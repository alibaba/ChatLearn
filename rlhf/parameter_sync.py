from collections import defaultdict
from rlhf.initialize import patch_ray
import ray
patch_ray()


class ParameterSyncGroup:

    def __init__(self, src_model, tgt_model, group_name):
        self.src_model = src_model
        self.tgt_model = tgt_model
        self.group_name = group_name
        self.send_recv_actor_mappings = defaultdict(list)
        self.recv_send_actor_mappings = defaultdict(list)
        self.actor2rank = {}
        self.setup_collective_group()
        self.build_rank_mapping()

    
    def setup_collective_group(self):
        refs = []
        # we put src_model first, so we don't need to change the rank of training model
        models = [self.src_model, self.tgt_model]
        world_size = sum(model.actor_num for model in models)

        rank_offset = 0
        for i, model in enumerate(models):
            print(f"start setup_collective_group for {model.name}, group_name: {self.group_name}, world_size: {world_size}, rank_offset: {rank_offset}", flush=True)
            refs += model._setup_collective_group(rank_offset, world_size, self.group_name)
            rank_offset += model.actor_num

        ray.get(refs)
        print(f"init collective group done for {self.group_name}")


    def destroy_collective_group(self):
        try:
            self.src_model.destroy_collective_group()
            self.tgt_model.destroy_collective_group()
            print(f"destroy_collective_group success for {self.group_name}", flush=True)
        except Exception as e:
            print(f"destroy_collective_group fail for {self.group_name}", e, flush=True)


    def add_recv_actor(self, src_rank, dest_rank):
        src_actor = self.src_model.get(src_rank)
        self.actor2rank[src_actor] = src_rank
        dest_actor = self.tgt_model.get(dest_rank)
        self.actor2rank[dest_actor] = dest_rank
        self.send_recv_actor_mappings[src_actor].append(dest_actor)
        self.recv_send_actor_mappings[dest_actor].append(src_actor)



    def build_rank_mapping(self):
        # setup rank mapping for src parameter and tgt parameter

        # get rank for one src_model, without model replicas
        src_ranks = ray.get(self.src_model.all_actors[0][0].get_param_ranks.remote())
        tgt_ranks = self.tgt_model.all_ranks

        assert len(src_ranks) % len(tgt_ranks[0]) == 0, f"src training model ranks should be times of tgt ranks, but got {len(src_ranks)}and{len(tgt_ranks[0])}"
        mapping_interval = len(src_ranks) // len(tgt_ranks[0])

        for tgt_replica_ranks in tgt_ranks:
            for i, tgt_rank in enumerate(tgt_replica_ranks):
                for j in range(i*mapping_interval, (i+1)*mapping_interval):
                    self.add_recv_actor(src_ranks[j], tgt_rank)


    def sync(self):
        for send_actor in self.send_recv_actor_mappings:
            tensor_names = ray.get(send_actor.get_parameter_names.remote())
            # TODO: current only test one recv actor
            assert len(self.send_recv_actor_mappings[send_actor]) == 1, f'got {self.send_recv_actor_mappings[send_actor]} recv actors'
            recv_actor = self.send_recv_actor_mappings[send_actor][0]
            # TODO: optimize with coleased buffer
            for send_name in tensor_names:
                # TODO: add name mapping for models with different parameter name
                name = '.'.join(send_name.split('.')[1:])
                recv_tensor_exist = ray.get(recv_actor.exist_parameter.remote(name))
                if not recv_tensor_exist:
                    print(f"recv tensor {name} not exists")
                    continue

                send_ref = send_actor.send_parameter.remote(send_name, self.actor2rank[recv_actor], self.group_name)
                recv_ref = recv_actor.recv_parameter.remote(name, self.actor2rank[send_actor], self.group_name)
                ray.get([send_ref, recv_ref])