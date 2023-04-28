from collections import defaultdict
from rlhf.initialize import patch_ray
from rlhf.logger import logger
from rlhf import utils
from rlhf import get_args
import threading
from itertools import cycle
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
            logger.info(f"start setup_collective_group for {model.name}, group_name: {self.group_name}, world_size: {world_size}, rank_offset: {rank_offset}")
            for replica in model.replicas:
                refs += replica._setup_collective_group(rank_offset, world_size, self.group_name)
                rank_offset += replica.actor_num

        utils.get(refs)
        logger.info(f"init collective group done for {self.group_name}")


    def destroy_collective_group(self):
        try:
            self.src_model.destroy_collective_group()
            self.tgt_model.destroy_collective_group()
            logger.info(f"destroy_collective_group success for {self.group_name}")
        except Exception as e:
            logger.exception(f"destroy_collective_group fail for {self.group_name} {e}")


    def add_recv_actor(self, src_rank, dest_rank):
        src_actor = self.src_model.get_actor(src_rank)
        self.actor2rank[src_actor] = src_rank
        dest_actor = self.tgt_model.get_actor(dest_rank)
        self.actor2rank[dest_actor] = dest_rank
        self.send_recv_actor_mappings[src_actor].append(dest_actor)
        self.recv_send_actor_mappings[dest_actor].append(src_actor)


    def build_rank_mapping(self):
        # setup rank mapping for src parameter and tgt parameter
        # get rank for one src_model, without model replicas
        src_ranks = utils.get(self.src_model.replicas[0].master.get_param_ranks.remote())
        tgt_ranks = self.tgt_model.all_ranks
        if src_ranks is None or tgt_ranks is None:
            if get_args().rlhf_args.debug:
                return
            else:
                raise Exception(f"src_ranks {src_ranks} or tgt_ranks {tgt_ranks} should not be None")
            

        assert len(src_ranks[0]) % len(tgt_ranks[0]) == 0, f"src training model ranks should be times of tgt ranks, but got {len(src_ranks)} and {len(tgt_ranks[0])}"
        mapping_interval = len(src_ranks[0]) // len(tgt_ranks[0])

        replica_rank_iter = cycle(iter(src_ranks))

        for tgt_replica_ranks in tgt_ranks:
            src_replica_ranks = next(replica_rank_iter)
            for i, tgt_rank in enumerate(tgt_replica_ranks):
                for j in range(i*mapping_interval, (i+1)*mapping_interval):
                    self.add_recv_actor(src_replica_ranks[j], tgt_rank)

    def sync_send_recv(self, send_actor, recv_actor):
        rank = self.actor2rank[send_actor]
        tgt_names = src_names = None

        if len(self.send_recv_actor_mappings) > 1:
            # TODO: make it an attribute of actor, so we can get fast
            num_pipeline_stage = utils.get(send_actor.pipeline_model_parallel_size.remote())
            if num_pipeline_stage > 1:
                tgt_src_mappings = utils.get(send_actor.build_pipeline_layer_name_mapping.remote())
                tgt_names = tgt_src_mappings.keys()
                src_names = tgt_src_mappings.values()
        if tgt_names is None:
            tgt_names = src_names = utils.get(send_actor.get_parameter_names.remote())
        
        # TODO: optimize with coleased buffer
        for send_name, tgt_name in zip(src_names, tgt_names):
            # TODO: add name mapping for models with different parameter name
            tgt_name = '.'.join(tgt_name.split('.')[1:])
            recv_tensor_exist = utils.get(recv_actor.exist_parameter.remote(tgt_name))
            if not recv_tensor_exist:
                logger.info(f"recv tensor {tgt_name} not exists")
                all_tgt_layer_names = utils.get(recv_actor.get_parameter_names.remote())
                raise Exception(f"recv tensor {tgt_name} not exists, while recv model has following layers {all_tgt_layer_names}")
            send_ref = send_actor.send_parameter.remote(send_name, self.actor2rank[recv_actor], self.group_name)
            recv_ref = recv_actor.recv_parameter.remote(tgt_name, self.actor2rank[send_actor], self.group_name)
            utils.get([send_ref, recv_ref])
        logger.info(f"sync all parameters from {send_actor} to {recv_actor}, total param num {len(src_names)}")


    def sync(self):
        threads = []
        use_threads = True
        for send_actor in self.send_recv_actor_mappings:
            recv_actors = self.send_recv_actor_mappings[send_actor]
            for recv_actor in recv_actors:
                if use_threads:
                    thread = threading.Thread(target=self.sync_send_recv, args=(send_actor, recv_actor))
                    threads.append(thread)
                else:
                    self.sync_send_recv(send_actor, recv_actor)
        if use_threads:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        logger.info(f"sync all parameters done")
