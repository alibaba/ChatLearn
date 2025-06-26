"""fsdp to vllm parameter sync group"""
import ray
from chatlearn.utils import future
from chatlearn.runtime.dist_actor import DistModel
from chatlearn.utils.error_monitor import ErrorSignalActor


def flatten(lst: list, reverse=False):
    result = []
    for item in lst:
        if reverse:
            result += item[::-1]
        else:
            result += item
    return result


class FSDP2VllmParameterSyncGroup:
    """fsdp to vllm parameter sync group
    """
    def __init__(
        self,
        src_model: DistModel,
        dst_model: DistModel,
        group_name: str,
        frequency: int,
        error_signal: ErrorSignalActor,
    ):
        self.src_model = src_model
        self.dst_model = dst_model
        self.group_name = group_name
        self.error_signal = error_signal
        self.frequency = frequency

        self.setup_collective_group()

    def setup_collective_group(self):
        # we put src_model first, so we don't need to change the rank of training model
        models = [self.src_model, self.dst_model]

        rank_offset = 0
        for model in models:
            for replica in model.replicas:
                replica._setup_ranks(rank_offset)
                rank_offset += replica.actor_num

    def sync(self,  *args, **kwargs):  # pylint: disable=unused-argument
        """
        sync function for fsdp to vllm
        """
        # for fsdp to vllm, we only need to find the src and dst actors that are on the same GPU.
        src_model_ranks = flatten(self.src_model.all_ranks)
        # adapt for model manager: models_to_revert
        dst_model_ranks = flatten(self.dst_model.all_ranks, reverse=True)

        param_name_list = ray.get(self.src_model.get_actor(0).get_fsdp_param_name.remote())
        print("debughh", len(param_name_list), param_name_list)
        for param_name in param_name_list:

            refs = []
            for src_rank, dst_rank in zip(src_model_ranks, dst_model_ranks):
                src_actor = self.src_model.get_actor(src_rank)
                dst_actor = self.dst_model.get_actor(dst_rank)
                reduce_data_ref = src_actor.get_weight_ipc_handles_by_name.remote(param_name)
                ref = dst_actor.update_weights_from_ipc_handles.remote(reduce_data_ref)
                refs.append(ref)
            future.wait(refs, return_output=True)
