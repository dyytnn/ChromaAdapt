import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from configs.config import configurable
from utils import is_dist_available_and_initialized


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps = 1e-8)
    
    def pairwise_NNs_inner(self, x: Tensor):
        """Pairwise nearest neighbors for L2-norm vectors"""
        
        #todo pairwise dot product = inverse distance
        dots = torch.matmul(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1) # trick to fill diagonal with -1
        _, indices = torch.max(dots, dim = 1) # max inner prod -> min distance
        
        return indices
    
    def forward(self, student_output: Tensor, eps = 1e-8) -> Tensor:
        """
        Compute the KoLeoLoss from the student output.

        The KoLeoLoss regularizer is based on the nearest neighbor
        search in the feature space. It encourages the student to
        produce features that are dissimilar to the features of
        the other samples in the same batch.

        Args:
            student_output (Tensor): The output of the student network
            eps (float): The epsilon value for numerical stability

        Returns:
            Tensor: The KoLeoLoss value
        """
        
        with torch.autocast("cuda", enabled = False):
            student_output = F.normalize(student_output, dim = -1, p = 2, eps = eps)
            indices = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[indices]) # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
            
        return loss

class KoLeoLossDistributed(nn.Module):
    @configurable
    def __init__(self, *, topk = 1, loss_group_size: int = None):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps = 1e-8)
        self.topk = topk
        self.loss_group_size = loss_group_size # Size of nearest neightbor search group, None for all-gather
    
    @classmethod
    def from_config(cls, config: dict):
        return {
            "topk": config.koleo_topk if hasattr(config, "koleo_topk") else 1,
            "loss_group_size": config.koleo_distributed_loss_group_size if hasattr(config, "koleo_distributed_loss_group_size") else None
        }
    
    def pairwise_NNs_inner(self, x: Tensor, all_x: Tensor, rank: int) -> Tensor:
        """Pairwise nearsest neighbors for L2-norm vectors"""
        dots = torch.mm(x, all_x.t()) # local B x global B
        local_B, global_B = dots.shape
        dots.view(-1)[rank * local_B :: (global_B + 1)].fill_(-1) # trick to fill diagonal with -1
        _, indices = torch.topk(dots, dim = 1, k = self.topk) # max inner prod -> min distance
        
        return indices
    
    def forward(self, student_output: Tensor, eps: float = 1e-8) -> Tensor:
        with torch.autocast("cuda", enabled = False):
            student_output = F.normalize(student_output, eps = eps, p = 2, dim = -1)
            if is_dist_available_and_initialized():
                all_student_output = torch.cat(dist.nn.all_gather(student_output), dim = 0) # global_B x D
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                all_student_output = student_output
                world_size = 1
                rank = 0
                
            # Group the global batch into groups of size `loss_group_size` and use the features of the group
            # The local rank falls into as the nearest neighbor search set for the local rank
            local_B = len(student_output)
            global_B = len(all_student_output)
            loss_group_size = self.loss_group_size if self.loss_group_size is not None else global_B
            assert global_B % loss_group_size == 0, f"Global batch size {global_B} must be divisible by loss group size {loss_group_size}"
            
            n_groups = global_B // loss_group_size
            ranks_per_group = world_size // n_groups
            rank_in_group = rank % ranks_per_group
            group = rank // ranks_per_group
            all_student_outputs = all_student_outputs.view(n_groups, loss_group_size, student_output.shape[1])
            all_student_outputs = all_student_outputs[group]  # loss_group_size x D

            with torch.no_grad():
                indices = self.pairwise_NNs_inner(student_output, all_student_outputs, rank_in_group)  # local_B x topk

            student_output_expanded = (
                student_output.unsqueeze(1).repeat(1, self.topk, 1).flatten(0, 1)
            )  # (local_B * topk) x D
            distances = self.pdist(student_output_expanded, all_student_outputs[indices].flatten(0, 1))  # BxD, BxD -> B
            loss = -torch.log(distances.float() + eps).mean()

        return loss
