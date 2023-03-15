

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn


class Merge(nn.Module):
    def __init__(self, class_token=True,trace_source=False) -> None:
        super().__init__()
        self.class_token = class_token
        self.train_source = trace_source
        self.merge_func = None
    
    def merge(self, x, mode):
        return self.merge_func(x, mode)
        
        
    def forward(self, x: torch.Tensor, kept_number: int):
        if x.shape[0] <= kept_number:
            return x
        
        metric = x.detach()
        metric = metric/metric.norm(dim=1, keepdim=True)
        unimportant_tokens_metric = metric[:, kept_number:]
        compress_number = unimportant_tokens_metric.shape[0]
        important_tokens_metric = metric[:,:kept_number]
        similarity = unimportant_tokens_metric@important_tokens_metric.transpose(-1,-2)
        if self.class_token:
            similarity[..., :, 0] = -math.inf
        node_max, node_idx = similarity.max(dim=-1)
        dst_idx = node_idx[..., None][:,:compress_number]
        
        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            src = x[:,kept_number:]
            dst = x[:,:kept_number]
            n, t1, c = src.shape
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, compress_number, c), src, reduce=mode) 
            if self.training:
                return torch.cat([dst, src], dim=1)
            else:
                return dst

        self.merge_func = merge
        x = merge(x, mode="mean")
        # size = merge(size, mode="sum")
        # source = merge(source, mode="amax")
        
        return x
    

def get_merge_func(metric: torch.Tensor, kept_number: int, class_token: bool = True):
    with torch.no_grad():
        metric = metric/metric.norm(dim=-1, keepdim=True)
        unimportant_tokens_metric = metric[:, kept_number:]
        compress_number = unimportant_tokens_metric.shape[1]
        important_tokens_metric = metric[:,:kept_number]
        similarity = unimportant_tokens_metric@important_tokens_metric.transpose(-1,-2)
        if class_token:
            similarity[..., :, 0] = -math.inf
        node_max, node_idx = similarity.max(dim=-1)
        dst_idx = node_idx[..., None]
    def merge(x: torch.Tensor, mode="mean", training=False) -> torch.Tensor:
        src = x[:,kept_number:]
        dst = x[:,:kept_number]
        n, t1, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, compress_number, c), src, reduce=mode) 
        if training:
            return torch.cat([dst, src], dim=1)
        else:
            return dst
    return merge, node_max



def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by average the similar pairs
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x, mode="mean")
    size = merge(size, mode="sum")

    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    source = merge(source, mode="amax")
    return source
