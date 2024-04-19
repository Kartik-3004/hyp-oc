import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
from hyptorch.pmath import dist_matrix
from hyptorch import pmath
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def TPC_loss_hyp(features, c):
    batch_size = features.shape[0]
    assert batch_size % 2 == 0, "batch size not even"
    left_batch = features[:int(0.5*features.shape[0])]

    right_batch = features[int(0.5*features.shape[0]):]

    loss = 0.0
    for left_vec, right_vec in zip(left_batch, right_batch):
        mobius_add_term = torch.norm(pmath.mobius_add(-left_vec, right_vec, c=c),2,-1)
        sqrt_c = float(math.sqrt(c))
        num = torch.atan(sqrt_c*mobius_add_term)*2.0
        distance = num/sqrt_c
        loss += distance

    l = loss / float(batch_size)
    return l
