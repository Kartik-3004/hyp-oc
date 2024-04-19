import json
import math
import pandas as pd
import torch
import os
import sys
import shutil
import pickle

def save_checkpoint(epoch, encoder, model, metrics, filename):

    state = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'classifier_state_dict': model.state_dict(),
        'metrics': metrics,
    }
    torch.save(state, filename)

def load_dict(model, fname):
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))