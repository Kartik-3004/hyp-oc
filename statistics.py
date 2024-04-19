import json
import math
import numpy as np
import pandas as pd
import torch
import os
import sys
import shutil
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



def calculate_accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  output = torch.tensor(output)
  target = torch.tensor(target)
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def eval_state(probs, labels, thr):
  predict = probs >= thr
  TN = np.sum((labels == 0) & (predict == False))
  FN = np.sum((labels == 1) & (predict == False))
  FP = np.sum((labels == 0) & (predict == True))
  TP = np.sum((labels == 1) & (predict == True))
  return TN, FN, FP, TP

def calculate(probs, labels, thr):
  TN, FN, FP, TP = eval_state(probs, labels, thr)
  APCER = 1.0 if (FP + TN == 0) else FP / float(FP + TN)
  NPCER = 1.0 if (FN + TP == 0) else FN / float(FN + TP)
  ACER = (APCER + NPCER) / 2.0
  ACC = (TP + TN) / labels.shape[0]
  return APCER, NPCER, ACER, ACC

def calculate_threshold(probs, labels, threshold):
  TN, FN, FP, TP = eval_state(probs, labels, threshold)
  ACC = (TP + TN) / labels.shape[0]
  return ACC

def get_threshold(probs, grid_density):
  Min, Max = min(probs), max(probs)
  thresholds = []
  for i in range(grid_density + 1):
    thresholds.append(0.0 + i * 1.0 / float(grid_density))
  thresholds.append(1.1)
  return thresholds

def get_EER_states(probs, labels, grid_density=10000):
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if (FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif (FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list

def get_HTER_at_thr(probs, labels, thr):
  TN, FN, FP, TP = eval_state(probs, labels, thr)
  if (FN + TP == 0):
    FRR = 1.0
    FAR = FP / float(FP + TN)
  elif (FP + TN == 0):
    FAR = 1.0
    FRR = FN / float(FN + TP)
  else:
    FAR = FP / float(FP + TN)
    FRR = FN / float(FN + TP)
  HTER = (FAR + FRR) / 2.0
  return HTER


def roc_curve_and_rate(label_list, prob_list):
    fpr, tpr, thr = roc_curve(label_list, prob_list)
    tpr_filtered = tpr[fpr <= 1 / 100]
    if len(tpr_filtered) == 0:
        rate = 0
    else:
        rate = tpr_filtered[-1]
    return fpr, tpr, thr, rate

def calculate_metrics(labels, predictions):
  EER, threshold, FRR_list, FAR_list = get_EER_states(predictions, labels)
  accuracy_threshold = calculate_threshold(predictions, labels, threshold)
  roc_auc = roc_auc_score(labels, predictions)
  HTER = get_HTER_at_thr(predictions, labels, threshold)
  APCER, NPCER, ACER, ACC = calculate(predictions, labels, threshold)
  return APCER, NPCER, ACER, EER, HTER, roc_auc, threshold, accuracy_threshold
