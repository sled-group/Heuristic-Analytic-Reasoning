# This file includes evaluation/metric functions that may be useful for any task
from collections import defaultdict
import numpy as np
from scipy.stats import sem
from sklearn.metrics import label_ranking_average_precision_score, accuracy_score, top_k_accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from copy import deepcopy

def top_k_accuracy_score_multilabel(labels, scores, k=1):
    count = 0
    for l, s in zip(labels, scores):
        top_k = (-s).argsort(kind='stable')[:k]
        assert len(top_k.shape) == 1
        if len(np.intersect1d(top_k, l)) > 0:
            count += 1
    return count / len(labels)

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    metric_names = list(metrics_list[0].keys())
    metrics_agg = {}
    for metric in metric_names:
        metrics_agg[metric + '_mean'] = np.mean([ma[metric] for ma in metrics_list])
        metrics_agg[metric + '_std'] = np.std([ma[metric] for ma in metrics_list])
    return metrics_agg