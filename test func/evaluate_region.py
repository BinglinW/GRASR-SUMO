# -*- coding: utf-8 -*-

import numpy as np
import evaluate_BF
import pandas as pd
import scipy.stats
from scipy.stats import norm
from sklearn.metrics.pairwise import cosine_distances



def BF(analyzed_grad, n_evaluate,file_i):

    score_list=[]
    for i in np.arange(0,n_evaluate):
        subgrad= np.column_stack((analyzed_grad[:,i],analyzed_grad[:,-1]))
        subgrad_score=evaluate_BF.evaluate_BF(subgrad,file_i)
        score_list.append(subgrad_score)
    
    return score_list


def variation_coefficient(data):
    mean = np.average(data, axis=0)  # 按列平均值
    std = np.std(data, axis=0) # 标准差 自由度
    cv = abs(std / mean) # 这里取了绝对值，避免出现负值从而最小的情况
    
    return cv

def JS_div(arr1,arr2,num_bins):
    max0 = max(np.max(arr1),np.max(arr2))
    min0 = min(np.min(arr1),np.min(arr2))
    bins = np.linspace(min0-1e-4, max0-1e-4, num=num_bins)
    PDF1 = pd.cut(arr1,bins).value_counts() / len(arr1)
    PDF2 = pd.cut(arr2,bins).value_counts() / len(arr2)
    return JS_divergence(PDF1.values,PDF2.values)

def KL_divergence(p, q):

    return scipy.stats.entropy(p, q)

def JS_divergence(p, q):

    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)

  
 