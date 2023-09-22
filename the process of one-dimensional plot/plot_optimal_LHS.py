# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 08:44:57 2023

@author: bingl
"""
import numpy as np
import numberical_grad
import test_function_set
#import math
#import evaluate_region 
import add_fun
#import copy
import divide
#import heapq
import openturns as ot
#import single_stage_ot
from bayesianoptimization import bayesianoptimization
from openturns.viewer import View
import openturns.viewer as viewer
from matplotlib import pylab as plt
ot.Log.Show(ot.Log.NONE)

from matplotlib.pyplot import MultipleLocator

y_major2=MultipleLocator(50)
y_major1=MultipleLocator(1)
uselabelsize=10
size={"figsize": (4, 1.25)}


d = 1
whole_region_initial = ([[0,1]])
in_min=0.2
in_max=50
ot.RandomGenerator.SetSeed(int(7))      
selected_function = ot.PythonFunction(d, 1, 
                                      func=test_function_set.santetal03dc,#hig02
                                      func_sample = test_function_set.santetal03dc)

whole_region_initial_ot=divide.ot_region(whole_region_initial, d)
dimension = 1

# Define the function.
g = selected_function
# Create the design of experiments.
xMin = whole_region_initial[0][0]
xMax = whole_region_initial[0][1]
X_distr = ot.Uniform(xMin, xMax)

# %%
# The following `sqrt` function will be used later to compute the standard deviation from the variance.
sqrt = ot.SymbolicFunction(["x"], ["sqrt(x)"])

sampleSize=5

for sampleSize in range(2,8):


    X = ot.LHSExperiment(X_distr, sampleSize, False, False).generate()
    Y = g(X)
    
    
    print(X)
    
    
    
    krigResult = add_fun.createMyBasicKriging(X, Y, in_min)
    graph0 = add_fun.plotMyBasicKriging_initial(krigResult, xMin, xMax, X, Y, selected_function,in_min)
    view0 = viewer.View(graph0,
                        figure_kw=size)
    axes_view0 = view0.getAxes()
    _ = axes_view0[0].set_ylim(-1.1, 1.1)
    
    _ = axes_view0[0].yaxis.set_major_locator(y_major1)
    _ = axes_view0[0].tick_params(labelsize=uselabelsize)
    labels_view0 = axes_view0[0].get_xticklabels()+ axes_view0[0].get_yticklabels()
    [label_view0.set_fontname('Times New Roman') for label_view0 in labels_view0]
    _ = axes_view0[0].set_xticklabels(['']*10)
    fig_view0 = view0.getFigure
    fig_view0()
   
    _ = plt.rcParams['xtick.direction'] = 'in'
    _ = plt.rcParams['ytick.direction'] = 'in'
    #plt.show()
    
    view0.save('LHS_size'+ str(sampleSize)+'.png', dpi=300)
 

