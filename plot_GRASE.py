# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 20:20:24 2022

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


n_min=5
n_0 =2
d = 1
whole_region_initial = ([[0,1]])
in_min=0.9
in_max=50
#ot.RandomGenerator.SetSeed(int(7))      
selected_function = ot.PythonFunction(d, 1, 
                                      func=test_function_set.santetal03dc,#hig02
                                      func_sample = test_function_set.santetal03dc)
strategy=1 # 2 conservative 保守的 1 modest 中庸的 0 avant 激进的
getatable_1stgrad = True
if_optim= True # 极大加快运算速度
if_alBO = True # 是不是不管啥情况都用BO二阶梯度
basis =  ot.ConstantBasisFactory(d).build()
covarianceModel =  ot.MaternModel([1.]*d, 1.5)
"""
1st time region segmentation
"""

whole_region_initial_ot=divide.ot_region(whole_region_initial, d)

experiment = ot.MonteCarloExperiment( whole_region_initial_ot, n_0)
X_0 = experiment.generate()

X_0 = ot.LHSExperiment(whole_region_initial_ot, n_0,True, False).generate()
    
Y_0 = selected_function(X_0)
whole_region_0 = whole_region_initial
BO_2stgrad=if_alBO 
'''
2nd time region segmentation
'''
whole_region=whole_region_0
X_preorder_ot=X_0
    
X_preorder=np.array(X_preorder_ot)
whole_region_ot=divide.ot_region(whole_region, d)
       
# settings 贝叶斯优化的设置 
number_of_samples = 10000 
number_of_iteration = n_min#
acquisition_function_flag =2 #0: Estimated y-values 1: Mutual information (MI), 2: Expected Improvement(EI), 3: Probability of improvement (PI)
  

experiment_candidate_ot = ot.LHSExperiment(whole_region_ot, number_of_samples, False, True)
X_candidate_ot = experiment_candidate_ot.generate()
        
X_candidate = np.array(X_candidate_ot)
   
grad_1st = np.zeros(shape=(X_preorder_ot.getSize(),d))
               
if getatable_1stgrad:
    for i in range(0,X_preorder_ot.getSize()):
      grad_1st[i] = numberical_grad.grad(selected_function, X_preorder_ot[i])
else:
    Y_preorder_ot = selected_function(X_preorder_ot)
    algo_preorder = ot.KrigingAlgorithm(X_preorder_ot, 
                                     Y_preorder_ot, 
                                     covarianceModel, 
                                     basis)
    ot.Log.SetFile('Warnings.log')
    algo_preorder.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_preorder.run()
    result_preorder = algo_preorder.getResult()

    krig_preorder = result_preorder.getMetaModel()
            
    for i in range(0,X_preorder_ot.getSize()):
        grad_1st[i] =numberical_grad.grad_ungetatable_1st_ot(krig_preorder, X_preorder_ot[i])

   
algo_gradient = ot.KrigingAlgorithm(X_preorder_ot, 
                                   ot.Sample(grad_1st), 
                                   covarianceModel, basis)
algo_gradient.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
algo_gradient.run()
result_gradient = algo_gradient.getResult()
myKriging_grad = result_gradient.getMetaModel()

grad_2st = np.zeros(shape=(X_preorder_ot.getSize(),d))
for i in range(0,X_preorder.shape[0]): 
     grad_2st[i]=numberical_grad.grad_2nd_ot_1dim(myKriging_grad, X_preorder_ot[i])
        
    # 下面开始调用     
X_train = X_preorder
        
sum_grad_2st = abs(grad_2st)
           
y_train = sum_grad_2st
cumulative_variance = np.zeros(len(X_candidate))
        
   
for iteration in range(number_of_iteration):
            # Bayesian optimization
    selected_candidate_number, selected_X_candidate, cumulative_variance = bayesianoptimization(X_train, 
                                                                           y_train, 
                                                                           X_candidate ,
                                                                           acquisition_function_flag,
                                                                           cumulative_variance)

    X_train = np.append(X_train, 
                        np.reshape(selected_X_candidate, (1, X_train.shape[1])),
                        0)
        
    if getatable_1stgrad:
        grad_1st_add = numberical_grad.grad(selected_function, 
                                           ot.Point(selected_X_candidate))
    else:
               
        X_train_ot= ot.Sample(X_train)
        Y_true_response = selected_function(X_train_ot)

        algo_ungetatable_1stgrad = ot.KrigingAlgorithm(X_train_ot, 
                                     Y_true_response, 
                                    covarianceModel,
                                    basis)
        ot.Log.SetFile('Warnings.log')
        algo_ungetatable_1stgrad.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
        algo_ungetatable_1stgrad.run()
        result_ungetatable_1stgrad = algo_ungetatable_1stgrad.getResult()

        krig_ungetatable_1stgrad = result_ungetatable_1stgrad.getMetaModel()
        grad_1st_add = numberical_grad.grad_ungetatable_1st_ot(krig_ungetatable_1stgrad, 
                                                                      ot.Point(selected_X_candidate))        
                
            
    grad_1st= np.append(grad_1st, 
                        np.reshape(grad_1st_add, (1, grad_1st.shape[1])),
                        0)
          
    algo_BO = ot.KrigingAlgorithm(ot.Sample(X_train), 
                                  ot.Sample(grad_1st), 
                                   covarianceModel, basis)
    algo_BO.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_BO.run()
    result_BO = algo_BO.getResult()
    myKriging_grad_BO = result_BO.getMetaModel()

    grad_2st_BO = numberical_grad.grad_2nd_ot(myKriging_grad_BO, ot.Point(selected_X_candidate))       
            
        
    sum_grad_2st_BO = abs(grad_2st_BO) 
    y_add = sum_grad_2st_BO 
            
    y_train = np.append(y_train, y_add)
    X_candidate = np.delete(X_candidate , selected_candidate_number, 0)
    cumulative_variance = np.delete(cumulative_variance, selected_candidate_number)
  

X_next=X_train
Y_next = selected_function(X_next)

X_next




from openturns.viewer import View
import openturns.viewer as viewer
from matplotlib import pylab as plt
ot.Log.Show(ot.Log.NONE)

#sampleSize = 5
dimension = 1

# Define the function.
g = selected_function
# Create the design of experiments.
xMin = whole_region_initial[0][0]
xMax = whole_region_initial[0][1]
X_distr = ot.Uniform(xMin, xMax)
X =X_0
#X = ot.LHSExperiment(X_distr, sampleSize, False, False).generate()
Y = g(X)



# %%
# The following `sqrt` function will be used later to compute the standard deviation from the variance.
sqrt = ot.SymbolicFunction(["x"], ["sqrt(x)"])



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
 
current_size= np.size(X)
view0.save('GRASE_size' + str(current_size) +'.png', dpi=300)

in_min_1=0.1
graph0_second_gradient =add_fun.plot_second_gradient_initial(xMin, xMax, X, Y,
                                                     getatable_1stgrad,
                                                     selected_function,
                                                     in_min_1)
view0_second_gradient = viewer.View(graph0_second_gradient,
                                    figure_kw=size)
axes_view0_second_gradient = view0_second_gradient.getAxes()
#_ = axes[0].set_ylim(-100.0, 120.0)

#_ = axes_view0_second_gradient[0].yaxis.set_major_locator(y_major2)
_ = axes_view0_second_gradient[0].tick_params(labelsize=uselabelsize)
labels_view0_second_gradient = axes_view0_second_gradient[0].get_xticklabels()+ axes_view0_second_gradient[0].get_yticklabels()
[label_view0_second_gradient.set_fontname('Times New Roman') for label_view0_second_gradient in labels_view0_second_gradient]
_ = axes_view0_second_gradient[0].set_xticklabels(['']*10)
fig_view0_second_gradient = view0_second_gradient.getFigure
fig_view0_second_gradient()

_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view0_second_gradient.save('GradView_size' + str(current_size) +'.png', dpi=300)




X_add=X_next[n_0:n_0+n_min]

def getNewPoint(X_add):
    '''
    Returns a new point to be added to the design of experiments.
    This point maximizes the conditional variance of the kriging.
    '''
    xNew = X_add[0]
    xNew = ot.Point(xNew)
    
    return xNew


# %%
krigingStep = 0



# %%
xNew = getNewPoint(X_add)
X_add= np.delete(X_add, 0, 0)
yNew = g(xNew)
X.add(xNew)
Y.add(yNew)
krigResult = add_fun.createMyBasicKriging(X, Y,in_min)
krigingStep += 1
graph1 = add_fun.plotMyBasicKriging(krigResult, xMin, xMax, X, Y, selected_function,in_min)
view1=View(graph1,figure_kw=size)
axes_view1 = view1.getAxes()
_ = axes_view1[0].set_ylim(-1.1, 1.1)

_ = axes_view1[0].yaxis.set_major_locator(y_major1)
_ = axes_view1[0].tick_params(labelsize=uselabelsize)
labels_view1= axes_view1[0].get_xticklabels()+ axes_view1[0].get_yticklabels()
[label_view1.set_fontname('Times New Roman') for label_view1 in labels_view1]
_ = axes_view1[0].set_xticklabels(['']*10)
fig_view1 = view1.getFigure
fig_view1()

_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'


current_size= np.size(X)
view1.save('GRASE_size' + str(current_size) +'.png', dpi=300)

graph1_second_gradient =add_fun.plot_second_gradient(xMin, xMax, X, Y,
                                                     getatable_1stgrad,
                                                     selected_function,
                                                     in_min)
view1_second_gradient = viewer.View(graph1_second_gradient,
                                    figure_kw=size)
axes_view1_second_gradient = view1_second_gradient.getAxes()

_ = axes_view1_second_gradient[0].tick_params(labelsize=uselabelsize)
labels_view1_second_gradient = axes_view1_second_gradient[0].get_xticklabels()+ axes_view1_second_gradient[0].get_yticklabels()
[label_view1_second_gradient.set_fontname('Times New Roman') for label_view1_second_gradient in labels_view1_second_gradient]
_ = axes_view1_second_gradient[0].set_xticklabels(['']*10)
fig_view1_second_gradient = view1_second_gradient.getFigure
fig_view1_second_gradient()
# 坐标轴的刻度设置向内(in)或向外(out)
_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view1_second_gradient.save('GradView_size' + str(current_size) +'.png', dpi=300)


xNew = getNewPoint(X_add)
X_add= np.delete(X_add, 0, 0)
yNew = g(xNew)
X.add(xNew)
Y.add(yNew)
krigResult = add_fun.createMyBasicKriging(X, Y,in_min)
krigingStep += 1
graph2 = add_fun.plotMyBasicKriging(krigResult, xMin, xMax, X, Y, selected_function,in_min)
view2=View(graph2,figure_kw=size)
axes_view2 = view2.getAxes()
_ = axes_view2[0].set_ylim(-1.1, 1.1)

_ = axes_view2[0].yaxis.set_major_locator(y_major1)
_ = axes_view2[0].tick_params(labelsize=uselabelsize)
labels_view2= axes_view2[0].get_xticklabels()+ axes_view2[0].get_yticklabels()
[label_view2.set_fontname('Times New Roman') for label_view2 in labels_view2]
_ = axes_view2[0].set_xticklabels(['']*10)
fig_view2 = view2.getFigure
fig_view2()

_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view2.save('GRASE_size' + str(current_size) +'.png', dpi=300)



graph2_second_gradient =add_fun.plot_second_gradient(xMin, xMax, X, Y,
                                                     getatable_1stgrad,
                                                     selected_function,
                                                     in_min)
view2_second_gradient = viewer.View(graph2_second_gradient,
                                    figure_kw=size)
axes_view2_second_gradient = view2_second_gradient.getAxes()


_ = axes_view2_second_gradient[0].tick_params(labelsize=uselabelsize)
labels_view2_second_gradient = axes_view2_second_gradient[0].get_xticklabels()+ axes_view2_second_gradient[0].get_yticklabels()
[label_view2_second_gradient.set_fontname('Times New Roman') for label_view2_second_gradient in labels_view2_second_gradient]
_ = axes_view2_second_gradient[0].set_xticklabels(['']*10)
fig_view2_second_gradient = view2.getFigure
fig_view2_second_gradient()

_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view2_second_gradient.save('GradView_size' + str(current_size) +'.png', dpi=300)


xNew = getNewPoint(X_add)
X_add= np.delete(X_add, 0, 0)
yNew = g(xNew)
X.add(xNew)
Y.add(yNew)
krigResult = add_fun.createMyBasicKriging(X, Y, in_min)
krigingStep += 1
graph3 = add_fun.plotMyBasicKriging(krigResult, xMin, xMax, X, Y, selected_function,in_min)
view3=View(graph3,
                                    figure_kw=size)
axes_view3 = view3.getAxes()
_ = axes_view3[0].set_ylim(-1.1, 1.1)

_ = axes_view3[0].yaxis.set_major_locator(y_major1)
_ = axes_view3[0].tick_params(labelsize=uselabelsize)
labels_view3 = axes_view3[0].get_xticklabels()+ axes_view3[0].get_yticklabels()
[label_view3.set_fontname('Times New Roman') for label_view3 in labels_view3]
_ = axes_view3[0].set_xticklabels(['']*10)
fig_view3 = view3.getFigure
fig_view3()

_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view3.save('GRASE_size' + str(current_size) +'.png', dpi=300)


graph3_second_gradient =add_fun.plot_second_gradient(xMin, xMax, X, Y,
                                                     getatable_1stgrad,
                                                     selected_function,
                                                     in_min)
view3_second_gradient = viewer.View(graph3_second_gradient,
                                    figure_kw=size)
axes_view3_second_gradient = view3_second_gradient.getAxes()
#_ = axes[0].set_ylim(-100.0, 120.0)

#_ = axes_view3_second_gradient[0].yaxis.set_major_locator(y_major2)
_ = axes_view3_second_gradient[0].tick_params(labelsize=uselabelsize)
labels_view3_second_gradient = axes_view3_second_gradient[0].get_xticklabels()+ axes_view3_second_gradient[0].get_yticklabels()
[label_view3_second_gradient.set_fontname('Times New Roman') for label_view3_second_gradient in labels_view3_second_gradient]
_ = axes_view3_second_gradient[0].set_xticklabels(['']*10)
fig_view3_second_gradient = view3_second_gradient.getFigure
fig_view3_second_gradient()

_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view3_second_gradient.save('GradView_size' + str(current_size) +'.png', dpi=300)




xNew = getNewPoint(X_add)
X_add= np.delete(X_add, 0, 0)
yNew = g(xNew)
X.add(xNew)
Y.add(yNew)
krigResult = add_fun.createMyBasicKriging(X, Y, in_min)
krigingStep += 1
graph4 = add_fun.plotMyBasicKriging(krigResult, xMin, xMax, X, Y, selected_function,in_min)
view4=View(graph4,
                                    figure_kw=size)
axes_view4 = view4.getAxes()
_ = axes_view4[0].set_ylim(-1.1, 1.1)

_ = axes_view4[0].yaxis.set_major_locator(y_major1)
_ = axes_view4[0].tick_params(labelsize=uselabelsize)
labels_view4 = axes_view4[0].get_xticklabels()+ axes_view4[0].get_yticklabels()
[label_view4.set_fontname('Times New Roman') for label_view4 in labels_view4]
_ = axes_view4[0].set_xticklabels(['']*10)
fig_view4 = view4.getFigure
fig_view4()

_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view4.save('GRASE_size' + str(current_size) +'.png', dpi=300)


graph4_second_gradient =add_fun.plot_second_gradient(xMin, xMax, X, Y,
                                                     getatable_1stgrad,
                                                     selected_function,
                                                     in_min)
view4_second_gradient = viewer.View(graph4_second_gradient,
                                    figure_kw=size)
axes_view4_second_gradient = view4_second_gradient.getAxes()
#_ = axes[0].set_ylim(-100.0, 120.0)

#_ = axes_view4_second_gradient[0].yaxis.set_major_locator(y_major2)
_ = axes_view4_second_gradient[0].tick_params(labelsize=uselabelsize)
labels_view4_second_gradient = axes_view4_second_gradient[0].get_xticklabels()+ axes_view4_second_gradient[0].get_yticklabels()
[label_view4_second_gradient.set_fontname('Times New Roman') for label_view4_second_gradient in labels_view4_second_gradient]
_ = axes_view4_second_gradient[0].set_xticklabels(['']*10)
fig_view4_second_gradient = view4_second_gradient.getFigure
fig_view4_second_gradient()

_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view4_second_gradient.save('GradView_size' + str(current_size) +'.png', dpi=300)





xNew = getNewPoint(X_add)
X_add= np.delete(X_add, 0, 0)
yNew = g(xNew)
X.add(xNew)
Y.add(yNew)
krigResult = add_fun.createMyBasicKriging(X, Y, in_min)
krigingStep += 1
graph5 = add_fun.plotMyBasicKriging(krigResult, xMin, xMax, X, Y, selected_function,in_min)
view5=View(graph5,
                                    figure_kw=size)
axes_view5 = view5.getAxes()
_ = axes_view5[0].set_ylim(-1.1, 1.1)

_ = axes_view5[0].yaxis.set_major_locator(y_major1)
_ = axes_view5[0].tick_params(labelsize=uselabelsize)
labels_view5 = axes_view5[0].get_xticklabels()+ axes_view5[0].get_yticklabels()
[label_view5.set_fontname('Times New Roman') for label_view5 in labels_view5]
_ = axes_view5[0].set_xticklabels(['']*10)
fig_view5 = view5.getFigure
fig_view5()

_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view5.save('GRASE_size' + str(current_size) +'.png', dpi=300)


graph5_second_gradient =add_fun.plot_second_gradient(xMin, xMax, X, Y,
                                                     getatable_1stgrad,
                                                     selected_function,
                                                     in_min)
view5_second_gradient = viewer.View(graph5_second_gradient,
                                    figure_kw=size)
axes_view5_second_gradient = view5_second_gradient.getAxes()
#_ = axes[0].set_ylim(-100.0, 120.0)

#_ = axes_view5_second_gradient[0].yaxis.set_major_locator(y_major2)
_ = axes_view5_second_gradient[0].tick_params(labelsize=uselabelsize)
labels_view5_second_gradient = axes_view5_second_gradient[0].get_xticklabels()+ axes_view5_second_gradient[0].get_yticklabels()
[label_view5_second_gradient.set_fontname('Times New Roman') for label_view5_second_gradient in labels_view5_second_gradient]
_ = axes_view5_second_gradient[0].set_xticklabels(['']*10)
fig_view5_second_gradient = view5_second_gradient.getFigure
fig_view5_second_gradient()

_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view5_second_gradient.save('GradView_size' + str(current_size) +'.png', dpi=300)





















