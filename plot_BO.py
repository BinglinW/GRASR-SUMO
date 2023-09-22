# -*- coding: utf-8 -*-
"""

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
import copy
from matplotlib.pyplot import MultipleLocator

y_major2=MultipleLocator(50)
y_major1=MultipleLocator(1)
uselabelsize=10
size={"figsize": (4, 1.25)}


n_min=8
n_0 =2
d = 1
whole_region_initial = ([[0,1]])
in_min=0.5
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

#sampleSize = 5
dimension = 1

# Define the function.
g = selected_function
# Create the design of experiments.
xMin = whole_region_initial[0][0]
xMax = whole_region_initial[0][1]
X_distr = ot.Uniform(xMin, xMax)

X_0 = ot.LHSExperiment(X_distr, n_0, False, False).generate()

X_train= np.array(copy.deepcopy(X_0))
y_train =np.array( g(X_train))




# settings 贝叶斯优化的设置 
number_of_samples = 100 
number_of_iteration = n_min
acquisition_function_flag =1# 0: Estimated y-values 1: Mutual information (MI), 2: Expected Improvement(EI), 3: Probability of improvement (PI)
  

experiment_candidate_ot = ot.LHSExperiment(whole_region_initial_ot, number_of_samples, False, True)
X_candidate_ot = experiment_candidate_ot.generate()
        
X_candidate = np.array(X_candidate_ot)

cumulative_variance = np.zeros(len(X_candidate))
for iteration in range(number_of_iteration):
    print([iteration+1, number_of_iteration])
    # Bayesian optimization
    selected_candidate_number, selected_X_candidate, cumulative_variance = bayesianoptimization(X_train, y_train, X_candidate,
                                                                                                acquisition_function_flag,
                                                                                                cumulative_variance)
    X_app= np.reshape(X_candidate[selected_candidate_number, :], (1, X_train.shape[1]))
    X_train = np.append(X_train, X_app, 0)
    
    y_app= np.array(g(X_app))[0]
    y_train = np.append(y_train, y_app )

    X_candidate = np.delete(X_candidate, selected_candidate_number, 0)
    cumulative_variance = np.delete(cumulative_variance, selected_candidate_number)

  

X_next=X_train
Y_next = selected_function(X_next)

X_next




from openturns.viewer import View
import openturns.viewer as viewer
from matplotlib import pylab as plt
ot.Log.Show(ot.Log.NONE)



# %%
# The following `sqrt` function will be used later to compute the standard deviation from the variance.
sqrt = ot.SymbolicFunction(["x"], ["sqrt(x)"])

X =X_0
#X = ot.LHSExperiment(X_distr, sampleSize, False, False).generate()
Y = g(X)


krigResult = add_fun.createMyBasicKriging(X, Y, in_min)
graph0 = add_fun.plotMyBasicKriging_initial(krigResult, xMin, xMax, X, Y, selected_function,in_min)
view0 = viewer.View(graph0,
                    figure_kw=size)
current_size=np.size(X)

axes_view0 = view0.getAxes()
_ = axes_view0[0].set_ylim(-1.1, 1.1)

_ = axes_view0[0].yaxis.set_major_locator(y_major1)
_ = axes_view0[0].tick_params(labelsize=uselabelsize)
labels_view0 = axes_view0[0].get_xticklabels()+ axes_view0[0].get_yticklabels()
[label_view0.set_fontname('Times New Roman') for label_view0 in labels_view0]
_ = axes_view0[0].set_xticklabels(['']*10)
fig_view0 = view0.getFigure
fig_view0()
# 坐标轴的刻度设置向内(in)或向外(out)
_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'
#plt.show()

view0.save('BO_size' + str(current_size) +'.png', dpi=300)
 

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
# 坐标轴的刻度设置向内(in)或向外(out)
_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view1.save('BO_size' + str(current_size) +'.png', dpi=300)



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
# 坐标轴的刻度设置向内(in)或向外(out)
_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view2.save('BO_size' + str(current_size) +'.png', dpi=300)



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
# 坐标轴的刻度设置向内(in)或向外(out)
_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view3.save('BO_size' + str(current_size) +'.png', dpi=300)



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
# 坐标轴的刻度设置向内(in)或向外(out)
_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view4.save('BO_size' + str(current_size) +'.png', dpi=300)




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
# 坐标轴的刻度设置向内(in)或向外(out)
_ = plt.rcParams['xtick.direction'] = 'in'
_ = plt.rcParams['ytick.direction'] = 'in'

current_size= np.size(X)
view5.save('BO_size' + str(current_size) +'.png', dpi=300)

