# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:53:35 2022

@author: wbl
"""
import numpy as np
import numberical_grad
import test_function_set

import matplotlib.pyplot as plt
import divide
#import heapq
import openturns as ot
#import single_stage_ot
from bayesianoptimization import bayesianoptimization
from bayesianoptimization import bayesianoptimization_af
# %%
def createMyBasicKriging(X, Y, in_min):
    '''
    Create a kriging from a pair of X and Y samples.
    We use a 3/2 Matérn covariance model and a constant trend.
    '''
    d=1
    
    in_max=100
    basis =  ot.ConstantBasisFactory(d).build()
    covarianceModel =  ot.MaternModel([1.]*d, 1.5)
    algo = ot.KrigingAlgorithm(X, Y, covarianceModel, basis)
    algo.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo.run()
    krigResult = algo.getResult()
    return krigResult

# %%
def linearSample(xmin, xmax, npoints):
    '''Returns a sample created from a regular grid
    from xmin to xmax with npoints points.'''
    step = (xmax-xmin)/(npoints-1)
    rg = ot.RegularGrid(xmin, step, npoints)
    vertices = rg.getVertices()
    return vertices


# %%
def plot_kriging_bounds(vLow, vUp, n_test):
    '''
    From two lists containing the lower and upper bounds of the region,
    create a PolygonArray.
    '''
    
    #palette = ot.Drawable.BuildDefaultPalette(2)
    myPaletteColor = '#CCECFF'#palette[1]
    polyData = [[vLow[i], vLow[i+1], vUp[i+1], vUp[i]]
                for i in range(n_test-1)]
    polygonList = [ot.Polygon(
        polyData[i], myPaletteColor, myPaletteColor) for i in range(n_test-1)]
    boundsPoly = ot.PolygonArray(polygonList)
    boundsPoly.setLegend("95% bounds")
    return boundsPoly


# %%
def plotMyBasicKriging(krigResult, xMin, xMax, X, Y, selected_function, in_min, level=0.95):
    '''
    Given a kriging result, plot the data, the kriging metamodel
    and a confidence interval.
    '''
    g=selected_function
    sqrt = ot.SymbolicFunction(["x"], ["sqrt(x)"])
    
    in_max=50
    samplesize = X.getSize()
    meta = krigResult.getMetaModel()
    #graphKriging = meta.draw(xMin, xMax)
    #graphKriging.setLegends(["GRASE-SUMO"])
    # Create a grid of points and evaluate the function and the kriging
    nbpoints = 250
    xGrid = linearSample(xMin, xMax, nbpoints)
    yFunction = g(xGrid)
    yKrig = meta(xGrid)
    # Compute the conditional covariance
    epsilon = ot.Sample(nbpoints, [1.e-8])
    conditionalVariance = krigResult.getConditionalMarginalVariance(
        xGrid)+epsilon
    conditionalSigma = sqrt(conditionalVariance)
    # Compute the quantile of the Normal distribution
    alpha = 1-(1-level)/2
    quantileAlpha = ot.DistFunc.qNormal(alpha)
    # Graphics of the bounds
    epsilon = 1.e-8
    dataLower = [yKrig[i, 0] - quantileAlpha * conditionalSigma[i, 0]
                 for i in range(nbpoints)]
    dataUpper = [yKrig[i, 0] + quantileAlpha * conditionalSigma[i, 0]
                 for i in range(nbpoints)]
    # Coordinates of the vertices of the Polygons
    vLow = [[xGrid[i, 0], dataLower[i]] for i in range(nbpoints)]
    vUp = [[xGrid[i, 0], dataUpper[i]] for i in range(nbpoints)]
    # Compute the Polygon graphics
    boundsPoly = plot_kriging_bounds(vLow, vUp, nbpoints)
    #boundsPoly.setLegend("95% bounds")
    # Validate the kriging metamodel
    mmv = ot.MetaModelValidation(xGrid, yFunction, meta)
    Q2 = mmv.computePredictivityFactor()[0]
    # Plot the function
    graphFonction = ot.Curve(xGrid, yFunction)
    graphFonction.setLineStyle("dashed")
    graphFonction.setColor("magenta")
    graphFonction.setLineWidth(2)
    #graphFonction.setLegend("True Function")
    # Draw the X and Y observed
    cloudDOE = ot.Cloud(X[0:-1], Y[0:-1])
    cloudDOE.setPointStyle("circle")
    cloudDOE.setColor("#33CC33")
    
    #cloudDOE.setLegend("Existing Data")
    
    cloudDOE_new = ot.Cloud(X[-1], Y[-1])
    cloudDOE_new.setPointStyle("star")
    cloudDOE_new.setColor("red")
    
    #cloudDOE_new.setLegend("Added Data")

    graphKriging = ot.Curve(xGrid, yKrig)
    graphKriging.setLineStyle("solid")
    graphKriging.setColor("blue")
    graphKriging.setLineWidth(2)     
    
    graph = ot.Graph()
    graph.add(boundsPoly)
    graph.add(graphFonction)
    graph.add(graphKriging)
    #graph.add(graphFonction_lhs)
    graph.add(cloudDOE)
    graph.add(cloudDOE_new)
    #graph.add(cloudDOE_lhs)

    #graph.setLegendPosition("topright")
    graph.setAxes(True)
    graph.setGrid(False)
    graph.setXMargin(0.01)
    
    #graph.setTitle("GRASE-SUMO: Sample Size = %d, Q2=%.2f%%" % (samplesize, 100*Q2))
    #graph.setTitle("The SUMO construction (Sample Size = %d )" % samplesize)
    #graph.setXTitle("$X$" )
    #graph.setYTitle("$Y$")
    return graph

def plotMyBasicKriging_initial(krigResult, xMin, xMax, X, Y, selected_function, in_min, level=0.95):
    '''
    Given a kriging result, plot the data, the kriging metamodel
    and a confidence interval.
    '''
    g=selected_function
    sqrt = ot.SymbolicFunction(["x"], ["sqrt(x)"])
    
    in_max=50
    samplesize = X.getSize()
    meta = krigResult.getMetaModel()
    #graphKriging = meta.draw(xMin, xMax)
    
  
    
    #graphKriging.setLegends(["GRASE-SUMO"])
    # Create a grid of points and evaluate the function and the kriging
    nbpoints = 250
    xGrid = linearSample(xMin, xMax, nbpoints)
    yFunction = g(xGrid)
    yKrig = meta(xGrid)
    # Compute the conditional covariance
    epsilon = ot.Sample(nbpoints, [1.e-8])
    conditionalVariance = krigResult.getConditionalMarginalVariance(
        xGrid)+epsilon
    conditionalSigma = sqrt(conditionalVariance)
    # Compute the quantile of the Normal distribution
    alpha = 1-(1-level)/2
    quantileAlpha = ot.DistFunc.qNormal(alpha)
    # Graphics of the bounds
    epsilon = 1.e-8
    dataLower = [yKrig[i, 0] - quantileAlpha * conditionalSigma[i, 0]
                 for i in range(nbpoints)]
    dataUpper = [yKrig[i, 0] + quantileAlpha * conditionalSigma[i, 0]
                 for i in range(nbpoints)]
    # Coordinates of the vertices of the Polygons
    vLow = [[xGrid[i, 0], dataLower[i]] for i in range(nbpoints)]
    vUp = [[xGrid[i, 0], dataUpper[i]] for i in range(nbpoints)]
    # Compute the Polygon graphics
    boundsPoly = plot_kriging_bounds(vLow, vUp, nbpoints)
    #boundsPoly.setLegend("95% bounds")
    # Validate the kriging metamodel
    mmv = ot.MetaModelValidation(xGrid, yFunction, meta)
    Q2 = mmv.computePredictivityFactor()[0]
    # Plot the function
    graphFonction = ot.Curve(xGrid, yFunction)
    graphFonction.setLineStyle("dashed")
    graphFonction.setColor("magenta")
    graphFonction.setLineWidth(2)
    #graphFonction.setLegend("True Function")
    # Draw the X and Y observed
    cloudDOE = ot.Cloud(X, Y)
    cloudDOE.setPointStyle("circle")
    cloudDOE.setColor("#33CC33")
    
    #cloudDOE.setLegend("Existing Data")
    
    graphKriging = ot.Curve(xGrid, yKrig)
    graphKriging.setLineStyle("solid")
    graphKriging.setColor("blue")
    graphKriging.setLineWidth(2) 
    
    graph = ot.Graph()
    graph.add(boundsPoly)
    graph.add(graphFonction)
    #graph.add(graphFonction_lhs)
    graph.add(graphKriging)
    graph.add(cloudDOE)
    
    
    #graph.setLegendPosition("topright")
    graph.setAxes(True)
    graph.setGrid(False)
    graph.setXMargin(0.01)
    #plt.ylim(-1, 1.1)
    #graph.setTitle("GRASE-SUMO: Sample Size = %d, Q2=%.2f%%" % (samplesize, 100*Q2))
    #graph.setTitle("The SUMO construction (Sample Size = %d )" % samplesize)
    #graph.setXTitle("$X$" )
    graph.setYTitle("$Y$")
    
    return graph




def model_second_gradient(X_preorder_ot, myKriging_grad,d=1):
    
    grad_2st = np.zeros(shape=(X_preorder_ot.getSize(),d))
    for i in range(0,X_preorder_ot.getSize()): # 这个地方就是后面要换，根据代理模型来建立
         grad_2st[i]=numberical_grad.grad_2nd_ot_1dim(myKriging_grad, X_preorder_ot[i])
    
    sum_grad_2st = abs(grad_2st)
               
    y = sum_grad_2st
    return ot.Sample(y)


def plot_second_gradient(xMin, xMax, X, Y, getatable_1stgrad, selected_function, in_min, level=0.95):
    d=1
    samplesize = X.getSize()
    sqrt = ot.SymbolicFunction(["x"], ["sqrt(x)"])
    X_plot=X
    Y_plot=Y
    #in_min=3
    in_max=200
    basis =  ot.ConstantBasisFactory(d).build()
    covarianceModel =  ot.MaternModel([1.]*d, 1.5)
    grad_1st = np.zeros(shape=(X_plot.getSize(),d))
    if getatable_1stgrad:
        for i in range(0,X_plot.getSize()):
            grad_1st[i] = numberical_grad.grad(selected_function, X_plot[i])
    else:
        Y_plot = selected_function(X_plot)
        algo_preorder = ot.KrigingAlgorithm(X_plot, 
                                            Y_plot, 
                                            covarianceModel, 
                                            basis)
        ot.Log.SetFile('Warnings.log')
        algo_preorder.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
        algo_preorder.run()
        result_preorder = algo_preorder.getResult()

        krig_preorder = result_preorder.getMetaModel()
            
        for i in range(0,X_plot.getSize()):
            grad_1st[i] =numberical_grad.grad_ungetatable_1st_ot(krig_preorder, X_plot[i])
        
# 然后是对梯度建立代理模型
    algo_gradient = ot.KrigingAlgorithm(X_plot, 
                                        ot.Sample(grad_1st), 
                                        covarianceModel, basis)
    algo_gradient.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_gradient.run()
    result_gradient = algo_gradient.getResult()
    myKriging_grad = result_gradient.getMetaModel()
    
    #experiment = ot.MonteCarloExperiment( whole_region_initial_ot, int(1000))
    #x_fix = experiment.generate()
    x_fix =X_plot
    
    grad_2st = np.zeros(shape=(x_fix.getSize(),d))
    for i in range(0,x_fix.getSize()): # 这个地方就是后面要换，根据代理模型来建立
        grad_2st[i]=numberical_grad.grad_2nd_ot_1dim(myKriging_grad, x_fix[i])
        
    sum_grad_2st = abs(grad_2st)
                   
    algo_second_gradient = ot.KrigingAlgorithm(x_fix, 
                                        ot.Sample(sum_grad_2st), 
                                        covarianceModel, basis)
    algo_second_gradient.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_second_gradient.run()
    result_second_gradient = algo_second_gradient.getResult()
    myKriging_second_grad = result_second_gradient.getMetaModel()
    
    
    
    #下面是画图部分
    meta = myKriging_second_grad#myKriging_grad
    graphKriging = meta.draw(xMin, xMax)
    #graphKriging.setLegends(["GPR"])
    # Create a grid of points and evaluate the function and the kriging
    nbpoints = 250
    xGrid = linearSample(xMin, xMax, nbpoints)
    #yFunction = g(xGrid)
    yKrig = meta(xGrid)#model_second_gradient(xGrid, myKriging_grad,d) #
    # Compute the conditional covariance
    epsilon = ot.Sample(nbpoints, [1.e-8])
    conditionalVariance = result_second_gradient.getConditionalMarginalVariance(xGrid)+epsilon
    conditionalSigma = sqrt(conditionalVariance)
    # Compute the quantile of the Normal distribution
    level=0.95
    alpha = 1-(1-level)/2
    quantileAlpha = ot.DistFunc.qNormal(alpha)
    # Graphics of the bounds
    epsilon = 1.e-8
    dataLower = [yKrig[i, 0] - quantileAlpha * conditionalSigma[i, 0]
                 for i in range(nbpoints)]
    dataUpper = [yKrig[i, 0] + quantileAlpha * conditionalSigma[i, 0]
                 for i in range(nbpoints)]
    # Coordinates of the vertices of the Polygons
    vLow = [[xGrid[i, 0], dataLower[i]] for i in range(nbpoints)]
    vUp = [[xGrid[i, 0], dataUpper[i]] for i in range(nbpoints)]
    # Compute the Polygon graphics
    boundsPoly = plot_kriging_bounds(vLow, vUp, nbpoints)
    #boundsPoly.setLegend("95% bounds")

    # Plot the function

    grad_2st_plot = np.zeros(shape=(X.getSize(),d))
    for i in range(0,X.getSize()): # 这个地方就是后面要换，根据代理模型来建立
        grad_2st_plot[i]=numberical_grad.grad_2nd_ot_1dim(myKriging_grad, X[i])
        
    grad_2st_plot = abs(grad_2st_plot)
    
    cloudDOE = ot.Cloud(X[0:-1], grad_2st_plot[0:-1])
    cloudDOE.setPointStyle("circle")
    cloudDOE.setColor("#33CC33")
    #cloudDOE.setLegend("Existing Data")
    
    cloudDOE_new = ot.Cloud(X[-1], grad_2st_plot[-1])
    cloudDOE_new.setPointStyle("star")
    cloudDOE_new.setColor("red")
    #cloudDOE_new.setLegend("Added Data")

    graphFonction = ot.Curve(xGrid, yKrig)
    graphFonction.setLineStyle("dotdash")
    graphFonction.setColor("black")
    graphFonction.setLineWidth(2)

    
    # Assemble the graphics
    graph = ot.Graph()
    graph.add(boundsPoly)
    graph.add(graphFonction)
    graph.add(cloudDOE)
    graph.add(cloudDOE_new)
    #graph.add(graphKriging)
    #graph.setLegendPosition("topright")
    graph.setAxes(True)
    graph.setGrid(False)
    graph.setXMargin(0.01)
    #graph.setTitle("Gradient view for Bayesian optimization (Sample Size = %d)" % (samplesize))
    #graph.setXTitle("$X$" )
    graph.setYTitle("$Y_{BO}$")
    #plt.ylim(-100, 125)
    return graph



def plot_second_gradient_initial(xMin, xMax, X, Y, getatable_1stgrad, selected_function, in_min, level=0.95):
    d=1
    samplesize = X.getSize()
    sqrt = ot.SymbolicFunction(["x"], ["sqrt(x)"])
    X_plot=X
    Y_plot=Y
    
    in_max=50
    basis =  ot.ConstantBasisFactory(d).build()
    covarianceModel =  ot.MaternModel([1.]*d, 1.5)
    grad_1st = np.zeros(shape=(X_plot.getSize(),d))
    if getatable_1stgrad:
        for i in range(0,X_plot.getSize()):
            grad_1st[i] = numberical_grad.grad(selected_function, X_plot[i])
    else:
        Y_plot = selected_function(X_plot)
        algo_preorder = ot.KrigingAlgorithm(X_plot, 
                                            Y_plot, 
                                            covarianceModel, 
                                            basis)
        ot.Log.SetFile('Warnings.log')
        algo_preorder.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
        algo_preorder.run()
        result_preorder = algo_preorder.getResult()

        krig_preorder = result_preorder.getMetaModel()
            
        for i in range(0,X_plot.getSize()):
            grad_1st[i] =numberical_grad.grad_ungetatable_1st_ot(krig_preorder, X_plot[i])
        
# 然后是对梯度建立代理模型
    algo_gradient = ot.KrigingAlgorithm(X_plot, 
                                        ot.Sample(grad_1st), 
                                        covarianceModel, basis)
    algo_gradient.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_gradient.run()
    result_gradient = algo_gradient.getResult()
    myKriging_grad = result_gradient.getMetaModel()
    
    #experiment = ot.MonteCarloExperiment( whole_region_initial_ot, int(1000))
    #x_fix = experiment.generate()
    x_fix =X_plot
    
    grad_2st = np.zeros(shape=(x_fix.getSize(),d))
    for i in range(0,x_fix.getSize()): # 这个地方就是后面要换，根据代理模型来建立
        grad_2st[i]=numberical_grad.grad_2nd_ot_1dim(myKriging_grad, x_fix[i])
        
    sum_grad_2st = abs(grad_2st)
                   
    algo_second_gradient = ot.KrigingAlgorithm(x_fix, 
                                        ot.Sample(sum_grad_2st), 
                                        covarianceModel, basis)
    algo_second_gradient.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_second_gradient.run()
    result_second_gradient = algo_second_gradient.getResult()
    myKriging_second_grad = result_second_gradient.getMetaModel()
    
    
    
    #下面是画图部分
    meta = myKriging_second_grad#myKriging_grad
    #graphKriging = meta.draw(xMin, xMax)
    #graphKriging.setLegends(["GPR"])
    # Create a grid of points and evaluate the function and the kriging
    nbpoints = 250
    xGrid = linearSample(xMin, xMax, nbpoints)
    #yFunction = g(xGrid)
    yKrig = meta(xGrid)#model_second_gradient(xGrid, myKriging_grad,d) #
    # Compute the conditional covariance
    epsilon = ot.Sample(nbpoints, [1.e-8])
    conditionalVariance = result_second_gradient.getConditionalMarginalVariance(xGrid)+epsilon
    conditionalSigma = sqrt(conditionalVariance)
    # Compute the quantile of the Normal distribution
    level=0.95
    alpha = 1-(1-level)/2
    quantileAlpha = ot.DistFunc.qNormal(alpha)
    # Graphics of the bounds
    epsilon = 1.e-8
    dataLower = [yKrig[i, 0] - quantileAlpha * conditionalSigma[i, 0]
                 for i in range(nbpoints)]
    dataUpper = [yKrig[i, 0] + quantileAlpha * conditionalSigma[i, 0]
                 for i in range(nbpoints)]
    # Coordinates of the vertices of the Polygons
    vLow = [[xGrid[i, 0], dataLower[i]] for i in range(nbpoints)]
    vUp = [[xGrid[i, 0], dataUpper[i]] for i in range(nbpoints)]
    # Compute the Polygon graphics
    boundsPoly = plot_kriging_bounds(vLow, vUp, nbpoints)
    #boundsPoly.setLegend("95% bounds")

    # Plot the function

    grad_2st_plot = np.zeros(shape=(X.getSize(),d))
    for i in range(0,X.getSize()): # 这个地方就是后面要换，根据代理模型来建立
        grad_2st_plot[i]=numberical_grad.grad_2nd_ot_1dim(myKriging_grad, X[i])
        
    grad_2st_plot = abs(grad_2st_plot)


    graphFonction = ot.Curve(xGrid, yKrig)
    graphFonction.setLineStyle("dotdash")
    graphFonction.setColor("black")
    graphFonction.setLineWidth(2)
    #graphFonction.setLegend("Absolute of estimated 2nd-order gradient")

    
    cloudDOE = ot.Cloud(X, grad_2st_plot)
    cloudDOE.setPointStyle("circle")
    cloudDOE.setColor("#33CC33")
    #cloudDOE.setLegend("Existing Data")
    
    
    
    # Assemble the graphics
    graph = ot.Graph()
    graph.add(boundsPoly)
    graph.add(graphFonction)
    graph.add(cloudDOE)
    #graph.add(graphKriging)
    #graph.setLegendPosition("topright")
    graph.setAxes(True)
    graph.setGrid(False)
    graph.setXMargin(0.01)
    
    #graph.setTitle("Gradient view for Bayesian optimization (Sample Size = %d)" % (samplesize))
    #graph.setXTitle("$X$" )
    graph.setYTitle("$Y_{BO}$")

    return graph


def plot_AF_initial(xMin, xMax, X, Y, getatable_1stgrad, selected_function, in_min, level=0.95):
    d=1
    samplesize = X.getSize()
    sqrt = ot.SymbolicFunction(["x"], ["sqrt(x)"])
    X_plot=X
    Y_plot=Y
    
    in_max=50
    basis =  ot.ConstantBasisFactory(d).build()
    covarianceModel =  ot.MaternModel([1.]*d, 1.5)
    grad_1st = np.zeros(shape=(X_plot.getSize(),d))
    if getatable_1stgrad:
        for i in range(0,X_plot.getSize()):
            grad_1st[i] = numberical_grad.grad(selected_function, X_plot[i])
    else:
        Y_plot = selected_function(X_plot)
        algo_preorder = ot.KrigingAlgorithm(X_plot, 
                                            Y_plot, 
                                            covarianceModel, 
                                            basis)
        ot.Log.SetFile('Warnings.log')
        algo_preorder.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
        algo_preorder.run()
        result_preorder = algo_preorder.getResult()

        krig_preorder = result_preorder.getMetaModel()
            
        for i in range(0,X_plot.getSize()):
            grad_1st[i] =numberical_grad.grad_ungetatable_1st_ot(krig_preorder, X_plot[i])
        
# 然后是对梯度建立代理模型
    algo_gradient = ot.KrigingAlgorithm(X_plot, 
                                        ot.Sample(grad_1st), 
                                        covarianceModel, basis)
    algo_gradient.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_gradient.run()
    result_gradient = algo_gradient.getResult()
    myKriging_grad = result_gradient.getMetaModel()
    
    #experiment = ot.MonteCarloExperiment( whole_region_initial_ot, int(1000))
    #x_fix = experiment.generate()
    x_fix =X_plot
    
    grad_2st = np.zeros(shape=(x_fix.getSize(),d))
    for i in range(0,x_fix.getSize()): # 这个地方就是后面要换，根据代理模型来建立
        grad_2st[i]=numberical_grad.grad_2nd_ot_1dim(myKriging_grad, x_fix[i])


    # 下面开始调用     
    X_train = X
        
    sum_grad_2st = abs(grad_2st)
           
    y_train = sum_grad_2st
    
    # settings 贝叶斯优化的设置 
    number_of_samples = 300 #候选点的数目，这个选的比较大，因为是从这个里面选的，如果不想让点之间太近，把这个减小应该也可以
    acquisition_function_flag =2 #直接对二阶梯度做0，把二阶梯度作为正则项是1-3 0: Estimated y-values 1: Mutual information (MI), 2: Expected Improvement(EI), 3: Probability of improvement (PI)

            
    X_candidate = np.linspace(0, 1, 100) .reshape(1, -1).T
    X_train=np.array(X_train)
    
    cumulative_variance = np.zeros(len(X_candidate))

    out_af=bayesianoptimization_af(X_train, y_train, X_candidate ,
                            acquisition_function_flag, cumulative_variance)


        
    sum_grad_2st = abs(grad_2st)
                   
    algo_second_gradient = ot.KrigingAlgorithm(x_fix, 
                                        ot.Sample(sum_grad_2st), 
                                        covarianceModel, basis)
    algo_second_gradient.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_second_gradient.run()
    result_second_gradient = algo_second_gradient.getResult()
    myKriging_second_grad = result_second_gradient.getMetaModel()
   
      
    #下面是画图部分
    meta = myKriging_second_grad#myKriging_grad
    #graphKriging = meta.draw(xMin, xMax)
    #graphKriging.setLegends(["GPR"])
    # Create a grid of points and evaluate the function and the kriging
    nbpoints = 250
    xGrid = linearSample(xMin, xMax, nbpoints)
    #yFunction = g(xGrid)
    yKrig = meta(xGrid)#model_second_gradient(xGrid, myKriging_grad,d) #
    # Compute the conditional covariance
    epsilon = ot.Sample(nbpoints, [1.e-8])
    conditionalVariance = result_second_gradient.getConditionalMarginalVariance(xGrid)+epsilon
    conditionalSigma = sqrt(conditionalVariance)
    # Compute the quantile of the Normal distribution
    level=0.95
    alpha = 1-(1-level)/2
    quantileAlpha = ot.DistFunc.qNormal(alpha)
    # Graphics of the bounds
    epsilon = 1.e-8
    dataLower = [yKrig[i, 0] - quantileAlpha * conditionalSigma[i, 0]
                 for i in range(nbpoints)]
    dataUpper = [yKrig[i, 0] + quantileAlpha * conditionalSigma[i, 0]
                 for i in range(nbpoints)]
    # Coordinates of the vertices of the Polygons
    vLow = [[xGrid[i, 0], dataLower[i]] for i in range(nbpoints)]
    vUp = [[xGrid[i, 0], dataUpper[i]] for i in range(nbpoints)]
    # Compute the Polygon graphics
    boundsPoly = plot_kriging_bounds(vLow, vUp, nbpoints)
    #boundsPoly.setLegend("95% bounds")

    # Plot the function

    grad_2st_plot = np.zeros(shape=(X.getSize(),d))
    for i in range(0,X.getSize()): # 这个地方就是后面要换，根据代理模型来建立
        grad_2st_plot[i]=numberical_grad.grad_2nd_ot_1dim(myKriging_grad, X[i])
        
    grad_2st_plot = abs(grad_2st_plot)


    graphFonction = ot.Curve(xGrid, yKrig)
    graphFonction.setLineStyle("dotdash")
    graphFonction.setColor("black")
    graphFonction.setLineWidth(2)
    #graphFonction.setLegend("Absolute of estimated 2nd-order gradient")
 
     
    cloudDOE = ot.Cloud(X, grad_2st_plot)
    cloudDOE.setPointStyle("circle")
    cloudDOE.setColor("#33CC33")
    #cloudDOE.setLegend("Existing Data")
    
    
    
    # Assemble the graphics
    graph = ot.Graph()
    graph.add(boundsPoly)
    graph.add(graphFonction)
    graph.add(cloudDOE)
    #graph.add(graphKriging)
    #graph.setLegendPosition("topright")
    graph.setAxes(True)
    graph.setGrid(False)
    graph.setXMargin(0.01)
     
    #graph.setTitle("Gradient view for Bayesian optimization (Sample Size = %d)" % (samplesize))
    #graph.setXTitle("$X$" )
    graph.setYTitle("$Y_{BO}$")

    return graph



