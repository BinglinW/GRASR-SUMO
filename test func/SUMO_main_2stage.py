# -*- coding: utf-8 -*-
"""

@author: wbl
"""


import numpy as np
import numberical_grad
import phy_sim
import math
import evaluate_region 

import copy
import divide
import heapq
import openturns as ot
import single_stage_ot
from bayesianoptimization import bayesianoptimization
import multiprocessing
import os
import sys 
import time   
import getopt
import math
import scipy.integrate as integrate

d=2
n_0 =20

whole_region_initial = ([[-10, 10],[-10, 10]])
selected_function = ot.PythonFunction(d, 1, 
                                   func=phy_sim.dixonpr, #mccorm_ot_single,
                                   func_sample=phy_sim.dixonpr) #mccorm_ot_multi

                                 
strategy=1
if_optim= True
if_alBO = True
basis =  ot.ConstantBasisFactory(d).build()
covarianceModel =  ot.MaternModel([1.]*d, 1.5)

max_distance_0=whole_region_initial[0][1]-whole_region_initial[0][0]
max_distance_1=whole_region_initial[1][1]-whole_region_initial[1][0]
max_distance=max(max_distance_0,max_distance_1)


in_min=0.1
in_max=100

if __name__=='__main__':
    opts,_=getopt.getopt(sys.argv[1:],'i:n:',[])
    for opt,value in opts:
        if opt == "-i":
            n_min=value
        if opt == "-n":
            file_i=value
    ot.RandomGenerator.SetSeed(int(file_i))            

    """
    能直接获得一阶段梯度
    """

    getatable_1stgrad= True
    

    """
    1st time region segmentation
    """

    whole_region_initial_ot=divide.ot_region(whole_region_initial, d)


    bounds = whole_region_initial_ot.getRange()
    lhs = ot.LHSExperiment(whole_region_initial_ot, n_0)
    lhs.setAlwaysShuffle(True)
    space_filling = ot.SpaceFillingPhiP()
    temperatureProfile = ot.GeometricProfile(10.0, 0.95, 1000)
    algo = ot.SimulatedAnnealingLHS(lhs, space_filling, temperatureProfile)

    X_0 = algo.generate()
    X_write0 = np.array(X_0, copy=False)
    
    Y_0 = selected_function(X_0)

    algo_0 = ot.KrigingAlgorithm(X_0, Y_0, covarianceModel, basis)
    algo_0.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    ot.Log.SetFile('Warnings.log')
    algo_0.run()
    result = algo_0.getResult()


    krig = result.getMetaModel()
    y_0_pred= krig (X_0)
    


    Y_gradient_0 = np.zeros(shape=(n_0,d))
    for i in range(0,n_0):
        if getatable_1stgrad:
            Y_gradient_0[i] = numberical_grad.grad(selected_function, X_0[i])
        else:
            Y_gradient_0[i] =numberical_grad.grad_ungetatable_1st_ot(krig, X_0[i])


    algo_gradient_0 = ot.KrigingAlgorithm(X_0, 
                                      ot.Sample(Y_gradient_0), 
                                      covarianceModel, basis)
    algo_gradient_0.setOptimizationBounds(ot.Interval([in_min]*d*2, [in_max]*d*2))
    algo_gradient_0.run()
    result_gradient_0 = algo_gradient_0.getResult()

    myKriging_grad0 = result_gradient_0.getMetaModel()

    experiment_virtual0_ot = ot.LHSExperiment(whole_region_initial_ot, 300, False, True)
    X_virtual0_ot = experiment_virtual0_ot.generate()
    X_virtual0 = np.array(X_virtual0_ot)

    grad_virtual0_ot = myKriging_grad0 (X_virtual0_ot)
    grad_virtual0 = np.array(grad_virtual0_ot)

    absY_gradient0 = abs(grad_virtual0)

    cv_whole = evaluate_region.variation_coefficient(absY_gradient0)

    disper_1=cv_whole
    print(disper_1)
    
    disper_1=disper_1.tolist()
    min_index_initial=[]
    min_index=disper_1.index(min(disper_1))


    sum_grad_1=sum(absY_gradient0[:,min_index])
    weight_1=absY_gradient0[:,min_index]/sum_grad_1
    centroid=sum(X_virtual0[:,min_index]*weight_1)


    subgrad_score_dim_0= np.zeros(shape=(9,1))
    coff=0.1

    for i_range in range(1,10):
        coff_bandwidth=0.1*i_range  
        whole_region_0 = divide.shrink_range(coff_bandwidth, 
                                           whole_region_initial, 
                                           min_index, 
                                           centroid)

        whole_region_0_ot=divide.ot_region(whole_region_0[0], d)
      
        n_per_v=50
        experiment_shrink_ot = ot.LHSExperiment(whole_region_0_ot, n_per_v, False, True)
        X_shrink_ot = experiment_shrink_ot.generate()
        X_shrink = np.array(X_shrink_ot)

        if whole_region_initial[min_index][0] != whole_region_0[0][min_index][0]\
        and whole_region_initial[min_index][1] != whole_region_0[0][min_index][1]:
         
            region_right = copy.deepcopy(whole_region_initial)
            region_right[min_index][0] = whole_region_0[0][min_index][1] 
      
            region_left = copy.deepcopy(whole_region_initial)
            region_left[min_index][1] = whole_region_0[0][min_index][0]

            length_right=abs( whole_region_initial[min_index][1] - whole_region_0[0][min_index][1])
            length_left=abs( whole_region_initial[min_index][0] - whole_region_0[0][min_index][0])
            if length_right<length_left:
                n_right=math.ceil( n_per_v*(length_right/(length_right+length_left)))
                n_left=n_per_v-n_right
            else:
                n_left=math.ceil( n_per_v*(length_left/(length_right+length_left)))
                n_right=n_per_v-n_left

            region_right_ot=divide.ot_region(region_right, d)
            right_apart_shrink_ot = ot.LHSExperiment(region_right_ot, 
                                               n_right, False, True)
            X_apart_right_ot = right_apart_shrink_ot.generate()      
            X_apart_right = np.array(X_apart_right_ot)

            region_left_ot=divide.ot_region(region_left, d)
            left_apart_shrink_ot = ot.LHSExperiment(region_left_ot, 
                                               n_left, False, True)
            X_apart_left_ot = left_apart_shrink_ot.generate()     
            X_apart_left = np.array(X_apart_left_ot)
          
            X_apart=np.row_stack((X_apart_right,X_apart_left))

        elif whole_region_initial[min_index][0] == whole_region_0[0][min_index][0]:
            region_right = copy.deepcopy(whole_region_initial)
            region_right[min_index][0] = whole_region_0[0][min_index][1] 
           
            region_right_ot=divide.ot_region(region_right, d)
            experiment_apart_shrink_ot = ot.LHSExperiment(region_right_ot, 
                                               n_per_v, False, True)
            X_apart_shrink_ot = experiment_apart_shrink_ot.generate()
            X_apart=np.array(X_apart_shrink_ot)
        
        elif whole_region_initial[min_index][1] == whole_region_0[0][min_index][1]:
            region_left = copy.deepcopy(whole_region_initial)
            region_left[min_index][1] = whole_region_0[0][min_index][0]
           
            region_left_ot=divide.ot_region(region_left, d)
           
            experiment_apart_shrink_ot = ot.LHSExperiment(region_left_ot, 
                                               n_per_v, False, True)
            X_apart_shrink_ot = experiment_apart_shrink_ot.generate()      
            X_apart=np.array(X_apart_shrink_ot)
 
        elif whole_region_initial[min_index][1] == whole_region_0[0][min_index][1]:
            region_left = copy.deepcopy(whole_region_initial)
            region_left[min_index][1] = whole_region_0[0][min_index][0]

            region_left_ot=divide.ot_region(region_left, d)

            experiment_apart_shrink_ot = ot.LHSExperiment(region_left_ot, 
                                               n_per_v, False, True)
            X_apart_shrink_ot = experiment_apart_shrink_ot.generate()      
            X_apart=np.array(X_apart_shrink_ot)
         


        judge_X_apart= np.zeros(shape=(n_per_v,1 ))
        analyzed_grad_part1=np.column_stack((judge_X_apart,X_apart))

        judge_X_shrink= np.ones(shape=(n_per_v,1 ))
        analyzed_grad_part2=np.column_stack((judge_X_shrink,X_shrink))
     
        X_vir2=np.row_stack((X_apart, X_shrink))
        judge_region_0=np.row_stack((judge_X_apart, judge_X_shrink))
        grad_virtual2_ot = myKriging_grad0 (X_vir2)
        grad_virtual2 = np.array(grad_virtual2_ot)
      
        absY_gradient2= abs(grad_virtual2)

        sumY_gradient2 = np.sum(absY_gradient2, axis=1)

        analyzed_grad_0 = np.column_stack((judge_region_0,sumY_gradient2))

        n_repeat=1
        subgrad_score_dim_0_temp =np.zeros(shape=(n_repeat,1))

        try:
            for i in range(0,n_repeat):
                subgrad_score_dim_0_temp[i,0]=evaluate_region.BF(analyzed_grad_0, 1,file_i)[0]
        except:
            for i in range(0,n_repeat):
                subgrad_score_dim_0_temp[i,0]=0
   
        subgrad_score_dim_0_temp0 = np.mean(subgrad_score_dim_0_temp)

        subgrad_score_dim_0[i_range-1,0]=subgrad_score_dim_0_temp0

    subgrad_array=  np.asarray(subgrad_score_dim_0)  
    max_index_array = np.argsort(-subgrad_array.T)[0][0:3]
    max_index_0= max_index_array.tolist()


    possible_range_set= (np.asarray(max_index_0)+1)* coff
    possible_range_set = sorted(possible_range_set,reverse=True)

    print(possible_range_set)

    sorted_index = sorted(np.asarray(max_index_0),reverse=True)

    BF_evidence=subgrad_score_dim_0[sorted_index[strategy]]

    min_index_initial=[]
    if BF_evidence>1.8:
        range_selected=possible_range_set[strategy]
        whole_region_0 = divide.shrink_range(range_selected, 
                                        whole_region_initial, 
                                        min_index, 
                                        centroid)[0] 
        BO_2stgrad=True
        min_index_0=[min_index]
        min_bo_0=[min_index]

    else:
        print("no enough evidence")
        whole_region_0 = whole_region_initial    
        BO_2stgrad=if_alBO 
        min_bo_0=min_index_initial#!!!!!!!!!!!!!!!!
        min_index_0=[min_index]#!!!!!!!!!!!!!!!!

    out = single_stage_ot.single_stage_BO_traversal( selected_function,
                                       whole_region_0, 
                                       strategy, d,
                                       n_0, 
                                       min_index_0, 
                                       min_bo_0,# !!!!!!!!! 
                                       X_0, 
                                       n_min, 
                                       getatable_1stgrad, 
                                       BO_2stgrad,
                                       if_optim, 
                                       if_alBO,
                                       Y_0,
                                       file_i,
                                       max_distance )

    whole_region_1=copy.deepcopy(out[0]) 
    X_1 =copy.deepcopy(out[1]) 
    Y_1 =copy.deepcopy(out[2]) 
    min_index_1 =copy.deepcopy(out[3]) 
    BO_2stgrad=copy.deepcopy(out[6]) 
    min_bo_1=copy.deepcopy(out[7]) 

  
    X_seg_True = X_1
    X_seg_True_ot = ot.Sample(X_seg_True)
     
    N_seg=X_seg_True.shape[0]

    Y_seg_True_ot = Y_1
    Y_seg_True = np.array(Y_seg_True_ot)

    algo_seg_True = ot.KrigingAlgorithm(X_seg_True_ot, 
                                        Y_seg_True_ot, 
                                        covarianceModel,
                                        basis)
    algo_seg_True.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_seg_True.run()
    result_seg_True = algo_seg_True.getResult()

    krig_seg_True = result_seg_True.getMetaModel()


  
    experiment_compare1_ot = ot.LHSExperiment(whole_region_initial_ot, N_seg, False, False)
    X_compare1_ot = experiment_compare1_ot.generate()
    Y_compare1_ot= selected_function(X_compare1_ot)


    algo_compare1_opi = ot.KrigingAlgorithm(X_compare1_ot, 
                                       Y_compare1_ot, 
                                       covarianceModel, 
                                       basis)
    algo_compare1_opi.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_compare1_opi.run()
    result_compare1_opi = algo_compare1_opi.getResult()

    krig_compare1_opi = result_compare1_opi.getMetaModel()



    lhs_compare2 = ot.LHSExperiment(whole_region_initial_ot, N_seg)
    lhs_compare2.setAlwaysShuffle(True)  # randomized

    space_filling_compare2 = ot.SpaceFillingPhiP()

    temperatureProfile_compare2 = ot.GeometricProfile(10.0, 0.95, 1000)
    algo_compare2 = ot.SimulatedAnnealingLHS(lhs_compare2, 
                                    space_filling_compare2, 
                                    temperatureProfile_compare2)
    X_compare2_ot = algo_compare2.generate()
        
    Y_compare2_ot= selected_function(X_compare2_ot)


    algo_compare2_opi = ot.KrigingAlgorithm(X_compare2_ot, 
                                        Y_compare2_ot, 
                                        covarianceModel, 
                                        basis)

    algo_compare2_opi.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_compare2_opi.run()
    result_compare2_opi = algo_compare2_opi.getResult()

    krig_compare2_opi = result_compare2_opi.getMetaModel()


    number_of_samples = 5000 #
    number_of_iteration = int(n_min)#
    acquisition_function_flag = 1  #
    X_train=X_write0
    y_train=np.array(Y_0)
    experiment_candidate_ot = ot.LHSExperiment(whole_region_initial_ot, number_of_samples, False, True)
    X_candidate_ot = experiment_candidate_ot.generate()
 
    X_candidate = np.array(X_candidate_ot)
    cumulative_variance = np.zeros(len(X_candidate))
    for iteration in range(number_of_iteration):

           selected_candidate_number, selected_X_candidate, cumulative_variance = bayesianoptimization(X_train, 
                                                                                  y_train, 
                                                                                  X_candidate ,
                                                                                  acquisition_function_flag,
                                                                                  cumulative_variance)

           X_append = np.reshape(selected_X_candidate, (1, X_train.shape[1]))
           X_train = np.append(X_train, 
                               X_append,
                               0)

           y_add = np.array(selected_function(X_append))
                      
           y_train = np.append(y_train, y_add)
           X_candidate = np.delete(X_candidate, selected_candidate_number, 0)
           cumulative_variance = np.delete(cumulative_variance, selected_candidate_number)

    X_BO = X_train
    Y_BO = np.expand_dims(y_train, axis=1)
    algo_BO = ot.KrigingAlgorithm(ot.Sample(X_BO), 
                                        ot.Sample(Y_BO), 
                                        covarianceModel, 
                                        basis)

    algo_BO.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_BO.run()
    result_BO = algo_BO.getResult()

    krig_BO = result_BO.getMetaModel()


    experiment_test_ot = ot.MonteCarloExperiment(whole_region_initial_ot, 2000)
    X_test_ot = experiment_test_ot.generate()
    Y_test_ot= selected_function(X_test_ot)
    Y_test = np.array(Y_test_ot)


    y_test_seg_True_ot = krig_seg_True (X_test_ot)

    Y_hat_seg_True = np.array(y_test_seg_True_ot)

    y_test_compare1_ot_opi= krig_compare1_opi (X_test_ot)
    y_test_compare2_ot_opi= krig_compare2_opi (X_test_ot)
    y_test_BO= krig_BO (X_test_ot)

    Y_hat_compare1_opi = np.array(y_test_compare1_ot_opi)
    Y_hat_compare2_opi = np.array(y_test_compare2_ot_opi)
    Y_hat_BO = np.array(y_test_BO)
    
    MSE_K_compare1_opi = np.linalg.norm(Y_test-Y_hat_compare1_opi, ord=2)**2/len(Y_test)
    MSE_K_compare2_opi = np.linalg.norm(Y_test-Y_hat_compare2_opi, ord=2)**2/len(Y_test)
    MSE_K_seg_True = np.linalg.norm(Y_test-Y_hat_seg_True, ord=2)**2/len(Y_test)
    MSE_K_BO = np.linalg.norm(Y_test-Y_hat_BO, ord=2)**2/len(Y_test)
    

    getatable_1stgrad= False


 
    whole_region_initial_ot=divide.ot_region(whole_region_initial, d)


    bounds = whole_region_initial_ot.getRange()
    lhs = ot.LHSExperiment(whole_region_initial_ot, n_0)
    lhs.setAlwaysShuffle(True)
    space_filling = ot.SpaceFillingPhiP()
    temperatureProfile = ot.GeometricProfile(10.0, 0.95, 1000)
    algo = ot.SimulatedAnnealingLHS(lhs, space_filling, temperatureProfile)

    X_0 = algo.generate()
    X_write0 = np.array(X_0, copy=False)

    Y_0 = selected_function(X_0)

    algo_0 = ot.KrigingAlgorithm(X_0, Y_0, covarianceModel, basis)
    ot.Log.SetFile('Warnings.log')
    algo_0.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_0.run()
    result = algo_0.getResult()

    krig = result.getMetaModel()
    y_0_pred= krig (X_0)
    


    Y_gradient_0 = np.zeros(shape=(n_0,d))
    for i in range(0,n_0):
        if getatable_1stgrad:
            Y_gradient_0[i] = numberical_grad.grad(selected_function, X_0[i])
        else:
            Y_gradient_0[i] =numberical_grad.grad_ungetatable_1st_ot(krig, X_0[i])


    algo_gradient_0 = ot.KrigingAlgorithm(X_0, 
                                      ot.Sample(Y_gradient_0), 
                                      covarianceModel, basis)

    algo_gradient_0.setOptimizationBounds(ot.Interval([in_min]*d*2, [in_max]*d*2))
    algo_gradient_0.run()
    result_gradient_0 = algo_gradient_0.getResult()
   
    myKriging_grad0 = result_gradient_0.getMetaModel()

    experiment_virtual0_ot = ot.LHSExperiment(whole_region_initial_ot, 300, False, True)
    X_virtual0_ot = experiment_virtual0_ot.generate()
    X_virtual0 = np.array(X_virtual0_ot)

    grad_virtual0_ot = myKriging_grad0 (X_virtual0_ot)
    grad_virtual0 = np.array(grad_virtual0_ot)

    absY_gradient0 = abs(grad_virtual0)

    cv_whole = evaluate_region.variation_coefficient(absY_gradient0)

    disper_1=cv_whole
    print(disper_1)
    
    disper_1=disper_1.tolist()
    min_index=disper_1.index(min(disper_1))


    sum_grad_1=sum(absY_gradient0[:,min_index])
    weight_1=absY_gradient0[:,min_index]/sum_grad_1
    centroid=sum(X_virtual0[:,min_index]*weight_1)


    subgrad_score_dim_0= np.zeros(shape=(9,1))
    coff=0.1

    for i_range in range(1,10):
        coff_bandwidth=0.1*i_range  
        whole_region_0 = divide.shrink_range(coff_bandwidth, 
                                           whole_region_initial, 
                                           min_index, 
                                           centroid)

        whole_region_0_ot=divide.ot_region(whole_region_0[0], d)
      
        n_per_v=50
        experiment_shrink_ot = ot.LHSExperiment(whole_region_0_ot, n_per_v, False, True)
        X_shrink_ot = experiment_shrink_ot.generate()
        X_shrink = np.array(X_shrink_ot)

        if whole_region_initial[min_index][0] != whole_region_0[0][min_index][0]\
        and whole_region_initial[min_index][1] != whole_region_0[0][min_index][1]:
         
            region_right = copy.deepcopy(whole_region_initial)
            region_right[min_index][0] = whole_region_0[0][min_index][1] 
      
            region_left = copy.deepcopy(whole_region_initial)
            region_left[min_index][1] = whole_region_0[0][min_index][0]

            length_right=abs( whole_region_initial[min_index][1] - whole_region_0[0][min_index][1])
            length_left=abs( whole_region_initial[min_index][0] - whole_region_0[0][min_index][0])
            if length_right<length_left:
                n_right=math.ceil( n_per_v*(length_right/(length_right+length_left)))
                n_left=n_per_v-n_right
            else:
                n_left=math.ceil( n_per_v*(length_left/(length_right+length_left)))
                n_right=n_per_v-n_left

            region_right_ot=divide.ot_region(region_right, d)
            right_apart_shrink_ot = ot.LHSExperiment(region_right_ot, 
                                               n_right, False, True)
            X_apart_right_ot = right_apart_shrink_ot.generate()      
            X_apart_right = np.array(X_apart_right_ot)

            region_left_ot=divide.ot_region(region_left, d)
            left_apart_shrink_ot = ot.LHSExperiment(region_left_ot, 
                                               n_left, False, True)
            X_apart_left_ot = left_apart_shrink_ot.generate()     
            X_apart_left = np.array(X_apart_left_ot)
          
            X_apart=np.row_stack((X_apart_right,X_apart_left))

        elif whole_region_initial[min_index][0] == whole_region_0[0][min_index][0]:
            region_right = copy.deepcopy(whole_region_initial)
            region_right[min_index][0] = whole_region_0[0][min_index][1] 
           
            region_right_ot=divide.ot_region(region_right, d)
            experiment_apart_shrink_ot = ot.LHSExperiment(region_right_ot, 
                                               n_per_v, False, True)
            X_apart_shrink_ot = experiment_apart_shrink_ot.generate()
            X_apart=np.array(X_apart_shrink_ot)
        
        elif whole_region_initial[min_index][1] == whole_region_0[0][min_index][1]:
            region_left = copy.deepcopy(whole_region_initial)
            region_left[min_index][1] = whole_region_0[0][min_index][0]
           
            region_left_ot=divide.ot_region(region_left, d)
           
            experiment_apart_shrink_ot = ot.LHSExperiment(region_left_ot, 
                                               n_per_v, False, True)
            X_apart_shrink_ot = experiment_apart_shrink_ot.generate()      
            X_apart=np.array(X_apart_shrink_ot)
 
        elif whole_region_initial[min_index][1] == whole_region_0[0][min_index][1]:
            region_left = copy.deepcopy(whole_region_initial)
            region_left[min_index][1] = whole_region_0[0][min_index][0]

            region_left_ot=divide.ot_region(region_left, d)

            experiment_apart_shrink_ot = ot.LHSExperiment(region_left_ot, 
                                               n_per_v, False, True)
            X_apart_shrink_ot = experiment_apart_shrink_ot.generate()      
            X_apart=np.array(X_apart_shrink_ot)
         


        judge_X_apart= np.zeros(shape=(n_per_v,1 ))
        analyzed_grad_part1=np.column_stack((judge_X_apart,X_apart))

        judge_X_shrink= np.ones(shape=(n_per_v,1 ))
        analyzed_grad_part2=np.column_stack((judge_X_shrink,X_shrink))
     
        X_vir2=np.row_stack((X_apart, X_shrink))
        judge_region_0=np.row_stack((judge_X_apart, judge_X_shrink))
        grad_virtual2_ot = myKriging_grad0 (X_vir2)
        grad_virtual2 = np.array(grad_virtual2_ot)
      
        absY_gradient2= abs(grad_virtual2)

        sumY_gradient2 = np.sum(absY_gradient2, axis=1)

        analyzed_grad_0 = np.column_stack((judge_region_0,sumY_gradient2))

        n_repeat=1
        subgrad_score_dim_0_temp =np.zeros(shape=(n_repeat,1))
        try:
            for i in range(0,n_repeat):
                subgrad_score_dim_0_temp[i,0]=evaluate_region.BF(analyzed_grad_0, 1,file_i)[0]
        except:
            for i in range(0,n_repeat):
                subgrad_score_dim_0_temp[i,0]=0
   
        subgrad_score_dim_0_temp0 = np.mean(subgrad_score_dim_0_temp)

        subgrad_score_dim_0[i_range-1,0]=subgrad_score_dim_0_temp0

    subgrad_array=  np.asarray(subgrad_score_dim_0)  
    max_index_array = np.argsort(-subgrad_array.T)[0][0:3]
    max_index_0= max_index_array.tolist()


    possible_range_set= (np.asarray(max_index_0)+1)* coff
    possible_range_set = sorted(possible_range_set,reverse=True)

    print(possible_range_set)

    sorted_index = sorted(np.asarray(max_index_0),reverse=True)

    BF_evidence=subgrad_score_dim_0[sorted_index[strategy]]

    min_index_initial=[]#!!!!!!!!!!
    if BF_evidence>1.8:
        range_selected=possible_range_set[strategy]
        whole_region_0 = divide.shrink_range(range_selected, 
                                        whole_region_initial, 
                                        min_index, 
                                        centroid)[0] 
        BO_2stgrad=True
        min_index_0=[min_index]
        min_bo_0=[min_index]

    else:
        print("no enough evidence")
        whole_region_0 = whole_region_initial    
        BO_2stgrad=if_alBO 
        min_bo_0=min_index_initial
        min_index_0=[min_index]
         
    out = single_stage_ot.single_stage_BO_traversal( selected_function,
                                       whole_region_0, 
                                       strategy, d,
                                       n_0, 
                                       min_index_0, 
                                       min_bo_0,# !!!!!!!!! 
                                       X_0, 
                                       n_min, 
                                       getatable_1stgrad, 
                                       BO_2stgrad,
                                       if_optim, 
                                       if_alBO,
                                       Y_0,
                                       file_i,
                                       max_distance )

    whole_region_1=copy.deepcopy(out[0]) 
    X_1 =copy.deepcopy(out[1]) 
    Y_1 =copy.deepcopy(out[2]) 
    min_index_1 =copy.deepcopy(out[3]) 
    BO_2stgrad=copy.deepcopy(out[6]) 
    min_bo_1=copy.deepcopy(out[7]) 


    '''
    construction sumo
    '''
    X_seg_False = X_1
    X_seg_False_ot = ot.Sample(X_seg_False)

    Y_seg_False_ot = Y_1
    Y_seg_False = np.array(Y_seg_False_ot)

    algo_seg_False = ot.KrigingAlgorithm(X_seg_False_ot, 
                                   Y_seg_False_ot, 
                                   covarianceModel, 
                                   basis)

    algo_seg_False.setOptimizationBounds(ot.Interval([in_min]*d, [in_max]*d))
    algo_seg_False.run()
    result_seg_False = algo_seg_False.getResult()

    krig_seg_False = result_seg_False.getMetaModel()


    y_test_seg_False_ot = krig_seg_False (X_test_ot)

    Y_hat_seg_False = np.array(y_test_seg_False_ot)

    MSE_K_seg_False = np.linalg.norm(Y_test-Y_hat_seg_False, ord=2)**2/len(Y_test)


    file='result_2stage.csv'
    if not os.path.exists(file):
        with open(file,'a') as f:
            print('file_i,n_min,MSE_K_compare1_opi,MSE_K_compare2_opi,MSE_K_seg_True,MSE_K_seg_False,MSE_K_BO',file=f)
    with open(file,'a') as f:
            print(str(file_i)+","+str(n_min)+","+str(MSE_K_compare1_opi)+","+str(MSE_K_compare2_opi)+","+str(MSE_K_seg_True)+","+str(MSE_K_seg_False)+","+str(MSE_K_BO),file=f)

    experiment = ot.LHSExperiment(whole_region_initial_ot, 10000, False, True)
    x_extremum = experiment.generate()

    out_set_compare1_opi = krig_compare1_opi(x_extremum)
    out_set_compare2_opi = krig_compare2_opi(x_extremum)
    out_set_seg_True = krig_seg_True(x_extremum)
    out_set_seg_False = krig_seg_False(x_extremum)
    out_set_BO = krig_BO(x_extremum)
    
    maxY_compare1_opi = np.array(out_set_compare1_opi.getMax())[0]
    minY_compare1_opi = np.array(out_set_compare1_opi.getMin())[0]

    maxY_compare2_opi = np.array(out_set_compare2_opi.getMax())[0]
    minY_compare2_opi = np.array(out_set_compare2_opi.getMin())[0]

    maxY_seg_True = np.array(out_set_seg_True.getMax())[0]
    minY_seg_True = np.array(out_set_seg_True.getMin())[0]

    maxY_seg_False = np.array(out_set_seg_False.getMax())[0]
    minY_seg_False = np.array(out_set_seg_False.getMin())[0]

    maxY_BO = np.array(out_set_BO.getMax())[0]
    minY_BO = np.array(out_set_BO.getMin())[0]


    file='max_2stage.csv'
    if not os.path.exists(file):
        with open(file,'a') as f:
            print('file_i,n_min,maxY_compare1_opi,maxY_compare2_opi,maxY_seg_True,maxY_seg_False,maxY_BO',file=f)
    with open(file,'a') as f:
            print(str(file_i)+","+str(n_min)+","+str(maxY_compare1_opi)+","+str(maxY_compare2_opi)+","+str(maxY_seg_True)+","+str(maxY_seg_False)+","+str(maxY_BO),file=f)

 
    file='min_2stage.csv'
    if not os.path.exists(file):
        with open(file,'a') as f:
            print('file_i,n_min,minY_compare1_opi,minY_compare2_opi,minY_seg_True,minY_seg_False,minY_BO',file=f)
    with open(file,'a') as f:
            print(str(file_i)+","+str(n_min)+","+str(minY_compare1_opi)+","+str(minY_compare2_opi)+","+str(minY_seg_True)+","+str(minY_seg_False)+","+str(minY_BO),file=f)

    def bounds_x2():
        return [whole_region_initial[1][0], whole_region_initial[1][1]]

    def bounds_x1(*args):
        return [whole_region_initial[0][0], whole_region_initial[0][1]]


    def get_variance_compare1_opi(x1,x2, meta=result_compare1_opi):
        xPoint = ot.Point([x1,x2])
        conditionalVariance = meta.getConditionalMarginalVariance(xPoint)
        return conditionalVariance

    def get_variance_compare2_opi(x1,x2, meta=result_compare2_opi):
        xPoint = ot.Point([x1,x2])
        conditionalVariance = meta.getConditionalMarginalVariance(xPoint)
        return conditionalVariance

    def get_variance_seg_True(x1,x2, meta=result_seg_True ):
        xPoint = ot.Point([x1,x2])
        conditionalVariance = meta.getConditionalMarginalVariance(xPoint)
        return conditionalVariance

    def get_variance_seg_False(x1,x2, meta=result_seg_False):
        xPoint = ot.Point([x1,x2])
        conditionalVariance = meta.getConditionalMarginalVariance(xPoint)
        return conditionalVariance
    
    #MPV_compare1_opi = integrate.nquad(get_variance_compare1_opi, [bounds_x1, bounds_x2])[0]

    #MPV_compare2_opi = integrate.nquad(get_variance_compare2_opi, [bounds_x1, bounds_x2])[0]

    #MPV_seg_True = integrate.nquad(get_variance_seg_True, [bounds_x1, bounds_x2])[0]

    #MPV_seg_False = integrate.nquad(get_variance_seg_False, [bounds_x1, bounds_x2])[0]


 
    
    