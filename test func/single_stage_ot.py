import numpy as np
import numberical_grad
from if_inPoly import if_inPoly
import evaluate_region 
import divide
import heapq
from bayesianoptimization import bayesianoptimization
import openturns as ot
import copy
import math



def single_stage_BO_traversal(
                 selected_function,
                 whole_region, 
                 strategy, 
                 d, 
                 n_0,
                 min_index_preorder,
                 min_bo_preorder,
                 X_preorder_ot,
                 n_min,
                 getatable_1stgrad,
                 BO_2stgrad,
                 if_optim,
                 if_alBO,
                 Y_preorder_ot,
                 file_i,
                 max_distance):
   basis =  ot.ConstantBasisFactory(d).build() #ot.QuadraticBasisFactory(d).build()#
   covarianceModel =  ot.MaternModel([1.]*d, 1.5)
   in_min=0.2
   in_max=100

   X_preorder=np.array(X_preorder_ot)
   
   whole_region_ot=divide.ot_region(whole_region, d)
       
   n_1=n_min 
 
   # settings
   number_of_samples = 5000 
   number_of_iteration = int(n_1)#
   acquisition_function_flag = 2  #
   
   if BO_2stgrad:
       
       experiment_candidate_ot = ot.LHSExperiment(whole_region_ot, number_of_samples, False, True)
       X_candidate_ot = experiment_candidate_ot.generate()
       
       X_candidate = np.array(X_candidate_ot)
       
       grad_1st = np.zeros(shape=(X_preorder_ot.getSize(),d))

       if getatable_1stgrad: 
           for i in range(0,X_preorder_ot.getSize()):
              grad_1st[i] = numberical_grad.grad(selected_function, X_preorder_ot[i])
       else:
            
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
       algo_gradient.setOptimizationBounds(ot.Interval([in_min]*d*2, [in_max]*d*2))

       algo_gradient.run()
       result_gradient = algo_gradient.getResult()
       myKriging_grad = result_gradient.getMetaModel()

       grad_2st = np.zeros(shape=(X_preorder_ot.getSize(),d))
       for i in range(0,X_preorder.shape[0]): 
            grad_2st[i]=numberical_grad.grad_2nd_ot(myKriging_grad, X_preorder_ot[i])
       
        
       X_train = X_preorder
       
       
       abs_grad_2st = np.delete(abs(grad_2st),min_bo_preorder,axis=1)
       sum_grad_2st = np.sum(abs_grad_2st, axis=1)
          
       y_train = sum_grad_2st
     
       cumulative_variance = np.zeros(len(X_candidate))
       
       
       Y_true_response = Y_preorder_ot
       
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
           
           
           if getatable_1stgrad:
               grad_1st_add = numberical_grad.grad(selected_function, 
                                                   ot.Point(selected_X_candidate))
               Y_true_response_normal = np.append(np.array(Y_true_response), 
                                   [np.array(selected_function(ot.Point(X_append[0])))],
                                   0)
               Y_true_response = ot.Sample(Y_true_response_normal)
           else:
               
               X_train_ot= ot.Sample(X_train)
               
               
               Y_true_response_normal = np.append(np.array(Y_true_response), 
                                   [np.array(selected_function(ot.Point(X_append[0])))],
                                   0)
               Y_true_response = ot.Sample(Y_true_response_normal)

               
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
           algo_BO.setOptimizationBounds(ot.Interval([in_min]*d*2, [in_max]*d*2))
           algo_BO.run()
           result_BO = algo_BO.getResult()
           myKriging_grad_BO = result_BO.getMetaModel()


           grad_2st_BO = numberical_grad.grad_2nd_ot(myKriging_grad_BO, ot.Point(selected_X_candidate))       
           
           
           abs_grad_2st_BO = np.delete(abs(grad_2st_BO),min_bo_preorder,axis=1)
           sum_grad_2st_BO = np.sum(abs_grad_2st_BO, axis=1) 
           y_add = sum_grad_2st_BO 
                      
           y_train = np.append(y_train, y_add)
           X_candidate = np.delete(X_candidate, selected_candidate_number, 0)
           cumulative_variance = np.delete(cumulative_variance, selected_candidate_number)

   X_next = X_train
   Y_next = np.array(Y_true_response)
   grad_all = grad_1st
   # 
   
   algo_gradient_virtual = ot.KrigingAlgorithm(ot.Sample(X_next), 
                                      ot.Sample(grad_all), 
                                      covarianceModel, 
                                      basis)

   algo_gradient_virtual.setOptimizationBounds(ot.Interval([in_min]*d*2, [in_max]*d*2))
   algo_gradient_virtual.run()
   result_gradient_virtual = algo_gradient_virtual.getResult()
   myKriging_grad_virtual = result_gradient_virtual.getMetaModel()

   experiment_virtual_ot = ot.LHSExperiment(whole_region_ot, 300, False, True)
   X_virtual_ot = experiment_virtual_ot.generate()
   X_virtual = np.array(X_virtual_ot)

   grad_virtual_ot = myKriging_grad_virtual (X_virtual_ot)
   grad_virtual = np.array(grad_virtual_ot)

   # 这个时候为了减少已经分析过得信息的干扰，需要删掉前序已经处理过的变量min_index_preorder
   #absY_gradient_virtual = np.delete(abs(grad_virtual),min_index_preorder,axis=1)
   absY_gradient_virtual = abs(grad_virtual)

   cv_whole = evaluate_region.variation_coefficient(absY_gradient_virtual)

   disper_virtual=cv_whole
   print(disper_virtual)

   for i in range(0,d):
     if not np.argsort(disper_virtual)[i] in min_index_preorder:
       min_index_1= np.argsort(disper_virtual)[i]
       break

   sum_grad_1=sum(absY_gradient_virtual[:,min_index_1])
   weight_1=absY_gradient_virtual[:,min_index_1]/sum_grad_1
   centroid=sum(X_virtual[:,min_index_1]*weight_1)



   # 
   subgrad_score_dim= np.zeros(shape=(9,1))
   coff=0.1

   for i_range in range(1,10):
     
     coff_bandwidth=0.1*i_range  
     next_whole_region = divide.shrink_range(coff_bandwidth, 
                                              whole_region, 
                                              min_index_1, 
                                              centroid)
     

     next_whole_region_ot=divide.ot_region(next_whole_region[0], d)
     
     n_per_v=50
     experiment_shrink_ot = ot.LHSExperiment(next_whole_region_ot, n_per_v, False, True)
     X_shrink_ot = experiment_shrink_ot.generate()
     X_shrink = np.array(X_shrink_ot)

     if whole_region[min_index_1][0] != next_whole_region[0][min_index_1][0]\
     and whole_region[min_index_1][1] != next_whole_region[0][min_index_1][1]:
        
         region_right = copy.deepcopy(whole_region)
         region_right[min_index_1][0] = next_whole_region[0][min_index_1][1] 
         
         region_left = copy.deepcopy(whole_region)
         region_left[min_index_1][1] = next_whole_region[0][min_index_1][0]

         length_right=abs( whole_region[min_index_1][1] - next_whole_region[0][min_index_1][1])
         length_left=abs( whole_region[min_index_1][0] - next_whole_region[0][min_index_1][0])
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

     elif whole_region[min_index_1][0] == next_whole_region[0][min_index_1][0]:
          region_right = copy.deepcopy(whole_region)
          region_right[min_index_1][0] = next_whole_region[0][min_index_1][1] 
          
          region_right_ot=divide.ot_region(region_right, d)
          experiment_apart_shrink_ot = ot.LHSExperiment(region_right_ot, 
                                                  n_per_v, False, True)
          X_apart_shrink_ot = experiment_apart_shrink_ot.generate()
          X_apart=np.array(X_apart_shrink_ot)

     elif whole_region[min_index_1][1] == next_whole_region[0][min_index_1][1]:
          region_left = copy.deepcopy(whole_region)
          region_left[min_index_1][1] = next_whole_region[0][min_index_1][0]
          
          region_left_ot=divide.ot_region(region_left, d)
          
          experiment_apart_shrink_ot = ot.LHSExperiment(region_left_ot, 
                                                  n_per_v, False, True)
          X_apart_shrink_ot = experiment_apart_shrink_ot.generate()      
          X_apart=np.array(X_apart_shrink_ot)
         
         
     judge_X_apart= np.zeros(shape=(n_per_v,1 ))
     judge_X_shrink= np.ones(shape=(n_per_v,1 ))
     
     X_vir2=np.row_stack((X_apart, X_shrink))
     judge_region_0=np.row_stack((judge_X_apart, judge_X_shrink))
     grad_virtual2_ot = myKriging_grad_virtual (X_vir2)
     grad_virtual2 = np.array(grad_virtual2_ot)

     
     absY_gradient2= np.delete(abs(grad_virtual2),min_index_preorder,axis=1)

     sumY_gradient2 = np.sum(absY_gradient2, axis=1)

     analyzed_grad_0 = np.column_stack((judge_region_0,sumY_gradient2))


     n_repeat=1
     subgrad_score_dim_temp =np.zeros(shape=(n_repeat,1))
     
     try:
        for i in range(0,n_repeat):
           subgrad_score_dim_temp[i,0]=evaluate_region.BF(analyzed_grad_0, 1,file_i)[0]
     except:
        for i in range(0,n_repeat):
           subgrad_score_dim_temp[i,0]=0

     subgrad_score_dim_temp1 = np.mean(subgrad_score_dim_temp)

     subgrad_score_dim[i_range-1,0]=subgrad_score_dim_temp1

   
   subgrad_array=  np.asarray(subgrad_score_dim)  
   max_index_array = np.argsort(-subgrad_array.T)[0][0:3]
   max_index_1= max_index_array.tolist()
   
   possible_range_set= (np.asarray(max_index_1)+1)* coff
   possible_range_set = sorted(possible_range_set,reverse=True)

   print(possible_range_set)

   sorted_index = sorted(np.asarray(max_index_1),reverse=True)
   BF_evidence=subgrad_score_dim[sorted_index[strategy]]

   new_min_index0 = np.hstack((min_index_preorder, min_index_1))
   new_bo_index0 = np.hstack((min_bo_preorder, min_index_1))

   if BF_evidence>1.8:
     range_selected=possible_range_set[strategy]#whole_region#
     #next_whole_region = whole_region
     next_whole_region = divide.shrink_range(range_selected, 
                                        whole_region, 
                                        min_index_1, 
                                        centroid)[0] 
     BO_next_2stgrad=True
     new_min_index=new_min_index0
     new_BO_index=new_bo_index0
   else:
     print("no enough evidence")
     next_whole_region = whole_region
     BO_next_2stgrad=if_alBO 
     new_min_index=new_min_index0
     new_BO_index=min_bo_preorder
   
   return [next_whole_region, X_next, Y_next, new_min_index, subgrad_score_dim, n_1, BO_next_2stgrad, new_BO_index]

