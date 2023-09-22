# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:09:21 2021
This is the set of test function
@author: bingl
"""

import numpy as np
import math  
import numpy.matlib
import sys    
import os
import time
import sdf

def trid(xx): 
    
    # TRID FUNCTION  xi ∈ [-d2, d2], for all i = 1, …, d.
    xx = np.array(xx, copy=False)     
    d = 2 ##len(xx)

    if len(xx.shape)!=1:
        sum1 = (xx[:,0] - 1) ** 2
        sum2 = 0
        for ii in range(1,d):
            xi = xx[:,ii]
            xold = xx[:,ii-1]
            sum1 = sum1 + (xi - 1) ** 2
            sum2 = sum2 + xi * xold

    elif len(xx.shape)==1: 
        sum1 = (xx[0] - 1) ** 2
        sum2 = 0
        for ii in range(1,d):
            xi = xx[ii]
            xold = xx[ii-1]
            sum1 = sum1 + (xi - 1) ** 2
            sum2 = sum2 + xi * xold
    
    y = sum1 - sum2
    
    if len(xx.shape)!=1:
       y = np.expand_dims(y, axis=1)

    elif len(xx.shape)==1:  
       y = [y]          

    return y




def hart3(xx): 
    
    
    # HARTMANN 3-DIMENSIONAL FUNCTION xi ∈ (0, 1), for all i = 1, 2, 3.
    xx = np.array(xx, copy=False)     
    alpha = np.transpose(np.array([1.0,1.2,3.0,3.2]))
    A = np.array([[3.0,10,30],
                  [0.1,10,35],
                  [3.0,10,30],
                  [0.1,10,35]])
    P = 10 ** (- 4) * np.array([[3689,1170,2673],
                                [4699,4387,7470],
                                [1091,8732,5547],
                                [381,5743,8828]])
    outer = 0
    
    if len(xx.shape)!=1:
        for ii in range(0,4):
            inner = 0
            for jj in range(0,3):
                xj = xx[:,jj] 
                Aij = A[ii,jj]
                Pij = P[ii,jj]

                inner = inner + Aij * (xj - Pij) ** 2
            new = alpha[ii] * np.exp(- inner)
            outer = outer + new
    elif len(xx.shape)==1:
        for ii in range(0,4):
            inner = 0
            for jj in range(0,3):
                xj = xx[jj] 
                Aij = A[ii,jj]
                Pij = P[ii,jj]

                inner = inner + Aij * (xj - Pij) ** 2
            new = alpha[ii] * np.exp(- inner)
            outer = outer + new
      
    
    y = - outer

    if len(xx.shape)!=1:
       y = np.expand_dims(y, axis=1)

    elif len(xx.shape)==1:  
       y = [y]  



    return y
        

    

    
def hart6(xx): 
    
    # HARTMANN 6-DIMENSIONAL FUNCTION xi ∈ (0, 1), for all i = 1, …, 6.
    xx = np.array(xx, copy=False)         
    alpha = np.array([1.0,1.2,3.0,3.2])
    A = np.array([[10,3,17,3.5,1.7,8],
                  [0.05,10,17,0.1,8,14],
                  [3,3.5,1.7,10,17,8],
                  [17,8,0.05,10,0.1,14]])
    P = 10 ** (- 4) * np.array([[1312,1696,5569,124,8283,5886],
                                [2329,4135,8307,3736,1004,9991],
                                [2348,1451,3522,2883,3047,6650],
                                [4047,8828,8732,5743,1091,381]])
    outer = 0
    
    if len(xx.shape)!=1:
        for ii in range(0,4):
            inner = 0
            for jj in range(0,6):
                xj = xx[:,jj]
                Aij = A[ii,jj]
                Pij = P[ii,jj]
                inner = inner + Aij * (xj - Pij) ** 2
            new = alpha[ii] * np.exp(- inner)
            outer = outer + new
     
    elif len(xx.shape)==1:  
        for ii in range(0,4):
            inner = 0
            for jj in range(0,6):
                xj = xx[jj]
                Aij = A[ii,jj]
                Pij = P[ii,jj]
                inner = inner + Aij * (xj - Pij) ** 2
            new = alpha[ii] * np.exp(- inner)
            outer = outer + new
     
    
    y = - (2.58 + outer) / 1.94
    if len(xx.shape)!=1:
       y = np.expand_dims(y, axis=1)

    elif len(xx.shape)==1:  
       y = [y]  
    
    return y



def run(a, X_0):
    
    with open('/PARA2/paratera_blsca_056/wbl/new_test_fun/output1','a') as m:
        print("start call sr"+str(a)+".sh", file=m)
        time.sleep(1)
        os.system("sed -i 's/a0=50/a0="+str(X_0[a,0])+"/g' /PARA2/paratera_blsca_056/wbl/epoch-4.17.15/epoch1d/Data"+str(a)+"/input.deck")
        os.system("sed -i 's/n1=150.0/n1="+str(X_0[a,1])+"/g' /PARA2/paratera_blsca_056/wbl/epoch-4.17.15/epoch1d/Data"+str(a)+"/input.deck")
        os.system("sed -i 's/l=0.17/l="+str(X_0[a,2])+"/g' /PARA2/paratera_blsca_056/wbl/epoch-4.17.15/epoch1d/Data"+str(a)+"/input.deck")
        os.system('yhbatch -J sr_RPA_'+str(a)+' /PARA2/paratera_blsca_056/wbl/epoch-4.17.15/epoch1d/run'+str(a)+'.sh')
        print("=======================Epoch"+str(a)+"end", file=m)
    
    
def sim(xx):
       
    xx = np.array(xx, copy=False)   
    if len(xx.shape)!=1:
       x1 = xx[:,0]
       x2 = xx[:,1]
       x3 = xx[:,2]
       n_sample=xx.shape[0]
       
       y1 = np.zeros(shape=(n_sample,1))   
       for i in range(0,n_sample):                                     
           data=sdf.read('/WORK/paratera_39/wbl/epoch-4.17.15/epoch1d/Data'+str(i)+'/Proton0005.sdf')
           En=data.dist_fn_energy_Proton.data
           x=data.dist_fn_energy_Proton.grid.data
           MeV=1.69e-13
           x0=np.array(x,dtype='float64')
           x0=x0.T/MeV
           energy=x0[np.argmax(En),0]
           y1 [i]=energy
           y = y1

    elif len(xx.shape)==1:  
       x1 = xx[0]
       x2 = xx[1]
       x3 = xx[2]
       n_sample=1
       os.system('rm -rf /WORK/paratera_39/wbl/epoch-4.17.15/epoch1d/Data/*')
       os.system('cp /WORK/paratera_39/wbl/epoch-4.17.15/epoch1d/input.deck /WORK/paratera_39/wbl/epoch-4.17.15/epoch1d/Data')
       os.system("sed -i 's/a0=50/a0="+str(x1)+"/g' /WORK/paratera_39/wbl/epoch-4.17.15/epoch1d/Data/input.deck")
       os.system("sed -i 's/n1=150.0/n1="+str(x2)+"/g' /WORK/paratera_39/wbl/epoch-4.17.15/epoch1d/Data/input.deck")
       os.system("sed -i 's/l=0.17/l="+str(x3)+"/g' /WORK/paratera_39/wbl/epoch-4.17.15/epoch1d/Data/input.deck")    
       os.system('yhbatch -J sr_RPA /WORK/paratera_39/wbl/epoch-4.17.15/epoch1d/run.sh')
       time.sleep(30)  
       while not os.path.exists(r'/WORK/paratera_39/wbl/epoch-4.17.15/epoch1d/Data/Proton0005.sdf'):
            time.sleep(15) 
       data=sdf.read('/WORK/paratera_39/wbl/epoch-4.17.15/epoch1d/Data/Proton0005.sdf')
       En=data.dist_fn_energy_Proton.data
       x=data.dist_fn_energy_Proton.grid.data
       MeV=1.69e-13
       x0=np.array(x,dtype='float64')
       x0=x0.T/MeV
       energy=x0[np.argmax(En),0]
       y = [float(energy)]
    with open('/WORK/paratera_39/wbl/SUMO_RPA_2/output','a') as n:
        print(y, file=n)  
       
    return y



def curretal88exp(xx): 
    # xi ∈ [0, 1], for all i = 1, 2.
    #https://www.sfu.ca/~ssurjano/curretal88exp.html
    xx = np.array(xx, copy=False)   
    if len(xx.shape)!=1:
       x1 = xx[:,0]
       x2 = xx[:,1]

    elif len(xx.shape)==1:  
       x1 = xx[0]
       x2 = xx[1]

    fact1 = 1 - np.exp(- 1 / (2 * x2))
    fact2 = 2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60
    fact3 = 100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20
    y = fact1 * fact2 / fact3
    
    if len(xx.shape)!=1:
      y = np.expand_dims(y, axis=1)
    elif len(xx.shape)==1:  
      y = [y]      
    
    return y


def curretal88exp_ot_multi(X): 
    xx = np.array(X, copy=False)    
    x1 = xx[:,0]
    x2 = xx[:,1]

    fact1 = 1 - np.exp(- 1 / (2 * x2))
    fact2 = 2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60
    fact3 = 100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20
    y = fact1 * fact2 / fact3
    return np.expand_dims(y, axis=1)

def curretal88exp_ot_single(X): 
    
    xx = np.array(X, copy=False)   
    x1 = xx[0]
    x2 = xx[1]

    fact1 = 1 - np.exp(- 1 / (2 * x2))
    fact2 = 2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60
    fact3 = 100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20
    y = fact1 * fact2 / fact3
    return [y]



def camel6(xx): 
    
    # SIX-HUMP CAMEL FUNCTION https://www.sfu.ca/~ssurjano/camel6.html
    # 符合设计情况的为x1 ∈ [-2, 2], x2 ∈ [-1, 1]. , 推荐的为 x1 ∈ [-3, 3], x2 ∈ [-2, 2].

    if len(xx.shape)!=1:
      x1 = xx[:,0]
      x2 = xx[:,1]
    elif len(xx.shape)==1:  
      x1 = xx[0]
      x2 = xx[1]
    
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (- 4 + 4 * x2 ** 2) * x2 ** 2
    y = term1 + term2 + term3
    return y


def camel6_ot_multi(X): 
    xx = np.array(X, copy=False)    
   
    x1 = xx[:,0]
    x2 = xx[:,1]
    
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (- 4 + 4 * x2 ** 2) * x2 ** 2
    y = term1 + term2 + term3
    return np.expand_dims(y, axis=1)


def camel6_ot_single(X): 
    
    xx = np.array(X, copy=False)   
   
    x1 = xx[0]
    x2 = xx[1]
    
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (- 4 + 4 * x2 ** 2) * x2 ** 2
    y = term1 + term2 + term3
    return [y]




def camel3(xx): 
    # xi ∈ [-2, 2] https://www.sfu.ca/~ssurjano/camel3.html
   
    # THREE-HUMP CAMEL FUNCTION
    
    if len(xx.shape)!=1:
      x1 = xx[:,0]
      x2 = xx[:,1]
    elif len(xx.shape)==1:  
      x1 = xx[0]
      x2 = xx[1]
    
    term1 = 2 * x1 ** 2
    term2 = - 1.05 * x1 ** 4
    term3 = x1 ** 6 / 6
    term4 = x1 * x2
    term5 = x2 ** 2
    y = term1 + term2 + term3 + term4 + term5
    return y

def camel3_ot_multi(X): 
    xx = np.array(X, copy=False)    
    # xi ∈ [-2, 2] https://www.sfu.ca/~ssurjano/camel3.html
   
    # THREE-HUMP CAMEL FUNCTION
    x1 = xx[:,0]
    x2 = xx[:,1]
    
    term1 = 2 * x1 ** 2
    term2 = - 1.05 * x1 ** 4
    term3 = x1 ** 6 / 6
    term4 = x1 * x2
    term5 = x2 ** 2
    y = term1 + term2 + term3 + term4 + term5
    return np.expand_dims(y, axis=1)

def camel3_ot_single(X): 
    # xi ∈ [-2, 2] https://www.sfu.ca/~ssurjano/camel3.html
   
    # THREE-HUMP CAMEL FUNCTION
    
    xx = np.array(X, copy=False)   
    x1 = xx[0]
    x2 = xx[1]
    
    term1 = 2 * x1 ** 2
    term2 = - 1.05 * x1 ** 4
    term3 = x1 ** 6 / 6
    term4 = x1 * x2
    term5 = x2 ** 2
    y = term1 + term2 + term3 + term4 + term5
    return [y]

def limetal02pol(xx): 
    # xi ∈ [0, 1], for all i = 1, 2.
    
    # LIM ET AL. (2002) POLYNOMIAL FUNCTION

    if len(xx.shape)!=1:
       x1 = xx[:,0]
       x2 = xx[:,1]

    elif len(xx.shape)==1:  
       x1 = xx[0]
       x2 = xx[1]

    term1 = (5 / 2) * x1 - (35 / 2) * x2
    term2 = (5 / 2) * x1 * x2 + 19 * x2 ** 2
    term3 = - (15 / 2) * x1 ** 3 - (5 / 2) * x1 * x2 ** 2
    term4 = - (11 / 2) * x2 ** 4 + (x1 ** 3) * (x2 ** 2)
    y = 9 + term1 + term2 + term3 + term4
    return y
    

def limetal02pol_ot_multi(X): 
    xx = np.array(X, copy=False)    
   
    x1 = xx[:,0]
    x2 = xx[:,1]

    term1 = (5 / 2) * x1 - (35 / 2) * x2
    term2 = (5 / 2) * x1 * x2 + 19 * x2 ** 2
    term3 = - (15 / 2) * x1 ** 3 - (5 / 2) * x1 * x2 ** 2
    term4 = - (11 / 2) * x2 ** 4 + (x1 ** 3) * (x2 ** 2)
    y = 9 + term1 + term2 + term3 + term4
    return np.expand_dims(y, axis=1)

def limetal02pol_ot_single(X): 
    
    xx = np.array(X, copy=False)   

    x1 = xx[0]
    x2 = xx[1]

    term1 = (5 / 2) * x1 - (35 / 2) * x2
    term2 = (5 / 2) * x1 * x2 + 19 * x2 ** 2
    term3 = - (15 / 2) * x1 ** 3 - (5 / 2) * x1 * x2 ** 2
    term4 = - (11 / 2) * x2 ** 4 + (x1 ** 3) * (x2 ** 2)
    y = 9 + term1 + term2 + term3 + term4
    return [y]


def zakharov(xx): 
    xx = np.array(xx, copy=False)   
    # ZAKHAROV FUNCTION xi ∈ [-5, 10], for all i = 1, …, d.
    # https://www.sfu.ca/~ssurjano/zakharov.html
    if len(xx.shape)!=1:
      d=np.size(xx,1)
      #d=3
      sum1 = 0
      sum2 = 0
      for ii in range(0,d):
        xi = xx[:,ii]
        sum1 = sum1 + xi ** 2
        sum2 = sum2 + 0.5 * (ii+1) * xi
    
      y = sum1 + sum2 ** 2 + sum2 ** 4
      
    elif len(xx.shape)==1:  
      d=np.size(xx,0)
      sum1 = 0
      sum2 = 0
      for ii in range(0,d):
        xi = xx[ii]
        sum1 = sum1 + xi ** 2
        sum2 = sum2 + 0.5 * (ii+1) * xi
    
      y = sum1 + sum2 ** 2 + sum2 ** 4
    
    if len(xx.shape)!=1:
       y = np.expand_dims(y, axis=1)

    elif len(xx.shape)==1:  
       y = [y]        
    
    return y


def zakharov_ot_multi(X): 
    xx = np.array(X, copy=False) 
    
    d=np.size(xx,1)
    #d=3
    sum1 = 0
    sum2 = 0
    for ii in range(0,d):
      xi = xx[:,ii]
      sum1 = sum1 + xi ** 2
      sum2 = sum2 + 0.5 * (ii+1) * xi
    
      y = sum1 + sum2 ** 2 + sum2 ** 4
      
      
    return np.expand_dims(y, axis=1)

def zakharov_ot_single(X): 
    
    xx = np.array(X, copy=False)   
    
    # ZAKHAROV FUNCTION

    d=np.size(xx,0)
    sum1 = 0
    sum2 = 0
    for ii in range(0,d):
        xi = xx[ii]
        sum1 = sum1 + xi ** 2
        sum2 = sum2 + 0.5 * (ii+1) * xi
    
    y = sum1 + sum2 ** 2 + sum2 ** 4
      
    return [y]






def branin(xx): 
   
    # BRANIN FUNCTION https://www.sfu.ca/~ssurjano/branin.html
    # x1 ∈ [-5, 10], x2 ∈ [0, 15]
    xx = np.array(xx, copy=False)   
    if len(xx.shape)!=1:
       x1 = xx[:,0]
       x2 = xx[:,1]

    elif len(xx.shape)==1:  
       x1 = xx[0]
       x2 = xx[1]    
       
    t = 1 / (8 * math.pi)
    s = 10
    r = 6
    c = 5 / math.pi
    b = 5.1 / (4 * math.pi ** 2)
    a = 1
    
    term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)
    y = term1 + term2 + s
    
    if len(xx.shape)!=1:
      y = np.expand_dims(y, axis=1)
    elif len(xx.shape)==1:  
      y = [y]      
    
    return y

    
    
  
def michal(xx): 
    # https://www.sfu.ca/~ssurjano/michal.html
    # MICHALEWICZ FUNCTION xi ∈ [0, π], for all i = 1, …, d.
  
    m = 10
    
    xx = np.array(xx, copy=False)   
   
    sum1 = 0

    if len(xx.shape)!=1:
        d = xx.shape[1]
        for ii in range(0,d):
            xi = xx[:, ii]
            new = np.sin(xi) * (np.sin((ii+1) * xi ** 2 / math.pi)) ** (2 * m)
            sum1 = sum1 + new
           
    elif len(xx.shape)==1: 
        d =  len(xx)
        for ii in range(0,d):
            xi = xx[ii]
            new = np.sin(xi) * (np.sin((ii+1) * xi ** 2 / math.pi)) ** (2 * m)
            sum1 = sum1 + new

    y = - sum1
    
    if len(xx.shape)!=1:
       y = np.expand_dims(y, axis=1)

    elif len(xx.shape)==1:  
       y = [y]    
       
    return y    
 
    



