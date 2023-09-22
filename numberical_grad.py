# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:46:08 2021

@author: wbl
"""


import numpy as np
import copy
def grad_ungetatable_1st_ot(f, x):

    #h = 5e-3                    
    grad = np.zeros_like(x)      
    for idx in range(x.getSize()):   
        h=0.01*abs(x[idx])
        tmp_val = x[idx]
        x[idx] = tmp_val + h*tmp_val        
        fxh1 = f(x)                    #(x+h)

        x[idx] = tmp_val - h*tmp_val           #f(x-h)
        fxh2 = f(x)

        grad[idx] =np.array((fxh1 - fxh2) / (2*h*tmp_val), copy=True)     
        x[idx] = tmp_val
        grad=grad.reshape(grad.shape[0],1)

        out_grad=grad.T
    return out_grad

def grad_2nd_ot(f, x):

    #h = 5e-3  
               
    grad = np.zeros_like(x)    
    for idx in range(x.getSize()):
        h=0.01*abs(x[idx])
     
        tmp_val = x[idx]
        x[idx] = tmp_val + h *tmp_val           
        fxh1 = f(x)[idx]                     

        x[idx] = tmp_val - h *tmp_val        
        fxh2 = f(x)[idx]

        grad[idx] =np.array((fxh1 - fxh2) / (2*h*tmp_val), copy=True)    
        x[idx] = tmp_val
        grad=grad.reshape(grad.shape[0],1)

        out_grad=grad.T
    return out_grad



def grad_2nd_ot_1dim(f, x):

    #h = 5e-3
                   
    grad = np.zeros_like(x)     
    for idx in range(x.getSize()):  
        h=0.01*abs(x[idx])
        tmp_val = x[idx]
        x[idx] = tmp_val + h *tmp_val           
        fxh1 = f(x)[idx]                 

        x[idx] = tmp_val - h *tmp_val         
        fxh2 = f(x)[idx]

        grad[idx] =np.array((fxh1 - fxh2) / (2*h*tmp_val), copy=True)   
        x[idx] = tmp_val
        grad=grad.reshape(grad.shape[0],1)

        out_grad=grad.T
    return out_grad




def grad(f, x):

    #h = 5e-3                  
    grad = np.zeros_like(x)      
    for idx in range(x.getSize()):    
        h=0.01*abs(x[idx])
        tmp_val = x[idx]
        x[idx] = tmp_val + h *tmp_val           
        fxh1 = f(x)                     

        x[idx] = tmp_val - h *tmp_val          )
        fxh2 = f(x)

        grad[idx] =np.array((fxh1 - fxh2) / (2*h*tmp_val), copy=True)    
        x[idx] = tmp_val
        grad=grad.reshape(grad.shape[0],1)

        out_grad=grad.T
    return out_grad


def uq_grad(x, uq, model):

    #h = 5e-3                     
    grad = np.zeros_like(x)     
    for idx in range(x.size):    
        h=0.01*abs(x[idx])
        tmp_val = x[idx]
        x[idx] = tmp_val + h *tmp_val          
        fxh1 = uq.evalModel(model, x)   

        x[idx] = tmp_val - h *tmp_val        
        fxh2 = uq.evalModel(model, x)

        grad[idx] = (fxh1 - fxh2) / (2*h*tmp_val)    
        x[idx] = tmp_val
        grad=grad.reshape(grad.shape[0],1)

        out_grad=grad.T
    return out_grad