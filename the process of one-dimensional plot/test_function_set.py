# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:09:21 2021
This is the set of test function
@author: bingl
"""

import numpy as np
import math  
import numpy.matlib

def santetal03dc(x): 
    #https://www.sfu.ca/~ssurjano/santetal03dc.html
# SANTNER ET AL. (2003) DAMPED COSINE FUNCTION
    x = np.array(x, copy=False)   
    
    fact1 = np.exp(- 1.4 * x)
    fact2 = np.cos(3.5 * math.pi * x)
    y = fact1 * fact2

    return y



