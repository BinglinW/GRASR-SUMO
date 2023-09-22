# -*- coding: utf-8 -*-

import numpy as np
import copy
from itertools import product

def divide_region(whole_range, dim, sub_per_dim):

  n_sub_total = sub_per_dim ** dim
  
  dividepoint=np.mean(whole_range, axis=1)


  dim_th=[]
  
  for i in range(0,dim):
     dim_th.append([0, 1]) 
  
  tdim=[dim_th[i] for i in range(0,dim)]
  
  if dim==2:
     interval_choice=list(product(tdim[0],tdim[1]))
  elif dim==3:
     interval_choice=list(product(tdim[0],tdim[1],tdim[2]))
  elif dim==4:
     interval_choice=list(product(tdim[0],tdim[1],tdim[2],tdim[3]))
  elif dim==5:
     interval_choice=list(product(tdim[0],tdim[1],tdim[2],tdim[3],tdim[4]))
  elif dim==6:
     interval_choice=list(product(tdim[0],tdim[1],tdim[2],tdim[3],tdim[4],tdim[5]))
  elif dim==7:
     interval_choice=list(product(tdim[0],tdim[1],tdim[2],tdim[3],tdim[4],tdim[5],tdim[6]))
  elif dim==8:
     interval_choice=list(product(tdim[0],tdim[1],tdim[2],tdim[3],tdim[4],tdim[5],tdim[6],tdim[7]))
  elif dim==9:
     interval_choice=list(product(tdim[0],tdim[1],tdim[2],tdim[3],tdim[4],tdim[5],tdim[6],tdim[7],tdim[8]))
  elif dim==10:
     interval_choice=list(product(tdim[0],tdim[1],tdim[2],tdim[3],tdim[4],tdim[5],tdim[6],tdim[7],tdim[8],tdim[9]))


  for i in np.arange(0,n_sub_total):
      locals()['subregion_'+str(i)]= copy.deepcopy(whole_range)
  
  for i in np.arange(0,n_sub_total):
    for k in np.arange(0,dim):
      locals()['subregion_'+str(i)][k][interval_choice[i][k]] = dividepoint[k]
  
  d = dict(); 
  for i in np.arange(0,n_sub_total):
     d[i]= locals()['subregion_'+str(i)]
   
  
  return d



def shrink_range(coff_bandwidth, Original_region, shrink_index, centroid):


  half_bandwidth=coff_bandwidth/2 * (Original_region[shrink_index][1]-Original_region[shrink_index][0])

  lower_bound_ideal=centroid-half_bandwidth
  upper_bound_ideal=centroid+half_bandwidth


  if upper_bound_ideal>Original_region[shrink_index][1]:
    overflow= upper_bound_ideal-Original_region[shrink_index][1]
    upper_bound=Original_region[shrink_index][1]
    lower_bound=lower_bound_ideal-overflow
  elif lower_bound_ideal<Original_region[shrink_index][0]:
    overflow=Original_region[shrink_index][0]-lower_bound_ideal
    upper_bound=upper_bound_ideal+overflow
    lower_bound=Original_region[shrink_index][0]
  else:
    upper_bound=upper_bound_ideal
    lower_bound=lower_bound_ideal
    
  shrink_range=[lower_bound, upper_bound]


  shrink_region=[]

  shrink_region.append(copy.deepcopy(Original_region))

  shrink_region[0][shrink_index]=shrink_range
  
  return shrink_region

import openturns as ot


def ot_region(whole_region, d):
   temp_0=whole_region
   
   if d==1:
       whole_region_ot=ot.ComposedDistribution([ot.Uniform(temp_0[0][0], temp_0[0][1])])
   
   if d==2:
       whole_region_ot=ot.ComposedDistribution([ot.Uniform(temp_0[0][0], temp_0[0][1]), 
                            ot.Uniform(temp_0[1][0], temp_0[1][1])])
   elif d==3:
       whole_region_ot=ot.ComposedDistribution([ot.Uniform(temp_0[0][0], temp_0[0][1]), 
                            ot.Uniform(temp_0[1][0], temp_0[1][1]), 
                            ot.Uniform(temp_0[2][0], temp_0[2][1])])
       
   elif d==4:
       whole_region_ot=ot.ComposedDistribution([ot.Uniform(temp_0[0][0], temp_0[0][1]), 
                            ot.Uniform(temp_0[1][0], temp_0[1][1]), 
                            ot.Uniform(temp_0[2][0], temp_0[2][1]), 
                            ot.Uniform(temp_0[3][0], temp_0[3][1])])       
   elif d==5:
       whole_region_ot=ot.ComposedDistribution([ot.Uniform(temp_0[0][0], temp_0[0][1]), 
                            ot.Uniform(temp_0[1][0], temp_0[1][1]), 
                            ot.Uniform(temp_0[2][0], temp_0[2][1]), 
                            ot.Uniform(temp_0[3][0], temp_0[3][1]), 
                            ot.Uniform(temp_0[4][0], temp_0[4][1])])          
   elif d==6:
       whole_region_ot=ot.ComposedDistribution([ot.Uniform(temp_0[0][0], temp_0[0][1]), 
                            ot.Uniform(temp_0[1][0], temp_0[1][1]), 
                            ot.Uniform(temp_0[2][0], temp_0[2][1]), 
                            ot.Uniform(temp_0[3][0], temp_0[3][1]), 
                            ot.Uniform(temp_0[4][0], temp_0[4][1]), 
                            ot.Uniform(temp_0[5][0], temp_0[5][1])])         
   elif d==7:
       whole_region_ot=ot.ComposedDistribution([ot.Uniform(temp_0[0][0], temp_0[0][1]), 
                            ot.Uniform(temp_0[1][0], temp_0[1][1]), 
                            ot.Uniform(temp_0[2][0], temp_0[2][1]), 
                            ot.Uniform(temp_0[3][0], temp_0[3][1]), 
                            ot.Uniform(temp_0[4][0], temp_0[4][1]), 
                            ot.Uniform(temp_0[5][0], temp_0[5][1]), 
                            ot.Uniform(temp_0[6][0], temp_0[6][1])])  
   elif d==8:
       whole_region_ot=ot.ComposedDistribution([ot.Uniform(temp_0[0][0], temp_0[0][1]), 
                            ot.Uniform(temp_0[1][0], temp_0[1][1]), 
                            ot.Uniform(temp_0[2][0], temp_0[2][1]), 
                            ot.Uniform(temp_0[3][0], temp_0[3][1]), 
                            ot.Uniform(temp_0[4][0], temp_0[4][1]), 
                            ot.Uniform(temp_0[5][0], temp_0[5][1]), 
                            ot.Uniform(temp_0[6][0], temp_0[6][1]), 
                            ot.Uniform(temp_0[7][0], temp_0[7][1])])  
   elif d==20:
       whole_region_ot=ot.ComposedDistribution([ot.Uniform(temp_0[0][0], temp_0[0][1]), 
                               ot.Uniform(temp_0[1][0], temp_0[1][1]), 
                               ot.Uniform(temp_0[2][0], temp_0[2][1]), 
                               ot.Uniform(temp_0[3][0], temp_0[3][1]), 
                               ot.Uniform(temp_0[4][0], temp_0[4][1]), 
                               ot.Uniform(temp_0[5][0], temp_0[5][1]), 
                               ot.Uniform(temp_0[6][0], temp_0[6][1]), 
                               ot.Uniform(temp_0[7][0], temp_0[7][1]), 
                               ot.Uniform(temp_0[8][0], temp_0[8][1]), 
                               ot.Uniform(temp_0[9][0], temp_0[9][1]), 
                               ot.Uniform(temp_0[10][0], temp_0[10][1]), 
                               ot.Uniform(temp_0[11][0], temp_0[11][1]), 
                               ot.Uniform(temp_0[12][0], temp_0[12][1]), 
                               ot.Uniform(temp_0[13][0], temp_0[13][1]), 
                               ot.Uniform(temp_0[14][0], temp_0[14][1]), 
                               ot.Uniform(temp_0[15][0], temp_0[15][1]), 
                               ot.Uniform(temp_0[16][0], temp_0[16][1]), 
                               ot.Uniform(temp_0[17][0], temp_0[17][1]), 
                               ot.Uniform(temp_0[18][0], temp_0[18][1]), 
                               ot.Uniform(temp_0[19][0], temp_0[19][1])])  
    
   return whole_region_ot

    
 
 
