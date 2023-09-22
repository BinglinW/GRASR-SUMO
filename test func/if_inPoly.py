# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:58:03 2021

@author: wbl
"""

import numpy as np


def if_inPoly(Point, n_sub_total, polygon, dim):
     judge_result = np.zeros(shape=(1,n_sub_total))
     
     
     if dim==2:
      for i_ploy in range(0,n_sub_total): 
       if     polygon[i_ploy][0][0] < Point[0] and Point[0] < polygon[i_ploy][0][1]\
          and polygon[i_ploy][1][0] < Point[1] and Point[1] < polygon[i_ploy][1][1]:
          judge_result[0,i_ploy]=1

     elif dim==3:
      for i_ploy in range(0,n_sub_total): 
       if     polygon[i_ploy][0][0] < Point[0] and Point[0] < polygon[i_ploy][0][1]\
          and polygon[i_ploy][1][0] < Point[1] and Point[1] < polygon[i_ploy][1][1]\
          and polygon[i_ploy][2][0] < Point[2] and Point[2] < polygon[i_ploy][2][1]:
          judge_result[0,i_ploy]=1
     
     elif dim==4:
      for i_ploy in range(0,n_sub_total): 
       if     polygon[i_ploy][0][0] < Point[0] and Point[0] < polygon[i_ploy][0][1]\
          and polygon[i_ploy][1][0] < Point[1] and Point[1] < polygon[i_ploy][1][1]\
          and polygon[i_ploy][2][0] < Point[2] and Point[2] < polygon[i_ploy][2][1]\
          and polygon[i_ploy][3][0] < Point[3] and Point[3] < polygon[i_ploy][3][1]:
          judge_result[0,i_ploy]=1

     elif dim==5:
      for i_ploy in range(0,n_sub_total): 
       if     polygon[i_ploy][0][0] < Point[0] and Point[0] < polygon[i_ploy][0][1]\
          and polygon[i_ploy][1][0] < Point[1] and Point[1] < polygon[i_ploy][1][1]\
          and polygon[i_ploy][2][0] < Point[2] and Point[2] < polygon[i_ploy][2][1]\
          and polygon[i_ploy][3][0] < Point[3] and Point[3] < polygon[i_ploy][3][1]\
          and polygon[i_ploy][4][0] < Point[4] and Point[4] < polygon[i_ploy][4][1]:
          judge_result[0,i_ploy]=1

     elif dim==6:
      for i_ploy in range(0,n_sub_total): 
       if     polygon[i_ploy][0][0] < Point[0] and Point[0] < polygon[i_ploy][0][1]\
          and polygon[i_ploy][1][0] < Point[1] and Point[1] < polygon[i_ploy][1][1]\
          and polygon[i_ploy][2][0] < Point[2] and Point[2] < polygon[i_ploy][2][1]\
          and polygon[i_ploy][3][0] < Point[3] and Point[3] < polygon[i_ploy][3][1]\
          and polygon[i_ploy][4][0] < Point[4] and Point[4] < polygon[i_ploy][4][1]\
          and polygon[i_ploy][5][0] < Point[5] and Point[5] < polygon[i_ploy][5][1]:
          judge_result[0,i_ploy]=1        
     
     elif dim==7:
      for i_ploy in range(0,n_sub_total): 
       if     polygon[i_ploy][0][0] < Point[0] and Point[0] < polygon[i_ploy][0][1]\
          and polygon[i_ploy][1][0] < Point[1] and Point[1] < polygon[i_ploy][1][1]\
          and polygon[i_ploy][2][0] < Point[2] and Point[2] < polygon[i_ploy][2][1]\
          and polygon[i_ploy][3][0] < Point[3] and Point[3] < polygon[i_ploy][3][1]\
          and polygon[i_ploy][4][0] < Point[4] and Point[4] < polygon[i_ploy][4][1]\
          and polygon[i_ploy][5][0] < Point[5] and Point[5] < polygon[i_ploy][5][1]\
          and polygon[i_ploy][6][0] < Point[6] and Point[6] < polygon[i_ploy][6][1]:
          judge_result[0,i_ploy]=1
          
     elif dim==8:
      for i_ploy in range(0,n_sub_total): 
       if     polygon[i_ploy][0][0] < Point[0] and Point[0] < polygon[i_ploy][0][1]\
          and polygon[i_ploy][1][0] < Point[1] and Point[1] < polygon[i_ploy][1][1]\
          and polygon[i_ploy][2][0] < Point[2] and Point[2] < polygon[i_ploy][2][1]\
          and polygon[i_ploy][3][0] < Point[3] and Point[3] < polygon[i_ploy][3][1]\
          and polygon[i_ploy][4][0] < Point[4] and Point[4] < polygon[i_ploy][4][1]\
          and polygon[i_ploy][5][0] < Point[5] and Point[5] < polygon[i_ploy][5][1]\
          and polygon[i_ploy][6][0] < Point[6] and Point[6] < polygon[i_ploy][6][1]\
          and polygon[i_ploy][7][0] < Point[7] and Point[7] < polygon[i_ploy][7][1]:
          judge_result[0,i_ploy]=1
          
     elif dim==9:
      for i_ploy in range(0,n_sub_total): 
       if     polygon[i_ploy][0][0] < Point[0] and Point[0] < polygon[i_ploy][0][1]\
          and polygon[i_ploy][1][0] < Point[1] and Point[1] < polygon[i_ploy][1][1]\
          and polygon[i_ploy][2][0] < Point[2] and Point[2] < polygon[i_ploy][2][1]\
          and polygon[i_ploy][3][0] < Point[3] and Point[3] < polygon[i_ploy][3][1]\
          and polygon[i_ploy][4][0] < Point[4] and Point[4] < polygon[i_ploy][4][1]\
          and polygon[i_ploy][5][0] < Point[5] and Point[5] < polygon[i_ploy][5][1]\
          and polygon[i_ploy][6][0] < Point[6] and Point[6] < polygon[i_ploy][6][1]\
          and polygon[i_ploy][7][0] < Point[7] and Point[7] < polygon[i_ploy][7][1]\
          and polygon[i_ploy][8][0] < Point[8] and Point[8] < polygon[i_ploy][8][1]:
          judge_result[0,i_ploy]=1

     elif dim==10:
      for i_ploy in range(0,n_sub_total): 
       if     polygon[i_ploy][0][0] < Point[0] and Point[0] < polygon[i_ploy][0][1]\
          and polygon[i_ploy][1][0] < Point[1] and Point[1] < polygon[i_ploy][1][1]\
          and polygon[i_ploy][2][0] < Point[2] and Point[2] < polygon[i_ploy][2][1]\
          and polygon[i_ploy][3][0] < Point[3] and Point[3] < polygon[i_ploy][3][1]\
          and polygon[i_ploy][4][0] < Point[4] and Point[4] < polygon[i_ploy][4][1]\
          and polygon[i_ploy][5][0] < Point[5] and Point[5] < polygon[i_ploy][5][1]\
          and polygon[i_ploy][6][0] < Point[6] and Point[6] < polygon[i_ploy][6][1]\
          and polygon[i_ploy][7][0] < Point[7] and Point[7] < polygon[i_ploy][7][1]\
          and polygon[i_ploy][8][0] < Point[8] and Point[8] < polygon[i_ploy][8][1]\
          and polygon[i_ploy][9][0] < Point[9] and Point[9] < polygon[i_ploy][9][1]:
          judge_result[0,i_ploy]=1         
          
     return judge_result
 
         


def if_inPoly_single(Point, polygon1):
     judge_result = np.zeros(shape=(1,1))
     if polygon1[0][0] < Point[0] and Point[0] < polygon1[0][1]\
        and polygon1[1][0] < Point[1] and Point[1] < polygon1[1][1]:
        judge_result[0,0]=1   
        
     return judge_result    
