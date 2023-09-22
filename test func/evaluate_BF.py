# -*- coding: utf-8 -*-

import numpy as np
import subprocess




def evaluate_BF(data):
    
    np.savetxt('/PARA2/paratera_blsca_056/wbl/analyzed_data.csv', data, delimiter = ',')

    subprocess.call(["/PARA2/paratera_blsca_056/.conda/envs/py373/bin/R","--slave","--no-restore","--file=/PARA2/paratera_blsca_056/wbl/SUMO_RPA_2dim/two-group-evaluation.R" ] )
    
    score = np.loadtxt(open("/PARA2/paratera_blsca_056/wbl/BF_ru.csv","rb"),delimiter=",") 
    
    return score
