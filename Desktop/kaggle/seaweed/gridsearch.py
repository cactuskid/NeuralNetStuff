import numpy as np


def genGrid(params)
    
    values = {}

    #for each parameter define a range and a step [ start, stop , step]
    
    for p in params.keys():
        values[p]=np.linspace(params[p][0],params[p][1],params[p][2])

    return values



