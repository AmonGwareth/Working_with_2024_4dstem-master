
import numpy as np

def Set_Wrong_Pixels_to_NaN(Array, treshhold):
    
    
    sum = np.sum(Array)
    # print(sum)
    if sum >treshhold*0.9 and sum <treshhold*1.1:
        Return_array = Array
        
    else:
        Return_array = Array*np.nan

    return Return_array


def Set_Wrong_Pixels_to_Zero(Array, treshhold):
    
    
    sum = np.sum(Array)
    # print(sum)
    if sum >treshhold*0.9 and sum <treshhold*1.1:
        Return_array = Array
        
    else:
        Return_array = Array*0

    return Return_array

def Set_Wrong_Pixels_to_MeanValue(Array, treshhold, MeanValue):
    
    
    sum = np.sum(Array)
    # print(sum)
    if sum >treshhold*0.9 and sum <treshhold*1.1:
        Return_array = Array
        
    else:
        Return_array = Array*0+1*MeanValue

    return Return_array

def Normalize_Array_to_Sum(Array):

    Array_norm = Array/np.sum(Array)

    return Array_norm

def Normalize_Array_to_Max(Array):

    Array_norm = Array/np.max(Array)

    return Array_norm