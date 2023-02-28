
import pandas as pd
import numpy as np

def types_cleaning(df):
    '''Data types correction for all variables'''
    try :
        
        #booleans
        bool_columns = bool_vars
        for column in bool_columns:
            data[column] = data[column].astype('bool')
       
        #integers
        int_columns = []
        for column in int_vars:
            data[column] = data[column].astype('int64')

        #floats
        float_columns = float_vars
        for column in float_columns:
            data[column] = data[column].astype('float64')
 
        #objects
        object_columns = object_vars
        for column in object_columns:
            data[column] = data[column].astype('object')
       
        #categories
        category_columns = category_vars
        for column in category_columns:
            data[column] = data[column].astype('category')
            
        #time
        for column in time_vars_t:
            data[column] = pd.to_datetime(data[column], unit='s', errors='coerce')
        for column in time_vars_datetime:
            data[column] = pd.to_datetime(data[column], errors='coerce')

    except :
        print('Data type conversion issue!')
        
    return df
