import pandas as pd
import numpy as np

def convert_dtypes(df):
    """
    Convert the data types of all variables in a pandas DataFrame.
    
    Parameters:
        df (pandas DataFrame): The DataFrame to convert.
    
    Returns:
        pandas DataFrame: The converted DataFrame.
    """
    # Get the data types of all variables
    dtypes = df.dtypes
    
    # Convert each variable to the appropriate data type
    for col in df.columns:
        dtype = dtypes[col]
        if dtype == 'bool':
            df[col] = df[col].astype(bool)
        elif dtype == 'int64':
            df[col] = df[col].astype(int)
        elif dtype == 'float64':
            df[col] = df[col].astype(float)
        elif dtype == 'object':
            df[col] = df[col].astype(str)
        elif dtype.name.startswith('datetime64'):
            df[col] = pd.to_datetime(df[col])
        elif pd.api.types.is_categorical_dtype(dtype):
            df[col] = df[col].astype('category')
            
    return df
