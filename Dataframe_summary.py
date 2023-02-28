import pandas as pd
import numpy as np

def nb_rows(df):     
    ''' returns number of rows'''
    return df.shape[0]

def nb_columns(df):
    ''' returns number of columns'''
    return df.shape[1]

def missing_values(df):
    ''' returns number of missing values'''
    return df.isna().sum().sum()

def missing_values_percent(df):
    ''' returns percentage of missing values'''
    return df.isnull().mean().mean()

def count_duplicated_columns(df):
    ''' returns number of duplicated columns'''
    dupli = df.columns.duplicated()
    dupli_column = 0
    for i in range (len(dupli)):
        if dupli[i] == True:
            dupli_column += 1
    return dupli_column, dupli_column/len(df.columns)

def count_duplicated_rows(df):
    ''' returns number of duplicated rows'''
    return df.duplicated().sum()
#len(df)-len(df.drop_duplicates())

def count_duplicated_rows_percent(df):
    '''  returns percentage of duplicated rows'''
    return count_duplicated_rows(df)/nb_rows(df)

def data_overview(df):    
    '''  prints a dfframe summary containing:number of rows, columns, missing cells and duplicated rows'''
   
    print('Number of variables : {}'.format(nb_columns(df)))
    print('Number of rows : {}'.format(nb_rows(df)))
    print('Missing values : {0} ({1:.2%})'.format(missing_values(df), missing_values_percent(df)))
    print('Duplicated columns : {0} ({1:.2%})'.format(count_duplicated_columns(df)[0], count_duplicated_columns(df)[1]))
    print('Duplicated rows : {0} ({1:.2%})'.format(count_duplicated_rows(df), count_duplicated_rows_percent(df)))
    return None
