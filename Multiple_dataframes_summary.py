import csv
import pandas as pd

data_01 = pd.read_csv('data_01.csv', sep=',')
data_02 = pd.read_csv('data_02.csv', sep=',')
data_03 = pd.read_csv('data_03.csv', sep=',')
data_04 = pd.read_csv('data_04.csv', sep=',')
data_05 = pd.read_csv('data_05.csv', sep=',')

list_data = {'data_01': data_01, 
              'data_02': data_02,
              'data_03': data_03,
              'data_04': data_04,
              'data_05': data_05}

def dataframes_info(list_data):
    list_info = []
    for name, df in list_data.items():
        list_info.append([name, df.shape[0], df.shape[1],
                          df.count().sum(),
                          df.isna().mean().mean()*100,
                          df.duplicated().sum()])

    df_info = pd.DataFrame(list_info, columns=['name', 'rows', 'columns', 'count',
                                               'NaNs (%)', 'duplicates'])
    return df_info


print(dataframes_info(list_data))
