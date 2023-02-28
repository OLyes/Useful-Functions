#!/usr/bin/env python
# coding: utf-8

# # Projet 3:  Fonctions utilisées

# In[9]:


# Installation des librairies et des fonctions usuelles de python                                 

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import missingno as msno
from datetime import datetime
import plotly.graph_objs as go
import plotly
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy import stats


# In[16]:


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


# In[11]:


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


# In[12]:


def null_var(df, threshold):
    '''Variables to drop from DataFrame according to their null values rate'''
    
    null_var = (df.isnull().mean().sort_values(ascending=False) * 100).reset_index()
    null_var.columns = ['variable','prc_nan']
    null_var_rate = null_var[null_var['prc_nan']>=threshold]
    
    print(f'{null_var_rate.variable.count()} variables ({round(100 * (null_var_rate.shape[0] / df.shape[1]), 2)}%) with NaN rate higher or equal to {threshold}%')
    
    return null_var_rate


# In[13]:


def correlation_graph(pca, 
                      x_y, 
                      features) : 
    """ PCA : Correlation graph """

    # Extrait x et y 
    x,y=x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante : 
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0,0, 
                pca.components_[0, i],  
                pca.components_[1, i],  
                head_width=0.07,
                head_length=0.07, 
                width=0.02, )

        # Les labels
        plt.text(pca.components_[0, i] + 0.05,
                pca.components_[1, i] + 0.05,
                features[i])
        
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))


    plt.title("Correlation graph (F{} et F{})".format(x+1, y+1))

    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)


# In[14]:


def display_factorial_planes(   X_projected, 
                                x_y, 
                                pca=None, 
                                labels = None,
                                clusters=None, 
                                alpha=1,
                                figsize=[10,8], 
                                marker="." ):
    """ PCA : Factorial planes """

    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (7,6)

    # On gère les labels
    if  labels is None : 
        labels = []
    try : 
        len(labels)
    except Exception as e : 
        raise e

    # On vérifie la variable axis 
    if not len(x_y) ==2 : 
        raise AttributeError("2 axes sont demandées")   
    if max(x_y )>= X_.shape[1] : 
        raise AttributeError("la variable axis n'est pas bonne")   

    # on définit x et y 
    x, y = x_y

    # Initialisation de la figure       
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters
 
    # Les points    
    plt.scatter(   X_[:, x], 
                        X_[:, y], 
                        alpha=alpha, 
                        c=c, 
                        cmap="Set1", 
                        marker=marker)


    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe 
    if pca : 
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else : 
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    # On borne x et y 
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels) : 
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center') 

    # Titre et display
    plt.title(f"Projection on factorial planes ( F{x+1} & F{y+1})")
    plt.show()


# In[21]:


def test_chi2(ar1, ar2):
    """ Test Chi2 """
    alpha = 0.05    
    tab_contingence = pd.crosstab(ar1, ar2)
    stat_chi2, p, dof, expected_table = chi2_contingency(tab_contingence.values)
    print('Statistique khi2 [chi2] :'+str(stat_chi2))
#    print('Nombe de degrés de liberté [dof] :'+str(dof))
    print('Pvalue [p] :'+str(p))
    
    critical = chi2.ppf(1-alpha, dof) 

    if p <= alpha:
        print('Variables dépendantes (H0 rejetée) car p = {} <= alpha = {}'.format(p, alpha))
        return False
    
    else:
        print('H0 non rejetée car p = {} >= alpha = {}'.format(p, alpha))
        return True


# In[ ]:




