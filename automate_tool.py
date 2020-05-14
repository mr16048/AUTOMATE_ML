# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:07:15 2020

@author: morir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def is_categorical(data, key):
    
    col_type=data[key].dtype
    
    if col_type=='int':
        
        nunique=data[key].nunique()
        return nunique<6
    
    elif col_type=="float":
        return False
    
    else:
        return True
    
def visualize_data(data, target_col):
    
    for key in data.keys():
        
        if key==target_col:
            continue
            
        length=10
        subplot_size=(length, length/2)
        
        if is_categorical(data, key) and is_categorical(data, target_col):

            fig, axes=plt.subplots(1, 2, figsize=subplot_size)
            sns.countplot(x=key, data=data, ax=axes[0])
            sns.countplot(x=key, data=data, hue=target_col, ax=axes[1])
            plt.tight_layout()
            plt.show()

        elif is_categorical(data, key) and not is_categorical(data, target_col):

            fig, axes=plt.subplots(1, 2, figsize=subplot_size)
            sns.countplot(x=key, data=data, ax=axes[0])
            sns.violinplot(x=key, y=target_col, data=data, ax=axes[1])
            plt.tight_layout()
            plt.show()

        elif not is_categorical(data, key) and is_categorical(data, target_col):

            fig, axes=plt.subplots(1, 2, figsize=subplot_size)
            sns.distplot(data[key], ax=axes[0], kde=False)
            g=sns.FacetGrid(data, hue=target_col)
            g.map(sns.distplot, key, ax=axes[1], kde=False)
            axes[1].legend()
            plt.tight_layout()
            plt.close()
            plt.show()

        else:

            sns.jointplot(x=key, y=target_col, data=data, height=length*2/3)
            plt.show()            

def summarize_data(df):

    df_summary=pd.DataFrame({'nunique':np.zeros(df.shape[1])}, index=df.keys())

    df_summary['nunique']=df.nunique()
    df_summary['dtype']=df.dtypes
    df_summary['isnull']=df.isnull().sum()
    df_summary['first_val']=df.iloc[0]
    df_summary['first_val']=df.iloc[0]
    df_summary['max']=df.max(numeric_only=True)
    df_summary['min']=df.min(numeric_only=True)
    df_summary['mean']=df.mean(numeric_only=True)
    df_summary['std']=df.std(numeric_only=True)
    df_summary['mode']=df.mode().iloc[0]

    pd.set_option('display.max_rows', len(df.keys()))
    
    return df_summary