# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:07:15 2020

@author: morir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from sklearn.metrics import mean_squared_error as mse

def is_categorical(data, key):
    
    col_type=data[key].dtype
    
    if col_type=='int':
        
        nunique=data[key].nunique()
        return nunique<6
    
    elif col_type=="float":
        return False
    
    else:
        return True
    
def visualize_data(data, target_col, categorical_keys=None, hue=None):
     
    keys=data.keys()
        
    if categorical_keys is None:
        
        categorical_keys=keys[[is_categorical(data, key) for key in keys]]
   
    for key in keys:
        
        if key==target_col or key==hue:
            continue
            
        length=10
        subplot_size=(length, length/2)
        
        if (key in categorical_keys) and (target_col in categorical_keys):

            r=cramerV(key, target_col, data)
            
            fig, axes=plt.subplots(1, 2, figsize=subplot_size)
            sns.countplot(x=key, data=data, ax=axes[0])
            sns.catplot(x=key, data=data, hue=target_col, row=hue, kind='count')
            plt.title(r)
            plt.tight_layout()
            plt.show()

        elif (key in categorical_keys) and not (target_col in categorical_keys):

            r=correlation_ratio(cat_key=key, num_key=target_col, data=data)
            
            fig, axes=plt.subplots(1, 2, figsize=subplot_size)
            sns.countplot(x=key, data=data, ax=axes[0])
            sns.violinplot(x=key, y=target_col, data=data, ax=axes[1], hue=hue)
            plt.title(r)
            plt.tight_layout()
            plt.show()

        elif not (key in categorical_keys) and (target_col in categorical_keys):

            r=correlation_ratio(cat_key=target_col, num_key=key, data=data)
            
            fig, axes=plt.subplots(1, 2, figsize=subplot_size)            
            sns.distplot(data[key], ax=axes[0], kde=False)
            g=sns.FacetGrid(data, hue=target_col, col=hue)
            g.map(sns.distplot, key, ax=axes[1], kde=False)
            axes[1].set_title(r)
            axes[1].legend()            
            plt.tight_layout()
            plt.close()
            plt.show()

        else:

            r=data.corr().loc[key, target_col]
            
            sg=sns.jointplot(x=key, y=target_col, data=data, height=length*2/3, hue=hue)
            plt.title(r)
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

def correlation_ratio(cat_key, num_key, data):
    
    categorical=data[cat_key]
    numerical=data[num_key]
    
    mean=numerical.dropna().mean()
    all_var=((numerical-mean)**2).sum()
    
    unique_cat=pd.Series(categorical.unique())
    unique_cat=list(unique_cat.dropna())
    
    categorical_num=[numerical[categorical==cat] for cat in unique_cat]
    categorical_var=[len(x.dropna())*(x.dropna().mean()-mean)**2 for x in categorical_num]    

    r=sum(categorical_var)/all_var
    
    return r

def cramerV(x, y, data):
    
    table=pd.crosstab(data[x], data[y])
    x2, p, dof, e=st.chi2_contingency(table, False)
    
    n=table.sum().sum()
    r=np.sqrt(x2/(n*(np.min(table.shape)-1)))

    return r

def get_corr(data, categorical_keys=None):
    
    keys=data.keys()
        
    if categorical_keys is None:
        
        categorical_keys=keys[[is_categorical(data, key) for key in keys]]
        
    corr=pd.DataFrame({})
    corr_ratio=pd.DataFrame({})
    corr_cramer=pd.DataFrame({})
        
    for key1 in keys:
        for key2 in keys:

            if (key1 in categorical_keys) and (key2 in categorical_keys):

                r=cramerV(key1, key2, data)
                corr_cramer.loc[key1, key2]=r                

            elif (key1 in categorical_keys) and (key2 not in categorical_keys):

                r=correlation_ratio(cat_key=key1, num_key=key2, data=data)
                corr_ratio.loc[key1, key2]=r                

            elif (key1 not in categorical_keys) and (key2 in categorical_keys):

                r=correlation_ratio(cat_key=key2, num_key=key1, data=data)
                corr_ratio.loc[key1, key2]=r                

            else:

                r=data.corr().loc[key1, key2]
                corr.loc[key1, key2]=r                
        
    return corr, corr_ratio, corr_cramer

def get_pred_ratio(y_real, y_pred):
    
    ratios=[]
    
    for real, pred in zip(y_real, y_pred):
        
        if real!=0:
            ratio=abs(real-pred)/real
        else:
            ratio=abs(real-pred)
            
        ratios.append(ratio)
        
    return ratios

def eval_regression(y_pred, y_real):

    y_ratio=get_pred_ratio(y_real, y_pred)

    percent=[50, 75, 90]
    ratio_p=[]

    for p in percent:
        ratio_p.append(np.percentile(y_ratio, p))

    print('mse: ', mse(y_pred, y_real))
    for p, r in zip(percent, ratio_p):
        print('error {0}%: {1}'.format(p, r))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y_real, y_pred, 'bo')
    plt.plot(y_real, y_real, 'r')
    plt.xlabel('real', fontsize=12)
    plt.ylabel('predeiction', fontsize=12)
    
    plt.subplot(1, 2, 2)
    plt.plot(percent, ratio_p, 'bo')
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    plt.xlabel('Percent (%)', fontsize=12)
    plt.ylabel('Prediction ratio', fontsize=12)
    
    plt.subplots_adjust(wspace=0.7)
    
def wrong_rate(data, target_key, pred, real):
    
    first_key=data.keys()[0]
    if first_key==target_key:
        first_key=data.keys()[1]
    
    wrong=pred!=real
    data_wrong=data[wrong]
    data_wrong_group=data_wrong.groupby(target_key).count()[first_key].reset_index().rename(columns={first_key: 'Count'}).set_index(target_key)
    
    data_count=data.groupby(target_key)[first_key].count().reset_index().rename(columns={first_key: 'Count'}).set_index(target_key)
    
    data_wrong_group/=data_count
    
    return data_wrong_group