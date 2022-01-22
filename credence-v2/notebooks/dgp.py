#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 01:19:09 2021

@author: harshparikh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special as special
import sklearn.datasets as dg

def dgp1(n=1000,p=10,r=0):
    X = np.random.normal(0,1,size=(n,p))
    Y0 = np.mean(X,axis=1) + np.random.normal(0,1,size=(n))
    Y1 = Y0**2 + np.random.normal(np.mean(X,axis=1),5)
    TE = Y1-Y0
    pi =  special.expit( r*(np.mean(X,axis=1)) )
    T = np.random.binomial(1,pi)
    df = pd.DataFrame(X,columns=['X%d'%(i) for i in range(X.shape[1])])
    df['Y'] = T*Y1 + (1-T)*Y0
    df['T'] = T
    df_true = df.copy(deep=True)
    df_true['Y0'] = Y0
    df_true['Y1'] = Y1
    df_true['TE'] = TE
    df_true['pi'] = pi
    return df, df_true


def dgp2(n=1000,p=10,r=0,e=1):
    def u(x):
        T = []
        for row in x:
            l = special.expit(row[0]+row[1]-0.5+np.random.normal(0,1))
            t = int( l > 0.5 )
            T.append(t)
        return np.array(T)
    def TE(X):
        return X[:,2]*np.cos(np.pi * X[:, 0] * X[:, 1])
    ##GENERATE DATA
    X,Y0 = dg.make_friedman1(n_samples=n, n_features=p, noise=e, random_state=0)
    Y1 = Y0 + TE(X)
    T = u(X)
    Y = T*Y1 + (1-T)*Y0
    columns = ['X%d'%(i) for i in range(p)] + ['Y','T']
    df = pd.DataFrame(np.hstack((X,Y.reshape(-1,1),T.reshape(-1,1))),columns=columns)
    df_true = pd.DataFrame(np.hstack((Y0.reshape(-1,1),Y1.reshape(-1,1),TE(X).reshape(-1,1),T.reshape(-1,1))),columns=['Y0','Y1','TE','T'])
    return df, df_true

def project_star():
    STAR_High_School = pd.read_spss('PROJECTSTAR/STAR_High_Schools.sav')
    STAR_K3_School = pd.read_spss('PROJECTSTAR/STAR_K-3_Schools.sav').set_index('schid')
    STAR_Students = pd.read_spss('PROJECTSTAR/STAR_Students.sav').set_index('stdntid')
    Comparison_Students = pd.read_spss('PROJECTSTAR/Comparison_Students.sav').set_index('stdntid')
    
    # pre-treatment covariates
    gk_cols = list(filter(lambda x: 'gk' in x, STAR_Students.columns))
    g1_cols = list(filter(lambda x: 'g1' in x, STAR_Students.columns))
    g2_cols = list(filter(lambda x: 'g2' in x, STAR_Students.columns))
    g3_cols = list(filter(lambda x: 'g3' in x, STAR_Students.columns))
    g_cols = gk_cols+g1_cols+g2_cols+g3_cols
    
    personal_cols = ['gender','race','birthmonth','birthday','birthyear']
    
    cols_cond = ['surban',
                'tgen',
                'trace',
                'thighdegree',
                'tcareer',
                'tyears',
                'classsize',
                'freelunch']
    
    class_sizes = ['g1classsize',
                 'g2classsize']
    
    g3scores = ['g3treadss',
                'g3tmathss',
                'g3tlangss',
                'g3socialsciss']
    
    g_cols_cond = list(filter(lambda s: np.sum(list(map(lambda x: x in s,cols_cond)))>0,g_cols))
    df_exp = STAR_Students[personal_cols]#+class_sizes]
    df_exp['g3avgscore'] = STAR_Students[g3scores].mean(axis=1)
    df_exp['g3smallclass'] = (STAR_Students['g3classsize']<=17).astype(int)
    
    df_obs = Comparison_Students[personal_cols]#+class_sizes]
    df_obs['g3avgscore'] = Comparison_Students[g3scores].mean(axis=1)
    df_obs['g3smallclass'] = (Comparison_Students['g3classsize']<=17).astype(int)
    
    df_exp = df_exp.dropna()
    df_obs = df_obs.dropna()
    
    df_exp_dummified = pd.get_dummies(df_exp,columns=['gender','race'],drop_first=True)
    df_obs_dummified = pd.get_dummies(df_obs,columns=['gender','race'],drop_first=True)
    df_exp_dummified.columns = df_exp_dummified.columns.str.replace(" ", "_")
    df_obs_dummified.columns = df_obs_dummified.columns.str.replace(" ", "_")
    
    df_exp_dummified['birthmonth'].replace({'JANUARY':1.0,
                                   'FEBRUARY':2.0,
                                   'MARCH':3.0,
                                   'APRIL':4.0,
                                    'ARPIL':4.0,
                                   'MAY':5.0,
                                   'JUNE':6.0,
                                   'JULY':7.0,
                                   'AUGUST':8.0,
                                   'SEPTEMBER':9.0,
                                   'OCTOBER':10.0,
                                   'NOVEMBER':11.0,
                                    'DECEMBER':12.0},inplace=True)
    
    df_obs_dummified['birthmonth'].replace({'JANUARY':1.0,
                                   'FEBRUARY':2.0,
                                   'MARCH':3.0,
                                   'APRIL':4.0,
                                    'ARPIL':4.0,
                                   'MAY':5.0,
                                   'JUNE':6.0,
                                   'JULY':7.0,
                                   'AUGUST':8.0,
                                   'SEPTEMBER':9.0,
                                   'OCTOBER':10.0,
                                   'NOVEMBER':11.0,
                                    'DECEMBER':12.0},inplace=True)
    
    return df_exp_dummified.astype(float), df_obs_dummified.astype(float)

def lalonde():
    nsw = pd.read_stata('http://www.nber.org/~rdehejia/data/nsw.dta')
    psid_control = pd.read_stata('http://www.nber.org/~rdehejia/data/psid_controls.dta')
    df_exp = nsw.drop(columns=['data_id'])
    df_obs = nsw.drop(columns=['data_id']).append(psid_control.drop(columns=['data_id','re74']),ignore_index=True)
    return df_exp, df_obs