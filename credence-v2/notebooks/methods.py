#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 20:58:33 2021

@author: harshparikh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from econml.dml import NonParamDML, LinearDML
from econml.dr import DRLearner
from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner

import zepid
from zepid.causal.doublyrobust import TMLE

import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r, pandas2ri

pandas2ri.activate()

def matchit(outcome, treatment, data, method='nearest', distance='glm', replace=False):
    if replace:
        replace = 'TRUE'
    else:
        replace = 'FALSE'
    data.to_csv('data.csv',index=False)
    formula_cov = treatment+' ~ '
    i = 0
    for cov in data.columns:
        if cov!=outcome and cov!=treatment:
            if i!=0:
                formula_cov += '+' 
            formula_cov += str(cov)
            i += 1
    string = """
    library('MatchIt')
    data <- read.csv('data.csv')
    r <- matchit( %s, method = "%s", data = data, replace = %s)
    matrix <- r$match.matrix[,]
    names <- as.numeric(names(r$match.matrix[,]))
    mtch <- data[as.numeric(names(r$match.matrix[,])),]
    hh <- data[as.numeric(names(r$match.matrix[,])),'%s']- data[as.numeric(r$match.matrix[,]),'%s']
    
    data2 <- data
    data2$%s <- 1 - data2$%s
    r2 <- matchit( %s, method = "%s", distance="%s", data = data2, replace = %s)
    matrix2 <- r2$match.matrix[,]
    names2 <- as.numeric(names(r2$match.matrix[,]))
    mtch2 <- data2[as.numeric(names(r2$match.matrix[,])),]
    hh2 <- data2[as.numeric(r2$match.matrix[,]),'%s'] - data2[as.numeric(names(r2$match.matrix[,])),'%s']
    """%( formula_cov,method,distance,replace,outcome,outcome, treatment, treatment, formula_cov,method,replace,outcome,outcome)
    
    psnn = SignatureTranslatedAnonymousPackage(string, "powerpack")
    match = psnn.mtch
    match2 = psnn.mtch2
    t_hat = pd.DataFrame(np.hstack((np.array(psnn.hh),np.array(psnn.hh2))),
                         index=list(psnn.names.astype(int))+list(psnn.names2.astype(int)),
                         columns=['CATE'])
    ate = np.mean(t_hat['CATE'])
    return ate

def bart(outcome,treatment,data):
    utils = importr('utils')
    dbarts = importr('dbarts')
    cate_est = pd.DataFrame()
    df_train = data
    df_est = data
    
    covariates = set(data.columns) - set([outcome,treatment])
    
    Xc = np.array(df_train.loc[df_train[treatment]==0,covariates])
    Yc = np.array(df_train.loc[df_train[treatment]==0,outcome])
    
    Xt = np.array(df_train.loc[df_train[treatment]==1,covariates])
    Yt = np.array(df_train.loc[df_train[treatment]==1,outcome])
    #
    Xtest = df_train[covariates]
    bart_res_c = dbarts.bart(Xc,Yc,Xtest,keeptrees=True,verbose=False)
    y_c_hat_bart = np.array(bart_res_c[7])
    bart_res_t = dbarts.bart(Xt,Yt,Xtest,keeptrees=True,verbose=False)
    y_t_hat_bart = np.array(bart_res_t[7])
    t_hat_bart = np.array(y_t_hat_bart - y_c_hat_bart)
    cate_est_i = pd.DataFrame(t_hat_bart, index=df_train.index, columns=['CATE'])
    cate_est = pd.concat([cate_est, cate_est_i], join='outer', axis=1)
    cate_est['avg.CATE'] = cate_est.mean(axis=1)
    cate_est['std.CATE'] = cate_est.std(axis=1)
    return np.mean(cate_est)

def metalearner(outcome,treatment,data,est='T',method='linear'):
    if method=='linear':
        models = RidgeCV()
        propensity_model = LogisticRegressionCV()
    if method=='GBR':
        models = GradientBoostingRegressor()
        propensity_model = GradientBoostingClassifier() 
    if est=='T':
        T_learner = TLearner(models=models)
        T_learner.fit(data[outcome], data[treatment], X=data.drop(columns=[outcome,treatment]))
        point = T_learner.ate(X=data.drop(columns=[outcome,treatment]))
    elif est=='S':
        S_learner = SLearner(models=models)
        S_learner.fit(data[outcome], data[treatment], X=data.drop(columns=[outcome,treatment]))
        point = S_learner.ate(X=data.drop(columns=[outcome,treatment]))
    elif est=='X':
        X_learner = XLearner(models=models,propensity_model=propensity_model)
        X_learner.fit(data[outcome], data[treatment], X=data.drop(columns=[outcome,treatment]))
        point = X_learner.ate(X=data.drop(columns=[outcome,treatment]))
    return point

def dml(outcome,treatment,data,method='GBR'):
    if method=='GBR':
        est = NonParamDML(model_y=GradientBoostingRegressor(),model_t=GradientBoostingClassifier(),model_final=GradientBoostingRegressor(),discrete_treatment=True)
        est.fit(data[outcome], data[treatment], 
                X=data.drop(columns=[outcome, treatment]), 
                W=data.drop(columns=[outcome, treatment]))
        point = est.ate(data.drop(columns=[outcome, treatment]), T0=0, T1=1)
    if method=='linear':
        est = LinearDML(discrete_treatment=True)
        est.fit(data[outcome], data[treatment], 
                X=data.drop(columns=[outcome, treatment]), 
                W=data.drop(columns=[outcome, treatment]))
        point = est.ate(data.drop(columns=[outcome, treatment]), T0=0, T1=1)
    return point


def doubleRobust(outcome,treatment,data):
    est = DRLearner()
    est.fit(data[outcome], data[treatment], X=data.drop(columns=[outcome, treatment]), W=data.drop(columns=[outcome, treatment]))
    point = est.ate(data.drop(columns=[outcome, treatment]), T0=0, T1=1)
    return point

def tmle(outcome,treatment,data):
    tml = TMLE(data, exposure=treatment, outcome=outcome)
    cols = data.drop(columns=[outcome, treatment]).columns
    s = str(cols[0])
    for j in range(1,len(cols)):
        s = s + ' + ' + str(cols[j])
    tml.exposure_model('')
    tml.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    tml.fit()
    return tml.average_treatment_effect

def estimate_ate(outcome, treatment, data):
    ate = {}
    ate['gbr_dml'] = dml(outcome, treatment, data, method='GBR', n_estimators=100)
    ate['linear_dml'] = dml(outcome, treatment, data, method='linear', n_estimators=100)
    ate['dr'] = doubleRobust(outcome, treatment, data)
    ate['linear_t_learner'] = metalearner(outcome,treatment, data, est='T', method='linear')
    ate['linear_s_learner'] = metalearner(outcome,treatment, data, est='S', method='linear')
    ate['linear_x_learner'] = metalearner(outcome,treatment, data, est='X', method='linear')
    ate['bart'] = bart(outcome,treatment, data)
    ate['psm'] = matchit(outcome,treatment, data, method='nearest', distance='bart')
    ate['knn'] = matchit(outcome,treatment, data, method='nearest', distance='mahalanobis')
    ate['tmle'] = tmle(outcome, treatment, data)
    return pd.DataFrame.from_dict( ate, orient='index' )
    
    
    