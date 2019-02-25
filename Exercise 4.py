#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm  
from sklearn import linear_model
from sklearn import metrics

k =[23,41,68]   

print(k)

df=pd.read_sas(r"C:\Users\rahul.slathia\Desktop\College Stuff\Ex 4 Survival\loans_expanded.sas7bdat",encoding='utf-7')
df.head()

df['cubic_spline_b1'] = (df['event_time']>k[0])*(df['event_time']-k[0])**3-df['event_time']**3+3*k[0]*df['event_time']**2-3*k[0]**2*df['event_time']

df['cubic_spline_b2'] = (df['event_time']>k[1])*(df['event_time']-k[1])**3-df['event_time']**3+3*k[1]*df['event_time']**2-3*k[1]**2*df['event_time']

df['cubic_spline_b3'] = (df['event_time']>k[2])*(df['event_time']-k[2])**3-df['event_time']**3+3*k[2]*df['event_time']**2-3*k[2]**2*df['event_time']

df.head()

list(df)

x=df[['DBT_RATIO', 'cred_score', 'event_time',
           'cubic_spline_b1','cubic_spline_b2','cubic_spline_b3' ]]

x.head()

i =df[['ID']]

y=df[['event_type']]

y['event_type'] =y['event_type'].astype(str).replace('\.0+$', '', regex=True)

xc= sm.add_constant(x)

mlogit =sm.MNLogit(y,xc)

fmlogit = mlogit.fit()

summary = fmlogit.summary()

print(summary)

odds = np.exp(fmlogit.params)

print(odds)

y.head()

y['event_type'].value_counts()

print(fmlogit.aic)

pred = fmlogit.predict(xc)

final = pd.concat([i,x,y,pred],axis=1)


data = pd.Series(range(0,181))

d1 = pd.DataFrame(data=data, columns=['event_time'])

d1['cred_score'] = 690

d2 = pd.DataFrame(data=data, columns=['event_time'])

d2['cred_score'] = 733

d3 = pd.DataFrame(data=data, columns=['event_time'])

d3['cred_score'] = 774

plot = pd.concat([d1,d2,d3])

plot['cubic_spline_b1'] = (plot['event_time']>k[0])*(plot['event_time']-k[0])**3-plot['event_time']**3+3*k[0]*plot['event_time']**2-3*k[0]**2*plot['event_time']

plot['cubic_spline_b2'] = (plot['event_time']>k[1])*(plot['event_time']-k[1])**3-plot['event_time']**3+3*k[1]*plot['event_time']**2-3*k[1]**2*plot['event_time']

plot['cubic_spline_b3'] = (plot['event_time']>k[2])*(plot['event_time']-k[2])**3-plot['event_time']**3+3*k[2]*plot['event_time']**2-3*k[2]**2*plot['event_time']

plot['const'] = 1

plot['DBT_RATIO'] = 0.277

plotc = plot[['const','DBT_RATIO','event_time',
 'cred_score',
 'cubic_spline_b1',
 'cubic_spline_b2',
 'cubic_spline_b3']]

plotc['event_type'] = 0

ys = pd.DataFrame(plotc['event_type'])

xcs = plotc[['const','DBT_RATIO','event_time',
 'cred_score',
 'cubic_spline_b1',
 'cubic_spline_b2',
 'cubic_spline_b3']]

plotc = plotc.reshape(-1,1)

pred_p = mlogit.predict(plotc)


