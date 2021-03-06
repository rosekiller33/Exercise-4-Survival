import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm  
#from sklearn import linear_model
#from sklearn import metrics

k =[23,41,68]   

#Change file location to read SAS file
#df=pd.read_sas(r"/Users/slathia/Desktop/BAN 5763 SEM 4/Survival Miniung/loans_expanded.sas7bdat")
#
#df.to_csv(r"/Users/slathia/Desktop/BAN 5763 SEM 4/Survival Miniung/ex4.csv")

url = 'https://github.com/slathia18/Sas-DataSet/blob/master/ex4.csv?raw=true'
df = pd.read_csv(url,sep=',',index_col=0,engine='python')
#

df['cubic_spline_b1'] = (df['event_time']>k[0])*(df['event_time']-k[0])**3-df['event_time']**3+3*k[0]*df['event_time']**2-3*k[0]**2*df['event_time']
df['cubic_spline_b2'] = (df['event_time']>k[1])*(df['event_time']-k[1])**3-df['event_time']**3+3*k[1]*df['event_time']**2-3*k[1]**2*df['event_time']
df['cubic_spline_b3'] = (df['event_time']>k[2])*(df['event_time']-k[2])**3-df['event_time']**3+3*k[2]*df['event_time']**2-3*k[2]**2*df['event_time']

x=df[['DBT_RATIO', 'cred_score', 'event_time',
           'cubic_spline_b1','cubic_spline_b2','cubic_spline_b3' ]]

x.head()

i =df[['ID']]

y=df[['event_type']]

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
plot = pd.concat([d1,d2,d3],ignore_index=True)

plot['cubic_spline_b1'] = (plot['event_time']>k[0])*(plot['event_time']-k[0])**3-plot['event_time']**3+3*k[0]*plot['event_time']**2-3*k[0]**2*plot['event_time']
plot['cubic_spline_b2'] = (plot['event_time']>k[1])*(plot['event_time']-k[1])**3-plot['event_time']**3+3*k[1]*plot['event_time']**2-3*k[1]**2*plot['event_time']
plot['cubic_spline_b3'] = (plot['event_time']>k[2])*(plot['event_time']-k[2])**3-plot['event_time']**3+3*k[2]*plot['event_time']**2-3*k[2]**2*plot['event_time']
plot['const'] = 1
plot['DBT_RATIO'] = 0.277

plot = plot[['const','DBT_RATIO','cred_score','event_time',
 'cubic_spline_b1',
 'cubic_spline_b2',
 'cubic_spline_b3']]

pred_p = fmlogit.predict(plot)

plot_pd = pd.concat([plot,pred_p],axis=1)

fig, ax = plt.subplots(figsize=(8,6))
bp = plot_pd.groupby('cred_score').plot(x = 'event_time',y= 1, ax=ax, title = "Hazard of Default by Time and Credit Risk")
plt.xlabel('Time in months')
plt.ylabel('Hazard')
plt.gca().legend(('Credit Score = 690','Credit Score = 733', 'Credit Score = 774'))
plt.show()

fig, ax = plt.subplots(figsize=(8,6))
bp = plot_pd.groupby('cred_score').plot(x = 'event_time',y= 2, ax=ax, title = "Hazard of Default by Time and Credit Risk")
plt.xlabel('Time in months')
plt.ylabel('Hazard')
plt.gca().legend(('Credit Score = 690','Credit Score = 733', 'Credit Score = 774'))
plt.show()


data1 = pd.Series(range(0,181))
d1 = pd.DataFrame(data=data1, columns=['event_time'])
d1['DBT_RATIO'] = 0.19
d2 = pd.DataFrame(data=data1, columns=['event_time'])
d2['DBT_RATIO'] = 0.28
d3 = pd.DataFrame(data=data1, columns=['event_time'])
d3['DBT_RATIO'] = 0.36
plot1 = pd.concat([d1,d2,d3],ignore_index=True)

plot1['cubic_spline_b1'] = (plot1['event_time']>k[0])*(plot1['event_time']-k[0])**3-plot1['event_time']**3+3*k[0]*plot1['event_time']**2-3*k[0]**2*plot1['event_time']
plot1['cubic_spline_b2'] = (plot1['event_time']>k[1])*(plot1['event_time']-k[1])**3-plot1['event_time']**3+3*k[1]*plot1['event_time']**2-3*k[1]**2*plot1['event_time']
plot1['cubic_spline_b3'] = (plot1['event_time']>k[2])*(plot1['event_time']-k[2])**3-plot1['event_time']**3+3*k[2]*plot1['event_time']**2-3*k[2]**2*plot1['event_time']
plot1['const'] = 1
plot1['cred_score'] = 733

plot1 = plot1[['const','DBT_RATIO','cred_score','event_time',
 'cubic_spline_b1',
 'cubic_spline_b2',
 'cubic_spline_b3']]

pred_p1 = fmlogit.predict(plot1)

plot_pd1 = pd.concat([plot1,pred_p1],axis=1)

fig, ax = plt.subplots(figsize=(8,6))
bp = plot_pd1.groupby('DBT_RATIO').plot(x = 'event_time',y= 1, ax=ax, title = "Hazard of Default by Time and Credit Risk")
plt.xlabel('Time in months')
plt.ylabel('Hazard')
plt.gca().legend(('DBT_RATIO =  0.19','DBT_RATIO =  0.28', 'DBT_RATIO =  0.36'))
plt.show()

fig, ax = plt.subplots(figsize=(8,6))
bp = plot_pd1.groupby('DBT_RATIO').plot(x = 'event_time',y= 2, ax=ax, title = "Hazard of Default by Time and Credit Risk")
plt.xlabel('Time in months')
plt.ylabel('Hazard')
plt.gca().legend(('DBT_RATIO =  0.19','DBT_RATIO =  0.28', 'DBT_RATIO =  0.36'))
plt.show()



