import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm  
from sklearn import linear_model
from sklearn import metrics

k =[23,41,68]   

print(k)

 #Change file location to read SAS file
df=pd.read_sas(r"/Users/slathia/Desktop/BAN 5763 SEM 4/Survival Miniung/loans_expanded.sas7bdat")

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


columns = ['DBT_RATIO', 'cred_score', 'event_time']
plot = pd.DataFrame(columns=columns)
plot.head()

cred_score = []
event_time= []
for y in range(0,181):
    for x in (690, 733, 774):
        cred_score.append(x)
        event_time.append(y)

plot.event_time = event_time
plot.cred_score = cred_score

#data = pd.Series(range(0,181))
##
##d1 = pd.DataFrame(data=data, columns=['event_time'])
##
##d1['cred_score'] = 690
##
##d2 = pd.DataFrame(data=data, columns=['event_time'])
##
##d2['cred_score'] = 733
##
##d3 = pd.DataFrame(data=data, columns=['event_time'])
##
##d3['cred_score'] = 774
##
##plot = pd.concat([d1,d2,d3])

plot['cubic_spline_b1'] = (plot['event_time']>k[0])*(plot['event_time']-k[0])**3-plot['event_time']**3+3*k[0]*plot['event_time']**2-3*k[0]**2*plot['event_time']

plot['cubic_spline_b2'] = (plot['event_time']>k[1])*(plot['event_time']-k[1])**3-plot['event_time']**3+3*k[1]*plot['event_time']**2-3*k[1]**2*plot['event_time']

plot['cubic_spline_b3'] = (plot['event_time']>k[2])*(plot['event_time']-k[2])**3-plot['event_time']**3+3*k[2]*plot['event_time']**2-3*k[2]**2*plot['event_time']

plot['const'] = 1

plot['DBT_RATIO'] = 0.277

plot_sorted = plot.sort_values(by = ['cred_score', 'event_time'])
#df_score_sorted['const'] = 1
plot_sorted.reset_index()

#plotc = plot[['const','DBT_RATIO','event_time',
# 'cred_score',
# 'cubic_spline_b1',
# 'cubic_spline_b2',
# 'cubic_spline_b3']]

plot_sorted = plot_sorted[['const','DBT_RATIO','cred_score','event_time',
 'cubic_spline_b1',
 'cubic_spline_b2',
 'cubic_spline_b3']]

#plotc['event_type'] = 0

#ys = pd.DataFrame(plotc['event_type'])

#xcs = plotc[['const','DBT_RATIO','event_time',
# 'cred_score',
# 'cubic_spline_b1',
# 'cubic_spline_b2',
# 'cubic_spline_b3']]

#plotc = plotc.reshape(-1,1)# not working

pred_p = fmlogit.predict(plot_sorted)

plot_pd = pd.concat([plot_sorted,pred_p],axis=1)

fig, ax = plt.subplots(figsize=(8,6))
bp = plot_pd.groupby('cred_score').plot(x = 'event_time',y= 1, ax=ax, title = "Hazard of Default by Time and Credit Risk")
plt.xlabel('Time in months')
plt.ylabel('Hazard')
plt.gca().legend(('Credit Score =  690','Credit Score  = 733', 'Credit Score  = 774'))
plt.show()





