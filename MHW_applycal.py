#Apply a MHW calibration to an independent SST data set

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

plt.rcParams['figure.dpi'] = 300; plt.rcParams['savefig.dpi'] = 300

filename = input("What is the name of calibration file? (include .pkl extension): ")

with open(filename,"rb") as f:
    pls_model = pickle.load(f)

filename = input("What is the name of you proxy file? (include .csv extension): ")
proxy = pd.read_csv(filename)

#plot proxy data
fig, ax = plt.subplots()
ax.hist(proxy.sst, bins=int(len(proxy)/4), density=True,color="grey")
sns.kdeplot(proxy.sst, ax=ax,color="red")
plt.show()

#calculate distribution statistics
stdz = (proxy.sst-np.mean(proxy.sst))/np.std(proxy.sst)
spw = stats.shapiro(stdz)           #shapiro-wilk
kurt = stats.kurtosis(stdz)         #kurtosis
skew = stats.skew(stdz)             #skewness
percs = np.linspace(0,100,21)
q = np.percentile(stdz,percs)
tmp = pd.DataFrame ({'data': [spw[0],kurt,skew]},index = ['shapiro_stat','kurt','skew'])
tmpqq = pd.DataFrame(np.transpose(q),index = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21'])
tmpqq.columns = ['data']
summary = pd.concat([tmp,tmpqq])

MHWpred_proxy = pls_model.predict(summary.T)

print('reconstructed MHW metric = ', round(MHWpred_proxy.item() ,3)) 

