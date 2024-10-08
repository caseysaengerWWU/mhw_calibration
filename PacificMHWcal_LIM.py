#An adaptable MHW calibration script

import requests
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tarfile
import netCDF4 as nc
import pandas as pd
import scipy.stats as stats
import random
import seaborn as sns
import pickle
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

plt.rcParams['figure.dpi'] = 300; plt.rcParams['savefig.dpi'] = 300

filename = input("What is the name of your monthly sst file? (include extension): ")
file = pd.read_csv(filename, parse_dates=['date'])
file.set_index('date', inplace = True)  
#the latitude and longitude of the site
lat = file.lat[0]
lon = file.lon[0]
if lon < 0: lon1 = lon + 360

file.drop(['lat','lon'], axis = 1, inplace=True)

#define the duration (m, months) and threshold (s, number of standard deviations) of MHWs
m = int(input("Define a minimum duration for a marine heatwave (integer in months). Values less than 6 suggested: "))
s = float(input("Define a standard deviation threshold for defining marine heat waves. Values between 1 and 2 suggested:  "))

#select the foram species. Options are: NPachyderma, NIncompta, GBulloides, GRuber, TSacculifer
#select maximum depth over which to average (up to 100 m)
species=input("What foram species do you want to model? Choices are NPachyderma, GBulloides, NIncompta, GRruber, or TSacculifer: ")

###############################################################################

#function to get closest grid cell assuming flat earth
def flat_fast(latvar,lonvar,lat0,lon0):
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:]
    lonvals = lonvar[:]
    ny,nx = latvals.shape
    dist_sq = (latvals-lat0)**2 + (lonvals-lon0)**2
    minindex_flattened = dist_sq.argmin()  # 1D index of min element
    iy_min,ix_min = np.unravel_index(minindex_flattened, latvals.shape)
    return iy_min,ix_min

#function to get MHW from timeseries with mean and climatology removed
#duration and time should have the same units
def mhw_alt(time,signal,threshold,duration):
    thresholded = signal > threshold
    edges = np.convolve([1, -1], thresholded, mode='same')
    tmpstart = np.where(edges==1)[0]
    tmpend = np.where(edges==-1)[0]
    if len(tmpstart) > len(tmpend): tmpend = np.append(tmpend,len(time))
    tmpdur = tmpend-tmpstart
    istart = tmpstart[np.where(tmpdur >= duration)]
    iend = tmpend[np.where(tmpdur >= duration)]-1
    tstart = time[istart]
    tend = time[(iend)]
    mean_intensity = np.zeros(len(istart))
    total_intensity = np.zeros(len(istart))
    count = len(istart)
    dur = tmpdur[np.where(tmpdur >= duration)]
    months = np.zeros(0)
    if any(dur) > 0: 
        months = np.zeros([len(istart),max(dur)])
        for x in range(0,len(istart)): 
            mean_intensity[x] = np.mean(signal[istart[x]:iend[x]+1]-threshold)
            total_intensity[x] = np.sum(signal[istart[x]:iend[x]+1]-threshold)
            months[x,0:dur[x]]=(time.month[istart[x]:iend[x]+1])            
    return count, tstart, tend, dur, mean_intensity, total_intensity, months  

# A function for calculating distribution statistics. 
#sst is a timeseries of sea surface temperature
#monthlyf is an array of 12 values representing the relative monthly conc of forams
#n multiplied by the number of years in the record is the number of random draws
#an n of 1000 is used for the true value. Smaller values reflect sensitivity to undersampling

def dist_stat(sst,monthlyf,n):
    import warnings
    warnings.filterwarnings('ignore')
    yrs=int(len(sst)/12)
    prob = list(monthlyf/yrs)*yrs
    #weighted = random.choices(stdz,prob,k=n)
    weighted = random.choices(sst,prob,k=n)
    weighted = (weighted-np.mean(weighted))/np.std(weighted)
    spw = stats.shapiro(weighted)           #shapiro-wilk
    kurt = stats.kurtosis(weighted)         #kurtosis
    skew = stats.skew(weighted)             #skewness
    percs = np.linspace(0,100,21)
    q = np.percentile(weighted,percs)
    tmp = pd.DataFrame ({'data': [spw[0],spw[1],kurt,skew]},index = ['shapiro_stat','shapiro_p','kurt','skew'])
    tmpqq = pd.DataFrame(np.transpose(q),index = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21'])
    tmpqq.columns = ['data']
    summary = pd.concat([tmp,tmpqq])
    return summary

###############################################################################
###############################################################################
#Get ERSSTv5 data
print('fetching ERSST data...')
base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nceiErsstv5.nc?sst%5B(1854-01-15T00:00:00Z):1:(2023-12-15T00:00:00Z)%5D%5B(0.0):1:(0.0)%5D%5B("
request_url = f"{base_url}{lat}):1:({lat})%5D%5B({lon1}):1:({lon1})%5D"
response = requests.get(request_url)
if response.status_code == 200:
    dat = xr.open_dataset(response.content)
    ER_sst = dat['sst'].values
    ER_lat = dat['latitude'].values
    ER_lon = dat['longitude'].values
    ER_time = dat['time'].values
    ER_df = pd.DataFrame(np.squeeze(ER_sst), index = ER_time, columns = ['sst'])
    
#Get HadISST data
print('fetching HadISST data...')
base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdHadISSTDecomp.nc?sst%5B(1870-01-16T11:59:59Z):1:(2022-12-16T12:00:00Z)%5D%5B("
request_url = f"{base_url}{lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"
response = requests.get(request_url)
if response.status_code == 200:
    dat = xr.open_dataset(response.content)
    HD_sst = dat['sst'].values
    HD_lat = dat['latitude'].values
    HD_lon = dat['longitude'].values
    HD_time = dat['time'].values
    HD_df = pd.DataFrame(np.squeeze(HD_sst), index = HD_time, columns = ['sst'])
    HD_df.index = HD_df.index.normalize()

#Get COBEv2 data
print('fetching COBE sst data...')
base_url = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/COBE2/sst.mon.mean.nc?var=sst&north="
request_url = f"{base_url}{lat}&west={lon}&east={lon}&south={lat}&horizStride=1&time_start=1850-01-01T00:00:00Z&time_end=2023-12-01T00:00:00Z&&&accept=netcdf3"
response = requests.get(request_url)
if response.status_code == 200:
    dat = xr.open_dataset(response.content)
    CB_sst = dat['sst'].values
    CB_lat = dat['lat'].values
    CB_lon = dat['lon'].values
    CB_time = dat['time'].values
    CB_df = pd.DataFrame(np.squeeze(CB_sst), index = CB_time, columns = ['sst'])

#Highpass filter and remove climatology
filtyr = 30
sst_lowess = lowess(ER_df.sst, ER_df.index, filtyr/(ER_df.size/12)) 
sst_filt = ER_df.sst.values-sst_lowess[:,1]
ER_df['sst_filt'] = sst_filt
climER=ER_df.groupby(ER_df.index.month).sst_filt.mean()
tmp = ER_df.groupby(ER_df.index.month).transform(lambda sst_filt: sst_filt-sst_filt.mean())
ER_df['sstWOclim'] = tmp.sst_filt

sst_lowess = lowess(HD_df.sst, HD_df.index, filtyr/(HD_df.size/12)) 
sst_filt = HD_df.sst.values-sst_lowess[:,1]
HD_df['sst_filt'] = sst_filt
climHD=HD_df.groupby(HD_df.index.month).sst_filt.mean()
tmp = HD_df.groupby(HD_df.index.month).transform(lambda sst_filt: sst_filt-sst_filt.mean())
HD_df['sstWOclim'] = tmp.sst_filt

sst_lowess = lowess(CB_df.sst, CB_df.index, filtyr/(CB_df.size/12)) 
sst_filt = CB_df.sst.values-sst_lowess[:,1]
CB_df['sst_filt'] = sst_filt    
climCB=CB_df.groupby(CB_df.index.month).sst_filt.mean()
tmp = CB_df.groupby(CB_df.index.month).transform(lambda sst_filt: sst_filt-sst_filt.mean())
CB_df['sstWOclim'] = tmp.sst_filt

#average climatology to be added to LIM since it doesn't have seasonal cycle
clim = np.mean(np.array([climCB, climHD, climER]), axis=0)

###############################################################################  
#Get monthly foram concentration per m3 from Kretschmer et al. 2018 doi.org/10.5194/bg-15-4405-2018
print('fetching seasonal foram concentrations...')
if species == "NPachyderma":
    request_url = "https://store.pangaea.de/Publications/Kretschmer-etal_2018/Cold-WaterPlankForams.tar.gz"
    with requests.get(request_url, stream = 'TRUE') as rx, tarfile.open(fileobj=rx.raw, mode="r|gz") as tarobj : tarobj.extractall()
    dat = nc.Dataset('PLAFOM2.0_GLOBAL_MONTHLY_CONC_cold-waterPlankForamSpecies.nc')
if species in ("GBulloides" , "NIncompta"):
    request_url = "https://store.pangaea.de/Publications/Kretschmer-etal_2018/Temperate-WaterPlankForams.tar.gz"
    with requests.get(request_url, stream = 'TRUE') as rx, tarfile.open(fileobj=rx.raw, mode="r|gz") as tarobj : tarobj.extractall()
    dat = nc.Dataset('PLAFOM2.0_GLOBAL_MONTHLY_CONC_temperate-waterPlankForamSpecies.nc')
if species in ("GRruber" , "TSacculifer"):
    request_url ="https://store.pangaea.de/Publications/Kretschmer-etal_2018/Warm-WaterPlankForams.tar.gz"
    with requests.get(request_url, stream = 'TRUE') as rx, tarfile.open(fileobj=rx.raw, mode="r|gz") as tarobj : tarobj.extractall()
    dat = nc.Dataset('PLAFOM2.0_GLOBAL_MONTHLY_CONC_warm-waterPlankForamSpecies.nc')

iy,ix = flat_fast(dat['latitude'], dat['longitude'], lat, lon1)
conc = np.sum(dat[species][:,0:24,iy,ix].data, axis =1)
total_conc = np.sum(conc)
monthlyf = conc/total_conc

#plot
fig, ax = plt.subplots(1,1,figsize=(7,4))
ax.plot(np.arange(1, 13),monthlyf,'o')
ax.set(ylabel='relative foram conc.',xlabel='month')
plt.show()

###############################################################################
#Calculate marine heatwave metrics in observations
#And calculate distribution statistics in observations as if timeseries were forams without knowledge of climatology
#Data are standardized. This does not influence statistics, except to change quantiles by a proportional amount.

print('calculating mhw metrics and temperature distribution statistics in observations...')
ER_allmhws = mhw_alt(ER_df.index,ER_df.sstWOclim,np.std(ER_df.sstWOclim)*s,m)
ER_mhw_summary = np.zeros(5) 
yrs=int(len(ER_df)/12)
ER_mhw_summary[0] = ER_allmhws[0]/(yrs/10)             #n_tot/10yr
ER_mhw_summary[1] = sum(ER_allmhws[3])/(yrs/10)        #tot_mo/10yr
ER_mhw_summary[2] = sum(ER_allmhws[5])/(yrs/10)        #cum_int/10yr
ER_mhw_summary[3] = np.mean(ER_allmhws[3])             #mean_dur per event
ER_mhw_summary[4] = np.mean(ER_allmhws[4])             #mean_int per event

ER_dstats = dist_stat(ER_df.sst_filt,monthlyf,yrs*1000)

HD_allmhws = mhw_alt(HD_df.index,HD_df.sstWOclim,np.std(HD_df.sstWOclim)*s,m)
HD_mhw_summary = np.zeros(5) 
yrs=int(len(HD_df)/12)
HD_mhw_summary[0] = HD_allmhws[0]/(yrs/10)             #n_tot/10yr
HD_mhw_summary[1] = sum(HD_allmhws[3])/(yrs/10)        #tot_mo/10yr
HD_mhw_summary[2] = sum(HD_allmhws[5])/(yrs/10)        #cum_int/10yr
HD_mhw_summary[3] = np.mean(HD_allmhws[3])             #mean_dur per event
HD_mhw_summary[4] = np.mean(HD_allmhws[4])             #mean_int per event

HD_dstats = dist_stat(HD_df.sst_filt,monthlyf,yrs*1000)

CB_allmhws = mhw_alt(CB_df.index,CB_df.sstWOclim,np.std(CB_df.sstWOclim)*s,m)
CB_mhw_summary = np.zeros(5) 
yrs=int(len(CB_df)/12)
CB_mhw_summary[0] = CB_allmhws[0]/(yrs/10)             #n_tot/10yr
CB_mhw_summary[1] = sum(CB_allmhws[3])/(yrs/10)        #tot_mo/10yr
CB_mhw_summary[2] = sum(CB_allmhws[5])/(yrs/10)        #cum_int/10yr
CB_mhw_summary[3] = np.mean(CB_allmhws[3])             #mean_dur per event
CB_mhw_summary[4] = np.mean(CB_allmhws[4])             #mean_int per event

CB_dstats = dist_stat(CB_df.sst_filt,monthlyf,yrs*1000)

obs_dstats_df = pd.concat([ER_dstats,HD_dstats,CB_dstats],axis=1).T
obs_dstats_df.index=(['ER','HD','CB'])
obs_dstats_df = obs_dstats_df.drop('shapiro_p',axis = 1)
tmp = np.stack((ER_mhw_summary, HD_mhw_summary, CB_mhw_summary)).reshape(3,5)
obs_mhwmetrics_df = pd.DataFrame(tmp, index = ['ER','HD','CB'], columns = ['n_tot','tot_mo','cum_int','mean_dur','mean_int']) 

###############################################################################
#Calcualte statistics and mhw metrics
print('calculating distribution statistics and mhw metrics...')
#file = LIM_df
#Add a climatology
clim_y=np.tile(clim,int(file.shape[0]/12))
clim_i=np.tile(clim_y,(int(file.shape[1]),1))
file = file + clim_i.T

summary = np.zeros((file.shape[1],5))
dstats = np.zeros((file.shape[1],25))

for x in range(0,file.shape[1]):
    tmp_df = pd.DataFrame(file.iloc[:,x].values, index = file.index, columns = ['sst'])

    #Remove climatology for consistency
    tmp = tmp_df.groupby(tmp_df.index.month).transform(lambda sst: sst-sst.mean())
    tmp_df['sstWOclim'] = tmp.sst
    
    #Calculate marine heatwave metrics and distribution statistics
    mhws = mhw_alt(tmp_df.index,tmp_df.sstWOclim,np.std(tmp_df.sstWOclim)*s,m) 
    yrs=int(len(tmp_df)/12)
    if mhws[0] > 0:
        summary[x,0] = mhws[0]/(yrs/10)             #n_tot/10yr
        summary[x,1] = sum(mhws[3])/(yrs/10)        #tot_mo/10yr
        summary[x,2] = sum(mhws[5])/(yrs/10)        #cum_int/10yr
        summary[x,3] = np.mean(mhws[3])             #mean_dur
        summary[x,4] = np.mean(mhws[4])             #mean_int 

    dstats[x,] = np.squeeze(dist_stat(tmp_df.sst,monthlyf,yrs*1000))

LIM_mhwmetrics_df = pd.DataFrame(summary, columns = ['n_tot','tot_mo','cum_int','mean_dur','mean_int']) 
LIM_dstats_df = pd.DataFrame(dstats, columns = ['shapiro_stat','shapiro_p','kurt','skew', 'Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21'])
LIM_all_df = pd.concat([LIM_mhwmetrics_df,LIM_dstats_df],axis=1)
LIM_all_df['source_id']= file.columns
#p value is always very low, never yields a good calibration and throws errors during regression
LIM_dstats_df = LIM_dstats_df.drop('shapiro_p',axis = 1)


###############################################################################
#Partial least squares regression
print('regressing mhw metrics against distribution stats...')
pls_stats = np.zeros((LIM_mhwmetrics_df.shape[1],4))
for i in range(0,LIM_mhwmetrics_df.shape[1]):
    # Extract features (X) and target variable (y)
    X = LIM_dstats_df
    y = LIM_mhwmetrics_df.iloc[:,i]
    rmse = np.zeros((30,10))
    # Test PLS model with up to 10 components.     
    for x in range(0,30):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        for z in range(0,10):
            n_components = z+1
            pls_model = PLSRegression(n_components=n_components)
            pls_model.fit(X_train, y_train)
            y_pred = pls_model.predict(X_test)
            rmse[x,z] = (mean_squared_error(y_test, y_pred))**0.5
    rmse_df = pd.DataFrame(rmse, columns = range(1,11))
    #user defines the lowest mse from iterations above
    ax = sns.stripplot(rmse_df,color='k',alpha = 0.3)
    ax.set(xlabel='number of components', ylabel='RMSE', title=LIM_mhwmetrics_df.columns[i])
    sns.pointplot(rmse_df)
    tmp=rmse_df.T.pct_change().mean(axis=1)
    plt.scatter((tmp.loc[tmp>-0.01].index[0]-2),rmse_df.values.min(),marker = '^',color='gold',s=100)
    plt.scatter(np.argmin(rmse_df.T.mean(axis=1)),rmse_df.values.min(),marker = '^',color='r')
    plt.show()
    n_components = int(input("Enter the number of components for the PLS regression: "))
    
    #pls_stats[i,0] = int(n_components)
    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X, y)
    y_pred = pls_model.predict(X)
    print('r2 = ', round(pls_model.score(X, y),2)) 
    print('rmse= ', round(np.mean(rmse_df[n_components]),3))
    obs_pred = pls_model.predict(obs_dstats_df)
    print('rmse_obs= ', round(mean_squared_error(obs_mhwmetrics_df.iloc[:,i], obs_pred)**0.5,3))
    
    plt.scatter(y, y_pred, c='red', alpha = 0.5, label='LIM data')
    plt.plot([min(y), max(y)], [min(y), max(y)], '--', c='red', label='1:1 line')
    plt.scatter(obs_mhwmetrics_df.iloc[:,i],obs_pred,c='k', label = 'observations')
    plt.xlabel("Actual MHW metric")
    plt.ylabel("Predicted MHW metric")
    plt.title(LIM_mhwmetrics_df.columns[i]+" "+species)
    plt.legend()
    plt.show()
    
    #In practice only a small fraction of months will be sampled by forams.
    #Evaluate this influence by undersampling the original SST timeseries,
    #recalculating distribution statistics, applying the model to calculate 
    #mhw metric and then comparing to "true" mhw metric calculated from the 
    #entire dataset
    usampl = input("Evaluate undersampling (suggested only for promising relationships)? (y/n): ")
    if usampl == 'y':
        sedyrs = int(float(input("Each of your samples represent about how many years (can be calculated by sedimentation rate)?: ")))
        #nforams = int(input("About how many forams will you measure in each sample?: "))
        nforams = np.array([50,100,200,400,800])
        if sedyrs > 500 and sedyrs <2000:
            nforams = np.array([200,300,400,600,800])
        if sedyrs > 2000:
            nforams = np.array([800,1200,1600,2000])
        rmse_usampl = np.zeros((len(nforams)))
        rndm = random.sample(range(0, file.shape[1]), 500)
        for z in range(0, len(nforams)):
            f = nforams[z]/(sedyrs*12)
            y_pred_usampl = np.zeros((len(rndm)))
            y_tru_usampl = LIM_mhwmetrics_df.iloc[rndm,i]
            for x in range(0,len(rndm)):  
                tmp_df = pd.DataFrame(file.iloc[:,rndm[x]].values, index = file.index, columns = ['sst'])
                Xusampl = dist_stat(tmp_df.sst,monthlyf,int(len(tmp_df)*f)).T
                Xusampl = Xusampl.drop('shapiro_p',axis = 1)
                y_pred_usampl[x] = pls_model.predict(Xusampl)
            rmse_usampl[z] = round((np.sum((y_pred_usampl-y_tru_usampl)**2)/len(rndm))**0.5,3)
        print('sampling this many forams: ', nforams)
        print('gives these rmse values: ', rmse_usampl,3)
        print('the mean of true values is: ', round(np.mean(LIM_mhwmetrics_df.iloc[:,i]),3))
        
        b, a = np.polyfit(np.log(nforams), np.log(rmse_usampl), deg=1)
        
        plt.scatter(nforams, rmse_usampl)
        plt.plot(np.arange(min(nforams),max(nforams),10),np.exp(a+b*np.log(np.arange(min(nforams),max(nforams),10))))
        plt.xlabel("forams measured (n)")
        plt.ylabel("RMSE")
        plt.title(LIM_mhwmetrics_df.columns[i]+" "+species)
        plt.show()
        print('Calculate RMSE for any number of forams using: ln(rmse) = ', round(a,3), '+', round(b,3), '* ln(n forams)')
        
        ldngs=pd.DataFrame(pls_model.x_loadings_, columns = list(range(1,n_components+1)))
        ax = sns.lineplot(data=ldngs,marker = 'o', palette = "magma")
        ax.set(ylabel = 'loadings', title = LIM_mhwmetrics_df.columns[i]+" "+species) 
        ax.set_xticks(list(range(24)))
        ax.set_xticklabels(LIM_dstats_df.columns, rotation =90)
        ax.legend(title = 'component')
        ax.axhline(0,color='k')
        plt.show()
        
        n = int(len(y)*0.1)
        top10 = LIM_dstats_df.iloc[sorted(range(len(y)), key=lambda i: y[i], reverse=True)[:n],:]
        bot10 = LIM_dstats_df.iloc[sorted(range(len(y)), key=lambda i: y[i])[:n]] 
        diff = (np.mean(top10,0)-np.mean(bot10,0))/np.mean(top10,0)*100
        plt.plot(diff,'ko')
        plt.axhline(0,color='k')
        plt.xticks(rotation=90)
        plt.ylabel("Percent MHW metric change (top-bottom 10%)")
        plt.title(LIM_mhwmetrics_df.columns[i]+" "+species)
        plt.show()
        
        sav = input("Save this calibration? (y/n): ")
        if sav == 'y':
            name = filename+"_"+species+"_s"+str(int(s*10))+"m"+str(m)+".pkl"
            pickle.dump(pls_model, open(name, 'wb'))
        
        