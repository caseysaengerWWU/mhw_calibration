#Gather SST data from LIM simulations and export of file

"""
Created on Tue Apr  2 11:29:51 2024

@author: caseysaenger
"""
import pandas as pd
import numpy as np
import netCDF4 as nc

site=input("Enter a unique name for your site (no spaces): ")
lat=float(input("Enter the latitude of your site in decimal degrees (e.g. 44.88): "))
lon=float(input("Enter the longitude of your site in decimal degrees (e.g. -130.60): "))
path=input("Enter the path to a folder with the LIM ensemble data files. Be sure to end with a slash: ")

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

###############################################################################
#Get LIM ensemble data, which is divided up into files of 100 years
start=np.arange(1,2001,100)
end=np.arange(100,2100,100)
LIM_ssta = np.zeros((840,2000))                     # initialize

for i in range(0,len(end)):
    print('fetching LIM sst anomalies...')
    dat = nc.Dataset(path+'northPacific_ens'+str(start[i])+'to'+str(end[i])+'.nc')
    LIM_lat = dat.variables['lat'][:]
    LIM_lon = dat.variables['lon'][:]
    iy,ix = flat_fast(LIM_lat, LIM_lon, lat, lon)
    tmp = dat.variables['ens_ssta'][:,:,iy,ix]
    LIM_ssta[:,start[i]-1:end[i]] = np.transpose(tmp)

LIM_time = pd.date_range(start = '1/15/1950',freq = 'M', periods = 840)
tmp_df = pd.DataFrame({'lat': [lat] * len(LIM_time),'lon':[lon] * len(LIM_time)}, index = LIM_time)
LIM_df = pd.DataFrame(LIM_ssta, index = LIM_time)
df = pd.concat([tmp_df,LIM_df],axis = 1)
df.index.name = 'date'
df.to_csv(site+"_SST_LIM.csv")


