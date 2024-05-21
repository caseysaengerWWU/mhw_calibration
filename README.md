# mhw_calibration
Scripts and example code to apply the method described by Saenger et al. in the manuscript "A framework for reconstructing marine heatwaves from individual foraminifera in low-resolution sedimentary archives"

Please site the original publication when using this code.

Below if a brief description of the files included and what to do with them. I do not claim to be an expert scripter and you may well run into errors. I am happy to help troubleshoot these as I am able.

IF YOU HAVE NEVER GENERATED A CALIBRATION, first get the LIM ensemble. Unfortunately these data are not yet publically available, but can be accessed with permission here: https://drive.google.com/drive/u/1/folders/19oaETxaxbwCCmsf4MHsx6JIZgKQrKjMo. To gain access please contact saengec@wwu.edu with a brief description of your application. Once you have access it is recommended to download all 20 NetCDF files to a local drive.

IF YOU ARE GENERATING A CALIBRATION AT A NEW SITE, run LIM_sstget.py to extract the LIM SST timeseries from the gridbox closest to the site. You will be prompted assign a name to the site, and the script will generate a .csv with SST data at this site, appended with _SST_LIM.csv. Note that the LIM domain is currently restricted to the tropical and north Pacific.

TO EXPLORE MHW CALIBRATIONS, run PacificMHWcal_LIM.py. You will be prompted to enter a sitename_SST_LIM.csv file of SST data, define MHW intensity and duration thresholds and select one of 5 foraminifera species. You will subsequently be asked to select the number of components to include in a PLSR and if under-sampling should be evaluated. If under-sampling is evaluated, the user must enter the amount of time represented in 1 cm of sediment, which can be derived from sedimentation rates. If the user chooses to save a calibration it will be written as a .pkl file that includes the site name, species and defined thresholds. Intensity thresholds are represented by an s (for standard deviation) follow by the assigned value multiplied by 10 (to accommodate non-integer thresholds). Duration thresholds are represented by an m (for months).

TO APPLY A CALIBRATION TO PROXY DATA, format proxy data into a 3 column .csv with column headings of "sample", "proxy" and "sst". DSDP3618O_T.csv is an example. Then run MHW_applycal.py
