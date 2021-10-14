#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
from DPML import *
import numpy as np
import matplotlib.pyplot as plt
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Setup
#///////////////////////////////////////////
# %%--  Instructions:
'''
---Main Steps---
    1/  Choose SAVEDIR folder where to save the output files from DPML. (Use absolute path)
    2/  Choose FILEPATH of measurements. Check the sample_tempccs.csv file for correct formatting.
        Each pair of columns needs to match the elements of TEMPERATURE and DOPING in order.
    3/  Provide TEMPERATURE as a list of the temperature in Kelvin for each measurements
    4/  Provide DOPING as a list of the temperature in cm-3 for each measurements
    5/  Provide cell type 'n' or 'p'
    6/  NAME your experiment

---Other notes---
    Change hyper-parameters as desired.
    There are hidden parameters that can be specified in most functions, they
    use by the default the class-defined parameters
    Not executing the functions in the correct order can results in errors.
    Comment or uncomment load and save point as required.
'''
# %%-

# %%--  Inputs
SAVEDIR = "savedir_example\\"
FILEPATH = "advanced_example\\data\\sample_tempccs_L.csv"
TEMPERATURE = [158,182,206,230,254,278,300]
DOPING = [1.01e16]*len(TEMPERATURE)
WAFERTYPE = 'p'
NAME = 'advanced example - tempccs_L'
# %%-

# %%--  Hyper-parameters
PARAMETERS = {
    'name': NAME,
    'save': False,   # True to save a copy of the printed log, the outputed model and data
    'logML':True,   #   Log the output of the console to a text file
    'n_defects': 100000, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on
    'non-feature_col':['Name', 'Et_eV', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'logk', 'bandgap','CMn','CMp','CPn','CPp'] # columns to remove from dataframe in ML training
}
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Script
#///////////////////////////////////////////
# %%--  Define experiment
exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS)
exp.loadCSV(FilePath=FILEPATH,Temperature=TEMPERATURE,Doping=DOPING, Type=WAFERTYPE)
exp.interpolateSRH()
exp.plotSRH()
# %%-

# %%--  Simulate dataset
PARAM={
        'type': WAFERTYPE,                #   Wafer doping type
        'Et_min':-0.55,             #   Minimum defect energy level
        'Et_max':0.55,              #   Maximum defect energy level
        'S_min':1E-17,              #   Minimum capture cross section
        'S_max':1E-13,              #   Maximum capture cross section
        'Nt':1E12,                  #   Defect density
        'CMn_tab':['Radiative','Multiphonon emission', 'Cascade'],    #   Capture mode for Sn
        'CMp_tab':['Radiative','Multiphonon emission', 'Cascade'],    #   Capture mode for Sp
        'Force_same_CM':False,      #   Wether to force Sn and Sp to follow CMn_tab
        'check_auger':True,        #   Check wether to resample if lifetime is auger-limited
        'noise':'',     #   Enable noiseparam
        'noiseparam':0.000,     #   Adds noise proportional to the log of Delta n
}
db,dic=DPML.generateDB(PARAMETERS['n_defects'], TEMPERATURE, DOPING, PARAMETERS['dn_range'], PARAM)
''' IDs of datasets
0 --> all data
1 --> CMp = MPE
2 --> CMp = CAS
3 --> CMn = MPE
4 --> CMn = CAS
'''
#   Id 0
exp.uploadDB(db)
#   Id 1
locdf=db.copy(deep=True)
locdf=locdf[locdf['CMp']=='Multiphonon emission']
exp.uploadDB(locdf)
#   Id 2
locdf=db.copy(deep=True)
locdf=locdf[locdf['CMp']=='Cascade']
exp.uploadDB(locdf)
#   Id 3
locdf=db.copy(deep=True)
locdf=locdf[locdf['CMn']=='Multiphonon emission']
exp.uploadDB(locdf)
#   Id 4
locdf=db.copy(deep=True)
locdf=locdf[locdf['CMn']=='Cascade']
exp.uploadDB(locdf)
# %%-

# %%--  Regression of Defect parameter
ml = exp.newML(datasetID='0')
for trainKey in ['Et_eV_upper','Et_eV_lower','logk_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainRegressor(targetCol=targetCol, trainParameters=param)    #   RF regressor, 0.01 validation set
    ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})
# %%-

# %%--  Capture parameter regression
# 1 --> CMp = MPE
ml = exp.newML(datasetID='1')
for trainKey in ['CPp_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainRegressor(targetCol=targetCol, trainParameters=param)    #   RF regressor, 0.01 validation set
    ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})

# 2 --> CMp = CAS
ml = exp.newML(datasetID='2')
for trainKey in ['CPp_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainRegressor(targetCol=targetCol, trainParameters=param)    #   RF regressor, 0.01 validation set
    ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})

# 3 --> CMn = MPE
ml = exp.newML(datasetID='3')
for trainKey in ['CPn_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainRegressor(targetCol=targetCol, trainParameters=param)    #   RF regressor, 0.01 validation set
    ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})

# 4 --> CMp = CAS
ml = exp.newML(datasetID='4')
for trainKey in ['CPn_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainRegressor(targetCol=targetCol, trainParameters=param)    #   RF regressor, 0.01 validation set
    ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})
# %%-

# %%--  Mode classification
ml = exp.newML(datasetID='0')
for trainKey in ['bandgap_all','CMn_all','CMp_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainClassifier(targetCol=targetCol, trainParameters=param)
# %%-

# %%--  Make ML predictions
exp.predictML(header=PARAMETERS['non-feature_col'])
# %%-

# %%--  Export data
exp.exportDataset()
exp.exportValidationset()
# %%-
