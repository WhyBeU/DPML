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
FILEPATH = "advanced_example\\data\\sample_original_L.csv"
TEMPERATURE = [158,182,206,230,254,278,300]
DOPING = [1.01e16]*len(TEMPERATURE)
WAFERTYPE = 'p'
NAME = 'advanced example - multi_level_L'
# %%-

# %%--  Hyper-parameters
PARAMETERS = {
    'name': NAME,
    'save': False,   # True to save a copy of the printed log, the outputed model and data
    'logML':True,   #   Log the output of the console to a text file
    'n_defects': 1000, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on
    'non-feature_col':['Mode','Label',"Name","Et_eV_1","Sn_cm2_1","Sp_cm2_1",'k_1','logSn_1','logSp_1','logk_1','bandgap_1',"Et_eV_2","Sn_cm2_2","Sp_cm2_2",'k_2','logSn_2','logSp_2','logk_2','bandgap_2']
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
# %%--  Simulate datasets
PARAM={
        'type': 'p',                #   Wafer doping type
        'Et_min_1':-0.55,             #   Minimum defect energy level
        'Et_max_1':0.55,              #   Maximum defect energy level
        'Et_min_2':-0.55,             #   Minimum defect energy level
        'Et_max_2':0.55,              #   Maximum defect energy level
        'S_min_1':1E-17,              #   Minimum capture cross section
        'S_max_1':1E-13,              #   Maximum capture cross section
        'S_min_2':1E-17,              #   Minimum capture cross section
        'S_max_2':1E-13,              #   Maximum capture cross section
        'Nt':1E12,                  #   Defect density
        'check_auger':True,     #   Check wether to resample if lifetime is auger-limited
        'noise':'',             #   Enable noiseparam
        'noiseparam':0,         #   Adds noise proportional to the log of Delta n
}
db_multi=DPML.generateDB_multi(PARAMETERS['n_defects'], TEMPERATURE, DOPING, PARAMETERS['dn_range'], PARAM)
db_sah=DPML.generateDB_sah(PARAMETERS['n_defects'], TEMPERATURE, DOPING, PARAMETERS['dn_range'], PARAM)
db_multi['Mode']=['Two one-level']*len(db_multi)
db_sah['Mode']=['Single two-level']*len(db_sah)
dataDf=pd.concat([db_multi,db_sah])
dataDf['Label']=[0 if mode=="Two one-level" else 1 for mode in dataDf['Mode']]
exp.uploadDB(dataDf)
vocab={
    '0':'Two one-level',
    '1':'Single two-level',
}
# %%-

# %%--  Train classifier
ml = exp.newML()
for trainKey in ['Label_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainClassifier(targetCol=targetCol, trainParameters=param)
# %%-

# %%--  Make ML predictions
exp.predictML(header=PARAMETERS['non-feature_col'])
print(vocab)
# %%-

# %%--  Export data
exp.exportDataset()
exp.exportValidationset()
# %%-
