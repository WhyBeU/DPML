#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
from DPML.si import *
from DPML.main import *
from DPML.utils import *
import numpy as np
from DPML.utils.matplotlibstyle import *
import matplotlib.pyplot as plt
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Setup
#///////////////////////////////////////////
# %%--  Instructions:
'''
---Main Steps---
      Choose SAVEDIR folder where to save the output files from DPML. (Use absolute path)
      Provide TEMPERATURE as a list of the temperature in Kelvin for each measurements
      Provide DOPING as a list of the temperature in cm-3 for each measurements
      Provide cell type 'n' or 'p'
      NAME your experiment

---Other notes---
    Change hyper-parameters as desired.
    Adjust file specific inputs
    Adjust regression_training_keys to desired value: Et_upper, Et_lower or logk_all
'''
# %%-

# %%--  Inputs
SAVEDIR = "savedir_example\\"
TEMPERATURE = [200,250,300,350,400]
DOPING = [1e15,1e15,1e15,1e15,1e15]
WAFERTYPE = 'p'
NAME = 'Dataset size dependecy'
#   File specific inputs
RANGE = np.logspace(2,5,100)

# %%-

# %%--  Hyper-parameters
PARAMETERS = {
    'name': NAME,
    'save': False,   # True to save a copy of the printed log, the outputed model and data
    'logML':False,   #   Log the output of the console to a text file
    'n_defects': 100000, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on
    'classification_training_keys': ['bandgap_all'], # for parameter prediction
    'regression_training_keys': ['logk_all'], # for parameter prediction
    'non-feature_col':['Name', 'Et_eV', 'Sn_cm2', 'Sp_cm2', 'k', 'logSn', 'logSp', 'logk', 'bandgap'] # columns to remove from dataframe in ML training
}
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Script
#///////////////////////////////////////////

# %%--  Define experiment and generate defect database
exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS)
exp.updateParameters({'type':WAFERTYPE,'temperature':TEMPERATURE,'doping':DOPING})
exp.generateDB()
# %%-

# %%--  Loop
DB = exp.logDataset['0'].copy(deep=True)
TIME=[]
SCORE=[]
for N in RANGE:
    exp.logDataset['0'] = DB.sample(int(N)).copy(deep=True)
    ml = exp.newML()
    for trainKey in exp.parameters['regression_training_keys']:
        targetCol, bandgapParam = trainKey.rsplit('_',1)
        param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col'],'validation_fraction': 0.2}
        ml.trainRegressor(targetCol=targetCol, trainParameters=param)
        TIME.append(float(ml.logTrain[trainKey]['results']['training_time'].split()[0]))
        SCORE.append(float(ml.logTrain[trainKey]['results']['validation_r2'].split()[0]))

exp.logDataset['0'] = DB.copy(deep=True)
# %%-

# %%--  Plot
plotParam={
    'figsize':(6,6),
    'xlabel':'Dataset size',
    'ylabel':'Training time (s)',
}

plt.figure(figsize=plotParam['figsize'])
ax = plt.gca()
ax.set_xlabel(plotParam['xlabel'])
ax.set_ylabel(plotParam['ylabel'])
ax.semilogx()
ax.set_title("Random Forest k prediction", fontsize=14)
ax.scatter(RANGE,TIME)
plt.show()


plotParam={
    'figsize':(6,6),
    'xlabel':'Dataset size',
    'ylabel':'$R^2$ score',
}
plt.figure(figsize=plotParam['figsize'])
ax = plt.gca()
ax.set_xlabel(plotParam['xlabel'])
ax.set_ylabel(plotParam['ylabel'])
# ax.set_ylim(bottom=0)
ax.semilogx()
ax.set_title(exp.parameters['regression_training_keys'], fontsize=14)
ax.scatter(RANGE,SCORE)
plt.show()
# %%-
