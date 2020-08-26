#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
from DPML import *
import numpy as np
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Setup
#///////////////////////////////////////////
# %%--  Instructions:
'''
---Main Steps---
    1/  Choose SAVEDIR folder where to save the output files from DPML. (Use absolute path)
    2/  Choose FILEPATH of measurements. Check the sample.csv file for correct formatting.
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
'''
# %%-

# %%--  Inputs
SAVEDIR = "DPML\\savedir_example\\"
FILEPATH = "DPML\\example\\sample.csv"
TEMPERATURE = [227.3,251.8,275.8,301.4,320.5,344.3,367.9,391.3]
DOPING = [5.1e15,5.1e15,5.1e15,5.1e15,5.1e15,5.1e15,5.1e15,5.1e15]
WAFERTYPE = 'n'
NAME = 'Example github'
# %%-

# %%--  Hyper-parameters
PARAMETERS = {
    'name': NAME,
    'save': True,   # True to save a copy of the printed log, the outputed model and data
    'logML':True,   #   Log the output of the console to a text file
    'n_defects': 2000, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on
    'classification_training_keys': ['bandgap_all'], # for k prediction
    'regression_training_keys': ['Et_eV_upper','Et_eV_lower','logk_all'], # for k prediction
}
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Script
#///////////////////////////////////////////
# %%--  Define experiment and generate defect database
exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS)
exp.loadCSV(FilePath=FILEPATH,Temperature=TEMPERATURE,Doping=DOPING, Type=WAFERTYPE)
exp.interpolateSRH()
exp.plotSRH()
exp.generateDB()
# exp.saveExp()
# %%-

# %%--  Train machine learning algorithms
# exp = Experiment.loadExp(SAVEDIR+"objects\\")
ml = exp.newML()
for trainKey in exp.parameters['regression_training_keys']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    ml.trainRegressor(targetCol=targetCol, trainParameters={'bandgap':bandgapParam})
for trainKey in exp.parameters['classification_training_keys']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    ml.trainClassifier(targetCol=targetCol, trainParameters={'bandgap':bandgapParam})
for trainKey in exp.parameters['regression_training_keys']: ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})
# ml.saveML()
# exp.saveExp()
# %%-

# %%--  Make ML predictions
# exp = Experiment.loadExp(SAVEDIR+"objects\\")
exp.predictML()
# exp.saveExp()
# %%-

# %%--  Export data
# exp = Experiment.loadExp(SAVEDIR+"objects\\")
# ml = exp.loadML()
exp.exportDataset()
exp.exportValidationset()
# exp.saveExp()
# %%-

# %%--  Reload specific experiment
# expRef = Experiment.loadExp(SAVEDIR+"objects\\", filename='experimentObj_Example github_2020-03-12-15-14-50')
# exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS)
# for key,value in expRef.__dict__.items():
#     exp.__dict__[key]= value
# %%-
