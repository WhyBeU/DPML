# %%--  Playground
%reload_ext autoreload
%autoreload 2
# %%-

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
# %%--  Inputs
SAVEDIR = "C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\06-Github-TDD-savedire\\compareAlgorithm\\"
# TEMPERATURE = [200,250,300,350,400]
TEMPERATURE = [200,200,200,200,200,250,250,250,250,250,300,300,300,300,300,350,350,350,350,350,400,400,400,400,400]
# TEMPERATURE = [300,300,300,300,300]
# DOPING = [1e15,1e15,1e15,1e15,1e15]
# DOPING = [5e14,1e15,5e15,1e16,5e16]
DOPING = [5e14,1e15,5e15,1e16,5e16,5e14,1e15,5e15,1e16,5e16,5e14,1e15,5e15,1e16,5e16,5e14,1e15,5e15,1e16,5e16,5e14,1e15,5e15,1e16,5e16]
WAFERTYPE = 'n'
# %%-
# %%--  Hyper-parameters
PARAMETERS = {
    'name': 'Doping-temperature-combine-variation-ntype',
    'save': True,   # True to save a copy of the printed log, the outputed model and data
    'logML':True,   #   Log the output of the console to a text file
    'n_defects': 100000, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on
    'regression_training_keys': ['Et_eV_upper','Et_eV_lower','logk_all'], # for k prediction
    'classification_training_keys': ['bandgap_all'], # for k prediction
    # 'training_keys': ['Et_eV_upper','Et_eV_lower','logSn_upper','logSn_lower','logSp_upper','logSp_lower'] # for Sn,Sp prediction
}
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Run experiment
#///////////////////////////////////////////
# %%--  Define experiment and generate defect database
exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS)
exp.updateParameters({'type':WAFERTYPE,'temperature':TEMPERATURE,'doping':DOPING})
exp.generateDB()
# exp.saveExp()
# %%-

# %%--  Train machine learning algorithms
# exp = Experiment.loadExp(SAVEDIR+"objects\\", filename='experimentObj_Doping variation_2020-03-06-11-33-49')
ml = exp.newML()
for trainKey in exp.parameters['regression_training_keys']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    ml.trainRegressor(targetCol=targetCol, trainParameters={'bandgap':bandgapParam})
for trainKey in exp.parameters['classification_training_keys']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    ml.trainClassifier(targetCol=targetCol, trainParameters={'bandgap':bandgapParam})
for trainKey in exp.parameters['regression_training_keys']: ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})
#

# exp.saveExp()
# %%-
# %%--  Save
ml.saveML()
exp.saveExp()
# %%-

# %%--  Inspect
exp.logbook
exp.logDataset['1'].head()
exp.parameter
exp.logML['0'].logTrain
ml.__dict__
exp.__dict__
ml.logger = None
np.power(10,exp.predictCsv['0']['logk_all'])
# %%-
