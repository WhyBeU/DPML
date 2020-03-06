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
TEMPERATURE = [300,300,300,300,300]
# DOPING = [1e15,1e15,1e15,1e15,1e15]
DOPING = [5e14,1e15,5e15,1e16,5e16]
WAFERTYPE = 'p'
# %%-
# %%--  Hyper-parameters
PARAMETERS = {
    'name': 'test',
    'save': False,   # True to save a copy of the printed log, the outputed model and data
    'logML':True,   #   Log the output of the console to a text file
    'n_defects': 1000, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,10),# Number of points to interpolate the curves on
    'training_keys': ['Et_eV_upper','Et_eV_lower','logk_upper','logk_lower'] # for k prediction
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
exp.saveExp()
# %%-

# %%--  Train machine learning algorithms
exp = Experiment.loadExp(SAVEDIR+"objects\\", filename='experimentObj_Doping variation_2020-03-06-11-33-49')
ml = exp.newML()
for trainKey in exp.parameters['training_keys']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    ml.trainRegressor(targetCol=targetCol, trainParameters={'bandgap':bandgapParam})
    ml.plotRegressor(trainKey, plotParameters={'scatter_c':'darkred'})

ml.trainRegressor(targetCol='Et_eV', trainParameters={'bandgap':'upper'})
len(ml.dataset)

mlID=exp.updateLogMLmodel(ml,ml.logID)
exp.saveExp()
# %%-
# %%--  Make ML predictions
ML.DefaultParameters
# %%-
df_Val = ml.logTrain['Et_eV_upper']['validation_data']

# %%--
exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS)
exp.loadCSV(FilePath=FILEPATH,Temperature=TEMPERATURE,Doping=DOPING, Type=WAFERTYPE)
exp.interpolateSRH()
exp.plotSRH()
exp.generateDB()
exp.saveExp()
exp.loadLTS()
exp.logbook
exp.logDataset['0']

exp = Experiment.loadExp(SAVEDIR+"objects\\")
ml = exp.newML()
ml.__dict__
ml.logTrain
ml.trainRegressor(targetCol='Et_eV')
ml.trainClassifier(targetCol='bandgap')
