# %%--  Playground
%reload_ext autoreload
%autoreload 2
import os
import pandas as pd
import datetime
import pickle
dir()
exp.__dict__
datetime.datetime.now()
d1=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
d2=datetime.datetime.strptime(d1,"%Y-%m-%d-%H-%M-%S")


plt.plot([0,1],[0,1])
import matplotlib.pyplot as plt
import json
df = pd.read_csv("C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\04-Experimental data -ML\\ML\\data\\2019-05-24-16-46_SRH-data_N-100000_T-[227.29999999999998, 251.79999999999998, 275.79999999999995, 301.4, 320.5, 344.29999999999995, 367.9, 391.29999999999995]_Dn-100pts_type-n_Ndop-5E+15.csv")
df.to_csv("csv-file.csv")
df.to_json("json-file.json")
SaveObj(df,"","pickle-file")
def SaveObj(obj, folder, name):
    if '.pkl' in name:
        with open(folder + name, 'rb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(folder + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def LoadObj(folder, name):
    if '.pkl' in name:
        with open(folder + name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(folder + name + '.pkl', 'rb') as f:
            return pickle.load(f)

df2 =  LoadObj("","pickle-file")

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
# %%--  Instructions:
'''
    1/ Choose SAVEDIR folder where to save the output files from DPML. (Use absolute path)
    2/ Choose FILENAME of measurements. Check out the sample.csv file for correct formattingself.
    header doesn't matter as long as they math the temperature and doping order in input
    3/ Provide TEMPERATURE as a list of the temperature in Kelvin for each measurements
    4/ Provide DOPING as a list of the temperature in cm-3 for each measurements
    5/ Provide cell type 'n' or 'p'
    6/ Change any other hyper-parameters as desired
'''
# %%-
# %%--  Inputs
SAVEDIR = "C:\\Users\\z5189526\\Documents\\GitHub\\DPML\\savedir_example\\"
FILEPATH = "C:\\Users\\z5189526\\Documents\\GitHub\\DPML\\example\\sample.csv"
TEMPERATURE = [227.3,251.8,275.8,301.4,320.5,344.3,367.9,391.3]
DOPING = [5.1e15,5.1e15,5.1e15,5.1e15,5.1e15,5.1e15,5.1e15,5.1e15]
WAFERTYPE = 'n'
# %%-
# %%--  Hyper-parameters
PARAMETERS = {
    'name': 'Example github test',
    'save': True,   # True to save a copy of the printed log, the outputed model and data
    'n_defects': 1000, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,10),# Number of points to interpolate the curves on
    'training_keys': ['Et_eV_upper','Et_eV_lower','logk_upper','logk_lower'] # for k prediction
    # 'training_keys': ['Et_eV_upper','Et_eV_lower','logSn_upper','logSn_lower','logSp_upper','logSp_lower'] # for Sn,Sp prediction
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
exp.saveExp()
# %%-

# %%--  Train machine learning algorithms
exp = Experiment.loadExp(SAVEDIR+"objects\\")
ml = exp.newML()
for trainKey in exp.parameters['training_keys']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    ml.trainRegressor(targetCol=targetCol, trainParameters={'bandgap':bandgapParam})

ml.trainClassifier(targetCol='bandgap')
mlID=exp.updateLogMLmodel(ml,ml.logID)
exp.saveExp()
# %%-
# %%--  Make ML predictions

# %%-

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




ml.plotRegressor()
ml.train2Classifier()
ml.plotClassifier()
ml.saveMLModel()
exp.loadMLModel()
exp.predict(ml) # or load oldest model

#other functions:
ml.retrainModel()
# index dataset and training with Ids to match what trained what
# %%-
