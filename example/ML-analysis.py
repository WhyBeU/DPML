#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
from DPML import *
import numpy as np
# %%-
print('Hello')
# %%--  Inputs
SAVEDIR = "C:\\Users\\z5189526\\Documents\\GitHub\\DPML\\savedir_example\\"
# FILEPATH = "C:\\Users\\z5189526\\Documents\\GitHub\\DPML\\example\\sample.csv"
TEMPERATURE = [200,250,300,350,400]
DOPING = [1e15,1e15,1e15,1e15,1e15]
WAFERTYPE = 'p'
NAME = 'Dataset size dependecy'
# %%-

# %%--  Hyper-parameters
PARAMETERS = {
    'name': NAME,
    'save': False,   # True to save a copy of the printed log, the outputed model and data
    'logML':False,   #   Log the output of the console to a text file
    'n_defects': 100000, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on
    'classification_training_keys': ['bandgap_all'], # for k prediction
    'regression_training_keys': ['logk_all'], # for k prediction
}
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Dependencies to dataset size
#///////////////////////////////////////////

# %%--  Define experiment and generate defect database
exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS)
exp.updateParameters({'type':WAFERTYPE,'temperature':TEMPERATURE,'doping':DOPING})
exp.generateDB()
# %%-

# %%--  Loop
RANGE = np.logspace(2,5,100)
DB = exp.logDataset['0'].copy(deep=True)
TIME=[]
SCORE=[]
for N in RANGE:
    exp.logDataset['0'] = DB.sample(int(N)).copy(deep=True)
    ml = exp.newML()
    for trainKey in exp.parameters['regression_training_keys']:
        targetCol, bandgapParam = trainKey.rsplit('_',1)
        ml.trainRegressor(targetCol=targetCol, trainParameters={'bandgap':bandgapParam,'validation_fraction': 0.2})
        TIME.append(float(ml.logTrain[trainKey]['results']['training_time'].split()[0]))
        SCORE.append(float(ml.logTrain[trainKey]['results']['validation_r2'].split()[0]))

exp.logDataset['0'] = DB.copy(deep=True)
# %%-

# %%--  Plot
from DPML.utils.matplotlibstyle import *
import matplotlib.pyplot as plt
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
ax.set_title("Random Forest k prediction", fontsize=14)
ax.scatter(RANGE,SCORE)
plt.show()


# %%-

ml.logTrain['logk_all']['results']
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
