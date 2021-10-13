#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
from DPML.si import *
from DPML.main import *
from DPML.utils import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Set-up
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
    There are hidden parameters that can be specified in most functions, they
    use by the default the class-defined parameters
    Adjust ML pipeline to test for different ML algorithms. Don't forget to import the ML functions from Sklearn
'''
# %%-

# %%--  Inputs
SAVEDIR = "savedir_example\\"
TEMPERATURE = [200,250,300,350,400]
DOPING = [1e15,1e15,1e15,1e15,1e15]
WAFERTYPE = 'p'
NAME = 'Main'
#   File specific inputs
ML_REGRESSION_PIPELINE={
    "Random Forest": RandomForestRegressor(n_estimators=100, verbose =2, n_jobs=-1),
    "Adaptive Boosting": AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=100, loss='linear'),
    "Gradient Boosting": GradientBoostingRegressor(verbose=2,loss='ls',max_depth=10),
    "Neural Network": MLPRegressor((100,100),alpha=0.001, activation = 'relu',verbose=2,learning_rate='adaptive'),
    "Support Vector": SVR(kernel='rbf',C=5,verbose=2, gamma="auto"),
}
ML_CLASSIFICATION_PIPELINE={
    "Random Forest": RandomForestClassifier(n_estimators=100, verbose =2,n_jobs=-1),
    "Adaptive Boosting": AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators=10),
    "Gradient Boosting": GradientBoostingClassifier(verbose=2,loss='deviance'),
    "Neural Network": MLPClassifier((100,100),alpha=0.001, activation = 'relu',verbose=2,learning_rate='adaptive'),
    "Nearest Neighbors":KNeighborsClassifier(n_neighbors = 5, weights='distance',n_jobs=-1),
}
# %%-

# %%--  Hyper-parameters
PARAMETERS = {
    'name': NAME,
    'save': False,   # True to save a copy of the printed log, the outputed model and data
    'logML': False,   #   Log the output of the console to a text file
    'n_defects': 1000, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on
    'classification_training_keys': ['bandgap_all'], # for  prediction
    'regression_training_keys': ['Et_eV_upper','Et_eV_lower','logk_all'], # for prediction
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

# %%--  Train machine learning algorithms loop
for modelName,model in ML_REGRESSION_PIPELINE.items():
    ml = exp.newML(mlParameters={'name':exp.parameters['name']+"_"+modelName})
    for trainKey in exp.parameters['regression_training_keys']:
        targetCol, bandgapParam = trainKey.rsplit('_',1)
        param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col'],'base_model':model}
        ml.trainRegressor(targetCol=targetCol, trainParameters=param)
        ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})
for modelName,model in ML_CLASSIFICATION_PIPELINE.items():
    ml = exp.newML(mlParameters={'name':exp.parameters['name']+"_"+modelName})
    for trainKey in exp.parameters['classification_training_keys']:
        targetCol, bandgapParam = trainKey.rsplit('_',1)
        param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col'],'base_model':model}
        ml.trainClassifier(targetCol=targetCol, trainParameters=param)

# %%-

# %%--  Export data
exp.exportDataset()
exp.exportValidationset()
# %%-
