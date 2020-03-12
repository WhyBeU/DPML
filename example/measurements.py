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
NAME = 'Example github'
# %%-
# %%--  Hyper-parameters
PARAMETERS = {
    'name': NAME,
    'save': True,   # True to save a copy of the printed log, the outputed model and data
    'logML':True,   #   Log the output of the console to a text file
    'n_defects': 1000, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,100),# Number of points to interpolate the curves on
    'classification_training_keys': ['bandgap_all'], # for k prediction
    'regression_training_keys': ['Et_eV_upper','Et_eV_lower','logk_all'], # for k prediction
    # 'regression_training_keys': ['Et_eV_upper','Et_eV_lower','logSn_upper','logSn_lower','logSp_upper','logSp_lower'], # for Sn,Sp prediction
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
for trainKey in exp.parameters['regression_training_keys']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    ml.trainRegressor(targetCol=targetCol, trainParameters={'bandgap':bandgapParam})
for trainKey in exp.parameters['classification_training_keys']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    ml.trainClassifier(targetCol=targetCol, trainParameters={'bandgap':bandgapParam})
for trainKey in exp.parameters['regression_training_keys']: ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})

mlID=exp.updateLogMLmodel(ml,logID=ml.parameters['mlID'])
ml.saveML()
exp.saveExp()
# %%-
# %%--  Make ML predictions
exp = Experiment.loadExp(SAVEDIR+"objects\\")
exp.predictML()
exp.predictCsv
# %%-

# %%--  Export data
exp = Experiment.loadExp(SAVEDIR+"objects\\")
ml = exp.loadML()

exp.exportDataset()
exp.exportSRHCurves()
exp.exportValidationData()
exp.exportPrediction()
# %%-


# %%--  Inspect
exp.logbook
exp.logDataset['1'].head()
exp.parameter
exp.logML['1'].logTrain
ml.__dict__
exp.__dict__
ml.logger = None
np.power(10,exp.predictCsv['0']['logk_all'])
# %%-

# %%--  Reload specific experiment
expRef = Experiment.loadExp(SAVEDIR+"objects\\", filename='experimentObj_Example github_2020-03-12-15-14-50')
exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS)
for key,value in expRef.__dict__.items():
    exp.__dict__[key]= value
# %%-

# %%--  Test area
selfexp = exp
mlIDs=None
header=None
#   Check for applicabiliy
if mlIDs == None: mlIDs = [str(i) for i in range(len(selfexp.logML))]
if header == None: header = ['Et_eV','Sn_cm2','Sp_cm2','k','logSn','logSp','logk']
selfexp.predictCsv = {}
for mlID in mlIDs:
    ml = selfexp.logML[mlID]
    selfexp.predictCsv[mlID]={}
    #   Create dataset feature vector
    for trainKey, mlDic in ml.logTrain.items():
        vector = [t for key in selfexp.expKeys for t in selfexp.expDic[key]['tau_interp']]
        if trainKey=='scaler': continue
        targetCol, bandgapParam = trainKey.rsplit('_',1)
        if mlDic['train_parameters']['normalize']:
            #   scale the feature vector
            if len(vector) != len(ml.logTrain['scaler'])-len(header): raise ValueError('Feature vector is not the same size as the trained ML')
            i=0
            for scaler_key,scaler_value in mlDic['scaler'].items():
                if scaler_key in header: continue
                vector[i]=np.log10(vector[i])
                vector[i]=scaler_value.transform(vector[i].reshape(-1,1))[0][0]
                i+=1
        #   Call ML model and predict on sample data
        selfexp.predictCsv[mlID][trainKey] = mlDic['scaler'][targetCol].inverse_transform([mlDic['model'].predict([vector])])[0][0]

ml.logTrain.keys()
# %%-
trainKey='Et_eV_upper'
mlDic = ml.logTrain[trainKey]
mlDic.keys()
mlDic['scaler'][targetCol].__dict__
