#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
from DPML import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#   For DPDL
from DPDL import *
import torch
import copy
import torch.nn as nn
import torch.utils.data as Data
from torchvision import datasets, models, transforms

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

# %%--  Adapt set for CNN
dataDf=exp.logDataset['0'].copy(deep=True)
dataDf['Labels_CMp']=[0 if m=='Radiative' else (1 if m=='Multiphonon emission' else 2) for m in dataDf['CMp']]
dataDf['Labels_CMn']=[0 if m=='Radiative' else (1 if m=='Multiphonon emission' else 2) for m in dataDf['CMn']]
Mode=[]
for CMn, CMp in zip(dataDf['CMn'],dataDf['CMp']):
    if CMn=='Radiative':
        if CMp=='Radiative': Mode.append(0)
        if CMp=='Multiphonon emission': Mode.append(1)
        if CMp=='Cascade': Mode.append(2)
    if CMn=='Multiphonon emission':
        if CMp=='Radiative': Mode.append(3)
        if CMp=='Multiphonon emission': Mode.append(4)
        if CMp=='Cascade': Mode.append(5)

    if CMn=='Cascade':
        if CMp=='Radiative': Mode.append(6)
        if CMp=='Multiphonon emission': Mode.append(7)
        if CMp=='Cascade': Mode.append(8)

dataDf['Mode']=Mode
vocab={
    '0':'n_Radiative + p_Radiative',
    '1':'n_Radiative + p_Multiphonon emission',
    '2':'n_Radiative + p_Cascade',
    '3':'n_Multiphonon emission + p_Radiative',
    '4':'n_Multiphonon emission + p_Multiphonon emission',
    '5':'n_Multiphonon emission + p_Cascade',
    '6':'n_Cascade + p_Radiative',
    '7':'n_Cascade + p_Multiphonon emission',
    '8':'n_Cascade + p_Cascade',
}
# %%-

# %%--  Train CNN for feature extraction - CMp
#  <subcell>    Settings
save=PARAMETERS['save']
Ycol="Mode"
nb_classes=9
subset_size = None
batch_size = 17
split_size = 0.1
n_epochs = 50
CM_fz = 12
img_size = 64
comment="SRH_bandgap_50epochs"
transform_augment = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
#  </subcell>
CNN=dualCNN(nb_classes=nb_classes,input_size=img_size,input_channel=1,dropout_rate=0.2)
dpdl = DPDL(SAVEDIR, CNN,name='tempccs_L_9classes',save=save)
dpdl.subset_size = subset_size
dpdl.batch_size = batch_size
dpdl.split_size = split_size
dpdl.n_epochs = n_epochs
dpdl.CM_fz = CM_fz
dpdl.non_feature_col = PARAMETERS['non-feature_col']+['Labels_CMn','Labels_CMp','Mode']
dpdl.dn_len=len(PARAMETERS['dn_range'])
dpdl.t_len=len(TEMPERATURE)
dpdl.initTraining()
dpdl.trainModel(df=dataDf,Ycol=Ycol,transform=transform,transformTrain=transform_augment,randomSeed=None,comment=comment)
if dpdl.predictType=="Regression":
    res=dpdl.regResults[0]
    res['scaler']=dpdl.scaler
else:
    res=dpdl.classResults[0]
    res['vocab'] = dpdl.vocab
    res["CM"] = dpdl.CM
res['tracefile']=dpdl.tracefile
res['predictfile']=dpdl.predictfile
res['model']=dpdl.model
res['transform']=transform
res['non_feature_col']=dpdl.non_feature_col

#   Feature extraction
CNN.FC = nn.Sequential(nn.ReLU())
CNN.cuda()
CNN.eval()
data=dataDf.copy(deep=True)
Set = Dataset(data, Ycol, PARAMETERS['non-feature_col']+['Labels_CMn','Labels_CMp','Mode'], len(PARAMETERS['dn_range']), len(TEMPERATURE), transform)
Loader = Data.DataLoader(Set,batch_size=39,shuffle=False,num_workers=0)
print('Extract Set')
p=0
first=True
for inputs, targets in Loader:
    p+=1
    print('Batch '+str(p)+' of '+str(len(Loader)))
    inputs = inputs.cuda()
    outputs = CNN(inputs)
    outputs = outputs.cpu().detach().numpy().tolist()
    if first==True:
        first=False
        l = len(outputs[0])
        Xcols = []
        extracted = {}
        for j in range(l):
            extracted['CNN_'+str(j)]=[]
            Xcols.append('CNN_'+str(j))
    inputs=inputs.cpu()
    for k in range(len(outputs)):
        for j in range(l):
            extracted['CNN_'+str(j)].append(outputs[k][j])

    del inputs, targets, outputs
    torch.cuda.empty_cache()
CNN.cpu()
extracted_df=pd.DataFrame(extracted)
for col in extracted_df.columns: extracted_df[col]=10**extracted_df[col].values
for col in PARAMETERS['non-feature_col']: extracted_df[col]=data[col].values
exp.uploadDB(extracted_df) # ID=5
# %%-

# %%--  New extracted datasets
''' IDs of datasets
5 --> all data
6 --> CMp = MPE
7 --> CMp = CAS
8 --> CMn = MPE
9 --> CMn = CAS
'''

#   Id 6
locdf=extracted_df.copy(deep=True)
locdf=locdf[locdf['CMp']=='Multiphonon emission']
exp.uploadDB(locdf)
#   Id 7
locdf=extracted_df.copy(deep=True)
locdf=locdf[locdf['CMp']=='Cascade']
exp.uploadDB(locdf)
#   Id 8
locdf=extracted_df.copy(deep=True)
locdf=locdf[locdf['CMn']=='Multiphonon emission']
exp.uploadDB(locdf)
#   Id 9
locdf=extracted_df.copy(deep=True)
locdf=locdf[locdf['CMn']=='Cascade']
exp.uploadDB(locdf)
# %%-

# %%--  Capture parameter regression
# 1 --> CMp = MPE
ml = exp.newML(datasetID='6')
for trainKey in ['CPp_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainRegressor(targetCol=targetCol, trainParameters=param)    #   RF regressor, 0.01 validation set
    ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})

# 2 --> CMp = CAS
ml = exp.newML(datasetID='7')
for trainKey in ['CPp_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainRegressor(targetCol=targetCol, trainParameters=param)    #   RF regressor, 0.01 validation set
    ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})

# 3 --> CMn = MPE
ml = exp.newML(datasetID='8')
for trainKey in ['CPn_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainRegressor(targetCol=targetCol, trainParameters=param)    #   RF regressor, 0.01 validation set
    ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})

# 4 --> CMp = CAS
ml = exp.newML(datasetID='9')
for trainKey in ['CPn_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainRegressor(targetCol=targetCol, trainParameters=param)    #   RF regressor, 0.01 validation set
    ml.plotRegressor(trainKey, plotParameters={'scatter_c':'black'})
# %%-

# %%--  Mode classification
ml = exp.newML(datasetID='5')
for trainKey in ['CMn_all','CMp_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainClassifier(targetCol=targetCol, trainParameters=param)
# %%-

# %%--  Predict
header=PARAMETERS['non-feature_col']
predictCsv = {}
vector = [t for key in exp.expKeys for t in exp.expDic[key]['tau_interp']]
extracted_vector = CNN(transform(Set.row_to_img(pd.Series(vector))).unsqueeze(0)).squeeze(0).detach().numpy()
mlIDs = [str(i) for i in range(len(exp.logML))]
if header == None: header = ['Et_eV','Sn_cm2','Sp_cm2','k','logSn','logSp','logk']
for mlID in mlIDs:
    ml = exp.logML[mlID]
    if int(mlID)<6:continue
    predictCsv[mlID]={}
    for trainKey, mlDic in ml.logTrain.items():
        if trainKey=='scaler': continue
        targetCol, bandgapParam = trainKey.rsplit('_',1)
        if mlDic['train_parameters']['normalize']:
            #   scale the feature vector
            if len(extracted_vector) != len(mlDic['scaler']): raise ValueError('Feature vector is not the same size as the trained ML')
            i=0
            for scaler_key,scaler_value in mlDic['scaler'].items():
                if scaler_key in header: continue
                extracted_vector[i]=scaler_value.transform(extracted_vector[i].reshape(-1,1))[0][0]
                i+=1
        #   Call ML model and predict on sample data
        if mlDic['prediction_type'] == 'regression':
            predictCsv[mlID][trainKey] = mlDic['model'].predict([extracted_vector])[0]
            # if mlDic['train_parameters']['normalize']:  self.predictCsv[mlID][mlID][trainKey] = mlDic['scaler'][targetCol].inverse_transform([mlDic['model'].predict([vector])])[0][0]
        if mlDic['prediction_type'] == 'classification':
            predictCsv[mlID][trainKey] = (mlDic['model'].predict([extracted_vector])[0],mlDic['model'].predict_proba([extracted_vector])[0])

#   Log in the trace
    Logger.printTitle(' ML PREDICTION',titleLen=60, newLine=False)
    Logger.printTitle('mlID '+mlID,titleLen=40)
    Logger.printDic(predictCsv[mlID])
    exp.updateLogbook('prediction_made')
# %%-
