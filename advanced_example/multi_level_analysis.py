#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
from DPML import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    'n_defects': 100000, # Size of simulated defect data set for machine learning
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

# %%--  Train CNN for feature extraction
#  <subcell>    Settings
dataDf=exp.logDataset['0'].copy(deep=True)
save=PARAMETERS['save']
Ycol="Label"
nb_classes=2
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
dpdl = DPDL(SAVEDIR, CNN,name='DPDL_multi_level_analysis',save=save)
dpdl.subset_size = subset_size
dpdl.batch_size = batch_size
dpdl.split_size = split_size
dpdl.n_epochs = n_epochs
dpdl.CM_fz = CM_fz
dpdl.non_feature_col = PARAMETERS['non-feature_col']
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
# %%-

# %%--  Feature extraction
Ycol='Label'
CNN.FC = nn.Sequential(nn.ReLU())
CNN.cuda()
CNN.eval()
data=dataDf.copy(deep=True)
Set = Dataset(data, Ycol, PARAMETERS['non-feature_col'], len(PARAMETERS['dn_range']), len(TEMPERATURE), transform)
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
exp.uploadDB(extracted_df) # ID=1
# %%-

# %%--  Train machine learning algorithms
ml = exp.newML(datasetID='1')
for trainKey in ['Label_all']:
    targetCol, bandgapParam = trainKey.rsplit('_',1)
    param={'bandgap':bandgapParam,'non-feature_col':PARAMETERS['non-feature_col']}
    ml.trainClassifier(targetCol=targetCol, trainParameters=param)
# %%-

# %%--  Predict
header=PARAMETERS['non-feature_col']
predictCsv = {}
vector = [t for key in exp.expKeys for t in exp.expDic[key]['tau_interp']]
extracted_vector = CNN(transform(Set.row_to_img(pd.Series(vector))).unsqueeze(0)).squeeze(0).detach().numpy()
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
        predictCsv[trainKey] = mlDic['model'].predict([extracted_vector])[0]
        # if mlDic['train_parameters']['normalize']:  self.predictCsv[mlID][trainKey] = mlDic['scaler'][targetCol].inverse_transform([mlDic['model'].predict([vector])])[0][0]
    if mlDic['prediction_type'] == 'classification':
        predictCsv[trainKey] = (mlDic['model'].predict([extracted_vector])[0],mlDic['model'].predict_proba([extracted_vector])[0])

#   Log in the trace
Logger.printTitle(' ML PREDICTION',titleLen=60, newLine=False)
Logger.printDic(predictCsv)
exp.updateLogbook('prediction_made')
# %%-
