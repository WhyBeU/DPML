"Experimental main functions"
from ..si import Cell,Defect,LTS
import numpy as np
import os
import warnings
import pandas as  pd

class Experiment():
    #****   Constant declaration    ****#

    #****   Method declaration      ****#
    def __init__(self,SaveDir,Parameters):
        #   Check applicability of method
        if not os.path.exists(SaveDir): raise ValueError('%s does not exists'%(SaveDir))

        #   Create directory for computation on this dataset
        self.pathDic = {
            'savedir':      SaveDir,
            'figures':      SaveDir+"\\figures\\",
            'objects':      SaveDir+"\\objects\\",
            'traces':       SaveDir+"\\traces\\",
            'outputs':      SaveDir+"\\outputs\\",
        }
        for key, value in self.pathDic.items():
                if key in ['figures','objects','traces','outputs']:
                    if not os.path.exists(value):   os.makedirs(value)

        #   define hyper parameters for experiment
        self.parameters = Parameters

    def loadCSV(self,FilePath, Temperature, Doping, Type):
        #   Check applicability of method
        if not os.path.exists(FilePath): raise ValueError('%s does not exists'%(FilePath))
        if '.csv' not in FilePath: raise ValueError('%s is not a csv file'%(FilePath))
        if len(Temperature)!=len(Doping): raise ValueError('Doping and Temperature array needs to be the same length')
        if Type not in ['n','p']: raise ValueError('%s needs to be either n or p'%(Type))

        #   Load csv file and check the correct format
        self.pathDic['csvfile']=FilePath
        self.csvDf = pd.read_csv(FilePath)
        if len(self.csvDf.columns)!=2*len(Temperature): raise ValueError('csv file does not match to Temperature and Doping array given')

        #   Create experimental data storage for further calculations
        self.expKeys = ["Exp#"+str(i) for i in range(len(Temperature))]
        new_columns = [[k1+"_Dn",k2+"_Tau"] for k1,k2 in zip(self.expKeys,self.expKeys)]
        self.csvDf.columns = [l1 for subL in new_columns for l1 in subL]
        self.expDic={}
        for key,T,Ndop in zip(self.expKeys,Temperature,Doping):
            self.expDic[key]={
                'T':T,
                'Ndop':Ndop,
                'Dn_csv':self.csvDf[key+"_Dn"].dropna().values,
                'Tau_csv':self.csvDf[key+"_Tau"].dropna().values,
                'cell': Cell(T=T,Ndop=Ndop,type=Type),
            }

        #   Save parameter in Parameter dicitionary
        changedParameter = {'type':Type,'temperature':Temperature,'doping':Doping}
        self.updateParameters(changedParameter)



    def updateParameters(self,Parameters):
        return None
