#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
for i in range(1):  #---[CELL]    Import modules
    #   General packages
    import os
    import datetime
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd
    pd.options.mode.chained_assignment = None  # default='warn'
    import numpy as np
    from scipy.stats import linregress
    import scipy.constants as const
    from scipy.optimize import curve_fit, minimize
    #   Sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.svm import SVR, SVC
    from sklearn.linear_model import LogisticRegression, Lasso
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.externals import joblib
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import RandomizedSearchCV, cross_validate
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.linear_model import SGDClassifier
    from sklearn import preprocessing

    #   Module classes
    from DPML.cell import cell
    from DPML.defect import defect
    from DPML.logger import Logger
    from DPML.MLProcess import MLProcess
    from DPML.Nostdstreams import NoStdStreams
    from DPML.simulation import simulation
for i in range(1):  #   [CELL]  Matplotlib style sheet
    mpl.style.use('seaborn-paper')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] ='STIXGeneral'
    mpl.rcParams['mathtext.default'] = 'rm'
    mpl.rcParams['mathtext.fallback_to_cm'] = False
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['axes.labelweight'] = 'normal'
    mpl.rcParams['axes.grid.which']='both'
    mpl.rcParams['grid.linewidth']= 0
    mpl.rcParams['axes.xmargin']=0.05
    mpl.rcParams['axes.ymargin']=0.05
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams['figure.titlesize'] = 18
    mpl.rcParams['figure.figsize'] = (16.18,10)
    mpl.rcParams['figure.autolayout'] = False
    mpl.rcParams['image.cmap'] = "viridis"
    mpl.rcParams['figure.dpi'] = 75
    mpl.rcParams['savefig.dpi'] = 150
    mpl.rcParams['errorbar.capsize'] = 3
    mpl.rcParams['axes.prop_cycle'] = plt.cycler(color = plt.cm.viridis(np.linspace(0.1,0.9,10)))

|#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Test
#///////////////////////////////////////////
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#****** Experimental data visualization
#/////////////////////////////////////////////////////////////////////

#   Dataset need to be predicted Et = -0.28 eV or 0.3 eV k=0.89 Sn*Nt = 3E-3 --> Sn ~ 3e-15
Ndop_exp = 5.1e15
Trange_exp_C = [-45.85,-21.35,2.65,28.25,47.35,71.15,94.75,118.15]
Trange_exp_K = [273.15+T for T in Trange_exp_C]
File_exp = "C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\04-Experimental data -ML\\ML\\data\\2019-05-27_first-defect-lifetime-table.csv"
DF_exp = pd.read_csv(File_exp)
Cell_exp=[cell(T,Ndop_exp,type='n') for T in Trange_exp_K]
Trange_keys = ["-50c","-25c","0c","30c","50c","75c","100c","125c"]

#   Store experimental data in dictionnary
Exp = {}
for key,c,T in zip(Trange_keys,Cell_exp,Trange_exp_K):
    line={
        "n0" : c.n0,
        "p0" : c.p0,
        "X_csv" : DF_exp["X-Y_"+key].dropna().values,
        "Dn_csv" : DF_exp["DN_"+key].dropna().values,
        "Tau_csv" : DF_exp["Tau_"+key].dropna().values,
        "T": T,
    }
    Exp[key]=line

#   Calculate evenly space Dn range for experimental data
Dn_range = np.logspace(13,17,100)
for key in Exp:
    xp =Exp[key]
    slope,intercept,_,_,_ = linregress(xp["X_csv"],xp["Tau_csv"])
    Exp[key]["Tau_calc"] = [slope*(xp["p0"]+dn)/(xp["n0"]+dn)+intercept for dn in Dn_range]

#   Check actual data vs extrapolated data
for i in range(1):
    plt.figure(figsize=(8.09,5))
    plt.title("Experimental data - extrapolated vs actual data")
    ax1 = plt.gca()
    ax1.set_xlabel('Excess carrier concentration ($cm^{-3}$)')
    ax1.set_ylabel('Lifetime ($\mu$s)')
    for key,c in zip(Exp,plt.cm.viridis(np.linspace(0.1,0.9,len(Exp)))):
        xp=Exp[key]
        ax1.plot(Dn_range,xp["Tau_calc"],c=c,label="T=%s - fit"%(key))
        ax1.scatter(xp["Dn_csv"],xp["Tau_csv"],c=c,label="T=%s - data"%(key))
    ax1.loglog()
    ax1.grid(which='minor',linewidth=0.5)
    ax1.grid(which='major',linewidth=1)
    #ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
    #plt.savefig('C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\00-TIDLS Simulation\\graphes\\2019-05-14\\Et-var-5T-k-100.png',transparent=True,bbox_inches='tight')
    ax1.legend(ncol=2)
    ax1.set_axisbelow(True)
    plt.minorticks_on()
    plt.show()

#   Feature vector to use in machine learning
Exp_feature = [t for key in Exp for t in Exp[key]["Tau_calc"]]

for i in range(1):
    Snrange = [0.00332547,0.00315966,0.00360459,0.00339791,0.0038242,0.00368272,0.00302364,0.00211294]
    Snrange = [sn*1e-12 for sn in Snrange]
    krange = [1.190001109,0.959997812,1.01000028,0.889000228,1.220000064,1.379998801,1.230001953,0.830999154]
    Sprange = [sn/k for (sn,k) in zip(Snrange,krange)]
    #k = Exp_k
    Et = 0.2835
    plt.figure(figsize=(16.18,10))
    ax1 = plt.gca()
    ax1.set_xlabel('$Excess\: carrier\: concentration\: (cm^{-3})$')
    ax1.set_ylabel('$Lifetime\: (\mu\:s)$')
    for key,c,Sn,k in zip(Exp,plt.cm.viridis(np.linspace(0.1,0.9,len(Exp))),Snrange,krange):
        xp=Exp[key]
        ax1.plot(Dn_range,simulation(cell(xp["T"],Ndop_exp,type='n'),defect(Et,Sn,Sn/k),Dn_range).tauSRH,c=c,label="T=%s - ML"%(key))
        ax1.scatter(xp["Dn_csv"],xp["Tau_csv"],c=c,label="T=%s - data"%(key))
    ax1.loglog()
    ax1.grid(which='minor',linewidth=0.5)
    ax1.grid(which='major',linewidth=1)
    #ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
    #plt.savefig(MLProcess.WORKDIR+"figures\\"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+"Predict-vs-actual.png",transparent=True,bbox_inches='tight')
    ax1.legend(ncol=2)
    ax1.set_axisbelow(True)
    plt.minorticks_on()
    plt.show()


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#****** Dataset simulation
#/////////////////////////////////////////////////////////////////////

N=100
Ndop=Ndop_exp
type = 'n'
db = defect.random_db(N,Et_min = -0.55, Et_max = 0.55, S_min = 1E-17, S_max = 1E-13, Nt = None)
Trange = Trange_exp_K
cref = cell(300, type=type, Ndop=Ndop)
cdb = [cref.changeT(T) for T in Trange]
dnrange = np.logspace(13,17,100)
columns_name = ["Name","Et_eV","Sn_cm2","Sp_cm2",'k']
Save = []
firstPass = True
noiseparam = 0
for d in db:
    col = [d.name,d.Et,d.Sn,d.Sp,d.Sn/d.Sp]
    for c in cdb:
        s = simulation(c,d,dnrange,noise="", noiseparam=noiseparam)    # No noise
        for t,dn in zip(s.tauSRH_noise,s.dnrange):
            if firstPass: columns_name.append("%s K _ %.0E cm-3" % (c.T,dn))
            col.append(t)
    Save.append(col)
    firstPass = False

df = pd.DataFrame(Save)
df.columns = columns_name
#outputpath = MLProcess.WORKDIR +"data\\"+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+"_SRH-data_N-%s_T-%s_Dn-%spts_type-%s_Ndop-%.0E.csv" %(N,Trange,len(dnrange),type,Ndop)
#df.to_csv(outputpath, encoding='utf-8', index=False)

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#****** Experimental prediction pipeline
#/////////////////////////////////////////////////////////////////////
datafile=MLProcess.WORKDIR +"data\\"+"2019-05-27-12-58_SRH-data_N-100000_T-[227.29999999999998, 251.79999999999998, 275.79999999999995, 301.4, 320.5, 344.29999999999995, 367.9, 391.29999999999995]_Dn-100pts_type-n_Ndop-5E+15.csv"
data = MLProcess.loadData(datafile, normalize = True)
loaded_scaler = MLProcess.SCALER

#   Bypass reload of data
MLProcess.SCALER = loaded_scaler
MLProcess.DATAFILE = datafile
MLProcess.NORMALIZE = True

#   Machine learning training
for i in range(1):
    subsetSize = 1000
    SAVE = False
    pipeline ={
        "Random Forest": RandomForestRegressor(n_estimators=100, verbose =2),
        "Ada Boost linear": AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators=100, loss='linear'),
        "Gradient Boost ls": GradientBoostingRegressor(verbose=2,loss='ls',max_depth=10),
        "Neural Network relu": MLPRegressor((100,100),alpha=0.001, activation = 'relu',verbose=2,learning_rate='adaptive'),
    }
    Exp_transform = MLProcess.SCALER["data"].transform([np.log10(Exp_feature)])

    Etp_results = {}
    for key in pipeline:
        Process = MLProcess(model=pipeline[key], name=key, save=SAVE)
        Process.initTraining()
        X,Y = Process.prepData(data, predictColumn="Et_eV", subsetSize=subsetSize, BG=1)
        Process.trainModel(X,Y)
        Process.regResults[0]['Etp_pred'] = MLProcess.SCALER["Et_eV"].inverse_transform([Process.model.predict(Exp_transform)])[0][0]
        Etp_results[key] = Process.regResults[0]

    resultsTab=[]
    title=[]
    for key,dics in Etp_results.items():
        if not resultsTab: title.append("Model")
        line=[]
        line.append(key)
        for k,v in dics.items():
            if not resultsTab: title.append(k)
            line.append(v)
        resultsTab.append(line)
    resEtp = pd.DataFrame(resultsTab)
    resEtp.columns=title
    resEtp.to_csv(MLProcess.WORKDIR+"2019-08-22_Etp_exp_results_2.csv",encoding='utf-8', index=False)


    Etm_results = {}
    for key in pipeline:
        Process = MLProcess(model=pipeline[key], name=key, save=SAVE)
        Process.initTraining()
        X,Y = Process.prepData(data, predictColumn="Et_eV", subsetSize=subsetSize, BG=0)
        Process.trainModel(X,Y)
        Process.regResults[0]['Etm_pred'] = MLProcess.SCALER["Et_eV"].inverse_transform([Process.model.predict(Exp_transform)])[0][0]
        Etm_results[key] = Process.regResults[0]

    resultsTab=[]
    title=[]
    for key,dics in Etm_results.items():
        if not resultsTab: title.append("Model")
        line=[]
        line.append(key)
        for k,v in dics.items():
            if not resultsTab: title.append(k)
            line.append(v)
        resultsTab.append(line)
    resEtm = pd.DataFrame(resultsTab)
    resEtm.columns=title
    resEtm.to_csv(MLProcess.WORKDIR+"2019-08-22_Etm_exp_results_2.csv",encoding='utf-8', index=False)


    k_results = {}
    for key in pipeline:
        Process = MLProcess(model=pipeline[key], name=key, save=SAVE)
        Process.initTraining()
        X,Y = Process.prepData(data, predictColumn="k", subsetSize=subsetSize, BG=None)
        Process.trainModel(X,Y)
        Process.regResults[0]['k_pred'] = np.power(10,MLProcess.SCALER["k"].inverse_transform([Process.model.predict(Exp_transform)]))[0][0]
        k_results[key] = Process.regResults[0]

    resultsTab=[]
    title=[]
    for key,dics in k_results.items():
        if not resultsTab: title.append("Model")
        line=[]
        line.append(key)
        for k,v in dics.items():
            if not resultsTab: title.append(k)
            line.append(v)
        resultsTab.append(line)
    resk = pd.DataFrame(resultsTab)
    resk.columns=title
    resk.to_csv(MLProcess.WORKDIR+"2019-08-22_k_exp_results_2.csv",encoding='utf-8', index=False)


    pipeline ={
        "Random Forest Classification": RandomForestClassifier(n_estimators=100, verbose =2),
        "Ada Boost Classification": AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators=100),
        "Gradient Boost Classification": GradientBoostingClassifier(verbose=2,loss='deviance'),
        "Neural Network relu": MLPClassifier((100,100),alpha=0.001, activation = 'relu',verbose=2,learning_rate='adaptive'),
        "5-neighbors classifier":KNeighborsClassifier(n_neighbors = 5, weights='distance'),
    }

    BG_results = {}
    for key in pipeline:
        Process = MLProcess(model=pipeline[key], name=key, save=SAVE)
        Process.initTraining()
        X,Y = Process.prepData(data, predictColumn="BG", subsetSize=subsetSize, BG=None)
        Process.trainModel(X,Y)
        Process.regResults[0]['BG_pred'] = (Process.model.predict(Exp_transform)[0],Process.model.predict_proba(Exp_transform)[0])
        BG_results[key] = Process.regResults[0]

    resultsTab=[]
    title=[]
    for key,dics in BG_results.items():
        if not resultsTab: title.append("Model")
        line=[]
        line.append(key)
        for k,v in dics.items():
            if not resultsTab: title.append(k)
            line.append(v)
        resultsTab.append(line)
    resBG = pd.DataFrame(resultsTab)
    resBG.columns=title
    resBG.to_csv(MLProcess.WORKDIR+"2019-08-22_BG_exp_results_2.csv",encoding='utf-8', index=False)

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#****** DPSS
#/////////////////////////////////////////////////////////////////////

dref = defect(0.285,1E-14,1E-14)
cref = cell(300, type='n', Ndop=5.1e15)
#Trange = [200,250,300,350,400]
Trange_exp_C = [-45.85,-21.35,2.65,28.25,47.35,71.15,94.75,118.15]
Trange_exp_K = [273.15+T for T in Trange_exp_C]
Trange = Trange_exp_K
cdb = [cref.changeT(T) for T in Trange]
dnrange=np.logspace(13,17,100)
sdb = [simulation(c,dref,dnrange,noise="logNorm", noiseparam=0.01) for c in cdb]
Etrange=np.arange(-0.55,0.56,0.01)
DPSS_Taun0 =[]
DPSS_k = []
for s in sdb:
    X = [1/(s.cell.n0+s.cell.p0+dn) for dn in dnrange]
    slope,intercept,_,_,_ = linregress(X,s.tauSRH_noise)
    dpss_Taun0 = [(slope-intercept*(s.cell.ni*np.exp(Et/(s.cell.kb*s.cell.T))-s.cell.p0))/(s.cell.ni*np.exp(-Et/(s.cell.kb*s.cell.T))+s.cell.p0-s.cell.ni*np.exp(Et/(s.cell.kb*s.cell.T))-s.cell.n0) for Et in Etrange]
    dpss_k = [s.cell.Vp/s.cell.Vn*(intercept/taun0-1) for taun0 in dpss_Taun0]
    DPSS_Taun0.append(dpss_Taun0)
    DPSS_k.append(dpss_k)


#   Predict back
for i in range(1):
    datafile = "C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\04-Experimental data -ML\\ML\\data\\2019-09-17-14-28_SRH-data_N-100000_T-[227.29999999999998, 251.79999999999998, 275.79999999999995, 301.4, 320.5, 344.29999999999995, 367.9, 391.29999999999995]_Dn-100pts_type-n_Ndop-5E+15.csv"
    data = MLProcess.loadData(datafile, normalize = True)
    loaded_scaler = MLProcess.SCALER

    EtScaler = preprocessing.MinMaxScaler()
    EtScaler.scale_=[0.90920781]
    EtScaler.min_ =[0.50006353]
    EtScaler.data_min_ = [-0.54999916]
    EtScaler.data_max_ = [0.5498594]
    EtScaler.data_range_ = [1.09985857]

    kScaler = preprocessing.MinMaxScaler()
    kScaler.scale_=[0.12530088]
    kScaler.min_ =[0.50037123]
    kScaler.data_min_ = [-3.99335779]
    kScaler.data_max_ = [3.98743241]
    kScaler.data_range_ = [7.9807902]

    dataScaler = loaded_scaler['data']

pipelineEtm ={
    "Random Forest": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_18-38__model_Random Forest_1_.sav"),
    "Ada Boost": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_18-44__model_Ada Boost linear_1_.sav"),
    "Gradient Boost": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_19-24__model_Gradient Boost ls_1_.sav"),
    "Neural Network": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_19-51__model_Neural Network relu_1_.sav"),
}
pipelineEtp ={
    "Random Forest": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_17-22__model_Random Forest_1_.sav"),
    "Ada Boost": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_17-26__model_Ada Boost linear_1_.sav"),
    "Gradient Boost": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_18-10__model_Gradient Boost ls_1_.sav"),
    "Neural Network": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_18-37__model_Neural Network relu_1_.sav"),
}
pipelinek ={
    "Random Forest": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_14-31__model_Random Forest_1_.sav"),
    "Ada Boost": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_14-39__model_Ada Boost linear_1_.sav"),
    "Gradient Boost": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_16-08__model_Gradient Boost ls_1_.sav"),
    "Neural Network": joblib.load(MLProcess.WORKDIR+"models\\2019-09-17_17-19__model_Neural Network relu_1_.sav"),
}
pipelineBG ={
    "Random Forest": joblib.load(MLProcess.WORKDIR+"models\\2019-08-26_19-33__model_Random Forest Classification_1_.sav"),
    "Ada Boost": joblib.load(MLProcess.WORKDIR+"models\\2019-08-26_19-38__model_Ada Boost Classification_1_.sav"),
    "Gradient Boost": joblib.load(MLProcess.WORKDIR+"models\\2019-08-26_19-40__model_Gradient Boost Classification_1_.sav"),
    "Neural Network": joblib.load(MLProcess.WORKDIR+"models\\2019-08-26_19-46__model_Neural Network relu_1_.sav"),
}

feature = [[t for s in sdb for t in s.tauSRH_noise]]
feature = dataScaler.transform(np.log10(feature))
res={}
for modelP,modelM in zip(pipelineEtm,pipelineEtp):
    res[modelP]={'Et':[EtScaler.inverse_transform([pipelineEtm[modelM].predict(feature)])[0][0],EtScaler.inverse_transform([pipelineEtp[modelP].predict(feature)])[0][0]]}
for model in pipelinek:
    res[model]['k']=[np.power(10,kScaler.inverse_transform([pipelinek[model].predict(feature)]))[0][0]]*2
for model in pipelineBG:
    res[model]['BG']=pipelineBG[model].predict_proba(feature)[0]


for i in range(1):  #   [CELL]  Plot DPSS curve
    fig, (ax2) = plt.subplots(figsize=(6,6))
    for k,T,c in zip(DPSS_k,Trange,plt.cm.viridis(np.linspace(0.1,0.9,len(Trange)))):
        ax2.plot(Etrange,k,label="T=%.0F K"%(T),c=c,alpha=0.5)
    markers =['o','v','s','D']
    colors =['b','r','green','brown']
    for key,m,c in zip(res,markers,colors):
        if res[key]['BG'][0]>res[key]['BG'][1]:
            pred=" +"
            prob = " %.2F"%(res[key]['BG'][0])
        else:
            pred=" -"
            prob = " %.2F"%(res[key]['BG'][1])
        ax2.errorbar(res[key]['Et'],res[key]['k'],xerr=[0,0],yerr=[0,0],
        marker=m,
        c=c,
        ms=5,
        alpha=0.8,
        linewidth=0,
        label=key+pred+prob
        )
    ax2.semilogy()
    ax2.set_xlabel('Defect energy level $\it{E_t}$ (eV)')
    ax2.set_ylabel('Capture cross-section ratio $\it{k}$')
    ax2.set_ylim(ymin=1e-1, ymax=1e1)
    ax2.set_xlim(xmin=-0.55, xmax=0.55)
    ax2.grid(which='minor',linewidth=0)
    ax2.grid(which='major',linewidth=0)
    ax2.locator_params(axis='x',nbins=6)
    ax2.tick_params(axis='x',direction='in', which='both',top=True)
    ax2.tick_params(axis='y',direction='in', which='both',right=True)
    ax2.minorticks_on()
    ax2.set_axisbelow(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

#   Find Et with MRL minimization method
def minEtfunc(Et,sdb):
    Equation ={
        "A":[],
        "B":[],
        "L":[],
        "E":[],
    }
    for s in sdb:
        X = [1/(s.cell.n0+s.cell.p0+dn) for dn in dnrange]
        slope,intercept,_,_,_ = linregress(X,s.tauSRH_noise)
        Equation['A'].append(slope)
        Equation['B'].append(intercept)
        Equation['L'].append([1/s.cell.Vn,1/s.cell.Vp])
        Equation['E'].append([(s.cell.ni*np.exp(-Et/(s.cell.kb*s.cell.T))-s.cell.n0)/s.cell.Vn,(s.cell.ni*np.exp(Et/(s.cell.kb*s.cell.T))-s.cell.p0)/s.cell.Vp])
    return Equation

custom_sdb=[]

for key,c in zip(Trange,Cell_exp):
    s = simulation(c,defect(0,1E-13,1E-13),dnrange)
    s.tauSRH_noise = Exp[key]["Tau_calc"]
    custom_sdb.append(s)
Etrange=np.arange(-0.55,0.551,0.001)
Norm_names = [np.inf,-np.inf]
Norms = {}
Norm_min = {}
for s in Norm_names: Norms[str(s)]=[]
for s in Norm_names: Norm_min[str(s)]=[np.inf,0]
custom_sdb=sdb['D-0454']
for Et in Etrange:
    Equation=minEtfunc(Et,custom_sdb)
    norm = Equation['E'] @ np.linalg.pinv(Equation['L']) @ Equation['B'] - Equation['A']
    for n in Norm_names:
        eq_norm = np.linalg.norm(norm,n)
        if eq_norm < Norm_min[str(n)][0]: Norm_min[str(n)]=[eq_norm,Et]
        Norms[str(n)].append(eq_norm)


for i in range(1):  #   [CELL]  Plot minimization curve
    fig, (ax2) = plt.subplots(figsize=(6,6))

    for n,c in zip(Norms,plt.cm.viridis(np.linspace(0.1,0.9,len(Norms)))):
        ax2.plot(Etrange,Norms[n], label="Norm %s - minEt = %.3F eV [%.2E]"%(n,Norm_min[n][1],Norm_min[n][0]), c=c)

    ax2.semilogy()
    ax2.set_xlabel('Defect energy level $\it{E_t}$ (eV)')
    ax2.set_ylabel('Normal equation norm')
    #ax2.set_ylim(ymin=1e-1, ymax=1e1)
    ax2.set_xlim(xmin=-0.55, xmax=0.55)
    ax2.grid(which='minor',linewidth=0)
    ax2.grid(which='major',linewidth=0)
    ax2.locator_params(axis='x',nbins=6)
    ax2.tick_params(axis='x',direction='in', which='both',top=True)
    ax2.tick_params(axis='y',direction='in', which='both',right=True)
    ax2.minorticks_on()
    ax2.set_axisbelow(True)
    ax2.legend(bbox_to_anchor=(0.5, -.15), loc="upper center", borderaxespad=0.)
    plt.show()

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#****** Noise accuracy of DPSS/MRL/ML
#/////////////////////////////////////////////////////////////////////
MLProcess.WORKDIR="C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\03-ML-prediction\\2019-T1\\2019-04-12_MLProcess\\"

#   Load Scaler
for i in range(1):
    datafileBG = "C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\03-ML-prediction\\2019-T1\\2019-02-21_DPML algorithms comparison\\data\\2019-03-22-12-20_SRH-data_N-500000_T-[200, 250, 300, 350, 400]_Dn-100pts_type-p_Ndop-1E+15.csv"
    dataBG = MLProcess.loadData(datafileBG, normalize = True)
    BG_EtScaler = MLProcess.SCALER['Et_eV']
    BG_kScaler = MLProcess.SCALER['k']
    BG_dataScaler = MLProcess.SCALER['data']

#   Load Models

ML = joblib.load(MLProcess.WORKDIR+"models\\2019-04-26_18-21__model_Neural Network relu_4_.sav")

#   Generate random noise dataset
N=10000
ddb = defect.random_db(N,Et_min = -0.55, Et_max = 0.55, S_min = 1e-17, S_max = 1e-13, Nt = 1e12)
Trange = [200,250,300,350,400]
dnrange=np.logspace(13,17,100)
Etrange=np.arange(-0.55,0.551,0.001)
cref = cell(300, type='p', Ndop=1e15)
cdb = [cref.changeT(T) for T in Trange]
noiseparam=0.01
sdb = {}
for d in ddb:
    sdb[d.name]=[simulation(c,d,dnrange,noise="logNorm", noiseparam=noiseparam) for c in cdb]
def minEtfunc(Et,sdb):
    Equation ={
        "A":[],
        "B":[],
        "L":[],
        "E":[],
    }
    for s in sdb:
        X = [1/(s.cell.n0+s.cell.p0+dn) for dn in dnrange]
        slope,intercept,_,_,_ = linregress(X,s.tauSRH_noise)
        Equation['A'].append(slope)
        Equation['B'].append(intercept)
        Equation['L'].append([1/s.cell.Vn,1/s.cell.Vp])
        Equation['E'].append([(s.cell.ni*np.exp(-Et/(s.cell.kb*s.cell.T))-s.cell.n0)/s.cell.Vn,(s.cell.ni*np.exp(Et/(s.cell.kb*s.cell.T))-s.cell.p0)/s.cell.Vp])
    return Equation

Result = {}
for key,simu in sdb.items():
    Result[key]= {'Actual': 0 if simu[0].defect.Et<0 else 1 }

    #   DPSS solver
    DPSS_Taun0 =[]
    DPSS_k = []
    for s in simu:
        X = [1/(s.cell.n0+s.cell.p0+dn) for dn in dnrange]
        slope,intercept,_,_,_ = linregress(X,s.tauSRH_noise)
        dpss_Taun0 = [(slope-intercept*(s.cell.ni*np.exp(Et/(s.cell.kb*s.cell.T))-s.cell.p0))/(s.cell.ni*np.exp(-Et/(s.cell.kb*s.cell.T))+s.cell.p0-s.cell.ni*np.exp(Et/(s.cell.kb*s.cell.T))-s.cell.n0) for Et in Etrange]
        dpss_k = [s.cell.Vp/s.cell.Vn*(intercept/taun0-1) for taun0 in dpss_Taun0]
        dpss_k = np.array(dpss_k)
        dpss_k[dpss_k<0]='NaN'
        DPSS_Taun0.append(dpss_Taun0)
        DPSS_k.append(dpss_k)
    err_k_tab=[]
    for i in range(len(Etrange)):
        err_k = []
        for j in range(len(Trange)):
            err_k.append(DPSS_k[j][i])
        err_k_tab.append(err_k)
    DPSS_L = []
    DPSS_minEt = [-0.55,np.infty]
    for i in range(len(Etrange)):
        mean = np.nanmean(err_k_tab[i])
        min = np.nanmin(err_k_tab[i])
        max = np.nanmax(err_k_tab[i])
        l = np.log(max/min)
        if l ==0: l = 'NaN'
        DPSS_L.append(l)
        if l =='NaN': continue
        if l< DPSS_minEt[1]:
            DPSS_minEt[0]=Etrange[i]
            DPSS_minEt[1]=l
    Result[key]['DPSS']= 0 if DPSS_minEt[0]<0 else 1
    # ML
    extractfeature = [[t for s in simu for t in s.tauSRH_noise]]
    featureBG = BG_dataScaler.transform(np.log10(extractfeature))
    Result[key]['ML']= ML.predict(featureBG)[0]
    # MRL inf
    Norm_inf_min=np.inf
    Norm_Et_min = -0.55
    for Et in Etrange:
        Equation=minEtfunc(Et,simu)
        norm = Equation['E'] @ np.linalg.pinv(Equation['L']) @ Equation['B'] - Equation['A']
        eq_norm = np.linalg.norm(norm,np.inf)
        if eq_norm< Norm_inf_min:
            Norm_inf_min = eq_norm
            Norm_Et_min = Et
    Result[key]['MRL']= 0 if Norm_Et_min<0 else 1

column_names = ["Et","k","Actual","DPSS","ML","MRL"]
line=[]
for key in Result:
    line.append((sdb[key][0].defect.Et,sdb[key][0].defect.k,Result[key]['Actual'],Result[key]['DPSS'],Result[key]['ML'],Result[key]['MRL']))
df = pd.DataFrame(line)
df.columns = column_names
df.to_csv("C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\05-ML-noise\\"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+"DPSS-ML-MRL-comparison-10000.csv",encoding='utf-8', index=False)

FigDF.describe()
FigDF = pd.read_csv("C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\05-ML-noise\\2019-11-20-10-21DPSS-ML-MRL-comparison-10000.csv", index_col=None)
FigDF['Correct DPSS']= [ 1 if dpss==actual else 0 for dpss,actual in zip(FigDF.DPSS,FigDF.Actual)]
FigDF['Correct MRL']= [ 1 if mrl==actual else 0 for mrl,actual in zip(FigDF.MRL,FigDF.Actual)]
FigDF['Correct ML']= [ 1 if nl==actual else 0 for nl,actual in zip(FigDF.ML,FigDF.Actual)]

for i in range(1):  #   [CELL]  Plot
    fig = plt.figure(figsize=(6,4.5))
    #plt.title("Defect energy level variation lifetime curve",fontsize=16)
    ax1 = plt.gca()
    ax1.set_xlabel('Defect energy level $\it{E_t}$ (eV)')
    ax1.set_ylabel('Capture cross-section ratio $\it{k}$')
    #ax1.annotate("$\it{T} =$ 400 K",xy=(0.05,0.9),xycoords='axes fraction', fontsize=14)
    ax1.scatter(FigDF.Et,FigDF.k,c=FigDF['Correct MRL'],marker=".",alpha=1,s=9)
    # ax1.plot([-0.55,0.55],[0.83,0.83],"k--")
    ax1.semilogy()
    ax1.set_xlim(xmin=-0.55,xmax=0.55)
    ax1.set_ylim(ymin=0.0001,ymax=10000)
    ax1.grid(which='minor',linewidth=0)
    ax1.grid(which='major',linewidth=0)
    ax1.minorticks_on()
    ax1.locator_params(axis='x',nbins=7)
    ax1.tick_params(axis='x',direction='in', which='both',top=True)
    ax1.tick_params(axis='y',direction='in', which='both',right=True)
    ax1.set_axisbelow(True)
    # #ax1.set_facecolor('#F2F2F2')
    # divider = make_axes_locatable(ax1)
    # cax1 = divider.append_axes("right", size="5%", pad="1%")
    # s=Fraction(0.3,AxesX(cax1))
    # cbar=fig.colorbar(sc, ax=ax1)
    # cbar.set_label("Correct label prediction probability")
    #cbar.set_ticks([0.2,0.49,1])
    plt.show()
