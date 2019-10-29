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
