#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
%reload_ext autoreload
%autoreload 2
from DPML import *
import numpy as np
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Genarating images
#///////////////////////////////////////////
# %%-- Extra Imports
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import datetime
# %%-

# %%--  Create one defect
parameters={
    # 'temperature':[200,250,300,350,400],
    'temperature':[200+T*10 for T in range(20)],
    'dn_range':np.logspace(13,17,11),
}

d = Defect(0.33,1E-14,1E-15)
c = Cell(T=300,Ndop=1E15,type='p')
SRHmap = []
for T in parameters['temperature']:
    lts = LTS(c.changeT(T),d,parameters['dn_range'],noise=parameters['noise_model'], noiseparam=parameters['noise_parameter'])
    SRHmap.append(np.log10(lts.tauSRH_noise))

SRHmapNorm = [[(256*(tau-parameters['log_tau_min'])/(parameters['log_tau_max']-parameters['log_tau_min'])) for tau in Tline] for Tline in SRHmap]
SRHmapImg = cv2.resize(SRHmapImg, (32,32))
cv2.imwrite('test.png',SRHmapImg)
map = cv2.imread('test.png')

plt.imshow(SRHmap)
plt.imshow(map)
# %%-

# %%--  Create multiple defect image
parameters={
    'temperature':[200+T*10 for T in range(21)],
    'dn_range':np.logspace(13,17,100),
}


db = Defect.randomDB(N=1)
c = Cell(T=300,Ndop=1E15,type='p')
for d in db:
    SRHmap = []
    for T in parameters['temperature']:
        lts = LTS(c.changeT(T),d,parameters['dn_range'])
        SRHmap.append(np.log10(lts.tauSRH))


    SRHmapImg = cv2.normalize(np.float32(SRHmap), None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX)
    SRHmapImg = cv2.resize(np.float32(SRHmapImg), (128,128))

    plt.imshow(SRHmap)
    plt.title("%.2F ; %.2F"%(d.Et, np.log10(d.k)))
    plt.show()


# %%-

# %%--  Generate database and save data and csv with target
start = datetime.datetime.now()
SAVEDIR = "C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\07-Github-SRHmap\\dataset\\128-fix-local\\"
parameters={
    'temperature':[200+T*200/128 for T in range(128)],
    'dn_range':np.logspace(13,17,128),
    'type':'p',
    'log_tau_max':-1,   #   Max Tau for normalization
    'log_tau_min':-7,   #   Min tau for normalization
    'Et_min':-0.55,  #   Minimum energy level
    'Et_max':0.55,  #   maximum energy level
    'S_min':1E-18,   #   minimum capture cross section
    'S_max':1E-12,  #   maximum capture cross section
    'Nt':1E12,  #   maximum energy level
    'noise_model':"",  #   Type of noise in SRH generation
    'noise_parameter':0, #Parameter used to vary noise level from noise model
}
db = Defect.randomDB(N=10000,
                Et_min = parameters['Et_min'],
                Et_max = parameters['Et_max'],
                S_min = parameters['S_min'],
                S_max = parameters['S_max'],
                Nt = parameters['Nt']
                )
c = Cell(T=300,Ndop=1E15,type=parameters['type'])
i=1
columns_name = ["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap','path']
imageDB=[]
for d in db:
    print(i, " out of ", len(db))
    SRHmap = []
    bandgap = 1 if d.Et>0 else 0
    col = [d.name,d.Et,d.Sn,d.Sp,d.k, np.log10(d.Sn),np.log10(d.Sp),np.log10(d.k),bandgap]

    #   Generate map
    for T in parameters['temperature']:
        lts = LTS(c.changeT(T),d,parameters['dn_range'],noise=parameters['noise_model'], noiseparam=parameters['noise_parameter'])
        SRHmap.append(np.log10(lts.tauSRH))

    #   Check if map is within bound
    if np.max(SRHmap)>parameters['log_tau_max'] or np.min(SRHmap)<parameters['log_tau_min']:
        newDefect = Defect.randomDB(
                N=1,
                Et_min = parameters['Et_min'],
                Et_max = parameters['Et_max'],
                S_min = parameters['S_min'],
                S_max = parameters['S_max'],
                Nt = parameters['Nt']
                )[0]
        newDefect.name = d.name
        db.append(newDefect)
        continue

    #   Upload data and make filename
    filename = SAVEDIR+"images\\"+d.name+".png"
    col.append(filename)
    i+=1
    imageDB.append(col)

    #   Normalize array
    SRHmapNorm = [[(256*(tau-np.min(SRHmap))/(np.max(SRHmap)-np.min(SRHmap))) for tau in Tline] for Tline in SRHmap]
    SRHmapImg = cv2.resize(np.float32(SRHmapNorm), (128,128))
    cv2.imwrite(filename,SRHmapImg)

imageDB = pd.DataFrame(imageDB)
imageDB.columns = columns_name
imageDB.to_csv(SAVEDIR+"data.csv",encoding='utf-8', index=False)

finish = datetime.datetime.now()
print((finish-start))
    # map = cv2.imread('test.png')
    # plt.imshow(map)
    # plt.imshow(SRHmapImg)
    # plt.imshow(SRHmapNorm)
    # plt.title("%.2F ; %.2F; %.2F"%(d.Et, np.log10(d.k),np.mean(SRHmapNorm)))
    # plt.show()
# %%-

# %%--  Generate noisy database and save data and csv with target
start = datetime.datetime.now()
SAVEDIR = "C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\07-Github-SRHmap\\dataset\\128-fix-local-noisy-100\\"
parameters={
    'temperature':[200+T*200/128 for T in range(128)],
    'dn_range':np.logspace(13,17,128),
    'type':'p',
    'log_tau_max':-1,   #   Max Tau for normalization
    'log_tau_min':-7,   #   Min tau for normalization
    'Et_min':-0.55,  #   Minimum energy level
    'Et_max':0.55,  #   maximum energy level
    'S_min':1E-18,   #   minimum capture cross section
    'S_max':1E-12,  #   maximum capture cross section
    'Nt':1E12,  #   maximum energy level
    'noise_model':"logNorm",  #   Type of noise in SRH generation
    'noise_parameter':0.1, #Parameter used to vary noise level from noise model
}
db = Defect.randomDB(N=1000,
                Et_min = parameters['Et_min'],
                Et_max = parameters['Et_max'],
                S_min = parameters['S_min'],
                S_max = parameters['S_max'],
                Nt = parameters['Nt']
                )
c = Cell(T=300,Ndop=1E15,type=parameters['type'])
i=1
columns_name = ["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap','path']
imageDB=[]
for d in db:
    print(i, " out of ", len(db))
    SRHmap = []
    bandgap = 1 if d.Et>0 else 0
    col = [d.name,d.Et,d.Sn,d.Sp,d.k, np.log10(d.Sn),np.log10(d.Sp),np.log10(d.k),bandgap]

    #   Generate map
    for T in parameters['temperature']:
        lts = LTS(c.changeT(T),d,parameters['dn_range'],noise=parameters['noise_model'], noiseparam=parameters['noise_parameter'])
        SRHmap.append(np.log10(lts.tauSRH_noise))

    #   Check if map is within bound
    if np.max(SRHmap)>parameters['log_tau_max'] or np.min(SRHmap)<parameters['log_tau_min']:
        newDefect = Defect.randomDB(
                N=1,
                Et_min = parameters['Et_min'],
                Et_max = parameters['Et_max'],
                S_min = parameters['S_min'],
                S_max = parameters['S_max'],
                Nt = parameters['Nt']
                )[0]
        newDefect.name = d.name
        db.append(newDefect)
        continue

    #   Upload data and make filename
    filename = SAVEDIR+"images\\"+d.name+".png"
    col.append(filename)
    i+=1
    imageDB.append(col)

    #   Normalize array
    SRHmapNorm = [[(256*(tau-np.min(SRHmap))/(np.max(SRHmap)-np.min(SRHmap))) for tau in Tline] for Tline in SRHmap]
    SRHmapImg = cv2.resize(np.float32(SRHmapNorm), (128,128))
    cv2.imwrite(filename,SRHmapImg)

imageDB = pd.DataFrame(imageDB)
imageDB.columns = columns_name
imageDB.to_csv(SAVEDIR+"data.csv",encoding='utf-8', index=False)

finish = datetime.datetime.now()
print((finish-start))
    # map = cv2.imread('test.png')
    # plt.imshow(map)
    # plt.imshow(SRHmapImg)
    # plt.imshow(SRHmapNorm)
    # plt.title("%.2F ; %.2F; %.2F"%(d.Et, np.log10(d.k),np.mean(SRHmapNorm)))
    # plt.show()
# %%-
