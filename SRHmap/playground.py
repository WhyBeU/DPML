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
# %%-

# %%--  Create one defect
parameters={
    'temperature':[200+T*10 for T in range(20)],
    'dn_range':np.logspace(13,17,10),
}

d = Defect(-0.33,1E-16,1E-16)
c = Cell(T=300,Ndop=1E15,type='p')
SRHmap = []
for T in parameters['temperature']:
    lts = LTS(c.changeT(T),d,parameters['dn_range'])
    SRHmap.append(np.log10(lts.tauSRH))


SRHmapImg = cv2.normalize(np.float32(SRHmap), None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX)
SRHmapImg = cv2.resize(SRHmapImg, (32,32))

plt.imshow(SRHmapImg)

# %%-

# %%--  Create multiple defect image
parameters={
    'temperature':[200+T*10 for T in range(21)],
    'dn_range':np.logspace(13,17,100),
}


db = Defect.randomDB(N=10)
c = Cell(T=300,Ndop=1E15,type='p')
for d in db:
    SRHmap = []
    for T in parameters['temperature']:
        lts = LTS(c.changeT(T),d,parameters['dn_range'])
        SRHmap.append(np.log10(lts.tauSRH))


    SRHmapImg = cv2.normalize(np.float32(SRHmap), None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX)
    SRHmapImg = cv2.resize(SRHmapImg, (32,32))

    plt.imshow(SRHmapImg)
    plt.title("%.2F ; %.2F"%(d.Et, np.log10(d.k)))
    plt.show()

# %%-

# %%--  Generate database and save data and csv with target
#   Add global normalization in destination range
#   restrict Et, k in center value
#   Start with 32 by 32 images

# %%-
