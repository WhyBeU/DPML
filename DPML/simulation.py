import numpy as np
import warnings

class simulation():
    #****   Constant declaration    ****#

    #****   Method declaration      ****#
    def __init__(self,cell,defect,dnrange,noise="", noiseparam=0):
        '''
        ---Doc---
            Description:
                Defines the boundary of the simulation, using the clel and defect parameters to calculate SRH lifetime.
                It is spossible to add noise to the lifetime, according to different noise models.

            Inputs:
                cell        object      Cell object previously created
                defect      object      Defect object previously created
                dnrange     array       Excess carrier concentration values at which to simulate data
                noise       String      Define which noise model to used
                noiseparam  float       hyper-parameter for noise level

            Outputs:
                Object

            Exemple:
                >> simulation(cell(300,1E15,"p"),defect(0.33,1E-15,4E-14,1E12),[1E-13,1E-15,1E-17], noise="log-gaussian")
        '''
        self.cell = cell
        self.defect = defect
        self.dnrange = dnrange
        self.n1 = self.cell.ni*np.exp(self.defect.Et/(self.cell.kb*self.cell.T))
        self.p1 = self.cell.ni*np.exp(-self.defect.Et/(self.cell.kb*self.cell.T))
        self.taun0 = 1/(self.defect.Sn * self.cell.Vn*self.defect.Nt)
        self.taup0 = 1/(self.defect.Sp * self.cell.Vp*self.defect.Nt)
        self.tauSRH = [(self.taun0*(self.cell.p0+self.p1+dn)+self.taup0*(self.cell.n0+self.n1+dn))/(self.cell.n0+self.cell.p0+dn) for dn in self.dnrange]

        #   Add noise
        self.noisemodel=noise
        self.noiseparam=noiseparam
        if self.noisemodel=="logNorm":
            noise = np.random.normal(0,noiseparam,len(dnrange))
            self.tauSRH_noise = [t*(1+e*np.log(np.max(self.dnrange)/dn)) for t,e,dn in zip(self.tauSRH,noise,self.dnrange)]
            self.noisemodel="logNorm"
            self.noiseparam=noiseparam
        else:
            self.tauSRH_noise=self.tauSRH
    def print_self(self):
        '''
        ---Doc---
            Description:
                Prints to the console the values stored in the object.

            Inputs:
                None

            Outputs:
                None

            Exemple:
                >>  mydefect.print_self()
        '''
        for attr in dir(self):
            if not attr.startswith("__"):
                value = getattr(self, attr)
                if not callable(value):
                    print(str(attr)+" : "+str(value))
