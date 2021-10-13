import numpy as np
import warnings
from decimal import Decimal

class LTS():
    #****   Constant declaration    ****#
    N0_eeh=3.3e17   #cm-3
    N0_ehh =7.0e17 #cm-3
    B_temp = [77,90,112,170,195,249,300]    #K
    B_values = [8.01e-14,4.57e-14,2.14e-14,8.84e-15,7.35e-15,5.48e-15,4.73e-15] #cm3/s
    #****   Method declaration      ****#
    def __init__(self,cell,defect,dnrange,noise="", noiseparam=0):
        '''
        ---Doc---
            Description:
                Defines the boundary of the lifetime simulation, using the cell and defect parameters to calculate SRH lifetime.
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
                >> LTS(cell(300,1E15,"p"),defect(0.33,1E-15,4E-14,1E12),[1E-13,1E-15,1E-17], noise="log-gaussian")
        '''
        self.cell = cell
        self.defect = defect
        self.dnrange = dnrange
        self.n1 = float(Decimal(self.cell.ni)*np.exp(Decimal(self.defect.Et/(self.cell.kb*self.cell.T))))
        self.p1 = float(Decimal(self.cell.ni)*np.exp(Decimal(-self.defect.Et/(self.cell.kb*self.cell.T))))
        self.taun0 = float(1/(Decimal(self.defect.Sn) * Decimal(self.cell.Vn)*Decimal(self.defect.Nt)))
        self.taup0 = float(1/(Decimal(self.defect.Sp) * Decimal(self.cell.Vp)*Decimal(self.defect.Nt)))
        self.tauSRH = [float((Decimal(self.taun0)*(Decimal(self.cell.p0)+Decimal(self.p1)+Decimal(dn))+Decimal(self.taup0)*(Decimal(self.cell.n0)+Decimal(self.n1)+Decimal(dn)))/(Decimal(self.cell.n0)+Decimal(self.cell.p0)+Decimal(dn))) for dn in self.dnrange]
        # self.n1 = self.cell.ni*np.exp(self.defect.Et/(self.cell.kb*self.cell.T))
        # self.p1 = self.cell.ni*np.exp(-self.defect.Et/(self.cell.kb*self.cell.T))
        # self.taun0 = 1/(self.defect.Sn * self.cell.Vn*self.defect.Nt)
        # self.taup0 = 1/(self.defect.Sp * self.cell.Vp*self.defect.Nt)
        # self.tauSRH = [(self.taun0*(self.cell.p0+self.p1+dn)+self.taup0*(self.cell.n0+self.n1+dn))/(self.cell.n0+self.cell.p0+dn) for dn in self.dnrange]

        #   Add noise
        self.noisemodel=noise
        self.noiseparam=noiseparam
        if self.noisemodel=="logNorm":
            noise = np.random.normal(0,noiseparam,len(dnrange))
            self.tauSRH_noise = [float(Decimal(t)*(Decimal(1)+Decimal(e*np.log(np.max(self.dnrange)/dn)))) for t,e,dn in zip(self.tauSRH,noise,self.dnrange)]
            # self.tauSRH_noise = [t*1+e*np.log(np.max(self.dnrange)/dn) for t,e,dn in zip(self.tauSRH,noise,self.dnrange)]
            self.noisemodel="logNorm"
            self.noiseparam=noiseparam
        else:
            self.tauSRH_noise=self.tauSRH
    def checkAuger(self):
        '''
        ---Doc---
            Description:
                Return a boolean to check if generated lifetime is smaller than Auger lifetime and a clip lifetime to Auger if needed
                Auger Constants are defined from Richter et al [2012] -DOI 10.1103/PhysRevB.86.165202
                Radiative contants from Altermat et al [2005] - DOI 10.1109/NUSOD.2005.1518128 and Trupke et al [2003] - DOI 10.1063/1.1610231

            Inputs:

            Outputs:
                breakAuger  boolean     True if lifetime is higher than Auger limit at any carrier concentration
                tauAuger    array       Auger lifetime array, the length of self.tauSRH
        '''
        g_eeh = 1+13*(1-np.tanh(np.power(self.cell.n0/LTS.N0_eeh,0.66)))
        g_ehh = 1+13*(1-np.tanh(np.power(self.cell.n0/LTS.N0_ehh,0.63)))
        tauAuger = np.zeros(len(self.tauSRH))
        bmin = 0.20+(0.00-0.20)/(1+np.power(self.cell.T/320,2.5))
        b1 = 1.5e18+(1e7-1.5e18)/(1+np.power(self.cell.T/550,3))
        b3 = 4e18+(1e9-4e18)/(1+np.power(self.cell.T/365,3.54))
        for k in range(len(LTS.B_temp)):
            if self.cell.T < LTS.B_temp[k]:
                if k==0: B_low = B_V[0]
                B_low=(LTS.B_values[k]-LTS.B_values[k-1])/(LTS.B_temp[k]-LTS.B_temp[k-1])*(self.cell.T-LTS.B_temp[k-1])+LTS.B_values[k-1]
            else:
                if k==len(LTS.B_temp)-1: B_low = LTS.B_values[-1]
        breakAuger=False
        for i in range(len(self.dnrange)):
            dn = self.dnrange[i]
            fact1 = (self.cell.n0+dn)*(self.cell.p0+dn)-self.cell.ni**2
            fact2 = 2.5e-31*g_eeh*self.cell.n0+8.5e-32*g_ehh*self.cell.p0+3e-29*np.power(dn,0.92)
            factB = B_low*(bmin+(1-bmin)/(1+np.power((self.cell.n0+dn+self.cell.p0+dn)/b1,0.54)+np.power((self.cell.n0+dn+self.cell.p0+dn)/b3,1.25)))
            tauAuger[i]=dn/(fact1*(fact2+factB))
            if tauAuger[i]<self.tauSRH[i]: breakAuger=True

        return(breakAuger,tauAuger)
    def __repr__(self):
        return f'{self.__class__.__name__}('f'{self.cell!r}, {self.defect!r}, dnrange([{np.min(self.dnrange)},{np.max(self.dnrange)}]))'
