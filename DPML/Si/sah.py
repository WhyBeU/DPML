import numpy as np
import warnings
from decimal import Decimal

class Sah():
    #****   Constant declaration    ****#
    N0_eeh=3.3e17   #cm-3
    N0_ehh =7.0e17 #cm-3
    B_temp = [77,90,112,170,195,249,300]    #K
    B_values = [8.01e-14,4.57e-14,2.14e-14,8.84e-15,7.35e-15,5.48e-15,4.73e-15] #cm3/s
    #****   Method declaration      ****#
    def __init__(self,c,d1,d2,dnrange):
        '''
        ---Doc---
            Description:
                Defines the boundary of the lifetime simulation, using the cell and defect parameters to calculate Sah-Shockley lifetime.

            Inputs:
                c        object      Cell object previously created
                d1      object      Defect level 1 object previously created
                d2      object      Defect level 2 object previously created
                dnrange     array       Excess carrier concentration values at which to simulate data

            Outputs:
                Object
        '''
        #   Check for applicability
        if d1.Nt != d2.Nt:
            warnings.warn("In Sah.__init__ : defects Nt are different : d1.Nt [%s] d2.Nt [%s].d2.Nt will be changed to d1.Nt." %(d1.Nt,d2.Nt))
            d2.Nt = d1.Nt

        self.cell = c
        self.defect_1 = d1
        self.defect_2 = d2
        self.dnrange = dnrange
        self.n1_1 = float(Decimal(c.ni)*np.exp(Decimal(d1.Et/(c.kb*c.T))))
        self.p1_1 = float(Decimal(c.ni)*np.exp(Decimal(-d1.Et/(c.kb*c.T))))
        self.taun0_1 = float(1/(Decimal(d1.Sn)*Decimal(c.Vn)*Decimal(d1.Nt)))
        self.taup0_1 = float(1/(Decimal(d1.Sp)*Decimal(c.Vp)*Decimal(d1.Nt)))
        self.n1_2 = float(Decimal(c.ni)*np.exp(Decimal(d2.Et/(c.kb*c.T))))
        self.p1_2 = float(Decimal(c.ni)*np.exp(Decimal(-d2.Et/(c.kb*c.T))))
        self.taun0_2 = float(1/(Decimal(d2.Sn)*Decimal(c.Vn)*Decimal(d2.Nt)))
        self.taup0_2 = float(1/(Decimal(d2.Sp)*Decimal(c.Vp)*Decimal(d2.Nt)))
        self.tauSah = [float((1+(Decimal(c.p0)+Decimal(dn)+Decimal(d1.k*self.n1_1))/(Decimal(self.p1_1)+Decimal(d1.k)*(Decimal(c.n0)+Decimal(dn)))+(Decimal(self.p1_2)+Decimal(d2.k)*(Decimal(c.n0)+Decimal(dn)))/(Decimal(c.p0)+Decimal(dn)+Decimal(d2.k*self.n1_2)))/((Decimal(c.n0)+Decimal(c.p0)+Decimal(dn))*(1/(Decimal(self.taun0_1*self.p1_1)+Decimal(self.taup0_1)*(Decimal(c.n0)+Decimal(dn)))+1/(Decimal(self.taun0_2)*(Decimal(c.p0)+Decimal(dn))+Decimal(self.taup0_2)*Decimal(self.n1_2))))) for dn in dnrange]
        # self.n1_1 = c.ni*np.exp(d1.Et/(c.kb*c.T))
        # self.p1_1 = c.ni*np.exp(-d1.Et/(c.kb*c.T))
        # self.taun0_1 = 1/(d1.Sn * c.Vn*d1.Nt)
        # self.taup0_1 = 1/(d1.Sp * c.Vp*d1.Nt)
        # self.n1_2 = c.ni*np.exp(d2.Et/(c.kb*c.T))
        # self.p1_2 = c.ni*np.exp(-d2.Et/(c.kb*c.T))
        # self.taun0_2 = 1/(d2.Sn * c.Vn*d2.Nt)
        # self.taup0_2 = 1/(d2.Sp * c.Vp*d2.Nt)
        # self.tauSah = [(1+(c.p0+dn+d1.k*self.n1_1)/(self.p1_1+d1.k*(c.n0+dn))+(self.p1_2+d2.k*(c.n0+dn))/(c.p0+dn+d2.k*self.n1_2))/((c.n0+c.p0+dn)*(1/(self.taun0_1*self.p1_1+self.taup0_1*(c.n0+dn))+1/(self.taun0_2*(c.p0+dn)+self.taup0_2*self.n1_2))) for dn in dnrange]
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
        g_eeh = 1+13*(1-np.tanh(np.power(self.cell.n0/Sah.N0_eeh,0.66)))
        g_ehh = 1+13*(1-np.tanh(np.power(self.cell.n0/Sah.N0_ehh,0.63)))
        tauAuger = np.zeros(len(self.tauSah))
        bmin = 0.20+(0.00-0.20)/(1+np.power(self.cell.T/320,2.5))
        b1 = 1.5e18+(1e7-1.5e18)/(1+np.power(self.cell.T/550,3))
        b3 = 4e18+(1e9-4e18)/(1+np.power(self.cell.T/365,3.54))
        for k in range(len(Sah.B_temp)):
            if self.cell.T < Sah.B_temp[k]:
                if k==0: B_low = B_V[0]
                B_low=(Sah.B_values[k]-Sah.B_values[k-1])/(Sah.B_temp[k]-Sah.B_temp[k-1])*(self.cell.T-Sah.B_temp[k-1])+Sah.B_values[k-1]
            else:
                if k==len(Sah.B_temp)-1: B_low = Sah.B_values[-1]
        breakAuger=False
        for i in range(len(self.dnrange)):
            dn = self.dnrange[i]
            fact1 = (self.cell.n0+dn)*(self.cell.p0+dn)-self.cell.ni**2
            fact2 = 2.5e-31*g_eeh*self.cell.n0+8.5e-32*g_ehh*self.cell.p0+3e-29*np.power(dn,0.92)
            factB = B_low*(bmin+(1-bmin)/(1+np.power((self.cell.n0+dn+self.cell.p0+dn)/b1,0.54)+np.power((self.cell.n0+dn+self.cell.p0+dn)/b3,1.25)))
            tauAuger[i]=dn/(fact1*(fact2+factB))
            if tauAuger[i]<self.tauSah[i]: breakAuger=True

        return(breakAuger,tauAuger)
    def __repr__(self):
        return f'{self.__class__.__name__}('f'{self.cell!r}, {self.defect_1!r}, {self.defect_2!r}, dnrange([{np.min(self.dnrange)},{np.max(self.dnrange)}]))'
