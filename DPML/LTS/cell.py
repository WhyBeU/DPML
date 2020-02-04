"Cell related functions"
from semiconductor.recombination import SRH
from ..utils import NoStdStreams
import scipy.constants as const
import numpy as np
import warnings


class Cell():
    #****   Constant declaration    ****#
    kb = const.physical_constants['Boltzmann constant in eV/K'][0]
    Tmin = 75
    Tmax = 400
    Ndopmin = 1e10
    Ndopmax = 1e20
    DefaultNdop = 1E15
    DefaultType = "p"
    DefaultT=300

    #****   Method declaration      ****#
    def __init__(self,T=None,Ndop=None,type=None,):
        '''
        ---Doc---
            Description:
                Validates Temperature range (Tmin-Tmax), and cell-type, defaulting to p-type
                Calculates Bang Gap Narrowing intrinsic carrier ni
                Calculates thermal velocity vn, vp
                Calculate n0,p0

                Calculation are made using MK8J semiconductor package :
                https://github.com/MK8J/semiconductor

                Models used:
                    - Band Gap Narrowing : Yan, 2014
                    - Intrinsici carrier : Couderc 2014 doi- 10.1063/1.4867776
                    - Thermal velocity : Green 1990 doi- 10.1063/1.345414
                    - Bandgap : Passler 2002 doi- 10.1103/PhysRevB.66.085201

            Inputs:
                T       Float       Temperature of the cell in Kelvin
                Ndop    Float       Doping level of the sample
                type    String      Cell type. Have "n" or "p"

            Outputs:
                object  represents defined cell

            Exemple:
                >>  Cell(T=300, Ndop = 1E15, type="p")
        '''
        if Ndop is None: Ndop = Cell.DefaultNdop
        if type is None: type = Cell.DefaultType
        if T is None: T = Cell.DefaultT

        #   Check dopping range
        if Ndop<Cell.Ndopmin:
            self.Ndop = Cell.Ndopmin
            warnings.warn("In Cell.__init__ : Ndop value out of bound, got %s, and expected within [%s,%s]. Et will be clipped to %s." %(Ndop,Cell.Ndopmin,Cell.Ndopmax,Cell.Ndopmin))
        elif Ndop>Cell.Ndopmax:
            self.Ndop = Cell.Ndopmax
            warnings.warn("In Cell.__init__ : Ndop value out of bound, got %s, and expected within [%s,%s]. Et will be clipped to %s." %(Ndop,Cell.Ndopmin,Cell.Ndopmax,Cell.Ndopmax))
        else:
            self.Ndop = Ndop

        #   Check temperature range
        if T<Cell.Tmin:
            self.T = Cell.Tmin
            warnings.warn("In Cell.__init__ : T value out of bound, got %s, and expected within [%s,%s]. Et will be clipped to %s." %(T,Cell.Tmin,Cell.Tmax,Cell.Tmin))
        elif T>Cell.Tmax:
            self.T = Cell.Tmax
            warnings.warn("In Cell.__init__ : T value out of bound, got %s, and expected within [%s,%s]. Et will be clipped to %s." %(T,Cell.Tmin,Cell.Tmax,Cell.Tmax))
        else:
            self.T = T

        #   Check type validity
        if type not in ["p","n"]:
            warnings.warn("In Cell.__init__ : Incorrect type value, expected n or p. Value will be defaulted to %s"%(Cell.DefaultType))
            self.type = Cell.DefaultType
        else:
            self.type = type

        #   Calculate Ni, Vn using semiconductor package
        with NoStdStreams():
            if type == "p": Tmodel=SRH(material="Si",temp = self.T, Nt = 1, nxc = 1e15, Na = self.Ndop, Nd= 0, BGN_author = "Yan_2014fer")
            if type == "n": Tmodel=SRH(material="Si",temp = self.T, Nt = 1, nxc = 1e15, Na = 0, Nd= self.Ndop, BGN_author = "Yan_2013fer")
        self.ni = Tmodel.nieff[0]
        self.Vn = Tmodel.vel_th_e[0]
        self.Vp = Tmodel.vel_th_h
        #Calculate n0 and p0
        if type == "p":
            self.p0 = (0.5 * (np.abs(Ndop - 0) + np.sqrt((0 - Ndop)**2 + 4 * self.ni**2)))
            self.n0 = (self.ni**2)/self.p0
        if type == "n":
            self.n0 = (0.5 * (np.abs(Ndop - 0) + np.sqrt((0 - Ndop)**2 + 4 * self.ni**2)))
            self.p0 = (self.ni**2)/self.n0
    def changeT(self,T):
        '''
        ---Doc---
            Description:
                Create a new cell instance, based on the same parameters as .self except for the new inputted temperature.

            Inputs:
                T       Float       Temperature of the cell in Kelvin

            Outputs:
                object  represents defined cell

            Exemple:
                >>  cell2 = Cell(T = 300, Ndop = 1E15, type="p").changeT(350)
        '''
        return Cell(T,self.Ndop,self.type)
    def changeNdop(self,T):
        '''
        ---Doc---
            Description:
                Create a new cell instance, based on the same parameters as .self except for the new inputted doping.

            Inputs:
                Ndop       Float       Doping level of the sample

            Outputs:
                object  represents defined cell

            Exemple:
                >>  cell2 = Cell(T = 300, Ndop = 1e15, type="p").changeNdop(1e16)
        '''
        return Cell(T,self.Ndop,self.type)
