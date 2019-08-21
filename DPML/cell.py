from semiconductor.recombination import SRH
from DPML.Nostdstreams import NoStdStreams

class cell():
    #****   Constant declaration    ****#
    kb = const.physical_constants['Boltzmann constant in eV/K'][0]
    Tmin =75
    Tmax =400
    DefaultNdop = 1E15
    DefaultType = "p"
    #****   Method declaration      ****#
    def __init__(self,T,Ndop=None,type=None,):
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
                >>  cell(300, Ndop = 1E15, type="p")
        '''
        if Ndop is None: Ndop = cell.DefaultNdop
        if type is None: type = cell.DefaultType
        self.Ndop = Ndop
        #   Check temperature range
        if T<cell.Tmin:
            self.T = cell.Tmin
            warnings.warn("In cell.__init__ : T value out of bound, got %s, and expected within [%s,%s]. Et will be clipped to %s." %(T,cell.Tmin,cell.Tmax,cell.Tmin))
        elif T>cell.Tmax:
            self.T = cell.Tmax
            warnings.warn("In cell.__init__ : T value out of bound, got %s, and expected within [%s,%s]. Et will be clipped to %s." %(T,cell.Tmin,cell.Tmax,cell.Tmax))
        else:
            self.T = T

        #   Check type validity
        if type not in ["p","n"]:
            warnings.warn("In cell.__init__ : Incorrect type value, expected n or p. Value will be defaulted to %s"%(cell.DefaultType))
            self.type = cell.DefaultType
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
                >>  cell2 = cell(300, Ndop = 1E15, type="p").changeT(350)
        '''
        return cell(T,self.Ndop,self.type)
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
