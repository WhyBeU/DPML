'Defect parameters and generation functions'
import numpy as np
import warnings

class Defect():
    #****   Constant declaration    ****#
    Etbound = 0.6       #   Absolute bound for Defect level
    Sminbound = 1E-20   #   Absolute minimum bound for Capture Cross Section
    Smaxbound = 1E-10   #   Absolute maximum bound for Capture Cross Section
    DefaultNt = 1E12    #   Default value for Defect density if none is provided

    #****   Method declaration      ****#
    def __init__(self,Et,Sn,Sp,Nt=None, name=""):
        '''
        ---Doc---
            Description:
                Take Defect parameters as inputs.
                Note that Et is defined as Et-Ei with Ei = Eg/2 and must therefore be between -Defect.Etbound and Defect.Etbound
                if outside the range, will clip to the side and throw a warning.
                Electron and hole capture-cross section are also bound between Sminbount and Smaxbound and will be clipped if exceeded.
                Calculates An and Ap: An = 1/(Sn*Nt) ;; Ap = 1/(Sp*Nt)
                Calculate k = Sn/Sp = Ap/An

            Inputs:
                Et  Float       Defect energy level, relative to the intrinsic mid-gap
                Sn  Float       Capture-cross section for electrons
                Sp  Float       Capture-cross section for holes
                Nt  Float       Defect density

            Outputs:
                object  represents defined Defect

            Exemple:
                >>  myDefect=Defect(0.33,1E-14,3E-15, 1E12)
        '''
        if Nt is None : Nt = Defect.DefaultNt
        if Et<-Defect.Etbound:
            self.Et = -Defect.Etbound
            warnings.warn("In Defect.__init__ : Et value out of bound, got %s, and expected within [%s,%s]. Et will be clipped to %s." %(Et,-Defect.Etbound,Defect.Etbound,-Defect.Etbound))
        elif Et>Defect.Etbound:
            self.Et = Defect.Etbound
            warnings.warn("In Defect.__init__ : Et value out of bound, got %s, and expected within [%s,%s]. Et will be clipped to %s." %(Et,-Defect.Etbound,Defect.Etbound,Defect.Etbound))
        else:
            self.Et=Et

        if Sn<Defect.Sminbound:
            self.Sn = Defect.Sminbound
            warnings.warn("In Defect.__init__ : Sn value out of bound, got %s, and expected within [%s,%s]. Sn will be clipped to %s." %(Sn,Defect.Sminbound,Defect.Smaxbound,Defect.Sminbound))
        elif Sn>Defect.Smaxbound:
            self.Sn = Defect.Smaxbound
            warnings.warn("In Defect.__init__ : Sn value out of bound, got %s, and expected within [%s,%s]. Sn will be clipped to %s." %(Sn,Defect.Sminbound,Defect.Smaxbound,Defect.Sminbound))
        else:
            self.Sn=Sn

        if Sp<Defect.Sminbound:
            self.Sp = Defect.Sminbound
            warnings.warn("In Defect.__init__ : Sp value out of bound, got %s, and expected within [%s,%s]. Sp will be clipped to %s." %(Sp,Defect.Sminbound,Defect.Smaxbound,Defect.Sminbound))
        elif Sp>Defect.Smaxbound:
            self.Sp = Defect.Smaxbound
            warnings.warn("In Defect.__init__ : Sp value out of bound, got %s, and expected within [%s,%s]. Sp will be clipped to %s." %(Sp,Defect.Sminbound,Defect.Smaxbound,Defect.Sminbound))
        else:
            self.Sp=Sp

        self.Nt=Nt
        self.An=1/(Sn*Nt)
        self.Ap=1/(Sp*Nt)
        self.k=Sn/Sp
        self.name=name
    def randomDB(N,Et_min = None, Et_max = None, S_min = None, S_max = None, Nt = None):
        '''
        ---Doc---
            Description:
                Generates N Defects objects bound by the imputted limits and store them in an array.
            Inputs:
                N           Integer     Number of Defects to generate in the database
                Et_min      Float       Lower boundary for Defect Energy level
                Et_max      Float       Higher boundary for Defect Energy level
                S_min       Float       Lower boundary for Capture Cross-Section
                S_max       Float       Higher boundary for Capture Cross-Section
                Nt          Float       Density of Defects
            Outputs:
                Res         array       Database of Defects
            Exemple:
                >>  Defect.random_db(100)
        '''
        #   If values are not define, then default thermal
        if Et_min is None: Et_min = - Defect.Etbound
        if Et_max is None: Et_max =  Defect.Etbound
        if S_min is None: S_min =  Defect.Sminbound
        if S_max is None: S_max =  Defect.Smaxbound
        if Nt is None: Nt = Defect.DefaultNt
        Res=[]
        for i in range(N):
            Et = np.random.rand(1)[0]*(Et_max-Et_min)+Et_min
            Sn = np.exp(np.random.rand(1)[0]*(np.log(S_max)-np.log(S_min))+np.log(S_min))
            Sp= np.exp(np.random.rand(1)[0]*(np.log(S_max)-np.log(S_min))+np.log(S_min))
            d = Defect(Et, Sn, Sp, Nt=Nt,name="D-"+str(i).zfill(1+int(np.trunc(np.log10(N))))) #    Will display leading '0' for naming ordering purposes
            Res.append(d)
        return Res
