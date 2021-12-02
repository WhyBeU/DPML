'Defect parameters and generation functions'
import numpy as np
import warnings

class Defect():
    #****   Constant declaration    ****#
    Etbound = 0.6       #   Absolute bound for Defect level
    Sminbound = 1E-20   #   Absolute minimum bound for Capture Cross Section
    Smaxbound = 1E-10   #   Absolute maximum bound for Capture Cross Section
    DefaultNt = 1E12    #   Default value for Defect density if none is provided
    MPE_Einf_min = 0.01    # 0.00  Minimum value for Multiphonon emission activation energy [Rein]
    MPE_Einf_max = 0.25 #   0.056 -Maximum value for Multiphonon emission activation energy [Rein]
    CAS_Alpha_min = 1.1    #   Minimum value for Cascade power law [Rein]
    CAS_Alpha_max = 4    #   Maximum value for Cascade power law [Rein]
    Capture_modes = ['Radiative','Multiphonon emission','Cascade']
    DefaultCM = ['Radiative']
    kB=8.617e-5 #eV/K
    T0=300 #K
    #****   Method declaration      ****#
    def __init__(self,Et,Sn,Sp,Nt=None, name="", Capture_mode_n=None,Capture_mode_p=None, Capture_param_n=None, Capture_param_p=None):
        '''
        ---Doc---
            Description:
                Take Defect parameters as inputs.
                Note that Et is defined as Et-Ei with Ei = Eg/2 and must therefore be between -Defect.Etbound and Defect.Etbound
                if outside the range, will clip to the side and throw a warning.
                Electron and hole capture-cross section are also bound between Sminbount and Smaxbound and will be clipped if exceeded.
                Calculates An and Ap: An = 1/(Sn*Nt) ;; Ap = 1/(Sp*Nt)
                Calculate k = Sn/Sp = Ap/An
                Add a modal capture mechanism, applying a temperature dependence to the capture-cross-section

            Inputs:
                Et  Float       Defect energy level, relative to the intrinsic mid-gap
                Sn  Float       Capture-cross section for electrons
                Sp  Float       Capture-cross section for holes
                Nt  Float       Defect density
                Capture_mode_n  String      Defines which capture mode for Sn
                Capture_mode_p  String      Defines which capture mode for Sp
                Capture_param_n   Float       Parameter for capture mechanism for Sn
                Capture_param_p   Float       Parameter for capture mechanism for Sp

            Outputs:
                object  represents defined Defect
        '''
        if Nt is None : Nt = Defect.DefaultNt
        if Capture_mode_n is None: self.CMn=Defect.DefaultCM[0]
        if Capture_mode_p is None: self.CMp=Defect.DefaultCM[0]
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

        if Capture_mode_n not in self.Capture_modes: warnings.warn("In Defect.__init__ : Incorrect capture mode for Sn. Got %s and expeceted in %s" %(Capture_mode_n,Defect.Capture_modes))
        if Capture_mode_p not in self.Capture_modes: warnings.warn("In Defect.__init__ : Incorrect capture mode for Sp. Got %s and expeceted in %s" %(Capture_mode_p,Defect.Capture_modes))
        if Capture_mode_n == 'Radiative': self.CMn=Capture_mode_n ; self.CPn = np.nan
        if Capture_mode_p == 'Radiative': self.CMp=Capture_mode_p ; self.CPp = np.nan
        if Capture_mode_n == 'Multiphonon emission':
            self.CMn = Capture_mode_n
            if Capture_param_n is None:
                self.CPn = Defect.MPE_Einf_min
                warnings.warn("In Defect.__init__ :Capture_param_n was None. expected within [%s,%s]. Capture_param_n will be clipped to %s." %(Defect.MPE_Einf_min,Defect.MPE_Einf_max,Defect.MPE_Einf_min))
            if Capture_param_n<self.MPE_Einf_min:
                self.CPn = Defect.MPE_Einf_min
                warnings.warn("In Defect.__init__ :Capture_param_n value out of bound, got %s, and expected within [%s,%s]. Capture_param_n will be clipped to %s." %(Capture_param_n,Defect.MPE_Einf_min,Defect.MPE_Einf_max,Defect.MPE_Einf_min))
            elif Capture_param_n>self.MPE_Einf_max:
                self.CPn = Defect.MPE_Einf_max
                warnings.warn("In Defect.__init__ :Capture_param_n value out of bound, got %s, and expected within [%s,%s]. Capture_param_n will be clipped to %s." %(Capture_param_n,Defect.MPE_Einf_min,Defect.MPE_Einf_max,Defect.MPE_Einf_min))
            else:
                self.CPn = Capture_param_n
            self.Sn = self.Sn*np.exp(self.CPn/(Defect.kB*Defect.T0))
        if Capture_mode_p == 'Multiphonon emission':
            self.CMp = Capture_mode_p
            if Capture_param_p is None:
                self.CPp = Defect.MPE_Einf_min
                warnings.warn("In Defect.__init__ :Capture_param_p was None. expected within [%s,%s]. Capture_param_p will be clipped to %s." %(Defect.MPE_Einf_min,Defect.MPE_Einf_max,Defect.MPE_Einf_min))
            if Capture_param_p<self.MPE_Einf_min:
                self.CPp = Defect.MPE_Einf_min
                warnings.warn("In Defect.__init__ :Capture_param_p value out of bound, got %s, and expected within [%s,%s]. Capture_param_p will be clipped to %s." %(Capture_param_p,Defect.MPE_Einf_min,Defect.MPE_Einf_max,Defect.MPE_Einf_min))
            elif Capture_param_p>self.MPE_Einf_max:
                self.CPp = Defect.MPE_Einf_max
                warnings.warn("In Defect.__init__ :Capture_param_p value out of bound, got %s, and expected within [%s,%s]. Capture_param_p will be clipped to %s." %(Capture_param_p,Defect.MPE_Einf_min,Defect.MPE_Einf_max,Defect.MPE_Einf_min))
            else:
                self.CPp = Capture_param_p
            self.Sp = self.Sp*np.exp(self.CPp/(Defect.kB*Defect.T0))
        if Capture_mode_n == 'Cascade':
            self.CMn = Capture_mode_n
            if Capture_param_n is None:
                self.CPn = Defect.CAS_Alpha_min
                warnings.warn("In Defect.__init__ :Capture_param_n was None. expected within [%s,%s]. Capture_param_n will be clipped to %s." %(Defect.MPE_Einf_min,Defect.MPE_Einf_max,Defect.MPE_Einf_min))
            if Capture_param_n<self.CAS_Alpha_min:
                self.CPn = Defect.CAS_Alpha_min
                warnings.warn("In Defect.__init__ :Capture_param_n value out of bound, got %s, and expected within [%s,%s]. Capture_param_n will be clipped to %s." %(Capture_param_n,Defect.CAS_Alpha_min,Defect.CAS_Alpha_max,Defect.CAS_Alpha_min))
            elif Capture_param_n>self.CAS_Alpha_max:
                self.CPn = Defect.CAS_Alpha_max
                warnings.warn("In Defect.__init__ :Capture_param_n value out of bound, got %s, and expected within [%s,%s]. Capture_param_n will be clipped to %s." %(Capture_param_n,Defect.CAS_Alpha_min,Defect.CAS_Alpha_max,Defect.CAS_Alpha_min))
            else:
                self.CPn = Capture_param_n
            self.Sn = self.Sn*Defect.T0**self.CPn
        if Capture_mode_p == 'Cascade':
            self.CMp = Capture_mode_p
            if Capture_param_p is None:
                self.CPp = Defect.CAS_Alpha_min
                warnings.warn("In Defect.__init__ :Capture_param_p was None. expected within [%s,%s]. Capture_param_p will be clipped to %s." %(Defect.MPE_Einf_min,Defect.MPE_Einf_max,Defect.MPE_Einf_min))
            if Capture_param_p<self.CAS_Alpha_min:
                self.CPp = Defect.CAS_Alpha_min
                warnings.warn("In Defect.__init__ :Capture_param_p value out of bound, got %s, and expected within [%s,%s]. Capture_param_p will be clipped to %s." %(Capture_param_p,Defect.CAS_Alpha_min,Defect.CAS_Alpha_max,Defect.CAS_Alpha_min))
            elif Capture_param_p>self.CAS_Alpha_max:
                self.CPp = Defect.CAS_Alpha_max
                warnings.warn("In Defect.__init__ :Capture_param_p value out of bound, got %s, and expected within [%s,%s]. Capture_param_p will be clipped to %s." %(Capture_param_p,Defect.CAS_Alpha_min,Defect.CAS_Alpha_max,Defect.CAS_Alpha_min))
            else:
                self.CPp = Capture_param_p
            self.Sp = self.Sp*Defect.T0**self.CPp
        self.Nt=Nt
        self.An=1/(self.Sn*Nt)
        self.Ap=1/(self.Sp*Nt)
        self.k=self.Sn/self.Sp
        self.name=name
    def randomDB(N,Et_min = None, Et_max = None, S_min = None, S_max = None, Nt = None,CMn_tab=None,CMp_tab=None,Force_same_CM=False):
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
                CMn_tab     array       List of possible capture modes for n
                CMp_tab     array       List of possible capture modes for p
                Force_same_CM   Bool    If True, both n and p will have the same capture modes
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
        if CMn_tab is None: CMn_tab = Defect.DefaultCM
        if CMp_tab is None: CMp_tab = Defect.DefaultCM
        Res=[]
        for i in range(N):
            Et = np.random.rand(1)[0]*(Et_max-Et_min)+Et_min
            Sn = np.exp(np.random.rand(1)[0]*(np.log(S_max)-np.log(S_min))+np.log(S_min))
            Sp= np.exp(np.random.rand(1)[0]*(np.log(S_max)-np.log(S_min))+np.log(S_min))
            name = "D-"+str(i).zfill(1+int(np.trunc(np.log10(N)))) #    Will display leading '0' for naming ordering purposes
            Capture_param_n = None
            Capture_param_p = None
            Capture_mode_n = np.random.choice(CMn_tab)
            if Force_same_CM:
                Capture_mode_p = Capture_mode_n
            else:
                Capture_mode_p = np.random.choice(CMp_tab)
            if Capture_mode_n == 'Multiphonon emission': Capture_param_n = np.random.rand(1)[0]*(Defect.MPE_Einf_max-Defect.MPE_Einf_min)+Defect.MPE_Einf_min
            if Capture_mode_p == 'Multiphonon emission': Capture_param_p = np.random.rand(1)[0]*(Defect.MPE_Einf_max-Defect.MPE_Einf_min)+Defect.MPE_Einf_min
            if Capture_mode_n == 'Cascade': Capture_param_n = np.random.rand(1)[0]*(Defect.CAS_Alpha_max-Defect.CAS_Alpha_min)+Defect.CAS_Alpha_min
            if Capture_mode_p == 'Cascade': Capture_param_p = np.random.rand(1)[0]*(Defect.CAS_Alpha_max-Defect.CAS_Alpha_min)+Defect.CAS_Alpha_min
            d = Defect(Et, Sn, Sp, Nt=Nt,name=name,Capture_mode_n=Capture_mode_n,Capture_mode_p=Capture_mode_p,Capture_param_n=Capture_param_n,Capture_param_p=Capture_param_p)
            Res.append(d)
        return Res
    def __repr__(self):
        return f'{self.__class__.__name__}('f'{self.Et!r}, {self.Sn!r}, {self.Sp!r})'
    def copy(self):
        dref = Defect(
            Et=self.Et,
            Sn=self.Sn,
            Sp=self.Sp,
            Nt=self.Nt,
            name=self.name,
            Capture_mode_n=self.CMn,
            Capture_mode_p=self.CMp,
            Capture_param_n=self.CPn,
            Capture_param_p=self.CPp
        )
        return(dref)
