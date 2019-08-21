class defect():
    #****   Constant declaration    ****#
    Etbound = 0.6       #   Absolute bound for Defect level
    Sminbound = 1E-18   #   Absolute minimum bound for Capture Cross Section
    Smaxbound = 1E-12   #   Absolute maximum bound for Capture Cross Section
    DefaultNt = 1E12    #   Default value for defect density if none is provided
    #****   Method declaration      ****#
    def __init__(self,Et,Sn,Sp,Nt=None, name=""):
        '''
        ---Doc---
            Description:
                Take defect parameters as inputs.
                Note that Et is defined as Et-Ei with Ei = Eg/2 and must therefore be between -defect.Etbound and defect.Etbound
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
                object  represents defined defect

            Exemple:
                >>  mydefect=defect(0.33,1E-14,3E-15, 1E12)
        '''
        if Nt is None : Nt = defect.DefaultNt
        if Et<-defect.Etbound:
            self.Et = -defect.Etbound
            warnings.warn("In defect.__init__ : Et value out of bound, got %s, and expected within [%s,%s]. Et will be clipped to %s." %(Et,-defect.Etbound,defect.Etbound,-defect.Etbound))
        elif Et>defect.Etbound:
            self.Et = defect.Etbound
            warnings.warn("In defect.__init__ : Et value out of bound, got %s, and expected within [%s,%s]. Et will be clipped to %s." %(Et,-defect.Etbound,defect.Etbound,defect.Etbound))
        else:
            self.Et=Et

        if Sn<defect.Sminbound:
            self.Sn = defect.Sminbound
            warnings.warn("In defect.__init__ : Sn value out of bound, got %s, and expected within [%s,%s]. Sn will be clipped to %s." %(Sn,defect.Sminbound,defect.Smaxbound,defect.Sminbound))
        elif Sn>defect.Smaxbound:
            self.Sn = defect.Smaxbound
            warnings.warn("In defect.__init__ : Sn value out of bound, got %s, and expected within [%s,%s]. Sn will be clipped to %s." %(Sn,defect.Sminbound,defect.Smaxbound,defect.Sminbound))
        else:
            self.Sn=Sn

        if Sp<defect.Sminbound:
            self.Sp = defect.Sminbound
            warnings.warn("In defect.__init__ : Sp value out of bound, got %s, and expected within [%s,%s]. Sp will be clipped to %s." %(Sp,defect.Sminbound,defect.Smaxbound,defect.Sminbound))
        elif Sp>defect.Smaxbound:
            self.Sp = defect.Smaxbound
            warnings.warn("In defect.__init__ : Sp value out of bound, got %s, and expected within [%s,%s]. Sp will be clipped to %s." %(Sp,defect.Sminbound,defect.Smaxbound,defect.Sminbound))
        else:
            self.Sp=Sp

        self.Nt=Nt
        self.An=1/(Sn*Nt)
        self.Ap=1/(Sp*Nt)
        self.k=Sn/Sp
        self.name=name
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
    def random_db(N,Et_min = None, Et_max = None, S_min = None, S_max = None, Nt = None):
        '''
        ---Doc---
            Description:
                Generates N defects objects bound by the imputted limits and store them in an array.
            Inputs:
                N           Integer     Number of defects to generate in the database
                Et_min      Float       Lower boundary for Defect Energy level
                Et_max      Float       Higher boundary for defect Energy level
                S_min       Float       Lower boundary for Capture Cross-Section
                S_max       Float       Higher boundary for Capture Cross-Section
                Nt          Float       Density of defects
            Outputs:
                Res         array       Database of defects
            Exemple:
                >>  defect.random_db(100)
        '''
        #   If values are not define, then default thermal
        if Et_min is None: Et_min = - defect.Etbound
        if Et_max is None: Et_max =  defect.Etbound
        if S_min is None: S_min =  defect.Sminbound
        if S_max is None: S_max =  defect.Smaxbound
        if Nt is None: Nt = defect.DefaultNt
        Res=[]
        for i in range(N):
            Et = np.random.rand(1)[0]*(Et_max-Et_min)+Et_min
            Sn = np.exp(np.random.rand(1)[0]*(np.log(S_max)-np.log(S_min))+np.log(S_min))
            Sp= np.exp(np.random.rand(1)[0]*(np.log(S_max)-np.log(S_min))+np.log(S_min))
            d = defect(Et, Sn, Sp, Nt=Nt,name="D-"+str(i).zfill(1+int(np.trunc(np.log10(N))))) #    Will display leading '0' for naming ordering purposes
            Res.append(d)
        return Res
