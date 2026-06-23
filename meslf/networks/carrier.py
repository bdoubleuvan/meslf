"""Carrier base class"""

# ===========================================================================
class Carrier():
    """Carrier class"""
    def __init__(self,name):
        """Creates a Carrier object
        
        Parameters
        ----------
        name : string
            name of the carrier
        """
        self.name = name
        
# ===========================================================================
class Gas(Carrier):
    """Gas carrier class"""
    def __init__(self,name,S,R_air,Z,pn,Tn,T,mu=None):
        """Creates a Gas object
        
        Parameters
        ----------
        name : string
            name of the carrier
        S : float
            specific gravitiy of the gas
        R_air : float
            specific constan of air in Nm/kgK (=J/kgK)
        Z : float
            compressibility factor of the gas
        pn : float
            standard pressure in N/m^2 (=Pa)
        Tn : float
            standard temperature in K
        T : float
            temperature of the gas in K
        mu : float, optional
            Kinematic viscosity of gas in m^2/s. Default is None
        """
        super().__init__(name)
        self.S = S
        self.R_air = R_air
        self.Z = Z
        self.pn = pn
        self.Tn = Tn
        self.T = T
        self.mu = mu
        
    @property
    def S(self):
        """getter of _S
        """
        return self._S
    
    @S.setter
    def S(self,S):
        """setter of _S
        """
        self._S = S
        
    @property
    def R_air(self):
        """getter of _R_air
        """
        return self._R_air
    
    @R_air.setter
    def R_air(self,R_air):
        """setter of _R_air
        """
        self._R_air = R_air
        
    @property
    def Z(self):
        """getter of _Z
        """
        return self._Z
    
    @Z.setter
    def Z(self,Z):
        """setter of _Z
        """
        self._Z = Z
        
    @property
    def pn(self):
        """getter of _pn
        """
        return self._pn
    
    @pn.setter
    def pn(self,pn):
        """setter of _pn
        """
        self._pn = pn
        
    @property
    def Tn(self):
        """getter of _Tn
        """
        return self._Tn
    
    @Tn.setter
    def Tn(self,Tn):
        """setter of _Tn
        """
        self._Tn = Tn
        
    @property
    def T(self):
        """getter of _T
        """
        return self._T
    
    @T.setter
    def T(self,T):
        """setter of _T
        """
        self._T = T
    
    @property
    def mu(self):
        """getter of _mu
        """
        return self._mu
    
    @mu.setter
    def mu(self,mu):
        """setter of _mu
        """
        self._mu = mu
        
    @property
    def rhon(self):
        """getter of rhon
        """
        return (self.pn*self.S)/(self.R_air*self.Tn)
    
# ===========================================================================
class Water(Carrier):
    """Water carrier class"""
    def __init__(self,name,Cp,rho=None,mu=None,g=9.81):
        """Creates a Water object
        
        Parameters
        ----------
        name : string
            name of the carrier
        Cp : float
            specific heat of the carrier in J/(kgK)
        rho : float, optional
            Density of carrier in kg/m^3. Default is None
        mu : float, optional
            Viscosity of carrier in m^2/s. Default is None
        g : float, optional
            Gravitational constant in m/s^2. Default is 9.81
        """
        super().__init__(name)
        self.Cp = Cp
        self.rhon = rho
        self.mu = mu
        self.g = g
        
    @property
    def Cp(self):
        """getter of _Cp
        """
        return self._Cp
    
    @Cp.setter
    def Cp(self,Cp):
        """setter of _Cp
        """
        self._Cp = Cp
        
    def get_Cp(self,scale_var=None,scale_var_params=None):
        """Get specific heat of carrier, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        Cp : float
            Specific heat of the carrier, possibly scaled. When unscaled in J/(kgK).
        """
        Cp = self.Cp
        if scale_var == 'per_unit':
            Cb = scale_var_params['phibase']/(scale_var_params['qbase']*scale_var_params['Tbase'])
            Cp = Cp/Cb
        return Cp
    
    @property
    def rhon(self):
        """getter of _rhon
        """
        return self._rhon
    
    @rhon.setter
    def rhon(self,rhon):
        """setter of _rho
        """
        self._rhon = rhon
        
    @property
    def mu(self):
        """getter of _mu
        """
        return self._mu
    
    @mu.setter
    def mu(self,mu):
        """setter of _mu
        """
        self._mu = mu
    
    @property
    def g(self):
        """getter of _g
        """
        return self._g
    
    @g.setter
    def g(self,g):
        """setter of _g
        """
        self._g = g
    
