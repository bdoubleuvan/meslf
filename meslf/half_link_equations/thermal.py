"""Thermal half link equations, such as heat exchangers"""

# ===========================================================================
def dummy():
    """Creates all link equations needed for a dummy half link (i.e, a half link without a model).
    
    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the half link
        
    Returns
    --------
    m_of_phi : function
        Flow through the half link as a function of heat power (and return and outflow temperature)
    phi_of_m : function
        Heat power absorbed by half link as a functin of flow (and return and outflow temperature)
    dm_dTs : function
        Derivative of flow to the nodal supply temperature
    dm_dTr : function
        Derivative of flow to the nodal return temperature
    """    
    def m_of_phi(phi,Ts,Tr,scale_var=None,scale_var_params=None):
        """Half link flow as a function of heat power difference, and half links temperatures
        
        Parameters
        ----------
        phi : float
            Heat power difference
        Ts : float
            Supply temperature
        Tr : float
            Return temperature
        
        Returns
        -------
        m : float
            Link flow in kg/s
        """
        pass
    
    def phi_of_m(m,Ts,Tr,scale_var=None,scale_var_params=None):
        """Half link heat power difference as a function of flow, and half links temperatures
        
        Parameters
        ----------
        m : float
            Link flow in kg/s
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line
        
        Returns
        -------
        phi : float
            Heat power difference
        """
        pass
    
    def dm_dTs(phi,Tr,Ts,scale_var=None,scale_var_params=None):
        """Derivative of half link flow to half link temperature near supply line
        
        Parameters
        ----------
        phi : float
            Heat power
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line
        
        Returns
        -------
        dm_dTs : float
            Derivative of half link flow to supply temperature
        """
        return 0. 
    
    def dm_dTr(phi,Ts,Tr,scale_var=None,scale_var_params=None):
        """Derivative of half link flow to half link temperature near return line

        Parameters
        ----------
        phi : float
            Heat power
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line

        Returns
        -------
        dm_dTr : float
            Derivative of half link flow to supply temperature
        """
        return 0.
    
    def dphi_dm(m,Ts,Tr,scale_var=None,scale_var_params=None):
        """Derivative of half link heat power difference to flow.
        
        Parameters
        ----------
        m : float
            Link flow in kg/s
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line
        
        Returns
        -------
        dphi_dm : float
            Derivative of half link heat power to flow
        """
        return 0.
    
    def dphi_dTs(m,Tr,Ts,scale_var=None,scale_var_params=None):
        """Derivative of half link heat power difference to half link temperature near supply line
        
        Parameters
        ----------
        m : float
            Link flow in kg/s
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line
        
        Returns
        -------
        dphi_dTs : float
            Derivative of half link flow to supply temperature
        """
        return 0. 
    
    def dphi_dTr(m,Ts,Tr,scale_var=None,scale_var_params=None):
        """Derivative of half link heat power difference to half link temperature near return line

        Parameters
        ----------
        m : float
            Link flow in kg/s
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line

        Returns
        -------
        dphi_dTr : float
            Derivative of half link flow to supply temperature
        """
        return 0.

    return m_of_phi, phi_of_m, dm_dTs, dm_dTr,dphi_dm,dphi_dTs,dphi_dTr

# ===========================================================================
def heat_exchanger(carrier):
    """Creates all link equations needed for a heat exchanger half link.
    
    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the half link
        
    Returns
    --------
    m_of_phi : function
        Flow through the half link as a function of heat power difference (and return and outflow temperature)
    phi_of_m : function
        Heat power exchanged with environment by half link as a function of flow (and return and outflow temperature)
    dm_dTs : function
        Derivative of flow to the nodal supply temperature
    dm_dTr : function
        Derivative of flow to the nodal return temperature
    """
    
    def m_of_phi(phi,Ts,Tr,scale_var=None,scale_var_params=None):
        """Half link flow as a function of heat power difference, and half links temperatures
        
        Parameters
        ----------
        phi : float
            Heat power difference
        Ts : float
            Supply temperature
        Tr : float
            Return temperature
        
        Returns
        -------
        m : float
            Link flow in kg/s
        """
        Cp = carrier.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
        return phi/(Cp*(Ts-Tr))
    
    def phi_of_m(m,Ts,Tr,scale_var=None,scale_var_params=None):
        """Half link heat power difference as a function of flow, and half links temperatures
        
        Parameters
        ----------
        m : float
            Link flow in kg/s
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line
        
        Returns
        -------
        phi : float
            Heat power difference
        """
        Cp = carrier.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
        return Cp*m*(Ts-Tr)
    
    def dm_dTs(phi,Tr,Ts,scale_var=None,scale_var_params=None):
        """Derivative of half link flow to half link temperature near supply line
        
        Parameters
        ----------
        phi : float
            Heat power
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line
        
        Returns
        -------
        dm_dTs : float
            Derivative of half link flow to supply temperature
        """
        m = m_of_phi(phi,Ts,Tr,scale_var=scale_var,scale_var_params=scale_var_params)
        return -m/(Ts-Tr) 
    
    def dm_dTr(phi,Ts,Tr,scale_var=None,scale_var_params=None):
        """Derivative of half link flow to half link temperature near return line

        Parameters
        ----------
        phi : float
            Heat power
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line

        Returns
        -------
        dm_dTr : float
            Derivative of half link flow to supply temperature
        """
        m = m_of_phi(phi,Ts,Tr,scale_var=scale_var,scale_var_params=scale_var_params)
        return m/(Ts-Tr)

    def dphi_dm(m,Ts,Tr,scale_var=None,scale_var_params=None):
        """Derivative of half link heat power difference to flow.
        
        Parameters
        ----------
        m : float
            Link flow in kg/s
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line
        
        Returns
        -------
        dphi_dm : float
            Derivative of half link heat power to flow
        """
        Cp = carrier.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
        return Cp*(Ts-Tr)
    
    def dphi_dTs(m,Tr,Ts,scale_var=None,scale_var_params=None):
        """Derivative of half link heat power difference to half link temperature near supply line
        
        Parameters
        ----------
        m : float
            Link flow in kg/s
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line
        
        Returns
        -------
        dphi_dTs : float
            Derivative of half link flow to supply temperature
        """
        Cp = carrier.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
        return Cp*m  
    
    def dphi_dTr(m,Ts,Tr,scale_var=None,scale_var_params=None):
        """Derivative of half link heat power difference to half link temperature near return line

        Parameters
        ----------
        m : float
            Link flow in kg/s
        Tr : float
            Temperature near return line
        Ts : float
            Temperature near supply line

        Returns
        -------
        dphi_dTr : float
            Derivative of half link flow to supply temperature
        """
        Cp = carrier.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
        return -Cp*m
    
    return m_of_phi, phi_of_m, dm_dTs, dm_dTr, dphi_dm, dphi_dTs, dphi_dTr
