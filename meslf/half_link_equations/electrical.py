"""Electrical half link equations, such as a nodal shunt"""

# ===========================================================================
def flow():
    """Creates the half link equations for a basic (out)flow half link
    
    Returns
    --------
    P_of_V : function
        Active power of half link (i.e. injected active power) as function of nodal voltage amplitude.
    Q_of_V : function
        Reactive power of half link (i.e. injected reactive power) as function of nodal voltage amplitude.
    """
    def P_of_V(V,scale_var=None,scale_var_params=None):
        """Active power of half link (i.e. injected active power) as function of nodal voltage amplitude.
        
        Parameters
        ----------
        V : float
            Voltage amplitude of start node
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        pass
    
    def Q_of_V(V,scale_var=None,scale_var_params=None):
        """Reactive power of half link (i.e. injected reactive power) as function of nodal voltage amplitude.
        
        Parameters
        ----------
        V : float
            Voltage amplitude of start node
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        pass
    
    return P_of_V, Q_of_V

def shunt(sus,cond):
    """Creates the half link equations for a shunt half link
    
    Parameters
    ----------
    sus : float
        Susceptance of the shunt (:math:`b_{sh}`).
    cond : float
        Conductance of the shunt (:math:`g_{sh}`).
        
    Returns
    --------
    P_of_V : function
        Active power of half link (i.e. injected active power) as function of nodal voltage amplitude.
    Q_of_V : function
        Reactive power of half link (i.e. injected reactive power) as function of nodal voltage amplitude.
    """
    
    def admittance(sus=sus,cond=cond,scale_var=None,scale_var_params=None):
        """admittance of the half link used in calculations
        
        Parameters
        ----------
        sus : float 
            susceptance
        cond : float
            conductance
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        
        Returns:
        b : float
            Susceptance of the line, possibly scaled. 
        g : float
            Conductance of the line, possibly scaled. 
        """
        b = sus
        g = cond
        if scale_var == 'per_unit':
            yb = scale_var_params['Sbase']/scale_var_params['Vbase']**2
            b /= yb
            g /= yb
        return b,g
    
    def P_of_V(V,scale_var=None,scale_var_params=None):
        """Active power of half link (i.e. injected active power) as function of nodal voltage amplitude.
        
        Parameters
        ----------
        V : float
            Voltage amplitude of start node
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        b,g = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        return g*V**2
    
    def Q_of_V(V,scale_var=None,scale_var_params=None):
        """Reactive power of half link (i.e. injected reactive power) as function of nodal voltage amplitude.
        
        Parameters
        ----------
        V : float
            Voltage amplitude of start node
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        b,g = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        return -b*V**2
    
    return P_of_V, Q_of_V
    
    
