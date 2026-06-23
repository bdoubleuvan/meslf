"""Thermal link equations, such as temperature drops for pipes"""
import numpy as np

# ===========================================================================
def dummy():
    """Creates all link equations needed for a dummy link.

    Returns
    --------
    temp_drop_fac : function
        Temperature drop factor
    temp_drop_fac_dm : function
        Derivative of the temperature drop factor to (mass) flow
    Tend_of_Tstart : function
        Temperature at the end of the link, as a function of (mass) flow and temperature at the start of the link. Start and end with respect to actual direction of flow.
    dTend_dm : function
        Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to link mass flow. Start and end with respect to actual direction of flow.
    dTend_dTstart : function
        Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to temperature at start of link. Start and end with respect to actual direction of flow.
    """

    def temp_drop_fac(m,scale_var=None,scale_var_params=None):
        """Temperature drop factor

        Parameters
        ----------
        m : float
            (mass) flow in the link
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 1.

    def temp_drop_fac_dm(m,scale_var=None,scale_var_params=None):
        """Derivate of the temperature drop factor with respect to (mass) flow

        Parameters
        ----------
        m : float
            (mass) flow in the link
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 0

    def Tend_of_Tstart(m,Tstart,scale_var=None,scale_var_params=None):
        """Temperature at the end of the link, as a function of of mass flow and temperature at the start of the link. Start and end with respect to actual direction of flow.

        Parameters
        ----------
        m : float
            (mass) flow in the link
        Tstart : float
            Temperature at start of the link in the supply line
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return Tstart

    def dTend_dm(m,Tstart,scale_var=None,scale_var_params=None):
        """Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to link mass flow.
        
        Parameters
        ----------
        m : float
            (mass) flow in the link
        Tstart : float
            Temperature at start of the link in the supply line
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 0
    
    def dTend_dTstart(m,Tstart,scale_var=None,scale_var_params=None):
        """Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to temperature at start of link.
        
        Parameters
        ----------
        m : float
            (mass) flow in the link
        Tstart : float
            Temperature at start of the link in the supply line
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 1

    return temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart

# ===========================================================================
def perfect_isolated_pipe():
    """Creates all link equations needed for a perfectly isolated pipe.

    Returns
    --------
    temp_drop_fac : function
        Temperature drop factor
    temp_drop_fac_dm : function
        Derivative of the temperature drop factor to (mass) flow
    Tend_of_Tstart : function
        Temperature at the end of the link, as a function of (mass) flow and temperature at the start of the link. Start and end with respect to actual direction of flow.
    dTend_dm : function
        Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to link mass flow. Start and end with respect to actual direction of flow.
    dTend_dTstart : function
        Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to temperature at start of link. Start and end with respect to actual direction of flow.
    """
    def temp_drop_fac(m,scale_var=None,scale_var_params=None):
        """Temperature drop factor

        Parameters
        ----------
        m : float
            (mass) flow in the link
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 1.

    def temp_drop_fac_dm(m,scale_var=None,scale_var_params=None):
        """Derivate of the temperature drop factor with respect to (mass) flow

        Parameters
        ----------
        m : float
            (mass) flow in the link
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 0.

    def Tend_of_Tstart(m,Tstart,scale_var=None,scale_var_params=None):
        """Temperature at the end of the link, as a function of of mass flow and temperature at the start of the link. Start and end with respect to actual direction of flow.

        Parameters
        ----------
        m : float
            (mass) flow in the link
        Tstart : float
            Temperature at start of the link in the supply line
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return temp_drop_fac(m,scale_var=scale_var,scale_var_params=scale_var_params)*Tstart

    def dTend_dm(m,Tstart,scale_var=None,scale_var_params=None):
        """Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to link mass flow.
        
        Parameters
        ----------
        m : float
            (mass) flow in the link
        Tstart : float
            Temperature at start of the link in the supply line
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return temp_drop_fac_dm(m,scale_var=scale_var,scale_var_params=scale_var_params)*Tstart
    
    def dTend_dTstart(m,Tstart,scale_var=None,scale_var_params=None):
        """Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to temperature at start of link.
        
        Parameters
        ----------
        m : float
            (mass) flow in the link
        Tstart : float
            Temperature at start of the link in the supply line
        carrier : Carrier
            Carrier flowing through the pipe
        U : float
            Heat transfer coefficient of the pipe in W/(m^2 K)
        L : float
            Length of the pipe in m
        D : float
            Diameter of the pipe in m
        Ta : float
            Temperature directly outside pipe (ambient temperature). Unscaled
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return temp_drop_fac(m,scale_var=scale_var,scale_var_params=scale_var_params)

    return temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart

# ===========================================================================
def standard_pipe(carrier, U, L, D, Ta=None):
    """Creates all link equations needed for a perfectly isolated pipe.

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    U : float
        Heat transfer coefficient of the pipe in W/(m^2 K)
    L : float
        Length of the pipe in m
    D : float
        Diameter of the pipe in m
    Ta : float, optional
        Temperature directly outside pipe (ambient temperature). Default is None, such that temperature is set when link is added to a network.

    Returns
    --------
    temp_drop_fac : function
        Temperature drop factor
    temp_drop_fac_dm : function
        Derivative of the temperature drop factor to (mass) flow
    Tend_of_Tstart : function
        Temperature at the end of the link, as a function of (mass) flow and temperature at the start of the link. Start and end with respect to actual direction of flow.
    dTend_dm : function
        Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to link mass flow. Start and end with respect to actual direction of flow.
    dTend_dTstart : function
        Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to temperature at start of link. Start and end with respect to actual direction of flow.
    """

    def temp_drop_fac(m,carrier=carrier,U=U,L=L,D=D,scale_var=None,scale_var_params=None):
        """Temperature drop factor (uses the unscaled flow)

        Parameters
        ----------
        m : float
            (mass) flow in the link
        carrier : Carrier
            Carrier flowing through the pipe
        U : float
            Heat transfer coefficient of the pipe in W/(m^2 K)
        L : float
            Length of the pipe in m
        D : float
            Diameter of the pipe in m
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        if scale_var == 'per_unit':
            m *= scale_var_params['qbase']
        return np.exp(-np.pi*D*U*L/(carrier.Cp*np.abs(m)))

    def temp_drop_fac_dm(m,carrier=carrier,U=U,L=L,D=D,scale_var=None,scale_var_params=None):
        """Derivate of the temperature drop factor with respect to (mass) flow (uses the unscaled flow)

        Parameters
        ----------
        m : float
            (mass) flow in the link
        carrier : Carrier
            Carrier flowing through the pipe
        U : float
            Heat transfer coefficient of the pipe in W/(m^2 K)
        L : float
            Length of the pipe in m
        D : float
            Diameter of the pipe in m
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        psi_ac = temp_drop_fac(m,carrier=carrier,U=U,L=L,D=D,scale_var=scale_var,scale_var_params=scale_var_params) # scales m itself
        if scale_var == 'per_unit':
            m *= scale_var_params['qbase']
        der = np.sign(m)*psi_ac*np.pi*D*U*L/(carrier.Cp)*1./(np.abs(m)**2)
        if scale_var == 'per_unit':
            der *= scale_var_params['qbase']
        return der

    def Tend_of_Tstart(m,Tstart,carrier=carrier,U=U,L=L,D=D,Ta=Ta,scale_var=None,scale_var_params=None):
        """Temperature at the end of the link, as a function of of mass flow and temperature at the start of the link. Start and end with respect to actual direction of flow.

        Parameters
        ----------
        m : float
            (mass) flow in the link
        Tstart : float
            Temperature at start of the link in the supply line
        carrier : Carrier
            Carrier flowing through the pipe
        U : float
            Heat transfer coefficient of the pipe in W/(m^2 K)
        L : float
            Length of the pipe in m
        D : float
            Diameter of the pipe in m
        Ta : float
            Temperature directly outside pipe (ambient temperature). Unscaled
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        if scale_var == 'per_unit':
            Ta /= scale_var_params['Tbase']
        return temp_drop_fac(m,carrier=carrier,U=U,L=L,D=D,scale_var=scale_var,scale_var_params=scale_var_params)*(Tstart-Ta) + Ta

    def dTend_dm(m,Tstart,carrier=carrier,U=U,L=L,D=D,Ta=Ta,scale_var=None,scale_var_params=None):
        """Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to link mass flow.
        
        Parameters
        ----------
        m : float
            (mass) flow in the link
        Tstart : float
            Temperature at start of the link in the supply line
        carrier : Carrier
            Carrier flowing through the pipe
        U : float
            Heat transfer coefficient of the pipe in W/(m^2 K)
        L : float
            Length of the pipe in m
        D : float
            Diameter of the pipe in m
        Ta : float
            Temperature directly outside pipe (ambient temperature). Unscaled
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        if scale_var == 'per_unit':
            Ta /= scale_var_params['Tbase']
        return temp_drop_fac_dm(m,carrier=carrier,U=U,L=L,D=D,scale_var=scale_var,scale_var_params=scale_var_params)*(Tstart-Ta)
    
    def dTend_dTstart(m,Tstart,carrier=carrier,U=U,L=L,D=D,Ta=Ta,scale_var=None,scale_var_params=None):
        """Derivative of temperature at start of the link, as function of mass flow and temperature at start of the link, to temperature at start of link.
        
        Parameters
        ----------
        m : float
            (mass) flow in the link
        Tstart : float
            Temperature at start of the link in the supply line
        carrier : Carrier
            Carrier flowing through the pipe
        U : float
            Heat transfer coefficient of the pipe in W/(m^2 K)
        L : float
            Length of the pipe in m
        D : float
            Diameter of the pipe in m
        Ta : float
            Temperature directly outside pipe (ambient temperature). Unscaled
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return temp_drop_fac(m,carrier=carrier,U=U,L=L,D=D,scale_var=scale_var,scale_var_params=scale_var_params)
        
    return temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart
