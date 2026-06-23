"""Electrical link equations"""
from math import cos, sin

# ===========================================================================
def dummy():
    """Creates all link equations needed for a dummy link.
    
    Returns
    --------
    Pstart_of_V_delta : function
        Active power at start of line as function of nodal voltage amplitudes and angles.
    Qstart_of_V_delta : function
        Reactive power at start of line as function of nodal voltage amplitudes and angles.
    Pend_of_V_delta : function
        Active power at end of line as function of nodal voltage amplitudes and angles.
    Qend_of_V_delta : function
        Reactive power at end of line as function of nodal voltage amplitudes and angles.
    """

    def Pstart_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Active power at start of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        pass
    
    def Qstart_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Reactive power at start of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        pass
    
    def Pend_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Active power at end of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        pass
    
    def Qend_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Reactive power at end of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None 
        """
        pass
        
    return Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta

# ===========================================================================
def short_line(sus,cond):
    """Creates the link equations for a short line model
    
    Parameters
    ----------
    sus : float
        susceptance of line (b)
    cond : float
        conductance of line (g)
        
    Returns
    --------
    Pstart_of_V_delta : function
        Active power at start of line as function of nodal voltage amplitudes and angles.
    Qstart_of_V_delta : function
        Reactive power at start of line as function of nodal voltage amplitudes and angles.
    Pend_of_V_delta : function
        Active power at end of line as function of nodal voltage amplitudes and angles.
    Qend_of_V_delta : function
        Reactive power at end of line as function of nodal voltage amplitudes and angles.
    """
    def admittance(sus=sus,cond=cond,scale_var=None,scale_var_params=None):
        """admittance of the line used in calculations
        
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
            
        
    def dV(V_start,V_end):
        """Voltage amplitude drop function
        
        Parameters
        ----------
        V_start : float
            Voltage amplitude at start
        V_end : float 
            Voltage amplitude at end
            
        Returns
        -------
        dV : float
            Voltage amplitude drop
        """
        return V_start - V_end
    
    def ddelta(delta_start,delta_end):
        """Voltage angle drop function
        
        Parameters
        ----------
        delta_start : float
            Voltage angle at start
        delta_end : float 
            Voltage angle at end
            
        Returns
        -------
        ddelta : float
            Voltage angle drop
        """
        return delta_start - delta_end
    
    def Pstart_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Active power at start of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        b,g = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end)
        return g*V_start**2-V_start*V_end*(g*cos(dd)+b*sin(dd))

    def Qstart_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Reactive power at start of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        b,g = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end)
        return -b*V_start**2-V_start*V_end*(g*sin(dd)-b*cos(dd))
    
    def Pend_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Active power at end of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        b,g = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end)
        return g*V_end**2-V_start*V_end*(g*cos(dd)-b*sin(dd))
    
    def Qend_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Reactive power at end of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None 
        """
        b,g = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end)
        return -b*V_end**2-V_start*V_end*(-g*sin(dd)-b*cos(dd))
    
    return Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta
    
# ===========================================================================
def pi_line(sus,cond,sus_sh,cond_sh):
    """Creates the link equations for a pi-line model
    
    Parameters
    ----------
    sus : float
        susceptance of line (b)
    cond : float
        conductance of line (g)
    sus_sh : float
        susceptance of shunt (b_sh)
    cond_sh : float
        conductance of shunt (g_sh)
        
    Returns
    --------
    Pstart_of_V_delta : function
        Active power at start of line as function of nodal voltage amplitudes and angles.
    Qstart_of_V_delta : function
        Reactive power at start of line as function of nodal voltage amplitudes and angles.
    Pend_of_V_delta : function
        Active power at end of line as function of nodal voltage amplitudes and angles.
    Qend_of_V_delta : function
        Reactive power at end of line as function of nodal voltage amplitudes and angles.
    """
    def admittance(sus=sus,cond=cond,sus_sh=sus_sh,cond_sh=cond_sh,scale_var=None,scale_var_params=None):
        """admittance of the line and shunts used in calculations
        
        Parameters
        ----------
        sus : float
            susceptance of line
        cond : float
            conductance of line
        sus_sh : float
            susceptance of shunt
        cond_sh : float
            conductance of shunt
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        
        Returns:
        b : float
            Susceptance of the line, possibly scaled. 
        g : float
            Conductance of the line, possibly scaled. 
        b_sh : float
            Susceptance of the shunt, possibly scaled. 
        g_sh : float
            Conductance of the shunt, possibly scaled. 
        """
        b = sus
        g = cond
        b_sh = sus_sh
        g_sh = cond_sh
        if scale_var == 'per_unit':
            yb = scale_var_params['Sbase']/scale_var_params['Vbase']**2
            b /= yb
            g /= yb
            b_sh /= yb
            g_sh /= yb
        return b,g,b_sh,g_sh
    
    def dV(V_start,V_end):
        """Voltage amplitude drop function
        
        Parameters
        ----------
        V_start : float
            Voltage amplitude at start
        V_end : float 
            Voltage amplitude at end
            
        Returns
        -------
        dV : float
            Voltage amplitude drop
        """
        return V_start - V_end
    
    def ddelta(delta_start,delta_end):
        """Voltage angle drop function
        
        Parameters
        ----------
        delta_start : float
            Voltage angle at start
        delta_end : float 
            Voltage angle at end
            
        Returns
        -------
        ddelta : float
            Voltage angle drop
        """
        return delta_start - delta_end
    
    def Pstart_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Active power at start of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        b,g,b_sh,g_sh = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end)
        return (g+g_sh/2)*V_start**2-V_start*V_end*(g*cos(dd)+b*sin(dd))

    def Qstart_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Reactive power at start of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        b,g,b_sh,g_sh = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end)
        return -(b+b_sh/2)*V_start**2-V_start*V_end*(g*sin(dd)-b*cos(dd))
    
    def Pend_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Active power at end of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        b,g,b_sh,g_sh = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end)
        return (g+g_sh/2)*V_end**2-V_start*V_end*(g*cos(dd)-b*sin(dd))
    
    def Qend_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Reactive power at end of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None 
        """
        b,g,b_sh,g_sh = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end)
        return -(b+b_sh/2)*V_end**2-V_start*V_end*(-g*sin(dd)-b*cos(dd))
    
    return Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta

# ===========================================================================
def pi_line_trafo(sus,cond,sus_sh,cond_sh,ratio,phase_shift):
    """Creates the link equations for a pi-line model with a transformer at the start end of the link.
    
    Parameters
    ----------
    sus : float
        susceptance of line (b)
    cond : float
        conductance of line (g)
    sus_sh : float
        susceptance of shunt (b_sh)
    cond_sh : float
        conductance of shunt (g_sh)
    ratio : float  
        Tap ratio magnitude of the transformer (:math:`\\tau`).
    phase : float
        Phase shif angle of the transformer (:math:`\\delta_\\text{shift}`).
        
    Returns
    --------
    Pstart_of_V_delta : function
        Active power at start of line as function of nodal voltage amplitudes and angles.
    Qstart_of_V_delta : function
        Reactive power at start of line as function of nodal voltage amplitudes and angles.
    Pend_of_V_delta : function
        Active power at end of line as function of nodal voltage amplitudes and angles.
    Qend_of_V_delta : function
        Reactive power at end of line as function of nodal voltage amplitudes and angles.
    """
    def admittance(sus=sus,cond=cond,sus_sh=sus_sh,cond_sh=cond_sh,scale_var=None,scale_var_params=None):
        """admittance of the line and shunts used in calculations
        
        Parameters
        ----------
        sus : float
            susceptance of line
        cond : float
            conductance of line
        sus_sh : float
            susceptance of shunt
        cond_sh : float
            conductance of shunt
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        
        Returns
        -------
        b : float
            Susceptance of the line, possibly scaled. 
        g : float
            Conductance of the line, possibly scaled. 
        b_sh : float
            Susceptance of the shunt, possibly scaled. 
        g_sh : float
            Conductance of the shunt, possibly scaled. 
        """
        b = sus
        g = cond
        b_sh = sus_sh
        g_sh = cond_sh
        if scale_var == 'per_unit':
            yb = scale_var_params['Sbase']/scale_var_params['Vbase']**2
            b /= yb
            g /= yb
            b_sh /= yb
            g_sh /= yb
        return b,g,b_sh,g_sh
    
    def tap_ratio(ratio=ratio,phase_shift=phase_shift,scale_var=None,scale_var_params=None):
        """tap ratio magnitude and phase shift of the transformer used in calculations
        
        Parameters
        ----------
        ratio : float  
            Tap ratio magnitude of the transformer (:math:`\\tau`).
        phase_shift : float
            Phase shif angle of the transformer (:math:`\\delta_\\text{shift}`).
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        
        Returns
        -------
        tau : float
            Tap ratio magnitude of the transformer, possibly scaled.
        delta_s : float
            Phase shif angle of the transformer, possibly scaled
        """
        tau = ratio
        delta_s = phase_shift
        if scale_var == 'per_unit':
            raise NotImplementedError('Per unit scaling is not implemented for the tap ratio of the transformer')
        return tau, delta_s
            
        
    def dV(V_start,V_end):
        """Voltage amplitude drop function
        
        Parameters
        ----------
        V_start : float
            Voltage amplitude at start
        V_end : float 
            Voltage amplitude at end
            
        Returns
        -------
        dV : float
            Voltage amplitude drop
        """
        return V_start - V_end
    
    def ddelta(delta_start,delta_end):
        """Voltage angle drop function
        
        Parameters
        ----------
        delta_start : float
            Voltage angle at start
        delta_end : float 
            Voltage angle at end
            
        Returns
        -------
        ddelta : float
            Voltage angle drop
        """
        return delta_start - delta_end
    
    def Pstart_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Active power at start of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        b,g,b_sh,g_sh = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        tau, delta_s = tap_ratio(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end) - delta_s
        return 1/tau**2*(g+g_sh/2)*V_start**2-1/tau*V_start*V_end*(g*cos(dd)+b*sin(dd))

    def Qstart_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Reactive power at start of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        b,g,b_sh,g_sh = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        tau, delta_s = tap_ratio(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end) - delta_s
        return -1/tau**2*(b+b_sh/2)*V_start**2-1/tau*V_start*V_end*(g*sin(dd)-b*cos(dd))
    
    def Pend_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Active power at end of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        b,g,b_sh,g_sh = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        tau, delta_s = tap_ratio(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end) - delta_s
        return (g+g_sh/2)*V_end**2-1/tau*V_start*V_end*(g*cos(dd)-b*sin(dd))
    
    def Qend_of_V_delta(V_start,V_end,delta_start,delta_end,scale_var=None,scale_var_params=None):
        """Reactive power at end of link as function of start and end voltage amplitudes and angles
        
        Parameters
        ----------
        V_start : float
            start voltage amplitude
        V_end : float
            end voltage amplitude
        delta_start : float
            start voltage angle
        delta_end : float
            end voltage angle
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None 
        """
        b,g,b_sh,g_sh = admittance(scale_var=scale_var,scale_var_params=scale_var_params)
        tau, delta_s = tap_ratio(scale_var=scale_var,scale_var_params=scale_var_params)
        dd = ddelta(delta_start,delta_end) - delta_s
        return -(b+b_sh/2)*V_end**2-1/tau*V_start*V_end*(-g*sin(dd)-b*cos(dd))
    
    return Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta
