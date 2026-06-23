"""This modulde defines the hydraulic link equations.
The options for the links are dummy_, pipe_linear_, pipe_high_pressure_implicit_, pipe_high_pressure_, pipe_low_pressure_, pipe_low_pressure_implicit_, compressor_, and resistor_.


For a link :math:`k` between nodes :math:`i` and :math:`j`, the most general form of the link equation is :math:`f_k(q_k,p_i,p_j)=0`. For most pipes the link equation can be reduced to :math:`f_k(q_k,\Delta p_k(p_i,p_j))=0` by defining a pressure drop function :math:`\Delta p_k(p_i,p_j)`. For most pipe models, is pressure drop function is of the form :math:`\Delta p_k(p_i,p_j) = g(p_i)-g(p_j)` with :math:`g(p)` some function pressure. The link flow through the pipes can be expressed as a function of the pressure drop [1]_:

.. math::
    q_k =C_k\\text{sign}(\Delta p_k)\sqrt{ \\frac{|\Delta p_k|}{f_k} }

with :math:`f_k` the friction factor of the pipe, and :math:`C_k` a pipe constant. Rewriting this equations expresses the pressure drop as a function of the link flow:

.. math::
    \Delta p_k(q_k) = \\frac{1}{C_k^2}f_kq_k|q_k|

There are then two ways to formulate the link equation:

.. math::
    f^a_k(q_k,\Delta p_k(p_i,p_j)) = q_k - q_k(\Delta p_k)

    f^b_k(q_k,\Delta p_k(p_i,p_j)) = \Delta p_k - \Delta p_k(q_k)

If possible, the models of all non-pipe elements are reformulated in this form, with :math:`f_k:=1`. Hence, non-pipe elements can have a 'pipe constant' :math:`C_k`.

This means that most of the links will have the same form for the link equations :math:`f^a` and :math:`f^b`. For these elements, the following holds for the derivative of the first link equation :math:`f^a_k(q_k,p_i,p_j)`:

.. math::
    \\frac{\partial f^a_k}{\partial\Delta p_k}=-\\frac{dq_k(\Delta p_k)}{d\Delta p_k}

    \\frac{\partial f^a_k}{\partial p_i}=\\frac{\partial f^a_k}{\partial\Delta p_k} \\frac{\partial\Delta p_k(p_i,p_j)}{\partial p_i}

    \\frac{\partial f^a_k}{\partial p_j}=\\frac{\partial f^a_k}{\partial\Delta p_k} \\frac{\partial\Delta p_k(p_i,p_j)}{\partial p_j}

    \\frac{\partial f^a_k}{\partial q_k}=1

And similar expressions hold for the derivatives of the second link equation :math:`f^b_k(q_k,p_i,p_j)` (unless stated otherwise):

.. math::
    \\frac{\partial f^b_k}{\partial\Delta p_k}=1

    \\frac{\partial f^b_k}{\partial p_i}=\\frac{\partial\Delta p_k(p_i,p_j)}{\partial p_i}

    \\frac{\partial f^b_k}{\partial p_j}=\\frac{\partial\Delta p_k(p_i,p_j)}{\partial p_j}

    \\frac{\partial f^b_k}{\partial q_k}=-\\frac{d\Delta p_k(q_k)}{dq_k}

All links are assumed to have the structure as described above, such that:

.. function:: meslf.link_equations.hydraulic.<link>(*args)

    Parameters
    -----------
    ``*args``
        Input arguments needed to create the link.

    Returns
    --------
    pipe_const : function
        The pipe constant :math:`C_k` used in the rest of the link equations.
    dp : function
        The pressure drop function of the link :math:`\Delta p_k(p_i,p_j)`.
    fa : function
        The first option for the link equation :math:`f_k(q_k,p_i,p_j)=0`, namely :math:`f^a_k(q_k,p_i,p_j):=q_k - q_k(p_i,p_j)=0`.
    fb : function
        Second option for a link equation :math:`f_k(q_k,p_i,p_j)=0`, namely :math:`f^b_k(q_k,p_i,p_j):=\Delta p_k(p_i,p_j) - \Delta p_k(q)=0`.
    q_of_dp : function
        Link flow as function of pressure drop function :math:`q_k = q_k(\Delta p_k(p_i,p_j))`.
    dp_of_q : function
        Pressure drop function as function of link flow :math:`\Delta p_k = \Delta p_k(q_k)`.
    ddp_dp : function
        Derivative of pressure drop function to start pressure, :math:`\\frac{\partial\Delta p_k(p_i,p_j)}{\partial p_i}`, and end pressures, :math:`\\frac{\partial \Delta p_k(p_i,p_j)}{\partial p_j}`.
    ddp_dq : function
        Derivative of pressure drop function to link flow :math:`\\frac{d\Delta p_k(q_k)}{dq_k}`.
    dq_ddp : function
        Derivative of link flow to the pressure drop function :math:`\\frac{dq_k(\Delta p_k)}{d\Delta p_k}`.
    dfa_ddp : function
        Derivative of the first link equation to the pressure drop function :math:`\\frac{\partial f^a_k}{\partial\Delta p_k}`.
    dfb_ddp : function
        Derivative of the second link equation to the pressure drop function :math:`\\frac{\partial f^b_k}{\partial\Delta p_k}`.
    dfa_dp : function
        Derivative of the first link equation to the start pressure, :math:`\\frac{\partial f^a_k}{\partial p_i}`, and end pressures, :math:`\\frac{\partial f^a_k}{\partial p_j}`.
    dfb_dp : function
        Derivative of the second link equation to the start pressure, :math:`\\frac{\partial f^b_k}{\partial p_i}`, and end pressures, :math:`\\frac{\partial f^b_k}{\partial p_j}`.
    dfa_dq : function
        Derivative of the first link equation to link flow :math:`\\frac{\partial f^a_k}{\partial q_k}`.
    dfb_dq : function
        Derivative of the second link equation to link flow :math:`\\frac{\partial f^b_k}{\partial q_k}`.


See the links (dummy_, pipe_linear_, pipe_high_pressure_implicit_, pipe_high_pressure_, pipe_low_pressure_, pipe_low_pressure_implicit_, compressor_, or resistor_) for the specific models used for the all equations, and the required input parameters.

References
----------
.. [1] Osiadacz
"""
import numpy as np
import scipy as sp
import scipy.optimize
import warnings

# ===========================================================================
def dummy():
    """Creates all link equations needed for a dummy link.
    """

    def pipe_const(scale_var=None,scale_var_params=None):
        """Pipe constant

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        pass

    def dp(p_start,p_end):
        """Pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure
        p_end : float
            end pressure

        Returns
        -------
        dp : float
            basic pressure drop, i.e. p_start - p_end
        """
        return p_start - p_end

    def q_of_dp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Link flow as function of start and end pressures.

        Parameters
        ----------
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        """
        pass

    def dp_of_q(q,scale_var=None,scale_var_params=None):
        """Pressure 'drop' as a function of link flow

        Parameters
        ----------
        q : float
            link flow
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        pass

    def fa(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            link flow
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        pass

    def fb(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            link flow
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        pass

    def ddp_dp(p_start,p_end):
        """Derivative of pressure drop equation to start and end pressure

        Parameters
        ----------
        p_start : float
            start pressure
        p_end : float
            end pressure

        Returns
        -------
        ddp_dp_start : float
            Derivative of pressure drop function to the start pressure
        ddp_dp_end : float
            Derivative of pressure drop function to the end pressure
        """
        return 1.,-1.

    def ddp_dq(q,scale_var=None,scale_var_params=None):
        """Derivative of pressure drop equation to link flow

        Parameters
        ----------
        q : float
            link flow
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 0.

    def dq_ddp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link flow to pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dq_ddp : float
            Derivative of link flow to pressure drop
        """
        return 0.

    def dfa_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 0.

    def dfb_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 0.

    def dfa_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        return 0., 0.

    def dfb_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        return 0., 0.

    def dfa_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            link flow
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return 1.

    def dfb_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            link flow
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return 1.

    return pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq

# ===========================================================================
def pipe_linear(alpha):
    """Creates all link equations needed for a pipe with linear flow

    Parameters
    ----------
    alpha : float
        Parameter :math:`\\alpha` for linear pipe flow.
    """
    def pipe_const(alpha=alpha,scale_var=None,scale_var_params=None):
        """Pipe constant

        Parameters
        ----------
        alpha : float
            Parameter for linear pipe flow
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        C : float
            Pipe constant, possibly scaled.
        """
        if scale_var == 'per_unit':
            alpha *= scale_var_params['pbase']/scale_var_params['qbase']
        return alpha

    def dp(p_start,p_end):
        """Pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure
        p_end : float
            end pressure

        Returns
        -------
        dp : float
            basic pressure drop, i.e. p_start - p_end
        """
        return p_start - p_end


    def q_of_dp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Link flow as function of start and end pressures.

        Parameters
        ----------
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        q : float
            Link flow.
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return C*dp(p_start,p_end)

    def dp_of_q(q,scale_var=None,scale_var_params=None):
        """Pressure 'drop' as a function of link flow

        Parameters
        ----------
        q : float
            link flow
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dp : float
            pressure drop function
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return q/C

    def fa(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            link flow
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return q - q_of_dp(p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def fb(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            link flow
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return dp(p_start,p_end)-dp_of_q(q,scale_var=scale_var,scale_var_params=scale_var_params)

    def ddp_dp(p_start,p_end):
        """Derivative of pressure drop equation to start and end pressure

        Parameters
        ----------
        p_start : float
            start pressure
        p_end : float
            end pressure

        Returns
        -------
        ddp_dp_start : float
            Derivative of pressure drop function to the start pressure
        ddp_dp_end : float
            Derivative of pressure drop function to the end pressure
        """
        return 1.,-1.

    def ddp_dq(q,scale_var=None,scale_var_params=None):
        """Derivative of pressure drop equation to link flow

        Parameters
        ----------
        q : float
            link flow
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return 1/C

    def dq_ddp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link flow to pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dq_ddp : float
            Derivative of link flow to pressure drop
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return C

    def dfa_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return -dq_ddp(p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfb_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 1.

    def dfa_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfb_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure
        p_end : float
            end pressure
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        return ddp_dp(p_start,p_end)

    def dfa_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            link flow
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return 1.

    def dfb_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            link flow
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return -ddp_dq(q,scale_var=scale_var,scale_var_params=scale_var_params)

    return pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq

# ===========================================================================
def pipe_high_pressure_implicit(carrier,fric_fac,D,L,eps,*args,fric_fac_der_q=lambda carrier,D,L,eps,q : 0,**kwargs):
    """Creates all link equations needed for high pressure pipe flow, where the friction factor is a function of the flow q

    Parameters
    ----------
    carrier :  Carrier
        Carrier flowing through the pipe
    fric_fac : function
        (Fanning) friction factor as a function of carrier, D, and L, eps, q, and ``**kwargs``
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    eps : float
        Relative roughness of the pipe
    fric_fac_der_q : function, optional
        Derivative of the friction factor function to the link flow, as a function of carrier, D, and L, eps, q. Default is 0
    """
    def pipe_const(carrier=carrier,D=D,L=L,scale_var=None,scale_var_params=None):
        """Pipe constant

        Parameters
        ----------
        carrier : Carrier
            Gas carrier
        D : float
            Pipe diameter in m
        L : float
            Pipe length in m
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        C : float
            Pipe constant, possibly scaled.
        """
        C = np.pi/8*np.sqrt(carrier.S*D**5/(carrier.R_air*L*carrier.T*carrier.Z))
        if scale_var == 'per_unit':
            C *= scale_var_params['pbase']/scale_var_params['qbase']
        return C

    def dp(p_start,p_end):
        """Pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2

        Returns
        -------
        dp : float
            pressure drop function for the standard pipe equation
        """
        return p_start**2 - p_end**2


    def q_of_dp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Link flow as function of start and end pressures.

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        q : float
            Link flow in kg/s.
        """
        #C=pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        #dp_func = dp(p_start,p_end)
        #q_init = C*np.sign(dp_func)*np.sqrt(np.abs(dp_func))
        #def flow(q):
            #return C*np.sign(dp_func)*np.sqrt(np.abs(dp_func)/fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)) - q
        #q = sp.optimize.fsolve(flow,q_init)
        #return q[0]
        pass

    def dp_of_q(q,scale_var=None,scale_var_params=None):
        """Pressure 'drop' as a function of link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dp : float
            pressure drop function
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)/C**2*q*np.abs(q)

    def fa(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return q - C*np.sign(dp_func)*np.sqrt(np.abs(dp_func)/fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs))

    def fb(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return dp(p_start,p_end)-dp_of_q(q,scale_var=scale_var,scale_var_params=scale_var_params)

    def ddp_dp(p_start,p_end):
        """Derivative of pressure drop equation to start and end pressure

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        ddp_dp_start : float
            Derivative of pressure drop function to the start pressure
        ddp_dp_end : float
            Derivative of pressure drop function to the end pressure
        """
        return 2.*p_start,-2.*p_end

    def ddp_dq(q,scale_var=None,scale_var_params=None):
        """Derivative of pressure drop equation to link flow

        Parameters
        ----------
        q : float
            link flow
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return 1/C**2 * np.abs(q) * (q*fric_fac_der_q(carrier,D,L,eps,q,scale_var=scale_var,scale_var_params=scale_var_params) + 2*fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs))

    def dq_ddp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link flow to pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dq_ddp : float
            Derivative of link flow to pressure drop
        """
        #C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        #q = q_of_dp(p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)
        #dp_func = dp(p_start,p_end)
        #return 1/(1+1/2*C*fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)**(-3/2)*fric_fac_der_q(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)*np.sign(dp_func)*np.sqrt(np.abs(dp_func)))*1/2*C*1/np.sqrt(fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)*np.abs(dp_func))
        pass

    def dfa_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return -1/2*C*1/np.sqrt(fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)*np.abs(dp_func))

    def dfb_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 1.

    def dfa_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfb_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfb_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfb_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfa_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return 1.+1/2*C*fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)**(-3/2)*fric_fac_der_q(carrier,D,L,eps,q,scale_var=scale_var,scale_var_params=scale_var_params)*np.sign(dp_func)*np.sqrt(np.abs(dp_func))

    def dfb_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return -ddp_dq(q,scale_var=scale_var,scale_var_params=scale_var_params)

    return pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq

# ===========================================================================
def pipe_high_pressure(carrier,fric_fac,D,L,*args,**kwargs):
    """Creates all link equations needed for high pressure pipe flow

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    fric_fac : function
        (Fanning) friction factor as a function of carrier, D, and L, and ``**kwargs``
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    """
    def pipe_const(carrier=carrier,D=D,L=L,scale_var=None,scale_var_params=None):
        """Pipe constant

        Parameters
        ----------
        carrier : Carrier
            Gas carrier
        D : float
            Pipe diameter in m
        L : float
            Lenght of the pipe in m
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        C : float
            Pipe constant, possibly scaled.
        """
        C = np.pi/8*np.sqrt(carrier.S*D**5/(carrier.R_air*L*carrier.T*carrier.Z))
        if scale_var == 'per_unit':
            C *= scale_var_params['pbase']/scale_var_params['qbase']
        return C

    def dp(p_start,p_end):
        """Pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2

        Returns
        -------
        dp : float
            pressure drop function for the standard pipe equation
        """
        return p_start**2 - p_end**2


    def q_of_dp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Link flow as function of start and end pressures.

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        q : float
            Link flow in kg/s.
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return C*np.sign(dp_func)*np.sqrt(np.abs(dp_func)/fric_fac(carrier,D,L,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs))

    def dp_of_q(q,scale_var=None,scale_var_params=None):
        """Pressure 'drop' as a function of link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dp : float
            pressure drop function
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return fric_fac(carrier,D,L,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)/C**2*q*np.abs(q)

    def fa(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return q - q_of_dp(p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def fb(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return dp(p_start,p_end)-dp_of_q(q,scale_var=scale_var,scale_var_params=scale_var_params)

    def ddp_dp(p_start,p_end):
        """Derivative of pressure drop equation to start and end pressure

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        ddp_dp_start : float
            Derivative of pressure drop function to the start pressure
        ddp_dp_end : float
            Derivative of pressure drop function to the end pressure
        """
        return 2.*p_start,-2.*p_end

    def ddp_dq(q,scale_var=None,scale_var_params=None):
        """Derivative of pressure drop equation to link flow

        Parameters
        ----------
        q : float
            link flow
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return fric_fac(carrier,D,L,*args,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)/C**2*2*np.abs(q)

    def dq_ddp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link flow to pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dq_ddp : float
            Derivative of link flow to pressure drop
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return 1/2*C/np.sqrt(np.abs(dp_func)*fric_fac(carrier,D,L,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs))

    def dfa_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return -dq_ddp(p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfb_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 1.

    def dfa_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfb_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfb_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfb_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfa_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return 1.

    def dfb_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return -ddp_dq(q,scale_var=scale_var,scale_var_params=scale_var_params)

    return pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq

# ===========================================================================
def pipe_low_pressure(carrier,fric_fac,D,L,*args,**kwargs):
    """Creates all link equations needed for a pipe with standard flow in a low pressure system

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    fric_fac : function
        (Fanning) friction factor as a function of carrier, D, and L, and ``**kwargs``
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    """
    def pipe_const(carrier=carrier,D=D,L=L,scale_var=None,scale_var_params=None):
        """Pipe constant

        Parameters
        ----------
        carrier : Carrier
            Carrier in the pipe
        D : float
            Inner pipe diameter in m
        L : float
            Pipe length in m
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        C : float
            Pipe constant, possibly scaled.
        """
        C = np.pi/8 * np.sqrt(2*carrier.rhon*D**5 / L)
        if scale_var == 'per_unit':
            C *= np.sqrt(scale_var_params['pbase'])/scale_var_params['qbase']
        return C

    def dp(p_start,p_end):
        """Pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2

        Returns
        -------
        dp : float
            pressure drop function for the standard pipe equation
        """
        return p_start - p_end


    def q_of_dp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Link flow as function of start and end pressures.

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        q : float
            Link flow in kg/s.
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return C*np.sign(dp_func)*np.sqrt(np.abs(dp_func)/fric_fac(carrier,D,L,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs))

    def dp_of_q(q,scale_var=None,scale_var_params=None):
        """Pressure 'drop' as a function of link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dp : float
            pressure drop function
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return fric_fac(carrier,D,L,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)/C**2*q*np.abs(q)

    def fa(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return q - q_of_dp(p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def fb(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return dp(p_start,p_end)-dp_of_q(q,scale_var=scale_var,scale_var_params=scale_var_params)

    def ddp_dp(p_start,p_end):
        """Derivative of pressure drop equation to start and end pressure

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2


        Returns
        -------
        ddp_dp_start : float
            Derivative of pressure drop function to the start pressure
        ddp_dp_end : float
            Derivative of pressure drop function to the end pressure
        """
        return 1.,-1.

    def ddp_dq(q,scale_var=None,scale_var_params=None):
        """Derivative of pressure drop equation to link flow

        Parameters
        ----------
        q : float
            link flow
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return fric_fac(carrier,D,L,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)/C**2*2*np.abs(q)

    def dq_ddp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link flow to pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dq_ddp : float
            Derivative of link flow to pressure drop
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return 1/2*C/np.sqrt(np.abs(dp_func)*fric_fac(carrier,D,L,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs))

    def dfa_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return -dq_ddp(p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfb_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 1.

    def dfa_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfb_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfb_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfb_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfa_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return 1.

    def dfb_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return -ddp_dq(q,scale_var=scale_var,scale_var_params=scale_var_params)

    return pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq

# ===========================================================================
def pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,*args,fric_fac_der_q=lambda carrier,D,L,eps,q : 0,**kwargs):
    """Creates all link equations needed for low pressure pipe flow, where the friction factor is a function of the flow q

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    fric_fac : function
        (Fanning) friction factor as a function of carrier, D, and L, eps, q, and ``**kwargs``
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    eps : float
        Relative roughness of the pipe
    fric_fac_der_q : function, optional
        Derivative of the friction factor function to the link flow, as a function of carrier, D, and L, eps, q. Default is 0
    """
    def pipe_const(carrier=carrier,D=D,L=L,scale_var=None,scale_var_params=None):
        """Pipe constant

        Parameters
        ----------
        carrier : Carrier
            Gas carrier
        D : float
            Pipe diameter in m
        L : float
            Pipe length in m
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        C : float
            Pipe constant, possibly scaled.
        """
        C = np.pi/8 * np.sqrt(2*carrier.rhon*D**5 / L)
        if scale_var == 'per_unit':
            C *= np.sqrt(scale_var_params['pbase'])/scale_var_params['qbase']
        return C

    def dp(p_start,p_end):
        """Pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2

        Returns
        -------
        dp : float
            pressure drop function for the standard pipe equation
        """
        return p_start - p_end


    def q_of_dp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Link flow as function of start and end pressures.

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        q : float
            Link flow in kg/s.
        """
        #C=pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        #dp_func = dp(p_start,p_end)
        #q_init = C*np.sign(dp_func)*np.sqrt(np.abs(dp_func))
        #def flow(q):
            #return C*np.sign(dp_func)*np.sqrt(np.abs(dp_func)/fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)) - q
        #q = sp.optimize.fsolve(flow,q_init)
        #return q[0]
        pass

    def dp_of_q(q,scale_var=None,scale_var_params=None):
        """Pressure 'drop' as a function of link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dp : float
            pressure drop function
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)/C**2*q*np.abs(q)

    def fa(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return q - C*np.sign(dp_func)*np.sqrt(np.abs(dp_func)/fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs))

    def fb(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return dp(p_start,p_end)-dp_of_q(q,scale_var=scale_var,scale_var_params=scale_var_params)

    def ddp_dp(p_start,p_end):
        """Derivative of pressure drop equation to start and end pressure

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        ddp_dp_start : float
            Derivative of pressure drop function to the start pressure
        ddp_dp_end : float
            Derivative of pressure drop function to the end pressure
        """
        return 1.,-1.

    def ddp_dq(q,scale_var=None,scale_var_params=None):
        """Derivative of pressure drop equation to link flow

        Parameters
        ----------
        q : float
            link flow
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return 1/C**2*np.abs(q)*(q*fric_fac_der_q(carrier,D,L,eps,q,scale_var=scale_var,scale_var_params=scale_var_params) + 2*fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs))

    def dq_ddp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link flow to pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dq_ddp : float
            Derivative of link flow to pressure drop
        """
        #C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        #q = q_of_dp(p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)
        #dp_func = dp(p_start,p_end)
        #return 1/(1+1/2*C*fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)**(-3/2)*fric_fac_der_q(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)*np.sign(dp_func)*np.sqrt(np.abs(dp_func)))*1/2*C*1/np.sqrt(fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)*np.abs(dp_func))
        pass

    def dfa_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return -1/2*C*1/np.sqrt(fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)*np.abs(dp_func))

    def dfb_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 1.

    def dfa_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfb_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfb_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfb_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfa_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return 1.+1/2*C*fric_fac(carrier,D,L,eps,q,*args,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)**(-3/2)*fric_fac_der_q(carrier,D,L,eps,q,scale_var=scale_var,scale_var_params=scale_var_params)*np.sign(dp_func)*np.sqrt(np.abs(dp_func))

    def dfb_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return -ddp_dq(q,scale_var=scale_var,scale_var_params=scale_var_params)

    return pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq

# ===========================================================================
def compressor(r,*args,**kwargs):
    """Create all link equations needed for a compressor link

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    r : float
        Compressor ratio :math:`r`.
    """

    def pipe_const(r=r,scale_var=None,scale_var_params=None):
        """Pipe constant

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return r

    def dp(p_start,p_end):
        """Pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2

        Returns
        -------
        dp : float
            pressure drop function for the standard pipe equation
        """
        return p_start*r - p_end


    def q_of_dp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Link flow as function of start and end pressures.

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        q : float
            Link flow in kg/s.
        """
        pass

    def dp_of_q(q,scale_var=None,scale_var_params=None):
        """Pressure 'drop' as a function of link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dp : float
            pressure drop function
        """
        pass

    def fa(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return -dp(p_start,p_end)

    def fb(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return fa(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def ddp_dp(p_start,p_end):
        """Derivative of pressure drop equation to start and end pressure

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2

        Returns
        -------
        ddp_dp_start : float
            Derivative of pressure drop function to the start pressure
        ddp_dp_end : float
            Derivative of pressure drop function to the end pressure
        """
        return r,-1.

    def ddp_dq(q,scale_var=None,scale_var_params=None):
        """Derivative of pressure drop equation to link flow

        Parameters
        ----------
        q : float
            link flow
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 0.

    def dq_ddp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link flow to pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dq_ddp : float
            Derivative of link flow to pressure drop
        """
        pass

    def dfa_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return -1.

    def dfb_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfa_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfb_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        return dfa_dp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfa_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return 0.

    def dfb_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return dfa_dq(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    return pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq

# ===========================================================================
def resistor(C,*args,**kwargs):
    """Create all link equations needed for a non-linear resistor link

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    C : float
        Pipe constant / 'resistance' (NB. Not the same C as in my notes on this model. This C matches the rest of the link equations.
    """

    def pipe_const(C=C,scale_var=None,scale_var_params=None):
        """Pipe constant

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        if scale_var == 'per_unit':
            C *= np.sqrt(scale_var_params['pbase'])/scale_var_params['qbase']
        return C

    def dp(p_start,p_end):
        """Pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2

        Returns
        -------
        dp : float
            pressure drop function for the standard pipe equation
        """
        return p_start - p_end


    def q_of_dp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Link flow as function of start and end pressures.

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        q : float
            Link flow in kg/s.
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return C*np.sign(dp_func)*np.sqrt(np.abs(dp_func))

    def dp_of_q(q,scale_var=None,scale_var_params=None):
        """Pressure 'drop' as a function of link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dp : float
            pressure drop function
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return 1/C**2*q*np.abs(q)

    def fa(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return q - q_of_dp(p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def fb(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """ Link equation.

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            value of link equation
        """
        return dp(p_start,p_end)-dp_of_q(q,scale_var=scale_var,scale_var_params=scale_var_params)

    def ddp_dp(p_start,p_end):
        """Derivative of pressure drop equation to start and end pressure

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2

        Returns
        -------
        ddp_dp_start : float
            Derivative of pressure drop function to the start pressure
        ddp_dp_end : float
            Derivative of pressure drop function to the end pressure
        """
        return 1.,-1.

    def ddp_dq(q,scale_var=None,scale_var_params=None):
        """Derivative of pressure drop equation to link flow

        Parameters
        ----------
        q : float
            link flow
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        return 1/C**2*2*np.abs(q)

    def dq_ddp(p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link flow to pressure drop function

        Parameters
        ----------
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dq_ddp : float
            Derivative of link flow to pressure drop
        """
        C = pipe_const(scale_var=scale_var,scale_var_params=scale_var_params)
        dp_func = dp(p_start,p_end)
        return 1/2*C/np.sqrt(np.abs(dp_func))

    def dfa_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return -dq_ddp(p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfb_ddp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equation to pressure drop

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None
        """
        return 1.

    def dfa_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfa_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfb_dp(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of link equation to the start and end pressure

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dp_start : float
            Derivative of the link equation to the start pressure
        df_dp_end : float
            Derivative of the link equation to the end pressure
        """
        ddp_dp_start,ddp_dp_end = ddp_dp(p_start,p_end)
        return ddp_dp_start*dfb_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params), ddp_dp_end*dfb_ddp(q,p_start,p_end,scale_var=scale_var,scale_var_params=scale_var_params)

    def dfa_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return 1.

    def dfb_dq(q,p_start,p_end,scale_var=None,scale_var_params=None):
        """Derivative of Link equations to link flow

        Parameters
        ----------
        q : float
            Link flow in kg/s.
        p_start : float
            start pressure in N/m^2
        p_end : float
            end pressure in N/m^2
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dq : float
            Derivative of the link equations to the link flow
        """
        return -ddp_dq(q,scale_var=scale_var,scale_var_params=scale_var_params)

    return pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq

# ===========================================================================
def fric_fac_pole(carrier,D,L,*args,scale_var=None,scale_var_params=None):
    """Friction factor for Pole's equations (for low pressure).

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    scale_var : string, optional
        How to scale the variable. Default is no scaling
    scale_var_params: dict, optional
        Dictionary with values needed for scaling variables. Default is None

    Returns
    -------
    fric_fac : float
        The (Fanning) friction factor
    """
    return 0.0065

def fric_fac_weymouth(carrier,D,L,*args,scale_var=None,scale_var_params=None,E=1,**kwargs):
    """Friction factor for the Weymouth equation (for high pressure).

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    D : float
        Inner pipe diameter in m
    E : float, optional
        Efficiency factor. Default is 1.
    scale_var : string, optional
        How to scale the variable. Default is no scaling
    scale_var_params: dict, optional
        Dictionary with values needed for scaling variables. Default is None

    Returns
    -------
    fric_fac : float
        The (Fanning) friction factor
    """
    return 1/(20.64**2*D**(1/3)*E**2)

def fric_fac_colebrook(carrier,D,L,eps,q,scale_var=None,scale_var_params=None,**kwargs):
    """Colebrook friction factor.

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    eps : float
        relative roughness of the pipe in m
    q : float
        link flow in kg/s
    scale_var : string, optional
        How to scale the variable. Default is no scaling
    scale_var_params: dict, optional
        Dictionary with values needed for scaling variables. Default is None

    Returns
    -------
    fric_fac : float
        The (Fanning) friction factor
    """
    if scale_var == 'per_unit':
        q *= scale_var_params['qbase']
    v = 4*np.abs(q)/(carrier.rhon*np.pi*(D)**2)
    Re = v*D/carrier.mu
    a = 2.51/Re
    b = eps/(3.7*D)
    def fric_fac_implicit(X):
        return X + 2*np.log10(a*X+b)
    X_init = 64/Re
    X = sp.optimize.fsolve(fric_fac_implicit,X_init)
    fric_fac = 1/(4*X[0].real**2)
    return fric_fac

def fric_fac_der_qcolebrook(carrier,D,L,eps,q,scale_var=None,scale_var_params=None,**kwargs):
    """Derivative of Colebrook friction factor to link flow.

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    eps : float
        relative roughness :math:`\\varepsilon` of the pipe in m
    q : float
        link flow in kg/s
    scale_var : string, optional
        How to scale the variable. Default is no scaling
    scale_var_params: dict, optional
        Dictionary with values needed for scaling variables. Default is None

    Returns
    -------
    fric_fac_der_q : float
        The derivative of the (Fanning) friction factor to link flow q
    """
    if scale_var == 'per_unit':
        q *= scale_var_params['qbase']
    v = 4*np.abs(q)/(carrier.rhon*np.pi*(D)**2)
    Re = v*D/carrier.mu
    a = 2.51/Re
    b = eps/(3.7*D)
    def fric_fac_implicit(X):
        return X + 2*np.log10(a*X+b)
    X_init = 64/Re
    X = sp.optimize.fsolve(fric_fac_implicit,X_init)
    X = X[0].real
    fric_fac_der = 1/(4*X*q)*1/((X+b/a)/2*np.log(10)+1)
    return fric_fac_der

def fric_fac_churchill(carrier,D,L,eps,q,scale_var=None,scale_var_params=None,**kwargs):
    """Churchill friction factor. Valid for both laminar and turbulent flow.

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    eps : float
        relative roughness of the pipe in m
    q : float
        link flow in kg/s
    scale_var : string, optional
        How to scale the variable. Default is no scaling
    scale_var_params: dict, optional
        Dictionary with values needed for scaling variables. Default is None

    Returns
    -------
    fric_fac : float
        The (Fanning) friction factor
    """
    if scale_var == 'per_unit':
        q *= scale_var_params['qbase']
    v = 4*np.abs(q)/(carrier.rhon*np.pi*(D)**2)
    Re = v*D/carrier.mu
    theta1 = (-2.457*np.log((7/Re)**(0.9)+0.27*eps/D))**16
    theta2 = (37530/Re)**16
    fric_fac = 2*((8/Re)**12 + 1/(theta1+theta2)**(1.5))**(1/12)
    return fric_fac

def fric_fac_der_qchurchill(carrier,D,L,eps,q,scale_var=None,scale_var_params=None,**kwargs):
    """Derivative of Churchill friction factor to link flow.

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    eps : float
        relative roughness :math:`\\varepsilon` of the pipe in m
    q : float
        link flow in kg/s
    scale_var : string, optional
        How to scale the variable. Default is no scaling
    scale_var_params: dict, optional
        Dictionary with values needed for scaling variables. Default is None

    Returns
    -------
    fric_fac_der_q : float
        The derivative of the (Fanning) friction factor to link flow q
    """
    if scale_var == 'per_unit':
        q *= scale_var_params['qbase']
    v = 4*np.abs(q)/(carrier.rhon*np.pi*(D)**2)
    Re = v*D/carrier.mu
    a = 8/Re
    c = 7/Re
    d = (37530/Re)
    theta1 = (-2.457*np.log(c**(0.9)+0.27*eps/D))**16
    theta2 = d**16
    b = theta1 + theta2
    fric_fac_der = -2/(q*Re)*(a**12+1/b**(1.5))**(1/12-1)*(a**11+1/b**(2.5)*(-theta1/np.log(c**(0.9)+0.27*eps/D)*0.9*c**(-0.1)/(c**(0.9)+0.27*eps/D)-d**5))
    return fric_fac_der

def fric_fac_blasius(carrier,D,L,eps,q,*args,scale_var=None,scale_var_params=None,**kwargs):
    """Blasius friction factor. Valid for turbulent flow: 4000 < Re < 1e5

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    eps : float
        relative roughness of the pipe in m
    q : float
        link flow in kg/s
    scale_var : string, optional
        How to scale the variable. Default is no scaling
    scale_var_params: dict, optional
        Dictionary with values needed for scaling variables. Default is None

    Returns
    -------
    fric_fac : float
        The (Fanning) friction factor
    """
    if scale_var == 'per_unit':
        q *= scale_var_params['qbase']
    v = 4*np.abs(q)/(carrier.rhon*np.pi*(D)**2)
    Re = v*D/carrier.mu
    if Re < 4000 or Re > 1e5:
        warnings.warn("Blasius is only valid for 4000 < Re < 1e5, but Re = {}".format(Re))
    fric_fac = .316/4*Re**(-1/4)
    return fric_fac

def fric_fac_der_qblasius(carrier,D,L,eps,q,*args,scale_var=None,scale_var_params=None,**kwargs):
    """Derivative of Blasius friction factor to link flow.

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    eps : float
        relative roughness :math:`\\varepsilon` of the pipe in m
    q : float
        link flow in kg/s
    scale_var : string, optional
        How to scale the variable. Default is no scaling
    scale_var_params: dict, optional
        Dictionary with values needed for scaling variables. Default is None

    Returns
    -------
    fric_fac_der_q : float
        The derivative of the (Fanning) friction factor to link flow q
    """
    if scale_var == 'per_unit':
        q *= scale_var_params['qbase']
    v = 4*np.abs(q)/(carrier.rhon*np.pi*(D)**2)
    Re = v*D/carrier.mu
    if Re < 4000 or Re > 1e5:
        warnings.warn("Blasius is only valid for 4000 < Re < 1e5, but Re = {}".format(Re))
    fric_fac_der = -.316/(16*q)*Re**(-5/4)
    return fric_fac_der

def fric_fac_hagen_poiseuille(carrier,D,L,eps,q,*args,scale_var=None,scale_var_params=None,**kwargs):
    """Hagen-Poiseuille friction factor. Valid for laminar flow: Re < 2000

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    eps : float
        relative roughness of the pipe in m
    q : float
        link flow in kg/s
    scale_var : string, optional
        How to scale the variable. Default is no scaling
    scale_var_params: dict, optional
        Dictionary with values needed for scaling variables. Default is None

    Returns
    -------
    fric_fac : float
        The (Fanning) friction factor
    """
    if scale_var == 'per_unit':
        q *= scale_var_params['qbase']
    v = 4*np.abs(q)/(carrier.rhon*np.pi*(D)**2)
    Re = v*D/carrier.mu
    if Re > 2000:
        warnings.warn("Hagen-Poiseuille is only valid for Re < 2000, but Re = {}".format(Re))
    fric_fac = 16/Re
    return fric_fac

def fric_fac_der_qhagen_poiseuille(carrier,D,L,eps,q,*args,scale_var=None,scale_var_params=None,**kwargs):
    """Derivative of Hagen-Poiseuille friction factor to link flow.

    Parameters
    ----------
    carrier : Carrier
        Carrier flowing through the pipe
    D : float
        Inner pipe diameter in m
    L : float
        Pipe length in m
    eps : float
        relative roughness :math:`\\varepsilon` of the pipe in m
    q : float
        link flow in kg/s
    scale_var : string, optional
        How to scale the variable. Default is no scaling
    scale_var_params: dict, optional
        Dictionary with values needed for scaling variables. Default is None

    Returns
    -------
    fric_fac_der_q : float
        The derivative of the (Fanning) friction factor to link flow q
    """
    if scale_var == 'per_unit':
        q *= scale_var_params['qbase']
    v = 4*np.abs(q)/(carrier.rhon*np.pi*(D)**2)
    Re = v*D/carrier.mu
    if Re > 2000:
        warnings.warn("Hagen-Poiseuille is only valid for Re < 2000, but Re = {}".format(Re))
    fric_fac_der = -16/(q*Re)
    return fric_fac_der
