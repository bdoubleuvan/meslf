"""Node equations for (heterogeneous) coupling nodes"""
import numpy as np
import warnings

# ===========================================================================
def dummy():
    """Creates all node equations needed for a dummy node.

    Returns
    -------
    Eout_of_Ein : function
        Outgoing energy as function of incoming energy
    Ein_of_Eout : function
        Incoming energy as function of outgoing energy
    dEout_dEin : function
        Derivative of outgoing energy to incoming energy
    dEin_dEout : function
        Derivative of incoming energy to outgoing energy
    node_law_Eout : function
        Node law, using outgoing energy as function of incoming energy
    node_law_Ein : function
        Node law, using incoming energy as function of outgoing energy
    der_node_law_Eout_dEout : function
        Derivative of node law (using outgoing energy as function of incoming energy) to outgoing energy
    der_node_law_Eout_dEin : function
        Derivative of node law (using outgoing energy as function of incoming energy) to incoming energy
    der_node_law_Ein_dEout : function
        Derivative of node law (using incoming energy as function of outgoing energy) to outgoing energy
    der_node_law_Ein_dEin : function
        Derivative of node law (using incoming energy as function of outgoing energy) to incoming energy
    heat_power_eq : function
        Nodal heat power equation
    der_heat_power_eq_dEout : function
        Derivative of nodal heat power equation to outgoing energy
    der_heat_power_eq_dm : function
        Derivative of nodal heat power equation to outgoing flow
    der_heat_power_eq_dTs : function
        Derivative of nodal heat power equation to supply temperature
    der_heat_power_eq_dTr : function
        Derivative of nodal heat power equation to return temperature
    """
    def Eout_of_Ein(Ein):
        """Energy going out of node as a function of energy coming in.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).

        Returns
        -------
        Eout : np array
            Outgoing energies
        """
        pass

    def Ein_of_Eout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Energy coming in to the node as a function of energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).

        Returns
        -------
        Ein : np array
            Incoming energies
        """
        pass

    def dEout_dEin(Ein):
        """Derivative of energy going out of node (as a function of energy coming in) to energy coming in.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        dEout_dEin : np array
            Derevatives of outgoing energies to incoming energies
        """
        pass

    def dEin_dEout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of energy coming in to the node (as a function of energy going out) to energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        dEin_dEout : np array
            Derivatives of incoming energies to outgoing energies
        """
        pass

    def node_law_Eout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses outgoing energy as a function of incoming energy.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        pass

    def node_law_Ein(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses incoming energy as a function of outgoing energy.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        pass

    def der_node_law_Eout_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Eout - Eout(Ein) to Eout
        """
        pass

    def der_node_law_Eout_dEin(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEin : np array
            derivative of Eout - Eout(Ein) to Ein
        """
        pass

    def der_node_law_Eout_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses outgoing energy as a function of incoming energy) to supply temperature

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        pass

    def der_node_law_Ein_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Eout
        """
        pass

    def der_node_law_Ein_dEin(Ein,Eout):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        pass

    def der_node_law_Ein_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outflow energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dTs : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        pass

    def heat_power_eq(Eout,m,Ts,Tr,Cp):
        """Heat power equation for the coupling node, based on conservation of energy, and the assumption that there is one heat half link, and one heat dummy link connected to a coupling node

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        fphi : float
            Nodal heat power equation
        """
        pass

    def der_heat_power_eq_dEout(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing energy

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dEout : float
            Derivative of nodal heat power equation to outgoing energy
        """
        pass

    def der_heat_power_eq_dm(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing flow

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dm : float
            Derivative of nodal heat power equation to water mass flow
        """
        pass

    def der_heat_power_eq_dTs(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outflow temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTs: float
            Derivative of nodal heat power equation to supply temperature
        """
        pass

    def der_heat_power_eq_dTr(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the return temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTr: float
            Derivative of nodal heat power equation to return temperature
        """
        pass

    return Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr

# ===========================================================================
def ge_gas_fired_gen(eta):
    """Creates all node equations needed for an gas-fired generator.

    Parameters
    ----------
    eta : float
        Efficiency of the gas-fired generator

    Returns
    -------
    Eout_of_Ein : function
        Outgoing energy as function of incoming energy
    Ein_of_Eout : function
        Incoming energy as function of outgoing energy
    dEout_dEin : function
        Derivative of outgoing energy to incoming energy
    dEin_dEout : function
        Derivative of incoming energy to outgoing energy
    node_law_Eout : function
        Node law, using outgoing energy as function of incoming energy
    node_law_Ein : function
        Node law, using incoming energy as function of outgoing energy
    der_node_law_Eout_dEout : function
        Derivative of node law (using outgoing energy as function of incoming energy) to outgoing energy
    der_node_law_Eout_dEin : function
        Derivative of node law (using outgoing energy as function of incoming energy) to incoming energy
    der_node_law_Ein_dEout : function
        Derivative of node law (using incoming energy as function of outgoing energy) to outgoing energy
    der_node_law_Ein_dEin : function
        Derivative of node law (using incoming energy as function of outgoing energy) to incoming energy
    heat_power_eq : function
        Nodal heat power equation
    der_heat_power_eq_dEout : function
        Derivative of nodal heat power equation to outgoing energy
    der_heat_power_eq_dm : function
        Derivative of nodal heat power equation to outgoing flow
    der_heat_power_eq_dTs : function
        Derivative of nodal heat power equation to supply temperature
    der_heat_power_eq_dTr : function
        Derivative of nodal heat power equation to return temperature
    """
    def efficiency(eta=eta,scale_var=None,scale_var_params=None):
        """Efficiency of the gas fired generator.

        Parameters
        ----------
        eta : float
            Efficiency of the gas-fired generator
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        C : float
            Efficiency of the gas-fired generator, possibly scaled.
        """
        if scale_var == 'per_unit':
            eta /= scale_var_params['Sbase']/scale_var_params['Ebase']
        return eta

    def Eout_of_Ein(Ein,scale_var=None,scale_var_params=None):
        """Energy going out of node as a function of energy coming in.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        eta : float
            Efficiency of the gas-fired generator

        Returns
        -------
        Eout : float
            Outgoing energy, i.e. active power
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return eta*Ein

    def Ein_of_Eout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Energy coming in to the node as a function of energy going out.

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. active power
        eta : float
            Efficiency of the gas-fired generator

        Returns
        -------
        Ein : float
            Incoming energy, i.e. gas
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return 1/eta*Eout

    def dEout_dEin(Ein,scale_var=None,scale_var_params=None):
        """Derivative of energy going out of node (as a function of energy coming in) to energy coming in.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        dEout_dEin : np array
            Derevatives of outgoing energies to incoming energies
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([eta])

    def dEin_dEout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of energy coming in to the node (as a function of energy going out) to energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        dEin_dEout : np array
            Derivatives of incoming energies to outgoing energies
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([1/eta])

    def node_law_Eout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses outgoing energy as a function of incoming energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : float
            Outgoing energy, i.e. active power
        eta : float
            Efficiency of the gas-fired generator

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return Eout - Eout_of_Ein(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def node_law_Ein(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses incoming energy as a function of outgoing energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : float
            Outgoing energy, i.e. active power
        eta : float
            Efficiency of the gas-fired generator

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return Ein - Ein_of_Eout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Eout - Eout(Ein) to Eout
        """
        return np.array([1.])

    def der_node_law_Eout_dEin(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEin : np array
            derivative of Eout - Eout(Ein) to Ein
        """
        return -dEout_dEin(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses outgoing energy as a function of incoming energy) to supply temperature

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def der_node_law_Ein_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Eout
        """
        return -dEin_dEout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Ein_dEin(Ein,Eout):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.eye(len(Ein))

    def der_node_law_Ein_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outflow energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dTs : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def heat_power_eq(Eout,m,Ts,Tr,Cp):
        """Heat power equation for the coupling node, based on conservation of energy, and the assumption that there is one heat half link, and one heat dummy link connected to a coupling node

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        fphi : float
            Nodal heat power equation
        """
        pass

    def der_heat_power_eq_dEout(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing energy

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dEout : float
            Derivative of nodal heat power equation to outgoing energy
        """
        pass

    def der_heat_power_eq_dm(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing flow

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dm : float
            Derivative of nodal heat power equation to water mass flow
        """
        pass

    def der_heat_power_eq_dTs(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outflow temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTs: float
            Derivative of nodal heat power equation to supply temperature
        """
        pass

    def der_heat_power_eq_dTr(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the return temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTr: float
            Derivative of nodal heat power equation to return temperature
        """
        pass

    return Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr

# ===========================================================================
def ge_gas_fired_gen_valve_point(a,b,c,d,e,Pmin):
    """Creates all node equations needed for an gas-fired generator with valve point effect:
    E^g = a*E^P**2 + b*E^P + c + np.abs(d*np.sin(e*(Pmin-E^P))

    Parameters
    ----------
    a : float
        Valve point effect parameter (assuming E^g and P in SI units)
    b : float
        Valve point effect parameter (assuming E^g and P in SI units)
    c : float
        Valve point effect parameter (assuming E^g and P in SI units)
    d : float
        Valve point effect parameter (assuming E^g and P in SI units)
    e : float
        Valve point effect parameter (assuming E^g and P in SI units)
    Pmin : float
        Minimum amount of generated active power.

    Returns
    -------
    Eout_of_Ein : function
        Outgoing energy as function of incoming energy
    Ein_of_Eout : function
        Incoming energy as function of outgoing energy
    dEout_dEin : function
        Derivative of outgoing energy to incoming energy
    dEin_dEout : function
        Derivative of incoming energy to outgoing energy
    node_law_Eout : function
        Node law, using outgoing energy as function of incoming energy
    node_law_Ein : function
        Node law, using incoming energy as function of outgoing energy
    der_node_law_Eout_dEout : function
        Derivative of node law (using outgoing energy as function of incoming energy) to outgoing energy
    der_node_law_Eout_dEin : function
        Derivative of node law (using outgoing energy as function of incoming energy) to incoming energy
    der_node_law_Ein_dEout : function
        Derivative of node law (using incoming energy as function of outgoing energy) to outgoing energy
    der_node_law_Ein_dEin : function
        Derivative of node law (using incoming energy as function of outgoing energy) to incoming energy
    heat_power_eq : function
        Nodal heat power equation
    der_heat_power_eq_dEout : function
        Derivative of nodal heat power equation to outgoing energy
    der_heat_power_eq_dm : function
        Derivative of nodal heat power equation to outgoing flow
    der_heat_power_eq_dTs : function
        Derivative of nodal heat power equation to supply temperature
    der_heat_power_eq_dTr : function
        Derivative of nodal heat power equation to return temperature
    """
    def paramaters(a=a,b=b,c=c,d=d,e=e,Pmin=Pmin,scale_var=None,scale_var_params=None):
        """Valve point effect parameters

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        a : float
            Valve point effect parameter, unscaled (assuming E^g and P in SI units)
        b : float
            Valve point effect parameter, unscaled (assuming E^g and P in SI units)
        c : float
            Valve point effect parameter, unscaled (assuming E^g and P in SI units)
        d : float
            Valve point effect parameter, unscaled (assuming E^g and P in SI units)
        e : float
            Valve point effect parameter, unscaled (assuming E^g and P in SI units)

        Returns
        -------
        a : float
            Valve point effect parameter, scaled
        b : float
            Valve point effect parameter, scaled
        c : float
            Valve point effect parameter, scaled
        d : float
            Valve point effect parameter, scaled
        e : float
            Valve point effect parameter, scaled
        """
        if scale_var == 'per_unit':
            Pb = scale_var_params['Sbase']
            Eb = scale_var_params['Ebase']
            a *= Pb**2/Eb
            b *= Pb/Eb
            c *= 1/Eb
            d *= 1/Eb
            e *= Pb
            Pmin *= 1/Pb
        return a,b,c,d,e,Pmin

    def Eout_of_Ein(Ein,scale_var=None,scale_var_params=None):
        """Energy going out of node as a function of energy coming in. NB. not defined for this unit

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        a : float
            Valve point effect parameter
        b : float
            Valve point effect parameter
        c : float
            Valve point effect parameter
        d : float
            Valve point effect parameter
        e : float
            Valve point effect parameter
        Pmin : float
            Minimum amount of generated active power.

        Returns
        -------
        Eout : float
            Outgoing energy, i.e. active power
        """
        pass

    def Ein_of_Eout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Energy coming in to the node as a function of energy going out.

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. active power
        a : float
            Valve point effect parameter
        b : float
            Valve point effect parameter
        c : float
            Valve point effect parameter
        d : float
            Valve point effect parameter
        e : float
            Valve point effect parameter
        Pmin : float
            Minimum amount of generated active power.

        Returns
        -------
        Ein : float
            Incoming energy, i.e. gas
        """
        a,b,c,d,e,Pmin = paramaters(scale_var=scale_var,scale_var_params=scale_var_params)
        return a*Eout**2 + b*Eout + c + np.abs(d*np.sin(e*(Pmin-Eout)))

    def dEout_dEin(Ein,scale_var=None,scale_var_params=None):
        """Derivative of energy going out of node (as a function of energy coming in) to energy coming in.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        dEout_dEin : np array
            Derevatives of outgoing energies to incoming energies
        """
        return np.array([0.])

    def dEin_dEout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of energy coming in to the node (as a function of energy going out) to energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        dEin_dEout : np array
            Derivatives of incoming energies to outgoing energies
        """
        a,b,c,d,e,Pmin = paramaters(scale_var=scale_var,scale_var_params=scale_var_params)
        return 2*a*Eout + b + np.sign(d*np.sin(e*(Pmin-Eout)))*np.abs(d*np.cos(e*(Pmin-Eout)))*-e

    def node_law_Eout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses outgoing energy as a function of incoming energy. NB. Not defined for this unit.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : float
            Outgoing energy, i.e. active power
        a : float
            Valve point effect parameter
        b : float
            Valve point effect parameter
        c : float
            Valve point effect parameter
        d : float
            Valve point effect parameter
        e : float
            Valve point effect parameter
        Pmin : float
            Minimum amount of generated active power.

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return Eout - Eout_of_Ein(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def node_law_Ein(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses incoming energy as a function of outgoing energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : float
            Outgoing energy, i.e. active power
        a : float
            Valve point effect parameter
        b : float
            Valve point effect parameter
        c : float
            Valve point effect parameter
        d : float
            Valve point effect parameter
        e : float
            Valve point effect parameter
        Pmin : float
            Minimum amount of generated active power.

        Returns
        -------
        f : np array
            Node law, Ein - Ein(Eout)
        """
        return Ein - Ein_of_Eout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Eout - Eout(Ein) to Eout
        """
        return np.array([1.])

    def der_node_law_Eout_dEin(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEin : np array
            derivative of Eout - Eout(Ein) to Ein
        """
        return -dEout_dEin(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses outgoing energy as a function of incoming energy) to supply temperature

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def der_node_law_Ein_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Eout
        """
        return -dEin_dEout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Ein_dEin(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.array([1.])

    def der_node_law_Ein_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outflow energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dTs : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def heat_power_eq(Eout,m,Ts,Tr,Cp,scale_var=None,scale_var_params=None):
        """Heat power equation for the coupling node, based on conservation of energy, and the assumption that there is one heat half link, and one heat dummy link connected to a coupling node

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        fphi : float
            Nodal heat power equation
        """
        pass

    def der_heat_power_eq_dEout(Eout,m,Ts,Tr,Cp,scale_var=None,scale_var_params=None):
        """Derivative of heat power equation to the outgoing energy

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dEout : float
            Derivative of nodal heat power equation to outgoing energy
        """
        pass

    def der_heat_power_eq_dm(Eout,m,Ts,Tr,Cp,scale_var=None,scale_var_params=None):
        """Derivative of heat power equation to the outgoing flow

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dm : float
            Derivative of nodal heat power equation to water mass flow
        """
        pass

    def der_heat_power_eq_dTs(Eout,m,Ts,Tr,Cp,a=a,b=b,c=c,d=d,e=e,Pmin=Pmin):
        """Derivative of heat power equation to the outflow temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTs: float
            Derivative of nodal heat power equation to supply temperature
        """
        pass

    def der_heat_power_eq_dTr(Eout,m,Ts,Tr,Cp,a=a,b=b,c=c,d=d,e=e,Pmin=Pmin):
        """Derivative of heat power equation to the return temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTr: float
            Derivative of nodal heat power equation to return temperature
        """
        pass

    return Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr

# ===========================================================================
def gh_gas_boiler(eta):
    """Creates all node equations needed for a gas boiler.

    Parameters
    ----------
    eta : float
        Efficiency of the gas boiler

    Returns
    -------
    Eout_of_Ein : function
        Outgoing energy as function of incoming energy
    Ein_of_Eout : function
        Incoming energy as function of outgoing energy
    dEout_dEin : function
        Derivative of outgoing energy to incoming energy
    dEin_dEout : function
        Derivative of incoming energy to outgoing energy
    node_law_Eout : function
        Node law, using outgoing energy as function of incoming energy
    node_law_Ein : function
        Node law, using incoming energy as function of outgoing energy
    der_node_law_Eout_dEout : function
        Derivative of node law (using outgoing energy as function of incoming energy) to outgoing energy
    der_node_law_Eout_dEin : function
        Derivative of node law (using outgoing energy as function of incoming energy) to incoming energy
    der_node_law_Ein_dEout : function
        Derivative of node law (using incoming energy as function of outgoing energy) to outgoing energy
    der_node_law_Ein_dEin : function
        Derivative of node law (using incoming energy as function of outgoing energy) to incoming energy
    heat_power_eq : function
        Nodal heat power equation
    der_heat_power_eq_dEout : function
        Derivative of nodal heat power equation to outgoing energy
    der_heat_power_eq_dm : function
        Derivative of nodal heat power equation to outgoing flow
    der_heat_power_eq_dTs : function
        Derivative of nodal heat power equation to supply temperature
    der_heat_power_eq_dTr : function
        Derivative of nodal heat power equation to return temperature
    """
    def efficiency(eta=eta,scale_var=None,scale_var_params=None):
        """Efficiency of the gas boiler.

        Parameters
        ----------
        eta : float
            Efficiency of the gas-fired generator
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        eta : float
            Efficiency of the gas-fired generator, possibly scaled.
        """
        if scale_var == 'per_unit':
            eta /= scale_var_params['phibase']/scale_var_params['Ebase']
        return eta

    def Eout_of_Ein(Ein,scale_var=None,scale_var_params=None):
        """Energy going out of node as a function of energy coming in.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas

        Returns
        -------
        Eout : float
            Outgoing energy, i.e. heat
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([eta])*Ein

    def Ein_of_Eout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Energy coming in to the node as a function of energy going out.

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat

        Returns
        -------
        Ein : float
            Incoming energy, i.e. gas
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return 1/eta*Eout

    def dEout_dEin(Ein,scale_var=None,scale_var_params=None):
        """Derivative of energy going out of node (as a function of energy coming in) to energy coming in.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).

        Returns
        -------
        dEout_dEin : np array
            Derevatives of outgoing energies to incoming energies
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([eta])

    def dEin_dEout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of energy coming in to the node (as a function of energy going out) to energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).

        Returns
        -------
        dEin_dEout : np array
            Derivatives of incoming energies to outgoing energies
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([1/eta])

    def node_law_Eout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses outgoing energy as a function of incoming energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : float
            Outgoing energy, i.e. heat

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return Eout - Eout_of_Ein(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def node_law_Ein(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses incoming energy as a function of outgoing energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : float
            Outgoing energy, i.e. heat

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return Ein - Ein_of_Eout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.

        Returns
        -------
        df_dEout : np array
            derivative of Eout - Eout(Ein) to Eout
        """
        return np.array([1.])

    def der_node_law_Eout_dEin(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.

        Returns
        -------
        df_dEin : np array
            derivative of Eout - Eout(Ein) to Ein
        """
        return -dEout_dEin(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses outgoing energy as a function of incoming energy) to supply temperature

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def der_node_law_Ein_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Eout
        """
        return -dEin_dEout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Ein_dEin(Ein,Eout):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.array([1.])

    def der_node_law_Ein_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outflow energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dTs : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def heat_power_eq(Eout,m,Ts,Tr,Cp):
        """Heat power equation for the coupling node, based on conservation of energy, and the assumption that there is one heat half link, and one heat dummy link connected to a coupling node

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        fphi : float
            Nodal heat power equation
        """
        return -Eout + Cp*(m*(Ts-Tr))

    def der_heat_power_eq_dEout(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing energy

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dEout : float
            Derivative of nodal heat power equation to outgoing energy
        """
        return -1.

    def der_heat_power_eq_dm(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing flow

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dm : float
            Derivative of nodal heat power equation to water mass flow
        """
        return Cp*(Ts-Tr)

    def der_heat_power_eq_dTs(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outflow temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTs: float
            Derivative of nodal heat power equation to supply temperature
        """
        return Cp*m

    def der_heat_power_eq_dTr(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the return temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTr: float
            Derivative of nodal heat power equation to return temperature
        """
        return -Cp*m

    return Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr

# ===========================================================================
def gh_gas_boiler_part_load(a,b,Ess):
    """Creates all node equations needed for a gas boiler, taking part load effect into account.

    Parameters
    ----------
    a : float
        Part-load effect parameter
    b : float
        Part-load effect parameter
    Ess : float
        Steady-state operation energy (input?)

    Returns
    -------
    Eout_of_Ein : function
        Outgoing energy as function of incoming energy
    Ein_of_Eout : function
        Incoming energy as function of outgoing energy
    dEout_dEin : function
        Derivative of outgoing energy to incoming energy
    dEin_dEout : function
        Derivative of incoming energy to outgoing energy
    node_law_Eout : function
        Node law, using outgoing energy as function of incoming energy
    node_law_Ein : function
        Node law, using incoming energy as function of outgoing energy
    der_node_law_Eout_dEout : function
        Derivative of node law (using outgoing energy as function of incoming energy) to outgoing energy
    der_node_law_Eout_dEin : function
        Derivative of node law (using outgoing energy as function of incoming energy) to incoming energy
    der_node_law_Ein_dEout : function
        Derivative of node law (using incoming energy as function of outgoing energy) to outgoing energy
    der_node_law_Ein_dEin : function
        Derivative of node law (using incoming energy as function of outgoing energy) to incoming energy
    heat_power_eq : function
        Nodal heat power equation
    der_heat_power_eq_dEout : function
        Derivative of nodal heat power equation to outgoing energy
    der_heat_power_eq_dm : function
        Derivative of nodal heat power equation to outgoing flow
    der_heat_power_eq_dTs : function
        Derivative of nodal heat power equation to supply temperature
    der_heat_power_eq_dTr : function
        Derivative of nodal heat power equation to return temperature
    """
    def paramaters(a=a,b=b,Ess=Ess,scale_var=None,scale_var_params=None):
        """Part load effect parameters.

        Parameters
        ----------
        a : float
            Part-load effect parameter
        b : float
            Part-load effect parameter
        Ess : float
            Steady-state operation energy (input?)
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        a : float
            Part-load effect parameter, possibly scaled
        b : float
            Part-load effect parameter, possibly scaled
        Ess : float
            Steady-state operation energy (input?), possibly scaled
        """
        if scale_var == 'per_unit':
            b /= scale_var_params['phibase']/scale_var_params['Ebase']
            Ess /= scale_var_params['phibase']
        return a,b,Ess

    def Eout_of_Ein(Ein,scale_var=None,scale_var_params=None):
        """Energy going out of node as a function of energy coming in.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas

        Returns
        -------
        Eout : float
            Outgoing energy, i.e. heat
        """
        a,b,Ess = paramaters(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([b])*Ein - a*Ess

    def Ein_of_Eout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Energy coming in to the node as a function of energy going out.

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat

        Returns
        -------
        Ein : float
            Incoming energy, i.e. gas
        """
        a,b,Ess = paramaters(scale_var=scale_var,scale_var_params=scale_var_params)
        return 1/b*(Eout + a*Ess)

    def dEout_dEin(Ein,scale_var=None,scale_var_params=None):
        """Derivative of energy going out of node (as a function of energy coming in) to energy coming in.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).

        Returns
        -------
        dEout_dEin : np array
            Derevatives of outgoing energies to incoming energies
        """
        a,b,Ess = paramaters(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([b])

    def dEin_dEout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of energy coming in to the node (as a function of energy going out) to energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).

        Returns
        -------
        dEin_dEout : np array
            Derivatives of incoming energies to outgoing energies
        """
        a,b,Ess = paramaters(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([1/b])

    def node_law_Eout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses outgoing energy as a function of incoming energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : float
            Outgoing energy, i.e. heat

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return Eout - Eout_of_Ein(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def node_law_Ein(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses incoming energy as a function of outgoing energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : float
            Outgoing energy, i.e. heat

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return Ein - Ein_of_Eout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.

        Returns
        -------
        df_dEout : np array
            derivative of Eout - Eout(Ein) to Eout
        """
        return np.array([1.])

    def der_node_law_Eout_dEin(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.

        Returns
        -------
        df_dEin : np array
            derivative of Eout - Eout(Ein) to Ein
        """
        return -dEout_dEin(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses outgoing energy as a function of incoming energy) to supply temperature

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def der_node_law_Ein_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Eout
        """
        return -dEin_dEout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Ein_dEin(Ein,Eout):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.array([1.])

    def der_node_law_Ein_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outflow energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dTs : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def heat_power_eq(Eout,m,Ts,Tr,Cp):
        """Heat power equation for the coupling node, based on conservation of energy, and the assumption that there is one heat half link, and one heat dummy link connected to a coupling node

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        fphi : float
            Nodal heat power equation
        """
        return -Eout + Cp*(m*(Ts-Tr))

    def der_heat_power_eq_dEout(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing energy

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dEout : float
            Derivative of nodal heat power equation to outgoing energy
        """
        return -1.

    def der_heat_power_eq_dm(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing flow

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dm : float
            Derivative of nodal heat power equation to water mass flow
        """
        return Cp*(Ts-Tr)

    def der_heat_power_eq_dTs(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outflow temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTs: float
            Derivative of nodal heat power equation to supply temperature
        """
        return Cp*m

    def der_heat_power_eq_dTr(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the return temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTr: float
            Derivative of nodal heat power equation to return temperature
        """
        return -Cp*m

    return Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr

# ===========================================================================
def eh_elec_boiler(eta):
    """Creates all node equations needed for an electrical boiler.

    Parameters
    ----------
    eta : float
        Efficiency of the electrical boiler

    Returns
    -------
    Eout_of_Ein : function
        Outgoing energy as function of incoming energy
    Ein_of_Eout : function
        Incoming energy as function of outgoing energy
    dEout_dEin : function
        Derivative of outgoing energy to incoming energy
    dEin_dEout : function
        Derivative of incoming energy to outgoing energy
    node_law_Eout : function
        Node law, using outgoing energy as function of incoming energy
    node_law_Ein : function
        Node law, using incoming energy as function of outgoing energy
    der_node_law_Eout_dEout : function
        Derivative of node law (using outgoing energy as function of incoming energy) to outgoing energy
    der_node_law_Eout_dEin : function
        Derivative of node law (using outgoing energy as function of incoming energy) to incoming energy
    der_node_law_Ein_dEout : function
        Derivative of node law (using incoming energy as function of outgoing energy) to outgoing energy
    der_node_law_Ein_dEin : function
        Derivative of node law (using incoming energy as function of outgoing energy) to incoming energy
    heat_power_eq : function
        Nodal heat power equation
    der_heat_power_eq_dEout : function
        Derivative of nodal heat power equation to outgoing energy
    der_heat_power_eq_dm : function
        Derivative of nodal heat power equation to outgoing flow
    der_heat_power_eq_dTs : function
        Derivative of nodal heat power equation to supply temperature
    der_heat_power_eq_dTr : function
        Derivative of nodal heat power equation to return temperature
    """
    def efficiency(eta=eta,scale_var=None,scale_var_params=None):
        """Efficiency of the electrical boiler.

        Parameters
        ----------
        eta : float
            Efficiency of the electrical boiler
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        C : float
            Efficiency of the electrical boiler, possibly scaled.
        """
        if scale_var == 'per_unit':
            eta /= scale_var_params['Sbase']/scale_var_params['phibase']
        return eta

    def Eout_of_Ein(Ein,scale_var=None,scale_var_params=None):
        """Energy going out of node as a function of energy coming in.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. active power
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        Eout : float
            Outgoing energy, i.e. heat
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return eta*Ein

    def Ein_of_Eout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Energy coming in to the node as a function of energy going out.

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        Ein : float
            Incoming energy, i.e. active power
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return 1/eta*Eout

    def dEout_dEin(Ein,scale_var=None,scale_var_params=None):
        """Derivative of energy going out of node (as a function of energy coming in) to energy coming in.

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        dEout_dEin : np array
            Derevatives of outgoing energies to incoming energies
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([eta])

    def dEin_dEout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of energy coming in to the node (as a function of energy going out) to energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies.
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        dEin_dEout : np array
            Derivatives of incoming energies to outgoing energies
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([1/eta])

    def node_law_Eout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses outgoing energy as a function of incoming energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. active power
        Eout : float
            Outgoing energy, i.e. heat
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return Eout - Eout_of_Ein(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def node_law_Ein(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses incoming energy as a function of outgoing energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. active power
        Eout : float
            Outgoing energy, i.e. heat
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return Ein - Ein_of_Eout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        df_dEout : np array
            derivative of Eout - Eout(Ein) to Eout
        """
        return np.array([1.])

    def der_node_law_Eout_dEin(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        df_dEin : np array
            derivative of Eout - Eout(Ein) to Ein
        """
        return -dEout_dEin(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses outgoing energy as a function of incoming energy) to supply temperature

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def der_node_law_Ein_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Eout
        """
        return -dEin_dEout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Ein_dEin(Ein,Eout):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.array([1.])

    def der_node_law_Ein_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outflow energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dTs : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def heat_power_eq(Eout,m,Ts,Tr,Cp):
        """Heat power equation for the coupling node, based on conservation of energy, and the assumption that there is one heat half link, and one heat dummy link connected to a coupling node

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        fphi : float
            Nodal heat power equation
        """
        return -Eout + Cp*(m*(Ts-Tr))

    def der_heat_power_eq_dEout(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing energy

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dEout : float
            Derivative of nodal heat power equation to outgoing energy
        """
        return -1.

    def der_heat_power_eq_dm(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing flow

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dm : float
            Derivative of nodal heat power equation to water mass flow
        """
        return Cp*(Ts-Tr)

    def der_heat_power_eq_dTs(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outflow temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTs: float
            Derivative of nodal heat power equation to supply temperature
        """
        return Cp*m

    def der_heat_power_eq_dTr(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the return temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTr: float
            Derivative of nodal heat power equation to return temperature
        """
        return -Cp*m

    return Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr

# ===========================================================================
def geh_CHP(eta):
    """Creates all node equations needed for a combined heat and power plant (CHP).

    Parameters
    ----------
    eta : np array
        Efficiency of the CHP [eta_ge eta_gh]

    Returns
    -------
    Eout_of_Ein : function
        Outgoing energy as function of incoming energy
    Ein_of_Eout : function
        Incoming energy as function of outgoing energy
    dEout_dEin : function
        Derivative of outgoing energy to incoming energy
    dEin_dEout : function
        Derivative of incoming energy to outgoing energy
    node_law_Eout : function
        Node law, using outgoing energy as function of incoming energy
    node_law_Ein : function
        Node law, using incoming energy as function of outgoing energy
    der_node_law_Eout_dEout : function
        Derivative of node law (using outgoing energy as function of incoming energy) to outgoing energy
    der_node_law_Eout_dEin : function
        Derivative of node law (using outgoing energy as function of incoming energy) to incoming energy
    der_node_law_Ein_dEout : function
        Derivative of node law (using incoming energy as function of outgoing energy) to outgoing energy
    der_node_law_Ein_dEin : function
        Derivative of node law (using incoming energy as function of outgoing energy) to incoming energy
    heat_power_eq : function
        Nodal heat power equation
    der_heat_power_eq_dEout : function
        Derivative of nodal heat power equation to outgoing energy
    der_heat_power_eq_dm : function
        Derivative of nodal heat power equation to outgoing flow
    der_heat_power_eq_dTs : function
        Derivative of nodal heat power equation to supply temperature
    der_heat_power_eq_dTr : function
        Derivative of nodal heat power equation to return temperature
    """
    def efficiency(eta=eta,scale_var=None,scale_var_params=None):
        """Efficiency of the CHP.

        Parameters
        ----------
        eta : float
            Efficiency
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        eta : float
            Efficiency, possibly scaled.
        """
        if scale_var == 'per_unit':
            eta_b = np.array([scale_var_params['Sbase']/scale_var_params['Ebase'], scale_var_params['phibase']/scale_var_params['Ebase']])
            eta = eta/eta_b
        return eta

    def Eout_of_Ein(Ein,scale_var=None,scale_var_params=None):
        """Energy going out of node as a function of energy coming in. NB. returns total outgoing energy

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        eta : float
            Efficiency of the CHP

        Returns
        -------
        Eout : float
            Tstal outgoing energy
        """
        pass

    def Ein_of_Eout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Energy coming in to the node as a function of energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies, i.e. active power and heat
        eta : float
            Efficiency of the CHP

        Returns
        -------
        Ein : float
            Incoming energy, i.e. gas
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.sum(1/eta*Eout)

    def dEout_dEin(Ein,scale_var=None,scale_var_params=None):
        """Derivative of energy going out of node (as a function of energy coming in) to energy coming in.

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        dEout_dEin : np array
            Derevatives of outgoing energies to incoming energies
        """
        pass

    def dEin_dEout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of energy coming in to the node (as a function of energy going out) to energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies.
        eta : float
            Efficiency of the electrical boiler

        Returns
        -------
        dEin_dEout : np array
            Derivatives of incoming energies to outgoing energies
        """
        eta = efficiency(scale_var=scale_var,scale_var_params=scale_var_params)
        return 1/eta

    def node_law_Eout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses outgoing energy as a function of incoming energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : np array
            Outgoing energies, i.e. active power and heat
        eta : float
            Efficiency of the CHP

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return np.sum(Eout) - Eout_of_Ein(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def node_law_Ein(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses incoming energy as a function of outgoing energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : np array
            Outgoing energies, i.e. active power and heat
        eta : float
            Efficiency of the CHP

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return Ein - Ein_of_Eout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        eta : float
            Efficiency of the CHP

        Returns
        -------
        df_dEout : np array
            derivative of Eout - Eout(Ein) to Eout
        """
        return np.array([1,1])

    def der_node_law_Eout_dEin(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        eta : float
            Efficiency of the CHP

        Returns
        -------
        df_dEin : np array
            derivative of Eout - Eout(Ein) to Ein
        """
        return -dEout_dEin(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses outgoing energy as a function of incoming energy) to supply temperature

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def der_node_law_Ein_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        eta : float
            Efficiency of the CHP

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Eout
        """
        return -dEin_dEout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Ein_dEin(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        eta : float
            Efficiency of the CHP

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.array([1.])

    def der_node_law_Ein_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outflow energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dTs : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return 0

    def heat_power_eq(Eout,m,Ts,Tr,Cp):
        """Heat power equation for the coupling node, based on conservation of energy, and the assumption that there is one heat half link, and one heat dummy link connected to a coupling node

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        fphi : float
            Nodal heat power equation
        """
        return -Eout + Cp*(m*(Ts-Tr))

    def der_heat_power_eq_dEout(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing energy

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dEout : float
            Derivative of nodal heat power equation to outgoing energy
        """
        return -1.

    def der_heat_power_eq_dm(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing flow

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dm : float
            Derivative of nodal heat power equation to water mass flow
        """
        return Cp*(Ts-Tr)

    def der_heat_power_eq_dTs(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outflow temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTs: float
            Derivative of nodal heat power equation to supply temperature
        """
        return Cp*m

    def der_heat_power_eq_dTr(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the return temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTr: float
            Derivative of nodal heat power equation to return temperature
        """
        return -Cp*m

    return Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr

# ===========================================================================
def geh_CHP_part_load(eta,a,b,d,L1,L2,r1,r2,phimin,phimax):
    """Creates all node equations needed for a combined heat and power plant (CHP), taken part load effect into account. NB. this is a not a continuous function.

    Parameters
    ----------
    eta : float
        Tstal efficiency of the CHP
    a : float
        Parameter of CHP
    b : float
        Parameter of CHP
    d : float
        Parameter of CHP
    L1 : float
        Upper limit to determine which part load effect is taken into account. Fraction of phimax, so 0<=L2<L1<=1 (with 0 indicating no load, and 1 indicating full load)
    L2 : float
        Lower limit to determine which part load effect is taken into account. Fraction of phimax, so 0<=L2<L1<=1 (with 0 indicating no load, and 1 indicating full load)
    r1 : float
        Parameter of part load effect of CHP
    r2 : float
        Parameter of part load effect of CHP
    phimin : float
        Minimum heat demand of CHP.
    phimax : float
        Maximum heat power produces by CHP.

    Returns
    -------
    Eout_of_Ein : function
        Outgoing energy as function of incoming energy
    Ein_of_Eout : function
        Incoming energy as function of outgoing energy
    dEout_dEin : function
        Derivative of outgoing energy to incoming energy
    dEin_dEout : function
        Derivative of incoming energy to outgoing energy
    node_law_Eout : function
        Node law, using outgoing energy as function of incoming energy
    node_law_Ein : function
        Node law, using incoming energy as function of outgoing energy
    der_node_law_Eout_dEout : function
        Derivative of node law (using outgoing energy as function of incoming energy) to outgoing energy
    der_node_law_Eout_dEin : function
        Derivative of node law (using outgoing energy as function of incoming energy) to incoming energy
    der_node_law_Eout_dTs : function
        Derivative of node law (using outgoing energy as function of incoming energy) to supply temperature
    der_node_law_Ein_dEout : function
        Derivative of node law (using incoming energy as function of outgoing energy) to outgoing energy
    der_node_law_Ein_dEin : function
        Derivative of node law (using incoming energy as function of outgoing energy) to incoming energy
    der_node_law_Ein_dEin : function
        Derivative of node law (using incoming energy as function of outgoing energy) to supply temperature
    heat_power_eq : function
        Nodal heat power equation
    der_heat_power_eq_dEout : function
        Derivative of nodal heat power equation to outgoing energy
    der_heat_power_eq_dm : function
        Derivative of nodal heat power equation to outgoing flow
    der_heat_power_eq_dTs : function
        Derivative of nodal heat power equation to supply temperature
    der_heat_power_eq_dTr : function
        Derivative of nodal heat power equation to return temperature
    """
    def parameters(eta=eta,a=a,b=b,d=d,L1=L1,L2=L2,r1=r1,r2=r2,phimin=phimin,phimax=phimax,scale_var=None,scale_var_params=None):
        """Parameters for the CHP with valve point effect. NB. per unit scaling is not implemented!

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        eta : float
            Tstal efficiency of the CHP
        a : float
            Parameter of CHP
        b : float
            Parameter of CHP
        d : float
            Parameter of CHP
        L1 : float
            Upper limit to determine which part load effect is taken into account. Fraction of phimax, so 0<=L2<L1<=1 (with 0 indicating no load, and 1 indicating full load)
        L2 : float
            Lower limit to determine which part load effect is taken into account. Fraction of phimax, so 0<=L2<L1<=1 (with 0 indicating no load, and 1 indicating full load)
        r1 : float
            Parameter of part load effect of CHP
        r2 : float
            Parameter of part load effect of CHP
        phimin : float
            Minimum heat demand of CHP.
        phimax : float
            Maximum heat power produces by CHP.

        Returns
        -------
        eta : float
            Tstal efficiency of the CHP, possibly scaled
        a : float
            Parameter of CHP, possibly scaled
        b : float
            Parameter of CHP
        d : float
            Parameter of CHP
        L1 : float
            Upper limit to determine which part load effect is taken into account. Fraction of phimax, so 0<=L2<L1<=1 (with 0 indicating no load, and 1 indicating full load)
        L2 : float
            Lower limit to determine which part load effect is taken into account. Fraction of phimax, so 0<=L2<L1<=1 (with 0 indicating no load, and 1 indicating full load)
        r1 : float
            Parameter of part load effect of CHP
        r2 : float
            Parameter of part load effect of CHP
        phimin : float
            Minimum heat demand of CHP.
        phimax : float
            Maximum heat power produces by CHP.
        """
        if scale_var == 'per_unit':
            eta_b = np.array([scale_var_params['Sbase']/scale_var_params['Ebase'], scale_var_params['phibase']/scale_var_params['Ebase']])
            eta = eta/eta_b
            a /= scale_var_params.get('Sbase')/scale_var.get('phibase')
            b /= scale_var_params.get('Sbase')/scale_var.get('Tbase')
            d /= scale_var_params.get('Sbase')
            rb = scale_var_params.get('Sbase')/scale_var.get('phibase')
            r1 /= rb
            r2 /= rb
            phimin /= scale_var_params.get('phibase')
            phimax /= scale_var_params.get('phibase')
        return eta,a,b,d,L1,L2,r1,r2,phimin,phimax

    def Eout_of_Ein(Ein,scale_var=None,scale_var_params=None):
        """Energy going out of node as a function of energy coming in. NB. returns total outgoing energy

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        eta : float
            Efficiency of the CHP

        Returns
        -------
        Eout : float
            Tstal outgoing energy
        """
        pass

    def Ein_of_Eout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Energy coming in to the node as a function of energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies, i.e. active power and heat
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        Ein : float
            Incoming energy, i.e. gas
        """
        eta,a,b,d,L1,L2,r1,r2,phimin,phimax = parameters(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.sum(1/eta*Eout)

    def dEout_dEin(Ein,scale_var=None,scale_var_params=None):
        """Derivative of energy going out of node (as a function of energy coming in) to energy coming in.

        Parameters
        ----------
        Ein : np array
            Incoming energy.

        Returns
        -------
        dEout_dEin : np array
            Derevatives of outgoing energies to incoming energies
        """
        pass

    def dEin_dEout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of energy coming in to the node (as a function of energy going out) to energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        dEin_dEout : np array
            Derivatives of incoming energies to outgoing energies
        """
        eta,a,b,d,L1,L2,r1,r2,phimin,phimax = parameters(scale_var=scale_var,scale_var_params=scale_var_params)
        return 1/eta

    def node_law_Eout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses outgoing energy as a function of incoming energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : np array
            Outgoing energies, i.e. active power and heat
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return np.sum(Eout) - Eout_of_Ein(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def node_law_Ein(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses incoming energy as a function of outgoing energy.

        Parameters
        ----------
        Ein : float
            Incoming energy, i.e. gas
        Eout : np array
            Outgoing energies, i.e. active power and heat
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        eta,a,b,d,L1,L2,r1,r2,phimin,phimax = parameters(scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([Ein - Ein_of_Eout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params), Eout[0]-power_to_heat_ratio(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)])

    def der_node_law_Eout_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Eout - Eout(Ein) to Eout
        """
        return np.array([1,1])

    def der_node_law_Eout_dEin(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEin : np array
            derivative of Eout - Eout(Ein) to Ein
        """
        return -dEout_dEin(Ein,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses outgoing energy as a function of incoming energy) to supply temperature

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.array([0])

    def der_node_law_Ein_dEout(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Eout
        """
        return np.vstack([-dEin_dEout(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params),(np.array([1, 0])-der_power_to_heat_ratio_dEout (Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params))])

    def der_node_law_Ein_dEin(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.array([[1.],[0]])

    def der_node_law_Ein_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outflow energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.array([0,-der_power_to_heat_ratio_dTs(Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)])

    def heat_power_eq(Eout,m,Ts,Tr,Cp):
        """Heat power equation for the coupling node, based on conservation of energy, and the assumption that there is one heat half link, and one heat dummy link connected to a coupling node

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        fphi : float
            Nodal heat power equation
        """
        return -Eout + Cp*(m*(Ts-Tr))

    def der_heat_power_eq_dEout(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing energy

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dEout : float
            Derivative of nodal heat power equation to outgoing energy
        """
        return -1.

    def der_heat_power_eq_dm(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outgoing flow

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dm : float
            Derivative of nodal heat power equation to water mass flow
        """
        return Cp*(Ts-Tr)

    def der_heat_power_eq_dTs(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the outflow temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTs: float
            Derivative of nodal heat power equation to supply temperature
        """
        return Cp*m

    def der_heat_power_eq_dTr(Eout,m,Ts,Tr,Cp):
        """Derivative of heat power equation to the return temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTr: float
            Derivative of nodal heat power equation to return temperature
        """
        return -Cp*m

    def w(Eout,scale_var=None,scale_var_params=None):
        """Determines which part load effect is used, based on the lower and upper limits provided

        Parameters
        ----------
        Eout : np array
            Outgoing energies, i.e. active power and heat

        Returns
        -------
        w : float
            Modification made to the active power
        """
        phi = Eout[1]
        eta,a,b,d,L1,L2,r1,r2,phimin,phimax = parameters(scale_var=scale_var,scale_var_params=scale_var_params)
        if (phimin <= phi) and (phi <= L2*phimax):
            w = (L1*phimax - phi)*r1 + (L2*phimax - phi)*r2
        elif (L2*phimax <= phi) and (phi <= L1*phimax):
            w = (L1*phimax - phi)*r1
        elif (L1*phimax <= phi) and (phi <= phimax):
            w = 0
        else:
            w = 0
            warnings.warn("phi outside of feasible bounds of the CHP with part load effect: phi = {:.4e}, phimin = {:.4e}, phimax = {:.4e}".format(phi,phimin,phimax))
        return w

    def der_w_dEout(Eout,scale_var=None,scale_var_params=None):
        """Derivative of w wrt Eout, i.e. wrt phi

        Parameters
        ----------
        Eout : np array
            Outgoing energies, i.e. active power and heat

        Returns
        -------
        dw_dEout : float
            Derivative of w
        """
        phi = Eout[1]
        eta,a,b,d,L1,L2,r1,r2,phimin,phimax = parameters(scale_var=scale_var,scale_var_params=scale_var_params)
        if (phimin <= phi) and (phi <= L2*phimax):
            dw_dphi = -r1 -r2
        elif (L2*phimax <= phi) and (phi <= L1*phimax):
            dw_dphi = -r1
        elif (L1*phimax <= phi) and (phi <= phimax):
            dw_dphi = 0
        else:
            dw_dphi = 0
            warnings.warn("phi outside of feasible bounds of the CHP with part load effect: phi = {:.4e}, phimin = {:.4e}, phimax = {:.4e}".format(phi,phimin,phimax))
        dw_dEout = np.array([0, dw_dphi]) #[dw_dP, dw_dphi]
        return dw_dEout

    def power_to_heat_ratio(Eout,Ts,scale_var=None,scale_var_params=None):
        """The active power produced as function of the heat power produced

        Parameters
        ----------
        Eout : np array
            Outgoing energies, i.e. active power and heat
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        P : float
            Active power
        """
        eta,a,b,d,L1,L2,r1,r2,phimin,phimax = parameters(scale_var=scale_var,scale_var_params=scale_var_params)
        phi = Eout[1]
        return a*phi + b*Ts + d - w(Eout,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_power_to_heat_ratio_dEout(Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of the active power produced (as function of the heat power produced and outflow temperature) to the outgoing energy

        Parameters
        ----------
        Eout : np array
            Outgoing energies, i.e. active power and heat
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        dP_dphi : float
            Active power
        """
        eta,a,b,d,L1,L2,r1,r2,phimin,phimax = parameters(scale_var=scale_var,scale_var_params=scale_var_params)
        dw_dEout = der_w_dEout(Eout,scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([0, a-dw_dEout[1]]) #[dP_dP, dP_dphi]

    def der_power_to_heat_ratio_dTs(Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of the active power produced (as function of the heat power produced and outflow temperature) to the outgoing energy

        Parameters
        ----------
        Eout : np array
            Outgoing energies, i.e. active power and heat
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        dP_dTs : float
            Active power
        """
        eta,a,b,d,L1,L2,r1,r2,phimin,phimax = parameters(scale_var=scale_var,scale_var_params=scale_var_params)
        return b #[dP_dTs]

    return Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr

# ===========================================================================
def EH(C,Ein_carriers,Eout_carriers):
    """Creates all node equations needed for an energy hub node.

    Parameters
    ----------
    C : np array
        Coupling matrix of the energy hub

    Returns
    -------
    Eout_of_Ein : function
        Outgoing energy as function of incoming energy
    Ein_of_Eout : function
        Incoming energy as function of outgoing energy
    dEout_dEin : function
        Derivative of outgoing energy to incoming energy
    dEin_dEout : function
        Derivative of incoming energy to outgoing energy
    node_law_Eout : function
        Node law, using outgoing energy as function of incoming energy
    node_law_Ein : function
        Node law, using incoming energy as function of outgoing energy
    der_node_law_Eout_dEout : function
        Derivative of node law (using outgoing energy as function of incoming energy) to outgoing energy
    der_node_law_Eout_dEin : function
        Derivative of node law (using outgoing energy as function of incoming energy) to incoming energy
    der_node_law_Ein_dEout : function
        Derivative of node law (using incoming energy as function of outgoing energy) to outgoing energy
    der_node_law_Ein_dEin : function
        Derivative of node law (using incoming energy as function of outgoing energy) to incoming energy
    heat_power_eq : function
        Nodal heat power equation
    der_heat_power_eq_dEout : function
        Derivative of nodal heat power equation to outgoing energy
    der_heat_power_eq_dm : function
        Derivative of nodal heat power equation to outgoing flow
    der_heat_power_eq_dTs : function
        Derivative of nodal heat power equation to supply temperature
    der_heat_power_eq_dTr : function
        Derivative of nodal heat power equation to return temperature
    """
    def coupling_matrix(C=C,Ein_carriers=Ein_carriers,Eout_carriers=Eout_carriers,scale_var=None,scale_var_params=None):
        """Coupling matrix of the CHP.

        Parameters
        ----------
        C : np array
            Coupling matrix. Unscaled.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params: dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        coup_matr : np array
            Coupling matrix, possibly scaled.
        """
        if scale_var == 'per_unit':
            Eout_b = np.zeros(len(Eout_carriers))
            for ind, carrier in enumerate(Eout_carriers):
                if carrier == 'g':
                    Eout_b[ind] = scale_var_params.get('Ebase')
                elif carrier == 'e':
                    Eout_b[ind] = scale_var_params.get('Sbase')
                elif carrier == 'h':
                    Eout_b[ind] = scale_var_params.get('phibase')
            Ein_b = np.zeros(len(Ein_carriers))
            for ind, carrier in enumerate(Ein_carriers):
                if carrier == 'g':
                    Ein_b[ind] = scale_var_params.get('Ebase')
                elif carrier == 'e':
                    Ein_b[ind] = scale_var_params.get('Sbase')
                elif carrier == 'h':
                    Ein_b[ind] = scale_var_params.get('phibase')
            coup_matr = np.diag(1/Eout_b).dot(C.dot(np.diag(Ein_b)))
        else:
            coup_matr = C
        return coup_matr

    def Eout_of_Ein(Ein,C=C,scale_var=None,scale_var_params=None):
        """Energy going out of node as a function of energy coming in.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        Eout : np array
            Outgoing energies
        """
        coup_matr = coupling_matrix(C=C,scale_var=scale_var,scale_var_params=scale_var_params)
        return coup_matr.dot(Ein)

    def Ein_of_Eout(Eout,Ts,C=C,scale_var=None,scale_var_params=None):
        """Energy coming in to the node as a function of energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        Ein : np array
            Incoming energies
        """
        coup_matr = coupling_matrix(C=C,scale_var=scale_var,scale_var_params=scale_var_params)
        return np.linalg.solve(coup_matr,Eout)

    def dEout_dEin(Ein,C=C,scale_var=None,scale_var_params=None):
        """Derivative of energy going out of node (as a function of energy coming in) to energy coming in.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        dEout_dEin : np array
            Derevatives of outgoing energies to incoming energies
        """
        coup_matr = coupling_matrix(C=C,scale_var=scale_var,scale_var_params=scale_var_params)
        return coup_matr

    def dEin_dEout(Eout,Ts,C=C,scale_var=None,scale_var_params=None):
        """Derivative of energy coming in to the node (as a function of energy going out) to energy going out.

        Parameters
        ----------
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        dEin_dEout : np array
            Derivatives of incoming energies to outgoing energies
        """
        coup_matr = coupling_matrix(C=C,scale_var=scale_var,scale_var_params=scale_var_params)
        return np.linalg.inv(coup_matr)

    def node_law_Eout(Ein,Eout,Ts,C=C,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses outgoing energy as a function of incoming energy.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        f : np array
            Node law, Eout - Eout(Ein)
        """
        return Eout - Eout_of_Ein(Ein,C=C,scale_var=scale_var,scale_var_params=scale_var_params)

    def node_law_Ein(Ein,Eout,Ts,C=C,scale_var=None,scale_var_params=None):
        """Node law of the coupling node, based on conservation of energy. Uses incoming energy as a function of outgoing energy.

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        f : np array
            Node law, Ein - Ein(Eout)
        """
        return Ein - Ein_of_Eout(Eout,Ts,C=C,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dEout(Ein,Eout,Ts,C=C,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Eout - Eout(Ein) to Eout
        """
        return np.eye(len(Eout))

    def der_node_law_Eout_dEin(Ein,Eout,Ts,C=C,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (using outgoing energy as a function of incoming energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEin : np array
            derivative of Eout - Eout(Ein) to Ein
        """
        return -dEout_dEin(Ein,C=C,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Eout_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses outgoing energy as a function of incoming energy) to supply temperature

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.zeros(len(Eout))

    def der_node_law_Ein_dEout(Ein,Eout,Ts,C=C,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outgoing energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Eout
        """
        return -dEin_dEout(Eout,Ts,C=C,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_Ein_dEin(Ein,Eout,Ts,C=C,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to incoming energy

        Parameters
        ----------
        Ein : np array
            Incoming energy (dimensions must match the coupling matrix C).
        Eout : np array
            Outgoing energies (dimensions must match the coupling matrix C).
        C : np array
            Coupling matrix of the energy hub

        Returns
        -------
        df_dEout : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.eye(len(Ein))

    def der_node_law_Ein_dTs(Ein,Eout,Ts,scale_var=None,scale_var_params=None):
        """Derivative of node law of the coupling node (uses incoming energy as a function of outgoing energy) to outflow energy

        Parameters
        ----------
        Ein : np array
            Incoming energy.
        Eout : np array
            Outgoing energies.
        Ts : float
            Temperature of the water at supply line side.

        Returns
        -------
        df_dTs : np array
            derivative of Ein - Ein(Eout) to Ein
        """
        return np.zeros(len(Ein))

    def heat_power_eq(Eout,m,Ts,Tr,Cp,C=C):
        """Heat power equation for the coupling node, based on conservation of energy, and the assumption that there is one heat half link, and one heat dummy link connected to a coupling node

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        fphi : float
            Nodal heat power equation
        """
        return -Eout + Cp*(m*(Ts-Tr))

    def der_heat_power_eq_dEout(Eout,m,Ts,Tr,Cp,C=C):
        """Derivative of heat power equation to the outgoing energy

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dEout : float
            Derivative of nodal heat power equation to outgoing energy
        """
        return -1.

    def der_heat_power_eq_dm(Eout,m,Ts,Tr,Cp,C=C):
        """Derivative of heat power equation to the outgoing flow

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dm : float
            Derivative of nodal heat power equation to water mass flow
        """
        return Cp*(Ts-Tr)

    def der_heat_power_eq_dTs(Eout,m,Ts,Tr,Cp,C=C):
        """Derivative of heat power equation to the outflow temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTs: float
            Derivative of nodal heat power equation to supply temperature
        """
        return Cp*m

    def der_heat_power_eq_dTr(Eout,m,Ts,Tr,Cp,C=C):
        """Derivative of heat power equation to the return temperature

        Parameters
        ----------
        Eout : float
            Outgoing energy, i.e. heat
        m : float
            Outgoing flow, i.e. water
        Ts : float
            Temperature of the water at supply line side.
        Tr : float
            Temperature of the water at return line side.
        Cp : float
            Specific heat constant (possibly scaled)

        Returns
        -------
        dfphi_dTr: float
            Derivative of nodal heat power equation to return temperature
        """
        return -Cp*m

    return Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr
