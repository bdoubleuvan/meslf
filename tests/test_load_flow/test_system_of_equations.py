"""Test the creation of the system of equations and Jacobian matrix
"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Gas, Water
from meslf.load_flow.system_of_equations import NonLinearSystem, NonLinearSystemGas, NonLinearSystemElectrical, NonLinearSystemHeat, NonLinearSystemHeterogeneous
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.utils.constants import mbar, bar, atm, hour, cm, km, MW
import numpy as np
import scipy.sparse as sps
import pytest

# =========================================================
# Gas network
def create_test_gas_network():
    gas_net = GasNetwork('test gas network')
    g0 = GasNode('g0',node_type=0,p=30.) # reference node
    g1 = GasNode('g1',node_type=3,p=10.,q=250.) # reference load node
    g2 = GasNode('g2',node_type=2) # slack node
    g3 = GasNode('g3',node_type=1,q=180.)

    l0 = GasLink('l0',g0,g1,link_type = 'pipe_linear',link_params = {'alpha':10})
    l1 = GasLink('l1',g0,g2,link_type = 'pipe_linear',link_params = {'alpha':10})
    l2 = GasLink('l2',g0,g3,link_type = 'pipe_linear',link_params = {'alpha':10})
    l3 = GasLink('l3',g2,g1,link_type = 'pipe_linear',link_params = {'alpha':10})
    l4 = GasLink('l4',g2,g3,link_type = 'pipe_linear',link_params = {'alpha':10})
    gas_net.add_link(l0)
    gas_net.add_link(l1)
    gas_net.add_link(l2)
    gas_net.add_link(l3)
    gas_net.add_link(l4)
    return gas_net

def test_F_gas_form_nodal():
    """Test the creation of the system of equations for a gas network using the nodal formulation
    """
    # Given
    form = 'nodal'
    gas_net = create_test_gas_network()
    x_init = np.array([24.,28.])
    nlsys = NonLinearSystemGas(gas_net,formulation=form)

    # When
    F = nlsys.F(x_init)
    F_expected = np.array([90, -200.0])

    # Then
    assert np.all(F == F_expected)

def test_F_gas_form_full():
    """Test the creation of the system of equations for a gas network using the full formulation
    """
    # Given
    form = 'full'
    gas_net = create_test_gas_network()
    p = np.array([24.,28.])
    q = np.array([200.,150.,160.,50.,15.])
    x_init = np.concatenate((q,p))
    nlsys = NonLinearSystemGas(gas_net,formulation=form)

    # When
    F = nlsys.F(x_init)
    node_law_expected = np.array([0., -5.0])
    link_equations_expected = np.array([0.,90.,140.,-90.,55.])
    F_expected = np.concatenate((node_law_expected,link_equations_expected))

    # Then
    assert np.all(F == F_expected)

def test_J_gas_form_nodal():
    """Test the creation of the Jacobian matrix for a gas network using the nodal formulation
    """
    # Given
    form = 'nodal'
    gas_net = create_test_gas_network()
    x_init = np.array([24.,28.])
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation=form)

    # When
    J = nlsys.J_dense(x_init)
    J_expected = np.array([[ 10., 0.],[ 10., -20.]])

    # Then
    assert np.all(J == J_expected)

def test_J_gas_form_full():
    """Test the creation of the Jacobian matrix for a gas network using the full formulation
    """
    # Given
    form = 'full'
    gas_net = create_test_gas_network()
    p = np.array([24.,28.])
    q = np.array([200.,150.,160.,50.,15.])
    x_init = np.concatenate((q,p))
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation=form)

    # When
    J = nlsys.J_dense(x_init)
    J_expected = np.array([[ 1.,  0.,  0.,  1.,  0.,  0.,  0.],
                           [ 0.,  0.,  1.,  0.,  1.,  0.,  0.],
                           [ 1.,  0.,  0.,  0.,  0.,  0.,  0.],
                           [ 0.,  1.,  0.,  0.,  0.,  10.,  0.],
                           [ 0.,  0.,  1.,  0.,  0.,  0., 10.],
                           [ 0.,  0.,  0.,  1.,  0.,  -10., 0.],
                           [ 0.,  0.,  0.,  0.,  1.,  -10., 10.]])

    # Then
    assert np.all(J == J_expected)

def create_test_gas_network_one_link():
    """Create a gas network consisting of one single link (a low pressure pipe)

    Returns
    -------
    gas_net : GasNetwork
        The test network
    carrier : Carrier
        The carrier in the network
    """
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    gas_net = GasNetwork('single pipe')
    gn0 = GasNode('hn0',node_type=0,p=30*mbar) # ref node
    gn1 = GasNode('hn2',node_type=1,q=218.45689801329863*carrier.rhon/hour) # load node

    # pipe parameters
    L = 680. #m
    D = .150 #[m]
    link_type = 'pipe_low_pres_pole'
    link_params = {'L':L,'D':D,'carrier':carrier}
    gl0 = GasLink('gl0',gn0,gn1,link_type=link_type,link_params=link_params)

    gas_net.add_link(gl0)
    return gas_net, carrier

def test_F_gas_one_link_full():
    """Test the system of equations for a gas network with only one low pressure pipe, using the full formulation
    """
    # Given
    gas_net, gas = create_test_gas_network_one_link()
    q = 300.*gas.rhon/hour
    q_init = np.array([q])
    p_init = np.array([20.*mbar])
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation='full')

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([0.01614,-0.002065])
    rel_tol = 1e-4
    assert np.allclose(F,F_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_J_gas_one_link_full():
    """Test the creation of the Jacobian matrix for a gas network with only one low pressure pipe
    """
    # Given
    gas_net, gas = create_test_gas_network_one_link()
    q = 300.*gas.rhon/hour
    q_init = np.array([q])
    p_init = np.array([20.*mbar])
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation='full')

    # When
    J = nlsys.J_dense(x_init)

    # Then
    J_expected = np.array([[1, 0],
                          [1, 3.0725e-5]])
    rel_tol = 1e-4
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def create_test_gas_network_one_link_fb():
    """Create a gas network consisting of one single link (a low pressure pipe), with the alternative formulation of the link equation

    Returns
    -------
    gas_net : GasNetwork
        The test network
    carrier : Carrier
        The carrier in the network
    """
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    Z = 1 #low pressure
    T = Tn #low pressure
    carrier = Gas('low pres gas',S,R_air,Z,pn,Tn,T)
    gas_net = GasNetwork('single pipe')
    gn0 = GasNode('hn0',node_type=0,p=30*mbar) # ref node
    gn1 = GasNode('hn2',node_type=1,q=0.043) # load node

    # pipe parameters
    L = 680. #[m]
    D = .150 #[m]
    link_type = 'pipe_low_pres_pole'
    link_params = {'L':L,'D':D,'carrier':carrier}
    gl0 = GasLink('gl0',gn0,gn1,link_type=link_type,link_params=link_params,link_eq_form='dp_of_q')

    gas_net.add_link(gl0)
    return gas_net, carrier

def test_F_gas_one_link_full_fb():
    """Test the system of equations for a gas network with only one low pressure pipe, using the full formulation
    """
    # Given
    gas_net, gas = create_test_gas_network_one_link_fb()
    q_init = np.array([0.1]) #[kg/s]
    p_init = np.array([25.*mbar]) #[Pa]
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation='full')

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([0.057,-2148.225])
    rel_tol = 1e-6
    assert np.allclose(F,F_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_J_gas_one_link_fb():
    """Test the creation of the Jacobian matrix for a gas network with only one low pressure pipe
    """
    # Given
    gas_net, gas = create_test_gas_network_one_link_fb()
    q = 300.*gas.rhon/hour
    q_init = np.array([0.1]) #[kg/s]
    p_init = np.array([25.*mbar]) #[Pa]
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation='full')

    # When
    J = nlsys.J_dense(x_init)

    # Then
    J_expected = np.array([[1, 0],
                          [-52964.505, -1]])
    rel_tol = 1e-6
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def create_test_gas_network_one_link_hpw():
    """Create a gas network consisting of one single link (a high pressure pipe with Weymouth friction factor)

    Returns
    -------
    gas_net : GasNetwork
        The test network
    carrier : Carrier
        The carrier in the network
    """
    # carrier
    Z = 0.96
    T = 300 #[K]
    S = 0.7
    Tn = 288 #[K]
    pn = 1.013*bar #[Pa]
    R = 8.314 #[J/molK]
    M = 29e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    gas_net = GasNetwork('single pipe')
    gn0 = GasNode('hn0',node_type=0,p=32*bar) # ref node
    gn1 = GasNode('hn2',node_type=1,q=1.066e6*carrier.rhon/(24*hour)) # load node

    # pipe parameters
    L = 10.*km #[m]
    D = 0.3 #[m]
    E = 1
    link_type = 'pipe_high_pres_weymouth'
    link_params = {'carrier':carrier, 'D':D, 'L':L, 'E':E}
    gl0 = GasLink('gl0',gn0,gn1,link_type=link_type,link_params=link_params)

    gas_net.add_link(gl0)
    return gas_net, carrier

def test_F_gas_one_link_full_hpw():
    """Test the system of equations for a gas network with only one low pressure pipe, using the full formulation
    """
    # Given
    gas_net, gas = create_test_gas_network_one_link_hpw()
    q_init = np.array([5])
    p_init = np.array([25.6*bar])
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation='full')

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([-5.596109,-13.275709729681854])
    rel_tol = 1e-6
    assert np.allclose(F,F_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_J_gas_one_link_hpw():
    """Test the creation of the Jacobian matrix for a gas network with only one low pressure pipe
    """
    # Given
    gas_net, gas = create_test_gas_network_one_link_hpw()
    q_init = np.array([5])
    p_init = np.array([25.6*bar])
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation='full')

    # When
    J = nlsys.J_dense(x_init)

    # Then
    J_expected = np.array([[1, 0],
                          [1, 1.2691465e-5]])
    rel_tol = 1e-6
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def create_test_gas_network_one_link_hpw_fb():
    """Create a gas network consisting of one single link (a high pressure pipe with Weymouth friction factor)

    Returns
    -------
    gas_net : GasNetwork
        The test network
    carrier : Carrier
        The carrier in the network
    """
    # carrier
    Z = 0.96
    T = 300 #[K]
    S = 0.7
    Tn = 288 #[K]
    pn = 1.013*bar #[Pa]
    R = 8.314 #[J/molK]
    M = 29e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    gas_net = GasNetwork('single pipe')
    gn0 = GasNode('hn0',node_type=0,p=32*bar) # ref node
    gn1 = GasNode('hn2',node_type=1,q=1.066e6*carrier.rhon/(24*hour)) # load node

    # pipe parameters
    L = 10.*km #[m]
    D = 0.3 #[m]
    E = 1
    link_type = 'pipe_high_pres_weymouth'
    link_params = {'carrier':carrier, 'D':D, 'L':L, 'E':E}
    gl0 = GasLink('gl0',gn0,gn1,link_type=link_type,link_params=link_params,link_eq_form='dp_of_q')

    gas_net.add_link(gl0)
    return gas_net, carrier

def test_F_gas_one_link_full_hpw_fb():
    """Test the system of equations for a gas network with only one low pressure pipe, using the full formulation
    """
    # Given
    gas_net, gas = create_test_gas_network_one_link_hpw_fb()
    q_init = np.array([5])
    p_init = np.array([25.6*bar])
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation='full')

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([-5.596109,3.41047e12])
    rel_tol = 1e-6
    assert np.allclose(F,F_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_J_gas_one_link_hpw_fb():
    """Test the creation of the Jacobian matrix for a gas network with only one low pressure pipe
    """
    # Given
    gas_net, gas = create_test_gas_network_one_link_hpw_fb()
    q_init = np.array([5])
    p_init = np.array([25.6*bar])
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation='full')

    # When
    J = nlsys.J_dense(x_init)

    # Then
    J_expected = np.array([[1, 0],
                          [-1.103707e11, -51.2*bar]])
    rel_tol = 1e-6
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def create_test_gas_network_one_link_hpp():
    """Create a gas network consisting of one single link (a high pressure pipe with Pole friction factor)

    Returns
    -------
    gas_net : GasNetwork
        The test network
    carrier : Carrier
        The carrier in the network
    """
    # carrier
    Z = 1
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    T = Tn
    R = 8.314 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    gas_net = GasNetwork('single pipe')
    p_ref = 0.1*mbar+atm #[Pa]
    q_out = 7e-5 #[kg/s]
    gn0 = GasNode('gn0',node_type=0,p=p_ref) # ref node
    gn1 = GasNode('gn1',node_type=1,q=q_out) # load node

    # pipe parameters
    L = 100 #[m]
    D = 5*cm #[m]
    link_type = 'pipe_high_pres_pole'
    link_params = {'carrier':carrier, 'D':D, 'L':L}
    gl0 = GasLink('gl0',gn0,gn1,link_type=link_type,link_params=link_params)

    gas_net.add_link(gl0)
    return gas_net, carrier

def test_F_gas_one_link_full_hpp():
    """Test the system of equations for a gas network with only pipe, using a high pressure pipe model with Pole friction factor, and using the full formulation
    """
    # Given
    gas_net, gas = create_test_gas_network_one_link_hpp()
    p_ref = gas_net.nodes[0].p
    q_out = gas_net.nodes[1].half_links[0].q
    q_init = np.array([0.5*q_out])
    p_init = np.array([0.999999*p_ref])
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation='full')

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([q_init[0] - q_out,-6.88336497e-5])
    rel_tol = 1e-6
    assert np.allclose(F,F_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_J_gas_one_link_hpp():
    """Test the creation of the Jacobian matrix for a gas network with only one low pressure pipe
    """
    # Given
    gas_net, gas = create_test_gas_network_one_link_hpp()
    p_ref = gas_net.nodes[0].p
    q_out = gas_net.nodes[1].half_links[0].q
    q_init = np.array([0.5*q_out])
    p_init = np.array([0.999999*p_ref])
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation='full')

    # When
    J = nlsys.J_dense(x_init)

    # Then
    J_expected = np.array([[1, 0],
                          [1, 5.13976823e-4]])
    rel_tol = 1e-6
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

# =========================================================
# Electrical network
def create_test_elec_network():
    """Creates and returns an electrial test network with only short lines
    """
    elec_net = ElectricalNetwork('test electrical network')
    en0 = ElectricalNode('en0',node_type=0,V=1.,delta=0.) # reference node
    en1 = ElectricalNode('en1',node_type=2,P=0.9931,Q=0.0492) # load node
    en2 = ElectricalNode('en2',node_type=1,P=0.9931,V=0.98) # generator node

    b = -10.
    g = 1.
    el0 = ElectricalLink('el0',en0,en1,link_type = 'short_line',link_params = {'b':b,'g':g})
    el1 = ElectricalLink('el1',en0,en2,link_type = 'short_line',link_params = {'b':b,'g':g})
    el2 = ElectricalLink('el2',en1,en2,link_type = 'short_line',link_params = {'b':b,'g':g})

    elec_net.add_link(el0)
    elec_net.add_link(el1)
    elec_net.add_link(el2)
    return elec_net

def test_F_electrical():
    """Test the system of equations for an electrical network with only short lines
    """
    # Given
    elec_net = create_test_elec_network()
    delta_init = np.array([0.,0.])
    V_init = np.array([1.])
    x_init = np.concatenate((delta_init,V_init))
    elec_net.initialize()
    nlsys = NonLinearSystemElectrical(elec_net)

    # When
    F = nlsys.F(x_init)

    # Then
    nodal_P_expected = np.array([1.0131000000000001, 0.9538999999999999])
    nodal_Q_expected = np.array([0.24919999999999928])
    F_expected = np.concatenate((nodal_P_expected,nodal_Q_expected))
    assert np.allclose(F,F_expected)

def test_J_electrical():
    """Test the creation of the Jacobian matrix for an electrical network with only short lines
    """
    # Given
    elec_net = create_test_elec_network()
    delta_init = np.array([0.,0.])
    V_init = np.array([1.])
    x_init = np.concatenate((delta_init,V_init))
    elec_net.initialize()
    nlsys = NonLinearSystemElectrical(elec_net)

    # When
    J = nlsys.J_dense(x_init)
    J_expected = np.array([[ 19.8 ,  -9.8 ,   2.02],
        [ -9.8 ,  19.6 ,  -0.98],
        [ -1.98,   0.98,  20.2 ]])

    # Then
    assert np.all(J == J_expected)

# =========================================================
# Heat network
def create_test_heat_network_one_link():
    """Create a heat network consisting of one single link

    Returns
    -------
    heat_net : HeatNetwork
        The test network
    water : Carrier
        The carrier in the network
    """
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1e3 #[kg/m^3]
    water = Water('water',Cp,rho=rho)
    Ta = 20. #[C]
    heat_net = HeatNetwork('test heat network',Ta=Ta)
    hn0 = HeatNode('hn0',node_type=0,Ts=70.,p=5*bar) # source slack node
    hn0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn1 = HeatNode('hn2',node_type=1,Tr_hl=50.,dphi=2.3*MW) # load node (sink)
    hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})

    L = 1*km #[m]
    D = 0.2 #[m]
    U = 10 #[W/(m^2 K)]
    link_type = 'standard_pipe_low_pres_pole'
    link_params = {'L':L,'D':D,'U':U,'carrier':water}
    hl0 = HeatLink('hl0',hn0,hn1,link_type=link_type,link_params=link_params)

    heat_net.add_link(hl0)
    return heat_net, water

def test_F_heat_one_link():
    """Test the system of equations for a heat network with only standard pipes and a given pipe constant
    """
    # Given
    heat_net, water = create_test_heat_network_one_link()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])#-Ta
    Tr_init = np.array([44.,45.])#-Ta
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net)

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([-5.266689,1.35*bar,258.643183,5.2780857,-420.1408738])#np.array([-5.266689,1.35*bar,153.30940,5.2780857,-315.0803055])

    rel_tol = 1e-3
    assert np.allclose(F,F_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_J_heat_one_link():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant
    """
    # Given
    heat_net, water = create_test_heat_network_one_link()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])
    Tr_init = np.array([44.,45.])
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net)

    # When
    J = nlsys.J_dense(x_init)

    # Then
    J_expected = np.array([[1, 0, 2.4455077, 0, 0],
                          [-4138.02852, -1, 0, 0, 0],
                          [-49.944560-Ta, 0, -122.2753854, 0, 0],
                          [-0.97228, 0, 0, m, -29.9482],
                          [25+Ta, 0, 122.2753854, 0, m]])
    rel_tol = 1e-5
    print('J_expected = \n{}'.format(J_expected))
    print('J = \n{}'.format(J))
    print('J_expected - J = \n{}'.format(J_expected - J))
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_F_heat_one_link_scaled():
    """Test the system of equations for a heat network with only standard pipes and a given pipe constant
    """
    # Given
    heat_net, water = create_test_heat_network_one_link()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])
    Tr_init = np.array([44.,45.])
    mbase = 10
    pbase = bar
    Tbase = 70
    phibase = MW
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'Tbase':Tbase,'phibase':phibase}
    x_init = np.concatenate((m_init/mbase,p_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([-0.5266689094086106,1.35,0.36949026119923722,0.0075401223796875971,-0.6005915534054211])

    rel_tol = 1e-6
    assert np.allclose(F,F_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_J_heat_one_link_scaled():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant
    """
    # Given
    heat_net, water = create_test_heat_network_one_link()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])
    Tr_init = np.array([44.,45.])
    mbase = 10
    pbase = bar
    Tbase = 70
    phibase = MW
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'Tbase':Tbase,'phibase':phibase}
    x_init = np.concatenate((m_init/mbase,p_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    J = nlsys.J_dense(x_init)

    # Then
    J_expected = np.array([[1, 0, 17.118553960659217, 0, 0],
                          [-0.41380285203892775, -1, 0, 0, 0],
                          [-0.99920800219259109, 0, -12.227538543328011, 0, 0],
                          [-0.013889715382009893, 0, 0, m/mbase, -2.9948166047830762],
                          [0.6428571428571429, 0, 12.227538543328013, 0, m/mbase]])
    rel_tol = 1e-6
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_F_heat_one_link_unknown_half_link():
    """Test the system of equations for a heat network with only standard pipes and a given pipe constant, using the 'half_link_flow' formulation.
    """
    # Given
    heat_net, water = create_test_heat_network_one_link()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    m_hl_init = np.array([40.])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])
    Tr_init = np.array([44.,45.])
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([-8.584073464102065,1.35*bar,474.27316689050303,5.2780857,-586.2833058845929,0.208*MW])

    rel_tol = 1e-6
    assert np.allclose(F,F_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_J_heat_one_link_unknown_half_link():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant, using the 'half_link_flow' formulation.
    """
    # Given
    heat_net, water = create_test_heat_network_one_link()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    m_hl_init = np.array([40.])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])
    Tr_init = np.array([44.,45.])
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    J = nlsys.J_dense(x_init)

    # Then
    J_expected = np.array([[1, -1, 0, 0, 0, 0],
                          [-4138.02852, 0,  -1, 0, 0, 0],
                          [-49.944560-Ta, 65., 0, 40., 0, 0],
                          [-0.97228, 0, 0, 0, m, -29.9482],
                          [25+Ta, -50., 0, 0, 0, m],
                          [0, 62700, 0, 167200, 0, 0]])
    rel_tol = 1e-5
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_F_heat_one_link_unknown_half_link_scaled():
    """Test the system of equations for a heat network with only standard pipes and a given pipe constant, using the 'half_link_flow' formulation, and using per unit scaling.
    """
    # Given
    heat_net, water = create_test_heat_network_one_link()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    m_hl_init = np.array([40.])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])
    Tr_init = np.array([44.,45.])
    mbase = 10
    pbase = bar
    Tbase = 70
    phibase = MW
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'Tbase':Tbase,'phibase':phibase}
    x_init = np.concatenate((m_init/mbase,m_hl_init/mbase,p_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow',scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([-8.584073464102065/mbase,1.35*bar/pbase,474.27316689050303/(mbase*Tbase),5.2780857/(mbase*Tbase),-586.2833058845929/(mbase*Tbase),0.208*MW/phibase])

    rel_tol = 1e-6
    assert np.allclose(F,F_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_J_heat_one_link_unknown_half_link_scaled():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant, using the 'half_link_flow' formulation, and using per unit scaling.
    """
    # Given
    heat_net, water = create_test_heat_network_one_link()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    m_hl_init = np.array([40.])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])
    Tr_init = np.array([44.,45.])
    mbase = 10
    pbase = bar
    Tbase = 70
    phibase = MW
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'Tbase':Tbase,'phibase':phibase}
    x_init = np.concatenate((m_init/mbase,m_hl_init/mbase,p_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow',scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    J = nlsys.J_dense(x_init)

    # Then
    D_F = np.diag([1/mbase,1/pbase,1/(mbase*Tbase),1/(mbase*Tbase),1/(mbase*Tbase),1/phibase])
    D_x_inv = np.diag([mbase,mbase,pbase,Tbase,Tbase,Tbase])
    J_expected = D_F.dot(np.array([[1, -1, 0, 0, 0, 0],
                          [-4138.02852, 0,  -1, 0, 0, 0],
                          [-49.944560-Ta, 65., 0, 40., 0, 0],
                          [-0.97228, 0, 0, 0, m, -29.9482],
                          [25+Ta, -50., 0, 0, 0, m],
                          [0, 62700, 0, 167200, 0, 0]])).dot(D_x_inv)
    rel_tol = 1e-5
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_F_heat_one_link_unknown_half_link_dT():
    """Test the system of equations for a heat network with only standard pipes and a given pipe constant, using the 'half_link_flow' formulation, and assuming dT known for the sink.
    """
    # Given
    heat_net, water = create_test_heat_network_one_link()
    heat_net.nodes[1].node_type = 12
    dT1 = 20
    heat_net.nodes[1].half_links[0].dT = dT1
    heat_net.nodes[1].half_links[0].bc_type = 5
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    m_hl_init = np.array([40.])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])
    Tr_init = np.array([44.,45.])
    Trhl_init = np.array([46.])
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init,Trhl_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    F = nlsys.F(x_init)

    # Then
    Cp = water.Cp
    U = heat_net.links[0].link_params['U']
    L = heat_net.links[0].link_params['L']
    psi = np.exp(-np.pi*D*U*L/(Cp*np.abs(m)))
    mhl1 = m_hl_init[0]
    mhl0 = -m_init[0]
    Ts0 = heat_net.nodes[0].Ts
    Tr0 = Tr_init[0]
    Ts1 = Ts_init[0]
    Tr1 = Tr_init[1]
    Trhl1 = Trhl_init[0]
    dphi1 = heat_net.nodes[1].half_links[0].dphi
    F_expected = np.array([-8.584073464102065,1.35*bar,
                           mhl1*Ts1 -m*(psi*(Ts0-Ta)+Ta),
                           -mhl0*Tr0 - m*(psi*(Tr1-Ta)+Ta),
                           -mhl1*Trhl1 + m*Tr1,
                           -dphi1 + Cp*mhl1*(Ts1 - Trhl1),
                           Ts1 - Trhl1 - dT1])

    rel_tol = 1e-6
    print('F = \n{}'.format(F))
    print('F expected = \n{}'.format(F_expected))
    assert np.allclose(F,F_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_J_heat_one_link_unknown_half_link_dT():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant, using the 'half_link_flow' formulation, and assuming dT known for the sink.
    """
    # Given
    heat_net, water = create_test_heat_network_one_link()
    heat_net.nodes[1].node_type = 12
    dT1 = 20
    heat_net.nodes[1].half_links[0].dT = dT1
    heat_net.nodes[1].half_links[0].bc_type = 5
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    m_hl_init = np.array([40.])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])
    Tr_init = np.array([44.,45.])
    Trhl_init = np.array([46.])
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init,Trhl_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    J = nlsys.J_dense(x_init)

    # Then
    Cp = water.Cp
    U = heat_net.links[0].link_params['U']
    L = heat_net.links[0].link_params['L']
    psi = np.exp(-np.pi*D*U*L/(Cp*np.abs(m)))
    dpsi_dm = np.sign(m)*psi*np.pi*D*U*L/(Cp)*1./(np.abs(m)**2)
    mhl1 = m_hl_init[0]
    mhl0 = -m_init[0]
    Ts0 = heat_net.nodes[0].Ts
    Tr0 = Tr_init[0]
    Ts1 = Ts_init[0]
    Tr1 = Tr_init[1]
    Trhl1 = Trhl_init[0]
    J_expected = np.array([[1, -1, 0, 0, 0, 0, 0],
                          [-4138.02852, 0,  -1, 0, 0, 0, 0],
                          [-(psi*(Ts0-Ta)+Ta) - m*dpsi_dm*(Ts0-Ta), Ts1, 0, mhl1, 0, 0, 0],
                          [Tr0-(psi*(Tr1-Ta)+Ta) - m*dpsi_dm*(Tr1-Ta), 0, 0, 0, -mhl0, -m*psi, 0],
                          [Tr1, -Trhl1, 0, 0, 0, m,-mhl1],
                          [0, Cp*(Ts1-Trhl1), 0, Cp*mhl1, 0, 0, -Cp*mhl1],
                          [0, 0, 0, 1, 0, 0, -1]])
    rel_tol = 1e-5
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(sps.csr_matrix(J_expected)))
    print('J - J expected = \n{}'.format(J-sps.csr_matrix(J_expected)))
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def create_test_heat_network_one_link_fa():
    """Create a heat network consisting of one single link

    Returns
    -------
    heat_net : HeatNetwork
        The test network
    water : Carrier
        The carrier in the network
    """
    # carrier
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1e3 #[kg/m^3]
    water = Water('water',Cp,rho=rho)
    Ta = 20. #[C]
    heat_net = HeatNetwork('test heat network',Ta=Ta)
    hn0 = HeatNode('hn0',node_type=0,Ts=70.,p=5*bar) # source slack node
    hn0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn1 = HeatNode('hn2',node_type=1,Tr_hl=50.,dphi=2.3*MW) # load node (sink)
    hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})

    # pipe parameters
    L = 1*km #[m]
    D = 0.2 #[m]
    U = 10 #[W/(m^2 K)]
    link_type = 'standard_pipe_low_pres_pole'
    link_params = {'L':L,'D':D,'U':U,'carrier':water}
    hl0 = HeatLink('hl0',hn0,hn1,link_type=link_type,link_params=link_params,hydr_eq_form='q_of_dp')

    heat_net.add_link(hl0)
    return heat_net, water

def test_F_heat_one_link_fa():
    """Test the system of equations for a heat network with only standard pipes and a given pipe constant
    """
    # Given
    heat_net, water = create_test_heat_network_one_link_fa()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])#-Ta
    Tr_init = np.array([44.,45.])#-Ta
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net)

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([-5.266689,-23.691254,258.64318284,5.2780857,-420.1408738])#np.array([-5.266689,-23.691254,153.30940,5.2780857,-315.0803055])
    rel_tol = 1e-3
    assert np.allclose(F,F_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_J_heat_one_link_fa():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant
    """
    # Given
    heat_net, water = create_test_heat_network_one_link_fa()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    D = heat_net.links[0].link_params['D']
    v = 1. #[m/s]
    m = np.pi*D**2*rho*v/4
    m_init = np.array([m])
    p_init = np.array([3.*bar])
    Ts_init = np.array([65.])#-Ta
    Tr_init = np.array([44.,45.])#-Ta
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net)

    # When
    J = nlsys.J_dense(x_init)

    # Then
    J_expected = np.array([[1, 0, 2.4455077, 0, 0],
                          [1, 1.3776795e-4, 0, 0, 0],
                          [-49.944560-Ta, 0, -122.2753854, 0, 0],
                          [-0.97228, 0, 0, m, -29.9482],
                          [25+Ta, 0, 122.2753854, 0, m]])
    rel_tol = 1e-5
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def create_test_heat_network():
    """Create a test heat network

    Returns
    -------
    heat_net : HeatNetwork
        The test network
    water : Carrier
        The carrier in the network
    """
    # carrier
    rho = 960. #[kg/m^3]
    Cp = 4.182e3 #[J/(kg K)]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp,rho=rho)
    # network
    Ta = 10. #[C]
    heat_net = HeatNetwork('test heat network',Ta=Ta)
    hn0 = HeatNode('hn0',node_type=0,Ts=100.,p=11.*rho*g) # source slack node
    hn0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn1 = HeatNode('hn1',node_type=1,Ts_hl=100.,dphi=-0.5*MW) # source node
    hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn2 = HeatNode('hn2',node_type=1,Tr_hl=50.,dphi=1.3*MW) # sink node
    hn2.half_links[0].set_type('heat_exchanger',{'carrier':water})

    C = np.sqrt(1/(rho*g))
    L = 400. #[m]
    D = 0.01 #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D) #[W/(m^2 K)]
    link_type = 'standard_resistor'
    link_params = {'L':L,'D':D,'U':U,'C':C,'carrier':water}
    hl0 = HeatLink('hl0',hn0,hn1,link_type=link_type,link_params=link_params)
    hl1 = HeatLink('hl1',hn0,hn2,link_type=link_type,link_params=link_params.copy())
    hl2 = HeatLink('hl2',hn1,hn2,link_type=link_type,link_params=link_params.copy())

    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    return heat_net, water

def test_F_heat():
    """Test the system of equations for a heat network with only standard pipes and a given pipe constant
    """
    # Given
    heat_net, water = create_test_heat_network()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    m_init = np.array([1.,np.sqrt(10.),3.])
    p_init = np.array([10.*rho*g,2.*rho*g])
    Ts_init = np.array([100.,100.])#-Ta
    Tr_init = np.array([50.,50.,50.])#-Ta
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net)

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([0.39120038259206114,-0.054843334570979785, 5.4569682106375694e-12, -9417.600000000006, -9417.600000000006, -37.41473685630069, 8.9169875737028406, 1.5207860982750958, 20.322768817777046, -2.7421667285489093])#np.array([0.39120038259206114,-0.054843334570979785, 5.4569682106375694e-12, -9417.600000000006, -9417.600000000006, -33.50273303,   8.36855423,  1.5207861, 16.41076499,  -2.19373338])

    assert np.allclose(F,F_expected)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_J_heat():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant
    """
    # Given
    heat_net, water = create_test_heat_network()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    m_init = np.array([1.,np.sqrt(10.),3.])
    p_init = np.array([10.*rho*g,2.*rho*g])
    Ts_init = np.array([100.,100.])
    Tr_init = np.array([50.,50.,50.])
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net)

    # When
    J = nlsys.J_dense(x_init)

    # Then
    J_data = np.array([1.0, -1.0, 0.0478240076518, 1.0, 1.0, 0.124342419895, -18835.2, -1.0, -59562.1321848, -1.0, -56505.6, 1.0, -1.0, -99.9837411336, 100, 3.0, -4.782400765184122, -99.9983598883, -99.998178051, -2.9809312578, -6.217120994739359, 50.0, -49.9991902449, 5.782400765184122, -2.9809312578, 50.0, 50.0, 6.217120994739359, 6.16227766017, 0.00722616283363, 0.000728938517447, -0.981052206634, -3.14320580108, 4.16227766017])#np.array([1.0, -1.0, 0.0478240076518, 1.0, 1.0, 0.124342419895, -18835.2, -1.0, -59562.1321848, -1.0, -56505.6, 1.0, -1.0, -89.9837411336, 90, 3.0, -4.30416068867, -89.9983598883, -89.998178051, -2.9809312578, -4.97369679579, 40.0, -39.9991902449, 5.30416068867, -2.9809312578, 40.0, 40.0, 4.97369679579, 6.16227766017, 0.00722616283363, 0.000728938517447, -0.981052206634, -3.14320580108, 4.16227766017])
    J_indices = np.array([0, 2, 7, 1, 2, 6, 0, 3, 1, 4, 2, 3, 4, 0, 2, 5, 7, 1, 2, 5, 6, 0, 2, 7, 8, 1, 2, 6, 8, 0, 1, 7, 8, 9])
    J_indptr = np.array([0, 3, 6, 8, 10, 13, 17, 21, 25, 29, 34])
    F_ind_v2 = [0,1,2,3,4,5,6,9,7,8] # reorder, because data is taken from older version of code
    x_ind_v2 = [0,1,2,3,4,5,6,9,7,8]
    J_expected = sps.csr_matrix((J_data, J_indices, J_indptr))[F_ind_v2,:][:,x_ind_v2].todense()

    print('J_expected = \n{}'.format(J_expected))
    print('J = \n{}'.format(J))
    print('J_expected - J = \n{}'.format(J_expected - J))
    assert np.allclose(J,J_expected)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_J_heat_general_slack_node():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant, and a general slack node instead of a source slack node
    """
    # Given
    heat_net, water = create_test_heat_network()
    heat_net.nodes[0].node_type = 10 #general slack node instead of source slack node. Same variables are assumed known / unknown
    hl_params = heat_net.nodes[0].half_links[0].link_params
    heat_net.nodes[0].half_links[0].set_type('heat_exchanger',hl_params)
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    m_init = np.array([1.,np.sqrt(10.),3.])
    p_init = np.array([10.*rho*g,2.*rho*g])
    Ts_init = np.array([100.,100.])
    Tr_init = np.array([50.,50.,50.])
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net)

    # When
    J = nlsys.J_dense(x_init)

    # Then
    J_data = np.array([1.0, -1.0, 0.0478240076518, 1.0, 1.0, 0.124342419895, -18835.2, -1.0, -59562.1321848, -1.0, -56505.6, 1.0, -1.0, -99.9837411336, 100, 3.0, -4.782400765184122, -99.9983598883, -99.998178051, -2.9809312578, -6.217120994739359, 50.0, -49.9991902449, 5.782400765184122, -2.9809312578, 50.0, 50.0, 6.217120994739359, 6.16227766017, 0.00722616283363, 0.000728938517447, -0.981052206634, -3.14320580108, 4.16227766017])
    J_indices = np.array([0, 2, 7, 1, 2, 6, 0, 3, 1, 4, 2, 3, 4, 0, 2, 5, 7, 1, 2, 5, 6, 0, 2, 7, 8, 1, 2, 6, 8, 0, 1, 7, 8, 9])
    J_indptr = np.array([0, 3, 6, 8, 10, 13, 17, 21, 25, 29, 34])
    F_ind_v2 = [0,1,2,3,4,5,6,9,7,8] # reorder, because data is taken older version of code
    x_ind_v2 = [0,1,2,3,4,5,6,9,7,8]
    J_expected = sps.csr_matrix((J_data, J_indices, J_indptr))[F_ind_v2,:][:,x_ind_v2].todense()

    assert np.allclose(J,J_expected)

def create_test_heat_network_scaled():
    """Create a test heat network, used for the tests with per unit scaling

    Returns
    -------
    heat_net : HeatNetwork
        The test network
    water : Carrier
        The carrier in the network
    """
    # carrier
    rho = 960. #[kg/m^3]
    Cp = 4.182e3 #[J/(kg K)]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp,rho=rho)
    # network
    MW = 1e6 #[W]
    Ta = 10. #[C]
    heat_net = HeatNetwork('test heat network',Ta=Ta)
    hn0 = HeatNode('hn0',node_type=0,Ts=100.,p=(12.+2*np.sqrt(5.))*rho*g) # source slack node
    hn0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn1 = HeatNode('hn1',node_type=2) # junction node
    hn2 = HeatNode('hn2',node_type=1,Tr_hl=50.,dphi=0.86220755547510397*MW) # load node
    hn2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn3 = HeatNode('hn3',node_type=1,Ts_hl=100.,dphi=-0.21069237249666534*MW) # source node
    hn3.half_links[0].set_type('heat_exchanger',{'carrier':water})

    C = np.sqrt(1/(rho*g))
    L = 400. #[m]
    D = 0.01 #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D) #[W/(m^2 K)]
    link_type = 'standard_resistor'
    link_params = {'L':L,'D':D,'U':U,'C':C,'carrier':water}
    hl0 = HeatLink('hl0',hn0,hn1,link_type=link_type,link_params=link_params)
    hl1 = HeatLink('hl1',hn1,hn2,link_type=link_type,link_params=link_params.copy())
    hl2 = HeatLink('hl2',hn1,hn3,link_type=link_type,link_params=link_params.copy())
    hl3 = HeatLink('hl3',hn3,hn2,link_type=link_type,link_params=link_params.copy())

    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    heat_net.add_link(hl3)
    return heat_net, water

def test_F_heat_scaled():
    """Test the system of equations for a heat network with only standard pipes and a given pipe constant
    """
    # Given
    heat_net, water = create_test_heat_network_scaled()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    m_init = np.array([4., 2., -2., 2.])
    p_init = np.array([6.,2.,4.])*rho*g
    Ts_init = np.array([99.,98.,100.])
    Tr_init = np.array([49.,49.,50.,48.])
    scale_var = 'per_unit'
    Tbase = 80. #[C]
    phibase = 1e6 #[W]
    mbase = phibase/(water.Cp*Tbase)
    pbase = mbase**2*rho*g
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}
    x_init = np.concatenate((m_init/mbase,p_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    F = nlsys.F(x_init)

    # Then
    F_expected = np.array([1.3382399999999999,-0.09877259245850656,-1.0140978884666687,-0.61873599832418513, 0.0, 0.67158236159999984, -0.22386078719999999,-1.6668155129949147,0.1101561596068259,1.2676223605833359,0.0031125513337061372,0.81449274499333091,-0.06173287028656671,-0.62726086171817319])

    assert np.allclose(F,F_expected)

def test_J_heat_scaled():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant, using per unit scaling
    """
    # Given
    heat_net, water = create_test_heat_network_scaled()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    m_init = np.array([4., 2., -2., 2.])
    p_init = np.array([6.,2.,4.])*rho*g
    Ts_init = np.array([99.,98.,100.])
    Tr_init = np.array([49.,49.,50.,48.])
    heat_net.initialize()
    scale_var = 'per_unit'
    Tbase = 80. #[C]
    phibase = 1e6 #[W]
    mbase = phibase/(water.Cp*Tbase)
    pbase = mbase**2*rho*g
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}
    x_init = np.concatenate((m_init/mbase,p_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    nlsys = NonLinearSystemHeat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    J = nlsys.J_dense(x_init)

    dm_dm = sps.csr_matrix((np.array([1,-1,-1,1,1,1,-1]),(np.array([0,0,0,1,1,2,2]),np.array([0,1,2,1,3,2,3]))),shape=(3,4)) # data,(row,col)
    dH_dm = sps.diags(np.array([-2.67648, -1.33824,  -1.33824, -1.33824])).tocsr()
    dTs_dm = sps.csr_matrix((np.array([-1.2499871758,1.2375,1.2499488664125828,-1.2374494345635541,-1.2499488664125828,-1.25,1.25]),(np.array([0,0,0,1,1,2,2]),np.array([0,1,2,1,3,2,3]))),shape=(3,4))
    dTr_dm = sps.csr_matrix((np.array([5.5571508744134412e-06,0.6125,-0.62497727396114788,-0.6125,0.625,0.625,0.6124778421121192,-0.62497727396114788]),(np.array([0,1,1,1,2,2,3,3]),np.array([0,0,1,2,1,3,2,3]))),shape=(4,4))
    dH_dh = sps.csr_matrix((np.array([-1,1,-1,1,-1,-1,1]),(np.array([0,1,1,2,2,3,3]),np.array([0,0,1,0,2,1,2]))),shape=(4,3))
    dm_dTs = sps.csr_matrix((np.array([2.39502098743]),(np.array([1]),np.array([1]))),shape=(3,3))
    dTs_dTs = sps.csr_matrix((np.array([0.66912,-0.66275051001,-0.66275051001,-1.496888117,-0.66275051001,1.33824]),(np.array([0,0,1,1,1,2]),np.array([0,2,0,1,2,2]))),shape=(3,3))
    dTr_dTs = sps.csr_matrix((np.array([1.496888117]),(np.array([2]),np.array([1]))),shape=(4,3))
    dm_dTr = sps.csr_matrix((np.array([0.49868017]),(np.array([2]),np.array([3]))),shape=(3,4))
    dTs_dTr = sps.csr_matrix((np.array([-0.623350214]),(np.array([2]),np.array([3]))),shape=(3,4))
    dTr_dTr = sps.csr_matrix((np.array([1.33824,-1.331855279,2.00736,-0.66275051,1.33824,-0.66275051,-0.66275051,0.623350214]),(np.array([0,0,1,1,2,3,3,3]),np.array([0,1,1,2,2,1,2,3]))),shape=(4,4))
    J_expected = sps.bmat([
                [dm_dm,None,dm_dTs,dm_dTr],
                 [dH_dm,dH_dh,None,None],
                 [dTs_dm,None,dTs_dTs,dTs_dTr],
                 [dTr_dm,None,dTr_dTs,dTr_dTr]]).todense()

    rel_tol = 1e-5
    print('J_expected = \n{}'.format(J_expected))
    print('J = \n{}'.format(J))
    print('J_expected - J = \n{}'.format(J_expected - J))
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_F_heat_unknown_half_link_scaled():
    """Test the system of equations for a heat network of 4 nodes with only standard pipes and a given pipe constant, using the 'half_link_flow' formulation, and using per unit scaling.
    """
    # Given
    heat_net, water = create_test_heat_network_scaled()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    m_init = np.array([4., 2., -2., 2.])
    m_hl_init = np.array([3,-2])
    p_init = np.array([6.,2.,4.])*rho*g
    Ts_init = np.array([99.,98.,100.])
    Tr_init = np.array([49.,49.,50.,48.])
    scale_var = 'per_unit'
    Tbase = 80. #[C]
    phibase = 1e6 #[W]
    mbase = phibase/(water.Cp*Tbase)
    pbase = mbase**2*rho*g
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}
    x_init = np.concatenate((m_init/mbase,m_hl_init/mbase,p_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params,formulation='half_link_flow')

    # When
    F = nlsys.F(x_init)

    # Then
    F_m_expected = np.array([4,1,-2])/mbase #[p.u.]
    F_l_expected = np.array([-0.61873599832418513, 0.0, 0.67158236159999984, -0.22386078719999999]) #[p.u.]
    F_Ts_expected = np.array([-398.5689892383823,-100.59212485768654,200])/(mbase*Tbase) #[p.u.]
    F_Tr_expected = np.array([0.74427339399957759,194.76153634465106,50,-100.49596571931417])/(mbase*Tbase) #[p.u.]
    F_phi_expected = np.array([-259999.55547510402,-224235.62750333466])/phibase #[p.u]
    F_expected = np.concatenate((F_m_expected,F_l_expected,F_Ts_expected,F_Tr_expected,F_phi_expected))

    assert np.allclose(F,F_expected)

def test_J_heat_unknown_half_link_scaled():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant, using per unit scaling
    """
    # Given
    heat_net, water = create_test_heat_network_scaled()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    m_init = np.array([4., 2., -2., 2.])
    m_hl_init = np.array([3,-2])
    p_init = np.array([6.,2.,4.])*rho*g
    Ts_init = np.array([99.,98.,100.])
    Tr_init = np.array([49.,49.,50.,48.])
    heat_net.initialize()
    scale_var = 'per_unit'
    Tbase = 80. #[C]
    phibase = 1e6 #[W]
    mbase = phibase/(water.Cp*Tbase)
    pbase = mbase**2*rho*g
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}
    x_init = np.concatenate((m_init/mbase,m_hl_init/mbase,p_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    nlsys = NonLinearSystemHeat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params,formulation='half_link_flow')

    # When
    J = nlsys.J_dense(x_init)

    dm_dm = sps.csr_matrix((np.array([1,-1,-1,1,1,1,-1]),(np.array([0,0,0,1,1,2,2]),np.array([0,1,2,1,3,2,3]))),shape=(3,4)) # data,(row,col)
    dH_dm = sps.diags(np.array([-75340.8, -37670.4,  -37670.4, -37670.4])).tocsr()*mbase/pbase
    dTs_dm = sps.csr_matrix((np.array([-99.998974064453947,99,99.995909313006621,-98.995954765084321,-99.995909313006621,-100,100]),(np.array([0,0,0,1,1,2,2]),np.array([0,1,2,1,3,2,3]))),shape=(3,4))*1/Tbase
    dTr_dm = sps.csr_matrix((np.array([0.00044457206995218712,49,-49.998181916891831,-49,50,50,48.998227368969538,-49.998181916891831]),(np.array([0,1,1,1,2,2,3,3]),np.array([0,0,1,2,1,3,2,3]))),shape=(4,4))*1/Tbase
    dm_dmhl = sps.csr_matrix((np.array([-1,-1]),(np.array([1,2]),np.array([0,1]))),shape=(3,2))
    dTs_dmhl = sps.csr_matrix((np.array([98,100]),(np.array([1,2]),np.array([0,1]))),shape=(3,2))*1/Tbase
    dTr_dmhl = sps.csr_matrix((np.array([-50,-48]),(np.array([2,3]),np.array([0,1]))),shape=(4,2))*1/Tbase
    dphi_dmhl = sps.csr_matrix((np.array([200736.0,217464.0]),(np.array([0,1]),np.array([0,1]))),shape=(2,2))*mbase/phibase
    dH_dh = sps.csr_matrix((np.array([-1,1,-1,1,-1,-1,1]),(np.array([0,1,1,2,2,3,3]),np.array([0,0,1,0,2,1,2]))),shape=(4,3))
    dTs_dTs = sps.csr_matrix((np.array([2,-1.9809615913837237,-1.9809615913837237,3,-1.9809615913837237,4]),(np.array([0,0,1,1,1,2]),np.array([0,2,0,1,2,2]))),shape=(3,3))*1/mbase
    dphi_dTs = sps.csr_matrix((np.array([12546]),(np.array([0]),np.array([1]))),shape=(2,3))*Tbase/phibase
    dTr_dTr = sps.csr_matrix((np.array([4,-3.9809160668205239,6,-1.9809615913837237,4,-1.9809615913837237,-1.9809615913837237,2]),(np.array([0,0,1,1,2,3,3,3]),np.array([0,1,1,2,2,1,2,3]))),shape=(4,4))*1/mbase
    dphi_dTr = sps.csr_matrix((np.array([8364]),(np.array([1]),np.array([3]))),shape=(2,4))*Tbase/phibase
    J_expected = sps.bmat([
                [dm_dm,dm_dmhl,None,None,None],
                 [dH_dm,None,dH_dh,None,None],
                 [dTs_dm,dTs_dmhl,None,dTs_dTs,None],
                 [dTr_dm,dTr_dmhl,None,None,dTr_dTr],
                 [None,dphi_dmhl,None,dphi_dTs,dphi_dTr]]).todense()

    rel_tol = 1e-6
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_J_heat_unknown_half_link_4N():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant, using per unit scaling
    """
    # Given
    heat_net, water = create_test_heat_network_scaled()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    m_init = np.array([4., 2., -2., 2.])
    m_hl_init = np.array([3,-2])
    p_init = np.array([6.,2.,4.])*rho*g
    Ts_init = np.array([99.,98.,100.])
    Tr_init = np.array([49.,49.,50.,48.])
    heat_net.initialize()
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    J = nlsys.J_dense(x_init)

    dm_dm = sps.csr_matrix((np.array([1,-1,-1,1,1,1,-1]),(np.array([0,0,0,1,1,2,2]),np.array([0,1,2,1,3,2,3]))),shape=(3,4)) # data,(row,col)
    dH_dm = sps.diags(np.array([-75340.8, -37670.4,  -37670.4, -37670.4])).tocsr()
    dTs_dm = sps.csr_matrix((np.array([-99.998974064453947,99,99.995909313006621,-98.995954765084321,-99.995909313006621,-100,100]),(np.array([0,0,0,1,1,2,2]),np.array([0,1,2,1,3,2,3]))),shape=(3,4))
    dTr_dm = sps.csr_matrix((np.array([0.00044457206995218712,49,-49.998181916891831,-49,50,50,48.998227368969538,-49.998181916891831]),(np.array([0,1,1,1,2,2,3,3]),np.array([0,0,1,2,1,3,2,3]))),shape=(4,4))
    dm_dmhl = sps.csr_matrix((np.array([-1,-1]),(np.array([1,2]),np.array([0,1]))),shape=(3,2))
    dTs_dmhl = sps.csr_matrix((np.array([98,100]),(np.array([1,2]),np.array([0,1]))),shape=(3,2))
    dTr_dmhl = sps.csr_matrix((np.array([-50,-48]),(np.array([2,3]),np.array([0,1]))),shape=(4,2))
    dphi_dmhl = sps.csr_matrix((np.array([200736.0,217464.0]),(np.array([0,1]),np.array([0,1]))),shape=(2,2))
    dH_dh = sps.csr_matrix((np.array([-1,1,-1,1,-1,-1,1]),(np.array([0,1,1,2,2,3,3]),np.array([0,0,1,0,2,1,2]))),shape=(4,3))
    dTs_dTs = sps.csr_matrix((np.array([2,-1.9809615913837237,-1.9809615913837237,3,-1.9809615913837237,4]),(np.array([0,0,1,1,1,2]),np.array([0,2,0,1,2,2]))),shape=(3,3))
    dphi_dTs = sps.csr_matrix((np.array([12546]),(np.array([0]),np.array([1]))),shape=(2,3))
    dTr_dTr = sps.csr_matrix((np.array([4,-3.9809160668205239,6,-1.9809615913837237,4,-1.9809615913837237,-1.9809615913837237,2]),(np.array([0,0,1,1,2,3,3,3]),np.array([0,1,1,2,2,1,2,3]))),shape=(4,4))
    dphi_dTr = sps.csr_matrix((np.array([8364]),(np.array([1]),np.array([3]))),shape=(2,4))
    J_expected = sps.bmat([
                [dm_dm,dm_dmhl,None,None,None],
                 [dH_dm,None,dH_dh,None,None],
                 [dTs_dm,dTs_dmhl,None,dTs_dTs,None],
                 [dTr_dm,dTr_dmhl,None,None,dTr_dTr],
                 [None,dphi_dmhl,None,dphi_dTs,dphi_dTr]]).todense()

    rel_tol = 1e-6
    assert np.allclose(J,J_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def create_test_heat_network_3N_line(Ts1=93.72276227854158,p1=10*bar,Tr1=-3.2139721171872058,To2=102.12615673584322,To3=50.,dT2=50,dT3=50,phi2=-2*MW,phi3=1*MW,slack_node='source',source_node='outflow',sink_node='outflow'):
    """Create a test heat network consisting of three nodes (1-3) in a line. Node 1 is taken as slack, node 2 is a source, and node 3 is taken as a sink.

    Parameters
    ----------
    slack_node : str, optional
        Determines if the slack node (node 1) should be a source (ref.) slack node, a sink (ref.) slack node, or a general (ref.) slack node. Options are 'source', 'sink', , 'general_Ts', 'general_Tr', or 'general'. Default is 'source'.

    Returns
    -------
    heat_net : HeatNetwork
        The test network
    """
    # water carrier
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    water = Water('water',Cp_w,rho=rho_w)

    # physical parameters of network and pipes
    Ta = 10. #[C]
    L12 = 4*km #[m]
    L23 = 5*km #[m]
    D = 0.15 #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D) #[W/(m^2 K)]

    heat_net = HeatNetwork('3N one line',Ta=Ta)
    if slack_node == 'source':
        hn1 = HeatNode('hn1',node_type=0,x=0,y=0,Ts=Ts1,p=p1) # source slack node
        hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif slack_node == 'general_Ts':
        hn1 = HeatNode('hn1',node_type=10,x=0,y=0,Ts=Ts1,p=p1) # general slack node (with p and Ts known)
        hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif slack_node == 'sink':
        hn1 = HeatNode('hn1',node_type=8,x=0,y=0,Tr=Tr1,p=p1) # sink slack node
        hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif slack_node == 'general_Tr':
        hn1 = HeatNode('hn1',node_type=11,x=0,y=0,Tr=Tr1,p=p1) # general slack node (with p and Tr known)
        hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif slack_node == 'general':
        hn1 = HeatNode('hn1',node_type=9,x=0,y=0,p=p1) # general slack node (with p known)
        hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    else:
        raise ValueError('Enter valid value for slack_node')
    if source_node == 'outflow':
        hn2 = HeatNode('hn2',node_type=1,x=1,y=0,Ts_hl=To2,dphi=phi2) # load node (source)
        hn2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif source_node == 'delta':
        hn2 = HeatNode('hn2',node_type=12,x=1,y=0,dphi=phi2,dT=dT2) # source temp. diff. node
        hn2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    else:
        raise ValueError('Enter valid value for source_node')
    if sink_node == 'outflow':
        hn3 = HeatNode('hn3',node_type=1,x=2,y=0,Tr_hl=To3,dphi=phi3) # load node (sink)
        hn3.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif sink_node == 'delta':
        hn3 = HeatNode('hn3',node_type=12,x=2,y=0,dphi=phi3,dT=dT3) # sink temp. diff. node
        hn3.half_links[0].set_type('heat_exchanger',{'carrier':water})
    else:
        raise ValueError('Enter valid value for sink_node')

    link_type = 'standard_pipe_low_pres_pole'
    link_params12 = {'L':L12,'D':D,'U':U,'carrier':water}
    link_params23 = link_params12.copy()
    link_params23['L'] = L23

    hl0 = HeatLink('hl0',hn1,hn2,link_type=link_type,link_params=link_params12)
    hl1 = HeatLink('hl1',hn2,hn3,link_type=link_type,link_params=link_params23)

    heat_net.add_link(hl0)
    heat_net.add_link(hl1)

    return heat_net

def test_F_heat_3N_line_source_slack_solution():
    """Test the system of equations for a heat network of three nodes (1-3) in a line, using the solution. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink
    """
    # Given
    heat_net = create_test_heat_network_3N_line(slack_node='source')
    m_sol = np.array([-2.,5.])
    p_sol = np.array([10.046254718677874,9.684889729006985])*bar
    Ts_sol = np.array([102.12615673584322,97.82400765184121])#np.array([123.10587724713142,97.82400765184121])
    Tr_sol = np.array([-3.2139721171872058,33.806145804641474,50.])#np.array([77.58524408442334,54.78586631592967,50.])
    x_sol = np.concatenate((m_sol,p_sol,Ts_sol,Tr_sol))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='standard')

    # When
    F = nlsys.F(x_sol)

    # Then
    assert np.allclose(F,np.zeros(len(F)))

def test_F_heat_3N_line_source_slack_delta_loads():
    """Test the system of equations for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink
    """
    # Given
    Ts1 = 101 #[C]
    p1 = 9.8*bar
    dphi2 = -.4*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Ts1=Ts1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='source',source_node='delta',sink_node='delta')

    m_init = np.array([2.1,3.4])
    p_init  = np.array([10.1,9.4])*bar
    Ts_init  = np.array([103,97])
    Tr_init  = np.array([12,38,47])
    x_init  = np.concatenate((m_init,p_init ,Ts_init ,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='standard')

    # When
    F = nlsys.F(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    m1_hl = -m12
    Tr1_init = Tr_init[0]
    Ts2_init = Ts_init[0]
    Tr2_init = Tr_init[1]
    Ts2_hl = dT2+Tr2_init
    m2_hl = dphi2/(Cp*dT2)
    Ts3_init = Ts_init[1]
    Tr3_init = Tr_init[2]
    Tr3_hl = Ts3_init - dT3
    m3_hl = dphi3/(Cp*dT3)

    F_expected = np.array([m12-m23-m2_hl,
                           m23 - m3_hl,
                           link12.link_equation(),
                           link23.link_equation(),
                           m23*Ts2_init - m12*(link12.temp_drop_fac(m12)*(Ts1-Ta)+Ta) + m2_hl*Ts2_hl,
                           -m23*(link23.temp_drop_fac(m23)*(Ts2_init-Ta)+Ta) + m3_hl*Ts3_init,
                           -m12*(link12.temp_drop_fac(m12)*(Tr2_init-Ta)+Ta) - m1_hl*Tr1_init,
                           -m23*(link23.temp_drop_fac(m23)*(Tr3_init-Ta)+Ta) + m12*Tr2_init -m2_hl*Tr2_init,
                           m23*Tr3_init - m3_hl*Tr3_hl])
    assert np.allclose(F,F_expected)

def test_J_heat_3N_line_source_slack_delta_loads():
    """Test the Jacobian for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink
    """
    # Given
    Ts1 = 101 #[C]
    p1 = 9.8*bar
    dphi2 = -.4*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Ts1=Ts1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='source',source_node='delta',sink_node='delta')

    m_init = np.array([2.1,3.4])
    p_init  = np.array([10.1,9.4])*bar
    Ts_init  = np.array([103,97])
    Tr_init  = np.array([12,38,47])
    x_init  = np.concatenate((m_init,p_init ,Ts_init ,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='standard')

    # When
    J = nlsys.J(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    df12_dp1,df12_dp2 = link12.f_der_dp_func()*np.array(link12.pres_drop_func_der_p())
    df12_dm12 = link12.f_der_m()
    psi12 = link12.temp_drop_fac(m12)
    dpsi12_dm12 = link12.temp_drop_fac_dm(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    df23_dp2,df23_dp3 = link23.f_der_dp_func()*np.array(link23.pres_drop_func_der_p())
    df23_dm23 = link23.f_der_m()
    psi23 = link23.temp_drop_fac(m23)
    dpsi23_dm23 = link23.temp_drop_fac_dm(m23)
    m1_hl = -m12
    dm1_dm12 = -1
    Tr1_init = Tr_init[0]
    Ts2_init = Ts_init[0]
    Tr2_init = Tr_init[1]
    Ts2_hl = dT2+Tr2_init
    m2_hl = dphi2/(Cp*dT2)
    dm2_dTr2 = 0
    dTs2hl_dTr2 = 1
    Ts3_init = Ts_init[1]
    Tr3_init = Tr_init[2]
    Tr3_hl = Ts3_init - dT3
    m3_hl = dphi3/(Cp*dT3)
    dm3_dTs3 = 0
    dTr3hl_dTs3 = 1

    J_expected = np.array([[1,    -1,    0,    0,    0,    0,    0,    -dm2_dTr2,    0],
                           [0,    1,    0,    0,    0,    -dm3_dTs3,    0,    0,    0],
                           [df12_dm12,    0,    df12_dp2,    0,    0,    0,    0,    0,    0],
                           [0,    df23_dm23,    df23_dp2,    df23_dp3,    0,    0,    0,    0,    0],
                           [-(psi12*(Ts1-Ta)+Ta)-m12*dpsi12_dm12*(Ts1-Ta),     Ts2_init,     0,    0,    m23,    0,    0,    dm2_dTr2*Ts2_init+m2_hl*dTs2hl_dTr2,    0],
                           [0,    -(psi23*(Ts2_init-Ta)+Ta)-m23*dpsi23_dm23*(Ts2_init-Ta),    0,    0,    -m23*psi23,    dm3_dTs3*Ts3_init+m3_hl,    0,    0,    0],
                           [-(psi12*(Tr2_init-Ta)+Ta)-m12*dpsi12_dm12*(Tr2_init-Ta)-dm1_dm12*Tr1_init,    0,    0,    0,    0,    0,    -m1_hl,    -m12*psi12,    0],
                           [Tr2_init,    -(psi23*(Tr3_init-Ta)+Ta)-m23*dpsi23_dm23*(Tr3_init-Ta),    0,    0,    0,    0,    0,    m12-dm2_dTr2*Tr2_init-m2_hl,    -m23*psi23],
                           [0,    Tr3_init,    0,    0,    0,    -dm3_dTs3*Tr3_hl-m3_hl*dTr3hl_dTs3,    0,    0,    m23]])
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(sps.csr_matrix(J_expected)))
    print('J - J expected = \n{}'.format(J-sps.csr_matrix(J_expected)))
    assert np.allclose(J.todense(),J_expected)

def test_F_heat_unknown_half_link_3N_line_source_slack_delta_loads():
    """Test the system of equations for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink
    """
    # Given
    Ts1 = 101 #[C]
    p1 = 9.8*bar
    dphi2 = -.4*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Ts1=Ts1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='source',source_node='delta',sink_node='delta')

    m_init = np.array([2.1,3.4])
    m_hl_init = np.array([-.8,2.7])
    p_init  = np.array([10.1,9.4])*bar
    Ts_init  = np.array([103,97])
    Tr_init  = np.array([12,38,47])
    Ts_hl_init = np.array([102])
    Tr_hl_init = np.array([52.])
    x_init  = np.concatenate((m_init,m_hl_init,p_init ,Ts_init ,Tr_init,Ts_hl_init,Tr_hl_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    F = nlsys.F(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    m1_hl = -m12
    Tr1_init = Tr_init[0]
    Ts2_init = Ts_init[0]
    Tr2_init = Tr_init[1]
    Ts2_hl_init = Ts_hl_init[0]
    m2_hl_init = m_hl_init[0]
    Ts3_init = Ts_init[1]
    Tr3_init = Tr_init[2]
    Tr3_hl_init = Tr_hl_init[0]
    m3_hl_init = m_hl_init[1]

    F_expected = np.array([m12-m23-m2_hl_init,
                           m23 - m3_hl_init,
                           link12.link_equation(),
                           link23.link_equation(),
                           m23*Ts2_init - m12*(link12.temp_drop_fac(m12)*(Ts1-Ta)+Ta) + m2_hl_init*Ts2_hl_init,
                           -m23*(link23.temp_drop_fac(m23)*(Ts2_init-Ta)+Ta) + m3_hl_init*Ts3_init,
                           -m12*(link12.temp_drop_fac(m12)*(Tr2_init-Ta)+Ta) - m1_hl*Tr1_init,
                           -m23*(link23.temp_drop_fac(m23)*(Tr3_init-Ta)+Ta) + m12*Tr2_init -m2_hl_init*Tr2_init,
                           m23*Tr3_init - m3_hl_init*Tr3_hl_init,
                           -dphi2 + Cp*m2_hl_init*(Ts2_hl_init-Tr2_init),
                           -dphi3 + Cp*m3_hl_init*(Ts3_init-Tr3_hl_init),
                           Ts2_hl_init - Tr2_init - dT2,
                           Ts3_init - Tr3_hl_init - dT3])
    assert np.allclose(F,F_expected)

def test_J_heat_unknown_half_link_3N_line_source_slack_delta_loads():
    """Test the Jacobian for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink
    """
    # Given
    Ts1 = 101 #[C]
    p1 = 9.8*bar
    dphi2 = -.4*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Ts1=Ts1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='source',source_node='delta',sink_node='delta')

    m_init = np.array([2.1,3.4])
    m_hl_init = np.array([-.8,2.7])
    p_init  = np.array([10.1,9.4])*bar
    Ts_init  = np.array([103,97])
    Tr_init  = np.array([12,38,47])
    Ts_hl_init = np.array([102])
    Tr_hl_init = np.array([52.])
    x_init  = np.concatenate((m_init,m_hl_init,p_init ,Ts_init ,Tr_init,Ts_hl_init,Tr_hl_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    J = nlsys.J(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    df12_dp1,df12_dp2 = link12.f_der_dp_func()*np.array(link12.pres_drop_func_der_p())
    df12_dm12 = link12.f_der_m()
    psi12 = link12.temp_drop_fac(m12)
    dpsi12_dm12 = link12.temp_drop_fac_dm(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    df23_dp2,df23_dp3 = link23.f_der_dp_func()*np.array(link23.pres_drop_func_der_p())
    df23_dm23 = link23.f_der_m()
    psi23 = link23.temp_drop_fac(m23)
    dpsi23_dm23 = link23.temp_drop_fac_dm(m23)
    m1_hl = -m12
    dm1_dm12 = -1
    Tr1_init = Tr_init[0]
    Ts2_init = Ts_init[0]
    Tr2_init = Tr_init[1]
    Ts2_hl_init = Ts_hl_init[0]
    m2_hl_init = m_hl_init[0]
    Ts3_init = Ts_init[1]
    Tr3_init = Tr_init[2]
    Tr3_hl_init = Tr_hl_init[0]
    m3_hl_init = m_hl_init[1]

    J_expected = np.array([[1,    -1,    -1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
                           [0,    1,    0,    -1,    0,    0,    0,   0,    0,    0,    0,    0,    0],
                           [df12_dm12,    0,    0,    0,    df12_dp2,    0,    0,    0,    0,    0,    0,    0,    0],
                           [0,    df23_dm23,    0,    0,    df23_dp2,    df23_dp3,    0,    0,    0,    0,    0,    0,    0],
                           [-(psi12*(Ts1-Ta)+Ta)-m12*dpsi12_dm12*(Ts1-Ta),     Ts2_init,    Ts2_hl_init,    0,     0,    0,    m23,    0,    0,    0,    0,    m2_hl_init,    0],
                           [0,    -(psi23*(Ts2_init-Ta)+Ta)-m23*dpsi23_dm23*(Ts2_init-Ta),    0,    Ts3_init,   0,    0,    -m23*psi23,    m3_hl_init,    0,    0,    0,    0,    0],
                           [-(psi12*(Tr2_init-Ta)+Ta)-m12*dpsi12_dm12*(Tr2_init-Ta)-dm1_dm12*Tr1_init,    0,    0,    0,    0,    0,    0,    0,    -m1_hl,    -m12*psi12,    0,    0,    0],
                           [Tr2_init,    -(psi23*(Tr3_init-Ta)+Ta)-m23*dpsi23_dm23*(Tr3_init-Ta),    -Tr2_init,    0,    0,    0,    0,    0,    0,    m12-m2_hl_init,    -m23*psi23,    0,    0],
                           [0,    Tr3_init,    0,    -Tr3_hl_init,    0,    0,    0,    0,    0,    0,    m23,    0,    -m3_hl_init],
                           [0,    0,    Cp*(Ts2_hl_init-Tr2_init),    0,    0,    0,    0,    0,    0,   -Cp*m2_hl_init,    0,    Cp*m2_hl_init,    0],
                           [0,    0,    0,    Cp*(Ts3_init-Tr3_hl_init),    0,    0,    0,    Cp*m3_hl_init,    0,    0,    0,    0,    -Cp*m3_hl_init],
                           [0,    0,    0,    0,    0,    0,    0,    0,    0,    -1,    0,    1,    0],
                           [0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    -1]])
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(sps.csr_matrix(J_expected)))
    print('J - J expected = \n{}'.format(J-sps.csr_matrix(J_expected)))
    assert np.allclose(J.todense(),J_expected)

def test_F_heat_standard_3N_line_general_Ts_slack():
    """Test the system of equations for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. source slack, node 2 is a source, and node 3 is taken as a sink
    """
    # Given
    heat_net = create_test_heat_network_3N_line(slack_node='general_Ts')
    m_init = np.array([1.,1.])
    p_init = np.array([8.,6.])*bar
    Ts_init = np.array([100.,100.])
    Tr_init = np.array([50.,50.,50.])
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='standard')

    # When
    F = nlsys.F(x_init)

    # Then
    F_m_expected = np.array([9.1746659732073184,-3.7824007651841223]) #[kg/s]
    F_l_expected = np.array([198843.63203305315,198554.54004131645])
    F_Ts_expected = np.array([-916.11900521420364,397.38124338637829])
    F_Tr_expected = np.array([6.9644774474204354,467.24048393501749,-189.12003825920613])
    F_expected = np.concatenate((F_m_expected,F_l_expected,F_Ts_expected,F_Tr_expected))

    assert np.allclose(F,F_expected)

def test_F_heat_unknown_half_link_3N_line_sink_slack():
    """Test the system of equations for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink
    """
    # Given
    heat_net = create_test_heat_network_3N_line(slack_node='sink')
    m_init = np.array([-1.,1.])
    p_init = np.array([8.,6.])*bar
    Ts_init = np.array([100.,100.,100.])
    Tr_init = np.array([50.,50.])
    x_init_standard = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    heat_net.update(x_init_standard)
    m_hl_init = np.array([heat_net.nodes[1].half_links[0].flow(),heat_net.nodes[2].half_links[0].flow()])
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    F = nlsys.F(x_init)

    # Then
    F_m_expected = np.array([7.1746659732073184,-3.7824007651841223]) #[kg/s]
    F_l_expected = np.array([201156.36796694685,198554.54004131645])
    F_Ts_expected = np.array([15.670074256695969,-736.97337517877816,397.38124338637829])
    F_Tr_expected = np.array([418.1537457821799,-189.12003825920613])
    F_phi_expected = np.array([0.,0.])
    F_expected = np.concatenate((F_m_expected,F_l_expected,F_Ts_expected,F_Tr_expected,F_phi_expected))

    assert np.allclose(F,F_expected)

def test_F_heat_3N_line_sink_slack_delta_loads():
    """Test the system of equations for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink.
    """
    # Given
    Tr1 = 40.
    p1 = 9.8*bar
    dphi2 = -2.8*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Tr1=Tr1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='sink',source_node='delta',sink_node='delta')
    m_init = np.array([-2.1,3.4])
    p_init = np.array([10.1,6.7])*bar
    Ts_init = np.array([103.,97.,88.])
    Tr_init = np.array([38.,52.])
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='standard')

    # When
    F = nlsys.F(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    psi12 = link12.temp_drop_fac(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    psi23 = link23.temp_drop_fac(m23)
    m1_hl = -m12
    Ts1_init = Ts_init[0]
    Ts2_init = Ts_init[1]
    Tr2_init = Tr_init[0]
    Ts2_hl = dT2+Tr2_init
    m2_hl = dphi2/(Cp*dT2)
    Ts3_init = Ts_init[2]
    Tr3_init = Tr_init[1]
    Tr3_hl = Ts3_init - dT3
    m3_hl = dphi3/(Cp*dT3)

    F_expected = np.array([m12-m23-m2_hl, #Fm2
                           m23 - m3_hl, #Fm3
                           link12.link_equation(),#Fdp12
                           link23.link_equation(), #Fdp23
                           m12*(psi12*(Ts2_init-Ta)+Ta)+m1_hl*Ts1_init, #FTs1
                           m23*Ts2_init-m12*Ts2_init+m2_hl*Ts2_hl, #FTs2
                           -m23*(psi23*(Ts2_init-Ta)+Ta)+m3_hl*Ts3_init, #FTs3
                           -m23*(psi23*(Tr3_init-Ta)+Ta) + m12*(psi12*(Tr1-Ta)+Ta) -m2_hl*Tr2_init, #FTr2
                           m23*Tr3_init - m3_hl*Tr3_hl]) #FTr3
    print('F = \n{}'.format(F))
    print('F expected = \n{}'.format(F_expected))
    print('F - F exp = \n{}'.format(F-F_expected))
    assert np.allclose(F,F_expected)

def test_J_heat_3N_line_sink_slack_delta_loads():
    """Test the Jacobian for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink.
    """
    # Given
    Tr1 = 40.
    p1 = 9.8*bar
    dphi2 = -2.8*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Tr1=Tr1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='sink',source_node='delta',sink_node='delta')
    m_init = np.array([-2.1,3.4])
    p_init = np.array([10.1,6.7])*bar
    Ts_init = np.array([103.,97.,88.])
    Tr_init = np.array([38.,52.])
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='standard')

    # When
    J = nlsys.J(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    df12_dp1,df12_dp2 = link12.f_der_dp_func()*np.array(link12.pres_drop_func_der_p())
    df12_dm12 = link12.f_der_m()
    psi12 = link12.temp_drop_fac(m12)
    dpsi12_dm12 = link12.temp_drop_fac_dm(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    df23_dp2,df23_dp3 = link23.f_der_dp_func()*np.array(link23.pres_drop_func_der_p())
    df23_dm23 = link23.f_der_m()
    psi23 = link23.temp_drop_fac(m23)
    dpsi23_dm23 = link23.temp_drop_fac_dm(m23)
    m1_hl = -m12
    dm1_dm12 = -1
    Ts1_init = Ts_init[0]
    Ts2_init = Ts_init[1]
    Tr2_init = Tr_init[0]
    Ts2_hl = dT2+Tr2_init
    m2_hl = dphi2/(Cp*dT2)
    dm2_dTr2 = 0
    dTs2hl_dTr2 = 1
    Ts3_init = Ts_init[2]
    Tr3_init = Tr_init[1]
    Tr3_hl = Ts3_init - dT3
    m3_hl = dphi3/(Cp*dT3)
    dm3_dTs3 = 0
    dTr3hl_dTs3 = 1

    J_expected = np.array([[1,    -1,    0,    0,    0,    0,    0,    -dm2_dTr2,    0],
                           [0,    1,    0,    0,    0,    -dm3_dTs3,    0,    0,    0],
                           [df12_dm12,    0,    df12_dp2,    0,    0,    0,    0,    0,    0],
                           [0,    df23_dm23,    df23_dp2,    df23_dp3,    0,    0,    0,    0,    0],
                           [psi12*(Ts2_init-Ta)+Ta+m12*dpsi12_dm12*(Ts2_init-Ta)+dm1_dm12*Ts1_init,     0,     0,    0,    m1_hl,    m12*psi12,    0,    0,    0],    #FTs1
                           [-Ts2_init,    Ts2_init,    0,    0,    0,    m23-m12,    0,    dm2_dTr2*Ts2_hl+m2_hl*dTs2hl_dTr2,    0],    #FTs2
                           [0,    -(psi23*(Ts2_init-Ta)+Ta)-m23*dpsi23_dm23*(Ts2_init-Ta),    0,    0,    0,    -m23*psi23,    dm3_dTs3*Ts3_init+m3_hl,    0,    0],    #FTs3
                           [psi12*(Tr1-Ta)+Ta+m12*dpsi12_dm12*(Tr1-Ta),    -(psi23*(Tr3_init-Ta)+Ta)-m23*dpsi23_dm23*(Tr3_init-Ta),    0,    0,    0,    0,    0,    -dm2_dTr2*Tr2_init-m2_hl,    -m23*psi23],    #FTr2
                           [0,    Tr3_init,    0,    0,    0,   0,     -dm3_dTs3*Tr3_hl-m3_hl*dTr3hl_dTs3,    0,    m23]]) #FTr3
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(sps.csr_matrix(J_expected)))
    print('J - J expected = \n{}'.format(J-sps.csr_matrix(J_expected)))
    assert np.allclose(J.todense(),J_expected)

def test_F_heat_unknown_half_link_3N_line_sink_slack_delta_loads():
    """Test the system of equations for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink.
    """
    # Given
    Tr1 = 40.
    p1 = 9.8*bar
    dphi2 = -2.8*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Tr1=Tr1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='sink',source_node='delta',sink_node='delta')
    m_init = np.array([-2.1,3.4])
    m_hl_init = np.array([-.8,2.7])
    p_init = np.array([10.1,6.7])*bar
    Ts_init = np.array([103.,97.,88.])
    Tr_init = np.array([38.,52.])
    Ts_hl_init = np.array([102])
    Tr_hl_init = np.array([49.])
    x_init  = np.concatenate((m_init,m_hl_init,p_init ,Ts_init ,Tr_init,Ts_hl_init,Tr_hl_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    F = nlsys.F(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    psi12 = link12.temp_drop_fac(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    psi23 = link23.temp_drop_fac(m23)
    m1_hl = -m12
    Ts1_init = Ts_init[0]
    Ts2_init = Ts_init[1]
    Tr2_init = Tr_init[0]
    Ts2_hl_init = Ts_hl_init[0]
    m2_hl_init = m_hl_init[0]
    Ts3_init = Ts_init[2]
    Tr3_init = Tr_init[1]
    Tr3_hl_init = Tr_hl_init[0]
    m3_hl_init = m_hl_init[1]

    F_expected = np.array([m12-m23-m2_hl_init, #Fm2
                           m23 - m3_hl_init, #Fm3
                           link12.link_equation(),#Fdp12
                           link23.link_equation(), #Fdp23
                           m12*(psi12*(Ts2_init-Ta)+Ta)+m1_hl*Ts1_init, #FTs1
                           m23*Ts2_init-m12*Ts2_init+m2_hl_init*Ts2_hl_init, #FTs2
                           -m23*(psi23*(Ts2_init-Ta)+Ta)+m3_hl_init*Ts3_init, #FTs3
                           -m23*(psi23*(Tr3_init-Ta)+Ta) + m12*(psi12*(Tr1-Ta)+Ta) -m2_hl_init*Tr2_init, #FTr2
                           m23*Tr3_init - m3_hl_init*Tr3_hl_init, #FTr3
                           -dphi2 + Cp*m2_hl_init*(Ts2_hl_init-Tr2_init), #Fdphi2
                           -dphi3 + Cp*m3_hl_init*(Ts3_init-Tr3_hl_init), #Fdphi3
                           Ts2_hl_init - Tr2_init - dT2, #FdT2
                           Ts3_init - Tr3_hl_init - dT3]) #FdT3
    print('F = \n{}'.format(F))
    print('F expected = \n{}'.format(F_expected))
    print('F - F exp = \n{}'.format(F-F_expected))
    assert np.allclose(F,F_expected)

def test_J_heat_unknown_half_link_3N_line_sink_slack_delta_loads():
    """Test the Jacobian for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink.
    """
    # Given
    Tr1 = 40.
    p1 = 9.8*bar
    dphi2 = -2.8*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Tr1=Tr1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='sink',source_node='delta',sink_node='delta')
    m_init = np.array([-2.1,3.4])
    m_hl_init = np.array([-.8,2.7])
    p_init = np.array([10.1,6.7])*bar
    Ts_init = np.array([103.,97.,88.])
    Tr_init = np.array([38.,52.])
    Ts_hl_init = np.array([102])
    Tr_hl_init = np.array([49.])
    x_init  = np.concatenate((m_init,m_hl_init,p_init ,Ts_init ,Tr_init,Ts_hl_init,Tr_hl_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    J = nlsys.J(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    df12_dp1,df12_dp2 = link12.f_der_dp_func()*np.array(link12.pres_drop_func_der_p())
    df12_dm12 = link12.f_der_m()
    psi12 = link12.temp_drop_fac(m12)
    dpsi12_dm12 = link12.temp_drop_fac_dm(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    df23_dp2,df23_dp3 = link23.f_der_dp_func()*np.array(link23.pres_drop_func_der_p())
    df23_dm23 = link23.f_der_m()
    psi23 = link23.temp_drop_fac(m23)
    dpsi23_dm23 = link23.temp_drop_fac_dm(m23)
    m1_hl = -m12
    dm1_dm12 = -1
    Ts1_init = Ts_init[0]
    Ts2_init = Ts_init[1]
    Tr2_init = Tr_init[0]
    Ts2_hl_init = Ts_hl_init[0]
    m2_hl_init = m_hl_init[0]
    Ts3_init = Ts_init[2]
    Tr3_init = Tr_init[1]
    Tr3_hl_init = Tr_hl_init[0]
    m3_hl_init = m_hl_init[1]

    J_expected = np.array([[1,    -1,    -1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0], #Fm2
                           [0,    1,    0,    -1,    0,    0,    0,   0,    0,    0,    0,    0,    0], #Fm3
                           [df12_dm12,    0,    0,    0,    df12_dp2,    0,    0,    0,    0,    0,    0,    0,    0], #Fdp12
                           [0,    df23_dm23,    0,    0,    df23_dp2,    df23_dp3,    0,    0,    0,    0,    0,    0,    0], #Fdp23
                           [psi12*(Ts2_init-Ta)+Ta+m12*dpsi12_dm12*(Ts2_init-Ta)+dm1_dm12*Ts1_init,     0,    0,    0,     0,    0,    m1_hl,    m12*psi12,    0,    0,    0,    0,    0], #FTs1
                           [-Ts2_init,   Ts2_init,    Ts2_hl_init,    0,   0,    0,    0,    m23-m12,    0,    0,    0,    m2_hl_init,    0], #FTs2
                           [0,    -(psi23*(Ts2_init-Ta)+Ta)-m23*dpsi23_dm23*(Ts2_init-Ta),    0,    Ts3_init,    0,    0,    0,    -m23*psi23,    m3_hl_init,    0,    0,    0,    0], #FTs3
                           [psi12*(Tr1-Ta)+Ta+m12*dpsi12_dm12*(Tr1-Ta),    -(psi23*(Tr3_init-Ta)+Ta)-m23*dpsi23_dm23*(Tr3_init-Ta),    -Tr2_init,    0,    0,    0,    0,    0,    0,    -m2_hl_init,    -m23*psi23,    0,    0], #FTr2
                           [0,    Tr3_init,    0,    -Tr3_hl_init,    0,    0,    0,    0,    0,    0,    m23,    0,    -m3_hl_init], #FTr3
                           [0,    0,    Cp*(Ts2_hl_init-Tr2_init),    0,    0,    0,    0,    0,    0,   -Cp*m2_hl_init,    0,    Cp*m2_hl_init,    0], #Fdphi2
                           [0,    0,    0,    Cp*(Ts3_init-Tr3_hl_init),    0,    0,    0,    0,    Cp*m3_hl_init,    0,    0,    0,    -Cp*m3_hl_init], #Fdphi3
                           [0,    0,    0,    0,    0,    0,    0,    0,    0,    -1,    0,    1,    0], #FdT2
                           [0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    -1]]) #FdT3
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(sps.csr_matrix(J_expected)))
    print('J - J expected = \n{}'.format(J-sps.csr_matrix(J_expected)))
    assert np.allclose(J.todense(),J_expected)

def test_F_heat_unknown_half_link_3N_line_general_slack():
    """Test the system of equations for a heat network of three nodes (1-3) in a line. Node 1 is taken as general ref. slack, node 2 is a source, and node 3 is taken as a sink.

    NB. Is ill-posed, I think. The return line mixing rule for node 1 is useless...
    """
    # Given
    heat_net = create_test_heat_network_3N_line(slack_node='general')
    m_init = np.array([-1.,1.])
    p_init = np.array([8.,6.])*bar
    Ts_init = np.array([100.,100.,100.])
    Tr_init = np.array([50.,50.,50.])
    x_init_standard = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    heat_net.update(x_init_standard)
    m_hl_init = np.array([heat_net.nodes[1].half_links[0].flow(),heat_net.nodes[2].half_links[0].flow()])
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    F = nlsys.F(x_init)

    # Then
    F_m_expected = np.array([7.1746659732073184,-3.7824007651841223]) #[kg/s]
    F_l_expected = np.array([201156.36796694685,198554.54004131645])
    F_Ts_expected = np.array([15.670074256695969,-736.97337517877816,397.38124338637829])
    F_Tr_expected = np.array([0.,374.20496138243794,-189.12003825920613])
    F_phi_expected = np.array([0.,0.])
    F_expected = np.concatenate((F_m_expected,F_l_expected,F_Ts_expected,F_Tr_expected,F_phi_expected))

    assert np.allclose(F,F_expected)

def test_J_heat_unknown_half_link_3N_line_sink_slack():
    """Test the Jacobian for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink
    """
    # Given
    heat_net = create_test_heat_network_3N_line(slack_node='sink')
    m_init = np.array([-1.,1.])
    p_init = np.array([8.,6.])*bar
    Ts_init = np.array([100.,100.,100.])
    Tr_init = np.array([50.,50.])
    x_init_standard = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    heat_net.update(x_init_standard)
    m_hl_init = np.array([heat_net.nodes[1].half_links[0].flow(),heat_net.nodes[2].half_links[0].flow()])
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    J = nlsys.J(x_init)

    # Then
    Cp = heat_net.links[0].link_params.get('carrier').Cp
    To2 = heat_net.nodes[1].half_links[0].Ts #source
    To3 = heat_net.nodes[2].half_links[0].Tr #sink
    dFm_dm = sps.csr_matrix((np.array([1,-1,1]),(np.array([0,0,1]),np.array([0,1,1]))),shape=(2,2)) # data,(row,col)
    dFl_dm = sps.diags(np.array([-2312.7359338936913, -2890.9199173671141])).tocsr()
    dFTs_dm = sps.csr_matrix((np.array([-1.451054506661734,-Ts_init[1],Ts_init[1],-97.802600021568637]),(np.array([0,1,1,2]),np.array([0,0,1,1]))),shape=(3,2))
    dFTr_dm = sps.csr_matrix((np.array([-3.0009255195033511,-49.023377787363842,Tr_init[1]]),(np.array([0,0,1]),np.array([0,1,1]))),shape=(2,2))
    dFm_dmhl = -1*sps.eye(2).tocsr()
    dFTs_dmhl = sps.csr_matrix((np.array([To2,Ts_init[2]]),(np.array([1,2]),np.array([0,1]))),shape=(3,2))
    dFTr_dmhl = sps.csr_matrix((np.array([-Tr_init[0],-To3]),(np.array([0,1]),np.array([0,1]))),shape=(2,2))
    dFphi_dmhl = sps.diags(np.array([Cp*(To2-Tr_init[0]),Cp*(Ts_init[2]-To3)])).tocsr()
    dFl_dp = sps.csr_matrix((np.array([-1,1,-1]),(np.array([0,1,1]),np.array([0,0,1]))),shape=(2,2))
    dFTs_dTs = sps.csr_matrix((np.array([-m_init[0],-0.8258880638144892,np.abs(m_init[0])+m_init[1],-0.78732036813371076,m_hl_init[1]]),(np.array([0,0,1,2,2]),np.array([0,1,1,1,2]))),shape=(3,3))
    dFphi_dTs = sps.csr_matrix((np.array([Cp*m_hl_init[1]]),(np.array([1]),np.array([2]))),shape=(2,3))
    dFTr_dTr = sps.csr_matrix((np.array([-m_hl_init[0],-0.78732036813371076,m_init[1]]),(np.array([0,0,1]),np.array([0,1,1]))),shape=(2,2))
    dFphi_dTr = sps.csr_matrix((np.array([-Cp*m_hl_init[0]]),(np.array([0]),np.array([0]))),shape=(2,2))

    J_expected = sps.bmat([
                [dFm_dm,dFm_dmhl,None,None,None],
                 [dFl_dm,None,dFl_dp,None,None],
                 [dFTs_dm,dFTs_dmhl,None,dFTs_dTs,None],
                 [dFTr_dm,dFTr_dmhl,None,None,dFTr_dTr],
                 [None,dFphi_dmhl,None,dFphi_dTs,dFphi_dTr]])
    print('dFTs_dTs expected = \n{}'.format(dFTs_dTs.todense()))
    print('dFTr_dTr expected = \n{}'.format(dFTr_dTr.todense()))
    assert np.allclose(J.todense(),J_expected.todense())

def test_J_heat_unknown_half_link_3N_line_general_slack():
    """Test the Jacobian for a heat network of three nodes (1-3) in a line. Node 1 is taken as general ref. slack, node 2 is a source, and node 3 is taken as a sink.

    NB. Is ill-posed, I think. The return line mixing rule for node 1 is useless...
    """
    # Given
    heat_net = create_test_heat_network_3N_line(slack_node='general')
    m_init = np.array([-1.,1.])
    p_init = np.array([8.,6.])*bar
    Ts_init = np.array([100.,100.,100.])
    Tr_init = np.array([50.,50.,50.])
    x_init_standard = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    heat_net.update(x_init_standard)
    m_hl_init = np.array([heat_net.nodes[1].half_links[0].flow(),heat_net.nodes[2].half_links[0].flow()])
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    J = nlsys.J(x_init)

    # Then
    Cp = heat_net.links[0].link_params.get('carrier').Cp
    To2 = heat_net.nodes[1].half_links[0].Ts #source
    To3 = heat_net.nodes[2].half_links[0].Tr #sink
    dFm_dm = sps.csr_matrix((np.array([1,-1,1]),(np.array([0,0,1]),np.array([0,1,1]))),shape=(2,2)) # data,(row,col)
    dFl_dm = sps.diags(np.array([-2312.7359338936913, -2890.9199173671141])).tocsr()
    dFTs_dm = sps.csr_matrix((np.array([-1.451054506661734,-Ts_init[1],Ts_init[1],-97.802600021568637]),(np.array([0,1,1,2]),np.array([0,0,1,1]))),shape=(3,2))
    dFTr_dm = sps.csr_matrix((np.array([49.355086885928124,-49.023377787363842,Tr_init[2]]),(np.array([1,1,2]),np.array([0,1,1]))),shape=(3,2))
    dFm_dmhl = -1*sps.eye(2).tocsr()
    dFTs_dmhl = sps.csr_matrix((np.array([To2,Ts_init[2]]),(np.array([1,2]),np.array([0,1]))),shape=(3,2))
    dFTr_dmhl = sps.csr_matrix((np.array([-Tr_init[1],-To3]),(np.array([1,2]),np.array([0,1]))),shape=(3,2))
    dFphi_dmhl = sps.diags(np.array([Cp*(To2-Tr_init[1]),Cp*(Ts_init[2]-To3)])).tocsr()
    dFl_dp = sps.csr_matrix((np.array([-1,1,-1]),(np.array([0,1,1]),np.array([0,0,1]))),shape=(2,2))
    dFTs_dTs = sps.csr_matrix((np.array([-m_init[0],-0.8258880638144892,np.abs(m_init[0])+m_init[1],-0.78732036813371076,m_hl_init[1]]),(np.array([0,0,1,2,2]),np.array([0,1,1,1,2]))),shape=(3,3))
    dFphi_dTs = sps.csr_matrix((np.array([Cp*m_hl_init[1]]),(np.array([1]),np.array([2]))),shape=(2,3))
    dFTr_dTr = sps.csr_matrix((np.array([-0.8258880638144892,-m_hl_init[0],-0.78732036813371076,m_init[1]]),(np.array([1,1,1,2]),np.array([0,1,2,2]))),shape=(3,3))
    dFphi_dTr = sps.csr_matrix((np.array([-Cp*m_hl_init[0]]),(np.array([0]),np.array([1]))),shape=(2,3))

    print('dFTs_dm expected=\n{}'.format(dFTs_dm.todense()))
    print('dFTr_dm expected=\n{}'.format(dFTr_dm.todense()))
    print('dFTr_dmhl expected=\n{}'.format(dFTr_dmhl.todense()))
    print('dFTr_dTr expected=\n{}'.format(dFTr_dTr.todense()))
    J_expected = sps.bmat([
                [dFm_dm,dFm_dmhl,None,None,None],
                 [dFl_dm,None,dFl_dp,None,None],
                 [dFTs_dm,dFTs_dmhl,None,dFTs_dTs,None],
                 [dFTr_dm,dFTr_dmhl,None,None,dFTr_dTr],
                 [None,dFphi_dmhl,None,dFphi_dTs,dFphi_dTr]])

    print('J_expected = \n{}'.format(J_expected))
    print('J = \n{}'.format(J))
    print('J_expected - J = \n{}'.format(J_expected - J))
    assert np.linalg.det(J.todense())==0#np.allclose(J.todense(),J_expected.todense())

def test_J_heat_unknown_half_link_3N_line_general_slack_ill_posedness():
    """Test the ill-posedness for a heat network of three nodes (1-3) in a line. Node 1 is taken as general ref. slack, node 2 is a source, and node 3 is taken as a sink. Test by checking that FTr has a zero row for node 1 (implying |J|==0)

    NB. Is ill-posed, I think. The return line mixing rule for node 1 is useless...
    """
    # Given
    heat_net = create_test_heat_network_3N_line(slack_node='general')
    m_init = np.array([-1.,1.])
    p_init = np.array([8.,6.])*bar
    Ts_init = np.array([100.,100.,100.])
    Tr_init = np.array([50.,50.,50.])
    x_init_standard = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    heat_net.update(x_init_standard)
    m_hl_init = np.array([heat_net.nodes[1].half_links[0].flow(),heat_net.nodes[2].half_links[0].flow()])
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    J = nlsys.J(x_init)

    # Then
    # 7 is the row index of FTr for node 1. The corresponding column indices are stored in indices[indptr[7]:indptr[7+1]]. So if that one is empty, then the row is empty
    print('number of entries in row 7 = {}'.format(len(J.indices[J.indptr[7]:J.indptr[7+1]])))
    if len(J.indices[J.indptr[7]:J.indptr[7+1]])==0:
        zero_row = True
    else:
        print('entries in row 4 = \n{}'.format(J.data[J.indptr[7]:J.indptr[7+1]]))
        if np.all(J.data[J.indptr[7]:J.indptr[7+1]]==0):
            zero_row = True
        else:
            zero_row = False
    assert zero_row

def test_F_heat_3N_line_source_slack_delta_loads_opposite_flow():
    """Test the system of equations for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink. The initial value of the link flow is chosen such that node 2 has only inflow and node 3 has only outflow in the supply line.
    """
    # Given
    Ts1 = 101 #[C]
    p1 = 9.8*bar
    dphi2 = -.4*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Ts1=Ts1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='source',source_node='delta',sink_node='delta')

    m_init = np.array([2.1,-3.4])
    p_init  = np.array([10.1,9.4])*bar
    Ts_init  = np.array([103,97])
    Tr_init  = np.array([12,38,47])
    x_init  = np.concatenate((m_init,p_init ,Ts_init ,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='standard')

    # When
    F = nlsys.F(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    m1_hl = -m12
    Tr1_init = Tr_init[0]
    Ts2_init = Ts_init[0]
    Tr2_init = Tr_init[1]
    Ts2_hl = dT2+Tr2_init
    m2_hl = dphi2/(Cp*dT2)
    Ts3_init = Ts_init[1]
    Tr3_init = Tr_init[2]
    Tr3_hl = Ts3_init - dT3
    m3_hl = dphi3/(Cp*dT3)

    F_expected = np.array([m12-m23-m2_hl, #Fm1
                           m23 - m3_hl, #Fm2
                           link12.link_equation(), #Fdp12
                           link23.link_equation(), #Fdp23
                           m23*(link23.temp_drop_fac(m23)*(Ts3_init-Ta)+Ta) - m12*(link12.temp_drop_fac(m12)*(Ts1-Ta)+Ta) + m2_hl*Ts2_hl + (-m23+m12-m2_hl)*Ts2_init, #FTs2
                           -m23*Ts3_init + m3_hl*Ts3_init, #FTs3
                           -m12*(link12.temp_drop_fac(m12)*(Tr2_init-Ta)+Ta) - m1_hl*Tr1_init, #FTr1
                           -m23*Tr2_init + m12*Tr2_init -m2_hl*Tr2_init, #FTr2
                           m23*(link23.temp_drop_fac(m23)*(Tr2_init-Ta)+Ta) - m3_hl*Tr3_hl+ (-m23+m3_hl)*Tr3_init]) #FTr3
    print('F=\n{}'.format(F))
    print('F expected = \n{}'.format(F_expected))
    print('F-F expected = \n{}'.format(F-F_expected))
    assert np.allclose(F,F_expected)

def test_J_heat_3N_line_source_slack_delta_loads_opposite_flow():
    """Test the Jacobian for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink. The initial value of the link flow is chosen such that node 2 has only inflow and node 3 has only outflow in the supply line.
    """
    # Given
    Ts1 = 101 #[C]
    p1 = 9.8*bar
    dphi2 = -.4*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Ts1=Ts1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='source',source_node='delta',sink_node='delta')

    m_init = np.array([2.1,-3.4])
    p_init  = np.array([10.1,9.4])*bar
    Ts_init  = np.array([103,97])
    Tr_init  = np.array([12,38,47])
    x_init  = np.concatenate((m_init,p_init ,Ts_init ,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='standard')

    # When
    J = nlsys.J(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    df12_dp1,df12_dp2 = link12.f_der_dp_func()*np.array(link12.pres_drop_func_der_p())
    df12_dm12 = link12.f_der_m()
    psi12 = link12.temp_drop_fac(m12)
    dpsi12_dm12 = link12.temp_drop_fac_dm(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    df23_dp2,df23_dp3 = link23.f_der_dp_func()*np.array(link23.pres_drop_func_der_p())
    df23_dm23 = link23.f_der_m()
    psi23 = link23.temp_drop_fac(m23)
    dpsi23_dm23 = link23.temp_drop_fac_dm(m23)
    m1_hl = -m12
    dm1_dm12 = -1
    Tr1_init = Tr_init[0]
    Ts2_init = Ts_init[0]
    Tr2_init = Tr_init[1]
    Ts2_hl = dT2+Tr2_init
    m2_hl = dphi2/(Cp*dT2)
    dm2_dTr2 = 0
    dTs2hl_dTr2 = 1
    Ts3_init = Ts_init[1]
    Tr3_init = Tr_init[2]
    Tr3_hl = Ts3_init - dT3
    m3_hl = dphi3/(Cp*dT3)
    dm3_dTs3 = 0
    dTr3hl_dTs3 = 1

    J_expected = np.array([[1,    -1,    0,    0,    0,    0,    0,    -dm2_dTr2,    0], #Fm1
                           [0,    1,    0,    0,    0,    -dm3_dTs3,    0,    0,    0],    #Fm2
                           [df12_dm12,    0,    df12_dp2,    0,    0,    0,    0,    0,    0],    #Fdp12
                           [0,    df23_dm23,    df23_dp2,    df23_dp3,    0,    0,    0,    0,    0],    #Fdp23
                           [-(psi12*(Ts1-Ta)+Ta)-m12*dpsi12_dm12*(Ts1-Ta)+Ts2_init,    (psi23*(Ts3_init-Ta)+Ta)+m23*dpsi23_dm23*(Ts3_init-Ta) - Ts2_init,     0,    0,    -m23+m12-m2_hl,    m23*psi23,    0,    dm2_dTr2*Ts2_init+m2_hl*dTs2hl_dTr2 -dm2_dTr2*Ts2_init,    0],    #FTs2
                           [0,    -Ts3_init,    0,    0,   0,    -m23 + dm3_dTs3*Ts3_init+m3_hl,    0,    0,    0],    #FTs3
                           [-(psi12*(Tr2_init-Ta)+Ta)-m12*dpsi12_dm12*(Tr2_init-Ta)-dm1_dm12*Tr1_init,    0,    0,    0,    0,    0,    -m1_hl,    -m12*psi12,    0],    #FTr1
                           [Tr2_init,    -Tr2_init,    0,    0,    0,    0,    0,    -m23+m12-dm2_dTr2*Tr2_init-m2_hl,    0],    #FTr2
                           [0,     (psi23*(Tr2_init-Ta)+Ta)+m23*dpsi23_dm23*(Tr2_init-Ta) - Tr3_init,    0,    0,    0,    -dm3_dTs3*Tr3_hl-m3_hl*dTr3hl_dTs3+dm3_dTs3*Tr3_init,    0,    m23*psi23,    -m23+m3_hl]]) #FTr3
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(sps.csr_matrix(J_expected)))
    print('J - J expected = \n{}'.format(J-sps.csr_matrix(J_expected)))
    assert np.allclose(J.todense(),J_expected)

def test_F_heat_unknown_half_link_3N_line_source_slack_delta_loads_opposite_flow():
    """Test the system of equations for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink. The initial value of the link flow is chosen such that node 2 has only inflow and node 3 has only outflow in the supply line. The unknown half link flow formulation is used.
    """
    # Given
    Ts1 = 101 #[C]
    p1 = 9.8*bar
    dphi2 = -.4*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Ts1=Ts1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='source',source_node='delta',sink_node='delta')

    m_init = np.array([2.1,-3.4])
    m_hl_init = np.array([-.8,2.7])
    p_init  = np.array([10.1,9.4])*bar
    Ts_init  = np.array([103,97])
    Tr_init  = np.array([12,38,47])
    Ts_hl_init = np.array([102])
    Tr_hl_init = np.array([52.])
    x_init  = np.concatenate((m_init,m_hl_init,p_init ,Ts_init ,Tr_init,Ts_hl_init,Tr_hl_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    F = nlsys.F(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    m1_hl = -m12
    Tr1_init = Tr_init[0]
    Ts2_init = Ts_init[0]
    Tr2_init = Tr_init[1]
    Ts2_hl_init = Ts_hl_init[0]
    m2_hl_init = m_hl_init[0]
    Ts3_init = Ts_init[1]
    Tr3_init = Tr_init[2]
    Tr3_hl_init = Tr_hl_init[0]
    m3_hl_init = m_hl_init[1]

    F_expected = np.array([m12-m23-m2_hl_init, #Fm2
                           m23 - m3_hl_init, #Fm3
                           link12.link_equation(), #Fdp12
                           link23.link_equation(), #Fdp23
                           m23*(link23.temp_drop_fac(m23)*(Ts3_init-Ta)+Ta) - m12*(link12.temp_drop_fac(m12)*(Ts1-Ta)+Ta) + m2_hl_init*Ts2_hl_init + (-m23+m12-m2_hl_init)*Ts2_init, #FTs2
                           -m23*Ts3_init + m3_hl_init*Ts3_init, #FTs3
                           -m12*(link12.temp_drop_fac(m12)*(Tr2_init-Ta)+Ta) - m1_hl*Tr1_init, #FTr1
                           -m23*Tr2_init + m12*Tr2_init -m2_hl_init*Tr2_init, #FTr2
                           m23*(link23.temp_drop_fac(m23)*(Tr2_init-Ta)+Ta) - m3_hl_init*Tr3_hl_init+ (-m23+m3_hl_init)*Tr3_init, #FTr3
                           -dphi2 + Cp*m2_hl_init*(Ts2_hl_init-Tr2_init), #Fdphi2
                           -dphi3 + Cp*m3_hl_init*(Ts3_init-Tr3_hl_init), #Fdphi3
                           Ts2_hl_init - Tr2_init - dT2, #FdT2
                           Ts3_init - Tr3_hl_init - dT3]) #FdT3
    assert np.allclose(F,F_expected)

def test_J_heat_unknown_half_link_3N_line_source_slack_delta_loads_opposite_flow():
    """Test the Jacobian for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink. The initial value of the link flow is chosen such that node 2 has only inflow and node 3 has only outflow in the supply line. The unknown half link flow formulation is used.
    """
    # Given
    Ts1 = 101 #[C]
    p1 = 9.8*bar
    dphi2 = -.4*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Ts1=Ts1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='source',source_node='delta',sink_node='delta')

    m_init = np.array([2.1,-3.4])
    m_hl_init = np.array([-.8,2.7])
    p_init  = np.array([10.1,9.4])*bar
    Ts_init  = np.array([103,97])
    Tr_init  = np.array([12,38,47])
    Ts_hl_init = np.array([102])
    Tr_hl_init = np.array([52.])
    x_init  = np.concatenate((m_init,m_hl_init,p_init ,Ts_init ,Tr_init,Ts_hl_init,Tr_hl_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    J = nlsys.J(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    df12_dp1,df12_dp2 = link12.f_der_dp_func()*np.array(link12.pres_drop_func_der_p())
    df12_dm12 = link12.f_der_m()
    psi12 = link12.temp_drop_fac(m12)
    dpsi12_dm12 = link12.temp_drop_fac_dm(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    df23_dp2,df23_dp3 = link23.f_der_dp_func()*np.array(link23.pres_drop_func_der_p())
    df23_dm23 = link23.f_der_m()
    psi23 = link23.temp_drop_fac(m23)
    dpsi23_dm23 = link23.temp_drop_fac_dm(m23)
    m1_hl = -m12
    dm1_dm12 = -1
    Tr1_init = Tr_init[0]
    Ts2_init = Ts_init[0]
    Tr2_init = Tr_init[1]
    Ts2_hl_init = Ts_hl_init[0]
    m2_hl_init = m_hl_init[0]
    Ts3_init = Ts_init[1]
    Tr3_init = Tr_init[2]
    Tr3_hl_init = Tr_hl_init[0]
    m3_hl_init = m_hl_init[1]

    J_expected = np.array([[1,    -1,    -1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0], #Fm2
                           [0,    1,    0,    -1,    0,    0,    0,   0,    0,    0,    0,    0,    0], #Fm3
                           [df12_dm12,    0,    0,    0,    df12_dp2,    0,    0,    0,    0,    0,    0,    0,    0], #Fdp12
                           [0,    df23_dm23,    0,    0,    df23_dp2,    df23_dp3,    0,    0,    0,    0,    0,    0,    0], #Fdp23
                           [-(psi12*(Ts1-Ta)+Ta)-m12*dpsi12_dm12*(Ts1-Ta) + Ts2_init,     (psi23*(Ts3_init-Ta)+Ta)+m23*dpsi23_dm23*(Ts3_init-Ta) -Ts2_init,    Ts2_hl_init -Ts2_init,    0,     0,    0,    -m23+m12-m2_hl_init,    m23*psi23,    0,    0,    0,    m2_hl_init,    0], #FTs2
                           [0,    -Ts3_init,    0,    Ts3_init,   0,    0,    0,    -m23+m3_hl_init,    0,    0,    0,    0,    0], #FTs3
                           [-(psi12*(Tr2_init-Ta)+Ta)-m12*dpsi12_dm12*(Tr2_init-Ta)-dm1_dm12*Tr1_init,    0,    0,    0,    0,    0,    0,    0,    -m1_hl,    -m12*psi12,    0,    0,    0], #FTr1
                           [Tr2_init,    -Tr2_init,    -Tr2_init,    0,    0,    0,    0,    0,    0,    -m23+m12-m2_hl_init,    0,    0,    0], #FTr2
                           [0,    (psi23*(Tr2_init-Ta)+Ta)+m23*dpsi23_dm23*(Tr2_init-Ta) - Tr3_init,    0,    -Tr3_hl_init + Tr3_init,    0,    0,    0,    0,    0,    m23*psi23,    -m23+m3_hl_init,    0,    -m3_hl_init], #FTr3
                           [0,    0,    Cp*(Ts2_hl_init-Tr2_init),    0,    0,    0,    0,    0,    0,   -Cp*m2_hl_init,    0,    Cp*m2_hl_init,    0], #Fdphi2
                           [0,    0,    0,    Cp*(Ts3_init-Tr3_hl_init),    0,    0,    0,    Cp*m3_hl_init,    0,    0,    0,    0,    -Cp*m3_hl_init], #Fdphi3
                           [0,    0,    0,    0,    0,    0,    0,    0,    0,    -1,    0,    1,    0], #FdT2
                           [0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    -1]]) #FdT3
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(sps.csr_matrix(J_expected)))
    print('J - J expected = \n{}'.format(J-sps.csr_matrix(J_expected)))
    assert np.allclose(J.todense(),J_expected)

def test_F_heat_3N_line_sink_slack_delta_loads_opposite_flow():
    """Test the system of equations for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink.  The initial value of the link flow is chosen such that node 1 functions as a source, while it is defined as a sink.
    """
    # Given
    Tr1 = 40.
    p1 = 9.8*bar
    dphi2 = -2.8*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Tr1=Tr1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='sink',source_node='delta',sink_node='delta')
    m_init = np.array([2.1,3.4])
    p_init = np.array([10.1,6.7])*bar
    Ts_init = np.array([103.,97.,88.])
    Tr_init = np.array([38.,52.])
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='standard')

    # When
    F = nlsys.F(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    psi12 = link12.temp_drop_fac(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    psi23 = link23.temp_drop_fac(m23)
    m1_hl = -m12
    Ts1_init = Ts_init[0]
    Ts2_init = Ts_init[1]
    Tr2_init = Tr_init[0]
    Ts2_hl = dT2+Tr2_init
    m2_hl = dphi2/(Cp*dT2)
    Ts3_init = Ts_init[2]
    Tr3_init = Tr_init[1]
    Tr3_hl = Ts3_init - dT3
    m3_hl = dphi3/(Cp*dT3)

    F_expected = np.array([m12-m23-m2_hl, #Fm2
                           m23 - m3_hl, #Fm3
                           link12.link_equation(),#Fdp12
                           link23.link_equation(), #Fdp23
                           m12*Ts1_init+m1_hl*Ts1_init, #FTs1 (This one should be zero)
                           m23*Ts2_init-m12*(psi12*(Ts1_init-Ta)+Ta)+m2_hl*Ts2_hl, #FTs2
                           -m23*(psi23*(Ts2_init-Ta)+Ta)+m3_hl*Ts3_init, #FTs3
                           -m23*(psi23*(Tr3_init-Ta)+Ta) + m12*Tr2_init -m2_hl*Tr2_init, #FTr2
                           m23*Tr3_init - m3_hl*Tr3_hl]) #FTr3
    print('F = \n{}'.format(F))
    print('F expected = \n{}'.format(F_expected))
    print('F - F exp = \n{}'.format(F-F_expected))
    assert np.allclose(F,F_expected)

def test_J_heat_3N_line_sink_slack_delta_loads_opposite_flow():
    """Test the Jacobian for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink. The initial value of the link flow is chosen such that node 1 functions as a source, while it is defined as a sink.
    """
    # Given
    Tr1 = 40.
    p1 = 9.8*bar
    dphi2 = -2.8*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Tr1=Tr1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='sink',source_node='delta',sink_node='delta')
    m_init = np.array([2.1,3.4])
    p_init = np.array([10.1,6.7])*bar
    Ts_init = np.array([103.,97.,88.])
    Tr_init = np.array([38.,52.])
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='standard')

    # When
    J = nlsys.J(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    df12_dp1,df12_dp2 = link12.f_der_dp_func()*np.array(link12.pres_drop_func_der_p())
    df12_dm12 = link12.f_der_m()
    psi12 = link12.temp_drop_fac(m12)
    dpsi12_dm12 = link12.temp_drop_fac_dm(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    df23_dp2,df23_dp3 = link23.f_der_dp_func()*np.array(link23.pres_drop_func_der_p())
    df23_dm23 = link23.f_der_m()
    psi23 = link23.temp_drop_fac(m23)
    dpsi23_dm23 = link23.temp_drop_fac_dm(m23)
    m1_hl = -m12
    dm1_dm12 = -1
    Ts1_init = Ts_init[0]
    Ts2_init = Ts_init[1]
    Tr2_init = Tr_init[0]
    Ts2_hl = dT2+Tr2_init
    m2_hl = dphi2/(Cp*dT2)
    dm2_dTr2 = 0
    dTs2hl_dTr2 = 1
    Ts3_init = Ts_init[2]
    Tr3_init = Tr_init[1]
    Tr3_hl = Ts3_init - dT3
    m3_hl = dphi3/(Cp*dT3)
    dm3_dTs3 = 0
    dTr3hl_dTs3 = 1

    J_expected = np.array([[1,    -1,    0,    0,    0,    0,    0,    -dm2_dTr2,    0],
                           [0,    1,    0,    0,    0,    -dm3_dTs3,    0,    0,    0],
                           [df12_dm12,    0,    df12_dp2,    0,    0,    0,    0,    0,    0],
                           [0,    df23_dm23,    df23_dp2,    df23_dp3,    0,    0,    0,    0,    0],
                           [Ts1_init+dm1_dm12*Ts1_init,     0,     0,    0,    m12+m1_hl,    0,    0,    0,    0],    #FTs1 (should be a zero row)
                           [-(psi12*(Ts1_init-Ta)+Ta)-m12*dpsi12_dm12*(Ts1_init-Ta),    Ts2_init,    0,    0,    -m12*psi12,    m23,    0,    dm2_dTr2*Ts2_hl+m2_hl*dTs2hl_dTr2,    0],    #FTs2
                           [0,    -(psi23*(Ts2_init-Ta)+Ta)-m23*dpsi23_dm23*(Ts2_init-Ta),    0,    0,    0,    -m23*psi23,    dm3_dTs3*Ts3_init+m3_hl,    0,    0],    #FTs3
                           [Tr2_init,    -(psi23*(Tr3_init-Ta)+Ta)-m23*dpsi23_dm23*(Tr3_init-Ta),    0,    0,    0,    0,    0,    m12-dm2_dTr2*Tr2_init-m2_hl,    -m23*psi23],    #FTr2
                           [0,    Tr3_init,    0,    0,    0,   0,     -dm3_dTs3*Tr3_hl-m3_hl*dTr3hl_dTs3,    0,    m23]]) #FTr3
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(sps.csr_matrix(J_expected)))
    print('J - J expected = \n{}'.format(J-sps.csr_matrix(J_expected)))
    # 4 is the row index of FTs for node 1. The corresponding column indices are stored in indices[indptr[4]:indptr[4+1]]. So if that one is empty, then the row is a zero-row
    print('number of entries in row 4 = {}'.format(len(J.indices[J.indptr[4]:J.indptr[4+1]])))
    if len(J.indices[J.indptr[4]:J.indptr[4+1]])==0:
        zero_row = True
    else:
        print('entries in row 4 = \n{}'.format(J.data[J.indptr[4]:J.indptr[4+1]]))
        if np.all(J.data[J.indptr[4]:J.indptr[4+1]]==0):
            zero_row = True
        else:
            zero_row = False
    assert np.allclose(J.todense(),J_expected) and zero_row

def test_F_heat_unknown_half_link_3N_line_sink_slack_delta_loads_opposite_flow():
    """Test the system of equations for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink. The initial value of the link flow is chosen such that node 1 functions as a source, while it is defined as a sink.
    """
    # Given
    Tr1 = 40.
    p1 = 9.8*bar
    dphi2 = -2.8*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Tr1=Tr1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='sink',source_node='delta',sink_node='delta')
    m_init = np.array([2.1,3.4])
    m_hl_init = np.array([-.8,2.7])
    p_init = np.array([10.1,6.7])*bar
    Ts_init = np.array([103.,97.,88.])
    Tr_init = np.array([38.,52.])
    Ts_hl_init = np.array([102])
    Tr_hl_init = np.array([49.])
    x_init  = np.concatenate((m_init,m_hl_init,p_init ,Ts_init ,Tr_init,Ts_hl_init,Tr_hl_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    F = nlsys.F(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    psi12 = link12.temp_drop_fac(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    psi23 = link23.temp_drop_fac(m23)
    m1_hl = -m12
    Ts1_init = Ts_init[0]
    Ts2_init = Ts_init[1]
    Tr2_init = Tr_init[0]
    Ts2_hl_init = Ts_hl_init[0]
    m2_hl_init = m_hl_init[0]
    Ts3_init = Ts_init[2]
    Tr3_init = Tr_init[1]
    Tr3_hl_init = Tr_hl_init[0]
    m3_hl_init = m_hl_init[1]

    F_expected = np.array([m12-m23-m2_hl_init, #Fm2
                           m23 - m3_hl_init, #Fm3
                           link12.link_equation(),#Fdp12
                           link23.link_equation(), #Fdp23
                           m12*Ts1_init+m1_hl*Ts1_init, #FTs1 (This one should be zero)
                           m23*Ts2_init-m12*(psi12*(Ts1_init-Ta)+Ta)+m2_hl_init*Ts2_hl_init, #FTs2
                           -m23*(psi23*(Ts2_init-Ta)+Ta)+m3_hl_init*Ts3_init, #FTs3
                           -m23*(psi23*(Tr3_init-Ta)+Ta) + m12*Tr2_init -m2_hl_init*Tr2_init, #FTr2
                           m23*Tr3_init - m3_hl_init*Tr3_hl_init, #FTr3
                           -dphi2 + Cp*m2_hl_init*(Ts2_hl_init-Tr2_init), #Fdphi2
                           -dphi3 + Cp*m3_hl_init*(Ts3_init-Tr3_hl_init), #Fdphi3
                           Ts2_hl_init - Tr2_init - dT2, #FdT2
                           Ts3_init - Tr3_hl_init - dT3]) #FdT3
    print('F = \n{}'.format(F))
    print('F expected = \n{}'.format(F_expected))
    print('F - F exp = \n{}'.format(F-F_expected))
    assert np.allclose(F,F_expected)

def test_J_heat_unknown_half_link_3N_line_sink_slack_delta_loads_opposite_flow():
    """Test the Jacobian for a heat network of three nodes (1-3) in a line. Node 1 is taken as ref. sink slack, node 2 is a source, and node 3 is taken as a sink. The initial value of the link flow is chosen such that node 1 functions as a source, while it is defined as a sink.
    """
    # Given
    Tr1 = 40.
    p1 = 9.8*bar
    dphi2 = -2.8*MW
    dT2 = 48
    dphi3 = 1.3*MW
    dT3 = 51
    heat_net = create_test_heat_network_3N_line(Tr1=Tr1,p1=p1,dT2=dT2,dT3=dT3,phi2=dphi2,phi3=dphi3,slack_node='sink',source_node='delta',sink_node='delta')
    m_init = np.array([2.1,3.4])
    m_hl_init = np.array([-.8,2.7])
    p_init = np.array([10.1,6.7])*bar
    Ts_init = np.array([103.,97.,88.])
    Tr_init = np.array([38.,52.])
    Ts_hl_init = np.array([102])
    Tr_hl_init = np.array([49.])
    x_init  = np.concatenate((m_init,m_hl_init,p_init ,Ts_init ,Tr_init,Ts_hl_init,Tr_hl_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    J = nlsys.J(x_init)

    # Then
    link12 = heat_net.links[0]
    m12 = m_init[0]
    df12_dp1,df12_dp2 = link12.f_der_dp_func()*np.array(link12.pres_drop_func_der_p())
    df12_dm12 = link12.f_der_m()
    psi12 = link12.temp_drop_fac(m12)
    dpsi12_dm12 = link12.temp_drop_fac_dm(m12)
    water = link12.link_params.get('carrier')
    Ta = heat_net.get_Ta()
    Cp = water.get_Cp()
    link23 = heat_net.links[1]
    m23 = m_init[1]
    df23_dp2,df23_dp3 = link23.f_der_dp_func()*np.array(link23.pres_drop_func_der_p())
    df23_dm23 = link23.f_der_m()
    psi23 = link23.temp_drop_fac(m23)
    dpsi23_dm23 = link23.temp_drop_fac_dm(m23)
    m1_hl = -m12
    dm1_dm12 = -1
    Ts1_init = Ts_init[0]
    Ts2_init = Ts_init[1]
    Tr2_init = Tr_init[0]
    Ts2_hl_init = Ts_hl_init[0]
    m2_hl_init = m_hl_init[0]
    Ts3_init = Ts_init[2]
    Tr3_init = Tr_init[1]
    Tr3_hl_init = Tr_hl_init[0]
    m3_hl_init = m_hl_init[1]

    J_expected = np.array([[1,    -1,    -1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0], #Fm2
                           [0,    1,    0,    -1,    0,    0,    0,   0,    0,    0,    0,    0,    0], #Fm3
                           [df12_dm12,    0,    0,    0,    df12_dp2,    0,    0,    0,    0,    0,    0,    0,    0], #Fdp12
                           [0,    df23_dm23,    0,    0,    df23_dp2,    df23_dp3,    0,    0,    0,    0,    0,    0,    0], #Fdp23
                           [Ts1_init+dm1_dm12*Ts1_init,     0,    0,    0,     0,    0,    m12+m1_hl,   0,    0,    0,    0,    0,    0], #FTs1 (should be a zero row)
                           [-(psi12*(Ts1_init-Ta)+Ta)-m12*dpsi12_dm12*(Ts1_init-Ta),    Ts2_init,    Ts2_hl_init,    0,   0,    0,    -m12*psi12,    m23,    0,    0,    0,    m2_hl_init,    0], #FTs2
                           [0,    -(psi23*(Ts2_init-Ta)+Ta)-m23*dpsi23_dm23*(Ts2_init-Ta),    0,    Ts3_init,    0,    0,    0,    -m23*psi23,    m3_hl_init,    0,    0,    0,    0], #FTs3
                           [Tr2_init,    -(psi23*(Tr3_init-Ta)+Ta)-m23*dpsi23_dm23*(Tr3_init-Ta),    -Tr2_init,    0,    0,    0,    0,    0,    0,    m12-m2_hl_init,    -m23*psi23,    0,    0], #FTr2
                           [0,    Tr3_init,    0,    -Tr3_hl_init,    0,    0,    0,    0,    0,    0,    m23,    0,    -m3_hl_init], #FTr3
                           [0,    0,    Cp*(Ts2_hl_init-Tr2_init),    0,    0,    0,    0,    0,    0,   -Cp*m2_hl_init,    0,    Cp*m2_hl_init,    0], #Fdphi2
                           [0,    0,    0,    Cp*(Ts3_init-Tr3_hl_init),    0,    0,    0,    0,    Cp*m3_hl_init,    0,    0,    0,    -Cp*m3_hl_init], #Fdphi3
                           [0,    0,    0,    0,    0,    0,    0,    0,    0,    -1,    0,    1,    0], #FdT2
                           [0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    -1]]) #FdT3
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(sps.csr_matrix(J_expected)))
    print('J - J expected = \n{}'.format(J-sps.csr_matrix(J_expected)))
    # 4 is the row index of FTs for node 1. The corresponding column indices are stored in indices[indptr[4]:indptr[4+1]]. So if that one is empty, then the row is a zero-row
    print('number of entries in row 4 = {}'.format(len(J.indices[J.indptr[4]:J.indptr[4+1]])))
    if len(J.indices[J.indptr[4]:J.indptr[4+1]])==0:
        zero_row = True
    else:
        print('entries in row 4 = \n{}'.format(J.data[J.indptr[4]:J.indptr[4+1]]))
        if np.all(J.data[J.indptr[4]:J.indptr[4+1]]==0):
            zero_row = True
        else:
            zero_row = False
    assert np.allclose(J.todense(),J_expected) and zero_row

def create_test_heat_network_extra_links(dummy_links=True):
    """Create a test heat network with extra links to model connections to coupling"""
    # water carrier
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    grav_const = 9.81 #[m/s^2]
    water = Water('water',Cp_w,rho=rho_w)
    # physical parameters of network and pipes
    Ta = 0.
    L_h = 500. #[m]
    D_h = .15 #[m]
    lam = .2 #[W/(mK)]
    U = lam/(np.pi*D_h) #[W/(m^2 K)]
    heat_link_type = 'standard_pipe_low_pres_pole'
    heat_link_params = {'L':L_h,'U':U,'D':D_h,'carrier':water}
    heat_link_type_extra = 'isolated_pipe_low_pres_pole'
    heat_link_params_extra = {'L':L_h,'D':D_h,'carrier':water}
    # boundary conditions
    ph0 = 100*rho_w*grav_const #[Pa]
    Ts0 = 100 #[C]
    phi0_sink = 2*MW
    To0_sink = 50 #[C]
    phi1_sink = 2.5*MW
    To1_sink = 50 #[C]
    phi1_source = 1.5*MW
    To1_source = 99

    h0 = HeatNode('hn0',node_type=0,p=ph0,Ts=Ts0) # slack
    h0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h1 = HeatNode('hn1',node_type=1,Tr_hl=To1_sink,dphi=phi1_sink) # load node (sink)
    h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hl0 = HeatLink('hl0',h0,h1,link_type = heat_link_type,link_params = heat_link_params)
    heat_net = HeatNetwork('2 nodes',Ta=Ta)
    heat_net.add_link(hl0)

    if dummy_links:
        h0_extra = HeatNode('hn0_extra',node_type=3,Tr_hl=To0_sink,dphi=phi0_sink,p=ph0) # ref. load node (sink) (since only connected with a dummy link, so no way to determine pressure)
        h1_extra = HeatNode('hn1_extra',node_type=3,Ts_hl=To1_source,dphi=-phi1_source,p=ph0) # ref. load node (source) (since only connected with a dummy link, so no way to determine pressure)
        hl1 = HeatLink('hl1',h0,h0_extra,link_type = 'dummy',link_params = {'carrier':water})
        hl2 = HeatLink('hl2',h1_extra,h1,link_type = 'dummy',link_params = {'carrier':water})
    else:
        h0_extra = HeatNode('hn0_extra',node_type=1,Tr_hl=To0_sink,dphi=phi0_sink) # ref. load node (sink)
        h1_extra = HeatNode('hn1_extra',node_type=1,Ts_hl=To1_source,dphi=-phi1_source) # ref. load node (source)
        hl1 = HeatLink('hl1',h0,h0_extra,link_type = heat_link_type_extra,link_params = heat_link_params_extra)
        hl2 = HeatLink('hl2',h1_extra,h1,link_type = heat_link_type_extra,link_params = heat_link_params_extra)
    h0_extra.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h1_extra.half_links[0].set_type('heat_exchanger',{'carrier':water})
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    return heat_net, water

def test_J_heat_dummy_links():
    """Test the Jacobian matrix for a heat network with extra dummy links to model connections to coupling"""
    # Given
    heat_net, water = create_test_heat_network_extra_links(dummy_links=True)
    m_ic = 5
    m_hl_ic = m_ic
    ph_ic = 99*water.rhon*water.g #[Pa]
    Ts_ic = 100
    Tr_ic = 50
    m_init = np.array([m_ic,m_ic,m_ic]) #[kg/s]
    m_hl_init = np.array([m_hl_ic,m_hl_ic,-m_hl_ic]) #[kg/s]
    p_init = np.array([ph_ic])
    Ts_init = np.array([Ts_ic,Ts_ic,Ts_ic]) #[C]
    Tr_init = np.array([Tr_ic,Tr_ic,Tr_ic,Tr_ic]) #[C]
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation='half_link_flow')

    # When
    J = nlsys.J(x_init)

    # Then
    heat_net_extra_pipes, _ = create_test_heat_network_extra_links(dummy_links=False)
    heat_net_extra_pipes.initialize()
    nlsys_extra_pipes = NonLinearSystemHeat(heat_net_extra_pipes,formulation='half_link_flow')
    x_init_extra_pipes = np.concatenate((m_init,m_hl_init,np.array([ph_ic,ph_ic,ph_ic]),Ts_init,Tr_init))
    J_extra_pipes = nlsys_extra_pipes.J(x_init_extra_pipes)
    F_ind_extra_pipes = [4,5] # indices of F related to link equations of the extra pipes. Not present in the network with dummy links
    x_ind_extra_pipes = [7,8] # indices of x related to pressures of extra nodes. Not present in the network with dummy links
    F_ind = [ind for ind in range(J_extra_pipes.shape[0]) if not ind in F_ind_extra_pipes]
    x_ind = [ind for ind in range(J_extra_pipes.shape[1]) if not ind in x_ind_extra_pipes]
    J_expected = J_extra_pipes[F_ind,:][:,x_ind]
    assert np.allclose(J.todense(),J_expected.todense())


# =========================================================
# Heterogeneous network
def create_test_heterogeneous_network():
    """Create a test heterogeneous network with a single CHP

    Returns
    -------
    het_net : HeterogeneousNetwork
        The test network
    gas_net : GasNetwork
        The test gas subnetwork
    elec_net : ElectricalNetwork
        The test electrical subnetwork
    heat_net : HeatNetwork
        The test heat subnetwork
    water : Carrier
        The carrier of water in the network
    gas : Carrier
        The carrier of gas in the network
    """
    # water carrier
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp_w,rho=rho_w)

    # carrier
    mu_g = 0.288e-6 #[m^2/s]
    S = 0.6106
    Z = 0.8
    pn = 1.01325*bar #[Pa]
    Tn = 273.15 #[K]
    T = 281.15 #[K]
    R = 8.314413 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('high pres gas',S,R_air,Z,pn,Tn,T,mu=mu_g)
    rhon_g = gas.rhon

    # nodes
    g0 = GasNode('gn0',node_type=0,p=37.*bar) # reference node
    g1 = GasNode('gn1',node_type=3,p=30.*bar,q=20.*rhon_g/hour) # reference load node
    g2 = GasNode('gn2',node_type=1,q=61.86112692100001*rhon_g/hour) # load node
    e0 = ElectricalNode('en0',node_type=0,V=1.,delta=0.) # reference node
    e1 = ElectricalNode('en1',node_type=1,P=1.49307157,V=0.98) # generator node
    e2 = ElectricalNode('en2',node_type=2,P=0.99307157,Q=0.04920407) # load node
    h0 = HeatNode('hn0',node_type=1,Ts_hl=97.079557294922552,dphi=-0.445273104) # source node
    h0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h1 = HeatNode('hn1',node_type=3,Tr_hl=50.,dphi=0.20551716490915198*MW,p=2.*rho_w*g) # load reference node
    h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h2 = HeatNode('hn2',node_type=1,Tr_hl=50.,dphi=0.65884601901145079*MW) # load node
    h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    To_c = 103.265154207 #[C]
    Pb = 1*MW
    eff_CHP = np.array([0.9/Pb, 0.9])
    GHV = 50.2*1e6 #[J/kg]
    c0 = HeterogeneousNode('cn0',node_type=1,unit_type='geh_CHP',unit_params={'eta':eff_CHP,'GHV':GHV}) # To known

    # links
    alpha = 10.
    gas_link_params = {'alpha':alpha}
    gl0 = GasLink('gl0',g0,g1,link_type = 'pipe_linear',link_params = gas_link_params)
    gl1 = GasLink('gl1',g0,g2,link_type = 'pipe_linear',link_params = gas_link_params)
    gl2 = GasLink('gl2',g1,g2,link_type = 'pipe_linear',link_params = gas_link_params)
    gl3 = GasLink('gl3',g2,c0)
    b = -10.
    g = 1.
    elec_link_params = {'b':b,'g':g}
    el0 = ElectricalLink('el0',e0,e1,link_type='short_line',link_params=elec_link_params)
    el1 = ElectricalLink('el1',e0,e2,link_type='short_line',link_params=elec_link_params)
    el2 = ElectricalLink('el2',e1,e2,link_type='short_line',link_params=elec_link_params)
    el3 = ElectricalLink('el3',c0,e1)
    C = 1.
    L = 400. #[m]
    D = 0.01 #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D) #[W/(m^2 K)]
    heat_link_params = {'L':L,'D':D,'U':U,'C':C,'carrier':water}
    hl0 = HeatLink('hl0',h0,h1,link_type='standard_resistor',link_params=heat_link_params)
    hl1 = HeatLink('hl1',h0,h2,link_type='standard_resistor',link_params=heat_link_params.copy())
    hl2 = HeatLink('hl2',h1,h2,link_type='standard_resistor',link_params=heat_link_params.copy())
    hl3 = HeatLink('hl3',c0,h0,link_params={'carrier':water},bc_type=6,Tsstart=To_c) # To of coupling (source) is known

    # network
    gas_net = GasNetwork('test gas network')
    gas_net.add_link(gl0)
    gas_net.add_link(gl1)
    gas_net.add_link(gl2)
    gas_net.add_link(gl3)

    elec_net = ElectricalNetwork('test electrical network')
    elec_net.add_link(el0)
    elec_net.add_link(el1)
    elec_net.add_link(el2)
    elec_net.add_link(el3)

    Ta = 10 #[C]
    heat_net = HeatNetwork('test heat network',Ta=Ta)
    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    heat_net.add_link(hl3)

    het_net = HeterogeneousNetwork('test heterogeneous network')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    het_net.add_node(c0)

    return het_net, gas_net, elec_net, heat_net, water, gas

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_F_heterogeneous():
    """Test the system of equations for a heterogeneous network with only one CHP
    """
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_heterogeneous_network()
    rho_w = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    rhon_g = gas.rhon #[kg/m^3]
    Ta = heat_net.Ta
    gp_init = np.array([30*bar])
    delta_init = np.array([0.,0.1])
    V_init = np.array([2.])
    m_init = np.array([3.,2.,-1.])
    hp_init = np.array([4.,2.])*rho_w*g
    Ts_init = np.array([100.,95.,90.])#-Ta
    Tr_init = np.array([35.,37.,50.])#-Ta
    qc_init = np.array([100.*0.7/hour])
    Sc_init = np.array([2.,0.])
    mc_init = np.array([3.])
    phic_init = np.array([1.])*MW

    mbase = 1.
    pbase = 1.
    Tbase = 1.
    phibase = MW #since P and Q are in p.u., so for the coupling, the heat (and gas) power also need to be scaled to p.u.
    Sbase = 1. #since P and Q are in p.u.
    Vbase = 1.
    deltabase = 1.
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase,'Vbase':Vbase,'deltabase':deltabase,'Sbase':Sbase,'Ebase':phibase}

    xg_init = gp_init/pbase
    xe_init = np.concatenate((delta_init/deltabase,V_init/Vbase))
    xh_init = np.concatenate((m_init/mbase,hp_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    x_init = np.concatenate((xg_init,xe_init,xh_init,qc_init/mbase,Sc_init/Sbase,mc_init/mbase,phic_init/phibase)) # scaled
    het_net.initialize()
    nlsys = NonLinearSystemHeterogeneous(het_net,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysg = NonLinearSystemGas(gas_net,formulation='nodal',scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysh = NonLinearSystemHeat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    F = nlsys.F(x_init)

    # Then
    Fg = nlsysg.F(xg_init)
    Fe = nlsyse.F(xe_init)
    Fh = nlsysh.F(xh_init)
    Fc_expected = np.array([-2.357222, -0.14354537531897804])#np.array([2.1215, -0.14354537531897804])
    F_expected = np.concatenate((Fg,Fe,Fh,Fc_expected))

    assert np.allclose(F,F_expected)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def test_J_heterogenous():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant
    """
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_heterogeneous_network()
    rho_w = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    rhon_g = gas.rhon #[kg/m^3]
    Ta = heat_net.Ta
    gp_init = np.array([30*bar])
    delta_init = np.array([0.,0.1])
    V_init = np.array([2.])
    m_init = np.array([3.,2.,-1.])
    hp_init = np.array([4.,2.])*rho_w*g
    Ts_init = np.array([100.,95.,90.])
    Tr_init = np.array([35.,37.,50.])
    qc_init = np.array([100.*0.7/hour])
    Sc_init = np.array([2.,0.])
    mc_init = np.array([3.])
    phic_init = np.array([1.])*MW

    mbase = 10.
    pbase = 1.
    Tbase = 1.
    phibase = MW #since P and Q are in p.u., so for the coupling, the heat (and gas) power also need to be scaled to p.u.
    Sbase = 1. #since P and Q are in p.u.
    Vbase = 1.
    deltabase = 1.
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase,'Vbase':Vbase,'deltabase':deltabase,'Sbase':Sbase,'Ebase':phibase}
    Egbase = scale_var_params['Ebase']

    xg_init = gp_init/pbase
    xe_init = np.concatenate((delta_init/deltabase,V_init/Vbase))
    xh_init = np.concatenate((m_init/mbase,hp_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    x_init = np.concatenate((xg_init,xe_init,xh_init,qc_init/mbase,Sc_init/Sbase,mc_init/mbase,phic_init/phibase)) # scaled
    het_net.initialize()
    nlsys = NonLinearSystemHeterogeneous(het_net,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysg = NonLinearSystemGas(gas_net,formulation='nodal',scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysh = NonLinearSystemHeat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    J = nlsys.J_dense(x_init)

    # Then
    Jgg = nlsysg.J(xg_init)
    Jee = nlsyse.J(xe_init)
    Jhh = nlsysh.J(xh_init)
    c0 = het_net.nodes[-1]
    eta = c0.unit_params['eta']
    GHV = c0.unit_params['GHV']
    hl_coupling = heat_net.links[-1]
    Cp = hl_coupling.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
    To_c = hl_coupling.get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params)
    Jcc_expected = np.array([[GHV*mbase/Egbase,-1/(eta[0]/(Sbase/phibase)), 0, 0 , -1/eta[1]],
    [0, 0, 0, Cp*(To_c-Tr_init[0])/Tbase, -1]])
    Jgc_expected = sps.csr_matrix(([-1],([1],[0])),shape=(2,5))
    Jec_expected = sps.csr_matrix(([-1],([0],[1])),shape=(3,5))
    Jhc_expected = sps.csr_matrix(([1,-To_c/Tbase,Tr_init[0]/Tbase],([0,6,9],[3,3,3])),shape=(12,5))
    Jch_expected = sps.csr_matrix(([-Cp*mc_init[0]/mbase],([1],[8])),shape=(2,11))
    J_expected = sps.bmat([[Jgg,None,None,Jgc_expected],[None,Jee,None,Jec_expected],[None,None,Jhh,Jhc_expected],[None,None,Jch_expected,Jcc_expected]],format='csr').todense()

    assert np.allclose(J,J_expected)

def create_test_heterogeneous_network_EH():
    """Create a test heterogeneous network with an energy hub

    Returns
    -------
    het_net : HeterogeneousNetwork
        The test network
    gas_net : GasNetwork
        The test gas subnetwork
    elec_net : ElectricalNetwork
        The test electrical subnetwork
    heat_net : HeatNetwork
        The test heat subnetwork
    water : Carrier
        The carrier of water in the network
    gas : Carrier
        The carrier of gas in the network
    """
    # water carrier
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp_w,rho=rho_w)

    # carrier
    mu_g = 0.288e-6 #[m^2/s]
    S = 0.6106
    Z = 0.8
    pn = 1.01325*bar #[Pa]
    Tn = 273.15 #[K]
    T = 281.15 #[K]
    R = 8.314413 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('high pres gas',S,R_air,Z,pn,Tn,T,mu=mu_g)
    rhon_g = gas.rhon

    # nodes
    g0 = GasNode('gn0',node_type=0,p=37.*bar) # reference node
    g1 = GasNode('gn1',node_type=1,q=20.*rhon_g/hour) # load node
    g2 = GasNode('gn2',node_type=1,q=61.86112692100001*rhon_g/hour) # load node
    e0 = ElectricalNode('en0',node_type=0,V=1.,delta=0.) # reference node
    e1 = ElectricalNode('en1',node_type=1,P=1.49307157,V=0.98) # generator node
    e2 = ElectricalNode('en2',node_type=2,P=0.99307157,Q=0.04920407) # load node
    h0 = HeatNode('hn0',node_type=4,Ts_hl=97.079557294922552,Ts=100.,dphi=-0.445273104) # source temperature node
    h0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h1 = HeatNode('hn1',node_type=3,Tr_hl=50.,dphi=0.20551716490915198*MW,p=2.*rho_w*g) # load reference node
    h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h2 = HeatNode('hn2',node_type=1,Tr_hl=50.,dphi=0.65884601901145079*MW) # load node
    h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    eta = 0.9
    nu = 0.5/0.95 # dispatch factor (to active power)
    GHV = 50.2*1e6 #[J/kg]
    Pb = MW # power is already in p.u.
    C = np.array([[0,0,0],[eta*nu/Pb,0,0],[eta*(1-nu),0,0]])
    Ein_carriers = ['g','e','h']
    Eout_carriers = ['g','e','h']
    c0 = HeterogeneousNode('cn0',node_type=0,unit_type='EH',unit_params={'C':C,'GHV':GHV,'Eout_carriers':Eout_carriers,'Ein_carriers':Ein_carriers},Ein_len=np.array([1,1,1]),Eout_len=np.array([1,1,1])) # To unknown

    # links
    alpha = 10.
    gas_link_params = {'alpha':alpha}
    gl0 = GasLink('gl0',g0,g1,link_type = 'pipe_linear',link_params = gas_link_params)
    gl1 = GasLink('gl1',g0,g2,link_type = 'pipe_linear',link_params = gas_link_params)
    gl2 = GasLink('gl2',g1,g2,link_type = 'pipe_linear',link_params = gas_link_params)
    gl3 = GasLink('gl3',g2,c0)
    b = -10.
    g = 1.
    elec_link_params = {'b':b,'g':g}
    el0 = ElectricalLink('el0',e0,e1,link_type='short_line',link_params=elec_link_params)
    el1 = ElectricalLink('el1',e0,e2,link_type='short_line',link_params=elec_link_params)
    el2 = ElectricalLink('el2',e1,e2,link_type='short_line',link_params=elec_link_params)
    el3 = ElectricalLink('el3',c0,e1)
    C = 1.
    L = 400. #[m]
    D = 0.01 #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D) #[W/(m^2 K)]
    heat_link_params = {'L':L,'D':D,'U':U,'C':C,'carrier':water}
    hl0 = HeatLink('hl0',h0,h1,link_type='standard_resistor',link_params=heat_link_params.copy())
    hl1 = HeatLink('hl1',h0,h2,link_type='standard_resistor',link_params=heat_link_params.copy())
    hl2 = HeatLink('hl2',h1,h2,link_type='standard_resistor',link_params=heat_link_params.copy())
    hl3 = HeatLink('hl3',c0,h0,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown

    # network
    gas_net = GasNetwork('test gas network')
    gas_net.add_link(gl0)
    gas_net.add_link(gl1)
    gas_net.add_link(gl2)
    gas_net.add_link(gl3)

    elec_net = ElectricalNetwork('test electrical network')
    elec_net.add_link(el0)
    elec_net.add_link(el1)
    elec_net.add_link(el2)
    elec_net.add_link(el3)

    Ta = 10 #[C]
    heat_net = HeatNetwork('test heat network',Ta=Ta)
    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    heat_net.add_link(hl3)

    het_net = HeterogeneousNetwork('test heterogeneous network')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    het_net.add_node(c0)

    return het_net, gas_net, elec_net, heat_net, water, gas

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_F_heterogeneous_EH():
    """Test the system of equations for a heterogeneous network with one EH
    """
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_heterogeneous_network_EH()
    rho_w = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    rhon_g = gas.rhon #[kg/m^3]
    Ta = heat_net.Ta
    gp_init = np.array([35,30])*bar
    delta_init = np.array([0.,0.1])
    V_init = np.array([2.])
    m_init = np.array([3.,2.,-1.])
    hp_init = np.array([4.,2.])*rho_w*g
    Ts_init = np.array([95.,90.])
    Tr_init = np.array([35.,37.,50.])
    qc_init = np.array([100.*0.7/hour])
    Sc_init = np.array([2.,0.])
    mc_init = np.array([3.])
    phic_init = np.array([1.])*MW
    To_init = np.array([90.])

    mbase = 1.
    pbase = 1.
    Tbase = 1.
    phibase = MW #since P and Q are in p.u., so for the coupling, the heat (and gas) power also need to be scaled to p.u.
    Sbase = 1. #since P and Q are in p.u.
    Vbase = 1.
    deltabase = 1.
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase,'Vbase':Vbase,'deltabase':deltabase,'Sbase':Sbase,'Ebase':phibase}

    xg_init = gp_init/pbase
    xe_init = np.concatenate((delta_init/deltabase,V_init/Vbase))
    xh_init = np.concatenate((m_init/mbase,hp_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    x_init = np.concatenate((xg_init,xe_init,xh_init,qc_init/mbase,Sc_init/Sbase,mc_init/mbase,phic_init/phibase,To_init/Tbase)) # scaled
    het_net.initialize()
    nlsys = NonLinearSystemHeterogeneous(het_net,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysg = NonLinearSystemGas(gas_net,formulation='nodal',scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysh = NonLinearSystemHeat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    F = nlsys.F(x_init)

    # Then
    Fg = nlsysg.F(xg_init)
    Fe = nlsyse.F(xe_init)
    Fh = nlsysh.F(xh_init)
    Fc_expected = np.array([0.,1.5376315789473685, 0.58386842105263148, -0.30996999999999997])
    F_expected = np.concatenate((Fg,Fe,Fh,Fc_expected))

    assert np.allclose(F,F_expected)


@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def test_J_heterogenous_EH():
    """Test the creation of the Jacobian matrix for a heat network with only standard pipes and a given pipe constant, with one EH
    """
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_heterogeneous_network_EH()
    rho_w = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    rhon_g = gas.rhon #[kg/m^3]
    Ta = heat_net.Ta
    gp_init = np.array([35,30])*bar
    delta_init = np.array([0.,0.1])
    V_init = np.array([2.])
    m_init = np.array([3.,2.,-1.])
    hp_init = np.array([4.,2.])*rho_w*g
    Ts_init = np.array([95.,90.])
    Tr_init = np.array([35.,37.,50.])
    qc_init = np.array([100.*0.7/hour])
    Sc_init = np.array([2.,0.])
    mc_init = np.array([3.])
    phic_init = np.array([1.])*MW
    To_init = np.array([90.])#-Ta

    mbase = 1.
    pbase = 1.
    Tbase = 1.
    phibase = MW #since P and Q are in p.u., so for the coupling, the heat (and gas) power also need to be scaled to p.u.
    Sbase = 1. #since P and Q are in p.u.
    Vbase = 1.
    deltabase = 1.
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase,'Vbase':Vbase,'deltabase':deltabase,'Sbase':Sbase,'Ebase':phibase}

    xg_init = gp_init/pbase
    xe_init = np.concatenate((delta_init/deltabase,V_init/Vbase))
    xh_init = np.concatenate((m_init/mbase,hp_init/pbase,Ts_init/Tbase,Tr_init/Tbase))
    x_init = np.concatenate((xg_init,xe_init,xh_init,qc_init/mbase,Sc_init/Sbase,mc_init/mbase,phic_init/phibase,To_init/Tbase)) # scaled
    het_net.initialize()
    nlsys = NonLinearSystemHeterogeneous(het_net,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysg = NonLinearSystemGas(gas_net,formulation='nodal',scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysh = NonLinearSystemHeat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    J = nlsys.J_dense(x_init)

    # Then
    Jgg = nlsysg.J(xg_init)
    Jee = nlsyse.J(xe_init)
    Jhh = nlsysh.J(xh_init)

    c0 = het_net.nodes[-1]
    C = c0.unit_params['C']
    GHV = c0.unit_params['GHV']
    hl_coupling = heat_net.links[-1]
    Cp = hl_coupling.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
    To_c = hl_coupling.get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params)
    Jcc_expected = np.array([[0, 0, 0, 0, 0, 0],
                             [-C[1,0]*GHV/Sbase, 1, 0, 0, 0, 0],
                             [-C[2,0]*GHV/phibase, 0, 0, 0, 1, 0],
                             [0, 0, 0, Cp*(To_c-Tr_init[0])/Tbase, -1, Cp*mc_init[0]/mbase]])
    Jgc_expected = sps.csr_matrix(([-1],([1],[0])),shape=(2,6))
    Jec_expected = sps.csr_matrix(([-1],([0],[1])),shape=(3,6))
    Jhc_expected = sps.csr_matrix(([1,-To_c/Tbase,Tr_init[0]/Tbase,-mc_init[0]/mbase],([0,6,9,6],[3,3,3,5])),shape=(12,6))
    Jch_expected = sps.csr_matrix(([-Cp*mc_init[0]/mbase],([3],[7])),shape=(4,10))
    J_expected = sps.bmat([[Jgg,None,None,Jgc_expected],[None,Jee,None,Jec_expected],[None,None,Jhh,Jhc_expected],[None,None,Jch_expected,Jcc_expected]],format='csr').todense()

    assert np.allclose(J,J_expected)

def create_test_network_only_coupling_ge():
    """Create a heterogeneous network, consisting of only a gas-fired generator. The active and reactive power are assumed known, the gas flow is assumed unknown.
    """
    eta_GG = .6
    GHV = 60134305#[J/kg]
    c0 = HeterogeneousNode('cn0',node_type=0,unit_type='ge_gas_fired_gen',unit_params={'eta':eta_GG,'GHV':GHV})
    hlg = GasHalfLink('cn0_hlg',c0,-0.05,bc_type=0) # gas flows into coupling node, q is unknown
    hle = ElectricalHalfLink('cn0_hle',c0,P=2*MW,Q=2*MW,bc_type=1) # P and Q are known
    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)

    coupling_net.initialize()
    return coupling_net

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_F_coupling_ge():
    """Test the system of equations for the coupling network, consisting of only a gas-fired generator (i.e., there are only halflinks connected to the coupling node). The active and reactive power are assumed known, the gas flow is assumed unknown. """
    # Given
    coupling_net = create_test_network_only_coupling_ge()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[0].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[1].P = Pc_hl
    cn.half_links[1].Q = Qc_hl
    xc = coupling_net.set_x_init()
    nlsys = NonLinearSystemHeterogeneous(coupling_net)

    # When
    F = nlsys.F(xc)

    # Then
    F_expected = np.array([Pc_hl - cn.unit_params.get('eta')*cn.unit_params.get('GHV')*-qc_hl_init])
    assert (len(F)>0 and np.allclose(F,F_expected))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_J_coupling_ge():
    """Test the Jacobian for the coupling network, consisting of only a gas-fired generator (i.e., there are only halflinks connected to the coupling node). The active and reactive power are assumed known, the gas flow is assumed unknown. """
    # Given
    coupling_net = create_test_network_only_coupling_ge()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[0].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[1].P = Pc_hl
    cn.half_links[1].Q = Qc_hl
    xc = coupling_net.set_x_init()
    nlsys = NonLinearSystemHeterogeneous(coupling_net)

    # When
    J = nlsys.J_dense(xc)

    # Then
    J_expected = np.array([-cn.unit_params.get('eta')*cn.unit_params.get('GHV')])
    assert (J.size>0 and np.allclose(J,J_expected))

def create_test_network_only_coupling_gh():
    """Create a heterogeneous network, consisting of only a gas boiler. The heat power and outflow temperature are assumed known, the gas flow is assumed unknown.
    """
    eta_GB = .7
    GHV = 60134305#[J/kg]
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    grav_const = 9.81 #[m/s^2]
    water = Water('water',Cp_w,rho=rho_w)
    c0 = HeterogeneousNode('cn0',node_type=1,unit_type='gh_gas_boiler',unit_params={'eta':eta_GB,'GHV':GHV}) # To known
    hlh = HeatHalfLink('cn0_hlh',c0,Ts=100,dphi=-1*MW,link_type='heat_exchanger',bc_type=2,link_params={'carrier':water}) # Ts and dphi known, m unknown (and Tr)
    hlg = GasHalfLink('cn0_hlg',c0,-0.05,bc_type=0) # gas flows into coupling node, q is unknown
    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)

    coupling_net.initialize()
    return coupling_net

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_F_coupling_gh():
    """Test the system of equations for the coupling network, consisting of only a gas boiler (i.e., there are only halflinks connected to the coupling node). The heat power and outflow temperature are assumed known, the gas flow is assumed unknown. """
    # Given
    coupling_net = create_test_network_only_coupling_gh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[1].q = qc_hl_init
    mc_hl_init = -1.01
    phic_hl = -2*MW
    Tsc_hl = 100.
    Trc_hl = 49.
    hlh = cn.half_links[0]
    hlh.m = mc_hl_init
    hlh.dphi = phic_hl
    hlh.Ts = Tsc_hl
    hlh.Tr = Trc_hl
    xc = coupling_net.set_x_init()
    nlsys = NonLinearSystemHeterogeneous(coupling_net)

    # When
    F = nlsys.F(xc)

    # Then
    F_expected = np.array([-phic_hl - cn.unit_params.get('eta')*cn.unit_params.get('GHV')*-qc_hl_init, phic_hl + hlh.link_params.get('carrier').get_Cp()*-mc_hl_init*(Tsc_hl-Trc_hl)])
    print('F = {}'.format(F))
    print('F expected = {}'.format(F_expected))
    assert (len(F)>0 and np.allclose(F,F_expected))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_J_coupling_gh():
    """Test the Jacobian for the coupling network, consisting of only a gas boiler (i.e., there are only halflinks connected to the coupling node). The heat power and outflow temperature are assumed known, the gas flow is assumed unknown. """
    # Given
    coupling_net = create_test_network_only_coupling_gh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[1].q = qc_hl_init
    mc_hl_init = -1.01
    phic_hl = -2*MW
    Tsc_hl = 100.
    Trc_hl = 49.
    hlh = cn.half_links[0]
    hlh.m = mc_hl_init
    hlh.dphi = phic_hl
    hlh.Ts = Tsc_hl
    hlh.Tr = Trc_hl
    xc = coupling_net.set_x_init()
    nlsys = NonLinearSystemHeterogeneous(coupling_net)

    # When
    J = nlsys.J_dense(xc)

    # Then
    J_expected = np.array([[-cn.unit_params.get('eta')*cn.unit_params.get('GHV'),0],[0,hlh.link_params.get('carrier').get_Cp()*(Tsc_hl-Trc_hl)]])
    assert (J.size>0 and np.allclose(J,J_expected))

def create_test_network_only_coupling_geh():
    """Create a heterogeneous network, consisting of only a CHP. The active and reactive power, heat power, and outflow temperature are assumed known, the gas flow is assumed unknown.
    """
    eta_CHP = np.array([.6,.7])
    GHV = 60134305#[J/kg]
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    grav_const = 9.81 #[m/s^2]
    water = Water('water',Cp_w,rho=rho_w)
    c0 = HeterogeneousNode('cn0',node_type=1,unit_type='geh_CHP',unit_params={'eta':eta_CHP,'GHV':GHV}) # To known
    hlh = HeatHalfLink('cn0_hlh',c0,Ts=100,dphi=-1*MW,link_type='heat_exchanger',bc_type=2,link_params={'carrier':water}) # Ts and dphi known, m unknown (and Tr)
    hlg = GasHalfLink('cn0_hlg',c0,-0.05,bc_type=0) # gas flows into coupling node, q is unknown
    hle = ElectricalHalfLink('cn0_hle',c0,P=2*MW,Q=2*MW,bc_type=1) # P and Q are known
    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)

    coupling_net.initialize()
    return coupling_net

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_F_coupling_geh():
    """Test the system of equations for the coupling network, consisting of only a CHP (i.e., there are only halflinks connected to the coupling node). The active and reactive power, heat power, and outflow temperature are assumed known, the gas flow is assumed unknown. """
    # Given
    coupling_net = create_test_network_only_coupling_geh()
    cn = coupling_net.nodes[0]
    eta_CHP = cn.unit_params.get('eta')
    qc_hl_init = -0.08
    cn.half_links[1].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[2].P = Pc_hl
    cn.half_links[2].Q = Qc_hl
    mc_hl_init = -1.01
    phic_hl = -2*MW
    Tsc_hl = 100.
    Trc_hl = 49.
    hlh = cn.half_links[0]
    hlh.m = mc_hl_init
    hlh.dphi = phic_hl
    hlh.Ts = Tsc_hl
    hlh.Tr = Trc_hl
    xc = coupling_net.set_x_init()
    nlsys = NonLinearSystemHeterogeneous(coupling_net)

    # When
    F = nlsys.F(xc)

    # Then
    F_expected = np.array([cn.unit_params.get('GHV')*-qc_hl_init - Pc_hl/eta_CHP[0] - -phic_hl/eta_CHP[1], phic_hl + hlh.link_params.get('carrier').get_Cp()*-mc_hl_init*(Tsc_hl-Trc_hl)])
    assert (len(F)>0 and np.allclose(F,F_expected))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_J_coupling_geh():
    """Test the Jacobian for the coupling network, consisting of only a CHP (i.e., there are only halflinks connected to the coupling node). The active and reactive power, heat power, and outflow temperature are assumed known, the gas flow is assumed unknown."""
    # Given
    coupling_net = create_test_network_only_coupling_geh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.08
    cn.half_links[1].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[2].P = Pc_hl
    cn.half_links[2].Q = Qc_hl
    mc_hl_init = -1.01
    phic_hl = -2*MW
    Tsc_hl = 100.
    Trc_hl = 49.
    hlh = cn.half_links[0]
    hlh.m = mc_hl_init
    hlh.dphi = phic_hl
    hlh.Ts = Tsc_hl
    hlh.Tr = Trc_hl
    xc = coupling_net.set_x_init()
    nlsys = NonLinearSystemHeterogeneous(coupling_net)

    # When
    J = nlsys.J_dense(xc)

    # Then
    J_expected = np.array([[cn.unit_params.get('GHV'),0],[0,hlh.link_params.get('carrier').get_Cp()*(Tsc_hl-Trc_hl)]])
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(J_expected))
    assert (J.size>0 and np.allclose(J,J_expected))

def create_test_network_only_coupling_geh_EH():
    """Create a heterogeneous network, consisting of only an EH. The reactive power, heat power, and outflow temperature are assumed known, the gas flow, water mass flow, and active power are assumed unknown.
    """
    eta_CHP = np.array([.6,.7])
    nu = 0.4
    eta_GG = .6
    eta_GB = .7
    C = np.array([[eta_GG*nu],[eta_GB*(1-nu)]])
    GHV = 60134305#[J/kg]
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    grav_const = 9.81 #[m/s^2]
    water = Water('water',Cp_w,rho=rho_w)
    c0 = HeterogeneousNode('cn0',node_type=1,unit_type='EH',unit_params={'C':C,'GHV':GHV}) # To known
    hlh = HeatHalfLink('cn0_hlh',c0,Ts=100,dphi=-1*MW,link_type='heat_exchanger',bc_type=2,link_params={'carrier':water}) # Ts and dphi known, m unknown (and Tr)
    hlg = GasHalfLink('cn0_hlg',c0,-0.05,bc_type=0) # gas flows into coupling node, q is unknown
    hle = ElectricalHalfLink('cn0_hle',c0,Q=2*MW,bc_type=3) # P is unknown, Q is known
    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)

    coupling_net.initialize()
    return coupling_net

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_J_coupling_geh_EH():
    """Test the Jacobian for the coupling network, consisting of only an EH (i.e., there are only halflinks connected to the coupling node). The reactive power, heat power, and outflow temperature are assumed known, the gas flow, water mass flow, and active power are assumed unknown."""
    # Given
    coupling_net = create_test_network_only_coupling_geh_EH()
    qc_init = 0.08
    Tsc = 103.
    Trc = 48.
    Pc_init = 1.7*MW
    mc_init = 5.3
    phic = 3.1*MW

    cn = coupling_net.nodes[0]
    cn.half_links[1].q = -qc_init
    cn.half_links[2].P = Pc_init
    hlh = cn.half_links[0]
    hlh.m = mc_init
    hlh.Ts = Tsc # is known
    hlh.Tr = Trc # set to known value (is usually taken from heat network)
    hlh.dphi = -phic # is known. <0, since it is a source
    xc = coupling_net.set_x_init() #[qc,Pc,mc]
    nlsys = NonLinearSystemHeterogeneous(coupling_net)

    # When
    J = nlsys.J_dense(xc)

    # Then
    C = cn.unit_params.get('C')
    GHV = cn.unit_params.get('GHV')
    J_expected = np.array([[-C[0,0]*GHV,1,0],
                           [-C[1,0]*GHV,0,0],
                           [0,0,hlh.link_params.get('carrier').get_Cp()*(Tsc-Trc)]])
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(J_expected))
    assert (J.size>0 and np.allclose(J,J_expected))

def create_test_network_only_coupling_geh_EH_dT():
    """Create a heterogeneous network, consisting of only an EH. The reactive power, heat power, and temperature difference are assumed known. The gas flow, water mass flow, outflow temperature, and active power are assumed unknown.
    """
    eta_CHP = np.array([.6,.7])
    nu = 0.4
    eta_GG = .6
    eta_GB = .7
    C = np.array([[eta_GG*nu],[eta_GB*(1-nu)]])
    GHV = 60134305#[J/kg]
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    grav_const = 9.81 #[m/s^2]
    water = Water('water',Cp_w,rho=rho_w)
    c0 = HeterogeneousNode('cn0',node_type=2,unit_type='EH',unit_params={'C':C,'GHV':GHV}) # dT known
    hlh = HeatHalfLink('cn0_hlh',c0,dT=50,dphi=-1*MW,link_type='heat_exchanger',bc_type=4,link_params={'carrier':water}) # dT and dphi known, m unknown (and Tr)
    hlg = GasHalfLink('cn0_hlg',c0,-0.05,bc_type=0) # gas flows into coupling node, q is unknown
    hle = ElectricalHalfLink('cn0_hle',c0,Q=2*MW,bc_type=3) # P is unknown, Q is known
    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)

    coupling_net.initialize()
    return coupling_net

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_J_coupling_geh_EH():
    """Test the Jacobian for the coupling network, consisting of only an EH (i.e., there are only halflinks connected to the coupling node). The reactive power, heat power, and outflow temperature are assumed known, the gas flow, water mass flow, and active power are assumed unknown."""
    # Given
    coupling_net = create_test_network_only_coupling_geh_EH_dT()
    qc_init = 0.08
    dTc = 51.
    Trc = 48.
    Tsc_init = Trc + dTc + 2 # to make sure the initial guess is wrong
    Pc_init = 1.7*MW
    mc_init = 5.3
    phic = 3.1*MW

    cn = coupling_net.nodes[0]
    cn.half_links[1].q = -qc_init
    cn.half_links[2].P = Pc_init
    hlh = cn.half_links[0]
    hlh.m = -mc_init
    hlh.dT = dTc # is known
    hlh.Ts = Tsc_init
    hlh.Tr = Trc # set to known value (is usually taken from heat network)
    hlh.dphi = -phic # is known. <0, since it is a source
    xc = coupling_net.set_x_init() #[qc,Pc,mc]
    nlsys = NonLinearSystemHeterogeneous(coupling_net)

    # When
    J = nlsys.J_dense(xc)

    # Then
    C = cn.unit_params.get('C')
    GHV = cn.unit_params.get('GHV')
    Cp = hlh.link_params.get('carrier').get_Cp()
    J_expected = np.array([[-C[0,0]*GHV,1,0,0],
                           [-C[1,0]*GHV,0,0,0],
                           [0,0,Cp*(Tsc_init-Trc),Cp*mc_init],
                           [0,0,0,1]])
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(J_expected))
    assert (J.size>0 and np.allclose(J,J_expected))

def create_coupling_eh_single_network_m_known():
    """Create a coupling network consisting of two nodes, for electricity-heat coupling. Assume m known for both heat halflinks"""
    # parameters
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    grav_const = 9.81 #[m/s^2]
    water = Water('water',Cp_w,rho=rho_w)
    GHV = 60134305#[J/kg]
    eta_GG0 = .6
    eta_GG1 = .7
    eta_GB0 = .75
    eta_GB1 = .8
    eta_CHP0 = np.array([eta_GG0,eta_GB0])
    eta_CHP1 = np.array([eta_GG1,eta_GB1])
    unit_type_CHP = 'geh_CHP'
    unit_params_CHP0 = {'eta':eta_CHP0,'GHV':GHV}
    unit_params_CHP1 = {'eta':eta_CHP1,'GHV':GHV}
    # boundary conditions
    P0_sol = 1*MW
    Q0 = .5*MW
    P1 = 3.5*MW
    Q1 = 2*MW
    Ts0 = 101
    Tr0 = 50
    m0 = -1 #[kg/s]
    dphi0 = Cp_w*(Ts0-Tr0) #[W] (Needs to satisfy the heat power equation to have realistic values. Should not be necessary to run load flow for this network
    dphi1_sol = -1.5*MW
    m1 = -1.5 #[kg/s]
    Tr1 = 49 #[C]
    q0 = -(P0_sol/eta_CHP0[0] + -dphi0/eta_CHP0[1])/GHV #[kg/s]
    q1 = -(P1/eta_CHP1[0] + -dphi1_sol/eta_CHP0[1])/GHV #[kg/s]

    c0 = HeterogeneousNode('cn0',node_type=4,unit_type=unit_type_CHP,unit_params=unit_params_CHP0) # To known, no heat power equations
    c0_hl_heat = HeatHalfLink('cn_hlh',c0,link_type='heat_exchanger',link_params={'carrier':water},bc_type=14,Ts=Ts0,dphi=dphi0,m=m0) # m, Ts and dphi known (and Tr known), source
    c0_hl_heat.Tr = Tr0 # Set to solution
    c0_hl_gas = GasHalfLink('cn0_hlg',c0,q0,bc_type=1) # gas flows into coupling node, q known
    c0_hl_elec = ElectricalHalfLink('cn0_hle',c0,Q=Q0,bc_type=3) # P is unknown and Q is known
    c1 = HeterogeneousNode('cn1',node_type=0,unit_type=unit_type_CHP,unit_params=unit_params_CHP1) # To unknown
    c1_hl_heat = HeatHalfLink('cn_hlh',c1,link_type='heat_exchanger',link_params={'carrier':water},bc_type=12,m=m1) # m (and Tr) known, Ts and dphi unknown, source
    c1_hl_heat.Tr = Tr1 # Set to solution
    c1_hl_gas = GasHalfLink('cn1_hlg',c1,q1,bc_type=1) # gas flows into coupling node, q known
    c1_hl_elec = ElectricalHalfLink('cn1_hle',c1,P=P1,Q=Q1,bc_type=1) # P and Q are known

    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)
    coupling_net.add_node(c1)

    return coupling_net

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_F_coupling_eh_single_network_m_known():
    """Test the system of equations for a coupling network consisting of two nodes, for electricity-heat coupling. Assume m known for both heat halflinks"""
    # given
    coupling_net = create_coupling_eh_single_network_m_known()
    P0_init = 1.8*MW
    dphi1_init = -2.1*MW
    Ts1_init = 99
    x_init = np.array([P0_init,-dphi1_init,Ts1_init])
    coupling_net.initialize()
    coupling_net.update(x_init)
    xc = coupling_net.set_x_init()

    # when
    nlsys = NonLinearSystemHeterogeneous(coupling_net)

    # When
    F = nlsys.F(xc)

    # Then
    GHV = coupling_net.nodes[0].unit_params.get('GHV')
    cn0 = coupling_net.nodes[0]
    eta_CHP0 = cn0.unit_params.get('eta')
    cn1 = coupling_net.nodes[1]
    eta_CHP1 = cn1.unit_params.get('eta')
    cn0_hlh = cn0.half_links[0]
    cn0_hlg = cn0.half_links[1]
    cn0_hle = cn0.half_links[2]
    cn1_hlh = cn1.half_links[0]
    cn1_hlg = cn1.half_links[1]
    cn1_hle = cn1.half_links[2]
    F_expected = np.array([GHV*-cn0_hlg.q - P0_init/eta_CHP0[0] - -cn0_hlh.dphi/eta_CHP0[1],
                           GHV*-cn1_hlg.q - cn1_hle.P/eta_CHP1[0] - -dphi1_init/eta_CHP1[1],
                           dphi1_init + cn1_hlh.link_params.get('carrier').get_Cp()*-cn1_hlh.m*(Ts1_init-cn1_hlh.Tr)])
    assert (len(F)>0 and np.allclose(F,F_expected))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_J_coupling_eh_single_network_m_known():
    """Test the Jacobian matrix for a coupling network consisting of two nodes, for electricity-heat coupling. Assume m known for both heat halflinks"""
    # given
    coupling_net = create_coupling_eh_single_network_m_known()
    P0_init = 1.8*MW
    dphi1_init = -2.1*MW
    Ts1_init = 99
    x_init = np.array([P0_init,-dphi1_init,Ts1_init])
    coupling_net.initialize()
    coupling_net.update(x_init)
    xc = coupling_net.set_x_init()

    # when
    nlsys = NonLinearSystemHeterogeneous(coupling_net)

    # When
    J = nlsys.J_dense(xc)

    # Then
    GHV = coupling_net.nodes[0].unit_params.get('GHV')
    cn0 = coupling_net.nodes[0]
    eta_CHP0 = cn0.unit_params.get('eta')
    cn1 = coupling_net.nodes[1]
    eta_CHP1 = cn1.unit_params.get('eta')
    cn1_hlh = cn1.half_links[0]
    J_expected = np.array([[-1/eta_CHP0[0],0,0],
                           [0,-1/eta_CHP1[1], 0],
                           [0,-1,-cn1_hlh.link_params.get('carrier').get_Cp()*cn1_hlh.m]])
    print('J = \n{}'.format(J))
    print('J expected = \n{}'.format(J_expected))
    assert (J.size>0 and np.allclose(J,J_expected))
