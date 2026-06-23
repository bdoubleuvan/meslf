"""Test the heterogeneous network classes and methods"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water, Gas
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.utils.constants import MW, bar, hour, BTU, MBTU
import numpy as np
import pytest


def create_test_network():
    """Create a test heterogeneous network

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
    h0 = HeatNode('hn0',node_type=1,Ts_hl=97.079557294922552,dphi=-0.445273104*MW) # source node
    h0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h1 = HeatNode('hn1',node_type=3,Tr_hl=50.,dphi=0.20551716490915198*MW,p=2.*rho_w*g) # load (sink) reference node
    h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h2 = HeatNode('hn2',node_type=1,Tr_hl=50.,dphi=0.65884601901145079*MW) # load (sink) node
    h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    To_c = 103.265154207 #[C]
    eff_CHP = np.array([0.9, 0.9])
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
def test_x_init():
    """Test the initial guess for the heterogeneous network
    """
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_network()
    rho_w = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    rhon_g = gas.rhon #[kg/m^3]
    Ta = heat_net.Ta

    # When
    gp2 = 30*bar
    q3 = 100.*rhon_g/hour
    m0 = 3
    m1 = 2
    m2 = -1
    m3 = 3.
    delta1 = 0.
    delta2 = 0.1
    V2 = 2.
    P3 = 2.
    hp0 = 4.*rho_w*g
    hp2 = 2.*rho_w*g
    Ts0 = 100.
    Ts1 = 95.
    Ts2 = 90.
    Tr0 = 35.
    Tr1 = 37.
    Tr2 = 50.
    cphi = 1.*MW

    gas_net.nodes[2].p = gp2
    gas_net.links[3].q = q3
    elec_net.nodes[1].delta = delta1
    elec_net.nodes[2].delta = delta2
    elec_net.nodes[2].V = V2
    elec_net.links[3].Pstart = P3
    heat_net.links[0].m = m0
    heat_net.links[1].m = m1
    heat_net.links[2].m = m2
    heat_net.links[3].m = m3
    heat_net.nodes[0].p = hp0
    heat_net.nodes[2].p = hp2
    heat_net.nodes[0].Ts = Ts0
    heat_net.nodes[1].Ts = Ts1
    heat_net.nodes[2].Ts = Ts2
    heat_net.nodes[0].Tr = Tr0
    heat_net.nodes[1].Tr = Tr1
    heat_net.nodes[2].Tr = Tr2
    heat_net.links[3].dphistart = -cphi

    x0 = het_net.set_x_init()

    # Then
    x0_expected = np.array([gp2,delta1,delta2,V2,m0,m1,m2,hp0,hp2,Ts0,Ts1,Ts2,Tr0,Tr1,Tr2,q3,P3,0.,m3,cphi])#np.array([gp2,delta1,delta2,V2,m0,m1,m2,hp0,hp2,Ts0-Ta,Ts1-Ta,Ts2-Ta,Tr0-Ta,Tr1-Ta,Tr2-Ta,q3,P3,0.,m3,cphi])
    assert np.all(x0_expected==x0)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_update():
    """Test updating of heterogeneous network
    """
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_network()
    rho_w = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    rhon_g = gas.rhon #[kg/m^3]
    Ta = heat_net.Ta

    gp2 = 30*bar
    q3 = 100.*rhon_g/hour
    m0 = 3
    m1 = 2
    m2 = -1
    m3 = 3.
    delta1 = 0.
    delta2 = 0.1
    V2 = 2.
    P3 = 2.
    hp0 = 4.*rho_w*g
    hp2 = 2.*rho_w*g
    Ts0 = 100.
    Ts1 = 95.
    Ts2 = 90.
    Tr0 = 35.
    Tr1 = 37.
    Tr2 = 50.
    cphi = 1.*MW

    x = np.array([gp2,delta1,delta2,V2,m0,m1,m2,hp0,hp2,Ts0-Ta,Ts1-Ta,Ts2-Ta,Tr0-Ta,Tr1-Ta,Tr2-Ta,q3,P3,0.,m3,cphi])

    # When
    het_net.update(x)

    # Then
    assert np.all(x==het_net.set_x_init())

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_update_scaled():
    """Test updating of heterogeneous network, using per unit scaling
    """
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_network()
    rho_w = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    rhon_g = gas.rhon #[kg/m^3]
    Ta = heat_net.Ta
    mbase = 2
    pbase = 4*rho_w*g
    Tbase = 100.
    phibase = 1*MW#[W]
    Sbase = 1. #since P and Q are given in p.u.
    Vbase = 1.
    deltabase = 1.
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase,'Vbase':Vbase,'deltabase':deltabase,'Sbase':Sbase}
    gp = np.array([30*bar])
    delta = np.array([0.,0.1])
    V = np.array([2.])
    m = np.array([3.,2.,-1.])
    hp = np.array([4.,2.])*rho_w*g
    Ts = np.array([100.,95.,90.])#-Ta
    Tr = np.array([35.,37.,50.])#-Ta
    qc = np.array([100.*rhon_g/hour])
    Sc = np.array([2.,0.])
    mc = np.array([3.])
    phic = np.array([1.*MW])
    x = np.concatenate((gp,delta,V,m,hp,Ts,Tr,qc,Sc,mc,phic)) # unscaled
    x_scaled = np.concatenate((gp/pbase,delta/deltabase,V/Vbase,m/mbase,hp/pbase,Ts/Tbase,Tr/Tbase,qc/mbase,Sc/Sbase,mc/mbase,phic/phibase)) # scaled

    # When
    het_net.update(x)

    # Then
    assert np.allclose(x_scaled,het_net.set_x_init(scale_var=scale_var,scale_var_params=scale_var_params))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_update_full():
    """Test full updating of heterogeneous network
    """
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_network()
    het_net.initialize()
    rho_w = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    rhon_g = gas.rhon #[kg/m^3]
    Ta = heat_net.Ta

    gp = np.array([25.*bar])
    delta = np.array([-0.1, -0.1])
    V = np.array([0.98])
    m = np.array([2.,np.sqrt(5),1.])
    hp = np.array([6.,1.])*rho_w*g
    Ts = np.array([100.,99.143272336,98.683552499])#-Ta
    Tr = np.array([49.463145599,49.621044450,50.])#-Ta
    qc = np.array([108.1388730791121*rhon_g/hour])
    Sc = np.array([0.5,0.])
    mc = np.array([2.])
    phic = np.array([0.45*MW])
    x_sol = np.concatenate((gp,delta,V,m,hp,Ts,Tr,qc,Sc,mc,phic)) # unscaled

    # When
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(x_sol)
    m_hl = np.array([m for m_hl_list in m_hl_vec for m in m_hl_list])

    # Then
    m_hl_expected = np.array([-2.2360719125571014, 1.0000001199898445, 3.2360683951823188])
    assert np.allclose(m_hl_expected,m_hl)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dEin_dE_CHP():
    """Test the derivative of the incoming energy vector to total energy vector for the mes with a single CHP"""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_network()
    het_net.initialize()

    # When
    cn = list(het_net.get_nodes(carriers='het'))[0]
    dEin_dE = cn.dEin_dE()

    # Then
    dEin_dE_expected = np.array([cn.unit_params.get('GHV'),0,0]) # only active power is taken into account, not reactive power
    assert np.allclose(dEin_dE_expected,dEin_dE)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dEout_dE_CHP():
    """Test the derivative of the outgoing energy vector to total energy vector for the mes with a single CHP"""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_network()
    het_net.initialize()

    # When
    cn = list(het_net.get_nodes(carriers='het'))[0]
    dEout_dE = cn.dEout_dE()

    # Then
    dEout_dE_expected = np.array([[0,1,0],
                                  [0,0,1]]) # only active power is taken into account, not reactive power
    assert np.allclose(dEout_dE_expected,dEout_dE)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_node_law_der_E_coupling_ge():
    """Test the derivative of the node law for the coupling network, consisting of only a gas-fired generator (i.e., there are only halflinks connected to the coupling node)"""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_network()
    het_net.initialize()

    # When
    cn = list(het_net.get_nodes(carriers='het'))[0]
    dfc_dE = cn.der_node_law_dE()

    # Then
    dfc_dE_expected = np.array([cn.unit_params.get('GHV'), -1/cn.unit_params.get('eta')[0], -1/cn.unit_params.get('eta')[1]])
    assert np.allclose(dfc_dE_expected,dfc_dE)

def create_test_network_EH():
    """Create a small test heterogeneous network with one energy hub

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
    g2 = GasNode('gn2',node_type=0,p=1e5) # ref node
    e1 = ElectricalNode('en1',node_type=0,V=1.,delta=0.) # ref node
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp_w,rho=rho_w)
    h0 = HeatNode('hn0',node_type=0,Ts=100.,p=1e5) # source ref. slack node
    h0.half_links[0].set_type('heat_exchanger',{'carrier':water},bc_type=0)
    eta = 0.9
    nu = 0.5/0.95 # dispatch factor (to active power)
    GHV = 50.2*1e6 #[J/kg]
    C = np.array([[0,0,0],[eta*nu,0,0],[eta*(1-nu),0,0]])
    Ein_carriers = ['g','e','h']
    Eout_carriers = ['g','e','h']
    c0 = HeterogeneousNode('cn0',node_type=0,unit_type='EH',unit_params={'C':C,'GHV':GHV,'Ein_carriers':Ein_carriers,'Eout_carriers':Eout_carriers},Ein_len=np.array([1,1,1]),Eout_len=np.array([1,1,1])) # To unknown
    gl3 = GasLink('gl3',g2,c0)
    el3 = ElectricalLink('el3',c0,e1)
    hl3 = HeatLink('hl3',c0,h0,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
    gas_net = GasNetwork('test gas network')
    gas_net.add_link(gl3)

    elec_net = ElectricalNetwork('test electrical network')
    elec_net.add_link(el3)

    Ta = 10 #[C]
    heat_net = HeatNetwork('test heat network',Ta=Ta)
    heat_net.add_link(hl3)

    het_net = HeterogeneousNetwork('test heterogeneous network')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    het_net.add_node(c0)
    return het_net, gas_net, elec_net, heat_net, water

def initialize_test_network_EH(het_net):
    """Create the small test heterogeneous network with one energy hub"""
    Tr_init = np.array([50.])
    qc_init = np.array([100.*0.7/hour])
    Sc_init = np.array([2.,0.])
    mc_init = np.array([3.])
    phic_init = np.array([1.])*MW
    Tsc_init = np.array([90.])
    mbase = 1.
    pbase = 1.
    Tbase = 1.
    phibase = MW #since P and Q are in p.u., so for the coupling, the heat (and gas) power also need to be scaled to p.u.
    Sbase = 1. #since P and Q are in p.u.
    Vbase = 1.
    deltabase = 1.
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase,'Vbase':Vbase,'deltabase':deltabase,'Sbase':Sbase,'Ebase':phibase}

    x_init = np.concatenate((Tr_init/Tbase,qc_init/mbase,Sc_init/Sbase,mc_init/mbase,phic_init/phibase,Tsc_init/Tbase)) # scaled
    het_net.initialize()
    het_net.update(x_init,scale_var=scale_var,scale_var_params=scale_var_params)
    return x_init, scale_var, scale_var_params

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_get_nodes_EH():
    """Test getting only the heterogeneous nodes for the energy hub"""
    # Given
    het_net, gas_net, elec_net, heat_net, water = create_test_network_EH()

    # When
    het_net.initialize()
    EH_node = list(het_net.get_nodes(carriers=['het']))[0]

    # Then
    EH_node_expteced = het_net.nodes[-1]
    assert EH_node == EH_node_expteced

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_Eout_EH_full_C():
    """Test the outgoing energy vector for the energy hub, for the network with a 'full' 3x3 coupling matrix"""
    # Given
    het_net, gas_net, elec_net, heat_net, water = create_test_network_EH()
    x_init, scale_var, scale_var_params = initialize_test_network_EH(het_net)
    # When
    c0 = het_net.nodes[-1]
    Eout = c0.Eout(scale_var=scale_var,scale_var_params=scale_var_params)

    # Then
    Pc_init = elec_net.links[-1].Pstart
    phic_init = -heat_net.links[-1].dphistart
    phibase = scale_var_params['phibase']
    Eout_expected = np.array([0.,Pc_init,phic_init/phibase])
    assert np.allclose(Eout_expected,Eout)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_Ein_EH_full_C():
    """Test the incoming energy vector for the energy hub, for the network with a 'full' 3x3 coupling matrix"""
    # Given
    het_net, gas_net, elec_net, heat_net, water = create_test_network_EH()
    x_init, scale_var, scale_var_params = initialize_test_network_EH(het_net)

    # When
    c0 = het_net.nodes[-1]
    Ein = c0.Ein(scale_var=scale_var,scale_var_params=scale_var_params)

    # Then
    GHV = c0.unit_params['GHV']
    phibase = scale_var_params['phibase']
    qc_init = gas_net.links[-1].q
    Ein_expected = np.array([qc_init*GHV/phibase,0.,0.])
    assert np.allclose(Ein_expected,Ein)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dEoutdE_EH_full_C():
    """Test the derivative of the outgoing energy vector for the energy hub, for the network with a 'full' 3x3 coupling matrix"""
    # Given
    het_net, gas_net, elec_net, heat_net, water = create_test_network_EH()
    x_init, scale_var, scale_var_params = initialize_test_network_EH(het_net)

    # When
    c0 = het_net.nodes[-1]
    dEoutdE = c0.dEout_dE(scale_var=scale_var,scale_var_params=scale_var_params)

    # Then
    dEoutdE_expected = np.array([[0.,0.,0.],
                               [0.,1.,0.],
                               [0.,0.,1.]])
    assert np.allclose(dEoutdE_expected,dEoutdE)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dEindE_EH_full_C():
    """Test the derivative of the incoming energy vector for the energy hub, for the network with a 'full' 3x3 coupling matrix"""
    # Given
    het_net, gas_net, elec_net, heat_net, water = create_test_network_EH()
    x_init, scale_var, scale_var_params = initialize_test_network_EH(het_net)

    # When
    c0 = het_net.nodes[-1]
    dEindE = c0.dEin_dE(scale_var=scale_var,scale_var_params=scale_var_params)

    # Then
    GHV = c0.unit_params['GHV']
    phibase = scale_var_params['phibase']
    dEindE_expected = np.array([[GHV/phibase,0.,0.],
                               [0.,0.,0.],
                               [0.,0.,0.]])
    assert np.allclose(dEindE_expected,dEindE)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_node_law_der_E_EH_full_C():
    """Test the derivative of node law for the energy hub with a 'full' 3x3 coupling matrix"""
    # Given
    het_net, gas_net, elec_net, heat_net, water = create_test_network_EH()
    x_init, scale_var, scale_var_params = initialize_test_network_EH(het_net)

    # When
    c0 = het_net.nodes[-1]
    dfc_dq,dfc_dP,dfc_dphi = c0.der_node_law_dE(scale_var=scale_var,scale_var_params=scale_var_params)
    dfc_dE = np.vstack((dfc_dq,dfc_dP,dfc_dphi))
    # Then
    C = c0.unit_params['C']
    GHV = c0.unit_params['GHV']
    phibase = scale_var_params['phibase']
    dfc_dE_expected = np.array([[0, 0, 0],
                             [-C[1,0]*GHV, 1, 0],
                             [-C[2,0]*GHV/phibase, 0, 1]])
    assert np.allclose(dfc_dE_expected,dfc_dE)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_node_law_der_E_CHP_part_load():
    """Test the derivative of node law for the CHP with part load effect"""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_test_network()
    rhon_g = gas.rhon
    GHV = 40.611 #[MBTU/m^3]
    GHV *= MBTU*BTU #[Wh/m^3]
    GHV *= hour/rhon_g #[J/kg]
    a_CHP = .463
    b_CHP = -.04532*MW
    d_CHP = 4.49*MW
    eta_CHP = np.array([0.88, 0.88])
    phimin = 10*MW #??
    phimax = 14*MW/.48 # maximum (?) active power / power-to-heat-ratio
    L1 = .8
    # Should stay above the upper limit L1, so these following values shouldn't matter
    L2 = .6
    r1 = .0736
    r2 = .0845
    To_c_CHP = 130
    unit_type_CHP = 'geh_CHP_part_load'
    unit_params_CHP = {'eta':eta_CHP,'a':a_CHP,'b':b_CHP,'d':d_CHP,'L1':L1,'L2':L2,'r1':r1,'r1':r1,'phimin':phimin,'phimax':phimax,'GHV':GHV}
    c0 = het_net.nodes[-1]
    c0.set_type(unit_type_CHP,unit_params_CHP)
    heat_net.links[-1].dphistart = -25*MW # Need some value between the operating limits of the CHP

    # When
    Ein = c0.Ein()
    Eout = c0.Eout()
    Ts = heat_net.links[-1].get_Tsstart()
    dfc_dE = c0.der_node_law_dE()

    # Then
    dfc_dE_expected = np.array([[1*GHV, -1/eta_CHP[0], -1/eta_CHP[1]],
                             [0, 1, -a_CHP]])
    assert np.allclose(dfc_dE_expected,dfc_dE)

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

def test_x_init_coupling_ge():
    """Test the initial guess for coupling network"""
    # Given
    coupling_net = create_test_network_only_coupling_ge()

    # When
    qc_hl_init = -0.05
    coupling_net.nodes[0].half_links[0].q = qc_hl_init

    x0 = coupling_net.set_x_init()

    # Then
    x0_expected = np.array([-qc_hl_init])
    assert np.all(x0_expected==x0)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_update_coupling_ge():
    """Test updating the coupling network"""
    # Given
    coupling_net = create_test_network_only_coupling_ge()

    # When
    x = np.array([0.07])
    coupling_net.update(x)

    # Then
    assert np.all(-x[0]==coupling_net.nodes[0].half_links[0].q)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_Eout_coupling_ge():
    """Test the outgoing energy vector for the coupling network, consisting of only a gas-fired generator"""
    # Given
    coupling_net = create_test_network_only_coupling_ge()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[0].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[1].P = Pc_hl
    cn.half_links[1].Q = Qc_hl

    # When
    Eout = cn.Eout()

    # Then
    Eout_expected = np.array([Pc_hl]) # only active power is taken into account, not reactive power
    assert np.allclose(Eout_expected,Eout)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_Ein_coupling_ge():
    """Test the outgoing energy vector for the coupling network, consisting of only a gas-fired generator"""
    # Given
    coupling_net = create_test_network_only_coupling_ge()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[0].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[1].P = Pc_hl
    cn.half_links[1].Q = Qc_hl

    # When
    Ein = cn.Ein()

    # Then
    Ein_expected = np.array([-cn.unit_params.get('GHV')*qc_hl_init]) # only active power is taken into account, not reactive power
    assert np.allclose(Ein_expected,Ein)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dEout_dE_coupling_ge():
    """Test the derivative of the outgoing energy vector to total energy vector for the coupling network, consisting of only a gas-fired generator"""
    # Given
    coupling_net = create_test_network_only_coupling_ge()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[0].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[1].P = Pc_hl
    cn.half_links[1].Q = Qc_hl

    # When
    dEout_dE = cn.dEout_dE()

    # Then
    dEout_dE_expected = np.array([0,1,0]) # only active power is taken into account, not reactive power
    assert (len(dEout_dE>0) and np.allclose(dEout_dE_expected,dEout_dE))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dEin_dE_coupling_ge():
    """Test the derivative of the incoming energy vector to total energy vector for the coupling network, consisting of only a gas-fired generator"""
    # Given
    coupling_net = create_test_network_only_coupling_ge()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[0].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[1].P = Pc_hl
    cn.half_links[1].Q = Qc_hl

    # When
    dEin_dE = cn.dEin_dE()

    # Then
    dEin_dE_expected = np.array([cn.unit_params.get('GHV'),0,0]) # only active power is taken into account, not reactive power
    assert (len(dEin_dE)>0 and np.allclose(dEin_dE_expected,dEin_dE))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_node_law_coupling_ge():
    """Test the node law for the coupling network, consisting of only a gas-fired generator (i.e., there are only halflinks connected to the coupling node)"""
    # Given
    coupling_net = create_test_network_only_coupling_ge()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[0].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[1].P = Pc_hl
    cn.half_links[1].Q = Qc_hl

    # When
    F = cn.node_law()

    # Then
    F_expected = Pc_hl - cn.unit_params.get('eta')*cn.unit_params.get('GHV')*-qc_hl_init
    assert np.allclose(F_expected,F)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_node_law_der_Eout_coupling_ge():
    """Test the derivative of the node law to outgoing energy for the coupling network, consisting of only a gas-fired generator (i.e., there are only halflinks connected to the coupling node)"""
    # Given
    coupling_net = create_test_network_only_coupling_ge()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[0].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[1].P = Pc_hl
    cn.half_links[1].Q = Qc_hl

    # When
    Ein = cn.Ein()
    Eout = cn.Eout()
    dfc_dEout = cn.df_dEout(Ein,Eout,None)

    # Then
    dfc_dEout_expected = np.array([1])
    assert np.allclose(dfc_dEout_expected,dfc_dEout)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_node_law_der_Ein_coupling_ge():
    """Test the derivative of the node law to outgoing energy for the coupling network, consisting of only a gas-fired generator (i.e., there are only halflinks connected to the coupling node)"""
    # Given
    coupling_net = create_test_network_only_coupling_ge()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[0].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[1].P = Pc_hl
    cn.half_links[1].Q = Qc_hl

    # When
    Ein = cn.Ein()
    Eout = cn.Eout()
    dfc_dEin = cn.df_dEin(Ein,Eout,None)

    # Then
    dfc_dEin_expected = np.array([-cn.unit_params.get('eta')])
    assert np.allclose(dfc_dEin_expected,dfc_dEin)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_node_law_der_E_coupling_ge():
    """Test the derivative of the node law for the coupling network, consisting of only a gas-fired generator (i.e., there are only halflinks connected to the coupling node)"""
    # Given
    coupling_net = create_test_network_only_coupling_ge()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[0].q = qc_hl_init
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[1].P = Pc_hl
    cn.half_links[1].Q = Qc_hl

    # When
    Ein = cn.Ein()
    Eout = cn.Eout()
    dfc_dE = cn.der_node_law_dE()

    # Then
    dfc_dE_expected = np.array([-cn.unit_params.get('eta')*cn.unit_params.get('GHV'), 1, 0])
    assert np.allclose(dfc_dE_expected,dfc_dE)

def create_test_network_only_coupling_gh():
    """Create a heterogeneous network, consisting of only a gas boiler. The heat power and outflow temperature are assumed known, the gas flow is assumed unknown.
    """
    eta_GB = .7
    GHV = 60134305#[J/kg]
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    grav_const = 9.81 #[m/s^2]
    water = Water('water',Cp_w,rho=rho_w)
    To_c = 100.
    c0 = HeterogeneousNode('cn0',node_type=1,unit_type='gh_gas_boiler',unit_params={'eta':eta_GB,'GHV':GHV}) # To known
    hlh = HeatHalfLink('cn0_hlh',c0,Ts=To_c,dphi=-1*MW,link_type='heat_exchanger',bc_type=2,link_params={'carrier':water}) # Ts and dphi known, m unknown (and Tr)
    hlg = GasHalfLink('cn0_hlg',c0,-0.05,bc_type=0) # gas flows into coupling node, q is unknown
    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)

    coupling_net.initialize()
    return coupling_net

def test_x_init_coupling_gh():
    """Test the initial guess for coupling network, consisting of only a gas boiler."""
    # Given
    coupling_net = create_test_network_only_coupling_gh()

    # When
    qc_hl_init = -0.08
    mc_hl_init = -1.01
    coupling_net.nodes[0].half_links[0].m = mc_hl_init
    coupling_net.nodes[0].half_links[1].q = qc_hl_init

    x0 = coupling_net.set_x_init()

    # Then
    x0_expected = np.array([-qc_hl_init,-mc_hl_init])
    assert np.all(x0_expected==x0)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_update_coupling_gh():
    """Test updating the coupling network, consisting of only a gas boiler."""
    # Given
    coupling_net = create_test_network_only_coupling_gh()

    # When
    x = np.array([0.07,1.03])
    coupling_net.update(x)

    # Then
    assert np.all(-x==[coupling_net.nodes[0].half_links[1].q,coupling_net.nodes[0].half_links[0].m])

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_Eout_coupling_gh():
    """Test the outgoing energy vector for the coupling network, consisting of only a gas boiler."""
    # Given
    coupling_net = create_test_network_only_coupling_gh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl

    # When
    Eout = cn.Eout()

    # Then
    Eout_expected = np.array([-phic_hl])
    assert np.allclose(Eout_expected,Eout)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_Ein_coupling_gh():
    """Test the outgoing energy vector for the coupling network, consisting of only a gas boiler."""
    # Given
    coupling_net = create_test_network_only_coupling_gh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl

    # When
    Ein = cn.Ein()

    # Then
    Ein_expected = np.array([-cn.unit_params.get('GHV')*qc_hl_init])
    assert np.allclose(Ein_expected,Ein)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dEout_dE_coupling_gh():
    """Test the derivative of the outgoing energy vector to total energy vector for the coupling network, consisting of only a gas boiler."""
    # Given
    coupling_net = create_test_network_only_coupling_gh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl

    # When
    dEout_dE = cn.dEout_dE()

    # Then
    dEout_dE_expected = np.array([0,0,1])
    assert (len(dEout_dE>0) and np.allclose(dEout_dE_expected,dEout_dE))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dEin_dE_coupling_gh():
    """Test the derivative of the incoming energy vector to total energy vector for the coupling network, consisting of only a gas boiler."""
    # Given
    coupling_net = create_test_network_only_coupling_gh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl

    # When
    dEin_dE = cn.dEin_dE()

    # Then
    dEin_dE_expected = np.array([cn.unit_params.get('GHV'),0,0])
    assert (len(dEin_dE)>0 and np.allclose(dEin_dE_expected,dEin_dE))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_node_law_coupling_gh():
    """Test the node law for the coupling network, consisting of only a gas boiler."""
    # Given
    coupling_net = create_test_network_only_coupling_gh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl

    # When
    F = cn.node_law()

    # Then
    F_expected = -phic_hl - cn.unit_params.get('eta')*cn.unit_params.get('GHV')*-qc_hl_init
    assert np.allclose(F_expected,F)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_node_law_der_E_coupling_gh():
    """Test the derivative of the node law for the coupling network, consisting of only a gas boiler."""
    # Given
    coupling_net = create_test_network_only_coupling_gh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.05
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl

    # When
    Ein = cn.Ein()
    Eout = cn.Eout()
    dfc_dE = cn.der_node_law_dE()

    # Then
    dfc_dE_expected = np.array([-cn.unit_params.get('eta')*cn.unit_params.get('GHV'), 0, 1])
    assert np.allclose(dfc_dE_expected,dfc_dE)

def create_test_network_only_coupling_geh():
    """Create a heterogeneous network, consisting of only a CHP. The active and reactive power, heat power, and outflow temperature are assumed known, the gas flow is assumed unknown.
    """
    eta_CHP = np.array([.6,.7])
    GHV = 60134305#[J/kg]
    rho_w = 960. #[kg/m^3]
    Cp_w = 4.182e3 #[J/(kg K)]
    grav_const = 9.81 #[m/s^2]
    water = Water('water',Cp_w,rho=rho_w)
    To_c = 100.
    c0 = HeterogeneousNode('cn0',node_type=1,unit_type='geh_CHP',unit_params={'eta':eta_CHP,'GHV':GHV}) # To known
    hlh = HeatHalfLink('cn0_hlh',c0,Ts=To_c,dphi=-1*MW,link_type='heat_exchanger',bc_type=2,link_params={'carrier':water}) # Ts and dphi known, m unknown (and Tr)
    hlg = GasHalfLink('cn0_hlg',c0,-0.05,bc_type=0) # gas flows into coupling node, q is unknown
    hle = ElectricalHalfLink('cn0_hle',c0,P=2*MW,Q=2*MW,bc_type=1) # P and Q are known
    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)

    coupling_net.initialize()
    return coupling_net

def test_x_init_coupling_geh():
    """Test the initial guess for coupling network, consisting of only a CHP."""
    # Given
    coupling_net = create_test_network_only_coupling_geh()

    # When
    qc_hl_init = -0.08
    mc_hl_init = -1.01
    coupling_net.nodes[0].half_links[0].m = mc_hl_init
    coupling_net.nodes[0].half_links[1].q = qc_hl_init

    x0 = coupling_net.set_x_init()

    # Then
    x0_expected = np.array([-qc_hl_init,-mc_hl_init])
    assert np.all(x0_expected==x0)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_Eout_coupling_geh():
    """Test the outgoing energy vector for the coupling network, consisting of only a CHP."""
    # Given
    coupling_net = create_test_network_only_coupling_geh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.08
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[2].P = Pc_hl
    cn.half_links[2].Q = Qc_hl

    # When
    Eout = cn.Eout()

    # Then
    Eout_expected = np.array([Pc_hl, -phic_hl]) # only active power is taken into account, not reactive power
    assert np.allclose(Eout_expected,Eout)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_Ein_coupling_geh():
    """Test the outgoing energy vector for the coupling network, consisting of only a CHP"""
    # Given
    coupling_net = create_test_network_only_coupling_geh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.08
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[2].P = Pc_hl
    cn.half_links[2].Q = Qc_hl

    # When
    Ein = cn.Ein()

    # Then
    Ein_expected = np.array([-cn.unit_params.get('GHV')*qc_hl_init])
    assert np.allclose(Ein_expected,Ein)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dEout_dE_coupling_geh():
    """Test the derivative of the outgoing energy vector to total energy vector for the coupling network, consisting of only a CHP."""
    # Given
    coupling_net = create_test_network_only_coupling_geh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.08
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[2].P = Pc_hl
    cn.half_links[2].Q = Qc_hl

    # When
    dEout_dE = cn.dEout_dE()

    # Then
    dEout_dE_expected = np.array([[0,1,0],
                                 [0,0,1]])
    assert (len(dEout_dE>0) and np.allclose(dEout_dE_expected,dEout_dE))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dEin_dE_coupling_geh():
    """Test the derivative of the incoming energy vector to total energy vector for the coupling network, consisting of only a CHP."""
    # Given
    coupling_net = create_test_network_only_coupling_geh()
    cn = coupling_net.nodes[0]
    qc_hl_init = -0.08
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[2].P = Pc_hl
    cn.half_links[2].Q = Qc_hl

    # When
    dEin_dE = cn.dEin_dE()

    # Then
    dEin_dE_expected = np.array([cn.unit_params.get('GHV'),0,0])
    assert (len(dEin_dE)>0 and np.allclose(dEin_dE_expected,dEin_dE))

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_node_law_coupling_geh():
    """Test the node law for the coupling network, consisting of only a CHP."""
    # Given
    coupling_net = create_test_network_only_coupling_geh()
    cn = coupling_net.nodes[0]
    eta_CHP = cn.unit_params.get('eta')
    qc_hl_init = -0.08
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[2].P = Pc_hl
    cn.half_links[2].Q = Qc_hl

    # When
    F = cn.node_law()

    # Then
    F_expected = cn.unit_params.get('GHV')*-qc_hl_init - Pc_hl/eta_CHP[0] - -phic_hl/eta_CHP[1]
    assert np.allclose(F_expected,F)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_node_law_der_E_coupling_geh():
    """Test the derivative of the node law for the coupling network, consisting of only a CHP."""
    # Given
    coupling_net = create_test_network_only_coupling_geh()
    cn = coupling_net.nodes[0]
    eta_CHP = cn.unit_params.get('eta')
    qc_hl_init = -0.08
    cn.half_links[1].q = qc_hl_init
    phic_hl = -2*MW
    cn.half_links[0].dphi = phic_hl
    Pc_hl = 2*MW
    Qc_hl = 3*MW
    cn.half_links[2].P = Pc_hl
    cn.half_links[2].Q = Qc_hl

    # When
    Ein = cn.Ein()
    Eout = cn.Eout()
    dfc_dE = cn.der_node_law_dE()

    # Then
    dfc_dE_expected = np.array([cn.unit_params.get('GHV'), -1/eta_CHP[0], -1/eta_CHP[1]])
    assert np.allclose(dfc_dE_expected,dfc_dE)
