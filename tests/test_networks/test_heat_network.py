"""Test the heat network classes and methods"""
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
import numpy as np

def create_test_network():
    """Create a test heat network

    Returns
    -------
    heat_net : HeatNetwork
        The test network
    water : Carrier
        The carrier in the network
    """
    MW = 1e6 #[W]
    # carrier
    rho = 960. #[kg/m^3]
    Cp = 4.182e3 #[J/(kg K)]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp,rho=rho)

    Ta = 10. #[C]
    heat_net = HeatNetwork('test heat network',Ta=Ta)

    hn0 = HeatNode('hn0',node_type=0,Ts=100.,p=11.*rho*g) # source slack node
    hn0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn1 = HeatNode('hn1',node_type=1,Ts_hl=100.,dphi=-0.5*MW) # source node
    hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn2 = HeatNode('hn2',node_type=1,Tr_hl=50.,dphi=1.3*MW) # load node
    hn2.half_links[0].set_type('heat_exchanger',{'carrier':water})

    C = 1.
    L = 400. #[m]
    D = 0.01 #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D) #[W/(m^2 K)]
    link_type = 'standard_resistor'
    link_params = {'L':L,'D':D,'U':U,'C':C,'carrier':water}
    hl0 = HeatLink('hl0',hn0,hn1,link_type=link_type,link_params=link_params)
    hl1 = HeatLink('hl1',hn0,hn2,link_type=link_type,link_params=link_params.copy()) # If I don't make a copy, then these link parameters are the same object. If Ta is then adjusted when hl0 is added, so are Ta in the link parameters of the other links. There link type / equations are then not updated when adding the link
    hl2 = HeatLink('hl2',hn1,hn2,link_type=link_type,link_params=link_params.copy())


    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    return heat_net, water

def test_incidence_matrix():
    """Test creating a heat network by checking the incidence matrix.
    """
    # Given
    heat_net, _ = create_test_network()

    # When
    heat_net.initialize()

    # Then
    A_test = np.array([[-1., -1.,  0.],
                    [ 1.,  0., -1.],
                    [ 0.,  1.,  1.]])
    assert np.all(A_test == heat_net.A)

def test_adjacency_matrix():
    """Test creating a heat network by checking the adjacency matrix.
    """
    # Given
    heat_net, _ = create_test_network()

    # When
    heat_net.initialize()

    # Then
    Ad_test = np.array([[ 0.,  1.,  1.],
                    [ 0.,  0.,  1.],
                    [ 0.,  0.,  0.]])
    assert np.all(Ad_test == heat_net.Ad)

def test_half_links():
    """Test the creation and adding of half links to the nodes
    """
    # Given
    heat_net, _ = create_test_network()

    # When
    n = heat_net.nodes[1] # the load node

    # Then
    number_hl_expected = 1
    assert number_hl_expected == len(n.half_links)

def test_x_init():
    """Test the initial guess for the heat network
    """
    # Given
    heat_net, water = create_test_network()
    rho = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    Ta = heat_net.Ta

    # When
    m0 = 2.
    m1 = 2.
    m2 = -1.
    m_init = np.array([m0,m1,m2])
    p1 = 4.*rho*g
    p2 = 3.*rho*g
    Ts1 = 100.
    Ts2 = 80.
    Tr0 = 30.
    Tr1 = 50.
    Tr2 = 35.
    for n in heat_net.get_nodes():
        if n.name == 'hn0':
            n.Tr = Tr0
        elif n.name == 'hn1':
            n.p = p1
            n.Ts = Ts1
            n.Tr = Tr1
        elif n.name == 'hn2':
            n.p = p2
            n.Ts = Ts2
            n.Tr = Tr2
    for ind_l,l in enumerate(heat_net.get_links()):
        l.m = m_init[ind_l]
    x0 = heat_net.set_x_init()

    # Then
    x0_expected = np.concatenate([m_init,np.array([p1,p2,Ts1,Ts2,Tr0,Tr1,Tr2])])#np.concatenate([m_init,np.array([p1,p2,Ts1-Ta,Ts2-Ta,Tr0-Ta,Tr1-Ta,Tr2-Ta])])
    assert np.all(x0_expected == x0)

def test_x_init_half_link_flows():
    """Test the initial guess for the heat network when the 'half_link_flow' formulation is used.
    """
    # Given
    heat_net, water = create_test_network()
    rho = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    Ta = heat_net.Ta
    heat_net.initialize() # needed to add the half links to the network

    # When
    m0 = 2.
    m1 = 2.
    m2 = -1.
    m_init = np.array([m0,m1,m2])
    m1_hl = -1. # source node
    m2_hl = 1. # sink node
    p1 = 4.*rho*g
    p2 = 3.*rho*g
    Ts1 = 100.
    Ts2 = 80.
    Tr0 = 30.
    Tr1 = 50.
    Tr2 = 35.
    for n in heat_net.get_nodes():
        if n.name == 'hn0':
            n.Tr = Tr0
        elif n.name == 'hn1':
            n.p = p1
            n.Ts = Ts1
            n.Tr = Tr1
            n.half_links[0].m = m1_hl
        elif n.name == 'hn2':
            n.p = p2
            n.Ts = Ts2
            n.Tr = Tr2
            n.half_links[0].m = m2_hl
    for ind_l,l in enumerate(heat_net.get_links()):
        l.m = m_init[ind_l]
    x0 = heat_net.set_x_init(formulation='half_link_flow')

    # Then
    x0_expected = np.concatenate([m_init,np.array([m1_hl,m2_hl,p1,p2,Ts1,Ts2,Tr0,Tr1,Tr2])])
    assert np.all(x0_expected == x0)
    
def test_update():
    """Test updating of heat network
    """
    # Given
    heat_net, water = create_test_network()
    heat_net.initialize()
    rho = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    Ta = heat_net.Ta
    x = np.array([2,2,-1,4*rho*g,3*rho*g,100,80,30,50,35])#np.array([2,2,-1,4*rho*g,3*rho*g,100-Ta,80-Ta,30-Ta,50-Ta,35-Ta])

    # When
    heat_net.update(x)

    # Then
    assert np.all(x == heat_net.set_x_init())
    
def test_update_half_link_flows():
    """Test updating of heat network when the 'half_link_flow' formulation is used.
    """
    # Given
    heat_net, water = create_test_network()
    heat_net.initialize()
    rho = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    Ta = heat_net.Ta
    x = np.array([2,2,-1,-1,1,4*rho*g,3*rho*g,100,80,30,50,35])

    # When
    heat_net.update(x,formulation='half_link_flow')

    # Then
    assert np.all(x == heat_net.set_x_init(formulation='half_link_flow'))

def test_update_scaled():
    """Test updating of heat network, using per unit scaling
    """
    # Given
    heat_net, water = create_test_network()
    heat_net.initialize()
    rho = water.rhon #[kg/m^3]
    g = water.g #[m/s^2]
    Ta = heat_net.Ta
    mbase = 2
    pbase = 4*rho*g
    Tbase = 100.
    phibase = 1e6 #[W]
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}
    m = np.array([2,2,-1])
    p = np.array([4*rho*g,3*rho*g])
    Ts = np.array([100,80])
    Tr = np.array([30,50,35])
    x = np.concatenate((m,p,Ts,Tr)) # unscaled
    x_scaled = np.concatenate((m/mbase,p/pbase,Ts/Tbase,Tr/Tbase)) # scaled
    # When
    heat_net.update(x)

    # Then
    assert np.allclose(x_scaled,heat_net.set_x_init(scale_var=scale_var,scale_var_params=scale_var_params))

def test_node_law():
    """Test the node law for a heat network with standard resistor pipes
    """
    # Given
    heat_net,water = create_test_network()
    heat_net.initialize()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    m_init = np.array([1.,np.sqrt(10.),3.])
    p_init = np.array([10.*rho*g,1.*rho*g])
    Ts_init = np.array([100.,100.])
    Tr_init = np.array([50.,50.,50.])
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.update(x_init)

    # When
    m_mismatch_expected = np.array([0.39120038259206114,-0.054843334570979785])
    m_mismatch_calc = np.array([n.node_law() for n in heat_net.nodes[1:]]) # the source node and the load node

    # Then
    assert np.allclose(m_mismatch_expected,m_mismatch_calc)

def test_node_law_scaled():
    """Test the node law for a heat network with standard resistor pipes, using per unit scaling
    """
    # Given
    heat_net,water = create_test_network()
    heat_net.initialize()
    rho = water.rhon
    g = water.g
    Ta = heat_net.Ta
    mbase = 2
    pbase = 4*rho*g
    Tbase = 100.
    phibase = 1e6 #[W]
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}
    m_init = np.array([1.,np.sqrt(10.),3.])
    p_init = np.array([10.*rho*g,1.*rho*g])
    Ts_init = np.array([100.,100.])
    Tr_init = np.array([50.,50.,50.])
    x_init = np.concatenate((m_init/mbase,p_init/pbase,Ts_init/Tbase,Tr_init/Tbase)) #scaled
    heat_net.update(x_init,scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    m_mismatch_expected = np.array([0.39120038259206114,-0.054843334570979785])/mbase
    m_mismatch_calc = np.array([n.node_law(scale_var=scale_var,scale_var_params=scale_var_params) for n in heat_net.nodes[1:]]) # the source node and the load node

    # Then
    assert np.allclose(m_mismatch_expected,m_mismatch_calc)

def test_mixing_rule():
    """Test the mixing rule for a heat network with standard resistor pipes
    """
    # Given
    heat_net,water = create_test_network()
    heat_net.initialize()
    Ta = heat_net.Ta
    rho = water.rhon
    g = water.g
    m_init = np.array([1.,np.sqrt(10.),3.])
    p_init = np.array([10.,1.])*rho*g
    Ts_init = np.array([100.,100.])
    Tr_init = np.array([50.,50.,50.])
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.update(x_init)

    # When
    Ts_mismatch_calc = list()
    Tr_mismatch_calc = list()
    for n in heat_net.nodes[1:]:# the source node and the load node
        fTs,fTr = n.mixing_rule(network=heat_net)
        Ts_mismatch_calc.append(fTs)
        Tr_mismatch_calc.append(fTr)
    FT_mismatch_calc = np.array([Ts_mismatch_calc + Tr_mismatch_calc]) # the source node and the load node

    # Then
    FT_mismatch_expected = np.array([-37.41473685630069, 8.9169875737028406, 20.322768817777046, -2.7421667285489093 ])#np.array([-33.502733030380057, 8.36855423, 16.410764991856468, -2.1937333828392127])
    assert np.allclose(FT_mismatch_expected,FT_mismatch_calc)

def test_mixing_rule_scaled():
    """Test the mixing rule for a heat network with standard resistor pipes, using per unit scaling
    """
    # Given
    heat_net,water = create_test_network()
    heat_net.initialize()
    Ta = heat_net.Ta
    rho = water.rhon
    g = water.g
    mbase = 2
    pbase = 4*rho*g
    Tbase = 100.
    phibase = 1e6 #[W]
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}
    m_init = np.array([1.,np.sqrt(10.),3.])
    p_init = np.array([10.*rho*g,1.*rho*g])
    Ts_init = np.array([100.,100.])
    Tr_init = np.array([50.,50.,50.])
    x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.update(x_init)

    # When
    Ts_mismatch_calc = list()
    Tr_mismatch_calc = list()
    T_shift = heat_net.get_Ta(scale_var=scale_var,scale_var_params=scale_var_params)
    for n in heat_net.nodes[1:]:
        print("node: {}, Ts={}, Tr={}, Ts'={}, Tr'={}".format(n.name,n.Ts,n.Tr,n.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),n.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)))
        fTs,fTr = n.mixing_rule(network=heat_net,scale_var=scale_var,scale_var_params=scale_var_params)
        Ts_mismatch_calc.append(fTs)
        Tr_mismatch_calc.append(fTr)
    FT_mismatch_calc = np.array([Ts_mismatch_calc + Tr_mismatch_calc]) # the source node and the load node

    # Then
    FT_mismatch_expected = np.array([-37.41473685630069, 8.9169875737028406, 20.322768817777046, -2.7421667285489093 ])/(mbase*Tbase)#np.array([-33.502733030380057, 8.36855423, 16.410764991856468, -2.1937333828392127])/(mbase*Tbase)
    assert np.allclose(FT_mismatch_expected,FT_mismatch_calc)

def test_update_full():
    """Test full updating of heat network
    """
    # Given
    heat_net,water = create_test_network()
    heat_net.initialize()
    Ta = heat_net.Ta
    m_sol = np.array([0.73722693, 3.20278262, 3.11677927])
    p_sol = np.array([98475.1010112, 6989.58696913])
    Ts_sol = np.array([99.45471939, 99.18949369])
    Tr_sol = np.array([49.57003606, 49.75524712, 50.])
    x_sol= np.concatenate((m_sol,p_sol,Ts_sol,Tr_sol))

    # When
    m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(x_sol)
    m_hl = np.array([m for m_hl_list in m_hl_vec for m in m_hl_list])

    # Then
    m_hl_expected = np.array([-3.9400095536145106, -2.3795523366391302, 6.3195618902536408])
    assert np.allclose(m_hl_expected,m_hl)
    
def test_standard_pipe_heat_loss():
    # Given
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1000 #[kg/m^3]
    water = Water('water',Cp,rho=rho)
    L = 1e3 #[m]
    D = 0.2 #[m]
    U = 10 #[W/(m^2 K)
    Ta = 20 #[C]
    Tsstart = 70. #[C]
    Tsend = 67.664 #[C]
    v = 1 #[m/s]
    m = np.pi*D**2/4 * v*rho #[kg/s]
    
    hn0 = HeatNode('hn0',Ts=Tsstart) 
    hn1 = HeatNode('hn1',Ts=Tsend) 
    link_type = 'standard_resistor'
    link_params = {'L':L,'D':D,'U':U,'C':1,'carrier':water,'Ta':Ta}
    pipe = HeatLink('hl0',hn0,hn1,link_type=link_type,link_params=link_params)
    
    pipe.m = m
    pipe.Tsstart = Tsstart
    pipe.Tsend = Tsend
    
    # When
    pipe.phisstart = pipe.supply_heat_power_start()
    pipe.phisend = pipe.supply_heat_power_end()
    philoss = pipe.heat_loss_supply()

    # Then
    philoss_expected = 306760.18634124444 #[W]
    assert np.isclose(philoss_expected,philoss)
