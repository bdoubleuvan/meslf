"""Test the electrical network classes and methods
"""
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
import numpy as np

def create_test_network():
    """Create a test electical network
    
    Returns 
    -------
    elec_net : ElectricalNetwork
        The test network
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
    
def test_admittance_matrix():
    """Test the electrical network by checking the admittance matrix
    """
    # Given
    elec_net = create_test_network()
    
    # When
    elec_net.initialize()
    
    # Then
    Y_expected = np.array([[ 2.-20.*1j, -1.+10.*1j, -1.+10.*1j],
                        [-1.+10.*1j,  2.-20.*1j, -1.+10.*1j],
                        [-1.+10.*1j, -1.+10.*1j,  2.-20.*1j]])
    assert np.all(Y_expected == elec_net.Y)

def test_half_links():
    """Test the creation and adding of half links to the nodes
    """
    # Given 
    elec_net = create_test_network()
    
    # When
    n = elec_net.nodes[1] # the load node
    
    # Then
    number_hl_expected = 1
    assert number_hl_expected == len(n.half_links)
    
def test_x_init():
    """Test the initial guess for the electrical network
    """
    # Given
    elec_net = create_test_network()
    
    # When
    V1 = 0.99
    delta1 = -0.1
    delta2 = -0.2
    for n in elec_net.get_nodes():
        if n.name == 'en1':
            n.delta = delta1
            n.V = V1
        elif n.name == 'en2':
            n.delta = delta2
    x0 = elec_net.set_x_init()

    # Then
    x0_expected = np.array([delta1, delta2, V1])
    assert np.all(x0_expected == x0)
    
def test_update():
    """Test updating of electical network
    """
    # Given 
    elec_net = create_test_network()
    elec_net.initialize()
    x = np.array([-0.1,-0.2,.99])
    
    # When
    elec_net.update(x)
    
    # Then
    assert np.all(x == elec_net.set_x_init())

def test_node_law():
    """Test the node law for an electrical network with only short lines
    """
    # Given 
    elec_net = create_test_network()
    elec_net.initialize()
    x = np.array([0.,0.,1.])
    elec_net.update(x)
    
    # When
    en = elec_net.nodes[1] # the load node
    S_mismatch_expected = np.array([1.0131000000000001, 0.24919999999999928])
    S_mismatch_calc = np.array([en.node_law()])
    
    # Then
    assert np.allclose(S_mismatch_expected,S_mismatch_calc)
    
def test_update_full():
    """Test full updating of electrical network
    """
    # Given 
    elec_net = create_test_network()
    elec_net.initialize()
    delta_sol = np.array([-0.10000294, -0.10000295])
    V_sol = np.array([0.97999991])
    x_sol = np.concatenate((delta_sol,V_sol))
    
    # When
    delta_vec,V_mag_vec,S_inj,P_edge,Q_edge = elec_net.update_full(x_sol)
    S_inj_expected = np.array([-2.00658483-0.30224574*1j, 0.99310000+0.0492*1j, 0.99310000+0.04919747*1j],dtype=complex)
    
    # Then
    assert np.allclose(S_inj_expected,S_inj)
    
    
