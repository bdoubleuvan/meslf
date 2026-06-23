"""Test the gas network classes and methods
"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
import numpy as np
import pandas as pd
import os.path

def create_test_network():
    gas_net = GasNetwork('test gas network')
    g0 = GasNode('g0',node_type=0,p=30.) # reference node 
    g1 = GasNode('g1',node_type=3,p=25.,q=250.) # reference load node 
    g2 = GasNode('g2',node_type=2) # slack node
    g3 = GasNode('g3',node_type=1,q=180.) # load node
    
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

def test_incidence_matrix():
    """Test the creating of a gas network by checking the incidence matrix.
    """
    # Given 
    gas_net = create_test_network()
    
    # When
    gas_net.initialize()
    
    # Then
    A_test = np.array([[-1., -1., -1.,  0.,  0.],
        [ 1.,  0.,  0.,  1.,  0.],
        [ 0.,  1.,  0., -1., -1.],
        [ 0.,  0.,  1.,  0.,  1.]])
    assert np.all(A_test == gas_net.A)

def test_half_links():
    """Test the creation and adding of half links to the nodes
    """
    # Given 
    gas_net = create_test_network()
    
    # When
    n = gas_net.nodes[-1] # the load node
    
    # Then
    number_hl_expected = 1
    assert number_hl_expected == len(n.half_links)
    
def test_x_init_form_nodal():
    """Test the creation of the initial gues of the gas network using the nodal formulation.
    """
    # Given
    form = 'nodal'
    gas_net = create_test_network()
    
    # When
    p2 = 24.
    p3 = 28.
    for n in gas_net.get_nodes():
        if n.name == 'g2':
            n.p = p2
        elif n.name == 'g3':
            n.p = p3
    
    # Then 
    x0 = np.array([p2,p3])
    assert np.all(x0 == gas_net.set_x_init(formulation=form))

def test_x_init_form_full():
    """Test the creationg of the initial guess for the gas network using the full formulation.
    """
    # Given
    form = 'full'
    gas_net = create_test_network()
    
    # When
    p2 = 24.
    p3 = 28.
    q = np.array([200.,150.,150.,50.,15.])
    for n in gas_net.get_nodes():
        if n.name == 'g2':
            n.p = p2
        elif n.name == 'g3':
            n.p = p3
    for ind_e,e in enumerate(gas_net.get_links()):
        e.q = q[ind_e]
    
    # Then
    x0 = np.concatenate((q,np.array([p2,p3])))
    assert np.all(x0 == gas_net.set_x_init(formulation=form))
    
def test_update_form_nodal():
    """Test updating of the gas network using the nodal formulation
    """
    # Given
    form = 'nodal'
    gas_net = create_test_network()
    gas_net.initialize()
    x = np.array([24.,28.])
    
    # When
    gas_net.update(x,formulation=form)
    
    # Then
    assert (np.all(x == gas_net.set_x_init(formulation=form)))
    
    
def test_update_form_full():
    """Test updating of the gas network using the full formulation
    """
    # Given
    form = 'full'
    gas_net = create_test_network()
    gas_net.initialize()
    p = np.array([24.,28.])
    q = np.array([200.,150.,150.,50.,15.])
    x = np.concatenate((q,p))
    
    # When
    gas_net.update(x,formulation=form)
    
    # Then
    assert np.all(x == gas_net.set_x_init(formulation=form))
    
