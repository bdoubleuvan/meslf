"""Test the reading and writing of network from and to pd from_pd_dataframes"""
from meslf.networks.read_write_network import to_pd_dataframes, from_pd_dataframes
from meslf.networks.network import Network, Node, Link, HalfLink
from meslf.networks.create_network import create_radial_line_network, add_gas_data, add_elec_data, add_heat_data
from meslf.networks.carrier import Gas, Water
import examples.network_data.N_1C.make_GN_1C as make_GN_1C
import pandas as pd
import numpy as np
import os.path

import pytest
# ===================================================================================
# Test writing a network to a pandas dataframe
def test_to_pd_dataframe_nodes():
    """Test writing a network to a pandas dataframe. Check the nodes
    """
    # Given
    node0 = Node('n0')
    node1 = Node('n1')
    node1.x = 1
    node1.y = 2
    halflink = HalfLink('n1_hl',node1)
    link = Link('l0',node0,node1)
    net = Network('net')
    net.add_link(link)
    net.add_half_link(halflink)

    # When
    nodes, _, _ = to_pd_dataframes(net)

    # Then
    node_names_expected = ['n0','n1']
    carrier_nodes_expected = ['no_type']*2
    node_types_expected = ['no_type']*2
    x_expected = [0,1]
    y_expected = [0,2]
    nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected]).T,columns=['name','carrier','type','x','y'])
    pd.testing.assert_frame_equal(nodes_expected,nodes.loc['net'])

def test_to_pd_dataframe_links():
    """Test writing a network to a pandas dataframe. Check the links
    """
    # Given
    node0 = Node('n0')
    node1 = Node('n1')
    node1.x = 1
    node1.y = 2
    halflink = HalfLink('n1_hl',node1)
    link = Link('l0',node0,node1)
    net = Network('net')
    net.add_link(link)
    net.add_half_link(halflink)

    # When
    _, links, _ = to_pd_dataframes(net)

    # Then
    link_names_expected = ['l0']
    carrier_links_expected = ['no_type']
    start_nodes_expected = [('net',0)]
    end_nodes_expected = [('net',1)]

    links_expected = pd.DataFrame({'name':link_names_expected,'carrier':carrier_links_expected,'start_node':start_nodes_expected,'end_node':end_nodes_expected},columns=['name','carrier','start_node','end_node'])

    pd.testing.assert_frame_equal(links_expected,links.loc['net'])

def test_to_pd_dataframe_halflinks():
    """Test writing a network to a pandas dataframe. Check the half links
    """
    # Given
    node0 = Node('n0')
    node1 = Node('n1')
    node1.x = 1
    node1.y = 2
    halflink = HalfLink('n1_hl',node1)
    link = Link('l0',node0,node1)
    net = Network('net')
    net.add_link(link)
    net.add_half_link(halflink)

    # When
    _, _, halflinks = to_pd_dataframes(net)

    # Then
    halflink_names_expected = ['n1_hl']
    carrier_halflinks_expected = ['no_type']
    start_nodes_expected = [('net',1)]

    halflinks_expected = pd.DataFrame({'name':halflink_names_expected,'carrier':carrier_halflinks_expected,'start_node':start_nodes_expected},columns=['name','carrier','start_node'])

    pd.testing.assert_frame_equal(halflinks_expected,halflinks.loc['net'])

def test_to_pd_dataframe_nodes_1sub():
    """Test writing a network to a pandas dataframe for a network with one subnetwork. Check the nodes
    """
    # Given
    subnode0 = Node('n0')
    subnode1 = Node('n1')
    subnode1.x = 1
    halflink_sn0 = HalfLink('n0_hl',subnode0)
    halflink_sn1 = HalfLink('n1_hl',subnode1)
    link_sl0 = Link('l0',subnode0,subnode1)
    subnet = Network('subnet')
    subnet.add_link(link_sl0)
    subnet.add_half_link(halflink_sn0)
    subnet.add_half_link(halflink_sn1)

    node0 = Node('n0')
    node0.x = -1
    halflink_n0 = HalfLink('n0_hl',node0)
    link = Link('l0',node0,subnode0)
    net = Network('net')
    net.add_node(node0)
    net.add_network(subnet)
    net.add_link(link)
    net.add_half_link(halflink_n0)

    # When
    nodes, _, _ = to_pd_dataframes(net)

    # Then
    node_names_expected = ['n0','n0','n1']
    carrier_nodes_expected = ['no_type']*3
    node_types_expected = ['no_type']*3
    x_expected = [-1,0,1]
    y_expected = [0,0,0]
    node_mis = [('net','no_net',0),('net','subnet',0),('net','subnet',1)]
    level_names = ['level 1','level 0',None]
    df_index = pd.MultiIndex.from_tuples(node_mis,names = level_names)
    nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected]).T,columns=['name','carrier','type','x','y'],index=df_index)
    pd.testing.assert_frame_equal(nodes_expected,nodes)

def test_to_pd_dataframe_links_1sub():
    """Test writing a network to a pandas dataframe for a network with one subnetwork. Check the links
    """
    # Given
    subnode0 = Node('n0')
    subnode1 = Node('n1')
    subnode1.x = 1
    halflink_sn0 = HalfLink('n0_hl',subnode0)
    halflink_sn1 = HalfLink('n1_hl',subnode1)
    link_sl0 = Link('l0',subnode0,subnode1)
    subnet = Network('subnet')
    subnet.add_link(link_sl0)
    subnet.add_half_link(halflink_sn0)
    subnet.add_half_link(halflink_sn1)

    node0 = Node('n0')
    node0.x = -1
    halflink_n0 = HalfLink('n0_hl',node0)
    link = Link('l0',node0,subnode0)
    net = Network('net')
    net.add_node(node0)
    net.add_network(subnet)
    net.add_link(link)
    net.add_half_link(halflink_n0)

    # When
    _, links, _ = to_pd_dataframes(net)

    # Then
    link_names_expected = ['l0','l0']
    carrier_links_expected = ['no_type']*2
    start_nodes_expected = [('net','subnet',0),('net','no_net',0)]
    end_nodes_expected = [('net','subnet',1),('net','subnet',0)]
    links_mis = [('net','subnet',0),('net','no_net',1)]
    level_names = ['level 1','level 0',None]
    df_index = pd.MultiIndex.from_tuples(links_mis,names = level_names)
    links_expected = pd.DataFrame({'name':link_names_expected,'carrier':carrier_links_expected,'start_node':start_nodes_expected,'end_node':end_nodes_expected},columns=['name','carrier','start_node','end_node'],index=df_index)
    pd.testing.assert_frame_equal(links_expected,links)

def test_to_pd_dataframe_halflinks_1sub():
    """Test writing a network to a pandas dataframe for a network with one subnetwork. Check the halflinks
    """
    # Given
    subnode0 = Node('n0')
    subnode1 = Node('n1')
    subnode1.x = 1
    halflink_sn0 = HalfLink('n0_hl',subnode0)
    halflink_sn1 = HalfLink('n1_hl',subnode1)
    link_sl0 = Link('l0',subnode0,subnode1)
    subnet = Network('subnet')
    subnet.add_link(link_sl0)
    subnet.add_half_link(halflink_sn0)
    subnet.add_half_link(halflink_sn1)

    node0 = Node('n0')
    node0.x = -1
    halflink_n0 = HalfLink('n0_hl',node0)
    link = Link('l0',node0,subnode0)
    net = Network('net')
    net.add_node(node0)
    net.add_network(subnet)
    net.add_link(link)
    net.add_half_link(halflink_n0)

    # When
    _, _, halflinks = to_pd_dataframes(net)

    # Then
    halflink_names_expected = ['n0_hl','n1_hl','n0_hl']
    carrier_halflinks_expected = ['no_type']*3
    start_nodes_expected = [('net','subnet',0),('net','subnet',1),('net','no_net',0)]
    halflinks_mis = [('net','subnet',0),('net','subnet',1),('net','no_net',2)]
    level_names = ['level 1','level 0',None]
    df_index = pd.MultiIndex.from_tuples(halflinks_mis,names = level_names)
    halflinks_expected = pd.DataFrame({'name':halflink_names_expected,'carrier':carrier_halflinks_expected,'start_node':start_nodes_expected},columns=['name','carrier','start_node'],index=df_index)
    pd.testing.assert_frame_equal(halflinks_expected,halflinks)

def create_net_2levels():
    """Create a test network with 2 hierarchical levels
    """
    # subsubnet 0
    ssnode0 = Node('n0')
    ssnode1 = Node('n1')
    sshalflink0 = HalfLink('n1_hl',ssnode1)
    sslink0 = Link('l0',ssnode0,ssnode1)
    subsubnet0 = Network('subsubnet 0')
    subsubnet0.add_link(sslink0)
    subsubnet0.add_half_link(sshalflink0)

    # subsubnet 1
    ssnode2 = Node('n2')
    ssnode3 = Node('n3')
    sslink1 = Link('l1',ssnode2,ssnode3)
    sshalflink1 = HalfLink('n3_hl',ssnode3)
    subsubnet1 = Network('subsubnet 1')
    subsubnet1.add_link(sslink1)
    subsubnet1.add_half_link(sshalflink1)

    # subnet
    snode0 = Node('n4')
    slink0 = Link('l2',snode0,ssnode0)
    slink1 = Link('l3',snode0,ssnode2)
    subnet = Network('subnet')
    subnet.add_node(snode0)
    subnet.add_network(subsubnet0)
    subnet.add_network(subsubnet1)
    subnet.add_link(slink0)
    subnet.add_link(slink1)

    # net
    node0 = Node('n5')
    halflink = HalfLink('n5_hl',node0)
    link = Link('l4',node0,snode0)
    net = Network('net')
    net.add_node(node0)
    net.add_network(subnet)
    net.add_link(link)
    net.add_half_link(halflink)
    return net

def test_to_pd_dataframe_nodes_2lev():
    """Test writing a network to a pandas dataframe for a network with 2 hierarchical levels. Check the nodes
    """
    # Given
    net = create_net_2levels()

    # When
    nodes, _, _ = to_pd_dataframes(net)

    # Then
    node_names_expected = ['n5','n4','n0','n1','n2','n3']
    carrier_nodes_expected = ['no_type']*6
    node_types_expected = ['no_type']*6
    x_expected = [0]*6
    y_expected = [0]*6
    node_mis = [('net','no_net','no_net',0),('net','subnet','no_net',0),('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',0),('net','subnet','subsubnet 1',1)]
    level_names = ['level 2','level 1','level 0',None]
    df_index = pd.MultiIndex.from_tuples(node_mis,names = level_names)
    nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected]).T,columns=['name','carrier','type','x','y'],index=df_index)
    pd.testing.assert_frame_equal(nodes_expected,nodes)

def test_to_pd_dataframe_links_2lev():
    """Test writing a network to a pandas dataframe for a network with 2 hierarchical levels. Check the links
    """
    # Given
    net = create_net_2levels()

    # When
    _, links, _ = to_pd_dataframes(net)

    # Then
    link_names_expected = ['l0','l1','l2','l3','l4']
    carrier_links_expected = ['no_type']*5
    start_nodes_expected = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',0),('net','subnet','no_net',0),('net','no_net','no_net',0)]
    end_nodes_expected = [('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',1),('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',0)]
    links_mis = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',2),('net','subnet','no_net',3),('net','no_net','no_net',4)]
    level_names = ['level 2','level 1','level 0',None]
    df_index = pd.MultiIndex.from_tuples(links_mis,names = level_names)
    links_expected = pd.DataFrame({'name':link_names_expected,'carrier':carrier_links_expected,'start_node':start_nodes_expected,'end_node':end_nodes_expected},columns=['name','carrier','start_node','end_node'],index=df_index)
    pd.testing.assert_frame_equal(links_expected,links)

def test_to_pd_dataframe_halflinks_2lev():
    """Test writing a network to a pandas dataframe for a network with 2 hierarchical levels. Check the halflinks
    """
    # Given
    net = create_net_2levels()

    # When
    _, _, halflinks = to_pd_dataframes(net)

    # Then
    halflink_names_expected = ['n1_hl','n3_hl','n5_hl']
    carrier_halflinks_expected = ['no_type']*3
    start_nodes_expected = [('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',1),('net','no_net','no_net',0)]
    halflinks_mis = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','no_net','no_net',2)]
    level_names = ['level 2','level 1','level 0',None]
    df_index = pd.MultiIndex.from_tuples(halflinks_mis,names = level_names)
    halflinks_expected = pd.DataFrame({'name':halflink_names_expected,'carrier':carrier_halflinks_expected,'start_node':start_nodes_expected},columns=['name','carrier','start_node'],index=df_index)
    pd.testing.assert_frame_equal(halflinks_expected,halflinks)

def create_gas_net_with_data():
    """Creates one gas network with data"""
    n = 5
    m = 2
    carrier = 'g'

    net = create_radial_line_network(n,carrier=carrier,m=m)
    number_of_nodes = len(net.nodes)

    # node data
    mbar = 1e2 #[Pa]
    p_ref = 50*mbar #[Pa]
    np.random.seed(0)
    q_min = 0.02/n #[kg/s]
    q_var = 0.02/n #[kg/s]
    q_inj_loads = np.random.rand(n)*q_var + q_min

    # link data
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('gas',S,R_air,1,pn,Tn,Tn)
    L = 500 #[m]
    L_unit = 'm'
    D = .01 #[m]
    D_unit = 'm'
    number_of_links = len(net.links)
    link_types = ['pipe_low_pres_pole']*number_of_links
    link_params = [{'carrier':gas, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit}]*number_of_links

    add_gas_data(net,p_ref,q_inj_loads,link_types,link_params)
    net.initialize()
    return net

def test_gas_node_data_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the node data."""
    # Given
    net = create_gas_net_with_data()
    number_of_nodes = len(net.nodes)

    # When
    nodes, _, _ = to_pd_dataframes(net)

    # Then
    node_names_expected = ['source','gn0','gn1','gn2','gn3','gn4','gn5','gn6','gn7']
    carrier_nodes_expected = ['g']*number_of_nodes
    node_types_expected = [0,1,1,1,1,1,1,1,1]
    x_expected = [-1,0,0,1,1,2,0,1,2]
    y_expected = [1,0,2,0,2,0,1,1,1]
    node_data = list()
    for n in net.get_nodes():
        node_data.append({'p':n.p,'p_unit':'Pa'})
    node_mis = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5),('S0',6),('S0',7),('S0',8)]
    level_names = ['level 0',None]
    df_index = pd.MultiIndex.from_tuples(node_mis,names = level_names)
    nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected,node_data]).T,columns=['name','carrier','type','x','y','data'],index=df_index)

    pd.testing.assert_frame_equal(nodes_expected,nodes)

def test_gas_link_data_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the link data."""
    # Given
    net = create_gas_net_with_data()
    number_of_links = len(net.links)

    # When
    _, links, _ = to_pd_dataframes(net)

    # Then
    link_names_expected = ['gl0','gl1','gl2','gl3','gl4','gl5','gl6','gl7']
    carrier_links_expected = ['g']*number_of_links
    start_nodes_expected = [('S0',0),('S0',6),('S0',7),('S0',6),('S0',6),('S0',7),('S0',7),('S0',8)]
    end_nodes_expected = [('S0',6),('S0',7),('S0',8),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5)]
    link_hydr_eq_form_expected = ['q_of_dp']*number_of_links
    link_types = list()
    link_params = list()
    link_data = list()
    link_bc_types_expected = np.zeros(number_of_links)
    for e in net.get_links():
        link_types.append(e.link_type)
        link_params.append(e.link_params)
        link_data.append({'q':e.q,'q_unit':'kg/s'})
    links_mis = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5),('S0',6),('S0',7)]
    level_names = ['level 0',None]
    df_index = pd.MultiIndex.from_tuples(links_mis,names = level_names)
    links_expected = pd.DataFrame(np.array([link_names_expected,carrier_links_expected,start_nodes_expected,end_nodes_expected,link_types,link_params,link_hydr_eq_form_expected,link_bc_types_expected,link_data]).T,columns=['name','carrier','start_node','end_node','type','parameters','hydr_eq_form','bc_type','data'],index=df_index)
    links_expected = links_expected.astype({'bc_type':'int64'}) # somehow, the bc types are stored as ints and not as objects

    pd.testing.assert_frame_equal(links_expected,links)

def test_gas_halflink_data_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the halflink data."""
    # Given
    net = create_gas_net_with_data()
    number_of_halflinks = len(net.half_links)

    # When
    _, _, halflinks= to_pd_dataframes(net)

    # Then
    halflink_names_expected = ['source_hl','gn0_hl','gn1_hl','gn2_hl','gn3_hl','gn4_hl','gn5_hl','gn6_hl','gn7_hl']
    carrier_halflinks_expected = ['g']*number_of_halflinks
    start_nodes_halflinks_expected = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5),('S0',6),('S0',7),('S0',8)]
    halflink_types_expected = ['flow']*number_of_halflinks
    halflink_params_expected = [{}]*number_of_halflinks
    halflinks_mis = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5),('S0',6),('S0',7),('S0',8)]
    halflink_data_expected = list()
    halflink_bc_types_expected = list()
    for hl in net.get_half_links():
        halflink_data_expected.append({'q':hl.q,'q_unit':'kg/s'})
        halflink_bc_types_expected.append(hl.bc_type)
    level_names = ['level 0',None]
    df_index = pd.MultiIndex.from_tuples(halflinks_mis,names = level_names)
    halflinks_expected = pd.DataFrame({'name':halflink_names_expected,'carrier':carrier_halflinks_expected,'type':halflink_types_expected,'start_node':start_nodes_halflinks_expected,'parameters':halflink_params_expected,'bc_type':halflink_bc_types_expected,'data':halflink_data_expected},columns=['name','carrier','start_node','type','parameters','bc_type','data'],index=df_index)

    pd.testing.assert_frame_equal(halflinks_expected,halflinks)

def create_elec_net_with_data():
    """Creates one electrical network with data"""
    n = 5
    m = 2
    carrier = 'e'

    net = create_radial_line_network(n,carrier=carrier,m=m)
    number_of_nodes = len(net.nodes)

    # node data
    V_ref = 1. #[p.u.]
    delta_ref = 0. #[rad]
    np.random.seed(0)
    S_min = 0.5 #[kW]? [p.u.]?
    S_var = 1. #[kW]? [p.u.]?
    P_inj = np.random.rand(n)*S_var + S_min
    Q_inj = np.random.rand(n)*S_var + S_min

    # link data
    b = -10. # [p.u.]
    g = 1. # [p.u.]
    number_of_links = len(net.links)
    link_types = ['short_line']*number_of_links
    link_params = [{'b':b,'g':g}]*number_of_links

    add_elec_data(net,V_ref,delta_ref,P_inj,Q_inj,link_types,link_params)
    net.initialize()
    return net

def test_elec_node_data_double_load_n_odd():
    """Test the addition of electrical data to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the node data."""
    # Given
    net = create_elec_net_with_data()
    number_of_nodes = len(net.nodes)

    # When
    nodes, _, _ = to_pd_dataframes(net)

    # Then
    node_names_expected = ['source','en0','en1','en2','en3','en4','en5','en6','en7']
    carrier_nodes_expected = ['e']*number_of_nodes
    node_types_expected = [0,2,2,2,2,2,2,2,2]
    x_expected = [-1,0,0,1,1,2,0,1,2]
    y_expected = [1,0,2,0,2,0,1,1,1]
    node_data = list()
    for n in net.get_nodes():
        node_data.append({'V':n.V,'V_unit':'p.u.','delta':n.delta,'delta_unit':'rad'})
    node_mis = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5),('S0',6),('S0',7),('S0',8)]
    level_names = ['level 0',None]
    df_index = pd.MultiIndex.from_tuples(node_mis,names = level_names)
    nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected,node_data]).T,columns=['name','carrier','type','x','y','data'],index=df_index)

    pd.testing.assert_frame_equal(nodes_expected,nodes)

def test_elec_link_data_double_load_n_odd():
    """Test the addition of electrical data to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the link data."""
    # Given
    net = create_elec_net_with_data()
    number_of_links = len(net.links)

    # When
    _, links, _ = to_pd_dataframes(net)

    # Then
    link_names_expected = ['el0','el1','el2','el3','el4','el5','el6','el7']
    carrier_links_expected = ['e']*number_of_links
    start_nodes_expected = [('S0',0),('S0',6),('S0',7),('S0',6),('S0',6),('S0',7),('S0',7),('S0',8)]
    end_nodes_expected = [('S0',6),('S0',7),('S0',8),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5)]
    link_hydr_eq_form_expected = [None]*number_of_links
    link_types = list()
    link_params = list()
    link_data = list()
    link_bc_types_expected = [0]*number_of_links
    for e in net.get_links():
        link_types.append(e.link_type)
        link_params.append(e.link_params)
        link_data.append({'Pstart':e.Pstart,'Qstart':e.Qstart,'Pend':e.Pend,'Qend':e.Qend,'P_unit':'p.u.','Q_unit':'p.u.'})
    links_mis = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5),('S0',6),('S0',7)]
    level_names = ['level 0',None]
    df_index = pd.MultiIndex.from_tuples(links_mis,names = level_names)
    links_expected = pd.DataFrame(np.array([link_names_expected,carrier_links_expected,start_nodes_expected,end_nodes_expected,link_types,link_params,link_hydr_eq_form_expected,link_bc_types_expected,link_data]).T,columns=['name','carrier','start_node','end_node','type','parameters','hydr_eq_form','bc_type','data'],index=df_index)
    links_expected = links_expected.astype({'bc_type':'int64'}) # somehow, the bc types are stored as ints and not as objects
    
    pd.testing.assert_frame_equal(links_expected,links)

def test_elec_halflink_data_double_load_n_odd():
    """Test the addition of electrical data to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the halflink data."""
    # Given
    net = create_elec_net_with_data()
    number_of_halflinks = len(net.half_links)

    # When
    _, _, halflinks= to_pd_dataframes(net)

    # Then
    halflink_names_expected = ['source_hl','en0_hl','en1_hl','en2_hl','en3_hl','en4_hl','en5_hl','en6_hl','en7_hl']
    carrier_halflinks_expected = ['e']*number_of_halflinks
    start_nodes_halflinks_expected = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5),('S0',6),('S0',7),('S0',8)]
    halflink_types_expected = ['flow']*number_of_halflinks
    halflink_params_expected = [{}]*number_of_halflinks
    halflinks_mis = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5),('S0',6),('S0',7),('S0',8)]
    halflink_data_expected = list()
    halflink_bc_types_expected = [0]*number_of_halflinks
    for hl in net.get_half_links():
        halflink_data_expected.append({'P':hl.P,'Q':hl.Q,'P_unit':'p.u.','Q_unit':'p.u.'})
    level_names = ['level 0',None]
    df_index = pd.MultiIndex.from_tuples(halflinks_mis,names = level_names)
    halflinks_expected = pd.DataFrame({'name':halflink_names_expected,'carrier':carrier_halflinks_expected,'type':halflink_types_expected,'start_node':start_nodes_halflinks_expected,'parameters':halflink_params_expected,'bc_type':halflink_bc_types_expected,'data':halflink_data_expected},columns=['name','carrier','start_node','type','parameters','bc_type','data'],index=df_index)

    pd.testing.assert_frame_equal(halflinks_expected,halflinks)

def create_heat_net_with_data():
    """Creates one heat network with data"""
    n = 5
    m = 2
    carrier = 'h'

    net = create_radial_line_network(n,carrier=carrier,m=m)
    number_of_nodes = len(net.nodes)

    # carrier
    rho = 960. #[kg/m^3]
    Cp = 4.182e3 #[J/(kg K)]
    mu = 0.294e-6 #[m^2/s]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp,rho=rho,mu=mu)
    Ta = 0. #[C]

    # node data
    h_ref = 100. #[m]
    p_ref = h_ref*rho*g #[Pa]
    Ts_ref = 120. #[C]
    # link data
    L = 500 #[m]
    L_unit = 'm'
    D = .14 #[m]
    D_unit = 'm'
    eps = 1.25*1e-3 #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D) #[W/(m^2 K)]
    # half link data
    Tr_hl = 50.*np.ones(n+1) #[C]
    Ts_hl = Ts_ref*np.ones(n+1) #source node
    dT = 50.*np.ones(n+1) #[C]
    halflink_bc_types = [2]+[3]*n # one source and the rest sinks
    np.random.seed(0)
    phi_min = 50. #[W]
    phi_var = 450. #[W]
    phi_inj = np.zeros(n+1)
    phi_inj[1:] = np.random.rand(n)*phi_var + phi_min
    halflink_types = ['heat_exchanger']*(n+1)
    halflink_params = [{'carrier':water}]*(n+1)

    # link data
    number_of_links = len(net.links)
    link_types = ['standard_pipe_low_pres_colebrook']*number_of_links
    link_params = [{'L':L,'L_unit':L_unit,'D':D,'D_unit':D_unit,'eps':eps,'U':U,'carrier':carrier}]*number_of_links

    add_heat_data(net,p_ref,Ts_ref,phi_inj,Ts_hl,Tr_hl,dT,link_types,link_params,halflink_types,halflink_params,halflink_bc_types)
    net.initialize()
    return net

def test_heat_node_data_double_load_n_odd():
    """Test the addition of heat data to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the node data."""
    # Given
    net = create_heat_net_with_data()
    number_of_nodes = len(net.nodes)

    # When
    nodes, _, _ = to_pd_dataframes(net)

    # Then
    node_names_expected = ['source','hn0','hn1','hn2','hn3','hn4','hn5','hn6','hn7']
    carrier_nodes_expected = ['h']*number_of_nodes
    node_types_expected = [0,1,1,1,1,1,2,2,2]
    x_expected = [-1,0,0,1,1,2,0,1,2]
    y_expected = [1,0,2,0,2,0,1,1,1]
    node_data = list()
    for n in net.get_nodes():
        node_data.append({'p':n.p,'p_unit':'Pa','Ts':n.Ts,'Ts_unit':'C','Tr':n.Tr,'Tr_unit':'C'})
    node_mis = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5),('S0',6),('S0',7),('S0',8)]
    level_names = ['level 0',None]
    df_index = pd.MultiIndex.from_tuples(node_mis,names = level_names)
    nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected,node_data]).T,columns=['name','carrier','type','x','y','data'],index=df_index)

    pd.testing.assert_frame_equal(nodes_expected,nodes)

def test_heat_link_data_double_load_n_odd():
    """Test the addition of heat data to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the link data."""
    # Given
    net = create_heat_net_with_data()
    number_of_links = len(net.links)

    # When
    _, links, _ = to_pd_dataframes(net)

    # Then
    link_names_expected = ['hl0','hl1','hl2','hl3','hl4','hl5','hl6','hl7']
    carrier_links_expected = ['h']*number_of_links
    start_nodes_expected = [('S0',0),('S0',6),('S0',7),('S0',6),('S0',6),('S0',7),('S0',7),('S0',8)]
    end_nodes_expected = [('S0',6),('S0',7),('S0',8),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5)]
    link_hydr_eq_form_expected = ['dp_of_q']*number_of_links
    link_types = list()
    link_params = list()
    link_data = list()
    link_bc_types_expected = np.zeros(number_of_links)
    for e in net.get_links():
        link_types.append(e.link_type)
        link_params.append(e.link_params)
        link_data.append({'m':e.m,'m_unit':'kg/s','Tsstart':e.Tsstart,'Trstart':e.Trstart,'Tsend':e.Tsend,'Trend':e.Trend,'dphistart':e.dphistart,'dphiend':e.dphiend,'Ts_unit':'C','Tr_unit':'C','phi_unit':'W'})
    links_mis = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5),('S0',6),('S0',7)]
    level_names = ['level 0',None]
    df_index = pd.MultiIndex.from_tuples(links_mis,names = level_names)
    links_expected = pd.DataFrame(np.array([link_names_expected,carrier_links_expected,start_nodes_expected,end_nodes_expected,link_types,link_params,link_hydr_eq_form_expected,link_bc_types_expected,link_data]).T,columns=['name','carrier','start_node','end_node','type','parameters','hydr_eq_form','bc_type','data'],index=df_index)
    links_expected = links_expected.astype({'bc_type':'int64'}) # somehow, the bc types are stored as ints and not as objects

    pd.testing.assert_frame_equal(links_expected,links)

def test_heat_halflink_data_double_load_n_odd():
    """Test the addition of heat data to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the halflink data."""
    # Given
    net = create_heat_net_with_data()
    number_of_halflinks = len(net.half_links)

    # When
    _, _, halflinks= to_pd_dataframes(net)

    # Then
    halflink_names_expected = ['source_hl','hn0_hl','hn1_hl','hn2_hl','hn3_hl','hn4_hl']
    carrier_halflinks_expected = ['h']*number_of_halflinks
    start_nodes_halflinks_expected = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5)]
    halflink_types_expected = ['heat_exchanger']*number_of_halflinks
    halflinks_mis = [('S0',0),('S0',1),('S0',2),('S0',3),('S0',4),('S0',5)]
    halflink_data_expected = list()
    halflink_params_expected= list()
    halflink_bc_types_expected = list()
    for hl in net.get_half_links():
        halflink_data_expected.append({'m':hl.m,'m_unit':'kg/s','Ts':hl.Ts,'Tr':hl.Tr,'dT':hl.dT,'phi':hl.dphi,'T_unit':'C','phi_unit':'W'})
        halflink_params_expected.append(hl.link_params)
        halflink_bc_types_expected.append(hl.bc_type)
    level_names = ['level 0',None]
    df_index = pd.MultiIndex.from_tuples(halflinks_mis,names = level_names)

    halflinks_expected = pd.DataFrame({'name':halflink_names_expected,'carrier':carrier_halflinks_expected,'type':halflink_types_expected,'start_node':start_nodes_halflinks_expected,'parameters':halflink_params_expected,'bc_type':halflink_bc_types_expected,'data':halflink_data_expected},columns=['name','carrier','start_node','type','parameters','bc_type','data'],index=df_index)

    pd.testing.assert_frame_equal(halflinks_expected,halflinks)

# ===================================================================================
# Test creation of network from a pandas dataframe
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_nodes():
    """Test reading a network to a pandas dataframe for a network without subnetworks. Check if writing gives back the input dataframe for the nodes
    """
    # Given
    level_names = ['level 0',None]

    node_names_input = ['n0','n1']
    carrier_nodes_input = ['no_type']*2
    node_types_input = ['no_type']*2
    x_input = [0,1]
    y_input = [0,2]
    nodes_mis = [('net',0),('net',1)]
    nodes_index = pd.MultiIndex.from_tuples(nodes_mis,names = level_names)
    nodes_input = pd.DataFrame(np.array([node_names_input,carrier_nodes_input,node_types_input,x_input,y_input]).T,columns=['name','carrier','type','x','y'],index=nodes_index)

    link_names_input = ['l0']
    carrier_links_input = ['no_type']
    start_nodes_input = [('net',0)]
    end_nodes_input = [('net',1)]
    links_mis = [('net',0)]
    links_index = pd.MultiIndex.from_tuples(links_mis,names = level_names)
    links_input = pd.DataFrame({'name':link_names_input,'carrier':carrier_links_input,'start_node':start_nodes_input,'end_node':end_nodes_input},columns=['name','carrier','start_node','end_node'],index=links_index)

    halflink_names_input = ['n1_hl']
    carrier_halflinks_input = ['no_type']
    start_nodes_input = [('net',1)]
    halflinks_mis = [('net',0)]
    halflinks_index = pd.MultiIndex.from_tuples(halflinks_mis,names = level_names)
    halflinks_input = pd.DataFrame({'name':halflink_names_input,'carrier':carrier_halflinks_input,'start_node':start_nodes_input},columns=['name','carrier','start_node'],index=halflinks_index)

    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # When
    nodes, links, halflinks = to_pd_dataframes(net)

    # Then
    pd.testing.assert_frame_equal(nodes_input,nodes)
    
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_links():
    """Test reading a network to a pandas dataframe for a network without subnetworks. Check if writing gives back the input dataframe for the links
    """
    # Given
    level_names = ['level 0',None]

    node_names_input = ['n0','n1']
    carrier_nodes_input = ['no_type']*2
    node_types_input = ['no_type']*2
    x_input = [0,1]
    y_input = [0,2]
    nodes_mis = [('net',0),('net',1)]
    nodes_index = pd.MultiIndex.from_tuples(nodes_mis,names = level_names)
    nodes_input = pd.DataFrame(np.array([node_names_input,carrier_nodes_input,node_types_input,x_input,y_input]).T,columns=['name','carrier','type','x','y'],index=nodes_index)

    link_names_input = ['l0']
    carrier_links_input = ['no_type']
    start_nodes_input = [('net',0)]
    end_nodes_input = [('net',1)]
    links_mis = [('net',0)]
    links_index = pd.MultiIndex.from_tuples(links_mis,names = level_names)
    links_input = pd.DataFrame({'name':link_names_input,'carrier':carrier_links_input,'start_node':start_nodes_input,'end_node':end_nodes_input},columns=['name','carrier','start_node','end_node'],index=links_index)

    halflink_names_input = ['n1_hl']
    carrier_halflinks_input = ['no_type']
    start_nodes_input = [('net',1)]
    halflinks_mis = [('net',0)]
    halflinks_index = pd.MultiIndex.from_tuples(halflinks_mis,names = level_names)
    halflinks_input = pd.DataFrame({'name':halflink_names_input,'carrier':carrier_halflinks_input,'start_node':start_nodes_input},columns=['name','carrier','start_node'],index=halflinks_index)

    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # When
    nodes, links, halflinks = to_pd_dataframes(net)

    # Then
    pd.testing.assert_frame_equal(links_input,links)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_halflinks():
    """Test reading a network to a pandas dataframe for a network without subnetworks. Check if writing gives back the input dataframe for the halflinks
    """
    # Given
    level_names = ['level 0',None]

    node_names_input = ['n0','n1']
    carrier_nodes_input = ['no_type']*2
    node_types_input = ['no_type']*2
    x_input = [0,1]
    y_input = [0,2]
    nodes_mis = [('net',0),('net',1)]
    nodes_index = pd.MultiIndex.from_tuples(nodes_mis,names = level_names)
    nodes_input = pd.DataFrame(np.array([node_names_input,carrier_nodes_input,node_types_input,x_input,y_input]).T,columns=['name','carrier','type','x','y'],index=nodes_index)

    link_names_input = ['l0']
    carrier_links_input = ['no_type']
    start_nodes_input = [('net',0)]
    end_nodes_input = [('net',1)]
    links_mis = [('net',0)]
    links_index = pd.MultiIndex.from_tuples(links_mis,names = level_names)
    links_input = pd.DataFrame({'name':link_names_input,'carrier':carrier_links_input,'start_node':start_nodes_input,'end_node':end_nodes_input},columns=['name','carrier','start_node','end_node'],index=links_index)

    halflink_names_input = ['n1_hl']
    carrier_halflinks_input = ['no_type']
    start_nodes_input = [('net',1)]
    halflinks_mis = [('net',0)]
    halflinks_index = pd.MultiIndex.from_tuples(halflinks_mis,names = level_names)
    halflinks_input = pd.DataFrame({'name':halflink_names_input,'carrier':carrier_halflinks_input,'start_node':start_nodes_input},columns=['name','carrier','start_node'],index=halflinks_index)

    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # When
    nodes, links, halflinks = to_pd_dataframes(net)

    # Then
    pd.testing.assert_frame_equal(halflinks_input,halflinks)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_nodes_2lev():
    """Test reading a network to a pandas dataframe for a network with 2 hierarchical levels. Check if writing gives back the input dataframe for the nodes
    """
    # Given
    node_names_input = ['n5','n4','n0','n1','n2','n3']
    carrier_nodes_input = ['no_type']*6
    node_types_input = ['no_type']*6
    x_input = [0]*6
    y_input = [0]*6
    node_mis = [('net','no_net','no_net',0),('net','subnet','no_net',0),('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',0),('net','subnet','subsubnet 1',1)]
    level_names = ['level 2','level 1','level 0',None]
    nodes_index = pd.MultiIndex.from_tuples(node_mis,names = level_names)
    nodes_input = pd.DataFrame(np.array([node_names_input,carrier_nodes_input,node_types_input,x_input,y_input]).T,columns=['name','carrier','type','x','y'],index=nodes_index)

    link_names_input = ['l0','l1','l2','l3','l4']
    carrier_links_input = ['no_type']*5
    start_nodes_input = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',0),('net','subnet','no_net',0),('net','no_net','no_net',0)]
    end_nodes_input = [('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',1),('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',0)]
    links_mis = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',2),('net','subnet','no_net',3),('net','no_net','no_net',4)]
    level_names = ['level 2','level 1','level 0',None]
    links_index = pd.MultiIndex.from_tuples(links_mis,names = level_names)
    links_input = pd.DataFrame({'name':link_names_input,'carrier':carrier_links_input,'start_node':start_nodes_input,'end_node':end_nodes_input},columns=['name','carrier','start_node','end_node'],index=links_index)

    halflink_names_input = ['n1_hl','n3_hl','n5_hl']
    carrier_halflinks_input = ['no_type']*3
    start_nodes_input = [('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',1),('net','no_net','no_net',0)]
    halflinks_mis = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','no_net','no_net',2)]
    level_names = ['level 2','level 1','level 0',None]
    halflinks_index = pd.MultiIndex.from_tuples(halflinks_mis,names = level_names)
    halflinks_input = pd.DataFrame({'name':halflink_names_input,'carrier':carrier_halflinks_input,'start_node':start_nodes_input},columns=['name','carrier','start_node'],index=halflinks_index)

    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # When
    nodes, links, halflinks = to_pd_dataframes(net)

    print('It should be:')
    print(nodes_input)
    print('It is:')
    print(nodes)
    # Then
    pd.testing.assert_frame_equal(nodes_input,nodes)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_links_2lev():
    """Test reading a network to a pandas dataframe for a network with 2 hierarchical levels. Check if writing gives back the input dataframe for the links
    """
    # Given
    node_names_input = ['n5','n4','n0','n1','n2','n3']
    carrier_nodes_input = ['no_type']*6
    node_types_input = ['no_type']*6
    x_input = [0]*6
    y_input = [0]*6
    node_mis = [('net','no_net','no_net',0),('net','subnet','no_net',0),('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',0),('net','subnet','subsubnet 1',1)]
    level_names = ['level 2','level 1','level 0',None]
    nodes_index = pd.MultiIndex.from_tuples(node_mis,names = level_names)
    nodes_input = pd.DataFrame(np.array([node_names_input,carrier_nodes_input,node_types_input,x_input,y_input]).T,columns=['name','carrier','type','x','y'],index=nodes_index)

    link_names_input = ['l0','l1','l2','l3','l4']
    carrier_links_input = ['no_type']*5
    start_nodes_input = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',0),('net','subnet','no_net',0),('net','no_net','no_net',0)]
    end_nodes_input = [('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',1),('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',0)]
    links_mis = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',2),('net','subnet','no_net',3),('net','no_net','no_net',4)]
    level_names = ['level 2','level 1','level 0',None]
    links_index = pd.MultiIndex.from_tuples(links_mis,names = level_names)
    links_input = pd.DataFrame({'name':link_names_input,'carrier':carrier_links_input,'start_node':start_nodes_input,'end_node':end_nodes_input},columns=['name','carrier','start_node','end_node'],index=links_index)

    halflink_names_input = ['n1_hl','n3_hl','n5_hl']
    carrier_halflinks_input = ['no_type']*3
    start_nodes_input = [('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',1),('net','no_net','no_net',0)]
    halflinks_mis = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','no_net','no_net',2)]
    level_names = ['level 2','level 1','level 0',None]
    halflinks_index = pd.MultiIndex.from_tuples(halflinks_mis,names = level_names)
    halflinks_input = pd.DataFrame({'name':halflink_names_input,'carrier':carrier_halflinks_input,'start_node':start_nodes_input},columns=['name','carrier','start_node'],index=halflinks_index)

    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # When
    nodes, links, halflinks = to_pd_dataframes(net)

    # Then
    pd.testing.assert_frame_equal(links_input,links)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_halflinks_2lev():
    """Test reading a network to a pandas dataframe for a network with 2 hierarchical levels. Check if writing gives back the input dataframe for the halflinks
    """
    # Given
    node_names_input = ['n5','n4','n0','n1','n2','n3']
    carrier_nodes_input = ['no_type']*6
    node_types_input = ['no_type']*6
    x_input = [0]*6
    y_input = [0]*6
    node_mis = [('net','no_net','no_net',0),('net','subnet','no_net',0),('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',0),('net','subnet','subsubnet 1',1)]
    level_names = ['level 2','level 1','level 0',None]
    nodes_index = pd.MultiIndex.from_tuples(node_mis,names = level_names)
    nodes_input = pd.DataFrame(np.array([node_names_input,carrier_nodes_input,node_types_input,x_input,y_input]).T,columns=['name','carrier','type','x','y'],index=nodes_index)

    link_names_input = ['l0','l1','l2','l3','l4']
    carrier_links_input = ['no_type']*5
    start_nodes_input = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',0),('net','subnet','no_net',0),('net','no_net','no_net',0)]
    end_nodes_input = [('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',1),('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',0)]
    links_mis = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','subnet','no_net',2),('net','subnet','no_net',3),('net','no_net','no_net',4)]
    level_names = ['level 2','level 1','level 0',None]
    links_index = pd.MultiIndex.from_tuples(links_mis,names = level_names)
    links_input = pd.DataFrame({'name':link_names_input,'carrier':carrier_links_input,'start_node':start_nodes_input,'end_node':end_nodes_input},columns=['name','carrier','start_node','end_node'],index=links_index)

    halflink_names_input = ['n1_hl','n3_hl','n5_hl']
    carrier_halflinks_input = ['no_type']*3
    start_nodes_input = [('net','subnet','subsubnet 0',1),('net','subnet','subsubnet 1',1),('net','no_net','no_net',0)]
    halflinks_mis = [('net','subnet','subsubnet 0',0),('net','subnet','subsubnet 1',0),('net','no_net','no_net',2)]
    level_names = ['level 2','level 1','level 0',None]
    halflinks_index = pd.MultiIndex.from_tuples(halflinks_mis,names = level_names)
    halflinks_input = pd.DataFrame({'name':halflink_names_input,'carrier':carrier_halflinks_input,'start_node':start_nodes_input},columns=['name','carrier','start_node'],index=halflinks_index)

    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # When
    nodes, links, halflinks = to_pd_dataframes(net)

    # Then
    pd.testing.assert_frame_equal(halflinks_input,halflinks)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_gas_node_data():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the node data."""
    # Given
    nodes_input, links_input, halflinks_input = to_pd_dataframes(create_gas_net_with_data())

    # When
    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # Then
    p_expected = [node_data.get('p') for node_data in nodes_input['data']]

    p = list()
    for n in net.get_nodes():
        p.append(n.p)

    assert np.all(p_expected == p)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_gas_link_data_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the link data."""
    # Given
    nodes_input, links_input, halflinks_input = to_pd_dataframes(create_gas_net_with_data())

    # When
    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # Then
    link_types_expected = ['pipe_low_pres_pole']*len(net.links)
    new_link_types = list()
    for e in net.get_links():
        new_link_types.append(e.link_type)
    assert np.all(link_types_expected == new_link_types)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_gas_halflink_data_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the halflink data."""
    # Given
    nodes_input, links_input, halflinks_input = to_pd_dataframes(create_gas_net_with_data())

    # When
    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # Then
    q_inj_expected = [hl_data.get('q') for hl_data in halflinks_input['data']]
    q_inj = list()
    for node in net.get_nodes():
        for hl in node.get_half_links():
            q_inj.append(hl.q)

    assert np.all(q_inj_expected == q_inj)
    
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframes_gas_1street():
    """Test creation of network from pandas dataframes
    """
    # Given
    path_to_data = './examples/network_data/GN_1street'
    nodes = pd.read_pickle(os.path.join(path_to_data, 'GN_1street_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'GN_1street_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'GN_1street_halflinks.pkl'))

    # When
    gas_net = from_pd_dataframes(nodes,links,halflinks,flatten=False)
    gas_net.initialize()

    # Then
    A_test = np.array([[-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],
            [1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,1.,-1.,0.,0.,0.,0.,0.,0.,-1.,-1.,0.,0.,0.,0.,0.,0.],
            [0.,0.,1.,-1.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,0.,0.,0.,0.],
            [0.,0.,0.,1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.,0.],
            [0.,0.,0.,0.,1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.],
            [0.,0.,0.,0.,0.,1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.],
            [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.]])
    assert np.all(A_test == gas_net.A)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframes_nodes_gas_2streets():
    """Test creation of network from pandas dataframes, where the network consists of subnetworks. Check if writing gives back the input dataframe for the nodes
    """
    # Given
    path_to_data = './examples/network_data/GN_2streets'
    nodes_input = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_nodes.pkl'))
    links_input = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_links.pkl'))
    halflinks_input = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_halflinks.pkl'))
    gas_net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)
    gas_net.initialize()

    # When
    nodes, _, _ = to_pd_dataframes(gas_net)

    # Then
    print('It should be:')
    print(nodes_input['name'])
    print('It is:')
    print(nodes['name'])

    pd.testing.assert_frame_equal(nodes_input,nodes)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframes_links_gas_2streets():
    """Test creation of network from pandas dataframes, where the network consists of subnetworks. Check if writing gives back the input dataframe for the links
    """
    # Given
    path_to_data = './examples/network_data/GN_2streets'
    nodes_input = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_nodes.pkl'))
    links_input = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_links.pkl'))
    halflinks_input = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_halflinks.pkl'))
    gas_net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)
    gas_net.initialize()

    # When
    _, links, _ = to_pd_dataframes(gas_net)

    # Then

    pd.testing.assert_frame_equal(links_input,links)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframes_links_gas_2streets():
    """Test creation of network from pandas dataframes, where the network consists of subnetworks. Check if writing gives back the input dataframe for the links
    """
    # Given
    path_to_data = './examples/network_data/GN_2streets'
    nodes_input = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_nodes.pkl'))
    links_input = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_links.pkl'))
    halflinks_input = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_halflinks.pkl'))
    gas_net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)
    gas_net.initialize()

    # When
    _, _, halflinks = to_pd_dataframes(gas_net)

    # Then

    pd.testing.assert_frame_equal(halflinks_input,halflinks)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframes_gas_2streets():
    """Test creation of network from pandas dataframes, where the network consists of subnetworks.
    """
    # Given
    path_to_data = './examples/network_data/GN_2streets'
    nodes = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_halflinks.pkl'))


    # When
    gas_net = from_pd_dataframes(nodes,links,halflinks,flatten=False)
    gas_net.initialize()

    # Then
    A_test = np.array([[0., 0., 0., 0., 0., 0., -1.,  0., 0., 0., -1.],
                    [-1., 0.,  0.,  0.,  0.,  0., 1.,   0.,  0.,  0., 0.],
                    [0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                    [1., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  1., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0., 1.],
                    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.],
                    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]])

    assert np.all(A_test == gas_net.A)

def test_from_pd_dataframes_gas_2streets_flatten():
    """Test creation of network from pandas dataframes, where the network is flat.
    """
    # Given
    path_to_data = './examples/network_data/GN_2streets'
    nodes = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'GN_2streets_halflinks.pkl'))

    print(nodes)
    print(links)
    # When
    gas_net = from_pd_dataframes(nodes,links,halflinks)
    gas_net.initialize()

    # Then
    A_test = np.array([[0., 0., 0., 0., 0., 0., -1.,  0., 0., 0., -1.],
                    [-1., 0.,  0.,  0.,  0.,  0., 1.,   0.,  0.,  0., 0.],
                    [0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                    [1., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  1., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0., 1.],
                    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.],
                    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]])
    print('It should be: {}'.format(A_test))
    print('It is: {}'.format(gas_net.A.todense()))
    assert np.all(A_test == gas_net.A)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_elec_node_data():
    """Test reading an electrical network, with data, from a pd dataframe. Check the node data"""
    # Given
    nodes_input, links_input, halflinks_input = to_pd_dataframes(create_elec_net_with_data())

    # When
    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # Then
    V_expected = [node_data.get('V') for node_data in nodes_input['data']]
    delta_expected = [node_data.get('delta') for node_data in nodes_input['data']]

    V = list()
    delta = list()
    for n in net.get_nodes():
        V.append(n.V)
        delta.append(n.delta)

    assert np.all(V_expected+delta_expected == V+delta)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_elec_link_data_double_load_n_odd():
    """Test reading an electrical network, with data, from a pd dataframe. Check the link data."""
    # Given
    nodes_input, links_input, halflinks_input = to_pd_dataframes(create_elec_net_with_data())

    # When
    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # Then
    link_types_expected = ['short_line']*len(net.links)
    new_link_types = list()
    for e in net.get_links():
        new_link_types.append(e.link_type)
    assert np.all(link_types_expected == new_link_types)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_elec_halflink_data_double_load_n_odd():
    """Test reading an electrical network, with data, from a pd dataframe. Check the halflink data."""
    # Given
    nodes_input, links_input, halflinks_input = to_pd_dataframes(create_elec_net_with_data())

    # When
    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # Then
    P_inj_expected = [hl_data.get('P') for hl_data in halflinks_input['data']]
    Q_inj_expected = [hl_data.get('Q') for hl_data in halflinks_input['data']]
    P_inj = list()
    Q_inj = list()
    for node in net.get_nodes():
        for hl in node.get_half_links():
            Q_inj.append(hl.Q)
            P_inj.append(hl.P)

    assert np.all(P_inj_expected+Q_inj_expected == P_inj+Q_inj)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_heat_node_data():
    """Test reading a heat network, with data, from a pd dataframe. Check the node data"""
    # Given
    nodes_input, links_input, halflinks_input = to_pd_dataframes(create_heat_net_with_data())

    # When
    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # Then
    p_expected = [node_data.get('p') for node_data in nodes_input['data']]
    Ts_expected = [node_data.get('Ts') for node_data in nodes_input['data']]
    Tr_expected = [node_data.get('Tr') for node_data in nodes_input['data']]

    p = list()
    Ts = list()
    Tr = list()
    for n in net.get_nodes():
        p.append(n.p)
        Ts.append(n.Ts)
        Tr.append(n.Tr)

    assert np.all(p_expected+Ts_expected+Tr_expected == p+Ts+Tr)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_heat_link_data_double_load_n_odd():
    """Test reading a heat network, with data, from a pd dataframe. Check the link data."""
    # Given
    nodes_input, links_input, halflinks_input = to_pd_dataframes(create_heat_net_with_data())

    # When
    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    # Then
    link_types_expected = ['standard_pipe_low_pres_colebrook']*len(net.links)
    new_link_types = list()
    for e in net.get_links():
        new_link_types.append(e.link_type)
    assert np.all(link_types_expected == new_link_types)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_pd_dataframe_heat_halflink_data_double_load_n_odd():
    """Test reading a heat network, with data, from a pd dataframe. Check the halflink data."""
    # Given
    nodes_input, links_input, halflinks_input = to_pd_dataframes(create_heat_net_with_data())

    # When
    net = from_pd_dataframes(nodes_input,links_input,halflinks_input,flatten=False)

    print(halflinks_input)
    # Then
    m_inj_expected = [hl_data.get('m') for hl_data in halflinks_input['data']]
    phi_inj_expected = [hl_data.get('phi') for hl_data in halflinks_input['data']]
    Ts_hl_expected = [hl_data.get('Ts') for hl_data in halflinks_input['data']]
    Tr_hl_expected = [hl_data.get('Tr') for hl_data in halflinks_input['data']]
    m_inj = list()
    phi_inj = list()
    Ts_hl = list()
    Tr_hl = list()
    for node in net.get_nodes():
        for hl in node.get_half_links():
            print('half link name: {}'.format(hl.name))
            m_inj.append(hl.m)
            phi_inj.append(hl.dphi)
            Ts_hl.append(hl.Ts)
            Tr_hl.append(hl.Tr)

    print('length phi: {}, length phi_expected: {}'.format(len(phi_inj),len(phi_inj_expected)))
    assert np.all(m_inj_expected+phi_inj_expected+Ts_hl_expected+Tr_hl_expected == m_inj+phi_inj+Ts_hl+Tr_hl)
