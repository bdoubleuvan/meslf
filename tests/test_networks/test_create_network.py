"""Test the create network functions
"""
from meslf.networks.create_network import create_radial_line_network_pd, combine_line_networks_full_pd, create_radial_line_network, combine_networks_as_tree, add_gas_data
from meslf.networks.carrier import Gas
import numpy as np
import pandas as pd

# All of the below are depricated. Don't create networks based on pd dataframes, but create them based on Networks!
# ===================================================================================
# Test radial line network based on pd dataframes
def test_nodes_radial_line_default_pd():
    """Test the nodes created for a radial line network with (almost all) default input
    """
    # Given
    n = 2
    carrier = 'g'

    # When
    nodes, _, _ = create_radial_line_network_pd(n,carrier=carrier)

    # Then
    node_names_expected = ['source','gn0','gn1','gn2','gn3']
    carrier_nodes_expected = ['g']*(2*n+1)
    node_types_expected = [0,1,1,1,1]
    x_expected = [-1,0,1,0,1]
    y_expected = [1,0,0,1,1]
    nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected]).T,columns=['name','carrier','type','x','y'])
    pd.testing.assert_frame_equal(nodes_expected,nodes)

def test_links_radial_line_default_pd():
    """Test the links created for a radial line network with (almost all) default input
    """
    # Given
    n = 2
    carrier = 'g'

    # When
    _, links, _ = create_radial_line_network_pd(n,carrier=carrier)

    # Then
    link_names_expected = ['gl0','gl1','gl2','gl3']
    carrier_links_expected = ['g']*(2*n)
    start_nodes_expected = ['source','gn2','gn2','gn3']
    end_nodes_expected = ['gn2','gn3','gn0','gn1']
    links_expected = pd.DataFrame(np.array([link_names_expected,carrier_links_expected,start_nodes_expected,end_nodes_expected]).T,columns=['name','carrier','start_node','end_node'])
    pd.testing.assert_frame_equal(links_expected,links)

def test_halflinks_radial_line_default_pd():
    """Test the halflinks created for a radial line network with (almost all) default input
    """
    # Given
    n = 2
    carrier = 'g'

    # When
    _, _, halflinks = create_radial_line_network_pd(n,carrier=carrier)

    # Then
    halflink_names_expected = ['source_hl','gn0_hl','gn1_hl','gn2_hl','gn3_hl']
    carrier_halflinks_expected = ['g']*(2*n+1)
    start_nodes_halflinks_expected = ['source','gn0','gn1','gn2','gn3']
    halflink_types_expected = [-1,1,1,0,0]
    halflinks_expected = pd.DataFrame(np.array([halflink_names_expected,carrier_halflinks_expected,halflink_types_expected,start_nodes_halflinks_expected]).T,columns=['name','carrier','type','start_node'])
    pd.testing.assert_frame_equal(halflinks_expected,halflinks)

def test_nodes_radial_line_coor_pd():
    """Test the nodes created for a radial line network with given coordinates
    """
    # Given
    n = 2
    carrier = 'g'
    x = [1.3, 1]
    y = [-0.1,0.8]

    # When
    nodes, _, _= create_radial_line_network_pd(n,carrier=carrier,x=x,y=y)

    # Then
    node_names_expected = ['source','gn0','gn1','gn2','gn3']
    carrier_nodes_expected = ['g']*(2*n+1)
    node_types_expected = [0,1,1,1,1]
    x_expected = [0,1.3,1,1.3,1]
    y_expected = [1.8,-0.1,0.8,0.9,1.8]
    nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected]).T,columns=['name','carrier','type','x','y'])
    pd.testing.assert_frame_equal(nodes_expected,nodes)

def test_links_radial_line_coor_pd():
    """Test the links created for a radial line network with with given coordinates
    """
    # Given
    n = 2
    carrier = 'g'
    x = [1.3, 1]
    y = [-0.1,0.8]

    # When
    _, links, _ = create_radial_line_network_pd(n,carrier=carrier,x=x,y=y)

    # Then
    link_names_expected = ['gl0','gl1','gl2','gl3']
    carrier_links_expected = ['g']*(2*n)
    start_nodes_expected = ['source','gn3','gn2','gn3']
    end_nodes_expected = ['gn3','gn2','gn0','gn1']
    links_expected = pd.DataFrame(np.array([link_names_expected,carrier_links_expected,start_nodes_expected,end_nodes_expected]).T,columns=['name','carrier','start_node','end_node'])
    pd.testing.assert_frame_equal(links_expected,links)

def test_halflinks_radial_line_coor_pd():
    """Test the halflinks created for a radial line network with with given coordinates
    """
    # Given
    n = 2
    carrier = 'g'
    x = [1.3, 1]
    y = [-0.1,0.8]

    # When
    _, _, halflinks = create_radial_line_network_pd(n,carrier=carrier,x=x,y=y)

    # Then
    halflink_names_expected = ['source_hl','gn0_hl','gn1_hl','gn2_hl','gn3_hl']
    carrier_halflinks_expected = ['g']*(2*n+1)
    start_nodes_halflinks_expected = ['source','gn0','gn1','gn2','gn3']
    halflink_types_expected = [-1,1,1,0,0]
    halflinks_expected = pd.DataFrame(np.array([halflink_names_expected,carrier_halflinks_expected,halflink_types_expected,start_nodes_halflinks_expected]).T,columns=['name','carrier','type','start_node'])
    pd.testing.assert_frame_equal(halflinks_expected,halflinks)

def test_nodes_radial_no_links_to_loads_pd():
    """Test the nodes created for a radial line network with (almost all) default input
    """
    # Given
    n = 2
    carrier = 'g'

    # When
    nodes, _, _ = create_radial_line_network_pd(n,carrier=carrier,link_to_loads=False)

    # Then
    node_names_expected = ['source','gn0','gn1']
    carrier_nodes_expected = ['g']*(n+1)
    node_types_expected = [0,1,1]
    x_expected = [-1,0,1]
    y_expected = [0,0,0]
    nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected]).T,columns=['name','carrier','type','x','y'])
    pd.testing.assert_frame_equal(nodes_expected,nodes)

def test_links_radial_line_no_links_to_loads_pd():
    """Test the links created for a radial line network with (almost all) default input, but no links between the junctions and the loads.
    """
    # Given
    n = 2
    carrier = 'g'

    # When
    _, links, _ = create_radial_line_network_pd(n,carrier=carrier,link_to_loads=False)

    # Then
    link_names_expected = ['gl0','gl1']
    carrier_links_expected = ['g']*(n)
    start_nodes_expected = ['source','gn0']
    end_nodes_expected = ['gn0','gn1']
    links_expected = pd.DataFrame(np.array([link_names_expected,carrier_links_expected,start_nodes_expected,end_nodes_expected]).T,columns=['name','carrier','start_node','end_node'])
    pd.testing.assert_frame_equal(links_expected,links)

def test_halflinks_radial_line_no_links_to_loads_pd():
    """Test the halflinks created for a radial line network with (almost all) default input, but no links between the junctions and the loads.
    """
    # Given
    n = 2
    carrier = 'g'

    # When
    _, _, halflinks = create_radial_line_network_pd(n,carrier=carrier,link_to_loads=False)

    # Then
    halflink_names_expected = ['source_hl','gn0_hl','gn1_hl']
    carrier_halflinks_expected = ['g']*(n+1)
    start_nodes_halflinks_expected = ['source','gn0','gn1']
    halflink_types_expected = [-1,1,1]
    halflinks_expected = pd.DataFrame(np.array([halflink_names_expected,carrier_halflinks_expected,halflink_types_expected,start_nodes_halflinks_expected]).T,columns=['name','carrier','type','start_node'])
    pd.testing.assert_frame_equal(halflinks_expected,halflinks)

def test_nodes_radial_line_double_load_n_odd_pd():
    """Test the nodes created for a radial line network with two loads per junction for some nodes, and otherwise default input
    """
    # Given
    n = 5
    m = 2
    carrier = 'g'

    # When
    nodes, _, _ = create_radial_line_network_pd(n,carrier=carrier,m=m)

    # Then
    node_names_expected = ['source','gn0','gn1','gn2','gn3','gn4','gn5','gn6','gn7']
    carrier_nodes_expected = ['g']*(2*n-m+1)
    node_types_expected = [0,1,1,1,1,1,1,1,1]
    x_expected = [-1,0,0,1,1,2,0,1,2]
    y_expected = [1,0,2,0,2,0,1,1,1]
    nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected]).T,columns=['name','carrier','type','x','y'])
    pd.testing.assert_frame_equal(nodes_expected,nodes)

def test_links_radial_line_double_load_n_odd_pd():
    """Test the links created for a radial line network with two loads per junction for some nodes, and otherwise default input
    """
    # Given
    n = 5
    m = 2
    carrier = 'g'

    # When
    _, links, _ = create_radial_line_network_pd(n,carrier=carrier,m=m)

    # Then
    link_names_expected = ['gl0','gl1','gl2','gl3','gl4','gl5','gl6','gl7']
    carrier_links_expected = ['g']*(2*n-m)
    start_nodes_expected = ['source','gn5','gn6','gn5','gn5','gn6','gn6','gn7']
    end_nodes_expected = ['gn5','gn6','gn7','gn0','gn1','gn2','gn3','gn4']
    links_expected = pd.DataFrame(np.array([link_names_expected,carrier_links_expected,start_nodes_expected,end_nodes_expected]).T,columns=['name','carrier','start_node','end_node'])
    pd.testing.assert_frame_equal(links_expected,links)

def test_halflinks_radial_line_double_load_n_odd_pd():
    """Test the halflinks created for a radial line network with two loads per junction for some nodes, and otherwise default input
    """
    # Given
    n = 5
    m = 2
    carrier = 'g'

    # When
    _, _, halflinks = create_radial_line_network_pd(n,carrier=carrier,m=m)

    # Then
    halflink_names_expected = ['source_hl','gn0_hl','gn1_hl','gn2_hl','gn3_hl','gn4_hl','gn5_hl','gn6_hl','gn7_hl']
    carrier_halflinks_expected = ['g']*(2*n-m+1)
    start_nodes_halflinks_expected = ['source','gn0','gn1','gn2','gn3','gn4','gn5','gn6','gn7']
    halflink_types_expected = [-1,1,1,1,1,1,0,0,0]
    halflinks_expected = pd.DataFrame(np.array([halflink_names_expected,carrier_halflinks_expected,halflink_types_expected,start_nodes_halflinks_expected]).T,columns=['name','carrier','type','start_node'])
    pd.testing.assert_frame_equal(halflinks_expected,halflinks)

def test_nodes_radial_line_no_links_to_loads_double_load_n_odd_pd():
    """Test the nodes created for a radial line network with two loads per junction for some nodes, and with no links to the loads
    """
    # Given
    n = 5
    m = 2
    carrier = 'g'

    # When
    nodes, _, _ = create_radial_line_network_pd(n,carrier=carrier,m=m,link_to_loads=False)

    # Then
    node_names_expected = ['source','gn0','gn1','gn2']
    carrier_nodes_expected = ['g']*(n-m+1)
    node_types_expected = [0,1,1,1]
    x_expected = [-1,0,1,2]
    y_expected = [0,0,0,0]
    nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected]).T,columns=['name','carrier','type','x','y'])
    pd.testing.assert_frame_equal(nodes_expected,nodes)

def test_links_radial_line_no_links_to_loads_double_load_n_odd_pd():
    """Test the links created for a radial line network with two loads per junction for some nodes, and with no links to the loads
    """
    # Given
    n = 5
    m = 2
    carrier = 'g'

    # When
    _, links, _ = create_radial_line_network_pd(n,carrier=carrier,m=m,link_to_loads=False)

    # Then
    link_names_expected = ['gl0','gl1','gl2']
    carrier_links_expected = ['g']*(n-m)
    start_nodes_expected = ['source','gn0','gn1']
    end_nodes_expected = ['gn0','gn1','gn2']
    links_expected = pd.DataFrame(np.array([link_names_expected,carrier_links_expected,start_nodes_expected,end_nodes_expected]).T,columns=['name','carrier','start_node','end_node'])
    pd.testing.assert_frame_equal(links_expected,links)

def test_halflinks_radial_line_no_links_to_loads_double_load_n_odd_pd():
    """Test the halflinks created for a radial line network with two loads per junction for some nodes, and with no links to the loads
    """
    # Given
    n = 5
    m = 2
    carrier = 'g'

    # When
    _, _, halflinks = create_radial_line_network_pd(n,carrier=carrier,m=m,link_to_loads=False)

    # Then
    halflink_names_expected = ['source_hl','gn0_hl0','gn0_hl1','gn1_hl0','gn1_hl1','gn2_hl']
    carrier_halflinks_expected = ['g']*(n+1)
    start_nodes_halflinks_expected = ['source','gn0','gn0','gn1','gn1','gn2']
    halflink_types_expected = [-1,1,1,1,1,1]
    halflinks_expected = pd.DataFrame(np.array([halflink_names_expected,carrier_halflinks_expected,halflink_types_expected,start_nodes_halflinks_expected]).T,columns=['name','carrier','type','start_node'])
    pd.testing.assert_frame_equal(halflinks_expected,halflinks)

def test_halflinks_radial_line_no_links_to_loads_double_load_n_even_pd():
    """Test the halflinks created for a radial line network with two loads per junction for some nodes, and with no links to the loads
    """
    # Given
    n = 4
    m = 2
    carrier = 'g'

    # When
    _, _, halflinks = create_radial_line_network_pd(n,carrier=carrier,m=m,link_to_loads=False)

    # Then
    halflink_names_expected = ['source_hl','gn0_hl0','gn0_hl1','gn1_hl0','gn1_hl1']
    carrier_halflinks_expected = ['g']*(n+1)
    start_nodes_halflinks_expected = ['source','gn0','gn0','gn1','gn1']
    halflink_types_expected = [-1,1,1,1,1]
    halflinks_expected = pd.DataFrame(np.array([halflink_names_expected,carrier_halflinks_expected,halflink_types_expected,start_nodes_halflinks_expected]).T,columns=['name','carrier','type','start_node'])
    pd.testing.assert_frame_equal(halflinks_expected,halflinks)

# # ===================================================================================
# # Test combinations of radial line networks based on pd dataframes
# def test_nodes_combine_line_networks_full_pd_same_pd():
#     """Test the nodes for a combination of the same two line networks into one network
#     """
#     # GIVES ERRORS ON LAPTOP DUE TO OLD VERSION OF PANDAS
#     #Given
#     n = 3
#     carrier = 'g'
#     nodes1, links1, halflinks1 = create_radial_line_network_pd(n,carrier=carrier)
#
#     # When
#     nodes, _, _ = combine_line_networks_full_pd([nodes1,nodes1.copy()],[links1,links1.copy()],[halflinks1,halflinks1.copy()],keys=['street_1','street_2'],adjust_coor=True)
#
#     # Then
#     node_names_expected = ['source','source','gn0','gn1','gn2','gn3','gn4','gn5','source','gn0','gn1','gn2','gn3','gn4','gn5']
#     carrier_nodes_expected = ['g']*(len(nodes1.index)+len(nodes1.index)+1)
#     node_types_expected = [0]+[1]*(len(nodes1.index)+len(nodes1.index))
#     x_expected = [-2,-1,0,1,2,0,1,2,-1,0,1,2,0,1,2]
#     y_expected = [1.5,1.,0.,0.,0.,1.,1.,1.,3.,2.,2.,2.,3.,3.,3.]
#     nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected]).T,columns=['name','carrier','type','x','y'],index=[['coupling']+['street_1']*len(nodes1.index)+['street_2']*len(nodes1.index),[0]+nodes1.index.tolist()+nodes1.index.tolist()])
#     pd.testing.assert_frame_equal(nodes_expected,nodes)
#
# def test_nodes_combine_line_networks_full_pd_different_pd():
#     """Test the nodes for a combination of two different line networks into one network
#     """
#     # GIVES ERRORS ON LAPTOP DUE TO OLD VERSION OF PANDAS
#     #Given
#     n1 = 3
#     n2 = 5
#     m2 = 2
#     carrier = 'g'
#     nodes1, links1, halflinks1 = create_radial_line_network_pd(n1,carrier=carrier)
#     nodes2, links2, halflinks2 = create_radial_line_network_pd(n2,carrier=carrier,m=m2,link_to_loads=False)
#
#     # When
#     nodes, _, _ = combine_line_networks_full_pd([nodes1,nodes2],[links1,links2],[halflinks1,halflinks2],keys=['street_1','street_2'],adjust_coor=True)
#
#     # Then
#     node_names_expected = ['source','source','gn0','gn1','gn2','gn3','gn4','gn5','source','gn0','gn1','gn2']
#     carrier_nodes_expected = ['g']*(len(nodes1.index)+len(nodes2.index)+1)
#     node_types_expected = [0]+ [1]*(len(nodes1.index)+len(nodes2.index))
#     x_expected = [-2,-1,0,1,2,0,1,2,-1,0,1,2]
#     y_expected = [1.,1.,0.,0.,0.,1.,1.,1.,2.,2.,2.,2.]
#     nodes_expected = pd.DataFrame(np.array([node_names_expected,carrier_nodes_expected,node_types_expected,x_expected,y_expected]).T,columns=['name','carrier','type','x','y'],index=[['coupling']+['street_1']*len(nodes1.index)+['street_2']*len(nodes2.index),[0]+nodes1.index.tolist()+nodes2.index.tolist()])
#     pd.testing.assert_frame_equal(nodes_expected,nodes)
#
# def test_links_combine_line_networks_full_pd_different_pd():
#     """Test the links for a combination of two different line networks into one network
#     """
#     # GIVES ERRORS ON LAPTOP DUE TO OLD VERSION OF PANDAS
#     #Given
#     n1 = 3
#     n2 = 5
#     m2 = 2
#     carrier = 'g'
#     nodes1, links1, halflinks1 = create_radial_line_network_pd(n1,carrier=carrier)
#     nodes2, links2, halflinks2 = create_radial_line_network_pd(n2,carrier=carrier,m=m2,link_to_loads=False)
#
#     # When
#     _, links, _ = combine_line_networks_full_pd([nodes1,nodes2],[links1,links2],[halflinks1,halflinks2],keys=['street_1','street_2'],adjust_coor=True)
#
#     # Then
#     link_names_expected = ['gl0','gl1','gl0','gl1','gl2','gl3','gl4','gl5','gl0','gl1','gl2']
#     carrier_links_expected = ['g']*(len(links1.index)+len(links2.index)+2)
#     start_nodes_expected = ['','','source','gn3','gn4','gn3','gn4','gn5','source','gn0','gn1']
#     end_nodes_expected = ['','','gn3','gn4','gn5','gn0','gn1','gn2','gn0','gn1','gn2']
#     links_expected = pd.DataFrame(np.array([link_names_expected,carrier_links_expected,start_nodes_expected,end_nodes_expected]).T,columns=['name','carrier','start_node','end_node'],index=[['coupling']*2+['street_1']*len(links1.index)+['street_2']*len(links2.index),[0,1]+links1.index.tolist()+links2.index.tolist()])
#     links_expected.loc['coupling',0]['start_node'] = ['coupling','source']
#     links_expected.loc['coupling',1]['start_node'] = ['coupling','source']
#     links_expected.loc['coupling',0]['end_node'] = ['street_1','source']
#     links_expected.loc['coupling',1]['end_node'] = ['street_2','source']
#     pd.testing.assert_frame_equal(links_expected,links)
#
#
# def test_halflinks_combine_line_networks_full_pd_different_pd():
#     """Test the halflinks for a combination of two different line networks into one network
#     """
#     # GIVES ERRORS ON LAPTOP DUE TO OLD VERSION OF PANDAS
#     #Given
#     n1 = 3
#     n2 = 5
#     m2 = 2
#     carrier = 'g'
#     nodes1, links1, halflinks1 = create_radial_line_network_pd(n1,carrier=carrier)
#     nodes2, links2, halflinks2 = create_radial_line_network_pd(n2,carrier=carrier,m=m2,link_to_loads=False)
#
#     # When
#     _, _, halflinks = combine_line_networks_full_pd([nodes1,nodes2],[links1,links2],[halflinks1,halflinks2],keys=['street_1','street_2'],adjust_coor=True)
#
#     # Then
#     halflink_names_expected = ['source_hl','source_hl','gn0_hl','gn1_hl','gn2_hl','gn3_hl','gn4_hl','gn5_hl','source_hl','gn0_hl0','gn0_hl1','gn1_hl0','gn1_hl1','gn2_hl']
#     carrier_halflinks_expected = ['g']*(len(halflinks1.index)+len(halflinks2.index)+1)
#     start_nodes_halflinks_expected = ['source','source','gn0','gn1','gn2','gn3','gn4','gn5','source','gn0','gn0','gn1','gn1','gn2']
#     halflink_types_expected = [-1,0,1,1,1,0,0,0,0,1,1,1,1,1]
#     halflinks_expected = pd.DataFrame(np.array([halflink_names_expected,carrier_halflinks_expected,halflink_types_expected,start_nodes_halflinks_expected]).T,columns=['name','carrier','type','start_node'],index=[['coupling']+['street_1']*len(halflinks1.index)+['street_2']*len(halflinks2.index),[0]+halflinks1.index.tolist()+halflinks2.index.tolist()])
#     pd.testing.assert_frame_equal(halflinks_expected,halflinks)

# ===================================================================================
# Test radial line network based on Network
def test_net_radial_line_default():
    """Test the network created for a radial line network with (almost all) default input, by checking the adjacency matrix
    """
    # Given
    n = 2
    carrier = 'g'

    # When
    net = create_radial_line_network(n,carrier=carrier)
    net.initialize()

    # Then
    A_test = np.array([[-1.,0.,0.,0.],
                       [0.,0.,1.,0.],
                       [0.,0.,0.,1.],
                       [1.,-1.,-1.,0.],
                       [0.,1.,0.,-1.]])

    assert np.all(A_test == net.A)

def test_halflinks_radial_line_default():
    """Test the halflinks of the network created for a radial line network with (almost all) default input, by checking the names of the start nodes of the halflinks
    """
    # Given
    n = 2
    carrier = 'g'

    # When
    net = create_radial_line_network(n,carrier=carrier)
    net.initialize()

    # Then
    start_nodes_expected = ['source','gn0','gn1']
    start_nodes = list()
    for hl in net.get_half_links():
        start_nodes.append(hl.start_node.name)
    assert np.all(start_nodes_expected == start_nodes)

def test_net_radial_no_links_to_loads():
    """Test the network created for a radial line network with (almost all) default input but no links to the loads, by checking the adjacency matrix
    """
    # Given
    n = 2
    carrier = 'g'

    # When
    net = create_radial_line_network(n,carrier=carrier,link_to_loads=False)
    net.initialize()

    # Then
    A_test = np.array([[-1.,0.],
                       [1.,-1.],
                       [0.,1.]])
    assert np.all(A_test == net.A)

def test_halflinks_radial_no_links_to_loads():
    """Test the halflinks of the network created for a radial line network with (almost all) default input but no links to the loads, by checking the names of the start nodes of the halflinks
    """
    # Given
    n = 2
    carrier = 'g'

    # When
    net = create_radial_line_network(n,carrier=carrier,link_to_loads=False)
    net.initialize()

    # Then
    start_nodes_expected = ['source','gn0','gn1']
    start_nodes = list()
    for hl in net.get_half_links():
        start_nodes.append(hl.start_node.name)
    assert np.all(start_nodes_expected == start_nodes)

def test_net_radial_line_double_load_n_odd():
    """Test the network created for a radial line network with two loads per junction for some nodes, and otherwise default input, by checking the adjacency matrix
    """
    # Given
    n = 5
    m = 2
    carrier = 'g'

    # When
    net = create_radial_line_network(n,carrier=carrier,m=m)
    net.initialize()

    # Then
    A_test = np.array([[-1.,0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,1.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,1.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,1.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,1.,0.],
                       [0.,0.,0.,0.,0.,0.,0.,1.],
                       [1.,-1.,0.,-1.,-1.,0.,0.,0.],
                       [0.,1.,-1.,0.,0.,-1.,-1.,0.],
                       [0.,0.,1.,0.,0.,0.,0.,-1.]])
    assert np.all(A_test == net.A)

def test_halflinks_radial_line_double_load_n_odd():
    """Test the half links of the network created for a radial line network with two loads per junction for some nodes, and otherwise default input, by checking the start nodes of the halflinks
    """
    # Given
    n = 5
    m = 2
    carrier = 'g'

    # When
    net = create_radial_line_network(n,carrier=carrier,m=m)
    net.initialize()

    # Then
    start_nodes_expected = ['source','gn0','gn1','gn2','gn3','gn4']
    start_nodes = list()
    for hl in net.get_half_links():
        start_nodes.append(hl.start_node.name)
    assert np.all(start_nodes_expected == start_nodes)

def test_net_radial_line_no_links_to_loads_double_load_n_odd():
    """Test the network created for a radial line network with two loads per junction for some nodes, and with no links to the loads, by checking the adjacency matrix
    """
    # Given
    n = 5
    m = 2
    carrier = 'g'

    # When
    net = create_radial_line_network(n,carrier=carrier,m=m,link_to_loads=False)
    net.initialize()

    # Then
    A_test = np.array([[-1.,0.,0.],
                       [1.,-1.,0.],
                       [0.,1.,-1.],
                       [0.,0.,1.]])
    assert np.all(A_test == net.A)

def test_halflinks_radial_line_no_links_to_loads_double_load_n_odd():
    """Test the halflinks of the network created for a radial line network with two loads per junction for some nodes, and with no links to the loads, by checking the start nodes of the halflinks
    """
    # Given
    n = 5
    m = 2
    carrier = 'g'

    # When
    net = create_radial_line_network(n,carrier=carrier,m=m,link_to_loads=False)
    net.initialize()

    # Then
    start_nodes_expected = ['source','gn0','gn0','gn1','gn1','gn2']
    start_nodes = list()
    for hl in net.get_half_links():
        start_nodes.append(hl.start_node.name)
    assert np.all(start_nodes_expected == start_nodes)

# Test combining networks based on Network
def test_net_combine_line_networks():
    """Test the links for a combination of two different line networks into one network, by checking the adjacency matrix
    """
    #Given
    n1 = 3
    n2 = 5
    m2 = 2
    carrier = 'g'
    net1 = create_radial_line_network(n1,carrier=carrier)
    net2 = create_radial_line_network(n2,carrier=carrier,m=m2,link_to_loads=False)

    # When
    net = combine_networks_as_tree([net1,net2])
    net.initialize()

    # Then
    A_test = np.array([[0.,0.,0.,0.,0.,0.,-1.,0.,0.,0.,-1.],
                       [-1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
                       [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
                       [1.,-1.,0.,-1.,0.,0.,0.,0.,0.,0.,0.],
                       [0.,1.,-1.,0.,-1.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,1.,0.,0.,-1.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.,1.],
                       [0.,0.,0.,0.,0.,0.,0.,1.,-1.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.,0.,1.,-1.,0.],
                       [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]])
    assert np.all(A_test == net.A)

def test_halflinks_combine_line_networks():
    """Test the links for a combination of two different line networks into one network, by checking the start nodes of the halflinks
    """
    #Given
    n1 = 3
    n2 = 5
    m2 = 2
    carrier = 'g'
    net1 = create_radial_line_network(n1,carrier=carrier)
    net2 = create_radial_line_network(n2,carrier=carrier,m=m2,link_to_loads=False)

    # When
    net = combine_networks_as_tree([net1,net2])
    net.initialize()

    # Then
    start_nodes_expected = ['source','source','gn0','gn1','gn2','source','gn0','gn0','gn1','gn1','gn2']
    start_nodes = list()
    for hl in net.get_half_links():
        start_nodes.append(hl.start_node.name)
    print(start_nodes)
    assert np.all(start_nodes_expected == start_nodes)

def test_net_combine_line_networks_twice():
    """Test the links for a combination of two different line networks into one network (2 times), which are in turn combined into one network, by checking the adjacency matrix
    """
    #Given
    n1 = 3
    n2 = 5
    m2 = 2
    carrier = 'g'
    S1 = create_radial_line_network(n1,carrier=carrier)
    S3 = create_radial_line_network(n1,carrier=carrier)
    S2 = create_radial_line_network(n2,carrier=carrier,m=m2,link_to_loads=False)
    S4 = create_radial_line_network(n2,carrier=carrier,m=m2,link_to_loads=False)

    # When
    Q1 = combine_networks_as_tree([S1,S2])
    Q2 = combine_networks_as_tree([S3,S4])
    D1 = combine_networks_as_tree([Q1,Q2])
    D1.initialize()

    # Then
    A_S = np.array([[0.,0.,0.,0.,0.,0.,-1.,0.,0.,0.,-1.],
                       [-1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
                       [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
                       [1.,-1.,0.,-1.,0.,0.,0.,0.,0.,0.,0.],
                       [0.,1.,-1.,0.,-1.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,1.,0.,0.,-1.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.,1.],
                       [0.,0.,0.,0.,0.,0.,0.,1.,-1.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.,0.,1.,-1.,0.],
                       [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]])
    col = np.zeros((12,1))
    col[0] = 1.
    A_test = np.concatenate((np.concatenate((np.zeros((1,11)),-1.*np.ones((1,1)),np.zeros((1,11)),-1.*np.ones((1,1))),axis=1),
                       np.concatenate((A_S,col,np.zeros((12,11)),np.zeros((12,1))),axis=1),
                       np.concatenate((np.zeros((12,11)),np.zeros((12,1)),A_S,col),axis=1)))
    assert np.all(A_test == D1.A)

# ===================================================================================
# Test adding data to network
def test_gas_data_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check if the network structure hasn't been changed."""
    # Given
    n = 5
    m = 2
    carrier = 'g'

    net = create_radial_line_network(n,carrier=carrier,m=m)
    net.initialize()
    A_before = net.A.copy().todense()

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

    # When
    add_gas_data(net,p_ref,q_inj_loads,link_types,link_params)
    net.initialize()

    # Then
    assert np.all(A_before == net.A)

def test_gas_node_data_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the node data."""
    # Given
    n = 5
    m = 2
    carrier = 'g'

    net = create_radial_line_network(n,carrier=carrier,m=m)
    net.initialize()
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

    # When
    add_gas_data(net,p_ref,q_inj_loads,link_types,link_params)

    # Then
    p_expected = np.zeros(number_of_nodes)
    p_expected[0] = p_ref
    p = list()
    for n in net.get_nodes():
        p.append(n.p)
    assert np.all(p_expected == p)

def test_gas_link_data_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the link type."""
    # Given
    n = 5
    m = 2
    carrier = 'g'

    net = create_radial_line_network(n,carrier=carrier,m=m)
    net.initialize()
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

    # When
    add_gas_data(net,p_ref,q_inj_loads,link_types,link_params)

    # Then
    new_link_types = list()
    for e in net.get_links():
        new_link_types.append(e.link_type)
    assert np.all(link_types == new_link_types)

def test_gas_halflink_data_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the halflink data."""
    # Given
    n = 5
    m = 2
    carrier = 'g'

    net = create_radial_line_network(n,carrier=carrier,m=m)
    net.initialize()


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

    # When
    add_gas_data(net,p_ref,q_inj_loads,link_types,link_params)
    number_of_halflinks = len(net.half_links)

    # Then
    q_inj_expected = np.zeros(number_of_halflinks)
    q_inj_expected[1:n+1] = q_inj_loads
    q_inj = list()
    for node in net.get_nodes():
        for hl in node.get_half_links():
            q_inj.append(hl.q)

    assert np.all(q_inj_expected == q_inj)

def test_gas_data_no_links_to_loads_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, without links to loads, and multiple loads per junction. Check if the network structure hasn't been changed."""
    # Given
    n = 5
    m = 2
    carrier = 'g'

    net = create_radial_line_network(n,carrier=carrier,m=m,link_to_loads=False)
    net.initialize()
    A_before = net.A.copy().todense()

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

    # When
    add_gas_data(net,p_ref,q_inj_loads,link_types,link_params)
    net.initialize()

    # Then
    assert np.all(A_before == net.A)

def test_gas_node_data_no_links_to_loads_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the node data."""
    # Given
    n = 5
    m = 2
    carrier = 'g'

    net = create_radial_line_network(n,carrier=carrier,m=m,link_to_loads=False)
    net.initialize()
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

    # When
    add_gas_data(net,p_ref,q_inj_loads,link_types,link_params)

    # Then
    p_expected = np.zeros(number_of_nodes)
    p_expected[0] = p_ref
    p = list()
    for n in net.get_nodes():
        p.append(n.p)
    assert np.all(p_expected == p)

def test_gas_link_data_no_links_to_loads_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the link type."""
    # Given
    n = 5
    m = 2
    carrier = 'g'

    net = create_radial_line_network(n,carrier=carrier,m=m,link_to_loads=False)
    net.initialize()
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

    # When
    add_gas_data(net,p_ref,q_inj_loads,link_types,link_params)

    # Then
    new_link_types = list()
    for e in net.get_links():
        new_link_types.append(e.link_type)
    assert np.all(link_types == new_link_types)

def test_gas_halflink_no_links_to_loads_data_double_load_n_odd():
    """Test the addition of gas to a radial / tree-like network, with links to loads, and multiple loads per junction. Check the halflink data."""
    # Given
    n = 5
    m = 2
    carrier = 'g'

    net = create_radial_line_network(n,carrier=carrier,m=m,link_to_loads=False)
    net.initialize()


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

    # When
    add_gas_data(net,p_ref,q_inj_loads,link_types,link_params)
    number_of_halflinks = len(net.half_links)

    # Then
    q_inj_expected = np.zeros(number_of_halflinks)
    q_inj_expected[1:n+1] = q_inj_loads
    q_inj = list()
    for node in net.get_nodes():
        for hl in node.get_half_links():
            q_inj.append(hl.q)

    assert np.all(q_inj_expected == q_inj)
