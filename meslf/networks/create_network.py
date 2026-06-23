"""Functions to automatically create network"""
import numpy as np
import pandas as pd
from statistics import median
from meslf.networks.network import Network, Node, Link, HalfLink
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink


# ====================================== Create / combine using pd dataframes ==========================
def create_radial_line_network_pd(n,names=None,carrier='no type',x=None,y=None,link_to_loads=True,m=0):
    """Creates a radial network in the shape of a line or 'street', of one carrier type. Links and junctions needed to connect the houses are generated automatically.  
    
    Parameters
    ----------
    n : int
        Number of loads (or 'houses')
    names : list, optional
        List of lenght n with names. If no list is provided, default names are used
    carrier : string, optional
        Carrier type of the network, options are 'g','e','h','c','no type'. Default is 'no type'
    x : list, optional
        List of length n with x-coordinates (floats). If no list is provided, coordinates are automatically generated
    y : list, optional
        List of length n with x-coordinates (floats). If no list is provided, coordinates are automatically generated
    link_to_loads : boolean, optional
        Specifies if the load are connected to be junction by links (True) or not (False). Default is True. 
    m : int, optional
        Number of junctions with two loads. Must be between 0 and n/2, default is 0. 
        
    Returns
    -------
    nodes : pd DataFrame
        Node data
    links : pd DataFrame
        Link data
    halflinks: pd DataFrame
        Halflink data
        
    Raises
    ------
    ValueError
        If one of the lists names, x, or y provided is not of length n
    ValueError
        If only x or only y is provided
    ValueError
        If m is not between 0 and n/2.
    """
    # check input
    if names:
        if not len(names) == n:
            raise ValueError("names must be of length {}".format(n))
    else:
        names = list()
    if x:
        if not len(x) == n:
            raise ValueError("x must be of length {}".format(n))
        if y:
            if not len(x) == len(y):
                raise ValueError("x and y must have the same length")
        else:
            raise ValueError("x and y must have the same length")
    else:
        x = list()
    if y:
        if not len(y) == n:
            raise ValueError("y must be of length {}".format(n))
    else:
        y = list()
    if m<0 or m>n/2:
        raise ValueError("m must be between 0 and {}/2".format(n))
    
    # create node topology data
    if link_to_loads:
        if m:
            number_of_nodes = 2*n-m+1 # m junctions connected to two load, the other to one load. And there is one source
        else:
            number_of_nodes = 2*n+1 # every load is connected to a junction. And there is one source node
    else:
        if m:
            number_of_nodes = n-m+1 # m load nodes and one source node 
        else:
            number_of_nodes = n+1 # load nodes and one source node 
    node_types = np.ones(number_of_nodes,dtype=int) 
    # adjust lists to also include junction nodes
    junction_names = list()
    junction_y = list()
    for ind in range(n):
        # add names of loads
        if len(names)<n:
            if m and (not link_to_loads): # more than one halflink per (some) nodes
                if ind%2 and ind < 2*m:
                    names.append(carrier+'n'+str((ind-1)//2))
                elif ind>=2*m:
                    names.append(carrier+'n'+str(ind-m))
            else:
                names.append(carrier+'n'+str(ind)) 
        if link_to_loads:
            # add names of junction nodes
            if m:
                if ind%2 and ind < 2*m:
                    junction_names.append(carrier+'n'+str(n+((ind-1)//2))) 
                elif ind>=2*m:
                    junction_names.append(carrier+'n'+str(n+(ind-m)))
            else:
                junction_names.append(carrier+'n'+str(n+ind)) 
            # add coordinates of load nodes to x and y, and add y coordinates of junction nodes
            if len(x)<n:               
                if ind < 2*m:
                    x.append((ind-ind%2)//2)
                    y.append(2*(ind%2))
                    if ind%2:                        
                        junction_y.append(1)                      
                else:
                    x.append(ind-m) 
                    y.append(0)                    
                    junction_y.append(1)
            else:
                junction_y.append(y[ind]+1)
        else:
            if len(x)<n:
                if m:
                    if ind%2 and ind < 2*m:
                        x.append((ind-1)//2)
                        y.append(0)                     
                    elif ind>=2*m:
                        x.append(ind-m) 
                        y.append(0)
                else:
                    x.append(ind)
                    y.append(0)
    # create total lists for nodes
    names = ['source'] + names + junction_names
    carrier_list = [carrier] * number_of_nodes
    node_types[0] = 0 # source node is the reference node
    if link_to_loads:
        y = [junction_y[x.index(min(x))]] + y + junction_y # source is at same height as junction it is connected to
        x = [min(x)-1] + x + [x[ind] for ind in range(n) if (not ind%2) or ind > 2*m ] # source node to the left of all the other nodes
    else:
        y = [y[x.index(min(x))]] + y
        x = [min(x)-1] + x 
        
    # create link topology data
    if link_to_loads:
        number_of_links = 2*n-m
        nodes_sorted = [name for _,name in sorted(zip(x[n+1:],junction_names))] # sort junction nodes from 'left' to 'right', i.e. based on x coordinates
        end_nodes = nodes_sorted+names[1:n+1]
        if m: 
            start_nodes = ['source']+nodes_sorted[:-1]+[junction_names[ind] for ind in range(m) for _ in range(2)]+junction_names[m:]
        else:
            start_nodes = ['source']+nodes_sorted[:-1]+junction_names
    else:
        number_of_links = n-m
        nodes_sorted = [name for _,name in sorted(zip(x,names))] # sort nodes from 'left' to 'right', i.e. based on x coordinates
        start_nodes = nodes_sorted[:-1]
        end_nodes = nodes_sorted[1:]
        
    link_names = list()
    for ind in range(number_of_links):
        link_names.append(carrier+'l'+str(ind))
    carrier_links = carrier_list[:number_of_links]
    
    # create halflink topology data
    if m and (not link_to_loads): # more than one halflink per (some) nodes
        start_nodes_halflinks =  [names[0]] + [name for name in names[1:m+1] for _ in range(2)] + names[m+1:]
        halflink_names = [names[0]+'_hl'] + [name + '_hl' + str(ind_hl) for name in names[1:m+1] for ind_hl in range(2)] + [name + '_hl' for name in names[m+1:]]
        halflink_types = [-1 if 'source' in name else 1 for name in halflink_names] # no junction nodes in this case. All nodes have at least one source or load halflink connected to it
    else: # one half link for every node
        halflink_names = [name + '_hl' for name in names]
        start_nodes_halflinks = names
        halflink_types = [-1 if name=='source' else 0 if name in junction_names else 1 for name in names]
    carrier_halflinks = [carrier] * len(halflink_names)
    
    # create dataframes
    nodes = pd.DataFrame(np.array([names,carrier_list,node_types,x,y]).T,columns=['name','carrier','type','x','y'])
    links = pd.DataFrame(np.array([link_names,carrier_links,start_nodes,end_nodes]).T,columns=['name','carrier','start_node','end_node'])
    halflinks = pd.DataFrame(np.array([halflink_names,carrier_halflinks,halflink_types,start_nodes_halflinks]).T,columns=['name','carrier','type','start_node'])
    return nodes, links, halflinks
    
def combine_line_networks_full_pd(nodes_list,links_list,halflinks_list,keys=None,adjust_coor=False):
    """Combines single line networks to form a bigger network. 
    
    Parameters
    ----------
    nodes_list : list
        list of nodes dataframes of single line networks
    links_list : list
        list of links dataframes of single line networks
    halflinks_list : list
        list of halflinks dataframes of single line networks
    keys : list
        list of strings with the keys used to indicate the different blocks
    adjust_coor : boolean, optional
        automatically adjusts the coordinates. Only usefull if the single line networks have artifically created coordinates. Default is False
    
    Returns
    -------
    nodes : pd DataFrame
        Node data of combined network
    links : pd DataFrame
        Link data of combined network
    halflinks: pd DataFrame
        Halflink data of combined network
        
    Raises
    ------
    ValueError
        If the input lists are not of the same length.
    """
    if keys:
        it = iter([nodes_list, links_list, halflinks_list,keys])
    else:
        it = iter([nodes_list, links_list, halflinks_list])
    the_len = len(next(it))
    if not all(len(input_list) == the_len for input_list in it):
        raise ValueError("Not all input lists have the same length")
    
    if not keys:
        keys = list()
        for i in range(the_len):
            keys.append('network_'+str(i))   
            
    # combine nodes of different networks
    if adjust_coor:
        y_max = [int(float(max(node['y']))) for node in nodes_list]
        for ind,node in enumerate(nodes_list):
            node['y'] = node['y'].astype(float)+ sum(y_max[:ind])+1*ind            
    nodes = pd.concat(nodes_list, keys=keys)
    nodes.loc[nodes['name']=='source','type']=1 #change the sources from reference nodes to load nodes
    # add coupling node
    source_nodes = nodes.loc[nodes['name'] == 'source']
    source_carriers = source_nodes['carrier'].tolist()
    if all(carrier == source_carriers[0] for carrier in source_carriers):
        coupling_carrier = source_carriers[0] #homogeneous coupling node
    else:
        coupling_carrier = 'c' # heterogeneous coupling node
    new_index_nodes = [['coupling']]
    for level in range(nodes.index.nlevels-2):
        new_index_nodes.append([''])
    new_index_nodes.append([0])
    coupling_node = pd.DataFrame([{'name':'source','carrier':coupling_carrier,'type':0,'x':int(min(nodes['x']))-1,'y':median(nodes['y'].astype(float))}], index=new_index_nodes)
    nodes = pd.concat([coupling_node,nodes])[nodes_list[0].columns]
    nodes['type'] = nodes['type'].astype(str)
    nodes['x'] = nodes['x'].astype(str)
    nodes['y'] = nodes['y'].astype(str)
    
    # combine links of different networks
    links = pd.concat(links_list, keys=keys)
    coupling_links = pd.DataFrame()
    for ind,key in enumerate(keys):
        coupling_link_carrier = source_nodes.loc[key,'carrier'].iloc[0]
        coupling_link_end_node = source_nodes.loc[key,'name'].iloc[0]
        new_index_links = [['coupling']]
        for level in range(links.index.nlevels-2):
            new_index_links.append([''])
        new_index_links.append([ind])
        start_node = coupling_node.index.tolist()
        coupling_links = pd.concat([coupling_links,pd.DataFrame([{'name':'gl'+str(ind),'carrier':coupling_link_carrier,'start_node':['coupling','source'],'end_node':[key,coupling_link_end_node]}], index=new_index_links)])
    links = pd.concat([coupling_links,links])[links_list[0].columns]
    
    # combine halflinks of different networks
    halflinks = pd.concat(halflinks_list, keys=keys)
    halflinks.loc[halflinks['name']=='source_hl','type']=0 #change the sources from inflow halflinks to junctions
    new_index_halflinks = [['coupling']]
    for level in range(halflinks.index.nlevels-2):
        new_index_halflinks.append([''])
    new_index_halflinks.append([0])
    coupling_halflink = pd.DataFrame([{'name':'source_hl','carrier':coupling_carrier,'type':-1,'start_node':'source'}], index=new_index_halflinks)
    halflinks = pd.concat([coupling_halflink,halflinks])[halflinks_list[0].columns]
    halflinks['type'] = halflinks['type'].astype(str)
    return nodes, links, halflinks

# ====================================== Create / combine using Network ==========================
def create_radial_line_network(n,net_name=None,carrier='no type',link_to_loads=True,m=0):
    """Creates a radial network in the shape of a line or 'street', of one carrier type. Links and junctions needed to connect the houses are generated automatically.  
    
    Parameters
    ----------
    n : int
        Number of loads (or 'houses')
    net_name : string, optional
        Name of the network. 
    carrier : string, optional
        Carrier type of the network, options are 'g','e','h','c','no type'. Default is 'no type'
    link_to_loads : boolean, optional
        Specifies if the load are connected to be junction by links (True) or not (False). Default is True. 
    m : int, optional
        Number of junctions with two load. Must be between 0 and n/2, default is 0. 
        
    Returns
    -------
    network : Network
        The network
        
    Raises
    ------
    ValueError
        If m is not between 0 and n/2.
    ValueError
        If carrier is not 'g','e','h','c', or 'no type'
    """
    # check input
    if m<0 or m>n/2:
        raise ValueError('m must be between 0 and {}/2'.format(n))
    
    if not net_name:
        net_name = 'S0'  
        
    # create empty network
    if carrier == 'no_type':
        net = Network(net_name)
    elif carrier == 'g':
        net = GasNetwork(net_name)
    elif carrier == 'e':
        net = ElectricalNetwork(net_name)
    elif carrier == 'h':
        net = HeatNetwork(net_name)
    elif carrier == 'c':
        raise NotImplementedError("Only standard, gas network, electrical network, and heat network are currently implemented")
    else:
        raise ValueError("Enter a valid value for carrier: 'g','e','h','c','no type'.")
        
    # create nodes
    if link_to_loads:
        if m:
            number_of_nodes = 2*n-m+1 # m junctions connected to two load, the other to one load. And there is one source
        else:
            number_of_nodes = 2*n+1 # every load is connected to a junction. And there is one source node
    else:
        if m:
            number_of_nodes = n-m+1 # m load nodes and one source node 
        else:
            number_of_nodes = n+1 # load nodes and one source node 
    # node data
    for ind in range(number_of_nodes):
        if ind == 0: #source node 
            node_name = 'source'
            node_type = 0
            node_x = -1
            if link_to_loads:
                node_y = 1
            else:
                node_y = 0 
        elif ind < n+1: # loads / demands / sinks
            node_name = carrier+'n'+str(ind-1)
            if carrier == 'e':
                node_type = 2
            else:
                node_type = 1
            if link_to_loads:
                if ind < 2*m+1:
                    node_x = (ind-1-(ind-1)%2)//2
                    node_y = 2*((ind-1)%2)                     
                else:
                    node_x = ind-1-m
                    node_y = 0
            else:
                node_x = ind-1
                node_y = 0
        else: # junctions
            if ind < 2*m+1:
                node_x = (ind-1-(ind-1)%2)//2                     
            else:
                node_x = ind-1-n
            node_name = carrier+'n'+str(ind-1)
            if carrier == 'e' or carrier == 'h':
                node_type = 2
            else:
                node_type = 1
            if link_to_loads:
                node_y = 1
            else:
                node_y = 0
        # create node object    
        if carrier == 'no_type':
            node = Node(node_name,x=node_x,y=node_y)
        elif carrier == 'g':
            node = GasNode(node_name,x=node_x,y=node_y,node_type=node_type)
        elif carrier == 'e':
            node = ElectricalNode(node_name,x=node_x,y=node_y,node_type=node_type)
        elif carrier == 'h':
            node = HeatNode(node_name,x=node_x,y=node_y,node_type=node_type)
        elif carrier == 'c':
            raise NotImplementedError("Only standard, gas network, electrical network, and heat network are currently implemented")
        else:
            raise ValueError("Enter a valid value for carrier: 'g','e','h','c','no type'.")        
        # add node to network
        net.add_node(node)
    
    # create links
    if link_to_loads:
        number_of_links = 2*n-m
    else:
        number_of_links = n-m

    if link_to_loads:
        if m:
            start_nodes = [net.nodes[0]] + net.nodes[n+1:2*n-m] + [n for n in net.nodes[n+1:n+m+1] for i in range(2)] + net.nodes[n+m+1:] # source + junctions to other junctions + junctions to two loads + junction to one load
            end_nodes = net.nodes[n+1:2*n-m+1] + net.nodes[1:n+1] #junctions from other junctions + loads
        else:
            start_nodes = [net.nodes[0]] + net.nodes[n+1:2*n] + net.nodes[n+1:2*n+1] #source + junctions to other junctions + junctions to loads
            end_nodes = net.nodes[n+1:2*n+1] + net.nodes[1:n+1] #junctions from other junctions + loads
    else:
        start_nodes = [net.nodes[0]] + net.nodes[1:n+1]
        end_nodes = net.nodes[1:]
    # add links to network
    for ind, [start_node, end_node] in enumerate(zip(start_nodes,end_nodes)):
        link_name = carrier+'l'+str(ind)
        if carrier == 'no_type':
            link = Link(link_name,start_node,end_node)
        elif carrier == 'g':
            link = GasLink(link_name,start_node,end_node)
        elif carrier == 'e':
            link = ElectricalLink(link_name,start_node,end_node)
        elif carrier == 'h':
            link = HeatLink(link_name,start_node,end_node)
        elif carrier == 'c':
            raise NotImplementedError("Only standard, gas network, electrical network, and heat network are currently implemented")
        else:
            raise ValueError("Enter a valid value for carrier: 'g','e','h','c','no type'.") 
        net.add_link(link)
                
    # create half links (there are n+1 half links, one for every load / sink, and one for the source)
    if m and (not link_to_loads): # more than one halflink per (some) nodes
        start_nodes_halflinks =  [net.nodes[0]] + [node for node in net.nodes[1:m+1] for _ in range(2)] + net.nodes[m+1:]
        halflink_names = [node.name + '_hl' + str(ind_hl) for node in net.nodes[1:m+1] for ind_hl in range(2)] + [node.name + '_hl' for node in [net.nodes[0]]+net.nodes[m+1:]]
    else: # one half link for every node
        start_nodes_halflinks = net.nodes[:n+1]
        halflink_names = [node.name + '_hl' for node in start_nodes_halflinks]
    # add half links to network and to the nodes (creating the half link automatically adds it to the node)
    for ind, [start_node_halflink, half_link_name] in enumerate(zip(start_nodes_halflinks,halflink_names)):
        if carrier == 'no_type':
            half_link = HalfLink(half_link_name,start_node_halflink)
        elif carrier == 'g':
            half_link = GasHalfLink(half_link_name,start_node_halflink,q=0.)
        elif carrier == 'e':
            half_link = ElectricalHalfLink(half_link_name,start_node_halflink,P=0.,Q=0.)
        elif carrier == 'h':
            if start_node_halflink.node_type in [0,8]: # for source reference slack nodes and load reference slack nodes a half link is automatically created when the node is made. But it is not added to the network yet.
                half_link = start_node_halflink.half_links[0]
                half_link.name = half_link_name
            else:
                half_link = HeatHalfLink(half_link_name,start_node_halflink,Tr=0.,dphi=0.,bc_type=3)
        elif carrier == 'c':
            raise NotImplementedError("Only standard, gas network, electrical network, and heat network are currently implemented")
        else:
            raise ValueError("Enter a valid value for carrier: 'g','e','h','c','no type'.") 
        net.add_half_link(half_link)

    return net

def combine_networks_as_tree(nets,net_name=None):
    """Combines single line networks to form a bigger network. The networks are all connected from the first node to one single (new) node. 
    
    Parameters
    ----------
    nets : list
        List of Network objects to be combined
    net_name : str, optional
        Name of the comined Network 
    
    Returns
    -------
    combined_net : Network
        The combined network. If all the networks are of the same carrier type, then the combined network will also be of type carrier. If not, the combined network will be heterogeneous, and the node used to connect the separate networks will be a coupling node. NB: combining the networks add links and possibly half links to the in and out links of nodes in the separate networks. These objects are therefore altered. 
    """       
    # create network with connection node (which will be first in the list of nodes of the combined network)
    if not net_name:
        net_name = 'Q0'
        
    if all(isinstance(net,GasNetwork) for net in nets):
        combined_net = GasNetwork(net_name)
        connection_node = GasNode('source',node_type=0) # refence node
        connection_half_link = GasHalfLink('source'+'_hl',connection_node,q=0.)
    elif all(isinstance(net,ElectricalNetwork) for net in nets):
        combined_net = ElectricalNetwork(net_name)
        connection_node = ElectricalNode('source',node_type=0) # refence node
        connection_half_link = ElectricalHalfLink('source'+'_hl',connection_node,P=0.,Q=0.)
    elif all(isinstance(net,HeatNetwork) for net in nets):
        combined_net = HeatNetwork(net_name)
        connection_node = HeatNode('source',node_type=0) # refence node (half link is automatically created)
        connection_half_link = connection_node.half_links[0]
    elif all(isinstance(net,Network) for net in nets):
        combined_net = Network(net_name)
        connection_node = Node('source')
        connection_half_link = HalfLink('source'+'_hl',connection_node)
    else:
        raise ValueError("nets must be a list of Network objects")
    combined_net.add_node(connection_node)
    combined_net.add_half_link(connection_half_link)
    
    # collect nodes to connect to of the separate networks, add network to bigger network, connect node in separate network to connection node
    connection_node_x = 0
    connection_node_y = 0
    for ind,net in enumerate(nets):
        combined_net.add_network(net)
        node = net.nodes[0]
        connection_node_x = min(connection_node_x, node.x)
        connection_node_y += node.y
        link_name = 'l'+str(ind)
        if isinstance(node,GasNode):
            link_name = 'g'+link_name
            link = GasLink(link_name,connection_node,node)
            if node.node_type == 0:
                node.node_type = 1 #change node type from reference node to load node, which functions as a junction
        elif isinstance(node,ElectricalNode):
            link_name = 'e'+link_name
            link = ElectricalLink(link_name,connection_node,node)
            if node.node_type == 0:
                node.node_type = 2 #change node type from slack node to load node, which functions as a junction
        elif isinstance(node,HeatNode):
            link_name = 'h'+link_name
            link = HeatLink(link_name,connection_node,node)
            if node.node_type == 0:
                node.node_type = 2 # change node type from slack node to a junction node
                for hl in node.get_half_links(): # a junction node doesn't have half links, so remove them
                    if hl in net.half_links:
                        net.remove_half_link(hl)
                        combined_net.remove_half_link(hl)
                    node.remove_half_link(hl)
        elif isinstance(node,Node):
            link = Link(link_name,connection_node,node)
        else:
            raise ValueError("Only standard and gas network are currently implemented")
        combined_net.add_link(link)
    # place new node to the left, and vertically in the middle of the separate networks
    connection_node.x = connection_node_x-1
    connection_node.y = connection_node_y/len(nets)
        
    return combined_net

def add_gas_data(net,p_ref,q,link_types,link_params):
    """Add network data for a gas network. The data are added to the Network objects, and to the Nodes, Links, and HalfLinks. Currently assumes a tree-like structure of the network as created by create_radial_line_network.
    
    Parameters
    ----------
    net : GasNetwork
        The gas network to which the data should be added
    p_ref : float
        Reference pressure of source node
    q : np array
        Array with outflow data for the load nodes. A flow bigger than 0 represents outflow, and smaller than 0 inflow. The length must be equal to the number of sinks in the network (is currently not checked). 
    link_types : list
        List with link types for the links
    link_params : list
        List with link parameter dictionaries for the links 
    """
    if not isinstance(net,GasNetwork):
        raise ValueError("net must be a GasNetwork instance.")
    
    if not len(link_types) == len(net.links):
        raise ValueError("link_types must be the same lenght as the number of links in the network, i.e. must be lenght {}".format(len(net.links)))
    if not len(link_params) == len(net.links):
        raise ValueError("link_params must be the same lenght as the number of links in the network, i.e. must be lenght {}".format(len(net.links)))
    
    # add node data (assumes the tree-like structure of the network as created by create_radial_line_network)
    net.nodes[0].p = p_ref
    load_ind = 0
    for n in net.get_nodes([1]):
        if load_ind < len(q): # sinks
            if n.half_links:
                for ind_hl,hl in enumerate(n.get_half_links()):
                    hl.q = q[load_ind]
                    load_ind += 1
                    if not hl in net.half_links:
                        net.add_half_link(hl)
            else:
                GasHalfLink(n.name+'_hl',n,q=q[load_ind])
                net.add_half_link(halflink)
                load_ind += 1
        else: # junctions
            if n.half_links: 
                hl = n.half_links[0]
                hl.q = 0.
                if not hl in net.half_links:
                    net.add_half_link(hl)
            else: # junctions
                halflink = GasHalfLink(n.name+'_hl',n,q=0.)
                net.add_half_link(halflink)
        
    # add link data
    for ind_e,e in enumerate(net.get_links()):
        e.set_type(link_types[ind_e],link_params[ind_e],link_eq_form=e.link_eq_form)
        
def add_elec_data(net,V_ref,delta_ref,P,Q,link_types,link_params):
    """Add network data for an electrical network. The data are added to the Network objects, and to the Nodes, Links, and HalfLinks. Currently assumes a tree-like structure of the network as created by create_radial_line_network.
    
    Parameters
    ----------
    net : ElecticalNetwork
        The gas network to which the data should be added
    V_ref : float
        Reference voltage magnitude of source node
    delta_ref : float
        Reference voltage angle of source node
    P : np array
        Array with active power data for the load nodes. A flow bigger than 0 represents outflow, and smaller than 0 inflow. The length must be equal to the number of sinks in the network (is currently not checked). 
    Q : np array
        Array with reactive power data for the load nodes. A flow bigger than 0 represents outflow, and smaller than 0 inflow. The length must be equal to the number of sinks in the network (is currently not checked). 
    link_types : list
        List with link types for the links
    link_params : list
        List with link parameter dictionaries for the links
    """
    if not isinstance(net,ElectricalNetwork):
        raise ValueError("net must be a ElecNetwork instance.")
    
    if not len(link_types) == len(net.links):
        raise ValueError("link_types must be the same lenght as the number of links in the network, i.e. must be lenght {}".format(len(net.links)))
    if not len(link_params) == len(net.links):
        raise ValueError("link_params must be the same lenght as the number of links in the network, i.e. must be lenght {}".format(len(net.links)))
    
    # add node data (assumes the tree-like structure of the network as created by create_radial_line_network)
    net.nodes[0].V = V_ref
    net.nodes[0].delta = delta_ref
    load_ind = 0
    for n in net.get_nodes([2]):
        if load_ind < len(P): # sinks
            if n.half_links:
                for ind_hl,hl in enumerate(n.get_half_links()):
                    hl.P = P[load_ind]
                    hl.Q = Q[load_ind]
                    load_ind += 1
                    if not hl in net.half_links:
                        net.add_half_link(hl)
            else:
                ElectricalHalfLink(n.name+'_hl',n,P=P[load_ind],Q=Q[load_ind])
                net.add_half_link(halflink)
                load_ind += 1
        else: # junctions
            if n.half_links: 
                hl = n.half_links[0]
                hl.P = 0.
                hl.Q = 0.
                if not hl in net.half_links:
                    net.add_half_link(hl)
            else: # junctions
                halflink = ElectricalHalfLink(n.name+'_hl',n,P=0.,Q=0.)
                net.add_half_link(halflink)
        
    # add link data
    for ind_e,e in enumerate(net.get_links()):
        e.set_type(link_types[ind_e],link_params[ind_e])    
        
def add_heat_data(net,p_ref,Ts_ref,phi,Ts_hl,Tr_hl,dT,link_types,link_params,halflink_types,halflink_params,halflink_bc_types):
    """Add network data for a heat network. The data are added to the Network objects, and to the Nodes, Links, and HalfLinks. Currently assumes a tree-like structure of the network as created by create_radial_line_network.
    
    Parameters
    ----------
    net : ElecticalNetwork
        The gas network to which the data should be added
    p_ref : float
        Reference pressure of source node
    Ts_ref : float
        Reference supply temperature of source node
    phi : np array
        Array with heat power data for the load nodes. A flow bigger than 0 represents outflow, and smaller than 0 inflow. The length must be equal to the number of sinks in the network (is currently not checked). 
    Ts_hl : np array
        Array with outflow temperatures for the load nodes, near supply line. The length must be equal to the number of sinks in the network (is currently not checked). 
    Tr_hl : np array
        Array with outflow temperatures for the load nodes, near return line. The length must be equal to the number of sinks in the network (is currently not checked). 
    dT : np array
        Array with temperature differences over load. The length must be equal to the number of sinks in the network (is currently not checked). 
    link_types : list
        List with link types for the links
    link_params : list
        List with link parameter dictionaries for the links
    halflink_types : list
        List with half link types for the half links
    halflink_params : list
        List with half link parameter dictionaries for the half links
    halflink_bc_types : list
        List with boundary condition types for the half links
    """
    if not isinstance(net,HeatNetwork):
        raise ValueError("net must be a HeatNetwork instance.")
    
    if not len(link_types) == len(net.links):
        raise ValueError("link_types must be the same lenght as the number of links in the network, i.e. must be lenght {}".format(len(net.links)))
    if not len(link_params) == len(net.links):
        raise ValueError("link_params must be the same lenght as the number of links in the network, i.e. must be lenght {}".format(len(net.links)))
    
    # add node data (assumes the tree-like structure of the network as created by create_radial_line_network)
    net.nodes[0].p = p_ref
    net.nodes[0].Ts = Ts_ref
    load_ind = 0
    for n in net.get_nodes(): 
        if n.node_type in [0,1,3,8,9,10,11,12,13]: # sinks and sources (i.e. not a junction)
            if n.half_links:
                for ind_hl,hl in enumerate(n.get_half_links()):
                    hl.dphi = phi[load_ind]
                    hl.Ts = Ts_hl[load_ind]
                    hl.Tr = Tr_hl[load_ind]
                    hl.dT = dT[load_ind]
                    if not hl in net.half_links:
                        net.add_half_link(hl)
                    if not hl.link_type == halflink_types[load_ind]:
                        hl.set_type(halflink_types[load_ind],halflink_params[load_ind],bc_type=halflink_bc_types[load_ind])
                    load_ind += 1
            else:
                hl = HeatHalfLink(n.name+'_hl',n,dphi=phi[load_ind],Ts=Ts_hl[load_ind],Tr=Tr_hl[load_ind],dT=dT[load_ind],bc_type=halflink_bc_types[load_ind])
                net.add_half_link(halflink)
                load_ind += 1
        else: # junctions
            if n.half_links: 
                for hl in n.half_links:
                    net.remove_half_link(hl)
                    n.remove_half_link(hl)
        
    # add link data
    for ind_e,e in enumerate(net.get_links()):
        e.set_type(link_types[ind_e],link_params[ind_e],hydr_eq_form=e.hydr_eq_form)    
