"""Create an electrical network consisting of 1 city, which consists of (hierarchically) nD districts, nQ quarters, and nS streets of n loads each"""
from meslf.networks.create_network import create_radial_line_network, combine_networks_as_tree, add_elec_data
from meslf.networks.read_write_network import to_pd_dataframes
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.utils.constants import kW, MW, kV, km, cm, mm
import numpy as np
import os 
import argparse 

command_line_input = argparse.ArgumentParser()
command_line_input.add_argument(
    "--n", # name on the command line Drop the `--` for positional/required parameters
    type=int,
    default = 10, # default if nothing is provided
    )
command_line_input.add_argument(
    "--m", # name on the command line Drop the `--` for positional/required parameters
    type=int,
    default = 5, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nS", # name on the command line Drop the `--` for positional/required parameters
    type=int,
    default = 4, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nQ", # name on the command line Drop the `--` for positional/required parameters
    type=int,
    default = 4, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nD", # name on the command line Drop the `--` for positional/required parameters
    type=int,
    default = 5, # default if nothing is provided
    )

V_ref_C = 50*kV #[V]
V_ref_D = 10*kV #[V]
V_ref_Q = 10*kV #[V]
V_ref_S = .4*kV #[V]

ratio_SQ = V_ref_Q/V_ref_S
ratio_DC = V_ref_C/V_ref_D

def create_line_data(L,D):
    """Create the needed data for transmission lines.
    
    Parameters
    ----------
    L : float
        Length of the line in m
    D : float
        Diameter of the line in m
        
    Returns
    -------
    b : float
        b
    g : float
        g
    b_sh : float
        b_sh
    """
    #x = 0.2 #[Ohm/km]
    #r = 0.1 #[Ohm/km]
    c = 0#100 #[nF/km]
    f = 50 #[Hz]
    #x_line = x*L/km #[Ohm]
    #r_line = r*L/km #[Ohm]
    rho = 1.6e-8 #[Ohm m]
    r_line = 4*rho*L/(np.pi*D**2) #[Ohm]
    x_line = 10*r_line #[Ohm]
    #r_line = 0# DC power flow
    z_abs2_line = r_line**2 + x_line**2
    b = -x_line/(z_abs2_line) 
    g = r_line/(z_abs2_line)
    b_sh = 2*np.pi*f*c*1e-9*L/km
    return b,g,b_sh
    
    
def create_streets(n,m,number_of_streets):
    """Create a list of street networks
    
    Parameters
    ----------
    n : int
        number of loads in the street
    m : int
        number of junctions with two loads
    number_of_streets : int
        number of street networks to be created
        
    Returns 
    -------
    streets : list
        list with ElectricalNetwork objects
    """
    streets = list()
    # node data
    V_ref = 0.99*V_ref_Q #[V] voltage before trafo (only 99%, because you want it to be lower than the voltage in the rest of the quarters).
    V_ref_low = V_ref_S #[V] voltage after trafo
    delta_ref = 0. #[rad]
    # half link data
    np.random.seed(0)
    S_min = 0.2*kW #[W]
    S_var = 0.2*kW #[W]
    y_shift = 0
    for i in range(number_of_streets):
        # topology
        street = create_radial_line_network(n,carrier='e',m=m,net_name='S'+str(i))

        # set initial data for voltage amplitude (flat initialization)
        for ind_n,node in enumerate(street.get_nodes(node_types=[1,2])):
            node.V = V_ref_low
            
        # add extra node connected to trafo, and add the trafo
        source_node = street.nodes[0]
        street.add_node(ElectricalNode('ene'+str(i),x=source_node.x-1,y=source_node.y),position=0)
        street.add_link(ElectricalLink('elt'+str(i),street.nodes[0],source_node))
        
        # link data
        L_S = 100 #[m]
        D_S = 3*mm#1*mm #[m]
        b_S,g_S,b_sh_S = create_line_data(L_S,D_S)
        L_junc_to_load = 10 #[m]
        D_junc_to_load = 1*mm #[m]
        b_junc_to_load,g_junc_to_load,b_sh_junc_to_load = create_line_data(L_junc_to_load,D_junc_to_load)
        L_trafo = 1 #[m]
        D_trafo = 1*mm #[m]
        b_trafo,g_trafo,b_sh_trafo = create_line_data(L_trafo,D_trafo)
        # links between junctions (and source) + links between junctions and loads + trafo link 
        link_params = [{'b':b_S,'g':g_S}]*(n-m) + [{'b':b_junc_to_load,'g':g_junc_to_load}]*n + [{'b':b_trafo,'g':g_trafo,'b_sh':b_sh_trafo,'g_sh':0,'ratio':ratio_SQ,'phase_shift':0}]
        #link_params = [{'b':0.0625,'g':-0.00625}]*(n-m) + [{'b':0.0625,'g':-0.00625}]*n + [{'b':0.0625,'g':-0.00625,'b_sh':b_sh_trafo,'g_sh':0,'ratio':ratio_SQ,'phase_shift':0}]
        link_types = ['short_line']*(2*n-m) + ['pi_line_trafo'] 
        
        # half link data
        P_inj = np.random.rand(n)*S_var + S_min
        Q_inj = np.random.rand(n)*S_var + S_min
        
        # add data (sets half link data, and also the voltage for the extra node)
        add_elec_data(street,V_ref,delta_ref,P_inj,Q_inj,link_types,link_params)
        
        # adjust y coordinates and rename nodes
        for node in street.get_nodes():
            node.y += y_shift
            node.name += street.name
        y_shift += 3
        
        # rename links
        for link in street.get_links():
            link.name += street.name
            
        # rename source node
        street.nodes[1].name = 's'+street.name
        
        if source_node.node_type == 0:
            source_node.node_type = 2 #change node type from reference node to load node, which functions as a junction
            source_node.V = V_ref_low #set initial data for voltage amplitude
            
        streets.append(street)
    
    return streets
    
def create_quarters(streets):
    """Create a list of quarter networks. 
    
    Parameters
    ----------
    streets : list
        List of list of street networks. For every list of streets, a quarter network is created consisting of those streets
    """
    quarters = list()
    y_shift = 0
    for i,list_of_streets in enumerate(streets):
        # topology
        quarter = combine_networks_as_tree(list_of_streets,net_name = 'Q'+str(i))
        
        # link data (for links from quarter source to sources of streets)
        L_Q = 1*km #[m]
        D_Q = 1*cm #[m]
        b_Q,g_Q,b_sh_Q = create_line_data(L_Q,D_Q)
        link_type = 'short_line'
        link_params = {'b':b_Q,'g':g_Q}
        #link_params = {'b':.001,'g':-.0001}
        for link in quarter.nodes[0].get_out_links():
            if link in quarter.links: # not a half link
                link.set_type(link_type,link_params)
        
        # adjust y coordinates
        for node in quarter.get_nodes():
            node.y += y_shift
            node.name += quarter.name
        y_shift += 3 * len(list_of_streets) 
        
        # rename links
        for link in quarter.get_links():
            link.name += quarter.name
            
        # rename source node
        source_node = quarter.nodes[0]
        source_node.name = 's'+quarter.name
        
        # set initial data for voltage amplitude (flat initialization)
        source_node.V = V_ref_Q
        if source_node.node_type == 0:
            source_node.node_type = 2 #change node type from reference node to load node, which functions as a junction
        
        quarters.append(quarter)

    return quarters

def create_districts(quarters):
    """Create a list of district networks. 
    
    Parameters
    ----------
    quarters : list
        List of list of quarter networks. For every list of quarters, a district network is created consisting of those quarters.
    """
    districts = list()
    y_shift = 0
    for i,list_of_quarters in enumerate(quarters):
        # topology
        district = combine_networks_as_tree(list_of_quarters,net_name = 'D'+str(i))
        quarter_source_nodes = [quarter.nodes[0] for quarter in list_of_quarters]
        
        # add extra node connected to trafo, and add the trafo
        source_node = district.nodes[0]
        if source_node.node_type == 0:
            source_node.node_type = 2 #change node type from reference node to load node, which functions as a junction
        district.add_node(ElectricalNode('ene'+str(i),x=source_node.x-1,y=source_node.y),position=0)
        L_trafo = 1 #[m]
        D_trafo = 1*mm #[m]
        b_trafo,g_trafo,b_sh_trafo = create_line_data(L_trafo,D_trafo)
        district.add_link(ElectricalLink('elt'+str(i),district.nodes[0],source_node,link_type='pi_line_trafo',link_params={'b':b_trafo,'g':g_trafo,'b_sh':b_sh_trafo,'g_sh':0,'ratio':ratio_DC,'phase_shift':0}))
        #district.add_link(ElectricalLink('elt'+str(i),district.nodes[0],source_node,link_type='pi_line_trafo',link_params={'b':.01,'g':-.001,'b_sh':b_sh_trafo,'g_sh':0,'ratio':ratio_DC,'phase_shift':0}))
        
        # link data (for links from district source to sources of quarters)
        L_D = 2*km #[m]
        D_D = 1*cm #[m]
        b_D,g_D,b_sh_D = create_line_data(L_D,D_D)
        link_type = 'short_line'
        link_params = {'b':b_D,'g':g_D}
        #link_params = {'b':.01,'g':-.001}
        for link in source_node.get_out_links():
            if link in district.links: # not a half link
                link.set_type(link_type,link_params)
            
        # links between quarter sources
        for ind_n, node in enumerate(quarter_source_nodes[:-1]):
            district.add_link(ElectricalLink('el'+str(len(quarters)+ind_n),node,quarter_source_nodes[ind_n+1],link_type=link_type,link_params=link_params))
            # set initial data for voltage amplitude (flat initialization)
            node.V = V_ref_Q
            if node.node_type == 0:
                node.node_type = 2 #change node type from reference node to load node, which functions as a junction
                
        # adjust y coordinates and rename nodes
        for node in district.get_nodes():
            node.y += y_shift
            node.name += district.name
        for quarter in list_of_quarters:
            y_shift += 3*len(quarter.networks)
        
        # rename links
        for link in district.get_links():
            link.name += district.name
            
        # rename source node
        source_node.name = 's'+district.name
        # set initial data for voltage amplitude
        source_node.V = V_ref_D
        
        districts.append(district)
    
    return districts

def create_city(districts):
    """Creates one (!) city network.
    
    Parameters
    ----------
    districts : list
        List of district networks.
        
    Returns
    -------
    city : ElectricalNetwork
        The city network
    """
    district_trafo_nodes = [district.nodes[0] for district in districts]
    
    city = ElectricalNetwork('C0')
    connection_node = ElectricalNode('source',node_type=0) # refence node
    connection_half_link = ElectricalHalfLink('source'+'_hl',connection_node,P=0.,Q=0.)
    city.add_node(connection_node)
    city.add_half_link(connection_half_link)
    nD = len(districts)
    
    connection_node_x = 0
    connection_node_y = 0
    ind_link = 0
    for ind,district in enumerate(districts):
        city.add_network(district)
        node = district.nodes[0]
        node.V = .99*V_ref_C # set initial data for voltage amplitude (these are the nodes before the trafo's)
        connection_node_x = min(connection_node_x, node.x)
        connection_node_y += node.y
        if ind == 0 or ind == round(nD/2) or ind == nD-1:
            link_name = 'l'+str(ind)
            link_name = 'e'+link_name
            link = ElectricalLink(link_name,connection_node,node)
            if node.node_type == 0:
                node.node_type = 2 #change node type from reference node to load node, which functions as a junction
            city.add_link(link)
            ind_link += 1
    # place new node to the left, and vertically in the middle of the separate networks
    connection_node.x = connection_node_x-1
    connection_node.y = connection_node_y/nD
    
    # link data (for links from city source to sources of districts, and for link between district sources)
    L_C = 5*km #[m]
    D_C = 1*cm #[m]
    b_C,g_C,b_sh_C = create_line_data(L_C,D_C)
    link_type = 'pi_line'
    link_params = {'b':b_C,'g':g_C,'b_sh':b_sh_C,'g_sh':0}
    #link_params = {'b':.004,'g':-.0004,'b_sh':b_sh_C,'g_sh':0}
    for link in city.nodes[0].get_out_links():
        if link in district.links: # not a half link
            link.set_type(link_type,link_params)
                
    # links between district sources (or, more precisely, between the extra trafo nodes connected to the district sources)
    for ind_n, node in enumerate(district_trafo_nodes[:-1]):
        city.add_link(ElectricalLink('el'+str(ind_link),node,district_trafo_nodes[ind_n+1],link_type=link_type,link_params=link_params))
        ind_link += 1
        
    # rename source node
    city_source = city.nodes[0]
    city_source.name = 's'+city.name
    # set initial data for voltage amplitude
    city_source.V = V_ref_C
    
    return city

def create_network(n,m,nS,nQ,nD):
    """Create the complete city network
    
    Parameters
    ----------
    n : int
        number of loads, for each street
    m : int
        number of junctions with two loads, for each street
    nS : int
        number of street networks
    nQ : int
        number of quarter networks
    nD : int
        number of district networks
        
    Returns
    -------
    nodes : pd DataFrame
        Node data
    links : pd DataFrame
        Link data
    halflinks: pd DataFrame
        Halflink data
    """
    quarters = list()
    for d in range(nD):
        quarters_per_district = list()
        streets = list()
        for i in range(nQ):
            streets.append(create_streets(n,m,nS))
        quarters_per_district = create_quarters(streets)
        quarters.append(quarters_per_district)
    districts = (create_districts(quarters))
    city_net = create_city(districts)
    
    L_C = 5*km #[m]
    D_C = 1*cm #[m]
    b_C,g_C,b_sh_C = create_line_data(L_C,D_C)
    for link in city_net.get_links():
        if link.start_node == city_net.nodes[0]:
            link.set_type('pi_line',{'b':b_C,'g':g_C,'b_sh':b_sh_C,'g_sh':0})
            #link.set_type('pi_line',{'b':.004,'g':-.0004,'b_sh':b_sh_C,'g_sh':0})
    
    # rotate network
    for node in city_net.get_nodes():
        node.x, node.y = node.y, -node.x 
        
    nodes, links, halflinks = to_pd_dataframes(city_net)
    return nodes, links, halflinks 

def main(n,m,nS,nQ,nD,dir_path):
    """Create the network, and write to pd dataframes. """
    
    # creat network
    nodes, links, halflinks = create_network(n,m,nS,nQ,nD)
    
    # save data
    nodes.to_pickle(os.path.join(dir_path,'EN_1C_nodes.pkl'))
    links.to_pickle(os.path.join(dir_path,'EN_1C_links.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'EN_1C_halflinks.pkl'))
    
if __name__== '__main__':
    # parse the command line
    args = command_line_input.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(args.n,args.m,args.nS,args.nQ,args.nD,dir_path)

