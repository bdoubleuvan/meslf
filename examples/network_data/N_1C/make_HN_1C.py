"""Create a heat network consisting of 1 city, which consists of (hierarchically) 5 districts, 4 quarters, and 10 streets of approximately 20 loads each"""
from meslf.networks.create_network import create_radial_line_network, combine_networks_as_tree, add_heat_data
from meslf.networks.read_write_network import to_pd_dataframes
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
from meslf.utils.constants import mm, cm, km, bar

import numpy as np
import os 

import argparse 

command_line_input = argparse.ArgumentParser()
command_line_input.add_argument(
    "--n", # number of loads, for each street
    type=int,
    default = 10, # default if nothing is provided
    )
command_line_input.add_argument(
    "--m", # number of junctions with two loads, for each street
    type=int,
    default = 5, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nS", # number of street networks
    type=int,
    default = 4, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nQ", # number of quarter networks
    type=int,
    default = 4, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nD", # number of district networks
    type=int,
    default = 5, # default if nothing is provided
    )
command_line_input.add_argument(
    "--Ta", # Ambient temperature of the heat network
    type=int,
    default = 0, # default if nothing is provided
    )
command_line_input.add_argument(
    "--p_low", # Lowest value of fraction of reference pressure that is used in the initial linear pressure profile
    type=float,
    default = .98, # default if nothing is provided
    )
command_line_input.add_argument(
    "--p_high", # Highest value of fraction of reference pressure that is used in the initial linear pressure profile
    type=float,
    default = .99, # default if nothing is provided
    )

# node data
p_ref = 9*bar #[Pa]
Ts_ref = 100. #[C]
def create_streets(n,m,number_of_streets,Ta,p_init_low,p_init_high,carrier,heat_load='outflow'):
    """Create a list of street networks
    
    Parameters
    ----------
    n : int
        number of loads in the street
    m : int
        number of junctions with two loads
    number_of_streets : int
        number of street networks to be created
    Ta : float
        Ambient temperature of the heat network
    p_init_low : float
        Lowest value of fraction of reference pressure that is used in the initial linear pressure profile.
    p_init_high : float
        Highest value of fraction of reference pressure that is used in the initial linear pressure profile.
        
        
    Returns 
    -------
    streets : list
        list with HeatNetwork objects
    """
    # carrier
    rho = carrier.rhon
    g = carrier.g
    
    streets = list()
    
    # link data
    L_unit = 'm'
    D_unit = 'm'
    lam = 0.2 #[W/(mK)]
    # half link data
    To = 50. #[C]
    np.random.seed(0)
    phi_min = 50. #[W]
    phi_var = 450. #[W]
    
    y_shift = 0
    for i in range(number_of_streets):
        # topology
        street = create_radial_line_network(n,carrier='h',m=m,net_name='S'+str(i))

        # set initial data for pressure
        p_init = p_ref*np.linspace(p_init_high,p_init_low,len(street.nodes))
        for ind_n,node in enumerate(street.get_nodes()):
            node.p = p_init[ind_n]
            if node.node_type in [1,3,4] and heat_load == 'delta':
                node.node_type = 12 # change node type from load to a load with known dT
            
        # add extra node connected to valve, and add the valve
        source_node = street.nodes[0]
        if source_node.node_type == 0:
            source_node.node_type = 2 #change node type from reference slack node to a junction
            for hl in source_node.get_half_links():
                street.remove_half_link(hl)
                source_node.remove_half_link(hl)
        street.add_node(HeatNode('hne'+str(i),node_type=2,x=source_node.x-1,y=source_node.y),position=0) # reference pressure is set with the add_heat_data
        street.add_link(HeatLink('hlv'+str(i),street.nodes[0],source_node))
        
        # link data
        D = 5*cm #[m]
        U = 1.3 #[W/(m^2K)] for pipes from junctions to loads
        r = 1
        # links between junctions (and source) + links between junctions and loads + valve link
        link_params = [{'L':100.,'L_unit':L_unit,'D':D,'D_unit':D_unit,'U':U/10,'carrier':carrier,'Ta':Ta}]*(n-m) + [{'L':10.,'L_unit':L_unit,'D':1*cm,'D_unit':D_unit,'U':U,'carrier':carrier,'Ta':Ta}]*n + [{'r':1,'carrier':carrier,'Ta':Ta}]
        #link_types = ['standard_pipe_low_pres_pole']*(2*n-m) + ['isolated_pump']
        link_types = ['isolated_pipe_low_pres_pole']*(2*n-m) + ['isolated_pump']
        
        # half link data
        phi_inj = np.random.rand(n)*phi_var + phi_min
        Tr_inj = To*np.ones(n) # sink
        Ts_inj = Ts_ref*np.ones(n) # need some value
        dT = Ts_ref - Tr_inj # only sink nodes
        halflink_types = ['heat_exchanger']*n
        halflink_params = [{'carrier':carrier}]*n
        halflink_bc_types = [3]*n # sinks with dphi and To known
        
        # add data
        add_heat_data(street,p_ref,Ts_ref,phi_inj,Ts_inj,Tr_inj,dT,link_types,link_params,halflink_types,halflink_params,halflink_bc_types)
        
        # adjust y coordinates, and rename nodes
        for node in street.get_nodes():
            node.y += y_shift
            node.name += street.name
        y_shift += 3
        
        # rename links
        for link in street.get_links():
            link.name += street.name
        
        # rename source node
        street.nodes[1].name = 's'+street.name
        
        streets.append(street)
    
    return streets
    
def create_quarters(streets,Ta,carrier):
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
        D_unit = 'm'
        L_unit = 'm'
        L = 1*km #[m]
        D = 10*cm #[m]
        U = .006 #[W/(m^2K)]
        #link_type = 'standard_pipe_low_pres_pole'
        link_type = 'isolated_pipe_low_pres_pole'
        link_params = {'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit,'U':U,'Ta':Ta}
        for link in quarter.nodes[0].get_out_links():
            if link in quarter.links: # not a half link
                link.set_type(link_type,link_params)
        
        # adjust y coordinates, and rename nodes
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
        # set initial guess for pressure
        source_node.p = p_ref
        if source_node.node_type == 0:
            source_node.node_type = 2 #change node type from reference node to a junction
            for hl in source_node.get_half_links():
                quarter.remove_half_link(hl)
                source_node.remove_half_link(hl)
            
        quarters.append(quarter)

    return quarters

def create_districts(quarters,Ta,p_init_low,p_init_high,carrier):
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
        p_init = p_ref*np.linspace(p_init_high,p_init_low,len(quarter_source_nodes))
        
        # add extra node connected to valve, and add the valve
        source_node = district.nodes[0]
        if source_node.node_type == 0:
            source_node.node_type = 2 #change node type from reference node to a junction
            for hl in source_node.get_half_links():
                district.remove_half_link(hl)
                source_node.remove_half_link(hl)
        district.add_node(HeatNode('hne'+str(i),x=source_node.x-1,y=source_node.y,p=p_init_high*p_ref),position=0)
        district.add_link(HeatLink('hlv'+str(i),district.nodes[0],source_node,link_type='isolated_pump',link_params={'r':1,'Ta':Ta,'carrier':carrier})) 
        
        # link data (for links from district source to sources of quarters)
        D_unit = 'm'
        L_unit = 'm'
        U = .006/2 #[W/(m^2K)]
        L = 2*km
        D = 10*cm
        #link_type = 'standard_pipe_low_pres_pole'
        link_type = 'isolated_pipe_low_pres_pole'
        link_params = {'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit,'U':U,'Ta':Ta}
        for link in source_node.get_out_links():
            if link in district.links: # not a half link
                link.set_type(link_type,link_params)
            
        # links between quarter sources, and set initial guess for pressures
        for ind_n, node in enumerate(quarter_source_nodes[:-1]):
            district.add_link(HeatLink('hl'+str(len(quarters)+ind_n),node,quarter_source_nodes[ind_n+1],link_type=link_type,link_params=link_params))
            node.p = p_init[ind_n]
            if node.node_type == 0:
                node.node_type = 2 #change node type from reference node to a junction
                for hl in node.get_half_links():
                    district.remove_half_link(hl)
                    node.remove_half_link(hl)
        quarter_source_nodes[-1].p = p_init[-1]
        if quarter_source_nodes[-1].node_type == 0:
            quarter_source_nodes[-1].node_type = 2 #change node type from reference node to a junction
            for hl in quarter_source_nodes[-1].get_half_links():
                district.remove_half_link(hl)
                quarter_source_nodes[-1].remove_half_link(hl)
        
        # adjust y coordinates, and rename nodes
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
        # set initial guess pressure
        source_node.p = p_ref # node after valve
        
        districts.append(district)
    
    return districts

def create_city(districts,Ta,carrier):
    """Creates one (!) city network.
    
    Parameters
    ----------
    districts : list
        List of district networks.
        
    Returns
    -------
    city : HeatNetwork
        The city network
    """
    district_valve_nodes = [district.nodes[0] for district in districts]
    
    city = HeatNetwork('C0')
    connection_node = HeatNode('source',node_type=0) # refence node
    connection_half_link = connection_node.half_links[0]
    city.add_node(connection_node)
    city.add_half_link(connection_half_link)
    nD = len(districts)
    
    connection_node_x = 0
    connection_node_y = 0
    ind_link = 0
    for ind,district in enumerate(districts):
        city.add_network(district)
        node = district.nodes[0]
        connection_node_x = min(connection_node_x, node.x)
        connection_node_y += node.y
        if ind == 0 or ind == round(nD/2) or ind == nD-1:
            link_name = 'l'+str(ind)
            link_name = 'h'+link_name
            link = HeatLink(link_name,connection_node,node)
            city.add_link(link)
            ind_link += 1
        if node.node_type == 0:
            node.node_type = 2 #change node type from reference node to a junction
            for hl in node.get_half_links():
                city.remove_half_link(hl)
                node.remove_half_link(hl)
    # place new node to the left, and vertically in the middle of the separate networks
    connection_node.x = connection_node_x-1
    connection_node.y = connection_node_y/nD
    
    # link data (for links from city source to sources of districts, and for link between district sources)
    D_unit = 'm'
    L_unit = 'm'
    U = .004/5 #[W/(m^2K)]
    L = 5*km
    D = 15*cm
    #link_type = 'standard_pipe_low_pres_pole'
    link_type = 'isolated_pipe_low_pres_pole'
    link_params = {'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit,'U':U,'Ta':Ta}
    for link in city.nodes[0].get_out_links():
        if link in district.links: # not a half link
            link.set_type(link_type,link_params)
                
    # links between district sources (or, more precisely, between the extra valve nodes connected to the district sources)
    for ind_n, node in enumerate(district_valve_nodes[:-1]):
        city.add_link(HeatLink('hl'+str(ind_link),node,district_valve_nodes[ind_n+1],link_type=link_type,link_params=link_params))
        ind_link += 1
    
    # asign reference pressure to new source node
    city_source = city.nodes[0]
    city_source.p = p_ref
    # rename source node
    city_source.name = 's'+city.name
    
    return city

def create_network(n,m,nS,nQ,nD,Ta,p_init_low,p_init_high,carrier,heat_load='outflow'):
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
            streets.append(create_streets(n,m,nS,Ta,p_init_low,p_init_high,carrier,heat_load=heat_load))
        quarters_per_district = create_quarters(streets,Ta,carrier)
        quarters.append(quarters_per_district)
    districts = (create_districts(quarters,Ta,p_init_low,p_init_high,carrier))
    city_net = create_city(districts,Ta,carrier)
    
    D_unit = 'm'
    L_unit = 'm'
    U = .004/5 #[W/(m^2K)]
    L = 5*km
    D = 15*cm
    #link_type = 'standard_pipe_low_pres_pole'
    link_type = 'isolated_pipe_low_pres_pole'
    link_params = {'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit,'U':U,'Ta':Ta}
    city_source = city_net.nodes[0]
    for link in city_net.get_links():
        if link.start_node == city_source:
            #link.set_type('standard_pipe_low_pres_pole',link_params)
            link.set_type('isolated_pipe_low_pres_pole',link_params)
    
    # set half link type
    city_source.half_links[0].set_type('heat_exchanger',{'carrier':carrier},bc_type=0) # city source is source, and slack
    
    # rotate network
    for node in city_net.get_nodes():
        node.x, node.y = node.y, -node.x 
        
    nodes, links, halflinks = to_pd_dataframes(city_net)
    return nodes, links, halflinks 

def main(n,m,nS,nQ,nD,Ta,p_init_low,p_init_high,dir_path,heat_load='outflow'):
    """Create the network, and write to pd dataframes. """
    # carrier
    rho = 960. #[kg/m^3]
    Cp = 4.182e3 #[J/(kg K)]
    mu = 0.294e-6 #[m^2/s]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp,rho=rho,mu=mu)
    
    # creat network
    nodes, links, halflinks = create_network(n,m,nS,nQ,nD,Ta,p_init_low,p_init_high,water,heat_load=heat_load)
    
    # save data
    nodes.to_pickle(os.path.join(dir_path,'HN_1C_nodes.pkl'))
    links.to_pickle(os.path.join(dir_path,'HN_1C_links.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'HN_1C_halflinks.pkl'))
    
if __name__== '__main__':  
    args = command_line_input.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(args.n,args.m,args.nS,args.nQ,args.nD,args.Ta,args.p_low,args.p_high,dir_path)
