"""Create a gasnetwork consisting of 1 city, which consists of (hierarchically) nD districts, nQ quarters, and nS streets of n loads each"""
from meslf.networks.create_network import create_radial_line_network, combine_networks_as_tree, add_gas_data
from meslf.networks.read_write_network import to_pd_dataframes
from meslf.networks.carrier import Gas
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.utils.constants import cm, km, mbar, bar
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
command_line_input.add_argument(
    "--p_low", # name on the command line Drop the `--` for positional/required parameters
    type=float,
    default = .98, # default if nothing is provided
    )
command_line_input.add_argument(
    "--p_high", # name on the command line Drop the `--` for positional/required parameters
    type=float,
    default = .99, # default if nothing is provided
    )

def create_streets(n,m,number_of_streets,carrier,p_init_low,p_init_high,hydr_eq='dp_of_q'):
    """Create a list of street networks
    
    Parameters
    ----------
    n : int
        number of loads in the street
    m : int
        number of junctions with two loads
    number_of_streets : int
        number of street networks to be created
    carrier : Carrier
        the gas Carrier object used for every network
        
    Returns 
    -------
    streets : list
        list with GasNetwork objects
    """
    streets = list()
    # node data
    p_ref = 0.8*100*mbar #[Pa] pressure before valve (only 80%, because you want it to be lower than the pressure in the rest of the quarters.)
    p_ref_low = 30*mbar #[Pa] pressure after valve
    # link data
    D_unit = 'm'
    L_unit = 'm'
    # half link data
    np.random.seed(0)
    q_min = 5e-3#0.05 #[kg/s]
    q_var = 2e-3#0.02 #[kg/s]
    q_unit = 'kg/s'
    y_shift = 0
    for i in range(number_of_streets):
        # topology
        street = create_radial_line_network(n,carrier='g',m=m,net_name='S'+str(i))
            
        # set initial data for pressure
        p_init = p_ref_low*np.linspace(p_init_high,p_init_low,len(street.nodes))
        for ind_n,node in enumerate(street.get_nodes(node_types=[1])):
            node.p = p_init[ind_n]
            
        # add extra node connected to valve, and add the valve
        source_node = street.nodes[0]
        if source_node.node_type == 0:
            source_node.node_type = 1 #change node type from reference node to load node, which functions as a junction
        source_node.p = p_ref_low
        street.add_node(GasNode('gne'+str(i),x=source_node.x-1,y=source_node.y),position=0) # reference pressure is set with the add_gas_data
        street.add_link(GasLink('glv'+str(i),street.nodes[0],source_node,link_eq_form=hydr_eq))
        
        # link data
        D = 5*cm
        # links between junctions (and source) + links between junctions and loads + valve link 
        link_params = [{'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':100,'L_unit':L_unit}]*(n-m) + [{'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':10,'L_unit':L_unit}]*n + [{'r':p_ref_low/p_ref}]
        link_types = ['pipe_low_pres_pole']*(2*n-m) + ['compressor']
        
        # half link data
        q_inj_source = np.array([0])
        q_inj = np.concatenate((q_inj_source,np.random.rand(n)*q_var + q_min))
        
        # add data
        add_gas_data(street,p_ref,q_inj,link_types,link_params)
        
        # adjust y coordinates, and rename nodes
        for ind_n,node in enumerate(street.get_nodes()):
            node.y += y_shift
            node.name += street.name
        y_shift += 3
        
        # rename links
        for link in street.get_links():
            link.name += street.name
            # set link equation formulation to fb (because the ones created with create_radial_line_network use fa)
            link.set_type(link.link_type,link.link_params,link_eq_form=hydr_eq)
        
        # rename source node
        street.nodes[1].name = 's'+street.name
        
        streets.append(street)
    
    return streets
    
def create_quarters(streets,carrier,hydr_eq='dp_of_q'):
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
        L = 1*km
        D = 10*cm
        link_type = 'pipe_low_pres_pole'
        link_params = {'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit}
        for link in quarter.nodes[0].get_out_links():
            if link in quarter.links: # not a half link
                link.set_type(link_type,link_params,link_eq_form=hydr_eq)
        
        # adjust y coordinates, and rename nodes
        for node in quarter.get_nodes():
            node.y += y_shift
            node.name += quarter.name
        y_shift += 3 * len(list_of_streets) 
        
        # rename links
        for link in quarter.get_links():
            link.name += quarter.name
            # set link equation formulation to fb (because the ones created with combine_networks_as_tree use fa)
            link.set_type(link.link_type,link.link_params,link_eq_form=hydr_eq)
        # rename source node
        source_node = quarter.nodes[0]
        source_node.name = 's'+quarter.name
        # set initial guess for pressure
        source_node.p = 100*mbar
        if source_node.node_type == 0:
            source_node.node_type = 1 #change node type from reference node to load node, which functions as a junction
    
        quarters.append(quarter)

    return quarters

def create_districts(quarters,carrier,p_init_low,p_init_high,hydr_eq='dp_of_q'):
    """Create a list of district networks. 
    
    Parameters
    ----------
    quarters : list
        List of list of quarter networks. For every list of quarters, a district network is created consisting of those quarters.
    """
    districts = list()
    y_shift = 0
    p_ref_quarters = 100*mbar #[Pa]
    p_ref_district = 100*mbar #[Pa]
    for i,list_of_quarters in enumerate(quarters):
        # topology
        district = combine_networks_as_tree(list_of_quarters,net_name = 'D'+str(i))
        quarter_source_nodes = [quarter.nodes[0] for quarter in list_of_quarters]
        p_init = p_ref_quarters*np.linspace(p_init_high,p_init_low,len(quarter_source_nodes))
        
        # add extra node connected to valve, and add the valve
        source_node = district.nodes[0]
        if source_node.node_type == 0:
            source_node.node_type = 1 #change node type from reference node to load node, which functions as a junction
        district.add_node(GasNode('gne'+str(i),x=source_node.x-1,y=source_node.y,p=8*bar),position=0)
        district.add_link(GasLink('glv'+str(i),district.nodes[0],source_node,link_type='compressor',link_params={'r':0.0125},link_eq_form=hydr_eq)) #only some length for the plotting
        
        # link data (for links from district source to sources of quarters)
        D_unit = 'm'
        L_unit = 'm'
        L = 2*km
        D = 10*cm
        link_type = 'pipe_low_pres_pole'
        link_params = {'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit}
        for link in source_node.get_out_links():
            if link in district.links: # not a half link
                link.set_type(link_type,link_params,link_eq_form=hydr_eq)
            
        # links between quarter sources
        for ind_n, node in enumerate(quarter_source_nodes[:-1]):
            district.add_link(GasLink('gl'+str(len(quarters)+ind_n),node,quarter_source_nodes[ind_n+1],link_type=link_type,link_params=link_params,link_eq_form=hydr_eq))
            node.p = p_init[ind_n]
            if node.node_type == 0:
                node.node_type = 1 #change node type from reference node to load node, which functions as a junction
        quarter_source_nodes[-1].p = p_init[-1]
        if quarter_source_nodes[-1].node_type == 0:
            quarter_source_nodes[-1].node_type = 1 #change node type from reference node to load node, which functions as a junction
        
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
        source_node.p = p_ref_district
    
        districts.append(district)
    
    return districts

def create_city(districts,carrier,p_init_low,p_init_high,hydr_eq='dp_of_q'):
    """Creates one (!) city network.
    
    Parameters
    ----------
    districts : list
        List of district networks.
        
    Returns
    -------
    city : GasNetwork
        The city network
    """
    district_valve_nodes = [district.nodes[0] for district in districts]
    p_ref_district = 8*bar #[Pa] before valve
    p_init = p_ref_district*np.linspace(p_init_high,p_init_low,len(district_valve_nodes))
    city = GasNetwork('C0')
    connection_node = GasNode('source',node_type=0) # refence node
    connection_half_link = GasHalfLink('source'+'_hl',connection_node,q=0.)
    city.add_node(connection_node)
    city.add_half_link(connection_half_link)
    nD = len(districts)
    
    connection_node_x = 0
    connection_node_y = 0
    ind_link = 0
    for ind,district in enumerate(districts):
        city.add_network(district)
        node = district.nodes[0]
        node.p = p_init[ind] # set initial guess for pressure
        connection_node_x = min(connection_node_x, node.x)
        connection_node_y += node.y
        if ind == 0 or ind == round(nD/2) or ind == nD-1:
            link_name = 'l'+str(ind)
            link_name = 'g'+link_name
            link = GasLink(link_name,connection_node,node,link_eq_form=hydr_eq)
            city.add_link(link)
            ind_link += 1
        if node.node_type == 0:
            node.node_type = 1 #change node type from reference node to load node, which functions as a junction
    # place new node to the left, and vertically in the middle of the separate networks
    connection_node.x = connection_node_x-1
    connection_node.y = connection_node_y/nD
    
    # link data (for links from city source to sources of districts, and for link between district sources)
    D_unit = 'm'
    L_unit = 'm'
    L = 5*km
    D = 15*cm
    E = 0.98 #efficiency factor
    link_type = 'pipe_high_pres_weymouth'
    link_params = {'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit,'E':E}
    for link in city.nodes[0].get_out_links():
        if link in district.links: # not a half link
            link.set_type(link_type,link_params,link_eq_form=hydr_eq)
                
    # links between district sources (or, more precisely, between the extra valve nodes connected to the district sources)
    for ind_n, node in enumerate(district_valve_nodes[:-1]):
        city.add_link(GasLink('gl'+str(ind_link),node,district_valve_nodes[ind_n+1],link_type=link_type,link_params=link_params,link_eq_form=hydr_eq))
        #node.p = p_init[ind_n]
        ind_link += 1
    #district_valve_nodes[-1].p = p_init[-1]
    
    # asign reference pressure to new source node
    p_ref = 8*bar #[Pa]
    city_source = city.nodes[0]
    city_source.p = p_ref
    # rename source node
    city_source.name = 's'+city.name
    
    return city

def create_network(n,m,nS,nQ,nD,carrier,p_init_low,p_init_high,hydr_eq='dp_of_q'):
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
    carrier : GasCarrier
        The gas through the network
        
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
            streets.append(create_streets(n,m,nS,carrier,p_init_low,p_init_high,hydr_eq=hydr_eq))
        quarters_per_district = create_quarters(streets,carrier,hydr_eq=hydr_eq)
        quarters.append(quarters_per_district)
    districts = (create_districts(quarters,carrier,p_init_low,p_init_high,hydr_eq=hydr_eq))
    city_net = create_city(districts,carrier,p_init_low,p_init_high,hydr_eq=hydr_eq)
    
    for link in city_net.get_links():
        if link.start_node == city_net.nodes[0]:
            link.set_type('pipe_high_pres_weymouth',{'carrier':carrier, 'D':10*cm, 'D_unit':'m','L':5*km,'L_unit':'m','E':.98},link_eq_form=hydr_eq)
            #link.set_type('pipe_low_pres_pole',{'carrier':carrier, 'D':10*cm, 'D_unit':'m','L':2*km,'L_unit':'m'},link_eq_form=hydr_eq)
    
    # rotate network
    for node in city_net.get_nodes():
        node.x, node.y = node.y, -node.x 
        
    nodes, links, halflinks = to_pd_dataframes(city_net)
    return nodes, links, halflinks 

def main(n,m,nS,nQ,nD,dir_path,p_init_low,p_init_high,hydr_eq='dp_of_q'):
    """Create the network, and write to pd dataframes. """
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    T = Tn
    Z = 1.
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('gas',S,R_air,Z,pn,Tn,T)   
    
    # creat network
    nodes, links, halflinks = create_network(n,m,nS,nQ,nD,gas,p_init_low,p_init_high,hydr_eq=hydr_eq)
    
    # save data
    nodes.to_pickle(os.path.join(dir_path,'GN_1C_nodes.pkl'))
    links.to_pickle(os.path.join(dir_path,'GN_1C_links.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'GN_1C_halflinks.pkl'))
    
if __name__== '__main__':
    # parse the command line
    args = command_line_input.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(args.n,args.m,args.nS,args.nQ,args.nD,dir_path,args.p_low,args.p_high)
    
    
    
    
    
        
    
        
        
