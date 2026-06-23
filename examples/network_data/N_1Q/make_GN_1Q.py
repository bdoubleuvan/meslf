"""Create a gasnetwork consisting of 1 quarter, which consists of 10 streets of approximately 20 loads each"""
from meslf.networks.create_network import create_radial_line_network, combine_networks_as_tree, add_gas_data
from meslf.networks.read_write_network import to_pd_dataframes
from meslf.networks.carrier import Gas
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
import numpy as np
import os 
import sys

# constants
cm = 1e-2 #[m]
km = 1e3 #[m]
mbar = 1e2 #[Pa]
    
def create_streets(n,m,number_of_streets,carrier):
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
    p_ref = 0.1*mbar #[Pa] pressure before valve
    # link data
    D_unit = 'm'
    L_unit = 'm'
    # half link data
    np.random.seed(0)
    q_min = 5e-5 #[kg/s]
    q_var = 2e-5 #[kg/s]
    q_unit = 'kg/s'
    y_shift = 0
    for i in range(number_of_streets):
        # topology
        street = create_radial_line_network(n,carrier='g',m=m,net_name='S'+str(i))
        
        # link data
        D = 5*cm
        # links between junctions (and source) + links between junctions and loads
        link_params = [{'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':100,'L_unit':L_unit}]*(n-m) + [{'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':10,'L_unit':L_unit}]*n
        link_types = ['pipe_low_pres_pole']*(2*n-m)
        #link_params = [{'carrier':carrier, 'alpha':0.0025}]*(n-m) + [{'carrier':carrier, 'alpha':0.025}]*n
        #link_types = ['pipe_linear']*(2*n-m)
        # half link data
        q_inj = np.random.rand(n)*q_var + q_min
        
        # add data
        add_gas_data(street,p_ref,q_inj,link_types,link_params)
        
        # adjust y coordinates
        for node in street.get_nodes():
            node.y += y_shift
        y_shift += 3
        
        # rename source node
        street.nodes[0].name = 's'+street.name
        
        streets.append(street)
    
    return streets
    
def create_network(n,m,nS,carrier):
    """Create a the quarter network. 
    
    Parameters
    ----------
    n : int
        number of loads, for each street
    m : int
        number of junctions with two loads, for each street
    nS : int
        number of street networks
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
    streets = create_streets(n,m,nS,carrier)
    quarter = combine_networks_as_tree(streets,net_name = 'Q0')
    
    # asign reference pressure to new source node
    p_ref = 0
    for street in streets:
        p_ref += street.nodes[0].p
    p_ref /= nS
    quarter.nodes[0].p = p_ref
        
    # link data (for links from quarter source to sources of streets)
    D_unit = 'm'
    L_unit = 'm'
    L = 1*km
    D = 10*cm
    link_type = 'pipe_low_pres_pole'
    link_params = {'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit}
    #link_type = 'pipe_linear'
    #link_params = {'carrier':carrier, 'alpha':0.005}
    for link in quarter.nodes[0].get_out_links():
        if link in quarter.links: # not a half link
            link.set_type(link_type,link_params)

    # rename source node
    quarter.nodes[0].name = 's'+quarter.name
    
    nodes, links, halflinks = to_pd_dataframes(quarter)
    return nodes, links, halflinks

def main(n,m,nS,dir_path):
    """Create the network, and write to pd dataframes. """
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('gas',S,R_air,1,pn,Tn,Tn)    
    
    # create network
    nodes, links, halflinks = create_network(n,m,nS,gas)
    
    # save data
    nodes.to_pickle(os.path.join(dir_path,'GN_1Q_nodes.pkl'))
    links.to_pickle(os.path.join(dir_path,'GN_1Q_links.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'GN_1Q_halflinks.pkl'))
    
if __name__== '__main__':
    if(len(sys.argv) > 1):
        n = int(sys.argv[1]) # numer of loads, for each street
        m = int(sys.argv[2]) # number of junctions with two loads, for each street
        nS = int(sys.argv[3]) # number of street networks
    else:
        n = 10#10#3#20 # numer of loads, for each street
        m = 5#5#1#10 # number of junctions with two loads, for each street
        nS = 4#3#2#10 # number of street networks
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(n,m,nS,dir_path)
    
    
    
        
    
        
        
