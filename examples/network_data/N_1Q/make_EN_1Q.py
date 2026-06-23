"""Create an electrical network consisting of 1 quarter, which consists of 10 streets of approximately 20 loads each"""
from meslf.networks.create_network import create_radial_line_network, combine_networks_as_tree, add_elec_data
from meslf.networks.read_write_network import to_pd_dataframes
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink

import numpy as np
import os
import sys

kW = 1e3 #[W]
kV = 1e3 #[V]
Sbase = 1*kW #[W]
Vbase = 1*kV #[V]
Ybase = Sbase/(Vbase**2)
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
    V_ref = 1. #[p.u]
    delta_ref = 0. #[rad]
    # link data
    b = -10./Ybase # [p.u.]
    g = 1./Ybase  # [p.u.]
    # half link data
    np.random.seed(0)
    S_min = 1. #[p.u.]
    S_var = 0.8 #[p.u.]
    y_shift = 0
    for i in range(number_of_streets):
        # topology
        street = create_radial_line_network(n,carrier='e',m=m,net_name='S'+str(i))

        # link data
        # links between junctions (and source) + links between junctions and loads
        link_params = [{'b':b,'g':g}]*(n-m) + [{'b':b,'g':g}]*n
        link_types = ['short_line']*(2*n-m)

        # half link data
        P_inj = np.random.rand(n)*S_var + S_min
        Q_inj = np.random.rand(n)*S_var + S_min

        # add data
        add_elec_data(street,V_ref,delta_ref,P_inj,Q_inj,link_types,link_params)

        # adjust y coordinates
        for node in street.get_nodes():
            node.y += y_shift
        y_shift += 3

        # rename source node
        street.nodes[0].name = 's'+street.name

        streets.append(street)

    return streets

def create_network(n,m,nS):
    """Create a the quarter network.

    Parameters
    ----------
    n : int
        number of loads, for each street
    m : int
        number of junctions with two loads, for each street
    nS : int
        number of street networks

    Returns
    -------
    nodes : pd DataFrame
        Node data
    links : pd DataFrame
        Link data
    halflinks: pd DataFrame
        Halflink data
    """
    streets = create_streets(n,m,nS)
    quarter = combine_networks_as_tree(streets,net_name = 'Q0')

    # asign reference voltage angle and amplitude to new source node
    V_ref = 0
    delta_ref = 0
    for street in streets:
        V_ref += street.nodes[0].V
        delta_ref += street.nodes[0].delta
    V_ref /= nS
    delta_ref /= nS
    quarter.nodes[0].V = V_ref
    quarter.nodes[0].delta = delta_ref

    # link data (for links from quarter source to sources of streets)
    b = -10./Ybase  # [p.u.]
    g = 1./Ybase  # [p.u.]
    link_type = 'short_line'
    link_params = {'b':b,'g':g}
    for link in quarter.nodes[0].get_out_links():
        if link in quarter.links: # not a half link
            link.set_type(link_type,link_params)

    # rename source node
    quarter.nodes[0].name = 's'+quarter.name

    nodes, links, halflinks = to_pd_dataframes(quarter)
    return nodes, links, halflinks

def main(n,m,nS,dir_path):
    """Create the network, and write to pd dataframes. """
    # create network
    nodes, links, halflinks = create_network(n,m,nS)

    # save data
    nodes.to_pickle(os.path.join(dir_path,'EN_1Q_nodes.pkl'))
    links.to_pickle(os.path.join(dir_path,'EN_1Q_links.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'EN_1Q_halflinks.pkl'))
    
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

    
