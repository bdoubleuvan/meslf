"""Create a heat network consisting of 1 quarter, which consists of 10 streets of approximately 20 loads each"""
from meslf.networks.create_network import create_radial_line_network, combine_networks_as_tree, add_heat_data
from meslf.networks.read_write_network import to_pd_dataframes
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water

import numpy as np
import os
import sys

# constants
cm = 1e-2 #[m]
mm = 1e-3 #[m]
km = 1e3 #[m]
kW = 1e3 #[W]
MW = 1e6 #[W]

Ta = 0. #[C] #A value for Ta is needed, otherwise 'None' is used, which will give errors later.

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

    Returns
    -------
    streets : list
        list with HeatNetwork objects
    """
    # carrier
    rho = carrier.rhon
    g = carrier.g

    streets = list()

    # node data
    h_ref = 0.001#100. #[m]
    p_ref = h_ref*rho*g #[Pa]
    Ts_ref = 100.#120. #[C]
    # link data
    L_unit = 'm'
    D_unit = 'm'
    eps = 0.1*mm #[m]
    lam = 0.2 #[W/(mK)]
    # half link data
    To = 50. #[C]
    np.random.seed(0)
    phi_min = 1*kW#50. #[W]
    phi_var = 1*kW#450. #[W]

    y_shift = 0
    for i in range(number_of_streets):
        # topology
        street = create_radial_line_network(n,carrier='h',m=m,net_name='S'+str(i))

        # link data
        D = 5*cm #[m]
        U = lam/(np.pi*D) #[W/(m^2 K)]
        # links between junctions (and source) + links between junctions and loads
        #C = np.sqrt(1/(rho*g))
        #link_params = [{'L':100.,'L_unit':L_unit,'U':U/10,'C':C,'carrier':carrier,'Ta':Ta}]*(n-m) + [{'L':10.,'L_unit':L_unit,'U':U,'C':C,'carrier':carrier,'Ta':Ta}]*n
        #link_types = ['standard_resistor']*(2*n-m)
        link_params = [{'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':100.,'L_unit':L_unit,'U':U/10,'Ta':Ta}]*(n-m) + [{'L':10.,'L_unit':L_unit,'U':U,'D':D, 'D_unit':D_unit,'carrier':carrier,'Ta':Ta}]*n
        link_types = ['standard_pipe_low_pres_pole']*(2*n-m)
        #link_params = [{'L':100.,'L_unit':L_unit,'D':D,'D_unit':D_unit,'eps':eps,'U':U,'carrier':carrier,'Ta':Ta}]*(n-m) + [{'L':10.,'L_unit':L_unit,'D':D,'D_unit':D_unit,'eps':eps,'U':U,'carrier':carrier,'Ta':Ta]*n
        #link_types = ['standard_pipe_low_pres_colebrook']*(2*n-m)

        # half link data
        phi_inj = np.zeros(n+1)
        phi_inj[1:] = np.random.rand(n)*phi_var + phi_min
        Tr_inj = To*np.ones(n+1)
        Ts_inj = Ts_ref*np.ones(n+1) #source node
        dT = Ts_inj - Tr_inj
        halflink_types = ['heat_exchanger']*(n+1)
        halflink_params = [{'carrier':carrier}]*(n+1)
        halflink_bc_types = [0]+[3]*n # source slack + sinks with dphi and To known

        # add data
        add_heat_data(street,p_ref,Ts_ref,phi_inj,Ts_inj,Tr_inj,dT,link_types,link_params,halflink_types,halflink_params,halflink_bc_types)

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
    carrier : WaterCarrier
        The water through the network

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

    # asign reference pressure and supply temperature to new source node
    p_ref = 0
    Ts_ref = 0
    for street in streets:
        p_ref += street.nodes[0].p
        Ts_ref += street.nodes[0].Ts
    p_ref /= nS
    Ts_ref /= nS
    quarter.nodes[0].p = p_ref
    quarter.nodes[0].Ts = Ts_ref

    # link data (for links from quarter source to sources of streets)
    D_unit = 'm'
    L_unit = 'm'
    eps = 0.1*mm #[m]
    lam = 0.002 #[W/(mK)]
    L = 1*km #[m]
    D = 10*cm #[m]
    U = lam/(np.pi*D) #[W/(m^2 K)]
    rho = carrier.rhon
    g = carrier.g
    #C = np.sqrt(1/(rho*g))
    #link_params = {'L':L,'L_unit':L_unit,'U':U,'C':C,'carrier':carrier}
    #link_type = 'standard_resistor'
    link_params = {'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit,'U':U,'Ta':Ta}
    link_type = 'standard_pipe_low_pres_pole'
    #link_type = 'standard_pipe_low_pres_colebrook'
    #link_params = {'carrier':carrier, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit,'eps':eps,'U':U,'Ta':Ta}
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
    rho = 960. #[kg/m^3]
    Cp = 4.182e3 #[J/(kg K)]
    mu = 0.294e-6 #[m^2/s]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp,rho=rho,mu=mu)

    # create network
    nodes, links, halflinks = create_network(n,m,nS,water)

    # save data
    nodes.to_pickle(os.path.join(dir_path,'HN_1Q_nodes.pkl'))
    links.to_pickle(os.path.join(dir_path,'HN_1Q_links.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'HN_1Q_halflinks.pkl'))

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
