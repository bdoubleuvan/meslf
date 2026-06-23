"""Make a gas network consisting of two subnetworks, which are 'streets'. Save data
"""
from meslf.networks.create_network import create_radial_line_network, combine_networks_as_tree, add_gas_data
from meslf.networks.read_write_network import to_pd_dataframes
from meslf.networks.carrier import Gas
from meslf.networks.gas_network import GasNetwork
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    # topologies   
    carrier = 'g'
    n1 = 3
    n2 = 5
    m2 = 2
    net1 = create_radial_line_network(n1,carrier=carrier,net_name='S0')
    net2 = create_radial_line_network(n2,carrier=carrier,m=m2,link_to_loads=False,net_name='S1')
    for node in net1.get_nodes():
        node.y -= 2
    
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('gas',S,R_air,1,pn,Tn,Tn)
    
    # link data
    number_of_links1 = len(net1.links)#len(links1.index)
    number_of_links2 = len(net2.links)#len(links2.index)
    # physical data
    L = 500. #[m]
    L_unit = 'm'
    D = .1 #[m]
    D_unit = 'm'
    link_type1 = ['pipe_low_pres_pole']*number_of_links1
    link_params1 = [{'carrier':gas, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit}]*number_of_links1
    link_type2 = ['pipe_low_pres_pole']*number_of_links2
    link_params2 = [{'carrier':gas, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit}]*number_of_links2
    
    # halflink data
    np.random.seed(0)
    q_min = 0.02 #[kg/s]
    q_var = 0.021 #[kg/s]
    q_unit = 'kg/s'
    q_inj1 = np.random.rand(n1)*q_var + q_min
    q_inj2 = np.random.rand(n2)*q_var + q_min
    
    add_gas_data(net1,0.,q_inj1,link_type1,link_params1)
    add_gas_data(net2,0.,q_inj2,link_type2,link_params2)
    
    # combine networks
    net = combine_networks_as_tree([net1,net2])
    
    # node data
    p_ref = 50 #[mbar]
    p_ref *= 1e2 #[Pa]
    net.nodes[0].p = p_ref
    net.nodes[0].y = (net1.nodes[0].y+net2.nodes[0].y)/2
    
    # data of added links and half links
    link_type_coupling = 'pipe_low_pres_pole'
    link_params_coupling = {'carrier':gas, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit}
    coupling_link1 = net.links[number_of_links1]
    coupling_link1.set_type(link_type_coupling,link_params_coupling)
    coupling_link2 = net.links[-1]
    coupling_link2.set_type(link_type_coupling,link_params_coupling)
    
    nodes, links, halflinks = to_pd_dataframes(net)
    print(nodes)
    print(links)
    print(halflinks)
    
    # save data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    nodes.to_pickle(os.path.join(dir_path,'GN_2streets_nodes.pkl'))
    links.to_pickle(os.path.join(dir_path,'GN_2streets_links.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'GN_2streets_halflinks.pkl'))
    
