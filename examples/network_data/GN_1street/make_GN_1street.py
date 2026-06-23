"""Make a gas network based on the network generator. Save data. 
"""
from meslf.networks.create_network import create_radial_line_network, add_gas_data
from meslf.networks.read_write_network import to_pd_dataframes
from meslf.networks.carrier import Gas
from meslf.networks.gas_network import GasNetwork
import numpy as np
import pandas as pd
import os 

if __name__ == '__main__':
    # topology
    #nodes,links,halflinks = network_topology()
    
    carrier = 'g'
    n = 10
    m = 3
    net = create_radial_line_network(n,carrier=carrier,m=m)
    
    # node data
    p_ref = 50 #[mbar]
    p_ref *= 1e2 #[Pa]
    
    # link data
    number_of_links = len(net.links)
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('gas',S,R_air,1,pn,Tn,Tn)
    # physical data
    L = 500. #[m]
    L_unit = 'm'
    D = .1 #[m]
    D_unit = 'm'
    link_type = ['pipe_low_pres_pole']*number_of_links
    link_params = [{'carrier':gas, 'D':D, 'D_unit':D_unit,'L':L,'L_unit':L_unit}]*number_of_links
    
    np.random.seed(0)
    q_min = 0.02/n #[kg/s]
    q_var = 0.02/n #[kg/s]
    q_inj_loads = np.random.rand(n)*q_var + q_min
    
    #nodes, links, halflinks = add_data(nodes,links,halflinks,nodes_data,links_data,halflinks_data)
    add_gas_data(net,p_ref,q_inj_loads,link_type,link_params)  
    nodes, links, halflinks = to_pd_dataframes(net)
    print(nodes)
    print(links)
    print(halflinks)
    
    # save data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    nodes.to_pickle(os.path.join(dir_path,'GN_1street_nodes.pkl'))
    links.to_pickle(os.path.join(dir_path,'GN_1street_links.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'GN_1street_halflinks.pkl'))
    
    
