"""Network consisting of 'one street', based on the GN1_street data."""
from meslf.networks.gas_network import GasNetwork
from meslf.networks.read_write_network import from_pd_dataframes
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # create gas network from data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','GN_1street')
    nodes = pd.read_pickle(os.path.join(path_to_data, 'GN_1street_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'GN_1street_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'GN_1street_halflinks.pkl'))
    #gas_net = GasNetwork.from_pd_dataframes('test',nodes,links,halflinks)
    gas_net = from_pd_dataframes(nodes,links,halflinks)
    print(nodes)
    print(halflinks)
    for n in gas_net.get_nodes():
        if n.half_links:
            print(n.half_links[0].q)
    for e in gas_net.get_links():
        print(e.link_type)
    # plot network topology
    plt.figure('Network topology')
    ax = plt.gca()
    gas_net.draw_network(ax)
    plt.axis('equal')
    plt.axis('off')
    plt.plot()
    
    
    # solve network
    #p_ref = nodes.loc[nodes['name']=='source'].iloc[0]['p']
    #p_unit = nodes.loc[nodes['name']=='source'].iloc[0]['p_unit']
    #if p_unit == 'mbar':
        #p_ref *= 1e2 #[Pa]
    p_ref = gas_net.nodes[0].p
    p_init = p_ref*np.linspace(0.95,0.9,len(list(gas_net.get_nodes()))-1) # initial pressure deviades 5% - 10% from reference pressure
    form = 'nodal'    
    gas_net.update(p_init,formulation=form) # update without scaling, since x_init is unscaled
    for n in gas_net.get_nodes():
        print('type = {}, p = {}'.format(n.node_type,n.p))
    scale_var = 'per_unit'
    qbase = 0.02
    pbase = p_ref
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    gas_net.initialize()
    x0 = gas_net.set_x_init(formulation=form,scale_var=scale_var,scale_var_params=scale_var_params)
    
    tol = 1e-6
    max_iter = 100
    print("\nSolving system using analytical Jacobian")
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,formulation=form,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    print("\nFull solution using analytical method:")
    print("Solution for pressure: p = {}".format(p_sol))
    print("Solution for injected flow: q_inj = {}".format((q_inj)))
    print("Solution for edge flows: q = {}".format((q_sol)))
    plt.show()
