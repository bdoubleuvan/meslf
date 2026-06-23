"""Electrical network consisting of 3 demand/source nodes. Also called the reduced benchmark problem."""
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.utils.constants import kV, MW
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import os
import pandas as pd
from meslf.networks.read_write_network import from_pd_dataframes

# Read the scenario data
def read_scen_data(path_to_data,c_hl=True,topology=1):
    """Read the scenario data
    
    Parameters
    ----------
    c_hl : bool, optional
        If true, half links are added to the nodes with values equal to the flow going to the coupling components. Default is True.
    topology : int, optional
        Determines which topology is used in the MES, hence, which is used in the elec ntework when the coupling components are taken into account seperately. Options are 1-4. Default is 1. 
        
    Returns
    -------
    elec_net_scen : ElecNetwork
        The single-carrier scenario network.
    x : list
        List with node x-coordinates. 
    y : list
        List with node y-coordinates.
    mes_net_scen : HeterogeneousNetwork
        The multi-carrier scenario network. Is None if c_hl is False.

    """
    nodes = pd.read_pickle(os.path.join(path_to_data, 'EN_BP_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'EN_BP_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'EN_BP_halflinks.pkl'))
    elec_net_scen = from_pd_dataframes(nodes,links,halflinks)

    # (full) solution of scenario (somehow, the values of the half links are not correctly stored in the pd halflinks (I think this was a choice with being able to run load flow after reading a pd or something), so the scenario is taken from the network, and not from the pd df's)
    elec_net_scen.initialize()
    x_scen = elec_net_scen.set_x_init()
    elec_net_scen.update_full(x_scen)
    delta_scen  = np.zeros(len(elec_net_scen.nodes))
    V_scen  = np.zeros(len(elec_net_scen.nodes))
    P_inj_scen  = [hl.P for node in elec_net_scen.get_nodes() for hl in node.get_half_links()]
    Q_inj_scen  = [hl.Q for node in elec_net_scen.get_nodes() for hl in node.get_half_links()]
    P_edge_scen = np.zeros(2*len(elec_net_scen.links))
    Q_edge_scen = np.zeros(2*len(elec_net_scen.links))
    for ind_e,link in enumerate(elec_net_scen.get_links()):
        P_edge_scen[ind_e] = link.get_Pstart()
        P_edge_scen[ind_e+len(elec_net_scen.links)] =link.get_Pend()
        Q_edge_scen[ind_e] = link.get_Qstart()
        Q_edge_scen[ind_e+len(elec_net_scen.links)] =link.get_Qend()
    for ind_n,node in enumerate(elec_net_scen.get_nodes()):
        delta_scen[ind_n] = node.get_delta()
        V_scen[ind_n] = node.get_V()
        
    if c_hl:
        mes_data = 'top'+str(topology)+'_2c'
        nodes_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_nodes_'+mes_data+'.pkl'))
        links_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_links_'+mes_data+'.pkl'))
        halflinks_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_halflinks_'+mes_data+'.pkl'))
        
        mes_net_scen = from_pd_dataframes(nodes_mes,links_mes,halflinks_mes)
        # (full) solution of scenario, multi-carrier
        delta_mes_scen = [node.get_delta() for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode)]
        V_mes_scen = [node.get_V() for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode)]
        P_inj_mes_scen = [hl.get_P() for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode) for hl in node.get_half_links()]
        Q_inj_mes_scen = [hl.get_Q() for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode) for hl in node.get_half_links()]
        x = [node.x for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode)]
        y = [node.y for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode)]
    else:
        mes_net_scen = None
        delta_mes_scen = []
        V_mes_scen = []
        P_inj_mes_scen = []
        Q_inj_mes_scen = []
        x = [node.x for node in elec_net_scen.get_nodes()]
        y = [node.y for node in elec_net_scen.get_nodes()]
    return elec_net_scen,delta_scen,V_scen,P_inj_scen,Q_inj_scen,P_edge_scen,Q_edge_scen,x,y,mes_net_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen

def create_network(path_to_data,c_hl=True,topology=1):
    """Create an electrical network consisting of 3 demand/source nodes. The values are given in p.u., assuming a base value for S of 1MW.
    
    Parameters
    ----------
    c_hl : bool, optional
        If true, half links are added to the nodes with values equal to the flow going to / coming from the coupling components. Default is True.
        
    Returns
    -------
    elec_net : ElectricalNetwork
        The electrical network
    """
    if not topology in [1,2,3,4]:
        raise ValueError('Enter valid value for topology')
    elec_net_scen,delta_scen,V_scen,P_inj_scen,Q_inj_scen,P_edge_scen,Q_edge_scen,x,y,mes_net_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen = read_scen_data(path_to_data,c_hl=c_hl,topology=topology)

    # physical arameters of the lines
    elec_link_params = elec_net_scen.links[0].link_params.copy()
    elec_link_type = elec_net_scen.links[0].link_type
    
    
    # boundary conditions
    V0 = V_scen[0]
    delta0 = delta_scen[0]
    V1 = V_scen[1]
    if mes_net_scen:
        P1_load = P_inj_mes_scen[1]
        P2_load = P_inj_mes_scen[2]
        Q2_load = Q_inj_mes_scen[2]
    else:
        P1_load = P_inj_scen[1]
        P2_load = P_inj_scen[2]
        Q2_load = Q_inj_scen[2]
        
    e0 = ElectricalNode('en0',node_type=0,x=x[0],y=y[0],V=V0,delta=delta0) # ref
    e1 = ElectricalNode('en1',node_type=1,x=x[1],y=y[1],P=P1_load,V=V1) # generator 
    e2 = ElectricalNode('en2',x=x[2],y=y[2],node_type=2,P=P2_load,Q=Q2_load) # load
    
    if c_hl:
        Pc = P_edge_scen[3]
        Qc = Q_edge_scen[3]
        ElectricalHalfLink('en1_hl0',start_node=e1,P=Pc,Q=Qc) # power from coupling
        
    el0 = ElectricalLink('el0',e0,e1,link_type=elec_link_type,link_params=elec_link_params)
    el1 = ElectricalLink('el1',e0,e2,link_type=elec_link_type,link_params=elec_link_params.copy())
    el2 = ElectricalLink('el2',e1,e2,link_type=elec_link_type,link_params=elec_link_params.copy())
    
    elec_net = ElectricalNetwork('3 nodes')
    elec_net.add_link(el0)
    elec_net.add_link(el1)
    elec_net.add_link(el2)
    return elec_net

def initialize_network(elec_net,V=10/np.sqrt(3)*kV,scale_var=None,scale_var_params=None):
    """Initialize the electrical network, consisting of 3 demand/source nodes.
    
    Parameters
    ----------
    elec_net : ElectricalNetwork
        The electrical network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    x_entries, unknown_delta_nodes, unknown_V_nodes = elec_net.get_x_entries()
    
    delta_init = np.zeros(len(unknown_delta_nodes))
    V_init = V*np.ones(len(unknown_V_nodes))
    x_init = np.concatenate((delta_init,V_init))
    
    elec_net.initialize()
    elec_net.update(x_init)
    x0 = elec_net.set_x_init(scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def scaling_matrices(elec_net,Vbase,Sbase):
    xe_entries, unknown_delta_nodes, unknown_V_nodes = elec_net.get_x_entries()
    
    xb_delta = np.ones(len(unknown_delta_nodes))
    xb_V = Vbase*np.ones(len(unknown_V_nodes))
    
    xb = np.concatenate((xb_delta,xb_V))
    Dx = sps.diags(1/xb)
    
    Fe_entries, known_P_nodes, known_Q_nodes = elec_net.get_F_entries()
    Fb = Sbase*np.ones(len(Fe_entries))
    DF = sps.diags(1/Fb)
    
    return DF,Dx

def run_load_flow(path_to_data,c_hl=True,topology=1,V_init=10/np.sqrt(3)*kV,Vbase=10/np.sqrt(3)*kV,Sbase=1*MW,tol=1e-6,max_iter=50):
    """Stead-state load flow analysis of electrical network. Per unit system is already assumed, and the default values are used for initialization.
    """
    # create network
    elec_net = create_network(path_to_data,c_hl=c_hl,topology=topology)
    # initialize
    x0 = initialize_network(elec_net,V=V_init)
    
    # solve network
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR',scale_var='matrix',scale_var_params={'deltabase':1,'Vbase':Vbase,'Sbase':Sbase})
    
    return elec_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge,tol

def comp_conv(path_to_data):
    """Compare convergence"""
    # make figure to plot convergence
    fig_conv_elec = plt.figure('Convergence plot electrical network')
    ax_conv_elec = fig_conv_elec.gca()
    topology = 1
    max_iters_used = 0
    for c_hl in [True,False]:
        print('seperate couplings used: {}'.format(c_hl))
        elec_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge,tol = run_load_flow(path_to_data,c_hl=c_hl,topology=topology)
        print('Solution:')
        print('delta = {} rad.'.format(delta_sol))
        print('|V| = {} V'.format(V_sol))
        print('P edge = {} W'.format(P_edge))
        print('Q edge = {} W'.format(Q_edge))
        print('S nodal inj = {} W'.format(S_inj))
        print('P hl = {} W'.format([hl.P for node in elec_net.get_nodes() for hl in node.get_half_links()]))
        print('Q hl = {} W'.format([hl.Q for node in elec_net.get_nodes() for hl in node.get_half_links()]))
        # plot convergence
        if c_hl:
            ls = '--'
            label = 'sep. coup.'
        else:
            ls = '-'
            label = 'int. coup.'
        ax_conv_elec.semilogy(err_vec,ls=ls,color='tab:red',marker='.',label=label)
        max_iters_used = max(max_iters_used,iters)
        fig_top = plt.figure('Network topology, seperate coupling {}'.format(c_hl))
        ax_top = fig_top.gca()
        elec_net.draw_network(ax_top,halflink_angle=2,halflink_length=1)
        plt.axis('equal')
        plt.axis('off')
    ax_conv_elec.set_xlabel(r'Iteration $k$')
    ax_conv_elec.set_ylabel(r'Error $||D_F F(x^k)||_2$')
    ax_conv_elec.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
    ax_conv_elec.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_elec.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_elec.legend()
    xmin = 0
    xmax = max_iters_used
    xticks = range(xmin,xmax+1,2) # make sure the xticks are integers
    ax_conv_elec.set_xlim(left=xmin,right=xmax+1)
    ax_conv_elec.set_xticks(xticks)

def comp_conv_scaling(path_to_data,topology=1):
    """Compare convergence of NR for different ways of scaling."""
    # base values
    Sbase = 1*MW #[W]
    Vbase = 10/np.sqrt(3)*kV #[V]
    
    # create networks
    c_hl = False # no separate coupling
    # network with values specified in S.I.
    elec_net_SI = create_network(path_to_data,c_hl=c_hl,topology=topology)
    # network with values specified in p.u.
    elec_net_pu = create_network(path_to_data,c_hl=c_hl,topology=topology)
    for link in elec_net_pu.get_links():
        if not link.link_type == 'short_line':
            raise ValueError('Cannot create network with values specified in p.u. Link type is wrong')
        else:
            Ybase = Sbase/(Vbase**2)
            b_SI = link.b
            g_SI = link.g
            b_pu = b_SI/Ybase
            g_pu = g_SI/Ybase
            link.set_type('short_line',{'b':b_pu,'g':g_pu})
    for node in elec_net_pu.get_nodes():
        node.V /= Vbase
        for hl in node.get_half_links():
            if not hl.link_type == 'flow':
                raise ValueError('Cannot create network with values specified in p.u. HalfLink type is wrong')
            else:
                hl.P /= Sbase
                hl.Q /= Sbase
    
    # initial conditions
    V_init=10/np.sqrt(3)*kV
    
    tol=1e-6
    max_iter=50
    # run load flow for network with values specified in S.I., using matrix scaling
    # initialize
    x0_SI = initialize_network(elec_net_SI,V=V_init)
    # solve network
    x_sol_scaled,iters_scaled,err_vec_scaled,delta_sol_scaled,V_sol_scaled,S_inj_scaled,P_edge_scaled,Q_edge_scaled = elec_net_SI.solve_network(tol,max_iter,solver='NR',scale_var='matrix',scale_var_params={'deltabase':1,'Vbase':Vbase,'Sbase':Sbase})
    
    # run load flow for network with values specified in S.I., using p.u. scaling
    scale_var = 'per_unit'
    scale_var_params = {'Sbase':Sbase,'Vbase':Vbase,'deltabase':1}
    # initialize
    elec_net_SI.reset_network(x0_SI)
    elec_net_SI.update_full(x0_SI)
    x0_SI_pu = initialize_network(elec_net_SI,V=V_init,scale_var=scale_var,scale_var_params=scale_var_params)
    # solve network
    x_sol_SI_pu,iters_SI_pu,err_vec_SI_pu,delta_sol_SI_pu,V_sol_SI_pu,S_inj_SI_pu,P_edge_SI_pu,Q_edge_SI_pu = elec_net_SI.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    
    # run load flow for network with values specified in p.u., without scaling
    # initialize
    x0_pu = initialize_network(elec_net_pu,V=V_init/Vbase)
    x_sol_pu,iters_pu,err_vec_pu,delta_sol_pu,V_sol_pu,S_inj_pu,P_edge_pu,Q_edge_pu = elec_net_pu.solve_network(tol,max_iter,solver='NR')
    
    print('Errors. Par. in S.I., matrix scaling:\n{}'.format(err_vec_scaled))
    print('Errors. Par. in S.I., p.u. scaling:\n{}'.format(err_vec_SI_pu))
    print('Errors. Par. in p.u., unscaled:\n{}'.format(err_vec_pu))
    print('Solution. Par. in S.I., matrix scaling:\n{}'.format(x_sol_scaled))
    print('Solution. Par. in S.I., p.u. scaling:\n{}'.format(x_sol_SI_pu))
    print('Solution. Par. in p.u., unscaled:\n{}'.format(x_sol_pu))
    
    # make figure to plot convergence
    fig_conv_elec = plt.figure('Convergence plot electrical network, scaling')
    ax_conv_elec = fig_conv_elec.gca()
    max_iters_used = max([iters_scaled,iters_SI_pu,iters_pu])
    ls = '-'
    ax_conv_elec.semilogy(err_vec_scaled,ls=ls,color='tab:blue',marker='.',label='matrix scaling')
    ax_conv_elec.semilogy(err_vec_SI_pu,ls=ls,color='tab:orange',marker='.',label='p.u. scaling')
    ax_conv_elec.semilogy(err_vec_pu,ls=ls,color='tab:red',marker='.',label='specified in p.u.')
    ax_conv_elec.set_xlabel(r'Iteration $k$')
    ax_conv_elec.set_ylabel(r'Error ($||D_F F(x^k)||_2$ or $||F(x^k)||_2$)')
    ax_conv_elec.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
    ax_conv_elec.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_elec.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_elec.legend()
    xmin = 0
    xmax = max_iters_used
    xticks = range(xmin,xmax+1,2) # make sure the xticks are integers
    ax_conv_elec.set_xlim(left=xmin,right=xmax+1)
    ax_conv_elec.set_xticks(xticks)
    
if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','N_BP')
    #comp_conv(path_to_data)
    comp_conv_scaling(path_to_data)

    plt.show()
