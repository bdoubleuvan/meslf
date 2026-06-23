"""Network consisting of one city, based on the GN_1C data."""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink
from meslf.networks.read_write_network import from_pd_dataframes
import numpy as np
import pandas as pd
import scipy.sparse as sps
import os.path
import matplotlib.pyplot as plt
from meslf.networks.read_write_network import to_pd_dataframes
import argparse # To get arguments (as list) from command line
from meslf.utils.constants import mbar, mbar, bar

from meslf.load_flow.system_of_equations import NonLinearSystemGas

def create_network(dir_path):
    """Create the gas network from the data
    
    Parameters
    ----------
    dir_path : string
        Path to the directory where this file is located
        
    Returns 
    -------
    gas_net : GasNetwork
        The network created from the data
    """
    # create gas network from data
    path_to_data = os.path.join(dir_path,'network_data','N_1C')
    nodes = pd.read_pickle(os.path.join(path_to_data, 'GN_1C_nodes.pkl'))    
    links = pd.read_pickle(os.path.join(path_to_data, 'GN_1C_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'GN_1C_halflinks.pkl'))
    gas_net = from_pd_dataframes(nodes,links,halflinks)
    
    # determine number of streets etc.
    nC = len(nodes.index.levels[0])
    nD = len(nodes.index.levels[1])-1
    nQ = len(nodes.index.levels[2])-1
    nS = len(nodes.index.levels[3])-1
    number_of_street_nodes = len(nodes.index.levels[4])
    n = 0
    for halflink_data in halflinks.loc['C0','D0','Q0','S0'].get('data'):
        if halflink_data.get('q') > 0:
            n+=1
    m = 2*n - (number_of_street_nodes - 2) # -2 because of source node and extra node due to compressor / valve
    return gas_net, nC, nD, nQ, nS, n, m

def initialize_network(gas_net,formulation='full'):
    """Initialize the network.
    
    Parameters
    ----------
    gas_net : GasNetwork
        The network to be initialized
    p_perc_low : float
        Lowest value of fraction of reference pressure that is used in the initial linear pressure profile.
    p_perc_high : float
        Highest value of fraction of reference pressure that is used in the initial linear pressure profile.
    use_fa : bool, optional
        If True, fb is (also) used in the gas network to solve the network. Default is True
        
    Returns
    -------
    x0_nodal : np array
        Vector with initial guess using nodal formulation
    x0_full : np array
        Vector with initial guess using full formulation
    """
    gas_net.initialize()
    gas_source = gas_net.nodes[0]
    gas = gas_net.links[0].link_params.get('carrier')
    p_ref = gas_source.p #[bar]
    q_init_value = np.mean([hl.q for node in gas_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.q>0])
    for link in gas_net.get_links():
        #link.q = q_init_value
        if link.link_type == 'compressor':
            link.q = q_init_value
        else:
            link.q = link.flow()
    x0 = gas_net.set_x_init(formulation=formulation)        
    
    return x0

def scaling_matrices(gas_net, nC, nD, nQ, nS, n, formulation='full'):
    """Determine the matrices needed for scaling
    
    Parameters
    ----------
    gas_net : GasNetwork
        The gas network
    pbase : float
        Base value used for pressure.
    qbase : float
        Base value used for flow.
    use_fb : bool
        If True, fb is (also) used in the gas network to solve the network.
    use_fa : bool, optional
        If True, fb is (also) used in the gas network to solve the network. Default is True
        
    Returns
    -------
    D_F : scipy.sparse diagonal matrix
        Diagonal matrix with function scaling factors (when using the full formulation and fb instead of fa) 
    D_x : scipy.sparse diagonal matrix
        Diagonal matrix with variable scaling factors (when using the full formulation)
    """
    qb_street = 5e-3 #[kg/s]
    qb_quarter = nS*n*qb_street
    qb_district = nQ*nS*n*qb_street
    qb_city = nD*nQ*nS*n*qb_street
    pb_city = 8*bar
    pb_district = 100*mbar
    pb_quarter = 100*mbar
    pb_street = 30*mbar
    # full formulation
    F_entries = gas_net.get_F_entries(formulation)
    x_entries = gas_net.get_x_entries(formulation)
    Fb = np.zeros(len(F_entries))
    for ind_el,el in enumerate(F_entries):
        if 'S' in el.name:
            pbase = pb_street
            qbase = qb_street
        elif 'Q' in el.name:
            pbase = pb_quarter
            qbase = qb_quarter#n*qb_street
        elif 'D' in el.name:
            pbase = pb_district
            qbase = qb_district#nS*n*qb_street
        elif 'C' in el.name:
            pbase = pb_city
            qbase = qb_city#nQ*nS*n*qb_street
        else:
            pbase = pb_city
            qbase = qb_city#nQ*nS*n*qb_street
        if isinstance(el,GasNode):
            Fb[ind_el] = qbase #np.max([link.q for link in el.get_links() if isinstance(link,GasLink)]) #qbase
        elif isinstance(el,GasLink):
            if el.link_eq_form == 'q_of_dp':
                Fb[ind_el] = qbase
            if el.link_type == 'compressor':
                Fb[ind_el] = pbase #el.end_node.p #pbase
            elif 'high_pres' in el.link_type:
                Fb[ind_el] = pbase**2 #el.end_node.p**2 #pbase**2
            else:
                Fb[ind_el] = pbase #el.end_node.p #pbase
    D_F = sps.diags(1/Fb)
    xb = np.zeros(len(x_entries))
    for ind_el,el in enumerate(x_entries):
        if 'S' in el.name:
            if 'ene' in el.name:
                pbase = pb_quarter # node before compressor
            else:
                pbase = pb_street
            qbase = qb_street
        elif 'Q' in el.name:
            pbase = pb_quarter
            qbase = qb_quarter#n*qb_street
        elif 'D' in el.name:
            if 'ene' in el.name:
                pbase = pb_city # node before compressor
            else:
                pbase = pb_district
            qbase = qb_district#nS*n*qb_street
        elif 'C' in el.name:
            pbase = pb_city
            qbase = qb_city#nQ*nS*n*qb_street
        else:
            pbase = pb_city
            qbase = qb_city#nQ*nS*n*qb_street
        if isinstance(el,GasNode):
            xb[ind_el] = pbase #el.p#pbase # NB: I think el.p is wrong!! Now two nodes connected to one pipe can / will have a different base value for the pressure, meaning my analysis (wrt p.u.?) is wrong...
        elif isinstance(el,GasLink):
            xb[ind_el] = qbase #el.q#qbase
    D_x = sps.diags(1/xb)
    return D_F, D_x

def solve_GN_1C(dir_path,max_number_of_nodes, formulation='full', show_plots=True):
    """
    Parameters
    ----------
    dir_path : string
        Path to the directory where this file is located
    p_perc_low : float
        Lowest value of fraction of reference pressure that is used in the initial linear pressure profile.
    p_perc_high : float
        Highest value of fraction of reference pressure that is used in the initial linear pressure profile.
    max_number_of_nodes : int
        Maximum number of total nodes for which FD is used, and for which the topology is plotted, etc.
    use_fb : bool
        If True, fb is also used to solve the network (and not only fa)
    save_fig : bool
        If True, the convergence plot is saved as a pgf. 
    """
    # create gas network from data
    gas_net, nC, nD, nQ, nS, n, m = create_network(dir_path)
    number_of_nodes = len(gas_net.nodes)
    
    # initalize network
    x0 = initialize_network(gas_net,formulation=formulation)
    
    if number_of_nodes<max_number_of_nodes:   
        form = formulation
        if show_plots:
            # plot the gas_network
            plt.figure('Network topology gas')
            ax = plt.gca()
            gas_net.draw_network(ax,halflink_length=0.5)
            plt.axis('equal')
            plt.axis('off')
            plt.plot()
    
        for link in gas_net.get_links():
            print('Link {} with eq form {}, link type {}, link params {}, and q = {:2e} [kg/s]'.format(link.name,link.link_eq_form,link.link_type,link.link_params,link.q))
            ddp_dp_start,ddp_dp_end = link.pres_drop_func_der_p()
            print('Derivatives: ddp_dp = {:2e}, {:2e} and df_ddp = {:2e}'.format(ddp_dp_start,ddp_dp_end, link.f_der_dp_func()))
        for node in gas_net.get_nodes():
            print('Node {} with node type {} and p = {:2e} [Pa]'.format(node.name,node.node_type,node.p))
            for hl in node.get_half_links():
                print('Half link {} with q = {:2e} [kg/s]'.format(hl.name,hl.q))
        
        nlsys = NonLinearSystemGas(gas_net,formulation=form)
        F0 = nlsys.F(x0)
        J0 = nlsys.J(x0)
        print('length of x0 = {}, and length of F(x0) = {}'.format(len(x0),len(F0)))
        print('determinant: |J(x0)|={}'.format(np.linalg.det(J0.todense())))
        print('J(x0) = {}'.format(J0))
        
        if show_plots:
            # spy plot of Jacobian
            fig_J = nlsys.spy_plot_J(x0)
            ax_J = plt.gca()
            nlsys.plot_J_overlay(ax_J)
            
            # colormap of Jacobian
            fig_J_map = plt.figure('Jacobian gas')
            plt.imshow(J0.todense())
            ax_J_map = plt.gca()
            nlsys.plot_J_overlay(ax_J_map)
            plt.colorbar()
        
    # scaling in solver
    q_init_value = np.mean([hl.q for node in gas_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.q>0])
    qbase = q_init_value
    gas_source = gas_net.nodes[0]
    pbase = gas_source.p
    D_F, D_x = scaling_matrices(gas_net, nC, nD, nQ, nS, n, formulation=formulation)
    
    if number_of_nodes < max_number_of_nodes:
        D_x_inv = sps.diags(1/D_x.data[0])
        J_scaled = D_F.dot(J0.dot(D_x_inv))
        print('D_F J D_x = {}'.format(J_scaled))
        
        if show_plots:
            # spy plot of scaled J over original
            ax_J.spy(J_scaled,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5)
            
            # colormap of Jacobian
            fig_J_scaled_map = plt.figure('Scaled Jacobian gas')
            plt.imshow(J_scaled.todense())
            ax_J_scaled_map = plt.gca()
            nlsys.plot_J_overlay(ax_J_scaled_map)
            plt.colorbar()
    
    # solve the system
    tol = 1e-6
    max_iter = 100
    det_tol = 1e-12
    x_sol,iters,err_vec,_,_,_ = gas_net.solve_network(tol,max_iter,formulation=formulation,solver='NR',D_F=D_F,D_x=D_x,det_tol=det_tol)    
    
    if show_plots:
        # plot convergence
        fig_conv_gas_one_solver = plt.figure('Convergence plot gas (max iters = {}, len(x) = {}, nC = {}, nQ = {}, nD = {}, nS = {}, n = {}, m = {})'.format(max_iter,len(x0),nC,nQ,nD,nS,n,m))
        ax_conv_gas_one_solver = fig_conv_gas_one_solver.gca()
        plt.cla()
        plt.xlabel('Iteration k')
        plt.ylabel('Error $||D_F F(x^k)||_2$')
        ax_conv_gas_one_solver.semilogy([0,iters+1],[tol,tol],'r:')
        ax_conv_gas_one_solver.semilogy(err_vec,marker='o',ls='-',color='tab:blue')
        plt.grid(which='major',color='k', linestyle='--', alpha=.2)
        plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
        # plot solution
        if number_of_nodes<max_number_of_nodes:
            plt.figure('Network solution gas')
            ax_sol = plt.gca()
            gas_net.draw_network_value(ax_sol,halflink_length=0.5,halflink_angle=1)
            plt.axis('equal')
            plt.axis('off')
            plt.plot()
        
    return x_sol, iters, err_vec
        
if __name__ == '__main__':
    max_number_of_nodes = 500 # Maximum number of total nodes for which the topology is plotted, etc. 
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    solve_GN_1C(dir_path,max_number_of_nodes)    
    
    plt.show()
