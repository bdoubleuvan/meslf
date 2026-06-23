"""Network consisting of one city, based on the HN_1C data."""
from meslf.networks.read_write_network import from_pd_dataframes
from meslf.load_flow.system_of_equations import NonLinearSystemHeat
from meslf.utils.constants import bar, kW
import numpy as np
import scipy.sparse as sps
import pandas as pd
import os.path
import matplotlib.pyplot as plt

def create_network(dir_path):
    """Create the heat network from the data
    
    Parameters
    ----------
    dir_path : string
        Path to the directory where this file is located
        
    Returns 
    -------
    heat_net : HeatNetwork
        The network created from the data
    """
    # create heat network from data
    path_to_data = os.path.join(dir_path,'network_data','N_1C')
    nodes = pd.read_pickle(os.path.join(path_to_data, 'HN_1C_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'HN_1C_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'HN_1C_halflinks.pkl'))
    heat_net = from_pd_dataframes(nodes,links,halflinks)
    
    # determine number of streets etc.
    nC = len(nodes.index.levels[0])
    nD = len(nodes.index.levels[1])-1
    nQ = len(nodes.index.levels[2])-1
    nS = len(nodes.index.levels[3])-1
    number_of_street_nodes = len(nodes.index.levels[4])
    n = 0
    for halflink_data in halflinks.loc['C0','D0','Q0','S0'].get('data'):
        if halflink_data.get('phi') > 0:
            n+=1
    m = 2*n - (number_of_street_nodes - 2) # -2 because of source node and extra node due to pumps
    return heat_net, nC, nD, nQ, nS, n, m

def initialize_network(heat_net, nC, nD, nQ, nS, n, form = 'half_link_flow'):
    """Initialize the network.
    
    Parameters
    ----------
    heat_net : HeatNetwork
        The network to be initialized
        
    Returns
    -------
    x0 : np array
        Vector with initial guess
    """
    Ta = heat_net.links[0].link_params.get('Ta')
    heat_net.Ta = Ta
    heat_net.initialize()
    hl_flow = list()
    hl_Tr = list()
    for node in heat_net.get_nodes(node_types=[1,3,4,12,13]):
        for hl in node.get_half_links():
            m_hl = hl.flow()
            hl.m = m_hl
            hl_flow.append(m_hl)
            if hl.sink:
                hl_Tr.append(hl.Tr)
    m_init_value = np.mean(hl_flow) 
    Tr_init = np.min(hl_Tr)
    for node in heat_net.get_nodes(node_types=[0,1,2,3,4,5,6,7,12,13]):
        node.Tr = Tr_init
    m_init_street = n/2*m_init_value
    m_init_quarter = nS*n*m_init_value
    m_init_district = nQ*nS*n*m_init_value
    m_init_city = nD*nQ*nS*n*m_init_value
    for link in heat_net.get_links():
        #if link.link_type == 'isolated_pump':
            #link.m = m_init_value
        #else:
            #link.m = m_init_value # link.flow() #(setting it equal to link.flow() will mean the link equations will be satisfied. Setting it equal to m_init_value means that conservation of mass is satisfied in each street.
        if 'S' in link.name:
            link.m = m_init_street
        elif 'Q' in link.name:
            link.m = m_init_quarter
        elif 'D' in link.name:
            link.m = m_init_district
        elif 'C' in link.name:
            link.m = m_init_city 
        else:
            link.m = m_init_city 
    x0 = heat_net.set_x_init(formulation=form)        
    return x0

def scaling_matrices(heat_net, nC, nD, nQ, nS, n, form = 'half_link_flow'):
    """Determine the matrices needed for scaling
    
    Parameters
    ----------
    heat_net : HeatNetwork
        The heat network
        
    Returns
    -------
    D_F : scipy.sparse diagonal matrix
        Diagonal matrix with function scaling factors
    D_x : scipy.sparse diagonal matrix
        Diagonal matrix with variable scaling factors
    """
    pbase = 9*bar #[Pa]
    Tbase = 100 #[C]
    mbase_load = 1e-3 #[kg/s]
    mb_street = n/2*mbase_load
    mb_quarter = nS*n*mbase_load
    mb_district = nQ*nS*n*mbase_load
    mb_city = nD*nQ*nS*n*mbase_load
    phibase_load = .5*kW #[W]
    phib_street = n/2*phibase_load
    phib_quarter = nS*n*phibase_load
    phib_district = nQ*nS*n*phibase_load
    phib_city = nD*nQ*nS*n*phibase_load
    x_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks = heat_net.get_x_entries(formulation=form)
    xb_m = np.zeros(len(unknown_m_links))
    for ind_el, el in enumerate(unknown_m_links):
        if 'S' in el.name:
            mbase = mb_street
        elif 'Q' in el.name:
            mbase = mb_quarter
        elif 'D' in el.name:
            mbase = mb_district
        elif 'C' in el.name:
            mbase = mb_city
        else:
            mbase = mb_city
        xb_m[ind_el] = mbase
    xb_m_hl = np.zeros(len(unknown_m_halflinks))
    for ind_el, el in enumerate(unknown_m_halflinks):
        if 'S' in el.start_node.name:
            mbase_hl = mbase_load#mb_street
        elif 'Q' in el.start_node.name:
            mbase_hl = mb_quarter
        elif 'D' in el.start_node.name:
            mbase_hl = mb_district
        elif 'C' in el.start_node.name:
            mbase_hl = mb_city
        xb_m_hl[ind_el] = mbase_hl
    xb_p = pbase*np.ones(len(unknown_p_nodes))
    xb_Ts = Tbase*np.ones(len(unknown_Ts_nodes))
    xb_Tr = Tbase*np.ones(len(unknown_Tr_nodes))
    xb_Tshl = Tbase*np.ones(len(unknown_Ts_halflinks))
    xb_Trhl = Tbase*np.ones(len(unknown_Tr_halflinks))
    xb = np.concatenate((xb_m,xb_m_hl,xb_p,xb_Ts,xb_Tr,xb_Tshl,xb_Trhl))
    D_x = sps.diags(1/xb)
    F_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_dT_halflinks = heat_net.get_F_entries(formulation=form)
    Fb_m = mbase*np.zeros(len(F_m_nodes))
    for ind_el, el in enumerate(F_m_nodes):
        if 'S' in el.name:
            mbase = mb_street
        elif 'Q' in el.name:
            mbase = mb_quarter
        elif 'D' in el.name:
            mbase = mb_district
        elif 'C' in el.name:
            mbase = mb_city
        Fb_m[ind_el] = mbase
    Fb_deltap = pbase*np.ones(len(F_deltap_links))
    Fb_Ts = np.zeros(len(F_Ts_nodes))
    for ind_el, el in enumerate(F_Ts_nodes):
        if 'S' in el.name:
            mbase = mb_street
        elif 'Q' in el.name:
            mbase = mb_quarter
        elif 'D' in el.name:
            mbase = mb_district
        elif 'C' in el.name:
            mbase = mb_city
        Fb_Ts[ind_el] = mbase*Tbase
    Fb_Tr = np.zeros(len(F_Tr_nodes))
    for ind_el, el in enumerate(F_Tr_nodes):
        if 'S' in el.name:
            mbase = mb_street
        elif 'Q' in el.name:
            mbase = mb_quarter
        elif 'D' in el.name:
            mbase = mb_district
        elif 'C' in el.name:
            mbase = mb_city
        Fb_Tr[ind_el] = mbase*Tbase
    Fb_phi = np.zeros(len(F_phi_halflinks))
    for ind_el, el in enumerate(F_phi_halflinks):
        if 'S' in el.start_node.name:
            phibase = phib_street
        elif 'Q' in el.start_node.name:
            phibase = phib_quarter
        elif 'D' in el.start_node.name:
            phibase = phib_district
        elif 'C' in el.start_node.name:
            phibase = phib_city
        Fb_phi[ind_el] = phibase
    Fb_dT = Tbase*np.ones(len(F_dT_halflinks))
    Fb = np.concatenate((Fb_m,Fb_deltap,Fb_Ts,Fb_Tr,Fb_phi,Fb_dT))
    D_F = sps.diags(1/Fb)
    return D_F, D_x

def solve_HN_1C(dir_path,max_number_of_nodes, form = 'half_link_flow', show_plots=True):
    """
    Parameters
    ----------
    dir_path : string
        Path to the directory where this file is located
    max_number_of_nodes : int
        Maximum number of total nodes for which FD is used, and for which the topology is plotted, etc.
    """
    # create heat network from data
    heat_net, nC, nD, nQ, nS, n, m = create_network(dir_path)
    number_of_nodes = len(heat_net.nodes)
    
    # initalize network
    x0 = initialize_network(heat_net, nC, nD, nQ, nS, n, form=form)
    
    if number_of_nodes<max_number_of_nodes: 
        if show_plots:
            # plot the network
            plt.figure('Network topology heat')
            ax = plt.gca()
            heat_net.draw_network(ax,halflink_length=0.5)
            plt.axis('equal')
            plt.axis('off')
            plt.plot()
        
        print('\nNode info:')
        for node in heat_net.get_nodes():
            print('Node {} with node type {} and p = {:2e} [Pa], Ts = {:2f} [C], Tr = {:2f} [C] '.format(node.name,node.node_type,node.p,node.Ts,node.Tr))
            for hl in node.get_half_links():
                print('Half link {}, link type {}, m = {:2e} [kg/s], dphi = {:2e} [W], Ts = {:2f} [C], Ts = {:2f} [C]'.format(hl.name,hl.link_type,hl.m,hl.dphi,hl.Ts,hl.Tr))
        print('\nLink info:')
        for link in heat_net.get_links():
            #print('Link {}, link type {}, link params {}, and m = {:2e} [kg/s]'.format(link.name,link.link_type,link.link_params,link.m))
            print('Link {}, link type {}, and m = {:2e} [kg/s]'.format(link.name,link.link_type,link.m))
        
        nlsys = NonLinearSystemHeat(heat_net,formulation=form)
        F0 = nlsys.F(x0)
        print('x0 = {}'.format(x0))
        print('F(x0) = {}'.format(F0))
        J0 = nlsys.J(x0)
        print('length of x0 = {}, and length of F(x0) = {}'.format(len(x0),len(F0)))
        print('determinant: |J(x0)|={}'.format(np.linalg.det(J0.todense())))
        
        if show_plots:
            # spy plot of Jacobian
            fig_J = nlsys.spy_plot_J(x0)
            ax_J = plt.gca()
            nlsys.plot_J_overlay(ax_J)
            
            # colormap of Jacobian
            fig_J_map = plt.figure('Jacobian heat')
            plt.imshow(J0.todense())
            ax_J_map = plt.gca()
            nlsys.plot_J_overlay(ax_J_map)
            plt.colorbar()
        
    # scaling in solver
    D_F, D_x = scaling_matrices(heat_net, nC, nD, nQ, nS, n,form=form)
    
    if number_of_nodes < max_number_of_nodes:
        print('D_F = {}'.format(D_F))
        #print('D_x = {}'.format(D_x))
        D_x_inv = sps.diags(1/D_x.data[0])
        print('D_x_inv = {}'.format(D_x_inv))
        print('D_F_inv = {}'.format(sps.diags(1/D_F.data[0])))
        J_scaled = D_F.dot(J0.dot(D_x_inv))
        #print('D_F J D_x = {}'.format(J_scaled))
        
        if show_plots:
            # spy plot of scaled J over original
            ax_J.spy(J_scaled,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5)
            
            # colormap of Jacobian
            fig_J_scaled_map = plt.figure('Scaled Jacobian heat')
            plt.imshow(J_scaled.todense())
            ax_J_scaled_map = plt.gca()
            nlsys.plot_J_overlay(ax_J_scaled_map)
            plt.colorbar()
    
    # solve the system
    tol = 1e-6
    max_iter = 50
    x_sol,iters,err_vec,_,_,_,_,_,_,_,_ = heat_net.solve_network(tol,max_iter,D_F=D_F,D_x=D_x,formulation=form)
    
    if show_plots:
        # plot convergence
        fig_conv_heat_one_solver = plt.figure('Convergence plot heat (max iters = {}, len(x) = {}, nC = {}, nQ = {}, nD = {}, nS = {}, n = {}, m = {})'.format(max_iter,len(x0),nC,nQ,nD,nS,n,m))
        ax_conv_heat_one_solver = fig_conv_heat_one_solver.gca()
        plt.cla()
        plt.xlabel('Iteration k')
        plt.ylabel('Error ($||D_F F(x^k)||_2$)')
        ax_conv_heat_one_solver.semilogy([0,iters+1],[tol,tol],'r:')
        ax_conv_heat_one_solver.semilogy(err_vec,marker='o',ls='-',color='tab:blue')
        plt.grid(which='major',color='k', linestyle='--', alpha=.2)
        plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
        
        # plot solution
        if number_of_nodes<max_number_of_nodes:
            heat_net.update_full(x_sol,formulation=form)
            plt.figure('Network solution heat')
            ax_sol = plt.gca()
            heat_net.draw_network_value(ax_sol,halflink_length=0.5,halflink_angle=1)
            plt.axis('equal')
            plt.axis('off')
            plt.plot()
        
    return x_sol, iters, err_vec
        
if __name__ == '__main__':
    max_number_of_nodes = 500 # Maximum number of total nodes for which the topology is plotted, etc. 
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    solve_HN_1C(dir_path,max_number_of_nodes) 
    
    plt.show()

