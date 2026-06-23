"""Network consisting of one city, based on the EN_1C data."""
from meslf.networks.read_write_network import from_pd_dataframes
from meslf.load_flow.system_of_equations import NonLinearSystemElectrical
from meslf.utils.constants import kV,kW,MW
import numpy as np
import scipy.sparse as sps
import pandas as pd
import os.path
import matplotlib.pyplot as plt

def create_network(dir_path):
    """Create the gas network from the data
    
    Parameters
    ----------
    dir_path : string
        Path to the directory where this file is located
        
    Returns 
    -------
    elec_net : ElectricalNetwork
        The network created from the data
    """
    # create electrical network from data
    path_to_data = os.path.join(dir_path,'network_data','N_1C')
    nodes = pd.read_pickle(os.path.join(path_to_data, 'EN_1C_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'EN_1C_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'EN_1C_halflinks.pkl'))
    elec_net = from_pd_dataframes(nodes,links,halflinks)
    
    # determine number of streets etc.
    nC = len(nodes.index.levels[0])
    nD = len(nodes.index.levels[1])-1
    nQ = len(nodes.index.levels[2])-1
    nS = len(nodes.index.levels[3])-1
    number_of_street_nodes = len(nodes.index.levels[4])
    n = 0
    for halflink_data in halflinks.loc['C0','D0','Q0','S0'].get('data'):
        if halflink_data.get('P') > 0:
            n+=1
    m = 2*n - (number_of_street_nodes - 2) # -2 because of source node and extra node due to trafo
    return elec_net, nC, nD, nQ, nS, n, m

def scaling_matrices(elec_net,nC, nD, nQ, nS, n):
    """Determine the matrices needed for scaling
    
    Parameters
    ----------
    elec_net : ElectricalNetwork
        The electrical network network
        
    Returns
    -------
    D_F : scipy.sparse diagonal matrix
        Diagonal matrix with function scaling factors
    D_x : scipy.sparse diagonal matrix
        Diagonal matrix with variable scaling factors
    """
    #Sb_street = n/2*.3*kW #[W]
    #ratio_SQ = 25
    #ratio_DC = 5
    #Sb_quarter = nS*n*Sb_street*ratio_SQ
    #Sb_district = nQ*nS*n*Sb_street*ratio_SQ
    #Sb_city = nD*nQ*nS*n*Sb_street*ratio_SQ*ratio_DC
    Sb_street = 7*kW #[W]
    Sb_SQ = 77.76*kW
    Sb_quarter = 4.86*MW
    Sb_district = 2.43*MW
    Sb_DC = 48.6*MW
    Sb_city = 24.3*MW
    F_entries, known_P_nodes, known_Q_nodes = elec_net.get_F_entries(formulation='complex_power')
    Fb = np.zeros(len(F_entries))
    for ind_el, el in enumerate(F_entries):
        if 'S' in el.name:
            if 'ene' in el.name: # node before trafo
                Sbase = Sb_SQ #Sb_street # Sb_quarter
            else:
                Sbase = Sb_street
        elif 'Q' in el.name:
            Sbase = Sb_quarter
        elif 'D' in el.name:
            if 'ene' in el.name: # node before trafo
                Sbase = Sb_DC #Sb_district #Sb_city
            else:
                Sbase = Sb_district
        elif 'C' in el.name:
            Sbase = Sb_city
        Fb[ind_el] = Sbase
    D_F = sps.diags(1/Fb)
    x_entries, unknown_delta_nodes, unknown_V_nodes = elec_net.get_x_entries(formulation='complex_power')
    xb_delta = np.ones(len(unknown_delta_nodes))
    xb_V = np.zeros(len(unknown_V_nodes))
    Vb_city = 50*kV
    Vb_district = 10*kV
    Vb_quarter = 10*kV
    Vb_street = .4*kV
    for ind_el, el in enumerate(unknown_V_nodes):
        if 'S' in el.name:
            if 'ene' in el.name:
                Vbase = Vb_street#Vb_quarter # node before trafo
            else:
                Vbase = Vb_street
        elif 'Q' in el.name:
            Vbase = Vb_quarter
        elif 'D' in el.name:
            if 'ene' in el.name:
                Vbase = Vb_district#Vb_city # node before trafo
            else:
                Vbase = Vb_district
        elif 'C' in el.name:
            Vbase = Vb_city
        xb_V[ind_el] = Vbase
    xb = np.concatenate((xb_delta,xb_V))
    D_x = sps.diags(1/xb)
    return D_F, D_x

    
def initialize_network(elec_net):
    """Initialize the network.
    
    Parameters
    ----------
    elec_net : ElectricalNetwork
        The network to be initialized
        
    Returns
    -------
    x0 : np array
        Vector with initial guess using nodal formulation
    """
    elec_net.initialize()
    x0 = elec_net.set_x_init(formulation='complex_power')        
    
    return x0

def solve_EN_1C(dir_path,max_number_of_nodes,show_plots=True):
    """
    Parameters
    ----------
    dir_path : string
        Path to the directory where this file is located
    max_number_of_nodes : int
        Maximum number of total nodes for which FD is used, and for which the topology is plotted, etc.
    """
    # create electrical network from data
    elec_net, nC, nD, nQ, nS, n, m = create_network(dir_path)
    number_of_nodes = len(elec_net.nodes)
    
    # initalize network
    x0 = initialize_network(elec_net)
    
    if number_of_nodes<max_number_of_nodes:   
        for node in elec_net.get_nodes():
            print('node {} with number: {} and type: {},  and V = {}V, and F = {}'.format(node.name,node.number,node.node_type,node.V,node.node_law()))
        for link in elec_net.get_links():
            print('link {} with type: {} and parameters: {}, and P_start = {}, P_end = {}, Q_start = {}, Q_end = {}'.format(link.name,link.link_type,link.link_params,link.active_power_start(),link.active_power_end(),link.reactive_power_start(),link.reactive_power_end()))
            
        if show_plots:
            # plot the electrical network
            plt.figure('Network topology electricity'.format(nC,nQ,nD,nS,n,m))
            ax = plt.gca()
            elec_net.draw_network(ax,halflink_length=0.5)
            plt.axis('equal')
            plt.axis('off')
            plt.plot()
        
        nlsys = NonLinearSystemElectrical(elec_net,formulation='complex_power')
        F0 = nlsys.F(x0)
        print('x0 = {}'.format(x0))
        print('F(x0) = {}'.format(F0))
        for link in elec_net.get_links():
            print('link {} with type, and P_start = {}, P_end = {}, Q_start = {}, Q_end = {}'.format(link.name,link.link_type,link.active_power_start(),link.active_power_end(),link.reactive_power_start(),link.reactive_power_end()))
        for node in elec_net.get_nodes():
            print('node {} with number: {}, F = {}'.format(node.name,node.number,node.node_law()))
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
            fig_J_map = plt.figure('Jacobian electricity')
            plt.imshow(J0.todense())
            ax_J_map = plt.gca()
            nlsys.plot_J_overlay(ax_J_map)
            plt.colorbar()
    
    # scaling in solver
    D_F, D_x = scaling_matrices(elec_net, nC, nD, nQ, nS, n)
    
    if number_of_nodes < max_number_of_nodes:
        #print('D_F = {}'.format(D_F))
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
            fig_J_scaled_map = plt.figure('Scaled Jacobian electricity')
            plt.imshow(J_scaled.todense())
            ax_J_scaled_map = plt.gca()
            nlsys.plot_J_overlay(ax_J_scaled_map)
            plt.colorbar()
        
    # solve the system
    tol = 1e-6
    max_iter = 100
    x_sol,iters,err_vec,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR')
    
    if show_plots:
        # plot convergence
        fig_conv_elec_one_solver = plt.figure('Convergence plot electricity (max iters = {}, len(x) = {}, nC = {}, nQ = {}, nD = {}, nS = {}, n = {}, m = {})'.format(max_iter,len(x0),nC,nQ,nD,nS,n,m))
        ax_conv_elec_one_solver = fig_conv_elec_one_solver.gca()
        plt.cla()
        plt.xlabel('Iteration k')
        plt.ylabel('Error ($||F(x^k)||_2$)')
        ax_conv_elec_one_solver.semilogy([0,iters+1],[tol,tol],'r:')
        ax_conv_elec_one_solver.semilogy(err_vec,marker='o',ls='-',color='tab:blue')
        plt.grid(which='major',color='k', linestyle='--', alpha=.2)
        plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
        
        # plot solution
        if number_of_nodes<max_number_of_nodes:
            elec_net.update_full(x_sol)
            plt.figure('Network solution electricity')
            ax_sol = plt.gca()
            elec_net.draw_network_value(ax_sol,halflink_length=0.5,halflink_angle=1)
            plt.axis('equal')
            plt.axis('off')
            plt.plot()
        
    return x_sol, iters, err_vec
        
if __name__ == '__main__':
    max_number_of_nodes = 500 # Maximum number of total nodes for which the topology is plotted, etc. 
    
    dir_path = os.path.dirname(os.path.realpath(__file__))     
    solve_EN_1C(dir_path,max_number_of_nodes) 
    
    plt.show()
