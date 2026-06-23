"""Network consisting of one quarter, based on the HN_1Q data."""
from meslf.networks.read_write_network import from_pd_dataframes
from meslf.networks.heat_network import HeatLink
import pandas as pd
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

from meslf.load_flow.system_of_equations import NonLinearSystemHeat

import argparse # To get arguments (as list) from command line

command_line_input = argparse.ArgumentParser()

command_line_input.add_argument(
    "--max_nodes", # name on the command line Drop the `--` for positional/required parameters
    type=int,
    default = 50, # default if nothing is provided
    )
command_line_input.add_argument(
    "--save_fig", # name on the command line Drop the `--` for positional/required parameters
    type=bool,
    default = False, # default if nothing is provided
    )
command_line_input.add_argument(
    "--p_low", # name on the command line Drop the `--` for positional/required parameters
    type=float,
    default = .9, # default if nothing is provided
    )
command_line_input.add_argument(
    "--p_high", # name on the command line Drop the `--` for positional/required parameters
    type=float,
    default = .95, # default if nothing is provided
    )
command_line_input.add_argument(
    "--T_low", # name on the command line Drop the `--` for positional/required parameters
    type=float,
    default = .9, # default if nothing is provided
    )
command_line_input.add_argument(
    "--T_high", # name on the command line Drop the `--` for positional/required parameters
    type=float,
    default = .95, # default if nothing is provided
    )
command_line_input.add_argument(
    "--show_plots", # name on the command line Drop the `--` for positional/required parameters
    type=bool,
    default = True, # default if nothing is provided
    )

kW = 1e3 #[W]

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
    path_to_data = os.path.join(dir_path,'network_data','N_1Q')
    nodes = pd.read_pickle(os.path.join(path_to_data, 'HN_1Q_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'HN_1Q_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'HN_1Q_halflinks.pkl'))
    heat_net = from_pd_dataframes(nodes,links,halflinks)
    
    return heat_net 

def initialize_network(heat_net,p_perc_low,p_perc_high,T_perc_low,T_perc_high,formulation='standard'):
    """Initialize the network.
    
    Parameters
    ----------
    heat_net : HeatNetwork
        The network to be initialized
    p_perc_low : float
        Lowest value of fraction of reference pressure that is used in the initial linear pressure profile.
    p_perc_high : float
        Highest value of fraction of reference pressure that is used in the initial linear pressure profile.
    T_perc_low : float
        Lowest value of fraction of reference temperature that is used in the initial linear temperature profile.
    T_perc_high : float
        Highest value of fraction of reference temperature that is used in the initial linear temperature profile.
        
    Returns
    -------
    x0 : np array
        Vector with initial guess
    """
    Ta = heat_net.links[0].link_params.get('Ta')
    heat_net.Ta = Ta
    #print('Ta = {}'.format(Ta))
    water = heat_net.links[0].link_params.get('carrier')
    heat_source = heat_net.nodes[0]
    heat_source.half_links[0].set_type('heat_exchanger',{'carrier':water},bc_type=0)

    x_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks = heat_net.get_x_entries(formulation=formulation)
    m_init = 1e-2*np.ones(len(unknown_m_links))
    m_hl_init = 1e-2*np.ones(len(unknown_m_halflinks))
    hp_ref = heat_source.p
    Ts_ref = heat_source.Ts
    #print('reference pressure heat: {}'.format(hp_ref))
    hp_init = hp_ref*np.linspace(p_perc_high,p_perc_low,len(unknown_p_nodes)) 
    Ts_init = Ts_ref*np.linspace(T_perc_high,T_perc_low,len(unknown_Ts_nodes))
    Tr_init = 50.*np.linspace(T_perc_low,T_perc_high,len(unknown_Tr_nodes))
    x_init = np.concatenate((m_init,m_hl_init,hp_init,Ts_init,Tr_init))
    heat_net.initialize()
    heat_net.update(x_init,formulation=formulation)
    x0 = heat_net.set_x_init(formulation=formulation)
    
    return x0

def compare_conv_solvers(dir_path,p_perc_low,p_perc_high,T_perc_low,T_perc_high,max_number_of_nodes,save_fig):
    """
    Parameters
    ----------
    p_perc_low : float
        Lowest value of fraction of reference pressure that is used in the initial linear pressure profile.
    p_perc_high : float
        Highest value of fraction of reference pressure that is used in the initial linear pressure profile.
    T_perc_low : float
        Lowest value of fraction of reference temperature that is used in the initial linear temperature profile.
    T_perc_high : float
        Highest value of fraction of reference temperature that is used in the initial linear temperature profile.    
    max_number_of_nodes : int
        Maximum number of total nodes for which FD is used, and for which the topology is plotted, etc.
    save_fig : bool
        If True, the convergence plot is saved as a pgf. 
    """
    # create heat network from data
    heat_net = create_network(dir_path)
    number_of_nodes = len(heat_net.nodes)

    formulation = 'standard'
    formulation_hl = 'half_link_flow'
    
    # initialize
    x0 = initialize_network(heat_net,p_perc_low,p_perc_high,T_perc_low,T_perc_high,formulation=formulation)

    if number_of_nodes<max_number_of_nodes:
        for node in heat_net.get_nodes():
            print('node {} with node type {}'.format(node.name,node.node_type))
            for hl in node.get_half_links():
                print('half link {} with type {}, and m = {}, phi = {}, To = {}'.format(hl.name,hl.link_type, hl.m, hl.phi, hl.To))
        for link in heat_net.get_links():
                print('link {} with link type {}, and L = {}, m = {}, D = {}, U = {}, Ta = {}'.format(link.name,link.link_type,link.link_params['L'],link.m,link.link_params['D'],link.link_params['U'],link.link_params['Ta']))
        # also plot initial condition on the networks
        fig_heat_init = plt.figure('Initial guess heat network')
        ax_heat_init = plt.gca()
        heat_net.draw_network_value(ax_heat_init)
        plt.axis('equal')
        plt.axis('off')
        plt.plot()
        
    # scaling
    heat_source = heat_net.nodes[0]
    hp_ref = heat_source.p
    mbase = 1e-2
    pbase = hp_ref
    Tbase = 100.
    phibase = 1.*kW
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase,'mbase':mbase,'phbase':pbase}
    x0_pu = heat_net.set_x_init(scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)
    
    if number_of_nodes<max_number_of_nodes:
        print('x0_pu = {}'.format(x0_pu))
        nlsys = NonLinearSystemHeat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)
        print('F(x0_pu) = {}'.format(nlsys.F(x0_pu)))
        J0 = nlsys.J(x0_pu)
        J0_FD = nlsys.J_FD(x0_pu,1e-10)
        print('J(x0_pu) = {}'.format(J0))
        print('cond(J(x0_pu)) = {}, det(J(x0_pu)) = {}'.format(np.linalg.cond(J0.todense()),np.linalg.det(J0.todense())))
        print('cond(J_FD(x0_pu)) = {}, det(J_FD(x0_pu)) = {}'.format(np.linalg.cond(J0_FD),np.linalg.det(J0_FD)))
        print('nans at: {}'.format(np.where(np.isnan(J0.todense()))))
        print('nans in J_FD at: {}'.format(np.where(np.isnan(J0_FD))))
        fig_spy = plt.figure('Spy plot Jacobian heat')
        ax_spy = fig_spy.gca()
        plt.spy(J0)
        nlsys.plot_J_overlay(ax_spy)

        J_diff = J0-J0_FD
        fig_J = plt.figure('Jacobian difference heat (using per unit scaling)')
        plt.imshow(J_diff)
        ax_J = plt.gca()
        nlsys.plot_J_overlay(ax_J)
        plt.colorbar()

        # solve network
        tol = 1e-6
        max_iter = 10
        x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,To_hl_vec = heat_net.solve_network(tol,max_iter,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)

        x_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes = heat_net.get_x_entries(formulation=formulation)
        F_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks = heat_net.get_F_entries(formulation=formulation)
        #print('errors: {}'.format(err_vec))

        print('x sol = {}'.format(x_sol))
        print('x_m = {}, x_p = {}, x_Ts = {}, x_Tr = {}'.format(x_sol[0:len(unknown_m_links)],x_sol[len(unknown_m_links):len(unknown_m_links)+len(unknown_p_nodes)],x_sol[len(unknown_m_links)+len(unknown_p_nodes):len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes)],x_sol[len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes):len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes)+len(unknown_Tr_nodes)]))
        J_sol = nlsys.J(x_sol)
        J_FD = nlsys.J_FD(x_sol,1e-6)
        print('J(x_sol) = {}'.format(J_sol))
        print('cond(J(x_sol)) = {}, det(J(x_sol)) = {}'.format(np.linalg.cond(J_sol.todense()),np.linalg.det(J_sol.todense())))
        print('cond(J_FD(x_sol)) = {}, det(J_FD(x_sol)) = {}'.format(np.linalg.cond(J_FD),np.linalg.det(J_FD)))

        print('submatrices:')
        print('dFm_dm = {}'.format(J_sol[0:len(F_m_nodes),0:len(unknown_m_links)]))
        print('dFm_dTs = {}'.format(J_sol[0:len(F_m_nodes),len(unknown_m_links)+len(unknown_p_nodes):len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes)]))
        print('dFm_dTr = {}'.format(J_sol[0:len(F_m_nodes),len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes):len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes)+len(unknown_Tr_nodes)]))
        print('dTs_dm = {}'.format(J_sol[len(F_m_nodes)+len(F_deltap_links):len(F_m_nodes)+len(F_deltap_links)+len(F_Ts_nodes),0:len(unknown_m_links)]))
        print('dTs_dTs = {}'.format(J_sol[len(F_m_nodes)+len(F_deltap_links):len(F_m_nodes)+len(F_deltap_links)+len(F_Ts_nodes),len(unknown_m_links)+len(unknown_p_nodes):len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes)]))
        print('dTs_dTr = {}'.format(J_sol[len(F_m_nodes)+len(F_deltap_links):len(F_m_nodes)+len(F_deltap_links)+len(F_Ts_nodes),len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes):len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes)+len(unknown_Tr_nodes)]))
        print('dTr_dm = {}'.format(J_sol[len(F_m_nodes)+len(F_deltap_links)+len(F_Ts_nodes):len(F_m_nodes)+len(F_deltap_links)+len(F_Ts_nodes)+len(F_Tr_nodes),0:len(unknown_m_links)]))
        print('dTr_dTs = {}'.format(J_sol[len(F_m_nodes)+len(F_deltap_links)+len(F_Ts_nodes):len(F_m_nodes)+len(F_deltap_links)+len(F_Ts_nodes)+len(F_Tr_nodes),len(unknown_m_links)+len(unknown_p_nodes):len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes)]))
        print('dTr_dTr = {}'.format(J_sol[len(F_m_nodes)+len(F_deltap_links)+len(F_Ts_nodes):len(F_m_nodes)+len(F_deltap_links)+len(F_Ts_nodes)+len(F_Tr_nodes),len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes):len(unknown_m_links)+len(unknown_p_nodes)+len(unknown_Ts_nodes)+len(unknown_Tr_nodes)]))
        print('nans at: {}'.format(np.where(np.isnan(J_sol.todense()))))

        print('errors: {}'.format(err_vec))
        print('x sol = {}'.format(x_sol))
        for node in heat_net.get_nodes():
            print('node {} with node type {}'.format(node.name,node.node_type))
            for hl in node.get_half_links():
                print('half link {} with type {}, and m = {}, phi = {}, To = {}'.format(hl.name,hl.link_type, hl.m, hl.phi, hl.To))

        # plot the network
        plt.figure('Network topology heat')
        ax = plt.gca()
        heat_net.draw_network(ax)
        plt.axis('equal')
        plt.axis('off')
        plt.plot()

    # compare convergence
    h = 1e-6
    tol = 1e-6
    max_iter = 50
    # solve when everything is specified in S.I., unscaled
    print('\nSolving heat network, unscaled')
    heat_net.reset_network(x0,formulation=formulation)
    x_sol_SI,iters_SI,err_vec_SI,_,_,_,_,_,_,_,_ = heat_net.solve_network(tol,max_iter,formulation=formulation)
    # solve when everything is specified in per unit
    print('\nSolving heat network, per unit')
    heat_net.reset_network(x0_pu,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)
    x_sol_pu,iters_pu,err_vec_pu,_,_,_,_,_,_,_,_  = heat_net.solve_network(tol,max_iter,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)
    # solve when everything is specified in S.I., using scaling in solver
    print('\nSolving heat network, scaling in solver')
    heat_net.reset_network(x0,formulation=formulation)
    x_sol_scaled,iters_scaled,err_vec_scaled,_,_,_,_,_,_,_,_ = heat_net.solve_network(tol,max_iter,scale_var='matrix',scale_var_params=scale_var_params,formulation=formulation)
    if number_of_nodes<max_number_of_nodes:
        # solve when everything is specified in per unit, using FD
        print('\nSolving heat network, per unit, finite-difference Jacobian')
        heat_net.reset_network(x0_pu,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)
        x_sol_pu_FD,iters_pu_FD,err_vec_pu_FD,_,_,_,_,_,_,_  = heat_net.solve_network(tol,max_iter,scale_var=scale_var,scale_var_params=scale_var_params,solver='NR_FD',h=h,formulation=formulation)
        # solve when everything is specified in S.I., using scaling in solver
        print('\nSolving heat network, scaling in solver, finite-difference Jacobian')
        heat_net.reset_network(x0,formulation=formulation)
        x_sol_scaled_FD,iters_scaled_FD,err_vec_scaled_FD,_,_,_,_,_,_,_,_ = heat_net.solve_network(tol,max_iter,scale_var='matrix',scale_var_params=scale_var_params,solver='NR_FD',h=h,formulation=formulation)
    # using 'half_link_flow' formulation
    heat_net.reset_network(x0,formulation=formulation)
    x0_hl = initialize_network(heat_net,p_perc_low,p_perc_high,T_perc_low,T_perc_high,formulation=formulation_hl)
    # solve when everything is specified in S.I., unscaled
    print('\nSolving heat network, unscaled')
    heat_net.reset_network(x0_hl,formulation=formulation_hl)
    x_sol_SI_hl,iters_SI_hl,err_vec_SI_hl,_,_,_,_,_,_,_,_ = heat_net.solve_network(tol,max_iter,formulation=formulation_hl)
    # solve when everything is specified in per unit
    print('\nSolving heat network, per unit')
    heat_net.reset_network(x0_hl,formulation=formulation_hl)
    x0_pu_hl = heat_net.set_x_init(scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation_hl)
    x_sol_pu_hl,iters_pu_hl,err_vec_pu_hl,_,_,_,_,_,_,_,_  = heat_net.solve_network(tol,max_iter,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation_hl)
    # solve when everything is specified in S.I., using scaling in solver
    print('\nSolving heat network, scaling in solver')
    heat_net.reset_network(x0_hl,formulation=formulation_hl)
    x_sol_scaled_hl,iters_scaled_hl,err_vec_scaled_hl,_,_,_,_,_,_,_,_ = heat_net.solve_network(tol,max_iter,scale_var='matrix',scale_var_params=scale_var_params,formulation=formulation_hl)
    if number_of_nodes<max_number_of_nodes:
        # solve when everything is specified in per unit, using FD
        print('\nSolving heat network, per unit, finite-difference Jacobian')
        heat_net.reset_network(x0_pu_hl,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation_hl)
        x_sol_pu_FD_hl,iters_pu_FD_hl,err_vec_pu_FD_hl,_,_,_,_,_,_,_  = heat_net.solve_network(tol,max_iter,scale_var=scale_var,scale_var_params=scale_var_params,solver='NR_FD',h=h,formulation=formulation_hl)
        # solve when everything is specified in S.I., using scaling in solver
        print('\nSolving heat network, scaling in solver, finite-difference Jacobian')
        heat_net.reset_network(x0_hl,formulation=formulation_hl)
        x_sol_scaled_FD_hl,iters_scaled_FD_hl,err_vec_scaled_FD_hl,_,_,_,_,_,_,_,_ = heat_net.solve_network(tol,max_iter,scale_var='matrix',scale_var_params=scale_var_params,solver='NR_FD',h=h,formulation=formulation_hl)
        
    # plot convergence
    nS = len([link for link in heat_net.nodes[0].get_out_links() if isinstance(link,HeatLink)])
    n = int((len([node for node in heat_net.get_nodes(node_types=[1]) if (len(list(node.get_in_links()))==1 and len(list(node.get_out_links()))==1)]))/nS)
    m = int(2*n+1 - (number_of_nodes-1)/nS) #int((len([node for node in heat_net.get_nodes(node_types=[2]) if len(list(node.get_out_links())) > 1]))/nS)
    fig_conv_heat = plt.figure('Convergence plot heat (n = {}, m = {}, nS = {}, max iters = {})'.format(n,m,nS,max_iter))
    ax_conv_heat = fig_conv_heat.gca()
    plt.cla()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    if number_of_nodes<max_number_of_nodes:
        max_iter_used = np.max([iters_pu,iters_SI,iters_scaled,iters_pu_FD,iters_scaled_FD,iters_pu_hl,iters_SI_hl,iters_scaled_hl,iters_pu_FD_hl,iters_scaled_FD_hl])
    else:
        max_iter_used = np.max([iters_pu,iters_SI,iters_scaled,iters_pu_hl,iters_SI_hl,iters_scaled_hl])
    ax_conv_heat.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax_conv_heat.semilogy(np.asarray(range(0,iters_pu+1)),err_vec_pu,marker='o',ls='-',color='tab:green',label='p.u.')
    ax_conv_heat.semilogy(np.asarray(range(0,iters_SI+1)),err_vec_SI,marker='o',ls='-',color='tab:orange',label='S.I., unscaled')
    ax_conv_heat.semilogy(np.asarray(range(0,iters_scaled+1)),err_vec_scaled,marker='o',ls='-',color='tab:blue',label='S.I., scaled solver')
    if number_of_nodes<max_number_of_nodes:
        ax_conv_heat.semilogy(np.asarray(range(0,iters_pu_FD+1)),err_vec_pu_FD,marker='o',ls='--',color='tab:green',label='p.u. FD')
        ax_conv_heat.semilogy(np.asarray(range(0,iters_scaled_FD+1)),err_vec_scaled_FD,marker='o',ls='--',color='tab:blue',label='S.I., scaled solver FD')
    ax_conv_heat.semilogy(np.asarray(range(0,iters_pu_hl+1)),err_vec_pu_hl,marker='d',ls='-',color='tab:green',label='p.u., hl')
    ax_conv_heat.semilogy(np.asarray(range(0,iters_SI_hl+1)),err_vec_SI_hl,marker='d',ls='-',color='tab:orange',label='S.I., unscaled, hl')
    ax_conv_heat.semilogy(np.asarray(range(0,iters_scaled_hl+1)),err_vec_scaled_hl,marker='d',ls='-',color='tab:blue',label='S.I., scaled solver, hl')
    if number_of_nodes<max_number_of_nodes:
        ax_conv_heat.semilogy(np.asarray(range(0,iters_pu_FD_hl+1)),err_vec_pu_FD_hl,marker='d',ls='--',color='tab:green',label='p.u. FD, hl')
        ax_conv_heat.semilogy(np.asarray(range(0,iters_scaled_FD_hl+1)),err_vec_scaled_FD_hl,marker='d',ls='--',color='tab:blue',label='S.I., scaled solver FD, hl')
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)

    if save_fig:
        # enlarge figure to maximum size to prevent the legend and labels from being to crowded (TkAgg backend)
        manager = plt.get_current_fig_manager()
        screen_y = manager.window.winfo_screenheight()
        screen_x = manager.window.winfo_screenwidth()
        d = 10  # width of the window border in pixels
        manager.resize(*(screen_x/2 - 2*d,screen_y - 4*d)) # screen_x/2 to compensate for using 2 monitors
        # figure setting to save figure
        fig_width_pt = 469.75502  #[pt] Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]
        pgf_with_latex = {
            "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
            "text.usetex": False,                # True: use LaTeX to write all text
            "font.family": "DejaVu Sans",#"serif",
            "font.serif": [],                   # blank entries should cause plots
            "font.monospace": [],               # to inherit fonts from the document
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.loc": "upper right",
            "legend.fontsize": 9,               # Make the legend/label fonts
            "xtick.labelsize": 10,               # a little smaller
            "ytick.labelsize": 10,
            "figure.figsize": fig_size,
            }
        mpl.rcParams.update(pgf_with_latex)

        path_to_fig = os.path.join(dir_path,'Figures','N_1Q')
        file_name = 'convergence_plot_HN_1Q'+'_n'+str(n)+'_m'+str(m)+'_nS'+str(nS)+'_'+str(int(p_perc_low*100))+'ppercent'+'_'+str(int(T_perc_low*100))+'Tpercent'+'.pgf'
        plt.savefig(os.path.join(path_to_fig, file_name))
        
    return x_sol_scaled

def solve_HN_1Q(dir_path,p_perc_low,p_perc_high,T_perc_low,T_perc_high,save_fig,formulation='standard'):
    """Solve this example, using fb and scaling in the solver.
    
    Parameters
    ----------
    p_perc_low : float
        Lowest value of fraction of reference pressure that is used in the initial linear pressure profile.
    p_perc_high : float
        Highest value of fraction of reference pressure that is used in the initial linear pressure profile.
    T_perc_low : float
        Lowest value of fraction of reference temperature that is used in the initial linear temperature profile.
    T_perc_high : float
        Highest value of fraction of reference temperature that is used in the initial linear temperature profile.
    save_fig : bool
        If True, the convergence plot is saved as a pgf. 
        
    Returns
    -------
    x_sol : np array
        Solution vector
    """
    # create heat network from data
    heat_net = create_network(dir_path)
    
    # initialize
    x0 = initialize_network(heat_net,p_perc_low,p_perc_high,T_perc_low,T_perc_high,formulation=formulation)
    
    # scaling in solver
    heat_source = heat_net.nodes[0]
    hp_ref = heat_source.p
    mbase = 1e-2
    pbase = hp_ref
    Tbase = 100.
    phibase = 1.*kW
    # solve the system
    tol = 1e-6
    max_iter = 50
    x_sol,iters,err_vec,_,_,_,_,_,_,_,_ = heat_net.solve_network(tol,max_iter,scale_var='matrix',scale_var_params={'mbase':mbase,'phbase':pbase,'Tbase':Tbase,'phibase':phibase},formulation=formulation)
    
    # plot convergence
    nS = len([link for link in heat_net.nodes[0].get_out_links() if isinstance(link,HeatLink)])
    n = int((len([node for node in heat_net.get_nodes(node_types=[1]) if (len(list(node.get_in_links()))==1 and len(list(node.get_out_links()))==1)]))/nS)
    m = int(2*n+1 - (len(heat_net.nodes)-1)/nS) #int((len([node for node in heat_net.get_nodes(node_types=[2]) if len(list(node.get_out_links())) > 1]))/nS)
    fig_conv_heat_one_solver = plt.figure('Convergence plot heat (n = {}, m = {}, nS = {}, max iters = {}, len(x)={})'.format(n,m,nS,max_iter,len(x0)))
    ax_conv_heat_one_solver = fig_conv_heat_one_solver.gca()
    plt.cla()
    plt.xlabel('Iteration k')
    plt.ylabel('Error $||D_F F(x^k)||_2$')
    ax_conv_heat_one_solver.semilogy([0,iters+1],[tol,tol],'r:')
    ax_conv_heat_one_solver.semilogy(err_vec,marker='o',ls='-',color='tab:blue')
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    if save_fig:
        # enlarge figure to maximum size to prevent the legend and labels from being to crowded (TkAgg backend)
        manager = plt.get_current_fig_manager()
        screen_y = manager.window.winfo_screenheight()
        screen_x = manager.window.winfo_screenwidth()
        d = 10  # width of the window border in pixels
        manager.resize(*(screen_x/2 - 2*d,screen_y - 4*d)) # screen_x/2 to compensate for using 2 monitors
        # figure setting to save figure
        fig_width_pt = 469.75502  #[pt] Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]
        pgf_with_latex = {
            "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
            "text.usetex": False,                # True: use LaTeX to write all text
            "font.family": "DejaVu Sans",#"serif",
            "font.serif": [],                   # blank entries should cause plots
            "font.monospace": [],               # to inherit fonts from the document
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.loc": "upper right",
            "legend.fontsize": 9,               # Make the legend/label fonts
            "xtick.labelsize": 10,               # a little smaller
            "ytick.labelsize": 10,
            "figure.figsize": fig_size,
            }
        mpl.rcParams.update(pgf_with_latex)

        path_to_fig = os.path.join(dir_path,'Figures','N_1Q')
        file_name = 'convergence_plot_HN_1Q'+'_n'+str(n)+'_m'+str(m)+'_nS'+str(nS)+'_'+str(int(p_perc_low*100))+'ppercent'+'_'+str(int(T_perc_low*100))+'Tpercent'+'_one_solver'+'.pgf'
        plt.savefig(os.path.join(path_to_fig, file_name))
    return x_sol

if __name__ == '__main__':
    # parse the command line
    args = command_line_input.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    compare_conv_solvers(dir_path,args.p_low,args.p_high,args.T_low,args.T_high,args.max_nodes,args.save_fig)

    if args.show_plots:
        plt.show()
