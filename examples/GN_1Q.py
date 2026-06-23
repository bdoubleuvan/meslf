"""Network consisting of one city, based on the GN_1Q data."""
from meslf.networks.read_write_network import from_pd_dataframes
import pandas as pd
import os.path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from meslf.networks.gas_network import GasNode, GasLink
import scipy.sparse as sps
import argparse # To get arguments (as list) from command line

mbar = 1e2 #[Pa]
command_line_input = argparse.ArgumentParser()
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
    "--max_nodes", # name on the command line Drop the `--` for positional/required parameters
    type=int,
    default = 50, # default if nothing is provided
    )
command_line_input.add_argument(
    "--fb", # name on the command line Drop the `--` for positional/required parameters
    type=bool,
    default = True, # default if nothing is provided
    )
command_line_input.add_argument(
    "--save_fig", # name on the command line Drop the `--` for positional/required parameters
    type=bool,
    default = False, # default if nothing is provided
    )
command_line_input.add_argument(
    "--print_sol", # name on the command line Drop the `--` for positional/required parameters
    type=bool,
    default = False, # default if nothing is provided
    )
command_line_input.add_argument(
    "--show_plots", # name on the command line Drop the `--` for positional/required parameters
    type=bool,
    default = True, # default if nothing is provided
    )

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
    path_to_data = os.path.join(dir_path,'network_data','N_1Q')
    nodes = pd.read_pickle(os.path.join(path_to_data, 'GN_1Q_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'GN_1Q_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'GN_1Q_halflinks.pkl'))
    gas_net = from_pd_dataframes(nodes,links,halflinks)
    
    return gas_net

def initialize_network(gas_net,p_perc_low,p_perc_high,use_fa=True):
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
    gas_source = gas_net.nodes[0]
    gas = gas_net.links[0].link_params.get('carrier')
    p_ref = gas_source.p #[bar]
    #print('ref. pressure = {} [bar]'.format(p_ref))
    gas_net.initialize()
    if use_fa:
        # nodal
        x_entries_nodal = gas_net.get_x_entries(formulation='nodal')
        unknown_p_nodes_nodal = x_entries_nodal
        p_init_nodal = p_ref*np.linspace(p_perc_high,p_perc_low,len(unknown_p_nodes_nodal))
        x_init_nodal = p_init_nodal
        gas_net.update(x_init_nodal,formulation='nodal')
        x0_nodal = gas_net.set_x_init(formulation='nodal')
    else:
        x0_nodal = np.array([])
    # full
    x_entries_full = gas_net.get_x_entries(formulation='full')
    unknown_p_nodes_full = []
    unknown_q_links_full = []
    for el in x_entries_full:
        if isinstance(el,GasNode):
            unknown_p_nodes_full.append(el)
        elif isinstance(el,GasLink):
            unknown_q_links_full.append(el)
    p_init_full = p_ref*np.linspace(p_perc_high,p_perc_low,len(unknown_p_nodes_full))
    q_init_full = 1e-5*np.ones(len(unknown_q_links_full))#1e-4*np.ones(len(unknown_q_links_full))
    x_init_full = np.concatenate((q_init_full,p_init_full))
    gas_net.update(x_init_full,formulation='full')
    x0_full = gas_net.set_x_init(formulation='full')
    
    return x0_nodal, x0_full

def compare_conv_solvers(dir_path,p_perc_low,p_perc_high,max_number_of_nodes,use_fb,save_fig,print_sol):
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
    gas_net = create_network(dir_path)
    number_of_nodes = len(gas_net.nodes)

    # initalize network
    x0_nodal, x0_full = initialize_network(gas_net,p_perc_low,p_perc_high)
    
    if number_of_nodes<max_number_of_nodes:
        for node in gas_net.get_nodes():
            print('Node {} with node type {} and p = {:2e} [Pa]'.format(node.name,node.node_type,node.p))
        for link in gas_net.get_links():
            print('Link {} with link type {} and paramaters: {}'.format(link.name,link.link_type,link.link_params))

        # plot the gas network
        plt.figure('Network topology gas')
        ax = plt.gca()
        gas_net.draw_network(ax)
        plt.axis('equal')
        plt.axis('off')
        plt.plot()

    # Scaling
    # per unit
    scale_var = 'per_unit'
    qbase_pu = 1e-4 #[kg/s]
    gas_source = gas_net.nodes[0]
    pbase_pu = gas_source.p #[bar]
    scale_var_params = {'qbase':qbase_pu,'pbase':pbase_pu}
    # scaling in solver
    qbase = qbase_pu
    pbase = pbase_pu
    scale_var_solver = 'matrix'
    scale_var_params_solver = {'qbase':qbase_pu,'pgbase':pbase_pu}

    # compare convergence
    h = 1e-6
    tol = 1e-6
    max_iter = 100
    det_tol = 1e-12
    # nodal
    print('\nSolving gas network using nodal formulation, unscaled')
    gas_net.reset_network(x0_nodal,formulation='nodal')
    x_sol_nodal_SI,iters_nodal_SI,err_vec_nodal_SI,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='nodal',solver='NR',det_tol=det_tol)
    # nodal PU
    print('\nSolving gas network using nodal formulation, per unit')
    gas_net.reset_network(x0_nodal,formulation='nodal')
    x0_nodal_pu = gas_net.set_x_init(formulation='nodal',scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol_nodal_pu,iters_nodal_pu,err_vec_nodal_pu,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='nodal',solver='NR',scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol)
    # nodal scaled solver
    print('\nSolving gas network using nodal formulation, scaling in solver')
    gas_net.reset_network(x0_nodal,formulation='nodal')
    x_sol_nodal_scaled,iters_nodal_scaled,err_vec_nodal_scaled,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='nodal',solver='NR',scale_var=scale_var_solver,scale_var_params=scale_var_params_solver,det_tol=det_tol)
    # full
    print('\nSolving gas network using full formulation, unscaled')
    gas_net.reset_network(x0_full,formulation='full')
    x_sol_full_SI,iters_full_SI,err_vec_full_SI,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='full',solver='NR',det_tol=det_tol)
    # full PU
    print('\nSolving gas network using full formulation, per unit')
    gas_net.reset_network(x0_full,formulation='full')
    x0_full_pu = gas_net.set_x_init(formulation='full',scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol_full_pu,iters_full_pu,err_vec_full_pu,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='full',solver='NR',scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol)
    # full scaled solver (Dx en DF)
    print('\nSolving gas network using full formulation, scaling in solver')
    gas_net.reset_network(x0_full,formulation='full')
    x_sol_full_scaled,iters_full_scaled,err_vec_full_scaled,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='full',solver='NR',scale_var=scale_var_solver,scale_var_params=scale_var_params_solver,det_tol=det_tol)
    if number_of_nodes<max_number_of_nodes:
        # nodal PU FD
        print('\nSolving gas network using nodal formulation, per unit, finite-difference Jacobian')
        gas_net.reset_network(x0_nodal,formulation='nodal')
        x0_nodal_pu = gas_net.set_x_init(formulation='nodal',scale_var=scale_var,scale_var_params=scale_var_params)
        x_sol_nodal_pu_FD,iters_nodal_pu_FD,err_vec_nodal_pu_FD,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='nodal',solver='NR_FD',h=h,scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol)
        # full PU FD
        print('\nSolving gas network using full formulation, per unit, finite-difference Jacobian')
        gas_net.reset_network(x0_full,formulation='full')
        x0_full_pu = gas_net.set_x_init(formulation='full',scale_var=scale_var,scale_var_params=scale_var_params)
        x_sol_full_pu_FD,iters_full_pu_FD,err_vec_full_pu_FD,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='full',solver='NR_FD',h=h,scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol)
    if use_fb:
        # set fb as link equation on all links
        for link in gas_net.get_links():
            link.set_type(link.link_type,link.link_params,link_eq_form='dp_of_q')
        # full fb
        print('\nSolving gas network using full formulation, unscaled, fb')
        gas_net.reset_network(x0_full,formulation='full')
        x_sol_full_SI_fb,iters_full_SI_fb,err_vec_full_SI_fb,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='full',solver='NR',det_tol=det_tol)
        # full PU fb
        print('\nSolving gas network using full formulation, per unit, fb')
        gas_net.reset_network(x0_full,formulation='full')
        x_sol_full_pu_fb,iters_full_pu_fb,err_vec_full_pu_fb,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='full',solver='NR',scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol)
        # full scaled solver (Dx en DF) fb
        print('\nSolving gas network using full formulation, scaling in solver, fb')
        gas_net.reset_network(x0_full,formulation='full')
        x_sol_full_scaled_fb,iters_full_scaled_fb,err_vec_full_scaled_fb,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='full',solver='NR',scale_var=scale_var_solver,scale_var_params=scale_var_params_solver,det_tol=det_tol)
        if number_of_nodes<max_number_of_nodes:
            # full PU FD
            print('\nSolving gas network using full formulation, per unit, finite-difference Jacobian, fb')
            gas_net.reset_network(x0_full,formulation='full')
            x_sol_full_pu_FD_fb,iters_full_pu_FD_fb,err_vec_full_pu_FD_fb,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='full',solver='NR_FD',h=h,scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol)
        # set fa (back) as link equation on all links
        for link in gas_net.get_links():
            link.set_type(link.link_type,link.link_params,link_eq_form='q_of_dp')

    nS = len([link for link in gas_source.get_out_links() if isinstance(link,GasLink)])
    n = int((len([node for node in gas_net.get_nodes(node_types=[1]) if (len(list(node.get_in_links()))==1 and len(list(node.get_out_links()))==1)]))/nS)
    m = int(2*n+1 - (number_of_nodes-1)/nS) #int((len([node for node in gas_net.get_nodes(node_types=[1]) if len(list(node.get_out_links())) > 2]))/nS)
    fig_conv_gas = plt.figure('Convergence plot gas (n = {}, m = {}, nS = {}, max iters = {})'.format(n,m,nS,max_iter))
    # possible colors: one of {'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'} which are the Tableau Colors from the ‘T10’ categorical palette (which is the default color cycle).
    # fa = '*'
    # fb = 'o'
    # nodal = 's'
    # unscaled = 'tab:orange'
    # p.u. = 'tab:green'
    # scaled solver = 'tab:blue'
    # F.D = '--' and 'tab:red'
    ax_conv_gas = fig_conv_gas.gca()
    plt.cla()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    if number_of_nodes<max_number_of_nodes and use_fb:
        max_iter_used = np.max([iters_nodal_pu,iters_nodal_SI,iters_nodal_scaled,iters_full_pu,iters_full_SI,iters_full_scaled,iters_nodal_pu_FD,iters_full_pu_FD,iters_full_pu_fb,iters_full_SI_fb,iters_full_scaled_fb,iters_full_pu_FD_fb])
    elif number_of_nodes<max_number_of_nodes:
        max_iter_used = np.max([iters_nodal_pu,iters_nodal_SI,iters_nodal_scaled,iters_full_pu,iters_full_SI,iters_full_scaled,iters_nodal_pu_FD,iters_full_pu_FD])
    elif use_fb:
        max_iter_used = np.max([iters_nodal_pu,iters_nodal_SI,iters_nodal_scaled,iters_full_pu,iters_full_SI,iters_full_scaled,iters_full_pu_fb,iters_full_SI_fb,iters_full_scaled_fb])
    else:
        max_iter_used = np.max([iters_nodal_pu,iters_nodal_SI,iters_nodal_scaled,iters_full_pu,iters_full_SI,iters_full_scaled])
    ax_conv_gas.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax_conv_gas.semilogy(np.asarray(range(0,iters_nodal_pu+1)),err_vec_nodal_pu,marker='s',ls='-',color='tab:green',label='nodal p.u.')
    if number_of_nodes<max_number_of_nodes:
        ax_conv_gas.semilogy(np.asarray(range(0,iters_nodal_pu_FD+1)),err_vec_nodal_pu_FD,marker='s',ls='--',color='tab:red',label='nodal p.u. FD')
    ax_conv_gas.semilogy(np.asarray(range(0,iters_nodal_SI+1)),err_vec_nodal_SI,marker='s',ls='-',color='tab:orange',label='nodal  S.I., unscaled')
    ax_conv_gas.semilogy(np.asarray(range(0,iters_nodal_scaled+1)),err_vec_nodal_scaled,marker='s',ls='-',color='tab:blue',label='nodal S.I., scaled solver')
    ax_conv_gas.semilogy(np.asarray(range(0,iters_full_pu+1)),err_vec_full_pu,marker='*',ls='-',color='tab:green',label='full p.u.')
    ax_conv_gas.semilogy(np.asarray(range(0,iters_full_SI+1)),err_vec_full_SI,marker='*',ls='-',color='tab:orange',label='full  S.I., unscaled')
    ax_conv_gas.semilogy(np.asarray(range(0,iters_full_scaled+1)),err_vec_full_scaled,marker='*',ls='-',color='tab:blue',label='full S.I., scaled solver')
    if number_of_nodes<max_number_of_nodes:
        ax_conv_gas.semilogy(np.asarray(range(0,iters_full_pu_FD+1)),err_vec_full_pu_FD,marker='*',ls='--',color='tab:red',label='full p.u. FD')
    if use_fb:
        ax_conv_gas.semilogy(np.asarray(range(0,iters_full_pu_fb+1)),err_vec_full_pu_fb,marker='o',ls='-',color='tab:green',label='full p.u., fb')
        ax_conv_gas.semilogy(np.asarray(range(0,iters_full_SI_fb+1)),err_vec_full_SI_fb,marker='o',ls='-',color='tab:orange',label='full  S.I., unscaled, fb')
        ax_conv_gas.semilogy(np.asarray(range(0,iters_full_scaled_fb+1)),err_vec_full_scaled_fb,marker='o',ls='-',color='tab:blue',label='full S.I., scaled solver, fb')
        if number_of_nodes<max_number_of_nodes:
            ax_conv_gas.semilogy(np.asarray(range(0,iters_full_pu_FD_fb+1)),err_vec_full_pu_FD_fb,marker='o',ls='--',color='tab:red',label='full p.u. FD, fb')
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
        if use_fb:
            file_name = 'convergence_plot_GN_1Q'+'_n'+str(n)+'_m'+str(m)+'_nS'+str(nS)+'_'+str(int(p_perc_low*100))+'percent'+'_fb'+'.pgf'
        else:
            file_name = 'convergence_plot_GN_1Q'+'_n'+str(n)+'_m'+str(m)+'_nS'+str(nS)+'_'+str(int(p_perc_low*100))+'percent'+'.pgf'
        plt.savefig(os.path.join(path_to_fig, file_name))
     
    if print_sol:
        gas_net.update_full(x_sol_full_scaled_fb,formulation='full')
        print('\nSolution of gas network based on full formulation, using fb, and scaling in solver:')
        for node in gas_net.get_nodes():
            print('node {} with p = {}[mbar] and q = {}[kg/s]'.format(node.name,node.p/mbar,node.half_links[0].q))
        for link in gas_net.get_links():
            print('link {} with q = {}[kg/s]'.format(link.name,link.q))
    
    if use_fb:
        return x_sol_full_scaled_fb
    else:
        return x_sol_full_scaled

def solve_GN_1Q(dir_path,p_perc_low,p_perc_high,save_fig):
    """Solve this example, using fb and scaling in the solver.
    
    Parameters
    ----------
    p_perc_low : float
        Lowest value of fraction of reference pressure that is used in the initial linear pressure profile.
    p_perc_high : float
        Highest value of fraction of reference pressure that is used in the initial linear pressure profile.
    save_fig : bool
        If True, the convergence plot is saved as a pgf. 
        
    Returns
    -------
    x_sol : np array
        Solution vector
    """
    # create gas network from data
    gas_net = create_network(dir_path)
    # set fb as link equation on all links
    for link in gas_net.get_links():
        link.set_type(link.link_type,link.link_params,link_eq_form='dp_of_q')
    # initalize network
    _, x0_full = initialize_network(gas_net,p_perc_low,p_perc_high,use_fa=False)
    # scaling in solver
    qbase = 1e-4
    gas_source = gas_net.nodes[0]
    pbase = gas_source.p
    scale_var_solver = 'matrix'
    scale_var_params_solver = {'qbase':qbase,'pgbase':pbase}
    # solve the system
    tol = 1e-6
    max_iter = 100
    det_tol = 1e-12
    x_sol,iters,err_vec,_,_,_ = gas_net.solve_network(tol,max_iter,formulation='full',solver='NR',scale_var=scale_var_solver,scale_var_params=scale_var_params_solver,det_tol=det_tol)
    nS = len([link for link in gas_source.get_out_links() if isinstance(link,GasLink)])
    n = int((len([node for node in gas_net.get_nodes(node_types=[1]) if (len(list(node.get_in_links()))==1 and len(list(node.get_out_links()))==1)]))/nS)
    m = int(2*n+1 - (len(gas_net.nodes)-1)/nS) #int((len([node for node in gas_net.get_nodes(node_types=[1]) if len(list(node.get_out_links())) > 2]))/nS)
    # plot convergence
    fig_conv_gas_one_solver = plt.figure('Convergence plot gas (n = {}, m = {}, nS = {}, max iters = {}, len(x) = {})'.format(n,m,nS,max_iter,len(x0_full)))
    ax_conv_gas_one_solver = fig_conv_gas_one_solver.gca()
    plt.cla()
    plt.xlabel('Iteration k')
    plt.ylabel('Error $||D_F F(x^k)||_2$')
    ax_conv_gas_one_solver.semilogy([0,iters+1],[tol,tol],'r:')
    ax_conv_gas_one_solver.semilogy(err_vec,marker='o',ls='-',color='tab:blue')
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
        file_name = 'convergence_plot_GN_1Q'+'_n'+str(n)+'_m'+str(m)+'_nS'+str(nS)+'_'+str(int(p_perc_low*100))+'percent'+'_one_solver'+'.pgf'
        plt.savefig(os.path.join(path_to_fig, file_name))
    return x_sol
        
if __name__ == '__main__':
    # parse the command line
    args = command_line_input.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    compare_conv_solvers(dir_path,args.p_low,args.p_high,args.max_nodes,args.fb,args.save_fig,args.print_sol)

    if args.show_plots:
        plt.show()
