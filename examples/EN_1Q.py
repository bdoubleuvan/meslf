"""Network consisting of one quarter, based on the EN_1Q data."""
from meslf.networks.read_write_network import from_pd_dataframes
from meslf.networks.electrical_network import ElectricalLink
import pandas as pd
import numpy as np
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt

from meslf.load_flow.system_of_equations import NonLinearSystemElectrical
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
    "--show_plots", # name on the command line Drop the `--` for positional/required parameters
    type=bool,
    default = True, # default if nothing is provided
    )

def create_network(dir_path):
    """Create the electrical network from the data
    
    Parameters
    ----------
    dir_path : string
        Path to the directory where this file is located
        
    Returns 
    -------
    elec_net : GasNetwork
        The network created from the data
    """
    path_to_data = os.path.join(dir_path,'network_data','N_1Q')
    nodes = pd.read_pickle(os.path.join(path_to_data, 'EN_1Q_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'EN_1Q_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'EN_1Q_halflinks.pkl'))
    elec_net = from_pd_dataframes(nodes,links,halflinks)
    return elec_net

def initialize_network(elec_net):
    """Initialize the network.
    
    Parameters
    ----------
    elec_net : ElectricalNetwork
        The network to be initialized
    
    Returns
    -------
    x0 : np array
        Vector with initial guess
    """
    xe_entries, unknown_delta_nodes, unknown_V_nodes = elec_net.get_x_entries()
    delta_init = -0.2*np.ones(len(unknown_delta_nodes))
    V_init = 1.*np.linspace(0.99,0.95,len(unknown_V_nodes)) # initial voltage amplitudes 1% - 5% from reference value
    x_init = np.concatenate((delta_init,V_init))
    elec_net.initialize()
    elec_net.update(x_init)
    x0 = elec_net.set_x_init()
    return x0
    
def compare_conv_solvers(dir_path,max_number_of_nodes,save_fig):
    """
    Parameters
    ----------
    max_number_of_nodes : int
        Maximum number of total nodes for which FD is used, and for which the topology is plotted, etc.
    save_fig : bool
        If True, the convergence plot is saved as a pgf. 
    """
    # create electrical network from data
    elec_net = create_network(dir_path)
    number_of_nodes = len(elec_net.nodes)
    
    # initialize
    # for link in elec_net.get_links():
    #     link.b /= 1e-3
    #     link.g /= 1e-3
    x0 = initialize_network(elec_net)
    
    if number_of_nodes<max_number_of_nodes:
        for node in elec_net.get_nodes():
            print('node {} with node type {}, V = {}, delta = {}'.format(node.name,node.node_type,node.V,node.delta))
            for hl in node.get_half_links():
                print('With half link {} with half link type {}, and P = {}, Q = {}'.format(hl.name,hl.link_type,hl.P,hl.Q))

        print('x0 = {}'.format(x0))
        nlsys = NonLinearSystemElectrical(elec_net)
        print('F(x0) = {}'.format(nlsys.F(x0)))
        J0 = nlsys.J(x0)
        J0_FD = nlsys.J_FD(x0,1e-6)
        print('J(x0) = {}'.format(J0))
        print('cond(J(x0)) = {}, det(J(x0)) = {}'.format(np.linalg.cond(J0.todense()),np.linalg.det(J0.todense())))
        print('cond(J_FD(x0)) = {}, det(J_FD(x0)) = {}'.format(np.linalg.cond(J0_FD),np.linalg.det(J0_FD)))
        fig_spy = plt.figure('Spy plot Jacobian electrical')
        ax_spy = fig_spy.gca()
        plt.spy(J0)
        nlsys.plot_J_overlay(ax_spy)

    # solve network
    tol = 1e-6
    max_iter = 10
    print('\nSolving electrical network, per unit')
    x_sol,iters,err_vec,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR')
    if number_of_nodes<max_number_of_nodes:
        print('errors: {}'.format(err_vec))
        print('x sol = {}'.format(x_sol))

        for node in elec_net.get_nodes():
            print('node {} with node type {}, V = {}, delta = {}'.format(node.name,node.node_type,node.V,node.delta))
            for hl in node.get_half_links():
                print('With half link {} with half link type {}, and P = {}, Q = {}'.format(hl.name,hl.link_type,hl.P,hl.Q))


        # plot the network
        plt.figure('Network topology electrical')
        ax = plt.gca()
        elec_net.draw_network(ax)
        plt.axis('equal')
        plt.axis('off')
        plt.plot()

    # plot convergence
    nS = len([link for link in elec_net.nodes[0].get_out_links() if isinstance(link,ElectricalLink)])
    n = int((len([node for node in elec_net.get_nodes(node_types=[2]) if (len(list(node.get_in_links()))==1 and len(list(node.get_out_links()))==1)]))/nS)
    m = int(2*n+1 - (number_of_nodes-1)/nS) #int((len([node for node in elec_net.get_nodes(node_types=[2]) if len(list(node.get_out_links())) > 2]))/nS)
    fig_conv_elec = plt.figure('Convergence plot electrical (n = {}, m = {}, nS = {}, max iters = {}, len(x) = {})'.format(n,m,nS,max_iter,len(x0)))
    ax_conv_elec = fig_conv_elec.gca()
    plt.cla()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$)')
    max_iter_used = np.max([iters])
    ax_conv_elec.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax_conv_elec.semilogy(np.asarray(range(0,iters+1)),err_vec,marker='*',ls='-',color='tab:green',label='p.u.')
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
            "legend.loc": "upper right",
            "font.size": 10,
            "legend.fontsize": 9,               # Make the legend/label fonts
            "xtick.labelsize": 10,               # a little smaller
            "ytick.labelsize": 10,
            "figure.figsize": fig_size,
            }
        mpl.rcParams.update(pgf_with_latex)

        path_to_fig = os.path.join(dir_path,'Figures','N_1Q')
        file_name = 'convergence_plot_EN_1Q'+'_n'+str(n)+'_m'+str(m)+'_nS'+str(nS)+'.pgf'
        plt.savefig(os.path.join(path_to_fig, file_name))
        
    return x_sol
    
def solve_EN_1Q(dir_path,save_fig):
    """Solve this example.
    
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
    # create electrical network from data
    elec_net = create_network(dir_path)
    
    # initialize
    x0 = initialize_network(elec_net)
    
    # solve the system
    tol = 1e-6
    max_iter = 10
    x_sol,iters,err_vec,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR')
    
    # plot convergence
    nS = len([link for link in elec_net.nodes[0].get_out_links() if isinstance(link,ElectricalLink)])
    n = int((len([node for node in elec_net.get_nodes(node_types=[2]) if (len(list(node.get_in_links()))==1 and len(list(node.get_out_links()))==1)]))/nS)
    m = int(2*n+1 - (len(elec_net.nodes)-1)/nS) #int((len([node for node in elec_net.get_nodes(node_types=[2]) if len(list(node.get_out_links())) > 2]))/nS)
    fig_conv_elec_one_solver = plt.figure('Convergence plot electrical (n = {}, m = {}, nS = {}, max iters = {}, len(x) = {})'.format(n,m,nS,max_iter,len(x0)))
    ax_conv_elec_one_solver = fig_conv_elec_one_solver.gca()
    plt.cla()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_elec_one_solver.semilogy([0,iters+1],[tol,tol],'r:')
    ax_conv_elec_one_solver.semilogy(err_vec,marker='*',ls='-',color='tab:green')
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
            "legend.loc": "upper right",
            "font.size": 10,
            "legend.fontsize": 9,               # Make the legend/label fonts
            "xtick.labelsize": 10,               # a little smaller
            "ytick.labelsize": 10,
            "figure.figsize": fig_size,
            }
        mpl.rcParams.update(pgf_with_latex)

        path_to_fig = os.path.join(dir_path,'Figures','N_1Q')
        file_name = 'convergence_plot_EN_1Q'+'_n'+str(n)+'_m'+str(m)+'_nS'+str(nS)+'_one_solver'+'.pgf'
        plt.savefig(os.path.join(path_to_fig, file_name))
        
    return x_sol

if __name__ == '__main__':
    # parse the command line
    args = command_line_input.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    compare_conv_solvers(dir_path,args.max_nodes,args.save_fig)
    
    if args.show_plots:
        plt.show()
