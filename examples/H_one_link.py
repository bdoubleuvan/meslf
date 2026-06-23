"""Example of a heat network consisting of one link"""
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
import numpy as np
import scipy.sparse as sps
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
import argparse # To get arguments (as list) from command line

from meslf.load_flow.system_of_equations import NonLinearSystemHeat as nlsysh

# some constants
from meslf.utils.constants import mbar, bar, cm, km, hour, MW, kW

command_line_input = argparse.ArgumentParser()
command_line_input.add_argument(
    "-ex", # name on the command line
    nargs = "*", # 0 or more values expected => creates a list
    type=str,
    default = ['slpp1', 'slpp2'], # default if nothing is provided
    )
command_line_input.add_argument(
    "-solv_un", # name on the command line
    nargs = "*", # 0 or more values expected => creates a list
    type=str,
    default = ['NR','NR_FD'], # default if nothing is provided
    )
command_line_input.add_argument(
    "-solv_pu", # name on the command line
    nargs = "*", # 0 or more values expected => creates a list
    type=str,
    default = ['NR','NR_FD'], # default if nothing is provided
    )
command_line_input.add_argument(
    "-solv_sc", # name on the command line
    nargs = "*", # 0 or more values expected => creates a list
    type=str,
    default = ['NR','NR_FD'], # default if nothing is provided
    )
command_line_input.add_argument(
    "--tol", # name on the command line Drop the `--` for positional/required parameters
    type=float,
    default = 1e-12, # default if nothing is provided
    )
command_line_input.add_argument(
    "--plot_sol", # name on the command line. Drop the `--` for positional/required parameters
    type=bool,
    default = False,
    )
command_line_input.add_argument(
    "--p_perc", # name on the command line Drop the `--` for positional/required parameters
    type=float,
    default = .95, # default if nothing is provided
    )
command_line_input.add_argument(
    "--m_perc", # name on the command line Drop the `--` for positional/required parameters
    type=float,
    default = 1.1, # default if nothing is provided
    )
command_line_input.add_argument(
    "--T_perc", # name on the command line Drop the `--` for positional/required parameters
    type=float,
    default = .9, # default if nothing is provided
    )
command_line_input.add_argument(
    "--m_perc_list", # name on the command line Drop the `--` for positional/required parameters
    nargs = "*", # 0 or more values expected => creates a list
    type=float,
    default = [], # default if nothing is provided
    )
command_line_input.add_argument(
    "--T_perc_list", # name on the command line Drop the `--` for positional/required parameters
    nargs = "*", # 0 or more values expected => creates a list
    type=float,
    default = [], # default if nothing is provided
    )
command_line_input.add_argument(
    "--save_fig", 
    type=bool,
    default = False,
    )
command_line_input.add_argument(
    "--show_plots", 
    type=bool,
    default = False, 
    )

def create_test_heat_network_one_link(carrier,p_ref,Ts_ref,phi_out,To,Ta,link_type,link_params):
    """Create a heat network consisting of one single link

    Parameters
    ----------
    carrier : Carrier
        The carrier in the network
    p_ref : float
        Pressure at reference (inflow) node [Pa]
    Ts_ref : float
        Supply temperature at reference (inflow) node [C]
    phi_out : float
        Heat power demand at outlfow node [W]
    To : float
        Outflow temperature of heat load [C]
    Ta : float
        Ambient temperature of the network [C]
    link_type : string
        Type of the link.
    link_params : dict
        Dictionary with the link parameters needed for the link type.

    Returns
    -------
    heat_net : HeatNetwork
        The test network
    """
    heat_net = HeatNetwork('single pipe',Ta=Ta)
    hn0 = HeatNode('hn0',node_type=0,Ts=Ts_ref,p=p_ref) # source slack node
    hn0.half_links[0].set_type('heat_exchanger_source',{'carrier':carrier})
    hn1 = HeatNode('hn2',node_type=1,To=To,phi=phi_out) # load node
    hn1.half_links[0].set_type('heat_exchanger_sink',{'carrier':carrier})

    hl0 = HeatLink('hl0',hn0,hn1,link_type=link_type,link_params=link_params)

    heat_net.add_link(hl0)
    return heat_net

def create_test_heat_network_one_link_fa(carrier,p_ref,Ts_ref,phi_out,To,Ta,link_type,link_params):
    """Create a heat network consisting of one single link, with 'fa' as link equation. That is, it uses f(q,p1,p2) = q - q(delta_p) as link equation.

    Parameters
    ----------
    carrier : Carrier
        The carrier in the network
    p_ref : float
        Pressure at reference (inflow) node [Pa]
    Ts_ref : float
        Supply temperature at reference (inflow) node [C]
    phi_out : float
        Heat power demand at outlfow node [W]
    To : float
        Outflow temperature of heat load [C]
    Ta : float
        Ambient temperature of the network [C]
    link_type : string
        Type of the link.
    link_params : dict
        Dictionary with the link parameters needed for the link type.

    Returns
    -------
    heat_net : HeatNetwork
        The test network
    """
    heat_net_fa = HeatNetwork('single pipe',Ta=Ta)
    hn0_fa = HeatNode('hn0',node_type=0,Ts=Ts_ref,p=p_ref) # source slack node
    hn0_fa.half_links[0].set_type('heat_exchanger_source',{'carrier':carrier})
    hn1_fa = HeatNode('hn2',node_type=1,Ts=Ts_ref,To=To,phi=phi_out) # load node
    hn1_fa.half_links[0].set_type('heat_exchanger_sink',{'carrier':carrier})

    hl0_fa = HeatLink('hl0',hn0_fa,hn1_fa,link_type=link_type,link_params=link_params,hydr_eq_form='q_of_dp')

    heat_net_fa.add_link(hl0_fa)
    return heat_net_fa

def initialize_network(heat_net,p_perc,m_perc,T_perc,formulation='standard',scale_var=None,scale_var_params=None):
    """
    Parameters
    ----------
    heat_net : HeatNetwork
        The network to be initialized
    p_perc : float
        Percentage of the reference pressure that is used to set the initial guess of the unknown pressure.
    m_perc : float
        Percentage of the outgoing flow that is used to set the initial guess of the link flow.
    T_perc : float
        Percentage of the known temperatures that is used to set the intitial guess of the temperatures

    Returns
    -------
    x0 : np array
        initial guess
    """
    Ta = heat_net.Ta
    p_init = np.array([p_perc*heat_net.nodes[0].p])
    Ts_init = np.array([T_perc*heat_net.nodes[0].Ts])
    heat_net.nodes[1].Ts = Ts_init[0] # set this such that the half link flow is based on the initial Ts, i.e., the heat power equation will be satisfied for the initial condition
    m1 = heat_net.nodes[1].half_links[0].flow()
    m_init = np.array([m_perc*m1])
    if formulation == 'half_link_flow':
        m_hl_init = np.array([m1])
    To = heat_net.nodes[1].half_links[0].To
    Tr_init = np.array([T_perc*To, To])
    if formulation == 'half_link_flow':
        x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    else:
        x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    heat_net.update(x_init,formulation=formulation) # update without scaling, since x_init is unscaled
    x0 = heat_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def create_network_sp():
    """Create the network with a standard pipe using Pole friction factor. Based on the email conversation with Johan on 22-03-2019.
    This also determines the (analytical) solution of the load flow problem.
    """
    print('creating network with standard pipe and Pole friction factor')
    Ta = 20. #[C]
    # carrier
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1e3 #[kg/m^3]
    water = Water('water',Cp,rho=rho)
    # pipe parameters
    L = 1*km #[m]
    D = 0.2 #[m]
    U = 10 #[W/(m^2 K)]
    link_type = 'standard_pipe_low_pres_pole'
    link_params = {'L':L,'D':D,'U':U,'carrier':water}
    # boundary conditions
    p_ref = 5*bar #[Pa]
    Ts_ref = 70.
    phi_out = 2.3*MW #[W]
    To = 50. #[C]
    # create the networks
    heat_net = create_test_heat_network_one_link(water,p_ref,Ts_ref,phi_out,To,Ta,link_type,link_params)
    for node in heat_net.get_nodes():
        print('node {} with p = {}, Ts = {}, Tr = {}'.format(node.name, node.p, node.Ts, node.Tr))
        for hl in node.get_half_links():
            print('half link {} with m = {}, phi = {}, To = {}, m(phi,To) = {}'.format(hl.name,hl.m,hl.phi,hl.To,hl.flow()))
    heat_net_fa = create_test_heat_network_one_link_fa(water,p_ref,Ts_ref,phi_out,To,Ta,link_type,link_params)
    # determine analytical solution
    C = np.pi/8 * np.sqrt(2*rho*D**5/L)
    f = 0.0065
    m_sol = 31.4149
    p_sol = 4.35*bar #[Pa]
    psi = np.exp(-np.pi*D*U*L/(Cp*m_sol))
    Ts_sol = 67.6639  #[C]
    Tr_sol = 48.5984  #[C]
    return heat_net, heat_net_fa, m_sol, p_sol, Ts_sol, Tr_sol

def create_network_sp_v2():
    """Create the network with a standard pipe using Pole friction factor.
    This also determines the (analytical) solution of the load flow problem.
    """
    print('creating network with standard pipe and Pole friction factor, option 2')
    Ta = 0. #[C]
    # carrier
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1e3 #[kg/m^3]
    water = Water('water',Cp,rho=rho)
    # pipe parameters
    L = 100 #[m]
    D = 5*cm #[m]
    lam = 0.2/10 #[W/(m K)]
    U = lam/(np.pi*D) #[W/(m^2 K)]
    link_type = 'standard_pipe_low_pres_pole'
    link_params = {'L':L,'D':D,'U':U,'carrier':water}
    # boundary conditions
    h_ref = 0.001 #[m]
    p_ref = h_ref*water.rhon*water.g #[Pa]
    Ts_ref = 100. #[C]
    phi_out = 1*kW #[W]

    m_sol = 0.01 #[kg/s]
    psi = np.exp(-lam*L/(Cp*m_sol))
    Ts_sol = psi*(Ts_ref-Ta)+Ta #[C]
    
    To = Ts_sol - phi_out/(m_sol*Cp) #[C]
    # create the networks
    heat_net = create_test_heat_network_one_link(water,p_ref,Ts_ref,phi_out,To,Ta,link_type,link_params)
    heat_net_fa = create_test_heat_network_one_link_fa(water,p_ref,Ts_ref,phi_out,To,Ta,link_type,link_params)
    # determine analytical solution
    C = np.pi/8 * np.sqrt(2*rho*D**5/L)
    f = 0.0065
    p_sol = p_ref - f/C**2*m_sol**2 #[Pa]
    Tr_sol = psi*(To-Ta)+Ta #[C] return temperature in node 0, since Tr=To in node 1
    return heat_net, heat_net_fa, m_sol, p_sol, Ts_sol, Tr_sol

def solve_network(heat_net,x0,tol,max_iter,solvers,formulation='standard',scaling=None,scale_params=None,h=1e-6,label=None,det_tol=1e-12,return_all_x=False):
    """Solve the network in different ways.

    Parameters
    ----------
    heat_net : HeatNetwork
        The network to be solved.
    x0 : np array
        Vector with initial guess, assuming 'full' formulation. Unscaled!
    solvers : list
        List of solvers to use.
    h : float, optional
        Step size used for FD Jaciobian. Defaul is 1e-6. This h is only used when 'NR_FD' in the list of solvers.
    label : string, optional
        String to add to the end of the keys used in the dictionary.

    Returns
    -------
    x : dict
        Dictionary of solution vectors, one for each solver and each formulation
    errors : list
        Dictionary of error (vectors), one for each solver and each formulation
    iters : list
        Dictionary of error (vectors), one for each solver and each formulation
    """
    x = dict()
    errors = dict()
    iters = dict()
    if return_all_x:
        x_mat = dict()
    scale_var = None
    scale_var_params = None
    D_F = np.array([])
    D_x = np.array([])
    if scaling == 'per_unit':
        scale_var = scaling
        scale_var_params = scale_params
    elif scaling == 'matrix':
        scale_var = scaling
        if scale_params:
            scale_var_params = {'mbase':scale_params.get('qbase'),'phbase':scale_params.get('pbase'),'Tbase':scale_params.get('Tbase'),'phibase':scale_params.get('phibase')}
        else:
            scale_var_params = None
    for solver in solvers:
        heat_net.reset_network(x0,formulation=formulation)
        if label:
            key = solver+label
        else:
            key = solver
        if return_all_x and solver == 'NR':
            x[key],x_mat[key],iters[key],errors[key],_,_,_,_,_,_,_ = heat_net.solve_network(tol,max_iter,formulation=formulation,solver=solver,h=h,scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol,return_all_x=return_all_x)
        else:
            x[key],iters[key],errors[key],_,_,_,_,_,_,_ = heat_net.solve_network(tol,max_iter,formulation=formulation,solver=solver,h=h,scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol)
    if return_all_x:
        return x, x_mat, iters, errors
    else:
        return x, iters, errors

def plot_convergence(ax,errors,iters,tol):
    """Plots the convergence on a semilogy scale for all the error (vectors) in the errors dictionary

    Parameters
    ----------
    ax : matplotlib Axex
        Axes to plot on.
    errors : dict
        Dictionary with errors of the solver
    iters : dict
        Dictionary with corresponding number of iterations needed.
    """
    max_iters_used = 0
    for key, err in errors.items():
        if 'fb' in key:
            ls = 'o'
        else:
            ls = '*'
        if 'root' in key or 'fsolver' in key:
            ls += '-.'
        elif 'FD' in key:
            ls += '--'
        else:
            ls += '-'
        iters_used = iters.get(key)
        max_iters_used = max(max_iters_used,iters_used)
        if len(err) != iters_used+1: # for fsolver and root
            ax.semilogy(np.asarray([0,iters_used]),err,ls,label=key)
        else:
            ax.semilogy(err,ls,label=key)
    ax.semilogy([0,max_iters_used+1],[tol,tol],'r:',label='tolerance')
    ax.legend()
    ax.set_xlabel('Iteration k')
    ax.set_ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    return max_iters_used

def plot_pres_sol_error(ax,p_sol,x_sol):
    """Plots the error between the found solution and the actual solution for pressure"""
    xticks = []
    ind = 0
    for key, x in x_sol.items():
        x_p = x[1]
        ax.plot(ind,(p_sol-x_p),'k*')
        xticks.append(key)
        ind += 1
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    #ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
    ax.set_ylabel('p - $x^p$ [Pa]')
    ax.set_xticks(range(ind))
    ax.set_xticklabels(xticks,rotation='vertical')
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)

def plot_flow_sol_error(ax,m_sol,x_sol):
    """Plots the error between the found solution and the actual solution for flow"""
    xticks = []
    ind = 0
    for key, x in x_sol.items():
        x_m = x[0]
        ax.plot(ind,(m_sol-x_m),'k*')
        xticks.append(key)
        ind += 1
    ax.set_ylabel('m - $x^m$ [kg/s]')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    #ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
    ax.set_xticks(range(ind))
    ax.set_xticklabels(xticks,rotation='vertical')
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)

def plot_Ts_sol_error(ax,Ts_sol,x_sol):
    """Plots the error between the found solution and the actual solution for flow"""
    xticks = []
    ind = 0
    for key, x in x_sol.items():
        x_Ts = x[2]
        ax.plot(ind,(Ts_sol-x_Ts),'k*')
        xticks.append(key)
        ind += 1
    ax.set_ylabel('$T^s$ - $x^{Ts}$ [C]')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    #ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
    ax.set_xticks(range(ind))
    ax.set_xticklabels(xticks,rotation='vertical')
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)

def plot_Tr_sol_error(ax,Tr_sol,x_sol):
    """Plots the error between the found solution and the actual solution for flow"""
    xticks = []
    ind = 0
    for key, x in x_sol.items():
        x_Tr = x[-2]
        ax.plot(ind,(Tr_sol-x_Tr),'k*')
        xticks.append(key)
        ind += 1
    ax.set_ylabel('$T^r_0$ - $x^{Tr}$ [C]')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    #ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
    ax.set_xticks(range(ind))
    ax.set_xticklabels(xticks,rotation='vertical')
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)

def standard_low_pres_pole_1(p_perc,m_perc,T_perc,solvers_unscaled,solvers_pu,solvers_scaled,tol=1e-12,plot_sol=True):
    """Solve the network with a low pressure pipe using the Pole friction factor, option 1 for values.

    Parameters
    ----------
    p_perc : float
        Percentage of the reference pressure that is used to set the initial guess of the unknown pressure.
    m_perc : float
        Percentage of the outgoing flow that is used to set the initial guess of the link flow.
    T_perc : float
        Percentage of the known temperatures that is used to set the intitial guess of the temperatures
    solvers_unscaled : list
        List of strings indicating which solvers need to be used to solve the unscaled problem.
    solver_pu : list
        List of string indicating which solvers need to be used to solve the scaled problem (per unit scaling).
    solvers_unscaled : list
        List of strings indicating which solvers need to be used to solve the scaled problem (using matrix scaling).
    tol: float, optional
        Tolerance used in solvers. Default is tol = 1e-12
    plot_sol : bool, optional
        If True, plots with the difference between the calculated solution and the actual solution are shown. Default is True
    """
    print('\nLow pressure pipe with Pole friction factor, option 1')
    # create network and solution
    heat_net_sp, heat_net_sp_fa, m_sol_sp, p_sol_sp, Ts_sol_sp, Tr_sol_sp = create_network_sp()
    print('Actual solution: m = {:.3e} [kg/s], p1 = {:.3e} [Pa], Ts1 = {:.3e} [C], Tr0 = {:.3e} [C]'.format(m_sol_sp, p_sol_sp, Ts_sol_sp, Tr_sol_sp))
    # initialize networks
    x0_sp = initialize_network(heat_net_sp,p_perc,m_perc,T_perc)
    print('x0 with fb: m01 = {:.3e} [kg/s], p1 = {:.3e} [Pa], Ts1 = {:.3e} [C], Tr0 = {:.3e} [C], Tr1 = {:.3e} [C]'.format(x0_sp[0], x0_sp[1], x0_sp[2], x0_sp[3], x0_sp[4]))
    x0_sp_fa = initialize_network(heat_net_sp_fa,p_perc,m_perc,T_perc)
    print('x0 with fa: m01 = {:.3e} [kg/s], p1 = {:.3e} [Pa], Ts1 = {:.3e} [C], Tr0 = {:.3e} [C], Tr1 = {:.3e} [C]'.format(x0_sp_fa[0], x0_sp_fa[1], x0_sp_fa[2], x0_sp_fa[3], x0_sp_fa[4]))
    Ta_sp = heat_net_sp.Ta
    scale_params_sp = {'qbase':10,'pbase':bar,'Tbase':70,'phibase':MW}
    
    # Solve networks in different ways, compare convergence and solutions
    max_iter_sp = 20
    # unscaled
    x_sp, iters_sp, errors_sp = solve_network(heat_net_sp,x0_sp,tol,max_iter_sp,solvers_unscaled,label='_fb')
    x_sp_fa, iters_sp_fa, errors_sp_fa = solve_network(heat_net_sp_fa,x0_sp_fa,tol,max_iter_sp,solvers_unscaled,label='_fa')
    # per unit
    x_sp_pu, iters_sp_pu, errors_sp_pu = solve_network(heat_net_sp,x0_sp,tol,max_iter_sp,solvers_pu,label='_fb',scaling='per_unit',scale_params=scale_params_sp)
    x_sp_fa_pu, iters_sp_fa_pu, errors_sp_fa_pu = solve_network(heat_net_sp_fa,x0_sp_fa,tol,max_iter_sp,solvers_pu,label='_fa',scaling='per_unit',scale_params=scale_params_sp)
    # scaling in solver (in this case using the same base values as for the per unit scaling.)
    x_sp_ss, iters_sp_ss, errors_sp_ss = solve_network(heat_net_sp,x0_sp,tol,max_iter_sp,solvers_scaled,label='_fb',scaling='matrix',scale_params=scale_params_sp)
    x_sp_fa_ss, iters_sp_fa_ss, errors_sp_fa_ss = solve_network(heat_net_sp_fa,x0_sp_fa,tol,max_iter_sp,solvers_scaled,label='_fa',scaling='matrix',scale_params=scale_params_sp)
    # plot convergence
    fig_conv_sp, (ax_conv_sp, ax_conv_sp_pu, ax_conv_sp_ss) = plt.subplots(3, 1, sharex=True)
    fig_conv_sp.canvas.set_window_title('Convergence standard pipe Pole (high value for phi, large pipe)')
    plot_convergence(ax_conv_sp,{**errors_sp,**errors_sp_fa},{**iters_sp,**iters_sp_fa},tol)
    ax_conv_sp.set_title('unscaled')
    ax_conv_sp.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_sp.set_xlabel('')
    plot_convergence(ax_conv_sp_pu,{**errors_sp_pu,**errors_sp_fa_pu},{**iters_sp_pu,**iters_sp_fa_pu},tol)
    ax_conv_sp_pu.set_title('per unit scaling')
    ax_conv_sp_pu.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_sp_pu.set_xlabel('')
    plot_convergence(ax_conv_sp_ss,{**errors_sp_ss,**errors_sp_fa_ss},{**iters_sp_ss,**iters_sp_fa_ss},tol)
    ax_conv_sp_ss.set_title('scaled')
    ax_conv_sp_ss.set_ylabel('Error ($||D_F F(x^k)||_2$)')
    if plot_sol:
        # plot error w.r.t solution for pressure
        fig_pres_err_sp, (ax_pres_err_sp, ax_pres_err_sp_pu, ax_pres_err_sp_ss) = plt.subplots(1, 3, sharey=True)
        fig_pres_err_sp.canvas.set_window_title('Pressure error standard pipe Pole, with p = {:.2e} [Pa] = {:.2e} [bar]'.format(p_sol_sp,p_sol_sp/bar))
        x_sp_conv = dict() # not all the unscaled methods have converged. Only plot errors for the ones that did.
        for key, x in {**x_sp,**x_sp_fa}.items():
            if {**errors_sp,**errors_sp_fa}.get(key)[-1] < tol:
                x_sp_conv[key] = x
        plot_pres_sol_error(ax_pres_err_sp,p_sol_sp,x_sp_conv)
        ax_pres_err_sp.set_title('unscaled')
        x_sp_pu_conv = dict()
        for key, x in {**x_sp_pu,**x_sp_fa_pu}.items():
            if {**errors_sp_pu,**errors_sp_fa_pu}.get(key)[-1] < tol:
                x_sp_pu_conv[key] = x
        x_sp_pu_SI = dict()
        for key, x in x_sp_pu_conv.items():
            x_sp_pu_SI[key] = x*np.array([scale_params_sp['qbase'],scale_params_sp['pbase'],scale_params_sp['Tbase'],scale_params_sp['Tbase'],scale_params_sp['Tbase']])
        plot_pres_sol_error(ax_pres_err_sp_pu,p_sol_sp,x_sp_pu_SI)
        ax_pres_err_sp_pu.set_title('per unit scaling')
        x_sp_ss_conv = dict()
        for key, x in {**x_sp_ss,**x_sp_fa_ss}.items():
            if {**errors_sp_ss,**errors_sp_fa_ss}.get(key)[-1] < tol:
                x_sp_ss_conv[key] = x
        plot_pres_sol_error(ax_pres_err_sp_ss,p_sol_sp,x_sp_ss_conv)
        ax_pres_err_sp_ss.set_title('scaled')
        # plot error w.r.t solution for flow
        fig_flow_err_sp, (ax_flow_err_sp, ax_flow_err_sp_pu, ax_flow_err_sp_ss) = plt.subplots(1, 3, sharey=True)
        fig_flow_err_sp.canvas.set_window_title('Flow error standard pipe Pole, with q = {:.2e} [kg/s]'.format(m_sol_sp))
        plot_flow_sol_error(ax_flow_err_sp,m_sol_sp,x_sp_conv)
        ax_flow_err_sp.set_title('unscaled')
        plot_flow_sol_error(ax_flow_err_sp_pu,m_sol_sp,x_sp_pu_SI)
        ax_flow_err_sp_pu.set_title('per unit scaling')
        plot_flow_sol_error(ax_flow_err_sp_ss,m_sol_sp,x_sp_ss_conv)
        ax_flow_err_sp_ss.set_title('scaled')
        # plot error w.r.t solution for supply temperature
        fig_Ts_err_sp, (ax_Ts_err_sp, ax_Ts_err_sp_pu, ax_Ts_err_sp_ss) = plt.subplots(1, 3, sharey=True)
        fig_Ts_err_sp.canvas.set_window_title('Supply temperature error standard pipe Pole, with $T^s_1$ - Ta = {:.2e} [C]'.format(Ts_sol_sp-Ta_sp))
        plot_Ts_sol_error(ax_Ts_err_sp,Ts_sol_sp,x_sp_conv)
        ax_Ts_err_sp.set_title('unscaled')
        plot_Ts_sol_error(ax_Ts_err_sp_pu,Ts_sol_sp,x_sp_pu_SI)
        ax_Ts_err_sp_pu.set_title('per unit scaling')
        plot_Ts_sol_error(ax_Ts_err_sp_ss,Ts_sol_sp,x_sp_ss_conv)
        ax_Ts_err_sp_ss.set_title('scaled')
        # plot error w.r.t solution for supply temperature
        fig_Tr_err_sp, (ax_Tr_err_sp, ax_Tr_err_sp_pu, ax_Tr_err_sp_ss) = plt.subplots(1, 3, sharey=True)
        fig_Tr_err_sp.canvas.set_window_title('Return temperature error standard pipe Pole, with $T^r_0$ - Ta = {:.2e} [C]'.format(Tr_sol_sp-Ta_sp))
        plot_Tr_sol_error(ax_Tr_err_sp,Tr_sol_sp,x_sp_conv)
        ax_Tr_err_sp.set_title('unscaled')
        plot_Tr_sol_error(ax_Tr_err_sp_pu,Tr_sol_sp,x_sp_pu_SI)
        ax_Tr_err_sp_pu.set_title('per unit scaling')
        plot_Tr_sol_error(ax_Tr_err_sp_ss,Tr_sol_sp,x_sp_ss_conv)
        ax_Tr_err_sp_ss.set_title('scaled')

def standard_low_pres_pole_2(p_perc,m_perc,T_perc,solvers_unscaled,solvers_pu,solvers_scaled,tol=1e-12,plot_sol=True):
    """Solve the network with a low pressure pipe using the Pole friction factor, option 2 for values.

    Parameters
    ----------
    p_perc : float
        Percentage of the reference pressure that is used to set the initial guess of the unknown pressure.
    m_perc : float
        Percentage of the outgoing flow that is used to set the initial guess of the link flow.
    T_perc : float
        Percentage of the known temperatures that is used to set the intitial guess of the temperatures
    solvers_unscaled : list
        List of strings indicating which solvers need to be used to solve the unscaled problem.
    solver_pu : list
        List of string indicating which solvers need to be used to solve the scaled problem (per unit scaling).
    solvers_unscaled : list
        List of strings indicating which solvers need to be used to solve the scaled problem (using matrix scaling).
    tol: float, optional
        Tolerance used in solvers. Default is tol = 1e-12
    plot_sol : bool, optional
        If True, plots with the difference between the calculated solution and the actual solution are shown. Default is True
    """
    print('\nLow pressure pipe with Pole friction factor, option 2')
    # create network and solution
    heat_net_sp2, heat_net_sp2_fa, m_sol_sp2, p_sol_sp2, Ts_sol_sp2, Tr_sol_sp2 = create_network_sp_v2()
    print('Actual solution: m = {:.3e} [kg/s], p1 = {:.3e} [Pa], Ts1 = {:.3e} [C], Tr0 = {:.3e} [C]'.format(m_sol_sp2, p_sol_sp2, Ts_sol_sp2, Tr_sol_sp2))
    # initialize networks
    x0_sp2 = initialize_network(heat_net_sp2,p_perc,m_perc,T_perc)
    print('x0 with fb: m01 = {:.3e} [kg/s], p1 = {:.3e} [Pa], Ts1 = {:.3e} [C], Tr0 = {:.3e} [C], Tr1 = {:.3e} [C]'.format(x0_sp2[0], x0_sp2[1], x0_sp2[2], x0_sp2[3], x0_sp2[4]))
    x0_sp2_fa = initialize_network(heat_net_sp2_fa,p_perc,m_perc,T_perc)
    print('x0 with fa: m01 = {:.3e} [kg/s], p1 = {:.3e} [Pa], Ts1 = {:.3e} [C], Tr0 = {:.3e} [C], Tr1 = {:.3e} [C]'.format(x0_sp2_fa[0], x0_sp2_fa[1], x0_sp2_fa[2], x0_sp2_fa[3], x0_sp2_fa[4]))
    Ta_sp2 = heat_net_sp2.Ta
    scale_params_sp2 = {'qbase':1e-2,'pbase':bar,'Tbase':10,'phibase':kW}
    
    # Solve networks in different ways, compare convergence and solutions
    max_iter_sp2 = 20
    # unscaled
    x_sp2, iters_sp2, errors_sp2 = solve_network(heat_net_sp2,x0_sp2,tol,max_iter_sp2,solvers_unscaled,label='_fb')
    x_sp2_fa, iters_sp2_fa, errors_sp2_fa = solve_network(heat_net_sp2_fa,x0_sp2_fa,tol,max_iter_sp2,solvers_unscaled,label='_fa')
    # per unit
    x_sp2_pu, iters_sp2_pu, errors_sp2_pu = solve_network(heat_net_sp2,x0_sp2,tol,max_iter_sp2,solvers_pu,label='_fb',scaling='per_unit',scale_params=scale_params_sp2)
    x_sp2_fa_pu, iters_sp2_fa_pu, errors_sp2_fa_pu = solve_network(heat_net_sp2_fa,x0_sp2_fa,tol,max_iter_sp2,solvers_pu,label='_fa',scaling='per_unit',scale_params=scale_params_sp2)
    # scaling in solver (in this case using the same base values as for the per unit scaling.)
    x_sp2_ss, iters_sp2_ss, errors_sp2_ss = solve_network(heat_net_sp2,x0_sp2,tol,max_iter_sp2,solvers_scaled,label='_fb',scaling='matrix',scale_params=scale_params_sp2)
    x_sp2_fa_ss, iters_sp2_fa_ss, errors_sp2_fa_ss = solve_network(heat_net_sp2_fa,x0_sp2_fa,tol,max_iter_sp2,solvers_scaled,label='_fa',scaling='matrix',scale_params=scale_params_sp2)
    # plot convergence
    fig_conv_sp2, (ax_conv_sp2, ax_conv_sp2_pu, ax_conv_sp2_ss) = plt.subplots(3, 1, sharex=True)
    fig_conv_sp2.canvas.set_window_title('Convergence standard pipe Pole, version 2 (smaller value for phi, small pipe)')
    plot_convergence(ax_conv_sp2,{**errors_sp2,**errors_sp2_fa},{**iters_sp2,**iters_sp2_fa},tol)
    ax_conv_sp2.set_title('unscaled')
    ax_conv_sp2.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_sp2.set_xlabel('')
    plot_convergence(ax_conv_sp2_pu,{**errors_sp2_pu,**errors_sp2_fa_pu},{**iters_sp2_pu,**iters_sp2_fa_pu},tol)
    ax_conv_sp2_pu.set_title('per unit scaling')
    ax_conv_sp2_pu.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_sp2_pu.set_xlabel('')
    plot_convergence(ax_conv_sp2_ss,{**errors_sp2_ss,**errors_sp2_fa_ss},{**iters_sp2_ss,**iters_sp2_fa_ss},tol)
    ax_conv_sp2_ss.set_title('scaled')
    ax_conv_sp2_ss.set_ylabel('Error ($||D_F F(x^k)||_2$)')
    if plot_sol:
        # plot error w.r.t solution for pressure
        fig_pres_err_sp2, (ax_pres_err_sp2, ax_pres_err_sp2_pu, ax_pres_err_sp2_ss) = plt.subplots(1, 3, sharey=True)
        fig_pres_err_sp2.canvas.set_window_title('Pressure error standard pipe Pole, with p = {:.2e} [Pa] = {:.2e} [bar]'.format(p_sol_sp2,p_sol_sp2/bar))
        x_sp2_conv = dict() # not all the unscaled methods have converged. Only plot errors for the ones that did.
        for key, x in {**x_sp2,**x_sp2_fa}.items():
            if {**errors_sp2,**errors_sp2_fa}.get(key)[-1] < tol:
                x_sp2_conv[key] = x
        plot_pres_sol_error(ax_pres_err_sp2,p_sol_sp2,x_sp2_conv)
        ax_pres_err_sp2.set_title('unscaled')
        x_sp2_pu_conv = dict()
        for key, x in {**x_sp2_pu,**x_sp2_fa_pu}.items():
            if {**errors_sp2_pu,**errors_sp2_fa_pu}.get(key)[-1] < tol:
                x_sp2_pu_conv[key] = x
        x_sp2_pu_SI = dict()
        for key, x in x_sp2_pu_conv.items():
            x_sp2_pu_SI[key] = x*np.array([scale_params_sp2['qbase'],scale_params_sp2['pbase'],scale_params_sp2['Tbase'],scale_params_sp2['Tbase'],scale_params_sp2['Tbase']])
        plot_pres_sol_error(ax_pres_err_sp2_pu,p_sol_sp2,x_sp2_pu_SI)
        ax_pres_err_sp2_pu.set_title('per unit scaling')
        x_sp2_ss_conv = dict()
        for key, x in {**x_sp2_ss,**x_sp2_fa_ss}.items():
            if {**errors_sp2_ss,**errors_sp2_fa_ss}.get(key)[-1] < tol:
                x_sp2_ss_conv[key] = x
        plot_pres_sol_error(ax_pres_err_sp2_ss,p_sol_sp2,x_sp2_ss_conv)
        ax_pres_err_sp2_ss.set_title('scaled')
        # plot error w.r.t solution for flow
        fig_flow_err_sp2, (ax_flow_err_sp2, ax_flow_err_sp2_pu, ax_flow_err_sp2_ss) = plt.subplots(1, 3, sharey=True)
        fig_flow_err_sp2.canvas.set_window_title('Flow error standard pipe Pole, with q = {:.2e} [kg/s]'.format(m_sol_sp2))
        plot_flow_sol_error(ax_flow_err_sp2,m_sol_sp2,x_sp2_conv)
        ax_flow_err_sp2.set_title('unscaled')
        plot_flow_sol_error(ax_flow_err_sp2_pu,m_sol_sp2,x_sp2_pu_SI)
        ax_flow_err_sp2_pu.set_title('per unit scaling')
        plot_flow_sol_error(ax_flow_err_sp2_ss,m_sol_sp2,x_sp2_ss_conv)
        ax_flow_err_sp2_ss.set_title('scaled')
        # plot error w.r.t solution for supply temperature
        fig_Ts_err_sp2, (ax_Ts_err_sp2, ax_Ts_err_sp2_pu, ax_Ts_err_sp2_ss) = plt.subplots(1, 3, sharey=True)
        fig_Ts_err_sp2.canvas.set_window_title('Supply temperature error standard pipe Pole, with $T^s_1$ - Ta = {:.2e} [C]'.format(Ts_sol_sp2-Ta_sp2))
        plot_Ts_sol_error(ax_Ts_err_sp2,Ts_sol_sp2,x_sp2_conv)
        ax_Ts_err_sp2.set_title('unscaled')
        plot_Ts_sol_error(ax_Ts_err_sp2_pu,Ts_sol_sp2,x_sp2_pu_SI)
        ax_Ts_err_sp2_pu.set_title('per unit scaling')
        plot_Ts_sol_error(ax_Ts_err_sp2_ss,Ts_sol_sp2,x_sp2_ss_conv)
        ax_Ts_err_sp2_ss.set_title('scaled')
        # plot error w.r.t solution for supply temperature
        fig_Tr_err_sp2, (ax_Tr_err_sp2, ax_Tr_err_sp2_pu, ax_Tr_err_sp2_ss) = plt.subplots(1, 3, sharey=True)
        fig_Tr_err_sp2.canvas.set_window_title('Return temperature error standard pipe Pole, with $T^r_0$ - Ta = {:.2e} [C]'.format(Tr_sol_sp2-Ta_sp2))
        plot_Tr_sol_error(ax_Tr_err_sp2,Tr_sol_sp2,x_sp2_conv)
        ax_Tr_err_sp2.set_title('unscaled')
        plot_Tr_sol_error(ax_Tr_err_sp2_pu,Tr_sol_sp2,x_sp2_pu_SI)
        ax_Tr_err_sp2_pu.set_title('per unit scaling')
        plot_Tr_sol_error(ax_Tr_err_sp2_ss,Tr_sol_sp2,x_sp2_ss_conv)
        ax_Tr_err_sp2_ss.set_title('scaled')

def compare_init_cond_low_pres_pole_2(p_perc,m_perc_list,T_perc_list,solvers_unscaled,solvers_scaled,dir_path,save_fig,tol=1e-12):
    """Compare convergence for different initial conditions, for the network with a low pressure pipe using the Pole friction factor, option 2 for values Only uses unscaled and scaled solver (no per unit scaling).
    
    Parameters
    ----------
    p_perc : float
        Percentage of the reference pressure that is used to set the initial guess of the unknown pressure.
    m_perc : float
        Percentage of the outgoing flow that is used to set the initial guess of the link flow.
    T_perc : float
        Percentage of the known temperatures that is used to set the intitial guess of the temperatures
    solvers_unscaled : list
        List of strings indicating which solvers need to be used to solve the unscaled problem.
    solvers_unscaled : list
        List of strings indicating which solvers need to be used to solve the scaled problem (using matrix scaling).
    tol: float, optional
        Tolerance used in solvers. Default is tol = 1e-12
    """
    # set figure parameters
    if save_fig:
        d = 10  # width of the window border in pixels
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
        
    # create network and solution
    heat_net_sp2, heat_net_sp2_fa, m_sol_sp2, p_sol_sp2, Ts_sol_sp2, Tr_sol_sp2 = create_network_sp_v2()
    print('Actual solution: m = {:.3e} [kg/s], p1 = {:.3e} [Pa], Ts1 = {:.3e} [C], Tr0 = {:.3e} [C]'.format(m_sol_sp2, p_sol_sp2, Ts_sol_sp2, Tr_sol_sp2))
    scale_params_sp2 = {'qbase':1e-2,'pbase':bar,'Tbase':10,'phibase':kW}
    max_iter_sp2 = 20
    det_tol = 1e-30
    
    Ts0 = heat_net_sp2.nodes[0].Ts
    To = heat_net_sp2.nodes[1].half_links[0].To
    
    x_sol = {}
    x_mat = {}
    iters = {}
    errors = {}
    x_sol_ss = {}
    x_mat_ss = {}
    iters_ss = {}
    errors_ss = {}
    if save_fig:
        file_name_m = '_m'
        file_name_T = '_T'
    for T_perc in T_perc_list:
        if save_fig:
            file_name_T += r'_{:.2f}'.format(T_perc*Ts0)
        if m_perc_list:
            for m_perc in m_perc_list:
                # initialize network
                x0_sp2 = initialize_network(heat_net_sp2,p_perc,m_perc,T_perc)
                print('\nx0 with fb: m01 = {:.3e} [kg/s], p1 = {:.3e} [Pa], Ts1 = {:.3e} [C], Tr0 = {:.3e} [C], Tr1 = {:.3e} [C]'.format(x0_sp2[0], x0_sp2[1], x0_sp2[2], x0_sp2[3], x0_sp2[4]))
                label = r' $m_{01}^0$'+' = {:.2f}'.format(m_perc)+r'$m_1$ $(T^s_1)^0$'+' = {:.1f}'.format(x0_sp2[2])
                if save_fig:
                    file_name_m += r'_{:.2f}m1'.format(m_perc)
                # Solve networks in different ways, compare convergence and solutions
                # unscaled
                x_sp2, x_sp2_mat, iters_sp2, errors_sp2 = solve_network(heat_net_sp2,x0_sp2,tol,max_iter_sp2,solvers_unscaled,label=label,return_all_x=True)
                x_sol.update(x_sp2)
                x_mat.update(x_sp2_mat)
                iters.update(iters_sp2)
                errors.update(errors_sp2)
                # scaling in solver (in this case using the same base values as for the per unit scaling.)
                x_sp2_ss, x_sp2_mat_ss, iters_sp2_ss, errors_sp2_ss = solve_network(heat_net_sp2,x0_sp2,tol,max_iter_sp2,solvers_scaled,label=label,scaling='matrix',scale_params=scale_params_sp2,det_tol=det_tol,return_all_x=True)
                x_sol_ss.update(x_sp2_ss)
                x_mat_ss.update(x_sp2_mat_ss)
                iters_ss.update(iters_sp2_ss)
                errors_ss.update(errors_sp2_ss)
                # use formulation with unknown half link flows
                heat_net_sp2.reset_network(x0_sp2)
                x0_sp2_hl = initialize_network(heat_net_sp2,p_perc,m_perc,T_perc,formulation='half_link_flow')
                # unscaled
                x_sp2_hl, x_sp2_mat_hl, iters_sp2_hl, errors_sp2_hl = solve_network(heat_net_sp2,x0_sp2_hl,tol,max_iter_sp2,solvers_unscaled,formulation='half_link_flow',label=' hl'+label,return_all_x=True)
                x_sol.update(x_sp2_hl)
                x_mat.update(x_sp2_mat_hl)
                iters.update(iters_sp2_hl)
                errors.update(errors_sp2_hl)
                # scaling in solver (in this case using the same base values as for the per unit scaling.)
                x_sp2_ss_hl, x_sp2_mat_ss_hl, iters_sp2_ss_hl, errors_sp2_ss_hl = solve_network(heat_net_sp2,x0_sp2_hl,tol,max_iter_sp2,solvers_scaled,formulation='half_link_flow',label=' hl'+label,scaling='matrix',scale_params=scale_params_sp2,det_tol=det_tol,return_all_x=True)
                x_sol_ss.update(x_sp2_ss_hl)
                x_mat_ss.update(x_sp2_mat_ss_hl)
                iters_ss.update(iters_sp2_ss_hl)
                errors_ss.update(errors_sp2_ss_hl)
                
                # initialize network with negative initial guess for the flow
                x0_sp2 = initialize_network(heat_net_sp2,p_perc,-m_perc,T_perc)
                print('\nx0 with fb: m01 = {:.3e} [kg/s], p1 = {:.3e} [Pa], Ts1 = {:.3e} [C], Tr0 = {:.3e} [C], Tr1 = {:.3e} [C]'.format(x0_sp2[0], x0_sp2[1], x0_sp2[2], x0_sp2[3], x0_sp2[4]))
                label = r' $m_{01}^0$'+' = {:.2f}'.format(-m_perc)+r'$m_1$ $(T^s_1)^0$'+' = {:.1f}'.format(x0_sp2[2])
                # Solve networks in different ways, compare convergence and solutions
                # unscaled
                x_sp2, x_sp2_mat, iters_sp2, errors_sp2 = solve_network(heat_net_sp2,x0_sp2,tol,max_iter_sp2,solvers_unscaled,label=label,return_all_x=True)
                x_sol.update(x_sp2)
                x_mat.update(x_sp2_mat)
                iters.update(iters_sp2)
                errors.update(errors_sp2)
                # scaling in solver (in this case using the same base values as for the per unit scaling.)
                x_sp2_ss, x_sp2_mat_ss, iters_sp2_ss, errors_sp2_ss = solve_network(heat_net_sp2,x0_sp2,tol,max_iter_sp2,solvers_scaled,label=label,scaling='matrix',scale_params=scale_params_sp2,det_tol=det_tol,return_all_x=True)
                x_sol_ss.update(x_sp2_ss)
                x_mat_ss.update(x_sp2_mat_ss)
                iters_ss.update(iters_sp2_ss)
                errors_ss.update(errors_sp2_ss)
                # use formulation with unknown half link flows
                heat_net_sp2.reset_network(x0_sp2)
                x0_sp2_hl = initialize_network(heat_net_sp2,p_perc,-m_perc,T_perc,formulation='half_link_flow')
                # unscaled
                x_sp2_hl, x_sp2_mat_hl, iters_sp2_hl, errors_sp2_hl = solve_network(heat_net_sp2,x0_sp2_hl,tol,max_iter_sp2,solvers_unscaled,formulation='half_link_flow',label=' hl'+label,return_all_x=True)
                x_sol.update(x_sp2_hl)
                x_mat.update(x_sp2_mat_hl)
                iters.update(iters_sp2_hl)
                errors.update(errors_sp2_hl)
                # scaling in solver (in this case using the same base values as for the per unit scaling.)
                x_sp2_ss_hl, x_sp2_mat_ss_hl, iters_sp2_ss_hl, errors_sp2_ss_hl = solve_network(heat_net_sp2,x0_sp2_hl,tol,max_iter_sp2,solvers_scaled,formulation='half_link_flow',label=' hl'+label,scaling='matrix',scale_params=scale_params_sp2,det_tol=det_tol,return_all_x=True)
                x_sol_ss.update(x_sp2_ss_hl)
                x_mat_ss.update(x_sp2_mat_ss_hl)
                iters_ss.update(iters_sp2_ss_hl)
                errors_ss.update(errors_sp2_ss_hl)
        else:
            heat_net_sp2.initialize()
            
        # initialize network with solution for pressure, link mass flow, and both return temperatures
        Ts_init = T_perc*Ts0
        x0_sol_sp2 = np.array([m_sol_sp2,p_sol_sp2,Ts_init,Tr_sol_sp2,To])
        print('\nx0 with fb: m01 = {:.3e} [kg/s], p1 = {:.3e} [Pa], Ts1 = {:.3e} [C], Tr0 = {:.3e} [C], Tr1 = {:.3e} [C]'.format(x0_sol_sp2[0], x0_sol_sp2[1], x0_sol_sp2[2], x0_sol_sp2[3], x0_sol_sp2[4]))
        label = r' $m_{01}^0 = m_{01}$ $(T^s_1)^0$'+' = {:.1f}'.format(x0_sol_sp2[2])
        
        # plot the supply line mixing rule for node 1 (i.e. plot FTs1), and its derivative to Ts1
        heat_net_sp2.update_full(x0_sol_sp2)
        def FTs1(Ts1):
            node0 = heat_net_sp2.nodes[0]
            node1 = heat_net_sp2.nodes[1]
            link = heat_net_sp2.links[0]
            Ta = heat_net_sp2.Ta
            m1 = node1.half_links[0].phi/(link.link_params['carrier'].Cp*(Ts1 - To))
            return m1*Ts1 - link.m*(link.psi()*(node0.Ts - Ta)+Ta)
        def dFTs1_dTs1(Ts1):
            node1 = heat_net_sp2.nodes[1]
            link = heat_net_sp2.links[0]
            m1 = node1.half_links[0].phi/(link.link_params['carrier'].Cp*(Ts1 - To))
            return -m1/(Ts1-To)*To
        eps = 5e-1
        Ts1_low = np.linspace(60,To-eps,100)
        Ts1_high = np.linspace(To+eps,110,100)
        FTs1_min = min(np.concatenate((FTs1(Ts1_low),FTs1(Ts1_high))))
        FTs1_max = max(np.concatenate((FTs1(Ts1_low),FTs1(Ts1_high))))
        fig_FTs1 = plt.figure(r'$F^{T^s}_1$')
        ax_FTs1 = plt.gca()
        ax_FTs1.plot(Ts1_low,FTs1(Ts1_low),color='tab:blue')
        ax_FTs1.plot(Ts1_high,FTs1(Ts1_high),color='tab:blue')
        ax_FTs1.plot([To,To],[FTs1_min,FTs1_max],'k:')
        ax_FTs1.plot([min(Ts1_low),max(Ts1_high)],[FTs1(min(Ts1_low)),FTs1(min(Ts1_low))],'k:')
        ax_FTs1.plot([min(Ts1_low),max(Ts1_high)],[FTs1(max(Ts1_high)),FTs1(max(Ts1_high))],'k:')
        ax_FTs1.set_xlabel(r'$T^s_1$ [C]')
        ax_FTs1.set_ylabel(r'$F^{T^s}_1$ [kg/s C]')
        ax_FTs1.set_xlim(min(Ts1_low),max(Ts1_high))
        ax_FTs1.set_ylim(FTs1_min,FTs1_max)
        ax_FTs1.grid(which='major',color='k', linestyle='--', alpha=.2)
        ax_FTs1.grid(which='minor',color='k', linestyle=':', alpha=.05)
        ax_FTs1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_FTs1.set_xticks([min(Ts1_low),To,max(Ts1_high)])
        ax_FTs1.text(max(Ts1_high)+.5,FTs1(min(Ts1_low)),'{:.2f}'.format(FTs1(min(Ts1_low))), horizontalalignment='left',verticalalignment='center',color='k')
        ax_FTs1.text(max(Ts1_high)+.5,FTs1(max(Ts1_high)),'{:.2f}'.format(FTs1(max(Ts1_high))), horizontalalignment='left',verticalalignment='center',color='k')        
        if save_fig:
            # enlarge figure to maximum size to prevent the legend and labels from being to crowded (TkAgg backend)
            manager = plt.get_current_fig_manager()
            screen_y = manager.window.winfo_screenheight()
            screen_x = manager.window.winfo_screenwidth()
            manager.resize(*(screen_x/2 - 2*d,screen_y - 4*d)) # screen_x/2 to compensate for using 2 monitors

            path_to_fig = os.path.join(dir_path,'Figures','H_one_link')
            file_name = 'FTs1.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))
        
        FTs1_der_min = min(np.concatenate((dFTs1_dTs1(Ts1_low),dFTs1_dTs1(Ts1_high))))
        FTs1_der_max = max(np.concatenate((dFTs1_dTs1(Ts1_low),dFTs1_dTs1(Ts1_high))))
        fig_FTs1_der = plt.figure(r'Derivative of $F^{T^s}_1$ to $T^s_0$')
        ax_FTs1_der = plt.gca()
        ax_FTs1_der.plot(Ts1_low,dFTs1_dTs1(Ts1_low),color='tab:blue')
        ax_FTs1_der.plot(Ts1_high,dFTs1_dTs1(Ts1_high),color='tab:blue')
        ax_FTs1_der.plot([To,To],[FTs1_der_min,FTs1_der_max],'k:')
        ax_FTs1_der.set_xlabel(r'$T^s_1$ [C]')
        ax_FTs1_der.set_ylabel(r'$\frac{\partial F^{T^s}_1}{\partial T^s_1}$ [kg/s?]')
        ax_FTs1_der.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_FTs1_der.set_xticks([min(Ts1_low),To,max(Ts1_high)])
        ax_FTs1_der.grid(which='major',color='k', linestyle='--', alpha=.2)
        ax_FTs1_der.grid(which='minor',color='k', linestyle=':', alpha=.05)
        ax_FTs1_der.set_xlim(min(Ts1_low),max(Ts1_high))
        ax_FTs1_der.set_ylim(FTs1_der_min,FTs1_der_max)
        if save_fig:
            # enlarge figure to maximum size to prevent the legend and labels from being to crowded (TkAgg backend)
            manager = plt.get_current_fig_manager()
            screen_y = manager.window.winfo_screenheight()
            screen_x = manager.window.winfo_screenwidth()
            manager.resize(*(screen_x/2 - 2*d,screen_y - 4*d)) # screen_x/2 to compensate for using 2 monitors

            path_to_fig = os.path.join(dir_path,'Figures','H_one_link')
            file_name = 'FTs1_der.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))
        
        # Solve networks in different ways, compare convergence and solutions
        # unscaled
        x_sp2, x_sp2_mat, iters_sp2, errors_sp2 = solve_network(heat_net_sp2,x0_sol_sp2,tol,max_iter_sp2,solvers_unscaled,label=label,return_all_x=True)
        x_sol.update(x_sp2)
        x_mat.update(x_sp2_mat)
        iters.update(iters_sp2)
        errors.update(errors_sp2)
        # scaling in solver (in this case using the same base values as for the per unit scaling.)
        x_sp2_ss, x_sp2_mat_ss, iters_sp2_ss, errors_sp2_ss = solve_network(heat_net_sp2,x0_sol_sp2,tol,max_iter_sp2,solvers_scaled,label=label,scaling='matrix',scale_params=scale_params_sp2,det_tol=det_tol,return_all_x=True)
        x_sol_ss.update(x_sp2_ss)
        x_mat_ss.update(x_sp2_mat_ss)
        iters_ss.update(iters_sp2_ss)
        errors_ss.update(errors_sp2_ss)
        # use formulation with unknown half link flows
        x0_sol_sp2_hl = np.array([m_sol_sp2,m_sol_sp2,p_sol_sp2,Ts_init,Tr_sol_sp2,To])
        heat_net_sp2.reset_network(x0_sol_sp2)
        # unscaled
        x_sp2_hl, x_sp2_mat_hl, iters_sp2_hl, errors_sp2_hl = solve_network(heat_net_sp2,x0_sol_sp2_hl,tol,max_iter_sp2,solvers_unscaled,formulation='half_link_flow',label=' hl'+label,return_all_x=True)
        x_sol.update(x_sp2_hl)
        x_mat.update(x_sp2_mat_hl)
        iters.update(iters_sp2_hl)
        errors.update(errors_sp2_hl)
        # scaling in solver (in this case using the same base values as for the per unit scaling.)
        x_sp2_ss_hl, x_sp2_mat_ss_hl, iters_sp2_ss_hl, errors_sp2_ss_hl = solve_network(heat_net_sp2,x0_sol_sp2_hl,tol,max_iter_sp2,solvers_scaled,formulation='half_link_flow',label=' hl'+label,scaling='matrix',scale_params=scale_params_sp2,det_tol=det_tol,return_all_x=True)
        x_sol_ss.update(x_sp2_ss_hl)
        x_mat_ss.update(x_sp2_mat_ss_hl)
        iters_ss.update(iters_sp2_ss_hl)
        errors_ss.update(errors_sp2_ss_hl)
    
    # plot convergence
    fig_conv_init2, (ax_conv_init2, ax_conv_init2_ss) = plt.subplots(2, 1, sharex=True)
    fig_conv_init2.canvas.set_window_title('Convergence standard pipe Pole, different x0, version 2 (smaller value for phi, small pipe)')
    plot_convergence(ax_conv_init2,{**errors},{**iters},tol)
    ax_conv_init2.set_title('unscaled')
    ax_conv_init2.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_init2.set_xlabel('')
    box_conv_init2 = ax_conv_init2.get_position()
    ax_conv_init2.set_position([box_conv_init2.x0, box_conv_init2.y0, box_conv_init2.width * 0.8, box_conv_init2.height]) # Shrink current axis by 20%
    lgd_conv_init2 = ax_conv_init2.legend(loc='center left', bbox_to_anchor=(1, -.1)) # Put a legend to the right of the current axis
    plot_convergence(ax_conv_init2_ss,{**errors_ss},{**iters_ss},tol)
    ax_conv_init2_ss.set_title('scaled')
    ax_conv_init2_ss.set_ylabel('Error ($||D_F F(x^k)||_2$)')
    box_conv_init2_ss = ax_conv_init2_ss.get_position()
    ax_conv_init2_ss.set_position([box_conv_init2_ss.x0, box_conv_init2_ss.y0, box_conv_init2_ss.width * 0.8, box_conv_init2_ss.height]) # Shrink current axis by 20%
    ax_conv_init2_ss.get_legend().remove()
    if save_fig:
        # enlarge figure to maximum size to prevent the legend and labels from being to crowded (TkAgg backend)
        manager = plt.get_current_fig_manager()
        screen_y = manager.window.winfo_screenheight()
        screen_x = manager.window.winfo_screenwidth()
        manager.resize(*(screen_x/2 - 2*d,screen_y - 4*d)) # screen_x/2 to compensate for using 2 monitors

        path_to_fig = os.path.join(dir_path,'Figures','H_one_link')
        file_name = 'Convergence_H_one_link' + file_name_m + file_name_T +'.pgf'
        plt.savefig(os.path.join(path_to_fig, file_name), bbox_extra_artists=(lgd_conv_init2,), bbox_inches='tight')
    
    # plot x for every iteration
    fig_x_hydr, (ax_m, ax_p) = plt.subplots(2, 1, sharex=True, num='fig_x_hydr')
    fig_x_therm, (ax_Ts1, ax_Tr0, ax_Tr1) = plt.subplots(3, 1, sharex=True, num='fig_x_therm')
    fig_x_hydr.canvas.set_window_title('Hydraulic x per iteration (using scaling)')
    fig_x_therm.canvas.set_window_title('Thermal x per iteration (using scaling)')
    max_iters_used = 0
    for key, all_x in x_mat_ss.items():
        if len(all_x.shape) > 1:
            iters_used = iters_ss.get(key)
            max_iters_used = max(max_iters_used,iters_used)
            m = all_x[:,0]
            if ' hl' in key:
                m_hl = all_x[:,1]
                p = all_x[:,2]
                Ts1 = all_x[:,3]
                Tr0 = all_x[:,4]
                Tr1 = all_x[:,5]
            else:
                p = all_x[:,1]
                Ts1 = all_x[:,2]
                Tr0 = all_x[:,3]
                Tr1 = all_x[:,4]
            ax_m.plot(m,'*--',label=key)
            ax_p.plot(p,'*--',label=key)
            ax_Ts1.plot(Ts1,'*--',label=key)
            if np.any(Ts1 > 10*Ts_sol_sp2) or np.any(Ts1 < .1*Ts_sol_sp2):
                print('Ts1 for {} = {}'.format(key,Ts1))
                ax_Ts1.set_yscale('symlog') #semilogy cannot handle negative values.
            ax_Tr0.plot(Tr0,'*--',label=key)
            if np.any(Tr0 > 10*Tr_sol_sp2) or np.any(Tr0 < .1*Tr_sol_sp2):
                ax_Tr0.set_yscale('symlog')
            ax_Tr1.plot(Tr1,'*--',label=key)   
            if np.any(Tr1 > 10*To) or np.any(Tr1 < .1*To):
                ax_Tr1.set_yscale('symlog')
            
    #hydraulic part
    ax_m.plot([0,max_iters_used],[m_sol_sp2,m_sol_sp2],':g',label='sol')
    ax_m.legend()
    ax_m.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_m.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_m.set_title('Link flow')
    ax_m.set_ylabel(r'$m_{01}$')
    box_m = ax_m.get_position()
    ax_m.set_position([box_m.x0, box_m.y0, box_m.width * 0.8, box_m.height]) # Shrink current axis by 20%
    lgd_x_hydr = ax_m.legend(loc='center left', bbox_to_anchor=(1, -.1)) # Put a legend to the right of the current axis
    ax_p.plot([0,max_iters_used],[p_sol_sp2,p_sol_sp2],':g',label='sol')
    ax_p.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_p.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_p.set_title('Pressure')
    ax_p.set_ylabel(r'$p_1$')
    ax_p.set_xlabel('Iteration k')
    box_p = ax_p.get_position()
    ax_p.set_position([box_p.x0, box_p.y0, box_p.width * 0.8, box_p.height]) # Shrink current axis by 20%
    if save_fig:
        plt.figure('fig_x_hydr')
        # enlarge figure to maximum size to prevent the legend and labels from being to crowded (TkAgg backend)
        manager = plt.get_current_fig_manager()
        screen_y = manager.window.winfo_screenheight()
        screen_x = manager.window.winfo_screenwidth()
        manager.resize(*(screen_x/2 - 2*d,screen_y - 4*d)) # screen_x/2 to compensate for using 2 monitors

        path_to_fig = os.path.join(dir_path,'Figures','H_one_link')
        file_name = 'Hydraulic_solution' + file_name_m + file_name_T +'.pgf'
        plt.savefig(os.path.join(path_to_fig, file_name), bbox_extra_artists=(lgd_x_hydr,), bbox_inches='tight')
        
    #thermal part
    ax_Ts1.plot([0,max_iters_used],[Ts_sol_sp2,Ts_sol_sp2],':g',label='sol')
    ax_Ts1.plot([0,max_iters_used],[To,To],':r',label='$T^o$')
    ax_Ts1.plot([0,max_iters_used],[Ts0,Ts0],':k',label='$T^s_0$')
    ax_Ts1.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_Ts1.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_Ts1.set_title('Supply temperature')
    ax_Ts1.set_ylabel(r'$T^s_1$')
    legend_handles, legend_labels = ax_Ts1.get_legend_handles_labels() # Use these handles and labels for the legend
    box_Ts1 = ax_Ts1.get_position()
    ax_Ts1.set_position([box_Ts1.x0, box_Ts1.y0, box_Ts1.width * 0.8, box_Ts1.height]) # Shrink current axis by 20%
    ax_Tr0.plot([0,max_iters_used],[Tr_sol_sp2,Tr_sol_sp2],':g',label='sol')
    ax_Tr0.legend()
    ax_Tr0.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_Tr0.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_Tr0.set_title('Return temperature')
    ax_Tr0.set_ylabel(r'$T^r_0$')
    box_Tr0 = ax_Tr0.get_position()
    ax_Tr0.set_position([box_Tr0.x0, box_Tr0.y0, box_Tr0.width * 0.8, box_Tr0.height]) # Shrink current axis by 20%
    lgd_x_therm = ax_Tr0.legend(handles=legend_handles,labels=legend_labels,loc='center left', bbox_to_anchor=(1, 0.5)) # Put a legend to the right of the current axis
    ax_Tr1.plot([0,max_iters_used],[To,To],':g',label='sol')
    ax_Tr1.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_Tr1.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_Tr1.set_ylabel(r'$T^r_1$')
    ax_Tr1.set_xlabel('Iteration k')
    box_Tr1 = ax_Tr1.get_position()
    ax_Tr1.set_position([box_Tr1.x0, box_Tr1.y0, box_Tr1.width * 0.8, box_Tr1.height]) # Shrink current axis by 20%
    if save_fig:
        plt.figure('fig_x_therm')
        # enlarge figure to maximum size to prevent the legend and labels from being to crowded (TkAgg backend)
        manager = plt.get_current_fig_manager()
        screen_y = manager.window.winfo_screenheight()
        screen_x = manager.window.winfo_screenwidth()
        manager.resize(*(screen_x/2 - 2*d,screen_y - 4*d)) # screen_x/2 to compensate for using 2 monitors
        
        path_to_fig = os.path.join(dir_path,'Figures','H_one_link')
        file_name = 'Thermal_solution' + file_name_m + file_name_T +'.pgf'
        plt.savefig(os.path.join(path_to_fig, file_name), bbox_extra_artists=(lgd_x_therm,), bbox_inches='tight')
        
    # check if solution is correct
    nlsys_sp2 = nlsysh(heat_net_sp2)
    x_sol = np.array([m_sol_sp2,p_sol_sp2,Ts_sol_sp2,Tr_sol_sp2,To])
    heat_net_sp2.update_full(x_sol)
    D_F = nlsysh.DF()
    print('\nCheck solution: ||F(x_sol)||_2 = {:.3e}, ||D_F F(x_sol)||_2 = {:.3e}'.format(np.linalg.norm(nlsys_sp2.F(x_sol)),np.linalg.norm(D_F.dot(nlsys_sp2.F(x_sol)))))        
            
def main(examples,p_perc,m_perc,T_perc,m_perc_list,T_perc_list,solvers_unscaled,solvers_pu,solvers_scaled,dir_path,save_fig,tol=1e-12,plot_sol=False):
    """Solve a network with only one link for different link models. Plot convergence for different solvers. Optionally, also plot difference between calculated solution and actual solution.

    Parameters
    ----------
    examples : list
        Determines which networks are solved. Options are 'slpp1', 'slpp2'.
    p_perc : float
        Percentage of the reference pressure that is used to set the initial guess of the unknown pressure.
    m_perc : float
        Percentage of the outgoing flow that is used to set the initial guess of the link flow.
    T_perc : float
        Percentage of the known temperatures that is used to set the intitial guess of the temperatures
    solvers_unscaled : list
        List of strings indicating which solvers need to be used to solve the unscaled problem.
    solver_pu : list
        List of string indicating which solvers need to be used to solve the scaled problem (per unit scaling).
    solvers_unscaled : list
        List of strings indicating which solvers need to be used to solve the scaled problem (using matrix scaling).
    tol: float, optional
        Tolerance used in solvers. Default is tol = 1e-12
    plot_sol : bool, optional
        If True, plots with the difference between the calculated solution and the actual solution are shown. Default is False
    """
    if 'slpp1' in examples:
        standard_low_pres_pole_1(p_perc,m_perc,T_perc,solvers_unscaled,solvers_pu,solvers_scaled,tol=tol,plot_sol=plot_sol)
    if 'slpp2' in examples:
        if m_perc_list and T_perc_list:
            compare_init_cond_low_pres_pole_2(p_perc,m_perc_list,T_perc_list,solvers_unscaled,solvers_scaled,dir_path,save_fig,tol=tol)
        elif m_perc_list:
            compare_init_cond_low_pres_pole_2(p_perc,m_perc_list,[T_perc],solvers_unscaled,solvers_scaled,dir_path,save_fig,tol=tol)
        elif T_perc_list:
            compare_init_cond_low_pres_pole_2(p_perc,[],T_perc_list,solvers_unscaled,solvers_scaled,dir_path,save_fig,tol=tol)
        else:
            standard_low_pres_pole_2(p_perc,m_perc,T_perc,solvers_unscaled,solvers_pu,solvers_scaled,tol=tol,plot_sol=plot_sol)
        
            
if __name__ == '__main__':
    # parse the command line
    args = command_line_input.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    main(args.ex,args.p_perc,args.m_perc,args.T_perc,args.m_perc_list,args.T_perc_list,args.solv_un,args.solv_pu,args.solv_sc,dir_path,args.save_fig,tol=args.tol,plot_sol=args.plot_sol)

    if args.show_plots:
        plt.show()

