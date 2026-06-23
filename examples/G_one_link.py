"""Example of a gas network consisting of one link"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink
from meslf.networks.carrier import Gas, Water
import numpy as np
import scipy.sparse as sps
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import argparse # To get arguments (as list) from command line

# some constants
mbar = 1e2 #[Pa]
bar = 1e5 #[Pa]
cm = 1e-2 #[m]
km = 1e3 #[m]
hour = 3600 #[s]
atm = 1.01*bar #[Pa]

command_line_input = argparse.ArgumentParser()
command_line_input.add_argument(
    "-ex", # name on the command line 
    nargs = "*", # 0 or more values expected => creates a list
    type=str,
    default = ['lpp', 'lphp', 'lppw', 'hpw', 'hpb', 'hpp', 'res'], # default if nothing is provided
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
    "--plot_top", 
    type=bool,
    default = False,
    )
def create_test_gas_network_one_link(carrier,p_ref,q_out,link_type,link_params):
    """Create a gas network consisting of one single link (a low pressure pipe)

    Parameters
    ----------
    carrier : Carrier
        The carrier in the network
    p_ref : float
        Pressure at reference (inflow) node [Pa]
    q_out : float
        Outflow of load node [kg/s]
    link_type : string
        Type of the link.
    link_params : dict
        Dictionary with the link parameters needed for the link type.

    Returns
    -------
    gas_net : GasNetwork
        The gas network of one link
    """
    gas_net = GasNetwork('single pipe')
    gn0 = GasNode('gn0',node_type=0,x=0,y=0,p=p_ref) # ref node
    gn1 = GasNode('gn2',node_type=1,x=1,y=0,q=q_out) # load node
    gl0 = GasLink('gl0',gn0,gn1,link_type=link_type,link_params=link_params)

    gas_net.add_link(gl0)
    return gas_net

def create_test_gas_network_one_link_fb(carrier,p_ref,q_out,link_type,link_params):
    """Create a gas network consisting of one single link, with the 'fb' link equations. That is, it uses f(q,p1,p2) = delta_p - delta_p(q) as link equation.

    Parameters
    ----------
    carrier : Carrier
        The carrier in the network
    p_ref : float
        Pressure at reference (inflow) node [Pa]
    q_out : float
        Outflow of load node [kg/s]
    link_type : string
        Type of the link.
    link_params : dict
        Dictionary with the link parameters needed for the link type.

    Returns
    -------
    gas_net : GasNetwork
        The gas network of one link
    """
    gas_net_fb = GasNetwork('single pipe')
    gn0_fb = GasNode('gn0',node_type=0,x=0,y=0,p=p_ref) # ref node
    gn1_fb = GasNode('gn2',node_type=1,x=1,y=0,q=q_out) # load node
    gl0_fb = GasLink('gl0',gn0_fb,gn1_fb,link_type=link_type,link_params=link_params,link_eq_form='dp_of_q')

    gas_net_fb.add_link(gl0_fb)
    return gas_net_fb

def initialize_network(gas_net,p_perc=.95,q_perc=1.1,scale_var=None,scale_var_params=None):
    """
    Parameters
    ----------
    gas_net : GasNetwork
        The network to be initialized
    p_perc : float, optional
        Percentage of the reference pressure that is used to set the initial guess of the unknown pressure.
    q_perc : float, optional
        Percentage of the outgoing flow that is used to set the initial guess of the link flow. 

    Returns
    -------
    x0 : np array
        initial guess
    """
    p_init = p_perc*gas_net.nodes[0].p
    q_init = q_perc*gas_net.nodes[1].half_links[0].q
    x_init = np.array([q_init,p_init])
    gas_net.initialize()
    gas_net.update(x_init,formulation='full') # update without scaling, since x_init is unscaled
    x0 = gas_net.set_x_init(formulation='full',scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def create_network_lpp():
    """Create the network with a low pressure pipe using the Pole friction factor.
    This also determines the (analytical) solution of the load flow problem. 
    """
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('gas',S,R_air,1,pn,Tn,Tn)
    # pipe parameters
    L = 100 #[m]
    D = 5*cm #[m]
    link_type = 'pipe_low_pres_pole'
    link_params = {'carrier':gas, 'D':D, 'L':L}
    # boundary conditions
    p_ref = 0.1*mbar #[Pa]
    q_out = 7e-5 #[kg/s]
    # create the networks
    gas_net = create_test_gas_network_one_link(gas,p_ref,q_out,link_type,link_params)
    gas_net_fb = create_test_gas_network_one_link_fb(gas,p_ref,q_out,link_type,link_params)
    # determine analytical solution
    C = np.pi/8 * np.sqrt(2*pn*S*D**5/(Tn*R_air*L))
    f = 0.0065
    q_sol = q_out #[kg/s]
    p_sol = p_ref - f/C**2*q_sol**2
    return gas_net, gas_net_fb, q_sol, p_sol

def create_network_lphp():
    """Create the network with a low pressure pipe using the Hagen-Poiseuille friction factor.
    This also determines the (analytical) solution of the load flow problem. 
    """
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('gas',S,R_air,1,pn,Tn,Tn)
    rhon_g = gas.rhon
    dyn_vis = 1e-5 #[Pa/s] = [kg/(m s)]
    kin_vis = dyn_vis/rhon_g #[m^s/s]
    gas.mu = kin_vis # about 1.4e-5
    # pipe parameters
    L = 100 #[m]
    D = 5*cm #[m]
    link_type = 'pipe_low_pres_hagen_poiseuille'
    link_params = {'carrier':gas, 'D':D, 'L':L}
    # boundary conditions
    p_ref = 0.1*mbar #[Pa]
    q_out = 7e-5 #[kg/s]
    # create the networks
    gas_net = create_test_gas_network_one_link(gas,p_ref,q_out,link_type,link_params)
    gas_net_fb = create_test_gas_network_one_link_fb(gas,p_ref,q_out,link_type,link_params)
    # determine analytical solution
    C = np.pi/8 * np.sqrt(2*pn*S*D**5/(Tn*R_air*L))
    f = 0.0065
    q_sol = q_out #[kg/s]
    p_sol = p_ref - f/C**2*q_sol**2 # about 0.0936mbar
    return gas_net, gas_net_fb, q_sol, p_sol

def create_network_lpp_water():
    """Create the network with a low pressure pipe using the Pole friction factor, with water instead of gas
    This also determines the (analytical) solution of the load flow problem. 
    """
    # carrier
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1e3 #[kg/m^3]
    gas = Water('water',Cp,rho=rho)
    # pipe parameters
    L = 1*km #[m]
    D = 0.2 #[m]
    link_type = 'pipe_low_pres_pole'
    link_params = {'carrier':gas, 'D':D, 'L':L}
    # boundary conditions
    p_ref = 5*bar #[Pa]
    q_out = 31.4149
    # create the networks
    gas_net = create_test_gas_network_one_link(gas,p_ref,q_out,link_type,link_params)
    gas_net_fb = create_test_gas_network_one_link_fb(gas,p_ref,q_out,link_type,link_params)
    # determine analytical solution
    q_sol = q_out #[kg/s]
    p_sol = 4.35*bar #[Pa]
    return gas_net, gas_net_fb, q_sol, p_sol

def create_network_hpw():
    """Create the network with a high pressure pipe using the Weymouth friction factor.
    This also determines the (analytical) solution of the load flow problem. 
    """
    # carrier
    Z = 0.96
    T = 300 #[K]
    S = 0.7
    Tn = 288 #[K]
    pn = 1.013*bar #[Pa]
    R = 8.314 #[J/molK]
    M = 29e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    rhon_g = gas.rhon
    dyn_vis = 1e-5 #[Pa/s] = [kg/(m s)]
    kin_vis = dyn_vis/rhon_g #[m^s/s]
    gas.mu = kin_vis
    # pipe parameters
    L = 10.*km #[m]
    D = 0.3 #[m]
    E = 1
    link_type = 'pipe_high_pres_weymouth'
    link_params = {'carrier':gas, 'D':D, 'L':L, 'E':E}
    # boundary conditions
    p_ref = 32*bar #[Pa]
    q_out = 1.066e6 #[m^3/dag]
    q_out /= (24*hour) #[m^3/s]
    q_out *= gas.rhon #[kg/s]
    # create the networks
    gas_net = create_test_gas_network_one_link(gas,p_ref,q_out,link_type,link_params)
    gas_net_fb = create_test_gas_network_one_link_fb(gas,p_ref,q_out,link_type,link_params)
    # determine analytical solution
    C = np.pi/8 * np.sqrt(S*D**5/(T*R_air*L*Z))
    f = 1/(20.64**2*D**(1/3)*E**2)
    q_sol = q_out #[kg/s]
    p_sol = 30*bar #[Pa] Based on the email conversation with Johan on 22-03-2019. See also test_pipe_high_pressure_weymouth_2() in test_hydraulic
    return gas_net, gas_net_fb, q_sol, p_sol

def create_network_hpb():
    """Create the network with a high pressure pipe using the Blasius friction factor.
    This also determines the (analytical) solution of the load flow problem. 
    """
    # carrier
    Z = 0.96
    T = 300 #[K]
    S = 0.7
    Tn = 288 #[K]
    pn = 1.013*bar #[Pa]
    R = 8.314 #[J/molK]
    M = 29e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    rhon_g = gas.rhon
    dyn_vis = 1e-5 #[Pa/s] = [kg/(m s)]
    kin_vis = dyn_vis/rhon_g #[m^s/s]
    gas.mu = kin_vis
    # pipe parameters
    L = 10.*km #[m]
    D = 0.3 #[m]
    link_type = 'pipe_high_pres_blasius'
    link_params = {'carrier':gas, 'D':D, 'L':L}
    # boundary conditions
    p_ref = 32*bar #[Pa]
    q_out = 1.066e6 #[m^3/dag]
    q_out /= (24*hour) #[m^3/s]
    q_out *= gas.rhon #[kg/s]
    # create the networks
    gas_net = create_test_gas_network_one_link(gas,p_ref,q_out,link_type,link_params)
    gas_net_fb = create_test_gas_network_one_link_fb(gas,p_ref,q_out,link_type,link_params)
    # determine analytical solution
    q_sol = q_out #[kg/s]
    p_sol = 30*bar #[Pa] Based on the email conversation with Johan on 22-03-2019. See also test_pipe_high_pressure_weymouth_2() in test_hydraulic
    return gas_net, gas_net_fb, q_sol, p_sol

def create_network_hpp():
    """Create the network with a `high pressure pipe' using the Pole friction factor. That is, use the `high pressure pipe', which is the general steady-state flow equation, but use the same values as for the low pressure pipe with Pole friction factor. Since the high pressure pipe model uses abolute pressures instead of gauge pressures, 1atm has to be added. The (analytical) solution of the load flow problem should then be the same as for the low pressure pipe. 
    """
    # carrier
    Z = 1
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    T = Tn
    R = 8.314 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    # pipe parameters
    L = 100 #[m]
    D = 5*cm #[m]
    link_type = 'pipe_high_pres_pole'
    link_params = {'carrier':gas, 'D':D, 'L':L}
    # boundary conditions
    p_ref = 0.1*mbar+atm #[Pa]
    q_out = 7e-5 #[kg/s]
    # create the networks
    gas_net = create_test_gas_network_one_link(gas,p_ref,q_out,link_type,link_params)
    gas_net_fb = create_test_gas_network_one_link_fb(gas,p_ref,q_out,link_type,link_params)
    # determine analytical solution (based on low pressure equation)
    C = np.pi/8 * np.sqrt(2*pn*S*D**5/(Tn*R_air*L))
    f = 0.0065
    q_sol = q_out #[kg/s]
    p_sol = p_ref - f/C**2*q_sol**2 #[Pa]
    return gas_net, gas_net_fb, q_sol, p_sol

def create_network_resistor():
    """Create the network with a resistor. Use data similar to the high pressure Weymouth pipe
    This also determines the (analytical) solution of the load flow problem. 
    """
    # carrier
    Z = 0.96
    T = 300 #[K]
    S = 0.7
    Tn = 288 #[K]
    pn = 1.013*bar #[Pa]
    R = 8.314 #[J/molK]
    M = 29e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    rhon_g = gas.rhon
    dyn_vis = 1e-5 #[Pa/s] = [kg/(m s)]
    kin_vis = dyn_vis/rhon_g #[m^s/s]
    gas.mu = kin_vis
    # pipe parameters
    C = 0.022 # based on C/sqrt(f) of the high pressure weymouth pipe
    link_type = 'resistor'
    link_params = {'carrier':gas, 'C':C}
    # boundary conditions
    p_ref = 32*bar #[Pa]
    q_out = 1.066e6 #[m^3/dag]
    q_out /= (24*hour) #[m^3/s]
    q_out *= gas.rhon #[kg/s]
    # create the networks
    gas_net = create_test_gas_network_one_link(gas,p_ref,q_out,link_type,link_params)
    gas_net_fb = create_test_gas_network_one_link_fb(gas,p_ref,q_out,link_type,link_params)
    # determine analytical solution
    q_sol = q_out #[kg/s]
    p_sol = p_ref - 1/C**2*q_sol**2 #[Pa] 
    return gas_net, gas_net_fb, q_sol, p_sol
        
def solve_network(gas_net,x0,tol,max_iter,solvers,scaling=None,scale_params=None,solve_both_forms=True,h=1e-6,label=None,plot_top=False):
    """Solve the network in different ways.
    
    Parameters
    ----------
    gas_net : GasNetwork
        The network to be solved.
    x0 : np array
        Vector with initial guess, assuming 'full' formulation.
    solvers : list
        List of solvers to use.
    solve_both_forms : bool, optional
        When True both 'nodal' and 'full' formulation are used. When False, only 'full' formulation is used.
    h : float, optional
        Step size used for FD Jaciobian. Defaul is 1e-6. This h is only used when 'NR_FD' in the list of solvers.
    label : string, optional
        String to add to the end of the keys used in the dictionary. 
    plot_top : bool, optional
        If True, plot topology of the (solved) network. Default is False
        
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
    scale_var = None
    scale_var_params = None
    if scaling == 'per_unit':
        scale_var = scaling
        scale_var_params = scale_params
    elif scaling == 'matrix':
        if scale_params:
            qbase = scale_params['qbase']
            pbase = scale_params['pbase']
        else:
            qbase = None
            pbase = None
        scale_var_params = {'qbase':qbase,'pgbase':pbase}    
    for solver in solvers:
        gas_net.reset_network(x0,formulation='full')
        if label:
            key_full = solver+' full'+label
        else: 
            key_full = solver+' full'
        x[key_full],iters[key_full],errors[key_full],_,_,_ = gas_net.solve_network(tol,max_iter,formulation='full',solver=solver,h=h,scale_var=scaling,scale_var_params=scale_var_params)
        if solve_both_forms:
            gas_net.reset_network(x0,'full')
            x0_nodal = gas_net.set_x_init('nodal')
            if label:
                key_nodal = solver+' nodal'+label
            else: 
                key_nodal = solver+' nodal'
            x[key_nodal],iters[key_nodal],errors[key_nodal],_,_,_ = gas_net.solve_network(tol,max_iter,formulation='nodal',solver=solver,h=h,scale_var=scaling,scale_var_params=scale_var_params)
    if plot_top:
        plt.figure('Network topology')
        ax_top = plt.gca()
        gas_net.draw_network(ax_top)
        plt.axis('equal')
        plt.axis('off')
        plt.plot()
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
        # different markers for the different link equations fa and fb
        if 'fb' in key:
            ls = 'o'
        else:
            ls = '*'           
        
        if 'root' in key or 'fsolver' in key:
            ls += '--' # use dashed lines is 'root' or 'fsolver' are used, since they only give the first and final error, and no information in between.
        elif 'NR_FD' in key: 
            ls += '-.' # use dashed dotted lines for FD solver, to be able to see if it is on top of the analytical solver.
        else:
            ls += '-'

        label = key.replace("_"," ")
        iters_used = iters.get(key)
        max_iters_used = max(max_iters_used,iters_used)
        if len(err) != iters_used+1: # for fsolver and root
            ax.semilogy(np.asarray([0,iters_used]),err,ls,label=label)
        else:
            ax.semilogy(err,ls,label=label)
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
        if 'nodal' in key:
            x_p = x[0]
        else:
            x_p = x[1]
        ax.plot(ind,(p_sol-x_p),'k*')
        xtick = key
        xtick = xtick.replace("_"," ")
        xticks.append(xtick)
        ind += 1
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    #ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
    ax.set_ylabel('p - $x^p$ [Pa]')
    ax.set_xticks(range(ind))
    ax.set_xticklabels(xticks,rotation='vertical')
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)

def plot_flow_sol_error(ax,q_sol,x_sol):
    """Plots the error between the found solution and the actual solution for flow"""
    xticks = []
    ind = 0
    for key, x in x_sol.items():
        if 'nodal' in key:
            pass
        else:
            x_q = x[0]
            ax.plot(ind,(q_sol-x_q),'k*')
            xtick = key
            xtick = xtick.replace("_"," ")
            xticks.append(xtick)
            ind += 1
    ax.set_ylabel('m - $x^m$ [kg/s]')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    #ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
    ax.set_xticks(range(ind))
    ax.set_xticklabels(xticks,rotation='vertical')
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)

def low_pres_pole(solvers_unscaled,solvers_pu,solvers_scaled,tol=1e-12,plot_sol=True,plot_top=False):
    """Solve the network with a low pressure pipe using the Pole friction factor.
    
    Parameters
    ----------
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
    print('\nLow pressure pipe with Pole friction factor')
    # create network and solution
    gas_net_lpp, gas_net_lpp_fb, q_sol_lpp, p_sol_lpp = create_network_lpp()
    print('solution for low pressure Pole: m = {:.3e} [kg/s], p = {:.3e} [Pa]'.format(q_sol_lpp,p_sol_lpp))
    # initialize networks
    x0_lpp = initialize_network(gas_net_lpp)
    x0_lpp_fb = initialize_network(gas_net_lpp_fb)
    scale_params_lpp = {'qbase':1e-5,'pbase':1*mbar}
    # Solve networks in different ways, compare convergence and solutions
    max_iter_lpp = 50
    # low pressure pipe with Pole friction factor
    print('solving system with low pressure pipe with Pole friction factor')
    # unscaled
    x_lpp, iters_lpp, errors_lpp = solve_network(gas_net_lpp,x0_lpp,tol,max_iter_lpp,solvers_unscaled,label=' fa')
    x_lpp_fb, iters_lpp_fb, errors_lpp_fb = solve_network(gas_net_lpp_fb,x0_lpp_fb,tol,max_iter_lpp,solvers_unscaled,solve_both_forms=False,label=' fb',plot_top=plot_top)
    # per unit 
    x_lpp_pu, iters_lpp_pu, errors_lpp_pu = solve_network(gas_net_lpp,x0_lpp,tol,max_iter_lpp,solvers_pu,label=' fa',scaling='per_unit',scale_params=scale_params_lpp)
    x_lpp_fb_pu, iters_lpp_fb_pu, errors_lpp_fb_pu = solve_network(gas_net_lpp_fb,x0_lpp_fb,tol,max_iter_lpp,solvers_pu,solve_both_forms=False,label=' fb',scaling='per_unit',scale_params=scale_params_lpp)
    # scaling in solver (in this case using the same base values as for the per unit scaling.)
    x_lpp_ss, iters_lpp_ss, errors_lpp_ss = solve_network(gas_net_lpp,x0_lpp,tol,max_iter_lpp,solvers_scaled,label=' fa',scaling='matrix',scale_params=scale_params_lpp)
    x_lpp_fb_ss, iters_lpp_fb_ss, errors_lpp_fb_ss = solve_network(gas_net_lpp_fb,x0_lpp_fb,tol,max_iter_lpp,solvers_scaled,solve_both_forms=False,label=' fb',scaling='matrix',scale_params=scale_params_lpp)
    # plot convergence
    fig_conv_lpp, (ax_conv_lpp, ax_conv_lpp_pu, ax_conv_lpp_ss) = plt.subplots(3, 1, sharex=True)
    fig_conv_lpp.canvas.set_window_title('Convergence low pres. Pole')
    plot_convergence(ax_conv_lpp,{**errors_lpp,**errors_lpp_fb},{**iters_lpp,**iters_lpp_fb},tol)
    ax_conv_lpp.set_title('unscaled')
    ax_conv_lpp.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_lpp.set_xlabel('')
    plot_convergence(ax_conv_lpp_pu,{**errors_lpp_pu,**errors_lpp_fb_pu},{**iters_lpp_pu,**iters_lpp_fb_pu},tol)
    ax_conv_lpp_pu.set_title('per unit scaling')
    ax_conv_lpp_pu.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_lpp_pu.set_xlabel('')
    plot_convergence(ax_conv_lpp_ss,{**errors_lpp_ss,**errors_lpp_fb_ss},{**iters_lpp_ss,**iters_lpp_fb_ss},tol)
    ax_conv_lpp_ss.set_title('scaled')
    ax_conv_lpp_ss.set_ylabel('Error ($||D_F F(x^k)||_2$)')
    if plot_sol:
        # plot error w.r.t solution for pressure
        fig_pres_err_lpp, (ax_pres_err_lpp, ax_pres_err_lpp_pu, ax_pres_err_lpp_ss) = plt.subplots(1, 3, sharey=True)
        fig_pres_err_lpp.canvas.set_window_title('Pressure error low pres. Pole, with gauge p = {:.2e} [Pa] = {:.2e} [bar]'.format(p_sol_lpp,p_sol_lpp/bar))
        plot_pres_sol_error(ax_pres_err_lpp,p_sol_lpp,{**x_lpp,**x_lpp_fb})
        ax_pres_err_lpp.set_title('unscaled')
        x_lpp_pu_SI = dict()
        for key, x in {**x_lpp_pu,**x_lpp_fb_pu}.items():
            if 'nodal' in key:
                x_lpp_pu_SI[key] = x*scale_params_lpp['pbase']
            else:
                x_lpp_pu_SI[key] = x*np.array([scale_params_lpp['qbase'],scale_params_lpp['pbase']])
        plot_pres_sol_error(ax_pres_err_lpp_pu,p_sol_lpp,x_lpp_pu_SI)
        ax_pres_err_lpp_pu.set_title('per unit scaling')
        plot_pres_sol_error(ax_pres_err_lpp_ss,p_sol_lpp,{**x_lpp_ss,**x_lpp_fb_ss})
        ax_pres_err_lpp_ss.set_title('scaled')
        # plot error w.r.t solution for flow
        fig_flow_err_lpp, (ax_flow_err_lpp, ax_flow_err_lpp_pu, ax_flow_err_lpp_ss) = plt.subplots(1, 3, sharey=True)
        fig_flow_err_lpp.canvas.set_window_title('Flow error low pres. Pole, with q = {:.2e} [kg/s]'.format(q_sol_lpp))
        plot_flow_sol_error(ax_flow_err_lpp,q_sol_lpp,{**x_lpp,**x_lpp_fb})
        print('solution for lpp: {}'.format({**x_lpp,**x_lpp_fb}))
        ax_flow_err_lpp.set_title('unscaled')
        plot_flow_sol_error(ax_flow_err_lpp_pu,q_sol_lpp,x_lpp_pu_SI)
        ax_flow_err_lpp_pu.set_title('per unit scaling')
        plot_flow_sol_error(ax_flow_err_lpp_ss,q_sol_lpp,{**x_lpp_ss,**x_lpp_fb_ss})
        ax_flow_err_lpp_ss.set_title('scaled')

def low_pres_hagen_poiseuille(solvers_unscaled,solvers_pu,solvers_scaled,tol=1e-12,plot_sol=True,plot_top=False):
    """Solve the network with a low pressure pipe using Hagen-Poiseuille friction factor.
    
    Parameters
    ----------
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
    print('\nLow pressure pipe with Hagen-Poiseuille friction factor')
    # create network and solution
    gas_net_lphp, gas_net_lphp_fb, q_sol_lphp, p_sol_lphp = create_network_lphp()
    print('solution for low pressure Hagen-Poiseuille: m = {:.3e} [kg/s], p = {:.3e} [Pa]'.format(q_sol_lphp,p_sol_lphp))
    # initialize network
    x0_lphp = initialize_network(gas_net_lphp)
    x0_lphp_fb = initialize_network(gas_net_lphp_fb)
    scale_params_lphp = {'qbase':1e-5,'pbase':1*mbar}
    # Solve networks in different ways, compare convergence and solutions
    max_iter_lphp = 10
    print('\nsolving system with low pressure pipe with Hagen-Poiseuille friction factor')
    # unscaled
    x_lphp, iters_lphp, errors_lphp = solve_network(gas_net_lphp,x0_lphp,tol,max_iter_lphp,solvers_unscaled,solve_both_forms=False,label=' fa')
    x_lphp_fb, iters_lphp_fb, errors_lphp_fb = solve_network(gas_net_lphp_fb,x0_lphp_fb,tol,max_iter_lphp,solvers_unscaled,solve_both_forms=False,label=' fb',plot_top=plot_top)
    # per unit 
    x_lphp_pu, iters_lphp_pu, errors_lphp_pu = solve_network(gas_net_lphp,x0_lphp,tol,max_iter_lphp,solvers_pu,solve_both_forms=False,label=' fa',scaling='per_unit',scale_params=scale_params_lphp)
    x_lphp_fb_pu, iters_lphp_fb_pu, errors_lphp_fb_pu = solve_network(gas_net_lphp_fb,x0_lphp_fb,tol,max_iter_lphp,solvers_pu,solve_both_forms=False,label=' fb',scaling='per_unit',scale_params=scale_params_lphp)
    # scaling in solver (in this case using the same base values as for the per unit scaling.)
    x_lphp_ss, iters_lphp_ss, errors_lphp_ss = solve_network(gas_net_lphp,x0_lphp,tol,max_iter_lphp,solvers_scaled,solve_both_forms=False,label=' fa',scaling='matrix',scale_params=scale_params_lphp)
    x_lphp_fb_ss, iters_lphp_fb_ss, errors_lphp_fb_ss = solve_network(gas_net_lphp_fb,x0_lphp_fb,tol,max_iter_lphp,solvers_scaled,solve_both_forms=False,label=' fb',scaling='matrix',scale_params=scale_params_lphp)
    # plot convergence
    fig_conv_lphp, (ax_conv_lphp, ax_conv_lphp_pu, ax_conv_lphp_ss) = plt.subplots(3, 1, sharex=True)
    fig_conv_lphp.canvas.set_window_title('Convergence low pres. Hagen-Poiseuille')
    plot_convergence(ax_conv_lphp,{**errors_lphp,**errors_lphp_fb},{**iters_lphp,**iters_lphp_fb},tol)
    ax_conv_lphp.set_title('unscaled')
    ax_conv_lphp.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_lphp.set_xlabel('')
    plot_convergence(ax_conv_lphp_pu,{**errors_lphp_pu,**errors_lphp_fb_pu},{**iters_lphp_pu,**iters_lphp_fb_pu},tol)
    ax_conv_lphp_pu.set_title('per unit scaling')
    ax_conv_lphp_pu.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_lphp_pu.set_xlabel('')
    plot_convergence(ax_conv_lphp_ss,{**errors_lphp_ss,**errors_lphp_fb_ss},{**iters_lphp_ss,**iters_lphp_fb_ss},tol)
    ax_conv_lphp_ss.set_title('scaled')
    ax_conv_lphp_ss.set_ylabel('Error ($||D_F F(x^k)||_2$)')
    if plot_sol:
        # plot error w.r.t solution for pressure
        fig_pres_err_lphp, (ax_pres_err_lphp, ax_pres_err_lphp_pu, ax_pres_err_lphp_ss) = plt.subplots(1, 3, sharey=True)
        fig_pres_err_lphp.canvas.set_window_title('Pressure error low pres. Hagen-Poiseuille, with gauge p = {:.2e} [Pa] = {:.2e} [bar]'.format(p_sol_lphp,p_sol_lphp/bar))
        plot_pres_sol_error(ax_pres_err_lphp,p_sol_lphp,{**x_lphp,**x_lphp_fb})
        ax_pres_err_lphp.set_title('unscaled')
        x_lphp_pu_SI = dict()
        for key, x in {**x_lphp_pu,**x_lphp_fb_pu}.items():
            if 'nodal' in key:
                x_lphp_pu_SI[key] = x*scale_params_lphp['pbase']
            else:
                x_lphp_pu_SI[key] = x*np.array([scale_params_lphp['qbase'],scale_params_lphp['pbase']])
        plot_pres_sol_error(ax_pres_err_lphp_pu,p_sol_lphp,x_lphp_pu_SI)
        ax_pres_err_lphp_pu.set_title('per unit scaling')
        plot_pres_sol_error(ax_pres_err_lphp_ss,p_sol_lphp,{**x_lphp_ss,**x_lphp_fb_ss})
        ax_pres_err_lphp_ss.set_title('scaled')
        # plot error w.r.t solution for flow
        fig_flow_err_lphp, (ax_flow_err_lphp, ax_flow_err_lphp_pu, ax_flow_err_lphp_ss) = plt.subplots(1, 3, sharey=True)
        fig_flow_err_lphp.canvas.set_window_title('Flow error low pres. Hagen-Poiseuille, with q = {:.2e} [kg/s]'.format(q_sol_lphp))
        plot_flow_sol_error(ax_flow_err_lphp,q_sol_lphp,{**x_lphp,**x_lphp_fb})
        print('solution for lphp: {}'.format({**x_lphp,**x_lphp_fb}))
        ax_flow_err_lphp.set_title('unscaled')
        plot_flow_sol_error(ax_flow_err_lphp_pu,q_sol_lphp,x_lphp_pu_SI)
        ax_flow_err_lphp_pu.set_title('per unit scaling')
        plot_flow_sol_error(ax_flow_err_lphp_ss,q_sol_lphp,{**x_lphp_ss,**x_lphp_fb_ss})
        ax_flow_err_lphp_ss.set_title('scaled')
        
def low_pres_pole_water(solvers_unscaled,solvers_pu,solvers_scaled,tol=1e-12,plot_sol=True,plot_top=False):
    """Solve the network with a low pressure pipe using the Pole friction factor, with water as a carrier.
    
    Parameters
    ----------
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
    print('\nLow pressure pipe with Pole friction factor, and water as carrier')
    # create network and solution
    gas_net_lpp_w, gas_net_lpp_w_fb, q_sol_lpp_w, p_sol_lpp_w = create_network_lpp_water()
    print('solution for low pressure Pole, with water as carrier: m = {:.3e} [kg/s], p = {:.3e} [Pa]'.format(q_sol_lpp_w,p_sol_lpp_w))
    # initialize network
    x0_lpp_w = initialize_network(gas_net_lpp_w)
    x0_lpp_w_fb = initialize_network(gas_net_lpp_w_fb)
    scale_params_lpp_w = {'qbase':10,'pbase':1*bar}
    # Solve networks in different ways, compare convergence and solutions
    max_iter_lpp_w = 10
    print('\nsolving system with low pressure pipe with Pole friction factor, with water as carrier')
    # unscaled
    x_lpp_w, iters_lpp_w, errors_lpp_w = solve_network(gas_net_lpp_w,x0_lpp_w,tol,max_iter_lpp_w,solvers_unscaled,label=' fa')
    x_lpp_w_fb, iters_lpp_w_fb, errors_lpp_w_fb = solve_network(gas_net_lpp_w_fb,x0_lpp_w_fb,tol,max_iter_lpp_w,solvers_unscaled,solve_both_forms=False,label=' fb',plot_top=plot_top)
    # per unit 
    x_lpp_w_pu, iters_lpp_w_pu, errors_lpp_w_pu = solve_network(gas_net_lpp_w,x0_lpp_w,tol,max_iter_lpp_w,solvers_pu,label=' fa',scaling='per_unit',scale_params=scale_params_lpp_w)
    x_lpp_w_fb_pu, iters_lpp_w_fb_pu, errors_lpp_w_fb_pu = solve_network(gas_net_lpp_w_fb,x0_lpp_w_fb,tol,max_iter_lpp_w,solvers_pu,solve_both_forms=False,label=' fb',scaling='per_unit',scale_params=scale_params_lpp_w)
    # scaling in solver (in this case using the same base values as for the per unit scaling.)
    x_lpp_w_ss, iters_lpp_w_ss, errors_lpp_w_ss = solve_network(gas_net_lpp_w,x0_lpp_w,tol,max_iter_lpp_w,solvers_scaled,label=' fa',scaling='matrix',scale_params=scale_params_lpp_w)
    x_lpp_w_fb_ss, iters_lpp_w_fb_ss, errors_lpp_w_fb_ss = solve_network(gas_net_lpp_w_fb,x0_lpp_w_fb,tol,max_iter_lpp_w,solvers_scaled,solve_both_forms=False,label=' fb',scaling='matrix',scale_params=scale_params_lpp_w)
    # plot convergence
    fig_conv_lpp_w, (ax_conv_lpp_w, ax_conv_lpp_w_pu, ax_conv_lpp_w_ss) = plt.subplots(3, 1, sharex=True)
    fig_conv_lpp_w.canvas.set_window_title('Convergence water')
    plot_convergence(ax_conv_lpp_w,{**errors_lpp_w,**errors_lpp_w_fb},{**iters_lpp_w,**iters_lpp_w_fb},tol)
    ax_conv_lpp_w.set_title('unscaled')
    ax_conv_lpp_w.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_lpp_w.set_xlabel('')
    plot_convergence(ax_conv_lpp_w_pu,{**errors_lpp_w_pu,**errors_lpp_w_fb_pu},{**iters_lpp_w_pu,**iters_lpp_w_fb_pu},tol)
    ax_conv_lpp_w_pu.set_title('per unit scaling')
    ax_conv_lpp_w_pu.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_lpp_w_pu.set_xlabel('')
    plot_convergence(ax_conv_lpp_w_ss,{**errors_lpp_w_ss,**errors_lpp_w_fb_ss},{**iters_lpp_w_ss,**iters_lpp_w_fb_ss},tol)
    ax_conv_lpp_w_ss.set_title('scaled')
    ax_conv_lpp_w_ss.set_ylabel('Error ($||D_F F(x^k)||_2$)')
    if plot_sol:
        # plot error w.r.t solution for pressure
        fig_pres_err_lpp_w, (ax_pres_err_lpp_w, ax_pres_err_lpp_w_pu, ax_pres_err_lpp_w_ss) = plt.subplots(1, 3, sharey=True)
        x_lpp_w_conv = dict() # not all the unscaled methods have converged. Only plot errors for the ones that did.
        for key, x in {**x_lpp_w,**x_lpp_w_fb}.items():
            if {**errors_lpp_w,**errors_lpp_w_fb}.get(key)[-1] < tol:
                x_lpp_w_conv[key] = x
        fig_pres_err_lpp_w.canvas.set_window_title('Pressure error water, with (gauge?) p = {:.2e} [Pa] = {:.2e} [bar]'.format(p_sol_lpp_w,p_sol_lpp_w/bar))
        plot_pres_sol_error(ax_pres_err_lpp_w,p_sol_lpp_w,x_lpp_w_conv)
        ax_pres_err_lpp_w.set_title('unscaled')
        x_lpp_w_pu_SI = dict()
        for key, x in {**x_lpp_w_pu,**x_lpp_w_fb_pu}.items():
            if 'nodal' in key:
                x_lpp_w_pu_SI[key] = x*scale_params_lpp_w['pbase']
            else:
                x_lpp_w_pu_SI[key] = x*np.array([scale_params_lpp_w['qbase'],scale_params_lpp_w['pbase']])
        plot_pres_sol_error(ax_pres_err_lpp_w_pu,p_sol_lpp_w,x_lpp_w_pu_SI)
        ax_pres_err_lpp_w_pu.set_title('per unit scaling')
        plot_pres_sol_error(ax_pres_err_lpp_w_ss,p_sol_lpp_w,{**x_lpp_w_ss,**x_lpp_w_fb_ss})
        ax_pres_err_lpp_w_ss.set_title('scaled')
        # plot error w.r.t solution for flow
        fig_flow_err_lpp_w, (ax_flow_err_lpp_w, ax_flow_err_lpp_w_pu, ax_flow_err_lpp_w_ss) = plt.subplots(1, 3, sharey=True)
        fig_flow_err_lpp_w.canvas.set_window_title('Flow error water, with q = {:.2e} [kg/s]'.format(q_sol_lpp_w))
        plot_flow_sol_error(ax_flow_err_lpp_w,q_sol_lpp_w,x_lpp_w_conv)
        ax_flow_err_lpp_w.set_title('unscaled')
        plot_flow_sol_error(ax_flow_err_lpp_w_pu,q_sol_lpp_w,x_lpp_w_pu_SI)
        ax_flow_err_lpp_w_pu.set_title('per unit scaling')
        plot_flow_sol_error(ax_flow_err_lpp_w_ss,q_sol_lpp_w,{**x_lpp_w_ss,**x_lpp_w_fb_ss})
        ax_flow_err_lpp_w_ss.set_title('scaled')
        
def high_pres_weymouth(solvers_unscaled,solvers_pu,solvers_scaled,tol=1e-12,plot_sol=True,plot_top=False):
    """Solve the network with a high pressure pipe using Weymouth friction factor.
    
    Parameters
    ----------
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
    print('\nHigh pressure pipe with Weymouth friction factor')
    # create network and solution
    gas_net_hpw, gas_net_hpw_fb, q_sol_hpw, p_sol_hpw = create_network_hpw()
    print('solution for Weymouth: m = {:.3e} [kg/s], p = {:.3e} [Pa]'.format(q_sol_hpw,p_sol_hpw))
    # initialize network
    p_perc_hpw = 0.8
    q_perc_hpw = 0.5
    x0_hpw = initialize_network(gas_net_hpw,p_perc=p_perc_hpw,q_perc=q_perc_hpw)
    x0_hpw_fb = initialize_network(gas_net_hpw_fb,p_perc=p_perc_hpw,q_perc=q_perc_hpw)
    scale_params_hpw = {'qbase':10,'pbase':10*bar}
    # Solve networks in different ways, compare convergence and solutions
    max_iter_hpw = 8
    print('\nsolving system with high pressure pipe with Weymouth friction factor')
    # unscaled
    x_hpw, iters_hpw, errors_hpw = solve_network(gas_net_hpw,x0_hpw,tol,max_iter_hpw,solvers_unscaled,label=' fa')
    x_hpw_fb, iters_hpw_fb, errors_hpw_fb = solve_network(gas_net_hpw_fb,x0_hpw_fb,tol,max_iter_hpw,solvers_unscaled,solve_both_forms=False,label=' fb',plot_top=plot_top)
    # per unit 
    x_hpw_pu, iters_hpw_pu, errors_hpw_pu = solve_network(gas_net_hpw,x0_hpw,tol,max_iter_hpw,solvers_pu,label=' fa',scaling='per_unit',scale_params=scale_params_hpw)
    x_hpw_fb_pu, iters_hpw_fb_pu, errors_hpw_fb_pu = solve_network(gas_net_hpw_fb,x0_hpw_fb,tol,max_iter_hpw,solvers_pu,solve_both_forms=False,label=' fb',scaling='per_unit',scale_params=scale_params_hpw)
    # scaling in solver (in this case using the same base values as for the per unit scaling.)
    x_hpw_ss, iters_hpw_ss, errors_hpw_ss = solve_network(gas_net_hpw,x0_hpw,tol,max_iter_hpw,solvers_scaled,label=' fa',scaling='matrix',scale_params=scale_params_hpw)
    x_hpw_fb_ss, iters_hpw_fb_ss, errors_hpw_fb_ss = solve_network(gas_net_hpw_fb,x0_hpw_fb,tol,max_iter_hpw,solvers_scaled,solve_both_forms=False,label=' fb',scaling='matrix',scale_params=scale_params_hpw)
    # plot convergence
    fig_conv_hpw, (ax_conv_hpw, ax_conv_hpw_pu, ax_conv_hpw_ss) = plt.subplots(3, 1, sharex=True)
    fig_conv_hpw.canvas.set_window_title('Convergence high pres. Weymouth')
    plot_convergence(ax_conv_hpw,{**errors_hpw,**errors_hpw_fb},{**iters_hpw,**iters_hpw_fb},tol)
    ax_conv_hpw.set_title('unscaled')
    ax_conv_hpw.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_hpw.set_xlabel('')
    plot_convergence(ax_conv_hpw_pu,{**errors_hpw_pu,**errors_hpw_fb_pu},{**iters_hpw_pu,**iters_hpw_fb_pu},tol)
    ax_conv_hpw_pu.set_title('per unit scaling')
    ax_conv_hpw_pu.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_hpw_pu.set_xlabel('')
    plot_convergence(ax_conv_hpw_ss,{**errors_hpw_ss,**errors_hpw_fb_ss},{**iters_hpw_ss,**iters_hpw_fb_ss},tol)
    ax_conv_hpw_ss.set_title('scaled')
    ax_conv_hpw_ss.set_ylabel('Error ($||D_F F(x^k)||_2$)')
    if plot_sol:
        # plot error w.r.t solution for pressure
        fig_pres_err_hpw, (ax_pres_err_hpw, ax_pres_err_hpw_pu, ax_pres_err_hpw_ss) = plt.subplots(1, 3, sharey=True)
        fig_pres_err_hpw.canvas.set_window_title('Pressure error high pres. Weymouth, with p = {:.2e} [Pa] = {:.2e} [bar]'.format(p_sol_hpw,p_sol_hpw/bar))
        x_hpw_conv = dict() # not all the unscaled methods have converged. Only plot errors for the ones that did.
        for key, x in {**x_hpw,**x_hpw_fb}.items():
            if {**errors_hpw,**errors_hpw_fb}.get(key)[-1] < tol:
                x_hpw_conv[key] = x
        plot_pres_sol_error(ax_pres_err_hpw,p_sol_hpw,x_hpw_conv)
        ax_pres_err_hpw.set_title('unscaled')
        x_hpw_pu_SI = dict()
        for key, x in {**x_hpw_pu,**x_hpw_fb_pu}.items():
            if 'nodal' in key:
                x_hpw_pu_SI[key] = x*scale_params_hpw['pbase']
            else:
                x_hpw_pu_SI[key] = x*np.array([scale_params_hpw['qbase'],scale_params_hpw['pbase']])
        plot_pres_sol_error(ax_pres_err_hpw_pu,p_sol_hpw,x_hpw_pu_SI)
        ax_pres_err_hpw_pu.set_title('per unit scaling')
        plot_pres_sol_error(ax_pres_err_hpw_ss,p_sol_hpw,{**x_hpw_ss,**x_hpw_fb_ss})
        ax_pres_err_hpw_ss.set_title('scaled')
        # plot error w.r.t solution for flow
        fig_flow_err_hpw, (ax_flow_err_hpw, ax_flow_err_hpw_pu, ax_flow_err_hpw_ss) = plt.subplots(1, 3, sharey=True)
        fig_flow_err_hpw.canvas.set_window_title('Flow error high pres. Weymouth, with q = {:.2e} [kg/s]'.format(q_sol_hpw))
        plot_flow_sol_error(ax_flow_err_hpw,q_sol_hpw,x_hpw_conv)
        ax_flow_err_hpw.set_title('unscaled')
        plot_flow_sol_error(ax_flow_err_hpw_pu,q_sol_hpw,x_hpw_pu_SI)
        ax_flow_err_hpw_pu.set_title('per unit scaling')
        plot_flow_sol_error(ax_flow_err_hpw_ss,q_sol_hpw,{**x_hpw_ss,**x_hpw_fb_ss})
        ax_flow_err_hpw_ss.set_title('scaled')

def high_pres_blasius(solvers_unscaled,solvers_pu,solvers_scaled,tol=1e-12,plot_sol=True,plot_top=False):
    """Solve the network with a high pressure pipe using Weymouth friction factor.
    
    Parameters
    ----------
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
    print('\nHigh pressure pipe with Blasius friction factor')
    # create network and solution
    gas_net_hpb, gas_net_hpb_fb, q_sol_hpb, p_sol_hpb = create_network_hpb()
    print('solution for Blasius: m = {:.3e} [kg/s], p = {:.3e} [Pa]'.format(q_sol_hpb,p_sol_hpb))
    # initialize network
    p_perc_hpb = 0.8
    q_perc_hpb = 0.5
    x0_hpb = initialize_network(gas_net_hpb,p_perc=p_perc_hpb,q_perc=q_perc_hpb)
    x0_hpb_fb = initialize_network(gas_net_hpb_fb,p_perc=p_perc_hpb,q_perc=q_perc_hpb)
    scale_params_hpb = {'qbase':10,'pbase':10*bar}
    # Solve networks in different ways, compare convergence and solutions
    max_iter_hpb = 8
    print('\nsolving system with high pressure pipe with Blasius friction factor')
    # unscaled
    x_hpb, iters_hpb, errors_hpb = solve_network(gas_net_hpb,x0_hpb,tol,max_iter_hpb,solvers_unscaled,solve_both_forms=False,label=' fa')
    x_hpb_fb, iters_hpb_fb, errors_hpb_fb = solve_network(gas_net_hpb_fb,x0_hpb_fb,tol,max_iter_hpb,solvers_unscaled,solve_both_forms=False,label=' fb',plot_top=plot_top)
    # per unit 
    x_hpb_pu, iters_hpb_pu, errors_hpb_pu = solve_network(gas_net_hpb,x0_hpb,tol,max_iter_hpb,solvers_pu,solve_both_forms=False,label=' fa',scaling='per_unit',scale_params=scale_params_hpb)
    x_hpb_fb_pu, iters_hpb_fb_pu, errors_hpb_fb_pu = solve_network(gas_net_hpb_fb,x0_hpb_fb,tol,max_iter_hpb,solvers_pu,solve_both_forms=False,label=' fb',scaling='per_unit',scale_params=scale_params_hpb)
    # scaling in solver (in this case using the same base values as for the per unit scaling.)
    x_hpb_ss, iters_hpb_ss, errors_hpb_ss = solve_network(gas_net_hpb,x0_hpb,tol,max_iter_hpb,solvers_scaled,solve_both_forms=False,label=' fa',scaling='matrix',scale_params=scale_params_hpb)
    x_hpb_fb_ss, iters_hpb_fb_ss, errors_hpb_fb_ss = solve_network(gas_net_hpb_fb,x0_hpb_fb,tol,max_iter_hpb,solvers_scaled,solve_both_forms=False,label=' fb',scaling='matrix',scale_params=scale_params_hpb)
    # plot convergence
    fig_conv_hpb, (ax_conv_hpb, ax_conv_hpb_pu, ax_conv_hpb_ss) = plt.subplots(3, 1, sharex=True)
    fig_conv_hpb.canvas.set_window_title('Convergence high pres. Blasius')
    plot_convergence(ax_conv_hpb,{**errors_hpb,**errors_hpb_fb},{**iters_hpb,**iters_hpb_fb},tol)
    ax_conv_hpb.set_title('unscaled')
    ax_conv_hpb.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_hpb.set_xlabel('')
    plot_convergence(ax_conv_hpb_pu,{**errors_hpb_pu,**errors_hpb_fb_pu},{**iters_hpb_pu,**iters_hpb_fb_pu},tol)
    ax_conv_hpb_pu.set_title('per unit scaling')
    ax_conv_hpb_pu.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_hpb_pu.set_xlabel('')
    plot_convergence(ax_conv_hpb_ss,{**errors_hpb_ss,**errors_hpb_fb_ss},{**iters_hpb_ss,**iters_hpb_fb_ss},tol)
    ax_conv_hpb_ss.set_title('scaled')
    ax_conv_hpb_ss.set_ylabel('Error ($||D_F F(x^k)||_2$)')
    if plot_sol:
        # plot error w.r.t solution for pressure
        fig_pres_err_hpb, (ax_pres_err_hpb, ax_pres_err_hpb_pu, ax_pres_err_hpb_ss) = plt.subplots(1, 3, sharey=True)
        fig_pres_err_hpb.canvas.set_window_title('Pressure error high pres. Blasius, with p = {:.2e} [Pa] = {:.2e} [bar]'.format(p_sol_hpb,p_sol_hpb/bar))
        x_hpb_conv = dict() # not all the unscaled methods have converged. Only plot errors for the ones that did.
        for key, x in {**x_hpb,**x_hpb_fb}.items():
            if {**errors_hpb,**errors_hpb_fb}.get(key)[-1] < tol:
                x_hpb_conv[key] = x
        plot_pres_sol_error(ax_pres_err_hpb,p_sol_hpb,x_hpb_conv)
        ax_pres_err_hpb.set_title('unscaled')
        x_hpb_pu_SI = dict()
        for key, x in {**x_hpb_pu,**x_hpb_fb_pu}.items():
            if 'nodal' in key:
                x_hpb_pu_SI[key] = x*scale_params_hpb['pbase']
            else:
                x_hpb_pu_SI[key] = x*np.array([scale_params_hpb['qbase'],scale_params_hpb['pbase']])
        plot_pres_sol_error(ax_pres_err_hpb_pu,p_sol_hpb,x_hpb_pu_SI)
        ax_pres_err_hpb_pu.set_title('per unit scaling')
        plot_pres_sol_error(ax_pres_err_hpb_ss,p_sol_hpb,{**x_hpb_ss,**x_hpb_fb_ss})
        ax_pres_err_hpb_ss.set_title('scaled')
        # plot error w.r.t solution for flow
        fig_flow_err_hpb, (ax_flow_err_hpb, ax_flow_err_hpb_pu, ax_flow_err_hpb_ss) = plt.subplots(1, 3, sharey=True)
        fig_flow_err_hpb.canvas.set_window_title('Flow error high pres. Blasius, with q = {:.2e} [kg/s]'.format(q_sol_hpb))
        plot_flow_sol_error(ax_flow_err_hpb,q_sol_hpb,x_hpb_conv)
        ax_flow_err_hpb.set_title('unscaled')
        plot_flow_sol_error(ax_flow_err_hpb_pu,q_sol_hpb,x_hpb_pu_SI)
        ax_flow_err_hpb_pu.set_title('per unit scaling')
        plot_flow_sol_error(ax_flow_err_hpb_ss,q_sol_hpb,{**x_hpb_ss,**x_hpb_fb_ss})
        ax_flow_err_hpb_ss.set_title('scaled')
        
def high_pres_pole(solvers_unscaled,solvers_pu,solvers_scaled,tol=1e-12,plot_sol=True,plot_top=False):
    """Solve the network with a high pressure pipe using Pole friction factor. Should give the same solution as the low pressure pipe with the Pole friction factor.
    
    Parameters
    ----------
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
    print('\nHigh pressure pipe with Pole friction factor')
    # create network and solution
    gas_net_hpp, gas_net_hpp_fb, q_sol_hpp, p_sol_hpp = create_network_hpp()
    print('solution for high pressure Pole: m = {:.3e} [kg/s], gauge p = {:.3e} [Pa]'.format(q_sol_hpp,p_sol_hpp-atm))
    # initialize network
    p_perc_hpp = 0.999999
    q_perc_hpp = 0.5
    x0_hpp = initialize_network(gas_net_hpp,p_perc=p_perc_hpp,q_perc=q_perc_hpp)
    x0_hpp_fb = initialize_network(gas_net_hpp_fb,p_perc=p_perc_hpp,q_perc=q_perc_hpp)
    scale_params_hpp = {'qbase':1e-5,'pbase':1*bar}
    # Solve networks in different ways, compare convergence and solutions
    max_iter_hpp = 10
    print('\nsolving system with high pressure pipe with Pole friction factor')
    # unscaled
    x_hpp, iters_hpp, errors_hpp = solve_network(gas_net_hpp,x0_hpp,tol,max_iter_hpp,solvers_unscaled,label=' fa')
    x_hpp_fb, iters_hpp_fb, errors_hpp_fb = solve_network(gas_net_hpp_fb,x0_hpp_fb,tol,max_iter_hpp,solvers_unscaled,solve_both_forms=False,label=' fb',plot_top=plot_top)
    # per unit 
    x_hpp_pu, iters_hpp_pu, errors_hpp_pu = solve_network(gas_net_hpp,x0_hpp,tol,max_iter_hpp,solvers_pu,label=' fa',scaling='per_unit',scale_params=scale_params_hpp)
    x_hpp_fb_pu, iters_hpp_fb_pu, errors_hpp_fb_pu = solve_network(gas_net_hpp_fb,x0_hpp_fb,tol,max_iter_hpp,solvers_pu,solve_both_forms=False,label=' fb',scaling='per_unit',scale_params=scale_params_hpp)
    # scaling in solver (in this case using the same base values as for the per unit scaling.)
    x_hpp_ss, iters_hpp_ss, errors_hpp_ss = solve_network(gas_net_hpp,x0_hpp,tol,max_iter_hpp,solvers_scaled,label=' fa',scaling='matrix',scale_params=scale_params_hpp)
    x_hpp_fb_ss, iters_hpp_fb_ss, errors_hpp_fb_ss = solve_network(gas_net_hpp_fb,x0_hpp_fb,tol,max_iter_hpp,solvers_scaled,solve_both_forms=False,label=' fb',scaling='matrix',scale_params=scale_params_hpp)
    # plot convergence
    fig_conv_hpp, (ax_conv_hpp, ax_conv_hpp_pu, ax_conv_hpp_ss) = plt.subplots(3, 1, sharex=True)
    fig_conv_hpp.canvas.set_window_title('Convergence high pres. Pole')
    plot_convergence(ax_conv_hpp,{**errors_hpp,**errors_hpp_fb},{**iters_hpp,**iters_hpp_fb},tol)
    ax_conv_hpp.set_title('unscaled')
    ax_conv_hpp.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_hpp.set_xlabel('')
    plot_convergence(ax_conv_hpp_pu,{**errors_hpp_pu,**errors_hpp_fb_pu},{**iters_hpp_pu,**iters_hpp_fb_pu},tol)
    ax_conv_hpp_pu.set_title('per unit scaling')
    ax_conv_hpp_pu.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_hpp_pu.set_xlabel('')
    plot_convergence(ax_conv_hpp_ss,{**errors_hpp_ss,**errors_hpp_fb_ss},{**iters_hpp_ss,**iters_hpp_fb_ss},tol)
    ax_conv_hpp_ss.set_title('scaled')
    ax_conv_hpp_ss.set_ylabel('Error ($||D_F F(x^k)||_2$)')
    if plot_sol:
        # plot error w.r.t solution for pressure
        fig_pres_err_hpp, (ax_pres_err_hpp, ax_pres_err_hpp_pu, ax_pres_err_hpp_ss) = plt.subplots(1, 3, sharey=True)
        fig_pres_err_hpp.canvas.set_window_title('Pressure error high pres. Pole, with p = {:.2e} [Pa] = {:.2e} [bar]'.format(p_sol_hpp,p_sol_hpp/bar))
        x_hpp_conv = dict() # not all the unscaled methods have converged. Only plot errors for the ones that did.
        for key, x in {**x_hpp,**x_hpp_fb}.items():
            if {**errors_hpp,**errors_hpp_fb}.get(key)[-1] < tol:
                x_hpp_conv[key] = x
        plot_pres_sol_error(ax_pres_err_hpp,p_sol_hpp,x_hpp_conv)
        ax_pres_err_hpp.set_title('unscaled')
        x_hpp_pu_SI = dict()
        for key, x in {**x_hpp_pu,**x_hpp_fb_pu}.items():
            if 'nodal' in key:
                x_hpp_pu_SI[key] = x*scale_params_hpp['pbase']
            else:
                x_hpp_pu_SI[key] = x*np.array([scale_params_hpp['qbase'],scale_params_hpp['pbase']])
        plot_pres_sol_error(ax_pres_err_hpp_pu,p_sol_hpp,x_hpp_pu_SI)
        ax_pres_err_hpp_pu.set_title('per unit scaling')
        plot_pres_sol_error(ax_pres_err_hpp_ss,p_sol_hpp,{**x_hpp_ss,**x_hpp_fb_ss})
        ax_pres_err_hpp_ss.set_title('scaled')
        # plot error w.r.t solution for flow
        fig_flow_err_hpp, (ax_flow_err_hpp, ax_flow_err_hpp_pu, ax_flow_err_hpp_ss) = plt.subplots(1, 3, sharey=True)
        fig_flow_err_hpp.canvas.set_window_title('Flow error high pres. Pole, with q = {:.2e} [kg/s]'.format(q_sol_hpp))
        plot_flow_sol_error(ax_flow_err_hpp,q_sol_hpp,x_hpp_conv)
        ax_flow_err_hpp.set_title('unscaled')
        plot_flow_sol_error(ax_flow_err_hpp_pu,q_sol_hpp,x_hpp_pu_SI)
        ax_flow_err_hpp_pu.set_title('per unit scaling')
        plot_flow_sol_error(ax_flow_err_hpp_ss,q_sol_hpp,{**x_hpp_ss,**x_hpp_fb_ss})
        ax_flow_err_hpp_ss.set_title('scaled')

def resistor_link(solvers_unscaled,solvers_pu,solvers_scaled,tol=1e-12,plot_sol=True,plot_top=False):
    """Solve the network with a resistor instead of a pipe.
    
    Parameters
    ----------
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
    print('\nResistor')
    # create network and solution
    gas_net_res, gas_net_res_fb, q_sol_res, p_sol_res = create_network_resistor()
    print('solution for resistor: m = {:.3e} [kg/s], p = {:.3e} [Pa]'.format(q_sol_res,p_sol_res))
    # initialize network
    p_perc_res = 0.8
    q_perc_res = 0.5
    x0_res = initialize_network(gas_net_res,p_perc=p_perc_res,q_perc=q_perc_res)
    x0_res_fb = initialize_network(gas_net_res_fb,p_perc=p_perc_res,q_perc=q_perc_res)
    scale_params_res = {'qbase':10,'pbase':10*bar}
    # Solve networks in different ways, compare convergence and solutions
    max_iter_res = 10
    print('\nsolving system with resistor')
    # unscaled
    x_res, iters_res, errors_res = solve_network(gas_net_res,x0_res,tol,max_iter_res,solvers_unscaled,label=' fa')
    x_res_fb, iters_res_fb, errors_res_fb = solve_network(gas_net_res_fb,x0_res_fb,tol,max_iter_res,solvers_unscaled,solve_both_forms=False,label=' fb',plot_top=plot_top)
    # per unit 
    x_res_pu, iters_res_pu, errors_res_pu = solve_network(gas_net_res,x0_res,tol,max_iter_res,solvers_pu,label=' fa',scaling='per_unit',scale_params=scale_params_res)
    x_res_fb_pu, iters_res_fb_pu, errors_res_fb_pu = solve_network(gas_net_res_fb,x0_res_fb,tol,max_iter_res,solvers_pu,solve_both_forms=False,label=' fb',scaling='per_unit',scale_params=scale_params_res)
    # scaling in solver (in this case using the same base values as for the per unit scaling.)
    x_res_ss, iters_res_ss, errors_res_ss = solve_network(gas_net_res,x0_res,tol,max_iter_res,solvers_scaled,label=' fa',scaling='matrix',scale_params=scale_params_res)
    x_res_fb_ss, iters_res_fb_ss, errors_res_fb_ss = solve_network(gas_net_res_fb,x0_res_fb,tol,max_iter_res,solvers_scaled,solve_both_forms=False,label=' fb',scaling='matrix',scale_params=scale_params_res)
    # plot convergence
    fig_conv_res, (ax_conv_res, ax_conv_res_pu, ax_conv_res_ss) = plt.subplots(3, 1, sharex=True)
    fig_conv_res.canvas.set_window_title('Convergence resistor')
    plot_convergence(ax_conv_res,{**errors_res,**errors_res_fb},{**iters_res,**iters_res_fb},tol)
    ax_conv_res.set_title('unscaled')
    ax_conv_res.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_res.set_xlabel('')
    plot_convergence(ax_conv_res_pu,{**errors_res_pu,**errors_res_fb_pu},{**iters_res_pu,**iters_res_fb_pu},tol)
    ax_conv_res_pu.set_title('per unit scaling')
    ax_conv_res_pu.set_ylabel('Error ($||F(x^k)||_2$)')
    ax_conv_res_pu.set_xlabel('')
    plot_convergence(ax_conv_res_ss,{**errors_res_ss,**errors_res_fb_ss},{**iters_res_ss,**iters_res_fb_ss},tol)
    ax_conv_res_ss.set_title('scaled')
    ax_conv_res_ss.set_ylabel('Error ($||D_F F(x^k)||_2$)')
    if plot_sol:
        # plot error w.r.t solution for pressure
        fig_pres_err_res, (ax_pres_err_res, ax_pres_err_res_pu, ax_pres_err_res_ss) = plt.subplots(1, 3, sharey=True)
        fig_pres_err_res.canvas.set_window_title('Pressure error resistor, with (gauge?) p = {:.2e} [Pa] = {:.2e} [bar]'.format(p_sol_res,p_sol_res/bar))
        plot_pres_sol_error(ax_pres_err_res,p_sol_res,{**x_res,**x_res_fb})
        ax_pres_err_res.set_title('unscaled')
        x_res_pu_SI = dict()
        for key, x in {**x_res_pu,**x_res_fb_pu}.items():
            if 'nodal' in key:
                x_res_pu_SI[key] = x*scale_params_res['pbase']
            else:
                x_res_pu_SI[key] = x*np.array([scale_params_res['qbase'],scale_params_res['pbase']])
        plot_pres_sol_error(ax_pres_err_res_pu,p_sol_res,x_res_pu_SI)
        ax_pres_err_res_pu.set_title('per unit scaling')
        plot_pres_sol_error(ax_pres_err_res_ss,p_sol_res,{**x_res_ss,**x_res_fb_ss})
        ax_pres_err_res_ss.set_title('scaled')
        # plot error w.r.t solution for flow
        fig_flow_err_res, (ax_flow_err_res, ax_flow_err_res_pu, ax_flow_err_res_ss) = plt.subplots(1, 3, sharey=True)
        fig_flow_err_res.canvas.set_window_title('Flow error resistor, with q = {:.2e} [kg/s]'.format(q_sol_res))
        plot_flow_sol_error(ax_flow_err_res,q_sol_res,{**x_res,**x_res_fb})
        ax_flow_err_res.set_title('unscaled')
        plot_flow_sol_error(ax_flow_err_res_pu,q_sol_res,x_res_pu_SI)
        ax_flow_err_res_pu.set_title('per unit scaling')
        plot_flow_sol_error(ax_flow_err_res_ss,q_sol_res,{**x_res_ss,**x_res_fb_ss})
        ax_flow_err_res_ss.set_title('scaled')
        
def main(examples,solvers_unscaled,solvers_pu,solvers_scaled,tol=1e-12,plot_sol=False,plot_top=False):
    """Solve a network with only one link for different link models. Plot convergence for different solvers. Optionally, also plot difference between calculated solution and actual solution.
    
    Parameters
    ----------
    examples : list
        Determines which networks are solved. Options are 'lpp', 'lphp', 'lppw', 'hpw', 'hpb', 'hpp', 'res'.
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
    plot_top : bool, optional
        If True, plot topology of the network. Default is False
    """
    if 'lpp' in examples:
        low_pres_pole(solvers_unscaled,solvers_pu,solvers_scaled,tol=tol,plot_sol=plot_sol,plot_top=plot_top)
    if 'lphp' in examples:
        low_pres_hagen_poiseuille(solvers_unscaled,solvers_pu,solvers_scaled,tol=tol,plot_sol=plot_sol,plot_top=plot_top)
    if 'lppw' in examples:
        low_pres_pole_water(solvers_unscaled,solvers_pu,solvers_scaled,tol=tol,plot_sol=plot_sol,plot_top=plot_top)
    if 'hpw' in examples:
        high_pres_weymouth(solvers_unscaled,solvers_pu,solvers_scaled,tol=tol,plot_sol=plot_sol,plot_top=plot_top)
    if 'hpb' in examples:
        high_pres_blasius(solvers_unscaled,solvers_pu,solvers_scaled,tol=tol,plot_sol=plot_sol,plot_top=plot_top)
    if 'hpp' in examples:
        high_pres_pole(solvers_unscaled,solvers_pu,solvers_scaled,tol=tol,plot_sol=plot_sol,plot_top=plot_top)
    if 'res' in examples:
        resistor_link(solvers_unscaled,solvers_pu,solvers_scaled,tol=tol,plot_sol=plot_sol,plot_top=plot_top)
    
if __name__ == '__main__':
    # parse the command line
    args = command_line_input.parse_args()
    
    if args.plot_sol or args.plot_top:
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
            "font.sans-serif": [],              # to inherit fonts from the document
            "font.monospace": [],
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.fontsize": 9,               # Make the legend/label fonts 
            "xtick.labelsize": 10,               # a little smaller
            "ytick.labelsize": 10,
            "figure.figsize": fig_size,     
            }
        mpl.rcParams.update(pgf_with_latex)
            
    main(args.ex,args.solv_un,args.solv_pu,args.solv_sc,tol=args.tol,plot_sol=args.plot_sol,plot_top=args.plot_top)
    
    plt.show()
