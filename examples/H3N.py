"""Example of a heat network with 3 nodes, based on the example in Shabanpour-Haghighi & Seifi.
"""
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
from meslf.utils.constants import mm, MW
from meslf.load_flow.system_of_equations import NonLinearSystemHeat
import numpy as np
import scipy.sparse as sps
import scipy.optimize as spo
import pytest
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import warnings
from meslf.utils.hide_print import HiddenPrints
import os
import sys
import time
import ipopt

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

colors_method = {'trust-constr':'tab:blue','SLSQP':'tab:orange','ipopt':'tab:green'}
markers_bounds = {'soft':'s','hard':'*'}
linestyles_derivatives = {'num':'--','an':'-','direct':':','adjoint':'-.'}
linestyles_contraints = {'eq':'-','ineq':'--','bound':':'}
marker_size = 10
legend_handles = [Line2D([0], [0], color=colors_method.get('trust-constr'), label='trust-constr'),
    Line2D([0], [0], color=colors_method.get('SLSQP'), label='SLSQP'),
    Line2D([0], [0], color=colors_method.get('ipopt'), label='ipopt'),
    Line2D([0], [0], marker=markers_bounds.get('soft'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Soft constraints'),
    Line2D([0], [0], marker=markers_bounds.get('hard'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Hard constraints'),
    Line2D([0], [0], color='k',ls=linestyles_derivatives.get('num'), label='Numerical derivatives'),
    Line2D([0], [0], color='k',ls=linestyles_derivatives.get('an'), label='Analytical derivatives'),
    Line2D([0], [0], color='k',ls=linestyles_derivatives.get('direct'), label='Direct approach'),
    Line2D([0], [0], color='k',ls=linestyles_derivatives.get('adjoint'), label='Adjoint approach')]

def create_network():
    """Create a test heat network

    Returns
    -------
    heat_net : HeatNetwork
        The test network
    water : Carrier
        The carrier in the network
    """
    # carrier
    rho = 960. #[kg/m^3]
    Cp = 4.182e3 #[J/(kg K)]
    mu = 0.294e-6 #[m^2/s]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp,rho=rho,mu=mu)

    # nodes
    Ta = 10. #[C]
    heat_net = HeatNetwork('test heat network',Ta=Ta)
    hn0 = HeatNode('hn0',node_type=0,x=1,y=1,Ts=120.,p=5517*rho*g) # source slack node
    hn0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn1 = HeatNode('hn1',node_type=1,x=0,y=0,Tr_hl=50.,dphi=35*MW) # sink node
    hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn2 = HeatNode('hn2',node_type=1,x=2,y=0,Ts_hl=135.3422,dphi=-9*MW) # source node
    hn2.half_links[0].set_type('heat_exchanger',{'carrier':water})

    # links
    L = 30000. #[m]
    D = 150.*mm #[m]
    eps = 1.25*mm #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D) #[W/(m^2 K)]

    link_type = 'standard_pipe_low_pres_colebrook'
    link_params = {'L':L,'D':D,'eps':eps,'U':U,'carrier':water}
    hl0 = HeatLink('hl0',hn0,hn1,link_type=link_type,link_params=link_params)
    hl1 = HeatLink('hl1',hn0,hn2,link_type=link_type,link_params=link_params.copy())
    hl2 = HeatLink('hl2',hn1,hn2,link_type=link_type,link_params=link_params.copy())

    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    return heat_net, water

def update_bc(heat_net,To2,phi2,scale_var=None,scale_var_params=None):
    """Updates the boundary conditions of the heat network, based on the state variables of the OPF"""
    if scale_var == 'per_unit':
        To2 = To2*scale_var_params.get('Tbase')
        phi2 = phi2*scale_var_params.get('phibase')
    heat_net.nodes[2].half_links[0].Ts = To2
    heat_net.nodes[2].half_links[0].dphi = phi2 #<0
    return heat_net

def initialize_network(network,carrier,scale_var=None,scale_var_params=None,formulation='standard'):
    """Sets values of network variables to be used for initial guess.

    Parameters
    ----------
    network : HeatlNetwork
        The network to be initialized
    carrier : Water
        The carrier in the network

    Returns
    -------
    x0 : np array
        initial guess
    """
    Ta = network.Ta
    rho = carrier.rhon
    g = carrier.g
    m_init = np.array([60, 30, -50])
    p_init = np.array([10,4000])*rho*g
    Ts_init = np.array([100,100])
    Tr_init = np.array([50.,50.,50.])
    if formulation == 'half_link_flow':
        m_hl_init = np.array([10,-10])
        if network.nodes[2].node_type == 12:
            Ts_hl_init = np.array([100])
        else:
            Ts_hl_init = np.array([])
        if network.nodes[1].node_type == 12:
            Tr_hl_init = np.array([50])
        else:
            Tr_hl_init = np.array([])
        x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init,Ts_hl_init,Tr_hl_init))

    else:
        x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    network.initialize()
    network.update(x_init,formulation=formulation)  # update without scaling, since x_init is unscaled
    x0 = network.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def run_load_flow(scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,formulation='standard'):
    # creat network
    heat_net, water = create_network()

    # initialize
    x0 = initialize_network(heat_net, water,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)

    # solve network
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,formulation=formulation,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution after {} iterations (final error = {:.4e}):'.format(iters,err_vec[-1]))
    rho_w = water.rhon
    grav_const = water.g
    print('p heat = {} m'.format(p_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {}'.format(Ts_hl_vec))
    print('Tr hl = {}'.format(Tr_hl_vec))
    print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
    return heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def example_h3n():
    """Check solution of the example network"""
    # Given
    tol = 1e-6
    max_iter = 10
    formulation='standard'

    # When
    heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = run_load_flow(tol=tol,max_iter=max_iter,formulation=formulation)

    # Then
    water = heat_net.links[0].link_params.get('carrier')
    rho = water.rhon
    g = water.g
    m_sol_expected = np.array([64.5075, 31.4083, -56.3316])
    p_sol_expected = np.array([254.3706, 4268.1087])*rho*g
    Ts_sol_expected = np.array([119.2591, 124.0494])
    Tr_sol_expected = np.array([48.5087, 50., 48.9941])
    x_sol_expected = np.concatenate((m_sol_expected,p_sol_expected,Ts_sol_expected,Tr_sol_expected))

    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def example_h3n_pu():
    """Check solution of the example network, using per unit scaling"""
    # Given
    tol = 1e-6
    max_iter = 10
    formulation='standard'
    #scaling
    scale_var = 'per_unit'
    _, water = create_network()
    rho = water.rhon
    g = water.g
    phibase = 1*MW #[W]
    Tbase = 100. #[C]
    mbase = 1.
    pbase = 5517*rho*g
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}

    # When
    heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = run_load_flow(tol=tol,max_iter=max_iter,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)

    # Then
    m_sol_expected = np.array([64.5075, 31.4083, -56.3316])/mbase
    p_sol_expected = np.array([254.3706, 4268.1087])*rho*g/pbase
    Ts_sol_expected = np.array([119.2591, 124.0494])/Tbase
    Tr_sol_expected = np.array([48.5087, 50., 48.9941])/Tbase
    x_sol_expected = np.concatenate((m_sol_expected,p_sol_expected,Ts_sol_expected,Tr_sol_expected))

    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def example_h3n_scaled_solver():
    """Check solution of the example network, using scaling in solver"""
    # Given
    tol = 1e-6
    max_iter = 10
    formulation='standard'
    #scaling
    scale_var = 'matrix'
    _, water = create_network()
    rho = water.rhon
    g = water.g
    phibase = 1*MW #[W]
    Tbase = 100. #[C]
    mbase = 1.
    pbase = 5517*rho*g
    scale_var_params = {'mbase':mbase,'phbase':pbase,'phibase':phibase,'Tbase':Tbase}

    # When
    heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = run_load_flow(tol=tol,max_iter=max_iter,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)

    # Then
    m_sol_expected = np.array([64.5075, 31.4083, -56.3316])
    p_sol_expected = np.array([254.3706, 4268.1087])*rho*g
    Ts_sol_expected = np.array([119.2591, 124.0494])
    Tr_sol_expected = np.array([48.5087, 50., 48.9941])
    x_sol_expected = np.concatenate((m_sol_expected,p_sol_expected,Ts_sol_expected,Tr_sol_expected))

    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def example_h3n_general_slack_node():
    """Check solution of the example network"""
    # Given
    heat_net, water = create_network()
    heat_net.nodes[0].node_type = 10 #general slack node instead of source slack node. Same variables are assumed known / unknown
    hl_params = heat_net.nodes[0].half_links[0].link_params
    heat_net.nodes[0].half_links[0].set_type('heat_exchanger',hl_params)
    x0 = initialize_network(heat_net, water)

    # When
    tol = 1e-6
    max_iter = 10
    formulation='standard'
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,formulation=formulation,solver='NR')

    # Then
    rho = water.rhon
    g = water.g
    m_sol_expected = np.array([64.5075, 31.4083, -56.3316])
    p_sol_expected = np.array([254.3706, 4268.1087])*rho*g
    Ts_sol_expected = np.array([119.2591, 124.0494])
    Tr_sol_expected = np.array([48.5087, 50., 48.9941])
    x_sol_expected = np.concatenate((m_sol_expected,p_sol_expected,Ts_sol_expected,Tr_sol_expected))

    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def example_h3n_temp_diff_half_link_flow():
    """Check solution of the example network, using the unknown half link flow formulation."""
    # Given
    heat_net, water = create_network()
    rho = water.rhon
    g = water.g
    hn1 = heat_net.nodes[1]
    hn2 = heat_net.nodes[2]
    Ts1  = 119.2591
    To1 = 50.
    dT1 = Ts1 - To1
    hn1.node_type=12
    hn1.half_links[0].dT = dT1
    hn1.half_links[0].bc_type = 5 #dphi and dT known, sink
    Tr2 = 48.9941
    To2 = 135.3422
    dT2 = To2 - Tr2
    hn2.node_type=12
    hn2.half_links[0].dT = dT2
    hn2.half_links[0].bc_type = 4 #dphi and dT known, source

    # When
    tol = 1e-6
    max_iter = 10
    formulation = 'half_link_flow'
    x0 = initialize_network(heat_net, water, formulation=formulation)
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,formulation=formulation,solver='NR')
    x_sol_full = np.concatenate([x_sol[0:5],x_sol[7:]]) # solution without pressure, with outflow temperatures of half links

    # Then (check everything expect for pressure, since pressure has a different order of magnitude)
    m_sol_expected = np.array([64.5075, 31.4083, -56.3316])
    m_hl_sol_expected = np.array([120.8391,-24.9236])
    Ts_sol_expected = np.array([119.2591, 124.0494])
    Tr_sol_expected = np.array([48.5087, 50., 48.9941])
    To_hl_sol_expected = np.array([To2,To1]) #sources first, then sinks
    x_sol_expected = np.concatenate((m_sol_expected,m_hl_sol_expected,Ts_sol_expected,Tr_sol_expected,To_hl_sol_expected))

    rel_tol = 1e-3
    print('x_sol = \n{}'.format(x_sol_full))
    print('x_expected = \n{}'.format(x_sol_expected))
    assert np.allclose(x_sol_full,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def compare_scaling_formulation():
    """Compare the convergence of NR for different ways of scaling the system of equations and variables, and using different formulations."""
    #scaling
    _, water = create_network()
    rho = water.rhon
    g = water.g
    phibase = 1*MW #[W]
    Tbase = 100.#[C]
    mbase = 1.
    pbase = 5517*rho*g
    scale_var_params_pu = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}
    scale_var_params_matrix = {'mbase':mbase,'phbase':pbase,'phibase':phibase,'Tbase':Tbase}

    # compare convergence
    tol = 1e-6
    max_iter = 500
    # solve when everything is specified in S.I., unscaled
    heat_net,x_sol_SI,iters_SI,err_vec_SI,_,_,_,_,_,_,_,_ = run_load_flow(scale_var=None,scale_var_params=None,tol=tol,max_iter=max_iter,formulation='standard')
    # solve when everything is specified in per unit
    _,x_sol_pu,iters_pu,err_vec_pu,_,_,_,_,_,_,_,_ = run_load_flow(scale_var='per_unit',scale_var_params=scale_var_params_pu,tol=tol,max_iter=max_iter,formulation='standard')
    # solve when everything is specified in S.I., using scaling in solver
    _,x_sol_scaled,iters_scaled,err_vec_scaled,_,_,_,_,_,_,_,_ = run_load_flow(scale_var='matrix',scale_var_params=scale_var_params_matrix,tol=tol,max_iter=max_iter,formulation='standard')
    # using 'half_link_flow' formulation
    # solve when everything is specified in S.I., unscaled
    _,x_sol_SI_hl,iters_SI_hl,err_vec_SI_hl,_,_,_,_,_,_,_,_ = run_load_flow(scale_var=None,scale_var_params=None,tol=tol,max_iter=max_iter,formulation='half_link_flow')
    # solve when everything is specified in per unit
    _,x_sol_pu_hl,iters_pu_hl,err_vec_pu_hl,_,_,_,_,_,_,_,_ = run_load_flow(scale_var='per_unit',scale_var_params=scale_var_params_pu,tol=tol,max_iter=max_iter,formulation='half_link_flow')
    # solve when everything is specified in S.I., using scaling in solver
    _,x_sol_scaled_hl,iters_scaled_hl,err_vec_scaled_hl,_,_,_,_,_,_,_,_ = run_load_flow(scale_var='matrix',scale_var_params=scale_var_params_matrix,tol=tol,max_iter=max_iter,formulation='half_link_flow')

    fig = plt.figure('Convergence plot H3N')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    max_iter_used = np.max([iters_pu,iters_SI,iters_scaled,iters_pu_hl,iters_SI_hl,iters_scaled_hl])
    ax.semilogy(np.asarray(range(0,iters_pu+1)),err_vec_pu,'s-',label='p.u.')
    ax.semilogy(np.asarray(range(0,iters_SI+1)),err_vec_SI,'o-',label='S.I., unscaled')
    ax.semilogy(np.asarray(range(0,iters_scaled+1)),err_vec_scaled,'.-',label='S.I., scaled solver')
    ax.semilogy(np.asarray(range(0,iters_pu_hl+1)),err_vec_pu_hl,'s--',label='p.u., hl')
    ax.semilogy(np.asarray(range(0,iters_SI_hl+1)),err_vec_SI_hl,'o--',label='S.I., unscaled, hl')
    ax.semilogy(np.asarray(range(0,iters_scaled_hl+1)),err_vec_scaled_hl,'.--',label='S.I., scaled solver, hl')
    layout_convergence(ax,tol,max_iter_used)

    fig = plt.figure('Convergence order H3N')
    ax_ord = fig.gca()
    plt.xlabel(r'$||\Delta e^k||_2 / ||\Delta e^0||_2$')
    plt.ylabel(r'$||\Delta e^{k+1}||_2 / ||\Delta e^0||_2$')
    ax_ord.loglog(err_vec_pu[:-1]/err_vec_pu[0],err_vec_pu[1:]/err_vec_pu[0],'s-',label='p.u.')
    ax_ord.loglog(err_vec_SI[:-1]/err_vec_SI[0],err_vec_SI[1:]/err_vec_SI[0],'o-',label='S.I., unscaled')
    ax_ord.loglog(err_vec_scaled[:-1]/err_vec_scaled[0],err_vec_scaled[1:]/err_vec_scaled[0],'.-',label='S.I., scaled solver')
    ax_ord.loglog(err_vec_pu_hl[:-1]/err_vec_pu_hl[0],err_vec_pu_hl[1:]/err_vec_pu_hl[0],'s--',label='p.u., hl')
    ax_ord.loglog(err_vec_SI_hl[:-1]/err_vec_SI_hl[0],err_vec_SI_hl[1:]/err_vec_SI_hl[0],'o--',label='S.I., unscaled, hl')
    ax_ord.loglog(err_vec_scaled_hl[:-1]/err_vec_scaled_hl[0],err_vec_scaled_hl[1:]/err_vec_scaled_hl[0],'.--',label='S.I., scaled solver, hl')
    layout_convergence_order(ax_ord)

    # plot network
    fig_top = plt.figure('Network topology')
    ax_top = fig_top.gca()
    heat_net.draw_network(ax_top)
    plt.axis('equal')
    plt.axis('off')

def xh_from_xopf(x_opf):
    xh = x_opf[4:]
    return xh

def price_heat(dphi0,dphi2,a0=0,a2=0,b0=.04,b2=.05,c0=4e-4,c2=4.5e-4,scale_var=None,scale_var_params=None,Dy=None, Dy_inv=None, Df=None, Dh=None):
    """Determine the cost of the total heat input into the network

    Parameters
    ----------
    dphi0 : float
        Heat on source half link of node 0, in W. Since it is a source, it is assumed to be negative. Scaled when per unit scaling is unsed, unscaled otherwise.
    dphi2 : float
        Heat on source half link of node 2, in W. Since it is a source, it is assumed to be negative. Scaled when per unit scaling is unsed, unscaled otherwise.
    a0, a1 : float
        Parameter of price function. Scaled when per unit scaling is used, unscaled otherwise in euros.
    b0, b2 : float
        Parameter of price function. Scaled when per unit scaling is used, unscaled otherwise in euros/W.
    c0, c2 : float
        Parameter of price function. Scaled when per unit scaling is used, unscaled otherwise in euros/W^2.

    Returns
    -------
    f : float
        Total price of the two heat sources, in euros.
    """
    f = a0 + b0*-dphi0 + c0*dphi0**2  + a2 + b2*-dphi2 + c2*dphi2**2
    if scale_var == 'matrix':
        f = Df[0]*f
    return f

def jac_objective(y,dphi0_ind,dphi2_ind,a0=0,a2=0,b0=.04,b2=.05,c0=4e-4,c2=4.5e-4,scale_var=None,scale_var_params=None,Dy=None, Dy_inv=None, Df=None, Dh=None):
    """Gradient vector / Jacobian of objective function

    Parameters
    ----------
    y : np array
        Vector with unknowns of OF. Scaled when per unit scaling is unsed, unscaled otherwise.
    dphi0_ind, dphi2_ind : float
        Indices in y of the heat power of the source connected to node 0 and node 2
    a0, a1 : float
        Parameter of price function. Scaled when per unit scaling is used, unscaled otherwise in euros.
    b0, b2 : float
        Parameter of price function. Scaled when per unit scaling is used, unscaled otherwise in euros/W.
    c0, c2 : float
        Parameter of price function. Scaled when per unit scaling is used, unscaled otherwise in euros/W^2.

    Returns
    -----------
    df_dy : np.array
        Derivatives of the objective function. Scaled when per unit scaling or matrix scaling is used.
    """
    dphi0 = y[dphi0_ind] #<0
    dphi2 = y[dphi2_ind] #<0
    df_dy = np.zeros(len(y))
    df_dy[dphi0_ind] = -b0 + 2*c0*dphi0
    df_dy[dphi2_ind] = -b2 + 2*c2*dphi2
    if scale_var == 'matrix':
        df_dy = Df[0]*(df_dy.dot(Dy_inv))
    return df_dy

def hess_objective(y,dphi0_ind,dphi2_ind,a0=0,a2=0,b0=.04,b2=.05,c0=4e-4,c2=4.5e-4,scale_var=None,scale_var_params=None,Dy=None, Dy_inv=None, Df=None, Dh=None):
    """Hessian of objective function

    Parameters
    ----------
    y : np array
        Vector with unknowns of OF. Scaled when per unit scaling is unsed, unscaled otherwise.
    dphi0_ind, dphi2_ind : float
        Indices in y of the heat power of the source connected to node 0 and node 2
    a0, a1 : float
        Parameter of price function. Scaled when per unit scaling is used, unscaled otherwise in euros.
    b0, b2 : float
        Parameter of price function. Scaled when per unit scaling is used, unscaled otherwise in euros/W.
    c0, c2 : float
        Parameter of price function. Scaled when per unit scaling is used, unscaled otherwise in euros/W^2.

    Returns
    -----------
    hess : np.array
        Hessian of the objective function. Scaled when per unit scaling or matrix scaling is used.
    """
    hess_cost_diag = np.zeros(len(y))
    hess_cost_diag[dphi0_ind] = 2*c0
    hess_cost_diag[dphi2_ind] = 2*c2
    hess = np.diag(hess_cost_diag)
    if scale_var == 'matrix':
        hess = Df[0]*(np.transpose(Dy_inv).dot(hess.dot(Dy_inv)))
    return hess

def h(y,dphi0_ind,dphi2_ind,network=None,nlsys=None,scale_var=None,scale_var_params=None, Dy=None, Dy_inv=None, Df=None, Dh=None,formulation='standard'):
    """Equality constraints h(x)=0. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Variables. Scaled when per unit scaling is used, unscaled otherwise.
    network : ElectricalNetwork
        The electrical network that is being optimizes. Assumed to have the updated BC's.
    scale_var : str, optional
        Which scaling is used. Options are 'per_unit', 'matrix', or None. Default is None.

    Returns
    -------
    h : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    # update network and set correct values on half links etc.
    xh = xh_from_xopf(y)
    network.update(xh,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    # evaluate load flow equations
    network.reset_network(xh,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    F = nlsys.F(xh)
    # evaluate heat equation in slack node (use m0 determined by conservation of mass)
    m0, dphi0 = y[dphi2_ind+1:dphi0_ind+1] # scaled for p.u., unscaled for matrix
    if scale_var == 'per_unit':
        network.nodes[0].half_links[0].dphi = dphi0*scale_var_params.get('phibase')
    else:
        network.nodes[0].half_links[0].dphi = dphi0
    heat_power_eq = network.nodes[0].half_links[0].heat_power_equation(scale_var=scale_var,scale_var_params=scale_var_params)
    # evaluate conservation of mass in slack node
    if scale_var == 'per_unit':
        network.nodes[0].half_links[0].m = m0*scale_var_params.get('mbase')
    else:
        network.nodes[0].half_links[0].m = m0
    cons_mass = network.nodes[0].node_law(network=network,scale_var=scale_var,scale_var_params=scale_var_params)
    h = np.concatenate((np.array([cons_mass,heat_power_eq]),F)) # already scaled if per unit is used
    if scale_var == 'matrix':
        h = Dh.dot(h)
    return h

def h_der(y,dphi0_ind,dphi2_ind,network=None,nlsys=None,scale_var=None,scale_var_params=None, Dy=None, Dy_inv=None, Df=None, Dh=None,formulation='standard'):
    """First derivatives of equality constraints h(x)=0. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Variables. Scaled when per unit scaling is used, unscaled otherwise.
    network : ElectricalNetwork
        The electrical network that is being optimizes. Assumed to have the updated BC's.
    scale_var : str, optional
        Which scaling is used. Options are 'per_unit', 'matrix', or None. Default is None.

    Returns
    -------
    dh_dy : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    # determine indices
    N = len(network.nodes) # number of nodes in the network
    E = len(network.links) # number of links in the network
    T = len(network.half_links) # number of halflinks in the network.
    F_ind = nlsys.Fm + [N+ind for ind in nlsys.Fdeltap] + [N+E+ind for ind in nlsys.FTs] + [2*N+E+ind for ind in nlsys.FTr] + [3*N+E+ind for ind in nlsys.Fphi] + [3*N+E+T+ind for ind in nlsys.FdT]
    G_ind = [0,3*N+E]
    xlf_ind = nlsys.xm + [E + ind for ind in nlsys.xmhl] + [E+T + ind for ind in nlsys.xp] + [E+T+N + ind for ind in nlsys.xTs] + [E+T+2*N + ind for ind in nlsys.xTr] + [E+T+3*N + ind for ind in nlsys.xTshl] + [E+2*T+3*N + ind for ind in nlsys.xTrhl]
    Tshl2_ind = E+T+3*N+2
    mhl0_ind = E
    len_u = 2 #dphi2_ind+1
    len_slack = 2#dphi0_ind-dphi2_ind
    dh_dy = np.zeros((len(y)-len_u,len(y)))
    dh_dx = np.zeros((len(y)-len_u,len_slack+len(xlf_ind)))
    dh_du = np.zeros((len(y)-len_u,len_u))
    # evaluate Jacobian of LF equations
    xh = xh_from_xopf(y)
    network.reset_network(xh,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params) # resets all halflinks as well.
    J_full = nlsys.J_dense(xh,return_full=True) #This also updates the network
    dh_du[13,1] = -1 #dFphi2_dphi2
    dh_dx[:len_slack,:][:,len_slack:] = J_full[G_ind,:][:,xlf_ind]
    dh_du[:,0] = J_full[G_ind+F_ind,:][:,Tshl2_ind].ravel() #dH_dTs2,0
    dh_dx[len_slack:,:][:,len_slack:] = J_full[F_ind,:][:,xlf_ind] # LF part
    # determine derivatives of the slack equations
    water = network.links[0].link_params.get('carrier')
    dh_dx[0,0] = -1 #dG0_dm0
    dh_dx[0,len_slack+0] = -1 #dG0_dm01
    dh_dx[0,len_slack+1] = -1 #dG0_dm02
    dh_dx[1,1] = -1 #dG1_dphi0
    dT_source = network.nodes[0].half_links[0].get_Ts(scale_var=scale_var,scale_var_params=scale_var_params) - network.nodes[0].half_links[0].get_Tr(scale_var=scale_var,scale_var_params=scale_var_params)
    dh_dx[1,len_slack+0] = -water.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*(dT_source) #dG1_dm01
    dh_dx[1,len_slack+1] = -water.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*(dT_source) #dG1_dm02
    # fill in full Jacobian matrix
    dh_dy[:,:][:,:len_u] = dh_du
    dh_dy[:,:][:,len_u:] = dh_dx
    if scale_var == 'matrix':
        dh_dy = Dh.dot(dh_dy.dot(Dy_inv))
    return dh_dy

def run_optimal_load_flow(To2_init=135.3422,To2_bounds=np.array([.7*135.3422,135.3422]),phi2_init=-9*MW,phi2_bounds=np.array([1.3*-9*MW,-9*MW]),m0_bounds=([-500,0]),phi0_bounds=np.array([-40*MW,-30*MW]),m01_bounds=np.array([-200,200]),m02_bounds=np.array([-200,200]),m12_bounds=np.array([-200,200]),m1_bounds=np.array([0,500]),m2_bounds=np.array([-500,0]),p1_bounds=np.array([10,6000*960*9.81]),p2_bounds=np.array([10,6000*960*9.81]),Ts1_bounds=np.array([60,140]),Ts2_bounds=np.array([60,140]),Tr0_bounds=np.array([10,60]),Tr1_bounds=np.array([10,60]),Tr2_bounds=np.array([10,60]),a0=0,a2=0,b0=.04,b2=.05,c0=4e-4,c2=4.5e-4,scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,max_iters_lf=10,formulation='standard',ineq_constr='control',derivatives=False,optimization_method='trust-constr',stay_within_bounds=False,fb=None):
    """Run optimal power flow.

    Parameters
    ----------
    derivatives : bool, optional
        If True, analytical expressions for the gradient and Hessian of the objective function and of the (nonlinear) constraints are used. Otherwise, numerical approximations are used. Default is False.
    """
    if formulation != 'half_link_flow':
        raise ValueError('OPF not implemented for other formulation than half_link_flow')
    print('\nRunning OPF for heat network (inequality constraints on: {}, analytical derivatives: {}, hard bounds: {}, method: {} '.format(ineq_constr, derivatives,stay_within_bounds,optimization_method))
    # create network
    heat_net, water = create_network()

    # update the boundary conditions of the heat network to match the initial guess of opf
    heat_net = update_bc(heat_net,To2_init,phi2_init)

    # run load flow once, to make sure that the initial guess of opf is at least a solution of LF
    x0 = initialize_network(heat_net, water,formulation=formulation)
    x_LF,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iters_lf,formulation=formulation,solver='NR')
    nlsys = NonLinearSystemHeat(heat_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # initial guess for OPF (unscaled)
    m0_init = heat_net.nodes[0].half_links[0].get_m() #<0
    dphi0_init = heat_net.nodes[0].half_links[0].get_dphi() #<0
    slack_init = np.array([m0_init,dphi0_init])
    u_init = np.array([To2_init,phi2_init])
    x_opf0 = np.concatenate((u_init,slack_init,x_LF))

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('Tbase'),scale_var_params.get('phibase')])
        slack_base = np.array([scale_var_params.get('mbase'),scale_var_params.get('phibase')])
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((ubase,slack_base,1/Dx.data[0])))
        DH = np.diag(np.concatenate((np.array([1/scale_var_params.get('mbase'),1/scale_var_params.get('phibase')]),DF.data[0])))
        Df = np.array([1/fb])
        x_opf0 = Dy.dot(x_opf0) # scale y
    else:
        Dy=np.eye(len(x_opf0))
        Dy_inv=np.eye(len(x_opf0))
        Df=np.eye(1)
        DH=np.eye(len(slack_init)+len(x_LF))

    if scale_var == 'per_unit':
        a0 = a0/fb
        a2 = a2/fb
        b0 = b0/(fb/scale_var_params.get('phibase'))
        b2 = b2/(fb/scale_var_params.get('phibase'))
        c0 = c0/(fb/scale_var_params.get('phibase')**2)
        c2 = c2/(fb/scale_var_params.get('phibase')**2)

    # define objective function and its derivatives
    def obj(y,a0=a0,b0=b0,c0=c0,a2=a2,b2=b2,c2=c2,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH):
        """Define the cost function for OPF.

        Parameters
        ----------------
        y : np array
            Variable vector used in OPF.

        Returns
        -----------
        f : float
            The value of the cost function
        """
        global x_f_vec
        x_f_vec = y.copy()
        global dphi0_f_global
        global dphi2_f_global
        global To2_f_global
        global f_vec_global
        dphi0_f_global.append(y[len(u_init)+len(slack_init)-1])
        dphi2_f_global.append(y[len(u_init)-1])
        To2_f_global.append(y[0])
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        dphi0 = y[len(u_init)+len(slack_init)-1] #<0
        dphi2 = y[len(u_init)-1] #<0
        f = price_heat(dphi0,dphi2,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)
        f_vec_global.append(f)
        return f

    def obj_grad(y,a0=a0,b0=b0,c0=c0,a2=a2,b2=b2,c2=c2,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        dphi0_ind = len(u_init)+len(slack_init)-1
        dphi2_ind = len(u_init)-1
        return jac_objective(y,dphi0_ind,dphi2_ind,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)

    def obj_hess(y,a0=a0,b0=b0,c0=c0,a2=a2,b2=b2,c2=c2,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        dphi0_ind = len(u_init)+len(slack_init)-1
        dphi2_ind = len(u_init)-1
        return hess_objective(y,dphi0_ind,dphi2_ind,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)

    # define nonlinear equality constriants (load flow equations)
    def eq_constr(y,network=heat_net,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH, formulation=formulation):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        To2, dphi2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network = update_bc(network,To2,dphi2,scale_var=scale_var,scale_var_params=scale_var_params)
        dphi0_ind = len(u_init)+len(slack_init)-1
        dphi2_ind = len(u_init)-1
        return h(y,dphi0_ind,dphi2_ind,network=network,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH, formulation=formulation)
    def jac_eq_constr(y,network=heat_net,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH, formulation=formulation):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        To2, dphi2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network = update_bc(network,To2,dphi2,scale_var=scale_var,scale_var_params=scale_var_params)
        dphi0_ind = len(u_init)+len(slack_init)-1
        dphi2_ind = len(u_init)-1
        return h_der(y,dphi0_ind,dphi2_ind,network=network,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH, formulation=formulation)

    lb_nleq = np.zeros(len(x_LF)+2)
    ub_nleq = np.zeros(len(x_LF)+2)
    if derivatives:
        if optimization_method == 'trust-constr':
            nonlinear_constraint = spo.NonlinearConstraint(eq_constr,lb_nleq,ub_nleq,jac=jac_eq_constr,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            nonlinear_constraint = {'type':'eq','fun':eq_constr,'jac':jac_eq_constr}
    else:
        if optimization_method == 'trust-constr':
            nonlinear_constraint = spo.NonlinearConstraint(eq_constr,lb_nleq,ub_nleq,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            nonlinear_constraint = {'type':'eq','fun':eq_constr}

    # define linear inequality constraints, i.e. define bounds
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        To2_bounds = To2_bounds/scale_var_params.get('Tbase')
        phi2_bounds = phi2_bounds/scale_var_params.get('phibase')
        m0_bounds = m0_bounds/scale_var_params.get('mbase')
        phi0_bounds = phi0_bounds/scale_var_params.get('phibase')
        m01_bounds = m01_bounds/scale_var_params.get('mbase')
        m02_bounds = m02_bounds/scale_var_params.get('mbase')
        m12_bounds = m12_bounds/scale_var_params.get('mbase')
        m1_bounds = m1_bounds/scale_var_params.get('mbase')
        m2_bounds = m2_bounds/scale_var_params.get('mbase')
        p1_bounds = p1_bounds/scale_var_params.get('phbase')
        p2_bounds = p2_bounds/scale_var_params.get('phbase')
        Ts1_bounds = Ts1_bounds/scale_var_params.get('Tbase')
        Ts2_bounds = Ts2_bounds/scale_var_params.get('Tbase')
        Tr0_bounds = Tr0_bounds/scale_var_params.get('Tbase')
        Tr1_bounds = Tr1_bounds/scale_var_params.get('Tbase')
        Tr2_bounds = Tr2_bounds/scale_var_params.get('Tbase')
    if ineq_constr == 'control':
        # define linear inequality constraints (on the control variables)
        lb_ineq = -np.inf*np.ones(len(x_opf0)) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
        ub_ineq = np.inf*np.ones(len(x_opf0))
        lb_ineq[:len(u_init)] = np.array([To2_bounds[0],phi2_bounds[0]])
        ub_ineq[:len(u_init)]  = np.array([To2_bounds[1],phi2_bounds[1]])
    elif ineq_constr == 'all':
        lb_ineq = np.array([To2_bounds[0],phi2_bounds[0],m0_bounds[0],phi0_bounds[0],m01_bounds[0],m02_bounds[0],m12_bounds[0],m1_bounds[0],m2_bounds[0],p1_bounds[0],p2_bounds[0],Ts1_bounds[0],Ts2_bounds[0],Tr0_bounds[0],Tr1_bounds[0],Tr2_bounds[0]])
        ub_ineq  = np.array([To2_bounds[1],phi2_bounds[1],m0_bounds[1],phi0_bounds[1],m01_bounds[1],m02_bounds[1],m12_bounds[1],m1_bounds[1],m2_bounds[1],p1_bounds[1],p2_bounds[1],Ts1_bounds[1],Ts2_bounds[1],Tr0_bounds[1],Tr1_bounds[1],Tr2_bounds[1]])

    if optimization_method == 'ipopt':
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
    if optimization_method == 'ipopt' and ineq_constr != None:
        bounds = [(lb,ub) for lb, ub in zip(lb_ineq,ub_ineq)]
    elif ineq_constr != None:
        bounds = spo.Bounds(lb_ineq,ub_ineq,keep_feasible=stay_within_bounds)
    else:
        bounds = None

    # make sure initial guess satisfies bounds (NB. If adjustments are made, LF is not necessarily satisfied anymore)
    if ineq_constr != None and (optimization_method == 'SLSQP' or stay_within_bounds):
        for ind, x0 in enumerate(x_opf0):
            if lb_ineq[ind] > x0:
                x_opf0[ind] = lb_ineq[ind]
            elif ub_ineq[ind] < x0:
                x_opf0[ind] = ub_ineq[ind]

    f_vec = list()
    dphi0_vec = list()
    dphi2_vec = list()
    To2_vec = list()
    global dphi0_f_global
    global dphi2_f_global
    global To2_f_global
    global f_vec_global
    global x_f_vec
    dphi0_f_global = list()
    dphi2_f_global = list()
    To2_f_global = list()
    f_vec_global = list()
    x_f_vec = list()
    if optimization_method == 'trust-constr':
        def callback(xk, state):
            """Called after every iteration"""
            f_vec.append(state.fun)
            To2_vec.append(xk[0])
            dphi0_vec.append(xk[3])
            dphi2_vec.append(xk[1])
            return False
    elif optimization_method == 'SLSQP':
        f_vec.append(obj(x_opf0))
        To2_vec.append(x_opf0[0])
        dphi0_vec.append(x_opf0[3])
        dphi2_vec.append(x_opf0[1])
        def callback(xk):
            """Called after every iteration"""
            f_vec.append(obj(xk))
            To2_vec.append(xk[0])
            dphi0_vec.append(xk[3])
            dphi2_vec.append(xk[1])
            return False
    elif optimization_method == 'ipopt':
        # callback is not implemented in the ipopt (cyipopt) package / wrapper.
        pass

    # solve OPF
    opf_start_time = time.time()
    try:
        if derivatives:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, x_opf0, method=optimization_method,jac=obj_grad,hess=obj_hess, constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol, callback=callback)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, x_opf0, method=optimization_method,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, x_opf0,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
        else:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, x_opf0, method=optimization_method, constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol, callback=callback)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, x_opf0, method=optimization_method, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol,callback=callback)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, x_opf0, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
    except:
        print('Exception made for {}, hard bounds: {}, analytical der.: {}'.format(optimization_method,stay_within_bounds,derivatives))
        if len(f_vec_global) == 0:
            obj(x_opf0)
            nit = 0
            nfev = 0
            njev = 0
            nhev = 0
        else:
            nit = 0
            nfev = len(f_vec_global)
            njev = 0
            nhev = 0
        execution_time = opf_start_time - time.time()
        res = spo.OptimizeResult({'success':False,'x':np.array(x_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

    if optimization_method == 'ipopt':
        if res.nit > 0:
            dphi0_vec = [dphi0_f_global[ind] for ind in range(0,len(dphi0_f_global),round(len(dphi0_f_global)/res.nit))]
            dphi2_vec = [dphi2_f_global[ind] for ind in range(0,len(dphi2_f_global),round(len(dphi2_f_global)/res.nit))]
            To2_vec = [To2_f_global[ind] for ind in range(0,len(To2_f_global),round(len(To2_f_global)/res.nit))]
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            dphi0_vec = dphi0_f_global
            dphi2_vec = dphi2_f_global
            To2_vec = To2_f_global
            f_vec = f_vec_global

    if scale_var == 'matrix' or scale_var == 'per_unit':
        x_opf = Dy_inv.dot(res.x)
    else:
        x_opf = res.x

    # print solution
    To2,dphi2 = x_opf[:len(u_init)]
    heat_net = update_bc(heat_net,To2,dphi2)
    xh_opt = xh_from_xopf(x_opf)
    m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(xh_opt,formulation=formulation)
    rho_w = water.rhon
    grav_const = water.g
    print('Solution OPF (inequality constraints on control variables: {})'.format(ineq_constr))
    print('p heat = {} m'.format(p_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {}'.format(Ts_hl_vec))
    print('Tr hl = {}'.format(Tr_hl_vec))
    print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
    return xh_opt, res, f_vec, To2_vec, dphi0_vec, dphi2_vec, execution_time

def solve_lf_in_lf(network,u,max_iters=10,tol=1e-6,formulation='standard',scale_var=None,scale_var_params=None):
    """Solve steady-state LF within an optmization context.

    Parameters
    ----------
    u : np arrays
        Vector with control variables. Scaled when using per unit scaling, unscaled otherwise
    """
    To2,dphi2 = u
    xh_init = network.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    network.reset_network(xh_init,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    network = update_bc(network,To2,dphi2,scale_var=scale_var,scale_var_params=scale_var_params)
    x_LF,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = network.solve_network(tol,max_iters,formulation=formulation,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    dphi0 = network.nodes[0].half_links[0].get_dphi(scale_var=scale_var,scale_var_params=scale_var_params) # Scaled when per unit is used. Is a source, so the half link power <0
    return dphi0, network

def run_optimal_load_flow_separate_LF(To2_init=135.3422,To2_bounds=np.array([.7*135.3422,135.3422]),phi2_init=-9*MW,phi2_bounds=np.array([1.3*-9*MW,-9*MW]),m0_bounds=([-500,0]),phi0_bounds=np.array([-40*MW,-30*MW]),m01_bounds=np.array([-200,200]),m02_bounds=np.array([-200,200]),m12_bounds=np.array([-200,200]),m1_bounds=np.array([0,500]),m2_bounds=np.array([-500,0]),p1_bounds=np.array([10,6000*960*9.81]),p2_bounds=np.array([10,6000*960*9.81]),Ts1_bounds=np.array([60,140]),Ts2_bounds=np.array([60,140]),Tr0_bounds=np.array([10,60]),Tr1_bounds=np.array([10,60]),Tr2_bounds=np.array([10,60]),a0=0,a2=0,b0=.04,b2=.05,c0=4e-4,c2=4.5e-4,scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,max_iters_lf=10,formulation='standard',ineq_constr='control',approach='direct',optimization_method='trust-constr',stay_within_bounds=False,fb=None):
    """Run optimal power flow.

    Parameters
    ----------
    approach : str, optional
        Approach used to compute the gradient and Jacobians. Either 'direct' or 'adjoint'. Default is 'direct'.
    """
    print('\nRunning OPF with separate LF for heat network (inequality constraints on control variables: {}), approach: {}, hard bounds: {}, method: {}, scaling: {} '.format(ineq_constr, approach,stay_within_bounds, optimization_method, scale_var))
    if formulation != 'half_link_flow':
        raise ValueError('OPF with substituted LF not implemented for other formulation than half_link_flow')
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        To2_bounds = To2_bounds/scale_var_params.get('Tbase')
        phi2_bounds = phi2_bounds/scale_var_params.get('phibase')
        m0_bounds = m0_bounds/scale_var_params.get('mbase')
        phi0_bounds = phi0_bounds/scale_var_params.get('phibase')
        m01_bounds = m01_bounds/scale_var_params.get('mbase')
        m02_bounds = m02_bounds/scale_var_params.get('mbase')
        m12_bounds = m12_bounds/scale_var_params.get('mbase')
        m1_bounds = m1_bounds/scale_var_params.get('mbase')
        m2_bounds = m2_bounds/scale_var_params.get('mbase')
        p1_bounds = p1_bounds/scale_var_params.get('phbase')
        p2_bounds = p2_bounds/scale_var_params.get('phbase')
        Ts1_bounds = Ts1_bounds/scale_var_params.get('Tbase')
        Ts2_bounds = Ts2_bounds/scale_var_params.get('Tbase')
        Tr0_bounds = Tr0_bounds/scale_var_params.get('Tbase')
        Tr1_bounds = Tr1_bounds/scale_var_params.get('Tbase')
        Tr2_bounds = Tr2_bounds/scale_var_params.get('Tbase')

    # create network
    heat_net, water = create_network()

    # update the boundary conditions of the heat network to match the initial guess of opf
    heat_net = update_bc(heat_net,To2_init,phi2_init)
    xh0 = initialize_network(heat_net, water,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation) # initialize network, and set reasonable values as first initial guess for LF (if reasonable values are not set, division by 0 etc might occur during LF)

    nlsys = NonLinearSystemHeat(heat_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # initial guess for opf (unscaled)
    u0 = np.array([To2_init,phi2_init])

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('Tbase'),scale_var_params.get('phibase')])
        slack_base = np.array([scale_var_params.get('mbase'),scale_var_params.get('phibase')])
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((ubase,slack_base,1/Dx.data[0])))
        Du = np.diag(1/ubase)
        Du_inv = np.diag(ubase)
        DH = np.diag(np.concatenate((np.array([1/scale_var_params.get('mbase'),1/scale_var_params.get('phibase')]),DF.data[0])))
        Df = np.array([1/fb])
        u0 = Du.dot(u0) # scale u
    else:
        Dy=np.eye(16)
        Dy_inv=np.eye(16)
        Du=np.eye(2)
        Du_inv=np.eye(2)
        Df=np.eye(1)
        DH=np.eye(14)

    if scale_var == 'per_unit':
        a0 = a0/fb
        a2 = a2/fb
        b0 = b0/(fb/scale_var_params.get('phibase'))
        b2 = b2/(fb/scale_var_params.get('phibase'))
        c0 = c0/(fb/scale_var_params.get('phibase')**2)
        c2 = c2/(fb/scale_var_params.get('phibase')**2)

    # define cost function / objective function
    def obj(u,a0=a0,b0=b0,c0=c0,a2=a2,b2=b2,c2=c2,network=heat_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH):
        """Define the cost function for OPF.

        Parameters
        ----------------
        x_opf : np array
            Variable vector used in OPF.

        Returns
        -----------
        f : float
            The value of the cost function
        """
        global x_f_vec
        x_f_vec = u.copy()
        global dphi0_f_global
        global dphi2_f_global
        global To2_f_global
        global f_vec_global
        dphi2_f_global.append(u[1])
        To2_f_global.append(u[0])
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            u = Du_inv.dot(u)
        dphi0, network = solve_lf_in_lf(network,u,tol=tol,max_iters=max_iters_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        if scale_var == 'matrix':
            # Optimizers uses scaled x, but solve return unscaled dphi0 when matrix scaling is used
            dphi0_f_global.append(dphi0/scale_var_params.get('phibase'))
        else:
            dphi0_f_global.append(dphi0)
        dphi2 = u[1] #<0
        f = price_heat(dphi0,dphi2,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)
        f_vec_global.append(f)
        return f

    # gradient of objective function
    def obj_grad(u,a0=a0,b0=b0,c0=c0,a2=a2,b2=b2,c2=c2,formulation=formulation,network=heat_net,nlsys=nlsys,method=approach,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            u = Du_inv.dot(u)
        # update network and solve LF
        To2,dphi2 = u
        dphi0, network = solve_lf_in_lf(network,u,tol=tol,max_iters=max_iters_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        m0 = network.nodes[0].half_links[0].get_m(scale_var=scale_var,scale_var_params=scale_var_params)
        dphi0_ind = 3
        dphi2_ind = 1
        # partial derivatives of objective
        x_LF = network.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        y = np.concatenate((u,np.array([m0,dphi0]),x_LF))
        deltaf_deltay = jac_objective(y,dphi0_ind,dphi2_ind,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)
        deltaf_deltau = np.zeros((1,2))
        deltaf_deltax = np.zeros((1,14))
        deltaf_deltau[0,:] = deltaf_deltay[:len(u)]
        deltaf_deltax[0,:] = deltaf_deltay[len(u):]
        # partial derivatives of equatilty constraints / load-flow equations
        deltah_deltay = h_der(y,dphi0_ind,dphi2_ind,network=network,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH, formulation=formulation)
        deltah_deltau = deltah_deltay[:,:len(u)]
        deltah_deltax = deltah_deltay[:,len(u):]
        # gradient objective
        df_du = deltaf_deltau.copy() # first part of gradient
        if method == 'direct':
            w = np.linalg.solve(deltah_deltax,-deltah_deltau)
            df_du += np.dot(deltaf_deltax,w)
        elif method == 'adjoint':
            v = np.linalg.solve(np.transpose(deltah_deltax),np.transpose(deltaf_deltax))
            df_du += np.dot(np.transpose(v),-deltah_deltau)
        df_du = df_du.ravel() # jac needs to be 'array_like, shape (n,)'
        # print('df_du={}\ndeltaf_deltau={}\ndeltaf_deltax={}\ndeltah_deltau={}\ndeltah_deltax={}'.format(df_du,deltaf_deltau,deltaf_deltax,deltah_deltau,deltah_deltax))
        return df_du

    if ineq_constr == 'all':
        lb_ineq_state = np.array([m0_bounds[0],phi0_bounds[0],m01_bounds[0],m02_bounds[0],m12_bounds[0],m1_bounds[0],m2_bounds[0],p1_bounds[0],p2_bounds[0],Ts1_bounds[0],Ts2_bounds[0],Tr0_bounds[0],Tr1_bounds[0],Tr2_bounds[0]])
        ub_ineq_state = np.array([m0_bounds[1],phi0_bounds[1],m01_bounds[1],m02_bounds[1],m12_bounds[1],m1_bounds[1],m2_bounds[1],p1_bounds[1],p2_bounds[1],Ts1_bounds[1],Ts2_bounds[1],Tr0_bounds[1],Tr1_bounds[1],Tr2_bounds[1]])
        # print('ub phi0 = {}'.format(ub_ineq_state[1]))
        # define inequality constraints
        def g(u,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation,network=heat_net,Dy=Dy, Dy_inv=Dy_inv, Du=Du,Du_inv=Du_inv,lb_ineq_state = lb_ineq_state, ub=ub_ineq_state):
            """Determine the nonlinear inequality constraints g(x(u)) >= 0"""
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                u = Du_inv.dot(u)
            # update network and solve LF
            To2,dphi2 = u
            dphi0, network = solve_lf_in_lf(network,u,tol=tol,max_iters=max_iters_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            m0 = network.nodes[0].half_links[0].get_m(scale_var=scale_var,scale_var_params=scale_var_params)
            # print('In g: m0 = {}, dphi0 = {}'.format(m0,dphi0))
            x_LF = network.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            x = np.concatenate((np.array([m0,dphi0]),x_LF))
            if scale_var == 'matrix': # lb_ineq_state and ub_ineq_state are scaled, so scale x as wel
                x = Dy[len(u):,len(u):].dot(x)
            # print('lb = {}\nx = {}\nub = {}\nx-lb > 0: {}\nub-x > 0 : {}'.format(lb_ineq_state,x,ub_ineq_state,(x-lb_ineq_state>=0),(ub_ineq_state-x>=0)))
            # print('All x-lb > 0: {}\nAll ub-x > 0 : {}'.format(np.all((x-lb_ineq_state>=0)),np.all((ub_ineq_state-x>=0))))
            # print('lb = {}\nu = {}\nub = {}\nlb<u={}\nu<ub={}'.format([To2_bounds[0],phi2_bounds[0]],u,[To2_bounds[1],phi2_bounds[1]],np.array([To2_bounds[0],phi2_bounds[0]])<=u,u<=np.array([To2_bounds[1],phi2_bounds[1]])))
            g = np.concatenate((x-lb_ineq_state,ub_ineq_state-x))
            # print('g>=0: {}'.format(g>=0))
            return g
        def g_jac(u,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation,network=heat_net,nlsys=nlsys,method=approach,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH,Du=Du,Du_inv=Du_inv):
            """Jacobian of inequality constraints"""
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                u = Du_inv.dot(u)
            # Jacobian of inequality constraints wrt state variables x
            deltag_deltax = np.vstack((np.eye(14),-np.eye(14)))
            deltag_deltau = np.zeros((28,2))
            # update network and solve LF
            To2,dphi2 = u
            dphi0, network = solve_lf_in_lf(network,u,tol=tol,max_iters=max_iters_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            m0 = network.nodes[0].half_links[0].get_m(scale_var=scale_var,scale_var_params=scale_var_params)
            dphi0_ind = 3
            dphi2_ind = 1
            # partial derivatives of objective
            x_LF = network.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            y = np.concatenate((u,np.array([m0,dphi0]),x_LF))
            # partial derivatives of equatilty constraints / load-flow equations
            deltah_deltay = h_der(y,dphi0_ind,dphi2_ind,network=network,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH, formulation=formulation)
            deltah_deltau = deltah_deltay[:,:len(u)]
            deltah_deltax = deltah_deltay[:,len(u):]
            # jacobian inequality constraints
            dg_du = deltag_deltau.copy() # first part of gradient
            if method == 'direct':
                w = np.linalg.solve(deltah_deltax,-deltah_deltau)
                dg_du += np.dot(deltag_deltax,w)
            elif method == 'adjoint':
                v = np.linalg.solve(np.transpose(deltah_deltax),np.transpose(deltag_deltax))
                dg_du += np.dot(np.transpose(v),-deltah_deltau)
            # print('dim dg_du = {}\ndg_du = {}'.format(dg_du.shape,dg_du))
            return dg_du
        if optimization_method == 'trust-constr':
            ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(28),np.inf*np.ones(28),jac=g_jac,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            ineq_constr_fun  = {'type':'ineq','fun':g,'jac':g_jac}
    else:
        ineq_constr_fun = None

    # define bounds
    if ineq_constr != None:
        # define linear inequality constraints (on the control variables)
        lb_ineq = np.array([To2_bounds[0],phi2_bounds[0]])
        ub_ineq = np.array([To2_bounds[1],phi2_bounds[1]])
    else:
        bounds = None

    if optimization_method == 'ipopt':
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
    if optimization_method == 'ipopt' and ineq_constr != None:
        bounds = [(lb,ub) for lb, ub in zip(lb_ineq,ub_ineq)]
    elif ineq_constr != None:
        bounds = spo.Bounds(lb_ineq,ub_ineq,keep_feasible=stay_within_bounds)
    else:
        bounds = None

    # make sure initial guess satisfies bounds (NB. If adjustments are made, LF is not necessarily satisfied anymore)
    if ineq_constr != None and (optimization_method == 'SLSQP' or stay_within_bounds):
        for ind, x0 in enumerate(u0):
            if lb_ineq[ind] > x0:
                u0[ind] = lb_ineq[ind]
            elif ub_ineq[ind] < x0:
                u0[ind] = ub_ineq[ind]

    f_vec = list()
    dphi0_vec = list()
    dphi2_vec = list()
    To2_vec = list()
    global dphi0_f_global
    global dphi2_f_global
    global To2_f_global
    global f_vec_global
    global x_f_vec
    dphi0_f_global = list()
    dphi2_f_global = list()
    To2_f_global = list()
    f_vec_global = list()
    x_f_vec = list()
    if optimization_method == 'trust-constr':
        def callback(xk, state):
            """Called after every iteration"""
            f_vec.append(state.fun)
            To2_vec.append(xk[0])
            dphi2_vec.append(xk[1])
            return False
    elif optimization_method == 'SLSQP':
        f_vec.append(obj(u0))
        To2_vec.append(u0[0])
        dphi2_vec.append(u0[1])
        def callback(xk):
            """Called after every iteration"""
            f_vec.append(obj(xk))
            To2_vec.append(xk[0])
            dphi2_vec.append(xk[1])
            return False
    elif optimization_method == 'ipopt':
        # callback is not implemented in the ipopt (cyipopt) package / wrapper.
        pass

    # solve OPF
    opf_start_time = time.time()
    try:
        if optimization_method == 'trust-constr':
            res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=[ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds, callback=callback)
            execution_time = res.execution_time
        elif optimization_method == 'SLSQP':
            res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
            execution_time = opf_start_time - time.time()
        elif optimization_method == 'ipopt':
            res = ipopt.minimize_ipopt(obj, u0,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
            execution_time = opf_start_time - time.time()
    except:
        print('Exception made for {}, hard bounds: {}, approach: {}, scaling: {}'.format(optimization_method,stay_within_bounds,approach,scale_var))
        if len(f_vec) == 0:
            obj(u0)
            nit = 0
            nfev = 0
            njev = 0
            nhev = 0
        else:
            nit = 0
            nfev = len(f_vec)
            njev = 0
            nhev = 0
        execution_time = opf_start_time - time.time()
        res = spo.OptimizeResult({'success':False,'x':np.array(x_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

    if optimization_method == 'ipopt':
        if res.nit > 0:
            dphi0_vec = [dphi0_f_global[ind] for ind in range(0,len(dphi0_f_global),round(len(dphi0_f_global)/res.nit))]
            dphi2_vec = [dphi2_f_global[ind] for ind in range(0,len(dphi2_f_global),round(len(dphi2_f_global)/res.nit))]
            To2_vec = [To2_f_global[ind] for ind in range(0,len(To2_f_global),round(len(To2_f_global)/res.nit))]
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            dphi0_vec = dphi0_f_global
            dphi2_vec = dphi2_f_global
            To2_vec = To2_f_global
            f_vec = f_vec_global
    else:
        if len(dphi2_vec) == 0:
            dphi0_vec = list()
        elif res.nfev < len(dphi2_vec):
            dphi0_vec = dphi0_f_global
            dphi2_vec = dphi2_f_global
        else:
            dphi0_vec = [dphi0_f_global[ind] for ind in range(0,len(dphi0_f_global),round(len(dphi0_f_global)/len(dphi2_vec)))]
            if len(dphi0_vec) < len(dphi2_vec):
                dphi0_vec.append(dphi0_f_global[-1])

    if scale_var == 'matrix':
        u_opf = Du_inv.dot(res.x)
    else:
        u_opf = res.x
    # print('ub phi0 = {}'.format(ub_ineq_state[1]))
    # print solution
    dphi0, heat_net = solve_lf_in_lf(heat_net,u_opf,tol=tol,max_iters=max_iters_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    xh_opt = heat_net.set_x_init(formulation=formulation) # unscaled
    heat_net.reset_network(xh_opt,formulation=formulation)
    m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(xh_opt,formulation=formulation)
    rho_w = water.rhon
    grav_const = water.g
    print('Solution OPF (inequality constraints on control variables: {})'.format(ineq_constr))
    print('p heat = {} m'.format(p_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {}'.format(Ts_hl_vec))
    print('Tr hl = {}'.format(Tr_hl_vec))
    print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
    return xh_opt, res, f_vec, To2_vec, dphi0_vec, dphi2_vec, execution_time

def compare_opf_derivatives(dir_path,number_runs=10,save_figs=False,save_tables=False):
    """Compare the optimal flow problem, using different ways to determine the gradients (and Hessians). p0 and Ts0 are taken as BC, and the inequality constraints are imposed on the control variables only. Print results to a table that can be read by LaTeX."""
    # solver info
    tol = 1e-6
    max_iter = 10
    formulation = 'half_link_flow'
    ineq_constr = True # inequality constraints on control variables are used
    max_iter_opf = 500
    # parameters for objective function
    a0=0
    a2=0
    b0=.04
    b2=.05
    c0=4e-4
    c2=4.5e-4
    # steady-state LF solution
    with HiddenPrints():
        heat_net,x_sol_LF,iters,err_vec,m_vec_LF,p_vec_LF,Ts_vec_LF,Tr_vec_LF,m_hl_vec_LF,phi_hl_vec_LF,Ts_hl_vec_LF,Tr_hl_vec_LF = run_load_flow(tol=tol,max_iter=max_iter,formulation=formulation)
        dphi0_sol = heat_net.nodes[0].half_links[0].get_dphi()
        dphi2_sol = heat_net.nodes[2].half_links[0].get_dphi()
        To2_sol = heat_net.nodes[2].half_links[0].get_Ts()
        f_LF = price_heat(dphi0_sol, dphi2_sol,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2) # value of objective function for the LF solution
    # limits for inequality constraints
    To2_lb=To2_sol
    To2_ub=1.1*To2_sol
    phi2_lb=1.3*dphi2_sol
    phi2_ub=dphi2_sol
    # initial guess
    To2_init = 1.08*To2_sol
    phi2_init = 1.1*dphi2_sol

    # run the various optimizations. Run several times, take average of run time. For the other data (which seemed to be the same every time), the last run is used.
    exec_times = list()
    exec_times_num_der = list()
    exec_times_sepLF_direct = list()
    exec_times_sepLF_adjoint = list()
    for run in range(number_runs):
        # LF is included as (nonlinear) equality constriant. Analytical expressions for gradients and Hessian of objective function and Jacobian of equality constraints are used
        xh_opt, res, f_vec, To2_vec, dphi0_vec, dphi2_vec, exec_time = run_optimal_load_flow(To2_init=To2_init,To2_lb=To2_lb,To2_ub=To2_ub,phi2_init=phi2_init,phi2_lb=phi2_ub,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,tol=tol,max_iter=max_iter_opf,formulation=formulation,ineq_constr=ineq_constr, derivatives=True)
        x_opf, obj_fun, nfev, nit = res.x, res.fun, res.nfev, res.nit
        exec_times.append(exec_time)
        # LF is included as (nonlinear) equality constriant. Numerical approximation for gradients and Hessian of objective function and Jacobian of equality constraints are used
        xh_opt_num_der, res_num_der, f_vec_num_der, To2_vec_num_der, dphi0_vec_num_der, dphi2_vec_num_der, exec_time_num_der = run_optimal_load_flow(To2_init=To2_init,To2_lb=To2_lb,To2_ub=To2_ub,phi2_init=phi2_init,phi2_lb=phi2_ub,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,tol=tol,max_iter=max_iter_opf,formulation=formulation,ineq_constr=ineq_constr)
        x_opf_num_der, obj_fun_num_der, nfev_num_der, nit_num_der = res_num_der.x, res_num_der.fun, res_num_der.nfev, res_num_der.nit
        exec_times_num_der.append(exec_time_num_der)
        # LF is not included as (nonlinear) equality constriant. Analytical expressions for gradient of objective function are determined using the direct approach
        xh_opt_sepLF_direct, res_sep_LF_direct, f_vec_sepLF_direct, To2_vec_sepLF_direct, dphi0_vec_sepLF_direct, dphi2_vec_sepLF_direct, exec_time_sepLF_direct = run_optimal_load_flow_separate_LF(To2_init=To2_init,To2_lb=To2_lb,To2_ub=To2_ub,phi2_init=phi2_init,phi2_lb=phi2_ub,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,tol=tol,max_iter=max_iter_opf,formulation=formulation,ineq_constr=ineq_constr, approach='direct')
        x_opf_sep_LF_direct, obj_fun_sep_LF_direct, nfev_sep_LF_direct, nit_sep_LF_direct = res_sep_LF_direct.x, res_sep_LF_direct.fun, res_sep_LF_direct.nfev, res_sep_LF_direct.nit
        exec_times_sepLF_direct.append(exec_time_sepLF_direct)
        # LF is not included as (nonlinear) equality constriant. Analytical expressions for gradient of objective function are determined using the adjoint approach
        xh_opt_sepLF_adjoint, res_sep_LF_adjoint, f_vec_sepLF_adjoint, To2_vec_sepLF_adjoint, dphi0_vec_sepLF_adjoint, dphi2_vec_sepLF_adjoint, exec_time_sepLF_adjoint = run_optimal_load_flow_separate_LF(To2_init=To2_init,To2_lb=To2_lb,To2_ub=To2_ub,phi2_init=phi2_init,phi2_lb=phi2_ub,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,tol=tol,max_iter=max_iter_opf,formulation=formulation,ineq_constr=ineq_constr, approach='adjoint')
        x_opf_sep_LF_adjoint, obj_fun_sep_LF_adjoint, nfev_sep_LF_adjoint, nit_sep_LF_adjoint = res_sep_LF_adjoint.x, res_sep_LF_adjoint.fun, res_sep_LF_adjoint.nfev, res_sep_LF_adjoint.nit
        exec_times_sepLF_adjoint.append(exec_time_sepLF_adjoint)

    exec_time = np.mean(exec_times)
    exec_time_num_der = np.mean(exec_times_num_der)
    exec_time_sepLF_direct = np.mean(exec_times_sepLF_direct)
    exec_time_sepLF_adjoint = np.mean(exec_times_sepLF_adjoint)
    print('exec time = {}'.format(exec_times))
    print('exec time num. der. = {}'.format(exec_times_num_der))
    print('exec time sep. LF direct = {}'.format(exec_times_sepLF_direct))
    print('exec time sep. LF adjoint = {}'.format(exec_times_sepLF_adjoint))

    # create (and save) table with optimal solution in network
    E = len(heat_net.links)
    N = len(heat_net.nodes)
    T = len(heat_net.half_links)

    # solution with LF as equality constraints
    heat_net, water = create_network()
    heat_net.initialize()
    To2,dphi2 = x_opf[:2]
    heat_net = update_bc(heat_net,To2,dphi2)
    xh_opt = xh_from_xopf(x_opf)
    m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(xh_opt,formulation=formulation)
    # solution with LF as equality constraints, using numerical derivatives
    heat_net, water = create_network()
    heat_net.initialize()
    To2_num_der,dphi2_num_der = x_opf_num_der[:2]
    heat_net = update_bc(heat_net,To2_num_der,dphi2_num_der)
    xh_opt_num_der = xh_from_xopf(x_opf_num_der)
    heat_net.reset_network(xh_opt_num_der,formulation=formulation)
    m_vec_num_der,p_vec_num_der,Ts_vec_num_der,Tr_vec_num_der,m_hl_vec_num_der,phi_hl_vec_num_der,Ts_hl_vec_num_der,Tr_hl_vec_num_der = heat_net.update_full(xh_opt_num_der,formulation=formulation)
    # solution with LF implicit, direct approach
    heat_net, water = create_network()
    heat_net.initialize()
    To2_sepLF_direct,dphi2_sepLF_direct = x_opf_sepLF_direct
    heat_net = update_bc(heat_net,To2_sepLF_direct,dphi2_sepLF_direct)
    heat_net.reset_network(x_sol_LF,formulation=formulation) # set reasonable initial guess
    xh_opt_sepLF_direct,_,_,m_vec_sepLF_direct,p_vec_sepLF_direct,Ts_vec_sepLF_direct,Tr_vec_sepLF_direct,m_hl_vec_sepLF_direct,phi_hl_vec_sepLF_direct,Ts_hl_vec_sepLF_direct,Tr_hl_vec_sepLF_direct = heat_net.solve_network(tol,max_iter,formulation=formulation,solver='NR')
    # solution with LF implicit, adjoint approach
    heat_net, water = create_network()
    heat_net.initialize()
    To2_sepLF_adjoint,dphi2_sepLF_adjoint = x_opf_sepLF_adjoint
    heat_net = update_bc(heat_net,To2_sepLF_adjoint,dphi2_sepLF_adjoint)
    heat_net.reset_network(x_sol_LF,formulation=formulation) # set reasonable initial guess
    xh_opt_sepLF_adjoint,_,_,m_vec_sepLF_adjoint,p_vec_sepLF_adjoint,Ts_vec_sepLF_adjoint,Tr_vec_sepLF_adjoint,m_hl_vec_sepLF_adjoint,phi_hl_vec_sepLF_adjoint,Ts_hl_vec_sepLF_adjoint,Tr_hl_vec_sepLF_adjoint = heat_net.solve_network(tol,max_iter,formulation=formulation,solver='NR')

    rho_w = water.rhon
    grav_const = water.g
    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','H3N')
        with open(os.path.join(path_to_tables,'network_solution_derivatives.txt'), "w") as table:
            table.write(r'$m_{01}$'+r' & {:5f}  & {:5f}  & {:5f}  & {:5f}  & {:5f}\\ '.format(m_vec_LF[0],m_vec_num_der[0],m_vec[0],m_vec_sepLF_direct[0],m_vec_sepLF_adjoint[0]))
            table.write(r'$m_{02}$'+r' & {:5f}  & {:5f}  & {:5f}  & {:5f}  & {:5f}\\ '.format(m_vec_LF[1],m_vec_num_der[1],m_vec[1],m_vec_sepLF_direct[1],m_vec_sepLF_adjoint[1]))
            table.write(r'$m_{03}$'+r' & {:5f}  & {:5f}  & {:5f}  & {:5f}  & {:5f}\\ '.format(m_vec_LF[2],m_vec_num_der[2],m_vec[2],m_vec_sepLF_direct[2],m_vec_sepLF_adjoint[2]))
            for ind_hl in range(T):
                table.write(r'$m_{'+r'{:d}'.format(ind_hl)+r',0}$'+r' & {:5f}  & {:5f}  & {:5f}  & {:5f}  & {:5f}\\ '.format(m_hl_vec_LF[ind_hl][0],m_hl_vec_num_der[ind_hl][0],m_hl_vec[ind_hl][0],m_hl_vec_sepLF_direct[ind_hl][0],m_hl_vec_sepLF_adjoint[ind_hl][0]))
            for ind_n in range(N):
                table.write(r'$h_{:d}$ [m]'.format(ind_n)+r' & {:5f}  & {:5f}  & {:5f}  & {:5f}  & {:5f}\\ '.format(p_vec_LF[ind_n]/(rho_w*grav_const),p_vec_num_der[ind_n]/(rho_w*grav_const),p_vec[ind_n]/(rho_w*grav_const),p_vec_sepLF_direct[ind_n]/(rho_w*grav_const),p_vec_sepLF_adjoint[ind_n]/(rho_w*grav_const)))
            for ind_n in range(N):
                table.write(r'$T^s_{:d}$'.format(ind_n)+r' & {:5f}  & {:5f}  & {:5f}  & {:5f}  & {:5f}\\ '.format(Ts_vec_LF[ind_n],Ts_vec_num_der[ind_n],Ts_vec[ind_n],Ts_vec_sepLF_direct[ind_n],Ts_vec_sepLF_adjoint[ind_n]))
            for ind_n in range(N):
                table.write(r'$T^r_{:d}$'.format(ind_n)+r' & {:5f}  & {:5f}  & {:5f}  & {:5f}  & {:5f}\\ '.format(Tr_vec_LF[ind_n],Tr_vec_num_der[ind_n],Tr_vec[ind_n],Tr_vec_sepLF_direct[ind_n],Tr_vec_sepLF_adjoint[ind_n]))
            for ind_hl in range(T):
                table.write(r'$T^s_{'+r'{:d}'.format(ind_hl)+r',0}$'+r' & {:5f}  & {:5f}  & {:5f}  & {:5f}  & {:5f}\\ '.format(Ts_hl_vec_LF[ind_hl][0],Ts_hl_vec_num_der[ind_hl][0],Ts_hl_vec[ind_hl][0],Ts_hl_vec_sepLF_direct[ind_hl][0],Ts_hl_vec_sepLF_adjoint[ind_hl][0]))
            for ind_hl in range(T):
                table.write(r'$T^r_{'+r'{:d}'.format(ind_hl)+r',0}$'+r' & {:5f}  & {:5f}  & {:5f}  & {:5f}  & {:5f}\\ '.format(Tr_hl_vec_LF[ind_hl][0],Tr_hl_vec_num_der[ind_hl][0],Tr_hl_vec[ind_hl][0],Tr_hl_vec_sepLF_direct[ind_hl][0],Tr_hl_vec_sepLF_adjoint[ind_hl][0]))
            for ind_hl in range(T):
                table.write(r'$\Delta \varphi_{:d}$ [MW]'.format(ind_hl)+r' & {:5f}  & {:5f}  & {:5f}  & {:5f}  & {:5f}\\ '.format(phi_hl_vec_LF[ind_hl][0]/MW,phi_hl_vec_num_der[ind_hl][0]/MW,phi_hl_vec[ind_hl][0]/MW,phi_hl_vec_sepLF_direct[ind_hl][0]/MW,phi_hl_vec_sepLF_adjoint[ind_hl][0]/MW))

    # print results of optimizer, and create (and save) table
    print('\nopf num. der.   opf     opf sep. LF direct    opf sep. LF adjoint')
    print('obj. func:  {:.5e}  , {:.5e} , {:.5e} , {:.5e}'.format(obj_fun_num_der,obj_fun,obj_fun_sepLF_direct,obj_fun_sepLF_adjoint))
    print('numb. fev.:  {:d}  , {:d} , {:d} , {:d}'.format(nfev_num_der,nfev,nfev_sepLF_direct,nfev_sepLF_adjoint))
    print('iters:  {:d}  , {:d}  , {:d} , {:d}'.format(nit_num_der,nit,nit_sepLF_direct,nit_sepLF_adjoint))
    print('time:  {:.5f}  , {:.5f} , {:5f} , {:.5f}\n'.format(exec_time_num_der,exec_time,exec_time_sepLF_direct,exec_time_sepLF_adjoint))
    if save_tables:
        with open(os.path.join(path_to_tables,'optimizer_info_derivatives.txt'), "w") as table:
            table.write(r'$f$ & {:.5e} &  {:.5e} &  {:.5e}  & {:.5e} & {:.5e}  \\ '.format(f_LF,obj_fun_num_der,obj_fun,obj_fun_sepLF_direct,obj_fun_sepLF_adjoint))
            table.write(r'func. eval. & & {:d} & {:d}  & {:d}  & {:d}  \\ '.format(nfev_num_der,nfev,nfev_sepLF_direct,nfev_sepLF_adjoint))
            table.write(r'iterations & & {:d} & {:d}  & {:d}  & {:d}  \\ '.format(nit_num_der,nit,nit_sepLF_direct,nit_sepLF_adjoint))
            table.write(r'time [s] & & {:.5f} & {:.5f}  & {:.5f}  & {:.5f}  \\ '.format(exec_time_num_der,exec_time,exec_time_sepLF_direct,exec_time_sepLF_adjoint))

    # plots
    colors = {'LF':'k', 'OPF':'tab:blue', 'OPF direct':'tab:orange', 'OPF adjoint':'tab:green', 'OPF num. der.':'tab:red'}
    dphi0 = np.linspace(-20*MW,-45*MW,500)
    dphi2 = np.linspace(-8.5*MW,-10*MW,100)
    PHI0, PHI2 = np.meshgrid(dphi0,dphi2)
    f = price_heat(PHI0, PHI2,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2)
    fig_cost = plt.figure('cost_function_3d')
    ax_cost = fig_cost.gca(projection='3d')
    surf = ax_cost.plot_surface(PHI0,PHI2,f,cmap=cm.coolwarm)
    ax_cost.set_xlabel(r'$\Delta \varphi_0$ [MW]')
    ax_cost.set_ylabel(r'$\Delta \varphi_2$ [MW]')
    ax_cost.set_zlabel('f')

    fig_contour = plt.figure('cost_function_contour')
    ax_contour = fig_contour.gca()
    n_levels = 20 #number of contour levels - 1
    Cplt = plt.contourf(PHI0,PHI2,f,n_levels,cmap=cm.coolwarm)
    Cbar = plt.colorbar(Cplt) # show colorbar
    ax_contour.set_xlabel(r'$\Delta \varphi_0$ [MW]')
    ax_contour.set_ylabel(r'$\Delta \varphi_2$ [MW]')

    fig_f = plt.figure('cost_function_OF')
    ax_f = fig_f.gca()
    ax_f.set_xlabel('Iteration')
    ax_f.set_ylabel('f')

    fig_To2 = plt.figure('Ts_source_2')
    ax_To2 = fig_To2.gca()
    ax_To2.set_xlabel('Iteration')
    ax_To2.set_ylabel(r'$T^s_{2,0}$ [C]')

    fig_dphi0 = plt.figure('heat_power_0')
    ax_dphi0 = fig_dphi0.gca()
    ax_dphi0.set_xlabel('Iteration')
    ax_dphi0.set_ylabel(r'$\Delta \varphi_0$ [MW]')

    fig_dphi2 = plt.figure('heat_power_2')
    ax_dphi2 = fig_dphi2.gca()
    ax_dphi2.set_xlabel('Iteration')
    ax_dphi2.set_ylabel(r'$\Delta \varphi_2$ [MW]')

    fig_dphi = plt.figure('heat_power_sources_sum')
    ax_dphi = fig_dphi.gca()
    ax_dphi.set_xlabel('Iteration')
    ax_dphi.set_ylabel(r'$\Delta \varphi_0+\Delta \varphi_2$ [MW]')

    # plot results with LF as equality constraints
    ax_f.plot(f_vec,color=colors.get('OPF'),label='OF')
    ax_To2.plot(To2_vec,color=colors.get('OPF'),label='OF')
    ax_dphi0.plot(np.array(dphi0_vec)/MW,color=colors.get('OPF'),label='OF')
    ax_dphi2.plot(np.array(dphi2_vec)/MW,color=colors.get('OPF'),label='OF')
    ax_dphi.plot((np.array(dphi0_vec)+np.array(dphi2_vec))/MW,color=colors.get('OPF'),label='OF')
    ax_cost.plot(dphi0_vec,dphi2_vec,f_vec,marker='.',ls='-',color=colors.get('OPF'),label='OF')
    ax_contour.plot(dphi0_vec,dphi2_vec,marker='.',ls='-',color=colors.get('OPF'),label='OF')
    # plot results with LF as equality constraints, using numerical derivatives
    ax_f.plot(f_vec_num_der,color=colors.get('OPF num. der.'),label='OF num. der.')
    ax_To2.plot(To2_vec_num_der,color=colors.get('OPF num. der.'),label='OF num. der.')
    ax_dphi0.plot(np.array(dphi0_vec_num_der)/MW,color=colors.get('OPF num. der.'),label='OF num. der.')
    ax_dphi2.plot(np.array(dphi2_vec_num_der)/MW,color=colors.get('OPF num. der.'),label='OF num. der.')
    ax_dphi.plot((np.array(dphi0_vec_num_der)+np.array(dphi2_vec_num_der))/MW,color=colors.get('OPF num. der.'),label='OF num. der.')
    ax_cost.plot(dphi0_vec_num_der,dphi2_vec_num_der,f_vec_num_der,marker='.',ls='-',color=colors.get('OPF num. der.'),label='OF num. der.')
    ax_contour.plot(dphi0_vec_num_der,dphi2_vec_num_der,marker='.',ls='-',color=colors.get('OPF num. der.'),label='OF num. der.')
    # plot results with LF implicit, direct approach
    ax_f.plot(f_vec_sepLF_direct ,color=colors.get('OPF direct'),label='sep. LF direct')
    ax_To2.plot(To2_vec_sepLF_direct,color=colors.get('OPF direct'),label='sep. LF direct')
    ax_dphi0.plot(np.array(dphi0_vec_sepLF_direct)/MW,color=colors.get('OPF direct'),label='sep. LF direct')
    ax_dphi2.plot(np.array(dphi2_vec_sepLF_direct)/MW,color=colors.get('OPF direct'),label='sep. LF direct')
    ax_dphi.plot((np.array(dphi0_vec_sepLF_direct)+np.array(dphi2_vec_sepLF_direct))/MW,color=colors.get('OPF direct'),label='sep. LF direct')
    ax_cost.plot(dphi0_vec_sepLF_direct,dphi2_vec_sepLF_direct,f_vec_sepLF_direct,marker='*',ls='-',color=colors.get('OPF direct'),label='sep. LF direct')
    ax_contour.plot(dphi0_vec_sepLF_direct,dphi2_vec_sepLF_direct,marker='*',ls='-',color=colors.get('OPF direct'),label='sep. LF direct')
    # plot results with LF implicit, adjoint approach
    ax_f.plot(f_vec_sepLF_adjoint ,color=colors.get('OPF adjoint'),label='sep. LF adjoint')
    ax_To2.plot(To2_vec_sepLF_adjoint,color=colors.get('OPF adjoint'),label='sep. LF adjoint')
    ax_dphi0.plot(np.array(dphi0_vec_sepLF_adjoint)/MW,color=colors.get('OPF adjoint'),label='sep. LF adjoint')
    ax_dphi2.plot(np.array(dphi2_vec_sepLF_adjoint)/MW,color=colors.get('OPF adjoint'),label='sep. LF adjoint')
    ax_dphi.plot((np.array(dphi0_vec_sepLF_adjoint)+np.array(dphi2_vec_sepLF_adjoint))/MW,color=colors.get('OPF adjoint'),label='sep. LF adjoint')
    ax_cost.plot(dphi0_vec_sepLF_adjoint,dphi2_vec_sepLF_adjoint,f_vec_sepLF_adjoint,marker='s',ls='-',color=colors.get('OPF adjoint'),label='sep. LF adjoint')
    ax_contour.plot(dphi0_vec_sepLF_adjoint,dphi2_vec_sepLF_adjoint,marker='s',ls='-',color=colors.get('OPF adjoint'),label='sep. LF adjoint')

    # layout
    ax_cost.plot([dphi0_sol],[dphi2_sol],[f_LF],label='LF',marker='.',color='k')
    ax_contour.plot([dphi0_sol],[dphi2_sol],label='LF',marker='.',color='k')
    nit_max = np.max([nit_num_der,nit,nit_sepLF_direct,nit_sepLF_adjoint])

    ax_f.plot([0,nit_max],[f_LF,f_LF],ls=':',color=colors.get('LF'),label='LF')
    ax_f.set_xlim(left=0,right=nit_max)
    ax_f.legend()
    ax_To2.plot([0,nit_max],[To2_lb,To2_lb],ls=':',color='k',label='lower bound')
    ax_To2.plot([0,nit_max],[To2_ub,To2_ub],ls='-.',color='k',label='upper bound')
    ax_To2.legend()
    ax_To2.set_xlim(left=0,right=nit_max)
    ax_dphi0.set_xlim(left=0,right=nit_max)
    ax_dphi0.legend()
    ax_dphi2.plot([0,nit_max],[phi2_lb/MW,phi2_lb/MW],ls=':',color='k',label='lower bound phi2')
    ax_dphi2.plot([0,nit_max],[phi2_ub/MW,phi2_ub/MW],ls='-.',color='k',label='upper bound phi2')
    ax_dphi2.set_xlim(left=0,right=nit_max)
    ax_dphi2.legend()
    ax_dphi.set_xlim(left=0,right=nit_max)
    ax_dphi.legend()
    ax_cost.legend()
    ticks_dphi0 = ax_cost.get_xticks()
    ax_cost.set_xticklabels(ticks_dphi0/MW)
    ticks_dphi2 = ax_cost.get_yticks()
    ax_cost.set_yticklabels(ticks_dphi2/MW)
    ax_contour.legend()
    ax_contour.set_xticklabels(ticks_dphi0/MW)
    ax_contour.set_yticklabels(ticks_dphi2/MW)

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','H3N')
        for fig_num in plt.get_figlabels():
            if not '3d' in fig_num:
                plt.figure(fig_num)
                file_name = fig_num+'.pgf'
                plt.savefig(os.path.join(path_to_fig, file_name))

def compare_opf_methods(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF for different optimization methods. Without scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    tol=1e-6
    max_iter=100
    max_iters_lf=10
    scale_var = None
    fb = None
    a0=0
    a2=0
    b0=.4
    b2=.5
    c0=4e-4
    c2=4.5e-4
    formulation = 'half_link_flow'

    #LF solution
    with HiddenPrints():
        heat_net_LF,x_LF,_,_,_,_,_,_,_,_,_,_ = run_load_flow(tol=tol,max_iter=max_iters_lf,formulation=formulation)
        dphi0_sol = heat_net_LF.nodes[0].half_links[0].get_dphi()
        m0_sol = heat_net_LF.nodes[0].half_links[0].get_m()
        dphi2_sol = heat_net_LF.nodes[2].half_links[0].get_dphi()
        To2_sol = heat_net_LF.nodes[2].half_links[0].get_Ts()
        x_opf_LF = np.concatenate((np.array([To2_sol,dphi2_sol,m0_sol,dphi0_sol]),x_LF))

    # initial guesses
    To2_init = .9*To2_sol
    phi2_init = 1.1*dphi2_sol

    # bounds
    To2_bounds=np.array([.8*To2_sol,1*To2_sol])
    phi2_bounds=np.array([1.3*dphi2_sol,dphi2_sol])
    m0_bounds=np.array([-500,0])
    phi0_bounds=np.array([1.3*dphi0_sol,dphi0_sol])
    m01_bounds=np.array([-200,200])
    m02_bounds=np.array([-200,200])
    m12_bounds=np.array([-200,200])
    m1_bounds=np.array([0,500])
    m2_bounds=np.array([-500,0])
    p1_bounds=np.array([10,6000*960*9.81])
    p2_bounds=np.array([10,6000*960*9.81])
    Ts1_bounds=np.array([60,140])
    Ts2_bounds=np.array([60,140])
    Tr0_bounds=np.array([10,60])
    Tr1_bounds=np.array([10,60])
    Tr2_bounds=np.array([10,60])

    result = dict()
    xh_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']
    ineq_constrs = [None,'control','all']

    # values used for plots
    dphi0 = np.linspace(-20*MW,-45*MW,500)
    dphi2 = np.linspace(-8.5*MW,-10*MW,100)
    PHI0, PHI2 = np.meshgrid(dphi0,dphi2)
    f = price_heat(PHI0, PHI2,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2)

    # Optimal Flow
    for ineq_constr in ineq_constrs:
        # plots
        fig_contour = plt.figure('obj_contour_ineq_constr_{}'.format(ineq_constr))
        ax_contour = fig_contour.gca()
        n_levels = 20 #number of contour levels - 1
        Cplt = plt.contourf(PHI0,PHI2,f,n_levels,cmap=cm.coolwarm)
        Cbar = plt.colorbar(Cplt) # show colorbar
        ax_contour.set_xlabel(r'$\Delta \varphi_0$ [W]')
        ax_contour.set_ylabel(r'$\Delta \varphi_2$ [W]')

        fig_f = plt.figure('obj_OF_ineq_constr_{}'.format(ineq_constr))
        ax_f = fig_f.gca()
        ax_f.set_xlabel('Iteration')
        ax_f.set_ylabel('f')

        fig_To2 = plt.figure('Ts_source_2_ineq_constr_{}'.format(ineq_constr))
        ax_To2 = fig_To2.gca()
        ax_To2.set_xlabel('Iteration')
        ax_To2.set_ylabel(r'$T^s_{2,0}$ [C]')

        fig_dphi0 = plt.figure('heat_power_0_ineq_constr_{}'.format(ineq_constr))
        ax_dphi0 = fig_dphi0.gca()
        ax_dphi0.set_xlabel('Iteration')
        ax_dphi0.set_ylabel(r'$\Delta \varphi_0$ [W]')

        fig_dphi2 = plt.figure('heat_power_2_ineq_constr_{}'.format(ineq_constr))
        ax_dphi2 = fig_dphi2.gca()
        ax_dphi2.set_xlabel('Iteration')
        ax_dphi2.set_ylabel(r'$\Delta \varphi_2$ [W]')

        max_fev = 0
        for method in methods:
            for bound in bounds:
                if bound == 'soft':
                    stay_within_bounds = False
                else:
                    stay_within_bounds = True
                for der in ders:
                    if der == 'an':
                        derivatives = True
                    else:
                        derivatives = False
                    xh_opt, res, f_vec, To2_vec, dphi0_vec, dphi2_vec, execution_time = run_optimal_load_flow(To2_init=To2_init,To2_bounds=To2_bounds,phi2_init=phi2_init,phi2_bounds=phi2_bounds,m0_bounds=m0_bounds,phi0_bounds=phi0_bounds,m01_bounds=m01_bounds,m02_bounds=m02_bounds,m12_bounds=m12_bounds,m1_bounds=m1_bounds,m2_bounds=m2_bounds,p1_bounds=p1_bounds,p2_bounds=p2_bounds,Ts1_bounds=Ts1_bounds,Ts2_bounds=Ts2_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,Tr2_bounds=Tr2_bounds,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,scale_var=scale_var,scale_var_params=None,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,formulation=formulation,ineq_constr=ineq_constr,derivatives=derivatives,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb)
                    # save result in dictionaries
                    result[method+'_'+bound+'_'+der+'_{}'.format(ineq_constr)] = res
                    xh_res[method+'_'+bound+'_'+der+'_{}'.format(ineq_constr)] = xh_opt
                    max_fev = max(max_fev,len(f_vec))
                    # plot results
                    ax_contour.plot(dphi0_vec,dphi2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der),label=method+' '+bound+' '+der)
                    ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_To2.plot(To2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_dphi0.plot(dphi0_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_dphi2.plot(dphi2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))

        ax_contour.plot([dphi0_sol],[dphi2_sol],'.r')
        ax_f.plot([0,max_fev],[price_heat(dphi0_sol, dphi2_sol,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2),price_heat(dphi0_sol, dphi2_sol,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2)],':r')
        ax_To2.plot([0,max_fev],[To2_bounds[0],To2_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_To2.plot([0,max_fev],[To2_bounds[1],To2_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_To2.plot([0,max_fev],[To2_sol,To2_sol],':r')
        ax_dphi0.plot([0,max_fev],[phi0_bounds[0],phi0_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0.plot([0,max_fev],[phi0_bounds[1],phi0_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0.plot([0,max_fev],[dphi0_sol,dphi0_sol],':r')
        ax_dphi2.plot([0,max_fev],[phi2_bounds[0],phi2_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi2.plot([0,max_fev],[phi2_bounds[1],phi2_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi2.plot([0,max_fev],[dphi2_sol,dphi2_sol],':r')

        ax_contour.legend(handles=legend_handles)
        ax_f.legend(handles=legend_handles)
        ax_To2.legend(handles=legend_handles)
        ax_dphi0.legend(handles=legend_handles)
        ax_dphi2.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','H3N')
        for bound in bounds:
            for der in ders:
                with open(os.path.join(path_to_tables,'network_solution_errors_methods_'+bound+'_'+der+'.txt'), "w") as table:
                    variable_names = [r'$T^s_{2,0}$',r'$\Delta \varphi_2$',r'$m_0$',r'$\Delta \varphi_0$',r'$m_{01}$',r'$m_{02}$',r'$m_{12}$',r'$m_{1}$',r'$m_{2}$',r'$p_1$',r'$p_2$',r'$T^s_1$',r'$T^s_2$',r'$T^r_0$',r'$T^r_1$',r'$T^r_2$']
                    res_trust_None = result.get('trust-constr_'+bound+'_'+der+'_None')
                    res_slsqp_None = result.get('SLSQP_'+bound+'_'+der+'_None')
                    res_ipopt_None = result.get('ipopt_'+bound+'_'+der+'_None')
                    res_trust_control = result.get('trust-constr_'+bound+'_'+der+'_control')
                    res_slsqp_control = result.get('SLSQP_'+bound+'_'+der+'_control')
                    res_ipopt_control = result.get('ipopt_'+bound+'_'+der+'_control')
                    res_trust_all = result.get('trust-constr_'+bound+'_'+der+'_all')
                    res_slsqp_all = result.get('SLSQP_'+bound+'_'+der+'_all')
                    res_ipopt_all = result.get('ipopt_'+bound+'_'+der+'_all')
                    for ind_var, var in enumerate(variable_names):
                        table.write(r'{} & {:.3e} & {:.3e}  & {:.3e}  & {:.3e} & {:.3e}  & {:.3e}  & {:.3e} & {:.3e}  & {:.3e}  & {:.3e} \\ '.format(var,x_opf_LF[ind_var],error(res_trust_None.x[ind_var],x_opf_LF[ind_var]),error(res_slsqp_None.x[ind_var],x_opf_LF[ind_var]),error(res_ipopt_None.x[ind_var],x_opf_LF[ind_var]),error(res_trust_control.x[ind_var],x_opf_LF[ind_var]),error(res_slsqp_control.x[ind_var],x_opf_LF[ind_var]),error(res_ipopt_control.x[ind_var],x_opf_LF[ind_var]),error(res_trust_all.x[ind_var],x_opf_LF[ind_var]),error(res_slsqp_all.x[ind_var],x_opf_LF[ind_var]),error(res_ipopt_all.x[ind_var],x_opf_LF[ind_var])))
        with open(os.path.join(path_to_tables,'optimizer_info_methods.txt'), "w") as table:
            for ineq_constr in ineq_constrs:
                for bound in bounds:
                    for der in ders:
                        res_trust = result.get('trust-constr_'+bound+'_'+der+'_{}'.format(ineq_constr))
                        res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_{}'.format(ineq_constr))
                        res_ipopt = result.get('ipopt_'+bound+'_'+der+'_{}'.format(ineq_constr))
                        table.write(r'{} & {} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(ineq_constr,bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(res_trust.x,x_opf_LF),error(res_slsqp.x,x_opf_LF),error(res_ipopt.x,x_opf_LF)))
                        print('\nConstraints: {}, bounds: {}, der: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\ntrust-constr:{}\nSLSQP: {}\nIPOPT: {}'.format(ineq_constr,bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message))
                table.write(r'\hline ')

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','H3N')
        for fig_num in plt.get_figlabels():
            if not '3d' in fig_num:
                plt.figure(fig_num)
                file_name = fig_num+'.pgf'
                plt.savefig(os.path.join(path_to_fig, file_name))

def compare_opf_scaling(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF for different scaling options, and for different optimization methods."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    tol=1e-6
    max_iter=100
    max_iters_lf=10
    a0=0
    a2=0
    b0=.4
    b2=.5
    c0=4e-4
    c2=4.5e-4
    formulation = 'half_link_flow'

    #LF solution, unscaled
    with HiddenPrints():
        heat_net_LF,x_LF,_,_,_,_,_,_,_,_,_,_ = run_load_flow(tol=tol,max_iter=max_iters_lf,formulation=formulation)
        dphi0_sol = heat_net_LF.nodes[0].half_links[0].get_dphi()
        m0_sol = heat_net_LF.nodes[0].half_links[0].get_m()
        dphi2_sol = heat_net_LF.nodes[2].half_links[0].get_dphi()
        To2_sol = heat_net_LF.nodes[2].half_links[0].get_Ts()

    # initial guesses
    To2_init = .95*To2_sol
    phi2_init = 1.2*dphi2_sol

    # bounds
    ineq_constr = 'all'
    To2_bounds=np.array([.9*To2_sol,1*To2_sol])
    phi2_bounds=np.array([1.3*dphi2_sol,dphi2_sol])
    m0_bounds=np.array([-500,0])
    phi0_bounds=np.array([1.3*dphi0_sol,dphi0_sol])
    m01_bounds=np.array([-200,200])
    m02_bounds=np.array([-200,200])
    m12_bounds=np.array([-200,200])
    m1_bounds=np.array([0,500])
    m2_bounds=np.array([-500,0])
    p1_bounds=np.array([10,6000*960*9.81])
    p2_bounds=np.array([10,6000*960*9.81])
    Ts1_bounds=np.array([60,140])
    Ts2_bounds=np.array([60,140])
    Tr0_bounds=np.array([10,60])
    Tr1_bounds=np.array([10,60])
    Tr2_bounds=np.array([10,60])

    # base values
    water = heat_net_LF.links[0].link_params.get('carrier')
    rho = water.rhon
    g = water.g
    phibase = 1.*MW #[W]
    Tbase = 100.#[C]
    mbase = 1.
    pbase = 5517*rho*g
    scale_var_params = {'mbase':mbase,'phbase':pbase,'phibase':phibase,'Tbase':Tbase,'qbase':mbase,'pbase':pbase}
    fb = 1000.*MW

    x_opf_LF = np.concatenate((np.array([To2_sol/scale_var_params.get('Tbase'),dphi2_sol/scale_var_params.get('phibase'),m0_sol/scale_var_params.get('mbase'),dphi0_sol/scale_var_params.get('phibase')]),x_LF/np.array([scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('pbase'),scale_var_params.get('pbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase')]))) # scaled

    result = dict()
    xh_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']
    scaling = ['matrix','per_unit']

    # Optimal Flow
    for scale_var in scaling:
        # values used for plots
        dphi0 = np.linspace(-20*MW,-45*MW,500)
        dphi2 = np.linspace(-8.5*MW,-10*MW,100)
        if scale_var == 'per_unit':
            dphi0 = dphi0/scale_var_params.get('phibase')
            dphi2 = dphi2/scale_var_params.get('phibase')
        PHI0, PHI2 = np.meshgrid(dphi0,dphi2)
        if scale_var == 'per_unit':
            f = price_heat(PHI0/scale_var_params.get('phibase'), PHI2/scale_var_params.get('phibase'),a0=a0/fb,a2=a2/fb,b0=b0/(fb/scale_var_params.get('phibase')),b2=b2/(fb/scale_var_params.get('phibase')),c0=c0/(fb/scale_var_params.get('phibase')**2),c2=c2/(fb/scale_var_params.get('phibase')**2))
        else:
            f = price_heat(PHI0, PHI2,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2)/fb
        # plots
        fig_contour = plt.figure('obj_contour_scaling_{}'.format(scale_var))
        ax_contour = fig_contour.gca()
        n_levels = 20 #number of contour levels - 1
        Cplt = plt.contourf(PHI0/scale_var_params.get('phibase'),PHI2/scale_var_params.get('phibase'),f,n_levels,cmap=cm.coolwarm)
        Cbar = plt.colorbar(Cplt) # show colorbar
        ax_contour.set_xlabel(r'$\Delta \varphi_0$')
        ax_contour.set_ylabel(r'$\Delta \varphi_2$')

        fig_f = plt.figure('obj_OF_scaling_{}'.format(scale_var))
        ax_f = fig_f.gca()
        ax_f.set_xlabel('Iteration')
        ax_f.set_ylabel('f')

        fig_To2 = plt.figure('Ts_source_2_scaling_{}'.format(scale_var))
        ax_To2 = fig_To2.gca()
        ax_To2.set_xlabel('Iteration')
        ax_To2.set_ylabel(r'$T^s_{2,0}$')

        fig_dphi0 = plt.figure('heat_power_0_scaling_{}'.format(scale_var))
        ax_dphi0 = fig_dphi0.gca()
        ax_dphi0.set_xlabel('Iteration')
        ax_dphi0.set_ylabel(r'$\Delta \varphi_0$')

        fig_dphi2 = plt.figure('heat_power_2_scaling_{}'.format(scale_var))
        ax_dphi2 = fig_dphi2.gca()
        ax_dphi2.set_xlabel('Iteration')
        ax_dphi2.set_ylabel(r'$\Delta \varphi_2$')

        max_fev = 0
        for method in methods:
            for bound in bounds:
                if bound == 'soft':
                    stay_within_bounds = False
                else:
                    stay_within_bounds = True
                for der in ders:
                    if der == 'an':
                        derivatives = True
                    else:
                        derivatives = False
                    xh_opt, res, f_vec, To2_vec, dphi0_vec, dphi2_vec, execution_time = run_optimal_load_flow(To2_init=To2_init,To2_bounds=To2_bounds,phi2_init=phi2_init,phi2_bounds=phi2_bounds,m0_bounds=m0_bounds,phi0_bounds=phi0_bounds,m01_bounds=m01_bounds,m02_bounds=m02_bounds,m12_bounds=m12_bounds,m1_bounds=m1_bounds,m2_bounds=m2_bounds,p1_bounds=p1_bounds,p2_bounds=p2_bounds,Ts1_bounds=Ts1_bounds,Ts2_bounds=Ts2_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,Tr2_bounds=Tr2_bounds,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,formulation=formulation,ineq_constr=ineq_constr,derivatives=derivatives,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb)
                    # save result in dictionaries
                    result[method+'_'+bound+'_'+der+'_'+scale_var] = res
                    xh_res[method+'_'+bound+'_'+der+'_'+scale_var] = xh_opt
                    max_fev = max(max_fev,len(f_vec))
                    # plot results
                    ax_contour.plot(dphi0_vec,dphi2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der),label=method+' '+bound+' '+der)
                    ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_To2.plot(To2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_dphi0.plot(dphi0_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_dphi2.plot(dphi2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))

        ax_contour.plot([dphi0_sol/scale_var_params.get('phibase')],[dphi2_sol/scale_var_params.get('phibase')],'.r')

        if scale_var == 'per_unit':
            ax_f.plot([0,max_fev],[price_heat(dphi0_sol, dphi2_sol,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2)/fb,price_heat(dphi0_sol, dphi2_sol,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2)/fb],':r')
        else:
            ax_f.plot([0,max_fev],[price_heat(dphi0_sol/scale_var_params.get('phibase'), dphi2_sol/scale_var_params.get('phibase'),a0=a0/fb,a2=a2/fb,b0=b0/(fb/scale_var_params.get('phibase')),b2=b2/(fb/scale_var_params.get('phibase')),c0=c0/(fb/scale_var_params.get('phibase')**2),c2=c2/(fb/scale_var_params.get('phibase')**2)),price_heat(dphi0_sol/scale_var_params.get('phibase'), dphi2_sol/scale_var_params.get('phibase'),a0=a0/fb,a2=a2/fb,b0=b0/(fb/scale_var_params.get('phibase')),b2=b2/(fb/scale_var_params.get('phibase')),c0=c0/(fb/scale_var_params.get('phibase')**2),c2=c2/(fb/scale_var_params.get('phibase')**2))],':r')

        ax_To2.plot([0,max_fev],[To2_bounds[0]/scale_var_params.get('Tbase'),To2_bounds[0]/scale_var_params.get('Tbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_To2.plot([0,max_fev],[To2_bounds[1]/scale_var_params.get('Tbase'),To2_bounds[1]/scale_var_params.get('Tbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_To2.plot([0,max_fev],[To2_sol/scale_var_params.get('Tbase'),To2_sol/scale_var_params.get('Tbase')],':r')
        ax_dphi0.plot([0,max_fev],[phi0_bounds[0]/scale_var_params.get('phibase'),phi0_bounds[0]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0.plot([0,max_fev],[phi0_bounds[1]/scale_var_params.get('phibase'),phi0_bounds[1]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0.plot([0,max_fev],[dphi0_sol/scale_var_params.get('phibase'),dphi0_sol/scale_var_params.get('phibase')],':r')
        ax_dphi2.plot([0,max_fev],[phi2_bounds[0]/scale_var_params.get('phibase'),phi2_bounds[0]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi2.plot([0,max_fev],[phi2_bounds[1]/scale_var_params.get('phibase'),phi2_bounds[1]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi2.plot([0,max_fev],[dphi2_sol/scale_var_params.get('phibase'),dphi2_sol/scale_var_params.get('phibase')],':r')

        ax_contour.legend(handles=legend_handles)
        ax_f.legend(handles=legend_handles)
        ax_To2.legend(handles=legend_handles)
        ax_dphi0.legend(handles=legend_handles)
        ax_dphi2.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','H3N')
        with open(os.path.join(path_to_tables,'optimizer_info_methods_scaling.txt'), "w") as table:
            for bound in bounds:
                for der in ders:
                    res_trust_mat = result.get('trust-constr_'+bound+'_'+der+'_matrix')
                    res_slsqp_mat = result.get('SLSQP_'+bound+'_'+der+'_matrix')
                    res_ipopt_mat = result.get('ipopt_'+bound+'_'+der+'_matrix')
                    res_trust_pu = result.get('trust-constr_'+bound+'_'+der+'_per_unit')
                    res_slsqp_pu = result.get('SLSQP_'+bound+'_'+der+'_per_unit')
                    res_ipopt_pu = result.get('ipopt_'+bound+'_'+der+'_per_unit')
                    table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} \\ '.format(bound,der,res_trust_mat.success,res_trust_pu.success,res_slsqp_mat.success,res_slsqp_pu.success,res_ipopt_mat.success,res_ipopt_pu.success,res_trust_mat.nit,res_trust_pu.nit,res_slsqp_pu.nit,res_slsqp_mat.nit,res_ipopt_mat.nit,res_ipopt_pu.nit,error(res_trust_mat.x,x_opf_LF),error(res_trust_pu.x,x_opf_LF),error(res_slsqp_mat.x,x_opf_LF),error(res_slsqp_pu.x,x_opf_LF),error(res_ipopt_mat.x,x_opf_LF),error(res_ipopt_pu.x,x_opf_LF)))
    for bound in bounds:
        for der in ders:
            for scale_var in scaling:
                res_trust = result.get('trust-constr_'+bound+'_'+der+'_'+scale_var)
                res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_'+scale_var)
                res_ipopt = result.get('ipopt_'+bound+'_'+der+'_'+scale_var)
                print('\nBounds: {}, der: {}, scaling: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\ntrust-constr:{}\nSLSQP: {}\nIPOPT: {}'.format(bound,der,scale_var,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','H3N')
        for fig_num in plt.get_figlabels():
            if not '3d' in fig_num:
                plt.figure(fig_num)
                file_name = fig_num+'.pgf'
                plt.savefig(os.path.join(path_to_fig, file_name))

def compare_opf_methods_sep_LF(dir_path=None,save_tables=False,save_figs=False,To2_factor = 0.9,phi2_factor = 1.1,phi0_factor_ub = 0.5):
    """Compare OPF with LF substituted for different optimization methods. Without scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    tol=1e-6
    max_iter=100
    max_iters_lf=10
    scale_var = None
    fb = None
    a0=0
    a2=0
    b0=.4
    b2=.5
    c0=4e-4
    c2=4.5e-4
    formulation = 'half_link_flow'

    #LF solution
    with HiddenPrints():
        heat_net_LF,xh_LF,_,_,_,_,_,_,_,_,_,_ = run_load_flow(tol=tol,max_iter=max_iters_lf,formulation=formulation)
        dphi0_sol = heat_net_LF.nodes[0].half_links[0].get_dphi()
        m0_sol = heat_net_LF.nodes[0].half_links[0].get_m()
        dphi2_sol = heat_net_LF.nodes[2].half_links[0].get_dphi()
        To2_sol = heat_net_LF.nodes[2].half_links[0].get_Ts()
        x_opf_LF = np.concatenate((np.array([To2_sol,dphi2_sol,m0_sol,dphi0_sol]),xh_LF))

    # initial guesses
    To2_init = To2_factor*To2_sol
    phi2_init = phi2_factor*dphi2_sol

    # bounds
    ineq_constr='all'
    To2_bounds=np.array([.8*To2_sol,1*To2_sol])
    phi2_bounds=np.array([1.3*dphi2_sol,dphi2_sol])
    m0_bounds=np.array([-500,0])
    phi0_bounds=np.array([1.3*dphi0_sol,phi0_factor_ub*dphi0_sol])
    m01_bounds=np.array([-200,200])
    m02_bounds=np.array([-200,200])
    m12_bounds=np.array([-200,200])
    m1_bounds=np.array([0,500])
    m2_bounds=np.array([-500,0])
    p1_bounds=np.array([10,6000*960*9.81])
    p2_bounds=np.array([10,6000*960*9.81])
    Ts1_bounds=np.array([60,140])
    Ts2_bounds=np.array([60,140])
    Tr0_bounds=np.array([10,60])
    Tr1_bounds=np.array([10,60])
    Tr2_bounds=np.array([10,60])

    result = dict()
    xh_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']

    # values used for plots
    dphi0 = np.linspace(-20*MW,-45*MW,500)
    dphi2 = np.linspace(-8.5*MW,-10*MW,100)
    PHI0, PHI2 = np.meshgrid(dphi0,dphi2)
    f = price_heat(PHI0, PHI2,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2)
    # plots
    fig_contour = plt.figure('obj_contour_sep_LF_{}To2_{}phi2_{}phi0ub'.format(To2_factor,phi2_factor,phi0_factor_ub))
    ax_contour = fig_contour.gca()
    n_levels = 20 #number of contour levels - 1
    Cplt = plt.contourf(PHI0,PHI2,f,n_levels,cmap=cm.coolwarm)
    Cbar = plt.colorbar(Cplt) # show colorbar
    ax_contour.set_xlabel(r'$\Delta \varphi_0$ [W]')
    ax_contour.set_ylabel(r'$\Delta \varphi_2$ [W]')

    fig_f = plt.figure('obj_OF_sep_LF_{}To2_{}phi2_{}phi0ub'.format(To2_factor,phi2_factor,phi0_factor_ub))
    ax_f = fig_f.gca()
    ax_f.set_xlabel('Iteration')
    ax_f.set_ylabel('f')

    fig_To2 = plt.figure('Ts_source_2_sep_LF_{}To2_{}phi2_{}phi0ub'.format(To2_factor,phi2_factor,phi0_factor_ub))
    ax_To2 = fig_To2.gca()
    ax_To2.set_xlabel('Iteration')
    ax_To2.set_ylabel(r'$T^s_{2,0}$ [C]')

    fig_dphi0 = plt.figure('heat_power_0_sep_LF_{}To2_{}phi2_{}phi0ub'.format(To2_factor,phi2_factor,phi0_factor_ub))
    ax_dphi0 = fig_dphi0.gca()
    ax_dphi0.set_xlabel('Iteration')
    ax_dphi0.set_ylabel(r'$\Delta \varphi_0$ [W]')

    fig_dphi2 = plt.figure('heat_power_2_sep_LF_{}To2_{}phi2_{}phi0ub'.format(To2_factor,phi2_factor,phi0_factor_ub))
    ax_dphi2 = fig_dphi2.gca()
    ax_dphi2.set_xlabel('Iteration')
    ax_dphi2.set_ylabel(r'$\Delta \varphi_2$ [W]')

    max_fev = 0

    for method in methods:
        for bound in bounds:
            if bound == 'soft':
                stay_within_bounds = False
            else:
                stay_within_bounds = True
            for approach in approaches:
                if approach == 'direct' or approach == 'adjoint':
                    approach_legend = approach
                    xh_opt, res, f_vec, To2_vec, dphi0_vec, dphi2_vec, execution_time = run_optimal_load_flow_separate_LF(To2_init=To2_init,To2_bounds=To2_bounds,phi2_init=phi2_init,phi2_bounds=phi2_bounds,m0_bounds=m0_bounds,phi0_bounds=phi0_bounds,m01_bounds=m01_bounds,m02_bounds=m02_bounds,m12_bounds=m12_bounds,m1_bounds=m1_bounds,m2_bounds=m2_bounds,p1_bounds=p1_bounds,p2_bounds=p2_bounds,Ts1_bounds=Ts1_bounds,Ts2_bounds=Ts2_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,Tr2_bounds=Tr2_bounds,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,scale_var=scale_var,scale_var_params=None,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,formulation=formulation,ineq_constr=ineq_constr,optimization_method=method,approach=approach,stay_within_bounds=stay_within_bounds,fb=fb)
                else:
                    approach_legend = 'an'
                    xh_opt, res, f_vec, To2_vec, dphi0_vec, dphi2_vec, execution_time = run_optimal_load_flow(To2_init=To2_init,To2_bounds=To2_bounds,phi2_init=phi2_init,phi2_bounds=phi2_bounds,m0_bounds=m0_bounds,phi0_bounds=phi0_bounds,m01_bounds=m01_bounds,m02_bounds=m02_bounds,m12_bounds=m12_bounds,m1_bounds=m1_bounds,m2_bounds=m2_bounds,p1_bounds=p1_bounds,p2_bounds=p2_bounds,Ts1_bounds=Ts1_bounds,Ts2_bounds=Ts2_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,Tr2_bounds=Tr2_bounds,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,scale_var=scale_var,scale_var_params=None,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,formulation=formulation,ineq_constr=ineq_constr,derivatives=True,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb)
                # save result in dictionaries
                result[method+'_'+bound+'_'+approach] = res
                xh_res[method+'_'+bound+'_'+approach] = xh_opt
                max_fev = max(max_fev,len(f_vec))
                # plot results
                ax_contour.plot(dphi0_vec,dphi2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend))
                ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                ax_To2.plot(To2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                ax_dphi0.plot(dphi0_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                ax_dphi2.plot(dphi2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
    ax_contour.plot([dphi0_sol],[dphi2_sol],'.r')
    ax_f.plot([0,max_fev],[price_heat(dphi0_sol, dphi2_sol,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2),price_heat(dphi0_sol, dphi2_sol,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2)],':r')
    ax_To2.plot([0,max_fev],[To2_bounds[0],To2_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_To2.plot([0,max_fev],[To2_bounds[1],To2_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_To2.plot([0,max_fev],[To2_sol,To2_sol],':r')
    ax_dphi0.plot([0,max_fev],[phi0_bounds[0],phi0_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_dphi0.plot([0,max_fev],[phi0_bounds[1],phi0_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_dphi0.plot([0,max_fev],[dphi0_sol,dphi0_sol],':r')
    ax_dphi2.plot([0,max_fev],[phi2_bounds[0],phi2_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_dphi2.plot([0,max_fev],[phi2_bounds[1],phi2_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_dphi2.plot([0,max_fev],[dphi2_sol,dphi2_sol],':r')

    ax_contour.legend(handles=legend_handles)
    ax_f.legend(handles=legend_handles)
    ax_To2.legend(handles=legend_handles)
    ax_dphi0.legend(handles=legend_handles)
    ax_dphi2.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','H3N')
        with open(os.path.join(path_to_tables,'optimizer_info_methods_sep_LF_{}To2_{}phi2_{}phi0ub.txt'.format(To2_factor,phi2_factor,phi0_factor_ub)), "w") as table:
            for bound in bounds:
                for approach in approaches:
                    if approach == 'eq_constr':
                        approach_label = 'eq. constr.'
                    else:
                        approach_label = approach
                    xh_opt_trust = xh_res.get('trust-constr_'+bound+'_'+approach)
                    xh_opt_slsqp = xh_res.get('SLSQP_'+bound+'_'+approach)
                    xh_opt_ipopt = xh_res.get('ipopt_'+bound+'_'+approach)
                    res_trust = result.get('trust-constr_'+bound+'_'+approach)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+approach)
                    res_ipopt = result.get('ipopt_'+bound+'_'+approach)
                    table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(bound,approach_label,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(xh_opt_trust,xh_LF),error(xh_opt_slsqp,xh_LF),error(xh_opt_ipopt,xh_LF)))
                table.write(r'\hline ')

    for bound in bounds:
        for approach in approaches:
            xh_opt_trust = xh_res.get('trust-constr_'+bound+'_'+approach)
            xh_opt_slsqp = xh_res.get('SLSQP_'+bound+'_'+approach)
            xh_opt_ipopt = xh_res.get('ipopt_'+bound+'_'+approach)
            res_trust = result.get('trust-constr_'+bound+'_'+approach)
            res_slsqp = result.get('SLSQP_'+bound+'_'+approach)
            res_ipopt = result.get('ipopt_'+bound+'_'+approach)
            print('\nBounds: {}, approach: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\ntrust-constr:{}\nSLSQP: {}\nIPOPT: {}\nErrors for t-c: {}, SLSQP: {}, IPOPT: {}'.format(bound,approach,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(xh_opt_trust,xh_LF),error(xh_opt_slsqp,xh_LF),error(xh_opt_ipopt,xh_LF)))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','H3N')
        for fig_num in plt.get_figlabels():
            if not '3d' in fig_num:
                plt.figure(fig_num)
                file_name = fig_num+'.pgf'
                plt.savefig(os.path.join(path_to_fig, file_name))

def compare_opf_scaling_sep_LF(dir_path=None,save_tables=False,save_figs=False,To2_factor = 0.9,phi2_factor = 1.1,phi0_factor_ub = 0.5):
    """Compare OPF for for different scaling options, and for different optimization methods. LF is substituted."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    tol=1e-6
    max_iter=100
    max_iters_lf=10
    a0=0
    a2=0
    b0=.4
    b2=.5
    c0=4e-4
    c2=4.5e-4
    formulation = 'half_link_flow'

    #LF solution
    with HiddenPrints():
        heat_net_LF,xh_LF,_,_,_,_,_,_,_,_,_,_ = run_load_flow(tol=tol,max_iter=max_iters_lf,formulation=formulation)
        dphi0_sol = heat_net_LF.nodes[0].half_links[0].get_dphi()
        m0_sol = heat_net_LF.nodes[0].half_links[0].get_m()
        dphi2_sol = heat_net_LF.nodes[2].half_links[0].get_dphi()
        To2_sol = heat_net_LF.nodes[2].half_links[0].get_Ts()
        x_opf_LF = np.concatenate((np.array([To2_sol,dphi2_sol,m0_sol,dphi0_sol]),xh_LF))

    # initial guesses
    To2_init = To2_factor*To2_sol
    phi2_init = phi2_factor*dphi2_sol

    # bounds
    ineq_constr='all'
    To2_bounds=np.array([.8*To2_sol,1*To2_sol])
    phi2_bounds=np.array([1.3*dphi2_sol,dphi2_sol])
    m0_bounds=np.array([-500,0])
    phi0_bounds=np.array([1.3*dphi0_sol,phi0_factor_ub*dphi0_sol])
    m01_bounds=np.array([-200,200])
    m02_bounds=np.array([-200,200])
    m12_bounds=np.array([-200,200])
    m1_bounds=np.array([0,500])
    m2_bounds=np.array([-500,0])
    p1_bounds=np.array([10,6000*960*9.81])
    p2_bounds=np.array([10,6000*960*9.81])
    Ts1_bounds=np.array([60,140])
    Ts2_bounds=np.array([60,140])
    Tr0_bounds=np.array([10,60])
    Tr1_bounds=np.array([10,60])
    Tr2_bounds=np.array([10,60])

    # base values
    water = heat_net_LF.links[0].link_params.get('carrier')
    rho = water.rhon
    g = water.g
    phibase = 1.*MW #[W]
    Tbase = 100.#[C]
    mbase = 1.
    pbase = 5517*rho*g
    scale_var_params = {'mbase':mbase,'phbase':pbase,'phibase':phibase,'Tbase':Tbase,'qbase':mbase,'pbase':pbase}
    fb = 1000.*MW

    result = dict()
    xh_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']
    scaling = ['matrix','per_unit']

    for scale_var in scaling:
        # values used for plots
        dphi0 = np.linspace(-20*MW,-45*MW,500)
        dphi2 = np.linspace(-8.5*MW,-10*MW,100)
        if scale_var == 'per_unit':
            dphi0 = dphi0/scale_var_params.get('phibase')
            dphi2 = dphi2/scale_var_params.get('phibase')
        PHI0, PHI2 = np.meshgrid(dphi0,dphi2)
        if scale_var == 'per_unit':
            f = price_heat(PHI0/scale_var_params.get('phibase'), PHI2/scale_var_params.get('phibase'),a0=a0/fb,a2=a2/fb,b0=b0/(fb/scale_var_params.get('phibase')),b2=b2/(fb/scale_var_params.get('phibase')),c0=c0/(fb/scale_var_params.get('phibase')**2),c2=c2/(fb/scale_var_params.get('phibase')**2))
        else:
            f = price_heat(PHI0, PHI2,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2)/fb
        # plots
        fig_contour = plt.figure('obj_contour_sep_LF_{}To2_{}phi2_{}phi0ub_{}'.format(To2_factor,phi2_factor,phi0_factor_ub,scale_var))
        ax_contour = fig_contour.gca()
        n_levels = 20 #number of contour levels - 1
        Cplt = plt.contourf(PHI0,PHI2,f,n_levels,cmap=cm.coolwarm)
        Cbar = plt.colorbar(Cplt) # show colorbar
        ax_contour.set_xlabel(r'$\Delta \varphi_0$ [W]')
        ax_contour.set_ylabel(r'$\Delta \varphi_2$ [W]')

        fig_f = plt.figure('obj_OF_sep_LF_{}To2_{}phi2_{}phi0ub_{}'.format(To2_factor,phi2_factor,phi0_factor_ub,scale_var))
        ax_f = fig_f.gca()
        ax_f.set_xlabel('Iteration')
        ax_f.set_ylabel('f')

        fig_To2 = plt.figure('Ts_source_2_sep_LF_{}To2_{}phi2_{}phi0ub_{}'.format(To2_factor,phi2_factor,phi0_factor_ub,scale_var))
        ax_To2 = fig_To2.gca()
        ax_To2.set_xlabel('Iteration')
        ax_To2.set_ylabel(r'$T^s_{2,0}$ [C]')

        fig_dphi0 = plt.figure('heat_power_0_sep_LF_{}To2_{}phi2_{}phi0ub_{}'.format(To2_factor,phi2_factor,phi0_factor_ub,scale_var))
        ax_dphi0 = fig_dphi0.gca()
        ax_dphi0.set_xlabel('Iteration')
        ax_dphi0.set_ylabel(r'$\Delta \varphi_0$ [W]')

        fig_dphi2 = plt.figure('heat_power_2_sep_LF_{}To2_{}phi2_{}phi0ub_{}'.format(To2_factor,phi2_factor,phi0_factor_ub,scale_var))
        ax_dphi2 = fig_dphi2.gca()
        ax_dphi2.set_xlabel('Iteration')
        ax_dphi2.set_ylabel(r'$\Delta \varphi_2$ [W]')

        max_fev = 0

        for method in methods:
            for bound in bounds:
                if bound == 'soft':
                    stay_within_bounds = False
                else:
                    stay_within_bounds = True
                for approach in approaches:
                    if approach == 'direct' or approach == 'adjoint':
                        approach_legend = approach
                        xh_opt, res, f_vec, To2_vec, dphi0_vec, dphi2_vec, execution_time = run_optimal_load_flow_separate_LF(To2_init=To2_init,To2_bounds=To2_bounds,phi2_init=phi2_init,phi2_bounds=phi2_bounds,m0_bounds=m0_bounds,phi0_bounds=phi0_bounds,m01_bounds=m01_bounds,m02_bounds=m02_bounds,m12_bounds=m12_bounds,m1_bounds=m1_bounds,m2_bounds=m2_bounds,p1_bounds=p1_bounds,p2_bounds=p2_bounds,Ts1_bounds=Ts1_bounds,Ts2_bounds=Ts2_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,Tr2_bounds=Tr2_bounds,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,formulation=formulation,ineq_constr=ineq_constr,optimization_method=method,approach=approach,stay_within_bounds=stay_within_bounds,fb=fb)
                    else:
                        approach_legend = 'an'
                        xh_opt, res, f_vec, To2_vec, dphi0_vec, dphi2_vec, execution_time = run_optimal_load_flow(To2_init=To2_init,To2_bounds=To2_bounds,phi2_init=phi2_init,phi2_bounds=phi2_bounds,m0_bounds=m0_bounds,phi0_bounds=phi0_bounds,m01_bounds=m01_bounds,m02_bounds=m02_bounds,m12_bounds=m12_bounds,m1_bounds=m1_bounds,m2_bounds=m2_bounds,p1_bounds=p1_bounds,p2_bounds=p2_bounds,Ts1_bounds=Ts1_bounds,Ts2_bounds=Ts2_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,Tr2_bounds=Tr2_bounds,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,formulation=formulation,ineq_constr=ineq_constr,derivatives=True,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb)
                    # save result in dictionaries
                    result[method+'_'+bound+'_'+approach+'_'+scale_var] = res
                    xh_res[method+'_'+bound+'_'+approach+'_'+scale_var] = xh_opt
                    max_fev = max(max_fev,len(f_vec))
                    # plot results
                    ax_contour.plot(dphi0_vec,dphi2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend))
                    ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_To2.plot(To2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_dphi0.plot(dphi0_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_dphi2.plot(dphi2_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
        ax_contour.plot([dphi0_sol/scale_var_params.get('phibase')],[dphi2_sol/scale_var_params.get('phibase')],'.r')
        if scale_var == 'per_unit':
            ax_f.plot([0,max_fev],[price_heat(dphi0_sol, dphi2_sol,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2)/fb,price_heat(dphi0_sol, dphi2_sol,a0=a0,a2=a2,b0=b0,b2=b2,c0=c0,c2=c2)/fb],':r')
        else:
            ax_f.plot([0,max_fev],[price_heat(dphi0_sol/scale_var_params.get('phibase'), dphi2_sol/scale_var_params.get('phibase'),a0=a0/fb,a2=a2/fb,b0=b0/(fb/scale_var_params.get('phibase')),b2=b2/(fb/scale_var_params.get('phibase')),c0=c0/(fb/scale_var_params.get('phibase')**2),c2=c2/(fb/scale_var_params.get('phibase')**2)),price_heat(dphi0_sol/scale_var_params.get('phibase'), dphi2_sol/scale_var_params.get('phibase'),a0=a0/fb,a2=a2/fb,b0=b0/(fb/scale_var_params.get('phibase')),b2=b2/(fb/scale_var_params.get('phibase')),c0=c0/(fb/scale_var_params.get('phibase')**2),c2=c2/(fb/scale_var_params.get('phibase')**2))],':r')
        ax_To2.plot([0,max_fev],[To2_bounds[0]/scale_var_params.get('Tbase'),To2_bounds[0]/scale_var_params.get('Tbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_To2.plot([0,max_fev],[To2_bounds[1]/scale_var_params.get('Tbase'),To2_bounds[1]/scale_var_params.get('Tbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_To2.plot([0,max_fev],[To2_sol/scale_var_params.get('Tbase'),To2_sol/scale_var_params.get('Tbase')],':r')
        ax_dphi0.plot([0,max_fev],[phi0_bounds[0]/scale_var_params.get('phibase'),phi0_bounds[0]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0.plot([0,max_fev],[phi0_bounds[1]/scale_var_params.get('phibase'),phi0_bounds[1]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0.plot([0,max_fev],[dphi0_sol/scale_var_params.get('phibase'),dphi0_sol/scale_var_params.get('phibase')],':r')
        ax_dphi2.plot([0,max_fev],[phi2_bounds[0]/scale_var_params.get('phibase'),phi2_bounds[0]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi2.plot([0,max_fev],[phi2_bounds[1]/scale_var_params.get('phibase'),phi2_bounds[1]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi2.plot([0,max_fev],[dphi2_sol/scale_var_params.get('phibase'),dphi2_sol/scale_var_params.get('phibase')],':r')

        ax_contour.legend(handles=legend_handles)
        ax_f.legend(handles=legend_handles)
        ax_To2.legend(handles=legend_handles)
        ax_dphi0.legend(handles=legend_handles)
        ax_dphi2.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','H3N')
        with open(os.path.join(path_to_tables,'optimizer_info_scaling_sep_LF_{}To2_{}phi2_{}phi0ub.txt'.format(To2_factor,phi2_factor,phi0_factor_ub)), "w") as table:
            for bound in bounds:
                for approach in approaches:
                    if approach == 'eq_constr':
                        approach_label = 'eq. constr.'
                    else:
                        approach_label = approach
                    xh_opt_trust_mat = xh_res.get('trust-constr_'+bound+'_'+approach+'_matrix')
                    xh_opt_slsqp_mat = xh_res.get('SLSQP_'+bound+'_'+approach+'_matrix')
                    xh_opt_ipopt_mat = xh_res.get('ipopt_'+bound+'_'+approach+'_matrix')
                    res_trust_mat = result.get('trust-constr_'+bound+'_'+approach+'_matrix')
                    res_slsqp_mat = result.get('SLSQP_'+bound+'_'+approach+'_matrix')
                    res_ipopt_mat = result.get('ipopt_'+bound+'_'+approach+'_matrix')
                    xh_opt_trust_pu = xh_res.get('trust-constr_'+bound+'_'+approach+'_per_unit')
                    xh_opt_slsqp_pu = xh_res.get('SLSQP_'+bound+'_'+approach+'_per_unit')
                    xh_opt_ipopt_pu = xh_res.get('ipopt_'+bound+'_'+approach+'_per_unit')
                    res_trust_pu = result.get('trust-constr_'+bound+'_'+approach+'_per_unit')
                    res_slsqp_pu = result.get('SLSQP_'+bound+'_'+approach+'_per_unit')
                    res_ipopt_pu = result.get('ipopt_'+bound+'_'+approach+'_per_unit')
                    table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} \\ '.format(bound,approach_label,res_trust_mat.success,res_trust_pu.success,res_slsqp_mat.success,res_slsqp_pu.success,res_ipopt_mat.success,res_ipopt_pu.success,res_trust_mat.nit,res_trust_pu.nit,res_slsqp_pu.nit,res_slsqp_mat.nit,res_ipopt_mat.nit,res_ipopt_pu.nit,error(xh_opt_trust_mat,xh_LF),error(xh_opt_slsqp_mat,xh_LF),error(xh_opt_ipopt_mat,xh_LF),error(xh_opt_trust_pu,xh_LF),error(xh_opt_slsqp_pu,xh_LF),error(xh_opt_ipopt_pu,xh_LF)))
                table.write(r'\hline ')

    for bound in bounds:
        for approach in approaches:
            for scale_var in scaling:
                xh_opt_trust = xh_res.get('trust-constr_'+bound+'_'+approach+'_'+scale_var)
                xh_opt_slsqp = xh_res.get('SLSQP_'+bound+'_'+approach+'_'+scale_var)
                xh_opt_ipopt = xh_res.get('ipopt_'+bound+'_'+approach+'_'+scale_var)
                res_trust = result.get('trust-constr_'+bound+'_'+approach+'_'+scale_var)
                res_slsqp = result.get('SLSQP_'+bound+'_'+approach+'_'+scale_var)
                res_ipopt = result.get('ipopt_'+bound+'_'+approach+'_'+scale_var)
                print('\nBounds: {}, approach: {}, scaling: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\ntrust-constr:{}\nSLSQP: {}\nIPOPT: {}\nErrors for t-c: {}, SLSQP: {}, IPOPT: {}'.format(bound,approach,scale_var,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(xh_opt_trust,xh_LF),error(xh_opt_slsqp,xh_LF),error(xh_opt_ipopt,xh_LF)))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','H3N')
        for fig_num in plt.get_figlabels():
            if not '3d' in fig_num:
                plt.figure(fig_num)
                file_name = fig_num+'.pgf'
                plt.savefig(os.path.join(path_to_fig, file_name))

def error(x_res,x_sol):
    """Relative error between solution and result.

    Parameters
    ----------
    x_res : np array
        Variables result. Unscaled
    x_sol : np array
        Variables solution. Unscaled
    """
    return np.max(np.abs(x_sol-x_res)/np.abs(x_sol))

def layout_convergence(ax,tol,max_iters):
    ax.semilogy([0,max_iters+1],[tol,tol],'k:',label='tolerance')
    xmin = 0
    xmax = max_iters
    xticks = range(xmin,xmax+1,max(1,int(xmax/10))) # make sure the xticks are integers
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

def layout_convergence_order(ax):
    x_min,x_max = ax.get_xlim()
    x_slope = np.linspace(x_min,x_max)
    y_slope2 = x_slope**2
    ax.loglog(x_slope,x_slope,linestyle=':',color='k',label='slope 1')
    ax.loglog(x_slope,y_slope2,linestyle='-.',color='k',label='slope 2')
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # compare_opf_derivatives(dir_path,number_runs=1,save_figs=False,save_tables=False)
    # compare_scaling_formulation()
    # compare_opf_methods(dir_path=dir_path,save_tables=False,save_figs=False)
    # compare_opf_scaling(dir_path=dir_path,save_tables=False,save_figs=False)
    # compare_opf_methods_sep_LF(dir_path=dir_path,save_tables=False,save_figs=False,To2_factor = 0.9,phi2_factor = 1.1,phi0_factor_ub=1)
    # compare_opf_methods_sep_LF(dir_path=dir_path,save_tables=False,save_figs=False,To2_factor = 0.9,phi2_factor = 1.1,phi0_factor_ub=.8)
    compare_opf_scaling_sep_LF(dir_path=dir_path,save_tables=False,save_figs=False,To2_factor = 0.9,phi2_factor = 1.1,phi0_factor_ub=1)

    plt.show()
