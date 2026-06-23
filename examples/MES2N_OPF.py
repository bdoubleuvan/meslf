"""
MES consisting of 2 nodes per carrier (i.e. only one link in every single-carrier network), with multiple couplings.

Optimal power flow is conducted for these networks
"""
from examples import MES2N as MES
from meslf.networks.gas_network import GasHalfLink
import warnings
from meslf.utils.constants import bar, mbar, kV, MW
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pytest
from meslf.load_flow.system_of_equations import NonLinearSystemGas, NonLinearSystemElectrical, NonLinearSystemHeterogeneous
import scipy.optimize as spo
import scipy.sparse as sps
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

def create_gas_network():
    """Create a gas  network, consisting of two gas and two electrical nodes. The node types are slightly different from the ones used in MES2N, hence the network is created (again)"""
    gas_net = MES.create_gas_network()
    GasHalfLink('gn0_slack',gas_net.nodes[0],0) # slack
    GasHalfLink('gn0_load',gas_net.nodes[0],MES.qc0_sol,bc_type=1) # gas sink (flow to coupling), q known
    return gas_net

def update_bc_gas(gas_net,q0c,q1c):
    """Updates the boundary conditions of the gas network, based on the state variables of the OPF"""
    gas_net.nodes[0].half_links[1].q = q0c #>0
    gas_net.nodes[1].half_links[0].q = q1c #>0
    return gas_net

def update_bc_gas2(gas_net,q1c):
    """Updates the boundary conditions of the gas network, based on the state variables of the OPF"""
    gas_net.nodes[1].half_links[0].q = q1c #>0
    return gas_net

def update_bc_electrical(elec_net,V0,P0c):
    """Updates the boundary conditions of the electricity network, based on the state variables of the OPF"""
    elec_net.nodes[0].V = V0
    elec_net.nodes[0].half_links[0].P = -P0c #generator, so source, so <0 (and P0c > 0)
    return elec_net

def create_mes_ge_network():
    """Create a combined gas-electricity network, consisting of two gas and two electrical nodes. The node types are slightly different from the ones used in MES2N, hence the network is created (again)"""
    het_net, gas_net, elec_net = MES.create_mes_ge_network()
    # change the node types / BCs
    gas_net.nodes[0].node_type = 0 # slack node
    elec_net.links[1].bc_type = 1 # Pstart known (i.e. active power produced by coupling node 0 is known)
    elec_net.links[1].Pstart = MES.Pc0_sol
    elec_net.links[1].Pend = -MES.Pc0_sol # apparently, both need to be set?!

    return het_net, gas_net, elec_net

def update_bc_mes_ge(het_net, gas_net, elec_net,V0,P0c,scale_var=None,scale_var_params=None):
    """Updates the boundary conditions of the gas-electricity network, based on the control variables of the OPF"""
    if scale_var == 'per_unit':
        V0 = V0*scale_var_params.get('Vbase')
        P0c = P0c*scale_var_params.get('Sbase')
    elec_net.nodes[0].V = V0
    elec_net.links[1].Pstart = P0c
    elec_net.links[1].Pend = -P0c # apparently, both need to be set?!
    return het_net, gas_net, elec_net

def run_mes_ge_load_flow(max_iter=MES.max_iter_outer,tol=MES.tol,plot_top=False,plot_jac=False,plot_sol=False,scale_var=None,scale_var_params=None,formulation=MES.formulation):
    """Steady-state load flow analysis of combined gas and electrical network, without scaling. The default values are used for initialization.
    """
    # create network
    het_net, gas_net, elec_net = create_mes_ge_network()

    # initialize network
    x0 = MES.initialize_mes_ge_network(het_net)

    if plot_jac:
        from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
        nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation)
        fig_J = nlsys.spy_plot_J(x0,title='Jacobian spy plot, scaling = {}'.format(scale_var))
        fig_J_map = nlsys.imshow_J(x0,title='Colormap Jacobian, scaling = {}'.format(scale_var))
    # solve network
    print('\nRunning load flow for multi-carrier gas-electricity network')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/MES.Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))

    if plot_top:
        # plot topology
        fig_top = plt.figure('Network topology')
        ax_top = fig_top.gca()
        het_net.draw_network(ax_top,halflink_angle=2,halflink_length=.5)
        plt.axis('equal')
        plt.axis('off')

    if plot_sol:
        # plot solution
        fig_sol = plt.figure('Network solution, scaling = {}'.format(scale_var))
        ax_sol = fig_sol.gca()
        het_net.draw_network_value(ax_sol,halflink_angle=2,halflink_length=.5)
        plt.axis('equal')
        plt.axis('off')

    return het_net, gas_net, elec_net,x_sol,iters,err_vec

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_mes_ge_load_flow():
    # Given + When
    max_iter = 50
    _, _, _, x_sol, _, _ = run_mes_ge_load_flow(max_iter=max_iter)

    # Then
    _, _, _, x_sol_original, _, _ = MES.run_mes_ge_load_flow(max_iter=max_iter)
    x_sol_expected = np.concatenate((x_sol_original[:5],x_sol_original[6:]))
    assert np.allclose(x_sol,x_sol_expected)

def xmes_from_xopf(x_opf):
    xmes = x_opf[3:]
    return xmes

def create_coupling_ge_single_network():
    """Create a coupling network consisting of two nodes, for a gas-electricity coupling. The halflink types are slightly different from the ones used in MES2N, hence the network is created (again).

    NB. The boundary conditions are such that the resulting system of loadflow equations in underdetermined.
    """
    coupling_net = MES.create_coupling_ge_single_network()
    coupling_net.nodes[0].half_links[0].bc_type = 0 # q unknown
    return coupling_net

def update_bc_coupling_ge(coupling_net,Q0c,P1c,Q1c):
    coupling_net.nodes[0].half_links[1].Q = Q0c
    coupling_net.nodes[1].half_links[1].P = P1c
    coupling_net.nodes[1].half_links[1].Q = Q1c
    return coupling_net

def update_bc_coupling_ge2(coupling_net,q0c,Q0c,P1c,Q1c):
    coupling_net.nodes[0].half_links[0].q = -q0c
    coupling_net.nodes[0].half_links[1].Q = Q0c
    coupling_net.nodes[1].half_links[1].P = P1c
    coupling_net.nodes[1].half_links[1].Q = Q1c
    return coupling_net

def initialize_coupling_ge_single_network(coupling_net,qc_ic=MES.qc_ic,Pc_ic=MES.Pc_ic,formulation=MES.formulation):
    """Initialize the coupling network consisting of two nodes, for a gas-electricity coupling

    Parameters
    ----------
    coupling_net : HeterogeneousNetwork
        The coupling network to be initialized

    Returns
    -------
    x0 : np array
        initial guess
    """
    x_init = np.array([qc_ic,qc_ic,Pc_ic])
    coupling_net.initialize()
    coupling_net.update(x_init,formulation=formulation)
    x0 = coupling_net.set_x_init(formulation=formulation)
    return x0

def price_gas(q0,a=MES.q0_source**2,b=2*MES.q0_source,c=1,scale_var=None,scale_var_params=None,fb=None):
    """Determine the cost of the gas input into the network.

    Parameters
    ----------
    q0 : float
        Gas mass flow source half link of node 0, in kg/s. Since it is a source, it is assumed to be negative.
    a : float
        Parameter of price function, in euros.
    b : float
        Parameter of price function, in euros/(kg/s).
    c : float
        Parameter of price function, in euros/(kg/s)^2.

    Returns
    -------
    f : float
        Total price of the gas sources, in euros.
    """
    f = a + b*-q0 + c*q0**2
    if scale_var == 'matrix':
        f *= 1/fb
    return f

def price_gas_first_der(q0,a=MES.q0_source**2,b=2*MES.q0_source,c=1,scale_var=None,scale_var_params=None,fb=None):
    """First derivative of the cost of the gas input into the network.

    Parameters
    ----------
    q0 : float
        Gas mass flow source half link of node 0, in kg/s. Since it is a source, it is assumed to be negative.
    a : float
        Parameter of price function, in euros.
    b : float
        Parameter of price function, in euros/(kg/s).
    c : float
        Parameter of price function, in euros/(kg/s)^2.

    Returns
    -------
    df_dq0 : float
        First deriviative of total price of the gas sources.
    """
    df_dq0 = -b + 2*c*q0
    if scale_var == 'matrix':
        df_dq0 *= (scale_var_params.get('qbase')/fb)
    return df_dq0

def price_gas_second_der(q0,a=MES.q0_source**2,b=2*MES.q0_source,c=1,scale_var=None,scale_var_params=None,fb=None):
    """second derivative of the cost of the gas input into the network.

    Parameters
    ----------
    q0 : float
        Gas mass flow source half link of node 0, in kg/s. Since it is a source, it is assumed to be negative.
    a : float
        Parameter of price function, in euros.
    b : float
        Parameter of price function, in euros/(kg/s).
    c : float
        Parameter of price function, in euros/(kg/s)^2.

    Returns
    -------
    d2f_dq02 : float
        Second derivative of total price of the gas sources.
    """
    d2f_dq02 = 2*c
    if scale_var == 'matrix':
        d2f_dq02 *= (scale_var_params.get('qbase')**2)/fb
    return d2f_dq02

def price_gas_electricity(q0,P0c,P1c,a0=0,b0=.01*MES.GHV,c0=1e-6*MES.GHV**2,a0c=0,b0c=.3,c0c=3e-5,a1c=0,b1c=.2,c1c=2e-5,scale_var=None,scale_var_params=None,fb=None):
    """Determine the cost of the gas input into the network.

    Parameters
    ----------
    q0 : float
        Gas mass flow source half link of node 0, in kg/s. Since it is a source, it is assumed to be negative.
    P0c : float
        Active power produced by coupling 0, in W. Assumed to be positive.
    P1c : float
        Active power produced by coupling 1, in W. Assumed to be positive
    a0, a0c, a1c : float
        Parameter of price function, in euros.
    b0, b0c, b1c : float
        Parameter of price function, in euros/(kg/s) for b0 or in euros/W for b0c and b1c.
    c0, c0c, c1c : float
        Parameter of price function, in euros/(kg/s)^2 for c0 or in euros/W^2 for c0c and c1c.

    Returns
    -------
    f : float
        Total price of the gas sources and the conversions of gas to electricity, in euros.
    """
    f = a0 + b0*-q0 + c0*q0**2 + a0c + b0c*P0c + c0c*P0c**2 + a1c + b1c*P1c + c1c*P1c**2
    if scale_var == 'matrix':
        f *= (1/fb)
    return f

def price_gas_electricity_first_der(q0,P0c,P1c,a0=0,b0=.01*MES.GHV,c0=1e-6*MES.GHV**2,a0c=0,b0c=.3,c0c=3e-5,a1c=0,b1c=.2,c1c=2e-5,scale_var=None,scale_var_params=None,fb=None):
    """First (partial) derivative of the cost of the gas input into the network.

    Parameters
    ----------
    q0 : float
        Gas mass flow source half link of node 0, in kg/s. Since it is a source, it is assumed to be negative.
    P0c : float
        Active power produced by coupling 0, in W. Assumed to be positive.
    P1c : float
        Active power produced by coupling 1, in W. Assumed to be positive
    a0, a0c, a1c : float
        Parameter of price function, in euros.
    b0, b0c, b1c : float
        Parameter of price function, in euros/(kg/s) for b0 or in euros/W for b0c and b1c.
    c0, c0c, c1c : float
        Parameter of price function, in euros/(kg/s)^2 for c0 or in euros/W^2 for c0c and c1c.

    Returns
    -------
    df_dq0, df_dP0c, df_dP1c : float
        First (partial) derivatives of total price of the gas sources and the conversions of gas to electricity.
    """
    df_dq0, df_dP0c, df_dP1c = -b0 + 2*c0*q0, b0c + 2*c0c*P0c, b1c + 2*c1c*P1c
    if scale_var == 'matrix':
        df_dq0 *= (scale_var_params.get('qbase')/fb)
        df_dP0c *= (scale_var_params.get('Sbase')/fb)
        df_dP1c  *= (scale_var_params.get('Sbase')/fb)
    return df_dq0, df_dP0c, df_dP1c

def price_gas_electricity_second_der(q0,P0c,P1c,a0=0,b0=.01*MES.GHV,c0=1e-6*MES.GHV**2,a0c=0,b0c=.3,c0c=3e-5,a1c=0,b1c=.2,c1c=2e-5,scale_var=None,scale_var_params=None,fb=None):
    """Second (partial) derivatives of the cost of the gas input into the network.

    Parameters
    ----------
    q0 : float
        Gas mass flow source half link of node 0, in kg/s. Since it is a source, it is assumed to be negative.
    P0c : float
        Active power produced by coupling 0, in W. Assumed to be positive.
    P1c : float
        Active power produced by coupling 1, in W. Assumed to be positive
    a0, a0c, a1c : float
        Parameter of price function, in euros.
    b0, b0c, b1c : float
        Parameter of price function, in euros/(kg/s) for b0 or in euros/W for b0c and b1c.
    c0, c0c, c1c : float
        Parameter of price function, in euros/(kg/s)^2 for c0 or in euros/W^2 for c0c and c1c.

    Returns
    -------
    d2f_dq02, d2f_dP0c2, d2f_dP1c2 : float
        Second (partial) derivatives of total price of the gas sources and the conversions of gas to electricity.
    """
    d2f_dq02, d2f_dP0c2, d2f_dP1c2 = 2*c0, 2*c0c, 2*c1c
    if scale_var == 'matrix':
        d2f_dq02 *= ((scale_var_params.get('qbase')**2)/fb)
        d2f_dP0c2 *= ((scale_var_params.get('Sbase')**2)/fb)
        d2f_dP1c2  *= ((scale_var_params.get('Sbase')**2)/fb)
    return d2f_dq02, d2f_dP0c2, d2f_dP1c2

def run_ge_optimal_load_flow(P0c_init=MES.Pc_ic,P0c_lb=.5*MES.Pc0_sol,P0c_ub=1*MES.Pc0_sol, V0_init=1.1*MES.Vbase,V0_lb=0.8*MES.V0_sol,V0_ub=1*MES.V0_sol,q0_lb=1.5*MES.q0_source,q0_ub=.5*MES.q0_source,q01_lb=-5,q01_ub=5,p1_lb=1*mbar,p1_ub=1.1*MES.pg0,delta0_lb=-np.pi,delta0_ub=np.pi,q0c_lb=0,q0c_ub=1.5*MES.qc0_sol,q1c_lb=0,q1c_ub=1.5*MES.qc1_sol,P1c_lb=0,P1c_ub=1.5*MES.Pc1_sol,Q0c_lb=-2*MES.Qc0_sol,Q0c_ub=2*MES.Qc0_sol,Q1c_lb=-2*MES.Qc1_sol,Q1c_ub=2*MES.Qc1_sol,max_iter=MES.max_iter_outer,tol=MES.tol,scale_var=None,scale_var_params=None,formulation=MES.formulation,a=MES.q0_source**2,b=2*MES.q0_source,c=1,a0=0,b0=.01*MES.GHV,c0=1e-6*MES.GHV**2,a0c=0,b0c=.3,c0c=3e-5,a1c=0,b1c=.2,c1c=2e-5,fb=None,ineq_constr='control',derivatives=False,objective='gas',optimization_method='trust-constr',stay_within_bounds=False):
    """Run optimal power flow for the combined gas-electricity network. The constraints on the control variables, specifically on Pc0, and the cost function are chosen such that the solution to OPF is equal to the solution of LF. In this case, the coupling flows are divided over the control and state variables in such a way that an integrated load flow problem could be formulated for the MES.

    Parameters
    ----------------
    derivatives : bool, optional
        If True, analytical expressions for the gradient and Hessian of the objective function and of the (nonlinear) constraints are used. Otherwise, numerical approximations are used. Default is False.
    """
    print('\nRunning OPF for gas-electricty network, using the integrated MES (method: {}, obj: {}, ineq. constr. on: {}, hard bounds: {}, an der: {})'.format(optimization_method,objective,ineq_constr,stay_within_bounds,derivatives))
    # create network
    het_net, gas_net, elec_net = create_mes_ge_network()

    if scale_var == 'matrix' and scale_var_params == None:
        scale_var_params = MES.scale_var_params

    # update the boundary conditions of the MES to match the initial guess of opf
    het_net, gas_net, elec_net = update_bc_mes_ge(het_net, gas_net, elec_net,V0_init,P0c_init)

    # run load flow once, to make sure that the initial guess of opf is at least a solution of LF
    x0 = MES.initialize_mes_ge_network(het_net)
    x_LF,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
    het_net.reset_network(x_LF,formulation=formulation)
    print('For initial guess, final error LF = {:.4e}'.format(err_vec[-1]))

    # initial guess for OPF (unscaled)
    u_init = np.array([V0_init,P0c_init])
    q0 = q_inj[0]
    slack_init = np.array([q0])
    x_opf0 = np.concatenate((u_init,slack_init,x_LF)) # initial guess for y

    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    Ng = len(gas_net.nodes)
    Eg = 1 # non-dummy links in gasnetwork
    Ne = len(elec_net.nodes)
    Eg_dummy = 2
    Ee_dummy = 2
    Fe_ind = nlsys.nlsystemse[0].FP + [Ne+ind for ind in nlsys.nlsystemse[0].FQ]
    V0_ind = Ne
    q0_ind = 2
    P0c_ind = 1
    P1c_ind = 8

    DF = nlsys.DF()
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        DH = np.diag(np.concatenate((np.array([1/scale_var_params.get('qbase')]),DF.data[0])))
        Dy = np.diag(np.concatenate((np.array([1/scale_var_params.get('Vbase'),1/scale_var_params.get('Sbase'),1/scale_var_params.get('qbase')]),Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((np.array([scale_var_params.get('Vbase'),scale_var_params.get('Sbase'),scale_var_params.get('qbase')]),1/Dx.data[0])))
        x_opf0 = Dy.dot(x_opf0) # scale y
    else:
        DH = np.eye(1+DF.shape[0])
        Dy=np.eye(len(x_opf0))
        Dy_inv=np.eye(len(x_opf0))
    # print('DH = {}'.format(DH))

    if scale_var == 'per_unit':
        a = a/fb
        b = b/(fb/scale_var_params.get('qbase'))
        c = c/(fb/(scale_var_params.get('qbase')**2))
        a0 = a0/fb
        b0 = b0/(fb/scale_var_params.get('qbase'))
        c0 = c0/(fb/(scale_var_params.get('qbase')**2))
        a0c = a0c/fb
        b0c = b0c/(fb/scale_var_params.get('Sbase'))
        c0c = c0c/(fb/(scale_var_params.get('Sbase')**2))
        a1c = a1c/fb
        b1c = b1c/(fb/scale_var_params.get('Sbase'))
        c1c = c1c/(fb/(scale_var_params.get('Sbase')**2))

    def obj(y,a=a,b=b,c=c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb):
        """Define the cost function for OPF

        Parameters
        ----------------
        x_opf : np array
            Variable vector used in OPF. Is assumed to be [V0, P0c, q0, q01, p1, delta0, q0c, q1c, P1c, Q0c, Q1c]

        Returns
        -----------
        f : float
            The value of the cost function
        """
        global f_vec_global
        global x_f_vec
        x_f_vec = y.copy()
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        q0 = y[q0_ind] #<0, scaled
        if objective == 'gas':
            f = price_gas(q0,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
        elif objective == 'gas_elec':
            P0c = y[P0c_ind] #>0, scaled
            P1c = y[P1c_ind] #>0, scaled
            f = price_gas_electricity(q0,P0c,P1c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    # gradient and Hessian of objective function
    def obj_grad(y,a=a,b=b,c=c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        df_dy = np.zeros(len(y))
        q0 = y[q0_ind] #<0, scaled
        if objective == 'gas':
            df_dq0 = price_gas_first_der(q0,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
            df_dy[q0_ind] = df_dq0
        elif objective == 'gas_elec':
            P0c = y[P0c_ind] #>0, scaled
            P1c = y[P1c_ind] #>0, scaled
            df_dq0, df_dP0c, df_dP1c = price_gas_electricity_first_der(q0,P0c,P1c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
            df_dy[q0_ind] = df_dq0
            df_dy[P0c_ind] = df_dP0c
            df_dy[P1c_ind] = df_dP1c
        # print('dfdy = {}'.format(df_dy))
        return df_dy
    def obj_hess(y,a=a,b=b,c=c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        hess_cost_diag = np.zeros(len(y))
        q0 = y[q0_ind] #<0, scaled
        if objective == 'gas':
            d2f_dq02 = price_gas_second_der(q0,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
            hess_cost_diag[q0_ind] = d2f_dq02
        elif objective == 'gas_elec':
            P0c = y[P0c_ind] #>0, scaled
            P1c = y[P1c_ind] #>0, scaled
            d2f_dq02, d2f_dP0c2, d2f_dP1c2 = price_gas_electricity_second_der(q0,P0c,P1c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
            hess_cost_diag[q0_ind] = d2f_dq02
            hess_cost_diag[P0c_ind] = d2f_dP0c2
            hess_cost_diag[P1c_ind] = d2f_dP1c2
        # print('hess diag = {}'.format(hess_cost_diag))
        return np.diag(hess_cost_diag)

    # define nonlinear equality constriants (load flow equations)
    def nonlinear_equality_constraints(y,network_mes=het_net,network_g=gas_net,network_e=elec_net,scale_var=scale_var,scale_var_params=scale_var_params):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        V0 = y[0]
        P0c = y[P0c_ind]
        network_mes, network_g, network_e = update_bc_mes_ge(network_mes, network_g, network_e,V0,P0c,scale_var=scale_var,scale_var_params=scale_var_params)
        # evaluate conservation of mass in slack node of gas network
        xmes = xmes_from_xopf(y)
        network_mes.reset_network(xmes,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        q0 = y[P0c_ind+1]
        if scale_var == 'per_unit':
            network_g.nodes[0].half_links[0].q = q0*scale_var_params.get('qbase')
        else:
            network_g.nodes[0].half_links[0].q = q0
        cons_mass = network_g.nodes[0].node_law(scale_var=scale_var,scale_var_params=scale_var_params)
        # evaluate load flow equations
        network_mes.reset_network(xmes,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        F = nlsys.F(xmes)
        H = np.concatenate((np.array([cons_mass]),F))
        if scale_var == 'matrix':
            H = DH.dot(H)
        return H

    # Jacobian of nonlinear constraints
    def jac_nleq(y,network_mes=het_net,network_g=gas_net,network_e=elec_net,scale_var=scale_var,scale_var_params=scale_var_params):
        if scale_var == 'matrix':
            y = Dy_inv.dot(y)
        # update bc of the network
        V0 = y[0]
        P0c = y[P0c_ind]
        network_mes, network_g, network_e = update_bc_mes_ge(network_mes, network_g, network_e,V0,P0c,scale_var=scale_var,scale_var_params=scale_var_params)
        xmes = xmes_from_xopf(y)
        network_mes.reset_network(xmes,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        J_lf = nlsys.J_dense(xmes)
        xe = np.array([xmes[2]])
        nlsys.nlsystemse[0].V_vec_mag[0] = V0
        Je_full = nlsys.nlsystemse[0].J_dense(xe,return_full=True)
        dH_dy = np.zeros((len(slack_init)+len(x_LF),len(y)))
        dH_dy[3:7,0] = Je_full[Fe_ind,V0_ind].ravel() # dFe_dV0
        dH_dy[:,P0c_ind] = np.array([0,0,0,-1,0,0,0,1,0])
        dH_dy[0,:] = np.array([0,0,-1,-1,0,0,-1,0,0,0,0])
        dH_dy[1:,3:] = J_lf #dF_dxlf
        if scale_var == 'matrix':
            dH_dy = DH.dot(dH_dy.dot(Dy_inv))
        # if scale_var == 'matrix':
        #     plt.figure('jac obj')
        #     plt.spy(dH_dy)
        #     plt.show()
        return dH_dy

    lb_nleq = np.zeros(len(x_LF)+len(slack_init))
    ub_nleq = np.zeros(len(x_LF)+len(slack_init))
    if derivatives:
        if optimization_method == 'trust-constr':
            nonlinear_constraint = spo.NonlinearConstraint(nonlinear_equality_constraints,lb_nleq,ub_nleq,jac=jac_nleq,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            nonlinear_constraint = {'type':'eq','fun':nonlinear_equality_constraints,'jac':jac_nleq}
    else:
        if optimization_method == 'trust-constr':
            nonlinear_constraint = spo.NonlinearConstraint(nonlinear_equality_constraints,lb_nleq,ub_nleq,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            nonlinear_constraint = {'type':'eq','fun':nonlinear_equality_constraints}

    # define linear inequality constraints, i.e. define bounds
    lb_ineq = np.array([V0_lb,P0c_lb,q0_lb,q01_lb,p1_lb,delta0_lb,q0c_lb,q1c_lb,P1c_lb,Q0c_lb,Q1c_lb])
    ub_ineq = np.array([V0_ub,P0c_ub,q0_ub,q01_ub,p1_ub,delta0_ub,q0c_ub,q1c_ub,P1c_ub,Q0c_ub,Q1c_ub])
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        lb_ineq = Dy.dot(lb_ineq)
        ub_ineq = Dy.dot(ub_ineq)
    if ineq_constr == 'control':
        lb_ineq[2:] = -np.inf*np.ones(len(x_opf0)-2) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
        ub_ineq[2:] = np.inf*np.ones(len(x_opf0)-2)

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

    global f_vec_global
    global x_f_vec
    f_vec_global = list()
    f_vec = list()
    x_f_vec = list()
    if optimization_method == 'trust-constr':
        def callback_opf(xk, state):
            f_vec.append(state.fun)
            return False
    elif optimization_method == 'SLSQP':
        f_vec.append(obj(x_opf0))
        def callback_opf(xk):
            f_vec.append(obj(xk))
            return False

    # solve OPF
    opf_start_time = time.time()
    try:
        if derivatives:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, x_opf0, method=optimization_method, jac=obj_grad,hess=obj_hess, constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter,'gtol':tol,'xtol':tol}, bounds=bounds,tol=tol, callback=callback_opf)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                opf_start_time = time.time()
                res = spo.minimize(obj, x_opf0, method=optimization_method,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback_opf)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, x_opf0, jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
        else:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, x_opf0, method=optimization_method, constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter,'gtol':tol,'xtol':tol}, bounds=bounds,tol=tol,callback=callback_opf)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, x_opf0, method=optimization_method, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback_opf)
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
            res = spo.OptimizeResult({'success':False,'x':np.array(x_f_vec),'fun':obj(np.array(x_f_vec)),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

    if optimization_method == 'ipopt':
        if res.nit > 0:
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            f_vec = f_vec_global

    if scale_var == 'matrix' or scale_var == 'per_unit':
        x_opf = Dy_inv.dot(res.x)
    else:
        x_opf = res.x

    # print solution
    V0 = x_opf[0]
    P0c = x_opf[P0c_ind]
    het_net, gas_net, elec_net = update_bc_mes_ge(het_net, gas_net, elec_net,V0,P0c)
    xmes_opt = xmes_from_xopf(x_opf)
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('Solution OPF (inequality constraints on control variables: {})'.format(ineq_constr))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/MES.Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    # print('objective function = {}'.format(obj(x_opf)))
    return x_opf, xmes_opt, f_vec, res.fun, res.nfev, res.nit, res.njev, execution_time, res.success

def solve_ge_lf_in_of(network_mes, network_g, network_e,u,max_iters=10,tol=MES.tol,formulation=MES.formulation,scale_var=None,scale_var_params=None):
    """Solve steady-state LF within an optmization context.

    Parameters
    ----------
    u : np arrays
        Vector with control variables. Scaled when using per unit scaling, unscaled otherwise
    """
    V0, P0c = u
    xmes = network_mes.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    network_mes.reset_network(xmes,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    network_mes, network_g, network_e = update_bc_mes_ge(network_mes, network_g, network_e,V0,P0c,scale_var=scale_var,scale_var_params=scale_var_params)
    with HiddenPrints():
        x_LF,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = network_mes.solve_network(tol,max_iters,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    q0 = network_g.nodes[0].half_links[0].get_q(scale_var=scale_var,scale_var_params=scale_var_params) #<0
    P1c = network_e.links[2].get_Pstart(scale_var=scale_var,scale_var_params=scale_var_params) #>0
    return q0, P1c, network_mes, network_g, network_e

def run_ge_optimal_load_flow_separate_LF(P0c_init=MES.Pc_ic,P0c_lb=.5*MES.Pc0_sol,P0c_ub=1*MES.Pc0_sol, V0_init=1.1*MES.Vbase,V0_lb=0.8*MES.V0_sol,V0_ub=1*MES.V0_sol,q0_lb=1.5*MES.q0_source,q0_ub=.5*MES.q0_source,q01_lb=-5,q01_ub=5,p1_lb=1*mbar,p1_ub=1.1*MES.pg0,delta0_lb=-np.pi,delta0_ub=np.pi,q0c_lb=0,q0c_ub=1.5*MES.qc0_sol,q1c_lb=0,q1c_ub=1.5*MES.qc1_sol,P1c_lb=0,P1c_ub=1.5*MES.Pc1_sol,Q0c_lb=-2*MES.Qc0_sol,Q0c_ub=2*MES.Qc0_sol,Q1c_lb=-2*MES.Qc1_sol,Q1c_ub=2*MES.Qc1_sol,max_iter=MES.max_iter_outer,max_iters_lf=10,tol=MES.tol,scale_var=None,scale_var_params=None,formulation=MES.formulation,a=MES.q0_source**2,b=2*MES.q0_source,c=1,a0=0,b0=.01*MES.GHV,c0=1e-6*MES.GHV**2,a0c=0,b0c=.3,c0c=3e-5,a1c=0,b1c=.2,c1c=2e-5,fb=None,ineq_constr='control',objective='gas',optimization_method='trust-constr',stay_within_bounds=False,approach='direct'):
    """Run optimal power flow for the combined gas-electricity network, where the LF is included implicitly. The constraints on the control variables, specifically on Pc0, and the cost function are chosen such that the solution to OPF is equal to the solution of LF. In this case, the coupling flows are divided over the control and state variables in such a way that an integrated load flow problem could be formulated for the MES.

    Parameters
    ----------------
    max_iter : int, optional
        Maximum number of iteration used for the OPF (for OPF, the number of functions evalutions might be more).
    max_iters_lf : int, optional
        Maximum number of iteration used for steady-state load flow
    approach : str, optional
        Approach used to compute the gradient and Jacobians. Either 'direct' or 'adjoint'. Default is 'direct'.
    """
    print('\nRunning OPF for gas-electricty network, using the integrated MES, with LF separate and {} approach (method: {}, obj: {}, ineq. constr. on: {}, hard bounds: {})'.format(approach,optimization_method,objective,ineq_constr,stay_within_bounds))
    # create network
    het_net, gas_net, elec_net = create_mes_ge_network()

    if scale_var == 'matrix' and scale_var_params == None:
        scale_var_params = MES.scale_var_params

    # update the boundary conditions of the MES to match the initial guess of opf
    het_net, gas_net, elec_net = update_bc_mes_ge(het_net, gas_net, elec_net,V0_init,P0c_init)
    x0 = MES.initialize_mes_ge_network(het_net) # initialize network, and set reasonable values as first initial guess for LF (if reasonable values are not set, division by 0 etc might occur during LF)

    # initial guess for OF
    u0 = np.array([V0_init,P0c_init])

    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    Ng = len(gas_net.nodes)
    Eg = 1 # non-dummy links in gasnetwork
    Ne = len(elec_net.nodes)
    Eg_dummy = 2
    Ee_dummy = 2
    Fe_ind = nlsys.nlsystemse[0].FP + [Ne+ind for ind in nlsys.nlsystemse[0].FQ]
    V0_ind = Ne
    q0_ind = 2
    P0c_ind = 1
    P1c_ind = 8
    len_x = 9

    DF = nlsys.DF()
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        DH = np.diag(np.concatenate((np.array([1/scale_var_params.get('qbase')]),DF.data[0])))
        Du = np.diag(np.array([1/scale_var_params.get('Vbase'),1/scale_var_params.get('Sbase')]))
        Du_inv = np.diag(np.array([scale_var_params.get('Vbase'),scale_var_params.get('Sbase')]))
        Dy = np.diag(np.concatenate((np.array([1/scale_var_params.get('Vbase'),1/scale_var_params.get('Sbase'),1/scale_var_params.get('qbase')]),Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((np.array([scale_var_params.get('Vbase'),scale_var_params.get('Sbase'),scale_var_params.get('qbase')]),1/Dx.data[0])))
        u0 = Du.dot(u0) # scale u
    else:
        DH = np.eye(1+DF.shape[0])
        Du = np.eye(len(u0))
        Du_inv= np.eye(len(u0))
        Dy=np.eye(len_x+len(u0))
        Dy_inv=np.eye(len_x+len(u0))
    # print('DH = {}'.format(DH))

    if scale_var == 'per_unit':
        a = a/fb
        b = b/(fb/scale_var_params.get('qbase'))
        c = c/(fb/(scale_var_params.get('qbase')**2))
        a0 = a0/fb
        b0 = b0/(fb/scale_var_params.get('qbase'))
        c0 = c0/(fb/(scale_var_params.get('qbase')**2))
        a0c = a0c/fb
        b0c = b0c/(fb/scale_var_params.get('Sbase'))
        c0c = c0c/(fb/(scale_var_params.get('Sbase')**2))
        a1c = a1c/fb
        b1c = b1c/(fb/scale_var_params.get('Sbase'))
        c1c = c1c/(fb/(scale_var_params.get('Sbase')**2))

    # values used for bounds an inequality constraints
    lb_ineq = np.array([V0_lb,P0c_lb,q0_lb,q01_lb,p1_lb,delta0_lb,q0c_lb,q1c_lb,P1c_lb,Q0c_lb,Q1c_lb])
    ub_ineq = np.array([V0_ub,P0c_ub,q0_ub,q01_ub,p1_ub,delta0_ub,q0c_ub,q1c_ub,P1c_ub,Q0c_ub,Q1c_ub])
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        lb_ineq = Dy.dot(lb_ineq)
        ub_ineq = Dy.dot(ub_ineq)

    def obj(u,network_mes=het_net,network_g=gas_net,network_e=elec_net,a=a,b=b,c=c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb):
        """Define the cost function for OPF

        Parameters
        ----------------
        u : np array
            Variable vector used in OPF. Is assumed to be [V0, P0c]

        Returns
        -----------
        f : float
            The value of the cost function
        """
        global f_vec_global
        global x_f_vec
        x_f_vec = u.copy()
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            u = Du_inv.dot(u)
        q0, P1c, network_mes, network_g, network_e = solve_ge_lf_in_of(network_mes, network_g, network_e,u,max_iters=max_iters_lf,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        if objective == 'gas':
            f = price_gas(q0,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
        elif objective == 'gas_elec':
            P0c = u[1]
            f = price_gas_electricity(q0,P0c,P1c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    # gradient of objective function
    def obj_grad(u,network_mes=het_net,network_g=gas_net,network_e=elec_net,method=approach,a=a,b=b,c=c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb,Dy=Dy, Dy_inv=Dy_inv, Dh=DH,Du=Du,Du_inv=Du_inv):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            u = Du_inv.dot(u)
        # update network and solve LF
        q0, P1c, network_mes, network_g, network_e = solve_ge_lf_in_of(network_mes, network_g, network_e,u,max_iters=max_iters_lf,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        # partial derivatives of objective
        x_LF = network_mes.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        y = np.concatenate((u,np.array([q0]),x_LF))
        deltaf_deltay = np.zeros(len(y))
        if objective == 'gas':
            df_dq0 = price_gas_first_der(q0,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
            deltaf_deltay[q0_ind] = df_dq0
        elif objective == 'gas_elec':
            P0c = y[P0c_ind] #>0, scaled
            P1c = y[P1c_ind] #>0, scaled
            df_dq0, df_dP0c, df_dP1c = price_gas_electricity_first_der(q0,P0c,P1c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
            deltaf_deltay[q0_ind] = df_dq0
            deltaf_deltay[P0c_ind] = df_dP0c
            deltaf_deltay[P1c_ind] = df_dP1c
        deltaf_deltau = np.zeros((1,len(u)))
        deltaf_deltax = np.zeros((1,len_x))
        deltaf_deltau[0,:] = deltaf_deltay[:len(u)]
        deltaf_deltax[0,:] = deltaf_deltay[len(u):]
        # partial derivatives of equatilty constraints / load-flow equations
        network_mes.reset_network(x_LF,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        V0, P0c = u
        J_lf = nlsys.J_dense(x_LF)
        xe = np.array([x_LF[2]])
        nlsys.nlsystemse[0].V_vec_mag[0] = V0
        Je_full = nlsys.nlsystemse[0].J_dense(xe,return_full=True)
        deltah_deltay = np.zeros((len_x,len(y)))
        deltah_deltay[3:7,0] = Je_full[Fe_ind,V0_ind].ravel() # dFe_dV0
        deltah_deltay[:,P0c_ind] = np.array([0,0,0,-1,0,0,0,1,0])
        deltah_deltay[0,:] = np.array([0,0,-1,-1,0,0,-1,0,0,0,0])
        deltah_deltay[1:,3:] = J_lf #dF_dxlf
        if scale_var == 'matrix':
            deltah_deltay = DH.dot(deltah_deltay.dot(Dy_inv))
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
        return df_du

    # define (non)linear inequality constraints on state variables
    if ineq_constr == 'all':
        lb_ineq_state = lb_ineq[len(u0):]
        ub_ineq_state = ub_ineq[len(u0):]
        def g(u,scale_var=scale_var,scale_var_params=scale_var_params,network_mes=het_net,network_g=gas_net,network_e=elec_net,Dy=Dy, Dy_inv=Dy_inv, Du=Du,Du_inv=Du_inv):
            """Determine the nonlinear inequality constraints g(x(u)) >= 0"""
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                u = Du_inv.dot(u)
            # update network and solve LF
            q0, P1c, network_mes, network_g, network_e = solve_ge_lf_in_of(network_mes, network_g, network_e,u,max_iters=max_iters_lf,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            x_LF = network_mes.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            network_mes.reset_network(x_LF,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            x = np.concatenate((np.array([q0]),x_LF))
            if scale_var == 'matrix': # lb_ineq_state and ub_ineq_state are scaled, so scale x as well
                x = Dy[len(u):,len(u):].dot(x)
            g = np.concatenate((x-lb_ineq_state,ub_ineq_state-x))
            return g
        def g_jac(u,scale_var=scale_var,scale_var_params=scale_var_params,network_mes=het_net,network_g=gas_net,network_e=elec_net,nlsys=nlsys,method=approach,Dy=Dy, Dy_inv=Dy_inv, fb=fb, Dh=DH,Du=Du,Du_inv=Du_inv):
            """Jacobian of inequality constraints"""
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                u = Du_inv.dot(u)
            # Jacobian of inequality constraints wrt state variables x
            deltag_deltax = np.vstack((np.eye(len_x),-np.eye(len_x)))
            deltag_deltau = np.zeros((2*len_x,len(u)))
            # update network and solve LF
            q0, P1c, network_mes, network_g, network_e = solve_ge_lf_in_of(network_mes, network_g, network_e,u,max_iters=max_iters_lf,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            x_LF = network_mes.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            y = np.concatenate((u,np.array([q0]),x_LF))
            # partial derivatives of equatilty constraints / load-flow equations
            network_mes.reset_network(x_LF,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            V0, P0c = u
            J_lf = nlsys.J_dense(x_LF)
            xe = np.array([x_LF[2]])
            nlsys.nlsystemse[0].V_vec_mag[0] = V0
            Je_full = nlsys.nlsystemse[0].J_dense(xe,return_full=True)
            deltah_deltay = np.zeros((len_x,len(y)))
            deltah_deltay[3:7,0] = Je_full[Fe_ind,V0_ind].ravel() # dFe_dV0
            deltah_deltay[:,P0c_ind] = np.array([0,0,0,-1,0,0,0,1,0])
            deltah_deltay[0,:] = np.array([0,0,-1,-1,0,0,-1,0,0,0,0])
            deltah_deltay[1:,3:] = J_lf #dF_dxlf
            if scale_var == 'matrix':
                deltah_deltay = DH.dot(deltah_deltay.dot(Dy_inv))
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
            return dg_du
        if optimization_method == 'trust-constr':
            ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(2*len_x),np.inf*np.ones(2*len_x),jac=g_jac,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            ineq_constr_fun  = {'type':'ineq','fun':g,'jac':g_jac}
    else:
        ineq_constr_fun = None

    # define linear inequality constraints (bounds) on the control variables
    if ineq_constr != None:
        # define linear inequality constraints (on the control variables)
        lb_ineq_bounds = lb_ineq[:len(u0)]
        ub_ineq_bounds = ub_ineq[:len(u0)]
    else:
        bounds = None

    if optimization_method == 'ipopt':
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
    if optimization_method == 'ipopt' and ineq_constr != None:
        bounds = [(lb,ub) for lb, ub in zip(lb_ineq_bounds,ub_ineq_bounds)]
    elif ineq_constr != None:
        bounds = spo.Bounds(lb_ineq_bounds,ub_ineq_bounds,keep_feasible=stay_within_bounds)
    else:
        bounds = None

    # make sure initial guess satisfies bounds (NB. If adjustments are made, LF is not necessarily satisfied anymore)
    if ineq_constr != None and (optimization_method == 'SLSQP' or stay_within_bounds):
        for ind, x0 in enumerate(u0):
            if lb_ineq_bounds[ind] > x0:
                u0[ind] = lb_ineq_bounds[ind]
            elif ub_ineq_bounds[ind] < x0:
                u0[ind] = ub_ineq_bounds[ind]

    global f_vec_global
    global x_f_vec
    f_vec_global = list()
    f_vec = list()
    x_f_vec = list()
    if optimization_method == 'trust-constr':
        def callback_opf(xk, state):
            f_vec.append(state.fun)
            return False
    elif optimization_method == 'SLSQP':
        f_vec.append(obj(u0))
        def callback_opf(xk):
            f_vec.append(obj(xk))
            return False

    # solve OPF
    opf_start_time = time.time()
    try:
        if ineq_constr_fun != None:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=[ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds, callback=callback_opf)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback_opf)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, u0,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
        else:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, options={'verbose': 1,'maxiter':max_iter}, bounds=bounds, callback=callback_opf)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback_opf)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, u0,jac=obj_grad,  options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
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
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            f_vec = f_vec_global

    if scale_var == 'matrix':
        u_opf = Du_inv.dot(res.x)
    else:
        u_opf = res.x
    # print solution
    q0, P1c, het_net, gas_net, elec_net = solve_ge_lf_in_of(het_net, gas_net, elec_net,u_opf,max_iters=max_iters_lf,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    xmes_opt = het_net.set_x_init(formulation=formulation)
    het_net.reset_network(xmes_opt,formulation=formulation) # if you don't do this first, the update_full will set the wrong values for the half link powers.
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation)
    print('Solution OPF (inequality constraints on control variables: {})'.format(ineq_constr))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/MES.Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('objective function = {}'.format(obj(u_opf)))
    return u_opf, xmes_opt, f_vec, res.fun, res.nfev, res.nit, execution_time, res.success

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_opf_ge_separate_LF_direct():
    """Test OF against the solution of LF, using the integrated MES for load flow, and with inequality constraints on the control variables. The load flow equations are included implicitly. Analytical expression are used for the gradient of the objective function, using the direct approach."""
    # Given + When
    max_iters_lf = 10
    tol = 1e-6
    scale_var = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    het_net_LF, gas_net_LF, elec_net_LF, xmes_LF, _, _ = run_mes_ge_load_flow(max_iter=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation)
    max_iter = 500
    a=0
    b=.01*MES.GHV
    c=1e-6*(MES.GHV)**2
    P0c_init = 1.3*MW
    P0c_lb=1*MES.Pc0_sol
    P0c_ub=1.5*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    x_opf, xmes_opt, _, _, _, _, _, success = run_ge_optimal_load_flow_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,a=a,b=b,c=c,formulation=formulation,ineq_constr=True, approach='direct')

    # Then
    assert success and np.allclose(xmes_opt,xmes_LF)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_opf_ge_separate_LF_adjoint():
    """Test OF against the solution of LF, using the integrated MES for load flow, and with inequality constraints on the control variables. The load flow equations are included implicitly. Analytical expression are used for the gradient of the objective function, using the adjoint approach."""
    # Given + When
    max_iters_lf = 10
    tol = 1e-6
    scale_var = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    het_net_LF, gas_net_LF, elec_net_LF, xmes_LF, _, _ = run_mes_ge_load_flow(max_iter=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation)
    max_iter = 500
    a=0
    b=.01*MES.GHV
    c=1e-6*(MES.GHV)**2
    P0c_init = 1.3*MW
    P0c_lb=1*MES.Pc0_sol
    P0c_ub=1.5*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    x_opf, xmes_opt, _, _, _, _, _, success = run_ge_optimal_load_flow_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,a=a,b=b,c=c,formulation=formulation,ineq_constr=True, approach='adjoint')

    # Then
    assert success and np.allclose(xmes_opt,xmes_LF)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_opf_ge_combined_objective_separate_LF_direct():
    """Test OF against the solution of LF, using the integrated MES for load flow, and with inequality constraints on the control variables. As objective function, the total price for gas input and conversion is used. The load flow equations are included implicitly. Analytical expression are used for the gradient of the objective function, using the direct approach."""
    # Given + When
    max_iters_lf = 10
    tol = 1e-6
    scale_var = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    het_net_LF, gas_net_LF, elec_net_LF, xmes_LF, _, _ = run_mes_ge_load_flow(max_iter=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation)
    max_iter = 500
    P0c_init = 1.3*MW
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.3
    c0c=3e-5
    a1c=0
    b1c=.2
    c1c=2e-5
    x_opf, xmes_opt, _, _, _, _, _, success = run_ge_optimal_load_flow_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,ineq_constr=True,objective='gas_elec', approach='direct')

    # Then
    assert success and np.allclose(xmes_opt,xmes_LF)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_opf_ge_combined_objective_separate_LF_adjoint():
    """Test OF against the solution of LF, using the integrated MES for load flow, and with inequality constraints on the control variables. As objective function, the total price for gas input and conversion is used. The load flow equations are included implicitly. Analytical expression are used for the gradient of the objective function, using the adjoint approach."""
    # Given + When
    max_iters_lf = 10
    tol = 1e-6
    scale_var = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    het_net_LF, gas_net_LF, elec_net_LF, xmes_LF, _, _ = run_mes_ge_load_flow(max_iter=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation)
    max_iter = 500
    P0c_init = 1.3*MW
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.3
    c0c=3e-5
    a1c=0
    b1c=.2
    c1c=2e-5
    x_opf, xmes_opt, _, _, _, _, _, success = run_ge_optimal_load_flow_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,ineq_constr=True,objective='gas_elec', approach='adjoint')

    # Then
    assert success and np.allclose(xmes_opt,xmes_LF)

def xseparate_from_xopf(x_opf):
    xg = x_opf[8:10]
    xe = np.array([x_opf[-1]])
    xc = x_opf[1:4] # these are actually control variables
    return xg, xe, xc

def run_ge_optimal_load_flow_dd(P0c_init=MES.Pc_ic,P0c_lb=1*MES.Pc0_sol,P0c_ub=1.5*MES.Pc0_sol, V0_init=1.1*MES.Vbase,V0_lb=0.8*MES.V0_sol,V0_ub=1*MES.V0_sol,q0c_init=MES.qc_ic,q0c_lb=1*MES.qc0_sol,q0c_ub=1.3*MES.qc0_sol,q1c_init=MES.qc_ic,q1c_lb=.7*MES.qc1_sol,q1c_ub=1*MES.qc1_sol,max_iter=MES.max_iter_outer,tol=MES.tol,scale_var=None,scale_var_params=None,formulation=MES.formulation,a=MES.q0_source**2,b=2*MES.q0_source,c=1,a0=0,b0=.01*MES.GHV,c0=1e-6*MES.GHV**2,a0c=0,b0c=.3,c0c=3e-5,a1c=0,b1c=.2,c1c=2e-5,ineq_constr=True,objective='gas'):
    """Run optimal power flow for the combined gas-electricity network. As many of the coupling energies as possible (based on single-carrier network physics?) are taken as control variables. The decomposed networks are then used for load flow.
    """
    print('\nRunning OPF for gas-electricty network, using the decomposed MES (inequality constraints on control variables: {})'.format(ineq_constr))
    # create networks
    gas_net = create_gas_network()
    xg0 = MES.initialize_gas_network(gas_net)
    elec_net = MES.create_electrical_network()
    xe0 = MES.initialize_electrical_network(elec_net)
    coupling_net = create_coupling_ge_single_network()
    xc0 = initialize_coupling_ge_single_network(coupling_net,formulation=formulation)

    if scale_var == 'matrix' and scale_var_params == None:
        scale_var_params = MES.scale_var_params

    # update the boundary conditions of the single-carrier networks to mathc the intitial guess of OPF
    gas_net = update_bc_gas(gas_net,q0c_init,q1c_init)
    elec_net = update_bc_electrical(elec_net,V0_init,P0c_init)

    # run load flow of the single-carrier networks once, to make sure that initial guess of OPF at least satisfies those LFs
    xg_LF,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    xe_LF,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    # initial guess for OPF
    u_init = np.array([V0_init,q0c_init,q1c_init,P0c_init])
    q0 = gas_net.nodes[0].half_links[0].get_q()
    Q0_gen = elec_net.nodes[0].half_links[0].get_Q() # is generator in elec net, so is <0
    P1_gen = elec_net.nodes[1].half_links[0].get_P() - MES.P1_load
    Q1_gen = elec_net.nodes[1].half_links[0].get_Q() - MES.Q1_load
    slack_init = np.array([q0,P1_gen,Q0_gen,Q1_gen])
    x_opf0 = np.concatenate((u_init,slack_init,xg_LF,xe_LF))

    def cost_function(x_opf):
        """Define the cost function for OPF

        Parameters
        ----------------
        x_opf : np array
            Variable vector used in OPF. Is assumed to be [V0, P0c, q0, q01, p1, delta0, q0c, q1c, P1c, Q0c, Q1c]

        Returns
        -----------
        f : float
            The value of the cost function
        """
        q0 = x_opf[len(u_init)] #<0
        if objective == 'gas':
            f = price_gas(q0,a=a,b=b,c=c)
        elif objective == 'gas_elec':
            P0c = x_opf[len(u_init)-1] #>0
            P1c = x_opf[len(u_init)+1] #>0
            f = price_gas_electricity(q0,P0c,P1c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)
        return f

    # define nonlinear equality constriants (load flow equations)
    nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    def nonlinear_equality_constraints(x_opf,network_c=coupling_net,network_g=gas_net,network_e=elec_net):
        # update BCs of the single-carrier networks
        xg, xe, xc = xseparate_from_xopf(x_opf)
        V0,q0c,q1c,P0c= x_opf[:len(u_init)]
        network_g = update_bc_gas(network_g,q0c,q1c)
        cons_mass = np.array([network_g.nodes[0].node_law()])
        network_e = update_bc_electrical(network_e,V0,P0c)
        P1_gen,Q0_gen,Q1_gen = x_opf[len(u_init)+1:len(u_init)+len(slack_init)]
        network_c = update_bc_coupling_ge(network_c,-Q0_gen,-P1_gen,-Q1_gen)
        # evaluate conservation of energy in the slack and generator node
        network_e.reset_network(xe,formulation=formulation.get('elec'))
        network_e.update(xe,formulation=formulation.get('elec')) # this should set the correct values on the links.
        _,fQ0 = network_e.nodes[0].node_law(network=network_e,scale_var=scale_var,scale_var_params=scale_var_params) + np.array([0,Q0_gen])
        fP1,fQ1 = network_e.nodes[1].node_law(network=network_e,scale_var=scale_var,scale_var_params=scale_var_params) + np.array([P1_gen + MES.P1_load,Q1_gen + MES.Q1_load]) #Node 1 is a slack node, so it has no half links connected to it (yet)
        cons_energy = np.array([fP1,fQ0,fQ1])
        # evaluate load flow equations
        network_g.reset_network(xg,formulation=formulation.get('gas'))
        network_e.reset_network(xe,formulation=formulation.get('elec'))
        network_c.reset_network(xc,formulation=formulation)
        Fg = nlsysg.F(xg)
        Fe = nlsyse.F(xe)
        Fc = nlsysc.F(xc) # This one is underdetermined, i.e. |x| > |F|, but it should be possible to make F
        return np.concatenate((cons_mass,Fg,cons_energy,Fe,Fc))
    lb_nleq = np.zeros(len(slack_init)+len(xg_LF)+len(xe_LF)+2)
    ub_nleq = np.zeros(len(slack_init)+len(xg_LF)+len(xe_LF)+2)
    nonlinear_constraint = spo.NonlinearConstraint(nonlinear_equality_constraints,lb_nleq,ub_nleq)
    if ineq_constr:
        # define linear inequality constraints (on the control variables)
        lb_ineq = -np.inf*np.ones(len(x_opf0)) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
        ub_ineq = np.inf*np.ones(len(x_opf0))
        lb_ineq[:len(u_init)] = np.array([V0_lb,q0c_lb,q1c_lb,P0c_lb])
        ub_ineq[:len(u_init)]  = np.array([V0_ub,q0c_ub,q1c_ub,P0c_ub])
        bounds = spo.Bounds(lb_ineq,ub_ineq)
    else:
        bounds = None

    f_vec = list()
    def callback_opf_dd(xk, state):
        f_vec.append(state.fun)
        return False

    # solve OPF
    res = spo.minimize(cost_function, x_opf0, method='trust-constr', constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol,callback=callback_opf_dd)
    x_opf = res.x
    # print solution
    V0,q0c,q1c,P0c= x_opf[:len(u_init)]
    gas_net = update_bc_gas(gas_net,q0c,q1c)
    elec_net = update_bc_electrical(elec_net,V0,P0c)
    P1_gen,Q0_gen,Q1_gen = x_opf[len(u_init)+1:len(u_init)+len(slack_init)]
    coupling_net = update_bc_coupling_ge(coupling_net,-Q0_gen,-P1_gen,-Q1_gen)
    xg_opt, xe_opt, xc_opt = xseparate_from_xopf(x_opf)
    xmes_opt = np.concatenate((xg_opt,xe_opt,xc_opt[:2],-np.array([P1_gen,Q0_gen,Q1_gen])))
    print('Solution OPF (inequality constraints on control variables: {})'.format(ineq_constr))
    p_sol,q_sol,q_inj = gas_net.update_full(xg_opt,formulation=formulation.get('gas'))
    print('p = {} mbar'.format(p_sol/mbar))
    print('q = {} kg/s'.format(q_sol))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.update_full(xe_opt,formulation=formulation.get('elec'))
    print('delta = {}'.format(delta_sol))
    print('|V| = {} V'.format(V_sol))
    print('|V| = {} p.u.'.format(V_sol/MES.Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.update_full(xc_opt,formulation=formulation)
    print('q hl coupling = {} kg/s'.format([hl.q for node in coupling_net.get_nodes() for hl in node.get_half_links(carriers=['gas'])]))
    print('P hl coupling = {} MW'.format([hl.P/MW for node in coupling_net.get_nodes() for hl in node.get_half_links(carriers=['elec'])]))
    print('Q hl coupling = {} MW'.format([hl.Q/MW for node in coupling_net.get_nodes() for hl in node.get_half_links(carriers=['elec'])]))
    print('objective function = {}'.format(cost_function(x_opf)))
    return x_opf, xmes_opt, f_vec, res.fun, res.nfev, res.nit, res.execution_time, res.success

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_opf_ge_dd():
    """Test OPF against the solution of LF, using the decomposed MES for load flow, and with inequality constraints on the control variables"""
    # Given + When
    max_iter = 300
    tol = 1e-6
    scale_var = None
    P0c_init = 1.1*MES.Pc0_sol
    P0c_lb=1*MES.Pc0_sol
    P0c_ub=1.5*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    q0c_init = .9*MES.qc0_sol
    q0c_lb=1*MES.qc0_sol
    q0c_ub=1.3*MES.qc0_sol
    q1c_init = 1.1*MES.qc1_sol
    q1c_lb=.7*MES.qc1_sol
    q1c_ub=1*MES.qc1_sol
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    x_opf, xmes_opt, _, _, _, _, _, success = run_ge_optimal_load_flow_dd(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0c_init=q0c_init,q0c_lb=q0c_lb,q0c_ub=q0c_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation,objective='gas')

    # Then
    _, _, _, xmes_LF, _, _ = run_mes_ge_load_flow(max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation)
    print('xmes_opt = \n{}\nxmes_LF = \n{}\ndifference = \n{}\nnormalized difference = \n{}\nnormalized difference reversed = \n{}'.format(xmes_opt,xmes_LF,xmes_opt-xmes_LF,(xmes_opt-xmes_LF)/xmes_LF,(xmes_LF-xmes_opt)/xmes_opt))
    rel_tol = 100*tol
    abs_tol = rel_tol
    print('|a-b|=\n{}\natol+rtol*|b|=\n{}'.format(np.abs(xmes_opt-xmes_LF),abs_tol+rel_tol*np.abs(xmes_LF)))
    assert success and np.allclose(xmes_opt,xmes_LF,rtol=rel_tol,atol=abs_tol)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_opf_ge_dd_combined_objective():
    """Test OPF against the solution of LF, using the decomposed MES for load flow, and with inequality constraints on the control variables. As objective function, the total price for gas input and conversion is used."""
    # Given + When
    max_iter = 300
    tol = 1e-6
    scale_var = None
    P0c_init = 1.1*MW
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    q0c_init = 1.5*MES.qc0_sol
    q0c_lb=1*MES.qc0_sol
    q0c_ub=1.3*MES.qc0_sol
    q1c_init = 1.1*MES.qc1_sol
    q1c_lb=1*MES.qc1_sol
    q1c_ub=1.3*MES.qc1_sol
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.5
    c0c=5e-5
    a1c=0
    b1c=b0c
    c1c=c0c
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    x_opf, xmes_opt, f_vec, _, _, _, _, success = run_ge_optimal_load_flow_dd(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0c_init=q0c_init,q0c_lb=q0c_lb,q0c_ub=q0c_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation,objective='gas_elec')

    # Then
    _, _, _, xmes_LF, _, _ = run_mes_ge_load_flow(max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation)
    assert success and np.allclose(xmes_opt,xmes_LF)

def xseparate_from_xopf2(x_opf):
    xg = x_opf[8:10]
    xe = np.array([x_opf[-1]])
    xc = x_opf[2:4] # these are actually control variables
    return xg, xe, xc

def run_ge_optimal_load_flow_dd2(P0c_init=MES.Pc_ic,P0c_lb=.5*MES.Pc0_sol,P0c_ub=1*MES.Pc0_sol, V0_init=1.1*MES.Vbase,V0_lb=0.8*MES.V0_sol,V0_ub=1*MES.V0_sol,q0_init=MES.q0_source,q0_lb=1.3*MES.q0_source,q0_ub=1*MES.q0_source,q1c_init=MES.qc_ic,q1c_lb=.7*MES.qc1_sol,q1c_ub=1*MES.qc1_sol,max_iter=MES.max_iter_outer,tol=MES.tol,scale_var=None,scale_var_params=None,formulation=MES.formulation,a=MES.q0_source**2,b=2*MES.q0_source,c=1,a0=0,b0=.01*MES.GHV,c0=1e-6*MES.GHV**2,a0c=0,b0c=.3,c0c=3e-5,a1c=0,b1c=.2,c1c=2e-5,ineq_constr=True,derivatives=False):
    """Run optimal power flow for the combined gas-electricity network. As many of the coupling energies as possible (based on single-carrier network physics?) are taken as control variables. The decomposed networks are then used for load flow.

    Parameters
    ----------------
    derivatives : bool, optional
        If True, analytical expressions for the gradient and Hessian of the objective function and of the (nonlinear) constraints are used. Otherwise, numerical approximations are used. Default is False.
    """
    print('\nRunning OPF for gas-electricty network, using the decomposed MES with the second node set (inequality constraints on control variables: {})'.format(ineq_constr))
    # create networks
    gas_net = MES.create_gas_network()
    xg0 = MES.initialize_gas_network(gas_net)
    elec_net = MES.create_electrical_network()
    xe0 = MES.initialize_electrical_network(elec_net)
    coupling_net = MES.create_coupling_ge_single_network()
    xc0 = MES.initialize_coupling_ge_single_network(coupling_net)

    if scale_var == 'matrix' and scale_var_params == None:
        scale_var_params = MES.scale_var_params

    # update the boundary conditions of the single-carrier networks to mathc the intitial guess of OPF
    gas_net = update_bc_gas2(gas_net,q1c_init)
    elec_net = update_bc_electrical(elec_net,V0_init,P0c_init)

    # run load flow of the single-carrier networks once, to make sure that initial guess of OPF at least satisfies those LFs
    xg_LF,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    xe_LF,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    # initial guess for OPF
    u_init = np.array([q0_init,V0_init,q1c_init,P0c_init])
    q0_load = gas_net.nodes[0].half_links[0].get_q() - q0_init #>0
    Q0_gen = elec_net.nodes[0].half_links[0].get_Q() # is generator in elec net, so is <0
    P1_gen = elec_net.nodes[1].half_links[0].get_P() - MES.P1_load
    Q1_gen = elec_net.nodes[1].half_links[0].get_Q() - MES.Q1_load
    slack_init = np.array([q0_load,P1_gen,Q0_gen,Q1_gen])
    x_opf0 = np.concatenate((u_init,slack_init,xg_LF,xe_LF))

    q0_ind = 0
    P0c_ind = 3
    P1_gen_ind = 5
    def cost_function(x_opf):
        """Define the cost function for OPF

        Parameters
        ----------------
        x_opf : np array
            Variable vector used in OPF. Is assumed to be [V0, P0c, q0, q01, p1, delta0, q0c, q1c, P1c, Q0c, Q1c]

        Returns
        -----------
        f : float
            The value of the cost function
        """
        q0 = x_opf[q0_ind] #<0
        P0c = x_opf[P0c_ind] #>0
        P1c = -x_opf[P1_gen_ind] #>0
        f = price_gas_electricity(q0,P0c,P1c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)
        return f

    if derivatives:
        # gradient and Hessiand of cost function
        def jac_cost(x_opf):
            df_dy = np.zeros(len(x_opf))
            q0 = x_opf[q0_ind] #<0
            P0c = x_opf[P0c_ind] #>0
            P1c = -x_opf[P1_gen_ind] #>0
            df_dq0, df_dP0c, df_dP1c = price_gas_electricity_first_der(q0,P0c,P1c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)
            df_dy[q0_ind] = df_dq0
            df_dy[P0c_ind] = df_dP0c
            df_dy[P1_gen_ind] = -df_dP1c
            return df_dy
        def hess_cost(x_opf):
            hess_cost_diag = np.zeros(len(x_opf))
            q0 = x_opf[q0_ind] #<0
            P0c = x_opf[P0c_ind] #>0
            P1c = -x_opf[P1_gen_ind] #>0
            d2f_dq02, d2f_dP0c2, d2f_dP1c2 = price_gas_electricity_second_der(q0,P0c,P1c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)
            hess_cost_diag[q0_ind] = d2f_dq02
            hess_cost_diag[P0c_ind] = d2f_dP0c2
            hess_cost_diag[P1_gen_ind] = d2f_dP1c2
            return np.diag(hess_cost_diag)
    # define nonlinear equality constriants (load flow equations)
    nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    Ne = len(elec_net.nodes)
    Fe_ind = nlsyse.FP + [Ne+ind for ind in nlsyse.FQ]
    Ge_ind = [1,Ne+0,Ne+1]
    xelf_ind = nlsyse.xdelta + [Ne+ind for ind in nlsyse.xV]
    V0_ind = Ne
    def nonlinear_equality_constraints(x_opf,network_c=coupling_net,network_g=gas_net,network_e=elec_net):
        # update BCs of the single-carrier networks
        xg, xe, xc = xseparate_from_xopf2(x_opf)
        q0,V0,q1c,P0c= x_opf[:len(u_init)]
        network_g = update_bc_gas2(network_g,q1c)
        q0_load,P1_gen,Q0_gen,Q1_gen = x_opf[len(u_init):len(u_init)+len(slack_init)]
        q01 = xg[0]
        cons_mass = np.array([-q01-q0_load-q0])  #node 0 is slack node, so it has only one half link connected to it with flow 0
        network_e = update_bc_electrical(network_e,V0,P0c)
        network_c = update_bc_coupling_ge2(network_c,q0_load,-Q0_gen,-P1_gen,-Q1_gen)
        # evaluate conservation of energy in the slack and generator node
        network_e.reset_network(xe,formulation=formulation.get('elec'))
        network_e.update(xe,formulation=formulation.get('elec')) # this should set the correct values on the links.
        _,fQ0 = network_e.nodes[0].node_law(network=network_e,scale_var=scale_var,scale_var_params=scale_var_params) + np.array([0,Q0_gen])
        fP1,fQ1 = network_e.nodes[1].node_law(network=network_e,scale_var=scale_var,scale_var_params=scale_var_params) + np.array([P1_gen + MES.P1_load,Q1_gen + MES.Q1_load]) #Node 1 is a slack node, so it has no half links connected to it (yet)
        cons_energy = np.array([fP1,fQ0,fQ1])
        # evaluate load flow equations
        network_g.reset_network(xg,formulation=formulation.get('gas'))
        network_e.reset_network(xe,formulation=formulation.get('elec'))
        network_c.reset_network(xc,formulation=formulation)
        Fg = nlsysg.F(xg)
        Fe = nlsyse.F(xe)
        Fc = nlsysc.F(xc)
        return np.concatenate((cons_mass,Fg,cons_energy,Fe,Fc))
    if derivatives:
        # Jacobian of nonlinear constraints
        def jac_nleq(x_opf,network_c=coupling_net,network_g=gas_net,network_e=elec_net):
            # update BCs of the single-carrier networks
            xg, xe, xc = xseparate_from_xopf2(x_opf)
            q0,V0,q1c,P0c= x_opf[:len(u_init)]
            network_g = update_bc_gas2(network_g,q1c)
            q0_load,P1_gen,Q0_gen,Q1_gen = x_opf[len(u_init):len(u_init)+len(slack_init)]
            network_e = update_bc_electrical(network_e,V0,P0c)
            network_c = update_bc_coupling_ge2(network_c,q0_load,-Q0_gen,-P1_gen,-Q1_gen)
            # reset networks
            network_g.reset_network(xg,formulation=formulation.get('gas'))
            network_e.reset_network(xe,formulation=formulation.get('elec'))
            network_c.reset_network(xc,formulation=formulation)
            # determine LF jacobians
            Jgg_full = nlsysg.J_dense(xg,return_full=True)
            Jee_full = nlsyse.J_dense(xe,return_full=True)
            Jcc = nlsysc.J_dense(xc)
            # create and collect Jacobian of equality constraints
            dH_dy = np.zeros((len(slack_init)+len(xg_LF)+len(xe_LF)+2,len(x_opf)))
            dH_dy[0,0] = -1 #dGg_dq0
            dH_dy[1,2] = -1 #dFg_dq1c
            dH_dy[3:7,1] = Jee_full[Ge_ind+Fe_ind,V0_ind].ravel()#dHe_dV0
            dH_dy[6,3] = -1#dFe_dP0c
            dH_dy[7:,2:4] = Jcc #dFc_dxc
            dH_dy[0,4:] = np.array([-1,0,0,0,-1,0,0]) #dGg_dx
            dH_dy[1:3,8:10] = Jgg_full[1:,[0,2]] #dFg_dxlf
            dH_dy[3:6,10] = Jee_full[Ge_ind,xelf_ind].ravel() #dGe_dxlf
            dH_dy[3,5] = 1 #dGe_dP1,c
            dH_dy[4,6] = 1 #dGe_dQ0,c
            dH_dy[5,7] = 1 #dGe_dQ1,c
            dH_dy[6,10] = Jee_full[Fe_ind,xelf_ind].ravel() #dFe_dxlf
            dH_dy[7,4] = -MES.GHV*MES.eta_GG0 #dFc_dq0,c
            dH_dy[8,5] = -1 #dFc_dP1,c
            return dH_dy
    lb_nleq = np.zeros(len(slack_init)+len(xg_LF)+len(xe_LF)+2)
    ub_nleq = np.zeros(len(slack_init)+len(xg_LF)+len(xe_LF)+2)
    if derivatives:
        nonlinear_constraint = spo.NonlinearConstraint(nonlinear_equality_constraints,lb_nleq,ub_nleq,jac=jac_nleq)
    else:
        nonlinear_constraint = spo.NonlinearConstraint(nonlinear_equality_constraints,lb_nleq,ub_nleq)
    if ineq_constr:
        # define linear inequality constraints (on the control variables)
        lb_ineq = -np.inf*np.ones(len(x_opf0)) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
        ub_ineq = np.inf*np.ones(len(x_opf0))
        lb_ineq[:len(u_init)] = np.array([q0_lb,V0_lb,q1c_lb,P0c_lb])
        ub_ineq[:len(u_init)]  = np.array([q0_ub,V0_ub,q1c_ub,P0c_ub])
        bounds = spo.Bounds(lb_ineq,ub_ineq)
    else:
        bounds = None

    f_vec = list()
    def callback_opf_dd2(xk, state):
        f_vec.append(state.fun)
        return False

    # solve OPF
    if derivatives:
        res = spo.minimize(cost_function, x_opf0, method='trust-constr', jac=jac_cost,hess=hess_cost, constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol,callback=callback_opf_dd2)
    else:
        res = spo.minimize(cost_function, x_opf0, method='trust-constr', constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol,callback=callback_opf_dd2)
    x_opf = res.x
    # print solution
    q0,V0,q1c,P0c= x_opf[:len(u_init)]
    xg_opt, xe_opt, xc_opt = xseparate_from_xopf2(x_opf)
    print('q0 sol = {}'.format(q0))
    gas_net = update_bc_gas2(gas_net,q1c)
    gas_net.reset_network(xg_opt,formulation=formulation.get('gas'))
    elec_net = update_bc_electrical(elec_net,V0,P0c)
    elec_net.reset_network(xe_opt,formulation=formulation.get('elec'))
    q0_load,P1_gen,Q0_gen,Q1_gen = x_opf[len(u_init):len(u_init)+len(slack_init)]
    coupling_net = update_bc_coupling_ge2(coupling_net,q0_load,-Q0_gen,-P1_gen,-Q1_gen)
    xmes_opt = np.concatenate((xg_opt,xe_opt,np.array([q0_load,xc_opt[0]]),-np.array([P1_gen,Q0_gen,Q1_gen])))
    print('Solution OPF (inequality constraints on control variables: {})'.format(ineq_constr))
    p_sol,q_sol,q_inj = gas_net.update_full(xg_opt,formulation=formulation.get('gas'))
    print('p = {} mbar'.format(p_sol/mbar))
    print('q = {} kg/s'.format(q_sol))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.update_full(xe_opt,formulation=formulation.get('elec'))
    print('delta = {}'.format(delta_sol))
    print('|V| = {} V'.format(V_sol))
    print('|V| = {} p.u.'.format(V_sol/MES.Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.update_full(xc_opt,formulation=formulation)
    print('q hl coupling = {} kg/s'.format([hl.q for node in coupling_net.get_nodes() for hl in node.get_half_links(carriers=['gas'])]))
    print('P hl coupling = {} MW'.format([hl.P/MW for node in coupling_net.get_nodes() for hl in node.get_half_links(carriers=['elec'])]))
    print('Q hl coupling = {} MW'.format([hl.Q/MW for node in coupling_net.get_nodes() for hl in node.get_half_links(carriers=['elec'])]))
    print('objective function = {}'.format(cost_function(x_opf)))
    return x_opf, xmes_opt, f_vec, res.fun, res.nfev, res.nit, res.execution_time, res.success

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_opf_ge_dd2():
    """Test OPF against the solution of LF, using the decomposed MES for load flow, and with inequality constraints on the control variables. As objective function, the total price for gas input and conversion is used. The second node set is used."""
    # Given + When
    max_iter = 300
    tol = 1e-6
    scale_var = None
    P0c_init = 1.1*MES.Pc0_sol
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    q0_init = 1.1*MES.q0_source
    q0_lb=1.3*MES.q0_source
    q0_ub=1*MES.q0_source
    q1c_init = 1.1*MES.qc1_sol
    q1c_lb=.7*MES.qc1_sol
    q1c_ub=1*MES.qc1_sol
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.3
    c0c=3e-5
    a1c=0
    b1c=.2
    c1c=2e-5
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    x_opf, xmes_opt, _, _, _, _, _, success = run_ge_optimal_load_flow_dd2(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_init=q0_init,q0_lb=q0_lb,q0_ub=q0_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation)

    # Then
    _, _, _, xmes_LF, _, _ = run_mes_ge_load_flow(max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation)
    assert success and np.allclose(xmes_opt,xmes_LF)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_opf_ge_dd2_derivatives():
    """Test OPF against the solution of LF, using the decomposed MES for load flow, and with inequality constraints on the control variables. The gradient and Hessian of the objective function, and the Jacobian of the equality constraints are determined analytically. As objective function, the total price for gas input and conversion is used. The second node set is used."""
    # Given + When
    max_iter = 300
    tol = 1e-6
    scale_var = None
    P0c_init = 1.1*MES.Pc0_sol
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    q0_init = 1.1*MES.q0_source
    q0_lb=1.3*MES.q0_source
    q0_ub=1*MES.q0_source
    q1c_init = 1.1*MES.qc1_sol
    q1c_lb=.7*MES.qc1_sol
    q1c_ub=1*MES.qc1_sol
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.3
    c0c=3e-5
    a1c=0
    b1c=.2
    c1c=2e-5
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    x_opf, xmes_opt, _, _, _, _, _, success = run_ge_optimal_load_flow_dd2(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_init=q0_init,q0_lb=q0_lb,q0_ub=q0_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation,derivatives=True)

    # Then
    _, _, _, xmes_LF, _, _ = run_mes_ge_load_flow(max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation)
    assert success and np.allclose(xmes_opt,xmes_LF)

def run_ge_optimal_load_flow_dd2_separate_LF(P0c_init=MES.Pc_ic,P0c_lb=.5*MES.Pc0_sol,P0c_ub=1*MES.Pc0_sol, V0_init=1.1*MES.Vbase,V0_lb=1*MES.V0_sol,V0_ub=1.3*MES.V0_sol,q0_init=MES.q0_source,q0_lb=1.3*MES.q0_source,q0_ub=1*MES.q0_source,q1c_init=MES.qc_ic,q1c_lb=.7*MES.qc1_sol,q1c_ub=1*MES.qc1_sol,max_iter=MES.max_iter_outer,max_iters_lf=10,tol=MES.tol,scale_var=None,scale_var_params=None,formulation=MES.formulation,a=MES.q0_source**2,b=2*MES.q0_source,c=1,a0=0,b0=.01*MES.GHV,c0=1e-6*MES.GHV**2,a0c=0,b0c=.3,c0c=3e-5,a1c=0,b1c=.2,c1c=2e-5,ineq_constr=True,Fc_eq_constr=True,approach='direct'):
    """Run optimal power flow for the combined gas-electricity network. As many of the coupling energies as possible (based on single-carrier network physics?) are taken as control variables. The decomposed networks are then used for load flow.

    Parameters
    ----------------
    Fc_eq_constr : bool, optional
        If True, the load flow equations of the coupling network are (still) taken as equality constraints. If False, they are also included implicitly. Default is True
    """
    if not Fc_eq_constr:
        raise NotImplementedError("OPF with implicit LF is not implemented if Fc are also taken implicitly")
    print('\nRunning OPF for gas-electricty network, using the decomposed MES with the second node set, with LF separate (Fc as equality constraints: {}, inequality constraints on control variables: {})'.format(Fc_eq_constr,ineq_constr))
    # create networks
    gas_net = MES.create_gas_network()
    gas_net.initialize()
    elec_net = MES.create_electrical_network()
    elec_net.initialize()
    coupling_net = MES.create_coupling_ge_single_network()
    coupling_net.initialize()

    if scale_var == 'matrix' and scale_var_params == None:
        scale_var_params = MES.scale_var_params

    # update the boundary conditions of the single-carrier networks to mathc the intitial guess of OPF
    gas_net = update_bc_gas2(gas_net,q1c_init)
    elec_net = update_bc_electrical(elec_net,V0_init,P0c_init)

    # initial guess for OPF
    x_opf0 = np.array([q0_init,V0_init,q1c_init,P0c_init])

    def cost_function(x_opf,network_c=coupling_net,network_g=gas_net,network_e=elec_net):
        """Define the cost function for OPF

        Parameters
        ----------------
        x_opf : np array
            Variable vector used in OPF. Is assumed to be [V0, P0c, q0, q01, p1, delta0, q0c, q1c, P1c, Q0c, Q1c]

        Returns
        -----------
        f : float
            The value of the cost function
        """
        q0,V0,q1c,P0c= x_opf
        network_e = update_bc_electrical(network_e,V0,P0c)
        xe = network_e.set_x_init(formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
        network_e.reset_network(xe,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
        xe_LF,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iters_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        P1c = -(network_e.nodes[1].half_links[0].get_P() - MES.P1_load)
        f = price_gas_electricity(q0,P0c,P1c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)
        return f

    # gradient of objective function
    def deltaf_deltau(x_opf):
        """Partial derivative of objective function to control variables u"""
        df_du = np.zeros(len(x_opf))
        q0,V0,q1c,P0c= x_opf
        # should actually also calculate updated P1c, but the objective function is a sum of quadratic functions, so deriviative to P0c is independent of q0 and P1c
        df_dq0, df_dP0c, _ = price_gas_electricity_first_der(q0,P0c,0,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)
        df_du[0] = df_dq0
        df_du[3] = df_dP0c
        return df_du
    # define nonlinear equality constriants (load flow equations)
    nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    Ne = len(elec_net.nodes)
    Fe_ind = nlsyse.FP + [Ne+ind for ind in nlsyse.FQ]
    Ge_ind = [1,Ne+0,Ne+1]
    xelf_ind = nlsyse.xdelta + [Ne+ind for ind in nlsyse.xV]
    V0_ind = Ne
    len_x = 7
    dH_dx = np.zeros((len_x,len_x))
    dH_du = np.zeros((len_x,4))
    def par_der(x_opf,network_c=coupling_net,network_g=gas_net,network_e=elec_net):
        """Partial derivative of equality constraints H and of objective function. (In separate function to try to reduce number of times LF is solved)"""
        q0,V0,q1c,P0c= x_opf
        network_g = update_bc_gas2(network_g,q1c)
        xg = network_g.set_x_init(formulation=formulation.get('gas'))
        network_g.reset_network(xg,formulation=formulation.get('gas'))
        xg_LF,iters,err_vec,p_sol,q_sol,q_inj = network_g.solve_network(tol,max_iters_lf,solver='NR',formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
        network_e = update_bc_electrical(network_e,V0,P0c)
        xe = network_e.set_x_init(formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
        network_e.reset_network(xe,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
        xe_LF,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = network_e.solve_network(tol,max_iters_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        deltaf_deltax = np.zeros(len_x)
        P1c = -(network_e.nodes[1].half_links[0].get_P() - MES.P1_load)
        _, _, df_dP1c = price_gas_electricity_first_der(q0,P0c,P1c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)
        deltaf_deltax[1] = -df_dP1c #df_dP1,c
        # determine LF jacobians
        Jgg_full = nlsysg.J_dense(xg,return_full=True)
        Jee_full = nlsyse.J_dense(xe,return_full=True)
        dH_du[0,0] = -1 #dGg_dq0
        dH_du[1,2] = -1 #dFg_dq1c
        dH_du[3:7,1] = Jee_full[Ge_ind+Fe_ind,V0_ind].ravel()#dHe_dV0
        dH_du[6,3] = -1#dFe_dP0c
        dH_dx[0,:] = np.array([-1,0,0,0,-1,0,0]) #dGg_dx
        dH_dx[1:3,4:6] = Jgg_full[1:,[0,2]] #dFg_dxlf
        dH_dx[3:6,6] = Jee_full[Ge_ind,xelf_ind].ravel() #dGe_dxlf
        dH_dx[3,1] = 1 #dGe_dP1,c
        dH_dx[4,2] = 1 #dGe_dQ0,c
        dH_dx[5,3] = 1 #dGe_dQ1,c
        dH_dx[6,6] = Jee_full[Fe_ind,xelf_ind].ravel() #dFe_dxlf
        return deltaf_deltax,dH_du,dH_dx

    def jac_cost(x_opf,network_c=coupling_net,network_g=gas_net,network_e=elec_net,method=approach):
        """Gradient vector / Jacobian of objective function"""
        df_du = deltaf_deltau(x_opf) # first part of gradient
        deltaf_deltax,dH_du,dH_dx = par_der(x_opf)
        if method == 'direct':
            w = np.linalg.solve(dH_dx,-dH_du)
            df_du += np.dot(deltaf_deltax,w)
        elif method == 'adjoint':
            v = np.linalg.solve(np.transpose(dH_dx),deltaf_deltax)
            df_du += np.dot(v,-dH_du)
        return df_du

    def nonlinear_equality_constraints(x_opf,network_c=coupling_net,network_g=gas_net,network_e=elec_net):
        # update BCs of the single-carrier networks
        q0,V0,q1c,P0c= x_opf
        network_g = update_bc_gas2(network_g,q1c)
        xg = network_g.set_x_init(formulation=formulation.get('gas'))
        network_g.reset_network(xg,formulation=formulation.get('gas'))
        xg_LF,iters,err_vec,p_sol,q_sol,q_inj = network_g.solve_network(tol,max_iters_lf,solver='NR',formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
        q0_load = network_g.nodes[0].half_links[0].get_q() - q0 #>0
        if q0_load <= 0:
            warnings.warn('encountered a negative q0c. It is set equal to -q0 instead.')
            print('encountered a negative q0c in iteration {}. It is set equal to -q0 instead.')
            q0_load = -q0
        network_e = update_bc_electrical(network_e,V0,P0c)
        xe = network_e.set_x_init(formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
        network_e.reset_network(xe,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
        xe_LF,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = network_e.solve_network(tol,max_iters_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        Q0_gen = network_e.nodes[0].half_links[0].get_Q() # is generator in elec net, so is <0
        P1_gen = network_e.nodes[1].half_links[0].get_P() - MES.P1_load
        Q1_gen = network_e.nodes[1].half_links[0].get_Q() - MES.Q1_load
        network_c = update_bc_coupling_ge2(network_c,q0_load,-Q0_gen,-P1_gen,-Q1_gen)
        xc = np.array([q1c,P0c])
        network_c.reset_network(xc,formulation=formulation)
        Fc = nlsysc.F(xc)
        return Fc

    def jac_nleq(x_opf,network_c=coupling_net,network_g=gas_net,network_e=elec_net,method=approach):
        # update BCs of the single-carrier networks
        q0,V0,q1c,P0c= x_opf
        _,dH_du,dH_dx = par_der(x_opf) # this runs loadflow and (fully) updates the gas and electrical network
        q0_load = network_g.nodes[0].half_links[0].get_q() - q0 #>0
        if q0_load <= 0:
            warnings.warn('encountered a negative q0c. It is set equal to -q0 instead.')
            print('encountered a negative q0c in iteration {}. It is set equal to -q0 instead.')
            q0_load = -q0
        Q0_gen = network_e.nodes[0].half_links[0].get_Q() # is generator in elec net, so is <0
        P1_gen = network_e.nodes[1].half_links[0].get_P() - MES.P1_load
        Q1_gen = network_e.nodes[1].half_links[0].get_Q() - MES.Q1_load
        network_c = update_bc_coupling_ge2(network_c,q0_load,-Q0_gen,-P1_gen,-Q1_gen)
        # reset networks
        xc = np.array([q1c,P0c])
        network_c.reset_network(xc,formulation=formulation)
        Jcc = nlsysc.J_dense(xc)
        # create and collect Jacobian of equality constraints
        deltaFc_deltau = np.zeros((2,len(x_opf)))
        deltaFc_deltau[:,2:] = Jcc #dFc_dxc
        deltaFc_deltax = np.zeros((2,len_x))
        deltaFc_deltax[0,0] = -MES.GHV*MES.eta_GG0 #dFc_dq0,c
        deltaFc_deltax[1,1] = -1 #dFc_dP1,c
        dFc_du = deltaFc_deltau.copy()
        if method == 'direct':
            w = np.linalg.solve(dH_dx,-dH_du)
            dFc_du += np.dot(deltaFc_deltax,w)
        elif method == 'adjoint':
            v = np.linalg.solve(np.transpose(dH_dx),np.transpose(deltaFc_deltax))
            dFc_du += np.dot(np.transpose(v),-dH_du)
        return dFc_du
    lb_nleq = np.zeros(2)
    ub_nleq = np.zeros(2)
    nonlinear_constraint = spo.NonlinearConstraint(nonlinear_equality_constraints,lb_nleq,ub_nleq,jac=jac_nleq)
    if ineq_constr:
        # define linear inequality constraints (on the control variables)
        lb_ineq = np.array([q0_lb,V0_lb,q1c_lb,P0c_lb])
        ub_ineq = np.array([q0_ub,V0_ub,q1c_ub,P0c_ub])
        bounds = spo.Bounds(lb_ineq,ub_ineq)
    else:
        bounds = None

    f_vec = list()
    def callback_opf_dd2_separate_LF(xk,state):
        f_vec.append(state.fun)
        return False
    # solve OPF
    res = spo.minimize(cost_function, x_opf0, method='trust-constr', jac=jac_cost,constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol,callback=callback_opf_dd2_separate_LF)
    x_opf = res.x
    # print solution
    q0,V0,q1c,P0c= x_opf
    gas_net = update_bc_gas2(gas_net,q1c)
    xg = gas_net.set_x_init(formulation=formulation.get('gas'))
    gas_net.reset_network(xg,formulation=formulation.get('gas'))
    xg_opt,_,_,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iters_lf,solver='NR',formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    q0_load = gas_net.nodes[0].half_links[0].get_q() - q0 #>0
    elec_net = update_bc_electrical(elec_net,V0,P0c)
    xe = elec_net.set_x_init(formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net.reset_network(xe,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    xe_opt,_,_,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iters_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    Q0_gen = elec_net.nodes[0].half_links[0].get_Q() # is generator in elec net, so is <0
    P1_gen = elec_net.nodes[1].half_links[0].get_P() - MES.P1_load
    Q1_gen = elec_net.nodes[1].half_links[0].get_Q() - MES.Q1_load
    coupling_net = update_bc_coupling_ge2(coupling_net,q0_load,-Q0_gen,-P1_gen,-Q1_gen)
    xc = coupling_net.set_x_init(formulation=formulation)
    coupling_net.reset_network(xc,formulation=formulation)
    xc_opt,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.solve_network(tol,max_iters_lf,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    xmes_opt = np.concatenate((xg_opt,xe_opt,np.array([q0_load,q1c,-P1_gen,-Q0_gen,-Q1_gen])))
    print('q0 sol = {}'.format(q0))
    print('Solution OPF (inequality constraints on control variables: {})'.format(ineq_constr))
    print('p = {} mbar'.format(p_sol/mbar))
    print('q = {} kg/s'.format(q_sol))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    print('delta = {}'.format(delta_sol))
    print('|V| = {} V'.format(V_sol))
    print('|V| = {} p.u.'.format(V_sol/MES.Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('q hl coupling = {} kg/s'.format([hl.q for node in coupling_net.get_nodes() for hl in node.get_half_links(carriers=['gas'])]))
    print('P hl coupling = {} MW'.format([hl.P/MW for node in coupling_net.get_nodes() for hl in node.get_half_links(carriers=['elec'])]))
    print('Q hl coupling = {} MW'.format([hl.Q/MW for node in coupling_net.get_nodes() for hl in node.get_half_links(carriers=['elec'])]))
    print('objective function = {}'.format(cost_function(x_opf)))
    return x_opf, xmes_opt, f_vec, res.fun, res.nfev, res.nit, res.execution_time, res.success

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_opf_ge_dd2_separate_LF_Fc_equality_direct():
    """Test OPF against the solution of LF, using the decomposed MES for load flow, and with inequality constraints on the control variables. The gas and electrical load flow equations are included implicitly. Analytical expression are used for the gradient of the objective function, using the direct approach. The second node set is used."""
    # Given + When
    max_iter = 300
    max_iters_lf = 20
    tol = 1e-6
    scale_var = None
    P0c_init = 1.1*MES.Pc0_sol
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=1*MES.V0_sol
    V0_ub=1.5*MES.V0_sol
    q0_init = 1.5*MES.q0_source
    q0_lb=1.3*MES.q0_source
    q0_ub=1*MES.q0_source
    q1c_init = 1.1*MES.qc1_sol
    q1c_lb=1*MES.qc1_sol
    q1c_ub=1.3*MES.qc1_sol
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.5
    c0c=5e-5
    a1c=0
    b1c=b0c
    c1c=c0c
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    x_opf, xmes_opt, f_vec, _, _, _, _, success = run_ge_optimal_load_flow_dd2_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_init=q0_init,q0_lb=q0_lb,q0_ub=q0_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,Fc_eq_constr=True,approach='direct')

    # Then
    _, _, _, xmes_LF, _, _ = run_mes_ge_load_flow(max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation)
    assert success and np.allclose(xmes_opt,xmes_LF)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_opf_ge_dd2_separate_LF_Fc_equality_adjoint():
    """Test OPF against the solution of LF, using the decomposed MES for load flow, and with inequality constraints on the control variables. The gas and electrical load flow equations are included implicitly. Analytical expression are used for the gradient of the objective function, using the direct approach. The second node set is used."""
    # Given + When
    max_iter = 300
    max_iters_lf = 20
    tol = 1e-6
    scale_var = None
    P0c_init = 1.1*MES.Pc0_sol
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=1*MES.V0_sol
    V0_ub=1.5*MES.V0_sol
    q0_init = 1.5*MES.q0_source
    q0_lb=1.3*MES.q0_source
    q0_ub=1*MES.q0_source
    q1c_init = 1.3*MES.qc1_sol
    q1c_lb=1*MES.qc1_sol
    q1c_ub=1.3*MES.qc1_sol
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.3
    c0c=3e-5
    a1c=0
    b1c=b0c
    c1c=c0c
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    x_opf, xmes_opt, f_vec, _, _, _, _, success = run_ge_optimal_load_flow_dd2_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_init=q0_init,q0_lb=q0_lb,q0_ub=q0_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,Fc_eq_constr=True,approach='adjoint')

    # Then
    _, _, _, xmes_LF, _, _ = run_mes_ge_load_flow(max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation)
    assert success and np.allclose(xmes_opt,xmes_LF)

def price_electricity_heat(P0c,P1c,dphi0c,dphi1c,a0=0,b0=.3,c0=3e-5,a1=0,b1=.2,c1=2e-5,a2=0,b2=.04,c2=4e-4,a3=0,b3=.05,c3=4.5e-4,scale_var=None,scale_var_params=None,fb=None):
    """Determine the cost of the active power and heat power generation of the coupling.

    Parameters
    ----------
    P0c : float
        Active power produced by coupling 0, in W. Assumed to be positive.
    P1c : float
        Active power produced by coupling 1, in W. Assumed to be positive
    dphi0c : float
        Heat power produced by coupling 0, in W. Assumed to be positive.
    dphi1c : float
        Heat power produced by coupling 1, in W. Assumed to be positive.
    a0, a1, a2, a3 : float
        Parameter of price function, in euros.
    b0, b1, b2, b3 : float
        Parameter of price function, in euros/(kg/s) for b0 or in euros/W for b0c and b1c.
    c0, c1, c2, c3 : float
        Parameter of price function, in euros/(kg/s)^2 for c0 or in euros/W^2 for c0c and c1c.

    Returns
    -------
    f : float
        Total price of the conversions of gas to electricity and heat, in euros.
    """
    f = a0 + b0*P0c + c0*P0c**2 + a1 + b1*P1c + c1*P1c**2 + a2 + b2*dphi0c + c2*dphi0c**2 + a3 + b3*dphi1c + c3*dphi1c**2
    if scale_var == 'matrix':
        f *= (1/fb)
    return f

def price_electricity_heat_first_der(P0c,P1c,dphi0c,dphi1c,a0=0,b0=.3,c0=3e-5,a1=0,b1=.2,c1=2e-5,a2=0,b2=.04,c2=4e-4,a3=0,b3=.05,c3=4.5e-4,scale_var=None,scale_var_params=None,fb=None):
    """First (partial) derivative of the cost of the active power and heat power generation of the coupling.

    Parameters
    ----------
    P0c : float
        Active power produced by coupling 0, in W. Assumed to be positive.
    P1c : float
        Active power produced by coupling 1, in W. Assumed to be positive
    dphi0c : float
        Heat power produced by coupling 0, in W. Assumed to be positive.
    dphi1c : float
        Heat power produced by coupling 1, in W. Assumed to be positive
    a0, a1, a2, a3 : float
        Parameter of price function, in euros.
    b0, b1, b2, b3 : float
        Parameter of price function, in euros/(kg/s) for b0 or in euros/W for b0c and b1c.
    c0, c1, c2, c3 : float
        Parameter of price function, in euros/(kg/s)^2 for c0 or in euros/W^2 for c0c and c1c.

    Returns
    -------
    df_dP0c, df_dP1c, df_dphi0c, df_dphi1c  : float
        First (partial) derivatives of the total price of the conversions of gas to electricity and heat, in euros.
    """
    df_dP0c, df_dP1c, df_dphi0c, df_dphi1c = b0 + 2*c0*P0c, b1 + 2*c1*P1c, b2 + 2*c2*dphi0c, b3 + 2*c3*dphi1c
    if scale_var == 'matrix':
        df_dP0c *= (scale_var_params.get('Sbase')/fb)
        df_dP1c  *= (scale_var_params.get('Sbase')/fb)
        df_dphi0c *= (scale_var_params.get('phibase')/fb)
        df_dphi1c  *= (scale_var_params.get('phibase')/fb)
    return df_dP0c, df_dP1c, df_dphi0c, df_dphi1c

def price_electricity_heat_second_der(P0c,P1c,dphi0c,dphi1c,a0=0,b0=.3,c0=3e-5,a1=0,b1=.2,c1=2e-5,a2=0,b2=.04,c2=4e-4,a3=0,b3=.05,c3=4.5e-4,scale_var=None,scale_var_params=None,fb=None):
    """Second (partial) derivatives of the cost of the active power and heat power generation of the coupling.

    Parameters
    ----------
    P0c : float
        Active power produced by coupling 0, in W. Assumed to be positive.
    P1c : float
        Active power produced by coupling 1, in W. Assumed to be positive
    dphi0c : float
        Heat power produced by coupling 0, in W. Assumed to be positive.
    dphi1c : float
        Heat power produced by coupling 1, in W. Assumed to be positive.
    a0, a1, a2, a3 : float
        Parameter of price function, in euros.
    b0, b1, b2, b3 : float
        Parameter of price function, in euros/(kg/s) for b0 or in euros/W for b0c and b1c.
    c0, c1, c2, c3 : float
        Parameter of price function, in euros/(kg/s)^2 for c0 or in euros/W^2 for c0c and c1c.

    Returns
    -------
    d2f_dP0c2, d2f_dP1c2, d2f_dphi0c2, d2f_dphi1c2  : float
        Second (partial) derivatives of the total price of the conversions of gas to electricity and heat, in euros.
    """
    d2f_dP0c2, d2f_dP1c2, d2f_dphi0c2, d2f_dphi1c2 = 2*c0, 2*c1, 2*c2, 2*c3
    if scale_var == 'matrix':
        d2f_dP0c2 *= ((scale_var_params.get('Sbase')**2)/fb)
        d2f_dP1c2  *= ((scale_var_params.get('Sbase')**2)/fb)
        d2f_dphi0c2 *= ((scale_var_params.get('phibase')**2)/fb)
        d2f_dphi1c2  *= ((scale_var_params.get('phibase')**2)/fb)
    return d2f_dP0c2, d2f_dP1c2, d2f_dphi0c2, d2f_dphi1c2

def update_bc_mes_eh(het_net, elec_net, heat_net, V0, q0c, q1c,scale_var=None,scale_var_params=None):
    """Update the boundary conditions of the electricity-heat network, based on the control variables of OF"""
    # print('in update mes: V0={}kV, q0c={}, q1c={}'.format(V0/kV, q0c, q1c))
    if scale_var == 'per_unit':
        V0 = V0*scale_var_params.get('Vbase')
        q0c = q0c*scale_var_params.get('qbase')
        q1c = q1c*scale_var_params.get('qbase')
    elec_net.nodes[0].V = V0
    het_net.nodes[2].V = V0
    het_net.nodes[4].half_links[0].q = -q0c # source. Assume q0c > 0, then the half link flow has to be -q0c
    het_net.nodes[5].half_links[0].q = -q1c # source.
    return het_net, elec_net, heat_net

def run_eh_optimal_load_flow(V0_init=1.1*MES.Vbase,V0_bounds=np.array([0.8*MES.V0_sol,1*MES.V0_sol]),q0c_init=1.3*MES.qc0_sol_CHP,q0c_bounds=np.array([1*MES.qc0_sol_CHP,1.5*MES.qc0_sol_CHP]),q1c_init=1.3*MES.qc1_sol_CHP,q1c_bounds=np.array([1*MES.qc1_sol_CHP,1.5*MES.qc1_sol_CHP]),delta0_bounds=np.array([-np.pi,np.pi]),m01_bounds=np.array([-3*MES.m01_sol,3*MES.m01_sol]),m0_bounds=np.array([0,5*MES.m0_sink]),m1_bounds=np.array([0,5*MES.m1_sink]),p1_bounds=np.array([10,5*MES.ph1_sol]),Ts0_bounds=np.array([60,140]),Ts1_bounds=np.array([60,140]),Tr0_bounds=np.array([10,60]),Tr1_bounds=np.array([10,60]),P0c_init=1.1*MES.Pc0_sol,P0c_bounds=np.array([0,3*MES.Pc0_sol]),P1c_init=1.1*MES.Pc1_sol,P1c_bounds=np.array([0,3*MES.Pc1_sol]),Q0c_bounds=np.array([-3*MES.Qc0_sol,3*MES.Qc0_sol]),Q1c_bounds=np.array([-3*MES.Qc1_sol,3*MES.Qc1_sol]),m0c_bounds=np.array([0,3*MES.mc0_sol]),m1c_bounds=np.array([0,3*MES.mc1_sol]),dphi0c_init=1.1*MES.phic0_sol,dphi0c_bounds=np.array([0,3*MES.phic0_sol]), dphi1c_init=1.1*MES.phic1_sol,dphi1c_bounds=np.array([0,3*MES.phic1_sol]),max_iter=MES.max_iter_outer,tol=MES.tol,scale_var=None,scale_var_params=None,a0=0,b0=.3,c0=3e-5,a1=0,b1=.2,c1=2e-5,a2=0,b2=.04,c2=4e-4,a3=0,b3=.05,c3=4.5e-4,formulation=MES.formulation,ineq_constr='control',derivatives=False,optimization_method='trust-constr',stay_within_bounds=False,fb=None):
    """Run optimal load flow for the combined electricity-heat network.

    Parameters
    ----------------
    derivatives : bool, optional
        If True, analytical expressions for the gradient and Hessian of the objective function and of the (nonlinear) constraints are used. Otherwise, numerical approximations are used. Default is False.
    """
    print('\nRunning OF for electricty-heat network, using the integrated MES (method: {}, ineq. constr. on: {}, hard bounds: {}, an der: {}, scaling: {})'.format(optimization_method,ineq_constr,stay_within_bounds,derivatives,scale_var))
    # create network
    het_net, elec_net, heat_net = MES.create_mes_eh_network()

    if scale_var == 'matrix' and scale_var_params == None:
        scale_var_params = MES.scale_var_params

    # update the boundary conditions of the MES to match the initial guess of opf
    het_net, elec_net, heat_net = update_bc_mes_eh(het_net, elec_net, heat_net, V0_init, q0c_init, q1c_init)

    # run steady-state load flow once, to make sure that hte intiial guess of of is at least a solution of LF
    x0 = MES.initialize_mes_eh_network(het_net)
    # x_LF,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,10,solver='NR',formulation=formulation)
    # print('Final LF error for initial guess = {:.4e}, in {} iterations'.format(err_vec[-1],iters))

    # initial guess for OF (unscaled)
    u_init = np.array([V0_init,q0c_init,q1c_init])
    x_opf0 = np.concatenate((u_init,x0))
    P0c_ind = len(u_init) + 9 # index of P0c within x
    P1c_ind = len(u_init) + 10 # index of P1c within x
    dphi0c_ind = len(u_init) + 15 # index of dphi0c within x
    dphi1c_ind = len(u_init) + 16 # index of dphi1c within x
    x_opf0[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]] = [P0c_init, P1c_init, dphi0c_init, dphi1c_init]

    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    Ne = len(elec_net.nodes)
    Fe_ind = nlsys.nlsystemse[0].FP + [Ne+ind for ind in nlsys.nlsystemse[0].FQ]
    V0_ind = Ne

    DF = nlsys.DF()
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dy = np.diag(np.concatenate((np.array([1/scale_var_params.get('Vbase'),1/scale_var_params.get('qbase'),1/scale_var_params.get('qbase')]),Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((np.array([scale_var_params.get('Vbase'),scale_var_params.get('qbase'),scale_var_params.get('qbase')]),1/Dx.data[0])))
        x_opf0 = Dy.dot(x_opf0) # scale y
    else:
        Dy=np.eye(len(x_opf0))
        Dy_inv=np.eye(len(x_opf0))

    if scale_var == 'per_unit':
        a0 = a0/fb
        b0 = b0/(fb/scale_var_params.get('Sbase'))
        c0 = c0/(fb/(scale_var_params.get('Sbase')**2))
        a1 = a1/fb
        b1 = b1/(fb/scale_var_params.get('Sbase'))
        c1 = c1/(fb/(scale_var_params.get('Sbase')**2))
        a2 = a2/fb
        b2 = b2/(fb/scale_var_params.get('phibase'))
        c2 = c2/(fb/(scale_var_params.get('phibase')**2))
        a3 = a3/fb
        b3 = b3/(fb/scale_var_params.get('phibase'))
        c3 = c3/(fb/(scale_var_params.get('phibase')**2))
        Eb = scale_var_params.get('Ebase')
        qb = scale_var_params.get('qbase')
        GHVb = Eb/qb

    def obj(y,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb):
        global f_vec_global
        global u_mat_global
        global F_mat_global
        global E_mat_global
        global x_f_vec
        x_f_vec = y.copy()
        u_mat_global = np.vstack((u_mat_global,y[:len(u_init)]))
        F_mat_global = np.vstack((F_mat_global,nonlinear_equality_constraints(y)))
        E_mat_global = np.vstack((E_mat_global,y[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]]))
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        P0c, P1c, dphi0c, dphi1c = y[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]].copy() # >0
        f = price_electricity_heat(P0c,P1c,dphi0c,dphi1c,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    # gradient and Hessian of objective function
    def obj_grad(y,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        df_dy = np.zeros(len(y))
        P0c, P1c, dphi0c, dphi1c = y[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]].copy() # >0
        df_dy[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]] = price_electricity_heat_first_der(P0c,P1c,dphi0c,dphi1c,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
        # print('gradient f = {}'.format(df_dy))
        return df_dy
        hess_cost_diag = np.zeros(len(x_opf0))
    def obj_hess(y,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        hess_cost_diag = np.zeros(len(y))
        P0c, P1c, dphi0c, dphi1c = y[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]].copy() # >0
        hess_cost_diag[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]] = price_electricity_heat_second_der(P0c,P1c,dphi0c,dphi1c,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
        # print('Hessian diag f = {}'.format(hess_cost_diag))
        return np.diag(hess_cost_diag)

    # define nonlinear equality constraints (load flow equations)
    def nonlinear_equality_constraints(y,network_mes=het_net,network_e=elec_net,network_h=heat_net,scale_var=scale_var,scale_var_params=scale_var_params):
        if scale_var == 'matrix':
            y = Dy_inv.dot(y)
        # update bc of the network
        V0, q0c, q1c = y[:len(u_init)]
        network_mes, network_e, network_h = update_bc_mes_eh(network_mes, network_e, network_h, V0, q0c, q1c,scale_var=scale_var,scale_var_params=scale_var_params)
        # evaluate load flow equations
        xmes = y[len(u_init):]
        network_mes.reset_network(xmes,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        F = nlsys.F(xmes)
        if scale_var == 'matrix':
            F = DF.dot(F)
        return F

    # Jacobian of nonlinear constraints
    def jac_nleq(y,network_mes=het_net,network_e=elec_net,network_h=heat_net,scale_var=scale_var,scale_var_params=scale_var_params):
        if scale_var == 'matrix':
            y = Dy_inv.dot(y)
        # update bc of the network
        V0, q0c, q1c = y[:len(u_init)]
        network_mes, network_e, network_h = update_bc_mes_eh(network_mes, network_e, network_h, V0, q0c, q1c,scale_var=scale_var,scale_var_params=scale_var_params)
        # evaluate load flow jacobian
        xmes = y[len(u_init):]
        network_mes.reset_network(xmes,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        J_lf = nlsys.J_dense(xmes)
        xe = np.array([xmes[0]])
        network_e.reset_network(xe,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        nlsys.nlsystemse[0].V_vec_mag[0] = V0
        Je_full = nlsys.nlsystemse[0].J_dense(xe,return_full=True)
        dH_dy = np.zeros((len(x_opf0)-len(u_init),len(x_opf0)),dtype='double')
        # dH_dy[13,1] = MES.GHV #dF_dq0c
        # dH_dy[14,2] = MES.GHV #dF_dq1c
        # if scale_var == 'per_unit':
        #     dH_dy[13,1] = dH_dy[13,1]/GHVb
        #     dH_dy[14,2] = dH_dy[14,2]/GHVb
        dH_dy[13,1] = het_net.nodes[-2].der_node_law_dE(scale_var=scale_var,scale_var_params=scale_var_params)[0]
        dH_dy[14,2] = het_net.nodes[-1].der_node_law_dE(scale_var=scale_var,scale_var_params=scale_var_params)[0]
        dH_dy[0:4,0] = Je_full[Fe_ind,V0_ind].ravel()#dFe_dV0
        dH_dy[:,len(u_init):] = J_lf #dF_dxlf
        if scale_var == 'matrix':
            dH_dy = DF.dot(dH_dy.dot(Dy_inv))
        # plt.figure('A')
        # plt.spy(dH_dy)
        # plt.show()
        return dH_dy
    lb_nleq = np.zeros(len(x_opf0)-len(u_init))
    ub_nleq = np.zeros(len(x_opf0)-len(u_init))
    if derivatives:
        if optimization_method == 'trust-constr':
            nonlinear_constraint = spo.NonlinearConstraint(nonlinear_equality_constraints,lb_nleq,ub_nleq,jac=jac_nleq,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            nonlinear_constraint = {'type':'eq','fun':nonlinear_equality_constraints,'jac':jac_nleq}
    else:
        if optimization_method == 'trust-constr':
            nonlinear_constraint = spo.NonlinearConstraint(nonlinear_equality_constraints,lb_nleq,ub_nleq,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            nonlinear_constraint = {'type':'eq','fun':nonlinear_equality_constraints}

    # define linear inequality constraints, i.e. define bounds
    lb_ineq = np.array([V0_bounds[0],q0c_bounds[0],q1c_bounds[0],delta0_bounds[0],m01_bounds[0],m0_bounds[0],m1_bounds[0],p1_bounds[0],Ts0_bounds[0],Ts1_bounds[0],Tr0_bounds[0],Tr1_bounds[0],P0c_bounds[0],P1c_bounds[0],Q0c_bounds[0],Q1c_bounds[0],m0c_bounds[0],m1c_bounds[0],dphi0c_bounds[0],dphi1c_bounds[0]])
    ub_ineq = np.array([V0_bounds[1],q0c_bounds[1],q1c_bounds[1],delta0_bounds[1],m01_bounds[1],m0_bounds[1],m1_bounds[1],p1_bounds[1],Ts0_bounds[1],Ts1_bounds[1],Tr0_bounds[1],Tr1_bounds[1],P0c_bounds[1],P1c_bounds[1],Q0c_bounds[1],Q1c_bounds[1],m0c_bounds[1],m1c_bounds[1],dphi0c_bounds[1],dphi1c_bounds[1]])
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        lb_ineq = Dy.dot(lb_ineq)
        ub_ineq = Dy.dot(ub_ineq)
    if ineq_constr == 'control':
        lb_ineq[len(u_init):] = -np.inf*np.ones(len(x_opf0)-len(u_init)) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
        ub_ineq[len(u_init):] = np.inf*np.ones(len(x_opf0)-len(u_init))

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

    global f_vec_global
    global x_f_vec
    global u_mat_global
    global F_mat_global
    global E_mat_global
    f_vec_global = list()
    u_mat_global = x_opf0[:len(u_init)].copy()
    F_mat_global = nonlinear_equality_constraints(x_opf0)
    E_mat_global = x_opf0[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]].copy()
    u_mat = np.zeros((max_iter+2,len(u_init)))
    F_mat = np.zeros((max_iter+2,DF.shape[0]))
    E_mat = np.zeros((max_iter+2,4))
    x_f_vec = list()
    if optimization_method == 'trust-constr':
        f_vec = list()
        def callback_opf(xk, state):
            f_vec.append(state.fun)
            u_mat[state.nit-1,:] = xk[:len(u_init)]
            F_mat[state.nit-1,:] = nonlinear_equality_constraints(xk)
            E_mat[state.nit-1,:] = xk[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]]
            return False
    elif optimization_method == 'SLSQP':
        f_vec = [obj(x_opf0)] # this call to obj() alters all the global variables.
        u_mat[len(f_vec)-1,:] = x_opf0[:len(u_init)]
        F_mat[len(f_vec)-1,:] = nonlinear_equality_constraints(x_opf0)
        E_mat[len(f_vec)-1,:] = x_opf0[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]]
        def callback_opf(xk):
            f_vec.append(obj(xk))
            u_mat[len(f_vec)-1,:] = xk[:len(u_init)]
            F_mat[len(f_vec)-1,:] = nonlinear_equality_constraints(xk)
            E_mat[len(f_vec)-1,:] = xk[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]]
            return False

    # solve OPF
    opf_start_time = time.time()
    try:
        if derivatives:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, x_opf0, method=optimization_method, jac=obj_grad,hess=obj_hess, constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter,'gtol':tol,'xtol':tol}, bounds=bounds,tol=tol, callback=callback_opf)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                opf_start_time = time.time()
                res = spo.minimize(obj, x_opf0, method=optimization_method,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback_opf)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, x_opf0, jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
        else:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, x_opf0, method=optimization_method, constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter,'gtol':tol,'xtol':tol}, bounds=bounds,tol=tol,callback=callback_opf)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, x_opf0, method=optimization_method, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback_opf)
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
                nfev = len(f_vec_global)
                njev = 0
                nhev = 0
                if optimization_method == 'ipopt':
                    nit = 0
                else: # append value of iterate to output of the iteration in which the error occured
                    f_vec.append(f_vec_global[-1])
                    nit = len(f_vec)
                    u_mat[len(f_vec)-1,:] = x_f_vec[:len(u_init)]
                    F_mat[len(f_vec)-1,:] = nonlinear_equality_constraints(x_f_vec)
                    E_mat[len(f_vec)-1,:] = x_f_vec[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]]
            execution_time = opf_start_time - time.time()
            res = spo.OptimizeResult({'success':False,'x':np.array(x_f_vec),'fun':obj(np.array(x_f_vec)),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

    if optimization_method == 'ipopt':
        if res.nit > 0:
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
            indices = list(range(0,len(f_vec_global),round(len(f_vec_global)/len(f_vec))))
            u_mat = u_mat_global[indices,:]
            F_mat = F_mat_global[indices,:]
            E_mat = E_mat_global[indices,:]
        else:
            f_vec = [f_vec_global[-1]]
            u_mat = np.array([u_mat_global[-1,:]])
            F_mat = np.array([F_mat_global[-1,:]])
            E_mat = np.array([E_mat_global[-1,:]])

    if scale_var == 'matrix' or scale_var == 'per_unit':
        x_opf = Dy_inv.dot(res.x)
    else:
        x_opf = res.x

    # print solution
    V0, q0c, q1c = x_opf[:len(u_init)]
    het_net, elec_net, heat_net = update_bc_mes_eh(het_net, elec_net, heat_net, V0, q0c, q1c)
    xmes_opt = x_opf[len(u_init):]
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation)
    print('Solution OPF (analytical derivatives: {})'.format(derivatives))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/MES.Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('p heat = {} m'.format(p_h_vec/(MES.rho_w*MES.grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {} C'.format(Ts_hl_vec))
    print('Tr hl = {} C'.format(Tr_hl_vec))
    print('dphi hl = {}'.format(phi_hl_vec))
    print('m c = {}'.format(mc_vec))
    print('phi c = {}'.format(phic_vec))
    print('Ts c = {} C'.format(Tsc_vec))
    print('Tr c = {} C'.format(Trc_vec))
    return xmes_opt, res, f_vec, u_mat, F_mat, E_mat, execution_time

def solve_eh_lf_in_of(network_mes, network_e, network_h,u,P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind,max_iters=10,tol=MES.tol,formulation=MES.formulation,scale_var=None,scale_var_params=None):
    """Solve steady-state LF within an optmization context.

    Parameters
    ----------
    u : np arrays
        Vector with control variables. Scaled when using per unit scaling, unscaled otherwise
    """
    V0, q0c, q1c = u
    network_mes, network_e, network_h = update_bc_mes_eh(network_mes, network_e, network_h, V0, q0c, q1c,scale_var=scale_var,scale_var_params=scale_var_params)
    xmes0 = network_mes.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    network_mes.reset_network(xmes0,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    with HiddenPrints():
        xmes,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = network_mes.solve_network(tol,max_iters,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    global err_LF_vec_global
    err_LF_vec_global.append(err_vec[-1])
    P0c, P1c, dphi0c, dphi1c = xmes[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]] # >0
    return P0c, P1c, dphi0c, dphi1c, network_mes, network_e, network_h

def run_eh_optimal_load_flow_separate_LF(V0_init=1.1*MES.Vbase,V0_bounds=np.array([0.8*MES.V0_sol,1*MES.V0_sol]),q0c_init=1.3*MES.qc0_sol_CHP,q0c_bounds=np.array([1*MES.qc0_sol_CHP,1.5*MES.qc0_sol_CHP]),q1c_init=1.3*MES.qc1_sol_CHP,q1c_bounds=np.array([1*MES.qc1_sol_CHP,1.5*MES.qc1_sol_CHP]),delta0_bounds=np.array([-np.pi,np.pi]),m01_bounds=np.array([-3*MES.m01_sol,3*MES.m01_sol]),m0_bounds=np.array([0,5*MES.m0_sink]),m1_bounds=np.array([0,5*MES.m1_sink]),p1_bounds=np.array([10,5*MES.ph1_sol]),Ts0_bounds=np.array([60,140]),Ts1_bounds=np.array([60,140]),Tr0_bounds=np.array([10,60]),Tr1_bounds=np.array([10,60]),P0c_init=1.1*MES.Pc0_sol,P0c_bounds=np.array([0,3*MES.Pc0_sol]),P1c_init=1.1*MES.Pc1_sol,P1c_bounds=np.array([0,3*MES.Pc1_sol]),Q0c_bounds=np.array([-3*MES.Qc0_sol,3*MES.Qc0_sol]),Q1c_bounds=np.array([-3*MES.Qc1_sol,3*MES.Qc1_sol]),m0c_bounds=np.array([0,3*MES.mc0_sol]),m1c_bounds=np.array([0,3*MES.mc1_sol]),dphi0c_init=1.1*MES.phic0_sol,dphi0c_bounds=np.array([0,3*MES.phic0_sol]), dphi1c_init=1.1*MES.phic1_sol,dphi1c_bounds=np.array([0,3*MES.phic1_sol]),max_iter=MES.max_iter_outer,max_iters_lf=10,tol=MES.tol,scale_var=None,scale_var_params=None,a0=0,b0=.3,c0=3e-5,a1=0,b1=.2,c1=2e-5,a2=0,b2=.04,c2=4e-4,a3=0,b3=.05,c3=4.5e-4,formulation=MES.formulation,ineq_constr='control',approach='direct',stay_within_bounds=False,optimization_method='trust-constr',fb=None):
    """Run optimal load flow for the combined electricity-heat network, where LF is included implicitly.

    Parameters
    ----------------
    max_iter : int, optional
        Maximum number of iteration used for the OPF (for OPF, the number of functions evalutions might be more).
    max_iters_lf : int, optional
        Maximum number of iteration used for steady-state load flow
    approach : str, optional
        Approach used to compute the gradient and Jacobians. Either 'direct' or 'adjoint'. Default is 'direct'.
    """
    print('\nRunning OF for electricty-heat network, using the integrated MES and implicit LF (approach: {}, method: {}, ineq. constr. on: {}, hard bounds: {})'.format(approach,optimization_method,ineq_constr,stay_within_bounds))
    # create network
    het_net, elec_net, heat_net = MES.create_mes_eh_network()

    if scale_var == 'matrix' and scale_var_params == None:
        scale_var_params = MES.scale_var_params

    # update the boundary conditions of the MES to match the initial guess of opf
    het_net, elec_net, heat_net = update_bc_mes_eh(het_net, elec_net, heat_net, V0_init, q0c_init, q1c_init)

    # run steady-state load flow once, to make sure that hte intiial guess of of is at least a solution of LF
    x0 = MES.initialize_mes_eh_network(het_net) # initialize network, and set reasonable values as first initial guess for LF (if reasonable values are not set, division by 0 etc might occur during LF)

    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    Ne = len(elec_net.nodes)
    Fe_ind = nlsys.nlsystemse[0].FP + [Ne+ind for ind in nlsys.nlsystemse[0].FQ]
    V0_ind = Ne
    P0c_ind = 9 # index of P0c within x_LF
    P1c_ind = 10 # index of P1c within x_LF
    dphi0c_ind = 15 # index of dphi0c within x_LF
    dphi1c_ind = 16 # index of dphi1c within x_LF
    len_x = 17

    x0[[P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind]] = [P0c_init,P1c_init,dphi0c_init,dphi1c_init]
    het_net.reset_network(x0,formulation=formulation)

    # initial guess for OF
    u0 = np.array([V0_init,q0c_init,q1c_init])

    DF = nlsys.DF()
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        DH = DF.copy()
        Dy = np.diag(np.concatenate((np.array([1/scale_var_params.get('Vbase'),1/scale_var_params.get('qbase'),1/scale_var_params.get('qbase')]),Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((np.array([scale_var_params.get('Vbase'),scale_var_params.get('qbase'),scale_var_params.get('qbase')]),1/Dx.data[0])))
        Du = np.diag(np.array([1/scale_var_params.get('Vbase'),1/scale_var_params.get('qbase'),1/scale_var_params.get('qbase')]))
        Du_inv = np.diag(np.array([scale_var_params.get('Vbase'),scale_var_params.get('qbase'),scale_var_params.get('qbase')]))
        u0 = Du.dot(u0) # scale u
    else:
        DH = np.eye(DF.shape[0])
        Du = np.eye(len(u0))
        Du_inv= np.eye(len(u0))
        Dy=np.eye(len_x+len(u0))
        Dy_inv=np.eye(len_x+len(u0))

    if scale_var == 'per_unit':
        a0 = a0/fb
        b0 = b0/(fb/scale_var_params.get('Sbase'))
        c0 = c0/(fb/(scale_var_params.get('Sbase')**2))
        a1 = a1/fb
        b1 = b1/(fb/scale_var_params.get('Sbase'))
        c1 = c1/(fb/(scale_var_params.get('Sbase')**2))
        a2 = a2/fb
        b2 = b2/(fb/scale_var_params.get('phibase'))
        c2 = c2/(fb/(scale_var_params.get('phibase')**2))
        a3 = a3/fb
        b3 = b3/(fb/scale_var_params.get('phibase'))
        c3 = c3/(fb/(scale_var_params.get('phibase')**2))
        Eb = scale_var_params.get('Ebase')
        qb = scale_var_params.get('qbase')
        GHVb = Eb/qb

    # values used for bounds an inequality constraints
    lb_ineq = np.array([V0_bounds[0],q0c_bounds[0],q1c_bounds[0],delta0_bounds[0],m01_bounds[0],m0_bounds[0],m1_bounds[0],p1_bounds[0],Ts0_bounds[0],Ts1_bounds[0],Tr0_bounds[0],Tr1_bounds[0],P0c_bounds[0],P1c_bounds[0],Q0c_bounds[0],Q1c_bounds[0],m0c_bounds[0],m1c_bounds[0],dphi0c_bounds[0],dphi1c_bounds[0]])
    ub_ineq = np.array([V0_bounds[1],q0c_bounds[1],q1c_bounds[1],delta0_bounds[1],m01_bounds[1],m0_bounds[1],m1_bounds[1],p1_bounds[1],Ts0_bounds[1],Ts1_bounds[1],Tr0_bounds[1],Tr1_bounds[1],P0c_bounds[1],P1c_bounds[1],Q0c_bounds[1],Q1c_bounds[1],m0c_bounds[1],m1c_bounds[1],dphi0c_bounds[1],dphi1c_bounds[1]])
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        lb_ineq = Dy.dot(lb_ineq)
        ub_ineq = Dy.dot(ub_ineq)

    def obj(u,network_mes=het_net,network_e=elec_net,network_h=heat_net,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb,Du_inv=Du_inv):
        global f_vec_global
        global u_mat_global
        global E_mat_global
        global x_f_vec
        x_f_vec = u.copy()
        u_mat_global = np.vstack((u_mat_global,u))
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            u = Du_inv.dot(u)
        # update network and solve LF
        P0c, P1c, dphi0c, dphi1c, network_mes, network_e, network_h = solve_eh_lf_in_of(network_mes, network_e, network_h,u,P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind,max_iters=max_iters_lf,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        if scale_var == 'matrix':
            E_mat_global = np.vstack((E_mat_global,np.array([P0c/scale_var_params.get('Sbase'),P1c/scale_var_params.get('Sbase'),dphi0c/scale_var_params.get('phibase'),dphi1c/scale_var_params.get('phibase')])))
        else:
            E_mat_global = np.vstack((E_mat_global,np.array([P0c,P1c,dphi0c,dphi1c])))
        f = price_electricity_heat(P0c,P1c,dphi0c,dphi1c,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    # gradient of objective function
    def obj_grad(u,network_mes=het_net,network_e=elec_net,network_h=heat_net,method=approach,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb,Dy=Dy, Dy_inv=Dy_inv, Dh=DH,Du=Du,Du_inv=Du_inv):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            u = Du_inv.dot(u)
        # update network and solve LF
        P0c, P1c, dphi0c, dphi1c, network_mes, network_e, network_h = solve_eh_lf_in_of(network_mes, network_e, network_h,u,P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind,max_iters=max_iters_lf,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        # partial derivatives of objective
        x_LF = network_mes.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        y = np.concatenate((u,x_LF))
        deltaf_deltay = np.zeros(len(y))
        deltaf_deltay[[P0c_ind+len(u),P1c_ind+len(u),dphi0c_ind+len(u),dphi1c_ind+len(u)]] = price_electricity_heat_first_der(P0c,P1c,dphi0c,dphi1c,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
        deltaf_deltau = np.zeros((1,len(u)))
        deltaf_deltax = np.zeros((1,len_x))
        deltaf_deltau[0,:] = deltaf_deltay[:len(u)]
        deltaf_deltax[0,:] = deltaf_deltay[len(u):]
        # partial derivatives of equatilty constraints / load-flow equations
        network_mes.reset_network(x_LF,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        V0, q0c, q1c = u
        J_lf = nlsys.J_dense(x_LF)
        xe = np.array([x_LF[0]])
        network_e.reset_network(xe,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        nlsys.nlsystemse[0].V_vec_mag[0] = V0
        Je_full = nlsys.nlsystemse[0].J_dense(xe,return_full=True)
        deltah_deltay = np.zeros((len_x,len(y)))
        deltah_deltay[13,1] = het_net.nodes[-2].der_node_law_dE(scale_var=scale_var,scale_var_params=scale_var_params)[0]
        deltah_deltay[14,2] = het_net.nodes[-1].der_node_law_dE(scale_var=scale_var,scale_var_params=scale_var_params)[0]
        deltah_deltay[0:4,0] = Je_full[Fe_ind,V0_ind].ravel()#dFe_dV0
        deltah_deltay[:,len(u):] = J_lf #dF_dxlf
        if scale_var == 'matrix':
            deltah_deltay = DH.dot(deltah_deltay.dot(Dy_inv))
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
        return df_du

    # define (non)linear inequality constraints on state variables
    if ineq_constr == 'all':
        lb_ineq_state = lb_ineq[len(u0):]
        ub_ineq_state = ub_ineq[len(u0):]
        def g(u,scale_var=scale_var,scale_var_params=scale_var_params,network_mes=het_net,network_e=elec_net,network_h=heat_net,Dy=Dy, Dy_inv=Dy_inv, Du=Du,Du_inv=Du_inv):
            """Determine the nonlinear inequality constraints g(x(u)) >= 0"""
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                u = Du_inv.dot(u)
            # update network and solve LF
            P0c, P1c, dphi0c, dphi1c, network_mes, network_e, network_h = solve_eh_lf_in_of(network_mes, network_e, network_h,u,P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind,max_iters=max_iters_lf,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            x = network_mes.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            network_mes.reset_network(x,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            if scale_var == 'matrix': # lb_ineq_state and ub_ineq_state are scaled, so scale x as well
                x = Dy[len(u):,len(u):].dot(x)
            g = np.concatenate((x-lb_ineq_state,ub_ineq_state-x))
            return g
        def g_jac(u,scale_var=scale_var,scale_var_params=scale_var_params,network_mes=het_net,network_e=elec_net,network_h=heat_net,nlsys=nlsys,method=approach,Dy=Dy, Dy_inv=Dy_inv, fb=fb, Dh=DH,Du=Du,Du_inv=Du_inv):
            """Jacobian of inequality constraints"""
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                u = Du_inv.dot(u)
            # Jacobian of inequality constraints wrt state variables x
            deltag_deltax = np.vstack((np.eye(len_x),-np.eye(len_x)))
            deltag_deltau = np.zeros((2*len_x,len(u)))
            # update network and solve LF
            P0c, P1c, dphi0c, dphi1c, network_mes, network_e, network_h = solve_eh_lf_in_of(network_mes, network_e, network_h,u,P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind,max_iters=max_iters_lf,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            x_LF = network_mes.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            y = np.concatenate((u,x_LF))
            # partial derivatives of equatilty constraints / load-flow equations
            network_mes.reset_network(x_LF,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            V0, q0c, q1c = u
            J_lf = nlsys.J_dense(x_LF)
            xe = np.array([x_LF[0]])
            network_e.reset_network(xe,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            nlsys.nlsystemse[0].V_vec_mag[0] = V0
            Je_full = nlsys.nlsystemse[0].J_dense(xe,return_full=True)
            deltah_deltay = np.zeros((len_x,len(y)))
            deltah_deltay[13,1] = het_net.nodes[-2].der_node_law_dE(scale_var=scale_var,scale_var_params=scale_var_params)[0]
            deltah_deltay[14,2] = het_net.nodes[-1].der_node_law_dE(scale_var=scale_var,scale_var_params=scale_var_params)[0]
            deltah_deltay[0:4,0] = Je_full[Fe_ind,V0_ind].ravel()#dFe_dV0
            deltah_deltay[:,len(u):] = J_lf #dF_dxlf
            if scale_var == 'matrix':
                deltah_deltay = DH.dot(deltah_deltay.dot(Dy_inv))
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
            return dg_du
        if optimization_method == 'trust-constr':
            ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(2*len_x),np.inf*np.ones(2*len_x),jac=g_jac,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            ineq_constr_fun  = {'type':'ineq','fun':g,'jac':g_jac}
    else:
        ineq_constr_fun = None

    # define linear inequality constraints (bounds) on the control variables
    if ineq_constr != None:
        # define linear inequality constraints (on the control variables)
        lb_ineq_bounds = lb_ineq[:len(u0)]
        ub_ineq_bounds = ub_ineq[:len(u0)]
    else:
        bounds = None

    if optimization_method == 'ipopt':
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
    if optimization_method == 'ipopt' and ineq_constr != None:
        bounds = [(lb,ub) for lb, ub in zip(lb_ineq_bounds,ub_ineq_bounds)]
    elif ineq_constr != None:
        bounds = spo.Bounds(lb_ineq_bounds,ub_ineq_bounds,keep_feasible=stay_within_bounds)
    else:
        bounds = None

    # make sure initial guess satisfies bounds (NB. If adjustments are made, LF is not necessarily satisfied anymore)
    if ineq_constr != None and (optimization_method == 'SLSQP' or stay_within_bounds):
        for ind, x0 in enumerate(u0):
            if lb_ineq_bounds[ind] > x0:
                u0[ind] = lb_ineq_bounds[ind]
            elif ub_ineq_bounds[ind] < x0:
                u0[ind] = ub_ineq_bounds[ind]

    global f_vec_global
    global x_f_vec
    global u_mat_global
    global E_mat_global
    global err_LF_vec_global
    f_vec_global = list()
    u_mat_global = u0.copy()
    err_LF_vec_global = list()
    if scale_var == 'matrix':
        u0_E_mat = Du_inv.dot(u0)
    else:
        u0_E_mat = u0.copy()
    P0c, P1c, dphi0c, dphi1c, het_net, elec_net, heat_net = solve_eh_lf_in_of(het_net, elec_net, heat_net,u0_E_mat,P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind,max_iters=max_iters_lf,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    x_LF0 = het_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    het_net.reset_network(x_LF0,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    if scale_var == 'matrix':
        E_mat_global = np.array([P0c/scale_var_params.get('Sbase'),P1c/scale_var_params.get('Sbase'),dphi0c/scale_var_params.get('phibase'),dphi1c/scale_var_params.get('phibase')])
    else:
        E_mat_global = np.array([P0c, P1c, dphi0c, dphi1c])
    x_f_vec = list()
    if optimization_method == 'trust-constr':
        f_vec = list()
        def callback_opf(xk, state):
            f_vec.append(state.fun)
            return False
    elif optimization_method == 'SLSQP':
        f_vec = [obj(u0)] # this call to obj() alters all the global variables.
        def callback_opf(xk):
            f_vec.append(obj(xk))
            return False

    # solve OPF
    opf_start_time = time.time()
    try:
        if ineq_constr_fun != None:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=[ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds, callback=callback_opf)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback_opf)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, u0,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
        else:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, options={'verbose': 1,'maxiter':max_iter}, bounds=bounds, callback=callback_opf)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback_opf)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, u0,jac=obj_grad,  options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
    except:
        print('Exception made for {}, hard bounds: {}, approach: {}, scaling: {}'.format(optimization_method,stay_within_bounds,approach,scale_var))
        if len(f_vec_global) == 0:
            obj(u0)
            nit = 0
            nfev = 0
            njev = 0
            nhev = 0
        else:
            nfev = len(f_vec_global)
            njev = 0
            nhev = 0
            if optimization_method == 'ipopt':
                nit = 0
            else: # append value of iterate to output of the iteration in which the error occured
                f_vec.append(f_vec_global[-1])
                nit = len(f_vec)
        execution_time = opf_start_time - time.time()
        res = spo.OptimizeResult({'success':False,'x':np.array(x_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

    if optimization_method == 'ipopt':
        if res.nit > 0:
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            f_vec = f_vec_global

    if len(f_vec_global) > len(f_vec):
        indices = [int(round(ind)) for ind in np.linspace(0,len(f_vec_global)-1,len(f_vec))]
        u_mat = u_mat_global[indices,:]
        E_mat = E_mat_global[indices,:]
        err_LF_vec = [err_LF_vec_global[ind] for ind in indices]
    else:
        obj(u0)
        f_vec = [f_vec_global[-1]]
        err_LF_vec = [err_LF_vec_global[-1]]
        u_mat = np.array([u_mat_global[-1,:]])
        E_mat = np.array([E_mat_global[-1,:]])

    if scale_var == 'matrix':
        u_opf = Du_inv.dot(res.x)
    else:
        u_opf = res.x
    # print solution
    P0c, P1c, dphi0c, dphi1c, het_net, elec_net, heat_net = solve_eh_lf_in_of(het_net, elec_net, heat_net,u_opf,P0c_ind,P1c_ind,dphi0c_ind,dphi1c_ind,max_iters=max_iters_lf,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    xmes_opt = het_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    het_net.reset_network(xmes_opt,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('Solution OPF (approach: {})'.format(approach))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/MES.Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('p heat = {} m'.format(p_h_vec/(MES.rho_w*MES.grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {} C'.format(Ts_hl_vec))
    print('Tr hl = {} C'.format(Tr_hl_vec))
    print('dphi hl = {}'.format(phi_hl_vec))
    print('m c = {}'.format(mc_vec))
    print('phi c = {}'.format(phic_vec))
    print('Ts c = {} C'.format(Tsc_vec))
    print('Tr c = {} C'.format(Trc_vec))
    # print('objective function = {}'.format(obj(u0)))
    return xmes_opt, res, f_vec, u_mat, err_LF_vec, E_mat, execution_time

def compare_ge_opf_integrated_LF(dir_path,number_runs=10,save_figs=False,save_tables=False):
    """Compare OF for the different versions with the integrated MES for LF"""
    # solver info
    max_iters_lf = 10
    max_iter = 100
    tol = 1e-6
    scale_var = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    ineq_constr = True

    # parameter values for the two objective functions
    a=0
    b=.01*MES.GHV
    c=1e-6*(MES.GHV)**2
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.3
    c0c=3e-5
    a1c=0
    b1c=.2
    c1c=2e-5

    # initial guesses and limits for the inequality constraints (when used)
    P0c_init = 1.3*MW
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, gas_net_LF, elec_net_LF, xmes_LF, iters_LF, err_vec_LF = run_mes_ge_load_flow(max_iter=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,plot_top=False,plot_jac=False,plot_sol=False)
        q0_sol = gas_net_LF.nodes[0].half_links[0].get_q()
        P0c_sol = elec_net_LF.links[1].get_Pstart()
        P1c_sol = elec_net_LF.links[2].get_Pstart()
        # value of objective functions for LF solution
        f_LF_gas = price_gas(q0_sol,a=a,b=b,c=c)
        f_LF_comb_obj = price_gas_electricity(q0_sol,P0c_sol,P1c_sol,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)

    # run the various optimizations. Run several times, take average of run time. For the other data (which seemed to be the same every time), the last run is used.
    exec_times = list()
    exec_times_num_der = list()
    exec_times_sepLF_direct = list()
    exec_times_sepLF_adjoint = list()
    exec_times_comb_obj = list()
    exec_times_num_der_comb_obj = list()
    exec_times_sepLF_direct_comb_obj = list()
    exec_times_sepLF_adjoint_comb_obj = list()
    for run in range(number_runs):
        # LF is included as (nonlinear) equality constriant. Analytical expressions for gradients and Hessian of objective function and Jacobian of equality constraints are used. Objective function for gas input only.
        x_opf, xmes_opt, f_vec, obj_fun, nfev, nit, _, exec_time, _ = run_ge_optimal_load_flow(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation,a=a,b=b,c=c,ineq_constr=ineq_constr,derivatives=True,objective='gas')
        exec_times.append(exec_time)
        # LF is included as (nonlinear) equality constriant. Numerical approximations for gradients and Hessian of objective function and Jacobian of equality constraints are used. Objective function for gas input only.
        x_opf_num_der, xmes_opt_num_der, f_vec_num_der, obj_fun_num_der, nfev_num_der, nit_num_der, _, exec_time_num_der, _ = run_ge_optimal_load_flow(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation,a=a,b=b,c=c,ineq_constr=ineq_constr,derivatives=False,objective='gas')
        exec_times_num_der.append(exec_time_num_der)
        # LF is not included as (nonlinear) equality constriant. Analytical expressions for gradient of objective function are determined using the direct approach
        x_opf_sepLF_direct, xmes_opt_sepLF_direct, f_vec_sepLF_direct, obj_fun_sepLF_direct, nfev_sepLF_direct, nit_sepLF_direct, exec_time_sepLF_direct, _ = run_ge_optimal_load_flow_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,a=a,b=b,c=c,ineq_constr=ineq_constr,objective='gas',approach='direct')
        exec_times_sepLF_direct.append(exec_time_sepLF_direct)
        # LF is not included as (nonlinear) equality constriant. Analytical expressions for gradient of objective function are determined using the adjoint approach
        x_opf_sepLF_adjoint, xmes_opt_sepLF_adjoint, f_vec_sepLF_adjoint, obj_fun_sepLF_adjoint, nfev_sepLF_adjoint, nit_sepLF_adjoint, exec_time_sepLF_adjoint, _ = run_ge_optimal_load_flow_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,a=a,b=b,c=c,ineq_constr=ineq_constr,objective='gas',approach='adjoint')
        exec_times_sepLF_adjoint.append(exec_time_sepLF_adjoint)
        # LF is included as (nonlinear) equality constriant. Analytical expressions for gradients and Hessian of objective function and Jacobian of equality constraints are used. Objective function for gas input and coupling active powers.
        x_opf_comb_obj, xmes_opt_comb_obj, f_vec_comb_obj, obj_fun_comb_obj, nfev_comb_obj, nit_comb_obj, _, exec_time_comb_obj, _ = run_ge_optimal_load_flow(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,ineq_constr=ineq_constr,derivatives=True,objective='gas_elec')
        exec_times_comb_obj.append(exec_time_comb_obj)
        # LF is included as (nonlinear) equality constriant. Numerical approximations for gradients and Hessian of objective function and Jacobian of equality constraints are used. Objective function for gas input and coupling active powers.
        x_opf_num_der_comb_obj, xmes_opt_num_der_comb_obj, f_vec_num_der_comb_obj, obj_fun_num_der_comb_obj, nfev_num_der_comb_obj, nit_num_der_comb_obj, _, exec_time_num_der_comb_obj, _ = run_ge_optimal_load_flow(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,ineq_constr=ineq_constr,derivatives=False,objective='gas_elec')
        exec_times_num_der_comb_obj.append(exec_time_num_der_comb_obj)
        # LF is not included as (nonlinear) equality constriant. Analytical expressions for gradient of objective function are determined using the direct approach. Objective function for gas input and coupling active powers.
        x_opf_sepLF_direct_comb_obj, xmes_opt_sepLF_direct_comb_obj, f_vec_sepLF_direct_comb_obj, obj_fun_sepLF_direct_comb_obj, nfev_sepLF_direct_comb_obj, nit_sepLF_direct_comb_obj,  exec_time_sepLF_direct_comb_obj, _ = run_ge_optimal_load_flow_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,ineq_constr=ineq_constr,objective='gas_elec',approach='direct')
        exec_times_sepLF_direct_comb_obj.append(exec_time_sepLF_direct_comb_obj)
        # LF is not included as (nonlinear) equality constriant. Analytical expressions for gradient of objective function are determined using the adjoint approach. Objective function for gas input and coupling active powers.
        x_opf_sepLF_adjoint_comb_obj, xmes_opt_sepLF_adjoint_comb_obj, f_vec_sepLF_adjoint_comb_obj, obj_fun_sepLF_adjoint_comb_obj, nfev_sepLF_adjoint_comb_obj, nit_sepLF_adjoint_comb_obj, exec_time_sepLF_adjoint_comb_obj, _ = run_ge_optimal_load_flow_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,ineq_constr=ineq_constr,objective='gas_elec',approach='adjoint')
        exec_times_sepLF_adjoint_comb_obj.append(exec_time_sepLF_adjoint_comb_obj)

    exec_time = np.mean(exec_times)
    exec_time_num_der = np.mean(exec_times_num_der)
    exec_time_sepLF_direct = np.mean(exec_times_sepLF_direct)
    exec_time_sepLF_adjoint = np.mean(exec_times_sepLF_adjoint)
    exec_time_comb_obj = np.mean(exec_times_comb_obj)
    exec_time_num_der_comb_obj = np.mean(exec_times_num_der_comb_obj)
    exec_time_sepLF_direct_comb_obj = np.mean(exec_times_sepLF_direct_comb_obj)
    exec_time_sepLF_adjoint_comb_obj = np.mean(exec_times_sepLF_adjoint_comb_obj)

    # create (and save) table with difference between OF and LF solution. For intergrated MES, x_LF = [q01 p1 delta0 q0c q1c P1c Q0c Q1c]
    def rel_diff(x1,x2):
        return np.abs(x1-x2)/np.abs(x1)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','MES2N')
        variable_names = [r'$q_{01}$',r'$p_1$',r'$\delta_0$',r'$q_{0c}',r'$q_{1c}$',r'$P_{1c}$',r'$Q_{0c}$',r'$Q_{1c}$']
        with open(os.path.join(path_to_tables,'solution_LF_vs_OF_integrated_MES_gas.txt'), "w") as table:
            for ind_var,var in enumerate(variable_names):
                table.write(r'{} & {:.5e} & {:.5e} &  {:.5e} &  {:.5e}  & {:.5e}   \\ '.format(var,xmes_LF[ind_var],rel_diff(xmes_LF,xmes_opt_num_der)[ind_var],rel_diff(xmes_LF,xmes_opt)[ind_var],rel_diff(xmes_LF,xmes_opt_sepLF_direct)[ind_var],rel_diff(xmes_LF,xmes_opt_sepLF_adjoint)[ind_var]))
            table.write(r'\hline ')
            table.write(r'max. & & {:.5e} &  {:.5e} &  {:.5e}  & {:.5e}  \\ '.format(np.max(rel_diff(xmes_LF,xmes_opt_num_der)),np.max(rel_diff(xmes_LF,xmes_opt)),np.max(rel_diff(xmes_LF,xmes_opt_sepLF_direct)),np.max(rel_diff(xmes_LF,xmes_opt_sepLF_adjoint))))
        with open(os.path.join(path_to_tables,'solution_LF_vs_OF_integrated_MES_comb_obj.txt'), "w") as table:
            for ind_var,var in enumerate(variable_names):
                table.write(r'{} & {:.5e} & {:.5e} &  {:.5e} &  {:.5e}  & {:.5e}  \\ '.format(var,xmes_LF[ind_var],rel_diff(xmes_LF,xmes_opt_num_der_comb_obj)[ind_var],rel_diff(xmes_LF,xmes_opt_comb_obj)[ind_var],rel_diff(xmes_LF,xmes_opt_sepLF_direct_comb_obj)[ind_var],rel_diff(xmes_LF,xmes_opt_sepLF_adjoint_comb_obj)[ind_var]))
            table.write(r'\hline ')
            table.write(r'max. & & {:.5e} &  {:.5e} &  {:.5e}  & {:.5e}  \\ '.format(np.max(rel_diff(xmes_LF,xmes_opt_num_der_comb_obj)),np.max(rel_diff(xmes_LF,xmes_opt_comb_obj)),np.max(rel_diff(xmes_LF,xmes_opt_sepLF_direct_comb_obj)),np.max(rel_diff(xmes_LF,xmes_opt_sepLF_adjoint_comb_obj))))
    # print results of optimizer, and create (and save) table
    print('\nopf num. der.   opf     opf sep. LF direct    opf sep. LF adjoint    num. der. comb. obj.   opf comb. obj.    opf sep. LF direct comb. obj.   opf sep. LF adjoint comb. obj.')
    print('obj. func:  {:.5e}  , {:.5e} , {:.5e} , {:.5e},   {:.5e}  , {:.5e} , {:.5e} , {:.5e}'.format(obj_fun_num_der,obj_fun,obj_fun_sepLF_direct,obj_fun_sepLF_adjoint,obj_fun_num_der_comb_obj,obj_fun_comb_obj,obj_fun_sepLF_direct_comb_obj,obj_fun_sepLF_adjoint_comb_obj))
    print('numb. fev.:  {:d}  , {:d} , {:d} , {:d},  {:d}  , {:d} , {:d} , {:d}'.format(nfev_num_der,nfev,nfev_sepLF_direct,nfev_sepLF_adjoint,nfev_num_der_comb_obj,nfev_comb_obj,nfev_sepLF_direct_comb_obj,nfev_sepLF_adjoint_comb_obj))
    print('iters:  {:d}  , {:d}  , {:d} , {:d},  {:d}  , {:d} , {:d} , {:d}'.format(nit_num_der,nit,nit_sepLF_direct,nit_sepLF_adjoint,nit_num_der_comb_obj,nit_comb_obj,nit_sepLF_direct_comb_obj,nit_sepLF_adjoint_comb_obj))
    print('time:  {:.5f}  , {:.5f} , {:5f} , {:.5f},  {:.5f}  , {:.5f} , {:5f} , {:.5f}\n'.format(exec_time_num_der,exec_time,exec_time_sepLF_direct,exec_time_sepLF_adjoint,exec_time_num_der_comb_obj,exec_time_comb_obj,exec_time_sepLF_direct_comb_obj,exec_time_sepLF_adjoint_comb_obj))
    if save_tables:
        with open(os.path.join(path_to_tables,'optimizer_info_integrated_MES_gas.txt'), "w") as table:
            table.write(r'$f$ & {:.5e} &  {:.5e} &  {:.5e}  & {:.5e} & {:.5e} \\ '.format(f_LF_gas,obj_fun_num_der,obj_fun,obj_fun_sepLF_direct,obj_fun_sepLF_adjoint))
            table.write(r'func. eval. & & {:d} & {:d}  & {:d}  & {:d} \\ '.format(nfev_num_der,nfev,nfev_sepLF_direct,nfev_sepLF_adjoint))
            table.write(r'iterations & & {:d} & {:d}  & {:d}  & {:d} \\ '.format(nit_num_der,nit,nit_sepLF_direct,nit_sepLF_adjoint))
            table.write(r'time [s] & & {:.5f} & {:.5f}  & {:.5f}  & {:.5f} \\ '.format(exec_time_num_der,exec_time,exec_time_sepLF_direct,exec_time_sepLF_adjoint))
        with open(os.path.join(path_to_tables,'optimizer_info_integrated_MES_comb_obj.txt'), "w") as table:
            table.write(r'$f$ & {:.5e} &  {:.5e} &  {:.5e}  & {:.5e} & {:.5e}  \\ '.format(f_LF_comb_obj,obj_fun_num_der_comb_obj,obj_fun_comb_obj,obj_fun_sepLF_direct_comb_obj,obj_fun_sepLF_adjoint_comb_obj))
            table.write(r'func. eval. & & {:d} & {:d}  & {:d}  & {:d} \\ '.format(nfev_num_der_comb_obj,nfev_comb_obj,nfev_sepLF_direct_comb_obj,nfev_sepLF_adjoint_comb_obj))
            table.write(r'iterations & & {:d} & {:d}  & {:d}  & {:d} \\ '.format(nit_num_der_comb_obj,nit_comb_obj,nit_sepLF_direct_comb_obj,nit_sepLF_adjoint_comb_obj))
            table.write(r'time [s] & & {:.5f} & {:.5f}  & {:.5f}  & {:.5f} \\ '.format(exec_time_num_der_comb_obj,exec_time_comb_obj,exec_time_sepLF_direct_comb_obj,exec_time_sepLF_adjoint_comb_obj))
    # plots
    colors = {'LF':'k', 'OPF':'tab:blue', 'OPF direct':'tab:orange', 'OPF adjoint':'tab:green', 'OPF num. der.':'tab:red'}
    fig_f_gas = plt.figure('objective_function_gas_OF_integrated_MES')
    ax_f_gas = fig_f_gas.gca()
    ax_f_gas.set_xlabel('Iteration')
    ax_f_gas.set_ylabel('f')

    fig_f_comb_obj = plt.figure('objective_function_comb_obj_OF_integrated_MES')
    ax_f_comb_obj = fig_f_comb_obj.gca()
    ax_f_comb_obj.set_xlabel('Iteration')
    ax_f_comb_obj.set_ylabel('f')

    # plot results with LF as equality constraints
    ax_f_gas.plot(f_vec,color=colors.get('OPF'),label='OF')
    ax_f_comb_obj.plot(f_vec_comb_obj,color=colors.get('OPF'),label='OF')
    # plot results with LF as equality constraints, using numerical derivatives
    ax_f_gas.plot(f_vec_num_der,color=colors.get('OPF num. der.'),label='OF num. der.')
    ax_f_comb_obj.plot(f_vec_num_der_comb_obj,color=colors.get('OPF num. der.'),label='OF num. der.')
    # plot results with LF implicit, direct approach
    ax_f_gas.plot(f_vec_sepLF_direct ,color=colors.get('OPF direct'),label='sep. LF direct')
    ax_f_comb_obj.plot(f_vec_sepLF_direct_comb_obj ,color=colors.get('OPF direct'),label='sep. LF direct')
    # plot results with LF implicit, adjoint approach
    ax_f_gas.plot(f_vec_sepLF_adjoint ,color=colors.get('OPF adjoint'),label='sep. LF adjoint')
    ax_f_comb_obj.plot(f_vec_sepLF_adjoint_comb_obj ,color=colors.get('OPF adjoint'),label='sep. LF adjoint')

    # layout
    nit_max_gas = np.max([nit_num_der,nit,nit_sepLF_direct,nit_sepLF_adjoint])
    ax_f_gas.plot([0,nit_max_gas],[f_LF_gas,f_LF_gas],ls=':',color=colors.get('LF'),label='LF')
    ax_f_gas.set_xlim(left=0,right=nit_max_gas)
    ax_f_gas.legend()
    nit_max_comb_obj = np.max([nit_num_der_comb_obj,nit_comb_obj,nit_sepLF_direct_comb_obj,nit_sepLF_adjoint_comb_obj])
    ax_f_comb_obj.plot([0,nit_max_comb_obj],[f_LF_comb_obj,f_LF_comb_obj],ls=':',color=colors.get('LF'),label='LF')
    ax_f_comb_obj.set_xlim(left=0,right=nit_max_comb_obj)
    ax_f_comb_obj.legend()

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','MES2N')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def compare_ge_opf_integrated_LF_methods(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF of gas-electricity network for different optimization methods, objective function, and bounds. Without scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')
    # solver info
    max_iters_lf = 10
    max_iter = 100
    tol = 1e-6
    scale_var = None
    fb = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

    # parameter values for the two objective functions
    a=0
    b=.01*MES.GHV
    c=1e-6*(MES.GHV)**2
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.3
    c0c=3e-5
    a1c=0
    b1c=.2
    c1c=2e-5

    # initial guesses and limits for the inequality constraints (when used)
    P0c_init = 1.3*MW
    V0_init = 1.05*10/np.sqrt(3)*kV

    # bounds
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    q0_lb=1.5*MES.q0_source
    q0_ub=.5*MES.q0_source
    q01_lb=-5
    q01_ub=5
    p1_lb=1*mbar
    p1_ub=1.1*MES.pg0
    delta0_lb=-np.pi
    delta0_ub=np.pi
    q0c_lb=0
    q0c_ub=1.5*MES.qc0_sol
    q1c_lb=0
    q1c_ub=1.5*MES.qc1_sol
    P1c_lb=0
    P1c_ub=1.5*MES.Pc1_sol
    Q0c_lb=-2*MES.Qc0_sol
    Q0c_ub=2*MES.Qc0_sol
    Q1c_lb=-5*MES.Qc1_sol
    Q1c_ub=5*MES.Qc1_sol

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, gas_net_LF, elec_net_LF, xmes_LF, iters_LF, err_vec_LF = run_mes_ge_load_flow(max_iter=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,plot_top=False,plot_jac=False,plot_sol=False)
        q0_sol = gas_net_LF.nodes[0].half_links[0].get_q()
        V0_sol = elec_net_LF.nodes[0].get_V()
        P0c_sol = elec_net_LF.links[1].get_Pstart()
        P1c_sol = elec_net_LF.links[2].get_Pstart()
        # value of objective functions for LF solution
        f_LF_gas = price_gas(q0_sol,a=a,b=b,c=c)
        f_LF_comb_obj = price_gas_electricity(q0_sol,P0c_sol,P1c_sol,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)
        x_opf_LF = np.concatenate((np.array([V0_sol,P0c_sol,q0_sol]),xmes_LF))

    result = dict()
    xmes_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']
    ineq_constrs = ['control','all']
    obj_funs = ['gas','gas_elec']
    # Optimal Flow
    for obj_fun in obj_funs:
        for ineq_constr in ineq_constrs:
            # plots
            fig_f = plt.figure('obj_OPF_ineq_constr_{}_obj_{}'.format(ineq_constr,obj_fun))
            ax_f = fig_f.gca()
            ax_f.set_xlabel('Iteration')
            ax_f.set_ylabel('f')

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
                        x_opf, xmes_opt, f_vec, fun, nfev, nit, njev, execution_time, success = run_ge_optimal_load_flow(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_lb=q0_lb,q0_ub=q0_ub,q01_lb=q01_lb,q01_ub=q01_ub,p1_lb=p1_lb,p1_ub=p1_ub,delta0_lb=delta0_lb,delta0_ub=delta0_ub,q0c_lb=q0c_lb,q0c_ub=q0c_ub,q1c_lb=q1c_lb,q1c_ub=q1c_ub,P1c_lb=P1c_lb,P1c_ub=P1c_ub,Q0c_lb=Q0c_lb,Q0c_ub=Q0c_ub,Q1c_lb=Q1c_lb,Q1c_ub=Q1c_ub,max_iter=max_iter,tol=tol,scale_var=scale_var,scale_var_params=None,formulation=formulation,a=a,b=b,c=c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,fb=fb,ineq_constr=ineq_constr,derivatives=derivatives,objective=obj_fun,optimization_method=method,stay_within_bounds=stay_within_bounds)
                        result[method+'_'+bound+'_'+der+'_{}_{}'.format(ineq_constr,obj_fun)] = spo.OptimizeResult({'success':success,'x':x_opf,'nit':nit,'nfev':nfev,'njev':njev,'execution_time':execution_time})
                        xmes_res[method+'_'+bound+'_'+der+'_{}_{}'.format(ineq_constr,obj_fun)] = xmes_opt
                        max_fev = max(max_fev,len(f_vec))
                        # plot results
                        ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
            if obj_fun == 'gas':
                f_LF_sol = f_LF_gas
            else:
                f_LF_sol = f_LF_comb_obj
            ax_f.plot([0,max_fev],[f_LF_sol,f_LF_sol],':r')
            ax_f.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','MES2N')
        variable_names = [r'$|V_0|$',r'$P_{0c}$',r'$q_0$',r'$q_{01}$',r'$p_1$',r'$\delta_0$',r'$q_{0c}',r'$q_{1c}$',r'$P_{1c}$',r'$Q_{0c}$',r'$Q_{1c}$']
        for obj_fun in obj_funs:
            if obj_fun == 'gas':
                obj_fun_label = 'gas'
            else:
                obj_fun_label = 'combined'
            for bound in bounds:
                for der in ders:
                    with open(os.path.join(path_to_tables,'solution_LF_vs_OF_integrated_MES_ge_methods_'+obj_fun_label+'_'+bound+'_'+der+'.txt'), "w") as table:
                        res_trust_control = result.get('trust-constr_'+bound+'_'+der+'_control'+'_'+obj_fun)
                        res_slsqp_control = result.get('SLSQP_'+bound+'_'+der+'_control'+'_'+obj_fun)
                        res_ipopt_control = result.get('ipopt_'+bound+'_'+der+'_control'+'_'+obj_fun)
                        res_trust_all = result.get('trust-constr_'+bound+'_'+der+'_all'+'_'+obj_fun)
                        res_slsqp_all = result.get('SLSQP_'+bound+'_'+der+'_all'+'_'+obj_fun)
                        res_ipopt_all = result.get('ipopt_'+bound+'_'+der+'_all'+'_'+obj_fun)
                        for ind_var, var in enumerate(variable_names):
                            table.write(r'{} & {:.3e} & {:.3e}  & {:.3e}  & {:.3e} & {:.3e}  & {:.3e}  & {:.3e} \\ '.format(var,x_opf_LF[ind_var],error(res_trust_control.x[ind_var],x_opf_LF[ind_var]),error(res_slsqp_control.x[ind_var],x_opf_LF[ind_var]),error(res_ipopt_control.x[ind_var],x_opf_LF[ind_var]),error(res_trust_all.x[ind_var],x_opf_LF[ind_var]),error(res_slsqp_all.x[ind_var],x_opf_LF[ind_var]),error(res_ipopt_all.x[ind_var],x_opf_LF[ind_var])))
                        table.write(r'\hline ')
                        table.write(r'max. &  & {:.3e}  & {:.3e}  & {:.3e} & {:.3e}  & {:.3e}  & {:.3e} \\ '.format(error(res_trust_control.x,x_opf_LF),error(res_slsqp_control.x,x_opf_LF),error(res_ipopt_control.x,x_opf_LF),error(res_trust_all.x,x_opf_LF),error(res_slsqp_all.x,x_opf_LF),error(res_ipopt_all.x,x_opf_LF)))
        with open(os.path.join(path_to_tables,'optimizer_info_integrated_MES_ge_methods.txt'), "w") as table:
            for obj_fun in obj_funs:
                if obj_fun == 'gas':
                    obj_fun_label = 'gas'
                else:
                    obj_fun_label = 'combined'
                for ineq_constr in ineq_constrs:
                    for bound in bounds:
                        for der in ders:
                            res_trust = result.get('trust-constr_'+bound+'_'+der+'_{}_{}'.format(ineq_constr,obj_fun))
                            res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_{}_{}'.format(ineq_constr,obj_fun))
                            res_ipopt = result.get('ipopt_'+bound+'_'+der+'_{}_{}'.format(ineq_constr,obj_fun))
                            table.write(r'{} & {} & {} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(obj_fun_label,ineq_constr,bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(res_trust.x,x_opf_LF),error(res_slsqp.x,x_opf_LF),error(res_ipopt.x,x_opf_LF)))
                            print('\nObj: {}, constraints: {}, bounds: {}, der: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nError for t-c:{}, SLSQP: {}, IPOPT: {}'.format(obj_fun,ineq_constr,bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,error(res_trust.x,x_opf_LF),error(res_slsqp.x,x_opf_LF),error(res_ipopt.x,x_opf_LF)))
                    table.write(r'\cline{2-16} ')
                table.write(r'\hline ')

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','MES2N')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def compare_ge_opf_integrated_LF_scaling(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF of gas-electricity network for different optimization methods, objective function, and bounds. With scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')
    # solver info
    max_iters_lf = 10
    max_iter = 50
    tol = 1e-6
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

    # parameter values for the two objective functions
    a=0
    b=.01*MES.GHV
    c=1e-6*(MES.GHV)**2
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.3
    c0c=3e-5
    a1c=0
    b1c=.2
    c1c=2e-5

    # initial guesses and limits for the inequality constraints (when used)
    P0c_init = 1.3*MW
    V0_init = 1.05*10/np.sqrt(3)*kV

    # bounds
    ineq_constr = 'all'
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    q0_lb=1.5*MES.q0_source
    q0_ub=.5*MES.q0_source
    q01_lb=-5
    q01_ub=5
    p1_lb=1*mbar
    p1_ub=1.1*MES.pg0
    delta0_lb=-np.pi
    delta0_ub=np.pi
    q0c_lb=0
    q0c_ub=1.5*MES.qc0_sol
    q1c_lb=0
    q1c_ub=1.5*MES.qc1_sol
    P1c_lb=0
    P1c_ub=1.5*MES.Pc1_sol
    Q0c_lb=-2*MES.Qc0_sol
    Q0c_ub=2*MES.Qc0_sol
    Q1c_lb=-5*MES.Qc1_sol
    Q1c_ub=5*MES.Qc1_sol

    # steady-state LF solution, unscaled
    with HiddenPrints():
        het_net_LF, gas_net_LF, elec_net_LF, xmes_LF, iters_LF, err_vec_LF = run_mes_ge_load_flow(max_iter=max_iters_lf,tol=tol,scale_var=None,formulation=formulation,plot_top=False,plot_jac=False,plot_sol=False)
        q0_sol = gas_net_LF.nodes[0].half_links[0].get_q()
        V0_sol = elec_net_LF.nodes[0].get_V()
        P0c_sol = elec_net_LF.links[1].get_Pstart()
        P1c_sol = elec_net_LF.links[2].get_Pstart()
        x_opf_LF = np.concatenate((np.array([V0_sol,P0c_sol,q0_sol]),xmes_LF))

    # base values
    Sbase = 1*MW #[W]
    Vbase = 10/np.sqrt(3)*kV #[V]
    deltabase = 1.
    pbase = 10*mbar #[Pa]
    qbase = .01 #[kg/s]
    Egbase = 1*MW #[W]
    scale_var_params={'qbase':qbase,'pbase':pbase,'pgbase':pbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'Ebase':Egbase}
    obj_base = {'gas':1e7,'gas_elec':1e8}

    result = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']
    obj_funs = ['gas','gas_elec']
    scaling = ['matrix','per_unit']
    # Optimal Flow
    for obj_fun in obj_funs:
        if obj_fun == 'gas':
            obj_fun_label = 'gas'
        else:
            obj_fun_label = 'combined'
        fb = obj_base.get(obj_fun)
        for scale_var in scaling:
            # plots
            fig_f = plt.figure('obj_OPF_integrated_MES_ge_'+obj_fun_label+'_'+scale_var)
            ax_f = fig_f.gca()
            ax_f.set_xlabel('Iteration')
            ax_f.set_ylabel('f')
            # value of objective functions for LF solution
            if scale_var == 'per_unit':
                if obj_fun == 'gas':
                    f_LF_sol = price_gas(q0_sol/scale_var_params.get('qbase'),a=a/fb,b=b/(fb/scale_var_params.get('qbase')),c=c/(fb/scale_var_params.get('qbase')**2))
                else:
                    f_LF_sol = price_gas_electricity(q0_sol/scale_var_params.get('qbase'),P0c_sol/scale_var_params.get('Sbase'),P1c_sol/scale_var_params.get('Sbase'),a0=a0/fb,b0=b0/(fb/scale_var_params.get('qbase')),c0=c0/(fb/scale_var_params.get('qbase')**2),a0c=a0c/fb,b0c=b0c/(fb/scale_var_params.get('Sbase')),c0c=c0c/(fb/scale_var_params.get('Sbase')**2),a1c=a1c/fb,b1c=b1c/(fb/scale_var_params.get('Sbase')),c1c=c1c/(fb/scale_var_params.get('Sbase')**2))
            else:
                if obj_fun == 'gas':
                    f_LF_sol = price_gas(q0_sol,a=a,b=b,c=c)/fb
                else:
                    f_LF_sol = price_gas_electricity(q0_sol,P0c_sol,P1c_sol,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)/fb

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
                        x_opf, xmes_opt, f_vec, fun, nfev, nit, njev, execution_time, success = run_ge_optimal_load_flow(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_lb=q0_lb,q0_ub=q0_ub,q01_lb=q01_lb,q01_ub=q01_ub,p1_lb=p1_lb,p1_ub=p1_ub,delta0_lb=delta0_lb,delta0_ub=delta0_ub,q0c_lb=q0c_lb,q0c_ub=q0c_ub,q1c_lb=q1c_lb,q1c_ub=q1c_ub,P1c_lb=P1c_lb,P1c_ub=P1c_ub,Q0c_lb=Q0c_lb,Q0c_ub=Q0c_ub,Q1c_lb=Q1c_lb,Q1c_ub=Q1c_ub,max_iter=max_iter,tol=tol,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation,a=a,b=b,c=c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,fb=fb,ineq_constr=ineq_constr,derivatives=derivatives,objective=obj_fun,optimization_method=method,stay_within_bounds=stay_within_bounds)
                        result[method+'_'+bound+'_'+der+'_'+obj_fun+'_'+scale_var] = spo.OptimizeResult({'success':success,'x':x_opf,'nit':nit,'nfev':nfev,'njev':njev,'execution_time':execution_time}) # x_opf is unscaled, so in this case, res.x is also unscaled
                        max_fev = max(max_fev,len(f_vec))
                        # plot results
                        ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
            ax_f.plot([0,max_fev],[f_LF_sol,f_LF_sol],':r')
            ax_f.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','MES2N')
        with open(os.path.join(path_to_tables,'optimizer_info_integrated_MES_ge_scaling.txt'), "w") as table:
            for obj_fun in obj_funs:
                if obj_fun == 'gas':
                    obj_fun_label = 'gas'
                else:
                    obj_fun_label = 'combined'
                for bound in bounds:
                    for der in ders:
                        res_trust_mat = result.get('trust-constr_'+bound+'_'+der+'_'+obj_fun+'_matrix')
                        res_slsqp_mat = result.get('SLSQP_'+bound+'_'+der+'_'+obj_fun+'_matrix')
                        res_ipopt_mat = result.get('ipopt_'+bound+'_'+der+'_'+obj_fun+'_matrix')
                        res_trust_pu = result.get('trust-constr_'+bound+'_'+der+'_'+obj_fun+'_per_unit')
                        res_slsqp_pu = result.get('SLSQP_'+bound+'_'+der+'_'+obj_fun+'_per_unit')
                        res_ipopt_pu = result.get('ipopt_'+bound+'_'+der+'_'+obj_fun+'_per_unit')
                        table.write(r'{} & {} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} \\ '.format(obj_fun_label,bound,der,res_trust_mat.success,res_trust_pu.success,res_slsqp_mat.success,res_slsqp_pu.success,res_ipopt_mat.success,res_ipopt_pu.success,res_trust_mat.nit,res_trust_pu.nit,res_slsqp_pu.nit,res_slsqp_mat.nit,res_ipopt_mat.nit,res_ipopt_pu.nit,error(res_trust_mat.x,x_opf_LF),error(res_trust_pu.x,x_opf_LF),error(res_slsqp_mat.x,x_opf_LF),error(res_slsqp_pu.x,x_opf_LF),error(res_ipopt_mat.x,x_opf_LF),error(res_ipopt_pu.x,x_opf_LF)))
                table.write(r'\hline ')

    for obj_fun in obj_funs:
        for bound in bounds:
            for der in ders:
                for scale_var in scaling:
                    res_trust = result.get('trust-constr_'+bound+'_'+der+'_'+obj_fun+'_'+scale_var)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_'+obj_fun+'_'+scale_var)
                    res_ipopt = result.get('ipopt_'+bound+'_'+der+'_'+obj_fun+'_'+scale_var)
                    print('\nObj: {}, bounds: {}, der: {}, scaling: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nError for t-c:{}, SLSQP: {}, IPOPT: {}'.format(obj_fun,bound,der,scale_var,res_trust.success,res_slsqp.success,res_ipopt.success,error(res_trust.x,x_opf_LF),error(res_slsqp.x,x_opf_LF),error(res_ipopt.x,x_opf_LF)))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','MES2N')
        for fig_num in plt.get_figlabels():
            if not '3d' in fig_num:
                plt.figure(fig_num)
                file_name = fig_num+'.pgf'
                plt.savefig(os.path.join(path_to_fig, file_name))

def compare_ge_opf_integrated_LF_methods_sep_LF(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF of gas-electricity network with integrated LF substituted for different optimization methods, objective function, and bounds. Without scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')
    # solver info
    max_iters_lf = 10
    max_iter = 50#100
    tol = 1e-6
    scale_var = None
    fb = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

    # parameter values for the two objective functions
    a=0
    b=.01*MES.GHV
    c=1e-6*(MES.GHV)**2
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.3
    c0c=3e-5
    a1c=0
    b1c=.2
    c1c=2e-5

    # initial guesses and limits for the inequality constraints (when used)
    P0c_init = .8*MES.Pc0_sol
    V0_init = .9*MES.V0_sol

    # bounds
    ineq_constr='all'
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    q0_lb=1.5*MES.q0_source
    q0_ub=.5*MES.q0_source
    q01_lb=-5
    q01_ub=5
    p1_lb=1*mbar
    p1_ub=1.1*MES.pg0
    delta0_lb=-np.pi
    delta0_ub=np.pi
    q0c_lb=0
    q0c_ub=1.5*MES.qc0_sol
    q1c_lb=0
    q1c_ub=1.5*MES.qc1_sol
    P1c_lb=0
    P1c_ub=1.5*MES.Pc1_sol
    Q0c_lb=-2*MES.Qc0_sol
    Q0c_ub=2*MES.Qc0_sol
    Q1c_lb=-5*MES.Qc1_sol
    Q1c_ub=5*MES.Qc1_sol

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, gas_net_LF, elec_net_LF, xmes_LF, iters_LF, err_vec_LF = run_mes_ge_load_flow(max_iter=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,plot_top=False,plot_jac=False,plot_sol=False)
        q0_sol = gas_net_LF.nodes[0].half_links[0].get_q()
        V0_sol = elec_net_LF.nodes[0].get_V()
        P0c_sol = elec_net_LF.links[1].get_Pstart()
        P1c_sol = elec_net_LF.links[2].get_Pstart()
        # value of objective functions for LF solution
        f_LF_gas = price_gas(q0_sol,a=a,b=b,c=c)
        f_LF_comb_obj = price_gas_electricity(q0_sol,P0c_sol,P1c_sol,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)

    result = dict()
    xmes_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']
    obj_funs = ['gas','gas_elec']

    # Optimal Flow
    for obj_fun in obj_funs:
        # plots
        fig_f = plt.figure('obj_OPF_integrated_MES_ge_sep_LF_obj_{}'.format(obj_fun))
        ax_f = fig_f.gca()
        ax_f.set_xlabel('Iteration')
        ax_f.set_ylabel('f')

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
                        x_opf, xmes_opt, f_vec, fun, nfev, nit, execution_time, success = run_ge_optimal_load_flow_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_lb=q0_lb,q0_ub=q0_ub,q01_lb=q01_lb,q01_ub=q01_ub,p1_lb=p1_lb,p1_ub=p1_ub,delta0_lb=delta0_lb,delta0_ub=delta0_ub,q0c_lb=q0c_lb,q0c_ub=q0c_ub,q1c_lb=q1c_lb,q1c_ub=q1c_ub,P1c_lb=P1c_lb,P1c_ub=P1c_ub,Q0c_lb=Q0c_lb,Q0c_ub=Q0c_ub,Q1c_lb=Q1c_lb,Q1c_ub=Q1c_ub,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,scale_var_params=None,formulation=formulation,a=a,b=b,c=c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,fb=fb,ineq_constr=ineq_constr,objective=obj_fun,optimization_method=method,stay_within_bounds=stay_within_bounds,approach=approach)
                    else:
                        approach_legend = 'an'
                        x_opf, xmes_opt, f_vec, fun, nfev, nit, njev, execution_time, success = run_ge_optimal_load_flow(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_lb=q0_lb,q0_ub=q0_ub,q01_lb=q01_lb,q01_ub=q01_ub,p1_lb=p1_lb,p1_ub=p1_ub,delta0_lb=delta0_lb,delta0_ub=delta0_ub,q0c_lb=q0c_lb,q0c_ub=q0c_ub,q1c_lb=q1c_lb,q1c_ub=q1c_ub,P1c_lb=P1c_lb,P1c_ub=P1c_ub,Q0c_lb=Q0c_lb,Q0c_ub=Q0c_ub,Q1c_lb=Q1c_lb,Q1c_ub=Q1c_ub,max_iter=max_iter,tol=tol,scale_var=scale_var,scale_var_params=None,formulation=formulation,a=a,b=b,c=c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,fb=fb,ineq_constr=ineq_constr,derivatives=True,objective=obj_fun,optimization_method=method,stay_within_bounds=stay_within_bounds)
                    result[method+'_'+bound+'_'+approach+'_'+obj_fun] = spo.OptimizeResult({'success':success,'x':x_opf,'nit':nit,'nfev':nfev,'execution_time':execution_time})
                    xmes_res[method+'_'+bound+'_'+approach+'_'+obj_fun] = xmes_opt
                    max_fev = max(max_fev,len(f_vec))
                    # plot results
                    ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
        if obj_fun == 'gas':
            f_LF_sol = f_LF_gas
        else:
            f_LF_sol = f_LF_comb_obj
        ax_f.plot([0,max_fev],[f_LF_sol,f_LF_sol],':r')
        ax_f.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','MES2N')
        with open(os.path.join(path_to_tables,'optimizer_info_integrated_MES_ge_methods_sep_LF.txt'), "w") as table:
            for obj_fun in obj_funs:
                if obj_fun == 'gas':
                    obj_fun_label = 'gas'
                else:
                    obj_fun_label = 'combined'
                for bound in bounds:
                    for approach in approaches:
                        if approach == 'eq_constr':
                            approach_label = 'eq. constr.'
                        else:
                            approach_label = approach
                        res_trust = result.get('trust-constr_'+bound+'_'+approach+'_'+obj_fun)
                        res_slsqp = result.get('SLSQP_'+bound+'_'+approach+'_'+obj_fun)
                        res_ipopt = result.get('ipopt_'+bound+'_'+approach+'_'+obj_fun)
                        if approach == 'eq_constr':
                            y_trust = res_trust.x
                            y_slsqp = res_slsqp.x
                            y_ipopt = res_ipopt.x
                            x_opf_LF = np.concatenate((np.array([V0_sol,P0c_sol,q0_sol]),xmes_LF))
                        else:
                            y_trust = np.concatenate((res_trust.x,xmes_res.get('trust-constr_'+bound+'_'+approach+'_'+obj_fun)))
                            y_slsqp = np.concatenate((res_slsqp.x,xmes_res.get('SLSQP_'+bound+'_'+approach+'_'+obj_fun)))
                            y_ipopt = np.concatenate((res_ipopt.x,xmes_res.get('ipopt_'+bound+'_'+approach+'_'+obj_fun)))
                            x_opf_LF = np.concatenate((np.array([V0_sol,P0c_sol]),xmes_LF))
                        table.write(r'{} & {} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(obj_fun_label,bound,approach_label,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(y_trust,x_opf_LF),error(y_slsqp,x_opf_LF),error(y_ipopt,x_opf_LF)))
                    table.write(r'\cline{2-15} ')
                table.write(r'\hline ')

    for obj_fun in obj_funs:
        for bound in bounds:
            for approach in approaches:
                res_trust = result.get('trust-constr_'+bound+'_'+approach+'_'+obj_fun)
                res_slsqp = result.get('SLSQP_'+bound+'_'+approach+'_'+obj_fun)
                res_ipopt = result.get('ipopt_'+bound+'_'+approach+'_'+obj_fun)
                if approach == 'eq_constr':
                    y_trust = res_trust.x
                    y_slsqp = res_slsqp.x
                    y_ipopt = res_ipopt.x
                    x_opf_LF = np.concatenate((np.array([V0_sol,P0c_sol,q0_sol]),xmes_LF))
                else:
                    y_trust = np.concatenate((res_trust.x,xmes_res.get('trust-constr_'+bound+'_'+approach+'_'+obj_fun)))
                    y_slsqp = np.concatenate((res_slsqp.x,xmes_res.get('SLSQP_'+bound+'_'+approach+'_'+obj_fun)))
                    y_ipopt = np.concatenate((res_ipopt.x,xmes_res.get('ipopt_'+bound+'_'+approach+'_'+obj_fun)))
                    x_opf_LF = np.concatenate((np.array([V0_sol,P0c_sol]),xmes_LF))
                print('\nObj: {}, bounds: {}, approach: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nError for t-c:{}, SLSQP: {}, IPOPT: {}'.format(obj_fun,bound,approach,res_trust.success,res_slsqp.success,res_ipopt.success,error(y_trust,x_opf_LF),error(y_slsqp,x_opf_LF),error(y_ipopt,x_opf_LF)))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','MES2N')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def compare_ge_opf_integrated_LF_scaling_sep_LF(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF of gas-electricity network with integrated LF substituted for different optimization methods, objective function, and bounds. With scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')
    # solver info
    max_iters_lf = 10
    max_iter = 50
    tol = 1e-6
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

    # parameter values for the two objective functions
    a=0
    b=.01*MES.GHV
    c=1e-6*(MES.GHV)**2
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.3
    c0c=3e-5
    a1c=0
    b1c=.2
    c1c=2e-5

    # initial guesses and limits for the inequality constraints (when used)
    P0c_init = .8*MES.Pc0_sol
    V0_init = .9*MES.V0_sol

    # bounds
    ineq_constr='all'
    P0c_lb=.5*MES.Pc0_sol
    P0c_ub=1*MES.Pc0_sol
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    q0_lb=1.5*MES.q0_source
    q0_ub=.5*MES.q0_source
    q01_lb=-5
    q01_ub=5
    p1_lb=1*mbar
    p1_ub=1.1*MES.pg0
    delta0_lb=-np.pi
    delta0_ub=np.pi
    q0c_lb=0
    q0c_ub=1.5*MES.qc0_sol
    q1c_lb=0
    q1c_ub=1.5*MES.qc1_sol
    P1c_lb=0
    P1c_ub=1.5*MES.Pc1_sol
    Q0c_lb=-2*MES.Qc0_sol
    Q0c_ub=2*MES.Qc0_sol
    Q1c_lb=-5*MES.Qc1_sol
    Q1c_ub=5*MES.Qc1_sol

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, gas_net_LF, elec_net_LF, xmes_LF, iters_LF, err_vec_LF = run_mes_ge_load_flow(max_iter=max_iters_lf,tol=tol,scale_var=None,formulation=formulation,plot_top=False,plot_jac=False,plot_sol=False)
        q0_sol = gas_net_LF.nodes[0].half_links[0].get_q()
        V0_sol = elec_net_LF.nodes[0].get_V()
        P0c_sol = elec_net_LF.links[1].get_Pstart()
        P1c_sol = elec_net_LF.links[2].get_Pstart()
        # value of objective functions for LF solution
        f_LF_gas = price_gas(q0_sol,a=a,b=b,c=c)
        f_LF_comb_obj = price_gas_electricity(q0_sol,P0c_sol,P1c_sol,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)

    # base values
    Sbase = 1*MW #[W]
    Vbase = 10/np.sqrt(3)*kV #[V]
    deltabase = 1.
    pbase = 10*mbar #[Pa]
    qbase = .01 #[kg/s]
    Egbase = 1*MW #[W]
    scale_var_params={'qbase':qbase,'pbase':pbase,'pgbase':pbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'Ebase':Egbase}
    obj_base = {'gas':1e7,'gas_elec':1e8}

    result = dict()
    xmes_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']
    obj_funs = ['gas','gas_elec']
    scaling = ['matrix','per_unit']
    # Optimal Flow
    for obj_fun in obj_funs:
        if obj_fun == 'gas':
            obj_fun_label = 'gas'
        else:
            obj_fun_label = 'combined'
        fb = obj_base.get(obj_fun)
        for scale_var in scaling:
            # plots
            fig_f = plt.figure('obj_OPF_integrated_MES_ge_sep_LF_'+obj_fun_label+'_'+scale_var)
            ax_f = fig_f.gca()
            ax_f.set_xlabel('Iteration')
            ax_f.set_ylabel('f')
            # value of objective functions for LF solution
            if scale_var == 'per_unit':
                if obj_fun == 'gas':
                    f_LF_sol = price_gas(q0_sol/scale_var_params.get('qbase'),a=a/fb,b=b/(fb/scale_var_params.get('qbase')),c=c/(fb/scale_var_params.get('qbase')**2))
                else:
                    f_LF_sol = price_gas_electricity(q0_sol/scale_var_params.get('qbase'),P0c_sol/scale_var_params.get('Sbase'),P1c_sol/scale_var_params.get('Sbase'),a0=a0/fb,b0=b0/(fb/scale_var_params.get('qbase')),c0=c0/(fb/scale_var_params.get('qbase')**2),a0c=a0c/fb,b0c=b0c/(fb/scale_var_params.get('Sbase')),c0c=c0c/(fb/scale_var_params.get('Sbase')**2),a1c=a1c/fb,b1c=b1c/(fb/scale_var_params.get('Sbase')),c1c=c1c/(fb/scale_var_params.get('Sbase')**2))
            else:
                if obj_fun == 'gas':
                    f_LF_sol = price_gas(q0_sol,a=a,b=b,c=c)/fb
                else:
                    f_LF_sol = price_gas_electricity(q0_sol,P0c_sol,P1c_sol,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)/fb

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
                            x_opf, xmes_opt, f_vec, fun, nfev, nit, execution_time, success = run_ge_optimal_load_flow_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_lb=q0_lb,q0_ub=q0_ub,q01_lb=q01_lb,q01_ub=q01_ub,p1_lb=p1_lb,p1_ub=p1_ub,delta0_lb=delta0_lb,delta0_ub=delta0_ub,q0c_lb=q0c_lb,q0c_ub=q0c_ub,q1c_lb=q1c_lb,q1c_ub=q1c_ub,P1c_lb=P1c_lb,P1c_ub=P1c_ub,Q0c_lb=Q0c_lb,Q0c_ub=Q0c_ub,Q1c_lb=Q1c_lb,Q1c_ub=Q1c_ub,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation,a=a,b=b,c=c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,fb=fb,ineq_constr=ineq_constr,objective=obj_fun,optimization_method=method,stay_within_bounds=stay_within_bounds,approach=approach)
                        else:
                            approach_legend = 'an'
                            x_opf, xmes_opt, f_vec, fun, nfev, nit, njev, execution_time, success = run_ge_optimal_load_flow(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_lb=q0_lb,q0_ub=q0_ub,q01_lb=q01_lb,q01_ub=q01_ub,p1_lb=p1_lb,p1_ub=p1_ub,delta0_lb=delta0_lb,delta0_ub=delta0_ub,q0c_lb=q0c_lb,q0c_ub=q0c_ub,q1c_lb=q1c_lb,q1c_ub=q1c_ub,P1c_lb=P1c_lb,P1c_ub=P1c_ub,Q0c_lb=Q0c_lb,Q0c_ub=Q0c_ub,Q1c_lb=Q1c_lb,Q1c_ub=Q1c_ub,max_iter=max_iter,tol=tol,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation,a=a,b=b,c=c,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,fb=fb,ineq_constr=ineq_constr,derivatives=True,objective=obj_fun,optimization_method=method,stay_within_bounds=stay_within_bounds)
                        result[method+'_'+bound+'_'+approach+'_'+obj_fun+'_'+scale_var] = spo.OptimizeResult({'success':success,'x':x_opf,'nit':nit,'nfev':nfev,'execution_time':execution_time})
                        xmes_res[method+'_'+bound+'_'+approach+'_'+obj_fun+'_'+scale_var] = xmes_opt
                        max_fev = max(max_fev,len(f_vec))
                        # plot results
                        ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
            ax_f.plot([0,max_fev],[f_LF_sol,f_LF_sol],':r')
            ax_f.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','MES2N')
        with open(os.path.join(path_to_tables,'optimizer_info_integrated_MES_ge_scaling_sep_LF.txt'), "w") as table:
            for obj_fun in obj_funs:
                if obj_fun == 'gas':
                    obj_fun_label = 'gas'
                else:
                    obj_fun_label = 'combined'
                for bound in bounds:
                    for approach in approaches:
                        if approach == 'eq_constr':
                            approach_label = 'eq. constr.'
                        else:
                            approach_label = approach
                        res_trust_mat = result.get('trust-constr_'+bound+'_'+approach+'_'+obj_fun+'_matrix')
                        res_slsqp_mat = result.get('SLSQP_'+bound+'_'+approach+'_'+obj_fun+'_matrix')
                        res_ipopt_mat = result.get('ipopt_'+bound+'_'+approach+'_'+obj_fun+'_matrix')
                        xmes_opt_trust_mat = xmes_res.get('trust-constr_'+bound+'_'+approach+'_'+obj_fun+'_matrix')
                        xmes_opt_slsqp_mat = xmes_res.get('SLSQP_'+bound+'_'+approach+'_'+obj_fun+'_matrix')
                        xmes_opt_ipopt_mat = xmes_res.get('ipopt_'+bound+'_'+approach+'_'+obj_fun+'_matrix')
                        res_trust_pu = result.get('trust-constr_'+bound+'_'+approach+'_'+obj_fun+'_per_unit')
                        res_slsqp_pu = result.get('SLSQP_'+bound+'_'+approach+'_'+obj_fun+'_per_unit')
                        res_ipopt_pu = result.get('ipopt_'+bound+'_'+approach+'_'+obj_fun+'_per_unit')
                        xmes_opt_trust_pu = xmes_res.get('trust-constr_'+bound+'_'+approach+'_'+obj_fun+'_per_unit')
                        xmes_opt_slsqp_pu = xmes_res.get('SLSQP_'+bound+'_'+approach+'_'+obj_fun+'_per_unit')
                        xmes_opt_ipopt_pu = xmes_res.get('ipopt_'+bound+'_'+approach+'_'+obj_fun+'_per_unit')
                        table.write(r'{} & {} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e}\\ '.format(obj_fun_label,bound,approach_label,res_trust_mat.success,res_trust_pu.success,res_slsqp_mat.success,res_slsqp_pu.success,res_ipopt_mat.success,res_ipopt_pu.success,res_trust_mat.nit,res_trust_pu.nit,res_slsqp_pu.nit,res_slsqp_mat.nit,res_ipopt_mat.nit,res_ipopt_pu.nit,error(xmes_opt_trust_mat,xmes_LF),error(xmes_opt_slsqp_mat,xmes_LF),error(xmes_opt_ipopt_mat,xmes_LF),error(xmes_opt_trust_pu,xmes_LF),error(xmes_opt_slsqp_pu,xmes_LF),error(xmes_opt_ipopt_pu,xmes_LF)))
                    table.write(r'\cline{2-21} ')
                table.write(r'\hline ')

    for obj_fun in obj_funs:
        for bound in bounds:
            for approach in approaches:
                for scale_var in scaling:
                    res_trust = result.get('trust-constr_'+bound+'_'+approach+'_'+obj_fun+'_'+scale_var)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+approach+'_'+obj_fun+'_'+scale_var)
                    res_ipopt = result.get('ipopt_'+bound+'_'+approach+'_'+obj_fun+'_'+scale_var)
                    xmes_opt_trust = xmes_res.get('trust-constr_'+bound+'_'+approach+'_'+obj_fun+'_'+scale_var)
                    xmes_opt_slsqp = xmes_res.get('SLSQP_'+bound+'_'+approach+'_'+obj_fun+'_'+scale_var)
                    xmes_opt_ipopt = xmes_res.get('ipopt_'+bound+'_'+approach+'_'+obj_fun+'_'+scale_var)
                    print('\nObj: {}, bounds: {}, approach: {}, scaling: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nError for t-c:{}, SLSQP: {}, IPOPT: {}'.format(obj_fun,bound,approach,scaling,res_trust.success,res_slsqp.success,res_ipopt.success,error(xmes_opt_trust,xmes_LF),error(xmes_opt_slsqp,xmes_LF),error(xmes_opt_ipopt,xmes_LF)))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','MES2N')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def compare_ge_opf_DD_LF(dir_path,number_runs=10,save_figs=False,save_tables=False):
    """Compare OF for the different versions with the DD MES for LF"""
    # solver info
    max_iters_lf = 20
    max_iter = 300
    tol = 1e-6
    scale_var = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    ineq_constr = True
    Fc_eq_constr = True

    # parameter values for the two objective functions
    a=0
    b=.01*MES.GHV
    c=1e-6*(MES.GHV)**2
    a0=0
    b0=.01*MES.GHV
    c0=1e-6*(MES.GHV)**2
    a0c=0
    b0c=.3
    c0c=3e-5
    a1c=0
    b1c=.2
    c1c=2e-5

    # initial guesses, and limist for the inequality constraints
    P0c_init = 1.1*MES.Pc0_sol
    P0c_lb=1*MES.Pc0_sol
    P0c_ub=1.5*MES.Pc0_sol
    V0_init = 1.05*10/np.sqrt(3)*kV
    V0_lb=0.8*MES.V0_sol
    V0_ub=1*MES.V0_sol
    q0_init = 1.1*MES.q0_source
    q0_lb=1.3*MES.q0_source
    q0_ub=1*MES.q0_source
    q0c_init = .9*MES.qc0_sol
    q0c_lb=1*MES.qc0_sol
    q0c_ub=1.3*MES.qc0_sol
    q1c_init = 1.1*MES.qc1_sol
    q1c_lb=.7*MES.qc1_sol
    q1c_ub=1*MES.qc1_sol

    # steady-state LF solution (of integrated MES)
    with HiddenPrints():
        het_net_LF, gas_net_LF, elec_net_LF, xmes_LF, iters_LF, err_vec_LF = run_mes_ge_load_flow(max_iter=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,plot_top=False,plot_jac=False,plot_sol=False)
        q0_sol = gas_net_LF.nodes[0].half_links[0].get_q()
        P0c_sol = elec_net_LF.links[1].get_Pstart()
        P1c_sol = elec_net_LF.links[2].get_Pstart()
        # value of objective functions for LF solution
        f_LF_gas = price_gas(q0_sol,a=a,b=b,c=c)
        f_LF_comb_obj = price_gas_electricity(q0_sol,P0c_sol,P1c_sol,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c)

    # run the various optimizations. Run several times, take average of run time. For the other data (which seemed to be the same every time), the last run is used.
    exec_times_num_der_DD1_comb_obj = list()
    exec_times_num_der_DD1 = list()
    exec_times_DD2 = list()
    exec_times_num_der_DD2 = list()
    exec_times_sepLF_direct_DD2 = list()
    exec_times_sepLF_adjoint_DD2 = list()
    for run in range(number_runs):
        # LF is included as (nonlinear) equality constriant. Numerical expressions for gradients and Hessian of objective function and Jacobian of equality constraints are used. Objective function for gas input only. First node type set for DD.
        x_opf_num_der_DD1, xmes_opt_num_der_DD1, f_vec_num_der_DD1, obj_fun_num_der_DD1, nfev_num_der_DD1, nit_num_der_DD1, exec_time_num_der_DD1 = run_ge_optimal_load_flow_dd(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0c_init=q0c_init,q0c_lb=q0c_lb,q0c_ub=q0c_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation,a=a,b=b,c=c,ineq_constr=ineq_constr,objective='gas')
        exec_times_num_der_DD1.append(exec_time_num_der_DD1)
        # LF is included as (nonlinear) equality constriant. Numerical expressions for gradients and Hessian of objective function and Jacobian of equality constraints are used. Objective function for gas input and coupling active powers. First node type set for DD.
        x_opf_num_der_DD1_comb_obj, xmes_opt_num_der_DD1_comb_obj, f_vec_num_der_DD1_comb_obj, obj_fun_num_der_DD1_comb_obj, nfev_num_der_DD1_comb_obj, nit_num_der_DD1_comb_obj, exec_time_num_der_DD1_comb_obj = run_ge_optimal_load_flow_dd(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0c_init=q0c_init,q0c_lb=q0c_lb,q0c_ub=q0c_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation,a=a,b=b,c=c,ineq_constr=ineq_constr,objective='gas_elec')
        exec_times_num_der_DD1_comb_obj.append(exec_time_num_der_DD1_comb_obj)
        # LF is included as (nonlinear) equality constriant. Numerical expressions for gradients and Hessian of objective function and Jacobian of equality constraints are used. Objective function for gas input and coupling active powers. Second node type set for DD.
        x_opf_num_der_DD2, xmes_opt_num_der_DD2, f_vec_num_der_DD2, obj_fun_num_der_DD2, nfev_num_der_DD2, nit_num_der_DD2, exec_time_num_der_DD2 = run_ge_optimal_load_flow_dd2(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_init=q0_init,q0_lb=q0_lb,q0_ub=q0_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation,derivatives=False)
        exec_times_num_der_DD2.append(exec_time_num_der_DD2)
        # LF is included as (nonlinear) equality constriant. Analytical expressions for gradients and Hessian of objective function and Jacobian of equality constraints are used. Objective function for gas input and coupling active powers. Second node type set for DD.
        x_opf_DD2, xmes_opt_DD2, f_vec_DD2, obj_fun_DD2, nfev_DD2, nit_DD2, exec_time_DD2 = run_ge_optimal_load_flow_dd2(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_init=q0_init,q0_lb=q0_lb,q0_ub=q0_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,max_iter=max_iter,tol=tol,scale_var=scale_var,formulation=formulation,derivatives=True)
        exec_times_DD2.append(exec_time_DD2)
        # LF of gas and electrical single-carrier parts is included as implicitly, LF of coupling part in included as (nonlinear) equality constraints. Analytical expressions, using the direct approach, for gradients and Hessian of objective function and Jacobian of equality constraints are used. Objective function for gas input and coupling active powers. Second node type set for DD.
        x_opf_sepLF_direct_DD2, xmes_opt_sepLF_direct_DD2, f_vec_sepLF_direct_DD2, obj_fun_sepLF_direct_DD2, nfev_sepLF_direct_DD2, nit_sepLF_direct_DD2, exec_time_sepLF_direct_DD2 = run_ge_optimal_load_flow_dd2_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_init=q0_init,q0_lb=q0_lb,q0_ub=q0_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,ineq_constr=ineq_constr,Fc_eq_constr=True,approach='direct')
        exec_times_sepLF_direct_DD2.append(exec_time_sepLF_direct_DD2)
        # LF of gas and electrical single-carrier parts is included as implicitly, LF of coupling part in included as (nonlinear) equality constraints. Analytical expressions, using adjoint approach, for gradients and Hessian of objective function and Jacobian of equality constraints are used. Objective function for gas input and coupling active powers. Second node type set for DD.
        x_opf_sepLF_adjoint_DD2, xmes_opt_sepLF_adjoint_DD2, f_vec_sepLF_adjoint_DD2, obj_fun_sepLF_adjoint_DD2, nfev_sepLF_adjoint_DD2, nit_sepLF_adjoint_DD2, exec_time_sepLF_adjoint_DD2 = run_ge_optimal_load_flow_dd2_separate_LF(P0c_init=P0c_init,P0c_lb=P0c_lb,P0c_ub=P0c_ub, V0_init=V0_init,V0_lb=V0_lb,V0_ub=V0_ub,q0_init=q0_init,q0_lb=q0_lb,q0_ub=q0_ub,q1c_init=q1c_init,q1c_lb=q1c_lb,q1c_ub=q1c_ub,a0=a0,b0=b0,c0=c0,a0c=a0c,b0c=b0c,c0c=c0c,a1c=a1c,b1c=b1c,c1c=c1c,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,formulation=formulation,ineq_constr=ineq_constr,Fc_eq_constr=True,approach='adjoint')
        exec_times_sepLF_adjoint_DD2.append(exec_time_sepLF_adjoint_DD2)

    exec_time_num_der_DD1_comb_obj = np.mean(exec_times_num_der_DD1_comb_obj)
    exec_time_num_der_DD1 = np.mean(exec_times_num_der_DD1)
    exec_time_DD2 = np.mean(exec_times_DD2)
    exec_time_num_der_DD2 = np.mean(exec_times_num_der_DD2)
    exec_time_sepLF_direct_DD2 = np.mean(exec_times_sepLF_direct_DD2)
    exec_time_sepLF_adjoint_DD2 = np.mean(exec_times_sepLF_adjoint_DD2)

    # create (and save) table with difference between OF and LF solution. For intergrated MES, x_LF = [q01 p1 delta0 q0c q1c P1c Q0c Q1c]
    def rel_diff(x1,x2):
        return np.abs(x1-x2)/np.abs(x1)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','MES2N')
        variable_names = [r'$q_{01}$',r'$p_1$',r'$\delta_0$',r'$q_{0c}',r'$q_{1c}$',r'$P_{1c}$',r'$Q_{0c}$',r'$Q_{1c}$']
        with open(os.path.join(path_to_tables,'solution_LF_vs_OF_DD_MES.txt'), "w") as table:
            for ind_var,var in enumerate(variable_names):
                table.write(r'{} & {:.5e} & {:.5e} &  {:.5e} &  {:.5e}  & {:.5e}  &  {:.5e}  & {:.5e}  \\ '.format(var,xmes_LF[ind_var],rel_diff(xmes_LF,xmes_opt_num_der_DD1)[ind_var],rel_diff(xmes_LF,xmes_opt_num_der_DD1_comb_obj)[ind_var],rel_diff(xmes_LF,xmes_opt_num_der_DD2)[ind_var],rel_diff(xmes_LF,xmes_opt_DD2)[ind_var],rel_diff(xmes_LF,xmes_opt_sepLF_adjoint_DD2)[ind_var],rel_diff(xmes_LF,xmes_opt_sepLF_direct_DD2)[ind_var]))
            table.write(r'\hline ')
            table.write(r'max. & & {:.5e} &  {:.5e} &  {:.5e}  & {:.5e}  &  {:.5e}  & {:.5e} \\ '.format(np.max(rel_diff(xmes_LF,xmes_opt_num_der_DD1)),np.max(rel_diff(xmes_LF,xmes_opt_num_der_DD1_comb_obj)),np.max(rel_diff(xmes_LF,xmes_opt_num_der_DD2)),np.max(rel_diff(xmes_LF,xmes_opt_DD2)),np.max(rel_diff(xmes_LF,xmes_opt_sepLF_direct_DD2)),np.max(rel_diff(xmes_LF,xmes_opt_sepLF_adjoint_DD2))))
    # print results of optimizer, and create (and save) table
    print('\nopf num. der. DD1    num. der. DD1 comb obj   num. der. DD2    opf DD2  opf sep. LF direct DD2    opf sep. LF adjoint DD2')
    print('obj. func:  {:.5e}  , {:.5e} , {:.5e} , {:.5e},   {:.5e}  , {:.5e}'.format(obj_fun_num_der_DD1,obj_fun_num_der_DD1_comb_obj,obj_fun_num_der_DD2,obj_fun_DD2,obj_fun_sepLF_direct_DD2,obj_fun_sepLF_adjoint_DD2))
    print('numb. fev.:  {:d}  , {:d} , {:d} , {:d},  {:d}  , {:d}'.format(nfev_num_der_DD1,nfev_num_der_DD1_comb_obj,nfev_num_der_DD2,nfev_DD2,nfev_sepLF_direct_DD2,nfev_sepLF_adjoint_DD2))
    print('iters:  {:d}  , {:d}  , {:d} , {:d},  {:d}  , {:d}'.format(nit_num_der_DD1,nit_num_der_DD1_comb_obj,nit_num_der_DD2,nit_DD2,nit_sepLF_direct_DD2,nit_sepLF_adjoint_DD2))
    print('time:  {:.5f}  , {:.5f} , {:5f} , {:.5f},  {:.5f}  , {:.5f}\n'.format(exec_time_num_der_DD1,exec_time_num_der_DD1_comb_obj,exec_time_num_der_DD2,exec_time_DD2,exec_time_sepLF_direct_DD2,exec_time_sepLF_adjoint_DD2))
    if save_tables:
        with open(os.path.join(path_to_tables,'optimizer_info_DD_MES.txt'), "w") as table:
            table.write(r'$f$ & {:.5e} & {:.5e} &  {:.5e} &  {:.5e}  & {:.5e} & {:.5e}   & {:.5e} & {:.5e} \\ '.format(f_LF_gas,obj_fun_num_der_DD1,f_LF_comb_obj,obj_fun_num_der_DD1_comb_obj,obj_fun_num_der_DD2,obj_fun_DD2,obj_fun_sepLF_direct_DD2,obj_fun_sepLF_adjoint_DD2))
            table.write(r'func. eval. & & {:d} & & {:d}  & {:d}  & {:d}   & {:d}  & {:d} \\ '.format(nfev_num_der_DD1,nfev_num_der_DD1_comb_obj,nfev_num_der_DD2,nfev_DD2,nfev_sepLF_direct_DD2,nfev_sepLF_adjoint_DD2))
            table.write(r'iterations & & {:d} & & {:d}  & {:d}  & {:d}   & {:d}  & {:d} \\ '.format(nit_num_der_DD1,nit_num_der_DD1_comb_obj,nit_num_der_DD2,nit_DD2,nit_sepLF_direct_DD2,nit_sepLF_adjoint_DD2))
            table.write(r'time [s] & & {:.5f} & & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f} \\ '.format(exec_time_num_der_DD1,exec_time_num_der_DD1_comb_obj,exec_time_num_der_DD2,exec_time_DD2,exec_time_sepLF_direct_DD2,exec_time_sepLF_adjoint_DD2))

    # plots
    colors = {'LF':'k', 'OPF':'tab:blue', 'OPF direct':'tab:orange', 'OPF adjoint':'tab:green', 'OPF num. der.':'tab:red','DD1':'tab:purple'}
    fig_f_gas = plt.figure('objective_function_gas_OF_DD_MES')
    ax_f_gas = fig_f_gas.gca()
    ax_f_gas.set_xlabel('Iteration')
    ax_f_gas.set_ylabel('f')

    fig_f_comb_obj = plt.figure('objective_function_comb_obj_OF_DD_MES')
    ax_f_comb_obj = fig_f_comb_obj.gca()
    ax_f_comb_obj.set_xlabel('Iteration')
    ax_f_comb_obj.set_ylabel('f')

    # plot results with node set 1
    ax_f_gas.plot(f_vec_num_der_DD1,color=colors.get('DD1'),label='OF num. der. DD1')
    ax_f_comb_obj.plot(f_vec_num_der_DD1_comb_obj,color=colors.get('DD1'),label='OF num. der. DD1')
    # plot results with node set 2
    ax_f_comb_obj.plot(f_vec_num_der_DD2,color=colors.get('OPF num. der.'),label='OF num. der. DD2')
    ax_f_comb_obj.plot(f_vec_DD2,color=colors.get('OPF'),label='OF DD2')
    ax_f_comb_obj.plot(f_vec_sepLF_direct_DD2,color=colors.get('OPF direct'),label='sep. LF direct DD2')
    ax_f_comb_obj.plot(f_vec_sepLF_adjoint_DD2,color=colors.get('OPF adjoint'),label='sep. LF adjoint DD2')

    # layout
    nit_max_gas = np.max([nit_num_der_DD1,nit_num_der_DD1_comb_obj])
    ax_f_gas.plot([0,nit_max_gas],[f_LF_gas,f_LF_gas],ls=':',color=colors.get('LF'),label='LF')
    ax_f_gas.set_xlim(left=0,right=nit_max_gas)
    ax_f_gas.legend()
    nit_max_comb_obj = np.max([nit_num_der_DD2,nit_DD2,nit_sepLF_direct_DD2,nit_sepLF_adjoint_DD2])
    ax_f_comb_obj.plot([0,nit_max_comb_obj],[f_LF_comb_obj,f_LF_comb_obj],ls=':',color=colors.get('LF'),label='LF')
    ax_f_comb_obj.set_xlim(left=0,right=nit_max_comb_obj)
    ax_f_comb_obj.legend()

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','MES2N')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def compare_eh_opf_integrated_LF_methods(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF of electricity-heat network for different optimization methods, objective function, and bounds. Without scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')
    # solver info
    max_iter = 100
    tol = 1e-6
    scale_var = None
    fb = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

    # parameter values for the objective function
    # costs for active power coupling 0
    a0=0
    b0=.3
    c0=3e-5
    # costs for active power coupling 1
    a1=0
    b1=.2
    c1=2e-5
    # costs for heat power coupling 0
    a2=0#a1
    b2=.04#b1
    c2=4e-4#c1
    # costs for active power coupling 1
    a3=0#a0
    b3=.05#b0
    c3=4.5e-4#c0

    # initial guesses and limits for the inequality constraints (when used)
    V0_init = .9*MES.V0_sol
    q0c_init=1.3*MES.qc0_sol_CHP
    q1c_init=1.3*MES.qc1_sol_CHP
    P0c_init=1.3*MES.Pc0_sol
    P1c_init=1.5*MES.Pc1_sol
    dphi0c_init=1.5*MES.phic0_sol
    dphi1c_init=0.8*MES.phic1_sol

    # bounds
    V0_bounds=np.array([0.8*MES.V0_sol,1*MES.V0_sol])
    q0c_bounds=np.array([1*MES.qc0_sol_CHP,1.5*MES.qc0_sol_CHP])
    q1c_bounds=np.array([1*MES.qc1_sol_CHP,1.5*MES.qc1_sol_CHP])
    delta0_bounds=np.array([-np.pi,np.pi])
    m01_bounds=np.array([-3*MES.m01_sol,3*MES.m01_sol])
    m0_bounds=np.array([0,5*MES.m0_sink])
    m1_bounds=np.array([0,5*MES.m1_sink])
    p1_bounds=np.array([10,5*MES.ph1_sol])
    Ts0_bounds=np.array([60,140])
    Ts1_bounds=np.array([60,140])
    Tr0_bounds=np.array([10,60])
    Tr1_bounds=np.array([10,60])
    P0c_bounds=np.array([0,3*MES.Pc0_sol])
    P1c_bounds=np.array([0,3*MES.Pc1_sol])
    Q0c_bounds=np.array([-3*MES.Qc0_sol,3*MES.Qc0_sol])
    Q1c_bounds=np.array([-3*MES.Qc1_sol,3*MES.Qc1_sol])
    m0c_bounds=np.array([0,3*MES.mc0_sol])
    m1c_bounds=np.array([0,3*MES.mc1_sol])
    dphi0c_bounds=np.array([0,3*MES.phic0_sol])
    dphi1c_bounds=np.array([0,3*MES.phic1_sol])

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, elec_net_LF, heat_net_LF, xmes_LF, iters_LF, err_vec_LF = MES.run_mes_eh_load_flow(max_iter=10)
        V0_sol = elec_net_LF.nodes[0].get_V()
        q0c_sol = -het_net_LF.nodes[4].half_links[0].get_q()
        q1c_sol = -het_net_LF.nodes[5].half_links[0].get_q()
        P0c_sol, P1c_sol, dphi0c_sol, dphi1c_sol =  xmes_LF[[9,10,15,16]]
        # value of objective functions for LF solution
        f_LF_sol = price_electricity_heat(P0c_sol, P1c_sol, dphi0c_sol, dphi1c_sol,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3)
        x_opf_LF = np.concatenate((np.array([V0_sol,q0c_sol,q1c_sol]),xmes_LF))

    result = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']
    ineq_constrs = ['control','all']
    # Optimal Flow
    for ineq_constr in ineq_constrs:
        # plots
        fig_f = plt.figure('obj_OPF_integrated_MES_eh_'+ineq_constr)
        ax_f = fig_f.gca()
        ax_f.set_xlabel('Iteration')
        ax_f.set_ylabel('f')

        fig_LF_error = plt.figure('LF_error_OPF_integrated_MES_eh_'+ineq_constr)
        ax_LF_error = fig_LF_error.gca()
        ax_LF_error.set_xlabel('Iteration')
        ax_LF_error.set_ylabel(r'$||F||_2$')

        fig_q0c = plt.figure('q_0c_OPF_integrated_MES_eh_'+ineq_constr)
        ax_q0c = fig_q0c.gca()
        ax_q0c.set_xlabel('Iteration')
        ax_q0c.set_ylabel(r'$q_{0c}$ [kg/s]')

        fig_q1c = plt.figure('q_1c_OPF_integrated_MES_eh_'+ineq_constr)
        ax_q1c = fig_q1c.gca()
        ax_q1c.set_xlabel('Iteration')
        ax_q1c.set_ylabel(r'$q_{1c}$ [kg/s]')

        fig_P0c = plt.figure('P_0c_OPF_integrated_MES_eh_'+ineq_constr)
        ax_P0c = fig_P0c.gca()
        ax_P0c.set_xlabel('Iteration')
        ax_P0c.set_ylabel(r'$P_{0c}$ [W]')

        fig_P1c = plt.figure('P_1c_OPF_integrated_MES_eh_'+ineq_constr)
        ax_P1c = fig_P1c.gca()
        ax_P1c.set_xlabel('Iteration')
        ax_P1c.set_ylabel(r'$P_{1c}$ [W]')

        fig_dphi0c = plt.figure('phi_0c_OPF_integrated_MES_eh_'+ineq_constr)
        ax_dphi0c = fig_dphi0c.gca()
        ax_dphi0c.set_xlabel('Iteration')
        ax_dphi0c.set_ylabel(r'$\Delta \varphi_{0c}$ [W]')

        fig_dphi1c = plt.figure('phi_1c_OPF_integrated_MES_eh_'+ineq_constr)
        ax_dphi1c = fig_dphi1c.gca()
        ax_dphi1c.set_xlabel('Iteration')
        ax_dphi1c.set_ylabel(r'$\Delta \varphi_{1c}$ [W]')

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
                    xmes_opt, res, f_vec, u_mat, F_mat, E_mat, execution_time = run_eh_optimal_load_flow(V0_init=V0_init,V0_bounds=V0_bounds,q0c_init=q0c_init,q0c_bounds=q0c_bounds,q1c_init=q1c_init,q1c_bounds=q1c_bounds,delta0_bounds=delta0_bounds,m01_bounds=m01_bounds,m0_bounds=m0_bounds,m1_bounds=m1_bounds,p1_bounds=p1_bounds,Ts0_bounds=Ts0_bounds,Ts1_bounds=Ts1_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,P0c_init=P0c_init,P0c_bounds=P0c_bounds,P1c_init=P1c_init,P1c_bounds=P1c_bounds,Q0c_bounds=Q0c_bounds,Q1c_bounds=Q1c_bounds,m0c_bounds=m0c_bounds,m1c_bounds=m1c_bounds,dphi0c_init=dphi0c_init,dphi0c_bounds=dphi0c_bounds, dphi1c_init=dphi1c_init,dphi1c_bounds=dphi1c_bounds,max_iter=max_iter,tol=tol,scale_var=None,scale_var_params=None,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,formulation=formulation,ineq_constr=ineq_constr,derivatives=derivatives,stay_within_bounds=stay_within_bounds,optimization_method=method,fb=None)
                    # save result in dictionaries
                    result[method+'_'+bound+'_'+der+'_'+ineq_constr] = res
                    max_fev = max(max_fev,len(f_vec))
                    ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_LF_error.plot([np.linalg.norm(F_mat[ind,:]) for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_q0c.plot([u_mat[ind,1] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_q1c.plot([u_mat[ind,2] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_P0c.plot([E_mat[ind,0] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_P1c.plot([E_mat[ind,1] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_dphi0c.plot([E_mat[ind,2] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_dphi1c.plot([E_mat[ind,3] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
        ax_f.plot([0,max_fev],[f_LF_sol,f_LF_sol],':r')
        ax_f.legend(handles=legend_handles)
        ax_LF_error.plot([0,max_fev],[tol,tol],':k')
        ax_LF_error.legend(handles=legend_handles)
        ax_q0c.plot([0,max_fev],[q0c_sol,q0c_sol],':r')
        ax_q0c.plot([0,max_fev],[q0c_bounds[0],q0c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q0c.plot([0,max_fev],[q0c_bounds[1],q0c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q0c.legend(handles=legend_handles)
        ax_q1c.plot([0,max_fev],[q1c_sol,q1c_sol],':r')
        ax_q1c.plot([0,max_fev],[q1c_bounds[0],q1c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q1c.plot([0,max_fev],[q1c_bounds[1],q1c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q1c.legend(handles=legend_handles)
        ax_P0c.plot([0,max_fev],[P0c_sol,P0c_sol],':r')
        ax_P0c.plot([0,max_fev],[P0c_bounds[0],P0c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P0c.plot([0,max_fev],[P0c_bounds[1],P0c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P0c.legend(handles=legend_handles)
        ax_P1c.plot([0,max_fev],[P1c_sol,P1c_sol],':r')
        ax_P1c.plot([0,max_fev],[P1c_bounds[0],P1c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P1c.plot([0,max_fev],[P1c_bounds[1],P1c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P1c.legend(handles=legend_handles)
        ax_dphi0c.plot([0,max_fev],[dphi0c_sol,dphi0c_sol],':r')
        ax_dphi0c.plot([0,max_fev],[dphi0c_bounds[0],dphi0c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0c.plot([0,max_fev],[dphi0c_bounds[1],dphi0c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0c.legend(handles=legend_handles)
        ax_dphi1c.plot([0,max_fev],[dphi1c_sol,dphi1c_sol],':r')
        ax_dphi1c.plot([0,max_fev],[dphi1c_bounds[0],dphi1c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi1c.plot([0,max_fev],[dphi1c_bounds[1],dphi1c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi1c.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','MES2N')
        with open(os.path.join(path_to_tables,'optimizer_info_integrated_MES_eh_methods.txt'), "w") as table:
            for ineq_constr in ineq_constrs:
                for bound in bounds:
                    for der in ders:
                        res_trust = result.get('trust-constr_'+bound+'_'+der+'_'+ineq_constr)
                        res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_'+ineq_constr)
                        res_ipopt = result.get('ipopt_'+bound+'_'+der+'_'+ineq_constr)
                        table.write(r'{} & {} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(ineq_constr,bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(res_trust.x,x_opf_LF),error(res_slsqp.x,x_opf_LF),error(res_ipopt.x,x_opf_LF)))
                table.write(r'\hline ')

    for ineq_constr in ineq_constrs:
        for bound in bounds:
            for der in ders:
                res_trust = result.get('trust-constr_'+bound+'_'+der+'_{}'.format(ineq_constr))
                res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_{}'.format(ineq_constr))
                res_ipopt = result.get('ipopt_'+bound+'_'+der+'_{}'.format(ineq_constr))
                print('\nConstraints: {}, bounds: {}, der: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\ntrust-constr:{}\nSLSQP: {}\nIPOPT: {}'.format(ineq_constr,bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','MES2N')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def compare_eh_opf_integrated_LF_scaling(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF of electricity-heat network for different optimization methods, using per unit or matrix scaling"""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')
    # solver info
    max_iter = 50
    tol = 1e-6
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    ineq_constr = 'all'

    # parameter values for the objective function
    # costs for active power coupling 0
    a0=0
    b0=.3
    c0=3e-5
    # costs for active power coupling 1
    a1=0
    b1=.2
    c1=2e-5
    # costs for heat power coupling 0
    a2=0#a1
    b2=.04#b1
    c2=4e-4#c1
    # costs for active power coupling 1
    a3=0#a0
    b3=.05#b0
    c3=4.5e-4#c0

    # initial guesses and limits for the inequality constraints (when used)
    V0_init = .9*MES.V0_sol
    q0c_init=1.3*MES.qc0_sol_CHP
    q1c_init=1.3*MES.qc1_sol_CHP
    P0c_init=1.3*MES.Pc0_sol
    P1c_init=1.5*MES.Pc1_sol
    dphi0c_init=1.5*MES.phic0_sol
    dphi1c_init=0.8*MES.phic1_sol

    # bounds
    V0_bounds=np.array([0.8*MES.V0_sol,1*MES.V0_sol])
    q0c_bounds=np.array([1*MES.qc0_sol_CHP,1.5*MES.qc0_sol_CHP])
    q1c_bounds=np.array([1*MES.qc1_sol_CHP,1.5*MES.qc1_sol_CHP])
    delta0_bounds=np.array([-np.pi,np.pi])
    m01_bounds=np.array([-3*MES.m01_sol,3*MES.m01_sol])
    m0_bounds=np.array([0,5*MES.m0_sink])
    m1_bounds=np.array([0,5*MES.m1_sink])
    p1_bounds=np.array([10,5*MES.ph1_sol])
    Ts0_bounds=np.array([60,140])
    Ts1_bounds=np.array([60,140])
    Tr0_bounds=np.array([10,60])
    Tr1_bounds=np.array([10,60])
    P0c_bounds=np.array([0,3*MES.Pc0_sol])
    P1c_bounds=np.array([0,3*MES.Pc1_sol])
    Q0c_bounds=np.array([-3*MES.Qc0_sol,3*MES.Qc0_sol])
    Q1c_bounds=np.array([-3*MES.Qc1_sol,3*MES.Qc1_sol])
    m0c_bounds=np.array([0,3*MES.mc0_sol])
    m1c_bounds=np.array([0,3*MES.mc1_sol])
    dphi0c_bounds=np.array([0,3*MES.phic0_sol])
    dphi1c_bounds=np.array([0,3*MES.phic1_sol])

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, elec_net_LF, heat_net_LF, xmes_LF, iters_LF, err_vec_LF = MES.run_mes_eh_load_flow(max_iter=10)
        V0_sol = elec_net_LF.nodes[0].get_V()
        q0c_sol = -het_net_LF.nodes[4].half_links[0].get_q()
        q1c_sol = -het_net_LF.nodes[5].half_links[0].get_q()
        P0c_sol, P1c_sol, dphi0c_sol, dphi1c_sol =  xmes_LF[[9,10,15,16]]

    # base values
    Sbase = 1*MW #[W]
    Vbase = 10/np.sqrt(3)*kV #[V]
    deltabase = 1.
    qbase = .1 #[kg/s]
    water = heat_net_LF.links[0].link_params.get('carrier')
    rho = water.rhon
    g = water.g
    phibase = 1.*MW #[W]
    Tbase = 100.#[C]
    mbase = 1.
    pbase = 100*rho*g
    Egbase = 1*MW #[W]
    scale_var_params={'qbase':qbase,'pbase':pbase,'pgbase':pbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':pbase,'phibase':phibase,'Tbase':Tbase,'Ebase':Egbase}
    fb = 1000.*MW

    # make scaled LF solution
    x_opf_LF = np.concatenate((np.array([V0_sol/scale_var_params.get('Vbase'),q0c_sol/scale_var_params.get('qbase'),q1c_sol/scale_var_params.get('qbase')]),xmes_LF/np.array([scale_var_params.get('deltabase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('pbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('phibase'),scale_var_params.get('phibase')])))

    result = dict()
    xh_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']
    scaling = ['matrix','per_unit']
    # Optimal Flow
    for scale_var in scaling:
        if scale_var == 'per_unit':
            f_LF_sol = price_electricity_heat(P0c_sol/scale_var_params.get('Sbase'), P1c_sol/scale_var_params.get('Sbase'), dphi0c_sol/scale_var_params.get('phibase'), dphi1c_sol/scale_var_params.get('phibase'),a0=a0/fb,b0=b0/(fb/scale_var_params.get('Sbase')),c0=c0/(fb/scale_var_params.get('Sbase')**2),a1=a1/fb,b1=b1/(fb/scale_var_params.get('Sbase')),c1=c1/(fb/scale_var_params.get('Sbase')**2),a2=a2/fb,b2=b2/(fb/scale_var_params.get('phibase')),c2=c2/(fb/scale_var_params.get('phibase')**2),a3=a3/fb,b3=b3/(fb/scale_var_params.get('phibase')),c3=c3/(fb/scale_var_params.get('phibase')**2))
        else:
            f_LF_sol = price_electricity_heat(P0c_sol, P1c_sol, dphi0c_sol, dphi1c_sol,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3)/fb
        # plots
        fig_f = plt.figure('obj_OPF_integrated_MES_eh_'+scale_var)
        ax_f = fig_f.gca()
        ax_f.set_xlabel('Iteration')
        ax_f.set_ylabel('f')

        fig_LF_error = plt.figure('LF_error_OPF_integrated_MES_eh_'+scale_var)
        ax_LF_error = fig_LF_error.gca()
        ax_LF_error.set_xlabel('Iteration')
        ax_LF_error.set_ylabel(r'$||F||_2$')

        fig_q0c = plt.figure('q_0c_OPF_integrated_MES_eh_'+scale_var)
        ax_q0c = fig_q0c.gca()
        ax_q0c.set_xlabel('Iteration')
        ax_q0c.set_ylabel(r'$q_{0c}$ [kg/s]')

        fig_q1c = plt.figure('q_1c_OPF_integrated_MES_eh_'+scale_var)
        ax_q1c = fig_q1c.gca()
        ax_q1c.set_xlabel('Iteration')
        ax_q1c.set_ylabel(r'$q_{1c}$ [kg/s]')

        fig_P0c = plt.figure('P_0c_OPF_integrated_MES_eh_'+scale_var)
        ax_P0c = fig_P0c.gca()
        ax_P0c.set_xlabel('Iteration')
        ax_P0c.set_ylabel(r'$P_{0c}$ [W]')

        fig_P1c = plt.figure('P_1c_OPF_integrated_MES_eh_'+scale_var)
        ax_P1c = fig_P1c.gca()
        ax_P1c.set_xlabel('Iteration')
        ax_P1c.set_ylabel(r'$P_{1c}$ [W]')

        fig_dphi0c = plt.figure('phi_0c_OPF_integrated_MES_eh_'+scale_var)
        ax_dphi0c = fig_dphi0c.gca()
        ax_dphi0c.set_xlabel('Iteration')
        ax_dphi0c.set_ylabel(r'$\Delta \varphi_{0c}$ [W]')

        fig_dphi1c = plt.figure('phi_1c_OPF_integrated_MES_eh_'+scale_var)
        ax_dphi1c = fig_dphi1c.gca()
        ax_dphi1c.set_xlabel('Iteration')
        ax_dphi1c.set_ylabel(r'$\Delta \varphi_{1c}$ [W]')

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
                    xmes_opt, res, f_vec, u_mat, F_mat, E_mat, execution_time = run_eh_optimal_load_flow(V0_init=V0_init,V0_bounds=V0_bounds,q0c_init=q0c_init,q0c_bounds=q0c_bounds,q1c_init=q1c_init,q1c_bounds=q1c_bounds,delta0_bounds=delta0_bounds,m01_bounds=m01_bounds,m0_bounds=m0_bounds,m1_bounds=m1_bounds,p1_bounds=p1_bounds,Ts0_bounds=Ts0_bounds,Ts1_bounds=Ts1_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,P0c_init=P0c_init,P0c_bounds=P0c_bounds,P1c_init=P1c_init,P1c_bounds=P1c_bounds,Q0c_bounds=Q0c_bounds,Q1c_bounds=Q1c_bounds,m0c_bounds=m0c_bounds,m1c_bounds=m1c_bounds,dphi0c_init=dphi0c_init,dphi0c_bounds=dphi0c_bounds, dphi1c_init=dphi1c_init,dphi1c_bounds=dphi1c_bounds,max_iter=max_iter,tol=tol,scale_var=scale_var,scale_var_params=scale_var_params,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,formulation=formulation,ineq_constr=ineq_constr,derivatives=derivatives,stay_within_bounds=stay_within_bounds,optimization_method=method,fb=fb)
                    # save result in dictionaries
                    result[method+'_'+bound+'_'+der+'_'+scale_var] = res
                    max_fev = max(max_fev,len(f_vec))
                    ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_LF_error.plot([np.linalg.norm(F_mat[ind,:]) for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_q0c.plot([u_mat[ind,1] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_q1c.plot([u_mat[ind,2] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_P0c.plot([E_mat[ind,0] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_P1c.plot([E_mat[ind,1] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_dphi0c.plot([E_mat[ind,2] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_dphi1c.plot([E_mat[ind,3] for ind in range(len(f_vec))],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
        ax_f.plot([0,max_fev],[f_LF_sol,f_LF_sol],':r')
        ax_f.legend(handles=legend_handles)
        ax_LF_error.plot([0,max_fev],[tol,tol],':k')
        ax_LF_error.legend(handles=legend_handles)
        ax_q0c.plot([0,max_fev],[q0c_sol/scale_var_params.get('qbase'),q0c_sol/scale_var_params.get('qbase')],':r')
        ax_q0c.plot([0,max_fev],[q0c_bounds[0]/scale_var_params.get('qbase'),q0c_bounds[0]/scale_var_params.get('qbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q0c.plot([0,max_fev],[q0c_bounds[1]/scale_var_params.get('qbase'),q0c_bounds[1]/scale_var_params.get('qbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q0c.legend(handles=legend_handles)
        ax_q1c.plot([0,max_fev],[q1c_sol/scale_var_params.get('qbase'),q1c_sol/scale_var_params.get('qbase')],':r')
        ax_q1c.plot([0,max_fev],[q1c_bounds[0]/scale_var_params.get('qbase'),q1c_bounds[0]/scale_var_params.get('qbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q1c.plot([0,max_fev],[q1c_bounds[1]/scale_var_params.get('qbase'),q1c_bounds[1]/scale_var_params.get('qbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q1c.legend(handles=legend_handles)
        ax_P0c.plot([0,max_fev],[P0c_sol/scale_var_params.get('Sbase'),P0c_sol/scale_var_params.get('Sbase')],':r')
        ax_P0c.plot([0,max_fev],[P0c_bounds[0]/scale_var_params.get('Sbase'),P0c_bounds[0]/scale_var_params.get('Sbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P0c.plot([0,max_fev],[P0c_bounds[1]/scale_var_params.get('Sbase'),P0c_bounds[1]/scale_var_params.get('Sbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P0c.legend(handles=legend_handles)
        ax_P1c.plot([0,max_fev],[P1c_sol/scale_var_params.get('Sbase'),P1c_sol/scale_var_params.get('Sbase')],':r')
        ax_P1c.plot([0,max_fev],[P1c_bounds[0]/scale_var_params.get('Sbase'),P1c_bounds[0]/scale_var_params.get('Sbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P1c.plot([0,max_fev],[P1c_bounds[1]/scale_var_params.get('Sbase'),P1c_bounds[1]/scale_var_params.get('Sbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P1c.legend(handles=legend_handles)
        ax_dphi0c.plot([0,max_fev],[dphi0c_sol/scale_var_params.get('phibase'),dphi0c_sol/scale_var_params.get('phibase')],':r')
        ax_dphi0c.plot([0,max_fev],[dphi0c_bounds[0]/scale_var_params.get('phibase'),dphi0c_bounds[0]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0c.plot([0,max_fev],[dphi0c_bounds[1]/scale_var_params.get('phibase'),dphi0c_bounds[1]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0c.legend(handles=legend_handles)
        ax_dphi1c.plot([0,max_fev],[dphi1c_sol/scale_var_params.get('phibase'),dphi1c_sol/scale_var_params.get('phibase')],':r')
        ax_dphi1c.plot([0,max_fev],[dphi1c_bounds[0]/scale_var_params.get('phibase'),dphi1c_bounds[0]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi1c.plot([0,max_fev],[dphi1c_bounds[1]/scale_var_params.get('phibase'),dphi1c_bounds[1]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi1c.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','MES2N')
        with open(os.path.join(path_to_tables,'optimizer_info_integrated_MES_eh_scaling.txt'), "w") as table:
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
        path_to_fig = os.path.join(dir_path,'Figures','MES2N')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def compare_eh_opf_integrated_LF_methods_sep_LF(dir_path=None,save_tables=False,save_figs=False,q_ub_fac=1.5,q_lb_fac=1,q_init_fac=1.3):
    """Compare OPF of electricity-heat network with integrated LF, which is substituted, for different optimization methods, and bounds. Without scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')
    # solver info
    max_iter = 50
    max_iters_lf = 10
    tol = 1e-6
    scale_var = None
    fb = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

    # parameter values for the objective function
    # costs for active power coupling 0
    a0=0
    b0=.3
    c0=3e-5
    # costs for active power coupling 1
    a1=0
    b1=.2
    c1=2e-5
    # costs for heat power coupling 0
    a2=0#a1
    b2=.04#b1
    c2=4e-4#c1
    # costs for active power coupling 1
    a3=0#a0
    b3=.05#b0
    c3=4.5e-4#c0

    # initial guesses (when used)
    V0_init = .9*MES.V0_sol
    q0c_init=q_init_fac*MES.qc0_sol_CHP#1.3*MES.qc0_sol_CHP
    q1c_init=q_init_fac*MES.qc1_sol_CHP#1.3*MES.qc1_sol_CHP
    P0c_init=1.3*MES.Pc0_sol
    P1c_init=1.5*MES.Pc1_sol
    dphi0c_init=1.5*MES.phic0_sol
    dphi1c_init=0.8*MES.phic1_sol

    # bounds
    ineq_constr='all'
    V0_bounds=np.array([0.8*MES.V0_sol,1*MES.V0_sol])
    q0c_bounds=np.array([q_lb_fac*MES.qc0_sol_CHP,q_ub_fac*MES.qc0_sol_CHP])
    q1c_bounds=np.array([q_lb_fac*MES.qc1_sol_CHP,q_ub_fac*MES.qc1_sol_CHP])
    delta0_bounds=np.array([-np.pi,np.pi])
    m01_bounds=np.array([-3*MES.m01_sol,3*MES.m01_sol])
    m0_bounds=np.array([0,5*MES.m0_sink])
    m1_bounds=np.array([0,5*MES.m1_sink])
    p1_bounds=np.array([10,5*MES.ph1_sol])
    Ts0_bounds=np.array([60,140])
    Ts1_bounds=np.array([60,140])
    Tr0_bounds=np.array([10,60])
    Tr1_bounds=np.array([10,60])
    P0c_bounds=np.array([0,3*MES.Pc0_sol])
    P1c_bounds=np.array([0,3*MES.Pc1_sol])
    Q0c_bounds=np.array([-3*MES.Qc0_sol,3*MES.Qc0_sol])
    Q1c_bounds=np.array([-3*MES.Qc1_sol,3*MES.Qc1_sol])
    m0c_bounds=np.array([0,3*MES.mc0_sol])
    m1c_bounds=np.array([0,3*MES.mc1_sol])
    dphi0c_bounds=np.array([0,3*MES.phic0_sol])
    dphi1c_bounds=np.array([0,3*MES.phic1_sol])

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, elec_net_LF, heat_net_LF, xmes_LF, iters_LF, err_vec_LF = MES.run_mes_eh_load_flow(max_iter=10)
        V0_sol = elec_net_LF.nodes[0].get_V()
        q0c_sol = -het_net_LF.nodes[4].half_links[0].get_q()
        q1c_sol = -het_net_LF.nodes[5].half_links[0].get_q()
        P0c_sol, P1c_sol, dphi0c_sol, dphi1c_sol =  xmes_LF[[9,10,15,16]]
        # value of objective functions for LF solution
        f_LF_sol = price_electricity_heat(P0c_sol, P1c_sol, dphi0c_sol, dphi1c_sol,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3)
        x_opf_LF = np.concatenate((np.array([V0_sol,q0c_sol,q1c_sol]),xmes_LF))

    result = dict()
    xmes_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']
    # plots
    fig_f = plt.figure('obj_OPF_integrated_MES_eh_methods_sep_LF_qlb{}_qinit{}_qub{}'.format(q_lb_fac,q_init_fac,q_ub_fac))
    ax_f = fig_f.gca()
    ax_f.set_xlabel('Iteration')
    ax_f.set_ylabel('f')

    fig_LF_error = plt.figure('LF_error_OPF_integrated_MES_eh_methods_sep_LF_qlb{}_qinit{}_qub{}'.format(q_lb_fac,q_init_fac,q_ub_fac))
    ax_LF_error = fig_LF_error.gca()
    ax_LF_error.set_xlabel('Iteration')
    ax_LF_error.set_ylabel(r'$||F||_2$')

    fig_q0c = plt.figure('q_0c_OPF_integrated_MES_eh_methods_sep_LF_qlb{}_qinit{}_qub{}'.format(q_lb_fac,q_init_fac,q_ub_fac))
    ax_q0c = fig_q0c.gca()
    ax_q0c.set_xlabel('Iteration')
    ax_q0c.set_ylabel(r'$q_{0c}$ [kg/s]')

    fig_q1c = plt.figure('q_1c_OPF_integrated_MES_eh_methods_sep_LF_qlb{}_qinit{}_qub{}'.format(q_lb_fac,q_init_fac,q_ub_fac))
    ax_q1c = fig_q1c.gca()
    ax_q1c.set_xlabel('Iteration')
    ax_q1c.set_ylabel(r'$q_{1c}$ [kg/s]')

    fig_P0c = plt.figure('P_0c_OPF_integrated_MES_eh_methods_sep_LF_qlb{}_qinit{}_qub{}'.format(q_lb_fac,q_init_fac,q_ub_fac))
    ax_P0c = fig_P0c.gca()
    ax_P0c.set_xlabel('Iteration')
    ax_P0c.set_ylabel(r'$P_{0c}$ [W]')

    fig_P1c = plt.figure('P_1c_OPF_integrated_MES_eh_methods_sep_LF_qlb{}_qinit{}_qub{}'.format(q_lb_fac,q_init_fac,q_ub_fac))
    ax_P1c = fig_P1c.gca()
    ax_P1c.set_xlabel('Iteration')
    ax_P1c.set_ylabel(r'$P_{1c}$ [W]')

    fig_dphi0c = plt.figure('phi_0c_OPF_integrated_MES_eh_methods_sep_LF_qlb{}_qinit{}_qub{}'.format(q_lb_fac,q_init_fac,q_ub_fac))
    ax_dphi0c = fig_dphi0c.gca()
    ax_dphi0c.set_xlabel('Iteration')
    ax_dphi0c.set_ylabel(r'$\Delta \varphi_{0c}$ [W]')

    fig_dphi1c = plt.figure('phi_1c_OPF_integrated_MES_eh_methods_sep_LF_qlb{}_qinit{}_qub{}'.format(q_lb_fac,q_init_fac,q_ub_fac))
    ax_dphi1c = fig_dphi1c.gca()
    ax_dphi1c.set_xlabel('Iteration')
    ax_dphi1c.set_ylabel(r'$\Delta \varphi_{1c}$ [W]')

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
                    xmes_opt, res, f_vec, u_mat, err_LF_vec, E_mat, execution_time = run_eh_optimal_load_flow_separate_LF(V0_init=V0_init,V0_bounds=V0_bounds,q0c_init=q0c_init,q0c_bounds=q0c_bounds,q1c_init=q1c_init,q1c_bounds=q1c_bounds,delta0_bounds=delta0_bounds,m01_bounds=m01_bounds,m0_bounds=m0_bounds,m1_bounds=m1_bounds,p1_bounds=p1_bounds,Ts0_bounds=Ts0_bounds,Ts1_bounds=Ts1_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,P0c_init=P0c_init,P0c_bounds=P0c_bounds,P1c_init=P1c_init,P1c_bounds=P1c_bounds,Q0c_bounds=Q0c_bounds,Q1c_bounds=Q1c_bounds,m0c_bounds=m0c_bounds,m1c_bounds=m1c_bounds,dphi0c_init=dphi0c_init,dphi0c_bounds=dphi0c_bounds, dphi1c_init=dphi1c_init,dphi1c_bounds=dphi1c_bounds,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,scale_var_params=None,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,formulation=formulation,ineq_constr=ineq_constr,stay_within_bounds=stay_within_bounds,optimization_method=method,fb=fb,approach=approach)
                else:
                    approach_legend = 'an'
                    xmes_opt, res, f_vec, u_mat, F_mat, E_mat, execution_time = run_eh_optimal_load_flow(V0_init=V0_init,V0_bounds=V0_bounds,q0c_init=q0c_init,q0c_bounds=q0c_bounds,q1c_init=q1c_init,q1c_bounds=q1c_bounds,delta0_bounds=delta0_bounds,m01_bounds=m01_bounds,m0_bounds=m0_bounds,m1_bounds=m1_bounds,p1_bounds=p1_bounds,Ts0_bounds=Ts0_bounds,Ts1_bounds=Ts1_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,P0c_init=P0c_init,P0c_bounds=P0c_bounds,P1c_init=P1c_init,P1c_bounds=P1c_bounds,Q0c_bounds=Q0c_bounds,Q1c_bounds=Q1c_bounds,m0c_bounds=m0c_bounds,m1c_bounds=m1c_bounds,dphi0c_init=dphi0c_init,dphi0c_bounds=dphi0c_bounds, dphi1c_init=dphi1c_init,dphi1c_bounds=dphi1c_bounds,max_iter=max_iter,tol=tol,scale_var=scale_var,scale_var_params=None,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,formulation=formulation,ineq_constr=ineq_constr,derivatives=True,stay_within_bounds=stay_within_bounds,optimization_method=method,fb=fb)
                    err_LF_vec = [np.linalg.norm(F_mat[ind,:]) for ind in range(len(f_vec))]
                # save result in dictionaries
                result[method+'_'+bound+'_'+approach] = res
                xmes_res[method+'_'+bound+'_'+approach] = xmes_opt
                max_fev = max(max_fev,len(f_vec))
                # plot results
                ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                ax_LF_error.semilogy(err_LF_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                ax_q0c.plot(u_mat[:,1],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                ax_q1c.plot(u_mat[:,2],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                ax_P0c.plot(E_mat[:,0],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                ax_P1c.plot(E_mat[:,1],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                ax_dphi0c.plot(E_mat[:,2],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                ax_dphi1c.plot(E_mat[:,3],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
    ax_f.plot([0,max_fev],[f_LF_sol,f_LF_sol],':r')
    ax_f.legend(handles=legend_handles)
    ax_LF_error.semilogy([0,max_fev],[tol,tol],':k')
    ax_LF_error.legend(handles=legend_handles)
    ax_q0c.plot([0,max_fev],[q0c_sol,q0c_sol],':r')
    ax_q0c.plot([0,max_fev],[q0c_bounds[0],q0c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_q0c.plot([0,max_fev],[q0c_bounds[1],q0c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_q0c.legend(handles=legend_handles)
    ax_q1c.plot([0,max_fev],[q1c_sol,q1c_sol],':r')
    ax_q1c.plot([0,max_fev],[q1c_bounds[0],q1c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_q1c.plot([0,max_fev],[q1c_bounds[1],q1c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_q1c.legend(handles=legend_handles)
    ax_P0c.plot([0,max_fev],[P0c_sol,P0c_sol],':r')
    ax_P0c.plot([0,max_fev],[P0c_bounds[0],P0c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_P0c.plot([0,max_fev],[P0c_bounds[1],P0c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_P0c.legend(handles=legend_handles)
    ax_P1c.plot([0,max_fev],[P1c_sol,P1c_sol],':r')
    ax_P1c.plot([0,max_fev],[P1c_bounds[0],P1c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_P1c.plot([0,max_fev],[P1c_bounds[1],P1c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_P1c.legend(handles=legend_handles)
    ax_dphi0c.plot([0,max_fev],[dphi0c_sol,dphi0c_sol],':r')
    ax_dphi0c.plot([0,max_fev],[dphi0c_bounds[0],dphi0c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_dphi0c.plot([0,max_fev],[dphi0c_bounds[1],dphi0c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_dphi0c.legend(handles=legend_handles)
    ax_dphi1c.plot([0,max_fev],[dphi1c_sol,dphi1c_sol],':r')
    ax_dphi1c.plot([0,max_fev],[dphi1c_bounds[0],dphi1c_bounds[0]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_dphi1c.plot([0,max_fev],[dphi1c_bounds[1],dphi1c_bounds[1]],ls=linestyles_contraints.get('bounds'),color='k')
    ax_dphi1c.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','MES2N')
        with open(os.path.join(path_to_tables,'optimizer_info_integrated_MES_eh_methods_sep_LF_qlb{}_qinit{}_qub{}.txt'.format(q_lb_fac,q_init_fac,q_ub_fac)), "w") as table:
            for bound in bounds:
                for approach in approaches:
                    if approach == 'eq_constr':
                        approach_label = 'eq. constr.'
                    else:
                        approach_label = approach
                    res_trust = result.get('trust-constr_'+bound+'_'+approach)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+approach)
                    res_ipopt = result.get('ipopt_'+bound+'_'+approach)
                    if approach == 'eq_constr':
                        y_trust = res_trust.x
                        y_slsqp = res_slsqp.x
                        y_ipopt = res_ipopt.x
                    else:
                        y_trust = np.concatenate((res_trust.x,xmes_res.get('trust-constr_'+bound+'_'+approach)))
                        y_slsqp = np.concatenate((res_slsqp.x,xmes_res.get('SLSQP_'+bound+'_'+approach)))
                        y_ipopt = np.concatenate((res_ipopt.x,xmes_res.get('ipopt_'+bound+'_'+approach)))
                    table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(bound,approach_label,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(y_trust,x_opf_LF),error(y_slsqp,x_opf_LF),error(y_ipopt,x_opf_LF)))
                table.write(r'\hline ')

    for bound in bounds:
        for approach in approaches:
            res_trust = result.get('trust-constr_'+bound+'_'+approach)
            res_slsqp = result.get('SLSQP_'+bound+'_'+approach)
            res_ipopt = result.get('ipopt_'+bound+'_'+approach)
            if approach == 'eq_constr':
                y_trust = res_trust.x
                y_slsqp = res_slsqp.x
                y_ipopt = res_ipopt.x
            else:
                y_trust = np.concatenate((res_trust.x,xmes_res.get('trust-constr_'+bound+'_'+approach)))
                y_slsqp = np.concatenate((res_slsqp.x,xmes_res.get('SLSQP_'+bound+'_'+approach)))
                y_ipopt = np.concatenate((res_ipopt.x,xmes_res.get('ipopt_'+bound+'_'+approach)))
            print('\nBounds: {}, approach: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nt-c:{}\nSLSQP:{}\nIPOPT:{}\nError for t-c:{}, SLSQP: {}, IPOPT: {}'.format(bound,approach,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(y_trust,x_opf_LF),error(y_slsqp,x_opf_LF),error(y_ipopt,x_opf_LF)))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','MES2N')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def compare_eh_opf_integrated_LF_scaling_sep_LF(dir_path=None,save_tables=False,save_figs=False,q_ub_fac=1.5,q_lb_fac=1,q_init_fac=1.3):
    """Compare OPF of electricity-heat network with integrated LF, which is substituted, for different optimization methods, and bounds. With scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')
    # solver info
    max_iter = 50
    max_iters_lf = 10
    tol = 1e-6
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

    # parameter values for the objective function
    # costs for active power coupling 0
    a0=0
    b0=.3
    c0=3e-5
    # costs for active power coupling 1
    a1=0
    b1=.2
    c1=2e-5
    # costs for heat power coupling 0
    a2=0#a1
    b2=.04#b1
    c2=4e-4#c1
    # costs for active power coupling 1
    a3=0#a0
    b3=.05#b0
    c3=4.5e-4#c0

    # initial guesses (when used)
    V0_init = .9*MES.V0_sol
    q0c_init=q_init_fac*MES.qc0_sol_CHP
    q1c_init=q_init_fac*MES.qc1_sol_CHP
    P0c_init=1.3*MES.Pc0_sol
    P1c_init=1.5*MES.Pc1_sol
    dphi0c_init=1.5*MES.phic0_sol
    dphi1c_init=0.8*MES.phic1_sol

    # bounds
    ineq_constr='all'
    V0_bounds=np.array([0.8*MES.V0_sol,1*MES.V0_sol])
    q0c_bounds=np.array([q_lb_fac*MES.qc0_sol_CHP,q_ub_fac*MES.qc0_sol_CHP])
    q1c_bounds=np.array([q_lb_fac*MES.qc1_sol_CHP,q_ub_fac*MES.qc1_sol_CHP])
    delta0_bounds=np.array([-np.pi,np.pi])
    m01_bounds=np.array([-3*MES.m01_sol,3*MES.m01_sol])
    m0_bounds=np.array([0,5*MES.m0_sink])
    m1_bounds=np.array([0,5*MES.m1_sink])
    p1_bounds=np.array([10,5*MES.ph1_sol])
    Ts0_bounds=np.array([60,140])
    Ts1_bounds=np.array([60,140])
    Tr0_bounds=np.array([10,60])
    Tr1_bounds=np.array([10,60])
    P0c_bounds=np.array([0,3*MES.Pc0_sol])
    P1c_bounds=np.array([0,3*MES.Pc1_sol])
    Q0c_bounds=np.array([-3*MES.Qc0_sol,3*MES.Qc0_sol])
    Q1c_bounds=np.array([-3*MES.Qc1_sol,3*MES.Qc1_sol])
    m0c_bounds=np.array([0,3*MES.mc0_sol])
    m1c_bounds=np.array([0,3*MES.mc1_sol])
    dphi0c_bounds=np.array([0,3*MES.phic0_sol])
    dphi1c_bounds=np.array([0,3*MES.phic1_sol])

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, elec_net_LF, heat_net_LF, xmes_LF, iters_LF, err_vec_LF = MES.run_mes_eh_load_flow(max_iter=10)
        V0_sol = elec_net_LF.nodes[0].get_V()
        q0c_sol = -het_net_LF.nodes[4].half_links[0].get_q()
        q1c_sol = -het_net_LF.nodes[5].half_links[0].get_q()
        P0c_sol, P1c_sol, dphi0c_sol, dphi1c_sol =  xmes_LF[[9,10,15,16]]
        # value of objective functions for LF solution
        f_LF_sol = price_electricity_heat(P0c_sol, P1c_sol, dphi0c_sol, dphi1c_sol,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3)
        x_opf_LF = np.concatenate((np.array([V0_sol,q0c_sol,q1c_sol]),xmes_LF))

    # base values
    Sbase = 1*MW #[W]
    Vbase = 10/np.sqrt(3)*kV #[V]
    deltabase = 1.
    qbase = .1 #[kg/s]
    water = heat_net_LF.links[0].link_params.get('carrier')
    rho = water.rhon
    g = water.g
    phibase = 1.*MW #[W]
    Tbase = 100.#[C]
    mbase = 1.
    pbase = 100*rho*g
    Egbase = 1*MW #[W]
    scale_var_params={'qbase':qbase,'pbase':pbase,'pgbase':pbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':pbase,'phibase':phibase,'Tbase':Tbase,'Ebase':Egbase}
    fb = 1000.*MW

    # make scaled LF solution
    x_opf_LF = np.concatenate((np.array([V0_sol/scale_var_params.get('Vbase'),q0c_sol/scale_var_params.get('qbase'),q1c_sol/scale_var_params.get('qbase')]),xmes_LF/np.array([scale_var_params.get('deltabase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('pbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('phibase'),scale_var_params.get('phibase')])))

    result = dict()
    xmes_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']
    scaling = ['matrix','per_unit']
    # Optimal Flow
    for scale_var in scaling:
        if scale_var == 'per_unit':
            f_LF_sol = price_electricity_heat(P0c_sol/scale_var_params.get('Sbase'), P1c_sol/scale_var_params.get('Sbase'), dphi0c_sol/scale_var_params.get('phibase'), dphi1c_sol/scale_var_params.get('phibase'),a0=a0/fb,b0=b0/(fb/scale_var_params.get('Sbase')),c0=c0/(fb/scale_var_params.get('Sbase')**2),a1=a1/fb,b1=b1/(fb/scale_var_params.get('Sbase')),c1=c1/(fb/scale_var_params.get('Sbase')**2),a2=a2/fb,b2=b2/(fb/scale_var_params.get('phibase')),c2=c2/(fb/scale_var_params.get('phibase')**2),a3=a3/fb,b3=b3/(fb/scale_var_params.get('phibase')),c3=c3/(fb/scale_var_params.get('phibase')**2))
        else:
            f_LF_sol = price_electricity_heat(P0c_sol, P1c_sol, dphi0c_sol, dphi1c_sol,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3)/fb

        # plots
        fig_f = plt.figure('obj_OPF_integrated_MES_eh_sep_LF_qlb{}_qinit{}_qub{}_{}'.format(q_lb_fac,q_init_fac,q_ub_fac,scale_var))
        ax_f = fig_f.gca()
        ax_f.set_xlabel('Iteration')
        ax_f.set_ylabel('f')

        fig_LF_error = plt.figure('LF_error_OPF_integrated_MES_eh_sep_LF_qlb{}_qinit{}_qub{}_{}'.format(q_lb_fac,q_init_fac,q_ub_fac,scale_var))
        ax_LF_error = fig_LF_error.gca()
        ax_LF_error.set_xlabel('Iteration')
        ax_LF_error.set_ylabel(r'$||F||_2$')

        fig_q0c = plt.figure('q_0c_OPF_integrated_MES_eh_sep_LF_qlb{}_qinit{}_qub{}_{}'.format(q_lb_fac,q_init_fac,q_ub_fac,scale_var))
        ax_q0c = fig_q0c.gca()
        ax_q0c.set_xlabel('Iteration')
        ax_q0c.set_ylabel(r'$q_{0c}$ [kg/s]')

        fig_q1c = plt.figure('q_1c_OPF_integrated_MES_eh_sep_LF_qlb{}_qinit{}_qub{}_{}'.format(q_lb_fac,q_init_fac,q_ub_fac,scale_var))
        ax_q1c = fig_q1c.gca()
        ax_q1c.set_xlabel('Iteration')
        ax_q1c.set_ylabel(r'$q_{1c}$ [kg/s]')

        fig_P0c = plt.figure('P_0c_OPF_integrated_MES_eh_sep_LF_qlb{}_qinit{}_qub{}_{}'.format(q_lb_fac,q_init_fac,q_ub_fac,scale_var))
        ax_P0c = fig_P0c.gca()
        ax_P0c.set_xlabel('Iteration')
        ax_P0c.set_ylabel(r'$P_{0c}$ [W]')

        fig_P1c = plt.figure('P_1c_OPF_integrated_MES_eh_sep_LF_qlb{}_qinit{}_qub{}_{}'.format(q_lb_fac,q_init_fac,q_ub_fac,scale_var))
        ax_P1c = fig_P1c.gca()
        ax_P1c.set_xlabel('Iteration')
        ax_P1c.set_ylabel(r'$P_{1c}$ [W]')

        fig_dphi0c = plt.figure('phi_0c_OPF_integrated_MES_eh_sep_LF_qlb{}_qinit{}_qub{}_{}'.format(q_lb_fac,q_init_fac,q_ub_fac,scale_var))
        ax_dphi0c = fig_dphi0c.gca()
        ax_dphi0c.set_xlabel('Iteration')
        ax_dphi0c.set_ylabel(r'$\Delta \varphi_{0c}$ [W]')

        fig_dphi1c = plt.figure('phi_1c_OPF_integrated_MES_eh_sep_LF_qlb{}_qinit{}_qub{}_{}'.format(q_lb_fac,q_init_fac,q_ub_fac,scale_var))
        ax_dphi1c = fig_dphi1c.gca()
        ax_dphi1c.set_xlabel('Iteration')
        ax_dphi1c.set_ylabel(r'$\Delta \varphi_{1c}$ [W]')

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
                        xmes_opt, res, f_vec, u_mat, err_LF_vec, E_mat, execution_time = run_eh_optimal_load_flow_separate_LF(V0_init=V0_init,V0_bounds=V0_bounds,q0c_init=q0c_init,q0c_bounds=q0c_bounds,q1c_init=q1c_init,q1c_bounds=q1c_bounds,delta0_bounds=delta0_bounds,m01_bounds=m01_bounds,m0_bounds=m0_bounds,m1_bounds=m1_bounds,p1_bounds=p1_bounds,Ts0_bounds=Ts0_bounds,Ts1_bounds=Ts1_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,P0c_init=P0c_init,P0c_bounds=P0c_bounds,P1c_init=P1c_init,P1c_bounds=P1c_bounds,Q0c_bounds=Q0c_bounds,Q1c_bounds=Q1c_bounds,m0c_bounds=m0c_bounds,m1c_bounds=m1c_bounds,dphi0c_init=dphi0c_init,dphi0c_bounds=dphi0c_bounds, dphi1c_init=dphi1c_init,dphi1c_bounds=dphi1c_bounds,max_iter=max_iter,max_iters_lf=max_iters_lf,tol=tol,scale_var=scale_var,scale_var_params=scale_var_params,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,formulation=formulation,ineq_constr=ineq_constr,stay_within_bounds=stay_within_bounds,optimization_method=method,fb=fb,approach=approach)
                    else:
                        approach_legend = 'an'
                        xmes_opt, res, f_vec, u_mat, F_mat, E_mat, execution_time = run_eh_optimal_load_flow(V0_init=V0_init,V0_bounds=V0_bounds,q0c_init=q0c_init,q0c_bounds=q0c_bounds,q1c_init=q1c_init,q1c_bounds=q1c_bounds,delta0_bounds=delta0_bounds,m01_bounds=m01_bounds,m0_bounds=m0_bounds,m1_bounds=m1_bounds,p1_bounds=p1_bounds,Ts0_bounds=Ts0_bounds,Ts1_bounds=Ts1_bounds,Tr0_bounds=Tr0_bounds,Tr1_bounds=Tr1_bounds,P0c_init=P0c_init,P0c_bounds=P0c_bounds,P1c_init=P1c_init,P1c_bounds=P1c_bounds,Q0c_bounds=Q0c_bounds,Q1c_bounds=Q1c_bounds,m0c_bounds=m0c_bounds,m1c_bounds=m1c_bounds,dphi0c_init=dphi0c_init,dphi0c_bounds=dphi0c_bounds, dphi1c_init=dphi1c_init,dphi1c_bounds=dphi1c_bounds,max_iter=max_iter,tol=tol,scale_var=scale_var,scale_var_params=scale_var_params,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2,a3=a3,b3=b3,c3=c3,formulation=formulation,ineq_constr=ineq_constr,derivatives=True,stay_within_bounds=stay_within_bounds,optimization_method=method,fb=fb)
                        err_LF_vec = [np.linalg.norm(F_mat[ind,:]) for ind in range(len(f_vec))]
                    if scale_var == 'matrix':
                        xmes_opt = xmes_opt/np.array([scale_var_params.get('deltabase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('pbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Tbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('phibase'),scale_var_params.get('phibase')])
                    # save result in dictionaries
                    result[method+'_'+bound+'_'+approach+'_'+scale_var] = res
                    xmes_res[method+'_'+bound+'_'+approach+'_'+scale_var] = xmes_opt
                    max_fev = max(max_fev,len(f_vec))
                    # plot results
                    ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_LF_error.semilogy(err_LF_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_q0c.plot(u_mat[:,1],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_q1c.plot(u_mat[:,2],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_P0c.plot(E_mat[:,0],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_P1c.plot(E_mat[:,1],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_dphi0c.plot(E_mat[:,2],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_dphi1c.plot(E_mat[:,3],color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
        ax_f.plot([0,max_fev],[f_LF_sol,f_LF_sol],':r')
        ax_f.legend(handles=legend_handles)
        ax_LF_error.semilogy([0,max_fev],[tol,tol],':k')
        ax_LF_error.legend(handles=legend_handles)
        ax_q0c.plot([0,max_fev],[q0c_sol/scale_var_params.get('qbase'),q0c_sol/scale_var_params.get('qbase')],':r')
        ax_q0c.plot([0,max_fev],[q0c_bounds[0]/scale_var_params.get('qbase'),q0c_bounds[0]/scale_var_params.get('qbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q0c.plot([0,max_fev],[q0c_bounds[1]/scale_var_params.get('qbase'),q0c_bounds[1]/scale_var_params.get('qbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q0c.legend(handles=legend_handles)
        ax_q1c.plot([0,max_fev],[q1c_sol/scale_var_params.get('qbase'),q1c_sol/scale_var_params.get('qbase')],':r')
        ax_q1c.plot([0,max_fev],[q1c_bounds[0]/scale_var_params.get('qbase'),q1c_bounds[0]/scale_var_params.get('qbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q1c.plot([0,max_fev],[q1c_bounds[1]/scale_var_params.get('qbase'),q1c_bounds[1]/scale_var_params.get('qbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_q1c.legend(handles=legend_handles)
        ax_P0c.plot([0,max_fev],[P0c_sol/scale_var_params.get('Sbase'),P0c_sol/scale_var_params.get('Sbase')],':r')
        ax_P0c.plot([0,max_fev],[P0c_bounds[0]/scale_var_params.get('Sbase'),P0c_bounds[0]/scale_var_params.get('Sbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P0c.plot([0,max_fev],[P0c_bounds[1]/scale_var_params.get('Sbase'),P0c_bounds[1]/scale_var_params.get('Sbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P0c.legend(handles=legend_handles)
        ax_P1c.plot([0,max_fev],[P1c_sol/scale_var_params.get('Sbase'),P1c_sol/scale_var_params.get('Sbase')],':r')
        ax_P1c.plot([0,max_fev],[P1c_bounds[0]/scale_var_params.get('Sbase'),P1c_bounds[0]/scale_var_params.get('Sbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P1c.plot([0,max_fev],[P1c_bounds[1]/scale_var_params.get('Sbase'),P1c_bounds[1]/scale_var_params.get('Sbase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_P1c.legend(handles=legend_handles)
        ax_dphi0c.plot([0,max_fev],[dphi0c_sol/scale_var_params.get('phibase'),dphi0c_sol/scale_var_params.get('phibase')],':r')
        ax_dphi0c.plot([0,max_fev],[dphi0c_bounds[0]/scale_var_params.get('phibase'),dphi0c_bounds[0]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0c.plot([0,max_fev],[dphi0c_bounds[1]/scale_var_params.get('phibase'),dphi0c_bounds[1]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi0c.legend(handles=legend_handles)
        ax_dphi1c.plot([0,max_fev],[dphi1c_sol/scale_var_params.get('phibase'),dphi1c_sol/scale_var_params.get('phibase')],':r')
        ax_dphi1c.plot([0,max_fev],[dphi1c_bounds[0]/scale_var_params.get('phibase'),dphi1c_bounds[0]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi1c.plot([0,max_fev],[dphi1c_bounds[1]/scale_var_params.get('phibase'),dphi1c_bounds[1]/scale_var_params.get('phibase')],ls=linestyles_contraints.get('bounds'),color='k')
        ax_dphi1c.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','MES2N')
        with open(os.path.join(path_to_tables,'optimizer_info_integrated_MES_eh_scaling_sep_LF_qlb{}_qinit{}_qub{}.txt'.format(q_lb_fac,q_init_fac,q_ub_fac)), "w") as table:
            for bound in bounds:
                for approach in approaches:
                    if approach == 'eq_constr':
                        approach_label = 'eq. constr.'
                    else:
                        approach_label = approach
                    res_trust_mat = result.get('trust-constr_'+bound+'_'+approach+'_matrix')
                    res_slsqp_mat = result.get('SLSQP_'+bound+'_'+approach+'_matrix')
                    res_ipopt_mat = result.get('ipopt_'+bound+'_'+approach+'_matrix')
                    xmes_opt_trust_mat = xmes_res.get('trust-constr_'+bound+'_'+approach+'_matrix')
                    xmes_opt_slsqp_mat = xmes_res.get('SLSQP_'+bound+'_'+approach+'_matrix')
                    xmes_opt_ipopt_mat = xmes_res.get('ipopt_'+bound+'_'+approach+'_matrix')
                    res_trust_pu = result.get('trust-constr_'+bound+'_'+approach+'_per_unit')
                    res_slsqp_pu = result.get('SLSQP_'+bound+'_'+approach+'_per_unit')
                    res_ipopt_pu = result.get('ipopt_'+bound+'_'+approach+'_per_unit')
                    xmes_opt_trust_pu = xmes_res.get('trust-constr_'+bound+'_'+approach+'_per_unit')
                    xmes_opt_slsqp_pu = xmes_res.get('SLSQP_'+bound+'_'+approach+'_per_unit')
                    xmes_opt_ipopt_pu = xmes_res.get('ipopt_'+bound+'_'+approach+'_per_unit')
                    if approach == 'eq_constr':
                        y_trust_mat = res_trust_mat.x
                        y_slsqp_mat = res_slsqp_mat.x
                        y_ipopt_mat = res_ipopt_mat.x
                        y_trust_pu = res_trust_pu.x
                        y_slsqp_pu = res_slsqp_pu.x
                        y_ipopt_pu = res_ipopt_pu.x
                    else:
                        y_trust_mat = np.concatenate((res_trust_mat.x,xmes_opt_trust_mat))
                        y_slsqp_mat = np.concatenate((res_slsqp_mat.x,xmes_opt_slsqp_mat))
                        y_ipopt_mat = np.concatenate((res_ipopt_mat.x,xmes_opt_ipopt_mat))
                        y_trust_pu = np.concatenate((res_trust_pu.x,xmes_opt_trust_pu))
                        y_slsqp_pu = np.concatenate((res_slsqp_pu.x,xmes_opt_slsqp_pu))
                        y_ipopt_pu = np.concatenate((res_ipopt_pu.x,xmes_opt_ipopt_pu))
                    table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e}\\ '.format(bound,approach_label,res_trust_mat.success,res_trust_pu.success,res_slsqp_mat.success,res_slsqp_pu.success,res_ipopt_mat.success,res_ipopt_pu.success,res_trust_mat.nit,res_trust_pu.nit,res_slsqp_pu.nit,res_slsqp_mat.nit,res_ipopt_mat.nit,res_ipopt_pu.nit,error(y_trust_mat,x_opf_LF),error(y_trust_pu,x_opf_LF),error(y_slsqp_mat,x_opf_LF),error(y_slsqp_pu,x_opf_LF),error(y_ipopt_mat,x_opf_LF),error(y_ipopt_pu,x_opf_LF)))
                table.write(r'\hline ')

    for scale_var in scaling:
        for bound in bounds:
            for approach in approaches:
                res_trust = result.get('trust-constr_'+bound+'_'+approach+'_'+scale_var)
                res_slsqp = result.get('SLSQP_'+bound+'_'+approach+'_'+scale_var)
                res_ipopt = result.get('ipopt_'+bound+'_'+approach+'_'+scale_var)
                if approach == 'eq_constr':
                    y_trust = res_trust.x
                    y_slsqp = res_slsqp.x
                    y_ipopt = res_ipopt.x
                else:
                    y_trust = np.concatenate((res_trust.x,xmes_res.get('trust-constr_'+bound+'_'+approach+'_'+scale_var)))
                    y_slsqp = np.concatenate((res_slsqp.x,xmes_res.get('SLSQP_'+bound+'_'+approach+'_'+scale_var)))
                    y_ipopt = np.concatenate((res_ipopt.x,xmes_res.get('ipopt_'+bound+'_'+approach+'_'+scale_var)))
                print('\nScaling: {}, bounds: {}, approach: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nt-c:{}\nSLSQP:{}\nIPOPT:{}\nError for t-c:{}, SLSQP: {}, IPOPT: {}'.format(scale_var,bound,approach,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(y_trust,x_opf_LF),error(y_slsqp,x_opf_LF),error(y_ipopt,x_opf_LF)))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','MES2N')
        for fig_num in plt.get_figlabels():
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

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "No single-carrier subnetworks found",UserWarning)
        # compare_ge_opf_integrated_LF(dir_path,number_runs=1,save_figs=False,save_tables=False)
        # compare_ge_opf_DD_LF(dir_path,number_runs=1,save_figs=False,save_tables=False)
        # compare_ge_opf_integrated_LF_methods(dir_path=dir_path,save_tables=False,save_figs=False)
        # compare_ge_opf_integrated_LF_scaling(dir_path=dir_path,save_tables=False,save_figs=False)
        # compare_ge_opf_integrated_LF_methods_sep_LF(dir_path=dir_path,save_tables=False,save_figs=True)
        # compare_ge_opf_integrated_LF_scaling_sep_LF(dir_path=dir_path,save_tables=False,save_figs=False)
        # compare_eh_opf_integrated_LF_methods(dir_path=dir_path,save_tables=False,save_figs=False)
        # compare_eh_opf_integrated_LF_scaling(dir_path=dir_path,save_tables=False,save_figs=False)
        # compare_eh_opf_integrated_LF_methods_sep_LF(dir_path=dir_path,save_tables=False,save_figs=False,q_ub_fac=1.05,q_lb_fac=.95,q_init_fac=1.01)
        # compare_eh_opf_integrated_LF_methods_sep_LF(dir_path=dir_path,save_tables=False,save_figs=False,q_ub_fac=1.5,q_lb_fac=1,q_init_fac=1.3)
        # compare_eh_opf_integrated_LF_scaling_sep_LF(dir_path=dir_path,save_tables=False,save_figs=False,q_ub_fac=1.05,q_lb_fac=.95,q_init_fac=1.01)
        compare_eh_opf_integrated_LF_scaling_sep_LF(dir_path=dir_path,save_tables=False,save_figs=False,q_ub_fac=1.5,q_lb_fac=1,q_init_fac=1.3)


    plt.show()
