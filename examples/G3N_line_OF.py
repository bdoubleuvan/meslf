"""Optimal load flow of a gas network with 3 nodes connected in a single line"""
from examples import G3N_line as GN
from examples import MES3N_streets as MES
from meslf.networks.gas_network import GasNode, GasLink, GasHalfLink
import warnings
import numpy as np
from meslf.utils.constants import mbar, bar, hour, MBTU, BTU, km, cm
from meslf.utils.hide_print import HiddenPrints
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from meslf.load_flow.system_of_equations import NonLinearSystemGas
import scipy.optimize as spo
import scipy.sparse as sps
import os
import sys
import time
import ipopt

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

colors_method = {'trust-constr':'tab:blue','SLSQP':'tab:orange','ipopt':'tab:green'}
linestyles_approaches = {'eq_constr':'-','direct':':','adjoint':'-.'}
markers_forms = {'full_fa':'.','full_fb':'*','nodal_slack':'d','nodal_ineq':'s','nodal':'x'}
marker_size = 10
legend_handles = [Line2D([0], [0], color=colors_method.get('trust-constr'), label='trust-constr'),
    Line2D([0], [0], color=colors_method.get('SLSQP'), label='SLSQP'),
    Line2D([0], [0], color=colors_method.get('ipopt'), label='IPOPT'),
    Line2D([0], [0], color='w',marker=markers_forms.get('full_fa'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Full, $q(\Delta p)$'),
    Line2D([0], [0], color='w',marker=markers_forms.get('full_fb'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Full, $\Delta p(q)$'),
    Line2D([0], [0], color='w',marker=markers_forms.get('nodal_slack'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Nodal, $q$ slack'),
    Line2D([0], [0], color='w',marker=markers_forms.get('nodal_ineq'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Nodal, $q$ ineq. constr.'),
    Line2D([0], [0], color='w',marker=markers_forms.get('nodal'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Nodal'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('eq_constr'), label='LF as eq. constr.'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('direct'), label='Direct approach'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('adjoint'), label='Adjoint approach')]
# GHV is used to set 'realistic' values for the parameters in the objective function
GHV_BTU = 40.611 #[MBTU/m^3]
GHV = GHV_BTU * MBTU*BTU * hour/GN.gas.rhon #[J/kg]

def create_network(n=0,m=0,s=0,p1=9*bar,q2=-.5,q3=1,hydr_eq='fb',L12=4*km,L23=5*km,D=10*cm,Lstreets=5*km,Dstreets=10*cm,E=.98,link_type='pipe_high_pres_weymouth'):
    """Create gas network, with or without streets connected to it.

    Parameters
    ----------
    n : int, optional
        Number of loads load per street. Default is 0.
    m : int, optional
        Number of junctions that are connected to two loads, per street. Default is 0.
    s : int, optional
        Number of streets. When s=0, the base case without streets is used. Default is 0.

    Returns
    -------
    gas_net : GasNetwork
        The gas network
    """
    # base case
    gas_net = GN.create_network(p1=p1,q3=q3,hydr_eq=hydr_eq,L12=L12,L23=L23,D=D,E=E,link_type=link_type)

    gas_net = update_bc(gas_net,q2)

    if s > 0:
        gas = gas_net.links[0].link_params.get('carrier')
        gn3 = gas_net.nodes[2]
        gas_streets = MES.create_streets_gas(n,m,s,gas,Lstreets,Dstreets,E,q3)
        for gas_street in gas_streets:
            for node in gas_street.get_nodes():
                gas_net.add_node(node)
                for halflink in node.get_half_links():
                    gas_net.add_half_link(halflink)
            for link in gas_street.get_links():
                link.name += '_'+gas_street.name
                gas_net.add_link(link)
            gas_street_source = gas_street.nodes[0]
            gl = GasLink('gl_'+gas_street_source.name,gn3,gas_street_source,link_type=link_type,link_params={'carrier':gas, 'D':Dstreets, 'L':Lstreets,'E':E})
            gas_net.add_link(gl)
            gas_street_source.node_type = 1 # junction node (load node)
            for hl in gas_street_source.get_half_links():
                hl.q = 0
        gn3.node_type = 1 # junction node (load node)
        for hl in gn3.get_half_links():
            hl.q = 0
    return gas_net

def set_x_LF_init(gas_net,nlsys, nodal_link_slack=True,xLF_init_base=np.array([1,1,8*bar,7*bar]),n=0,m=0,s=0,p_perc_high=.99,p_perc_low=.98):
    """Determine initial guess for LF of the gas network, with or without streets connected to it.

    Parameters
    ----------
    n : int, optional
        Number of loads load per street. Default is 0.
    m : int, optional
        Number of junctions that are connected to two loads, per street. Default is 0.
    s : int, optional
        Number of streets. When s=0, the base case without streets is used. Default is 0.
    """
    if s>0: # the network has additional load / streets connected to it
        N = 2*n-m+1 #number of nodes per streets (+1 because of street source)
        NJ = n-m #number of junctions per street

        if nlsys.gas_formulation == 'full':
            p_init_base = xLF_init_base[2:]
            q_init_base = xLF_init_base[:2]
        else:
            p_init_base = xLF_init_base[:]

        q_load_avg = np.mean([hl.q for node in gas_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.q>0])
        q_load_tot = np.sum([hl.q for node in gas_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.q>0])

        pg_ref = gas_net.nodes[0].p #[bar]
        pg_init = np.zeros(len(nlsys.ind_p))
        pg_init[:2] = p_init_base
        pg_ind = 2
        for num_S in range(s): # gas nodes per street
            # nodes are numbered such that the load come first, and then the junctions
            pg_init_street = pg_ref*np.linspace(p_perc_high,p_perc_low,N)
            pg_init[num_S*N+pg_ind] = pg_init_street[0] #street source
            pg_init[num_S*N+pg_ind+1:num_S*N+n+pg_ind+1] = pg_init_street[NJ+1:N] #loads (lowest pressures)
            pg_init[num_S*N+n+pg_ind+1:num_S*N+N+pg_ind] = pg_init_street[1:NJ+1] #junctions (highest pressures)
        if nlsys.gas_formulation == 'full':
            q_init = np.concatenate((q_init_base,q_load_avg*np.ones(len(nlsys.ind_q)-2)))
            xg_init = np.concatenate((q_init,pg_init))
        elif nlsys.gas_formulation == 'nodal' and nodal_link_slack:
            for ind_l, link in enumerate(gas_net.get_links()):
                if ind_l < 2:
                    link.q = q_load_tot
                else:
                    link.q = q_load_avg
            xg_init = pg_init

        else:
            xg_init = pg_init
    else:
        xg_init = xLF_init_base
    gas_net.update(xg_init,formulation=nlsys.gas_formulation)
    gas_net.set_x_init(formulation=nlsys.gas_formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    return xg_init

def xg_from_yopf(y,nlsys=None):
    """Returns the variables of steady-state LF, from the variables y of OF."""
    xg = y[-len(nlsys.x_entries):]
    return xg

def objective_function(y,y_ind,a=np.array([0,0]),b=np.array([.01*GHV,.02*GHV]),c=np.array([1e-6*(GHV)**2,2e-6*(GHV)**2]),scale_var=None,scale_var_params=None,fb=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    E : np array
        Array with flows used in objective. Gas flows are assumed to be in kg/s, active powers in W, and heat power in W. Scaled when per unit scaling is used, unscaled otherwise.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [q1,q2]. Scaled when per unit scaling is used, unscaled otherwise.
    scale_var : str, optional
        Which scaling is used. Default is None.
    scale_var_params : dict, optional
        Dictionary with base values. Only used if scale_var is not None.
    fb : float, optional
        Base value with which to scale the objective function.

    Returns
    -------
    f : float
        The value of the objective function. Scaled when scaling is used.
    """
    f = np.sum(a+b*np.sign(y[np.array(y_ind)])*y[np.array(y_ind)]+c*y[np.array(y_ind)]**2)
    if scale_var == 'matrix':
        f *= (1/fb)
    return f

def grad_objective(y,y_ind,a=np.array([0,0]),b=np.array([.01*GHV,.02*GHV]),c=np.array([1e-6*(GHV)**2,2e-6*(GHV)**2]),scale_var=None,scale_var_params=None,fb=None,Dy_inv=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [q1,q2]. Scaled when per unit scaling is used, unscaled otherwise.
    scale_var : str, optional
        Which scaling is used. Default is None.
    scale_var_params : dict, optional
        Dictionary with base values. Only used if scale_var is not None.
    fb : float, optional
        Base value with which to scale the objective function.
    Dy_inv : np array, optional
        Inverse of base values with which the vector of variables y of OF is scaled.

    Returns
    -------
    df_dy : float
        Gradient of the objective function. Scaled when scaling is used.
    """
    df_dy = np.zeros(len(y))
    df_dy[np.array(y_ind)] = b*np.sign(y[np.array(y_ind)]) + 2*c*y[np.array(y_ind)]
    if scale_var == 'matrix':
        df_dy = (1/fb)*(df_dy.dot(Dy_inv))
    return df_dy

def hess_objective(y,y_ind,a=np.array([0,0]),b=np.array([.01*GHV,.02*GHV]),c=np.array([1e-6*(GHV)**2,2e-6*(GHV)**2]),scale_var=None,scale_var_params=None,fb=None,Dy_inv=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [q1,q2]. Scaled when per unit scaling is used, unscaled otherwise.
    scale_var : str, optional
        Which scaling is used. Default is None.
    scale_var_params : dict, optional
        Dictionary with base values. Only used if scale_var is not None.
    fb : float, optional
        Base value with which to scale the objective function.
    Dy_inv : np array, optional
        Inverse of base values with which the vector of variables y of OF is scaled.

    Returns
    -------
    hess : float
        Hessian of the objective function. Scaled when scaling is used.
    """
    hess_cost_diag = np.zeros(len(y))
    hess_cost_diag[np.array(y_ind)] = 2*c
    hess = np.diag(hess_cost_diag)
    if scale_var == 'matrix':
        hess = (1/fb)*(np.transpose(Dy_inv).dot(hess.dot(Dy_inv)))
    return hess

def h(y, nlsys=None, Dh=None, nodal_link_slack=True):
    """Equality constraints h(x)=0. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeterogeneous
        Nonlinear system of the heterogenous network. Constains the networks, information about scaling, etc.

    Returns
    -------
    h : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    len_u = 1
    network = nlsys.gasnetwork
    E = len(network.links)
    if nlsys.gas_formulation == 'nodal' and nodal_link_slack:
        len_slack = 3 + E
    else:
        len_slack = 1
    xg = xg_from_yopf(y,nlsys=nlsys)
    # reseting network for a gas net is not needed, only once to set / create a half link at slack node
    if len(network.nodes[0].half_links) == 0:
        network.reset_network(xg,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,formulation=nlsys.gas_formulation)
    # evaluate load flow equations
    F = nlsys.F(xg)
    # evaluate conservation of mass in gas slack node
    q1 = y[len_u]
    if nlsys.scale_var == 'per_unit':
        network.nodes[0].half_links[0].q = q1*nlsys.scale_var_params.get('qbase')
    else:
        network.nodes[0].half_links[0].q = q1
    cons_mass1 = network.nodes[0].node_law(network=network,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    if nlsys.gas_formulation == 'nodal' and nodal_link_slack:
        # evaluate link equations
        link_eqs = np.zeros(E)
        for ind_link,link in enumerate(network.get_links()):
            ind_qk = len_u + 1 + ind_link # index of qk wihtin y
            qk = y[ind_qk]
            if nlsys.scale_var == 'per_unit':
                qk = qk*nlsys.scale_var_params.get('qbase')
            link.q = qk
            link_eqs[ind_link] = link.link_equation(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        G = np.concatenate((np.array([cons_mass1]),link_eqs))
    else:
        G = np.array([cons_mass1])
    h = np.concatenate((G,F)) # already scaled if per unit is used
    if nlsys.scale_var == 'matrix':
        h = Dh.dot(h)
    return h

def h_der(y, nlsys=None, Dy_inv=None, Dh=None, nodal_link_slack=True):
    """First derivative of quality constraints h(x)=0. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeterogeneous
        Nonlinear system of the heterogenous network. Constains the networks, information about scaling, etc.

    Returns
    -------
    dh_dy : np array
        The first derivative of the (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    len_u = 1
    network = nlsys.gasnetwork
    E = len(network.links)
    if nlsys.gas_formulation == 'nodal' and nodal_link_slack:
        len_slack = 1 + E
    else:
        len_slack = 1
    xg = xg_from_yopf(y,nlsys=nlsys)
    # reseting network for a gas net is not needed
    # network.reset_network(xg,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,formulation=nlsys.gas_formulation)
    # Jacobian of LF equations
    J_full = nlsys.J_dense(xg,return_full=True)
    N = len(network.nodes)
    dh_dy = np.zeros((len(xg)+len_slack,len(y)))
    F_ind = nlsys.ind_Fn + [N+ind for ind in nlsys.ind_Fl]
    if nlsys.gas_formulation=='nodal':
        xlf_ind = nlsys.ind_p
    else:
        xlf_ind = nlsys.ind_q + [E+ind for ind in nlsys.ind_p]
    dh_dy[len_slack:,:][:,-len(xg):] = J_full[F_ind,:][:,xlf_ind] #J_lf
    # derivative to half link flows
    dh_dy[0,len_u] = -1 #dfq1_dq1
    dh_dy[len_slack,0] = -1 #dfq2_dq2
    # derivative of slack conservation of mass to load flow variables
    G_ind = [0]
    dh_dy[:len(G_ind),:][:,-len(xg):] = J_full[G_ind,:][:,xlf_ind] #dG_dxlf
    # derivatives of slack link equations to y
    if nlsys.gas_formulation == 'nodal' and nodal_link_slack:
        for ind_link,link in enumerate(network.get_links()):
            qk = link.flow() # unscaled
            link.q = qk
            dfk_dpi, dfk_dpj = link.f_der_p(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) # scaled when per unit scaling is used, unscaled otherwise
            ind_qk = len_u + 1 + ind_link # index of qk wihtin y
            if link.start_node.number in nlsys.ind_p: # node 1 is reference, so is not part of ind_p
                ind_pi = len_u + len_slack + nlsys.ind_p.index(link.start_node.number) # index of pi wihtin y
                dh_dy[1+ind_link,ind_pi] = dfk_dpi
            ind_pj = len_u + len_slack + nlsys.ind_p.index(link.end_node.number)
            if ind_link == 0:
                dh_dy[0,ind_pj] = dfk_dpj #dG0_dp2
            dh_dy[1+ind_link,ind_qk] = 1 #dfk_dqk
            dh_dy[1+ind_link,ind_pj] = dfk_dpj
    if nlsys.scale_var == 'matrix':
        dh_dy = Dh.dot(dh_dy.dot(Dy_inv))
    return dh_dy

def gamma(y, nlsys=None, nodal_link_slack=True, ineq_constr_lb=np.array([-2,-2]), ineq_constr_ub=np.array([2,2])):
    """The nonlinear inequality constraints gamma(x)>=0 on the link flows for nodal formulation. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeterogeneous
        Nonlinear system of the heterogenous network. Constains the networks, information about scaling, etc.
    ineq_constr_lb, ineq_constr_ub : np arrays
        Lower and upper bound for the link flows. Scaled when per unit or matrix scaling is used.

    Returns
    -------
    gam : np array
        The nonlinear inequality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    if nlsys.gas_formulation == 'full' or nodal_link_slack:
        raise ValueError('This function should only be called for nodal formulation and nodal_link_slack=False')
    len_u = 1
    len_slack = 1
    xg = xg_from_yopf(y,nlsys=nlsys)
    network = nlsys.gasnetwork
    # reseting network for a gas net is not needed
    # network.reset_network(xg,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,formulation=nlsys.gas_formulation)
    network.update(xg,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,formulation=nlsys.gas_formulation)
    link_flows = list()
    for link in network.get_links():
        qk = link.flow(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) # scaled when per unit scaling is used, unscaled otherwise
        if nlsys.scale_var == 'matrix': # lower and upper bounds are scaled
            qk = qk/nlsys.scale_var_params.get('qbase')
        link_flows.append(qk)
    q = np.array(link_flows)
    return np.concatenate((q-ineq_constr_lb,ineq_constr_ub-q))

def gamma_der(y, nlsys=None, Dy_inv=None, nodal_link_slack=True):
    """The nonlinear inequality constraints gamma(x)>=0 on the link flows for nodal formulation. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeterogeneous
        Nonlinear system of the heterogenous network. Constains the networks, information about scaling, etc.

    Returns
    -------
    gam : np array
        The nonlinear inequality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    if nlsys.gas_formulation == 'full' or nodal_link_slack:
        raise ValueError('This function should only be called for nodal formulation and nodal_link_slack=False')
    len_u = 1
    len_slack = 1
    xg = xg_from_yopf(y,nlsys=nlsys)
    network = nlsys.gasnetwork
    # reseting network for a gas net is not needed
    # network.reset_network(xg,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,formulation=nlsys.gas_formulation)
    network.update(xg,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,formulation=nlsys.gas_formulation)
    len_E =len(network.links)
    dgamma_dy = np.zeros((2*len_E,len(y)))
    dq_dx = np.zeros((len_E,len(xg)))
    for ind_link,link in enumerate(network.get_links()):
        qk = link.flow() # unscaled
        link.q = qk
        dqk_dpi, dqk_dpj = -np.array(link.f_der_p(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)) # scaled when per unit scaling is used, unscaled otherwise
        if link.start_node.number in nlsys.ind_p: # node 1 is reference, so is not part of ind_p
            ind_pi = nlsys.ind_p.index(link.start_node.number)
            dq_dx[ind_link,ind_pi] = dqk_dpi
        ind_pj = nlsys.ind_p.index(link.end_node.number)
        dq_dx[ind_link,ind_pj] = dqk_dpj
    dgamma_dy[:,-len(xg):] = np.vstack((np.eye(len_E),-np.eye(len_E))).dot(dq_dx)
    if nlsys.scale_var == 'matrix':
        dgamma_dy = np.diag(1/nlsys.scale_var_params.get('qbase')*np.ones(2*len_E)).dot(dgamma_dy.dot(Dy_inv))
    return dgamma_dy

def update_bc(gas_net,q2,scale_var=None,scale_var_params=None):
    """Update the boundary conditions of the gasn etwork, based on the control variables of OF"""
    if scale_var == 'per_unit':
        q2 = q2*scale_var_params.get('qbase')
    gas_net.nodes[1].half_links[0].q = q2
    return gas_net

def run_optimal_load_flow(u_lb=np.array([-2]),u_ub=np.array([0]),u_init=np.array([-1]),slack_lb_base=np.array([-2]),slack_ub_base=np.array([0]),slack_init_base=np.array([-1]),xLF_lb_base=np.array([-2,-2,1*mbar,1*mbar]),xLF_ub_base=np.array([2,2,10*bar,10*bar]),xLF_init_base=np.array([1,1,8*bar,7*bar]),p_perc_high=.99,p_perc_low=.98,n=0,m=0,s=0,p1=9*bar,q3=1,hydr_eq='fb',L12=4*km,L23=5*km,D=10*cm,Lstreets=5*km,Dstreets=10*cm,E=.98,link_type='pipe_high_pres_weymouth',a=np.array([0,0]),b=np.array([.01*GHV,.02*GHV]),c=np.array([1e-6*(GHV)**2,2e-6*(GHV)**2]),ineq_constr_lb_base=np.array([-2,-2]),ineq_constr_ub_base=np.array([2,2]),scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,formulation='full',nodal_link_slack=True,ineq_constr='all',optimization_method='trust-constr',stay_within_bounds=False,fb=None):
    """Run optimal load flow, with LF as equality constraints.

    Parameters
    ----------
    nodal_link_slack : bool, optional
        Determines if the link flows are taken as slack variables when using nodal formulation. Only used when ineq_constr is 'all'. Default is True.
    ineq_contrs : str, optional
        Determines on which variables the bounds (or inequality constraints) are imposed. Options are 'nodal' or 'all'. For 'nodal', bounds are imposed on nodal (and injected) variables only, so not on the link flows. Default is 'all'.

    Returns
    -------
    xg_opt : np array
        Solution of LF variables of OF. Unscaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_time : float
        Time of the optimization (excluding creation of network etc.)
    """
    print('\nRunning OPF for G3N line (formulation: {}, inequality constraints on: {}, nodal link flows as slack: {}, hard bounds: {}, method: {}, scaling: {})'.format(formulation,ineq_constr, nodal_link_slack,stay_within_bounds,optimization_method,scale_var))

    # create network
    if formulation == 'nodal' and hydr_eq == 'fb':
        raise ValueError("Nodal formulation is not compatible with link equations that express pressure drop as function of link flow. Use 'fa' for hydr_eq or 'full' for formulation instead.")

    # gas_net = GN.create_network(p1=p1,q3=q3,hydr_eq=hydr_eq,L12=L12,L23=L23,D=D,E=E,link_type=link_type)
    q2 = u_init
    gas_net= create_network(n=n,m=m,s=s,p1=p1,q2=q2,q3=q3,hydr_eq=hydr_eq,L12=L12,L23=L23,D=D,Lstreets=Lstreets,Dstreets=Dstreets,E=E,link_type=link_type)


    # set initial guess and initialize network
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    xLF_init = set_x_LF_init(gas_net,nlsys,xLF_init_base=xLF_init_base,n=n,m=m,s=s,p_perc_high=p_perc_high,p_perc_low=p_perc_low)
    if s>0 and formulation == 'nodal' and ineq_constr == 'all' and nodal_link_slack:
        slack_init = [slack_init_base[0]]
        for link in gas_net.get_links():
            slack_init.append(link.get_q()) #unscaled
        slack_init = np.array(slack_init)
    else:
        slack_init = slack_init_base

    # initial guess for OF (unscaled)
    y0 = np.concatenate((u_init,slack_init,xLF_init))

    # indices within y of values to be used in objective function
    y_ind = [1,0] # objective is defined with a,b, and c assuming [q1,q2] as input

    # adjust values of bounds to match size of network when there are streets
    len_E = len(gas_net.links)
    if s>0:
        len_xp = len(nlsys.ind_p)
        pLF_lb = np.zeros(len_xp)
        pLF_lb[:2] = xLF_lb_base[-2:]
        pLF_lb[2:] = xLF_lb_base[-1]
        pLF_ub = np.zeros(len_xp)
        pLF_ub[:2] = xLF_ub_base[-2:]
        pLF_ub[2:] = xLF_ub_base[-1]
        if formulation == 'full':
            len_xq = len(nlsys.ind_q)
            qLF_lb = np.zeros(len_xq)
            qLF_lb[:2] = xLF_lb_base[:2]
            qLF_lb[2:] = xLF_lb_base[1]
            qLF_ub = np.zeros(len_xq)
            qLF_ub[:2] = xLF_ub_base[:2]
            qLF_ub[2:] = xLF_ub_base[1]
            xLF_lb = np.concatenate((qLF_lb,pLF_lb))
            xLF_ub = np.concatenate((qLF_ub,pLF_ub))
            slack_lb = slack_lb_base
            slack_ub = slack_ub_base
        elif formulation == 'nodal' and ineq_constr == 'all' and (not nodal_link_slack):
            xLF_lb = pLF_lb
            xLF_ub = pLF_ub
            ineq_constr_lb = np.zeros(len_E)
            ineq_constr_lb[:2] = ineq_constr_lb_base
            ineq_constr_lb[2:] = ineq_constr_lb_base[-1]
            ineq_constr_ub = np.zeros(len_E)
            ineq_constr_ub[:2] = ineq_constr_ub_base
            ineq_constr_ub[2:] = ineq_constr_ub_base[-1]
            slack_lb = slack_lb_base
            slack_ub = slack_ub_base
        elif formulation == 'nodal' and ineq_constr == 'all' and nodal_link_slack:
            xLF_lb = pLF_lb
            xLF_ub = pLF_ub
            slack_lb = np.zeros(1+len_E)
            slack_lb[:3] = slack_lb_base[:]
            slack_lb[3:] = slack_lb_base[-1]
            slack_ub = np.zeros(1+len_E)
            slack_ub[:3] = slack_ub_base[:]
            slack_ub[3:] = slack_ub_base[-1]
        else:
            xLF_lb = pLF_lb
            xLF_ub = pLF_ub
            slack_lb = slack_lb_base
            slack_ub = slack_ub_base
    else:
        slack_lb = slack_lb_base
        slack_ub = slack_ub_base
        xLF_lb = xLF_lb_base
        xLF_ub = xLF_ub_base
        if formulation == 'nodal' and ineq_constr == 'all' and (not nodal_link_slack):
            ineq_constr_lb = ineq_constr_lb_base
            ineq_constr_ub = ineq_constr_ub_base

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('qbase')])
        if formulation == 'nodal' and ineq_constr == 'all' and nodal_link_slack:
            slack_base = scale_var_params.get('qbase')*np.ones(1+len_E)
            G_base = scale_var_params.get('qbase')*np.ones(1+len_E)
        else:
            slack_base = np.array([scale_var_params.get('qbase')])
            G_base = np.array([scale_var_params.get('qbase')])
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((ubase,slack_base,1/Dx.data[0])))
        Dh = np.diag(np.concatenate((1/G_base,DF.data[0])))
        y0 = Dy.dot(y0) # scale y
    else:
        Dy=np.eye(len(y0))
        Dy_inv=np.eye(len(y0))
        Dh=np.eye(len(slack_init)+len(xLF_init))

    if scale_var == 'per_unit':
        a = a/fb
        b = b/(fb/np.diag(Dy_inv)[np.array(y_ind)])
        c = c/(fb/np.diag(Dy_inv)[np.array(y_ind)]**2)

    # define objective function
    def obj(y,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb):
        global y_f_vec
        y_f_vec = y.copy()
        global f_vec_global
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
            y = Dy_inv.dot(y)
        f = objective_function(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    def obj_grad(y,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb, formulation=formulation,Dy_inv=Dy_inv):
        if nlsys.scale_var == 'matrix':
            y = Dy_inv.dot(y)
        return grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb,Dy_inv=Dy_inv)

    def obj_hess(y,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb, formulation=formulation):
        if nlsys.scale_var == 'matrix':
            y = Dy_inv.dot(y)
        return hess_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb,Dy_inv=Dy_inv)

    # define nonlinear equality constraints (load flow equations)
    def eq_constr(y,nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh, nodal_link_slack=nodal_link_slack):
        network = nlsys.gasnetwork
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        q2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network = update_bc(network, q2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        H = h(y, nlsys=nlsys, Dh=Dh, nodal_link_slack=nodal_link_slack)
        global err_LF_vec_global
        F = H[len(u_init)+len(slack_init):]
        err_LF_vec_global.append(np.linalg.norm(F))
        return H

    def jac_eq_constr(y,nlsys=nlsys, Dy_inv=Dy_inv, Dh=Dh, nodal_link_slack=nodal_link_slack):
        network = nlsys.gasnetwork
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        q2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network = update_bc(network, q2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dh_dy = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh, nodal_link_slack=nodal_link_slack)
        return dh_dy

    lb_nleq = np.zeros(len(slack_lb)+len(xLF_lb))
    ub_nleq = np.zeros(len(slack_ub)+len(xLF_ub))
    if optimization_method == 'trust-constr':
        nonlinear_constraint = spo.NonlinearConstraint(eq_constr,lb_nleq,ub_nleq,jac=jac_eq_constr,keep_feasible=stay_within_bounds)
    elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        nonlinear_constraint = {'type':'eq','fun':eq_constr,'jac':jac_eq_constr}

    # define nonlinear inequality constraints on link flows (only for nodal formulation)
    if formulation == 'nodal' and ineq_constr == 'all' and (not nodal_link_slack):
        def g(y,nlsys=nlsys, nodal_link_slack=nodal_link_slack, ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub, Dy_inv=Dy_inv):
            network = nlsys.gasnetwork
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                y = Dy_inv.dot(y)
            # update bc of the network
            q2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
            network = update_bc(network, q2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            gam = gamma(y,nlsys=nlsys, nodal_link_slack=nodal_link_slack, ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub)
            return gam
        def g_jac(y,nlsys=nlsys, nodal_link_slack=nodal_link_slack, Dy_inv=Dy_inv):
            network = nlsys.gasnetwork
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                y = Dy_inv.dot(y)
            # update bc of the network
            q2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
            network = update_bc(network, q2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            dgamma_dy = gamma_der(y, nlsys=nlsys, nodal_link_slack=nodal_link_slack, Dy_inv=Dy_inv)
            return dgamma_dy
        if optimization_method == 'trust-constr':
            ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(2*len(ineq_constr_lb)),np.inf*np.ones(2*len(ineq_constr_ub)),jac=g_jac,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            ineq_constr_fun  = {'type':'ineq','fun':g,'jac':g_jac}
    else:
        ineq_constr_fun = None

    # define linear inequality constraints, i.e. define bounds
    lb_ineq = np.concatenate((u_lb,slack_lb,xLF_lb))
    ub_ineq = np.concatenate((u_ub,slack_ub,xLF_ub))
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        lb_ineq = Dy.dot(lb_ineq)
        ub_ineq = Dy.dot(ub_ineq)
    if ineq_constr == 'nodal' and formulation == 'full': # do not impose bounds on link flows
        lb_ineq[-len(nlsys.x_entries):-len(nlsys.ind_p)] = -np.inf*np.ones(len_E) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
        ub_ineq[-len(nlsys.x_entries):-len(nlsys.ind_p)] = np.inf*np.ones(len_E)

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

    # make sure initial guess satisfies bounds when hard bounds are used.
    if optimization_method == 'SLSQP' or stay_within_bounds:
        for ind, x0 in enumerate(y0):
            if lb_ineq[ind] > x0:
                y0[ind] = lb_ineq[ind]
            elif ub_ineq[ind] < x0:
                y0[ind] = ub_ineq[ind]

    global f_vec_global
    global y_f_vec
    global err_LF_vec_global
    f_vec_global = list()
    y_f_vec = y0.copy()
    err_LF_vec_global = list()
    if optimization_method == 'trust-constr':
        f_vec = list()
        def callback(xk, state):
            f_vec.append(state.fun)
            return False
    elif optimization_method == 'SLSQP':
        f_vec = [obj(y0)] # this call to obj() alters all the global variables.
        def callback(xk):
            f_vec.append(obj(xk))
            return False

    # solve OPF
    opf_start_time = time.perf_counter()
    try:
        if ineq_constr_fun != None:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, y0, method=optimization_method,jac=obj_grad,hess=obj_hess, constraints=[nonlinear_constraint, ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter,'factorization_method':'SVDFactorization'}, bounds=bounds,tol=tol, callback=callback)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, y0, method=optimization_method,jac=obj_grad, constraints=[nonlinear_constraint,ineq_constr_fun], options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                execution_time = time.perf_counter() - opf_start_time
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, y0,jac=obj_grad, constraints=[nonlinear_constraint,ineq_constr_fun], options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = time.perf_counter() - opf_start_time
        else:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, y0, method=optimization_method,jac=obj_grad,hess=obj_hess, constraints=nonlinear_constraint, options={'verbose': 1,'maxiter':max_iter,'factorization_method':'SVDFactorization'}, bounds=bounds,tol=tol, callback=callback) #specify factorization method to avoid weird error about matrix not being square. Another options seems to set the bounds of the equality constraint not exactly equal. See https://stackoverflow.com/questions/61753007/how-to-solve-the-problem-of-the-valueerror-expected-square-matrix-in-a-constra
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, y0, method=optimization_method,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                execution_time = time.perf_counter() - opf_start_time
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, y0,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = time.perf_counter() - opf_start_time
    except:
        print('Exception made for {}, ineq. constr: {}, hard bounds: {}, nodal link flow slack: {}, scaling: {}'.format(optimization_method,ineq_constr,stay_within_bounds,nodal_link_slack,scale_var))
        if len(f_vec_global) == 0:
            obj(y0)
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
        execution_time = time.perf_counter() - opf_start_time
        res = spo.OptimizeResult({'success':False,'x':np.array(y_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

    if optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        res.execution_time = execution_time
    if optimization_method == 'ipopt':
        if res.nit > 0:
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            f_vec = f_vec_global

    if len(err_LF_vec_global) >= len(f_vec):
        indices = [int(round(ind)) for ind in np.linspace(0,len(err_LF_vec_global)-1,len(f_vec))]
        err_LF_vec = [err_LF_vec_global[ind] for ind in indices]
    elif len(err_LF_vec_global) == 0:
        eq_constr(y0)
        err_LF_vec = [err_LF_vec_global[-1]]
    else:
        err_LF_vec = err_LF_vec_global

    if scale_var == 'matrix' or scale_var == 'per_unit':
        y_opf = Dy_inv.dot(res.x)
    else:
        y_opf = res.x

    # print solution
    q2 = y_opf[:len(u_init)] # unscaled
    gas_net = update_bc(gas_net,q2)
    xg_opt = xg_from_yopf(y_opf,nlsys=nlsys) #unscaled
    print('Solution OF (success: {})'.format(res.success))
    if len(xg_opt) < 10:
        # reseting network for a gas net is not needed
        # gas_net.reset_network(xg_opt,formulation=formulation)
        p_vec,q_vec,q_inj = gas_net.update_full(xg_opt,formulation=formulation)
        q1 = y_opf[len(u_init)] # unscaled
        gas_net.nodes[0].half_links[0].q = q1
        q_inj[0] = q1 # if converged, this should be the same value as calculated by the network. If not converged, the value of q1 in y might be different from that calculated by the network.
        if formulation == 'nodal' and nodal_link_slack:
            link_flows = list()
            for ind_link,link in enumerate(gas_net.get_links()):
                ind_qk = len(u_init) + 1 + ind_link # index of qk wihtin y
                qk = y_opf[ind_qk]
                link_flows.append(qk)
                link.q = qk
            q_vec = np.array(link_flows)
        print('p = {} bar'.format(p_vec/bar))
        print('q = {} kg/s'.format(q_vec))
        print('q nodal inj = {} kg/s'.format(q_inj))
    return xg_opt, res, f_vec, err_LF_vec, execution_time

def get_u_from_net(gas_net,scale_var=None,scale_var_params=None):
    """Get the current values of the control variables u from the network.
    """
    q2 = gas_net.nodes[1].half_links[0].get_q(scale_var=scale_var,scale_var_params=scale_var_params)
    return np.array([q2])

def solve_lf_in_of(u,nlsys,max_iters=10,tol=1e-6,xLF_init=np.array([1,1,8*bar,7*bar]),nodal_link_slack=True):
    """Solve steady-state LF within an optimization context.
    """
    global err_LF_vec_global
    network = nlsys.gasnetwork
    u_net = get_u_from_net(network,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    if len(err_LF_vec_global)==0 or (np.linalg.norm(u-u_net) > tol) or (err_LF_vec_global[-1] > tol):
        q2 = u[0]
        network = update_bc(network, q2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        # reseting network for a gas net is not needed
        # xg0 = network.set_x_init(formulation=nlsys.gas_formulation, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        # network.reset_network(xg0,formulation=nlsys.gas_formulation, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        with HiddenPrints():
            xg,iters,err_vec,p_vec,q_vec,q_inj = network.solve_network(tol,max_iters,solver='NR',formulation=nlsys.gas_formulation, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        if err_vec[-1] >= tol:
            # reseting network for a gas net is not needed
            # network.reset_network(xLF_init,formulation=nlsys.gas_formulation, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            network.update(xLF_init,formulation=nlsys.gas_formulation, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            with HiddenPrints():
                xg,iters,err_vec,p_vec,q_vec,q_inj = network.solve_network(tol,max_iters,solver='NR',formulation=nlsys.gas_formulation, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        err_LF = err_vec[-1]
        q1 = q_inj[0]
        if nlsys.scale_var == 'per_unit':
            q1 = q1/nlsys.scale_var_params.get('qbase')
    else:
        xg = network.set_x_init(formulation=nlsys.gas_formulation, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        q1 = network.nodes[0].half_links[0].get_q(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        err_LF = err_LF_vec_global[-1]
    err_LF_vec_global.append(err_LF)
    if nlsys.gas_formulation == 'nodal' and nodal_link_slack:
        slack = np.zeros(1+len(network.links))
        slack[0] = q1
        for ind_link,link in enumerate(network.get_links()):
            qk = link.flow(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) # unscaled
            slack[1+ind_link] = link.get_q(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    else:
        slack = np.array([q1])
    x = np.concatenate((slack,xg))
    return x

def run_optimal_load_flow_separate_LF(u_lb=np.array([-2]),u_ub=np.array([0]),u_init=np.array([-1]),slack_lb_base=np.array([-2]),slack_ub_base=np.array([0]),slack_init_base=np.array([-1]),xLF_lb_base=np.array([-2,-2,1*mbar,1*mbar]),xLF_ub_base=np.array([2,2,10*bar,10*bar]),xLF_init_base=np.array([1,1,8*bar,7*bar]),p_perc_high=.99,p_perc_low=.98,n=0,m=0,s=0,p1=9*bar,q3=1,hydr_eq='fb',L12=4*km,L23=5*km,D=10*cm,Lstreets=5*km,Dstreets=10*cm,E=.98,link_type='pipe_high_pres_weymouth',a=np.array([0,0]),b=np.array([.01*GHV,.02*GHV]),c=np.array([1e-6*(GHV)**2,2e-6*(GHV)**2]),ineq_constr_lb_base=np.array([-2,-2]),ineq_constr_ub_base=np.array([2,2]),scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,max_iters_lf=10,formulation='full',nodal_link_slack=True,ineq_constr='all',approach='direct',optimization_method='trust-constr',stay_within_bounds=False,fb=None):
    """Run optimal load flow, with LF included implicitly.

    Parameters
    ----------
    nodal_link_slack : bool, optional
        Determines if the link flows are taken as slack variables when using nodal formulation. Only used when ineq_constr is 'all'. Default is True.
    ineq_contrs : str, optional
        Determines on which variables the bounds (or inequality constraints) are imposed. Options are 'nodal' or 'all'. For 'nodal', bounds are imposed on nodal (and injected) variables only, so not on the link flows. Default is 'all'.

    Returns
    -------
    xg_opt : np array
        Solution of LF variables of OF. Unscaled.
    y_opt : np array
        Full solution of OF. That is, control variables and all state variables. Scaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_time : float
        Time of the optimization (excluding creation of network etc.)
    """
    print('\nRunning OPF for G3N line with separate LF (formulation: {}, hydr. eq: {}, inequality constraints on: {}, nodal link flows as slack: {}, hard bounds: {}, method: {}, scaling: {}, approach: {})'.format(formulation,hydr_eq,ineq_constr, nodal_link_slack,stay_within_bounds,optimization_method,scale_var,approach))

    # create network
    if formulation == 'nodal' and hydr_eq == 'fb':
        raise ValueError("Nodal formulation is not compatible with link equations that express pressure drop as function of link flow. Use 'fa' for hydr_eq or 'full' for formulation instead.")
    q2 = u_init
    gas_net= create_network(n=n,m=m,s=s,p1=p1,q2=q2,q3=q3,hydr_eq=hydr_eq,L12=L12,L23=L23,D=D,Lstreets=Lstreets,Dstreets=Dstreets,E=E,link_type=link_type)

    # set initial guess and initialize network
    gas_net.initialize()
    nlsys = NonLinearSystemGas(gas_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    xLF_init = set_x_LF_init(gas_net,nlsys,xLF_init_base=xLF_init_base,n=n,m=m,s=s,p_perc_high=p_perc_high,p_perc_low=p_perc_low)
    if s>0 and formulation == 'nodal' and ineq_constr == 'all' and nodal_link_slack:
        slack_init = [slack_init_base[0]]
        for link in gas_net.get_links():
            slack_init.append(link.get_q()) #unscaled
        slack_init = np.array(slack_init)
    else:
        slack_init = slack_init_base

    # initial guess for OF (unscaled)
    u0 = u_init

    # indices within y of values to be used in objective function
    y_ind = [1,0] # objective is defined with a,b, and c assuming [q1,q2] as input

    # adjust values of bounds to match size of network when there are streets
    len_E = len(gas_net.links)
    ineq_constr_lb = None
    ineq_constr_ub = None
    if s>0:
        len_xp = len(nlsys.ind_p)
        pLF_lb = np.zeros(len_xp)
        pLF_lb[:2] = xLF_lb_base[-2:]
        pLF_lb[2:] = xLF_lb_base[-1]
        pLF_ub = np.zeros(len_xp)
        pLF_ub[:2] = xLF_ub_base[-2:]
        pLF_ub[2:] = xLF_ub_base[-1]
        if formulation == 'full':
            len_xq = len(nlsys.ind_q)
            qLF_lb = np.zeros(len_xq)
            qLF_lb[:2] = xLF_lb_base[:2]
            qLF_lb[2:] = xLF_lb_base[1]
            qLF_ub = np.zeros(len_xq)
            qLF_ub[:2] = xLF_ub_base[:2]
            qLF_ub[2:] = xLF_ub_base[1]
            xLF_lb = np.concatenate((qLF_lb,pLF_lb))
            xLF_ub = np.concatenate((qLF_ub,pLF_ub))
            slack_lb = slack_lb_base
            slack_ub = slack_ub_base
        elif formulation == 'nodal' and ineq_constr == 'all' and (not nodal_link_slack):
            xLF_lb = pLF_lb
            xLF_ub = pLF_ub
            ineq_constr_lb = np.zeros(len_E)
            ineq_constr_lb[:2] = ineq_constr_lb_base
            ineq_constr_lb[2:] = ineq_constr_lb_base[-1]
            ineq_constr_ub = np.zeros(len_E)
            ineq_constr_ub[:2] = ineq_constr_ub_base
            ineq_constr_ub[2:] = ineq_constr_ub_base[-1]
            slack_lb = slack_lb_base
            slack_ub = slack_ub_base
        elif formulation == 'nodal' and ineq_constr == 'all' and nodal_link_slack:
            xLF_lb = pLF_lb
            xLF_ub = pLF_ub
            slack_lb = np.zeros(1+len_E)
            slack_lb[:3] = slack_lb_base[:]
            slack_lb[3:] = slack_lb_base[-1]
            slack_ub = np.zeros(1+len_E)
            slack_ub[:3] = slack_ub_base[:]
            slack_ub[3:] = slack_ub_base[-1]
        else:
            xLF_lb = pLF_lb
            xLF_ub = pLF_ub
            slack_lb = slack_lb_base
            slack_ub = slack_ub_base
    else:
        slack_lb = slack_lb_base
        slack_ub = slack_ub_base
        xLF_lb = xLF_lb_base
        xLF_ub = xLF_ub_base
        if formulation == 'nodal' and ineq_constr == 'all' and (not nodal_link_slack):
            ineq_constr_lb = ineq_constr_lb_base
            ineq_constr_ub = ineq_constr_ub_base

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('qbase')])
        if formulation == 'nodal' and ineq_constr == 'all' and nodal_link_slack:
            slack_base = scale_var_params.get('qbase')*np.ones(1+len_E)
            G_base = scale_var_params.get('qbase')*np.ones(1+len_E)
        else:
            slack_base = np.array([scale_var_params.get('qbase')])
            G_base = np.array([scale_var_params.get('qbase')])
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((ubase,slack_base,1/Dx.data[0])))
        Du = np.diag(1/ubase)
        Du_inv = np.diag(ubase)
        Dh = np.diag(np.concatenate((1/G_base,DF.data[0])))
        u0 = Du.dot(u0) # scale y
    else:
        Dy=np.eye(len(u0)+len(slack_init)+len(xLF_init))
        Dy_inv=np.eye(len(u0)+len(slack_init)+len(xLF_init))
        Du = np.eye(len(u0))
        Du_inv= np.eye(len(u0))
        Dh=np.eye(len(slack_init)+len(xLF_init))

    if scale_var == 'per_unit':
        a = a/fb
        b = b/(fb/np.diag(Dy_inv)[np.array(y_ind)])
        c = c/(fb/np.diag(Dy_inv)[np.array(y_ind)]**2)

    # limits on inequality constraints on state variables
    lb_ineq_state = np.concatenate((slack_lb,xLF_lb))
    ub_ineq_state = np.concatenate((slack_ub,xLF_ub))
    if scale_var == 'matrix' or scale_var == 'per_unit':
        lb_ineq_state = Dy[len(u_lb):,len(u_lb):].dot(lb_ineq_state)
        ub_ineq_state = Dy[len(u_ub):,len(u_ub):].dot(ub_ineq_state)
        u_lb = Du.dot(u_lb)
        u_ub = Du.dot(u_ub)
    if ineq_constr == 'nodal' and formulation == 'full': # do not impose bounds on link flows
        lb_ineq_state = np.concatenate((np.array([lb_ineq_state[0]]),lb_ineq_state[-len(nlsys.ind_p):])) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
        ub_ineq_state = np.concatenate((np.array([ub_ineq_state[0]]),ub_ineq_state[-len(nlsys.ind_p):]))

    # define objective function
    def obj(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init,nodal_link_slack=nodal_link_slack):
        global u_f_vec
        u_f_vec = u.copy()
        global f_vec_global
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,nodal_link_slack=nodal_link_slack)
        y = np.concatenate((u,x))
        f = objective_function(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    def obj_grad(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb, Dy_inv=Dy_inv, Dh=Dh, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init,nodal_link_slack=nodal_link_slack,method=approach):
        if nlsys.scale_var == 'matrix':
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,nodal_link_slack=nodal_link_slack)
        y = np.concatenate((u,x))
        # partial derivatives of objective
        deltaf_deltay = grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb,Dy_inv=Dy_inv)
        deltaf_deltau = np.zeros((1,len(u)))
        deltaf_deltax = np.zeros((1,len(x)))
        deltaf_deltau[0,:] = deltaf_deltay[:len(u)]
        deltaf_deltax[0,:] = deltaf_deltay[len(u):]
        # partial derivatives of equatilty constraints / load-flow equations
        q2 = y[:len(u)] # scaled for p.u., unscaled for matrix
        network = nlsys.gasnetwork
        network = update_bc(network, q2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        deltah_deltay = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh, nodal_link_slack=nodal_link_slack)
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
    def g(u,nlsys=nlsys, nodal_link_slack=nodal_link_slack, ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub, Dy=Dy, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init):
        if nlsys.scale_var == 'matrix':
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,nodal_link_slack=nodal_link_slack)
        y = np.concatenate((u,x))
        if scale_var == 'matrix': # lb_ineq_state and ub_ineq_state are scaled, so scale x as well
            x = Dy[len(u):,len(u):].dot(x)
        if ineq_constr == 'nodal' and formulation == 'full':
            x = np.concatenate((np.array([x[0]]),x[-len(nlsys.ind_p):]))
        g = np.concatenate((x-lb_ineq_state,ub_ineq_state-x))
        if formulation == 'nodal' and ineq_constr == 'all' and (not nodal_link_slack):
            q2 = y[:len(u)] # scaled for p.u., unscaled for matrix
            network = nlsys.gasnetwork
            network = update_bc(network, q2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            gam = gamma(y,nlsys=nlsys, nodal_link_slack=nodal_link_slack, ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub)
            g = np.concatenate((gam,g))
        return g
    def g_jac(u,nlsys=nlsys, nodal_link_slack=nodal_link_slack, Dy_inv=Dy_inv, Dh=Dh, Dy=Dy, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init,method=approach):
        if nlsys.scale_var == 'matrix':
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,nodal_link_slack=nodal_link_slack)
        y = np.concatenate((u,x))
        # Jacobian of inequality constraints on state variables

        if ineq_constr == 'nodal' and nlsys.gas_formulation == 'full':
            len_state = len(nlsys.ind_p) + 1
            I = np.zeros((len_state,len(x)))
            I[0,0] = 1 # slack var.
            for ind in range(len(nlsys.ind_p)):
                I[1+ind,1+len(nlsys.ind_q)+ind] = 1
        else:
            len_state = len(x)
            I = np.eye(len(x))
        deltag_deltax = np.vstack((I,-I))
        deltag_deltau = np.zeros((2*len_state,len(u)))
        # partial derivatives of equatilty constraints / load-flow equations
        q2 = y[:len(u)] # scaled for p.u., unscaled for matrix
        network = nlsys.gasnetwork
        network = update_bc(network, q2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        deltah_deltay = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh, nodal_link_slack=nodal_link_slack)
        deltah_deltau = deltah_deltay[:,:len(u)]
        deltah_deltax = deltah_deltay[:,len(u):]
        if formulation == 'nodal' and ineq_constr == 'all' and (not nodal_link_slack):
            deltagamma_deltay = gamma_der(y, nlsys=nlsys, nodal_link_slack=nodal_link_slack, Dy_inv=Dy_inv)
            deltagamma_deltau = deltagamma_deltay[:,:len(u)]
            deltagamma_deltax = deltagamma_deltay[:,len(u):]
            deltag_deltau = np.vstack((deltagamma_deltau,deltag_deltau))
            deltag_deltax = np.vstack((deltagamma_deltax,deltag_deltax))
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
        len_g = 2*len(lb_ineq_state)
        if formulation == 'nodal' and ineq_constr == 'all' and (not nodal_link_slack):
            len_g += 2*len(ineq_constr_lb)
        ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(len_g),np.inf*np.ones(len_g),jac=g_jac,keep_feasible=stay_within_bounds)
    elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        ineq_constr_fun  = {'type':'ineq','fun':g,'jac':g_jac}

    # define linear inequality constraints (bounds) on the control variables
    if optimization_method == 'ipopt':
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
    if optimization_method == 'ipopt' and ineq_constr != None:
        bounds = [(lb,ub) for lb, ub in zip(u_lb,u_ub)]
    elif ineq_constr != None:
        bounds = spo.Bounds(u_lb,u_ub,keep_feasible=stay_within_bounds)
    else:
        bounds = None

    # make sure initial guess satisfies bounds
    if ineq_constr != None and (optimization_method == 'SLSQP' or stay_within_bounds):
        for ind, x0 in enumerate(u0):
            if u_lb[ind] > x0:
                u0[ind] = u_lb[ind]
            elif u_ub[ind] < x0:
                u0[ind] = u_ub[ind]

    global f_vec_global
    global u_f_vec
    global err_LF_vec_global
    f_vec_global = list()
    u_f_vec = u0.copy()
    err_LF_vec_global = list()
    if optimization_method == 'trust-constr':
        f_vec = list()
        def callback(xk, state):
            f_vec.append(state.fun)
            return False
    elif optimization_method == 'SLSQP':
        f_vec = [obj(u0)] # this call to obj() alters all the global variables.
        def callback(xk):
            f_vec.append(obj(xk))
            return False

    # solve OPF
    opf_start_time = time.perf_counter()
    try:
        if optimization_method == 'trust-constr':
            res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=[ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter,'factorization_method':'SVDFactorization'}, bounds=bounds, callback=callback)
            execution_time = res.execution_time
        elif optimization_method == 'SLSQP':
            res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
            execution_time = time.perf_counter() - opf_start_time
        elif optimization_method == 'ipopt':
            res = ipopt.minimize_ipopt(obj, u0,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
            execution_time = time.perf_counter() - opf_start_time
    except:
        print('Exception made for {} (formulation: {}, inequality constraints on: {}, nodal link flows as slack: {}, hard bounds: {}, scaling: {}, approach: {})'.format(optimization_method,formulation,ineq_constr, nodal_link_slack,stay_within_bounds,scale_var,approach))
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
        execution_time = time.perf_counter() - opf_start_time
        res = spo.OptimizeResult({'success':False,'x':np.array(u_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

    if optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        res.execution_time = execution_time

    if optimization_method == 'ipopt':
        if res.nit > 0:
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            f_vec = f_vec_global

    if len(f_vec_global) >= len(f_vec):
        indices = [int(round(ind)) for ind in np.linspace(0,len(f_vec_global)-1,len(f_vec))]
        err_LF_vec = [err_LF_vec_global[ind] for ind in indices]
    else:
        obj(u0)
        f_vec = [f_vec_global[-1]]
        err_LF_vec = [err_LF_vec_global[-1]]

    if scale_var == 'matrix' or scale_var == 'per_unit':
        u_opf = Du_inv.dot(res.x)
    else:
        u_opf = res.x
    # print solution
    x = solve_lf_in_of(res.x,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,nodal_link_slack=nodal_link_slack) #scaled, when using per unit, unscaled otherwise
    if scale_var == 'per_unit':
        y_opt = Dy_inv.dot(np.concatenate((res.x,x))) # unscaled
    else:
        y_opt = np.concatenate((u_opf,x)) # unscaled
    xg_opt = xg_from_yopf(y_opt,nlsys=nlsys) # unscaled
    print('Solution OF (success: {})'.format(res.success))
    if len(xg_opt) < 10:
        # reseting network for a gas net is not needed
        # gas_net.reset_network(xg_opt,formulation=formulation)
        p_vec,q_vec,q_inj = gas_net.update_full(xg_opt,formulation=formulation)
        q1 = y_opt[len(u_init)] # unscaled
        gas_net.nodes[0].half_links[0].q = q1
        q_inj[0] = q1 # if converged, this should be the same value as calculated by the network. If not converged, the value of q1 in y might be different from that calculated by the network.
        if formulation == 'nodal' and nodal_link_slack:
            link_flows = list()
            for ind_link,link in enumerate(gas_net.get_links()):
                ind_qk = len(u_init) + 1 + ind_link # index of qk wihtin y
                qk = y_opt[ind_qk]
                link_flows.append(qk)
                link.q = qk
            q_vec = np.array(link_flows)
        print('p = {} bar'.format(p_vec/bar))
        print('q = {} kg/s'.format(q_vec))
        print('q nodal inj = {} kg/s'.format(q_inj))
    if scale_var == 'matrix' or scale_var == 'per_unit':
        y_opt = Dy.dot(y_opt) # scaled
    return xg_opt, y_opt, res, f_vec, err_LF_vec, execution_time

def error(x_res,x_sol):
    """Relative error between solution and result.

    Parameters
    ----------
    x_res : np array or float
        Variables result.
    x_sol : np array or float
        Variables solution.
    """
    return np.max(np.abs(x_sol-x_res)/np.abs(x_sol))

def compare_forms(dir_path=None,save_tables=False,save_figs=False,number_of_runs=1,n=0,m=0,s=0,N_max = 100,max_iter=25,scale_var=None):
    """Compare OF for different optimization methods, formulations of LF, formulations of OF, link equations, and bounds."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    max_iters_lf = 10
    tol = 1e-6

    # scaling
    if scale_var == None:
        scale_var_params = None
        fb = None
        qbase = 1.
        scale_label = 'unscaled'
    else:
        pbase = 1*bar
        qbase = 1.
        scale_var_params = {'pbase':pbase,'pgbase':pbase,'qbase':qbase}
        fb = 1e9
        if scale_var == 'matrix':
            scale_label = 'matrix'
        else:
            scale_label = 'pu'

    # parameter values for the two objective functions
    a=np.array([0,0])
    b=np.array([.01*GHV,.02*GHV])
    c=np.array([1e-6*(GHV)**2,2e-6*(GHV)**2])

    # Network parameters and boundary conditions (of LF)
    p1 = 50*bar
    q3 = 1
    q2_opt = (b[1]-b[0]-2*c[0]*q3)/(2*(c[0]+c[1])) # optimal solution
    L12=4*km
    L23=5*km
    D=10*cm
    Lstreets=5*km
    Dstreets=10*cm
    E=.98
    link_type='pipe_high_pres_weymouth'

    # bounds (assuming full formulation)
    u_lb=np.array([-2*q3])
    u_ub=np.array([0])
    q1_lb=np.array([-2*q3])
    q1_ub=np.array([0])
    xLF_lb_full=np.array([-2,-2,1*mbar,1*mbar])
    xLF_ub_full=np.array([2,2,60*bar,60*bar])
    xLF_lb_nodal=xLF_lb_full[2:]
    xLF_ub_nodal=xLF_ub_full[2:]
    q_ineq_constr_lb=xLF_lb_full[:2] # bound on link flows q when included as inequality constraints
    q_ineq_constr_ub=xLF_ub_full[:2]

    # initial guess (assuming full formulation)
    u_init=np.array([-.5*q3])
    q1_init=np.array([-.5*q3])
    p_perc_high=.98
    p_perc_low=.97
    xLF_init_full=np.array([1,1,.99*p1,.985*p1])#np.array([1,1,8*bar,7*bar])
    xLF_init_nodal=xLF_init_full[2:]

    # steady-state LF solution, using the optimal q2 (without scaling)
    with HiddenPrints():
        gas_net_LF = create_network(n=n,m=m,s=s,p1=p1,q2=q2_opt,q3=q3,hydr_eq='fb',L12=L12,L23=L23,D=D,Lstreets=Lstreets,Dstreets=Dstreets,E=E,link_type=link_type)
        gas_net_LF.initialize()
        nlsys_LF = NonLinearSystemGas(gas_net_LF,formulation='full')
        x0_LF =set_x_LF_init(gas_net_LF,nlsys_LF,xLF_init_base=xLF_init_full,n=n,m=m,s=s,p_perc_high=p_perc_high,p_perc_low=p_perc_low)
        xg_LF,iters_LF,err_vec_LF,p_sol,q_sol,q_inj = gas_net_LF.solve_network(tol,max_iter,solver='NR',formulation='full')
    y_LF = np.concatenate((np.array([q2_opt,q_inj[0]]),xg_LF)) # assuming full formulation, unscaled
    if scale_var != None:
        nlsys_LF = NonLinearSystemGas(gas_net_LF,formulation='full',scale_var=scale_var,scale_var_params=scale_var_params)
        Dx = nlsys_LF.Dx()
        ubase = np.array([scale_var_params.get('qbase')])
        slack_base = np.array([scale_var_params.get('qbase')]) # assuming full formulation
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        f_LF = objective_function(y_LF,[1,0],a=a,b=b,c=c)/fb #scaled
        y_LF = Dy.dot(y_LF) # assuming full formulation, scaled
    else:
        f_LF = objective_function(y_LF,[1,0],a=a,b=b,c=c,scale_var=scale_var,fb=fb)

    result = dict()
    y_res = dict()
    N = len(gas_net_LF.nodes)
    if N > N_max:
        bounds = ['hard']
    else:
        bounds = ['soft', 'hard']
    ineq_constrs = ['all','nodal']
    formulations = ['full','nodal']
    if N > N_max:
        link_flows = ['ineq']
    else:
        link_flows = ['ineq','slack'] # If the link variables should be included as slack variables in the nodal approach
    hydr_eqs = ['fa','fb'] # q=q(dp) , dp = dp(q)
    methods = ['trust-constr','SLSQP','ipopt']
    approaches = ['eq_constr','direct','adjoint']
    total_cases_general = len(bounds)*len(methods)*len(approaches)
    if 'full' in formulations:
        total_cases_full = total_cases_general*len(hydr_eqs)*len(ineq_constrs)
    else:
        total_cases_full = 0
    total_cases_nodal = 0
    if 'nodal' in formulations:
        if 'all' in ineq_constrs:
            total_cases_nodal += total_cases_general*len(link_flows)
        if 'nodal' in ineq_constrs:
            total_cases_nodal += total_cases_general

    total_cases = total_cases_full + total_cases_nodal
    case_number = 1
    total_time = 0
    for bound in bounds:
        if bound == 'soft':
            stay_within_bounds = False
        else:
            stay_within_bounds = True
        for ineq_constr in ineq_constrs:
            fig_f = plt.figure('gas_obj_'+bound+'_'+ineq_constr+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
            ax_f = fig_f.gca()
            ax_f.set_xlabel('Iteration?')
            ax_f.set_ylabel('f')

            fig_LF_error = plt.figure('gas_error_LF_in_OF_'+bound+'_'+ineq_constr+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
            ax_LF_error = fig_LF_error.gca()
            ax_LF_error.set_xlabel('Iteration?')
            ax_LF_error.set_ylabel(r'$||F||_2$')

            max_fev = 0
            for form in formulations:
                if form == 'nodal':
                    xLF_lb = xLF_lb_nodal
                    xLF_ub = xLF_ub_nodal
                    xLF_init =xLF_init_nodal
                else:
                    xLF_lb = xLF_lb_full
                    xLF_ub = xLF_ub_full
                    xLF_init =xLF_init_full
                for link_flow in link_flows:
                    if link_flow == 'slack':
                        nodal_link_slack = True
                    else:
                        nodal_link_slack = False
                    if form == 'nodal' and ineq_constr == 'nodal' and nodal_link_slack:
                        continue
                    if form == 'nodal' and ineq_constr == 'all' and nodal_link_slack:
                        slack_lb = np.concatenate((q1_lb,q_ineq_constr_lb))
                        slack_ub = np.concatenate((q1_ub,q_ineq_constr_lb))
                        slack_init =np.concatenate((q1_init,xLF_init_full[:2]))
                    else:
                        slack_lb = q1_lb
                        slack_ub = q1_ub
                        slack_init = q1_init
                    if form == 'full' and nodal_link_slack: #nodal_link_slack has no effect for 'full', so only run OF once.
                        continue
                    for hydr_eq in hydr_eqs:
                        if form == 'nodal' and hydr_eq == 'fb':
                            continue
                        if form == 'full':
                            marker_key = form+'_'+hydr_eq
                        elif form == 'nodal' and ineq_constr == 'nodal':
                            marker_key = form
                        else:
                            marker_key = form+'_'+link_flow
                        for method in methods:
                            for approach in approaches:
                                exec_times = list()
                                for run in range(number_of_runs):
                                    start_time = time.perf_counter()
                                    if approach == 'direct' or approach == 'adjoint':
                                        xg_opt, y_opt, res, f_vec, err_LF_vec, execution_time = run_optimal_load_flow_separate_LF(u_lb=u_lb,u_ub=u_ub,u_init=u_init,slack_lb_base=slack_lb,slack_ub_base=slack_ub,slack_init_base=slack_init,xLF_lb_base=xLF_lb,xLF_ub_base=xLF_ub,xLF_init_base=xLF_init,p_perc_high=p_perc_high,p_perc_low=p_perc_low,n=n,m=m,s=s,p1=p1,q3=q3,hydr_eq=hydr_eq,L12=L12,L23=L23,D=D,Lstreets=Lstreets,Dstreets=Dstreets,E=E,link_type=link_type,a=a,b=b,c=c,ineq_constr_lb_base=q_ineq_constr_lb,ineq_constr_ub_base=q_ineq_constr_ub,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,formulation=form,nodal_link_slack=nodal_link_slack,ineq_constr=ineq_constr,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,approach=approach)
                                    else:
                                        xg_opt, res, f_vec, err_LF_vec, execution_time = run_optimal_load_flow(u_lb=u_lb,u_ub=u_ub,u_init=u_init,slack_lb_base=slack_lb,slack_ub_base=slack_ub,slack_init_base=slack_init,xLF_lb_base=xLF_lb,xLF_ub_base=xLF_ub,xLF_init_base=xLF_init,p_perc_high=p_perc_high,p_perc_low=p_perc_low,n=n,m=m,s=s,p1=p1,q3=q3,hydr_eq=hydr_eq,L12=L12,L23=L23,D=D,Lstreets=Lstreets,Dstreets=Dstreets,E=E,link_type=link_type,a=a,b=b,c=c,ineq_constr_lb_base=q_ineq_constr_lb,ineq_constr_ub_base=q_ineq_constr_ub,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,formulation=form,nodal_link_slack=nodal_link_slack,ineq_constr=ineq_constr,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb)
                                        y_opt = res.x
                                    exec_times.append(execution_time)
                                    end_time = time.perf_counter()
                                    total_time += end_time-start_time
                                    avg_time = total_time/((case_number-1)*number_of_runs+run+1)
                                    end_time = avg_time*(number_of_runs-(run+1)*avg_time + (total_cases-case_number)*number_of_runs)
                                    end_time_hour, rem = divmod(end_time, 3600)
                                    end_time_min, end_time_sec = divmod(rem, 60)
                                    print('Finished run {} of {}, for case {} of {}, in {:.2f}s (N={}). Expected time till end: {:0>2}:{:0>2}:{:05.2f} '.format(run+1,number_of_runs,case_number,total_cases,execution_time,N,int(end_time_hour),int(end_time_min),end_time_sec))
                                case_number+=1
                                res.execution_time = np.mean(execution_time)
                                result[method+'_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach] = res
                                y_res[method+'_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach] = y_opt
                                max_fev = max(max_fev,len(f_vec))
                                # plot results
                                ax_f.semilogy(f_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(marker_key),alpha=.5)
                                ax_LF_error.semilogy(err_LF_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(marker_key),alpha=.5)
            ax_f.semilogy([0,max_fev],[f_LF,f_LF],':r',alpha=.5)
            ax_f.legend(handles=legend_handles)
            ax_LF_error.semilogy([0,max_fev],[tol,tol],':k')
            ax_LF_error.legend(handles=legend_handles)

    if save_figs:
        if s > 0:
            path_to_fig = os.path.join(dir_path,'Figures','MES3N_streets')
        else:
            path_to_fig = os.path.join(dir_path,'Figures','MES3N_line')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

    if save_tables:
        if s > 0:
            path_to_tables = os.path.join(dir_path,'Tables','MES3N_streets')
        else:
            path_to_tables = os.path.join(dir_path,'Tables','MES3N_line')
        for ineq_constr in ineq_constrs:
            for bound in bounds:
                with open(os.path.join(path_to_tables,'gas_optimizer_info_forms_'+bound+'_'+ineq_constr+'_'+scale_label+'_{}_{}_{}.txt'.format(n,m,s)), "w") as table:
                    for form in formulations:
                        for hydr_eq in hydr_eqs:
                            if form == 'nodal' and hydr_eq == 'fb':
                                continue
                            for link_flow in link_flows:
                                if link_flow == 'slack':
                                    nodal_link_slack = True
                                else:
                                    nodal_link_slack = False
                                if form == 'nodal' and ineq_constr == 'nodal' and nodal_link_slack:
                                    continue
                                if form == 'full' and nodal_link_slack: #nodal_link_slack has no effect for 'full', so only run OF once.
                                    continue
                                if form == 'full':
                                    if hydr_eq == 'fa':
                                        link_label = r'$f^{q(\Delta p)}$'
                                    else:
                                        link_label = r'$f^{\Delta p(q)}$'
                                elif link_flow == 'ineq':
                                    link_label = 'ineq.'#'ineq. constr.'
                                else:
                                    link_label = link_flow
                                for approach in approaches:
                                    if approach == 'eq_constr':
                                        approach_label = 'eq. constr.'
                                    else:
                                        approach_label = approach
                                    res_trust = result.get('trust-constr_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                                    res_slsqp = result.get('SLSQP_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                                    res_ipopt = result.get('ipopt_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                                    y_trust = y_res.get('trust-constr_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                                    y_slsqp = y_res.get('SLSQP_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                                    y_ipopt = y_res.get('ipopt_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                                    if len(y_trust) < 1+len(gas_net_LF.links)+len(nlsys_LF.ind_p): # link flows are not part of y
                                        y_LF_sol = y_LF[[0,1]+[2+len(gas_net_LF.links)+ind for ind in range(len(nlsys_LF.ind_p))]]
                                    else:
                                        y_LF_sol = y_LF
                                    table.write(r'{} & {} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e}\\ '.format(form,link_label,approach_label,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,res_trust.execution_time,res_slsqp.execution_time,res_ipopt.execution_time,error(y_trust,y_LF_sol),error(y_slsqp,y_LF_sol),error(y_ipopt,y_LF_sol)))
                                table.write(r'\cline{2-18} ')
                        table.write(r'\hline ')

    for ineq_constr in ineq_constrs:
        for bound in bounds:
            for form in formulations:
                for link_flow in link_flows:
                    if link_flow == 'slack':
                        nodal_link_slack = True
                    else:
                        nodal_link_slack = False
                    if form == 'nodal' and ineq_constr == 'nodal' and nodal_link_slack:
                        continue
                    if form == 'full' and nodal_link_slack: #nodal_link_slack has no effect for 'full', so only run OF once.
                        continue
                    for hydr_eq in hydr_eqs:
                        if form == 'nodal' and hydr_eq == 'fb':
                            continue
                        for approach in approaches:
                            res_trust = result.get('trust-constr_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                            res_slsqp = result.get('SLSQP_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                            res_ipopt = result.get('ipopt_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                            y_trust = y_res.get('trust-constr_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                            y_slsqp = y_res.get('SLSQP_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                            y_ipopt = y_res.get('ipopt_'+form+'_'+hydr_eq+'_'+ineq_constr+'_'+link_flow+'_'+bound+'_'+approach)
                            if len(y_trust) < 1+len(gas_net_LF.links)+len(nlsys_LF.ind_p): # link flows are not part of y
                                y_LF_sol = y_LF[[0,1]+[2+len(gas_net_LF.links)+ind for ind in range(len(nlsys_LF.ind_p))]]
                            else:
                                y_LF_sol = y_LF
                            print('\nLimits: {}, bounds: {}, form: {}, hydr eq: {}, approach: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nt-c:{}\nSLSQP: {}\nIPOPT: {}\nError for t-c:{:.4e}, SLSQP: {:.4e}, IPOPT:{:.4e}\nExec. time for t-c:{:.2f}, SLSQP: {:.2f}, IPOPT:{:.2f}'.format(ineq_constr,bound,form,hydr_eq,approach,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(y_trust,y_LF_sol),error(y_slsqp,y_LF_sol),error(y_ipopt,y_LF_sol),res_trust.execution_time,res_slsqp.execution_time,res_ipopt.execution_time))

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # base case (3 nodes)
    n=0
    m=0
    s=0
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=1,N_max=100,n=n,m=m,s=s,max_iter=25) # unscaled
    compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=1,N_max=100,scale_var='matrix',n=n,m=m,s=s,max_iter=25)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=1,N_max=100,scale_var='per_unit',n=n,m=m,s=s,max_iter=25)

    # medium case (30 nodes)
    # n=5
    # m=2
    # s=3
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=1,N_max=100,n=n,m=m,s=s,max_iter=25) # unscaled
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=1,N_max=100,scale_var='matrix',n=n,m=m,s=s,max_iter=25)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=1,N_max=100,scale_var='per_unit',n=n,m=m,s=s,max_iter=25)

    # medium / large case (163 nodes)
    # n=10
    # m=5
    # s=10
    # runs = 3
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,N_max=100,scale_var='matrix',n=n,m=m,s=s,max_iter=25)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,N_max=100,scale_var='per_unit',n=n,m=m,s=s,max_iter=25)

    # large case (323 nodes)
    # n=10
    # m=5
    # s=20
    # # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=1,N_max=100,scale_var='matrix',n=n,m=m,s=s,max_iter=40)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=1,N_max=100,scale_var='per_unit',n=n,m=m,s=s,max_iter=40)

    plt.show()
