"""Optimal power flow of an electrical network with 3 nodes connected in a single line, with possible streets connected to the third node."""
from examples import E3N_line as EN
from examples import MES3N_streets as MES
from meslf.networks.electrical_network import ElectricalLink
import warnings
import numpy as np
from meslf.utils.hide_print import HiddenPrints
from meslf.utils.constants import kV, MW, km, cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from meslf.load_flow.system_of_equations import NonLinearSystemElectrical
import scipy.optimize as spo
import scipy.sparse as sps
import os
import sys
import time
import ipopt
import datetime

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

colors_method = {'trust-constr':'tab:blue','SLSQP':'tab:orange','ipopt':'tab:green'}
linestyles_approaches = {'eq_constr':'-','direct':':','adjoint':'-.'}
markers_forms = {1:'.',2:'*',3:'d'}
marker_size = 10
legend_handles = [Line2D([0], [0], color=colors_method.get('trust-constr'), label='trust-constr'),
    Line2D([0], [0], color=colors_method.get('SLSQP'), label='SLSQP'),
    Line2D([0], [0], color=colors_method.get('ipopt'), label='IPOPT'),
    Line2D([0], [0], color='w',marker=markers_forms.get(1), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'$|S_{ij}|^2$ unbounded'),
    Line2D([0], [0], color='w',marker=markers_forms.get(2), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'$|S_{ij}|^2$ bounded, not in $x^G$'),
    Line2D([0], [0], color='w',marker=markers_forms.get(3), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'$|S_{ij}|^2$ bounded, in $x^G$'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('eq_constr'), label='LF as eq. constr.'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('direct'), label='Direct approach'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('adjoint'), label='Adjoint approach')]

def create_network(n=0,m=0,s=0,V1=50*kV,delta1=0.,V2=50*kV,P2=-1*MW,P3=1.5*MW,Q3=1.5*MW,L12=4*km,L23=5*km,D=1*cm,link_type='pi_line',Lstreets=5*km,Dstreets=1*cm):
    """Create electrical network, with or without streets connected to it.

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
    elec_net : ElectricalNetwork
        The electrical network
    """
    # base case (with node 2 a generator node)
    elec_net = EN.create_network(V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,node_set=2,L12=L12,L23=L23,D=D,link_type=link_type)

    if s > 0: # connect streets to network
        en3 = elec_net.nodes[2]
        elec_streets = MES.create_streets_elec(n,m,s,Lstreets,Dstreets,P3,Q3)
        for elec_street in elec_streets:
            for node in elec_street.get_nodes():
                elec_net.add_node(node)
                for halflink in node.get_half_links():
                    elec_net.add_half_link(halflink)
            for link in elec_street.get_links():
                link.name += '_'+elec_street.name
                elec_net.add_link(link)
            elec_street_source = elec_street.nodes[0]
            b,g,b_sh = EN.create_line_data(Lstreets,Dstreets)
            if link_type == 'pi_line':
                link_params = {'b':b,'g':g,'b_sh':b_sh,'g_sh':0}
            elif link_type == 'short_line':
                link_params = {'b':b,'g':g,'b_sh':0,'g_sh':0}
            hl = ElectricalLink('el_'+elec_street_source.name,en3,elec_street_source,link_type=link_type,link_params=link_params)
            elec_net.add_link(hl)
            elec_street_source.node_type = 2 # junction node (load node)
            for hl in elec_street_source.get_half_links():
                hl.P = 0
                hl.Q = 0
        en3.node_type = 2 # junction node (load node)
        for hl in en3.get_half_links():
            hl.P = 0
            hl.Q = 0
    return elec_net

def set_x_LF_init(nlsys):
    """Set the unscaled initial guess for the LF variables"""
    elec_net = nlsys.elecnetwork
    delta_init = elec_net.nodes[0].get_delta()*np.ones(len(nlsys.unknown_delta_nodes))
    V_init = elec_net.nodes[0].get_V()*np.ones(len(nlsys.unknown_V_nodes))
    xe_init = np.concatenate((delta_init,V_init)) # unscaled
    return xe_init

def xe_from_yopf(y,nlsys=None):
    """Returns the variables of steady-state LF, from the variables y of OF."""
    xe = y[-len(nlsys.x_entries):]
    return xe

def objective_function(y,y_ind,a=np.array([0,0]),b=np.array([.2,.3]),c=np.array([2e-5,3e-5]),scale_var=None,scale_var_params=None,fb=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    E : np array
        Array with flows used in objective. Gas flows are assumed to be in kg/s, active powers in W, and heat power in W. Scaled when per unit scaling is used, unscaled otherwise.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [P1,P2]. Scaled when per unit scaling is used, unscaled otherwise.
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
    # print('P in obj: {}'.format(y[np.array(y_ind)]))
    f = np.sum(a+b*np.sign(y[np.array(y_ind)])*y[np.array(y_ind)]+c*y[np.array(y_ind)]**2)
    if scale_var == 'matrix':
        f *= (1/fb)
    return f

def grad_objective(y,y_ind,a=np.array([0,0]),b=np.array([.2,.3]),c=np.array([2e-5,3e-5]),scale_var=None,scale_var_params=None,fb=None,Dy_inv=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [P1,P2]. Scaled when per unit scaling is used, unscaled otherwise.
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

def hess_objective(y,y_ind,a=np.array([0,0]),b=np.array([.2,.3]),c=np.array([2e-5,3e-5]),scale_var=None,scale_var_params=None,fb=None,Dy_inv=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [P1,P2]. Scaled when per unit scaling is used, unscaled otherwise.
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

def h(y, nlsys=None, Dh=None):
    """Equality constraints h(x)=0. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemElectrical
        Nonlinear system of the electrical network. Constains the networks, information about scaling, etc.

    Returns
    -------
    h : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    network = nlsys.elecnetwork
    network = reset_network_without_update(network)

    xe = xe_from_yopf(y,nlsys=nlsys)
    network.update(xe,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) # this should set the correct values on the links and half links.

    xG = y[2:-len(xe)] # scaled when per unit scaling is used
    # evaluate conservation of energy in the slack and generator node
    len_inj_S = 3
    P1_gen, Q1_gen, Q2_gen = xG[:len_inj_S]
    if nlsys.scale_var == 'per_unit':
        network.nodes[1].half_links[0].Q = Q2_gen*nlsys.scale_var_params.get('Sbase')
    else:
        network.nodes[1].half_links[0].Q = Q2_gen
    fP1,fQ1 = network.nodes[0].node_law(network=network,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) + np.array([P1_gen,Q1_gen]) #Node 1 is a slack node, so it has no half links connected to it (yet)
    _,fQ2 = network.nodes[1].node_law(network=network,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    cons_energy = np.array([fP1,fQ1,fQ2])
    # link power equations |S_ij|^2 - P_ij^2 - Q_ij^2
    if len(xG) > len_inj_S: # (squared) link powers are part of extended state varialbes
        fS = np.zeros(len(network.links))
        for ind_e, link in enumerate(network.get_links()):
            fS[ind_e] = xG[len_inj_S+ind_e] - link.get_Pstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)**2 - link.get_Qstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)**2
        G = np.concatenate((cons_energy,fS))
    else:
        G = cons_energy
    # evaluate load flow equations
    network = reset_network_without_update(network)
    F = nlsys.F(xe)

    h = np.concatenate((G,F)) # already scaled if per unit is used
    if nlsys.scale_var == 'matrix':
        h = Dh.dot(h)
    return h

def link_power_derivatives(link,scale_var=None,scale_var_params=None):
    adm = 0
    adm_sh = 0
    if link.b:
        adm = np.complex(link.g,link.b)
    if link.link_params.get('b_sh'):
        adm_sh = np.complex(link.g_sh/2,link.b_sh/2)
    if scale_var == 'per_unit':
        adm_base = scale_var_params.get('Sbase')/scale_var_params.get('Vbase')**2
        adm /= adm_base
        adm_sh /= adm_base
    Pij = link.get_Pstart(scale_var=scale_var,scale_var_params=scale_var_params)
    Qij = link.get_Qstart(scale_var=scale_var,scale_var_params=scale_var_params)
    Vi = link.start_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params)
    Vj = link.end_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params)
    dPij_ddeltai = -Qij - (adm.imag+adm_sh.imag)*Vi**2
    dQij_ddeltai = Pij - (adm.real+adm_sh.real)*Vi**2
    dPij_ddeltaj = Qij + (adm.imag+adm_sh.imag)*Vi**2
    dQij_ddeltaj = -Pij + (adm.real+adm_sh.real)*Vi**2
    dPij_dVi = Pij/Vi + (adm.real+adm_sh.real)*Vi
    dQij_dVi = Qij/Vi - (adm.imag+adm_sh.imag)*Vi
    dPij_dVj = Pij/Vj - (adm.real+adm_sh.real)*Vi**2/Vj
    dQij_dVj = Qij/Vj + (adm.imag+adm_sh.imag)*Vi**2/Vj
    return dPij_ddeltai, dQij_ddeltai, dPij_ddeltaj, dQij_ddeltaj, dPij_dVi, dQij_dVi, dPij_dVj, dQij_dVj

def h_der(y, nlsys=None, Dy_inv=None, Dh=None):
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
    len_h = Dh.shape[0]
    len_u = len(y) - len_h
    dh_dy = np.zeros((len_h,len(y)))

    network = nlsys.elecnetwork
    network = reset_network_without_update(network)
    xe = xe_from_yopf(y,nlsys=nlsys)
    xG = y[len_u:-len(xe)]
    # update vector with voltage amplitudes and magnitudes, since nlsys.J() only updates the voltage amplitudes and angels that are in x
    V2, P2 = y[:len_u]
    nlsys.V_vec_mag[1] = V2
    # evaluate Jacobian of LF equations
    J_full = nlsys.J_dense(xe,return_full=True) #This also updates the network
    N = len(network.nodes) # number of nodes in the network
    F_ind = nlsys.FP + [N+ind for ind in nlsys.FQ]
    G_ind = [0,N,N+1] # indices of conservation of energy equations within full J
    xlf_ind = nlsys.xdelta + [N+ind for ind in nlsys.xV]
    dh_dy[:len(G_ind),:][:,len_u+len(xG):] = J_full[G_ind,:][:,xlf_ind] #dG_dxlf (except the link power equations)
    dh_dy[len(xG):,:][:,len_u+len(xG):] = J_full[F_ind,:][:,xlf_ind] #J_lf
    # derivatives of G (except the link power equations) and F to u
    V2_ind = N+1
    dh_dy[:len(G_ind),:][:,0] = J_full[G_ind,V2_ind].ravel() #dG_dV3 (except the link power equations)
    dh_dy[len(xG):,:][:,0] = J_full[F_ind,V2_ind].ravel()
    dh_dy[len(xG),1] = 1 #dFP2_dP2
    # derivatives to xG
    dh_dy[:len(xG),:][:,len_u:len_u+len(xG)] = np.eye(len(xG))
    # derivatives of link power equations |S_ij|^2 - P_ij^2 - Q_ij^2
    len_inj_S = 2*N - len(nlsys.known_P_nodes) - len(nlsys.known_Q_nodes)
    # print('xG: {}, inj S: {}'.format(len(xG),len_inj_S))
    if len(xG) > len_inj_S: # (squared) link powers are part of extended state varialbes
        for ind_e, link in enumerate(network.get_links()):
            Pij = link.get_Pstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            Qij = link.get_Qstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            Vi = link.start_node.get_V(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            Vj = link.end_node.get_V(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            dPij_ddeltai, dQij_ddeltai, dPij_ddeltaj, dQij_ddeltaj, dPij_dVi, dQij_dVi, dPij_dVj, dQij_dVj = link_power_derivatives(link,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            if link.start_node in nlsys.unknown_delta_nodes:
                dh_dy[len(G_ind)+ind_e,len_u+len(xG)+nlsys.unknown_delta_nodes.index(link.start_node)] = -2*Pij*dPij_ddeltai - 2*Qij*dQij_ddeltai
            if link.end_node in nlsys.unknown_delta_nodes:
                dh_dy[len(G_ind)+ind_e,len_u+len(xG)+nlsys.unknown_delta_nodes.index(link.end_node)] = -2*Pij*dPij_ddeltaj - 2*Qij*dQij_ddeltaj
            if link.start_node in nlsys.unknown_V_nodes:
                dh_dy[len(G_ind)+ind_e,len_u+len(xG)+len(nlsys.unknown_delta_nodes)+nlsys.unknown_V_nodes.index(link.start_node)] = -2*Pij*dPij_dVi - 2*Qij*dQij_dVi
            if link.end_node in nlsys.unknown_V_nodes:
                dh_dy[len(G_ind)+ind_e,len_u+len(xG)+len(nlsys.unknown_delta_nodes)+nlsys.unknown_V_nodes.index(link.end_node)] = -2*Pij*dPij_dVj - 2*Qij*dQij_dVj
            if link.start_node == network.nodes[1]: # derivative to V2
                dh_dy[len(G_ind)+ind_e,0] = -2*Pij*dPij_dVi - 2*Qij*dQij_dVi
            elif link.end_node == network.nodes[1]: # derivative to V2
                dh_dy[len(G_ind)+ind_e,0] = -2*Pij*dPij_dVj - 2*Qij*dQij_dVj
    if nlsys.scale_var == 'matrix':
        dh_dy = Dh.dot(dh_dy.dot(Dy_inv))
    # plt.figure('Jacobian')
    # plt.spy(dh_dy)
    # plt.figure('Jacobian values')
    # jac_val = dh_dy.copy()
    # jac_val[jac_val == 0] = np.nan
    # plt.imshow(jac_val)
    # plt.show()
    return dh_dy

def gamma(y, nlsys=None, ineq_constr_ub=np.array([(3*MW)**2,(3*MW)**2])):
    """The nonlinear inequality constraints gamma(x)>=0 on the link powers. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeterogeneous
        Nonlinear system of the heterogenous network. Constains the networks, information about scaling, etc.
    ineq_constr_ub : np arrays
        Upper bound for the link flows. Scaled when per unit or matrix scaling is used.

    Returns
    -------
    gam : np array
        The nonlinear inequality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    network = nlsys.elecnetwork
    network = reset_network_without_update(network)

    xe = xe_from_yopf(y,nlsys=nlsys)
    network.update(xe,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) # this should set the correct values on the links and half links.
    link_powers_squared = np.zeros(len(network.links))
    for ind_e, link in enumerate(network.get_links()):
        Sk2 =  link.get_Pstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)**2 + link.get_Qstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)**2
        if nlsys.scale_var == 'matrix': # lower and upper bounds are scaled
            Sk2 = Sk2/nlsys.scale_var_params.get('Sbase')**2
        link_powers_squared[ind_e] = Sk2
    return ineq_constr_ub - link_powers_squared

def gamma_der(y, nlsys=None, Dy_inv=None):
    """The nonlinear inequality constraints gamma(x)>=0 on the link powers. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

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
    network = nlsys.elecnetwork
    network = reset_network_without_update(network)
    xe = xe_from_yopf(y,nlsys=nlsys)
    len_u = 2
    xG = y[len_u:-len(xe)]
    network.update(xe,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) # this should set the correct values on the links and half links.
    len_E = len(network.links)
    dgamma_dy = np.zeros((len_E,len(y)))
    for ind_e, link in enumerate(network.get_links()):
        Pij = link.get_Pstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        Qij = link.get_Qstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        Vi = link.start_node.get_V(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        Vj = link.end_node.get_V(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dPij_ddeltai, dQij_ddeltai, dPij_ddeltaj, dQij_ddeltaj, dPij_dVi, dQij_dVi, dPij_dVj, dQij_dVj = link_power_derivatives(link,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        if link.start_node in nlsys.unknown_delta_nodes:
            dgamma_dy[ind_e,len_u+len(xG)+nlsys.unknown_delta_nodes.index(link.start_node)] = -2*Pij*dPij_ddeltai - 2*Qij*dQij_ddeltai
        if link.end_node in nlsys.unknown_delta_nodes:
            dgamma_dy[ind_e,len_u+len(xG)+nlsys.unknown_delta_nodes.index(link.end_node)] = -2*Pij*dPij_ddeltaj - 2*Qij*dQij_ddeltaj
        if link.start_node in nlsys.unknown_V_nodes:
            dgamma_dy[ind_e,len_u+len(xG)+len(nlsys.unknown_delta_nodes)+nlsys.unknown_V_nodes.index(link.start_node)] = -2*Pij*dPij_dVi - 2*Qij*dQij_dVi
        if link.end_node in nlsys.unknown_V_nodes:
            dgamma_dy[ind_e,len_u+len(xG)+len(nlsys.unknown_delta_nodes)+nlsys.unknown_V_nodes.index(link.end_node)] = -2*Pij*dPij_dVj - 2*Qij*dQij_dVj
        if link.start_node == network.nodes[1]: # derivative to V2
            dgamma_dy[ind_e,0] = -2*Pij*dPij_dVi - 2*Qij*dQij_dVi
        elif link.end_node == network.nodes[1]: # derivative to V2
            dgamma_dy[ind_e,0] = -2*Pij*dPij_dVj - 2*Qij*dQij_dVj
    if nlsys.scale_var == 'matrix':
        dgamma_dy = np.diag(1/(nlsys.scale_var_params.get('Sbase')**2)*np.ones(len_E)).dot(dgamma_dy.dot(Dy_inv))
    # plt.figure('Jacobian gamma')
    # plt.spy(dgamma_dy)
    return dgamma_dy

def reset_network_without_update(elec_net):
    """Reset the electrical network, but do not update the network values. That is, remove the half link connected to the slack node, and set Q of generator to zero."""
    elec_net.nodes[1].half_links[0].Q = 0
    elec_net.nodes[0].half_links = list()
    return elec_net

def update_bc(elec_net,V2,P2,scale_var=None,scale_var_params=None):
    """Update the boundary conditions of the electrical network, based on the control variables of OF"""
    if scale_var == 'per_unit':
        V2 = V2*scale_var_params.get('Vbase')
        P2 = P2*scale_var_params.get('Sbase')
    elec_net.nodes[1].V = V2
    elec_net.nodes[1].half_links[0].P = P2
    return elec_net

def run_optimal_load_flow(u_lb=np.array([.98*50*kV,-1.5*MW]),u_ub=np.array([1.02*50*kV,0]),u_init=np.array([50*kV,-1*MW]),slack_lb=np.array([-2*1.5*MW,-2*1.5*MW,-1.5*MW]),slack_ub=np.array([0,2*1.5*MW,1.5*MW]),slack_init_base=np.array([-1.5*MW,-1.5*MW,-1.5*MW]),xLF_lb=np.array([-np.pi,-np.pi,.98*50*kV]),xLF_ub=np.array([np.pi,np.pi,1.02*50*kV]),n=0,m=0,s=0,V1=50*kV,delta1=0.,P3=1.5*MW,Q3=1.5*MW,L12=4*km,L23=5*km,D=1*cm,link_type='pi_line',Lstreets=5*km,Dstreets=1*cm,y_ind = [2,1],a=np.array([0,0]),b=np.array([.2,.3]),c=np.array([2e-5,3e-5]),link_powers_slack=False,link_powers_limit=False,ineq_constr_ub=np.array([(3*MW)**2,(3*MW)**2]),scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,optimization_method='trust-constr',stay_within_bounds=False,fb=None,runs=1):
    """Run optimal load flow, with LF as equality constraints.

    Parameters
    ----------
    link_powers_slack : bool, optional
        Determines if the squared complex link power |S_ij|^2 should be included as slack variables. This parameter is ignored if link_powers_limit if False. Default is False.
    link_powers_limit : bool, optional
        Determines if limits are imposed on the squared complex link power |S_ij|^2. Default is False.

    Returns
    -------
    xe_opt : np array
        Solution of LF variables of OF. Unscaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_times : float
        List with time of the optimization (excluding creation of network etc.).
    """
    print('\nRunning OPF for E3N line with LF as eq. constr. (link powers limit: {}, link powers as slack: {}, hard bounds: {}, method: {}, scaling: {})'.format(link_powers_slack,link_powers_limit,stay_within_bounds,optimization_method,scale_var))

    # create network
    V2, P2 = u_init
    elec_net = create_network(n=n,m=m,s=s,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,L12=L12,L23=L23,D=D,link_type=link_type,Lstreets=Lstreets,Dstreets=Dstreets)

    # set initial guess and initialize network, using default initial guess
    elec_net.initialize()
    nlsys = NonLinearSystemElectrical(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    len_N = len(elec_net.nodes)
    len_inj_S = 2*len_N - len(nlsys.known_P_nodes) - len(nlsys.known_Q_nodes)
    len_E = len(elec_net.links)
    xLF_init = set_x_LF_init(nlsys)# unscaled
    if s>0 and link_powers_limit and link_powers_slack:
        link_powers = np.zeros(len_E-2)
        for ind, link in enumerate(elec_net.get_links()):
            if link.number >= 2: # the first two links are already included in the base case
                link_powers[ind-2] = link.get_Pstart()**2 + link.get_Qstart()**2 #unscaled
        slack_init = np.concatenate((slack_init_base,link_powers))
    else:
        slack_init = slack_init_base
    # initial guess for OF (unscaled)
    y0 = np.concatenate((u_init,slack_init,xLF_init))

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('Vbase'), scale_var_params.get('Sbase')])
        slack_base = scale_var_params.get('Sbase')*np.ones(len_inj_S)
        G_base = scale_var_params.get('Sbase')*np.ones(len_inj_S)
        if link_powers_limit and link_powers_slack:
            slack_base = np.concatenate((slack_base,scale_var_params.get('Sbase')**2*np.ones(len_E)))
            G_base = np.concatenate((G_base,scale_var_params.get('Sbase')**2*np.ones(len_E)))
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

    def obj_grad(y,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb, Dy_inv=Dy_inv):
        if nlsys.scale_var == 'matrix':
            y = Dy_inv.dot(y)
        return grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb,Dy_inv=Dy_inv)

    def obj_hess(y,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb):
        if nlsys.scale_var == 'matrix':
            y = Dy_inv.dot(y)
        return hess_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb,Dy_inv=Dy_inv)

    # define nonlinear equality constraints (load flow equations)
    def eq_constr(y,nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh):
        network = nlsys.elecnetwork
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        V2, P2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network = update_bc(network, V2,P2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        H = h(y, nlsys=nlsys, Dh=Dh)
        global err_LF_vec_global
        F = H[len(u_init)+len(slack_init):]
        err_LF_vec_global.append(np.linalg.norm(F))
        # print('dim h: {}'.format(H.shape))
        return H

    def jac_eq_constr(y,nlsys=nlsys, Dy_inv=Dy_inv, Dh=Dh):
        network = nlsys.elecnetwork
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        V2, P2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network = update_bc(network, V2,P2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dh_dy = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh)
        return dh_dy

    lb_nleq = np.zeros(len(slack_lb)+len(xLF_lb))
    ub_nleq = np.zeros(len(slack_ub)+len(xLF_ub))
    if optimization_method == 'trust-constr':
        nonlinear_constraint = spo.NonlinearConstraint(eq_constr,lb_nleq,ub_nleq,jac=jac_eq_constr,keep_feasible=stay_within_bounds)
    elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        nonlinear_constraint = {'type':'eq','fun':eq_constr,'jac':jac_eq_constr}

    # define nonlinear inequality constraints on squared link powers
    if link_powers_limit and (not link_powers_slack):
        def g(y,nlsys=nlsys, ineq_constr_ub=ineq_constr_ub, Dy_inv=Dy_inv):
            network = nlsys.elecnetwork
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                y = Dy_inv.dot(y)
            # update bc of the network
            V2, P2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
            network = update_bc(network, V2,P2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            gam = gamma(y,nlsys=nlsys, ineq_constr_ub=ineq_constr_ub)
            return gam
        def g_jac(y,nlsys=nlsys, Dy_inv=Dy_inv):
            network = nlsys.elecnetwork
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                y = Dy_inv.dot(y)
            # update bc of the network
            V2, P2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
            network = update_bc(network, V2,P2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            dgamma_dy = gamma_der(y, nlsys=nlsys, Dy_inv=Dy_inv)
            return dgamma_dy
        if optimization_method == 'trust-constr':
            ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(len(ineq_constr_ub)),np.inf*np.ones(len(ineq_constr_ub)),jac=g_jac,keep_feasible=stay_within_bounds)
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
    if link_powers_limit and link_powers_slack:
        lb_ineq[len(u_lb)+len_inj_S:len(u_lb)+len_inj_S+len_E] = -np.inf # use link powers squared, so no lower limit

    if optimization_method == 'ipopt':
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
        bounds = [(lb,ub) for lb, ub in zip(lb_ineq,ub_ineq)]
    else:
        bounds = spo.Bounds(lb_ineq,ub_ineq,keep_feasible=stay_within_bounds)

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
    execution_times = list()
    success_previous_run = 1
    for run in range(runs):
        print('Run {} of {}'.format(run+1,runs))
        if success_previous_run == 1:
            if run > 0:
                elec_net = reset_network_without_update(elec_net)
                V2_init, P2_init = u_init # unscaled
                elec_net = update_bc(elec_net, V2_init, P2_init)
                elec_net.update(xLF_init) # unscaled
                f_vec_global = list()
                y_f_vec = y0.copy()
                err_LF_vec_global = list()
                if optimization_method == 'trust-constr':
                    f_vec = list()
                elif optimization_method == 'SLSQP':
                    f_vec = [obj(y0)] # this call to obj() alters all the global variables.
            opf_start_time = time.perf_counter()
            try:
                if ineq_constr_fun != None:
                    if optimization_method == 'trust-constr':
                        res = spo.minimize(obj, y0, method=optimization_method,jac=obj_grad,hess=obj_hess, constraints=[nonlinear_constraint, ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter,'factorization_method':'SVDFactorization'}, bounds=bounds,tol=tol, callback=callback) #specify factorization method to avoid weird error about matrix not being square. Another options seems to set the bounds of the equality constraint not exactly equal. See https://stackoverflow.com/questions/61753007/how-to-solve-the-problem-of-the-valueerror-expected-square-matrix-in-a-constra
                        execution_times.append(res.execution_time)
                    elif optimization_method == 'SLSQP':
                        res = spo.minimize(obj, y0, method=optimization_method,jac=obj_grad, constraints=[nonlinear_constraint,ineq_constr_fun], options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                        execution_times.append(time.perf_counter() - opf_start_time)
                    elif optimization_method == 'ipopt':
                        res = ipopt.minimize_ipopt(obj, y0,jac=obj_grad, constraints=[nonlinear_constraint,ineq_constr_fun], options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                        execution_times.append(time.perf_counter() - opf_start_time)
                else:
                    if optimization_method == 'trust-constr':
                        res = spo.minimize(obj, y0, method=optimization_method,jac=obj_grad,hess=obj_hess, constraints=nonlinear_constraint, options={'verbose': 1,'maxiter':max_iter,'factorization_method':'SVDFactorization'}, bounds=bounds,tol=tol, callback=callback)
                        execution_times.append(res.execution_time)
                    elif optimization_method == 'SLSQP':
                        res = spo.minimize(obj, y0, method=optimization_method,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                        execution_times.append(time.perf_counter() - opf_start_time)
                    elif optimization_method == 'ipopt':
                        res = ipopt.minimize_ipopt(obj, y0,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                        execution_times.append(time.perf_counter() - opf_start_time)
            except:
                print('Exception made for {}, link powers limit: {}, link powers as slack: {}, hard bounds: {}, scaling: {}'.format(optimization_method,link_powers_slack,link_powers_limit,stay_within_bounds,scale_var))
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
                execution_times.append(time.perf_counter() - opf_start_time)
                res = spo.OptimizeResult({'success':False,'x':np.array(y_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})
            success_previous_run = res.success
        else:
            print('Previous run was unsuccesful, so no further runs are done.')
            break
    res.execution_time = np.mean(execution_times)

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
    V2, P2 = y_opf[:len(u_init)] # unscaled
    elec_net = update_bc(elec_net, V2,P2)
    xe_opt = xe_from_yopf(y_opf,nlsys=nlsys) #unscaled
    print('Solution OF (success: {})'.format(res.success))
    if len(xe_opt) < 10:
        elec_net = reset_network_without_update(elec_net)
        delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.update_full(xe_opt)
        print('delta = {}'.format(delta_sol))
        print('|V| = {} kV'.format(V_sol/kV))
        print('P edge = {} MW'.format(P_edge/MW))
        print('Q edge = {} MW'.format(Q_edge/MW))
        print('S nodal inj = {} MW'.format(S_inj/MW))
        print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
        print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    return xe_opt, res, f_vec, err_LF_vec, execution_times

def get_u_from_net(elec_net,scale_var=None,scale_var_params=None):
    """Get the current values of the control variables u from the network.
    """
    V2 = elec_net.nodes[1].get_V(scale_var=scale_var,scale_var_params=scale_var_params)
    P2 = elec_net.nodes[1].half_links[0].get_P(scale_var=scale_var,scale_var_params=scale_var_params)
    return np.array([V2,P2])

def solve_lf_in_of(u,nlsys,max_iters=10,tol=1e-6,xLF_init=np.array([0,0,50*kV]),link_powers_slack=False):
    """Solve steady-state LF within an optimization context.
    """
    global err_LF_vec_global
    network = nlsys.elecnetwork
    u_net = get_u_from_net(network,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    if len(err_LF_vec_global)==0 or (np.linalg.norm(u-u_net) > tol) or (err_LF_vec_global[-1] > tol):
        V2, P2 = u
        network = reset_network_without_update(network)
        network = update_bc(network, V2,P2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        with HiddenPrints():
            xe,iters,err_vec,delta,V_vec,S_inj_vec,P_edge,Q_edge = network.solve_network(tol,max_iters,solver='NR', scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        if err_vec[-1] >= tol:
            network = reset_network_without_update(network)
            network.update(xLF_init, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            xe,iters,err_vec,delta_vec,V_vec,S_inj_vec,P_edge,Q_edge = network.solve_network(tol,max_iters,solver='NR', scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        err_LF = err_vec[-1]
    else:
        network = reset_network_without_update(network)
        xe = network.set_x_init(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        delta_vec,V_vec,S_inj_vec,P_edge,Q_edge = network.update_full(xe,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        err_LF = err_LF_vec_global[-1]
    err_LF_vec_global.append(err_LF)
    P1 = S_inj_vec[0].real
    Q1 = S_inj_vec[0].imag
    Q2 = S_inj_vec[1].imag
    if nlsys.scale_var == 'per_unit':
        P1 = P1 / nlsys.scale_var_params.get('Sbase')
        Q1 = Q1 / nlsys.scale_var_params.get('Sbase')
        Q2 = Q2 / nlsys.scale_var_params.get('Sbase')
    if link_powers_slack:
        slack = np.zeros(3+len(network.links))
        slack[:3] = np.array([P1,Q1,Q2])
        for ind_link,link in enumerate(network.get_links()):
            Sk_squared = link.get_Pstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)**2 + link.get_Qstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)**2
            slack[3+ind_link] = Sk_squared
    else:
        slack = np.array([P1,Q1,Q2])
    x = np.concatenate((slack,xe)) # unscale unless per unit scaling is used
    return x

def run_optimal_load_flow_separate_LF(u_lb=np.array([.98*50*kV,-1.5*MW]),u_ub=np.array([1.02*50*kV,0]),u_init=np.array([50*kV,-1*MW]),slack_lb=np.array([-2*1.5*MW,-2*1.5*MW,-1.5*MW]),slack_ub=np.array([0,2*1.5*MW,1.5*MW]),slack_init_base=np.array([-1.5*MW,-1.5*MW,-1.5*MW]),xLF_lb=np.array([-np.pi,-np.pi,.98*50*kV]),xLF_ub=np.array([np.pi,np.pi,1.02*50*kV]),n=0,m=0,s=0,V1=50*kV,delta1=0.,P3=1.5*MW,Q3=1.5*MW,L12=4*km,L23=5*km,D=1*cm,link_type='pi_line',Lstreets=5*km,Dstreets=1*cm,y_ind = [2,1],a=np.array([0,0]),b=np.array([.2,.3]),c=np.array([2e-5,3e-5]),link_powers_slack=False,link_powers_limit=False,ineq_constr_ub=np.array([(3*MW)**2,(3*MW)**2]),scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,max_iters_lf=10,approach='direct',optimization_method='trust-constr',stay_within_bounds=False,fb=None,runs=1):
    """Run optimal load flow, with LF included implicitely.

    Parameters
    ----------
    link_powers_slack : bool, optional
        Determines if the squared complex link power |S_ij|^2 should be included as slack variables. This parameter is ignored if link_powers_limit if False. Default is False.
    link_powers_limit : bool, optional
        Determines if limits are imposed on the squared complex link power |S_ij|^2. Default is False.

    Returns
    -------
    xe_opt : np array
        Solution of LF variables of OF. Unscaled.
    y_opt : np array
        Full solution of OF. That is, control variables and all state variables. Scaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_time : float
        List with times of the optimization (excluding creation of network etc.).
    """
    print('\nRunning OPF for E3N line with separate LF (link powers limit: {}, link powers as slack: {}, hard bounds: {}, method: {}, scaling: {}, approach: {})'.format(link_powers_slack,link_powers_limit,stay_within_bounds,optimization_method,scale_var,approach))

    # create network
    V2, P2 = u_init
    elec_net = create_network(n=n,m=m,s=s,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,L12=L12,L23=L23,D=D,link_type=link_type,Lstreets=Lstreets,Dstreets=Dstreets)

    # set initial guess and initialize network, using default initial guess
    elec_net.initialize()
    nlsys = NonLinearSystemElectrical(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    len_N = len(elec_net.nodes)
    len_inj_S = 2*len_N - len(nlsys.known_P_nodes) - len(nlsys.known_Q_nodes)
    len_E = len(elec_net.links)
    xLF_init = set_x_LF_init(nlsys)# unscaled
    if s>0 and link_powers_limit and link_powers_slack:
        link_powers = np.zeros(len_E-2)
        for ind, link in enumerate(elec_net.get_links()):
            if link.number >= 2: # the first two links are already included in the base case
                link_powers[ind-2] = link.get_Pstart()**2 + link.get_Qstart()**2 #unscaled
        slack_init = np.concatenate((slack_init_base,link_powers))
    else:
        slack_init = slack_init_base
    # initial guess for OF (unscaled)
    u0 = u_init

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('Vbase'), scale_var_params.get('Sbase')])
        slack_base = scale_var_params.get('Sbase')*np.ones(len_inj_S)
        G_base = scale_var_params.get('Sbase')*np.ones(len_inj_S)
        if link_powers_limit and link_powers_slack:
            slack_base = np.concatenate((slack_base,scale_var_params.get('Sbase')**2*np.ones(len_E)))
            G_base = np.concatenate((G_base,scale_var_params.get('Sbase')**2*np.ones(len_E)))
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((ubase,slack_base,1/Dx.data[0])))
        Du = np.diag(1/ubase)
        Du_inv = np.diag(ubase)
        Dh = np.diag(np.concatenate((1/G_base,DF.data[0])))
        u0 = Du.dot(u0) # scale u
        if scale_var == 'per_unit':
            xLF_init = Dx.dot(xLF_init) # scale x
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
    if link_powers_limit and link_powers_slack:
        lb_ineq_state = np.concatenate((lb_ineq_state[:len_inj_S],lb_ineq_state[len_inj_S+len_E:])) # use link powers squared, so no lower limit

    # define objective function
    def obj(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb,xLF_init=xLF_init,link_powers_slack=link_powers_slack):
        global u_f_vec
        u_f_vec = u.copy()
        global f_vec_global
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,link_powers_slack=link_powers_slack)
        y = np.concatenate((u,x))
        f = objective_function(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    def obj_grad(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb,xLF_init=xLF_init,link_powers_slack=link_powers_slack,Dy_inv=Dy_inv,Dh=Dh,method=approach):
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,link_powers_slack=link_powers_slack)
        y = np.concatenate((u,x))
        # partial derivatives of objective
        deltaf_deltay = grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb,Dy_inv=Dy_inv)
        deltaf_deltau = np.zeros((1,len(u)))
        deltaf_deltax = np.zeros((1,len(x)))
        deltaf_deltau[0,:] = deltaf_deltay[:len(u)]
        deltaf_deltax[0,:] = deltaf_deltay[len(u):]
        # partial derivatives of equatilty constraints / load-flow equations
        V2, P2 = u
        network = nlsys.elecnetwork
        network = update_bc(network, V2,P2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        deltah_deltay = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh)
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
    def g(u,nlsys=nlsys, lb_ineq_state=lb_ineq_state, ub_ineq_state=ub_ineq_state, ineq_constr_ub=ineq_constr_ub, link_powers_slack=link_powers_slack, link_powers_limit = link_powers_limit, Dy=Dy, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init):
        if nlsys.scale_var == 'matrix':
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,link_powers_slack=link_powers_slack)
        y = np.concatenate((u,x))
        if scale_var == 'matrix': # lb_ineq_state and ub_ineq_state are scaled, so scale x as well
            x = Dy[len(u):,len(u):].dot(x)
        if link_powers_slack:
            x_lb = np.concatenate((x[:len_inj_S],x[len_inj_S+len_E:])) # use link powers squared, so no lower limit
        else:
            x_lb = x
        g = np.concatenate((x_lb-lb_ineq_state,ub_ineq_state-x))
        if link_powers_limit and (not link_powers_slack):
            # update bc of the network
            V2, P2 = u
            network = nlsys.elecnetwork
            network = update_bc(network, V2,P2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            gam = gamma(y,nlsys=nlsys, ineq_constr_ub=ineq_constr_ub)
            g = np.concatenate((gam,g))
        return g
    def g_jac(u,nlsys=nlsys, link_powers_slack=link_powers_slack, link_powers_limit = link_powers_limit, Dy=Dy, Dh=Dh, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init,method=approach):
        if nlsys.scale_var == 'matrix':
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,link_powers_slack=link_powers_slack)
        y = np.concatenate((u,x))
        # Jacobian of inequality constraints on state variables (without gamma, that part is added later)
        len_x = len(x)
        len_state = len_x-len_E
        if link_powers_slack:
            I_lb = np.zeros((len_state,len_x))
            for ind in range(len_inj_S):
                I_lb[ind,ind] = 1
            for ind in range(len_state-len_inj_S):
                I_lb[len_inj_S+ind,len_inj_S+len_E+ind] = 1
        else:
            I_lb = np.eye(len_x)
        I = np.eye(len_x)
        deltag_deltax = np.vstack((I_lb,-I))
        deltag_deltau = np.zeros((deltag_deltax.shape[0],len(u)))
        # partial derivatives of equatilty constraints / load-flow equations
        V2, P2 = u
        network = nlsys.elecnetwork
        network = update_bc(network, V2,P2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        deltah_deltay = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh)
        deltah_deltau = deltah_deltay[:,:len(u)]
        deltah_deltax = deltah_deltay[:,len(u):]
        if link_powers_limit and (not link_powers_slack):
            deltagamma_deltay = gamma_der(y, nlsys=nlsys, Dy_inv=Dy_inv)
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
        len_g = len(lb_ineq_state) + len(ub_ineq_state)
        if link_powers_limit and (not link_powers_slack):
            len_g += len(ineq_constr_ub)
        ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(len_g),np.inf*np.ones(len_g),jac=g_jac,keep_feasible=stay_within_bounds)
    elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        ineq_constr_fun  = {'type':'ineq','fun':g,'jac':g_jac}

    # define linear inequality constraints (bounds) on the control variables
    if optimization_method == 'ipopt':
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
        bounds = [(lb,ub) for lb, ub in zip(u_lb,u_ub)]
    else:
        bounds = spo.Bounds(u_lb,u_ub,keep_feasible=stay_within_bounds)

    # make sure initial guess satisfies bounds
    if optimization_method == 'SLSQP' or stay_within_bounds:
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
    execution_times = list()
    success_previous_run = 1
    for run in range(runs):
        print('Run {} of {}'.format(run+1,runs))
        if success_previous_run == 1:
            if run > 0:
                elec_net = reset_network_without_update(elec_net)
                V2_init, P2_init = u_init # unscaled
                elec_net = update_bc(elec_net, V2_init, P2_init)
                elec_net.update(xLF_init) # unscaled
                f_vec_global = list()
                u_f_vec = u0.copy()
                err_LF_vec_global = list()
                if optimization_method == 'trust-constr':
                    f_vec = list()
                elif optimization_method == 'SLSQP':
                    f_vec = [obj(u0)] # this call to obj() alters all the global variables.
            opf_start_time = time.perf_counter()
            try:
                if optimization_method == 'trust-constr':
                    res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=[ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter,'factorization_method':'SVDFactorization'}, bounds=bounds, callback=callback)
                    execution_times.append(res.execution_time)
                elif optimization_method == 'SLSQP':
                    res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                    execution_times.append(time.perf_counter() - opf_start_time)
                elif optimization_method == 'ipopt':
                    res = ipopt.minimize_ipopt(obj, u0,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                    execution_times.append(time.perf_counter() - opf_start_time)
            except:
                print('Exception made for {} (link powers limit: {}, link powers as slack: {}, hard bounds: {}, scaling: {}, approach: {})'.format(optimization_method,link_powers_slack,link_powers_limit,stay_within_bounds,scale_var,approach))
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
                execution_times.append(time.perf_counter() - opf_start_time)
                res = spo.OptimizeResult({'success':False,'x':np.array(u_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})
            success_previous_run = res.success
        else:
            print('Previous run was unsuccesful, so no further runs are done.')
            break
    res.execution_time = np.mean(execution_times)

    if optimization_method == 'ipopt':
        if res.nit > 0:
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            f_vec = f_vec_global

    if len(err_LF_vec_global) >= len(f_vec):
        indices = [int(round(ind)) for ind in np.linspace(0,len(err_LF_vec_global)-1,len(f_vec))]
        err_LF_vec = [err_LF_vec_global[ind] for ind in indices]
    elif len(err_LF_vec_global) == 0:
        obj(u0)
        err_LF_vec = [err_LF_vec_global[-1]]
    else:
        err_LF_vec = err_LF_vec_global

    if scale_var == 'matrix' or scale_var == 'per_unit':
        u_opf = Du_inv.dot(res.x)
    else:
        u_opf = res.x

    # print solution
    if scale_var == 'per_unit':
        x = solve_lf_in_of(res.x,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,link_powers_slack=link_powers_slack)
        y_opt = Dy_inv.dot(np.concatenate((res.x,x))) # unscaled
    else:
        x = solve_lf_in_of(u_opf,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,link_powers_slack=link_powers_slack)
        y_opt = np.concatenate((u_opf,x)) # unscaled
    xe_opt = xe_from_yopf(y_opt,nlsys=nlsys) # unscaled
    print('Solution OF (success: {})'.format(res.success))
    if len(xe_opt) < 10:
        elec_net = reset_network_without_update(elec_net)
        delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.update_full(xe_opt)
        print('delta = {}'.format(delta_sol))
        print('|V| = {} kV'.format(V_sol/kV))
        print('P edge = {} MW'.format(P_edge/MW))
        print('Q edge = {} MW'.format(Q_edge/MW))
        print('S nodal inj = {} MW'.format(S_inj/MW))
        print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
        print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    if scale_var == 'matrix' or scale_var == 'per_unit':
        y_opt = Dy.dot(y_opt) # scaled
    return xe_opt, y_opt, res, f_vec, err_LF_vec, execution_times


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
    """Compare OF for different optimization methods, formulations of OF, scalings, and bounds."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    max_iters_lf = 20
    tol = 1e-6

    # scaling
    if scale_var == None:
        scale_var_params = None
        fb = None
        scale_label = 'unscaled'
    else:
        deltabase = 1.
        Vbase = 50*kV
        Sbase = 1*MW
        scale_var_params = {'deltabase':deltabase,'Vbase':Vbase,'Sbase':Sbase}
        fb = 1e7
        if scale_var == 'matrix':
            scale_label = 'matrix'
        else:
            scale_label = 'pu'

    # parameter values for the objective function
    a=np.array([0,0])
    b=np.array([.2,.3])
    c=np.array([2e-5,3e-5])
    y_ind = [2,1] # indices of P1 and P2 within y. The parameters a, b, and c assume [P1,P2]

    # Network parameters and boundary conditions (of LF)
    V1 = 50*kV #[V]
    delta1 = 0.#[rad]
    if s == 0:
        V2 = 49.9854119*kV#49.98856105*kV#.99*V1 #[V]
        P2 = -.4*MW#-0.5999727456238263*MW#-.5*MW #[W]
        P3 = 1.5*MW #[W]
        Q3 = 1.5*MW #[Var]
    else:
        V2 = 49.98856105*kV#.99*V1 #[V]
        P2 = -0.5999727456238263*MW#-.5*MW #[W]
        P3 = 7*-P2 #[W]
        Q3 = 7*-P2 #[Var]
    L12=4*km
    L23=5*km
    D=1*cm
    link_type='pi_line'
    Lstreets=5*km
    Dstreets=1*cm

    # initial guess
    u_init=np.array([50*kV,-1.3*MW])
    slack_init_Sinj=np.array([-1.5*MW,-1.5*MW,-1.5*MW])
    link_powers_init = np.array([(1.5*MW)**2,(1.5*MW)**2])

    # steady-state LF solution (without scaling)
    with HiddenPrints():
        elec_net_LF = create_network(n=n,m=m,s=s,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,L12=L12,L23=L23,D=D,link_type=link_type,Lstreets=Lstreets,Dstreets=Dstreets)
        elec_net_LF.initialize()
        nlsys_LF = NonLinearSystemElectrical(elec_net_LF) # unscaled
        xLF_init = set_x_LF_init(nlsys_LF)
        # xLF_init = np.array([0,0,50*kV])
        elec_net_LF.update(xLF_init)
        x_sol_LF,iters_LF,err_vec_LF,delta_LF,V_LF,S_inj_LF,P_edge_LF,Q_edge_LF = elec_net_LF.solve_network(tol,max_iters_lf,solver='NR')
    y_LF = np.concatenate((np.array([V2,P2,S_inj_LF[0].real,S_inj_LF[0].imag,S_inj_LF[1].imag]),x_sol_LF))
    Sk_squared = np.zeros(len(elec_net_LF.links))
    for ind,link in enumerate(elec_net_LF.get_links()):
        Sk_squared[ind] = link.get_Pstart()**2 + link.get_Qstart()**2

    # bounds (partly based on LF solution)
    if s==0:
        u_lb=np.array([.98*50*kV,-1*MW])
        u_ub=np.array([1.01*50*kV,-.1*MW])
        slack_lb_Sinj=np.array([-2*1.5*MW,-2*1.5*MW,-1*1.5*MW])
        slack_ub_Sinj=np.array([0,2*1.5*MW,1.5*MW])
        ineq_constr_ub_base=np.array([(2*MW)**2+(2*MW)**2,(2*MW)**2+(2*MW)**2])
        xLF_lb_base=np.array([-np.pi/6,-np.pi/6,.98*50*kV])
        xLF_ub_base=np.array([np.pi/6,np.pi/6,1.02*50*kV])
    else:
        u_lb=np.array([.98*V2,1.3*P2])
        u_ub=np.array([1.02*V2,-.1*MW])
        slack_lb_Sinj=np.array([1.5*S_inj_LF[0].real,-1.5*np.abs(S_inj_LF[0].imag),-1.5*np.abs(S_inj_LF[1].imag)])
        slack_ub_Sinj=np.array([0,1.5*np.abs(S_inj_LF[0].imag),1.5*np.abs(S_inj_LF[1].imag)])
        ineq_constr_ub_base=np.array([(1.5*np.abs(P_edge_LF[0]))**2+(1.5*np.abs(Q_edge_LF[0]))**2,(1.5*np.abs(P_edge_LF[1]))**2+(1.5*np.abs(Q_edge_LF[1]))**2])
        xLF_lb_base=np.array([-np.pi/6,-np.pi/6,.98*np.min(V_LF)])
        xLF_ub_base=np.array([np.pi/6,np.pi/6,1.02*np.max(V_LF)])

    # adjust values of bounds to match size of network when there are streets
    ineq_constr_ub = ineq_constr_ub_base
    len_N = len(elec_net_LF.nodes)
    len_inj_S = 2*len_N - len(nlsys_LF.known_P_nodes) - len(nlsys_LF.known_Q_nodes)
    len_E = len(elec_net_LF.links)
    len_xdelta = len(nlsys_LF.unknown_delta_nodes)
    len_xV = len(nlsys_LF.unknown_V_nodes)
    if s>0:
        deltaLF_lb = np.zeros(len_xdelta)
        deltaLF_lb[:2] = xLF_lb_base[:2]
        deltaLF_lb[2:] = xLF_lb_base[1]
        deltaLF_ub = np.zeros(len_xdelta)
        deltaLF_ub[:2] = xLF_ub_base[:2]
        deltaLF_ub[2:] = xLF_ub_base[1]
        VLF_lb = xLF_lb_base[-1]*np.ones(len_xV)
        VLF_ub = xLF_ub_base[-1]*np.ones(len_xV)
        xLF_lb = np.concatenate((deltaLF_lb, VLF_lb))
        xLF_ub = np.concatenate((deltaLF_ub, VLF_ub))
    else:
        xLF_lb = xLF_lb_base
        xLF_ub = xLF_ub_base

    # scale LF solution
    if scale_var != None:
        nlsys_LF = NonLinearSystemElectrical(elec_net_LF,scale_var=scale_var,scale_var_params=scale_var_params)
        Dx = nlsys_LF.Dx()
        ubase = np.array([scale_var_params.get('Vbase'), scale_var_params.get('Sbase')])
        slack_base = scale_var_params.get('Sbase')*np.ones(3)
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        f_LF = objective_function(y_LF,y_ind,a=a,b=b,c=c)/fb #scaled
        y_LF = Dy.dot(y_LF) # scaled
        Sk_squared /= scale_var_params.get('Sbase')**2
    else:
        f_LF = objective_function(y_LF,y_ind,a=a,b=b,c=c,scale_var=scale_var,fb=fb)

    result = dict()
    y_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['eq_constr','direct','adjoint']
    forms_OF = [1,2,3] #[no limits on S_ij^2, limits on S_ij^2 with S_ij^2 not in xG, limits on S_ij^2 with S_ij^2 not in xG]
    total_cases = len(methods)*len(bounds)*len(approaches)*len(forms_OF)
    case_number = 1
    start_wall_clock = datetime.datetime.now().time()
    for bound in bounds:
        if bound == 'soft':
            stay_within_bounds = False
        else:
            stay_within_bounds = True
        fig_delta = plt.figure('delta_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
        ax_delta = fig_delta.gca()
        ax_delta.set_ylabel(r'$\delta$')
        fig_V = plt.figure('V_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
        ax_V = fig_V.gca()
        ax_V.set_ylabel(r'$|V|$')
        fig_P = plt.figure('Pinj_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
        ax_P = fig_P.gca()
        ax_P.set_ylabel(r'$P$')
        fig_Q = plt.figure('Qinj_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
        ax_Q = fig_Q.gca()
        ax_Q.set_ylabel(r'$Q$')
        fig_Sk = plt.figure('Sk_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
        ax_Sk = fig_Sk.gca()
        ax_Sk.set_ylabel(r'$|S_k|^2$')
        for form_OF in forms_OF:
            fig_f = plt.figure('elec_obj_'+str(form_OF)+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
            ax_f = fig_f.gca()
            ax_f.set_xlabel('Iteration?')
            ax_f.set_ylabel('f')

            fig_LF_error = plt.figure('elec_error_LF_in_OF_'+str(form_OF)+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
            ax_LF_error = fig_LF_error.gca()
            ax_LF_error.set_xlabel('Iteration?')
            ax_LF_error.set_ylabel(r'$||F||_2$')

            max_fev = 0

            slack_lb_base = slack_lb_Sinj.copy()
            slack_ub_base = slack_ub_Sinj.copy()
            slack_init_base = slack_init_Sinj.copy()
            slack_lb = slack_lb_base
            slack_ub = slack_ub_base
            if form_OF == 1:
                link_powers_limit = False
                link_powers_slack = False # is not used / required within OF
            elif form_OF == 2:
                link_powers_limit = True
                link_powers_slack = False
                ineq_constr_ub = np.zeros(len_E)
                ineq_constr_ub[:len(ineq_constr_ub_base)] = ineq_constr_ub_base[:]
                ineq_constr_ub[len(ineq_constr_ub_base):] = ineq_constr_ub_base[0]
            elif form_OF == 3:
                link_powers_limit = True
                link_powers_slack = True
                slack_lb_base = np.concatenate((slack_lb_Sinj,np.array([0,0])))
                slack_ub_base = np.concatenate((slack_ub_Sinj,ineq_constr_ub_base))
                slack_init_base = np.concatenate((slack_init_Sinj,link_powers_init))
                slack_lb = np.zeros(len_inj_S+len_E)
                slack_lb[:len(slack_lb_base)] = slack_lb_base[:]
                slack_ub = np.zeros(len_inj_S+len_E)
                slack_ub[:len(slack_lb_base)] = slack_ub_base[:]
                slack_ub[len(slack_lb_base):] = slack_ub_base[-2]
            for method in methods:
                for approach in approaches:
                    if approach == 'direct' or approach == 'adjoint':
                        xe_opt, y_opt, res, f_vec, err_LF_vec, execution_times = run_optimal_load_flow_separate_LF(u_lb=u_lb,u_ub=u_ub,u_init=u_init,slack_lb=slack_lb,slack_ub=slack_ub,slack_init_base=slack_init_base,xLF_lb=xLF_lb,xLF_ub=xLF_ub,n=n,m=m,s=s,V1=V1,delta1=delta1,P3=P3,Q3=Q3,L12=L12,L23=L23,D=D,link_type=link_type,Lstreets=Lstreets,Dstreets=Dstreets,y_ind = y_ind,a=a,b=b,c=c,link_powers_slack=link_powers_slack,link_powers_limit=link_powers_limit,ineq_constr_ub=ineq_constr_ub,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,approach=approach,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,runs=number_of_runs)
                    else:
                        xe_opt, res, f_vec, err_LF_vec, execution_times = run_optimal_load_flow(u_lb=u_lb,u_ub=u_ub,u_init=u_init,slack_lb=slack_lb,slack_ub=slack_ub,slack_init_base=slack_init_base,xLF_lb=xLF_lb,xLF_ub=xLF_ub,n=n,m=m,s=s,V1=V1,delta1=delta1,P3=P3,Q3=Q3,L12=L12,L23=L23,D=D,link_type=link_type,Lstreets=Lstreets,Dstreets=Dstreets,y_ind = y_ind,a=a,b=b,c=c,link_powers_slack=link_powers_slack,link_powers_limit=link_powers_limit,ineq_constr_ub=ineq_constr_ub,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,runs=number_of_runs)
                        y_opt = res.x
                    print('Finished case {} of {}, in average of {:.2f}s per run. Started at {}'.format(case_number,total_cases,res.execution_time,start_wall_clock))
                    case_number += 1
                    result[method+'_'+str(form_OF)+'_'+bound+'_'+approach] = res
                    y_res[method+'_'+str(form_OF)+'_'+bound+'_'+approach] = y_opt
                    max_fev = max(max_fev,len(f_vec))
                    # plot results
                    ax_f.semilogy(f_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form_OF),alpha=.5)
                    ax_LF_error.semilogy(err_LF_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form_OF),alpha=.5)
                    ax_delta.plot(y_opt[-len_xdelta-len_xV:-len_xV],color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form_OF),alpha=.5)
                    V_opt = np.concatenate((np.array([y_opt[0]]),y_opt[-len_xV:])) #scaled
                    ax_V.plot(V_opt,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form_OF),alpha=.5)
                    ax_P.plot(y_opt[y_ind],color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form_OF),alpha=.5)
                    ax_Q.plot(y_opt[3:5],color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form_OF),alpha=.5)
                    if scale_var != None:
                        elec_net_LF = update_bc(elec_net_LF,y_opt[0]*scale_var_params.get('Vbase'),y_opt[1]*scale_var_params.get('Sbase'))
                    else:
                        elec_net_LF = update_bc(elec_net_LF,y_opt[0],y_opt[1])
                    elec_net_LF.update(xe_opt)
                    Sk_squared_opt = np.zeros(len_E)
                    for ind, link in enumerate(elec_net_LF.get_links()):
                        Sk_squared_opt[ind] = link.get_Pstart(scale_var=nlsys_LF.scale_var,scale_var_params=nlsys_LF.scale_var_params)**2 + link.get_Qstart(scale_var=nlsys_LF.scale_var,scale_var_params=nlsys_LF.scale_var_params)**2
                    if scale_var == 'matrix':
                        Sk_squared_opt /= scale_var_params.get('Sbase')**2
                    ax_Sk.plot(Sk_squared_opt[:3+s+n+m],color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form_OF),alpha=.5)
                    if form_OF == 2:
                        if scale_var != None:
                            ax_Sk.plot(ineq_constr_ub[:3+s+n+m]/(scale_var_params.get('Sbase')**2),'k-')
                        else:
                            ax_Sk.plot(ineq_constr_ub[:3+s+n+m],'k-')
                    elif form_OF == 3 and (not 2 in forms_OF):
                        if scale_var != None:
                            ax_Sk.plot(slack_ub[len_inj_S:len_inj_S+3+s+n+m]/(scale_var_params.get('Sbase')**2),'k-')
                        else:
                            ax_Sk.plot(slack_ub[len_inj_S:len_inj_S+3+s+n+m],'k-')
            ax_f.semilogy([0,max_fev],[f_LF,f_LF],':r',alpha=.5)
            ax_LF_error.semilogy([0,max_fev],[tol,tol],':k')
            if scale_var != None:
                ax_delta.plot(xLF_lb[-len_xdelta-len_xV:-len_xV]/scale_var_params.get('deltabase'),'k-')
                ax_delta.plot(xLF_ub[-len_xdelta-len_xV:-len_xV]/scale_var_params.get('deltabase'),'k-')
                ax_delta.plot(delta_LF[1:]/scale_var_params.get('deltabase'),':r',alpha=.5)
                ax_V.plot(np.concatenate((np.array([u_lb[0]]),xLF_lb[-len_xV:]))/scale_var_params.get('Vbase'),'k-')
                ax_V.plot(np.concatenate((np.array([u_ub[0]]),xLF_ub[-len_xV:]))/scale_var_params.get('Vbase'),'k-')
                ax_V.plot(V_LF[1:]/scale_var_params.get('Vbase'),':r',alpha=.5)
                ax_P.plot([slack_lb[0]/scale_var_params.get('Sbase'),u_lb[1]/scale_var_params.get('Sbase')],'k-')
                ax_P.plot([slack_ub[0]/scale_var_params.get('Sbase'),u_ub[1]/scale_var_params.get('Sbase')],'k-')
                ax_P.plot(S_inj_LF.real[:3+s+n+m]/scale_var_params.get('Sbase'),':r',alpha=.5)
                ax_Q.plot(slack_lb[1:3]/scale_var_params.get('Sbase'),'k-')
                ax_Q.plot(slack_ub[1:3]/scale_var_params.get('Sbase'),'k-')
                ax_Q.plot(S_inj_LF.imag[:3+s+n+m]/scale_var_params.get('Sbase'),':r',alpha=.5)
            else:
                ax_delta.plot(xLF_lb[-len_xdelta-len_xV:-len_xV],'k-')
                ax_delta.plot(xLF_ub[-len_xdelta-len_xV:-len_xV],'k-')
                ax_delta.plot(delta_LF[1:],':r',alpha=.5)
                ax_V.plot(np.concatenate((np.array([u_lb[0]]),xLF_lb[-len_xV:])),'k-')
                ax_V.plot(np.concatenate((np.array([u_ub[0]]),xLF_ub[-len_xV:])),'k-')
                ax_V.plot(V_LF[1:],':r',alpha=.5)
                ax_P.plot([slack_lb[0],u_lb[1]],'k-')
                ax_P.plot([slack_ub[0],u_ub[1]],'k-')
                ax_P.plot(S_inj_LF.real[:3+s+n+m],':r',alpha=.5)
                ax_Q.plot(slack_lb[1:3],'k-')
                ax_Q.plot(slack_ub[1:3],'k-')
                ax_Q.plot(S_inj_LF.imag[:3+s+n+m],':r',alpha=.5)
            ax_Sk.plot(Sk_squared[:2+s+n+m],':r',alpha=.5)
            ax_delta.set_xticklabels([int(xtick+1+1) for xtick in ax_delta.get_xticks()]) #+1 since the first node has known delta, +1 since the first node is called 1 and not 0
            ax_delta.set_xlabel('Node number')
            ax_V.set_xticklabels([int(xtick+1+1) for xtick in ax_V.get_xticks()]) #+1 since the first node has known V, +1 since the first node is called 1 and not 0
            ax_V.set_xlabel('Node number')
            ax_P.set_xticklabels([int(xtick+1) for xtick in ax_P.get_xticks()]) #+1 since the first node is called 1 and not 0
            ax_P.set_xlabel('Node number (of first {} nodes)'.format(3+s+n+m))
            ax_Q.set_xticklabels([int(xtick+1) for xtick in ax_Q.get_xticks()]) #+1 since the first node is called 1 and not 0
            ax_Q.set_xlabel('Node number (of first {} nodes)'.format(3+s+n+m))
            ax_Sk.set_xlabel('Link number (of first {} links)'.format(2+s+n+m))
            if not save_figs:
                ax_f.legend(handles=legend_handles)
                ax_LF_error.legend(handles=legend_handles)
                ax_Sk.legend(handles=legend_handles)
                ax_delta.legend(handles=legend_handles)
                ax_V.legend(handles=legend_handles)
                ax_P.legend(handles=legend_handles)
                ax_Q.legend(handles=legend_handles)
    if save_figs:
        fig_legend = plt.figure('legend_elec_OF')
        ax_legend = fig_legend.gca()
        ax_legend.axis('off')
        fig_legend.patch.set_visible(False)
        ax_legend.legend(handles=legend_handles,loc='center')

    print('\nError LF: {} after {} iters.'.format(err_vec_LF[-1],iters_LF))
    if len(x_sol_LF) < 10:
        elec_net_LF = update_bc(elec_net_LF,V2,P2)
        elec_net_LF.update(x_sol_LF)
        print('delta = {}'.format(delta_LF))
        print('|V| = {} kV'.format(V_LF/kV))
        print('P edge = {} MW'.format(P_edge_LF/MW))
        print('Q edge = {} MW'.format(Q_edge_LF/MW))
        print('S nodal inj = {} MW'.format(S_inj_LF/MW))
        print('P hl = {} MW'.format([hl.P/MW for node in elec_net_LF.get_nodes() for hl in node.get_half_links()]))
        print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net_LF.get_nodes() for hl in node.get_half_links()]))

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
        for bound in bounds:
            with open(os.path.join(path_to_tables,'elec_optimizer_info_forms_'+bound+'_'+scale_label+'_{}_{}_{}.txt'.format(n,m,s)), "w") as table:
                for form_OF in forms_OF:
                    if form_OF == 1:
                        form_label = r'$|S_k|^2$ no lim.'
                    elif form_OF == 2:
                        form_label = r'$|S_k|^2$ lim., not in $\vec{x^G}$'
                    elif form_OF == 3:
                        form_label = r'$|S_k|^2$ lim., in $\vec{x^G}$'
                    if form_OF == 3:
                        y_LF_sol = np.concatenate((y_LF[:5],Sk_squared,y_LF[5:]))
                    else:
                        y_LF_sol = y_LF
                    for approach in approaches:
                        if approach == 'eq_constr':
                            approach_label = 'eq. constr.'
                        else:
                            approach_label = approach
                        res_trust = result.get('trust-constr_'+str(form_OF)+'_'+bound+'_'+approach)
                        res_slsqp = result.get('SLSQP_'+str(form_OF)+'_'+bound+'_'+approach)
                        res_ipopt = result.get('ipopt_'+str(form_OF)+'_'+bound+'_'+approach)
                        y_trust = y_res.get('trust-constr_'+str(form_OF)+'_'+bound+'_'+approach)
                        y_slsqp = y_res.get('SLSQP_'+str(form_OF)+'_'+bound+'_'+approach)
                        y_ipopt = y_res.get('ipopt_'+str(form_OF)+'_'+bound+'_'+approach)
                        table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e}\\ '.format(form_label,approach_label,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,res_trust.execution_time,res_slsqp.execution_time,res_ipopt.execution_time,error(y_trust,y_LF_sol),error(y_slsqp,y_LF_sol),error(y_ipopt,y_LF_sol)))
                    table.write(r'\hline ')

    for bound in bounds:
        for form_OF in forms_OF:
            if form_OF == 3:
                y_LF_sol = np.concatenate((y_LF[:5],Sk_squared,y_LF[5:]))
            else:
                y_LF_sol = y_LF
            for approach in approaches:
                res_trust = result.get('trust-constr_'+str(form_OF)+'_'+bound+'_'+approach)
                res_slsqp = result.get('SLSQP_'+str(form_OF)+'_'+bound+'_'+approach)
                res_ipopt = result.get('ipopt_'+str(form_OF)+'_'+bound+'_'+approach)
                y_trust = y_res.get('trust-constr_'+str(form_OF)+'_'+bound+'_'+approach)
                y_slsqp = y_res.get('SLSQP_'+str(form_OF)+'_'+bound+'_'+approach)
                y_ipopt = y_res.get('ipopt_'+str(form_OF)+'_'+bound+'_'+approach)
                print('\nForm OF: {}, bounds: {}, approach: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nt-c:{}\nSLSQP: {}\nIPOPT: {}\nError for t-c:{:.4e}, SLSQP: {:.4e}, IPOPT:{:.4e}\nExec. time for t-c:{:.2f}, SLSQP: {:.2f}, IPOPT:{:.2f}'.format(form_OF,bound,approach,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(y_trust,y_LF_sol),error(y_slsqp,y_LF_sol),error(y_ipopt,y_LF_sol),res_trust.execution_time,res_slsqp.execution_time,res_ipopt.execution_time))

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # base case (3 nodes)
    n=0
    m=0
    s=0
    runs = 1#10
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,n=n,m=m,s=s,max_iter=50) # unscaled
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,scale_var='matrix',n=n,m=m,s=s,max_iter=50)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,scale_var='per_unit',n=n,m=m,s=s,max_iter=50)

    # medium case (30 nodes)
    n=5
    m=2
    s=3
    runs = 1#10
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,n=n,m=m,s=s,max_iter=50) # unscaled
    compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,scale_var='matrix',n=n,m=m,s=s,max_iter=50)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,scale_var='per_unit',n=n,m=m,s=s,max_iter=50)

    # medium / large case (163 nodes)
    n=10
    m=5
    s=10
    runs = 1#5
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,n=n,m=m,s=s,max_iter=50) # unscaled
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,scale_var='matrix',n=n,m=m,s=s,max_iter=50)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,scale_var='per_unit',n=n,m=m,s=s,max_iter=50)

    # large case (323 nodes)
    n=10
    m=5
    s=20
    runs = 1#5
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,scale_var='matrix',n=n,m=m,s=s,max_iter=50)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,number_of_runs=runs,scale_var='per_unit',n=n,m=m,s=s,max_iter=50)

    plt.show()
