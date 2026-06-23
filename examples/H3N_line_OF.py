"""Optimal power flow of a heat network with 3 nodes connected in a single line, with possible streets connected to the third node."""
from examples import H3N_line as HN
from examples import MES3N_streets as MES
from meslf.networks.heat_network import HeatLink
import warnings
import numpy as np
from meslf.utils.hide_print import HiddenPrints
from meslf.utils.constants import bar, MW, km, cm
from meslf.utils.post_processing import error, fexp, fman, exp_tex
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from meslf.load_flow.system_of_equations import NonLinearSystemHeat
import scipy.optimize as spo
import scipy.sparse as sps
import os
import sys
import time
import ipopt
import datetime
from decimal import Decimal

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

colors_method = {'trust-constr':'tab:blue','SLSQP':'tab:orange','ipopt':'tab:green'}
linestyles_approaches = {'eq_constr':'-','direct':':','adjoint':'-.'}
markers_forms = {1:'.',2:'*',3:'d',4:'s',5:'x'}
OF_forms_label = {1:r'terminal flow, $m_{i,l}$ unbounded',2:r'terminal flow, $m_{i,l}$ bounded',3:r'standard, $m_{i,l}$ unbounded',4:r'standard, $m_{i,l}$ bounded, in $x^G$',5:r'standard, $m_{i,l}$ bounded, not in $x^G$'}
marker_size = 10
legend_handles = [Line2D([0], [0], color=colors_method.get('trust-constr'), label='trust-constr'),
    Line2D([0], [0], color=colors_method.get('SLSQP'), label='SLSQP'),
    Line2D([0], [0], color=colors_method.get('ipopt'), label='IPOPT'),
    Line2D([0], [0], color='w',marker=markers_forms.get(1), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=OF_forms_label.get(1)),
    Line2D([0], [0], color='w',marker=markers_forms.get(2), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=OF_forms_label.get(2)),
    Line2D([0], [0], color='w',marker=markers_forms.get(3), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=OF_forms_label.get(3)),
    Line2D([0], [0], color='w',marker=markers_forms.get(4), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=OF_forms_label.get(4)),
    Line2D([0], [0], color='w',marker=markers_forms.get(5), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=OF_forms_label.get(5)),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('eq_constr'), label='LF as eq. constr.'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('direct'), label='Direct approach'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('adjoint'), label='Adjoint approach')]

def create_network(n=0,m=0,s=0,Ts1=100.,p1=9*bar,To2=100.,To3=50.,phi2=-1*MW,phi3=1.5*MW,L12 =4*km,L23=5*km,D=0.15,lam=0.2,link_type='standard_pipe_low_pres_pole',Lstreets=5*km,Dstreets=0.15,lamstreets=0.2):
    """Create heat network, with or without streets connected to it.

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
    heat_net : HeatNetwork
        The heat network
    """
    # base case (with node 2 a source node)
    heat_net = HN.create_network(Ts1=Ts1,p1=p1,To2=To2,To3=To3,phi2=phi2,phi3=phi3,L12 =L12,L23=L23,D=D,lam=lam,link_type=link_type,node_set=2,slack_node='source',source_node='outflow',sink_node='outflow')

    if s > 0: # connect streets to network
        heat_net.initialize() #THIS IS SOMEHOW NEEDED WHEN USING THE HALF LINK FLOW FORMULATION???!!! If I don't include this, NR diverges like crazy. Don't know why?? Initialize adds the half links to the heat network... Maybe it has to do with numbering? Or maybe with some half link value that is taken somewhere?

        hn3 = heat_net.nodes[2]
        water = heat_net.links[0].link_params.get('carrier')
        heat_streets = MES.create_streets_heat(n,m,s,water,Lstreets,Dstreets,lamstreets,phi3,heat_load='outflow')

        U = lamstreets/(np.pi*Dstreets) #[W/(m^2 K)]
        for heat_street in heat_streets:
            for node in heat_street.get_nodes():
                heat_net.add_node(node)
                for halflink in node.get_half_links():
                    heat_net.add_half_link(halflink)
            for link in heat_street.get_links():
                link.name += '_'+heat_street.name
                heat_net.add_link(link)
            heat_street_source = heat_street.nodes[0]
            hl = HeatLink('hl_'+heat_street_source.name,hn3,heat_street_source,link_type=link_type,link_params={'L':Lstreets,'D':Dstreets,'U':U,'carrier':water})
            heat_net.add_link(hl)
            heat_street_source.node_type = 2 # junction node
            for hl in heat_street_source.get_half_links():
                heat_street_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
        hn3.node_type = 2 # junction node
        for hl in hn3.get_half_links():
                hn3.remove_half_link(hl)
                heat_net.remove_half_link(hl)
    return heat_net

def set_x_LF_init(nlsys,n=0,m=0,s=0,m_init=10,Tr_sink_init=50,p_perc_high=.99,p_perc_low=.98,T_perc_high=.99,T_perc_low=.98):
    """Set the unscaled initial guess for the LF variables"""
    heat_net = nlsys.heatnetwork
    ph_ref = heat_net.nodes[0].p
    Ts_ref = heat_net.nodes[0].Ts
    ph_init = np.zeros(len(nlsys.unknown_p_nodes))
    ph_ind = 0
    for i in range(3):
        if heat_net.nodes[i] in nlsys.unknown_p_nodes: # heat node i
            ph_init[ph_ind] = ph_ref*(1-(1-p_perc_high)*(ph_ind+1)/4)
            ph_ind += 1
    Ts_init = np.zeros(len(nlsys.unknown_Ts_nodes))
    Ts_ind = 0
    for i in range(3):
        if heat_net.nodes[i] in nlsys.unknown_Ts_nodes: # heat node i
            Ts_init[Ts_ind] = Ts_ref*(1-(1-T_perc_high)*(Ts_ind+1)/4)
            Ts_ind += 1

    N = 2*n-m+1 #number of nodes per streets (+1 because of street source)
    NJ = n-m #number of junctions per street
    for num_S in range(s): # heat nodes per street
        # nodes are numbered such that the load come first, and then the junctions
        ph_init_street = ph_ref*np.linspace(p_perc_high,p_perc_low,N)
        ph_init[num_S*N+ph_ind ] = ph_init_street[0] #street source
        ph_init[num_S*N+ph_ind+1:num_S*N+n+ph_ind+1] = ph_init_street[NJ+1:N] #loads (lowest pressures)
        ph_init[num_S*N+n+ph_ind+1:num_S*N+N+ph_ind] = ph_init_street[1:NJ+1] #junctions (highest pressures)
        Ts_init_street = Ts_ref*np.linspace(T_perc_high,T_perc_low,N)
        Ts_init[num_S*N+Ts_ind] = Ts_init_street[0] #street source
        Ts_init[num_S*N+Ts_ind+1:num_S*N+n+Ts_ind+1] = Ts_init_street[NJ+1:N] #loads (lowest temperatures)
        Ts_init[num_S*N+n+Ts_ind+1:num_S*N+N+Ts_ind] = Ts_init_street[1:NJ+1] #junctions (highest temperatures)
    Tr_init = Tr_sink_init*np.ones(len(nlsys.unknown_Tr_nodes))

    if n*s > 0:
        m_sink_init = m_init/(n*s)
    else:
        m_sink_init = m_init
    m_init = np.concatenate((np.array([m_init,m_init]),m_sink_init*np.ones(len(nlsys.unknown_m_links)-2)))
    m_hl_init = m_sink_init*np.ones(len(nlsys.unknown_m_halflinks))
    if len(m_hl_init):
        m_hl_init[0] = -m_sink_init # node 2 is a source

    xh_init = np.concatenate((m_init,m_hl_init,ph_init,Ts_init,Tr_init)) # unscaled
    return xh_init

def xh_from_yopf(y,nlsys=None):
    """Returns the variables of steady-state LF, from the variables y of OF."""
    xh = y[-len(nlsys.x_entries):]
    return xh

def objective_function(y,y_ind,a=np.array([0,0]),b=np.array([.04,.05]),c=np.array([4e-4,4.5e-4]),scale_var=None,scale_var_params=None,fb=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    E : np array
        Array with flows used in objective. Gas flows are assumed to be in kg/s, active powers in W, and heat power in W. Scaled when per unit scaling is used, unscaled otherwise.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [dphi1,dphi2]. Scaled when per unit scaling is used, unscaled otherwise.
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
    # print('within obj: E = {}'.format(y[np.array(y_ind)]))
    f = np.sum(a+b*np.sign(y[np.array(y_ind)])*y[np.array(y_ind)]+c*y[np.array(y_ind)]**2)
    if scale_var == 'matrix':
        f *= (1/fb)
    return f

def grad_objective(y,y_ind,a=np.array([0,0]),b=np.array([.04,.05]),c=np.array([4e-4,4.5e-4]),scale_var=None,scale_var_params=None,fb=None,Dy_inv=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [dphi1,dphi2]. Scaled when per unit scaling is used, unscaled otherwise.
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

def hess_objective(y,y_ind,a=np.array([0,0]),b=np.array([.04,.05]),c=np.array([4e-4,4.5e-4]),scale_var=None,scale_var_params=None,fb=None,Dy_inv=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [dphi1,dphi2]. Scaled when per unit scaling is used, unscaled otherwise.
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
    nlsys : NonLinearSystemHeat
        Nonlinear system of the heat network. Constains the networks, information about scaling, etc.

    Returns
    -------
    h : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    network = nlsys.heatnetwork

    xh = xh_from_yopf(y,nlsys=nlsys)
    # evaluate load flow equations
    network = reset_network_without_update(network)
    F = nlsys.F(xh) # Also updates network. Should set correct values on links and halflinks

    xG = y[2:-len(xh)] # scaled when per unit scaling is used
    G = np.zeros(len(xG))
    # evaluate heat equation in slack node (use m1 determined by conservation of mass)
    m1 = xG[0] # scaled for p.u., unscaled for matrix
    dphi1 = xG[-1] # scaled for p.u., unscaled for matrix
    if nlsys.scale_var == 'per_unit':
        network.nodes[0].half_links[0].dphi = dphi1*nlsys.scale_var_params.get('phibase')
    else:
        network.nodes[0].half_links[0].dphi = dphi1
    G[-1] = network.nodes[0].half_links[0].heat_power_equation(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    # evaluate conservation of mass in slack node
    if nlsys.scale_var == 'per_unit':
        network.nodes[0].half_links[0].m = m1*nlsys.scale_var_params.get('mbase')
    else:
        network.nodes[0].half_links[0].m = m1
    G[0] = network.nodes[0].node_law(network=network,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    if len(xG) > 2:
        for ind_hl,hl in enumerate(network.get_half_links(bc_types=[2,3])):
            ind_G = ind_hl+1
            G[ind_G] = xG[ind_G] - hl.get_m(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) # the update in F(x) should have set the correct value for m
    h = np.concatenate((G,F)) # already scaled if per unit is used
    if nlsys.scale_var == 'matrix':
        h = Dh.dot(h)
    return h

def h_der(y, nlsys=None, Dy_inv=None, Dh=None):
    """First derivative of quality constraints h(x)=0. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeat
        Nonlinear system of the heat network. Constains the networks, information about scaling, etc.

    Returns
    -------
    dh_dy : np array
        The first derivative of the (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    len_h = Dh.shape[0]
    len_u = len(y) - len_h
    len_slack = 2 # number of unknowns in slack node
    dh_dy = np.zeros((len_h,len(y)))

    network = nlsys.heatnetwork
    network = reset_network_without_update(network)
    xh = xh_from_yopf(y,nlsys=nlsys)
    xG = y[len_u:-len(xh)]
    # determine indices
    N = len(network.nodes) # number of nodes in the network
    E = len(network.links) # number of links in the network
    T = len(network.half_links) # number of halflinks in the network.
    F_ind = nlsys.Fm + [N+ind for ind in nlsys.Fdeltap] + [N+E+ind for ind in nlsys.FTs] + [2*N+E+ind for ind in nlsys.FTr] + [3*N+E+ind for ind in nlsys.Fphi] + [3*N+E+T+ind for ind in nlsys.FdT]
    xlf_ind = nlsys.xm.copy()
    len_flows = E
    if len(nlsys.xmhl)>0:
        xlf_ind += [len_flows + ind for ind in nlsys.xmhl]
        len_flows += T
    xlf_ind += [len_flows + ind for ind in nlsys.xp] + [len_flows+N + ind for ind in nlsys.xTs] + [len_flows+2*N + ind for ind in nlsys.xTr] + [len_flows+3*N + ind for ind in nlsys.xTshl] + [len_flows+T+3*N + ind for ind in nlsys.xTrhl]
    # evaluate Jacobian of LF equations
    J_full = nlsys.J_dense(xh,return_full=True) #This also updates the network
    # derivative of slack conservation of mass to xF
    dh_dy[0,len_u+len(xG)] = -1 #dGm1_dm12
    dh_dy[len(xG):,:][:,len_u+len(xG):] = J_full[F_ind,:][:,xlf_ind] #J_lf
    # derivative of slack heat power equation to xF
    water = network.links[0].link_params.get('carrier')
    dT_source = network.nodes[0].half_links[0].get_Ts(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) - network.nodes[0].half_links[0].get_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    dh_dy[len(xG)-1,len_u+len(xG)] = water.get_Cp(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)*dT_source #dGphi1_dm12
    dh_dy[len(xG)-1,len_u+len(xG)+len(nlsys.xm) + len(nlsys.xmhl) + len(nlsys.xp) + len(nlsys.xTs)] = network.links[0].get_m(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)*water.get_Cp(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)#dGphi1_dTr1
    # derivatives of the half link flow equation in G to xF
    if len(xG) > 2:
        for ind_hl,hl in enumerate(network.get_half_links(bc_types=[2,3])):
            ind_G = ind_hl+1
            dm_dTs = hl.m_der_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            dm_dTr = hl.m_der_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            Ts_ind = len(nlsys.xm) + len(nlsys.xmhl) + len(nlsys.xp) + nlsys.unknown_Ts_nodes.index(hl.start_node)
            Tr_ind = len(nlsys.xm) + len(nlsys.xmhl) + len(nlsys.xp) + len(nlsys.xTs) + nlsys.unknown_Tr_nodes.index(hl.start_node)# index of Tr of start node within xF
            dh_dy[ind_G,len_u+len(xG)+Ts_ind] = dm_dTs
            dh_dy[ind_G,len_u+len(xG)+Tr_ind] = dm_dTr
    # derivatives of G and F to u (der. of G to u is 0)
    if nlsys.heat_formulation == 'half_link_flow':
        dh_dy[len(xG)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links)+len(nlsys.F_Ts_nodes)+len(nlsys.F_Tr_nodes),1] = -1 #dFphi2_dphi2
        Tshl2_ind = len_flows+3*N+1 # index of Ts2,0 within x for full jabocian
        dh_dy[len(xG):,0] = J_full[F_ind,:][:,Tshl2_ind].ravel() #dh_dTs2,0
    else:
        hl2 = network.nodes[1].half_links[0]
        Ts2, dphi2 = y[:len_u]
        Tr2 = network.nodes[1].get_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dT2 = Ts2 - hl2.get_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dm2_dphi2 = 1/(water.get_Cp(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)*dT2)
        dm2_dTs2hl = hl2.dm_dTs(dphi2,Ts2,Tr2,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dh_dy[len(xG),0] = -dm2_dTs2hl # dFm2_dTs2,0
        dh_dy[len(xG)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links),0] = dm2_dTs2hl*Ts2 + hl2.get_m(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)# dFTs2_dTs2,0
        dh_dy[len(xG)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links)+len(nlsys.F_Ts_nodes)+1,0] = -dm2_dTs2hl * Tr2# dFTr2_dTs2,0
        dh_dy[len(xG),1] = -dm2_dphi2 # dFm2_dphi2
        dh_dy[len(xG)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links),1] = dm2_dphi2 * Ts2# dFTs2_dphi2
        dh_dy[len(xG)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links)+len(nlsys.F_Ts_nodes)+1,1] = -dm2_dphi2 * Tr2# dFTr2_dphi2
        if len(xG) > len_slack:
            dh_dy[1,0] = -dm2_dTs2hl#dGm2_dTs2,0
            dh_dy[1,1] = -dm2_dphi2 #dGm2_dphi2
    # derivatives of G and F to xG (by definition, dF_dxG = 0)
    dh_dy[0,len_u] = -1 #dG0_dm1
    dh_dy[len(xG)-1,len_u+len(xG)-1] = -1 #dG_dphi1
    if len(xG) > len_slack: # in case of standard formulation, and half links flows in xG
        dh_dy[1:len(xG)-1,:][:,len_u+1:len_u+len(xG)-1] = np.eye(T-1) #dG_dmhl
    if nlsys.scale_var == 'matrix':
        dh_dy = Dh.dot(dh_dy.dot(Dy_inv))
    # plt.figure('Jacobian')
    # plt.spy(dh_dy)
    # plt.figure('Jacobian values')
    # jac_val = dh_dy.copy()
    # jac_val[jac_val == 0] = np.nan
    # plt.imshow(jac_val)
    # plt.plot([len_u-.5,len_u-.5],[0-.5,len_h-.5],'k')
    # plt.plot([len_u+len(xG)-.5,len_u+len(xG)-.5],[0-.5,len_h-.5],'k')
    # plt.plot([0-.5,len(y)-.5],[len(xG)-.5,len(xG)-.5],'k')
    # plt.show()
    return dh_dy

def gamma(y, nlsys=None, ineq_constr_lb=np.array([-10,0]), ineq_constr_ub=np.array([0,10])):
    """The nonlinear inequality constraints gamma(x)>=0 on the half link flow. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeat
        Nonlinear system of the heat network. Constains the networks, information about scaling, etc.
    ineq_constr_lb : np arrays
        Lower bound for the half link flows. Scaled when per unit or matrix scaling is used.
    ineq_constr_ub : np arrays
        Upper bound for the half link flows. Scaled when per unit or matrix scaling is used.

    Returns
    -------
    gam : np array
        The nonlinear inequality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    network = nlsys.heatnetwork
    network = reset_network_without_update(network)

    xh = xh_from_yopf(y,nlsys=nlsys)
    network.update(xh,formulation=nlsys.heat_formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    hl_flows = np.zeros(len(network.half_links)-1) #-1 for half link of slack node
    for ind_hl,hl in enumerate(network.get_half_links(bc_types=[2,3])):
        hl_flows[ind_hl] = hl.flow(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    return np.concatenate((hl_flows-ineq_constr_lb,ineq_constr_ub-hl_flows))

def gamma_der(y, nlsys=None, Dy_inv=None):
    """The nonlinear inequality constraints gamma(x)>=0 on the link powers. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeat
        Nonlinear system of the heat network. Constains the networks, information about scaling, etc.

    Returns
    -------
    gam : np array
        The nonlinear inequality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    network = nlsys.heatnetwork
    network = reset_network_without_update(network)
    len_u = 2

    xh = xh_from_yopf(y,nlsys=nlsys)
    network.update(xh,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)

    water = network.links[0].link_params.get('carrier')
    hl2 = network.nodes[1].half_links[0]
    Ts2, dphi2 = y[:len_u]
    Tr2 = network.nodes[1].get_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    dT2 = Ts2 - hl2.get_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    dm2_dphi2 = 1/(water.get_Cp(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)*dT2)
    dm2_dTs2 = hl2.dm_dTs(dphi2,Ts2,Tr2)

    T_loads = len(network.half_links)-1
    N_Ts = len(nlsys.unknown_Ts_nodes)
    N_Tr = len(nlsys.unknown_Tr_nodes)
    # derivatives to u
    dgamma_dy = np.zeros((2*T_loads,len(y)))
    dgamma_dy[0,0] = dm2_dTs2
    dgamma_dy[T_loads,0] = -dm2_dTs2
    dgamma_dy[0,1] = dm2_dphi2
    dgamma_dy[T_loads,1] = -dm2_dphi2
    # derivatives to xF
    dmhl_dx = np.zeros((T_loads,len(xh)))
    for ind_hl,hl in enumerate(network.get_half_links(bc_types=[2,3])):
        dm_dTs = hl.m_der_Ts(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dm_dTr = hl.m_der_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        Ts_ind = len(nlsys.xm) + len(nlsys.xmhl) + len(nlsys.xp) + nlsys.unknown_Ts_nodes.index(hl.start_node)
        Tr_ind = len(nlsys.xm) + len(nlsys.xmhl) + len(nlsys.xp) + len(nlsys.xTs) + nlsys.unknown_Tr_nodes.index(hl.start_node)# index of Tr of start node within xF
        dmhl_dx[ind_hl,Ts_ind] = dm_dTs
        dmhl_dx[ind_hl,Tr_ind] = dm_dTr
    dgamma_dy[:,-len(xh):] = np.vstack((np.eye(T_loads),-np.eye(T_loads))).dot(dmhl_dx)
    if nlsys.scale_var == 'matrix':
        dgamma_dy = np.diag(1/(nlsys.scale_var_params.get('mbase'))*np.ones(2*T_loads)).dot(dgamma_dy.dot(Dy_inv))
    # plt.figure('Jacobian gamma')
    # plt.spy(dgamma_dy)
    return dgamma_dy

def reset_network_without_update(heat_net):
    """Reset the heat network, but do not update the network values. That is, set the mass flow of slack nodes to zero."""
    heat_net.nodes[0].half_links[0].m = 0
    return heat_net

def update_bc(heat_net,To2,dphi2,scale_var=None,scale_var_params=None):
    """Update the boundary conditions of the gasn etwork, based on the control variables of OF"""
    if scale_var == 'per_unit':
        To2 = To2*scale_var_params.get('Tbase')
        dphi2 = dphi2*scale_var_params.get('phibase')
    heat_net.nodes[1].half_links[0].Ts = To2
    heat_net.nodes[1].half_links[0].dphi = dphi2 #<0
    return heat_net

def run_optimal_load_flow(u_lb=np.array([80,-1.5*MW]),u_ub=np.array([110,0]),u_init=np.array([100,-1*MW]),slack_lb=np.array([-3,-2*MW]),slack_ub=np.array([0,0]),slack_init_base=np.array([-1,-1.*MW]),xLF_lb=np.array([-10,-10,0,0,60,60,0,0,0]),xLF_ub=np.array([10,10,10*bar,10*bar,110,110,55,55,55]),m_init=10,Tr_sink_init=50,p_perc_high=.99,p_perc_low=.98,T_perc_high=.99,T_perc_low=.98,n=0,m=0,s=0,Ts1=100.,p1=9*bar,To3=50.,phi3=1.5*MW,L12 =4*km,L23=5*km,D=0.15,lam=0.2,link_type='standard_pipe_low_pres_pole',Lstreets=5*km,Dstreets=0.15,lamstreets=0.2,y_ind = [5,1],a=np.array([0,0]),b=np.array([.04,.05]),c=np.array([4e-4,4.5e-4]),halflinks_slack=False,halflinks_limit=False, ineq_constr_lb=np.array([-10,0]), ineq_constr_ub=np.array([0,10]),formulation='standard',scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,optimization_method='trust-constr',stay_within_bounds=False,fb=None,runs=1):
    """Run optimal load flow, with LF as equality constraints.

    Parameters
    ----------
    halflinks_slack : bool, optional
        Determines if the half links flows should be included as slack variables. This parameter is ignored if halflinks_limit is False and/or if half_link_flow formulation is used. Default is False.
    halflinks_limit : bool, optional
        Determines if limits are imposed on the half link flows. Default is False.

    Returns
    -------
    xh_opt : np array
        Solution of LF variables of OF. Unscaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_times : float
        List with time of the optimization (excluding creation of network etc.).
    """
    print('\nRunning OPF for H3N line with LF as eq. constr. (half link flow limit: {}, half link flows as slack: {}, formulations: {}, hard bounds: {}, method: {}, scaling: {})'.format(halflinks_slack,halflinks_limit,formulation,stay_within_bounds,optimization_method,scale_var))

    # create network
    To2, dphi2 = u_init
    heat_net = create_network(n=n,m=m,s=s,Ts1=Ts1,p1=p1,To2=To2,To3=To3,phi2=dphi2,phi3=phi3,L12 =L12,L23=L23,D=D,lam=lam,link_type=link_type,Lstreets=Lstreets,Dstreets=Dstreets,lamstreets=lamstreets)

    # set initial guess and initialize network, using default initial guess
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    xLF_init = set_x_LF_init(nlsys,n=n,m=m,s=s,p_perc_high=p_perc_high,p_perc_low=p_perc_low,T_perc_high=T_perc_high,T_perc_low=T_perc_low) # unscaled
    T_loads = len(list(heat_net.get_half_links(bc_types=[2,3])))
    if formulation=='standard' and halflinks_limit and halflinks_slack:
        if s>0:
            halflink_flows = np.zeros(T_loads)
            for ind, hl in enumerate(heat_net.get_half_links(bc_types=[2,3])):
                if hl.sink:
                    halflink_flows[ind] = m_init/T_loads
                else: # only node 2 is a source
                    halflink_flows[ind] = -m_init
        else:
            halflink_flows = np.array([-m_init,m_init])
        slack_init = np.concatenate(([slack_init_base[0]],halflink_flows,[slack_init_base[-1]]))
    else:
        slack_init = slack_init_base
    # initial guess for OF (unscaled)
    y0 = np.concatenate((u_init,slack_init,xLF_init))

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('Tbase'), scale_var_params.get('phibase')])
        slack_base = np.array([scale_var_params.get('mbase'), scale_var_params.get('phibase')])
        G_base = np.array([scale_var_params.get('mbase'), scale_var_params.get('phibase')])
        if formulation=='standard' and halflinks_limit and halflinks_slack:
            slack_base = np.concatenate(([slack_base[0]],scale_var_params.get('mbase')*np.ones(T_loads),[slack_base[1]]))
            G_base = np.concatenate(([G_base[0]],scale_var_params.get('mbase')*np.ones(T_loads),[G_base[1]]))
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
        network = nlsys.heatnetwork
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        To2, dphi2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network = update_bc(network, To2,dphi2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        H = h(y, nlsys=nlsys, Dh=Dh)
        global err_LF_vec_global
        F = H[len(u_init)+len(slack_init):]
        err_LF_vec_global.append(np.linalg.norm(F))
        # print('dim h: {}'.format(H.shape))
        return H

    def jac_eq_constr(y,nlsys=nlsys, Dy_inv=Dy_inv, Dh=Dh):
        network = nlsys.heatnetwork
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        To2, dphi2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network = update_bc(network, To2,dphi2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dh_dy = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh)
        return dh_dy

    lb_nleq = np.zeros(len(slack_lb)+len(xLF_lb))
    ub_nleq = np.zeros(len(slack_ub)+len(xLF_ub))
    if optimization_method == 'trust-constr':
        nonlinear_constraint = spo.NonlinearConstraint(eq_constr,lb_nleq,ub_nleq,jac=jac_eq_constr,keep_feasible=stay_within_bounds)
    elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        nonlinear_constraint = {'type':'eq','fun':eq_constr,'jac':jac_eq_constr}

    # define nonlinear inequality constraints on half link flows
    if formulation == 'standard' and halflinks_limit and (not halflinks_slack):
        def g(y,nlsys=nlsys, ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub, Dy_inv=Dy_inv):
            network = nlsys.heatnetwork
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                y = Dy_inv.dot(y)
            # update bc of the network
            To2, dphi2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
            network = update_bc(network, To2,dphi2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            gam = gamma(y,nlsys=nlsys, ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub)
            return gam
        def g_jac(y,nlsys=nlsys, Dy_inv=Dy_inv):
            network = nlsys.heatnetwork
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                y = Dy_inv.dot(y)
            # update bc of the network
            To2, dphi2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
            network = update_bc(network, To2,dphi2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            dgamma_dy = gamma_der(y, nlsys=nlsys, Dy_inv=Dy_inv)
            return dgamma_dy
        if optimization_method == 'trust-constr':
            ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(2*len(ineq_constr_ub)),np.inf*np.ones(2*len(ineq_constr_ub)),jac=g_jac,keep_feasible=stay_within_bounds)
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
    if (not halflinks_limit) and formulation == 'half_link_flow':
        # half links flow are part of xF by default, but limits should not be imposed
        lb_ineq[len(u_lb)+len(slack_init)+len(nlsys.unknown_m_links):len(u_lb)+len(slack_init)+len(nlsys.unknown_m_links)+len(nlsys.unknown_m_halflinks)] = -np.inf
        ub_ineq[len(u_lb)+len(slack_init)+len(nlsys.unknown_m_links):len(u_lb)+len(slack_init)+len(nlsys.unknown_m_links)+len(nlsys.unknown_m_halflinks)] = np.inf

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
                heat_net = reset_network_without_update(heat_net)
                To2_init, dphi2_init = u_init # unscaled
                heat_net = update_bc(heat_net, To2_init, dphi2_init)
                heat_net.update(xLF_init,formulation=formulation) # unscaled
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
                print('Exception made for {}, formulation: {}, halflink flows limit: {}, haflink flows as slack: {}, hard bounds: {}, scaling: {}'.format(optimization_method,formulation,halflinks_slack,halflinks_limit,stay_within_bounds,scale_var))
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
    To2, dphi2 = y_opf[:len(u_init)] # unscaled
    heat_net = update_bc(heat_net, To2,dphi2)
    xh_opt = xh_from_yopf(y_opf,nlsys=nlsys) #unscaled
    print('Solution OF (success: {})'.format(res.success))
    if len(xh_opt) < 20:
        water = heat_net.links[0].link_params.get('carrier')
        rho_w = water.rhon
        grav_const = water.g
        heat_net = reset_network_without_update(heat_net)
        m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(xh_opt,formulation=formulation)
        print('p heat = {} m'.format(p_vec/(rho_w*grav_const)))
        print('m = {} kg/s'.format(m_vec))
        print('Ts = {} C'.format(Ts_vec))
        print('Tr = {} C'.format(Tr_vec))
        print('m hl = {} kg/s'.format(m_hl_vec))
        print('Ts hl = {}'.format(Ts_hl_vec))
        print('Tr hl = {}'.format(Tr_hl_vec))
        print('phi hl = {}'.format(phi_hl_vec))
        print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
    return xh_opt, res, f_vec, err_LF_vec, execution_times

def get_u_from_net(heat_net,scale_var=None,scale_var_params=None):
    """Get the current values of the control variables u from the network.
    """
    To2 = heat_net.nodes[1].half_links[0].get_Ts(scale_var=scale_var,scale_var_params=scale_var_params)
    dphi2 = heat_net.nodes[1].half_links[0].get_dphi(scale_var=scale_var,scale_var_params=scale_var_params)
    return np.array([To2,dphi2])

def solve_lf_in_of(u,nlsys,max_iters=10,tol=1e-6,xLF_init=np.array([1,1,8*bar,7*bar,100,100,50,50,50]),halflinks_slack=False):
    """Solve steady-state LF within an optimization context.
    """
    global err_LF_vec_global
    network = nlsys.heatnetwork
    u_net = get_u_from_net(network,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    if len(err_LF_vec_global)==0 or (np.linalg.norm(u-u_net) > tol) or (err_LF_vec_global[-1] > tol):
        To2, dphi2 = u
        network = reset_network_without_update(network)
        network = update_bc(network, To2,dphi2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        with HiddenPrints():
            xh,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = network.solve_network(tol,max_iters,solver='NR',formulation=nlsys.heat_formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            if err_vec[-1] >= tol:
                network = reset_network_without_update(network)
                network.update(xLF_init,formulation=nlsys.heat_formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
                xh,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = network.solve_network(tol,max_iters,solver='NR',formulation=nlsys.heat_formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        err_LF = err_vec[-1]
    else:
        network = reset_network_without_update(network)
        xh = network.set_x_init(formulation=nlsys.heat_formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = network.update_full(xh,formulation=nlsys.heat_formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        err_LF = err_LF_vec_global[-1]
    err_LF_vec_global.append(err_LF)
    phi1 = network.nodes[0].half_links[0].get_dphi(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    m1 = network.nodes[0].half_links[0].get_m(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    if nlsys.heat_formulation == 'standard' and halflinks_slack:
        hl_flows = np.zeros(len(network.half_links)-1) #-1 for half link of slack node
        for ind_hl,hl in enumerate(network.get_half_links(bc_types=[2,3])):
            hl_flows[ind_hl] = hl.flow(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        slack = np.concatenate(([m1],hl_flows,[phi1]))
    else:
        slack = np.array([m1,phi1])
    x = np.concatenate((slack,xh)) # unscaled, unless per unit scaling is used
    return x

def run_optimal_load_flow_separate_LF(u_lb=np.array([80,-1.5*MW]),u_ub=np.array([110,0]),u_init=np.array([100,-1*MW]),slack_lb=np.array([-3,-2*MW]),slack_ub=np.array([0,0]),slack_init_base=np.array([-1,-1.*MW]),xLF_lb=np.array([-10,-10,0,0,60,60,0,0,0]),xLF_ub=np.array([10,10,10*bar,10*bar,110,110,55,55,55]),m_init=10,Tr_sink_init=50,p_perc_high=.99,p_perc_low=.98,T_perc_high=.99,T_perc_low=.98,n=0,m=0,s=0,Ts1=100.,p1=9*bar,To3=50.,phi3=1.5*MW,L12 =4*km,L23=5*km,D=0.15,lam=0.2,link_type='standard_pipe_low_pres_pole',Lstreets=5*km,Dstreets=0.15,lamstreets=0.2,y_ind = [5,1],a=np.array([0,0]),b=np.array([.04,.05]),c=np.array([4e-4,4.5e-4]),halflinks_slack=False,halflinks_limit=False, ineq_constr_lb=np.array([-10,0]), ineq_constr_ub=np.array([0,10]),formulation='standard',scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,max_iters_lf=10,approach='direct',optimization_method='trust-constr',stay_within_bounds=False,fb=None,runs=1):
    """Run optimal load flow, with LF included implicitely.

    Parameters
    ----------
    halflinks_slack : bool, optional
        Determines if the half links flows should be included as slack variables. This parameter is ignored if halflinks_limit is False and/or if half_link_flow formulation is used. Default is False.
    halflinks_limit : bool, optional
        Determines if limits are imposed on the half link flows. Default is False.

    Returns
    -------
    xh_opt : np array
        Solution of LF variables of OF. Unscaled.
    y_opt : np array
        Full solution of OF. That is, control variables and all state variables. Scaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_times : float
        List with time of the optimization (excluding creation of network etc.).
    """
    print('\nRunning OPF for H3N line with separate LF (half link flow limit: {}, half link flows as slack: {}, formulations: {}, hard bounds: {}, method: {}, scaling: {}, approach: {})'.format(halflinks_slack,halflinks_limit,formulation,stay_within_bounds,optimization_method,scale_var,approach))

    # create network
    To2, dphi2 = u_init
    heat_net = create_network(n=n,m=m,s=s,Ts1=Ts1,p1=p1,To2=To2,To3=To3,phi2=dphi2,phi3=phi3,L12 =L12,L23=L23,D=D,lam=lam,link_type=link_type,Lstreets=Lstreets,Dstreets=Dstreets,lamstreets=lamstreets)

    # set initial guess and initialize network, using default initial guess
    heat_net.initialize()
    nlsys = NonLinearSystemHeat(heat_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    xLF_init = set_x_LF_init(nlsys,n=n,m=m,s=s,p_perc_high=p_perc_high,p_perc_low=p_perc_low,T_perc_high=T_perc_high,T_perc_low=T_perc_low) # unscaled
    T_loads = len(list(heat_net.get_half_links(bc_types=[2,3])))
    if formulation=='standard' and halflinks_limit and halflinks_slack:
        if s>0:
            halflink_flows = np.zeros(T_loads)
            for ind, hl in enumerate(heat_net.get_half_links(bc_types=[2,3])):
                if hl.sink:
                    halflink_flows[ind] = m_init/T_loads
                else: # only node 2 is a source
                    halflink_flows[ind] = -m_init
        else:
            halflink_flows = np.array([-m_init,m_init])
        slack_init = np.concatenate(([slack_init_base[0]],halflink_flows,[slack_init_base[-1]]))
    else:
        slack_init = slack_init_base
    # initial guess for OF (unscaled)
    u0 = u_init

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('Tbase'), scale_var_params.get('phibase')])
        slack_base = np.array([scale_var_params.get('mbase'), scale_var_params.get('phibase')])
        G_base = np.array([scale_var_params.get('mbase'), scale_var_params.get('phibase')])
        if formulation=='standard' and halflinks_limit and halflinks_slack:
            slack_base = np.concatenate(([slack_base[0]],scale_var_params.get('mbase')*np.ones(T_loads),[slack_base[1]]))
            G_base = np.concatenate(([G_base[0]],scale_var_params.get('mbase')*np.ones(T_loads),[G_base[1]]))
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
    if (not halflinks_limit) and formulation == 'half_link_flow':
        # half links flow are part of xF by default, but limits should not be imposed
        lb_ineq_state = np.delete(lb_ineq_state,[len(slack_init)+len(nlsys.unknown_m_links)+ind_hl for ind_hl in range(len(nlsys.unknown_m_halflinks))])
        ub_ineq_state = np.delete(ub_ineq_state,[len(slack_init)+len(nlsys.unknown_m_links)+ind_hl for ind_hl in range(len(nlsys.unknown_m_halflinks))])

    # define objective function
    def obj(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb,xLF_init=xLF_init,halflinks_slack=halflinks_slack):
        global u_f_vec
        u_f_vec = u.copy()
        global f_vec_global
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,halflinks_slack=halflinks_slack)
        y = np.concatenate((u,x))
        f = objective_function(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    def obj_grad(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb,xLF_init=xLF_init,halflinks_slack=halflinks_slack,Dy_inv=Dy_inv,Dh=Dh,method=approach):
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,halflinks_slack=halflinks_slack)
        y = np.concatenate((u,x))
        # partial derivatives of objective
        deltaf_deltay = grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb,Dy_inv=Dy_inv)
        deltaf_deltau = np.zeros((1,len(u)))
        deltaf_deltax = np.zeros((1,len(x)))
        deltaf_deltau[0,:] = deltaf_deltay[:len(u)]
        deltaf_deltax[0,:] = deltaf_deltay[len(u):]
        # partial derivatives of equatilty constraints / load-flow equations
        To2, dphi2 = u # scaled for p.u., unscaled for matrix
        network = nlsys.heatnetwork
        network = update_bc(network, To2,dphi2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
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
    def g(u,nlsys=nlsys, lb_ineq_state=lb_ineq_state, ub_ineq_state=ub_ineq_state, ineq_constr_ub=ineq_constr_ub, halflinks_slack=halflinks_slack, halflinks_limit = halflinks_limit, Dy=Dy, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init):
        if nlsys.scale_var == 'matrix':
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,halflinks_slack=halflinks_slack)
        y = np.concatenate((u,x))
        if scale_var == 'matrix': # lb_ineq_state and ub_ineq_state are scaled, so scale x as well
            x = Dy[len(u):,len(u):].dot(x)
        if nlsys.heat_formulation == 'half_link_flow' and (not halflinks_limit):
            x = np.delete(x,[len(nlsys.unknown_m_links)+ind_hl for ind_hl in range(len(nlsys.unknown_m_halflinks))])
        g = np.concatenate((x-lb_ineq_state,ub_ineq_state-x))
        if nlsys.heat_formulation == 'standard' and halflinks_limit and (not halflinks_slack):
            To2, dphi2 = u # scaled for p.u., unscaled for matrix
            network = nlsys.heatnetwork
            network = update_bc(network, To2,dphi2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            deltah_deltay = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh)
            gam = gamma(y,nlsys=nlsys,ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub)
            g = np.concatenate((gam,g))
        return g
    def g_jac(u,nlsys=nlsys, halflinks_slack=halflinks_slack, halflinks_limit = halflinks_limit, Dy=Dy, Dh=Dh, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init,method=approach):
        if nlsys.scale_var == 'matrix':
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,halflinks_slack=halflinks_slack)
        y = np.concatenate((u,x))
        # Jacobian of inequality constraints on state variables (without gamma, that part is added later)
        len_x = len(x)
        I = np.eye(len_x)
        if nlsys.heat_formulation == 'half_link_flow' and (not halflinks_limit):
            I = np.delete(I,[len(nlsys.unknown_m_links)+ind_hl for ind_hl in range(len(nlsys.unknown_m_halflinks))],0)

        deltag_deltax = np.vstack((I,-I))
        deltag_deltau = np.zeros((deltag_deltax.shape[0],len(u)))
        # partial derivatives of equatilty constraints / load-flow equations
        To2, dphi2 = u # scaled for p.u., unscaled for matrix
        network = nlsys.heatnetwork
        network = update_bc(network, To2,dphi2, scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        deltah_deltay = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh)
        deltah_deltau = deltah_deltay[:,:len(u)]
        deltah_deltax = deltah_deltay[:,len(u):]
        if nlsys.heat_formulation == 'standard' and halflinks_limit and (not halflinks_slack):
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
        if formulation == 'standard' and halflinks_limit and (not halflinks_slack):
            len_g += len(ineq_constr_lb) + len(ineq_constr_ub)
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
                heat_net = reset_network_without_update(heat_net)
                To2_init, dphi2_init = u_init # unscaled
                heat_net = update_bc(heat_net, To2_init, dphi2_init)
                heat_net.update(xLF_init,formulation=formulation) # unscaled
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
                print('Exception made for {} (formulation: {}, halflink flows limit: {}, haflink flows as slack: {}, hard bounds: {}, scaling: {}, approach: {})'.format(optimization_method,formulation,halflinks_slack,halflinks_limit,stay_within_bounds,scale_var,approach))
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
        x = solve_lf_in_of(res.x,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,halflinks_slack=halflinks_slack)
        y_opt = Dy_inv.dot(np.concatenate((res.x,x))) # unscaled
    else:
        x = solve_lf_in_of(u_opf,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init,halflinks_slack=halflinks_slack)
        y_opt = np.concatenate((u_opf,x)) # unscaled
    xh_opt = xh_from_yopf(y_opt,nlsys=nlsys) #unscaled
    print('Solution OF (success: {})'.format(res.success))
    if len(xh_opt) < 20:
        water = heat_net.links[0].link_params.get('carrier')
        rho_w = water.rhon
        grav_const = water.g
        heat_net = reset_network_without_update(heat_net)
        m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(xh_opt,formulation=formulation)
        print('p heat = {} m'.format(p_vec/(rho_w*grav_const)))
        print('m = {} kg/s'.format(m_vec))
        print('Ts = {} C'.format(Ts_vec))
        print('Tr = {} C'.format(Tr_vec))
        print('m hl = {} kg/s'.format(m_hl_vec))
        print('Ts hl = {}'.format(Ts_hl_vec))
        print('Tr hl = {}'.format(Tr_hl_vec))
        print('phi hl = {}'.format(phi_hl_vec))
        print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
    if scale_var == 'matrix' or scale_var == 'per_unit':
        y_opt = Dy.dot(y_opt) # scaled
    return xh_opt, y_opt, res, f_vec, err_LF_vec, execution_times

def compare_forms(dir_path=None,save_tables=False,save_figs=False,save_CPU_times=False,number_of_runs=1,n=0,m=0,s=0,N_max = 100,max_iter=25,scale_var=None):
    """Compare OF for different optimization methods, formulations of OF, scalings, and bounds."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    max_iters_lf = 20
    tol = 1e-6

    pbase = 1*bar
    mbase = 1.
    Tbase = 100.
    phibase = 1*MW
    # scaling
    if scale_var == None:
        scale_var_params = None
        fb = None
        scale_label = 'unscaled'
    else:
        scale_var_params = {'phbase':pbase,'pbase':pbase,'mbase':mbase,'qbase':mbase,'Tbase':Tbase,'phibase':phibase}
        fb = 1e7
        if scale_var == 'matrix':
            scale_label = 'matrix'
        else:
            scale_label = 'pu'

    # parameter values for the objective function
    a=np.array([0,0])
    b=np.array([.04,.05])
    c=np.array([4e-4,4.5e-4])

    # Network parameters and boundary conditions (of LF)
    Ts1=100.
    p1=9*bar
    To3=50.
    L12 =4*km
    L23=5*km
    link_type='standard_pipe_low_pres_pole'
    if s == 0:
        To2=84.346642#84.82293161333068#100.
        phi2=-1.00175442*MW#-1.0469065288680548*MW#-1*MW
        phi3=1.5*MW
        D=15*cm
        lam=0.2
    else:
        To2=84.346642#84.82293161333068#.988*Ts1
        phi2=-1.00175442*MW#-1.0469065288680548*MW#-1*MW
        phi3=5.5*1.0469065288680548*MW#5.5*-phi2 #[W]
        D=30*cm
        lam=0.002
    Dstreets=D
    Lstreets=L23
    lamstreets=lam

    # initial guess
    if (n == 1 and m == 0) or n==m:
        p_perc_high = .99
        p_perc_low = .98
        T_perc_high = 1.
        T_perc_low = 1.
        u_init = np.array([100,-1*MW])
    else:
        p_perc_step = .001/(2*n-m)
        p_perc_high = min(.99,1-p_perc_step)
        p_per_low = min(.97,1-p_perc_step*(n-m))
        p_perc_low = max(.1,p_per_low)
        T_perc_step = .001/(2*n-m)
        T_perc_high = 1-T_perc_step#1.
        T_perc_low = max(.5,1-T_perc_step*(n-m))
        u_init=np.array([1.05*To2,1.19*phi2])
    m_init=10
    Tr_sink_init=50

    # steady-state LF solution
    with HiddenPrints():
        formulation = 'half_link_flow'
        heat_net_LF = create_network(n=n,m=m,s=s,Ts1=Ts1,p1=p1,To2=To2,To3=To3,phi2=phi2,phi3=phi3,L12 =L12,L23=L23,D=D,lam=lam,link_type=link_type,Lstreets=Lstreets,Dstreets=Dstreets,lamstreets=lamstreets)
        heat_net_LF.initialize()
        nlsys_LF = NonLinearSystemHeat(heat_net_LF,formulation=formulation)
        xh_init = set_x_LF_init(nlsys_LF,n=n,m=m,s=s,m_init=m_init,Tr_sink_init=Tr_sink_init,p_perc_high=p_perc_high,p_perc_low=p_perc_low,T_perc_high=T_perc_high,T_perc_low=T_perc_low)
        heat_net_LF.update(xh_init,formulation=formulation)
        x_sol_LF,iters_LF,err_vec_LF,m_vec_LF,p_vec_LF,Ts_vec_LF,Tr_vec_LF,m_hl_vec_LF,phi_hl_vec_LF,Ts_hl_vec_LF,Tr_hl_vec_LF = heat_net_LF.solve_network(tol,max_iters_lf,solver='NR',formulation=formulation)
        marker_LF = markers_forms.get(1)
        legend_label_LF = 'terminal link, unscaled'
        print('LF {}: error = {:.3e} after {} iters.'.format(legend_label_LF,err_vec_LF[-1],iters_LF))
        if (err_vec_LF[-1] > tol) and (scale_var != None): # solve using the same scaling as OF, in case NR did not converge
            scale_var_params = {'phbase':pbase,'pbase':pbase,'mbase':mbase,'qbase':mbase,'Tbase':Tbase,'phibase':phibase}
            heat_net_LF.reset_network(xh_init,formulation=formulation)
            x_sol_LF,iters_LF,err_vec_LF,m_vec_LF,p_vec_LF,Ts_vec_LF,Tr_vec_LF,m_hl_vec_LF,phi_hl_vec_LF,Ts_hl_vec_LF,Tr_hl_vec_LF = heat_net_LF.solve_network(tol,max_iters_lf,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            legend_label_LF = 'terminal link, scaled'
            print('LF {}: error = {:.3e} after {} iters.'.format(legend_label_LF,err_vec_LF[-1],iters_LF))
        if (err_vec_LF[-1] > tol) and formulation == 'half_link_flow':
            heat_net_LF.reset_network(xh_init,formulation=formulation)
            _,iters_LF,err_vec_LF,m_vec_LF,p_vec_LF,Ts_vec_LF,Tr_vec_LF,m_hl_vec_LF,phi_hl_vec_LF,Ts_hl_vec_LF,Tr_hl_vec_LF = heat_net_LF.solve_network(tol,max_iters_lf,solver='NR',formulation='standard',scale_var=scale_var,scale_var_params=scale_var_params)
            x_sol_LF = heat_net_LF.set_x_init(formulation=formulation)
            marker_LF = markers_forms.get(3)
            if scale_var == None:
                legend_label_LF = 'standard, unscaled'
            else:
                legend_label_LF = 'standard, scaled'
            print('LF {}: error = {:.3e} after {} iters.'.format(legend_label_LF,err_vec_LF[-1],iters_LF))

    phi1 = heat_net_LF.nodes[0].half_links[0].get_dphi()
    m1 = heat_net_LF.nodes[0].half_links[0].get_m()
    if n == 0:
        T_loads = 2
        mhl_LF = np.array([m_hl for ind, m_hl in enumerate(m_hl_vec_LF) if ind > 0]).ravel() # half link flow solution of LF, except at slack node
    else:
        T_loads = n*s+1
        mhl_LF = np.zeros(T_loads) # half link flow solution of LF, except at slack node
        for ind, hl in enumerate(heat_net_LF.get_half_links(bc_types=[2,3])):
            mhl_LF[ind] = hl.get_m()
    y_ind = [3,1]
    y_LF = np.concatenate((np.array([To2,phi2,m1,phi1]),x_sol_LF))

    # bounds. Based on LF solution, so LF has to solve!!!
    E = len(heat_net_LF.links)
    N = len(heat_net_LF.nodes)
    if n == 0:
        m_max = 10
        Ts_max = 110
        Ts_min = 60
        Tr_max = 55
        Tr_min = 0
        slack_lb_base=np.array([-3,-2*MW])
        slack_init_base=np.array([-1,-1.*MW])
    else:
        m_max = 1.1*heat_net_LF.links[1].get_m()
        Ts_max = 1.1*np.max(Ts_vec_LF)
        Tr_min = 0
        Tr_max = 1*np.max(Tr_vec_LF)
        Ts_min = 1.1*Tr_max
        slack_lb_base=np.array([1.5*m1,1.5*phi1])
        slack_init_base = np.array([1.1*m_init,1.3*phi1])
    mhl_lb = np.zeros(T_loads)
    mhl_lb[0] = -m_max
    mhl_ub = m_max*np.ones(T_loads)#10*np.ones(T_loads)
    mhl_ub[0] = 0.1
    ineq_constr_lb = mhl_lb.copy()
    ineq_constr_ub = mhl_ub.copy()

    slack_ub_base=np.array([-0.1,-0.1*MW])
    u_lb=np.array([.9*To2,1.2*phi2])#np.array([80,-1.5*MW])
    u_ub=np.array([1.1*To2,.9*phi2])#np.array([110,0])
    xLF_lb_standard=np.concatenate((-m_max*np.ones(E),np.zeros(N-1),Ts_min*np.ones(N-1),Tr_min*np.ones(N)))
    xLF_ub_standard=np.concatenate((m_max*np.ones(E),10*bar*np.ones(N-1),Ts_max*np.ones(N-1),Tr_max*np.ones(N)))

    # scale LF solution (assuming half_link_flow formulation)
    if scale_var != None:
        nlsys_LF = NonLinearSystemHeat(heat_net_LF,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        Dx = nlsys_LF.Dx()
        ubase = np.array([scale_var_params.get('Tbase'), scale_var_params.get('phibase')])
        slack_base = np.array([scale_var_params.get('mbase'), scale_var_params.get('phibase')])
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        f_LF = objective_function(y_LF,y_ind,a=a,b=b,c=c)/fb #scaled
        y_LF = Dy.dot(y_LF) # scaled
        mhl_LF /= scale_var_params.get('mbase')
    else:
        f_LF = objective_function(y_LF,y_ind,a=a,b=b,c=c,scale_var=scale_var,fb=fb)

    result = dict()
    y_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft']#, 'hard']
    approaches = ['eq_constr','direct','adjoint']
    forms_OF = [1,2,3,4,5] #[half_link_flow & no limits on m_i,l; half_link_flow & limits on m_i,l; standard & no limits on m_i,l; standard & limits on m_i,l & m_i,l in xG; standard & limits on m_i,l & m_i,l not in xG]
    total_cases = len(methods)*len(bounds)*len(approaches)*len(forms_OF)
    case_number = 1
    start_wall_clock = datetime.datetime.now().time()
    for bound in bounds:
        if bound == 'soft':
            stay_within_bounds = False
        else:
            stay_within_bounds = True
        for form_OF in forms_OF:
            fig_f = plt.figure('heat_obj_'+str(form_OF)+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
            ax_f = fig_f.gca()
            ax_f.set_xlabel('Iteration?')
            ax_f.set_ylabel('f')

            fig_LF_error = plt.figure('heat_error_LF_in_OF_'+str(form_OF)+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
            ax_LF_error = fig_LF_error.gca()
            ax_LF_error.set_xlabel('Iteration?')
            ax_LF_error.set_ylabel(r'$||F||_2$')

            max_fev = 0
            slack_lb = slack_lb_base.copy()
            slack_ub = slack_ub_base.copy()
            if form_OF == 1:
                formulation = 'half_link_flow'
                halflinks_limit = False
                halflinks_slack = False
                y_ind = [3,1]
            elif form_OF == 2:
                formulation = 'half_link_flow'
                halflinks_limit = True
                halflinks_slack = False
                y_ind = [3,1]
            elif form_OF == 3:
                formulation = 'standard'
                halflinks_limit = False
                halflinks_slack = False
                y_ind = [3,1]
            elif form_OF == 4:
                formulation = 'standard'
                halflinks_limit = True
                halflinks_slack = True
                y_ind = [3+len(mhl_lb),1]
                slack_lb = np.concatenate(([slack_lb[0]],mhl_lb,[slack_lb[-1]]))
                slack_ub = np.concatenate(([slack_ub[0]],mhl_ub,[slack_ub[-1]]))
            elif form_OF == 5:
                formulation = 'standard'
                halflinks_limit = True
                halflinks_slack = False
                y_ind = [3,1]
            if formulation == 'half_link_flow':
                xLF_lb = np.concatenate((xLF_lb_standard[:E],mhl_lb,xLF_lb_standard[E:]))
                xLF_ub = np.concatenate((xLF_ub_standard[:E],mhl_ub,xLF_ub_standard[E:]))
            else:
                xLF_lb = xLF_lb_standard.copy()
                xLF_ub = xLF_ub_standard.copy()
            for method in methods:
                for approach in approaches:
                    if approach == 'direct' or approach == 'adjoint':
                        xh_opt, y_opt, res, f_vec, err_LF_vec, execution_times = run_optimal_load_flow_separate_LF(u_lb=u_lb,u_ub=u_ub,u_init=u_init,slack_lb=slack_lb,slack_ub=slack_ub,slack_init_base=slack_init_base,xLF_lb=xLF_lb,xLF_ub=xLF_ub,m_init=m_init,Tr_sink_init=Tr_sink_init,p_perc_high=p_perc_high,p_perc_low=p_perc_low,T_perc_high=T_perc_high,T_perc_low=T_perc_low,n=n,m=m,s=s,Ts1=Ts1,p1=p1,To3=To3,phi3=phi3,L12 =L12,L23=L23,D=D,lam=lam,link_type=link_type,Lstreets=Lstreets,Dstreets=Dstreets,lamstreets=lamstreets,y_ind=y_ind,a=a,b=b,c=c,halflinks_slack=halflinks_slack,halflinks_limit=halflinks_limit, ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,approach=approach,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,runs=number_of_runs)
                    else:
                        xh_opt, res, f_vec, err_LF_vec, execution_times = run_optimal_load_flow(u_lb=u_lb,u_ub=u_ub,u_init=u_init,slack_lb=slack_lb,slack_ub=slack_ub,slack_init_base=slack_init_base,xLF_lb=xLF_lb,xLF_ub=xLF_ub,m_init=m_init,Tr_sink_init=Tr_sink_init,p_perc_high=p_perc_high,p_perc_low=p_perc_low,T_perc_high=T_perc_high,T_perc_low=T_perc_low,n=n,m=m,s=s,Ts1=Ts1,p1=p1,To3=To3,phi3=phi3,L12 =L12,L23=L23,D=D,lam=lam,link_type=link_type,Lstreets=Lstreets,Dstreets=Dstreets,lamstreets=lamstreets,y_ind=y_ind,a=a,b=b,c=c,halflinks_slack=halflinks_slack,halflinks_limit=halflinks_limit, ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,runs=number_of_runs)
                        y_opt = res.x
                    print('Finished case {} of {}, in average of {:.2f}s per run. Started at {}'.format(case_number,total_cases,res.execution_time,start_wall_clock))
                    case_number += 1
                    result[method+'_'+str(form_OF)+'_'+bound+'_'+approach] = res
                    y_res[method+'_'+str(form_OF)+'_'+bound+'_'+approach] = y_opt
                    max_fev = max(max_fev,len(f_vec))
                    # plot results
                    ax_f.semilogy(f_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form_OF),alpha=.5)
                    ax_LF_error.semilogy(err_LF_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form_OF),alpha=.5)
            ax_f.semilogy([0,max_fev],[f_LF,f_LF],':r',alpha=.5)
            ax_LF_error.semilogy([0,max_fev],[tol,tol],':k')
            if not save_figs:
                ax_f.legend(handles=legend_handles)
                ax_LF_error.legend(handles=legend_handles)
    if save_figs:
        fig_legend = plt.figure('legend_heat_OF')
        ax_legend = fig_legend.gca()
        ax_legend.axis('off')
        fig_legend.patch.set_visible(False)
        ax_legend.legend(handles=legend_handles,loc='center')

    if save_figs:
        if s > 0:
            path_to_fig = os.path.join(dir_path,'Figures','MES3N_streets')
        else:
            path_to_fig = os.path.join(dir_path,'Figures','MES3N_line')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

    # Plot NR error of LF solve as well
    fig_NR = plt.figure('Error NR LF')
    ax_NR = fig_NR.gca()
    ax_NR.semilogy(err_vec_LF,':r',marker=marker_LF,alpha=.5,label=legend_label_LF) # unscaled, unless that wouldn't solve and scaling is used in OF
    ax_NR.semilogy([0,len(err_vec_LF)-1],[tol,tol],':k',label='tolerance')
    ax_NR.set_xlabel('Iteration')
    ax_NR.set_ylabel(r'$||F||_2$')
    ax_NR.legend()

    if save_tables:
        if s > 0:
            path_to_tables = os.path.join(dir_path,'Tables','MES3N_streets')
        else:
            path_to_tables = os.path.join(dir_path,'Tables','MES3N_line')
        for bound in bounds:
            with open(os.path.join(path_to_tables,'heat_optimizer_info_forms_'+bound+'_'+scale_label+'_{}_{}_{}_noCPU.txt'.format(n,m,s)), "w") as table, open(os.path.join(path_to_tables,'heat_optimizer_info_forms_'+bound+'_'+scale_label+'_{}_{}_{}_CPU_{}runs.txt'.format(n,m,s,number_of_runs)), "w") as table_CPU:
                for form_OF in forms_OF:
                    if form_OF in [3,5]: # [standard, no limits on m_i,l; standard, limits on m_i,l, m_i,l in xG]
                        y_LF_sol = np.concatenate((y_LF[:len(u_init)+len(slack_init_base)+len(nlsys_LF.unknown_m_links)],y_LF[-len(nlsys_LF.unknown_p_nodes)-len(nlsys_LF.unknown_Ts_nodes)-len(nlsys_LF.unknown_Ts_nodes)-1:]))
                    elif form_OF == 4: #standard with m_i,l in xG
                        y_LF_sol = np.concatenate((y_LF[:len(u_init)+1],mhl_LF,y_LF[len(u_init)+len(slack_init_base)-1:len(u_init)+len(slack_init_base)+len(nlsys_LF.unknown_m_links)],y_LF[-len(nlsys_LF.unknown_p_nodes)-len(nlsys_LF.unknown_Ts_nodes)-len(nlsys_LF.unknown_Ts_nodes)-1:]))
                    else: # half link flow, or
                        y_LF_sol = y_LF.copy()
                    for ind_app,approach in enumerate(approaches):
                        if approach == 'eq_constr':
                            approach_label = 'eq. constr.'
                        else:
                            approach_label = approach
                        if ind_app == 0:
                            table.write(r'\multirow{'+str(len(approaches))+r'}{*}{'+OF_forms_label.get(form_OF)+r'} & ')
                            if save_CPU_times:
                                table_CPU.write(r'\multirow{'+str(len(approaches))+r'}{*}{'+OF_forms_label.get(form_OF)+r'} & ')
                        else:
                            table.write(r' & ')
                            if save_CPU_times:
                                table_CPU.write(r' & ')
                        tables_iters = ' '
                        tables_fcals = ' '
                        tables_err = ' '
                        if save_CPU_times:
                            tables_cpu = ' '
                        for method in methods:
                            res_method = result.get(method+'_'+str(form_OF)+'_'+bound+'_'+approach)
                            y_method = y_res.get(method+'_'+str(form_OF)+'_'+bound+'_'+approach)
                            if res_method.success:
                                tables_iters += r' {:d} & '.format(res_method.nit)
                                tables_fcals += r' {:d} & '.format(res_method.nfev)
                                tables_err +=' '+exp_tex(error(y_method,y_LF_sol))+' & '
                                if save_CPU_times:
                                    tables_cpu += ' '+exp_tex(res_method.execution_time)+' & '
                            else:
                                tables_iters += r' & '
                                tables_fcals += r' & '
                                tables_err += r' & '
                                if save_CPU_times:
                                    tables_cpu += r' & '
                        table.write(r'{} & '.format(approach_label)+tables_iters+tables_fcals+tables_err)
                        if save_CPU_times:
                            table_CPU.write(r'{} & '.format(approach_label)+tables_iters+tables_fcals+tables_cpu+tables_err)
                    table.write(r'\hline ')
                    table_CPU.write(r'\hline ')

    print('\nError LF: {} after {} iters, using: {}.'.format(err_vec_LF[-1],iters_LF,legend_label_LF))
    if len(x_sol_LF) < 20:
        water = heat_net_LF.links[0].link_params.get('carrier')
        rho_w = water.rhon
        grav_const = water.g
        print('p heat = {} m'.format(p_vec_LF/(rho_w*grav_const)))
        print('m = {} kg/s'.format(m_vec_LF))
        print('Ts = {} C'.format(Ts_vec_LF))
        print('Tr = {} C'.format(Tr_vec_LF))
        print('m hl = {} kg/s'.format(m_hl_vec_LF))
        print('Ts hl = {}'.format(Ts_hl_vec_LF))
        print('Tr hl = {}'.format(Tr_hl_vec_LF))
        print('phi hl = {}'.format(phi_hl_vec_LF))
        print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net_LF.get_half_links()]))

    for bound in bounds:
        for form_OF in forms_OF:
            if form_OF in [3,5]: # [standard, no limits on m_i,l; standard, limits on m_i,l, m_i,l in xG]
                y_LF_sol = np.concatenate((y_LF[:len(u_init)+len(slack_init_base)+len(nlsys_LF.unknown_m_links)],y_LF[-len(nlsys_LF.unknown_p_nodes)-len(nlsys_LF.unknown_Ts_nodes)-len(nlsys_LF.unknown_Ts_nodes)-1:]))
            elif form_OF == 4: #standard with m_i,l in xG
                y_LF_sol = np.concatenate((y_LF[:len(u_init)+1],mhl_LF,y_LF[len(u_init)+len(slack_init_base)-1:len(u_init)+len(slack_init_base)+len(nlsys_LF.unknown_m_links)],y_LF[-len(nlsys_LF.unknown_p_nodes)-len(nlsys_LF.unknown_Ts_nodes)-len(nlsys_LF.unknown_Ts_nodes)-1:]))
            else: # half link flow, or
                y_LF_sol = y_LF.copy()
            for approach in approaches:
                print('\nForm OF: {}, bounds: {}, approach: {}.'.format(form_OF,bound,approach))
                for method in methods:
                    res_method= result.get(method+'_'+str(form_OF)+'_'+bound+'_'+approach)
                    y_method = y_res.get(method+'_'+str(form_OF)+'_'+bound+'_'+approach)
                    print('For {}: success: {}, iters: {}, error: {:.4e}, exec time: {:.2f}, message: {}'.format(method,res_method.success,res_method.nit,error(y_method,y_LF_sol),res_method.execution_time,res_method.message))

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # base case (3 nodes)
    n=0
    m=0
    s=0
    runs = 1#10
    max_iters = 50
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_CPU_times=False,number_of_runs=runs,n=n,m=m,s=s,max_iter=max_iters) # unscaled
    compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_CPU_times=False,number_of_runs=runs,scale_var='matrix',n=n,m=m,s=s,max_iter=max_iters)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_CPU_times=False,number_of_runs=runs,scale_var='per_unit',n=n,m=m,s=s,max_iter=max_iters)

    # medium case (30 nodes)
    n=5
    m=2
    s=3
    runs = 1#5
    max_iters = 50
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_CPU_times=False,number_of_runs=runs,scale_var='matrix',n=n,m=m,s=s,max_iter=max_iters)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_CPU_times=False,number_of_runs=runs,scale_var='per_unit',n=n,m=m,s=s,max_iter=max_iters)

    # medium / large case (163 nodes)
    n=10
    m=5
    s=10
    runs = 1#5
    max_iters = 50
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_CPU_times=False,number_of_runs=runs,scale_var='matrix',n=n,m=m,s=s,max_iter=max_iters)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_CPU_times=False,number_of_runs=runs,scale_var='per_unit',n=n,m=m,s=s,max_iter=max_iters)

    # large case (323 nodes)
    n=10
    m=5
    s=20
    runs = 1#5
    max_iters = 50
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_CPU_times=False,number_of_runs=runs,scale_var='matrix',n=n,m=m,s=s,max_iter=max_iters)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_CPU_times=False,number_of_runs=runs,scale_var='per_unit',n=n,m=m,s=s,max_iter=max_iters)

    plt.show()
