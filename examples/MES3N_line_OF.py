"""Optimal power flow of a multi-carrier network with 3 nodes connected in a single line, with possible streets connected to the third node."""
from examples import MES3N_streets as MES
import examples.MES3N_line as MES3N_line
from examples.E3N_line_OF import link_power_derivatives
from meslf.networks.gas_network import GasNode, GasLink, GasHalfLink
import warnings
import numpy as np
from meslf.utils.hide_print import HiddenPrints
from meslf.utils.constants import mbar, bar, MW, km, cm, kV
from meslf.utils.post_processing import error, fexp, fman, exp_tex
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
import scipy.optimize as spo
import scipy.sparse as sps
import os
import sys
import time
import ipopt
import datetime
import pickle

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

colors_method = {'trust-constr':'tab:blue','SLSQP':'tab:orange','ipopt':'tab:green'}
colors_energy = ['tab:green','red','lightcoral','firebrick','royalblue','mediumblue'] #(assumes E = [q1,P1,P2,P1c,dphi2,dphi1c])
labels_energy = [r'$-q_{1,0}-q_{3,0}$',r'$-P_{1,0}$',r'$-P_{2,0}$',r'$P_{1^c1^e}$',r'$-\Delta\varphi_{2,0}$',r'$\Delta\varphi_{1^c1^h}$']
linestyles_approaches = {'eq_constr':'-','direct':':','adjoint':'-.'}
markers_forms = {'nodal, standard, limits':'.','nodal, term., limits':'*','full, standard, limits':'d','full, term., limits':'s','nodal, standard, no limits':'x','nodal, term., no limits':'o','full, standard, no limits':'+','full, term., no limits':'^'}
marker_size = 10
legend_handles = [Line2D([0], [0], color=colors_method.get('trust-constr'), label='trust-constr'),
    Line2D([0], [0], color=colors_method.get('SLSQP'), label='SLSQP'),
    Line2D([0], [0], color=colors_method.get('ipopt'), label='IPOPT'),
    Line2D([0], [0], color='w',marker=markers_forms.get('nodal, standard, limits'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='nodal, standard, limits'),
    Line2D([0], [0], color='w',marker=markers_forms.get('nodal, term., limits'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='nodal, term., limits'),
    Line2D([0], [0], color='w',marker=markers_forms.get('full, standard, limits'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='full, standard, limits'),
    Line2D([0], [0], color='w',marker=markers_forms.get('full, term., limits'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='full, term., limits'),
    Line2D([0], [0], color='w',marker=markers_forms.get('nodal, standard, no limits'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='nodal, standard, no limits'),
    Line2D([0], [0], color='w',marker=markers_forms.get('nodal, term., no limits'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='nodal, term., no limits'),
    Line2D([0], [0], color='w',marker=markers_forms.get('full, standard, no limits'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='full, standard, no limits'),
    Line2D([0], [0], color='w',marker=markers_forms.get('full, term., no limits'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='full, term., no limits'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('eq_constr'), label='LF as eq. constr.'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('direct'), label='Direct approach'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('adjoint'), label='Adjoint approach')]
legend_handles_energy = [plt.Rectangle((0,0),1,1, color=colors_energy[ind], label=labels_energy[ind]) for ind in range(len(colors_energy))]

def create_network(n=0,m=0,s=0,p1g=9*bar,q3=1,V1=50*kV,delta1=0.,V2=50*kV,P2=-1*MW,P3=1.5*MW,Q3=1.5*MW,Ts1=100.,p1h=9*bar,To2=100.,To3=50.,phi2=-1*MW,phi3=1.5*MW,L12=4*km,L23=5*km,Dg=10*cm,E=.98,link_type_g='pipe_high_pres_weymouth',hydr_eq_gas='fb',De=1*cm,link_type_e='pi_line',Dh=0.15,lam=0.2,link_type_h='standard_pipe_low_pres_pole',coupling_type='EH',coupling_point=1):
    """Create multi-carrier network, with or without streets connected to it.

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
    elec_net : ElectricalNetwork
        The electrical network
    heat_net : HeatNetwork
        The heat network
    het_net : HeterogeneousNetwork
        The multi-carrier network
    """
    if s == 0: # base case
        gas_net,elec_net,heat_net,het_net = MES3N_line.create_network(p1g=p1g,q3=q3,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi2=phi2,phi3=phi3,L12=L12,L23=L23,Dg=Dg,E=E,link_type_g=link_type_g,hydr_eq_gas=hydr_eq_gas,De=De,link_type_e=link_type_e,Dh=Dh,lam=lam,link_type_h=link_type_h,node_set_elec=2,node_set_heat=2,heat_load='outflow',coupling_type=coupling_type,coupling_point=coupling_point,node_set=1)
    else: # with streets
        gas_net,elec_net,heat_net,het_net = MES.create_network(n,m,s,p1g=p1g,q3=q3,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi2=phi2,phi3=phi3,L12=L12,L23=L23,Dg=Dg,E=E,link_type_g=link_type_g,hydr_eq_gas=hydr_eq_gas,De=De,link_type_e=link_type_e,Dh=Dh,lam=lam,link_type_h=link_type_h,node_set_elec=2,node_set_heat=2,heat_load='outflow',coupling_type=coupling_type,coupling_point=coupling_point,node_set=1)
    return gas_net,elec_net,heat_net,het_net

def set_x_LF_init(nlsys,n=0,m=0,s=0,q_init=1,m_init=10,Tr_sink_init=50,p_perc_high=.99,p_perc_low=.98,T_perc_high=.99,T_perc_low=.98,qc_init=1,Pc_init=1*MW,Qc_init=1*MW,phic_init=1*MW):
    """Set the unscaled initial guess for the LF variables"""
    gas_net = nlsys.nlsystemsg[0].gasnetwork
    elec_net = nlsys.nlsystemse[0].elecnetwork
    heat_net = nlsys.nlsystemsh[0].heatnetwork
    Ts_ref = heat_net.nodes[0].get_Ts()
    het_net = nlsys.hetnetwork
    if s == 0: # base case
        unknown_pg_nodes = []
        unknown_q_links = []
        for el in nlsys.xg_entries:
            if isinstance(el,GasNode):
                unknown_pg_nodes.append(el)
            elif isinstance(el,GasLink):
                unknown_q_links.append(el)
        pg_ref = gas_net.nodes[0].get_p()
        delta_ref = elec_net.nodes[0].get_delta()
        V_ref = elec_net.nodes[0].get_V()
        ph_ref = heat_net.nodes[0].get_p()
        pg_init = np.zeros(len(unknown_pg_nodes))
        ph_init = np.zeros(len(nlsys.unknown_p_nodes))
        ph_ind = 0
        pg_ind = 0
        Ts_init = np.zeros(len(nlsys.unknown_Ts_nodes))
        Ts_ind = 0
        for i in range(3):
            if gas_net.nodes[i] in unknown_pg_nodes: # heat node i
                pg_init[pg_ind] = pg_ref*(1-(1-p_perc_high)*(pg_ind+1)/4)
                pg_ind += 1
            if heat_net.nodes[i] in nlsys.unknown_p_nodes: # heat node i
                ph_init[ph_ind] = ph_ref*(1-(1-p_perc_high)*(ph_ind+1)/4)
                ph_ind += 1
            if heat_net.nodes[i] in nlsys.unknown_Ts_nodes: # heat node i
                Ts_init[Ts_ind] = Ts_ref*(1-(1-T_perc_high)*(Ts_ind+1)/4)
                Ts_ind += 1
        if unknown_q_links:
            q_vec_init = q_init*np.ones(len(unknown_q_links))
            xg_init = np.concatenate((q_vec_init,pg_init))
        else:
            xg_init = pg_init

        delta_init = elec_net.nodes[0].get_delta()*np.ones(len(nlsys.unknown_delta_nodes))
        V_init = elec_net.nodes[0].get_V()*np.ones(len(nlsys.unknown_V_nodes))
        xe_init = np.concatenate((delta_init,V_init)) # unscaled

        m_vec_init = m_init*np.ones(len(nlsys.unknown_m_links))
        m_hl_init = m_init*np.ones(len(nlsys.unknown_m_halflinks))
        if len(m_hl_init): #NB: only true when coupled at node 1!!!!!
            m_hl_init[0] = -m_init # node 2 is a source
        Tr_init = Tr_sink_init*np.ones(len(nlsys.unknown_Tr_nodes))
        xh_init = np.concatenate((m_vec_init,m_hl_init,ph_init,Ts_init,Tr_init)) # unscaled

        #coupling
        qc_vec_init = qc_init*np.ones(len(nlsys.unknown_qc_links))
        Pc_vec_init = Pc_init*np.ones(len(nlsys.unknown_Pc_links))
        Qc_vec_init = Qc_init*np.ones(len(nlsys.unknown_Qc_links))
        phic_vec_init = phic_init*np.ones(len(nlsys.unknown_dphi_links))
        mc_init = m_init*np.ones(len(nlsys.unknown_mc_links))
        Toc_init = Ts_ref*np.ones(len(nlsys.unknown_Ts_links))
        xc_init = np.concatenate((qc_vec_init,Pc_vec_init,Qc_vec_init,mc_init,phic_vec_init,Toc_init))

        x_init = np.concatenate((xg_init,xe_init,xh_init,xc_init))
    else:
        with HiddenPrints():
            x_init = MES.initialize_network(het_net, gas_net, elec_net, heat_net,n,m,s,p_perc_high,p_perc_low,T_perc_high,T_perc_low,q3=qc_init,P3=Pc_init,Q3=Qc_init,phi3=phic_init,Ts1=Ts_ref,formulation=nlsys.formulation)
    return x_init

def xmes_from_yopf(y,nlsys=None):
    """Returns the variables of steady-state LF, from the variables y of OF."""
    xmes = y[-len(nlsys.x_entries):]
    return xmes

def objective_function(y,y_ind,a=np.array([0,0,0,0,0,0]),b=np.array([.01*MES3N_line.GHV,.3,.3,.2,.05,.04]),c=np.array([1e-6*(MES3N_line.GHV)**2,3e-5,3e-5,2e-5,4.5e-4,4e-4]),scale_var=None,scale_var_params=None,fb=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    E : np array
        Array with flows used in objective. Gas flows are assumed to be in kg/s, active powers in W, and heat power in W. Scaled when per unit scaling is used, unscaled otherwise.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [q1,P1,P2,P1c,dphi2,dphi1c]. Scaled when per unit scaling is used, unscaled otherwise.
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

def grad_objective(y,y_ind,a=np.array([0,0,0,0,0,0]),b=np.array([.01*MES3N_line.GHV,.3,.3,.2,.05,.04]),c=np.array([1e-6*(MES3N_line.GHV)**2,3e-5,3e-5,2e-5,4.5e-4,4e-4]),scale_var=None,scale_var_params=None,fb=None,Dy_inv=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [q1,P1,P2,P1c,dphi2,dphi1c]. Scaled when per unit scaling is used, unscaled otherwise.
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

def hess_objective(y,y_ind,a=np.array([0,0,0,0,0,0]),b=np.array([.01*MES3N_line.GHV,.3,.3,.2,.05,.04]),c=np.array([1e-6*(MES3N_line.GHV)**2,3e-5,3e-5,2e-5,4.5e-4,4e-4]),scale_var=None,scale_var_params=None,fb=None,Dy_inv=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [q1,P1,P2,P1c,dphi2,dphi1c]. Scaled when per unit scaling is used, unscaled otherwise.
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
    nlsys : NonLinearSystemHeterogeneous
        Nonlinear system of the multi-carrier network. Constains the networks, information about scaling, etc.

    Returns
    -------
    h : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    network_g = nlsys.nlsystemsg[0].gasnetwork
    network_e = nlsys.nlsystemse[0].elecnetwork
    network_h = nlsys.nlsystemsh[0].heatnetwork
    network_mes = nlsys.hetnetwork

    xmes = xmes_from_yopf(y,nlsys=nlsys)

    # evaluate load flow equations
    network_g, network_e, network_h, network_mes = reset_network_without_update(network_g, network_e, network_h, network_mes)
    F = nlsys.F(xmes) # Also updates network. Should set correct values on links and halflinks

    # evaluate additional equations
    len_u = 4
    xG = y[len_u:-len(xmes)] # scaled when per unit scaling is used
    G = np.zeros(len(xG))
    # NB: Assumes coupling at node 1 with an EH using node set 1
    # evaluate conservation of mass in gas slack node
    q1 = xG[0]
    if nlsys.scale_var == 'per_unit':
        q1 *= nlsys.scale_var_params.get('qbase')
    # set / create a half link at the gas slack node
    gn1 = network_g.nodes[0]
    if len(gn1.half_links) == 0:
        GasHalfLink(gn1.name + "_hl",gn1,q1)
    else:
        gn1.half_links[0].q = q1
    G[0] = gn1.node_law(network=network_mes,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    # evaluate conservation of energy in the QVdelta and generator node
    P1_gen, Q2_gen = xG[1:3]
    if nlsys.scale_var == 'per_unit':
        network_e.nodes[0].half_links[0].P = P1_gen*nlsys.scale_var_params.get('Sbase')
        network_e.nodes[1].half_links[0].Q = Q2_gen*nlsys.scale_var_params.get('Sbase')
    else:
        network_e.nodes[0].half_links[0].P = P1_gen
        network_e.nodes[1].half_links[0].Q = Q2_gen
    fP1, _ = network_e.nodes[0].node_law(network=network_mes,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    _, fQ2 = network_e.nodes[1].node_law(network=network_mes,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    cons_energy = np.array([fP1,fQ2])
    G[1:3] = cons_energy

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
    dh_dy = np.zeros((len_h,len(y)))

    network_g = nlsys.nlsystemsg[0].gasnetwork
    nlsyse = nlsys.nlsystemse[0]
    network_e = nlsyse.elecnetwork
    network_h = nlsys.nlsystemsh[0].heatnetwork
    network_mes = nlsys.hetnetwork

    xmes = xmes_from_yopf(y,nlsys=nlsys)
    xG = y[len_u:-len(xmes)] # scaled when per unit scaling is used
    V2,P2,To2,dphi2 = y[:len_u]
    network_g, network_e, network_h, network_mes = reset_network_without_update(network_g, network_e, network_h, network_mes)
    # update vector with voltage amplitudes and magnitudes, since nlsys.J() only updates the voltage amplitudes and angels that are in x
    nlsyse.V_vec_mag[1] = V2

    # Jacobian of LF equations
    J_lf = nlsys.J_dense(xmes)
    dh_dy = np.zeros((len_h,len(y)))
    dh_dy[len(xG):,len_u+len(xG):] = J_lf #dF_dxF
    # Derivatives of F to u
    xe = xmes[len(nlsys.xg_entries):len(nlsys.xg_entries)+len(nlsys.xe_entries)]
    Ne = len(network_e.nodes)
    V2_ind = Ne+1 # index of V2 within the full Jee
    Fe_ind = nlsyse.FP + [Ne+ind for ind in nlsyse.FQ]
    Je_full = nlsyse.J_dense(xe,return_full=True)
    dh_dy[len(xG)+len(nlsys.Fg_entries):len(xG)+len(nlsys.Fg_entries)+len(Fe_ind),0] = Je_full[Fe_ind,V2_ind].ravel() # dFe_dV2
    dh_dy[len(xG)+len(nlsys.Fg_entries),1] = 1 #dFP2_dP2
    water = network_h.links[0].link_params.get('carrier')
    if nlsys.formulation.get('heat') == 'half_link_flow':
        m2 = xmes[len(nlsys.xg_entries)+len(xe)+len(nlsys.unknown_m_links)]
        dh_dy[len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links)+len(nlsys.F_Ts_nodes)+len(nlsys.F_Tr_nodes),3] = -1 #dFphi2_dphi2
        dh_dy[len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links)+len(nlsys.F_Ts_nodes)+len(nlsys.F_Tr_nodes),2] = m2*water.get_Cp(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) #dFphi2_dTs2,0
        dh_dy[len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links)+1,2] = m2 #dFTs2_dTs2,0
    else:
        hl2 = network_h.nodes[1].half_links[0]
        Tr2 = network_h.nodes[1].get_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dT2 = To2 - hl2.get_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dm2_dphi2 = 1/(water.get_Cp(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)*dT2)
        dm2_dTs2hl = hl2.dm_dTs(dphi2,To2,Tr2,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dh_dy[len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+1,2] = -dm2_dTs2hl # dFm2_dTs2,0
        dh_dy[len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links)+1,2] = dm2_dTs2hl*To2 + hl2.get_m(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)# dFTs2_dTs2,0
        dh_dy[len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links)+len(nlsys.F_Ts_nodes)+1,2] = -dm2_dTs2hl * Tr2# dFTr2_dTs2,0
        dh_dy[len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+1,3] = -dm2_dphi2 # dFm2_dphi2
        dh_dy[len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links)+1,3] = dm2_dphi2 * To2# dFTs2_dphi2
        dh_dy[len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links)+len(nlsys.F_Ts_nodes)+1,3] = -dm2_dphi2 * Tr2# dFTr2_dphi2

    # Derivatives of G to xG
    dh_dy[0,len_u] = -1 # der. of cons. of mass in node 1g to q1
    dh_dy[1,len_u+1] = 1 # der. of cons. of energy (active power) in node 1e to P1
    dh_dy[2,len_u+2] = 1 # der. of cons. of energy (reactive power) in node 2e to Q2
    # Derivatives of G to xF
    if nlsys.formulation.get('gas') == 'nodal':
        _, df12_dp2 = network_g.links[0].f_der_p(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) # since -dq12_dp2 = df_dp2 in this case
        dh_dy[0,len_u+len(xG)] = df12_dp2 # der. of cons. of mass in node 1g to p2
    else:
        dh_dy[0,len_u+len(xG)] = -1 # der. of cons. of mass in node 1g to q12
    Ge_ind = [0,Ne+1] # indices of conservation of energy equations within full Jee
    xlfe_ind = nlsyse.xdelta + [Ne+ind for ind in nlsyse.xV]
    dh_dy[1:len(Ge_ind)+1,:][:,len_u+len(xG)+len(nlsys.xg_entries):len_u+len(xG)+len(nlsys.xg_entries)+len(xe)] = Je_full[Ge_ind,:][:,xlfe_ind] #dGe_dxFe
    dh_dy[0,-6] = -1 # der. of cons. of mass in node 1g to q1g1c
    dh_dy[1,-5] = -1 # der. of cons. of energy (active power) in node 1e to P1c1e
    # Derivatives of G to u
    dh_dy[1:len(Ge_ind)+1,0] = Je_full[Ge_ind,V2_ind].ravel() #dGe_dV2
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
    # plt.plot([len_u+len(xG)+len(nlsys.xg_entries)-.5,len_u+len(xG)+len(nlsys.xg_entries)-.5],[0-.5,len_h-.5],'--k')
    # plt.plot([len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.xe_entries)-.5,len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.xe_entries)-.5],[0-.5,len_h-.5],'--k')
    # plt.plot([len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.xh_entries)-.5,len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.xh_entries)-.5],[0-.5,len_h-.5],'--k')
    # plt.plot([0-.5,len(y)-.5],[len(xG)-.5,len(xG)-.5],'k')
    # plt.plot([0-.5,len(y)-.5],[len(xG)+len(nlsys.Fg_entries)-.5,len(xG)+len(nlsys.Fg_entries)-.5],'--k')
    # plt.plot([0-.5,len(y)-.5],[len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)-.5,len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)-.5],'--k')
    # plt.plot([0-.5,len(y)-.5],[len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+len(nlsys.Fh_entries)-.5,len(xG)+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+len(nlsys.Fh_entries)-.5],'--k')
    # plt.show()
    return dh_dy

def gamma(y, nlsys=None, ineq_constr_lb=np.array([-2,-2,-10,0]), ineq_constr_ub=np.array([2,2,(3*MW)**2,(3*MW)**2,0,10])):
    """The nonlinear inequality constraints gamma(x)>=0 on the gas link flows, the electrical link powers, and the heat half link flow. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function. For the links, the dummy links are not taken into account, since those variables are part of xF.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeat
        Nonlinear system of the heterogenous network. Constains the networks, information about scaling, etc.
    ineq_constr_lb : np arrays
        Lower bounds. Scaled when per unit or matrix scaling is used.
    ineq_constr_ub : np arrays
        Upper bounds. Scaled when per unit or matrix scaling is used.

    Returns
    -------
    gam : np array
        The nonlinear inequality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    network_g = nlsys.nlsystemsg[0].gasnetwork
    network_e = nlsys.nlsystemse[0].elecnetwork
    network_h = nlsys.nlsystemsh[0].heatnetwork
    network_mes = nlsys.hetnetwork

    xmes = xmes_from_yopf(y,nlsys=nlsys)
    network_g, network_e, network_h, network_mes = reset_network_without_update(network_g, network_e, network_h, network_mes)
    network_mes.update(xmes,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,formulation=nlsys.formulation)

    # gas
    gas_link_flows = list()
    if nlsys.formulation.get('gas') == 'nodal':
        for link in network_g.get_links():
            if not link.link_type == 'dummy':
                qk = link.flow(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params) # scaled when per unit scaling is used, unscaled otherwise
                if nlsys.scale_var == 'matrix': # lower and upper bounds are scaled
                    qk = qk/nlsys.scale_var_params.get('qbase')
                gas_link_flows.append(qk)
    q = np.array(gas_link_flows)

    # electricity
    link_powers_squared = list()
    for link in network_e.get_links():
        if not link.link_type == 'dummy':
            Sk2 =  link.get_Pstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)**2 + link.get_Qstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)**2
            if nlsys.scale_var == 'matrix': # lower and upper bounds are scaled
                Sk2 = Sk2/nlsys.scale_var_params.get('Sbase')**2
            link_powers_squared.append(Sk2)
    S_squared = np.array(link_powers_squared)

    # heat
    if nlsys.formulation.get('heat') == 'standard':
        hl_flows = np.zeros(len(network_h.half_links))
        for ind_hl,hl in enumerate(network_h.get_half_links(bc_types=[2,3])):
            hl_flows[ind_hl] = hl.flow(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    else:
        hl_flows = np.array([])

    x_lb = np.concatenate([q,hl_flows])# variables that have a lower bound
    x_ub = np.concatenate([q,S_squared,hl_flows])# variables that have an upper bound
    return np.concatenate((x_lb-ineq_constr_lb,ineq_constr_ub-x_ub))

def gamma_der(y, nlsys=None, Dy_inv=None):
    """The nonlinear inequality constraints gamma(x)>=0 on the gas link flows, the electrical link powers, and the heat half link flow. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

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
    nlsysg = nlsys.nlsystemsg[0]
    network_g = nlsysg.gasnetwork
    network_e = nlsys.nlsystemse[0].elecnetwork
    nlsysh = nlsys.nlsystemsh[0]
    network_h = nlsysh.heatnetwork
    network_mes = nlsys.hetnetwork

    xmes = xmes_from_yopf(y,nlsys=nlsys)
    len_u = 4
    xG = y[len_u:-len(xmes)]
    network_g, network_e, network_h, network_mes = reset_network_without_update(network_g, network_e, network_h, network_mes)
    network_mes.update(xmes,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,formulation=nlsys.formulation)

    # gas
    if nlsys.formulation.get('gas') == 'nodal':
        link_type_g = network_g.links[0].link_type
        len_Eg = len(list(network_g.get_links(link_types=[link_type_g])))  # without dummy links
        dgammag_dy = np.zeros((2*len_Eg,len(y)))
        dq_dx = np.zeros((len_Eg,len(nlsys.xg_entries)))
        for ind_link,link in enumerate(network_g.get_links(link_types=[link_type_g])):
            qk = link.flow() # unscaled
            link.q = qk
            dqk_dpi, dqk_dpj = -np.array(link.f_der_p(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)) # scaled when per unit scaling is used, unscaled otherwise
            if link.start_node.number in nlsysg.ind_p: # node 1 is reference, so is not part of ind_p
                ind_pi = nlsysg.ind_p.index(link.start_node.number)
                dq_dx[ind_link,ind_pi] = dqk_dpi
            ind_pj = nlsysg.ind_p.index(link.end_node.number)
            dq_dx[ind_link,ind_pj] = dqk_dpj
        dgammag_dy[:,len_u+len(xG):len_u+len(xG)+len(nlsys.xg_entries)] = np.vstack((np.eye(len_Eg),-np.eye(len_Eg))).dot(dq_dx)
        if nlsys.scale_var == 'matrix':
            gammag_base = 1/nlsys.scale_var_params.get('qbase')*np.ones(2*len_Eg)

    # electricity
    link_type_e = network_e.links[0].link_type
    len_Ee = len(list(network_e.get_links(link_types=[link_type_e])))
    dgammae_dy = np.zeros((len_Ee,len(y)))
    V2_ind = 0 # index of V2 within y
    for ind_e, link in enumerate(network_e.get_links(link_types=[link_type_e])):
        Pij = link.get_Pstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        Qij = link.get_Qstart(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        Vi = link.start_node.get_V(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        Vj = link.end_node.get_V(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dPij_ddeltai, dQij_ddeltai, dPij_ddeltaj, dQij_ddeltaj, dPij_dVi, dQij_dVi, dPij_dVj, dQij_dVj = link_power_derivatives(link,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        if link.start_node in nlsys.unknown_delta_nodes:
            dgammae_dy[ind_e,len_u+len(xG)+len(nlsys.xg_entries)+nlsys.unknown_delta_nodes.index(link.start_node)] = -2*Pij*dPij_ddeltai - 2*Qij*dQij_ddeltai
        if link.end_node in nlsys.unknown_delta_nodes:
            dgammae_dy[ind_e,len_u+len(xG)+len(nlsys.xg_entries)+nlsys.unknown_delta_nodes.index(link.end_node)] = -2*Pij*dPij_ddeltaj - 2*Qij*dQij_ddeltaj
        if link.start_node in nlsys.unknown_V_nodes:
            dgammae_dy[ind_e,len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.unknown_delta_nodes)+nlsys.unknown_V_nodes.index(link.start_node)] = -2*Pij*dPij_dVi - 2*Qij*dQij_dVi
        if link.end_node in nlsys.unknown_V_nodes:
            dgammae_dy[ind_e,len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.unknown_delta_nodes)+nlsys.unknown_V_nodes.index(link.end_node)] = -2*Pij*dPij_dVj - 2*Qij*dQij_dVj
        if link.start_node == network_e.nodes[1]: # derivative to V2
            dgammae_dy[ind_e,V2_ind] = -2*Pij*dPij_dVi - 2*Qij*dQij_dVi
        elif link.end_node == network_e.nodes[1]: # derivative to V2
            dgammae_dy[ind_e,V2_ind] = -2*Pij*dPij_dVj - 2*Qij*dQij_dVj
    if nlsys.scale_var == 'matrix':
        gammae_base = 1/(nlsys.scale_var_params.get('Sbase')**2)*np.ones(len_Ee)

    # heat
    if nlsys.formulation.get('heat') == 'standard':
        T_loads = len(network_h.half_links)
        dgammah_dy = np.zeros((2*T_loads,len(y)))
        water = network_h.links[0].link_params.get('carrier')
        V2, P2, To2, dphi2 = y[:len_u]
        hl2 = network_h.nodes[1].half_links[0]
        Tr2 = network_h.nodes[1].get_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dT2 = To2 - hl2.get_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dm2_dphi2 = 1/(water.get_Cp(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)*dT2)
        dm2_dTs2hl = hl2.dm_dTs(dphi2,To2,Tr2,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        # derivatives to u
        dgammah_dy[0,2] = dm2_dTs2hl
        dgammah_dy[T_loads,2] = -dm2_dTs2hl
        dgammah_dy[0,3] = dm2_dphi2
        dgammah_dy[T_loads,3] = -dm2_dphi2
        # derivatives to xF
        dmhl_dx = np.zeros((T_loads,len(nlsys.xh_entries)))
        for ind_hl,hl in enumerate(network_h.get_half_links(bc_types=[2,3])):
            dm_dTs = hl.m_der_Ts(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            dm_dTr = hl.m_der_Tr(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            Ts_ind = len(nlsysh.xm) + len(nlsysh.xmhl) + len(nlsysh.xp) + nlsys.unknown_Ts_nodes.index(hl.start_node)
            Tr_ind = len(nlsysh.xm) + len(nlsysh.xmhl) + len(nlsysh.xp) + len(nlsysh.xTs) + nlsys.unknown_Tr_nodes.index(hl.start_node)# index of Tr of start node within xF
            dmhl_dx[ind_hl,Ts_ind] = dm_dTs
            dmhl_dx[ind_hl,Tr_ind] = dm_dTr
        dgammah_dy[:,len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.xe_entries):len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.xh_entries)] = np.vstack((np.eye(T_loads),-np.eye(T_loads))).dot(dmhl_dx)
        if nlsys.scale_var == 'matrix':
            gammah_base = 1/(nlsys.scale_var_params.get('mbase'))*np.ones(2*T_loads)

    if nlsys.formulation.get('gas') == 'nodal':
        dgamma_dy = np.vstack((dgammag_dy,dgammae_dy))
        if nlsys.scale_var == 'matrix':
            gamma_base = np.concatenate((gammag_base,gammae_base))
    else:
        dgamma_dy = dgammae_dy.copy()
        if nlsys.scale_var == 'matrix':
            gamma_base = gammae_base.copy()
    if nlsys.formulation.get('heat') == 'standard':
        dgamma_dy = np.vstack((dgamma_dy,dgammah_dy))
        if nlsys.scale_var == 'matrix':
            gamma_base = np.concatenate((gamma_base,gammah_base))
    if nlsys.scale_var == 'matrix':
        dgamma_dy = np.diag(gamma_base).dot(dgamma_dy.dot(Dy_inv))
    # plt.figure('Jacobian gamma')
    # plt.spy(dgamma_dy)
    # len_gam = dgamma_dy.shape[0]
    # plt.plot([len_u-.5,len_u-.5],[0-.5,len_gam-.5],'k')
    # plt.plot([len_u+len(xG)-.5,len_u+len(xG)-.5],[0-.5,len_gam-.5],'k')
    # plt.plot([len_u+len(xG)+len(nlsys.xg_entries)-.5,len_u+len(xG)+len(nlsys.xg_entries)-.5],[0-.5,len_gam-.5],'--k')
    # plt.plot([len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.xe_entries)-.5,len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.xe_entries)-.5],[0-.5,len_gam-.5],'--k')
    # plt.plot([len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.xh_entries)-.5,len_u+len(xG)+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.xh_entries)-.5],[0-.5,len_gam-.5],'--k')
    # if nlsys.formulation.get('gas') == 'nodal':
    #     plt.plot([0-.5,len(y)-.5],[2*len_Eg-.5,2*len_Eg-.5],'--k')
    #     plt.plot([0-.5,len(y)-.5],[2*len_Eg+len_Ee-.5,2*len_Eg+len_Ee-.5],'--k')
    # else:
    #     plt.plot([0-.5,len(y)-.5],[len_Ee-.5,len_Ee-.5],'--k')
    # if nlsys.formulation.get('heat') == 'standard':
    #     plt.plot([0-.5,len(y)-.5],[len_gam-2*T_loads-.5,len_gam-2*T_loads-.5],'--k')
    # plt.show()
    return dgamma_dy

def reset_network_without_update(gas_net,elec_net,heat_net,het_net):
    """Reset the heterogeneous network, but do not update the network values. That is, remove the half link connected to the electrical slack node, and set Q of electrical generator to zero, and set the mass flow of heat slack nodes to zero."""
    #NB: Currently assumes coupling at node 1, with node set 1!!!!
    elec_net.nodes[0].half_links[0].P = 0
    elec_net.nodes[1].half_links[0].Q = 0
    return gas_net,elec_net,heat_net,het_net

def update_bc(gas_net,elec_net,heat_net,het_net,V2,P2,To2,dphi2,scale_var=None,scale_var_params=None):
    """Update the boundary conditions of the multi-carrier network, based on the control variables of OF"""
    #NB: Currently assumes coupling at node 1, with node set 1!!!!
    if scale_var == 'per_unit':
        V2 = V2*scale_var_params.get('Vbase')
        P2 = P2*scale_var_params.get('Sbase')
        To2 = To2*scale_var_params.get('Tbase')
        dphi2 = dphi2*scale_var_params.get('phibase')
    elec_net.nodes[1].V = V2
    elec_net.nodes[1].half_links[0].P = P2 #<0
    heat_net.nodes[1].half_links[0].Ts = To2
    heat_net.nodes[1].half_links[0].dphi = dphi2 #<0
    return gas_net,elec_net,heat_net,het_net

def run_optimal_load_flow(u_lb=np.array([.98*50*kV,-1.5*MW,80,-1.5*MW]),u_ub=np.array([1.02*50*kV,0,110,0]),u_init=np.array([50*kV,-1*MW,100,-1*MW]),slack_lb=np.array([-2,-2*1.5*MW,-2*1.5*MW]),slack_ub=np.array([0,0,2*1.5*MW]),slack_init=np.array([-1,-1.5*MW,-1.5*MW]),xLF_lb=np.array([1*mbar,1*mbar,-np.pi,-np.pi,.98*50*kV,-10,-10,0,0,60,60,0,0,0,0,-3*MW,0,0,0,60]),xLF_ub=np.array([10*bar,10*bar,np.pi,np.pi,1.02*50*kV,10,10,10*bar,10*bar,110,110,55,55,55,10,3*MW,3*MW,10,3*MW,100]),xLF_init=[],n=0,m=0,s=0,p1g=9*bar,q3=1,V1=50*kV,delta1=0.,P3=1.5*MW,Q3=1.5*MW,Ts1=100.,p1h=9*bar,To3=50.,phi3=1.5*MW,L12=4*km,L23=5*km,Dg=10*cm,E=.98,link_type_g='pipe_high_pres_weymouth',hydr_eq_gas='fb',De=1*cm,link_type_e='pi_line',Dh=0.15,lam=0.2,link_type_h='standard_pipe_low_pres_pole',coupling_type='EH',coupling_point=1,q_init=1,m_init=10,Tr_sink_init=50,p_perc_high=.99,p_perc_low=.98,T_perc_high=.99,T_perc_low=.98,qc_init=1,Pc_init=1*MW,Qc_init=1*MW,phic_init=1*MW,y_ind=[4,5,1,22,3,25],a=np.array([0,0,0,0,0,0]),b=np.array([.01*MES3N_line.GHV,.3,.3,.2,.05,.04]),c=np.array([1e-6*(MES3N_line.GHV)**2,3e-5,3e-5,2e-5,4.5e-4,4e-4]), ineq_constr_lb=np.array([-2,-2,-10,0]), ineq_constr_ub=np.array([2,2,(3*MW)**2,(3*MW)**2,0,10]),link_limits=False,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None},scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,optimization_method='trust-constr',stay_within_bounds=False,fb=None,runs=1,derivatives=False):
    """Run optimal load flow, with LF as equality constraints.

    Parameters
    ----------
    link_limits : bool, optional
        Determines if limits are imposed on the gas link flows, the electrical link powers, and the water half link flows. Default is False.

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
    if coupling_type != 'EH' or coupling_point != 1:
        raise ValueError('Only implemented for coupling at node 1 with an EH (using node set 1)')
    print('\nRunning OPF for MES3N line with LF as eq. constr. (link limits: {}, formulations: {}, hard bounds: {}, method: {}, scaling: {})'.format(link_limits,formulation,stay_within_bounds,optimization_method,scale_var))
    # create network
    V2, P2, To2, dphi2 = u_init
    with HiddenPrints():
        gas_net,elec_net,heat_net,het_net = create_network(n=n,m=m,s=s,p1g=p1g,q3=q3,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi2=dphi2,phi3=phi3,L12=L12,L23=L23,Dg=Dg,E=E,link_type_g=link_type_g,hydr_eq_gas=hydr_eq_gas,De=De,link_type_e=link_type_e,Dh=Dh,lam=lam,link_type_h=link_type_h,coupling_type=coupling_type,coupling_point=coupling_point)

    # set initial guess and initialize network, using default initial guess
    het_net.initialize() # needed to create nlsys
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    if len(xLF_init) == 0:
        xLF_init = set_x_LF_init(nlsys,n=n,m=m,s=s,q_init=q_init,m_init=m_init,Tr_sink_init=Tr_sink_init,p_perc_high=p_perc_high,p_perc_low=p_perc_low,T_perc_high=T_perc_high,T_perc_low=T_perc_low,qc_init=qc_init,Pc_init=Pc_init,Qc_init=Qc_init,phic_init=phic_init)
    # initial guess for OF (unscaled)
    y0 = np.concatenate((u_init,slack_init,xLF_init))

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('Vbase'), scale_var_params.get('Sbase'),scale_var_params.get('Tbase'), scale_var_params.get('phibase')])
        slack_base = np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase')])
        G_base = np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase')])
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
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_h = nlsys.nlsystemsh[0].heatnetwork
        network_mes = nlsys.hetnetwork
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        V2, P2, To2, dphi2  = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network_g,network_e,network_h,network_mes = update_bc(network_g,network_e,network_h,network_mes,V2,P2,To2,dphi2,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        H = h(y, nlsys=nlsys, Dh=Dh)
        global err_LF_vec_global
        F = H[-len(nlsys.F_entries):]
        err_LF_vec_global.append(np.linalg.norm(F))
        return H

    def jac_eq_constr(y,nlsys=nlsys, Dy_inv=Dy_inv, Dh=Dh):
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_h = nlsys.nlsystemsh[0].heatnetwork
        network_mes = nlsys.hetnetwork
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        V2, P2, To2, dphi2  = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network_g,network_e,network_h,network_mes = update_bc(network_g,network_e,network_h,network_mes,V2,P2,To2,dphi2,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dh_dy = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh)
        return dh_dy

    lb_nleq = np.zeros(len(slack_lb)+len(xLF_lb))
    ub_nleq = np.zeros(len(slack_ub)+len(xLF_ub))
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

    # define nonlinear inequality constraints on the gas link flows, the electrical link powers, and the heat half link flow.
    if link_limits:
        def g(y,nlsys=nlsys, ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub, Dy_inv=Dy_inv):
            network_g = nlsys.nlsystemsg[0].gasnetwork
            network_e = nlsys.nlsystemse[0].elecnetwork
            network_h = nlsys.nlsystemsh[0].heatnetwork
            network_mes = nlsys.hetnetwork
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                y = Dy_inv.dot(y)
            # update bc of the network
            V2, P2, To2, dphi2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
            network_g,network_e,network_h,network_mes = update_bc(network_g,network_e,network_h,network_mes,V2,P2,To2,dphi2,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            gam = gamma(y,nlsys=nlsys, ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub)
            return gam
        def g_jac(y,nlsys=nlsys, Dy_inv=Dy_inv):
            network_g = nlsys.nlsystemsg[0].gasnetwork
            network_e = nlsys.nlsystemse[0].elecnetwork
            network_h = nlsys.nlsystemsh[0].heatnetwork
            network_mes = nlsys.hetnetwork
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                y = Dy_inv.dot(y)
            # update bc of the network
            V2, P2, To2, dphi2 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
            network_g,network_e,network_h,network_mes = update_bc(network_g,network_e,network_h,network_mes,V2,P2,To2,dphi2,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            dgamma_dy = gamma_der(y, nlsys=nlsys, Dy_inv=Dy_inv)
            return dgamma_dy
        if derivatives:
            if optimization_method == 'trust-constr':
                ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(len(ineq_constr_lb)+len(ineq_constr_ub)),np.inf*np.ones(len(ineq_constr_lb)+len(ineq_constr_ub)),jac=g_jac,keep_feasible=stay_within_bounds)
            elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
                ineq_constr_fun  = {'type':'ineq','fun':g,'jac':g_jac}
        else:
            if optimization_method == 'trust-constr':
                ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(len(ineq_constr_lb)+len(ineq_constr_ub)),np.inf*np.ones(len(ineq_constr_lb)+len(ineq_constr_ub)),keep_feasible=stay_within_bounds)
            elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
                ineq_constr_fun  = {'type':'ineq','fun':g}
    else:
        ineq_constr_fun = None

    # define linear inequality constraints, i.e. define bounds
    lb_ineq = np.concatenate((u_lb,slack_lb,xLF_lb))
    ub_ineq = np.concatenate((u_ub,slack_ub,xLF_ub))
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        lb_ineq = Dy.dot(lb_ineq)
        ub_ineq = Dy.dot(ub_ineq)
    if not link_limits: # no limits on gas link flows or on heat half links flows.
        if formulation.get('gas') == 'full':
            lb_ineq[len(u_lb)+len(slack_init):len(u_lb)+len(slack_init)+len(nlsys.ind_xg_q)] = -np.inf
            ub_ineq[len(u_ub)+len(slack_init):len(u_lb)+len(slack_init)+len(nlsys.ind_xg_q)] = np.inf
        if formulation.get('heat') == 'half_link_flow':
            lb_ineq[len(u_lb)+len(slack_init)+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.unknown_m_links):len(u_lb)+len(slack_init)+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.unknown_m_links)+len(nlsys.unknown_m_halflinks)] = -np.inf
            ub_ineq[len(u_ub)+len(slack_init)+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.unknown_m_links):len(u_lb)+len(slack_init)+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.unknown_m_links)+len(nlsys.unknown_m_halflinks)] = np.inf

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
                gas_net,elec_net,heat_net,het_net = reset_network_without_update(gas_net,elec_net,heat_net,het_net)
                V2, P2, To2, dphi2 = u_init # unscaled
                gas_net,elec_net,heat_net,het_net = update_bc(gas_net,elec_net,heat_net,het_net,V2,P2,To2,dphi2)
                het_net.update(xLF_init,formulation=formulation) # unscaled
                f_vec_global = list()
                y_f_vec = y0.copy()
                err_LF_vec_global = list()
                if optimization_method == 'trust-constr':
                    f_vec = list()
                elif optimization_method == 'SLSQP':
                    f_vec = [obj(y0)] # this call to obj() alters all the global variables.
            opf_start_time = time.perf_counter()
            try:
                if derivatives:
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
                else:
                    if ineq_constr_fun != None:
                        if optimization_method == 'trust-constr':
                            res = spo.minimize(obj, y0, method=optimization_method, constraints=[nonlinear_constraint, ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter,'factorization_method':'SVDFactorization'}, bounds=bounds,tol=tol, callback=callback) #specify factorization method to avoid weird error about matrix not being square. Another options seems to set the bounds of the equality constraint not exactly equal. See https://stackoverflow.com/questions/61753007/how-to-solve-the-problem-of-the-valueerror-expected-square-matrix-in-a-constra
                            execution_times.append(res.execution_time)
                        elif optimization_method == 'SLSQP':
                            res = spo.minimize(obj, y0, method=optimization_method, constraints=[nonlinear_constraint,ineq_constr_fun], options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                            execution_times.append(time.perf_counter() - opf_start_time)
                        elif optimization_method == 'ipopt':
                            res = ipopt.minimize_ipopt(obj, y0, constraints=[nonlinear_constraint,ineq_constr_fun], options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                            execution_times.append(time.perf_counter() - opf_start_time)
                    else:
                        if optimization_method == 'trust-constr':
                            res = spo.minimize(obj, y0, method=optimization_method, constraints=nonlinear_constraint, options={'verbose': 1,'maxiter':max_iter,'factorization_method':'SVDFactorization'}, bounds=bounds,tol=tol, callback=callback)
                            execution_times.append(res.execution_time)
                        elif optimization_method == 'SLSQP':
                            res = spo.minimize(obj, y0, method=optimization_method, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                            execution_times.append(time.perf_counter() - opf_start_time)
                        elif optimization_method == 'ipopt':
                            res = ipopt.minimize_ipopt(obj, y0, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                            execution_times.append(time.perf_counter() - opf_start_time)
            except:
                print('Exception made for {}, link limit: {}, gas form.: {}, heat form.: {}, hard bounds: {}, scaling: {}'.format(optimization_method,link_limits,formulation.get('gas'),formulation.get('heat'),stay_within_bounds,scale_var))
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
        y_opf = res.x.copy()

    # print solution
    V2, P2, To2, dphi2 = y_opf[:len(u_init)] # unscaled
    gas_net,elec_net,heat_net,het_net = update_bc(gas_net,elec_net,heat_net,het_net,V2,P2,To2,dphi2)
    xmes_opt = xmes_from_yopf(y_opf,nlsys=nlsys) # unscaled
    print('Solution OF (success: {})'.format(res.success))
    if len(xmes_opt) < 40:
        gas_net,elec_net,heat_net,het_net = reset_network_without_update(gas_net,elec_net,heat_net,het_net)
        p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation)
        print('p gas = {} bar'.format(p_g_vec/bar))
        print('q = {} kg/s'.format(q_vec))
        print('q nodal inj = {} kg/s'.format(q_inj))
        print('delta = {}'.format(delta_vec))
        print('|V| = {} kV'.format(V_mag_vec/kV))
        print('P edge = {} MW'.format(P_edge/MW))
        print('Q edge = {} MW'.format(Q_edge/MW))
        print('S nodal inj = {} MW'.format(S_inj/MW))
        water = heat_net.links[0].link_params.get('carrier')
        rho_w = water.rhon
        grav_const = water.g
        print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
        print('m = {} kg/s'.format(m_vec))
        print('Ts = {} C'.format(Ts_vec))
        print('Tr = {} C'.format(Tr_vec))
        print('m hl = {} kg/s'.format(m_hl_vec))
        print('Ts hl = {}'.format(Ts_hl_vec))
        print('Tr hl = {}'.format(Tr_hl_vec))
        print('phi hl = {}'.format(phi_hl_vec))
        print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
        print('q c = {}'.format(qc_vec))
        print('P c = {} MW'.format(np.array([Pc_vec])/MW))
        print('Q c = {} MW'.format(np.array([Qc_vec])/MW))
        print('m c = {}'.format(mc_vec))
        print('phi c = {} MW'.format(np.array([phic_vec])/MW))
        print('Ts c = {} C'.format(Tsc_vec))
        print('Tr c = {} C'.format(Trc_vec))
    return xmes_opt, res, f_vec, err_LF_vec, execution_times

def get_u_from_net(gas_net,elec_net,heat_net,het_net,scale_var=None,scale_var_params=None):
    """Get the current values of the control variables u from the network.
    """
    V2 = elec_net.nodes[1].get_V(scale_var=scale_var,scale_var_params=scale_var_params)
    P2 = elec_net.nodes[1].half_links[0].get_P(scale_var=scale_var,scale_var_params=scale_var_params)
    To2 = heat_net.nodes[1].half_links[0].get_Ts(scale_var=scale_var,scale_var_params=scale_var_params)
    dphi2 = heat_net.nodes[1].half_links[0].get_dphi(scale_var=scale_var,scale_var_params=scale_var_params)
    return np.array([V2,P2,To2,dphi2])

def solve_lf_in_of(u,nlsys,max_iters=10,tol=1e-6,xLF_init=np.array([])):
    """Solve steady-state LF within an optimization context.
    """
    global err_LF_vec_global
    network_g = nlsys.nlsystemsg[0].gasnetwork
    network_e = nlsys.nlsystemse[0].elecnetwork
    network_h = nlsys.nlsystemsh[0].heatnetwork
    network_mes = nlsys.hetnetwork

    u_net = get_u_from_net(network_g,network_e,network_h,network_mes,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    if len(err_LF_vec_global)==0 or (np.linalg.norm(u-u_net) > tol) or (err_LF_vec_global[-1] > tol):
        V2,P2,To2,dphi2 = u
        network_g,network_e,network_h,network_mes = reset_network_without_update(network_g,network_e,network_h,network_mes)
        network_g,network_e,network_h,network_mes = update_bc(network_g,network_e,network_h,network_mes,V2,P2,To2,dphi2,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        with HiddenPrints():
            xmes,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = network_mes.solve_network(tol,max_iters,solver='NR',formulation=nlsys.formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            if err_vec[-1] >= tol:
                network_g,network_e,network_h,network_mes = reset_network_without_update(network_g,network_e,network_h,network_mes)
                network_mes.update(xLF_init,formulation=nlsys.formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
                xmes,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = network_mes.solve_network(tol,max_iters,solver='NR',formulation=nlsys.formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        err_LF = err_vec[-1]
    else:
        network_g,network_e,network_h,network_mes = reset_network_without_update(network_g,network_e,network_h,network_mes)
        xmes = network_mes.set_x_init(formulation=nlsys.formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = network_mes.update_full(xmes,formulation=nlsys.formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        err_LF = err_LF_vec_global[-1]
    err_LF_vec_global.append(err_LF)
    q1 = network_g.nodes[0].half_links[0].get_q(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    P1 = network_e.nodes[0].half_links[0].get_P(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    Q2 = network_e.nodes[1].half_links[0].get_Q(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    slack = np.array([q1,P1,Q2])
    x = np.concatenate((slack,xmes)) # unscaled unless per unit scaling is used
    return x

def run_optimal_load_flow_separate_LF(u_lb=np.array([.98*50*kV,-1.5*MW,80,-1.5*MW]),u_ub=np.array([1.02*50*kV,0,110,0]),u_init=np.array([50*kV,-1*MW,100,-1*MW]),slack_lb=np.array([-2,-2*1.5*MW,-2*1.5*MW]),slack_ub=np.array([0,0,2*1.5*MW]),slack_init=np.array([-1,-1.5*MW,-1.5*MW]),xLF_lb=np.array([1*mbar,1*mbar,-np.pi,-np.pi,.98*50*kV,-10,-10,0,0,60,60,0,0,0,0,-3*MW,0,0,0,60]),xLF_ub=np.array([10*bar,10*bar,np.pi,np.pi,1.02*50*kV,10,10,10*bar,10*bar,110,110,55,55,55,10,3*MW,3*MW,10,3*MW,100]),xLF_init=[],n=0,m=0,s=0,p1g=9*bar,q3=1,V1=50*kV,delta1=0.,P3=1.5*MW,Q3=1.5*MW,Ts1=100.,p1h=9*bar,To3=50.,phi3=1.5*MW,L12=4*km,L23=5*km,Dg=10*cm,E=.98,link_type_g='pipe_high_pres_weymouth',hydr_eq_gas='fb',De=1*cm,link_type_e='pi_line',Dh=0.15,lam=0.2,link_type_h='standard_pipe_low_pres_pole',coupling_type='EH',coupling_point=1,q_init=1,m_init=10,Tr_sink_init=50,p_perc_high=.99,p_perc_low=.98,T_perc_high=.99,T_perc_low=.98,qc_init=1,Pc_init=1*MW,Qc_init=1*MW,phic_init=1*MW,y_ind=[4,5,1,22,3,25],a=np.array([0,0,0,0,0,0]),b=np.array([.01*MES3N_line.GHV,.3,.3,.2,.05,.04]),c=np.array([1e-6*(MES3N_line.GHV)**2,3e-5,3e-5,2e-5,4.5e-4,4e-4]), ineq_constr_lb=np.array([-2,-2,-10,0]), ineq_constr_ub=np.array([2,2,(3*MW)**2,(3*MW)**2,0,10]),link_limits=False,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None},scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,max_iters_lf=10,approach='direct',optimization_method='trust-constr',stay_within_bounds=False,fb=None,runs=1,derivatives=False):
    """Run optimal load flow, with LF as equality constraints.

    Parameters
    ----------
    link_limits : bool, optional
        Determines if limits are imposed on the gas link flows, the electrical link powers, and the water half link flows. Default is False.

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
    if coupling_type != 'EH' or coupling_point != 1:
        raise ValueError('Only implemented for coupling at node 1 with an EH (using node set 1)')
    print('\nRunning OPF for MES3N line with separate LF (link limits: {}, formulations: {}, hard bounds: {}, method: {}, scaling: {}, approach: {})'.format(link_limits,formulation,stay_within_bounds,optimization_method,scale_var,approach))

    # create network
    V2, P2, To2, dphi2 = u_init
    with HiddenPrints():
        gas_net,elec_net,heat_net,het_net = create_network(n=n,m=m,s=s,p1g=p1g,q3=q3,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi2=dphi2,phi3=phi3,L12=L12,L23=L23,Dg=Dg,E=E,link_type_g=link_type_g,hydr_eq_gas=hydr_eq_gas,De=De,link_type_e=link_type_e,Dh=Dh,lam=lam,link_type_h=link_type_h,coupling_type=coupling_type,coupling_point=coupling_point)

    # set initial guess and initialize network, using default initial guess
    het_net.initialize() # needed to create nlsys
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    if len(xLF_init) == 0:
        xLF_init = set_x_LF_init(nlsys,n=n,m=m,s=s,q_init=q_init,m_init=m_init,Tr_sink_init=Tr_sink_init,p_perc_high=p_perc_high,p_perc_low=p_perc_low,T_perc_high=T_perc_high,T_perc_low=T_perc_low,qc_init=qc_init,Pc_init=Pc_init,Qc_init=Qc_init,phic_init=phic_init)
    # initial guess for OF (unscaled)
    u0 = u_init

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('Vbase'), scale_var_params.get('Sbase'),scale_var_params.get('Tbase'), scale_var_params.get('phibase')])
        slack_base = np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase')])
        G_base = np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase')])
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
    if not link_limits: # no limits on gas link flows or on heat half links flows.
        if formulation.get('gas') == 'full':
            # link flows are part of xF by default, but limits should not be imposed
            lb_ineq_state = np.delete(lb_ineq_state,[len(slack_init)+ind_l for ind_l in range(len(nlsys.ind_xg_q))])
            ub_ineq_state = np.delete(ub_ineq_state,[len(slack_init)+ind_l for ind_l in range(len(nlsys.ind_xg_q))])
        if formulation.get('heat') == 'half_link_flow':
            # half link flows are part of xF by default, but limits should not be imposed
            lb_ineq_state = np.delete(lb_ineq_state,[len(slack_init)+len(nlsys.ind_xg_p)+len(nlsys.xe_entries)+len(nlsys.unknown_m_links)+ind_hl for ind_hl in range(len(nlsys.unknown_m_halflinks))])
            ub_ineq_state = np.delete(ub_ineq_state,[len(slack_init)+len(nlsys.ind_xg_p)+len(nlsys.xe_entries)+len(nlsys.unknown_m_links)+ind_hl for ind_hl in range(len(nlsys.unknown_m_halflinks))])

    # define objective function
    def obj(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb,xLF_init=xLF_init):
        global u_f_vec
        u_f_vec = u.copy()
        global f_vec_global
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init)
        y = np.concatenate((u,x))
        f = objective_function(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f
    def obj_grad(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb,xLF_init=xLF_init,Dy_inv=Dy_inv,Dh=Dh,method=approach):
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init)
        y = np.concatenate((u,x))
        # partial derivatives of objective
        deltaf_deltay = grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb,Dy_inv=Dy_inv)
        deltaf_deltau = np.zeros((1,len(u)))
        deltaf_deltax = np.zeros((1,len(x)))
        deltaf_deltau[0,:] = deltaf_deltay[:len(u)]
        deltaf_deltax[0,:] = deltaf_deltay[len(u):]
        # partial derivatives of equatilty constraints / load-flow equations
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_h = nlsys.nlsystemsh[0].heatnetwork
        network_mes = nlsys.hetnetwork
        V2, P2, To2, dphi2 = u # scaled for p.u., unscaled for matrix
        network_g,network_e,network_h,network_mes = update_bc(network_g,network_e,network_h,network_mes,V2,P2,To2,dphi2,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
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
    def g(u,nlsys=nlsys, lb_ineq_state=lb_ineq_state, ub_ineq_state=ub_ineq_state, ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub, link_limits=link_limits, Dy=Dy, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init):
        if nlsys.scale_var == 'matrix':
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init)
        y = np.concatenate((u,x))
        if scale_var == 'matrix': # lb_ineq_state and ub_ineq_state are scaled, so scale x as well
            x = Dy[len(u):,len(u):].dot(x)
        if not link_limits: # no limits on gas link flows or on heat half links flows.
            len_slack = len(x) - len(nlsys.x_entries)
            if formulation.get('gas') == 'full':
                # link flows are part of xF by default, but limits should not be imposed
                x = np.delete(x,[len_slack+ind_l for ind_l in range(len(nlsys.ind_xg_q))])
            if formulation.get('heat') == 'half_link_flow':
                # half link flows are part of xF by default, but limits should not be imposed
                x = np.delete(x,[len_slack+len(nlsys.ind_xg_p)+len(nlsys.xe_entries)+len(nlsys.unknown_m_links)+ind_hl for ind_hl in range(len(nlsys.unknown_m_halflinks))])
        g = np.concatenate((x-lb_ineq_state,ub_ineq_state-x))
        if link_limits:
            network_g = nlsys.nlsystemsg[0].gasnetwork
            network_e = nlsys.nlsystemse[0].elecnetwork
            network_h = nlsys.nlsystemsh[0].heatnetwork
            network_mes = nlsys.hetnetwork
            V2, P2, To2, dphi2 = u # scaled for p.u., unscaled for matrix
            network_g,network_e,network_h,network_mes = update_bc(network_g,network_e,network_h,network_mes,V2,P2,To2,dphi2,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
            gam = gamma(y,nlsys=nlsys,ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub)
            g = np.concatenate((gam,g))
        return g
    def g_jac(u,nlsys=nlsys, link_limits=link_limits, Dy=Dy, Dh=Dh, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init,method=approach):
        if nlsys.scale_var == 'matrix':
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init)
        y = np.concatenate((u,x))
        # Jacobian of inequality constraints on state variables (without gamma, that part is added later)
        len_x = len(x)
        I = np.eye(len_x)
        if not link_limits: # no limits on gas link flows or on heat half links flows.
            if formulation.get('gas') == 'full':
                # link flows are part of xF by default, but limits should not be imposed
                I = np.delete(I,[ind_l for ind_l in range(len(nlsys.ind_xg_q))],0)
            if formulation.get('heat') == 'half_link_flow':
                # half link flows are part of xF by default, but limits should not be imposed
                I = np.delete(I,[len(nlsys.ind_xg_p)+len(nlsys.xe_entries)+len(nlsys.unknown_m_links)+ind_hl for ind_hl in range(len(nlsys.unknown_m_halflinks))],0)
        deltag_deltax = np.vstack((I,-I))
        deltag_deltau = np.zeros((deltag_deltax.shape[0],len(u)))
        # partial derivatives of equatilty constraints / load-flow equations
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_h = nlsys.nlsystemsh[0].heatnetwork
        network_mes = nlsys.hetnetwork
        V2, P2, To2, dphi2 = u # scaled for p.u., unscaled for matrix
        network_g,network_e,network_h,network_mes = update_bc(network_g,network_e,network_h,network_mes,V2,P2,To2,dphi2,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        deltah_deltay = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh)
        deltah_deltau = deltah_deltay[:,:len(u)]
        deltah_deltax = deltah_deltay[:,len(u):]
        if link_limits:
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

    if derivatives:
        if optimization_method == 'trust-constr':
            len_g = len(lb_ineq_state) + len(ub_ineq_state)
            if link_limits:
                len_g += len(ineq_constr_lb) + len(ineq_constr_ub)
            ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(len_g),np.inf*np.ones(len_g),jac=g_jac,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            ineq_constr_fun  = {'type':'ineq','fun':g,'jac':g_jac}
    else:
        if optimization_method == 'trust-constr':
            len_g = len(lb_ineq_state) + len(ub_ineq_state)
            if link_limits:
                len_g += len(ineq_constr_lb) + len(ineq_constr_ub)
            ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(len_g),np.inf*np.ones(len_g),keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            ineq_constr_fun  = {'type':'ineq','fun':g}

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
                gas_net,elec_net,heat_net,het_net = reset_network_without_update(gas_net,elec_net,heat_net,het_net)
                V2, P2, To2, dphi2 = u_init # unscaled
                gas_net,elec_net,heat_net,het_net = update_bc(gas_net,elec_net,heat_net,het_net,V2,P2,To2,dphi2)
                het_net.update(xLF_init,formulation=formulation) # unscaled
                f_vec_global = list()
                u_f_vec = u0.copy()
                err_LF_vec_global = list()
                if optimization_method == 'trust-constr':
                    f_vec = list()
                elif optimization_method == 'SLSQP':
                    f_vec = [obj(u0)] # this call to obj() alters all the global variables.
            opf_start_time = time.perf_counter()
            try:
                if derivatives:
                    if optimization_method == 'trust-constr':
                        res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=[ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter,'factorization_method':'SVDFactorization'}, bounds=bounds, callback=callback)
                        execution_times.append(res.execution_time)
                    elif optimization_method == 'SLSQP':
                        res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                        execution_times.append(time.perf_counter() - opf_start_time)
                    elif optimization_method == 'ipopt':
                        res = ipopt.minimize_ipopt(obj, u0,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                        execution_times.append(time.perf_counter() - opf_start_time)
                else:
                    if optimization_method == 'trust-constr':
                        res = spo.minimize(obj, u0, method=optimization_method, constraints=[ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter,'factorization_method':'SVDFactorization'}, bounds=bounds, callback=callback)
                        execution_times.append(res.execution_time)
                    elif optimization_method == 'SLSQP':
                        res = spo.minimize(obj, u0, method=optimization_method, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                        execution_times.append(time.perf_counter() - opf_start_time)
                    elif optimization_method == 'ipopt':
                        res = ipopt.minimize_ipopt(obj, u0,constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                        execution_times.append(time.perf_counter() - opf_start_time)
            except:
                print('Exception made for {} (link limit: {}, gas form.: {}, heat form.: {}, hard bounds: {}, scaling: {}, approach: {})'.format(optimization_method,link_limits,formulation.get('gas'),formulation.get('heat'),stay_within_bounds,scale_var,approach))
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
        x = solve_lf_in_of(res.x,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init)
        y_opt = Dy_inv.dot(np.concatenate((res.x,x))) # unscaled
    else:
        x = solve_lf_in_of(u_opf,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=xLF_init)
        y_opt = np.concatenate((u_opf,x)) # unscaled
    xmes_opt = xmes_from_yopf(y_opt,nlsys=nlsys) # unscaled
    print('Solution OF (success: {})'.format(res.success))
    if len(xmes_opt) < 40:
        gas_net,elec_net,heat_net,het_net = reset_network_without_update(gas_net,elec_net,heat_net,het_net)
        p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation)
        print('p gas = {} bar'.format(p_g_vec/bar))
        print('q = {} kg/s'.format(q_vec))
        print('q nodal inj = {} kg/s'.format(q_inj))
        print('delta = {}'.format(delta_vec))
        print('|V| = {} kV'.format(V_mag_vec/kV))
        print('P edge = {} MW'.format(P_edge/MW))
        print('Q edge = {} MW'.format(Q_edge/MW))
        print('S nodal inj = {} MW'.format(S_inj/MW))
        water = heat_net.links[0].link_params.get('carrier')
        rho_w = water.rhon
        grav_const = water.g
        print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
        print('m = {} kg/s'.format(m_vec))
        print('Ts = {} C'.format(Ts_vec))
        print('Tr = {} C'.format(Tr_vec))
        print('m hl = {} kg/s'.format(m_hl_vec))
        print('Ts hl = {}'.format(Ts_hl_vec))
        print('Tr hl = {}'.format(Tr_hl_vec))
        print('phi hl = {}'.format(phi_hl_vec))
        print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
        print('q c = {}'.format(qc_vec))
        print('P c = {} MW'.format(np.array([Pc_vec])/MW))
        print('Q c = {} MW'.format(np.array([Qc_vec])/MW))
        print('m c = {}'.format(mc_vec))
        print('phi c = {} MW'.format(np.array([phic_vec])/MW))
        print('Ts c = {} C'.format(Tsc_vec))
        print('Tr c = {} C'.format(Trc_vec))
    return xmes_opt, y_opt, res, f_vec, err_LF_vec, execution_times

def compare_forms(dir_path=None,save_tables=False,save_figs=False,save_CPU_times=False,save_data=False,save_parameters=False,number_of_runs=1,n=0,m=0,s=0,N_max = 100,max_iter=25,scale_var=None):
    """Compare OF for different optimization methods, formulations of OF, scalings, and bounds."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    max_iters_lf = 20
    tol = 1e-6

    # scaling (is used in solving LF by default)
    pgbase = 10*bar
    qbase = 1.
    deltabase = 1.
    Vbase = 50*kV
    Sbase = 1*MW
    phbase = 10*bar
    mbase = 1
    Tbase = 100
    phibase = 1*MW
    pbase = pgbase
    Ebase = 1*MW
    if scale_var == None:
        scale_var_params = None
        fb = None
        scale_label = 'unscaled'
    else:
        fb = 1#1e8
        scale_var_params = {'pgbase':pgbase,'qbase':qbase,'deltabase':deltabase,'Vbase':Vbase,'Sbase':Sbase,'phbase':phbase,'pbase':pbase,'mbase':mbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Ebase}
        if scale_var == 'matrix':
            scale_label = 'matrix'
        else:
            scale_label = 'pu'

    # parameter values for the objective function (assumes E = [q1,P1,P2,P1c,dphi2,dphi1c])
    a=np.array([0,0,0,0,0,0])#np.array([0,0,0,0,0,0])
    b=np.array([15*MES3N_line.GHV,40,40,2,16,1])/MW#np.array([.01*MES3N_line.GHV,.2,.3,.3,.05,.04])
    c=np.array([0,0,0,b[3],0,b[5]])/(100*MW)

    # Network parameters and boundary conditions (of LF)
    p1g = 50*bar
    q3 = 1
    Dg=10*cm
    E=.98
    link_type_g='pipe_high_pres_weymouth'
    V1 = 50*kV #[V]
    delta1 = 0.#[rad]
    De=1*cm
    link_type_e='pi_line'
    Ts1=100.
    p1h=9*bar
    To3=50.
    link_type_h='standard_pipe_low_pres_pole'
    if s == 0:
        To2=84.346642#84.82293161333068#100.
        phi2=-1.00175442*MW#-1.0469065288680548*MW#-1*MW
        phi3=1.5*MW
        Dh=15*cm
        lam=0.2
        V2 = 49.9854119*kV#49.98856105*kV #[V]
        P2 = -.4*MW#-0.5999727456238263*MW #[W]
        P3 = 1.5*MW #[W]
        Q3 = 1.5*MW #[Var]
    else:
        To2=84.346642#84.82293161333068#100.
        phi2=-1.00175442*MW#-1.0469065288680548*MW#-1*MW
        phi3=5.5*1.0469065288680548*MW#5.5*-phi2 #[W]
        Dh=30*cm
        lam=0.002
        V2 = 49.9854119*kV#49.98856105*kV #[V]
        P2 = -.4*MW#-0.5999727456238263*MW #[W]
        P3 = .9*phi3#7*-P2 #[W]
        Q3 = .9*phi3#7*-P2 #[Var]
    L12=4*km
    L23=5*km
    coupling_type='EH'
    coupling_point=1

    # initial guess
    if (n == 1 and m == 0) or n==m:
        p_perc_high = .99
        p_perc_low = .98
        T_perc_high = 1.
        T_perc_low = 1.
    else:
        p_perc_step = .001/(2*n-m)
        if 3+s*(2*n-m+1) < 35:
            p_perc_high = min(.99,1-p_perc_step)
            p_per_low = min(.97,1-p_perc_step*(n-m))
        else:
            p_perc_high = 1-p_perc_step
            p_per_low = 1-p_perc_step*(n-m)
        p_perc_low = max(.1,p_per_low)
        T_perc_step = .001/(2*n-m)
        T_perc_high = 1-T_perc_step#1.
        T_perc_low = max(.5,1-T_perc_step*(n-m))
    print('p perc high = {}, p perc low = {}\nT perc high = {}, T perc low = {}'.format(p_perc_high,p_perc_low,T_perc_high,T_perc_low))

    q_init=1
    m_init=10
    Tr_sink_init=50
    Pc_init=P3+P2#1.5*MW#1*MW
    Qc_init=Q3#1*MW
    phic_init=phi3#1.5*MW#1*MW
    qc_init=(P3+phi3)/MES3N_line.GHV#.1#1
    u_init=np.array([50*kV,1.2*P2,100,1.1*phi2])#np.array([50*kV,-1*MW,100,-1*MW])
    slack_init=np.array([-(q3+qc_init),.5*P2,-1.5*MW])#np.array([-(q3+qc_init),-1.5*MW,-1.5*MW])
    # steady-state LF solution
    with HiddenPrints():
        formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
        hydr_eq_gas = 'fb'
        scale_var_params_LF = {'pgbase':pgbase,'qbase':qbase,'deltabase':deltabase,'Vbase':Vbase,'Sbase':Sbase,'phbase':phbase,'pbase':pbase,'mbase':mbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Ebase}
        gas_net_LF,elec_net_LF,heat_net_LF,het_net_LF = create_network(n=n,m=m,s=s,p1g=p1g,q3=q3,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi2=phi2,phi3=phi3,L12=L12,L23=L23,Dg=Dg,E=E,link_type_g=link_type_g,hydr_eq_gas=hydr_eq_gas,De=De,link_type_e=link_type_e,Dh=Dh,lam=lam,link_type_h=link_type_h,coupling_type=coupling_type,coupling_point=coupling_point)
        het_net_LF.initialize()
        nlsys_LF = NonLinearSystemHeterogeneous(het_net_LF,formulation=formulation,scale_var='matrix',scale_var_params=scale_var_params_LF)
        x_init = set_x_LF_init(nlsys_LF,n=n,m=m,s=s,q_init=q_init,m_init=m_init,Tr_sink_init=Tr_sink_init,p_perc_high=p_perc_high,p_perc_low=p_perc_low,T_perc_high=T_perc_high,T_perc_low=T_perc_low)
        het_net_LF.update(x_init,formulation=formulation)
        x_sol_LF,iters_LF,err_vec_LF,p_g_vec_LF,q_vec_LF,q_inj_LF,delta_vec_LF,V_mag_vec_LF,S_inj_LF,P_edge_LF,Q_edge_LF,m_vec_LF,p_h_vec_LF,Ts_vec_LF,Tr_vec_LF,m_hl_vec_LF,phi_hl_vec_LF,Ts_hl_vec_LF,Tr_hl_vec_LF,qc_vec_LF,Pc_vec_LF,Qc_vec_LF,mc_vec_LF,phic_vec_LF,Tsc_vec_LF,Trc_vec_LF = het_net_LF.solve_network(tol,max_iters_lf,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params=scale_var_params_LF)
        q1 = gas_net_LF.nodes[0].half_links[0].get_q()
        P1 = elec_net_LF.nodes[0].half_links[0].get_P()
        Q2 = elec_net_LF.nodes[1].half_links[0].get_Q()
        y_LF = np.concatenate((np.array([V2,P2,To2,phi2,q1,P1,Q2]),x_sol_LF)) # unscaled
        y_ind=[len(u_init),len(u_init)+1,1,len(u_init)+len(slack_init)+len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries)+len(nlsys_LF.xh_entries)+1,3,len(u_init)+len(slack_init)+len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries)+len(nlsys_LF.xh_entries)+4]
        E_LF = y_LF[np.array(y_ind)]
        E_LF[0] = E_LF[0]*MES3N_line.GHV
        E_LF = E_LF/MW
        powers = np.array([qc_vec_LF[0]*MES3N_line.GHV,P1,P2,Pc_vec_LF[0],phi2,phic_vec_LF[0],Q2,Qc_vec_LF[0]])/MW
        DF = nlsys_LF.DF()
    print('err LF = {:.3e} after {} iters'.format(err_vec_LF[-1],iters_LF))

    # bounds (partly based on LF solution)
    pg_min = .8*p1g
    pg_max = 1.1*p1g
    q_min = -2
    q_max = 2
    xLF_g_lb_nodal = pg_min*np.ones(len(nlsys_LF.ind_xg_p))
    xLF_g_lb_full = np.concatenate((q_min*np.ones(len(nlsys_LF.ind_xg_q)),xLF_g_lb_nodal))
    xLF_g_ub_nodal = pg_max*np.ones(len(nlsys_LF.ind_xg_p))
    xLF_g_ub_full = np.concatenate((q_max*np.ones(len(nlsys_LF.ind_xg_q)),xLF_g_ub_nodal))
    dummy_links_gas = len(list(gas_net_LF.get_links(link_types=['dummy'])))
    ineq_constr_g_lb=q_min*np.ones(len(gas_net_LF.links)-dummy_links_gas)
    ineq_constr_g_ub=q_max*np.ones(len(gas_net_LF.links)-dummy_links_gas)
    if s==0:
        u_e_lb=np.array([.98*50*kV,-1*MW])
        u_e_ub=np.array([1.01*50*kV,-.4*MW])#np.array([1.01*50*kV,-.1*MW])
        slack_e_lb=np.array([-2*1.5*MW,-2*1.5*MW])
        slack_e_ub=np.array([0,2*1.5*MW])
        ineq_constr_e_ub=np.array([(2*MW)**2+(2*MW)**2,(2*MW)**2+(2*MW)**2])
        xLF_e_lb=np.array([-np.pi/6,-np.pi/6,.98*V1])
        xLF_e_ub=np.array([np.pi/6,np.pi/6,1.02*V1])
        T_loads = 2
        m_max = 10
        Ts_max = 110
        Ts_min = 60
        Tr_max = 55
        Tr_min = 0
    else:
        u_e_lb=np.array([.98*V2,1.3*P2])
        u_e_ub=np.array([1.02*V2,-.4*MW])#np.array([1.02*V2,-.1*MW])
        slack_e_lb=np.array([-2*np.abs(S_inj_LF[0].real),-2*np.abs(S_inj_LF[1].imag)])
        slack_e_ub=np.array([0,2*np.abs(S_inj_LF[1].imag)])
        dummy_links_elec = len(list(elec_net_LF.get_links(link_types=['dummy'])))
        Ne = len(elec_net_LF.links)
        non_dummy_links_elec = np.array([ind_l for ind_l, link in enumerate(elec_net_LF.get_links()) if link.link_type != 'dummy'])
        ineq_constr_e_ub=(1.5*np.abs(P_edge_LF[non_dummy_links_elec]))**2+(1.5*np.abs(Q_edge_LF[non_dummy_links_elec]))**2
        xLF_e_lb=np.concatenate((-np.pi*np.ones(len(nlsys_LF.unknown_delta_nodes)),.98*np.min(V_mag_vec_LF)*np.ones(len(nlsys_LF.unknown_V_nodes))))
        xLF_e_ub=np.concatenate((np.pi*np.ones(len(nlsys_LF.unknown_delta_nodes)),1.02*np.max(V_mag_vec_LF)*np.ones(len(nlsys_LF.unknown_V_nodes))))
        print('delta between {} and {}'.format(np.min(delta_vec_LF),np.max(delta_vec_LF)))
        T_loads = n*s+1
        m_max = 1.1*heat_net_LF.links[1].get_m()
        Ts_max = 1.1*np.max(Ts_vec_LF)
        Tr_min = 0
        Tr_max = 1*np.max(Tr_vec_LF)
        Ts_min = 1.1*Tr_max#1.5*Tr_max
    mhl_lb = np.zeros(T_loads)
    mhl_lb[0] = -m_max
    mhl_ub = m_max*np.ones(T_loads)
    mhl_ub[0] = 0.1
    xLF_h_lb_standard=np.concatenate((-m_max*np.ones(len(nlsys_LF.unknown_m_links)),np.zeros(len(nlsys_LF.unknown_p_nodes)),Ts_min*np.ones(len(nlsys_LF.unknown_Ts_nodes)),Tr_min*np.ones(len(nlsys_LF.unknown_Tr_nodes))))
    xLF_h_ub_standard=np.concatenate((m_max*np.ones(len(nlsys_LF.unknown_m_links)),10*bar*np.ones(len(nlsys_LF.unknown_p_nodes)),Ts_max*np.ones(len(nlsys_LF.unknown_Ts_nodes)),Tr_max*np.ones(len(nlsys_LF.unknown_Tr_nodes))))
    xLF_h_lb_term = np.concatenate((xLF_h_lb_standard[:len(nlsys_LF.unknown_m_links)],mhl_lb,xLF_h_lb_standard[len(nlsys_LF.unknown_m_links):]))
    xLF_h_ub_term = np.concatenate((xLF_h_ub_standard[:len(nlsys_LF.unknown_m_links)],mhl_ub,xLF_h_ub_standard[len(nlsys_LF.unknown_m_links):]))
    ineq_constr_h_lb=mhl_lb.copy()
    ineq_constr_h_ub=mhl_ub.copy()
    if s == 0:
        xLF_c_lb=np.array([0.1*MW/MES3N_line.GHV,0.1*MW,-2*np.abs(Q3),0.1,0.1*MW,Ts_min])
        xLF_c_ub=np.array([2*(np.abs(P3)+np.abs(phi3))/MES3N_line.GHV,2*np.abs(P3),2*np.abs(Q3),m_max,2*phi3,Ts_max])
        slack_lb=np.concatenate([np.array([1.2*q1]),slack_e_lb])
        slack_ub=np.concatenate([np.array([-q3]),slack_e_ub])
        u_lb=np.concatenate((u_e_lb,np.array([.9*To2,1.2*phi2])))
        u_ub=np.concatenate((u_e_ub,np.array([1.1*To2,.9*phi2])))#np.concatenate((u_e_ub,np.array([1.1*To2,.7*phi2])))
    else:
        xLF_c_lb=np.array([0.1*MW/MES3N_line.GHV,0.1*MW,-2*np.abs(Q3),0.1,0.1*MW,Ts_min])
        xLF_c_ub=np.array([2*(np.abs(P3)+np.abs(phi3))/MES3N_line.GHV,1.5*np.abs(P3),1.5*np.abs(Q3),m_max,2*phi3,Ts_max])
        slack_lb=np.concatenate([np.array([1.2*q1]),slack_e_lb])
        slack_ub=np.concatenate([np.array([-q3]),slack_e_ub])
        u_lb=np.concatenate((u_e_lb,np.array([.9*To2,1.2*phi2])))
        u_ub=np.concatenate((u_e_ub,np.array([1.1*To2,.9*phi2])))#np.concatenate((u_e_ub,np.array([1.1*To2,.7*phi2])))

    xg_sol_LF_full = x_sol_LF[:len(nlsys_LF.xg_entries)]
    xg_sol_LF_nodal = xg_sol_LF_full[-len(nlsys_LF.ind_xg_p):]
    xe_sol_LF = x_sol_LF[len(nlsys_LF.xg_entries):len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries)]
    xh_sol_LF_term = x_sol_LF[len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries):len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries)+len(nlsys_LF.xh_entries)]
    xh_sol_LF_standard = np.concatenate((xh_sol_LF_term[:len(nlsys_LF.unknown_m_links)],xh_sol_LF_term[len(nlsys_LF.unknown_m_links)+len(nlsys_LF.unknown_m_halflinks):]))
    xc_sol_LF = x_sol_LF[-len(nlsys_LF.xc_entries):]

    # scale LF solution (assuming half_link_flow formulation)
    if scale_var != None:
        nlsys_LF = NonLinearSystemHeterogeneous(het_net_LF,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        Dx = nlsys_LF.Dx()
        ubase = np.array([scale_var_params.get('Vbase'),scale_var_params.get('Sbase'),scale_var_params.get('Tbase'), scale_var_params.get('phibase')])
        slack_base = np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase')])
        xbase_inv = Dx.data[0]
        xbase_g_inv = xbase_inv[:len(nlsys_LF.xg_entries)]
        xbase_e_inv = xbase_inv[len(nlsys_LF.xg_entries):len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries)]
        xbase_h_inv = xbase_inv[len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries):len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries)+len(nlsys_LF.xh_entries)]
        xbase_c_inv = xbase_inv[-len(nlsys_LF.xc_entries):]
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,xbase_inv)))
        f_LF = objective_function(y_LF,y_ind,a=a,b=b,c=c)/fb #scaled
        y_LF = Dy.dot(y_LF) # scaled
    else:
        f_LF = objective_function(y_LF,y_ind,a=a,b=b,c=c,scale_var=scale_var,fb=fb)

    result = dict()
    y_res = dict()
    y_inds = dict()
    err_LF_res = dict()
    f_vec_res = dict()
    parameters = dict()
    parameters['a'] = a.copy()
    parameters['b'] = b.copy()
    parameters['c'] = c.copy()
    parameters['fb'] = fb
    parameters['q3'] = q3
    parameters['scale_var_params'] = scale_var_params
    y_res['LF'] = y_LF.copy()
    err_LF_res['LF'] = err_vec_LF.copy()
    methods = ['SLSQP','ipopt']#['trust-constr','SLSQP','ipopt']
    bounds = ['soft']#, 'hard']
    approaches = ['eq_constr','direct','adjoint']
    forms_gas = ['full','nodal']
    forms_heat = ['half_link_flow','standard']
    link_limits = [False,True]
    an_ders = [True]#,False]
    total_cases = len(methods)*len(bounds)*len(approaches)*len(forms_gas)*len(forms_heat)*len(link_limits)*len(an_ders)
    case_number = 1
    start_wall_clock = datetime.datetime.now().time()
    for derivative in an_ders:
        if derivative:
            der_label = 'an'
        else:
            der_label = 'num'
        for bound in bounds:
            if bound == 'soft':
                stay_within_bounds = False
            else:
                stay_within_bounds = True

            for link_limit in link_limits:
                for form_gas in forms_gas:
                    marker_key_gas = form_gas+', '
                    if form_gas == 'nodal':
                        hydr_eq_gas = 'fa'
                        xLF_g_lb = xLF_g_lb_nodal.copy()
                        xLF_g_ub = xLF_g_ub_nodal.copy()
                        xLF_g_init = xg_sol_LF_nodal.copy()
                    else:
                        hydr_eq_gas = 'fb'
                        xLF_g_lb = xLF_g_lb_full.copy()
                        xLF_g_ub = xLF_g_ub_full.copy()
                        xLF_g_init = xg_sol_LF_full.copy()
                    for form_heat in forms_heat:
                        if form_heat == 'standard':
                            marker_key_heat = form_heat+', '
                            xLF_h_lb = xLF_h_lb_standard.copy()
                            xLF_h_ub = xLF_h_ub_standard.copy()
                            xLF_h_init = xh_sol_LF_standard.copy()
                        else:
                            marker_key_heat = 'term., '
                            xLF_h_lb = xLF_h_lb_term.copy()
                            xLF_h_ub = xLF_h_ub_term.copy()
                            xLF_h_init = xh_sol_LF_term.copy()
                        formulation = {'gas':form_gas,'elec':'complex_power','heat':form_heat,'het':None}
                        xLF_lb = np.concatenate((xLF_g_lb,xLF_e_lb,xLF_h_lb,xLF_c_lb))
                        xLF_ub = np.concatenate((xLF_g_ub,xLF_e_ub,xLF_h_ub,xLF_c_ub))
                        y_ind=[len(u_init),len(u_init)+1,1,len(u_init)+len(slack_init)+len(xLF_g_lb)+len(xLF_e_lb)+len(xLF_h_lb)+1,3,len(u_init)+len(slack_init)+len(xLF_g_lb)+len(xLF_e_lb)+len(xLF_h_lb)+4]
                        y_inds[form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound] = y_ind
                        parameters['y_ind_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound] = y_ind
                        xLF_init = []#np.concatenate((xLF_g_init,xe_sol_LF,xLF_h_init,xc_sol_LF))
                        if link_limit:
                            marker_key_lim = 'limits'
                            if form_gas == 'nodal' and form_heat == 'standard':
                                ineq_constr_lb = np.concatenate((ineq_constr_g_lb,ineq_constr_h_lb))
                                ineq_constr_ub = np.concatenate((ineq_constr_g_ub,ineq_constr_e_ub,ineq_constr_h_ub))
                            elif form_gas == 'nodal':
                                ineq_constr_lb = ineq_constr_g_lb.copy()
                                ineq_constr_ub = np.concatenate((ineq_constr_g_ub,ineq_constr_e_ub))
                            elif form_heat == 'standard':
                                ineq_constr_lb = ineq_constr_h_lb.copy()
                                ineq_constr_ub = np.concatenate((ineq_constr_e_ub,ineq_constr_h_ub))
                            else:
                                ineq_constr_lb = np.array([])
                                ineq_constr_ub = ineq_constr_e_ub.copy()
                        else:
                            marker_key_lim = 'no limits'
                            ineq_constr_lb = np.array([])
                            ineq_constr_ub = np.array([])

                        marker_key = marker_key_gas + marker_key_heat + marker_key_lim
                        fig_f = plt.figure('mes_obj_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
                        ax_f = fig_f.gca()
                        ax_f.set_xlabel('Iteration?')
                        ax_f.set_ylabel('f')

                        fig_LF_error = plt.figure('mes_error_LF_in_OF_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
                        ax_LF_error = fig_LF_error.gca()
                        ax_LF_error.set_xlabel('Iteration?')
                        ax_LF_error.set_ylabel(r'$||F||_2$')

                        fig_E = plt.figure('mes_OF_E_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
                        ax_E = fig_E.gca()
                        ax_E.set_ylabel(r'$E$ [MW]')
                        E_lb = np.concatenate((u_lb,slack_lb,xLF_lb))[np.array(y_ind)]
                        E_lb[0] = E_lb[0]*MES3N_line.GHV
                        E_lb /= MW
                        xticks_E = list(range(len(E_lb)))
                        ax_E.plot(xticks_E,E_lb,'-k',alpha=.5)
                        E_ub = np.concatenate((u_ub,slack_ub,xLF_ub))[np.array(y_ind)]
                        E_ub[0] = E_ub[0]*MES3N_line.GHV
                        E_ub /= MW
                        ax_E.plot(xticks_E,E_ub,'-k',alpha=.5)

                        fig_coupling = plt.figure('mes_OF_coupling_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
                        ax_coupling = fig_coupling.gca()
                        ax_coupling.set_ylabel(r'coupling $E$ [MW]')
                        power_lb = E_lb.copy()
                        power_lb[0] = xLF_lb[-6]*MES3N_line.GHV/MW
                        power_lb = np.concatenate((power_lb,np.array([slack_e_lb[1],xLF_c_lb[2]])/MW))
                        xticks_coupling = list(range(len(power_lb)))
                        ax_coupling.plot(xticks_coupling,power_lb,'-k',alpha=.5)
                        power_ub = E_ub.copy()
                        power_ub[0] = xLF_ub[-6]*MES3N_line.GHV/MW
                        power_ub = np.concatenate((power_ub,np.array([slack_e_ub[1],xLF_c_ub[2]])/MW))
                        ax_coupling.plot(xticks_coupling,power_ub,'-k',alpha=.5)

                        fig_V = plt.figure('mes_V_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
                        ax_V = fig_V.gca()
                        xticks_V = range(0,len(V_mag_vec_LF[1:]))
                        ax_V.set_ylabel(r'$|V|$ [p.u.] (base = {} kV)'.format(scale_var_params_LF.get('Vbase')/kV))
                        V_lb = np.concatenate((np.array([u_lb[0]]),xLF_e_lb[-len(nlsys_LF.unknown_V_nodes):]))/scale_var_params_LF.get('Vbase')
                        V_ub = np.concatenate((np.array([u_ub[0]]),xLF_e_ub[-len(nlsys_LF.unknown_V_nodes):]))/scale_var_params_LF.get('Vbase')
                        ax_V.plot(xticks_V,V_lb,'-k',alpha=.5)
                        ax_V.plot(xticks_V,V_ub,'-k',alpha=.5)

                        max_fev = 0
                        for method in methods:
                            for approach in approaches:
                                start_runs_wall_clock = datetime.datetime.now().time()
                                if approach == 'direct' or approach == 'adjoint':
                                    xmes_opt, y_opt, res, f_vec, err_LF_vec, execution_times = run_optimal_load_flow_separate_LF(u_lb=u_lb,u_ub=u_ub,u_init=u_init,slack_lb=slack_lb,slack_ub=slack_ub,slack_init=slack_init,xLF_lb=xLF_lb,xLF_ub=xLF_ub,xLF_init=xLF_init,n=n,m=m,s=s,p1g=p1g,q3=q3,V1=V1,delta1=delta1,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To3=To3,phi3=phi3,L12=L12,L23=L23,Dg=Dg,E=E,link_type_g=link_type_g,hydr_eq_gas=hydr_eq_gas,De=De,link_type_e=link_type_e,Dh=Dh,lam=lam,link_type_h=link_type_h,coupling_type=coupling_type,coupling_point=coupling_point,q_init=q_init,m_init=m_init,Tr_sink_init=Tr_sink_init,p_perc_high=p_perc_high,p_perc_low=p_perc_low,T_perc_high=T_perc_high,T_perc_low=T_perc_low,qc_init=qc_init,Pc_init=Pc_init,Qc_init=Qc_init,phic_init=phic_init,y_ind=y_ind,a=a,b=b,c=c,ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub,link_limits=link_limit,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,approach=approach,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,runs=number_of_runs,derivatives=derivative)
                                else:
                                    xmes_opt, res, f_vec, err_LF_vec, execution_times = run_optimal_load_flow(u_lb=u_lb,u_ub=u_ub,u_init=u_init,slack_lb=slack_lb,slack_ub=slack_ub,slack_init=slack_init,xLF_lb=xLF_lb,xLF_ub=xLF_ub,xLF_init=xLF_init,n=n,m=m,s=s,p1g=p1g,q3=q3,V1=V1,delta1=delta1,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To3=To3,phi3=phi3,L12=L12,L23=L23,Dg=Dg,E=E,link_type_g=link_type_g,hydr_eq_gas=hydr_eq_gas,De=De,link_type_e=link_type_e,Dh=Dh,lam=lam,link_type_h=link_type_h,coupling_type=coupling_type,coupling_point=coupling_point,q_init=q_init,m_init=m_init,Tr_sink_init=Tr_sink_init,p_perc_high=p_perc_high,p_perc_low=p_perc_low,T_perc_high=T_perc_high,T_perc_low=T_perc_low,qc_init=qc_init,Pc_init=Pc_init,Qc_init=Qc_init,phic_init=phic_init,y_ind=y_ind,a=a,b=b,c=c,ineq_constr_lb=ineq_constr_lb, ineq_constr_ub=ineq_constr_ub,link_limits=link_limit,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,runs=number_of_runs,derivatives=derivative)
                                    y_opt = res.x
                                print('Finished case {} of {}, in average of {:.2f}s per run. Started at {}, started these runs at {}'.format(case_number,total_cases,res.execution_time,start_wall_clock,start_runs_wall_clock))
                                case_number += 1
                                result[method+'_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound+'_'+approach] = res
                                y_res[method+'_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound+'_'+approach] = y_opt # unscaled for direct and adjoint approach. Scaled with eq_constr approach!!!!
                                err_LF_res[method+'_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound+'_'+approach] = err_LF_vec
                                f_vec_res[method+'_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound+'_'+approach] = f_vec
                                max_fev = max(max_fev,len(f_vec))
                                # plot results
                                ax_f.plot(f_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(marker_key),alpha=.5)
                                ax_LF_error.semilogy(err_LF_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(marker_key),alpha=.5)

                                E_opt = y_opt[np.array(y_ind)]
                                E_opt[0] = E_opt[0]*MES3N_line.GHV
                                if scale_var == None or approach != 'eq_constr':
                                    E_opt /= MW
                                else:
                                    E_opt[0] /= MW
                                ax_E.plot(xticks_E,E_opt,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(marker_key),alpha=.5)

                                power_opt = E_opt.copy()
                                power_opt[0] = y_opt[-6]*MES3N_line.GHV
                                power_opt = np.concatenate((power_opt,np.array([y_opt[6],y_opt[-4]])))
                                if scale_var == None or approach != 'eq_constr':
                                    power_opt[len(E_opt):] /= MW
                                power_opt[0] /= MW
                                ax_coupling.plot(xticks_coupling,power_opt,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(marker_key),alpha=.5)

                                V_opt = np.concatenate((np.array([y_opt[0]]),y_opt[len(u_init)+len(slack_init)+len(nlsys_LF.xg_entries)+len(nlsys_LF.unknown_delta_nodes):len(u_init)+len(slack_init)+len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries)]))
                                if scale_var == None or approach != 'eq_constr':
                                    V_opt /= scale_var_params_LF.get('Vbase')
                                ax_V.plot(xticks_V,V_opt,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(marker_key),alpha=.5)

                        ax_f.plot([0,max_fev],[f_LF,f_LF],':r',alpha=.5)
                        ax_LF_error.semilogy([0,max_fev],[tol,tol],':k')
                        ax_E.plot(xticks_E,E_LF,':r',alpha=.5)
                        ax_E.set_xticks(xticks_E)
                        ax_E.set_xticklabels([r'GHV$q_1$',r'$P_1$',r'$P_2$',r'$P_{1^c1^e}$',r'$\Delta\varphi_2$',r'$\Delta\varphi_{1^c1^h}$'])
                        ax_coupling.plot(xticks_coupling,powers,':r',alpha=.5)
                        ax_coupling.plot(xticks_coupling,np.zeros(len(xticks_coupling)),':k',alpha=.5)
                        ax_coupling.set_xticks(xticks_coupling)
                        ax_coupling.set_xticklabels([r'GHV$q_{1^g1^c}$',r'$P_1$',r'$P_2$',r'$P_{1^c1^e}$',r'$\Delta\varphi_2$',r'$\Delta\varphi_{1^c1^h}$',r'$Q_2$',r'$Q_{1^c1^e}$'])
                        ax_V.plot(V_mag_vec_LF[1:]/scale_var_params_LF.get('Vbase'),':r',alpha=.5)
                        # ax_V.set_xticklabels([ind+1 for ind in range(len(V_mag_vec_LF[1:])+1)])
                        ax_V.set_xlim(left=0-.1,right=len(V_mag_vec_LF[1:])-1+.1)
                        ax_V.set_xticks(xticks_V)
                        ax_V.set_xticklabels([ind+1 for ind in xticks_V])#+1 since the first node has known V, +1 since the first node is called 1 and not 0
                        ax_V.set_xlabel('Node number')
                        if not save_figs:
                            ax_f.legend(handles=legend_handles)
                            ax_LF_error.legend(handles=legend_handles)
                            ax_E.legend(handles=legend_handles)
                            ax_coupling.legend(handles=legend_handles)
                            ax_V.legend(handles=legend_handles)

    u_LF_sol = y_LF[:len(u_init)]
    xG_LF_sol = y_LF[len(u_init):len(u_init)+len(slack_init)]
    xg_LF = y_LF[len(u_init)+len(slack_init):len(u_init)+len(slack_init)+len(nlsys_LF.xg_entries)]
    xe_LF_sol = y_LF[len(u_init)+len(slack_init)+len(nlsys_LF.xg_entries):len(u_init)+len(slack_init)+len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries)]
    xh_LF = y_LF[len(u_init)+len(slack_init)+len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries):len(u_init)+len(slack_init)+len(nlsys_LF.xg_entries)+len(nlsys_LF.xe_entries)+len(nlsys_LF.xh_entries)]
    xc_LF_sol = y_LF[-len(nlsys_LF.xc_entries):]
    for derivative in an_ders:
        for bound in bounds:
            for link_limit in link_limits:
                for form_gas in forms_gas:
                    if form_gas == 'nodal':
                        xg_LF_sol = xg_LF[len(nlsys_LF.ind_xg_q):]
                        if scale_var != None:
                            xbase_g_inv_OF = xbase_g_inv[len(nlsys_LF.ind_xg_q):]
                        else:
                            xbase_g_inv_OF = np.ones(len(xbase_g_inv[len(nlsys_LF.ind_xg_q):]))
                    else:
                        xg_LF_sol = xg_LF.copy()
                        if scale_var != None:
                            xbase_g_inv_OF = xbase_g_inv.copy()
                        else:
                            xbase_g_inv_OF = np.ones(len(xbase_g_inv))
                    for form_heat in forms_heat:
                        if form_heat == 'standard':
                            xh_LF_sol = np.concatenate((xh_LF[:len(nlsys_LF.unknown_m_links)],xh_LF[len(nlsys_LF.unknown_m_links)+len(nlsys_LF.unknown_m_halflinks):]))
                            if scale_var != None:
                                xbase_h_inv_OF = np.concatenate((xbase_h_inv[:len(nlsys_LF.unknown_m_links)],xbase_h_inv[len(nlsys_LF.unknown_m_links)+len(nlsys_LF.unknown_m_halflinks):]))
                            else:
                                xbase_h_inv_OF = np.ones(len(nlsys_LF.unknown_m_links)+len(xbase_h_inv[len(nlsys_LF.unknown_m_links)+len(nlsys_LF.unknown_m_halflinks):]))
                        else:
                            xh_LF_sol = xh_LF.copy()
                            if scale_var != None:
                                xbase_h_inv_OF = xbase_h_inv.copy()
                            else:
                                xbase_h_inv_OF = np.ones(len(xbase_h_inv))
                        y_LF_sol = np.concatenate((u_LF_sol,xG_LF_sol,xg_LF_sol,xe_LF_sol,xh_LF_sol,xc_LF_sol))
                        key = form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound
                        y_res['LF_'+key] = y_LF_sol.copy()
                        if scale_var == 'matrix' or scale_var == 'per_unit':
                            xbase_inv_OF = np.concatenate((xbase_g_inv_OF,xbase_e_inv,xbase_h_inv_OF,xbase_c_inv))
                            Dx_inv = np.diag(1/xbase_inv_OF)
                            Dx = np.diag(xbase_inv_OF)
                            Dy_inv = np.diag(np.concatenate((ubase,slack_base,1/xbase_inv_OF)))
                            Dy = np.diag(np.concatenate((1/ubase,1/slack_base,xbase_inv_OF)))
                        else:
                            Dx_inv = np.ones(len(xbase_inv_OF))
                            Dx = np.ones(len(xbase_inv_OF))
                            Dy_inv = np.ones(len(ubase)+len(slack_base)+len(xbase_inv_OF))
                            Dy = np.ones(len(ubase)+len(slack_base)+len(xbase_inv_OF))
                        parameters['Dx_'+key] = Dx
                        parameters['Dx_inv_'+key] = Dx_inv
                        parameters['Dy_'+key] = Dy
                        parameters['Dy_inv_'+key] = Dy_inv
    if save_data:
        if s > 0:
            path_to_data = os.path.join(dir_path,'Optimization_data','MES3N_streets')
        else:
            path_to_data = os.path.join(dir_path,'Optimization_data','MES3N_line')
        with open(os.path.join(path_to_data,'y_res_'+scale_label+'_{}_{}_{}.pkl'.format(n,m,s)), "wb") as y_file:
            pickle.dump(y_res,y_file)
        with open(os.path.join(path_to_data,'opt_res_'+scale_label+'_{}_{}_{}.pkl'.format(n,m,s)), "wb") as res_file:
            pickle.dump(result,res_file)
        with open(os.path.join(path_to_data,'errLF_res_'+scale_label+'_{}_{}_{}.pkl'.format(n,m,s)), "wb") as errLF_file:
            pickle.dump(err_LF_res,errLF_file)
        with open(os.path.join(path_to_data,'f_res_'+scale_label+'_{}_{}_{}.pkl'.format(n,m,s)), "wb") as f_file:
            pickle.dump(f_vec_res,f_file)
    elif save_parameters:
        if s > 0:
            path_to_data = os.path.join(dir_path,'Optimization_data','MES3N_streets')
        else:
            path_to_data = os.path.join(dir_path,'Optimization_data','MES3N_line')
        with open(os.path.join(path_to_data,'parameters_'+scale_label+'_{}_{}_{}.pkl'.format(n,m,s)), "wb") as params_file:
            pickle.dump(parameters,params_file)

    if save_figs:
        fig_legend = plt.figure('legend_mes_OF')
        ax_legend = fig_legend.gca()
        ax_legend.axis('off')
        fig_legend.patch.set_visible(False)
        ax_legend.legend(handles=legend_handles,loc='center')

    print('\nError LF: {} after {} iters.'.format(err_vec_LF[-1],iters_LF))
    print('P1 LF = {}MW'.format(P1/MW))
    if len(x_sol_LF) < 25:
        print('p gas = {} bar'.format(p_g_vec_LF/bar))
        print('q = {} kg/s'.format(q_vec_LF))
        print('q nodal inj = {} kg/s'.format(q_inj_LF))
        print('delta = {}'.format(delta_vec_LF))
        print('|V| = {} kV'.format(V_mag_vec_LF/kV))
        print('P edge = {} MW'.format(P_edge_LF/MW))
        print('Q edge = {} MW'.format(Q_edge_LF/MW))
        print('S nodal inj = {} MW'.format(S_inj_LF/MW))
        water = heat_net_LF.links[0].link_params.get('carrier')
        rho_w = water.rhon
        grav_const = water.g
        print('p heat = {} m'.format(p_h_vec_LF/(rho_w*grav_const)))
        print('m = {} kg/s'.format(m_vec_LF))
        print('Ts = {} C'.format(Ts_vec_LF))
        print('Tr = {} C'.format(Tr_vec_LF))
        print('m hl = {} kg/s'.format(m_hl_vec_LF))
        print('Ts hl = {}'.format(Ts_hl_vec_LF))
        print('Tr hl = {}'.format(Tr_hl_vec_LF))
        print('phi hl = {}'.format(phi_hl_vec_LF))
        print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net_LF.get_half_links()]))
        print('q c = {}'.format(qc_vec_LF))
        print('P c = {} MW'.format(np.array([Pc_vec_LF])/MW))
        print('Q c = {} MW'.format(np.array([Qc_vec_LF])/MW))
        print('m c = {}'.format(mc_vec_LF))
        print('phi c = {} MW'.format(np.array([phic_vec_LF])/MW))
        print('Ts c = {} C'.format(Tsc_vec_LF))
        print('Tr c = {} C'.format(Trc_vec_LF))

    if save_tables:
        cols = 4+3*len(methods) # number of columns in table
        if save_CPU_times:
            cols_cpu = cols + len(methods)
        if s > 0:
            path_to_tables = os.path.join(dir_path,'Tables','MES3N_streets')
        else:
            path_to_tables = os.path.join(dir_path,'Tables','MES3N_line')
        for bound in bounds:
            for derivative in an_ders:
                if derivative:
                    der_label = 'an'
                else:
                    der_label = 'num'
                with open(os.path.join(path_to_tables,'mes_optimizer_info_forms_'+der_label+'_'+bound+'_'+scale_label+'_{}_{}_{}_noCPU.txt'.format(n,m,s)), "w") as table, open(os.path.join(path_to_tables,'mes_optimizer_info_forms_'+der_label+'_'+bound+'_'+scale_label+'_{}_{}_{}_noCPU_LFerr.txt'.format(n,m,s)), "w") as table_LFerr, open(os.path.join(path_to_tables,'mes_optimizer_info_forms_'+der_label+'_'+bound+'_'+scale_label+'_{}_{}_{}_CPU_{}runs.txt'.format(n,m,s,number_of_runs)), "w") as table_CPU:
                    for link_limit in link_limits:
                        new_lim = True
                        for ind_gas, form_gas in enumerate(forms_gas):
                            new_gas = True
                            if form_gas == 'nodal':
                                xg_LF_sol = xg_LF[len(nlsys_LF.ind_xg_q):]
                                if scale_var != None:
                                    xbase_g_inv_OF = xbase_g_inv[len(nlsys_LF.ind_xg_q):]
                            else:
                                xg_LF_sol = xg_LF.copy()
                                if scale_var != None:
                                    xbase_g_inv_OF = xbase_g_inv.copy()
                            for ind_heat,form_heat in enumerate(forms_heat):
                                new_heat = True
                                if form_heat == 'standard':
                                    heat_label = form_heat
                                    xh_LF_sol = np.concatenate((xh_LF[:len(nlsys_LF.unknown_m_links)],xh_LF[len(nlsys_LF.unknown_m_links)+len(nlsys_LF.unknown_m_halflinks):]))
                                    if scale_var != None:
                                        xbase_h_inv_OF = np.concatenate((xbase_h_inv[:len(nlsys_LF.unknown_m_links)],xbase_h_inv[len(nlsys_LF.unknown_m_links)+len(nlsys_LF.unknown_m_halflinks):]))
                                else:
                                    heat_label = 'term. link'
                                    xh_LF_sol = xh_LF.copy()
                                    if scale_var != None:
                                        xbase_h_inv_OF = xbase_h_inv.copy()
                                y_LF_sol = np.concatenate((u_LF_sol,xG_LF_sol,xg_LF_sol,xe_LF_sol,xh_LF_sol,xc_LF_sol))
                                if link_limit:
                                    limit_label = 'yes'
                                else:
                                    limit_label = 'no'
                                for ind_app,approach in enumerate(approaches):
                                    if approach == 'eq_constr':
                                        approach_label = 'I'
                                    elif approach == 'direct':
                                        approach_label = 'II.A'
                                    else:
                                        approach_label = 'II.B'
                                    if new_lim:
                                        table.write(r'\multirow{'+str(len(forms_gas)*len(forms_heat)*len(approaches))+r'}{*}{'+limit_label+r'} & ')
                                        table_LFerr.write(r'\multirow{'+str(len(forms_gas)*len(forms_heat)*len(approaches))+r'}{*}{'+limit_label+r'} & ')
                                        if save_CPU_times:
                                            table_CPU.write(r'\multirow{'+str(len(forms_gas)*len(forms_heat)*len(approaches))+r'}{*}{'+limit_label+r'} & ')
                                        new_lim = False
                                    else:
                                        table.write(r' & ')
                                        table_LFerr.write(r' & ')
                                        if save_CPU_times:
                                            table_CPU.write(r' & ')
                                    if new_gas:
                                        table.write(r'\multirow{'+str(len(forms_heat)*len(approaches))+r'}{*}{'+form_gas+r'} & ')
                                        table_LFerr.write(r'\multirow{'+str(len(forms_heat)*len(approaches))+r'}{*}{'+form_gas+r'} & ')
                                        if save_CPU_times:
                                            table_CPU.write(r'\multirow{'+str(len(forms_heat)*len(approaches))+r'}{*}{'+form_gas+r'} & ')
                                        new_gas = False
                                    else:
                                        table.write(r' & ')
                                        table_LFerr.write(r' & ')
                                        if save_CPU_times:
                                            table_CPU.write(r' & ')
                                    if new_heat:
                                        table.write(r'\multirow{'+str(len(approaches))+r'}{*}{'+heat_label+r'} & ')
                                        table_LFerr.write(r'\multirow{'+str(len(approaches))+r'}{*}{'+heat_label+r'} & ')
                                        if save_CPU_times:
                                            table_CPU.write(r'\multirow{'+str(len(approaches))+r'}{*}{'+heat_label+r'} & ')
                                        new_heat = False
                                    else:
                                        table.write(r' & ')
                                        table_LFerr.write(r' & ')
                                        if save_CPU_times:
                                            table_CPU.write(r' & ')
                                    tables_iters = ' '
                                    tables_fcals = ' '
                                    tables_err = ' '
                                    tables_err_LF = ' '
                                    if save_CPU_times:
                                        tables_cpu = ' '
                                    for method in methods:
                                        key = method+'_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound+'_'+approach
                                        res_method= result.get(key)
                                        y_method = y_res.get(key)
                                        err_LF_method = err_LF_res.get(key)
                                        if approach != 'eq_constr' and (scale_var == 'matrix' or scale_var == 'per_unit'): # y_opt for direct and adjoint approach is unscaled
                                            xbase_inv_OF = np.concatenate((xbase_g_inv_OF,xbase_e_inv,xbase_h_inv_OF,xbase_c_inv))
                                            Dy = np.diag(np.concatenate((1/ubase,1/slack_base,xbase_inv_OF)))
                                            y_method = Dy.dot(y_method)
                                        if res_method.success:
                                            tables_iters += r' {:d} & '.format(res_method.nit)
                                            tables_fcals += r' {:d} & '.format(res_method.nfev)
                                            tables_err +=' '+exp_tex(error(y_method,y_LF_sol))+' & '
                                            tables_err_LF += ' '+exp_tex(err_LF_method[-1])+' & '
                                            if save_CPU_times:
                                                tables_cpu += ' '+exp_tex(res_method.execution_time)+' & '
                                        else:
                                            tables_iters += r' & '
                                            tables_fcals += r' & '
                                            tables_err += r' & '
                                            tables_err_LF += r' & '
                                            if save_CPU_times:
                                                tables_cpu += r' & '
                                    table.write(r'{} & '.format(approach_label)+tables_iters+tables_fcals+tables_err)
                                    table_LFerr.write(r'{} & '.format(approach_label)+tables_iters+tables_fcals+tables_err_LF)
                                    if save_CPU_times:
                                        table_CPU.write(r'{} & '.format(approach_label)+tables_iters+tables_fcals+tables_cpu+tables_err)
                                table.write(r'\cline{3-'+str(cols)+r'} ')
                                table_LFerr.write(r'\cline{3-'+str(cols)+r'} ')
                                if save_CPU_times:
                                    table_CPU.write(r'\cline{3-'+str(cols_cpu)+r'} ')
                            table.write(r'\cline{2-'+str(cols)+r'} ')
                            table_LFerr.write(r'\cline{2-'+str(cols)+r'} ')
                            if save_CPU_times:
                                table_CPU.write(r'\cline{2-'+str(cols_cpu)+r'} ')
                        table.write(r'\hline ')
                        table_LFerr.write(r'\hline ')
                        if save_CPU_times:
                            table_CPU.write(r'\hline ')

    for derivative in an_ders:
        if derivative:
            der_label = 'an'
        else:
            der_label = 'num'
        for bound in bounds:
            fig_en_bars = plt.figure('mes_energy_inputs_'+der_label+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
            ax_en_bar = fig_en_bars.add_subplot(1,1,1)
            fig_cost_bars = plt.figure('mes_cost_inputs_'+der_label+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
            ax_cost_bar = fig_cost_bars.add_subplot(1,1,1)
            fig_en_bars_nogas = plt.figure('mes_energy_inputs_nogas_'+der_label+'_'+bound+'_'+scale_label+'_{}_{}_{}'.format(n,m,s))
            ax_en_bar_nogas = fig_en_bars_nogas.add_subplot(1,1,1)
            scale_bars = [20,1.5e9,1]
            x_en_bar = 0
            width_bar = .35
            xticks_en_bar = list()
            xticks_en_bar_labels = list()
            x_approach = -width_bar
            x_heat = -3*width_bar
            x_gas = -3*width_bar
            x_link = -3*width_bar
            y_pos = -1
            dy_pos = 0.2
            plot_LF_sol = True
            for link_limit in link_limits:
                if link_limit:
                    limit_label = 'limits'
                else:
                    limit_label = 'no limits'
                for form_gas in forms_gas:
                    if form_gas == 'nodal':
                        xg_LF_sol = xg_LF[len(nlsys_LF.ind_xg_q):]
                        if scale_var != None:
                            xbase_g_inv_OF = xbase_g_inv[len(nlsys_LF.ind_xg_q):]
                    else:
                        xg_LF_sol = xg_LF.copy()
                        if scale_var != None:
                            xbase_g_inv_OF = xbase_g_inv.copy()
                    for form_heat in forms_heat:
                        if form_heat == 'standard':
                            heat_label = form_heat
                            xh_LF_sol = np.concatenate((xh_LF[:len(nlsys_LF.unknown_m_links)],xh_LF[len(nlsys_LF.unknown_m_links)+len(nlsys_LF.unknown_m_halflinks):]))
                            if scale_var != None:
                                xbase_h_inv_OF = np.concatenate((xbase_h_inv[:len(nlsys_LF.unknown_m_links)],xbase_h_inv[len(nlsys_LF.unknown_m_links)+len(nlsys_LF.unknown_m_halflinks):]))
                        else:
                            heat_label = 'term. link'
                            xh_LF_sol = xh_LF.copy()
                            if scale_var != None:
                                xbase_h_inv_OF = xbase_h_inv.copy()
                        y_LF_sol = np.concatenate((u_LF_sol,xG_LF_sol,xg_LF_sol,xe_LF_sol,xh_LF_sol,xc_LF_sol))
                        y_ind = y_inds.get(form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound)
                        for approach in approaches:
                            if approach == 'eq_constr':
                                approach_label = 'eq. constr.'
                            else:
                                approach_label = approach
                            print('\nLink limits: {}, form gas: {}, form heat: {}, an der: {}, bounds: {}, approach: {}.'.format(link_limit,form_gas,form_heat,derivative,bound,approach))
                            for method in methods:
                                key = method+'_'+form_gas+'_'+form_heat+'_'+str(link_limit)+'_'+der_label+'_'+bound+'_'+approach
                                label = method+', '+form_gas+', '+heat_label+', '+limit_label+', '+approach_label
                                res_method= result.get(key)
                                y_method = y_res.get(key)
                                print('For {}: success: {}, iters: {}, error: {:.4e}, exec time: {:.2f}, message: {}'.format(method,res_method.success,res_method.nit,error(y_method,y_LF_sol),res_method.execution_time,res_method.message))
                                if scale_var == 'matrix' or scale_var == 'per_unit':
                                    xbase_inv_OF = np.concatenate((xbase_g_inv_OF,xbase_e_inv,xbase_h_inv_OF,xbase_c_inv))
                                    Dy_inv = np.diag(np.concatenate((ubase,slack_base,1/xbase_inv_OF)))
                                    if approach == 'eq_constr': # y_opt for direct and adjoint approach is unscaled
                                        y_method = Dy_inv.dot(y_method) # y_method is now unscaled
                                    y_LF_sol = Dy_inv.dot(y_LF_sol)
                                if plot_LF_sol:
                                    e_opt_sum_LF = 0
                                    e_opt_sum_nogas_LF = 0
                                    price_sum_LF = 0
                                    for ind, e_opt in enumerate(y_LF_sol[np.array(y_ind)]):
                                        if ind == 0:
                                            e_opt += q3 # only the gas input dedicated to the couplings
                                        price = a[ind]+b[ind]*np.abs(e_opt)+c[ind]*e_opt**2
                                        e_opt /= MW
                                        if ind == 0:
                                            e_opt *= MES3N_line.GHV
                                        else:
                                            ax_en_bar_nogas.bar(-2*width_bar,np.abs(e_opt),width_bar,bottom=e_opt_sum_nogas_LF,color=colors_energy[ind],linewidth=.1,edgecolor='k')
                                            e_opt_sum_nogas_LF += np.abs(e_opt)
                                        ax_en_bar.bar(-2*width_bar,np.abs(e_opt),width_bar,bottom=e_opt_sum_LF,color=colors_energy[ind],linewidth=.1,edgecolor='k')
                                        e_opt_sum_LF += np.abs(e_opt)
                                        ax_cost_bar.bar(-2*width_bar,price,width_bar,bottom=price_sum_LF,color=colors_energy[ind],linewidth=.1,edgecolor='k')
                                        price_sum_LF += price
                                        xticks_en_bar_labels.append('LF')
                                        xticks_en_bar.append(-2*width_bar)
                                    plot_LF_sol = False
                                if res_method.success:
                                    e_opt_sum = 0
                                    e_opt_sum_nogas = 0
                                    price_sum = 0
                                    for ind, e_opt in enumerate(y_method[np.array(y_ind)]):
                                        if ind == 0:
                                            e_opt += q3 # only the gas input dedicated to the couplings
                                        price = a[ind]+b[ind]*np.abs(e_opt)+c[ind]*e_opt**2
                                        e_opt /= MW
                                        if ind == 0:
                                            e_opt *= MES3N_line.GHV
                                        else:
                                            ax_en_bar_nogas.bar(x_en_bar,np.abs(e_opt),width_bar,bottom=e_opt_sum_nogas,color=colors_energy[ind],linewidth=.1,edgecolor='k')
                                            e_opt_sum_nogas += np.abs(e_opt)
                                        ax_en_bar.bar(x_en_bar,np.abs(e_opt),width_bar,bottom=e_opt_sum,color=colors_energy[ind],linewidth=.1,edgecolor='k')
                                        e_opt_sum += np.abs(e_opt)
                                        ax_cost_bar.bar(x_en_bar,price,width_bar,bottom=price_sum,color=colors_energy[ind],linewidth=.1,edgecolor='k')
                                        price_sum += price
                                    xticks_en_bar_labels.append(method)
                                    xticks_en_bar.append(x_en_bar)
                                    x_en_bar += width_bar
                                if method == methods[-1]:
                                    x_en_bar += width_bar
                                if scale_var != None:
                                    Dy = np.diag(np.concatenate((1/ubase,1/slack_base,xbase_inv_OF)))
                                    y_method = Dy.dot(y_method)
                                    y_LF_sol = Dy.dot(y_LF_sol)
                            for ind, ax in enumerate([ax_en_bar,ax_cost_bar,ax_en_bar_nogas]):
                                ax.text(x_approach+(x_en_bar-x_approach)/2,y_pos*scale_bars[ind],approach_label,ha='center')
                            x_approach = x_en_bar-2*width_bar
                            # For what I should do: https://stackoverflow.com/questions/19184484/how-to-add-group-labels-for-bar-charts-in-matplotlib
                            # line = plt.Line2D([x_approach, x_approach], [y_pos + .1, y_pos], color='black')
                            # line.set_clip_on(False)
                            # ax_en_bar_nogas.add_line(line)
                        for ind, ax in enumerate([ax_en_bar,ax_cost_bar,ax_en_bar_nogas]):
                            ax.text(x_heat+(x_en_bar-x_heat)/2,(y_pos-dy_pos)*scale_bars[ind],heat_label,ha='center')
                        x_heat = x_en_bar-2*width_bar
                    for ind, ax in enumerate([ax_en_bar,ax_cost_bar,ax_en_bar_nogas]):
                        ax.text(x_gas+(x_en_bar-x_gas)/2,(y_pos-2*dy_pos)*scale_bars[ind],form_gas,ha='center')
                    x_gas = x_en_bar-width_bar
                for ind, ax in enumerate([ax_en_bar,ax_cost_bar,ax_en_bar_nogas]):
                    ax.text(x_link+(x_en_bar-x_link)/2,(y_pos-3*dy_pos)*scale_bars[ind],limit_label,ha='center')
                x_link = x_en_bar-width_bar

            ax_en_bar.set_ylabel('Energy [MW]')
            ax_en_bar.legend(handles=legend_handles_energy)
            ax_en_bar.plot([xticks_en_bar[0]-width_bar/2,xticks_en_bar[-1]+width_bar/2],[e_opt_sum_LF,e_opt_sum_LF],':k')
            ax_cost_bar.set_ylabel('Cost [euros/h]')
            ax_cost_bar.legend(handles=legend_handles_energy)
            ax_cost_bar.plot([xticks_en_bar[0]-width_bar/2,xticks_en_bar[-1]+width_bar/2],[price_sum_LF,price_sum_LF],':k')
            ax_en_bar_nogas.set_ylabel('Energy [MW]')
            ax_en_bar_nogas.legend(handles=legend_handles_energy[1:])
            ax_en_bar_nogas.plot([xticks_en_bar[0]-width_bar/2,xticks_en_bar[-1]+width_bar/2],[e_opt_sum_nogas_LF,e_opt_sum_nogas_LF],':k')
            for fig_num in [fig_en_bars.number,fig_cost_bars.number,fig_en_bars_nogas.number]:
                plt.figure(fig_num)
                plt.xticks(xticks_en_bar,xticks_en_bar_labels, rotation='vertical')
                plt.margins(0.2) # Pad margins so that markers don't get clipped by the axes
                plt.subplots_adjust(bottom=0.32)# Tweak spacing to prevent clipping of tick-labels

    if save_figs:
        if s > 0:
            path_to_fig = os.path.join(dir_path,'Figures','MES3N_streets')
        else:
            path_to_fig = os.path.join(dir_path,'Figures','MES3N_line')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def table_LF_results_base(path_to_data):
    """ Results LF of base case, using matrix scaling.

    NB: Should use the same initial guess en BCs as compare_forms(). So check to see if they're still the same.
    """
    # solver info
    max_iters_lf = 20
    tol = 1e-6

    # scaling (is used in solving LF by default)
    pgbase = 10*bar
    qbase = 1.
    deltabase = 1.
    Vbase = 50*kV
    Sbase = 1*MW
    phbase = 10*bar
    mbase = 1
    Tbase = 100
    phibase = 1*MW
    pbase = pgbase
    Ebase = 1*MW
    scale_var_params = {'pgbase':pgbase,'qbase':qbase,'deltabase':deltabase,'Vbase':Vbase,'Sbase':Sbase,'phbase':phbase,'pbase':pbase,'mbase':mbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Ebase}
    # Network parameters and boundary conditions (of LF)
    p1g = 50*bar
    q3 = 1
    Dg=10*cm
    E=.98
    link_type_g='pipe_high_pres_weymouth'
    V1 = 50*kV #[V]
    delta1 = 0.#[rad]
    De=1*cm
    link_type_e='pi_line'
    Ts1=100.
    p1h=9*bar
    To3=50.
    link_type_h='standard_pipe_low_pres_pole'
    To2=84.346642#84.82293161333068#100.
    phi2=-1.00175442*MW#-1.0469065288680548*MW#-1*MW
    phi3=1.5*MW
    Dh=15*cm
    lam=0.2
    V2 = 49.9854119*kV#49.98856105*kV #[V]
    P2 = -.4*MW#-0.5999727456238263*MW #[W]
    P3 = 1.5*MW #[W]
    Q3 = 1.5*MW #[Var]
    L12=4*km
    L23=5*km
    coupling_type='EH'
    coupling_point=1
    # initial conditions
    p_perc_high = .99
    p_perc_low = .98
    T_perc_high = 1.
    T_perc_low = 1.
    q_init=1
    m_init=10
    Tr_sink_init=50
    Pc_init=P3+P2#1.5*MW#1*MW
    Qc_init=Q3#1*MW
    phic_init=phi3#1.5*MW#1*MW
    qc_init=(P3+phi3)/MES3N_line.GHV#.1#1
    print('I.C., Pc = {}MW, Qc = {}MW, phic = {}MW, qc = {}kg/s'.format(Pc_init/MW,Qc_init/MW,phic_init/MW,qc_init))
    # steady-state LF solution
    n = 0
    m = 0
    s = 0
    with HiddenPrints():
        formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
        hydr_eq_gas = 'fb'
        gas_net,elec_net,heat_net,het_net = create_network(n=n,m=m,s=s,p1g=p1g,q3=q3,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi2=phi2,phi3=phi3,L12=L12,L23=L23,Dg=Dg,E=E,link_type_g=link_type_g,hydr_eq_gas=hydr_eq_gas,De=De,link_type_e=link_type_e,Dh=Dh,lam=lam,link_type_h=link_type_h,coupling_type=coupling_type,coupling_point=coupling_point)
        het_net.initialize()
        nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var='matrix',scale_var_params=scale_var_params)
        x_init = set_x_LF_init(nlsys,n=n,m=m,s=s,q_init=q_init,m_init=m_init,Tr_sink_init=Tr_sink_init,p_perc_high=p_perc_high,p_perc_low=p_perc_low,T_perc_high=T_perc_high,T_perc_low=T_perc_low)
        het_net.update(x_init,formulation=formulation)
        x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iters_lf,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params=scale_var_params)
        x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)
    print('x init = {}'.format(x_init))
    print('x sol = {}'.format(x_sol))

    nodes = ['1','2','3']
    links = ['1--2','2--3']
    with open(os.path.join(path_to_data,'gas_data_base_EH_node1_thesis.txt'), "w") as table:
        for ind in range(len(nodes)):
            node_data = ' ' + nodes[ind] + r' '
            if ind < len(links):
                link_data = ' ' + links[ind] + r' '
            else:
                link_data = ' '
            if gas_net.nodes[ind] in list(xg_entries):
                node_data += r' & {:.3f} '.format(gas_net.nodes[ind].get_p()/bar)
            else:
                node_data += r' & \textbf{'+'{:.3f}'.format(gas_net.nodes[ind].get_p()/bar)+'} '
            if gas_net.nodes[ind].half_links[0].bc_type == 1: # q known
                node_data += r' & \textbf{'+'{:.2f}'.format(gas_net.nodes[ind].half_links[0].get_q())+'} '
            else:
                node_data += r' & {:.2f} '.format(gas_net.nodes[ind].half_links[0].get_q())
            if ind < len(links):
                link_data += r' & {:.2f}'.format(gas_net.links[ind].get_q())
            else:
                link_data += r' & '
            table.write(node_data)
            table.write(r' & ')
            table.write(link_data)
            table.write(r' & ')

    P_loss_total = 0
    Q_loss_total = 0
    with open(os.path.join(path_to_data,'elec_data_base_EH_node1_thesis.txt'), "w") as table:
        for ind in range(len(nodes)):
            node_data = ' ' + nodes[ind] + r' '
            if ind < len(links):
                link_data = ' ' + links[ind] + r' '
            else:
                link_data = ' '
            if elec_net.nodes[ind] in list(unknown_V_nodes):
                node_data += r' & {:.3f} '.format(elec_net.nodes[ind].get_V()/kV)
            else:
                node_data += r' & \textbf{'+'{:.3f}'.format(elec_net.nodes[ind].get_V()/kV)+'} '
            if elec_net.nodes[ind] in list(unknown_delta_nodes):
                node_data += r' & {:.3f} '.format(elec_net.nodes[ind].get_delta())
            else:
                node_data += r' & \textbf{'+'{:.0f}'.format(elec_net.nodes[ind].get_delta())+'} '
            if len(elec_net.nodes[ind].half_links):
                hl = elec_net.nodes[ind].half_links[0]
                if hl.bc_type in [1,2]: # P known
                    node_data += r' & \textbf{'+'{:.1f} '.format(hl.get_P()/MW)+'} '
                else:
                    node_data += r' & {:.1f} '.format(hl.get_P()/MW)
                if hl.bc_type in [1,3]: # Q known
                    node_data += r'\textbf{'+'{:+.1f}$\iu$  '.format(hl.get_Q()/MW)+'} '
                else:
                    node_data += r'{:+.1f}$\iu$  '.format(hl.get_Q()/MW)
            else:
                node_data += r' & - '
            if ind < len(links):
                P_loss = elec_net.links[ind].complex_power_loss().real/MW
                Q_loss = elec_net.links[ind].complex_power_loss().imag/MW
                P_loss_total += P_loss
                Q_loss_total += Q_loss
                link_data += r' & {:.3f} {:+.3f}$\iu '.format(P_loss, Q_loss)
            else:
                link_data += r' & '
            table.write(node_data)
            table.write(r' & ')
            table.write(link_data)
            table.write(r' & ')
        table.write(r' \hline ')
        table.write(r' & & & & Total & {:.3f}  {:+.3f}$\iu '.format(P_loss_total,Q_loss_total))
        table.write(r' & ')

        with open(os.path.join(path_to_data,'heat_data_base_EH_node1_hydraulic_thesis.txt'), "w") as table:
            for ind in range(len(nodes)):
                node_data = ' ' + nodes[ind] + r' '
                if ind < len(links):
                    link_data = ' ' + links[ind] + r' '
                else:
                    link_data = ' '
                if heat_net.nodes[ind] in list(unknown_p_nodes):
                    node_data += r' & {:.3f} '.format(heat_net.nodes[ind].get_p()/bar)
                else:
                    node_data += r' & \textbf{'+'{:.3f}'.format(heat_net.nodes[ind].get_p()/bar)+'} '
                if len(heat_net.nodes[ind].half_links):
                    if formulation.get('heat') == 'standard' or heat_net.nodes[ind].half_links[0] in unknown_m_halflinks:
                        node_data += r' & {:.3f} '.format(heat_net.nodes[ind].half_links[0].get_m())
                    else:
                        node_data += r' & \textbf{'+'{:.3f} '.format(heat_net.nodes[ind].half_links[0].get_m())+'} '
                else:
                    node_data += r' & - '
                if ind < len(links):
                    link_data += r' & {:.3f}'.format(heat_net.links[ind].get_m())
                else:
                    link_data += r' & '
                table.write(node_data)
                table.write(r' & ')
                table.write(link_data)
                table.write(r' & ')

        phi_loss_total = 0
        with open(os.path.join(path_to_data,'heat_data_base_EH_node1_thermal_thesis.txt'), "w") as table:
            for ind in range(len(nodes)):
                node_data = ' ' + nodes[ind] + r' '
                if ind < len(links):
                    link_data = ' ' + links[ind] + r' '
                else:
                    link_data = ' '
                if heat_net.nodes[ind] in list(unknown_Ts_nodes):
                    node_data += r' & {:.3f} '.format(heat_net.nodes[ind].get_Ts())
                else:
                    node_data += r' & \textbf{'+'{:.0f}'.format(heat_net.nodes[ind].get_Ts())+'} '
                if heat_net.nodes[ind] in list(unknown_Tr_nodes):
                    node_data += r' & {:.3f} '.format(heat_net.nodes[ind].get_Tr())
                else:
                    node_data += r' & \textbf{'+'{:.0f}'.format(heat_net.nodes[ind].get_Tr())+'} '
                if len(heat_net.nodes[ind].half_links):
                    node_data += r' & \textbf{'+'{:.1f} '.format(heat_net.nodes[ind].half_links[0].get_dphi()/MW)+'} '
                else:
                    node_data += r' & - '
                if ind < len(links):
                    phi_loss = heat_net.links[ind].heat_loss_supply()/MW + heat_net.links[ind].heat_loss_return()/MW
                    phi_loss_total += phi_loss
                    link_data += r' & {:.3f}'.format(phi_loss)
                else:
                    link_data += r' & '
                table.write(node_data)
                table.write(r' & ')
                table.write(link_data)
                table.write(r' & ')
            table.write(r' \hline ')
            table.write(r' & & & & Total & {:.3f} '.format(phi_loss_total))
            table.write(r' & ')

        unit = [r'\acrshort{eh} at 1']
        with open(os.path.join(path_to_data,'coupling_data_base_EH_node1_thesis.txt'), "w") as table:
            for ind in range(len(unit)):
                link_data = ' ' + unit[ind] + r' '
                # gas data
                if gas_net.links[ind-len(unit)] in unknown_qc_links:
                    link_data += r' & {:.3f}'.format(gas_net.links[ind-len(unit)].get_q())
                else:
                    link_data += r' & \textbf{'+'{:.3f}'.format(gas_net.links[ind-len(unit)].get_q())+'} '
                # elec data
                if elec_net.links[ind-len(unit)] in unknown_Pc_links:
                    link_data += r' & {:.3f}'.format(elec_net.links[ind-len(unit)].get_Pstart()/MW)
                else:
                    link_data += r' & \textbf{'+'{:.3f}'.format(elec_net.links[ind-len(unit)].get_Pstart()/MW)+'} '
                if elec_net.links[ind-len(unit)] in unknown_Qc_links:
                    link_data += r' & {:.3f} '.format(elec_net.links[ind-len(unit)].get_Qstart()/MW)
                else:
                    link_data += r' & \textbf{'+'{:.3f}'.format(elec_net.links[ind-len(unit)].get_Qstart()/MW)+'} '
                # heat data
                if heat_net.links[ind-len(unit)] in unknown_mc_links:
                    link_data += r' & {:.3f}'.format(heat_net.links[ind-len(unit)].get_m())
                else:
                    link_data += r' & \textbf{'+'{:.3f}'.format(heat_net.links[ind-len(unit)].get_m())+'} '
                if heat_net.links[ind-len(unit)] in unknown_dphi_links:
                    link_data += r' & {:.3f}'.format(-heat_net.links[ind-len(unit)].get_dphistart()/MW)
                else:
                    link_data += r' & \textbf{'+'{:.3f}'.format(-heat_net.links[ind-len(unit)].get_dphistart()/MW)+'} '
                if heat_net.links[ind-len(unit)] in unknown_Ts_links:
                    link_data += r' & {:.3f}'.format(heat_net.links[ind-len(unit)].get_Tsstart())
                else:
                    link_data += r' & \textbf{'+'{:.3f}'.format(heat_net.links[ind-len(unit)].get_Tsstart())+'} '
                table.write(link_data)
                table.write(r' & ')

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','MES3N_line')
    table_LF_results_base(path_to_data)

    # base case (3 nodes)
    n=0
    m=0
    s=0
    runs = 1#5
    max_iter = 50
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_CPU_times=False,number_of_runs=runs,n=n,m=m,s=s,max_iter=max_iter) # unscaled
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_data=False,save_CPU_times=False,save_parameters=False,number_of_runs=runs,scale_var='matrix',n=n,m=m,s=s,max_iter=max_iter)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_data=False,save_parameters=False,save_CPU_times=False,number_of_runs=runs,scale_var='per_unit',n=n,m=m,s=s,max_iter=max_iter)

    # medium case (30 nodes)
    n=5
    m=2
    s=3
    runs = 1#5
    max_iter = 50
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_data=False,save_CPU_times=False,number_of_runs=runs,scale_var='matrix',n=n,m=m,s=s,max_iter=max_iter)
    # compare_forms(dir_path=dir_path,save_tables=False,save_figs=False,save_data=False,save_CPU_times=False,number_of_runs=runs,scale_var='per_unit',n=n,m=m,s=s,max_iter=max_iter)

    # medium / large case (163 nodes)
    n=10
    m=5
    s=10
    runs = 1#5
    max_iter = 50
    # print('Starting OF for n={}, m={}, s={}, using matrix scaling.'.format(n,m,s))
    # compare_forms(dir_path=dir_path,save_tables=True,save_figs=True,save_data=True,save_CPU_times=False,number_of_runs=runs,scale_var='matrix',n=n,m=m,s=s,max_iter=max_iter)
    # plt.close('all')
    # print('Finished OF for n={}, m={}, s={}, using matrix scaling.'.format(n,m,s))
    # print('Starting OF for n={}, m={}, s={}, using per unit scaling.'.format(n,m,s))
    # compare_forms(dir_path=dir_path,save_tables=True,save_figs=True,save_data=True,save_CPU_times=False,number_of_runs=runs,scale_var='per_unit',n=n,m=m,s=s,max_iter=max_iter)
    # print('Finished OF for n={}, m={}, s={}, using per unit scaling.'.format(n,m,s))
    # plt.close('all')

    #plt.show()
