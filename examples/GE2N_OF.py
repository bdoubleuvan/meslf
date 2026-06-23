"""
OF for MES consisting of 2 nodes per carrier (i.e. only one link in every single-carrier network), with multiple couplings.
"""
from examples import GE2N as MES
import warnings
import numpy as np
from meslf.utils.hide_print import HiddenPrints
from meslf.utils.constants import mbar, MW
from meslf.load_flow.system_of_equations import NonLinearSystemGas, NonLinearSystemElectrical, NonLinearSystemHeterogeneous
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.optimize as spo
import scipy.sparse as sps
import os
import sys
import time
import ipopt
import datetime

colors_method = {'trust-constr':'tab:blue','SLSQP':'tab:orange','ipopt':'tab:green'}
linestyles_approaches = {'eq_constr':'-','direct':':','adjoint':'-.'}
markers_forms = {'int. orig.':'.','int. reord.':'*','dd1':'d','dd2':'s'}
marker_size = 10
legend_handles = [Line2D([0], [0], color=colors_method.get('trust-constr'), label='trust-constr'),
    Line2D([0], [0], color=colors_method.get('SLSQP'), label='SLSQP'),
    Line2D([0], [0], color=colors_method.get('ipopt'), label='IPOPT'),
    Line2D([0], [0], color='w',marker=markers_forms.get('int. orig.'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='integrated, original'),
    Line2D([0], [0], color='w',marker=markers_forms.get('int. reord.'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='integrated, reordered'),
    Line2D([0], [0], color='w',marker=markers_forms.get('dd1'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='DD 1'),
    Line2D([0], [0], color='w',marker=markers_forms.get('dd2'), markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label='DD 2'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('eq_constr'), label='LF as eq. constr.'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('direct'), label='Direct approach'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('adjoint'), label='Adjoint approach')]

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

def xmes_from_yopf(y,nlsys=None):
    """Returns the variables of steady-state LF, from the variables y of OF."""
    xmes = y[-len(nlsys.x_entries):]
    return xmes

def xdd_from_yopf(y,len_xg=3, len_xe=4, len_xc=2):
    """Returns the variables of DD steady-state LF, from the variables y of OF."""
    len_u = len(y) - (len_xg+len_xe+len_xc)
    u = y[:len_u]
    xg = y[len_u:len_u+len_xg]
    xe = y[len_u+len_xg:len_u+len_xg+len_xe]
    xc = y[-len_xc:]
    return u, xg, xe, xc

def objective_function(y,y_ind,a=np.array([0,0,0]),b=np.array([15*MES.GHV,2,3])/MW,c=np.array([0,0.02,0.03])/MW,scale_var=None,scale_var_params=None,fb=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    E : np array
        Array with flows used in objective. Gas flows are assumed to be in kg/s, active powers in W, and heat power in W. Scaled when per unit scaling is used, unscaled otherwise.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [-q0, Pc0, Pc1] or [-q0, Pc, -P1,c]. Scaled when per unit scaling is used, unscaled otherwise.
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
    # print('y ind = {}\ny[y_ind]={}'.format(y_ind,y[np.array(y_ind)]))
    f = np.sum(a+b*np.sign(y[np.array(y_ind)])*y[np.array(y_ind)]+c*y[np.array(y_ind)]**2)
    if scale_var == 'matrix':
        f *= (1/fb)
    return f

def grad_objective(y,y_ind,a=np.array([0,0,0]),b=np.array([15*MES.GHV,2,3])/MW,c=np.array([0,0.02,0.03])/MW,scale_var=None,scale_var_params=None,fb=None,Dy_inv=None, Tx_inv=None,reorder=False):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [-q0, Pc0, Pc1] or [-q0, Pc, -P1,c]. Scaled when per unit scaling is used, unscaled otherwise.
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
    if reorder:
        len_u=len(y)-Tx_inv.shape[0]
        grad_x = sps.csr_matrix(df_dy[len_u:])
        df_dy[len_u:] = grad_x.dot(Tx_inv).todense()
    return df_dy

def hess_objective(y,y_ind,a=np.array([0,0,0]),b=np.array([15*MES.GHV,2,3])/MW,c=np.array([0,0.02,0.03])/MW,scale_var=None,scale_var_params=None,fb=None,Dy_inv=None, Tx_inv=None,reorder=False):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [-q0, Pc0, Pc1] or [-q0, Pc, -P1,c]. Scaled when per unit scaling is used, unscaled otherwise.
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
        hess = np.transpose(Dy_inv).dot(hess.dot(Dy_inv))
    if reorder:
        len_u=len(y)-Tx_inv.shape[0]
        hess_x = sps.csr_matrix(hess[len_u:,:][:,len_u:])
        hess_x = Tx_inv.transpose().dot(hess_x.dot(Tx_inv))
        hess[len_u:,:][:,len_u:] = hess_x.todense()
    if scale_var == 'matrix':
        hess = (1/fb)*hess
    return hess

def h(y, nlsys=None, Dh=None, TF=None,reorder=False):
    """Equality constraints h(x)=0 for the integrated system. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeterogeneous
        Nonlinear system of the multi-carrier network. Constains the networks, information about scaling, etc.
    Dh : sps matrix or None
        Scaling matrix for h. Only used for matrix scaling.
    TF : sps matrix or empty np array
        Transformation matrix for F, in case of reordering.

    Returns
    -------
    h : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    network_g = nlsys.nlsystemsg[0].gasnetwork
    network_e = nlsys.nlsystemse[0].elecnetwork
    network_mes = nlsys.hetnetwork

    xmes = xmes_from_yopf(y,nlsys=nlsys)

    # evaluate load flow equations
    F = nlsys.F(xmes) # Also updates network. Should set correct values on links and halflinks

    # no additional equations for integrated system, so
    h = F.copy()
    if nlsys.scale_var == 'matrix':
        h = Dh.dot(h)
    # reorder
    if reorder:
        h = TF.dot(h)
    return h

def h_dd(y, nlsysg=None, nlsyse=None, nlsysc=None, len_xg=3, len_xe=4, len_xc=2, Dh=None):
    """Equality constraints h(x)=0 for the integrated system. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeterogeneous
        Nonlinear system of the multi-carrier network. Constains the networks, information about scaling, etc.
    Dh : sps matrix or None
        Scaling matrix for h. Only used for matrix scaling.

    Returns
    -------
    h : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    u, xg, xe, xc = xdd_from_yopf(y,len_xg=len_xg, len_xe=len_xe, len_xc=len_xc)
    q0_c = xg[-1]
    q0 = u[0]
    P1_c,Q0_c,Q1_c = xe[1:]
    # evaluate load flow equations
    Fg = nlsysg.F(xg[:len(nlsysg.x_entries)])
    Gg = -q0_c+ np.array([MES.G_gas(nlsysg.gasnetwork,scale_var=nlsysg.scale_var,scale_var_params=nlsysg.scale_var_params,q0=q0)])
    Fe = nlsyse.F(xe[:len(nlsyse.x_entries)])
    Ge = np.array([P1_c,Q0_c,Q1_c])- np.array(MES.G_elec(nlsyse.elecnetwork,scale_var=nlsyse.scale_var,scale_var_params=nlsyse.scale_var_params))
    if len_xc==0:
        xc = u[2:]
    with HiddenPrints():
        Fc = nlsysc.F(xc)
    # print('Fg: {},Gg: {},Fe: {},Ge: {},Fc: {}'.format(Fg.shape,Gg.shape,Fe.shape,Ge.shape,Fc.shape))
    h = np.concatenate((Fg,Gg,Fe,Ge,Fc))
    if nlsysg.scale_var == 'matrix':
        h = Dh.dot(h)
    return h

def h_der(y, nlsys=None, Dy_inv=None, Dh=None, TF=None, Tx_inv=None,reorder=False):
    """First derivative of quality constraints h(x)=0. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeat
        Nonlinear system of the heat network. Constains the networks, information about scaling, etc.
    TF : sps matrix or empty np array
        Transformation matrix for F, in case of reordering.
    Tx_inv : sps matrix or empty np array
        Inverse of transformation matrix for x, in case of reordering.

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
    network_mes = nlsys.hetnetwork

    xmes = xmes_from_yopf(y,nlsys=nlsys)
    q0, V0 = y[:len_u]
    # update vector with voltage amplitudes and magnitudes, since nlsys.J() only updates the voltage amplitudes and angels that are in x
    nlsyse.V_vec_mag[0] = V0

    # Jacobian of LF equations
    J_lf = nlsys.J_dense(xmes)
    dh_dy = np.zeros((len_h,len(y)))
    dh_dy[:,len_u:] = J_lf #dF_dxF
    # Derivatives of F to u
    xe = xmes[len(nlsys.xg_entries):len(nlsys.xg_entries)+len(nlsys.xe_entries)]
    Ne = len(network_e.nodes)
    V0_ind = Ne # index of V0 within the full Jee
    Fe_ind = nlsyse.FP + [Ne+ind for ind in nlsyse.FQ]
    Je_full = nlsyse.J_dense(xe,return_full=True)
    dh_dy[len(nlsys.Fg_entries):len(nlsys.Fg_entries)+len(Fe_ind),0] = Je_full[Fe_ind,V0_ind].ravel() # dFe_dV2
    dh_dy[0,0] = -1 #dFq0_dq0

    if nlsys.scale_var == 'matrix':
        dh_dy = Dh.dot(dh_dy.dot(Dy_inv))

    # reorder
    if reorder:
        dh_dx = sps.csr_matrix(dh_dy[:,len_u:])
        dh_dy[:,len_u:] = dh_dx.dot(Tx_inv).todense()
        dh_dy = TF.dot(dh_dy)
    return dh_dy

def h_der_dd(y, nlsysg=None, nlsyse=None, nlsysc=None, len_xg=3, len_xe=4, len_xc=2, Dy_inv=None, Dh=None):
    """Equality constraints h(x)=0 for the integrated system. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeterogeneous
        Nonlinear system of the multi-carrier network. Constains the networks, information about scaling, etc.
    Dh : sps matrix or None
        Scaling matrix for h. Only used for matrix scaling.

    Returns
    -------
    h : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    len_h = Dh.shape[0]
    len_u = len(y) - len_h # length of original u, without xc which might be part of u
    dh_dy = np.zeros((len_h,len(y)))

    u, xg, xe, xc = xdd_from_yopf(y,len_xg=len_xg, len_xe=len_xe, len_xc=len_xc)
    len_xc_u = len(u)-len_u # amount of xc in u

    # Jacobians of extended LF equations, gas part
    Ng = len(nlsysg.gasnetwork.nodes)
    Eg = len(nlsysg.gasnetwork.links)
    Fg_ind = nlsysg.ind_Fn + [Ng+ind for ind in nlsysg.ind_Fl]
    Gg_ind = [0]
    if nlsysg.gas_formulation=='nodal':
        xlfg_ind = nlsysg.ind_p
    else:
        xlfg_ind = nlsysg.ind_q + [Eg+ind for ind in nlsysg.ind_p]
    # len_slack_g = len(xg) - len(xlfg_ind)
    dhg_dxg = np.zeros((len_xg,len_xg))
    Jgg_full = nlsysg.J_dense(xg[:len(nlsysg.x_entries)],return_full=True)
    dhg_dxg[:len(Fg_ind),:][:,:len(xlfg_ind)] = Jgg_full[Fg_ind,:][:,xlfg_ind] #Jgg
    dhg_dxg[len(Fg_ind):,:][:,:len(xlfg_ind)] = Jgg_full[Gg_ind,:][:,xlfg_ind] #dGg_dxlfg
    dhg_dxg[len(Fg_ind):,:][:,len(xlfg_ind):] = -1 #dGg_dxGg
    dhg_dxc = np.zeros((len_xg,max(len_xc,len_xc_u)))
    dhg_dxc[0,0] = -1 # der. of cons. of mass in node 1 to (-qc1)

    # Jacobians of extended LF equations, electrical part
    V0 = y[1]
    # update vector with voltage amplitudes and magnitudes, since nlsys.J() only updates the voltage amplitudes and angels that are in x
    nlsyse.V_vec_mag[0] = V0
    Ne = len(nlsyse.elecnetwork.nodes)
    Fe_ind = nlsyse.FP + [Ne+ind for ind in nlsyse.FQ]
    Ge_ind = [1,Ne,Ne+1] # indices of conservation of energy equations within full J
    xlfe_ind = nlsyse.xdelta + [Ne+ind for ind in nlsyse.xV]
    len_slack_e = len(xe) - len(xlfe_ind)
    dhe_dxe = np.zeros((len_xe,len_xe))
    Jee_full = nlsyse.J_dense(xe[:len(nlsyse.x_entries)],return_full=True)
    dhe_dxe[:len(Fe_ind),:][:,:len(xlfe_ind)] = Jee_full[Fe_ind,:][:,xlfe_ind] #Jee
    dhe_dxe[len(Fe_ind):,:][:,:len(xlfe_ind)] = Jee_full[Ge_ind,:][:,xlfe_ind] #dGe_dxlfe
    dhe_dxe[len(Fe_ind):,:][:,len(xlfe_ind):] = np.eye(len_slack_e)
    dhe_dxc = np.zeros((len_xe,max(len_xc,len_xc_u)))
    dhe_dxc[0,1] = -1 # der. of cons. of active power in node 0 to Pc0

    # Jacobians of extended LF equations, coupling part
    if len_xc==0:
        xc = u[len(u)-len_u:]
    with HiddenPrints():
        Jcc = nlsysc.J_dense(xc)
    len_Fc = Jcc.shape[0]
    dhc_dxg = np.zeros((len_Fc,len_xg))
    dhc_dxg[0,len(xlfg_ind)] = -MES.eta_GG0*MES.GHV #dFc0_dq0,c
    dhc_dxe = np.zeros((len_Fc,len_xe))
    dhc_dxe[1,len(xlfe_ind)] = -1 #dFc1_dP1,c

    # derivatives of G and F to u, gas part
    dhg_du = np.zeros((len(xg),len_u))
    dhg_du[len(Fg_ind):,:][:,0] = -1 #dGg_du

    # derivatives of G and F to u, electrical part
    V0_ind = Ne # index of V0 within the full Jee
    dhe_du = np.zeros((len(xe),len_u))
    dhe_du[:len(Fe_ind),:][:,1] = Jee_full[Fe_ind,V0_ind].ravel() # dFe_dV2
    dhe_du[len(Fe_ind):len(Fe_ind)+len(Ge_ind),:][:,1] = Jee_full[Ge_ind,V0_ind].ravel() # dGe_dV2

    # Fill Jacobian
    if len_xc > 0:
        dh_dy[:len(Fg_ind)+len(Gg_ind),:][:,:len_u] = dhg_du
        dh_dy[:len(Fg_ind)+len(Gg_ind),:][:,len_u:len_u+len_xg] = dhg_dxg
        dh_dy[:len(Fg_ind)+len(Gg_ind),:][:,-len_xc:] = dhg_dxc
        dh_dy[len(Fg_ind)+len(Gg_ind):len(Fg_ind)+len(Gg_ind)+len(Fe_ind)+len(Ge_ind),:][:,:len_u] = dhe_du
        dh_dy[len(Fg_ind)+len(Gg_ind):len(Fg_ind)+len(Gg_ind)+len(Fe_ind)+len(Ge_ind),:][:,len_u+len_xg:len_u+len_xg+len_xe] = dhe_dxe
        dh_dy[len(Fg_ind)+len(Gg_ind):len(Fg_ind)+len(Gg_ind)+len(Fe_ind)+len(Ge_ind),:][:,-len_xc:] = dhe_dxc
        dh_dy[-len_Fc:,:][:,len_u:len_u+len_xg] = dhc_dxg
        dh_dy[-len_Fc:,:][:,len_u+len_xg:len_u+len_xg+len_xe] = dhc_dxe
        dh_dy[-len_Fc:,:][:,-len_xc:] = Jcc
    else: # xc is part of u
        dh_dy[:len(Fg_ind)+len(Gg_ind),:][:,:len_u] = dhg_du
        dh_dy[:len(Fg_ind)+len(Gg_ind),:][:,len_u+len_xc_u:len_u+len_xc_u+len_xg] = dhg_dxg
        dh_dy[:len(Fg_ind)+len(Gg_ind),:][:,len_u:len_u+len_xc_u] = dhg_dxc
        dh_dy[len(Fg_ind)+len(Gg_ind):len(Fg_ind)+len(Gg_ind)+len(Fe_ind)+len(Ge_ind),:][:,:len_u] = dhe_du
        dh_dy[len(Fg_ind)+len(Gg_ind):len(Fg_ind)+len(Gg_ind)+len(Fe_ind)+len(Ge_ind),:][:,len_u+len_xc_u+len_xg:len_u+len_xc_u+len_xg+len_xe] = dhe_dxe
        dh_dy[len(Fg_ind)+len(Gg_ind):len(Fg_ind)+len(Gg_ind)+len(Fe_ind)+len(Ge_ind),:][:,len_u:len_u+len_xc_u] = dhe_dxc
        dh_dy[-len_Fc:,:][:,len_u+len_xc_u:len_u+len_xc_u+len_xg] = dhc_dxg
        dh_dy[-len_Fc:,:][:,len_u+len_xc_u+len_xg:len_u+len_xc_u+len_xg+len_xe] = dhc_dxe
        dh_dy[-len_Fc:,:][:,len_u:len_u+len_xc_u] = Jcc

    if nlsysc.scale_var == 'matrix':
        dh_dy = Dh.dot(dh_dy.dot(Dy_inv))
    # plt.figure('Jacobian')
    # plt.spy(dh_dy)
    # plt.figure('Jacobian values')
    # jac_val = dh_dy.copy()
    # jac_val[jac_val == 0] = np.nan
    # plt.imshow(jac_val)
    # if len_xc > 0:
    #     plt.plot([len_u-.5,len_u-.5],[0-.5,len_h-.5],'k')
    #     plt.plot([len_u+len_xg-.5,len_u+len_xg-.5],[0-.5,len_h-.5],'k')
    #     plt.plot([len_u+len_xg+len_xe-.5,len_u+len_xg+len_xe-.5],[0-.5,len_h-.5],'k')
    # else:
    #     plt.plot([len(u)-.5,len(u)-.5],[0-.5,len_h-.5],'k')
    # plt.plot([0-.5,len(y)-.5],[len(Fg_ind)-.5,len(Fg_ind)-.5],':k')
    # plt.plot([0-.5,len(y)-.5],[dhg_dxg.shape[0]-.5,dhg_dxg.shape[0]-.5],'k')
    # plt.plot([0-.5,len(y)-.5],[len(Fe_ind)+dhg_dxg.shape[0]-.5,len(Fe_ind)+dhg_dxg.shape[0]-.5],':k')
    # plt.plot([0-.5,len(y)-.5],[dhg_dxg.shape[0]+dhe_dxe.shape[0]-.5,dhg_dxg.shape[0]+dhe_dxe.shape[0]-.5],'k')
    return dh_dy

def update_bc(gas_net,elec_net,het_net,q0,V0,scale_var=None,scale_var_params=None):
    """Update the boundary conditions of the multi-carrier network, based on the control variables of OF"""
    #NB: Currently assumes coupling at node 1, with node set 1!!!!
    if scale_var == 'per_unit':
        q0 = q0*scale_var_params.get('qbase')
        V0 = V0*scale_var_params.get('Vbase')
    gas_net.nodes[0].half_links[0].q = q0 #<0
    elec_net.nodes[0].V = V0
    return gas_net,elec_net,het_net

def trans_back(y,nlsys=None,Dy_inv=None,Tx_inv=None,reorder=False):
    """Transform the vector of variables back, to the unscaled vector in the original order."""
    y_back = y.copy()
    if reorder:
        x = xmes_from_yopf(y_back,nlsys=nlsys)
        x = Tx_inv.dot(x)
        len_u = len(y)-len(x)
        y_back[len_u:] = x.copy()
    if nlsys.scale_var == 'matrix':
        y_back = Dy_inv.dot(y_back)
    return y_back

def run_optimal_load_flow(u_lb=np.array([1.3*MES.q0_source,.95*MES.V0_sol]),u_ub=np.array([0,1.05*MES.V0_sol]),u_init=np.array([1.01*MES.q0_source,1.01*MES.V0_sol]),x_lb=np.array([-1*abs(1.1*MES.q0_source),0,-np.pi,.5*abs(MES.qc0_sol),.5*abs(MES.qc1_sol),.5*MES.Pc0_sol,.5*MES.Pc1_sol,.5*MES.Qc0_sol,.5*MES.Qc1_sol]),x_ub=np.array([abs(1.1*MES.q0_source),1.05*MES.pg0,np.pi,1.5*abs(MES.qc0_sol),1.5*abs(MES.qc1_sol),1.5*MES.Pc0_sol,1.5*MES.Pc1_sol,1.5*MES.Qc0_sol,1.5*MES.Qc1_sol]),x_init=np.array([MES.q_ic,.99*MES.pg_ic,0,MES.qc_ic,MES.qc_ic,MES.Pc_ic,MES.Pc_ic,MES.Qc_ic,MES.Qc_ic]),a=np.array([0,0,0]),b=np.array([15*MES.GHV,2,3])/MW,c=np.array([0,0.02,0.03])/MW,y_ind=[0,7,8],formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,optimization_method='trust-constr',stay_within_bounds=False,fb=None,runs=1,reorder=False):
    """Run optimal load flow, with LF as equality constraints.

    Parameters
    ----------
    y_ind :list, optional
        Indices within y of the variables for the objective function.
    reorder : bool, optional
        If True, the LF equations and variables are reorded to match the order of the system of LF eq's for DD.

    Returns
    -------
    xmes_opt : np array
        Solution of LF variables of OF. Unscaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_times : float
        List with time of the optimization (excluding creation of network etc.).
    """
    if scale_var == 'per_unit':
        raise NotImplementedError('OF not implemented for per unit scaling.')
    print('\nRunning OF for GE2N with LF as eq. constr. (formulations: {}, hard bounds: {}, method: {}, scaling: {}, reordered: {})'.format(formulation,stay_within_bounds,optimization_method,scale_var,reorder))

    # create network
    with HiddenPrints():
        het_net, gas_net, elec_net = MES.create_mes_network()
    # update BCs
    q0,V0 = u_init
    gas_net,elec_net,het_net = update_bc(gas_net,elec_net,het_net,q0,V0,scale_var=scale_var,scale_var_params=scale_var_params)

    # set initial guess and initialize network, using default initial guess
    if len(x_init):
        het_net.initialize()
        het_net.update(x_init,formulation=formulation)
    else:
        x_init = set_xmes_LF_init(het_net,formulation=formulation)
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # initial guess for OF (unscaled)
    y0 = np.concatenate((u_init,x_init))

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        Dh = nlsys.DF()
        ubase = np.array([scale_var_params.get('qbase'),scale_var_params.get('Vbase')])
        Dy = np.diag(np.concatenate((1/ubase,Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((ubase,1/Dx.data[0])))
        y0 = Dy.dot(y0) # scale y
    else:
        Dy=np.eye(len(y0))
        Dy_inv=np.eye(len(y0))
        Dh=np.eye(len(x_init))

    if scale_var == 'per_unit':
        a = a/fb
        b = b/(fb/np.diag(Dy_inv)[np.array(y_ind)])
        c = c/(fb/np.diag(Dy_inv)[np.array(y_ind)]**2)

    if reorder:
        Dx_red, Px, DF_red, PF = MES.trans_matr_mes()
        Tx = Px.dot(Dx_red)
        Px_inv = Px.transpose()
        Dx_red_inv = sps.diags(1/Dx_red.data[0])
        Tx_inv = Dx_red_inv.dot(Px_inv)
        TF = PF.dot(DF_red)
        PF_inv = PF.transpose()
        DF_red_inv = sps.diags(1/DF_red.data[0])
        TF_inv = DF_red_inv.dot(PF_inv)
    else:
        TF=np.eye(len(x_init))
        Tx=np.eye(len(x_init))
        TF_inv=np.eye(len(x_init))
        Tx_inv=np.eye(len(x_init))
    x0 = y0[len(u_init):].copy()
    x0 = Tx.dot(x0)
    y0[len(u_init):] = x0.copy()

    # define objective function
    def obj(y,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys,Dy_inv=Dy_inv,Tx_inv=Tx_inv,fb=fb,reorder=reorder):
        global y_f_vec
        y_f_vec = y.copy()
        global f_vec_global
        # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
        y = trans_back(y,nlsys=nlsys,Dy_inv=Dy_inv,Tx_inv=Tx_inv,reorder=reorder)
        f = objective_function(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    def obj_grad(y,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb, Dy_inv=Dy_inv,Tx_inv=Tx_inv,reorder=reorder):
        # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
        y = trans_back(y,nlsys=nlsys,Dy_inv=Dy_inv,Tx_inv=Tx_inv,reorder=reorder)
        return grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb,Dy_inv=Dy_inv,Tx_inv=Tx_inv,reorder=reorder)

    def obj_hess(y,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb,Dy_inv=Dy_inv,Tx_inv=Tx_inv,reorder=reorder):
        # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
        y = trans_back(y,nlsys=nlsys,Dy_inv=Dy_inv,Tx_inv=Tx_inv,reorder=reorder)
        return hess_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb,Dy_inv=Dy_inv,Tx_inv=Tx_inv,reorder=reorder)

    # define nonlinear equality constraints (load flow equations)
    def eq_constr(y,nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh,Tx_inv=Tx_inv,TF=TF,reorder=reorder):
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_mes = nlsys.hetnetwork
        # Optimizer uses scaled y, but eq. constr. function wants unscaled y (when using matrix scaling)
        y = trans_back(y,nlsys=nlsys,Dy_inv=Dy_inv,Tx_inv=Tx_inv,reorder=reorder)
        # update bc of the network
        q0,V0 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network_g,network_e,network_mes = update_bc(network_g,network_e,network_mes,q0,V0,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        H = h(y, nlsys=nlsys, Dh=Dh, TF=TF,reorder=reorder)
        global err_LF_vec_global
        err_LF_vec_global.append(np.linalg.norm(H))
        return H

    def jac_eq_constr(y,nlsys=nlsys, Dy_inv=Dy_inv, Dh=Dh,Tx_inv=Tx_inv,TF=TF,reorder=reorder):
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_mes = nlsys.hetnetwork
        # Optimizer uses scaled y, but eq. constr. function wants unscaled y (when using matrix scaling)
        y = trans_back(y,nlsys=nlsys,Dy_inv=Dy_inv,Tx_inv=Tx_inv,reorder=reorder)
        # update bc of the network
        q0,V0 = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network_g,network_e,network_mes = update_bc(network_g,network_e,network_mes,q0,V0,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        dh_dy = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh,TF=TF,Tx_inv=Tx_inv,reorder=reorder)
        return dh_dy

    lb_nleq = np.zeros(len(x_lb))
    ub_nleq = np.zeros(len(x_ub))
    if optimization_method == 'trust-constr':
        nonlinear_constraint = spo.NonlinearConstraint(eq_constr,lb_nleq,ub_nleq,jac=jac_eq_constr,keep_feasible=stay_within_bounds)
    elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        nonlinear_constraint = {'type':'eq','fun':eq_constr,'jac':jac_eq_constr}

    # define linear inequality constraints, i.e. define bounds
    lb_ineq = np.concatenate((u_lb,x_lb))
    ub_ineq = np.concatenate((u_ub,x_ub))
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        lb_ineq = Dy.dot(lb_ineq)
        ub_ineq = Dy.dot(ub_ineq)
    if reorder:
        # transform
        lb_ineq_reord = np.ravel(Tx.dot(lb_ineq[len(u_lb):]))
        ub_ineq_reord = np.ravel(Tx.dot(ub_ineq[len(u_ub):]))
        # sign of variable might be flipped, i.e. scaling with -1 might have occured, so switch lower and upper bounds
        for ind in range(len(x_init)):
            if lb_ineq_reord[ind] < ub_ineq_reord[ind]:
                lb_ineq[ind+len(u_init)] = lb_ineq_reord[ind]
                ub_ineq[ind+len(u_init)] = ub_ineq_reord[ind]
            else:
                lb_ineq[ind+len(u_init)] = ub_ineq_reord[ind]
                ub_ineq[ind+len(u_init)] = lb_ineq_reord[ind]

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

    if scale_var == 'matrix' or scale_var == 'per_unit' or reorder:
        y_opf = trans_back(res.x,nlsys=nlsys,Dy_inv=Dy_inv,Tx_inv=Tx_inv,reorder=reorder)
    else:
        y_opf = res.x.copy()

    # print solution
    q0, V0 = y_opf[:len(u_init)] # unscaled
    gas_net,elec_net,het_net = update_bc(gas_net,elec_net,het_net,q0,V0)
    xmes_opt = xmes_from_yopf(y_opf,nlsys=nlsys) # unscaled
    print('Solution OF (success: {})'.format(res.success))
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation)
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
    return xmes_opt, res, f_vec, err_LF_vec, execution_times

def update_bc_elec(elec_net,V0,P0_c,scale_var=None,scale_var_params=None):
    if scale_var == 'per_unit':
        V0 = V0*scale_var_params.get('Vbase')
    elec_net.nodes[0].V = V0
    elec_net = MES.update_bc_elec(elec_net,P0_c,scale_var=scale_var,scale_var_params=scale_var_params)
    return elec_net

def update_bc_dd(gas_net,elec_net,coupling_net,u,xg,xe,xc,scale_var=None,scale_var_params=None):
    """Update the BCs of all separate (single-carrier) networks used in DD"""
    if len(u) == 2:
        q0,V0 = u
        q1c,Pc0 = xc
    elif len(u) == 4:
        q0,V0,q1c,Pc0 = u
    qc1 = -q1c
    q1_c = MES.ic_gas(qc1)
    gas_net = MES.update_bc_gas(gas_net,q1_c,scale_var=scale_var,scale_var_params=scale_var_params) #q0 is only used in Gg, so q0 is never explicitly set (as BC) on any network element.
    P0_c = MES.ic_elec(Pc0)
    elec_net = update_bc_elec(elec_net,V0,P0_c,scale_var=scale_var,scale_var_params=scale_var_params)
    q0_c = xg[-1]
    P1_c,Q0_c,Q1_c = xe[1:]
    qc0,Pc1,Qc0,Qc1 = MES.ic_coupling(q0_c,P1_c,Q0_c,Q1_c)
    coupling_net = MES.update_bc_coupling(coupling_net,qc0,Pc1,Qc0,Qc1,scale_var=scale_var,scale_var_params=scale_var_params)
    return gas_net,elec_net,coupling_net

def run_optimal_load_flow_dd(u_lb=np.array([1.3*MES.q0_source,.95*MES.V0_sol]),u_ub=np.array([0,1.05*MES.V0_sol]),u_init=np.array([1.01*MES.q0_source,1.01*MES.V0_sol]),xg_lb=np.array([-1*abs(1.1*MES.q0_source),0,.5*abs(MES.qc0_sol)]),xe_lb=np.array([-np.pi,-1.5*MES.Pc1_sol,-1.5*MES.Qc0_sol,-1.5*MES.Qc1_sol]),xc_lb=np.array([.5*abs(MES.qc1_sol),.5*MES.Pc0_sol]),xg_ub=np.array([abs(1.1*MES.q0_source),1.05*MES.pg0,1.5*abs(MES.qc0_sol)]),xe_ub=np.array([np.pi,-.5*MES.Pc1_sol,-.5*MES.Qc0_sol,-.5*MES.Qc1_sol]),xc_ub=np.array([1.5*abs(MES.qc1_sol),1.5*MES.Pc0_sol]),xg_init=np.array([MES.q_ic,.99*MES.pg_ic,MES.qc_ic]),xe_init=np.array([0,-MES.Pc_ic,-MES.Qc_ic,-MES.Qc_ic]),xc_init=np.array([MES.qc_ic,MES.Pc_ic]),a=np.array([0,0,0]),b=np.array([15*MES.GHV,2,3])/MW,c=np.array([0,0.02,0.03])/MW,y_ind=[0,10,6],formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,optimization_method='trust-constr',stay_within_bounds=False,fb=None,runs=1):
    """Run optimal load flow, with LF as equality constraints, using DD in LF.

    Parameters
    ----------
    y_ind :list, optional
        Indices within y of the variables for the objective function.

    Returns
    -------
    xmes_opt : np array
        Solution of LF variables of OF. Unscaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_times : float
        List with time of the optimization (excluding creation of network etc.).
    """
    if scale_var == 'per_unit':
        raise NotImplementedError('OF not implemented for per unit scaling.')
    print('\nRunning OF for GE2N with DD LF as eq. constr. (formulations: {}, hard bounds: {}, method: {}, scaling: {})'.format(formulation,stay_within_bounds,optimization_method,scale_var))

    # create network
    with HiddenPrints():
        gas_net = MES.create_gas_network(scale_var=scale_var,scale_var_params=scale_var_params)
        elec_net = MES.create_electrical_network(scale_var=scale_var,scale_var_params=scale_var_params)
        coupling_net = MES.create_coupling_network(scale_var=scale_var,scale_var_params=scale_var_params)

    # set initial guess and initialize network, using default initial guess
    if len(xg_init):
        gas_net.initialize()
        gas_net.update(xg_init[:2],formulation=formulation.get('gas'))
    else:
        xg_init = MES.set_xg_LF_init(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    if len(xe_init):
        elec_net.initialize()
        elec_net.update(xe_init[:1],formulation=formulation.get('elec'))
    else:
        xe_init = MES.set_xe_LF_init(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    if len(u_init) == 2:
        if len(xc_init):
            coupling_net.initialize()
            coupling_net.update(xc_init,formulation=formulation)
        else:
            xc_init = MES.set_xc_LF_init(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    elif len(u_init) == 4:
        coupling_net.initialize()
        coupling_net.update(u_init[2:],formulation=formulation)
    nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # update BCs
    if len(u_init) == 2:
        q0,V0 = u_init
        q1c,Pc0 = xc_init
    elif len(u_init) == 4:
        q0,V0,q1c,Pc0 = u_init
    qc1 = -q1c
    q1_c = MES.ic_gas(qc1)
    gas_net = MES.update_bc_gas(gas_net,q1_c,scale_var=scale_var,scale_var_params=scale_var_params) #q0 is only used in Gg, so q0 is never explicitly set (as BC) on any network element.
    P0_c = MES.ic_elec(Pc0)
    elec_net = update_bc_elec(elec_net,V0,P0_c,scale_var=scale_var,scale_var_params=scale_var_params)
    q0_c = xg_init[-1]
    P1_c,Q0_c,Q1_c = xe_init[1:]
    qc0,Pc1,Qc0,Qc1 = MES.ic_coupling(q0_c,P1_c,Q0_c,Q1_c)
    coupling_net = MES.update_bc_coupling(coupling_net,qc0,Pc1,Qc0,Qc1,scale_var=scale_var,scale_var_params=scale_var_params)

    # initial guess for OF (unscaled)
    y0 = np.concatenate((u_init,xg_init,xe_init,xc_init))
    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dxg = nlsysg.Dx()
        DFg = nlsysg.DF()
        Gg_base = np.array([scale_var_params.get('qbase')])
        Dxe = nlsyse.Dx()
        DFe = nlsyse.DF()
        len_slack_e = len(xe_init)-Dxe.shape[0]
        Ge_base = scale_var_params.get('Sbase')*np.ones(len_slack_e)
        Dxc = nlsysc.Dx()
        DFc = nlsysc.DF()
        ubase = np.array([scale_var_params.get('qbase'),scale_var_params.get('Vbase')])
        slackg_base = np.array([scale_var_params.get('qbase')])
        slacke_base = scale_var_params.get('Sbase')*np.ones(len_slack_e)
        if len(u_init) == 2:
            Dy = np.diag(np.concatenate((1/ubase,Dxg.data[0],1/slackg_base,Dxe.data[0],1/slacke_base,Dxc.data[0])))
        elif len(u_init) == 4:
            Dy = np.diag(np.concatenate((1/ubase,Dxc.data[0],Dxg.data[0],1/slackg_base,Dxe.data[0],1/slacke_base)))
        Dy_inv = np.diag(1/np.diag(Dy))
        Dh = np.diag(np.concatenate((DFg.data[0],1/Gg_base,DFe.data[0],1/Ge_base,DFc.data[0])))
        y0 = Dy.dot(y0) # scale y
    else:
        Dy=np.eye(len(y0))
        Dy_inv=np.eye(len(y0))
        Dh=np.eye(len(y0)-len(u_init))

    if scale_var == 'per_unit':
        a = a/fb
        b = b/(fb/np.diag(Dy_inv)[np.array(y_ind)])
        c = c/(fb/np.diag(Dy_inv)[np.array(y_ind)]**2)

    # define objective function
    def obj(y,y_ind=y_ind,a=a,b=b,c=c,nlsysc=nlsysc,Dy_inv=Dy_inv,fb=fb):
        global y_f_vec
        y_f_vec = y.copy()
        global f_vec_global
        # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
        if nlsysc.scale_var == 'matrix':
            # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
            y = Dy_inv.dot(y)
        f = objective_function(y,y_ind,a=a,b=b,c=c,scale_var=nlsysc.scale_var,scale_var_params=nlsysc.scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    def obj_grad(y,y_ind=y_ind,a=a,b=b,c=c,nlsysc=nlsysc, fb=fb, Dy_inv=Dy_inv):
        # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
        if nlsysc.scale_var == 'matrix':
            # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
            y = Dy_inv.dot(y)
        return grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsysc.scale_var,scale_var_params=nlsysc.scale_var_params,fb=fb,Dy_inv=Dy_inv)

    def obj_hess(y,y_ind=y_ind,a=a,b=b,c=c,nlsysc=nlsysc, fb=fb,Dy_inv=Dy_inv):
        # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
        if nlsysc.scale_var == 'matrix':
            # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
            y = Dy_inv.dot(y)
        return hess_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsysc.scale_var,scale_var_params=nlsysc.scale_var_params,fb=fb,Dy_inv=Dy_inv)

    # define nonlinear equality constraints (load flow equations)
    def eq_constr(y,nlsysg=nlsysg, nlsyse=nlsyse, nlsysc=nlsysc, len_xg=len(xg_init), len_xe=len(xe_init), len_xc=len(xc_init),Dy_inv=Dy_inv, Dh=Dh):
        """(extended) LF equations using DD"""
        network_g = nlsysg.gasnetwork
        network_e = nlsyse.elecnetwork
        network_c = nlsysc.hetnetwork
        if nlsysc.scale_var == 'matrix':
            # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the networks
        u,xg,xe,xc = xdd_from_yopf(y,len_xg=len_xg, len_xe=len_xe, len_xc=len_xc)
        network_g,network_e,network_c = update_bc_dd(network_g,network_e,network_c,u,xg,xe,xc,scale_var=nlsysc.scale_var,scale_var_params=nlsysc.scale_var_params)
        network_e = MES.reset_network_without_update_elec(network_e)
        H = h_dd(y, nlsysg=nlsysg,nlsyse=nlsyse,nlsysc=nlsysc,len_xg=len_xg,len_xe=len_xe,len_xc=len_xc,Dh=Dh)
        global err_LF_vec_global
        err_LF_vec_global.append(np.linalg.norm(H))
        return H

    def jac_eq_constr(y,nlsysg=nlsysg, nlsyse=nlsyse, nlsysc=nlsysc, len_xg=len(xg_init), len_xe=len(xe_init), len_xc=len(xc_init),Dy_inv=Dy_inv, Dh=Dh):
        """Jacobian of (extended) LF equations using DD"""
        network_g = nlsysg.gasnetwork
        network_e = nlsyse.elecnetwork
        network_c = nlsysc.hetnetwork
        if nlsysc.scale_var == 'matrix':
            # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the networks
        u,xg,xe,xc = xdd_from_yopf(y,len_xg=len_xg, len_xe=len_xe, len_xc=len_xc)
        network_g,network_e,network_c = update_bc_dd(network_g,network_e,network_c,u,xg,xe,xc,scale_var=nlsysc.scale_var,scale_var_params=nlsysc.scale_var_params)
        network_e = MES.reset_network_without_update_elec(network_e)
        dh_dy = h_der_dd(y,nlsysg=nlsysg,nlsyse=nlsyse,nlsysc=nlsysc,len_xg=len_xg,len_xe=len_xe,len_xc=len_xc,Dh=Dh,Dy_inv=Dy_inv)
        return dh_dy

    lb_nleq = np.zeros(Dh.shape[0])
    ub_nleq = np.zeros(Dh.shape[0])
    if optimization_method == 'trust-constr':
        nonlinear_constraint = spo.NonlinearConstraint(eq_constr,lb_nleq,ub_nleq,jac=jac_eq_constr,keep_feasible=stay_within_bounds)
    elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        nonlinear_constraint = {'type':'eq','fun':eq_constr,'jac':jac_eq_constr}

    # define linear inequality constraints, i.e. define bounds
    lb_ineq = np.concatenate((u_lb,xg_lb,xe_lb,xc_lb))
    ub_ineq = np.concatenate((u_ub,xg_ub,xe_ub,xc_ub))
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        lb_ineq = Dy.dot(lb_ineq)
        ub_ineq = Dy.dot(ub_ineq)

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
                raise NotImplementedError('not implemented for more than one run')
                f_vec_global = list()
                y_f_vec = y0.copy()
                err_LF_vec_global = list()
                if optimization_method == 'trust-constr':
                    f_vec = list()
                elif optimization_method == 'SLSQP':
                    f_vec = [obj(y0)] # this call to obj() alters all the global variables.
            opf_start_time = time.perf_counter()
            try:
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
                print('Exception made for {}, formulations: {}, hard bounds: {}, scaling: {}'.format(optimization_method,formulation,stay_within_bounds,scale_var))
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
    u,xg,xe,xc = xdd_from_yopf(y_opf, len_xg=len(xg_init), len_xe=len(xe_init), len_xc=len(xc_init))
    gas_net,elec_net,coupling_net = update_bc_dd(gas_net,elec_net,coupling_net,u,xg,xe,xc)
    with HiddenPrints():
        het_net, gas_net_mes, elec_net_mes = MES.create_mes_network()
        het_net.initialize()
    xg_mes = xg[:2]
    xe_mes = np.array([xe[0]])
    q0_c = xg[-1]
    P1_c,Q0_c,Q1_c = xe[1:]
    if len(u_init) == 2:
        q1c,Pc0 = xc
    elif len(u_init) == 4:
        q0,V0,q1c,Pc0 = u
    qc0,Pc1,Qc0,Qc1 = MES.ic_coupling(q0_c,P1_c,Q0_c,Q1_c)
    xc_mes = np.array([qc0,q1c,Pc0,Pc1,Qc0,Qc1])
    xmes_opt = np.concatenate((xg_mes,xe_mes,xc_mes))
    gas_net_mes,elec_net_mes,het_net = update_bc(gas_net_mes,elec_net_mes,het_net,q0,V0)
    print('Solution OF (success: {})'.format(res.success))
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation)
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
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net_mes.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net_mes.get_nodes() for hl in node.get_half_links()]))
    return xmes_opt, res, f_vec, err_LF_vec, execution_times

def get_u_from_net(gas_net,elec_net,het_net,scale_var=None,scale_var_params=None):
    """Get the current values of the control variables u from the network.
    """
    q0 = gas_net.nodes[0].half_links[0].get_q(scale_var=scale_var,scale_var_params=scale_var_params)
    V0 = elec_net.nodes[0].get_V(scale_var=scale_var,scale_var_params=scale_var_params)
    return np.array([q0,V0])


def solve_lf_in_of(u,nlsys,max_iters=10,tol=1e-6,xLF_init=np.array([]),Tx=None,TF=None,reorder=False):
    """Solve steady-state LF within an optimization context.
    """
    global err_LF_vec_global
    network_g = nlsys.nlsystemsg[0].gasnetwork
    network_e = nlsys.nlsystemse[0].elecnetwork
    network_mes = nlsys.hetnetwork

    u_net = get_u_from_net(network_g,network_e,network_mes,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    if not reorder:
        TF=np.array([])
        Tx=np.array([])
    if len(err_LF_vec_global)==0 or (np.linalg.norm(u-u_net) > tol) or (err_LF_vec_global[-1] > tol):
        q0,V0 = u
        network_g,network_e,network_mes = update_bc(network_g,network_e,network_mes,q0,V0,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        with HiddenPrints():
            xmes,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = network_mes.solve_network(tol,max_iters,solver='NR',formulation=nlsys.formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,P_F=TF,P_x=Tx)
            if err_vec[-1] >= tol:
                network_mes.update(xLF_init,formulation=nlsys.formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
                xmes,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = network_mes.solve_network(tol,max_iters,solver='NR',formulation=nlsys.formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,P_F=TF,P_x=Tx)
        err_LF = err_vec[-1]
    else:
        xmes = network_mes.set_x_init(formulation=nlsys.formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = network_mes.update_full(xmes,formulation=nlsys.formulation,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,P_F=TF,P_x=Tx)
        err_LF = err_LF_vec_global[-1]
    err_LF_vec_global.append(err_LF)
    return xmes

def run_optimal_load_flow_separate_LF(u_lb=np.array([1.3*MES.q0_source,.95*MES.V0_sol]),u_ub=np.array([0,1.05*MES.V0_sol]),u_init=np.array([1.01*MES.q0_source,1.01*MES.V0_sol]),x_lb=np.array([-1*abs(1.1*MES.q0_source),0,-np.pi,.5*abs(MES.qc0_sol),.5*abs(MES.qc1_sol),.5*MES.Pc0_sol,.5*MES.Pc1_sol,.5*MES.Qc0_sol,.5*MES.Qc1_sol]),x_ub=np.array([abs(1.1*MES.q0_source),1.05*MES.pg0,np.pi,1.5*abs(MES.qc0_sol),1.5*abs(MES.qc1_sol),1.5*MES.Pc0_sol,1.5*MES.Pc1_sol,1.5*MES.Qc0_sol,1.5*MES.Qc1_sol]),x_init=np.array([MES.q_ic,.99*MES.pg_ic,0,MES.qc_ic,MES.qc_ic,MES.Pc_ic,MES.Pc_ic,MES.Qc_ic,MES.Qc_ic]),a=np.array([0,0,0]),b=np.array([15*MES.GHV,2,3])/MW,c=np.array([0,0.02,0.03])/MW,y_ind=[0,7,8],formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,max_iters_lf=10,approach='direct',optimization_method='trust-constr',stay_within_bounds=False,fb=None,runs=1,reorder=False):
    """Run optimal load flow, with LF as equality constraints.

    Parameters
    ----------
    y_ind :list, optional
        Indices within y of the variables for the objective function.
    reorder : bool, optional
        If True, the LF equations and variables are reorded to match the order of the system of LF eq's for DD.

    Returns
    -------
    xmes_opt : np array
        Solution of LF variables of OF. Unscaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_times : float
        List with time of the optimization (excluding creation of network etc.).
    """
    if scale_var == 'per_unit':
        raise NotImplementedError('OF not implemented for per unit scaling.')
    print('\nRunning OF for GE2N with LF as subsystem (formulations: {}, hard bounds: {}, method: {}, scaling: {}, approach: {})'.format(formulation,stay_within_bounds,optimization_method,scale_var,approach))

    # create network
    with HiddenPrints():
        het_net, gas_net, elec_net = MES.create_mes_network()
    # update BCs
    q0,V0 = u_init
    gas_net,elec_net,het_net = update_bc(gas_net,elec_net,het_net,q0,V0,scale_var=scale_var,scale_var_params=scale_var_params)

    # set initial guess and initialize network, using default initial guess
    if len(x_init):
        het_net.initialize()
        het_net.update(x_init,formulation=formulation)
    else:
        x_init = set_xmes_LF_init(het_net,formulation=formulation)
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # initial guess for OF (unscaled)
    u0 = u_init

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        Dh = nlsys.DF()
        ubase = np.array([scale_var_params.get('qbase'),scale_var_params.get('Vbase')])
        Dy = np.diag(np.concatenate((1/ubase,Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((ubase,1/Dx.data[0])))
        Du = np.diag(1/ubase)
        Du_inv = np.diag(ubase)
        u0 = Du.dot(u0) # scale u
        if scale_var == 'per_unit':
            x_init = Dx.dot(x_init) # scale x
    else:
        Dy=np.eye(len(u0)+len(x_init))
        Dy_inv=np.eye(len(u0)+len(x_init))
        Du = np.eye(len(u0))
        Du_inv= np.eye(len(u0))
        Dh=np.eye(len(x_init))

    if scale_var == 'per_unit':
        a = a/fb
        b = b/(fb/np.diag(Dy_inv)[np.array(y_ind)])
        c = c/(fb/np.diag(Dy_inv)[np.array(y_ind)]**2)

    if reorder:
        Dx_red, Px, DF_red, PF = MES.trans_matr_mes()
        Tx = Px.dot(Dx_red)
        Px_inv = Px.transpose()
        Dx_red_inv = sps.diags(1/Dx_red.data[0])
        Tx_inv = Dx_red_inv.dot(Px_inv)
        TF = PF.dot(DF_red)
        PF_inv = PF.transpose()
        DF_red_inv = sps.diags(1/DF_red.data[0])
        TF_inv = DF_red_inv.dot(PF_inv)
    else:
        TF=np.eye(len(x_init))
        Tx=np.eye(len(x_init))
        TF_inv=np.eye(len(x_init))
        Tx_inv=np.eye(len(x_init))

    # limits on inequality constraints on state variables
    lb_ineq_state = x_lb.copy()
    ub_ineq_state = x_ub.copy()
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        lb_ineq_state = Dy[len(u_lb):,len(u_lb):].dot(lb_ineq_state)
        ub_ineq_state = Dy[len(u_ub):,len(u_ub):].dot(ub_ineq_state)
        u_lb = Du.dot(u_lb)
        u_ub = Du.dot(u_ub)
    if reorder:
        # transform
        lb_ineq_reord = np.ravel(Tx.dot(lb_ineq_state))
        ub_ineq_reord = np.ravel(Tx.dot(ub_ineq_state))
        # sign of variable might be flipped, i.e. scaling with -1 might have occured, so switch lower and upper bounds
        for ind in range(len(x_init)):
            if lb_ineq_reord[ind] < ub_ineq_reord[ind]:
                lb_ineq_state[ind] = lb_ineq_reord[ind]
                ub_ineq_state[ind] = ub_ineq_reord[ind]
            else:
                lb_ineq_state[ind] = ub_ineq_reord[ind]
                ub_ineq_state[ind] = lb_ineq_reord[ind]

    # define objective function
    def obj(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys,xLF_init=x_init,Dy_inv=Dy_inv,Du_inv=Du_inv,Tx=Tx,TF=TF,fb=fb,reorder=reorder):
        global u_f_vec
        u_f_vec = u.copy()
        global f_vec_global
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=x_init,Tx=Tx,TF=TF,reorder=reorder) # original ordering, unscaled (unless p.u. is used)
        y = np.concatenate((u,x))
        f = objective_function(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f
    def obj_grad(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys, fb=fb,xLF_init=x_init,Dy_inv=Dy_inv,Du_inv=Du_inv,Dh=Dh,Tx=Tx,Tx_inv=Tx_inv,TF=TF,reorder=reorder,method=approach):
        if nlsys.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=x_init,Tx=Tx,TF=TF,reorder=reorder) # original ordering, unscaled (unless p.u. is used)
        y = np.concatenate((u,x))
        # partial derivatives of objective
        deltaf_deltay = grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,fb=fb,Dy_inv=Dy_inv,Tx_inv=Tx_inv,reorder=reorder)
        deltaf_deltau = np.zeros((1,len(u)))
        deltaf_deltax = np.zeros((1,len(x)))
        deltaf_deltau[0,:] = deltaf_deltay[:len(u)]
        deltaf_deltax[0,:] = deltaf_deltay[len(u):]
        # partial derivatives of equatilty constraints / load-flow equations
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_mes = nlsys.hetnetwork
        q0, V0 = u # scaled for p.u., unscaled for matrix
        network_g,network_e,network_mes = update_bc(network_g,network_e,network_mes,q0,V0,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        deltah_deltay = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh,TF=TF,Tx_inv=Tx_inv,reorder=reorder)
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
    def g(u,nlsys=nlsys, lb_ineq_state=lb_ineq_state, ub_ineq_state=ub_ineq_state, Dy=Dy, Du_inv=Du_inv,Tx=Tx,Tx_inv=Tx_inv,TF=TF,reorder=reorder, max_iters_lf=max_iters_lf,tol=tol,xLF_init=x_init):
        if nlsys.scale_var == 'matrix':
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=x_init,Tx=Tx,TF=TF,reorder=reorder) # original ordering, unscaled (unless p.u. is used)
        y = np.concatenate((u,x))
        if scale_var == 'matrix': # lb_ineq_state and ub_ineq_state are scaled, so scale x as well
            x = Dy[len(u):,len(u):].dot(x)
        if reorder: # lb_ineq_state and ub_ineq_state are reordered, so reorder x as well
            x = Tx.dot(x)
        g = np.concatenate((x-lb_ineq_state,ub_ineq_state-x))
        return g
    def g_jac(u,nlsys=nlsys, lb_ineq_state=lb_ineq_state, ub_ineq_state=ub_ineq_state, Dy=Dy, Du_inv=Du_inv,Tx=Tx,Tx_inv=Tx_inv,TF=TF,reorder=reorder, max_iters_lf=max_iters_lf,tol=tol,xLF_init=x_init,method=approach):
        if nlsys.scale_var == 'matrix':
            u = Du_inv.dot(u)
        x = solve_lf_in_of(u,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=x_init,Tx=Tx,TF=TF,reorder=reorder) # original ordering, unscaled (unless p.u. is used)
        y = np.concatenate((u,x))
        # Jacobian of inequality constraints on state variables (without gamma, that part is added later)
        len_x = len(x)
        I = np.eye(len_x)
        deltag_deltax = np.vstack((I,-I))
        deltag_deltau = np.zeros((deltag_deltax.shape[0],len(u)))
        # partial derivatives of equatilty constraints / load-flow equations
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_mes = nlsys.hetnetwork
        q0, V0 = u # scaled for p.u., unscaled for matrix
        network_g,network_e,network_mes = update_bc(network_g,network_e,network_mes,q0,V0,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
        deltah_deltay = h_der(y, nlsys=nlsys,Dy_inv=Dy_inv, Dh=Dh,TF=TF,Tx_inv=Tx_inv,reorder=reorder)
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
        len_g = len(lb_ineq_state) + len(ub_ineq_state)
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
                raise NotImplementedError('OF not implemented for more than one run')
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
                print('Exception made for {} (formulations: {}, hard bounds: {}, scaling: {}, approach: {})'.format(optimization_method,formulation,stay_within_bounds,scale_var,approach))
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
        x = solve_lf_in_of(res.x,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=x_init,Tx=Tx,TF=TF,reorder=reorder)
        y_opt = Dy_inv.dot(np.concatenate((res.x,x))) # unscaled
    else:
        x = solve_lf_in_of(u_opf,nlsys,max_iters=max_iters_lf,tol=tol,xLF_init=x_init,Tx=Tx,TF=TF,reorder=reorder)
        y_opt = np.concatenate((u_opf,x)) # unscaled
    xmes_opt = xmes_from_yopf(y_opt,nlsys=nlsys) # unscaled
    print('Solution OF (success: {})'.format(res.success))
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation)
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
    return xmes_opt, y_opt, res, f_vec, err_LF_vec, execution_times

def solve_lf_in_of_dd(u,nlsysg,nlsyse,nlsysc,max_iters=10,tol=1e-6,xg_init=np.array([]),xe_init=np.array([])):
    """Solve steady-state LF within an optimization context.
    """
    global err_LF_vec_global
    global err_LF_vecg_global
    global err_LF_vece_global
    network_g = nlsysg.gasnetwork
    network_e = nlsyse.elecnetwork
    network_c = nlsysc.hetnetwork

    len_xc = len(nlsysc.x_entries)
    q1_c_net = MES.get_bc_gas(network_g,scale_var=nlsysg.scale_var,scale_var_params=nlsysg.scale_var_params)
    P0_c_net = MES.get_bc_elec(network_e,scale_var=nlsyse.scale_var,scale_var_params=nlsyse.scale_var_params)
    V0_net = network_e.nodes[0].get_V(scale_var=nlsyse.scale_var,scale_var_params=nlsyse.scale_var_params)
    ue_net = np.array([V0_net,P0_c_net])
    q0, V0, q1c, Pc0 = u
    # print('Solving DD LF equations...')
    # evaluate load flow equations gas
    qc1 = -q1c
    q1_c = MES.ic_gas(qc1)
    if len(err_LF_vecg_global)==0 or (abs(q1_c-q1_c_net) > tol) or (err_LF_vecg_global[-1] > tol):
        network_g = MES.update_bc_gas(network_g,q1_c,scale_var=nlsysg.scale_var,scale_var_params=nlsysg.scale_var_params)
        # print('Solving LF in gas network')
        # print('q1_c = {}kg/s'.format(q1_c))
        with HiddenPrints():
            xg,itersg,err_vecg,_,_,_ = network_g.solve_network(tol,max_iters,solver='NR',formulation=nlsysg.gas_formulation,scale_var=nlsysg.scale_var,scale_var_params=nlsysg.scale_var_params)
        if err_vecg[-1] >= tol:
            network_g.update(xg_init,formulation=nlsysg.gas_formulation)
            xg,itersg,err_vecg,_,_,_ = network_g.solve_network(tol,max_iters,solver='NR',formulation=nlsysg.gas_formulation,scale_var=nlsysg.scale_var,scale_var_params=nlsysg.scale_var_params)
        err_LFg = err_vecg[-1]
    else:
        xg = network_g.set_x_init(formulation=nlsysg.gas_formulation,scale_var=nlsysg.scale_var,scale_var_params=nlsysg.scale_var_params)
        network_g.update_full(xg,formulation=nlsysg.gas_formulation,scale_var=nlsysg.scale_var,scale_var_params=nlsysg.scale_var_params)
        err_LFg = err_LF_vecg_global[-1]
    q0_c = MES.G_gas(network_g,scale_var=nlsysg.scale_var,scale_var_params=nlsysg.scale_var_params,q0=q0)
    # print('q0 = {}kg/s, q0_c = {}kg/s, q01 = {}kg/s,-q0-q0_c-q01={:.3e}kg/s'.format(q0,q0_c,network_g.links[0].get_q(scale_var=nlsysg.scale_var,scale_var_params=nlsysg.scale_var_params),-q0-q0_c-network_g.links[0].get_q(scale_var=nlsysg.scale_var,scale_var_params=nlsysg.scale_var_params)))
    slack_g = np.array([q0_c])
    err_LF_vecg_global.append(err_LFg)
    # evaluate load flow equations electricity
    P0_c = MES.ic_elec(Pc0)
    ue = np.array([V0,P0_c])
    if len(err_LF_vece_global)==0 or (np.linalg.norm(ue-ue_net) > tol) or (err_LF_vece_global[-1] > tol):
        network_e = update_bc_elec(network_e,V0,P0_c,scale_var=nlsyse.scale_var,scale_var_params=nlsyse.scale_var_params)
        network_e = MES.reset_network_without_update_elec(network_e)
        # print('Solving LF in electrical network')
        with HiddenPrints():
            xe,iterse,err_vece,delta_sol,V_sol,S_inj,P_edge,Q_edge = network_e.solve_network(tol,max_iters,solver='NR',scale_var=nlsyse.scale_var,scale_var_params=nlsyse.scale_var_params)
        if err_vece[-1] >= tol:
            network_e = MES.reset_network_without_update_elec(network_e)
            network_e.update(xe_init,scale_var=nlsyse.scale_var,scale_var_params=nlsyse.scale_var_params)
            xe,iterse,err_vece,delta_sol,V_sol,S_inj,P_edge,Q_edge = network_e.solve_network(tol,max_iters,solver='NR',scale_var=nlsyse.scale_var,scale_var_params=nlsyse.scale_var_params)
        err_LFe = err_vece[-1]
    else:
        network_e = MES.reset_network_without_update_elec(network_e)
        xe = network_e.set_x_init(scale_var=nlsyse.scale_var,scale_var_params=nlsyse.scale_var_params)
        network_e.update_full(xe_init,scale_var=nlsyse.scale_var,scale_var_params=nlsyse.scale_var_params)
        err_LFe = err_LF_vece_global[-1]
    P1_c,Q0_c,Q1_c = MES.G_elec(network_e,scale_var=nlsyse.scale_var,scale_var_params=nlsyse.scale_var_params)
    slack_e = np.array([P1_c,Q0_c,Q1_c])
    err_LF_vece_global.append(err_LFe)

    x = np.concatenate((xg,slack_g,xe,slack_e))
    return x

def hc_dd(u, nlsysc=None,Dh=None):
    """Equality constraints h(x)=0 for the coupling part. Assumes BCs are already updated."""
    xc = u[-len(nlsysc.x_entries):]
    with HiddenPrints():
        h = nlsysc.F(xc) # scaled when matrix scaling is used
    if nlsysc.scale_var == 'matrix':
        h = Dh.dot(h)
    return h

def hc_deru_dd(u,nlsysc=None,Du_inv=None, Dh=None):
    """Derivative of equality constraints h(x)=0 for the coupling part, to u. Assumes BCs are already updated."""
    len_xc = len(nlsysc.x_entries)
    xc = u[-len_xc:]
    with HiddenPrints():
        Jcc = nlsysc.J_dense(xc)
    dh_du = np.zeros((len_xc,len(u)))
    dh_du[:,:][:,-len_xc:] = Jcc
    if nlsysc.scale_var == 'matrix':
        dh_du = Dh.dot(dh_du.dot(Du_inv))
    return dh_du

def hc_derx_dd(y, nlsysg=None,nlsyse=None, nlsysc=None,Dx_inv=None, Dh=None,len_xg=3, len_xe=4, len_xc=0):
    """Derivative of equality constraints h(x)=0 for the coupling part, to x. Assumes BCs are already updated."""
    len_h = Dh.shape[0]
    u, xg, xe, xc = xdd_from_yopf(y,len_xg=len_xg, len_xe=len_xe, len_xc=len_xc)
    len_xg = len(xg)
    len_xe = len(xe)
    len_x = len_xg+len_xe
    dh_dx = np.zeros((len_h,len_x))
    # to gas part
    Eg = len(nlsysg.gasnetwork.links)
    if nlsysg.gas_formulation=='nodal':
        xlfg_ind = nlsysg.ind_p
    else:
        xlfg_ind = nlsysg.ind_q + [Eg+ind for ind in nlsysg.ind_p]
    dh_dxg = np.zeros((len_h,len_xg))
    dh_dxg[0,len(xlfg_ind)] = -MES.eta_GG0*MES.GHV #dFc0_dq0,c
    # to electrical part
    Ne = len(nlsyse.elecnetwork.nodes)
    xlfe_ind = nlsyse.xdelta + [Ne+ind for ind in nlsyse.xV]
    dh_dxe = np.zeros((len_h,len_xe))
    dh_dxe[1,len(xlfe_ind)] = -1 #dFc1_dP1,c

    dh_dx[:,:][:,:len_xg] = dh_dxg
    dh_dx[:,:][:,-len_xe:] = dh_dxe
    if nlsysc.scale_var == 'matrix':
        dh_dx = Dh.dot(dh_dx.dot(Dx_inv))
    return dh_dx

def run_optimal_load_flow_separate_LF_dd(u_lb=np.array([1.3*MES.q0_source,.95*MES.V0_sol,.5*abs(MES.qc1_sol),.5*MES.Pc0_sol]),u_ub=np.array([0,1.05*MES.V0_sol,1.5*abs(MES.qc1_sol),1.5*MES.Pc0_sol]),u_init=np.array([1.01*MES.q0_source,1.01*MES.V0_sol,MES.qc_ic,MES.Pc_ic]),xg_lb=np.array([-1*abs(1.1*MES.q0_source),0,.5*abs(MES.qc0_sol)]),xe_lb=np.array([-np.pi,-1.5*MES.Pc1_sol,-1.5*MES.Qc0_sol,-1.5*MES.Qc1_sol]),xg_ub=np.array([abs(1.1*MES.q0_source),1.05*MES.pg0,1.5*abs(MES.qc0_sol)]),xe_ub=np.array([np.pi,-.5*MES.Pc1_sol,-.5*MES.Qc0_sol,-.5*MES.Qc1_sol]),xg_init=np.array([MES.q_ic,.99*MES.pg_ic,MES.qc_ic]),xe_init=np.array([0,-MES.Pc_ic,-MES.Qc_ic,-MES.Qc_ic]),a=np.array([0,0,0]),b=np.array([15*MES.GHV,2,3])/MW,c=np.array([0,0.02,0.03])/MW,y_ind=[0,3,8],formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,max_iters_lf=10,approach='direct',optimization_method='trust-constr',stay_within_bounds=False,fb=None,runs=1):
    """Run optimal load flow, with LF as equality constraints. Using DD within LF. Only 'option 2' for DD, that is, xc is part of u

    Parameters
    ----------
    y_ind :list, optional
        Indices within y of the variables for the objective function.

    Returns
    -------
    xmes_opt : np array
        Solution of LF variables of OF. Unscaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_times : float
        List with time of the optimization (excluding creation of network etc.).
    """
    if scale_var == 'per_unit':
        raise NotImplementedError('OF not implemented for per unit scaling.')
    print('\nRunning OF for GE2N with DD LF as subsystem (formulations: {}, hard bounds: {}, method: {}, scaling: {}, approach: {})'.format(formulation,stay_within_bounds,optimization_method,scale_var,approach))

    # create networks
    with HiddenPrints():
        gas_net = MES.create_gas_network(scale_var=scale_var,scale_var_params=scale_var_params)
        elec_net = MES.create_electrical_network(scale_var=scale_var,scale_var_params=scale_var_params)
        coupling_net = MES.create_coupling_network(scale_var=scale_var,scale_var_params=scale_var_params)

    # set initial guess and initialize network, using default initial guess
    if len(xg_init):
        gas_net.initialize()
        gas_net.update(xg_init[:2],formulation=formulation.get('gas'))
    else:
        xg_init = MES.set_xg_LF_init(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    if len(xe_init):
        elec_net.initialize()
        elec_net.update(xe_init[:1],formulation=formulation.get('elec'))
    else:
        xe_init = MES.set_xe_LF_init(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    coupling_net.initialize()
    coupling_net.update(u_init[2:],formulation=formulation)
    nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    len_xc = 0#len(nlsysc.x_entries)

    # update BCs
    gas_net,elec_net,coupling_net = update_bc_dd(gas_net,elec_net,coupling_net,u_init,xg_init,xe_init,np.array([]),scale_var=scale_var,scale_var_params=scale_var_params)

    # initial guess for OF (unscaled)
    u0 = u_init
    x_init = np.concatenate((xg_init,xe_init))
    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dxg = nlsysg.Dx()
        DFg = nlsysg.DF()
        Gg_base = np.array([scale_var_params.get('qbase')])
        Dxe = nlsyse.Dx()
        DFe = nlsyse.DF()
        len_slack_e = len(xe_init)-Dxe.shape[0]
        Ge_base = scale_var_params.get('Sbase')*np.ones(len_slack_e)
        Dxc = nlsysc.Dx()
        DFc = nlsysc.DF()
        ubase = np.array([scale_var_params.get('qbase'),scale_var_params.get('Vbase')])
        slackg_base = np.array([scale_var_params.get('qbase')])
        slacke_base = scale_var_params.get('Sbase')*np.ones(len_slack_e)
        Dy = np.diag(np.concatenate((1/ubase,Dxc.data[0],Dxg.data[0],1/slackg_base,Dxe.data[0],1/slacke_base)))
        Dy_inv = np.diag(1/np.diag(Dy))
        Dx = np.diag(np.concatenate((Dxg.data[0],1/slackg_base,Dxe.data[0],1/slacke_base)))
        Dx_inv = np.diag(1/np.diag(Dx))
        Dh_full = np.diag(np.concatenate((DFg.data[0],1/Gg_base,DFe.data[0],1/Ge_base,DFc.data[0])))
        Dh = DFc.copy()
        Du = np.diag(np.concatenate((1/ubase,Dxc.data[0])))
        Du_inv = np.diag(1/np.diag(Du))
        u0 = Du.dot(u0) # scale u
        if scale_var == 'per_unit':
            x_init = Dx.dot(x_init) # scale x
    else:
        Dy=np.eye(len(u0)+len(x_init))
        Dy_inv=np.eye(len(u0)+len(x_init))
        Dx=np.eye(len(x_init))
        Dx_inv=np.eye(len(x_init))
        Du = np.eye(len(u0))
        Du_inv= np.eye(len(u0))
        Dh=np.eye(len(nlsysc.x_entries))
        Dh_full = np.eye(len(x_init)+len(nlsysc.x_entries))

    if scale_var == 'per_unit':
        a = a/fb
        b = b/(fb/np.diag(Dy_inv)[np.array(y_ind)])
        c = c/(fb/np.diag(Dy_inv)[np.array(y_ind)]**2)

    # limits on inequality constraints on state variables
    lb_ineq_state = np.concatenate((xg_lb,xe_lb))
    ub_ineq_state = np.concatenate((xg_ub,xe_ub))
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        lb_ineq_state = Dx.dot(lb_ineq_state)
        ub_ineq_state = Dx.dot(ub_ineq_state)
        u_lb = Du.dot(u_lb)
        u_ub = Du.dot(u_ub)

    # define objective function
    def obj(u,y_ind=y_ind,a=a,b=b,c=c,nlsysg=nlsysg,nlsyse=nlsyse,nlsysc=nlsysc,max_iters_lf=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init,Du_inv=Du_inv,fb=fb):
        global u_f_vec
        u_f_vec = u.copy()
        global f_vec_global
        if nlsysc.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of_dd(u,nlsysg,nlsyse,nlsysc,max_iters=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init) # unscaled (unless p.u. is used)
        y = np.concatenate((u,x))
        f = objective_function(y,y_ind,a=a,b=b,c=c,scale_var=nlsysc.scale_var,scale_var_params=nlsysc.scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f
    def obj_grad(u,y_ind=y_ind,a=a,b=b,c=c,nlsysg=nlsysg,nlsyse=nlsyse,nlsysc=nlsysc,max_iters_lf=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init,len_xg=len(xg_init), len_xe=len(xe_init), len_xc=len_xc,Du_inv=Du_inv,Dy_inv=Dy_inv,Dh_full=Dh_full,fb=fb,method=approach):
        if nlsysc.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of_dd(u,nlsysg,nlsyse,nlsysc,max_iters=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init) # unscaled (unless p.u. is used)
        y = np.concatenate((u,x))
        len_u = len(u)
        len_x = len(x)
        # partial derivatives of objective
        deltaf_deltay = grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=nlsysc.scale_var,scale_var_params=nlsysc.scale_var_params,fb=fb,Dy_inv=Dy_inv,reorder=False)
        deltaf_deltau = np.zeros((1,len_u))
        deltaf_deltax = np.zeros((1,len_x))
        deltaf_deltau[0,:] = deltaf_deltay[:len_u]
        deltaf_deltax[0,:] = deltaf_deltay[len_u:]
        # partial derivatives of equatilty constraints / load-flow equations
        network_g = nlsysg.gasnetwork
        network_e = nlsyse.elecnetwork
        network_c = nlsysc.hetnetwork
        u,xg,xe,xc = xdd_from_yopf(y,len_xg=len_xg, len_xe=len_xe, len_xc=len_xc)
        network_g,network_e,network_c = update_bc_dd(network_g,network_e,network_c,u,xg,xe,xc,scale_var=nlsysc.scale_var,scale_var_params=nlsysc.scale_var_params)
        network_e = MES.reset_network_without_update_elec(network_e)
        deltahfull_deltay = h_der_dd(y,nlsysg=nlsysg,nlsyse=nlsyse,nlsysc=nlsysc,len_xg=len_xg,len_xe=len_xe,len_xc=len_xc,Dh=Dh_full,Dy_inv=Dy_inv)
        len_Fc = len(nlsysc.F_entries)
        deltah_deltau = deltahfull_deltay[:-len_Fc,:][:,:len_u]
        deltah_deltax = deltahfull_deltay[:-len_Fc,:][:,len_u:]
        # print('dh_du={}\ndh_dx={}'.format(deltah_deltau,deltah_deltax))
        # plt.figure('Jacobian deltah_deltau')
        # plt.spy(deltah_deltau)
        # plt.figure('Jacobian values deltah_deltau')
        # jac_val = deltah_deltau.copy()
        # jac_val[jac_val == 0] = np.nan
        # plt.imshow(jac_val)
        # if len_xc > 0:
        #     plt.plot([len_u-.5,len_u-.5],[0-.5,len_h-.5],'k')
        #     plt.plot([len_u+len_xg-.5,len_u+len_xg-.5],[0-.5,len_h-.5],'k')
        #     plt.plot([len_u+len_xg+len_xe-.5,len_u+len_xg+len_xe-.5],[0-.5,len_h-.5],'k')
        # else:
        #     plt.plot([len_u-.5,len_u-.5],[0-.5,len_h-.5],'k')
        # plt.plot([0-.5,len(y)-.5],[len(Fg_ind)-.5,len(Fg_ind)-.5],':k')
        # plt.plot([0-.5,len(y)-.5],[dhg_dxg.shape[0]-.5,dhg_dxg.shape[0]-.5],'k')
        # plt.plot([0-.5,len(y)-.5],[len(Fe_ind)+dhg_dxg.shape[0]-.5,len(Fe_ind)+dhg_dxg.shape[0]-.5],':k')
        # plt.plot([0-.5,len(y)-.5],[dhg_dxg.shape[0]+dhe_dxe.shape[0]-.5,dhg_dxg.shape[0]+dhe_dxe.shape[0]-.5],'k')
        # plt.show()
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

    # define nonlinear equality constraints (load flow equations)
    def eq_constr(u,nlsysg=nlsysg, nlsyse=nlsyse, nlsysc=nlsysc, max_iters_lf=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init,len_xg=len(xg_init), len_xe=len(xe_init), len_xc=len_xc,Du_inv=Du_inv,Dy_inv=Dy_inv, Dh=Dh):
        if nlsysc.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of_dd(u,nlsysg,nlsyse,nlsysc,max_iters=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init) # unscaled (unless p.u. is used)
        y = np.concatenate((u,x))
        # update bc of the networks
        network_g = nlsysg.gasnetwork
        network_e = nlsyse.elecnetwork
        network_c = nlsysc.hetnetwork
        u,xg,xe,xc = xdd_from_yopf(y,len_xg=len_xg, len_xe=len_xe, len_xc=len_xc)
        network_g,network_e,network_c = update_bc_dd(network_g,network_e,network_c,u,xg,xe,xc,scale_var=nlsysc.scale_var,scale_var_params=nlsysc.scale_var_params)
        hc = hc_dd(u, nlsysc=nlsysc,Dh=Dh)
        global err_LF_vecc_global
        err_LFc = np.linalg.norm(hc)
        err_LF_vecc_global.append(np.linalg.norm(err_LFc))
        global err_LF_vecg_global
        global err_LF_vece_global
        global err_LF_vec_global
        err_LF_vec_global.append(max(err_LF_vecg_global[-1],err_LF_vece_global[-1],err_LFc))
        return hc
    def jac_eq_constr(u,nlsysg=nlsysg, nlsyse=nlsyse, nlsysc=nlsysc, max_iters_lf=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init,len_xg=len(xg_init), len_xe=len(xe_init), len_xc=len_xc,Du_inv=Du_inv,Dy_inv=Dy_inv,Dx_inv=Dx_inv, Dh=Dh,Dh_full=Dh_full,method=approach):
        if nlsysc.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of_dd(u,nlsysg,nlsyse,nlsysc,max_iters=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init) # unscaled (unless p.u. is used)
        y = np.concatenate((u,x))
        len_u = len(u)
        # update bc of the networks
        network_g = nlsysg.gasnetwork
        network_e = nlsyse.elecnetwork
        network_c = nlsysc.hetnetwork
        u,xg,xe,xc = xdd_from_yopf(y,len_xg=len_xg, len_xe=len_xe, len_xc=len_xc)
        network_g,network_e,network_c = update_bc_dd(network_g,network_e,network_c,u,xg,xe,xc,scale_var=nlsysc.scale_var,scale_var_params=nlsysc.scale_var_params)
        deltahc_deltau = hc_deru_dd(u,nlsysc=nlsysc,Du_inv=Du_inv, Dh=Dh)
        deltahc_deltax = hc_derx_dd(y,nlsysg=nlsysg,nlsyse=nlsyse,nlsysc=nlsysc,Dx_inv=Dx_inv, Dh=Dh)
        # plt.figure('Jacobian values deltahc_deltax')
        # jac_val = deltahc_deltax.copy()
        # jac_val[jac_val == 0] = np.nan
        # plt.imshow(jac_val)
        # plt.figure('Jacobian values deltahc_deltau')
        # jac_val = deltahc_deltau.copy()
        # jac_val[jac_val == 0] = np.nan
        # plt.imshow(jac_val)
        # partial derivatives of equatilty constraints / load-flow equations
        network_e = MES.reset_network_without_update_elec(network_e)
        deltahfull_deltay = h_der_dd(y,nlsysg=nlsysg,nlsyse=nlsyse,nlsysc=nlsysc,len_xg=len_xg,len_xe=len_xe,len_xc=len_xc,Dh=Dh_full,Dy_inv=Dy_inv)
        len_Fc = len(nlsysc.F_entries)
        deltah_deltau = deltahfull_deltay[:-len_Fc,:][:,:len_u]
        deltah_deltax = deltahfull_deltay[:-len_Fc,:][:,len_u:]
        # plt.figure('Jacobian values deltah_deltax')
        # jac_val = deltah_deltax.copy()
        # jac_val[jac_val == 0] = np.nan
        # plt.imshow(jac_val)
        # plt.figure('Jacobian values deltah_deltau')
        # jac_val = deltah_deltau.copy()
        # jac_val[jac_val == 0] = np.nan
        # plt.imshow(jac_val)
        # gradient eq. constraints
        dhc_du = deltahc_deltau.copy() # first part of gradient
        if method == 'direct':
            w = np.linalg.solve(deltah_deltax,-deltah_deltau)
            dhc_du += np.dot(deltahc_deltax,w)
        elif method == 'adjoint':
            v = np.linalg.solve(np.transpose(deltah_deltax),np.transpose(deltahc_deltax))
            dhc_du += np.dot(np.transpose(v),-deltah_deltau)
        return dhc_du
    lb_nleq = np.zeros(Dh.shape[0])
    ub_nleq = np.zeros(Dh.shape[0])
    if optimization_method == 'trust-constr':
        nonlinear_constraint = spo.NonlinearConstraint(eq_constr,lb_nleq,ub_nleq,jac=jac_eq_constr,keep_feasible=stay_within_bounds)
    elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        nonlinear_constraint = {'type':'eq','fun':eq_constr,'jac':jac_eq_constr}

    # define (non)linear inequality constraints on state variables
    def g(u,nlsysg=nlsysg, nlsyse=nlsyse, nlsysc=nlsysc, lb_ineq_state=lb_ineq_state, ub_ineq_state=ub_ineq_state, Dx=Dx, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init):
        if nlsysc.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of_dd(u,nlsysg,nlsyse,nlsysc,max_iters=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init) # unscaled (unless p.u. is used)
        if scale_var == 'matrix': # lb_ineq_state and ub_ineq_state are scaled, so scale x as well
            x = Dx.dot(x)
        g = np.concatenate((x-lb_ineq_state,ub_ineq_state-x))
        return g
    def g_jac(u,nlsysg=nlsysg, nlsyse=nlsyse, nlsysc=nlsysc, lb_ineq_state=lb_ineq_state, ub_ineq_state=ub_ineq_state, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init,len_xg=len(xg_init), len_xe=len(xe_init),len_xc=len_xc,method=approach):
        if nlsysc.scale_var == 'matrix':
            # Optimizer uses scaled u, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        x = solve_lf_in_of_dd(u,nlsysg,nlsyse,nlsysc,max_iters=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init) # unscaled (unless p.u. is used)
        y = np.concatenate((u,x))
        len_u = len(u)
        # Jacobian of inequality constraints on state variables
        len_x = len(x)
        I = np.eye(len_x)
        deltag_deltax = np.vstack((I,-I))
        deltag_deltau = np.zeros((deltag_deltax.shape[0],len(u)))
        # partial derivatives of equatilty constraints / load-flow equations
        u,xg,xe,xc = xdd_from_yopf(y,len_xg=len_xg, len_xe=len_xe, len_xc=len_xc)
        network_g = nlsysg.gasnetwork
        network_e = nlsyse.elecnetwork
        network_c = nlsysc.hetnetwork
        network_g,network_e,network_c = update_bc_dd(network_g,network_e,network_c,u,xg,xe,xc,scale_var=nlsysc.scale_var,scale_var_params=nlsysc.scale_var_params)
        network_e = MES.reset_network_without_update_elec(network_e)
        deltahfull_deltay = h_der_dd(y,nlsysg=nlsysg,nlsyse=nlsyse,nlsysc=nlsysc,len_xg=len_xg,len_xe=len_xe,len_xc=len_xc,Dh=Dh_full,Dy_inv=Dy_inv)
        len_Fc = len(nlsysc.F_entries)
        deltah_deltau = deltahfull_deltay[:-len_Fc,:][:,:len_u]
        deltah_deltax = deltahfull_deltay[:-len_Fc,:][:,len_u:]
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
    global err_LF_vecg_global
    global err_LF_vece_global
    global err_LF_vecc_global
    global err_LF_vec_global
    f_vec_global = list()
    u_f_vec = u0.copy()
    err_LF_vecg_global = list()
    err_LF_vece_global = list()
    err_LF_vecc_global = list()
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
                raise NotImplementedError('OF not implemented for more than one run')
            opf_start_time = time.perf_counter()
            # try:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=[nonlinear_constraint,ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter,'factorization_method':'SVDFactorization'}, bounds=bounds, callback=callback)
                execution_times.append(res.execution_time)
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=[nonlinear_constraint,ineq_constr_fun], options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                execution_times.append(time.perf_counter() - opf_start_time)
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, u0,jac=obj_grad, constraints=[nonlinear_constraint,ineq_constr_fun], options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_times.append(time.perf_counter() - opf_start_time)
            # except:
            #     print('Exception made for {} (formulations: {}, hard bounds: {}, scaling: {}, approach: {})'.format(optimization_method,formulation,stay_within_bounds,scale_var,approach))
            #     if len(f_vec_global) == 0:
            #         obj(u0)
            #         nit = 0
            #         nfev = 0
            #         njev = 0
            #         nhev = 0
            #     else:
            #         nfev = len(f_vec_global)
            #         njev = 0
            #         nhev = 0
            #         if optimization_method == 'ipopt':
            #             nit = 0
            #         else: # append value of iterate to output of the iteration in which the error occured
            #             f_vec.append(f_vec_global[-1])
            #             nit = len(f_vec)
            #     execution_times.append(time.perf_counter() - opf_start_time)
            #     res = spo.OptimizeResult({'success':False,'x':np.array(u_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})
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
    if len(err_LF_vecg_global) >= len(f_vec):
        indices = [int(round(ind)) for ind in np.linspace(0,len(err_LF_vecg_global)-1,len(f_vec))]
        err_LFg_vec = [err_LF_vecg_global[ind] for ind in indices]
    elif len(err_LF_vecg_global) == 0:
        obj(u0)
        err_LFg_vec = [err_LF_vecg_global[-1]]
    else:
        err_LFg_vec = err_LF_vecg_global
    if len(err_LF_vece_global) >= len(f_vec):
        indices = [int(round(ind)) for ind in np.linspace(0,len(err_LF_vece_global)-1,len(f_vec))]
        err_LFe_vec = [err_LF_vece_global[ind] for ind in indices]
    elif len(err_LF_vece_global) == 0:
        obj(u0)
        err_LFe_vec = [err_LF_vece_global[-1]]
    else:
        err_LFe_vec = err_LF_vece_global
    if len(err_LF_vecc_global) >= len(f_vec):
        indices = [int(round(ind)) for ind in np.linspace(0,len(err_LF_vecc_global)-1,len(f_vec))]
        err_LFc_vec = [err_LF_vecc_global[ind] for ind in indices]
    elif len(err_LF_vecc_global) == 0:
        obj(u0)
        err_LFc_vec = [err_LF_vecc_global[-1]]
    else:
        err_LFc_vec = err_LF_vecc_global

    if scale_var == 'matrix' or scale_var == 'per_unit':
        u_opf = Du_inv.dot(res.x)
    else:
        u_opf = res.x

    # print solution
    if scale_var == 'per_unit':
        x = solve_lf_in_of_dd(res.x,nlsysg,nlsyse,nlsysc,max_iters=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init)
        y_opt = Dy_inv.dot(np.concatenate((res.x,x))) # unscaled
    else:
        x = solve_lf_in_of_dd(u_opf,nlsysg,nlsyse,nlsysc,max_iters=max_iters_lf,tol=tol,xg_init=xg_init,xe_init=xe_init)
        y_opt = np.concatenate((u_opf,x)) # unscaled
    # print solution
    u,xg,xe,xc = xdd_from_yopf(y_opt, len_xg=len(xg_init), len_xe=len(xe_init), len_xc=len_xc)
    gas_net,elec_net,coupling_net = update_bc_dd(gas_net,elec_net,coupling_net,u,xg,xe,xc)
    with HiddenPrints():
        het_net, gas_net_mes, elec_net_mes = MES.create_mes_network()
        het_net.initialize()
    xg_mes = xg[:2]
    xe_mes = np.array([xe[0]])
    q0_c = xg[-1]
    P1_c,Q0_c,Q1_c = xe[1:]
    q0,V0,q1c,Pc0 = u
    qc0,Pc1,Qc0,Qc1 = MES.ic_coupling(q0_c,P1_c,Q0_c,Q1_c)
    xc_mes = np.array([qc0,q1c,Pc0,Pc1,Qc0,Qc1])
    xmes_opt = np.concatenate((xg_mes,xe_mes,xc_mes))
    gas_net_mes,elec_net_mes,het_net = update_bc(gas_net_mes,elec_net_mes,het_net,q0,V0)
    print('Solution OF (success: {})'.format(res.success))
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation)
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
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net_mes.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net_mes.get_nodes() for hl in node.get_half_links()]))
    return xmes_opt, y_opt, res, f_vec, err_LF_vec, err_LFg_vec, err_LFe_vec, err_LFc_vec, execution_times

def compare_forms(dir_path=None,save_tables=False,save_figs=False,save_data=False,save_parameters=False,number_of_runs=1,max_iter=25,scale_var=None):
    """Compare OF for different optimization methods, formulations of OF, solution methods / formulation of LF, and scalings."""
    # solver info
    max_iters_lf = 20
    tol = 1e-6
    formulation={'gas':'full','elec':'complex_power'}

    # scaling
    pbase = MES.pg0
    qbase = .1
    Vbase = MES.Vbase
    deltabase = 1
    Sbase = MES.Sbase
    Ebase = 1*MW
    if scale_var == None:
        scale_var_params = None
        fb = None
        scale_label = 'unscaled'
    else:
        fb = 5e2#1e2
        scale_var_params={'pgbase':pbase,'pbase':pbase,'qbase':qbase,'Vbase':Vbase,'deltabase':deltabase,'Sbase':Sbase,'Ebase':Ebase}
        if scale_var == 'matrix':
            scale_label = 'matrix'
        else:
            scale_label = 'pu'

    # parameter values for the objective function (assumes E = [q0,Pc0,Pc1])
    a=np.array([0,0,0])
    b=np.array([15*MES.GHV,2,3])/MW
    c=np.array([0,0.02,0.03])/MW
    y_ind_int = [0,7,8]
    y_ind_dd1 = [0,10,6]
    y_ind_dd2 = [0,3,8]

    # initial guess
    u_init=np.array([1.02*MES.q0_source,1.01*MES.V0_sol])#np.array([1.01*MES.q0_source,1.01*MES.V0_sol])
    x_init=np.array([MES.q_ic,.99*MES.pg_ic,0,MES.qc_ic,MES.qc_ic,MES.Pc_ic,3*MES.Pc_ic,MES.Qc_ic,MES.Qc_ic])
    xg_init=np.array([MES.q_ic,.99*MES.pg_ic,MES.qc_ic])
    xe_init=np.array([0,-3*MES.Pc_ic,-MES.Qc_ic,-MES.Qc_ic])
    xc_init=np.array([1.2*abs(MES.qc1_sol),1.8*MES.Pc0_sol])#np.array([MES.qc_ic,MES.Pc_ic])

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, gas_net_LF, elec_net_LF = MES.create_mes_network()
        het_net_LF.initialize()
        het_net_LF.update(x_init,formulation=formulation)
        x_sol_LF,iters_LF,err_vec_LF,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net_LF.solve_network(tol,max_iters_lf,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params) # without reordering
    q0 = gas_net_LF.nodes[0].half_links[0].get_q(scale_var=scale_var,scale_var_params=scale_var_params)
    V0 = elec_net_LF.nodes[0].get_V(scale_var=scale_var,scale_var_params=scale_var_params)
    y_LF = np.concatenate((np.array([q0,V0]),x_sol_LF))
    print('err LF = {:.3e} after {} iters'.format(err_vec_LF[-1],iters_LF))

    # bounds
    u_lb=np.array([1.3*MES.q0_source,.95*MES.V0_sol])
    u_ub=np.array([0,1.05*MES.V0_sol])
    x_lb=np.array([-1*abs(1.1*MES.q0_source),0,-np.pi,.5*abs(MES.qc0_sol),.5*abs(MES.qc1_sol),.5*MES.Pc0_sol,.5*MES.Pc1_sol,.5*MES.Qc0_sol,.5*MES.Qc1_sol])
    xg_lb=np.array([-1*abs(1.1*MES.q0_source),0,.5*abs(MES.qc0_sol)])
    xe_lb=np.array([-np.pi,-1.5*MES.Pc1_sol,-1.5*MES.Qc0_sol,-1.5*MES.Qc1_sol])
    xc_lb=np.array([.5*abs(MES.qc1_sol),.5*MES.Pc0_sol])
    x_ub=np.array([abs(1.1*MES.q0_source),1.05*MES.pg0,np.pi,1.5*abs(MES.qc0_sol),1.5*abs(MES.qc1_sol),1.5*MES.Pc0_sol,1.5*MES.Pc1_sol,1.5*MES.Qc0_sol,1.5*MES.Qc1_sol])
    xg_ub=np.array([abs(1.1*MES.q0_source),1.05*MES.pg0,1.5*abs(MES.qc0_sol)])
    xe_ub=np.array([np.pi,-.5*MES.Pc1_sol,-.5*MES.Qc0_sol,-.5*MES.Qc1_sol])
    xc_ub=np.array([1.5*abs(MES.qc1_sol),2*MES.Pc0_sol])#np.array([1.5*abs(MES.qc1_sol),1.5*MES.Pc0_sol])

    # scale LF solution
    if scale_var != None:
        nlsys_LF = NonLinearSystemHeterogeneous(het_net_LF,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        Dx = nlsys_LF.Dx()
        ubase = np.array([scale_var_params.get('qbase'),scale_var_params.get('Vbase')])
        Dy = np.diag(np.concatenate((1/ubase,Dx.data[0])))
        f_LF = objective_function(y_LF,y_ind_int,a=a,b=b,c=c)/fb #scaled
        y_LF = Dy.dot(y_LF) # scaled
    else:
        f_LF = objective_function(y_LF,y_ind_int,a=a,b=b,c=c,scale_var=scale_var,fb=fb)

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
    parameters['scale_var_params'] = scale_var_params
    y_res['LF'] = y_LF.copy()
    err_LF_res['LF'] = err_vec_LF.copy()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft']#, 'hard']
    approaches = ['eq_constr','direct','adjoint']
    forms = ['int. orig.','int. reord.','dd1','dd2']
    total_cases = len(methods)*len(bounds)*len(approaches)*len(forms)
    case_number = 1
    start_wall_clock = datetime.datetime.now().time()
    for bound in bounds:
        if bound == 'soft':
            stay_within_bounds = False
        else:
            stay_within_bounds = True
        for form in forms:
            if form == 'int. reord.':
                reorder = True
            elif form == 'int. orig.':
                reorder = False
            fig_f = plt.figure('mes_obj_'+bound+'_'+form+'_'+scale_label)
            ax_f = fig_f.gca()
            ax_f.set_xlabel('Iteration?')
            ax_f.set_ylabel('f')

            fig_LF_error = plt.figure('mes_error_LF_in_OF_'+bound+'_'+form+'_'+scale_label)
            ax_LF_error = fig_LF_error.gca()
            ax_LF_error.set_xlabel('Iteration?')
            ax_LF_error.set_ylabel(r'$||F||_2$')
            if form == 'dd2' and ('direct' in approaches or 'adjoint' in approaches):
                fig_LFg_error = plt.figure('gas_error_LF_in_OF_'+bound+'_'+form+'_'+scale_label)
                ax_LFg_error = fig_LFg_error.gca()
                ax_LFg_error.set_xlabel('Iteration?')
                ax_LFg_error.set_ylabel(r'$||F^g||_2$')

                fig_LFe_error = plt.figure('elec_error_LF_in_OF_'+bound+'_'+form+'_'+scale_label)
                ax_LFe_error = fig_LFe_error.gca()
                ax_LFe_error.set_xlabel('Iteration?')
                ax_LFe_error.set_ylabel(r'$||F^e||_2$')

                fig_LFc_error = plt.figure('coupling_error_LF_in_OF_'+bound+'_'+form+'_'+scale_label)
                ax_LFc_error = fig_LFc_error.gca()
                ax_LFc_error.set_xlabel('Iteration?')
                ax_LFc_error.set_ylabel(r'$||F^c||_2$')
            max_fev = 0
            for method in methods:
                for approach in approaches:
                    start_runs_wall_clock = datetime.datetime.now().time()
                    if form == 'dd1':
                        y_ind = y_ind_dd1
                        if approach == 'direct' or approach == 'adjoint':
                            case_number += 1
                            continue
                        else:
                            xmes_opt, res, f_vec, err_LF_vec, execution_times = run_optimal_load_flow_dd(u_lb=u_lb,u_ub=u_ub,u_init=u_init,xg_lb=xg_lb,xe_lb=xe_lb,xc_lb=xc_lb,xg_ub=xg_ub,xe_ub=xe_ub,xc_ub=xc_ub,xg_init=xg_init,xe_init=xe_init,xc_init=xc_init,y_ind=y_ind,a=a,b=b,c=c,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,runs=number_of_runs)
                            y_opt = res.x
                    elif form == 'dd2':
                        y_ind = y_ind_dd2
                        u_init_dd = np.concatenate((u_init,xc_init))
                        u_lb_dd = np.concatenate((u_lb,xc_lb))
                        u_ub_dd = np.concatenate((u_ub,xc_ub))
                        xc_init_dd = np.array([])
                        xc_lb_dd = np.array([])
                        xc_ub_dd = np.array([])
                        if approach == 'direct' or approach == 'adjoint':
                            xmes_opt, y_opt, res, f_vec, err_LF_vec, err_LFg_vec, err_LFe_vec, err_LFc_vec, execution_times = run_optimal_load_flow_separate_LF_dd(u_lb=u_lb_dd,u_ub=u_ub_dd,u_init=u_init_dd,xg_lb=xg_lb,xe_lb=xe_lb,xg_ub=xg_ub,xe_ub=xe_ub,xg_init=xg_init,xe_init=xe_init,y_ind=y_ind,a=a,b=b,c=c,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,approach=approach,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,runs=number_of_runs)
                        else:
                            xmes_opt, res, f_vec, err_LF_vec, execution_times = run_optimal_load_flow_dd(u_lb=u_lb_dd,u_ub=u_ub_dd,u_init=u_init_dd,xg_lb=xg_lb,xe_lb=xe_lb,xc_lb=xc_lb_dd,xg_ub=xg_ub,xe_ub=xe_ub,xc_ub=xc_ub_dd,xg_init=xg_init,xe_init=xe_init,xc_init=xc_init_dd,y_ind=y_ind,a=a,b=b,c=c,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,runs=number_of_runs)
                            y_opt = res.x
                    else:
                        y_ind = y_ind_int
                        if approach == 'direct' or approach == 'adjoint':
                            xmes_opt, y_opt, res, f_vec, err_LF_vec, execution_times = run_optimal_load_flow_separate_LF(u_lb=u_lb,u_ub=u_ub,u_init=u_init,x_lb=x_lb,x_ub=x_ub,x_init=x_init,y_ind=y_ind,a=a,b=b,c=c,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,approach=approach,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,runs=number_of_runs,reorder=reorder)
                        else:
                            xmes_opt, res, f_vec, err_LF_vec, execution_times = run_optimal_load_flow(u_lb=u_lb,u_ub=u_ub,u_init=u_init,x_lb=x_lb,x_ub=x_ub,x_init=x_init,y_ind=y_ind,a=a,b=b,c=c,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,runs=number_of_runs,reorder=reorder)
                            y_opt = res.x
                    print('Finished case {} of {}, in average of {:.2f}s per run. Started at {}, started these runs at {}'.format(case_number,total_cases,res.execution_time,start_wall_clock,start_runs_wall_clock))
                    case_number += 1
                    result[method+'_'+form+'_'+bound+'_'+approach] = res
                    y_res[method+'_'+form+'_'+bound+'_'+approach] = y_opt # unscaled for direct and adjoint approach. Scaled with eq_constr approach!!!!
                    err_LF_res[method+'_'+form+'_'+bound+'_'+approach] = err_LF_vec
                    f_vec_res[method+'_'+form+'_'+bound+'_'+approach] = f_vec
                    max_fev = max(max_fev,len(f_vec))
                    # plot results
                    ax_f.plot(f_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form),alpha=.5)
                    ax_LF_error.semilogy(err_LF_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form),alpha=.5)
                    if form == 'dd2' and (approach == 'direct' or approach == 'adjoint'):
                        ax_LFg_error.semilogy(err_LFg_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form),alpha=.5)
                        ax_LFe_error.semilogy(err_LFe_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form),alpha=.5)
                        ax_LFc_error.semilogy(err_LFc_vec,color=colors_method.get(method),ls=linestyles_approaches.get(approach),marker=markers_forms.get(form),alpha=.5)
            ax_f.plot([0,max_fev],[f_LF,f_LF],':r',alpha=.5)
            ax_LF_error.semilogy([0,max_fev],[tol,tol],':k')
            if form == 'dd2' and ('direct' in approaches or 'adjoint' in approaches):
                ax_LFg_error.semilogy([0,max_fev],[tol,tol],':k')
                ax_LFe_error.semilogy([0,max_fev],[tol,tol],':k')
                ax_LFc_error.semilogy([0,max_fev],[tol,tol],':k')
            if not save_figs:
                ax_f.legend(handles=legend_handles)
                ax_LF_error.legend(handles=legend_handles)
                if form == 'dd2' and ('direct' in approaches or 'adjoint' in approaches):
                    ax_LFg_error.legend(handles=legend_handles)
                    ax_LFe_error.legend(handles=legend_handles)
                    ax_LFc_error.legend(handles=legend_handles)

if __name__ == '__main__':
    max_iter = 30#50
    runs = 1
    # unscaled
    # compare_forms(number_of_runs=runs,max_iter=max_iter,scale_var=None)
    # matrix scaling
    compare_forms(number_of_runs=runs,max_iter=max_iter,scale_var='matrix')

    plt.show()
