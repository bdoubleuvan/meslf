"""Example of a MES with 3 nodes per carrier. Based on the example in Shabanpourt-Haghighi and Seifi. Uses energy hubs for the coupling nodes. Optimal flow."""
from examples import MES_Shabanpour_EH as MES
import warnings
import numpy as np
from meslf.utils.constants import mm, bar, hour, kV, MW, MSCM, BTU, MBTU
from meslf.utils.hide_print import HiddenPrints
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
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

variable_names_u = [r'$|V_2|',r'$T^s_{2^c2^h}']
variable_names = variable_names_u +  [r'$q_{0,0}$',r'$q_{01}$',r'$q_{02}$',r'$q_{32}$',r'$q_{13}$',r'$p^g_{1}$',r'$p^g_2',r'$p^g_{3}$',r'$\delta_1$',r'$\delta_2$',r'$|V_1|$',r'$m_{01}$',r'$m_{02}$',r'$m_{12}$',r'$m_{1}$',r'$m_{2}$',r'$p^h_1$',r'$p^h_2',r'$T^s_0$',r'$T^s_1$',r'$T^s_2$',r'$T^r_0$',r'$T^r_1$',r'$T^r_2$',r'$q_{0^g0^c}$',r'$q_{2^g1^c}$',r'$P_{0^c0^e}$',r'$P_{1^c2^e}$',r'$Q_{0^c0^e}$',r'$Q_{1^c2^e}$',r'$m_{0^c0^h}$',r'$m_{1^c2^h}$',r'$-\Delta\varphi_{1^c0^h}$',r'$-\Delta\varphi_{2^c2^h}$']

def xmes_from_yopf(y):
    """Returns the variables of steady-state LF, from the variables y of OF."""
    xmes = y[3:]
    return xmes

def objective_function(y,y_ind,a=np.array([0,0,0,0,0]),b=np.array([.01*MES.GHV,.3,.3,.04,.05]),c=np.array([1e-6*(MES.GHV)**2,3e-5,2e-5,4e-4,4.5e-4]),scale_var=None,scale_var_params=None,fb=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    E : np array
        Array with flows used in objective. Gas flows are assumed to be in kg/s, active powers in W, and heat power in W. Scaled when per unit scaling is used, unscaled otherwise.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [q,P,P,phi,phi]. Scaled when per unit scaling is used, unscaled otherwise.
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

def grad_objective(y,y_ind,a=np.array([0,0,0,0,0]),b=np.array([.01*MES.GHV,.3,.3,.04,.05]),c=np.array([1e-6*(MES.GHV)**2,3e-5,2e-5,4e-4,4.5e-4]),scale_var=None,scale_var_params=None,fb=None,Dy_inv=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [q,P,P,phi,phi]. Scaled when per unit scaling is used, unscaled otherwise.
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

def hess_objective(y,y_ind,a=np.array([0,0,0,0,0]),b=np.array([.01*MES.GHV,.3,.3,.04,.05]),c=np.array([1e-6*(MES.GHV)**2,3e-5,2e-5,4e-4,4.5e-4]),scale_var=None,scale_var_params=None,fb=None,Dy_inv=None):
    """Objective function for optimal dispatch problem

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    y_ind : list
        Array with indices in y of the flows used in objective function.
    a, b, c : np array
        Arrays with parameter values for the objective. The defaults assume E to be [q,P,P,phi,phi]. Scaled when per unit scaling is used, unscaled otherwise.
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
        Nonlinear system of the heterogenous network. Constains the networks, information about scaling, etc.

    Returns
    -------
    h : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    xmes = xmes_from_yopf(y)
    network_mes = nlsys.hetnetwork
    network_mes.reset_network(xmes,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,formulation=nlsys.formulation)
    # evaluate load flow equations
    F = nlsys.F(xmes)
    # evaluate conservation of mass in gas slack node
    q0 = y[2]
    network_g = nlsys.nlsystemsg[0].gasnetwork
    if nlsys.scale_var == 'per_unit':
        network_g.nodes[0].half_links[0].q = q0*nlsys.scale_var_params.get('qbase')
    else:
        network_g.nodes[0].half_links[0].q = q0
    cons_mass = network_g.nodes[0].node_law(network=network_g,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)
    h = np.concatenate((np.array([cons_mass]),F)) # already scaled if per unit is used
    if nlsys.scale_var == 'matrix':
        h = Dh.dot(h)
    return h

def h_der(y,nlsys=None, Dy_inv=None, Dh=None):
    """Fiurst derivative of equality constraints h(x)=0. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Array with all variables of OF. Scaled when per unit scaling is used, unscaled otherwise.
    nlsys : NonLinearSystemHeterogeneous
        Nonlinear system of the heterogenous network. Constains the networks, information about scaling, etc.

    Returns
    -------
    dh_dy : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    # update networks
    xmes = xmes_from_yopf(y)
    network_mes = nlsys.hetnetwork
    network_mes.reset_network(xmes,scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params,formulation=nlsys.formulation)
    # Jacobian of LF equations
    J_lf = nlsys.J_dense(xmes)
    dh_dy = np.zeros((Dh.shape[0],len(y)))
    dh_dy[1:,3:] = J_lf #dF_dxlf
    # derivative to amplitude of control voltage
    nlsyse = nlsys.nlsystemse[0]
    network_e = nlsyse.elecnetwork
    xe = xmes[len(nlsys.xg_entries):len(nlsys.xg_entries)+len(nlsys.xe_entries)]
    Je_full = nlsys.nlsystemse[0].J_dense(xe,return_full=True)
    Ne = len(network_e.nodes)
    V2_ind = 2*Ne-1
    Fe_ind = nlsyse.FP + [Ne+ind for ind in nlsyse.FQ]
    dh_dy[1+len(nlsys.Fg_entries):1+len(nlsys.Fg_entries)+len(Fe_ind),0] = Je_full[Fe_ind,V2_ind].ravel() # dFe_dV2
    # derivatives of slack equation in gas node
    dh_dy[0,[2,3,4,3+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.xh_entries)]] = -1 #dG_dy
    # derivatives to control temperature
    m1c_ind = 3+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.xh_entries)+7 # index of m1c within y
    m1c = y[m1c_ind]
    dh_dy[-1,1] = MES.water.get_Cp(scale_var=nlsys.scale_var,scale_var_params=nlsys.scale_var_params)*m1c #dFdphi1c_dTs1c
    if m1c > 0:
        dh_dy[1+len(nlsys.Fg_entries)+len(nlsys.Fe_entries)+len(nlsys.F_m_nodes)+len(nlsys.F_deltap_links)+2,1] = -m1c #dFTs2_dTs1c
    if nlsys.scale_var == 'matrix':
        dh_dy = Dh.dot(dh_dy.dot(Dy_inv))
    return dh_dy

def update_bc(het_net, gas_net, elec_net, heat_net, V2, Ts1c, scale_var=None,scale_var_params=None):
    """Update the boundary conditions of the MES, based on the control variables of OF"""
    if scale_var == 'per_unit':
        V2 = V2*scale_var_params.get('Vbase')
        Ts1c = Ts1c*scale_var_params.get('Tbase')
    elec_net.nodes[2].V = V2
    het_net.nodes[6].V = V2
    heat_net.links[-1].Tsstart = Ts1c
    heat_net.links[-1].Tsend = Ts1c
    return het_net, gas_net, elec_net, heat_net

def run_optimal_load_flow(u_lb=np.array([.8*MES.Vbase_shabanpour,100]),u_ub=np.array([1.2*MES.Vbase_shabanpour,135]),u_init=np.array([1.1*MES.Vbase_shabanpour,130]),slack_lb=np.array([-80*MSCM*MES.rhon_g/hour]),slack_ub=np.array([0]),slack_init=np.array([-50*MSCM*MES.rhon_g/hour]),xLF_lb=np.array([-80*MSCM*MES.rhon_g/hour,-80*MSCM*MES.rhon_g/hour,-80*MSCM*MES.rhon_g/hour,-80*MSCM*MES.rhon_g/hour,1*bar,1*bar,1*bar,-np.pi,-np.pi,0.8*MES.Vbase_shabanpour,-100,-100,-100,0,0,1*MES.rho_w*MES.grav_const,1*MES.rho_w*MES.grav_const,60,60,60,0,0,0,0,0,0,0,-50*MES.Sbase_shabanpour,-50*MES.Sbase_shabanpour,0,0,0,0]),xLF_ub=np.array([80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,60*bar,60*bar,60*bar,np.pi,np.pi,1.2*MES.Vbase_shabanpour,100,100,100,200,200,6000*MES.rho_w*MES.grav_const,6000*MES.rho_w*MES.grav_const,135,135,135,60,60,60,80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,60*MES.Sbase_shabanpour,60*MES.Sbase_shabanpour,60*MES.Sbase_shabanpour,60*MES.Sbase_shabanpour,150,150,35*MW,35*MW]),xLF_init=np.array([20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,45*bar,47*bar,45*bar,0,0,1*MES.Vbase_shabanpour,65,30,-60,20,20,200*MES.rho_w*MES.grav_const,4300*MES.rho_w*MES.grav_const,100,120,120,50,50,50,10*MSCM*MES.rhon_g/hour,3*MSCM*MES.rhon_g/hour,50*MES.Sbase_shabanpour,10*MES.Sbase_shabanpour,0*MES.Sbase_shabanpour,0*MES.Sbase_shabanpour,10,10,30*MW,25*MW]),a=np.array([0,0,0,0,0]),b=np.array([.01*MES.GHV,.3,.3,.04,.05]),c=np.array([1e-6*(MES.GHV)**2,3e-5,2e-5,4e-4,4.5e-4]),scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None},ineq_constr='control',derivatives=False,optimization_method='trust-constr',stay_within_bounds=False,fb=None,obj_fun='combined'):
    """Run optimal load flow, with LF as equality constraints.

    Parameters
    ----------
    ineq_contrs : str, optional
        Determines on which variables the bounds (or inequality constraints) are imposed. Options are 'control' or 'all'. Default is 'control'

    Returns
    -------
    xmes_opt : np array
        Solution of LF variables of OF. Unscaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_time : float
        Time of the optimization (excluding creation of network etc.)
    """
    if formulation.get('gas') != 'full':
        raise ValueError('OF not implemented for gas-formulation other than full')
    if formulation.get('heat') != 'half_link_flow':
        raise ValueError('OF not implemented for heat-formulation other than half_link_flow')
    print('\nRunning OPF for MES EH network (inequality constraints on: {}, analytical derivatives: {}, hard bounds: {}, method: {}, scaling:{})'.format(ineq_constr, derivatives,stay_within_bounds,optimization_method,scale_var))

    # create network
    with HiddenPrints():
        het_net, gas_net, elec_net, heat_net, _, _ = MES.create_network(hydr_eq='fa',heat_load='outflow',node_set=2) # Ts of both coupling node 'known'

    # update the BC of the MES to match initial guess of OF
    V2_init, Ts1c_init = u_init
    het_net, gas_net, elec_net, heat_net = update_bc(het_net, gas_net, elec_net, heat_net, V2_init, Ts1c_init, scale_var=scale_var,scale_var_params=scale_var_params)
    # set intial guess and initialize network
    het_net.initialize()
    het_net.update(xLF_init,formulation=formulation)
    het_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # initial guess for OF (unscaled)
    y0 = np.concatenate((u_init,slack_init,xLF_init))

    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # indices within y of values to be used in objective function
    if obj_fun == 'combined':
        xc_start_ind = len(u_init)+len(slack_init)+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.xh_entries)
        y_ind = [len(u_init),xc_start_ind+2,xc_start_ind+3,xc_start_ind+8,xc_start_ind+9]
    elif obj_fun == 'gas':
        y_ind = [len(u_init)]
    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('Vbase'),scale_var_params.get('Tbase')])
        slack_base = np.array([scale_var_params.get('qbase')])
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((ubase,slack_base,1/Dx.data[0])))
        Dh = np.diag(np.concatenate((np.array([1/scale_var_params.get('qbase')]),DF.data[0])))
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
    def obj(y,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params, fb=fb, formulation=formulation):
        global y_f_vec
        y_f_vec = y.copy()
        global f_vec_global
        if scale_var == 'matrix':
            # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
            y = Dy_inv.dot(y)
        f = objective_function(y,y_ind,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    def obj_grad(y,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params, fb=fb, formulation=formulation):
        if scale_var == 'matrix':
            y = Dy_inv.dot(y)
        return grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb,Dy_inv=Dy_inv)

    def obj_hess(y,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params, fb=fb, formulation=formulation):
        if scale_var == 'matrix':
            y = Dy_inv.dot(y)
        return hess_objective(y,y_ind,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb,Dy_inv=Dy_inv)

    # define nonlinear equality constraints (load flow equations)
    def eq_constr(y,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params, Dy_inv=Dy_inv, Dh=Dh, formulation=formulation):
        network_mes = nlsys.hetnetwork
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_h = nlsys.nlsystemsh[0].heatnetwork
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        V2, Ts1c = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network_mes, network_g, network_e, network_h = update_bc(network_mes, network_g, network_e, network_h, V2, Ts1c, scale_var=scale_var,scale_var_params=scale_var_params)
        H = h(y, nlsys=nlsys, Dh=Dh)
        global err_LF_vec_global
        F = H[len(slack_init):]
        err_LF_vec_global.append(np.linalg.norm(F))
        return H

    def jac_eq_constr(y,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params, Dy_inv=Dy_inv, Dh=Dh, formulation=formulation):
        network_mes = nlsys.hetnetwork
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_h = nlsys.nlsystemsh[0].heatnetwork
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        V2, Ts1c = y[:len(u_init)] # scaled for p.u., unscaled for matrix
        network_mes, network_g, network_e, network_h = update_bc(network_mes, network_g, network_e, network_h, V2, Ts1c, scale_var=scale_var,scale_var_params=scale_var_params)
        nlsys.nlsystemse[0].V_vec_mag[2] = V2
        dh_dy = h_der(y, nlsys=nlsys, Dy_inv=Dy_inv, Dh=Dh)
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

    # define linear inequality constraints, i.e. define bounds
    lb_ineq = np.concatenate((u_lb,slack_lb,xLF_lb))
    ub_ineq = np.concatenate((u_ub,slack_ub,xLF_ub))
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        lb_ineq = Dy.dot(lb_ineq)
        ub_ineq = Dy.dot(ub_ineq)
    if ineq_constr == 'control':
        lb_ineq[len(u_lb):] = -np.inf*np.ones(len(slack_lb)+len(xLF_lb)) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
        ub_ineq[len(u_ub):] = np.inf*np.ones(len(slack_ub)+len(xLF_ub))

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
    if ineq_constr != None and (optimization_method == 'SLSQP' or stay_within_bounds):
        for ind, x0 in enumerate(y0):
            if lb_ineq[ind] > x0:
                x_opf0[ind] = lb_ineq[ind]
            elif ub_ineq[ind] < x0:
                x_opf0[ind] = ub_ineq[ind]

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
    opf_start_time = time.time()
    try:
        if derivatives:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, y0, method=optimization_method,jac=obj_grad,hess=obj_hess, constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol, callback=callback)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, y0, method=optimization_method,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, y0,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
        else:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, y0, method=optimization_method, constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol, callback=callback)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, y0, method=optimization_method, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol,callback=callback)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, y0, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
    except:
        print('Exception made for objective {}, {}, ineq. constr: {}, hard bounds: {}, analytical der.: {}, scaling: {}'.format(obj_fun,optimization_method,ineq_constr,stay_within_bounds,derivatives,scale_var))
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
        execution_time = opf_start_time - time.time()
        res = spo.OptimizeResult({'success':False,'x':np.array(y_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

    if optimization_method == 'ipopt':
        if res.nit > 0:
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            f_vec = f_vec_global

    if len(err_LF_vec_global) > len(f_vec):
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
    V2, Ts1c = y_opf[:len(u_init)] # unscaled
    het_net, gas_net, elec_net, heat_net = update_bc(het_net, gas_net, elec_net, heat_net, V2, Ts1c)
    xmes_opt = xmes_from_yopf(y_opf) # unscaled
    het_net.reset_network(xmes_opt,formulation=formulation)
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation)
    print('Solution OF (success: {})'.format(res.success))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
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
    return xmes_opt, res, f_vec, err_LF_vec, execution_time

def get_u_from_net(het_net, gas_net, elec_net, heat_net,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}, scale_var=None,scale_var_params=None):
    """Get the current values of the control variables u from the network

    Returns
    -------
    u : np array
        Vector with control variables. Scaled when using per unit scaling, unscaled otherwise
    """
    V2 = elec_net.nodes[2].get_V(scale_var=scale_var,scale_var_params=scale_var_params)
    Ts1c = heat_net.links[-1].get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params)
    return np.array([V2,Ts1c])

def solve_lf_in_of(het_net, gas_net, elec_net, heat_net, u, max_iters=10,tol=1e-6,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}, scale_var=None,scale_var_params=None, xLF_init=np.array([20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,45*bar,47*bar,45*bar,0,0,1*MES.Vbase_shabanpour,65,30,-60,20,20,200*MES.rho_w*MES.grav_const,4300*MES.rho_w*MES.grav_const,100,120,120,50,50,50,10*MSCM*MES.rhon_g/hour,3*MSCM*MES.rhon_g/hour,50*MES.Sbase_shabanpour,10*MES.Sbase_shabanpour,0*MES.Sbase_shabanpour,0*MES.Sbase_shabanpour,10,10,30*MW,25*MW]),):
    """Solve steady-state LF within an optmization context.

    Parameters
    ----------
    u : np arrays
        Vector with control variables. Scaled when using per unit scaling, unscaled otherwise
    """
    global err_LF_vec_global
    u_net = get_u_from_net(het_net, gas_net, elec_net, heat_net,formulation=formulation, scale_var=scale_var,scale_var_params=scale_var_params)
    if len(err_LF_vec_global)==0 or (np.linalg.norm(u-u_net) > tol) or (err_LF_vec_global[-1] > tol):
        V2, Ts1c = u
        het_net, gas_net, elec_net, heat_net = update_bc(het_net, gas_net, elec_net, heat_net, V2, Ts1c, scale_var=scale_var,scale_var_params=scale_var_params)
        xmes0 = het_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        het_net.reset_network(xmes0,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        with HiddenPrints():
            xmes,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iters,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        if err_vec[-1] >= tol:
            het_net.reset_network(xLF_init,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            xmes,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iters,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        err_LF = err_vec[-1]
        q0 = q_inj[0]
        if scale_var == 'per_unit':
            q0 = q0/scale_var_params.get('qbase')
    else:
        xmes = het_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        q0 = gas_net.nodes[0].half_links[0].get_q(scale_var=scale_var,scale_var_params=scale_var_params)
        err_LF = err_LF_vec_global[-1]
    err_LF_vec_global.append(err_LF)
    x = np.concatenate((np.array([q0]),xmes)) #Scaled when using per unit scaling, unscaled otherwise
    return x, het_net, gas_net, elec_net, heat_net
def run_optimal_load_flow_separate_LF(u_lb=np.array([.8*MES.Vbase_shabanpour,100]),u_ub=np.array([1.2*MES.Vbase_shabanpour,135]),u_init=np.array([1.1*MES.Vbase_shabanpour,130]),slack_lb=np.array([-80*MSCM*MES.rhon_g/hour]),slack_ub=np.array([0]),slack_init=np.array([-50*MSCM*MES.rhon_g/hour]),xLF_lb=np.array([-80*MSCM*MES.rhon_g/hour,-80*MSCM*MES.rhon_g/hour,-80*MSCM*MES.rhon_g/hour,-80*MSCM*MES.rhon_g/hour,1*bar,1*bar,1*bar,-np.pi,-np.pi,0.8*MES.Vbase_shabanpour,-100,-100,-100,0,0,1*MES.rho_w*MES.grav_const,1*MES.rho_w*MES.grav_const,60,60,60,0,0,0,0,0,0,0,-50*MES.Sbase_shabanpour,-50*MES.Sbase_shabanpour,0,0,0,0]),xLF_ub=np.array([80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,60*bar,60*bar,60*bar,np.pi,np.pi,1.2*MES.Vbase_shabanpour,100,100,100,200,200,6000*MES.rho_w*MES.grav_const,6000*MES.rho_w*MES.grav_const,135,135,135,60,60,60,80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,60*MES.Sbase_shabanpour,60*MES.Sbase_shabanpour,60*MES.Sbase_shabanpour,60*MES.Sbase_shabanpour,150,150,35*MW,35*MW]),xLF_init=np.array([20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,45*bar,47*bar,45*bar,0,0,1*MES.Vbase_shabanpour,65,30,-60,20,20,200*MES.rho_w*MES.grav_const,4300*MES.rho_w*MES.grav_const,100,120,120,50,50,50,10*MSCM*MES.rhon_g/hour,3*MSCM*MES.rhon_g/hour,50*MES.Sbase_shabanpour,10*MES.Sbase_shabanpour,0*MES.Sbase_shabanpour,0*MES.Sbase_shabanpour,10,10,30*MW,25*MW]),a=np.array([0,0,0,0,0]),b=np.array([.01*MES.GHV,.3,.3,.04,.05]),c=np.array([1e-6*(MES.GHV)**2,3e-5,2e-5,4e-4,4.5e-4]),scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,max_iters_lf=10,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None},ineq_constr='control',approach='direct',optimization_method='trust-constr',stay_within_bounds=False,fb=None,obj_fun='combined'):
    """Run optimal load flow, with LF as equality constraints.

    Parameters
    ----------
    ineq_contrs : str, optional
        Determines on which variables the bounds (or inequality constraints) are imposed. Options are 'control' or 'all'. Default is 'control'

    Returns
    -------
    xmes_opt : np array
        Solution of LF variables of OF. Unscaled.
    y_opt : np array
        Full solution of OF. That is, control variables and all state variables. Scaled.
    res : OptimizeResult
        The optimization result. The solution is scaled.
    f_vec : list
        List with the values of the objective. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    err_LF_vec : list
        List with the (final) error of the LF solve. Should be per iteration. But when callback is not used, these values are taken based on the list with values per function call. So might not exactly be per iteration. Scaled.
    execution_time : float
        Time of the optimization (excluding creation of network etc.)
    """
    if formulation.get('gas') != 'full':
        raise ValueError('OF not implemented for gas-formulation other than full')
    if formulation.get('heat') != 'half_link_flow':
        raise ValueError('OF not implemented for heat-formulation other than half_link_flow')
    print('\nRunning OPF for MES EH network (inequality constraints on: {}, approach: {}, hard bounds: {}, method: {}, scaling: {})'.format(ineq_constr, approach,stay_within_bounds,optimization_method,scale_var))

    # create network
    with HiddenPrints():
        het_net, gas_net, elec_net, heat_net, _, _ = MES.create_network(hydr_eq='fa',heat_load='outflow',node_set=2) # Ts of both coupling node 'known'

    # update the BC of the MES to match initial guess of OF
    V2_init, Ts1c_init = u_init
    het_net, gas_net, elec_net, heat_net = update_bc(het_net, gas_net, elec_net, heat_net, V2_init, Ts1c_init, scale_var=scale_var,scale_var_params=scale_var_params)
    # set intial guess and initialize network
    het_net.initialize()
    het_net.update(xLF_init,formulation=formulation)
    het_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # initial guess for OF (unscaled)
    u0 = u_init

    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # indices within y of values to be used in objective function
    if obj_fun == 'combined':
        xc_start_ind = len(u_init)+len(slack_init)+len(nlsys.xg_entries)+len(nlsys.xe_entries)+len(nlsys.xh_entries)
        y_ind = [len(u_init),xc_start_ind+2,xc_start_ind+3,xc_start_ind+8,xc_start_ind+9]
    elif obj_fun == 'gas':
        y_ind = [len(u_init)]
    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        ubase = np.array([scale_var_params.get('Vbase'),scale_var_params.get('Tbase')])
        slack_base = np.array([scale_var_params.get('qbase')])
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((ubase,slack_base,1/Dx.data[0])))
        Du = np.diag(1/ubase)
        Du_inv = np.diag(ubase)
        Dh = np.diag(np.concatenate((np.array([1/scale_var_params.get('qbase')]),DF.data[0])))
        u0 = Du.dot(u0) # scale u
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

    lb_ineq_state = np.concatenate((slack_lb,xLF_lb))
    ub_ineq_state = np.concatenate((slack_ub,xLF_ub))
    if scale_var == 'matrix' or scale_var == 'per_unit':
        lb_ineq_state = Dy[len(u_lb):,len(u_lb):].dot(lb_ineq_state)
        ub_ineq_state = Dy[len(u_ub):,len(u_ub):].dot(ub_ineq_state)
        u_lb = Du.dot(u_lb)
        u_ub = Du.dot(u_ub)

    # define objective function
    def obj(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params, fb=fb, formulation=formulation, Du_inv=Du_inv, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init):
        global u_f_vec
        u_f_vec = u.copy()
        global f_vec_global
        if scale_var == 'matrix':
            # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        # update network and solve LF
        network_mes = nlsys.hetnetwork
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_h = nlsys.nlsystemsh[0].heatnetwork
        x, network_mes, network_g, network_e, network_h = solve_lf_in_of(network_mes, network_g, network_e, network_h, u, max_iters=max_iters_lf,tol=tol,formulation=formulation, scale_var=scale_var,scale_var_params=scale_var_params,xLF_init=xLF_init)
        # evaluate objective function
        y = np.concatenate((u,x))
        f = objective_function(y,y_ind,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
        f_vec_global.append(f)
        return f

    def obj_grad(u,y_ind=y_ind,a=a,b=b,c=c,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params, fb=fb, formulation=formulation, max_iters_lf=max_iters_lf,tol=tol,method=approach,xLF_init=xLF_init):
        if scale_var == 'matrix':
            # Optimizer uses scaled y, but objective function wants unscaled y (when using matrix scaling)
            u = Du_inv.dot(u)
        # update network and solve LF
        network_mes = nlsys.hetnetwork
        network_g = nlsys.nlsystemsg[0].gasnetwork
        network_e = nlsys.nlsystemse[0].elecnetwork
        network_h = nlsys.nlsystemsh[0].heatnetwork
        x, network_mes, network_g, network_e, network_h = solve_lf_in_of(network_mes, network_g, network_e, network_h, u, max_iters=max_iters_lf,tol=tol,formulation=formulation, scale_var=scale_var,scale_var_params=scale_var_params,xLF_init=xLF_init)
        # partial derivatives of objective
        y = np.concatenate((u,x))
        deltaf_deltay = grad_objective(y,y_ind,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,fb=fb,Dy_inv=Dy_inv)
        deltaf_deltau = np.zeros((1,len(u)))
        deltaf_deltax = np.zeros((1,len(x)))
        deltaf_deltau[0,:] = deltaf_deltay[:len(u)]
        deltaf_deltax[0,:] = deltaf_deltay[len(u):]
        # partial derivatives of equatilty constraints / load-flow equations
        V2 = u[0]
        nlsys.nlsystemse[0].V_vec_mag[2] = V2
        deltah_deltay = h_der(y, nlsys=nlsys, Dy_inv=Dy_inv, Dh=Dh) # also resets network as a first step
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
        def g(u,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params, Du_inv=Du_inv, Dy=Dy, formulation=formulation, max_iters_lf=max_iters_lf,tol=tol,xLF_init=xLF_init):
            """Determine the nonlinear inequality constraints g(x(u)) >= 0"""
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                u = Du_inv.dot(u)
            # update network and solve LF
            network_mes = nlsys.hetnetwork
            network_g = nlsys.nlsystemsg[0].gasnetwork
            network_e = nlsys.nlsystemse[0].elecnetwork
            network_h = nlsys.nlsystemsh[0].heatnetwork
            x, network_mes, network_g, network_e, network_h = solve_lf_in_of(network_mes, network_g, network_e, network_h, u, max_iters=max_iters_lf,tol=tol,formulation=formulation, scale_var=scale_var,scale_var_params=scale_var_params,xLF_init=xLF_init) # x is scaled when using per unit scaling, unscaled otherwise
            if scale_var == 'matrix': # lb_ineq_state and ub_ineq_state are scaled, so scale x as well
                x = Dy[len(u):,len(u):].dot(x)
            g = np.concatenate((x-lb_ineq_state,ub_ineq_state-x))
            return g
        def g_jac(u,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params, Du_inv=Du_inv, Dy_inv=Dy_inv, Dh=Dh, formulation=formulation, max_iters_lf=max_iters_lf,tol=tol,method=approach,xLF_init=xLF_init):
            """Jacobian of inequality constraints"""
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                u = Du_inv.dot(u)
            # update network and solve LF
            network_mes = nlsys.hetnetwork
            network_g = nlsys.nlsystemsg[0].gasnetwork
            network_e = nlsys.nlsystemse[0].elecnetwork
            network_h = nlsys.nlsystemsh[0].heatnetwork
            x, network_mes, network_g, network_e, network_h = solve_lf_in_of(network_mes, network_g, network_e, network_h, u, max_iters=max_iters_lf,tol=tol,formulation=formulation, scale_var=scale_var,scale_var_params=scale_var_params,xLF_init=xLF_init) # x is scaled when using per unit scaling, unscaled otherwise
            # Jacobian of inequality constraints wrt state variables x
            deltag_deltax = np.vstack((np.eye(len(x)),-np.eye(len(x))))
            deltag_deltau = np.zeros((2*len(x),len(u)))
            # partial derivatives of equatilty constraints / load-flow equations
            V2 = u[0]
            nlsys.nlsystemse[0].V_vec_mag[2] = V2
            y = np.concatenate((u,x))
            deltah_deltay = h_der(y, nlsys=nlsys, Dy_inv=Dy_inv, Dh=Dh) # also resets network as a first step
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
            ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(2*(Dy.shape[0]-len(u0))),np.inf*np.ones(2*(Dy.shape[0]-len(u0))),jac=g_jac,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            ineq_constr_fun  = {'type':'ineq','fun':g,'jac':g_jac}
    else:
        ineq_constr_fun = None

    # define linear inequality constraints (bounds) on the control variables
    if ineq_constr != None:
        # define linear inequality constraints (on the control variables)
        lb_ineq_bounds = u_lb
        ub_ineq_bounds = u_ub
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
    opf_start_time = time.time()
    try:
        if ineq_constr_fun != None:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=[ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds, callback=callback)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, u0,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
        else:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, options={'verbose': 1,'maxiter':max_iter}, bounds=bounds, callback=callback)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
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
        res = spo.OptimizeResult({'success':False,'x':np.array(u_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

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

    if scale_var == 'matrix' or scale_var == 'per_unit':
        u_opf = Du_inv.dot(res.x)
    else:
        u_opf = res.x
    # print solution
    x_opt, het_net, gas_net, elec_net, heat_net = solve_lf_in_of(het_net, gas_net, elec_net, heat_net, u_opf, max_iters=max_iters_lf,tol=tol,formulation=formulation,xLF_init=xLF_init) # unscaled
    y_opt = np.concatenate((u_opf,x_opt)) # unscaled
    if scale_var == 'matrix' or scale_var == 'per_unit':
        y_opt = Dy.dot(y_opt) # scaled
    xmes_opt = x_opt[len(slack_init):] # unscaled
    het_net.reset_network(xmes_opt,formulation=formulation)
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_opt,formulation=formulation)
    print('Solution OF (success: {})'.format(res.success))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
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
    return xmes_opt, y_opt, res, f_vec, err_LF_vec, execution_time

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

def par_obj(obj_fun):
    """Returns the parameter values used in the objective function, and the indices within y of the corresponding energies."""
    a_gas=0
    b_gas=15*MES.GHV#.01*MES.GHV
    c_gas=0#b_gas/100*MES.GHV#1e-6*(MES.GHV)**2
    y_ind_gas = 2
    if obj_fun == 'gas':
        a = np.array([a_gas])
        b = np.array([b_gas])/MW#np.array([b_gas])
        c = np.array([c_gas])
        y_ind = np.array([y_ind_gas])
    elif obj_fun == 'combined':
        # parameter values for the two objective functions
        a = np.array([a_gas,0,0,0,0])
        b = np.array([b_gas,2,5,1,0])/MW#np.array([b_gas,.3,.3,.04,.05])
        c = np.concatenate((np.array([c_gas]),b[1:]/100))/MW#np.array([c_gas,3e-5,2e-5,4e-4,4.5e-4])
        y_ind = [y_ind_gas,28,29,34,35]
    return a,b,c,y_ind

def bounds_OF():
    """Returns the lower and upper bounds for the control variables u, the slack varialbes, and the state variables x"""
    u_lb=np.array([.8*MES.Vbase_shabanpour,100])
    u_ub=np.array([1.2*MES.Vbase_shabanpour,135])
    slack_lb=np.array([-80*MSCM*MES.rhon_g/hour])
    slack_ub=np.array([-(10.8649+20)*MSCM*MES.rhon_g/hour])
    xLF_lb=np.array([-80*MSCM*MES.rhon_g/hour,-80*MSCM*MES.rhon_g/hour,-80*MSCM*MES.rhon_g/hour,-80*MSCM*MES.rhon_g/hour,1*bar,1*bar,1*bar,-np.pi,-np.pi,0.8*MES.Vbase_shabanpour,-100,-100,-100,0,0,1*MES.rho_w*MES.grav_const,1*MES.rho_w*MES.grav_const,60,60,60,0,0,0,0,0,0,0,-50*MES.Sbase_shabanpour,-50*MES.Sbase_shabanpour,0,0,0,0])
    xLF_ub=np.array([80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,60*bar,60*bar,60*bar,np.pi,np.pi,1.2*MES.Vbase_shabanpour,100,100,100,200,200,6000*MES.rho_w*MES.grav_const,6000*MES.rho_w*MES.grav_const,135,135,135,60,60,60,80*MSCM*MES.rhon_g/hour,80*MSCM*MES.rhon_g/hour,60*MES.Sbase_shabanpour,60*MES.Sbase_shabanpour,60*MES.Sbase_shabanpour,60*MES.Sbase_shabanpour,150,150,35*MW,35*MW])
    return u_lb, u_ub, slack_lb, slack_ub, xLF_lb, xLF_ub

def initial_guess():
    """Returns the initial guess for the control variables u, the slack varialbes, and the state variables x"""
    u_init=np.array([1.1*MES.Vbase_shabanpour,130])
    slack_init=np.array([-70*MSCM*MES.rhon_g/hour])
    xLF_init=np.array([20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,20*MSCM*MES.rhon_g/hour,45*bar,47*bar,45*bar,0,0,1*MES.Vbase_shabanpour,65,30,-60,20,20,200*MES.rho_w*MES.grav_const,4300*MES.rho_w*MES.grav_const,100,120,120,50,50,50,10*MSCM*MES.rhon_g/hour,3*MSCM*MES.rhon_g/hour,50*MES.Sbase_shabanpour,20*MES.Sbase_shabanpour,0*MES.Sbase_shabanpour,0*MES.Sbase_shabanpour,10,10,30*MW,25*MW])
    return u_init, slack_init, xLF_init

def compare_methods(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF of gas-electricity network for different optimization methods, objective function, and bounds. Without scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    max_iter = 40
    tol = 1e-6
    scale_var = None
    fb = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

    # bounds
    u_lb, u_ub, slack_lb, slack_ub, xLF_lb, xLF_ub = bounds_OF()

    # intial guess
    u_init, slack_init, xLF_init = initial_guess()

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, gas_net_LF, elec_net_LF, heat_net_LF, water, gas = MES.create_network(hydr_eq='fa',heat_load='outflow',node_set=2)
        x0_LF = MES.initialize_network(het_net_LF,gas_net_LF,elec_net_LF,heat_net_LF,water,gas,formulation,heat_load='outflow',node_set=2,scale_var=None,scale_var_params=None)
        xmes_LF, iters_LF, err_vec_LF,p_g_vec,_,q_inj,_,V_mag_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,Tsc_vec,_ = het_net_LF.solve_network(tol,10,solver='NR',formulation=formulation)
        V2_sol = V_mag_vec[2]
        Ts1c_sol = Tsc_vec[1]
        q0_sol = q_inj[0]
        y_LF = np.concatenate((np.array([V2_sol,Ts1c_sol,q0_sol]),xmes_LF))
        # value of objective functions for LF solution
        a_gas,b_gas,c_gas,y_ind_gas = par_obj('gas')
        f_LF_gas = objective_function(y_LF,y_ind_gas,a=a_gas,b=b_gas,c=c_gas)
        a_comb,b_comb,c_comb,y_ind_comb = par_obj('combined')
        f_LF_comb = objective_function(y_LF,y_ind_comb,a=a_comb,b=b_comb,c=c_comb)

    result = dict()
    xmes_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']
    ineq_constrs = ['control','all']
    obj_funs = ['gas','combined']

    for obj_fun in obj_funs:
        a,b,c,y_ind = par_obj(obj_fun)
        if obj_fun == 'gas':
            f_LF_sol = f_LF_gas
        else:
            f_LF_sol = f_LF_comb
        for ineq_constr in ineq_constrs:
            # plots
            fig_f = plt.figure('obj_OF_MES_EH_ineq_constr_{}_obj_{}'.format(ineq_constr,obj_fun))
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
                        xmes_opt, res, f_vec, err_LF_vec, execution_time = run_optimal_load_flow(u_lb=u_lb,u_init=u_init,u_ub=u_ub,slack_lb=slack_lb,slack_ub=slack_ub,slack_init=slack_init,xLF_lb=xLF_lb,xLF_ub=xLF_ub,xLF_init=xLF_init,a=a,b=b,c=c,scale_var=None,scale_var_params=None,tol=tol,max_iter=max_iter,formulation=formulation,ineq_constr=ineq_constr,derivatives=derivatives,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,obj_fun=obj_fun)
                        result[method+'_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun] = res
                        xmes_res[method+'_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun] = xmes_opt
                        max_fev = max(max_fev,len(f_vec))
                        # plot results
                        ax_f.semilogy(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
            ax_f.semilogy([0,max_fev],[f_LF_sol,f_LF_sol],':r')
            ax_f.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'Tables','N_Shabanpour')
        with open(os.path.join(path_to_tables,'optimizer_info_MES_EH_methods.txt'), "w") as table:
            for obj_fun in obj_funs:
                for ineq_constr in ineq_constrs:
                    for bound in bounds:
                        for der in ders:
                            res_trust = result.get('trust-constr_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                            res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                            res_ipopt = result.get('ipopt_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                            table.write(r'{} & {} & {} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(obj_fun,ineq_constr,bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(res_trust.x,y_LF),error(res_slsqp.x,y_LF),error(res_ipopt.x,y_LF)))
                    table.write(r'\cline{2-16} ')
                table.write(r'\hline ')

    for obj_fun in obj_funs:
        for ineq_constr in ineq_constrs:
            for bound in bounds:
                for der in ders:
                    res_trust = result.get('trust-constr_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                    res_ipopt = result.get('ipopt_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                    print('\nObj: {}, constraints: {}, bounds: {}, der: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nt-c:{}\nSLSQP: {}\nIPOPT: {}\nError for t-c:{}, SLSQP: {}, IPOPT:{}'.format(obj_fun,ineq_constr,bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(res_trust.x,y_LF),error(res_slsqp.x,y_LF),error(res_ipopt.x,y_LF)))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','N_Shabanpour')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def compare_methods_scaled(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF of gas-electricity network for different optimization methods, objective function, and bounds. Using matrix scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    max_iter = 40
    tol = 1e-6
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

    # bounds
    u_lb, u_ub, slack_lb, slack_ub, xLF_lb, xLF_ub = bounds_OF()

    # intial guess
    u_init, slack_init, xLF_init = initial_guess()

    # scaling
    scale_var = 'matrix'
    fb = 1e10
    qbase = 10*MSCM*MES.rhon_g/hour
    pgbase = 30*bar
    mbase = 50.
    phbase = 1000.*MES.rho_w*MES.grav_const
    Tbase = 130.
    Sbase = 100.*MES.Sbase_shabanpour #since P and Q are in p.u. (i.e. given in MW based on p.u. delta, V and line parameters)
    phibase =Sbase
    Vbase = 1.*MES.Vbase_shabanpour
    deltabase = 1.
    scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Sbase}

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, gas_net_LF, elec_net_LF, heat_net_LF, water, gas = MES.create_network(hydr_eq='fa',heat_load='outflow',node_set=2)
        x0_LF = MES.initialize_network(het_net_LF,gas_net_LF,elec_net_LF,heat_net_LF,water,gas,formulation,heat_load='outflow',node_set=2,scale_var=None,scale_var_params=None)
        xmes_LF, iters_LF, err_vec_LF,p_g_vec,_,q_inj,_,V_mag_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,Tsc_vec,_ = het_net_LF.solve_network(tol,10,solver='NR',formulation=formulation)
        V2_sol = V_mag_vec[2]
        Ts1c_sol = Tsc_vec[1]
        q0_sol = q_inj[0]
        nlsys = NonLinearSystemHeterogeneous(het_net_LF,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        Dx = nlsys.Dx()
        ubase = np.array([scale_var_params.get('Vbase'),scale_var_params.get('Tbase')])
        slack_base = np.array([scale_var_params.get('qbase')])
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        y_LF = np.concatenate((np.array([V2_sol,Ts1c_sol,q0_sol]),xmes_LF)) # unscaled
        # value of objective functions for LF solution
        a_gas,b_gas,c_gas,y_ind_gas = par_obj('gas')
        f_LF_gas = objective_function(y_LF,y_ind_gas,a=a_gas,b=b_gas,c=c_gas)/fb # scaled
        a_comb,b_comb,c_comb,y_ind_comb = par_obj('combined')
        f_LF_comb = objective_function(y_LF,y_ind_comb,a=a_comb,b=b_comb,c=c_comb)/fb # scaled
        y_LF = Dy.dot(y_LF) # scaled

    result = dict()
    xmes_res = dict() # unscaled
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']
    ineq_constrs = ['control','all']
    obj_funs = ['gas','combined']

    for obj_fun in obj_funs:
        a,b,c,y_ind = par_obj(obj_fun)
        if obj_fun == 'gas':
            f_LF_sol = f_LF_gas
        else:
            f_LF_sol = f_LF_comb
        for ineq_constr in ineq_constrs:
            # plots
            fig_f = plt.figure('obj_OF_MES_EH_ineq_constr_{}_obj_{}_scaled'.format(ineq_constr,obj_fun))
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
                        xmes_opt, res, f_vec, err_LF_vec, execution_time = run_optimal_load_flow(u_lb=u_lb,u_init=u_init,u_ub=u_ub,slack_lb=slack_lb,slack_ub=slack_ub,slack_init=slack_init,xLF_lb=xLF_lb,xLF_ub=xLF_ub,xLF_init=xLF_init,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,formulation=formulation,ineq_constr=ineq_constr,derivatives=derivatives,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,obj_fun=obj_fun)
                        result[method+'_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun] = res
                        xmes_res[method+'_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun] = xmes_opt # unscaled
                        max_fev = max(max_fev,len(f_vec))
                        # plot results
                        ax_f.semilogy(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
            ax_f.semilogy([0,max_fev],[f_LF_sol,f_LF_sol],':r')
            ax_f.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'Tables','N_Shabanpour')
        with open(os.path.join(path_to_tables,'optimizer_info_MES_EH_methods_scaled.txt'), "w") as table:
            for obj_fun in obj_funs:
                for ineq_constr in ineq_constrs:
                    for bound in bounds:
                        for der in ders:
                            res_trust = result.get('trust-constr_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                            res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                            res_ipopt = result.get('ipopt_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                            table.write(r'{} & {} & {} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(obj_fun,ineq_constr,bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(res_trust.x,y_LF),error(res_slsqp.x,y_LF),error(res_ipopt.x,y_LF)))
                    table.write(r'\cline{2-16} ')
                table.write(r'\hline ')

    for obj_fun in obj_funs:
        for ineq_constr in ineq_constrs:
            for bound in bounds:
                for der in ders:
                    res_trust = result.get('trust-constr_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                    res_ipopt = result.get('ipopt_'+bound+'_'+der+'_'+ineq_constr+'_'+obj_fun)
                    print('\nObj: {}, constraints: {}, bounds: {}, der: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nt-c:{}\nSLSQP: {}\nIPOPT: {}\nError for t-c:{}, SLSQP: {}, IPOPT:{}'.format(obj_fun,ineq_constr,bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(res_trust.x,y_LF),error(res_slsqp.x,y_LF),error(res_ipopt.x,y_LF)))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','N_Shabanpour')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def compare_methods_sep_LF(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF of gas-electricity network for different optimization methods, objective function, and bounds. LF is included implicitily. Without scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    max_iter = 40
    max_iters_lf = 10
    tol = 1e-6
    scale_var = None
    fb = None
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

    # bounds
    u_lb, u_ub, slack_lb, slack_ub, xLF_lb, xLF_ub = bounds_OF()

    # intial guess
    u_init, slack_init, xLF_init = initial_guess()

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, gas_net_LF, elec_net_LF, heat_net_LF, water, gas = MES.create_network(hydr_eq='fa',heat_load='outflow',node_set=2)
        x0_LF = MES.initialize_network(het_net_LF,gas_net_LF,elec_net_LF,heat_net_LF,water,gas,formulation,heat_load='outflow',node_set=2,scale_var=None,scale_var_params=None)
        xmes_LF, iters_LF, err_vec_LF,p_g_vec,_,q_inj,_,V_mag_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,Tsc_vec,_ = het_net_LF.solve_network(tol,max_iters_lf,solver='NR',formulation=formulation)
        V2_sol = V_mag_vec[2]
        Ts1c_sol = Tsc_vec[1]
        q0_sol = q_inj[0]
        y_LF = np.concatenate((np.array([V2_sol,Ts1c_sol,q0_sol]),xmes_LF))
        # value of objective functions for LF solution
        a_gas,b_gas,c_gas,y_ind_gas = par_obj('gas')
        f_LF_gas = objective_function(y_LF,y_ind_gas,a=a_gas,b=b_gas,c=c_gas)
        a_comb,b_comb,c_comb,y_ind_comb = par_obj('combined')
        f_LF_comb = objective_function(y_LF,y_ind_comb,a=a_comb,b=b_comb,c=c_comb)

    result = dict()
    y_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']
    ineq_constrs = ['control','all']
    obj_funs = ['gas','combined']
    for obj_fun in obj_funs:
        a,b,c,y_ind = par_obj(obj_fun)
        if obj_fun == 'gas':
            f_LF_sol = f_LF_gas
        else:
            f_LF_sol = f_LF_comb
        for ineq_constr in ineq_constrs:
            # plots
            fig_f = plt.figure('obj_OF_MES_EH_ineq_constr_{}_obj_{}_sep_LF'.format(ineq_constr,obj_fun))
            ax_f = fig_f.gca()
            ax_f.set_xlabel('Iteration')
            ax_f.set_ylabel('f')

            fig_LF_error = plt.figure('error_LF_in_OF_MES_EH_ineq_constr_{}_obj_{}_sep_LF'.format(ineq_constr,obj_fun))
            ax_LF_error = fig_LF_error.gca()
            ax_LF_error.set_xlabel('Iteration?')
            ax_LF_error.set_ylabel(r'$||F||_2$')

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
                            xmes_opt, y_opt, res, f_vec, err_LF_vec, execution_time = run_optimal_load_flow_separate_LF(u_lb=u_lb,u_init=u_init,u_ub=u_ub,slack_lb=slack_lb,slack_ub=slack_ub,slack_init=slack_init,xLF_lb=xLF_lb,xLF_ub=xLF_ub,xLF_init=xLF_init,a=a,b=b,c=c,scale_var=None,scale_var_params=None,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,formulation=formulation,ineq_constr=ineq_constr,approach=approach,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,obj_fun=obj_fun)
                        else:
                            approach_legend = 'an'
                            xmes_opt, res, f_vec, err_LF_vec, execution_time = run_optimal_load_flow(u_lb=u_lb,u_init=u_init,u_ub=u_ub,slack_lb=slack_lb,slack_ub=slack_ub,slack_init=slack_init,xLF_lb=xLF_lb,xLF_ub=xLF_ub,xLF_init=xLF_init,a=a,b=b,c=c,scale_var=None,scale_var_params=None,tol=tol,max_iter=max_iter,formulation=formulation,ineq_constr=ineq_constr,derivatives=True,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,obj_fun=obj_fun)
                            y_opt = res.x
                        result[method+'_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach] = res
                        y_res[method+'_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach] = y_opt
                        max_fev = max(max_fev,len(f_vec))
                        # plot results
                        ax_f.semilogy(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend))
                        ax_LF_error.semilogy(err_LF_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend))
            ax_f.semilogy([0,max_fev],[f_LF_sol,f_LF_sol],':r')
            ax_f.legend(handles=legend_handles)
            ax_LF_error.semilogy([0,max_fev],[tol,tol],':k')
            ax_LF_error.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'Tables','N_Shabanpour')
        with open(os.path.join(path_to_tables,'optimizer_info_MES_EH_methods_sep_LF.txt'), "w") as table:
            for obj_fun in obj_funs:
                for ineq_constr in ineq_constrs:
                    for bound in bounds:
                        for approach in approaches:
                            if approach == 'eq_constr':
                                approach_label = 'eq. constr.'
                            else:
                                approach_label = approach
                            res_trust = result.get('trust-constr_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            res_slsqp = result.get('SLSQP_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            res_ipopt = result.get('ipopt_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            y_trust = y_res.get('trust-constr_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            y_slsqp = y_res.get('SLSQP_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            y_ipopt = y_res.get('ipopt_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            table.write(r'{} & {} & {} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(obj_fun,ineq_constr,bound,approach_label,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(y_trust,y_LF),error(y_slsqp,y_LF),error(y_ipopt,y_LF)))
                    table.write(r'\cline{2-16} ')
                table.write(r'\hline ')

    for obj_fun in obj_funs:
        for ineq_constr in ineq_constrs:
            for bound in bounds:
                for approach in approaches:
                    res_trust = result.get('trust-constr_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    res_ipopt = result.get('ipopt_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    y_trust = y_res.get('trust-constr_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    y_slsqp = y_res.get('SLSQP_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    y_ipopt = y_res.get('ipopt_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    print('\nObj: {}, constraints: {}, bounds: {}, approach: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nt-c:{}\nSLSQP: {}\nIPOPT: {}\nError for t-c:{}, SLSQP: {}, IPOPT:{}'.format(obj_fun,ineq_constr,bound,approach,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(y_trust,y_LF),error(y_slsqp,y_LF),error(y_ipopt,y_LF)))
    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','N_Shabanpour')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def compare_methods_sep_LF_scaled(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF of gas-electricity network for different optimization methods, objective function, and bounds. LF is included implicitily. Without scaling."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    max_iter = 40
    max_iters_lf = 10
    tol = 1e-6
    formulation = {'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

    # bounds
    u_lb, u_ub, slack_lb, slack_ub, xLF_lb, xLF_ub = bounds_OF()

    # intial guess
    u_init, slack_init, xLF_init = initial_guess()

    # scaling
    scale_var = 'matrix'
    fb = 1e10
    qbase = 10*MSCM*MES.rhon_g/hour
    pgbase = 30*bar
    mbase = 50.
    phbase = 1000.*MES.rho_w*MES.grav_const
    Tbase = 130.
    Sbase = 100.*MES.Sbase_shabanpour #since P and Q are in p.u. (i.e. given in MW based on p.u. delta, V and line parameters)
    phibase =Sbase
    Vbase = 1.*MES.Vbase_shabanpour
    deltabase = 1.
    scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Sbase}

    # steady-state LF solution
    with HiddenPrints():
        het_net_LF, gas_net_LF, elec_net_LF, heat_net_LF, water, gas = MES.create_network(hydr_eq='fa',heat_load='outflow',node_set=2)
        x0_LF = MES.initialize_network(het_net_LF,gas_net_LF,elec_net_LF,heat_net_LF,water,gas,formulation,heat_load='outflow',node_set=2,scale_var=None,scale_var_params=None)
        xmes_LF, iters_LF, err_vec_LF,p_g_vec,_,q_inj,_,V_mag_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,Tsc_vec,_ = het_net_LF.solve_network(tol,max_iters_lf,solver='NR',formulation=formulation)
        V2_sol = V_mag_vec[2]
        Ts1c_sol = Tsc_vec[1]
        q0_sol = q_inj[0]
        nlsys = NonLinearSystemHeterogeneous(het_net_LF,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        Dx = nlsys.Dx()
        ubase = np.array([scale_var_params.get('Vbase'),scale_var_params.get('Tbase')])
        slack_base = np.array([scale_var_params.get('qbase')])
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        y_LF = np.concatenate((np.array([V2_sol,Ts1c_sol,q0_sol]),xmes_LF)) # unscaled
        # value of objective functions for LF solution
        a_gas,b_gas,c_gas,y_ind_gas = par_obj('gas')
        f_LF_gas = objective_function(y_LF,y_ind_gas,a=a_gas,b=b_gas,c=c_gas)/fb # scaled
        a_comb,b_comb,c_comb,y_ind_comb = par_obj('combined')
        f_LF_comb = objective_function(y_LF,y_ind_comb,a=a_comb,b=b_comb,c=c_comb)/fb # scaled
        y_LF = Dy.dot(y_LF) # scaled

    result = dict()
    y_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']
    ineq_constrs = ['control','all']
    obj_funs = ['gas','combined']
    for obj_fun in obj_funs:
        a,b,c,y_ind = par_obj(obj_fun)
        if obj_fun == 'gas':
            f_LF_sol = f_LF_gas
        else:
            f_LF_sol = f_LF_comb
        for ineq_constr in ineq_constrs:
            # plots
            fig_f = plt.figure('obj_OF_MES_EH_ineq_constr_{}_obj_{}_sep_LF_scaled'.format(ineq_constr,obj_fun))
            ax_f = fig_f.gca()
            ax_f.set_xlabel('Iteration')
            ax_f.set_ylabel('f')

            fig_LF_error = plt.figure('error_LF_in_OF_MES_EH_ineq_constr_{}_obj_{}_sep_LF_scaled'.format(ineq_constr,obj_fun))
            ax_LF_error = fig_LF_error.gca()
            ax_LF_error.set_xlabel('Iteration?')
            ax_LF_error.set_ylabel(r'$||F||_2$')

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
                            xmes_opt, y_opt, res, f_vec, err_LF_vec, execution_time = run_optimal_load_flow_separate_LF(u_lb=u_lb,u_init=u_init,u_ub=u_ub,slack_lb=slack_lb,slack_ub=slack_ub,slack_init=slack_init,xLF_lb=xLF_lb,xLF_ub=xLF_ub,xLF_init=xLF_init,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,formulation=formulation,ineq_constr=ineq_constr,approach=approach,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,obj_fun=obj_fun)
                        else:
                            approach_legend = 'an'
                            xmes_opt, res, f_vec, err_LF_vec, execution_time = run_optimal_load_flow(u_lb=u_lb,u_init=u_init,u_ub=u_ub,slack_lb=slack_lb,slack_ub=slack_ub,slack_init=slack_init,xLF_lb=xLF_lb,xLF_ub=xLF_ub,xLF_init=xLF_init,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,formulation=formulation,ineq_constr=ineq_constr,derivatives=True,optimization_method=method,stay_within_bounds=stay_within_bounds,fb=fb,obj_fun=obj_fun)
                            y_opt = res.x
                        result[method+'_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach] = res
                        y_res[method+'_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach] = y_opt
                        max_fev = max(max_fev,len(f_vec))
                        # plot results
                        ax_f.semilogy(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend))
                        ax_LF_error.semilogy(err_LF_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend))
            ax_f.semilogy([0,max_fev],[f_LF_sol,f_LF_sol],':r')
            ax_f.legend(handles=legend_handles)
            ax_LF_error.semilogy([0,max_fev],[tol,tol],':k')
            ax_LF_error.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'Tables','N_Shabanpour')
        with open(os.path.join(path_to_tables,'optimizer_info_MES_EH_methods_sep_LF_scaled.txt'), "w") as table:
            for obj_fun in obj_funs:
                for ineq_constr in ineq_constrs:
                    for bound in bounds:
                        for approach in approaches:
                            if approach == 'eq_constr':
                                approach_label = 'eq. constr.'
                            else:
                                approach_label = approach
                            res_trust = result.get('trust-constr_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            res_slsqp = result.get('SLSQP_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            res_ipopt = result.get('ipopt_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            y_trust = y_res.get('trust-constr_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            y_slsqp = y_res.get('SLSQP_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            y_ipopt = y_res.get('ipopt_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                            table.write(r'{} & {} & {} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(obj_fun,ineq_constr,bound,approach_label,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(y_trust,y_LF),error(y_slsqp,y_LF),error(y_ipopt,y_LF)))
                    table.write(r'\cline{2-16} ')
                table.write(r'\hline ')

    for obj_fun in obj_funs:
        for ineq_constr in ineq_constrs:
            for bound in bounds:
                for approach in approaches:
                    res_trust = result.get('trust-constr_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    res_ipopt = result.get('ipopt_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    y_trust = y_res.get('trust-constr_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    y_slsqp = y_res.get('SLSQP_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    y_ipopt = y_res.get('ipopt_'+bound+'_'+ineq_constr+'_'+obj_fun+'_'+approach)
                    print('\nObj: {}, constraints: {}, bounds: {}, approach: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\nt-c:{}\nSLSQP: {}\nIPOPT: {}\nError for t-c:{}, SLSQP: {}, IPOPT:{}'.format(obj_fun,ineq_constr,bound,approach,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(y_trust,y_LF),error(y_slsqp,y_LF),error(y_ipopt,y_LF)))
    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','N_Shabanpour')
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        # compare_methods(dir_path=dir_path,save_tables=False,save_figs=False)
        # compare_methods_scaled(dir_path=dir_path,save_tables=False,save_figs=False)
        # compare_methods_sep_LF(dir_path=dir_path,save_tables=False,save_figs=False)
        compare_methods_sep_LF_scaled(dir_path=dir_path,save_tables=False,save_figs=False)

    plt.show()
