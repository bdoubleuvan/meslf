"""
Simple optimization problem, based on a single gas-pipe, to test different optimizers, and the effect of a non-defined models for variables outside the bounds.
"""
import numpy as np
import scipy.optimize as spo
import ipopt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from meslf.utils.constants import mbar, bar
import os

# solution
x_min = np.array([1,5])#np.array([1,5*bar])

colors_method = {'trust-constr':'tab:blue','SLSQP':'tab:orange','ipopt':'tab:green'}
markers_bounds = {'soft':'s','hard':'*'}
linestyles_derivatives = {'num':'--','an':'-'}
linestyles_approaches = {'adjoint':'--','direct':'-','eq_constr':':'}
linestyles_contraints = {'eq':'-','ineq':'--','bound':':'}

marker_size = 10
legend_handles = [Line2D([0], [0], color=colors_method.get('trust-constr'), label='trust-constr'),
    Line2D([0], [0], color=colors_method.get('SLSQP'), label='SLSQP'),
    Line2D([0], [0], color=colors_method.get('ipopt'), label='ipopt'),
    Line2D([0], [0], marker=markers_bounds.get('soft'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Soft constraints'),
    Line2D([0], [0], marker=markers_bounds.get('hard'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Hard constraints'),
    Line2D([0], [0], color='k',ls=linestyles_derivatives.get('num'), label='Numerical derivatives'),
    Line2D([0], [0], color='k',ls=linestyles_derivatives.get('an'), label='Analytical derivatives')]
legend_handles_substituted = [Line2D([0], [0], color=colors_method.get('trust-constr'), label='trust-constr'),
    Line2D([0], [0], color=colors_method.get('SLSQP'), label='SLSQP'),
    Line2D([0], [0], color=colors_method.get('ipopt'), label='ipopt'),
    Line2D([0], [0], marker=markers_bounds.get('soft'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Soft constraints'),
    Line2D([0], [0], marker=markers_bounds.get('hard'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Hard constraints'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('adjoint'), label='Adjoint approach'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('direct'), label='Direct approach'),
    Line2D([0], [0], color='k',ls=linestyles_approaches.get('eq_constr'), label='Equality constraint')]

def f(x,a=3,b=-4,c=2,scale_var=None,scale_var_params=None):
    """The objective function. If per unit scaling is used, x, a, b, and c are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    x : np array
        Variables. Scaled when per unit scaling is used, unscaled otherwise.
    a, b, c : floats, optional
        Parameters. Scaled when per unit scaling is used, unscaled otherwise. Default are 3, -4, and 2 respectively.
    scale_var : str, optional
        Which scaling is used. Options are 'per_unit', 'matrix', or None. Default is None.

    Returns
    -------
    f : float
        The objective function. Scaled when per unit scaling or matrix scaling is used.
    """
    f = a + b*x[0] + c*x[0]**2
    global x0_f_global
    global x1_f_global
    global f_vec_global
    x0_f_global.append(x[0])
    x1_f_global.append(x[1])
    f_vec_global.append(f)
    return f

def f_der(x,a=3,b=-4,c=2,scale_var=None,scale_var_params=None):
    """The first derivative of the objective function. If per unit scaling is used, x, a, b, and c are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    x : np array
        Variables. Scaled when per unit scaling is used, unscaled otherwise.
    a, b, c : floats, optional
        Parameters. Scaled when per unit scaling is used, unscaled otherwise. Default are 3, -4, and 2 respectively.
    scale_var : str, optional
        Which scaling is used. Options are 'per_unit', 'matrix', or None. Default is None.

    Returns
    -------
    der : np array
        The first derivative of the objective function. Scaled when per unit scaling or matrix scaling is used.
    """
    der = np.zeros(len(x))
    der[0] = b + 2*c*x[0]
    return der

def f_hess(x,a=3,b=-4,c=2,scale_var=None,scale_var_params=None):
    """The Hessian of the objective function. If per unit scaling is used, x, a, b, and c are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    x : np array
        Variables. Scaled when per unit scaling is used, unscaled otherwise.
    a, b, c : floats, optional
        Parameters. Scaled when per unit scaling is used, unscaled otherwise. Default are 3, -4, and 2 respectively.
    scale_var : str, optional
        Which scaling is used. Options are 'per_unit', 'matrix', or None. Default is None.

    Returns
    -------
    hess : np array
        The Hessian of the objective function. Scaled when per unit scaling or matrix scaling is used.
    """
    hess = np.zeros((len(x),len(x)))
    hess[0,0] = 2*c
    return hess

def h(x,alpha=x_min[0]/np.sqrt(x_min[1]),scale_var=None,scale_var_params=None):
    """Equality constraints h(x)=0. If per unit scaling is used, x and alpha are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    x : np array
        Variables. Scaled when per unit scaling is used, unscaled otherwise.
    alpha : float, optional
        Parameters. Scaled when per unit scaling is used, unscaled otherwise. Default is approximately 0.000447.
    scale_var : str, optional
        Which scaling is used. Options are 'per_unit', 'matrix', or None. Default is None.

    Returns
    -------
    h : np array
        The equality. Scaled when per unit scaling or matrix scaling is used.
    """
    # print('In h: x[1]={}'.format(x[1]))
    global x0_h_global
    global x1_h_global
    x0_h_global.append(x[0])
    x1_h_global.append(x[1])
    h = x[0] - alpha*np.sqrt(x[1])
    return h

def solve_h(x,alpha=x_min[0]/np.sqrt(x_min[1]),scale_var=None,scale_var_params=None):
    """Solve the equatilty constraint h(x)=0 for x0 as a function of x1. If per unit scaling is used, x and alpha are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    x : np array
        Variables. Scaled when per unit scaling is used, unscaled otherwise.
    alpha : float, optional
        Parameters. Scaled when per unit scaling is used, unscaled otherwise. Default is approximately 0.000447.
    scale_var : str, optional
        Which scaling is used. Options are 'per_unit', 'matrix', or None. Default is None.

    Returns
    -------
    x0 : float
        x0. Scaled when per unit scaling or matrix scaling is used.
    """
    global x0_h_global
    global x1_h_global
    x1_h_global.append(x[0])
    x0 = alpha*np.sqrt(x[0])
    x0_h_global.append(x0)
    return x0

def h_der(x,alpha=x_min[0]/np.sqrt(x_min[1]),scale_var=None,scale_var_params=None):
    """First derivatives of equality constraints h(x)=0. If per unit scaling is used, x and alpha are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    x : np array
        Variables. Scaled when per unit scaling is used, unscaled otherwise.
    alpha : float, optional
        Parameters. Scaled when per unit scaling is used, unscaled otherwise. Default is approximately 0.000447.
    scale_var : str, optional
        Which scaling is used. Options are 'per_unit', 'matrix', or None. Default is None.

    Returns
    -------
    dh_dx : np array
        First derivative of the equality. Scaled when per unit scaling or matrix scaling is used.
    """
    # print('In h_der: x[1]={}'.format(x[1]))
    global x0_der_h_global
    global x1_der_h_global
    x0_der_h_global.append(x[0])
    x1_der_h_global.append(x[1])
    dh_dx = np.array([1,-alpha/(2*np.sqrt(x[1]))])
    return dh_dx

def optimize(x0,tol=1e-6,max_iter=150,optimization_method='trust-constr',stay_within_bounds=True,derivatives=False,a=3,b=-4,c=2,alpha=x_min[0]/np.sqrt(x_min[1]),scale_var=None,scale_var_params=None,lb=np.array([0.1,1*bar]),ub=np.array([5,10*bar])):
    """Minimize the testproblem based on a single gas pipe. The nonlinear equality constraint is taken as equality constraint.

    Parameters
    ----------
    x0 : np array
        Initial guess for the optimizer. caled when per unit scaling is used, unscaled otherwise.
    alpha : float, optional
        Parameters. Scaled when per unit scaling is used, unscaled otherwise. Default is 1.
    method : str, optional
        Which optimization method to use. Options are 'trust-constr', 'SLSQP', or 'ipopt'. Default is 'trust-constr'
    """
    # bounds (the optimizer uses the scaled x, so the bounds need to be scaled as well)
    lb_bounds = lb
    ub_bounds = ub
    if optimization_method == 'ipopt':
        bounds = [(lb,ub) for lb, ub in zip(lb_bounds,ub_bounds)]
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
    else:
        bounds = spo.Bounds(lb_bounds,ub_bounds,keep_feasible=stay_within_bounds)

    print('\nOptimizing using {}, with x0={}, alpha={}, hard bounds: {}, analytical der.: {}, scaling: {}'.format(optimization_method,x0,alpha,stay_within_bounds,derivatives,scale_var))

    # set value of alpha in objective and constraints
    def obj(x,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params):
        # if scale_var == 'matrix':
        #     # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
        #     x = Dx_inv.dot(x)
        return f(x,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params)
    def obj_grad(x,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params):
        # if scale_var == 'matrix':
        #     # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
        #     x = Dx_inv.dot(x)
        return f_der(x,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params)
    def obj_hess(x,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params):
        # if scale_var == 'matrix':
        #     # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
        #     x = Dx_inv.dot(x)
        return f_hess(x,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params)
    def h_alpha(x,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params):
        # if scale_var == 'matrix':
        #     # Optimizer uses scaled x, but equality constraints wants unscaled x (when using matrix scaling)
        #     x = Dx_inv.dot(x)
        return h(x,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params)
    def h_der_alpha(x,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params):
        # if scale_var == 'matrix':
        #     # Optimizer uses scaled x, but equality constraints wants unscaled x (when using matrix scaling)
        #     x = Dx_inv.dot(x)
        return h_der(x,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params)

    # equality constraints
    if derivatives:
        if optimization_method == 'trust-constr':
            eq_constr = spo.NonlinearConstraint(h_alpha,np.zeros(1),np.zeros(1),jac=h_der_alpha,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            eq_constr  = {'type':'eq','fun':h_alpha,'jac':h_der_alpha}
    else:
        if optimization_method == 'trust-constr':
            eq_constr = spo.NonlinearConstraint(h_alpha,np.zeros(1),np.zeros(1),keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            eq_constr  = {'type':'eq','fun':h_alpha}
    # callback
    f_vec = list()
    x0_vec = list()
    x1_vec = list()
    global x0_f_global
    global x1_f_global
    global f_vec_global
    x0_f_global = list()
    x1_f_global = list()
    f_vec_global = list()
    if optimization_method == 'trust-constr':
        nit = 0
        nfev = 0
        njev = 0
        nhev = 0
        def callback(xk,state):
            """Called after every iteration"""
            f_vec.append(state.fun)
            x0_vec.append(xk[0])
            x1_vec.append(xk[1])
            nit = state.nit
            nfev = state.nfev
            njev = state.njev
            nhev = state.nhev
            return False
    elif optimization_method == 'SLSQP':
        f_vec.append(obj(x0))
        x0_vec.append(x0[0])
        x1_vec.append(x0[1])
        def callback(xk):
            """Called after every iteration"""
            f_vec.append(obj(xk))
            x0_vec.append(xk[0])
            x1_vec.append(xk[1])
            return False
    elif optimization_method == 'ipopt':
        # callback is not implemented in the ipopt (cyipopt) package / wrapper.
        pass

    # solve minimization
    try:
        if derivatives:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, x0, method=optimization_method, jac=obj_grad, hess=obj_hess, constraints=[eq_constr], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol, callback=callback)
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, x0, method=optimization_method, jac=obj_grad, constraints=[eq_constr], options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, x0, jac=obj_grad, constraints=[eq_constr], options={'maxiter':max_iter,'disp': 1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
        else:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, x0, method=optimization_method, constraints=[eq_constr], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol, callback=callback)
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, x0, method=optimization_method, constraints=[eq_constr], options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, x0, method=optimization_method, constraints=[eq_constr], options={'maxiter':max_iter,'disp': 1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
    except:
        print('Exception made for {}, hard bounds: {}, analytical der.: {}, scaling: {}'.format(optimization_method,stay_within_bounds,derivatives,scale_var))
        message = 'An error occured during minimization'
        if optimization_method == 'trust-constr':
            if len(f_vec)>1:
                del f_vec[-1] #callback gets called in the iteration where it goes wrong, but does not get the updated (current) value of x. Hence, this final version is the same is the value of the previous (complete) iteration, despite x already being changed / updated during the final iterations. It is this updated x which causes an error.
                del x0_vec[-1]
                del x1_vec[-1]
            res = spo.OptimizeResult({'success':0,'x':np.array([x0_vec[-1],x1_vec[-1]]),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':message})
        elif optimization_method == 'SLSQP':
            res = spo.OptimizeResult({'success':0,'x':np.array([x0_vec[-1],x1_vec[-1]]),'nit':len(f_vec),'nfev':len(f_vec_global),'njev':'-','nhev':'-','message':message})
        elif optimization_method == 'ipopt':
            if len(f_vec_global) == 0:
                obj(x0)
                nit = 0
            else:
                nit = len(f_vec_global) #not correct number of iterations, but I need some value!
            res = spo.OptimizeResult({'success':0,'x':np.array([x0_f_global[-1],x1_f_global[-1]]),'nit':nit,'nfev':len(f_vec_global),'njev':'-','nhev':'-','message':message})
    if optimization_method == 'ipopt':
        if res.nit > 0:
            x0_vec = [x0_f_global[ind] for ind in range(0,len(x0_f_global),round(len(x0_f_global)/res.nit))]
            x1_vec = [x1_f_global[ind] for ind in range(0,len(x1_f_global),round(len(x1_f_global)/res.nit))]
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            x0_vec = x0_f_global
            x1_vec = x1_f_global
            f_vec = f_vec_global

    return res, f_vec, x0_vec, x1_vec

def optimize_substituted(x0,tol=1e-6,max_iter=150,optimization_method='trust-constr',stay_within_bounds=True,a=3,b=-4,c=2,alpha=x_min[0]/np.sqrt(x_min[1]),scale_var=None,scale_var_params=None,lb=np.array([0.1,1*bar]),ub=np.array([5,10*bar]),approach='direct'):
    """Minimize the Rosenbrock banana function. The nonlinear equality constraint is substituted to eliminate one of the variables.

    Parameters
    ----------
    x0 : np array
        Initial guess for the optimizer. caled when per unit scaling is used, unscaled otherwise.
    alpha : float, optional
        Parameters. Scaled when per unit scaling is used, unscaled otherwise. Default is 1.
    method : str, optional
        Which optimization method to use. Options are 'trust-constr', 'SLSQP', or 'ipopt'. Default is 'trust-constr'
    """
    # bounds (the optimizer uses the scaled x, so the bounds need to be scaled as well)
    lb_bounds = np.array([lb[1]])
    ub_bounds = np.array([ub[1]])
    if optimization_method == 'ipopt':
        bounds = [(lb,ub) for lb, ub in zip(lb_bounds,ub_bounds)]
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
    else:
        bounds = spo.Bounds(lb_bounds,ub_bounds,keep_feasible=stay_within_bounds)

    print('\nOptimizing using {}, with x0={}, alpha={}, hard bounds: {}, scaling: {}, approach: {}'.format(optimization_method,x0,alpha,stay_within_bounds,scale_var,approach))

    # set value of alpha in objective and constraints
    def obj(x,a=a,b=b,c=c,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params):
        # if scale_var == 'matrix':
        #     # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
        #     x = Dx_inv.dot(x)
        x0 = solve_h(x,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params)
        y = np.array([x0,x[0]])
        return f(y,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params)
    def obj_grad(x,a=a,b=b,c=c,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params,approach=approach):
        # if scale_var == 'matrix':
        #     # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
        #     x = Dx_inv.dot(x)
        x0 = solve_h(x,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params)
        y = np.array([x0,x[0]])
        parh_parx = h_der(y,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params)
        parf_parx = f_der(y,a=a,b=b,c=c,scale_var=scale_var,scale_var_params=scale_var_params)
        df_dx = np.zeros(len(x))
        if approach == 'direct':
            v = -parh_parx[1]/parh_parx[0]
            df_dx[0] = parf_parx[0]*v + parf_parx[1]
        elif approach == 'adjoint':
            lam = parf_parx[0]/parh_parx[0]
            df_dx[0] = -lam*parh_parx[1] + parf_parx[1]
        return df_dx
    def g(x,a=a,b=b,c=c,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params,lb=lb,ub=ub):
        """The (nonlinear) inequality constraints"""
        x0 = solve_h(x,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params)
        g = np.array([x0-lb[0],ub[0]-x0])
        return g
    def g_jac(x,a=a,b=b,c=c,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params,lb=lb,ub=ub):
        """First derivatives of the (nonlinear) inequality constraints"""
        x0 = solve_h(x,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params)
        y = np.array([x0,x[0]])
        parh_parx = h_der(y,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params)
        parg_parx = np.array([[1,0],[-1,0]])
        dg_dx = np.zeros((2,len(x)))
        if approach == 'direct':
            v = -parh_parx[1]/parh_parx[0]
            dg_dx[:,0] = parg_parx[:,0]*v + parg_parx[:,1]
        elif approach == 'adjoint':
            mu = np.transpose(parg_parx[:,0])/parh_parx[0]
            dg_dx[:,0] = -np.transpose(mu)*parh_parx[1] + parg_parx[:,1]
        return dg_dx

    # inequality constraints
    if optimization_method == 'trust-constr':
        ineq_constr = spo.NonlinearConstraint(g,np.zeros((2)),np.infty*np.ones((2)),jac=g_jac,keep_feasible=stay_within_bounds)
    elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        ineq_constr  = {'type':'ineq','fun':g,'jac':g_jac}

    # callback
    f_vec = list()
    x0_vec = list()
    x1_vec = list()
    global x0_f_global
    global x1_f_global
    global f_vec_global
    x0_f_global = list()
    x1_f_global = list()
    f_vec_global = list()
    if optimization_method == 'trust-constr':
        nit = 0
        nfev = 0
        njev = 0
        nhev = 0
        def callback(xk,state):
            """Called after every iteration"""
            f_vec.append(state.fun)
            x0_vec.append(solve_h(xk,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params))
            x1_vec.append(xk[0])
            nit = state.nit
            nfev = state.nfev
            njev = state.njev
            nhev = state.nhev
            return False
    elif optimization_method == 'SLSQP':
        f_vec.append(obj(x0))
        x0_vec.append(solve_h(x0,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params))
        x1_vec.append(x0[0])
        def callback(xk):
            """Called after every iteration"""
            f_vec.append(obj(xk))
            x0_vec.append(solve_h(xk,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params))
            x1_vec.append(xk[0])
            return False
    elif optimization_method == 'ipopt':
        # callback is not implemented in the ipopt (cyipopt) package / wrapper.
        pass

    # solve minimization
    try:
        if optimization_method == 'trust-constr':
            res = spo.minimize(obj, x0, method=optimization_method, jac=obj_grad, constraints=[ineq_constr], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol, callback=callback)
        elif optimization_method == 'SLSQP':
            res = spo.minimize(obj, x0, method=optimization_method, jac=obj_grad, constraints=[ineq_constr], options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
        elif optimization_method == 'ipopt':
            res = ipopt.minimize_ipopt(obj, x0, jac=obj_grad, constraints=[ineq_constr], options={'maxiter':max_iter,'disp': 1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
    except:
        print('Exception made for {}, hard bounds: {}, approach: {}, scaling: {}'.format(optimization_method,stay_within_bounds,approach,scale_var))
        message = 'An error occured during minimization'
        if optimization_method == 'trust-constr':
            if len(f_vec)>1:
                del f_vec[-1] #callback gets called in the iteration where it goes wrong, but does not get the updated (current) value of x. Hence, this final version is the same is the value of the previous (complete) iteration, despite x already being changed / updated during the final iterations. It is this updated x which causes an error.
                del x0_vec[-1]
                del x1_vec[-1]
            res = spo.OptimizeResult({'success':0,'x':np.array([x0_vec[-1]]),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':message})
        elif optimization_method == 'SLSQP':
            res = spo.OptimizeResult({'success':0,'x':np.array([x0_vec[-1]]),'nit':len(f_vec),'nfev':len(f_vec_global),'njev':'-','nhev':'-','message':message})
        elif optimization_method == 'ipopt':
            if len(f_vec_global) == 0:
                obj(x0)
                nit = 0
            else:
                nit = len(f_vec_global) #not correct number of iterations, but I need some value!
            res = spo.OptimizeResult({'success':0,'x':np.array([x0_f_global[-1]]),'nit':nit,'nfev':len(f_vec_global),'njev':'-','nhev':'-','message':message})

    # set final value for x0 in the global vectors
    solve_h(res.x,alpha=alpha,scale_var=scale_var,scale_var_params=scale_var_params)

    if optimization_method == 'ipopt':
        if res.nit > 0:
            x0_vec = [x0_f_global[ind] for ind in range(0,len(x0_f_global),round(len(x0_f_global)/res.nit))]
            x1_vec = [x1_f_global[ind] for ind in range(0,len(x1_f_global),round(len(x1_f_global)/res.nit))]
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            x0_vec = x0_f_global
            x1_vec = x1_f_global
            f_vec = f_vec_global

    return res, f_vec, x0_vec, x1_vec


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

def exp_format(f):
    s = '{:e}'.format(f)
    mantissa, exp = s.split('e')
    return str(int(exp))

def inspect_problem(x0 = np.linspace(-0.05,5.5,200),a=3,b=-4,c=2,alpha=x_min[0]/np.sqrt(x_min[1]),lb=np.array([0.1,1*bar]),ub=np.array([5,10*bar]),fig_obj_num='objective_function',fig_constr_num='constraints'):
    """Investigate the properties of the objective function and the constraints, etc. """
    fig_obj = plt.figure(fig_obj_num)
    ax_obj = fig_obj.gca()
    f_vec = np.zeros(len(x0))
    for ind_x0,q in enumerate(x0):
        f_vec[ind_x0] = f(np.array([q,0]),a=a,b=b,c=c)
    ax_obj.plot(x0,f_vec,'-k')
    # ax_obj.plot([x_min[0]],[f(x_min,a=a,b=b,c=c)],'.r')
    ax_obj.set_xlabel(r'$x_0$')
    ax_obj.set_ylabel('f')

    fig_constr = plt.figure(fig_constr_num)
    ax_constr = fig_constr.gca()
    y0 = x0**2/alpha**2 # eq. constr
    ax_constr.semilogy(x0,y0,ls=linestyles_contraints.get('eq'),color='k')
    ax_constr.semilogy([lb[0],lb[0]],[lb[1],ub[1]],ls=linestyles_contraints.get('bound'),color='k')# lower bound x0
    ax_constr.semilogy([ub[0],ub[0]],[lb[1],ub[1]],ls=linestyles_contraints.get('bound'),color='k')# upper bound x0
    ax_constr.semilogy([lb[0],ub[0]],[lb[1],lb[1]],ls=linestyles_contraints.get('bound'),color='k')# lower bound x1
    ax_constr.semilogy([lb[0],ub[0]],[ub[1],ub[1]],ls=linestyles_contraints.get('bound'),color='k')# upper bound x1
    # ax_constr.semilogy([x_min[0]],[x_min[0]**2/alpha**2],'.r')
    ax_constr.fill_between([lb[0],ub[0]], [lb[1],lb[1]], [ub[1],ub[1]], color='grey', alpha=0.6)
    ax_constr.set_xlabel(r'$x_0$')
    ax_constr.set_ylabel(r'$x_1$')
    return ax_obj, ax_constr

def compare_optimization_pressure(dir_path=None,save_tables=False,save_figs=False):
    """Compare the different optimization methods, for different order of magnitude for pressure"""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')
    # set solver info
    tol = 1e-6
    max_iter = 50

    result = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']

    # parameters
    c = 2
    b = -2*c
    a = -1.5*(b+c)

    # orders of magnitude for x1 (i.e. the pressure)
    orders_x1 = [1,mbar,bar]

    for order in orders_x1:
        # bounds
        lb = np.array([0.1,1*order])
        ub = np.array([5,10*order])

        # initial guess for optimizer
        x_init = np.array([.1,8*order])

        x_sol = np.array([1,5*order])
        alpha=x_sol[0]/np.sqrt(x_sol[1])

        # create plots
        ax_obj, ax_constr = inspect_problem(a=a,b=b,c=c,alpha=alpha,lb=lb,ub=ub,fig_obj_num='objective_function_orderx1_'+exp_format(order),fig_constr_num='constraints_orderx1_'+exp_format(order))
        fig_f = plt.figure('objective_optimization_orderx1_'+exp_format(order))
        ax_f = fig_f.gca()
        ax_f.set_xlabel('iteration')
        ax_f.set_ylabel(r'$f$')
        fig_x0 = plt.figure('x0_optimization_orderx1_'+exp_format(order))
        ax_x0 = fig_x0.gca()
        ax_x0.set_xlabel('iteration')
        ax_x0.set_ylabel(r'$x_0$')
        fig_x1 = plt.figure('x1_optimization_orderx1_'+exp_format(order))
        ax_x1 = fig_x1.gca()
        ax_x1.set_xlabel('iteration')
        ax_x1.set_ylabel(r'$x_1$')

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
                    res, f_vec, x0_vec, x1_vec = optimize(x_init,tol=tol,max_iter=max_iter,optimization_method=method,stay_within_bounds=stay_within_bounds,derivatives=derivatives,a=a,b=b,c=c,alpha=alpha,lb=lb,ub=ub)
                    result[method+'_'+bound+'_'+der+'_'+exp_format(order)] = res
                    ax_obj.plot(x0_vec,f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_constr.plot(x0_vec,x1_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_x0.plot(x0_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_x1.plot(x1_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    max_fev = max(max_fev,len(f_vec))

        ax_constr.plot([x_sol[0]],[x_sol[1]],'.r')
        ax_f.plot([0,max_fev],[f(x_sol),f(x_sol)],':r')
        ax_x0.plot([0,max_fev],[lb[0],lb[0]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_x0.plot([0,max_fev],[ub[0],ub[0]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_x0.plot([0,max_fev],[x_sol[0],x_sol[0]],':r')
        ax_x1.plot([0,max_fev],[lb[1],lb[1]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_x1.plot([0,max_fev],[ub[1],ub[1]],ls=linestyles_contraints.get('bounds'),color='k')
        ax_x1.plot([0,max_fev],[x_sol[1],x_sol[1]],':r')
        ax_f.legend(handles=legend_handles)
        ax_x0.legend(handles=legend_handles)
        ax_x1.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'Tables','Optimization_pipe')
        for bound in bounds:
            for der in ders:
                with open(os.path.join(path_to_tables,'optimizer_info_'+bound+'_'+der+'.txt'), "w") as table:
                    for order in orders_x1:
                        x_sol = np.array([1,5*order])
                        res_trust = result.get('trust-constr_'+bound+'_'+der+'_'+exp_format(order))
                        res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_'+exp_format(order))
                        res_ipopt = result.get('ipopt_'+bound+'_'+der+'_'+exp_format(order))
                        table.write(r'{} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(exp_format(order),res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(res_trust.x,x_sol),error(res_slsqp.x,x_sol),error(res_ipopt.x,x_sol)))
    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','Optimization_pipe')
        for fig_num in plt.get_figlabels():
            if not '3d' in fig_num:
                plt.figure(fig_num)
                file_name = fig_num+'.pgf'
                plt.savefig(os.path.join(path_to_fig, file_name))

def optimization_out_of_bound(dir_path=None,save_tables=False,save_figs=False):
    """Compare the different optimization methods, forcing the variables out-of-bound to where the model is not well-defined."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')
    # set solver info
    tol = 1e-6
    max_iter_soft_constr = 20
    max_iter_hard_constr = 50 # I've tried up to 5000, but trust-constr just seems to be stuck. It find the right solution for x1, but it somehow doesn't update x0 anymore. After 309 iterations it terminates based on xtol. It's optimality is very small, maybe that's the problem

    result = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']

    # parameters
    c = 20
    b = 2*c
    a = b-.5*c

    # bounds
    lb = np.array([0,1e-3])
    ub = np.array([50,10])

    # solution and parameter for equality constraint
    x_sol = np.array([.5,lb[1]])
    alpha=x_sol[0]/np.sqrt(x_sol[1])

    # initial guess for optimizer. Make sure it is feasible, i.e. it is on the equality constraint and within bounds
    x_inits = np.array([[alpha*np.sqrt(9),9],[9,9**2/alpha**2],[.3,2e-3],[9,9]])

    for ind_init in range(x_inits.shape[0]):
        x_init = x_inits[ind_init,:]
        for bound in bounds:
            if bound == 'soft':
                stay_within_bounds = False
                max_iter = max_iter_soft_constr
            else:
                stay_within_bounds = True
                max_iter = max_iter_hard_constr
            # create plots
            max_fev = 0
            x0_limits = [0,0]
            x1_limits = [0,0]
            ax_obj, ax_constr = inspect_problem(x0=np.linspace(lb[0]-.5,ub[0]+.5,1000),a=a,b=b,c=c,alpha=alpha,lb=lb,ub=ub,fig_obj_num='objective_function_oob_'+bound+'_constr'+'_xinit'+str(ind_init),fig_constr_num='constraints_oob_'+bound+'_constr'+'_xinit'+str(ind_init))
            fig_f = plt.figure('objective_optimization_oob_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_f = fig_f.gca()
            ax_f.set_xlabel('iteration')
            ax_f.set_ylabel(r'$f$')
            fig_x0 = plt.figure('x0_optimization_oob_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x0 = fig_x0.gca()
            ax_x0.set_xlabel('iteration')
            ax_x0.set_ylabel(r'$x_0$')
            fig_x1 = plt.figure('x1_optimization_oob_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x1 = fig_x1.gca()
            ax_x1.set_xlabel('iteration')
            ax_x1.set_ylabel(r'$x_1$')
            fig_x0_h = plt.figure('x0_h_oob_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x0_h = fig_x0_h.gca()
            ax_x0_h.set_xlabel('function call to equality constraint')
            ax_x0_h.set_ylabel(r'$x_0$')
            fig_x1_h = plt.figure('x1_h_oob_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x1_h = fig_x1_h.gca()
            ax_x1_h.set_xlabel('function call to equality constraint')
            ax_x1_h.set_ylabel(r'$x_1$')
            fig_x0_der_h = plt.figure('x0_der_h_oob_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x0_der_h = fig_x0_der_h.gca()
            ax_x0_der_h.set_xlabel('function call to gradient of equality constraint')
            ax_x0_der_h.set_ylabel(r'$x_0$')
            fig_x1_der_h = plt.figure('x1_der_h_oob_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x1_der_h = fig_x1_der_h.gca()
            ax_x1_der_h.set_xlabel('function call to gradient of equality constraint')
            ax_x1_der_h.set_ylabel(r'$x_1$')
            for method in methods:
                for der in ders:
                    if der == 'an':
                        derivatives = True
                    else:
                        derivatives = False
                    global x0_h_global
                    global x1_h_global
                    x0_h_global = list()
                    x1_h_global = list()
                    global x0_der_h_global
                    global x1_der_h_global
                    x0_der_h_global = list()
                    x1_der_h_global = list()
                    res, f_vec, x0_vec, x1_vec = optimize(x_init,tol=tol,max_iter=max_iter,optimization_method=method,stay_within_bounds=stay_within_bounds,derivatives=derivatives,a=a,b=b,c=c,alpha=alpha,lb=lb,ub=ub)
                    result[method+'_'+bound+'_'+der+'_'+str(ind_init)] = res
                    ax_obj.plot(x0_vec,f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_constr.plot(x0_vec,x1_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_x0.plot(x0_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_x1.plot(x1_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_x0_h.plot(x0_h_global,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_x1_h.plot(x1_h_global,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_x0_der_h.plot(x0_der_h_global,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    ax_x1_der_h.plot(x1_der_h_global,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(der))
                    max_fev = max(max_fev,len(f_vec))
                    print('x0 = {}'.format(x0_vec))
                    x0_limits[0] = min(x0_limits[0],np.min(x0_vec))
                    x0_limits[1] = max(x0_limits[1],np.max(x0_vec))
                    x1_limits[0] = min(x1_limits[0],np.min(x1_vec))
                    x1_limits[1] = max(x1_limits[1],np.max(x1_vec))

            ax_obj.plot([x_sol[0]],[f(x_sol,a=a,b=b,c=c)],'.r')
            ax_obj.set_xlim(left=x0_limits[0]-.1,right=x0_limits[1]+1)
            ax_obj.set_ylim(bottom=f(np.array([x0_limits[0]-.1,0]),a=a,b=b,c=c),top=f(np.array([x0_limits[1]+1,0]),a=a,b=b,c=c))
            ax_constr.plot([x_sol[0]],[x_sol[1]],'.r')
            ax_constr.set_xlim(left=x0_limits[0]-.1,right=x0_limits[1]+1)
            ax_constr.set_ylim(bottom=np.max([x1_limits[0]-.1,5e-4]),top=x1_limits[1]+10)
            ax_f.plot([0,max_fev],[f(x_sol,a=a,b=b,c=c),f(x_sol,a=a,b=b,c=c)],':r')
            ax_x0.plot([0,max_fev],[lb[0],lb[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0.plot([0,max_fev],[ub[0],ub[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0.plot([0,max_fev],[x_sol[0],x_sol[0]],':r')
            ax_x1.plot([0,max_fev],[lb[1],lb[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1.plot([0,max_fev],[ub[1],ub[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1.plot([0,max_fev],[x_sol[1],x_sol[1]],':r')
            x0_xlims = ax_x0_h.get_xlim()
            x0_ylims = ax_x0_h.get_ylim()
            ax_x0_h.plot(x0_xlims,[lb[0],lb[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0_h.plot(x0_xlims,[ub[0],ub[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0_h.plot(x0_xlims,[x_sol[0],x_sol[0]],':r')
            ax_x0_h.set_ylim(bottom=x0_ylims[0],top=x0_ylims[1]) # in case the bounds are far away from iterates
            x1_xlims = ax_x1_h.get_xlim()
            x1_ylims = ax_x1_h.get_ylim()
            ax_x1_h.plot(x1_xlims,[lb[1],lb[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1_h.plot(x1_xlims,[ub[1],ub[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1_h.plot(x1_xlims,[x_sol[1],x_sol[1]],':r')
            ax_x1_h.set_ylim(bottom=x1_ylims[0],top=x1_ylims[1]) # in case the bounds are far away from iterates
            x0_der_xlims = ax_x0_der_h.get_xlim()
            x0_der_ylims = ax_x0_der_h.get_ylim()
            ax_x0_der_h.plot(x0_der_xlims,[lb[0],lb[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0_der_h.plot(x0_der_xlims,[ub[0],ub[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0_der_h.plot(x0_der_xlims,[x_sol[0],x_sol[0]],':r')
            ax_x0_der_h.set_ylim(bottom=x0_der_ylims[0],top=x0_der_ylims[1]) # in case the bounds are far away from iterates
            x1_der_xlims = ax_x1_der_h.get_xlim()
            x1_der_ylims = ax_x1_der_h.get_ylim()
            ax_x1_der_h.plot(x1_der_xlims,[lb[1],lb[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1_der_h.plot(x1_der_xlims,[ub[1],ub[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1_der_h.plot(x1_der_xlims,[x_sol[1],x_sol[1]],':r')
            ax_x1_der_h.set_ylim(bottom=x1_der_ylims[0],top=x1_der_ylims[1]) # in case the bounds are far away from iterates
            ax_f.legend(handles=legend_handles)
            ax_x0.legend(handles=legend_handles)
            ax_x1.legend(handles=legend_handles)
            ax_x0_h.legend(handles=legend_handles)
            ax_x1_h.legend(handles=legend_handles)
            ax_x0_der_h.legend(handles=legend_handles)
            ax_x1_der_h.legend(handles=legend_handles)

    if save_tables:
        path_to_tables = os.path.join(dir_path,'Tables','Optimization_pipe')
        for ind_init in range(x_inits.shape[0]):
            with open(os.path.join(path_to_tables,'optimizer_info_oob_xinit'+str(ind_init)+'.txt'), "w") as table:
                for bound in bounds:
                    for der in ders:
                        res_trust = result.get('trust-constr_'+bound+'_'+der+'_'+str(ind_init))
                        res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_'+str(ind_init))
                        res_ipopt = result.get('ipopt_'+bound+'_'+der+'_'+str(ind_init))
                        table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(res_trust.x,x_sol),error(res_slsqp.x,x_sol),error(res_ipopt.x,x_sol)))
    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','Optimization_pipe')
        for fig_num in plt.get_figlabels():
            if not '3d' in fig_num:
                plt.figure(fig_num)
                file_name = fig_num+'.pgf'
                plt.savefig(os.path.join(path_to_fig, file_name))

def optimization_out_of_bound_substituted(dir_path=None,save_tables=False,save_figs=False):
    """Compare the different optimization methods, forcing the variables out-of-bound to where the model is not well-defined. The equality constraint is substituted to reduce the dimension of the variables."""
    if (save_tables or save_figs) and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')
    # set solver info
    tol = 1e-6
    max_iter_soft_constr = 20
    max_iter_hard_constr = 50 # I've tried up to 5000, but trust-constr just seems to be stuck. It find the right solution for x1, but it somehow doesn't update x0 anymore. After 309 iterations it terminates based on xtol. It's optimality is very small, maybe that's the problem

    result = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']

    # parameters
    c = 20
    b = 2*c
    a = b-.5*c

    # bounds
    lb = np.array([0,1e-3])
    ub = np.array([50,10])

    # solution and parameter for equality constraint
    x_sol = np.array([lb[1]])
    x0_sol = .5
    x_sol_vec = np.array([x0_sol,x_sol[0]])
    alpha=x0_sol/np.sqrt(x_sol[0])

    # initial guess for optimizer. Make sure it is feasible, i.e. it is on the equality constraint and within bounds
    x_inits = np.array([[9],[9**2/alpha**2],[2e-3]])

    for ind_init in range(x_inits.shape[0]):
        x_init = x_inits[ind_init,:]
        for bound in bounds:
            if bound == 'soft':
                stay_within_bounds = False
                max_iter = max_iter_soft_constr
            else:
                stay_within_bounds = True
                max_iter = max_iter_hard_constr
            # create plots
            max_fev = 0
            x0_limits = [0,0]
            x1_limits = [0,0]
            ax_obj, ax_constr = inspect_problem(x0=np.linspace(lb[0]-.5,ub[0]+.5,1000),a=a,b=b,c=c,alpha=alpha,lb=lb,ub=ub,fig_obj_num='objective_function_oob_substituted_'+bound+'_constr'+'_xinit'+str(ind_init),fig_constr_num='constraints_oob_substituted_'+bound+'_constr'+'_xinit'+str(ind_init))
            fig_f = plt.figure('objective_optimization_oob_substituted_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_f = fig_f.gca()
            ax_f.set_xlabel('iteration')
            ax_f.set_ylabel(r'$f$')
            fig_x0 = plt.figure('x0_optimization_oob_substituted_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x0 = fig_x0.gca()
            ax_x0.set_xlabel('iteration')
            ax_x0.set_ylabel(r'$x_0$')
            fig_x1 = plt.figure('x1_optimization_oob_substituted_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x1 = fig_x1.gca()
            ax_x1.set_xlabel('iteration')
            ax_x1.set_ylabel(r'$x_1$')
            fig_x0_h = plt.figure('x0_h_oob_substituted_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x0_h = fig_x0_h.gca()
            ax_x0_h.set_xlabel('function call to equality constraint')
            ax_x0_h.set_ylabel(r'$x_0$')
            fig_x1_h = plt.figure('x1_h_oob_substituted_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x1_h = fig_x1_h.gca()
            ax_x1_h.set_xlabel('function call to equality constraint')
            ax_x1_h.set_ylabel(r'$x_1$')
            fig_x0_der_h = plt.figure('x0_der_h_oob_substituted_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x0_der_h = fig_x0_der_h.gca()
            ax_x0_der_h.set_xlabel('function call to gradient of equality constraint')
            ax_x0_der_h.set_ylabel(r'$x_0$')
            fig_x1_der_h = plt.figure('x1_der_h_oob_substituted_'+bound+'_constr'+'_xinit'+str(ind_init))
            ax_x1_der_h = fig_x1_der_h.gca()
            ax_x1_der_h.set_xlabel('function call to gradient of equality constraint')
            ax_x1_der_h.set_ylabel(r'$x_1$')
            for method in methods:
                for approach in approaches:
                    global x0_h_global
                    global x1_h_global
                    x0_h_global = list()
                    x1_h_global = list()
                    global x0_der_h_global
                    global x1_der_h_global
                    x0_der_h_global = list()
                    x1_der_h_global = list()
                    if approach == 'direct' or approach == 'adjoint':
                        res, f_vec, x0_vec, x1_vec = optimize_substituted(x_init,tol=tol,max_iter=max_iter,optimization_method=method,stay_within_bounds=stay_within_bounds,approach=approach,a=a,b=b,c=c,alpha=alpha,lb=lb,ub=ub)
                    else:
                        res, f_vec, x0_vec, x1_vec = optimize(np.array([solve_h(x_init,alpha=alpha),x_init]),tol=tol,max_iter=max_iter,optimization_method=method,stay_within_bounds=stay_within_bounds,derivatives=True,a=a,b=b,c=c,alpha=alpha,lb=lb,ub=ub)
                    result[method+'_'+bound+'_'+approach+'_'+str(ind_init)] = res
                    ax_obj.plot(x0_vec,f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_approaches.get(approach))
                    ax_constr.plot(x0_vec,x1_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_approaches.get(approach))
                    ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_approaches.get(approach))
                    ax_x0.plot(x0_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_approaches.get(approach))
                    ax_x1.plot(x1_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_approaches.get(approach))
                    ax_x0_h.plot(x0_h_global,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_approaches.get(approach))
                    ax_x1_h.plot(x1_h_global,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_approaches.get(approach))
                    ax_x0_der_h.plot(x0_der_h_global,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_approaches.get(approach))
                    ax_x1_der_h.plot(x1_der_h_global,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_approaches.get(approach))
                    max_fev = max(max_fev,len(f_vec))
                    print('x0 = {}'.format(x0_vec))
                    x0_limits[0] = min(x0_limits[0],np.min(x0_vec))
                    x0_limits[1] = max(x0_limits[1],np.max(x0_vec))
                    x1_limits[0] = min(x1_limits[0],np.min(x1_vec))
                    x1_limits[1] = max(x1_limits[1],np.max(x1_vec))

            ax_obj.plot([x_sol_vec[0]],[f(x_sol_vec,a=a,b=b,c=c)],'.r')
            ax_obj.set_xlim(left=x0_limits[0]-.1,right=x0_limits[1]+1)
            ax_obj.set_ylim(bottom=f(np.array([x0_limits[0]-.1,0]),a=a,b=b,c=c),top=f(np.array([x0_limits[1]+1,0]),a=a,b=b,c=c))
            ax_constr.plot([x_sol_vec[0]],[x_sol_vec[1]],'.r')
            ax_constr.set_xlim(left=x0_limits[0]-.1,right=x0_limits[1]+1)
            ax_constr.set_ylim(bottom=np.max([x1_limits[0]-.1,5e-4]),top=x1_limits[1]+10)
            ax_f.plot([0,max_fev],[f(x_sol_vec,a=a,b=b,c=c),f(x_sol_vec,a=a,b=b,c=c)],':r')
            ax_x0.plot([0,max_fev],[lb[0],lb[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0.plot([0,max_fev],[ub[0],ub[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0.plot([0,max_fev],[x_sol_vec[0],x_sol_vec[0]],':r')
            ax_x1.plot([0,max_fev],[lb[1],lb[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1.plot([0,max_fev],[ub[1],ub[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1.plot([0,max_fev],[x_sol_vec[1],x_sol_vec[1]],':r')
            x0_xlims = ax_x0_h.get_xlim()
            x0_ylims = ax_x0_h.get_ylim()
            ax_x0_h.plot(x0_xlims,[lb[0],lb[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0_h.plot(x0_xlims,[ub[0],ub[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0_h.plot(x0_xlims,[x_sol_vec[0],x_sol_vec[0]],':r')
            ax_x0_h.set_ylim(bottom=x0_ylims[0],top=x0_ylims[1]) # in case the bounds are far away from iterates
            x1_xlims = ax_x1_h.get_xlim()
            x1_ylims = ax_x1_h.get_ylim()
            ax_x1_h.plot(x1_xlims,[lb[1],lb[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1_h.plot(x1_xlims,[ub[1],ub[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1_h.plot(x1_xlims,[x_sol_vec[1],x_sol_vec[1]],':r')
            ax_x1_h.set_ylim(bottom=x1_ylims[0],top=x1_ylims[1]) # in case the bounds are far away from iterates
            x0_der_xlims = ax_x0_der_h.get_xlim()
            x0_der_ylims = ax_x0_der_h.get_ylim()
            ax_x0_der_h.plot(x0_der_xlims,[lb[0],lb[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0_der_h.plot(x0_der_xlims,[ub[0],ub[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x0_der_h.plot(x0_der_xlims,[x_sol_vec[0],x_sol_vec[0]],':r')
            ax_x0_der_h.set_ylim(bottom=x0_der_ylims[0],top=x0_der_ylims[1]) # in case the bounds are far away from iterates
            x1_der_xlims = ax_x1_der_h.get_xlim()
            x1_der_ylims = ax_x1_der_h.get_ylim()
            ax_x1_der_h.plot(x1_der_xlims,[lb[1],lb[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1_der_h.plot(x1_der_xlims,[ub[1],ub[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_x1_der_h.plot(x1_der_xlims,[x_sol_vec[1],x_sol_vec[1]],':r')
            ax_x1_der_h.set_ylim(bottom=x1_der_ylims[0],top=x1_der_ylims[1]) # in case the bounds are far away from iterates
            ax_f.legend(handles=legend_handles_substituted)
            ax_x0.legend(handles=legend_handles_substituted)
            ax_x1.legend(handles=legend_handles_substituted)
            ax_x0_h.legend(handles=legend_handles_substituted)
            ax_x1_h.legend(handles=legend_handles_substituted)
            ax_x0_der_h.legend(handles=legend_handles_substituted)
            ax_x1_der_h.legend(handles=legend_handles_substituted)

    # legend
    fig_legend = plt.figure('Legend_oob_substituted')
    ax_legend = fig_legend.gca()
    ax_legend.axis('off')
    fig_legend.patch.set_visible(False)
    ax_legend.legend(handles=legend_handles_substituted,loc='center')

    if save_tables:
        path_to_tables = os.path.join(dir_path,'Tables','Optimization_pipe')
        for ind_init in range(x_inits.shape[0]):
            with open(os.path.join(path_to_tables,'optimizer_info_oob_substituted_xinit'+str(ind_init)+'.txt'), "w") as table:
                for bound in bounds:
                    for approach in approaches:
                        if approach == 'eq_constr':
                            approach_label = 'eq. constr.'
                            x_opt_sol = x_sol_vec.copy()
                        else:
                            approach_label = approach
                            x_opt_sol = x_sol.copy()
                        res_trust = result.get('trust-constr_'+bound+'_'+approach+'_'+str(ind_init))
                        res_slsqp = result.get('SLSQP_'+bound+'_'+approach+'_'+str(ind_init))
                        res_ipopt = result.get('ipopt_'+bound+'_'+approach+'_'+str(ind_init))
                        table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(bound,approach_label,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(res_trust.x,x_opt_sol),error(res_slsqp.x,x_opt_sol),error(res_ipopt.x,x_opt_sol)))
                    table.write('\hline ')
    for ind_init in range(x_inits.shape[0]):
        for bound in bounds:
            for approach in approaches:
                if approach == 'eq_constr':
                    x_opt_sol = x_sol_vec.copy()
                else:
                    x_opt_sol = x_sol.copy()
                res_trust = result.get('trust-constr_'+bound+'_'+approach+'_'+str(ind_init))
                res_slsqp = result.get('SLSQP_'+bound+'_'+approach+'_'+str(ind_init))
                res_ipopt = result.get('ipopt_'+bound+'_'+approach+'_'+str(ind_init))
                print('\nx init:{}, bounds: {}, approach: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\ntrust-constr:{}\nSLSQP: {}\nIPOPT: {}\nErrors for t-c: {}, SLSQP: {}, IPOPT: {}'.format(x_inits[ind_init,:],bound,approach,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(res_trust.x,x_opt_sol),error(res_slsqp.x,x_opt_sol),error(res_ipopt.x,x_opt_sol)))

    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','Optimization_pipe')
        for fig_num in plt.get_figlabels():
            if not '3d' in fig_num:
                plt.figure(fig_num)
                file_name = fig_num+'.pgf'
                plt.savefig(os.path.join(path_to_fig, file_name))

if __name__ == '__main__':
    x0_h_global = list()
    x1_h_global = list()
    x0_der_h_global = list()
    x1_der_h_global = list()
    f_vec_global = list()
    x0_f_global = list()
    x1_f_global = list()
    # inspect_problem()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # compare_optimization_pressure(dir_path=dir_path,save_tables=False,save_figs=False)
    # optimization_out_of_bound(dir_path=dir_path,save_tables=False,save_figs=False)
    optimization_out_of_bound_substituted(dir_path=dir_path,save_tables=False,save_figs=False)

    plt.show()
