"""Network with 3 nodes, 2 of which are sources, to be able to run OPF.

The nodes are connected in a triangle. The pipes connecting the links are all the same.
"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink
from meslf.networks.carrier import Gas
from meslf.load_flow.system_of_equations import NonLinearSystemGas
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import os
from meslf.utils.hide_print import HiddenPrints
import ipopt

def create_network(p0=50,q1=-100,q2=250,link_type='pipe_linear',link_params={'alpha':10}):
    """Create the gas network, consisting of 3 nodes. Nodes 0 and 1 are sources, node 2 is a sink."""
    gas_net = GasNetwork('G3N')
    n0 = GasNode('gn0',node_type=0,x=0,y=0,p=p0) # slack node
    n1 = GasNode('gn1',node_type=1,x=2,y=0,q=q1) # load node (source)
    n2 = GasNode('gn2',node_type=1,x=1,y=-1,q=q2) # load node (sink)

    l0 = GasLink('gl0',n0,n1,link_type=link_type,link_params=link_params.copy())
    l1 = GasLink('gl1',n0,n2,link_type=link_type,link_params=link_params.copy())
    l2 = GasLink('gl2',n1,n2,link_type=link_type,link_params=link_params.copy())

    gas_net.add_link(l0)
    gas_net.add_link(l1)
    gas_net.add_link(l2)
    return gas_net

def update_bc(gas_net,p0,q1):
    """Update the boundary conditions of the gas network"""
    gas_net.nodes[0].p = p0
    gas_net.nodes[1].half_links[0].q = q1
    return gas_net

def initialize_network(network,p1=40,p2=30,q01=40,q02=150,q12=120,scale_var=None,scale_var_params=None,formulation='full'):
    if formulation == 'full':
        q_init = np.array([q01,q02,q12])
    else:
        q_init = np.array([])
    p_init = np.array([p1,p2])
    x_init = np.concatenate((q_init,p_init))
    network.initialize()
    network.update(x_init,formulation=formulation) # update without scaling, since x_init is unscaled
    x0 = network.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def run_load_flow(p0=50,q1=-100,q2=250,link_type='pipe_linear',link_params={'alpha':10},p1=40,p2=30,q01=40,q02=150,q12=120,scale_var=None,scale_var_params=None,formulation='full',tol=1e-6,max_iter=150):
    """Stead-state load flow analysis of gas network, using matrix scaling

    Parameters
    ----------

    Returns
    -------

    """
    print('\nRunning steady-state load flow for gas network')
    # create network
    gas_net = create_network(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params)
    # initialize
    x0 = initialize_network(gas_net,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)

    # solve network
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('Error is {:.4e}, after {} iterations'.format(err_vec[-1],iters))
    print('Solution:')
    print('p = {} Pa'.format(p_sol))
    print('q = {} kg/s'.format(q_sol))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    return gas_net,x_sol,iters,err_vec,p_sol,q_sol,q_inj,tol

def xg_from_xopf(x_opf,p_BC=False):
    if p_BC:
        xg = x_opf[2:]
    else:
        xg = x_opf[3:]
    return xg

def run_optimal_load_flow(p0=50,q1=-100,q2=250,link_type='pipe_linear',link_params={'alpha':10},p1=40,p2=30,q01=40,q02=150,q12=120,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=None,scale_var_params=None,formulation='full',tol=1e-6,max_iter=150,lb_ineq_state = np.array([-150,5,10,5,45,33]),ub_ineq_state = np.array([-50,100,200,150,50,38]),p_BC=False,ineq_constr='all',derivatives=False,optimization_method='trust-constr',stay_within_bounds=False):
    """Run optimal power flow.

    Parameters
    ----------------
    p_BC : bool, optional
        When True, the pressure of the slack node is taken as boundary condition for the OPF. If False, it is taken is control variables. Default is False.
    ineq_constr : str, optional
        Determines on which variables the inequality constraints are imposed. If 'all', the inequality constraints are imposed on both state and control variables. If 'control', the inequality constraints are only imposed on the control variables. Default is 'all'.
    derivatives : bool, optional
        If True, analytical expressions for the gradient and Hessian of the objective function and of the (nonlinear) constraints are used. Otherwise, numerical approximations are used. Default is False.
    """
    if formulation != 'full':
        raise ValueError('OPF not implemented for other formulation than full')
    if scale_var:
        raise ValueError('OPF not implemented when using scaling')
    if derivatives and not p_BC and ineq_constr == 'all':
        raise ValueError('OPF with analytical derivatives only implemented if p0 is BC and inequality contraints are only imposed on control variables.')
    print('\nRunning OPF for gas network (p0 as BC: {}, inequality constraints: {}, analytical derivatives: {})'.format(p_BC,ineq_constr,derivatives))
    # create network
    gas_net = create_network(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params)

    # update the boundary conditions of the gas network to match the initial guess of opf
    gas_net = update_bc(gas_net,p0,q1)
    # run load flow once, to make sure that the initial guess of opf is at least a solution of LF
    x0 = initialize_network(gas_net,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)
    x_LF,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    q0 = q_inj[0]
    # initial guess for opf
    if p_BC:
        u_init = np.array([q1])
        p0_BC = p0
    else:
        u_init = np.array([q1, p0])
    slack_init = np.array([q0])
    x_opf0 = np.concatenate((u_init,slack_init,x_LF))

    # keep track of variables in cost function
    global f_vec
    global x_f_vec
    f_vec = list()
    x_f_vec = list()

    # define cost function / objective function
    def cost_function(x_opf):
        """Define the cost function for OPF.

        Parameters
        ----------------
        x_opf : np array
            Variable vector used in OPF. Is assumed to be [q1 p0 q0 q01 q02 q12 p1 p2]

        Returns
        -----------
        f : float
            The value of the cost function
        """
        q1 = x_opf[0] #<0
        if p_BC:
            q0 = x_opf[1] #<0
        else:
            q0 = x_opf[2] #<0
        f = a0 + b0*-q0 + c0*q0**2  + a1 + b1*-q1 + c1*q1**2
        global f_vec
        global x_f_vec
        f_vec.append(x_opf)
        x_f_vec = x_opf.copy()
        return f
    if derivatives:
        # gradient and Hessian of cost function
        def jac_cost(x_opf):
            """Gradient vector / Jacobian of cost function"""
            df_dy = np.zeros(len(x_opf))
            q1 = x_opf[0] #<0
            if p_BC:
                q0_ind = 1
            else:
                q0_ind = 2
            q0 = x_opf[q0_ind] #<0
            df_dy[0]  = -b1 + 2*c1*q1
            df_dy[q0_ind] = -b0 + 2*c0*q0
            return df_dy
        def hess_cost(x_opf):
            """Hessian of cost function"""
            hess_cost_diag = np.zeros(len(x_opf))
            if p_BC:
                q0_ind = 1
            else:
                q0_ind = 2
            hess_cost_diag[0]  = -2*c1
            hess_cost_diag[q0_ind] = 2*c0
            return np.diag(hess_cost_diag)

    # define linear equality constraints (conservation of mass in slack node)
    A_eq = np.zeros((1,len(x_opf0)))
    if p_BC:
        A_eq[0,1] = -1
        A_eq[0,2] = -1
        A_eq[0,3] = -1
    else:
        A_eq[0,2] = -1
        A_eq[0,3] = -1
        A_eq[0,4] = -1
    if optimization_method == 'trust-constr':
        linear_constraint = spo.LinearConstraint(A_eq,[0],[0],keep_feasible=stay_within_bounds)
    elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
        linear_constraint  = {'type':'eq','fun': lambda x: A_eq.dot(x), 'jac': lambda x: A_eq}

    # define nonlinear equality constriants (load flow equations)
    nlsys = NonLinearSystemGas(gas_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    def load_flow_in_opf(x_opf,network=gas_net):
        # update bc of network
        q1 = x_opf[0]
        if p_BC:
            p0 = p0_BC
        else:
            p0 = x_opf[1]
        network = update_bc(network,p0,q1)
        # evaluate load flow equations
        xg = xg_from_xopf(x_opf,p_BC=p_BC)
        network.reset_network(xg,formulation=formulation)
        F = nlsys.F(xg)
        return F
    if derivatives:
        # gradient and Hessian of nonlinear (equality) contraints
        def jac_nleq(x_opf,network=gas_net):
            # update bc of network
            q1 = x_opf[0]
            if p_BC:
                p0 = p0_BC
            else:
                p0 = x_opf[1]
            network = update_bc(network,p0,q1)
            # evaluate load flow equations
            xg = xg_from_xopf(x_opf,p_BC=p_BC)
            network.reset_network(xg,formulation=formulation)
            J_lf = nlsys.J_dense(xg) #seems that minimize needs the Jacobian of all the constraints to be sparse or dense
            dF_dq1 = np.zeros(len(xg))
            dF_dq1[0] = -1
            dF_dq0 = np.zeros(len(xg))
            dF_dy = np.zeros((len(xg),len(x_opf)))
            dF_dy[:,0] = dF_dq1
            dF_dy[:,1] = dF_dq0
            dF_dy[:,2:] = J_lf
            return dF_dy
        def hess_nleq(x_opf, v):
            """Hessian of dot(h,v) with h(x_opf) the nonlinear constraints, and v a vector (of Lagrange multipliers).
            In this case, h(x_opf) are the steady-state load flow equations, which are linear. So the Hessian of every equation of h is zero.
            """
            return np.zeros((len(x_opf),len(x_opf)))
    lb_lf = np.zeros(len(x0))
    ub_lf = np.zeros(len(x0))
    if derivatives:
        if optimization_method == 'trust-constr':
            nonlinear_constraint = spo.NonlinearConstraint(load_flow_in_opf,lb_lf,ub_lf,jac=jac_nleq,hess=hess_nleq,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            nonlinear_constraint  = {'type':'eq','fun':load_flow_in_opf,'jac':jac_nleq}
    else:
        if optimization_method == 'trust-constr':
            nonlinear_constraint = spo.NonlinearConstraint(load_flow_in_opf,lb_lf,ub_lf,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            nonlinear_constraint  = {'type':'eq','fun':load_flow_in_opf}
    # define linear inequality constraints
    q1_lb = -200
    q1_ub = -110
    p0_lb = 48
    p0_ub = 52
    if ineq_constr == 'all':
        if p_BC:
            lb_ineq = np.concatenate((np.array([q1_lb]),lb_ineq_state))
            ub_ineq = np.concatenate((np.array([q1_ub]),ub_ineq_state))
        else:
            lb_ineq = np.concatenate((np.array([q1_lb,p0_lb]),lb_ineq_state))
            ub_ineq = np.concatenate((np.array([q1_ub,p0_ub]),ub_ineq_state))
    elif ineq_constr == 'control':
        lb_ineq = -np.inf*np.ones(len(x_opf0)) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
        ub_ineq = np.inf*np.ones(len(x_opf0))
        if p_BC:
            lb_ineq[:len(u_init)] = np.array([q1_lb])
            ub_ineq[:len(u_init)] = np.array([q1_ub])
        else:
            lb_ineq[:len(u_init)] = np.array([q1_lb,p0_lb])
            ub_ineq[:len(u_init)] = np.array([q1_ub,p0_ub])
    else:
        raise ValueError('Enter valid value for ineq_constr')
    if optimization_method == 'ipopt':
        bounds = [(lb,ub) for lb, ub in zip(lb_ineq,ub_ineq)]
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
    else:
        bounds = spo.Bounds(lb_ineq,ub_ineq,keep_feasible=stay_within_bounds)

    # solve OPF
    try:
        if derivatives:
            if optimization_method == 'trust-constr':
                res = spo.minimize(cost_function, x_opf0, method=optimization_method,jac=jac_cost,hess=hess_cost, constraints=[linear_constraint, nonlinear_constraint], options={'verbose': 1}, bounds=bounds)
            elif optimization_method == 'SLSQP':
                res = spo.minimize(cost_function, x_opf0, method=optimization_method,jac=jac_cost, constraints=[linear_constraint, nonlinear_constraint], options={'disp': True}, bounds=bounds)
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(cost_function, x_opf0, jac=jac_cost, constraints=[linear_constraint, nonlinear_constraint], options={'disp': 1,'bound_relax_factor':bound_relax_factor}, bounds=bounds)
        else:
            if optimization_method == 'trust-constr':
                res = spo.minimize(cost_function, x_opf0, method=optimization_method, constraints=[linear_constraint, nonlinear_constraint], options={'verbose': 1}, bounds=bounds)
            elif optimization_method == 'SLSQP':
                res = spo.minimize(cost_function, x_opf0, method=optimization_method, constraints=[linear_constraint, nonlinear_constraint], options={'disp': True,'ftol':tol}, bounds=bounds)
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(cost_function, x_opf0, constraints=[linear_constraint, nonlinear_constraint], options={'disp': 1,'bound_relax_factor':bound_relax_factor}, bounds=bounds)
    except:
        print('Exception made for {}, hard bounds: {}, analytical der.: {}'.format(optimization_method,stay_within_bounds,derivatives))
        if len(f_vec) == 0:
            cost_function(x_opf0)
            nit = 0
            nfev = 0
            njev = 0
            nhev = 0
        else:
            nit = 0
            nfev = len(f_vec)
            njev = 0
            nhev = 0
        res = spo.OptimizeResult({'success':False,'x':np.array(x_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})
    x_opf = res.x
    # print solution
    print('x_opf = {}'.format(x_opf))
    xg_opt = xg_from_xopf(x_opf,p_BC=p_BC)
    p_sol,q_sol,q_inj = gas_net.update_full(xg_opt,formulation=formulation)
    print('Optimal solution:')
    print('p = {} Pa'.format(p_sol))
    print('q = {} kg/s'.format(q_sol))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    return xg_opt, res

def test_opf_pbc():
    """Test OPF againts the solution of LF (the constraints and cost function parameteres are chosen such that this should be the case), using the pressure of the slack node as boundary condition, and imposing inequality constraints only on the control variables."""
    # Given
    p_BC = True
    ineq_constr = 'control'
    p0=50 #BC
    q1=-180 # Inital guess
    q1_sol = -110 #BC
    q2=250 #BC
    link_type='pipe_linear'
    link_params={'alpha':10}
    p1=40 # Initial guess
    p2=30 # Initial guess
    q01=40 # Initial guess
    q02=150 # Initial guess
    q12=120 # Initial guess
    formulation='full'
    tol=1e-6
    max_iter=150
    scale_var = None

    # When
    xg_opt, _,  = run_optimal_load_flow(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=p_BC,ineq_constr=ineq_constr)

    # Then
    _,xg_LF,_,_,_,_,_,_ = run_load_flow(p0=p0,q1=q1_sol,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)
    print('xg opt = \n{}\nxg LF - \n{}\ndifference = \n{}'.format(xg_opt,xg_LF,xg_opt-xg_LF))
    assert np.allclose(xg_opt,xg_LF)

def test_opf_pbc_derivates():
    """Test OPF againts the solution of LF (the constraints and cost function parameteres are chosen such that this should be the case), using analytical expressions for the gradients and hessians. The pressure of the slack node as boundary condition, and imposing inequality constraints only on the control variables."""
    # Given
    p_BC = True
    ineq_constr = 'control'
    derivatives = True
    p0=50 #BC
    q1=-180 # Inital guess
    q1_sol = -110 #BC
    q2=250 #BC
    link_type='pipe_linear'
    link_params={'alpha':10}
    p1=40 # Initial guess
    p2=30 # Initial guess
    q01=40 # Initial guess
    q02=150 # Initial guess
    q12=120 # Initial guess
    formulation='full'
    tol=1e-6
    max_iter=150
    scale_var = None

    # When
    xg_opt, _, = run_optimal_load_flow(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=p_BC,ineq_constr=ineq_constr,derivatives=derivatives)

    # Then
    _,xg_LF,_,_,_,_,_,_ = run_load_flow(p0=p0,q1=q1_sol,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)
    print('xg opt = \n{}\nxg LF - \n{}\ndifference = \n{}'.format(xg_opt,xg_LF,xg_opt-xg_LF))
    assert np.allclose(xg_opt,xg_LF)

def run_optimal_load_flow_separate_LF_explicit(p0=50,q1=-100,q2=250,link_type='pipe_linear',link_params={'alpha':10},p1=40,p2=30,q01=40,q02=150,q12=120,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=None,scale_var_params=None,formulation='full',tol=1e-6,max_iter=150,p_BC=False):
    """Run optimal power flow, but the load flow equations are taken out of the optimal flow problem. That is, they are solved seperately, and then substituted in the objective function. The gradients and Hessians are determined by numerical methods

    Parameters
    ----------------
    p_BC : bool, optional
        When True, the pressure of the slack node is taken as boundary condition for the OPF. If False, it is taken is control variables. Default is False.
    """
    if formulation != 'full':
        raise ValueError('OPF, with LF separate, not implemented for other formulation than full')
    if scale_var:
        raise ValueError('OPF, with LF separate, not implemented when using scaling')

    print('\nRunning OPF, with LF separate, for gas network (p0 as BC: {})'.format(p_BC))
    # create network
    gas_net = create_network(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params)

    # run load flow once, to make sure that the initial guess of opf is at least a solution of LF
    x0 = initialize_network(gas_net,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)
    x_LF,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)


    # initial gues for opf
    q0 = q_inj[0]
    if p_BC:
        u_init = np.array([q1])
        p0_BC = p0
    else:
        u_init = np.array([q1, p0])
    slack_init = np.array([q0])
    x_opf0 = np.concatenate((u_init,slack_init))

    # define cost function / objective function
    def cost_function(x_opf):
        """Define the cost function for OPF.

        Parameters
        ----------------
        x_opf : np array
            Variable vector used in OPF. Is assumed to be [q1 p0 q0 q01 q02 q12 p1 p2]

        Returns
        -----------
        f : float
            The value of the cost function
        """
        q1 = x_opf[0] #<0
        q0 = x_opf[len(u_init)] #<0
        return a0 + b0*-q0 + c0*q0**2  + a1 + b1*-q1 + c1*q1**2

    # define (nonlinear??) equality constriants
    def solve_load_flow_in_opf(x_opf,network=gas_net):
        """Determines the slack flow of the gas network, given control variables, by solving the steady-state loadflow problem"""
        # update bc of network
        q1 = x_opf[0]
        if p_BC:
            p0 = p0_BC
        else:
            p0 = x_opf[1]
        network = update_bc(network,p0,q1)
        # solve load flow equations
        xg_init = network.set_x_init(formulation=formulation)
        network.reset_network(xg_init,formulation=formulation)
        x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        # determine value of slack flow
        q0_LF = q_inj[0] #<0
        # current slack flow in opf state variables
        q0 = x_opf[len(u_init)] #<0
        return q0 - q0_LF
    lb_eq = np.zeros(len(slack_init))
    ub_eq = np.zeros(len(slack_init))
    nonlinear_constraint = spo.NonlinearConstraint(solve_load_flow_in_opf,lb_eq,ub_eq)

    # define linear inequality constraints
    q1_lb = -200
    q1_ub = -110
    p0_lb = 48
    p0_ub = 52
    lb_ineq = -np.inf*np.ones(len(x_opf0)) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
    ub_ineq = np.inf*np.ones(len(x_opf0))
    if p_BC:
        lb_ineq[:len(u_init)] = np.array([q1_lb])
        ub_ineq[:len(u_init)] = np.array([q1_ub])
    else:
        lb_ineq[:len(u_init)] = np.array([q1_lb,p0_lb])
        ub_ineq[:len(u_init)] = np.array([q1_ub,p0_ub])
    bounds = spo.Bounds(lb_ineq,ub_ineq)

    # solve OPF
    res = spo.minimize(cost_function, x_opf0, method='trust-constr', constraints=[nonlinear_constraint], options={'verbose': 1}, bounds=bounds)
    x_opf = res.x
    # print solution
    xg_opt = gas_net.set_x_init(formulation=formulation)
    p_sol,q_sol,q_inj = gas_net.update_full(xg_opt,formulation=formulation)
    print('Optimal solution:')
    print('p = {} Pa'.format(p_sol))
    print('q = {} kg/s'.format(q_sol))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))

    return x_opf, res.fun,res.nfev,res.nit,res.execution_time

def test_opf_pbc_separate_LF_explicit():
    """Test OPF againts the solution of LF (the constraints and cost function parameteres are chosen such that this should be the case), using the pressure of the slack node as boundary condition, and imposing inequality constraints only on the control variables."""
    # Given
    p_BC = True
    p0=50 #BC
    q1=-180 # Inital guess
    q1_sol = -110 #BC
    q2=250 #BC
    link_type='pipe_linear'
    link_params={'alpha':10}
    p1=40 # Initial guess
    p2=30 # Initial guess
    q01=40 # Initial guess
    q02=150 # Initial guess
    q12=120 # Initial guess
    formulation='full'
    tol=1e-6
    max_iter=150
    scale_var = None

    # When
    x_opf, _, _, _, _  = run_optimal_load_flow_separate_LF_explicit(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=p_BC)
    q1_opt, q0_opt = x_opf
    _,xg_opt,_,_,_,_,_,_ = run_load_flow(p0=p0,q1=q1_opt,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)

    # Then
    _,xg_LF,_,_,_,_,_,_ = run_load_flow(p0=p0,q1=q1_sol,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)
    print('xg opt = \n{}\nxg LF - \n{}\ndifference = \n{}'.format(xg_opt,xg_LF,xg_opt-xg_LF))
    assert np.allclose(xg_opt,xg_LF)

def run_optimal_load_flow_separate_LF(p0=50,q1=-100,q2=250,link_type='pipe_linear',link_params={'alpha':10},p1=40,p2=30,q01=40,q02=150,q12=120,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=None,scale_var_params=None,formulation='full',tol=1e-6,max_iter=150,lb_ineq_state = np.array([-150,5,10,5,45,33]),ub_ineq_state = np.array([-50,100,200,150,50,38]),p_BC=False,approach='direct',ineq_constr='all',optimization_method='trust-constr',stay_within_bounds=False):
    """Optimal flow where the LF is included implicitely. The gradient and Hessian are determined analytically, either using a direct or adjoint approach. """
    if formulation != 'full':
        raise ValueError('OPF not implemented for other formulation than full')
    if scale_var:
        raise ValueError('OPF not implemented when using scaling')
    if not p_BC:
        raise ValueError('OPF with separate LF only implemented if p0 is taken as BC')
    print('\nRunning OPF with separate LF for gas network (p0 as BC: {}, inequality constraints: {}, method: {}, bounds: {}, approach:{})'.format(p_BC,ineq_constr,optimization_method,stay_within_bounds,approach))

    # create network
    gas_net = create_network(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params)

    # update the boundary conditions of the gas network to match the initial guess of opf. Initialize network
    gas_net = update_bc(gas_net,p0,q1)
    x0_LF = initialize_network(gas_net,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)
    # initial guess for opf
    x_opf0 = np.array([q1])

    # keep track of variables in cost function
    global f_vec
    global x_f_vec
    f_vec = list()
    x_f_vec = list()

    # define cost function / objective function
    def cost_function(x_opf,network=gas_net):
        """Define the cost function for OPF.

        Parameters
        ----------------
        x_opf : np array
            Variable vector used in OPF. Is assumed to be [q1 p0 q0 q01 q02 q12 p1 p2]

        Returns
        -----------
        f : float
            The value of the cost function
        """
        # solve LF
        q1 = x_opf[0]
        network = update_bc(network,p0,q1)
        xg_init = network.set_x_init(formulation=formulation)
        network.reset_network(xg_init,formulation=formulation)
        x_sol,iters,err_vec,p_sol,q_sol,q_inj = network.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        # determine value of slack flow
        q0 = q_inj[0] #<0
        f = a0 + b0*-q0 + c0*q0**2  + a1 + b1*-q1 + c1*q1**2
        global f_vec
        global x_f_vec
        f_vec.append(x_opf)
        x_f_vec = x_opf.copy()
        return f
    # gradient of objective function
    def deltaf_deltau(x_opf):
        """Partial derivative of objective function to control variable u"""
        q1 = x_opf[0]
        return np.array([-b1 + c1*q1])
    nlsys = NonLinearSystemGas(gas_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    def jac_cost(x_opf,method=approach,network=gas_net):
        """Gradient vector / Jacobian of cost function"""
        df_du = deltaf_deltau(x_opf) # first part of gradient
        # Jacobian of nonlinear equality constraints wrt state variables x
        q1 = x_opf[0]
        network = update_bc(network,p0,q1)
        xg_init = network.set_x_init(formulation=formulation)
        network.reset_network(xg_init,formulation=formulation)
        xg,iters,err_vec,p_sol,q_sol,q_inj = network.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params) #NB: for effiency, this solve should not be includes. That is, a solve is already done in the objective function, so that information should be reusable.
        J_lf = nlsys.J_dense(xg)
        q0 = q_inj[0] #<0
        dnleq_dx = np.zeros((len(xg)+1,len(xg)+1))
        dnleq_dq0 = np.zeros(len(xg)+1)
        dnleq_dx[0,:] = np.array([-1,-1,-1,0,0,0])
        dnleq_dx[1:,1:] = J_lf
        deltaf_deltax = np.array([-b0+c0*q0, 0, 0, 0, 0, 0])
        dnleq_du = np.array([0,-1,0,0,0,0])
        if method == 'direct':
            w = np.linalg.solve(dnleq_dx,-dnleq_du)
            df_du += np.dot(deltaf_deltax,w)
        elif method == 'adjoint':
            v = np.linalg.solve(np.transpose(dnleq_dx),deltaf_deltax)
            df_du += np.dot(v,-dnleq_du)
        return df_du
    def hess_cost(x_opf,method=approach,network=gas_net):
        """Gradient vector / Jacobian of cost function"""
        d2f_du2 = 2*c1 # first part of gradient
        # Jacobian of nonlinear equality constraints wrt state variables x
        q1 = x_opf[0]
        network = update_bc(network,p0,q1)
        xg_init = network.set_x_init(formulation=formulation)
        network.reset_network(xg_init,formulation=formulation)
        xg,iters,err_vec,p_sol,q_sol,q_inj = network.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params) #NB: for effiency, this solve should not be includes. That is, a solve is already done in the objective function, so that information should be reusable.
        J_lf = nlsys.J_dense(xg)
        q0 = q_inj[0] #<0
        dnleq_dx = np.zeros((len(xg)+1,len(xg)+1))
        dnleq_dq0 = np.zeros(len(xg)+1)
        dnleq_dx[0,:] = np.array([-1,-1,-1,0,0,0])
        dnleq_dx[1:,1:] = J_lf
        deltaf_deltax = np.array([-b0+c0*q0, 0, 0, 0, 0, 0])
        dnleq_du = np.array([0,-1,0,0,0,0])
        delta2f_deltax2 = np.zeros((len(xg)+1,len(xg)+1))
        delta2f_deltax2[0,0] = 2*c0
        if method == 'direct':
            w = np.linalg.solve(dnleq_dx,-dnleq_du)
            deltaw_deltau = np.linalg.solve(dnleq_dx,np.zeros(len(xg)+1))
            d2f_du2 += np.dot(w,np.dot(dnleq_dx,w)) + np.dot(deltaf_deltax,deltaw_deltau)
        elif method == 'adjoint':
            deltavJ_deltax = np.linalg.solve(dnleq_dx,delta2f_deltax2)
            deltavJ_deltau = np.linalg.solve(dnleq_dx,np.zeros(len(xg)+1))
            vJ = np.linalg.solve(np.transpose(dnleq_dx),np.dot(deltavJ_deltax,-dnleq_du))
            d2f_du2 += np.dot(vJ,-dnleq_du) + np.dot(deltavJ_deltau,-dnleq_du)
        return d2f_du2

    if ineq_constr == 'all':
        # define inequality constraints
        def g(x_opf,network=gas_net):
            # solve LF
            q1 = x_opf[0]
            network = update_bc(network,p0,q1)
            xg_init = network.set_x_init(formulation=formulation)
            network.reset_network(xg_init,formulation=formulation)
            x_sol,iters,err_vec,p_sol,q_sol,q_inj = network.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            x = np.concatenate((np.array([q_inj[0]]),x_sol))
            g = np.concatenate((x-lb_ineq_state,ub_ineq_state-x))
            return g
        def g_jac(x_opf,method=approach,network=gas_net):
            """Jacobian of inequality constraints"""
            # Jacobian of inequality constraints wrt state variables x
            deltag_deltax = np.vstack((np.eye(6),-np.eye(6)))
            deltag_deltau = np.zeros(12)
            # Jacobian of nonlinear equality constraints wrt state variables x
            q1 = x_opf[0]
            network = update_bc(network,p0,q1)
            xg_init = network.set_x_init(formulation=formulation)
            network.reset_network(xg_init,formulation=formulation)
            xg,iters,err_vec,p_sol,q_sol,q_inj = network.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params) #NB: for effiency, this solve should not be includes. That is, a solve is already done in the objective function, so that information should be reusable.
            J_lf = nlsys.J_dense(xg)
            q0 = q_inj[0] #<0
            dnleq_dx = np.zeros((len(xg)+1,len(xg)+1))
            dnleq_dq0 = np.zeros(len(xg)+1)
            dnleq_dx[0,:] = np.array([-1,-1,-1,0,0,0])
            dnleq_dx[1:,1:] = J_lf
            dnleq_du = np.array([0,-1,0,0,0,0])
            dg_du = np.zeros((12,1))
            dg_du[:,0] = deltag_deltau # first part of gradient
            if method == 'direct':
                w = np.linalg.solve(dnleq_dx,-dnleq_du)
                dg_du[:,0] += np.dot(deltag_deltax,w)
            elif method == 'adjoint':
                v = np.linalg.solve(np.transpose(dnleq_dx),np.transpose(deltag_deltax))
                dg_du[:,0] += np.dot(np.transpose(v),-dnleq_du)
            return dg_du
        if optimization_method == 'trust-constr':
            ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(12),np.inf*np.ones(12),jac=g_jac,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            ineq_constr_fun  = {'type':'ineq','fun':g,'jac':g_jac}
    else:
        ineq_constr_fun = None

    # define bounds
    q1_lb = -200
    q1_ub = -110
    lb_bounds = np.array([q1_lb])
    ub_bounds = np.array([q1_ub])
    if optimization_method == 'ipopt':
        bounds = [(lb,ub) for lb, ub in zip(lb_bounds,ub_bounds)]
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
    else:
        bounds = spo.Bounds(lb_bounds,ub_bounds,keep_feasible=stay_within_bounds)

    # make sure initial guess satisfies bounds (NB. If adjustments are made, LF is not necessarily satisfied anymore)
    if ineq_constr != None and (optimization_method == 'SLSQP' or stay_within_bounds):
        for ind, x0 in enumerate(x_opf0):
            if lb_bounds[ind] > x0:
                x_opf0[ind] = lb_bounds[ind]
            elif ub_bounds[ind] < x0:
                x_opf0[ind] = ub_bounds[ind]

    # solve OPF
    try:
        if optimization_method == 'trust-constr':
            res = spo.minimize(cost_function, x_opf0, method=optimization_method, jac=jac_cost, hess=hess_cost, constraints=[ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds,tol=tol)
        elif optimization_method == 'SLSQP':
            res = spo.minimize(cost_function, x_opf0, method=optimization_method, jac=jac_cost, constraints=[ineq_constr_fun], options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol)
        elif optimization_method == 'ipopt':
            res = ipopt.minimize_ipopt(cost_function, x_opf0, jac=jac_cost, constraints=[ineq_constr_fun], options={'maxiter':max_iter,'disp': 1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
    except:
        print('Exception made for {}, hard bounds: {}, approach: {}'.format(optimization_method,stay_within_bounds,approach))
        if len(f_vec) == 0:
            cost_function(x_opf0)
            nit = 0
            nfev = 0
            njev = 0
            nhev = 0
        else:
            nit = 0
            nfev = len(f_vec)
            njev = 0
            nhev = 0
        res = spo.OptimizeResult({'success':False,'x':np.array(x_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

    # print solution
    gas_net = update_bc(gas_net,p0,res.x[0])
    xg_opt = gas_net.set_x_init(formulation=formulation)
    p_sol,q_sol,q_inj = gas_net.update_full(xg_opt,formulation=formulation)
    print('Optimal solution:')
    print('p = {} Pa'.format(p_sol))
    print('q = {} kg/s'.format(q_sol))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    return xg_opt, res

def compare_opf(dir_path,number_runs=10):
    """Compare the different formulation of the optimal flow problem. Print results to a table that can be read by Tex."""
    # given (BCs, initial guesses, parameters, solver info, etc.)
    p0_BC=50 #BC
    p0_init=55 # Initial guess
    q1=-180 # Inital guess
    q1_BC= -110 #BC
    q2=250 #BC
    link_type='pipe_linear'
    link_params={'alpha':10}
    p1=40 # Initial guess
    p2=30 # Initial guess
    q01=40 # Initial guess
    q02=150 # Initial guess
    q12=120 # Initial guess
    formulation='full'
    tol=1e-6
    max_iter=150
    scale_var = None

    # run the various optimizations. Run several times, take average of run time. For the other data (which seemed to be the same every time), the last run is used.
    exec_times = list()
    exec_times_pBC = list()
    exec_times_control = list()
    exec_times_sepLF = list()
    exec_times_pBC_sepLF = list()
    for run in range(number_runs):
        # p0 control variable, inequality contraints on all variables, load flow as equality constraints
        xg_opt, res = run_optimal_load_flow(p0=p0_init,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=False,ineq_constr='all')
        x_opf, obj_fun, nfev, nit, exec_time = res.x, res.fun, res.nfev, res.nit, res.execution_time
        exec_times.append(exec_time)
        # p0 as BC, inequality contraints only on control variables, load flow as equality constraints
        xg_opt_pBC, res_pBC = run_optimal_load_flow(p0=p0_BC,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=True,ineq_constr='control')
        x_opf_pBC, obj_fun_pBC, nfev_pBC, nit_pBC, exec_time_pBC = res_pBC.x, res_pBC.fun, res_pBC.nfev, res_pBC.nit, res_pBC.execution_time
        exec_times_pBC.append(exec_time_pBC)
        # p0 control variable, inequality contraints only on control variables, load flow as equality constraints
        xg_opt_control, res_control = run_optimal_load_flow(p0=p0_init,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=False,ineq_constr='control')
        x_opf_control, obj_fun_control, nfev_control, nit_control, exec_time_control = res_control.x, res_control.fun, res_control.nfev, res_control.nit, res_control.execution_time
        exec_times_control.append(exec_time_control)
        # p0 control variable, inequality contraints only on control variables, load flow as a separate solver
        x_opf_sepLF, obj_fun_sepLF, nfev_sepLF, nit_sepLF, exec_time_sepLF = run_optimal_load_flow_separate_LF_explicit(p0=p0_init,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=False)
        exec_times_sepLF .append(exec_time_sepLF)
        # p0 as BC, inequality contraints only on control variables, load flow as a separate solver
        x_opf_pBC_sepLF, obj_fun_pBC_sepLF, nfev_pBC_sepLF, nit_pBC_sepLF, exec_time_pBC_sepLF = run_optimal_load_flow_separate_LF_explicit(p0=p0_BC,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=True)
        exec_times_pBC_sepLF .append(exec_time_pBC_sepLF)

    exec_time = np.mean(exec_times)
    exec_time_pBC = np.mean(exec_times_pBC)
    exec_time_control = np.mean(exec_times_control)
    exec_time_sepLF = np.mean(exec_times_sepLF)
    exec_time_pBC_sepLF = np.mean(exec_times_pBC_sepLF)
    print('exec time = {}'.format(exec_times))
    print('exec time pBC = {}'.format(exec_times_pBC))
    print('exec time ineq. control= {}'.format(exec_times_control))
    print('exec time sep. LF= {}'.format(exec_times_sepLF))
    print('exec time pBC sep. LF= {}'.format(exec_times_pBC_sepLF))

    path_to_tables = os.path.join(dir_path,'network_data','G3N_OPF')
    # create (and save) table with optimal solution in network
    p0_opt = x_opf[1]
    q0_opt = x_opf[2]
    xg_opt = xg_from_xopf(x_opf,p_BC=False)
    p0_opt_control = x_opf_control[1]
    q0_opt_control = x_opf_control[2]
    xg_opt_control = xg_from_xopf(x_opf_control,p_BC=False)
    q0_opt_pBC = x_opf_pBC[1]
    xg_opt_pBC = xg_from_xopf(x_opf_pBC,p_BC=True)
    q1_opt_sepLF, p0_opt_sepLF, q0_opt_sepLF = x_opf_sepLF
    with HiddenPrints():
        _,xg_opt_sepLF,_,_,_,_,_,_ = run_load_flow(p0=p0_opt_sepLF,q1=q1_opt_sepLF,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)
    q1_opt_pBC_sepLF, q0_opt_pBC_sepLF = x_opf_pBC_sepLF
    with HiddenPrints():
        _,xg_opt_pBC_sepLF,_,_,_,_,_,_ = run_load_flow(p0=p0_BC,q1=q1_opt_pBC_sepLF,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)
        _,xg_LF,_,_,_,_,q_inj_LF,_ = run_load_flow(p0=p0_BC,q1=q1_BC,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)
    q0_LF = q_inj_LF[0]
    with open(os.path.join(path_to_tables,'network_solution.txt'), "w") as table:
        table.write(r'$q_0$ & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f} \\ '.format(q0_LF,q0_opt,q0_opt_pBC,q0_opt_control,q0_opt_sepLF,q0_opt_pBC_sepLF))
        table.write(r'$p_0$ & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f} \\ '.format(p0_BC,p0_opt,p0_BC,p0_opt_control,p0_opt_sepLF,p0_BC))
        variable_names = [r'$q_{01}$',r'$q_{02}$',r'$q_{12}$',r'$p_1$',r'$p_2$']
        for ind_var, var in enumerate(variable_names):
            table.write(r'{} & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f} \\ '.format(var,xg_LF[ind_var],xg_opt[ind_var],xg_opt_pBC[ind_var],xg_opt_control[ind_var],xg_opt_sepLF[ind_var],xg_opt_pBC_sepLF[ind_var]))

    # print results of optimizer, and create (and save) table
    print('\nopf     opf pBC  opf all ineq    opf sep. LF     opf sep. LF pBC')
    print('obj. func:  {:.5f}  , {:.5f} , {:.5f}  , {:.5f}  , {:.5f}'.format(obj_fun,obj_fun_pBC,obj_fun_control,obj_fun_sepLF,obj_fun_pBC_sepLF))
    print('numb. fev.:  {:d}  , {:d} , {:d}  , {:d}  , {:d}'.format(nfev,nfev_pBC,nfev_control,nfev_sepLF,nfev_pBC_sepLF))
    print('iters:  {:d}  , {:d}  , {:d} , {:d}  , {:d}'.format(nit,nit_pBC,nit_control,nit_sepLF,nit_pBC_sepLF))
    print('time:  {:.5f}  , {:.5f} , {:5f}  , {:.5f}  , {:.5f}'.format(exec_time,exec_time_pBC,exec_time_control,exec_time_sepLF,exec_time_pBC_sepLF))
    with open(os.path.join(path_to_tables,'optimizer_info.txt'), "w") as table:
        table.write(r'$f$ &  {:.5f}  & {:.5f} & {:.5f}  & {:.5f}  & {:.5f} \\ '.format(obj_fun,obj_fun_pBC,obj_fun_control,obj_fun_sepLF,obj_fun_pBC_sepLF))
        table.write(r'func. eval. &  {:d}  & {:d}  & {:d}  & {:d}  & {:d} \\ '.format(nfev,nfev_pBC,nfev_control,nfev_sepLF,nfev_pBC_sepLF))
        table.write(r'iterations &  {:d}  & {:d}  & {:d}  & {:d}  & {:d} \\ '.format(nit,nit_pBC,nit_control,nit_sepLF,nit_pBC_sepLF))
        table.write(r'time [s] &  {:.5f}  & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f} \\ '.format(exec_time,exec_time_pBC,exec_time_control,exec_time_sepLF,exec_time_pBC_sepLF))

def compare_opf_derivatives(dir_path,number_runs=10):
    """Compare the the optimal flow problem, using different ways to determine the gradients an Hessians. p0 is taken as BC, and the inequality contraints are imposed on the control variables only. Print results to a table that can be read by Tex."""
    # given (BCs, initial guesses, parameters, solver info, etc.)
    p0 =50 #BC
    q1=-180 # Inital guess
    q1_BC= -110 #BC
    q2=250 #BC
    link_type='pipe_linear'
    link_params={'alpha':10}
    p1=40 # Initial guess
    p2=30 # Initial guess
    q01=40 # Initial guess
    q02=150 # Initial guess
    q12=120 # Initial guess
    formulation='full'
    tol=1e-6
    max_iter=150
    scale_var = None
    p_BC = True
    ineq_constr = 'control'

    # run the various optimizations. Run several times, take average of run time. For the other data (which seemed to be the same every time), the last run is used.
    exec_times = list()
    exec_times_sepLF_direct = list()
    exec_times_sepLF_adjoint = list()
    for run in range(number_runs):
        # LF is included as (nonlinear) equality constriant. Analytical expressions for gradients and Hessians of objective function and equality constraints are used
        xg_opt, res = run_optimal_load_flow(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=p_BC,ineq_constr=ineq_constr,derivatives=True)
        x_opf, obj_fun, nfev, nit, exec_time = res.x, res.fun, res.nfev, res.nit, res.execution_time
        exec_times.append(exec_time)
        # LF is not included as (nonlinear) equality constriant. Analytical expressions for gradients and Hessians of objective function are determined using the direct approach
        x_opf_sepLF_direct, obj_fun_sepLF_direct, nfev_sepLF_direct, nit_sepLF_direct, exec_time_sepLF_direct  = run_optimal_load_flow_separate_LF(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=p_BC,approach='direct')
        xg_opt_sepLF_direct, res_sepLF_direct = res_sepLF_direct.x, res_sepLF_direct.fun, res_sepLF_direct.nfev, res_sepLF_direct.nit, res_sepLF_direct.execution_time
        exec_times_sepLF_direct.append(exec_time_sepLF_direct)
        # LF is not included as (nonlinear) equality constriant. Analytical expressions for gradients and Hessians of objective function are determined using the adjoint approach
        xg_opt_sepLF_adjoint, res_sepLF_adjoint =  run_optimal_load_flow_separate_LF(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=p_BC,approach='adjoint')
        x_opf_sepLF_adjoint, obj_fun_sepLF_adjoint, nfev_sepLF_adjoint, nit_sepLF_adjoint, exec_time_sepLF_adjoint = res_sepLF_adjoint.x, res_sepLF_adjoint.fun, res_sepLF_adjoint.nfev, res_sepLF_adjoint.nit, res_sepLF_adjoint.execution_time
        exec_times_sepLF_adjoint.append(exec_time_sepLF_adjoint)

    exec_time = np.mean(exec_times)
    exec_time_sepLF_direct = np.mean(exec_times_sepLF_direct)
    exec_time_sepLF_adjoint = np.mean(exec_times_sepLF_adjoint)
    print('exec time = {}'.format(exec_times))
    print('exec time sep. LF direct = {}'.format(exec_times_sepLF_direct))
    print('exec time sep. LF adjoint = {}'.format(exec_times_sepLF_adjoint))

    path_to_tables = os.path.join(dir_path,'network_data','G3N_OPF')
    # create (and save) table with optimal solution in network
    q1_opt = x_opf[0]
    q0_opt = x_opf[1]
    xg_opt = xg_from_xopf(x_opf,p_BC=p_BC)
    q1_opt_sepLF_direct = x_opf_sepLF_direct[0]
    with HiddenPrints():
        _,xg_opt_sepLF_direct,_,_,_,_,q_inj_sepLF_direct,_ = run_load_flow(p0=p0,q1=q1_opt_sepLF_direct,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)
    q0_opt_sepLF_direct = q_inj_sepLF_direct[0]
    q1_opt_sepLF_adjoint = x_opf_sepLF_adjoint[0]
    with HiddenPrints():
        _,xg_opt_sepLF_adjoint,_,_,_,_,q_inj_sepLF_adjoint,_ = run_load_flow(p0=p0,q1=q1_opt_sepLF_adjoint,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)
        _,xg_LF,_,_,_,_,q_inj_LF,_ = run_load_flow(p0=p0,q1=q1_BC,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)
    q0_opt_sepLF_adjoint = q_inj_sepLF_adjoint[0]
    q0_LF = q_inj_LF[0]
    with open(os.path.join(path_to_tables,'network_solution_derivatives.txt'), "w") as table:
        table.write(r'$q_0$ & {:.5f}  & {:.5f}  & {:.5f} & {:.5f}  \\ '.format(q0_LF,q0_opt,q0_opt_sepLF_direct,q0_opt_sepLF_adjoint))
        table.write(r'$q_1$ & {:.5f}  & {:.5f}  & {:.5f} & {:.5f}  \\ '.format(q1_BC,q1_opt,q1_opt_sepLF_direct,q1_opt_sepLF_adjoint))
        variable_names = [r'$q_{01}$',r'$q_{02}$',r'$q_{12}$',r'$p_1$',r'$p_2$']
        for ind_var, var in enumerate(variable_names):
            table.write(r'{} & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f}  \\ '.format(var,xg_LF[ind_var],xg_opt[ind_var],xg_opt_sepLF_direct[ind_var],xg_opt_sepLF_adjoint[ind_var]))

    # print results of optimizer, and create (and save) table
    print('\nopf     opf sep. LF direct    opf sep. LF adjoint')
    print('obj. func:  {:.5f}  , {:.5f} , {:.5f}'.format(obj_fun,obj_fun_sepLF_direct,obj_fun_sepLF_adjoint))
    print('numb. fev.:  {:d}  , {:d} , {:d}'.format(nfev,nfev_sepLF_direct,nfev_sepLF_adjoint))
    print('iters:  {:d}  , {:d}  , {:d}'.format(nit,nit_sepLF_direct,nit_sepLF_adjoint))
    print('time:  {:.5f}  , {:.5f} , {:5f}'.format(exec_time,exec_time_sepLF_direct,exec_time_sepLF_adjoint))
    with open(os.path.join(path_to_tables,'optimizer_info_derivatives.txt'), "w") as table:
        table.write(r'$f$ &  {:.5f}  & {:.5f} & {:.5f}  \\ '.format(obj_fun,obj_fun_sepLF_direct,obj_fun_sepLF_adjoint))
        table.write(r'func. eval. &  {:d}  & {:d}  & {:d}  \\ '.format(nfev,nfev_sepLF_direct,nfev_sepLF_adjoint))
        table.write(r'iterations &  {:d}  & {:d}  & {:d}  \\ '.format(nit,nit_sepLF_direct,nit_sepLF_adjoint))
        table.write(r'time [s] &  {:.5f}  & {:.5f}  & {:.5f}  \\ '.format(exec_time,exec_time_sepLF_direct,exec_time_sepLF_adjoint))

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

def compare_opf_methods(dir_path=None,save_tables=False):
    """Compare OF for different optimization methods."""
    if save_tables and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # given (BCs, initial guesses, parameters, solver info, etc.)
    p0 =50 #BC
    q1=-180 # Inital guess
    q1_BC= -110 #BC
    q2=250 #BC
    link_type='pipe_linear'
    link_params={'alpha':10}
    p1=40 # Initial guess
    p2=30 # Initial guess
    q01=40 # Initial guess
    q02=150 # Initial guess
    q12=120 # Initial guess
    lb_ineq_state = np.array([-150,5,10,5,45,33])
    ub_ineq_state = np.array([-50,100,200,150,50,38])
    formulation='full'
    tol=1e-6
    max_iter=150
    scale_var = None
    p_BC = True
    ineq_constr = 'all'

    result = dict()
    xg_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']

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
                xg_opt, res = run_optimal_load_flow(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,p_BC=p_BC,lb_ineq_state=lb_ineq_state,ub_ineq_state=ub_ineq_state,ineq_constr=ineq_constr,derivatives=derivatives,optimization_method=method,stay_within_bounds=stay_within_bounds)
                result[method+'_'+bound+'_'+der] = res
                xg_res[method+'_'+bound+'_'+der] = xg_opt

    # LF solution
    with HiddenPrints():
        _,xg_LF,_,_,_,_,q_inj_LF,_ = run_load_flow(p0=p0,q1=q1_BC,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)
    q0_LF = q_inj_LF[0]
    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','G3N_OPF')
        for bound in bounds:
            for der in ders:
                xg_opt_trust = xg_res.get('trust-constr_'+bound+'_'+der)
                res_trust = result.get('trust-constr_'+bound+'_'+der)
                q1_opt_trust = res_trust.x[0]
                q0_opt_trust = res_trust.x[1]
                xg_opt_slsqp = xg_res.get('SLSQP_'+bound+'_'+der)
                res_slsqp = result.get('SLSQP_'+bound+'_'+der)
                q1_opt_slsqp = res_slsqp.x[0]
                q0_opt_slsqp = res_slsqp.x[1]
                xg_opt_ipopt = xg_res.get('ipopt_'+bound+'_'+der)
                res_ipopt = result.get('ipopt_'+bound+'_'+der)
                q1_opt_ipopt = res_ipopt.x[0]
                q0_opt_ipopt = res_ipopt.x[1]
                with open(os.path.join(path_to_tables,'network_solution_errors_methods'+'_'+bound+'_'+der+'.txt'), "w") as table:
                    table.write(r'$q_0$ & {:.2f}  & {:.3e}  & {:.3e} & {:.3e}  \\ '.format(q0_LF,error(q0_opt_trust,q0_LF),error(q0_opt_slsqp,q0_LF),error(q0_opt_ipopt,q0_LF)))
                    table.write(r'$q_1$ & {:.2f}  & {:.3e}  & {:.3e} & {:.3e}  \\ '.format(q1_BC,error(q1_opt_trust,q1_BC),error(q1_opt_slsqp,q1_BC),error(q1_opt_ipopt,q1_BC)))
                    variable_names = [r'$q_{01}$',r'$q_{02}$',r'$q_{12}$',r'$p_1$',r'$p_2$']
                    for ind_var, var in enumerate(variable_names):
                        table.write(r'{} & {:.2f}  & {:.3e}  & {:.3e} & {:.3e}  \\ '.format(var,xg_LF[ind_var],error(xg_opt_trust[ind_var],xg_LF[ind_var]),error(xg_opt_slsqp[ind_var],xg_LF[ind_var]),error(xg_opt_ipopt[ind_var],xg_LF[ind_var])))
        with open(os.path.join(path_to_tables,'optimizer_info_methods.txt'), "w") as table:
            for bound in bounds:
                for der in ders:
                    xg_opt_trust = xg_res.get('trust-constr_'+bound+'_'+der)
                    xg_opt_slsqp = xg_res.get('SLSQP_'+bound+'_'+der)
                    xg_opt_ipopt = xg_res.get('ipopt_'+bound+'_'+der)
                    res_trust = result.get('trust-constr_'+bound+'_'+der)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+der)
                    res_ipopt = result.get('ipopt_'+bound+'_'+der)
                    table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(xg_opt_trust,xg_LF),error(xg_opt_slsqp,xg_LF),error(xg_opt_ipopt,xg_LF)))

def compare_opf_methods_sep_LF(dir_path=None,save_tables=False):
    """Compare OF for different optimization methods, when LF is eliminated."""
    if save_tables and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # given (BCs, initial guesses, parameters, solver info, etc.)
    p0 =50 #BC
    q1=-180 # Inital guess
    q1_BC= -110 #BC
    q2=250 #BC
    link_type='pipe_linear'
    link_params={'alpha':10}
    p1=47 # Initial guess
    p2=35 # Initial guess
    q01=40 # Initial guess
    q02=150 # Initial guess
    q12=120 # Initial guess
    lb_ineq_state = np.array([-2*q2,-2*q2,-2*q2,-2*q2,1,1])
    ub_ineq_state = np.array([0,2*q2,2*q2,2*q2,1.5*p0,1.5*p0])
    formulation='full'
    tol=1e-6
    max_iter=250
    scale_var = None
    p_BC = True
    ineq_constr = 'all'

    result = dict()
    xg_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']

    for method in methods:
        for bound in bounds:
            if bound == 'soft':
                stay_within_bounds = False
            else:
                stay_within_bounds = True
            for approach in approaches:
                if approach == 'direct' or approach == 'adjoint':
                    xg_opt, res = run_optimal_load_flow_separate_LF(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,lb_ineq_state=lb_ineq_state,ub_ineq_state=ub_ineq_state,p_BC=p_BC,approach=approach,ineq_constr=ineq_constr,optimization_method=method,stay_within_bounds=stay_within_bounds)
                else:
                    xg_opt, res = run_optimal_load_flow(p0=p0,q1=q1,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter,lb_ineq_state=lb_ineq_state,ub_ineq_state=ub_ineq_state,p_BC=p_BC,ineq_constr=ineq_constr,derivatives=True,optimization_method=method,stay_within_bounds=stay_within_bounds)
                result[method+'_'+bound+'_'+approach] = res
                xg_res[method+'_'+bound+'_'+approach] = xg_opt

    # LF solution
    with HiddenPrints():
        _,xg_LF,_,_,_,_,q_inj_LF,_ = run_load_flow(p0=p0,q1=q1_BC,q2=q2,link_type=link_type,link_params=link_params,p1=p1,p2=p2,q01=q01,q02=q02,q12=q12,scale_var=scale_var,formulation=formulation,tol=tol,max_iter=max_iter)
    q0_LF = q_inj_LF[0]
    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','G3N_OPF')
        for bound in bounds:
            for approach in approaches:
                xg_opt_trust = xg_res.get('trust-constr_'+bound+'_'+approach)
                res_trust = result.get('trust-constr_'+bound+'_'+approach)
                q1_opt_trust = res_trust.x[0]
                xg_opt_slsqp = xg_res.get('SLSQP_'+bound+'_'+approach)
                res_slsqp = result.get('SLSQP_'+bound+'_'+approach)
                q1_opt_slsqp = res_slsqp.x[0]
                xg_opt_ipopt = xg_res.get('ipopt_'+bound+'_'+approach)
                res_ipopt = result.get('ipopt_'+bound+'_'+approach)
                q1_opt_ipopt = res_ipopt.x[0]
                with open(os.path.join(path_to_tables,'network_solution_errors_methods_sep_LF_'+bound+'_'+approach+'.txt'), "w") as table:
                    table.write(r'$q_1$ & {:.2f}  & {:.3e}  & {:.3e} & {:.3e}  \\ '.format(q1_BC,error(q1_opt_trust,q1_BC),error(q1_opt_slsqp,q1_BC),error(q1_opt_ipopt,q1_BC)))
                    variable_names = [r'$q_{01}$',r'$q_{02}$',r'$q_{12}$',r'$p_1$',r'$p_2$']
                    for ind_var, var in enumerate(variable_names):
                        table.write(r'{} & {:.2f}  & {:.3e}  & {:.3e} & {:.3e}  \\ '.format(var,xg_LF[ind_var],error(xg_opt_trust[ind_var],xg_LF[ind_var]),error(xg_opt_slsqp[ind_var],xg_LF[ind_var]),error(xg_opt_ipopt[ind_var],xg_LF[ind_var])))
        with open(os.path.join(path_to_tables,'optimizer_info_methods_sep_LF.txt'), "w") as table:
            for bound in bounds:
                for approach in approaches:
                    if approach == 'eq_constr':
                        approach_label = 'eq. constr.'
                    else:
                        approach_label = approach
                    xg_opt_trust = xg_res.get('trust-constr_'+bound+'_'+approach)
                    xg_opt_slsqp = xg_res.get('SLSQP_'+bound+'_'+approach)
                    xg_opt_ipopt = xg_res.get('ipopt_'+bound+'_'+approach)
                    res_trust = result.get('trust-constr_'+bound+'_'+approach)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+approach)
                    res_ipopt = result.get('ipopt_'+bound+'_'+approach)
                    table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(bound,approach_label,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(xg_opt_trust,xg_LF),error(xg_opt_slsqp,xg_LF),error(xg_opt_ipopt,xg_LF)))
                table.write('\hline ')

    for bound in bounds:
        for approach in approaches:
            xg_opt_trust = xg_res.get('trust-constr_'+bound+'_'+approach)
            xg_opt_slsqp = xg_res.get('SLSQP_'+bound+'_'+approach)
            xg_opt_ipopt = xg_res.get('ipopt_'+bound+'_'+approach)
            res_trust = result.get('trust-constr_'+bound+'_'+approach)
            res_slsqp = result.get('SLSQP_'+bound+'_'+approach)
            res_ipopt = result.get('ipopt_'+bound+'_'+approach)
            print('\nBounds: {}, approach: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\ntrust-constr:{}\nSLSQP: {}\nIPOPT: {}\nErrors for t-c: {}, SLSQP: {}, IPOPT: {}'.format(bound,approach,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(xg_opt_trust,xg_LF),error(xg_opt_slsqp,xg_LF),error(xg_opt_ipopt,xg_LF)))


if __name__ == '__main__':
    # load flow
    gas_net,x_sol,iters,err_vec,p_sol,q_sol,q_inj,tol = run_load_flow()

    # plot topology
    fig_top = plt.figure('Network topology')
    ax_top = fig_top.gca()
    gas_net.draw_network(ax_top,halflink_angle=2,halflink_length=.5)
    plt.axis('equal')
    plt.axis('off')

    #opf
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #compare_opf(dir_path,number_runs=100)
    # compare_opf_derivatives(dir_path,number_runs=100)
    # compare_opf_methods(dir_path=dir_path,save_tables=False)
    compare_opf_methods_sep_LF(dir_path=dir_path,save_tables=False)
    plt.show()
