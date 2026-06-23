"""Example of an electrical network with 3 nodes, connected with short lines. 
"""
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

MW = 1e6 #[W]
kV = 1e3 #[V]
def create_network(b,g):
    """Creates and returns an electrical test network with only short lines
    
    Parameters
    ----------
    b : float
        susceptance of line
    g : float
        reactance of line
        
    Returns
    -------
    elec_net : ElectricalNetwork
        The electrical network
    """
    elec_net = ElectricalNetwork('test electrical network')
    en0 = ElectricalNode('en0',node_type=0,V=1.,delta=0.) # reference node
    en1 = ElectricalNode('en1',node_type=2,P=0.9931,Q=0.0492) # load node
    en2 = ElectricalNode('en2',node_type=1,P=0.9931,V=0.98) # generator node

    el0 = ElectricalLink('el0',en0,en1,link_type = 'short_line',link_params = {'b':b,'g':g})
    el1 = ElectricalLink('el1',en0,en2,link_type = 'short_line',link_params = {'b':b,'g':g})
    el2 = ElectricalLink('el2',en1,en2,link_type = 'short_line',link_params = {'b':b,'g':g})

    elec_net.add_link(el0)
    elec_net.add_link(el1)
    elec_net.add_link(el2)
    return elec_net

def initialize_network(network,scale_var=None,scale_var_params=None):
    """Sets values of network variables to be used for initial guess. Default values are used.
    
    Parameters
    ----------
    network : ElectricalNetwork
        The network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    #delta_init = np.array([0.,0.])
    #V_init = np.array([1.])
    #x_init = np.concatenate((delta_init,V_init))
    network.initialize()
    #network.update(x_init) 
    x0 = network.set_x_init(scale_var=scale_var,scale_var_params=scale_var_params)
    return x0
    
def solve_system(network,tol,max_iter,h,x0,scale_var=None,scale_var_params=None,D_F=np.array([]),D_x=np.array([]),lin_solver='solve',FD=True):
    """Solve the network using analytical Jacobian and FD Jacobian, with basic NR
    
    Parameters
    ----------
    network : ElectricalNetwork
        The network to be solved
    tol : float
        tolerance of NR
    max_iter : int
        maximum number of iterations of NR
    h : float
        step size used for FD
    x0 : np array
        inital guess
        
    Returns
    -------
    x_sol_FD : np array
        solution vector, using FD Jacobian
    iters_FD : int
        total number of iterations, using FD Jacobian
    err_vec_FD : np array
        vector with the error of NR for every iteration, using FD Jacobian
    x_sol : np array
        solution vector, using analytical Jacobian
    iters : int
        total number of iterations, using analytical Jacobian 
    err_vec : np array
        vector with the error of NR for every iteration, using analytical Jacobian
    """
    if FD:
        network.update(x0,scale_var=scale_var,scale_var_params=scale_var_params)
        x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD = network.solve_network(tol,max_iter,h=h,solver='NR_FD',scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x)
    else:
        x_sol_FD = None
        iters_FD = None
        err_vec_FD = None
    
    network.update(x0,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = network.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x,lin_solver=lin_solver)
    
    return x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec

def print_results(x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec):
    """Print the results.
    
    Parameters
    ----------
    _sol_FD : np array
        solution vector, using FD Jacobian
    iters_FD : int
        total number of iterations, using FD Jacobian
    err_vec_FD : np array
        vector with the error of NR for every iteration, using FD Jacobian
    x_sol : np array
        solution vector, using analytical Jacobian
    iters : int
        total number of iterations, using analytical Jacobian 
    err_vec : np array
        vector with the error of NR for every iteration, using analytical Jacobian
    """
    print("\nFD solution:")
    print("Error vector = {}".format(err_vec_FD))
    print("Ater {} iterations the solution is {}".format(format(iters_FD),format(x_sol_FD)))

    
    print("\nAnalytical solution:")
    print("Error vector = {}".format(err_vec))
    print("Ater {} iterations the solution is {}".format(format(iters),format(x_sol)))
    
def example_e3n_pu():
    """Check solution of the example network, where everything is already in p.u.
    """
    # Given
    b = -10.
    g = 1.
    elec_net = create_network(b,g)
    x0 = initialize_network(elec_net)
    
    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(elec_net,tol,max_iter,h,x0)
    
    # Then
    delta_sol_expected = np.array([-0.10000294, -0.10000295])
    V_sol_expected = np.array([0.97999991])
    x_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    assert np.allclose(x_sol,x_sol_expected) 
    
def example_e3n():
    """Check solution of the example network, where everything is in S.I. units
    """
    # Given
    Sbase = 1*MW #[W] ?
    Vbase = 10/np.sqrt(3)*kV #[V] ?
    Ybase = Sbase/(Vbase**2)
    b_pu = -10.
    g_pu = 1.
    b = b_pu*Ybase
    g = g_pu*Ybase
    elec_net = create_network(b,g)
    # this network is created with the nodal voltages and injected powers in p.u., so change to S.I.
    for node in elec_net.get_nodes():
        node.V *= Vbase
        for hl in node.get_half_links():
            hl.P *= Sbase
            hl.Q *= Sbase
    x0 = initialize_network(elec_net)
    
    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(elec_net,tol,max_iter,h,x0)
    
    # Then
    delta_sol_expected = np.array([-0.10000294, -0.10000295])
    V_sol_expected = np.array([0.97999991])*Vbase
    x_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    assert np.allclose(x_sol,x_sol_expected) 
    
def example_e3n_SI_pu():
    """Check solution of the example network, where everything is in S.I. units, and then the per unit system is used
    """
    # Given
    Sbase = 1*MW #[W] ?
    Vbase = 10/np.sqrt(3)*kV #[V] ?
    Ybase = Sbase/(Vbase**2)
    scale_var = 'per_unit'
    scale_var_params = {'Sbase':Sbase,'Vbase':Vbase,'deltabase':1}
    b_pu = -10.
    g_pu = 1.
    b = b_pu*Ybase
    g = g_pu*Ybase
    elec_net = create_network(b,g)
    # this network is created with the nodal voltages and injected powers in p.u., so change to S.I.
    for node in elec_net.get_nodes():
        node.V *= Vbase
        for hl in node.get_half_links():
            hl.P *= Sbase
            hl.Q *= Sbase
    x0 = initialize_network(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    
    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(elec_net,tol,max_iter,h,x0,scale_var=scale_var,scale_var_params=scale_var_params)
    
    # Then
    delta_sol_expected = np.array([-0.10000294, -0.10000295])
    V_sol_expected = np.array([0.97999991])
    x_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    assert np.allclose(x_sol,x_sol_expected)

def example_e3n_scaled_solver():
    """Check solution of the example network, where everything is in S.I. units, and then everything is scaled in the solver
    """
    # Given
    Sbase = 1*MW #[W] ?
    Vbase = 10/np.sqrt(3)*kV #[V] ?
    Ybase = Sbase/(Vbase**2)
    b_pu = -10.
    g_pu = 1.
    b = b_pu*Ybase
    g = g_pu*Ybase
    elec_net = create_network(b,g)
    # this network is created with the nodal voltages and injected powers in p.u., so change to S.I.
    for node in elec_net.get_nodes():
        node.V *= Vbase
        for hl in node.get_half_links():
            hl.P *= Sbase
            hl.Q *= Sbase
    x0 = initialize_network(elec_net)
    
    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(elec_net,tol,max_iter,h,x0,scale_var='matrix',scale_var_params={'deltabase':1,'Vbase':Vbase,'Sbase':Sbase})
    
    # Then
    delta_sol_expected = np.array([-0.10000294, -0.10000295])
    V_sol_expected = np.array([0.97999991])*Vbase
    x_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    assert np.allclose(x_sol,x_sol_expected) 

def scaling():
    """Compare the different scaling options"""
    Sbase = 1*MW #[W] ?
    Vbase = 10/np.sqrt(3)*kV #[V] ?
    Ybase = Sbase/(Vbase**2)
    scale_var = 'per_unit'
    scale_var_params = {'Sbase':Sbase,'Vbase':Vbase,'deltabase':1}
    b_pu = -10.
    g_pu = 1.
    b = b_pu*Ybase
    g = g_pu*Ybase
    elec_net_pu = create_network(b_pu,g_pu)
    elec_net_SI = create_network(b,g)
    # the network is created with the nodal voltages and injected powers in p.u., so change to S.I.
    for node in elec_net_SI.get_nodes():
        node.V *= Vbase
        for hl in node.get_half_links():
            hl.P *= Sbase
            hl.Q *= Sbase
    
    # initialize the networks
    x0_pu = initialize_network(elec_net_pu)
    x0_SI = initialize_network(elec_net_SI)
    
    # compare convergence
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    # solve when everything is specified in per unit
    _,_,_,x_sol_pu,iters_pu,err_vec_pu = solve_system(elec_net_pu,tol,max_iter,h,x0_pu)
    # solve when everything is specified in S.I., without scaling
    _,_,_,x_sol_SI,iters_SI,err_vec_SI = solve_system(elec_net_SI,tol,max_iter,h,x0_SI)
    # solve when everything is specified in S.I., using per unit scaling
    elec_net_SI.reset_network(x0_SI)
    elec_net_SI.update_full(x0_SI)
    x0_SI_pu = initialize_network(elec_net_SI,scale_var=scale_var,scale_var_params=scale_var_params)
    _,_,_,x_sol_SI_pu,iters_SI_pu,err_vec_SI_pu = solve_system(elec_net_SI,tol,max_iter,h,x0_SI_pu,scale_var=scale_var,scale_var_params=scale_var_params)
    # solve when everything is specified in S.I., using scaling in solver
    elec_net_SI.reset_network(x0_SI)
    elec_net_SI.update_full(x0_SI)
    x0_SI_scaled = initialize_network(elec_net_SI)
    _,_,_,x_sol_scaled,iters_scaled,err_vec_scaled = solve_system(elec_net_SI,tol,max_iter,h,x0_SI_scaled,scale_var='matrix',scale_var_params=scale_var_params)
    # solve when everything is specified in per unit, without scaling, using scaled F for the stopping criterion (to match the accuracy of the unscaled solver)
    elec_net_pu.reset_network(x0_pu)
    elec_net_pu.update_full(x0_pu)
    x0_pu_DF = initialize_network(elec_net_pu)
    from meslf.load_flow.system_of_equations import NonLinearSystemElectrical
    nlsys = NonLinearSystemElectrical(elec_net_SI,scale_var='matrix',scale_var_params=scale_var_params)
    D_F = nlsys.DF()
    _,_,_,x_sol_pu_DF,iters_pu_DF,err_vec_pu_DF = solve_system(elec_net_pu,tol,max_iter,h,x0_pu_DF,D_F=D_F)
    
    
    fig = plt.figure('Convergence plot (|S_b| = {:.2e} W)'.format(Sbase))
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    max_iter_used = np.max([iters_pu,iters_SI,iters_SI_pu,iters_scaled])
    ax.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax.semilogy(np.asarray(range(0,iters_pu+1)),err_vec_pu,'s-',label='p.u.')
    ax.semilogy(np.asarray(range(0,iters_SI+1)),err_vec_SI,'*-',label='S.I., unscaled')
    ax.semilogy(np.asarray(range(0,iters_SI_pu+1)),err_vec_SI_pu,'d-',label='S.I., per unit scaling')
    ax.semilogy(np.asarray(range(0,iters_pu_DF+1)),err_vec_pu_DF,'o-',label='p.u. with $D_F$')
    ax.semilogy(np.asarray(range(0,iters_scaled+1)),err_vec_scaled,'.-',label='S.I., scaled solver')
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    # compare with actual solution. NB: THE 'ACTUAL SOLUTION' IS BASED ON THE SOLUTION I COMPUTED IN VERSION1, WHICH USED PER UNIT VALUES
    delta_sol_expected = np.array([-0.10000294, -0.10000295]) #[rad]
    V_sol_expected = np.array([0.97999991])*Vbase #[V]
    x_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))

    x_sol_pu_SI = x_sol_pu.copy() # solution of system when everything is specified in per unit, scaled back to S.I.
    x_sol_pu_SI[2] *= Vbase
    x_sol_SI_pu_SI = x_sol_SI_pu.copy() # solution of system when everything is specified in S.I., solved using per unit scaling, scaled back to S.I.
    x_sol_SI_pu_SI[2] *= Vbase
    x_sol_pu_DF_SI = x_sol_pu_DF.copy() # solution of system when everything is specified in per unit, scaled back to S.I.
    x_sol_pu_DF_SI[2] *= Vbase

    fig.suptitle('Final error ||x-x_sol||: p.u. = {:.2e}, S.I., unscaled = {:.2e}, S.I., per unit scaling = {:.2e}, p.u. with D_F = {:.2e}'.format(np.linalg.norm(x_sol_pu_SI-x_sol_expected),np.linalg.norm(x_sol_SI-x_sol_expected),np.linalg.norm(x_sol_SI_pu_SI-x_sol_expected),np.linalg.norm(x_sol_pu_DF_SI-x_sol_expected)))
    
    print('Errors. Par. in p.u., unscaled: {}'.format(err_vec_pu))
    print('Errors. Par. in S.I., p.u. scaling : {}'.format(err_vec_SI_pu))
    print('Errors. Par. in S.I., matrix scaling : {}'.format(err_vec_scaled))
    
def lin_solvers():
    """Compare the different linear solvers"""
    Sbase = 1*MW #[W] ?
    Vbase = 10/np.sqrt(3)*kV #[V] ?
    Ybase = Sbase/(Vbase**2)
    scale_var = 'per_unit'
    scale_var_params = {'Sbase':Sbase,'Vbase':Vbase,'deltabase':1}
    b_pu = -10.
    g_pu = 1.
    b = b_pu*Ybase
    g = g_pu*Ybase
    elec_net_pu = create_network(b_pu,g_pu)
    elec_net_SI = create_network(b,g)
    # the network is created with the nodal voltages and injected powers in p.u., so change to S.I.
    for node in elec_net_SI.get_nodes():
        node.V *= Vbase
        for hl in node.get_half_links():
            hl.P *= Sbase
            hl.Q *= Sbase
    
    # initialize the networks
    x0_pu = initialize_network(elec_net_pu)
    x0_SI = initialize_network(elec_net_SI)
    
    # compare convergence
    max_iter = 6
    tol = 1e-6
    
    fig_conv = plt.figure('Convergence plot S.I.')
    ax_conv = fig_conv.gca()
    
    fig_conv_pu = plt.figure('Convergence plot p.u.')
    ax_conv_pu = fig_conv_pu.gca()
    
    fig_conv_scaled = plt.figure('Convergence plot scaled')
    ax_conv_scaled = fig_conv_scaled.gca()
    
    max_iters_used = 0 
    h = None
    for lin_solver in ['solve','gmres','bicgstab']:
        _,_,_,x_sol_SI,iters_SI,err_vec_SI = solve_system(elec_net_SI,tol,max_iter,h,x0_SI,lin_solver=lin_solver,FD=False)
        # plot convergence
        max_iters_used = max(max_iters_used,iters_SI)
        ax_conv.semilogy(err_vec_SI,'.-',label=lin_solver)
    ax_conv.semilogy([0,max_iters_used],[tol,tol],'k:',label='tol')
    ax_conv.set_xlabel("Iteration k")
    ax_conv.set_ylabel("Error $||D_F F(x^k)||_2$")
    ax_conv.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv.legend()
    
    max_iters_used_pu = 0 
    h = None
    for lin_solver in ['solve','gmres','bicgstab']:
        _,_,_,x_sol_pu,iters_pu,err_vec_pu = solve_system(elec_net_pu,tol,max_iter,h,x0_pu,lin_solver=lin_solver,FD=False)
        # plot convergence
        max_iters_used_pu = max(max_iters_used_pu,iters_pu)
        ax_conv_pu.semilogy(err_vec_pu,'.-',label=lin_solver)
    ax_conv_pu.semilogy([0,max_iters_used_pu],[tol,tol],'k:',label='tol')
    ax_conv_pu.set_xlabel("Iteration k")
    ax_conv_pu.set_ylabel("Error $||D_F F(x^k)||_2$")
    ax_conv_pu.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_pu.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_pu.legend()
    
    max_iters_used_scaled = 0 
    h = None
    elec_net_SI.reset_network(x0_SI)
    elec_net_SI.update_full(x0_SI)
    x0_SI_scaled = initialize_network(elec_net_SI)
    for lin_solver in ['solve','gmres','bicgstab']:
        _,_,_,x_sol_scaled,iters_scaled,err_vec_scaled = solve_system(elec_net_SI,tol,max_iter,h,x0_SI_scaled,scale_var='matrix',scale_var_params={'deltabase':1,'Vbase':Vbase,'Sbase':Sbase},lin_solver=lin_solver,FD=False)
        # plot convergence
        max_iters_used_scaled = max(max_iters_used_scaled,iters_scaled)
        ax_conv_scaled.semilogy(err_vec_scaled,'.-',label=lin_solver)
    ax_conv_scaled.semilogy([0,max_iters_used_scaled],[tol,tol],'k:',label='tol')
    ax_conv_scaled.set_xlabel("Iteration k")
    ax_conv_scaled.set_ylabel("Error $||D_F F(x^k)||_2$")
    ax_conv_scaled.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_scaled.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_scaled.legend()
    
    # Use preconditioning instead of scaling beforehand
    elec_net_SI.reset_network(x0_SI)
    elec_net_SI.update_full(x0_SI)
    from meslf.load_flow.system_of_equations import NonLinearSystemElectrical
    nlsys = NonLinearSystemElectrical(elec_net_SI)
    F0 = nlsys.F(x0_SI)
    J0 = nlsys.J(x0_SI)
    
    from scipy.sparse.linalg import LinearOperator
    nlsys_scaled = NonLinearSystemElectrical(elec_net_SI,scale_var='matrix',scale_var_params=scale_var_params)
    D_F = nlsys_scaled.DF()
    D_F_inv = sps.diags(1/D_F.data[0])
    prec_list = ['I','D_F','D_F_inv','no prec']
    MM_norm = [np.linalg.norm(np.eye(len(x0_SI))), np.linalg.norm(D_F.todense()), np.linalg.norm(D_F_inv.todense()), 1]
    for ind_M,MM in enumerate([LinearOperator(shape=J0.shape, matvec=lambda x : np.eye(len(x0_SI)).dot(x), dtype=D_F.dtype),LinearOperator(shape=J0.shape, matvec=lambda x : D_F.dot(x), dtype=D_F.dtype), LinearOperator(shape=J0.shape, matvec=lambda x : D_F_inv.dot(x), dtype=D_F.dtype),None]):
        iter_gmres = 0
        error_gmres = np.linalg.norm(D_F.dot(F0))
        err_vec_gmres = list()
        err_vec_gmres.append(error_gmres)
        elec_net_SI.reset_network(x0_SI)
        elec_net_SI.update_full(x0_SI)
        x_new = x0_SI
        F_new = F0
        J_new = J0
        while error_gmres > tol and iter_gmres < max_iter:
            x_old = x_new
            F_old = F_new
            J_old = J_new
            bnrm2 = np.linalg.norm(F_old)
            bnrm2_scaled = np.linalg.norm(D_F.dot(F_old))
            
            tol_gmres = MM_norm[ind_M]*tol#/bnrm2_scaled
            print('bnrm2 = {}, bnrm2_scaled = {}'.format(bnrm2,bnrm2_scaled))
            print('tol_gmres = {}'.format(tol_gmres))
            global res_vec
            res_vec = list()

            dx = np.zeros(len(F_old))
            print('||F(x^k) - J(x^k)dx|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))))
            print('||J(x^k)dx - F(x^k)|| = {}'.format(np.linalg.norm(J_old.dot(dx) - F_old)))
            print('J(x^k)dx - F(x^k) = {}'.format(J_old.dot(dx) - F_old))
            print('||F(x^k) - J(x^k)dx||/||F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/bnrm2))
            dx,info = sps.linalg.gmres(J_old,F_old,M=MM,tol=tol_gmres,callback=callback_gmres,maxiter=20) # dx is unscaled
            print('||r^k|| = {}'.format(res_vec))
            print('||dx^k|| = {}'.format(dx))
            #print('||D_F F(x^k) - D_F J(x^k)dx|| = {}'.format(np.linalg.norm(D_F.dot(F_old) - D_F.dot(J_old).dot(dx))))
            #print('||D_F F(x^k) - D_F J(x^k)dx||/||D_F F(x^k)|| = {}'.format((np.linalg.norm(D_F.dot(F_old) - D_F.dot(J_old).dot(dx)))/bnrm2_scaled))
            #print('||F(x^k) - J(x^k)dx|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))))
            #print('||F(x^k) - J(x^k)dx||/||F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/bnrm2))
            #print('||D_F F(x^k) - D_F J(x^k)dx||/||F(x^k)|| = {}'.format((np.linalg.norm(D_F.dot(F_old) - D_F.dot(J_old).dot(dx)))/bnrm2))
            #print('||F(x^k) - J(x^k)dx||/||D_F F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/bnrm2_scaled))
            
            #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx|| = {}'.format(np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx))))
            #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx||/||D_F F(x^k)|| = {}'.format((np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx)))/bnrm2_scaled))
            #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx||/||F(x^k)|| = {}'.format((np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx)))/bnrm2))
            #print('||F(x^k) - J(x^k)dx||/||D_F F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/bnrm2_scaled))
            
            #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx|| = {}'.format(np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx))))
            #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx||/||D_F_inv F(x^k)|| = {}'.format((np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx)))/np.linalg.norm(D_F_inv.dot(F_old))))
            #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx||/||F(x^k)|| = {}'.format((np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx)))/bnrm2))
            #print('||D_F_inv (F(x^k) - J(x^k)dx)||/||F(x^k)|| = {}'.format(np.linalg.norm(D_F_inv.dot(F_old - J_old.dot(dx)))/bnrm2))
            #print('||F(x^k) - J(x^k)dx||/||D_F_inv F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/np.linalg.norm(D_F_inv.dot(F_old))))
            
            print('||F(x^k) - J(x^k)dx|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))))
            print('||J(x^k)dx - F(x^k)|| = {}'.format(np.linalg.norm(J_old.dot(dx) - F_old)))
            print('J(x^k)dx - F(x^k) = {}'.format(J_old.dot(dx) - F_old))
            print('||F(x^k) - J(x^k)dx||/||F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/bnrm2))
            if ind_M<3:
                print('||MM F(x^k) - MM J(x^k)dx|| = {}'.format(np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx)))))
                print('||MM F(x^k) - MM J(x^k)dx||/||MM F(x^k)|| = {}'.format((np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx))))/np.linalg.norm(MM.matvec(F_old))))
                print('||MM F(x^k) - MM J(x^k)dx||/||F(x^k)|| = {}'.format((np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx))))/bnrm2))
                print('||F(x^k) - J(x^k)dx||/||MM F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/np.linalg.norm(MM.matvec(F_old))))
            
            
            #print('||r|| = {} {} {}'.format(res_vec[0], res_vec[-2], res_vec[-1]))
            x_new = x_old - dx
            F_new = nlsys.F(x_new)
            J_new = nlsys.J(x_new)
            error_gmres = np.linalg.norm(D_F.dot(F_new)) # scaled error
            err_vec_gmres.append(error_gmres)
            print('||D_F F(x^k)|| = {}'.format(error_gmres))
            
            print('matvecs = {}, for NR iteration {}, with M={}\n'.format(len(res_vec),iter_gmres,prec_list[ind_M]))
            iter_gmres += 1
            max_iters_used_scaled = max(max_iters_used_scaled,iter_gmres)
            #print('iteration {}, error = {}'.format(iter_gmres,error_gmres))
        
        ax_conv_scaled.semilogy(err_vec_gmres,'.-',label='prec. gmres, M={}'.format(prec_list[ind_M]))
    
    elec_net_SI.reset_network(x0_SI)
    elec_net_SI.update_full(x0_SI)
    error_bicgstab = np.linalg.norm(D_F.dot(F0))
    err_vec_bicgstab = list()
    err_vec_bicgstab.append(error_bicgstab)
    x_new = x0_SI
    F_new = F0
    J_new = J0
    iter_bicgstab = 0
    while error_bicgstab > tol and iter_bicgstab < max_iter:
        x_old = x_new
        F_old = F_new
        J_old = J_new
        bnrm2 = np.linalg.norm(F_old)
        bnrm2_scaled = np.linalg.norm(D_F.dot(F_old))
        tol_bicgstab = tol/bnrm2_scaled
        print('bnrm2 = {}, bnrm2_scaled = {}'.format(bnrm2,bnrm2_scaled))
        print('tol_bicgstab = {}'.format(tol_bicgstab))
        res_vec = list()
        global A
        A = J_old
        global rhs
        rhs = F_old
        dx,info = sps.linalg.bicgstab(J_old,F_old,M=MM,tol=tol_bicgstab,callback=callback_bicgstab) # dx is unscaled. NB. The description for M is different from the description of M for gmres. Don't know if I need to use D_F or D_F_inv!!
        
        print('||r^k|| = {}'.format(res_vec))
        #print('1/r^k = {}'.format(1/np.array(res_vec)))
        print('dx^k = {}'.format(dx))
        #print('||D_F F(x^k) - D_F J(x^k)dx|| = {}'.format(np.linalg.norm(D_F.dot(F_old) - D_F.dot(J_old).dot(dx))))
        #print('||D_F F(x^k) - D_F J(x^k)dx||/||D_F F(x^k)|| = {}'.format((np.linalg.norm(D_F.dot(F_old) - D_F.dot(J_old).dot(dx)))/bnrm2_scaled))
        #print('||F(x^k) - J(x^k)dx|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))))
        #print('||F(x^k) - J(x^k)dx||/||F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/bnrm2))
        #print('||D_F F(x^k) - D_F J(x^k)dx||/||F(x^k)|| = {}'.format((np.linalg.norm(D_F.dot(F_old) - D_F.dot(J_old).dot(dx)))/bnrm2))
        ##print('||r|| = {} {} {}'.format(res_vec[0], res_vec[-2], res_vec[-1]))
        
        #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx|| = {}'.format(np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx))))
        #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx||/||D_F F(x^k)|| = {}'.format((np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx)))/bnrm2_scaled))
        #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx||/||F(x^k)|| = {}'.format((np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx)))/bnrm2))
        #print('||F(x^k) - J(x^k)dx||/||D_F F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/bnrm2_scaled))
        
        #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx|| = {}'.format(np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx))))
        #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx||/||D_F_inv F(x^k)|| = {}'.format((np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx)))/np.linalg.norm(D_F_inv.dot(F_old))))
        #print('||D_F_inv F(x^k) - D_F_inv J(x^k)dx||/||F(x^k)|| = {}'.format((np.linalg.norm(D_F_inv.dot(F_old) - D_F_inv.dot(J_old).dot(dx)))/bnrm2))
        #print('||D_F_inv (F(x^k) - J(x^k)dx)||/||F(x^k)|| = {}'.format(np.linalg.norm(D_F_inv.dot(F_old - J_old.dot(dx)))/bnrm2))
        #print('||F(x^k) - J(x^k)dx||/||D_F_inv F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/np.linalg.norm(D_F_inv.dot(F_old))))
        
        print('||F(x^k) - J(x^k)dx|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))))
        print('||J(x^k)dx - F(x^k)|| = {}'.format(np.linalg.norm(J_old.dot(dx) - F_old)))
        print('J(x^k)dx - F(x^k) = {}'.format(J_old.dot(dx) - F_old))
        print('||F(x^k) - J(x^k)dx||/||F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/bnrm2))
        if ind_M<3:
            print('||MM F(x^k) - MM J(x^k)dx|| = {}'.format(np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx)))))
            print('||MM F(x^k) - MM J(x^k)dx||/||MM F(x^k)|| = {}'.format((np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx))))/np.linalg.norm(MM.matvec(F_old))))
            print('||MM F(x^k) - MM J(x^k)dx||/||F(x^k)|| = {}'.format((np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx))))/bnrm2))
            print('||F(x^k) - J(x^k)dx||/||MM F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/np.linalg.norm(MM.matvec(F_old))))
        
        x_new = x_old - dx
        F_new = nlsys.F(x_new)
        J_new = nlsys.J(x_new)
        error_bicgstab = np.linalg.norm(D_F.dot(F_new)) # scaled error
        err_vec_bicgstab.append(error_bicgstab)
        
        print('||D_F F(x^k)|| = {}'.format(error_bicgstab))
        
        print('matvecs = {}, for NR iteration {}, with M={}\n'.format(len(res_vec),iter_gmres,prec_list[ind_M]))
        iter_bicgstab += 1
        max_iters_used_scaled = max(max_iters_used_scaled,iter_bicgstab)
        #print('iteration {}, error = {}'.format(iter_bicgstab,error_bicgstab))
    ax_conv_scaled.semilogy(err_vec_bicgstab,'.-',label='prec. bicgstab')
    
    ax_conv_scaled.semilogy([0,max_iters_used_scaled],[tol,tol],'k:')
    ax_conv_scaled.legend() 
    
if __name__ == '__main__':
    scaling()
    
    res_vec = []
    A = None
    rhs = None
    def callback_gmres(r):
        """
        Parameters
        ----------
        r : 
            Residual vector
        """
        print('residual = {}'.format(r))
        global res_vec
        res_vec.append(np.linalg.norm(r))
        
    def callback_bicgstab(x):
        """
        Parameters
        ----------
        x : 
            Solution of Ax=b
        """
        #print('x = {}'.format(x))
        global A
        global rhs
        r = rhs - A.dot(x)
        print('residual = {}'.format(r))
        global res_vec
        res_vec.append(np.linalg.norm(r))
        
    lin_solvers()
    
    plt.show()
    
    
