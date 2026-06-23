"""Example of a heat network with 4 nodes, connected with standard pipes with a constant pipe constant
"""
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink
from meslf.networks.carrier import Water
import meslf.load_flow.system_of_equations as NLS
import numpy as np
import scipy.sparse as sps
import pytest
import matplotlib.pyplot as plt

def create_network():
    """Create a test heat network

    Returns
    -------
    heat_net : HeatNetwork
        The test network
    water : Carrier
        The carrier in the network
    """
    # carrier
    rho = 960. #[kg/m^3]
    Cp = 4.182e3 #[J/(kg K)]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp,rho=rho)
    # network
    MW = 1e6 #[W]
    Ta = 10. #[C]
    heat_net = HeatNetwork('test heat network',Ta=Ta)
    hn0 = HeatNode('hn0',node_type=0,x=1,y=2,Ts=100.,p=(12.+2*np.sqrt(5.))*rho*g) # source slack node
    hn0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn1 = HeatNode('hn1',node_type=2,x=1,y=1) # junction node
    hn2 = HeatNode('hn2',node_type=1,x=0,y=0,Tr_hl=50.,dphi=0.86220755547510397*MW) # sink node
    hn2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hn3 = HeatNode('hn3',node_type=1,x=2,y=0,Ts_hl=100.,dphi=-0.21069237249666534*MW) # source node
    hn3.half_links[0].set_type('heat_exchanger',{'carrier':water})

    C = np.sqrt(1/(rho*g))
    L = 400. #[m]
    D = 0.01 #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D) #[W/(m^2 K)]
    link_type = 'standard_resistor'
    link_params = {'L':L,'D':D,'U':U,'C':C,'carrier':water}
    hl0 = HeatLink('hl0',hn0,hn1,link_type=link_type,link_params=link_params)
    hl1 = HeatLink('hl1',hn1,hn2,link_type=link_type,link_params=link_params.copy())
    hl2 = HeatLink('hl2',hn1,hn3,link_type=link_type,link_params=link_params.copy())
    hl3 = HeatLink('hl2',hn3,hn2,link_type=link_type,link_params=link_params.copy())

    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    heat_net.add_link(hl3)
    return heat_net, water

def initialize_network(network,carrier,scale_var=None,scale_var_params=None,formulation='standard'):
    """Sets values of network variables to be used for initial guess.

    Parameters
    ----------
    network : HeatlNetwork
        The network to be initialized
    carrier : Water
        The carrier in the network

    Returns
    -------
    x0 : np array
        initial guess
    """
    Ta = network.Ta
    rho = carrier.rhon
    g = carrier.g
    m_init = np.array([4., 2., -2., 3.])
    p_init = np.array([6.,2.,4.])*rho*g
    Ts_init = np.array([99.,98.,100.])#-Ta
    Tr_init = np.array([49.,49.,50.,48.])#-Ta
    if formulation == 'half_link_flow':
        m_hl_init = np.array([3,-2])
        x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    else:
        x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    network.initialize()
    network.update(x_init,formulation=formulation)
    x0 = network.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def solve_system(network,tol,max_iter,h,x0,formulation='standard',scale_var=None,scale_var_params=None,D_F=np.array([]),D_x=np.array([]),det_tol=1e-8):
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
    network.update(x0,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print("\nSolving system using FD Jacobian")
    x_sol_FD,iters_FD,err_vec_FD,m_vec_FD,p_vec_FD,Ts_vec_FD,Tr_vec_FD,m_hl_vec_FD,phi_hl_vec_FD,Ts_hl_vec_FD,Tr_hl_vec_FD = network.solve_network(tol,max_iter,h=h,formulation=formulation,solver='NR_FD',scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x)

    network.reset_network(x0,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    #network.update(x0,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print("\nSolving system using analytical Jacobian")
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = network.solve_network(tol,max_iter,formulation=formulation,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x,det_tol=det_tol)

    return x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning") # UserWarning: Tr>Ts for node hn1 at some iteration
def example_h4n_pu():
    """Check solution of the example network, using per unit scaling"""
    # Given
    heat_net, water = create_network()
    #scaling
    scale_var = 'per_unit'
    rho = water.rhon
    g = water.g
    pbase = 9.*rho*g
    mbase = 3.
    phibase = mbase * water.Cp
    Tbase = 80. #[C]
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}
    Ta = heat_net.Ta
    x0 = initialize_network(heat_net, water, scale_var=scale_var,scale_var_params=scale_var_params)
    print('x0 = {}'.format(x0))

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 1000
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(heat_net,tol,max_iter,h,x0, scale_var=scale_var,scale_var_params=scale_var_params)
    print('errors: {}'.format(err_vec))
    # Then
    m_sol_expected = np.array([3.25956121,  2.24954466,  1.01001655,  2.01005417])
    p_sol_expected = np.array([55068.44279688,    7411.13782231, 45461.23439602])
    Ts_sol_expected = np.array([99.47335791,   98.40153064,   98.89186955])#-Ta
    Tr_sol_expected = np.array([49.18784669,  49.41850719,  50.,  49.62112735])#-Ta
    if scale_var == 'per_unit':
        m_sol_expected = m_sol_expected/mbase
        p_sol_expected = p_sol_expected/pbase
        Ts_sol_expected = Ts_sol_expected/Tbase
        Tr_sol_expected = Tr_sol_expected/Tbase
    x_sol_expected = np.concatenate((m_sol_expected,p_sol_expected,Ts_sol_expected,Tr_sol_expected))

    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning") # UserWarning: Tr>Ts for node hn1 at some iteration
def example_h4n_scaled_solver():
    """Check solution of the example network, using scaling in solver"""
    # Given
    heat_net, water = create_network()
    #scaling
    rho = water.rhon
    g = water.g
    pbase = 9.*rho*g
    mbase = 3.
    phibase = mbase * water.Cp
    Tbase = 80. #[C]
    Ta = heat_net.Ta
    x0 = initialize_network(heat_net, water)
    print('x0 = {}'.format(x0))

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 1000
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(heat_net,tol,max_iter,h,x0,scale_var='matrix',scale_var_params={'mbase':mbase,'phbase':pbase,'Tbase':Tbase,'phibase':phibase})
    print('errors: {}'.format(err_vec))
    # Then
    m_sol_expected = np.array([3.25956121,  2.24954466,  1.01001655,  2.01005417])
    p_sol_expected = np.array([55068.44279688,    7411.13782231, 45461.23439602])
    Ts_sol_expected = np.array([99.47335791,   98.40153064,   98.89186955])#-Ta
    Tr_sol_expected = np.array([49.18784669,  49.41850719,  50.,  49.62112735])#-Ta
    x_sol_expected = np.concatenate((m_sol_expected,p_sol_expected,Ts_sol_expected,Tr_sol_expected))

    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning") # UserWarning: Tr>Ts for node hn1 at some iteration
def example_h4n_unknown_half_link_scaled_solver():
    """Check solution of the example network, using the 'half_link_flow' formulation, and using scaling in solver"""
    # Given
    heat_net, water = create_network()
    #scaling
    rho = water.rhon
    g = water.g
    pbase = 9.*rho*g
    mbase = 3.
    phibase = mbase * water.Cp
    Tbase = 80. #[C]
    Ta = heat_net.Ta
    x0 = initialize_network(heat_net, water, formulation='half_link_flow')
    print('x0 = {}'.format(x0))

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 1000
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(heat_net,tol,max_iter,h,x0,scale_var='matrix',scale_var_params={'mbase':mbase,'phbase':pbase,'Tbase':Tbase,'phibase':phibase}, formulation='half_link_flow')
    print('errors: {}'.format(err_vec))
    # Then
    m_sol_expected = np.array([3.25956121,  2.24954466,  1.01001655,  2.01005417])
    m_hl_sol_expected = np.array([4.2595988299999998, -1.00003762])
    p_sol_expected = np.array([55068.44279688,    7411.13782231, 45461.23439602])
    Ts_sol_expected = np.array([99.47335791,   98.40153064,   98.89186955])
    Tr_sol_expected = np.array([49.18784669,  49.41850719,  50.,  49.62112735])
    x_sol_expected = np.concatenate((m_sol_expected,m_hl_sol_expected,p_sol_expected,Ts_sol_expected,Tr_sol_expected))

    assert np.allclose(x_sol,x_sol_expected)
    
if __name__ == '__main__':
    heat_net, water = create_network()
    #scaling
    scale_var = 'per_unit'
    rho = water.rhon
    g = water.g
    pbase = 9.*rho*g
    mbase = 3.
    phibase = mbase * water.Cp
    Tbase = 80. #[C]
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}
    Ta = heat_net.Ta

    # initalize network, unscaled
    x0 = initialize_network(heat_net, water)
    nlsys = NLS.NonLinearSystemHeat(heat_net,formulation='standard')
    J0 = nlsys.J(x0)
    # initalize network, per unit
    x0_pu = initialize_network(heat_net, water, scale_var=scale_var,scale_var_params=scale_var_params)
    # initalize network, unscaled, using unknown half link formulation
    heat_net.reset_network(x0)
    x0_hl = initialize_network(heat_net, water, formulation='half_link_flow')
    nlsys_hl = NLS.NonLinearSystemHeat(heat_net,formulation='half_link_flow')
    J0_hl = nlsys_hl.J(x0_hl)
    # initalize network, per unit, using unknown half link formulation
    heat_net.reset_network(x0_hl,formulation='half_link_flow')
    x0_pu_hl = initialize_network(heat_net, water, formulation='half_link_flow', scale_var=scale_var,scale_var_params=scale_var_params)
    # plot network
    fig_top = plt.figure('Network topology')
    ax_top = fig_top.gca()
    heat_net.draw_network(ax_top)
    plt.axis('equal')
    plt.axis('off')

    # Jacobian matrices at initial guess 
    fig_J = plt.figure('Jacobian standard formulation')
    plt.spy(J0)
    ax_J = plt.gca()
    nlsys.plot_J_overlay(ax_J)
    
    fig_J_hl = plt.figure('Jacobian unknown half link flow formulation')
    plt.spy(J0_hl)
    ax_J_hl = plt.gca()
    nlsys_hl.plot_J_overlay(ax_J_hl)
    
    # compare convergence
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    # using 'standard' formulation
    # solve when everything is specified in S.I., unscaled
    _,_,_,x_sol_SI,iters_SI,err_vec_SI = solve_system(heat_net,tol,max_iter,h,x0)
    # solve when everything is specified in per unit
    heat_net.update_full(x0)
    _,_,_,x_sol_pu,iters_pu,err_vec_pu = solve_system(heat_net,tol,max_iter,h,x0_pu, scale_var=scale_var,scale_var_params=scale_var_params)
    # solve when everything is specified in S.I., using scaling in solver
    heat_net.update_full(x0)
    _,_,_,x_sol_scaled,iters_scaled,err_vec_scaled = solve_system(heat_net,tol,max_iter,h,x0,scale_var='matrix',scale_var_params={'mbase':mbase,'phbase':pbase,'Tbase':Tbase,'phibase':phibase})
    # using 'half_link_flow' formulation
    heat_net.reset_network(x0_hl,formulation='half_link_flow')
    # solve when everything is specified in S.I., unscaled
    _,_,_,x_sol_SI_hl,iters_SI_hl,err_vec_SI_hl = solve_system(heat_net,tol,max_iter,h,x0_hl,formulation='half_link_flow')
    # solve when everything is specified in per unit
    heat_net.reset_network(x0_hl,formulation='half_link_flow')
    _,_,_,x_sol_pu_hl,iters_pu_hl,err_vec_pu_hl = solve_system(heat_net,tol,max_iter,h,x0_pu_hl,formulation='half_link_flow',scale_var=scale_var,scale_var_params=scale_var_params)
    # solve when everything is specified in S.I., using scaling in solver
    heat_net.reset_network(x0_hl,formulation='half_link_flow')
    _,_,_,x_sol_scaled_hl,iters_scaled_hl,err_vec_scaled_hl = solve_system(heat_net,tol,max_iter,h,x0_hl,formulation='half_link_flow',scale_var='matrix',scale_var_params={'mbase':mbase,'phbase':pbase,'Tbase':Tbase,'phibase':phibase})
    
    # plot convergence
    fig = plt.figure('Convergence plot H4N')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    max_iter_used = np.max([iters_pu,iters_SI,iters_scaled,iters_pu_hl,iters_SI_hl,iters_scaled_hl])
    ax.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax.semilogy(np.asarray(range(0,iters_pu+1)),err_vec_pu,'s-',label='p.u.')
    ax.semilogy(np.asarray(range(0,iters_SI+1)),err_vec_SI,'o-',label='S.I., unscaled')
    ax.semilogy(np.asarray(range(0,iters_scaled+1)),err_vec_scaled,'.-',label='S.I., scaled solver')
    ax.semilogy(np.asarray(range(0,iters_pu_hl+1)),err_vec_pu_hl,'s--',label='p.u., hl')
    ax.semilogy(np.asarray(range(0,iters_SI_hl+1)),err_vec_SI_hl,'o--',label='S.I., unscaled, hl')
    ax.semilogy(np.asarray(range(0,iters_scaled_hl+1)),err_vec_scaled_hl,'.--',label='S.I., scaled solver, hl')
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)

    # plot solution 
    heat_net.update_full(x_sol_scaled)
    fig_sol = plt.figure('Network solution')
    ax_sol = fig_sol.gca()
    heat_net.draw_network_value(ax_sol,plot_loss=True)
    plt.axis('equal')
    plt.axis('off')
    
    plt.show()
