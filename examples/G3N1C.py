"""Example of a gas network with 3 (actual nodes) and 1 compressor"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink
from meslf.networks.carrier import Gas
import numpy as np

hour = 3600 #[s]
mbar = 1e2 #[Pa]

def create_network(carrier,r,D,L):
    """Create a network with low pressure flow pipelines and one compressor
    
    Parameters
    ----------
    carrier : Carrier
        gas carrier
    r : float
        compressor ratio
    D : list
        list of pipe diameters in m
    L : list
        list of pipe lengths in m
    
    Returns
    -------
    gas_net : GasNetwork
        The network
    """
    gas_net = GasNetwork('test gas network')
    g0 = GasNode('g0',node_type=0,p=30.*mbar) # reference node
    g1 = GasNode('g1',node_type=1,q=100.*carrier.rhon/hour) # load node
    g2 = GasNode('g2',node_type=2) # slack node
    g3 = GasNode('g3',node_type=3,q=0.,p=16.7083*mbar) # reference load node
    
    l0 = GasLink('l0',g0,g1,link_type = 'pipe_low_pres_pole',link_params = {'carrier':carrier, 'D':D[0], 'L':L[0]})
    l1 = GasLink('l1',g0,g2,link_type = 'pipe_low_pres_pole',link_params = {'carrier':carrier, 'D':D[1], 'L':L[1]})
    l2 = GasLink('l2',g3,g2,link_type = 'pipe_low_pres_pole',link_params = {'carrier':carrier, 'D':D[2], 'L':L[2]})
    l3 = GasLink('l3',g1,g3,link_type = 'compressor',link_params = {'carrier':carrier, 'r':r})
    
    gas_net.add_link(l0)
    gas_net.add_link(l1)
    gas_net.add_link(l2)
    gas_net.add_link(l3)
    return gas_net

def initialize_network(network,carrier,scale_var=None,scale_var_params=None):
    """Sets values of network variables to be used for initial guess.
    
    Parameters
    ----------
    network : GasNetwork
        The network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    form = 'full'
    p_init = np.array([15.,20.])*mbar
    q_init = np.array([200.,100.,100.,100.])*carrier.rhon/hour
    x_init = np.concatenate((q_init,p_init))
    network.initialize()
    network.update(x_init,formulation=form) # update without scaling, since x_init is unscaled
    x0 = network.set_x_init(formulation=form,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def solve_system(network,tol,max_iter,h,x0,scale_var=None,scale_var_params=None):
    """Solve the network using analytical Jacobian and FD Jacobian, with basic NR
    
    Parameters
    ----------
    network : GasNetwork
        The network to be initialized
    form : string
        Formulation to be used.
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
    form = 'full'
    network.update(x0,formulation=form,scale_var=scale_var,scale_var_params=scale_var_params)
    print("\nSolving system using FD Jacobian")
    x_sol_FD,iters_FD,err_vec_FD,p_sol_FD,q_sol_FD,q_inj_FD = network.solve_network(tol,max_iter,h=h,formulation=form,solver='NR_FD',scale_var=scale_var,scale_var_params=scale_var_params)
    
    network.update(x0,formulation=form,scale_var=scale_var,scale_var_params=scale_var_params)
    print("\nSolving system using analytical Jacobian")
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = network.solve_network(tol,max_iter,formulation=form,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    
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
    
def print_full_solution(x_sol,network):
    """Print the full set of network parameters
    
    Parameters
    ----------
    x_sol : np array
        solution vector
    network : GasNetwork
        The network corresponding to the solution vector
    form : string
        Formulation to be used.
    """
    form = 'full'
    p_vec,q_vec,q_inj = network.update_full(x_sol,formulation=form)
    
    node_order = []
    edge_order = []
    for n in network.get_nodes():
        node_order.append(n.number)
    for n in network.get_links():
        edge_order.append(n.number)
    
    np.set_printoptions(precision=4)
    print("\nFull solution using analytical method:")
    print("Solution for pressure: p = {}".format(p_vec[node_order]))
    print("Solution for injected flow: q_inj in m^3/h = {}".format(q_inj[node_order]))
    print("Solution for edge flows: q in m^3/h = {}".format(q_vec[edge_order]))
    
def example_g3nc1():
    """Check the solution of the example network
    """
    #Given
    r = 1.5
    # link parameters
    r = 1.5 # compressor ratio
    L = [500., 500., 500., 500.] #[m]
    D = [.100, .100, .100, .100] #[m]
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    scale_var = 'per_unit'
    qbase = 150.*gas.rhon/hour
    pbase = 30.*mbar
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    # create network
    gas_net = create_network(gas,r,D,L)
    # initalize network
    x0 = initialize_network(gas_net,gas,scale_var=scale_var,scale_var_params=scale_var_params)
    
    #When 
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,tol,max_iter,h,x0,scale_var=scale_var,scale_var_params=scale_var_params)
    p_sol_expected = np.array([11.13886667, 13.00551072])*mbar/pbase
    q_sol_expected = np.array([180.42676625688685, 171.26599093800243, 79.943208033602531, 79.943208033602531])*gas.rhon/hour/qbase
    x_sol_expected = np.concatenate((q_sol_expected,p_sol_expected))
    
    #Then
    assert np.allclose(x_sol,x_sol_expected,rtol=1e-2) 
    
if __name__ == '__main__':
    # link parameters
    r = 1.5 # compressor ratio
    L = [500., 500., 500., 500.] #[m]
    D = [.100, .100, .100, .100] #[m]
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    scale_var = 'per_unit'
    qbase = 150.*gas.rhon/hour
    pbase = 30.*mbar
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    
    # create networks
    gas_net = create_network(gas,r,D,L)
        
    # initalize network
    x0 = initialize_network(gas_net,gas,scale_var=scale_var,scale_var_params=scale_var_params)
    
    # solve the network  
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,tol,max_iter,h,x0)
    
    # print all results
    print_results(x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec)
    print_full_solution(x_sol,gas_net)
