"""Example of a gas network with 4 nodes and 5 links.

Based on the example shown in Figure 6.1 in Osiadacz, and described in example 6.2.3.
"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink
from meslf.networks.carrier import Gas
import numpy as np
import matplotlib.pyplot as plt
from meslf.utils.constants import hour, mbar, bar, mm


def create_test_network_linear(alpha):
    """Create the network with linear pipelines

    Parameters
    ----------
    alpha : float
        link parameter

    Returns
    -------
    gas_net : GasNetwork
        the network
    """
    gas_net = GasNetwork('test gas network')
    g0 = GasNode('g0',node_type=0,p=30.) # reference node
    g1 = GasNode('g1',node_type=3,p=9.625,q=250.) # reference load node
    g2 = GasNode('g2',node_type=2) # slack node
    g3 = GasNode('g3',node_type=1,q=180.)

    l0 = GasLink('l0',g0,g1,link_type = 'pipe_linear',link_params = {'alpha':alpha})
    l1 = GasLink('l1',g0,g2,link_type = 'pipe_linear',link_params = {'alpha':alpha})
    l2 = GasLink('l2',g0,g3,link_type = 'pipe_linear',link_params = {'alpha':alpha})
    l3 = GasLink('l3',g2,g1,link_type = 'pipe_linear',link_params = {'alpha':alpha})
    l4 = GasLink('l4',g2,g3,link_type = 'pipe_linear',link_params = {'alpha':alpha})
    gas_net.add_link(l0)
    gas_net.add_link(l1)
    gas_net.add_link(l2)
    gas_net.add_link(l3)
    gas_net.add_link(l4)
    return gas_net

def initialize_network_linear(network,form):
    """Sets values of network variables to be used for initial guess.

    Parameters
    ----------
    network : GasNetwork
        The network to be initialized
    form : string
        Formulation to be used.

    Returns
    -------
    x0 : np array
        initial guess
    """
    if form == 'nodal':
        x_init = np.array([24.,28.])
    elif form == 'full':
        p = np.array([24.,28.])
        q = np.array([200.,150.,160.,50.,15.])
        x_init = np.concatenate((q,p))
    network.initialize()
    network.update(x_init,formulation=form)
    x0 = network.set_x_init(formulation=form)
    return x0

def create_test_network_low_pres(carrier,D,L):
    """Create the network with low pressure pipelines

    Parameters
    ----------
    carrier : Carrier
        gas carrier
    D : list
        list of pipe diameters in m
    L : list of pipe lengths in m

    Returns
    -------
    gas_net : GasNetwork
        the network
    """
    rho = carrier.rhon
    gas_net = GasNetwork('test gas network')
    g0 = GasNode('g0',node_type=0,p=30.*mbar) # reference node
    g1 = GasNode('g1',node_type=3,p=25.03542399*mbar,q=250.*rho/hour) # reference load node
    g2 = GasNode('g2',node_type=2) # slack node
    g3 = GasNode('g3',node_type=1,q=180.*rho/hour)

    l0 = GasLink('l0',g0,g1,link_type = 'pipe_low_pres_pole',link_params = {'carrier':carrier, 'D':D[0], 'L':L[0]})
    l1 = GasLink('l1',g0,g2,link_type = 'pipe_low_pres_pole',link_params = {'carrier':carrier, 'D':D[1], 'L':L[1]})
    l2 = GasLink('l2',g0,g3,link_type = 'pipe_low_pres_pole',link_params = {'carrier':carrier, 'D':D[2], 'L':L[2]})
    l3 = GasLink('l3',g2,g1,link_type = 'pipe_low_pres_pole',link_params = {'carrier':carrier, 'D':D[3], 'L':L[3]})
    l4 = GasLink('l4',g2,g3,link_type = 'pipe_low_pres_pole',link_params = {'carrier':carrier, 'D':D[4], 'L':L[4]})
    gas_net.add_link(l0)
    gas_net.add_link(l1)
    gas_net.add_link(l2)
    gas_net.add_link(l3)
    gas_net.add_link(l4)
    return gas_net

def initialize_network_low_pres(network,form,carrier,scale_var=None,scale_var_params=None,**kwargs):
    """Sets values of network variables to be used for initial guess.

    Parameters
    ----------
    network : GasNetwork
        The network to be initialized
    form : string
        Formulation to be used.

    Returns
    -------
    x0 : np array
        initial guess
    """
    rho = carrier.rhon
    if form == 'nodal':
        x_init = np.array([24.,28.])*mbar
    elif form == 'full':
        p = np.array([24.,28.])*mbar
        q = np.array([200.,110.,200.,50.,15.])*rho/hour
        x_init = np.concatenate((q,p))
    network.initialize()
    network.update(x_init,formulation=form,**kwargs) # update without scaling, since x_init is unscaled
    x0 = network.set_x_init(formulation=form,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)
    return x0

def create_test_network_Osiadacz_Lacey():
    """Create the network with low pressure pipes with Lacey's equation (4.22) as used by Osiadacz

    Parameters
    ----------
    alpha : float
        link parameter

    Returns
    -------
    gas_net : GasNetwork
        the network
    """
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    rho = carrier.rhon

    gas_net = GasNetwork('test gas network')
    g0 = GasNode('1',x=1,y=-1,node_type=0,p=30.*mbar) # reference node
    g1 = GasNode('2',x=0,y=0,node_type=1,q=250.*rho/hour) # load node
    g2 = GasNode('3',x=1,y=0,node_type=1,q=100*rho/hour) # load node
    g3 = GasNode('4',x=2,y=0,node_type=1,q=180.*rho/hour) # load node

    link_type_Lacey = 'resistor'#'pipe_low_pres_pole'
    D = np.array([150,100,150,100,100])*mm # m (not used for resistor)
    L = [680,500,420,600,340] # m (not used for resistor)
    K = np.array([1.04782,5.85065,0.64718,7.02078,3.97844])*1e-4 # for mbar and m3/hour
    C = (1/np.sqrt(K*(mbar/(rho/hour)**2))) # for S.I. units
    l0 = GasLink('e1',g0,g1,link_type = link_type_Lacey,link_params = {'carrier':carrier, 'D':D[0], 'L':L[0],'C':C[0]})
    l1 = GasLink('e2',g0,g2,link_type = link_type_Lacey,link_params = {'carrier':carrier, 'D':D[1], 'L':L[1],'C':C[1]})
    l2 = GasLink('e3',g0,g3,link_type = link_type_Lacey,link_params = {'carrier':carrier, 'D':D[2], 'L':L[2],'C':C[2]})
    l3 = GasLink('e4',g2,g1,link_type = link_type_Lacey,link_params = {'carrier':carrier, 'D':D[3], 'L':L[3],'C':C[3]})
    l4 = GasLink('e5',g2,g3,link_type = link_type_Lacey,link_params = {'carrier':carrier, 'D':D[4], 'L':L[4],'C':C[4]})
    gas_net.add_link(l0)
    gas_net.add_link(l1)
    gas_net.add_link(l2)
    gas_net.add_link(l3)
    gas_net.add_link(l4)
    return gas_net

def initialize_network_Osiadacz_Lacey(network,form,scale_var=None,scale_var_params=None,**kwargs):
    """Sets values of network variables to be used for initial guess.

    Parameters
    ----------
    network : GasNetwork
        The network to be initialized
    form : string
        Formulation to be used.

    Returns
    -------
    x0 : np array
        initial guess
    """
    carrier = network.links[0].link_params.get('carrier')
    rho = carrier.rhon
    p_init = np.array([23.4511,24.1493,27.9031])*mbar
    if form == 'nodal':
        x_init = p_init
    elif form == 'full':
        p = p_init
        q = np.array([200.,110.,200.,50.,15.])*rho/hour
        x_init = np.concatenate((q,p))
    network.initialize()
    network.update(x_init,formulation=form,**kwargs) # update without scaling, since x_init is unscaled
    x0 = network.set_x_init(formulation=form,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)
    return x0

def solve_system(network,form,tol,max_iter,h,x0,scale_var=None,scale_var_params=None,D_F=np.array([]),D_x=np.array([]),det_tol=1e-8,**kwargs):
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
    network.update(x0,formulation=form,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)
    x_sol_FD,iters_FD,err_vec_FD,p_sol_FD,q_sol_FD,q_inj_FD = network.solve_network(tol,max_iter,h=h,formulation=form,solver='NR_FD',scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x,**kwargs)

    network.update(x0,formulation=form,scale_var=scale_var,scale_var_params=scale_var_params,**kwargs)
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = network.solve_network(tol,max_iter,formulation=form,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x,det_tol=det_tol,**kwargs)

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

def print_full_solution(x_sol,network,form,p_scale=1,q_scale=1):
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
    p_vec,q_vec,q_inj = network.update_full(x_sol,formulation=form)

    node_order = []
    edge_order = []
    for n in network.get_nodes():
        node_order.append(n.number)
    for n in network.get_links():
        edge_order.append(n.number)

    np.set_printoptions(precision=4)
    print("\nFull solution using analytical method")
    print("Solution for pressure: p = {}".format(p_vec[node_order]/p_vec))
    print("Solution for injected flow: q_inj = {}".format(q_inj[node_order]/q_scale))
    print("Solution for edge flows: q = {}".format(q_vec[edge_order]/q_scale))

def example_gn4_linear_form_nodal():
    """Check the solution of the example network, using the nodal formulation.
    """
    # Given
    # link parameters
    alpha = 10.
    # create network
    gas_net = create_test_network_linear(alpha)

    # formulation
    form = 'nodal'

    # initalize network
    x0 = initialize_network_linear(gas_net,form)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0)
    x_sol_expected = np.array([14.25, 13.125])

    # Then
    np.testing.assert_array_almost_equal(x_sol,x_sol_expected)

def example_gn4_linear_form_nodal_DF():
    """Check the solution of the example network, using the nodal formulation, and using the scaled / normalized stopping criterion
    """
    # Given
    # link parameters
    alpha = 10.
    # create network
    gas_net = create_test_network_linear(alpha)

    # formulation
    form = 'nodal'

    # initalize network
    x0 = initialize_network_linear(gas_net,form)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    D_F = np.diag(1/100*np.ones(len(x0)))

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0,D_F=D_F)
    x_sol_expected = np.array([14.25, 13.125])

    # Then
    np.testing.assert_array_almost_equal(x_sol,x_sol_expected)

def example_gn4_linear_form_full():
    """Check the solution of the example network, using the full formulation.
    """
    # Given
    # link parameters
    alpha = 10.
    # create network
    gas_net = create_test_network_linear(alpha)

    # formulation
    form = 'full'

    # initalize network
    x0 = initialize_network_linear(gas_net,form)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0)
    p_sol = np.array([14.25, 13.125])
    q_sol = np.array([203.75, 157.5, 168.75, 46.25, 11.25])
    x_sol_expected = np.concatenate((q_sol,p_sol))

    # Then
    np.testing.assert_array_almost_equal(x_sol,x_sol_expected)

def example_gn4_linear_form_full_DF():
    """Check the solution of the example network, using the full formulation.
    """
    # Given
    # link parameters
    alpha = 10.
    # create network
    gas_net = create_test_network_linear(alpha)

    # formulation
    form = 'full'

    # initalize network
    x0 = initialize_network_linear(gas_net,form)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    D_F = np.diag(1/100*np.ones(len(x0))) # all nodal and link equations are of the same scale as the flow

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0,D_F=D_F)
    p_sol = np.array([14.25, 13.125])
    q_sol = np.array([203.75, 157.5, 168.75, 46.25, 11.25])
    x_sol_expected = np.concatenate((q_sol,p_sol))

    # Then
    np.testing.assert_array_almost_equal(x_sol,x_sol_expected)

def example_gn4_low_pres_form_nodal():
    """Check the solution of the example network, using the nodal formulation.
    """
    # Given
    # link parameters
    L_low_pres = [680., 500., 420., 600., 340.] #[m]
    D_low_pres = [.150, .100, .150, .100, .100] #[m]
    # carrier
    S = 0.589
    pn = 1.*bar#[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas_low_pres = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    # create network
    gas_net = create_test_network_low_pres(gas_low_pres,D_low_pres,L_low_pres)

    # formulation
    form = 'nodal'

    # initalize network
    x0 = initialize_network_low_pres(gas_net,form,gas_low_pres)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    det_tol = 1e-12

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0,det_tol=det_tol)
    x_sol_expected = np.array([25.76864547, 26.65754075])*mbar

    # Then
    # NB: Dit is niet echt de goede x_sol_expected. Dit is op basis van de andere versie van m'n code, maar die werk met de K onder vgl 4.22 uit Osiadacz, waar dus afgeronde getallen in staan. Vandaar de rtol = 1e-2
    assert np.allclose(x_sol,x_sol_expected,rtol=1e-2)

def example_gn4_low_pres_form_nodal_PU():
    """Check the solution of the example network, using the nodal formulation.
    """
    # Given
    # link parameters
    L_low_pres = [680., 500., 420., 600., 340.] #[m]
    D_low_pres = [.150, .100, .150, .100, .100] #[m]
    # carrier
    S = 0.589
    pn = 1.*bar#[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas_low_pres = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    scale_var = 'per_unit'
    qbase = 200.*gas_low_pres.rhon/hour
    pbase = 30.*mbar
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    # create network
    gas_net = create_test_network_low_pres(gas_low_pres,D_low_pres,L_low_pres)

    # formulation
    form = 'nodal'

    # initalize network
    x0 = initialize_network_low_pres(gas_net,form,gas_low_pres,scale_var=scale_var,scale_var_params=scale_var_params)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol_expected = np.array([25.76864547, 26.65754075])*mbar/pbase

    # Then
    # NB: Dit is niet echt de goede x_sol_expected. Dit is op basis van de andere versie van m'n code, maar die werk met de K onder vgl 4.22 uit Osiadacz, waar dus afgeronde getallen in staan. Vandaar de rtol = 1e-2
    assert np.allclose(x_sol,x_sol_expected,rtol=1e-2)

def example_gn4_low_pres_form_nodal_DF():
    """Check the solution of the example network, using the nodal formulation.
    """
    # Given
    # link parameters
    L_low_pres = [680., 500., 420., 600., 340.] #[m]
    D_low_pres = [.150, .100, .150, .100, .100] #[m]
    # carrier
    S = 0.589
    pn = 1.*bar#[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas_low_pres = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)

    # create network
    gas_net = create_test_network_low_pres(gas_low_pres,D_low_pres,L_low_pres)

    # formulation
    form = 'nodal'

    # initalize network
    x0 = initialize_network_low_pres(gas_net,form,gas_low_pres)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    det_tol = 1e-12
    D_F = np.diag(1/(200*gas_low_pres.rhon/hour*np.ones(len(x0)))) # take same value as the flow base value used in the per unit case.

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0,D_F=D_F,det_tol=det_tol)
    x_sol_expected = np.array([25.76864547, 26.65754075])*mbar

    # Then
    # NB: Dit is niet echt de goede x_sol_expected. Dit is op basis van de andere versie van m'n code, maar die werk met de K onder vgl 4.22 uit Osiadacz, waar dus afgeronde getallen in staan. Vandaar de rtol = 1e-2
    assert np.allclose(x_sol,x_sol_expected,rtol=1e-2)

def example_gn4_low_pres_form_nodal_scaled_solver():
    """Check the solution of the example network, using the nodal formulation.
    """
    # Given
    # link parameters
    L_low_pres = [680., 500., 420., 600., 340.] #[m]
    D_low_pres = [.150, .100, .150, .100, .100] #[m]
    # carrier
    S = 0.589
    pn = 1.*bar#[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas_low_pres = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)

    # create network
    gas_net = create_test_network_low_pres(gas_low_pres,D_low_pres,L_low_pres)

    # formulation
    form = 'nodal'

    # initalize network
    x0 = initialize_network_low_pres(gas_net,form,gas_low_pres)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    det_tol = 1e-12
    scale_var = 'matrix'
    scale_var_params = {'qbase':200*gas_low_pres.rhon/hour,'pgbase':30.*mbar}

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0,scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol)
    x_sol_expected = np.array([25.76864547, 26.65754075])*mbar

    # Then
    # NB: Dit is niet echt de goede x_sol_expected. Dit is op basis van de andere versie van m'n code, maar die werk met de K onder vgl 4.22 uit Osiadacz, waar dus afgeronde getallen in staan. Vandaar de rtol = 1e-2
    assert np.allclose(x_sol,x_sol_expected,rtol=1e-2)

def example_gn4_low_pres_form_full():
    """Check the solution of the example network, using the full formulation.
    """
    # Given
    # link parameters
    L_low_pres = [680., 500., 420., 600., 340.] #[m]
    D_low_pres = [.150, .100, .150, .100, .100] #[m]
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas_low_pres = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    # create network
    gas_net = create_test_network_low_pres(gas_low_pres,D_low_pres,L_low_pres)

    # formulation
    form = 'full'

    # initalize network
    x0 = initialize_network_low_pres(gas_net,form,gas_low_pres)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    det_tol = 1e-12

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0,det_tol=det_tol)
    p_sol_expected = np.array([25.76864547, 26.65754075])*mbar
    q_sol_expected = np.array([218.73427878, 85.99809825, 228.67629991, 31.26571831, -48.6762995 ])*gas_low_pres.rhon/hour #based on gas flows found using nodal formulation
    x_sol_expected = np.concatenate((q_sol_expected,p_sol_expected))

    # Then
    # NB: Dit is niet echt de goede p_sol_expected. Dit is op basis van de andere versie van m'n code, maar die werk met de K onder vgl 4.22 uit Osiadacz, waar dus afgeronde getallen in staan. Vandaar de rtol = 1e-2
    assert np.allclose(x_sol,x_sol_expected,rtol=1e-2)

def example_gn4_low_pres_form_full_PU():
    """Check the solution of the example network, using the full formulation, and per unit scaling
    """
    # Given
    # link parameters
    L_low_pres = [680., 500., 420., 600., 340.] #[m]
    D_low_pres = [.150, .100, .150, .100, .100] #[m]
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas_low_pres = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    scale_var = 'per_unit'
    qbase = 200.*gas_low_pres.rhon/hour
    pbase = 30.*mbar
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    # create network
    gas_net = create_test_network_low_pres(gas_low_pres,D_low_pres,L_low_pres)

    # formulation
    form = 'full'

    # initalize network
    x0 = initialize_network_low_pres(gas_net,form,gas_low_pres,scale_var=scale_var,scale_var_params=scale_var_params)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0,scale_var=scale_var,scale_var_params=scale_var_params)
    p_sol_expected = np.array([25.76864547, 26.65754075])*mbar/pbase
    q_sol_expected = np.array([218.73427878, 85.99809825, 228.67629991, 31.26571831, -48.6762995 ])*gas_low_pres.rhon/hour/qbase #based on gas flows found using nodal formulation
    x_sol_expected = np.concatenate((q_sol_expected,p_sol_expected))

    # Then
    # NB: Dit is niet echt de goede p_sol_expected. Dit is op basis van de andere versie van m'n code, maar die werk met de K onder vgl 4.22 uit Osiadacz, waar dus afgeronde getallen in staan. Vandaar de rtol = 1e-2
    assert np.allclose(x_sol,x_sol_expected,rtol=1e-2)

def example_gn4_low_pres_form_full_DF():
    """Check the solution of the example network, using the full formulation.
    """
    # Given
    # link parameters
    L_low_pres = [680., 500., 420., 600., 340.] #[m]
    D_low_pres = [.150, .100, .150, .100, .100] #[m]
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas_low_pres = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    # create network
    gas_net = create_test_network_low_pres(gas_low_pres,D_low_pres,L_low_pres)

    # formulation
    form = 'full'

    # initalize network
    x0 = initialize_network_low_pres(gas_net,form,gas_low_pres)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    det_tol = 1e-12
    D_F = np.diag(1/(200*gas_low_pres.rhon/hour*np.ones(len(x0)))) # take same value as the flow base value used in the per unit case. All nodal and link equations are of the same scale as the flow.

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0,D_F=D_F,det_tol=det_tol)
    p_sol_expected = np.array([25.76864547, 26.65754075])*mbar
    q_sol_expected = np.array([218.73427878, 85.99809825, 228.67629991, 31.26571831, -48.6762995 ])*gas_low_pres.rhon/hour #based on gas flows found using nodal formulation
    x_sol_expected = np.concatenate((q_sol_expected,p_sol_expected))

    # Then
    # NB: Dit is niet echt de goede p_sol_expected. Dit is op basis van de andere versie van m'n code, maar die werk met de K onder vgl 4.22 uit Osiadacz, waar dus afgeronde getallen in staan. Vandaar de rtol = 1e-2
    assert np.allclose(x_sol,x_sol_expected,rtol=1e-2)

def example_gn4_low_pres_form_full_scaled_solver():
    """Check the solution of the example network, using the full formulation.
    """
    # Given
    # link parameters
    L_low_pres = [680., 500., 420., 600., 340.] #[m]
    D_low_pres = [.150, .100, .150, .100, .100] #[m]
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas_low_pres = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    # create network
    gas_net = create_test_network_low_pres(gas_low_pres,D_low_pres,L_low_pres)

    # formulation
    form = 'full'

    # initalize network
    x0 = initialize_network_low_pres(gas_net,form,gas_low_pres)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    det_tol = 1e-12
    scale_var = 'matrix'
    scale_var_params = {'qbase':200*gas_low_pres.rhon/hour,'pgbase':30.*mbar} # take same value as the flow base value used in the per unit case. All nodal and link equations are of the same scale as the flow.

    # When
    x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec = solve_system(gas_net,form,tol,max_iter,h,x0,scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol)
    p_sol_expected = np.array([25.76864547, 26.65754075])*mbar
    q_sol_expected = np.array([218.73427878, 85.99809825, 228.67629991, 31.26571831, -48.6762995 ])*gas_low_pres.rhon/hour #based on gas flows found using nodal formulation
    x_sol_expected = np.concatenate((q_sol_expected,p_sol_expected))

    # Then
    # NB: Dit is niet echt de goede p_sol_expected. Dit is op basis van de andere versie van m'n code, maar die werk met de K onder vgl 4.22 uit Osiadacz, waar dus afgeronde getallen in staan. Vandaar de rtol = 1e-2
    assert np.allclose(x_sol,x_sol_expected,rtol=1e-2)

def load_flow_Osiadacz():
    """Replicate the example from Osiadacz"""
    tol = 1e-6
    max_iters = 10
    print('\nExample Osiadacz:')
    gas_net_Osiadacz = create_test_network_Osiadacz_Lacey()
    gas = gas_net_Osiadacz.links[0].link_params.get('carrier')
    x0_nodal = initialize_network_Osiadacz_Lacey(gas_net_Osiadacz,'nodal')

    print('Nodal formulation, unscaled (in S.I.)')
    print('  Initial guess:')
    p_vec0_nodal, q_vec0_nodal,q_inj0_nodal  = gas_net_Osiadacz.update_full(x0_nodal,formulation='nodal')
    print('    Pressure p = {} mbar'.format(p_vec0_nodal/mbar))
    print('    Edge flows q = {} m3/hour'.format(q_vec0_nodal/(gas.rhon/hour)))
    # do one iteration, in order to compare with Osiadacz
    x1_nodal, iters, err_vec1_nodal, p_vec1_nodal, q_vec1_nodal, q_inj1_nodal = gas_net_Osiadacz.solve_network(tol,1,formulation='nodal',solver='NR')
    x1_expected = np.array([25.3393,26.2370,26.6477])*mbar
    print('  After 1 iteration')
    print('    Absolute error x with Osiadacz: {} mbar'.format(np.abs(x1_nodal/mbar-x1_expected/mbar)))
    print('    Relative error x with Osiadacz: {}'.format(np.abs(x1_nodal-x1_expected)/np.abs(x1_expected)))
    print('    Pressure p = {} mbar'.format(p_vec1_nodal/mbar))
    print('    Edge flows q = {} m3/hour'.format(q_vec1_nodal/(gas.rhon/hour)))
    print('    Injected flows q = {} m3/hour'.format(q_inj1_nodal/(gas.rhon/hour)))
    # solve using nodal formulation
    gas_net_Osiadacz.reset_network(x0_nodal,formulation='nodal')
    x_nodal, iters_nodal, err_vec_nodal, p_vec_nodal, q_vec_nodal, q_inj_nodal = gas_net_Osiadacz.solve_network(tol,max_iters,formulation='nodal',solver='NR')
    print('  Final solution, after {} iterations with final ||F|| = {:.4e}'.format(iters_nodal,err_vec_nodal[-1]))
    print('    Pressure p = {} mbar'.format(p_vec_nodal/mbar))
    print('    Edge flows q = {} m3/hour'.format(q_vec_nodal/(gas.rhon/hour)))
    print('    Injected flows q = {} m3/hour'.format(q_inj_nodal/(gas.rhon/hour)))

    scale_var_params = {'pbase':mbar,'qbase':gas.rhon/hour}
    gas_net_Osiadacz.reset_network(x0_nodal,formulation='nodal')
    print('Nodal formulation, scaled (p.u.)')
    print('  Initial guess:')
    p_vec0_nodal, q_vec0_nodal,q_inj0_nodal  = gas_net_Osiadacz.update_full(x0_nodal,formulation='nodal')
    print('    Pressure p = {} mbar'.format(p_vec0_nodal/mbar))
    print('    Edge flows q = {} m3/hour'.format(q_vec0_nodal/(gas.rhon/hour)))
    # do one iteration, in order to compare with Osiadacz
    x1_nodal, iters, err_vec1_nodal, p_vec1_nodal, q_vec1_nodal, q_inj1_nodal = gas_net_Osiadacz.solve_network(tol,1,formulation='nodal',solver='NR',scale_var='per_unit',scale_var_params=scale_var_params)
    x1_expected = np.array([25.3393,26.2370,26.6477])*mbar
    print('  After 1 iteration')
    print('    Absolute error x with Osiadacz: {} mbar'.format(np.abs(x1_nodal-x1_expected/mbar)))
    print('    Relative error x with Osiadacz: {}'.format(np.abs(x1_nodal-x1_expected/mbar)/np.abs(x1_expected/mbar)))
    print('    Pressure p = {} mbar'.format(p_vec1_nodal/mbar))
    print('    Edge flows q = {} m3/hour'.format(q_vec1_nodal/(gas.rhon/hour)))
    print('    Injected flows q = {} m3/hour'.format(q_inj1_nodal/(gas.rhon/hour)))
    # solve using nodal formulation
    gas_net_Osiadacz.reset_network(x0_nodal,formulation='nodal')
    x_nodal, iters_nodal, err_vec_nodal, p_vec_nodal, q_vec_nodal, q_inj_nodal = gas_net_Osiadacz.solve_network(tol,max_iters,formulation='nodal',solver='NR',scale_var='per_unit',scale_var_params=scale_var_params)
    print('  Final solution, after {} iterations with final ||F|| = {:.4e}'.format(iters_nodal,err_vec_nodal[-1]))
    print('    Pressure p = {} mbar'.format(p_vec_nodal/mbar))
    print('    Edge flows q = {} m3/hour'.format(q_vec_nodal/(gas.rhon/hour)))
    print('    Injected flows q = {} m3/hour'.format(q_inj_nodal/(gas.rhon/hour)))

    print('Full formulation, unscaled (in S.I.)')
    gas_net_Osiadacz.reset_network(x0_nodal,formulation='nodal')
    x0_full = initialize_network_Osiadacz_Lacey(gas_net_Osiadacz,'full')
    print('  Initial guess:')
    p_vec0_full, q_vec0_full,q_inj0_full  = gas_net_Osiadacz.update_full(x0_full,formulation='full')
    print('    Pressure p = {} mbar'.format(p_vec0_full/mbar))
    print('    Edge flows q = {} m3/hour'.format(q_vec0_full/(gas.rhon/hour)))
    # do one iteration, in order to compare with Osiadacz
    x1_full, iters, err_vec1_full, p_vec1_full, q_vec1_full, q_inj1_full = gas_net_Osiadacz.solve_network(tol,1,formulation='full',solver='NR')
    Ne = len(gas_net_Osiadacz.links)
    x1_expected = np.array([25.3393,26.2370,26.6477])*mbar
    print('  After 1 iteration')
    print('    Absolute error x with Osiadacz: {} mbar'.format(np.abs(x1_full[Ne:]/mbar-x1_expected/mbar)))
    print('    Relative error x with Osiadacz: {}'.format(np.abs(x1_full[Ne:]-x1_expected)/np.abs(x1_expected)))
    print('    Pressure p = {} mbar'.format(p_vec1_full/mbar))
    print('    Edge flows q = {} m3/hour'.format(q_vec1_full/(gas.rhon/hour)))
    print('    Injected flows q = {} m3/hour'.format(q_inj1_full/(gas.rhon/hour)))
    # solve using full formulation
    gas_net_Osiadacz.reset_network(x0_full,formulation='full')
    x_full, iters_full, err_vec_full, p_vec_full, q_vec_full, q_inj_full = gas_net_Osiadacz.solve_network(tol,max_iters,formulation='full',solver='NR')
    print('  Final solution, after {} iterations with final ||F|| = {:.4e}'.format(iters_full,err_vec_full[-1]))
    print('    Pressure p = {} mbar'.format(p_vec_full/mbar))
    print('    Edge flows q = {} m3/hour'.format(q_vec_full/(gas.rhon/hour)))
    print('    Injected flows q = {} m3/hour'.format(q_inj_full/(gas.rhon/hour)))

    fig_top = plt.figure('Topology Osiadacz')
    ax_top = fig_top.gca()
    gas_net_Osiadacz.draw_network(ax_top,halflink_angle=2,halflink_length=.2)
    plt.axis('equal')
    plt.axis('off')

if __name__ == '__main__':
    # link parameters
    alpha = 10.
    L_low_pres = [680., 500., 420., 600., 340.] #[m]
    D_low_pres = [.150, .100, .150, .100, .100] #[m]
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    gas_low_pres = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)

    # create networks
    gas_net_lin = create_test_network_linear(alpha)
    gas_net_low_pres = create_test_network_low_pres(gas_low_pres,D_low_pres,L_low_pres)

    # formulation
    form = 'nodal'

    # initalize network
    x0_lin = initialize_network_linear(gas_net_lin,form)
    x0_low_pres = initialize_network_low_pres(gas_net_low_pres,form,gas_low_pres)

    # solve the network
    h = 1e-6
    tol = 1e-6
    max_iter = 500
    det_tol = 1e-12
    x_sol_FD_lin,iters_FD_lin,err_vec_FD_lin,x_sol_lin,iters_lin,err_vec_lin = solve_system(gas_net_lin,form,tol,max_iter,h,x0_lin)
    x_sol_FD_low_pres,iters_FD_low_pres,err_vec_FD_low_pres,x_sol_low_pres,iters_low_pres,err_vec_low_pres = solve_system(gas_net_low_pres,form,tol,max_iter,h,x0_low_pres,det_tol=det_tol)

    # print all results
    print('Linear pipe lines, nodal formulation, no scaling:')
    print_results(x_sol_FD_lin,iters_FD_lin,err_vec_FD_lin,x_sol_lin,iters_lin,err_vec_lin)
    print_full_solution(x_sol_lin,gas_net_lin,form)
    print('Low pressure pipe lines, nodal formulation, no scaling:')
    print_results(x_sol_FD_low_pres,iters_FD_low_pres,err_vec_FD_low_pres,x_sol_low_pres,iters_low_pres,err_vec_low_pres)
    print_full_solution(x_sol_low_pres,gas_net_low_pres,form)

    # compare convergence of the low pressure pipes
    scale_var = 'per_unit'
    qbase = 200.*gas_low_pres.rhon/hour
    pbase = 30.*mbar
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    # nodal PU
    gas_net_low_pres.update_full(x0_low_pres,formulation='nodal')
    x0_low_pres_nodal_PU = initialize_network_low_pres(gas_net_low_pres,'nodal',gas_low_pres,scale_var=scale_var,scale_var_params=scale_var_params)
    _,_,_,_,iters_low_pres_nodal_PU,err_vec_low_pres_nodal_PU = solve_system(gas_net_low_pres,'nodal',tol,max_iter,h,x0_low_pres_nodal_PU,scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol)
    # nodal DF
    gas_net_low_pres.update_full(x0_low_pres,formulation='nodal')
    x0_low_pres_nodal_DF = initialize_network_low_pres(gas_net_low_pres,'nodal',gas_low_pres)
    D_F_nodal = np.diag(1/qbase*np.ones(len(x0_low_pres_nodal_DF))) # take same value as the flow base value used in the per unit case.
    _,_,_,_,iters_low_pres_nodal_DF,err_vec_low_pres_nodal_DF = solve_system(gas_net_low_pres,'nodal',tol,max_iter,h,x0_low_pres_nodal_DF,D_F=D_F_nodal,det_tol=det_tol)
    # nodal scaled solver (Dx en DF)
    gas_net_low_pres.update_full(x0_low_pres,formulation='nodal')
    x0_low_pres_nodal_scaled = initialize_network_low_pres(gas_net_low_pres,'nodal',gas_low_pres)
    _,_,_,_,iters_low_pres_nodal_scaled,err_vec_low_pres_nodal_scaled = solve_system(gas_net_low_pres,'nodal',tol,max_iter,h,x0_low_pres_nodal_scaled,scale_var='matrix',scale_var_params=scale_var_params,det_tol=det_tol)
    # full
    gas_net_low_pres.update_full(x0_low_pres,formulation='nodal')
    x0_low_pres_full = initialize_network_low_pres(gas_net_low_pres,'full',gas_low_pres)
    _,_,_,_,iters_low_pres_full,err_vec_low_pres_full = solve_system(gas_net_low_pres,'full',tol,max_iter,h,x0_low_pres_full,det_tol=det_tol)
    # full PU
    gas_net_low_pres.update_full(x0_low_pres,formulation='nodal')
    x0_low_pres_full_PU = initialize_network_low_pres(gas_net_low_pres,'full',gas_low_pres,scale_var=scale_var,scale_var_params=scale_var_params)
    _,_,_,_,iters_low_pres_full_PU,err_vec_low_pres_full_PU = solve_system(gas_net_low_pres,'full',tol,max_iter,h,x0_low_pres_full_PU,scale_var=scale_var,scale_var_params=scale_var_params,det_tol=det_tol)
    # full DF
    gas_net_low_pres.update_full(x0_low_pres,formulation='nodal')
    x0_low_pres_full_DF = initialize_network_low_pres(gas_net_low_pres,'full',gas_low_pres)
    D_F_full = np.diag(1/qbase*np.ones(len(x0_low_pres_full_DF))) # take same value as the flow base value used in the per unit case. All nodal and link equations are of the same scale as the flow.
    _,_,_,_,iters_low_pres_full_DF,err_vec_low_pres_full_DF = solve_system(gas_net_low_pres,'full',tol,max_iter,h,x0_low_pres_full_DF,D_F=D_F_full,det_tol=det_tol)
    # full scaled solver (Dx en DF)
    gas_net_low_pres.update_full(x0_low_pres,formulation='nodal')
    x0_low_pres_full_scaled = initialize_network_low_pres(gas_net_low_pres,'full',gas_low_pres)
    _,_,_,_,iters_low_pres_full_scaled,err_vec_low_pres_full_scaled = solve_system(gas_net_low_pres,'full',tol,max_iter,h,x0_low_pres_full_scaled,scale_var='matrix',scale_var_params=scale_var_params,det_tol=det_tol)

    fig = plt.figure('Convergence plot')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    max_iter_used = np.max([iters_low_pres,iters_low_pres_nodal_PU,iters_low_pres_nodal_DF,iters_low_pres_nodal_scaled,iters_low_pres_full,iters_low_pres_full_PU,iters_low_pres_full_DF,iters_low_pres_full_scaled])
    ax.semilogy([0,max_iter_used+1],[tol,tol],':r',label='tolerance')
    ax.semilogy(np.asarray(range(0,iters_low_pres+1)),err_vec_low_pres,'-s',label='nodal')
    ax.semilogy(np.asarray(range(0,iters_low_pres_nodal_PU+1)),err_vec_low_pres_nodal_PU,'-d',label='nodal p.u.')
    ax.semilogy(np.asarray(range(0,iters_low_pres_nodal_DF+1)),err_vec_low_pres_nodal_DF,'-*',label='nodal $D_F$')
    ax.semilogy(np.asarray(range(0,iters_low_pres_nodal_scaled+1)),err_vec_low_pres_nodal_scaled,'-o',label='nodal scaled solver')
    ax.semilogy(np.asarray(range(0,iters_low_pres_full+1)),err_vec_low_pres_full,'--s',label='full')
    ax.semilogy(np.asarray(range(0,iters_low_pres_full_PU+1)),err_vec_low_pres_full_PU,'--d',label='full p.u.')
    ax.semilogy(np.asarray(range(0,iters_low_pres_full_DF+1)),err_vec_low_pres_full_DF,'--*',label='full $D_F$')
    ax.semilogy(np.asarray(range(0,iters_low_pres_full_scaled+1)),err_vec_low_pres_full_scaled,'--o',label='full scaled solver')
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)

    load_flow_Osiadacz()

    plt.show()
