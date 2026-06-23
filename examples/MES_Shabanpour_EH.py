"""Example of a MES with 3 nodes per carrier. Based on the example in Shabanpourt-Haghighi and Seifi. Uses energy hubs for the coupling nodes."""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water, Gas
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.utils.constants import mm, bar, hour, kV, MW, MSCM, BTU, MBTU
import numpy as np
import scipy.sparse as sps
import pytest
import matplotlib.pyplot as plt

# base values to compensate for p.u. in electrical network in Shabanpour
Pb = 1 # base I assume for complex power S (which is 1, since I work in S.I.)
Sbase_shabanpour = 1*MW #[W]
Vbase_shabanpour = 10/np.sqrt(3)*kV #[V]
Ybase_shabanpour = Sbase_shabanpour/(Vbase_shabanpour**2)

# water carrier
rho_w = 960. #[kg/m^3]
mu_w = 0.294e-6 #[m^2/s]
Cp_w = 4.182e3 #[J/(kg K)]
grav_const = 9.81 #[m/s^2]
water = Water('water',Cp_w,rho=rho_w,mu=mu_w)

# gas carrier (part of this isn't used / isn't required for the pipe equation. But the carrier needs it)
mu_g = 0.288e-6 #[m^2/s]
S = 0.6106
Z = 0.8
pn = 1.01325*bar #[Pa]
Tn = 273.15 #[K]
T = 281.15 #[K]
R = 8.314413 #[J/molK]
M = 28.97e-3 #[kg/mol]
R_air = R/M #[J/kgK]
gas = Gas('high pres gas',S,R_air,Z,pn,Tn,T,mu=mu_g)
rhon_g = gas.rhon

GHV_BTU = 40.611 #[MBTU/m^3]
GHV = GHV_BTU * MBTU*BTU * hour/rhon_g #[J/kg]

def create_network(hydr_eq='fa',heat_load='outflow',node_set=1):
    """Create a test heterogeneous network with an energy hub

    Returns
    -------
    het_net : HeterogeneousNetwork
        The test network
    gas_net : GasNetwork
        The test gas subnetwork
    elec_net : ElectricalNetwork
        The test electrical subnetwork
    heat_net : HeatNetwork
        The test heat subnetwork
    water : Carrier
        The carrier of water in the network
    gas : Carrier
        The carrier of gas in the network
    """


    # nodes
    g0 = GasNode('gn0',node_type=0,x=5,y=12,p=50.*bar) # reference node
    g1 = GasNode('gn1',node_type=1,x=0,y=7,q=10.8649*MSCM*rhon_g/hour) # load node
    g2 = GasNode('gn2',node_type=1,x=5,y=2,q=20*MSCM*rhon_g/hour) # load node
    g3 = GasNode('gn3',node_type=1,x=2.5,y=4.5,q=0.) # load node
    if node_set == 1 or node_set == 2:
        e0 = ElectricalNode('en0',node_type=5,x=7,y=12,P=0.1451*Sbase_shabanpour,Q=0.*Sbase_shabanpour,V=1.06*Vbase_shabanpour,delta=0.) # PQVdelta node
        e1 = ElectricalNode('en1',node_type=2,x=2,y=7,P=30.*Sbase_shabanpour,Q=15.*Sbase_shabanpour) # load
        e2 = ElectricalNode('en2',node_type=6,x=7,y=2,P=30.136*Sbase_shabanpour,Q=15.*Sbase_shabanpour,V=1.*Vbase_shabanpour) # PQV
    elif node_set == 3:
        e0 = ElectricalNode('en0',node_type=6,x=7,y=12,P=0.1451*Sbase_shabanpour,Q=0.*Sbase_shabanpour,V=1.06*Vbase_shabanpour) # PQV node
        e1 = ElectricalNode('en1',node_type=2,x=2,y=7,P=30.*Sbase_shabanpour,Q=15.*Sbase_shabanpour) # load
        e2 = ElectricalNode('en2',node_type=5,x=7,y=2,P=30.136*Sbase_shabanpour,Q=15.*Sbase_shabanpour,V=1.*Vbase_shabanpour,delta=-0.10555751316061704) # PQVdelta
    else:
         raise ValueError("Enter valid value for 'node_set'.")
    h0 = HeatNode('hn0',node_type=5,x=9,y=12,p=5517.*rho_w*grav_const) # reference node
    if node_set == 1:
        if heat_load == 'outflow':
            h1 = HeatNode('hn1',node_type=1,x=4,y=7,Tr_hl=50.,dphi=35.*MW) # load node (sink)
            h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
            h2 = HeatNode('hn2',node_type=3,x=9,y=2,Tr_hl=50.,dphi=20.*MW,p=4324.9668*rho_w*grav_const) # load (sink) reference node
            h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
        elif heat_load == 'delta':
            h1 = HeatNode('hn1',node_type=12,x=4,y=7,dT=69.0382,dphi=35.*MW) # load node (sink)
            h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
            h2 = HeatNode('hn2',node_type=13,x=9,y=2,dT=73.5435,dphi=20.*MW,p=4324.9668*rho_w*grav_const) # load (sink) reference node
            h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
        else:
            raise ValueError("Enter valid value for 'heat_load'. Either 'outflow' or 'delta'.")
    elif node_set == 2 or node_set == 3:
        if heat_load == 'outflow':
            h1 = HeatNode('hn1',node_type=1,x=4,y=7,Tr_hl=50.,dphi=35.*MW) # load node (sink)
            h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
            h2 = HeatNode('hn2',node_type=1,x=9,y=2,Tr_hl=50.,dphi=20.*MW) # load node (sink)
            h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
        elif heat_load == 'delta':
            h1 = HeatNode('hn1',node_type=12,x=4,y=7,dT=69.0382,dphi=35.*MW) # load node (sink)
            h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
            h2 = HeatNode('hn2',node_type=12,x=9,y=2,dT=73.5435,dphi=20.*MW) # load node (sink)
            h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
        else:
            raise ValueError("Enter valid value for 'heat_load'. Either 'outflow' or 'delta'.")

    eta_GB = 0.88
    eta_CHP = 0.88
    To_c_EH0 = 120.
    To_c_EH1 = 126.49315908196526
    P3 = 50.498203335360927 #[MW]
    q3 = 9.3382*MSCM*rhon_g/hour #kg/s
    eta_GG = P3*MW/(GHV*q3)
    P4 = 10.533117232243029 #[MW]
    phi3 = 28.661590209014538  #[MW]
    phi4 = 29.0151926135801583 #[MW]
    nu0 = eta_GB*P3/(eta_GG*phi3+eta_GB*P3)
    nu1 = P4/(P4+phi4)

    C0 = np.array([[eta_GG*nu0/Pb],[eta_GB*(1-nu0)]])
    C1 = np.array([[eta_CHP*nu1/Pb],[eta_CHP*(1-nu1)]])
    Ein_carriers = ['g']
    Eout_carriers = ['e','h']
    Tr0 = 48.68310632
    Tr1 = 49.53799927
    if node_set == 1:
        if heat_load == 'outflow':
            c0 = HeterogeneousNode('cn0',node_type=1,x=7,y=14,unit_type='EH',unit_params={'C':C0,'GHV':GHV,'Eout_carriers':Eout_carriers,'Ein_carriers':Ein_carriers})# To known
        elif heat_load == 'delta':
            c0 = HeterogeneousNode('cn0',node_type=1,x=7,y=14,unit_type='EH',unit_params={'C':C0,'GHV':GHV,'Eout_carriers':Eout_carriers,'Ein_carriers':Ein_carriers})# To known
        c1 = HeterogeneousNode('cn1',node_type=0,x=7,y=0,unit_type='EH',unit_params={'C':C1,'GHV':GHV,'Eout_carriers':Eout_carriers,'Ein_carriers':Ein_carriers}) # To unknown
    elif node_set == 2 or node_set == 3:
        if heat_load == 'outflow':
            c0 = HeterogeneousNode('cn0',node_type=1,x=7,y=14,unit_type='EH',unit_params={'C':C0,'GHV':GHV,'Eout_carriers':Eout_carriers,'Ein_carriers':Ein_carriers})# To known
            c1 = HeterogeneousNode('cn1',node_type=1,x=7,y=0,unit_type='EH',unit_params={'C':C1,'GHV':GHV,'Eout_carriers':Eout_carriers,'Ein_carriers':Ein_carriers}) # To known
        elif heat_load == 'delta':
            c0 = HeterogeneousNode('cn0',node_type=1,x=7,y=14,unit_type='EH',unit_params={'C':C0,'GHV':GHV,'Eout_carriers':Eout_carriers,'Ein_carriers':Ein_carriers})# To known
            c1 = HeterogeneousNode('cn1',node_type=1,x=7,y=0,unit_type='EH',unit_params={'C':C1,'GHV':GHV,'Eout_carriers':Eout_carriers,'Ein_carriers':Ein_carriers}) # To known

    # links
    L_g = 30000. #[m]
    D_g = 150.*mm #[m]
    eps_g = 0.05*mm #[m]
    compr_ratio = 1.3
    gas_link_params = {'D':D_g,'L':L_g,'eps':eps_g,'carrier':gas}
    if hydr_eq =='fa':
        link_eq = 'q_of_dp'
    elif hydr_eq == 'fb':
        link_eq = 'dp_of_q'
    else:
        raise ValueError("Enter valid value for hydr_eq. Either 'fa' or 'fb'.")
    gl0 = GasLink('gl0',g0,g1,link_type = 'pipe_high_pres_colebrook',link_params = gas_link_params,link_eq_form=link_eq)
    gl1 = GasLink('gl1',g0,g2,link_type = 'pipe_high_pres_colebrook',link_params = gas_link_params,link_eq_form=link_eq)
    gl2 = GasLink('gl2',g3,g2,link_type = 'pipe_high_pres_colebrook',link_params = gas_link_params,link_eq_form=link_eq)
    gl3 = GasLink('gl3',g1,g3,link_type = 'compressor',link_params = {'r':compr_ratio},link_eq_form=link_eq)
    gl4 = GasLink('gl4',g0,c0)
    gl5 = GasLink('gl5',g2,c1)
    x = 0.5/100 # factor 100 because Shabanpour give a p.u. value, using Sbase = 100 MW, whereas I assume Sbase = 1 MW
    r = 0.05/100
    z = r+x*1j
    b = -x/(np.abs(z)**2)
    g = r/(np.abs(z)**2)
    b *= Ybase_shabanpour
    g *= Ybase_shabanpour
    elec_link_params = {'b':b,'g':g}
    el0 = ElectricalLink('el0',e0,e1,link_type='short_line',link_params=elec_link_params)
    el1 = ElectricalLink('el1',e0,e2,link_type='short_line',link_params=elec_link_params)
    el2 = ElectricalLink('el2',e1,e2,link_type='short_line',link_params=elec_link_params)
    el3 = ElectricalLink('el3',c0,e0)
    el4 = ElectricalLink('el4',c1,e2)
    L_h = L_g #[m]
    D_h = D_g #[m]
    lam = 0.2 #[W/(mK)]
    eps_h = 1.25*mm #[m]
    U = lam/(np.pi*D_h) #[W/(m^2 K)]
    heat_link_params = {'D':D_h,'L':L_h,'eps':eps_h,'U':U,'carrier':water}
    hl0 = HeatLink('hl0',h0,h1,link_type='standard_pipe_low_pres_colebrook',link_params=heat_link_params)
    hl1 = HeatLink('hl1',h0,h2,link_type='standard_pipe_low_pres_colebrook',link_params=heat_link_params.copy())
    hl2 = HeatLink('hl2',h1,h2,link_type='standard_pipe_low_pres_colebrook',link_params=heat_link_params.copy())
    if node_set == 1:
        if heat_load == 'outflow':
            hl3 = HeatLink('hl3',c0,h0,link_params={'carrier':water},bc_type=6,Tsstart=To_c_EH0) # To of coupling (source) is unknown
        elif heat_load == 'delta':
            hl3 = HeatLink('hl3',c0,h0,link_params={'carrier':water},bc_type=10,dTstart=To_c_EH0-Tr0) # dT of coupling (source) is known
        hl4 = HeatLink('hl4',c1,h2,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
    elif node_set == 2 or node_set == 3:
        if heat_load == 'outflow':
            hl3 = HeatLink('hl3',c0,h0,link_params={'carrier':water},bc_type=6,Tsstart=To_c_EH0) # To of coupling (source) is known
            hl4 = HeatLink('hl4',c1,h2,link_params={'carrier':water},bc_type=6,Tsstart=To_c_EH1) # To of coupling (source) is known
        elif heat_load == 'delta':
            hl3 = HeatLink('hl3',c0,h0,link_params={'carrier':water},bc_type=10,dTstart=To_c_EH0-Tr0) # dT of coupling (source) is known
            hl4 = HeatLink('hl4',c1,h2,link_params={'carrier':water},bc_type=10,dTstart=To_c_EH1-Tr1) # dT of coupling (source) is known

    # network
    gas_net = GasNetwork('test gas network')
    gas_net.add_link(gl0)
    gas_net.add_link(gl1)
    gas_net.add_link(gl2)
    gas_net.add_link(gl3)
    gas_net.add_link(gl4)
    gas_net.add_link(gl5)

    elec_net = ElectricalNetwork('test electrical network')
    elec_net.add_link(el0)
    elec_net.add_link(el1)
    elec_net.add_link(el2)
    elec_net.add_link(el3)
    elec_net.add_link(el4)

    Ta = 10. #[C]
    heat_net = HeatNetwork('test heat network',Ta=Ta)
    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    heat_net.add_link(hl3)
    heat_net.add_link(hl4)

    het_net = HeterogeneousNetwork('test heterogeneous network')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    het_net.add_node(c0)
    het_net.add_node(c1)

    return het_net, gas_net, elec_net, heat_net, water, gas

def initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,heat_load='outflow',node_set=1,scale_var=None,scale_var_params=None,qc_init = np.array([10,3])*MSCM*rhon_g/hour,Pc_init = np.array([50.,10.])*Sbase_shabanpour,phic_init = np.array([30.,25.])*MW):
    """Sets values of network variables to be used for initial guess.

    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The test network
    gas_net : GasNetwork
        The test gas subnetwork
    elec_net : ElectricalNetwork
        The test electrical subnetwork
    heat_net : HeatNetwork
        The test heat subnetwork
    water : Carrier
        The carrier of water in the network
    gas : Carrier
        The carrier of gas in the network

    Returns
    -------
    x0 : np array
        initial guess
    """
    rho_w = water.rhon #[kg/m^3]
    grav_const = water.g #[m/s^2]
    rhon_g = gas.rhon #[kg/m^3]
    Ta = heat_net.Ta
    gp_init = np.array([45.,47.,45.])*bar
    q_init = np.array([20.,20.,20.,20.])*MSCM*rhon_g/hour
    delta_init = np.array([0.,0.]) # default values
    V_init = np.array([1.])*Vbase_shabanpour # default values
    m_init = np.array([65.,30.,-60.])
    if formulation['heat'] == 'half_link_flow':
        m_hl_init = np.array([20.,20.])
        if heat_load == 'delta':
                To_hl_init = np.array([50.,50.]) #[C]
    if node_set == 1:
        hp_init = np.array([254.3706])*rho_w*grav_const
    elif node_set == 2 or node_set == 3:
        hp_init = np.array([254.3706,4300])*rho_w*grav_const
    Ts_init = np.array([100.,120.,120.])
    Tr_init = np.array([50.,50.,50.])
    qc_init = np.array([10,3])*MSCM*rhon_g/hour
    Pc_init = np.array([50.,10.])*Sbase_shabanpour
    Qc_init = np.array([0.,0.])*Sbase_shabanpour
    Sc_init = np.concatenate((Pc_init,Qc_init))
    mc_init = np.array([10.,10.])
    phic_init = np.array([30.,25.])*MW
    if node_set == 1:
        Toc_init = np.array([120.])
    elif node_set == 2 or node_set == 3:
        Toc_init = np.array([])

    xg_init = np.concatenate((q_init,gp_init))
    xe_init = np.concatenate((delta_init,V_init))
    if formulation['heat'] == 'half_link_flow':
        if heat_load == 'delta':
            xh_init = np.concatenate((m_init,m_hl_init,hp_init,Ts_init,Tr_init,To_hl_init))
        else:
            xh_init = np.concatenate((m_init,m_hl_init,hp_init,Ts_init,Tr_init))
    else:
        xh_init = np.concatenate((m_init,hp_init,Ts_init,Tr_init))
    if het_net.get_nodes(node_types=[2],unit_types=['gh_gas_boiler','eh_elec_boiler','geh_CHP','EH']): # dT is used instead of To
        x_init = np.concatenate((xg_init,xe_init,xh_init,qc_init,Sc_init,mc_init,phic_init,Toc_init)) # unscaled
    else:
        x_init = np.concatenate((xg_init,xe_init,xh_init,qc_init,Sc_init,mc_init,phic_init)) # unscaled
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def solve_system(network,tol,max_iter,h,x0,formulation,scale_var=None,scale_var_params=None,D_F=np.array([]),D_x=np.array([]),solver = 'NR'):
    """Solve the network using analytical Jacobian and FD Jacobian, with basic NR

    Parameters
    ----------
    network : HeterogeneousNetwork
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
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = network.solve_network(tol,max_iter,h=h,solver=solver,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x)

    import examples.GN_Shabanpour as GasNet
    import examples.HN_Shabanpour as HeatNet
    gas = GasNet.gas
    rhon_g = gas.rhon
    rho_w = HeatNet.rho_w
    grav_const = HeatNet.grav_const
    print('Solution after {} it. (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print('p = {} bar'.format(p_g_vec/bar))
    print('q = {} MSCM/hour'.format(q_vec/(MSCM*rhon_g/hour)))
    print('q nodal inj = {} MSCM/hour'.format(q_inj/(MSCM*rhon_g/hour)))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase_shabanpour))
    print('P edge = {} p.u.'.format(P_edge/Sbase_shabanpour))
    print('Q edge = {} p.u.'.format(Q_edge/Sbase_shabanpour))
    print('S nodal inj = {} p.u.'.format(S_inj/Sbase_shabanpour))
    print('P hl = {} p.u.'.format([hl.P/Sbase_shabanpour for node in network.get_nodes() if isinstance(node,ElectricalNode) for hl in node.get_half_links()]))
    print('Q hl = {} p.u.'.format([hl.Q/Sbase_shabanpour for node in network.get_nodes() if isinstance(node,ElectricalNode) for hl in node.get_half_links()]))
    print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format([hl.get_m() for node in network.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Ts hl = {} C'.format([hl.get_Ts() for node in network.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Tr hl = {} C'.format([hl.get_Tr() for node in network.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in network.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Ts hl c = {} C'.format([hl.get_Ts() for node in network.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('Tr hl c = {} C'.format([hl.get_Tr() for node in network.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('dphi hl c = {} MW'.format([hl.get_dphi()/MW for node in network.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('m hl c = {} kg/s'.format([hl.get_m() for node in network.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    return x_sol,iters,err_vec

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_EH_unscaled():
    """Check solution of the example network"""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_network()
    #scaling
    rho_w = water.rhon #[kg/m^3]
    grav_const = water.g #[m/s^2]
    Ta = heat_net.Ta
    rhon_g = gas.rhon #[kg/m^3]
    formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation)

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 100
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation)

    # Then
    p_g_sol_expected = np.array([29.1044, 34.0812, 37.83572])*bar
    q_sol_expected = np.array([18.232, 16.406, 7.3671, 7.3671])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-0.12198978806814366, -0.10558020244089296])
    V_sol_expected = np.array([0.9801])*Vbase_shabanpour
    m_sol_expected = np.array([65.4362, 30.6842, -57.7832])
    p_h_sol_expected = np.array([101.814])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 117.9211, 120.9902])
    Tr_sol_expected = np.array([48.6882, 50., 49.5471])
    qc_sol_expected = np.array([12.0761681, 3.7731])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.5053, 10.5261])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3517, 10.152 ])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([96.1205, 94.466])
    phic_sol_expected = np.array([28.665633629224711,28.995986860744686])*MW
    Toc_sol_expected = np.array([122.94406491035727])
    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected,Toc_sol_expected))

    rel_tol = 1e-3
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_EH_pu():
    """Check solution of the example network, using per unit scaling"""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_network()
    #scaling
    rho_w = water.rhon #[kg/m^3]
    grav_const = water.g #[m/s^2]
    Ta = heat_net.Ta
    rhon_g = gas.rhon #[kg/m^3]
    mbase = 1.
    pbase = 1.*bar
    Tbase = 130.
    Sbase = 100.*Sbase_shabanpour #since P and Q are in p.u. (i.e. given in MW based on p.u. delta, V and line parameters)
    phibase = Sbase
    Vbase = 1.*Vbase_shabanpour
    deltabase = 1.
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase,'Vbase':Vbase,'deltabase':deltabase,'Sbase':Sbase,'Ebase':phibase}
    formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 10
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # Then
    p_g_sol_expected = np.array([29.1044, 34.0812, 37.83572])*bar
    q_sol_expected = np.array([18.232, 16.406, 7.3671, 7.3671])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-0.12198978806814366, -0.10558020244089296])
    V_sol_expected = np.array([0.9801])*Vbase_shabanpour
    m_sol_expected = np.array([65.4362, 30.6842, -57.7832])
    p_h_sol_expected = np.array([101.814])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 117.9211, 120.9902])
    Tr_sol_expected = np.array([48.6882, 50., 49.5471])
    qc_sol_expected = np.array([12.0761681, 3.7731])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.5053, 10.5261])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3517, 10.152 ])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([96.1205, 94.466])
    phic_sol_expected = np.array([28.665633629224711,28.995986860744686])*MW
    Toc_sol_expected = np.array([122.94406491035727])
    if scale_var == 'per_unit':
        q_sol_expected /= mbase
        p_g_sol_expected /= pbase
        delta_sol_expected /= deltabase
        V_sol_expected /= Vbase
        m_sol_expected /= mbase
        p_h_sol_expected /= pbase
        Ts_sol_expected /= Tbase
        Tr_sol_expected /= Tbase
        qc_sol_expected /= mbase
        Sc_sol_expected /= Sbase
        mc_sol_expected /= mbase
        phic_sol_expected /= phibase
        Toc_sol_expected /= Tbase

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected,Toc_sol_expected))

    rel_tol = 1e-3
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_EH_scaled():
    """Check solution of the example network, using scaling in solver"""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_network()
    #scaling
    rho_w = water.rhon #[kg/m^3]
    grav_const = water.g #[m/s^2]
    Ta = heat_net.Ta
    rhon_g = gas.rhon #[kg/m^3]
    qbase = 10*MSCM*rhon_g/hour
    pgbase = 30*bar
    mbase = 50.
    phbase = 1000.*rho_w*grav_const
    Tbase = 130.
    Sbase = 100.*Sbase_shabanpour #since P and Q are in p.u. (i.e. given in MW based on p.u. delta, V and line parameters)
    phibase = Sbase
    Vbase = 1.*Vbase_shabanpour
    deltabase = 1.
    formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation)

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 10
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Sbase})

    # Then
    p_g_sol_expected = np.array([29.1044, 34.0812, 37.83572])*bar
    q_sol_expected = np.array([18.232, 16.406, 7.3671, 7.3671])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-0.12198978806814366, -0.10558020244089296])
    V_sol_expected = np.array([0.9801])*Vbase_shabanpour
    m_sol_expected = np.array([65.4362, 30.6842, -57.7832])
    p_h_sol_expected = np.array([101.814])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 117.9211, 120.9902])#-Ta
    Tr_sol_expected = np.array([48.6882, 50., 49.5471])#-Ta
    qc_sol_expected = np.array([12.0761681, 3.7731])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.5053, 10.5261])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3517, 10.152 ])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([96.1205, 94.466])
    phic_sol_expected = np.array([28.665633629224711,28.995986860744686])*MW
    Toc_sol_expected = np.array([122.94406491035727])#-Ta

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected,Toc_sol_expected))

    rel_tol = 1e-3
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def comp_conv_scaling(node_set=1):
    """compare the convergence of NR using different forms of scaling. Compare with NR using Finite-Difference Jacobian."""
    het_net, gas_net, elec_net, heat_net, water, gas = create_network(node_set=node_set)
    formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}

    # Scaling
    # per unit scaling (NB: not yet, on 28-03-2019, correctly implemented for EH. Only works if all energy base values are equal).
    rho_w = water.rhon #[kg/m^3]
    grav_const = water.g #[m/s^2]
    Ta = heat_net.Ta
    rhon_g = gas.rhon #[kg/m^3]
    mbase_pu = 1.
    pbase_pu = 1.*bar
    Tbase_pu = 130.
    Sbase_pu = 10.*Sbase_shabanpour
    phibase_pu = Sbase_pu # p.u. is not correctly implemented for EH, so Sbase = phibase = Ebase for it to work
    Vbase_pu = 1.*Vbase_shabanpour
    deltabase_pu = 1.
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase_pu,'pbase':pbase_pu,'phibase':phibase_pu,'Tbase':Tbase_pu,'Vbase':Vbase_pu,'deltabase':deltabase_pu,'Sbase':Sbase_pu,'Ebase':phibase_pu}
    # base values used for scaling in solver
    qbase = mbase_pu
    pgbase = pbase_pu
    mbase = mbase_pu
    phbase = pbase_pu
    Tbase = Tbase_pu
    Sbase = Sbase_pu
    phibase = Sbase_pu
    Vbase = 1.*Vbase_shabanpour
    deltabase = 1.
    Egbase = Sbase_pu
    scale_var_params_solver={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}

    # initalize network, unscaled
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,node_set=node_set)
    # initalize network, per unit
    het_net.reset_network(x0,formulation=formulation)
    x0_pu = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,scale_var=scale_var,scale_var_params=scale_var_params,node_set=node_set)

    # compare convergence
    h = 1e-6
    tol = 1e-6
    max_iter = 50
    # solve when everything is specified in S.I., unscaled
    het_net.reset_network(x0,formulation=formulation)
    x_sol_SI,iters_SI,err_vec_SI = solve_system(het_net,tol,max_iter,h,x0,formulation)
    # solve when everything is specified in S.I., unscaled, using FD
    het_net.reset_network(x0,formulation=formulation)
    x_sol_SI_FD,iters_SI_FD,err_vec_SI_FD = solve_system(het_net,tol,max_iter,h,x0,formulation,solver='NR_FD')
    # solve when everything is specified in per unit
    het_net.reset_network(x0,formulation=formulation)
    x_sol_pu,iters_pu,err_vec_pu = solve_system(het_net,tol,max_iter,h,x0_pu,formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    # solve when everything is specified in S.I., using scaling in solver
    het_net.reset_network(x0,formulation=formulation)
    x_sol_scaled,iters_scaled,err_vec_scaled = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params=scale_var_params_solver)
    # solve when everything is specified in S.I., using scaling in solver, using FD
    het_net.reset_network(x0,formulation=formulation)
    x_sol_scaled_FD,iters_scaled_FD,err_vec_scaled_FD = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params=scale_var_params_solver,solver='NR_FD')

    fig = plt.figure('Convergence plot, node set = {}'.format(node_set))
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    max_iter_used = np.max([iters_pu,iters_SI,iters_SI_FD,iters_scaled,iters_scaled_FD])
    ax.semilogy([0,max_iter_used+1],[tol,tol],'k:',label='tolerance')
    ax.semilogy(np.asarray(range(0,iters_pu+1)),err_vec_pu,'s-',label='p.u.')
    ax.semilogy(np.asarray(range(0,iters_SI+1)),err_vec_SI,'o-',label='S.I., unscaled')
    ax.semilogy(np.asarray(range(0,iters_SI_FD+1)),err_vec_SI_FD,'d-',label='S.I., unscaled, FD')
    ax.semilogy(np.asarray(range(0,iters_scaled+1)),err_vec_scaled,'.-',label='S.I., scaled solver')
    ax.semilogy(np.asarray(range(0,iters_scaled_FD+1)),err_vec_scaled_FD,'*-',label='S.I., scaled solver, FD')
    xmin = 0
    xmax = max_iter_used
    xticks = range(xmin,xmax+1) # make sure the xticks are integers
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

if __name__ == '__main__':
    comp_conv_scaling(node_set=1)
    comp_conv_scaling(node_set=2)

    plt.show()
