"""Example of a MES with 3 nodes per carrier. Based on the example in Shabanpourt-Haghighi and Seifi"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water, Gas
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
import examples.GN_Shabanpour as GasNet
import examples.EN_Shabanpour as ElecNet
import examples.HN_Shabanpour as HeatNet
import numpy as np
import scipy.sparse as sps
import pytest
import matplotlib.pyplot as plt
from meslf.utils.constants import mm, bar, hour, kV, MW, MSCM, BTU, MBTU,degree
import warnings
import os

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous

# base values to compensate for p.u. in electrical network in Shabanpour
Pb = 1 # base I assume for complex power S (which is 1, since I work in S.I.)
Pbs_MW = 100 # factor 100 because Shabanpour give a p.u. value, using Sbase = 100 MW
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

# refence pressures
pg0 = 50*bar
ph0 = 5517.*rho_w*grav_const

def create_network(hydr_eq='fa',heat_load='outflow',adjusted_coupling=False):
    """Create a test heterogeneous network with an energy hub

    Parameters
    ----------
    hydr_eq : str, optional
        Which hydraulic equation is used in the gas network. Options are 'fa' or 'fb'. Default is 'fa'.
    heat_load : str, optional
        Which heat load model is used for the heat exchangers. Options are 'outflow' or 'delta'. Default is outflow.
    adjusted_coupling : bool, optional
        Determines which coupling models are used for the gas-boiler and the CHP. If False, basic linear models are used. If True, more complicated models are used, which more closely match the ones from Shabanpourt-Haghighi and Seifi. Default is False.

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
    g0 = GasNode('gn0',node_type=0,x=5,y=12,p=pg0) # reference node
    g1 = GasNode('gn1',node_type=1,x=0,y=7,q=10.8649*MSCM*rhon_g/hour) # load node
    g2 = GasNode('gn2',node_type=3,x=5,y=2,p=34.07704*bar,q=20*MSCM*rhon_g/hour) # reference load node
    g3 = GasNode('gn3',node_type=1,x=2.5,y=4.5,q=0.) # load node
    e0 = ElectricalNode('en0',node_type=5,x=7,y=12,P=0.1451*Sbase_shabanpour,Q=0.*Sbase_shabanpour,V=1.06*Vbase_shabanpour,delta=0.) # PQVdelta node
    e1 = ElectricalNode('en1',node_type=2,x=2,y=7,P=30.*Sbase_shabanpour,Q=15.*Sbase_shabanpour) # load
    e2 = ElectricalNode('en2',node_type=6,x=7,y=2,P=30.136*Sbase_shabanpour,Q=15.*Sbase_shabanpour,V=1.*Vbase_shabanpour) # PQV
    h0 = HeatNode('hn0',node_type=5,x=9,y=12,p=ph0) # reference node
    if heat_load == 'outflow':
        h1 = HeatNode('hn1',node_type=1,x=4,y=7,Tr_hl=50.,dphi=35.*MW) # load node (sink)
        h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
        h2 = HeatNode('hn2',node_type=3,x=9,y=2,Tr_hl=50.,dphi=20.*MW,p=4268.1087*rho_w*grav_const) # load reference node (sink)
        h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif heat_load == 'delta':
        h1 = HeatNode('hn1',node_type=12,x=4,y=7,dT=69.0370,dphi=35.*MW) # load node (sink)
        h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
        h2 = HeatNode('hn2',node_type=13,x=9,y=2,dT=73.5410,dphi=20.*MW,p=4268.1087*rho_w*grav_const) # load reference node (sink)
        h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    else:
        raise ValueError("Enter valid value for 'heat_load'. Either 'outflow' or 'delta'.")

    Pbs_GG_MW = 1 # The values in Shabanpour are specified for P given in 1MW
    Pbs_GG = Pbs_GG_MW*MW
    qbs = 1*MSCM*rhon_g/hour
    GHVbs = MBTU*BTU*hour/(rhon_g)
    Ebs = qbs*GHVbs
    a_GG_pu = 0.01 # Assumes P to be in 100 MW, and q to be in MSCM
    b_GG_pu = 4.
    c_GG_pu = 150.
    d_GG_pu = 15.
    e_GG_pu = 0.5
    Pmin_pu = 0.
    a_GG = a_GG_pu * Ebs/Pbs_GG**2 *Pb**2
    b_GG = b_GG_pu * Ebs/Pbs_GG *Pb
    c_GG = c_GG_pu * Ebs
    d_GG = d_GG_pu * Ebs
    e_GG = e_GG_pu * 1/Pbs_GG *Pb
    Pmin = Pmin_pu * Pbs_GG / Pb
    print('a = {}, b = {}, c = {}, d = {}, e = {}, Pmin = {}'.format(a_GG,b_GG,c_GG,d_GG,e_GG,Pmin))
    c0 = HeterogeneousNode('cn0',node_type=0,x=6,y=13,unit_type='ge_gas_fired_gen_valve_point',unit_params={'a':a_GG,'b':b_GG,'c':c_GG,'d':d_GG,'e':e_GG,'Pmin':Pmin,'GHV':GHV})

    if adjusted_coupling:
        Cp = water.Cp
        m_CHP = 25*MW/(Cp*(130-49.5339))
        m_GB = 4*MW/(Cp*(110-49.5339))
        To_c_CHP = (130*m_CHP + 110*m_GB)/(m_CHP+m_GB)
        print('To CHP = {}'.format(To_c_CHP))
        To_c_GB = 120.
        Tr0 = 48.6805
        Tr1 = 49.5339
        Ess = 53*water.Cp*22.22
        print('Ess = {}'.format(Ess))
        a_GB = -.0377801*MW/Ess
        b_GB = .797347
        unit_type_GB = 'gh_gas_boiler_part_load'
        unit_params_GB = {'a':a_GB,'b':b_GB,'Ess':Ess,'GHV':GHV}
        a_CHP = .463*Pb # I haven't checked if the Pb is correct or if it should be /Pb
        b_CHP = -.04532*MW*Pb # I haven't checked if the Pb is correct or if it should be /Pb
        d_CHP = 4.49*MW*Pb # I haven't checked if the Pb is correct or if it should be /Pb
        eta_CHP = np.array([0.88/Pb, 0.88])
        phimin = 10*MW #??
        phimax = 14*MW/.48 # maximum (?) active power / power-to-heat-ratio
        L1 = .8
        # Should stay above the upper limit L1, so these following values shouldn't matter
        L2 = .6
        r1 = .0736
        r2 = .0845
        unit_type_CHP = 'geh_CHP_part_load'
        unit_params_CHP = {'eta':eta_CHP,'a':a_CHP,'b':b_CHP,'d':d_CHP,'L1':L1,'L2':L2,'r1':r1,'r2':r2,'phimin':phimin,'phimax':phimax,'GHV':GHV}
    else:
        eta_GB = 0.88
        eta_CHP = np.array([0.88/Pb, 0.88])
        To_c_CHP = 126.49315908196526
        To_c_GB = 120.
        Tr0 = 48.68310632
        Tr1 = 49.53799927
        unit_type_GB = 'gh_gas_boiler'
        unit_params_GB = {'eta':eta_GB,'GHV':GHV}
        unit_type_CHP = 'geh_CHP'
        unit_params_CHP = {'eta':eta_CHP,'GHV':GHV}
    if heat_load == 'outflow':
        c1 = HeterogeneousNode('cn1',node_type=1,x=8,y=14,unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
        c2 = HeterogeneousNode('cn2',node_type=1,x=7,y=0,unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To known
    elif heat_load == 'delta':
        c1 = HeterogeneousNode('cn1',node_type=2,x=8,y=14,unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
        c2 = HeterogeneousNode('cn2',node_type=2,x=7,y=0,unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To known

    # links
    L_g = 30000. #[m]
    D_g = 150.*mm #[mm]
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
    gl5 = GasLink('gl5',g0,c1)
    gl6 = GasLink('gl6',g2,c2)
    x = 0.5/Pbs_MW
    r = 0.05/Pbs_MW
    z = r+x*1j
    b = -x/(np.abs(z)**2)
    g = r/(np.abs(z)**2)
    b *= Ybase_shabanpour
    g *= Ybase_shabanpour
    print('b = {}, g = {}'.format(b,g))
    elec_link_params = {'b':b,'g':g}
    el0 = ElectricalLink('el0',e0,e1,link_type='short_line',link_params=elec_link_params)
    el1 = ElectricalLink('el1',e0,e2,link_type='short_line',link_params=elec_link_params)
    el2 = ElectricalLink('el2',e1,e2,link_type='short_line',link_params=elec_link_params)
    el3 = ElectricalLink('el3',c0,e0)
    el4 = ElectricalLink('el4',c2,e2)
    L_h = L_g #[m]
    D_h = D_g #[m]
    lam = 0.2 #[W/(mK)]
    eps_h = 1.25*mm #[m]
    U = lam/(np.pi*D_h) #[W/(m^2 K)]
    Ta = 10. #[C]
    heat_link_params = {'D':D_h,'L':L_h,'eps':eps_h,'U':U,'carrier':water,'Ta':Ta}
    heat_link_type = 'standard_pipe_low_pres_colebrook'
    hl0 = HeatLink('hl0',h0,h1,link_type=heat_link_type,link_params=heat_link_params)
    hl1 = HeatLink('hl1',h0,h2,link_type=heat_link_type,link_params=heat_link_params.copy())
    hl2 = HeatLink('hl2',h1,h2,link_type=heat_link_type,link_params=heat_link_params.copy())
    if heat_load == 'outflow':
        hl3 = HeatLink('hl3',c1,h0,link_params={'carrier':water},bc_type=6,Tsstart=To_c_GB) # To of coupling (source) is known
        hl4 = HeatLink('hl4',c2,h2,link_params={'carrier':water},bc_type=6,Tsstart=To_c_CHP) # To of coupling (source) is known
    elif heat_load == 'delta':
        hl3 = HeatLink('hl3',c1,h0,link_params={'carrier':water},bc_type=10,dTstart=To_c_GB-Tr0) # dT of coupling (source) is known
        hl4 = HeatLink('hl4',c2,h2,link_params={'carrier':water},bc_type=10,dTstart=To_c_CHP-Tr1) # dT of coupling (source) is known

    # network
    gas_net = GasNetwork('test gas network')
    gas_net.add_link(gl0)
    gas_net.add_link(gl1)
    gas_net.add_link(gl2)
    gas_net.add_link(gl3)
    gas_net.add_link(gl4)
    gas_net.add_link(gl5)
    gas_net.add_link(gl6)

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
    het_net.add_node(c2)

    return het_net, gas_net, elec_net, heat_net, water, gas

def initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,heat_load='outflow',scale_var=None,scale_var_params=None):
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

    gp_init = np.array([40.,40.])*bar
    q_init = np.array([20.,20.,20.,20.])*MSCM*rhon_g/hour
    delta_init = np.array([0.,0.]) # default values
    V_init = np.array([1.])*Vbase_shabanpour # default values
    m_init = np.array([60.,30.,60.])
    if formulation['heat'] == 'half_link_flow':
        m_hl_init = np.array([20.,20.])
        if heat_load == 'delta':
                Tr_hl_init = np.array([50.,50.]) #[C]
    hp_init = np.array([10.])*rho_w*grav_const
    Ts_init = np.array([100.,120.,120.])#-Ta
    Tr_init = np.array([50.,50.,50.])#-Ta
    qc_init = np.array([10,3,3])*MSCM*rhon_g/hour
    Pc_init = np.array([50.,10.])*Sbase_shabanpour
    Qc_init = np.array([0.,0.])*Sbase_shabanpour
    Sc_init = np.concatenate((Pc_init,Qc_init))
    mc_init = np.array([10.,10.])
    phic_init = np.array([30.,25.])*MW
    Toc_init = np.array([100.,100.])

    xg_init = np.concatenate((q_init,gp_init))
    xe_init = np.concatenate((delta_init,V_init))
    if formulation['heat'] == 'half_link_flow':
        if heat_load == 'delta':
            xh_init = np.concatenate((m_init,m_hl_init,hp_init,Ts_init,Tr_init,Tr_hl_init))
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

def solve_system(network,tol,max_iter,h,x0,formulation,scale_var=None,scale_var_params=None,D_F=np.array([]),D_x=np.array([]),P_F=np.array([]),P_x=np.array([]),solver = 'NR'):
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
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = network.solve_network(tol,max_iter,h=h,solver=solver,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x)

    gas = GasNet.gas
    rhon_g = gas.rhon
    rho_w = HeatNet.rho_w
    grav_const = HeatNet.grav_const
    print('Solution after {} it. (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print('p = {} bar'.format(p_g_vec/bar))
    print('q = {} MSCM/hour'.format(q_vec/(MSCM*rhon_g/hour)))
    print('q nodal inj = {} MSCM/hour'.format(q_inj/(MSCM*rhon_g/hour)))
    print('delta = {} degree'.format(delta_vec/degree))
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

def perm_matr_x(het_net,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}):
    """Determines the permutation matrices P_x and P_F, when only reordering by putting the coupling variables with the single-carrier parts, and by putting the heat coupling equations (Fphi and FdT) with the heat part

    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The network for which the permutation matrix needs to be made.

    Returns
    -------
    P_x : scipy sparse matrix
        Permutation matrix for the vector of variables x
    P_F : scipy sparse matrix
        Permutation matrix for the vector of equations F
    """
    # Determine new indices for x
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)
    x_length = len(x_entries)
    Px_row = list(range(x_length)) # New indices
    Px_data = [1]*x_length # Permutation matrix is binary matrix
    Px_col = [ind for ind,el in enumerate(x_entries) if 'Gas' in type(el).__name__] + [ind for ind,el in enumerate(x_entries) if 'Elec' in type(el).__name__] + [ind for ind,el in enumerate(x_entries) if 'Heat' in type(el).__name__] # Original indices
    P_x = sps.csr_matrix((Px_data,(Px_row,Px_col)),shape=(x_length,x_length))

    # Determine new indices for F
    F_entries, Fg_entries, Fe_entries, known_P_nodes, known_Q_nodes, Fh_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_dT_halflinks, Fc_entries, F_fc_nodes, F_fc_amount, F_phi_nodes, F_dT_nodes = het_net.get_F_entries(formulation=formulation)
    F_length = len(Fg_entries) + len(Fe_entries) + len(Fh_entries) + np.sum(F_fc_amount) + len(F_phi_nodes) + len(F_dT_nodes)
    PF_row = list(range(F_length)) # New indices
    PF_data = [1]*F_length # Permutation matrix is binary matrix
    PF_col = [ind for ind,el in enumerate(F_entries) if el in Fg_entries] + [ind for ind,el in enumerate(F_entries) if el in Fe_entries] + [ind for ind,el in enumerate(F_entries) if el in Fh_entries] # Original indices
    for ind_el,el in enumerate(F_phi_nodes + F_dT_nodes):
        PF_col.append(len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+np.sum(F_fc_amount)+ind_el)
    for ind_el,el in enumerate(F_fc_nodes):
        ind = ind_el+len(Fg_entries)+len(Fe_entries)+len(Fh_entries)
        if ind_el>0:
            ind += np.sum(F_fc_amount[0:ind_el])-ind_el # -ind_el, because index is already shifted by one with respect to previous element because ind_el has increased 1
        fc_len = F_fc_amount[ind_el]
        for fc_ind in range(fc_len):
            PF_col.append(ind + fc_ind)
    P_F = sps.csr_matrix((PF_data,(PF_row,PF_col)),shape=(F_length,F_length))
    return P_x, P_F

def perm_matr_xF(het_net,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}):
    """Determines the permutation matrices P_x and P_F, reordering x and all coupling equations

    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The network for which the permutation matrix needs to be made.

    Returns
    -------
    P_x : scipy sparse matrix
        Permutation matrix for the vector of variables x
    P_F : scipy sparse matrix
        Permutation matrix for the vector of equations F
    """
    # Determine new indices for x
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)
    x_length = len(x_entries)
    Px_row = list(range(x_length)) # New indices
    Px_data = [1]*x_length # Permutation matrix is binary matrix
    xg_new = [ind for ind,el in enumerate(x_entries) if 'Gas' in type(el).__name__]
    xe_new = [ind for ind,el in enumerate(x_entries) if 'Elec' in type(el).__name__]
    xh_new = [ind for ind,el in enumerate(x_entries) if 'Heat' in type(el).__name__]
    Px_col = xg_new + xe_new + xh_new # Original indices
    P_x = sps.csr_matrix((Px_data,(Px_row,Px_col)),shape=(x_length,x_length))

    # Determine new indices for F
    F_entries, Fg_entries, Fe_entries, known_P_nodes, known_Q_nodes, Fh_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_dT_halflinks, Fc_entries, F_fc_nodes, F_fc_amount, F_phi_nodes, F_dT_nodes = het_net.get_F_entries(formulation=formulation)
    F_length = len(Fg_entries) + len(Fe_entries) + len(Fh_entries) + np.sum(F_fc_amount) + len(F_phi_nodes) + len(F_dT_nodes)
    PF_row = list(range(F_length)) # New indices
    PF_data = [1]*F_length # Permutation matrix is binary matrix
    Fg_new = [ind for ind,el in enumerate(F_entries) if el in Fg_entries]
    Fe_new = [ind for ind,el in enumerate(F_entries) if el in Fe_entries]
    Fh_new = [ind for ind,el in enumerate(F_entries) if el in Fh_entries]
    for ind_el,el in enumerate(F_phi_nodes + F_dT_nodes):
        Fh_new.append(len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+np.sum(F_fc_amount)+ind_el)
    Fc_new = list()
    for ind_el,el in enumerate(F_fc_nodes):
        ind = ind_el+len(Fg_entries)+len(Fe_entries)+len(Fh_entries)
        if ind_el>0:
            ind += np.sum(F_fc_amount[0:ind_el])-ind_el # -ind_el, because index is already shifted by one with respect to previous element because ind_el has increased 1
        fc_len = F_fc_amount[ind_el]
        for fc_ind in range(fc_len):
            Fc_new.append(ind + fc_ind)
    while len(xg_new) > len(Fg_new):
        if not Fc_new:
            print('No more coupling equations available to move. Unable to make gas part square: |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
            break
        for ind_el,el in enumerate(F_fc_nodes):
            ind = len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+int(np.sum(F_fc_amount[:ind_el]))
            if F_fc_amount[ind_el]>1:
                dfc_dE = el.der_node_law_dE()
                dfc_dq = dfc_dE[:,0]
                for ind_q,der_q in enumerate(dfc_dq): # this coupling function depends on q
                    if der_q:
                        Fg_new.append(ind + ind_q)
                        Fc_new.remove(ind + ind_q)
                        if len(xg_new) <= len(Fg_new):
                            print('Gas part is now square! |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
                            break
            else:
                dfc_dq,dfc_dP,dfc_dphi = el.der_node_law_dE()
                if dfc_dq != None: # this coupling function depends on q
                    Fg_new.append(ind)
                    Fc_new.remove(ind)
                    if len(xg_new) <= len(Fg_new):
                            print('Gas part is now square! |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
                            break
        print('No more coupling equations available to move. Unable to make gas part square: |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
        break
    while len(xe_new) > len(Fe_new):
        if not Fc_new:
            print('No more coupling equations available to move. Unable to make electrical part square: |xe| = {}, |Fe| = {}'.format(len(xe_new),len(Fe_new)))
            break
        for ind_el,el in enumerate(F_fc_nodes):
            ind = len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+int(np.sum(F_fc_amount[:ind_el]))
            if F_fc_amount[ind_el]>1:
                dfc_dE = el.der_node_law_dE()
                dfc_dP = dfc_dE[:,1]
                for ind_P,der_P in enumerate(dfc_dP): # this coupling function depends on P
                    if der_P:
                        if ind+ind_P in Fc_new: # check if not already moved to Fc
                            Fe_new.append(ind + ind_P)
                            Fc_new.remove(ind + ind_P)
            else:
                dfc_dq,dfc_dP,dfc_dphi = el.der_node_law_dE()
                if dfc_dP != None: # this coupling function depends on P
                    if ind in Fc_new: # check if not already moved to Fc
                        Fe_new.append(ind)
                        Fc_new.remove(ind)
        print('No more coupling equations available to move. Unable to make electrical part square: |xe| = {}, |Fe| = {}'.format(len(xe_new),len(Fe_new)))
        break
    while len(xh_new) > len(Fh_new):
        if not Fc_new:
            print('No more coupling equations available to move. Unable to make heat part square: |xh| = {}, |Fh| = {}'.format(len(xh_new),len(Fh_new)))
            break
        for ind_el,el in enumerate(F_fc_nodes):
            ind = len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+int(np.sum(F_fc_amount[:ind_el]))
            if F_fc_amount[ind_el]>1:
                dfc_dE = el.der_node_law_dE()
                dfc_dphi = dfc_dE[:,2]
                for ind_phi,der_phi in enumerate(dfc_dphi): # this coupling function depends on phi
                    if der_phi:
                        if ind+ind_phi in Fc_new: # check if not already moved to Fc
                            Fh_new.append(ind + ind_phi)
                            Fc_new.remove(ind + ind_phi)
            else:
                dfc_dq,dfc_dP,dfc_dphi = el.der_node_law_dE()
                if dfc_dphi != None: # this coupling function depends on phi
                    if ind in Fc_new: # check if not already moved to Fc
                        Fh_new.append(ind)
                        Fc_new.remove(ind)
        print('No more coupling equations available to move. Unable to make heat part square: |xh| = {}, |Fh| = {}'.format(len(xh_new),len(Fh_new)))
        break
    PF_col = Fg_new + Fe_new + Fh_new + Fc_new
    P_F = sps.csr_matrix((PF_data,(PF_row,PF_col)),shape=(F_length,F_length))
    return P_x, P_F

def jacobians(hydr_eq='fa',heat_load='outflow',formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}):
    """Plot Jacobian matrices, and eigenvalue spectra, for different indices / ordering"""

    # create network
    het_net, gas_net, elec_net, heat_net, water, gas = create_network(hydr_eq=hydr_eq,heat_load=heat_load)

    # initialize
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,heat_load=heat_load)

    # create system of equations
    from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
    rhon_g = gas.rhon
    rho_w = water.rhon
    grav_const = water.g
    pgbase=50*bar
    qbase=10*MSCM*rhon_g/hour
    Vbase=10/np.sqrt(3)*kV
    Sbase=10*MW
    phbase=1000.*rho_w*grav_const
    mbase=50
    Tbase=100
    phibase=10*MW
    Egbase=10*MW
    deltabase=1

    scale_var = 'matrix'
    scale_var_params = {'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsys_unscaled = NonLinearSystemHeterogeneous(het_net,formulation=formulation)
    F0 = nlsys.F(x0)
    J0 = nlsys.J(x0)

    # unscaled system
    fig_J = nlsys_unscaled.spy_plot_J(x0,title='Jacobian spy plot')
    ax_J = plt.gca()
    fig_J_map = nlsys_unscaled.imshow_J(x0,title='Jacobian')

    # spectrum of Jacobian
    fig_spec = nlsys_unscaled.spectrum_J(x0,title='Spectrum')

    # scaled
    D_x = nlsys.Dx()
    D_x_inv = sps.diags(1/D_x.data[0])
    D_F = nlsys.DF()
    J_scaled = D_F.dot(J0.dot(D_x_inv))
    # spy plot of scaled J over original
    nlsys.spy_plot_J(x0,ax=ax_J,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5)

    # colormap of Jacobian
    fig_J_scaled_map = nlsys.imshow_J(x0,title='Scaled Jacobian')

    # spectrum of (scaled) Jacobian
    fig_spectra = nlsys.spectrum_J(x0,title='Scaled spectra',color='tab:blue')
    ax_spectra = fig_spectra.gca()

    # reordering (coupling with single-carrier parts), first only reorder x
    P_x_re_x, P_F_re_x = perm_matr_x(het_net,formulation=formulation)
    fig_J_re_x = nlsys.spy_plot_J(x0,title='Reordered Jacobian, only x',P_F=P_F_re_x,P_x=P_x_re_x)

    # also reorder F (based on which parts or non-square)
    P_x_re_xF, P_F_re_xF = perm_matr_xF(het_net,formulation=formulation)
    fig_J_re_xF = nlsys.spy_plot_J(x0,title='Reordered Jacobian, both x and F',P_F=P_F_re_xF,P_x=P_x_re_xF)

    J_re_x = P_F_re_x.dot(J_scaled).dot(P_x_re_x.transpose())
    J_re_xF = P_F_re_xF.dot(J_scaled).dot(P_x_re_xF.transpose())
    print('\nFor scaled Jacobians:')
    print('det(J) = {}'.format(np.linalg.det(J_scaled.todense())))
    print('det(J), reordering x = {}'.format(np.linalg.det(J_re_x.todense())))
    print('det(J), reordering x and F = {}'.format(np.linalg.det(J_re_xF.todense())))
    print('cond(J) = {}'.format(np.linalg.cond(J_scaled.todense())))
    print('cond(J), reordering x = {}'.format(np.linalg.cond(J_re_x.todense())))
    print('cond(J), reordering x and F = {}'.format(np.linalg.cond(J_re_xF.todense())))

    # spectra of scaled systems in one plot
    nlsys.spectrum_J(x0,ax=ax_spectra,P_F=P_F_re_x,P_x=P_x_re_x,color='tab:red')
    nlsys.spectrum_J(x0,ax=ax_spectra,P_F=P_F_re_xF,P_x=P_x_re_xF,color='tab:green')
    from matplotlib.lines import Line2D
    ax_spectra.legend(handles=[Line2D([0], [0], color='tab:blue', marker='.',label='original'),
                       Line2D([0], [0], color='tab:red', marker='.',label='reordered x'),
                       Line2D([0], [0], color='tab:green', marker='.',label='reordered x and F')])

    max_iter = 50
    tol = 1e-6
    # solve system for the different reorderings
    het_net.reset_network(x0,formulation=formulation)
    print('\nSolving original system')
    x_sol,iters,err_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('Errors orig: {}'.format(err_vec))
    het_net.reset_network(x0,formulation=formulation)
    print('\nSolving system when reordering x (and coupling equations clearly related to heat)')
    x_sol_re_x,iters_re_x,err_vec_re_x,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,P_F=P_F_re_x,P_x=P_x_re_x)
    print('Errors reordering x (and coupling equations clearly related to heat): {}'.format(err_vec_re_x))
    het_net.reset_network(x0,formulation=formulation)
    print('\nSolving system when reordering x and F')
    x_sol_re_xF,iters_re_xF,err_vec_re_xF,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,P_F=P_F_re_xF,P_x=P_x_re_xF)
    print('Errors reordering x and F: {}'.format(err_vec_re_xF))

    fig = plt.figure('Convergence plot different orderings')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||D_F F(x^k)||_2$')
    max_iter_used = np.max([iters,iters_re_x,iters_re_xF])
    ax.semilogy([0,max_iter_used+1],[tol,tol],'k:',label='tolerance')
    ax.semilogy(np.asarray(range(0,iters+1)),err_vec,'.-',color='tab:blue',label='original')
    ax.semilogy(np.asarray(range(0,iters_re_x+1)),err_vec_re_x,'.-',color='tab:red',label='reordered x')
    ax.semilogy(np.asarray(range(0,iters_re_xF+1)),err_vec_re_xF,'.-',color='tab:green',label='reordered x and F')
    xmin = 0
    xmax = max_iter_used
    xticks = range(xmin,xmax+1) # make sure the xticks are integers
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_unscaled():
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
    max_iter = 10
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation)

    # Then
    p_g_sol_expected = np.array([29.1021,37.83273])*bar
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-0.12197757076337971, -0.10555751316061704])
    V_sol_expected = np.array([0.9801])*Vbase_shabanpour
    m_sol_expected = np.array([64.6877,31.4083,-56.5379])
    p_h_sol_expected = np.array([224.9319])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 119.0370, 123.5410])#-Ta
    Tr_sol_expected = np.array([48.6805,50.,49.5339])#-Ta
    qc_sol_expected = np.array([9.3382, 2.7363, 3.7756])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.4982, 10.5331])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3515, 10.1507])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([96.096, 90.1578])
    phic_sol_expected = np.array([28.6616,29.0152])*MW

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected))

    rel_tol = 1e-3
    print('difference between solution: {}'.format(x_sol-x_sol_expected))
    print('maximum difference: {}'.format(np.max(np.abs(x_sol-x_sol_expected))))
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_pu():
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
    print('errors: {}'.format(err_vec))

    # Then
    p_g_sol_expected = np.array([29.1021,37.83273])*bar
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-0.12197757076337971, -0.10555751316061704])
    V_sol_expected = np.array([0.9801])*Vbase_shabanpour
    m_sol_expected = np.array([64.6877,31.4083,-56.5379])
    p_h_sol_expected = np.array([224.9319])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 119.0370, 123.5410])
    Tr_sol_expected = np.array([48.6805,50.,49.5339])
    qc_sol_expected = np.array([9.3382, 2.7363, 3.7756])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.4982, 10.5331])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3515, 10.1507])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([96.096, 90.1578])
    phic_sol_expected = np.array([28.6616,29.0152])*MW
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

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected))

    rel_tol = 1e-3
    print('difference between solution: {}'.format(x_sol-x_sol_expected))
    print('maximum difference: {}'.format(np.max(np.abs(x_sol-x_sol_expected))))
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_scaled():
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
    Egbase = phibase
    Vbase = 1.*Vbase_shabanpour
    deltabase = 1.
    formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation)

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 10
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})

    # Then
    p_g_sol_expected = np.array([29.1021,37.83273])*bar
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-0.12197757076337971, -0.10555751316061704])
    V_sol_expected = np.array([0.9801])*Vbase_shabanpour
    m_sol_expected = np.array([64.6877,31.4083,-56.5379])
    p_h_sol_expected = np.array([224.9319])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 119.0370, 123.5410])#-Ta
    Tr_sol_expected = np.array([48.6805,50.,49.5339])#-Ta
    qc_sol_expected = np.array([9.3382, 2.7363, 3.7756])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.4982, 10.5331])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3515, 10.1507])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([96.096, 90.1578])
    phic_sol_expected = np.array([28.6616,29.0152])*MW

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected))

    rel_tol = 1e-3
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_unknown_half_link_scaled():
    """Check solution of the example network, using scaling in solver, and using the 'half_link_flow' formulation in the heat network."""
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
    Egbase = phibase
    Vbase = 1.*Vbase_shabanpour
    deltabase = 1.
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation)

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 10
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})

    # Then
    p_g_sol_expected = np.array([29.1021,37.83273])*bar
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-0.12197757076337971, -0.10555751316061704])
    V_sol_expected = np.array([0.9801])*Vbase_shabanpour
    m_sol_expected = np.array([64.6877,31.4083,-56.5379])
    m_hl_sol_expected = np.array([121.2256,65.0282])
    p_h_sol_expected = np.array([224.9319])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 119.0370, 123.5410])#-Ta
    Tr_sol_expected = np.array([48.6805,50.,49.5339])#-Ta
    qc_sol_expected = np.array([9.3382, 2.7363, 3.7756])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.4982, 10.5331])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3515, 10.1507])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([96.096, 90.1578])
    phic_sol_expected = np.array([28.6616,29.0152])*MW

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,m_hl_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected))

    rel_tol = 1e-3
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_scaled_dT():
    """Check solution of the example network, using scaling in solver, and assuming delta T known instead of To known for the coupling components"""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_network()
    Tr_sol_expected = np.array([48.6805,50.,49.5339])
    for node in het_net.get_nodes():
        if isinstance(node,HeterogeneousNode):
            if node.name == 'cn1':
                node.node_type = 2
                for hl in node.get_links():
                    if isinstance(hl,HeatLink):
                        To_c_GB = hl.Tsstart
                        hl.dTstart = hl.Tsstart - Tr_sol_expected[0]
                        hl.bc_type = 10
            elif node.name == 'cn2':
                node.node_type = 2
                for hl in node.get_links():
                    if isinstance(hl,HeatLink):
                        To_c_CHP = hl.Tsstart
                        hl.dTstart = hl.Tsstart - Tr_sol_expected[2]
                        hl.bc_type = 10
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
    Egbase = phibase
    Vbase = 1.*Vbase_shabanpour
    deltabase = 1.
    formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation)

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 10
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})

    # Then
    p_g_sol_expected = np.array([29.1021,37.83273])*bar
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-0.12197757076337971, -0.10555751316061704])
    V_sol_expected = np.array([0.9801])*Vbase_shabanpour
    m_sol_expected = np.array([64.6877,31.4083,-56.5379])
    p_h_sol_expected = np.array([224.9319])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 119.0370, 123.5410])
    qc_sol_expected = np.array([9.3382, 2.7363, 3.7756])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.4982, 10.5331])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3515, 10.1507])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([96.096, 90.1578])
    phic_sol_expected = np.array([28.6616,29.0152])*MW
    Toc_sol_expected = np.array([To_c_GB,To_c_CHP])

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected,Toc_sol_expected))

    rel_tol = 1e-3
    print('difference between solution: {}'.format(x_sol-x_sol_expected))
    print('maximum difference: {}'.format(np.max(np.abs(x_sol-x_sol_expected))))
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_unknown_half_link_scaled_dT():
    """Check solution of the example network, using scaling in solver, and using the 'half_link_flow' formulation in the heat network, and assuming delta T known instead of To known for the coupling components."""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_network()
    Tr_sol_expected = np.array([48.6805,50.,49.5339])
    for node in het_net.get_nodes():
        if isinstance(node,HeterogeneousNode):
            if node.name == 'cn1':
                node.node_type = 2
                for hl in node.get_links():
                    if isinstance(hl,HeatLink):
                        To_c_GB = hl.Tsstart
                        hl.dTstart = hl.Tsstart - Tr_sol_expected[0]
                        hl.bc_type = 10
            elif node.name == 'cn2':
                node.node_type = 2
                for hl in node.get_links():
                    if isinstance(hl,HeatLink):
                        To_c_CHP = hl.Tsstart
                        hl.dTstart = hl.Tsstart - Tr_sol_expected[2]
                        hl.bc_type = 10
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
    Egbase = phibase
    Vbase = 1.*Vbase_shabanpour
    deltabase = 1.
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation)

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 10
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})

    # Then
    p_g_sol_expected = np.array([29.1021,37.83273])*bar
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-0.12197757076337971, -0.10555751316061704])
    V_sol_expected = np.array([0.9801])*Vbase_shabanpour
    m_sol_expected = np.array([64.6877,31.4083,-56.5379])
    m_hl_sol_expected = np.array([121.2256,65.0282])
    p_h_sol_expected = np.array([224.9319])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 119.0370, 123.5410])#-Ta
    qc_sol_expected = np.array([9.3382, 2.7363, 3.7756])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.4982, 10.5331])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3515, 10.1507])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([96.096, 90.1578])
    phic_sol_expected = np.array([28.6616,29.0152])*MW
    Toc_sol_expected = np.array([To_c_GB,To_c_CHP])

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,m_hl_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected,Toc_sol_expected))

    rel_tol = 1e-3
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_known_phi():
    """Check solution of the example network, assuming the heat power produced by the CHP known. Use more appropriate values for the efficiencies."""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_network()
    rhon_g = gas.rhon #[kg/m^3]
    rho_w = water.rhon #[kg/m^3]
    grav_const = water.g #[m/s^2]
    # change the half link type to indicate that heat power is known
    for node in het_net.get_nodes(unit_types=['geh_CHP']):
        GHV = node.unit_params.get('GHV')
        eta_CHP =  (10.1730 + 29)*MW/(GHV*3.776*(MSCM*rhon_g/hour))
        print('eta CHP = {}'.format(eta_CHP))
        node.set_type('geh_CHP',{'eta':np.array([eta_CHP/Pb, eta_CHP]),'GHV':GHV})
        for hl in node.get_links():
            if isinstance(hl,HeatLink):
                hl.bc_type = 2 # both To and phi known
                hl.dphistart = -29*MW#.0152*MW #< 0 because it is a source
                Cp = water.Cp
                m_CHP = 25*MW/(Cp*(130-49.5339))
                m_GB = 4*MW/(Cp*(110-49.5339))
                To = (130*m_CHP + 110*m_GB)/(m_CHP+m_GB)
                hl.Tsstart = To
    for node in het_net.get_nodes(unit_types=['gh_gas_boiler']):
        GHV = node.unit_params.get('GHV')
        eta_GB = 28.6768*MW/(GHV*3.0177*(MSCM*rhon_g/hour))
        print('eta GB = {}'.format(eta_GB))
        node.set_type('gh_gas_boiler',{'eta':eta_GB,'GHV':GHV})
    # change the node type of heat node 2
    for node in heat_net.get_nodes():
        if node.name == 'hn2':
            node.node_type = 1 # load (sink) node
    # initialize network
    formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}
    gp_init = np.array([40.,40.])*bar
    q_init = np.array([20.,20.,20.,20.])*MSCM*rhon_g/hour
    delta_init = np.array([0.,0.]) # default values
    V_init = np.array([1.])*Vbase_shabanpour # default values
    m_init = np.array([60.,30.,60.])
    hp_init = np.array([10.,4000])*rho_w*grav_const
    Ts_init = np.array([100.,120.,120.])
    Tr_init = np.array([50.,50.,50.])
    qc_init = np.array([10,3,3])*MSCM*rhon_g/hour
    Pc_init = np.array([50.,10.])*Sbase_shabanpour
    Qc_init = np.array([0.,0.])*Sbase_shabanpour
    Sc_init = np.concatenate((Pc_init,Qc_init))
    mc_init = np.array([10.,10.])
    phic_init = np.array([30.])*MW
    xg_init = np.concatenate((q_init,gp_init))
    xe_init = np.concatenate((delta_init,V_init))
    xh_init = np.concatenate((m_init,hp_init,Ts_init,Tr_init))
    x_init = np.concatenate((xg_init,xe_init,xh_init,qc_init,Sc_init,mc_init,phic_init))
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 10
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation)

    # Then
    p_g_sol_expected = np.array([29.1021,37.83273])*bar
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-7.0219,-6.1156])*degree
    V_sol_expected = np.array([0.9800])*Vbase_shabanpour
    m_sol_expected = np.array([64.6943,31.4533,-56.5335])
    p_h_sol_expected = np.array([223.05247021, 4264.91754827])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 119.0370, 123.5410])
    Tr_sol_expected = np.array([48.6805,50.,49.5339])
    qc_sol_expected = np.array([9.4501, 3.0177, 3.7756])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.8662, 10.1730])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3630, 10.2181])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([28.6768*MW/(Cp*(120-48.6805)), 29*MW/(Cp*(To-49.5339))])
    phic_sol_expected = np.array([28.6768])*MW

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected))

    rel_tol = 1e-3
    print('solution: {}'.format(x_sol))
    print('difference between solution: {}'.format(x_sol-x_sol_expected))
    print('maximum difference: {}'.format(np.max(np.abs(x_sol-x_sol_expected))))
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_permuted():
    """Check solution of the example network, using scaling in solver"""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_network()
    formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation)
    rhon_g = gas.rhon #[kg/m^3]
    rho_w = water.rhon #[kg/m^3]
    grav_const = water.g #[m/s^2]
    # permutation
    P_x, P_F = perm_matr_xF(het_net,formulation=formulation)

    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 10
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation,P_F=P_F,P_x=P_x)

    # Then
    p_g_sol_expected = np.array([29.1021,37.83273])*bar
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-0.12197757076337971, -0.10555751316061704])
    V_sol_expected = np.array([0.9801])*Vbase_shabanpour
    m_sol_expected = np.array([64.6877,31.4083,-56.5379])
    p_h_sol_expected = np.array([224.9319])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 119.0370, 123.5410])#-Ta
    Tr_sol_expected = np.array([48.6805,50.,49.5339])#-Ta
    qc_sol_expected = np.array([9.3382, 2.7363, 3.7756])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.4982, 10.5331])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3515, 10.1507])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([96.096, 90.1578])
    phic_sol_expected = np.array([28.6616,29.0152])*MW

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected))

    rel_tol = 1e-3
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_scaled_permuted():
    """Check solution of the example network, using scaling in solver"""
    # Given
    het_net, gas_net, elec_net, heat_net, water, gas = create_network()
    formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation)
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
    Egbase = phibase
    Vbase = 1.*Vbase_shabanpour
    deltabase = 1.
    # permutation
    P_x, P_F = perm_matr_xF(het_net,formulation=formulation)
    # When
    h = 1e-6
    tol = 1e-6
    max_iter = 10
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase},P_x=P_x,P_F=P_F)

    # Then
    p_g_sol_expected = np.array([29.1021,37.83273])*bar
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-0.12197757076337971, -0.10555751316061704])
    V_sol_expected = np.array([0.9801])*Vbase_shabanpour
    m_sol_expected = np.array([64.6877,31.4083,-56.5379])
    p_h_sol_expected = np.array([224.9319])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 119.0370, 123.5410])#-Ta
    Tr_sol_expected = np.array([48.6805,50.,49.5339])#-Ta
    qc_sol_expected = np.array([9.3382, 2.7363, 3.7756])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.4982, 10.5331])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3515, 10.1507])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    mc_sol_expected = np.array([96.096, 90.1578])
    phic_sol_expected = np.array([28.6616,29.0152])*MW

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected))

    rel_tol = 1e-3
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def comp_conv_scaling():
    """compare the convergence of NR using different forms of scaling. Compare with NR using Finite-Difference Jacobian."""
    het_net, gas_net, elec_net, heat_net, water, gas = create_network()
    formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}

    # Scaling
    # per unit base values
    rho_w = water.rhon #[kg/m^3]
    grav_const = water.g #[m/s^2]
    Ta = heat_net.Ta
    rhon_g = gas.rhon #[kg/m^3]
    print('rho_g = {}'.format(rhon_g))
    mbase_pu = 1.
    pbase_pu = 1.*bar
    Tbase_pu = 130.
    Sbase_pu = 10.*Sbase_shabanpour #since P and Q are in p.u. (i.e. given in MW based on p.u. delta, V and line parameters)
    phibase_pu = Sbase_pu
    Vbase_pu = 1.*Vbase_shabanpour
    deltabase_pu = 1.
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase_pu,'pbase':pbase_pu,'phibase':phibase_pu,'Tbase':Tbase_pu,'Vbase':Vbase_pu,'deltabase':deltabase_pu,'Sbase':Sbase_pu,'Ebase':phibase_pu}
    # base values used for scaling in solver
    qbase = mbase_pu#10*MSCM*rhon_g/hour
    pgbase = pbase_pu#30*bar
    mbase = mbase_pu#50.
    phbase = pbase_pu#1000.*rho_w*grav_const
    Tbase = 130.
    Sbase = Sbase_pu#10. #since P and Q are in p.u. (i.e. given in MW based on p.u. delta, V and line parameters)
    phibase = Sbase
    Egbase = phibase
    Vbase = 1.*Vbase_shabanpour
    deltabase = 1.


    # initalize network, unscaled
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation)

    # check initial Jacobian
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation)
    h=1e-6
    print('maximum difference between J0 and J0_FD = {}'.format(nlsys.compare_J(x0,h)))
    J0 = nlsys.J(x0)
    J0_FD = nlsys.J_FD(x0,h)
    J_diff = J0-J0_FD
    fig_J = plt.figure('Jacobian difference between unscaled J0 and J0_FD')
    plt.imshow(J_diff)
    ax_J = plt.gca()
    nlsys.plot_J_overlay(ax_J)
    plt.colorbar()

    # initalize network, per unit
    het_net.reset_network(x0,formulation=formulation)
    x0_pu = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,scale_var=scale_var,scale_var_params=scale_var_params)

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
    x_sol_scaled,iters_scaled,err_vec_scaled = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
    # solve when everything is specified in S.I., using scaling in solver, using FD
    het_net.reset_network(x0,formulation=formulation)
    x_sol_scaled_FD,iters_scaled_FD,err_vec_scaled_FD = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase},solver='NR_FD')

    fig = plt.figure('Convergence plot different scaling')
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

def comp_conv_form():
    """Compare the convergence of NR between MES and single-carrier networks, using different formulation in the singl-carrier (part of the) network."""
    # plot convergence
    fig_conv_form_het, ax_conv_form = plt.subplots(2, 2, num='Convergence plot networks')
    ax_conv_form_het = ax_conv_form[0,0]
    ax_conv_form_gas = ax_conv_form[0,1]
    ax_conv_form_elec = ax_conv_form[1,0]
    ax_conv_form_heat = ax_conv_form[1,1]
    ax_conv_form_het.set_title('Heterogeneous network')
    ax_conv_form_gas.set_title('Gas network')
    ax_conv_form_elec.set_title('Electrical network')
    ax_conv_form_heat.set_title('Heat network')

    max_iters_used = 0
    linestyles = {'fa':'--','fb':'-'}
    colors = {'sep. coup.':'tab:blue','int. coup.':'tab:orange'}
    markers_heat = {'standard outflow':'.','standard delta':'*','half_link_flow outflow':'d','half_link_flow delta':'x'}

    # load flow
    hydr_eqs = ['fb','fa']
    forms_heat = ['standard','half_link_flow']
    heat_loads = ['outflow','delta']
    max_iter = 15
    # run load flow for SC
    gas = GasNet.gas
    rhon_g = gas.rhon
    for c_hl in [True,False]:
        for hydr_eq in hydr_eqs:
            print('\nHydraulic equation is {}, and separate couplings is {}'.format(hydr_eq,c_hl))
            # load flow (only full is used, since the pipes use a implicit friction factor.
            gas_net_single,x_sol_gas,iters_gas,err_vec_gas,p_sol_single,q_sol_single,q_inj_single,tol_gas = GasNet.run_load_flow(hydr_eq=hydr_eq,c_hl=c_hl,max_iter=max_iter)
            print('Solution:')
            print('p = {} bar'.format(p_sol_single/bar))
            print('q = {} MSCM/hour'.format(q_sol_single/(MSCM*rhon_g/hour)))
            print('q nodal inj = {} MSCM/hour'.format(q_inj_single/(MSCM*rhon_g/hour)))
            print('q hl = {} MSCM/hour'.format([hl.q/(MSCM*rhon_g/hour) for node in gas_net_single.get_nodes() for hl in node.get_half_links()]))
            if c_hl:
                key = 'sep. coup.'
            else:
                key = 'int. coup.'
            print('key = {}'.format(key))
            ax_conv_form_gas.semilogy(err_vec_gas,ls=linestyles.get(hydr_eq),color=colors.get(key),marker='.',label=key+' '+hydr_eq)
            max_iters_used = max(max_iters_used,iters_gas)
    for c_hl in [True,False]:
        elec_net_single,x_sol_elec,iters_elec,err_vec_elec,delta_sol_single,V_sol_single,S_inj_single,P_edge_single,Q_edge_single,tol_elec = ElecNet.run_load_flow(c_hl=c_hl)
        print('\nSeparate couplings used: {}'.format(c_hl))
        print('Solution:')
        print('delta = {}'.format(delta_sol_single))
        print('|V| = {} p.u.'.format(V_sol_single))
        print('P edge = {} p.u.'.format(P_edge_single))
        print('Q edge = {} p.u.'.format(Q_edge_single))
        print('S nodal inj = {} p.u.'.format(S_inj_single))
        print('P hl = {} p.u.'.format([hl.P for node in elec_net_single.get_nodes() for hl in node.get_half_links()]))
        print('Q hl = {} p.u.'.format([hl.Q for node in elec_net_single.get_nodes() for hl in node.get_half_links()]))
        if c_hl:
            key = 'sep. coup.'
        else:
            key = 'int. coup.'
        ax_conv_form_elec.semilogy(err_vec_elec,ls='-',color=colors.get(key),marker='.',label=key)
        max_iters_used = max(max_iters_used,iters_elec)
    rho_w = HeatNet.rho_w
    grav_const = HeatNet.grav_const
    for c_hl in [True,False]:
        for form in forms_heat:
            for heat_load in heat_loads:
                print('\nFormulation is {}, and heat load is {}, and seperate couplings is {}'.format(form,heat_load,c_hl))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                    heat_net_single,x_sol_heat,iters_heat,err_vec_heat,m_vec_single,p_vec_single,Ts_vec_single,Tr_vec_single,m_hl_vec_single,phi_hl_vec_single,Ts_hl_vec_single,Tr_hl_vec_single,tol_heat = HeatNet.run_load_flow(c_hl=c_hl,heat_load=heat_load,formulation=form,max_iter=max_iter)
                print('Solution:')
                print('p heat = {} m'.format(p_vec_single/(rho_w*grav_const)))
                print('m = {}'.format(m_vec_single))
                print('Ts = {}'.format(Ts_vec_single))
                print('Tr = {}'.format(Tr_vec_single))
                print('m hl = {}'.format(m_hl_vec_single))
                print('Ts hl = {}'.format(Ts_hl_vec_single))
                print('Tr hl = {}'.format(Tr_hl_vec_single))
                print('phi hl = {}'.format(phi_hl_vec_single))
                label = '{} {}'.format(form,heat_load)
                # plot convergence
                if c_hl:
                    key = 'sep. coup.'
                else:
                    key = 'int. coup.'
                ax_conv_form_heat.semilogy(err_vec_heat,ls='-',color=colors.get(key),marker=markers_heat.get(label),label=key+', '+label)
                max_iters_used = max(max_iters_used,iters_heat)
    # run load flow for MES
    h = 1e-6
    tol = tol_heat

    # Scaling
    qbase = 1.#10*MSCM*rhon_g/hour
    pgbase = 1.*bar#30*bar
    mbase = 1.#50.
    phbase = 1.*bar#1000.*rho_w*grav_const
    Tbase = 130.
    Sbase = 10.*Sbase_shabanpour #since P and Q are in p.u. (i.e. given in MW based on p.u. delta, V and line parameters)
    phibase = Sbase
    Egbase = phibase
    Vbase = 1.*Vbase_shabanpour
    deltabase = 1.
    for hydr_eq in hydr_eqs:
        for form_heat in forms_heat:
            for heat_load in heat_loads:
                formulation={'gas':'full','elec':'complex_power','heat':form_heat,'het':None}
                het_net, gas_net, elec_net, heat_net, water, gas = create_network(hydr_eq=hydr_eq,heat_load=heat_load)
                # initalize network, unscaled
                x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,heat_load=heat_load)
                # solve
                het_net.reset_network(x0,formulation=formulation)
                x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
                key_heat = '{} {}'.format(form_heat,heat_load)
                ax_conv_form_het.semilogy(err_vec,ls=linestyles.get(hydr_eq),color='tab:blue',marker=markers_heat.get(key_heat),label=hydr_eq+', '+key_heat)

    # plot layout
    xmin = 0
    xmax = max_iters_used
    xticks = range(xmin,xmax+1) # make sure the xticks are integers
    ax_conv_form_gas.semilogy([0,max_iters_used+1],[tol_gas,tol_gas],'k:',label='tolerance')
    ax_conv_form_elec.semilogy([0,max_iters_used+1],[tol_elec,tol_elec],'k:',label='tolerance')
    ax_conv_form_heat.semilogy([0,max_iters_used+1],[tol_heat,tol_heat],'k:',label='tolerance')
    ax_conv_form_het.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
    for ax_rows in ax_conv_form:
        for ax in ax_rows:
                ax.set_xlabel(r'Iteration $k$')
                ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
                ax.grid(which='major',color='k', linestyle='--', alpha=.2)
                ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
                ax.legend()
                ax.set_xlim(left=xmin,right=xmax+1)
                ax.set_xticks(xticks)

def comp_solver_time_reod(hydr_eq='fa',heat_load='outflow',formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},max_iter=10,maxmatvecs=5000,tol=1e-6):
    """Compare the time spent in the linear solver and the total time spent for different orderings and different linear solvers"""
    # create plot and lay-out
    fig = plt.figure('Convergence plot different solvers and permutations')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||T_F F(x^k)||_2$')
    colors_perm = {'no':'tab:blue','x':'tab:red','xF':'tab:green'}
    markers_solver = {'solve':'.','gmres':'*','bicgstab':'x'}

    # create network
    het_net, gas_net, elec_net, heat_net, water, gas = create_network(hydr_eq=hydr_eq,heat_load=heat_load)
    # initialize
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,heat_load=heat_load)

    # solver information
    lin_solvers = ['solve','gmres','bicgstab']
    rhon_g = gas.rhon
    rho_w = water.rhon
    grav_const = water.g
    pgbase=50*bar
    qbase=10*MSCM*rhon_g/hour
    Vbase=10/np.sqrt(3)*kV
    Sbase=10*MW
    phbase=1000.*rho_w*grav_const
    mbase=50
    Tbase=100
    phibase=10*MW
    Egbase=10*MW
    deltabase=1
    scale_var = 'matrix'
    scale_var_params = {'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}
    max_iters_used = 0

    # permutation matrices
    P_x_re_x, P_F_re_x = perm_matr_x(het_net,formulation=formulation)
    P_x_re_xF, P_F_re_xF = perm_matr_xF(het_net,formulation=formulation)
    perm_x = [np.array([]),P_x_re_x,P_x_re_xF]
    perm_F = [np.array([]),P_F_re_x,P_F_re_xF]
    perm_keys = ['no','x','xF']

    # load flow (for different orderings and using different linear solvers, using default scaling values)
    for ind_P,perm_key in enumerate(perm_keys):
        P_x = perm_x[ind_P]
        P_F = perm_F[ind_P]
        for lin_solver in lin_solvers:
            het_net.reset_network(x0,formulation=formulation)
            print('\nSolving system for {} perm, with {} as lin solver'.format(perm_key,lin_solver))
            x_sol,iters,err_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,P_F=P_F,P_x=P_x,lin_solver=lin_solver,max_iter_lin=maxmatvecs)
            ax.semilogy(err_vec,ls='-',color=colors_perm.get(perm_key),marker=markers_solver.get(lin_solver),label=lin_solver+', '+perm_key)
            max_iters_used = max(max_iters_used,iters)
            print('Final error is {:6.3e} after {} iterations'.format(err_vec[-1],iters))

    # plot layout
    xmin = 0
    xmax = max_iters_used
    xticks = range(xmin,xmax+1) # make sure the xticks are integers
    ax.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

if __name__ == '__main__':
    comp_conv_scaling()
    comp_conv_form()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        jacobians()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        comp_solver_time_reod(tol=1e-5)

    plt.show()
