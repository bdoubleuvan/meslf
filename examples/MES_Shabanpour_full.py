"""Example of a MES with 3 nodes per carrier. Based on the example in Shabanpourt-Haghighi and Seifi. With all conversion units."""
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

def create_network(hydr_eq='fa',heat_load='outflow'):
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
    g0 = GasNode('gn0',node_type=0,x=5,y=12,p=50.*bar) # reference node
    g1 = GasNode('gn1',node_type=1,x=0,y=7,q=10.8649*MSCM*rhon_g/hour) # load node
    g2 = GasNode('gn2',node_type=1,x=5,y=2,q=20*MSCM*rhon_g/hour) # load node
    g3 = GasNode('gn3',node_type=1,x=2.5,y=4.5,q=0.) # load node
    e0 = ElectricalNode('en0',node_type=5,x=7,y=12,P=0.1451*Sbase_shabanpour,Q=0.*Sbase_shabanpour,V=1.06*Vbase_shabanpour,delta=0.) # PQVdelta node
    e1 = ElectricalNode('en1',node_type=2,x=2,y=7,P=30.*Sbase_shabanpour,Q=15.*Sbase_shabanpour) # load
    e2 = ElectricalNode('en2',node_type=6,x=7,y=2,P=30.136*Sbase_shabanpour,Q=15.*Sbase_shabanpour,V=1.*Vbase_shabanpour) # PQV
    h0 = HeatNode('hn0',node_type=5,x=9,y=12,p=5517.*rho_w*grav_const) # reference  node
    Tr2 = 49.5339
    if heat_load == 'outflow':
        h1 = HeatNode('hn1',node_type=1,x=4,y=7,Tr_hl=50.,dphi=35.*MW) # load node (sink)
        h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
        h2 = HeatNode('hn2',node_type=1,x=9,y=2,Tr_hl=50.,dphi=20.*MW) # load node (sink)
        h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif heat_load == 'delta':
        h1 = HeatNode('hn1',node_type=12,x=4,y=7,dT=69.0370,dphi=35.*MW) # load node (sink), temp. diff.
        h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
        h2 = HeatNode('hn2',node_type=12,x=9,y=2,dT=73.5410,dphi=20.*MW) # load node (sink), temp. diff
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
    GHV = 40.611 #[MBTU/m^3]
    GHV *= MBTU*BTU #[Wh/m^3]
    GHV *= hour/rhon_g #[J/kg]
    print('GHV = {} J/kg'.format(GHV))
    c0 = HeterogeneousNode('cn0',node_type=0,x=6,y=13,unit_type='ge_gas_fired_gen_valve_point',unit_params={'a':a_GG,'b':b_GG,'c':c_GG,'d':d_GG,'e':e_GG,'Pmin':Pmin,'GHV':GHV})

    Cp = water.Cp
    To_c_CHP = 130 #[C]
    To_c_GB0 = 120 #[C]
    To_c_GB2 = 110 #[C]
    dphi_c_CHP = -25*MW
    dphi_c_GB2 = -4*MW
    m_CHP = -dphi_c_CHP/(Cp*(To_c_CHP-Tr2))
    m_GB = -dphi_c_GB2/(Cp*(To_c_GB2-Tr2))
    Tr0 = 48.6805
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
    if heat_load == 'outflow':
        c1 = HeterogeneousNode('cn1',node_type=1,x=8,y=13,unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
        c2 = HeterogeneousNode('cn2',node_type=1,x=6,y=0,unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To known
        c3 = HeterogeneousNode('cn3',node_type=1,x=8,y=0,unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
    elif heat_load == 'delta':
        c1 = HeterogeneousNode('cn1',node_type=2,x=8,y=13,unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
        c2 = HeterogeneousNode('cn2',node_type=2,x=6,y=0,unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To known
        c3 = HeterogeneousNode('cn3',node_type=2,x=8,y=0,unit_type=unit_type_GB,unit_params=unit_params_GB) # To known

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
    gl7 = GasLink('gl7',g2,c3)
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
        hl3 = HeatLink('hl3',c1,h0,link_params={'carrier':water},bc_type=6,Tsstart=To_c_GB0) # To of coupling (source) is known
        hl4 = HeatLink('hl4',c2,h2,link_params={'carrier':water},bc_type=2,Tsstart=To_c_CHP,dphistart=dphi_c_CHP) # To of coupling (source) is known, and dphi is known
        hl5 = HeatLink('hl5',c3,h2,link_params={'carrier':water},bc_type=2,Tsstart=To_c_GB2,dphistart=dphi_c_GB2) # To of coupling (source) is known, and dphi is known
    elif heat_load == 'delta':
        hl3 = HeatLink('hl3',c1,h0,link_params={'carrier':water},bc_type=10,dTstart=To_c_GB0-Tr0) # dT of coupling (source) is known
        hl4 = HeatLink('hl4',c2,h2,link_params={'carrier':water},bc_type=4,dTstart=To_c_CHP-Tr2,dphistart=dphi_c_CHP) # dT of coupling (source) is known, and dphi is known
        hl5 = HeatLink('hl5',c3,h2,link_params={'carrier':water},bc_type=4,dTstart=To_c_GB2-Tr2,dphistart=dphi_c_GB2) # dT of coupling (source) is known, and dphi is known
    # network
    gas_net = GasNetwork('test gas network')
    gas_net.add_link(gl0)
    gas_net.add_link(gl1)
    gas_net.add_link(gl2)
    gas_net.add_link(gl3)
    gas_net.add_link(gl4)
    gas_net.add_link(gl5)
    gas_net.add_link(gl6)
    gas_net.add_link(gl7)

    elec_net = ElectricalNetwork('test electrical network')
    elec_net.add_link(el0)
    elec_net.add_link(el1)
    elec_net.add_link(el2)
    elec_net.add_link(el3)
    elec_net.add_link(el4)

    heat_net = HeatNetwork('test heat network',Ta=Ta)
    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    heat_net.add_link(hl3)
    heat_net.add_link(hl4)
    heat_net.add_link(hl5)

    het_net = HeterogeneousNetwork('test heterogeneous network')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    het_net.add_node(c0)
    het_net.add_node(c1)
    het_net.add_node(c2)
    het_net.add_node(c3)

    return gas_net, elec_net, heat_net, het_net

def initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,heat_load='outflow'):
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
    gp_init = np.array([25.,30.,40.])*bar
    q_init = np.array([20.,20.,20.,20.])*MSCM*rhon_g/hour
    delta_init = np.array([0.,0.]) # default values
    V_init = np.array([1.])*Vbase_shabanpour # default values
    m_init = np.array([60.,30.,60.])
    if formulation['heat'] == 'half_link_flow':
        m_hl_init = np.array([20.,20.])
        if heat_load == 'delta':
            To_hl_init = np.array([50.,50.]) #[C]
    hp_init = np.array([10.,4000.])*rho_w*grav_const
    Ts_init = np.array([120.,120.,120.])
    Tr_init = np.array([50.,50.,50.])
    qc_init = np.array([10,3,3,3])*MSCM*rhon_g/hour
    Pc_init = np.array([50.,10.])*Sbase_shabanpour
    Qc_init = np.array([0.,0.])*Sbase_shabanpour
    Sc_init = np.concatenate((Pc_init,Qc_init))
    mc_init = np.array([10.,10.,10.])
    phic_init = np.array([30.])*MW
    Toc_init = np.array([100.,130.,110.])

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
    x0 = het_net.set_x_init(formulation=formulation)
    return x0

def run_load_flow(hydr_eq='fa',heat_load='outflow',pgbase=50*bar,qbase=10*MSCM*rhon_g/hour,Vbase=10/np.sqrt(3)*kV,Sbase=10*MW,phbase=1000.*rho_w*grav_const,mbase=50,Tbase=100,phibase=10*MW,Egbase=10*MW,deltabase=1,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},P_F=np.array([]),P_x=np.array([]),tol=1e-6,max_iter=50,plot_top=False,plot_jac=False):
    """Stead-state load flow analysis of heat network

    Parameters
    ----------
    phbase : float
        Base value used for pressure.
    mbase : float
        Base value used for link flow.
    Tbase : float
        Base value for temperature.
    phibase : float
        Base value for heat power.


    """
    # create network
    gas_net,elec_net,heat_net,het_net = create_network(hydr_eq=hydr_eq,heat_load=heat_load)

    # initialize
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,heat_load=heat_load)
    for node in het_net.get_nodes():
        if isinstance(node,HeterogeneousNode):
            print('node {} with type {}'.format(node.name,node.unit_type))
            if node.half_links:
                print('phi = {}, m = {}, To = {}'.format(node.half_links[0].phi, node.half_links[0].m, node.half_links[0].To))

    if plot_jac:
        from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
        nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
        F0 = nlsys.F(x0)
        J0 = nlsys.J(x0)
        D_x = nlsys.Dx()
        D_x_inv = sps.diags(1/D_x.data[0])
        D_F = nlsys.DF()
        J_scaled = D_F.dot(J0.dot(D_x_inv))
        print('x0: {}, Dx: {}, DF: {}'.format(len(x0),D_x.shape,D_F.shape))
        print('|J0|={}, |D_F J0 D_x_inv|={}'.format(np.linalg.det(J0.todense()),np.linalg.det(J_scaled.todense())))
        #print('Dx = {}'.format(D_x))
        #print('DF = {}'.format(D_F))
        #spy plot original
        fig_J = nlsys.spy_plot_J(x0,title='Jacobian spy plot')
        ax_J = plt.gca()
        fig_J_map = nlsys.imshow_J(x0,title=r'Jacobian')

    # solve network
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase},P_F=P_F,P_x=P_x)

    print('Solution after {} it. (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print('p = {} bar'.format(p_g_vec/bar))
    print('q = {} MSCM/hour'.format(q_vec/(MSCM*rhon_g/hour)))
    print('q nodal inj = {} MSCM/hour'.format(q_inj/(MSCM*rhon_g/hour)))
    print('delta = {} degree'.format(delta_vec/degree))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase_shabanpour))
    print('P edge = {} p.u.'.format(P_edge/Sbase_shabanpour))
    print('Q edge = {} p.u.'.format(Q_edge/Sbase_shabanpour))
    print('S nodal inj = {} p.u.'.format(S_inj/Sbase_shabanpour))
    print('P hl = {} p.u.'.format([hl.P/Sbase_shabanpour for node in het_net.get_nodes() if isinstance(node,ElectricalNode) for hl in node.get_half_links()]))
    print('Q hl = {} p.u.'.format([hl.Q/Sbase_shabanpour for node in het_net.get_nodes() if isinstance(node,ElectricalNode) for hl in node.get_half_links()]))
    print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Ts hl = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Tr hl = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Ts hl c = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('Tr hl c = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('dphi hl c = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('m hl c = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    if plot_top:
        # plot topology
        fig_top = plt.figure('Network topology')
        ax_top = fig_top.gca()
        het_net.draw_network(ax_top,halflink_angle=2,halflink_length=1)
        plt.axis('equal')
        plt.axis('off')

    return gas_net,elec_net,heat_net,het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol

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

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_full_known_phi():
    # Given + When
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        gas_net,elec_net,heat_net,het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow()

    # Then
    p_g_sol_expected = np.array([29.1021,34.077,37.83273])*bar
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-7.0219,-6.1156])*degree
    V_sol_expected = np.array([0.9800])*Vbase_shabanpour
    m_sol_expected = np.array([64.6943,31.4533,-56.5335])
    p_h_sol_expected = np.array([223.05247021, 4264.91754827])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 119.0370, 123.5410])
    Tr0 = 48.6805
    Tr2 = 49.5339
    Tr_sol_expected = np.array([Tr0,50.,Tr2])
    qc_sol_expected = np.array([9.4501, 3.0177, 3.3581,.4175])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.8662, 10.1730])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3630, 10.2181])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    phi_GB0 = 28.6768 #[MW]
    mc_sol_expected = np.array([phi_GB0*MW/(Cp_w*(120-Tr0)), 25*MW/(Cp_w*(130-Tr2)), 4*MW/(Cp_w*(110-Tr2))])
    phic_sol_expected = np.array([28.6768])*MW

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected))

    rel_tol = 1e-3 # P_CHP is of by .004 MW (see also test_coupling_nodes)
    print('solution: {}'.format(x_sol))
    print('difference between solution: {}'.format(x_sol-x_sol_expected))
    print('maximum difference: {}'.format(np.max(np.abs(x_sol-x_sol_expected))))
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_shabanpour_full_known_phi_permuted():
    # Given + When
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        # permutation
        gas_net,elec_net,heat_net,het_net = create_network() # use defaults
        P_x, P_F = perm_matr_xF(het_net)
        # run load flow (creates network again, uses scaling by default)
        gas_net,elec_net,heat_net,het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(P_x=P_x,P_F=P_F)

    # Then
    p_g_sol_expected = np.array([29.1021,34.077,37.83273])*bar
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour
    delta_sol_expected = np.array([-7.0219,-6.1156])*degree
    V_sol_expected = np.array([0.9800])*Vbase_shabanpour
    m_sol_expected = np.array([64.6943,31.4533,-56.5335])
    p_h_sol_expected = np.array([223.05247021, 4264.91754827])*rho_w*grav_const
    Ts_sol_expected = np.array([120., 119.0370, 123.5410])
    Tr0 = 48.6805
    Tr2 = 49.5339
    Tr_sol_expected = np.array([Tr0,50.,Tr2])
    qc_sol_expected = np.array([9.4501, 3.0177, 3.3581,.4175])*MSCM*rhon_g/hour
    Pc_sol_expected = np.array([50.8662, 10.1730])*Sbase_shabanpour
    Qc_sol_expected = np.array([27.3630, 10.2181])*Sbase_shabanpour
    Sc_sol_expected = np.concatenate((Pc_sol_expected,Qc_sol_expected))
    phi_GB0 = 28.6768 #[MW]
    mc_sol_expected = np.array([phi_GB0*MW/(Cp_w*(120-Tr0)), 25*MW/(Cp_w*(130-Tr2)), 4*MW/(Cp_w*(110-Tr2))])
    phic_sol_expected = np.array([28.6768])*MW

    xg_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    xe_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))
    xh_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))
    x_sol_expected = np.concatenate((xg_sol_expected,xe_sol_expected,xh_sol_expected,qc_sol_expected,Sc_sol_expected,mc_sol_expected,phic_sol_expected))

    rel_tol = 1e-3 # P_CHP is of by .004 MW (see also test_coupling_nodes)
    print('solution: {}'.format(x_sol))
    print('difference between solution: {}'.format(x_sol-x_sol_expected))
    print('maximum difference: {}'.format(np.max(np.abs(x_sol-x_sol_expected))))
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def jacobians(hydr_eq='fa',heat_load='outflow',pgbase=50*bar,qbase=10*MSCM*rhon_g/hour,Vbase=10/np.sqrt(3)*kV,Sbase=10*MW,phbase=1000.*rho_w*grav_const,mbase=50,Tbase=100,phibase=10*MW,Egbase=10*MW,deltabase=1,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}):
    """Plot Jacobian matrices, and eigenvalue spectra, for different indices / ordering"""

    # create network
    gas_net,elec_net,heat_net,het_net = create_network(hydr_eq=hydr_eq,heat_load=heat_load)

    # initialize
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,heat_load=heat_load)

    # create system of equations
    from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
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

def comp_conv_form():
    """Compare convergence using different formulations"""
    # create plot and lay-out
    fig = plt.figure('Convergence plot different formulations')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||D_F F(x^k)||_2$')

    fig_ord = plt.figure('Convergence order different formulations')
    ax_ord = fig_ord.gca()
    plt.xlabel(r'$||D_F F(x^k)||_2 / ||D_F F(x^0)||_2$')
    plt.ylabel(r'$||D_F F(x^{k+1})||_2 / ||D_F F(x^0)||_2$')

    max_iters_used = 0
    linestyles = {'fa':'--','fb':'-'}
    markers_heat = {'standard outflow':'.','standard delta':'*','half_link_flow outflow':'d','half_link_flow delta':'x'}

    # load flow (nodal in gas is not possible, since the gas network has a compressor)
    hydr_eqs = ['fb','fa']
    forms_heat = ['standard','half_link_flow']
    heat_loads = ['outflow','delta']
    max_iter = 15
    for hydr_eq in hydr_eqs:
        for form_heat in forms_heat:
            formulation={'gas':'full','elec':'complex_power','heat':form_heat,'het':None}
            for heat_load in heat_loads:
                # is scaled with default base values
                gas_net,elec_net,heat_net,het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(hydr_eq=hydr_eq,heat_load=heat_load,formulation=formulation)
                key_heat = '{} {}'.format(form_heat,heat_load)
                ax.semilogy(err_vec,ls=linestyles.get(hydr_eq),color='tab:blue',marker=markers_heat.get(key_heat),label=hydr_eq+', '+key_heat)
                ax_ord.loglog(err_vec[:-1]/err_vec[0],err_vec[1:]/err_vec[0],ls=linestyles.get(hydr_eq),color='tab:blue',marker=markers_heat.get(key_heat),label=hydr_eq+', '+key_heat)
                max_iters_used = max(max_iters_used,iters)

    # plot layout
    xmin = 0
    xmax = max_iters_used
    xticks = range(xmin,xmax+1) # make sure the xticks are integers
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

    x_min_ord,x_max_ord = ax_ord.get_xlim()
    x_slope = np.linspace(x_min_ord,x_max_ord)
    y_slope2 = x_slope**2
    ax_ord.loglog(x_slope,x_slope,linestyle='--',color='k',label='slope 1')
    ax_ord.loglog(x_slope,y_slope2,linestyle='-.',color='k',label='slope 2')
    ax_ord.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_ord.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_ord.legend()

def comp_solver_time_reod(hydr_eq='fa',heat_load='outflow',pgbase=50*bar,qbase=10*MSCM*rhon_g/hour,Vbase=10/np.sqrt(3)*kV,Sbase=10*MW,phbase=1000.*rho_w*grav_const,mbase=50,Tbase=100,phibase=10*MW,Egbase=10*MW,deltabase=1,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},max_iter=10,maxmatvecs=5000,tol=1e-6):
    """Compare the time spent in the linear solver and the total time spent for different orderings and different linear solvers"""
    # solver information
    lin_solvers = ['solve','gmres','bicgstab']
    scale_var = 'matrix'
    scale_var_params = {'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}
    max_iters_used = 0

    # create plot and lay-out
    fig = plt.figure('Convergence plot different solvers and permutations')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||T_F F(x^k)||_2$')
    colors_perm = {'no':'tab:blue','x':'tab:red','xF':'tab:green'}
    markers_solver = {'solve':'.','gmres':'*','bicgstab':'x'}

    # create network
    gas_net,elec_net,heat_net,het_net = create_network(hydr_eq=hydr_eq,heat_load=heat_load)
    # initialize
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,heat_load=heat_load)

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
            print('Checking sign on heat dummy links')
            for link in het_net.get_links(link_types=['dummy']):
                if isinstance(link,HeatLink):
                    print('For link {}'.format(link.name))
                    print('dphistart = {}'.format(link.get_dphistart()))
                    dphi = link.supply_heat_power_start()+link.return_heat_power_start()
                    print('phis + phir func = {}'.format(dphi))
                    print('(phis + phir) - dphistart = {}'.format(dphi - link.get_dphistart()))
                    print('|phis + phir| - |dphistart| = {}'.format(abs(dphi) - abs(link.get_dphistart())))
                    print('phis start func = {}'.format(link.supply_heat_power_start()))
                    print('phis start attr = {}'.format(link.get_phisstart()))
                    print('phir start func = {}'.format(link.return_heat_power_start()))
                    print('phir start attr = {}'.format(link.get_phirstart()))


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        comp_conv_form()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        jacobians()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        comp_solver_time_reod()

    plt.show()
