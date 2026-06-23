"""Example of a MES with 3 nodes per carrier. Based on the example in Shabanpourt-Haghighi and Seifi. Uses energy hubs for the coupling nodes. Use Domain Decomposition to solve the load flow problem"""
from meslf.networks.gas_network import GasHalfLink, GasNetwork
from meslf.networks.electrical_network import ElectricalHalfLink, ElectricalNetwork
from meslf.networks.heat_network import HeatHalfLink, HeatNetwork
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from examples import GN_Shabanpour as GN
from examples import EN_Shabanpour as EN
from examples import HN_Shabanpour as HN
from examples import MES_Shabanpour_EH as MES
from meslf.utils.constants import mm, bar, hour, kV, MW, MSCM, BTU, MBTU, degree
from meslf.load_flow.system_of_equations import NonLinearSystemGas, NonLinearSystemElectrical, NonLinearSystemHeat, NonLinearSystemHeterogeneous
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pytest
from meslf.utils.hide_print import HiddenPrints
import os

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

# get data from mes
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "Only a",UserWarning)
    het_net_mes, gas_net_mes, elec_net_mes, heat_net_mes, water, gas  = MES.create_network()
rhon_g = gas.rhon
rho_w = water.rhon
grav_const = water.g
formulation_mes={'gas':'full','elec':'complex_power','heat':'standard','het':None}
x0_mes = MES.initialize_network(het_net_mes,gas_net_mes,elec_net_mes,heat_net_mes,water,gas,formulation_mes)
het_net_mes.update(x0_mes,formulation=formulation_mes)
x_sol_mes,iters_mes,err_vec_mes,p_g_vec_mes,q_vec_mes,q_inj_mes,delta_vec_mes,V_mag_vec_mes,S_inj_mes,P_edge_mes,Q_edge_mes,m_vec_mes,p_h_vec_mes,Ts_vec_mes,Tr_vec_mes,m_hl_vec_mes,phi_hl_vec_mes,Ts_hl_vec_mes,Tr_hl_vec_mes,qc_vec_mes,Pc_vec_mes,Qc_vec_mes,mc_vec_mes,phic_vec_mes,Tsc_vec_mes,Trc_vec_mes = het_net_mes.solve_network(1e-6,10,solver='NR',formulation=formulation_mes)
print('Solving MES: error is {} after {} iterations'.format(err_vec_mes[-1],iters_mes))
P0_load = elec_net_mes.nodes[0].half_links[0].get_P() #power to gas compressor?
Q0_load = elec_net_mes.nodes[0].half_links[0].get_Q()
# solution in SC parts
ph2 = p_h_vec_mes[-1]
# solution in coupling part
qc0_sol = qc_vec_mes[0]
qc1_sol = qc_vec_mes[1]
Pc0_sol = Pc_vec_mes[0]
Pc1_sol = Pc_vec_mes[1]
Qc0_sol = Qc_vec_mes[0]
Qc1_sol = Qc_vec_mes[1]
Trc0_sol = Trc_vec_mes[0]
Tsc0_sol = Tsc_vec_mes[0]
mc0_sol = mc_vec_mes[0]
dphic0_sol = phic_vec_mes[0]
dphic1_sol = phic_vec_mes[1]
Trc1_sol = Trc_vec_mes[1]
Tsc1_sol = Tsc_vec_mes[1]
mc1_sol = mc_vec_mes[1]

unit_params_EH0 = het_net_mes.nodes[-2].unit_params.copy()
unit_params_EH1 = het_net_mes.nodes[-1].unit_params.copy()

# base values
pgbase=50*bar #[Pa]
qbase=10*MSCM*rhon_g/hour #[kg/s]
Sbase=100*MW #[W]
Vbase= 10/np.sqrt(3)*kV #[V]
deltabase = 1
mbase = 1
phbase = 1*bar
Tbase = 130.
phibase = 1*MW

def create_gas_network(hydr_eq='fa'):
    """Create the decoupled single-carrier gas network."""
    gas_net = GN.create_network(hydr_eq=hydr_eq,c_hl=True,pipe_type='pipe_high_pres_colebrook',node_set=2)
    # merge the two half links representing flow to GG and GB into one flow going to EH
    g0 = gas_net.nodes[0]
    g0.half_links[1].q = qc0_sol
    g0_hl2 = g0.half_links[2]
    g0.remove_half_link(g0_hl2)
    gas_net.remove_half_link(g0_hl2)

    return gas_net

def update_gas_network_bc(gas_net,xc):
    """Update the boundary values of the gas network, based on the outcome xc of the coupling gas-electricity-heat network"""
    qc0 = xc[0] # >0
    gas_net.nodes[0].half_links[1].q = qc0 # flow to coupling is outflow for the gas network
    return gas_net

def initialize_gas_network(gas_net,pg0_ic=55*bar,pg1_ic=45*bar,pg3_ic=45*bar,q_ic=20*MSCM*rhon_g/hour,formulation='full'):
    """Initialize the gas network"""
    if formulation == 'full':
        q_init = q_ic*np.ones(len(gas_net.links))
    else:
        q_init = np.array([])
    p_init = np.array([pg0_ic,pg1_ic,pg3_ic]) #[Pa]
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    gas_net.reset_network(x_init,formulation=formulation)
    x0 = gas_net.set_x_init(formulation=formulation)
    return x0

def run_gas_load_flow(hydr_eq='fa',pg0_ic=55*bar,pg1_ic=45*bar,pg3_ic=45*bar,q_ic=20*MSCM*rhon_g/hour,tol=1e-6,max_iter=10,scale_var=None,scale_var_params={'qbase':qbase,'pgbase':pgbase},formulation='full'):
    """Steady-state load flow analysis of gas network.
    """
    # create network
    gas_net = create_gas_network(hydr_eq=hydr_eq)
    # initialize
    x0 = initialize_gas_network(gas_net,pg0_ic=pg0_ic,pg1_ic=pg1_ic,pg3_ic=pg3_ic,q_ic=q_ic,formulation=formulation)
    # solve network
    print('\nRunning load flow for single-carrier gas network')
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution after {} iterations (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print('x = {}'.format(x_sol))
    print_results_SC(gas_net,x_sol,formulation=formulation)
    return gas_net,x_sol,iters,err_vec

def create_electrical_network():
    """Create the decoupled single-carrier electricity network."""
    elec_net = EN.create_network(c_hl=True,node_set=2,values='S.I.')
    return elec_net

def update_electrical_network_bc(elec_net,xc):
    """Update the boundary values of the electricity network, based on the outcome xc of the coupling electricity-heat network"""
    Pc1 = xc[1] # >0
    elec_net.nodes[2].half_links[0].P = -Pc1 #generator, so source, so <0

    return elec_net

def initialize_electrical_network(elec_net):
    x0 = EN.initialize_network(elec_net,node_set=2,values='S.I.')
    return x0

def run_electrical_load_flow(tol=1e-6,max_iter=10,scale_var=None,scale_var_params={'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase}):
    """Steady-state load flow analysis of electrical network.
    """
    # create network
    elec_net = create_electrical_network()
    # initialize
    x0 = initialize_electrical_network(elec_net)
    # solve network
    print('\nRunning load flow for single-carrier electrical network')
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print_results_SC(elec_net,x_sol,formulation='complex_power')

    return elec_net,x_sol,iters,err_vec

def create_heat_network():
    """Create the decoupled single-carrier heat network."""
    heat_net = HN.create_network(heat_load='outflow',c_hl=True)
    return heat_net

def update_heat_network_bc(heat_net,xc):
    """Update the boundary values of the heat network, based on the outcome xc of the coupling network"""
    #dphic0 = -xc[2] #<0
    dphic1 = -xc[3] #<0
    Tsc0 = xc[4]
    Tsc1 = xc[5]
    heat_net.nodes[0].Ts = Tsc0
    heat_net.nodes[3].half_links[0].dphi = dphic1
    heat_net.nodes[3].half_links[0].Ts = Tsc1
    return heat_net

def initialize_heat_network(heat_net,formulation='standard'):
    x0 = HN.initialize_network(heat_net,c_hl=True,formulation=formulation,heat_load='outflow')
    return x0

def run_heat_load_flow(tol=1e-6,max_iter=10,scale_var=None,scale_var_params={'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase},formulation='standard'):
    """Steady-state load flow analysis of heat network.
    """
    # create network
    heat_net = create_heat_network()
    # initialize
    x0 = initialize_heat_network(heat_net,formulation=formulation)
    # solve network
    print('\nRunning load flow for single-carrier heat network')
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print_results_SC(heat_net,x_sol,formulation=formulation)

    return heat_net,x_sol,iters,err_vec

def run_mes_load_flow(tol=1e-6,max_iter=10,scale_var=None,scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase},formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None},hydr_eq_gas='fa'):
    # create network
    het_net, gas_net, elec_net, heat_net, water, gas  = MES.create_network(hydr_eq=hydr_eq_gas)
    # initialize
    x0 = MES.initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation)
    het_net.update(x0,formulation=formulation)

    # solve network
    print('\nRunning load flow for MES')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('Solution after {} it. (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print_results_ILF(het_net,x_sol,formulation=formulation)

    return x_sol,iters,err_vec

def create_coupling_network():
    """Create a coupling network consisting of two nodes, representing EH's"""
    # get required data from mes
    het_net, gas_net, elec_net, heat_net, water, gas  = MES.create_network()
    c0_mes = het_net.nodes[-2] # first EH in the mes
    c1_mes = het_net.nodes[-1] # second EH in the mes
    formulation_mes={'gas':'full','elec':'complex_power','heat':'standard','het':None}
    x0 = MES.initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation_mes)

    c0 = HeterogeneousNode('EH0',node_type=0,x=7,y=14,unit_type='EH',unit_params=c0_mes.unit_params.copy())# To unknown, with a heat power equation
    c0_hl_heat = HeatHalfLink('EH0_hlh',c0,link_type='heat_exchanger',link_params={'carrier':water},bc_type=12,m=-mc0_sol) # m (and Tr) known, dphi and Ts unknown, source
    c0_hl_heat.Tr = Trc0_sol # Set to solution
    c0_hl_gas = GasHalfLink('EH0_hlg',c0,-1,bc_type=0) # q unknown
    c0_hl_elec = ElectricalHalfLink('EH0_hle',c0,P=Pc0_sol,Q=Qc0_sol,bc_type=1) # P and Q known
    c1 = HeterogeneousNode('EH1',node_type=0,x=7,y=0,unit_type='EH',unit_params=c1_mes.unit_params.copy()) # To unknown, with a heat power equation
    c1_hl_heat = HeatHalfLink('EH1_hlh',c1,link_type='heat_exchanger',link_params={'carrier':water},bc_type=12,m=-mc1_sol) # m (and Tr) known, dphi and Ts unknown, source
    c1_hl_heat.Tr = Trc1_sol # Set to solution
    c1_hl_gas = GasHalfLink('EH1_hlg',c1,-qc1_sol,bc_type=1) # q known
    c1_hl_elec = ElectricalHalfLink('EH1_hle',c1,Q=Qc1_sol,bc_type=3) # P unknown and Q known

    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)
    coupling_net.add_node(c1)

    return coupling_net

def update_coupling_bc(coupling_net,gas_net,elec_net,heat_net,xg,xe,xh,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Update the boundary values of the coupling network, based on the outcome xg, xe, and xh of the single-carrier networks."""
    # update separate network fully
    gas_net.reset_network(xg,formulation=formulation.get('gas'))
    gas_net.update_full(xg,formulation=formulation.get('gas'))
    elec_net.reset_network(xe)
    elec_net.update_full(xe)
    heat_net.reset_network(xh,formulation=formulation.get('heat'))
    heat_net.update_full(xh,formulation=formulation.get('heat'))

    # gas part
    q2c = gas_net.nodes[2].half_links[0].get_q()
    # electrical part
    P0c = -(elec_net.nodes[0].half_links[0].get_P()-P0_load) #is a source, so the half link power <0
    Q0c = -(elec_net.nodes[0].half_links[0].get_Q()-Q0_load) #is a source, so the half link power <0
    Q1c = -elec_net.nodes[2].half_links[0].get_Q() #is a source, so the half link power <0
    m0c = heat_net.nodes[0].half_links[0].get_m() #<0
    Tr0c = heat_net.nodes[0].half_links[0].get_Tr()
    m1c = heat_net.nodes[3].half_links[0].get_m() #<0
    Tr1c = heat_net.nodes[3].half_links[0].get_Tr()
    # update BCs
    cn0 = coupling_net.nodes[0]
    cn1 = coupling_net.nodes[1]
    cn0.half_links[2].P = P0c
    cn0.half_links[2].Q = Q0c
    cn0.half_links[0].m = m0c
    cn0.half_links[0].Tr = Tr0c
    cn1.half_links[1].q = -q2c
    cn1.half_links[2].Q = Q1c
    cn1.half_links[0].m = m1c
    cn1.half_links[0].Tr = Tr1c
    return coupling_net

def initialize_coupling_net(coupling_net,qc_ic=15*MSCM*rhon_g/hour,Pc_ic=20*MW,phic_ic=25*MW,Tsc_ic=125.,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Initialize the coupling network"""
    x_init = np.array([qc_ic,Pc_ic,phic_ic,phic_ic,Tsc_ic,Tsc_ic])
    coupling_net.initialize()
    coupling_net.update(x_init,formulation=formulation)
    x0 = coupling_net.set_x_init(formulation=formulation)
    return x0

def run_coupling_load_flow(qc_ic=15*MSCM*rhon_g/hour,Pc_ic=20*MW,phic_ic=25*MW,Tsc_ic=125.,tol=1e-6,max_iter=10,scale_var=None,scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase},formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    # create network
    coupling_net = create_coupling_network()
    #initialize
    x0 = initialize_coupling_net(coupling_net,qc_ic=qc_ic,Pc_ic=Pc_ic,phic_ic=phic_ic,Tsc_ic=Tsc_ic,formulation=formulation)

    # solve network
    print('\nRunning load flow for gas-electricity-heat network, only coupling')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('Solution after {} iterations (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print_results_SC(coupling_net,x_sol,formulation=formulation)
    return coupling_net, x_sol, iters, err_vec

def xseparate_to_xmes(xg,xe,xh,xc,gas_net,elec_net,heat_net,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Create x for the gas-electricity-heat mes from the separate x's.
    """
    # update separate network fully
    gas_net.reset_network(xg,formulation=formulation.get('gas'))
    gas_net.update_full(xg,formulation=formulation.get('gas'))
    elec_net.reset_network(xe)
    elec_net.update_full(xe)
    heat_net.reset_network(xh,formulation=formulation.get('heat'))
    heat_net.update_full(xh,formulation=formulation.get('heat'))

    # gas part
    pg2 = gas_net.nodes[2].p
    qc1 = gas_net.nodes[2].half_links[0].get_q()
    if formulation.get('gas') == 'full':
        q01,q02,q12,q13,pg0,pg1,pg3 = xg
    else:
        pg0,pg1,pg3 = xg
    # electrical part
    Pc0 = -(elec_net.nodes[0].half_links[0].get_P()-P0_load) #is a source, so the half link power <0
    Qc0 = -(elec_net.nodes[0].half_links[0].get_Q()-Q0_load) #is a source, so the half link power <0
    Qc1 = -elec_net.nodes[2].half_links[0].get_Q() #is a source, so the half link power <0
    delta1,delta2,V1 = xe
    # heat part
    mc0 = heat_net.nodes[0].half_links[0].get_m() #<0
    Trc0 = heat_net.nodes[0].half_links[0].get_Tr()
    mc1 = heat_net.nodes[3].half_links[0].get_m() #<0
    Trc1 = heat_net.nodes[3].half_links[0].get_Tr()
    if formulation.get('heat') == 'half_link_flow':
        m01,m02,m12,m23,m1,m2,m3,ph1,ph2,Ts1,Ts2,Ts3,Tr0,Tr1,Tr2,Tr3 = xh
    else:
        m01,m02,m12,m23,ph1,ph2,Ts1,Ts2,Ts3,Tr0,Tr1,Tr2,Tr3 = xh

    # coupling part
    qc0,Pc1,phic0,phic1,Tsc0,Tsc1 = xc

    # make x_mes
    if formulation.get('gas') == 'full':
        xg_mes = np.array([q01,q02,q12,q13,pg1,pg2,pg3])
    else:
        xg_mes = np.array([pg1,pg2,pg3])
    xe_mes = np.array([delta1,delta2,V1])
    if formulation.get('heat') == 'half_link_flow':
        xh_mes = np.array([m01,m02,m12,m1,m2,ph1,Tsc0,Ts1,Ts2,Tr0,Tr1,Tr3])
    else:
        xh_mes = np.array([m01,m02,m12,ph1,Tsc0,Ts1,Ts2,Tr0,Tr1,Tr3])
    xc_mes = np.array([qc0,qc1,Pc0,Pc1,Qc0,Qc1,-mc0,-mc1,phic0,phic1,Tsc1])
    x_mes = np.concatenate((xg_mes,xe_mes,xh_mes,xc_mes))
    return x_mes

def xmes_to_xseparate(xmes,het_net,gas_net,elec_net,heat_net,coupling_net,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Create the separate xg, xe, xh, and xc for the single-carrier network, based on x of the gas-electricity-heat mes.
    The values of the BCs in the single-carrier networks (including the coupling network) are changed in this function.

    Parameters
    -----------------
    xmes : np array
        x of the MES
    het_net : HeterogeneousNetwork
        The mes
    gas_net : GasNetwork
        The single-carrier gas network. That is, NOT the gas part of the MES.
    elec_net : ElectricalNetwork
        The single-carrier electrical network. That is, NOT the electrical part of the MES.
    heat_net : HeatNetwork
        The single-carrier heat network. That is, NOT the heat part of the MES.
    coupling_net : GasNetwork
        The coupling network. That is, NOT the coupling part of the MES.

    Returns
    -----------
    xg : np array
        x of the single-carrier gas network.
    xe : np array
        x of the single-carrier electrical network.
    xh : np array
        x of the single-carrier heat network.
    xc : np array
        x of the single-carrier coupling network.
    """
    # update the mes
    het_net.reset_network(xmes,formulation=formulation)
    het_net.update_full(xmes,formulation=formulation)

    # separate xmes into parts (still corresponding to the mes)
    if formulation.get('gas') == 'full' and formulation.get('heat')== 'half_link_flow':
        q01,q02,q12,q13,pg1,pg2,pg3,delta1,delta2,V1,m01,m02,m12,m1,m2,ph1,Tsc0,Ts1,Ts2,Tr0,Tr1,Tr2,qc0,qc1,Pc0,Pc1,Qc0,Qc1,mc0,mc1,phic0,phic1,Tsc1 = xmes
    elif formulation.get('gas') == 'full' and formulation.get('heat')== 'standard':
        q01,q02,q12,q13,pg1,pg2,pg3,delta1,delta2,V1,m01,m02,m12,ph1,Tsc0,Ts1,Ts2,Tr0,Tr1,Tr2,qc0,qc1,Pc0,Pc1,Qc0,Qc1,mc0,mc1,phic0,phic1,Tsc1 = xmes
    elif formulation.get('gas') == 'nodal' and formulation.get('heat')== 'half_link_flow':
        pg1,pg2,pg3,delta1,delta2,V1,m01,m02,m12,m1,m2,ph1,Tsc0,Ts1,Ts2,Tr0,Tr1,Tr2,qc0,qc1,Pc0,Pc1,Qc0,Qc1,mc0,mc1,phic0,phic1,Tsc1 = xmes
    elif formulation.get('gas') == 'nodal' and formulation.get('heat')== 'standard':
        pg1,pg2,pg3,delta1,delta2,V1,m01,m02,m12,ph1,Tsc0,Ts1,Ts2,Tr0,Tr1,Tr2,qc0,qc1,Pc0,Pc1,Qc0,Qc1,mc0,mc1,phic0,phic1,Tsc1 = xmes

    # create x's of the single-carrier networks, and update their boundary conditions
    # gas part
    gas_nodes = list(het_net.get_nodes(carriers=['gas']))
    pg0 = gas_nodes[0].get_p()
    q_slack = gas_nodes[0].half_links[0].get_q()
    if formulation.get('gas') == 'full':
        xg = np.array([q01,q02,q12,q13,pg0,pg1,pg3])
    else:
        xg = np.array([pg0,pg1,pg3])
    gas_net.nodes[0].half_links[0].q = q_slack
    gas_net.nodes[0].half_links[1].q = qc0 # flow to coupling is outflow for the gas network
    gas_net.nodes[2].p = pg2
    gas_net.nodes[2].half_links[0].q = qc1
    # electrical part
    xe = np.array([delta1,delta2,V1])
    elec_net.nodes[0].half_links[0].P = -Pc0 + P0_load
    elec_net.nodes[0].half_links[0].Q = -Qc0 + Q0_load
    elec_net.nodes[2].half_links[0].Q = -Qc1
    elec_net.nodes[2].half_links[0].P = -Pc1 #generator, so source, so <0
    # heat part
    heat_nodes = list(het_net.get_nodes(carriers=['heat']))
    ph2 = heat_nodes[2].get_p()
    if formulation.get('heat') == 'half_link_flow':
        xh = np.array([m01,m02,m12,mc1,m1,m2,-mc1,ph1,ph2,Ts1,Ts2,Tsc1,Tr0,Tr1,Tr2,Tr2])
    else:
        xh = np.array([m01,m02,m12,mc1,ph1,ph2,Ts1,Ts2,Ts3,Tr0,Tr1,Tr2,Tr2])
    heat_net.nodes[0].Ts = Tsc0
    heat_net.nodes[3].half_links[0].Ts = Tsc1
    heat_net.nodes[3].half_links[0].dphi = -phic1
    # coupling part
    xc = np.array([qc0,Pc1,phic0,phic1,Tsc0,Tsc1])
    cn0 = coupling_net.nodes[0]
    cn1 = coupling_net.nodes[1]
    cn0.half_links[2].P = Pc0
    cn0.half_links[2].Q = Qc0
    cn0.half_links[0].m = -mc0
    cn0.half_links[0].Tr = Tr0
    cn1.half_links[1].q = -qc1
    cn1.half_links[2].Q = Qc1
    cn1.half_links[0].m = -mc1
    cn1.half_links[0].Tr = Tr2
    return xg, xe, xh, xc, gas_net,elec_net,heat_net,coupling_net

def error_measure_dd_F(gas_net,elec_net,heat_net,coupling_net,xg,xe,xh,xc,scale_var=None,scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase},formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """The error measure for DD of a electricity-heat coupling, based on (residual of) the system of equations.
    The network are assumed to be fully updated.
    """
    # update all boundary values
    gas_net = update_gas_network_bc(gas_net,xc)
    elec_net = update_electrical_network_bc(elec_net,xc)
    heat_net = update_heat_network_bc(heat_net,xc)
    coupling_net = update_coupling_bc(coupling_net,gas_net,elec_net,heat_net,xg,xe,xh,formulation=formulation)

    # determine the value of (residual of) the system of equations
    nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysh = NonLinearSystemHeat(heat_net,formulation=formulation.get('heat'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    if scale_var == 'matrix':
        Fg  = nlsysg.DF().dot(nlsysg.F(xg))
        Fe = nlsyse.DF().dot(nlsyse.F(xe))
        Fh = nlsysh.DF().dot(nlsysh.F(xh))
        Fc = nlsysc.DF().dot(nlsysc.F(xc))
    else:
        Fg  = nlsysg.F(xg)
        Fe = nlsyse.F(xe)
        Fh = nlsysh.F(xh)
        Fc = nlsysc.F(xc)
    error = np.linalg.norm(np.concatenate((Fg,Fe,Fh,Fc)))
    return error

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_mes_dd_error_measure_unsc():
    """Check if the solution MES is a solution of DD. That is, check if the solution of the intgrated MES gives an error of 0, using 'F' as the error measure."""
    # Given
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    het_net, gas_net_mes, elec_net_mes, heat_net_mes, water, gas  = MES.create_network()
    MES.initialize_network(het_net,gas_net_mes,elec_net_mes,heat_net_mes,water,gas,formulation)
    tol = 1e-7
    x_sol_mes, _, _ = run_mes_load_flow(tol=tol,max_iter=10,formulation=formulation)
    gas_net = create_gas_network()
    gas_net.initialize() #needed for update() to work (needs the adjacenc matrix)
    elec_net = create_electrical_network()
    elec_net.initialize()
    heat_net = create_heat_network()
    heat_net.initialize()
    coupling_net = create_coupling_network()
    xg, xe, xh, xc, gas_net,elec_net,heat_net, coupling_net = xmes_to_xseparate(x_sol_mes,het_net,gas_net,elec_net,heat_net,coupling_net,formulation=formulation)

    # When
    error = error_measure_dd_F(gas_net,elec_net,heat_net,coupling_net,xg,xe,xh,xc)
    error_norm = np.linalg.norm(error)
    print('error norm = {}'.format(error))
    # Then
    assert error_norm < tol

def is_ddsol_ilfsol(xg_dd,xe_dd,xh_dd,xc_dd,xmes_dd,tol,scale_var=None,scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase},formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Check if the solution found by  DD is also a solution of the MES when integrated load flow (ILF) would have been used. To check this, the values of the BCs of the integrated MES are set to the value of the solution of the BCs of the networks used in DD. Then, x for ILF is made based x of DD and on the values in the networks used in DD. This x is said to be a solution of the integrated MES, if F(x) for the integrated MES is smallen than the (outer) tolerance used for DD."""
    # create MES
    het_net, gas_net_mes, elec_net_mes, heat_net_mes, water, gas  = MES.create_network()
    MES.initialize_network(het_net,gas_net_mes,elec_net_mes,heat_net_mes,water,gas,formulation)
    # update values of the networks used in DD
    gas_net = create_gas_network()
    gas_net.initialize() #needed for update() to work (needs the adjacency matrix)
    gas_net = update_gas_network_bc(gas_net,xc_dd)
    gas_net.update_full(xg_dd,formulation=formulation.get('gas'))
    heat_net = create_heat_network()
    heat_net.initialize()
    heat_net = update_heat_network_bc(heat_net,xc_dd)
    heat_net.update_full(xh_dd,formulation=formulation.get('heat'))

    # set the values of the BC of the MES, based on solution of DD
    pg0 = gas_net.nodes[0].get_p()
    gas_net_mes.nodes[0].p = pg0
    ph2 = heat_net.nodes[2].get_p()
    heat_net_mes.nodes[2].p = ph2
    Ts0 = heat_net.nodes[0].get_Ts()
    heat_net_mes.links[3].Tsstart = Ts0

    # Determine the error of the integrated system of load flow equations for this x
    het_net.reset_network(xmes_dd,formulation=formulation)
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    if scale_var == 'matrix':
        F = nlsys.DF().dot(nlsys.F(xmes_dd))
    else:
        F = nlsys.F(xmes_dd)
    error_norm = np.linalg.norm(F)
    return error_norm < tol

def run_load_flow_dd(pg0_ic=55*bar,pg1_ic=45*bar,pg3_ic=45*bar,q_ic=20*MSCM*rhon_g/hour,qc_ic=15*MSCM*rhon_g/hour,Pc_ic=20*MW,phic_ic=25*MW,Tsc_ic=125.,tol_outer=1e-6,tol_inner=1e-6,max_iter_outer=1000,max_iter_inner=10,error_measure='F',scale_var=None,scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase},formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None},hydr_eq_gas='fa'):
    """Steady-state load flow analysis of multi-carrier electricity-heat network, using domain decomposition"""
    # create networks
    gas_net = create_gas_network(hydr_eq=hydr_eq_gas)
    elec_net = create_electrical_network()
    heat_net = create_heat_network()
    coupling_net = create_coupling_network()

    # initialize networks, and set initial guesses
    xg_new = initialize_gas_network(gas_net,pg0_ic=pg0_ic,pg1_ic=pg1_ic,pg3_ic=pg3_ic,q_ic=q_ic,formulation=formulation.get('gas'))
    xe_new = initialize_electrical_network(elec_net)
    xh_new = initialize_heat_network(heat_net,formulation=formulation.get('heat'))
    xc_new = initialize_coupling_net(coupling_net,qc_ic=qc_ic,Pc_ic=Pc_ic,phic_ic=phic_ic,Tsc_ic=Tsc_ic,formulation=formulation)

    # determine initial error.
    print('\nRunning load flow with DD, gas-electricity-heat, error {}'.format(error_measure))
    err_vec = list()
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    if error_measure == 'F':
        error = error_measure_dd_F(gas_net,elec_net,heat_net,coupling_net,xg_new,xe_new,xh_new,xc_new,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)
        err_vec.append(error)
    elif error_measure == 'dE':
        error = 1 # set some error which is bigger than the tolerance
        if scale_var == 'matrix':
            nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            Dxc = nlsysc.Dx()
    elif error_measure == 'phi0':
        # use difference between the heat power determined by the heat network, and the heat power determined by the coupling. Both represent the heat produced by EH0, so they should be the same
        m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(xh_new,formulation=formulation.get('heat'))
        phi0 = phi_hl_vec[0][0] #<0, since node 0 is a source
        phi0c = xc_new[2] #>0, since this is heat power produced
        error = np.abs(phi0c-(-phi0))
        if scale_var:
            error /= scale_var_params.get('phibase')
        err_vec.append(error)
    else:
        raise ValueError('enter valid error measure')

    # iterate
    iter_nr = 0 # I assume solution is not found at 1 iteration
    errors_gas = dict()
    errors_elec = dict()
    errors_heat = dict()
    errors_coupling = dict()
    while error > tol_outer and iter_nr < max_iter_outer:
        xg_old = xg_new
        xe_old = xe_new
        xh_old = xh_new
        xc_old = xc_new

        # Gas network. Update boundary, then solve, then update the boundary values
        gas_net = update_gas_network_bc(gas_net,xc_old)
        gas_net.reset_network(xg_old,formulation=formulation.get('gas'))
        with HiddenPrints():
            xg_new,itersg,err_vecg,p_sol,q_sol,q_inj = gas_net.solve_network(tol_inner,max_iter_inner,solver='NR',formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
        errors_gas[iter_nr] = err_vecg

        # Electricity network. Update boundary, then solve, then update the boundary values
        elec_net = update_electrical_network_bc(elec_net,xc_old)
        elec_net.reset_network(xe_old)
        with HiddenPrints():
            xe_new,iterse,err_vece,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol_inner,max_iter_inner,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        errors_elec[iter_nr] = err_vece

        # Heat network. Update boundary, then solve, then update the boundary values
        heat_net = update_heat_network_bc(heat_net,xc_old)
        heat_net.reset_network(xh_old,formulation=formulation.get('heat'))
        with HiddenPrints():
            xh_new,itersh,err_vech,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol_inner,max_iter_inner,scale_var=scale_var,scale_var_params=scale_var_params,solver='NR',formulation=formulation.get('heat'))
        errors_heat[iter_nr] = err_vech

        # Coupling network. Update boundary, then solve, then update the boundary values
        coupling_net = update_coupling_bc(coupling_net,gas_net,elec_net,heat_net,xg_new,xe_new,xh_new,formulation=formulation)
        coupling_net.reset_network(xc_old,formulation=formulation)
        with HiddenPrints():
            xc_new,itersc,err_vec_c,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.solve_network(tol_inner,max_iter_inner,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        errors_coupling[iter_nr] = err_vec_c

        # Determine error
        if error_measure == 'F':
            error = error_measure_dd_F(gas_net,elec_net,heat_net,coupling_net,xg_new,xe_new,xh_new,xc_new,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)
        elif error_measure == 'dE':
            if scale_var == 'matrix':
                error = np.linalg.norm(Dxc.dot(xc_new) - Dxc.dot(xc_old))
            else:
                error = np.linalg.norm(xc_new - xc_old)
        elif error_measure == 'phi0':
            # use difference between the heat power determined by the heat network, and the heat power determined by the coupling. Both represent the heat produced by EH0, so they should be the same
            m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(xh_new,formulation=formulation.get('heat'))
            phi0 = phi_hl_vec[0][0] #<0, since node 0 is a source
            phi0c = xc_new[2] #>0, since this is heat power produced
            error = np.abs(phi0c-(-phi0))
            if scale_var:
                error /= scale_var_params.get('phibase')
        err_vec.append(error)
        iter_nr += 1

    # print results
    print('Solution after {} iterations (final error = {:.4e}):'.format(iter_nr,err_vec[-1]))
    print_results_DD(gas_net,elec_net,heat_net,coupling_net,xg_new,xe_new,xh_new,xc_new,formulation=formulation)
    # create x of mes
    x_mes = xseparate_to_xmes(xg_new, xe_new, xh_new, xc_new,gas_net,elec_net,heat_net,formulation=formulation)
    return xg_new, xe_new, xh_new, xc_new, x_mes, iter_nr, err_vec, errors_gas, errors_elec, errors_heat, errors_coupling

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_mes_dd_unsc_errphi_issolution():
    """Check if the solution of DD would also be a solution of the integrated MES. DD is solved without scaling, using 'phi0' as the error measure."""
    # Given
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    tol = 1e-6
    pg0_ic = 55*bar
    pg1_ic = 45*bar
    pg3_ic = 45*bar
    q_ic = 20*MSCM*rhon_g/hour
    qc_ic = 15*MSCM*rhon_g/hour
    Pc_ic = 20*MW
    phic_ic = 25*MW
    Tsc_ic = 125.
    hydr_eq_gas = 'fa'

    # When
    xg, xe, xh, xc, x_sol_full, _, _, _, _, _, _ = run_load_flow_dd(hydr_eq_gas=hydr_eq_gas,pg0_ic=pg0_ic,pg1_ic=pg1_ic,pg3_ic=pg3_ic,q_ic=q_ic,qc_ic=qc_ic,Pc_ic=Pc_ic,phic_ic=phic_ic,Tsc_ic=Tsc_ic,tol_outer=tol,tol_inner=tol,max_iter_outer=600,error_measure='phi0',formulation=formulation)

    # Then
    assert is_ddsol_ilfsol(xg,xe,xh,xc,x_sol_full,tol,formulation=formulation)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_mes_dd_unsc_errF_issolution():
    """Check if the solution of DD would also be a solution of the integrated MES. DD is solved without scaling, using 'F' as the error measure."""
    # Given
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    tol = 1e-6
    pg0_ic = 55*bar
    pg1_ic = 45*bar
    pg3_ic = 45*bar
    q_ic = 20*MSCM*rhon_g/hour
    qc_ic = 15*MSCM*rhon_g/hour
    Pc_ic = 20*MW
    phic_ic = 25*MW
    Tsc_ic = 125.
    hydr_eq_gas = 'fa'

    # When
    xg, xe, xh, xc, x_sol_full, _, _, _, _, _, _ = run_load_flow_dd(hydr_eq_gas=hydr_eq_gas,pg0_ic=pg0_ic,pg1_ic=pg1_ic,pg3_ic=pg3_ic,q_ic=q_ic,qc_ic=qc_ic,Pc_ic=Pc_ic,phic_ic=phic_ic,Tsc_ic=Tsc_ic,tol_outer=tol,tol_inner=tol,max_iter_outer=600,error_measure='F',formulation=formulation)

    # Then
    assert is_ddsol_ilfsol(xg,xe,xh,xc,x_sol_full,tol,formulation=formulation)

def compare_DD(pg0_ic=55*bar,pg1_ic=45*bar,pg3_ic=45*bar,q_ic=20*MSCM*rhon_g/hour,qc_ic=15*MSCM*rhon_g/hour,Pc_ic=20*MW,phic_ic=25*MW,Tsc_ic=125.,tol_outer=1e-6,tol_inner=1e-6,max_iter_outer=1000,max_iter_inner=10,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None},hydr_eq_gas='fa',save_tables=False,path_to_tables=None):
    """Compare the overall convergence of DD for the electricity-heat network with solving the integrated system of equations. Both error measures 'dE' and 'F' are used. Convergence plots are made.
    NB: For solving the MES using NR and the integrated system of equations, different BC (node types but also values) and different initial conditions are used than for DD!!!

    Parameters
    ----------------
    plot_details : bool, optional
        If True, additional plots are made with detailed convergence. That is, the convergence of the inner NR iterations of the single-carrier networks is plotted, as well as the overall convergence of DD. Default is False.
    """
    # unscaled
    xg_dE_unsc, xe_dE_unsc, xh_dE_unsc, xc_dE_unsc, xmes_dE_unsc, iters_dE_unsc, err_vec_dE_unsc, errors_gas_dE_unsc, errors_elec_dE_unsc, errors_heat_dE_unsc, errors_coupling_dE_unsc = run_load_flow_dd(pg0_ic=pg0_ic,pg1_ic=pg1_ic,pg3_ic=pg3_ic,q_ic=q_ic,qc_ic=qc_ic,Pc_ic=Pc_ic,phic_ic=phic_ic,Tsc_ic=Tsc_ic,tol_outer=tol_outer,tol_inner=tol_inner,max_iter_outer=max_iter_outer,max_iter_inner=max_iter_inner,error_measure='dE',formulation=formulation,hydr_eq_gas=hydr_eq_gas)
    xg_F_unsc, xe_F_unsc, xh_F_unsc, xc_F_unsc, xmes_F_unsc, iters_F_unsc, err_vec_F_unsc, errors_gas_F_unsc, errors_elec_F_unsc, errors_heat_F_unsc, errors_coupling_F_unsc = run_load_flow_dd(pg0_ic=pg0_ic,pg1_ic=pg1_ic,pg3_ic=pg3_ic,q_ic=q_ic,qc_ic=qc_ic,Pc_ic=Pc_ic,phic_ic=phic_ic,Tsc_ic=Tsc_ic,tol_outer=tol_outer,tol_inner=tol_inner,max_iter_outer=max_iter_outer,max_iter_inner=max_iter_inner,error_measure='F',formulation=formulation,hydr_eq_gas=hydr_eq_gas)
    xg_phi_unsc, xe_phi_unsc, xh_phi_unsc, xc_phi_unsc, xmes_phi_unsc, iters_phi_unsc, err_vec_phi_unsc, errors_gas_phi_unsc, errors_elec_phi_unsc, errors_heat_phi_unsc, errors_coupling_phi_unsc = run_load_flow_dd(pg0_ic=pg0_ic,pg1_ic=pg1_ic,pg3_ic=pg3_ic,q_ic=q_ic,qc_ic=qc_ic,Pc_ic=Pc_ic,phic_ic=phic_ic,Tsc_ic=Tsc_ic,tol_outer=tol_outer,tol_inner=tol_inner,max_iter_outer=max_iter_outer,max_iter_inner=max_iter_inner,error_measure='phi0',formulation=formulation,hydr_eq_gas=hydr_eq_gas)
    x_mes_unsc, iters_mes_unsc, err_vec_mes_unsc = run_mes_load_flow(tol=tol_inner,max_iter=max_iter_inner,formulation=formulation,hydr_eq_gas=hydr_eq_gas)
    # scaled
    xg_dE_sc, xe_dE_sc, xh_dE_sc, xc_dE_sc, xmes_dE_sc, iters_dE_sc, err_vec_dE_sc, errors_gas_dE_sc, errors_elec_dE_sc, errors_heat_dE_sc, errors_coupling_dE_sc = run_load_flow_dd(pg0_ic=pg0_ic,pg1_ic=pg1_ic,pg3_ic=pg3_ic,q_ic=q_ic,qc_ic=qc_ic,Pc_ic=Pc_ic,phic_ic=phic_ic,Tsc_ic=Tsc_ic,tol_outer=tol_outer,tol_inner=tol_inner,max_iter_outer=max_iter_outer,max_iter_inner=max_iter_inner,error_measure='dE',formulation=formulation,scale_var='matrix',hydr_eq_gas=hydr_eq_gas)
    xg_F_sc, xe_F_sc, xh_F_sc, xc_F_sc, xmes_F_sc, iters_F_sc, err_vec_F_sc, errors_gas_F_sc, errors_elec_F_sc, errors_heat_F_sc, errors_coupling_F_sc = run_load_flow_dd(pg0_ic=pg0_ic,pg1_ic=pg1_ic,pg3_ic=pg3_ic,q_ic=q_ic,qc_ic=qc_ic,Pc_ic=Pc_ic,phic_ic=phic_ic,Tsc_ic=Tsc_ic,tol_outer=tol_outer,tol_inner=tol_inner,max_iter_outer=max_iter_outer,max_iter_inner=max_iter_inner,error_measure='F',formulation=formulation,scale_var='matrix',hydr_eq_gas=hydr_eq_gas)
    xg_phi_sc, xe_phi_sc, xh_phi_sc, xc_phi_sc, xmes_phi_sc, iters_phi_sc, err_vec_phi_sc, errors_gas_phi_sc, errors_elec_phi_sc, errors_heat_phi_sc, errors_coupling_phi_sc = run_load_flow_dd(pg0_ic=pg0_ic,pg1_ic=pg1_ic,pg3_ic=pg3_ic,q_ic=q_ic,qc_ic=qc_ic,Pc_ic=Pc_ic,phic_ic=phic_ic,Tsc_ic=Tsc_ic,tol_outer=tol_outer,tol_inner=tol_inner,max_iter_outer=max_iter_outer,max_iter_inner=max_iter_inner,error_measure='phi0',formulation=formulation,scale_var='matrix',hydr_eq_gas=hydr_eq_gas)
    x_mes_sc, iters_mes_sc, err_vec_mes_sc = run_mes_load_flow(tol=tol_inner,max_iter=max_iter_inner,formulation=formulation,scale_var='matrix',hydr_eq_gas=hydr_eq_gas)

    # print some results.
    print('\nIterations unscaled: integrated system (ILF) = {}, DD dE (outer iters) = {}, DD F (outer iters) = {}, DD phi (outer iters) = {}'.format(iters_mes_unsc,iters_dE_unsc,iters_F_unsc,iters_phi_unsc))
    print('Difference between solutions (NB: this is only solutions, so the difference in values in BCs is not shown!!!):')
    print('||x ILF - x DD dE||_2 unscaled = {}'.format(np.linalg.norm(x_mes_unsc - xmes_dE_unsc)))
    print('||x ILF - x DD F||_2 unscaled = {}'.format(np.linalg.norm(x_mes_unsc - xmes_F_unsc)))
    print('||x ILF - x DD phi||_2 unscaled = {}'.format(np.linalg.norm(x_mes_unsc - xmes_phi_unsc)))
    print('Solution of DD is a solution of ILF MES (NB: this does update the values of the BC of the MES to match the DD networks!!!):')
    print('x DD dE: {}'.format(is_ddsol_ilfsol(xg_dE_unsc,xe_dE_unsc,xh_dE_unsc,xc_dE_unsc,xmes_dE_unsc,tol_outer,formulation=formulation)))
    print('x DD F: {}'.format(is_ddsol_ilfsol(xg_F_unsc,xe_F_unsc,xh_F_unsc,xc_F_unsc,xmes_F_unsc,tol_outer,formulation=formulation)))
    print('x DD phi: {}'.format(is_ddsol_ilfsol(xg_phi_unsc,xe_phi_unsc,xh_phi_unsc,xc_phi_unsc,xmes_phi_unsc,tol_outer,formulation=formulation)))
    print('Iterations scaled: integrated system = {}, DD dE (outer iters) = {}, DD F (outer iters) = {}, DD phi (outer iters) = {}'.format(iters_mes_sc,iters_dE_sc,iters_F_sc,iters_phi_sc))
    print('Difference between solutions (NB: this is only solutions, so the difference in values in BCs is not shown!!!')
    print('||x ILF - x DD dE||_2 scaled = {}'.format(np.linalg.norm(x_mes_sc - xmes_dE_sc)))
    print('||x ILF - x DD F||_2 scaled = {}'.format(np.linalg.norm(x_mes_sc - xmes_F_sc)))
    print('||x ILF - x DD phi||_2 scaled = {}'.format(np.linalg.norm(x_mes_sc - xmes_phi_sc)))
    print('Solution of DD is a solution of ILF MES (NB: this does update the values of the BC of the MES to match the DD networks!!!):')
    print('x DD dE: {}'.format(is_ddsol_ilfsol(xg_dE_sc,xe_dE_sc,xh_dE_sc,xc_dE_sc,xmes_dE_sc,tol_outer,formulation=formulation,scale_var='matrix')))
    print('x DD F: {}'.format(is_ddsol_ilfsol(xg_F_sc,xe_F_sc,xh_F_sc,xc_F_sc,xmes_F_sc,tol_outer,formulation=formulation,scale_var='matrix')))
    print('x DD phi: {}'.format(is_ddsol_ilfsol(xg_phi_sc,xe_phi_sc,xh_phi_sc,xc_phi_sc,xmes_phi_sc,tol_outer,formulation=formulation,scale_var='matrix')))

    # plot convergence
    fig = plt.figure(r'Convergence plot gas-electricity-heat, unscaled (hydr eq gas: {})'.format(hydr_eq_gas))
    ax_conv_unsc = fig.gca()
    ax_conv_unsc.semilogy(err_vec_mes_unsc,marker='.',label='Integrated system')
    ax_conv_unsc.semilogy(range(1,len(err_vec_dE_unsc)+1),err_vec_dE_unsc,marker='.',label=r'DD $\Delta E$')
    ax_conv_unsc.semilogy(err_vec_F_unsc,marker='.',label=r'DD $F$')
    ax_conv_unsc.semilogy(err_vec_phi_unsc,marker='.',label=r'DD $\Delta \varphi$')
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error')

    fig = plt.figure(r'Convergence plot gas-electricity-heat, scaled (hydr eq gas: {})'.format(hydr_eq_gas))
    ax_conv_sc = fig.gca()
    ax_conv_sc.semilogy(err_vec_mes_sc,marker='.',label='Integrated system')
    ax_conv_sc.semilogy(range(1,len(err_vec_dE_sc)+1),err_vec_dE_sc,marker='.',label=r'DD $\Delta E$')
    ax_conv_sc.semilogy(err_vec_F_sc,marker='.',label=r'DD $F$')
    ax_conv_sc.semilogy(err_vec_phi_sc,marker='.',label=r'DD $\Delta \varphi$')
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error')

    # plot convergence order
    fig = plt.figure(r'Convergence order gas-electricity-heat, unscaled (hydr eq gas: {})'.format(hydr_eq_gas))
    ax_ord_unsc = fig.gca()
    plt.xlabel(r'$||\Delta e^k||_2 / ||\Delta e^0||_2$')
    plt.ylabel(r'$||\Delta e^{k+1}||_2 / ||\Delta e^0||_2$')
    ax_ord_unsc.loglog(err_vec_mes_unsc[:-1]/err_vec_mes_unsc[0],err_vec_mes_unsc[1:]/err_vec_mes_unsc[0],marker='.',label='Integrated system')
    ax_ord_unsc.loglog(err_vec_dE_unsc[:-1]/err_vec_dE_unsc[0],err_vec_dE_unsc[1:]/err_vec_dE_unsc[0],marker='.',label=r'DD $\Delta E$')
    ax_ord_unsc.loglog(err_vec_F_unsc[:-1]/err_vec_F_unsc[0],err_vec_F_unsc[1:]/err_vec_F_unsc[0],marker='.',label=r'DD $F$')
    ax_ord_unsc.loglog(err_vec_phi_unsc[:-1]/err_vec_phi_unsc[0],err_vec_phi_unsc[1:]/err_vec_phi_unsc[0],marker='.',label=r'DD $\Delta \varphi$')

    fig = plt.figure(r'Convergence order gas-electricity-heat, scaled (hydr eq gas: {})'.format(hydr_eq_gas))
    ax_ord_sc = fig.gca()
    plt.xlabel(r'$||\Delta e^k||_2 / ||\Delta e^0||_2$')
    plt.ylabel(r'$||\Delta e^{k+1}||_2 / ||\Delta e^0||_2$')
    ax_ord_sc.loglog(err_vec_mes_sc[:-1]/err_vec_mes_sc[0],err_vec_mes_sc[1:]/err_vec_mes_sc[0],marker='.',label='Integrated system')
    ax_ord_sc.loglog(err_vec_dE_sc[:-1]/err_vec_dE_sc[0],err_vec_dE_sc[1:]/err_vec_dE_sc[0],marker='.',label=r'DD $\Delta E$')
    ax_ord_sc.loglog(err_vec_F_sc[:-1]/err_vec_F_sc[0],err_vec_F_sc[1:]/err_vec_F_sc[0],marker='.',label=r'DD $F$')
    ax_ord_sc.loglog(err_vec_phi_sc[:-1]/err_vec_phi_sc[0],err_vec_phi_sc[1:]/err_vec_phi_sc[0],marker='.',label=r'DD $\Delta \varphi$')

    max_iters_used = np.max([iters_mes_unsc,iters_dE_unsc,iters_F_unsc,iters_phi_unsc,iters_mes_sc,iters_dE_sc,iters_F_sc,iters_phi_sc])
    layout_convergence_dd(ax_conv_unsc,tol_outer,tol_inner,max_iters_used)
    layout_convergence_dd(ax_conv_sc,tol_outer,tol_inner,max_iters_used)
    layout_convergence_order(ax_ord_unsc)
    layout_convergence_order(ax_ord_sc)

    if save_tables:
        write_data_to_tables(xg_F_unsc,xe_F_unsc,xh_F_unsc,xc_F_unsc,x_mes_unsc,path_to_tables,formulation=formulation)

def layout_convergence(ax,tol,max_iters):
    ax.semilogy([0,max_iters+1],[tol,tol],'k:',label='tolerance')
    xmin = 0
    xmax = max_iters
    xticks = range(xmin,xmax+1,max(1,int(xmax/10))) # make sure the xticks are integers
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

def layout_convergence_order(ax):
    x_min,x_max = ax.get_xlim()
    x_slope = np.linspace(x_min,x_max)
    y_slope2 = x_slope**2
    ax.loglog(x_slope,x_slope,linestyle=':',color='k',label='slope 1')
    ax.loglog(x_slope,y_slope2,linestyle='-.',color='k',label='slope 2')
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)

def layout_convergence_dd(ax,tol,tol_sub,max_iters):
    ax.semilogy([0,max_iters+1],[tol,tol],'k:',label='tolerance')
    ax.semilogy([0,max_iters+1],[tol_sub,tol_sub],'k-.',label='tolerance subnetworks')
    #ax.semilogy([0,max_iters+1],[art_zero,art_zero],'k--',label='artifical 0')
    xmin = 0
    xmax = max_iters
    xticks = range(xmin,xmax+1,max(1,int(xmax/10))) # make sure the xticks are integers
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

def print_results_SC(net,x,formulation=None):
    """Print the outcome (i.e. flows etc. in the network) of load flow for single-carrier or coupling network"""
    if isinstance(net,GasNetwork):
        p_sol,q_sol,q_inj = net.update_full(x,formulation=formulation)
        print('p = {} bar'.format(p_sol/bar))
        print('q links = {} kg/s'.format(q_sol))
        print('q links = {} MSCM/hour'.format(q_sol/(MSCM*rhon_g/hour)))
        print('q nodal inj = {} kg/s'.format(q_inj))
        print('q hl = {} kg/s'.format([hl.q for node in net.get_nodes() for hl in node.get_half_links()]))
        print('q hl = {} MSCM/hour'.format([hl.q/(MSCM*rhon_g/hour) for node in net.get_nodes() for hl in node.get_half_links()]))
    elif isinstance(net,ElectricalNetwork):
        delta_sol,V_sol,S_inj,P_edge,Q_edge = net.update_full(x)
        print('delta = {}'.format(delta_sol))
        print('|V| = {} V'.format(V_sol))
        print('|V| = {} p.u.'.format(V_sol/Vbase))
        print('P edge = {} MW'.format(P_edge/MW))
        print('Q edge = {} MW'.format(Q_edge/MW))
        print('S nodal inj = {} MW'.format(S_inj/MW))
        print('P hl = {} MW'.format([hl.P/MW for node in net.get_nodes() for hl in node.get_half_links()]))
        print('Q hl = {} MW'.format([hl.Q/MW for node in net.get_nodes() for hl in node.get_half_links()]))
    elif isinstance(net,HeatNetwork):
        m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = net.update_full(x,formulation=formulation)
        print('p heat = {} m'.format(p_vec/(rho_w*grav_const)))
        print('m = {} kg/s'.format(m_vec))
        print('Ts = {} C'.format(Ts_vec))
        print('Tr = {} C'.format(Tr_vec))
        print('m hl = {}'.format(m_hl_vec))
        print('Ts hl = {}'.format(Ts_hl_vec))
        print('Tr hl = {}'.format(Tr_hl_vec))
        print('phi hl = {} MW'.format([hl.dphi/MW for hl in net.get_half_links()]))
    elif isinstance(net,HeterogeneousNetwork):
        _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = net.update_full(x,formulation=formulation)
        print('qc = {} kg/s'.format(qc_vec))
        print('qc = {} MSCM/hour'.format(np.array(qc_vec)/(MSCM*rhon_g/hour)))
        print('Pc = {} W'.format(Pc_vec))
        print('Pc = {} MW'.format(np.array(Pc_vec)/MW))
        print('phic = {} W'.format(phic_vec))
        print('phic = {} MW'.format(np.array(phic_vec)/MW))
        print('Tsc = {} C'.format(Tsc_vec))
        print('Trc = {} C'.format(Trc_vec))

def print_results_DD(gas_net,elec_net,heat_net,coupling_net,xg,xe,xh,xc,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Print the outcome (i.e. flows etc. in the network) of the load flow, when DD was used"""
    # gas part
    gas_net = update_gas_network_bc(gas_net,xc)
    print('Single-carrier gas network:')
    print_results_SC(gas_net,xg,formulation=formulation.get('gas'))
    #electrical part
    elec_net = update_electrical_network_bc(elec_net,xc)
    print('Single-carrier electrical network:')
    print_results_SC(elec_net,xe)
    # heat part
    heat_net = update_heat_network_bc(heat_net,xc)
    print('Single-carrier heat network:')
    print_results_SC(heat_net,xh,formulation=formulation.get('heat'))
    # coupling part
    coupling_net = update_coupling_bc(coupling_net,gas_net,elec_net,heat_net,xg,xe,xh,formulation=formulation)
    print('Coupling network:')
    print_results_SC(coupling_net,xc,formulation=formulation)

def print_results_ILF(het_net,x,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Print the outcome (i.e. flows etc. in the network) of the load flow for MES, when the integrated system of equations was solved"""
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(x,formulation=formulation)
    # gas part
    print('p = {} bar'.format(p_g_vec/bar))
    print('q links = {} MSCM/hour'.format(q_vec/(MSCM*rhon_g/hour)))
    print('q nodal inj = {} MSCM/hour'.format(q_inj/(MSCM*rhon_g/hour)))
    print('q hl = {} MSCM/hour'.format([hl.q/(MSCM*rhon_g/hour) for node in het_net.get_nodes(carriers=['gas']) for hl in node.get_half_links()]))
    # electrical part
    print('delta = {} rad'.format(delta_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/MES.Vbase_shabanpour))
    print('P edge = {} p.u.'.format(P_edge/MES.Sbase_shabanpour))
    print('Q edge = {} p.u.'.format(Q_edge/MES.Sbase_shabanpour))
    print('S nodal inj = {} p.u.'.format(S_inj/MES.Sbase_shabanpour))
    print('P hl = {} MW'.format([hl.P/MW for node in het_net.get_nodes(carriers=['elec']) for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in het_net.get_nodes(carriers=['elec']) for hl in node.get_half_links()]))
    # heat part
    print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {}'.format(m_hl_vec))
    print('Ts hl = {}'.format(Ts_hl_vec))
    print('Tr hl = {}'.format(Tr_hl_vec))
    print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes(carriers=['heat'])  for hl in node.get_half_links()]))
    # coupling part
    print('qc = {} MSCM/hour'.format(np.array(qc_vec)/(MSCM*rhon_g/hour)))
    print('Pc = {} MW'.format(np.array(Pc_vec)/MW))
    print('phic = {} MW'.format(np.array(phic_vec)/MW))
    print('Tsc = {} C'.format(Tsc_vec))
    print('Trc = {} C'.format(Trc_vec))

def write_data_to_tables(xg_dd,xe_dd,xh_dd,xc_dd,xmes_ilf,path_to_tables,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Write the data for load flow of the integrated system and for load flow using DD to a table, which can be read by latex"""
    # create MES
    het_net, gas_net_mes, elec_net_mes, heat_net_mes, water, gas  = MES.create_network()
    het_net.initialize()
    het_net.update_full(xmes_ilf,formulation=formulation)

    # create and update values of the networks used in DD
    gas_net = create_gas_network()
    gas_net.initialize() #needed for update() to work (needs the adjacency matrix)
    gas_net = update_gas_network_bc(gas_net,xc_dd)
    gas_net.update_full(xg_dd,formulation=formulation.get('gas'))
    elec_net = create_electrical_network()
    elec_net.initialize() #needed for update() to work (needs the adjacency matrix)
    elec_net = update_electrical_network_bc(elec_net,xc_dd)
    elec_net.update_full(xe_dd)
    heat_net = create_heat_network()
    heat_net.initialize()
    heat_net = update_heat_network_bc(heat_net,xc_dd)
    heat_net.update_full(xh_dd,formulation=formulation.get('heat'))
    coupling_net = create_coupling_network()
    coupling_net.initialize() #needed for update() to work (needs the adjacency matrix)
    coupling_net = update_coupling_bc(coupling_net,gas_net,elec_net,heat_net,xg_dd,xe_dd,xh_dd,formulation=formulation)
    coupling_net.update_full(xc_dd,formulation=formulation)

    if path_to_tables:
        with open(os.path.join(path_to_tables,'gas_data_EH_DD.txt'), "w") as table:
            table.write(r'\resizebox{\columnwidth}{!}{')
            table.write(r'\begin{tabular}{|c|c|c|c|c|c|c|c|c|} ')
            table.write(r'\hline ')
            table.write(r'\textbf{Node} & \multicolumn{2}{c|}{$p$~\sib{\bar}}  & \textbf{Half link} & \multicolumn{2}{c|}{$q$~\SIB{e3}{\cubic\meter \per \hour}} & \textbf{Link} & \multicolumn{2}{c|}{$q$~\SIB{e3}{\cubic\meter \per \hour}}\\ ')
            table.write(r'\hline ')
            table.write(r' & (a) & (b) & & (a) & (b) & & (a) & (b)  \\ ')
            table.write(r'\hline ')
            table.write(r'0 & {:.3f} & {:.3f} & 0 & {:.3f} & {:.3f} & 0-1  & {:.3f} & {:.3f}\\'.format(gas_net_mes.nodes[0].get_p()/bar,gas_net.nodes[0].get_p()/bar,gas_net_mes.nodes[0].half_links[0].get_q()/(MSCM*rhon_g/hour),gas_net.nodes[0].half_links[0].get_q()/(MSCM*rhon_g/hour),gas_net_mes.links[0].get_q()/(MSCM*rhon_g/hour),gas_net.links[0].get_q()/(MSCM*rhon_g/hour)))
            table.write(r' & - & - & 1 & - & {:.3f} &   &  - & -\\'.format(gas_net.nodes[0].half_links[1].get_q()/(MSCM*rhon_g/hour)))
            table.write(r'1 & {:.3f} & {:.3f} & 0 & {:.3f} & {:.3f} & 0-2  & {:.3f} & {:.3f}\\'.format(gas_net_mes.nodes[1].get_p()/bar,gas_net.nodes[1].get_p()/bar,gas_net_mes.nodes[1].half_links[0].get_q()/(MSCM*rhon_g/hour),gas_net.nodes[1].half_links[0].get_q()/(MSCM*rhon_g/hour),gas_net_mes.links[1].get_q()/(MSCM*rhon_g/hour),gas_net.links[1].get_q()/(MSCM*rhon_g/hour)))
            table.write(r'2 & {:.3f} & {:.3f} & 0 & {:.3f} & {:.3f} & 3-2  & {:.3f} & {:.3f}\\'.format(gas_net_mes.nodes[2].get_p()/bar,gas_net.nodes[2].get_p()/bar,gas_net_mes.nodes[2].half_links[0].get_q()/(MSCM*rhon_g/hour),gas_net.nodes[2].half_links[1].get_q()/(MSCM*rhon_g/hour),gas_net_mes.links[2].get_q()/(MSCM*rhon_g/hour),gas_net.links[2].get_q()/(MSCM*rhon_g/hour)))
            table.write(r' & - & - & 1 & - & {:.3f} &   &  - & -\\'.format(gas_net.nodes[2].half_links[0].get_q()/(MSCM*rhon_g/hour)))
            table.write(r'3 & {:.3f} & {:.3f} & 0 & {:.3f} & {:.3f} & 1-3  & {:.3f} & {:.3f}\\'.format(gas_net_mes.nodes[3].get_p()/bar,gas_net.nodes[3].get_p()/bar,gas_net_mes.nodes[3].half_links[0].get_q()/(MSCM*rhon_g/hour),gas_net.nodes[3].half_links[0].get_q()/(MSCM*rhon_g/hour),gas_net_mes.links[3].get_q()/(MSCM*rhon_g/hour),gas_net.links[3].get_q()/(MSCM*rhon_g/hour)))
            table.write(r'\hline ')
            table.write(r'\end{tabular}}%')
        with open(os.path.join(path_to_tables,'elec_data_EH_DD.txt'), "w") as table:
            table.write(r'\resizebox{\columnwidth}{!}{')
            table.write(r'\begin{tabular}{|c|c|c|c|c|c|c|c|} ')
            table.write(r'\hline ')
            table.write(r'\textbf{Node} & \multicolumn{2}{c|}{$|V|$~\sib{\perunit}}  & \multicolumn{2}{c|}{$\delta$~\sib{\degree}} & \textbf{HalfLink} & \multicolumn{2}{c|}{$S$~\sib{\mega \watt}}\\ ')
            table.write(r'\hline ')
            table.write(r' & (a) & (b) & (a) & (b) &  & (a) & (b)\\ ')
            table.write(r'\hline ')
            # node 0
            table.write(r'0 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & 0 & {:.3f} {:+.3f}$\iu$ & {:.3f} {:+.3f}$\iu$ \\ '.format(elec_net_mes.nodes[0].get_V()/MES.Vbase_shabanpour,elec_net.nodes[0].get_V()/MES.Vbase_shabanpour,elec_net_mes.nodes[0].get_delta()/degree,elec_net.nodes[0].get_delta()/degree,elec_net_mes.nodes[0].half_links[0].get_P()/MW,elec_net_mes.nodes[0].half_links[0].get_Q()/MW,P0_load/MW,Q0_load/MW))
            table.write(r' & - & - & - & - & 1 & - & {:.3f} {:+.3f}$\iu$ \\ '.format((elec_net.nodes[0].half_links[0].get_P()-P0_load)/MW,(elec_net.nodes[0].half_links[0].get_Q()-Q0_load)/MW))
            # node 1
            table.write(r'1 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & 0 & {:.3f} {:+.3f}$\iu$ & {:.3f} {:+.3f}$\iu$ \\ '.format(elec_net_mes.nodes[1].get_V()/MES.Vbase_shabanpour,elec_net.nodes[1].get_V()/MES.Vbase_shabanpour,elec_net_mes.nodes[1].get_delta()/degree,elec_net.nodes[1].get_delta()/degree,elec_net_mes.nodes[1].half_links[0].get_P()/MW,elec_net_mes.nodes[1].half_links[0].get_Q()/MW,elec_net.nodes[1].half_links[0].get_P()/MW,elec_net.nodes[1].half_links[0].get_Q()/MW))
            # node 2
            table.write(r'2 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & 0 & {:.3f} {:+.3f}$\iu$ & {:.3f} {:+.3f}$\iu$ \\ '.format(elec_net_mes.nodes[2].get_V()/MES.Vbase_shabanpour,elec_net.nodes[2].get_V()/MES.Vbase_shabanpour,elec_net_mes.nodes[2].get_delta()/degree,elec_net.nodes[2].get_delta()/degree,elec_net_mes.nodes[2].half_links[0].get_P()/MW,elec_net_mes.nodes[2].half_links[0].get_Q()/MW,elec_net.nodes[2].half_links[1].get_P()/MW,elec_net.nodes[2].half_links[1].get_Q()/MW))
            table.write(r' & - & - & - & - & 1 & - & {:.3f} {:+.3f}$\iu$ \\ '.format((elec_net.nodes[2].half_links[0].get_P())/MW,(elec_net.nodes[2].half_links[0].get_Q())/MW))
            table.write(r'\hline ')
            # link header
            table.write(r'\textbf{Link} & \multicolumn{2}{c|}{$S_{ij}$~\sib{\mega \watt}}  & \multicolumn{2}{c|}{$S_{ji}$~\sib{\mega \watt}}& \multicolumn{2}{c|}{$S^\text{loss}_{ij}$~\sib{\mega \watt}} & \\ ')
            table.write(r'\hline ')
            table.write(r' & (a) & (b) & (a) & (b) & (a) & (b) & \\ ')
            table.write(r'\hline ')
            # link 0-1
            table.write(r'0-1 & {:.3f} {:+.3f}$\iu$  & {:.3f} {:+.3f}$\iu$  & {:.3f} {:+.3f}$\iu$ & {:.3f} {:+.3f}$\iu$ &  {:.3f} {:+.3f}$\iu$  & {:.3f} {:+.3f}$\iu$ & \\ '.format(elec_net_mes.links[0].get_Pstart()/MW,elec_net_mes.links[0].get_Qstart()/MW,elec_net.links[0].get_Pstart()/MW,elec_net.links[0].get_Qstart()/MW,elec_net_mes.links[0].get_Pend()/MW,elec_net_mes.links[0].get_Qend()/MW,elec_net.links[0].get_Pend()/MW,elec_net.links[0].get_Qend()/MW,elec_net_mes.links[0].complex_power_loss().real/MW,elec_net_mes.links[0].complex_power_loss().imag/MW,elec_net.links[0].complex_power_loss().real/MW,elec_net.links[0].complex_power_loss().imag/MW))
            # link 0-2
            table.write(r'0-2 & {:.3f} {:+.3f}$\iu$  & {:.3f} {:+.3f}$\iu$  & {:.3f} {:+.3f}$\iu$ & {:.3f} {:+.3f}$\iu$ &  {:.3f} {:+.3f}$\iu$  & {:.3f} {:+.3f}$\iu$ & \\ '.format(elec_net_mes.links[1].get_Pstart()/MW,elec_net_mes.links[1].get_Qstart()/MW,elec_net.links[1].get_Pstart()/MW,elec_net.links[1].get_Qstart()/MW,elec_net_mes.links[1].get_Pend()/MW,elec_net_mes.links[1].get_Qend()/MW,elec_net.links[1].get_Pend()/MW,elec_net.links[1].get_Qend()/MW,elec_net_mes.links[1].complex_power_loss().real/MW,elec_net_mes.links[1].complex_power_loss().imag/MW,elec_net.links[1].complex_power_loss().real/MW,elec_net.links[1].complex_power_loss().imag/MW))
            # link 1-2
            table.write(r'1-2 & {:.3f} {:+.3f}$\iu$  & {:.3f} {:+.3f}$\iu$  & {:.3f} {:+.3f}$\iu$ & {:.3f} {:+.3f}$\iu$ &  {:.3f} {:+.3f}$\iu$  & {:.3f} {:+.3f}$\iu$ & \\ '.format(elec_net_mes.links[2].get_Pstart()/MW,elec_net_mes.links[2].get_Qstart()/MW,elec_net.links[2].get_Pstart()/MW,elec_net.links[2].get_Qstart()/MW,elec_net_mes.links[2].get_Pend()/MW,elec_net_mes.links[2].get_Qend()/MW,elec_net.links[2].get_Pend()/MW,elec_net.links[2].get_Qend()/MW,elec_net_mes.links[2].complex_power_loss().real/MW,elec_net_mes.links[2].complex_power_loss().imag/MW,elec_net.links[2].complex_power_loss().real/MW,elec_net.links[2].complex_power_loss().imag/MW))
            table.write(r'\hline ')
            table.write(r'\end{tabular}}%')
        with open(os.path.join(path_to_tables,'heat_hydr_data_EH_DD.txt'), "w") as table:
            table.write(r'\resizebox{\columnwidth}{!}{')
            table.write(r'\begin{tabular}{|c|c|c|c|c|c|c|c|c|} ')
            table.write(r'\hline ')
            table.write(r'\textbf{Node} & \multicolumn{2}{c|}{$h$~\sib{\meter}}  & \textbf{Half Link} & \multicolumn{2}{c|}{$m$~\sib{\kilo\gram \per \second}} & \textbf{Link} & \multicolumn{2}{c|}{$m$~\sib{\kilo\gram \per \second}}\\ ')
            table.write(r'\hline ')
            table.write(r' & (a) & (b) & & (a) & (b) &  & (a) & (b) \\ ')
            table.write(r'\hline ')
            table.write(r'0 & {:.3f} & {:.3f} & 0 & - & {:.3f} & 0-1  & {:.3f} & {:.3f}\\'.format(heat_net_mes.nodes[0].get_p()/(rho_w*grav_const),heat_net.nodes[0].get_p()/(rho_w*grav_const),heat_net.nodes[0].half_links[0].get_m(),heat_net_mes.links[0].get_m(),heat_net.links[0].get_m()))
            table.write(r'1 & {:.3f} & {:.3f} & 0 & {:.3f} & {:.3f} & 0-2  & {:.3f} & {:.3f}\\'.format(heat_net_mes.nodes[1].get_p()/(rho_w*grav_const),heat_net.nodes[1].get_p()/(rho_w*grav_const),heat_net_mes.nodes[1].half_links[0].get_m(),heat_net.nodes[1].half_links[0].get_m(),heat_net_mes.links[1].get_m(),heat_net.links[1].get_m()))
            table.write(r'2 & {:.3f} & {:.3f} & 0 & {:.3f} & {:.3f} & 1-2  & {:.3f} & {:.3f}\\'.format(heat_net_mes.nodes[2].get_p()/(rho_w*grav_const),heat_net.nodes[2].get_p()/(rho_w*grav_const),heat_net_mes.nodes[2].half_links[0].get_m(),heat_net.nodes[2].half_links[0].get_m(),heat_net_mes.links[2].get_m(),heat_net.links[2].get_m()))
            table.write(r'3 & - & {:.3f} & 0 & - & {:.3f} & 2-3  & - & {:.3f}\\'.format(heat_net.nodes[3].get_p()/(rho_w*grav_const),heat_net.nodes[3].half_links[0].get_m(),heat_net.links[3].get_m()))
            table.write(r'\hline ')
            table.write(r'\end{tabular}}%')
        with open(os.path.join(path_to_tables,'heat_therm_data_EH_DD.txt'), "w") as table:
            table.write(r'\resizebox{\columnwidth}{!}{')
            table.write(r'\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|} ')
            table.write(r'\hline ')
            table.write(r'\textbf{Node} & \multicolumn{2}{c|}{$T^s$~\sib{\celsius}} & \multicolumn{2}{c|}{$T^r$~\sib{\celsius}} & \textbf{Half Link} & \multicolumn{2}{c|}{$\Delta\varphi$~\sib{\mega \watt}} & \multicolumn{2}{c|}{$T^s$~\sib{\celsius}} & \multicolumn{2}{c|}{$T^r$~\sib{\celsius}} & \textbf{Link} & \multicolumn{2}{c|}{$\varphi^\text{loss}$~\sib{\mega \watt}}\\ ')
            table.write(r'\hline ')
            table.write(r' & (a) & (b) & (a) & (b) & & (a) & (b) & (a) & (b) & (a) & (b) & & (a) & (b) \\ ')
            table.write(r'\hline ')
            table.write(r'0 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & 0 & - & {:.3f} & - & {:.3f} & - & {:.3f} & 0-1 & {:.3f} & {:.3f}\\'.format(heat_net_mes.nodes[0].get_Ts(),heat_net.nodes[0].get_Ts(),heat_net_mes.nodes[0].get_Tr(),heat_net.nodes[0].get_Tr(),heat_net.nodes[0].half_links[0].get_dphi()/MW,heat_net.nodes[0].half_links[0].get_Ts(),heat_net.nodes[0].half_links[0].get_Tr(),heat_net_mes.links[0].heat_loss_supply()/MW+heat_net_mes.links[0].heat_loss_return()/MW,heat_net.links[0].heat_loss_supply()/MW+heat_net.links[0].heat_loss_return()/MW))
            table.write(r'1 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & 0 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & 0-2 & {:.3f} & {:.3f}\\'.format(heat_net_mes.nodes[1].get_Ts(),heat_net.nodes[1].get_Ts(),heat_net_mes.nodes[1].get_Tr(),heat_net.nodes[1].get_Tr(),heat_net_mes.nodes[1].half_links[0].get_dphi()/MW,heat_net.nodes[1].half_links[0].get_dphi()/MW,heat_net_mes.nodes[1].half_links[0].get_Ts(),heat_net.nodes[1].half_links[0].get_Ts(),heat_net_mes.nodes[1].half_links[0].get_Tr(),heat_net.nodes[1].half_links[0].get_Tr(),heat_net_mes.links[1].heat_loss_supply()/MW+heat_net_mes.links[1].heat_loss_return()/MW,heat_net.links[1].heat_loss_supply()/MW+heat_net.links[1].heat_loss_return()/MW))
            table.write(r'2 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & 0 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & 1-2 & {:.3f} & {:.3f}\\'.format(heat_net_mes.nodes[2].get_Ts(),heat_net.nodes[2].get_Ts(),heat_net_mes.nodes[2].get_Tr(),heat_net.nodes[2].get_Tr(),heat_net_mes.nodes[2].half_links[0].get_dphi()/MW,heat_net.nodes[2].half_links[0].get_dphi()/MW,heat_net_mes.nodes[2].half_links[0].get_Ts(),heat_net.nodes[2].half_links[0].get_Ts(),heat_net_mes.nodes[2].half_links[0].get_Tr(),heat_net.nodes[2].half_links[0].get_Tr(),heat_net_mes.links[2].heat_loss_supply()/MW+heat_net_mes.links[2].heat_loss_return()/MW,heat_net.links[2].heat_loss_supply()/MW+heat_net.links[2].heat_loss_return()/MW))
            table.write(r'3 & - & {:.3f} & - & {:.3f} & 0 & - & {:.3f} & - & {:.3f} & - & {:.3f} & 2-3 & - & {:.3f}\\'.format(heat_net.nodes[3].get_Ts(),heat_net.nodes[3].get_Tr(),heat_net.nodes[3].half_links[0].get_dphi()/MW,heat_net.nodes[3].half_links[0].get_Ts(),heat_net.nodes[3].half_links[0].get_Tr(),heat_net.links[3].heat_loss_supply()/MW+heat_net.links[3].heat_loss_return()/MW))
            table.write(r'\hline ')
            table.write(r'\end{tabular}}%')
        with open(os.path.join(path_to_tables,'coupling_data_EH_DD.txt'), "w") as table:
            table.write(r'\resizebox{\columnwidth}{!}{')
            table.write(r'\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|} ')
            table.write(r'\hline ')
            table.write(r'\textbf{Unit} & \multicolumn{2}{c|}{$q$~\SIB{e3}{\cubic\meter \per \hour}} & \multicolumn{2}{c|}{$P$~\sib{\mega\watt}}& \multicolumn{2}{c|}{$Q$~\sib{\mega\var}} & \multicolumn{2}{c|}{$m^\text{inj}$~\sib{\kilo\gram \per \second}} & \multicolumn{2}{c|}{$\varphi$~\sib{\mega \watt}} & \multicolumn{2}{c|}{$T^o$~\sib{\celsius}} \\ ')
            table.write(r'\hline ')
            table.write(r' & (a) & (b) & (a) & (b) & (a) & (b) & (a) & (b) & (a) & (b) & (a) & (b)\\ ')
            table.write(r'\hline ')
            table.write(r' EH 0 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {}\\ '.format(gas_net_mes.links[4].get_q()/(MSCM*rhon_g/hour),coupling_net.nodes[0].half_links[1].get_q()/(MSCM*rhon_g/hour),elec_net_mes.links[3].get_Pstart()/MW,coupling_net.nodes[0].half_links[2].get_P()/MW,elec_net_mes.links[3].get_Qstart()/MW,coupling_net.nodes[0].half_links[2].get_Q()/MW,heat_net_mes.links[3].get_m(),coupling_net.nodes[0].half_links[0].get_m(),-heat_net_mes.links[3].get_dphistart()/MW,coupling_net.nodes[0].half_links[0].get_dphi()/MW,heat_net_mes.links[3].get_Tsstart(),coupling_net.nodes[0].half_links[0].get_Ts()))
            table.write(r' EH 1 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {}\\ '.format(gas_net_mes.links[5].get_q()/(MSCM*rhon_g/hour),coupling_net.nodes[1].half_links[1].get_q()/(MSCM*rhon_g/hour),elec_net_mes.links[4].get_Pstart()/MW,coupling_net.nodes[1].half_links[2].get_P()/MW,elec_net_mes.links[4].get_Qstart()/MW,coupling_net.nodes[1].half_links[2].get_Q()/MW,heat_net_mes.links[4].get_m(),coupling_net.nodes[1].half_links[0].get_m(),-heat_net_mes.links[4].get_dphistart()/MW,coupling_net.nodes[1].half_links[0].get_dphi()/MW,heat_net_mes.links[4].get_Tsstart(),coupling_net.nodes[1].half_links[0].get_Ts()))
            table.write(r'\hline ')
            table.write(r'\end{tabular}}%')
    else:
        raise ValueError('Enter a valid path to save the tables.')

if __name__== '__main__':
    tol = 1e-6
    gas_net,x_sol_g,iters_g,err_vec_g = run_gas_load_flow(tol=tol)
    elec_net,x_sol_e,iters_e,err_vec_e = run_electrical_load_flow(tol=tol)
    heat_net,x_sol_h,iters_h,err_vec_h = run_heat_load_flow(tol=tol)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "No single-carrier subnetworks found",UserWarning)
        coupling_net, x_sol_c, iters_c, err_vec_c = run_coupling_load_flow(tol=tol)
    max_iters_used  = max([iters_g,iters_e,iters_h,iters_c])

    fig = plt.figure('Convergence plot single-carrier')
    ax_conv = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F(x^k)||_2$')
    ax_conv.semilogy(err_vec_g,marker='.',color='tab:green',label='gas')
    ax_conv.semilogy(err_vec_e,marker='.',color='tab:red',label='electricity')
    ax_conv.semilogy(err_vec_h,marker='.',color='tab:blue',label='heat')
    ax_conv.semilogy(err_vec_c,marker='.',color='tab:gray',label='coupling')
    layout_convergence(ax_conv,tol,max_iters_used)

    fig_ord = plt.figure('Convergence order of NR single-carrier')
    ax_ord = fig_ord.gca()
    plt.xlabel(r'$||F(x^k)||_2 / ||F(x^0)||_2$')
    plt.ylabel(r'$||F(x^{k+1})||_2 / ||F(x^0)||_2$')
    ax_ord.loglog(err_vec_g[:-1]/err_vec_g[0],err_vec_g[1:]/err_vec_g[0],marker='.',color='tab:green',label='gas')
    ax_ord.loglog(err_vec_e[:-1]/err_vec_e[0],err_vec_e[1:]/err_vec_e[0],marker='.',color='tab:red',label='electricity')
    ax_ord.loglog(err_vec_h[:-1]/err_vec_h[0],err_vec_h[1:]/err_vec_h[0],marker='.',color='tab:blue',label='heat')
    ax_ord.loglog(err_vec_c[:-1]/err_vec_c[0],err_vec_c[1:]/err_vec_c[0],marker='.',color='tab:gray',label='coupling')
    layout_convergence_order(ax_ord)

    pg0_ic = 55*bar
    pg1_ic = 45*bar
    pg3_ic = 45*bar
    q_ic = 20*MSCM*rhon_g/hour
    qc_ic = 15*MSCM*rhon_g/hour
    Pc_ic = 20*MW
    phic_ic = 25*MW
    Tsc_ic = 125.
    hydr_eq_gas = 'fa'
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    max_iter_outer = 600
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_tables = os.path.join(dir_path,'network_data','N_Shabanpour')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "No single-carrier subnetworks found",UserWarning)
        compare_DD(hydr_eq_gas=hydr_eq_gas,pg0_ic=pg0_ic,pg1_ic=pg1_ic,pg3_ic=pg3_ic,q_ic=q_ic,qc_ic=qc_ic,Pc_ic=Pc_ic,phic_ic=phic_ic,Tsc_ic=Tsc_ic,max_iter_outer=max_iter_outer,formulation=formulation,save_tables=False,path_to_tables=path_to_tables)
    plt.show()
