"""Make network, with one solution scenario, for the multi-carrier network of the Benchmark Problem (BP), topology 1.

The solution is determined using full formulation and link pressure drops as function of flow in gas, complex power formulation in electricity, and unknown half link flow formulation and known outflow temperature for loads. Tolerance of NR is 1e-6, and no scaling is used."""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.carrier import Gas
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.utils.constants import bar, mbar, hour, mm, MW, MBTU, BTU, kV
import numpy as np
from meslf.networks.read_write_network import from_pd_dataframes, to_pd_dataframes
import os
import pandas as pd
import warnings
import scipy.sparse as sps
import matplotlib.pyplot as plt

# Read the scenario data
path_to_data = os.path.dirname(os.path.realpath(__file__))

nodes_gas = pd.read_pickle(os.path.join(path_to_data, 'GN_BP_nodes.pkl'))
links_gas = pd.read_pickle(os.path.join(path_to_data, 'GN_BP_links.pkl'))
halflinks_gas = pd.read_pickle(os.path.join(path_to_data, 'GN_BP_halflinks.pkl'))
gas_net_scen = from_pd_dataframes(nodes_gas,links_gas,halflinks_gas)

nodes_elec = pd.read_pickle(os.path.join(path_to_data, 'EN_BP_nodes.pkl'))
links_elec = pd.read_pickle(os.path.join(path_to_data, 'EN_BP_links.pkl'))
halflinks_elec = pd.read_pickle(os.path.join(path_to_data, 'EN_BP_halflinks.pkl'))
elec_net_scen = from_pd_dataframes(nodes_elec,links_elec,halflinks_elec)

nodes_heat = pd.read_pickle(os.path.join(path_to_data, 'HN_BP_nodes.pkl'))
links_heat = pd.read_pickle(os.path.join(path_to_data, 'HN_BP_links.pkl'))
halflinks_heat = pd.read_pickle(os.path.join(path_to_data, 'HN_BP_halflinks.pkl'))
heat_net_scen = from_pd_dataframes(nodes_heat,links_heat,halflinks_heat)

# (full) solution of scenario
q_scen = np.zeros(len(gas_net_scen.links))
q_hl_scen = np.zeros(len(gas_net_scen.half_links))
pg_scen = np.zeros(len(gas_net_scen.nodes))
for ind_e,link in enumerate(gas_net_scen.get_links()):
    q_scen[ind_e] = link.get_q()
for ind_n,node in enumerate(gas_net_scen.get_nodes()):
    pg_scen[ind_n] = node.get_p()
for ind_hl,half_link in enumerate(gas_net_scen.get_half_links()):
    q_hl_scen[ind_hl] = half_link.get_q()

elec_net_scen.initialize()
x_scen = elec_net_scen.set_x_init()
elec_net_scen.update_full(x_scen)
delta_scen  = np.zeros(len(elec_net_scen.nodes))
V_scen  = np.zeros(len(elec_net_scen.nodes))
P_inj_scen  = [hl.P for node in elec_net_scen.get_nodes() for hl in node.get_half_links()]
Q_inj_scen  = [hl.Q for node in elec_net_scen.get_nodes() for hl in node.get_half_links()]
P_edge_scen = np.zeros(2*len(elec_net_scen.links))
Q_edge_scen = np.zeros(2*len(elec_net_scen.links))
for ind_e,link in enumerate(elec_net_scen.get_links()):
    P_edge_scen[ind_e] = link.get_Pstart()
    P_edge_scen[ind_e+len(elec_net_scen.links)] =link.get_Pend()
    Q_edge_scen[ind_e] = link.get_Qstart()
    Q_edge_scen[ind_e+len(elec_net_scen.links)] =link.get_Qend()
for ind_n,node in enumerate(elec_net_scen.get_nodes()):
    delta_scen[ind_n] = node.get_delta()
    V_scen[ind_n] = node.get_V()

m_scen = np.zeros(len(heat_net_scen.links))
m_hl_scen = np.zeros(len(heat_net_scen.half_links))
ph_scen = np.zeros(len(heat_net_scen.nodes))
Ts_scen = np.zeros(len(heat_net_scen.nodes))
Tr_scen = np.zeros(len(heat_net_scen.nodes))
Ts_hl_scen = np.zeros(len(heat_net_scen.half_links))
Tr_hl_scen = np.zeros(len(heat_net_scen.half_links))
phi_scen = np.zeros(len(heat_net_scen.half_links))
for ind_e,link in enumerate(heat_net_scen.get_links()):
    m_scen[ind_e] = link.m
for ind_n,node in enumerate(heat_net_scen.get_nodes()):
    ph_scen[ind_n] = node.p
    Ts_scen[ind_n] = node.Ts
    Tr_scen[ind_n] = node.Tr
for ind_hl,half_link in enumerate(heat_net_scen.get_half_links()):
    m_hl_scen[ind_hl] = half_link.m
    Ts_hl_scen[ind_hl] = half_link.Ts
    Tr_hl_scen[ind_hl] = half_link.Tr
    phi_scen[ind_hl] = half_link.dphi

# physical parameters of network
gas = gas_net_scen.links[0].link_params.get('carrier')
rhon_g = gas.rhon #[kg/m^3]
gas_link_params = gas_net_scen.links[0].link_params.copy()
gas_link_type = gas_net_scen.links[0].link_type
link_eq = 'dp_of_q'

elec_link_params = elec_net_scen.links[0].link_params.copy()
elec_link_type = elec_net_scen.links[0].link_type

Ta = heat_net_scen.links[0].link_params.get('Ta')
heat_net_scen.Ta = Ta
water = heat_net_scen.links[0].link_params.get('carrier')
rho_w = water.rhon #[kg/m^3]
grav_const = water.g #[m/s^2]
heat_link_params = heat_net_scen.links[0].link_params.copy()
heat_link_type = heat_net_scen.links[0].link_type

GHV = 40.611 #[MBTU/m^3]
GHV *= MBTU*BTU #[Wh/m^3]
GHV *= hour/rhon_g #[J/kg]

# coordinates
xc0 = 6
yc0 = 11
xc1 = 6
yc1 = 6

# Coupling parameters (based on desired coupling energies)
phi_c = -phi_scen[0] #[W] All the input of the heat network is replaced by the coupling
q_c = .7*q_hl_scen[2] # part of the load of node 2 is redirected to the coupling
Eg_c = GHV*q_c
q2_load = q_hl_scen[2] - q_c
P1_load = 1*MW #[W]
P_c = abs(P_inj_scen[1] - P1_load)
Q1_load = 1*MW #[W]
Q_c = abs(Q_inj_scen[1] - Q1_load)
# divide coupling flow over the 2 coupling nodes
To_GB = 1.1*Ts_scen[0]
To_CHP = .9*Ts_scen[0]
phi_c_GB = phi_c * (To_GB-Tr_scen[0])/(Ts_scen[0]-Tr_scen[0]) * (To_CHP - Ts_scen[0])/(To_CHP - To_GB)
phi_c_CHP = phi_c * (To_CHP-Tr_scen[0])/(Ts_scen[0]-Tr_scen[0]) * (Ts_scen[0] - To_GB)/(To_CHP - To_GB)
if not np.isclose(phi_c,phi_c_GB+phi_c_CHP):
    raise ValueError('Wrong values for coupling heat powers: the separate heat powers do not add up to total heat power')
eta_CHP = np.array([2*P_c/Eg_c, 2*phi_c/Eg_c])
eta_GB = eta_CHP[1]
q_c_GB = phi_c_GB/(eta_GB*GHV)
q_c_CHP = 1/GHV * (P_c/eta_CHP[0] + phi_c_CHP/eta_CHP[1])
if not np.isclose(q_c,q_c_GB+q_c_CHP):
    raise ValueError('Wrong values for coupling gas flows: the separate gas flows do not add up to total gas flow')
print('coupling parameters:')
print('q_c = {} kg/s'.format(q_c))
print('q_c GB = {} kg/s'.format(q_c_GB))
print('q_c CHP = {} kg/s'.format(q_c_CHP))
print('Eg_c = {} W'.format(Eg_c))
print('P_c = {} W'.format(P_c))
print('Q_c = {} W'.format(Q_c))
print('P1_load = {} W'.format(P1_load))
print('Q1_load = {} W'.format(Q1_load))
print('phi_c = {} W'.format(phi_c))
print('phi_c_GB = {} W'.format(phi_c_GB))
print('phi_c_CHP = {} W'.format(phi_c_CHP))
print('eta_CHP = {}'.format(eta_CHP))
print('eta_GB = {}'.format(eta_GB))
unit_type_GB='gh_gas_boiler'
unit_params_GB={'eta':eta_GB,'GHV':GHV}
unit_type_CHP='geh_CHP'
unit_params_CHP={'eta':eta_CHP,'GHV':GHV}

# solver information
formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
tol=1e-6
max_iter=50

def create_network():
    """Create a multi-carrier network consisting of 3 demand/source nodes per carrier, coupled through an EH."""
    # gas network
    g0 = GasNode('gn0',node_type=0,x=gas_net_scen.nodes[0].x,y=gas_net_scen.nodes[0].y,p=pg_scen[0]) # reference node
    g1 = GasNode('gn1',node_type=1,x=gas_net_scen.nodes[1].x,y=gas_net_scen.nodes[1].y,q=q_hl_scen[1]) # load node
    g2 = GasNode('gn2',node_type=1,x=gas_net_scen.nodes[2].x,y=gas_net_scen.nodes[2].y,q=q2_load) # load node

    gl0 = GasLink('gl0',g0,g1,link_type=gas_link_type,link_params=gas_link_params,link_eq_form=link_eq)
    gl1 = GasLink('gl1',g0,g2,link_type=gas_link_type,link_params=gas_link_params.copy(),link_eq_form=link_eq)
    gl2 = GasLink('gl2',g1,g2,link_type=gas_link_type,link_params=gas_link_params.copy(),link_eq_form=link_eq)

    gas_net = GasNetwork('3 nodes gas')
    gas_net.add_link(gl0)
    gas_net.add_link(gl1)
    gas_net.add_link(gl2)

    # electrical network
    e0 = ElectricalNode('en0',node_type=1,x=elec_net_scen.nodes[0].x,y=elec_net_scen.nodes[0].y,P=P_inj_scen[0],V=V_scen[0]) # gen
    e1 = ElectricalNode('en1',node_type=5,x=elec_net_scen.nodes[1].x,y=elec_net_scen.nodes[1].y,P=P1_load,Q=Q1_load,V=V_scen[1],delta=delta_scen[1]) # PQVdelta
    e2 = ElectricalNode('en2',node_type=2,x=elec_net_scen.nodes[2].x,y=elec_net_scen.nodes[2].y,P=P_inj_scen[2],Q=Q_inj_scen[2]) # load

    el0 = ElectricalLink('el0',e0,e1,link_type=elec_link_type,link_params=elec_link_params)
    el1 = ElectricalLink('el1',e0,e2,link_type=elec_link_type,link_params=elec_link_params.copy())
    el2 = ElectricalLink('el2',e1,e2,link_type=elec_link_type,link_params=elec_link_params.copy())

    elec_net = ElectricalNetwork('3 nodes elec')
    elec_net.add_link(el0)
    elec_net.add_link(el1)
    elec_net.add_link(el2)

    # heat network
    h0 = HeatNode('hn0',node_type=7,x=heat_net_scen.nodes[0].x,y=heat_net_scen.nodes[0].y,p=ph_scen[0],Ts=Ts_scen[0]) # reference temperature junction
    h1 = HeatNode('hn1',node_type=1,x=heat_net_scen.nodes[1].x,y=heat_net_scen.nodes[1].y,Tr_hl=Tr_hl_scen[1],dphi=phi_scen[1]) # load node (sink)
    h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h2 = HeatNode('hn2',node_type=1,x=heat_net_scen.nodes[2].x,y=heat_net_scen.nodes[2].y,Tr_hl=Tr_hl_scen[2],dphi=phi_scen[2]) # load  node (sink)
    h2.half_links[0].set_type('heat_exchanger',{'carrier':water})

    hl0 = HeatLink('hl0',h0,h1,link_type=heat_link_type,link_params=heat_link_params)
    hl1 = HeatLink('hl1',h0,h2,link_type=heat_link_type,link_params=heat_link_params.copy())
    hl2 = HeatLink('hl2',h1,h2,link_type=heat_link_type,link_params=heat_link_params.copy())

    heat_net = HeatNetwork('3 nodes heat',Ta=Ta)
    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)

    # coupling node
    cn0 = HeterogeneousNode('cn0',node_type=1,x=xc0,y=yc0,unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
    cn1 = HeterogeneousNode('cn1',node_type=1,x=xc1,y=yc1,unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To known

    # coupling links
    glc0 = GasLink('gl_c',g2,cn0)
    glc1 = GasLink('gl_c',g2,cn1)
    elc = ElectricalLink('el_c',cn1,e1)
    hlc0 = HeatLink('hl_c',cn0,h0,link_params={'carrier':water},bc_type=6,Tsstart=To_GB) # To of coupling (source) is known
    hlc1 = HeatLink('hl_c',cn1,h0,link_params={'carrier':water},bc_type=6,Tsstart=To_CHP) # To of coupling (source) is known
    gas_net.add_link(glc0)
    gas_net.add_link(glc1)
    elec_net.add_link(elc)
    heat_net.add_link(hlc0)
    heat_net.add_link(hlc1)

    # create MES and add coupling links and node
    het_net = HeterogeneousNetwork('3 nodes mes')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    het_net.add_node(cn0)
    het_net.add_node(cn1)

    return gas_net,elec_net,heat_net,het_net

def initialize_network(gas_net,elec_net,heat_net,het_net):
    """Initialize the multi-carrier network, consisting of 3 demand/source nodes per single-carrier network.

    Parameters
    ----------
    gas_net : GasNetwork
        The gas network to be initialized
    elec_net : ElectricalNetwork
        The electrical network to be initialized
    heat_net : HeatNetwork
        The heat network to be initialized
    het_net : HeterogeneousNetwork
        The gas network to be initialized

    Returns
    -------
    x0 : np array
        initial guess
    """
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks= het_net.get_x_entries(formulation=formulation)

    # gas part
    q_init = .05*np.ones(len(gas_net.links)-2) #-2, because of coupling links
    pg_init = np.array([29,28])*mbar
    xg_init = np.concatenate((q_init,pg_init))

    # electrical part
    delta_init = np.zeros(len(unknown_delta_nodes))
    V_init = V_scen[2]*np.ones(len(unknown_V_nodes))
    xe_init = np.concatenate((delta_init,V_init))

    m_init = np.array([6,6,1]) #[kg/s]
    m_hl_init = np.array([5,7]) #[kg/s]
    ph_init = np.array([99.5,99.4])*rho_w*grav_const #[Pa]
    Ts_init = np.array([99.6,99.3]) #[C]
    Tr_init = np.array([49.7,49.8,50.]) #[C]
    xh_init = np.concatenate((m_init,m_hl_init,ph_init,Ts_init,Tr_init))

    # coupling
    qc_init = q_c/2*np.ones(len(unknown_qc_links))
    Pc_init = P_c*np.ones(len(unknown_Pc_links))
    Qc_init = Q_c*np.zeros(len(unknown_Qc_links))
    Sc_init = np.concatenate((Pc_init,Qc_init))
    mc_init = 6*np.ones(len(unknown_mc_links))
    phic_init = phi_c/2*np.ones(len(unknown_dphi_links))
    Toc_init = Ts_scen[0]*np.ones(len(unknown_Ts_links))
    xc_init = np.concatenate((qc_init,Sc_init,mc_init,phic_init,Toc_init))

    # combine into multi-carrier
    x_init = np.concatenate((xg_init,xe_init,xh_init,xc_init))

    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)
    return x0

def run_load_flow():
    """Stead-state load flow analysis of gas network
    """
    # create network
    gas_net,elec_net,heat_net,het_net = create_network()
    # initialize
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net)

    # solve network
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)

    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    print('delta = {}'.format(delta_vec))
    Vbase = 10/np.sqrt(3)*kV #[V]
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('elec hl start nodes = {}'.format([hl.start_node.name for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Ts hl = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Tr hl = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Ts c = {} C'.format(Tsc_vec))
    print('Tr c = {} C'.format(Trc_vec))
    print('dphi c = {} MW'.format([phi/MW for phi in phic_vec]))
    print('m c = {} kg/s'.format(mc_vec))

    return gas_net,elec_net,heat_net,het_net,x_sol

def save_scenario():
    """Save the network scenario data"""
    # create network and run load flow
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        gas_net,elec_net,heat_net,het_net,x_sol = run_load_flow()

    fig_top = plt.figure('Network topology')
    ax_top = fig_top.gca()
    het_net.draw_network(ax_top,halflink_angle=2,halflink_length=1)
    plt.axis('equal')
    plt.axis('off')

    # make pd df
    for node in gas_net.get_nodes():
        for hl in node.get_half_links():
            het_net.add_half_link(hl)
    for node in elec_net.get_nodes():
        for hl in node.get_half_links():
            het_net.add_half_link(hl)
    for node in heat_net.get_nodes():
        for hl in node.get_half_links():
            het_net.add_half_link(hl)
    nodes, links, halflinks = to_pd_dataframes(het_net)

    # save data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    nodes.to_pickle(os.path.join(dir_path,'MES_BP_nodes_top1_2c.pkl'))
    links.to_pickle(os.path.join(dir_path,'MES_BP_links_top1_2c.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'MES_BP_halflinks_top1_2c.pkl'))

def table():
    """Write the full solution to a txt file, which can be written by latex to create a table."""
    # create network and run load flow
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        gas_net,elec_net,heat_net,het_net,x_sol = run_load_flow()

    path_to_data = os.path.dirname(os.path.realpath(__file__))
    coupling_nodes = [node for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode)]
    with open(os.path.join(path_to_data,'coupling_data_BP_top1_2c.txt'), "w") as table:
        table.write(r'\multirow{2}{*}{Top. 1}' )
        table.write(r' & GB & {:.3f} & - & - & {:.3f} & {:.3f} & {:.3f} \\ '.format(gas_net.links[3].get_q(),heat_net.links[3].get_m(),-heat_net.links[3].get_dphistart()/MW,heat_net.links[3].get_Tsstart()))
        table.write(r' & CHP & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\ '.format(gas_net.links[4].get_q(),elec_net.links[3].get_Pstart()/MW,elec_net.links[3].get_Qstart()/MW,heat_net.links[4].get_m(),-heat_net.links[4].get_dphistart()/MW,heat_net.links[4].get_Tsstart()))
        table.write(r'\hline ')

    with open(os.path.join(path_to_data,'coupling_params_BP_top1_2c.txt'), "w") as table:
        table.write(r'\multirow{2}{*}{\acrshort{gb}} & \glssymbol{efficiency} & & ')
        table.write(r' & '+'{:.2f}'.format(eta_GB)+ ' & & ')
        table.write(r'\multirow{2}{*}{\acrshort{chp}} & \glssymbol{efficiency}^{\glssymbol{sup:gas}\glssymbol{sup:elec}} & \glssymbol{efficiency}^{\glssymbol{sup:gas}\glssymbol{sup:heat}} & ')
        table.write(r' & '+'{:.2f}'.format(eta_CHP[0])+ ' & '+'{:.2f}'.format(eta_CHP[1])+ ' & ')

if __name__== '__main__':
    # save_scenario()
    table()
    # plt.show()
