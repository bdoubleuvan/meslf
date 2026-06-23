"""Make network, with one solution scenario, for the part of the Benchmark Problem (BP).

Tolerance of NR is 1e-6, and no scaling is used."""
from meslf.networks.gas_network import GasHalfLink
from meslf.networks.electrical_network import ElectricalHalfLink
from meslf.networks.heat_network import HeatHalfLink
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.utils.constants import MW, MBTU, BTU, hour
import numpy as np
from meslf.networks.read_write_network import from_pd_dataframes, to_pd_dataframes
import os
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# Read the scenario data (of the single-carrier parts)
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

# (required part of) solution of scenario
q_hl_scen = np.zeros(len(gas_net_scen.half_links))
for ind_hl,half_link in enumerate(gas_net_scen.get_half_links()):
    q_hl_scen[ind_hl] = half_link.get_q()
    
elec_net_scen.initialize()
x_scen = elec_net_scen.set_x_init()
elec_net_scen.update_full(x_scen)
V_scen  = np.zeros(len(elec_net_scen.nodes))
P_inj_scen  = [hl.P for node in elec_net_scen.get_nodes() for hl in node.get_half_links()]
Q_inj_scen  = [hl.Q for node in elec_net_scen.get_nodes() for hl in node.get_half_links()]

Ts_scen = np.zeros(len(heat_net_scen.nodes))
Tr_scen = np.zeros(len(heat_net_scen.nodes))
phi_scen = np.zeros(len(heat_net_scen.half_links))
for ind_n,node in enumerate(heat_net_scen.get_nodes()):
    Ts_scen[ind_n] = node.Ts
    Tr_scen[ind_n] = node.Tr
for ind_hl,half_link in enumerate(heat_net_scen.get_half_links()):
    phi_scen[ind_hl] = half_link.dphi
    
# physical parameters of network
gas = gas_net_scen.links[0].link_params.get('carrier') 
rhon_g = gas.rhon #[kg/m^3]

water = heat_net_scen.links[0].link_params.get('carrier')

GHV = 40.611 #[MBTU/m^3]
GHV *= MBTU*BTU #[Wh/m^3]
GHV *= hour/rhon_g #[J/kg]

# coordinates
xc0 = 6
yc0 = 8.5

# Coupling parameters (based on desired coupling energies)
phi_c = -phi_scen[0] #[W] All the input of the heat network is replaced by the coupling
P1_load = 1*MW #[W]
P_c = abs(P_inj_scen[1] - P1_load)
Q1_load = 1*MW #[W]
Q_c = abs(Q_inj_scen[1] - Q1_load)
q_c = .7*q_hl_scen[2] # part of the load of node 2 is redirected to the coupling
q2_load = q_hl_scen[2] - q_c
Eg_c = GHV*q_c

C_EH = np.array([[P_c/Eg_c],[phi_c/Eg_c]])
print('coupling parameters:')
print('q_c = {} kg/s'.format(q_c))
print('Eg_c = {} W'.format(Eg_c))
print('P_c = {} W'.format(P_c))
print('Q_c = {} W'.format(Q_c))
print('P1_load = {} W'.format(P1_load))
print('Q1_load = {} W'.format(Q1_load))
print('phi_c = {} W'.format(phi_c))
print('C = {}'.format(C_EH))
unit_params={'C':C_EH,'GHV':GHV}
To_EH = Ts_scen[0]
Tr_EH = Tr_scen[0]

# solver information
formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
tol=1e-6
max_iter=50

def create_network():
    """Create the coupling part of the network, consisting of an EH."""
    
    cn = HeterogeneousNode('cn',node_type=1,x=xc0,y=yc0,unit_type='EH',unit_params=unit_params) # To known
    hlh = HeatHalfLink('cn_hlh',cn,link_type='heat_exchanger',link_params={'carrier':water},bc_type=2,Ts=To_EH,dphi=-phi_c) # Ts and dphi known, source
    hlh.Tr = Tr_EH # Set to solution
    hlg = GasHalfLink('cn_hlg',cn,-q_c,bc_type=0) # gas flows into coupling node, q is unknown
    hle = ElectricalHalfLink('cn_hle',cn,Q=Q_c,bc_type=3) # P is unknown, Q is known
    het_net = HeterogeneousNetwork('EH')
    het_net.add_node(cn)
    
    return het_net

def initialize_network(het_net):
    """Initialize the coupling part of the network, consisting of an EH.
    
    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The heterogeneous network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    x_init = np.array([.05, 1*MW, 2]) # q, P, m
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)
    return x0    

def run_load_flow():
    """Stead-state load flow analysis of gas network    
    """
    # create network
    het_net = create_network()
    
    # initialize
    x0 = initialize_network(het_net)
    
    # solve network
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
    
    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    cn = het_net.nodes[0]
    hlh = cn.half_links[0]
    hlg = cn.half_links[1]
    hle = cn.half_links[2]
    print('q = {} kg/s'.format(hlg.get_q()))
    print('P = {} MW'.format(hle.get_P()/MW))
    print('Q = {} MW'.format(hle.get_Q()/MW))
    print('m = {} kg/s'.format(hlh.get_m()))
    print('phi = {} MW'.format(hlh.get_dphi()/MW))
    print('To = {} C'.format(hlh.get_Ts()))
    
    # plot topology
    fig_top = plt.figure('Network topology')
    ax_top = fig_top.gca()
    het_net.draw_network(ax_top,halflink_angle=2,halflink_length=1)
    plt.axis('equal')
    plt.axis('off')
    
    return het_net,x_sol
    
if __name__== '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "No single-carrier subnetworks found",UserWarning)
        run_load_flow()
    plt.show()
    
    
    
    
