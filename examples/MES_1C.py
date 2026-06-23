"""Network consisting of one city, based on the N_1C data."""
import os.path
from examples.network_data.N_1C import make_MES_1C
from examples.GN_1C import solve_GN_1C
from examples.EN_1C import solve_EN_1C
from examples.HN_1C import solve_HN_1C
from meslf.networks.read_write_network import from_pd_dataframes
from meslf.networks.gas_network import GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNode, HeatLink, HeatHalfLink
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
import matplotlib.pyplot as plt
import argparse 
import pandas as pd
import numpy as np
import scipy.sparse as sps
from meslf.utils.constants import hour, mbar, bar, kV, MW, kW, MBTU, BTU
import warnings

command_line_input = argparse.ArgumentParser()
command_line_input.add_argument(
    "-ex", # Which examples / cases to run. 'single_CHP' is one CHP, 'multi' is multiple CHP with slack nodes as city source nodes, 'multi2' is multiple CHP with source nodes as electrical and heat city source nodes.
    nargs = "*", # 0 or more values expected => creates a list
    type=str,
    default = ['single_CHP', 'single_EH', 'multi_CHP', 'multi_CHP2', 'multi_CHP3', 'multi_EH1', 'multi_EH2', 'multi_EH3', 'CHP_node1_set1', 'CHP_node1_set2', 'EH_node1_set1', 'EH_node1_set2', 'CHP_node2_set1', 'CHP_node2_set2', 'CHP_node2_set3', 'EH_node2_set1', 'EH_node2_set2', 'EH_node2_set3'], # default if nothing is provided
    )
command_line_input.add_argument(
    "--n", # number of loads, for each street
    type=int,
    default = 0, # default if nothing is provided
    )
command_line_input.add_argument(
    "--m", # number of junctions with two loads, for each street
    type=int,
    default = 0, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nS", # number of street networks
    type=int,
    default = 0, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nQ", # number of quarter networks
    type=int,
    default = 0, # default if nothing is provided
    )
command_line_input.add_argument(
    "--nD", # number of district networks
    type=int,
    default = 0, # default if nothing is provided
    )
command_line_input.add_argument(
    "--Ta", # Ambient temperature of the heat network
    type=int,
    default = 0, # default if nothing is provided
    )
command_line_input.add_argument(
    "--p_low", # Lowest value of fraction of reference pressure that is used in the initial linear pressure profile
    type=float,
    default = .98, # default if nothing is provided
    )
command_line_input.add_argument(
    "--p_high", # Highest value of fraction of reference pressure that is used in the initial linear pressure profile
    type=float,
    default = .99, # default if nothing is provided
    )
command_line_input.add_argument(
    "--max_nodes", # Maximum number of total nodes for which the topology is plotted, etc.
    type=int,
    default = 500, # default if nothing is provided
    )
command_line_input.add_argument(
    "--plot_all", # Maximum number of total nodes for which the topology is plotted, etc.
    type=bool,
    default = False, # default if nothing is provided
    )

# some values used throughout the entire script
form={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
tol = 1e-6
max_iter = 15

V_ref_C = 50*kV #[V]
V_ref_D = 10*kV #[V]
V_ref_Q = 10*kV #[V]
V_ref_S = .4*kV #[V]

delta_ref_Q = 0#-0.1

colors = {'CHP':'tab:blue','EH':'tab:orange'}
linestyles_gas = {'full fa':'--','full fb':'-','nodal fa':'-.'}
linestyles_heat = {'standard outflow':'-','standard delta':'--','half_link_flow outflow':'-.','half_link_flow delta':':'}
markers = {1:'o',2:'s',3:'*'}

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning   


def remake_network(n,m,nS,nQ,nD,Ta,p_init_low,p_init_high,heat_load='outflow',hydr_eq='dp_of_q'):
    """Recreate the pd dataframes for the single carrier networks.

    Parameters
    ----------
    n : int
        number of loads, for each street
    m : int
        number of junctions with two loads, for each street
    nS : int
        number of street networks
    nQ : int
        number of quarter networks
    nD : int
        number of district networks
    Ta : float
        Ambient temperature of the heat network
    p_init_low : float
        Lowest value of fraction of reference pressure that is used in the initial linear pressure profile.
    p_init_high : float
        Highest value of fraction of reference pressure that is used in the initial linear pressure profile.
    """
    if n and nS and nQ and nD: # m=0 is allowed
        print('Recreating the single-carrier networks with n={}, m={}, nS={}, nQ={}, and nD={}'.format(n,m,nS,nQ,nD))
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path_to_data = os.path.join(dir_path,'network_data','N_1C')
        make_MES_1C.create_single_carrier_networks(n,m,nS,nQ,nD,Ta,p_init_low,p_init_high,path_to_data,heat_load=heat_load,hydr_eq=hydr_eq)
    else:
        print('The existing data for the single-carrier networks is used.')
    
def create_network(coupling='single_CHP',heat_load='outflow',hydr_eq='dp_of_q'):
    """Create a test heterogeneous network with a single CHP as coupling node

    Parameters
    ----------
    coupling : str, optional
        Determines how the single carrier networks are coupled. Options are 'single_CHP', 'single_EH', 'multi_CHP', 'multi_CHP2', 'multi_CHP3', 'multi_EH1', 'multi_EH2', or 'multi_EH3'. The ones with single are coupled through a single coupling node connected at the city source of the single-carrier networkes. The ones with multi are coupled through multiple coupling nodes, connected at every quarter source node of the single-carrier networks. Default is 'single_CHP'.
        
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
    nC : int
        number of city networks
    nD : int
        number of district networks
    nQ : int
        number of quarter networks
    nS : int
        number of street networks
    n : int
        number of loads, for each street
    m : int
        number of junctions with two loads, for each street
    """
    print("\nCreating network with coupling {}".format(coupling))
    # create network from data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','N_1C')

    gas_nodes = pd.read_pickle(os.path.join(path_to_data, 'GN_1C_nodes.pkl'))
    gas_links = pd.read_pickle(os.path.join(path_to_data, 'GN_1C_links.pkl'))
    gas_halflinks = pd.read_pickle(os.path.join(path_to_data, 'GN_1C_halflinks.pkl'))
    gas_net = from_pd_dataframes(gas_nodes,gas_links,gas_halflinks)

    elec_nodes = pd.read_pickle(os.path.join(path_to_data, 'EN_1C_nodes.pkl'))
    elec_links = pd.read_pickle(os.path.join(path_to_data, 'EN_1C_links.pkl'))
    elec_halflinks = pd.read_pickle(os.path.join(path_to_data, 'EN_1C_halflinks.pkl'))
    elec_net = from_pd_dataframes(elec_nodes,elec_links,elec_halflinks)

    heat_nodes = pd.read_pickle(os.path.join(path_to_data, 'HN_1C_nodes.pkl'))
    heat_links = pd.read_pickle(os.path.join(path_to_data, 'HN_1C_links.pkl'))
    heat_halflinks = pd.read_pickle(os.path.join(path_to_data, 'HN_1C_halflinks.pkl'))
    heat_net = from_pd_dataframes(heat_nodes,heat_links,heat_halflinks)
    Ta = heat_net.links[0].link_params.get('Ta')
    heat_net.Ta = Ta

    # determine number of streets etc. (assumption is that all three networks have the same topology)
    nC = len(gas_nodes.index.levels[0])
    nD = len(gas_nodes.index.levels[1])-1
    nQ = len(gas_nodes.index.levels[2])-1
    nS = len(gas_nodes.index.levels[3])-1
    number_of_street_nodes = len(gas_nodes.index.levels[4])
    n = 0
    for halflink_data in gas_halflinks.loc['C0','D0','Q0','S0'].get('data'):
        if halflink_data.get('q') > 0:
            n+=1
    m = 2*n - (number_of_street_nodes - 2) # -2 because of source node and extra node due to compressor / valve
    
    # adjust position for plotting
    y_shift_elec = 0.3
    x_shift_elec = 0.7
    y_shift_heat = 2*y_shift_elec
    x_shift_heat = 2*x_shift_elec
    for en in elec_net.get_nodes():
        en.x += x_shift_elec
        en.y += y_shift_elec
    for hn in heat_net.get_nodes():
        hn.x += x_shift_heat
        hn.y += y_shift_heat
    
    # coupling part
    gas_source = gas_net.nodes[0]
    elec_source = elec_net.nodes[0]
    heat_source = heat_net.nodes[0]

    gas = gas_net.links[0].link_params.get('carrier')
    water = heat_net.links[0].link_params.get('carrier')
    rhon_g = gas.rhon
    GHV = 40.611 #[MBTU/m^3]
    GHV *= MBTU*BTU #[Wh/m^3]
    GHV *= hour/rhon_g #[J/kg]
    eta_CHP_SI = 0.88 
    coupling_nodes = list()
    coupling_links = list()
    if 'multi' in coupling or 'node2' in coupling:
        quarter_source_names = list()
        for ind_D in range(nD):
            for ind_Q in range(nQ):
                quarter_source_names.append('sQ'+str(ind_Q)+'D'+str(ind_D))
        print('quarter source names =  {}'.format(quarter_source_names))
        quarter_source_ind = list()
        for ind_n, node in enumerate(gas_net.get_nodes()): # assume that all the single-carrier nodes are called the same
            if node.name in quarter_source_names:
                quarter_source_ind.append(ind_n)
        
        np.random.seed(0)
        eta_CHP = .8 + .1*np.random.rand(len(quarter_source_ind))
    
        ph_ref = heat_net.nodes[0].p # needed to (re)set the values of the pressure at the quarter sources. 
        Ts_ref = heat_net.nodes[0].Ts # needed to (re)set the values of the supply line temperature at the quarter sources.

    if 'EH' in coupling:
        alpha = .5/.3 # phi = alpha*P
        
    if coupling == 'single_CHP' or coupling == 'CHP_node1_set1':
        eta_CHP = np.array([eta_CHP_SI, eta_CHP_SI]) # Since all three single-carrier networks are in S.I., a adjustement in base values is not needed.
        cn = HeterogeneousNode('CHP',node_type=0,unit_type='geh_CHP',unit_params={'eta':eta_CHP,'GHV':GHV}) # To unknown
        cn.y = (gas_source.y + elec_source.y + heat_source.y)/3
        cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-1
        coupling_nodes.append(cn)
        
        glc = GasLink('gl_c',gas_source,cn)
        coupling_links.append(glc)
        elc = ElectricalLink('el_c',cn,elec_source)
        coupling_links.append(elc)
        hlc = HeatLink('hl_c',cn,heat_source,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
        coupling_links.append(hlc)
        
        # change node types of homogeneous nodes which are connected to the heterogeneous coupling node
        elec_source.node_type = 5 # PQVdelta node
        for hl in elec_source.get_half_links():
            hl.P = 0
            hl.Q = 0
        heat_source.node_type = 7 # reference temperature (junction) node
        for hl in heat_source.get_half_links():
            heat_source.remove_half_link(hl)
            heat_net.remove_half_link(hl)
            
    elif coupling == 'CHP_node1_set2':
        eta_CHP = np.array([eta_CHP_SI, eta_CHP_SI]) # Since all three single-carrier networks are in S.I., a adjustement in base values is not needed.
        To_c = heat_net.nodes[0].Ts
        cn = HeterogeneousNode('CHP',node_type=1,unit_type='geh_CHP',unit_params={'eta':eta_CHP,'GHV':GHV}) # To known
        cn.y = (gas_source.y + elec_source.y + heat_source.y)/3
        cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-1
        coupling_nodes.append(cn)
        
        glc = GasLink('gl_c',gas_source,cn)
        coupling_links.append(glc)
        elc = ElectricalLink('el_c',cn,elec_source)
        coupling_links.append(elc)
        hlc = HeatLink('hl_c',cn,heat_source,link_params={'carrier':water},bc_type=6,Tsstart=To_c) # To of coupling (source) is known
        coupling_links.append(hlc)
        # change node types of homogeneous nodes which are connected to the heterogeneous coupling node
        elec_source.node_type = 5 # PQVdelta node
        for hl in elec_source.get_half_links():
            hl.P = 0
            hl.Q = 0
        heat_source.node_type = 5 # reference (junction) node
        for hl in heat_source.get_half_links():
            heat_source.remove_half_link(hl)
            heat_net.remove_half_link(hl)
        
    elif coupling == 'single_EH' or coupling == 'EH_node1_set1':
        coupling_matrix = np.array([[eta_CHP_SI/(1+alpha)],[eta_CHP_SI*alpha/(1+alpha)]])
        cn = HeterogeneousNode('EH'+gas_source.name,node_type=0,unit_type='EH',unit_params={'C':coupling_matrix,'GHV':GHV}) # To unknown
        cn.y = (gas_source.y + elec_source.y + heat_source.y)/3
        cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-1
        coupling_nodes.append(cn)
        
        glc = GasLink('gl_c',gas_source,cn)
        coupling_links.append(glc)
        elc = ElectricalLink('el_c',cn,elec_source)
        coupling_links.append(elc)
        hlc = HeatLink('hl_c',cn,heat_source,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
        coupling_links.append(hlc)

        # change node types of homogeneous nodes which are connected to the heterogeneous coupling node
        elec_source.node_type = 4 # QVdelta node 
        for hl in elec_source.get_half_links():
            hl.P = 0
            hl.Q = 0
        heat_source.node_type = 7 # reference temperature (junction) node
        for hl in heat_source.get_half_links():
            heat_source.remove_half_link(hl)
            heat_net.remove_half_link(hl)
            
    elif coupling == 'EH_node1_set2':
        coupling_matrix = np.array([[eta_CHP_SI/(1+alpha)],[eta_CHP_SI*alpha/(1+alpha)]])
        To_c = heat_net.nodes[0].Ts
        cn = HeterogeneousNode('EH'+gas_source.name,node_type=1,unit_type='EH',unit_params={'C':coupling_matrix,'GHV':GHV}) # To known
        cn.y = (gas_source.y + elec_source.y + heat_source.y)/3
        cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-1
        coupling_nodes.append(cn)
        
        glc = GasLink('gl_c',gas_source,cn)
        coupling_links.append(glc)
        elc = ElectricalLink('el_c',cn,elec_source)
        coupling_links.append(elc)
        hlc = HeatLink('hl_c',cn,heat_source,link_params={'carrier':water},bc_type=6,Tsstart=To_c) # To of coupling (source) is known
        coupling_links.append(hlc)

        # change node types of homogeneous nodes which are connected to the heterogeneous coupling node
        elec_source.node_type = 5 # PQVdelta node
        for hl in elec_source.get_half_links():
            hl.P = 0
            hl.Q = 0
            
    elif coupling == 'multi_CHP':
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            eta = np.array([eta_CHP[ind], eta_CHP[ind]])
            cn = HeterogeneousNode('CHP'+gas_source.name,node_type=0,unit_type='geh_CHP',unit_params={'eta':eta,'GHV':GHV}) # To unknown
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
            coupling_links.append(hlc)
            
            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            elec_source.node_type = 5 # PQVdelta node
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 7 # reference temperature (junction) node
            heat_source.p = ph_ref
            heat_source.Ts = Ts_ref
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
    elif coupling == 'multi_CHP2':
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            eta = np.array([eta_CHP[ind], eta_CHP[ind]])
            cn = HeterogeneousNode('CHP'+gas_source.name,node_type=0,unit_type='geh_CHP',unit_params={'eta':eta,'GHV':GHV}) # To unknown
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
            coupling_links.append(hlc)
            
            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            elec_source.node_type = 5 # PQVdelta node
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 7 # reference temperature (junction) node
            heat_source.p = ph_ref
            heat_source.Ts = Ts_ref
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
                
        # change node types of the city sources, and set values
        elec_source = elec_net.nodes[0]
        elec_source.node_type = 1 # generator node
        elec_source.half_links[0].P = -1/5*nD*nQ*nS*n*0.2*kW
        heat_source = heat_net.nodes[0]
        heat_source_hl = heat_source.half_links[0]
        if heat_load == 'delta':
            heat_source.node_type = 12
            heat_source_hl.bc_type = 4
            heat_source_hl.dT = heat_source.Ts - heat_source.Tr
        else:
            heat_source.node_type = 1 # source node
            heat_source_hl.bc_type = 2
            heat_source_hl.Ts = heat_source.Ts
        heat_source.p *=1.01 # set initial condition for p (such that flow is in right direction)
        heat_source_hl.dphi = -2*nD*nQ*nS*n*0.05*kW
        
    elif coupling == 'multi_CHP3': 
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            eta = np.array([eta_CHP[ind], eta_CHP[ind]])
            cn = HeterogeneousNode('CHP'+gas_source.name,node_type=1,unit_type='geh_CHP',unit_params={'eta':eta,'GHV':GHV}) # To known
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=6,Tsstart=110) # To of coupling (source) is known
            coupling_links.append(hlc)
            
            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            elec_source.node_type = 5 # PQVdelta node
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 6 # temperature (junction) nodee
            heat_source.Ts = Ts_ref
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
                
    elif coupling == 'CHP_node2_set1':
        if nD>1 or nQ>1: #node set only defined for a single district and quarter
            raise ValueError("This type of coupling can only be used with nD=nQ=1.")
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            eta = np.array([eta_CHP[ind], eta_CHP[ind]])
            To_c = .9*Ts_ref
            cn = HeterogeneousNode('CHP'+gas_source.name,node_type=1,unit_type='geh_CHP',unit_params={'eta':eta,'GHV':GHV}) # To known
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=6,Tsstart=To_c) # To of coupling (source) is known
            coupling_links.append(hlc)
            
            # change node types of the city source nodes (which were originally the slack nodes)
            q_demand = np.sum([hl.q for node in gas_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.q>0])            
            P_demand = np.sum([hl.P for node in elec_net.get_nodes(node_types=[2]) for hl in node.get_half_links()  if hl.P>0])
            phi_demand = np.sum([hl.dphi for node in heat_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.dphi>0])
            P_city_source =  .8*P_demand # power provided by city source, based on demand 
            phi_city_source = .8*phi_demand # power provided by city source, based on demand 
            loss_elec = 0#.1
            loss_heat = 0#.1
            Pc = (1+loss_elec)*(P_demand-P_city_source) # estimated power needed from the coupling (taking power provided by city source and estimated losses into account)
            phic = (1+loss_heat)*(phi_demand - phi_city_source) # estimated power needed from the coupling (taking power provided by city source and estimated losses into account)
            Ec = Pc/eta[0] + phic/eta[1] # estimated power needed from the coupling
            qc = Ec/GHV # estimated amount of gas consumed by the CHP
            q_city_source = -(q_demand + qc)
            gas_net.nodes[0].node_type = 3 # ref. load node (this node was originally a slack node, so it didn't have a half link)
            GasHalfLink(gas_net.nodes[0].name+'_hl',gas_net.nodes[0],q=q_city_source)
            print('P_demand = {}, phi_demand = {}, q_demand = {}, P_city_source = {}, Ec = {}, Pc = {}, qc = {}, phic = {}'.format(P_demand,phi_demand,q_demand,P_city_source,Ec,Pc,qc,phic))
            elec_net.nodes[0].node_type = 1 # generator node (this node was originally a slack node, so it didn't have a half link)
            elec_net.nodes[0].half_links[0].P=-P_city_source
            
            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            for hl in gas_source.get_half_links():
                hl.q = 0
            elec_source.node_type = 5 # PQVdelta node
            elec_source.V = V_ref_Q
            elec_source.delta = delta_ref_Q
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 2 # junction nodee
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
    elif coupling == 'CHP_node2_set2':
        if nD>1 or nQ>1: #node set only defined for a single district and quarter
            raise ValueError("This type of coupling can only be used with nD=nQ=1.")
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            eta = np.array([eta_CHP[ind], eta_CHP[ind]])
            To_c = 1.*Ts_ref
            cn = HeterogeneousNode('CHP'+gas_source.name,node_type=1,unit_type='geh_CHP',unit_params={'eta':eta,'GHV':GHV}) # To known
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=6,Tsstart=To_c) # To of coupling (source) is known
            coupling_links.append(hlc)
            
            # change node types of the city source nodes (which were originally the slack nodes)
            q_demand = np.sum([hl.q for node in gas_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.q>0])
            gas_net.nodes[0].node_type = 3 # ref. load node (this node was originally a slack node, so it didn't have a half link)
            GasHalfLink(gas_net.nodes[0].name+'_hl',gas_net.nodes[0],q=-1.5*q_demand)
            phi_demand = np.sum([hl.dphi for node in heat_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.dphi>0])
            if heat_load == 'delta':
                heat_net.nodes[0].node_type = 13
                heat_net.nodes[0].half_links[0].bc_type = 4
                heat_net.nodes[0].half_links[0].dT = Ts_ref - heat_net.nodes[0].Tr
            else:
                heat_net.nodes[0].node_type = 3 # source ref. node
                heat_net.nodes[0].half_links[0].bc_type = 2
                heat_net.nodes[0].half_links[0].Ts = Ts_ref
            heat_net.nodes[0].half_links[0].phi = -.9*phi_demand
            
            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            elec_source.node_type = 6 # PQV node
            elec_source.V = V_ref_Q
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 2 # junction nodee
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
    elif coupling == 'CHP_node2_set3':
        if nD>1 or nQ>1: #node set only defined for a single district and quarter
            raise ValueError("This type of coupling can only be used with nD=nQ=1.")
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            eta = np.array([eta_CHP[ind], eta_CHP[ind]])
            To_c = 1.*Ts_ref
            cn = HeterogeneousNode('CHP'+gas_source.name,node_type=1,unit_type='geh_CHP',unit_params={'eta':eta,'GHV':GHV}) # To known
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=6,Tsstart=To_c) # To of coupling (source) is known
            coupling_links.append(hlc)
            
            # change node types of the city source nodes (which were originally the slack nodes)
            P_demand = np.sum([hl.P for node in elec_net.get_nodes(node_types=[2]) for hl in node.get_half_links()  if hl.P>0])
            elec_net.nodes[0].node_type = 1 # generator node (this node was originally a slack node, so it didn't have a half link)
            elec_net.nodes[0].half_links[0].P=-.9*P_demand
            phi_demand = np.sum([hl.dphi for node in heat_net.get_nodes(node_types=[1,3,4]) for hl in node.get_half_links()  if hl.dphi>0])
            if heat_load == 'delta':
                heat_net.nodes[0].node_type = 13
                heat_net.nodes[0].half_links[0].bc_type = 4
                heat_net.nodes[0].half_links[0].dT = Ts_ref - heat_net.nodes[0].Tr
            else:
                heat_net.nodes[0].node_type = 3 # source ref. node
                heat_net.nodes[0].half_links[0].bc_type = 2
                heat_net.nodes[0].half_links[0].Ts = Ts_ref
            heat_net.nodes[0].half_links[0].phi = -.9*phi_demand
            
            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            for hl in gas_source.get_half_links():
                hl.q = 0
            elec_source.node_type = 5 # PQVdelta node
            elec_source.V = V_ref_Q
            elec_source.delta = delta_ref_Q
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 2 # junction nodee
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
                
    elif coupling == 'multi_EH1':
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            coupling_matrix = np.array([[eta_CHP[ind]/(1+alpha)],[eta_CHP[ind]*alpha/(1+alpha)]])
            cn = HeterogeneousNode('EH'+gas_source.name,node_type=0,unit_type='EH',unit_params={'C':coupling_matrix,'GHV':GHV}) #To unknown
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
            coupling_links.append(hlc)
            
            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            elec_source.node_type = 5 # PQVdelta node
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 6 # temperature (junction) node
            heat_source.Ts = Ts_ref
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
    elif coupling == 'multi_EH2':
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            coupling_matrix = np.array([[eta_CHP[ind]/(1+alpha)],[eta_CHP[ind]*alpha/(1+alpha)]])
            cn = HeterogeneousNode('EH'+gas_source.name,node_type=1,unit_type='EH',unit_params={'C':coupling_matrix,'GHV':GHV}) #To known
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=6,Tsstart=Ts_ref) # To of coupling (source) is known
            coupling_links.append(hlc)
            
            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            elec_source.node_type = 5 # PQVdelta node
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 2 # (junction) node
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
    elif coupling == 'multi_EH3':
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            coupling_matrix = np.array([[eta_CHP[ind]/(1+alpha)],[eta_CHP[ind]*alpha/(1+alpha)]])
            cn = HeterogeneousNode('EH'+gas_source.name,node_type=1,unit_type='EH',unit_params={'C':coupling_matrix,'GHV':GHV}) #To known
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=6,Tsstart=Ts_ref) # To of coupling (source) is known
            coupling_links.append(hlc)
            
            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            elec_source.node_type = 6 # PQV node
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 6 # temperature (junction) node
            heat_source.Ts = Ts_ref
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
                
    elif coupling == 'EH_node2_set1':
        if nD>1 or nQ>1: #node set only defined for a single district and quarter
            raise ValueError("This type of coupling can only be used with nD=nQ=1.")
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            coupling_matrix = np.array([[eta_CHP[ind]/(1+alpha)],[eta_CHP[ind]*alpha/(1+alpha)]])
            To_c = 1.*Ts_ref
            cn = HeterogeneousNode('EH'+gas_source.name,node_type=1,unit_type='EH',unit_params={'C':coupling_matrix,'GHV':GHV}) #To known
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=6,Tsstart=To_c) # To of coupling (source) is known
            coupling_links.append(hlc)
            
            # change node types of the city source nodes (which were originally the slack nodes)
            q_demand = np.sum([hl.q for node in gas_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.q>0])            
            P_demand = np.sum([hl.P for node in elec_net.get_nodes(node_types=[2]) for hl in node.get_half_links()  if hl.P>0])
            phi_demand = np.sum([hl.dphi for node in heat_net.get_nodes(node_types=[1,3,4]) for hl in node.get_half_links()  if hl.dphi>0])
            P_city_source =  .8*P_demand # power provided by city source, based on demand 
            phi_city_source = .8*phi_demand # power provided by city source, based on demand 
            loss_elec = 0#.1
            loss_heat = 0#.1
            Pc = (1+loss_elec)*(P_demand-P_city_source) # estimated power needed from the coupling (taking power provided by city source and estimated losses into account)
            phic = (1+loss_heat)*(phi_demand - phi_city_source) # estimated power needed from the coupling (taking power provided by city source and estimated losses into account)
            qc = np.mean([Pc/coupling_matrix[0,0], phic/coupling_matrix[1,0]])/GHV # estimated amount of gas consumed by the CHP
            q_city_source = -(q_demand + qc)
            gas_net.nodes[0].node_type = 3 # ref. load node (this node was originally a slack node, so it didn't have a half link)
            GasHalfLink(gas_net.nodes[0].name+'_hl',gas_net.nodes[0],q=q_city_source)
            print('P_demand = {}, phi_demand = {}, q_demand = {}, P_city_source = {}, Pc = {}, qc = {}, phic = {}'.format(P_demand,phi_demand,q_demand,P_city_source,Pc,qc,phic))
            
            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            for hl in gas_source.get_half_links():
                hl.q = 0
            elec_source.node_type = 6 # PQV node
            elec_source.V = V_ref_Q
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 2 # junction node
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
    elif coupling == 'EH_node2_set2':
        if nD>1 or nQ>1: #node set only defined for a single district and quarter
            raise ValueError("This type of coupling can only be used with nD=nQ=1.")
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            coupling_matrix = np.array([[eta_CHP[ind]/(1+alpha)],[eta_CHP[ind]*alpha/(1+alpha)]])
            To_c = 1*Ts_ref
            cn = HeterogeneousNode('EH'+gas_source.name,node_type=1,unit_type='EH',unit_params={'C':coupling_matrix,'GHV':GHV}) #To known
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=6,Tsstart=To_c) # To of coupling (source) is known
            coupling_links.append(hlc)
            
            # change node types of the city source nodes (which were originally the slack nodes)           
            P_demand = np.sum([hl.P for node in elec_net.get_nodes(node_types=[2]) for hl in node.get_half_links()  if hl.P>0])
            phi_demand = np.sum([hl.dphi for node in heat_net.get_nodes(node_types=[1,3,4]) for hl in node.get_half_links()  if hl.dphi>0])
            loss_elec = 0
            loss_heat = 0
            phi_city_source = 0*phi_demand # power provided by city source, based on demand 
            phic = (1+loss_heat)*(phi_demand - phi_city_source) # estimated power needed from the coupling (taking power provided by city source and estimated losses into account)
            Pc = coupling_matrix[0,0]/coupling_matrix[1,0]*phic
            P_city_source = (1+loss_elec)*(P_demand-Pc)

            print('P_demand = {}, phi_demand = {}, P_city_source = {}, Pc = {}, phic = {}'.format(P_demand,phi_demand,P_city_source,Pc,phic))
            elec_net.nodes[0].node_type = 1 # generator node (this node was originally a slack node, so it didn't have a half link)
            elec_net.nodes[0].half_links[0].P = -P_city_source

            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            for hl in gas_source.get_half_links():
                hl.q = 0
            elec_source.node_type = 5 # PQVdelta node
            elec_source.V = V_ref_Q
            elec_source.delta = delta_ref_Q
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 2 # junction node
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)
    elif coupling == 'EH_node2_set3':
        if nD>1 or nQ>1: #node set only defined for a single district and quarter
            raise ValueError("This type of coupling can only be used with nD=nQ=1.")
        for ind, ind_qs in enumerate(quarter_source_ind):
            gas_source = gas_net.nodes[ind_qs]
            elec_source = elec_net.nodes[ind_qs]
            heat_source = heat_net.nodes[ind_qs]
            coupling_matrix = np.array([[eta_CHP[ind]/(1+alpha)],[eta_CHP[ind]*alpha/(1+alpha)]])
            To_c = 1.*Ts_ref
            cn = HeterogeneousNode('EH'+gas_source.name,node_type=1,unit_type='EH',unit_params={'C':coupling_matrix,'GHV':GHV}) #To known
            cn.y = (gas_source.y + elec_source.y + heat_source.y)/3+.5
            cn.x = (gas_source.x + elec_source.x + heat_source.x)/3-.5
            coupling_nodes.append(cn)
            glc = GasLink('gl'+gas_source.name+'_c',gas_source,cn)
            coupling_links.append(glc)
            elc = ElectricalLink('el'+elec_source.name+'_c',cn,elec_source)
            coupling_links.append(elc)
            hlc = HeatLink('hl'+heat_source.name+'_c',cn,heat_source,link_params={'carrier':water},bc_type=6,Tsstart=To_c) # To of coupling (source) is known
            coupling_links.append(hlc)
            
            # change node types of the city source nodes (which were originally the slack nodes)
            phi_demand = np.sum([hl.dphi for node in heat_net.get_nodes(node_types=[1,3,4]) for hl in node.get_half_links()  if hl.dphi>0])
            if heat_load == 'delta':
                heat_net.nodes[0].node_type = 13
                heat_net.nodes[0].half_links[0].bc_type = 4
                heat_net.nodes[0].half_links[0].dT = Ts_ref - heat_net.nodes[0].Tr
            else:
                heat_net.nodes[0].node_type = 3 # source ref. node
                heat_net.nodes[0].half_links[0].bc_type = 2
                heat_net.nodes[0].half_links[0].Ts = Ts_ref
            heat_net.nodes[0].half_links[0].phi = -.9*phi_demand
            
            # change node types of homogeneous nodes which are connected to the heterogeneous coupling node, and set values
            for hl in gas_source.get_half_links():
                hl.q = 0
            elec_source.node_type = 6 # PQV node
            elec_source.V = V_ref_Q
            for hl in elec_source.get_half_links():
                hl.P = 0
                hl.Q = 0
            heat_source.node_type = 2 # junction node
            for hl in heat_source.get_half_links():
                heat_source.remove_half_link(hl)
                heat_net.remove_half_link(hl)            
    else:
        raise ValueError('Enter valid value for coupling')

    if heat_load == 'delta' and cn.node_type == 1:
        cn.node_type = 2
        Tr_ref = heat_net.nodes[0].Tr
        hlc.bc_type = 10
        hlc.dTstart = hlc.Tsstart - Tr_ref
        
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a GasNode object can be added",UserWarning) # A warning of (sub)class UserWarning, of which the message starts with 'Only a GasNode object can be added' is ignored
        warnings.filterwarnings("ignore", "Only a ElectricalNode object can be added",UserWarning) # A warning of (sub)class UserWarning, of which the message starts with 'Only a ElectricalNode object can be added' is ignored
        warnings.filterwarnings("ignore", "Only a HeatNode object can be added",UserWarning) # A warning of (sub)class UserWarning, of which the message starts with 'Only a HeatNode object can be added' is ignored
        for lc in coupling_links:
            if isinstance(lc,GasLink):
                gas_net.add_link(lc)
            elif isinstance(lc,ElectricalLink):
                elec_net.add_link(lc)
            elif isinstance(lc,HeatLink):
                heat_net.add_link(lc)
                
    het_net = HeterogeneousNetwork('MES_1C')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    
    for cn in coupling_nodes:
        het_net.add_node(cn)
    
    return het_net, gas_net, elec_net, heat_net, nC, nD, nQ, nS, n, m

def initialize_network(het_net, gas_net, elec_net, heat_net, nC, nD, nQ, nS, n, coupling='single_CHP'):
    """Initialize the network.
    
    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The test network to be initialized
    gas_net : GasNetwork
        The test gas subnetwork
    elec_net : ElectricalNetwork
        The test electrical subnetwork
    heat_net : HeatNetwork
        The test heat subnetwork
    nC : int
        number of city networks
    nD : int
        number of district networks
    nQ : int
        number of quarter networks
    nS : int
        number of street networks
    n : int
        number of loads, for each street
    coupling : str, optional
        Determines how the single carrier networks are coupled. Options are 'single_CHP', 'single_EH', 'multi_CHP', 'multi_CHP2', 'multi_CHP3', 'multi_EH1', 'multi_EH2', or 'multi_EH3'. The ones with single are coupled through a single coupling node connected at the city source of the single-carrier networkes. The ones with multi are coupled through multiple coupling nodes, connected at every quarter source node of the single-carrier networks. Default is 'single_CHP'.
        
    Returns
    -------
    x0 : np array
        Vector with initial guess
    """
    het_net.initialize()
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries()
    
    # gas
    q_init_value = np.mean([hl.q for node in gas_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.q>0])
    gas_source = gas_net.nodes[0]
    gas = gas_net.links[0].link_params.get('carrier')
    p_ref = gas_source.p #[bar]
    
    # electricity
    S_init_load = 1.5*kW
    
    # heat
    hl_flow = list()
    hl_heat = list()
    hl_To = list()
    for node in heat_net.get_nodes(node_types=[1,3,4,12,13]):
        for hl in node.get_half_links():
            # set temperatures to initial some value. Needed to do hl.flow()
            if hl in unknown_Ts_halflinks: 
                hl.Ts = Ts_ref
            if hl in unknown_Tr_halflinks: 
                hl.Tr = 50.
            if hl.bc_type in [4,5,10,11]: #dT known
                if hl.sink:
                    if not (hl in unknown_Ts_halflinks): 
                        hl.Ts = hl.start_node.Ts
                    hl.Tr = hl.Ts - hl.dT
                else:
                    if not (hl in unknown_Tr_halflinks):
                        hl.Tr = 50
                    hl.Ts = hl.dT + hl.Tr
            m_hl = hl.flow()
            hl.m = m_hl
            hl_flow.append(m_hl)
            hl_heat.append(hl.dphi)
            if hl.sink:
                hl_To.append(hl.Tr)
    m_init_value = np.mean(hl_flow) 
    phi_init_value = np.mean(hl_heat) 
    Tr_init = np.min(hl_To)
    m_init_street = n/2*m_init_value
    m_init_quarter = nS*n*m_init_value
    m_init_district = nQ*nS*n*m_init_value
    m_init_city = nD*nQ*nS*n*m_init_value  
    
    # coupling
    if ('single' in coupling) or ('node1' in coupling): 
        qc_init = nD*nQ*nS*n*q_init_value #1e-5
        Pc_init = nD*nQ*nS*n*S_init_load#24.3*MW#1.5*kW
        Qc_init = nD*nQ*nS*n*S_init_load#24.3*MW#1.5*kW
        phic_init = nD*nQ*nS*n*phi_init_value #1.*kW #[W]
        mc_init = m_init_city
        Toc_init = 100.#130.
        
    elif ('multi' in coupling) or ('node2' in coupling):                
        qc_init = nS*n*q_init_value #1e-5
        Pc_init = nS*n*S_init_load#1.5*kW
        Qc_init = nS*n*S_init_load#1.5*kW
        phic_init = nS*n*phi_init_value #1.*kW #[W]
        mc_init = m_init_quarter
        Toc_init = 110.
        
    else:
        raise ValueError('Enter valid value for coupling')

    # gas
    for link in gas_net.get_links():
        #link.q = q_init_value
        if link.link_type == 'compressor':
            link.q = q_init_value
        elif '_c' in link.name: #coupling dummy link
            link.q = qc_init
        else:
            link.q = link.flow()
            
    # electricity        
    for link in elec_net.get_links(['dummy']):#coupling dummy link
        link.Pstart = Pc_init
        link.Pend = Pc_init
        link.Qstart = Qc_init
        link.Qend = Qc_init
        
    # heat
    for node in heat_net.get_nodes(node_types=[0,1,2,3,4,5,6,7,12,13]):
        node.Tr = Tr_init
    for link in heat_net.get_links():
        if 'S' in link.name:
            link.m = m_init_street
        elif 'Q' in link.name:
            link.m = m_init_quarter
        elif 'D' in link.name:
            link.m = m_init_district
        elif 'C' in link.name:
            link.m = m_init_city 
        elif '_c' in link.name: #coupling dummy link
            link.m = mc_init
        else:
            link.m = m_init_city
    for l in heat_net.get_links(link_types=['dummy']): #coupling links
        l.m = mc_init
        l.dphistart = -phic_init
        if l.bc_type == 0 or l.bc_type == 10: # To unknown
            l.Tsstart = Toc_init
                
    x0 = het_net.set_x_init(formulation=form)        
    return x0

def scaling_matrices(het_net, nC, nD, nQ, nS, n, coupling='single_CHP'):
    """Determine the matrices needed for scaling
    
    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The heterogeneous network
    nC : int
        number of city networks
    nD : int
        number of district networks
    nQ : int
        number of quarter networks
    nS : int
        number of street networks
    n : int
        number of loads, for each street
    coupling : str, optional
        Determines how the single carrier networks are coupled. Options are 'single_CHP', 'single_EH', 'multi_CHP', 'multi_CHP2', 'multi_CHP3', 'multi_EH1', 'multi_EH2', or 'multi_EH3'. The ones with single are coupled through a single coupling node connected at the city source of the single-carrier networkes. The ones with multi are coupled through multiple coupling nodes, connected at every quarter source node of the single-carrier networks. Default is 'single_CHP'.
        
    Returns
    -------
    D_F : scipy.sparse diagonal matrix
        Diagonal matrix with function scaling factors
    D_x : scipy.sparse diagonal matrix
        Diagonal matrix with variable scaling factors
    """
    # base values
    #gas
    qb_street = 5e-3 #[kg/s]
    qb_quarter = nS*n*qb_street
    qb_district = nQ*nS*n*qb_street
    qb_city = nD*nQ*nS*n*qb_street
    pgb_city = 8*bar
    pgb_district = 100*mbar
    pgb_quarter = 100*mbar
    pgb_street = 30*mbar
    #electricity
    #Sb_street = n/2*.3*kW #[W]
    #ratio_SQ = 25
    #ratio_DC = 5
    #Sb_SQ = Sb_street
    #Sb_quarter = nS*n*Sb_street#*ratio_SQ
    #Sb_district = nQ*nS*n*Sb_street#*ratio_SQ
    #Sb_DC = Sb_district
    #Sb_city = nD*nQ*nS*n*Sb_street#*ratio_SQ*ratio_DC
    Sb_street = 7*kW #[W]
    Sb_SQ = 77.76*kW
    Sb_quarter = 4.86*MW
    Sb_district = 2.43*MW
    Sb_DC = 48.6*MW
    Sb_city = 24.3*MW
    Vb_city = 50*kV
    Vb_district = 10*kV
    Vb_quarter = 10*kV
    Vb_street = .4*kV
    #heat
    phbase = 9*bar #[Pa]
    Tbase = 100 #[C]
    mbase_load = 1e-3 #[kg/s]
    mb_street = n/2*mbase_load
    mb_quarter = nS*n*mbase_load
    mb_district = nQ*nS*n*mbase_load
    mb_city = nD*nQ*nS*n*mbase_load
    phibase_load = .5*kW #[W]
    phib_street = n/2*phibase_load
    phib_quarter = nS*n*phibase_load
    phib_district = nQ*nS*n*phibase_load
    phib_city = nD*nQ*nS*n*phibase_load
    #coupling
    if ('single' in coupling) or ('node1' in coupling):
        qb_coupling = qb_city
        Sb_coupling = Sb_city
        mb_coupling = mb_city
        phib_coupling = phib_city
    elif ('multi' in coupling) or ('node2' in coupling):
        qb_coupling = qb_quarter
        Sb_coupling = Sb_quarter
        mb_coupling = mb_quarter
        phib_coupling = phib_quarter
    
    # create x base
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=form)
    xb_g = np.zeros(len(xg_entries))
    for ind_el,el in enumerate(xg_entries):
        if 'S' in el.name:
            if 'ene' in el.name:
                pgbase = pgb_quarter # node before compressor
            else:
                pgbase = pgb_street
            qbase = qb_street
        elif 'Q' in el.name:
            pgbase = pgb_quarter
            qbase = qb_quarter#n*qb_street
        elif 'D' in el.name:
            if 'ene' in el.name:
                pgbase = pgb_city # node before compressor
            else:
                pgbase = pgb_district
            qbase = qb_district#nS*n*qb_street
        elif 'C' in el.name:
            pgbase = pgb_city
            qbase = qb_city#nQ*nS*n*qb_street
        else:
            pgbase = pgb_city
            qbase = qb_city
        if isinstance(el,GasNode):
            xb_g[ind_el] = pgbase 
        elif isinstance(el,GasLink):
            xb_g[ind_el] = qbase 
    xb_delta = np.ones(len(unknown_delta_nodes))
    xb_V = np.zeros(len(unknown_V_nodes))
    for ind_el, el in enumerate(unknown_V_nodes):
        if 'S' in el.name:
            if 'ene' in el.name:
                Vbase = Vb_street#Vb_quarter # node before trafo
            else:
                Vbase = Vb_street
        elif 'Q' in el.name:
            Vbase = Vb_quarter
        elif 'D' in el.name:
            if 'ene' in el.name:
                Vbase = Vb_district#Vb_city # node before trafo
            else:
                Vbase = Vb_district
        elif 'C' in el.name:
            Vbase = Vb_city
        xb_V[ind_el] = Vbase
    xb_m = np.zeros(len(unknown_m_links))
    for ind_el, el in enumerate(unknown_m_links):
        if 'S' in el.name:
            mbase = mb_street
        elif 'Q' in el.name:
            mbase = mb_quarter
        elif 'D' in el.name:
            mbase = mb_district
        elif 'C' in el.name:
            mbase = mb_city
        else:
            mbase = mb_city
        xb_m[ind_el] = mbase
    xb_m_hl = np.zeros(len(unknown_m_halflinks))
    for ind_el, el in enumerate(unknown_m_halflinks):
        if 'S' in el.start_node.name:
            mbase_hl = mbase_load#mb_street
        elif 'Q' in el.start_node.name:
            mbase_hl = mb_quarter
        elif 'D' in el.start_node.name:
            mbase_hl = mb_district
        elif 'C' in el.start_node.name:
            mbase_hl = mb_city
        xb_m_hl[ind_el] = mbase_hl
    xb_ph = phbase*np.ones(len(unknown_p_nodes))
    xb_Ts = Tbase*np.ones(len(unknown_Ts_nodes))
    xb_Tr = Tbase*np.ones(len(unknown_Tr_nodes))
    xb_Tshl = Tbase*np.ones(len(unknown_Ts_halflinks))
    xb_Trhl = Tbase*np.ones(len(unknown_Tr_halflinks))
    xb_qc = qb_coupling*np.ones(len(unknown_qc_links))
    xb_Pc = Sb_coupling*np.ones(len(unknown_Pc_links))
    xb_Qc = Sb_coupling*np.ones(len(unknown_Qc_links))
    xb_mc = mb_coupling*np.ones(len(unknown_mc_links))
    xb_phic = phib_coupling*np.ones(len(unknown_dphi_links))
    xb_Toc = Tbase*np.ones(len(unknown_Ts_links))
    xb = np.concatenate((xb_g,xb_delta,xb_V,xb_m,xb_m_hl,xb_ph,xb_Ts,xb_Tr,xb_Tshl,xb_Trhl,xb_qc,xb_Pc,xb_Qc,xb_mc,xb_phic,xb_Toc))
    D_x = sps.diags(1/xb)
    
    # create F base
    F_entries, Fg_entries, Fe_entries, known_P_nodes, known_Q_nodes, Fh_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_To_halflinks, Fc_entries, F_fc_nodes, F_fc_amount, F_phi_nodes, F_To_nodes = het_net.get_F_entries(formulation=form)
    Fb_g = np.zeros(len(Fg_entries))
    for ind_el,el in enumerate(Fg_entries):
        if 'S' in el.name:
            pgbase = pgb_street
            qbase = qb_street
        elif 'Q' in el.name:
            pgbase = pgb_quarter
            qbase = qb_quarter
        elif 'D' in el.name:
            pgbase = pgb_district
            qbase = qb_district
        elif 'C' in el.name:
            pgbase = pgb_city
            qbase = qb_city
        else:
            pgbase = pgb_city
            qbase = qb_city
        if isinstance(el,GasNode):
            Fb_g[ind_el] = qbase 
        elif isinstance(el,GasLink):
            if el.link_eq_form == 'q_of_dp':
                Fb_g[ind_el] = qbase 
            if el.link_type == 'compressor':
                Fb_g[ind_el] = pgbase 
            elif 'high_pres' in el.link_type:
                Fb_g[ind_el] = pgbase**2
            else:
                Fb_g[ind_el] = pgbase
    Fb_e = np.zeros(len(Fe_entries))
    for ind_el, el in enumerate(Fe_entries):
        if 'S' in el.name:
            if 'ene' in el.name: # node before trafo
                Sbase = Sb_SQ #Sb_street # Sb_quarter
            else:
                Sbase = Sb_street
        elif 'Q' in el.name:
            Sbase = Sb_quarter
        elif 'D' in el.name:
            if 'ene' in el.name: # node before trafo
                Sbase = Sb_DC #Sb_district #Sb_city
            else:
                Sbase = Sb_district
        elif 'C' in el.name:
            Sbase = Sb_city
        Fb_e[ind_el] = Sbase
    Fb_m = mbase*np.zeros(len(F_m_nodes))
    for ind_el, el in enumerate(F_m_nodes):
        if 'S' in el.name:
            mbase = mb_street
        elif 'Q' in el.name:
            mbase = mb_quarter
        elif 'D' in el.name:
            mbase = mb_district
        elif 'C' in el.name:
            mbase = mb_city
        Fb_m[ind_el] = mbase
    Fb_deltap = phbase*np.ones(len(F_deltap_links))
    Fb_Ts = np.zeros(len(F_Ts_nodes))
    for ind_el, el in enumerate(F_Ts_nodes):
        if 'S' in el.name:
            mbase = mb_street
        elif 'Q' in el.name:
            mbase = mb_quarter
        elif 'D' in el.name:
            mbase = mb_district
        elif 'C' in el.name:
            mbase = mb_city
        Fb_Ts[ind_el] = mbase*Tbase
    Fb_Tr = np.zeros(len(F_Tr_nodes))
    for ind_el, el in enumerate(F_Tr_nodes):
        if 'S' in el.name:
            mbase = mb_street
        elif 'Q' in el.name:
            mbase = mb_quarter
        elif 'D' in el.name:
            mbase = mb_district
        elif 'C' in el.name:
            mbase = mb_city
        Fb_Tr[ind_el] = mbase*Tbase
    Fb_phi = np.zeros(len(F_phi_halflinks))
    Fb_To = Tbase*np.ones(len(F_To_halflinks))
    for ind_el, el in enumerate(F_phi_halflinks):
        if 'S' in el.start_node.name:
            phibase = phib_street
        elif 'Q' in el.start_node.name:
            phibase = phib_quarter
        elif 'D' in el.start_node.name:
            phibase = phib_district
        elif 'C' in el.start_node.name:
            phibase = phib_city
        Fb_phi[ind_el] = phibase
    Fb_c = np.zeros(np.sum(F_fc_amount))
    Egbase = phib_coupling
    for ind_el,el in enumerate(F_fc_nodes):
        if ind_el == 0:
            ind = ind_el
        else:
            ind = np.sum(F_fc_amount[:ind_el])
        if el.unit_type == 'ge_gas_fired_gen':
            Fb_c[ind] = Sb_coupling
        elif el.unit_type == 'ge_gas_fired_gen_valve_point':
            Fb_c[ind] = Egbase
        elif el.unit_type == 'gh_gas_boiler':
            Fb_c[ind] = phib_coupling
        elif el.unit_type == 'eh_elec_boiler':
            Fb_c[ind] = phib_coupling
        elif el.unit_type == 'geh_CHP':
            Fb_c[ind] = Egbase
        elif el.unit_type == 'EH':
            Fb_c[ind] = Sb_coupling
            Fb_c[ind+1] = phib_coupling
    Fb_phic = phib_coupling*np.ones(len(F_phi_nodes))
    Fb_Toc = Tbase*np.ones(len(F_To_nodes))
    Fb = np.concatenate((Fb_g,Fb_e,Fb_m,Fb_deltap,Fb_Ts,Fb_Tr,Fb_phi,Fb_To,Fb_c,Fb_phic,Fb_Toc))
    D_F = sps.diags(1/Fb)
    return D_F, D_x

def solve_MES_1C(max_number_of_nodes,save_fig,plot_all, coupling='single_CHP', heat_load='outflow'):
    """Solve this example, using fb in the gas network, unknown half link flow formulation in heat, and scaling in the solver.
    
    Parameters
    ----------
    max_number_of_nodes : int
        Maximum number of nodes of the MES for which the topology is plotted, the Jacobian is plotted, information about nodes and links is printed, etc. 
    save_fig : bool
        If True, the convergence plot is saved as a pgf. 
    plot_all : bool
        If True, (all) the plots are made for this case.
    coupling : str, optional
        Determines how the single carrier networks are coupled. Options are 'single_CHP', 'single_EH', 'multi_CHP', 'multi_CHP2', 'multi_CHP3', 'multi_EH1', 'multi_EH2', or 'multi_EH3'. The ones with single are coupled through a single coupling node connected at the city source of the single-carrier networkes. The ones with multi are coupled through multiple coupling nodes, connected at every quarter source node of the single-carrier networks. Default is 'single_CHP'.
        
    Returns
    -------
    x_sol : np array
        Solution vector
    iters : int
        Number of iterations NR used
    err_vec : np array
        Error of NR for every iteration
    """
    
    # create network from data
    het_net, gas_net, elec_net, heat_net, nC, nD, nQ, nS, n, m = create_network(coupling=coupling,heat_load=heat_load)
    
    # determine number of nodes, links, variables and equations
    xm_len = 0
    xm_hl_len = 0
    xp_len = 0
    xTs_len = 0
    xTr_len = 0
    if coupling == 'single_CHP':
        N_SC = nD*(nQ*(nS*(2*n-m+2)+1)+2)+1
        number_of_nodes = 3*N_SC+1
        if nD == 1:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)
            number_of_links = 3*E_SC + 3
        elif nD == 2 or nD == 3:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+1
            number_of_links = 3*E_SC + 3
        else:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+2
            number_of_links = 3*E_SC + 3
        
        xg_len = N_SC-1 + E_SC
        xe_len = 2*N_SC - 2
        xh_len = 3*N_SC +n*nS*nQ*nD- 2 + E_SC
        xm_len = E_SC
        xm_hl_len = n*nS*nQ*nD
        xp_len = N_SC-1
        xTs_len = N_SC-1
        xTr_len = N_SC
        xc_len = 6
        
        Fg_len = N_SC-1 + E_SC
        Fe_len = 2*N_SC
        Fh_len = 3*N_SC +n*nS*nQ*nD+ E_SC
        Fc_len = 2
    elif coupling == 'single_EH':
        N_SC = nD*(nQ*(nS*(2*n-m+2)+1)+2)+1
        number_of_nodes = 3*N_SC+1
        if nD == 1:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)
            number_of_links = 3*E_SC + 3
        elif nD == 2 or nD == 3:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+1
            number_of_links = 3*E_SC + 3
        else:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+2
            number_of_links = 3*E_SC + 3
            
        xg_len = N_SC-1 + E_SC
        xe_len = 2*N_SC - 2
        xh_len = 3*N_SC +n*nS*nQ*nD- 2 + E_SC
        xm_len = E_SC
        xm_hl_len = n*nS*nQ*nD
        xp_len = N_SC-1
        xTs_len = N_SC-1
        xTr_len = N_SC
        xc_len = 6
        
        Fg_len = N_SC-1 + E_SC
        Fe_len = 2*N_SC
        Fh_len = 3*N_SC +n*nS*nQ*nD+ E_SC
        Fc_len = 2
    elif coupling == 'multi_CHP':
        N_SC = nD*(nQ*(nS*(2*n-m+2)+1)+2)+1
        number_of_nodes = 3*N_SC+nQ*nD
        if nD == 1:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)
            number_of_links = 3*E_SC + 3*nQ*nD
        elif nD == 2 or nD == 3:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+1
            number_of_links = 3*E_SC + 3*nQ*nD
        else:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+2
            number_of_links = 3*E_SC + 3*nQ*nD
            
        xg_len = N_SC-1 + E_SC
        xe_len = 2*N_SC - 2*nQ*nD - 2
        xh_len = 3*N_SC +n*nS*nQ*nD- 2*nQ*nD - 2 + E_SC
        xc_len = 6*nQ*nD
        
        Fg_len = N_SC-1 + E_SC
        Fe_len = 2*N_SC - 2
        Fh_len = 3*N_SC +n*nS*nQ*nD+ E_SC -2
        Fc_len = 2*nQ*nD
    elif coupling == 'multi_CHP2':
        N_SC = nD*(nQ*(nS*(2*n-m+2)+1)+2)+1
        number_of_nodes = 3*N_SC+nQ*nD
        if nD == 1:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)
            number_of_links = 3*E_SC + 3*nQ*nD
        elif nD == 2 or nD == 3:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+1
            number_of_links = 3*E_SC + 3*nQ*nD
        else:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+2
            number_of_links = 3*E_SC + 3*nQ*nD
        xg_len = N_SC-1 + E_SC
        xe_len = 2*N_SC - 2*nQ*nD - 1
        xh_len = 3*N_SC + n*nS*nQ*nD+1 - 2*nQ*nD + E_SC
        xc_len = 6*nQ*nD
        
        Fg_len = N_SC-1 + E_SC
        Fe_len = 2*N_SC - 1
        Fh_len = 3*N_SC + n*nS*nQ*nD+1 + E_SC 
        Fc_len = 2*nQ*nD
    elif coupling == 'multi_CHP3':
        N_SC = nD*(nQ*(nS*(2*n-m+2)+1)+2)+1
        number_of_nodes = 3*N_SC+nQ*nD
        if nD == 1:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)
        elif nD == 2 or nD == 3:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+1
        else:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+2
        number_of_links = 3*E_SC + 3*nQ*nD
        HL_h = n*nS*nQ*nD + 1# only in single-carrier part
        number_of_heat_half_links =  HL_h + nQ*nD
        
        xg_len = N_SC-1 + E_SC
        xe_len = 2*N_SC - 2*nQ*nD - 2
        xh_len = 3*N_SC +n*nS*nQ*nD- nQ*nD - 2 + E_SC
        xc_len = 5*nQ*nD
        
        Fg_len = N_SC-1 + E_SC
        Fe_len = 2*N_SC - 2
        Fh_len = 3*N_SC +n*nS*nQ*nD+ E_SC -2
        Fc_len = 2*nQ*nD
    elif coupling == 'multi_EH1':
        N_SC = nD*(nQ*(nS*(2*n-m+2)+1)+2)+1
        number_of_nodes = 3*N_SC+nQ*nD
        if nD == 1:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)
            number_of_links = 3*E_SC + 3*nQ*nD
        elif nD == 2 or nD == 3:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+1
            number_of_links = 3*E_SC + 3*nQ*nD
        else:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+2
            number_of_links = 3*E_SC + 3*nQ*nD
            
        xg_len = N_SC-1 + E_SC
        xe_len = 2*N_SC - 2*nQ*nD - 2
        xh_len = 3*N_SC + n*nS*nQ*nD - nQ*nD - 2 + E_SC
        xm_len = E_SC
        xm_hl_len = n*nS*nQ*nD
        xp_len = N_SC-1
        xTs_len = N_SC-1-nQ*nD
        xTr_len = N_SC
        xc_len = 6*nQ*nD
        
        Fg_len = N_SC-1 + E_SC
        Fe_len = 2*N_SC - 2
        Fh_len = 3*N_SC + n*nS*nQ*nD + E_SC -2
        Fc_len = 3*nQ*nD
    elif coupling == 'multi_EH2':
        N_SC = nD*(nQ*(nS*(2*n-m+2)+1)+2)+1
        number_of_nodes = 3*N_SC+nQ*nD
        if nD == 1:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)
            number_of_links = 3*E_SC + 3*nQ*nD
        elif nD == 2 or nD == 3:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+1
            number_of_links = 3*E_SC + 3*nQ*nD
        else:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+2
            number_of_links = 3*E_SC + 3*nQ*nD
            
        xg_len = N_SC-1 + E_SC
        xe_len = 2*N_SC - 2*nQ*nD - 2
        xh_len = 3*N_SC +n*nS*nQ*nD - 2 + E_SC
        xc_len = 5*nQ*nD
        
        Fg_len = N_SC-1 + E_SC
        Fe_len = 2*N_SC - 2
        Fh_len = 3*N_SC +n*nS*nQ*nD+ E_SC -2
        Fc_len = 3*nQ*nD
    elif coupling == 'multi_EH3':
        N_SC = nD*(nQ*(nS*(2*n-m+2)+1)+2)+1
        number_of_nodes = 3*N_SC+nQ*nD
        if nD == 1:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)
            number_of_links = 3*E_SC + 3*nQ*nD
        elif nD == 2 or nD == 3:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+1
            number_of_links = 3*E_SC + 3*nQ*nD
        else:
            E_SC = nD*(nQ*(nS*(2*n-m+2)+2)+1)+2
            number_of_links = 3*E_SC + 3*nQ*nD
            
        xg_len = N_SC-1 + E_SC
        xe_len = 2*N_SC - nQ*nD - 2
        xh_len = 3*N_SC + n*nS*nQ*nD - nQ*nD - 2 + E_SC
        xc_len = 5*nQ*nD
        
        Fg_len = N_SC-1 + E_SC
        Fe_len = 2*N_SC - 2
        Fh_len = 3*N_SC + n*nS*nQ*nD+ E_SC -2
        Fc_len = 3*nQ*nD
    else:
        #raise ValueError('Enter valid value for coupling')
        xg_len = xe_len = xh_len = xc_len = 0
        Fg_len = Fe_len = Fh_len = Fc_len = 0
        number_of_nodes = len(het_net.nodes)
        number_of_links = 0
    x_len = xg_len + xe_len + xh_len + xc_len
    F_len = Fg_len + Fe_len + Fh_len + Fc_len
    print('\nMy calculated network size: N = {} and E = {}'.format(number_of_nodes, number_of_links))
    print('Actual network size: N = {} and E = {}'.format(len(het_net.nodes),len(het_net.links)))
    print('My calcuated length of x: xg = {}, xe = {}, xh = {}, xc = {}, x = {}'.format(xg_len,xe_len,xh_len,xc_len,x_len))
    print('My calcuated length of xh: xm = {}, xm_hl = {}, xp = {}, xTs = {}, xTr = {}'.format(xm_len,xm_hl_len,xp_len,xTs_len,xTr_len))
    print('My calcuated length of F: Fg = {}, Fe = {}, Fh = {}, Fc = {}, F = {}'.format(Fg_len,Fe_len,Fh_len,Fc_len,F_len))
    
    # initalize network
    x0 = initialize_network(het_net, gas_net, elec_net, heat_net, nC, nD, nQ, nS, n, coupling=coupling)
    
    # extra information about system
    if number_of_nodes<max_number_of_nodes:
        for node in het_net.get_nodes():
            if isinstance(node,GasNode):
                print('gas node {} with node type {}, p = {}'.format(node.name,node.node_type,node.p))
            elif isinstance(node,ElectricalNode):
                print('elec node {} with node type {}, V = {}, delta = {}'.format(node.name,node.node_type,node.V,node.delta))
            elif isinstance(node,HeatNode):
                print('heat node {} with node type {}, p = {}, Ts = {}, Tr = {}'.format(node.name,node.node_type,node.p,node.Ts,node.Tr))
            else:
                print('coupling node {} with node type {}'.format(node.name,node.node_type))
            for hl in node.get_half_links():
                print('With half link {} with half link type {}'.format(hl.name,hl.link_type))
                if isinstance(hl,GasHalfLink):
                    print('q = {}'.format(hl.q))
                elif isinstance(hl,ElectricalHalfLink):
                    print('P = {}, Q = {}'.format(hl.P,hl.Q))
                elif isinstance(hl,HeatHalfLink):
                    print('m = {}, dphi = {}, Ts = {}, Tr = {}'.format(hl.m,hl.dphi,hl.Ts,hl.Tr))

        for link in het_net.get_links():
            if 'Ta' in link.link_params.keys():
                print('link {} with link type {} and Ta = {}'.format(link.name,link.link_type,link.link_params['Ta']))
            elif isinstance(link,GasLink):
                print('link {} with link type {} and q = {}'.format(link.name,link.link_type,link.q))
            else:
                print('link {} with link type {} and Pstart = {} and Qstart = {}'.format(link.name,link.link_type,link.Pstart,link.Qstart))
        print('')
        
        x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=form)
        print('Actual length of x: xg = {}, xe = {}, xh = {}, xc = {}, x = {}'.format(len(xg_entries),len(xe_entries),len(xh_entries),len(xc_entries),len(x_entries)))
        print('Actual length of xh: xm = {}, xm_hl = {}, xp = {}, xTs = {}, xTr = {}, sum = {}'.format(len(unknown_m_links),len(unknown_m_halflinks),len(unknown_p_nodes),len(unknown_Ts_nodes),len(unknown_Tr_nodes),np.sum([len(unknown_m_links),len(unknown_m_halflinks),len(unknown_p_nodes),len(unknown_Ts_nodes),len(unknown_Tr_nodes)])))
        F_entries, Fg_entries, Fe_entries, known_P_nodes, known_Q_nodes, Fh_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_To_halflinks, Fc_entries, F_fc_nodes, F_fc_amount, F_phi_nodes, F_To_nodes = het_net.get_F_entries(formulation=form)
        print('Actual length of F: Fg = {}, Fe = {}, Fh = {}, Fc = {}, F = {}'.format(len(Fg_entries),len(Fe_entries),len(Fh_entries),len(F_phi_nodes)+np.sum(F_fc_amount),len(Fg_entries) + len(Fe_entries) + len(Fh_entries) + Fc_len))
        # topology plot
        if plot_all:
            fig1 = plt.figure('Network topology MES {}'.format(coupling))
            ax1 = fig1.gca()
            het_net.draw_network(ax1,halflink_angle=3)
            plt.axis('equal')
            plt.axis('off')
        
        print('\nInfo about network and system of equations at initial conditions')
        nlsys = NonLinearSystemHeterogeneous(het_net,formulation=form)
        F0 = nlsys.F(x0)
        J0 = nlsys.J(x0)
        print('Actual length of x0 = {} and of F(x0) = {}'.format(len(x0),len(F0)))
        print('My calculation: length of x0 = {}, and length of F(x0) = {}'.format(x_len,F_len))
        print('determinant: |J(x0)|={}'.format(np.linalg.det(J0.todense())))
                
        # spy plot of Jacobian
        if plot_all:
            fig_J = nlsys.spy_plot_J(x0,title='Jacobian spy plot {}'.format(coupling))
            ax_J = plt.gca()
            
            # colormap of Jacobian
            fig_J_map = plt.figure(r'Jacobian {}'.format(coupling))
            J0_dense = np.matrix(np.nan*np.ones(J0.shape))
            indices = J0.indices
            indptr = J0.indptr
            for row_ind in range(J0.shape[0]):
                for col_ind in indices[indptr[row_ind]:indptr[row_ind+1]]:
                    J0_dense[row_ind,col_ind] = J0[row_ind,col_ind]
            plt.imshow(J0_dense)
            ax_J_map = plt.gca()
            nlsys.plot_J_overlay(ax_J_map)
            plt.colorbar()
    
    # scaling in solver
    D_F, D_x = scaling_matrices(het_net, nC, nD, nQ, nS, n, coupling=coupling)
    
    if number_of_nodes < max_number_of_nodes:
        D_x_inv = sps.diags(1/D_x.data[0])
        J_scaled = D_F.dot(J0.dot(D_x_inv))
        print('determinant of scaled Jacobian: |D_F J(x0) D_x_inv|={}'.format(np.linalg.det(J_scaled.todense())))
        # spy plot of scaled J over original
        if plot_all:
            ax_J.spy(J_scaled,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5)
            nlsys.plot_J_overlay(ax_J)
        
            # colormap of Jacobian
            fig_J_scaled_map = plt.figure('Scaled Jacobian {}'.format(coupling))
            J_scaled_dense = np.matrix(np.nan*np.ones(J_scaled.shape))
            indices = J_scaled.indices
            indptr = J_scaled.indptr
            for row_ind in range(J_scaled.shape[0]):
                for col_ind in indices[indptr[row_ind]:indptr[row_ind+1]]:
                    J_scaled_dense[row_ind,col_ind] = J_scaled[row_ind,col_ind]
            plt.imshow(J_scaled_dense)
            ax_J_scaled_map = plt.gca()
            nlsys.plot_J_overlay(ax_J_scaled_map)
            plt.colorbar()
        
    # solve the system
    x_sol,iters,err_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,D_F=D_F,D_x=D_x,formulation=form)
    # plot convergence
    if plot_all:
        fig_conv_het_one_solver = plt.figure('Convergence plot {} (max iters = {}, len(x) = {}, nC = {}, nQ = {}, nD = {}, nS = {}, n = {}, m = {})'.format(coupling,max_iter,x_len,nC,nQ,nD,nS,n,m))
        ax_conv_het_one_solver = fig_conv_het_one_solver.gca()
        plt.cla()
        plt.xlabel('Iteration k')
        plt.ylabel('Error ($||F(x^k)||_2$)')
        ax_conv_het_one_solver.semilogy([0,iters+1],[tol,tol],'r:')
        ax_conv_het_one_solver.semilogy(err_vec,marker='o',ls='-',color='tab:blue')
        plt.grid(which='major',color='k', linestyle='--', alpha=.2)
        plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
        
        # plot solution
        if number_of_nodes<max_number_of_nodes:
            het_net.update_full(x_sol,formulation=form)
            plt.figure('Network solution {}'.format(coupling))
            ax_sol = plt.gca()
            het_net.draw_network_value(ax_sol,halflink_length=0.5,halflink_angle=1,plot_loss=True)
            plt.axis('equal')
            plt.axis('off')
            plt.plot()
        
    return x_sol, iters, err_vec

def compare_conv_node_sets(dir_path,hydr_eq_gas='fb',form_heat='standard',heat_load='outflow',n=3,m=1,nS=1,Ta=10,p_low=.998,p_high=.999):
    coupling_types = ['CHP','EH']
    coupling_points = [1,2]
    node_sets = [1,2,3]
    
    x_sol_dict = dict()
    errors_dict = dict()
    iters_dict = dict()
    x_sol_sc = dict()
    errors_sc = dict()
    iters_sc = dict()
    
    # solver information
    tol = 1e-6
    max_iter = 50
    form_gas = 'full' # there are compressors in the network, so nodal is imposible
    formulation={'gas':form_gas,'elec':'complex_power','heat':form_heat,'het':None}
    
    
    # create network from data
    if hydr_eq_gas == 'fa':
        hydr_eq = 'q_of_dp'
    elif hydr_eq_gas == 'fb':
        hydr_eq = 'dp_of_q'
    remake_network(n,m,nS,1,1,Ta,p_low,p_high,heat_load=heat_load,hydr_eq=hydr_eq)
    
    # solve single carrier networks
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "indexing past lexsort depth may impact performance.")
        x_sol_g, iters_g, err_vec_g = solve_GN_1C(dir_path,1,show_plots=False,formulation=formulation.get('gas'))
        key = '{} {} {} {} {}'.format(form_gas,hydr_eq_gas,n,m,nS)
        x_sol_sc[key] = x_sol_g
        errors_sc[key] = err_vec_g
        iters_sc[key] = iters_g
        x_sol_e, iters_e, err_vec_e = solve_EN_1C(dir_path,1,show_plots=False)
        key = 'elec {} {} {}'.format(n,m,nS)
        x_sol_sc[key] = x_sol_e
        errors_sc[key] = err_vec_e
        iters_sc[key] = iters_e
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
            x_sol_h, iters_h, err_vec_h = solve_HN_1C(dir_path,1,form=formulation.get('heat'),show_plots=False)
            key = '{} {} {} {} {}'.format(form_heat,heat_load,n,m,nS)
            x_sol_sc[key] = x_sol_h
            errors_sc[key] = err_vec_h
            iters_sc[key] = iters_h
    
    # run load flow for MES
    for coupling_point in coupling_points:
        for coupling_type in coupling_types:
            for node_set in node_sets:
                if node_set == 3 and coupling_point == 1:
                    continue # continue to the next iteration
                print('\nRunning load flow with coupling point = {}, coupling = {}, node_set = {}'.format(coupling_point,coupling_type,node_set))
                coupling = '{}_node{}_set{}'.format(coupling_type,coupling_point,node_set)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                    warnings.filterwarnings("ignore", "Only a",UserWarning)
                    warnings.filterwarnings("ignore", "indexing past lexsort depth may impact performance.")
                    #try:
                    het_net, gas_net, elec_net, heat_net, nC, nD, nQ, nS, n, m = create_network(coupling=coupling,heat_load=heat_load)
                    # initalize network
                    x0 = initialize_network(het_net, gas_net, elec_net, heat_net, nC, nD, nQ, nS, n, coupling=coupling)
                    # scaling in solver
                    D_F, D_x = scaling_matrices(het_net, nC, nD, nQ, nS, n, coupling=coupling)
                    # solve
                    x_sol,iters,err_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,D_F=D_F,D_x=D_x,formulation=form)
                    #except:
                        #continue
                    key = '{} {} {} {} {} {} {} {} {} {}'.format(form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set,coupling_type,n,m,nS)
                    x_sol_dict[key] = x_sol
                    errors_dict[key] = err_vec
                    iters_dict[key] = iters
    return x_sol_dict, errors_dict, iters_dict, x_sol_sc, errors_sc, iters_sc, tol

def run_examples(dir_path,n,m,nS,nQ,nD,Ta,p_low,p_high,max_nodes,exs,plot_all,heat_load='outflow',hydr_eq='dp_of_q'):
    """Run steady-state powerflow for the examples"""
    # remake the network (if necessary)
    remake_network(n,m,nS,nQ,nD,Ta,p_low,p_high,heat_load=heat_load)
    
    # solve single carrier networks (and plot result)
    x_sol_g, iters_g, err_vec_g = solve_GN_1C(dir_path,max_nodes)
    x_sol_e, iters_e, err_vec_e = solve_EN_1C(dir_path,max_nodes)
    x_sol_h, iters_h, err_vec_h = solve_HN_1C(dir_path,max_nodes)
    
    # solve MES with different ways of coupling
    converged = list()  
    x_sol = dict()
    err_vec = dict()
    iters = dict()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning) # A warning of (sub)class UserWarning, of which the message starts with 'Only a GasNode object can be added' is ignored
        warnings.filterwarnings("ignore", "indexing past lexsort depth may impact performance.")
        for ex in exs:
            x_sol[ex], iters[ex], err_vec[ex] = solve_MES_1C(max_nodes,False,plot_all, coupling=ex, heat_load=heat_load)
            if err_vec.get(ex)[-1] < tol:
                converged.append(ex)    
    print('converged examples: {}'.format(converged))
    
    # plot convergence for all three cases
    labels = {'single_CHP':'single CHP', 'single_EH':'single EH', 'multi_CHP':'multiple CHP', 'multi_CHP2':'multiple CHP 2', 'multi_CHP3':'multiple CHP 3', 'multi_EH1':'multiple EH To unknown', 'multi_EH2':'multiple EH To unknown', 'multi_EH3':'multiple EH To known','CHP_node1_set1':'CHP node 1 set 1','CHP_node1_set2':'CHP node 1 set 2','EH_node1_set1':'EH node 1 set 1','EH_node1_set2':'EH node 1 set 2','CHP_node2_set1':'CHP node 1 set 1','CHP_node2_set2':'CHP node 1 set 2','CHP_node2_set3':'CHP node 1 set 3','EH_node2_set1':'EH node 2 set 1','EH_node2_set2':'EH node 2 set 2','EH_node2_set3':'EH node 2 set 3'}
    line_styles = {'single_CHP':'o-', 'single_EH':'.-', 'multi_CHP':'o-', 'multi_CHP2':'s-', 'multi_CHP3':'*-', 'multi_EH1':'o-', 'multi_EH2':'s-', 'multi_EH3':'*-','CHP_node1_set1':'o--','CHP_node1_set2':'s--','EH_node1_set1':'o--','EH_node1_set2':'s--','CHP_node2_set1':'o--','CHP_node2_set2':'s--','CHP_node2_set3':'*--','EH_node2_set1':'o--','EH_node2_set2':'s--','EH_node2_set3':'*--'}
    line_colors = {'single_CHP':'tab:blue', 'single_EH':'tab:red', 'multi_CHP':'tab:orange', 'multi_CHP2':'tab:orange', 'multi_CHP3':'tab:orange', 'multi_EH1':'tab:green', 'multi_EH2':'tab:green', 'multi_EH3':'tab:green','CHP_node1_set1':'tab:blue','CHP_node1_set2':'tab:blue','EH_node1_set1':'tab:red','EH_node1_set2':'tab:red','CHP_node2_set1':'tab:orange','CHP_node2_set2':'tab:orange','CHP_node2_set3':'tab:orange','EH_node2_set1':'tab:green','EH_node2_set2':'tab:green','EH_node2_set3':'tab:green'}
    if len(exs) > 0:
        fig_conv = plt.figure('Convergence plot comparison (n = {}, m = {}, nS = {}, nQ = {}, nD = {})'.format(n,m,nS,nQ,nD))
        ax_conv = fig_conv.gca()
        plt.cla()
        plt.xlabel(r'Iteration $k$')
        plt.ylabel(r'Error $||F(x^k)||_2$')
        
        
        max_iters = 0
        for key, err in err_vec.items():
            max_iters = max(max_iters, iters.get(key))
            ax_conv.semilogy(err,line_styles.get(key),color=line_colors.get(key),label=labels.get(key))
        ax_conv.semilogy([0,max_iters+1],[tol,tol],'r:')
        ax_conv.grid(which='major',color='k', linestyle='--', alpha=.2)
        ax_conv.grid(which='minor',color='k', linestyle=':', alpha=.05)
        xmin = 0
        xmax = max_iters
        xticks = range(xmin,xmax+1) # make sure the xticks are integers
        ax_conv.set_xlim(xmin=xmin,xmax=xmax+1)
        ax_conv.set_xticks(xticks)
        ax_conv.legend()
    
def plot_conv_mes_node_forms(x_sol, errors, iters, tol):
    """Create convergence plots for the dictionaries created by compare_conv_models"""
    max_iters_used = 0
    couplings_used = list()
    for key_mes, iter_mes in iters.items():
        max_iters_used = max(max_iters_used,iter_mes)
    for key_mes, err_vec in errors.items():
        key_list = key_mes.split(' ')
        form_gas = key_list[0]
        hydr_eq = key_list[1]
        form_heat = key_list[2]
        heat_load = key_list[3]
        coupling_point = int(key_list[4])
        node_set = int(key_list[5])
        coupling_type = key_list[6]
        n = int(key_list[7])
        m = int(key_list[8])
        nS = int(key_list[9])
        fig = plt.figure('conv_mes_n{}_m{}_nS{}_node{}_{}_{}_{}_{}'.format(n,m,nS,coupling_point,form_gas,hydr_eq,form_heat,heat_load))
        ax = fig.gca()
        if not ax.lines:
            ax.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
        ls = '-'
        color = colors.get(coupling_type)
        marker = markers.get(node_set)
        label = '{}, node {}, set {}'.format(coupling_type,coupling_point,node_set)
        ax.semilogy(err_vec,marker=marker,ls=ls,color=color,label=label)
        ax.set_xlabel(r'Iteration $k$')
        ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
        ax.grid(which='major',color='k', linestyle='--', alpha=.2)
        ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
        xmin = 0
        xmax = max_iters_used
        xticks = range(xmin,xmax+1) # make sure the xticks are integers
        ax.set_xlim(left=xmin,right=xmax+1)
        ax.set_xticks(xticks)
        ax.legend()
    return ax

def plots_pscc_2020_forms(dir_path,n=3,m=1,nS=1,Ta=10,p_low=.998,p_high=.999,save_fig=False):
    x_sol = dict()
    errors = dict()
    iters = dict()
    x_sol_sc = dict()
    errors_sc = dict()
    iters_sc = dict()
    for hydr_eq_gas in ['fa','fb']:
        for form_heat in ['standard','half_link_flow']:
            for heat_load in ['outflow','delta']:
                x_sol_dict, errors_dict, iters_dict, x_sol_sc_dict, errors_sc_dict, iters_sc_dict, tol = compare_conv_node_sets(dir_path,hydr_eq_gas=hydr_eq_gas,form_heat=form_heat,heat_load=heat_load,n=n,m=m,nS=nS,Ta=Ta,p_low=p_low,p_high=p_high)
                x_sol.update(x_sol_dict)
                errors.update(errors_dict)
                iters.update(iters_dict)
                x_sol_sc.update(x_sol_sc_dict)
                errors_sc.update(errors_sc_dict)
                iters_sc.update(iters_sc_dict)
    plot_conv_mes_node_forms(x_sol, errors, iters, tol)
    
    # plot convergence sc
    fig_conv_gas = plt.figure('conv_gas_n{}_m{}_nS{}'.format(n,m,nS))
    fig_conv_elec = plt.figure('conv_elec_n{}_m{}_nS{}'.format(n,m,nS))
    fig_conv_heat = plt.figure('conv_heat_n{}_m{}_nS{}'.format(n,m,nS))
    max_iters_sc = 0
    for key_sc, iter_sc in iters_sc.items():
        if 'nodal' in key_sc or 'full' in key_sc: #gas
            ax = plt.figure('conv_gas_n{}_m{}_nS{}'.format(n,m,nS)).gca()
            key_list = key_sc.split(' ')
            form_gas = key_list[0]
            hydr_eq = key_list[1]
            key_gas = '{} {}'.format(form_gas,hydr_eq)
            ls = linestyles_gas.get(key_gas)
            color = 'tab:green'
            marker = '.'
            label = '{}, {}'.format(form_gas,hydr_eq)
        elif 'elec' in key_sc: #electrical
            ax = plt.figure('conv_elec_n{}_m{}_nS{}'.format(n,m,nS)).gca()
            ls = '-'
            color = 'tab:red'
            marker = '.'
            label = None
        else: #heat
            ax = plt.figure('conv_heat_n{}_m{}_nS{}'.format(n,m,nS)).gca()
            key_list = key_sc.split(' ')
            form_heat = key_list[0]
            heat_load = key_list[1]
            key_heat = '{} {}'.format(form_heat,heat_load)
            ls = linestyles_heat.get(key_heat)
            color = 'tab:blue'
            marker = '.'
            label = '{}, {}'.format(form_heat,heat_load)
        ax.semilogy(errors_sc.get(key_sc),marker=marker,ls=ls,color=color,label=label)
        max_iters_sc = max(max_iters_sc,iter_sc)
    xmin = 0
    xmax = max_iters_sc
    xticks = range(xmin,xmax+1) # make sure the xticks are integers
    for fig_num in ['conv_gas_n{}_m{}_nS{}'.format(n,m,nS),'conv_elec_n{}_m{}_nS{}'.format(n,m,nS),'conv_heat_n{}_m{}_nS{}'.format(n,m,nS)]:
        ax = plt.figure(fig_num).gca()
        ax.set_xlabel(r'Iteration $k$')
        ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
        ax.semilogy([0,max_iters_sc+1],[tol,tol],'k:',label='tolerance')
        ax.grid(which='major',color='k', linestyle='--', alpha=.2)
        ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
        ax.set_xlim(left=xmin,right=xmax+1)
        ax.set_xticks(xticks)
        ax.legend()

    if save_fig:
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            path_to_fig = os.path.join(dir_path,'Figures','MES1C')
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))
    
    # print the iteration results, in order to put them in the table in the paper
    print('\nNR iterations (CHP & EH):')
    form_gas = 'full'
    for coupling_point in [1,2]:
        for hydr_eq_gas in ['fa','fb']:
            if not (hydr_eq_gas == 'fb' and form_gas == 'nodal'):
                for form_heat in ['standard','half_link_flow']:
                    for heat_load in ['outflow','delta']:
                        for node_set in [1,2,3]:
                            if not (coupling_point == 1 and node_set == 3):
                                for coupling_type in ['CHP','EH']:
                                    key_CHP = '{} {} {} {} {} {} CHP {} {} {}'.format(form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set,n,m,nS)
                                    key_EH = '{} {} {} {} {} {} EH {} {} {}'.format(form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set,n,m,nS)
                                if errors.get(key_CHP)[-1] < tol:
                                    result_CHP = iters.get(key_CHP)
                                else:
                                    result_CHP = 'not conv.'
                                if errors.get(key_EH)[-1] < tol:
                                    result_EH = iters.get(key_EH)
                                else:
                                    result_EH = 'not conv.'
                                print('c. point: {}, {} {}, {} {}, node set: {}, {} & {} '.format(coupling_point,form_gas,hydr_eq_gas,form_heat,heat_load,node_set,result_CHP,result_EH))
if __name__ == '__main__':
    ## parse the command line
    #args = command_line_input.parse_args()
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    
    #heat_load = 'outflow'
    #hydr_eq='dp_of_q'
    #run_examples(dir_path,args.n,args.m,args.nS,args.nQ,args.nD,args.Ta,args.p_low,args.p_high,args.max_nodes,args.ex,args.plot_all,heat_load=heat_load,hydr_eq=hydr_eq)   
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plots_pscc_2020_forms(dir_path,n=10,m=4,nS=1,Ta=10,p_low=.998,p_high=.999,save_fig=False)
    
    plt.show()
