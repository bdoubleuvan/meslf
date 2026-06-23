"""
MES consisting of 3 nodes per energy carrier. Also called the reduced benchmark problem.
"""
import warnings
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.utils.constants import mbar, bar, hour, kV, MW, MBTU, BTU
import examples.GN_BP as GasNet
import examples.EN_BP as ElecNet
import examples.HN_BP as HeatNet
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import os
import pandas as pd
from meslf.networks.read_write_network import from_pd_dataframes
import pytest

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

# Read the scenario data
def read_scen_data(path_to_data,topology=1,single_coupling=False):
    """Read the scenario data

    Parameters
    ----------
    topology : int, optional
        Determines which topology is used in the MES, hence, which is used in the heat ntework when the coupling components are taken into account separately. Options are 1-4. Default is 1.
    single_coupling : bool, optional
        Determines if a single coupling node (either CHP or EH) is used in the MES, when coupled to one heat node and one heat node (i.e., when topology 1 is used). Default is False.
    """
    mes_data = 'top'+str(topology)
    if single_coupling:
        mes_data += '_1c'
    else:
        mes_data += '_2c'
    nodes_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_nodes_'+mes_data+'.pkl'))
    links_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_links_'+mes_data+'.pkl'))
    halflinks_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_halflinks_'+mes_data+'.pkl'))
    mes_net_scen = from_pd_dataframes(nodes_mes,links_mes,halflinks_mes)

    heat_nodes = [node for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
    coupling_nodes = [node for node in mes_net_scen.get_nodes() if isinstance(node,HeterogeneousNode)]

    # (full) solution of scenario, multi-carrier
    # gas part
    pg_mes_scen = [node.get_p() for node in mes_net_scen.get_nodes() if isinstance(node,GasNode)]
    q_mes_scen = [link.get_q() for link in mes_net_scen.get_links() if isinstance(link,GasLink)]
    q_hl_mes_scen = [hl.get_q() for node in mes_net_scen.get_nodes() if isinstance(node,GasNode) for hl in node.get_half_links()]
    xg = [node.x for node in mes_net_scen.get_nodes() if isinstance(node,GasNode)]
    yg = [node.y for node in mes_net_scen.get_nodes() if isinstance(node,GasNode)]
    # electricity part
    delta_mes_scen = [node.get_delta() for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode)]
    V_mes_scen = [node.get_V() for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode)]
    P_edge_start_mes_scen = [link.get_Pstart() for link in mes_net_scen.get_links() if isinstance(link,ElectricalLink)]
    P_edge_end_mes_scen = [link.get_Pend() for link in mes_net_scen.get_links() if isinstance(link,ElectricalLink)]
    P_edge_mes_scen = P_edge_start_mes_scen + P_edge_end_mes_scen
    Q_edge_start_mes_scen = [link.get_Qstart() for link in mes_net_scen.get_links() if isinstance(link,ElectricalLink)]
    Q_edge_end_mes_scen = [link.get_Qend() for link in mes_net_scen.get_links() if isinstance(link,ElectricalLink)]
    Q_edge_mes_scen = Q_edge_start_mes_scen + Q_edge_end_mes_scen
    P_inj_mes_scen = [hl.get_P() for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode) for hl in node.get_half_links()]
    Q_inj_mes_scen = [hl.get_Q() for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode) for hl in node.get_half_links()]
    xe = [node.x for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode)]
    ye = [node.y for node in mes_net_scen.get_nodes() if isinstance(node,ElectricalNode)]
    # heat part
    m_mes_scen = [link.get_m() for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
    m_hl_mes_scen = [hl.get_m() for node in heat_nodes for hl in node.get_half_links()] + [hl.get_m() for node in coupling_nodes for hl in node.get_half_links()]
    ph_mes_scen = [node.get_p() for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
    Ts_mes_scen = [node.get_Ts() for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
    Tr_mes_scen = [node.get_Tr() for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
    Tsstart_mes_scen = [link.get_Tsstart() for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
    Trstart_mes_scen = [link.get_Trstart() for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
    dphistart_mes_scen = [link.get_dphistart() for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
    Ts_hl_mes_scen = [hl.get_Ts()  for node in heat_nodes for hl in node.get_half_links()] + [hl.get_Ts() for node in coupling_nodes for hl in node.get_half_links()]
    Tr_hl_mes_scen = [hl.get_Tr()  for node in heat_nodes for hl in node.get_half_links()] + [hl.get_Tr() for node in coupling_nodes for hl in node.get_half_links()]
    phi_hl_mes_scen = [hl.get_dphi() for node in heat_nodes for hl in node.get_half_links()] + [hl.get_dphi() for node in coupling_nodes for hl in node.get_half_links()]
    xh = [node.x for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
    yh = [node.y for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
    # coupling part (the actual values are included in the single-carrier parts)
    xc = [node.x for node in mes_net_scen.get_nodes() if isinstance(node,HeterogeneousNode)]
    yc = [node.y for node in mes_net_scen.get_nodes() if isinstance(node,HeterogeneousNode)]

    return mes_net_scen,xg,yg,xe,ye,xh,yh,xc,yc,q_mes_scen,pg_mes_scen,q_hl_mes_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen,P_edge_mes_scen,Q_edge_mes_scen,m_mes_scen,m_hl_mes_scen,ph_mes_scen,Ts_mes_scen,Tr_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_hl_mes_scen

def sol_from_scen_data(path_to_data,het_net,topology=1,single_coupling=False,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}):
    """Determine the solution vector from the scenario data.

    Parameters
    ----------
    topology : int, optional
        Determines which topology is used in the MES, hence, which is used in the heat ntework when the coupling components are taken into account separately. Options are 1-4. Default is 1.
    single_coupling : bool, optional
        Determines if a single coupling node (either CHP or EH) is used in the MES, when coupled to one heat node and one heat node (i.e., when topology 1 is used). Default is False.

    Returns
    -------
    xc_sol : np array
        Solution vector, in S.I. units.
    """
    # read scenario data (to get the values not included in the created network)
    mes_net_scen,_,_,_,_,_,_,_,yc,q_mes_scen,pg_mes_scen,q_hl_mes_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen,P_edge_mes_scen,Q_edge_mes_scen,m_mes_scen,m_hl_mes_scen,ph_mes_scen,Ts_mes_scen,Tr_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_hl_mes_scen = read_scen_data(path_to_data,topology=topology,single_coupling=single_coupling)

    # create solution vector from this data (since not all this data is assigned to the network when reading from a df)
    het_net.initialize() # to assign numbers to nodes and link
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)
    # gas part
    q_sol = np.array([q_mes_scen[entry.number] for entry in xg_entries if isinstance(entry,GasLink)])
    pg_sol = np.array([pg_mes_scen[entry.number] for entry in xg_entries if isinstance(entry,GasNode)])
    xg_sol = np.concatenate((q_sol,pg_sol))

    # electrical part
    delta_sol = np.array([delta_mes_scen[node.number] for node in unknown_delta_nodes])
    V_sol = np.array([V_mes_scen[node.number] for node in unknown_V_nodes])
    xe_sol = np.concatenate((delta_sol,V_sol))

    m_sol = np.array([m_mes_scen[link.number] for link in unknown_m_links]) #[kg/s]
    m_hl_sol = np.array([m_hl_mes_scen[ind_hl] for ind_hl,halflink in enumerate(unknown_m_halflinks)]) #[kg/s]
    ph_sol = np.array([ph_mes_scen[node.number] for node in unknown_p_nodes])
    Ts_sol = np.array([Ts_mes_scen[node.number] for node in unknown_Ts_nodes]) #[C]
    Tr_sol = np.array([Tr_mes_scen[node.number] for node in unknown_Tr_nodes]) #[C]
    Ts_hl_sol = np.array([Ts_hl_mes_scen[ind_hl] for ind_hl,halflink in enumerate(unknown_Ts_halflinks)])
    Tr_hl_sol = np.array([Tr_hl_mes_scen[ind_hl] for ind_hl,halflink in enumerate(unknown_Tr_halflinks)])
    xh_sol = np.concatenate((m_sol,m_hl_sol,ph_sol,Ts_sol,Tr_sol,Ts_hl_sol,Tr_hl_sol))

    # coupling
    qc_sol = np.array([q_mes_scen[link.number] for link in unknown_qc_links])
    Pc_sol = np.array([P_edge_mes_scen[link.number] for link in unknown_Pc_links])
    Qc_sol = np.array([Q_edge_mes_scen[link.number] for link in unknown_Qc_links])
    Sc_sol = np.concatenate((Pc_sol,Qc_sol))
    mc_sol = np.array([m_mes_scen[link.number] for link in unknown_mc_links])
    phic_sol = np.array([-dphistart_mes_scen[link.number] for link in unknown_dphi_links])
    Tsc_sol = np.array([Tsstart_mes_scen[link.number] for link in unknown_Ts_links])
    Trc_sol = np.array([Trstart_mes_scen[link.number] for link in unknown_Tr_links])

    xc_sol = np.concatenate((qc_sol,Sc_sol,mc_sol,phic_sol,Tsc_sol,Trc_sol))

    # combine into multi-carrier
    x_sol  = np.concatenate((xg_sol,xe_sol,xh_sol,xc_sol))
    return x_sol

def create_network(path_to_data,hydr_eq_gas='fa',heat_load='outflow',topology=1,node_set=1,single_coupling=False,EH=True):
    """Create the multi-carrier network.

    Parameters
    ----------------
    topology : int, optional
        Determines which topology is used. Options are 1-4. Default is 1.
    node_set : int, optional
        Determines which node set is used. Options depend on the chosen topology. Default is 1.
    single_coupling : bool, optional
        Determines if a single coupling node (either CHP or EH) is used in the MES, when coupled to one gas node and one heat node (i.e., when topology 1 is used). Default is False. Only used when topology is 1 and if c_hl is True.

    Returns
    -----------
    het_net : HeterogeneousNetwork
        The heterogeneous network
    """
    if not topology in [1,2,3,4]:
        raise ValueError('Enter valid value for topology')
    mes_net_scen,xg,yg,xe,ye,xh,yh,xc,yc,q_mes_scen,pg_mes_scen,q_hl_mes_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen,P_edge_mes_scen,Q_edge_mes_scen,m_mes_scen,m_hl_mes_scen,ph_mes_scen,Ts_mes_scen,Tr_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_hl_mes_scen = read_scen_data(path_to_data,topology=topology,single_coupling=single_coupling)

    # coupling nodes
    coupling_nodes = [node for node in mes_net_scen.get_nodes() if isinstance(node,HeterogeneousNode)]
    # create single carrier networks
    gas_net = GasNet.create_network(path_to_data,hydr_eq=hydr_eq_gas,c_hl=True,topology=topology,single_coupling=single_coupling)
    elec_net = ElecNet.create_network(path_to_data,c_hl=True)
    heat_net = HeatNet.create_network(path_to_data,c_hl=True,heat_load=heat_load,topology=topology,single_coupling=single_coupling)
    # remove the (half) links that have to do with the coupling (removing things from lists (generators) while iterating over them doesn't work. So overwrite the list with a new list with only the items you want)
    for gas_node in gas_net.get_nodes():
        gas_node.half_links[:] = (hl for ind_hl,hl in enumerate(gas_node.half_links) if ind_hl == 0)
    for elec_node in elec_net.get_nodes():
        elec_node.half_links[:] = (hl for ind_hl,hl in enumerate(elec_node.half_links) if ind_hl == 0)

    extra_nodes = [node for ind_n,node in enumerate(heat_net.nodes) if ind_n > 2]
    for heat_node in extra_nodes:
        heat_net.remove_node(heat_node) # automatically removes the (dummy) links they are connected to
    for heat_node in heat_net.get_nodes(): # remove additional half links, but they shouldn't be there, since for heat the coupling are modeled as additional nodes and dummy links instead of half links
        for ind_hl,hl in enumerate(heat_node.get_half_links()):
            if ind_hl > 0:
                heat_net.remove_half_link(hl)
        heat_node.half_links[:] = (hl for ind_hl,hl in enumerate(heat_node.half_links) if ind_hl == 0)
    water = heat_net.links[0].link_params.get('carrier')

    # coupling
    elec_coupling_node = elec_net.nodes[1]
    if topology == 1:
        gas_coupling_node = gas_net.nodes[2]
        heat_coupling_node = heat_net.nodes[0]
        if single_coupling:
            if EH:
                unit_type = coupling_nodes[0].unit_type
                unit_params = coupling_nodes[0].unit_params
            else:
                unit_type = 'geh_CHP'
                unit_params_EH = coupling_nodes[0].unit_params
                C_EH = unit_params_EH.get('C')
                eta_CHP = np.array([2*C_EH[0,0], 2*C_EH[1,0]])
                unit_params = {'eta':eta_CHP,'GHV':unit_params_EH.get('GHV')}
            To_c = Tsstart_mes_scen[3]
            dTc = To_c - Tr_mes_scen[0]
            if EH:
                if node_set in [1,2,3,4,6,8]:
                    cn = HeterogeneousNode('cn',node_type=0,x=xc[0],y=yc[0],unit_type=unit_type,unit_params=unit_params) # To unknown
                    hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
                elif node_set in [5,7,19,20,21,22]:
                    if heat_load == 'outflow':
                        cn = HeterogeneousNode('cn',node_type=1,x=xc[0],y=yc[0],unit_type=unit_type,unit_params=unit_params) # To known
                        hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c) # To of coupling (source) is known
                    else:
                        cn = HeterogeneousNode('cn',node_type=2,x=xc[0],y=yc[0],unit_type=unit_type,unit_params=unit_params) # dT known
                        hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dTc) # dT of coupling (source) is known
            else:
                if heat_load == 'outflow':
                    cn = HeterogeneousNode('cn',node_type=1,x=xc[0],y=yc[0],unit_type=unit_type,unit_params=unit_params) # To known
                    hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c) # To of coupling (source) is known
                else:
                    cn = HeterogeneousNode('cn',node_type=2,x=xc[0],y=yc[0],unit_type=unit_type,unit_params=unit_params) # dT known
                    hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dTc) # dT of coupling (source) is known
            glc = GasLink('gl_c',gas_coupling_node,cn)
            elc = ElectricalLink('el_c',cn,elec_coupling_node)
        else: # topology 1, two coupling nodes
            unit_type_GB=coupling_nodes[0].unit_type
            unit_params_GB=coupling_nodes[0].unit_params
            To_c_GB = Tsstart_mes_scen[3]
            dT_c_GB = To_c_GB - Tr_mes_scen[0]
            unit_type_CHP=coupling_nodes[1].unit_type
            unit_params_CHP=coupling_nodes[1].unit_params
            To_c_CHP = Tsstart_mes_scen[4] #[C]
            dT_c_CHP = To_c_CHP - Tr_mes_scen[0]
            if heat_load == 'outflow':
                cn0 = HeterogeneousNode('cn0',node_type=1,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
                cn1 = HeterogeneousNode('cn1',node_type=1,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To known
                hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c_GB) # To of coupling (source) is known
                hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c_CHP) # To of coupling (source) is known
            else:
                cn0 = HeterogeneousNode('cn0',node_type=2,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # dT known
                cn1 = HeterogeneousNode('cn1',node_type=2,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # dT known
                hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dT_c_GB) # dT of coupling (source) is known
                hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dT_c_CHP) # dT of coupling (source) is known
            glc0 = GasLink('gl_c0',gas_coupling_node,cn0)
            glc1 = GasLink('gl_c1',gas_coupling_node,cn1)
            elc = ElectricalLink('el_c',cn1,elec_coupling_node)
    elif topology == 2:
        if single_coupling:
            raise ValueError('single coupling not implemented for topology 2')
        else:
            gas_coupling_node0 = gas_net.nodes[1]
            gas_coupling_node1 = gas_net.nodes[2]
            heat_coupling_node = heat_net.nodes[0]
            unit_type_GB=coupling_nodes[0].unit_type
            unit_params_GB=coupling_nodes[0].unit_params
            To_c_GB = coupling_nodes[0].half_links[0].get_To()
            dT_c_GB = To_c_GB - Tr_mes_scen[0]
            unit_type_CHP=coupling_nodes[1].unit_type
            unit_params_CHP=coupling_nodes[1].unit_params
            To_c_CHP = coupling_nodes[1].half_links[0].get_To() #[C]
            dT_c_CHP = To_c_CHP - Tr_mes_scen[1]
            if node_set == 1:
                cn0 = HeterogeneousNode('cn0',node_type=0,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # To unknown
                cn1 = HeterogeneousNode('cn1',node_type=0,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To unknown
                hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
                hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
            elif node_set == 2:
                cn0 = HeterogeneousNode('cn0',node_type=0,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # To unknown
                hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
                if heat_load == 'outflow':
                    cn1 = HeterogeneousNode('cn1',node_type=1,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To known
                    hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c_CHP) # To of coupling (source) is known
                else:
                    cn1 = HeterogeneousNode('cn1',node_type=2,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # dT known
                    hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dT_c_CHP) # dT of coupling (source) is known
            elif node_set == 3:
                if heat_load == 'outflow':
                    cn0 = HeterogeneousNode('cn0',node_type=1,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
                    hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c_GB) # To of coupling (source) is known
                else:
                    cn0 = HeterogeneousNode('cn0',node_type=2,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB,) # dT known
                    hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dT_c_GB) # dT of coupling (source) is known
                cn1 = HeterogeneousNode('cn1',node_type=0,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To unknown
                hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
            elif node_set == 4 or node_set == 5:
                if heat_load == 'outflow':
                    cn0 = HeterogeneousNode('cn0',node_type=1,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
                    cn1 = HeterogeneousNode('cn1',node_type=1,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To known
                    hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c_GB) # To of coupling (source) is known
                    hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c_CHP) # To of coupling (source) is known
                else:
                    cn0 = HeterogeneousNode('cn0',node_type=2,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB,dT=dT_c_GB) # dT known
                    cn1 = HeterogeneousNode('cn1',node_type=2,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP,dT=dT_c_CHP) # dT known
                    hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dT_c_GB) # dT of coupling (source) is known
                    hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dT_c_CHP) # dT of coupling (source) is known
            else:
                raise ValueError('Enter valid node set for topology 2 with two coupling nodes')
            glc0 = GasLink('gl_c0',gas_coupling_node0,cn0)
            glc1 = GasLink('gl_c1',gas_coupling_node1,cn1)
            elc = ElectricalLink('el_c',cn1,elec_coupling_node)
    elif topology == 3:
        if single_coupling:
            raise ValueError('single coupling not implemented for topology 3')
        else:
            gas_coupling_node = gas_net.nodes[2]
            heat_coupling_node0 = heat_net.nodes[0]
            heat_coupling_node1 = heat_net.nodes[1]
            unit_type_GB=coupling_nodes[0].unit_type
            unit_params_GB=coupling_nodes[0].unit_params
            To_c_GB = coupling_nodes[0].half_links[0].get_To()
            dT_c_GB = To_c_GB - Tr_mes_scen[0]
            unit_type_CHP=coupling_nodes[1].unit_type
            unit_params_CHP=coupling_nodes[1].unit_params
            To_c_CHP = coupling_nodes[1].half_links[0].get_To() #[C]
            dT_c_CHP = To_c_CHP - Tr_mes_scen[1]
            if node_set == 1:
                cn0 = HeterogeneousNode('cn0',node_type=0,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # To unknown
                cn1 = HeterogeneousNode('cn1',node_type=0,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To unknown
                hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
                hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
            elif node_set == 2:
                cn0 = HeterogeneousNode('cn0',node_type=0,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # To unknown
                hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
                if heat_load == 'outflow':
                    cn1 = HeterogeneousNode('cn1',node_type=1,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To known
                    hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c_CHP) # To of coupling (source) is known
                else:
                    cn1 = HeterogeneousNode('cn1',node_type=2,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # dT known
                    hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dT_c_CHP) # dT of coupling (source) is known
            elif node_set == 3:
                if heat_load == 'outflow':
                    cn0 = HeterogeneousNode('cn0',node_type=1,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
                    hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c_GB) # To of coupling (source) is known
                else:
                    cn0 = HeterogeneousNode('cn0',node_type=2,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # dT known
                    hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dT_c_GB) # dT of coupling (source) is known
                cn1 = HeterogeneousNode('cn1',node_type=0,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To unknown
                hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
            elif node_set in [4,5,6,7,8]:
                if heat_load == 'outflow':
                    cn0 = HeterogeneousNode('cn0',node_type=1,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB,To=To_c_GB) # To known
                    cn1 = HeterogeneousNode('cn1',node_type=1,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP,To=To_c_CHP) # To known
                    hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c_GB) # To of coupling (source) is known
                    hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=6,Tsstart=To_c_CHP) # To of coupling (source) is known
                else:
                    cn0 = HeterogeneousNode('cn0',node_type=2,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB,dT=dT_c_GB) # dT known
                    cn1 = HeterogeneousNode('cn1',node_type=2,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP,dT=dT_c_CHP) # dT known
                    hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dT_c_GB) # dT of coupling (source) is known
                    hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node,link_params={'carrier':water},bc_type=10,dTstart=dT_c_CHP) # dT of coupling (source) is known
            else:
                raise ValueError('Enter valid node set for topology 3 with two coupling nodes')
            glc0 = GasLink('gl_c0',gas_coupling_node,cn0)
            glc1 = GasLink('gl_c1',gas_coupling_node,cn1)
            elc = ElectricalLink('el_c',cn1,elec_coupling_node)
    elif topology == 4:
        if single_coupling:
            raise ValueError('single coupling not implemented for topology 4')
        else:
            gas_coupling_node0 = gas_net.nodes[1]
            gas_coupling_node1 = gas_net.nodes[2]
            heat_coupling_node0 = heat_net.nodes[0]
            heat_coupling_node1 = heat_net.nodes[1]
            unit_type_GB=coupling_nodes[0].unit_type
            unit_params_GB=coupling_nodes[0].unit_params
            To_c_GB = Tsstart_mes_scen[3]
            dT_c_GB = To_c_GB - Tr_mes_scen[0]
            unit_type_CHP=coupling_nodes[1].unit_type
            unit_params_CHP=coupling_nodes[1].unit_params
            To_c_CHP = Tsstart_mes_scen[4] #[C]
            dT_c_CHP = To_c_CHP - Tr_mes_scen[1]
            if node_set == 1:
                cn0 = HeterogeneousNode('cn0',node_type=0,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # To unknown
                cn1 = HeterogeneousNode('cn1',node_type=0,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To unknown
                hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node0,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
                hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node1,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
            elif node_set == 2:
                if heat_load == 'outflow':
                    cn0 = HeterogeneousNode('cn0',node_type=1,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
                    cn1 = HeterogeneousNode('cn1',node_type=1,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To known
                    hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node0,link_params={'carrier':water},bc_type=6,Tsstart=To_c_GB) # To of coupling (source) is known
                    hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node1,link_params={'carrier':water},bc_type=6,Tsstart=To_c_CHP) # To of coupling (source) is known
                else:
                    cn0 = HeterogeneousNode('cn0',node_type=2,x=xc[0],y=yc[0],unit_type=unit_type_GB,unit_params=unit_params_GB) # dT known
                    cn1 = HeterogeneousNode('cn1',node_type=2,x=xc[1],y=yc[1],unit_type=unit_type_CHP,unit_params=unit_params_CHP) # dT known
                    hlc0 = HeatLink('hl_c0',cn0,heat_coupling_node0,link_params={'carrier':water},bc_type=10,dTstart=dT_c_GB) # dT of coupling (source) is known
                    hlc1 = HeatLink('hl_c1',cn1,heat_coupling_node1,link_params={'carrier':water},bc_type=10,dTstart=dT_c_CHP) # dT of coupling (source) is known
            else:
                raise ValueError('Enter valid node set for topology 4 with two coupling nodes')
            glc0 = GasLink('gl_c0',gas_coupling_node0,cn0)
            glc1 = GasLink('gl_c1',gas_coupling_node1,cn1)
            elc = ElectricalLink('el_c',cn1,elec_coupling_node)
    else:
        raise ValueError('Enter valid value for topology')


    # additional boundary conditions
    q0_load = q_hl_mes_scen[0]
    pg1 = pg_mes_scen[1]
    pg2 = pg_mes_scen[2]

    delta1 = delta_mes_scen[1]
    P0_load = P_inj_mes_scen[0]
    P1_load = P_inj_mes_scen[1]
    Q1_load = Q_inj_mes_scen[1]
    P2_load = P_inj_mes_scen[2]
    Q2_load = Q_inj_mes_scen[2]

    ph1 = ph_mes_scen[1]
    ph2 = ph_mes_scen[2]
    Tr1 = Tr_mes_scen[1]
    # in the MES, node 0 is a junction, so it doesn't have a half link connected to it. So the first half link is the one connected to node 1
    Ts_hl1 = Ts_hl_mes_scen[0]
    Tr_hl1 = Tr_hl_mes_scen[0]
    dT1 = Ts_hl1 - Tr_hl1
    Ts_hl2 = Ts_hl_mes_scen[1]
    Tr_hl2 = Tr_hl_mes_scen[1]
    dT2 = Ts_hl2 - Tr_hl2

    # change node types of homogeneous nodes
    if topology == 1:
        if single_coupling:
            if EH: # topology 1, EH
                if node_set == 1:
                    elec_net.nodes[0].node_type = 1 # gen
                    ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                    elec_coupling_node.node_type = 5 # PQVdelta
                    elec_coupling_node.half_links[0].Q = Q1_load
                    elec_coupling_node.delta = delta1
                    heat_coupling_node.node_type = 5 # reference (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                elif node_set == 2:
                    gas_net.nodes[2].node_type = 3 # ref. load node
                    gas_net.nodes[2].p = pg2
                    elec_net.nodes[0].node_type = 1 # gen
                    ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                    elec_coupling_node.node_type = 4 # QVdelta
                    elec_coupling_node.half_links[0].Q = Q1_load
                    heat_coupling_node.node_type = 5 # reference (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                elif node_set == 3:
                    elec_coupling_node.node_type = 5 # PQVdelta
                    elec_coupling_node.half_links[0].Q = Q1_load
                    elec_coupling_node.delta = delta1
                    heat_coupling_node.node_type = 5 # reference (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                elif node_set == 4:
                    gas_net.nodes[2].node_type = 3 # ref. load node
                    gas_net.nodes[2].p = pg2
                    elec_coupling_node.node_type = 6 # PQV
                    elec_coupling_node.half_links[0].Q = Q1_load
                    heat_coupling_node.node_type = 5 # reference (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                elif node_set == 5:
                    elec_net.nodes[0].node_type = 1 # gen
                    ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                    elec_coupling_node.node_type = 4 # QVdelta
                    elec_coupling_node.half_links[0].Q = Q1_load
                    heat_coupling_node.node_type = 5 # reference (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                elif node_set == 6:
                    elec_net.nodes[0].node_type = 1 # gen
                    ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                    elec_coupling_node.node_type = 4 # QVdelta
                    elec_coupling_node.half_links[0].Q = Q1_load
                    heat_coupling_node.node_type = 7 # reference temperature (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                elif node_set == 7:
                    elec_coupling_node.node_type = 6 # PQV
                    elec_coupling_node.half_links[0].Q = Q1_load
                    heat_coupling_node.node_type = 5 # reference (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                elif node_set == 8:
                    elec_coupling_node.node_type = 6 # PQV
                    elec_coupling_node.half_links[0].Q = Q1_load
                    heat_coupling_node.node_type = 7 # reference temperature (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                elif node_set == 19:
                    gas_net.nodes[2].node_type = 3 # ref. load node
                    gas_net.nodes[2].p = pg2
                    elec_net.nodes[0].node_type = 1 # gen
                    ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                    elec_coupling_node.node_type = 4 # QVdelta
                    elec_coupling_node.half_links[0].Q = Q1_load
                elif node_set == 20:
                    gas_net.nodes[2].node_type = 3 # ref. load node
                    gas_net.nodes[2].p = pg2
                    elec_net.nodes[0].node_type = 1 # gen
                    ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                    elec_coupling_node.node_type = 4 # QVdelta
                    elec_coupling_node.half_links[0].Q = Q1_load
                    heat_coupling_node.node_type = 2 # junction node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                    if heat_load == 'outflow':
                        heat_net.nodes[1].node_type = 8 # sink slack node
                    else:
                        heat_net.nodes[1].node_type = 11 # slack return temp. node
                    heat_net.nodes[1].p = ph1
                    heat_net.nodes[1].Tr = Tr1
                elif node_set == 21:
                    gas_net.nodes[2].node_type = 3 # ref. load node
                    gas_net.nodes[2].p = pg2
                    elec_coupling_node.node_type = 6 # PQV
                    elec_coupling_node.half_links[0].Q = Q1_load
                    heat_coupling_node.node_type = 2 # junction node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                    if heat_load == 'outflow':
                        heat_net.nodes[1].node_type = 8 # sink slack node
                    else:
                        heat_net.nodes[1].node_type = 11 # slack return temp. node
                    heat_net.nodes[1].p = ph1
                    heat_net.nodes[1].Tr = Tr1
                elif node_set == 22:
                    gas_net.nodes[2].node_type = 3 # ref. load node
                    gas_net.nodes[2].p = pg2
                    elec_coupling_node.node_type = 6 # PQV
                    elec_coupling_node.half_links[0].Q = Q1_load
                else:
                    raise ValueError('Enter valid node_set for topology 1, single EH')
            else: # topology 1, CHP
                if node_set == 1:
                    elec_net.nodes[0].node_type = 1 # gen
                    ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                    elec_coupling_node.node_type = 5 # PQVdelta
                    elec_coupling_node.half_links[0].Q = Q1_load
                    elec_coupling_node.delta = delta1
                    heat_coupling_node.node_type = 5 # reference (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                elif node_set == 2:
                    gas_net.nodes[2].node_type = 3 # ref. load node
                    gas_net.nodes[2].p = pg2
                    elec_net.nodes[0].node_type = 1 # gen
                    ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                    elec_coupling_node.node_type = 4 # QVdelta
                    elec_coupling_node.half_links[0].Q = Q1_load
                    heat_coupling_node.node_type = 5 # reference (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                elif node_set == 3:
                    elec_coupling_node.node_type = 5 # PQVdelta
                    elec_coupling_node.half_links[0].Q = Q1_load
                    elec_coupling_node.delta = delta1
                    heat_coupling_node.node_type = 5 # reference (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                elif node_set == 4:
                    gas_net.nodes[2].node_type = 3 # ref. load node
                    gas_net.nodes[2].p = pg2
                    elec_coupling_node.node_type = 6 # PQV
                    elec_coupling_node.half_links[0].Q = Q1_load
                    heat_coupling_node.node_type = 5 # reference (junction) node
                    for hl in heat_coupling_node.get_half_links():
                        heat_coupling_node.remove_half_link(hl)
                        heat_net.remove_half_link(hl)
                    heat_coupling_node.half_links[:] = list()
                else:
                    raise ValueError('Enter valid node_set for topology 1, single CHP')
        else: # topology 1, two coupling nodes
            if node_set == 1:
                elec_net.nodes[0].node_type = 1 # gen
                ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                elec_coupling_node.node_type = 5 # PQVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                elec_coupling_node.delta = delta1
                heat_coupling_node.node_type = 7 # reference temperature (junction) node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node.half_links[:] = list()
            elif node_set == 2:
                gas_net.nodes[2].node_type = 3 # ref. load node
                gas_net.nodes[2].p = pg2
                elec_net.nodes[0].node_type = 1 # gen
                ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                elec_coupling_node.node_type = 4 # QVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                heat_coupling_node.node_type = 7 # reference temperature (junction) node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node.half_links[:] = list()
            elif node_set == 3:
                elec_coupling_node.node_type = 5 # PQVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                elec_coupling_node.delta = delta1
                heat_coupling_node.node_type = 7 # reference temperature (junction) node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node.half_links[:] = list()
            elif node_set == 4:
                gas_net.nodes[2].node_type = 3 # ref. load node
                gas_net.nodes[2].p = pg2
                elec_coupling_node.node_type = 6 # PQV
                elec_coupling_node.half_links[0].Q = Q1_load
                heat_coupling_node.node_type = 7 # reference temperature (junction) node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node.half_links[:] = list()
            else:
                raise ValueError('Enter valid node_set for topology 1, mutliple coupling nodes')
    elif topology == 2:
        if single_coupling: #not yet implemented
            pass
        else: # topology 2, two coupling nodes
            if node_set == 1:
                gas_net.nodes[0].node_type = 3 # ref. load node
                GasHalfLink('gn0_hl',gas_net.nodes[0],q=q0_load) # was slack, so didn't have a half link yet
                gas_net.nodes[1].node_type = 3 # ref. load node
                gas_net.nodes[1] = pg1
                elec_coupling_node.node_type = 5 # PQVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                elec_coupling_node.delta = delta1
                heat_coupling_node.node_type = 5 # reference (junction) node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node.half_links[:] = list()
                if heat_load == 'outflow':
                    heat_net.nodes[1].node_type = 3 # sink ref node
                    heat_net.nodes[1].half_links[0].bc_type = 3
                    heat_net.nodes[1].half_links[0].Tr_hl = Tr_hl1
                else:
                    heat_net.nodes[1].node_type = 13 # sink ref. temp. diff. node
                    heat_net.nodes[1].half_links[0].bc_type = 5
                    heat_net.nodes[1].half_links[0].dT = dT1
                heat_net.nodes[1].p = ph1
            elif node_set == 2 or node_set == 3:
                gas_net.nodes[1].node_type = 3 # ref. load node
                gas_net.nodes[1].p = pg1
                elec_coupling_node.node_type = 5 # PQVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                elec_coupling_node.delta = delta1
                heat_coupling_node.node_type = 5 # reference (junction) node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node.half_links[:] = list()
                if heat_load == 'outflow':
                    heat_net.nodes[1].node_type = 3 # sink ref node
                    heat_net.nodes[1].half_links[0].bc_type = 3
                    heat_net.nodes[1].half_links[0].Tr_hl = Tr_hl1
                else:
                    heat_net.nodes[1].node_type = 13 # sink ref. temp. diff. node
                    heat_net.nodes[1].half_links[0].bc_type = 5
                    heat_net.nodes[1].half_links[0].dT = dT1
                heat_net.nodes[1].p = ph1
            elif node_set == 4:
                gas_net.nodes[1].node_type = 3 # ref. load node
                gas_net.nodes[1].p = pg1
                elec_coupling_node.node_type = 5 # PQVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                elec_coupling_node.delta = delta1
                heat_coupling_node.node_type = 5 # reference (junction) node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node.half_links[:] = list()
            elif node_set == 5:
                gas_net.nodes[1].node_type = 3 # ref. load node
                gas_net.nodes[1].p = pg1
                elec_coupling_node.node_type = 6 # PQV
                elec_coupling_node.half_links[0].Q = Q1_load
                heat_coupling_node.node_type = 5 # reference (junction) node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node.half_links[:] = list()
                if heat_load == 'outflow':
                    heat_net.nodes[1].node_type = 3 # sink ref node
                    heat_net.nodes[1].half_links[0].bc_type = 3
                    heat_net.nodes[1].half_links[0].Tr_hl = Tr_hl1
                else:
                    heat_net.nodes[1].node_type = 13 # sink ref. temp. diff. node
                    heat_net.nodes[1].half_links[0].bc_type = 5
                    heat_net.nodes[1].half_links[0].dT = dT1
                heat_net.nodes[1].p = ph1
    elif topology == 3:
        if single_coupling: #not yet implemented
            pass
        else: # topology 3, two coupling nodes
            if node_set == 1:
                gas_net.nodes[1].node_type = 3 # ref. load node
                gas_net.nodes[1].p = pg1
                elec_coupling_node.node_type = 5 # PQVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                elec_coupling_node.delta = delta1
                heat_coupling_node0.node_type = 7 # reference temperature (junction) node
                for hl in heat_coupling_node0.get_half_links():
                    heat_coupling_node0.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node0.half_links[:] = list()
                if heat_load == 'outflow':
                    heat_coupling_node1.node_type = 16 # sink return temp. node
                    heat_coupling_node1.half_links[0].bc_type = 3
                    heat_coupling_node1.half_links[0].Tr_hl = Tr_hl1
                else:
                    heat_coupling_node1.node_type = 15 # sink return temp. temp. dif. node
                    heat_coupling_node1.half_links[0].bc_type = 5
                    heat_coupling_node1.half_links[0].dT = dT1
                heat_coupling_node1.Tr = Tr1
            elif node_set == 2 or node_set == 3:
                gas_net.nodes[1].node_type = 3 # ref. load node
                gas_net.nodes[1].p = pg1
                elec_coupling_node.node_type = 5 # PQVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                elec_coupling_node.delta = delta1
                heat_coupling_node0.node_type = 5 # reference temperature (junction) node
                for hl in heat_coupling_node0.get_half_links():
                    heat_coupling_node0.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node0.half_links[:] = list()
                if heat_load == 'outflow':
                    heat_coupling_node1.node_type = 16 # sink return temp. node
                    heat_coupling_node1.half_links[0].bc_type = 3
                    heat_coupling_node1.half_links[0].Tr_hl = Tr_hl1
                else:
                    heat_coupling_node1.node_type = 15 # sink return temp. temp. dif. node
                    heat_coupling_node1.half_links[0].bc_type = 5
                    heat_coupling_node1.half_links[0].dT = dT1
                heat_coupling_node1.Tr = Tr1
            elif node_set == 4:
                gas_net.nodes[1].node_type = 3 # ref. load node
                gas_net.nodes[1].p = pg1
                elec_coupling_node.node_type = 5 # PQVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                elec_coupling_node.delta = delta1
                heat_coupling_node0.node_type = 5 # reference (junction) node
                for hl in heat_coupling_node0.get_half_links():
                    heat_coupling_node0.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node0.half_links[:] = list()
            elif node_set == 5:
                elec_coupling_node.node_type = 6 # PQV
                elec_coupling_node.half_links[0].Q = Q1_load
                heat_coupling_node0.node_type = 6 # temperature (junction) node
                for hl in heat_coupling_node0.get_half_links():
                    heat_coupling_node0.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node0.half_links[:] = list()
                if heat_load == 'outflow':
                    heat_coupling_node1.node_type = 16 # sink temp. node
                    heat_coupling_node1.half_links[0].bc_type = 3
                    heat_coupling_node1.half_links[0].Tr_hl = Tr_hl1
                    heat_net.nodes[2].node_type = 3 # source/sink ref. node
                    heat_net.nodes[2].half_links[0].bc_type = 3
                    heat_net.nodes[2].half_links[0].Tr_hl = Tr_hl2
                else:
                    heat_coupling_node1.node_type = 15 # sink return temp. temp. dif. node
                    heat_coupling_node1.half_links[0].bc_type = 5
                    heat_coupling_node1.half_links[0].dT = dT1
                    heat_net.nodes[2].node_type = 13 # source/sink ref. temp. dif. node
                    heat_net.nodes[2].bc_type = 5
                    heat_net.nodes[2].dT = dT2
                heat_net.nodes[2].p = ph2
                heat_coupling_node1.Tr = Tr1
            elif node_set == 6:
                gas_net.nodes[1].node_type = 3 # ref. load node
                gas_net.nodes[1].p = pg1
                elec_coupling_node.node_type = 6 # PQV
                elec_coupling_node.half_links[0].Q = Q1_load
                heat_coupling_node0.node_type = 5 # reference (junction) node
                for hl in heat_coupling_node0.get_half_links():
                    heat_coupling_node0.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node0.half_links[:] = list()
                if heat_load == 'outflow':
                    heat_coupling_node1.node_type = 3 # source/sink ref. node
                    heat_coupling_node1.half_links[0].bc_type = 3
                    heat_coupling_node1.half_links[0].Tr_hl = Tr_hl1
                else:
                    heat_coupling_node1.node_type = 13 # source/sink ref. temp. dif. node
                    heat_coupling_node1.half_links[0].bc_type = 5
                    heat_coupling_node1.half_links[0].dT = dT1
                heat_coupling_node1.p = ph1
            elif node_set == 7:
                gas_net.nodes[1].node_type = 3 # ref. load node
                gas_net.nodes[1].p = pg1
                elec_coupling_node.node_type = 6 # PQV
                elec_coupling_node.half_links[0].Q = Q1_load
                heat_coupling_node0.node_type = 6 # temperature (junction) node
                for hl in heat_coupling_node0.get_half_links():
                    heat_coupling_node0.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node0.half_links[:] = list()
                if heat_load == 'outflow':
                    heat_coupling_node1.node_type = 3 # source/sink ref. node
                    heat_coupling_node1.half_links[0].bc_type = 3
                    heat_coupling_node1.half_links[0].Tr_hl = Tr_hl1
                else:
                    heat_coupling_node1.node_type = 13 # source/sink ref. temp. dif. node
                    heat_coupling_node1.half_links[0].bc_type = 5
                    heat_coupling_node1.half_links[0].dT = dT1
                heat_coupling_node1.p = ph1
            elif node_set == 8: # same as the scenario file
                elec_net.nodes[0].node_type = 1 # gen
                ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                elec_coupling_node.node_type = 5 # PQVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                elec_coupling_node.delta = delta1
                heat_coupling_node0.node_type = 5 # reference (junction) node
                for hl in heat_coupling_node0.get_half_links():
                    heat_coupling_node0.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node0.half_links[:] = list()
                if heat_load == 'outflow':
                    heat_coupling_node1.node_type = 3 # source/sink ref. node
                    heat_coupling_node1.half_links[0].bc_type = 3
                    heat_coupling_node1.half_links[0].Tr_hl = Tr_hl1
                else:
                    heat_coupling_node1.node_type = 13 # source/sink ref. temp. dif. node
                    heat_coupling_node1.half_links[0].bc_type = 5
                    heat_coupling_node1.half_links[0].dT = dT1
                heat_coupling_node1.p = ph1
    elif topology == 4:
        if single_coupling: #not yet implemented
            pass
        else: #two coupling nodes
            if node_set == 1:
                gas_net.nodes[1].node_type = 3 # ref. load node
                gas_net.nodes[1].p = pg1
                elec_coupling_node.node_type = 5 # PQVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                elec_coupling_node.delta = delta1
                heat_coupling_node0.node_type = 7 # reference temperature (junction) node
                for hl in heat_coupling_node0.get_half_links():
                    heat_coupling_node0.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node0.half_links[:] = list()
                if heat_load == 'outflow':
                    heat_coupling_node1.node_type = 16 # sink temp. node
                    heat_coupling_node1.Tr = Tr1
                else:
                    heat_coupling_node1.node_type = 15 # sink return temp. temp. dif. node
                    heat_coupling_node1.Tr = Tr1
            elif node_set == 2:
                elec_net.nodes[0].node_type = 1 # gen
                ElectricalHalfLink('en0_hl',elec_net.nodes[0],P=P0_load) # was slack, so didn't have a half link
                elec_coupling_node.node_type = 5 # PQVdelta
                elec_coupling_node.half_links[0].Q = Q1_load
                elec_coupling_node.delta = delta1
                heat_coupling_node0.node_type = 5 # reference (junction) node
                for hl in heat_coupling_node0.get_half_links():
                    heat_coupling_node0.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
                heat_coupling_node0.half_links[:] = list()
                if heat_load == 'outflow':
                    heat_coupling_node1.node_type = 3 # source/sink ref. node
                    heat_coupling_node1.half_links[0].bc_type = 3
                    heat_coupling_node1.half_links[0].Tr_hl = Tr_hl1
                else:
                    heat_coupling_node1.node_type = 13 # source/sink ref. temp. dif. node
                    heat_coupling_node1.half_links[0].bc_type = 5
                    heat_coupling_node1.half_links[0].dT = dT1
                heat_coupling_node1.p = ph1

    if topology == 1 and single_coupling:
        gas_net.add_link(glc)
        elec_net.add_link(elc)
        heat_net.add_link(hlc)
    elif topology == 2 and single_coupling:
        pass
    elif topology == 3 and single_coupling:
        pass
    elif topology == 4 and single_coupling:
        pass
    elif topology == 1 or topology == 2 or topology == 3 or topology == 4:
        gas_net.add_link(glc0)
        gas_net.add_link(glc1)
        elec_net.add_link(elc)
        heat_net.add_link(hlc0)
        heat_net.add_link(hlc1)

    het_net = HeterogeneousNetwork('3 nodes mes')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)

    if topology == 1 and single_coupling:
        het_net.add_node(cn)
    elif topology == 2 and single_coupling:
        het_net.add_node(cn)
    elif topology == 3 and single_coupling:
        het_net.add_node(cn)
    elif topology == 4 and single_coupling:
        het_net.add_node(cn)
    elif topology == 1 or topology == 2 or topology == 3 or topology == 4:
        het_net.add_node(cn0)
        het_net.add_node(cn1)

    return gas_net,elec_net,heat_net,het_net

def initialize_network(gas_net,elec_net,heat_net,het_net,pg1=29*mbar,pg2=28*mbar,q=.05,V_init=10/np.sqrt(3)*kV,ph0=4*bar,ph1=6*bar,ph2=1*bar,m=6,Ts0=100.,Ts1=99.,Ts2=98.,Tr=50.,Toc0=105,Toc1=95,Pc=1.5*MW,Qc=MW,phic=2*MW,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},heat_load='outflow',scale_var=None,scale_var_params=None):
    """Initialize the gas network, consisting of 3 demand/source nodes, and one extra node due to an compressor.

    Parameters
    ----------
    gas_net : GasNetwork
        The gas network to be initialized
    formulation : str, optional
        Formulation of the non-linear system of equations used to solve the network.

    Returns
    -------
    x0 : np array
        initial guess
    """
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)

    # gas part
    q_init = list()
    pg_init = list()
    for ind_el,el in enumerate(xg_entries):
        if isinstance(el,GasNode):
            if el.name == 'gn1':
                pg_init.append(pg1)
            elif el.name == 'gn2':
                pg_init.append(pg2)
        elif isinstance(el,GasLink):
            if formulation == 'nodal':
                if link.link_eq_form == 'dp_of_q':
                    warnings.warn("The link {} uses press. drop as function of link flow (fb instead of fa), but formulation is 'nodal'. The link equation for this link is changed to fa!!".format(link.name))
                    link.set_type(link.link_type,link.link_params,link_eq_form='q_of_dp')
            q_init.append(q)
    xg_init = np.array(q_init+pg_init)

    # electrical part
    delta_init = np.zeros(len(unknown_delta_nodes))
    V_init = V_init*np.ones(len(unknown_V_nodes))
    xe_init = np.concatenate((delta_init,V_init))

    # heat part
    m_init = np.array([m,m,m/6])
    m_hl_init = np.zeros(len(unknown_m_halflinks))
    for ind_hl,hl in enumerate(unknown_m_halflinks):
        if hl.source:
            m_hl_init[ind_hl] = -m
        else:
            m_hl_init[ind_hl] = m
    ph_init = np.zeros(len(unknown_p_nodes))
    for ind_n,node in enumerate(unknown_p_nodes):
        if node.name == 'hn0':
            ph_init[ind_n] = ph0
        elif node.name == 'hn1':
            ph_init[ind_n] = ph1
        elif node.name == 'hn2':
            ph_init[ind_n] = ph2
    Ts_init = np.zeros(len(unknown_Ts_nodes))
    for ind_n,node in enumerate(unknown_Ts_nodes):
        if node.name == 'hn0':
            Ts_init[ind_n] = Ts0
        elif node.name == 'hn1':
            Ts_init[ind_n] = Ts1
        elif node.name == 'hn2':
            Ts_init[ind_n] = Ts2
    Tr_init = Tr*np.ones(len(unknown_Tr_nodes))
    Ts_hl_init = Ts0*np.ones(len(unknown_Ts_halflinks))
    Tr_hl_init = Tr*np.ones(len(unknown_Tr_halflinks))
    xh_init = np.concatenate((m_init,m_hl_init,ph_init,Ts_init,Tr_init,Ts_hl_init,Tr_hl_init))

    # coupling
    qc_init = q/len(unknown_qc_links)*np.ones(len(unknown_qc_links))
    Pc_init = Pc*np.ones(len(unknown_Pc_links))
    Qc_init = Qc*np.zeros(len(unknown_Qc_links))
    Sc_init = np.concatenate((Pc_init,Qc_init))
    mc_init = m*np.ones(len(unknown_mc_links))
    if unknown_dphi_links:
        phic_init = phic/len(unknown_dphi_links)*np.ones(len(unknown_dphi_links))
    else:
        phic_init = np.array([])
    Toc_init = np.zeros(len(unknown_Ts_links))
    for ind_l,l in enumerate(unknown_Ts_links):
        if l.start_node.name == 'cn0':
            Toc_init[ind_l] = Toc0
        elif l.start_node.name == 'cn1':
            Toc_init[ind_l] = Toc1
        else:
            Toc_init[ind_l] = Ts0
    xc_init = np.concatenate((qc_init,Sc_init,mc_init,phic_init,Toc_init))

    x_init = np.concatenate((xg_init,xe_init,xh_init,xc_init))
    #print('x init = \n{}'.format(x_init))
    #print('For creating x_init: xg: {}, xg entries: {}, xe: {}, xe entries: {}, xh: {}, xh entries: {}, xc: {}, xc entries: {}, x: {}, x entries: {}'.format(len(xg_init),len(xg_entries),len(xe_init),len(xe_entries),len(xh_init),len(xh_entries),len(xc_init),len(xc_entries),len(x_init),len(x_entries)))
    #print('unknown_m_links: {}, unknown_m_halflinks: {}, unknown_p_nodes: {}, unknown_Ts_nodes: {}, unknown_Tr_nodes: {}, unknown_Ts_halflinks: {}, unknown_Tr_halflinks: {}'.format(len(unknown_m_links), len(unknown_m_halflinks), len(unknown_p_nodes), len(unknown_Ts_nodes), len(unknown_Tr_nodes), len(unknown_Ts_halflinks), len(unknown_Tr_halflinks)))
    het_net.initialize()

    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def run_load_flow(path_to_data,hydr_eq_gas='fa',heat_load='outflow',topology=1,node_set=1,single_coupling=False, EH=True,pg1=29*mbar,pg2=28*mbar,q=.05,V_init=10/np.sqrt(3)*kV,ph1=6*bar,ph0=4*bar,ph2=18*bar,m=6,Ts0=100.,Ts1=95.,Ts2=90.,Tr=50.,Toc0=100,Toc1=100.,Pc=1.5*MW,Qc=MW,phic=2*MW,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},pgbase=50*mbar,qbase=.05,Vbase=10/np.sqrt(3)*kV,Sbase=MW,phbase=10.*bar,mbase=1.,Tbase=100.,phibase=MW,Egbase=MW,tol=1e-6,max_iter=50,plot_top=False):
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
    gas_net,elec_net,heat_net,het_net = create_network(path_to_data,hydr_eq_gas=hydr_eq_gas,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling, EH=EH)
    # initialize
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net,pg1=pg1,pg2=pg2,q=q,V_init=V_init,ph0=ph0,ph1=ph1,ph2=ph2,m=m,Ts0=Ts0,Ts1=Ts1,Ts2=Ts2,Tr=Tr,Toc0=Toc0,Toc1=Toc1,Pc=Pc,Qc=Qc,phic=phic,formulation=formulation,heat_load=heat_load)

    # solve network
    #from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
    #nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
    #F0 = nlsys.F(x0)
    #J0 = nlsys.J(x0)
    #D_x = nlsys.Dx()
    #D_F = nlsys.DF()
    #D_x_inv = sps.diags(1/D_x.data[0])
    #print('x0: {}, Dx: {}, F0: {}, DF: {}'.format(len(x0),D_x.shape,len(F0),D_F.shape))
    #J_scaled = D_F.dot(J0.dot(D_x_inv))
    #print('|J0|={}, |D_F J0 D_x_inv|={}'.format(np.linalg.det(J0.todense()),np.linalg.det(J_scaled.todense())))
    #print('Dx = {}'.format(D_x))
    #print('DF = {}'.format(D_F))
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})

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
    F_entries, Fg_entries, Fe_entries, known_P_nodes, known_Q_nodes, Fh_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_To_halflinks, Fc_entries, F_fc_nodes, F_fc_amount, F_phi_nodes, F_dT_nodes = het_net.get_F_entries(formulation=formulation)
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
    F_entries, Fg_entries, Fe_entries, known_P_nodes, known_Q_nodes, Fh_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_To_halflinks, Fc_entries, F_fc_nodes, F_fc_amount, F_phi_nodes, F_dT_nodes = het_net.get_F_entries(formulation=formulation)
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
def example_mes_bp_top1_EH_scaled():
    """Test the solution of the mes with topology 4 against the scenario data (using the same node set), but with scaling"""
    # Given
    topology = 1
    node_set = 1 # same one as used to create the scenario data
    single_coupling = True
    EH = True
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'delta'
    hydr_eq = 'fb'
    # scaling
    qbase = .05
    pgbase = 50*mbar
    Tbase = 100.
    phibase = MW
    mbase = 1
    phbase = 10*bar
    Sbase = MW
    Vbase = 10/np.sqrt(3)*kV
    deltabase = 1.
    Egbase = phibase

    # When
    path_to_data = './examples/network_data/N_BP'
    # run load flow
    gas_net,elec_net,heat_net,het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(path_to_data,hydr_eq_gas=hydr_eq,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH,formulation=formulation,pgbase=pgbase,qbase=qbase,Vbase=Vbase,Sbase=Sbase,phbase=phbase,mbase=mbase,Tbase=Tbase,phibase=phibase,Egbase=Egbase,tol=1e-6,max_iter=50,plot_top=False)

    # Then
    x_sol_expected_SI = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation) #S.I., but x_sol is also in S.I, despite using scaling
    print('x ({} entries) = \n{}'.format(len(x_sol),x_sol))
    print('x exp ({} entries) = \n{}'.format(len(x_sol_expected_SI),x_sol_expected_SI))
    assert np.allclose(x_sol,x_sol_expected_SI)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_bp_top1_EH_unscaled():
    """Test the solution of the mes with topology 4 against the scenario data (using the same node set), but with scaling"""
    # Given
    topology = 1
    node_set = 1 # same one as used to create the scenario data
    single_coupling = True
    EH = True
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'delta'
    hydr_eq = 'fb'
    # scaling
    qbase = pgbase = Tbase = phibase = mbase = phbase = Sbase = Vbase = deltabase = Egbase = 1

    # When
    path_to_data = './examples/network_data/N_BP'
    # run load flow
    gas_net,elec_net,heat_net,het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(path_to_data,hydr_eq_gas=hydr_eq,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH,formulation=formulation,pgbase=pgbase,qbase=qbase,Vbase=Vbase,Sbase=Sbase,phbase=phbase,mbase=mbase,Tbase=Tbase,phibase=phibase,Egbase=Egbase,tol=1e-6,max_iter=50,plot_top=False)

    # Then
    x_sol_expected_SI = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation) #S.I., but x_sol is also in S.I, despite using scaling
    assert np.allclose(x_sol,x_sol_expected_SI)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_bp_top4_scaled():
    """Test the solution of the mes with topology 4 against the scenario data (using the same node set), but with scaling"""
    # Given
    topology = 4
    node_set = 2 # same one as used to create the scenario data
    single_coupling = False
    EH = False
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'delta'
    hydr_eq = 'fb'
    # scaling
    qbase = .05
    pgbase = 50*mbar
    Tbase = 100.
    phibase = MW
    mbase = 1
    phbase = 10*bar
    Sbase = MW
    Vbase = 10/np.sqrt(3)*kV
    deltabase = 1.
    Egbase = phibase

    # When
    path_to_data = './examples/network_data/N_BP'
    # run load flow
    gas_net,elec_net,heat_net,het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(path_to_data,hydr_eq_gas=hydr_eq,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH,formulation=formulation,pgbase=pgbase,qbase=qbase,Vbase=Vbase,Sbase=Sbase,phbase=phbase,mbase=mbase,Tbase=Tbase,phibase=phibase,Egbase=Egbase,tol=1e-6,max_iter=50,plot_top=False)

    # Then
    x_sol_expected_SI = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation) #S.I., but x_sol is also in S.I, despite using scaling
    assert np.allclose(x_sol,x_sol_expected_SI)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_bp_top4_unscaled():
    """Test the solution of the mes with topology 4 against the scenario data (using the same node set), no scaling"""
    # Given
    topology = 4
    node_set = 2 # same one as used to create the scenario data
    single_coupling = False
    EH = False
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'delta'
    hydr_eq = 'fb'
    # scaling
    qbase = pgbase = Tbase = phibase = mbase = phbase = Sbase = Vbase = deltabase = Egbase = 1

    # When
    path_to_data = './examples/network_data/N_BP'
    # run load flow
    gas_net,elec_net,heat_net,het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(path_to_data,hydr_eq_gas=hydr_eq,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH,formulation=formulation,pgbase=pgbase,qbase=qbase,Vbase=Vbase,Sbase=Sbase,phbase=phbase,mbase=mbase,Tbase=Tbase,phibase=phibase,Egbase=Egbase,tol=1e-6,max_iter=50,plot_top=False)

    # Then
    x_sol_expected_SI = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation) #S.I., but x_sol is also in S.I, despite using scaling
    assert np.allclose(x_sol,x_sol_expected_SI)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_bp_top4_scaled():
    """Test the solution of the mes with topology 4 against the scenario data (using the same node set), but with scaling"""
    # Given
    topology = 4
    node_set = 2 # same one as used to create the scenario data
    single_coupling = False
    EH = False
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'delta'
    hydr_eq = 'fb'
    # scaling
    qbase = .05
    pgbase = 50*mbar
    Tbase = 100.
    phibase = MW
    mbase = 1
    phbase = 10*bar
    Sbase = MW
    Vbase = 10/np.sqrt(3)*kV
    deltabase = 1.
    Egbase = phibase

    # When
    path_to_data = './examples/network_data/N_BP'
    # run load flow
    gas_net,elec_net,heat_net,het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(path_to_data,hydr_eq_gas=hydr_eq,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH,formulation=formulation,pgbase=pgbase,qbase=qbase,Vbase=Vbase,Sbase=Sbase,phbase=phbase,mbase=mbase,Tbase=Tbase,phibase=phibase,Egbase=Egbase,tol=1e-6,max_iter=50,plot_top=False)

    # Then
    x_sol_expected_SI = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation) #S.I., but x_sol is also in S.I, despite using scaling
    assert np.allclose(x_sol,x_sol_expected_SI)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_bp_top1_2c_q_known():
    """Test the solution of the mes with topology 1, two coupling nodes, against the scenario data. Using a similar node set, but with one of the coupling q's known"""
    # Given
    topology = 1
    node_set = 1 # same one as used to create the scenario data
    single_coupling = False
    EH = False
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'outflow'
    hydr_eq = 'fb'
    # scenario data
    path_to_data = './examples/network_data/N_BP'
    mes_net_scen,_,_,_,_,_,_,_,_,q_mes_scen,pg_mes_scen,q_hl_mes_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen,P_edge_mes_scen,Q_edge_mes_scen,m_mes_scen,m_hl_mes_scen,ph_mes_scen,Ts_mes_scen,Tr_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_hl_mes_scen = read_scen_data(path_to_data,topology=topology,single_coupling=single_coupling)
    qc_GB = q_mes_scen[3]
    # create network
    gas_net,elec_net,heat_net,het_net = create_network(path_to_data,hydr_eq_gas=hydr_eq,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling, EH=EH)
    for link in gas_net.get_links(link_types=['dummy']):
        if link.end_node.unit_type == 'gh_gas_boiler':
            link.bc_type = 1
            link.q = qc_GB
    heat_net.nodes[0].node_type = 5 # ref. (junction) node

    # When
    # initialize
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net,formulation=formulation)
    # solve
    tol = 1e-6
    max_iter = 20
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)

    # Then
    x_sol_expected  = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation)
    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_bp_top1_1c_P_known():
    """Test the solution of the mes with topology 1, one coupling node (CHP), against the scenario data. Using a similar node set, but with the coupling P known"""
    # Given
    topology = 1
    node_set = 1
    single_coupling = True
    EH = False
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'outflow'
    hydr_eq = 'fb'
    # scenario data
    path_to_data = './examples/network_data/N_BP'
    mes_net_scen,_,_,_,_,_,_,_,_,q_mes_scen,pg_mes_scen,q_hl_mes_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen,P_edge_mes_scen,Q_edge_mes_scen,m_mes_scen,m_hl_mes_scen,ph_mes_scen,Ts_mes_scen,Tr_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_hl_mes_scen = read_scen_data(path_to_data,topology=topology,single_coupling=single_coupling)
    Pc = P_edge_mes_scen[3]
    # create network
    gas_net,elec_net,heat_net,het_net = create_network(path_to_data,hydr_eq_gas=hydr_eq,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling, EH=EH)
    for link in elec_net.get_links(link_types=['dummy']):
        if link.start_node.unit_type == 'geh_CHP':
            link.bc_type = 1
            link.Pstart = Pc
            link.Pend = -Pc
    en0 = elec_net.nodes[0]
    en0.node_type = 0 # slack
    for hl in en0.get_half_links(): # was a generator node, so remove half link
        en0.remove_half_link(hl)
    en0.delta = delta_mes_scen[0] # was a generator node, so set delta
    elec_net.nodes[1].node_type = 6 # PQV

    # When
    # initialize
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net,formulation=formulation)
    het_net.reset_network(x0,formulation=formulation)
    # solve
    tol = 1e-6
    max_iter = 20
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)

    # Then
    x_sol_expected  = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation)
    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_bp_top1_1c_PQ_known():
    """Test the solution of the mes with topology 1, one coupling node (CHP), against the scenario data. Using a similar node set, but with the coupling P known"""
    # Given
    topology = 1
    node_set = 1
    single_coupling = True
    EH = False
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'outflow'
    hydr_eq = 'fb'
    # scenario data
    path_to_data = './examples/network_data/N_BP'
    mes_net_scen,_,_,_,_,_,_,_,_,q_mes_scen,pg_mes_scen,q_hl_mes_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen,P_edge_mes_scen,Q_edge_mes_scen,m_mes_scen,m_hl_mes_scen,ph_mes_scen,Ts_mes_scen,Tr_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_hl_mes_scen = read_scen_data(path_to_data,topology=topology,single_coupling=single_coupling)
    Pc = P_edge_mes_scen[3]
    Qc = Q_edge_mes_scen[3]
    # create network
    gas_net,elec_net,heat_net,het_net = create_network(path_to_data,hydr_eq_gas=hydr_eq,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling, EH=EH)
    for link in elec_net.get_links(link_types=['dummy']):
        if link.start_node.unit_type == 'geh_CHP':
            link.bc_type = 3
            link.Pstart = Pc
            link.Pend = -Pc
            link.Qstart = Qc
            link.Qend = -Qc
    en0 = elec_net.nodes[0]
    en0.node_type = 0 # slack
    for hl in en0.get_half_links(): # was a generator node, so remove half link
        en0.remove_half_link(hl)
    en0.delta = delta_mes_scen[0] # was a generator node, so set delta
    elec_net.nodes[1].node_type = 2 # load

    # When
    # initialize
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net,formulation=formulation)
    het_net.reset_network(x0,formulation=formulation)
    # solve
    tol = 1e-6
    max_iter = 20
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)

    # Then
    x_sol_expected  = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation)
    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_bp_top1_1c_CHP_part_load():
    """Test the solution of the mes with topology 1, one coupling node (CHP), against the scenario data. A model for the CHP taking part load effect into account is used. This is just to check if NR convergences, hence the parameters of the CHP are adjusted to match a CHP without part load effect."""
    # Given
    topology = 1
    node_set = 1
    single_coupling = True
    EH = False
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'outflow'
    hydr_eq = 'fb'
    # scenario data
    path_to_data = './examples/network_data/N_BP'
    mes_net_scen,_,_,_,_,_,_,_,_,q_mes_scen,pg_mes_scen,q_hl_mes_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen,P_edge_mes_scen,Q_edge_mes_scen,m_mes_scen,m_hl_mes_scen,ph_mes_scen,Ts_mes_scen,Tr_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_hl_mes_scen = read_scen_data(path_to_data,topology=topology,single_coupling=single_coupling)
    coupling_nodes = [node for node in mes_net_scen.get_nodes() if isinstance(node,HeterogeneousNode)] # should only be one
    GHV = coupling_nodes[0].unit_params.get('GHV')
    # create network
    gas_net,elec_net,heat_net,het_net = create_network(path_to_data,hydr_eq_gas=hydr_eq,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling, EH=EH)
    water = heat_net.links[0].link_params.get('carrier')
    cn = [node for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode)][0] # should only be one
    a_CHP = .91 #An adjusted value is used, otherwise this CHP does not match the one used in the scenario. Actual value is .358
    b_CHP = -MW*.0349
    d_CHP = MW*3.23
    eta_tot = .87
    phimin = 1*MW
    p_to_h = .38
    P_max_CHP = MW*6.1
    phimax = P_max_CHP/p_to_h # maximum (?) active power / power-to-heat-ratio
    L1 = .0001 # An adjusted value is used, otherwise this CHP does not match the one used in the scenario. This adjusted value effectively ensures that the part load effect is not taken into account. Actual value is .75
    L2 = .00001 # An adjusted value is used, otherwise this CHP does not match the one used in the scenario. This adjusted value effectively ensures that the part load effect is not taken into account. Actual value is.5
    r1 = .1304
    r2 = .1184
    eta_CHP = cn.unit_params.get('eta')#np.array([eta_tot,eta_tot]) # take equal to original efficiency
    unit_type_CHP = 'geh_CHP_part_load'
    unit_params_CHP = {'eta':eta_CHP,'a':a_CHP,'b':b_CHP,'d':d_CHP,'L1':L1,'L2':L2,'r1':r1,'r2':r2,'phimin':phimin,'phimax':phimax,'GHV':GHV}
    cn.set_type(unit_type_CHP,unit_params_CHP)
    cn.node_type = 0
    heat_net.links[3].bc_type = 0 # both To and dphi unknown
    # To is unknown, so a Ts must be specified
    heat_net.nodes[0].node_type = 7 #ref. temp. (junction) node (was ref. junction node)
    heat_net.nodes[0].Ts = Ts_mes_scen[0]
    # one other BC should now be let go (and if phi and To are known, then P is determined. So the CHP doesn't count as slack for the electrical network anymore).
    elec_net.nodes[1].node_type = 6 #PQV (was PQVdelta)
    # there is now no reference delta in the network, so:
    en0 = elec_net.nodes[0]
    en0.node_type = 0 # slack
    for hl in en0.get_half_links(): # was a generator node, so remove half link
        en0.remove_half_link(hl)
    en0.delta = delta_mes_scen[0] # was a generator node, so set delta

    # When
    # initialize
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net,formulation=formulation)
    het_net.reset_network(x0,formulation=formulation)
    # solve
    tol = 1e-8
    max_iter = 20
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)

    # Then
    rel_tol = 1e-3 # A relative tolerance, since I'm using a different model for the CHP. So the solution will not be exact.
    x_sol_expected  = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation)
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def comp_conv_form(path_to_data):
    """Compare the convergence of NR between MES and single-carrier networks, using different formulation in the singl-carrier (part of the) network."""
    topology = 1
    single_coupling = True
    EH = True
    node_sets = [1]

    for node_set in node_sets:
        # plot convergence
        fig_conv_form_het, ax_conv_form = plt.subplots(2, 2, num='Convergence plot networks, node set {}, topology {}'.format(node_set,topology))
        ax_conv_form_het = ax_conv_form[0,0]
        ax_conv_form_gas = ax_conv_form[0,1]
        ax_conv_form_elec = ax_conv_form[1,0]
        ax_conv_form_heat = ax_conv_form[1,1]
        ax_conv_form_het.set_title('Heterogeneous network')
        ax_conv_form_gas.set_title('Gas network')
        ax_conv_form_elec.set_title('Electrical network')
        ax_conv_form_heat.set_title('Heat network')

        max_iters_used = 0
        colors = {'sep. coup.':'tab:blue','int. coup.':'tab:orange'}
        linestyles_gas = {'full fa':'--','full fb':'-','nodal fa':'-.'}
        markers_gas = {'full fa':'.','full fb':'*','nodal fa':'d'}
        markers_heat = {'standard outflow':'.','standard delta':'*','half_link_flow outflow':'d','half_link_flow delta':'x'}

        # load flow
        hydr_eqs = ['fb','fa']
        forms_gas = ['full','nodal']
        forms_heat = ['half_link_flow','standard']
        heat_loads = ['outflow','delta']
        tol = 1e-6
        max_iter = 15

        # initial conditions
        pg1 = 29. * mbar
        pg2 = 28. * mbar
        ph2 = 1*bar
        Ts0 = 100.
        Ts1 = 99.
        Ts2 = 98.
        Toc0 = 110.
        Toc1 = 90.

        # scaling
        qbase = .05
        pgbase = 50*mbar
        Tbase = 100.
        phibase = MW
        mbase = 1
        phbase = 10*bar
        Sbase = MW
        Vbase = 10/np.sqrt(3)*kV
        deltabase = 1.
        Egbase = phibase
        # run load flow for SC
        for c_hl in [True,False]:
            for hydr_eq in hydr_eqs:
                for form_gas in forms_gas:
                    if not (form_gas == 'nodal' and hydr_eq == 'fb'): #those cannot be combined
                        print('\nHydraulic equation is {}, and separate couplings is {}, formulation = {}'.format(hydr_eq,c_hl,form_gas))
                        # load flow (only full is used, since the pipes use a implicit friction factor.
                        gas_net_single,x_sol_gas,iters_gas,err_vec_gas,p_sol_single,q_sol_single,q_inj_single,tol_gas = GasNet.run_load_flow(path_to_data,hydr_eq=hydr_eq,c_hl=c_hl,topology=topology,single_coupling=single_coupling,p1=pg1,p2=pg2,tol=tol,max_iter=max_iter,formulation=form_gas,pgbase=pgbase,qbase=qbase)
                        print('Solution:')
                        print('p = {} mbar'.format(p_sol_single/mbar))
                        print('q = {} kg/s'.format(q_sol_single))
                        print('q nodal inj = {} kg/s'.format(q_inj_single))
                        print('q hl = {} kg/s'.format([hl.q for node in gas_net_single.get_nodes() for hl in node.get_half_links()]))
                        label = '{} {}'.format(form_gas,hydr_eq)
                        if c_hl:
                            key = 'sep. coup.'
                        else:
                            key = 'int. coup.'
                        ax_conv_form_gas.semilogy(err_vec_gas,ls='-',color=colors.get(key),marker=markers_gas.get(label),label=key+', '+label)
                        max_iters_used = max(max_iters_used,iters_gas)
        for c_hl in [True,False]:
            print('separate couplings used: {}'.format(c_hl))
            elec_net_single,x_sol_elec,iters_elec,err_vec_elec,delta_sol_single,V_sol_single,S_inj_single,P_edge_single,Q_edge_single,tol_elec = ElecNet.run_load_flow(path_to_data,c_hl=c_hl,topology=topology,max_iter=max_iter)
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
        for c_hl in [True,False]:
            for form in forms_heat:
                for heat_load in heat_loads:
                    print('\nFormulation is {}, and heat load is {}, and separate couplings is {}'.format(form,heat_load,c_hl))
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                        heat_net_single,x_sol_heat,iters_heat,err_vec_heat,m_vec_single,p_vec_single,Ts_vec_single,Tr_vec_single,m_hl_vec_single,phi_hl_vec_single,Ts_hl_vec_single,Tr_hl_vec_single,tol_heat = HeatNet.run_load_flow(path_to_data,c_hl=c_hl,heat_load=heat_load,formulation=form,p2=ph2,Ts0=Ts0,Ts1=Ts1,Ts2=Ts2,Ts3=Toc0,Ts4=Toc1,Toc0=Toc0,Toc1=Toc1,topology=topology,single_coupling=single_coupling,max_iter=max_iter,phbase=phbase,mbase=mbase,Tbase=Tbase,phibase=phibase)
                    print('Solution:')
                    print('p heat = {} bar'.format(p_vec_single/bar))
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
        plot_top = True
        for hydr_eq in hydr_eqs:
                for form_gas in forms_gas:
                    if not (form_gas == 'nodal' and hydr_eq == 'fb'): #those cannot be combined
                        for form_heat in forms_heat:
                            for heat_load in heat_loads:
                                formulation={'gas':form_gas,'elec':'complex_power','heat':form_heat,'het':None}
                                print('\nHydraulic equation is {}, formulation gas is {}, formulation heat is {}, and heat load is {}'.format(hydr_eq,form_gas,form_heat,heat_load))
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                                    warnings.filterwarnings("ignore", "Only a",UserWarning)
                                    gas_net,elec_net,heat_net,het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(path_to_data,hydr_eq_gas=hydr_eq,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling, EH=EH,formulation=formulation,pg1=pg1,pg2=pg2,ph2=ph2,Ts0=Ts0,Ts1=Ts1,Ts2=Ts2,Toc0=Toc0,Toc1=Toc1,pgbase=pgbase,qbase=qbase,Vbase=Vbase,Sbase=Sbase,phbase=phbase,mbase=mbase,Tbase=Tbase,phibase=phibase,Egbase=Egbase,tol=tol,max_iter=max_iter,plot_top=plot_top)
                                key_heat = '{} {}'.format(form_heat,heat_load)
                                key_gas = '{} {}'.format(form_gas,hydr_eq)
                                ax_conv_form_het.semilogy(err_vec,ls=linestyles_gas.get(key_gas),color='tab:blue',marker=markers_heat.get(key_heat),label=key_gas+', '+key_heat)
                                plot_top = False
                                print('Solution after {} it. (final error = {:.4e}):'.format(iters,err_vec[-1]))
                                print('p = {} mbar'.format(p_g_vec/mbar))
                                print('q = {} kg/s'.format(q_vec))
                                print('q nodal inj = {} kg/s'.format(q_inj))
                                print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
                                print('delta = {}'.format(delta_vec))
                                print('|V| = {} V'.format(V_mag_vec))
                                print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
                                print('P edge = {} W'.format(P_edge))
                                print('Q edge = {} W'.format(Q_edge))
                                print('S nodal inj = {} W'.format(S_inj))
                                print('P hl = {} W'.format([hl.P for node in elec_net.get_nodes() for hl in node.get_half_links()]))
                                print('Q hl = {} W'.format([hl.Q for node in elec_net.get_nodes() for hl in node.get_half_links()]))
                                print('p heat = {} bar'.format(p_h_vec/bar))
                                print('m = {}'.format(m_vec))
                                print('Ts = {}'.format(Ts_vec))
                                print('Tr = {}'.format(Tr_vec))
                                print('m hl = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('Ts hl = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('Tr hl = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('Ts c = {} C'.format(Tsc_vec))
                                print('Tr c = {} C'.format(Trc_vec))
                                print('dphi c = {} MW'.format([phi/MW for phi in phic_vec]))
                                print('m c = {} kg/s'.format(mc_vec))

        # layout convergence plots
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

    #from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
    #nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
    #with warnings.catch_warnings():
        #warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        ## create Jacobian matrices
        #J = nlsys.J(x_sol)
        #D_x = nlsys.Dx()
        #D_F = nlsys.DF()
        #D_x_inv = sps.diags(1/D_x.data[0])
        #J_scaled = D_F.dot(J.dot(D_x_inv))
        #J_scaled_dense = np.matrix(np.nan*np.ones(J_scaled.shape))
        ## spy plot original
        #fig_J = nlsys.spy_plot_J(x_sol,title='Jacobian spy plot')
        #ax_J = plt.gca()
        #fig_J_map = nlsys.imshow_J(x_sol,title=r'Jacobian')

        ## spy plot of scaled J over original
        #ax_J.spy(J_scaled,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5)
        #nlsys.plot_J_overlay(ax_J)
        ## colormap of Jacobian
        #fig_J_scaled_map = plt.figure('Scaled Jacobian')

        #indices = J_scaled.indices
        #indptr = J_scaled.indptr
        #for row_ind in range(J_scaled.shape[0]):
            #for col_ind in indices[indptr[row_ind]:indptr[row_ind+1]]:
                #J_scaled_dense[row_ind,col_ind] = J_scaled[row_ind,col_ind]
        #plt.imshow(J_scaled_dense)
        #ax_J_scaled_map = plt.gca()
        #nlsys.plot_J_overlay(ax_J_scaled_map)
        #plt.colorbar()

def comp_conv_scaling(path_to_data,hydr_eq_gas='fa',heat_load='outflow',topology=1,node_set=1,single_coupling=False,EH=False,consistent_base=False,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None},return_networks=False):
    """Compare convergence of NR for different ways of scaling."""
    # base values
    if consistent_base: # gas and heat must use the same (hydraulic) base values
        pgbase = 1.
        qbase =.05
    else:
        pgbase = 50*mbar
        qbase =.05
    Sbase = 1*MW #[W]
    Vbase = 10/np.sqrt(3)*kV #[V]
    deltabase = 1. # has to be
    if consistent_base: # gas and heat must use the same (hydraulic) base values
        phbase = pgbase # Required for the p.u. implementation
        mbase = qbase # Required for the p.u. implementation
    else:
        phbase = 10.*bar
        mbase = 1.
    Tbase = 100.
    if consistent_base: # all coupling energies must have the same base values
        phibase = Sbase
        Egbase = Sbase
    else:
        phibase = 1*MW
        Egbase = 1*MW

    # create networks
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        # network with values specified in S.I.
        gas_net_SI,elec_net_SI,heat_net_SI,het_net_SI = create_network(path_to_data,hydr_eq_gas=hydr_eq_gas,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH)
        # network with values specified in p.u.
        gas_net_pu,elec_net_pu,heat_net_pu,het_net_pu = create_network(path_to_data,hydr_eq_gas=hydr_eq_gas,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH)
        # scale gas part parameters
    for link in gas_net_pu.get_links():
        if link.link_type == 'dummy':
            pass
        elif not link.link_type == 'pipe_low_pres_pole':
            raise ValueError('Cannot create network with values specified in p.u. Link type is wrong')
        else:
            C_SI = link.pipe_const() # pipe constant of 'low pres.' pipe in S.I.
            C_b = qbase/np.sqrt(pgbase)
            fric_fac = .0065 # since the pipe is assumed to have Pole's friction factor
            C_pu = (C_SI/np.sqrt(fric_fac))/C_b# pipe constant for a resistor, equivalent to the 'low pres.' pipe, in p.u.
            link.set_type('resistor',{'C':C_pu},link_eq_form=link.link_eq_form)
        link.q /= qbase
    for node in gas_net_pu.get_nodes():
        node.p /= pgbase
        for hl in node.get_half_links():
            hl.q /= qbase
    # scale electrical part parameters
    for link in elec_net_pu.get_links():
        if link.link_type == 'dummy':
            pass
        elif not link.link_type == 'short_line':
            raise ValueError('Cannot create network with values specified in p.u. Link type is wrong')
        else:
            Ybase = Sbase/(Vbase**2)
            b_SI = link.b
            g_SI = link.g
            b_pu = b_SI/Ybase
            g_pu = g_SI/Ybase
            link.set_type('short_line',{'b':b_pu,'g':g_pu})
        link.Pstart /= Sbase
        link.Qstart /= Sbase
        link.Pend /= Sbase
        link.Qend /= Sbase
    for node in elec_net_pu.get_nodes():
        node.V /= Vbase
        for hl in node.get_half_links():
            if not hl.link_type == 'flow':
                raise ValueError('Cannot create network with values specified in p.u. HalfLink type is wrong')
            else:
                hl.P /= Sbase
                hl.Q /= Sbase
    # scale heat part parameters
    Ta_SI = heat_net_pu.Ta
    Ta_pu = Ta_SI/Tbase
    heat_net_pu.Ta = Ta_pu
    water_pu = heat_net_pu.links[0].link_params.get('carrier') # still in S.I. units
    Cp_b = phibase/(Tbase*mbase)
    water_pu.Cp /= Cp_b
    water_pu.name = 'water p.u.'
    for link in heat_net_pu.get_links():
        if link.link_type == 'dummy':
            link.link_params['carrier'] = water_pu
            link.set_type('dummy',{'carrier':water_pu,'Ta':Ta_pu})
        elif not link.link_type == 'standard_pipe_low_pres_pole':
            raise ValueError('Cannot create network with values specified in p.u. Link type is wrong')
        else:
            C_SI = link.pipe_const() # pipe constant of 'low pres.' pipe in S.I.
            C_b = mbase/np.sqrt(phbase)
            fric_fac = .0065 # since the pipe is assumed to have Pole's friction factor
            C_pu = (C_SI/np.sqrt(fric_fac))/C_b # pipe constant for a resistor, equivalent to the 'low pres.' pipe, in p.u.
            U_SI = link.link_params.get('U')
            L_SI = link.link_params.get('L')
            D_SI = link.link_params.get('D')
            U_b = Cp_b*mbase
            U_pu = U_SI/U_b
            link.set_type('standard_resistor',{'L':L_SI,'U':U_pu,'D':D_SI,'C':C_pu,'carrier':water_pu,'Ta':Ta_pu})
        link.dphistart /= phibase
        link.m /= mbase
        link.Tsstart /= Tbase
        link.Trstart /= Tbase
        link.dTstart /= Tbase
        link.Tsend /= Tbase
        link.Trend /= Tbase
        link.dTend /= Tbase
    for node in heat_net_pu.get_nodes():
        node.p /= phbase
        node.Ts /= Tbase
        node.Tr /= Tbase
        for hl in node.get_half_links():
            if not 'heat_exchanger' in hl.link_type:
                raise ValueError('Cannot create network with values specified in p.u. HalfLink type is wrong')
            else:
                hl.set_type(hl.link_type,{'carrier':water_pu})
                hl.dphi /= phibase
                hl.m /= mbase
                hl.Ts /= Tbase
                hl.Tr /= Tbase
                hl.dT /= Tbase
    # scale electrical part parameters
    het_net_pu.Ta = Ta_pu
    GHV_SI = het_net_pu.nodes[-1].unit_params.get('GHV')
    GHV_b = Egbase/qbase
    GHV_pu = GHV_SI/GHV_b
    eta_ge_b = Sbase/Egbase
    eta_gh_b = phibase/Egbase
    for node in het_net_pu.get_nodes():
        if isinstance(node,HeterogeneousNode):
            if node.unit_type == 'EH':
                C_EH_SI = node.unit_params.get('C')
                C_EH_pu = np.array([[C_EH_SI[0,0]/eta_ge_b],[C_EH_SI[1,0]/eta_gh_b]])
                unit_params_pu={'C':C_EH_pu,'GHV':GHV_pu}
            elif node.unit_type == 'geh_CHP':
                eta_CHP_SI = node.unit_params.get('eta')
                eta_CHP_pu = np.array([eta_CHP_SI[0]/eta_ge_b, eta_CHP_SI[1]/eta_gh_b])
                unit_params_pu = {'eta':eta_CHP_pu,'GHV':GHV_pu}
            elif node.unit_type == 'gh_gas_boiler':
                eta_GB_SI = node.unit_params.get('eta')
                eta_GB_pu = eta_GB_SI/eta_gh_b
                unit_params_pu={'eta':eta_GB_pu,'GHV':GHV_pu}
            node.set_type(node.unit_type,unit_params_pu)

    # initial conditions
    pg1=29*mbar
    pg2=28*mbar
    q=.05
    V_init=10/np.sqrt(3)*kV
    ph0=4*bar
    ph1=6*bar
    ph2=1*bar
    m=6
    Ts0=100.
    Ts1=95.
    Ts2=90.
    Tr=50.
    Toc0=105
    Toc1=95
    Pc=1.5*MW
    Qc=MW
    phic=2*MW

    tol=1e-6
    max_iter=10

    # initialize with values specified in S.I.
    x0_SI = initialize_network(gas_net_SI,elec_net_SI,heat_net_SI,het_net_SI,pg1=pg1,pg2=pg2,q=q,V_init=V_init,ph0=ph0,ph1=ph1,ph2=ph2,m=m,Ts0=Ts0,Ts1=Ts1,Ts2=Ts2,Tr=Tr,Toc0=Toc0,Toc1=Toc1,Pc=Pc,Qc=Qc,phic=phic,formulation=formulation,heat_load=heat_load)

    # run load flow for network with values specified in S.I., without scaling
    # solve network
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        x_sol_SI,x_mat_SI,iters_SI,err_vec_SI,p_g_vec_SI,q_vec_SI,q_inj_SI,delta_vec_SI,V_mag_vec_SI,S_inj_SI,P_edge_SI,Q_edge_SI,m_vec_SI,p_h_vec_SI,Ts_vec_SI,Tr_vec_SI,m_hl_vec_SI,phi_hl_vec_SI,Ts_hl_vec_SI,Tr_hl_vec_SI,qc_vec_SI,Pc_vec_SI,Qc_vec_SI,mc_vec_SI,phic_vec_SI,Tsc_vec_SI,Trc_vec_SI = het_net_SI.solve_network(tol,max_iter,solver='NR',formulation=formulation,return_all_x=True)
    from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
    nlsys = NonLinearSystemHeterogeneous(het_net_SI,formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
    D_F = nlsys.DF()
    scaled_err_vec_SI = np.zeros(iters_SI+1)
    for ind in range(iters_SI+1):
        x_SI = x_mat_SI[ind,:]
        F_SI = nlsys.F(x_SI)
        scaled_err_vec_SI[ind] = np.linalg.norm(D_F.dot(F_SI))

    # run load flow for network with values specified in S.I., using matrix scaling
    het_net_SI.reset_network(x0_SI,formulation=formulation)
    #het_net_SI.update_full(x0_SI)
    # solve network
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        x_sol_scaled,iters_scaled,err_vec_scaled,p_g_vec_scaled,q_vec_scaled,q_inj_scaled,delta_vec_scaled,V_mag_vec_scaled,S_inj_scaled,P_edge_scaled,Q_edge_scaled,m_vec_scaled,p_h_vec_scaled,Ts_vec_scaled,Tr_vec_scaled,m_hl_vec_scaled,phi_hl_vec_scaled,Ts_hl_vec_scaled,Tr_hl_vec_scaled,qc_vec_scaled,Pc_vec_scaled,Qc_vec_scaled,mc_vec_scaled,phic_vec_scaled,Tsc_vec_scaled,Trc_vec_scaled = het_net_SI.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})

    # run load flow for network with values specified in S.I., using p.u. scaling
    scale_var = 'per_unit'
    scale_var_params = {'qbase':qbase,'pbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}
    # initialize
    het_net_SI.reset_network(x0_SI,formulation=formulation)
    #het_net_SI.update_full(x0_SI)
    x0_SI_pu = initialize_network(gas_net_SI,elec_net_SI,heat_net_SI,het_net_SI,pg1=pg1,pg2=pg2,q=q,V_init=V_init,ph0=ph0,ph1=ph1,ph2=ph2,m=m,Ts0=Ts0,Ts1=Ts1,Ts2=Ts2,Tr=Tr,Toc0=Toc0,Toc1=Toc1,Pc=Pc,Qc=Qc,phic=phic,formulation=formulation,heat_load=heat_load,scale_var=scale_var,scale_var_params=scale_var_params)
    # solve network
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        x_sol_SI_pu,iters_SI_pu,err_vec_SI_pu,p_g_vec_SI_pu,q_vec_SI_pu,q_inj_SI_pu,delta_vec_SI_pu,V_mag_vec_SI_pu,S_inj_SI_pu,P_edge_SI_pu,Q_edge_SI_pu,m_vec_SI_pu,p_h_vec_SI_pu,Ts_vec_SI_pu,Tr_vec_SI_pu,m_hl_vec_SI_pu,phi_hl_vec_SI_pu,Ts_hl_vec_SI_pu,Tr_hl_vec_SI_pu,qc_vec_SI_pu,Pc_vec_SI_pu,Qc_vec_SI_pu,mc_vec_SI_pu,phic_vec_SI_pu,Tsc_vec_SI_pu,Trc_vec_SI_pu = het_net_SI.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # run load flow for network with values specified in p.u., without scaling
    # initialize
    x0_pu = initialize_network(gas_net_pu,elec_net_pu,heat_net_pu,het_net_pu,pg1=pg1/pgbase,pg2=pg2/pgbase,q=q/qbase,V_init=V_init/Vbase,ph0=ph0/phbase,ph1=ph1/phbase,ph2=ph2/phbase,m=m/mbase,Ts0=Ts0/Tbase,Ts1=Ts1/Tbase,Ts2=Ts2/Tbase,Tr=Tr/Tbase,Toc0=Toc0/Tbase,Toc1=Toc1/Tbase,Pc=Pc/Sbase,Qc=Qc/Sbase,phic=phic/phibase,formulation=formulation,heat_load=heat_load)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        x_sol_pu,iters_pu,err_vec_pu,p_g_vec_pu,q_vec_pu,q_inj_pu,delta_vec_pu,V_mag_vec_pu,S_inj_pu,P_edge_pu,Q_edge_pu,m_vec_pu,p_h_vec_pu,Ts_vec_pu,Tr_vec_pu,m_hl_vec_pu,phi_hl_vec_pu,Ts_hl_vec_pu,Tr_hl_vec_pu,qc_vec_pu,Pc_vec_pu,Qc_vec_pu,mc_vec_pu,phic_vec_pu,Tsc_vec_pu,Trc_vec_pu = het_net_pu.solve_network(tol,max_iter,solver='NR',formulation=formulation)

    print('Scaled Errors. Par. in S.I., matrix scaling:\n{}'.format(scaled_err_vec_SI))
    print('Errors. Par. in S.I., unscaled:\n{}'.format(err_vec_SI))
    print('Errors. Par. in S.I., matrix scaling:\n{}'.format(err_vec_scaled))
    print('Errors. Par. in S.I., p.u. scaling:\n{}'.format(err_vec_SI_pu))
    print('Errors. Par. in p.u., unscaled:\n{}'.format(err_vec_pu))

    # make figure to plot convergence
    fig_conv_het = plt.figure('Convergence plot heterogeneous network, scaling. Top = {}, single coup = {}, EH = {}'.format(topology,single_coupling,EH))
    ax_conv_het = fig_conv_het.gca()
    max_iters_used = max([iters_scaled,iters_SI_pu,iters_pu])
    linestyles_gas = {'full fa':'--','full fb':'-','nodal fa':'-.'}
    markers_heat = {'standard outflow':'.','standard delta':'*','half_link_flow outflow':'d','half_link_flow delta':'x'}
    ls = linestyles_gas.get(formulation.get('gas')+' '+hydr_eq_gas)
    marker = markers_heat.get(formulation.get('heat')+' '+heat_load)
    ax_conv_het.semilogy(err_vec_scaled,ls=ls,color='tab:blue',marker=marker,label='matrix scaling')
    if phbase == pgbase and mbase == qbase and Sbase == phibase and Sbase == Egbase:
        label_pu = 'p.u. scaling'
    else:
        label_pu = 'p.u. scaling. WRONG!! (due to base values)'
    ax_conv_het.semilogy(err_vec_SI_pu,ls=ls,color='tab:orange',marker=marker,label=label_pu)
    ax_conv_het.semilogy(err_vec_pu,ls=ls,color='tab:red',marker=marker,label='specified in p.u.')
    ax_conv_het.set_xlabel(r'Iteration $k$')
    ax_conv_het.set_ylabel(r'Error ($||D_F F(x^k)||_2$ or $||F(x^k)||_2$)')
    ax_conv_het.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
    ax_conv_het.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_het.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_het.legend()
    xmin = 0
    xmax = max_iters_used
    xticks = range(xmin,xmax+1,2) # make sure the xticks are integers
    ax_conv_het.set_xlim(left=xmin,right=xmax+1)
    ax_conv_het.set_xticks(xticks)

    if return_networks:
        return het_net_SI,x_sol_scaled,iters_scaled,err_vec_scaled,x_sol_SI_pu,iters_SI_pu,err_vec_SI_pu,x_sol_pu,iters_pu,err_vec_pu,x_sol_SI,iters_SI,err_vec_SI,scaled_err_vec_SI
    else:
        return x_sol_scaled,iters_scaled,err_vec_scaled,x_sol_SI_pu,iters_SI_pu,err_vec_SI_pu,x_sol_pu,iters_pu,err_vec_pu,x_sol_SI,iters_SI,err_vec_SI,scaled_err_vec_SI

def jacobians(path_to_data,hydr_eq_gas='fa',heat_load='outflow',topology=1,node_set=1,single_coupling=False,EH=False,pg1=29*mbar,pg2=28*mbar,q=.05,V_init=10/np.sqrt(3)*kV,ph1=6*bar,ph0=4*bar,ph2=18*bar,m=6,Ts0=100.,Ts1=95.,Ts2=90.,Tr=50.,Toc0=100,Toc1=100.,Pc=1.5*MW,Qc=MW,phic=2*MW,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},pgbase=50*mbar,qbase=.05,deltabase=1,Vbase=10/np.sqrt(3)*kV,Sbase=MW,phbase=10.*bar,mbase=1.,Tbase=100.,phibase=MW,Egbase=MW,tol=1e-6,max_iter=50):
    """Plot Jacobian matrices, and eigenvalue spectra, for different indices / ordering"""

    # create network
    gas_net,elec_net,heat_net,het_net = create_network(path_to_data,hydr_eq_gas=hydr_eq_gas,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH)
    # initialize
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net,pg1=pg1,pg2=pg2,q=q,V_init=V_init,ph0=ph0,ph1=ph1,ph2=ph2,m=m,Ts0=Ts0,Ts1=Ts1,Ts2=Ts2,Tr=Tr,Toc0=Toc0,Toc1=Toc1,Pc=Pc,Qc=Qc,phic=phic,formulation=formulation,heat_load=heat_load)

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

    # solve system for the different reorderings (and using scaling)
    het_net.reset_network(x0,formulation=formulation)
    print('\nSolving original system')
    D_x_inv = sps.diags(1/D_x.data[0])
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

    fig = plt.figure('Convergence plot different ordering')
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

def comp_lin_solvers(path_to_data,hydr_eq_gas='fa',heat_load='outflow',topology=1,node_set=1,single_coupling=False,EH=False,):
    """Compare the different linear solvers"""
    # solver information
    max_iter = 10
    maxmatvecs = 5000
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    tol=1e-6
    lin_solvers = ['solve','gmres','bicgstab']
    err_NR = dict()

    # create convergence plots
    fig_conv = plt.figure('Convergence plot unscaled, top. {}, node set {}'.format(topology,node_set))
    ax_conv = fig_conv.gca()

    fig_conv_scaled = plt.figure('Convergence plot scaled, top. {}, node set {}'.format(topology,node_set))
    ax_conv_scaled = fig_conv_scaled.gca()

    fig_conv_scaled_gmres = plt.figure('Convergence plot GMRES scaled, top. {}, node set {}'.format(topology,node_set))
    ax_conv_scaled_gmres = fig_conv_scaled_gmres.gca()

    fig_conv_scaled_bicgstab = plt.figure('Convergence plot BiCGStab scaled, top. {}, node set {}'.format(topology,node_set))
    ax_conv_scaled_bicgstab = fig_conv_scaled_bicgstab.gca()

    # create network
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        gas_net,elec_net,heat_net,het_net = create_network(path_to_data,hydr_eq_gas=hydr_eq_gas,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling, EH=EH)

    # scaling
    qbase = .05
    pgbase = 50*mbar
    Tbase = 100.
    phibase = 1*MW
    mbase = 1
    phbase = 10*bar
    Sbase = 1*MW
    Vbase = 10/np.sqrt(3)*kV
    deltabase = 1.
    Egbase = 1*MW

    # initial conditions
    pg1=29*mbar
    pg2=28*mbar
    q=.05
    V_init=10/np.sqrt(3)*kV
    ph0=4*bar
    ph1=6*bar
    ph2=1*bar
    m=6
    Ts0=100.
    Ts1=95.
    Ts2=90.
    Tr=50.
    Toc0=105
    Toc1=95
    Pc=1.5*MW
    Qc=MW
    phic=2*MW

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        x_sol_expected_SI = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation) #S.I.
    # load flow
    # initalize network, unscaled
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net,pg1=pg1,pg2=pg2,q=q,V_init=V_init,ph0=ph0,ph1=ph1,ph2=ph2,m=m,Ts0=Ts0,Ts1=Ts1,Ts2=Ts2,Tr=Tr,Toc0=Toc0,Toc1=Toc1,Pc=Pc,Qc=Qc,phic=phic,formulation=formulation,heat_load=heat_load)
    # unscaled
    max_iters_used = 0
    print('\nUnscaled NR:')
    for lin_solver in lin_solvers:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
            het_net.reset_network(x0,formulation=formulation)
            x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,lin_solver=lin_solver,max_iter_lin=maxmatvecs)
            print('Voor {}, x dichtbij x_sol: {}'.format(lin_solver,np.allclose(x_sol_expected_SI,x_sol)))

            # plot convergence
            max_iters_used = max(max_iters_used,iters)
            ax_conv.semilogy(err_vec,'.-',label=lin_solver)
    ax_conv.semilogy([0,max_iters_used],[tol,tol],'k:',label='tol')
    ax_conv.set_xlabel("Iteration k")
    ax_conv.set_ylabel("Error $||D_F F(x^k)||_2$")
    ax_conv.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv.legend()

    # scaled
    print('\nScaled NR:')
    max_iters_used_scaled = 0
    for lin_solver in lin_solvers:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
            het_net.reset_network(x0,formulation=formulation)
            x_sol_scaled,iters_scaled,err_vec_scaled,p_g_vec_scaled,q_vec_scaled,q_inj_scaled,delta_vec_scaled,V_mag_vec_scaled,S_inj_scaled,P_edge_scaled,Q_edge_scaled,m_vec_scaled,p_h_vec_scaled,Ts_vec_scaled,Tr_vec_scaled,m_hl_vec_scaled,phi_hl_vec_scaled,Ts_hl_vec_scaled,Tr_hl_vec_scaled,qc_vec_scaled,Pc_vec_scaled,Qc_vec_scaled,mc_vec_scaled,phic_vec_scaled,Tsc_vec_scaled,Trc_vec_scaled = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase},lin_solver=lin_solver,max_iter_lin=maxmatvecs)
            print('Voor {}, geschaald, x dichtbij x_sol: {}'.format(lin_solver,np.allclose(x_sol_expected_SI,x_sol_scaled)))
            err_NR[lin_solver] = err_vec_scaled

            # plot convergence
            max_iters_used_scaled = max(max_iters_used_scaled,iters_scaled)
            ax_conv_scaled.semilogy(err_vec_scaled,'s-',label=lin_solver)
            if lin_solver == 'gmres':
                ax_conv_scaled_gmres.semilogy(err_vec_scaled,'s-',label=lin_solver+', scaled input')
            elif lin_solver == 'bicgstab':
                ax_conv_scaled_bicgstab.semilogy(err_vec_scaled,'s-',label=lin_solver+', scaled input')
            else:
                ax_conv_scaled_gmres.semilogy(err_vec_scaled,'s-',label=lin_solver+', scaled input')
                ax_conv_scaled_bicgstab.semilogy(err_vec_scaled,'s-',label=lin_solver+', scaled input')

    ax_conv_scaled.semilogy([0,max_iters_used_scaled],[tol,tol],'k:',label='tol')
    ax_conv_scaled.set_xlabel("Iteration k")
    ax_conv_scaled.set_ylabel("Error $||D_F F(x^k)||_2$")
    ax_conv_scaled.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_scaled.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_scaled.legend()

    # Use preconditioning instead of scaling with matrices
    from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
        F0 = nlsys.F(x0)
        J0 = nlsys.J(x0)

    D_F = nlsys.DF()
    D_F_inv = sps.diags(1/D_F.data[0])
    print('\nHandmatige NR gmres:')
    err_vec = list()
    het_net.reset_network(x0,formulation=formulation)
    error_gmres = np.linalg.norm(D_F.dot(F0))
    err_vec_gmres = list()
    err_vec_gmres.append(error_gmres)
    x_new = x0
    F_new = F0
    J_new = J0

    from scipy.sparse.linalg import LinearOperator
    D_F_inv = sps.diags(1/D_F.data[0])
    prec_list = ['I','D_F','D_F_inv','no prec']
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        MM_norm = [np.linalg.norm(np.eye(len(x0))), np.linalg.norm(D_F.todense()), np.linalg.norm(D_F_inv.todense()), 1]

        for ind_M,MM in enumerate([LinearOperator(shape=J0.shape, matvec=np.eye(len(x0)).dot, matmat = np.eye(len(x0)).dot, dtype=D_F.dtype),LinearOperator(shape=J0.shape, matvec=D_F.dot, matmat = D_F.dot, dtype=D_F.dtype), LinearOperator(shape=J0.shape, matvec= D_F_inv.dot, matmat = D_F_inv.dot, dtype=D_F.dtype),None]):
            iter_gmres = 0
            error_gmres = np.linalg.norm(D_F.dot(F0))
            err_vec_gmres = list()
            err_vec_gmres.append(error_gmres)
            het_net.reset_network(x0,formulation=formulation)
            x_new = x0
            F_new = F0
            J_new = J0
            while error_gmres > tol and iter_gmres < max_iter:
                x_old = x_new
                F_old = F_new
                J_old = J_new
                bnrm2 = np.linalg.norm(F_old)
                bnrm2_scaled = np.linalg.norm(D_F.dot(F_old))
                global res_vec
                res_vec = list()

                tol_gmres = tol*bnrm2_scaled/bnrm2#MM_norm[ind_M]*tol
                #print('tol_gmres = {}'.format(tol_gmres))
                dx,info = sps.linalg.gmres(J_old,F_old,M=MM,tol=tol_gmres,callback=callback_gmres,maxiter=maxmatvecs,restart=max(int(maxmatvecs/10),20)) # dx is unscaled
                print('matvecs = {}, for NR iteration {}, with M={}'.format(len(res_vec),iter_gmres,prec_list[ind_M]))
                if info:
                    print('Linear solver did not convergence. Stopping NR after {} iterations'.format(iter_gmres))
                    break
                #print('||r^k|| = {}'.format(res_vec))
                #print('||F(x^k) - J(x^k)dx|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))))
                #print('||J(x^k)dx - F(x^k)|| = {}'.format(np.linalg.norm(J_old.dot(dx) - F_old)))
                #print('J(x^k)dx - F(x^k) = {}'.format(J_old.dot(dx) - F_old))
                #print('||F(x^k) - J(x^k)dx||/||F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/bnrm2))
                #if ind_M<3:
                    #print('||MM F(x^k) - MM J(x^k)dx|| = {}'.format(np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx)))))
                    #print('||MM F(x^k) - MM J(x^k)dx||/||MM F(x^k)|| = {}'.format((np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx))))/np.linalg.norm(MM.matvec(F_old))))
                    #print('||MM F(x^k) - MM J(x^k)dx||/||F(x^k)|| = {}'.format((np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx))))/bnrm2))
                    #print('||F(x^k) - J(x^k)dx||/||MM F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/np.linalg.norm(MM.matvec(F_old))))

                x_new = x_old - dx
                F_new = nlsys.F(x_new)
                J_new = nlsys.J(x_new)
                error_gmres = np.linalg.norm(D_F.dot(F_new))
                err_vec_gmres.append(error_gmres)
                print('||D_F F(x^k)|| = {}'.format(error_gmres))
                iter_gmres += 1
                max_iters_used_scaled = max(max_iters_used_scaled,iter_gmres)
                #print('iteration {}, error = {}'.format(iter_gmres,error_gmres))
            ax_conv_scaled_gmres.semilogy(err_vec_gmres,'*--',label='prec. gmres, M={}'.format(prec_list[ind_M]))
            err_NR['prec. gmres, M={}'.format(prec_list[ind_M])]=err_vec_gmres
            print('x_gmres dichtbij x_sol: {}, M={}\n'.format(np.allclose(x_sol_expected_SI,x_new),prec_list[ind_M]))

        print('\nHandmatige NR bicgstab:')
        for ind_M,MM in enumerate([LinearOperator(shape=J0.shape, matvec=np.eye(len(x0)).dot, matmat = np.eye(len(x0)).dot, dtype=D_F.dtype),LinearOperator(shape=J0.shape, matvec=D_F.dot, matmat = D_F.dot, dtype=D_F.dtype), LinearOperator(shape=J0.shape, matvec= D_F_inv.dot, matmat = D_F_inv.dot, dtype=D_F.dtype),None]):
            het_net.reset_network(x0,formulation=formulation)
            error_bicgstab = np.linalg.norm(D_F.dot(F0))
            err_vec_bicgstab = list()
            err_vec_bicgstab.append(error_bicgstab)
            x_new = x0
            F_new = F0
            J_new = J0
            iter_bicgstab = 0
            while error_bicgstab > tol and iter_bicgstab < max_iter:
                x_old = x_new
                F_old = F_new
                J_old = J_new
                bnrm2 = np.linalg.norm(F_old)
                bnrm2_scaled = np.linalg.norm(D_F.dot(F_old))
                tol_bicgstab = tol*bnrm2_scaled/bnrm2
                #print('tol_bicgstab = {}'.format(tol_bicgstab))

                res_vec = list()
                global A
                A = J_old
                global rhs
                rhs = F_old

                dx,info = sps.linalg.bicgstab(J_old,F_old,M=MM,tol=tol_bicgstab,callback=callback_bicgstab,maxiter=maxmatvecs) # dx is unscaled.
                if info:
                    print('Linear solver did not convergence. Stopping NR after {} iterations'.format(iter_bicgstab))
                    break
                #print('||F(x^k) - J(x^k)dx|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))))
                #print('||J(x^k)dx - F(x^k)|| = {}'.format(np.linalg.norm(J_old.dot(dx) - F_old)))
                #print('J(x^k)dx - F(x^k) = {}'.format(J_old.dot(dx) - F_old))
                #print('||F(x^k) - J(x^k)dx||/||F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/bnrm2))
                #if ind_M<3:
                    #print('||MM F(x^k) - MM J(x^k)dx|| = {}'.format(np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx)))))
                    #print('||MM F(x^k) - MM J(x^k)dx||/||MM F(x^k)|| = {}'.format((np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx))))/np.linalg.norm(MM.matvec(F_old))))
                    #print('||MM F(x^k) - MM J(x^k)dx||/||F(x^k)|| = {}'.format((np.linalg.norm(MM.matvec(F_old) - MM.matvec(J_old.dot(dx))))/bnrm2))
                    #print('||F(x^k) - J(x^k)dx||/||MM F(x^k)|| = {}'.format(np.linalg.norm(F_old - J_old.dot(dx))/np.linalg.norm(MM.matvec(F_old))))

                x_new = x_old - dx
                F_new = nlsys.F(x_new)
                J_new = nlsys.J(x_new)
                error_bicgstab = np.linalg.norm(D_F.dot(F_new)) # scaled error
                err_vec_bicgstab.append(error_bicgstab)
                print('||D_F F(x^k)|| = {}'.format(error_bicgstab))
                print('matvecs = {}, for NR iteration {}, with M={}'.format(len(res_vec),iter_bicgstab,prec_list[ind_M]))

                iter_bicgstab += 1
                max_iters_used_scaled = max(max_iters_used_scaled,iter_bicgstab)
                #print('iteration {}, error = {}'.format(iter_bicgstab,error_bicgstab))
            ax_conv_scaled_bicgstab.semilogy(err_vec_bicgstab,'x--',label='prec. bicgstab, M={}'.format(prec_list[ind_M]))
            err_NR['prec. bicstab, M={}'.format(prec_list[ind_M])]=err_vec_bicgstab
            print('x_bicgstab dichtbij x_sol: {}, M={}\n'.format(np.allclose(x_sol_expected_SI,x_new),prec_list[ind_M]))
    ax_conv_scaled_gmres.semilogy([0,max_iters_used_scaled],[tol,tol],'k:',label='tol')
    ax_conv_scaled_gmres.set_xlabel("Iteration k")
    ax_conv_scaled_gmres.set_ylabel("Error $||D_F F(x^k)||_2$")
    ax_conv_scaled_gmres.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_scaled_gmres.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_scaled_gmres.legend()
    ax_conv_scaled_bicgstab.semilogy([0,max_iters_used_scaled],[tol,tol],'k:',label='tol')
    ax_conv_scaled_bicgstab.set_xlabel("Iteration k")
    ax_conv_scaled_bicgstab.set_ylabel("Error $||D_F F(x^k)||_2$")
    ax_conv_scaled_bicgstab.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_scaled_bicgstab.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_scaled_bicgstab.legend()

    print('Errors NR:')
    for lin_solver in lin_solvers:
        print('{} : {}'.format(lin_solver,err_NR.get(lin_solver)))
    for prec in prec_list:
        key_gmres = 'prec. gmres, M={}'.format(prec)
        print('{}: {}'.format(key_gmres,err_NR.get(key_gmres)))
        key_bicgstab = 'prec. bicstab, M={}'.format(prec)
        print('{}: {}'.format(key_bicgstab,err_NR.get(key_bicgstab)))

def comp_solver_time_reod(path_to_data,hydr_eq_gas='fa',heat_load='outflow',topology=1,node_set=1,single_coupling=False,EH=False,pg1=29*mbar,pg2=28*mbar,q=.05,V_init=10/np.sqrt(3)*kV,ph1=6*bar,ph0=4*bar,ph2=18*bar,m=6,Ts0=100.,Ts1=95.,Ts2=90.,Tr=50.,Toc0=100,Toc1=100.,Pc=1.5*MW,Qc=MW,phic=2*MW,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},pgbase=50*mbar,qbase=.05,deltabase=1,Vbase=10/np.sqrt(3)*kV,Sbase=MW,phbase=10.*bar,mbase=1.,Tbase=100.,phibase=MW,Egbase=MW,tol=1e-6,max_iter=50,maxmatvecs=5000):
    """Compare the time spent in the linear solver and the total time spent for different orderings and different linear solvers"""
    # solver information
    lin_solvers = ['solve','gmres','bicgstab']
    scale_var = 'matrix'
    scale_var_params = {'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}
    max_iters_used = 0

    # create plot and lay-out
    fig = plt.figure('Convergence plot different solvers and permutations, top. {}, node set {}'.format(topology,node_set))
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||T_F F(x^k)||_2$')
    colors_perm = {'no':'tab:blue','x':'tab:red','xF':'tab:green'}
    markers_solver = {'solve':'.','gmres':'*','bicgstab':'x'}

    # create network
    gas_net,elec_net,heat_net,het_net = create_network(path_to_data,hydr_eq_gas=hydr_eq_gas,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH)
    # initialize
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net,pg1=pg1,pg2=pg2,q=q,V_init=V_init,ph0=ph0,ph1=ph1,ph2=ph2,m=m,Ts0=Ts0,Ts1=Ts1,Ts2=Ts2,Tr=Tr,Toc0=Toc0,Toc1=Toc1,Pc=Pc,Qc=Qc,phic=phic,formulation=formulation,heat_load=heat_load)

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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','N_BP')

    comp_conv_form(path_to_data)

    comp_conv_scaling(path_to_data,topology=1,single_coupling=False,EH=False,consistent_base=False)
    comp_conv_scaling(path_to_data,topology=1,single_coupling=True,EH=False,consistent_base=False)
    comp_conv_scaling(path_to_data,topology=1,single_coupling=True,EH=True,consistent_base=False,hydr_eq_gas='fb',heat_load='delta')

    res_vec = []
    A = None
    rhs = None
    def callback_gmres(r):
        """
        Parameters
        ----------
        r :
            Residual vector
        """
        #print('residual = {}'.format(r))
        global res_vec
        res_vec.append(np.linalg.norm(r))

    def callback_bicgstab(x):
        """
        Parameters
        ----------
        x :
            Solution of Ax=b
        """
        #print('x = {}'.format(x))
        global A
        global rhs
        r = rhs - A.dot(x)
        #print('residual = {}'.format(r))
        global res_vec
        res_vec.append(np.linalg.norm(r))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        comp_lin_solvers(path_to_data,topology=1,node_set=1,single_coupling=False,EH=False)
        comp_solver_time_reod(path_to_data,topology=1,node_set=1,single_coupling=False,EH=False)

    jacobians(path_to_data)

    plt.show()
