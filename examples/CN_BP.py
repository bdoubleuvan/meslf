"""The coupling part of the reduced benchmark problem."""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.utils.constants import mbar, bar, hour, kV, MW, MBTU, BTU
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
import pandas as pd
from meslf.networks.read_write_network import from_pd_dataframes
import pytest

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning 

# Read the scenario data of the MES
def read_scen_data(path_to_data,topology=1,single_coupling=False):
    """Read the scenario data
    
    Parameters
    ----------
    topology : int, optional
        Determines which topology is used in the MES, hence, which is used in the heat network when the coupling components are taken into account separately. Options are 1-4. Default is 1. 
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
    Tsstart_mes_scen = [link.get_Tsstart() for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
    Trstart_mes_scen = [link.get_Trstart() for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
    dphistart_mes_scen = [link.get_dphistart() for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
    ph_mes_scen = [node.get_p() for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
    Ts_mes_scen = [node.get_Ts() for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
    Tr_mes_scen = [node.get_Tr() for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
    Ts_hl_mes_scen = [hl.get_Ts() for node in heat_nodes for hl in node.get_half_links()] + [hl.get_Ts() for node in coupling_nodes for hl in node.get_half_links()]
    Tr_hl_mes_scen = [hl.get_Tr() for node in heat_nodes for hl in node.get_half_links()] + [hl.get_Tr() for node in coupling_nodes for hl in node.get_half_links()]
    phi_mes_scen = [hl.get_dphi() for node in heat_nodes for hl in node.get_half_links()] + [hl.get_dphi() for node in coupling_nodes for hl in node.get_half_links()]
    xh = [node.x for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
    yh = [node.y for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
    # coupling part (the actual values are included in the single-carrier parts)
    xc = [node.x for node in mes_net_scen.get_nodes() if isinstance(node,HeterogeneousNode)]
    yc = [node.y for node in mes_net_scen.get_nodes() if isinstance(node,HeterogeneousNode)]
    
    return mes_net_scen,xg,yg,xe,ye,xh,yh,xc,yc,q_mes_scen,pg_mes_scen,q_hl_mes_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen,P_edge_mes_scen,Q_edge_mes_scen,m_mes_scen,m_hl_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,ph_mes_scen,Ts_mes_scen,Tr_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_mes_scen

def sol_from_scen_data(path_to_data,het_net,topology=1,single_coupling=False,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}):
    # read scenario data (to get the values not included in the created network)
    mes_net_scen,_,_,_,_,_,_,_,yc,q_mes_scen,pg_mes_scen,q_hl_mes_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen,P_edge_mes_scen,Q_edge_mes_scen,m_mes_scen,m_hl_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,ph_mes_scen,Ts_mes_scen,Tr_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_mes_scen = read_scen_data(path_to_data,topology=topology,single_coupling=single_coupling)
    
    # create solution vector from this data (since not all this data is assigned to the network when reading from a df)
    het_net.initialize() # to assign numbers to nodes and link
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)
    # coupling
    qc_sol = np.array([q_mes_scen[ind_hl+3] for ind_hl,link in enumerate(unknown_qc_halflinks)])
    Pc_sol = np.array([P_edge_mes_scen[ind_hl+3] for ind_hl,link in enumerate(unknown_Pc_halflinks)])
    Qc_sol = np.array([Q_edge_mes_scen[ind_hl+3] for ind_hl,link in enumerate(unknown_Qc_halflinks)])
    Sc_sol = np.concatenate((Pc_sol,Qc_sol))
    mc_sol = np.array([m_mes_scen[ind_hl+3] for ind_hl,link in enumerate(unknown_mc_halflinks)])
    phic_sol = np.array([-dphistart_mes_scen[ind_hl+3] for ind_hl,link in enumerate(unknown_dphic_halflinks)])
    Tsc_sol = np.array([Tsstart_mes_scen[ind_hl+3] for ind_hl,link in enumerate(unknown_Tsc_halflinks)])
    Trc_sol = np.array([Trstart_mes_scen[ind_hl+3] for ind_hl,link in enumerate(unknown_Trc_halflinks)])
    xc_sol = np.concatenate((qc_sol,Sc_sol,mc_sol,phic_sol,Tsc_sol,Trc_sol))
    return xc_sol
    
def create_network(path_to_data,topology=1,node_set=1,heat_load='outflow',single_coupling=False,EH=True):
    """Create the coupling part of the multi-carrier network.
    
    Parameters
    ----------------
    topology : int, optional
        Determines which topology is used. Options are 1-4. Default is 1. 
    single_coupling : bool, optional
        Determines if a single coupling node (either CHP or EH) is used in the MES, when coupled to one gas node and one heat node (i.e., when topology 1 is used). Default is False. Only used when topology is 1.
    EH : bool, optional 
        Determines if an EH is used as the single coupling. When False, a CHP is used. Only used if single_coupling is True. Default is True.
        
    Returns
    -----------
    het_net : HeterogeneousNetwork
        The heterogeneous network
    """
    if not topology in [1,2,3,4]:
        raise ValueError('Enter valid value for topology')
    mes_net_scen,xg,yg,xe,ye,xh,yh,xc,yc,q_mes_scen,pg_mes_scen,q_hl_mes_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen,P_edge_mes_scen,Q_edge_mes_scen,m_mes_scen,m_hl_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,ph_mes_scen,Ts_mes_scen,Tr_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_mes_scen = read_scen_data(path_to_data,topology=topology,single_coupling=single_coupling)
    
    # coupling nodes
    coupling_nodes = [node for node in mes_net_scen.get_nodes() if isinstance(node,HeterogeneousNode)]
    
    heat_links = [link for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
    water = heat_links[0].link_params.get('carrier')
    
    # coupling
    if topology == 1:
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
            phi_c = dphistart_mes_scen[3] #<0
            To_c = Tsstart_mes_scen[3]
            Tr_c = Trstart_mes_scen[3] # This value must be set. Normally comes from the heat network
            dTc = To_c - Tr_mes_scen[0]
            Q_c = Q_edge_mes_scen[3]
            if EH:
                if heat_load == 'outflow':
                    cn = HeterogeneousNode('cn',node_type=1,x=xc[0],y=yc[0],unit_type=unit_type,unit_params=unit_params) # To known
                    hlh = HeatHalfLink('cn_hlh',cn,link_type='heat_exchanger',link_params={'carrier':water},bc_type=2,Ts=To_c,dphi=phi_c) # Ts and dphi known, source
                else:
                    cn = HeterogeneousNode('cn',node_type=2,x=xc[0],y=yc[0],unit_type=unit_type,unit_params=unit_params) # dT known
                    hlh = HeatHalfLink('cn_hlh',cn,link_type='heat_exchanger',link_params={'carrier':water},bc_type=4,dT=dTc,dphi=phi_c) # dT and dphi known, source
                hlh.Tr = Tr_c # Set to solution
                hlg = GasHalfLink('cn_hlg',cn,-0,bc_type=0) # gas flows into coupling node, q is unknown
                hle = ElectricalHalfLink('cn_hle',cn,Q=Q_c,bc_type=3) # P is unknown, Q is known
    
    het_net = HeterogeneousNetwork('coupling net')
    het_net.add_node(cn)
    
    return het_net

def initialize_network(het_net,q=.05,m=6,Toc0=105,Toc1=95,Pc=1.5*MW,Qc=MW,phic=2*MW,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},heat_load='outflow'):
    """Initialize the gas network, consisting of 3 demand/source nodes, and one extra node due to an compressor.
    
    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The heterogeneous network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """   
    if heat_load == 'outflow':
        x_init = np.array([q,Pc,m])
    else:
        x_init = np.array([q,Pc,m,Toc0])
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)
    return x0    
    
def run_load_flow(path_to_data,topology=1,node_set=1,heat_load='outflow',single_coupling=False,EH=True,q=.05,m=6,Toc0=105,Toc1=95,Pc=1.5*MW,Qc=MW,phic=2*MW,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},pgbase=50*mbar,qbase=.05,Vbase=10/np.sqrt(3)*kV,Sbase=MW,phbase=10.*bar,mbase=1.,Tbase=100.,phibase=MW,Egbase=MW,tol=1e-6,max_iter=50,plot_top=False,scale_var='matrix'):
    # create network
    # create network
    het_net = create_network(path_to_data,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling, EH=EH)
    
    # initialize
    x0 = initialize_network(het_net,q=q,m=m,Toc0=Toc0,Toc1=Toc1,Pc=Pc,Qc=Qc,phic=phic,formulation=formulation,heat_load=heat_load)
    
    # solve network
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
    
    if plot_top:
        # plot topology
        mes_net_scen,xg,yg,xe,ye,xh,yh,xc,yc,q_mes_scen,pg_mes_scen,q_hl_mes_scen,delta_mes_scen,V_mes_scen,P_inj_mes_scen,Q_inj_mes_scen,P_edge_mes_scen,Q_edge_mes_scen,m_mes_scen,m_hl_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,ph_mes_scen,Ts_mes_scen,Tr_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_mes_scen = read_scen_data(path_to_data,topology=topology,single_coupling=single_coupling)
        fig_top = plt.figure('Network topology')
        ax_top = fig_top.gca()
        het_net.draw_network(ax_top,halflink_angle=2,halflink_length=1)
        plt.axis('equal')
        plt.axis('off')
        
    return het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol
    
@pytest.mark.filterwarnings("ignore::UserWarning")
def example_cn_bp_top1_EH_unscaled():
    # Given
    topology = 1
    node_set = 1 # same one as used to create the scenario data??
    single_coupling = True
    EH = True
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'outflow'
    
    # When
    path_to_data = './examples/network_data/N_BP'
    # run load flow
    het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(path_to_data,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH,formulation=formulation,tol=1e-6,max_iter=50,plot_top=False,scale_var=None)
                
    # Then 
    x_sol_expected_SI = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation) #S.I., but x_sol is also in S.I, despite using scaling
    assert np.allclose(x_sol,x_sol_expected_SI)
    
@pytest.mark.filterwarnings("ignore::UserWarning")
def example_cn_bp_top1_EH_scaled():
    # Given
    topology = 1
    node_set = 1 # same one as used to create the scenario data??
    single_coupling = True
    EH = True
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'outflow'
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
    het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(path_to_data,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH,formulation=formulation,pgbase=pgbase,qbase=qbase,Vbase=Vbase,Sbase=Sbase,phbase=phbase,mbase=mbase,Tbase=Tbase,phibase=phibase,Egbase=Egbase,tol=1e-6,max_iter=50,plot_top=False,scale_var='matrix')
    # Then 
    x_sol_expected_SI = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation) #S.I., but x_sol is also in S.I, despite using scaling
    print('x_sol = {}'.format(x_sol))
    print('err_vec = {}'.format(err_vec))
    print('x_sol_expected_SI = {}'.format(x_sol_expected_SI))
    assert np.allclose(x_sol,x_sol_expected_SI)
    
@pytest.mark.filterwarnings("ignore::UserWarning")
def example_cn_bp_top1_EH_unscaled_dT():
    # Given
    topology = 1
    node_set = 1 # same one as used to create the scenario data??
    single_coupling = True
    EH = True
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'delta'
    
    # When
    path_to_data = './examples/network_data/N_BP'
    # run load flow
    het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(path_to_data,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH,formulation=formulation,tol=1e-6,max_iter=50,plot_top=False,scale_var=None)
                
    # Then 
    x_sol_expected_SI = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation) #S.I., but x_sol is also in S.I, despite using scaling
    print('x_sol = {}'.format(x_sol))
    print('err_vec = {}'.format(err_vec))
    print('x_sol_expected_SI = {}'.format(x_sol_expected_SI))
    assert np.allclose(x_sol,x_sol_expected_SI)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_cn_bp_top1_EH_scaled_dT():
    # Given
    topology = 1
    node_set = 1 # same one as used to create the scenario data??
    single_coupling = True
    EH = True
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
    heat_load = 'delta'
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
    het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,tol = run_load_flow(path_to_data,heat_load=heat_load,topology=topology,node_set=node_set,single_coupling=single_coupling,EH=EH,formulation=formulation,pgbase=pgbase,qbase=qbase,Vbase=Vbase,Sbase=Sbase,phbase=phbase,mbase=mbase,Tbase=Tbase,phibase=phibase,Egbase=Egbase,tol=1e-6,max_iter=50,plot_top=False,scale_var='matrix')
    # Then 
    x_sol_expected_SI = sol_from_scen_data(path_to_data,het_net,topology=topology,single_coupling=single_coupling,formulation=formulation) #S.I., but x_sol is also in S.I, despite using scaling
    print('x_sol = {}'.format(x_sol))
    print('err_vec = {}'.format(err_vec))
    print('x_sol_expected_SI = {}'.format(x_sol_expected_SI))
    assert np.allclose(x_sol,x_sol_expected_SI)
    
if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','N_BP')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "No single-carrier subnetworks found",UserWarning)
        run_load_flow(path_to_data,single_coupling=True)
    plt.show()
        
    
    
