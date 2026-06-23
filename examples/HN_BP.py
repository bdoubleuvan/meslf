"""Heat network consisting of 3 demand/source nodes. Also called the reduced benchmark problem."""
from meslf.networks.read_write_network import from_pd_dataframes
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
from meslf.utils.constants import MW, bar
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import warnings
import pytest
import os
import pandas as pd

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning  

# Read the scenario data
def read_scen_data(path_to_data,c_hl=True,topology=1,single_coupling=False):
    """Read the scenario data
    
    Parameters
    ----------
    c_hl : bool, optional
        If true, half links are added to the nodes with values equal to the flow going to the coupling components. Default is True.
    topology : int, optional
        Determines which topology is used in the MES, hence, which is used in the heat ntework when the coupling components are taken into account separately. Options are 1-4. Default is 1. 
    single_coupling : bool, optional
        Determines if a single coupling node (either CHP or EH) is used in the MES, when coupled to one heat node and one heat node (i.e., when topology 1 is used). Default is False. Only used when topology is 1 and if c_hl is True. 
        
    Returns
    -------
    heat_net_scen : HeatNetwork
        The single-carrier scenario network.
    x : list
        List with node x-coordinates. 
    y : list
        List with node y-coordinates.
    mes_net_scen : HeterogeneousNetwork
        The multi-carrier scenario network. Is None if c_hl is False.

    """
    nodes = pd.read_pickle(os.path.join(path_to_data, 'HN_BP_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'HN_BP_links.pkl'))
    #for ind_l in links.index:
        #print('link data = \n{}'.format(links.loc[(ind_l),'data']))
        #print('link data phi unit= {}'.format(links.loc[(ind_l),'data'].get('phi_unit')))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'HN_BP_halflinks.pkl'))
    heat_net_scen = from_pd_dataframes(nodes,links,halflinks)

    # (full) solution of scenario
    m_scen = np.zeros(len(heat_net_scen.links))
    m_hl_scen = np.zeros(len(heat_net_scen.half_links))
    p_scen = np.zeros(len(heat_net_scen.nodes))
    Ts_scen = np.zeros(len(heat_net_scen.nodes))
    Tr_scen = np.zeros(len(heat_net_scen.nodes))
    Ts_hl_scen = np.zeros(len(heat_net_scen.half_links))
    Tr_hl_scen = np.zeros(len(heat_net_scen.half_links))
    phi_scen = np.zeros(len(heat_net_scen.half_links))
    for ind_e,link in enumerate(heat_net_scen.get_links()):
        m_scen[ind_e] = link.m
    for ind_n,node in enumerate(heat_net_scen.get_nodes()):
        p_scen[ind_n] = node.p
        Ts_scen[ind_n] = node.Ts
        Tr_scen[ind_n] = node.Tr
    for ind_hl,half_link in enumerate(heat_net_scen.get_half_links()):
        m_hl_scen[ind_hl] = half_link.m
        Ts_hl_scen[ind_hl] = half_link.Ts
        Tr_hl_scen[ind_hl] = half_link.Tr
        phi_scen[ind_hl] = half_link.dphi
    
    if c_hl:
        mes_data = 'top'+str(topology)
        if single_coupling:
            mes_data += '_1c'
        else:
            mes_data += '_2c'
        nodes_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_nodes_'+mes_data+'.pkl'))
        links_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_links_'+mes_data+'.pkl'))
        halflinks_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_halflinks_'+mes_data+'.pkl'))
        mes_net_scen = from_pd_dataframes(nodes_mes,links_mes,halflinks_mes)

        # (full) solution of scenario, multi-carrier
        heat_nodes = [node for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
        from meslf.networks.heterogeneous_network import HeterogeneousNode
        coupling_nodes = [node for node in mes_net_scen.get_nodes() if isinstance(node,HeterogeneousNode)]
        m_mes_scen = [link.get_m() for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
        Tsstart_mes_scen = [link.get_Tsstart() for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
        Trstart_mes_scen = [link.get_Trstart() for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
        dphistart_mes_scen = [link.get_dphistart() for link in mes_net_scen.get_links() if isinstance(link,HeatLink)]
        m_hl_mes_scen = [hl.get_m() for node in heat_nodes for hl in node.get_half_links()] + [hl.get_m() for node in coupling_nodes for hl in node.get_half_links()]
        Ts_hl_mes_scen = [hl.get_Ts() for node in heat_nodes for hl in node.get_half_links()] + [hl.get_Ts() for node in coupling_nodes for hl in node.get_half_links()]
        Tr_hl_mes_scen = [hl.get_Tr() for node in heat_nodes for hl in node.get_half_links()] + [hl.get_Tr() for node in coupling_nodes for hl in node.get_half_links()]
        phi_mes_scen = [hl.get_dphi() for node in heat_nodes for hl in node.get_half_links()] + [hl.get_dphi() for node in coupling_nodes for hl in node.get_half_links()]
        x = [node.x for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
        y = [node.y for node in mes_net_scen.get_nodes() if isinstance(node,HeatNode)]
        xc = [node.x for node in mes_net_scen.get_nodes() if isinstance(node,HeterogeneousNode)]
        yc = [node.y for node in mes_net_scen.get_nodes() if isinstance(node,HeterogeneousNode)]
    else:
        mes_net_scen = None
        m_mes_scen = []
        Tsstart_mes_scen = []
        Trstart_mes_scen = []
        dphistart_mes_scen = []
        m_hl_mes_scen = []
        Ts_hl_mes_scen = []
        Tr_hl_mes_scen = []
        phi_mes_scen = []
        x = [node.x for node in heat_net_scen.get_nodes()]
        y = [node.y for node in heat_net_scen.get_nodes()]
        xc = []
        yc = []
    return heat_net_scen,m_scen,m_hl_scen,p_scen,Ts_scen,Tr_scen,Ts_hl_scen,Tr_hl_scen,phi_scen,x,y,mes_net_scen,m_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,m_hl_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_mes_scen,xc,yc

def create_network(path_to_data,c_hl=True,heat_load='outflow',topology=1,single_coupling=False):
    """Create a heat network consisting of 3 demand/source nodes. 
    
    Parameters
    ----------
    c_hl : bool, optional
        If true, dummy links and extra nodes (!! not half links !!) are added with values equal to the flow coming from the coupling components. Default is True.
    heat_load : str, optional
        Determines which heat load model is used. Options are 'outflow' and 'delta'. Default is 'outflow'.
    topology : int, optional
        Determines which topology is used in the MES, hence, which is used in the heat network when the coupling components are taken into account separately. Options are 1-4. Default is 1. 
    single_coupling : bool, optional
        Determines if a single coupling node (either CHP or EH) is used in the MES, when coupled to one heat node and one heat node (i.e., when topology 1 is used). Default is False. Only used when topology is 1 and if c_hl is True. 
        
    Returns
    -------
    heat_net : HeatNetwork
        The heat network
    """
    if not topology in [1,2,3,4]:
        raise ValueError('Enter valid value for topology')
    heat_net_scen,m_scen,m_hl_scen,p_scen,Ts_scen,Tr_scen,Ts_hl_scen,Tr_hl_scen,phi_scen,x,y,mes_net_scen,m_mes_scen,Tsstart_mes_scen,Trstart_mes_scen,dphistart_mes_scen,m_hl_mes_scen,Ts_hl_mes_scen,Tr_hl_mes_scen,phi_mes_scen,xc,yc = read_scen_data(path_to_data,c_hl=c_hl,topology=topology,single_coupling=single_coupling)
    
    # physical parameters of network and pipes
    Ta = heat_net_scen.links[0].link_params.get('Ta') 
    
    water = heat_net_scen.links[0].link_params.get('carrier')
    rho_w = water.rhon #[kg/m^3]
    grav_const = water.g #[m/s^2]
    
    heat_link_params = heat_net_scen.links[0].link_params.copy()
    heat_link_type = heat_net_scen.links[0].link_type
    
    # boundary conditions
    p0 = p_scen[0]
    p1 = p_scen[1]
    Ts0 = Ts_scen[0]
    if mes_net_scen:
        # in the MES, node 0 is a junction, so it doesn't have a half link connected to it. So the first half link is the one connected to node 1
        To1_sink = Tr_hl_mes_scen[0]
        To2_sink = Tr_hl_mes_scen[1]
        phi1_sink = phi_mes_scen[0]
        phi2_sink = phi_mes_scen[1]
    else:
        To1_sink = Tr_hl_scen[1]
        To2_sink = Tr_hl_scen[2]
        phi1_sink = phi_scen[1]
        phi2_sink = phi_scen[2] 
    
    dT1 = Ts_scen[1] - To1_sink
    dT2 = Ts_scen[2] - To2_sink
    
    if heat_load == 'outflow':
        h1 = HeatNode('hn1',node_type=1,x=x[1],y=y[1],Tr_hl=To1_sink,dphi=phi1_sink) # load node (sink)
        h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
        h2 = HeatNode('hn2',node_type=1,x=x[2],y=y[2],Tr_hl=To2_sink,dphi=phi2_sink) # load  node (sink)
        h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    else:
        h1 = HeatNode('hn1',node_type=12,x=x[1],y=y[1],dphi=phi1_sink,dT=dT1) # sink temp. diff. node
        h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
        h2 = HeatNode('hn2',node_type=12,x=x[2],y=y[2],dphi=phi2_sink,dT=dT2) # sink temp. diff. node
        h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
            
    if c_hl:
        h0 = HeatNode('hn0',node_type=5,x=x[0],y=y[0],p=p0) # junction ref
        if topology == 1 and single_coupling:
            phi_c = dphistart_mes_scen[-1] #<0
            To_c = Tsstart_mes_scen[-1]
            dTc = To_c - Tr_scen[0]
            h3 = HeatNode('hn_c0',node_type=0,x=xc[0],y=yc[0],Ts=To_c,p=p0) # slack
            h3.half_links[0].set_type('heat_exchanger',{'carrier':water})
        elif topology == 1 or topology == 2:
            h0 = HeatNode('hn0',node_type=5,x=x[0],y=y[0],p=p0) # junction ref
            phi_c0 = dphistart_mes_scen[-2] #<0
            To_c0 = Tsstart_mes_scen[-2]
            dTc0 = To_c0 - Tr_scen[0]
            phi_c1 = dphistart_mes_scen[-1] #<0
            To_c1 = Tsstart_mes_scen[-1]
            dTc1 = To_c1 - Tr_scen[0]
            if heat_load == 'outflow':
                h3 = HeatNode('hn_c0',node_type=0,x=xc[0],y=yc[0],Ts=To_c0,p=p0) # slack
                h3.half_links[0].set_type('heat_exchanger',{'carrier':water})
                h4 = HeatNode('hn_c1',node_type=3,x=xc[1],y=yc[1],Ts_hl=To_c1,dphi=phi_c1,p=p0) # ref. load (source) node (since only connected with a dummy link, so no way to determine pressure)
                h4.half_links[0].set_type('heat_exchanger',{'carrier':water})
            else:
                h3 = HeatNode('hn_c0',node_type=0,x=xc[0],y=yc[0],Ts=To_c0,p=p0) # slack
                h3.half_links[0].set_type('heat_exchanger',{'carrier':water})
                h4 = HeatNode('hn_c1',node_type=13,x=xc[0],y=yc[1],dphi=phi_c1,dT=dTc1,p=p0) # ref. source temp. diff. node (since only connected with a dummy link, so no way to determine pressure)
                h4.half_links[0].set_type('heat_exchanger',{'carrier':water})
        else: # topology 3 and 4
            phi_c0 = dphistart_mes_scen[-2] #<0
            To_c0 = Tsstart_mes_scen[-2]
            dTc0 = To_c0 - Tr_scen[0]
            phi_c1 = dphistart_mes_scen[-1] #<0
            To_c1 = Tsstart_mes_scen[-1]
            dTc1 = To_c1 - Tr_scen[0]
            h0 = HeatNode('hn0',node_type=5,x=x[0],y=y[0],p=p0) # junction ref.
            h3 = HeatNode('hn_c0',node_type=0,x=xc[0],y=yc[0],Ts=To_c0,dphi=phi_c0,p=p0) # slack (since only connected with a dummy link, so no way to determine pressure)
            h3.half_links[0].set_type('heat_exchanger',{'carrier':water})
            if heat_load == 'outflow':
                h4 = HeatNode('hn_c1',node_type=3,x=xc[1],y=yc[1],Ts_hl=To_c1,dphi=phi_c1,p=p1) # ref. load (source) node (since only connected with a dummy link, so no way to determine pressure)
                h4.half_links[0].set_type('heat_exchanger',{'carrier':water})
            else:
                h4 = HeatNode('hn_c1',node_type=13,x=xc[1],y=yc[1],dphi=phi_c1,dT=dTc1,p=p1) # ref. source temp. diff. node (since only connected with a dummy link, so no way to determine pressure)
                h4.half_links[0].set_type('heat_exchanger',{'carrier':water})
    else: # no separate coupling
        h0 = HeatNode('hn0',node_type=0,x=x[0],y=y[0],p=p0,Ts=Ts0) # slack. 
        h0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    
    hl0 = HeatLink('hl0',h0,h1,link_type=heat_link_type,link_params=heat_link_params.copy())
    hl1 = HeatLink('hl1',h0,h2,link_type=heat_link_type,link_params=heat_link_params.copy())
    hl2 = HeatLink('hl2',h1,h2,link_type=heat_link_type,link_params=heat_link_params.copy())
    
    heat_net = HeatNetwork('3 nodes',Ta=Ta)
    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    if c_hl:
        if topology == 1 and single_coupling:
            hl3 = HeatLink('hl3',h3,h0,link_params={'carrier':water})
            heat_net.add_link(hl3)
        elif topology == 1 or topology == 2:
            hl3 = HeatLink('hl3',h3,h0,link_params={'carrier':water})
            hl4 = HeatLink('hl4',h4,h0,link_params={'carrier':water})
            heat_net.add_link(hl3)
            heat_net.add_link(hl4)
        else:
            hl3 = HeatLink('hl3',h3,h0,link_params={'carrier':water})
            hl4 = HeatLink('hl4',h4,h1,link_params={'carrier':water})
            heat_net.add_link(hl3)
            heat_net.add_link(hl4)
    
    return heat_net

def initialize_network(heat_net,p1=6*bar,p2=1*bar,m=6,Ts0=100.,Ts1=95.,Ts2=90.,Ts3=100.,Ts4=90.,Tr=50.,Toc0=110,Toc1=90.,c_hl=True,formulation='standard',heat_load='outflow',topology=1,single_coupling=False,scale_var=None,scale_var_params=None):
    """Initialize the network, based on the topology and formulation.
    
    Parameters
    ----------
    heat_net : HeatNetwork
        The heat network to be initialized
        
    Returns
    -------
    x[0] : np array
        initial guess
    """
    p_init = np.array([p1,p2])
    if c_hl:
        if single_coupling:
            m_init = np.array([m,m,m/6,m])
        else:
            m_init = np.array([m,m,m/6,m,m])
    else:
        m_init = np.array([m,m,m/6])
        
    Tr_init = [Tr,Tr,Tr]
    if topology == 1:
        if c_hl:
            if single_coupling:
                Ts_init = [Ts0,Ts1,Ts2]
                Tr_init.append(Tr) #[C]
                if formulation == 'half_link_flow':
                    m_hl_init = np.array([m,m]) #[kg/s]
                    if heat_load == 'delta':
                        To_hl_init = np.array([Tr,Tr]) #[C]
            else:
                Ts_init = [Ts0,Ts1,Ts2,Ts4]
                Tr_init += [Tr,Tr]
                if formulation == 'half_link_flow':
                    m_hl_init = np.array([m,m,-m]) #[kg/s]
                    if heat_load == 'delta':
                        To_hl_init = np.array([Tr,Tr,Toc1]) #[C]
        else:
            Ts_init = [Ts1,Ts2]
            if formulation == 'half_link_flow':
                m_hl_init = np.array([m,m]) #[kg/s]
                if heat_load == 'delta':
                    To_hl_init = np.array([Tr,Tr]) #[C]
    elif topology == 2:
        if c_hl:
            Ts_init = [Ts0,Ts1,Ts2,Ts4]
            Tr_init += [Tr,Tr]
            if formulation == 'half_link_flow':
                m_hl_init = np.array([m,m,-m]) #[kg/s]
                if heat_load == 'delta':
                    To_hl_init = np.array([Tr,Tr,Toc1]) #[C]
        else:
            Ts_init = [Ts1,Ts2]
            if formulation == 'half_link_flow':
                m_hl_init = np.array([m,m]) #[kg/s]
                if heat_load == 'delta':
                    To_hl_init = np.array([Tr,Tr]) #[C]
    elif topology == 3 or topology == 4:
        if c_hl:
            Ts_init = [Ts0,Ts1,Ts2,Ts4]
            Tr_init += [Tr,Tr]
            if formulation == 'half_link_flow':
                m_hl_init = np.array([-m,m,-m]) #[kg/s]
                if heat_load == 'delta':
                    To_hl_init = np.array([Ts1,Tr,Toc1]) #[C]
        else:
            Ts_init = [Ts1,Ts2]
            if formulation == 'half_link_flow':
                m_hl_init = np.array([-m,m]) #[kg/s]
                if heat_load == 'delta':
                    To_hl_init = np.array([Ts1,Tr]) #[C]
        
    Ts_init = np.array(Ts_init)
    Tr_init = np.array(Tr_init)
    if formulation == 'half_link_flow':
        if heat_load == 'delta':
            x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init,To_hl_init))
        else:
            x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    else:
        x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    
    heat_net.initialize()
    heat_net.update(x_init,formulation=formulation)
    x0 = heat_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def run_load_flow(path_to_data,phbase=10.*bar,mbase=1.,Tbase=130.,phibase=MW,p1=6*bar,p2=18*bar,m=2,Ts0=100.,Ts1=95.,Ts2=90.,Ts3=100.,Ts4=90.,Tr=50.,Toc0=110.,Toc1=90.,c_hl=True,tol=1e-6,max_iter=50,formulation='standard',heat_load='outflow',topology=1,single_coupling=False):
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
    heat_net = create_network(path_to_data,c_hl=c_hl,heat_load=heat_load,topology=topology,single_coupling=single_coupling)
    # initialize
    x0 = initialize_network(heat_net,p1=p1,p2=p2,m=m,Ts0=Ts0,Ts1=Ts1,Ts2=Ts2,Ts3=Ts3,Ts4=Ts4,Tr=Tr,Toc0=Toc0,Toc1=Toc1,c_hl=c_hl,formulation=formulation,heat_load=heat_load,topology=topology,single_coupling=single_coupling)
    
    # solve network
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase})
    
    return heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,tol

def comp_conv_form(path_to_data):
    """Compare convergence for different formulation, and for different heat load models."""
    # make figure to plot convergence
    fig_conv_heat = plt.figure('Convergence plot heat network')
    ax_conv_heat = fig_conv_heat.gca()
    max_iters_used = 0
    markers_heat = {'standard outflow':'.','standard delta':'*','half_link_flow outflow':'d','half_link_flow delta':'x'}
    
    topology = 4
    single_coupling = False
    for c_hl in [False,True]:
        for form in ['standard', 'half_link_flow']:
            for heat_load in ['outflow','delta']:
                print('\nFormulation is {}, and heat load is {}, and separate couplings is {}'.format(form,heat_load,c_hl))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                    heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,tol = run_load_flow(path_to_data,c_hl=c_hl,heat_load=heat_load,formulation=form,topology=topology,single_coupling=single_coupling)
                print('Solution (with final error {:.4e}):'.format(err_vec[-1]))
                print('p heat = {} bar'.format(p_vec/(bar)))
                print('m = {}'.format(m_vec))
                print('Ts = {}'.format(Ts_vec))
                print('Tr = {}'.format(Tr_vec))
                print('m hl = {}'.format(m_hl_vec))
                print('Ts hl = {}'.format(Ts_hl_vec))
                print('Tr hl = {}'.format(Tr_hl_vec))
                print('phi hl = {}'.format(phi_hl_vec))
                print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
                key = '{} {}'.format(form,heat_load)
                # plot convergence
                if c_hl:
                    ls = '--'
                    label = key + ', sep. coup.'
                else:
                    ls = '-'
                    label = key
                ax_conv_heat.semilogy(err_vec,ls=ls,color='tab:blue',marker=markers_heat.get(key),label=label)
                max_iters_used = max(max_iters_used,iters)
                fig_top = plt.figure('Network topology {}, separate coupling {}'.format(topology,c_hl))
                ax_top = fig_top.gca()
                if not ax_top.lines:
                    heat_net.draw_network(ax_top,halflink_angle=1,halflink_length=1)
                    plt.axis('equal')
                    plt.axis('off')
    ax_conv_heat.set_xlabel(r'Iteration $k$')
    ax_conv_heat.set_ylabel(r'Error $||D_F F(x^k)||_2$')
    ax_conv_heat.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
    ax_conv_heat.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_heat.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_heat.legend()
    xmin = 0
    xmax = max_iters_used
    xticks = range(xmin,xmax+1,2) # make sure the xticks are integers
    ax_conv_heat.set_xlim(left=xmin,right=xmax+1)
    ax_conv_heat.set_xticks(xticks)

def comp_conv_scaling(path_to_data,heat_load='outflow',topology=1,single_coupling=False):
    """Compare convergence of NR for different ways of scaling."""
    # base values
    phbase=10.*bar
    mbase=1.
    Tbase=130.
    phibase=MW
    
            
    # create networks
    c_hl = False # no separate coupling
    # network with values specified in S.I.
    heat_net_SI = create_network(path_to_data,c_hl=c_hl,heat_load=heat_load,topology=topology,single_coupling=single_coupling)
    # network with values specified in p.u.
    heat_net_pu = create_network(path_to_data,c_hl=c_hl,heat_load=heat_load,topology=topology,single_coupling=single_coupling)
    Ta_SI = heat_net_pu.Ta
    Ta_pu = Ta_SI/Tbase
    heat_net_pu.Ta = Ta_pu
    water_pu = heat_net_pu.links[0].link_params.get('carrier') # still in S.I. units
    Cp_b = phibase/(Tbase*mbase)
    water_pu.Cp /= Cp_b
    for link in heat_net_pu.get_links():
        if not link.link_type == 'standard_pipe_low_pres_pole':
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
    
    # initial conditions
    p1=6*bar
    p2=1*bar
    m=6
    Ts0=100.
    Ts1=95.
    Ts2=90.
    Tr=50.
    
    formulation = 'half_link_flow'
    tol=1e-6
    max_iter=50
    # run load flow for network with values specified in S.I., using matrix scaling
    # initialize
    x0 = initialize_network(heat_net_SI,p1=p1,p2=p2,m=m,Ts0=Ts0,Ts1=Ts1,Ts2=Ts2,Tr=Tr,c_hl=c_hl,formulation=formulation,heat_load=heat_load,topology=topology,single_coupling=single_coupling)
    # solve network
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        x_sol_scaled,iters_scaled,err_vec_scaled,m_vec_scaled,p_vec_scaled,Ts_vec_scaled,Tr_vec_scaled,m_hl_vec_scaled,phi_hl_vec_scaled,Ts_hl_vec_scaled,Tr_hl_vec_scaled = heat_net_SI.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase})
    
    # run load flow for network with values specified in S.I., using p.u. scaling
    scale_var='per_unit'
    scale_var_params={'qbase':mbase,'pbase':phbase,'Tbase':Tbase,'phibase':phibase}
    # initialize
    heat_net_SI.reset_network(x0,formulation=formulation)
    x0_SI_pu = initialize_network(heat_net_SI,p1=p1,p2=p2,m=m,Ts0=Ts0,Ts1=Ts1,Ts2=Ts2,Tr=Tr,c_hl=c_hl,formulation=formulation,heat_load=heat_load,topology=topology,single_coupling=single_coupling,scale_var=scale_var,scale_var_params=scale_var_params)
    # solve network
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        x_sol_SI_pu,iters_SI_pu,err_vec_SI_pu,m_vec_SI_pu,p_vec_SI_pu,Ts_vec_SI_pu,Tr_vec_SI_pu,m_hl_vec_SI_pu,phi_hl_vec_SI_pu,Ts_hl_vec_SI_pu,Tr_hl_vec_SI_pu = heat_net_SI.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    
    # run load flow for network with values specified in p.u., without scaling
    # initialize
    x0_pu = initialize_network(heat_net_pu,p1=p1/phbase,p2=p2/phbase,m=m/mbase,Ts0=Ts0/Tbase,Ts1=Ts1/Tbase,Ts2=Ts2/Tbase,Tr=Tr/Tbase,c_hl=c_hl,formulation=formulation,heat_load=heat_load,topology=topology,single_coupling=single_coupling)
    # solve network
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        x_sol_pu,iters_pu,err_vec_pu,m_vec_pu,p_vec_pu,Ts_vec_pu,Tr_vec_pu,m_hl_vec_pu,phi_hl_vec_pu,Ts_hl_vec_pu,Tr_hl_vec_pu = heat_net_pu.solve_network(tol,max_iter,solver='NR',formulation=formulation)
    
    print('Errors. Par. in S.I., matrix scaling:\n{}'.format(err_vec_scaled))
    print('Errors. Par. in S.I., p.u. scaling:\n{}'.format(err_vec_SI_pu))
    print('Errors. Par. in p.u., unscaled:\n{}'.format(err_vec_pu))
    print('Solution. Par. in S.I., matrix scaling:\n{}'.format(x_sol_scaled))
    print('Solution. Par. in S.I., p.u. scaling:\n{}'.format(x_sol_SI_pu))
    print('Solution. Par. in p.u., unscaled:\n{}'.format(x_sol_pu))
    
    # make figure to plot convergence
    fig_conv_heat = plt.figure('Convergence plot heat network, scaling')
    ax_conv_heat = fig_conv_heat.gca()
    max_iters_used = max([iters_scaled,iters_SI_pu,iters_pu])
    markers_heat = {'standard outflow':'.','standard delta':'*','half_link_flow outflow':'d','half_link_flow delta':'x'}
    ls = '-'
    ax_conv_heat.semilogy(err_vec_scaled,ls=ls,color='tab:blue',marker=markers_heat.get(formulation+' '+heat_load),label='matrix scaling')
    ax_conv_heat.semilogy(err_vec_SI_pu,ls=ls,color='tab:orange',marker=markers_heat.get(formulation+' '+heat_load),label='p.u. scaling')
    ax_conv_heat.semilogy(err_vec_pu,ls=ls,color='tab:red',marker=markers_heat.get(formulation+' '+heat_load),label='specified in p.u.')
    ax_conv_heat.set_xlabel(r'Iteration $k$')
    ax_conv_heat.set_ylabel(r'Error ($||D_F F(x^k)||_2$ or $||F(x^k)||_2$)')
    ax_conv_heat.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
    ax_conv_heat.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_heat.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_heat.legend()
    xmin = 0
    xmax = max_iters_used
    xticks = range(xmin,xmax+1,2) # make sure the xticks are integers
    ax_conv_heat.set_xlim(left=xmin,right=xmax+1)
    ax_conv_heat.set_xticks(xticks)
    
if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','N_BP')
    comp_conv_form(path_to_data)
    comp_conv_scaling(path_to_data)
    plt.show()
