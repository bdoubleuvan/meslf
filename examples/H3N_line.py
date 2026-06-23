"""Example of a heat network with 3 nodes connected in a single line"""
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink
from meslf.networks.carrier import Water
from meslf.utils.constants import bar, MW, km
from meslf.utils.list_manipulation import flatten
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
import pytest 
import argparse 

command_line_input = argparse.ArgumentParser()
command_line_input.add_argument(
    "--sep_figs", # When True, the convergence plots are created in seperate figures as well, and are saved
    type=bool,
    default = False, # default if nothing is provided
    )

# water carrier
rho_w = 960. #[kg/m^3]
Cp_w = 4.182e3 #[J/(kg K)]
grav_const = 9.81 #[m/s^2]
water = Water('water',Cp_w,rho=rho_w)

# physical parameters of network and pipes
Ta = 10.
    
def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning 

def create_network(Ts1=100.,p1=9*bar,Tr1=50.,To2=100.,To3=50.,phi2=-1*MW,phi3=1.5*MW,dT2=50.,dT3=50.,L12 =4*km,L23=5*km,D=0.15,lam=0.2,link_type='standard_pipe_low_pres_pole',node_set=1,slack_node='source',source_node='outflow',sink_node='outflow'):
    """Create a heat network consisting of 3 nodes in one line.
    
    Parameters
    ----------
    node_set : int, optional
        Node set to use. Node set 1 corresponds to ... and node set 2 corresponds to ... . Default is node set 1
    slack_node : str, optional
        Determines if the slack node should be a source (ref.) slack node, a sink (ref.) slack node, or a general (ref.) slack node. Options are 'source', 'sink', , 'general_Ts', 'general_Tr', or 'general'. Default is 'source'.
    source_node : str, optional
        Determines which model is used for the load (source) at node 2. In both cases, the heat power is specified. If 'outflow', the outlfow temperature is specified. If 'delta', the temperature difference between supply and outflow temperature is specified. Default is 'outflow'. NB. Only used if node_set is 2, otherwise node 2 is a junction, i.e. there is no load connected to node 2
    sink_node : str, optional
        Determines which model is used for the load (sink) at node 3. In both cases, the heat power is specified. If 'outflow', the outlfow temperature is specified. If 'delta', the temperature difference between supply and outflow temperature is specified. Default is 'outflow'.
        
    Returns
    -------
    heat_net : HeatNetwork
        The test network
    """
    heat_net = HeatNetwork('3N one line',Ta=Ta)
    if slack_node == 'source':
        hn1 = HeatNode('hn1',node_type=0,x=0,y=0,Ts=Ts1,p=p1) # source slack node
        hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif slack_node == 'general_Ts':
        hn1 = HeatNode('hn1',node_type=10,x=0,y=0,Ts=Ts1,p=p1) # general slack node (with p and Ts known)
        hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})   
    elif slack_node == 'sink':
        hn1 = HeatNode('hn1',node_type=8,x=0,y=0,Tr=Tr1,p=p1) # sink slack node
        hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})  
    elif slack_node == 'general_Tr':
        hn1 = HeatNode('hn1',node_type=11,x=0,y=0,Tr=Tr1,p=p1) # general slack node (with p and Tr known)
        hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif slack_node == 'general':
        hn1 = HeatNode('hn1',node_type=9,x=0,y=0,p=p1) # general slack node (with p known)
        hn1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    else:
        raise ValueError('Enter valid value for slack_node')
    if node_set == 1:
        hn2 = HeatNode('hn2',node_type=2,x=1,y=0) # junction node
    elif node_set == 2:
        if source_node == 'outflow':
            hn2 = HeatNode('hn2',node_type=1,x=1,y=0,Ts_hl=To2,dphi=phi2) # load node (source)
            hn2.half_links[0].set_type('heat_exchanger',{'carrier':water})
        elif source_node == 'delta':
            hn2 = HeatNode('hn2',node_type=12,x=1,y=0,dphi=phi2,dT=dT2) # source temp. diff. node
            hn2.half_links[0].set_type('heat_exchanger',{'carrier':water})
        else:
            raise ValueError('Enter valid value for source_node')
    else:
        raise ValueError('Enter valid value for node_set')
    if sink_node == 'outflow':
        hn3 = HeatNode('hn3',node_type=1,x=2,y=0,Tr_hl=To3,dphi=phi3) # load node (sink)
        hn3.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif sink_node == 'delta':
        hn3 = HeatNode('hn3',node_type=12,x=2,y=0,dphi=phi3,dT=dT3) # sink temp. diff. node
        hn3.half_links[0].set_type('heat_exchanger',{'carrier':water})
    else:
        raise ValueError('Enter valid value for sink_node')
    
    U = lam/(np.pi*D) #[W/(m^2 K)]
    link_params12 = {'L':L12,'D':D,'U':U,'carrier':water}
    link_params23 = link_params12.copy()
    link_params23['L'] = L23
    
    hl0 = HeatLink('hl0',hn1,hn2,link_type=link_type,link_params=link_params12)
    hl1 = HeatLink('hl1',hn2,hn3,link_type=link_type,link_params=link_params23)

    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    
    return heat_net

def initialize_network(heat_net,m0_init=1,m1_init=1,p2_init=8*bar,p3_init=6*bar,Ts1_init=100.,Ts2_init=100.,Ts3_init=100.,Tr1_init=50.,Tr2_init=50.,Tr3_init=50.,formulation='standard',node_set=1,slack_node='source',source_node='outflow',sink_node='outflow'):
    """Create a heat network consisting of 3 nodes in one line.
    
    Parameters
    ----------
    heat_net : HeatNetwork
        The heat network to be initialized
    formulation : str, optional
        Formulation of the non-linear system of equations used to solve the network.
    node_set : int, optional
        Node set to use. Node set 1 corresponds to ... and node set 2 corresponds to ... . Default is node set 1
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    m_init = np.array([m0_init,m1_init]) #[kg/s]
    p_init = np.array([p2_init,p3_init]) #[Pa]
    if slack_node in ['sink','general_Tr']:
        Ts_init = np.array([Ts1_init,Ts2_init,Ts3_init]) #[C]
        Tr_init = np.array([Tr2_init,Tr3_init]) #[C]
    elif slack_node == 'general':
        Ts_init = np.array([Ts1_init,Ts2_init,Ts3_init]) #[C]
        Tr_init = np.array([Tr1_init,Tr2_init,Tr3_init]) #[C]
    else:
        Ts_init = np.array([Ts2_init,Ts3_init]) #[C]
        Tr_init = np.array([Tr1_init,Tr2_init,Tr3_init]) #[C]
    if formulation == 'half_link_flow':
        if heat_net.nodes[1].node_type == 12: #dT known, Ts unknown
            Ts_hl_init = np.array([Ts2_init])
        else:
            Ts_hl_init = np.array([])
        if heat_net.nodes[2].node_type == 12: #dT known, Tr unknown
            Tr_hl_init = np.array([Tr3_init])
        else:
            Tr_hl_init = np.array([])
        # doesn't use Ts_init and Tr_init, but the default 100 for Ts and 50 for Tr
        if node_set == 1:
            m_hl_init = np.array([1])
        elif node_set == 2:
            m_hl_init = np.array([-1,1]) 
        x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init,Ts_hl_init,Tr_hl_init))
    else:
        x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    heat_net.update(x_init,formulation=formulation)
    x0 = heat_net.set_x_init(formulation=formulation)
    return x0

def run_load_flow(phbase,mbase,Tbase,phibase,Ts1=100.,p1=9*bar,Tr1=50.,To2=100.,To3=50.,phi2=-1*MW,phi3=1.5*MW,dT2=50.,dT3=50.,m0_init=1,m1_init=1,p2_init=8*bar,p3_init=6*bar,Ts1_init=100.,Ts2_init=100.,Ts3_init=100.,Tr1_init=50.,Tr2_init=50.,Tr3_init=50.,L12 =4*km,L23=5*km,D=0.15,lam=0.2,link_type='standard_pipe_low_pres_pole',formulation='standard',node_set=1,slack_node='source',source_node='outflow',sink_node='outflow',tol = 1e-6,max_iter = 150):
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
    node_set : int, optional
        Node set to use. Node set 1 corresponds to ... and node set 2 corresponds to ... . Default is node set 1
    formulation : string
        Formulation that is used in the solver.
    node_set : int, optional
        Node set to use. Node set 1 corresponds to ... and node set 2 corresponds to ... . Default is node set 1
    slack_node : str, optional
        Determines if the slack node should be a source (ref.) slack node, a sink (ref.) slack node, or a general (ref.) slack node. Options are 'source', 'sink', , 'general_Ts', 'general_Tr', or 'general'. Default is 'source'.
    sink_node : str, optional
        Determines which model is used for the load (sink). In both cases, the heat power is specified. If 'outflow', the outlfow temperature is specified. If 'delta', the temperature difference between supply and outflow temperature is specified. Default is 'outflow'.
        
    """
    print('\nSolving heat network with node set {}, formulation {}, {} slack node, {} source node, {} sink node'.format(node_set,formulation,slack_node,source_node,sink_node))
    # create network
    heat_net = create_network(Ts1=Ts1,p1=p1,Tr1=Tr1,To2=To2,To3=To3,phi2=phi2,phi3=phi3,dT2=dT2,dT3=dT3,L12=L12,L23=L23,D=D,lam=lam,link_type=link_type,node_set=node_set,slack_node=slack_node,source_node=source_node,sink_node=sink_node)
    # initialize
    x0 = initialize_network(heat_net,m0_init=m0_init,m1_init=m1_init,p2_init=p2_init,p3_init=p3_init,Ts1_init=Ts1_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,formulation=formulation,node_set=node_set,slack_node=slack_node,source_node=source_node,sink_node=sink_node)
    
    # solve network
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase})
    print('Error is {} after {} iterations'.format(err_vec[-1],iters))
        
    return heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_H3N_line_node3_temp_diff():
    """Check solution for this topology, with node 1 a sink slack node, node 2 a normal source node, and node 3 a sink node with temperature difference known. Using the standard half link flow formulation."""
    # Given
    formulation = 'standard'
    # boundary conditions (for a standard load (sink) node for node 3)
    Ts1=93.72276227854158
    p1=10*bar
    Tr1=-3.2139721171872058
    To2=102.12615673584322
    To3=50.
    phi2=-2*MW
    phi3=1*MW
    # initial guess
    m0_init= -1. #[kg/s]
    m1_init=1 #[kg/s]
    p2_init=8*bar #[Pa]
    p3_init=6*bar #[Pa]
    Ts1_init=100. #[C]
    Ts2_init=100. #[C]
    Ts3_init=100. #[C]
    Tr1_init=Tr1 #[C]
    Tr2_init=50. #[C]
    Tr3_init=50. #[C]
    # solver information
    tol = 1e-6
    max_iter = 50
    # scaling
    phbase = bar
    mbase = 1.
    Tbase = 100.
    phibase = MW
    # solution (for a standard load (sink) node for node 3)
    _,_,_,_,m_vec_outflow,p_vec_outflow,Ts_vec_outflow,Tr_vec_outflow,m_hl_vec_outflow,phi_hl_vec_outflow,Ts_hl_vec_outflow,Tr_hl_vec_outflow = run_load_flow(phbase,mbase,Tbase,phibase,p1=p1,Tr1=Tr1,To2=To2,To3=To3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2_init,p3_init=p3_init,Ts1_init=Ts1_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,formulation=formulation,node_set=2,slack_node='sink',sink_node='outflow',tol=tol,max_iter=max_iter)
    full_sol_outflow = np.concatenate([m_vec_outflow,p_vec_outflow,Ts_vec_outflow,Tr_vec_outflow,flatten(m_hl_vec_outflow),flatten(phi_hl_vec_outflow),flatten(Ts_hl_vec_outflow),flatten(Tr_hl_vec_outflow)])
    
    # When
    Ts3_sol = Ts_vec_outflow[2] #solution for Ts3
    dT3 = Ts3_sol-To3 
    _,_,_,_,m_vec_delta,p_vec_delta,Ts_vec_delta,Tr_vec_delta,m_hl_vec_delta,phi_hl_vec_delta,Ts_hl_vec_delta,Tr_hl_vec_delta = run_load_flow(phbase,mbase,Tbase,phibase,p1=p1,Tr1=Tr1,To2=To2,dT3=dT3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2_init,p3_init=p3_init,Ts1_init=Ts1_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,formulation=formulation,node_set=2,slack_node='sink',sink_node='delta',tol=tol,max_iter=max_iter)
    full_sol_delta = np.concatenate([m_vec_delta,p_vec_delta,Ts_vec_delta,Tr_vec_delta,flatten(m_hl_vec_delta),flatten(phi_hl_vec_delta),flatten(Ts_hl_vec_delta),flatten(Tr_hl_vec_delta)])
    assert np.allclose(full_sol_delta,full_sol_outflow)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_H3N_line_node2_node3_temp_diff_half_link_flow():
    """Check solution for this topology, with node 1 a sink slack node, node 2 a normal source node, and node 3 a sink node with temperature difference known. Using the unknown half link flow formulation."""
    # Given
    formulation = 'half_link_flow'
    # boundary conditions (for a standard load (sink) node for node 3)
    Ts1=93.72276227854158
    p1=10*bar
    Tr1=-3.2139721171872058
    To2=102.12615673584322
    To3=50.
    phi2=-2*MW
    phi3=1*MW
    # initial guess
    m0_init= -1. #[kg/s]
    m1_init=1 #[kg/s]
    p2_init=8*bar #[Pa]
    p3_init=6*bar #[Pa]
    Ts1_init=100. #[C]
    Ts2_init=100. #[C]
    Ts3_init=100. #[C]
    Tr1_init=Tr1 #[C]
    Tr2_init=50. #[C]
    Tr3_init=50. #[C]
    # solver information
    tol = 1e-6
    max_iter = 50
    # scaling
    phbase = bar
    mbase = 1.
    Tbase = 100.
    phibase = MW
    # solution (for a standard load (sink) node for node 3)
    print('\nrunning load flow with outflow source and sink node, to determine solution')
    _,_,_,_,m_vec_outflow,p_vec_outflow,Ts_vec_outflow,Tr_vec_outflow,m_hl_vec_outflow,phi_hl_vec_outflow,Ts_hl_vec_outflow,Tr_hl_vec_outflow = run_load_flow(phbase,mbase,Tbase,phibase,p1=p1,Tr1=Tr1,To2=To2,To3=To3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2_init,p3_init=p3_init,Ts1_init=Ts1_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,formulation=formulation,node_set=2,slack_node='sink',sink_node='outflow',source_node='outflow',tol=tol,max_iter=max_iter)
    full_sol_outflow = np.concatenate([m_vec_outflow,p_vec_outflow,Ts_vec_outflow,Tr_vec_outflow,flatten(m_hl_vec_outflow),flatten(phi_hl_vec_outflow),flatten(Ts_hl_vec_outflow),flatten(Tr_hl_vec_outflow)])
    
    # When
    Tr2_sol = Tr_vec_outflow[1]
    dT2 = To2 - Tr2_sol
    Ts3_sol = Ts_vec_outflow[2] #solution for Ts3
    dT3 = Ts3_sol - To3 
    print('Solution: Tr2 = {}, dT2 = {}, Ts3 = {}, dT3 = {}'.format(Tr2_sol,dT2,Ts3_sol,dT3))
    print('\nrunning load flow with delta source and sink node')
    _,_,_,_,m_vec_delta,p_vec_delta,Ts_vec_delta,Tr_vec_delta,m_hl_vec_delta,phi_hl_vec_delta,Ts_hl_vec_delta,Tr_hl_vec_delta = run_load_flow(phbase,mbase,Tbase,phibase,p1=p1,Tr1=Tr1,dT2=dT2,dT3=dT3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2_init,p3_init=p3_init,Ts1_init=Ts1_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,formulation=formulation,node_set=2,slack_node='sink',source_node='delta',sink_node='delta',tol=tol,max_iter=max_iter)
    full_sol_delta = np.concatenate([m_vec_delta,p_vec_delta,Ts_vec_delta,Tr_vec_delta,flatten(m_hl_vec_delta),flatten(phi_hl_vec_delta),flatten(Ts_hl_vec_delta),flatten(Tr_hl_vec_delta)])
    
    print('sol delta = \n{}'.format(full_sol_delta))
    print('sol full = \n{}'.format(full_sol_outflow))
    assert np.allclose(full_sol_delta,full_sol_outflow)
    
def solvability(sep_figs=False):
    """Look at (physical) solvability of this network, by taking different BC's
    """
    # boundary conditions
    Ts1 = 100. #[C]
    Tr1 = 50. #[C]
    p1 = 9*bar #[Pa]
    To2_BC = Ts1*np.array([.9]) #[C]
    To3 = 50. #[C]
    phi = 1*MW #[W]
    phi2_BC = -phi*np.array([0.5,1,5]) #[W]
    phi3_BC = phi*np.array([1]) #[W]

    # initial guess
    m0_init_vec=np.array([1., -1.]) #[kg/s]
    m1_init=1 #[kg/s]
    p2_init=8*bar #[Pa]
    p3_init=6*bar #[Pa]
    Ts1_init=Ts1 #[C]
    Ts2_init=100. #[C]
    Ts3_init=100. #[C]
    Tr1_init=Tr1 #[C]
    Tr2_init=50. #[C]
    Tr3_init=50. #[C]
    dT3 = 40.
    
    slack_nodes =  ['general_Ts','general_Tr','sink','source','general']
    
    # solver information
    tol = 1e-6
    max_iter = 15

    # scaling
    phbase = bar
    mbase = 1.
    Tbase = 100.
    phibase = MW
    
    # make figure to plot convergence
    max_iters_used = 0
    fig_conv_heat_solv_sink, ax_conv_heat_solv_sink = plt.subplots(2, 2, sharex=True, num='Convergence plot heat network with different BCs, sink slack')
    ax_standard_sink = ax_conv_heat_solv_sink[0,0] 
    ax_halflink_sink = ax_conv_heat_solv_sink[1,0]
    ax_standard_general_Tr = ax_conv_heat_solv_sink[0,1]
    ax_halflink_general_Tr = ax_conv_heat_solv_sink[1,1]
    ax_halflink_sink.set_xlabel(r'Iteration $k$')
    ax_halflink_general_Tr.set_xlabel(r'Iteration $k$')
    ax_standard_sink.set_title(r'standard and sink slack')
    ax_halflink_sink.set_title(r'unknown half link flow and sink slack')
    
    ax_standard_general_Tr.set_title(r'standard and general slack (Tr)')
    ax_halflink_general_Tr.set_title(r'unknown half link flow and general slack (Tr)')
    
    fig_conv_heat_solv_source, ax_conv_heat_solv_source = plt.subplots(2, 2, sharex=True, num='Convergence plot heat network with different BCs, source slack')
    ax_standard_source = ax_conv_heat_solv_source[0,0] 
    ax_halflink_source = ax_conv_heat_solv_source[1,0]
    ax_standard_general_Ts = ax_conv_heat_solv_source[0,1]
    ax_halflink_general_Ts = ax_conv_heat_solv_source[1,1]
    ax_halflink_general_Ts.set_xlabel(r'Iteration $k$')
    ax_halflink_source.set_xlabel(r'Iteration $k$')
    ax_standard_general_Ts.set_title(r'standard and general slack (Ts)')
    ax_halflink_general_Ts.set_title(r'unknown half link flow and general slack (Ts)')
    ax_standard_source.set_title(r'standard and source slack')
    ax_halflink_source.set_title(r'unknown half link flow and source slack')
    
    fig_conv_heat_solv_general, ax_conv_heat_solv_general = plt.subplots(2, 1, sharex=True, num='Convergence plot heat network with different BCs, general slack')
    ax_standard_general = ax_conv_heat_solv_general[0] 
    ax_halflink_general = ax_conv_heat_solv_general[1]
    ax_halflink_general.set_xlabel(r'Iteration $k$')
    ax_halflink_general.set_xlabel(r'Iteration $k$')
    ax_standard_general.set_title(r'standard and general slack')
    ax_halflink_general.set_title(r'unknown half link flow and general slack')
    
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_ind = 0
    legend_handles = []
    # run load flow
    for phi2 in phi2_BC:
        for phi3 in phi3_BC:
            for To2 in To2_BC:
                dT2 = To2 - To3
                line_color = color_cycle[color_ind]
                label = r'$\varphi_2$={:.2f} MW, $\varphi_3$={:.2f} MW, $T^o_2$={:.2f} C'.format(phi2/MW,phi3/MW,To2)
                legend_handles.append(Line2D([0], [0], color=line_color, label=label))
                color_ind += 1
                for m0_init in m0_init_vec:
                    for slack_node in slack_nodes:
                        if sep_figs: # make seperate figures for convergence instead of subplots. The node types of node 2 and 3 are kept in the same figure, the rest are seperate figures. The different formulations are kept as subplots.
                            sep_fig, axes_sep_fig = plt.subplots(2, 1, sharex=True, num=r'Convergence node 1 = {}, $\varphi_2$={:.2f} MW, $\varphi_3$={:.2f} MW, $T^o_2$={:.2f} C, $m_{{12}}$={:.1f}'.format(slack_node,phi2/MW,phi3/MW,To2,m0_init))
                            ax_sep_fig_standard = axes_sep_fig[0] 
                            ax_sep_fig_standard.set_title(r'standard formulation')
                            ax_sep_fig_halflink = axes_sep_fig[1]
                            ax_sep_fig_halflink.set_title(r'unknown half link formulation')
                            ax_sep_fig_halflink.set_xlabel(r'Iteration $k$')
                            for ax_sep_fig in axes_sep_fig:
                                ax_sep_fig.set_ylabel(r'Error $||D_F F(x^k)||_2$')
                                ax_sep_fig.grid(which='major',color='k', linestyle='--', alpha=.2)
                                ax_sep_fig.grid(which='minor',color='k', linestyle=':', alpha=.05)
                        for formulation in ['standard','half_link_flow']:
                            for sink_node in ['outflow','delta']:
                                for source_node in ['outflow','delta']:
                                    # load flow
                                    print('\nRunning load flow with formulation {}, and {} slack node'.format(formulation,slack_node))
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                                        heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = run_load_flow(phbase,mbase,Tbase,phibase,Ts1=Ts1,p1=p1,Tr1=Tr1,To2=To2,To3=To3,phi2=phi2,phi3=phi3,dT2=dT2,dT3=dT3,m0_init=m0_init,m1_init=m1_init,p2_init=p2_init,p3_init=p3_init,Ts1_init=Ts1_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,formulation=formulation,node_set=2,slack_node=slack_node,source_node=source_node,sink_node=sink_node,tol=tol,max_iter=max_iter)
                                    # plot convergence
                                    if formulation == 'standard':
                                        if slack_node == 'general_Ts':
                                            ax = ax_standard_general_Ts
                                        elif slack_node == 'general_Tr':
                                            ax = ax_standard_general_Tr
                                        elif slack_node == 'source':
                                            ax = ax_standard_source
                                        elif slack_node == 'general':
                                            ax = ax_standard_general
                                        else:
                                            ax = ax_standard_sink
                                    else:
                                        if slack_node == 'general_Ts':
                                            ax = ax_halflink_general_Ts
                                        elif slack_node == 'general_Tr':
                                            ax = ax_halflink_general_Tr
                                        elif slack_node == 'source':
                                            ax = ax_halflink_source
                                        elif slack_node == 'general':
                                            ax = ax_halflink_general
                                        else:
                                            ax = ax_halflink_sink
                                    label += r', $m_{{12}}$={:.1f}'.format(m0_init)
                                    if m0_init > 0:
                                        line_style = '-'
                                    else:
                                        line_style = '--'
                                    if sink_node == 'outflow' and source_node == 'outflow':
                                        marker_style = '.'
                                        label += r', $T^o_2$, $T^o_3$ known'
                                    elif sink_node == 'delta' and source_node == 'delta':
                                        marker_style ='x'
                                        label += r', $\Delta T_2$, $\Delta T_3$ known'
                                    elif sink_node == 'delta':
                                        marker_style ='*'
                                        label += r', $T^o_2$, \Delta T_3$ known'
                                    elif source_node == 'delta':
                                        marker_style ='d'
                                        label += r', $\Delta T_2$, $T^o_3$ known'
                                    ax.semilogy(err_vec,marker=marker_style,ls=line_style,color=line_color,label=label)
                                    max_iters_used = max(max_iters_used,iters)
                                    if sep_figs:
                                        if formulation == 'standard':
                                            ax_sep_fig_standard.semilogy(err_vec,marker=marker_style,ls=line_style,color=line_color,label=label)
                                        else:
                                            ax_sep_fig_halflink.semilogy(err_vec,marker=marker_style,ls=line_style,color=line_color,label=label)
                        if sep_figs:
                            for ax_sep_fig in axes_sep_fig:
                                ax_sep_fig.semilogy([0,max_iters_used],[tol,tol],'r:',label='tolerance')
    
    marker_size = 10
    legend_handles += [Line2D([0], [0], marker='.',color='w',markerfacecolor='k', markersize=marker_size, label=r'$T^o_2$, $T^o_3$ known'),
                           Line2D([0], [0], marker='x',color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'$\Delta T_2$, $\Delta T_3$ known'),
                           Line2D([0], [0], marker='*',color='w',markerfacecolor='k', markersize=marker_size, label=r'$T^o_2$, $\Delta T_3$ known'),
                           Line2D([0], [0], marker='d',color='w',markerfacecolor='k', markersize=marker_size, label=r'$\Delta T_2$, $T^o_3$ known')]
    for m0_init in m0_init_vec:
        if m0_init > 0:
            line_style = '-'
        else:
            line_style = '--'
        legend_handles.append(Line2D([0], [0], color='k',ls=line_style, label=r'$m_{{12}}$={:.1f}'.format(m0_init)))
    
    for ax_rows in np.concatenate((ax_conv_heat_solv_sink,ax_conv_heat_solv_source)):
        for ax in ax_rows:
            ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
            ax.semilogy([0,max_iters_used+1],[tol,tol],'r:',label='tolerance')
            if not sep_figs: # legend is plotted in figure
                box_ax = ax.get_position()
                ax.set_position([box_ax.x0, box_ax.y0, box_ax.width, box_ax.height * 0.8]) # Shrink current axis by 20%
            ax.grid(which='major',color='k', linestyle='--', alpha=.2)
            ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    if sep_figs: 
        fig_legend = plt.figure('Legend')
        ax_legend = fig_legend.gca()
        ax_legend.axis('off')
        fig_legend.patch.set_visible(False)
        ax_legend.legend(handles=legend_handles,loc='center')
    else: # plots legend in figures
        ax_standard_sink.legend(handles=legend_handles,ncol=3,loc='upper left', bbox_to_anchor=(0., 1.5)) # Put a legend above the upper right (This assumes the other two plots have the same legend!!! This seems to be the case, but I'm not entirely sure)
        ax_standard_source.legend(handles=legend_handles,ncol=3,loc='upper left', bbox_to_anchor=(0., 1.5))
    
if __name__ == '__main__':
    # parse the command line
    args = command_line_input.parse_args()
    
    # boundary conditions
    Ts1 = 100. #[C]
    p1 = 9*bar
    To2 = 1.1*Ts1
    To3 = 50. #[C]
    phi2 = -1*MW #[W]
    phi3 = 1.5*-phi2 

    # initial guess
    m0_init=1 #[kg/s]
    m1_init=1 #[kg/s]
    p2_init=8*bar #[Pa]
    p3_init=6*bar #[Pa]
    Ts2_init=100. #[C]
    Ts3_init=100. #[C]
    Tr1_init=50. #[C]
    Tr2_init=50. #[C]
    Tr3_init=50. #[C]
    
    # solver information
    tol = 1e-6
    max_iter = 30
    formulation='standard'

    # scaling
    phbase = bar
    mbase = 1.
    Tbase = 100.
    phibase = MW

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        heat_net1,x_sol1,iters1,err_vec1,m_vec1,p_vec1,Ts_vec1,Tr_vec1,m_hl_vec1,phi_hl_vec1,Ts_hl_vec1,Tr_hl_vec1 = run_load_flow(phbase,mbase,Tbase,phibase,Ts1=Ts1,p1=p1,To2=To2,To3=To3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2_init,p3_init=p3_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,formulation=formulation,node_set=1,tol=tol,max_iter=max_iter)
        heat_net1_gen,x_sol1_gen,iters1_gen,err_vec1_gen,m_vec1_gen,p_vec1_gen,Ts_vec1_gen,Tr_vec1_gen,m_hl_vec1_gen,phi_hl_vec1_gen,Ts_hl_vec1_gen,Tr_hl_vec1_gen = run_load_flow(phbase,mbase,Tbase,phibase,Ts1=Ts1,p1=p1,To2=To2,To3=To3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2_init,p3_init=p3_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,formulation=formulation,node_set=1,slack_node='general_Ts',tol=tol,max_iter=max_iter)
        heat_net2,x_sol2,iters2,err_vec2,m_vec2,p_vec2,Ts_vec2,Tr_vec2,m_hl_vec2,phi_hl_vec2,Ts_hl_vec2,Tr_hl_vec2 = run_load_flow(phbase,mbase,Tbase,phibase,Ts1=Ts1,p1=p1,To2=To2,To3=To3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2_init,p3_init=p3_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,formulation=formulation,node_set=2,tol=tol,max_iter=max_iter)
        heat_net2_gen,x_sol2_gen,iters2_gen,err_vec2_gen,m_vec2_gen,p_vec2_gen,Ts_vec2_gen,Tr_vec2_gen,m_hl_vec2_gen,phi_hl_vec2_gen,Ts_hl_vec2_gen,Tr_hl_vec2_gen = run_load_flow(phbase,mbase,Tbase,phibase,Ts1=Ts1,p1=p1,To2=To2,To3=To3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2_init,p3_init=p3_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,formulation=formulation,node_set=2,slack_node='general_Ts',tol=tol,max_iter=max_iter)
        
    # plot network solutions
    fig_sol1 = plt.figure('Network solution heat network, node set 1')
    ax_sol1 = fig_sol1.gca()
    heat_net1.draw_network_value(ax_sol1)
    plt.axis('equal')
    plt.axis('off')
    
    fig_sol2 = plt.figure('Network solution heat network, node set 2')
    ax_sol2 = fig_sol2.gca()
    heat_net2.draw_network_value(ax_sol2)
    plt.axis('equal')
    plt.axis('off')
    
    # print solution
    print('\nSolution for node set 1')
    for node in heat_net1.get_nodes():
        print('node {} with p = {:.3e} bar, Ts = {:.3f} C, Tr = {:.3f} C'.format(node.name,node.p/bar,node.Ts,node.Tr))
        for hl in node.get_half_links():
            print('with half link {} of type {}, with m = {:.3f} kg/s, dphi = {:.3e} W, Ts = {:.3f} C, Tr = {:.3f} C'.format(hl.name,hl.link_type,hl.m,hl.dphi,hl.Ts,hl.Tr))
    for link in heat_net1.get_links():
        print('link {} from node {} to node {}, with m = {:.3f} kg/s'.format(link.name,link.start_node.name,link.end_node.name,link.m))
        
    print('\nSolution for node set 1, and general slack node')
    for node in heat_net1_gen.get_nodes():
        print('node {} with p = {:.3e} bar, Ts = {:.3f} C, Tr = {:.3f} C'.format(node.name,node.p/bar,node.Ts,node.Tr))
        for hl in node.get_half_links():
            print('with half link {} of type {}, with m = {:.3f} kg/s, dphi = {:.3e} W, Ts = {:.3f} C, Tr = {:.3f} C'.format(hl.name,hl.link_type,hl.m,hl.dphi,hl.Ts,hl.Tr))
    for link in heat_net1_gen.get_links():
        print('link {} from node {} to node {}, with m = {:.3f} kg/s'.format(link.name,link.start_node.name,link.end_node.name,link.m))
    
    print('\nSolution for node set 2')
    for node in heat_net2.get_nodes():
        print('node {} with p = {:.3e} bar, Ts = {:.3f} C, Tr = {:.3f} C'.format(node.name,node.p/bar,node.Ts,node.Tr))
        for hl in node.get_half_links():
            print('with half link {} of type {}, with m = {:.3f} kg/s, dphi = {:.3e} W, Ts = {:.3f} C, Tr = {:.3f} C'.format(hl.name,hl.link_type,hl.m,hl.dphi,hl.Ts,hl.Tr))
    for link in heat_net2.get_links():
        print('link {} from node {} to node {}, with m = {:.3f} kg/s'.format(link.name,link.start_node.name,link.end_node.name,link.m))
        
    print('\nSolution for node set 2, and general slack node')
    for node in heat_net2_gen.get_nodes():
        print('node {} with p = {:.3e} bar, Ts = {:.3f} C, Tr = {:.3f} C'.format(node.name,node.p/bar,node.Ts,node.Tr))
        for hl in node.get_half_links():
            print('with half link {} of type {}, with m = {:.3f} kg/s, dphi = {:.3e} W, Ts = {:.3f} C, Tr = {:.3f} C'.format(hl.name,hl.link_type,hl.m,hl.dphi,hl.Ts,hl.Tr))
    for link in heat_net2_gen.get_links():
        print('link {} from node {} to node {}, with m = {:.3f} kg/s'.format(link.name,link.start_node.name,link.end_node.name,link.m))
    
    # plot convergence
    fig_conv_heat = plt.figure('Convergence plot heat network')
    ax_conv_heat = fig_conv_heat.gca()
    ax_conv_heat.set_title(r'$T^s_1$={:.2f} C, $\varphi_2$={:.2f} MW, $T^o_2$={:.2f} C, $\varphi_3$={:.2f} MW'.format(Ts1,phi2/MW,To2,phi3/MW))
    plt.xlabel(r'Iteration $k$')
    plt.ylabel(r'Error $||D_F F(x^k)||_2$')
    max_iter_used = np.max([iters1,iters1_gen,iters2,iters2_gen])
    ax_conv_heat.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax_conv_heat.semilogy(err_vec1,marker='.',ls='-',color='tab:blue',label='node set 1')
    ax_conv_heat.semilogy(err_vec1_gen,marker='*',ls='-',color='tab:green',label='node set 1, general slack')
    ax_conv_heat.semilogy(err_vec2,marker='.',ls='-',color='tab:orange',label='node set 2')
    ax_conv_heat.semilogy(err_vec2_gen,marker='*',ls='-',color='tab:red',label='node set 2, general slack')
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)

    # compare convergence / solvability for different boundary conditions
    solvability(sep_figs=args.sep_figs)
    
    plt.show()
