"""Network consisting of one quarter, based on the N_1Q data."""

from meslf.networks.read_write_network import from_pd_dataframes
from meslf.networks.gas_network import GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNode, HeatLink, HeatHalfLink
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
import pandas as pd
import numpy as np
import scipy.sparse as sps
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
from examples.network_data.N_1Q import make_MES_1Q
from examples import GN_1Q, EN_1Q, HN_1Q
import argparse # To get arguments (as list) from command line
import warnings
from meslf.utils.constants import MW, kW, hour, mbar, km

from meslf.load_flow.system_of_equations import NonLinearSystem, NonLinearSystemGas, NonLinearSystemElectrical, NonLinearSystemHeat, NonLinearSystemHeterogeneous

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning 

command_line_input = argparse.ArgumentParser()
command_line_input.add_argument(
    "--n", # number of loads, for each street
    type=int,
    default = 0, 
    )
command_line_input.add_argument(
    "--m", # number of junctions with two loads, for each street
    type=int,
    default = 0, 
    )
command_line_input.add_argument(
    "--nS", # number of street networks
    type=int,
    default = 0, 
    )
command_line_input.add_argument(
    "--pg_low", # Lowest value of fraction of reference pressure that is used in the initial linear pressure profile
    type=float,
    default = .9,
    )
command_line_input.add_argument(
    "--pg_high", # Highest value of fraction of reference pressure that is used in the initial linear pressure profile
    type=float,
    default = .95, 
    )
command_line_input.add_argument(
    "--T_low", # Lowest value of fraction of reference temperature that is used in the initial linear (supply) temperature profile
    type=float,
    default = .9, 
    )
command_line_input.add_argument(
    "--T_high", # Highest value of fraction of reference temperature that is used in the initial linear (supply) temperature profile
    type=float,
    default = .95,
    )
command_line_input.add_argument(
    "--max_nodes", # Maximum number of total nodes for which the topology is plotted, etc.
    type=int,
    default = 50,
    )
command_line_input.add_argument(
    "--fb", 
    type=bool,
    default = False,
    )
command_line_input.add_argument(
    "--lin_pipe", 
    type=bool,
    default = False,
    )
command_line_input.add_argument(
    "--use_un_hl", 
    type=bool,
    default = False,
    )
command_line_input.add_argument(
    "--scn", 
    type=bool,
    default = False,
    )
command_line_input.add_argument(
    "--save_fig", 
    type=bool,
    default = False,
    )
command_line_input.add_argument(
    "--print_sol", 
    type=bool,
    default = False,
    )
command_line_input.add_argument(
    "--show_plots", 
    type=bool,
    default = False, 
    )
command_line_input.add_argument(
    "--comp_conv", 
    type=bool,
    default = False, 
    )

def remake_network(n,m,nS):
    """Recreate the pd dataframes for the single carrier networks.

    Parameters
    ----------
    n : int
        number of loads, for each street
    m : int
        number of junctions with two loads, for each street
    nS : int
        number of street networks
    """
    if n and nS: # m=0 is allowed
        print('Recreating the single-carrier networks with n={}, m={}, and nS={}'.format(n,m,nS))
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path_to_data = os.path.join(dir_path,'network_data','N_1Q')
        make_MES_1Q.create_single_carrier_networks(n,m,nS,path_to_data)
    else:
        print('The existing data for the single-carrier networks is used.')

def create_network():
    """Create a test heterogeneous network with a single CHP as coupling node

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
    # create network from data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','N_1Q')

    gas_nodes = pd.read_pickle(os.path.join(path_to_data, 'GN_1Q_nodes.pkl'))
    gas_links = pd.read_pickle(os.path.join(path_to_data, 'GN_1Q_links.pkl'))
    gas_halflinks = pd.read_pickle(os.path.join(path_to_data, 'GN_1Q_halflinks.pkl'))
    gas_net = from_pd_dataframes(gas_nodes,gas_links,gas_halflinks)

    elec_nodes = pd.read_pickle(os.path.join(path_to_data, 'EN_1Q_nodes.pkl'))
    elec_links = pd.read_pickle(os.path.join(path_to_data, 'EN_1Q_links.pkl'))
    elec_halflinks = pd.read_pickle(os.path.join(path_to_data, 'EN_1Q_halflinks.pkl'))
    elec_net = from_pd_dataframes(elec_nodes,elec_links,elec_halflinks)

    heat_nodes = pd.read_pickle(os.path.join(path_to_data, 'HN_1Q_nodes.pkl'))
    heat_links = pd.read_pickle(os.path.join(path_to_data, 'HN_1Q_links.pkl'))
    heat_halflinks = pd.read_pickle(os.path.join(path_to_data, 'HN_1Q_halflinks.pkl'))
    heat_net = from_pd_dataframes(heat_nodes,heat_links,heat_halflinks)
    Ta = heat_net.links[0].link_params.get('Ta')
    heat_net.Ta = Ta

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
    eta_CHP_pu = 0.88
    eta_b = 1/kW # base value for eta should be equal to phibase/Pb. So, since heat and gas are in S.I., whereas power is in p.u. (with Sbase = 1kW), this scaling is needed.
    eta_CHP = np.array([eta_CHP_pu*eta_b, eta_CHP_pu])
    GHV = 40.611*293.1e-6 #[MWh/m^3]
    GHV *= MW*hour/rhon_g #[J/kg]
    cn0 = HeterogeneousNode('CHP',node_type=0,unit_type='geh_CHP',unit_params={'eta':eta_CHP,'GHV':GHV}) # To unknown
    cn0.y = (gas_source.y + elec_source.y + heat_source.y)/3
    cn0.x = (gas_source.x + elec_source.x + heat_source.x)/3-1

    glc = GasLink('gl_c',gas_source,cn0)
    elc = ElectricalLink('el_c',cn0,elec_source)
    hlc = HeatLink('hl_c',cn0,heat_source,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown

    gas_net.add_link(glc)
    elec_net.add_link(elc)
    heat_net.add_link(hlc)

    # change node types of homogeneous nodes which are connected to the heterogeneous coupling node
    elec_source.node_type = 5 # PQVdelta node
    for hl in elec_source.get_half_links():
        hl.P = 0
        hl.Q = 0
    heat_source.node_type = 7 # reference temperature (junction) node
    for hl in heat_source.get_half_links():
        heat_net.remove_half_link(hl)
        heat_source.remove_half_link(hl)

    het_net = HeterogeneousNetwork('MES_1Q')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    het_net.add_node(cn0)

    return het_net, gas_net, elec_net, heat_net, water, gas

def initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,pg_perc_high,pg_perc_low,T_perc_high,T_perc_low,scale_var=None,scale_var_params=None):
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
    #print('rhon gas = {}'.format(rhon_g))
    Ta = heat_net.Ta

    xg_entries = gas_net.get_x_entries(formulation = formulation['gas'])
    xe_entries, unknown_delta_nodes, unknown_V_nodes = elec_net.get_x_entries(formulation = formulation['elec'])
    x_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks = heat_net.get_x_entries(formulation = formulation['heat'])
    gp_ref = gas_net.nodes[0].p
    #print('reference pressure gas: {} Pa'.format(gp_ref))
    unknown_pg_nodes = []
    unknown_q_links = []
    for el in xg_entries:
        if isinstance(el,GasNode):
            unknown_pg_nodes.append(el)
        elif isinstance(el,GasLink):
            unknown_q_links.append(el)
    gp_init = gp_ref*np.linspace(pg_perc_high,pg_perc_low,len(unknown_pg_nodes))
    q_init = 1e-5*np.ones(len(unknown_q_links))
    delta_init = -0.2*np.ones(len(unknown_delta_nodes))
    V_init = 1.*np.linspace(0.99,0.95,len(unknown_V_nodes)) # initial voltage amplitudes 1% - 5% from reference value
    m_init = 1e-2*np.ones(len(unknown_m_links))
    m_hl_init = 1e-2*np.ones(len(unknown_m_halflinks))
    hp_ref = heat_net.nodes[0].p
    Ts_ref = heat_net.nodes[0].Ts
    #print('reference pressure heat: {} Pa'.format(hp_ref))
    hp_init = hp_ref*np.linspace(pg_perc_high,pg_perc_low,len(unknown_p_nodes)) 
    Ts_init = Ts_ref*np.linspace(T_perc_high,T_perc_low,len(unknown_Ts_nodes))
    Tr_init = 50.*np.linspace(T_perc_low,T_perc_high,len(unknown_Tr_nodes))
    qc_init = np.array([1e-5])
    Pc_init = np.array([1.5])
    Qc_init = np.array([1.5])
    Sc_init = np.concatenate((Pc_init,Qc_init))
    mc_init = np.array([1e-2])
    phic_init = np.array([1.])*kW #[W]
    Toc_init = np.array([130.])

    if formulation['gas'] == 'nodal':
        xg_init = gp_init
    else:
        xg_init = np.concatenate((q_init,gp_init))
    xe_init = np.concatenate((delta_init,V_init))
    if formulation['heat'] == 'half_link_flow':
        xh_init = np.concatenate((m_init,m_hl_init,hp_init,Ts_init,Tr_init))
    else:
        xh_init = np.concatenate((m_init,hp_init,Ts_init,Tr_init))
    x_init = np.concatenate((xg_init,xe_init,xh_init,qc_init,Sc_init,mc_init,phic_init,Toc_init)) # unscaled
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def solve_system(network,tol,max_iter,h,x0,formulation,scale_var=None,scale_var_params=None,D_F=np.array([]),D_x=np.array([]),solver = 'NR'):
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
    x_sol : np array
        solution vector, using analytical Jacobian
    iters : int
        total number of iterations, using analytical Jacobian
    err_vec : np array
        vector with the error of NR for every iteration, using analytical Jacobian
    """
    network.update(x0,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = network.solve_network(tol,max_iter,h=h,solver=solver,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x)

    return x_sol,iters,err_vec

def compare_conv_solvers(pg_perc_low,pg_perc_high,T_perc_low,T_perc_high,max_number_of_nodes,use_fb,use_linear_pipe,use_un_hl,save_fig,use_scn):
    """
    Parameters
    ----------
    p_perc_low : float
        Lowest value of fraction of reference pressure that is used in the initial linear pressure profile.
    p_perc_high : float
        Highest value of fraction of reference pressure that is used in the initial linear pressure profile.
    max_number_of_nodes : int
        Maximum number of total nodes for which FD is used, and for which the topology is plotted, etc.
    use_fb : bool
        If True, fb is also used in the gas network to solve the network (and not only fa)
    use_linear_pipe : bool
        If True, a linear pipe model is used for all the pipes in the gas network.
    use_un_hl : bool
        If true, the unknown half link formulation in the heat network is also used to solve the system
    save_fig : bool
        If True, the convergence plot is saved as a pgf. 
    use_scn: bool
        If True, the single-carrier networks are solved, and their solution is used as intial guess for the MES
    """
    het_net, gas_net, elec_net, heat_net, water, gas = create_network()
    number_of_nodes = len(het_net.nodes)
    formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None}
    Ta = heat_net.Ta

    if use_linear_pipe:
        # change all the gas pipes to a linear pipe model
        for link in gas_net.get_links():
            if 'pipe' in link.link_type:
                if np.isclose(link.link_params['L'],1*km):
                    link_params_linear = {'carrier':link.link_params['carrier'], 'alpha':0.005}
                elif np.isclose(link.link_params['L'],100):
                    link_params_linear = {'carrier':link.link_params['carrier'], 'alpha':0.0025}
                elif np.isclose(link.link_params['L'],10):
                    link_params_linear = {'carrier':link.link_params['carrier'], 'alpha':0.025}
                link.set_type('pipe_linear',link_params_linear)
    # scaling
    gas_source = gas_net.nodes[0]
    pg_ref = gas_source.p
    heat_source = heat_net.nodes[0]
    ph_ref = heat_source.p
    mbase_pu = 1e-2
    pbase_pu = pg_ref
    Tbase_pu = 100.
    Sbase_pu = 1. #since P and Q are in p.u.
    phibase_pu = Sbase_pu*kW
    Vbase_pu = 1.
    deltabase_pu = 1.
    scale_var = 'per_unit'
    scale_var_params = {'qbase':mbase_pu,'pbase':pbase_pu,'phibase':phibase_pu,'Tbase':Tbase_pu,'Vbase':Vbase_pu,'deltabase':deltabase_pu,'Sbase':Sbase_pu,'Ebase':phibase_pu}
    # base values used for scaling in solver
    qbase = 1e-4
    pgbase = pg_ref
    mbase = 50.
    phbase = ph_ref
    Tbase = 100.
    Sbase = 1.
    phibase = Sbase*kW
    Egbase = phibase
    Vbase = 1.
    deltabase = 1.

    # initalize network, unscaled
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,pg_perc_high,pg_perc_low,T_perc_high,T_perc_low)
    # initalize network, per unit
    het_net.reset_network(x0,formulation=formulation)
    x0_pu = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,pg_perc_high,pg_perc_low,T_perc_high,T_perc_low,scale_var=scale_var,scale_var_params=scale_var_params)

    if number_of_nodes<max_number_of_nodes:
        nlsys_s = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        F0_pu = nlsys_s.F(x0_pu)
        J0_pu = nlsys_s.J(x0_pu)
        print('length of x0 = {}, and length of F(x0) = {}'.format(len(x0_pu),len(F0_pu)))
        print('scaled: |J(x0)|={}'.format(np.linalg.det(J0_pu.todense())))
    
        J0_pu_FD = nlsys_s.J_FD(x0_pu,1e-6)

        J_pu_diff = J0_pu-J0_pu_FD
        fig_J = plt.figure('Jacobian difference MES')
        plt.imshow(J_pu_diff)
        ax_J = plt.gca()
        nlsys_s.plot_J_overlay(ax_J)
        plt.colorbar()

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
                    print('m = {}, phi = {}, To = {}'.format(hl.m,hl.phi,hl.To))

        for link in het_net.get_links():
            if 'Ta' in link.link_params.keys():
                print('link {} with link type {} and Ta = {}'.format(link.name,link.link_type,link.link_params['Ta']))
            else:
                print('link {} with link type {}'.format(link.name,link.link_type))
    # spy plot of Jacobian
    if number_of_nodes<max_number_of_nodes:
        fig_J = nlsys_s.spy_plot_J(x0_pu)

    # compare convergence
    h = 1e-6
    tol = 1e-6
    max_iter = 50

    # solve when everything is specified in S.I., unscaled
    print('\nSolving heterogeneous network, using {} formulation in gas network, unscaled'.format(formulation['gas']))
    het_net.reset_network(x0,formulation=formulation)
    x_sol_SI,iters_SI,err_vec_SI = solve_system(het_net,tol,max_iter,h,x0,formulation)
    if number_of_nodes<max_number_of_nodes:
        # solve when everything is specified in S.I., unscaled, using FD
        print('\nSolving heterogeneous network, using {} formulation in gas network, unscaled, finite-difference Jacobian'.format(formulation['gas']))
        het_net.reset_network(x0,formulation=formulation)
        x_sol_SI_FD,iters_SI_FD,err_vec_SI_FD = solve_system(het_net,tol,max_iter,h,x0,formulation,solver='NR_FD')
    # solve when everything is specified in per unit
    print('\nSolving heterogeneous network, using {} formulation in gas network, per unit'.format(formulation['gas']))
    het_net.reset_network(x0,formulation=formulation)
    x_sol_pu,iters_pu,err_vec_pu = solve_system(het_net,tol,max_iter,h,x0_pu,formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    # solve when everything is specified in S.I., using scaling in solver
    print('\nSolving heterogeneous network, using {} formulation in gas network, scaling in solver'.format(formulation['gas']))
    het_net.reset_network(x0,formulation=formulation)
    x_sol_scaled,iters_scaled,err_vec_scaled = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
    if number_of_nodes<max_number_of_nodes:
        # solve when everything is specified in S.I., using scaling in solver, using FD
        print('\nSolving heterogeneous network, using {} formulation in gas network, scaling in solver, finite-difference Jacobian'.format(formulation['gas']))
        het_net.reset_network(x0,formulation=formulation)
        x_sol_scaled_FD,iters_scaled_FD,err_vec_scaled_FD = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase},solver='NR_FD')
        # solve when everything is specified in per unit, using FD
        print('\nSolving heterogeneous network, using {} formulation in gas network, per unit, finite-difference Jacobian'.format(formulation['gas']))
        het_net.reset_network(x0,formulation=formulation)
        x_sol_pu_FD,iters_pu_FD,err_vec_pu_FD = solve_system(het_net,tol,max_iter,h,x0_pu,formulation,scale_var=scale_var,scale_var_params=scale_var_params,solver='NR_FD')
    if use_fb: # use fb in gas, and 'standard' formulation in heat
        # set fb as link equation on all links
        for link in gas_net.get_links():
            link.set_type(link.link_type,link.link_params,link_eq_form='dp_of_q')
        formulation_fb = formulation.copy()
        if formulation_fb.get('gas') == 'nodal':
            het_net.reset_network(x0,formulation=formulation)
            formulation_fb['gas'] = 'full' #nodal is not defined when using fb
            # initalize network, unscaled
            x0_fb = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation_fb,pg_perc_high,pg_perc_low,T_perc_high,T_perc_low)
            # initalize network, per unit
            x0_pu_fb  = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation_fb ,pg_perc_high,pg_perc_low,T_perc_high,T_perc_low,scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            x0_fb = x0.copy()
            x0_pu_fb = x0_pu.copy()
        # solve when everything is specified in S.I., unscaled
        print('\nSolving heterogeneous network, using {} formulation in gas network, unscaled, fb in gas'.format(formulation_fb['gas']))
        het_net.reset_network(x0_fb,formulation=formulation_fb)
        x_sol_SI_fb,iters_SI_fb,err_vec_SI_fb = solve_system(het_net,tol,max_iter,h,x0_fb,formulation_fb)
        # solve when everything is specified in per unit
        print('\nSolving heterogeneous network, using {} formulation in gas network, per unit, fb in gas'.format(formulation_fb['gas']))
        het_net.reset_network(x0_fb,formulation=formulation_fb)
        x_sol_pu_fb,iters_pu_fb,err_vec_pu_fb = solve_system(het_net,tol,max_iter,h,x0_pu_fb,formulation_fb,scale_var=scale_var,scale_var_params=scale_var_params)
        # solve when everything is specified in S.I., using scaling in solver
        print('\nSolving heterogeneous network, using {} formulation in gas network, scaling in solver, fb in gas'.format(formulation_fb['gas']))
        het_net.reset_network(x0_fb,formulation=formulation_fb)
        x_sol_scaled_fb,iters_scaled_fb,err_vec_scaled_fb = solve_system(het_net,tol,max_iter,h,x0_fb,formulation_fb,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
        # set fa (back) as link equation on all links
        for link in gas_net.get_links():
            link.set_type(link.link_type,link.link_params,link_eq_form='q_of_dp')
    if use_un_hl: # use fa in gas, and 'half_link_flow' formulation in heat
        formulation_hl = formulation.copy()
        formulation_hl['heat'] = 'half_link_flow'
        # initalize network, unscaled
        x0_hl = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation_hl,pg_perc_high,pg_perc_low,T_perc_high,T_perc_low)
        # initalize network, per unit
        het_net.reset_network(x0_hl,formulation=formulation_hl)
        x0_pu_hl = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation_hl,pg_perc_high,pg_perc_low,T_perc_high,T_perc_low,scale_var=scale_var,scale_var_params=scale_var_params)
        # solve when everything is specified in S.I., unscaled
        print('\nSolving heterogeneous network, using {} formulation in gas network, unscaled, hl in heat'.format(formulation_hl['gas']))
        het_net.reset_network(x0_hl,formulation=formulation_hl)
        x_sol_SI_hl,iters_SI_hl,err_vec_SI_hl = solve_system(het_net,tol,max_iter,h,x0_hl,formulation_hl)
        # solve when everything is specified in per unit
        print('\nSolving heterogeneous network, using {} formulation in gas network, per unit, hl in heat'.format(formulation_hl['gas']))
        het_net.reset_network(x0_hl,formulation=formulation_hl)
        x_sol_pu_hl,iters_pu_hl,err_vec_pu_hl = solve_system(het_net,tol,max_iter,h,x0_pu_hl,formulation_hl,scale_var=scale_var,scale_var_params=scale_var_params)
        # solve when everything is specified in S.I., using scaling in solver
        print('\nSolving heterogeneous network, using {} formulation in gas network, scaling in solver, hl in heat'.format(formulation_hl['gas']))
        het_net.reset_network(x0_hl,formulation=formulation_hl)
        x_sol_scaled_hl,iters_scaled_hl,err_vec_scaled_hl = solve_system(het_net,tol,max_iter,h,x0_hl,formulation_hl,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
    if use_fb and use_un_hl: # use fb in gas, and 'half_link_flow' formulation in heat
        # set fb as link equation on all links
        for link in gas_net.get_links():
            link.set_type(link.link_type,link.link_params,link_eq_form='dp_of_q')
        formulation_fb_hl = formulation_fb.copy()
        formulation_fb_hl['heat'] = 'half_link_flow'
        # initalize network, unscaled
        x0_fb_hl = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation_fb_hl,pg_perc_high,pg_perc_low,T_perc_high,T_perc_low)
        # initalize network, per unit
        het_net.reset_network(x0_fb_hl,formulation=formulation_fb_hl)
        x0_pu_fb_hl = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation_fb_hl,pg_perc_high,pg_perc_low,T_perc_high,T_perc_low,scale_var=scale_var,scale_var_params=scale_var_params)
        # solve when everything is specified in S.I., unscaled
        print('\nSolving heterogeneous network, using {} formulation in gas network, unscaled, hl in heat'.format(formulation_fb_hl['gas']))
        het_net.reset_network(x0_fb_hl,formulation=formulation_fb_hl)
        x_sol_SI_fb_hl,iters_SI_fb_hl,err_vec_SI_fb_hl = solve_system(het_net,tol,max_iter,h,x0_fb_hl,formulation_fb_hl)
        # solve when everything is specified in per unit
        print('\nSolving heterogeneous network, using {} formulation in gas network, per unit, hl in heat'.format(formulation_fb_hl['gas']))
        het_net.reset_network(x0_fb_hl,formulation=formulation_fb_hl)
        x_sol_pu_fb_hl,iters_pu_fb_hl,err_vec_pu_fb_hl = solve_system(het_net,tol,max_iter,h,x0_pu_fb_hl,formulation_fb_hl,scale_var=scale_var,scale_var_params=scale_var_params)
        # solve when everything is specified in S.I., using scaling in solver
        print('\nSolving heterogeneous network, using {} formulation in gas network, scaling in solver, hl in heat'.format(formulation_fb_hl['gas']))
        het_net.reset_network(x0_fb_hl,formulation=formulation_fb_hl)
        x_sol_scaled_fb_hl,iters_scaled_fb_hl,err_vec_scaled_fb_hl = solve_system(het_net,tol,max_iter,h,x0_fb_hl,formulation_fb_hl,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
        for link in gas_net.get_links():
            link.set_type(link.link_type,link.link_params,link_eq_form='q_of_dp')
    if use_scn:
        # solve the network, using the solutions of the single-carrier networks as initial guess
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print('\nSolving heterogeneous network, using solution to single-carrier networks as initial guess. Setting initial condition....')
        xg_sol = GN_1Q.compare_conv_solvers(dir_path,pg_perc_low,pg_perc_high,1,use_fb,False,False) # uses 'full' formulation
        if (not use_fb) and (formulation['gas'] == 'nodal'):
            print('Only use pressure part of gas solution')
            xg_sol = xg_sol[len(gas_net.links)-1:]
        xe_sol = EN_1Q.compare_conv_solvers(dir_path,1,False)
        xh_sol = HN_1Q.compare_conv_solvers(dir_path,pg_perc_low,pg_perc_high,T_perc_low,T_perc_high,1,False)
        xc_init_scn = x0[len(x0)-6:]
        x_init_scn = np.concatenate((xg_sol,xe_sol,xh_sol,xc_init_scn))
        print('\nSolving heterogeneous network....')
        if use_fb:
            # set fb as link equation on all links
            for link in gas_net.get_links():
                link.set_type(link.link_type,link.link_params,link_eq_form='dp_of_q')
            het_net.reset_network(x_init_scn,formulation=formulation_fb)
            x_sol_scn,iters_scn,err_vec_scn = solve_system(het_net,tol,max_iter,h,x_init_scn,formulation_fb,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
            # set fa (back) as link equation on all links
            for link in gas_net.get_links():
                link.set_type(link.link_type,link.link_params,link_eq_form='q_of_dp')
        else:
            het_net.reset_network(x_init_scn,formulation=formulation)
            x_sol_scn,iters_scn,err_vec_scn = solve_system(het_net,tol,max_iter,h,x_init_scn,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':1,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
    print('\nDone solving the heterogeneous network in various ways.')
    
    # topology plot
    if number_of_nodes<max_number_of_nodes:
        fig1 = plt.figure('Network topology MES')
        ax1 = fig1.gca()
        het_net.draw_network(ax1,halflink_angle=3)
        plt.axis('equal')
        plt.axis('off')

    # convergence plot
    nS = len([link for link in gas_source.get_out_links() if isinstance(link,GasLink)]) -1 # -1, because one is the dummy link to the CHP
    n = int((len([node for node in gas_net.get_nodes(node_types=[1]) if (len(list(node.get_in_links()))==1 and len(list(node.get_out_links()))==1)]))/nS)
    m = int(2*n+1 - (len(gas_net.nodes)-1)/nS) #int((len([node for node in gas_net.get_nodes(node_types=[1]) if len(list(node.get_out_links())) > 2]))/nS)
    fig_conv_mes = plt.figure('Convergence plot MES (n = {}, m = {}, nS = {}, max iters = {})'.format(n,m,nS,max_iter))
    ax_conv_mes = fig_conv_mes.gca()
    plt.cla()
    plt.xlabel('Iteration k')
    plt.ylabel('Error $||F(x^k)||_2$ or $||D_F F(x^k)||_2$')
    max_iter_used = np.max([iters_pu,iters_SI,iters_scaled])
    if use_fb:
        max_iter_used = max(max_iter_used,np.max([iters_pu_fb,iters_SI_fb,iters_scaled_fb]))
    if use_un_hl:
        max_iter_used = max(max_iter_used,np.max([iters_pu_hl,iters_SI_hl,iters_scaled_hl]))
    if use_fb and use_un_hl:
        max_iter_used = max(max_iter_used,np.max([iters_pu_fb_hl,iters_SI_fb_hl,iters_scaled_fb_hl]))
    if number_of_nodes<max_number_of_nodes:
        max_iter_used = max(max_iter_used,np.max([iters_SI_FD,iters_scaled_FD]))
    if use_scn:
        max_iter_used = max(max_iter_used,iters_scn)
    if use_un_hl:
        ax_conv_mes.semilogy(np.asarray(range(0,iters_pu_hl+1)),err_vec_pu_hl,marker='d',ls='-',color='tab:green',label='p.u., hl')
        ax_conv_mes.semilogy(np.asarray(range(0,iters_SI_hl+1)),err_vec_SI_hl,marker='d',ls='-',color='tab:orange',label='S.I., unscaled, hl')
        ax_conv_mes.semilogy(np.asarray(range(0,iters_scaled_hl+1)),err_vec_scaled_hl,marker='d',ls='-',color='tab:blue',label='S.I., scaled solver, hl')
    if use_fb:
        ax_conv_mes.semilogy(np.asarray(range(0,iters_pu_fb+1)),err_vec_pu_fb,marker='o',ls='-',color='tab:green',label='p.u., fb')
        ax_conv_mes.semilogy(np.asarray(range(0,iters_SI_fb+1)),err_vec_SI_fb,marker='o',ls='-',color='tab:orange',label='S.I., unscaled, fb')
        ax_conv_mes.semilogy(np.asarray(range(0,iters_scaled_fb+1)),err_vec_scaled_fb,marker='o',ls='-',color='tab:blue',label='S.I., scaled solver, fb')
    if use_un_hl and use_fb:
        ax_conv_mes.semilogy(np.asarray(range(0,iters_pu_fb_hl+1)),err_vec_pu_fb_hl,marker='s',ls='-',color='tab:green',label='p.u., fb, hl')
        ax_conv_mes.semilogy(np.asarray(range(0,iters_SI_fb_hl+1)),err_vec_SI_fb_hl,marker='s',ls='-',color='tab:orange',label='S.I., unscaled, fb, hl')
        ax_conv_mes.semilogy(np.asarray(range(0,iters_scaled_fb_hl+1)),err_vec_scaled_fb_hl,marker='s',ls='-',color='tab:blue',label='S.I., scaled solver, fb, hl')
    ax_conv_mes.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax_conv_mes.semilogy(np.asarray(range(0,iters_pu+1)),err_vec_pu,marker='*',ls='-',color='tab:green',label='p.u.')
    ax_conv_mes.semilogy(np.asarray(range(0,iters_SI+1)),err_vec_SI,marker='*',ls='-',color='tab:orange',label='S.I., unscaled')
    ax_conv_mes.semilogy(np.asarray(range(0,iters_scaled+1)),err_vec_scaled,marker='*',ls='-',color='tab:blue',label='S.I., scaled solver')
    if number_of_nodes<max_number_of_nodes:
        ax_conv_mes.semilogy(np.asarray(range(0,iters_SI_FD+1)),err_vec_SI_FD,marker='*',ls='--',color='tab:orange',label='S.I., unscaled, FD')
        ax_conv_mes.semilogy(np.asarray(range(0,iters_scaled_FD+1)),err_vec_scaled_FD,marker='*',ls='--',color='tab:blue',label='S.I., scaled solver, FD')
        ax_conv_mes.semilogy(np.asarray(range(0,iters_pu_FD+1)),err_vec_pu_FD,marker='*',ls='--',color='tab:green',label='p.u. FD')
    if use_scn:
        ax_conv_mes.semilogy(np.asarray(range(0,iters_scn+1)),err_vec_scn,marker='.',ls='-',color='tab:purple',label='SC sol, scaled solver')
        print('errors using single-carrier network solutions as guess: {}'.format(err_vec_scn))
    plt.legend(loc='upper right')
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    plt.tight_layout()
    #fig_conv_mes.suptitle('Final error: p.u. = {:.2e}, scaled = {:.2e}, unscaled = {:.2e}'.format(err_vec_pu[-1],err_vec_scaled[-1],err_vec_SI[-1]))
    
    if save_fig:
        # enlarge figure to maximum size to prevent the legend and labels from being to crowded (TkAgg backend)
        manager = plt.get_current_fig_manager()
        #manager.resize(*manager.window.maxsize()) # maximizes window, fills both screens (when using 2 monitors)
        screen_y = manager.window.winfo_screenheight()
        screen_x = manager.window.winfo_screenwidth()
        d = 10  # width of the window border in pixels
        manager.resize(*(screen_x/2 - 2*d,screen_y - 4*d)) # screen_x/2 to compensate for using 2 monitors
        # figure setting to save figure
        fig_width_pt = 469.75502  #[pt] Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]
        pgf_with_latex = {
            "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
            "text.usetex": False,                # True: use LaTeX to write all text
            "font.family": "DejaVu Sans",#"serif",
            "font.serif": [],                   # blank entries should cause plots
            "font.monospace": [],               # to inherit fonts from the document
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.loc": "upper right",
            "legend.fontsize": 9,               # Make the legend/label fonts
            "xtick.labelsize": 10,               # a little smaller
            "ytick.labelsize": 10,
            "figure.figsize": fig_size,
            }
        mpl.rcParams.update(pgf_with_latex)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        path_to_fig = os.path.join(dir_path,'Figures','N_1Q')
        file_name = 'convergence_plot_MES_1Q'+'_n'+str(n)+'_m'+str(m)+'_nS'+str(nS)+'_'+str(int(pg_perc_low*100))+'ppercent'+'_'+str(int(T_perc_low*100))+'Tpercent'
        if use_linear_pipe:
            file_name += '_linear_pipe'
        if use_fb:
            file_name += '_fb'
        if use_un_hl:
            file_name += '_hl'
        file_name += '.pgf'
        plt.savefig(os.path.join(path_to_fig, file_name))
        
    if use_fb:
        het_net.update_full(x_sol_scaled_fb,formulation=formulation_fb)
        print('\nSolution of coupling dummy links, using fb in gas, and scaling in solver:')
        print('qc = {}[kg/s], Pc = {}[p.u.], phic = {}[W], mc = {}[kg/s]'.format(gas_net.links[-1].q,elec_net.links[-1].Pstart,-heat_net.links[-1].dphistart,heat_net.links[-1].m))
    else:
        het_net.update_full(x_sol_scaled,formulation=formulation)
        print('\nSolution of coupling dummy links, using fa in gas, and scaling in solver:')
        print('qc = {}[kg/s], Pc = {}[p.u.], phic = {}[W], mc = {}[kg/s]'.format(gas_net.links[-1].q,elec_net.links[-1].Pstart,-heat_net.links[-1].dphistart,heat_net.links[-1].m))

def solve_MES_1Q(dir_path,pg_perc_low,pg_perc_high,T_perc_low,T_perc_high,save_fig,use_scn):
    """Solve this example, using fb in the gas network and scaling in the solver.
    
    Parameters
    ----------
    p_perc_low : float
        Lowest value of fraction of reference pressure that is used in the initial linear pressure profile.
    p_perc_high : float
        Highest value of fraction of reference pressure that is used in the initial linear pressure profile.
    T_perc_low : float
        Lowest value of fraction of reference temperature that is used in the initial linear temperature profile.
    T_perc_high : float
        Highest value of fraction of reference temperature that is used in the initial linear temperature profile.
    save_fig : bool
        If True, the convergence plot is saved as a pgf. 
    use_scn: bool
        If True, the single-carrier networks are solved, and their solution is used as intial guess for the MES
        
    Returns
    -------
    x_sol : np array
        Solution vector
    """
    # create network from data
    het_net, gas_net, elec_net, heat_net, water, gas = create_network()
    formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}
    # set fb as link equation on all links
    for link in gas_net.get_links():
        link.set_type(link.link_type,link.link_params,link_eq_form='dp_of_q')
        
    # initialize network
    x0 = initialize_network(het_net,gas_net,elec_net,heat_net,water,gas,formulation,pg_perc_high,pg_perc_low,T_perc_high,T_perc_low)
    
    # Scaling in solver
    gas_source = gas_net.nodes[0]
    pg_ref = gas_source.p
    heat_source = heat_net.nodes[0]
    ph_ref = heat_source.p
    qbase = 1e-4
    pgbase = pg_ref
    mbase = 50.
    phbase = ph_ref
    Tbase = 100.
    Sbase = 1.
    phibase = Sbase*kW
    Egbase = phibase
    Vbase = 1.
    deltabase = 1.
    # solve network
    h = 1e-6
    tol = 1e-6
    max_iter = 200
    print('\nSolving heterogeneous network, using standard initial guess....')
    x_sol,iters,err_vec = solve_system(het_net,tol,max_iter,h,x0,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
    print('....Done')
    if use_scn:
        # solve the network, using the solutions of the single-carrier networks as initial guess
        print('\nSolving heterogeneous network, using solution to single-carrier networks as initial guess. Setting initial condition....')
        xg_sol = GN_1Q.solve_GN_1Q(dir_path,pg_perc_low,pg_perc_high,save_fig)
        xe_sol = EN_1Q.solve_EN_1Q(dir_path,save_fig)
        xh_sol = HN_1Q.solve_HN_1Q(dir_path,pg_perc_low,pg_perc_high,T_perc_low,T_perc_high,save_fig)
        xc_init_scn = x0[len(x0)-6:]
        x_init_scn = np.concatenate((xg_sol,xe_sol,xh_sol,xc_init_scn))
        print('Solving heterogeneous network....')
        het_net.reset_network(x_init_scn,formulation=formulation)
        x_sol_scn,iters_scn,err_vec_scn = solve_system(het_net,tol,max_iter,h,x_init_scn,formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
        print('....Done')
    # convergence plot
    nS = len([link for link in gas_source.get_out_links() if isinstance(link,GasLink)]) -1 # -1, because one is the dummy link to the CHP
    n = int((len([node for node in gas_net.get_nodes(node_types=[1]) if (len(list(node.get_in_links()))==1 and len(list(node.get_out_links()))==1)]))/nS)
    m = int(2*n+1 - (len(gas_net.nodes)-1)/nS) #int((len([node for node in gas_net.get_nodes(node_types=[1]) if len(list(node.get_out_links())) > 2]))/nS)
    fig_conv_mes_one_solver = plt.figure('Convergence plot MES (n = {}, m = {}, nS = {}, max iters = {}, len(x) = {})'.format(n,m,nS,max_iter,len(x0)))
    ax_conv_mes_one_solver = fig_conv_mes_one_solver.gca()
    plt.cla()
    plt.xlabel('Iteration k')
    plt.ylabel('Error $||D_F F(x^k)||_2$')
    max_iter_used = iters
    if use_scn:
        max_iter_used = max(max_iter_used,iters_scn)
    ax_conv_mes_one_solver.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax_conv_mes_one_solver.semilogy(err_vec,marker='o',ls='-',color='tab:blue',label='standard init. guess')
    if use_scn:
        ax_conv_mes_one_solver.semilogy(err_vec_scn,marker='.',ls='-',color='tab:purple',label='SC sol init. guess')
        plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    if save_fig:
        # enlarge figure to maximum size to prevent the legend and labels from being to crowded (TkAgg backend)
        manager = plt.get_current_fig_manager()
        #manager.resize(*manager.window.maxsize()) # maximizes window, fills both screens (when using 2 monitors)
        screen_y = manager.window.winfo_screenheight()
        screen_x = manager.window.winfo_screenwidth()
        d = 10  # width of the window border in pixels
        manager.resize(*(screen_x/2 - 2*d,screen_y - 4*d)) # screen_x/2 to compensate for using 2 monitors
        # figure setting to save figure
        fig_width_pt = 469.75502  #[pt] Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]
        pgf_with_latex = {
            "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
            "text.usetex": False,                # True: use LaTeX to write all text
            "font.family": "DejaVu Sans",#"serif",
            "font.serif": [],                   # blank entries should cause plots
            "font.monospace": [],               # to inherit fonts from the document
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.loc": "upper right",
            "legend.fontsize": 9,               # Make the legend/label fonts
            "xtick.labelsize": 10,               # a little smaller
            "ytick.labelsize": 10,
            "figure.figsize": fig_size,
            }
        mpl.rcParams.update(pgf_with_latex)

        path_to_fig = os.path.join(dir_path,'Figures','N_1Q')
        file_name = 'convergence_plot_MES_1Q'+'_n'+str(n)+'_m'+str(m)+'_nS'+str(nS)+'_'+str(int(pg_perc_low*100))+'ppercent'+'_'+str(int(T_perc_low*100))+'Tpercent'+'_one_solver'+'.pgf'
        plt.savefig(os.path.join(path_to_fig, file_name))
        
if __name__ == '__main__':
    # parse the command line
    args = command_line_input.parse_args()

    remake_network(args.n,args.m,args.nS)

    if args.comp_conv:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Only a",UserWarning)
            compare_conv_solvers(args.pg_low,args.pg_high,args.T_low,args.T_high,args.max_nodes,args.fb,args.lin_pipe,args.use_un_hl,args.save_fig,args.scn)

        if args.scn:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            GN_1Q.compare_conv_solvers(dir_path,args.pg_low,args.pg_high,args.max_nodes,args.fb,args.save_fig,args.print_sol)
            EN_1Q.compare_conv_solvers(dir_path,args.max_nodes,args.save_fig)
            HN_1Q.compare_conv_solvers(dir_path,args.pg_low,args.pg_high,args.T_low,args.T_high,args.max_nodes,args.save_fig)
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Only a",UserWarning)
            solve_MES_1Q(dir_path,args.pg_low,args.pg_high,args.T_low,args.T_high,args.save_fig,args.scn)
        
    if args.show_plots:
        plt.show()
