"""Example of an electrical network with 14 nodes.

Data taken from pandapower, which took it form PyPower, which took it from MATPOWER.
Case based on IEEE 14 bus system
"""
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
import pandapower as pp
import pandapower.networks as ppn
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from meslf.utils.constants import kV, kW, MW, degree

def create_network():
    """ Create the network based on the pandapower data
    
    Returns
    -------
    elec_net : ElectricalNetwork
        The electrical network
    """
    # load data and create empty network
    pp_net = ppn.case14()
    if int(pp.__version__[0]) > 1:
        # run power flow to fill the dataframes with the results. Only needed in version 2 or later
        pp.runpp(pp_net,numba=False)
        # reorder dataframes to match order of version 1
        pp_net.bus.sort_values(by=['name'],inplace=True)
        pp_net.gen.sort_values(by=['bus'],inplace=True)
        pp_net.load.sort_values(by=['bus'],inplace=True)
        pp_net.line.sort_values(by=['from_bus'],inplace=True)
        pp_net.trafo.sort_values(by=['hv_bus'],inplace=True)
        pp_net.shunt.sort_values(by=['bus'],inplace=True)
        pp_net.ext_grid.sort_values(by=['bus'],inplace=True)
        # reorder dataframes to match order of version 1
        pp_net.res_bus.sort_index(inplace=True)
        pp_net.res_gen.sort_index(inplace=True)
        pp_net.res_load.sort_index(inplace=True)
        pp_net.res_line.sort_index(inplace=True)
        pp_net.res_trafo.sort_index(inplace=True)
        pp_net.res_shunt.sort_index(inplace=True)
        pp_net.res_ext_grid.sort_index(inplace=True)
    elec_net = ElectricalNetwork(pp_net.name) 
    
    # some base values
    f = pp_net.f_hz
    if 'sn_kva' in pp_net.keys():
        S_base = pp_net.sn_kva*kW #[va]
    elif 'sn_mva' in pp_net.keys():
        S_base = pp_net.sn_mva*MW  #[kva] 
    else:
        ValueError('Cannot get network complex power base value!')  
    
    node_dict = {} 
    for ind_n in pp_net.bus.index: # assign all values in p.u., such that I don't have to use scaling when solving
        V_base = pp_net.bus['vn_kv'][ind_n]*kV
        if ind_n in pp_net.load['bus'].values: # loads / demands
            ind_l = pp_net.load.index[pp_net.load['bus']==ind_n][0]
            if 'p_kw' in pp_net.load.columns:
                P_inj = pp_net.load['p_kw'][ind_l]*kW
            elif 'p_mw' in pp_net.load.columns:
                P_inj = pp_net.load['p_mw'][ind_l]*MW
            else:
                ValueError('Unknown unit encountered for injected active power in load bus.')
            if 'q_kvar' in pp_net.load.columns:
                Q_inj = pp_net.load['q_kvar'][ind_l]*kW
            elif 'q_mvar' in pp_net.load.columns:
                Q_inj = pp_net.load['q_mvar'][ind_l]*MW
            else:
                ValueError('Unknown unit encountered for injected reactive power in load bus.')
            if ind_n in pp_net.gen['bus'].values: # node with both a load and a generator connected to it
                ind_g = pp_net.gen.index[pp_net.gen['bus']==ind_n][0]
                if 'p_kw' in pp_net.gen.columns:
                    P_inj_gen = pp_net.gen['p_kw'][ind_g]*kW
                elif 'p_mw' in pp_net.gen.columns:
                    P_inj_gen = pp_net.gen['p_mw'][ind_g]*MW
                else:
                    ValueError('Unknown unit encountered for injected active power in generator bus.')
                if P_inj_gen> 0:
                    P_inj_gen *= -1. 
                node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=1,P=(P_inj+P_inj_gen)/S_base,V=pp_net.gen['vm_pu'][ind_g])
            else: # node with only a load connected to it
                node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=2,P=P_inj/S_base,Q=Q_inj/S_base)
                node.V = 1.
        elif ind_n in pp_net.gen['bus'].values: # node with only a generator connected to it
            ind_g = pp_net.gen.index[pp_net.gen['bus']==ind_n][0]
            if 'p_kw' in pp_net.gen.columns:
                P_inj = pp_net.gen['p_kw'][ind_g]*kW
            elif 'p_mw' in pp_net.gen.columns:
                P_inj = pp_net.gen['p_mw'][ind_g]*MW
            else:
                ValueError('Unknown unit encountered for injected active power in generator bus.')
            if P_inj > 0:
                P_inj *= -1. 
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=1,P=P_inj/S_base,V=pp_net.gen['vm_pu'][ind_g])
        elif ind_n in pp_net.ext_grid['bus'].values: # reference
            ind_s = pp_net.ext_grid.index[pp_net.ext_grid['bus']==ind_n][0]
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=0,V=pp_net.ext_grid['vm_pu'][ind_s],delta=pp_net.ext_grid['va_degree'][ind_s]*degree) 
        else: # junctions
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=2,P=0.,Q=0.)
            node.V = 1.
        node.V_base = V_base
        elec_net.add_node(node)

    for ind_e in pp_net.line.index:
        start_node = elec_net.nodes[pp_net.line['from_bus'][ind_e]]
        end_node = elec_net.nodes[pp_net.line['to_bus'][ind_e]]
        V_base = start_node.V_base
        Z_base = V_base**2 / S_base
        Y_base = S_base / (V_base**2)
        L = pp_net.line['length_km'][ind_e]
        par = pp_net.line['parallel'][ind_e]
        x = pp_net.line['x_ohm_per_km'][ind_e]*L/par
        r = pp_net.line['r_ohm_per_km'][ind_e]*L/par
        z = r+x*1j
        b = -x/(np.abs(z)**2)
        g = r/(np.abs(z)**2) 
        b_sh = 2*np.pi*f*pp_net.line['c_nf_per_km'][ind_e]*1e-9*L*par #c_nf = capacitance in nano Farad
        g_sh = pp_net.line['g_us_per_km'][ind_e]*1e-6*L*par #g_us = dielectric conductance in micro Siemens
        b_pu = b/Y_base
        g_pu = g/Y_base
        bsh_pu = b_sh/Y_base
        gsh_pu = g_sh/Y_base
        link = ElectricalLink(str(pp_net.line['name'][ind_e]),start_node,end_node,link_type = 'pi_line',link_params = {'b':b_pu,'g':g_pu,'b_sh':bsh_pu,'g_sh':gsh_pu})
        elec_net.add_link(link)
        
    for ind_e in pp_net.trafo.index:
        if int(pp.__version__[0]) > 1:
            if pp_net.trafo['tap_side'][ind_e] == 'hv':
                ntap = 1+(pp_net.trafo['tap_pos'][ind_e] - pp_net.trafo['tap_neutral'][ind_e])*pp_net.trafo['tap_step_percent'][ind_e]/100
                Vref_hv_trafo = pp_net.trafo['vn_hv_kv'][ind_e]*ntap
                Vref_lv_trafo = pp_net.trafo['vn_lv_kv'][ind_e]
            elif pp_net.trafo['tap_side'][ind_e] == 'lv':
                ntap = 1+(pp_net.trafo['tap_pos'][ind_e] - pp_net.trafo['tap_neutral'][ind_e])*pp_net.trafo['tap_step_percent'][ind_e]/100
                Vref_hv_trafo = pp_net.trafo['vn_hv_kv'][ind_e]
                Vref_lv_trafo = pp_net.trafo['vn_lv_kv'][ind_e]*ntap
            else:
                Vref_hv_trafo = pp_net.trafo['vn_hv_kv'][ind_e]
                Vref_lv_trafo = pp_net.trafo['vn_lv_kv'][ind_e]
        else:
            if pp_net.trafo['tp_side'][ind_e] == 'hv':
                ntap = 1+(pp_net.trafo['tp_pos'][ind_e] - pp_net.trafo['tp_mid'][ind_e])*pp_net.trafo['tp_st_percent'][ind_e]/100
                Vref_hv_trafo = pp_net.trafo['vn_hv_kv'][ind_e]*ntap
                Vref_lv_trafo = pp_net.trafo['vn_lv_kv'][ind_e]
            elif pp_net.trafo['tp_side'][ind_e] == 'lv':
                ntap = 1+(pp_net.trafo['tp_pos'][ind_e] - pp_net.trafo['tp_mid'][ind_e])*pp_net.trafo['tp_st_percent'][ind_e]/100
                Vref_hv_trafo = pp_net.trafo['vn_hv_kv'][ind_e]
                Vref_lv_trafo = pp_net.trafo['vn_lv_kv'][ind_e]*ntap
            else:
                Vref_hv_trafo = pp_net.trafo['vn_hv_kv'][ind_e]
                Vref_lv_trafo = pp_net.trafo['vn_lv_kv'][ind_e]
        start_node = elec_net.nodes[pp_net.trafo['hv_bus'][ind_e]]
        end_node = elec_net.nodes[pp_net.trafo['lv_bus'][ind_e]]    
        ratio = Vref_hv_trafo/Vref_lv_trafo * end_node.V_base/start_node.V_base
        phase = pp_net.trafo['shift_degree'][ind_e]*degree
        V_base = pp_net.trafo['vn_lv_kv'][ind_e]*kV # at low voltage side
        Z_base = V_base**2 / S_base
        if int(pp.__version__[0]) > 1:
            Z_ref_trafo = ((pp_net.trafo['vn_lv_kv'][ind_e]*kV)**2) /(pp_net.trafo['sn_mva'][ind_e]*MW)
            z_abs = pp_net.trafo['vk_percent'][ind_e]/100 *1/(pp_net.trafo['sn_mva'][ind_e])
            r = pp_net.trafo['vkr_percent'][ind_e]/100 *1/(pp_net.trafo['sn_mva'][ind_e])
            ysh_abs = pp_net.trafo['i0_percent'][ind_e]/100
            gsh = pp_net.trafo['pfe_kw'][ind_e]/((pp_net.trafo['sn_mva'][ind_e])**2) #*MW?? but 'pfe_kw' = 0, so doesn't matter
        else:
            Z_ref_trafo = ((pp_net.trafo['vn_lv_kv'][ind_e])**2) *1e3/(pp_net.trafo['sn_kva'][ind_e])
            z_abs = pp_net.trafo['vsc_percent'][ind_e]/100 *1000/(pp_net.trafo['sn_kva'][ind_e])
            r = pp_net.trafo['vscr_percent'][ind_e]/100 *1000/(pp_net.trafo['sn_kva'][ind_e])
            ysh_abs = pp_net.trafo['i0_percent'][ind_e]/100
            gsh = pp_net.trafo['pfe_kw'][ind_e]/((pp_net.trafo['sn_kva'][ind_e])**2)
        x = np.sqrt(z_abs**2 - r**2)
        b = -x/(z_abs**2)
        g = r/(z_abs**2)
        bsh = np.sqrt(ysh_abs**2 - gsh**2)
        # Ik heb geen idee waarom, maar pandapower zegt dat het keer deze factor moet, maar als ik vergelijk met de matpower data dan moet het juist niet!
        b_pu = b#*Z_ref_trafo/Z_base
        g_pu = g#*Z_ref_trafo/Z_base
        bsh_pu = bsh*Z_base/Z_ref_trafo
        gsh_pu = gsh*Z_base/Z_ref_trafo
        link = ElectricalLink(str(pp_net.line['name'][ind_e]),start_node,end_node,link_type = 'pi_line_trafo',link_params = {'b':b_pu,'g':g_pu,'b_sh':bsh_pu,'g_sh':gsh_pu,'ratio':ratio,'phase_shift':phase})
        elec_net.add_link(link)
    
    for ind_s in pp_net.shunt.index:
        if int(pp.__version__[0]) > 1:
            g_pu = pp_net.shunt['p_mw'][ind_s] *MW* pp_net.shunt['step'][ind_s] / S_base
            b_pu = pp_net.shunt['q_mvar'][ind_s] *MW* pp_net.shunt['step'][ind_s] / S_base
        else:
            g_pu = pp_net.shunt['p_kw'][ind_s]*kW * pp_net.shunt['step'][ind_s] / S_base
            b_pu = pp_net.shunt['q_kvar'][ind_s]*kW * pp_net.shunt['step'][ind_s] / S_base
        start_node = elec_net.nodes[pp_net.shunt.bus[ind_s]]
        if pp_net.shunt['name'][ind_s]:
            hl_name = pp_net.shunt['name'][ind_s]
        else:
            hl_name = start_node.name + '_hl_sh'
        half_link = ElectricalHalfLink(hl_name,start_node,link_type='nodal_shunt',link_params = {'b':-b_pu,'g':g_pu})
        
    return elec_net,S_base,pp_net

def initialize_network(network,scale_var=None,scale_var_params=None):
    """Sets values of network variables to be used for initial guess.
    
    Parameters
    ----------
    network : ElectricalNetwork
        The network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    network.initialize()
    # since the network is created from pandapower data, the initial guesses for nodal voltages are already set when creating the network
    x0 = network.set_x_init(scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def solve_example_e14n():
    """Solve of the example network
    """
    # create network
    elec_net,S_base,pp_net = create_network()
    scale_var = None#'per_unit'
    scale_var_params = {'Sbase':S_base,'Vbase':1,'deltabase':1}
    x0 = initialize_network(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # solving using FD and analytical solution
    h = 1e-6
    tol = 1e-6
    max_iter = 20

    elec_net.update(x0,scale_var=scale_var,scale_var_params=scale_var_params)
    print("\nSolving system using FD Jacobian")
    x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD = elec_net.solve_network(tol,max_iter,h=h,solver='NR_FD',scale_var=scale_var,scale_var_params=scale_var_params)
    
    elec_net.reset_network(x0,scale_var=scale_var,scale_var_params=scale_var_params)
    print("\nSolving system using analytical Jacobian")
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params) 
    return elec_net,S_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge

def test_topology_e14n():
    # When
    elec_net,S_base,pp_net = create_network()
    number_of_links = len(elec_net.links)
    number_of_nodes = len(elec_net.nodes)
    # Then
    number_of_links_pp = 20
    number_of_nodes_pp = 14
    assert np.all([number_of_links,number_of_nodes] == [number_of_links_pp,number_of_nodes_pp])

def test_Y_e14n():
    # Given 
    elec_net,S_base,pp_net = create_network()
    # When 
    elec_net.initialize()
    Y = elec_net.Y
    # Then
    Y_pp_data = [602.502905577-1944.70702055j, -499.91316008+1526.30865232j, -102.589745497+423.498368233j, -499.91316008+1526.30865232j, 952.132361081-3027.21153988j, -113.501919231+478.186315176j, -168.603315061+511.583832587j, -170.113966709+519.392739797j, -113.501919231+478.186315176j, 312.099490223-982.238012935j, -198.597570993+506.881697759j, -168.603315061+511.583832587j, -198.597570993+506.881697759j, 1051.2989522-3865.41712076j, -684.09806615+2157.85539817j, 488.951266032j, 185.549955782j, -102.589745497+423.498368233j, -170.113966709+519.392739797j, -684.09806615+2157.85539817j, 956.801778356-3553.3639456j, 425.744533525j, -0+425.744533525j, 657.992340747-1734.07328099j, -195.502856318+409.407434424j, -152.596744045+317.596396503j, -309.892740384+610.275544819j, -0+488.951266032j, -1954.90059483j, 567.697984672j, 909.008271975j, -0+567.697984672j, -567.697984672j, -0+185.549955782j, -0+909.008271975j, 532.605503947-2409.25063753j, -390.204955245+1036.53941271j, -142.400548702+302.905045693j, -390.204955245+1036.53941271j, 578.293430615-1476.83378765j, -188.08847537+440.294374946j, -195.502856318+409.407434424j, -188.08847537+440.294374946j, 383.591331688-849.70180937j, -152.596744045+317.596396503j, 401.499202727-542.79385912j, -248.902458682+225.197462617j, -309.892740384+610.275544819j, -248.902458682+225.197462617j, 672.494614847-1066.96935495j, -113.699415781+231.496347511j, -142.400548702+302.905045693j, -113.699415781+231.496347511j, 256.099964483-534.401393204j]
    Y_pp_ind = [0, 1, 4, 0, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 6, 8, 0, 1, 3, 4, 5, 4, 5, 10, 11, 12, 3, 6, 7, 8, 6, 7, 3, 6, 8, 9, 13, 8, 9, 10, 5, 9, 10, 5, 11, 12, 5, 11, 12, 13, 8, 12, 13]
    Y_pp_indptr = [0, 3, 8, 11, 17, 22, 27, 31, 33, 38, 41, 44, 47, 51, 54]
    Y_pp = sps.csr_matrix((Y_pp_data, Y_pp_ind, Y_pp_indptr),dtype='complex128')
    assert np.allclose(Y_pp.todense(),Y.todense())
    
def test_FD_vs_an_e14n():
    # When
    elec_net,S_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e14n()
    # Then
    assert np.allclose(x_sol_FD,x_sol)
    
def test_delta_e14n():
    # When
    elec_net,S_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e14n()
    
    # Then
    delta_sol_pp = pp_net.res_bus['va_degree'].values*degree
    assert np.allclose(delta_sol_pp,delta_sol)
    
def test_V_e14n():
    # When
    elec_net,S_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e14n()

    # Then
    V_sol_pp = pp_net.res_bus['vm_pu'].values #[p.u]
    assert np.allclose(V_sol_pp,V_sol)

def test_Pline_e14n():
    # When
    elec_net,S_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e14n()

    # Then
    if 'p_from_kw' in pp_net.res_line.keys():
        P_line_pp = pp_net.res_line['p_from_kw'].append(pp_net.res_line['p_to_kw']).values*kW #[W]
    elif 'p_from_mw' in pp_net.res_line.keys():
        P_line_pp = pp_net.res_line['p_from_mw'].append(pp_net.res_line['p_to_mw']).values*MW #[W]
    P_line_pp /= S_base #[p.u.]
    number_of_lines = len(list(elec_net.get_links(link_types=['pi_line'])))
    P_line = np.zeros(number_of_lines*2)
    for ind,line in enumerate(elec_net.get_links(link_types=['pi_line'])):
        P_line[ind] = line.Pstart
        P_line[ind+number_of_lines] = line.Pend
    assert np.allclose(P_line_pp,P_line)

def test_Qlink_e14n():
    # When
    elec_net,S_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e14n()

    # Then
    if 'q_from_kvar' in pp_net.res_line.keys():
        Q_line_pp = pp_net.res_line['q_from_kvar'].append(pp_net.res_line['q_to_kvar']).values*kW #[var]
    elif 'q_from_mvar' in pp_net.res_line.keys():
        Q_line_pp = pp_net.res_line['q_from_mvar'].append(pp_net.res_line['q_to_mvar']).values*MW #[var]
    Q_line_pp /= S_base #[p.u.]
    number_of_lines = len(list(elec_net.get_links(link_types=['pi_line'])))
    Q_line = np.zeros(number_of_lines*2)
    for ind,line in enumerate(elec_net.get_links(link_types=['pi_line'])):
        Q_line[ind] = line.Qstart
        Q_line[ind+number_of_lines] = line.Qend
    assert np.allclose(Q_line_pp,Q_line)  
    
def test_S_inj_e14n():
    # When
    elec_net,S_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e14n()

    # Then
    if 'p_kw' in pp_net.res_bus.keys():
        S_inj_pp = (pp_net.res_bus['p_kw'].values + 1j*pp_net.res_bus['q_kvar'].values)*kW #[VA]
    elif 'p_mw' in pp_net.res_bus.keys():
        S_inj_pp = (pp_net.res_bus['p_mw'].values + 1j*pp_net.res_bus['q_mvar'].values)*MW #[VA]
    S_inj_pp /= S_base #[p.u.]
    assert np.allclose(S_inj_pp,S_inj)
    
if __name__ == '__main__':
    # create network
    elec_net,S_base,pp_net = create_network() # Network is created in p.u., because of trafo's
    scale_var = 'per_unit'
    scale_var_params = {'Sbase':S_base,'Vbase':1,'deltabase':1}
    
    # Solve network in different ways, compare convergence
    h = 1e-6
    tol = 1e-6
    max_iter = 50
    # Scaled (i.e., in p.u.)
    x0 = initialize_network(elec_net) # p.u.
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR')
    number_of_lines = len(list(elec_net.get_links(link_types=['pi_line'])))
    P_line = np.zeros(number_of_lines*2)
    for ind,line in enumerate(elec_net.get_links(link_types=['pi_line'])):
        P_line[ind] = line.Pstart
        P_line[ind+number_of_lines] = line.Pend
    Q_line = np.zeros(number_of_lines*2)
    for ind,line in enumerate(elec_net.get_links(link_types=['pi_line'])):
        Q_line[ind] = line.Qstart
        Q_line[ind+number_of_lines] = line.Qend
    # Scaled (i.e., in p.u.), with scaled F for stopping criterion (F is scaled back to match S.I. values)
    elec_net.reset_network(x0)
    Fb = S_base * np.ones(len(x0))
    DF = np.diag(Fb)
    x_sol_DF,iters_DF,err_vec_DF,delta_sol_DF,V_sol_DF,S_inj_DF,P_edge_DF,Q_edge_DF = elec_net.solve_network(tol,max_iter,solver='NR',D_F=DF)
    P_line_DF = np.zeros(number_of_lines*2)
    for ind,line in enumerate(elec_net.get_links(link_types=['pi_line'])):
        P_line_DF[ind] = line.Pstart
        P_line_DF[ind+number_of_lines] = line.Pend
    Q_line_DF = np.zeros(number_of_lines*2)
    for ind,line in enumerate(elec_net.get_links(link_types=['pi_line'])):
        Q_line_DF[ind] = line.Qstart
        Q_line_DF[ind+number_of_lines] = line.Qend
    
    # plot convergence
    fig_conv = plt.figure('Convergence plot')
    ax_conv = fig_conv.gca()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    max_iter_used = np.max([iters,iters_DF])
    ax_conv.semilogy([0,max_iter_used+1],[tol,tol],':r',label='tolerance')
    ax_conv.semilogy(np.asarray(range(0,iters+1)),err_vec,'-d',label='p.u.')
    ax_conv.semilogy(np.asarray(range(0,iters_DF+1)),err_vec_DF,'-*',label='p.u. $D_F$ (Sb = {})'.format(S_base))
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    
    # compare with solution
    delta_sol_pp = pp_net.res_bus['va_degree'].values*degree #[rad]
    V_sol_pp = pp_net.res_bus['vm_pu'].values #[p.u]
    if 'p_from_kw' in pp_net.res_line.keys():
        P_line_pp = pp_net.res_line['p_from_kw'].append(pp_net.res_line['p_to_kw']).values*kW #[W]
    elif 'p_from_mw' in pp_net.res_line.keys():
        P_line_pp = pp_net.res_line['p_from_mw'].append(pp_net.res_line['p_to_mw']).values*MW #[W]
    P_line_pp /= S_base #[p.u.]
    if 'q_from_kvar' in pp_net.res_line.keys():
        Q_line_pp = pp_net.res_line['q_from_kvar'].append(pp_net.res_line['q_to_kvar']).values*kW #[var]
    elif 'q_from_mvar' in pp_net.res_line.keys():
        Q_line_pp = pp_net.res_line['q_from_mvar'].append(pp_net.res_line['q_to_mvar']).values*MW #[var]
    Q_line_pp /= S_base #[p.u.]
    
    
    fig_V, (ax_delta, ax_V) = plt.subplots(2, 1, sharex=True)
    fig_V.canvas.set_window_title('Voltage error')
    ax_delta.set_ylabel('$\delta_{sol} - \delta$ in [rad]')
    ax_delta.plot(delta_sol_pp-delta_sol,'d',label='p.u.')
    ax_delta.plot(delta_sol_pp-delta_sol_DF,'*',label='p.u. $D_F$')
    ax_delta.legend()
    ax_delta.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_delta.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_V.set_xlabel('Node number')
    ax_V.set_ylabel('$|V|_{sol} - |V|$ in [p.u.]')
    ax_V.plot(V_sol_pp-V_sol,'d',label='p.u.')
    ax_V.plot(V_sol_pp-V_sol_DF,'*',label='p.u. $D_F$')
    ax_V.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_V.grid(which='minor',color='k', linestyle=':', alpha=.05)
    
    fig_Sline, (ax_Pline, ax_Qline) = plt.subplots(2, 1, sharex=True)
    fig_Sline.canvas.set_window_title('Link Power error')
    ax_Pline.set_ylabel('$P_{sol} - P$ in [p.u.]')
    ax_Pline.plot(P_line_pp-P_line,'d',label='p.u.')
    ax_Pline.plot(P_line_pp-P_line_DF,'*',label='p.u. $D_F$')
    ax_Pline.legend()
    ax_Pline.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_Pline.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_Pline.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    ax_Qline.set_xlabel('Link number')
    ax_Qline.set_ylabel('$Q_{sol} - Q$ in [p.u.]')
    ax_Qline.plot(Q_line_pp-Q_line,'d',label='p.u.')
    ax_Qline.plot(Q_line_pp-Q_line_DF,'*',label='p.u. $D_F$')
    ax_Qline.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_Qline.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_Qline.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    
    plt.show()
    
