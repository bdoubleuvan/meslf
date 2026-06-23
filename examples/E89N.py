"""Example of an electrical network.

Data taken from pandapower, which took it from MATPOWER.
Case based on 89 pegase case.
"""
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
import pandapower as pp
import pandapower.networks as ppn
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import pytest
from meslf.utils.constants import kV, kW, MW, degree

def create_network():
    """ Create the network based on the pandapower data
    
    Returns
    -------
    elec_net : ElectricalNetwork
        The electrical network
    """
    # load data and create empty network
    pp_net = ppn.case89pegase()
    if int(pp.__version__[0]) > 1:
        # run power flow to fill the dataframes with the results. Only needed in version 2 or later
        pp.runpp(pp_net,numba=False)
        # reorder dataframes to match order of version 1
        pp_net.bus.sort_index(inplace=True)
        pp_net.gen.sort_values(by=['bus'],inplace=True)
        pp_net.sgen.sort_values(by=['bus'],inplace=True)
        pp_net.load.sort_values(by=['bus'],inplace=True)
        pp_net.line.sort_index(inplace=True)
        pp_net.trafo.sort_index(inplace=True)
        pp_net.shunt.sort_values(by=['bus'],inplace=True)
        pp_net.ext_grid.sort_values(by=['bus'],inplace=True)
        # reorder dataframes to match order of version 1
        pp_net.res_bus.sort_index(inplace=True)
        pp_net.res_gen.sort_index(inplace=True)
        pp_net.res_sgen.sort_index(inplace=True)
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
            if ind_n in pp_net.gen['bus'].values and ind_n in pp_net.sgen['bus'].values: # node with both a load, a generator, and a static generator connected to it
                ind_g = pp_net.gen.index[pp_net.gen['bus']==ind_n][0]
                if 'p_kw' in pp_net.gen.columns:
                    P_inj_gen = pp_net.gen['p_kw'][ind_g]*kW
                elif 'p_mw' in pp_net.gen.columns:
                    P_inj_gen = pp_net.gen['p_mw'][ind_g]*MW
                else:
                    ValueError('Unknown unit encountered for injected active power in generator bus.')
                if int(pp.__version__[0]) > 1:
                    P_inj_gen *= -1. 
                ind_sg = pp_net.gen.index[pp_net.sgen['bus']==ind_n][0]
                if 'p_kw' in pp_net.sgen.columns:
                    P_inj_sgen = pp_net.sgen['p_kw'][ind_sg]*kW
                elif 'p_mw' in pp_net.sgen.columns:
                    P_inj_sgen = pp_net.sgen['p_mw'][ind_sg]*MW
                else:
                    ValueError('Unknown unit encountered for injected active power in static generator bus.')
                if int(pp.__version__[0]) > 1:
                    P_inj_sgen *= -1. 
                node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=1,P=(P_inj+P_inj_gen+P_inj_sgen)/S_base,V=pp_net.gen['vm_pu'][ind_g])
            elif ind_n in pp_net.gen['bus'].values: # node with both a load and a generator connected to it
                ind_g = pp_net.gen.index[pp_net.gen['bus']==ind_n][0]
                if 'p_kw' in pp_net.gen.columns:
                    P_inj_gen = pp_net.gen['p_kw'][ind_g]*kW
                elif 'p_mw' in pp_net.gen.columns:
                    P_inj_gen = pp_net.gen['p_mw'][ind_g]*MW
                else:
                    ValueError('Unknown unit encountered for injected active power in generator bus.')
                if int(pp.__version__[0]) > 1:
                    P_inj_gen *= -1. 
                node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=1,P=(P_inj+P_inj_gen)/S_base,V=pp_net.gen['vm_pu'][ind_g])
            elif ind_n in pp_net.sgen['bus'].values: # node with both a load and a static generator connected to it
                ind_sg = pp_net.sgen.index[pp_net.sgen['bus']==ind_n][0]
                if 'p_kw' in pp_net.sgen.columns:
                    P_inj_sgen = pp_net.sgen['p_kw'][ind_sg]*kW
                elif 'p_mw' in pp_net.sgen.columns:
                    P_inj_sgen = pp_net.sgen['p_mw'][ind_sg]*MW
                else:
                    ValueError('Unknown unit encountered for injected active power in static generator bus.')
                if int(pp.__version__[0]) > 1:
                    P_inj_sgen *= -1.
                if 'q_kvar' in pp_net.sgen.columns:
                    Q_inj_sgen = pp_net.sgen['q_kvar'][ind_sg]*kW
                elif 'q_mvar' in pp_net.sgen.columns:
                    Q_inj_sgen = pp_net.sgen['q_mvar'][ind_sg]*MW
                else:
                    ValueError('Unknown unit encountered for injected reactive power in static generator bus.')
                if int(pp.__version__[0]) > 1:
                    Q_inj_sgen *= -1
                #node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=2,P=(P_inj-P_inj_sgen)/S_base,Q=(P_inj+P_inj_sgen)/S_base)
                node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=2,P=(P_inj+P_inj_sgen)/S_base,Q=(Q_inj+Q_inj_sgen)/S_base)
            else: # node with only a load connected to it
                node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=2,P=P_inj/S_base,Q=Q_inj/S_base)
                node.V = 1.
        elif ind_n in pp_net.gen['bus'].values and ind_n in pp_net.sgen['bus'].values: # node with a generator and a static generator connected to it
            ind_g = pp_net.gen.index[pp_net.gen['bus']==ind_n][0]
            if 'p_kw' in pp_net.gen.columns:
                P_inj_gen = pp_net.gen['p_kw'][ind_g]*kW
            elif 'p_mw' in pp_net.gen.columns:
                P_inj_gen = pp_net.gen['p_mw'][ind_g]*MW
            else:
                ValueError('Unknown unit encountered for injected active power in generator bus.')
            if int(pp.__version__[0]) > 1:
                P_inj_gen *= -1. 
            ind_sg = pp_net.sgen.index[pp_net.sgen['bus']==ind_n][0]
            if 'p_kw' in pp_net.sgen.columns:
                P_inj_sgen = pp_net.sgen['p_kw'][ind_sg]*kW
            elif 'p_mw' in pp_net.sgen.columns:
                P_inj_sgen = pp_net.sgen['p_mw'][ind_sg]*MW
            else:
                ValueError('Unknown unit encountered for injected active power in static generator bus.')
            if int(pp.__version__[0]) > 1:
                P_inj_sgen *= -1.
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=1,P=(P_inj_gen+P_inj_sgen)/S_base,V=pp_net.gen['vm_pu'][ind_g])
        elif ind_n in pp_net.gen['bus'].values: # node with only a generator connected to it
            ind_g = pp_net.gen.index[pp_net.gen['bus']==ind_n][0]
            if 'p_kw' in pp_net.gen.columns:
                P_inj_gen = pp_net.gen['p_kw'][ind_g]*kW
            elif 'p_mw' in pp_net.gen.columns:
                P_inj_gen = pp_net.gen['p_mw'][ind_g]*MW
            else:
                ValueError('Unknown unit encountered for injected active power in generator bus.')
            if int(pp.__version__[0]) > 1:
                P_inj_gen *= -1. 
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=1,P=P_inj_gen/S_base,V=pp_net.gen['vm_pu'][ind_g])
        elif ind_n in pp_net.ext_grid['bus'].values: # reference
            ind_s = pp_net.ext_grid.index[pp_net.ext_grid['bus']==ind_n][0]
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=0,V=pp_net.ext_grid['vm_pu'][ind_s],delta=pp_net.ext_grid['va_degree'][ind_s]*degree) 
        elif ind_n in pp_net.sgen['bus'].values: # static generator, which are modelled as negative PQ loads
            ind_sg = pp_net.sgen.index[pp_net.sgen['bus']==ind_n][0]
            if 'p_kw' in pp_net.sgen.columns:
                P_inj_sgen = pp_net.sgen['p_kw'][ind_sg]*kW
            elif 'p_mw' in pp_net.sgen.columns:
                P_inj_sgen = pp_net.sgen['p_mw'][ind_sg]*MW
            else:
                ValueError('Unknown unit encountered for injected active power in static generator bus.')
            if int(pp.__version__[0]) > 1:
                P_inj_sgen *= -1.
            if 'q_kvar' in pp_net.sgen.columns:
                Q_inj_sgen = pp_net.sgen['q_kvar'][ind_sg]*kW
            elif 'q_mvar' in pp_net.sgen.columns:
                Q_inj_sgen = pp_net.sgen['q_mvar'][ind_sg]*MW
            else:
                ValueError('Unknown unit encountered for injected reactive power in load bus.')
            if int(pp.__version__[0]) > 1:
                Q_inj_sgen *= -1
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=2,P=P_inj_sgen/S_base,Q=Q_inj_sgen/S_base)
            node.V = 1.
        else: # junctions
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=2,P=0.,Q=0.)
            node.V = 1.
        node.V_base = V_base
        node.x = pp_net.bus_geodata['x'][ind_n]
        node.y = pp_net.bus_geodata['y'][ind_n]
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

def solve_example_e89n():
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
    max_iter = 50
    
    elec_net.reset_network(x0,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params) 
    return elec_net,S_base,pp_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge

def test_topology_e89n():
    # When
    elec_net,S_base,pp_net = create_network()
    number_of_links = len(elec_net.links)
    number_of_nodes = len(elec_net.nodes)
    # Then
    number_of_links_pp = 50 + 160 # trafo + line
    number_of_nodes_pp = 89
    assert np.all([number_of_links,number_of_nodes] == [number_of_links_pp,number_of_nodes_pp])

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_delta_e89n():
    # When
    elec_net,S_base,pp_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e89n()
    
    # Then
    delta_sol_pp = pp_net.res_bus['va_degree'].values*degree
    assert np.allclose(delta_sol_pp,delta_sol)
    
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_V_e89n():
    # When
    elec_net,S_base,pp_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e89n()

    # Then
    V_sol_pp = pp_net.res_bus['vm_pu'].values #[p.u]
    assert np.allclose(V_sol_pp,V_sol)

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_Pline_e89n():
    # When
    elec_net,S_base,pp_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e89n()

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

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_Qlink_e89n():
    # When
    elec_net,S_base,pp_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e89n()

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

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_S_inj_e89n():
    # When
    elec_net,S_base,pp_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e89n()

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
    
    # plot topology
    fig_top = plt.figure('Network topology')
    ax_top = fig_top.gca()
    elec_net.draw_network(ax_top)
    plt.axis('equal')
    plt.axis('off')
    
    plt.show()
