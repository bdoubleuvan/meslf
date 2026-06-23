"""Example of an electrical network with 9 nodes. 

Data taken from pandapower, which took it form PyPower, which took it from MATPOWER.
Case based on data from page 70 of:
    Chow, J. H., editor. Time-Scale Modeling of Dynamic Networks with
    Applications to Power Systems. Springer-Verlag, 1982.
    Part of the Lecture Notes in Control and Information Sciences book
    series (LNCIS, volume 46)
"""
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink
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
    pp_net = ppn.case9()
    if int(pp.__version__[0]) > 1:
        # run power flow to fill the dataframes with the results. Only needed in version 2 or later
        pp.runpp(pp_net,numba=False)
    elec_net = ElectricalNetwork(pp_net.name) 
    
    # some base values
    f = pp_net.f_hz
    if 'sn_kva' in pp_net.keys():
        S_base = pp_net.sn_kva*kW #[va]
    elif 'sn_mva' in pp_net.keys():
        S_base = pp_net.sn_mva*MW #[va]
    else:
        ValueError('Cannot get network complex power base value!')
    V_base = pp_net.bus['vn_kv'][0]*kV #[V]    
    
    for ind_n in pp_net.bus.index:
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
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=2,P=P_inj,Q=Q_inj)
            node.V = pp_net.bus['vn_kv'][ind_n]*kV
        elif ind_n in pp_net.gen['bus'].values: # generators
            ind_g = pp_net.gen.index[pp_net.gen['bus']==ind_n][0]
            if 'p_kw' in pp_net.gen.columns:
                P_inj = pp_net.gen['p_kw'][ind_g]*kW
            elif 'p_mw' in pp_net.gen.columns:
                P_inj = pp_net.gen['p_mw'][ind_g]*MW
            else:
                ValueError('Unknown unit encountered for injected active power in generator bus.')
            if P_inj > 0:
                P_inj *= -1. 
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=1,P=P_inj,V=pp_net.gen['vm_pu'][ind_g]*V_base)
        elif ind_n in pp_net.ext_grid['bus'].values: # reference
            ind_s = pp_net.ext_grid.index[pp_net.ext_grid['bus']==ind_n][0]
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=0,V=pp_net.ext_grid['vm_pu'][ind_s]*V_base,delta=pp_net.ext_grid['va_degree'][ind_s]*degree) 
        else: # junctions
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),node_type=2,P=0.,Q=0.)
            node.V = pp_net.bus['vn_kv'][ind_n]*kV
        elec_net.add_node(node)
        
    for ind_e in pp_net.line.index:
        start_node = elec_net.nodes[pp_net.line['from_bus'][ind_e]]
        end_node = elec_net.nodes[pp_net.line['to_bus'][ind_e]]
        L = pp_net.line['length_km'][ind_e]
        par = pp_net.line['parallel'][ind_e]
        x = pp_net.line['x_ohm_per_km'][ind_e]*L/par
        r = pp_net.line['r_ohm_per_km'][ind_e]*L/par
        z = r+x*1j
        b = -x/(np.abs(z)**2)
        g = r/(np.abs(z)**2) 
        b_sh = 2*np.pi*f*pp_net.line['c_nf_per_km'][ind_e]*1e-9*L*par #c_nf = capacitance in nano Farad
        g_sh = pp_net.line['g_us_per_km'][ind_e]*1e-6*L*par #g_us = dielectric conductance in micro Siemens
        link = ElectricalLink(str(pp_net.line['name'][ind_e]),start_node,end_node,link_type = 'pi_line',link_params = {'b':b,'g':g,'b_sh':b_sh,'g_sh':g_sh})
        elec_net.add_link(link)
    
    
    for node in elec_net.get_nodes():
        if node.half_links:
            print('node {} with V = {}, P = {}, Q = {}'.format(node.name,node.V,node.half_links[0].P,node.half_links[0].Q))
        else:
            print('node {} with V = {}'.format(node.name,node.V))
    return elec_net,S_base,V_base,pp_net

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

def solve_system(network,tol,max_iter,h,x0,scale_var=None,scale_var_params=None,D_F=np.array([]),D_x=np.array([]),det_tol=1e-8):
    """Solve the network using analytical Jacobian and FD Jacobian, with basic NR
    
    Parameters
    ----------
    network : ElectricalNetwork
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
    x_sol_FD : np array
        solution vector, using FD Jacobian
    iters_FD : int
        total number of iterations, using FD Jacobian
    err_vec_FD : np array
        vector with the error of NR for every iteration, using FD Jacobian
    x_sol : np array
        solution vector, using analytical Jacobian
    iters : int
        total number of iterations, using analytical Jacobian 
    err_vec : np array
        vector with the error of NR for every iteration, using analytical Jacobian
    """
    network.update(x0,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD = network.solve_network(tol,max_iter,h=h,solver='NR_FD',scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x)
    
    network.update(x0,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = network.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x,det_tol=det_tol)
    
    return x_sol_FD,iters_FD,err_vec_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge

def solve_example_e9n():
    """Solve of the example network
    """
    # create network
    elec_net,S_base,V_base,pp_net = create_network()
    scale_var = 'per_unit'
    scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}
    print('scaling parameters: {}'.format(scale_var_params))
    x0 = initialize_network(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # solving using FD and analytical solution
    h = 1e-6
    tol = 1e-6
    max_iter = 50

    elec_net.update(x0,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD = elec_net.solve_network(tol,max_iter,h=h,solver='NR_FD',scale_var=scale_var,scale_var_params=scale_var_params)
    
    elec_net.reset_network(x0,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params) 
    return elec_net,S_base,V_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge

def test_FD_vs_an_e9n():
    # When
    elec_net,S_base,V_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e9n()
    # Then
    assert np.allclose(x_sol_FD,x_sol)
    
def test_delta_e9n():
    # When
    elec_net,S_base,V_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e9n()
    # Then
    delta_sol_pp = pp_net.res_bus['va_degree'].values/180*np.pi
    assert np.allclose(delta_sol_pp,delta_sol)
    
def test_V_e9n():
    # When
    elec_net,S_base,V_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e9n()

    # Then
    V_sol_pp = pp_net.res_bus['vm_pu'].values*V_base
    assert np.allclose(V_sol_pp,V_sol)

def test_Plink_e9n():
    # When
    elec_net,S_base,V_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e9n()

    # Then
    if 'p_from_kw' in pp_net.res_line.keys():
        P_edge_pp = pp_net.res_line['p_from_kw'].append(pp_net.res_line['p_to_kw']).values*kW #[W]
    elif 'p_from_mw' in pp_net.res_line.keys():
        P_edge_pp = pp_net.res_line['p_from_mw'].append(pp_net.res_line['p_to_mw']).values*MW #[W]
    assert np.allclose(P_edge_pp,P_edge)

def test_Qlink_e9n():
    # When
    elec_net,S_base,V_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e9n()

    # Then
    if 'q_from_kvar' in pp_net.res_line.keys():
        Q_edge_pp = pp_net.res_line['q_from_kvar'].append(pp_net.res_line['q_to_kvar']).values*kW #[var]
    elif 'q_from_mvar' in pp_net.res_line.keys():
        Q_edge_pp = pp_net.res_line['q_from_mvar'].append(pp_net.res_line['q_to_mvar']).values*MW #[var]
    assert np.allclose(Q_edge_pp,Q_edge)  
    
def test_S_inj_e9n():
    # When
    elec_net,S_base,V_base,pp_net,x_sol_FD,iters_FD,err_vec_FD,delta_sol_FD,V_sol_FD,S_inj_FD,P_edge_FD,Q_edge_FD,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_example_e9n()

    # Then
    if 'p_kw' in pp_net.res_bus.keys():
        S_inj_pp = (pp_net.res_bus['p_kw'].values + 1j*pp_net.res_bus['q_kvar'].values)*kW #[VA]
    elif 'p_mw' in pp_net.res_bus.keys():
        S_inj_pp = (pp_net.res_bus['p_mw'].values + 1j*pp_net.res_bus['q_mvar'].values)*MW #[VA]
    assert np.allclose(S_inj_pp,S_inj)
    
if __name__ == '__main__':
    # create network
    elec_net,S_base,V_base,pp_net = create_network()
    scale_var = 'per_unit'
    scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}
    
    for node in elec_net.get_nodes():
        if node.half_links:
            print('node {} with V = {} [p.u.], P = {} [p.u.], Q = {} [p.u.]'.format(node.name,node.get_V(scale_var=scale_var,scale_var_params=scale_var_params),node.half_links[0].get_P(scale_var=scale_var,scale_var_params=scale_var_params),node.half_links[0].get_Q(scale_var=scale_var,scale_var_params=scale_var_params)))
        else:
            print('node {} with V = {} [p.u.]'.format(node.name,node.get_V(scale_var=scale_var,scale_var_params=scale_var_params)))
            
    # Solve network in different ways, compare convergence
    h = 1e-6
    tol = 1e-6
    max_iter = 50
    # Unscaled (i.e., in S.I.)
    x0 = initialize_network(elec_net) # S.I.
    print(x0)
    _,_,_,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = solve_system(elec_net,tol,max_iter,h,x0)
    # Unscaled (i.e., in S.I.), with scaled F for stopping criterion
    elec_net.reset_network(x0)
    Fb = S_base * np.ones(len(x0))
    DF = sps.diags(1/Fb)
    _,_,_,x_sol_DF,iters_DF,err_vec_DF,delta_sol_DF,V_sol_DF,S_inj_DF,P_edge_DF,Q_edge_DF = solve_system(elec_net,tol,max_iter,h,x0,D_F=DF)
    # Scaled in solver
    elec_net.reset_network(x0)
    x_entries, unknown_delta_nodes, unknown_V_nodes = elec_net.get_x_entries()
    xb_delta = np.ones(len(unknown_delta_nodes))
    xb_V = V_base*np.ones(len(unknown_V_nodes))
    xb = np.concatenate((xb_delta,xb_V))
    Dx = sps.diags(1/xb)
    _,_,_,x_sol_scaled,iters_scaled,err_vec_scaled,delta_sol_scaled,V_sol_scaled,S_inj_scaled,P_edge_scaled,Q_edge_scaled = solve_system(elec_net,tol,max_iter,h,x0,D_F=DF,D_x=Dx)
    # Scaled (i.e., in p.u.)
    elec_net.reset_network(x0)
    x0_pu = initialize_network(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    _,_,_,x_sol_pu,iters_pu,err_vec_pu,delta_sol_pu,V_sol_pu,S_inj_pu,P_edge_pu,Q_edge_pu = solve_system(elec_net,tol,max_iter,h,x0_pu,scale_var=scale_var,scale_var_params=scale_var_params)
    # Unscaled (i.e., in S.I.), using fsolver
    elec_net.reset_network(x0)
    x_sol_fsolver,iters_fsolver,err_vec_fsolver,_,_,_,_,_= elec_net.solve_network(tol,max_iter,x0,solver='fsolver')
    # Scaled in fsolver
    elec_net.reset_network(x0)
    x_sol_fsolver_scaled,iters_fsolver_scaled,err_vec_fsolver_scaled,_,_,_,_,_= elec_net.solve_network(tol,max_iter,x0,solver='fsolver',D_F=DF,D_x=Dx)
    # per unit in fsolver
    elec_net.reset_network(x0)
    x_sol_fsolver_pu,iters_fsolver_pu,err_vec_fsolver_pu,_,_,_,_,_= elec_net.solve_network(tol,max_iter,x0_pu,scale_var=scale_var,scale_var_params=scale_var_params,solver='fsolver')
    # Unscaled (i.e., in S.I.), using root
    elec_net.reset_network(x0)
    x_sol_root,iters_root,err_vec_root,_,_,_,_,_= elec_net.solve_network(tol,max_iter,x0,solver='root')
    # Scaled in root
    elec_net.reset_network(x0)
    try:
        x_sol_root_scaled,iters_root_scaled,err_vec_root_scaled,_,_,_,_,_= elec_net.solve_network(tol,max_iter,x0,solver='root',D_F=DF,D_x=Dx)
    except:
        iters_root_scaled = 0
    # per unit in root
    elec_net.reset_network(x0)
    x_sol_root_pu,iters_root_pu,err_vec_root_pu,_,_,_,_,_= elec_net.solve_network(tol,max_iter,x0_pu,scale_var=scale_var,scale_var_params=scale_var_params,solver='root')
    
    
    # plot convergence
    fig_conv = plt.figure('Convergence plot')
    ax_conv = fig_conv.gca()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    max_iter_used = np.max([iters,iters_pu,iters_DF,iters_scaled,iters_fsolver,iters_fsolver_scaled,iters_fsolver_pu,iters_root_pu])
    ax_conv.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax_conv.semilogy(np.asarray(range(0,iters+1)),err_vec,'s-',label='unscaled')
    ax_conv.semilogy(np.asarray(range(0,iters_pu+1)),err_vec_pu,'d-',label='p.u.')
    ax_conv.semilogy(np.asarray(range(0,iters_DF+1)),err_vec_DF,'*-',label='unscaled $D_F$')
    ax_conv.semilogy(np.asarray(range(0,iters_scaled+1)),err_vec_scaled,'.-',label='scaled solver')
    ax_conv.semilogy(np.asarray([0,iters_fsolver]),err_vec_fsolver,'o--',label='unscaled fsolver')
    ax_conv.semilogy(np.asarray([0,iters_fsolver_scaled]),err_vec_fsolver_scaled,'s--',label='scaled fsolver')
    ax_conv.semilogy(np.asarray([0,iters_fsolver_pu]),err_vec_fsolver_pu,'^--',label='p.u. fsolver')
    ax_conv.semilogy(np.asarray([0,iters_root]),err_vec_root,'d--',label='unscaled root')
    if iters_root_scaled:
        ax_conv.semilogy(np.asarray([0,iters_root_scaled]),err_vec_root_scaled,'*--',label='scaled root')
    ax_conv.semilogy(np.asarray([0,iters_root_pu]),err_vec_root_pu,'.--',label='p.u. root')
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    
    print(delta_sol)
    print(delta_sol_pu)
    # compare with solution
    delta_sol_pp = pp_net.res_bus['va_degree'].values/180*np.pi #[rad]
    print(delta_sol_pp)
    V_sol_pp = pp_net.res_bus['vm_pu'].values*V_base #[V]
    
    if 'p_from_kw' in pp_net.res_line.keys():
        P_edge_pp = pp_net.res_line['p_from_kw'].append(pp_net.res_line['p_to_kw']).values*kW #[W]
    elif 'p_from_mw' in pp_net.res_line.keys():
        P_edge_pp = pp_net.res_line['p_from_mw'].append(pp_net.res_line['p_to_mw']).values*MW #[W]
    if 'q_from_kvar' in pp_net.res_line.keys():
        Q_edge_pp = pp_net.res_line['q_from_kvar'].append(pp_net.res_line['q_to_kvar']).values*kW #[var]
    elif 'q_from_mvar' in pp_net.res_line.keys():
        Q_edge_pp = pp_net.res_line['q_from_mvar'].append(pp_net.res_line['q_to_mvar']).values*MW #[var]
    
    fig_V, (ax_delta, ax_V) = plt.subplots(2, 1, sharex=True)
    fig_V.canvas.set_window_title('Voltage error')
    ax_delta.set_ylabel('$\delta_{sol} - \delta$ in [rad]')
    ax_delta.plot(delta_sol_pp-delta_sol,'s',label='unscaled')
    ax_delta.plot(delta_sol_pp-delta_sol_pu,'d',label='p.u.')
    ax_delta.plot(delta_sol_pp-delta_sol_DF,'*',label='unscaled $D_F$')
    ax_delta.plot(delta_sol_pp-delta_sol_scaled,'.',label='scaled solver')
    ax_delta.legend()
    ax_delta.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_delta.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_V.set_xlabel('Node number')
    ax_V.set_ylabel('$|V|_{sol} - |V|$ in [V]')
    ax_V.plot(V_sol_pp-V_sol,'s',label='unscaled')
    ax_V.plot(V_sol_pp-V_sol_pu,'d',label='p.u.')
    ax_V.plot(V_sol_pp-V_sol_DF,'*',label='unscaled $D_F$')
    ax_V.plot(V_sol_pp-V_sol_scaled,'.',label='scaled solver')
    ax_V.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_V.grid(which='minor',color='k', linestyle=':', alpha=.05)
    
    fig_Sedge, (ax_Pedge, ax_Qedge) = plt.subplots(2, 1, sharex=True)
    fig_Sedge.canvas.set_window_title('Link Power error')
    ax_Pedge.set_ylabel('$P_{sol} - P$ in [W]')
    ax_Pedge.plot(P_edge_pp-P_edge,'s',label='unscaled')
    ax_Pedge.plot(P_edge_pp-P_edge_pu,'d',label='p.u.')
    ax_Pedge.plot(P_edge_pp-P_edge_DF,'*',label='unscaled $D_F$')
    ax_Pedge.plot(P_edge_pp-P_edge_scaled,'.',label='scaled solver')
    ax_Pedge.legend()
    ax_Pedge.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_Pedge.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_Pedge.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    ax_Qedge.set_xlabel('Link number')
    ax_Qedge.set_ylabel('$Q_{sol} - Q$ in [var]')
    ax_Qedge.plot(Q_edge_pp-Q_edge,'s',label='unscaled')
    ax_Qedge.plot(Q_edge_pp-Q_edge_pu,'d',label='p.u.')
    ax_Qedge.plot(Q_edge_pp-Q_edge_DF,'*',label='unscaled $D_F$')
    ax_Qedge.plot(Q_edge_pp-Q_edge_scaled,'.',label='scaled solver')
    ax_Qedge.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_Qedge.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_Qedge.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    
    plt.show()
    
