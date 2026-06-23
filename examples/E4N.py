"""Example of an electrical network with 4 nodes.

Data taken from pandapower, which took it form PyPower, which took it from MATPOWER. The case is called case4gs.

This is the 4 bus example from pp. 337-338 of "Power System Analysis",
   by John Grainger, Jr., William Stevenson, McGraw-Hill, 1994.
"""
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
import pandapower as pp
import pandapower.networks as ppn
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from meslf.utils.constants import kV, kW, MW, degree
from meslf.load_flow.system_of_equations import NonLinearSystemElectrical
import scipy.optimize as spo
import os
from meslf.utils.hide_print import HiddenPrints
import meslf.utils.basic_math as bm
import time
import ipopt
import warnings

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

colors_method = {'trust-constr':'tab:blue','SLSQP':'tab:orange','ipopt':'tab:green'}
markers_bounds = {'soft':'s','hard':'*'}
linestyles_derivatives = {'num':'--','an':'-','direct':':','adjoint':'-.'}
linestyles_contraints = {'eq':'-','ineq':'--','bound':':'}
marker_size = 10
legend_handles = [Line2D([0], [0], color=colors_method.get('trust-constr'), label='trust-constr'),
    Line2D([0], [0], color=colors_method.get('SLSQP'), label='SLSQP'),
    Line2D([0], [0], color=colors_method.get('ipopt'), label='ipopt'),
    Line2D([0], [0], marker=markers_bounds.get('soft'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Soft constraints'),
    Line2D([0], [0], marker=markers_bounds.get('hard'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'Hard constraints'),
    Line2D([0], [0], color='k',ls=linestyles_derivatives.get('num'), label='Numerical derivatives'),
    Line2D([0], [0], color='k',ls=linestyles_derivatives.get('an'), label='Analytical derivatives'),
    Line2D([0], [0], color='k',ls=linestyles_derivatives.get('direct'), label='Direct approach'),
    Line2D([0], [0], color='k',ls=linestyles_derivatives.get('adjoint'), label='Adjoint approach')]

# load data
pp_net = ppn.case4gs()
if int(pp.__version__[0]) > 1:
    # run power flow to fill the dataframes with the results. Only needed in version 2 or later
    pp.runpp(pp_net,numba=False)
# some base values
if 'sn_kva' in pp_net.keys():
    S_base = pp_net.sn_kva*kW #[va]
elif 'sn_mva' in pp_net.keys():
    S_base = pp_net.sn_mva*MW #[va]
else:
    ValueError('Cannot get network complex power base value!')
V_base = pp_net.bus['vn_kv'][0]*kV #[V]
# solution
P0_load = pp_net.load['p_mw'][0]*MW
Q0_load = pp_net.load['q_mvar'][0]*MW
V0_sol = pp_net.res_bus['vm_pu'][0]*V_base
delta0_sol = pp_net.res_bus['va_degree'][0]*degree
P0_gen_sol = pp_net.res_ext_grid['p_mw'][0]*MW #>0
Q0_gen_sol = pp_net.res_ext_grid['q_mvar'][0]*MW #>0
P3_gen_sol = pp_net.res_gen['p_mw'][0]*MW #>0
Q3_gen_sol = pp_net.res_gen['q_mvar'][0]*MW #>0
#P01_sol = pp_net.res_line['p_from_mw'][0]*MW
#P02_sol = pp_net.res_line['p_from_mw'][1]*MW
#P13_sol = pp_net.res_line['p_from_mw'][2]*MW
#P23_sol = pp_net.res_line['p_from_mw'][3]*MW
#Q01_sol = pp_net.res_line['q_from_mvar'][0]*MW
#Q02_sol = pp_net.res_line['q_from_mvar'][1]*MW
#Q13_sol = pp_net.res_line['q_from_mvar'][2]*MW
#Q23_sol = pp_net.res_line['q_from_mvar'][3]*MW

def create_network():
    """ Create the network based on the pandapower data

    Returns
    -------
    elec_net : ElectricalNetwork
        The electrical network
    """
    # load data and create empty network
    pp_net = ppn.case4gs()
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

    # create nodes
    for ind_n in pp_net.bus.index:
        x = pp_net.bus_geodata['x'][ind_n]
        y = pp_net.bus_geodata['y'][ind_n]
        if ind_n in pp_net.ext_grid['bus'].values: # reference
            ind_s = pp_net.ext_grid.index[pp_net.ext_grid['bus']==ind_n][0]
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),x=x,y=y,node_type=0,V=pp_net.ext_grid['vm_pu'][ind_s]*V_base,delta=pp_net.ext_grid['va_degree'][ind_s]*degree)
        elif ind_n in pp_net.load['bus'].values: # loads / demands
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
                node = ElectricalNode(str(pp_net.bus['name'][ind_n]),x=x,y=y,node_type=1,P=P_inj_gen,V=pp_net.gen['vm_pu'][ind_g]*V_base)
                ElectricalHalfLink(node.name+'_hl_load',start_node=node,P=P_inj,Q=Q_inj) # load
            else: # node with only a load connected to it
                node = ElectricalNode(str(pp_net.bus['name'][ind_n]),x=x,y=y,node_type=2,P=P_inj,Q=Q_inj)
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
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),x=x,y=y,node_type=1,P=P_inj,V=pp_net.gen['vm_pu'][ind_g]*V_base)
        else: # junctions
            node = ElectricalNode(str(pp_net.bus['name'][ind_n]),x=x,y=y,node_type=2,P=0.,Q=0.)
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

    return elec_net,S_base,V_base,pp_net

def update_bc(elec_net,delta0,V0,V3,P3_gen,scale_var=None,scale_var_params=None):
    """Updates the boundary conditions of the electrical network, based on the state variables of the OPF"""
    if scale_var == 'per_unit':
        delta0 = delta0*scale_var_params.get('deltabase')
        V0 = V0*scale_var_params.get('Vbase')
        V3 = V3*scale_var_params.get('Vbase')
        P3_gen = P3_gen*scale_var_params.get('Sbase')
    elec_net.nodes[0].delta = delta0
    elec_net.nodes[0].V = V0
    elec_net.nodes[3].V = V3
    elec_net.nodes[3].half_links[0].P = P3_gen # < 0, since it is a generator
    return elec_net

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

def run_load_flow(scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50):
    """Stead-state load flow analysis of electrical network.
    """
    # create network
    elec_net,S_base,V_base,pp_net = create_network()
    if (scale_var == 'matrix' or scale_var == 'per_unit') and scale_var_params == None:
        scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}
    # initialize
    x0 = initialize_network(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # solve network
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    print('Error is {:.4e}, after {} iterations'.format(err_vec[-1],iters))
    print('Solution:')
    print('delta = {}'.format(delta_sol))
    print('|V| = {} p.u.'.format(V_sol/V_base))
    print('P edge = {} p.u.'.format(P_edge/S_base))
    print('Q edge = {} p.u.'.format(Q_edge/S_base))
    print('S nodal inj = {} p.u.'.format(S_inj/S_base))
    print('P hl = {} p.u.'.format([hl.P/S_base for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} p.u.'.format([hl.Q/S_base for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    return elec_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge,pp_net

def test_delta_e4n():
    """Test the solution of the voltage angle against solution found by pandapower."""
    # Given
    tol = 1e-6
    max_iter = 50
    scale_var = 'matrix' # no parameters are provided, so base values of ppnet are used

    # When
    elec_net, x_sol, iters, err_vec, delta_sol, V_sol, S_inj, P_edge, Q_edge, pp_net = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iter)

    # Then
    delta_sol_pp = pp_net.res_bus['va_degree'].values*degree
    assert np.allclose(delta_sol_pp,delta_sol)

def test_V_e4n():
    # Given
    tol = 1e-6
    max_iter = 50
    scale_var = 'matrix' # no parameters are provided, so base values of ppnet are used

    # When
    elec_net, x_sol, iters, err_vec, delta_sol, V_sol, S_inj, P_edge, Q_edge, pp_net = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iter)

    # Then
    V_base = pp_net.bus['vn_kv'][0]*kV #[V]
    V_sol_pp = pp_net.res_bus['vm_pu'].values*V_base
    assert np.allclose(V_sol_pp,V_sol)

def test_Plink_e4n():
    # Given
    tol = 1e-6
    max_iter = 50
    scale_var = 'matrix' # no parameters are provided, so base values of ppnet are used

    # When
    elec_net, x_sol, iters, err_vec, delta_sol, V_sol, S_inj, P_edge, Q_edge, pp_net = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iter)

    # Then
    if 'p_from_kw' in pp_net.res_line.keys():
        P_edge_pp = pp_net.res_line['p_from_kw'].append(pp_net.res_line['p_to_kw']).values*kW #[W]
    elif 'p_from_mw' in pp_net.res_line.keys():
        P_edge_pp = pp_net.res_line['p_from_mw'].append(pp_net.res_line['p_to_mw']).values*MW #[W]
    assert np.allclose(P_edge_pp,P_edge)

def test_Qlink_e4n():
    # Given
    tol = 1e-6
    max_iter = 50
    scale_var = 'matrix' # no parameters are provided, so base values of ppnet are used

    # When
    elec_net, x_sol, iters, err_vec, delta_sol, V_sol, S_inj, P_edge, Q_edge, pp_net = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iter)

    # Then
    if 'q_from_kvar' in pp_net.res_line.keys():
        Q_edge_pp = pp_net.res_line['q_from_kvar'].append(pp_net.res_line['q_to_kvar']).values*kW #[var]
    elif 'q_from_mvar' in pp_net.res_line.keys():
        Q_edge_pp = pp_net.res_line['q_from_mvar'].append(pp_net.res_line['q_to_mvar']).values*MW #[var]
    assert np.allclose(Q_edge_pp,Q_edge)

def test_S_inj_e4n():
    # Given
    tol = 1e-6
    max_iter = 50
    scale_var = 'matrix' # no parameters are provided, so base values of ppnet are used

    # When
    elec_net, x_sol, iters, err_vec, delta_sol, V_sol, S_inj, P_edge, Q_edge, pp_net = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iter)

    # Then
    if 'p_kw' in pp_net.res_bus.keys():
        S_inj_pp = (pp_net.res_bus['p_kw'].values + 1j*pp_net.res_bus['q_kvar'].values)*kW #[VA]
    elif 'p_mw' in pp_net.res_bus.keys():
        S_inj_pp = (pp_net.res_bus['p_mw'].values + 1j*pp_net.res_bus['q_mvar'].values)*MW #[VA]
    assert np.allclose(S_inj_pp,S_inj)

def objective_function(P0_gen,P3_gen,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=None,scale_var_params=None, Dy=None, Dy_inv=None, Df=None, Dh=None):
    """Define the cost function for OPF

    Parameters
    ----------------
    P0_gen,P3_gen : np array
        Generator powers. Scaled when per unit scaling is used, unscaled otherwise.
    a0, a1, b0, b1, c0, c1 : float
        Parameters. Scaled when per unit scaling is used, unscaled otherwise.

    Returns
    -----------
    f : float
        The value of the cost function
    """
    f = a0 + b0*-P0_gen + c0*P0_gen**2  + a1 + b1*-P3_gen + c1*P3_gen**2
    if scale_var == 'matrix':
        f = Df[0]*f
    return f

def jac_objective(y,P0_ind,P3_ind,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=None,scale_var_params=None, Dy=None, Dy_inv=None, Df=None, Dh=None):
    """Gradient vector / Jacobian of objective function

    Parameters
    ----------------
    P0_gen,P3_gen : np array
        Generator powers. Scaled when per unit scaling is used, unscaled otherwise.
    a0, a1, b0, b1, c0, c1 : float
        Parameters. Scaled when per unit scaling is used, unscaled otherwise.

    Returns
    -----------
    df_dy : np.array
        Derivatives of the objective function. Scaled when per unit scaling or matrix scaling is used.
    """
    P0_gen = y[P0_ind]
    P3_gen = y[P3_ind]
    df_dy = np.zeros(len(y))
    df_dy[P0_ind] = -b0 + 2*c0*P0_gen
    df_dy[P3_ind] = -b1 + 2*c1*P3_gen
    if scale_var == 'matrix':
        df_dy = Df[0]*(df_dy.dot(Dy_inv))
    return df_dy

def hess_objective(y,P0_ind,P3_ind,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=None,scale_var_params=None, Dy=None, Dy_inv=None, Df=None, Dh=None):
    """Hesssian of objective function

    Parameters
    ----------------
    P0_gen,P3_gen : np array
        Generator powers. Scaled when per unit scaling is used, unscaled otherwise.
    a0, a1, b0, b1, c0, c1 : float
        Parameters. Scaled when per unit scaling is used, unscaled otherwise.

    Returns
    -----------
    hess : np.array
        Hessian of the objective function. Scaled when per unit scaling or matrix scaling is used.
    """
    hess_cost_diag = np.zeros(len(y))
    hess_cost_diag[P0_ind] = 2*c0
    hess_cost_diag[P3_ind] = 2*c1
    hess = np.diag(hess_cost_diag)
    if scale_var == 'matrix':
        hess = Df[0]*(np.transpose(Dy_inv).dot(hess.dot(Dy_inv)))
    return hess

def h(y,P0_ind,network=None,nlsys=None,scale_var=None,scale_var_params=None, Dy=None, Dy_inv=None, Df=None, Dh=None, V_BC=False):
    """Equality constraints h(x)=0. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Variables. Scaled when per unit scaling is used, unscaled otherwise.
    network : ElectricalNetwork
        The electrical network that is being optimizes. Assumed to have the updated BC's.
    scale_var : str, optional
        Which scaling is used. Options are 'per_unit', 'matrix', or None. Default is None.

    Returns
    -------
    h : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    P0_gen, Q0_gen, Q3_gen = y[P0_ind:P0_ind+3] # scaled
    # evaluate conservation of energy in the slack and generator node
    xe= xe_from_xopf(y,V_BC=V_BC)
    network.reset_network(xe,scale_var=scale_var,scale_var_params=scale_var_params)
    network.update(xe,scale_var=scale_var,scale_var_params=scale_var_params) # this should set the correct values on the links and half links.
    if scale_var == 'per_unit':
        network.nodes[3].half_links[0].Q = Q3_gen*scale_var_params.get('Sbase')
    else:
        network.nodes[3].half_links[0].Q = Q3_gen
    if scale_var == 'per_unit':
        P0_demand = P0_load/scale_var_params.get('Sbase')
        Q0_demand = Q0_load/scale_var_params.get('Sbase')
    else:
        P0_demand = P0_load
        Q0_demand = Q0_load
    fP0,fQ0 = network.nodes[0].node_law(network=network,scale_var=scale_var,scale_var_params=scale_var_params) + np.array([P0_gen + P0_demand,Q0_gen + Q0_demand]) #Node 0 is a slack node, so it has no half links connected to it (yet)
    _,fQ3 = network.nodes[3].node_law(network=network,scale_var=scale_var,scale_var_params=scale_var_params)
    cons_energy = np.array([fP0,fQ0,fQ3])
    # evaluate load flow equations
    network.reset_network(xe,scale_var=scale_var,scale_var_params=scale_var_params)
    F = nlsys.F(xe)
    h = np.concatenate((cons_energy,F)) # already scaled if per unit is used
    if scale_var == 'matrix':
        h = Dh.dot(h)
    return h

def h_der(y,P0_ind,network=None,nlsys=None,scale_var=None,scale_var_params=None, Dy=None, Dy_inv=None, Df=None, Dh=None, V_BC=False):
    """First derivatives of equality constraints h(x)=0. If per unit scaling is used, y and network are assumed to be scaled already. Hence, no scaling has to be done inside this function.

    Parameters
    ----------
    y : np array
        Variables. Scaled when per unit scaling is used, unscaled otherwise.
    network : ElectricalNetwork
        The electrical network that is being optimizes. Assumed to have the updated BC's.
    scale_var : str, optional
        Which scaling is used. Options are 'per_unit', 'matrix', or None. Default is None.

    Returns
    -------
    dh_dy : np array
        The (nonlinear) equality constraints. Scaled when per unit scaling or matrix scaling is used.
    """
    # evaluate Jacobian of LF equations
    xe = xe_from_xopf(y,V_BC=V_BC)
    network.reset_network(xe,scale_var=scale_var,scale_var_params=scale_var_params)
    J_full = nlsys.J_dense(xe,return_full=True) #This also updates the network
    dh_dy = np.zeros((len(y)-P0_ind,len(y)))
    dh_dy[5,0] = 1 #dH_dP3gen
    dh_dy[0,P0_ind] = 1 #dH_dP0gen
    dh_dy[1,P0_ind+1] = 1 #dH_dQ0gen
    dh_dy[2,P0_ind+2] = 1 #dH_dQ3gen
    N = len(network.nodes) # number of nodes in the network
    F_ind = nlsys.FP + [N+ind for ind in nlsys.FQ]
    G_ind = [0,N,N+3]
    xlf_ind = nlsys.xdelta + [N+ind for ind in nlsys.xV]
    V3_ind = N + 3
    dh_dy[:,1] = J_full[G_ind+F_ind,V3_ind].ravel()#dH_dV3
    dh_dy[:len(G_ind),:][:,5:] = J_full[G_ind,:][:,xlf_ind] #dG_dxlf
    dh_dy[len(G_ind):,:][:,5:] = J_full[F_ind,:][:,xlf_ind] #J_lf
    if scale_var == 'matrix':
        dh_dy = Dh.dot(dh_dy.dot(Dy_inv))
    return dh_dy

def xe_from_xopf(x_opf,V_BC=False):
    if V_BC:
        xe = x_opf[5:]
    else:
        xe = x_opf[7:]
    return xe

def run_optimal_load_flow(P3_gen=-180*MW, delta0=0, V0=230*kV, V3=230*kV, delta1=0, delta2=0, delta3=0, V1=230*kV, V2=230*kV,P3_gen_bounds=np.array([1.5*-P3_gen_sol,0]), delta0_bounds=np.array([-np.pi,np.pi]), V0_bounds=np.array([0.8*V_base,1.2*V_base]), V3_bounds=np.array([0.8*V_base,1.2*V_base]), P0_gen_bounds=np.array([-P0_gen_sol,0]), Q0_gen_bounds=np.array([-Q0_gen_sol,1.5*Q0_gen_sol]), Q3_gen_bounds=np.array([-Q3_gen_sol,1.5*Q3_gen_sol]), delta1_bounds=np.array([-np.pi,np.pi]), delta2_bounds=np.array([-np.pi,np.pi]), delta3_bounds=np.array([-np.pi,np.pi]), V1_bounds=np.array([0.8*V_base,1.2*V_base]), V2_bounds=np.array([0.8*V_base,1.2*V_base]),a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,V_BC=False,ineq_constr='all',derivatives=False,optimization_method='trust-constr',stay_within_bounds=False,fb=None):
    """Run optimal power flow for this network. The constraints on the generator power, and the (default) values for the cost functions are chosen such that the solution to OPF is equal to the solution of LF.

    The initial guesses, boundary conditions, and bounds are unscaled.

    Parameters
    ----------------
    V_BC : bool, optional
        When True, the voltage amplitude and angle of the slack node are taken as boundary conditions for the OPF. If False, they are taken is control variables. Default is False.
    max_iter : int, optional
        Maximum number of iteration used for the initial load flow, and for OPF (for OPF, the number of functions evalutions might be more).
    ineq_constr : str, optional
        Determines on which variables the inequality constraints are imposed. If 'all', the inequality constraints are imposed on both state and control variables. If 'control', the inequality constraints are only imposed on the control variables. Default is 'all'.
    derivatives : bool, optional
        If True, analytical expressions for the gradient and Hessian of the objective function and of the (nonlinear) constraints are used. Otherwise, numerical approximations are used. Default is False.
    """
    if derivatives and not V_BC:
        raise ValueError('OPF with analytical derivatives only implemented if V0 is BC.')
    print('\nRunning OPF for electrical network (V0 as BC: {}, inequality constraints: {}, method: {})'.format(V_BC,ineq_constr,optimization_method))

    # create network
    elec_net,_,_,pp_net = create_network()

    if (scale_var == 'matrix' or scale_var == 'per_unit') and scale_var_params == None:
        scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}

    # update the boundary conditions of the electrical network to match the inital guess of opf
    if V_BC:
        V0 = V0_sol
        delta0 = delta0_sol
    else:
        V0 = V0
        delta0 = delta0
    elec_net = update_bc(elec_net,delta0,V0,V3,P3_gen) #unscaled
    # run load flow ones, to make sure that the initial guess of opf is at least a solution of LF.
    initialize_network(elec_net)
    x_LF0 = np.array([delta1, delta2, delta3, V1, V2]) # unscaled
    elec_net.reset_network(x_LF0) # unscaled
    x_LF,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR') # unscaled
    nlsys = NonLinearSystemElectrical(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # initial guess for OPF (unscaled)
    P0_gen = elec_net.nodes[0].half_links[0].get_P() - P0_load #is a source, so the half link power <0
    Q0_gen = elec_net.nodes[0].half_links[0].get_Q() - Q0_load #is a source, so the half link power <0
    Q3_gen = elec_net.nodes[3].half_links[0].get_Q()
    if V_BC:
        u_init = np.array([P3_gen, V3])
    else:
        u_init = np.array([P3_gen, delta0, V0, V3])
    slack_init = np.array([P0_gen, Q0_gen, Q3_gen])
    x_opf0 = np.concatenate((u_init,slack_init,x_LF))

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        if V_BC:
            ubase = np.array([scale_var_params.get('Sbase'),scale_var_params.get('Vbase')])
        else:
            ubase = np.array([scale_var_params.get('Sbase'),scale_var_params.get('deltabase'),scale_var_params.get('Vbase'),scale_var_params.get('Vbase')])
        slack_base = np.array([scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase')])
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((ubase,slack_base,1/Dx.data[0])))
        DH = np.diag(np.concatenate((np.array([1/scale_var_params.get('Sbase'),1/scale_var_params.get('Sbase'),1/scale_var_params.get('Sbase')]),DF.data[0])))
        Df = np.array([1/fb])
        x_opf0 = Dy.dot(x_opf0) # scale y
    else:
        Dy=np.eye(len(x_opf0))
        Dy_inv=np.eye(len(x_opf0))
        Df=np.eye(1)
        DH=np.eye(len(slack_init)+len(x_LF))
    if scale_var == 'per_unit':
        a0 = a0/fb
        a1 = a1/fb
        b0 = b0/(fb/scale_var_params.get('Sbase'))
        b1 = b1/(fb/scale_var_params.get('Sbase'))
        c0 = c0/(fb/scale_var_params.get('Sbase')**2)
        c1 = c1/(fb/scale_var_params.get('Sbase')**2)

    # define objective function and its derivatives
    def obj(y,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH):
        global x_f_vec
        x_f_vec = y.copy()
        global P3_f_global
        global V3_f_global
        global f_vec_global
        P3_f_global.append(y[0])
        V3_f_global.append(y[1])
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        P0_gen = y[len(u_init)] # <0
        P3_gen = y[0] # <0
        f = objective_function(P0_gen,P3_gen,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)
        f_vec_global.append(f)
        return f
    def obj_grad(y,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        P0_ind = len(u_init)
        P3_ind = 0
        return jac_objective(y,P0_ind,P3_ind,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)

    def obj_hess(y,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        P0_ind = len(u_init)
        P3_ind = 0
        return hess_objective(y,P0_ind,P3_ind,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)

    # define nonlinear equality constraints (conservation of energy in the slack and generator node, and the load flow equations)
    def eq_constr(y,network=elec_net,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH, V_BC=V_BC):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        if V_BC:  # V0 and delta0 are BCs
            P3_gen, V3 = y[:len(u_init)]
            if scale_var == 'per_unit':
                V0 = V0_sol/scale_var_params.get('Vbase')
                delta0 = delta0_sol/scale_var_params.get('deltabase')
            else:
                V0 = V0_sol
                delta0 = delta0_sol
        else:
            P3_gen, delta0, V0, V3 = y[:len(u_init)]
        network = update_bc(network,delta0,V0,V3,P3_gen,scale_var=scale_var,scale_var_params=scale_var_params)
        P0_ind = len(u_init)
        return h(y,P0_ind,network=network,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH, V_BC=V_BC)

    def jac_eq_constr(y,network=elec_net,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH, V_BC=V_BC):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            y = Dy_inv.dot(y)
        # update bc of the network
        if V_BC:  # V0 and delta0 are BCs
            P3_gen, V3 = y[:len(u_init)]
            if scale_var == 'per_unit':
                V0 = V0_sol/scale_var_params.get('Vbase')
                delta0 = delta0_sol/scale_var_params.get('deltabase')
            else:
                V0 = V0_sol
                delta0 = delta0_sol
        else:
            P3_gen, delta0, V0, V3 = y[:len(u_init)]
        network = update_bc(network,delta0,V0,V3,P3_gen,scale_var=scale_var,scale_var_params=scale_var_params)
        # update the vectors with voltage amplitudes and angles, since nlsys.J() only updates the voltage amplitudes and angels that are in x
        if V_BC:
            nlsys.V_vec_mag[3] = V3
        else:
            nlsys.V_vec_ang[0] = delta0
            nlsys.V_vec_mag[0] = V0
            nlsys.V_vec_mag[3] = V3
        P0_ind = len(u_init)
        return h_der(y,P0_ind,network=network,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH, V_BC=V_BC)

    lb_nleq = np.zeros(len(x_LF0)+len(slack_init))
    ub_nleq = np.zeros(len(x_LF0)+len(slack_init))
    if derivatives:
        if optimization_method == 'trust-constr':
            nonlinear_constraint = spo.NonlinearConstraint(eq_constr,lb_nleq,ub_nleq,jac=jac_eq_constr,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            nonlinear_constraint = {'type':'eq','fun':eq_constr,'jac':jac_eq_constr}
    else:
        if optimization_method == 'trust-constr':
            nonlinear_constraint = spo.NonlinearConstraint(eq_constr,lb_nleq,ub_nleq,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            nonlinear_constraint = {'type':'eq','fun':eq_constr}

    # define bounds
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        P3_gen_bounds = P3_gen_bounds/scale_var_params.get('Sbase') # DO NOT use /=, then, the P3_gen_bounds and as such P3_gen_sol is overwritten, such that this value is changed for ANY subsequent calls to this function
        delta0_bounds = delta0_bounds/scale_var_params.get('deltabase')
        V0_bounds = V0_bounds/scale_var_params.get('Vbase')
        V3_bounds = V3_bounds/scale_var_params.get('Vbase')
        P0_gen_bounds = P0_gen_bounds/scale_var_params.get('Sbase')
        Q0_gen_bounds = Q0_gen_bounds/scale_var_params.get('Sbase')
        Q3_gen_bounds = Q3_gen_bounds/scale_var_params.get('Sbase')
        delta1_bounds = delta1_bounds/scale_var_params.get('deltabase')
        delta2_bounds = delta2_bounds/scale_var_params.get('deltabase')
        delta3_bounds = delta3_bounds/scale_var_params.get('deltabase')
        V1_bounds = V1_bounds/scale_var_params.get('Vbase')
        V2_bounds = V2_bounds/scale_var_params.get('Vbase')
    if ineq_constr == 'all':
        # x_opf = [P3_gen, (delta0, V0), V3, P0_gen, Q0_gen, Q3_gen, delta1, delta2, delta3, V1, V2]
        if V_BC:
            lb_ineq = np.array([P3_gen_bounds[0], V3_bounds[0], P0_gen_bounds[0], Q0_gen_bounds[0], Q3_gen_bounds[0], delta1_bounds[0], delta2_bounds[0], delta3_bounds[0], V1_bounds[0], V2_bounds[0]])
            ub_ineq = np.array([P3_gen_bounds[1], V3_bounds[1], P0_gen_bounds[1], Q0_gen_bounds[1], Q3_gen_bounds[1], delta1_bounds[1], delta2_bounds[1], delta3_bounds[1], V1_bounds[1], V2_bounds[1]])
        else:
            lb_ineq = np.array([P3_gen_bounds[0], delta0_bounds[0], V0_bounds[0], V3_bounds[0], P0_gen_bounds[0], Q0_gen_bounds[0], Q3_gen_bounds[0], delta1_bounds[0], delta2_bounds[0], delta3_bounds[0], V1_bounds[0], V2_bounds[0]])
            ub_ineq = np.array([P3_gen_bounds[1], delta0_bounds[1], V0_bounds[1], V3_bounds[1], P0_gen_bounds[1], Q0_gen_bounds[1], Q3_gen_bounds[1], delta1_bounds[1], delta2_bounds[1], delta3_bounds[1], V1_bounds[1], V2_bounds[1]])
    elif ineq_constr == 'control':
        lb_ineq = -np.inf*np.ones(len(x_opf0)) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
        ub_ineq = np.inf*np.ones(len(x_opf0))
        if V_BC:
            lb_ineq = np.array([P3_gen_bounds[0], V3_bounds[0]])
            ub_ineq = np.array([P3_gen_bounds[1], V3_bounds[1]])
        else:
            lb_ineq = np.array([P3_gen_bounds[0], delta0_bounds[0], V0_bounds[0], V3_bounds[0]])
            ub_ineq = np.array([P3_gen_bounds[1], delta0_bounds[1], V0_bounds[1], V3_bounds[1]])
    else:
        raise ValueError('Enter valid value for ineq_constr')
    if optimization_method == 'ipopt':
        bounds = [(lb,ub) for lb, ub in zip(lb_ineq,ub_ineq)]
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
    else:
        bounds = spo.Bounds(lb_ineq,ub_ineq,keep_feasible=stay_within_bounds)

    # make sure initial guess satisfies bounds (NB. If adjustments are made, LF is not necessarily satisfied anymore)
    if optimization_method == 'SLSQP' or stay_within_bounds:
        for ind, x0 in enumerate(x_opf0):
            if lb_ineq[ind] > x0:
                x_opf0[ind] = lb_ineq[ind]
            elif ub_ineq[ind] < x0:
                x_opf0[ind] = ub_ineq[ind]

    f_vec = list()
    P3_vec = list()
    V3_vec = list()
    global P3_f_global
    global V3_f_global
    global f_vec_global
    global x_f_vec
    P3_f_global = list()
    V3_f_global = list()
    f_vec_global = list()
    x_f_vec = list()
    if optimization_method == 'trust-constr':
        def callback(xk, state):
            """Called after every iteration"""
            f_vec.append(state.fun)
            P3_vec.append(xk[0])
            V3_vec.append(xk[1])
            return False
    elif optimization_method == 'SLSQP':
        f_vec.append(obj(x_opf0))
        P3_vec.append(x_opf0[0])
        V3_vec.append(x_opf0[1])
        def callback(xk):
            """Called after every iteration"""
            f_vec.append(obj(xk))
            P3_vec.append(xk[0])
            V3_vec.append(xk[1])
            return False
    elif optimization_method == 'ipopt':
        # callback is not implemented in the ipopt (cyipopt) package / wrapper.
        pass

    # solve OPF
    opf_start_time = time.time()
    try:
        if derivatives:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, x_opf0, method=optimization_method,jac=obj_grad,hess=obj_hess, constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds, callback=callback)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, x_opf0, method=optimization_method,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, x_opf0,jac=obj_grad, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
        else:
            if optimization_method == 'trust-constr':
                res = spo.minimize(obj, x_opf0, method=optimization_method, constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds, callback=callback)
                execution_time = res.execution_time
            elif optimization_method == 'SLSQP':
                res = spo.minimize(obj, x_opf0, method=optimization_method, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
                execution_time = opf_start_time - time.time()
            elif optimization_method == 'ipopt':
                res = ipopt.minimize_ipopt(obj, x_opf0, constraints=nonlinear_constraint, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
                execution_time = opf_start_time - time.time()
    except:
        print('Exception made for {}, hard bounds: {}, analytical der.: {}'.format(optimization_method,stay_within_bounds,derivatives))
        if len(f_vec) == 0:
            obj(x_opf0)
            nit = 0
            nfev = 0
            njev = 0
            nhev = 0
        else:
            nit = 0
            nfev = len(f_vec)
            njev = 0
            nhev = 0
        execution_time = opf_start_time - time.time()
        res = spo.OptimizeResult({'success':False,'x':np.array(x_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

    if optimization_method == 'ipopt':
        if res.nit > 0:
            P3_vec = [P3_f_global[ind] for ind in range(0,len(P3_f_global),round(len(P3_f_global)/res.nit))]
            V3_vec = [V3_f_global[ind] for ind in range(0,len(V3_f_global),round(len(V3_f_global)/res.nit))]
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            P3_vec = P3_f_global
            V3_vec = dphi2_f_global
            f_vec = f_vec_global

    if scale_var == 'matrix' or scale_var == 'per_unit':
        x_opf = Dy_inv.dot(res.x)
    else:
        x_opf = res.x

    # print solution
    xe_opt = xe_from_xopf(x_opf,V_BC=V_BC)
    delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.update_full(xe_opt)
    print('Solution OPF (success optimizer: {}):'.format(res.success))
    print('delta = {}'.format(delta_sol))
    print('|V| = {} p.u.'.format(V_sol/V_base))
    print('P edge = {} p.u.'.format(P_edge/S_base))
    print('Q edge = {} p.u.'.format(Q_edge/S_base))
    print('S nodal inj = {} p.u.'.format(S_inj/S_base))
    print('P hl = {} p.u.'.format([hl.P/S_base for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} p.u.'.format([hl.Q/S_base for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    return xe_opt, res, f_vec, P3_vec, V3_vec, execution_time

def test_opf_Vbc():
    """Test OPF againts the solution of LF (the constraints and cost function parameteres are chosen such that this should be the case), using the voltage angle and amplitude of the slack node as boundary condition."""
    # Given + When
    optimization_method='trust-constr'
    V_BC = True
    ineq_constr = 'all'
    scale_var = None
    tol=1e-6
    max_iter=400
    xe_opt, _, _, _, _, _ = run_optimal_load_flow(P3_gen=-180*MW, V3=230*kV, delta1=0, delta2=0, delta3=0, V1=230*kV, V2=230*kV,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=scale_var,tol=tol,max_iter=max_iter,V_BC=V_BC,ineq_constr=ineq_constr,optimization_method=optimization_method)

    # Then
    _, xe_LF, _, _, _, _, _, _, _, _ = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iter)
    assert np.allclose(xe_opt,xe_LF)

def test_opf_Vbc_scaled():
    """Test OPF againts the solution of LF (the constraints and cost function parameteres are chosen such that this should be the case), using the voltage angle and amplitude of the slack node as boundary condition."""
    # Given + When
    optimization_method='trust-constr'
    V_BC = True
    ineq_constr = 'all'
    scale_var = 'matrix'
    fb = 1*MW
    tol=1e-6
    max_iter=400
    xe_opt, _, _, _, _, _ = run_optimal_load_flow(P3_gen=-180*MW, V3=230*kV,  delta1=0, delta2=0, delta3=0, V1=230*kV, V2=230*kV,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=scale_var,tol=tol,max_iter=max_iter,V_BC=V_BC,ineq_constr=ineq_constr,optimization_method=optimization_method,fb=fb)

    # Then
    _, xe_LF, _, _, _, _, _, _, _, _ = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iter)
    assert np.allclose(xe_opt,xe_LF)

def test_opf_Vbc_derivatives():
    """Test OPF againts the solution of LF (the constraints and cost function parameteres are chosen such that this should be the case),, using analytical expressions for the gradients and hessians. The voltage angle and amplitude of the slack node are taken as boundary condition."""
    # Given + When
    optimization_method='trust-constr'
    V_BC = True
    ineq_constr = 'all'
    derivatives = True
    scale_var = None
    tol=1e-6
    max_iter=400
    xe_opt, res, _, _, _, _ = run_optimal_load_flow(P3_gen=-180*MW, V3=230*kV, delta1=0, delta2=0, delta3=0, V1=230*kV, V2=230*kV,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=scale_var,tol=tol,max_iter=max_iter,V_BC=V_BC,ineq_constr=ineq_constr,derivatives=derivatives,optimization_method=optimization_method)

    print('Succes: {}, message: {}'.format(res.success,res.message))
    # Then
    _, xe_LF, _, _, _, _, _, _, _, _ = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iter)
    assert np.allclose(xe_opt,xe_LF)

def test_opf_Vbc_derivatives_scaled():
    """Test OPF againts the solution of LF (the constraints and cost function parameteres are chosen such that this should be the case),, using analytical expressions for the gradients and hessians. The voltage angle and amplitude of the slack node are taken as boundary condition."""
    # Given + When
    optimization_method='trust-constr'
    V_BC = True
    ineq_constr = 'all'
    derivatives = True
    scale_var = 'matrix'
    fb = 1*MW
    tol=1e-6
    max_iter=400
    xe_opt, _, _, _, _, _ = run_optimal_load_flow(P3_gen=-180*MW, V3=230*kV, delta1=0, delta2=0, delta3=0, V1=230*kV, V2=230*kV,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=scale_var,tol=tol,max_iter=max_iter,V_BC=V_BC,ineq_constr=ineq_constr,derivatives=derivatives,optimization_method=optimization_method,fb=fb)

    # Then
    _, xe_LF, _, _, _, _, _, _, _, _ = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iter)
    assert np.allclose(xe_opt,xe_LF)

def test_opf_Vbc_derivatives_scaled_pu():
    """Test OPF againts the solution of LF (the constraints and cost function parameteres are chosen such that this should be the case),, using analytical expressions for the gradients and hessians. The voltage angle and amplitude of the slack node are taken as boundary condition."""
    # Given + When
    optimization_method='trust-constr'
    V_BC = True
    ineq_constr = 'all'
    derivatives = True
    scale_var = 'per_unit'
    scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}
    fb = 1*MW
    tol=1e-6
    max_iter=400
    xe_opt, res, _, _, _, _ = run_optimal_load_flow(P3_gen=-180*MW, V3=230*kV, delta1=0, delta2=0, delta3=0, V1=230*kV, V2=230*kV,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=scale_var,scale_var_params=scale_var_params,tol=tol,max_iter=max_iter,V_BC=V_BC,ineq_constr=ineq_constr,derivatives=derivatives,optimization_method=optimization_method,fb=fb) # xe_opt is unscaled.

    print('Succes: {}, message: {}'.format(res.success,res.message))
    # Then
    _, xe_LF, _, _, _, _, _, _, _, _ = run_load_flow(tol=tol,max_iter=max_iter)
    print('maximum error = {}'.format(error(xe_opt,xe_LF)))
    assert res.success and np.allclose(xe_opt,xe_LF)

def run_optimal_load_flow_separate_LF_explicit(P3_gen=-180*MW, delta0=0, V0=230*kV, V3=230*kV, P0_gen=-150*MW, Q0_gen=-100*MW, Q3_gen=-140*MW, delta1=0, delta2=0, delta3=0, V1=230*kV, V2=230*kV,a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,max_iters_lf=10,V_BC=False):
    """Run optimal power flow for this network, but the load flow equations are taken out of the optimal flow problem. That is, they are solved seperately, and then substituted in the objective function. The gradients and Hessian are determined using numerical methods. The constraints on the generator power, and the (default) values for the cost functions are chosen such that the solution to OPF is equal to the solution of LF.

    Parameters
    ----------------
    V_BC : bool, optional
        When True, the voltage amplitude and angle of the slack node are taken as boundary conditions for the OPF. If False, they are taken is control variables. Default is False.
    max_iter : int, optional
        Maximum number of iteration used for the OPF (for OPF, the number of functions evalutions might be more).
    max_iters_lf : int, optional
        Maximum number of iteration used for steady-state load flow
    """
    print('\nRunning OPF, with LF separate, for electrical network (V0 as BC: {})'.format(V_BC))
    # create network
    elec_net,S_base,V_base,pp_net = create_network()
    P0_load = pp_net.load['p_mw'][0]*MW
    Q0_load = pp_net.load['q_mvar'][0]*MW
    if scale_var == 'matrix' and scale_var_params == None:
        scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}

    # update the boundary conditions of the electrical network to match the inital guess of opf
    if V_BC:
        V0 = pp_net.res_bus['vm_pu'][0]*V_base
        delta0 = pp_net.res_bus['va_degree'][0]*degree
    else:
        V0 = V0
        delta0 = delta0
    elec_net = update_bc(elec_net,delta0,V0,V3,P3_gen)
    # run load flow ones, to make sure that the initial guess of opf is at least a solution of LF
    x0 = initialize_network(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    x_LF0 = np.array([delta1, delta2, delta3, V1, V2])
    elec_net.reset_network(x_LF0)
    x_LF,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iters_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)

    # initial guess for OPF
    P0_gen = elec_net.nodes[0].half_links[0].get_P() - P0_load #is a source, so the half link power <0
    Q0_gen = elec_net.nodes[0].half_links[0].get_Q() - Q0_load #is a source, so the half link power <0
    Q3_gen = elec_net.nodes[3].half_links[0].get_Q()
    if V_BC:
        u_init = np.array([P3_gen, V3])
    else:
        u_init = np.array([P3_gen, delta0, V0, V3])
    slack_init = np.array([P0_gen, Q0_gen, Q3_gen])
    x_opf0 = np.concatenate((u_init,slack_init))

    def cost_function(x_opf):
        """Define the cost function for OPF

        Parameters
        ----------------
        x_opf : np array
            Variable vector used in OPF. Is assumed to be [P3_gen, delta0, V0, V3, P0_gen, Q0_gen, Q3_gen, delta1, delta2, delta3, V1, V2]

        Returns
        -----------
        f : float
            The value of the cost function
        """
        P0_gen = x_opf[len(u_init)] # <0
        P3_gen = x_opf[0] # <0
        return a0 + b0*-P0_gen + c0*P0_gen**2  + a1 + b1*-P3_gen + c1*P3_gen**2

    # define (nonlinear?) equality constraints
    def nonlinear_equality_constraints(x_opf,network=elec_net):
        # update bc of the network
        if V_BC:  # V0 and delta0 are BCs
            P3_gen, V3 = x_opf[:len(u_init)]
            V0 = pp_net.res_bus['vm_pu'][0]*V_base
            delta0 = pp_net.res_bus['va_degree'][0]*degree
        else:
            P3_gen, delta0, V0, V3 = x_opf[:len(u_init)]
        network = update_bc(network,delta0,V0,V3,P3_gen)
        # solve load flow equations
        network.reset_network(x_LF0,scale_var=scale_var,scale_var_params=scale_var_params)
        x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = network.solve_network(tol,max_iters_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        # determine active and reactive power of source at the slack, and the reactive power of the source at the generator node
        P0_gen_LF = network.nodes[0].half_links[0].get_P() - P0_load #is a source, so the half link power <0
        Q0_gen_LF = network.nodes[0].half_links[0].get_Q() - Q0_load#is a source, so the half link power <0
        Q3_gen_LF = network.nodes[3].half_links[0].get_Q()
        slack_LF = np.array([P0_gen_LF,Q0_gen_LF,Q3_gen_LF])
        # current values of powers
        slack = x_opf[len(u_init):len(u_init)+len(slack_init)]
        return slack - slack_LF
    lb_nleq = np.zeros(len(slack_init))
    ub_nleq = np.zeros(len(slack_init))
    nonlinear_constraint = spo.NonlinearConstraint(nonlinear_equality_constraints,lb_nleq,ub_nleq)

    # define linear inequality constraints (based on the LF solution of pandapower)
    P3_gen_sol = pp_net.res_gen['p_mw'][0]*MW #>0
    P3_gen_lb = -1.5*P3_gen_sol
    P3_gen_ub = 0
    delta0_lb = -np.pi
    delta0_ub = np.pi
    V0_lb = 0.8*V_base
    V0_ub = 1.2*V_base
    V3_lb = 0.8*V_base
    V3_ub = 1.2*V_base
    lb_ineq = -np.inf*np.ones(len(x_opf0)) # bounds need to be of the same length as x. Use infinity as bound when you want unconstrained.
    ub_ineq = np.inf*np.ones(len(x_opf0))
    if V_BC:
        lb_ineq[:len(u_init)]  = np.array([P3_gen_lb,V3_lb])
        ub_ineq[:len(u_init)]  = np.array([P3_gen_ub,V3_ub])
    else:
        lb_ineq[:len(u_init)]  = np.array([P3_gen_lb,delta0_lb,V0_lb,V3_lb])
        ub_ineq[:len(u_init)]  = np.array([P3_gen_ub,delta0_ub,V0_ub,V3_ub])
    bounds = spo.Bounds(lb_ineq,ub_ineq)

    # solve OPF
    res = spo.minimize(cost_function, x_opf0, method='trust-constr', constraints=[nonlinear_constraint], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds)
    x_opf = res.x
    # print solution
    xe_opt = elec_net.set_x_init()
    elec_net.reset_network(xe_opt) # if you don't do this first, the update_full will set the wrong values for the half link powers.
    delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.update_full(xe_opt)
    print('Solution OPF:')
    print('delta = {}'.format(delta_sol))
    print('|V| = {} p.u.'.format(V_sol/V_base))
    print('P edge = {} p.u.'.format(P_edge/S_base))
    print('Q edge = {} p.u.'.format(Q_edge/S_base))
    print('S nodal inj = {} p.u.'.format(S_inj/S_base))
    print('P hl = {} p.u.'.format([hl.P/S_base for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} p.u.'.format([hl.Q/S_base for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    return x_opf, res.fun,res.nfev,res.nit,res.execution_time

def solve_lf_in_lf(network,u,max_iters=10,V_BC=False,scale_var=None,scale_var_params=None):
    """Solve steady-state LF within an optmization context.

    Parameters
    ----------
    u : np arrays
        Vector with control variables. Scaled when using per unit scaling, unscaled otherwise
    """
    if V_BC:  # V0 and delta0 are BCs
        P3_gen, V3 = u
        if scale_var == 'per_unit':
            V0 = V0_sol/scale_var_params.get('Vbase')
            delta0 = delta0_sol/scale_var_params.get('deltabase')
        else:
            V0 = V0_sol
            delta0 = delta0_sol
    else:
        P3_gen, delta0, V0, V3 = u
    xe_init = network.set_x_init(scale_var=scale_var,scale_var_params=scale_var_params)
    network.reset_network(xe_init,scale_var=scale_var,scale_var_params=scale_var_params)
    network = update_bc(network,delta0,V0,V3,P3_gen,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = network.solve_network(tol,max_iters,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    if scale_var == 'per_unit':
        P0_demand = P0_load/scale_var_params.get('Sbase')
    else:
        P0_demand = P0_load
    P0_gen = network.nodes[0].half_links[0].get_P(scale_var=scale_var,scale_var_params=scale_var_params) - P0_demand #Scaled when per unit is used. Is a source, so the half link power <0
    return P0_gen, network

def run_optimal_load_flow_separate_LF(P3_gen=-180*MW, delta0=0, V0=230*kV, V3=230*kV, delta1=0, delta2=0, delta3=0, V1=230*kV, V2=230*kV,P3_gen_bounds=np.array([1.5*-P3_gen_sol,0]), delta0_bounds=np.array([-np.pi,np.pi]), V0_bounds=np.array([0.8*V_base,1.2*V_base]), V3_bounds=np.array([0.8*V_base,1.2*V_base]), P0_gen_bounds=np.array([-P0_gen_sol,0]), Q0_gen_bounds=np.array([-Q0_gen_sol,1.5*Q0_gen_sol]), Q3_gen_bounds=np.array([-Q3_gen_sol,1.5*Q3_gen_sol]), delta1_bounds=np.array([-np.pi,np.pi]), delta2_bounds=np.array([-np.pi,np.pi]), delta3_bounds=np.array([-np.pi,np.pi]), V1_bounds=np.array([0.8*V_base,1.2*V_base]), V2_bounds=np.array([0.8*V_base,1.2*V_base]),a0=0,a1=0,b0=2,b1=3,c0=1,c1=2,scale_var=None,scale_var_params=None,tol=1e-6,max_iter=50,max_iters_lf=10,V_BC=False,ineq_constr='all',optimization_method='trust-constr',stay_within_bounds=False,fb=None,approach='direct'):
    """Optimal load flow where the LF is included implicitely. The gradients are determined analytically, either using a direct or an adjoint approach. The constraints on the generator power, and the (default) values for the cost functions are chosen such that the solution to OPF is equal to the solution of LF.

    Parameters
    ----------------
    V_BC : bool, optional
        When True, the voltage amplitude and angle of the slack node are taken as boundary conditions for the OPF. If False, they are taken is control variables. Default is False.
    max_iter : int, optional
        Maximum number of iteration used for the OPF (for OPF, the number of functions evalutions might be more).
    max_iters_lf : int, optional
        Maximum number of iteration used for steady-state load flow
    approach : str, optional
        Approach used to compute the gradient and Jacobians. Either 'direct' or 'adjoint'. Default is 'direct'.
    """
    if not V_BC:
        raise ValueError('OPF with separate LF with direct or adjoint approach only implented if V0 is taken as BC')
    print('\nRunning OPF, with LF separate and {} approach, for electrical network (V0 as BC: {}, method: {}, ineq. constr.: {}, bounds: {}, scaling: {})'.format(approach,V_BC,optimization_method,ineq_constr,stay_within_bounds,scale_var))

    # create network
    elec_net,_,_,pp_net = create_network()

    if scale_var == 'matrix' and scale_var_params == None:
        scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}
    # scale the bounds
    if scale_var == 'matrix' or scale_var == 'per_unit': # y is scaled
        P3_gen_bounds = P3_gen_bounds/scale_var_params.get('Sbase') # DO NOT use /=, then, the P3_gen_bounds and as such P3_gen_sol is overwritten, such that this value is changed for ANY subsequent calls to this function
        delta0_bounds = delta0_bounds/scale_var_params.get('deltabase')
        V0_bounds = V0_bounds/scale_var_params.get('Vbase')
        V3_bounds = V3_bounds/scale_var_params.get('Vbase')
        P0_gen_bounds = P0_gen_bounds/scale_var_params.get('Sbase')
        Q0_gen_bounds = Q0_gen_bounds/scale_var_params.get('Sbase')
        Q3_gen_bounds = Q3_gen_bounds/scale_var_params.get('Sbase')
        delta1_bounds = delta1_bounds/scale_var_params.get('deltabase')
        delta2_bounds = delta2_bounds/scale_var_params.get('deltabase')
        delta3_bounds = delta3_bounds/scale_var_params.get('deltabase')
        V1_bounds = V1_bounds/scale_var_params.get('Vbase')
        V2_bounds = V2_bounds/scale_var_params.get('Vbase')

    # update the boundary conditions of the electrical network to match the inital guess of opf. Initialize network
    if V_BC:
        V0 = V0_sol
        delta0 = delta0_sol
    else:
        V0 = V0
        delta0 = delta0
    elec_net = update_bc(elec_net,delta0,V0,V3,P3_gen) #unscaled
    elec_net.initialize() # needed to create admittance matrix
    nlsys = NonLinearSystemElectrical(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # initial guess for OPF (unscaled)
    u0 = np.array([P3_gen,V3])

    # scaling
    if scale_var == 'matrix' or scale_var == 'per_unit':
        Dx = nlsys.Dx()
        Dx_inv = np.diag(1/Dx.data[0])
        DF = nlsys.DF()
        if V_BC:
            ubase = np.array([scale_var_params.get('Sbase'),scale_var_params.get('Vbase')])
        else:
            ubase = np.array([scale_var_params.get('Sbase'),scale_var_params.get('deltabase'),scale_var_params.get('Vbase'),scale_var_params.get('Vbase')])
        slack_base = np.array([scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase')])
        Dy = np.diag(np.concatenate((1/ubase,1/slack_base,Dx.data[0])))
        Dy_inv = np.diag(np.concatenate((ubase,slack_base,1/Dx.data[0])))
        Du = np.diag(1/ubase)
        Du_inv = np.diag(ubase)
        DH = np.diag(np.concatenate((np.array([1/scale_var_params.get('Sbase'),1/scale_var_params.get('Sbase'),1/scale_var_params.get('Sbase')]),DF.data[0])))
        Df = np.array([1/fb])
        u0 = Du.dot(u0) # scale u
    else:
        if V_BC:
            Dy=np.eye(10)
            Dy_inv=np.eye(10)
            Du=np.eye(2)
            Du_inv=np.eye(2)
        else:
            Dy=np.eye(12)
            Dy_inv=np.eye(12)
            Du=np.eye(4)
            Du_inv=np.eye(5)
        Df=np.eye(1)
        DH=np.eye(8)
    if scale_var == 'per_unit':
        a0 = a0/fb
        a1 = a1/fb
        b0 = b0/(fb/scale_var_params.get('Sbase'))
        b1 = b1/(fb/scale_var_params.get('Sbase'))
        c0 = c0/(fb/scale_var_params.get('Sbase')**2)
        c1 = c1/(fb/scale_var_params.get('Sbase')**2)

    def obj(u,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,scale_var=scale_var,scale_var_params=scale_var_params,network=elec_net,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH,Du=Du,Du_inv=Du_inv):
        """Define the cost function for OPF

        Parameters
        ----------------
        x_opf : np array
            Variable vector used in OPF. Is assumed to be [P3_gen, delta0, V0, V3, P0_gen, Q0_gen, Q3_gen, delta1, delta2, delta3, V1, V2]

        Returns
        -----------
        f : float
            The value of the cost function
        """
        global x_f_vec
        x_f_vec = u.copy()
        global P3_f_global
        global V3_f_global
        global f_vec_global
        P3_f_global.append(u[0])
        V3_f_global.append(u[1])
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            u = Du_inv.dot(u)
        P3_gen = u[0]
        P0_gen, network = solve_lf_in_lf(network,u,max_iters=max_iters_lf,V_BC=V_BC,scale_var=scale_var,scale_var_params=scale_var_params)
        f = objective_function(P0_gen,P3_gen,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)
        f_vec_global.append(f)
        return f
    # gradient of objective function
    def obj_grad(u,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,scale_var=scale_var,scale_var_params=scale_var_params,network=elec_net,nlsys=nlsys,method=approach,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH,Du=Du,Du_inv=Du_inv):
        if scale_var == 'matrix':
            # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
            u = Du_inv.dot(u)
        # update network and solve LF
        P3_gen, V3 = u
        P0_gen, network = solve_lf_in_lf(network,u,max_iters=max_iters_lf,V_BC=V_BC,scale_var=scale_var,scale_var_params=scale_var_params)
        P0_ind = len(u)
        P3_ind = 0
        if scale_var_params == 'per_unit':
            Q0_demand = Q0_load/scale_var_params.get('Sbase')
            V0 = V0_sol/scale_var_params.get('Vbase')
            delta0 = delta0_sol/scale_var_params.get('deltabase')
        else:
            Q0_demand = Q0_load
            V0 = V0_sol
            delta0 = delta0_sol
        Q0_gen = network.nodes[0].half_links[0].get_Q(scale_var=scale_var,scale_var_params=scale_var_params) - Q0_demand
        Q3_gen = network.nodes[3].half_links[0].get_Q(scale_var=scale_var,scale_var_params=scale_var_params)
        # partial derivatives of objective
        x_LF = network.set_x_init(scale_var=scale_var,scale_var_params=scale_var_params)
        y = np.concatenate((u,np.array([P0_gen,Q0_gen,Q3_gen]),x_LF))
        deltaf_deltay = jac_objective(y,P0_ind,P3_ind,a0=a0,b0=b0,c0=c0,a1=a1,b1=b1,c1=c1,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)
        deltaf_deltau = np.zeros((1,2))
        deltaf_deltax = np.zeros((1,8))
        deltaf_deltau[0,:] = deltaf_deltay[:len(u)]
        deltaf_deltax[0,:] = deltaf_deltay[len(u):]
        # partial derivatives of equatilty constraints / load-flow equations
        if V_BC:
            nlsys.V_vec_mag[3] = V3
        else:
            nlsys.V_vec_ang[0] = delta0
            nlsys.V_vec_mag[0] = V0
            nlsys.V_vec_mag[3] = V3
        deltah_deltay = h_der(y,P0_ind,network=network,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params, V_BC=V_BC,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)
        deltah_deltau = deltah_deltay[:,:len(u)]
        deltah_deltax = deltah_deltay[:,len(u):]
        # gradient objective
        df_du = deltaf_deltau.copy() # first part of gradient
        if method == 'direct':
            w = np.linalg.solve(deltah_deltax,-deltah_deltau)
            df_du += np.dot(deltaf_deltax,w)
        elif method == 'adjoint':
            v = np.linalg.solve(np.transpose(deltah_deltax),np.transpose(deltaf_deltax))
            df_du += np.dot(np.transpose(v),-deltah_deltau)
        df_du = df_du.ravel() # jac needs to be 'array_like, shape (n,)'
        # print('df_du={}\ndeltaf_deltau={}\ndeltaf_deltax={}\ndeltah_deltau={}\ndeltah_deltax={}'.format(df_du,deltaf_deltau,deltaf_deltax,deltah_deltau,deltah_deltax))
        return df_du

    if ineq_constr == 'all':
        lb_ineq_state = np.array([P0_gen_bounds[0], Q0_gen_bounds[0], Q3_gen_bounds[0], delta1_bounds[0], delta2_bounds[0], delta3_bounds[0], V1_bounds[0], V2_bounds[0]])
        ub_ineq_state = np.array([P0_gen_bounds[1], Q0_gen_bounds[1], Q3_gen_bounds[1], delta1_bounds[1], delta2_bounds[1], delta3_bounds[1], V1_bounds[1], V2_bounds[1]])
        # define inequality constraints
        def g(u,scale_var=scale_var,scale_var_params=scale_var_params,network=elec_net,Dy=Dy, Dy_inv=Dy_inv, Du=Du,Du_inv=Du_inv):
            """Determine the nonlinear inequality constraints g(x(u)) >= 0"""
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                u = Du_inv.dot(u)
            # update network and solve LF
            P3_gen, V3 = u
            P0_gen, network = solve_lf_in_lf(network,u,max_iters=max_iters_lf,V_BC=V_BC,scale_var=scale_var,scale_var_params=scale_var_params)
            P0_ind = len(u)
            P3_ind = 0
            if scale_var_params == 'per_unit':
                Q0_demand = Q0_load/scale_var_params.get('Sbase')
                V0 = V0_sol/scale_var_params.get('Vbase')
                delta0 = delta0_sol/scale_var_params.get('deltabase')
            else:
                Q0_demand = Q0_load
                V0 = V0_sol
                delta0 = delta0_sol
            Q0_gen = network.nodes[0].half_links[0].get_Q(scale_var=scale_var,scale_var_params=scale_var_params) - Q0_demand
            Q3_gen = network.nodes[3].half_links[0].get_Q(scale_var=scale_var,scale_var_params=scale_var_params)
            x_LF = network.set_x_init(scale_var=scale_var,scale_var_params=scale_var_params)
            network.reset_network(x_LF,scale_var=scale_var,scale_var_params=scale_var_params)
            x = np.concatenate((np.array([P0_gen,Q0_gen,Q3_gen]),x_LF))
            if scale_var == 'matrix': # lb_ineq_state and ub_ineq_state are scaled, so scale x as wel
                x = Dy[len(u):,len(u):].dot(x)
            g = np.concatenate((x-lb_ineq_state,ub_ineq_state-x))
            return g
        def g_jac(u,scale_var=scale_var,scale_var_params=scale_var_params,network=elec_net,nlsys=nlsys,method=approach,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH,Du=Du,Du_inv=Du_inv):
            """Jacobian of inequality constraints"""
            if scale_var == 'matrix':
                # Optimizer uses scaled x, but objective function wants unscaled x (when using matrix scaling)
                u = Du_inv.dot(u)
            # Jacobian of inequality constraints wrt state variables x
            deltag_deltax = np.vstack((np.eye(8),-np.eye(8)))
            deltag_deltau = np.zeros((16,2))
            # update network and solve LF
            P3_gen, V3 = u
            P0_gen, network = solve_lf_in_lf(network,u,max_iters=max_iters_lf,V_BC=V_BC,scale_var=scale_var,scale_var_params=scale_var_params)
            P0_ind = len(u)
            P3_ind = 0
            if scale_var_params == 'per_unit':
                Q0_demand = Q0_load/scale_var_params.get('Sbase')
                V0 = V0_sol/scale_var_params.get('Vbase')
                delta0 = delta0_sol/scale_var_params.get('deltabase')
            else:
                Q0_demand = Q0_load
                V0 = V0_sol
                delta0 = delta0_sol
            Q0_gen = network.nodes[0].half_links[0].get_Q(scale_var=scale_var,scale_var_params=scale_var_params) - Q0_demand
            Q3_gen = network.nodes[3].half_links[0].get_Q(scale_var=scale_var,scale_var_params=scale_var_params)
            x_LF = network.set_x_init(scale_var=scale_var,scale_var_params=scale_var_params)
            y = np.concatenate((u,np.array([P0_gen,Q0_gen,Q3_gen]),x_LF))
            # partial derivatives of equatilty constraints / load-flow equations
            if V_BC:
                nlsys.V_vec_mag[3] = V3
            else:
                nlsys.V_vec_ang[0] = delta0
                nlsys.V_vec_mag[0] = V0
                nlsys.V_vec_mag[3] = V3
            deltah_deltay = h_der(y,P0_ind,network=network,nlsys=nlsys,scale_var=scale_var,scale_var_params=scale_var_params, V_BC=V_BC,Dy=Dy, Dy_inv=Dy_inv, Df=Df, Dh=DH)
            deltah_deltau = deltah_deltay[:,:len(u)]
            deltah_deltax = deltah_deltay[:,len(u):]
            # jacobian inequality constraints
            dg_du = deltag_deltau.copy() # first part of gradient
            if method == 'direct':
                w = np.linalg.solve(deltah_deltax,-deltah_deltau)
                dg_du += np.dot(deltag_deltax,w)
            elif method == 'adjoint':
                v = np.linalg.solve(np.transpose(deltah_deltax),np.transpose(deltag_deltax))
                dg_du += np.dot(np.transpose(v),-deltah_deltau)
            return dg_du
        if optimization_method == 'trust-constr':
            ineq_constr_fun = spo.NonlinearConstraint(g,np.zeros(16),np.inf*np.ones(16),jac=g_jac,keep_feasible=stay_within_bounds)
        elif optimization_method == 'SLSQP' or optimization_method == 'ipopt':
            ineq_constr_fun  = {'type':'ineq','fun':g,'jac':g_jac}
    else:
        ineq_constr_fun = None

    # define bounds
    if V_BC:
        lb_ineq = np.array([P3_gen_bounds[0], V3_bounds[0]])
        ub_ineq = np.array([P3_gen_bounds[1], V3_bounds[1]])
    else:
        lb_ineq = np.array([P3_gen_bounds[0], delta0_bounds[0], V0_bounds[0], V3_bounds[0]])
        ub_ineq = np.array([P3_gen_bounds[1], delta0_bounds[1], V0_bounds[1], V3_bounds[1]])
    if optimization_method == 'ipopt':
        bounds = [(lb,ub) for lb, ub in zip(lb_ineq,ub_ineq)]
        if stay_within_bounds:
            bound_relax_factor = 0.0 # no relaxation of bounds is allowed
        else:
            bound_relax_factor = 1e-8 # default value in Ipopt
    else:
        bounds = spo.Bounds(lb_ineq,ub_ineq,keep_feasible=stay_within_bounds)

    # make sure initial guess satisfies bounds (NB. If adjustments are made, LF is not necessarily satisfied anymore)
    if optimization_method == 'SLSQP' or stay_within_bounds:
        for ind, x0 in enumerate(u0):
            if lb_ineq[ind] > x0:
                u0[ind] = lb_ineq[ind]
            elif ub_ineq[ind] < x0:
                u0[ind] = ub_ineq[ind]

    f_vec = list()
    P3_vec = list()
    V3_vec = list()
    global P3_f_global
    global V3_f_global
    global f_vec_global
    global x_f_vec
    P3_f_global = list()
    V3_f_global = list()
    f_vec_global = list()
    x_f_vec = list()
    if optimization_method == 'trust-constr':
        def callback(xk, state):
            """Called after every iteration"""
            f_vec.append(state.fun)
            P3_vec.append(xk[0])
            V3_vec.append(xk[1])
            return False
    elif optimization_method == 'SLSQP':
        f_vec.append(obj(u0))
        P3_vec.append(u0[0])
        V3_vec.append(u0[1])
        def callback(xk):
            """Called after every iteration"""
            f_vec.append(obj(xk))
            P3_vec.append(xk[0])
            V3_vec.append(xk[1])
            return False
    elif optimization_method == 'ipopt':
        # callback is not implemented in the ipopt (cyipopt) package / wrapper.
        pass

    # solve OPF
    opf_start_time = time.time()
    try:
        if optimization_method == 'trust-constr':
            res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=[ineq_constr_fun], options={'verbose': 1,'maxiter':max_iter}, bounds=bounds, callback=callback)
            execution_time = res.execution_time
        elif optimization_method == 'SLSQP':
            res = spo.minimize(obj, u0, method=optimization_method,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':True,'ftol':tol}, bounds=bounds,tol=tol, callback=callback)
            execution_time = opf_start_time - time.time()
        elif optimization_method == 'ipopt':
            res = ipopt.minimize_ipopt(obj, u0,jac=obj_grad, constraints=ineq_constr_fun, options={'maxiter':max_iter,'disp':1,'bound_relax_factor':bound_relax_factor}, bounds=bounds,tol=tol)
            execution_time = opf_start_time - time.time()
    except:
        print('Exception made for {}, hard bounds: {}, approach: {}, scaling: {}'.format(optimization_method,stay_within_bounds,approach,scale_var))
        if len(f_vec) == 0:
            obj(u0)
            nit = 0
            nfev = 0
            njev = 0
            nhev = 0
        else:
            nit = 0
            nfev = len(f_vec)
            njev = 0
            nhev = 0
        execution_time = opf_start_time - time.time()
        res = spo.OptimizeResult({'success':False,'x':np.array(x_f_vec),'nit':nit,'nfev':nfev,'njev':njev,'nhev':nhev,'message':'An error occured during minimization'})

    if optimization_method == 'ipopt':
        if res.nit > 0:
            P3_vec = [P3_f_global[ind] for ind in range(0,len(P3_f_global),round(len(P3_f_global)/res.nit))]
            V3_vec = [V3_f_global[ind] for ind in range(0,len(V3_f_global),round(len(V3_f_global)/res.nit))]
            f_vec = [f_vec_global[ind] for ind in range(0,len(f_vec_global),round(len(f_vec_global)/res.nit))]
        else:
            P3_vec = P3_f_global
            V3_vec = dphi2_f_global
            f_vec = f_vec_global

    if scale_var == 'matrix':
        u_opf = Du_inv.dot(res.x)
    else:
        u_opf = res.x
    # print solution
    P0_gen, elec_net = solve_lf_in_lf(elec_net,u_opf,max_iters=max_iters_lf,V_BC=V_BC,scale_var=scale_var,scale_var_params=scale_var_params)
    xe_opt = elec_net.set_x_init() # unscaled
    elec_net.reset_network(xe_opt) # if you don't do this first, the update_full will set the wrong values for the half link powers.
    delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.update_full(xe_opt)
    print('Solution OPF:')
    print('delta = {}'.format(delta_sol))
    print('|V| = {} p.u.'.format(V_sol/V_base))
    print('P edge = {} p.u.'.format(P_edge/S_base))
    print('Q edge = {} p.u.'.format(Q_edge/S_base))
    print('S nodal inj = {} p.u.'.format(S_inj/S_base))
    print('P hl = {} p.u.'.format([hl.P/S_base for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} p.u.'.format([hl.Q/S_base for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    return xe_opt, res, f_vec, P3_vec, V3_vec, execution_time

def compare_opf(dir_path,number_runs=10):
    """Compare the different formulation of the optimal flow problem. Print results to a table that can be read by Tex."""
    # solver info
    tol=1e-6
    max_iter=400
    max_iters_lf=10
    scale_var = None
    # initial guesses
    P3_gen=-180*MW
    V3=230*kV
    P0_gen=-150*MW
    Q0_gen=-100*MW
    Q3_gen=-140*MW
    delta1=0
    delta2=0
    delta3=0
    V1=230*kV
    V2=230*kV
    delta0_init=0
    V0_init =230*kV

    # run the various optimizations. Run several times, take average of run time. For the other data (which seemed to be the same every time), the last run is used.
    exec_times = list()
    exec_times_VBC = list()
    exec_times_control = list()
    exec_times_sepLF = list()
    exec_times_VBC_sepLF = list()
    for run in range(number_runs):
        # p0 control variable, inequality contraints on all variables, load flow as equality constraints
        xe_opt, res, exec_tim = run_optimal_load_flow(P3_gen=P3_gen, delta0=delta0_init, V0=V0_init, V3=V3, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2,scale_var=scale_var,tol=tol,max_iter=max_iter,V_BC=False,ineq_constr='all')
        print('Finished OPF with p0 control variable, and inequality contraints on all variables. Run {}'.format(run))
        x_opf, obj_fun, nfev, nit = res.x, res.fun, res.nfev, res.nit
        exec_times.append(exec_time)
        # p0 as BC, inequality contraints only on control variables, load flow as equality constraints
        xe_opt_VBC, res_VBC, exec_time_VBC = run_optimal_load_flow(P3_gen=P3_gen, V3=V3, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2,scale_var=scale_var,tol=tol,max_iter=max_iter,V_BC=True,ineq_constr='control')
        print('Finished OPF with p0 as BC, and inequality contraints on control variables. Run {}'.format(run))
        x_opf_VBC, obj_fun_VBC, nfev_VBC, nit_VBC = res_VBC.x, res_VBC.fun, res_VBC.nfev, res_VBC.nit
        exec_times_VBC.append(exec_time_VBC)
        # p0 control variable, inequality contraints only on control variables, load flow as equality constraints
        xe_opt_control, res_control, exec_time_control = run_optimal_load_flow(P3_gen=P3_gen, delta0=delta0_init, V0=V0_init, V3=V3, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2,scale_var=scale_var,tol=tol,max_iter=max_iter,V_BC=False,ineq_constr='control')
        x_opf_control, obj_fun_control, nfev_control, nit_control = res_control.x, res_control.fun, res_control.nfev, res_control.nit
        print('Finished OPF with p0 control variable, and inequality contraints on control variables. Run {}'.format(run))
        exec_times_control.append(exec_time_control)
        # p0 control variable, inequality contraints only on control variables, load flow as a separate solver
        x_opf_sepLF, obj_fun_sepLF, nfev_sepLF, nit_sepLF, exec_time_sepLF =  run_optimal_load_flow_separate_LF_explicit(P3_gen=P3_gen, delta0=delta0_init, V0=V0_init, V3=V3, P0_gen=P0_gen, Q0_gen=Q0_gen, Q3_gen=Q3_gen, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2,scale_var=scale_var,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,V_BC=False)
        print('Finished OPF with p0 control variable, separate LF. Run {}'.format(run))
        exec_times_sepLF.append(exec_time_sepLF)
        # p0 as BC, inequality contraints only on control variables, load flow as a separate solver
        x_opf_VBC_sepLF, obj_fun_VBC_sepLF, nfev_VBC_sepLF, nit_VBC_sepLF, exec_time_VBC_sepLF =  run_optimal_load_flow_separate_LF_explicit(P3_gen=P3_gen, V3=V3, P0_gen=P0_gen, Q0_gen=Q0_gen, Q3_gen=Q3_gen, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2,scale_var=scale_var,tol=tol,max_iter=max_iter,max_iters_lf=max_iters_lf,V_BC=True)
        print('Finished OPF with p0 as BC, separate LF. Run {}'.format(run))
        exec_times_VBC_sepLF.append(exec_time_VBC_sepLF)
    exec_time = np.mean(exec_times)
    exec_time_VBC = np.mean(exec_times_VBC)
    exec_time_control = np.mean(exec_times_control)
    exec_time_sepLF = np.mean(exec_times_sepLF)
    exec_time_VBC_sepLF = np.mean(exec_times_VBC_sepLF)

    path_to_tables = os.path.join(dir_path,'network_data','E4N')
    # create (and save) table with optimal solution in network
    P3_gen_opt, delta0_opt, V0_opt, V3_opt = x_opf[:4]
    P0_gen_opt, Q0_gen_opt, Q3_gen_opt = x_opf[4:7]
    xe_opt = xe_from_xopf(x_opf,V_BC=False)
    P3_gen_opt_control, delta0_opt_control, V0_opt_control, V3_opt_control = x_opf_control[:4]
    P0_gen_opt_control, Q0_gen_opt_control, Q3_gen_opt_control = x_opf_control[4:7]
    xe_opt_control = xe_from_xopf(x_opf_control,V_BC=False)
    P3_gen_opt_VBC, V3_opt_VBC = x_opf_VBC[:2]
    P0_gen_opt_VBC, Q0_gen_opt_VBC, Q3_gen_opt_VBC = x_opf_VBC[2:5]
    xe_opt_VBC = xe_from_xopf(x_opf_VBC,V_BC=True)
    P3_gen_opt_sepLF, delta0_opt_sepLF, V0_opt_sepLF, V3_opt_sepLF = x_opf_sepLF[:4]
    P0_gen_opt_sepLF, Q0_gen_opt_sepLF, Q3_gen_opt_sepLF = x_opf_sepLF[4:]
    P3_gen_opt_VBC_sepLF, V3_opt_VBC_sepLF = x_opf_VBC_sepLF[:2]
    P0_gen_opt_VBC_sepLF, Q0_gen_opt_VBC_sepLF, Q3_gen_opt_VBC_sepLF = x_opf_VBC_sepLF[2:]
    with HiddenPrints():
        elec_net,xe_LF,_,_,_,_,S_inj,P_edge,Q_edge,pp_net = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iters_lf)
        if 'sn_kva' in pp_net.keys():
            S_base = pp_net.sn_kva*kW #[va]
        elif 'sn_mva' in pp_net.keys():
            S_base = pp_net.sn_mva*MW #[va]
        V_base = pp_net.bus['vn_kv'][0]*kV #[V]
        scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}
        P0_load = pp_net.load['p_mw'][0]*MW
        Q0_load = pp_net.load['q_mvar'][0]*MW
        V0_BC = pp_net.res_bus['vm_pu'][0]*V_base
        delta0_BC = pp_net.res_bus['va_degree'][0]*degree
        P0_gen_LF = elec_net.nodes[0].half_links[0].get_P() - P0_load #is a source, so the half link power <0
        Q0_gen_LF = elec_net.nodes[0].half_links[0].get_Q() - Q0_load#is a source, so the half link power <0
        Q3_gen_LF = elec_net.nodes[3].half_links[0].get_Q()
        P3_gen_LF = elec_net.nodes[3].half_links[0].get_P()
        V3_LF = elec_net.nodes[3].get_V()
        x_LF0 = np.array([delta1, delta2, delta3, V1, V2])
        elec_net = update_bc(elec_net,delta0_opt_sepLF,V0_opt_sepLF,V3_opt_sepLF,P3_gen_opt_sepLF)
        elec_net.reset_network(x_LF0)
        xe_opt_sepLF,_,_,_,_,_,_,_ = elec_net.solve_network(tol,max_iters_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        elec_net = update_bc(elec_net,delta0_BC,V0_BC,V3_opt_VBC_sepLF,P3_gen_opt_VBC_sepLF)
        elec_net.reset_network(x_LF0)
        xe_opt_VBC_sepLF,_,_,_,_,_,_,_ = elec_net.solve_network(tol,max_iters_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    with open(os.path.join(path_to_tables,'network_solution.txt'), "w") as table:
        table.write(r'$P_{3,0}$'+r' & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e} \\ '.format(P3_gen_LF,P3_gen_opt,P3_gen_opt_VBC,P3_gen_opt_control,P3_gen_opt_sepLF,P3_gen_opt_VBC_sepLF))
        table.write(r'$\delta_0$ & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e} \\ '.format(delta0_BC,delta0_opt,delta0_BC,delta0_opt_control,delta0_opt_sepLF,delta0_BC))
        table.write(r'$|V_0|$ & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e} \\ '.format(V0_BC,V0_opt,V0_BC,V0_opt_control,V0_opt_sepLF,V0_BC))
        table.write(r'$|V_3|$ & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e} \\ '.format(V3_LF,V3_opt,V3_opt_VBC,V3_opt_control,V3_opt_sepLF,V3_opt_VBC_sepLF))
        table.write(r'$P_{0,0}$'+r' & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e} \\ '.format(P0_gen_LF,P0_gen_opt,P0_gen_opt_VBC,P0_gen_opt_control,P0_gen_opt_sepLF,P0_gen_opt_VBC_sepLF))
        table.write(r'$Q_{0,0}$'+r' & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e} \\ '.format(Q0_gen_LF,Q0_gen_opt,Q0_gen_opt_VBC,Q0_gen_opt_control,Q0_gen_opt_sepLF,Q0_gen_opt_VBC_sepLF))
        table.write(r'$Q_{3,0}$'+r' & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e} \\ '.format(Q3_gen_LF,Q3_gen_opt,Q3_gen_opt_VBC,Q3_gen_opt_control,Q3_gen_opt_sepLF,Q3_gen_opt_VBC_sepLF))
        variable_names = [r'$\delta_1$',r'$\delta_2$',r'$\delta_3$',r'$|V_1|$',r'$|V_2|$']
        for ind_var, var in enumerate(variable_names):
            table.write(r'{} & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e}  & {:5e} \\ '.format(var,xe_LF[ind_var],xe_opt[ind_var],xe_opt_VBC[ind_var],xe_opt_control[ind_var],xe_opt_sepLF[ind_var],xe_opt_VBC_sepLF[ind_var]))

    # print results of optimizer, and create (and save) table
    print('\nopf     opf VBC  opf control ineq    opf sep. LF     opf sep. LF VBC')
    print('obj. func:  {:.5e}  , {:.5e} , {:.5e}  , {:.5e}  , {:.5e}'.format(obj_fun,obj_fun_VBC,obj_fun_control,obj_fun_sepLF,obj_fun_VBC_sepLF))
    print('numb. fev.:  {:d}  , {:d} , {:d}  , {:d}  , {:d}'.format(nfev,nfev_VBC,nfev_control,nfev_sepLF,nfev_VBC_sepLF))
    print('iters:  {:d}  , {:d}  , {:d} , {:d}  , {:d}'.format(nit,nit_VBC,nit_control,nit_sepLF,nit_VBC_sepLF))
    print('time:  {:.5f}  , {:.5f} , {:5f}  , {:.5f}  , {:.5f}'.format(exec_time,exec_time_VBC,exec_time_control,exec_time_sepLF,exec_time_VBC_sepLF))
    with open(os.path.join(path_to_tables,'optimizer_info.txt'), "w") as table:
        table.write(r'$f$ &  {:.5e}  & {:.5e} & {:.5e}  & {:.5e}  & {:.5e} \\ '.format(obj_fun,obj_fun_VBC,obj_fun_control,obj_fun_sepLF,obj_fun_VBC_sepLF))
        table.write(r'func. eval. &  {:d}  & {:d}  & {:d}  & {:d}  & {:d} \\ '.format(nfev,nfev_VBC,nfev_control,nfev_sepLF,nfev_VBC_sepLF))
        table.write(r'iterations &  {:d}  & {:d}  & {:d}  & {:d}  & {:d} \\ '.format(nit,nit_VBC,nit_control,nit_sepLF,nit_VBC_sepLF))
        table.write(r'time [s] &  {:.5f}  & {:.5f}  & {:.5f}  & {:.5f}  & {:.5f} \\ '.format(exec_time,exec_time_VBC,exec_time_control,exec_time_sepLF,exec_time_VBC_sepLF))

def compare_opf_derivatives(dir_path,number_runs=10,save_tables=True):
    """Compare the the optimal flow problem, using different ways to determine the gradients (and Hessians). V0 is taken as BC, and the inequality constraints are imposed on the control variables only. Print results to a table that can be read by LaTeX."""
    # solver info
    tol=1e-6
    max_iter=400
    max_iters_lf=10
    scale_var = None
    ineq_constr = 'control'
    V_BC = True
    # initial guesses
    P3_gen=-180*MW
    V3=230*kV
    P0_gen=-150*MW
    Q0_gen=-100*MW
    Q3_gen=-140*MW
    delta1=0
    delta2=0
    delta3=0
    V1=230*kV
    V2=230*kV
    delta0_init=0
    V0_init =230*kV

    # run the various optimizations. Run several times, take average of run time. For the other data (which seemed to be the same every time), the last run is used.
    exec_times = list()
    exec_times_sepLF_direct = list()
    exec_times_sepLF_adjoint = list()
    for run in range(number_runs):
        # LF is included as (nonlinear) equality constriant. Analytical expressions for gradients and Hessian of objective function and Jacobian of equality constraints are used
        xe_opt, res, exec_time = run_optimal_load_flow(P3_gen=P3_gen, V3=V3, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2,scale_var=scale_var,tol=tol,max_iter=max_iter,V_BC=V_BC,ineq_constr=ineq_constr, derivatives=True)
        x_opf, obj_fun, nfev, nit = res.x, res.fun, res.nfev, res.nit
        exec_times.append(exec_time)
        # LF is not included as (nonlinear) equality constriant. Analytical expressions for gradient of objective function are determined using the direct approach
        xe_opt_sepLF_direct, res_sepLF_direct, exec_time_sepLF_direct  = run_optimal_load_flow_separate_LF(P3_gen=P3_gen, V3=V3, P0_gen=P0_gen, Q0_gen=Q0_gen, Q3_gen=Q3_gen, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2,scale_var=scale_var,tol=tol,max_iter=max_iter,V_BC=V_BC,approach='direct')
        x_opf_sepLF_direct, obj_fun_sepLF_direct, nfev_sepLF_direct, nit_sepLF_direct = res_sepLF_direct.x, res_sepLF_direct.fun, res_sepLF_direct.nfev, res_sepLF_direct.nit
        exec_times_sepLF_direct.append(exec_time_sepLF_direct)
        # LF is not included as (nonlinear) equality constriant. Analytical expressions for gradient of objective function are determined using the adjoint approach
        xe_opt_sepLF_adjoint, res_sepLF_adjoint, exec_time_sepLF_adjoint  = run_optimal_load_flow_separate_LF(P3_gen=P3_gen, V3=V3, P0_gen=P0_gen, Q0_gen=Q0_gen, Q3_gen=Q3_gen, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2,scale_var=scale_var,tol=tol,max_iter=max_iter,V_BC=V_BC,approach='adjoint')
        x_opf_sepLF_adjoint, obj_fun_sepLF_adjoint, nfev_sepLF_adjoint, nit_sepLF_adjoint = res_sepLF_adjoint.x, res_sepLF_adjoint.fun, res_sepLF_adjoint.nfev, res_sepLF_adjoint.nit
        exec_times_sepLF_adjoint.append(exec_time_sepLF_adjoint)

    exec_time = np.mean(exec_times)
    exec_time_sepLF_direct = np.mean(exec_times_sepLF_direct)
    exec_time_sepLF_adjoint = np.mean(exec_times_sepLF_adjoint)
    print('exec time = {}'.format(exec_times))
    print('exec time sep. LF direct = {}'.format(exec_times_sepLF_direct))
    print('exec time sep. LF adjoint = {}'.format(exec_times_sepLF_adjoint))

    path_to_tables = os.path.join(dir_path,'network_data','E4N')
    # create (and save) table with optimal solution in network
    P3_gen_opt, V3_opt = x_opf[:2]
    P0_gen_opt, Q0_gen_opt, Q3_gen_opt = x_opf[2:5]
    xe_opt = xe_from_xopf(x_opf,V_BC=V_BC)
    P3_gen_opt_sepLF_direct, V3_opt_sepLF_direct = x_opf_sepLF_direct
    P3_gen_opt_sepLF_adjoint, V3_opt_sepLF_adjoint = x_opf_sepLF_adjoint
    with HiddenPrints():
        # LF solution
        elec_net,xe_LF,_,_,_,_,S_inj,P_edge,Q_edge,pp_net = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iters_lf)
        if 'sn_kva' in pp_net.keys():
            S_base = pp_net.sn_kva*kW #[va]
        elif 'sn_mva' in pp_net.keys():
            S_base = pp_net.sn_mva*MW #[va]
        V_base = pp_net.bus['vn_kv'][0]*kV #[V]
        scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}
        P0_load = pp_net.load['p_mw'][0]*MW
        Q0_load = pp_net.load['q_mvar'][0]*MW
        V0_BC = pp_net.res_bus['vm_pu'][0]*V_base
        delta0_BC = pp_net.res_bus['va_degree'][0]*degree
        P0_gen_LF = elec_net.nodes[0].half_links[0].get_P() - P0_load #is a source, so the half link power <0
        Q0_gen_LF = elec_net.nodes[0].half_links[0].get_Q() - Q0_load#is a source, so the half link power <0
        Q3_gen_LF = elec_net.nodes[3].half_links[0].get_Q()
        P3_gen_LF = elec_net.nodes[3].half_links[0].get_P()
        V3_LF = elec_net.nodes[3].get_V()
        # solution with separate LF
        x_LF0 = np.array([delta1, delta2, delta3, V1, V2])
        # direct approach
        elec_net = update_bc(elec_net,delta0_BC,V0_BC,V3_opt_sepLF_direct,P3_gen_opt_sepLF_direct)
        elec_net.reset_network(x_LF0)
        xe_opt_sepLF_direct,_,_,_,_,_,_,_ = elec_net.solve_network(tol,max_iters_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        P0_gen_opt_sepLF_direct = elec_net.nodes[0].half_links[0].get_P() - P0_load #is a source, so the half link power <0
        Q0_gen_opt_sepLF_direct = elec_net.nodes[0].half_links[0].get_Q() - Q0_load#is a source, so the half link power <0
        Q3_gen_opt_sepLF_direct = elec_net.nodes[3].half_links[0].get_Q()
        # adjoint approach
        elec_net = update_bc(elec_net,delta0_BC,V0_BC,V3_opt_sepLF_adjoint,P3_gen_opt_sepLF_adjoint)
        elec_net.reset_network(x_LF0)
        xe_opt_sepLF_adjoint,_,_,_,_,_,_,_ = elec_net.solve_network(tol,max_iters_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        P0_gen_opt_sepLF_adjoint = elec_net.nodes[0].half_links[0].get_P() - P0_load #is a source, so the half link power <0
        Q0_gen_opt_sepLF_adjoint = elec_net.nodes[0].half_links[0].get_Q() - Q0_load#is a source, so the half link power <0
        Q3_gen_opt_sepLF_adjoint = elec_net.nodes[3].half_links[0].get_Q()

    if save_tables:
        with open(os.path.join(path_to_tables,'network_solution_derivatives.txt'), "w") as table:
            table.write(r'$P_{3,0}$'+r' & {:5e}  & {:5e}  & {:5e}  & {:5e}\\ '.format(P3_gen_LF,P3_gen_opt,P3_gen_opt_sepLF_direct,P3_gen_opt_sepLF_adjoint))
            table.write(r'$\delta_0$ & {:5e}  & {:5e}  & {:5e}  & {:5e}  \\ '.format(delta0_BC,delta0_BC,delta0_BC,delta0_BC))
            table.write(r'$|V_0|$ & {:5e}  & {:5e}  & {:5e}  & {:5e}  \\ '.format(V0_BC,V0_BC,V0_BC,V0_BC))
            table.write(r'$|V_3|$ & {:5e}  & {:5e}  & {:5e}  & {:5e}  \\ '.format(V3_LF,V3_opt,V3_opt_sepLF_direct,V3_opt_sepLF_adjoint))
            table.write(r'$P_{0,0}$'+r' & {:5e}  & {:5e}  & {:5e}  & {:5e}  \\ '.format(P0_gen_LF,P0_gen_opt,P0_gen_opt_sepLF_direct,P0_gen_opt_sepLF_adjoint))
            table.write(r'$Q_{0,0}$'+r' & {:5e}  & {:5e}  & {:5e}  & {:5e}  \\ '.format(Q0_gen_LF,Q0_gen_opt,Q0_gen_opt_sepLF_direct,Q0_gen_opt_sepLF_adjoint))
            table.write(r'$Q_{3,0}$'+r' & {:5e}  & {:5e}  & {:5e}  & {:5e}  \\ '.format(Q3_gen_LF,Q3_gen_opt,Q3_gen_opt_sepLF_direct,Q3_gen_opt_sepLF_adjoint))
            variable_names = [r'$\delta_1$',r'$\delta_2$',r'$\delta_3$',r'$|V_1|$',r'$|V_2|$']
            for ind_var, var in enumerate(variable_names):
                table.write(r'{} & {:5e}  & {:5e}  & {:5e}  & {:5e}  \\ '.format(var,xe_LF[ind_var],xe_opt[ind_var],xe_opt_sepLF_direct[ind_var],xe_opt_sepLF_adjoint[ind_var]))

    # print results of optimizer, and create (and save) table
    print('\nopf     opf sep. LF direct    opf sep. LF adjoint')
    print('obj. func:  {:.5e}  , {:.5e} , {:.5e}'.format(obj_fun,obj_fun_sepLF_direct,obj_fun_sepLF_adjoint))
    print('numb. fev.:  {:d}  , {:d} , {:d}'.format(nfev,nfev_sepLF_direct,nfev_sepLF_adjoint))
    print('iters:  {:d}  , {:d}  , {:d}'.format(nit,nit_sepLF_direct,nit_sepLF_adjoint))
    print('time:  {:.5f}  , {:.5f} , {:5f}'.format(exec_time,exec_time_sepLF_direct,exec_time_sepLF_adjoint))
    if save_tables:
        with open(os.path.join(path_to_tables,'optimizer_info_derivatives.txt'), "w") as table:
            table.write(r'$f$ &  {:.5e}  & {:.5e} & {:.5e}  \\ '.format(obj_fun,obj_fun_sepLF_direct,obj_fun_sepLF_adjoint))
            table.write(r'func. eval. &  {:d}  & {:d}  & {:d}  \\ '.format(nfev,nfev_sepLF_direct,nfev_sepLF_adjoint))
            table.write(r'iterations &  {:d}  & {:d}  & {:d}  \\ '.format(nit,nit_sepLF_direct,nit_sepLF_adjoint))
            table.write(r'time [s] &  {:.5f}  & {:.5f}  & {:.5f}  \\ '.format(exec_time,exec_time_sepLF_direct,exec_time_sepLF_adjoint))

def error(x_res,x_sol):
    """Relative error between solution and result.

    Parameters
    ----------
    x_res : np array
        Variables result. Unscaled
    x_sol : np array
        Variables solution. Unscaled
    """
    return np.max(np.abs(x_sol-x_res)/np.abs(x_sol))

def compare_opf_methods(dir_path=None,save_tables=False):
    """Compare OPF for different optimization methods. Without scaling."""
    if save_tables and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    tol=1e-6
    max_iter=400
    max_iters_lf=10
    scale_var = None
    a0=0
    a1=0
    b0=.2
    b1=.3
    c0=1e-5
    c1=2e-5

    #LF solution
    with HiddenPrints():
        elec_net,xe_LF,_,_,_,_,S_inj,P_edge,Q_edge,pp_net = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iters_lf)
        if 'sn_kva' in pp_net.keys():
            S_base = pp_net.sn_kva*kW #[va]
        elif 'sn_mva' in pp_net.keys():
            S_base = pp_net.sn_mva*MW #[va]
        V_base = pp_net.bus['vn_kv'][0]*kV #[V]
        scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}
        P0_load = pp_net.load['p_mw'][0]*MW
        Q0_load = pp_net.load['q_mvar'][0]*MW
        V0_BC = pp_net.res_bus['vm_pu'][0]*V_base
        delta0_BC = pp_net.res_bus['va_degree'][0]*degree
        P0_gen_LF = elec_net.nodes[0].half_links[0].get_P() - P0_load #is a source, so the half link power <0
        Q0_gen_LF = elec_net.nodes[0].half_links[0].get_Q() - Q0_load#is a source, so the half link power <0
        Q3_gen_LF = elec_net.nodes[3].half_links[0].get_Q()
        P3_gen_LF = elec_net.nodes[3].half_links[0].get_P()
        V3_LF = elec_net.nodes[3].get_V()
        x_opf_LF = np.concatenate((np.array([P3_gen_LF, V3_LF, P0_gen_LF, Q0_gen_LF, Q3_gen_LF]),xe_LF))

    # initial guesses
    P3_gen=1.3*P3_gen_LF
    V3=1.05*V3_LF
    delta1=0
    delta2=0
    delta3=0
    V1=230*kV
    V2=230*kV

    # bounds
    ineq_constr='all'
    P3_gen_bounds = np.array([2.5*P3_gen_LF,P3_gen_LF])
    V3_bounds = np.array([V3_LF,1.2*V_base])
    P0_gen_bounds = np.array([2.5*P0_gen_LF,0])
    Q0_gen_bounds = np.array([2.5*Q0_gen_LF,0])
    Q3_gen_bounds = np.array([2.5*Q3_gen_LF,0])
    delta1_bounds = np.array([-np.pi,np.pi])
    delta2_bounds = np.array([-np.pi,np.pi])
    delta3_bounds = np.array([-np.pi,np.pi])
    V1_bounds = np.array([0.8*V_base,1.2*V_base])
    V2_bounds = np.array([0.8*V_base,1.2*V_base])

    # BC
    V_BC = True
    delta0 =delta0_BC
    V0 = V0_BC

    result = dict()
    xe_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    ders = ['an','num']

    for method in methods:
        for bound in bounds:
            if bound == 'soft':
                stay_within_bounds = False
            else:
                stay_within_bounds = True
            for der in ders:
                if der == 'an':
                    derivatives = True
                else:
                    derivatives = False
                xe_opt, res, _, _, _, _ = run_optimal_load_flow(P3_gen=P3_gen, delta0=delta0, V0=V0, V3=V3, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2, P3_gen_bounds=P3_gen_bounds, V3_bounds=V3_bounds, P0_gen_bounds=P0_gen_bounds, Q0_gen_bounds=Q0_gen_bounds, Q3_gen_bounds=Q3_gen_bounds, delta1_bounds=delta1_bounds, delta2_bounds=delta2_bounds, delta3_bounds=delta3_bounds, V1_bounds=V1_bounds, V2_bounds=V2_bounds, a0=a0, a1=a1, b0=b0, b1=b1, c0=c0, c1=c1, scale_var=scale_var, tol=tol, max_iter=max_iter, V_BC=V_BC, ineq_constr=ineq_constr, derivatives=derivatives, optimization_method=method, stay_within_bounds=stay_within_bounds)
                result[method+'_'+bound+'_'+der] = res
                xe_res[method+'_'+bound+'_'+der] = xe_opt

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','E4N')
        for bound in bounds:
            for der in ders:
                res_trust = result.get('trust-constr_'+bound+'_'+der)
                res_slsqp = result.get('SLSQP_'+bound+'_'+der)
                res_ipopt = result.get('ipopt_'+bound+'_'+der)
                with open(os.path.join(path_to_tables,'network_solution_errors_methods_'+bound+'_'+der+'.txt'), "w") as table:
                    variable_names = [r'$P_{3,0}$',r'$|V_3|$',r'$P_{0,0}$',r'$Q_{0,0}$',r'$Q_{3,0}$',r'$\delta_1$',r'$\delta_2$',r'$\delta_3$',r'$|V_1|$',r'$|V_2|$']
                    for ind_var, var in enumerate(variable_names):
                        table.write(r'{} & {:.3e}  & {:.3e}  & {:.3e}  & {:.3e}  \\ '.format(var,x_opf_LF[ind_var],error(res_trust.x[ind_var],x_opf_LF[ind_var]),error(res_slsqp.x[ind_var],x_opf_LF[ind_var]),error(res_ipopt.x[ind_var],x_opf_LF[ind_var])))
        with open(os.path.join(path_to_tables,'optimizer_info_methods.txt'), "w") as table:
            for bound in bounds:
                for der in ders:
                    res_trust = result.get('trust-constr_'+bound+'_'+der)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+der)
                    res_ipopt = result.get('ipopt_'+bound+'_'+der)
                    table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(res_trust.x,x_opf_LF),error(res_slsqp.x,x_opf_LF),error(res_ipopt.x,x_opf_LF)))
                    print('\nBounds: {}, der: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\ntrust-constr:{}\nSLSQP: {}\nIPOPT: {}'.format(bound,der,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message))

def compare_opf_scaling(dir_path=None,save_tables=False):
    """Compare OPF for different scaling options, and for different optimization methods."""
    if save_tables and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    tol=1e-6
    max_iter=400
    max_iters_lf=10
    a0=0
    a1=0
    b0=.2
    b1=.3
    c0=1e-5
    c1=2e-5

    #LF solution
    with HiddenPrints():
        elec_net,xe_LF,_,_,_,_,S_inj,P_edge,Q_edge,pp_net = run_load_flow(scale_var='matrix',tol=tol,max_iter=max_iters_lf)
        if 'sn_kva' in pp_net.keys():
            S_base = pp_net.sn_kva*kW #[va]
        elif 'sn_mva' in pp_net.keys():
            S_base = pp_net.sn_mva*MW #[va]
        V_base = pp_net.bus['vn_kv'][0]*kV #[V]
        scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}
        P0_load = pp_net.load['p_mw'][0]*MW
        Q0_load = pp_net.load['q_mvar'][0]*MW
        V0_BC = pp_net.res_bus['vm_pu'][0]*V_base
        delta0_BC = pp_net.res_bus['va_degree'][0]*degree
        P0_gen_LF = elec_net.nodes[0].half_links[0].get_P() - P0_load #is a source, so the half link power <0
        Q0_gen_LF = elec_net.nodes[0].half_links[0].get_Q() - Q0_load#is a source, so the half link power <0
        Q3_gen_LF = elec_net.nodes[3].half_links[0].get_Q()
        P3_gen_LF = elec_net.nodes[3].half_links[0].get_P()
        V3_LF = elec_net.nodes[3].get_V()
        x_opf_LF = np.concatenate((np.array([P3_gen_LF/S_base, V3_LF/V_base, P0_gen_LF/S_base, Q0_gen_LF/S_base, Q3_gen_LF/S_base]),xe_LF/np.array([1,1,1,V_base,V_base])))

    # initial guesses
    P3_gen=1.3*P3_gen_LF
    V3=1.05*V3_LF
    delta1=0
    delta2=0
    delta3=0
    V1=230*kV
    V2=230*kV

    # bounds
    ineq_constr='all'
    P3_gen_bounds = np.array([2.5*P3_gen_LF,P3_gen_LF])
    V3_bounds = np.array([V3_LF,1.2*V_base])
    P0_gen_bounds = np.array([P0_gen_LF,0])
    Q0_gen_bounds = np.array([2.5*Q0_gen_LF,0])
    Q3_gen_bounds = np.array([2.5*Q3_gen_LF,0])
    delta1_bounds = np.array([-np.pi,np.pi])
    delta2_bounds = np.array([-np.pi,np.pi])
    delta3_bounds = np.array([-np.pi,np.pi])
    V1_bounds = np.array([0.8*V_base,1.2*V_base])
    V2_bounds = np.array([0.8*V_base,1.2*V_base])

    # BC
    V_BC = True
    delta0 =delta0_BC
    V0 = V0_BC

    result = dict()
    xe_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    scaling = ['matrix','per_unit']
    bounds = ['soft', 'hard']
    ders = ['an','num']
    fb = 1*MW

    for method in methods:
        for bound in bounds:
            if bound == 'soft':
                stay_within_bounds = False
            else:
                stay_within_bounds = True
            for der in ders:
                if der == 'an':
                    derivatives = True
                else:
                    derivatives = False
                for scale_var in scaling:
                    xe_opt, res, _, _, _, _ = run_optimal_load_flow(P3_gen=P3_gen, delta0=delta0, V0=V0, V3=V3, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2, P3_gen_bounds=P3_gen_bounds, V3_bounds=V3_bounds, P0_gen_bounds=P0_gen_bounds, Q0_gen_bounds=Q0_gen_bounds, Q3_gen_bounds=Q3_gen_bounds, delta1_bounds=delta1_bounds, delta2_bounds=delta2_bounds, delta3_bounds=delta3_bounds, V1_bounds=V1_bounds, V2_bounds=V2_bounds, a0=a0, a1=a1, b0=b0, b1=b1, c0=c0, c1=c1, tol=tol, max_iter=max_iter, V_BC=V_BC, ineq_constr=ineq_constr, derivatives=derivatives, optimization_method=method, stay_within_bounds=stay_within_bounds, scale_var=scale_var,scale_var_params=scale_var_params,fb=fb)
                    result[method+'_'+bound+'_'+der+'_'+scale_var] = res
                    xe_res[method+'_'+bound+'_'+der+'_'+scale_var] = xe_opt

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','E4N')
        with open(os.path.join(path_to_tables,'optimizer_info_methods_scaling.txt'), "w") as table:
            for bound in bounds:
                for der in ders:
                    res_trust_mat = result.get('trust-constr_'+bound+'_'+der+'_matrix')
                    res_slsqp_mat = result.get('SLSQP_'+bound+'_'+der+'_matrix')
                    res_ipopt_mat = result.get('ipopt_'+bound+'_'+der+'_matrix')
                    res_trust_pu = result.get('trust-constr_'+bound+'_'+der+'_per_unit')
                    res_slsqp_pu = result.get('SLSQP_'+bound+'_'+der+'_per_unit')
                    res_ipopt_pu = result.get('ipopt_'+bound+'_'+der+'_per_unit')
                    table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} \\ '.format(bound,der,res_trust_mat.success,res_trust_pu.success,res_slsqp_mat.success,res_slsqp_pu.success,res_ipopt_mat.success,res_ipopt_pu.success,res_trust_mat.nit,res_trust_pu.nit,res_slsqp_pu.nit,res_slsqp_mat.nit,res_ipopt_mat.nit,res_ipopt_pu.nit,error(res_trust_mat.x,x_opf_LF),error(res_trust_pu.x,x_opf_LF),error(res_slsqp_mat.x,x_opf_LF),error(res_slsqp_pu.x,x_opf_LF),error(res_ipopt_mat.x,x_opf_LF),error(res_ipopt_pu.x,x_opf_LF)))
    for bound in bounds:
        for der in ders:
            for scale_var in scaling:
                res_trust = result.get('trust-constr_'+bound+'_'+der+'_'+scale_var)
                res_slsqp = result.get('SLSQP_'+bound+'_'+der+'_'+scale_var)
                res_ipopt = result.get('ipopt_'+bound+'_'+der+'_'+scale_var)
                print('\nBounds: {}, der: {}, scaling: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\ntrust-constr:{}\nSLSQP: {}\nIPOPT: {}'.format(bound,der,scale_var,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message))

def compare_opf_methods_sep_LF(dir_path=None,save_tables=False):
    """Compare OPF for different optimization methods. Without scaling."""
    if save_tables and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    tol=1e-6
    max_iter=400
    max_iters_lf=10
    scale_var = None
    a0=0
    a1=0
    b0=.2
    b1=.3
    c0=1e-5
    c1=2e-5

    #LF solution
    with HiddenPrints():
        elec_net,xe_LF,_,_,_,_,S_inj,P_edge,Q_edge,pp_net = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iters_lf)
        if 'sn_kva' in pp_net.keys():
            S_base = pp_net.sn_kva*kW #[va]
        elif 'sn_mva' in pp_net.keys():
            S_base = pp_net.sn_mva*MW #[va]
        V_base = pp_net.bus['vn_kv'][0]*kV #[V]
        scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}
        P0_load = pp_net.load['p_mw'][0]*MW
        Q0_load = pp_net.load['q_mvar'][0]*MW
        V0_BC = pp_net.res_bus['vm_pu'][0]*V_base
        delta0_BC = pp_net.res_bus['va_degree'][0]*degree
        P0_gen_LF = elec_net.nodes[0].half_links[0].get_P() - P0_load #is a source, so the half link power <0
        Q0_gen_LF = elec_net.nodes[0].half_links[0].get_Q() - Q0_load#is a source, so the half link power <0
        Q3_gen_LF = elec_net.nodes[3].half_links[0].get_Q()
        P3_gen_LF = elec_net.nodes[3].half_links[0].get_P()
        V3_LF = elec_net.nodes[3].get_V()

    # initial guesses
    P3_gen=1.3*P3_gen_LF
    V3=1.05*V3_LF
    delta1=0
    delta2=0
    delta3=0
    V1=230*kV
    V2=230*kV

    # bounds
    ineq_constr='all'
    P3_gen_bounds = np.array([2.5*P3_gen_LF,P3_gen_LF])
    V3_bounds = np.array([V3_LF,1.2*V_base])
    P0_gen_bounds = np.array([2.5*P0_gen_LF,0])
    Q0_gen_bounds = np.array([2.5*Q0_gen_LF,0])
    Q3_gen_bounds = np.array([2.5*Q3_gen_LF,0])
    delta1_bounds = np.array([-np.pi,np.pi])
    delta2_bounds = np.array([-np.pi,np.pi])
    delta3_bounds = np.array([-np.pi,np.pi])
    V1_bounds = np.array([0.8*V_base,1.2*V_base])
    V2_bounds = np.array([0.8*V_base,1.2*V_base])

    # BC
    V_BC = True
    delta0 =delta0_BC
    V0 = V0_BC

    result = dict()
    xe_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']

    for method in methods:
        for bound in bounds:
            if bound == 'soft':
                stay_within_bounds = False
            else:
                stay_within_bounds = True
            for approach in approaches:
                if approach == 'direct' or approach == 'adjoint':
                    xe_opt, res, f_vec, P3_vec, V3_vec, _ = run_optimal_load_flow_separate_LF(P3_gen=P3_gen, delta0=delta0, V0=V0, V3=V3, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2, P3_gen_bounds=P3_gen_bounds, V3_bounds=V3_bounds, P0_gen_bounds=P0_gen_bounds, Q0_gen_bounds=Q0_gen_bounds, Q3_gen_bounds=Q3_gen_bounds, delta1_bounds=delta1_bounds, delta2_bounds=delta2_bounds, delta3_bounds=delta3_bounds, V1_bounds=V1_bounds, V2_bounds=V2_bounds, a0=a0, a1=a1, b0=b0, b1=b1, c0=c0, c1=c1, scale_var=scale_var, tol=tol, max_iter=max_iter, max_iters_lf=max_iters_lf, V_BC=V_BC, ineq_constr=ineq_constr, approach=approach, optimization_method=method, stay_within_bounds=stay_within_bounds)
                else:
                    xe_opt, res, f_vec, P3_vec, V3_vec, _ = run_optimal_load_flow(P3_gen=P3_gen, delta0=delta0, V0=V0, V3=V3, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2, P3_gen_bounds=P3_gen_bounds, V3_bounds=V3_bounds, P0_gen_bounds=P0_gen_bounds, Q0_gen_bounds=Q0_gen_bounds, Q3_gen_bounds=Q3_gen_bounds, delta1_bounds=delta1_bounds, delta2_bounds=delta2_bounds, delta3_bounds=delta3_bounds, V1_bounds=V1_bounds, V2_bounds=V2_bounds, a0=a0, a1=a1, b0=b0, b1=b1, c0=c0, c1=c1, scale_var=scale_var, tol=tol, max_iter=max_iter, V_BC=V_BC, ineq_constr=ineq_constr, derivatives=True, optimization_method=method, stay_within_bounds=stay_within_bounds)
                result[method+'_'+bound+'_'+approach] = res
                xe_res[method+'_'+bound+'_'+approach] = xe_opt

    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','E4N')
        for bound in bounds:
            for approach in approaches:
                xe_opt_trust = xe_res.get('trust-constr_'+bound+'_'+approach)
                xe_opt_slsqp = xe_res.get('SLSQP_'+bound+'_'+approach)
                xe_opt_ipopt = xe_res.get('ipopt_'+bound+'_'+approach)
                with open(os.path.join(path_to_tables,'network_solution_errors_methods_sep_LF_'+bound+'_'+approach+'.txt'), "w") as table:
                    variable_names = [r'$\delta_1$',r'$\delta_2$',r'$\delta_3$',r'$|V_1|$',r'$|V_2|$']
                    for ind_var, var in enumerate(variable_names):
                        table.write(r'{} & {:.3e}  & {:.3e}  & {:.3e}  & {:.3e}  \\ '.format(var,xe_LF[ind_var],error(xe_opt_trust[ind_var],xe_LF[ind_var]),error(xe_opt_slsqp[ind_var],xe_LF[ind_var]),error(xe_opt_ipopt[ind_var],xe_LF[ind_var])))
        with open(os.path.join(path_to_tables,'optimizer_info_methods_sep_LF.txt'), "w") as table:
            for bound in bounds:
                for approach in approaches:
                    if approach == 'eq_constr':
                        approach_label = 'eq. constr.'
                    else:
                        approach_label = approach
                    xe_opt_trust = xe_res.get('trust-constr_'+bound+'_'+approach)
                    xe_opt_slsqp = xe_res.get('SLSQP_'+bound+'_'+approach)
                    xe_opt_ipopt = xe_res.get('ipopt_'+bound+'_'+approach)
                    res_trust = result.get('trust-constr_'+bound+'_'+approach)
                    res_slsqp = result.get('SLSQP_'+bound+'_'+approach)
                    res_ipopt = result.get('ipopt_'+bound+'_'+approach)
                    table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e}\\ '.format(bound,approach_label,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.nit,res_slsqp.nit,res_ipopt.nit,res_trust.nfev,res_slsqp.nfev,res_ipopt.nfev,error(xe_opt_trust,xe_LF),error(xe_opt_slsqp,xe_LF),error(xe_opt_ipopt,xe_LF)))
                table.write('\hline ')
    for bound in bounds:
        for approach in approaches:
            xe_opt_trust = xe_res.get('trust-constr_'+bound+'_'+approach)
            xe_opt_slsqp = xe_res.get('SLSQP_'+bound+'_'+approach)
            xe_opt_ipopt = xe_res.get('ipopt_'+bound+'_'+approach)
            res_trust = result.get('trust-constr_'+bound+'_'+approach)
            res_slsqp = result.get('SLSQP_'+bound+'_'+approach)
            res_ipopt = result.get('ipopt_'+bound+'_'+approach)
            print('\nBounds: {}, approach: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\ntrust-constr:{}\nSLSQP: {}\nIPOPT: {}\nErrors for t-c: {}, SLSQP: {}, IPOPT: {}'.format(bound,approach,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(xe_opt_trust,xe_LF),error(xe_opt_slsqp,xe_LF),error(xe_opt_ipopt,xe_LF)))

def compare_opf_scaling_sep_LF(dir_path=None,save_tables=False,save_figs=False):
    """Compare OPF for for different scaling options, and for different optimization methods. LF is substituted."""
    if save_tables and dir_path == None:
        raise ValueError('dir_path needs to be specified to save tables or figures')

    # solver info
    tol=1e-6
    max_iter=50
    max_iters_lf=10
    a0=0
    a1=0
    b0=.2
    b1=.3
    c0=1e-5
    c1=2e-5

    #LF solution
    with HiddenPrints():
        elec_net,xe_LF,_,_,_,_,S_inj,P_edge,Q_edge,pp_net = run_load_flow(scale_var=None,tol=tol,max_iter=max_iters_lf)
        if 'sn_kva' in pp_net.keys():
            S_base = pp_net.sn_kva*kW #[va]
        elif 'sn_mva' in pp_net.keys():
            S_base = pp_net.sn_mva*MW #[va]
        V_base = pp_net.bus['vn_kv'][0]*kV #[V]
        scale_var_params = {'Sbase':S_base,'Vbase':V_base,'deltabase':1}
        P0_load = pp_net.load['p_mw'][0]*MW
        Q0_load = pp_net.load['q_mvar'][0]*MW
        V0_BC = pp_net.res_bus['vm_pu'][0]*V_base
        delta0_BC = pp_net.res_bus['va_degree'][0]*degree
        P0_gen_LF = elec_net.nodes[0].half_links[0].get_P() - P0_load #is a source, so the half link power <0
        Q0_gen_LF = elec_net.nodes[0].half_links[0].get_Q() - Q0_load#is a source, so the half link power <0
        Q3_gen_LF = elec_net.nodes[3].half_links[0].get_Q()
        P3_gen_LF = elec_net.nodes[3].half_links[0].get_P()
        V3_LF = elec_net.nodes[3].get_V()

    # initial guesses
    P3_gen=1.3*P3_gen_LF
    V3=1.05*V3_LF
    delta1=0
    delta2=0
    delta3=0
    V1=230*kV
    V2=230*kV

    # bounds
    ineq_constr='all'
    P3_gen_bounds = np.array([2.5*P3_gen_LF,P3_gen_LF])
    V3_bounds = np.array([V3_LF,1.2*V_base])
    P0_gen_bounds = np.array([2.5*P0_gen_LF,0])
    Q0_gen_bounds = np.array([2.5*Q0_gen_LF,0])
    Q3_gen_bounds = np.array([2.5*Q3_gen_LF,0])
    delta1_bounds = np.array([-np.pi,np.pi])
    delta2_bounds = np.array([-np.pi,np.pi])
    delta3_bounds = np.array([-np.pi,np.pi])
    V1_bounds = np.array([0.8*V_base,1.2*V_base])
    V2_bounds = np.array([0.8*V_base,1.2*V_base])

    # BC
    V_BC = True
    delta0 =delta0_BC
    V0 = V0_BC

    result = dict()
    xe_res = dict()
    methods = ['trust-constr','SLSQP','ipopt']
    bounds = ['soft', 'hard']
    approaches = ['direct','adjoint','eq_constr']
    scaling = ['matrix','per_unit']
    fb = 100*MW

    V3_LF_scaled = V3_LF/scale_var_params.get('Vbase')
    P3_LF_scaled = P3_gen_LF/scale_var_params.get('Sbase')
    P0_LF_scaled = P0_gen_LF/scale_var_params.get('Sbase')
    V3_bounds_scaled = V3_bounds/scale_var_params.get('Vbase')
    P3_gen_bounds_scaled = P3_gen_bounds/scale_var_params.get('Sbase')
    for scale_var in scaling:
        if scale_var == 'per_unit':
            f = objective_function(P0_LF_scaled,P3_LF_scaled,a0=a0/fb,a1=a1/fb,b0=b0/(fb/scale_var_params.get('Sbase')),b1=b1/(fb/scale_var_params.get('Sbase')),c0=c0/(fb/scale_var_params.get('Sbase')**2),c1=c1/(fb/scale_var_params.get('Sbase')**2))
        else:
            f = objective_function(P0_gen_LF,P3_gen_LF,a0=a0,a1=a1,b0=b0,b1=b1,c0=c0,c1=c1)/fb
        for bound in bounds:
            if bound == 'soft':
                stay_within_bounds = False
            else:
                stay_within_bounds = True
            max_fev = 0
            # plots
            fig_f = plt.figure('obj_OF_scaling_{}_bounds_{}'.format(scale_var,bound))
            ax_f = fig_f.gca()
            ax_f.set_xlabel('Iteration')
            ax_f.set_ylabel('f')

            fig_P3 = plt.figure('active_power_3_scaling_{}_bounds_{}'.format(scale_var,bound))
            ax_P3 = fig_P3.gca()
            ax_P3.set_xlabel('Iteration')
            ax_P3.set_ylabel(r'$P_{3,0}$')

            fig_V3 = plt.figure('volt_amp_3_scaling_{}_bounds_{}'.format(scale_var,bound))
            ax_V3 = fig_V3.gca()
            ax_V3.set_xlabel('Iteration')
            ax_V3.set_ylabel(r'$|V_3|$')
            for method in methods:
                for approach in approaches:
                    if approach == 'direct' or approach == 'adjoint':
                        approach_legend = approach
                        xe_opt, res, f_vec, P3_vec, V3_vec, _ = run_optimal_load_flow_separate_LF(P3_gen=P3_gen, delta0=delta0, V0=V0, V3=V3, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2, P3_gen_bounds=P3_gen_bounds, V3_bounds=V3_bounds, P0_gen_bounds=P0_gen_bounds, Q0_gen_bounds=Q0_gen_bounds, Q3_gen_bounds=Q3_gen_bounds, delta1_bounds=delta1_bounds, delta2_bounds=delta2_bounds, delta3_bounds=delta3_bounds, V1_bounds=V1_bounds, V2_bounds=V2_bounds, a0=a0, a1=a1, b0=b0, b1=b1, c0=c0, c1=c1, scale_var=scale_var, scale_var_params=scale_var_params, tol=tol, max_iter=max_iter, max_iters_lf=max_iters_lf, V_BC=V_BC, ineq_constr=ineq_constr, approach=approach, optimization_method=method, stay_within_bounds=stay_within_bounds, fb=fb)
                    else:
                        approach_legend = 'an'
                        xe_opt, res, f_vec, P3_vec, V3_vec, _ = run_optimal_load_flow(P3_gen=P3_gen, delta0=delta0, V0=V0, V3=V3, delta1=delta1, delta2=delta2, delta3=delta3, V1=V1, V2=V2, P3_gen_bounds=P3_gen_bounds, V3_bounds=V3_bounds, P0_gen_bounds=P0_gen_bounds, Q0_gen_bounds=Q0_gen_bounds, Q3_gen_bounds=Q3_gen_bounds, delta1_bounds=delta1_bounds, delta2_bounds=delta2_bounds, delta3_bounds=delta3_bounds, V1_bounds=V1_bounds, V2_bounds=V2_bounds, a0=a0, a1=a1, b0=b0, b1=b1, c0=c0, c1=c1, scale_var=scale_var, scale_var_params=scale_var_params, tol=tol, max_iter=max_iter, V_BC=V_BC, ineq_constr=ineq_constr, derivatives=True, optimization_method=method, stay_within_bounds=stay_within_bounds,fb=fb)
                    # save result in dictionaries
                    result[method+'_'+bound+'_'+approach+'_'+scale_var] = res
                    xe_res[method+'_'+bound+'_'+approach+'_'+scale_var] = xe_opt
                    # plot results
                    max_fev = max(max_fev,len(f_vec))
                    ax_f.plot(f_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_P3.plot(P3_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
                    ax_V3.plot(V3_vec,color=colors_method.get(method),marker=markers_bounds.get(bound),ls=linestyles_derivatives.get(approach_legend),alpha=.5)
            ax_f.plot([0,max_fev],[f,f],':r')
            ax_P3.plot([0,max_fev],[P3_LF_scaled,P3_LF_scaled],':r')
            ax_P3.plot([0,max_fev],[P3_gen_bounds_scaled[0],P3_gen_bounds_scaled[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_P3.plot([0,max_fev],[P3_gen_bounds_scaled[1],P3_gen_bounds_scaled[1]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_V3.plot([0,max_fev],[V3_LF_scaled,V3_LF_scaled],':r')
            ax_V3.plot([0,max_fev],[V3_bounds_scaled[0],V3_bounds_scaled[0]],ls=linestyles_contraints.get('bounds'),color='k')
            ax_V3.plot([0,max_fev],[V3_bounds_scaled[1],V3_bounds_scaled[1]],ls=linestyles_contraints.get('bounds'),color='k')

            ax_f.legend(handles=legend_handles)
            ax_P3.legend(handles=legend_handles)
            ax_V3.legend(handles=legend_handles)


    if save_tables:
        path_to_tables = os.path.join(dir_path,'network_data','E4N')
        with open(os.path.join(path_to_tables,'optimizer_info_scaling_sep_LF.txt'), "w") as table:
            for bound in bounds:
                for approach in approaches:
                    if approach == 'eq_constr':
                        approach_label = 'eq. constr.'
                    else:
                        approach_label = approach
                    xe_opt_trust_mat = xe_res.get('trust-constr_'+bound+'_'+approach+'_matrix')
                    xe_opt_slsqp_mat = xe_res.get('SLSQP_'+bound+'_'+approach+'_matrix')
                    xe_opt_ipopt_mat = xe_res.get('ipopt_'+bound+'_'+approach+'_matrix')
                    res_trust_mat = result.get('trust-constr_'+bound+'_'+approach+'_matrix')
                    res_slsqp_mat = result.get('SLSQP_'+bound+'_'+approach+'_matrix')
                    res_ipopt_mat = result.get('ipopt_'+bound+'_'+approach+'_matrix')
                    xe_opt_trust_pu = xe_res.get('trust-constr_'+bound+'_'+approach+'_per_unit')
                    xe_opt_slsqp_pu = xe_res.get('SLSQP_'+bound+'_'+approach+'_per_unit')
                    xe_opt_ipopt_pu = xe_res.get('ipopt_'+bound+'_'+approach+'_per_unit')
                    res_trust_pu = result.get('trust-constr_'+bound+'_'+approach+'_per_unit')
                    res_slsqp_pu = result.get('SLSQP_'+bound+'_'+approach+'_per_unit')
                    res_ipopt_pu = result.get('ipopt_'+bound+'_'+approach+'_per_unit')
                    table.write(r'{} & {} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} & {:.3e} \\ '.format(bound,approach_label,res_trust_mat.success,res_trust_pu.success,res_slsqp_mat.success,res_slsqp_pu.success,res_ipopt_mat.success,res_ipopt_pu.success,res_trust_mat.nit,res_trust_pu.nit,res_slsqp_pu.nit,res_slsqp_mat.nit,res_ipopt_mat.nit,res_ipopt_pu.nit,error(xe_opt_trust_mat,xe_LF),error(xe_opt_slsqp_mat,xe_LF),error(xe_opt_ipopt_mat,xe_LF),error(xe_opt_trust_pu,xe_LF),error(xe_opt_slsqp_pu,xe_LF),error(xe_opt_ipopt_pu,xe_LF)))
                table.write('\hline ')
    for bound in bounds:
        for approach in approaches:
            for scale_var in scaling:
                xe_opt_trust = xe_res.get('trust-constr_'+bound+'_'+approach+'_'+scale_var)
                xe_opt_slsqp = xe_res.get('SLSQP_'+bound+'_'+approach+'_'+scale_var)
                xe_opt_ipopt = xe_res.get('ipopt_'+bound+'_'+approach+'_'+scale_var)
                res_trust = result.get('trust-constr_'+bound+'_'+approach+'_'+scale_var)
                res_slsqp = result.get('SLSQP_'+bound+'_'+approach+'_'+scale_var)
                res_ipopt = result.get('ipopt_'+bound+'_'+approach+'_'+scale_var)
                print('\nBounds: {}, approach: {}, scaling: {}. Success for t-c: {}, SLSQP: {}, IPOPT: {}\ntrust-constr:{}\nSLSQP: {}\nIPOPT: {}\nErrors for t-c: {}, SLSQP: {}, IPOPT: {}'.format(bound,approach,scale_var,res_trust.success,res_slsqp.success,res_ipopt.success,res_trust.message,res_slsqp.message,res_ipopt.message,error(xe_opt_trust,xe_LF),error(xe_opt_slsqp,xe_LF),error(xe_opt_ipopt,xe_LF)))
    if save_figs:
        path_to_fig = os.path.join(dir_path,'Figures','E4N')
        for fig_num in plt.get_figlabels():
            if not 'Network_topology' in fig_num:
                plt.figure(fig_num)
                file_name = fig_num+'.pgf'
                plt.savefig(os.path.join(path_to_fig, file_name))

if __name__ == '__main__':
    tol = 1e-6
    max_iter = 10
    scale_var = 'matrix' # no parameters are provided, so base values of ppnet are used
    elec_net, x_sol, iters, err_vec, delta_sol, V_sol, S_inj, P_edge, Q_edge, pp_net = run_load_flow(scale_var=scale_var,tol=tol,max_iter=max_iter)

    # plot topology
    fig_top = plt.figure('Network_topology')
    ax_top = fig_top.gca()
    elec_net.draw_network(ax_top,halflink_angle=2,halflink_length=.5)
    plt.axis('equal')
    plt.axis('off')

    ## run opf pandapower
    #print(pp_net.load)
    #print(pp_net.gen)
    #print(pp_net.ext_grid)
    ## create cost function
    #pp.create_poly_cost(pp_net,0,"ext_grid",10,cp2_eur_per_mw2=.001)
    #pp.create_poly_cost(pp_net,3,"gen",12,cp2_eur_per_mw2=.0012)
    ## set generators to controllable
    #pp_net.gen['controllable'] [0]= True
    #pp_net.load['controllable'] [0]= True
    #pp_net.ext_grid['max_p_mw'] [0]= 400
    #print(pp_net.load)
    #print(pp_net.gen)
    #print(pp_net.ext_grid)

    #pp.runopp(pp_net,delta=1e-6,numba=False,verbose=True)
    #print(pp_net)

    #opf
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # compare_opf(dir_path,number_runs=5)
    # compare_opf_derivatives(dir_path,number_runs=1,save_tables=False)
    # compare_opf_methods(dir_path=dir_path,save_tables=False)
    # compare_opf_scaling(dir_path=dir_path,save_tables=False)
    # compare_opf_methods_sep_LF(dir_path=dir_path,save_tables=False)
    compare_opf_scaling_sep_LF(dir_path=dir_path,save_tables=False,save_figs=False)


    plt.show()
