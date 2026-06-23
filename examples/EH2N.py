"""
MES consisting of 2 nodes per carrier (i.e. only one link in every single-carrier network), with multiple couplings. Electricity-Heat network.
"""
import warnings
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.carrier import Gas
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.load_flow.system_of_equations import NonLinearSystemElectrical, NonLinearSystemHeat, NonLinearSystemHeterogeneous
from examples.GE2N import layout_convergence, layout_convergence_order, plot_convergence, plot_convergence_order

from meslf.utils.constants import bar, mbar, kV, MW
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.sparse as sps
from meslf.utils.hide_print import HiddenPrints
from meslf.utils.post_processing import error
import os
import pickle

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

colors_carrier = {'heat':'tab:blue','elec':'tab:red','gas':'tab:green','coupling':'tab:gray'}
# colors_solver = {'mes':'tab:blue','dd_dE':'tab:purple','dd_F':'tab:red','dd':'tab:orange','mes reod':'tab:green','block jacobi':'tab:pink','dd fp dy':'tab:pink','dd fp F':'tab:brown','dd fp NR':'tab:olive','dd fp NR FD':'tab:cyan'}
colors_solver = {'mes':'tab:blue','dd':'tab:orange','fp':'tab:purple','mes reod 2':'tab:green','fp NR':'tab:red','fp dec':'tab:pink','fp reod':'tab:brown','fp reod dec':'tab:olive','fp NR reod':'tab:cyan','fp reod 2':'tab:orange','fp reod 2 dec':'tab:grey','fp NR reod 2':'navy'}
# markers_solver = {'mes':'.','dd_dE':'x','dd_F':'*','dd':'s','mes reod':'d','dd fp dy':'o','dd fp F':'+','dd fp NR':'v','dd fp NR FD':'^'}
markers_solver = {'mes':'.','dy':'x','F':'*','an':'o','FD':'s'}

# gas carrier
S = 0.589 # This value to ensure equivalence between different pipe models
Z = 1.
pn = 1.*bar #[Pa]
Tn = 288. #[K]
T = Tn
R = 8.314413 #[J/molK]
M = 28.97e-3 #[kg/mol]
R_air = R/M #[J/kgK]
gas = Gas('low pres gas',S,R_air,Z,pn,Tn,T)
rhon_g = gas.rhon #[kg/m^3]

Sbase = 1*MW #[W]
Vbase = 10/np.sqrt(3)*kV #[V]
Ybase = Sbase/(Vbase**2)
# physical parameters of the lines
b_pu = -10. #[p.u.]
g_pu = 1. #[p.u.]
b = b_pu*Ybase
g = g_pu*Ybase
elec_link_params = {'b':b,'g':g}
elec_link_type = 'short_line'
# coordinates
x0e = 0
y0e = 1
x1e = 0
y1e = 0

# water carrier
rho_w = 960. #[kg/m^3]
Cp_w = 4.182e3 #[J/(kg K)]
grav_const = 9.81 #[m/s^2]
water = Water('water',Cp_w,rho=rho_w)
# physical parameters of network and pipes
Ta = 0.
L_h = 500. #[m]
D_h = .15 #[m]
lam = .2 #[W/(mK)]
U = lam/(np.pi*D_h) #[W/(m^2 K)]
heat_link_type = 'standard_pipe_low_pres_pole'
heat_link_params = {'L':L_h,'U':U,'D':D_h,'carrier':water}
heat_link_type_extra = 'dummy'#'isolated_pipe_low_pres_pole'
heat_link_params_extra = {'carrier':water}#{'L':L_h,'D':D_h,'carrier':water}
# coordinates
x0h = 2
y0h = 1
x1h = 2
y1h = 0

# solution
eta_GG0 = .6
eta_GG1 = .7
GHV = 60134305#[J/kg]
eta_GB0 = .75
eta_GB1 = .8
eta_CHP0 = np.array([eta_GG0,eta_GB0])
eta_CHP1 = np.array([eta_GG1,eta_GB1])
Pc0_sol = 1*Sbase #[W]
Qc0_sol = .5*Sbase #[W]
Pc1_sol = 3.51427623481958*MW #[W]
Qc1_sol = 2.142762348195836*MW #[W]
phic0_sol = 3.014962931574398*MW#[W]
phic1_sol = 1.5*MW #[W]
qc0_sol = (Pc0_sol/eta_CHP0[0] + phic0_sol/eta_CHP0[1])/GHV #[kg/s]
qc1_sol = (Pc1_sol/eta_CHP1[0] + phic1_sol/eta_CHP1[1])/GHV #[kg/s]
V0_sol = 0.93108103*Vbase #[V]
ph1_sol = 99.64192067*rho_w*grav_const #[Pa]
m01_sol = 14.394908119159574 - 9.564801530368245
Ts1_sol = 99.50616179
Tr0_sol = 49.7530809
Tr1_sol = 50.

# physical parameters of the coupling unit
unit_type_EH = 'EH'
C0 = np.array([[Pc0_sol/(Pc0_sol+phic0_sol)],[phic0_sol/(Pc0_sol+phic0_sol)]])
C1 = np.array([[Pc1_sol/(GHV*qc1_sol)],[phic1_sol/(GHV*qc1_sol)]])
unit_params_EH0 = {'C':C0,'GHV':GHV}
unit_params_EH1 = {'C':C1,'GHV':GHV}
# coordinates
x0c = 1
y0c = 1
x1c = 1
y1c = 0

# boundary conditions
V1 = 1.*Vbase #[V]
delta1 = 0 #[rad]
P0_load = 2*Sbase #[W]
Q0_load = 1*Sbase #[W]
P1_load = 2.5*Sbase #[W]
Q1_load = 1.5*Sbase #[W]
P0_gen = Pc0_sol
Q0_gen = Qc0_sol
ph0 = 100*rho_w*grav_const #[Pa]
phi0_sink = 2*MW
phi1_sink = 2.5*MW
To1_sink = 50 #[C]
psi01 = np.exp(-np.pi*D_h*U*L_h/(Cp_w*np.abs(m01_sol)))# temp drop
To0_sink = psi01*To1_sink #50 #[C]
phi0_source = phic0_sol
phi1_source = phic1_sol
Ts0 = 100 #[C] (this is a boundary condition)
Toc0 = Ts0 #[C]
m1_sink = phi1_sink/(Cp_w*(Ts1_sol-To1_sink))
mc1_sol = m1_sink - m01_sol
Toc1 = phic1_sol/(Cp_w*mc1_sol)+Tr1_sol
To1_source = Toc1
m0_sink = phi0_sink/(Cp_w*(Ts0 - To0_sink))
mc0_sol = phic0_sol/(Cp_w*(Toc0-Tr0_sol))

# initial conditions
qc_ic = .6*(qc0_sol+qc1_sol)
Pc_ic = .9*(P0_load+P1_load)#1.5*MW
Qc_ic = 2*MW
phic_ic = .9*(phi0_sink+phi1_sink)#1*MW
m_ic = .5*(m0_sink+m1_sink)#5
m_hl_ic = m_ic
ph_ic = .8*ph0#.99*ph0
Ts_ic = .8*Ts0
Tr_ic = 1.1*To1_sink

def create_electrical_network(scale_var=None,scale_var_params=None):
    """Create an electrical network consisting of one source connected to one sink"""
    if scale_var == 'per_unit':
        P0_c = -P0_gen/scale_var_params.get('Sbase') #(-P0_gen, since P0_gen > 0, but is should be in input)
        P0 = P0_load/scale_var_params.get('Sbase')
        Q0 = Q0_load/scale_var_params.get('Sbase')
        V0_bc = V0_sol/scale_var_params.get('Vbase')
        V1_bc = V1/scale_var_params.get('Vbase')
    else:
        P0_c = -P0_gen #(-P0_gen, since P0_gen > 0, but is should be in input)
        P0 = P0_load
        Q0 = Q0_load
        V0_bc = V0_sol
        V1_bc = V1
    e0 = ElectricalNode('en0',node_type=1,x=x0e,y=y0e,P=P0_c,V=V0_bc) # gen
    ElectricalHalfLink('en0_load',e0,P=P0,Q=Q0,bc_type=1) # P and Q are known
    e1 = ElectricalNode('en1',x=x1e,y=y1e,node_type=0,V=V1_bc,delta=delta1) # slack

    el0 = ElectricalLink('el0',e0,e1,link_type=elec_link_type,link_params=elec_link_params)

    elec_net = ElectricalNetwork('2 nodes')
    elec_net.add_link(el0)
    return elec_net

def set_xe_LF_init(elec_net,scale_var=None,scale_var_params=None):
    """Initialize the electrical network, consisting of one source connected to one sink.

    Parameters
    ----------
    elec_net : ElectricalNetwork
        The electrical network to be initialized

    Returns
    -------
    x0 : np array
        initial guess
    """
    x_entries, unknown_delta_nodes, unknown_V_nodes = elec_net.get_x_entries()

    delta_init = np.zeros(len(unknown_delta_nodes)) # default flat initialization of 0
    V_init = Vbase*np.ones(len(unknown_V_nodes)) # default flat initialization corresponding to 1 p.u.
    if scale_var == 'per_unit':
        V_init /= scale_var_params.get('Vbase')
    x_init = np.concatenate((delta_init,V_init))

    elec_net.initialize()
    elec_net.update(x_init)
    x0 = elec_net.set_x_init()
    return x0

def run_LF_elec(max_iter=10,tol=1e-6,scale_var=None,scale_var_params=None):
    """Steady-state load flow analysis of electrical network. The default values are used for initialization.
    """
    # create network
    elec_net = create_electrical_network(scale_var=scale_var,scale_var_params=scale_var_params)

    x0 = set_xe_LF_init(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)

    # solve network
    print('\nRunning load flow for single-carrier electrical network')
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution after {} iterations (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print('delta = {}'.format(delta_sol))
    print('|V| = {} V'.format(V_sol))
    print('|V| = {} p.u.'.format(V_sol/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('elec hl start nodes = {}'.format([hl.start_node.name for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('P1 source = {} MW'.format(elec_net.nodes[1].half_links[0].P/MW - P1_load/MW)) # is negative
    print('Q1 source = {} MW'.format(elec_net.nodes[1].half_links[0].Q/MW - Q1_load/MW)) # is negative
    return elec_net,x_sol,iters,err_vec

def update_bc_elec(elec_net,P0_c,scale_var=None,scale_var_params=None):
    """Update the BCs of the single-carrier electrical network, based on the interface conditions"""
    if scale_var == 'per_unit':
        P0_c = P0_c*scale_var_params.get('Sbase')
    elec_net.nodes[0].half_links[0].P = P0_c
    return elec_net

def get_bc_elec(elec_net,scale_var=None,scale_var_params=None):
    """Get the BCs of the single-carrier electrical network, which is updated based on the interface conditions"""
    P0_c = elec_net.nodes[0].half_links[0].get_P(scale_var_params=scale_var_params)
    return P0_c

def G_elec(elec_net,scale_var=None,scale_var_params=None):
    """The additional equations needed for full LF in the single-carrier electrical network"""
    if scale_var == 'per_unit':
        Q0 = Q0_load/scale_var_params.get('Sbase')
        P1 = P1_load/scale_var_params.get('Sbase')
        Q1 = Q1_load/scale_var_params.get('Sbase')
    else:
        Q0 = Q0_load
        P1 = P1_load
        Q1 = Q1_load
    P1_c = -elec_net.links[0].get_Pend(scale_var=scale_var,scale_var_params=scale_var_params) - P1
    Q0_c = -elec_net.links[0].get_Qstart(scale_var=scale_var,scale_var_params=scale_var_params) - Q0
    Q1_c = -elec_net.links[0].get_Qend(scale_var=scale_var,scale_var_params=scale_var_params) - Q1
    return P1_c,Q0_c,Q1_c

def reset_network_without_update_elec(elec_net):
    """Reset the heterogeneous network, but do not update the network values. That is, remove the half link connected to the electrical slack node, and set Q of electrical generator to zero."""
    # gen node
    elec_net.nodes[0].half_links[0].Q = 0
    # slack node
    elec_net.nodes[1].half_links = list()
    return elec_net

def h_elec(elec_net,ve,be,max_iter_lf=10,tol_lf=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,solve_LF=True):
    """The LF solve used in DD"""
    if solve_LF:
        P0_c = be[0]
        elec_net = update_bc_elec(elec_net,P0_c,scale_var=scale_var,scale_var_params=scale_var_params)
        elec_net = reset_network_without_update_elec(elec_net)
        with HiddenPrints():
            xe,iterse,err_vece,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol_lf,max_iter_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
    else:
        elec_net = reset_network_without_update_elec(elec_net)
        nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
        xe = elec_net.set_x_init(formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
        Fe = nlsyse.DF().dot(nlsyse.F(xe))
        err_vece = np.array([np.linalg.norm(Fe)])
    P1_c,Q0_c,Q1_c = G_elec(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    return elec_net, np.array([P1_c,Q0_c,Q1_c]), err_vece

def hIF_elec(be,vc,scale_var=None,scale_var_params=None):
    """The IFCs to determine the BCs used in electrical SC network"""
    P0_c = np.array([0, 0, -1, 0]).dot(vc)
    return np.array([P0_c])

def create_heat_network(scale_var=None,scale_var_params=None):
    """Create a heat network consisting of two nodes, both with a source and a sink. Additional nodes and (dummy) links are used"""
    if scale_var == 'per_unit':
        raise NotImplementedError('Heat network is not implemented for p.u. scaling')
    h0 = HeatNode('hn0',node_type=0,x=x0h,y=y0h,p=ph0,Ts=Ts0) # slack
    h0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h1 = HeatNode('hn1',node_type=1,x=x1h,y=y1h,Tr_hl=To1_sink,dphi=phi1_sink) # load node (sink)
    h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    if heat_link_type_extra == 'dummy':
        h2 = HeatNode('hn0_extra',node_type=3,x=x0h+.5,y=y0h+.5,Tr_hl=To0_sink,dphi=phi0_sink,p=ph0) # ref. load node (sink) (since only connected with a dummy link, so no way to determine pressure)
        h3 = HeatNode('hn1_extra',node_type=3,x=x1h-.5,y=y1h-.5,Ts_hl=To1_source,dphi=-phi1_source,p=ph1_sol) # ref. load node (source) (since only connected with a dummy link, so no way to determine pressure)
    else:
        h2 = HeatNode('hn0_extra',node_type=1,x=x0h+.5,y=y0h+.5,Tr_hl=To0_sink,dphi=phi0_sink) # load node (sink)
        h3 = HeatNode('hn1_extra',node_type=1,x=x1h-.5,y=y1h-.5,Ts_hl=To1_source,dphi=-phi1_source) # load node (source)
    h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h3.half_links[0].set_type('heat_exchanger',{'carrier':water})

    hl0 = HeatLink('hl0',h0,h1,link_type = heat_link_type,link_params = heat_link_params)
    hl1 = HeatLink('hl1',h0,h2,link_type = heat_link_type_extra,link_params = heat_link_params_extra)
    hl2 = HeatLink('hl2',h3,h1,link_type = heat_link_type_extra,link_params = heat_link_params_extra)

    heat_net = HeatNetwork('2 nodes',Ta=Ta)
    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    return heat_net

def set_xh_LF_init(heat_net,scale_var=None,scale_var_params=None,formulation='half_link_flow'):
    """Initialize the heat network consisting of two nodes, both with a source and a sink. Additional nodes and (dummy) links are used

    Parameters
    ----------
    heat_net : HeatNetwork
        The heat network to be initialized

    Returns
    -------
    x0 : np array
        Initial guess
    """
    if scale_var == 'per_unit':
        raise NotImplementedError('Initialization of heat network not implemented for p.u. scaling')
    if formulation == 'half_link_flow':
        m_init = np.array([m_ic,m_ic,2*m_ic]) #[kg/s]
        m_hl_init = np.array([5*m_hl_ic,m_hl_ic,-m_hl_ic/2]) #[kg/s]
        Ts_init = np.array([Ts_ic,Ts_ic,Ts_ic]) #[C]
        Tr_init = np.array([Tr_ic,Tr_ic,Tr_ic,Tr_ic]) #[C]
        if heat_link_type_extra == 'dummy':
            p_init = np.array([ph_ic])
        else:
            p_init = np.array([ph_ic,ph_ic,ph0])
    else:
        raise NotImplementedError('Initialization of heat network not implemented for standard formulation')
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))

    heat_net.initialize()
    heat_net.update(x_init,formulation=formulation)
    x0 = heat_net.set_x_init(formulation=formulation)
    return x0

def run_LF_heat(max_iter=10,tol=1e-6,scale_var=None,scale_var_params=None,formulation='half_link_flow'):
    """Steady-state load flow analysis of heat network. The default values are used for initialization.
    """
    # create network
    heat_net = create_heat_network(scale_var=scale_var,scale_var_params=scale_var_params)

    x0 = set_xh_LF_init(heat_net,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)

    # solve network
    print('\nRunning load flow for single-carrier heat network')
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)

    print('Solution after {} iterations (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print('p heat = {} m'.format(p_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {}'.format(m_hl_vec))
    print('Ts hl = {}'.format(Ts_hl_vec))
    print('Tr hl = {}'.format(Tr_hl_vec))
    print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
    return heat_net,x_sol,iters,err_vec

def update_bc_heat(heat_net,dphi1_c,scale_var=None,scale_var_params=None):
    """Update the BCs of the single-carrier heat network, based on the interface conditions"""
    if scale_var == 'per_unit':
        dphi1_c = dphi1_c*scale_var_params.get('phibase')
    heat_net.nodes[3].half_links[0].dphi = dphi1_c
    return heat_net

def get_bc_heat(heat_net,scale_var=None,scale_var_params=None):
    """Get the BCs of the single-carrier heat network, which is updated based on the interface conditions"""
    dphi1_c = heat_net.nodes[3].half_links[0].get_dphi(scale_var_params=scale_var_params)
    return dphi1_c

def G_heat(heat_net,scale_var=None,scale_var_params=None):
    """The additional equations needed for full LF in the single-carrier heat network"""
    dphi0_c = heat_net.nodes[0].half_links[0].get_dphi(scale_var_params=scale_var_params)
    m0_c = heat_net.nodes[0].half_links[0].get_m(scale_var_params=scale_var_params)
    return m0_c,dphi0_c

def reset_network_without_update_heat(heat_net):
    """Reset the heat network, but do not update the network values. That is, set mass flow of half link of slack node to zero."""
    # slack node
    heat_net.nodes[0].half_links[0].m = 0
    return heat_net

def h_heat(heat_net,vh,bh,max_iter_lf=10,tol_lf=1e-6,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},scale_var=None,scale_var_params=None,solve_LF=True):
    """The LF solve used in DD"""
    if formulation.get('heat') == 'standard':
        raise NotImplementedError('h_heat not implemented for standard formulation')
    if solve_LF:
        dphi1_c = bh[0]
        heat_net = update_bc_heat(heat_net,dphi1_c,scale_var=scale_var,scale_var_params=scale_var_params)
        heat_net = reset_network_without_update_heat(heat_net)
        with HiddenPrints():
            xh,itersh,err_vech,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol_lf,max_iter_lf,solver='NR',formulation=formulation.get('heat'))
    else:
        heat_net = reset_network_without_update_heat(heat_net)
        nlsysh = NonLinearSystemHeat(heat_net,formulation=formulation.get('heat'),scale_var=scale_var,scale_var_params=scale_var_params)
        xh = heat_net.set_x_init(formulation=formulation.get('heat'),scale_var=scale_var,scale_var_params=scale_var_params)
        Fh = nlsysh.DF().dot(nlsysh.F(xh))
        err_vech = np.array([np.linalg.norm(Fh)])
    m0_c,dphi0_c = G_heat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params)
    if formulation.get('heat') == 'half_link_flow':
        m1_c = xh[5]
    return heat_net, np.array([m0_c,m1_c,dphi0_c]), err_vech

def hIF_heat(bh,vc,scale_var=None,scale_var_params=None):
    """The IFCs to determine the BCs used in heat SC network"""
    dphi1_c = np.array([0, 0, 0, -1]).dot(vc)
    return np.array([dphi1_c])

def create_mes_network():
    """Create a combined electricity and heat network, with two coupling nodes"""
    # create single-carrier networks
    elec_net = create_electrical_network()
    en0 = elec_net.nodes[0]
    en1 = elec_net.nodes[1]
    en0.node_type = 6 # PQV
    en0.V = V0_sol
    en0.remove_half_link(en0.half_links[0]) #remove the half link representing the coupling
    en1.node_type = 5 # PQVdelta
    ElectricalHalfLink('en1_hl',en1,P=P1_load,Q=Q1_load,bc_type=1) # was slack, so didn't have a half link.

    h0 = HeatNode('hn0',node_type=3,x=x0h,y=y0h,p=ph0,Tr_hl=To0_sink,dphi=phi0_sink) #ref. load (sink) node
    h0.half_links[0].set_type('heat_exchanger',{'carrier':water},bc_type=3) #phi and To known (sink)
    h1 = HeatNode('hn1',node_type=1,x=x1h,y=y1h,Tr_hl=To1_sink,dphi=phi1_sink) # load node (sink)
    h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hl0 = HeatLink('hl0',h0,h1,link_type = heat_link_type,link_params = heat_link_params)
    heat_net = HeatNetwork('2 nodes',Ta=Ta)
    heat_net.add_link(hl0)

    # coupling
    c0 = HeterogeneousNode('cn0',node_type=1,x=x0c,y=y0c,unit_type=unit_type_EH,unit_params=unit_params_EH0) # To known
    c0_hl_gas = GasHalfLink('EH0_hlg',c0,-qc_ic,bc_type=0) # q unknown (q is negative)
    c1 = HeterogeneousNode('cn1',node_type=1,x=x1c,y=y1c,unit_type=unit_type_EH,unit_params=unit_params_EH1) # To known
    c1_hl_gas = GasHalfLink('EH1_hlg',c1,-qc_ic,bc_type=0) # q unknown (q is negative)

    # dummy links
    el1 = ElectricalLink('el1',c0,en0)
    el2 = ElectricalLink('el2',c1,en1)
    elec_net.add_link(el1)
    elec_net.add_link(el2)
    hl1 = HeatLink('hl1',c0,h0,link_params={'carrier':water},bc_type=6,Tsstart=Toc0) # To of coupling (source) is known
    hl2 = HeatLink('hl2',c1,h1,link_params={'carrier':water},bc_type=6,Tsstart=Toc1) # To of coupling (source) is known
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)

    # mes
    het_net = HeterogeneousNetwork('2 nodes')
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    het_net.add_node(c0)
    het_net.add_node(c1)

    return het_net, elec_net, heat_net

def set_xmes_LF_init(het_net,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'}):
    """Initialize the combined electrical and heat network, with two coupling nodes.

    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The multi-carrier network to be initialized

    Returns
    -------
    x0 : np array
        initial guess
    """
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)

    # electrical part
    delta_init = np.zeros(len(unknown_delta_nodes)) # default flat initialization of 0
    V_init = Vbase*np.ones(len(unknown_V_nodes)) # default flat initialization corresponding to 1 p.u.
    xe_init = np.concatenate((delta_init,V_init))

    # heat part
    m_init = m_ic*np.ones(len(unknown_m_links)) #[kg/s]
    m_hl_init = m_hl_ic*np.ones(len(unknown_m_halflinks)) #[kg/s] All half links are sinks, so all have the same sign
    p_init = ph_ic*np.ones(len(unknown_p_nodes))
    Ts_init = Ts_ic*np.ones(len(unknown_Ts_nodes))
    Tr_init = Tr_ic*np.ones(len(unknown_Tr_nodes))
    Ts_hl_init = Ts_ic*np.ones(len(unknown_Ts_halflinks))
    Tr_hl_init = Tr_ic*np.ones(len(unknown_Tr_halflinks))
    xh_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init,Ts_hl_init,Tr_hl_init))

    # coupling
    qc_init = qc_ic*np.ones(len(unknown_qc_halflinks))
    Pc_init = Pc_ic*np.ones(len(unknown_Pc_links))
    Qc_init = Qc_ic*np.ones(len(unknown_Qc_links))
    Sc_init = np.concatenate((Pc_init,Qc_init))
    mc_init = m_ic*np.ones(len(unknown_mc_links))
    phic_init = phic_ic*np.ones(len(unknown_dphi_links))
    Toc_init = Ts_ic*np.ones(len(unknown_Ts_links))
    xc_init = np.concatenate((qc_init,Sc_init,mc_init,phic_init,Toc_init))
    x_init = np.concatenate((xe_init,xh_init,xc_init))
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)
    return x0

def trans_matr_mes():
    """Permutation and scaling matrices for the integrated system to mimic the DD system."""
    Dx = sps.diags([1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1])
    # the column is moved to the row
    xlength = 19
    Px_row = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    Px_col = np.array([0,12,13,14,1,3,2,16,4,5,6,7,8,15,17,9,10,11,18])
    Px_data = np.ones(xlength)
    Px = sps.csr_matrix((Px_data,(Px_row,Px_col)),shape=(xlength,xlength))

    Flength = xlength
    DF = sps.eye(Flength)
    PF_row = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    PF_col = np.array([0,1,2,3,5,6,7,8,9,10,12,11,18,4,17,13,14,15,16])
    PF_data = np.ones(xlength)
    PF = sps.csr_matrix((PF_data,(PF_row,PF_col)),shape=(xlength,xlength))
    return Dx, Px, DF, PF

def jacobians(het_net, x,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None},scale_var=None,scale_var_params=None):
    """Plot Jacobian matrices, and eigenvalue spectra, for different indices / ordering"""
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsys_unscaled = NonLinearSystemHeterogeneous(het_net,formulation=formulation)

    # unscaled system
    fig_J = nlsys_unscaled.spy_plot_J(x,title='Jacobian spy plot, unscaled')
    ax_J = plt.gca()
    fig_J_map = nlsys_unscaled.imshow_J(x,title='Jacobian, unscaled')
    # spectrum of Jacobian
    fig_spec = nlsys_unscaled.spectrum_J(x,title='Spectrum, unscaled',color=colors_solver.get('mes'))
    ax_spec = fig_spec.gca()

    # scaled system
    if scale_var != None:
        # spy plot of scaled J over original
        nlsys.spy_plot_J(x,ax=ax_J,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5)
        # colormap of Jacobian
        fig_J_scaled_map = nlsys.imshow_J(x,title='Scaled Jacobian')
        # spectrum of (scaled) Jacobian
        fig_spectra = nlsys.spectrum_J(x,title='Scaled spectra',color=colors_solver.get('mes'))
        ax_spectra = fig_spectra.gca()

    # reordered system, unscaled
    Dx, Px, DF, PF = trans_matr_mes()
    Tx = Px.dot(Dx)
    TF = PF.dot(DF)
    fig_J_re = nlsys_unscaled.spy_plot_J(x,title='Reordered Jacobian spy plot, unscaled',P_F=TF,P_x=Tx,overlay=False)
    ax_J_re = plt.gca()
    nlsys_unscaled.spectrum_J(x,ax=ax_spec,P_F=TF,P_x=Tx,color=colors_solver.get('mes reod 2'))

    # reordered system, scaled
    if scale_var != None:
        # colormap of Jacobian
        fig_J_re_scaled_map = nlsys.imshow_J(x,P_F=TF,P_x=Tx,title='Reordered, scaled, Jacobian',overlay=False)
        ax_J_re_scaled_map = fig_J_re_scaled_map.gca()
        # spectrum of (scaled) Jacobian
        nlsys.spectrum_J(x,ax=ax_spectra,P_F=TF,P_x=Tx,color=colors_solver.get('mes reod 2'))

    # plot overlay / lines in reoreded Jacobians
    xe_len = 4
    xh_len = 11
    xc_len = 4
    x_len = xe_len+xh_len+xc_len
    if scale_var !=None:
        axes = [ax_J_re,ax_J_re_scaled_map]
    else:
        axes = [ax_J_re]
    for ax in axes:
        ax.plot((-0.5,x_len-0.5),(xe_len-0.5,xe_len-0.5),'k-')
        ax.plot((-0.5,x_len-0.5),(xe_len+xh_len-0.5,xe_len+xh_len-0.5),'k-')
        ax.plot((xe_len-0.5,xe_len-0.5),(0-0.5,x_len-0.5),'k-')
        ax.plot((xe_len+xh_len-0.5,xe_len+xh_len-0.5),(0-0.5,x_len-0.5),'k-')

def run_LF_mes(max_iter=10,tol=1e-6,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None},scale_var=None,scale_var_params=None,plot_top=False,plot_jac=False,plot_sol=False,reorder=False):
    """Steady-state load flow analysis of combined gas and electrical network, without scaling. The default values are used for initialization.
    """
    if scale_var == 'per_unit':
        raise ValueError('LF mes not implemented for p.u. scaling')
    # create network
    with HiddenPrints():
        het_net, elec_net, heat_net = create_mes_network()

    # initialize network
    x0 = set_xmes_LF_init(het_net,formulation=formulation)
    if plot_jac:
        if reorder:
            jacobians(het_net,x0,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
            nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation)
            fig_J = nlsys.spy_plot_J(x0,title='Jacobian spy plot, scaling = {}'.format(scale_var))
            fig_J_map = nlsys.imshow_J(x0,title='Colormap Jacobian, scaling = {}'.format(scale_var))

    # solve network
    print('\nRunning load flow for multi-carrier gas-electricity network')
    if reorder:
        Dx, Px, DF, PF = trans_matr_mes()
        Tx = Px.dot(Dx)
        TF = PF.dot(DF)
    else:
        TF=np.array([])
        Tx=np.array([])
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,P_F=TF,P_x=Tx)

    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))

    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format([hl.get_m() for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('Ts hl = {} C'.format([hl.get_Ts() for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('Tr hl = {} C'.format([hl.get_Tr() for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('qc = {} kg/s'.format(qc_vec))
    print('Pc = {} MW'.format([Pc/MW for Pc in Pc_vec]))
    print('phi c = {} MW'.format([phic/MW for phic in phic_vec]))

    if plot_top:
        # plot topology
        fig_top = plt.figure('Network topology')
        ax_top = fig_top.gca()
        het_net.draw_network(ax_top,halflink_angle=2,halflink_length=.5)
        plt.axis('equal')
        plt.axis('off')

    if plot_sol:
        # plot solution
        fig_sol = plt.figure('Network solution, scaling = {}'.format(scale_var))
        ax_sol = fig_sol.gca()
        het_net.draw_network_value(ax_sol,halflink_angle=2,halflink_length=.5)
        plt.axis('equal')
        plt.axis('off')

    return het_net, elec_net, heat_net, x_sol, iters, err_vec

def create_coupling_network(scale_var=None,scale_var_params=None):
    """Create a coupling network consisting of two nodes, representing EH's, for an electicity-heat coupling"""
    if scale_var == 'per_unit':
        qc0 = -qc_ic/scale_var_params.get('qbase')
        qc1 = -qc_ic/scale_var_params.get('qbase')
        Pc0 = Pc_ic/scale_var_params.get('Sbase')
        Qc0 = Qc0_sol/scale_var_params.get('Sbase')
        qc1 = -qc_ic/scale_var_params.get('qbase')
        Pc1 = Pc1_sol/scale_var_params.get('Sbase')
        Qc1 = Qc1_sol/scale_var_params.get('Sbase')
        phic0 = phic0_sol/scale_var_params.get('phibase')
        phic1 = phic_ic/scale_var_params.get('phibase')
        mc0 = mc0_sol/scale_var_params.get('mbase')
        mc1 = mc1_sol/scale_var_params.get('mbase')
        Tsc0 = Toc0/scale_var_params.get('Tbase')
        Tsc1 = Toc1/scale_var_params.get('Tbase')
        Trc0 = To0_sink/scale_var_params.get('Tbase')
        Trc1 = To1_sink/scale_var_params.get('Tbase')
    else:
        qc0 = -qc_ic
        qc1 = -qc_ic
        Qc0 = Qc0_sol
        qc1 = -qc_ic
        Pc1 = Pc1_sol
        Qc1 = Qc1_sol
        phic0 = phic0_sol
        phic1 = phic_ic
        mc0 = mc0_sol
        mc1 = mc1_sol
        Tsc0 = Toc0
        Tsc1 = Toc1
        Trc0 = To0_sink
        Trc1 = To1_sink
    c0 = HeterogeneousNode('EH0',node_type=5,x=x0c,y=y0c,unit_type=unit_type_EH,unit_params=unit_params_EH0)# To known, without a heat power equation
    c0_hl_heat = HeatHalfLink('EH0_hlh',c0,link_type='heat_exchanger',link_params={'carrier':water},bc_type=14,m=-mc0,Ts=Tsc0,dphi=-phic0) # m, dphi and Ts known (source)
    c0_hl_heat.Tr = Trc0 # Set to solution
    c0_hl_gas = GasHalfLink('EH0_hlg',c0,qc0,bc_type=0) # q unknown (q is negative)
    c0_hl_elec = ElectricalHalfLink('EH0_hle',c0,Q=Qc0,bc_type=3) # P unknown and Q known

    c1 = HeterogeneousNode('EH1',node_type=5,x=x1c,y=y1c,unit_type=unit_type_EH,unit_params=unit_params_EH1) # To known, without a heat power equation
    c1_hl_heat = HeatHalfLink('EH1_hlh',c1,link_type='heat_exchanger',link_params={'carrier':water},bc_type=18,m=-mc1,Ts=Tsc1,dphi=-phic_ic) #m, Ts known (source)
    c1_hl_heat.Tr = Trc1 # Set to solution
    c1_hl_gas = GasHalfLink('EH1_hlg',c1,qc1,bc_type=0) # q unknown (q is negative)
    c1_hl_elec = ElectricalHalfLink('EH1_hle',c1,P=Pc1,Q=Qc1,bc_type=1) # P  and Q known

    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)
    coupling_net.add_node(c1)
    return coupling_net

def set_xc_LF_init(coupling_net,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},scale_var=None,scale_var_params=None):
    """Initialize the coupling network consisting of two nodes, for a gas-electricity coupling

    Parameters
    ----------
    coupling_net : HeterogeneousNetwork
        The coupling network to be initialized

    Returns
    -------
    x0 : np array
        initial guess
    """
    x_init = np.array([qc_ic,qc_ic,Pc_ic,phic_ic])
    if scale_var == 'per_unit':
        x_init /= np.array([scale_var_params.get('qbase'),scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('phibase')])
    coupling_net.initialize()
    print('set x init: {}'.format(coupling_net.set_x_init(formulation,formulation)))
    print('x init: {}'.format(x_init))
    coupling_net.update(x_init,formulation=formulation)
    x0 = coupling_net.set_x_init(formulation=formulation)
    return x0

def run_LF_coupling(max_iter=10,tol=1e-6,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},scale_var=None,scale_var_params=None):
    """Steady-state load flow analysis for the coupling network for the gas-electricity"""
    # create network
    with HiddenPrints():
        coupling_net = create_coupling_network(scale_var=scale_var,scale_var_params=scale_var_params)

    # initialize network
    with HiddenPrints():
        x0 = set_xc_LF_init(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # solve network
    print('\nRunning load flow for gas-electricity network, only coupling')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution after {} iterations (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print('qc = {} kg/s'.format([-hl.get_q() for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,GasHalfLink)]))
    print('Pc = {} MW'.format([hl.get_P()/MW for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,ElectricalHalfLink)]))
    print('Qc = {} MW'.format([hl.get_Q()/MW for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,ElectricalHalfLink)]))
    print('dphic = {} MW'.format([hl.get_dphi()/MW for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,HeatHalfLink)]))
    print('mc = {} kg/s'.format([hl.get_m() for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,HeatHalfLink)]))
    print('Ts = {}'.format([hl.get_Ts() for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,HeatHalfLink)]))
    return coupling_net, x_sol, iters, err_vec

def update_bc_coupling(coupling_net,Pc1,Qc0,Qc1,m0c,m1c,dphi0c,scale_var=None,scale_var_params=None):
    """Update the BCs of the coupling network, based on the interface conditions"""
    if scale_var == 'per_unit':
        Pc1 = Pc1*scale_var_params.get('Sbase')
        Qc0 = Qc0*scale_var_params.get('Sbase')
        Qc1 = Qc1*scale_var_params.get('Sbase')
        m0c = m0c*scale_var_params.get('mbase')
        m1c = m1c*scale_var_params.get('mbase')
        dphi0c = dphi0c*scale_var_params.get('phibase')
    coupling_net.nodes[0].half_links[2].Q = Qc0
    coupling_net.nodes[1].half_links[2].P = Pc1
    coupling_net.nodes[1].half_links[2].Q = Qc1
    coupling_net.nodes[0].half_links[0].m = m0c
    coupling_net.nodes[1].half_links[0].m = m1c
    coupling_net.nodes[0].half_links[0].dphi = dphi0c
    return coupling_net

def get_bc_coupling(coupling_net,scale_var=None,scale_var_params=None):
    """Get the BCs of the coupling network, which is updated based on the interface conditions"""
    Qc0 = coupling_net.nodes[0].half_links[2].get_Q(scale_var=scale_var,scale_var_params=scale_var_params)
    Pc1 = coupling_net.nodes[1].half_links[2].get_P(scale_var=scale_var,scale_var_params=scale_var_params)
    Qc1 = coupling_net.nodes[1].half_links[2].get_Q(scale_var=scale_var,scale_var_params=scale_var_params)
    m0c = coupling_net.nodes[0].half_links[0].get_m(scale_var=scale_var,scale_var_params=scale_var_params)
    m1c = coupling_net.nodes[1].half_links[0].get_m(scale_var=scale_var,scale_var_params=scale_var_params)
    dphi0c = coupling_net.nodes[0].half_links[0].get_dphi(scale_var=scale_var,scale_var_params=scale_var_params)
    return Pc1, Qc0, Qc1, m0c, m1c, dphi0c

def h_coupling(coupling_net,vc,bc,max_iter_lf=10,tol_lf=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,solve_LF=True):
    """The LF solve used in DD"""
    if solve_LF:
        Pc1, Qc0, Qc1, m0c, m1c, dphi0c = bc
        coupling_net = update_bc_coupling(coupling_net,Pc1,Qc0,Qc1,m0c,m1c,dphi0c,scale_var=scale_var,scale_var_params=scale_var_params)
        with HiddenPrints():
            xc,itersc,err_vecc,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = coupling_net.solve_network(tol_lf,max_iter_lf,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    else:
        xc = vc.copy()
        with HiddenPrints():
            nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            xc = coupling_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            Fc = nlsysc.DF().dot(nlsysc.F(xc))
        err_vecc = np.array([np.linalg.norm(Fc)])
    return coupling_net, xc, err_vecc

def hIF_coupling(bc,ve,vh,scale_var=None,scale_var_params=None):
    """The IFCs to determine the BCs used in electrical SC network"""
    Pc1,Qc0,Qc1 = -np.eye(len(ve)).dot(ve)
    m0c,m1c,dphi0c = np.eye(len(vh)).dot(vh)
    return np.array([Pc1,Qc0,Qc1,m0c,m1c,dphi0c])

def error_measure_F(elec_net,heat_net,coupling_net,P0_c,dphi1_c,Pc1,Qc0,Qc1,m0c,m1c,dphi0c,xe,xh,xc,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},scale_var=None,scale_var_params=None):
    """The error measure for DD of a electricity-heat coupling, based on (residual of) the system of equations"""
    # update all BCs
    elec_net = update_bc_elec(elec_net,P0_c,scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = reset_network_without_update_elec(elec_net)
    heat_net = update_bc_heat(heat_net,dphi1_c,scale_var=scale_var,scale_var_params=scale_var_params)
    heat_net = reset_network_without_update_heat(heat_net)
    coupling_net = update_bc_coupling(coupling_net,Pc1,Qc0,Qc1,m0c,m1c,dphi0c,scale_var=scale_var,scale_var_params=scale_var_params)
    # determine the value of (residual of) the system of equations
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysh = NonLinearSystemHeat(heat_net,formulation=formulation.get('heat'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    Fe = nlsyse.DF().dot(nlsyse.F(xe))
    Fh = nlsysh.DF().dot(nlsysh.F(xh))
    Fc = nlsysc.DF().dot(nlsysc.F(xc))
    error = np.linalg.norm(np.concatenate((Fe,Fh,Fc)))
    return error, Fe, Fh, Fc

def get_xLF_FP(elec_net,heat_net,coupling_net,vc,bc,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},scale_var=None,scale_var_params=None):
    """get the LF variables when FP iteration is used"""
    if formulation.get('heat')=='standard':
        raise NotImplementedError('get_xLF_FP not implemented for standard formulation in heat')
    xe = elec_net.set_x_init(formulation=formulation.get('elec'))
    xh = heat_net.set_x_init(formulation=formulation.get('heat'))
    Ts0 = heat_net.nodes[0].get_Ts(scale_var=scale_var,scale_var_params=scale_var_params)
    if formulation.get('heat')=='half_link_flow':
        xh_mes = np.array([xh[0],xh[4],xh[3],xh[6],Ts0,xh[7],xh[10],xh[11]])
    xc = coupling_net.set_x_init(formulation=formulation)
    min_q0,min_q1,Pc0,min_dphi1c = vc
    Pc1,Qc0,Qc1,m0c,m1c,dphi0c = bc
    xc_mes = np.array([min_q0,min_q1,Pc0,Pc1,Qc0,Qc1,-m0c,-m1c,-dphi0c,min_dphi1c])
    x_mes = np.concatenate((xe,xh_mes,xc_mes))
    return xe,xh,xc,xc_mes,x_mes

def get_suby_FP(y,ordering=1):
    """Split y into the subvectors"""
    len_ve = 3
    len_be = 1
    len_vh = 3
    len_bh = 1
    len_vc = 4
    len_bc = 6
    if ordering == 1:
        ve = y[:len_ve]
        be = y[len_ve:len_ve+len_be]
        vh = y[len_ve+len_be:len_ve+len_be+len_vh]
        bh = y[len_ve+len_be+len_vh:len_ve+len_be+len_vh+len_bh]
        vc = y[-len_vc-len_bc:-len_bc]
        bc = y[-len_bc:]
    elif ordering == 2:
        ve = y[:len_ve]
        vh = y[len_ve:len_ve+len_vh]
        vc = y[len_ve+len_vh:len_ve+len_vh+len_vc]
        be = y[len_ve+len_vh+len_vc:len_ve+len_vh+len_vc+len_be]
        bh = y[-len_bh-len_bc:-len_bc]
        bc = y[-len_bc:]
    elif ordering == 3:
        ve = y[:len_ve]
        vh = y[len_ve:len_ve+len_vh]
        bc = y[len_ve+len_vh:len_ve+len_vh+len_bc]
        vc = y[len_ve+len_vh+len_bc:len_ve+len_vh+len_bc+len_vc]
        be = y[-len_be-len_bh:-len_bh]
        bh = y[-len_bh:]
    return ve, be, vh, bh, vc, bc

def get_y_FP(elec_net,heat_net,coupling_net,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},scale_var=None,scale_var_params=None,ordering=1):
    """Set y used in FP iteration"""
    P1_c,Q0_c,Q1_c = G_elec(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    ve = np.array([P1_c,Q0_c,Q1_c])
    P0_c = get_bc_elec(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    be = np.array([P0_c])
    m0_c,dphi0_c = G_heat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params)
    m1_c = heat_net.nodes[3].half_links[0].get_m(scale_var_params=scale_var_params)
    vh = np.array([m0_c,m1_c,dphi0_c])
    dphi1_c = get_bc_heat(heat_net,scale_var=scale_var,scale_var_params=scale_var_params)
    bh = np.array([dphi1_c])
    with HiddenPrints():
        vc = coupling_net.set_x_init(formulation=formulation)
    Pc1, Qc0, Qc1, m0c, m1c, dphi0c = get_bc_coupling(coupling_net,scale_var=scale_var,scale_var_params=scale_var_params)
    bc = np.array([Pc1, Qc0, Qc1, m0c, m1c, dphi0c])
    if ordering == 1:
        y = np.concatenate((ve,be,vh,bh,vc,bc))
    elif ordering == 2:
        y = np.concatenate((ve,vh,vc,be,bh,bc))
    elif ordering == 3:
        y = np.concatenate((ve,vh,bc,vc,be,bh))
    return y

def g_FP(elec_net,heat_net,coupling_net,y,max_iter_lf=10,tol_lf=1e-6,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},scale_var=None,scale_var_params=None,solve_LFe=True,solve_LFh=True,solve_LFc=True,ordering=1):
    """The system g(y) used in FP iteration based on DD"""
    ve_old, be_old, vh_old, bh_old, vc_old, bc_old = get_suby_FP(y,ordering=ordering)

    elec_net, ve_new, err_vece = h_elec(elec_net,ve_old,be_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=solve_LFe)
    be_new = hIF_elec(be_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)

    heat_net, vh_new, err_vech = h_heat(heat_net,vh_old,bh_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=solve_LFh)
    bh_new = hIF_heat(bh_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)

    coupling_net, vc_new, err_vecc = h_coupling(coupling_net,vc_old,bc_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=solve_LFc)
    bc_new = hIF_coupling(bc_old,ve_old,vh_old,scale_var=scale_var,scale_var_params=scale_var_params)

    if ordering == 1:
        g = np.concatenate((ve_new,be_new,vh_new,bh_new,vc_new,bc_new))
    elif ordering == 2:
        g = np.concatenate((ve_new,vh_new,vc_new,be_new,bh_new,bc_new))
    elif ordering == 3:
        g = np.concatenate((ve_new,vh_new,bc_new,vc_new,be_new,bh_new))
    return elec_net, heat_net, coupling_net, g, err_vece, err_vech, err_vecc

def jac_FP(elec_net,heat_net,coupling_net,y,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},scale_var=None,scale_var_params=None,ordering=1,method=1):
    """Jacobian of g(y) for the system used in FP iteration based on DD"""
    if formulation.get('heat') == 'standard':
        raise NotImplementedError('Jacobian for FP not implemented for standard formulation in heat')
    ve,be,vh,bh,vc,bc = get_suby_FP(y,ordering=ordering) # unscaled
    xe,xh,xc,xc_mes,x_mes = get_xLF_FP(elec_net,heat_net,coupling_net,vc,bc,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    P0_c = be[0]
    dphi1_c = bh[0]
    Pc1,Qc0,Qc1,m0c,m1c,dphi0c = bc
    # update all BCs
    elec_net = update_bc_elec(elec_net,P0_c,scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = reset_network_without_update_elec(elec_net)
    heat_net = update_bc_heat(heat_net,dphi1_c,scale_var=scale_var,scale_var_params=scale_var_params)
    heat_net = reset_network_without_update_heat(heat_net)
    coupling_net = update_bc_coupling(coupling_net,Pc1,Qc0,Qc1,m0c,m1c,dphi0c,scale_var=scale_var,scale_var_params=scale_var_params)
    # nonlinear systems
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysh = NonLinearSystemHeat(heat_net,formulation=formulation.get('heat'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    # submatrices in LF electrical part
    Ne = len(nlsyse.elecnetwork.nodes)
    Fe_ind = nlsyse.FP + [Ne+ind for ind in nlsyse.FQ]
    Ge_ind = [1,Ne,Ne+1] # indices of conservation of energy equations within full J
    xlfe_ind = nlsyse.xdelta + [Ne+ind for ind in nlsyse.xV]
    if scale_var == 'matrix':
        Jee_full = nlsyse.J(xe[:len(nlsyse.x_entries)],return_full=True)
        Dxe_inv = sps.diags(1/nlsyse.Dx().data[0])
        Jee = (nlsyse.DF().dot(Jee_full[Fe_ind,:][:,xlfe_ind]).dot(Dxe_inv)).todense() #Jee
    else:
        Jee_full = nlsyse.J_dense(xe[:len(nlsyse.x_entries)],return_full=True)
        Jee = Jee_full[Fe_ind,:][:,xlfe_ind] #Jee
    dFe_dbe = np.ones(1) # scaling has no influence
    dhe_dve = np.eye(len(Ge_ind))
    if scale_var == 'matrix':
        dhe_dxe = ((1/scale_var_params.get('Sbase')*sps.eye(len(Ge_ind))).dot(Jee_full[Ge_ind,:][:,xlfe_ind].dot(Dxe_inv))).todense() #dGe_dxlfe
    else:
        dhe_dxe = Jee_full[Ge_ind,:][:,xlfe_ind] #dGe_dxlfe
    dhe_dbe = np.zeros((len(Ge_ind),len(be)))
    dge_dve = np.zeros((len(ve),len(ve)))#np.eye(len(ve))
    dxe_dbe = np.linalg.solve(Jee,-dFe_dbe)
    dve_dbe = np.linalg.solve(dhe_dve,-dhe_dbe - dhe_dxe*dxe_dbe[0])
    dge_dbe = dve_dbe
    # submatrices in heat part
    Nh = len(nlsysh.heatnetwork.nodes)
    Eh = len(nlsysh.heatnetwork.links)
    Th = len(nlsysh.heatnetwork.half_links)
    if method == 1: # 'old' method, see notes 18 around 19-02-2021
        if scale_var == 'matrix':
            Dxh_inv = sps.diags(1/nlsysh.Dx().data[0])
            Jhh = (nlsysh.DF().dot(nlsysh.J(xh)).dot(Dxh_inv)).todense() #Jhh
        else:
            Jhh = nlsysh.J_dense(xh)
        dFh_dbh = np.zeros(len(xh))
        dFh_dbh[-1] = -1 # scaling has no influence
        dhh_dvh = np.zeros((len(vh),len(vh)))
        dhh_dvh[0,0] = -1 # scaling has no influence
        dhh_dvh[-1,-1] = -1 # scaling has no influence
        Ts1c = heat_net.nodes[-1].half_links[0].get_Ts(scale_var=scale_var,scale_var_params=scale_var_params)
        Ts0 = heat_net.nodes[0].get_Ts(scale_var=scale_var,scale_var_params=scale_var_params)
        Tr0 = heat_net.nodes[0].get_Tr(scale_var=scale_var,scale_var_params=scale_var_params)
        Tr3 = heat_net.nodes[-1].get_Tr(scale_var=scale_var,scale_var_params=scale_var_params)
        m0c,m1c,_ = vh
        m01 = xh[0]
        m02 = xh[1]
        dphi0c_dm0c = water.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*(Ts0-Tr0)
        dphi1c_dm1c = water.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*(Ts1c-Tr3)
        dphi1c_dTr3 = -m1c*water.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
        dphi0c_dTr0 = -m0c*water.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
        if scale_var == 'matrix':
            dphi0c_dm0c = 1/scale_var_params.get('phibase')*dphi0c_dm0c*scale_var_params.get('mbase')
            dphi1c_dm1c = 1/scale_var_params.get('phibase')*dphi1c_dm1c*scale_var_params.get('mbase')
            dphi0c_dTr0 = 1/scale_var_params.get('phibase')*dphi0c_dTr0*scale_var_params.get('Tbase')
            dphi1c_dTr3 = 1/scale_var_params.get('phibase')*dphi1c_dTr3*scale_var_params.get('Tbase')
        dhh_dvh[2,0] = dphi0c_dm0c
        dhh_dvh[1,1] = dphi1c_dm1c
        dhh_dbh = np.zeros(len(vh))
        dhh_dbh[1] = -1 # scaling has no influence
        dhh_dxh = np.zeros((len(vh),len(xh)))
        dhh_dxh[0,0] = -1 # scaling has no influence
        dhh_dxh[0,1] = -1 # scaling has no influence
        dhh_dxh[1,5] = 0 #dphi1c_dm1c
        dhh_dxh[2,10] = dphi0c_dTr0
        dhh_dxh[1,-1] = dphi1c_dTr3
        dgh_dvh = np.zeros((len(vh),len(vh)))#np.eye(len(vh))
        dxh_dbh = np.linalg.solve(Jhh,-dFh_dbh)
        dvh_dbh = np.linalg.solve(dhh_dvh,-dhh_dbh - dhh_dxh.dot(dxh_dbh))
        dgh_dbh = dvh_dbh.ravel()
    elif method == 2: #'new' method, see notes 18 around 17-03-2021
        len_xGh = 2 # slack equations or derived variables
        if scale_var == 'matrix':
            Dxh_inv = sps.diags(1/nlsysh.Dx().data[0])
            Jhh = (nlsysh.DF().dot(nlsysh.J(xh)).dot(Dxh_inv)).todense() #Jhh
        else:
            Jhh = nlsysh.J_dense(xh)
        dFh_dbh = np.zeros(len(xh))
        dFh_dbh[-1] = -1 # scaling has no influence
        Ts1c = heat_net.nodes[-1].half_links[0].get_Ts(scale_var=scale_var,scale_var_params=scale_var_params)
        Ts0 = heat_net.nodes[0].get_Ts(scale_var=scale_var,scale_var_params=scale_var_params)
        Tr0 = heat_net.nodes[0].get_Tr(scale_var=scale_var,scale_var_params=scale_var_params)
        Tr3 = heat_net.nodes[-1].get_Tr(scale_var=scale_var,scale_var_params=scale_var_params)
        m0c,m1c,_ = vh
        m01 = xh[0]
        m02 = xh[1]
        dphi0c_dm0c = water.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*(Ts0-Tr0)
        dphi1c_dm1c = water.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*(Ts1c-Tr3)
        dphi1c_dTr3 = -m1c*water.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
        dphi0c_dTr0 = -m0c*water.get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
        if scale_var == 'matrix':
            dphi0c_dm0c = 1/scale_var_params.get('phibase')*dphi0c_dm0c*scale_var_params.get('mbase')
            dphi1c_dm1c = 1/scale_var_params.get('phibase')*dphi1c_dm1c*scale_var_params.get('mbase')
            dphi0c_dTr0 = 1/scale_var_params.get('phibase')*dphi0c_dTr0*scale_var_params.get('Tbase')
            dphi1c_dTr3 = 1/scale_var_params.get('phibase')*dphi1c_dTr3*scale_var_params.get('Tbase')
        dFh_dxGh = np.zeros((len(xh),len_xGh)) # zero by definition
        dGh_dxh = np.zeros((len_xGh,len(xh))) # derivative of slack equations to state variables
        dGh_dxh[0,0] = -1 # scaling has no influence
        dGh_dxh[0,1] = -1 # scaling has no influence
        dGh_dxh[-1,10] = dphi0c_dTr0
        dGh_dxGh = np.zeros((len_xGh,len_xGh)) # derivative of slack equations to derived variables
        dGh_dxGh[0,0] = -1 # scaling has no influence
        dGh_dxGh[-1,-1] = -1 # scaling has no influence
        dGh_dxGh[-1,0] = dphi0c_dm0c
        dGh_dbh = np.zeros(len_xGh)
        dH_dbh = np.concatenate((dFh_dbh,dGh_dbh))
        dH_dxh = np.zeros((len(xh)+len_xGh,len(xh)+len_xGh))
        dH_dxh[:len(xh),:][:,:len(xh)] = Jhh
        dH_dxh[:len(xh),:][:,len(xh):] = dFh_dxGh
        dH_dxh[len(xh):,:][:,:len(xh)] = dGh_dxh
        dH_dxh[len(xh):,:][:,len(xh):] = dGh_dxGh
        plt.figure('dH_dxh')
        plt.spy(dH_dxh)
        P = np.zeros((len(vh),len(xh)+len_xGh)) # permutation matrix
        P[0,-2] = 1 #m0c
        P[1,5] = 1 #m1c
        P[2,-1] = 1 #dphi0c
        dx_du = np.linalg.solve(dH_dxh,-dH_dbh)
        dvh_dbh = P.dot(dx_du)
        dgh_dbh = dvh_dbh.ravel()
        dgh_dvh = np.zeros((len(vh),len(vh)))
    # submatrices in LF coupling part
    with HiddenPrints():
        if scale_var == 'matrix':
            Dxc_inv = sps.diags(1/nlsysc.Dx().data[0])
            Jcc = nlsysc.DF().dot(nlsysc.J(xc)).dot(Dxc_inv).todense()
        else:
            Jcc = nlsysc.J_dense(xc)
    dhc_dbc = np.zeros((len(vc),len(bc)))
    dhc_dbc[2,0] = 1 # scaling has no influence
    dhc_dbc[1,-1] = -1 # scaling has no influence
    dgc_dvc = np.zeros((len(vc),len(vc)))#np.eye(len(vc))
    dvc_dbc = np.linalg.solve(Jcc,-dhc_dbc)
    dgc_dbc = dvc_dbc
    # submatrices IFCs
    dgifce_dvc = np.array([0, 0, -1, 0])
    dgifce_dbe = np.zeros((len(be),len(be)))#np.eye(len(be))
    dgifch_dvc = np.array([0,0,0,-1])
    dgifch_dbh = np.zeros((len(bh),len(bh)))#np.eye(len(be))
    dgifcc_dve = np.zeros((len(bc),len(ve)))
    dgifcc_dve[:len(ve),:][:,:len(ve)] = -np.eye(len(ve))
    dgifcc_dvh = np.zeros((len(bc),len(vh)))
    dgifcc_dvh[-len(vh):,:][:,-len(vh):] = np.eye(len(vh))
    dgifcc_dbc = np.zeros((len(bc),len(bc)))#np.eye(len(bc))
    # collect the submatrices
    dg_dy = np.zeros((len(y),len(y)))
    if ordering == 1:
        ve_ind = 0 # first index of this part of y
        be_ind = ve_ind + len(ve)
        vh_ind = be_ind + len(be)
        bh_ind = vh_ind + len(vh)
        vc_ind = bh_ind + len(bh)
        bc_ind = vc_ind + len(vc)
        # derivatives of electrical part
        dg_dy[:be_ind,:][:,:be_ind] = dge_dve
        dg_dy[:be_ind,:][:,be_ind:vh_ind] = dge_dbe
        dg_dy[be_ind:vh_ind,:][:,be_ind:vh_ind] = dgifce_dbe
        dg_dy[be_ind:vh_ind,:][:,vc_ind:bc_ind] = dgifce_dvc
        # derivatives of heat part
        dg_dy[vh_ind:bh_ind,:][:,vh_ind:bh_ind] = dgh_dvh
        dg_dy[vh_ind:bh_ind,:][:,bh_ind] = dgh_dbh
        dg_dy[bh_ind:vc_ind,:][:,bh_ind:vc_ind] = dgifch_dbh
        dg_dy[bh_ind:vc_ind,:][:,vc_ind:bc_ind] = dgifch_dvc
        # derivatives of coupling part
        dg_dy[vc_ind:bc_ind,:][:,vc_ind:bc_ind] = dgc_dvc
        dg_dy[vc_ind:bc_ind,:][:,bc_ind:] = dgc_dbc
        dg_dy[bc_ind:,:][:,bc_ind:] = dgifcc_dbc
        dg_dy[bc_ind:,:][:,:be_ind] = dgifcc_dve
        dg_dy[bc_ind:,:][:,vh_ind:bh_ind] = dgifcc_dvh
    elif ordering == 2:
        ve_ind = 0 # first index of this part of y
        vh_ind = ve_ind + len(ve)
        vc_ind = vh_ind + len(vh)
        be_ind = vc_ind + len(vc)
        bh_ind = be_ind + len(be)
        bc_ind = bh_ind + len(bh)
        # derivatives of electrical part
        dg_dy[:vh_ind,:][:,:vh_ind] = dge_dve
        dg_dy[:vh_ind,:][:,be_ind:bh_ind] = dge_dbe
        dg_dy[be_ind:bh_ind,:][:,be_ind:bh_ind] = dgifce_dbe
        dg_dy[be_ind:bh_ind,:][:,vc_ind:be_ind] = dgifce_dvc
        # derivatives of heat part
        dg_dy[vh_ind:vc_ind,:][:,vh_ind:vc_ind] = dgh_dvh
        dg_dy[vh_ind:vc_ind,:][:,bh_ind] = dgh_dbh
        dg_dy[bh_ind:bc_ind,:][:,bh_ind:bc_ind] = dgifch_dbh
        dg_dy[bh_ind:bc_ind,:][:,vc_ind:be_ind] = dgifch_dvc
        # derivatives of coupling part
        dg_dy[vc_ind:be_ind,:][:,vc_ind:be_ind] = dgc_dvc
        dg_dy[vc_ind:be_ind,:][:,bc_ind:] = dgc_dbc
        dg_dy[bc_ind:,:][:,bc_ind:] = dgifcc_dbc
        dg_dy[bc_ind:,:][:,:vh_ind] = dgifcc_dve
        dg_dy[bc_ind:,:][:,vh_ind:vc_ind] = dgifcc_dvh
    elif ordering == 3:
        ve_ind = 0 # first index of this part of y
        vh_ind = ve_ind + len(ve)
        bc_ind = vh_ind + len(vh)
        vc_ind = bc_ind + len(bc)
        be_ind = vc_ind + len(vc)
        bh_ind = be_ind + len(be)
        # derivatives of electrical part
        dg_dy[:vh_ind,:][:,:vh_ind] = dge_dve
        dg_dy[:vh_ind,:][:,be_ind:bh_ind] = dge_dbe
        dg_dy[be_ind:bh_ind,:][:,be_ind:bh_ind] = dgifce_dbe
        dg_dy[be_ind:bh_ind,:][:,vc_ind:be_ind] = dgifce_dvc
        # derivatives of heat part
        dg_dy[vh_ind:bc_ind,:][:,vh_ind:bc_ind] = dgh_dvh
        dg_dy[vh_ind:bc_ind,:][:,bh_ind] = dgh_dbh
        dg_dy[bh_ind:,:][:,bh_ind:] = dgifch_dbh
        dg_dy[bh_ind:,:][:,vc_ind:be_ind] = dgifch_dvc
        # derivatives of coupling part
        dg_dy[vc_ind:be_ind,:][:,vc_ind:be_ind] = dgc_dvc
        dg_dy[vc_ind:be_ind,:][:,bc_ind:vc_ind] = dgc_dbc
        dg_dy[bc_ind:vc_ind,:][:,bc_ind:vc_ind] = dgifcc_dbc
        dg_dy[bc_ind:vc_ind,:][:,:vh_ind] = dgifcc_dve
        dg_dy[bc_ind:vc_ind,:][:,vh_ind:bc_ind] = dgifcc_dvh
    return elec_net, heat_net, coupling_net, dg_dy

def jac_FP_FD(elec_net,heat_net,coupling_net,y,max_iter_lf=10,tol_lf=1e-6,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},scale_var=None,scale_var_params=None,solve_LFe=True,solve_LFh=True,solve_LFc=True,Dy=np.array([]),Dy_inv=np.array([]),ordering=1):
    """Jacobian of g(y) for the system used in FP iteration based on DD, using FD"""
    J_FD = np.zeros((len(y),len(y)),dtype='float64')
    ve,be,vh,bh,vc,bc = get_suby_FP(y,ordering=ordering) # unscaled
    xe,xh,xc,xc_mes,x_mes = get_xLF_FP(elec_net,heat_net,coupling_net,vc,bc,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    P0_c = be[0]
    dphi1_c = bh[0]
    Pc1,Qc0,Qc1,m0c,m1c,dphi0c = bc
    # update all BCs
    elec_net = update_bc_elec(elec_net,P0_c,scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = reset_network_without_update_elec(elec_net)
    heat_net = update_bc_heat(heat_net,dphi1_c,scale_var=scale_var,scale_var_params=scale_var_params)
    heat_net = reset_network_without_update_heat(heat_net)
    coupling_net = update_bc_coupling(coupling_net,Pc1,Qc0,Qc1,m0c,m1c,dphi0c,scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net, heat_net, coupling_net, g, err_vece, err_vech, err_vecc = g_FP(elec_net,heat_net,coupling_net,y,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LFe=solve_LFe,solve_LFh=solve_LFh,solve_LFc=solve_LFc,ordering=ordering)
    for i in range(len(y)):
        e = np.zeros(len(y))
        e[i] = 1.
        if scale_var == 'matrix':
            y_dy = y.copy()
            y_dy[i] = Dy.data[0][i]*(y[i]*Dy_inv.data[0][i] + e[i]*tol_lf*Dy_inv.data[0][i])
        else:
            y_dy = y + e*tol_lf
        elec_net = reset_network_without_update_elec(elec_net)
        elec_net.update(xe,formulation=formulation.get('elec'))
        heat_net = reset_network_without_update_heat(heat_net)
        heat_net.update(xh,formulation=formulation.get('heat'))
        coupling_net.update(xc,formulation=formulation)
        elec_net, heat_net, coupling_net, g_dy, _, _, _ = g_FP(elec_net,heat_net,coupling_net,y_dy,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LFe=solve_LFe,solve_LFh=solve_LFh,solve_LFc=solve_LFc,ordering=ordering)
        J_FD[:,i] = (g_dy-g)/tol_lf
    # solve again, as this also updates BCs etc.
    elec_net = reset_network_without_update_elec(elec_net)
    elec_net.update(xe,formulation=formulation.get('elec'))
    heat_net = reset_network_without_update_heat(heat_net)
    heat_net.update(xh,formulation=formulation.get('heat'))
    coupling_net.update(xc,formulation=formulation)
    elec_net, heat_net, coupling_net, g, err_vece, err_vech, err_vecc = g_FP(elec_net,heat_net,coupling_net,y,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LFe=solve_LFe,solve_LFh=solve_LFh,solve_LFc=solve_LFc,ordering=ordering)
    return elec_net, heat_net, coupling_net, J_FD, g, err_vece, err_vech, err_vecc

def run_LF_mes_DD_FP(max_iter_lf=10,max_iter_fp=200,tol_lf=1e-6,tol_fp=1e-6,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},scale_var=None,scale_var_params=None,error_measure='F',decoupled=False,ordering=1):
    """Steady-state load flow analysis of multi-carrier network, using fixed-point iteration based on domain decomposition. The default values are used for initialization.

    Uses the already implemented solvers for the single-carrier network.
    """
    if scale_var == 'per_unit':
        raise NotImplementedError("DD is not implemented for per unit scaling. Use 'None' or 'matrix' instead.")
    # create networks
    elec_net = create_electrical_network(scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = reset_network_without_update_elec(elec_net)
    heat_net = create_heat_network(scale_var=scale_var,scale_var_params=scale_var_params)
    heat_net = reset_network_without_update_heat(heat_net)
    with HiddenPrints():
        coupling_net = create_coupling_network(scale_var=scale_var,scale_var_params=scale_var_params)

    # initialize networks, and set initial guesses
    xe_new = set_xe_LF_init(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    xh_new = set_xh_LF_init(heat_net,formulation=formulation.get('heat'),scale_var=scale_var,scale_var_params=scale_var_params)
    with HiddenPrints():
        xc_new = set_xc_LF_init(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    y_new = get_y_FP(elec_net,heat_net,coupling_net,formulation=formulation,ordering=ordering) # unscaled
    ve_new,be_new,vh_new,bh_new,vc_new,bc_new = get_suby_FP(y_new,ordering=ordering) # unscaled

    # adjustments to initial guess in case of decoupling (as is done for GE2N)
    if decoupled:
        be_new = hIF_elec(be_new,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
        bh_new = hIF_heat(bh_new,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
        bc_new = hIF_coupling(bc_new,ve_new,vh_new,scale_var=scale_var,scale_var_params=scale_var_params)
        if ordering == 1:
            y_new = np.concatenate((ve_new,be_new,vh_new,bh_new,vc_new,bc_new))
        elif ordering == 2:
            y_new = np.concatenate((ve_new,vh_new,vc_new,be_new,bh_new,bc_new))
        elif ordering == 3:
            y_new = np.concatenate((ve_new,vh_new,bc_new,vc_new,be_new,bh_new))

    if scale_var == 'matrix':
        veb = scale_var_params.get('Sbase')*np.ones(3)
        beb = scale_var_params.get('Sbase')*np.ones(1)
        vhb = np.array([scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('phibase')])
        bhb = scale_var_params.get('phibase')*np.ones(1)
        vcb = np.array([scale_var_params.get('qbase'),scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('phibase')])
        bcb = np.array([scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('phibase')])
        if ordering == 1:
            Dy = sps.diags(1/np.concatenate((veb,beb,vhb,bhb,vcb,bcb)))
        elif ordering == 2:
            Dy = sps.diags(1/np.concatenate((veb,vhb,vcb,beb,bhb,bcb)))
        elif ordering == 3:
            Dy = sps.diags(1/np.concatenate((veb,vhb,bcb,vcb,beb,bhb)))
    else:
        Dy = sps.eye(len(y_new))

    # iterate
    print('\nRunning load flow with FP based on DD, electricity-heat, error {}, ordering {}, decoupled: {}'.format(error_measure,ordering,decoupled))
    err_vec = list()
    if error_measure == 'dy':
        error = 1 # set some initial value for the error
    elif error_measure == 'F':
        xe,xh,xc,xc_mes,x_mes = get_xLF_FP(elec_net,heat_net,coupling_net,vc_new,bc_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        P0_c = be_new[0]
        dphi1_c = bh_new[0]
        Pc1,Qc0,Qc1,m0c,m1c,dphi0c = bc_new
        with HiddenPrints():
            error, Fe, Fh, Fc = error_measure_F(elec_net,heat_net,coupling_net,P0_c,dphi1_c,Pc1,Qc0,Qc1,m0c,m1c,dphi0c,xe,xh,xc,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        err_vec.append(error)
    iter_nr = 0
    errors_elec = dict()
    err_vece = np.array([1])
    errors_heat= dict()
    err_vech = np.array([1])
    errors_coupling = dict()
    err_vecc = np.array([1])
    errors_elec_DD = dict()
    errors_heat_DD = dict()
    errors_coupling_DD = dict()
    if error_measure == 'F':
        errors_elec_DD[iter_nr] = np.linalg.norm(Fe)
        errors_heat_DD[iter_nr] = np.linalg.norm(Fh)
        errors_coupling_DD[iter_nr] = np.linalg.norm(Fc)

    while error > tol_fp and iter_nr < max_iter_fp:
        y_old = y_new.copy() # unscaled
        ve_old, be_old, vh_old, bh_old, vc_old, bc_old = get_suby_FP(y_old,ordering=ordering) # unscaled

        if ordering == 1:
            # g1
            if len(errors_elec)==0 or (np.linalg.norm(dbe) > tol_lf) or (err_vece[-1] > tol_lf):
                elec_net, ve_new, err_vece = h_elec(elec_net,ve_old,be_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_elec[iter_nr] = err_vece
            else:
                ve_new = ve_old.copy()
            be_new = hIF_elec(be_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbe = be_new-be_old

            if len(errors_heat)==0 or (np.linalg.norm(dbh) > tol_lf) or (err_vech[-1] > tol_lf):
                heat_net, vh_new, err_vech = h_heat(heat_net,vh_old,bh_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_heat[iter_nr] = err_vech
            else:
                vh_new = vh_old.copy()
            bh_new = hIF_heat(bh_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbh = bh_new-bh_old

            if len(errors_coupling)==0 or (np.linalg.norm(dbc) > tol_lf) or (err_vecc[-1] > tol_lf):
                coupling_net, vc_new, err_vecc = h_coupling(coupling_net,vc_old,bc_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_coupling[iter_nr] = err_vecc
            else:
                vc_new = vc_old.copy()
            # g2
            if decoupled:
                bc_new = hIF_coupling(bc_old,ve_new,vh_new,scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                bc_new = hIF_coupling(bc_old,ve_old,vh_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbc = bc_new-bc_old

            y_new = np.concatenate((ve_new,be_new,vh_new,bh_new,vc_new,bc_new)) # always unscaled
        elif ordering == 2:
            # g1
            if len(errors_elec)==0 or (np.linalg.norm(dbe) > tol_lf) or (err_vece[-1] > tol_lf):
                elec_net, ve_new, err_vece = h_elec(elec_net,ve_old,be_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_elec[iter_nr] = err_vece
            else:
                ve_new = ve_old.copy()
            if len(errors_heat)==0 or (np.linalg.norm(dbh) > tol_lf) or (err_vech[-1] > tol_lf):
                heat_net, vh_new, err_vech = h_heat(heat_net,vh_old,bh_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_heat[iter_nr] = err_vech
            else:
                vh_new = vh_old.copy()
            if len(errors_coupling)==0 or (np.linalg.norm(dbc) > tol_lf) or (err_vecc[-1] > tol_lf):
                coupling_net, vc_new, err_vecc = h_coupling(coupling_net,vc_old,bc_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_coupling[iter_nr] = err_vecc
            else:
                vc_new = vc_old.copy()
            # g2
            if decoupled:
                be_new = hIF_elec(be_old,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
                bh_new = hIF_heat(bh_old,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
                bc_new = hIF_coupling(bc_old,ve_new,vh_new,scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                be_new = hIF_elec(be_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
                bh_new = hIF_heat(bh_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
                bc_new = hIF_coupling(bc_old,ve_old,vh_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbe = be_new-be_old
            dbh = bh_new-bh_old
            dbc = bc_new-bc_old
            y_new = np.concatenate((ve_new,vh_new,vc_new,be_new,bh_new,bc_new))
        elif ordering == 3:
            # g1
            if len(errors_elec)==0 or (np.linalg.norm(dbe) > tol_lf) or (err_vece[-1] > tol_lf):
                elec_net, ve_new, err_vece = h_elec(elec_net,ve_old,be_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_elec[iter_nr] = err_vece
            else:
                ve_new = ve_old.copy()
            if len(errors_heat)==0 or (np.linalg.norm(dbh) > tol_lf) or (err_vech[-1] > tol_lf):
                heat_net, vh_new, err_vech = h_heat(heat_net,vh_old,bh_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_heat[iter_nr] = err_vech
            else:
                vh_new = vh_old.copy()
            # g2
            if decoupled:
                bc_new = hIF_coupling(bc_old,ve_new,vh_new,scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                bc_new = hIF_coupling(bc_old,ve_old,vh_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbc = bc_new-bc_old
            # g3
            if decoupled:
                if len(errors_coupling)==0 or (np.linalg.norm(dbc) > tol_lf) or (err_vecc[-1] > tol_lf):
                    coupling_net, vc_new, err_vecc = h_coupling(coupling_net,vc_old,bc_new,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                    errors_coupling[iter_nr] = err_vecc
                else:
                    vc_new = vc_old.copy()
            else:
                if len(errors_coupling)==0 or (np.linalg.norm(dbc) > tol_lf) or (err_vecc[-1] > tol_lf):
                    coupling_net, vc_new, err_vecc = h_coupling(coupling_net,vc_old,bc_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                    errors_coupling[iter_nr] = err_vecc
                else:
                    vc_new = vc_old.copy()
            # g4
            if decoupled:
                be_new = hIF_elec(be_old,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
                bh_new = hIF_heat(bh_old,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                be_new = hIF_elec(be_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
                bh_new = hIF_heat(bh_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbe = be_new-be_old
            dbh = bh_new-bh_old
            y_new = np.concatenate((ve_new,vh_new,bc_new,vc_new,be_new,bh_new))

        # Determine error
        if error_measure == 'F': #Also updates the BCs of all networks
            xe_new,xh_new,xc_new,xc_mes_new,x_mes_new = get_xLF_FP(elec_net,heat_net,coupling_net,vc_new,bc_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            P0_c = be_new[0]
            dphi1_c = bh_new[0]
            Pc1,Qc0,Qc1,m0c,m1c,dphi0c = bc_new
            with HiddenPrints():
                error, Fe, Fh, Fc = error_measure_F(elec_net,heat_net,coupling_net,P0_c,dphi1_c,Pc1,Qc0,Qc1,m0c,m1c,dphi0c,xe_new,xh_new,xc_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            errors_elec_DD[iter_nr+1] = np.linalg.norm(Fe)
            errors_heat_DD[iter_nr+1] = np.linalg.norm(Fh)
            errors_coupling_DD[iter_nr+1] = np.linalg.norm(Fc)
        elif error_measure == 'dy': # y is unscaled, so determine scaled error
            error = np.linalg.norm(Dy.dot(y_new)-Dy.dot(y_old))
        err_vec.append(error)
        iter_nr += 1

    # print results
    xe_new,xh_new,xc_new,xc_mes_new,x_mes_new = get_xLF_FP(elec_net,heat_net,coupling_net,vc_new,bc_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params) # unscaled
    with HiddenPrints():
        het_net, elec_net, heat_net = create_mes_network()
        het_net.initialize()
    with HiddenPrints():
        p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(x_mes_new,formulation=formulation)
    print('Solution after {} iterations (final error = {:.4e}):'.format(iter_nr,err_vec[-1]))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format([hl.get_m() for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('Ts hl = {} C'.format([hl.get_Ts() for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('Tr hl = {} C'.format([hl.get_Tr() for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('qc = {} kg/s'.format(qc_vec))
    print('Pc = {} MW'.format([Pc/MW for Pc in Pc_vec]))
    print('phi c = {} MW'.format([phic/MW for phic in phic_vec]))
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('||F mes|| = {:.4e}'.format(np.linalg.norm(nlsys.DF().dot(nlsys.F(x_mes_new)))))
    return het_net, elec_net, heat_net, xe_new, xh_new, xc_new, x_mes_new, iter_nr, err_vec, errors_elec, errors_heat, errors_coupling, errors_elec_DD, errors_heat_DD, errors_coupling_DD

def run_LF_mes_DD_FP_NR(max_iter_lf=10,max_iter_fp=200,tol_lf=1e-6,tol_fp=1e-6,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},scale_var=None,scale_var_params=None,plot_jac=False,der=True,ordering=1):
    """Steady-state load flow analysis of multi-carrier network, using NR iteration based on the fixed-point system, which is based on domain decomposition. The default values are used for initialization.
    """
    if scale_var == 'per_unit':
        raise NotImplementedError("DD is not implemented for per unit scaling. Use 'None' or 'matrix' instead.")
    print('\nRunning load flow with NR on FP based on DD, electricity-heat, ordering: {}, an. der: {}'.format(ordering,der))
    # create networks
    elec_net = create_electrical_network(scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = reset_network_without_update_elec(elec_net)
    heat_net = create_heat_network(scale_var=scale_var,scale_var_params=scale_var_params)
    heat_net = reset_network_without_update_heat(heat_net)
    with HiddenPrints():
        coupling_net = create_coupling_network(scale_var=scale_var,scale_var_params=scale_var_params)

    # initialize networks, and set initial guesses
    xe_new = set_xe_LF_init(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    xh_new = set_xh_LF_init(heat_net,formulation=formulation.get('heat'),scale_var=scale_var,scale_var_params=scale_var_params)
    with HiddenPrints():
        xc_new = set_xc_LF_init(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    y_new = get_y_FP(elec_net,heat_net,coupling_net,formulation=formulation,ordering=ordering) # unscaled
    ve_new,be_new,vh_new,bh_new,vc_new,bc_new = get_suby_FP(y_new,ordering=ordering) # unscaled

    # adjustments to initial guess in case of decoupling (as is done for GE2N)
    be_new = hIF_elec(be_new,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
    bh_new = hIF_heat(bh_new,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
    bc_new = hIF_coupling(bc_new,ve_new,vh_new,scale_var=scale_var,scale_var_params=scale_var_params)
    if ordering == 1:
        y_new = np.concatenate((ve_new,be_new,vh_new,bh_new,vc_new,bc_new))
    elif ordering == 2:
        y_new = np.concatenate((ve_new,vh_new,vc_new,be_new,bh_new,bc_new))
    elif ordering == 3:
        y_new = np.concatenate((ve_new,vh_new,bc_new,vc_new,be_new,bh_new))

    len_y = len(y_new)
    if scale_var == 'matrix':
        veb = scale_var_params.get('Sbase')*np.ones(3)
        beb = scale_var_params.get('Sbase')*np.ones(1)
        vhb = np.array([scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('phibase')])
        bhb = scale_var_params.get('phibase')*np.ones(1)
        vcb = np.array([scale_var_params.get('qbase'),scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('phibase')])
        bcb = np.array([scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('phibase')])
        if ordering == 1:
            Dy = sps.diags(1/np.concatenate((veb,beb,vhb,bhb,vcb,bcb)))
        elif ordering == 2:
            Dy = sps.diags(1/np.concatenate((veb,vhb,vcb,beb,bhb,bcb)))
        elif ordering == 3:
            Dy = sps.diags(1/np.concatenate((veb,vhb,bcb,vcb,beb,bhb)))
        Dy_inv = sps.diags(1/Dy.data[0])
    else:
        Dy = np.eye(len_y)
        Dy_inv = np.eye(len_y)

    # iterate
    if der: # use analytical derivatives
        elec_net, heat_net, coupling_net, g_new, err_vece, err_vech, err_vecc = g_FP(elec_net,heat_net,coupling_net,y_new,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,ordering=ordering)
        elec_net, heat_net, coupling_net, Jg_new = jac_FP(elec_net,heat_net,coupling_net,y_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,ordering=ordering)
    else:
        elec_net, heat_net, coupling_net, Jg_new, g_new, err_vece, err_vech, err_vecc = jac_FP_FD(elec_net,heat_net,coupling_net,y_new,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy,Dy_inv=Dy_inv,ordering=ordering)

    F_new = y_new - g_new
    if scale_var == 'matrix': # Shouldn't matter, since F and y scale with the same base values...
        y_new = Dy.dot(y_new)
        F_new = Dy.dot(F_new)
        if not der:
            Jg_new = Dy.todense().dot(Jg_new.dot(Dy_inv.todense()))
    J_new = np.eye(len_y,dtype='float64') - Jg_new
    if plot_jac:
        fig_J = plt.figure('Jacobian spy plot (I-dg/dy), scaling = {}, ordering {}'.format(scale_var, ordering))
        ax_J = fig_J.gca()
        if der:
            ax_J.spy(J_new,markerfacecolor='tab:blue',markeredgecolor='tab:blue',marker='s',alpha=1)
        else:
            ax_J.spy(J_new,markerfacecolor='k',markeredgecolor='k',marker='.',alpha=1)
        J_map = np.matrix(np.nan*np.ones(J_new.shape))
        for i in range(len_y):
            for j in range(len_y):
                if J_new[i,j] != 0:
                    J_map[i,j] = J_new[i,j]
        fig_J_map = plt.figure('Colormap Jacobian (I-dg/dy), scaling = {}, an. der: {}, ordering {}'.format(scale_var, der, ordering))
        ax_J_map = fig_J_map.gca()
        plt.imshow(J_map)
        plt.colorbar()
        # plot overlay / lines in spyplot and colormap Jacobian
        for ax in [ax_J,ax_J_map]:
            overlay_FP_jac(ax,y_new,ordering=ordering)

    error = np.linalg.norm(F_new)
    err_vec = [error]
    iter_nr = 0
    errors_elec = dict()
    errors_elec[iter_nr] = err_vece
    errors_heat= dict()
    errors_heat[iter_nr] = err_vech
    errors_coupling = dict()
    errors_coupling[iter_nr] = err_vecc
    errors_elec_DD = dict()
    errors_elec_DD[iter_nr] = err_vece[-1]
    errors_heat_DD = dict()
    errors_heat_DD[iter_nr] = err_vech[-1]
    errors_coupling_DD = dict()
    errors_coupling_DD[iter_nr] = err_vecc[-1]
    while error > tol_fp and iter_nr < max_iter_fp:
        y_old = np.squeeze(y_new)
        F_old = np.squeeze(F_new)
        J_old = J_new

        if sps.issparse(J_old):
            dy = sps.linalg.spsolve(J_old,-F_old)
        else:
            dy = np.linalg.solve(J_old,-F_old)

        y_new = y_old + dy # scaled

        if scale_var == 'matrix':
            dve, dbe, dvh, dbh, dvc, dbc = get_suby_FP(Dy_inv.dot(dy),ordering=ordering)
        else:
            dve, dbe, dvh, dbh, dvc, dbc = get_suby_FP(dy,ordering=ordering)

        if len(errors_elec)==0 or (np.linalg.norm(dbe) > tol_lf) or (err_vece[-1] > tol_lf):
            solve_LFe = True
        else:
            solve_LFe = False
        if len(errors_heat)==0 or (np.linalg.norm(dbh) > tol_lf) or (err_vech[-1] > tol_lf):
            solve_LFh = True
        else:
            solve_LFh = False
        if len(errors_coupling)==0 or (np.linalg.norm(dbc) > tol_lf) or (err_vecc[-1] > tol_lf):
            solve_LFc = True
        else:
            solve_LFc = False
        if der: # use analytical derivatives
            elec_net, heat_net, coupling_net, g_new, err_vece, err_vech, err_vecc = g_FP(elec_net,heat_net,coupling_net,Dy_inv.dot(y_new),max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LFe=solve_LFe,solve_LFh=solve_LFh,solve_LFc=solve_LFc,ordering=ordering)
            elec_net, heat_net, coupling_net, Jg_new = jac_FP(elec_net,heat_net,coupling_net,Dy_inv.dot(y_new),formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,ordering=ordering)
        else:
            elec_net, heat_net, coupling_net, Jg_new, g_new, err_vece, err_vech, err_vecc = jac_FP_FD(elec_net,heat_net,coupling_net,Dy_inv.dot(y_new),max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy,Dy_inv=Dy_inv,ordering=ordering)
            if scale_var == 'matrix':
                Jg_new = Dy.todense().dot(Jg_new.dot(Dy_inv.todense()))
        J_new = np.eye(len_y,dtype='float64') - Jg_new
        if scale_var == 'matrix':
            F_new = Dy_inv.dot(y_new) - g_new
            # scale F
            F_new = Dy.dot(F_new)
        else:
            F_new = y_new - g_new

        error = np.linalg.norm(F_new) # F is scaled when matrix scaling is used
        iter_nr += 1
        if solve_LFe:
            errors_elec[iter_nr] = err_vece
            errors_elec_DD[iter_nr] = err_vece[-1]
        if solve_LFh:
            errors_heat[iter_nr] = err_vech
            errors_heat_DD[iter_nr] = err_vech[-1]
        if solve_LFc:
            errors_coupling[iter_nr] = err_vecc
            errors_coupling_DD[iter_nr] = err_vecc[-1]
        err_vec.append(error)

    if scale_var == 'matrix':
        y_new = Dy_inv.dot(y_new)
    # print results
    ve_new,be_new,vh_new,bh_new,vc_new,bc_new = get_suby_FP(y_new,ordering=ordering) # unscaled
    xe_new,xh_new,xc_new,xc_mes_new,x_mes_new = get_xLF_FP(elec_net,heat_net,coupling_net,vc_new,bc_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params) # unscaled
    with HiddenPrints():
        het_net, elec_net, heat_net = create_mes_network()
        het_net.initialize()
    with HiddenPrints():
        p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(x_mes_new,formulation=formulation)
    print('Solution after {} iterations (final error = {:.4e}):'.format(iter_nr,err_vec[-1]))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format([hl.get_m() for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('Ts hl = {} C'.format([hl.get_Ts() for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('Tr hl = {} C'.format([hl.get_Tr() for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in heat_net.get_nodes()  for hl in node.get_half_links()]))
    print('qc = {} kg/s'.format(qc_vec))
    print('Pc = {} MW'.format([Pc/MW for Pc in Pc_vec]))
    print('phi c = {} MW'.format([phic/MW for phic in phic_vec]))
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('||F mes|| = {:.4e}'.format(np.linalg.norm(nlsys.DF().dot(nlsys.F(x_mes_new)))))
    return het_net, elec_net, heat_net, xe_new, xh_new, xc_new, x_mes_new, iter_nr, err_vec, errors_elec, errors_heat, errors_coupling, errors_elec_DD, errors_heat_DD, errors_coupling_DD

def plot_convergence_detailed(ax, iter_vec, err_vec, errors_elec, errors_heat, errors_coupling, errors_elec_DD, errors_heat_DD, errors_coupling_DD,tol_lf,tol_dd,art_zero,max_data_points,alpha=0.7):
    """Plot the errors and suberrors per outer iteration of DD. Assumes the results of one DD run."""
    # plot final error of LF subsolve per outer DD iter, for each SC network
    iterse = list()
    err_vec_elec = list()
    for itere, err_vece in errors_elec.items():
        iterse.append(itere)
        err_vec_elec.append(max(art_zero,err_vece[-1]))
    plot_convergence(ax,iterse,err_vec_elec,max_data_points,'elec subsystem',ls='--',marker='.',color=colors_carrier.get('elec'),alpha=alpha)
    itersh = list()
    err_vec_heat = list()
    for iterh, err_vech in errors_heat.items():
        itersh.append(iterh)
        err_vec_heat.append(max(art_zero,err_vech[-1]))
    plot_convergence(ax,itersh,err_vec_heat,max_data_points,'heat subsystem ',ls='--',marker='.',color=colors_carrier.get('heat'),alpha=alpha)
    itersc = list()
    err_vec_coupling = list()
    for iterc, err_vecc in errors_coupling.items():
        itersc.append(iterc)
        err_vec_coupling.append(max(art_zero,err_vecc[-1]))
    plot_convergence(ax,itersc,err_vec_coupling,max_data_points,'coupling subsystem',ls='--',marker='.',color=colors_carrier.get('coupling'),alpha=alpha)
    # plot the error ||F|| of the subnetwork per outer DD iter, for each SC network. Only plottes if ||F|| is used as error measure
    if len(errors_elec_DD):
        iterse_DD = list()
        err_vec_elec_DD = list()
        for itere, err_e in errors_elec_DD.items():
            iterse_DD.append(itere)
            err_vec_elec_DD.append(max(art_zero,err_e))
        plot_convergence(ax,iterse_DD,err_vec_elec_DD,max_data_points,r'$||F^e||_2$',ls='-',marker='.',color=colors_carrier.get('elec'),alpha=alpha)
    if len(errors_heat_DD):
        itersh_DD = list()
        err_vec_heat_DD = list()
        for iterh, err_h in errors_heat_DD.items():
            itersh_DD.append(iterh)
            err_vec_heat_DD.append(max(art_zero,err_h))
        plot_convergence(ax,itersh_DD,err_vec_heat_DD,max_data_points,r'$||F^h||_2$',ls='-',marker='.',color=colors_carrier.get('heat'),alpha=alpha)
    if len(errors_coupling_DD):
        itersc_DD = list()
        err_vec_coupling_DD = list()
        for iterc, err_c in errors_coupling_DD.items():
            itersc_DD.append(iterc)
            err_vec_coupling_DD.append(max(art_zero,err_c))
        plot_convergence(ax,itersc_DD,err_vec_coupling_DD,max_data_points,r'$||F^c||_2$',ls='-',marker='.',color=colors_carrier.get('coupling'),alpha=alpha)
    # plot error of outer DD iters
    plot_convergence(ax,iter_vec,err_vec,max_data_points,'DD',ls='-',marker='.',color=colors_solver.get('dd'))

def overlay_FP_jac(ax,y,ordering=1):
    """Plot overlay / lines in spyplot or colormap plot of the Jacobian for the FP system, which is based on DD"""
    ve,be,vh,bh,vc,bc = get_suby_FP(y,ordering=ordering)
    ye_len = 4
    yh_len = 4
    yc_len = 10
    y_len = ye_len+yh_len+yc_len
    len_v = len(ve)+len(vh)+len(vc)
    if ordering == 1:
        # horizontal lines
        ax.plot((-0.5,y_len-0.5),(len(ve)-0.5,len(ve)-0.5),'k--',alpha=.5)
        ax.text(len(ve)/2-0.5,-1,r'$v^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((-0.5,y_len-0.5),(ye_len-0.5,ye_len-0.5),'k-')
        ax.text(len(ve)+len(be)/2-0.5,-1,r'$b^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((-0.5,y_len-0.5),(ye_len+len(vh)-0.5,ye_len+len(vh)-0.5),'k--',alpha=.5)
        ax.text(ye_len+len(vh)/2-0.5,-1,r'$v^h$', horizontalalignment='center',verticalalignment='center',color='b')
        ax.plot((-0.5,y_len-0.5),(ye_len+yh_len-0.5,ye_len+yh_len-0.5),'k-')
        ax.text(ye_len+len(vh)+len(bh)/2-0.5,-1,r'$b^h$', horizontalalignment='center',verticalalignment='center',color='b')
        ax.plot((-0.5,y_len-0.5),(ye_len+yh_len+len(vc)-0.5,ye_len+yh_len+len(vc)-0.5),'k--',alpha=.5)
        ax.text(ye_len+yh_len+len(vc)/2-0.5,-1,r'$v^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.text(ye_len+yh_len+len(vc)+len(bc)/2-0.5,-1,r'$b^c$', horizontalalignment='center',verticalalignment='center',color='k')
        # vertical lines
        ax.plot((len(ve)-0.5,len(ve)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,len(ve)/2-0.5,r'$g^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((ye_len-0.5,ye_len-0.5),(0-0.5,y_len-0.5),'k-')
        ax.text(-1.5,len(ve)+len(be)/2-0.5,r'$g^{I,e}$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((ye_len+len(vh)-0.5,ye_len+len(vh)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,ye_len+len(vh)/2-0.5,r'$g^h$', horizontalalignment='center',verticalalignment='center',color='b')
        ax.plot((ye_len+yh_len-0.5,ye_len+yh_len-0.5),(0-0.5,y_len-0.5),'k-')
        ax.text(-1.5,ye_len+len(vh)+len(bh)/2-0.5,r'$g^{I,h}$', horizontalalignment='center',verticalalignment='center',color='b')
        ax.plot((ye_len+yh_len+len(vc)-0.5,ye_len+yh_len+len(vc)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,ye_len+yh_len+len(vc)/2-0.5,r'$g^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.text(-1.5,ye_len+yh_len+len(vc)+len(bc)/2-0.5,r'$g^{I,c}$', horizontalalignment='center',verticalalignment='center',color='k')
    elif ordering == 2:
        # horizontal lines
        ax.plot((-0.5,y_len-0.5),(len(ve)-0.5,len(ve)-0.5),'k--',alpha=.5)
        ax.text(len(ve)/2-0.5,-1,r'$v^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((-0.5,y_len-0.5),(len(ve)+len(vh)-0.5,len(ve)+len(vh)-0.5),'k--',alpha=.5)
        ax.text(len(ve)+len(vh)/2-0.5,-1,r'$v^h$', horizontalalignment='center',verticalalignment='center',color='b')
        ax.plot((-0.5,y_len-0.5),(len_v-0.5,len_v-0.5),'k-')
        ax.text(len(ve)+len(vh)+len(vc)/2-0.5,-1,r'$v^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.plot((-0.5,y_len-0.5),(len_v+len(be)-0.5,len_v+len(be)-0.5),'k--',alpha=.5)
        ax.text(len_v+len(be)/2-0.5,-1,r'$b^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((-0.5,y_len-0.5),(len_v+len(be)+len(bh)-0.5,len_v+len(be)+len(bh)-0.5),'k--',alpha=.5)
        ax.text(len_v+len(be)+len(bh)/2-0.5,-1,r'$b^h$', horizontalalignment='center',verticalalignment='center',color='b')
        ax.text(len_v+len(be)+len(bh)+len(bc)/2-0.5,-1,r'$b^c$', horizontalalignment='center',verticalalignment='center',color='k')
        # vertical lines
        ax.plot((len(ve)-0.5,len(ve)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,len(ve)/2-0.5,r'$g^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((len(ve)+len(vh)-0.5,len(ve)+len(vh)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,len(ve)+len(vh)/2-0.5,r'$g^h$', horizontalalignment='center',verticalalignment='center',color='b')
        ax.plot((len_v-0.5,len_v-0.5),(0-0.5,y_len-0.5),'k-')
        ax.text(-1.5,len(ve)+len(vh)+len(vc)/2-0.5,r'$g^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.plot((len_v+len(be)-0.5,len_v+len(be)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,len_v+len(be)/2-0.5,r'$g^{I,e}$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((len_v+len(be)+len(bh)-0.5,len_v+len(be)+len(bh)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,len_v+len(be)+len(bh)/2-0.5,r'$g^{I,h}$', horizontalalignment='center',verticalalignment='center',color='b')
        ax.text(-1.5,len_v+len(be)+len(bh)+len(bc)/2-0.5,r'$g^{I,c}$', horizontalalignment='center',verticalalignment='center',color='k')
    elif ordering == 3:
        # horizontal lines
        ax.plot((-0.5,y_len-0.5),(len(ve)-0.5,len(ve)-0.5),'k--',alpha=.5)
        ax.text(len(ve)/2-0.5,-1,r'$v^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((-0.5,y_len-0.5),(len(ve)+len(vh)-0.5,len(ve)+len(vh)-0.5),'k-')
        ax.text(len(ve)+len(vh)/2-0.5,-1,r'$v^h$', horizontalalignment='center',verticalalignment='center',color='b')
        ax.plot((-0.5,y_len-0.5),(len(ve)+len(vh)+len(bc)-0.5,len(ve)+len(vh)+len(bc)-0.5),'k-')
        ax.text(len(ve)+len(vh)+len(bc)/2-0.5,-1,r'$b^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.plot((-0.5,y_len-0.5),(len_v+len(bc)-0.5,len_v+len(bc)-0.5),'k-')
        ax.text(len(ve)+len(vh)+len(bc)+len(vc)/2-0.5,-1,r'$v^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.plot((-0.5,y_len-0.5),(len_v+len(bc)+len(be)-0.5,len_v+len(bc)+len(be)-0.5),'k--',alpha=.5)
        ax.text(len_v+len(bc)+len(be)/2-0.5,-1,r'$b^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.text(len_v+len(bc)+len(be)+len(bh)/2-0.5,-1,r'$b^h$', horizontalalignment='center',verticalalignment='center',color='b')
        # vertical lines
        ax.plot((len(ve)-0.5,len(ve)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,len(ve)/2-0.5,r'$v^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((len(ve)+len(vh)-0.5,len(ve)+len(vh)-0.5),(0-0.5,y_len-0.5),'k-')
        ax.text(-1.5,len(ve)+len(vh)/2-0.5,r'$g^h$', horizontalalignment='center',verticalalignment='center',color='b')
        ax.plot((len(ve)+len(vh)+len(bc)-0.5,len(ve)+len(vh)+len(bc)-0.5),(0-0.5,y_len-0.5),'k-')
        ax.text(-1.5,len(ve)+len(vh)+len(bc)/2-0.5,r'$g^{I,c}$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.plot((len_v+len(bc)-0.5,len_v+len(bc)-0.5),(0-0.5,y_len-0.5),'k-')
        ax.text(-1.,len(ve)+len(vh)+len(bc)+len(vc)/2-0.5,r'$g^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.plot((len_v+len(bc)+len(be)-0.5,len_v+len(bc)+len(be)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,len_v+len(bc)+len(be)/2-0.5,r'$g^{I,e}$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.text(-1.5,len_v+len(bc)+len(be)+len(bh)/2-0.5,r'$g^{I,h}$', horizontalalignment='center',verticalalignment='center',color='b')

def compare_DD(tol_lf = 1e-6,tol_dd = 1e-6,art_zero = 1e-16,max_iter_dd = 200,max_iter_lf = 10,max_data_points = 200,scale_var=None,scale_var_params=None,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'},save_data=False,path_to_data=None):
    """Compare the results and the convergende of DD with integrated LF."""
    if scale_var == None:
        scale_var_label = 'unscaled'
    else:
        scale_var_label = scale_var

    fig = plt.figure('conv_'+scale_var_label)
    ax = fig.gca()
    fig = plt.figure('conv_order_'+scale_var_label)
    ax_ord = fig.gca()

    x_mes_sols = dict()
    result_DD = dict()
    result_DD['tol LF'] = tol_lf
    result_DD['tol DD'] = tol_dd
    # integrated mes, original order
    het_net, elec_net, heat_net, x_sol, iters, err_vec = run_LF_mes(max_iter=max_iter_lf,tol=tol_dd,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation,plot_top=False,plot_jac=False,plot_sol=False)
    x_mes_sols['mes'] = x_sol
    result_DD['mes'] = {'x mes':x_sol,'outer errors':err_vec,'iters':iters}
    plot_convergence(ax,range(iters+1),err_vec,max_data_points,'int. LF',ls='-',marker=markers_solver.get('mes'),color=colors_solver.get('mes'),alpha=.5)
    plot_convergence_order(ax_ord,err_vec,max_data_points,'int. LF',ls='-',marker=markers_solver.get('mes'),color=colors_solver.get('mes'),alpha=.5)
    # integrated mes, reorder to match DD
    het_net, elec_net, heat_net, x_sol, iters, err_vec = run_LF_mes(max_iter=max_iter_lf,tol=tol_dd,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation,plot_top=False,plot_jac=False,plot_sol=False,reorder=True)
    x_mes_sols['mes reod 2'] = x_sol
    result_DD['mes reod 2'] = {'x mes':x_sol,'outer errors':err_vec,'iters':iters}
    plot_convergence(ax,range(iters+1),err_vec,max_data_points,'int. LF, reod.',ls='-',marker=markers_solver.get('mes'),color=colors_solver.get('mes reod 2'),alpha=.5)
    plot_convergence_order(ax_ord,err_vec,max_data_points,'int. LF, reod.',ls='-',marker=markers_solver.get('mes'),color=colors_solver.get('mes reod 2'),alpha=.5)

    error_measures=['F','dy']
    max_iters_used = 0
    decouplings = [False,True]
    derivatives = [True,False]
    orderings = [1,2,3]
    for error_measure in error_measures:
        for ordering in orderings:
            for decoupled in decouplings:
                label = 'DD FP '+error_measure
                key = 'fp'
                if ordering == 2:
                    key += ' reod'
                    label += ' ord. 2'
                if ordering == 3:
                    key += ' reod 2'
                    label += ' ord. 3'
                if decoupled:
                    label += ' decoup.'
                    key += ' dec'
                    fig = plt.figure('conv_detail_FP_decoup_'+error_measure+'_'+str(ordering)+'_'+scale_var_label)
                else:
                    fig = plt.figure('conv_detail_FP_'+error_measure+'_'+str(ordering)+'_'+scale_var_label)
                key_sol = key + ' ' + error_measure
                ax_conv = fig.gca()
                max_iters_used_det = 0
                het_net, elec_net, heat_net, xe_sol, xh_sol, xc_sol, x_mes_sol, iter_nr, err_vec, errors_elec, errors_heat, errors_coupling, errors_elec_DD, errors_heat_DD, errors_coupling_DD = run_LF_mes_DD_FP(max_iter_lf=max_iter_lf,max_iter_fp=max_iter_dd,tol_lf=tol_lf,tol_fp=tol_dd,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,error_measure=error_measure,decoupled=decoupled,ordering=ordering)
                x_mes_sols[key_sol] = x_mes_sol
                result_DD[key_sol] = {'x mes':x_mes_sol,'iters':iter_nr,'outer errors':err_vec,'inner LF errors elec':errors_elec,'outer error elec':errors_elec_DD,'inner LF errors heat':errors_heat,'outer error heat':errors_heat_DD,'inner LF errors coupling':errors_coupling,'outer error coupling':errors_coupling_DD}
                if error_measure == 'F':
                    iter_start = 0
                    iter_range = iter_nr - iter_start
                elif error_measure == 'dy':
                    iter_start = 1
                    iter_range = iter_nr - iter_start
                max_iters_used = max(max_iters_used,iter_nr)
                max_iters_used_det = max(max_iters_used_det,iter_nr)
                plot_convergence_detailed(ax_conv,range(iter_start,iter_nr+1), err_vec, errors_elec, errors_heat, errors_coupling, errors_elec_DD, errors_heat_DD, errors_coupling_DD,tol_lf,tol_dd,art_zero,max_data_points,alpha=.5)
                plot_convergence(ax,range(iter_start,iter_nr+1),err_vec,max_data_points,label,ls='-',marker=markers_solver.get(error_measure),color=colors_solver.get(key),alpha=.5)
                plot_convergence_order(ax_ord ,err_vec,max_data_points,label,ls='-',marker=markers_solver.get(error_measure),color=colors_solver.get(key),alpha=.5)
                layout_convergence(ax_conv,tol_lf,tol_dd,art_zero,max_iters_used_det)

    for der in derivatives:
        for ordering in orderings:
            label = 'DD FP NR'
            key_color = 'fp NR'
            if ordering == 2:
                key_color += ' reod'
                label += ' ord. 2'
            if ordering == 3:
                key_color += ' reod 2'
                label += ' ord. 3'
            if der:
                fig = plt.figure('conv_detail_FP_NR_'+str(ordering)+'_'+scale_var_label)
                key = 'an'
            else:
                fig = plt.figure('conv_detail_FP_NR_FD_'+str(ordering)+'_'+scale_var_label)
                label += ' FD'
                key = 'FD'
            key_sol = key_color + ' ' + key
            ax_conv = fig.gca()
            max_iters_used_det = 0
            het_net, elec_net, heat_net, xe_sol, xh_sol, xc_sol, x_mes_sol, iter_nr, err_vec, errors_elec, errors_heat, errors_coupling, errors_elec_DD, errors_heat_DD, errors_coupling_DD = run_LF_mes_DD_FP_NR(max_iter_lf=max_iter_lf,max_iter_fp=max_iter_dd,tol_lf=tol_lf,tol_fp=tol_dd,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,plot_jac=True,der=der,ordering=ordering)
            x_mes_sols[key_sol] = x_mes_sol
            result_DD[key_sol] = {'x mes':x_mes_sol,'iters':iter_nr,'outer errors':err_vec,'inner LF errors elec':errors_elec,'outer error elec':errors_elec_DD,'inner LF errors heat':errors_heat,'outer error heat':errors_heat_DD,'inner LF errors coupling':errors_coupling,'outer error coupling':errors_coupling_DD}
            iter_start = 0
            iter_range = iter_nr - iter_start
            max_iters_used = max(max_iters_used,iter_nr)
            max_iters_used_det = max(max_iters_used_det,iter_nr)
            plot_convergence_detailed(ax_conv,range(iter_start,iter_nr+1), err_vec, errors_elec, errors_heat, errors_coupling, errors_elec_DD, errors_heat_DD, errors_coupling_DD,tol_lf,tol_dd,art_zero,max_data_points,alpha=.5)
            layout_convergence(ax_conv,tol_lf,tol_dd,art_zero,iter_nr)
            plot_convergence(ax,range(iter_start,iter_nr+1),err_vec,max_data_points,label,ls='-',marker=markers_solver.get(key),color=colors_solver.get(key_color),alpha=.5)
            plot_convergence_order(ax_ord ,err_vec,max_data_points,label,ls='-',marker=markers_solver.get(key),color=colors_solver.get(key_color),alpha=.5)

    layout_convergence(ax,tol_lf,tol_dd,art_zero,max_iters_used)
    layout_convergence_order(ax_ord)
    ax.legend()
    ax_ord.legend()

    print('\nRelative error to LF solution of integrated mes:')
    for key, x_mes_sol in x_mes_sols.items():
        print('For {}: rel. err.= {:.4e}'.format(key,error(x_mes_sol,x_mes_sols.get('mes'))))

    if save_data:
        with open(os.path.join(path_to_data,'x_mes_DD_'+scale_var_label+'.pkl'), "wb") as x_mes_file:
            pickle.dump(x_mes_sols,x_mes_file)
        with open(os.path.join(path_to_data,'result_DD_'+scale_var_label+'.pkl'), "wb") as result_DD_file:
            pickle.dump(result_DD,result_DD_file)

def compare_jacobians(formulation={'gas':'full','elec':'complex_power'},scale_var_params={'pgbase':ph0,'pbase':ph0,'qbase':.1,'Vbase':Vbase,'deltabase':1,'Sbase':Sbase,'phbase':ph0,'mbase':1,'Tbase':100,'phibase':Sbase,'Ebase':Sbase}):
    """Plot the Jacobians and their spectra for the various solution methods"""
    marker_size = 10
    # Integrated MES
    with HiddenPrints():
        het_net, elec_net, heat_net = create_mes_network()
    x = set_xmes_LF_init(het_net,formulation=formulation)

    scale_var = 'matrix'
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsys_unscaled = NonLinearSystemHeterogeneous(het_net,formulation=formulation)

    # unscaled system
    fig_J = nlsys_unscaled.spy_plot_J(x,title='Jacobian spy plot')
    ax_J = plt.gca()
    fig_J_map = nlsys_unscaled.imshow_J(x,title='Jacobian, unscaled')
    # spectrum of Jacobian
    fig_spec = nlsys_unscaled.spectrum_J(x,title='Unscaled spectra',color=colors_solver.get('mes'))
    ax_spec = fig_spec.gca()
    # spy plot of scaled J over original
    nlsys.spy_plot_J(x,ax=ax_J,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5)
    # colormap of scaled Jacobian
    fig_J_scaled_map = nlsys.imshow_J(x,title='Scaled Jacobian')
    # spectrum of (scaled) Jacobian
    fig_spectra = nlsys.spectrum_J(x,title='Scaled spectra',color=colors_solver.get('mes'))
    ax_spectra = fig_spectra.gca()

    # reordered system, unscaled
    Dx, Px, DF, PF = trans_matr_mes()
    Tx = Px.dot(Dx)
    TF = PF.dot(DF)
    fig_J_re = nlsys_unscaled.spy_plot_J(x,title='Reordered Jacobian spy plot, unscaled',P_F=TF,P_x=Tx,overlay=False)
    ax_J_re = plt.gca()
    nlsys_unscaled.spectrum_J(x,ax=ax_spec,P_F=TF,P_x=Tx,color=colors_solver.get('mes reod 2'))
    # reordered system, scaled
    nlsys.spy_plot_J(x,ax=ax_J_re,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5,P_F=TF,P_x=Tx,overlay=False)
    fig_J_re_scaled_map = nlsys.imshow_J(x,P_F=TF,P_x=Tx,title='Reordered, scaled, Jacobian',overlay=False)
    ax_J_re_scaled_map = fig_J_re_scaled_map.gca()
    # spectrum of (scaled) Jacobian
    nlsys.spectrum_J(x,ax=ax_spectra,P_F=TF,P_x=Tx,color=colors_solver.get('mes reod 2'))

    # plot overlay / lines in reoreded Jacobians
    xe_len = 4
    xh_len = 11
    xc_len = 4
    x_len = xe_len+xh_len+xc_len
    axes = [ax_J_re,ax_J_re_scaled_map]
    for ax in axes:
        ax.plot((-0.5,x_len-0.5),(xe_len-0.5,xe_len-0.5),'k-')
        ax.plot((-0.5,x_len-0.5),(xe_len+xh_len-0.5,xe_len+xh_len-0.5),'k-')
        ax.plot((xe_len-0.5,xe_len-0.5),(0-0.5,x_len-0.5),'k-')
        ax.plot((xe_len+xh_len-0.5,xe_len+xh_len-0.5),(0-0.5,x_len-0.5),'k-')

    legend_handles_spec = [Line2D([0],[0],color=colors_solver.get('mes'),marker=markers_solver.get('mes'),label='Int. MES'),Line2D([0], [0], color=colors_solver.get('mes reod 2'), marker=markers_solver.get('mes reod 2'),label='Int. MES reord.')]
    legend_handles_spy = [Line2D([0],[0],color='w',markerfacecolor='k',markeredgecolor='k',marker='.', markersize=marker_size,label='Scaled, an.'),Line2D([0], [0],color='w', markerfacecolor='tab:red',markeredgecolor='tab:red',marker='*',markersize=marker_size,label='Scaled, FD'),Line2D([0],[0],color='w',markerfacecolor='tab:blue',markeredgecolor='tab:blue',marker='s', markersize=marker_size,label='Unscaled, an.'),Line2D([0], [0],color='w', markerfacecolor='tab:green',markeredgecolor='tab:green',marker='d',markersize=marker_size,label='Unscaled, FD')]

    # Decomposed FP MES
    for ordering in [1,2,3]:
        fig_dgdy = plt.figure('Jacobian dg/dy spy plot, ordering {}'.format(ordering))
        ax_dgdy = fig_dgdy.gca()
        fig_FP = plt.figure('Jacobian I-dg/dy spy plot, ordering {}'.format(ordering))
        ax_FP = fig_FP.gca()
        key_color_NR = 'fp NR'
        key_color_dgdy = 'fp'
        if ordering == 2:
            label += ' ord. 2'
            key_color_NR += ' reod'
            key_color_dgdy += ' reod'
        if ordering == 3:
            label += ' ord. 3'
            key_color_NR += ' reod 2'
            key_color_dgdy += ' reod 2'
        for scale_var in [None,'matrix']:
            if scale_var == 'matrix':
                ax_s = ax_spectra
            else:
                ax_s = ax_spec
            for der in [False,True]:
                label = 'dg/dy'
                if ordering == 2:
                    label += ' ord. 2'
                if ordering == 3:
                    label += ' ord. 3'
                if der:
                    key = 'an'
                else:
                    label += ' FD'
                    key = 'FD'
                # create networks
                elec_net = create_electrical_network(scale_var=scale_var,scale_var_params=scale_var_params)
                elec_net = reset_network_without_update_elec(elec_net)
                heat_net = create_heat_network(scale_var=scale_var,scale_var_params=scale_var_params)
                heat_net = reset_network_without_update_heat(heat_net)
                with HiddenPrints():
                    coupling_net = create_coupling_network(scale_var=scale_var,scale_var_params=scale_var_params)
                # initialize networks, and set initial guesses
                xe_new = set_xe_LF_init(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
                xh_new = set_xh_LF_init(heat_net,formulation=formulation.get('heat'),scale_var=scale_var,scale_var_params=scale_var_params)
                with HiddenPrints():
                    xc_new = set_xc_LF_init(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
                y_new = get_y_FP(elec_net,heat_net,coupling_net,formulation=formulation,ordering=ordering) # unscaled
                ve_new,be_new,vh_new,bh_new,vc_new,bc_new = get_suby_FP(y_new,ordering=ordering) # unscaled

                len_y = len(y_new)
                if scale_var == 'matrix':
                    veb = scale_var_params.get('Sbase')*np.ones(3)
                    beb = scale_var_params.get('Sbase')*np.ones(1)
                    vhb = np.array([scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('phibase')])
                    bhb = scale_var_params.get('phibase')*np.ones(1)
                    vcb = np.array([scale_var_params.get('qbase'),scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('phibase')])
                    bcb = np.array([scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('mbase'),scale_var_params.get('mbase'),scale_var_params.get('phibase')])
                    if ordering == 1:
                        Dy = sps.diags(1/np.concatenate((veb,beb,vhb,bhb,vcb,bcb)))
                    elif ordering == 2:
                        Dy = sps.diags(1/np.concatenate((veb,vhb,vcb,beb,bhb,bcb)))
                    elif ordering == 3:
                        Dy = sps.diags(1/np.concatenate((veb,vhb,bcb,vcb,beb,bhb)))
                    Dy_inv = sps.diags(1/Dy.data[0])
                else:
                    Dy = np.eye(len_y)
                    Dy_inv = np.eye(len_y)
                if der: # use analytical derivatives
                    elec_net, heat_net, coupling_net, g_new, err_vece, err_vech, err_vecc = g_FP(elec_net,heat_net,coupling_net,y_new,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,ordering=ordering)
                    elec_net, heat_net, coupling_net, Jg_new = jac_FP(elec_net,heat_net,coupling_net,y_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,ordering=ordering,method=1)
                else:
                    elec_net, heat_net, coupling_net, Jg_new, g_new, err_vece, err_vech, err_vecc = jac_FP_FD(elec_net,heat_net,coupling_net,y_new,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy,Dy_inv=Dy_inv,ordering=ordering)
                    if scale_var == 'matrix':
                        Jg_new = Dy.todense().dot(Jg_new.dot(Dy_inv.todense()))
                # spectra
                J_new = np.eye(len_y,dtype='float64') - Jg_new
                eigs_Jg,_ = sps.linalg.eigs(Jg_new,k=Jg_new.shape[0]-2)
                radius_Jg = np.max(np.abs(eigs_Jg))
                eigs_J,_ = sps.linalg.eigs(J_new,k=J_new.shape[0]-2)
                radius_J = np.max(np.abs(eigs_J))
                for eig in eigs_Jg:
                    ax_s.plot(eig.real,eig.imag,marker=markers_solver.get(key),color=colors_solver.get(key_color_dgdy))
                ax_s.add_artist(plt.Circle((0, 0), radius_Jg, ls = '-',color=colors_solver.get(key_color_dgdy), fill=False))
                for eig in eigs_J:
                    ax_s.plot(eig.real,eig.imag,marker=markers_solver.get(key),color=colors_solver.get(key_color_NR))
                ax_s.add_artist(plt.Circle((0, 0), radius_J, ls = '--',color=colors_solver.get(key_color_NR), fill=False))
                if scale_var == 'matrix': # legends are the same for scaled and unscaled
                    legend_handles_spec.append(Line2D([0],[0],ls = '-',marker=markers_solver.get(key),color=colors_solver.get(key_color_dgdy),label=label))
                    legend_handles_spec.append(Line2D([0],[0],ls = '--',marker=markers_solver.get(key),color=colors_solver.get(key_color_NR),label='I-'+label))
                # spy plots
                if scale_var == 'matrix':
                    if der:
                        ax_dgdy.spy(Jg_new,markerfacecolor='k',markeredgecolor='k',marker='.',alpha=1)
                        ax_FP.spy(J_new,markerfacecolor='k',markeredgecolor='k',marker='.',alpha=1)
                    else:
                        ax_dgdy.spy(Jg_new,markerfacecolor='tab:red',markeredgecolor='tab:red',marker='*',alpha=.5)
                        ax_FP.spy(J_new,markerfacecolor='tab:red',markeredgecolor='tab:red',marker='*',alpha=.5)
                else:
                    if der:
                        ax_dgdy.spy(Jg_new,markerfacecolor='tab:blue',markeredgecolor='tab:blue',marker='s',alpha=.5)
                        ax_FP.spy(J_new,markerfacecolor='tab:blue',markeredgecolor='tab:blue',marker='s',alpha=.5)
                    else:
                        ax_dgdy.spy(Jg_new,markerfacecolor='tab:green',markeredgecolor='tab:green',marker='d',alpha=.5)
                        ax_FP.spy(J_new,markerfacecolor='tab:green',markeredgecolor='tab:green',marker='d',alpha=.5)
        # plot overlay / lines in spyplot Jacobian
        for ax in [ax_dgdy, ax_FP]:
            overlay_FP_jac(ax,y_new,ordering=ordering)
        ax_dgdy.legend(handles=legend_handles_spy)
        ax_FP.legend(handles=legend_handles_spy)

    legend_handles_spec.append(Line2D([0],[0],ls = '-',color='k',label='conv. radius FP (for dg/dy)'))
    ax_spec.legend(handles=legend_handles_spec)
    ax_spec.add_artist(plt.Circle((0, 0), 1,color='k', fill=False))
    ax_spectra.legend(handles=legend_handles_spec)
    ax_spectra.add_artist(plt.Circle((0, 0), 1,color='k', fill=False))

def table_LF_results(path_to_data,max_iter=10,tol=1e-6,scale_var=None,scale_var_params=None,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'}):
    """Write the full solution to a txt file, which can be written by latex to create a table."""
    with HiddenPrints():
        het_net, elec_net, heat_net, x_sol, iters, err_vec = run_LF_mes(max_iter=max_iter,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,plot_top=False,plot_jac=False,plot_sol=False,reorder=False)
        x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)

    nodes = ['0','1']
    links = ['0--1']

    P_loss_total = 0
    Q_loss_total = 0
    with open(os.path.join(path_to_data,'elec_data_EH2N_thesis.txt'), "w") as table:
        for ind in range(len(nodes)):
            node_data = ' ' + nodes[ind] + r' '
            if ind < len(links):
                link_data = ' ' + links[ind] + r' '
            else:
                link_data = ' '
            if elec_net.nodes[ind] in list(unknown_V_nodes):
                node_data += r' & {:.3f} '.format(elec_net.nodes[ind].get_V()/kV)
            else:
                node_data += r' & \textbf{'+'{:.3f}'.format(elec_net.nodes[ind].get_V()/kV)+'} '
            if elec_net.nodes[ind] in list(unknown_delta_nodes):
                node_data += r' & {:.3f} '.format(elec_net.nodes[ind].get_delta())
            else:
                node_data += r' & \textbf{'+'{:.0f}'.format(elec_net.nodes[ind].get_delta())+'} '
            if len(elec_net.nodes[ind].half_links):
                hl = elec_net.nodes[ind].half_links[0]
                if hl.bc_type in [1,2]: # P known
                    node_data += r' & \textbf{'+'{:.1f} '.format(hl.get_P()/MW)+'} '
                else:
                    node_data += r' & {:.3f} '.format(hl.get_P()/MW)
                if hl.bc_type in [1,3]: # Q known
                    node_data += r'\textbf{'+'{:+.1f}$\iu$  '.format(hl.get_Q()/MW)+'} '
                else:
                    node_data += r'{:+.3f}$\iu$  '.format(hl.get_Q()/MW)
            else:
                node_data += r' & - '
            if ind < len(links):
                P_loss = elec_net.links[ind].complex_power_loss().real/MW
                Q_loss = elec_net.links[ind].complex_power_loss().imag/MW
                P_loss_total += P_loss
                Q_loss_total += Q_loss
                link_data += r' & {:.3f} {:+.3f}$\iu '.format(P_loss, Q_loss)
            else:
                link_data += r' & '
            table.write(node_data)
            table.write(r' & ')
            table.write(link_data)
            table.write(r' & ')
        # table.write(r' \hline ')
        # table.write(r' & & & & Total & {:.3f}  {:+.3f}$\iu '.format(P_loss_total,Q_loss_total))
        # table.write(r' & ')

    with open(os.path.join(path_to_data,'heat_data_EH2N_hydraulic_thesis.txt'), "w") as table:
        for ind in range(len(nodes)):
            node_data = ' ' + nodes[ind] + r' '
            if ind < len(links):
                link_data = ' ' + links[ind] + r' '
            else:
                link_data = ' '
            if heat_net.nodes[ind] in list(unknown_p_nodes):
                node_data += r' & {:.3f} '.format(heat_net.nodes[ind].get_p()/bar)
            else:
                node_data += r' & \textbf{'+'{:.3f}'.format(heat_net.nodes[ind].get_p()/bar)+'} '
            if len(heat_net.nodes[ind].half_links):
                if formulation.get('heat') == 'standard' or heat_net.nodes[ind].half_links[0] in unknown_m_halflinks:
                    node_data += r' & {:.3f} '.format(heat_net.nodes[ind].half_links[0].get_m())
                else:
                    node_data += r' & \textbf{'+'{:.3f} '.format(heat_net.nodes[ind].half_links[0].get_m())+'} '
            else:
                node_data += r' & - '
            if ind < len(links):
                link_data += r' & {:.3f}'.format(heat_net.links[ind].get_m())
            else:
                link_data += r' & '
            table.write(node_data)
            table.write(r' & ')
            table.write(link_data)
            table.write(r' & ')

    phi_loss_total = 0
    with open(os.path.join(path_to_data,'heat_data_EH2N_thermal_thesis.txt'), "w") as table:
        for ind in range(len(nodes)):
            node_data = ' ' + nodes[ind] + r' '
            if ind < len(links):
                link_data = ' ' + links[ind] + r' '
            else:
                link_data = ' '
            if heat_net.nodes[ind] in list(unknown_Ts_nodes):
                node_data += r' & {:.3f} '.format(heat_net.nodes[ind].get_Ts())
            else:
                node_data += r' & \textbf{'+'{:.0f}'.format(heat_net.nodes[ind].get_Ts())+'} '
            if heat_net.nodes[ind] in list(unknown_Tr_nodes):
                node_data += r' & {:.3f} '.format(heat_net.nodes[ind].get_Tr())
            else:
                node_data += r' & \textbf{'+'{:.0f}'.format(heat_net.nodes[ind].get_Tr())+'} '
            if len(heat_net.nodes[ind].half_links):
                node_data += r' & \textbf{'+'{:.1f} '.format(heat_net.nodes[ind].half_links[0].get_dphi()/MW)+'} '
            else:
                node_data += r' & - '
            if ind < len(links):
                phi_loss = heat_net.links[ind].heat_loss_supply()/MW + heat_net.links[ind].heat_loss_return()/MW
                phi_loss_total += phi_loss
                link_data += r' & {:.3f}'.format(phi_loss)
            else:
                link_data += r' & '
            table.write(node_data)
            table.write(r' & ')
            table.write(link_data)
            table.write(r' & ')
        # table.write(r' \hline ')
        # table.write(r' & & & & Total & {:.3f} '.format(phi_loss_total))
        # table.write(r' & ')

    unit = [r'\acrshort{eh} at 0',r'\acrshort{eh} at 1']
    with open(os.path.join(path_to_data,'coupling_data_EH2N_thesis.txt'), "w") as table:
        for ind in range(len(unit)):
            link_data = ' ' + unit[ind] + r' '
            # gas data
            if het_net.nodes[ind-2].half_links[0] in unknown_qc_halflinks:
                link_data += r' & {:.3f}'.format(het_net.nodes[ind-2].half_links[0].get_q())
            else:
                link_data += r' & \textbf{'+'{:.3f}'.format(het_net.nodes[ind-2].half_links[0].get_q())+'} '
            # elec data
            if elec_net.links[ind-2] in unknown_Pc_links:
                link_data += r' & {:.3f}'.format(elec_net.links[ind-2].get_Pstart()/MW)
            else:
                link_data += r' & \textbf{'+'{:.3f}'.format(elec_net.links[ind-2].get_Pstart()/MW)+'} '
            if elec_net.links[ind-2] in unknown_Qc_links:
                link_data += r' & {:.3f} '.format(elec_net.links[ind-2].get_Qstart()/MW)
            else:
                link_data += r' & \textbf{'+'{:.3f}'.format(elec_net.links[ind-2].get_Qstart()/MW)+'} '
            # heat data
            if heat_net.links[ind-len(unit)] in unknown_mc_links:
                link_data += r' & {:.3f}'.format(heat_net.links[ind-len(unit)].get_m())
            else:
                link_data += r' & \textbf{'+'{:.3f}'.format(heat_net.links[ind-len(unit)].get_m())+'} '
            if heat_net.links[ind-len(unit)] in unknown_dphi_links:
                link_data += r' & {:.3f}'.format(-heat_net.links[ind-len(unit)].get_dphistart()/MW)
            else:
                link_data += r' & \textbf{'+'{:.3f}'.format(-heat_net.links[ind-len(unit)].get_dphistart()/MW)+'} '
            if heat_net.links[ind-len(unit)] in unknown_Ts_links:
                link_data += r' & {:.3f}'.format(heat_net.links[ind-len(unit)].get_Tsstart())
            else:
                link_data += r' & \textbf{'+'{:.3f}'.format(heat_net.links[ind-len(unit)].get_Tsstart())+'} '
            table.write(link_data)
            table.write(r' & ')

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','EH2N')
    # run_LF_elec()
    # run_LF_heat()
    # run_LF_mes(plot_top=True,plot_jac=True,plot_sol=True)
    # run_LF_coupling()

    tol_lf = 1e-7#1e-6
    tol_dd = 10*tol_lf#1e-6
    art_zero = 1e-16
    max_iter_dd = 200#500
    max_iter_lf = 10
    max_data_points = 80
    formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow'}

    # # unscaled
    # scale_var=None
    # scale_var_params=None
    # compare_DD(tol_lf=tol_lf,tol_dd=tol_dd,art_zero=art_zero,max_iter_dd=max_iter_dd,max_iter_lf=max_iter_lf,max_data_points=max_data_points,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)

    # matrix scaling
    scale_var='matrix'
    qbase = .1
    pbase = ph0
    mbase = 1
    Tbase = 100.
    phibase = Sbase
    Egbase = Sbase
    scale_var_params={'pgbase':pbase,'pbase':pbase,'qbase':qbase,'Vbase':Vbase,'deltabase':1,'Sbase':Sbase,'phbase':pbase,'mbase':mbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}
    compare_DD(tol_lf=tol_lf,tol_dd=tol_dd,art_zero=art_zero,max_iter_dd=max_iter_dd,max_iter_lf=max_iter_lf,max_data_points=max_data_points,scale_var=scale_var,scale_var_params=scale_var_params,save_data=False,path_to_data=path_to_data)

    # # Spectra and spy plots of the various Jacobians
    # qbase = .1
    # pbase = ph0
    # mbase = 1
    # Tbase = 100.
    # phibase = Sbase
    # Egbase = Sbase
    # scale_var_params={'pgbase':pbase,'pbase':pbase,'qbase':qbase,'Vbase':Vbase,'deltabase':1,'Sbase':Sbase,'phbase':pbase,'mbase':mbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}
    # compare_jacobians(formulation=formulation,scale_var_params=scale_var_params)

    # table_LF_results(path_to_data,max_iter=max_iter_lf,tol=tol_lf,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)

    plt.show()
