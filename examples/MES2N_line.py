"""
MES consisting of 2 nodes per carrier (i.e. only one link in every single-carrier network.
"""
import warnings
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.carrier import Gas
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.utils.constants import bar, mbar, hour, mm, kV, MW, MBTU, BTU, MSCM
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import pytest

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning 

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
# physical parameters of network and pipes
L_g = 500. #[m]
D_g = .15 #[m]
gas_link_type = 'pipe_low_pres_pole'
gas_link_params = {'L':L_g,'D':D_g,'carrier':gas}
link_eq = 'dp_of_q'
# coordinates
x0g = 0
y0g = 0
x1g = 1
y1g = 0

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
x0e = 3
y0e = -1
x1e = 4
y1e = -1
# boundary conditions
V0 = 1.*Vbase #[V]
delta0 = -0.10478613961080965 #[rad]
P1 = 1*Sbase #[W]
Q1 = 1*Sbase #[W]

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
# coordinates
x0h = 3
y0h = 1
x1h = 4
y1h = 1
# boundary conditions
ph0 = 100*rho_w*grav_const #[Pa]
Ts0 = 100 #[C]
phi1 = 1*MW #[W]
To1 = 50. #[C]
# initial conditions
m_ic = 2 #[kg/s]
m_hl_ic = 1 #[kg/s]
p1h_ic = 99*rho_w*grav_const #[Pa]
Ts1_ic = 99 #[C]
Tr0_ic = 49 #[C]
Tr1_ic = 50 #[C]

# physical parameters of the coupling unit
eta_GG = .6
GHV = 40.611 #[MBTU/m^3]
GHV *= MBTU*BTU #[Wh/m^3]
GHV *= hour/rhon_g #[J/kg]
unit_type_GG = 'ge_gas_fired_gen'
unit_params_GG={'eta':eta_GG,'GHV':GHV}
eta_GB = .7
unit_type_GB = 'gh_gas_boiler'
unit_params_GB={'eta':eta_GB,'GHV':GHV}
eta_CHP = np.array([eta_GG,eta_GB])
unit_type_CHP = 'geh_CHP'
unit_params_CHP = {'eta':eta_CHP,'GHV':GHV}
Pbs_GG_MW = 1 # The values in Shabanpour are specified for P given in 1MW
Pb = 1
Pbs_GG = Pbs_GG_MW*MW
qbs = 1*MSCM*rhon_g/hour
GHVbs = MBTU*BTU*hour/(rhon_g)
Ebs = qbs*GHVbs
a_GG_pu = 0.01 # Assumes P to be in 100 MW, and q to be in MSCM
b_GG_pu = 4.
c_GG_pu = 150.
d_GG_pu = 15.
e_GG_pu = 0.5
Pmin_pu = 0.
a_GG = a_GG_pu * Ebs/Pbs_GG**2 *Pb**2
b_GG = b_GG_pu * Ebs/Pbs_GG *Pb
c_GG = c_GG_pu * Ebs
d_GG = d_GG_pu * Ebs
e_GG = e_GG_pu * 1/Pbs_GG *Pb
Pmin = Pmin_pu * Pbs_GG / Pb
unit_type_GG_VP='ge_gas_fired_gen_valve_point'
unit_params_GG_VP={'a':a_GG,'b':b_GG,'c':c_GG,'d':d_GG,'e':e_GG,'Pmin':Pmin,'GHV':GHV}    
# coordinates
x0c = 2
y0c = 0
# initial conditions
P_ic = 2*Sbase #[W]
Q_ic = 2*Sbase #[W]
phi_ic = 2*MW
Toc_ic = 110 #[C]
# solution (or boundary conditions when needed)
Pc_sol = 1.02619363*MW
Qc_sol = 1.26193629*MW
phic_sol = 1.0149629315743993*MW
Tsc_sol = Ts0
Trc_sol = 49.75308089619643
mc_sol = phic_sol/(Cp_w*(Tsc_sol-Trc_sol))
qc_ge_GG_VP_sol = 0.787060558753952
qc_ge_GG_sol = Pc_sol/(GHV*eta_GG)
qc_gh_sol = phic_sol/(GHV*eta_GB)
qc_geh_sol = 1/GHV * (Pc_sol/eta_CHP[0] + phic_sol/eta_CHP[1])

# boundary conditions gas network
pg0 = 50*mbar #[Pa]
q1 = P1/(eta_GG*GHV) #[kg/s]
# initial conditions
p1_ic = 40*mbar #[Pa]
q_ic = 0.05 #[kg/s]


# solver information
tol=1e-6
max_iter=15
formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}

def create_gas_network():
    """Create a gas network consisting of one source connected to one sink"""
    g0 = GasNode('gn0',node_type=0,x=x0g,y=y0g,p=pg0) # reference node
    g1 = GasNode('gn1',node_type=1,x=x1g,y=y1g,q=q1) # load node
    
    gl0 = GasLink('gl0',g0,g1,link_type = gas_link_type,link_params = gas_link_params,link_eq_form=link_eq)
    gas_net = GasNetwork('2 nodes')
    gas_net.add_link(gl0)
    
    return gas_net

def initialize_gas_network(gas_net):
    """Initialize the gas network, consisting of one source connected to one sink.
    
    Parameters
    ----------
    gas_net : GasNetwork
        The gas network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    q_init = q_ic*np.ones(len(gas_net.links))
    p_init = np.array([p1_ic])
    
    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    gas_net.update(x_init,formulation=formulation.get('gas'))
    x0 = gas_net.set_x_init(formulation=formulation.get('gas'))
    return x0

def run_gas_load_flow():
    """Steady-state load flow analysis of gas network, without scaling. The default values are used for initialization.
    """
    # create network
    gas_net = create_gas_network()
    # initialize
    x0 = initialize_gas_network(gas_net)
    
    # solve network
    print('\nRunning load flow for single-carrier gas network')
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('gas'))
    
    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_sol/mbar))
    print('q = {} kg/s'.format(q_sol))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    
    return gas_net,x_sol,iters,err_vec

def create_electrical_network():
    """Create an electrical network consisting of one source connected to one sink"""
    e0 = ElectricalNode('en0',node_type=0,x=x0e,y=y0e,V=V0,delta=delta0) # ref
    e1 = ElectricalNode('en1',x=x1e,y=y1e,node_type=2,P=P1,Q=Q1) # load
    
    el0 = ElectricalLink('el0',e0,e1,link_type=elec_link_type,link_params=elec_link_params)
    
    elec_net = ElectricalNetwork('2 nodes')
    elec_net.add_link(el0)
    
    return elec_net

def initialize_electrical_network(elec_net):
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
    x_init = np.concatenate((delta_init,V_init))
    
    elec_net.initialize()
    elec_net.update(x_init)
    x0 = elec_net.set_x_init()
    return x0

def run_electrical_load_flow():
    """Steady-state load flow analysis of electrical network, without scaling. The default values are used for initialization.
    """
    # create network
    elec_net = create_electrical_network()
    
    x0 = initialize_electrical_network(elec_net)
    
    # solve network
    print('\nRunning load flow for single-carrier electrical network')
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR')
    
    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('delta = {}'.format(delta_sol))
    print('|V| = {} V'.format(V_sol))
    print('|V| = {} p.u.'.format(V_sol/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('elec hl start nodes = {}'.format([hl.start_node.name for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    
    return elec_net,x_sol,iters,err_vec

def create_heat_network():
    """Create a heat network consisting of one source connected to one sink"""
    h0 = HeatNode('hn0',node_type=0,x=x0h,y=y0h,p=ph0,Ts=Ts0) # slack
    h0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h1 = HeatNode('hn1',node_type=1,x=x1h,y=y1h,Tr_hl=To1,dphi=phi1) # load node (sink)
    h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    
    hl0 = HeatLink('hl0',h0,h1,link_type=heat_link_type,link_params=heat_link_params)
    heat_net = HeatNetwork('2 nodes',Ta=Ta)
    heat_net.add_link(hl0)
    
    return heat_net

def initialize_heat_network(heat_net):
    """Initialize the heat network, consisting of one source connected to one sink.
    
    Parameters
    ----------
    heat_net : HeatNetwork
        The heat network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    m_init = np.array([m_ic]) #[kg/s]
    m_hl_init = np.array([m_hl_ic]) #[kg/s]
    p_init = np.array([p1h_ic])
    Ts_init = np.array([Ts1_ic]) #[C]
    Tr_init = np.array([Tr0_ic,Tr1_ic]) #[C]
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    heat_net.update(x_init,formulation=formulation.get('heat'))
    x0 = heat_net.set_x_init(formulation=formulation.get('heat'))
    return x0

def run_heat_load_flow():
    """Steady-state load flow analysis of heat network, without scaling. The default values are used for initialization.
    """
    # create network
    heat_net = create_heat_network()
    
    # initialize
    x0 = initialize_heat_network(heat_net)
    
    # solve network
    print('\nRunning load flow for single-carrier heat network')
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('heat'))
    
    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p heat = {} m'.format(p_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {}'.format(m_hl_vec))
    print('Ts hl = {}'.format(Ts_hl_vec))
    print('Tr hl = {}'.format(Tr_hl_vec))
    print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
    
    return heat_net,x_sol,iters,err_vec

def create_mes_ge_network(unit='GG'):
    """Create a combined gas and electrical network consisting of one source connected to one sink per carrier. Using one coupling node"""
    # create single-carrier networks
    gas_net = create_gas_network()
    elec_net = create_electrical_network()
    
    # coupling
    if unit == 'GG':
        unit_type = unit_type_GG
        unit_params = unit_params_GG
    elif unit == 'GG_VP':
        unit_type = unit_type_GG_VP
        unit_params = unit_params_GG_VP
    else:
        raise ValueError('Enter valid unit')
    c0 = HeterogeneousNode('cn0',node_type=0,x=x0c,y=y0c,unit_type=unit_type,unit_params=unit_params)
    
    # change node types/BCs single-carrier networks
    elec_coupling_node = elec_net.nodes[0]
    elec_coupling_node.node_type = 5 # PQVdelta
    ElectricalHalfLink('en0_hl',elec_coupling_node,P=0,Q=0) # was slack, so didn't have a half link. All the required power is taken from the gas network
    gas_coupling_node = gas_net.nodes[1]
    gas_coupling_node.half_links[0].q = 0 # The gas load is the electrical network
    
    # dummy links
    gl1 = GasLink('gl1',gas_coupling_node,c0)
    gas_net.add_link(gl1)
    el1 = ElectricalLink('el1',c0,elec_coupling_node)
    elec_net.add_link(el1)
    
    het_net = HeterogeneousNetwork('2 nodes')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_node(c0)
    
    return het_net, gas_net, elec_net

def initialize_mes_ge_network(het_net):
    """Initialize the combined gas and electrical network consisting of one source connected to one sink per carrier.
    
    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The multi-carrier network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    q_init = np.array([q_ic])
    gp_init = np.array([p1_ic])
    delta_init = np.zeros(1) # default flat initialization of 0
    V_init = Vbase*np.ones(1) # default flat initialization corresponding to 1 p.u.
    qc_init = np.array([q_ic])
    Pc_init = np.array([P_ic])*Sbase
    Qc_init = np.array([Q_ic])*Sbase
    Sc_init = np.concatenate((Pc_init,Qc_init))
    
    xg_init = np.concatenate((q_init,gp_init))
    xe_init = np.concatenate((delta_init,V_init))
    xc_init = np.concatenate((qc_init,Sc_init))
    x_init = np.concatenate((xg_init,xe_init,xc_init))
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)
    return x0

def run_mes_ge_load_flow(unit='GG'):
    """Steady-state load flow analysis of combined gas and electrical network, without scaling. The default values are used for initialization.
    """
    # create network
    het_net, gas_net, elec_net = create_mes_ge_network(unit=unit)
    
    # initialize network
    x0 = initialize_mes_ge_network(het_net)
    
    # solve network
    print('\nRunning load flow for multi-carrier gas-electricity network, with {} coupling'.format(unit))
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
    
    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    
    return het_net, gas_net, elec_net,x_sol,iters,err_vec

def create_mes_gh_network():
    """Create a combined gas and heat network consisting of one source connected to one sink per carrier. Using one coupling node"""
    # create single-carrier networks
    gas_net = create_gas_network()
    heat_net = create_heat_network()
    
    # coupling
    c0 = HeterogeneousNode('cn0',node_type=0,x=x0c,y=y0c,unit_type=unit_type_GB,unit_params=unit_params_GB) # To unknown
    
    # change node types/BCs single-carrier networks
    heat_coupling_node = heat_net.nodes[0]
    heat_coupling_node.node_type = 7 # reference temperature (junction) node
    for hl in heat_coupling_node.get_half_links(): # is a junction node now, so remove any half links
        heat_coupling_node.remove_half_link(hl)
        heat_net.remove_half_link(hl)
    heat_coupling_node.half_links[:] = list()
    gas_coupling_node = gas_net.nodes[1]
    gas_coupling_node.half_links[0].q = 0 # The gas load is the heat network
    
    # dummy links
    gl1 = GasLink('gl1',gas_coupling_node,c0)
    gas_net.add_link(gl1)
    hl1 = HeatLink('hl1',c0,heat_coupling_node,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
    heat_net.add_link(hl1)
    
    het_net = HeterogeneousNetwork('2 nodes')
    het_net.add_network(gas_net)
    het_net.add_network(heat_net)
    het_net.add_node(c0)
    
    return het_net, gas_net, heat_net

def initialize_mes_gh_network(het_net):
    """Initialize the combined gas and heat network consisting of one source connected to one sink per carrier.
    
    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The multi-carrier network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    q_init = np.array([q_ic])
    gp_init = np.array([p1_ic])
    m_init = np.array([m_ic]) #[kg/s]
    m_hl_init = np.array([m_hl_ic]) #[kg/s]
    hp_init = np.array([p1h_ic])
    Ts_init = np.array([Ts1_ic]) #[C]
    Tr_init = np.array([Tr0_ic,Tr1_ic]) #[C]
    qc_init = np.array([q_ic])
    mc_init = np.array([m_ic]) #[kg/s]
    phic_init = np.array([phi_ic]) #[W]
    Toc_init = np.array([Toc_ic])

    xg_init = np.concatenate((q_init,gp_init))
    xh_init = np.concatenate((m_init,m_hl_init,hp_init,Ts_init,Tr_init))
    xc_init = np.concatenate((qc_init,mc_init,phic_init,Toc_init))
    x_init = np.concatenate((xg_init,xh_init,xc_init))
    het_net.initialize()   
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)
    return x0

def run_mes_gh_load_flow():
    """Steady-state load flow analysis of combined gas and heat network, without scaling. The default values are used for initialization.
    """
    # create network
    het_net, gas_net, heat_net = create_mes_gh_network()
    
    # initialize network
    x0 = initialize_mes_gh_network(het_net)
    
    # solve network
    print('\nRunning load flow for multi-carrier gas-heat network')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
    
    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Tr hl = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Ts hl c = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('Tr hl c = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('dphi hl c = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('m hl c = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    
    return het_net, gas_net, heat_net, x_sol, iters, err_vec

def create_mes_geh_network():
    """Create a combined gas, electricity, and heat network consisting of one source connected to one sink per carrier. Using one coupling node"""
    # create single-carrier networks
    gas_net = create_gas_network()
    elec_net = create_electrical_network()
    heat_net = create_heat_network()
    
    # coupling
    c0 = HeterogeneousNode('cn0',node_type=0,x=x0c,y=y0c,unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To unknown
    
    # change node types/BCs single-carrier networks
    heat_coupling_node = heat_net.nodes[0]
    heat_coupling_node.node_type = 7 # reference temperature (junction) node
    for hl in heat_coupling_node.get_half_links(): # is a junction node now, so remove any half links
        heat_coupling_node.remove_half_link(hl)
        heat_net.remove_half_link(hl)
    heat_coupling_node.half_links[:] = list()
    elec_coupling_node = elec_net.nodes[0]
    elec_coupling_node.node_type = 5 # PQVdelta
    ElectricalHalfLink('en0_hl',elec_coupling_node,P=0,Q=0) # was slack, so didn't have a half link. All the required power is taken from the gas network
    gas_coupling_node = gas_net.nodes[1]
    gas_coupling_node.half_links[0].q = 0 # The gas load is the heat network
    
    # dummy links
    gl1 = GasLink('gl1',gas_coupling_node,c0)
    gas_net.add_link(gl1)
    el1 = ElectricalLink('el1',c0,elec_coupling_node)
    elec_net.add_link(el1)
    hl1 = HeatLink('hl1',c0,heat_coupling_node,link_params={'carrier':water},bc_type=0) # To of coupling (source) is unknown
    heat_net.add_link(hl1)
    
    het_net = HeterogeneousNetwork('2 nodes')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    het_net.add_node(c0)
    
    return het_net, gas_net, elec_net, heat_net

def initialize_mes_geh_network(het_net):
    """Initialize the combined gas, electrical, and heat network consisting of one source connected to one sink per carrier.
    
    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The multi-carrier network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    q_init = np.array([q_ic])
    gp_init = np.array([p1_ic])
    delta_init = np.zeros(1) # default flat initialization of 0
    V_init = Vbase*np.ones(1) # default flat initialization corresponding to 1 p.u.
    m_init = np.array([m_ic]) #[kg/s]
    m_hl_init = np.array([m_hl_ic]) #[kg/s]
    hp_init = np.array([p1h_ic])
    Ts_init = np.array([Ts1_ic]) #[C]
    Tr_init = np.array([Tr0_ic,Tr1_ic]) #[C]
    qc_init = np.array([q_ic])
    Pc_init = np.array([P_ic])*Sbase
    Qc_init = np.array([Q_ic])*Sbase
    Sc_init = np.concatenate((Pc_init,Qc_init))
    mc_init = np.array([m_ic]) #[kg/s]
    phic_init = np.array([phi_ic]) #[W]
    Toc_init = np.array([Toc_ic])
    
    xg_init = np.concatenate((q_init,gp_init))
    xe_init = np.concatenate((delta_init,V_init))
    xh_init = np.concatenate((m_init,m_hl_init,hp_init,Ts_init,Tr_init))
    xc_init = np.concatenate((qc_init,Sc_init,mc_init,phic_init,Toc_init))
    x_init = np.concatenate((xg_init,xe_init,xh_init,xc_init))
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)
    return x0

def run_mes_geh_load_flow():
    """Steady-state load flow analysis of combined gas, electrical, and heat network, without scaling. The default values are used for initialization.
    """
    # create network
    het_net, gas_net, elec_net, heat_net = create_mes_geh_network()
    
    # initialize network
    x0 = initialize_mes_geh_network(het_net)
    
    # solve network
    print('\nRunning load flow for multi-carrier gas-electricity-heat network')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
    
    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
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
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Tr hl = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
    print('Ts hl c = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('Tr hl c = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('dphi hl c = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    print('m hl c = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
    
    return het_net, gas_net, elec_net, heat_net, x_sol, iters, err_vec

def create_coupling_ge_network(unit='GG'):
    """Create a coupling network consisting of one node, for a gas-electricity coupling"""
    if unit == 'GG':
        unit_type = unit_type_GG
        unit_params = unit_params_GG
    elif unit == 'GG_VP':
        unit_type = unit_type_GG_VP
        unit_params = unit_params_GG_VP
    else:
        raise ValueError('Enter valid unit')
    c0 = HeterogeneousNode('cn0',node_type=0,x=x0c,y=y0c,unit_type=unit_type,unit_params=unit_params)
    hlg = GasHalfLink('cn0_hlg',c0,-q_ic,bc_type=0) # gas flows into coupling node, q is unknown
    hle = ElectricalHalfLink('cn0_hle',c0,P=Pc_sol,Q=Qc_sol,bc_type=1) # P and Q are known
    coupling_ge_net = HeterogeneousNetwork('coupling network')
    coupling_ge_net.add_node(c0)
    
    return coupling_ge_net

def initialize_coupling_ge_network(coupling_ge_net):
    """Initialize the coupling network consisting of one node, for a gas-electricity coupling
    
    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The coupling network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    x_init = np.array([q_ic])#q_ic,P_ic,Q_ic])
    coupling_ge_net.initialize()
    coupling_ge_net.update(x_init,formulation=formulation)
    x0 = coupling_ge_net.set_x_init(formulation=formulation)
    return x0

def run_mes_coupling_ge_load_flow(unit='GG'):
    """Steady-state load flow analysis for the coupling network for the gas-electricity"""
    # create network
    coupling_ge_net = create_coupling_ge_network(unit=unit)
    
    # initialize network
    x0 = initialize_coupling_ge_network(coupling_ge_net)
    
    # solve network
    print('\nRunning load flow for gas-electricity network, only coupling, with {}'.format(unit))
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_ge_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
    
    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    
    return coupling_ge_net, x_sol, iters, err_vec

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_coupling_ge_load_flow_GG():
    # Given + When
    coupling_net, x_sol, iters, err_vec = run_mes_coupling_ge_load_flow(unit='GG')
    
    # Then
    x_sol_expected = np.array([qc_ge_GG_sol])
    assert np.allclose(x_sol,x_sol_expected)
    
@pytest.mark.filterwarnings("ignore::UserWarning")
def example_coupling_ge_load_flow_GG_VP():
    # Given + When
    coupling_net, x_sol, iters, err_vec = run_mes_coupling_ge_load_flow(unit='GG_VP')
    
    # Then
    x_sol_expected = np.array([qc_ge_GG_VP_sol])
    assert np.allclose(x_sol,x_sol_expected)
    
def run_mes_ge_load_flow_dd1(error_measure='F',unit='GG'):
    """Steady-state load flow analysis of multi-carrier network, for the first option of domain decomposition, without scaling. The default values are used for initialization.
    
    Uses the already implemented solvers for the single-carrier network. Between single-carrier solves, the boundary values are given to the other domains. The coupling part is solved separately (since the heterogeneous network solver currently assumes qc, Pc, and Qc live on dummy links, not on half links)
    
    Parameters
    ----------
    error_measure : str, optional
        Determines which error measure to use for the outer iterations. Options are 'F', for the system of equations, or 'iters' for the difference between coupling energies between consecutive iterations. Default if 'F'. NB: for 'iters', the vector with errors is one longer, since an error for iteration 0 cannot be defined.
    """ 
    # create networks
    gas_net = create_gas_network()
    elec_net = create_electrical_network()
    coupling_ge_net = create_coupling_ge_network(unit=unit)
    cn = coupling_ge_net.nodes[0]
    hle = cn.half_links[1]
    
    # initialize networks
    xe_new = initialize_electrical_network(elec_net)
    xg_new = initialize_gas_network(gas_net)
    xc_new = initialize_coupling_ge_network(coupling_ge_net)
    xc_full_new = np.concatenate((xc_new,np.array([P_ic,Q_ic])))
    
    # iterate
    print('\nRunning load flow for first option DD, gas-electricity, with coupling {}'.format(unit))
    err_vec = list()
    if error_measure == 'F':
        from meslf.load_flow.system_of_equations import NonLinearSystemGas, NonLinearSystemElectrical, NonLinearSystemHeterogeneous
        gas_net.nodes[1].half_links[0].q = xc_new[0]
        nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'))
        nlsyse = NonLinearSystemElectrical(elec_net)
        nlsysc = NonLinearSystemHeterogeneous(coupling_ge_net,formulation=formulation)
        Fg = nlsysg.F(xg_new)
        Fe = nlsyse.F(xe_new)
        delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.update_full(xe_new)
        Fc = nlsysc.F(xc_new)
        error = np.linalg.norm(np.concatenate((Fg,Fe,Fc)))
        err_vec.append(error)
    elif error_measure == 'iters':
        error = 1 # set some error which is bigger than the tolerance
    iter_nr = 0 # I assume solution is not found at 1 iteration
    errors_gas = dict()
    errors_elec = dict()
    errors_coupling = dict()
    qc_new = xc_new[0]
    Pc_new = xc_full_new[1]
    Qc_new = xc_full_new[2]
    while error > tol and iter_nr < max_iter:
        xe_old = xe_new
        xg_old = xg_new
        xc_old = xc_new
        qc_old = qc_new
        Pc_old = Pc_new
        Qc_old = Qc_new
        # update the domain boundary values, and the rest of the network
        gas_net.nodes[1].half_links[0].q = qc_old
        gas_net.reset_network(xg_old,formulation=formulation.get('gas'))
        xg_new,itersg,err_vecg,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('gas'))
        errors_gas[iter_nr] = err_vecg
        elec_net.reset_network(xe_old)
        xe_new,iterse,err_vece,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR')
        errors_elec[iter_nr] = err_vece
        # update the domain boundary values
        P0c = -S_inj[0].real
        Q0c = -S_inj[0].imag
        hle.P = P0c
        hle.Q = Q0c
        coupling_ge_net.reset_network(xc_old,formulation=formulation)
        xc_new,itersc,err_vec_c,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = coupling_ge_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
        errors_coupling[iter_nr] = err_vec_c
        # new domain boundary values 
        xc_full_new = np.concatenate((xc_new,np.array([P0c,Q0c])))
        qc_new = xc_new[0]
        Pc_new = xc_full_new[1]
        Qc_new = xc_full_new[2]
        if error_measure == 'F':
            gas_net.nodes[1].half_links[0].q = qc_new
            Fg = nlsysg.F(xg_new)
            Fe = nlsyse.F(xe_new)
            Fc = nlsysc.F(xc_new)
            error = np.linalg.norm(np.concatenate((Fg,Fe,Fc)))
        elif error_measure == 'iters':
            error = np.linalg.norm(np.array([qc_new-qc_old,Pc_new-Pc_old,Qc_new-Qc_old]))
        iter_nr += 1
        err_vec.append(error)
        
    # print results
    print('errors coupling = {}'.format(errors_coupling))
    het_net, gas_net, elec_net = create_mes_ge_network()
    het_net.initialize()
    x_new = np.concatenate((xg_new,xe_new,xc_full_new))
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(x_new,formulation=formulation)
    print('errors = {}'.format(err_vec))
    print('Solution after {} iterations (final error = {:.4e}):'.format(iter_nr,err_vec[-1]))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    return xg_new, xe_new, xc_new, x_new, iter_nr, err_vec, errors_gas, errors_elec, errors_coupling

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_coupling_ge_load_flow_GG_dd1_gas():
    # Given + When
    xg_sol, xe_sol, xc_sol, x_sol_full, iter_nr, err_vec, errors_gas, errors_elec, errors_coupling = run_mes_ge_load_flow_dd1(error_measure='F',unit='GG')
    
    # Then
    x_sol_expected = np.array([qc_ge_GG_sol])
    assert np.allclose(xc_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_coupling_ge_load_flow_GG_dd1_full():
    # Given + When
    _, _, _, x_sol_full, _, _, _, _, _ = run_mes_ge_load_flow_dd1(error_measure='F',unit='GG')
    
    # Then
    _, _, _, x_sol_expected, _, _ = run_mes_ge_load_flow(unit='GG')
    assert np.allclose(x_sol_full,x_sol_expected)
    
def create_coupling_gh_network():
    """Create a coupling network consisting of one node, for a gas-heat coupling"""
    c0 = HeterogeneousNode('cn0',node_type=1,x=x0c,y=y0c,unit_type=unit_type_GB,unit_params=unit_params_GB) # To known
    hlh = HeatHalfLink('cn_hlh',c0,link_type='heat_exchanger',link_params={'carrier':water},bc_type=2,Ts=Ts0,dphi=-phic_sol) # Ts and dphi known, source
    hlh.Tr = Trc_sol # Set to solution
    hlg = GasHalfLink('cn0_hlg',c0,-q_ic) # gas flows into coupling node
    coupling_gh_net = HeterogeneousNetwork('coupling network')
    coupling_gh_net.add_node(c0)
    
    return coupling_gh_net

def initialize_coupling_gh_network(coupling_gh_net):
    """Initialize the coupling network consisting of one node, for a gas-heat coupling
    
    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The coupling network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    x_init = np.array([q_ic,m_ic])
    coupling_gh_net.initialize()
    coupling_gh_net.update(x_init,formulation=formulation)
    x0 = coupling_gh_net.set_x_init(formulation=formulation)
    return x0

def run_mes_coupling_gh_load_flow():
    """Steady-state load flow analysis for the coupling network for the gas-heat"""
    # create network
    coupling_gh_net = create_coupling_gh_network()
    
    # initialize network
    x0 = initialize_coupling_gh_network(coupling_gh_net)
    
    # solve network
    print('\nRunning load flow for gas-heat network, only coupling')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_gh_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
    
    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {} C'.format(Ts_hl_vec))
    print('Tr hl = {} C'.format(Tr_hl_vec))
    print('dphi hl = {}'.format(phi_hl_vec))
    print('m c = {}'.format(mc_vec))
    print('phi c = {}'.format(phic_vec))
    print('Ts c = {} C'.format(Tsc_vec))
    print('Tr c = {} C'.format(Trc_vec))
    
    return coupling_gh_net, x_sol, iters, err_vec

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_coupling_gh_load_flow():
    # Given + When
    coupling_net, x_sol, iters, err_vec = run_mes_coupling_gh_load_flow()
    
    # Then
    x_sol_expected = np.array([qc_gh_sol,mc_sol])
    assert np.allclose(x_sol,x_sol_expected)
    
def run_mes_gh_load_flow_dd1(error_measure='F'):
    """Steady-state load flow analysis of multi-carrier network, for the first option of domain decomposition, without scaling. The default values are used for initialization.
    
    Uses the already implemented solvers for the single-carrier network. Between single-carrier solves, the boundary values are given to the other domains. The coupling part is solved separately (since the heterogeneous network solver currently assumes qc, Pc, and Qc live on dummy links, not on half links)
    
    Parameters
    ----------
    error_measure : str, optional
        Determines which error measure to use for the outer iterations. Options are 'F', for the system of equations, or 'iters' for the difference between coupling energies between consecutive iterations. Default if 'F'. NB: for 'iters', the vector with errors is one longer, since an error for iteration 0 cannot be defined.
    """ 
    # create networks
    gas_net = create_gas_network()
    heat_net = create_heat_network()
    coupling_gh_net = create_coupling_gh_network()
    cn = coupling_gh_net.nodes[0]
    hlh = cn.half_links[0]
    
    # initialize networks
    xh_new = initialize_heat_network(heat_net)
    xg_new = initialize_gas_network(gas_net)
    xc_new = initialize_coupling_gh_network(coupling_gh_net)
    xc_full_new = np.concatenate((xc_new,np.array([m_ic,phi_ic,Toc_ic])))
    
    # iterate
    print('\nRunning load flow for first option DD, gas-heat')
    err_vec = list()
    if error_measure == 'F':
        from meslf.load_flow.system_of_equations import NonLinearSystemGas, NonLinearSystemHeat, NonLinearSystemHeterogeneous
        gas_net.nodes[1].half_links[0].q = -xc_new[0]
        nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'))
        nlsysh = NonLinearSystemHeat(heat_net,formulation=formulation.get('heat'))
        nlsysc = NonLinearSystemHeterogeneous(coupling_gh_net,formulation=formulation)
        Fg = nlsysg.F(xg_new)
        Fh = nlsysh.F(xh_new)
        m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(xh_new,formulation=formulation.get('heat'))
        Fc = nlsysc.F(xc_new)
        error = np.linalg.norm(np.concatenate((Fg,Fh,Fc)))
        err_vec.append(error)
    elif error_measure == 'iters':
        error = 1 # set some error which is bigger than the tolerance
    iter_nr = 0 # I assume solution is not found at 1 iteration
    errors_gas = dict()
    errors_heat = dict()
    errors_coupling = dict()
    qc_new = xc_new[0]
    mc_new = xc_full_new[1]
    phic_new = xc_full_new[2]
    Toc_new = xc_full_new[3]
    while error > tol and iter_nr < max_iter:
        xh_old = xh_new
        xg_old = xg_new
        xc_old = xc_new
        qc_old = qc_new
        mc_old = mc_new
        phic_old = phic_new
        Toc_old = Toc_new
        # update the domain boundary values, and the rest of the network
        gas_net.nodes[1].half_links[0].q = qc_old
        gas_net.reset_network(xg_old,formulation=formulation.get('gas'))
        xg_new,itersg,err_vecg,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('gas'))
        errors_gas[iter_nr] = err_vecg
        heat_net.reset_network(xh_old,formulation=formulation.get('heat'))
        xh_new,itersh,err_vech,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('heat'))
        errors_heat[iter_nr] = err_vech
        # update the domain boundary values
        phi0c = -dphi_hl_vec[0][0]
        Tsc = Ts_hl_vec[0][0]
        Trc = Tr_vec[0]
        hlh.dphi = -phi0c
        hlh.Ts = Tsc
        hlh.Tr = Trc
        coupling_gh_net.reset_network(xc_old,formulation=formulation)
        xc_new,itersc,err_vec_c,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = coupling_gh_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
        errors_coupling[iter_nr] = err_vec_c
        # new domain boundary values 
        xc_full_new = np.concatenate((xc_new,np.array([phi0c,Tsc])))
        qc_new = xc_new[0]
        mc_new = xc_full_new[1]
        phic_new = xc_full_new[2]
        Toc_new = xc_full_new[3]
        if error_measure == 'F':
            gas_net.nodes[1].half_links[0].q = qc_new
            Fg = nlsysg.F(xg_new)
            Fh = nlsysh.F(xh_new)
            Fc = nlsysc.F(xc_new)
            error = np.linalg.norm(np.concatenate((Fg,Fh,Fc)))
        elif error_measure == 'iters':
            error = np.linalg.norm(np.array([qc_new-qc_old,mc_new-mc_old,phic_new-phic_old,Toc_new-Toc_old]))
        iter_nr += 1
        err_vec.append(error)
        
    # print results
    het_net, gas_net, heat_net = create_mes_gh_network()
    het_net.initialize()
    x_new = np.concatenate((xg_new,xh_new,xc_full_new))
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(x_new,formulation=formulation)
    print('errors = {}'.format(err_vec))
    print('Solution after {} iterations (final error = {:.4e}):'.format(iter_nr,err_vec[-1]))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {} C'.format(Ts_hl_vec))
    print('Tr hl = {} C'.format(Tr_hl_vec))
    print('dphi hl = {}'.format(phi_hl_vec))
    print('m c = {}'.format(mc_vec))
    print('phi c = {}'.format(phic_vec))
    print('Ts c = {} C'.format(Tsc_vec))
    print('Tr c = {} C'.format(Trc_vec))
    return xg_new, xh_new, xc_new, x_new, iter_nr, err_vec, errors_gas, errors_heat, errors_coupling

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_coupling_gh_load_flow_dd1_xc():
    # Given + When
    xg_sol, xh_sol, xc_sol, x_sol_full, iter_nr, err_vec, errors_gas, errors_heat, errors_coupling = run_mes_gh_load_flow_dd1(error_measure='F')
    
    # Then
    x_sol_expected = np.array([qc_gh_sol,mc_sol])
    assert np.allclose(xc_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_coupling_gh_load_flow_dd1_full():
    # Given + When
    _, _, _, x_sol_full, _, _, _, _, _ = run_mes_gh_load_flow_dd1(error_measure='F')
    
    # Then
    _, _, _, x_sol_expected, _, _ = run_mes_gh_load_flow()
    assert np.allclose(x_sol_full,x_sol_expected)
    
def create_coupling_geh_network():
    """Create a coupling network consisting of one node, for a gas-electricity-heat coupling"""
    c0 = HeterogeneousNode('cn0',node_type=1,x=x0c,y=y0c,unit_type=unit_type_CHP,unit_params=unit_params_CHP) # To known
    hlh = HeatHalfLink('cn_hlh',c0,link_type='heat_exchanger',link_params={'carrier':water},bc_type=2,Ts=Ts0,dphi=-phic_sol) # Ts and dphi known, source
    hlh.Tr = Trc_sol # Set to solution
    hlg = GasHalfLink('cn0_hlg',c0,-q_ic,bc_type=0) # gas flows into coupling node, q is unknown
    hle = ElectricalHalfLink('cn0_hle',c0,P=Pc_sol,Q=Qc_sol,bc_type=1) # P and Q are known
    coupling_geh_net = HeterogeneousNetwork('coupling network')
    coupling_geh_net.add_node(c0)
    
    return coupling_geh_net

def initialize_coupling_geh_network(coupling_geh_net):
    """Initialize the coupling network consisting of one node, for a gas-electricity-heat coupling
    
    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The coupling network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    x_init = np.array([q_ic,m_ic])
    coupling_geh_net.initialize()
    coupling_geh_net.update(x_init,formulation=formulation)
    x0 = coupling_geh_net.set_x_init(formulation=formulation)
    return x0

def run_mes_coupling_geh_load_flow():
    """Steady-state load flow analysis for the coupling network for the gas-electricity-heat"""
    # create network
    coupling_geh_net = create_coupling_geh_network()
    
    # initialize network
    x0 = initialize_coupling_geh_network(coupling_geh_net)
    
    # solve network
    print('\nRunning load flow for gas-heat network, only coupling')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_geh_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
    
    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('p heat = {} m'.format(p_h_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {} C'.format(Ts_hl_vec))
    print('Tr hl = {} C'.format(Tr_hl_vec))
    print('dphi hl = {}'.format(phi_hl_vec))
    print('m c = {}'.format(mc_vec))
    print('phi c = {}'.format(phic_vec))
    print('Ts c = {} C'.format(Tsc_vec))
    print('Tr c = {} C'.format(Trc_vec))
    
    return coupling_geh_net, x_sol, iters, err_vec

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_coupling_geh_load_flow():
    # Given + When
    coupling_net, x_sol, iters, err_vec = run_mes_coupling_geh_load_flow()
    
    # Then
    x_sol_expected = np.array([qc_geh_sol,mc_sol])
    assert np.allclose(x_sol,x_sol_expected)
    
def run_mes_geh_load_flow_dd1(error_measure='F'):
    """Steady-state load flow analysis of multi-carrier network, for the first option of domain decomposition, without scaling. The default values are used for initialization.
    
    Uses the already implemented solvers for the single-carrier network. Between single-carrier solves, the boundary values are given to the other domains. The coupling part is solved separately (since the heterogeneous network solver currently assumes qc, Pc, and Qc live on dummy links, not on half links)
    
    Parameters
    ----------
    error_measure : str, optional
        Determines which error measure to use for the outer iterations. Options are 'F', for the system of equations, or 'iters' for the difference between coupling energies between consecutive iterations. Default if 'F'. NB: for 'iters', the vector with errors is one longer, since an error for iteration 0 cannot be defined.
    """ 
    # create networks
    gas_net = create_gas_network()
    heat_net = create_heat_network()
    elec_net = create_electrical_network()
    coupling_geh_net = create_coupling_geh_network()
    cn = coupling_geh_net.nodes[0]
    hlh = cn.half_links[0]
    hle = cn.half_links[2]
    
    # initialize networks
    xh_new = initialize_heat_network(heat_net)
    xe_new = initialize_electrical_network(elec_net)
    xg_new = initialize_gas_network(gas_net)
    xc_new = initialize_coupling_geh_network(coupling_geh_net)
    xc_full_new = np.concatenate((xc_new,np.array([P_ic,Q_ic]),np.array([m_ic,phi_ic,Toc_ic])))
    
    # iterate
    print('\nRunning load flow for first option DD, gas-electricity-heat')
    err_vec = list()
    if error_measure == 'F':
        from meslf.load_flow.system_of_equations import NonLinearSystemGas, NonLinearSystemElectrical, NonLinearSystemHeat, NonLinearSystemHeterogeneous
        gas_net.nodes[1].half_links[0].q = -xc_new[0]
        nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'))
        nlsyse = NonLinearSystemElectrical(elec_net)
        nlsysh = NonLinearSystemHeat(heat_net,formulation=formulation.get('heat'))
        nlsysc = NonLinearSystemHeterogeneous(coupling_geh_net,formulation=formulation)
        Fg = nlsysg.F(xg_new)
        Fe = nlsyse.F(xe_new)
        delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.update_full(xe_new)
        Fh = nlsysh.F(xh_new)
        m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(xh_new,formulation=formulation.get('heat'))
        Fc = nlsysc.F(xc_new)
        error = np.linalg.norm(np.concatenate((Fg,Fe,Fh,Fc)))
        err_vec.append(error)
    elif error_measure == 'iters':
        error = 1 # set some error which is bigger than the tolerance
    iter_nr = 0 # I assume solution is not found at 1 iteration
    errors_gas = dict()
    errors_elec = dict()
    errors_heat = dict()
    errors_coupling = dict()
    qc_new = xc_new[0]
    Pc_new = xc_full_new[1]
    Qc_new = xc_full_new[2]
    mc_new = xc_full_new[3]
    phic_new = xc_full_new[4]
    Toc_new = xc_full_new[5]
    while error > tol and iter_nr < max_iter:
        xh_old = xh_new
        xe_old = xe_new
        xg_old = xg_new
        xc_old = xc_new
        qc_old = qc_new
        Pc_old = Pc_new
        Qc_old = Qc_new
        mc_old = mc_new
        phic_old = phic_new
        Toc_old = Toc_new
        # update the domain boundary values, and the rest of the network
        gas_net.nodes[1].half_links[0].q = qc_old
        gas_net.reset_network(xg_old,formulation=formulation.get('gas'))
        xg_new,itersg,err_vecg,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('gas'))
        errors_gas[iter_nr] = err_vecg
        elec_net.reset_network(xe_old)
        xe_new,iterse,err_vece,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR')
        errors_elec[iter_nr] = err_vece
        heat_net.reset_network(xh_old,formulation=formulation.get('heat'))
        xh_new,itersh,err_vech,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('heat'))
        errors_heat[iter_nr] = err_vech
        # update the domain boundary values
        P0c = -S_inj[0].real
        Q0c = -S_inj[0].imag
        hle.P = P0c
        hle.Q = Q0c
        phi0c = -dphi_hl_vec[0][0]
        Tsc = Ts_hl_vec[0][0]
        Trc = Tr_vec[0]
        hlh.dphi = -phi0c
        hlh.Ts = Tsc
        hlh.Tr = Trc
        coupling_geh_net.reset_network(xc_old,formulation=formulation)
        xc_new,itersc,err_vec_c,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = coupling_geh_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
        errors_coupling[iter_nr] = err_vec_c
        # new domain boundary values 
        xc_full_new = np.concatenate((np.array([xc_new[0]]),np.array([P0c,Q0c]),np.array([xc_new[1]]),np.array([phi0c,Tsc])))
        qc_new = xc_new[0]
        Pc_new = xc_full_new[1]
        Qc_new = xc_full_new[2]
        mc_new = xc_full_new[3]
        phic_new = xc_full_new[4]
        Toc_new = xc_full_new[5]
        if error_measure == 'F':
            gas_net.nodes[1].half_links[0].q = qc_new
            Fg = nlsysg.F(xg_new)
            Fe = nlsyse.F(xe_new)
            Fh = nlsysh.F(xh_new)
            Fc = nlsysc.F(xc_new)
            error = np.linalg.norm(np.concatenate((Fg,Fe,Fh,Fc)))
        elif error_measure == 'iters':
            error = np.linalg.norm(np.array([qc_new-qc_old,Pc_new-Pc_old,Qc_new-Qc_old,mc_new-mc_old,phic_new-phic_old,Toc_new-Toc_old]))
        iter_nr += 1
        err_vec.append(error)
        
    # print results
    het_net, gas_net, elec_net, heat_net = create_mes_geh_network()
    het_net.initialize()
    x_new = np.concatenate((xg_new,xe_new,xh_new,xc_full_new))
    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(x_new,formulation=formulation)
    print('errors = {}'.format(err_vec))
    print('Solution after {} iterations (final error = {:.4e}):'.format(iter_nr,err_vec[-1]))
    print('p = {} mbar'.format(p_g_vec/mbar))
    print('q = {} kg/s'.format(q_vec))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
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
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {} C'.format(Ts_hl_vec))
    print('Tr hl = {} C'.format(Tr_hl_vec))
    print('dphi hl = {}'.format(phi_hl_vec))
    print('m c = {}'.format(mc_vec))
    print('phi c = {}'.format(phic_vec))
    print('Ts c = {} C'.format(Tsc_vec))
    print('Tr c = {} C'.format(Trc_vec))
    return xg_new, xe_new, xh_new, xc_new, x_new, iter_nr, err_vec, errors_gas, errors_elec, errors_heat, errors_coupling

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_coupling_geh_load_flow_dd1():
    # Given + When
    xg_sol, xe_sol, xh_sol, xc_sol, x_sol_full, iter_nr, err_vec, errors_gas, errors_elec, errors_heat, errors_coupling = run_mes_geh_load_flow_dd1(error_measure='F')
    
    # Then
    x_sol_expected = np.array([qc_geh_sol,mc_sol])
    assert np.allclose(xc_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_coupling_geh_load_flow_dd1_full():
    # Given + When
    _, _, _, _, x_sol_full, _, _, _, _, _, _ = run_mes_geh_load_flow_dd1(error_measure='F')
    
    # Then
    _, _, _, _, x_sol_expected, _, _ = run_mes_geh_load_flow()
    assert np.allclose(x_sol_full,x_sol_expected)
    
def inner_iters(ax,err_vec,errors_gas=dict(),errors_elec=dict(),errors_heat=dict(),errors_coupling=dict(),error_measure='F'):
    """Plot the convergence per inner iterations"""
    iter_gas = 0
    iter_elec = 0
    iter_heat = 0
    iter_coupling = 0
    if err_vec[-1] == 0:
        err_vec[-1] = 1e-16 # semilogy doesn't like an actual 0
    if error_measure == 'F':
        iter_outer = [0]
        marker = '.'
        total_iters = len(err_vec)-1
    elif error_measure == 'dE':
        iter_outer = list()
        marker = '*'
        total_iters = len(err_vec)
    else:
        raise ValueError('Enter valid error_measure')
    label = 'DD inner'
    for iter_nr in range(total_iters):
        if errors_gas.get(iter_nr):
            err_vec_gas = errors_gas.get(iter_nr)
            if iter_nr == 0:
                ax.semilogy([k_gas+iter_gas for k_gas in range(len(err_vec_gas))],err_vec_gas,marker=marker,color='tab:green',linestyle='--',label=label+' gas, '+error_measure)
            else:
                ax.semilogy([k_gas+iter_gas for k_gas in range(len(err_vec_gas))],err_vec_gas,marker=marker,color='tab:green',linestyle='--')
        else:
            err_vec_gas = list()
            
        if errors_elec.get(iter_nr):
            err_vec_elec = errors_elec.get(iter_nr)
            iter_elec += len(err_vec_gas)
            if errors_gas:
                iter_elec -= 1
            if iter_nr == 0:
                ax.semilogy([k_elec+iter_elec for k_elec in range(len(err_vec_elec))],err_vec_elec,marker=marker,color='tab:red',linestyle='--',label=label+' elec, '+error_measure)
            else:
                ax.semilogy([k_elec+iter_elec for k_elec in range(len(err_vec_elec))],err_vec_elec,marker=marker,color='tab:red',linestyle='--')
        else:
            err_vec_elec = list()
        
        if errors_heat.get(iter_nr):
            err_vec_heat = errors_heat.get(iter_nr)
            iter_heat += len(err_vec_gas) + len(err_vec_elec)
            if errors_gas:
                iter_heat -= 1
            if errors_elec:
                iter_heat -= 1
            if iter_nr == 0:
                ax.semilogy([k_heat+iter_heat for k_heat in range(len(err_vec_heat))],err_vec_heat,marker=marker,color='tab:blue',linestyle='--',label=label+' heat, '+error_measure)
            else:
                ax.semilogy([k_heat+iter_heat for k_heat in range(len(err_vec_heat))],err_vec_heat,marker=marker,color='tab:blue',linestyle='--')
        else:
            err_vec_heat = list()
        
        if errors_coupling.get(iter_nr):
            err_vec_coupling = errors_coupling.get(iter_nr)
            if err_vec_coupling[-1] == 0:
                err_vec_coupling[-1] = 1e-16 # semilogy doesn't like an actual 0
            iter_coupling += len(err_vec_gas) + len(err_vec_elec) + len(err_vec_heat)
            if errors_gas:
                iter_coupling -= 1
            if errors_elec:
                iter_coupling -= 1
            if errors_heat:
                iter_coupling -= 1
            if iter_nr == 0:
                ax.semilogy([k_coupling+iter_coupling for k_coupling in range(len(err_vec_coupling))],err_vec_coupling,marker=marker,color='tab:gray',linestyle='--',label=label+' coupling, '+error_measure)
            else:
                ax.semilogy([k_coupling+iter_coupling for k_coupling in range(len(err_vec_coupling))],err_vec_coupling,marker=marker,color='tab:gray',linestyle='--')
        else:
            err_vec_coupling = list()
                
        iter_gas += len(err_vec_gas) + len(err_vec_elec) + len(err_vec_heat) + len(err_vec_coupling)
        iter_elec += len(err_vec_elec) + len(err_vec_heat) + len(err_vec_coupling)
        iter_heat += len(err_vec_heat) + len(err_vec_coupling)
        iter_coupling += len(err_vec_coupling)
        if errors_gas:
            iter_gas -= 1
        if errors_elec:
            iter_gas -= 1
            iter_elec -= 1
        if errors_heat:
            iter_gas -= 1
            iter_elec -= 1
            iter_heat -= 1
        if errors_coupling:
            iter_gas -= 1
            iter_elec -= 1
            iter_heat -= 1
            iter_coupling -= 1
        iter_outer.append(np.max([iter_gas,iter_elec,iter_heat,iter_coupling]))
    ax.semilogy(iter_outer,err_vec,marker=marker,color='tab:orange',label=label+', '+error_measure)
    return iter_gas, iter_elec, iter_heat, iter_coupling, iter_outer
    
if __name__== '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "No single-carrier subnetworks found",UserWarning)
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        run_gas_load_flow()
        run_electrical_load_flow()
        run_heat_load_flow()
        # gas - electricity coupling
        het_net_ge, gas_net_ge, elec_net_ge, x_sol_ge, iters_ge, err_vec_ge = run_mes_ge_load_flow()
        xg_new_errF_ge, xe_new_errF_ge, xc_new_errF_ge, x_new_errF_ge, iters_dd1_errF_ge, err_vec_dd1_errF_ge, errors_gas_errF_ge, errors_elec_errF_ge, errors_coupling_errF_ge = run_mes_ge_load_flow_dd1(error_measure='F')
        xg_new_errdE_ge, xe_new_errdE_ge, xc_new_errdE_ge, x_new_errdE_ge, iters_dd1_errdE_ge, err_vec_dd1_errdE_ge, errors_gas_errdE_ge, errors_elec_errdE_ge, errors_coupling_errdE_ge = run_mes_ge_load_flow_dd1(error_measure='iters')
        # gas - heat coupling
        het_net_gh, gas_net_gh, heat_net_gh, x_sol_gh, iters_gh, err_vec_gh = run_mes_gh_load_flow()
        xg_new_errF_gh, xh_new_errF_gh, xc_new_errF_gh, x_new_errF_gh, iters_dd1_errF_gh, err_vec_dd1_errF_gh, errors_gas_errF_gh, errors_heat_errF_gh, errors_coupling_errF_gh = run_mes_gh_load_flow_dd1(error_measure='F')
        xg_new_errdE_gh, xh_new_errdE_gh, xc_new_errdE_gh, x_new_errdE_gh, iters_dd1_errdE_gh, err_vec_dd1_errdE_gh, errors_gas_errdE_gh, errors_heat_errdE_gh, errors_coupling_errdE_gh = run_mes_gh_load_flow_dd1(error_measure='iters')
        # gas - electricity - heat coupling 
        het_net_geh, gas_net_geh, elec_net_geh, heat_net_geh, x_sol_geh, iters_geh, err_vec_geh = run_mes_geh_load_flow()
        xg_new_errF_geh, xe_new_errF_geh, xh_new_errF_geh, xc_new_errF_geh, x_new_errF_geh, iters_dd1_errF_geh, err_vec_dd1_errF_geh, errors_gas_errF_geh, errors_elec_errF_geh, errors_heat_errF_geh, errors_coupling_errF_geh = run_mes_geh_load_flow_dd1(error_measure='F')
        xg_new_errdE_geh, xe_new_errdE_geh, xh_new_errdE_geh, xc_new_errdE_geh, x_new_errdE_geh, iters_dd1_errdE_geh, err_vec_dd1_errdE_geh, errors_gas_errdE_geh, errors_elec_errdE_geh, errors_heat_errdE_geh, errors_coupling_errdE_geh = run_mes_geh_load_flow_dd1(error_measure='iters')
        # gas - electricity coupling, GG with valve point effect (is nonlinear)
        het_net_ge_nl, gas_net_ge_nl, elec_net_ge_nl, x_sol_ge_nl, iters_ge_nl, err_vec_ge_nl = run_mes_ge_load_flow(unit='GG_VP')
        xg_new_errF_ge_nl, xe_new_errF_ge_nl, xc_new_errF_ge_nl, x_new_errF_ge_nl, iters_dd1_errF_ge_nl, err_vec_dd1_errF_ge_nl, errors_gas_errF_ge_nl, errors_elec_errF_ge_nl, errors_coupling_errF_ge_nl = run_mes_ge_load_flow_dd1(error_measure='F',unit='GG_VP')
        xg_new_errdE_ge_nl, xe_new_errdE_ge_nl, xc_new_errdE_ge_nl, x_new_errdE_ge_nl, iters_dd1_errdE_ge_nl, err_vec_dd1_errdE_ge_nl, errors_gas_errdE_ge_nl, errors_elec_errdE_ge_nl, errors_coupling_errdE_ge_nl = run_mes_ge_load_flow_dd1(error_measure='iters',unit='GG_VP')
        
    # plot convergence
    print('\nCreating convergence plot gas-electricity')
    fig = plt.figure(r'Convergence plot gas-electricity, error $\Delta E$')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||\Delta E||_2$')
    ax.semilogy(err_vec_ge,marker='x',color='tab:orange',label='Total system')
    iter_gas_errdE_ge, iter_elec_errdE_ge, iter_heat_errdE_ge, iter_coupling_errdE_ge, iter_outer_dd1_errdE_ge = inner_iters(ax,err_vec_dd1_errdE_ge,errors_gas=errors_gas_errdE_ge,errors_elec=errors_elec_errdE_ge,errors_coupling=errors_coupling_errdE_ge,error_measure='dE')
    max_iters_used = np.max([iters_ge,iters_dd1_errdE_ge,iter_gas_errdE_ge,iter_elec_errdE_ge])
    
    fig = plt.figure(r'Convergence plot gas-electricity, error $F$')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F||_2')
    ax.semilogy(err_vec_ge,marker='x',color='tab:orange',label='Total system')
    iter_gas_errF_ge, iter_elec_errF_ge, iter_heat_errF_ge, iter_coupling_errF_ge, iter_outer_dd1_errF_ge = inner_iters(ax,err_vec_dd1_errF_ge,errors_gas=errors_gas_errF_ge,errors_elec=errors_elec_errF_ge,errors_coupling=errors_coupling_errF_ge,error_measure='F')
    max_iters_used = np.max([max_iters_used,iters_ge,iters_dd1_errF_ge,iter_gas_errF_ge,iter_elec_errF_ge])
    
    print('\nCreating convergence plot gas-electricity VP')
    fig = plt.figure(r'Convergence plot gas-electricity VP, error $\Delta E$')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||\Delta E||_2$')
    ax.semilogy(err_vec_ge_nl,marker='x',color='tab:orange',label='Total system')
    iter_gas_errdE_ge_nl, iter_elec_errdE_ge_nl, iter_heat_errdE_ge_nl, iter_coupling_errdE_ge_nl, iter_outer_dd1_errdE_ge_nl = inner_iters(ax,err_vec_dd1_errdE_ge_nl,errors_gas=errors_gas_errdE_ge_nl,errors_elec=errors_elec_errdE_ge_nl,errors_coupling=errors_coupling_errdE_ge_nl,error_measure='dE')
    max_iters_used = np.max([max_iters_used,iters_ge_nl,iters_dd1_errdE_ge_nl,iter_gas_errdE_ge_nl,iter_elec_errdE_ge_nl,iter_elec_errdE_ge_nl])
    
    fig = plt.figure(r'Convergence plot gas-electricity VP, error $F$')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F||_2$')
    ax.semilogy(err_vec_ge_nl,marker='x',color='tab:orange',label='Total system')
    iter_gas_errF_ge_nl, iter_elec_errF_ge_nl, iter_heat_errF_ge_nl, iter_coupling_errF_ge_nl, iter_outer_dd1_errF_ge_nl = inner_iters(ax,err_vec_dd1_errF_ge_nl,errors_gas=errors_gas_errF_ge_nl,errors_elec=errors_elec_errF_ge_nl,errors_coupling=errors_coupling_errF_ge_nl,error_measure='F')
    max_iters_used = np.max([max_iters_used,iters_ge_nl,iters_dd1_errF_ge_nl,iter_gas_errF_ge_nl,iter_elec_errF_ge_nl,iter_coupling_errF_ge_nl])
    
    print('\nCreating convergence plot gas-heat')
    fig = plt.figure(r'Convergence plot gas-heat, error $\Delta E$')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||\Delta E||_2$')
    ax.semilogy(err_vec_gh,marker='x',color='tab:orange',label='Total system')
    iter_gas_errdE_gh, iter_elec_errdE_gh, iter_heat_errdE_gh, iter_coupling_errdE_gh, iter_outer_dd1_errdE_gh = inner_iters(ax,err_vec_dd1_errdE_gh,errors_gas=errors_gas_errdE_gh,errors_heat=errors_heat_errdE_gh,errors_coupling=errors_coupling_errdE_gh,error_measure='dE')
    max_iters_used = np.max([max_iters_used,iters_gh,iters_dd1_errdE_gh,iter_gas_errdE_gh,iter_heat_errdE_gh,iter_coupling_errdE_gh])
    
    print('\nCreating convergence plot gas-heat')
    fig = plt.figure(r'Convergence plot gas-heat, error $F$')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F||_2$')
    ax.semilogy(err_vec_gh,marker='x',color='tab:orange',label='Total system')
    iter_gas_errF_gh, iter_elec_errF_gh, iter_heat_errF_gh, iter_coupling_errF_gh, iter_outer_dd1_errF_gh = inner_iters(ax,err_vec_dd1_errF_gh,errors_gas=errors_gas_errF_gh,errors_heat=errors_heat_errF_gh,errors_coupling=errors_coupling_errF_gh,error_measure='F')
    max_iters_used = np.max([max_iters_used,iters_gh,iters_dd1_errdE_gh,iter_gas_errdE_gh,iter_heat_errdE_gh,iter_coupling_errdE_gh])
    
    print('\nCreating convergence plot gas-electricity-heat')
    fig = plt.figure(r'Convergence plot gas-electricity-heat, error $\Delta E$')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||\Delta E||_2$')
    ax.semilogy(err_vec_geh,marker='x',color='tab:orange',label='Total system')
    iter_gas_errdE_geh, iter_elec_errdE_geh, iter_heat_errdE_geh, iter_coupling_errdE_geh, iter_outer_dd1_errdE_geh = inner_iters(ax,err_vec_dd1_errdE_geh,errors_gas=errors_gas_errdE_geh,errors_elec=errors_elec_errdE_geh,errors_heat=errors_heat_errdE_geh,errors_coupling=errors_coupling_errdE_geh,error_measure='dE')
    max_iters_used = np.max([max_iters_used,iters_geh,iters_dd1_errdE_geh,iter_gas_errdE_geh,iter_heat_errdE_geh,iter_coupling_errdE_geh])
    
    print('\nCreating convergence plot gas-electricity-heat')
    fig = plt.figure(r'Convergence plot gas-electricity-heat, error $F$')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F||_2$')
    ax.semilogy(err_vec_geh,marker='x',color='tab:orange',label='Total system')
    iter_gas_errF_geh, iter_elec_errF_geh, iter_heat_errF_geh, iter_coupling_errF_geh, iter_outer_dd1_errF_geh = inner_iters(ax,err_vec_dd1_errF_geh,errors_gas=errors_gas_errF_geh,errors_elec=errors_elec_errF_geh,errors_heat=errors_heat_errF_geh,errors_coupling=errors_coupling_errF_geh,error_measure='F')
    max_iters_used = np.max([iters_geh,iters_dd1_errF_geh,iter_gas_errF_geh,iter_heat_errF_geh,iter_coupling_errF_geh])
    
    for i in plt.get_fignums():
        fig = plt.figure(i)
        ax = fig.gca()
        ax.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
        xmin = 0
        xmax = max_iters_used
        xticks = range(xmin,xmax+1) # make sure the xticks are integers
        ax.legend()
        ax.grid(which='major',color='k', linestyle='--', alpha=.2)
        ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
        ax.set_xlim(left=xmin,right=xmax+1)
        ax.set_xticks(xticks)
        
    plt.show()
    
    
    
