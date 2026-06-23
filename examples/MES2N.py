"""
MES consisting of 2 nodes per carrier (i.e. only one link in every single-carrier network), with multiple couplings.
"""
import warnings
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.carrier import Gas
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.utils.constants import bar, mbar, kV, MW
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
import scipy.sparse as sps
from meslf.utils.hide_print import HiddenPrints
import os

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
y0g = 1
x1g = 0
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
x0e = 2
y0e = 1
x1e = 2
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
heat_link_type_extra = 'isolated_pipe_low_pres_pole'
heat_link_params_extra = {'L':L_h,'D':D_h,'carrier':water}
# coordinates
x0h = 3
y0h = 0
x1h = 3
y1h = -1

# physical parameters of the coupling unit
eta_GG0 = .6
eta_GG1 = .7
GHV = 60134305#[J/kg]
unit_type_GG = 'ge_gas_fired_gen'
unit_params_GG0={'eta':eta_GG0,'GHV':GHV}
unit_params_GG1={'eta':eta_GG1,'GHV':GHV}
eta_GB0 = .75
eta_GB1 = .8
eta_CHP0 = np.array([eta_GG0,eta_GB0])
eta_CHP1 = np.array([eta_GG1,eta_GB1])
unit_type_CHP = 'geh_CHP'
unit_params_CHP0 = {'eta':eta_CHP0,'GHV':GHV}
unit_params_CHP1 = {'eta':eta_CHP1,'GHV':GHV}
# coordinates
x0c = 1
y0c = 1
x1c = 1
y1c = 0

# solution
Pc0_sol = 1*Sbase #[W]
Qc0_sol = .5*Sbase #[W]
Pc1_sol = 3.51427623481958*MW #[W]
Qc1_sol = 2.142762348195836*MW #[W]
phic0_sol = 3.014962931574398*MW#[W]
phic1_sol = 1.5*MW #[W]
qc0_sol = Pc0_sol/(GHV*eta_GG0) #[kg/s]
qc1_sol = Pc1_sol/(GHV*eta_GG1) #[kg/s]
qc0_sol_CHP = (Pc0_sol/eta_CHP0[0] + phic0_sol/eta_CHP0[1])/GHV #[kg/s]
qc1_sol_CHP = (Pc1_sol/eta_CHP1[0] + phic1_sol/eta_CHP1[1])/GHV #[kg/s]
V0_sol = 0.93108103*Vbase #[V]
ph1_sol = 99.64192067*rho_w*grav_const #[Pa]
m01_sol = 14.394908119159574 - 9.564801530368245
Ts1_sol = 99.50616179
Tr0_sol = 49.7530809
Tr1_sol = 50.
# boundary conditions
pg0 = 50*mbar
q1_load = .01 #[kg/s]
V1 = 1.*Vbase #[V]
delta1 = 0 #[rad]
P0_load = 2*Sbase #[W]
Q0_load = 1*Sbase #[W]
P1_load = 2.5*Sbase #[W]
Q1_load = 1.5*Sbase #[W]
P0_gen = Pc0_sol
Q0_gen = Qc0_sol
q0_source = -(q1_load + qc0_sol + qc1_sol) #[kg/s]
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
#Toc1 = psi01*Ts0 #99 #[C]
m1_sink = phi1_sink/(Cp_w*(Ts1_sol-To1_sink))
mc1_sol = m1_sink - m01_sol
Toc1 = phic1_sol/(Cp_w*mc1_sol)+Tr1_sol
To1_source = Toc1
m0_sink = phi0_sink/(Cp_w*(Ts0 - To0_sink))
mc0_sol = phic0_sol/(Cp_w*(Toc0-Tr0_sol))
#mc1_sol = phic1_sol/(Cp_w*(Toc1-Tr1_sol))

# initial conditions
q_ic = 2*q1_load
pg_ic = .99*pg0
qc_ic = 1.5*q1_load
Pc_ic = 1.5*MW
Qc_ic = 2*MW
phi_ic = 1*MW
m_ic = 5
m_hl_ic = m_ic
ph_ic = .99*ph0
Ts_ic = Ts0
Tr_ic = To1_sink
To_ic = Ts0

# solver information
tol=1e-6 # overall / global tolerance.
tol_sub=tol*1e-1 # tolerance of the subnetworks of DD
max_iter_outer=250
formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}
pgbase=50*mbar
qbase=1e-2
deltabase=1
mbase=1
phbase=100*rho_w*grav_const #[Pa]
Tbase=100
phibase=1*MW
Egbase=1*MW
scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'phbase':phbase,'mbase':mbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}

# for plotting
art_zero = 1e-16

def create_gas_network():
    """Create a gas network consisting of one source connected to one sink"""
    g0 = GasNode('gn0',node_type=0,x=x0g,y=y0g,p=pg0) # reference node
    g1 = GasNode('gn1',node_type=1,x=x1g,y=y1g,q=qc1_sol) # load node
    GasHalfLink('gn1_load',g1,q1_load,bc_type=1) # gas sink, q known

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
    p_init = np.array([pg_ic])

    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    gas_net.update(x_init,formulation=formulation.get('gas'))
    x0 = gas_net.set_x_init(formulation=formulation.get('gas'))
    return x0

def run_gas_load_flow(max_iter=max_iter_outer,scale_var=None):
    """Steady-state load flow analysis of gas network, without scaling. The default values are used for initialization.
    """
    # create network
    gas_net = create_gas_network()
    # initialize
    x0 = initialize_gas_network(gas_net)

    # solve network
    print('\nRunning load flow for single-carrier gas network')
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_sol/mbar))
    print('q = {} kg/s'.format(q_sol))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    print('q0 source = {} kg/s'.format(gas_net.nodes[0].half_links[0].q - qc0_sol)) #negative

    return gas_net,x_sol,iters,err_vec

def create_electrical_network():
    """Create an electrical network consisting of one source connected to one sink"""
    e0 = ElectricalNode('en0',node_type=1,x=x0e,y=y0e,P=-P0_gen,V=V0_sol) # gen (-P0_gen, since P0_gen > 0, but is should be in input)
    ElectricalHalfLink('en0_load',e0,P=P0_load,Q=Q0_load,bc_type=1) # P and Q are known
    e1 = ElectricalNode('en1',x=x1e,y=y1e,node_type=0,V=V1,delta=delta1) # slack

    el0 = ElectricalLink('el0',e0,e1,link_type=elec_link_type,link_params=elec_link_params)

    elec_net = ElectricalNetwork('2 nodes')
    elec_net.add_link(el0)

    return elec_net

def update_electrical_network_bc_eh(elec_net,xc):
    """Update the boundary values of the electricity network, based on the outcome xc of the coupling electricity-heat network"""
    P0c = xc[0] # >0
    elec_net.nodes[0].half_links[0].P = -P0c #generator, so source, so <0

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

def run_electrical_load_flow(max_iter=max_iter_outer,scale_var=None):
    """Steady-state load flow analysis of electrical network, without scaling. The default values are used for initialization.
    """
    # create network
    elec_net = create_electrical_network()

    x0 = initialize_electrical_network(elec_net)

    # solve network
    print('\nRunning load flow for single-carrier electrical network')
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)

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
    print('P1 source = {} MW'.format(elec_net.nodes[1].half_links[0].P/MW - P1_load/MW)) # is negative
    print('Q1 source = {} MW'.format(elec_net.nodes[1].half_links[0].Q/MW - Q1_load/MW)) # is negative
    return elec_net,x_sol,iters,err_vec

def create_heat_network(c_hl=False,dummy_links=True):
    """Create a heat network consisting of one source connected to one sink"""
    if c_hl:
        phi1 = phi1_sink
        To1 = To1_sink
    else:
        phi1 = Cp_w*m01_sol*(Ts1_sol-To1_sink) # match Ts, Tr, p, and m to the network with the explicit coupling
        #print('phi1 = {} MW'.format(phi1/MW))
        #print('m01 = {}'.format(m01_sol))
        #print('Ts1_sol = {}'.format(Ts1_sol))
        #print('Ts1 alt = {}'.format(phi1/(Cp_w*m01_sol) + To1_sink))
        To1 = To1_sink

    h0 = HeatNode('hn0',node_type=0,x=x0h,y=y0h,p=ph0,Ts=Ts0) # slack
    h0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h1 = HeatNode('hn1',node_type=1,x=x1h,y=y1h,Tr_hl=To1,dphi=phi1) # load node (sink)
    h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    hl0 = HeatLink('hl0',h0,h1,link_type = heat_link_type,link_params = heat_link_params)
    heat_net = HeatNetwork('2 nodes',Ta=Ta)
    heat_net.add_link(hl0)

    if c_hl:
        if dummy_links:
            h0_extra = HeatNode('hn0_extra',node_type=3,x=x0h+.5,y=y0h+.5,Tr_hl=To0_sink,dphi=phi0_sink,p=ph0) # ref. load node (sink) (since only connected with a dummy link, so no way to determine pressure)
            h1_extra = HeatNode('hn1_extra',node_type=3,x=x1h-.5,y=y1h-.5,Ts_hl=To1_source,dphi=-phi1_source,p=ph1_sol) # ref. load node (source) (since only connected with a dummy link, so no way to determine pressure)
            hl1 = HeatLink('hl1',h0,h0_extra,link_type = 'dummy',link_params = {'carrier':water})
            hl2 = HeatLink('hl2',h1_extra,h1,link_type = 'dummy',link_params = {'carrier':water})
        else:
            h0_extra = HeatNode('hn0_extra',node_type=1,x=x0h+.5,y=y0h+.5,Tr_hl=To0_sink,dphi=phi0_sink) # ref. load node (sink)
            h1_extra = HeatNode('hn1_extra',node_type=1,x=x1h-.5,y=y1h-.5,Ts_hl=To1_source,dphi=-phi1_source) # ref. load node (source)
            hl1 = HeatLink('hl1',h0,h0_extra,link_type = heat_link_type_extra,link_params = heat_link_params_extra)
            hl2 = HeatLink('hl2',h1_extra,h1,link_type = heat_link_type_extra,link_params = heat_link_params_extra)
        h0_extra.half_links[0].set_type('heat_exchanger',{'carrier':water})
        h1_extra.half_links[0].set_type('heat_exchanger',{'carrier':water})
        heat_net.add_link(hl1)
        heat_net.add_link(hl2)

    return heat_net

def update_heat_network_bc(heat_net,xc,c_hl=False,dummy_links=True):
    """Update the boundary values of the heat network, based on the outcome xc of the coupling network"""
    dphic1 = -xc[1] #<0
    if c_hl:
        halflink1_extra = heat_net.nodes[3].half_links[0]
        halflink1_extra.dphi = dphic1
    else:
        raise NotImplementedError("Updating heat network BC's from coupling data not implemented")
    return heat_net

def update_heat_network_bc_Toc1_unknown(heat_net,xc,c_hl=False,dummy_links=True):
    """Update the boundary values of the heat network, based on the outcome xc of the coupling network"""
    dphic1 = -xc[1] #<0
    Ts1c = xc[2]
    if c_hl:
        halflink1_extra = heat_net.nodes[3].half_links[0]
        halflink1_extra.Ts = Ts1c
        halflink1_extra.dphi = dphic1
    else:
        raise NotImplementedError("Updating heat network BC's from coupling data not implemented")
    return heat_net

def initialize_heat_network(heat_net,c_hl=False,dummy_links=True,formulation='half_link_flow'):
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
    if formulation=='half_link_flow':
        if c_hl:
            m_init = np.array([m_ic,m_ic,2*m_ic]) #[kg/s]
            m_hl_init = np.array([5*m_hl_ic,m_hl_ic,-m_hl_ic/2]) #[kg/s]
            Ts_init = np.array([Ts_ic,Ts_ic,Ts_ic]) #[C]
            Tr_init = np.array([Tr_ic,Tr_ic,Tr_ic,Tr_ic]) #[C]
            if dummy_links:
                p_init = np.array([ph_ic])
            else:
                p_init = np.array([ph_ic,ph_ic,ph0])
        else:
            m_init = np.array([m_ic]) #[kg/s]
            m_hl_init = np.array([m_hl_ic]) #[kg/s]
            p_init = np.array([ph_ic])
            Ts_init = np.array([Ts_ic]) #[C]
            Tr_init = np.array([Tr_ic,Tr_ic]) #[C]
    else:
        raise NotImplementedError('initialization of heat network not implemented for standard formulation')
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    heat_net.update(x_init,formulation=formulation)
    x0 = heat_net.set_x_init(formulation=formulation)
    return x0

def run_heat_load_flow(max_iter=max_iter_outer,c_hl=False,dummy_links=True):
    """Steady-state load flow analysis of heat network, without scaling. The default values are used for initialization.
    """
    # create network
    heat_net = create_heat_network(c_hl=c_hl,dummy_links=dummy_links)

    # initialize
    x0 = initialize_heat_network(heat_net,c_hl=c_hl,dummy_links=dummy_links)

    # solve network
    print('\nRunning load flow for single-carrier heat network, with explicit coupling input {}, and dummy links {}'.format(c_hl,dummy_links))
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

def create_mes_ge_network():
    """Create a combined gas and electrical network, with two coupling node"""
    # create single-carrier networks
    gas_net = create_gas_network()
    elec_net = create_electrical_network()

    # coupling
    c0 = HeterogeneousNode('cn0',node_type=0,x=x0c,y=y0c,unit_type=unit_type_GG,unit_params=unit_params_GG0)
    c1 = HeterogeneousNode('cn1',node_type=0,x=x1c,y=y1c,unit_type=unit_type_GG,unit_params=unit_params_GG1)

    # change node types/BCs single-carrier networks
    gn0 = gas_net.nodes[0]
    gn1 = gas_net.nodes[1]
    gn0.node_type = 3 # ref. load node
    GasHalfLink('gn0_hl',gn0,q0_source,bc_type=1)
    gn1.remove_half_link(gn1.half_links[0]) #remove the half link representing the coupling

    en0 = elec_net.nodes[0]
    en1 = elec_net.nodes[1]
    en0.node_type = 6 # PQV
    en0.V = V0_sol
    en0.remove_half_link(en0.half_links[0]) #remove the half link representing the coupling
    en1.node_type = 5 # PQVdelta
    ElectricalHalfLink('en1_hl',en1,P=P1_load,Q=Q1_load) # was slack, so didn't have a half link.

    # dummy links
    gl1 = GasLink('gl1',gn0,c0)
    gl2 = GasLink('gl2',gn1,c1)
    gas_net.add_link(gl1)
    gas_net.add_link(gl2)
    el1 = ElectricalLink('el1',c0,en0)
    el2 = ElectricalLink('el2',c1,en1)
    elec_net.add_link(el1)
    elec_net.add_link(el2)

    het_net = HeterogeneousNetwork('2 nodes')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_node(c0)
    het_net.add_node(c1)

    return het_net, gas_net, elec_net

def initialize_mes_ge_network(het_net):
    """Initialize the combined gas and electrical network, with two coupling nodes.

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

    # gas part
    q_init = list()
    pg_init = list()
    for ind_el,el in enumerate(xg_entries):
        if isinstance(el,GasNode):
            pg_init.append(pg_ic)
        elif isinstance(el,GasLink):
            if formulation == 'nodal':
                if link.link_eq_form == 'dp_of_q':
                    warnings.warn("The link {} uses press. drop as function of link flow (fb instead of fa), but formulation is 'nodal'. The link equation for this link is changed to fa!!".format(link.name))
                    link.set_type(link.link_type,link.link_params,link_eq_form='q_of_dp')
            q_init.append(q_ic)
    xg_init = np.array(q_init+pg_init)

    # electrical part
    delta_init = np.zeros(len(unknown_delta_nodes)) # default flat initialization of 0
    V_init = Vbase*np.ones(len(unknown_V_nodes)) # default flat initialization corresponding to 1 p.u.
    xe_init = np.concatenate((delta_init,V_init))

    # coupling
    qc_init = qc_ic*np.ones(len(unknown_qc_links))
    Pc_init = Pc_ic*np.ones(len(unknown_Pc_links))
    Qc_init = Qc_ic*np.ones(len(unknown_Qc_links))
    Sc_init = np.concatenate((Pc_init,Qc_init))
    xc_init = np.concatenate((qc_init,Sc_init))

    x_init = np.concatenate((xg_init,xe_init,xc_init))
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)
    return x0

def run_mes_ge_load_flow(max_iter=max_iter_outer,plot_top=False,plot_jac=False,plot_sol=False,scale_var=None):
    """Steady-state load flow analysis of combined gas and electrical network, without scaling. The default values are used for initialization.
    """
    # create network
    het_net, gas_net, elec_net = create_mes_ge_network()

    # initialize network
    x0 = initialize_mes_ge_network(het_net)
    print('x0 = {}'.format(x0))

    if plot_jac:
        from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
        nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation)
        fig_J = nlsys.spy_plot_J(x0,title='Jacobian spy plot, scaling = {}'.format(scale_var))
        fig_J_map = nlsys.imshow_J(x0,title='Colormap Jacobian, scaling = {}'.format(scale_var))
    # solve network
    print('\nRunning load flow for multi-carrier gas-electricity network')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

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

    return het_net, gas_net, elec_net,x_sol,iters,err_vec

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_mes_ge_load_flow():
    # Given + When
    _, _, _, x_sol, _, _ = run_mes_ge_load_flow(max_iter=250)

    # Then
    _, xg_sol, _, _ = run_gas_load_flow(max_iter=10)
    q01 = xg_sol[0]
    p1 = xg_sol[1]
    _, xe_sol, _, _ = run_electrical_load_flow(max_iter=10)
    delta0 = xe_sol[0]
    x_sol_expected = np.array([q01,p1,delta0,qc0_sol,qc1_sol,Pc0_sol,Pc1_sol,Qc0_sol,Qc1_sol])
    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_mes_ge_load_flow_scaled():
    # Given + When
    _, _, _, x_sol, _, _ = run_mes_ge_load_flow(max_iter=250,scale_var='matrix')

    # Then
    _, xg_sol, _, _ = run_gas_load_flow(max_iter=10)
    q01 = xg_sol[0]
    p1 = xg_sol[1]
    _, xe_sol, _, _ = run_electrical_load_flow(max_iter=10)
    delta0 = xe_sol[0]
    x_sol_expected = np.array([q01,p1,delta0,qc0_sol,qc1_sol,Pc0_sol,Pc1_sol,Qc0_sol,Qc1_sol])
    assert np.allclose(x_sol,x_sol_expected)

def create_mes_eh_network():
    """Create a combined electrical and heat network, with two coupling nodes"""
    # create single-carrier networks
    elec_net = create_electrical_network()
    heat_net = create_heat_network(c_hl=False)

    # coupling
    c0 = HeterogeneousNode('cn0',node_type=1,x=x0c,y=y0c,unit_type=unit_type_CHP,unit_params=unit_params_CHP0) # To known
    c0_hl_gas = GasHalfLink('cn0_hlg',c0,-qc0_sol_CHP,bc_type=1) # gas flows into coupling node, q known
    c1 = HeterogeneousNode('cn1',node_type=1,x=x1c,y=y1c,unit_type=unit_type_CHP,unit_params=unit_params_CHP1) # To known
    c1_hl_gas = GasHalfLink('cn1_hlg',c1,-qc1_sol_CHP,bc_type=1) # gas flows into coupling node, q known

    # change node types/BCs single-carrier networks
    en0 = elec_net.nodes[0]
    en1 = elec_net.nodes[1]
    en0.node_type = 6 # PQV
    en0.V = V0_sol
    en0.remove_half_link(en0.half_links[0]) #remove the half link representing the coupling
    en1.node_type = 5 # PQVdelta
    ElectricalHalfLink('en1_hl',en1,P=P1_load,Q=Q1_load) # was slack, so didn't have a half link.

    hn0 = heat_net.nodes[0]
    hn0_hl = hn0.half_links[0]
    hn1 = heat_net.nodes[1]
    hn1_hl = hn1.half_links[0]
    hn0.node_type = 3 #ref. load (sink) node
    hn0_hl.set_type('heat_exchanger',{'carrier':water},bc_type=3) #phi and To known (sink)
    hn0_hl.dphi = phi0_sink
    hn0_hl.Tr = To0_sink
    hn1_hl.dphi = phi1_sink
    hn1_hl.Tr = To1_sink

    # dummy links
    el1 = ElectricalLink('el1',c0,en0)
    el2 = ElectricalLink('el2',c1,en1)
    elec_net.add_link(el1)
    elec_net.add_link(el2)
    hl1 = HeatLink('hl1',c0,hn0,link_params={'carrier':water},bc_type=6,Tsstart=Toc0) # To of coupling (source) is known
    hl2 = HeatLink('hl2',c1,hn1,link_params={'carrier':water},bc_type=6,Tsstart=Toc1) # To of coupling (source) is known
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)

    # mes
    het_net = HeterogeneousNetwork('2 nodes')
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    het_net.add_node(c0)
    het_net.add_node(c1)

    return het_net, elec_net, heat_net

def initialize_mes_eh_network(het_net):
    """Initialize the combined electrical and heat network, with two coupling nodes

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
    Pc_init = Pc_ic*np.ones(len(unknown_Pc_links))
    Qc_init = Qc_ic*np.ones(len(unknown_Qc_links))
    Sc_init = np.concatenate((Pc_init,Qc_init))
    mc_init = m_ic*np.ones(len(unknown_mc_links))
    phic_init = phi_ic*np.ones(len(unknown_dphi_links))
    Toc_init = Ts_ic*np.ones(len(unknown_Ts_links))
    xc_init = np.concatenate((Sc_init,mc_init,phic_init,Toc_init))
    x_init = np.concatenate((xe_init,xh_init,xc_init))
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)
    return x0

def run_mes_eh_load_flow(max_iter=max_iter_outer,plot_top=False,plot_jac=False,plot_sol=False,scale_var=None):
    """Steady-state load flow analysis of combined electrical and heat network. The default values are used for initialization.
    """
    # create network
    het_net, elec_net, heat_net = create_mes_eh_network()

     # initialize network
    x0 = initialize_mes_eh_network(het_net)

    if plot_jac:
        from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
        nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation)
        fig_J = nlsys.spy_plot_J(x0,title='Jacobian spy plot, scaling = {}'.format(scale_var))
        fig_J_map = nlsys.imshow_J(x0,title='Colormap Jacobian, scaling = {}'.format(scale_var))
    # solve network
    print('\nRunning load flow for multi-carrier electricity-heat network')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution after {} iterations (final error = {:.4e}):'.format(iters,err_vec[-1]))
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

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_eh_load_flow():
    # Given + When
    _, _, _, x_sol, _, _ = run_mes_eh_load_flow(max_iter=10)

    # Then
    _, xe_sol, _, _ = run_electrical_load_flow(max_iter=10)
    delta0 = xe_sol[0]
    _, xh_sol, _, _ = run_heat_load_flow(max_iter=10,c_hl=True,dummy_links=True)
    m01 = xh_sol[0]
    m0 = xh_sol[4]
    m1 = xh_sol[3]
    ph1 = xh_sol[6]
    Ts1 = xh_sol[7]
    Tr0 = xh_sol[10]
    Tr1 = xh_sol[11]

    mc0 = phic0_sol/(Cp_w*(Toc0-Tr0))
    mc1 = phic1_sol/(Cp_w*(Toc1-Tr1))
    x_sol_expected = np.array([delta0,m01,m0,m1,ph1,Ts0,Ts1,Tr0,Tr1,Pc0_sol,Pc1_sol,Qc0_sol,Qc1_sol,mc0,mc1,phic0_sol,phic1_sol])
    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_eh_load_flow_scaled():
    # Given + When
    _, _, _, x_sol, _, _ = run_mes_eh_load_flow(max_iter=10,scale_var='matrix')

    # Then
    _, xe_sol, _, _ = run_electrical_load_flow(max_iter=10)
    delta0 = xe_sol[0]
    _, xh_sol, _, _ = run_heat_load_flow(max_iter=10,c_hl=True,dummy_links=True)
    m01 = xh_sol[0]
    m0 = xh_sol[4]
    m1 = xh_sol[3]
    ph1 = xh_sol[6]
    Ts1 = xh_sol[7]
    Tr0 = xh_sol[10]
    Tr1 = xh_sol[11]

    mc0 = phic0_sol/(Cp_w*(Toc0-Tr0))
    mc1 = phic1_sol/(Cp_w*(Toc1-Tr1))
    x_sol_expected = np.array([delta0,m01,m0,m1,ph1,Ts0,Ts1,Tr0,Tr1,Pc0_sol,Pc1_sol,Qc0_sol,Qc1_sol,mc0,mc1,phic0_sol,phic1_sol])
    assert np.allclose(x_sol,x_sol_expected)

def create_mes_eh_network_Toc1_unknown():
    """Create a combined electrical and heat network, with two coupling nodes"""
    # create single-carrier networks
    elec_net = create_electrical_network()
    heat_net = create_heat_network(c_hl=False)

    # coupling
    c0 = HeterogeneousNode('cn0',node_type=1,x=x0c,y=y0c,unit_type=unit_type_CHP,unit_params=unit_params_CHP0) # To known
    c0_hl_gas = GasHalfLink('cn0_hlg',c0,-qc0_sol_CHP,bc_type=1) # gas flows into coupling node, q known
    c1 = HeterogeneousNode('cn1',node_type=0,x=x1c,y=y1c,unit_type=unit_type_CHP,unit_params=unit_params_CHP1) # To unknown
    c1_hl_gas = GasHalfLink('cn1_hlg',c1,-qc1_sol_CHP,bc_type=1) # gas flows into coupling node, q known

    # change node types/BCs single-carrier networks
    en0 = elec_net.nodes[0]
    en1 = elec_net.nodes[1]
    en0.node_type = 6 # PQV
    en0.V = V0_sol
    en0.remove_half_link(en0.half_links[0]) #remove the half link representing the coupling
    en1.node_type = 5 # PQVdelta
    ElectricalHalfLink('en1_hl',en1,P=P1_load,Q=Q1_load) # was slack, so didn't have a half link.

    hn0 = heat_net.nodes[0]
    hn0_hl = hn0.half_links[0]
    hn1 = heat_net.nodes[1]
    hn1_hl = hn1.half_links[0]
    hn0.node_type = 3 #ref. load (sink) node
    hn0_hl.set_type('heat_exchanger',{'carrier':water},bc_type=3) #phi and To known (sink)
    hn0_hl.dphi = phi0_sink
    hn0_hl.Tr = To0_sink
    hn1.node_type = 16 # load (sink) temp. node (Tr known)
    hn1.Tr = Tr1_sol
    hn1_hl.dphi = phi1_sink
    hn1_hl.Tr = To1_sink

    # dummy links
    el1 = ElectricalLink('el1',c0,en0)
    el2 = ElectricalLink('el2',c1,en1)
    elec_net.add_link(el1)
    elec_net.add_link(el2)
    hl1 = HeatLink('hl1',c0,hn0,link_params={'carrier':water},bc_type=6,Tsstart=Toc0) # To of coupling (source) is known
    hl2 = HeatLink('hl2',c1,hn1,link_params={'carrier':water},bc_type=0,Tsstart=Toc1) # To of coupling (source) is unknown
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)

    # mes
    het_net = HeterogeneousNetwork('2 nodes')
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)
    het_net.add_node(c0)
    het_net.add_node(c1)

    return het_net, elec_net, heat_net

def run_mes_eh_load_flow_Toc1_unknown(max_iter=max_iter_outer,plot_top=False,plot_jac=False,plot_sol=False,scale_var=None):
    """Steady-state load flow analysis of combined electrical and heat network. The default values are used for initialization.
    """
    # create network
    het_net, elec_net, heat_net = create_mes_eh_network_Toc1_unknown()

     # initialize network
    x0 = initialize_mes_eh_network(het_net)

    if plot_jac:
        from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
        nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation)
        fig_J = nlsys.spy_plot_J(x0,title='Jacobian spy plot, scaling = {}'.format(scale_var))
        fig_J_map = nlsys.imshow_J(x0,title='Colormap Jacobian, scaling = {}'.format(scale_var))
    # solve network
    print('\nRunning load flow for multi-carrier electricity-heat network')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution after {} iterations (final error = {:.4e}):'.format(iters,err_vec[-1]))
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

#@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
#def example_mes_eh_load_flow_Toc1_unknown():
    #"""Test the load flow for electricity-heat MES with Toc1 unknown. For the current BC's / node set, this problem is ill-posed."""
    ## Given + When
    #_, _, _, x_sol, _, _ = run_mes_eh_load_flow_Toc1_unknown(max_iter=10)

    ## Then
    #_, xe_sol, _, _ = run_electrical_load_flow(max_iter=10)
    #delta0 = xe_sol[0]
    #_, xh_sol, _, _ = run_heat_load_flow(max_iter=10,c_hl=True,dummy_links=True)
    #m01 = xh_sol[0]
    #m0 = xh_sol[4]
    #m1 = xh_sol[3]
    #ph1 = xh_sol[6]
    #Ts1 = xh_sol[7]
    #Tr0 = xh_sol[10]
    #Toc1_sol = [9]

    #x_sol_expected = np.array([delta0,m01,m0,m1,ph1,Ts0,Ts1,Tr0,Pc0_sol,Pc1_sol,Qc0_sol,Qc1_sol,mc0_sol,mc1_sol,phic0_sol,phic1_sol,Toc1_sol])
    #assert np.allclose(x_sol,x_sol_expected)

def create_coupling_ge_single_network():
    """Create a coupling network consisting of two nodes, for a gas-electricity coupling"""
    c0 = HeterogeneousNode('cn0',node_type=0,x=x0c,y=y0c,unit_type=unit_type_GG,unit_params=unit_params_GG0)
    hlg = GasHalfLink('cn0_hlg',c0,-qc0_sol,bc_type=1) # gas flows into coupling node, q known
    hle = ElectricalHalfLink('cn0_hle',c0,P=Pc_ic,Q=Qc0_sol,bc_type=3) # P is unknown and Q is known
    c1 = HeterogeneousNode('cn1',node_type=0,x=x1c,y=y1c,unit_type=unit_type_GG,unit_params=unit_params_GG1)
    hlg = GasHalfLink('cn1_hlg',c1,-qc_ic,bc_type=0) # gas flows into coupling node, q is unknown
    hle = ElectricalHalfLink('cn1_hle',c1,P=Pc1_sol,Q=Qc1_sol,bc_type=1) # P and Q are known
    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)
    coupling_net.add_node(c1)

    return coupling_net

def initialize_coupling_ge_single_network(coupling_net):
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
    x_init = np.array([qc_ic,Pc_ic])
    coupling_net.initialize()
    coupling_net.update(x_init,formulation=formulation)
    x0 = coupling_net.set_x_init(formulation=formulation)
    return x0

def run_mes_coupling_ge_single_load_flow(max_iter=max_iter_outer,scale_var=None):
    """Steady-state load flow analysis for the coupling network for the gas-electricity"""
    # create network
    coupling_net = create_coupling_ge_single_network()

    # initialize network
    x0 = initialize_coupling_ge_single_network(coupling_net)

    # solve network
    print('\nRunning load flow for gas-electricity network, only coupling')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('qc = {} kg/s'.format([-hl.get_q() for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,GasHalfLink)]))
    print('Pc = {} MW'.format([hl.get_P()/MW for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,ElectricalHalfLink)]))
    print('Qc = {} MW'.format([hl.get_Q()/MW for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,ElectricalHalfLink)]))

    return coupling_net, x_sol, iters, err_vec

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_coupling_ge_load_flow_single_network():
    # Given + When
    _, x_sol, _, _ = run_mes_coupling_ge_single_load_flow(max_iter=250)

    # Then
    x_sol_expected = np.array([qc1_sol,Pc0_sol])
    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_coupling_ge_load_flow_single_network_scaled():
    # Given + When
    _, x_sol, _, _ = run_mes_coupling_ge_single_load_flow(max_iter=250,scale_var='matrix')

    # Then
    x_sol_expected = np.array([qc1_sol,Pc0_sol])
    assert np.allclose(x_sol,x_sol_expected)

def create_coupling_eh_single_network():
    """Create a coupling network consisting of two nodes, for electricity-heat coupling"""
    c0 = HeterogeneousNode('cn0',node_type=4,x=x0c,y=y0c,unit_type=unit_type_CHP,unit_params=unit_params_CHP0) # To known, no heat power equations
    c0_hl_heat = HeatHalfLink('cn0_hlh',c0,link_type='heat_exchanger',link_params={'carrier':water},bc_type=14,Ts=Toc0,dphi=-phic0_sol,m=-mc0_sol) # m, Ts and dphi known (and Tr known), source
    c0_hl_heat.Tr = Tr0_sol # Set to solution
    c0_hl_gas = GasHalfLink('cn0_hlg',c0,-qc0_sol_CHP,bc_type=1) # gas flows into coupling node, q known
    c0_hl_elec = ElectricalHalfLink('cn0_hle',c0,P=Pc_ic,Q=Qc0_sol,bc_type=3) # P is unknown and Q is known
    c1 = HeterogeneousNode('cn1',node_type=4,x=x1c,y=y1c,unit_type=unit_type_CHP,unit_params=unit_params_CHP1) # T known, no heat power equations
    c1_hl_heat = HeatHalfLink('cn1_hlh',c1,link_type='heat_exchanger',link_params={'carrier':water},bc_type=18,m=-mc1_sol) # m and Ts (and Tr) known, dphi unknown, source
    c1_hl_heat.Tr = Tr1_sol # Set to solution
    c1_hl_gas = GasHalfLink('cn1_hlg',c1,-qc1_sol_CHP,bc_type=1) # gas flows into coupling node, q known
    c1_hl_elec = ElectricalHalfLink('cn1_hle',c1,P=Pc1_sol,Q=Qc1_sol,bc_type=1) # P and Q are known

    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)
    coupling_net.add_node(c1)

    return coupling_net

def create_coupling_eh_single_network_Toc1_unknown():
    """Create a coupling network consisting of two nodes, for electricity-heat coupling. Outflow / supply temperature of coupling 1 is unknown in coupling network. """
    c0 = HeterogeneousNode('cn0',node_type=4,x=x0c,y=y0c,unit_type=unit_type_CHP,unit_params=unit_params_CHP0) # To known, no heat power equations
    c0_hl_heat = HeatHalfLink('cn0_hlh',c0,link_type='heat_exchanger',link_params={'carrier':water},bc_type=14,Ts=Toc0,dphi=-phic0_sol,m=-mc0_sol) # m, Ts and dphi known (and Tr known), source
    c0_hl_heat.Tr = Tr0_sol # Set to solution
    c0_hl_gas = GasHalfLink('cn0_hlg',c0,-qc0_sol_CHP,bc_type=1) # gas flows into coupling node, q known
    c0_hl_elec = ElectricalHalfLink('cn0_hle',c0,P=Pc_ic,Q=Qc0_sol,bc_type=3) # P is unknown and Q is known
    c1 = HeterogeneousNode('cn1',node_type=0,x=x1c,y=y1c,unit_type=unit_type_CHP,unit_params=unit_params_CHP1) # To unknown
    c1_hl_heat = HeatHalfLink('cn1_hlh',c1,link_type='heat_exchanger',link_params={'carrier':water},bc_type=12,m=-mc1_sol) # m (and Tr) known, Ts and dphi unknown, source
    c1_hl_heat.Tr = Tr1_sol # Set to solution
    c1_hl_gas = GasHalfLink('cn1_hlg',c1,-qc1_sol_CHP,bc_type=1) # gas flows into coupling node, q known
    c1_hl_elec = ElectricalHalfLink('cn1_hle',c1,P=Pc1_sol,Q=Qc1_sol,bc_type=1) # P and Q are known

    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)
    coupling_net.add_node(c1)

    return coupling_net

def update_coupling_eh_single_network_bc(coupling_net,elec_net,heat_net,xe,xh,c_hl=False,dummy_links=True,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Update the boundary values of the coupling electricity-heat network, based on the outcome xe and xh of the single-carrier networks.
    """
    # update separate network fully
    elec_net.reset_network(xe)
    elec_net.update_full(xe)
    heat_net.reset_network(xh,formulation=formulation.get('heat'))
    heat_net.update_full(xh,formulation=formulation.get('heat'))

    # electrical part
    Q0c = -elec_net.nodes[0].half_links[0].get_Q()
    elec_halflink1 = elec_net.nodes[1].half_links[0]  # slack, so this is one half link
    P1c = -(elec_halflink1.get_P() - P1_load) #>0, this is already the coupling energy. P1_load is assumed known
    Q1c = -(elec_halflink1.get_Q() - Q1_load)
    # heat part
    heat_halflink0 = heat_net.nodes[0].half_links[0]
    m0c = heat_halflink0.get_m() #<0
    dphi0c = heat_halflink0.get_dphi() #<0
    Ts0c = heat_net.nodes[0].get_Ts()
    Tr0c = heat_net.nodes[0].get_Tr()
    if c_hl:
        m1c = xh[5] #<0
        Ts1c = xh[-5]
        Tr1c = xh[-1]
    else:
        raise NotImplementedError("Updating coupling network BC's from heat data not implemented for heat net without extra links")

    # update BCs (fully. Not all of these BCs are needed for the load flow)
    cn0 = coupling_net.nodes[0]
    cn1 = coupling_net.nodes[1]
    cn0_hlh = cn0.half_links[0]
    cn0_hle = cn0.half_links[2]
    cn1_hlh = cn1.half_links[0]
    cn1_hle = cn1.half_links[2]
    cn0_hle.Q = Q0c
    cn0_hlh.m = m0c
    cn0_hlh.dphi = dphi0c
    cn0_hlh.Ts = Ts0c
    cn0_hlh.Tr = Tr0c
    cn1_hle.P = P1c
    cn1_hle.Q = Q1c
    cn1_hlh.m = m1c
    cn1_hlh.Tr = Tr1c
    cn1_hlh.Ts = Ts1c

    return coupling_net

def update_coupling_eh_single_network_bc_Toc1_unknown(coupling_net,elec_net,heat_net,xe,xh,c_hl=False,dummy_links=True,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Update the boundary values of the coupling electricity-heat network, based on the outcome xe and xh of the single-carrier networks. Outflow / supply temperature of coupling 1 is unknown in coupling network.
    """
    # update separate network fully
    elec_net.reset_network(xe)
    elec_net.update_full(xe)
    heat_net.reset_network(xh,formulation=formulation.get('heat'))
    heat_net.update_full(xh,formulation=formulation.get('heat'))

    # electrical part
    Q0c = -elec_net.nodes[0].half_links[0].get_Q()
    elec_halflink1 = elec_net.nodes[1].half_links[0]  # slack, so this is one half link
    P1c = -(elec_halflink1.get_P() - P1_load) #>0, this is already the coupling energy. P1_load is assumed known
    Q1c = -(elec_halflink1.get_Q() - Q1_load)
    # heat part
    heat_halflink0 = heat_net.nodes[0].half_links[0]
    m0c = heat_halflink0.get_m() #<0
    dphi0c = heat_halflink0.get_dphi() #<0
    Ts0c = heat_net.nodes[0].get_Ts()
    Tr0c = heat_net.nodes[0].get_Tr()
    if c_hl:
        m1c = xh[5] #<0
        Tr1c = xh[-1]
    else:
        raise NotImplementedError("Updating coupling network BC's from heat data not implemented for heat net without extra links")

    # update BCs (fully. Not all of these BCs are needed for the load flow)
    cn0 = coupling_net.nodes[0]
    cn1 = coupling_net.nodes[1]
    cn0_hlh = cn0.half_links[0]
    cn0_hle = cn0.half_links[2]
    cn1_hlh = cn1.half_links[0]
    cn1_hle = cn1.half_links[2]
    cn0_hle.Q = Q0c
    cn0_hlh.m = m0c
    cn0_hlh.dphi = dphi0c
    cn0_hlh.Ts = Ts0c
    cn0_hlh.Tr = Tr0c
    cn1_hle.P = P1c
    cn1_hle.Q = Q1c
    cn1_hlh.m = m1c
    cn1_hlh.Tr = Tr1c

    return coupling_net

def initialize_coupling_eh_single_network(coupling_net):
    """Initialize the coupling network consisting of two nodes, for a electricity-heat coupling

    Parameters
    ----------
    coupling_net : HeterogeneousNetwork
        The coupling network to be initialized

    Returns
    -------
    x0 : np array
        initial guess
    """
    x_init = np.array([Pc_ic,phi_ic])
    coupling_net.initialize()
    coupling_net.update(x_init,formulation=formulation)
    x0 = coupling_net.set_x_init(formulation=formulation)
    return x0

def initialize_coupling_eh_single_network_Toc1_unknown(coupling_net):
    """Initialize the coupling network consisting of two nodes, for a electricity-heat coupling

    Parameters
    ----------
    coupling_net : HeterogeneousNetwork
        The coupling network to be initialized

    Returns
    -------
    x0 : np array
        initial guess
    """
    x_init = np.array([Pc_ic,phi_ic,To_ic])
    coupling_net.initialize()
    coupling_net.update(x_init,formulation=formulation)
    x0 = coupling_net.set_x_init(formulation=formulation)
    return x0

def run_mes_coupling_eh_single_load_flow(max_iter=max_iter_outer,scale_var=None):
    """Steady-state load flow analysis for the coupling network for the heat-electricity"""
    # create network
    coupling_net = create_coupling_eh_single_network()

    # initialize network
    x0 = initialize_coupling_eh_single_network(coupling_net)

    # solve network
    print('\nRunning load flow for heat-electricity network, only coupling')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution after {} iterations (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print('qc = {} kg/s'.format(qc_vec))
    print('Pc = {} MW'.format([Pc/MW for Pc in Pc_vec]))
    print('Qc = {} MW'.format([Qc/MW for Qc in Qc_vec]))
    print('mc = {} kg/s'.format(mc_vec))
    print('phic = {} MW'.format([phic/MW for phic in phic_vec]))
    print('Tsc = {} C'.format(Tsc_vec))
    print('Trc = {} C'.format(Trc_vec))

    return coupling_net, x_sol, iters, err_vec

def run_mes_coupling_eh_single_load_flow_Toc1_unknown(max_iter=max_iter_outer,scale_var=None):
    """Steady-state load flow analysis for the coupling network for the heat-electricity"""
    # create network
    coupling_net = create_coupling_eh_single_network_Toc1_unknown()

    # initialize network
    x0 = initialize_coupling_eh_single_network_Toc1_unknown(coupling_net)

    # solve network
    print('\nRunning load flow for heat-electricity network, only coupling')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution after {} iterations (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print('qc = {} kg/s'.format(qc_vec))
    print('Pc = {} MW'.format([Pc/MW for Pc in Pc_vec]))
    print('Qc = {} MW'.format([Qc/MW for Qc in Qc_vec]))
    print('mc = {} kg/s'.format(mc_vec))
    print('phic = {} MW'.format([phic/MW for phic in phic_vec]))
    print('Tsc = {} C'.format(Tsc_vec))
    print('Trc = {} C'.format(Trc_vec))

    return coupling_net, x_sol, iters, err_vec

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_coupling_eh_load_flow_single_network():
    # Given + When
    _, x_sol, _, _ = run_mes_coupling_eh_single_load_flow(max_iter=250)

    # Then
    x_sol_expected = np.array([Pc0_sol,phic1_sol])
    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_coupling_eh_load_flow_single_network_scaled():
    # Given + When
    _, x_sol, _, _ = run_mes_coupling_eh_single_load_flow(max_iter=250,scale_var='matrix')

    # Then
    x_sol_expected = np.array([Pc0_sol,phic1_sol])
    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_coupling_eh_load_flow_single_network_Toc1_unknown():
    # Given + When
    _, x_sol, _, _ = run_mes_coupling_eh_single_load_flow_Toc1_unknown(max_iter=250)

    # Then
    x_sol_expected = np.array([Pc0_sol,phic1_sol,Toc1])
    assert np.allclose(x_sol,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_coupling_eh_load_flow_single_network_Toc1_unknown_scaled():
    # Given + When
    _, x_sol, _, _ = run_mes_coupling_eh_single_load_flow_Toc1_unknown(max_iter=250,scale_var='matrix')

    # Then
    x_sol_expected = np.array([Pc0_sol,phic1_sol,Toc1])
    assert np.allclose(x_sol,x_sol_expected)

def error_measure_dd_ge(gas_net,elec_net,coupling_net,xg,xe,xc,xc_full,scale_var='matrix'):
    """The error measure for DD of a gas-electricity coupling, based on (residual of) the system of equations

    Parameters
    ----------
    xc_full : np array
        The full vector of the coupling part, in this case [q0c, q1c, P0c, P1c, Q0c, Q1c]
    """
    # update all boundary values
    q0c, q1c, P0c, P1c, Q0c, Q1c = xc_full
    gas_net.nodes[1].half_links[0].q = q1c
    elec_net.nodes[0].half_links[0].P = - P0c
    cn0 = coupling_net.nodes[0]
    cn0.half_links[0].q = -q0c
    cn0.half_links[1].Q = Q0c
    cn1 = coupling_net.nodes[1]
    cn1.half_links[1].P = P1c
    cn1.half_links[1].Q = Q1c

    # determine the value of (residual of) the system of equations
    from meslf.load_flow.system_of_equations import NonLinearSystemGas, NonLinearSystemElectrical, NonLinearSystemHeterogeneous
    if scale_var:
        scale_var_params_err_measure = scale_var_params
    else:
        scale_var_params_err_measure = None
    nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params_err_measure)
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params_err_measure)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params_err_measure)
    Fg = nlsysg.DF().dot(nlsysg.F(xg))
    Fe = nlsyse.DF().dot(nlsyse.F(xe))
    Fc = nlsysc.DF().dot(nlsysc.F(xc))
    error = np.linalg.norm(np.concatenate((Fg,Fe,Fc)))

    return error

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_mes_ge_single_network_dd_error_measure():
    # Given
    het_net, gas_net, elec_net,x_sol,iters,err_vec = run_mes_ge_load_flow()
    #x_sol =[q01,p1,delta0,qc0_sol,qc1_sol,Pc0_sol,Pc1_sol,Qc0_sol,Qc1_sol]
    xg = x_sol[0:2]
    xe = np.array([x_sol[2]])
    xc_full = x_sol[3:]
    xc = np.array([xc_full[1],xc_full[2]])
    gas_net = create_gas_network()
    elec_net = create_electrical_network()
    coupling_net = create_coupling_ge_single_network()
    initialize_electrical_network(elec_net)
    initialize_gas_network(gas_net)
    initialize_coupling_ge_single_network(coupling_net)

    # When
    error = error_measure_dd_ge(gas_net,elec_net,coupling_net,xg,xe,xc,xc_full)
    error_norm = np.linalg.norm(error)

    # Then
    assert np.isclose(error_norm,0)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_mes_ge_single_network_dd_error_measure_scaled():
    # Given
    het_net, gas_net, elec_net,x_sol,iters,err_vec = run_mes_ge_load_flow(scale_var='matrix')
    #x_sol =[q01,p1,delta0,qc0_sol,qc1_sol,Pc0_sol,Pc1_sol,Qc0_sol,Qc1_sol]
    xg = x_sol[0:2]
    xe = np.array([x_sol[2]])
    xc_full = x_sol[3:]
    xc = np.array([xc_full[1],xc_full[2]])
    gas_net = create_gas_network()
    elec_net = create_electrical_network()
    coupling_net = create_coupling_ge_single_network()
    initialize_electrical_network(elec_net)
    initialize_gas_network(gas_net)
    initialize_coupling_ge_single_network(coupling_net)

    # When
    error = error_measure_dd_ge(gas_net,elec_net,coupling_net,xg,xe,xc,xc_full,scale_var='matrix')
    error_norm = np.linalg.norm(error)

    # Then
    assert np.isclose(error_norm,0)

def xmes_to_xseparate_eh(xmes,c_hl=False,dummy_links=True):
    """Split the x for the electricity-heat mes to the x's of the separate networks.
    xmes is given by [delta0,m01,m0,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c].
    """
    # electricity
    xe = np.array([xmes[0]])
    #heat
    delta0,m01,m0,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c = xmes
    Toc0 = Ts0
    Toc1 = Ts1
    if c_hl:
        xh_m = np.array([m01,m0,m1c])
        xh_m_hl = np.array([m1,m0,-m1c])
        if dummy_links:
            xh_p = np.array([p1])
        else:
            xh_p = np.array([p1,heat_net.nodes[2].p,heat_net.nodes[3]]) # pressure at additional nodes in not important
        xh_Ts = np.array([Ts1,Ts0,Toc1])
        xh_Tr = np.array([Tr0,Tr1,To0_sink,Tr1])
        xh = np.concatenate((xh_m,xh_m_hl,xh_p,xh_Ts,xh_Tr))
    else:
        raise NotImplementedError('Splitting xmes into separate xh is not implemented for heat network without extra links.')
    # coupling
    xc = np.array([P0c,phi1c])

    return xe,xh,xc

def xmes_to_xseparate_eh_Toc1_unknown(xmes,c_hl=False,dummy_links=True):
    """Split the x for the electricity-heat mes to the x's of the separate networks.
    xmes is given by [delta0,m01,m0,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c].
    """
    # electricity
    xe = np.array([xmes[0]])
    #heat
    delta0,m01,m0,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c = xmes
    Toc0 = Ts0
    Toc1 = Ts1
    if c_hl:
        xh_m = np.array([m01,m0,m1c])
        xh_m_hl = np.array([m1,m0,-m1c])
        if dummy_links:
            xh_p = np.array([p1])
        else:
            xh_p = np.array([p1,heat_net.nodes[2].p,heat_net.nodes[3]]) # pressure at additional nodes in not important
        xh_Ts = np.array([Ts1,Ts0,Toc1])
        xh_Tr = np.array([Tr0,Tr1,To0_sink,Tr1])
        xh = np.concatenate((xh_m,xh_m_hl,xh_p,xh_Ts,xh_Tr))
    else:
        raise NotImplementedError('Splitting xmes into separate xh is not implemented for heat network without extra links.')
    # coupling
    xc = np.array([P0c,phi1c,Toc1])

    return xe,xh,xc

def xseparate_to_xmes_eh(xe,xh,xc,elec_net,heat_net,c_hl=False,dummy_links=True,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Create x for the electricity-heat mes from the separate x's.
    """
    # update separate network fully
    elec_net.reset_network(xe)
    elec_net.update_full(xe)
    heat_net.reset_network(xh,formulation=formulation.get('heat'))
    heat_net.update_full(xh,formulation=formulation.get('heat'))

    # electrical part
    delta0 = xe[0]
    Q0c = -elec_net.nodes[0].half_links[0].get_Q()
    elec_halflink1 = elec_net.nodes[1].half_links[0]  # slack, so this is one half link
    P1c = -(elec_halflink1.get_P() - P1_load) #<0, since is is a source. P1_load is assumed known
    Q1c = -(elec_halflink1.get_Q() - Q1_load)
    # heat part
    if c_hl:
        if dummy_links:
            m01,m00extra,m1extra1,m1,m0extra,m1extra,p1,Ts1,Ts0extra,Ts1extra,Tr0,Tr1,Tr0extra,Tr1extra = xh
        else:
            m01,m00extra,m1extra1,m1,m0extra,m1extra,p1,p0extra,p1extra,Ts1,Ts0extra,Ts1extra,Tr0,Tr1,Tr0extra,Tr1extra = xh
    else:
        raise NotImplementedError('Creating xmese from xh is not implemented for heat network without extra links.')
    heat_halflink0 = heat_net.nodes[0].half_links[0]
    m0c = heat_halflink0.get_m() #<0
    dphi0c = heat_halflink0.get_dphi() #<0
    Ts0 = heat_net.nodes[0].Ts
    # couplig part
    P0c,phi1c = xc

    # xmes is given by [delta0,m01,m0,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c]
    xmes = np.array([delta0,m01,m00extra,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, -m0c, m1extra1, -dphi0c, phi1c])
    return xmes

def xseparate_to_xmes_eh_Toc1_unknown(xe,xh,xc,elec_net,heat_net,c_hl=False,dummy_links=True,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Create x for the electricity-heat mes from the separate x's.
    """
    # update separate network fully
    elec_net.reset_network(xe)
    elec_net.update_full(xe)
    heat_net.reset_network(xh,formulation=formulation.get('heat'))
    heat_net.update_full(xh,formulation=formulation.get('heat'))

    # electrical part
    delta0 = xe[0]
    Q0c = -elec_net.nodes[0].half_links[0].get_Q()
    elec_halflink1 = elec_net.nodes[1].half_links[0]  # slack, so this is one half link
    P1c = -(elec_halflink1.get_P() - P1_load) #<0, since is is a source. P1_load is assumed known
    Q1c = -(elec_halflink1.get_Q() - Q1_load)
    # heat part
    if c_hl:
        if dummy_links:
            m01,m00extra,m1extra1,m1,m0extra,m1extra,p1,Ts1,Ts0extra,Ts1extra,Tr0,Tr1,Tr0extra,Tr1extra = xh
        else:
            m01,m00extra,m1extra1,m1,m0extra,m1extra,p1,p0extra,p1extra,Ts1,Ts0extra,Ts1extra,Tr0,Tr1,Tr0extra,Tr1extra = xh
    else:
        raise NotImplementedError('Creating xmese from xh is not implemented for heat network without extra links.')
    heat_halflink0 = heat_net.nodes[0].half_links[0]
    m0c = heat_halflink0.get_m() #<0
    dphi0c = heat_halflink0.get_dphi() #<0
    Ts0 = heat_net.nodes[0].Ts
    # couplig part
    P0c,phi1c,Toc1 = xc

    # xmes is given by [delta0,m01,m0,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c]
    xmes = np.array([delta0,m01,m00extra,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, -m0c, m1extra1, -dphi0c, phi1c])
    return xmes

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def test_xseparate_to_xmes_eh():
    # Given + When
    c_hl = True
    dummy_links = True
    elec_net, xe_sol, _, _ = run_electrical_load_flow(max_iter=10)
    heat_net, xh_sol, _, _ = run_heat_load_flow(max_iter=10,c_hl=c_hl,dummy_links=dummy_links)
    xc_sol = np.array([Pc0_sol,phic1_sol]) # given at top of file

    # When
    x_mes  = xseparate_to_xmes_eh(xe_sol,xh_sol,xc_sol,elec_net,heat_net,c_hl=c_hl,dummy_links=dummy_links)

    # Then
    _, _, _, x_sol_expected, _, _ = run_mes_eh_load_flow(max_iter=10,scale_var='matrix')
    assert np.allclose(x_mes,x_sol_expected)

def error_measure_dd_eh(elec_net,heat_net,coupling_net,xe,xh,xc,scale_var='matrix',c_hl=False,dummy_links=True,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """The error measure for DD of a electricity-heat coupling, based on (residual of) the system of equations.
    The network are assumed to be fully updated.
    """
    # update all boundary values
    elec_net = update_electrical_network_bc_eh(elec_net,xc)
    heat_net = update_heat_network_bc(heat_net,xc,c_hl=c_hl,dummy_links=dummy_links)
    coupling_net = update_coupling_eh_single_network_bc(coupling_net,elec_net,heat_net,xe,xh,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation)

    # determine the value of (residual of) the system of equations
    from meslf.load_flow.system_of_equations import NonLinearSystemElectrical, NonLinearSystemHeat, NonLinearSystemHeterogeneous
    if scale_var:
        scale_var_params_err_measure = scale_var_params
    else:
        scale_var_params_err_measure = None
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params_err_measure)
    nlsysh = NonLinearSystemHeat(heat_net,formulation=formulation.get('heat'),scale_var=scale_var,scale_var_params=scale_var_params_err_measure)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params_err_measure)

    Fe = nlsyse.DF().dot(nlsyse.F(xe))
    Fh = nlsysh.DF().dot(nlsysh.F(xh))
    Fc = nlsysc.DF().dot(nlsysc.F(xc))
    error = np.linalg.norm(np.concatenate((Fe,Fh,Fc)))
    return error

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_eh_single_network_dd_error_measure_extra_links():
    # Given
    c_hl=True
    dummy_links=True
    het_net, elec_net, heat_net, x_sol, iters, err_vec = run_mes_eh_load_flow(max_iter=10)
    xe,xh,xc = xmes_to_xseparate_eh(x_sol,c_hl=c_hl,dummy_links=dummy_links)

    elec_net = create_electrical_network()
    heat_net = create_heat_network(c_hl=c_hl,dummy_links=dummy_links)
    coupling_net = create_coupling_eh_single_network()
    initialize_electrical_network(elec_net)
    initialize_heat_network(heat_net,c_hl=c_hl,dummy_links=dummy_links)
    initialize_coupling_eh_single_network(coupling_net)

    # When
    error =  error_measure_dd_eh(elec_net,heat_net,coupling_net,xe,xh,xc,c_hl=c_hl,dummy_links=dummy_links)
    error_norm = np.linalg.norm(error)

    # Then
    assert np.isclose(error_norm,0)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_eh_single_network_dd_error_measure_scaled_extra_links():
    # Given
    c_hl=True
    dummy_links=True
    het_net, elec_net, heat_net, x_sol, iters, err_vec = run_mes_eh_load_flow(max_iter=10,scale_var='matrix')
    xe,xh,xc = xmes_to_xseparate_eh(x_sol,c_hl=c_hl,dummy_links=dummy_links)

    elec_net = create_electrical_network()
    heat_net = create_heat_network(c_hl=c_hl,dummy_links=dummy_links)
    coupling_net = create_coupling_eh_single_network()
    initialize_electrical_network(elec_net)
    initialize_heat_network(heat_net,c_hl=c_hl,dummy_links=dummy_links)
    initialize_coupling_eh_single_network(coupling_net)

    # When
    error =  error_measure_dd_eh(elec_net,heat_net,coupling_net,xe,xh,xc,c_hl=c_hl,dummy_links=dummy_links,scale_var='matrix')
    error_norm = np.linalg.norm(error)

    # Then
    assert np.isclose(error_norm,0)

#@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
#def example_mes_eh_single_network_dd_error_measure():
    #"""Test error measure of DD for the electricity-heat network, where the heat network has no additional links to represent coupling. Currently not implemented"""
    ## Given
    #c_hl=False
    #het_net, elec_net, heat_net, x_sol, iters, err_vec = run_mes_eh_load_flow(max_iter=10)
    ## x_sol is given by [delta0,m01,m0,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c]

    #xe = np.array([x_sol[0]])
    #delta0,m01,m0,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c = x_sol
    #xh_m = np.array([m01])
    #xh_m_hl = np.array([m1])
    #xh_p = np.array([p1]) # pressure at additional nodes in not important
    #xh_Ts = np.array([Ts1])
    #xh_Tr = np.array([Tr0,Tr1])
    #xh = np.concatenate((xh_m,xh_m_hl,xh_p,xh_Ts,xh_Tr))
    #xc_full = np.array([P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c,Toc0,Toc1])
    #xc = np.array([P0c,m0c,m1c,phi1c])

    #elec_net = create_electrical_network()
    #heat_net = create_heat_network(c_hl=c_hl)
    #coupling_net = create_coupling_eh_single_network()
    #initialize_electrical_network(elec_net)
    #initialize_heat_network(heat_net,c_hl=c_hl)
    #initialize_coupling_eh_single_network(coupling_net)

    ## When
    #error =  error_measure_dd_eh(elec_net,heat_net,coupling_net,xe,xh,xc,xc_full,c_hl=c_hl)
    #error_norm = np.linalg.norm(error)

    ## Then
    #assert np.isclose(error_norm,0)

#@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
#def example_mes_eh_single_network_dd_error_measure_scaled():
    #"""Test error measure of DD for the electricity-heat network, using scaling, where the heat network has no additional links to represent coupling. Currently not implemented"""
    ## Given
    #c_hl=False
    #het_net, elec_net, heat_net, x_sol, iters, err_vec = run_mes_eh_load_flow(max_iter=10,scale_var='matrix')
    ## x_sol is given by [delta0,m01,m0,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c]

    #xe = np.array([x_sol[0]])
    #delta0,m01,m0,m1,p1,Ts0,Ts1,Tr0,Tr1,P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c = x_sol
    #xh_m = np.array([m01])
    #xh_m_hl = np.array([m1])
    #xh_p = np.array([p1]) # pressure at additional nodes in not important
    #xh_Ts = np.array([Ts1])
    #xh_Tr = np.array([Tr0,Tr1])
    #xh = np.concatenate((xh_m,xh_m_hl,xh_p,xh_Ts,xh_Tr))
    #xc_full = np.array([P0c, P1c, Q0c, Q1c, m0c, m1c, phi0c, phi1c,Toc0,Toc1])
    #xc = np.array([P0c,m0c,m1c,phi1c])

    #elec_net = create_electrical_network()
    #heat_net = create_heat_network(c_hl=c_hl)
    #coupling_net = create_coupling_eh_single_network()
    #initialize_electrical_network(elec_net)
    #initialize_heat_network(heat_net,c_hl=c_hl)
    #initialize_coupling_eh_single_network(coupling_net)

    ## When
    #error =  error_measure_dd_eh(elec_net,heat_net,coupling_net,xe,xh,xc,xc_full,c_hl=c_hl)
    #error_norm = np.linalg.norm(error)

    ## Then
    #assert np.isclose(error_norm,0)

def error_measure_dd_eh_Toc1_unknown(elec_net,heat_net,coupling_net,xe,xh,xc,scale_var='matrix',c_hl=False,dummy_links=True,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """The error measure for DD of a electricity-heat coupling, based on (residual of) the system of equations.
    The network are assumed to be fully updated.
    """
    # update all boundary values
    elec_net = update_electrical_network_bc_eh(elec_net,xc)
    heat_net = update_heat_network_bc_Toc1_unknown(heat_net,xc,c_hl=c_hl,dummy_links=dummy_links)
    coupling_net = update_coupling_eh_single_network_bc_Toc1_unknown(coupling_net,elec_net,heat_net,xe,xh,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation)

    # determine the value of (residual of) the system of equations
    from meslf.load_flow.system_of_equations import NonLinearSystemElectrical, NonLinearSystemHeat, NonLinearSystemHeterogeneous
    if scale_var:
        scale_var_params_err_measure = scale_var_params
    else:
        scale_var_params_err_measure = None
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params_err_measure)
    nlsysh = NonLinearSystemHeat(heat_net,formulation=formulation.get('heat'),scale_var=scale_var,scale_var_params=scale_var_params_err_measure)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params_err_measure)

    Fe = nlsyse.DF().dot(nlsyse.F(xe))
    Fh = nlsysh.DF().dot(nlsysh.F(xh))
    Fc = nlsysc.DF().dot(nlsysc.F(xc))
    error = np.linalg.norm(np.concatenate((Fe,Fh,Fc)))
    return error

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_eh_single_network_dd_error_measure_Toc1_unknown_extra_links():
    # Given
    c_hl=True
    dummy_links=True
    het_net, elec_net, heat_net, x_sol, iters, err_vec = run_mes_eh_load_flow(max_iter=10)
    xe,xh,xc = xmes_to_xseparate_eh_Toc1_unknown(x_sol,c_hl=c_hl,dummy_links=dummy_links)

    elec_net = create_electrical_network()
    heat_net = create_heat_network(c_hl=c_hl,dummy_links=dummy_links)
    coupling_net = create_coupling_eh_single_network()
    initialize_electrical_network(elec_net)
    initialize_heat_network(heat_net,c_hl=c_hl,dummy_links=dummy_links)
    initialize_coupling_eh_single_network(coupling_net)

    # When
    error =  error_measure_dd_eh_Toc1_unknown(elec_net,heat_net,coupling_net,xe,xh,xc,c_hl=c_hl,dummy_links=dummy_links)
    error_norm = np.linalg.norm(error)

    # Then
    assert np.isclose(error_norm,0)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_eh_single_network_dd_error_measure_Toc1_unknown_scaled_extra_links():
    # Given
    c_hl=True
    dummy_links=True
    het_net, elec_net, heat_net, x_sol, iters, err_vec = run_mes_eh_load_flow(max_iter=10,scale_var='matrix')
    xe,xh,xc = xmes_to_xseparate_eh_Toc1_unknown(x_sol,c_hl=c_hl,dummy_links=dummy_links)

    elec_net = create_electrical_network()
    heat_net = create_heat_network(c_hl=c_hl,dummy_links=dummy_links)
    coupling_net = create_coupling_eh_single_network()
    initialize_electrical_network(elec_net)
    initialize_heat_network(heat_net,c_hl=c_hl,dummy_links=dummy_links)
    initialize_coupling_eh_single_network(coupling_net)

    # When
    error =  error_measure_dd_eh_Toc1_unknown(elec_net,heat_net,coupling_net,xe,xh,xc,c_hl=c_hl,dummy_links=dummy_links,scale_var='matrix')
    error_norm = np.linalg.norm(error)

    # Then
    assert np.isclose(error_norm,0)

def run_mes_ge_load_flow_dd(max_iter=max_iter_outer,error_measure='F',scale_var=None):
    """Steady-state load flow analysis of multi-carrier network, using domain decomposition, without scaling. The default values are used for initialization.

    Uses the already implemented solvers for the single-carrier network. Between single-carrier solves, the boundary values are given to the other domains. The coupling part is solved separately (since the heterogeneous network solver currently assumes qc, Pc, and Qc live on dummy links, not on half links)

    Parameters
    ----------
    error_measure : str, optional
        Determines which error measure to use for the outer iterations. Options are 'F', for the system of equations, or 'dE' for the difference between coupling energies between consecutive iterations. Default if 'F'. NB: for 'dE', the vector with errors is one longer, since an error for iteration 0 cannot be defined.
    """
    # create networks
    gas_net = create_gas_network()
    elec_net = create_electrical_network()
    coupling_net = create_coupling_ge_single_network()
    cn0 = coupling_net.nodes[0]
    cn0_hlg = cn0.half_links[0]
    cn0_hle = cn0.half_links[1]
    cn1 = coupling_net.nodes[1]
    cn1_hlg = cn1.half_links[0]
    cn1_hle = cn1.half_links[1]

    # initialize networks, and set initial guesses
    xe_new = initialize_electrical_network(elec_net)
    xg_new = initialize_gas_network(gas_net)
    xc_new = initialize_coupling_ge_single_network(coupling_net) #[q1c, P0c]
    xc_full_new = np.array([qc_ic, xc_new[0], xc_new[1], Pc_ic, Qc_ic, Qc_ic])

    # iterate
    print('\nRunning load flow with DD, gas-electricity, error {}'.format(error_measure))
    err_vec = list()
    if error_measure == 'F':
        error = error_measure_dd_ge(gas_net,elec_net,coupling_net,xg_new,xe_new,xc_new,xc_full_new,scale_var=scale_var)
        err_vec.append(error)
    elif error_measure == 'dE':
        error = 1 # set some error which is bigger than the tolerance
    else:
        raise ValueError('enter valid error measure')
    if scale_var == 'matrix':
        xb_qc = scale_var_params.get('qbase')*np.ones(2)
        xb_Sc = scale_var_params.get('Sbase')*np.ones(4)
        Dxc_full = sps.diags(1/np.concatenate((xb_qc,xb_Sc)))
    else:
        Dxc_full = sps.eye(len(xc_full_new))
    iter_nr = 0 # I assume solution is not found at 1 iteration
    errors_gas = dict()
    errors_elec = dict()
    errors_coupling = dict()
    while error > tol and iter_nr < max_iter:
        xe_old = xe_new
        xg_old = xg_new
        xc_old = xc_new
        xc_full_old = xc_full_new
        q0c_old, q1c_old, P0c_old, P1c_old, Q0c_old, Q1c_old = xc_full_old

        # Gas network. Update boundary, then solve, then update the boundary values
        gas_net.nodes[1].half_links[0].q = q1c_old
        gas_net.reset_network(xg_old,formulation=formulation.get('gas'))
        with HiddenPrints():
            xg_new,itersg,err_vecg,_,_,q_inj = gas_net.solve_network(tol_sub,max_iter,solver='NR',formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
        errors_gas[iter_nr] = err_vecg
        q0c_new = gas_net.nodes[0].half_links[0].q - q0_source
        if q0c_new <= 0:
            warnings.warn('encountered a negative q0c in iteration {}. It is set equal to -q0_source instead.'.format(iter_nr))
            print('encountered a negative q0c in iteration {}. It is set equal to -q0_source instead.'.format(iter_nr))
            q0c_new = -q0_source

        # Electricity network. Update boundary, then solve, then update the boundary values
        elec_net.nodes[0].half_links[0].P = - P0c_old
        elec_net.reset_network(xe_old)
        with HiddenPrints():
            xe_new,iterse,err_vece,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol_sub,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        errors_elec[iter_nr] = err_vece
        Q0c_new = -elec_net.nodes[0].half_links[0].get_Q()
        P1c_new = -(elec_net.nodes[1].half_links[0].get_P() - P1_load)
        Q1c_new = -(elec_net.nodes[1].half_links[0].get_Q() - Q1_load)
        if P1c_new <= 0:
            warnings.warn('encountered a negative P1c in iteration {}. It is set equal to P1_load instead.'.format(iter_nr))
            print('encountered a negative P1c in iteration {}. It is set equal to P1_load instead.'.format(iter_nr))
            P1c_new = P1_load

        # Coupling network. Update boundary, then solve, then update the boundary values
        cn0_hlg.q = -q0c_new
        cn0_hle.Q = Q0c_new
        cn1_hle.P = P1c_new
        cn1_hle.Q = Q1c_new
        coupling_net.reset_network(xc_old,formulation=formulation)
        with HiddenPrints():
            xc_new,itersc,err_vec_c,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = coupling_net.solve_network(tol_sub,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        errors_coupling[iter_nr] = err_vec_c
        q1c_new = xc_new[0]
        P0c_new = xc_new[1]

        # Update / set the full new boundary values
        xc_full_new = np.array([q0c_new, q1c_new, P0c_new, P1c_new, Q0c_new, Q1c_new])

        # Determine error
        if error_measure == 'F':
            error = error_measure_dd_ge(gas_net,elec_net,coupling_net,xg_new,xe_new,xc_new,xc_full_new,scale_var=scale_var)
        elif error_measure == 'dE':
            error = np.linalg.norm(Dxc_full.dot(xc_full_new) - Dxc_full.dot(xc_full_old))
        err_vec.append(error)
        iter_nr += 1

    # print results
    het_net, gas_net, elec_net = create_mes_ge_network()
    het_net.initialize()
    x_new = np.concatenate((xg_new,xe_new,xc_full_new))
    with HiddenPrints():
        p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(x_new,formulation=formulation)
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
def example_mes_ge_load_flow_single_network_dd():
    # Given + When
    _, _, _, x_sol_full, _, _, _, _, _ = run_mes_ge_load_flow_dd(max_iter=250,error_measure='F')

    # Then
    _, _, _, x_sol_expected, _, _ = run_mes_ge_load_flow()
    rel_tol = 1e-4 # The coupling (active) powers differ about 100 W (the solution is around 1MW), so they're only 'close' for a rel_tol larger than the default
    atol = rel_tol
    assert np.allclose(x_sol_full,x_sol_expected,atol=atol,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning")
def example_mes_ge_load_flow_single_network_dd_scaled():
    # Given + When
    _, _, _, x_sol_full, _, _, _, _, _ = run_mes_ge_load_flow_dd(max_iter=250,error_measure='F',scale_var='matrix')

    # Then
    _, _, _, x_sol_expected, _, _ = run_mes_ge_load_flow()
    rel_tol = 1e-5 # The coupling (active) powers differ about 1 W (the solution is around 1MW), so they're only 'close' for a rel_tol larger than the default
    atol = rel_tol*1e-1
    assert np.allclose(x_sol_full,x_sol_expected,atol=atol,rtol=rel_tol)

def run_mes_eh_load_flow_dd(max_iter=max_iter_outer,error_measure='F',scale_var=None,c_hl=False,dummy_links=True,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Steady-state load flow analysis of multi-carrier electricity-heat network, using domain decomposition, without scaling. The default values are used for initialization. Toc1 is assumed known (both for the heat network and for the coupling network.)
    Currently not implemented if the coupling is not explicitly represented by extra links in the heat network (i.e. not implemented for c_hl=False).

    Uses the already implemented solvers for the single-carrier network. Between single-carrier solves, the boundary values are given to the other domains.

    Parameters
    ----------
    error_measure : str, optional
        Determines which error measure to use for the outer iterations. Options are 'F', for the system of equations, or 'dE' for the difference between coupling energies between consecutive iterations. Default if 'F'. NB: for 'dE', the vector with errors is one longer, since an error for iteration 0 cannot be defined.
    """
    # create networks
    elec_net = create_electrical_network()
    heat_net = create_heat_network(c_hl=c_hl,dummy_links=dummy_links)
    coupling_net = create_coupling_eh_single_network()

    # initialize networks, and set initial guesses
    xe_new = initialize_electrical_network(elec_net)
    xh_new = initialize_heat_network(heat_net,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation.get('heat'))
    xc_new = initialize_coupling_eh_single_network(coupling_net) #[P0c,phi1c]

    # determine initial error. Determine required scaling matrices
    print('\nRunning load flow with DD, electricity-heat, error {}'.format(error_measure))
    err_vec = list()
    if error_measure == 'F':
        error = error_measure_dd_eh(elec_net,heat_net,coupling_net,xe_new,xh_new,xc_new,c_hl=c_hl,dummy_links=dummy_links)
        err_vec.append(error)
    elif error_measure == 'dE':
        error = 1 # set some error which is bigger than the tolerance
    else:
        raise ValueError('enter valid error measure')
    if scale_var == 'matrix':
        xb_Sc = scale_var_params.get('Sbase')*np.ones(1)
        xb_phic = scale_var_params.get('phibase')*np.ones(1)
        Dxc = sps.diags(1/np.concatenate((xb_Sc,xb_phic)))
    else:
        Dxc = sps.eye(len(xc_new))
    # iterate
    iter_nr = 0 # I assume solution is not found at 1 iteration
    errors_elec = dict()
    errors_heat = dict()
    errors_coupling = dict()
    while error > tol and iter_nr < max_iter:
        xe_old = xe_new
        xh_old = xh_new
        xc_old = xc_new

        # Electricity network. Update boundary, then solve, then update the boundary values
        elec_net = update_electrical_network_bc_eh(elec_net,xc_old)
        elec_net.reset_network(xe_old)
        with HiddenPrints():
            xe_new,iterse,err_vece,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol_sub,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        errors_elec[iter_nr] = err_vece

        # Heat network. Update boundary, then solve, then update the boundary values
        heat_net = update_heat_network_bc(heat_net,xc_old,c_hl=c_hl,dummy_links=dummy_links)
        heat_net.reset_network(xh_old,formulation=formulation.get('heat'))
        with HiddenPrints():
            xh_new,itersh,err_vech,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol_sub,max_iter,solver='NR',formulation=formulation.get('heat'))
        errors_heat[iter_nr] = err_vech

        # Coupling network. Update boundary, then solve, then update the boundary values
        coupling_net = update_coupling_eh_single_network_bc(coupling_net,elec_net,heat_net,xe_new,xh_new,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation)
        coupling_net.reset_network(xc_old,formulation=formulation)
        with HiddenPrints():
            xc_new,itersc,err_vec_c,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = coupling_net.solve_network(tol_sub,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        errors_coupling[iter_nr] = err_vec_c

        # Determine error
        if error_measure == 'F':
            error = error_measure_dd_eh(elec_net,heat_net,coupling_net,xe_new,xh_new,xc_new,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation,scale_var=scale_var)
        elif error_measure == 'dE':
            error = np.linalg.norm(Dxc.dot(xc_new) - Dxc.dot(xc_old))
        err_vec.append(error)
        iter_nr += 1

    # print results
    elec_net = update_electrical_network_bc_eh(elec_net,xc_new)
    elec_net.reset_network(xe_new)
    delta_vec,V_mag_vec,S_inj,P_edge,Q_edge = elec_net.update_full(xe_new)
    Q0c = -elec_net.nodes[0].half_links[0].get_Q()
    P0c = -elec_net.nodes[0].half_links[0].get_Q()
    elec_halflink1 = elec_net.nodes[1].half_links[0]  # slack, so this is one half link
    P1c = -(elec_halflink1.get_P() - P1_load) #>0, this is already the coupling energy. P1_load is assumed known
    Q1c = -(elec_halflink1.get_Q() - Q1_load)
    print('Solution after {} iterations (final error = {:.4e}):'.format(iter_nr,err_vec[-1]))
    print('Single-carrier electrical network:')
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Pc = {} MW'.format([P0c/MW, P1c/MW]))
    print('Qc = {} MW'.format([Q0c/MW,Q1c/MW]))
    heat_net = update_heat_network_bc(heat_net,xc_new,c_hl=c_hl,dummy_links=dummy_links)
    heat_net.reset_network(xh_new,formulation=formulation.get('heat'))
    m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(xh_new,formulation=formulation.get('heat'))
    print('Single-carrier heat network:')
    print('p heat = {} m'.format(p_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {}'.format(m_hl_vec))
    print('Ts hl = {}'.format(Ts_hl_vec))
    print('Tr hl = {}'.format(Tr_hl_vec))
    print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
    print('Coupling electricity-heat network:')
    coupling_net = update_coupling_eh_single_network_bc(coupling_net,elec_net,heat_net,xe_new,xh_new,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation)
    coupling_net.reset_network(xc_new,formulation=formulation)
    _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.update_full(xc_new,formulation=formulation)
    print('qc = {} kg/s'.format(qc_vec))
    print('Pc = {} MW'.format([Pc/MW for Pc in Pc_vec]))
    print('Qc = {} MW'.format([Qc/MW for Qc in Qc_vec]))
    print('mc = {} kg/s'.format(mc_vec))
    print('phic = {} MW'.format([phic/MW for phic in phic_vec]))
    print('Tsc = {} C'.format(Tsc_vec))
    print('Trc = {} C'.format(Trc_vec))

    xmes_new = xseparate_to_xmes_eh(xe_new,xh_new,xc_new,elec_net,heat_net,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation)
    het_net, elec_net, heat_net = create_mes_eh_network()
    het_net.initialize()
    with HiddenPrints():
        p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_new,formulation=formulation)
    print('Solution in MES (using DD):')
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
    return xe_new, xh_new, xc_new, xmes_new, iter_nr, err_vec, errors_elec, errors_heat, errors_coupling

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_eh_load_flow_single_network_dd_extra_links_heat():
    # Given + When
    _, _, _, x_sol_full, _, _, _, _, _ = run_mes_eh_load_flow_dd(max_iter=1000,error_measure='F',c_hl=True,dummy_links=False)

    # Then
    _, _, _, x_sol_expected, _, _ = run_mes_eh_load_flow()
    assert np.allclose(x_sol_full,x_sol_expected)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_mes_eh_load_flow_single_network_dd_extra_links_heat_scaled():
    # Given + When
    _, _, _, x_sol_full, _, _, _, _, _ = run_mes_eh_load_flow_dd(max_iter=500,error_measure='F',c_hl=True,dummy_links=True,scale_var='matrix')

    # Then
    _, _, _, x_sol_expected, _, _ = run_mes_eh_load_flow()
    assert np.allclose(x_sol_full,x_sol_expected)

def run_mes_eh_load_flow_dd_Toc1_unknown(max_iter=max_iter_outer,error_measure='F',scale_var=None,c_hl=False,dummy_links=True,formulation={'gas':'full','elec':'complex_power','heat':'half_link_flow','het':None}):
    """Steady-state load flow analysis of multi-carrier electricity-heat network, using domain decomposition, without scaling. The default values are used for initialization. Toc1 is assumed unknown in the coupling network.
    Currently not implemented if the coupling is not explicitly represented by extra links in the heat network (i.e. not implemented for c_hl=False).

    Uses the already implemented solvers for the single-carrier network. Between single-carrier solves, the boundary values are given to the other domains.

    Parameters
    ----------
    error_measure : str, optional
        Determines which error measure to use for the outer iterations. Options are 'F', for the system of equations, or 'dE' for the difference between coupling energies between consecutive iterations. Default if 'F'. NB: for 'dE', the vector with errors is one longer, since an error for iteration 0 cannot be defined.
    """
    # create networks
    elec_net = create_electrical_network()
    heat_net = create_heat_network(c_hl=c_hl,dummy_links=dummy_links)
    coupling_net = create_coupling_eh_single_network_Toc1_unknown()

    # initialize networks, and set initial guesses
    xe_new = initialize_electrical_network(elec_net)
    xh_new = initialize_heat_network(heat_net,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation.get('heat'))
    xc_new = initialize_coupling_eh_single_network_Toc1_unknown(coupling_net) #[P0c,phi1c,Ts1c]

    # determine initial error. Determine required scaling matrices
    print('\nRunning load flow with DD, electricity-heat, error {}'.format(error_measure))
    err_vec = list()
    if error_measure == 'F':
        error = error_measure_dd_eh_Toc1_unknown(elec_net,heat_net,coupling_net,xe_new,xh_new,xc_new,c_hl=c_hl,dummy_links=dummy_links)
        err_vec.append(error)
    elif error_measure == 'dE':
        error = 1 # set some error which is bigger than the tolerance
    else:
        raise ValueError('enter valid error measure')
    if scale_var == 'matrix':
        xb_Sc = scale_var_params.get('Sbase')*np.ones(1)
        xb_phic = scale_var_params.get('phibase')*np.ones(1)
        xb_Toc = scale_var_params.get('Tbase')*np.ones(1)
        Dxc = sps.diags(1/np.concatenate((xb_Sc,xb_phic,xb_Toc)))
    else:
        Dxc = sps.eye(len(xc_new))
    # iterate
    iter_nr = 0 # I assume solution is not found at 1 iteration
    errors_elec = dict()
    errors_heat = dict()
    errors_coupling = dict()
    while error > tol and iter_nr < max_iter:
        xe_old = xe_new
        xh_old = xh_new
        xc_old = xc_new

        # Electricity network. Update boundary, then solve, then update the boundary values
        elec_net = update_electrical_network_bc_eh(elec_net,xc_old)
        elec_net.reset_network(xe_old)
        with HiddenPrints():
            xe_new,iterse,err_vece,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol_sub,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        errors_elec[iter_nr] = err_vece

        # Heat network. Update boundary, then solve, then update the boundary values
        heat_net = update_heat_network_bc_Toc1_unknown(heat_net,xc_old,c_hl=c_hl,dummy_links=dummy_links)
        heat_net.reset_network(xh_old,formulation=formulation.get('heat'))
        with HiddenPrints():
            xh_new,itersh,err_vech,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol_sub,max_iter,solver='NR',formulation=formulation.get('heat'))
        errors_heat[iter_nr] = err_vech

        # Coupling network. Update boundary, then solve, then update the boundary values
        coupling_net = update_coupling_eh_single_network_bc_Toc1_unknown(coupling_net,elec_net,heat_net,xe_new,xh_new,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation)
        coupling_net.reset_network(xc_old,formulation=formulation)
        with HiddenPrints():
            xc_new,itersc,err_vec_c,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = coupling_net.solve_network(tol_sub,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        errors_coupling[iter_nr] = err_vec_c

        # Determine error
        if error_measure == 'F':
            error = error_measure_dd_eh_Toc1_unknown(elec_net,heat_net,coupling_net,xe_new,xh_new,xc_new,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation,scale_var=scale_var)
        elif error_measure == 'dE':
            error = np.linalg.norm(Dxc.dot(xc_new) - Dxc.dot(xc_old))
        err_vec.append(error)
        iter_nr += 1

    # print results
    elec_net = update_electrical_network_bc_eh(elec_net,xc_new)
    elec_net.reset_network(xe_new)
    delta_vec,V_mag_vec,S_inj,P_edge,Q_edge = elec_net.update_full(xe_new)
    Q0c = -elec_net.nodes[0].half_links[0].get_Q()
    P0c = -elec_net.nodes[0].half_links[0].get_Q()
    elec_halflink1 = elec_net.nodes[1].half_links[0]  # slack, so this is one half link
    P1c = -(elec_halflink1.get_P() - P1_load) #>0, this is already the coupling energy. P1_load is assumed known
    Q1c = -(elec_halflink1.get_Q() - Q1_load)
    print('Solution after {} iterations (final error = {:.4e}):'.format(iter_nr,err_vec[-1]))
    print('Single-carrier electrical network:')
    print('delta = {}'.format(delta_vec))
    print('|V| = {} V'.format(V_mag_vec))
    print('|V| = {} p.u.'.format(V_mag_vec/Vbase))
    print('P edge = {} MW'.format(P_edge/MW))
    print('Q edge = {} MW'.format(Q_edge/MW))
    print('S nodal inj = {} MW'.format(S_inj/MW))
    print('P hl = {} MW'.format([hl.P/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Q hl = {} MW'.format([hl.Q/MW for node in elec_net.get_nodes() for hl in node.get_half_links()]))
    print('Pc = {} MW'.format([P0c/MW, P1c/MW]))
    print('Qc = {} MW'.format([Q0c/MW,Q1c/MW]))
    heat_net = update_heat_network_bc_Toc1_unknown(heat_net,xc_new,c_hl=c_hl,dummy_links=dummy_links)
    heat_net.reset_network(xh_new,formulation=formulation.get('heat'))
    m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.update_full(xh_new,formulation=formulation.get('heat'))
    print('Single-carrier heat network:')
    print('p heat = {} m'.format(p_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {}'.format(m_hl_vec))
    print('Ts hl = {}'.format(Ts_hl_vec))
    print('Tr hl = {}'.format(Tr_hl_vec))
    print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
    print('Coupling electricity-heat network:')
    coupling_net = update_coupling_eh_single_network_bc_Toc1_unknown(coupling_net,elec_net,heat_net,xe_new,xh_new,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation)
    coupling_net.reset_network(xc_new,formulation=formulation)
    _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.update_full(xc_new,formulation=formulation)
    print('qc = {} kg/s'.format(qc_vec))
    print('Pc = {} MW'.format([Pc/MW for Pc in Pc_vec]))
    print('Qc = {} MW'.format([Qc/MW for Qc in Qc_vec]))
    print('mc = {} kg/s'.format(mc_vec))
    print('phic = {} MW'.format([phic/MW for phic in phic_vec]))
    print('Tsc = {} C'.format(Tsc_vec))
    print('Trc = {} C'.format(Trc_vec))

    xmes_new = xseparate_to_xmes_eh_Toc1_unknown(xe_new,xh_new,xc_new,elec_net,heat_net,c_hl=c_hl,dummy_links=dummy_links,formulation=formulation)
    het_net, elec_net, heat_net = create_mes_eh_network_Toc1_unknown()
    het_net.initialize()

    with HiddenPrints():
        p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(xmes_new,formulation=formulation)
    print('Solution in MES (using DD):')
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
    return xe_new, xh_new, xc_new, xmes_new, iter_nr, err_vec, errors_elec, errors_heat, errors_coupling

def inner_iters(ax,err_vec,errors_gas=dict(),errors_elec=dict(),errors_heat=dict(),errors_coupling=dict(),error_measure='F'):
    """Plot the convergence per inner iterations"""
    iter_gas = 0
    iter_elec = 0
    iter_heat = 0
    iter_coupling = 0
    if err_vec[-1] == 0:
        err_vec[-1] = art_zero # semilogy doesn't like an actual 0
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
            if err_vec_gas[-1] == 0:
                err_vec_gas[-1] = art_zero # semilogy doesn't like an actual 0
            if iter_nr == 0:
                ax.semilogy([k_gas+iter_gas for k_gas in range(len(err_vec_gas))],err_vec_gas,marker=marker,color='tab:green',linestyle='--',alpha=.4,label=label+' gas, '+error_measure)
            else:
                ax.semilogy([k_gas+iter_gas for k_gas in range(len(err_vec_gas))],err_vec_gas,marker=marker,color='tab:green',linestyle='--',alpha=.4)
        else:
            err_vec_gas = list()

        if errors_elec.get(iter_nr):
            err_vec_elec = errors_elec.get(iter_nr)
            if err_vec_elec[-1] == 0:
                err_vec_elec[-1] = art_zero # semilogy doesn't like an actual 0
            iter_elec += len(err_vec_gas)
            if errors_gas:
                iter_elec -= 1
            if iter_nr == 0:
                ax.semilogy([k_elec+iter_elec for k_elec in range(len(err_vec_elec))],err_vec_elec,marker=marker,color='tab:red',linestyle='--',alpha=.4,label=label+' elec, '+error_measure)
            else:
                ax.semilogy([k_elec+iter_elec for k_elec in range(len(err_vec_elec))],err_vec_elec,marker=marker,color='tab:red',linestyle='--',alpha=.4)
        else:
            err_vec_elec = list()

        if errors_heat.get(iter_nr):
            err_vec_heat = errors_heat.get(iter_nr)
            if err_vec_heat[-1] == 0:
                err_vec_heat[-1] = art_zero # semilogy doesn't like an actual 0
            iter_heat += len(err_vec_gas) + len(err_vec_elec)
            if errors_gas:
                iter_heat -= 1
            if errors_elec:
                iter_heat -= 1
            if iter_nr == 0:
                ax.semilogy([k_heat+iter_heat for k_heat in range(len(err_vec_heat))],err_vec_heat,marker=marker,color='tab:blue',linestyle='--',alpha=.4,label=label+' heat, '+error_measure)
            else:
                ax.semilogy([k_heat+iter_heat for k_heat in range(len(err_vec_heat))],err_vec_heat,marker=marker,color='tab:blue',linestyle='--',alpha=.4)
        else:
            err_vec_heat = list()

        if errors_coupling.get(iter_nr):
            err_vec_coupling = errors_coupling.get(iter_nr)
            if err_vec_coupling[-1] == 0:
                err_vec_coupling[-1] = art_zero # semilogy doesn't like an actual 0
            iter_coupling += len(err_vec_gas) + len(err_vec_elec) + len(err_vec_heat)
            if errors_gas:
                iter_coupling -= 1
            if errors_elec:
                iter_coupling -= 1
            if errors_heat:
                iter_coupling -= 1
            if iter_nr == 0:
                ax.semilogy([k_coupling+iter_coupling for k_coupling in range(len(err_vec_coupling))],err_vec_coupling,marker=marker,color='tab:gray',linestyle='--',alpha=.4,label=label+' coupling, '+error_measure)
            else:
                ax.semilogy([k_coupling+iter_coupling for k_coupling in range(len(err_vec_coupling))],err_vec_coupling,marker=marker,color='tab:gray',linestyle='--',alpha=.4)
        else:
            err_vec_coupling = list()

        iter_gas += len(err_vec_gas) + len(err_vec_elec) + len(err_vec_heat) + len(err_vec_coupling)
        iter_elec += len(err_vec_elec) + len(err_vec_heat) + len(err_vec_coupling)
        iter_heat += len(err_vec_heat) + len(err_vec_coupling)
        iter_coupling += len(err_vec_coupling)
        if errors_gas and len(err_vec_gas)>1:
            iter_gas -= 1
        if errors_elec and len(err_vec_elec)>1:
            iter_gas -= 1
            iter_elec -= 1
        if errors_heat and len(err_vec_heat)>1:
            iter_gas -= 1
            iter_elec -= 1
            iter_heat -= 1
        if errors_coupling and len(err_vec_coupling)>1:
            iter_gas -= 1
            iter_elec -= 1
            iter_heat -= 1
            iter_coupling -= 1
        iter_outer.append(np.max([iter_gas,iter_elec,iter_heat,iter_coupling]))
    ax.semilogy(iter_outer,err_vec,marker=marker,color='tab:orange',label=label+', '+error_measure)
    return iter_gas, iter_elec, iter_heat, iter_coupling, iter_outer

def plot_convergence(ax,err_vec,max_data_points,label):
    """Plot the error per iteration. Only the number of data point equal to max_data_points+1 are plotted. These data points are equally distributed over the error vector."""
    marker = '.'
    if len(err_vec) > max_data_points:
        ax.semilogy(range(1,len(err_vec)+1,len(err_vec)//max_data_points),[err_vec[ind] for ind in range(0,len(err_vec),len(err_vec)//max_data_points)],marker=marker,label=label)
    else:
        ax.semilogy(range(1,len(err_vec)+1),err_vec,marker=marker,label=label)

def plot_convergence_order(ax,err_vec,max_data_points,label):
    """Plot the convergence order. That is, plot the (normalized) error of iteration k+1 vs the (normalized) error of iteration k. Only the number of data point equal to max_data_points+1 are plotted. These data points are equally distributed over the error vector."""
    if len(err_vec) > max_data_points:
        err_vec_k=  [err_vec[ind] for ind in range(0,len(err_vec)-1,(len(err_vec)-1)//max_data_points)]
        err_vec_kplus1=  [err_vec[ind+1] for ind in range(0,len(err_vec)-1,(len(err_vec)-1)//max_data_points)]
    else:
        err_vec_k = err_vec[:-1]
        err_vec_kplus1 = err_vec[1:]
    ax.loglog(err_vec_k/err_vec[0],err_vec_kplus1/err_vec[0],marker='.',label=label)

def layout_convergence(ax,tol,tol_sub,max_iters):
    ax.semilogy([0,max_iters+1],[tol,tol],'k:',label='tolerance')
    ax.semilogy([0,max_iters+1],[tol_sub,tol_sub],'k-.',label='tolerance subnetworks')
    ax.semilogy([0,max_iters+1],[art_zero,art_zero],'k--',label='artifical 0')
    xmin = 0
    xmax = max_iters
    xticks = range(xmin,xmax+1,max(1,int(xmax/10))) # make sure the xticks are integers
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

def compare_DD_ge(max_iter=max_iter_outer,plot_details=False,save_fig=False,dir_path=None,max_data_points=100):
    """Compare the overall convergence of DD for the gas-electricity network with solving the integrated system of equations. Both error measures 'dE' and 'F' are used. Convergence plots are made.

    Parameters
    ----------------
    plot_details : bool, optional
        If True, additional plots are made with detailed convergence. That is, the convergence of the inner NR iterations of the single-carrier networks is plotted, as well as the overall convergence of DD. Default is False.
    """
    if save_fig:
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
        "ytick.labelsize": 10
        }
        mpl.rcParams.update(pgf_with_latex)

    # unscaled
    het_net_unsc, _, _, x_sol_unsc, iters_unsc, err_vec_unsc = run_mes_ge_load_flow(plot_top=False,plot_jac=False,plot_sol=False)
    xg_dE_unsc, xe_dE_unsc, xc_dE_unsc, x_dE_unsc, iters_dE_unsc, err_vec_dE_unsc, errors_gas_dE_unsc, errors_elec_dE_unsc, errors_coupling_dE_unsc = run_mes_ge_load_flow_dd(max_iter=max_iter,error_measure='dE')
    xg_F_unsc, xe_F_unsc, xc_F_unsc, x_F_unsc, iters_F_unsc, err_vec_F_unsc, errors_gas_F_unsc, errors_elec_F_unsc, errors_coupling_F_unsc = run_mes_ge_load_flow_dd(max_iter=max_iter,error_measure='F')
    # scaled
    het_net_sc, _, _, x_sol_sc, iters_sc, err_vec_sc = run_mes_ge_load_flow(plot_top=False,plot_jac=False,plot_sol=False,scale_var='matrix')
    xg_dE_sc, xe_dE_sc, xc_dE_sc, x_dE_sc, iters_dE_sc, err_vec_dE_sc, errors_gas_dE_sc, errors_elec_dE_sc, errors_coupling_dE_sc = run_mes_ge_load_flow_dd(max_iter=max_iter,scale_var='matrix',error_measure='dE')
    xg_F_sc, xe_F_sc, xc_F_sc, x_F_sc, iters_F_sc, err_vec_F_sc, errors_gas_F_sc, errors_elec_F_sc, errors_coupling_F_sc = run_mes_ge_load_flow_dd(max_iter=max_iter,scale_var='matrix',error_measure='F')

    print('Iterations unscaled: integrated system = {}, DD dE (outer iters) = {}, DD F (outer iters) = {}'.format(iters_unsc,iters_dE_unsc,iters_F_unsc))
    print('Final errors unscaled: integrated system = {}, DD dE = {}, DD F = {}'.format(err_vec_unsc[-1],err_vec_dE_unsc[-1],err_vec_F_unsc[-1]))
    print('Iterations scaled: integrated system = {}, DD dE (outer iters) = {}, DD F (outer iters) = {}'.format(iters_sc,iters_dE_sc,iters_F_sc))
    print('Final errors scaled: integrated system = {}, DD dE = {}, DD F = {}'.format(err_vec_sc[-1],err_vec_dE_sc[-1],err_vec_F_sc[-1]))

    # plot convergence
    print('\nCreating convergence plot gas-electricity, unscaled')
    if plot_details:
        fig = plt.figure('Convergence_plot_ge_dE_unscaled_tol{}_subtol{}_detailed'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
        ax_dE_unsc_det = fig.gca()
        ax_dE_unsc_det.semilogy(err_vec_unsc,marker='.',label='Integrated system')
        plt.xlabel('Iteration k')
        plt.ylabel(r'Error $||\Delta E||_2$')
        inner_g_dE_unsc, inner_e_dE_unsc, _, inner_c_dE_unsc, outer_dE_unsc = inner_iters(ax_dE_unsc_det,err_vec_dE_unsc,errors_gas=errors_gas_dE_unsc,errors_elec=errors_elec_dE_unsc,errors_coupling=errors_coupling_dE_unsc,error_measure='dE')

    print('\nCreating convergence plot gas-electricity, unscaled')
    fig = plt.figure('Convergence_plot_ge_dE_unscaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_dE_unsc = fig.gca()
    plot_convergence(ax_dE_unsc,err_vec_unsc,max_data_points,'Integrated system')
    plot_convergence(ax_dE_unsc,err_vec_dE_unsc,max_data_points,'DD')
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||\Delta E||_2$')

    if plot_details:
        fig = plt.figure('Convergence_plot_ge_F_unscaled_tol{}_subtol{}_detailed'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
        ax_F_unsc_det = fig.gca()
        ax_F_unsc_det.semilogy(err_vec_unsc,marker='.',label='Integrated system')
        plt.xlabel('Iteration k')
        plt.ylabel(r'Error $||F||_2$')
        inner_g_F_unsc, inner_e_F_unsc, _, inner_c_F_unsc, outer_F_unsc = inner_iters(ax_F_unsc_det,err_vec_F_unsc,errors_gas=errors_gas_F_unsc,errors_elec=errors_elec_F_unsc,errors_coupling=errors_coupling_F_unsc,error_measure='F')

    fig = plt.figure('Convergence_plot_ge_F_unscaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_F_unsc = fig.gca()
    plot_convergence(ax_F_unsc,err_vec_unsc,max_data_points,'Integrated system')
    plot_convergence(ax_F_unsc,err_vec_F_unsc,max_data_points,'DD')
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F||_2$')

    print('\nCreating convergence plot gas-electricity, scaled')
    if plot_details:
        fig = plt.figure('Convergence_plot_ge_dE_scaled_tol{}_subtol{}_detailed'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
        ax_dE_sc_det = fig.gca()
        ax_dE_sc_det.semilogy(err_vec_sc,marker='.',label='Integrated system')
        plt.xlabel('Iteration k')
        plt.ylabel(r'Error $||\Delta E||_2$')
        inner_g_dE_sc, inner_e_dE_sc, _, inner_c_dE_sc, outer_dE_sc = inner_iters(ax_dE_sc_det,err_vec_dE_sc,errors_gas=errors_gas_dE_sc,errors_elec=errors_elec_dE_sc,errors_coupling=errors_coupling_dE_sc,error_measure='dE')

    fig = plt.figure('Convergence_plot_ge_dE_scaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_dE_sc = fig.gca()
    plot_convergence(ax_dE_sc,err_vec_sc,max_data_points,'Integrated system')
    plot_convergence(ax_dE_sc,err_vec_dE_sc,max_data_points,'DD')
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||\Delta E||_2$')

    if plot_details:
        fig = plt.figure('Convergence_plot_ge_F_scaled_tol{}_subtol{}_detailed'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
        ax_F_sc_det = fig.gca()
        ax_F_sc_det.semilogy(err_vec_sc,marker='.',label='Integrated system')
        plt.xlabel('Iteration k')
        plt.ylabel(r'Error $||F||_2$')
        inner_g_F_sc, inner_e_F_sc, _, inner_c_F_sc, outer_F_sc = inner_iters(ax_F_sc_det,err_vec_F_sc,errors_gas=errors_gas_F_sc,errors_elec=errors_elec_F_sc,errors_coupling=errors_coupling_F_sc,error_measure='F')

    fig = plt.figure('Convergence_plot_ge_F_scaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_F_sc = fig.gca()
    plot_convergence(ax_F_sc,err_vec_sc,max_data_points,'Integrated system')
    plot_convergence(ax_F_sc,err_vec_F_sc,max_data_points,'DD')
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F||_2$')

    print('\nCreating convergence order plot gas-electricity, scaled')
    fig = plt.figure('Convergence_order_ge_dE_scaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_ord_dE_sc = fig.gca()
    plot_convergence_order(ax_ord_dE_sc,err_vec_sc,max_data_points,'Integrated system')
    plot_convergence_order(ax_ord_dE_sc,err_vec_dE_sc,max_data_points,'DD')
    plt.xlabel(r'$||D_F F(x^k)||_2 / ||D_F F(x^0)||_2$')
    plt.ylabel(r'$||D_F F(x^{k+1})||_2 / ||D_F F(x^0)||_2$')
    x_min_ord_dE_sc,x_max_ord_dE_sc = ax_ord_dE_sc.get_xlim()
    x_slope_dE_sc = np.linspace(x_min_ord_dE_sc,x_max_ord_dE_sc)
    y_slope2_dE_sc = x_slope_dE_sc**2
    ax_ord_dE_sc.loglog(x_slope_dE_sc,x_slope_dE_sc,linestyle='--',color='k',label='slope 1')
    ax_ord_dE_sc.loglog(x_slope_dE_sc,y_slope2_dE_sc,linestyle='-.',color='k',label='slope 2')
    ax_ord_dE_sc.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_ord_dE_sc.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_ord_dE_sc.legend()

    fig = plt.figure('Convergence_order_ge_F_scaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_ord_F_sc = fig.gca()
    plot_convergence_order(ax_ord_F_sc,err_vec_sc,max_data_points,'Integrated system')
    plot_convergence_order(ax_ord_F_sc,err_vec_F_sc,max_data_points,'DD')
    plt.xlabel(r'$||D_F F(x^k)||_2 / ||D_F F(x^0)||_2$')
    plt.ylabel(r'$||D_F F(x^{k+1})||_2 / ||D_F F(x^0)||_2$')
    x_min_ord_F_sc,x_max_ord_F_sc = ax_ord_F_sc.get_xlim()
    x_slope_F_sc = np.linspace(x_min_ord_F_sc,x_max_ord_F_sc)
    y_slope2_F_sc = x_slope_F_sc**2
    ax_ord_F_sc.loglog(x_slope_F_sc,x_slope_F_sc,linestyle='--',color='k',label='slope 1')
    ax_ord_F_sc.loglog(x_slope_F_sc,y_slope2_F_sc,linestyle='-.',color='k',label='slope 2')
    ax_ord_F_sc.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_ord_F_sc.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_ord_F_sc.legend()

    if plot_details:
        max_iters_used_det = np.max([iters_unsc, iters_dE_unsc, iters_F_unsc, iters_sc, iters_dE_sc, iters_F_sc,outer_dE_unsc[-1],outer_F_unsc[-1],outer_dE_sc[-1],outer_F_sc[-1]])
        layout_convergence(ax_dE_unsc_det,tol,tol_sub,max_iters_used_det)
        layout_convergence(ax_F_unsc_det,tol,tol_sub,max_iters_used_det)
        layout_convergence(ax_dE_sc_det,tol,tol_sub,max_iters_used_det)
        layout_convergence(ax_F_sc_det,tol,tol_sub,max_iters_used_det)
    max_iters_used = np.max([iters_unsc, iters_dE_unsc, iters_F_unsc, iters_sc, iters_dE_sc, iters_F_sc])
    layout_convergence(ax_dE_unsc,tol,tol_sub,max_iters_used)
    layout_convergence(ax_F_unsc,tol,tol_sub,max_iters_used)
    layout_convergence(ax_dE_sc,tol,tol_sub,max_iters_used)
    layout_convergence(ax_F_sc,tol,tol_sub,max_iters_used)

def compare_DD_eh(max_iter=max_iter_outer,plot_details=False,save_fig=False,dir_path=None,max_data_points=100):
    """Compare the overall convergence of DD for the electricity-heat network with solving the integrated system of equations. Both error measures 'dE' and 'F' are used. Convergence plots are made.

    Parameters
    ----------------
    plot_details : bool, optional
        If True, additional plots are made with detailed convergence. That is, the convergence of the inner NR iterations of the single-carrier networks is plotted, as well as the overall convergence of DD. Default is False.
    """
    if save_fig:
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
        "ytick.labelsize": 10
        }
        mpl.rcParams.update(pgf_with_latex)

    c_hl = True
    dummy_links = True
    # unscaled
    het_net_unsc, _, _, x_sol_unsc, iters_unsc, err_vec_unsc = run_mes_eh_load_flow(plot_top=False,plot_jac=False,plot_sol=False)
    xe_dE_unsc, xh_dE_unsc, xc_dE_unsc, x_dE_unsc, iters_dE_unsc, err_vec_dE_unsc, errors_gas_dE_unsc, errors_elec_dE_unsc, errors_coupling_dE_unsc = run_mes_eh_load_flow_dd(max_iter=max_iter,error_measure='dE',c_hl=c_hl,dummy_links=dummy_links)
    xe_F_unsc, xh_F_unsc, xc_F_unsc, x_F_unsc, iters_F_unsc, err_vec_F_unsc, errors_gas_F_unsc, errors_elec_F_unsc, errors_coupling_F_unsc = run_mes_eh_load_flow_dd(max_iter=max_iter,error_measure='F',c_hl=c_hl,dummy_links=dummy_links)
    # scaled
    het_net_sc, _, _, x_sol_sc, iters_sc, err_vec_sc = run_mes_eh_load_flow(plot_top=False,plot_jac=False,plot_sol=False,scale_var='matrix')
    xe_dE_sc, xh_dE_sc, xc_dE_sc, x_dE_sc, iters_dE_sc, err_vec_dE_sc, errors_gas_dE_sc, errors_elec_dE_sc, errors_coupling_dE_sc = run_mes_eh_load_flow_dd(max_iter=max_iter,scale_var='matrix',error_measure='dE',c_hl=c_hl,dummy_links=dummy_links)
    xe_F_sc, xh_F_sc, xc_F_sc, x_F_sc, iters_F_sc, err_vec_F_sc, errors_gas_F_sc, errors_elec_F_sc, errors_coupling_F_sc = run_mes_eh_load_flow_dd(max_iter=max_iter,scale_var='matrix',error_measure='F',c_hl=c_hl,dummy_links=dummy_links)

    # scaled, outflow temperature of couplin 1 (Toc1 / Tsc1) unknown
    het_net_sc_Tunknown, _, _, x_sol_sc_Tunknown, iters_sc_Tunknown, err_vec_sc_Tunknown = run_mes_eh_load_flow_Toc1_unknown(plot_top=False,plot_jac=False,plot_sol=False,scale_var='matrix')
    xe_dE_sc_Tunknown, xh_dE_sc_Tunknown, xc_dE_sc_Tunknown, x_dE_sc_Tunknown, iters_dE_sc_Tunknown, err_vec_dE_sc_Tunknown, errors_gas_dE_sc_Tunknown, errors_elec_dE_sc_Tunknown, errors_coupling_dE_sc_Tunknown = run_mes_eh_load_flow_dd_Toc1_unknown(max_iter=max_iter,scale_var='matrix',error_measure='dE',c_hl=c_hl,dummy_links=dummy_links)
    xe_F_sc_Tunknown, xh_F_sc_Tunknown, xc_F_sc_Tunknown, x_F_sc_Tunknown, iters_F_sc_Tunknown, err_vec_F_sc_Tunknown, errors_gas_F_sc_Tunknown, errors_elec_F_sc_Tunknown, errors_coupling_F_sc_Tunknown = run_mes_eh_load_flow_dd_Toc1_unknown(max_iter=max_iter,scale_var='matrix',error_measure='F',c_hl=c_hl,dummy_links=dummy_links)

    # print some results
    print('Iterations unscaled: integrated system = {}, DD dE (outer iters) = {}, DD F (outer iters) = {}'.format(iters_unsc,iters_dE_unsc,iters_F_unsc))
    print('Final errors unscaled: integrated system = {}, DD dE = {}, DD F = {}'.format(err_vec_unsc[-1],err_vec_dE_unsc[-1],err_vec_F_unsc[-1]))
    print('Iterations scaled: integrated system = {}, DD dE (outer iters) = {}, DD F (outer iters) = {}'.format(iters_sc,iters_dE_sc,iters_F_sc))
    print('Final errors scaled: integrated system = {}, DD dE = {}, DD F = {}'.format(err_vec_sc[-1],err_vec_dE_sc[-1],err_vec_F_sc[-1]))
    print('Iterations scaled, Toc1 unknown: integrated system = {}, DD dE (outer iters) = {}, DD F (outer iters) = {}'.format(iters_sc_Tunknown,iters_dE_sc_Tunknown,iters_F_sc_Tunknown))
    print('Final errors scaled, Toc1 unknown: integrated system = {}, DD dE = {}, DD F = {}'.format(err_vec_sc_Tunknown[-1],err_vec_dE_sc_Tunknown[-1],err_vec_F_sc_Tunknown[-1]))
    xe_sol_sc,xh_sol_sc,xc_sol_sc = xmes_to_xseparate_eh(x_sol_sc,c_hl=c_hl,dummy_links=dummy_links)
    xe_sol_sc_Tunknown,xh_sol_sc_Tunknown,xc_sol_sc_Tunknown = xmes_to_xseparate_eh(x_sol_sc_Tunknown,c_hl=c_hl,dummy_links=dummy_links)
    print('xh from x mes Toc1 known  = \n{}'.format(xh_sol_sc))
    print('xh from DD (error F) Toc1 known  = \n{}'.format(xh_F_sc))
    print('xh from x mes Toc1 unknown = \n{}'.format(xh_sol_sc_Tunknown))
    print('xh from DD (error F)Toc1 unknown = \n{}'.format(xh_F_sc_Tunknown))
    print('Solutions in sc heat network close (mes Toc1 known vs DD Toc1 known): {}'.format(np.allclose(xh_sol_sc,xh_F_sc)))
    print('Solutions in sc heat network close (mes Toc1 known vs DD Toc1 unknown): {}'.format(np.allclose(xh_sol_sc,xh_F_sc_Tunknown)))

    # plot convergence
    print('\nCreating convergence plot electricity-heat, unscaled')
    if plot_details:
        fig = plt.figure('Convergence_plot_eh_dE_unscaled_tol{}_subtol{}_detailed'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
        ax_dE_unsc_det = fig.gca()
        ax_dE_unsc_det.semilogy(err_vec_unsc,marker='.',label='Integrated system')
        plt.xlabel('Iteration k')
        plt.ylabel(r'Error $||\Delta E||_2$')
        inner_g_dE_unsc, inner_e_dE_unsc, _, inner_c_dE_unsc, outer_dE_unsc = inner_iters(ax_dE_unsc_det,err_vec_dE_unsc,errors_gas=errors_gas_dE_unsc,errors_elec=errors_elec_dE_unsc,errors_coupling=errors_coupling_dE_unsc,error_measure='dE')

    print('\nCreating convergence plot electricity-heat, unscaled')
    fig = plt.figure('Convergence_plot_eh_dE_unscaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_dE_unsc = fig.gca()
    plot_convergence(ax_dE_unsc,err_vec_unsc,max_data_points,'Integrated system')
    plot_convergence(ax_dE_unsc,err_vec_dE_unsc,max_data_points,'DD')
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||\Delta E||_2$')

    if plot_details:
        fig = plt.figure('Convergence_plot_eh_F_unscaled_tol{}_subtol{}_detailed'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
        ax_F_unsc_det = fig.gca()
        ax_F_unsc_det.semilogy(err_vec_unsc,marker='.',label='Integrated system')
        plt.xlabel('Iteration k')
        plt.ylabel(r'Error $||F||_2$')
        inner_g_F_unsc, inner_e_F_unsc, _, inner_c_F_unsc, outer_F_unsc = inner_iters(ax_F_unsc_det,err_vec_F_unsc,errors_gas=errors_gas_F_unsc,errors_elec=errors_elec_F_unsc,errors_coupling=errors_coupling_F_unsc,error_measure='F')

    fig = plt.figure('Convergence_plot_eh_F_unscaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_F_unsc = fig.gca()
    plot_convergence(ax_F_unsc,err_vec_unsc,max_data_points,'Integrated system')
    plot_convergence(ax_F_unsc,err_vec_F_unsc,max_data_points,'DD')
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F||_2$')

    print('\nCreating convergence plot electricity-heat, scaled')
    if plot_details:
        fig = plt.figure('Convergence_plot_eh_dE_scaled_tol{}_subtol{}_detailed'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
        ax_dE_sc_det = fig.gca()
        ax_dE_sc_det.semilogy(err_vec_sc,marker='.',label='Integrated system')
        plt.xlabel('Iteration k')
        plt.ylabel(r'Error $||\Delta E||_2$')
        inner_g_dE_sc, inner_e_dE_sc, _, inner_c_dE_sc, outer_dE_sc = inner_iters(ax_dE_sc_det,err_vec_dE_sc,errors_gas=errors_gas_dE_sc,errors_elec=errors_elec_dE_sc,errors_coupling=errors_coupling_dE_sc,error_measure='dE')
        inner_g_dE_sc_Tunknown, inner_e_dE_sc_Tunknown, _, inner_c_dE_sc_Tunknown, outer_dE_sc_Tunknown = inner_iters(ax_dE_sc_det,err_vec_dE_sc_Tunknown,errors_gas=errors_gas_dE_sc_Tunknown,errors_elec=errors_elec_dE_sc_Tunknown,errors_coupling=errors_coupling_dE_sc_Tunknown,error_measure='dE')

    fig = plt.figure('Convergence_plot_eh_dE_scaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_dE_sc = fig.gca()
    plot_convergence(ax_dE_sc,err_vec_sc,max_data_points,'Integrated system')
    plot_convergence(ax_dE_sc,err_vec_dE_sc,max_data_points,'DD')
    plot_convergence(ax_dE_sc,err_vec_dE_sc_Tunknown,max_data_points,'DD, Toc1 unknown')
    plot_convergence(ax_dE_sc,err_vec_sc_Tunknown,max_data_points,'Integrated system, Toc unknown')
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||\Delta E||_2$')

    if plot_details:
        fig = plt.figure('Convergence_plot_eh_F_scaled_tol{}_subtol{}_detailed'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
        ax_F_sc_det = fig.gca()
        ax_F_sc_det.semilogy(err_vec_sc,marker='.',label='Integrated system')
        plt.xlabel('Iteration k')
        plt.ylabel(r'Error $||F||_2$')
        inner_g_F_sc, inner_e_F_sc, _, inner_c_F_sc, outer_F_sc = inner_iters(ax_F_sc_det,err_vec_F_sc,errors_gas=errors_gas_F_sc,errors_elec=errors_elec_F_sc,errors_coupling=errors_coupling_F_sc,error_measure='F')
        inner_g_F_sc_Tunknown, inner_e_F_sc_Tunknown, _, inner_c_F_sc_Tunknown, outer_F_sc_Tunknown = inner_iters(ax_F_sc_det,err_vec_F_sc_Tunknown,errors_gas=errors_gas_F_sc_Tunknown,errors_elec=errors_elec_F_sc_Tunknown,errors_coupling=errors_coupling_F_sc_Tunknown,error_measure='F')

    fig = plt.figure('Convergence_plot_eh_F_scaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_F_sc = fig.gca()
    plot_convergence(ax_F_sc,err_vec_sc,max_data_points,'Integrated system')
    plot_convergence(ax_F_sc,err_vec_F_sc,max_data_points,'DD')
    plot_convergence(ax_F_sc,err_vec_F_sc_Tunknown,max_data_points,'DD, Toc1 unknown')
    plot_convergence(ax_F_sc,err_vec_sc_Tunknown,max_data_points,'Integrated system, Toc unknown')
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F||_2$')

    print('\nCreating convergence order plot electricity-heat, scaled')
    fig = plt.figure('Convergence_order_eh_dE_scaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_ord_dE_sc = fig.gca()
    plot_convergence_order(ax_ord_dE_sc,err_vec_sc,max_data_points,'Integrated system')
    plot_convergence_order(ax_ord_dE_sc,err_vec_dE_sc,max_data_points,'DD')
    plot_convergence_order(ax_ord_dE_sc,err_vec_dE_sc_Tunknown,max_data_points,'DD, Toc1 unknown')
    if len(err_vec_sc_Tunknown) > 1:
        plot_convergence_order(ax_ord_dE_sc,err_vec_sc_Tunknown,max_data_points,'Integrated system, Toc1 unknown')
    plt.xlabel(r'$||D_F F(x^k)||_2 / ||D_F F(x^0)||_2$')
    plt.ylabel(r'$||D_F F(x^{k+1})||_2 / ||D_F F(x^0)||_2$')
    x_min_ord_dE_sc,x_max_ord_dE_sc = ax_ord_dE_sc.get_xlim()
    x_slope_dE_sc = np.linspace(x_min_ord_dE_sc,x_max_ord_dE_sc)
    y_slope2_dE_sc = x_slope_dE_sc**2
    ax_ord_dE_sc.loglog(x_slope_dE_sc,x_slope_dE_sc,linestyle='--',color='k',label='slope 1')
    ax_ord_dE_sc.loglog(x_slope_dE_sc,y_slope2_dE_sc,linestyle='-.',color='k',label='slope 2')
    ax_ord_dE_sc.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_ord_dE_sc.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_ord_dE_sc.legend()

    fig = plt.figure('Convergence_order_eh_F_scaled_tol{}_subtol{}'.format(int(abs(np.log10(tol))),int(abs(np.log10(tol_sub)))))
    ax_ord_F_sc = fig.gca()
    plot_convergence_order(ax_ord_F_sc,err_vec_sc,max_data_points,'Integrated system')
    plot_convergence_order(ax_ord_F_sc,err_vec_F_sc,max_data_points,'DD')
    plot_convergence_order(ax_ord_F_sc,err_vec_F_sc_Tunknown,max_data_points,'DD, Toc1 unknown')
    if len(err_vec_sc_Tunknown) > 1:
        plot_convergence_order(ax_ord_F_sc,err_vec_sc_Tunknown,max_data_points,'Integrated system, Toc1 unknown')
    plt.xlabel(r'$||D_F F(x^k)||_2 / ||D_F F(x^0)||_2$')
    plt.ylabel(r'$||D_F F(x^{k+1})||_2 / ||D_F F(x^0)||_2$')
    x_min_ord_F_sc,x_max_ord_F_sc = ax_ord_F_sc.get_xlim()
    x_slope_F_sc = np.linspace(x_min_ord_F_sc,x_max_ord_F_sc)
    y_slope2_F_sc = x_slope_F_sc**2
    ax_ord_F_sc.loglog(x_slope_F_sc,x_slope_F_sc,linestyle='--',color='k',label='slope 1')
    ax_ord_F_sc.loglog(x_slope_F_sc,y_slope2_F_sc,linestyle='-.',color='k',label='slope 2')
    ax_ord_F_sc.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_ord_F_sc.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_ord_F_sc.legend()

    if plot_details:
        max_iters_used_det = np.max([iters_unsc, iters_dE_unsc, iters_F_unsc, iters_sc, iters_dE_sc, iters_F_sc, iters_sc_Tunknown, iters_dE_sc_Tunknown, iters_F_sc_Tunknown, outer_dE_unsc[-1],outer_F_unsc[-1],outer_dE_sc[-1],outer_F_sc[-1],outer_dE_sc_Tunknown[-1],outer_F_sc_Tunknown[-1]])
        layout_convergence(ax_dE_unsc_det,tol,tol_sub,max_iters_used_det)
        layout_convergence(ax_F_unsc_det,tol,tol_sub,max_iters_used_det)
        layout_convergence(ax_dE_sc_det,tol,tol_sub,max_iters_used_det)
        layout_convergence(ax_F_sc_det,tol,tol_sub,max_iters_used_det)
    max_iters_used = np.max([iters_unsc, iters_dE_unsc, iters_F_unsc, iters_sc, iters_dE_sc, iters_F_sc, iters_sc_Tunknown, iters_dE_sc_Tunknown, iters_F_sc_Tunknown])
    layout_convergence(ax_dE_unsc,tol,tol_sub,max_iters_used)
    layout_convergence(ax_F_unsc,tol,tol_sub,max_iters_used)
    layout_convergence(ax_dE_sc,tol,tol_sub,max_iters_used)
    layout_convergence(ax_F_sc,tol,tol_sub,max_iters_used)

    if save_fig:
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            path_to_fig = os.path.join(dir_path,'Figures','MES2N')
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def perm_matr(het_net):
    """Determines the permutation matrices P_x and P_F, reordering x and all coupling equations

    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The network for which the permutation matrix needs to be made.

    Returns
    -------
    P_x : scipy sparse matrix
        Permutation matrix for the vector of variables x
    P_F : scipy sparse matrix
        Permutation matrix for the vector of equations F
    """
    # Determine new indices for x
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)
    x_length = len(x_entries)
    Px_row = list(range(x_length)) # New indices
    Px_data = [1]*x_length # Permutation matrix is binary matrix
    xg_new = [ind for ind,el in enumerate(x_entries) if 'Gas' in type(el).__name__]
    xe_new = [ind for ind,el in enumerate(x_entries) if 'Elec' in type(el).__name__]
    xh_new = [ind for ind,el in enumerate(x_entries) if 'Heat' in type(el).__name__]
    Px_col = xg_new + xe_new + xh_new # Original indices
    P_x = sps.csr_matrix((Px_data,(Px_row,Px_col)),shape=(x_length,x_length))

    # Determine new indices for F
    F_entries, Fg_entries, Fe_entries, known_P_nodes, known_Q_nodes, Fh_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_To_halflinks, Fc_entries, F_fc_nodes, F_fc_amount, F_phi_nodes, F_To_nodes = het_net.get_F_entries(formulation=formulation)
    F_length = len(Fg_entries) + len(Fe_entries) + len(Fh_entries) + np.sum(F_fc_amount) + len(F_phi_nodes) + len(F_To_nodes)
    PF_row = list(range(F_length)) # New indices
    PF_data = [1]*F_length # Permutation matrix is binary matrix
    Fg_new = [ind for ind,el in enumerate(F_entries) if el in Fg_entries]
    Fe_new = [ind for ind,el in enumerate(F_entries) if el in Fe_entries]
    Fh_new = [ind for ind,el in enumerate(F_entries) if el in Fh_entries]
    for ind_el,el in enumerate(F_phi_nodes + F_To_nodes):
        Fh_new.append(len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+np.sum(F_fc_amount)+ind_el)
    Fc_new = list()
    for ind_el,el in enumerate(F_fc_nodes):
        ind = ind_el+len(Fg_entries)+len(Fe_entries)+len(Fh_entries)
        if ind_el>0:
            ind += np.sum(F_fc_amount[0:ind_el])-ind_el # -ind_el, because index is already shifted by one with respect to previous element because ind_el has increased 1
        fc_len = F_fc_amount[ind_el]
        for fc_ind in range(fc_len):
            Fc_new.append(ind + fc_ind)
    while len(xg_new) > len(Fg_new):
        if not Fc_new:
            print('No more coupling equations available to move. Unable to make gas part square: |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
            break
        for ind_el,el in enumerate(F_fc_nodes):
            ind = len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+int(np.sum(F_fc_amount[:ind_el]))
            if F_fc_amount[ind_el]>1:
                dfc_dE = el.der_node_law_dE()
                dfc_dq = dfc_dE[:,0]
                for ind_q,der_q in enumerate(dfc_dq): # this coupling function depends on q
                    if der_q:
                        Fg_new.append(ind + ind_q)
                        Fc_new.remove(ind + ind_q)
                        if len(xg_new) <= len(Fg_new):
                            print('Gas part is now square! |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
                            break
            else:
                dfc_dq,dfc_dP,dfc_dphi = el.der_node_law_dE()
                if dfc_dq != None: # this coupling function depends on q
                    Fg_new.append(ind)
                    Fc_new.remove(ind)
                    if len(xg_new) <= len(Fg_new):
                            print('Gas part is now square! |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
                            break
        print('No more coupling equations available to move. Unable to make gas part square: |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
        break
    while len(xe_new) > len(Fe_new):
        if not Fc_new:
            print('No more coupling equations available to move. Unable to make electrical part square: |xe| = {}, |Fe| = {}'.format(len(xe_new),len(Fe_new)))
            break
        for ind_el,el in enumerate(F_fc_nodes):
            ind = len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+int(np.sum(F_fc_amount[:ind_el]))
            if F_fc_amount[ind_el]>1:
                dfc_dE = el.der_node_law_dE()
                dfc_dP = dfc_dE[:,1]
                for ind_P,der_P in enumerate(dfc_dP): # this coupling function depends on P
                    if der_P:
                        if ind+ind_P in Fc_new: # check if not already moved to Fc
                            Fe_new.append(ind + ind_P)
                            Fc_new.remove(ind + ind_P)
            else:
                dfc_dq,dfc_dP,dfc_dphi = el.der_node_law_dE()
                if dfc_dP != None: # this coupling function depends on P
                    if ind in Fc_new: # check if not already moved to Fc
                        Fe_new.append(ind)
                        Fc_new.remove(ind)
        print('No more coupling equations available to move. Unable to make electrical part square: |xe| = {}, |Fe| = {}'.format(len(xe_new),len(Fe_new)))
        break
    while len(xh_new) > len(Fh_new):
        if not Fc_new:
            print('No more coupling equations available to move. Unable to make heat part square: |xh| = {}, |Fh| = {}'.format(len(xh_new),len(Fh_new)))
            break
        for ind_el,el in enumerate(F_fc_nodes):
            ind = len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+int(np.sum(F_fc_amount[:ind_el]))
            if F_fc_amount[ind_el]>1:
                dfc_dE = el.der_node_law_dE()
                dfc_dphi = dfc_dE[:,2]
                for ind_phi,der_phi in enumerate(dfc_dphi): # this coupling function depends on phi
                    if der_phi:
                        if ind+ind_phi in Fc_new: # check if not already moved to Fc
                            Fh_new.append(ind + ind_phi)
                            Fc_new.remove(ind + ind_phi)
            else:
                dfc_dq,dfc_dP,dfc_dphi = el.der_node_law_dE()
                if dfc_dphi != None: # this coupling function depends on phi
                    if ind in Fc_new: # check if not already moved to Fc
                        Fh_new.append(ind)
                        Fc_new.remove(ind)
        print('No more coupling equations available to move. Unable to make heat part square: |xh| = {}, |Fh| = {}'.format(len(xh_new),len(Fh_new)))
        break
    PF_col = Fg_new + Fe_new + Fh_new + Fc_new
    P_F = sps.csr_matrix((PF_data,(PF_row,PF_col)),shape=(F_length,F_length))
    return P_x, P_F

def jacobians_ge():
    """Plot Jacobian matrices, and eigenvalue spectra, for different indices / ordering"""
    # create network
    het_net, gas_net, elec_net = create_mes_ge_network()

    # initialize network
    x0 = initialize_mes_ge_network(het_net)

    # create system of equations
    from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
    scale_var = 'matrix'

    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsys_unscaled = NonLinearSystemHeterogeneous(het_net,formulation=formulation)
    F0 = nlsys.F(x0)
    J0 = nlsys.J(x0)

    # unscaled system
    fig_J = nlsys_unscaled.spy_plot_J(x0,title='Jacobian spy plot')
    ax_J = plt.gca()
    fig_J_map = nlsys_unscaled.imshow_J(x0,title='Jacobian')

    # spectrum of Jacobian
    fig_spec = nlsys_unscaled.spectrum_J(x0,title='Spectrum')

    # scaled
    D_x = nlsys.Dx()
    D_x_inv = sps.diags(1/D_x.data[0])
    D_F = nlsys.DF()
    J_scaled = D_F.dot(J0.dot(D_x_inv))
    # spy plot of scaled J over original
    nlsys.spy_plot_J(x0,ax=ax_J,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5)

    # colormap of Jacobian
    fig_J_scaled_map = nlsys.imshow_J(x0,title='Scaled Jacobian')

    # spectrum of (scaled) Jacobian
    fig_spectra = nlsys.spectrum_J(x0,title='Scaled spectra',color='tab:blue')
    ax_spectra = fig_spectra.gca()

    # reordering (based on which parts or non-square)
    P_x, P_F = perm_matr(het_net)
    fig_J_re = nlsys.spy_plot_J(x0,title='Reordered Jacobian',P_F=P_F,P_x=P_x)

    J_re = P_F.dot(J_scaled).dot(P_x.transpose())
    print('\nFor scaled Jacobians:')
    print('det(J) = {}'.format(np.linalg.det(J_scaled.todense())))
    print('det(J), reordering x = {}'.format(np.linalg.det(J_re.todense())))
    print('cond(J) = {}'.format(np.linalg.cond(J_scaled.todense())))
    print('cond(J), reordering = {}'.format(np.linalg.cond(J_re.todense())))

    # spectra of scaled systems in one plot
    nlsys.spectrum_J(x0,ax=ax_spectra,P_F=P_F,P_x=P_x,color='tab:red')
    from matplotlib.lines import Line2D
    ax_spectra.legend(handles=[Line2D([0], [0], color='tab:blue', marker='.',label='original'),
                       Line2D([0], [0], color='tab:red', marker='.',label='reordered')])

if __name__== '__main__':
    #run_gas_load_flow()
    #run_electrical_load_flow()
    #run_heat_load_flow()
    #run_heat_load_flow(c_hl=True,dummy_links=True)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "No single-carrier subnetworks found",UserWarning)
        # compare_DD_ge(plot_details=False,max_data_points=100,save_fig=False,dir_path=dir_path)
        compare_DD_eh(max_iter=850,plot_details=False,max_data_points=100,save_fig=False,dir_path=dir_path)

    #with warnings.catch_warnings():
        #warnings.filterwarnings("ignore", "Only a",UserWarning)
        #jacobians_ge()

    plt.show()
