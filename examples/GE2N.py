"""
MES consisting of 2 nodes per carrier (i.e. only one link in every single-carrier network), with multiple couplings.
"""
import warnings
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.carrier import Gas
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.load_flow.system_of_equations import NonLinearSystemGas, NonLinearSystemElectrical, NonLinearSystemHeterogeneous
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
colors_solver = {'mes':'tab:blue','dd':'tab:orange','fp':'tab:purple','mes reod 2':'tab:green','fp NR':'tab:red','fp dec':'tab:pink','fp reod':'tab:brown','fp reod dec':'tab:olive','fp NR reod':'tab:cyan','fp reod 2':'darkgreen','fp reod 2 dec':'tab:grey','fp NR reod 2':'navy'}
# markers_solver = {'mes':'.','dd_dE':'x','dd_F':'*','dd':'s','mes reod':'d','dd fp dy':'o','dd fp F':'+','dd fp NR':'v','dd fp NR FD':'^'}
markers_solver = {'mes':'.','dE':'x','F':'*','an':'o','FD':'s'}

labels_energy = [r'$q_{0^g0^c}$',r'$q_{1^g1^c}$',r'$P_{0^c0^e}$',r'$Q_{0^c0^e}$',r'$P_{1^c1^e}$',r'$Q_{1^c1^e}$']

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

# physical parameters of the coupling unit
eta_GG0 = .6
eta_GG1 = .7
GHV = 60134305#[J/kg]
unit_type_GG = 'ge_gas_fired_gen'
unit_params_GG0={'eta':eta_GG0,'GHV':GHV}
unit_params_GG1={'eta':eta_GG1,'GHV':GHV}
# coordinates
x0c = 1
y0c = 1
x1c = 1
y1c = 0

# solution
V0_sol = 0.93108103*Vbase #[V]
Pc0_sol = 1*Sbase #[W]
Qc0_sol = .5*Sbase #[W]
Pc1_sol = 3.51427623481958*MW #[W]
Qc1_sol = 2.142762348195836*MW #[W]
qc0_sol = -Pc0_sol/(GHV*eta_GG0) #[kg/s]
qc1_sol = -Pc1_sol/(GHV*eta_GG1) #[kg/s]
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
q0_source = -(q1_load - qc0_sol - qc1_sol) #[kg/s]

# initial conditions
q_ic = 2*q1_load
pg_ic = .99*pg0
qc_ic = 1.5*q1_load
Pc_ic = 1.5*MW
Qc_ic = 2*MW

def create_gas_network(scale_var=None,scale_var_params=None):
    """Create a gas network consisting of one source connected to one sink"""
    if scale_var == 'per_unit':
        p0 = pg0/scale_var_params.get('pbase')
        q1_c = -qc1_sol/scale_var_params.get('qbase')
        q1 = q1_load/scale_var_params.get('qbase')
    else:
        p0 = pg0
        q1_c = -qc1_sol
        q1 = q1_load
    g0 = GasNode('gn0',node_type=0,x=x0g,y=y0g,p=p0) # reference node
    g1 = GasNode('gn1',node_type=1,x=x1g,y=y1g,q=q1_c) # load node
    GasHalfLink('gn1_load',g1,q1,bc_type=1) # gas sink, q known

    gl0 = GasLink('gl0',g0,g1,link_type = gas_link_type,link_params = gas_link_params,link_eq_form=link_eq)
    gas_net = GasNetwork('2 nodes')
    gas_net.add_link(gl0)
    return gas_net

def set_xg_LF_init(gas_net,formulation='full',scale_var=None,scale_var_params=None):
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
    if scale_var == 'per_unit':
        q_init /= scale_var_params.get('qbase')
        p_init /= scale_var_params.get('pbase')
    if formulation == 'full':
        x_init = np.concatenate((q_init,p_init))
    elif formulation == 'nodal':
        x_init = p_init
    else:
        raise ValueError("Enter valid formulation for formulation. Either 'full' or 'nodal'.")
    gas_net.initialize()
    gas_net.update(x_init,formulation=formulation)
    x0 = gas_net.set_x_init(formulation=formulation)
    return x0

def run_LF_gas(max_iter=10,tol=1e-6,formulation='full',scale_var=None,scale_var_params=None):
    """Steady-state load flow analysis of gas network. The default values are used for initialization.
    """
    # create network
    gas_net = create_gas_network(scale_var=scale_var,scale_var_params=scale_var_params)
    # initialize
    x0 = set_xg_LF_init(gas_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # solve network
    print('\nRunning load flow for single-carrier gas network')
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_sol/mbar))
    print('q = {} kg/s'.format(q_sol))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
    print('q0 source = {} kg/s'.format(gas_net.nodes[0].half_links[0].q + qc0_sol)) #negative
    return gas_net,x_sol,iters,err_vec

def update_bc_gas(gas_net,q1_c,scale_var=None,scale_var_params=None):
    """Update the BCs of the single-carrier gas network, based on the interface conditions"""
    if scale_var == 'per_unit':
        q1_c = q1_c*scale_var_params.get('qbase')
    gas_net.nodes[1].half_links[0].q = q1_c
    return gas_net

def get_bc_gas(gas_net,scale_var=None,scale_var_params=None):
    """Get the BCs of the single-carrier gas network, which is updated based on the interface conditions"""
    q1_c = gas_net.nodes[1].half_links[0].get_q(scale_var=scale_var,scale_var_params=scale_var_params)
    return q1_c

def G_gas(gas_net,scale_var=None,scale_var_params=None,q0=q0_source):
    """The additional equations needed for full LF in the single-carrier gas network"""
    if scale_var == 'per_unit':
        q0 = q0*scale_var_params.get('qbase')
    q0_c = -gas_net.links[0].get_q(scale_var=scale_var,scale_var_params=scale_var_params) - q0
    return q0_c

def h_gas(gas_net,vg,bg,max_iter_lf=10,tol_lf=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,q0=q0_source,solve_LF=True):
    """The LF solve used in DD"""
    if solve_LF:
        q1_c = bg[0]
        gas_net = update_bc_gas(gas_net,q1_c,scale_var=scale_var,scale_var_params=scale_var_params)
        with HiddenPrints():
            xg,itersg,err_vecg,_,_,q_inj = gas_net.solve_network(tol_lf,max_iter_lf,solver='NR',formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    else:
        nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
        xg = gas_net.set_x_init(formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
        Fg = nlsysg.DF().dot(nlsysg.F(xg))
        err_vecg = np.array([np.linalg.norm(Fg)])
    q0_c = G_gas(gas_net,scale_var=scale_var,scale_var_params=scale_var_params,q0=q0)
    return gas_net, np.array([q0_c]), err_vecg

def hIF_gas(bg,vc,scale_var=None,scale_var_params=None):
    """The IFCs to determine the BCs used in gas"""
    # qc1 = -vc[0]
    # q1_c = ic_gas(qc1)
    q1_c = np.array([1, 0]).dot(vc)
    return np.array([q1_c])

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
        xe = elec_net.set_x_init(formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
        Fe = nlsyse.DF().dot(nlsyse.F(xe))
        err_vece = np.array([np.linalg.norm(Fe)])
    P1_c,Q0_c,Q1_c = G_elec(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    return elec_net, np.array([P1_c,Q0_c,Q1_c]), err_vece

def hIF_elec(be,vc,scale_var=None,scale_var_params=None):
    """The IFCs to determine the BCs used in electrical SC network"""
    P0_c = np.array([0, -1]).dot(vc)
    return np.array([P0_c])

def create_mes_network():
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
    GasHalfLink('gn0_hl',gn0,q0_source,bc_type=1) # add halflink with known gas input to node
    gn1.remove_half_link(gn1.half_links[0]) #remove the half link representing the coupling

    en0 = elec_net.nodes[0]
    en1 = elec_net.nodes[1]
    en0.node_type = 6 # PQV
    en0.V = V0_sol
    en0.remove_half_link(en0.half_links[0]) #remove the half link representing the coupling
    en1.node_type = 5 # PQVdelta
    ElectricalHalfLink('en1_hl',en1,P=P1_load,Q=Q1_load,bc_type=1) # was slack, so didn't have a half link.

    # dummy links
    with HiddenPrints():
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

def set_xmes_LF_init(het_net,formulation={'gas':'full','elec':'complex_power'}):
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
    if formulation.get('gas') == 'nodal':
        raise ValueError('LF not implemented for nodal form. in gas network. Initial guess for MES cannot be made.')
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
    if formulation.get('gas') == 'full':
        xg_init = np.array(q_init+pg_init)
    elif formulation.get('gas') == 'nodal':
        xg_init = np.array([pg_init])
    else:
        raise ValueError("Enter valid formulation for formulation. Either 'full' or 'nodal'.")

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

def trans_matr_mes():
    """Permutation and scaling matrices for the integrated system to mimic the DD system."""
    Dx = sps.diags([1,1,1,1,1,1,-1,-1,-1])
    # the column is moved to the row
    xlength = 9
    Px_row = np.array([1,2,3,4,5,6,7,8,9])-1 #-1, since indexing starts at 0....
    Px_col = np.array([1,2,4,3,7,8,9,5,6])-1
    Px_data = np.ones(xlength)
    Px = sps.csr_matrix((Px_data,(Px_row,Px_col)),shape=(xlength,xlength))

    Flength = xlength
    DF = sps.eye(Flength)
    PF_row = np.array([1,2,3,4,5,6,7,8,9])-1
    PF_col = np.array([2,3,1,4,5,6,7,8,9])-1
    PF_data = np.ones(xlength)
    PF = sps.csr_matrix((PF_data,(PF_row,PF_col)),shape=(xlength,xlength))

    F = np.array([1,2,3,4,5,6,7,8,9])
    return Dx, Px, DF, PF

def block_jacobi_matr(J_reod):
    M_reod = J_reod.copy()
    x_len = J_reod.shape[0]
    M_reod[7,2] = 0
    M_reod[8,4] = 0
    M_reod[0,7] = 0
    M_reod[3,8] = 0
    N_reod = M_reod - J_reod
    return M_reod, N_reod

def jacobians(het_net, gas_net, elec_net,x,formulation={'gas':'full','elec':'complex_power','het':None},scale_var=None,scale_var_params=None):
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
    nlsys_unscaled.spectrum_J(x,ax=ax_spec,P_F=TF,P_x=Tx,color=colors_solver.get('mes reod'))

    # reordered system, scaled
    if scale_var != None:
        # colormap of Jacobian
        fig_J_re_scaled_map = nlsys.imshow_J(x,P_F=TF,P_x=Tx,title='Reordered, scaled, Jacobian',overlay=False)
        ax_J_re_scaled_map = fig_J_re_scaled_map.gca()
        # spectrum of (scaled) Jacobian
        nlsys.spectrum_J(x,ax=ax_spectra,P_F=TF,P_x=Tx,color=colors_solver.get('mes reod'))

    # iteration matrix block-jacobi method
    Tx_inv = sps.diags(1/Dx.data[0]).dot(Px.transpose())
    J_reod = TF.dot(nlsys_unscaled.J(x).dot(Tx_inv))
    M_reod, N_reod = block_jacobi_matr(J_reod)
    fig_M = plt.figure('Spy plot M Block-Jacobi')
    ax_M = fig_M.gca()
    ax_M.spy(M_reod,markerfacecolor='tab:blue',markeredgecolor='tab:blue',marker='s',alpha=1)
    fig_N = plt.figure('Spy plot N Block-Jacobi')
    ax_N = fig_N.gca()
    ax_N.spy(N_reod,markerfacecolor='tab:blue',markeredgecolor='tab:blue',marker='s',alpha=1)
    G = sps.linalg.spsolve(M_reod,N_reod)
    eigs_G,_ = sps.linalg.eigs(G,k=G.shape[0]-2)
    for eig in eigs_G:
        ax_spec.plot(eig.real,eig.imag,'.',color=colors_solver.get('block jacobi'))
    radius = np.max(np.abs(eigs_G))
    ax_spec.add_artist(plt.Circle((0, 0), radius, color=colors_solver.get('block jacobi'), fill=False))
    ax_spec.legend(handles=[Line2D([0],[0],color=colors_solver.get('mes'),marker='.',label='original'),Line2D([0], [0], color=colors_solver.get('mes reod'), marker='.',label='reordered'),Line2D([0], [0], color=colors_solver.get('block jacobi'), marker='.',label='G')])

    if scale_var != None:
        Tx_inv_scaled = sps.diags(1/nlsys.Dx().data[0]).dot(sps.diags(1/Dx.data[0]).dot(Px.transpose()))
        J_reod_scaled = TF.dot(nlsys.DF()).dot(nlsys.J(x).dot(Tx_inv_scaled))
        M_reod_scaled, N_reod_scaled = block_jacobi_matr(J_reod_scaled)
        ax_M.spy(M_reod_scaled,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5)
        ax_N.spy(N_reod_scaled,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5)
        G_scaled = sps.linalg.spsolve(M_reod_scaled,N_reod_scaled)
        eigs_G_scaled,_ = sps.linalg.eigs(G_scaled,k=G_scaled.shape[0]-2)
        for eig in eigs_G_scaled:
            ax_spectra.plot(eig.real,eig.imag,'.',color=colors_solver.get('block jacobi'))
        radius = np.max(np.abs(eigs_G_scaled))
        ax_spectra.add_artist(plt.Circle((0, 0), radius, color=colors_solver.get('block jacobi'), fill=False))
        ax_spectra.legend(handles=[Line2D([0],[0],color=colors_solver.get('mes'),marker='.',label='original'),Line2D([0], [0], color=colors_solver.get('mes reod'), marker='.',label='reordered'),Line2D([0], [0], color=colors_solver.get('block jacobi'), marker='.',label='G')])
    # plot overlay / lines in reoreded Jacobians
    xg_len = 3
    xe_len = 4
    xc_len = 2
    x_len = xg_len+xe_len+xc_len
    if scale_var !=None:
        axes = [ax_J_re,ax_M,ax_N,ax_J_re_scaled_map]
    else:
        axes = [ax_J_re,ax_M,ax_N]
    for ax in axes:
        ax.plot((-0.5,x_len-0.5),(xg_len-0.5,xg_len-0.5),'k-')
        ax.plot((-0.5,x_len-0.5),(xg_len+xe_len-0.5,xg_len+xe_len-0.5),'k-')
        ax.plot((xg_len-0.5,xg_len-0.5),(0-0.5,x_len-0.5),'k-')
        ax.plot((xg_len+xe_len-0.5,xg_len+xe_len-0.5),(0-0.5,x_len-0.5),'k-')


def run_LF_mes(max_iter=10,tol=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,plot_top=False,plot_jac=False,plot_sol=False,reorder=False):
    """Steady-state load flow analysis of combined gas and electrical network, without scaling. The default values are used for initialization.
    """
    if scale_var == 'per_unit':
        raise ValueError('LF mes not implemented for p.u. scaling')
    # create network
    het_net, gas_net, elec_net = create_mes_network()

    # initialize network
    x0 = set_xmes_LF_init(het_net,formulation=formulation)
    if plot_jac:
        if reorder:
            jacobians(het_net, gas_net, elec_net,x0,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
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

def block_jacobi(max_iter=10,max_iter_bj=100,tol=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None):
    """Use block jacobi to solve the LF problem"""
    raise NotImplementedError('This is not (yet) correctly implemented!')
    if scale_var == 'per_unit':
        raise ValueError('LF mes not implemented for p.u. scaling')
    # create network
    het_net, gas_net, elec_net = create_mes_network()

    # initialize network
    x0 = set_xmes_LF_init(het_net,formulation=formulation)

    Dx, Px, DF, PF = trans_matr_mes()
    Tx = Px.dot(Dx)
    TF = PF.dot(DF)

    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    T_F,T_x,T_x_inv,T_F_len,T_x_len = nlsys.scal_perm_matr(D_F=nlsys.DF(),D_x=nlsys.Dx(),P_F=TF,P_x=Tx)
    x_new = x0
    F_new = nlsys.F(x_new)
    J_new = nlsys.J(x_new)
    if T_F_len and T_x_len: # scale x, F and J
        x_new = T_x.dot(x_new)
        F_new = T_F.dot(F_new)
        J_new = T_F.dot(J_new.dot(T_x_inv))
    M_new, N_new = block_jacobi_matr(J_new)
    iter_nr = 0
    err_vec = []
    # check if initial guess happens to be correct
    if T_F_len and (not T_x_len): # Scale stopping criterion only when only T_F is specified. If T_x is specified, F_new is already scaled.
        error = np.linalg.norm(T_F.dot(F_new))
    else:
        error = np.linalg.norm(F_new)
    err_vec.append(error)
    while error > tol and iter_nr < max_iter:
        x_old = x_new
        F_old = F_new
        J_old = J_new
        M_old, N_old = block_jacobi_matr(J_old)
        # The actual algorithm part
        res = np.linalg.norm(F_old - J_old.dot(x_old))
        while res > tol and iter_nr < max_iter_bj:
            x_old = x_new
            F_old = F_new
            J_old = J_new
            M_old, N_old = block_jacobi_matr(J_old)
            x_new = sps.linalg.spsolve(M_old,N_old.dot(x_old)+F_old)
            res = np.linalg.norm(F_old - J_old.dot(x_old))
            if T_F_len and T_x_len:
                # F and J need unscaled x
                F_new = nlsys.F(T_x_inv.dot(x_new))
                J_new = nlsys.J(T_x_inv.dot(x_new))
                # scale F and J
                F_new = T_F.dot(F_new)
                J_new = T_F.dot(J_new.dot(T_x_inv))
            else:
                F_new = nlsys.F(x_new)
                J_new = nlsys.J(x_new)
        if T_F_len and (not T_x_len): # Scale stopping criterion only when only T_F is specified. If T_x is specified, F_new is already scaled.
            error = np.linalg.norm(T_F.dot(F_new))
        else:
            error = np.linalg.norm(F_new)
        iter_nr += 1
        err_vec.append(error)
    if T_F_len and T_x_len: # scale x back
        x_new = T_x_inv.dot(x_new)

    p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.update_full(x_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    return het_net, gas_net, elec_net, x_new, iter_nr, err_vec

def create_coupling_network(scale_var=None,scale_var_params=None):
    """Create a coupling network consisting of two nodes, for a gas-electricity coupling"""
    if scale_var == 'per_unit':
        qc0 = qc0_sol/scale_var_params.get('qbase')
        Pc0 = Pc_ic/scale_var_params.get('Sbase')
        Qc0 = Qc0_sol/scale_var_params.get('Sbase')
        qc1 = -qc_ic/scale_var_params.get('qbase')
        Pc1 = Pc1_sol/scale_var_params.get('Sbase')
        Qc1 = Qc1_sol/scale_var_params.get('Sbase')
    else:
        qc0 = qc0_sol
        Pc0 = Pc_ic
        Qc0 = Qc0_sol
        qc1 = -qc_ic
        Pc1 = Pc1_sol
        Qc1 = Qc1_sol
    c0 = HeterogeneousNode('cn0',node_type=0,x=x0c,y=y0c,unit_type=unit_type_GG,unit_params=unit_params_GG0)
    hlg = GasHalfLink('cn0_hlg',c0,qc0,bc_type=1) # gas flows into coupling node, q known
    hle = ElectricalHalfLink('cn0_hle',c0,P=Pc0,Q=Qc0,bc_type=3) # P is unknown and Q is known
    c1 = HeterogeneousNode('cn1',node_type=0,x=x1c,y=y1c,unit_type=unit_type_GG,unit_params=unit_params_GG1)
    hlg = GasHalfLink('cn1_hlg',c1,qc1,bc_type=0) # gas flows into coupling node, q is unknown
    hle = ElectricalHalfLink('cn1_hle',c1,P=Pc1,Q=Qc1,bc_type=1) # P and Q are known
    coupling_net = HeterogeneousNetwork('coupling network')
    coupling_net.add_node(c0)
    coupling_net.add_node(c1)
    return coupling_net

def set_xc_LF_init(coupling_net,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None):
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
    if scale_var == 'per_unit':
        x_init /= np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase')])
    coupling_net.initialize()
    coupling_net.update(x_init,formulation=formulation)
    x0 = coupling_net.set_x_init(formulation=formulation)
    return x0

def run_LF_coupling(max_iter=10,tol=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None):
    """Steady-state load flow analysis for the coupling network for the gas-electricity"""
    # create network
    coupling_net = create_coupling_network(scale_var=scale_var,scale_var_params=scale_var_params)

    # initialize network
    x0 = set_xc_LF_init(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # solve network
    print('\nRunning load flow for gas-electricity network, only coupling')
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = coupling_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('qc = {} kg/s'.format([-hl.get_q() for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,GasHalfLink)]))
    print('Pc = {} MW'.format([hl.get_P()/MW for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,ElectricalHalfLink)]))
    print('Qc = {} MW'.format([hl.get_Q()/MW for node in coupling_net.get_nodes() for hl in node.get_half_links() if isinstance(hl,ElectricalHalfLink)]))
    return coupling_net, x_sol, iters, err_vec

def update_bc_coupling(coupling_net,qc0,Pc1,Qc0,Qc1,scale_var=None,scale_var_params=None):
    """Update the BCs of the coupling network, based on the interface conditions"""
    if scale_var == 'per_unit':
        qc0 = qc0*scale_var_params.get('qbase')
        Pc1 = Pc1*scale_var_params.get('Sbase')
        Qc0 = Qc0*scale_var_params.get('Sbase')
        Qc1 = Qc1*scale_var_params.get('Sbase')
    coupling_net.nodes[0].half_links[0].q = qc0
    coupling_net.nodes[0].half_links[1].Q = Qc0
    coupling_net.nodes[1].half_links[1].P = Pc1
    coupling_net.nodes[1].half_links[1].Q = Qc1
    return coupling_net

def get_bc_coupling(coupling_net,scale_var=None,scale_var_params=None):
    """Get the BCs of the coupling network, which is updated based on the interface conditions"""
    qc0 = coupling_net.nodes[0].half_links[0].get_q(scale_var=scale_var,scale_var_params=scale_var_params)
    Qc0 = coupling_net.nodes[0].half_links[1].get_Q(scale_var=scale_var,scale_var_params=scale_var_params)
    Pc1 = coupling_net.nodes[1].half_links[1].get_P(scale_var=scale_var,scale_var_params=scale_var_params)
    Qc1 = coupling_net.nodes[1].half_links[1].get_Q(scale_var=scale_var,scale_var_params=scale_var_params)
    return qc0, Pc1, Qc0, Qc1

def h_coupling(coupling_net,vc,bc,max_iter_lf=10,tol_lf=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,solve_LF=True):
    """The LF solve used in DD"""
    if solve_LF:
        qc0,Pc1,Qc0,Qc1 = bc
        coupling_net = update_bc_coupling(coupling_net,qc0,Pc1,Qc0,Qc1,scale_var=scale_var,scale_var_params=scale_var_params)
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

def hIF_coupling(bc,vg,ve,scale_var=None,scale_var_params=None):
    """The IFCs to determine the BCs used in electrical SC network"""
    qc0,Pc1,Qc0,Qc1 = -np.eye(len(vg)+len(ve)).dot(np.concatenate((vg,ve)))
    return np.array([qc0,Pc1,Qc0,Qc1])

def ic_gas(qc1):
    """Determine the BC for the gas networks, based on the outcomes of the other networks. That is, use the interface conditions."""
    q1_c = -qc1
    return q1_c

def ic_elec(Pc0):
    """Determine the BC for the networks, based on the outcomes of the other networks. That is, use the interface conditions."""
    P0_c = -Pc0
    return P0_c

def ic_coupling(q0_c,P1_c,Q0_c,Q1_c):
    """Determine the BC for the networks, based on the outcomes of the other networks. That is, use the interface conditions."""
    qc0 = -q0_c
    Pc1 = -P1_c
    Qc0 = -Q0_c
    Qc1 = -Q1_c
    return qc0,Pc1,Qc0,Qc1

def error_measure_F(gas_net,elec_net,coupling_net,q1_c,P0_c,qc0,Pc1,Qc0,Qc1,xg,xe,xc,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None):
    """The error measure for DD of a gas-electricity coupling, based on (residual of) the system of equations"""
    # upade all BCs
    gas_net = update_bc_gas(gas_net,q1_c,scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = update_bc_elec(elec_net,P0_c,scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = reset_network_without_update_elec(elec_net)
    coupling_net = update_bc_coupling(coupling_net,qc0,Pc1,Qc0,Qc1,scale_var=scale_var,scale_var_params=scale_var_params)
    # determine the value of (residual of) the system of equations
    nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    Fg = nlsysg.DF().dot(nlsysg.F(xg))
    Fe = nlsyse.DF().dot(nlsyse.F(xe))
    Fc = nlsysc.DF().dot(nlsysc.F(xc))
    error = np.linalg.norm(np.concatenate((Fg,Fe,Fc)))
    return error, Fg, Fe, Fc

def run_LF_mes_DD(max_iter_lf=10,max_iter_dd=200,error_measure='F',tol_lf=1e-6,tol_dd=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None):
    """Steady-state load flow analysis of multi-carrier network, using domain decomposition. The default values are used for initialization.

    Uses the already implemented solvers for the single-carrier network. Between single-carrier solves, the boundary values are given to the other domains. The coupling part is solved separately (since the heterogeneous network solver currently assumes qc, Pc, and Qc live on dummy links, not on half links)

    Parameters
    ----------
    error_measure : str, optional
        Determines which error measure to use for the outer iterations. Options are 'F', for the system of equations, or 'dE' for the difference between coupling energies between consecutive iterations. Default is 'F'. NB: for 'dE', the vector with errors is one longer, since an error for iteration 0 cannot be defined.
    """
    # if scale_var == 'per_unit':
    #     raise NotImplementedError("DD is not implemented for per unit scaling. Use 'None' or 'matrix' instead.")
    # create networks
    gas_net = create_gas_network(scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = create_electrical_network(scale_var=scale_var,scale_var_params=scale_var_params)
    coupling_net = create_coupling_network(scale_var=scale_var,scale_var_params=scale_var_params)

    # initialize networks, and set initial guesses
    xg_new = set_xg_LF_init(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    xe_new = set_xe_LF_init(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    xc_new = set_xc_LF_init(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    # iterate
    print('\nRunning load flow with DD, gas-electricity, error {}'.format(error_measure))
    err_vec = list()
    q1c,Pc0 = xc_new
    qc1 = - q1c
    q1_c = ic_gas(qc1)
    P0_c = ic_elec(Pc0)
    q0_c = G_gas(gas_net,scale_var=scale_var,scale_var_params=scale_var_params)
    P1_c,Q0_c,Q1_c = G_elec(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    qc0,Pc1,Qc0,Qc1 = ic_coupling(q0_c,P1_c,Q0_c,Q1_c)
    xc_full_new = np.array([q0_c,q1c,Pc0,Pc1,Qc0,Qc1])
    if error_measure == 'F':
        error, Fg, Fe, Fc = error_measure_F(gas_net,elec_net,coupling_net,q1_c,P0_c,qc0,Pc1,Qc0,Qc1,xg_new,xe_new,xc_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        err_vec.append(error)
    elif error_measure == 'dE':
        error = 1 # set some error which is bigger than the tolerance
    else:
        raise ValueError('enter valid error measure')
    q1_c_old = get_bc_gas(gas_net,scale_var=scale_var,scale_var_params=scale_var_params)
    P0_c_old = get_bc_elec(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    qc0_old,Pc1_old,Qc0_old,Qc1_old = get_bc_coupling(coupling_net,scale_var=scale_var,scale_var_params=scale_var_params)

    if scale_var == 'matrix':
        xb_qc = scale_var_params.get('qbase')*np.ones(2)
        xb_Sc = scale_var_params.get('Sbase')*np.ones(4)
        Dxc_full = sps.diags(1/np.concatenate((xb_qc,xb_Sc)))
    else:
        Dxc_full = sps.eye(len(xc_full_new))

    iter_nr = 0
    errors_gas = dict()
    err_vecg = np.array([1])
    errors_elec = dict()
    err_vece = np.array([1])
    errors_coupling = dict()
    err_vecc = np.array([1])
    errors_gas_DD = dict()
    errors_elec_DD = dict()
    errors_coupling_DD = dict()
    if error_measure == 'F':
        errors_gas_DD[iter_nr] = np.linalg.norm(Fg)
        errors_elec_DD[iter_nr] = np.linalg.norm(Fe)
        errors_coupling_DD[iter_nr] = np.linalg.norm(Fc)
    while error > tol_dd and iter_nr < max_iter_dd:
        xe_old = xe_new.copy()
        xg_old = xg_new.copy()
        xc_old = xc_new.copy()
        xc_full_old = xc_full_new.copy()
        q0_c_old,q1c_old,Pc0_old,Pc1_old,Qc0_old,Qc1_old = xc_full_old

        # Gas network. Update boundary, then solve, then update the boundary values
        qc1 = -xc_old[0]
        q1_c = ic_gas(qc1)
        if len(errors_gas)==0 or (abs(q1_c-q1_c_old) > tol_dd) or (err_vecg[-1] > tol_lf):
            gas_net = update_bc_gas(gas_net,q1_c,scale_var=scale_var,scale_var_params=scale_var_params)
            with HiddenPrints():
                xg_new,itersg,err_vecg,_,_,q_inj = gas_net.solve_network(tol_lf,max_iter_lf,solver='NR',formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
            errors_gas[iter_nr] = err_vecg
        else:
            xg_new = xg_old.copy()
        q0_c = G_gas(gas_net,scale_var=scale_var,scale_var_params=scale_var_params)
        q1_c_old = get_bc_gas(gas_net,scale_var=scale_var,scale_var_params=scale_var_params)

        # Electricity network. Update boundary, then solve, then update the boundary values
        Pc0 = xc_new[1]
        P0_c = ic_elec(Pc0)
        if len(errors_elec)==0 or (abs(P0_c-P0_c_old) > tol_dd) or (err_vece[-1] > tol_lf):
            elec_net = update_bc_elec(elec_net,P0_c,scale_var=scale_var,scale_var_params=scale_var_params)
            elec_net = reset_network_without_update_elec(elec_net)
            with HiddenPrints():
                xe_new,iterse,err_vece,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol_lf,max_iter_lf,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
            errors_elec[iter_nr] = err_vece
        else:
            xe_new = xe_old.copy()
        P1_c,Q0_c,Q1_c = G_elec(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
        P0_c_old = get_bc_elec(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)

        # Coupling network. Update boundary, then solve, then update the boundary values
        qc0,Pc1,Qc0,Qc1 = ic_coupling(q0_c,P1_c,Q0_c,Q1_c)
        if len(errors_coupling)==0 or (np.linalg.norm(np.array([qc0,Pc1,Qc0,Qc1])-np.array([qc0_old,Pc1_old,Qc0_old,Qc1_old])) > tol_dd) or (err_vecc[-1] > tol_lf):
            coupling_net = update_bc_coupling(coupling_net,qc0,Pc1,Qc0,Qc1,scale_var=scale_var,scale_var_params=scale_var_params)
            with HiddenPrints():
                xc_new,itersc,err_vecc,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = coupling_net.solve_network(tol_lf,max_iter_lf,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            errors_coupling[iter_nr] = err_vecc
            q1c,Pc0 = xc_new
            qc1 = -q1c
            q1_c = ic_gas(qc1)
            P0_c = ic_elec(Pc0)
        else:
            xc_new = xc_old.copy()
        qc0_old,Pc1_old,Qc0_old,Qc1_old = get_bc_coupling(coupling_net,scale_var=scale_var,scale_var_params=scale_var_params)

        # Determine error. Also updates the BCs of all networks
        if error_measure == 'F':
            error, Fg, Fe, Fc = error_measure_F(gas_net,elec_net,coupling_net,q1_c,P0_c,qc0,Pc1,Qc0,Qc1,xg_new,xe_new,xc_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            errors_gas_DD[iter_nr+1] = np.linalg.norm(Fg)
            errors_elec_DD[iter_nr+1] = np.linalg.norm(Fe)
            errors_coupling_DD[iter_nr+1] = np.linalg.norm(Fc)
            xc_full_new = np.array([q0_c,q1c,Pc0,Pc1,Qc0,Qc1])
        elif error_measure == 'dE':
            gas_net = update_bc_gas(gas_net,q1_c,scale_var=scale_var,scale_var_params=scale_var_params)
            elec_net = update_bc_elec(elec_net,P0_c,scale_var=scale_var,scale_var_params=scale_var_params)
            coupling_net = update_bc_coupling(coupling_net,qc0,Pc1,Qc0,Qc1,scale_var=scale_var,scale_var_params=scale_var_params)
            xc_full_new = np.array([q0_c,q1c,Pc0,Pc1,Qc0,Qc1])
            error = np.linalg.norm(Dxc_full.dot(xc_full_new) - Dxc_full.dot(xc_full_old))
        err_vec.append(error)
        iter_nr += 1
    # print results
    with HiddenPrints():
        het_net, gas_net, elec_net = create_mes_network()
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
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('||F mes|| = {:.4e}'.format(np.linalg.norm(nlsys.DF().dot(nlsys.F(x_new)))))
    return het_net, gas_net, elec_net,xg_new, xe_new, xc_new, x_new, iter_nr, err_vec, errors_gas, errors_elec, errors_coupling, errors_gas_DD, errors_elec_DD, errors_coupling_DD

def get_xLF_FP(gas_net,elec_net,coupling_net,vc,bc,formulation={'gas':'full','elec':'complex_power'}):
    """get the LF variables when FP iteration is used"""
    xg = gas_net.set_x_init(formulation=formulation.get('gas'))
    xe = elec_net.set_x_init(formulation=formulation.get('elec'))
    # xc = coupling_net.set_x_init(formulation=formulation)
    # print('xc={}\nvc={}'.format(xc,vc))
    xc = vc.copy()
    q1c,Pc0 = vc
    qc0,Pc1,Qc0,Qc1 = bc
    xc_full = np.array([-qc0,q1c,Pc0,Pc1,Qc0,Qc1])
    x = np.concatenate((xg,xe,xc_full))
    return xg,xe,xc,xc_full,x

def get_suby_FP(y,ordering=1):
    """Split y into the subvectors"""
    len_vg = 1
    len_bg = 1
    len_ve = 3
    len_be = 1
    len_vc = 2
    len_bc = 4
    if ordering == 1:
        vg = y[:len_vg]
        bg = y[len_vg:len_vg+len_bg]
        ve = y[len_vg+len_bg:len_vg+len_bg+len_ve]
        be = y[len_vg+len_bg+len_ve:len_vg+len_bg+len_ve+len_be]
        vc = y[-len_vc-len_bc:-len_bc]
        bc = y[-len_bc:]
    elif ordering == 2:
        vg = y[:len_vg]
        ve = y[len_vg:len_vg+len_ve]
        vc = y[len_vg+len_ve:len_vg+len_ve+len_vc]
        bg = y[len_vg+len_ve+len_vc:len_vg+len_ve+len_vc+len_bg]
        be = y[-len_be-len_bc:-len_bc]
        bc = y[-len_bc:]
    elif ordering == 3:
        vg = y[:len_vg]
        ve = y[len_vg:len_vg+len_ve]
        bc = y[len_vg+len_ve:len_vg+len_ve+len_bc]
        vc = y[len_vg+len_ve+len_bc:len_vg+len_ve+len_bc+len_vc]
        bg = y[-len_bg-len_be:-len_be]
        be = y[-len_be:]
    return vg, bg, ve, be, vc, bc

def get_y_FP(gas_net,elec_net,coupling_net,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,ordering=1):
    """Set y used in FP iteration"""
    q0_c = G_gas(gas_net,scale_var=scale_var,scale_var_params=scale_var_params)
    vg = np.array([q0_c])
    q1_c = get_bc_gas(gas_net,scale_var=scale_var,scale_var_params=scale_var_params)
    bg = np.array([q1_c])
    P1_c,Q0_c,Q1_c = G_elec(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    ve = np.array([P1_c,Q0_c,Q1_c])
    P0_c = get_bc_elec(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    be = np.array([P0_c])
    with HiddenPrints():
        vc = coupling_net.set_x_init(formulation=formulation)
    qc0,Pc1,Qc0,Qc1 = get_bc_coupling(coupling_net,scale_var=scale_var,scale_var_params=scale_var_params)
    bc = np.array([qc0,Pc1,Qc0,Qc1])
    if ordering == 1:
        return np.concatenate((vg,bg,ve,be,vc,bc))
    elif ordering == 2:
        return np.concatenate((vg,ve,vc,bg,be,bc))
    elif ordering == 3:
        return np.concatenate((vg,ve,bc,vc,bg,be))

def g_FP(gas_net,elec_net,coupling_net,y,max_iter_lf=10,tol_lf=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,solve_LFg=True,solve_LFe=True,solve_LFc=True,ordering=1):
    """The system g(y) used in FP iteration based on DD"""
    vg_old, bg_old, ve_old, be_old, vc_old, bc_old = get_suby_FP(y,ordering=ordering)

    gas_net, vg_new, err_vecg = h_gas(gas_net,vg_old,bg_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,q0=q0_source,solve_LF=solve_LFg)
    bg_new = hIF_gas(bg_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)

    elec_net, ve_new, err_vece = h_elec(elec_net,ve_old,be_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=solve_LFe)
    be_new = hIF_elec(be_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)

    coupling_net, vc_new, err_vecc = h_coupling(coupling_net,vc_old,bc_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=solve_LFc)
    bc_new = hIF_coupling(bc_old,vg_old,ve_old,scale_var=scale_var,scale_var_params=scale_var_params)

    if ordering == 1:
        g = np.concatenate((vg_new,bg_new,ve_new,be_new,vc_new,bc_new))
    elif ordering == 2:
        g = np.concatenate((vg_new,ve_new,vc_new,bg_new,be_new,bc_new))
    elif ordering == 3:
        g = np.concatenate((vg_new,ve_new,bc_new,vc_new,bg_new,be_new))
    return gas_net, elec_net, coupling_net, g, err_vecg, err_vece, err_vecc

def jac_FP(gas_net,elec_net,coupling_net,y,formulation={'gas':'full','elec':'complex_power'},Dy_inv=None,ordering=1,scale_var=None,scale_var_params=None):
    """Jacobian of g(y) for the system used in FP iteration based on DD"""
    if formulation.get('gas') == 'nodal':
        raise NotImplementedError('Jacobian for FP not implemented for nodal formulation in gas')
    vg, bg, ve, be, vc, bc = get_suby_FP(y,ordering=ordering)
    xg,xe,xc,xc_full,x = get_xLF_FP(gas_net,elec_net,coupling_net,vc,bc,formulation=formulation)
    q1_c = bg[0]
    P0_c = be[0]
    qc0,Pc1,Qc0,Qc1 = bc
    # upade all BCs
    gas_net = update_bc_gas(gas_net,q1_c,scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = update_bc_elec(elec_net,P0_c,scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = reset_network_without_update_elec(elec_net)
    coupling_net = update_bc_coupling(coupling_net,qc0,Pc1,Qc0,Qc1,scale_var=scale_var,scale_var_params=scale_var_params)
    # determine the value of (residual of) the system of equations
    nlsysg = NonLinearSystemGas(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsyse = NonLinearSystemElectrical(elec_net,formulation=formulation.get('elec'),scale_var=scale_var,scale_var_params=scale_var_params)
    nlsysc = NonLinearSystemHeterogeneous(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    # submatrices in LF gas part
    if scale_var == 'matrix':
        Dxg_inv = sps.diags(1/nlsysg.Dx().data[0])
        Jgg = (nlsysg.DF().dot(nlsysg.J(xg)).dot(Dxg_inv)).todense()
    else:
        Jgg = nlsysg.J_dense(xg)
    dFg_dbg = np.zeros((len(xg),len(bg)))
    dFg_dbg[0,0] = -1# scaling has no influence
    # if scale_var == 'matrix':
    #     dFg_dbg[0,0] *= nlsysg.DF()[0,0]*Dy_inv[len(vg),len(vg)]
    dhg_dvg = -np.eye(len(vg)) # scaling has no influence
    dhg_dxg = np.zeros((len(vg),len(xg)))
    dhg_dxg[0,0] = -1 # scaling has no influence
    dhg_dbg = np.zeros((len(vg),len(bg)))
    dgg_dvg = np.zeros((len(vg),len(vg)))#np.eye(len(vg))
    dxg_dbg = np.linalg.solve(Jgg,-dFg_dbg)
    dvg_dbg = np.linalg.solve(dhg_dvg,-dhg_dbg - dhg_dxg.dot(dxg_dbg))
    dgg_dbg = dvg_dbg
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
    # submatrices in LF coupling part
    with HiddenPrints():
        if scale_var == 'matrix':
            Dxc_inv = sps.diags(1/nlsysc.Dx().data[0])
            Jcc = nlsysc.DF().dot(nlsysc.J(xc)).dot(Dxc_inv).todense()
        else:
            Jcc = nlsysc.J_dense(xc)
    dhc_dbc = np.zeros((len(vc),len(bc)))
    dhc_dbc[0,0] = eta_GG0*GHV #dFc0_dqc0
    if scale_var == 'matrix':
        dhc_dbc[0,0] *= 1/scale_var_params.get('Sbase')*scale_var_params.get('qbase')
    dhc_dbc[1,1] = 1 #dFc1_dPc1
    dgc_dvc = np.zeros((len(vc),len(vc)))#np.eye(len(vc))
    dvc_dbc = np.linalg.solve(Jcc,-dhc_dbc)
    dgc_dbc = dvc_dbc
    # submatrices IFCs
    dgifcg_dvc = np.array([1,0])
    dgifcg_dbg = np.zeros((len(bg),len(bg)))#np.eye(len(bg))
    dgifce_dvc = np.array([0,-1])
    dgifce_dbe = np.zeros((len(be),len(be)))#np.eye(len(be))
    dgifcc_dvg = np.zeros((len(bc),len(vg)))
    dgifcc_dvg[0,0] = -1
    dgifcc_dve = np.zeros((len(bc),len(ve)))
    dgifcc_dve[-len(ve):,:][:,-len(ve):] = -np.eye(len(ve))
    dgifcc_dbc = np.zeros((len(bc),len(bc)))#np.eye(len(bc))
    # collect the submatrices
    dg_dy = np.zeros((len(y),len(y)))
    if ordering == 1:
        vg_ind = 0 # first index of this part of y
        bg_ind = vg_ind + len(vg)
        ve_ind = bg_ind + len(bg)
        be_ind = ve_ind + len(ve)
        vc_ind = be_ind + len(be)
        bc_ind = vc_ind + len(vc)
        # dervivatives of gas part
        dg_dy[:bg_ind,:][:,:bg_ind] = dgg_dvg
        dg_dy[:bg_ind,:][:,bg_ind:ve_ind] = dgg_dbg
        dg_dy[bg_ind:ve_ind,:][:,bg_ind:ve_ind] = dgifcg_dbg
        dg_dy[bg_ind:ve_ind,:][:,vc_ind:bc_ind] = dgifcg_dvc
        # derivatives of electrical part
        dg_dy[ve_ind:be_ind,:][:,ve_ind:be_ind] = dge_dve
        dg_dy[ve_ind:be_ind,:][:,be_ind:vc_ind] = dge_dbe
        dg_dy[be_ind:vc_ind,:][:,be_ind:vc_ind] = dgifce_dbe
        dg_dy[be_ind:vc_ind,:][:,vc_ind:bc_ind] = dgifce_dvc
        # derivatives of coupling part
        dg_dy[vc_ind:bc_ind,:][:,vc_ind:bc_ind] = dgc_dvc
        dg_dy[vc_ind:bc_ind,:][:,bc_ind:] = dgc_dbc
        dg_dy[bc_ind:,:][:,bc_ind:] = dgifcc_dbc
        dg_dy[bc_ind:,:][:,:bg_ind] = dgifcc_dvg
        dg_dy[bc_ind:,:][:,ve_ind:be_ind] = dgifcc_dve
    elif ordering == 2:
        vg_ind = 0 # first index of this part of y
        ve_ind = vg_ind + len(vg)
        vc_ind = ve_ind + len(ve)
        bg_ind = vc_ind + len(vc)
        be_ind = bg_ind + len(bg)
        bc_ind = be_ind + len(be)
        # dervivatives of gas part
        dg_dy[:ve_ind,:][:,:ve_ind] = dgg_dvg
        dg_dy[:ve_ind,:][:,bg_ind:be_ind] = dgg_dbg
        dg_dy[bg_ind:be_ind,:][:,bg_ind:be_ind] = dgifcg_dbg
        dg_dy[bg_ind:be_ind,:][:,vc_ind:bg_ind] = dgifcg_dvc
        # derivatives of electrical part
        dg_dy[ve_ind:vc_ind,:][:,ve_ind:vc_ind] = dge_dve
        dg_dy[ve_ind:vc_ind,:][:,be_ind:bc_ind] = dge_dbe
        dg_dy[be_ind:bc_ind,:][:,be_ind:bc_ind] = dgifce_dbe
        dg_dy[be_ind:bc_ind,:][:,vc_ind:bg_ind] = dgifce_dvc
        # derivatives of coupling part
        dg_dy[vc_ind:bg_ind,:][:,vc_ind:bg_ind] = dgc_dvc
        dg_dy[vc_ind:bg_ind,:][:,bc_ind:] = dgc_dbc
        dg_dy[bc_ind:,:][:,bc_ind:] = dgifcc_dbc
        dg_dy[bc_ind:,:][:,:ve_ind] = dgifcc_dvg
        dg_dy[bc_ind:,:][:,ve_ind:vc_ind] = dgifcc_dve
    elif ordering == 3:
        vg_ind = 0 # first index of this part of y
        ve_ind = vg_ind + len(vg)
        bc_ind = ve_ind + len(ve)
        vc_ind = bc_ind + len(bc)
        bg_ind = vc_ind + len(vc)
        be_ind = bg_ind + len(bg)
        # dervivatives of gas part
        dg_dy[:ve_ind,:][:,:ve_ind] = dgg_dvg
        dg_dy[:ve_ind,:][:,bg_ind:be_ind] = dgg_dbg
        dg_dy[bg_ind:be_ind,:][:,bg_ind:be_ind] = dgifcg_dbg
        dg_dy[bg_ind:be_ind,:][:,vc_ind:bg_ind] = dgifcg_dvc
        # derivatives of electrical part
        dg_dy[ve_ind:bc_ind,:][:,ve_ind:bc_ind] = dge_dve
        dg_dy[ve_ind:bc_ind,:][:,be_ind:] = dge_dbe
        dg_dy[be_ind:,:][:,be_ind:] = dgifce_dbe
        dg_dy[be_ind:,:][:,vc_ind:bg_ind] = dgifce_dvc
        # derivatives of coupling part
        dg_dy[vc_ind:bg_ind,:][:,vc_ind:bg_ind] = dgc_dvc
        dg_dy[vc_ind:bg_ind,:][:,bc_ind:vc_ind] = dgc_dbc
        dg_dy[bc_ind:vc_ind,:][:,bc_ind:vc_ind] = dgifcc_dbc
        dg_dy[bc_ind:vc_ind,:][:,:ve_ind] = dgifcc_dvg
        dg_dy[bc_ind:vc_ind,:][:,ve_ind:bc_ind] = dgifcc_dve
    return gas_net, elec_net, coupling_net, dg_dy

def jac_FP_FD(gas_net,elec_net,coupling_net,y,max_iter_lf=10,tol_lf=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,solve_LFg=True,solve_LFe=True,solve_LFc=True,Dy=np.array([]),Dy_inv=np.array([]),ordering=1):
    """Jacobian of g(y) for the system used in FP iteration based on DD, using FD"""
    J_FD = np.zeros((len(y),len(y)),dtype='float64')
    vg,bg,ve,be,vc,bc = get_suby_FP(y,ordering=ordering)
    xg,xe,xc,xc_full,x = get_xLF_FP(gas_net,elec_net,coupling_net,vc,bc,formulation=formulation)
    gas_net.update(xg,formulation=formulation.get('gas'))
    elec_net = reset_network_without_update_elec(elec_net)
    elec_net.update(xe,formulation=formulation.get('elec'))
    coupling_net.update(xc,formulation=formulation)
    gas_net, elec_net, coupling_net, g, _, _, _ = g_FP(gas_net,elec_net,coupling_net,y,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LFg=solve_LFg,solve_LFe=solve_LFe,solve_LFc=solve_LFc,ordering=ordering)
    for i in range(len(y)):
        e = np.zeros(len(y))
        e[i] = 1.
        if scale_var == 'matrix':
            y_dy = y.copy()
            y_dy[i] = Dy.data[0][i]*(y[i]*Dy_inv.data[0][i] + e[i]*tol_lf*Dy_inv.data[0][i])
        else:
            y_dy = y + e*tol_lf
        gas_net.update(xg,formulation=formulation.get('gas'))
        elec_net = reset_network_without_update_elec(elec_net)
        elec_net.update(xe,formulation=formulation.get('elec'))
        coupling_net.update(xc,formulation=formulation)
        gas_net, elec_net, coupling_net, g_dy, _, _, _ = g_FP(gas_net,elec_net,coupling_net,y_dy,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LFg=solve_LFg,solve_LFe=solve_LFe,solve_LFc=solve_LFc,ordering=ordering)
        J_FD[:,i] = (g_dy-g)/tol_lf
    # solve again, as this also updates BCs etc.
    gas_net.update(xg,formulation=formulation.get('gas'))
    elec_net = reset_network_without_update_elec(elec_net)
    elec_net.update(xe,formulation=formulation.get('elec'))
    coupling_net.update(xc,formulation=formulation)
    gas_net, elec_net, coupling_net, g, err_vecg, err_vece, err_vecc = g_FP(gas_net,elec_net,coupling_net,y,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LFg=solve_LFg,solve_LFe=solve_LFe,solve_LFc=solve_LFc,ordering=ordering)
    return gas_net, elec_net, coupling_net, J_FD, g, err_vecg, err_vece, err_vecc

def run_LF_mes_DD_FP(max_iter_lf=10,max_iter_fp=200,tol_lf=1e-6,tol_fp=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,error_measure='F',decoupled=False,ordering=1):
    """Steady-state load flow analysis of multi-carrier network, using fixed-point iteration based on domain decomposition. The default values are used for initialization.

    Uses the already implemented solvers for the single-carrier network. Between single-carrier solves, the boundary values are given to the other domains. The coupling part is solved separately (since the heterogeneous network solver currently assumes qc, Pc, and Qc live on dummy links, not on half links)

    """
    # if scale_var == 'per_unit':
    #     raise NotImplementedError("DD is not implemented for per unit scaling. Use 'None' or 'matrix' instead.")
    # create networks
    gas_net = create_gas_network(scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = create_electrical_network(scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = reset_network_without_update_elec(elec_net)
    coupling_net = create_coupling_network(scale_var=scale_var,scale_var_params=scale_var_params)

    # initialize networks, and set initial guesses
    xg_new = set_xg_LF_init(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    xe_new = set_xe_LF_init(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    with HiddenPrints():
        xc_new = set_xc_LF_init(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    y_new = get_y_FP(gas_net,elec_net,coupling_net,formulation=formulation,ordering=ordering)
    vg_new, bg_new, ve_new, be_new, vc_new, bc_new = get_suby_FP(y_new,ordering=ordering)

    # match initial guess with the one of original DD
    if decoupled:
        q1c,Pc0 = vc_new
        qc1 = - q1c
        q1_c = ic_gas(qc1)
        P0_c = ic_elec(Pc0)
        q0_c = vg_new[0]
        P1_c,Q0_c,Q1_c = ve_new
        qc0,Pc1,Qc0,Qc1 = ic_coupling(q0_c,P1_c,Q0_c,Q1_c)
        bg_new = np.array([q1_c])
        be_new = np.array([P0_c])
        bc_new = np.array([qc0,Pc1,Qc0,Qc1])
        if ordering == 1:
            y_new = np.concatenate((vg_new,bg_new,ve_new,be_new,vc_new,bc_new))
        elif ordering == 2:
            y_new = np.concatenate((vg_new,ve_new,vc_new,bg_new,be_new,bc_new))
        elif ordering == 3:
            y_new = np.concatenate((vg_new,ve_new,bc_new,vc_new,bg_new,be_new))

    if scale_var == 'matrix':
        vgb = scale_var_params.get('qbase')*np.ones(1)
        bgb = scale_var_params.get('qbase')*np.ones(1)
        veb = scale_var_params.get('Sbase')*np.ones(3)
        beb = scale_var_params.get('Sbase')*np.ones(1)
        vcb = np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase')])
        bcb = np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase')])
        if ordering == 1:
            Dy = sps.diags(1/np.concatenate((vgb,bgb,veb,beb,vcb,bcb)))
        elif ordering == 2:
            Dy = sps.diags(1/np.concatenate((vgb,veb,vcb,bgb,beb,bcb)))
        elif ordering == 3:
            Dy = sps.diags(1/np.concatenate((vgb,veb,bcb,vcb,bgb,beb)))
    else:
        Dy = sps.eye(len(y_new))
    # iterate
    print('\nRunning load flow with FP based on DD, gas-electricity, error {}, ordering {}, decoupled: {}'.format(error_measure,ordering,decoupled))
    err_vec = list()
    if error_measure == 'dy':
        error = 1 # set some initial value for the error
    elif error_measure == 'F':
        xg_new,xe_new,xc_new,xc_full_new,x_new = get_xLF_FP(gas_net,elec_net,coupling_net,vc_new,bc_new,formulation=formulation)
        q1_c = bg_new[0]
        P0_c = be_new[0]
        qc0,Pc1,Qc0,Qc1 = bc_new
        with HiddenPrints():
            error, Fg, Fe, Fc = error_measure_F(gas_net,elec_net,coupling_net,q1_c,P0_c,qc0,Pc1,Qc0,Qc1,xg_new,xe_new,xc_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        err_vec.append(error)
    iter_nr = 0
    errors_gas = dict()
    err_vecg = np.array([1])
    errors_elec = dict()
    err_vece = np.array([1])
    errors_coupling = dict()
    err_vecc = np.array([1])
    errors_gas_DD = dict()
    errors_elec_DD = dict()
    errors_coupling_DD = dict()
    if error_measure == 'F':
        errors_gas_DD[iter_nr] = np.linalg.norm(Fg)
        errors_elec_DD[iter_nr] = np.linalg.norm(Fe)
        errors_coupling_DD[iter_nr] = np.linalg.norm(Fc)
    while error > tol_fp and iter_nr < max_iter_fp:
        y_old = y_new.copy()
        vg_old, bg_old, ve_old, be_old, vc_old, bc_old = get_suby_FP(y_old,ordering=ordering)

        if ordering == 1:
            if len(errors_gas)==0 or (np.linalg.norm(dbg) > tol_lf) or (err_vecg[-1] > tol_lf):
                gas_net, vg_new, err_vecg = h_gas(gas_net,vg_old,bg_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,q0=q0_source,solve_LF=True)
                errors_gas[iter_nr] = err_vecg
            else:
                vg_new = vg_old.copy()
            bg_new = hIF_gas(bg_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbg = bg_new-bg_old
            # if scale_var == 'matrix':
            #     dbg = dbg*bgb # unscaled, as is used in the DD algorithm

            if len(errors_elec)==0 or (np.linalg.norm(dbe) > tol_lf) or (err_vece[-1] > tol_lf):
                elec_net, ve_new, err_vece = h_elec(elec_net,ve_old,be_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_elec[iter_nr] = err_vece
            else:
                ve_new = ve_old.copy()
            be_new = hIF_elec(be_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbe = be_new-be_old
            # if scale_var == 'matrix':
            #     dbe = dbe*beb # unscaled, as is used in the DD algorithm

            if len(errors_coupling)==0 or (np.linalg.norm(dbc) > tol_lf) or (err_vecc[-1] > tol_lf):
                coupling_net, vc_new, err_vecc = h_coupling(coupling_net,vc_old,bc_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_coupling[iter_nr] = err_vecc
            else:
                vc_new = vc_old.copy()
            if decoupled:
                bc_new = hIF_coupling(bc_old,vg_new,ve_new,scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                bc_new = hIF_coupling(bc_old,vg_old,ve_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbc = bc_new-bc_old
            # if scale_var == 'matrix':
            #     dbc = dbc*bcb # unscaled, as is used in the DD algorithm

            y_new = np.concatenate((vg_new,bg_new,ve_new,be_new,vc_new,bc_new)) # always unscaled
        elif ordering == 2:
            if len(errors_gas)==0 or (np.linalg.norm(dbg) > tol_lf) or (err_vecg[-1] > tol_lf):
                gas_net, vg_new, err_vecg = h_gas(gas_net,vg_old,bg_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,q0=q0_source,solve_LF=True)
                errors_gas[iter_nr] = err_vecg
            else:
                vg_new = vg_old.copy()
            if len(errors_elec)==0 or (np.linalg.norm(dbe) > tol_lf) or (err_vece[-1] > tol_lf):
                elec_net, ve_new, err_vece = h_elec(elec_net,ve_old,be_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_elec[iter_nr] = err_vece
            else:
                ve_new = ve_old.copy()
            if len(errors_coupling)==0 or (np.linalg.norm(dbc) > tol_lf) or (err_vecc[-1] > tol_lf):
                coupling_net, vc_new, err_vecc = h_coupling(coupling_net,vc_old,bc_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_coupling[iter_nr] = err_vecc
            else:
                vc_new = vc_old.copy()
            if decoupled:
                bg_new = hIF_gas(bg_old,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
                be_new = hIF_elec(be_old,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
                bc_new = hIF_coupling(bc_old,vg_new,ve_new,scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                bg_new = hIF_gas(bg_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
                be_new = hIF_elec(be_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
                bc_new = hIF_coupling(bc_old,vg_old,ve_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbg = bg_new-bg_old
            dbe = be_new-be_old
            dbc = bc_new-bc_old
            # if scale_var == 'matrix':
            #     dbg = dbg*bgb # unscaled, as is used in the DD algorithm
            #     dbe = dbe*beb # unscaled, as is used in the DD algorithm
            #     dbc = dbc*bcb # unscaled, as is used in the DD algorithm

            y_new = np.concatenate((vg_new,ve_new,vc_new,bg_new,be_new,bc_new)) # always unscaled
        elif ordering == 3:
            # g1
            if len(errors_gas)==0 or (np.linalg.norm(dbg) > tol_lf) or (err_vecg[-1] > tol_lf):
                gas_net, vg_new, err_vecg = h_gas(gas_net,vg_old,bg_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,q0=q0_source,solve_LF=True)
                errors_gas[iter_nr] = err_vecg
            else:
                vg_new = vg_old.copy()
            if len(errors_elec)==0 or (np.linalg.norm(dbe) > tol_lf) or (err_vece[-1] > tol_lf):
                elec_net, ve_new, err_vece = h_elec(elec_net,ve_old,be_old,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LF=True)
                errors_elec[iter_nr] = err_vece
            else:
                ve_new = ve_old.copy()
            # g2
            if decoupled:
                bc_new = hIF_coupling(bc_old,vg_new,ve_new,scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                bc_new = hIF_coupling(bc_old,vg_old,ve_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbc = bc_new-bc_old
            # if scale_var == 'matrix':
            #     dbc = dbc*bcb # unscaled, as is used in the DD algorithm
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
                bg_new = hIF_gas(bg_old,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
                be_new = hIF_elec(be_old,vc_new,scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                bg_new = hIF_gas(bg_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
                be_new = hIF_elec(be_old,vc_old,scale_var=scale_var,scale_var_params=scale_var_params)
            dbg = bg_new-bg_old
            dbe = be_new-be_old
            # if scale_var == 'matrix':
            #     dbg = dbg*bgb # unscaled, as is used in the DD algorithm
            #     dbe = dbe*beb # unscaled, as is used in the DD algorithm
            y_new = np.concatenate((vg_new,ve_new,bc_new,vc_new,bg_new,be_new))

        # Determine error
        if error_measure == 'F': #Also updates the BCs of all networks
            xg_new,xe_new,xc_new,xc_full_new,x_new = get_xLF_FP(gas_net,elec_net,coupling_net,vc_new,bc_new,formulation=formulation)
            q1_c = bg_new[0]
            P0_c = be_new[0]
            qc0,Pc1,Qc0,Qc1 = bc_new
            with HiddenPrints():
                error, Fg, Fe, Fc = error_measure_F(gas_net,elec_net,coupling_net,q1_c,P0_c,qc0,Pc1,Qc0,Qc1,xg_new,xe_new,xc_new,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
            errors_gas_DD[iter_nr+1] = np.linalg.norm(Fg)
            errors_elec_DD[iter_nr+1] = np.linalg.norm(Fe)
            errors_coupling_DD[iter_nr+1] = np.linalg.norm(Fc)
        elif error_measure == 'dy':
            error = np.linalg.norm(Dy.dot(y_new)-Dy.dot(y_old))
        err_vec.append(error)
        iter_nr += 1

    # print results
    xg_new,xe_new,xc_new,xc_full_new,x_new = get_xLF_FP(gas_net,elec_net,coupling_net,vc_new,bc_new,formulation=formulation)
    with HiddenPrints():
        het_net, gas_net, elec_net = create_mes_network()
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
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('||F mes|| = {:.4e}'.format(np.linalg.norm(nlsys.DF().dot(nlsys.F(x_new)))))
    return het_net, gas_net, elec_net, xg_new, xe_new, xc_new, x_new, iter_nr, err_vec, errors_gas, errors_elec, errors_coupling, errors_gas_DD, errors_elec_DD, errors_coupling_DD

def run_LF_mes_DD_FP_NR(max_iter_lf=10,max_iter_fp=200,tol_lf=1e-6,tol_fp=1e-6,formulation={'gas':'full','elec':'complex_power'},scale_var=None,scale_var_params=None,plot_jac=False,der=True,ordering=1):
    """Steady-state load flow analysis of multi-carrier network, using NR iteration based on the fixed-point system, which is based on domain decomposition. The default values are used for initialization.

    Uses the already implemented solvers for the single-carrier network. Between single-carrier solves, the boundary values are given to the other domains. The coupling part is solved separately (since the heterogeneous network solver currently assumes qc, Pc, and Qc live on dummy links, not on half links)

    """
    # if scale_var == 'per_unit':
    #     raise NotImplementedError("DD is not implemented for per unit scaling. Use 'None' or 'matrix' instead.")
    print('\nRunning load flow with NR on FP based on DD, gas-electricity, an. der: {}'.format(der))
    # create networks
    gas_net = create_gas_network(scale_var=scale_var,scale_var_params=scale_var_params)
    elec_net = create_electrical_network(scale_var=scale_var,scale_var_params=scale_var_params)
    coupling_net = create_coupling_network(scale_var=scale_var,scale_var_params=scale_var_params)

    # initialize networks, and set initial guesses
    xg_new = set_xg_LF_init(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
    xe_new = set_xe_LF_init(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
    with HiddenPrints():
        xc_new = set_xc_LF_init(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    y_new = get_y_FP(gas_net,elec_net,coupling_net,formulation=formulation,ordering=ordering)
    vg_new, bg_new, ve_new, be_new, vc_new, bc_new = get_suby_FP(y_new,ordering=ordering)

    # match initial guess with the one of original DD
    q1c,Pc0 = vc_new
    qc1 = - q1c
    q1_c = ic_gas(qc1)
    P0_c = ic_elec(Pc0)
    bg_new = np.array([q1_c])
    be_new = np.array([P0_c])
    q0_c = vg_new[0]
    P1_c,Q0_c,Q1_c = ve_new
    qc0,Pc1,Qc0,Qc1 = ic_coupling(q0_c,P1_c,Q0_c,Q1_c)
    bc_new = np.array([qc0,Pc1,Qc0,Qc1])
    if ordering == 1:
        y_new = np.concatenate((vg_new,bg_new,ve_new,be_new,vc_new,bc_new))
    elif ordering == 2:
        y_new = np.concatenate((vg_new,ve_new,vc_new,bg_new,be_new,bc_new))
    elif ordering == 3:
        y_new = np.concatenate((vg_new,ve_new,bc_new,vc_new,bg_new,be_new))

    len_y = len(y_new)
    if scale_var == 'matrix':
        vgb = scale_var_params.get('qbase')*np.ones(1)
        bgb = scale_var_params.get('qbase')*np.ones(1)
        veb = scale_var_params.get('Sbase')*np.ones(3)
        beb = scale_var_params.get('Sbase')*np.ones(1)
        vcb = np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase')])
        bcb = np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase')])
        if ordering == 1:
            Dy = sps.diags(1/np.concatenate((vgb,bgb,veb,beb,vcb,bcb)),dtype='float64')
        elif ordering == 2:
            Dy = sps.diags(1/np.concatenate((vgb,veb,vcb,bgb,beb,bcb)),dtype='float64')
        elif ordering == 3:
            Dy = sps.diags(1/np.concatenate((vgb,veb,bcb,vcb,bgb,beb)),dtype='float64')
        Dy_inv = sps.diags(1/Dy.data[0],dtype='float64')
    else:
        Dy = np.eye(len_y)
        Dy_inv = np.eye(len_y)
    # print('Dy: {}, Dy inv: {}'.format(Dy.shape,Dy_inv.shape))

    # iterate
    if der: # use analytical derivatives
        gas_net, elec_net, coupling_net, g_new, err_vecg, err_vece, err_vecc = g_FP(gas_net,elec_net,coupling_net,y_new,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,ordering=ordering)
        gas_net, elec_net, coupling_net,Jg_new = jac_FP(gas_net,elec_net,coupling_net,y_new,formulation=formulation,Dy_inv=Dy_inv,ordering=ordering,scale_var=scale_var,scale_var_params=scale_var_params)
    else:
        gas_net, elec_net, coupling_net,Jg_new, g_new, err_vecg, err_vece, err_vecc = jac_FP_FD(gas_net,elec_net,coupling_net,y_new,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy,Dy_inv=Dy_inv,ordering=ordering)

    F_new = y_new - g_new
    if scale_var == 'matrix': # Should matter, since F and y scale with the same base values...
        # print('Dy={}\nDy_inv={}'.format(Dy,Dy_inv))
        # print('Jg = {}'.format(Jg_new))
        y_new = Dy.dot(y_new)#np.reshape(Dy.dot(y_new),(len_y,)).copy()
        F_new = Dy.dot(F_new)#np.squeeze(Dy.dot(F_new))
        if not der:
            Jg_new = Dy.todense().dot(Jg_new.dot(Dy_inv.todense()))
    J_new = np.eye(len_y,dtype='float64') - Jg_new
    # print('y = {}'.format(y_new.ravel()))
    # print('y: {}, F: {}, J: {}'.format(y_new.shape,F_new.shape,J_new.shape))
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
        # plot overlay / lines in spyplot Jacobian
        for ax in [ax_J,ax_J_map]:
            overlay_FP_jac(ax,y_new,ordering=ordering)
        # plt.show()

    error = np.linalg.norm(F_new)
    err_vec = [error]
    iter_nr = 0
    errors_gas = dict()
    errors_gas[iter_nr] = err_vecg
    errors_elec = dict()
    errors_elec[iter_nr] = err_vece
    errors_coupling = dict()
    errors_coupling[iter_nr] = err_vecc
    errors_gas_DD = dict()
    errors_gas_DD[iter_nr] = err_vecg[-1]
    errors_elec_DD = dict()
    errors_elec_DD[iter_nr] = err_vece[-1]
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

        y_new = y_old + dy # is scaled

        if scale_var == 'matrix':
            dvg,dbg,dve,dbe,dvc,dbc = get_suby_FP(Dy_inv.dot(dy),ordering=ordering)
        else:
            dvg,dbg,dve,dbe,dvc,dbc = get_suby_FP(dy,ordering=ordering)

        if len(errors_gas)==0 or (np.linalg.norm(dbg) > tol_lf) or (err_vecg[-1] > tol_lf):
            solve_LFg = True
        else:
            solve_LFg = False
        if len(errors_elec)==0 or (np.linalg.norm(dbe) > tol_lf) or (err_vece[-1] > tol_lf):
            solve_LFe = True
        else:
            solve_LFe = False
        if len(errors_coupling)==0 or (np.linalg.norm(dbc) > tol_lf) or (err_vecc[-1] > tol_lf):
            solve_LFc = True
        else:
            solve_LFc = False

        if der: # use analytical derivatives
            gas_net, elec_net, coupling_net, g_new, err_vecg, err_vece, err_vecc = g_FP(gas_net,elec_net,coupling_net,Dy_inv.dot(y_new),max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,solve_LFg=solve_LFg,solve_LFe=solve_LFe,solve_LFc=solve_LFc,ordering=ordering)
            gas_net, elec_net, coupling_net,Jg_new = jac_FP(gas_net,elec_net,coupling_net,Dy_inv.dot(y_new),formulation=formulation,Dy_inv=Dy_inv,ordering=ordering,scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            gas_net, elec_net, coupling_net,Jg_new, g_new, err_vecg, err_vece, err_vecc = jac_FP_FD(gas_net,elec_net,coupling_net,Dy_inv.dot(y_new),max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy,Dy_inv=Dy_inv,ordering=ordering)
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
        if solve_LFg:
            errors_gas[iter_nr] = err_vecg
            errors_gas_DD[iter_nr] = err_vecg[-1]
        if solve_LFe:
            errors_elec[iter_nr] = err_vece
            errors_elec_DD[iter_nr] = err_vece[-1]
        if solve_LFc:
            errors_coupling[iter_nr] = err_vecc
            errors_coupling_DD[iter_nr] = err_vecc[-1]
        err_vec.append(error)

    if scale_var == 'matrix':
        y_new = Dy_inv.dot(y_new)
    # print results
    vg_new,bg_new,ve_new,be_new,vc_new,bc_new = get_suby_FP(y_new,ordering=ordering)
    xg_new,xe_new,xc_new,xc_full_new,x_new = get_xLF_FP(gas_net,elec_net,coupling_net,vc_new,bc_new,formulation=formulation)
    with HiddenPrints():
        het_net, gas_net, elec_net = create_mes_network()
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
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('||F mes|| = {:.4e}'.format(np.linalg.norm(nlsys.DF().dot(nlsys.F(x_new)))))
    return het_net, gas_net, elec_net, xg_new, xe_new, xc_new, x_new, iter_nr, err_vec, errors_gas, errors_elec, errors_coupling, errors_gas_DD, errors_elec_DD, errors_coupling_DD

def layout_convergence(ax,tol_lf,tol_dd,art_zero,iters):
    ax.semilogy([0,iters],[tol_dd,tol_dd],'k:',label='tolerance DD')
    ax.semilogy([0,iters],[tol_lf,tol_lf],'k-.',label='tolerance subnetworks')
    ax.semilogy([0,iters],[art_zero,art_zero],'k--',label='artifical 0')
    xmin = 0
    xmax = iters
    xticks = range(xmin,xmax+1,max(1,int(xmax/10))) # make sure the xticks are integers
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

def layout_convergence_order(ax):
    ax.set_xlabel(r'$||D_F F(x^k)||_2 / ||D_F F(x^0)||_2$')
    ax.set_ylabel(r'$||D_F F(x^{k+1})||_2 / ||D_F F(x^0)||_2$')
    x_min,x_max = ax.get_xlim()
    x_slope = np.linspace(x_min,x_max)
    y_slope2 = x_slope**2
    ax.loglog(x_slope,x_slope,linestyle='--',color='k',label='slope 1')
    ax.loglog(x_slope,y_slope2,linestyle='-.',color='k',label='slope 2')
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)

def plot_convergence(ax,iter_vec,err_vec,max_data_points,label,ls='-',marker='.',color='tab:blue',art_zero=1e-16,alpha=.7):
    """Plot the error per iteration. Only the number of data point equal to max_data_points+1 are plotted. These data points are equally distributed over the error vector."""
    if len(err_vec) > max_data_points:
        indices = [ind for ind in range(0,len(err_vec),len(err_vec)//max_data_points)]
        if len(err_vec)-1 not in indices:
            indices.append(len(err_vec)-1)
        ax.semilogy([iter_vec[ind] for ind in indices],[max(err_vec[ind],art_zero) for ind in indices],marker=marker,ls=ls,label=label,color=color,alpha=alpha)
    else:
        ax.semilogy(iter_vec,[max(err,art_zero) for err in err_vec],ls=ls,marker=marker,label=label,color=color,alpha=alpha)

def plot_convergence_order(ax,err_vec,max_data_points,label,ls='-',marker='.',color='tab:blue',alpha=.7):
    """Plot the convergence order. That is, plot the (normalized) error of iteration k+1 vs the (normalized) error of iteration k. Only the number of data point equal to max_data_points+1 are plotted. These data points are equally distributed over the error vector."""

    if len(err_vec) > max_data_points:
        indices = [ind for ind in range(0,len(err_vec)-1,(len(err_vec)-1)//max_data_points)]
        err_vec_k=  [err_vec[ind] for ind in indices]
        err_vec_kplus1=  [err_vec[ind+1] for ind in indices]
    else:
        err_vec_k = err_vec[:-1]
        err_vec_kplus1 = err_vec[1:]
    ax.loglog(err_vec_k/err_vec[0],err_vec_kplus1/err_vec[0],marker=marker,label=label,color=color,alpha=alpha)

def plot_convergence_detailed(ax, iter_vec, err_vec, errors_gas, errors_elec, errors_coupling, errors_gas_DD, errors_elec_DD, errors_coupling_DD,tol_lf,tol_dd,art_zero,max_data_points,alpha=0.7):
    """Plot the errors and suberrors per outer iteration of DD. Assumes the results of one DD run."""
    # plot final error of LF subsolve per outer DD iter, for each SC network
    itersg = list()
    err_vec_gas = list()
    for iterg, err_vecg in errors_gas.items():
        itersg.append(iterg)
        err_vec_gas.append(max(art_zero,err_vecg[-1]))
    plot_convergence(ax,itersg,err_vec_gas,max_data_points,'gas subsystem ',ls='--',marker='.',color=colors_carrier.get('gas'),alpha=alpha)
    iterse = list()
    err_vec_elec = list()
    for itere, err_vece in errors_elec.items():
        iterse.append(itere)
        err_vec_elec.append(max(art_zero,err_vece[-1]))
    plot_convergence(ax,iterse,err_vec_elec,max_data_points,'elec subsystem',ls='--',marker='.',color=colors_carrier.get('elec'),alpha=alpha)
    itersc = list()
    err_vec_coupling = list()
    for iterc, err_vecc in errors_coupling.items():
        itersc.append(iterc)
        err_vec_coupling.append(max(art_zero,err_vecc[-1]))
    plot_convergence(ax,itersc,err_vec_coupling,max_data_points,'coupling subsystem',ls='--',marker='.',color=colors_carrier.get('coupling'),alpha=alpha)
    # plot the error ||F|| of the subnetwork per outer DD iter, for each SC network. Only plottes if ||F|| is used as error measure
    if len(errors_gas_DD):
        itersg_DD = list()
        err_vec_gas_DD = list()
        for iterg, err_g in errors_gas_DD.items():
            itersg_DD.append(iterg)
            err_vec_gas_DD.append(max(art_zero,err_g))
        plot_convergence(ax,itersg_DD,err_vec_gas_DD,max_data_points,r'$||F^g||_2$',ls='-',marker='.',color=colors_carrier.get('gas'),alpha=alpha)
    if len(errors_elec_DD):
        iterse_DD = list()
        err_vec_elec_DD = list()
        for itere, err_e in errors_elec_DD.items():
            iterse_DD.append(itere)
            err_vec_elec_DD.append(max(art_zero,err_e))
        plot_convergence(ax,iterse_DD,err_vec_elec_DD,max_data_points,r'$||F^e||_2$',ls='-',marker='.',color=colors_carrier.get('elec'),alpha=alpha)
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
    vg_new, bg_new, ve_new, be_new, vc_new, bc_new = get_suby_FP(y,ordering=ordering)
    yg_len = 2
    ye_len = 4
    yc_len = 6
    y_len = yg_len+ye_len+yc_len
    len_v = len(vg_new)+len(ve_new)+len(vc_new)
    if ordering == 1:
        # horizontal lines
        ax.plot((-0.5,y_len-0.5),(len(vg_new)-0.5,len(vg_new)-0.5),'k--',alpha=.5)
        ax.plot((-0.5,y_len-0.5),(yg_len-0.5,yg_len-0.5),'k-')
        ax.plot((-0.5,y_len-0.5),(yg_len+len(ve_new)-0.5,yg_len+len(ve_new)-0.5),'k--',alpha=.5)
        ax.plot((-0.5,y_len-0.5),(yg_len+ye_len-0.5,yg_len+ye_len-0.5),'k-')
        ax.plot((-0.5,y_len-0.5),(yg_len+ye_len+len(vc_new)-0.5,yg_len+ye_len+len(vc_new)-0.5),'k--',alpha=.5)
        # vertical lines
        ax.plot((len(vg_new)-0.5,len(vg_new)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.plot((yg_len-0.5,yg_len-0.5),(0-0.5,y_len-0.5),'k-')
        ax.plot((yg_len+len(ve_new)-0.5,yg_len+len(ve_new)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.plot((yg_len+ye_len-0.5,yg_len+ye_len-0.5),(0-0.5,y_len-0.5),'k-')
        ax.plot((yg_len+ye_len+len(vc_new)-0.5,yg_len+ye_len+len(vc_new)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
    elif ordering == 2:
        # horizontal lines
        ax.plot((-0.5,y_len-0.5),(len(vg_new)-0.5,len(vg_new)-0.5),'k--',alpha=.5)
        ax.text(len(vg_new)/2-0.5,-1,r'$v^g$', horizontalalignment='center',verticalalignment='center',color='g')
        ax.plot((-0.5,y_len-0.5),(len(vg_new)+len(ve_new)-0.5,len(vg_new)+len(ve_new)-0.5),'k--',alpha=.5)
        ax.text(len(vg_new)+len(ve_new)/2-0.5,-1,r'$v^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((-0.5,y_len-0.5),(len_v-0.5,len_v-0.5),'k-')
        ax.text(len(vg_new)+len(ve_new)+len(vc_new)/2-0.5,-1,r'$v^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.plot((-0.5,y_len-0.5),(len_v+len(bg_new)-0.5,len_v+len(bg_new)-0.5),'k--',alpha=.5)
        ax.text(len_v+len(bg_new)/2-0.5,-1,r'$b^g$', horizontalalignment='center',verticalalignment='center',color='g')
        ax.plot((-0.5,y_len-0.5),(len_v+len(bg_new)+len(be_new)-0.5,len_v+len(bg_new)+len(be_new)-0.5),'k--',alpha=.5)
        ax.text(len_v+len(bg_new)+len(be_new)/2-0.5,-1,r'$b^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.text(len_v+len(bg_new)+len(be_new)+len(bc_new)/2-0.5,-1,r'$b^c$', horizontalalignment='center',verticalalignment='center',color='k')
        # vertical lines
        ax.plot((len(vg_new)-0.5,len(vg_new)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,len(vg_new)/2-0.5,r'$g^g$', horizontalalignment='center',verticalalignment='center',color='g')
        ax.plot((len(vg_new)+len(ve_new)-0.5,len(vg_new)+len(ve_new)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,len(vg_new)+len(ve_new)/2-0.5,r'$g^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.plot((len_v-0.5,len_v-0.5),(0-0.5,y_len-0.5),'k-')
        ax.text(-1.5,len(vg_new)+len(ve_new)+len(vc_new)/2-0.5,r'$g^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.plot((len_v+len(bg_new)-0.5,len_v+len(bg_new)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,len_v+len(bg_new)/2-0.5,r'$g^{I,g}$', horizontalalignment='center',verticalalignment='center',color='g')
        ax.plot((len_v+len(bg_new)+len(be_new)-0.5,len_v+len(bg_new)+len(be_new)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.text(-1.5,len_v+len(bg_new)+len(be_new)/2-0.5,r'$g^{I,e}$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.text(-1.5,len_v+len(bg_new)+len(be_new)+len(bc_new)/2-0.5,r'$g^{I,c}$', horizontalalignment='center',verticalalignment='center',color='k')
    elif ordering == 3:
        # horizontal lines
        ax.plot((-0.5,y_len-0.5),(len(vg_new)-0.5,len(vg_new)-0.5),'k--',alpha=.5)
        ax.plot((-0.5,y_len-0.5),(len(vg_new)+len(ve_new)-0.5,len(vg_new)+len(ve_new)-0.5),'k-',alpha=.5)
        ax.plot((-0.5,y_len-0.5),(len_v+len(bc_new)-0.5,len_v+len(bc_new)-0.5),'k-')
        ax.plot((-0.5,y_len-0.5),(len(vg_new)+len(ve_new)+len(bc_new)-0.5,len(vg_new)+len(ve_new)+len(bc_new)-0.5),'k-')
        ax.plot((-0.5,y_len-0.5),(len_v+len(bc_new)+len(bg_new)-0.5,len_v+len(bc_new)+len(bg_new)-0.5),'k--',alpha=.5)
        # row annotation
        ax.text(-1.5,len(vg_new)/2-0.5,r'$g^g$', horizontalalignment='center',verticalalignment='center',color='g')
        ax.text(-1.5,len(vg_new)+len(ve_new)/2-0.5,r'$g^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.text(-1.5,len(vg_new)+len(ve_new)+len(bc_new)/2-0.5,r'$g^{I,c}$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.text(-1.5,len(vg_new)+len(ve_new)+len(bc_new)+len(vc_new)/2-0.5,r'$g^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.text(-1.5,len_v+len(bc_new)+len(bg_new)/2-0.5,r'$g^{I,g}$', horizontalalignment='center',verticalalignment='center',color='g')
        ax.text(-1.5,len_v+len(bc_new)+len(bg_new)+len(be_new)/2-0.5,r'$g^{I,e}$', horizontalalignment='center',verticalalignment='center',color='r')
        # vertical lines
        ax.plot((len(vg_new)-0.5,len(vg_new)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        ax.plot((len(vg_new)+len(ve_new)-0.5,len(vg_new)+len(ve_new)-0.5),(0-0.5,y_len-0.5),'k-',alpha=.5)
        ax.plot((len(vg_new)+len(ve_new)+len(bc_new)-0.5,len(vg_new)+len(ve_new)+len(bc_new)-0.5),(0-0.5,y_len-0.5),'k-',alpha=.5)
        ax.plot((len_v+len(bc_new)-0.5,len_v+len(bc_new)-0.5),(0-0.5,y_len-0.5),'k-',alpha=.5)
        ax.plot((len_v+len(bc_new)+len(bg_new)-0.5,len_v+len(bc_new)+len(bg_new)-0.5),(0-0.5,y_len-0.5),'k--',alpha=.5)
        # columns annotation
        ax.text(len(vg_new)/2-0.5,-1,r'$v^g$', horizontalalignment='center',verticalalignment='center',color='g')
        ax.text(len(vg_new)+len(ve_new)/2-0.5,-1,r'$v^e$', horizontalalignment='center',verticalalignment='center',color='r')
        ax.text(len(vg_new)+len(ve_new)+len(bc_new)/2-0.5,-1,r'$b^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.text(len(vg_new)+len(ve_new)+len(bc_new)+len(vc_new)/2-0.5,-1,r'$v^c$', horizontalalignment='center',verticalalignment='center',color='k')
        ax.text(len_v+len(bc_new)+len(bg_new)/2-0.5,-1,r'$b^g$', horizontalalignment='center',verticalalignment='center',color='g')
        ax.text(len_v+len(bc_new)+len(bg_new)+len(be_new)/2-0.5,-1,r'$b^e$', horizontalalignment='center',verticalalignment='center',color='r')

def plot_solution(ax_pres,ax_q,ax_V_amp,ax_V_ang,ax_P,ax_Q,ax_en,het_net,label,color='tab:blue',marker='.',alpha=.7):
    """Plot the solution per carrier ena variable"""
    q = list()
    xticks_q_label = list()
    P = list()
    xticks_P_label = list()
    Q = list()
    xticks_Q_label = list()
    Ec = list()
    for link in het_net.get_links():
        if isinstance(link,GasLink):
            if link.link_type == 'dummy':
                Ec.append(link.get_q()*GHV)
            else:
                q.append(link.get_q())
                xticks_q_label.append(r'$q_{01}$')
        elif isinstance(link,ElectricalLink):
            if link.link_type == 'dummy':
                Ec.append(link.get_Pstart())
                Ec.append(link.get_Qstart())
            else:
                P.append(link.get_Pstart())
                xticks_P_label.append(r'$P_{01}$')
                Q.append(link.get_Qstart())
                xticks_Q_label.append(r'$Q_{01}$')

    pres = list()
    V_amp = list()
    V_ang = list()
    for node in het_net.get_nodes():
        if isinstance(node,GasNode):
            pres.append(node.get_p())
            for hl in node.get_half_links():
                q.append(hl.get_q())
                xticks_q_label.append(r'$q_{'+str(hl.start_node.number)+r'}$')
        elif isinstance(node,ElectricalNode):
            V_amp.append(node.get_V())
            V_ang.append(node.get_delta())
            for hl in node.get_half_links():
                P.append(hl.get_P())
                xticks_P_label.append(r'$P_{'+str(hl.start_node.number)+r'}$')
                Q.append(hl.get_Q())
                xticks_Q_label.append(r'$Q_{'+str(hl.start_node.number)+r'}$')
    ax_pres.plot(pres,ls='-',marker=marker,color=color,label=label,alpha=alpha)
    ax_V_amp.plot(V_amp,ls='-',marker=marker,color=color,label=label,alpha=alpha)
    ax_V_ang.plot(V_ang,ls='-',marker=marker,color=color,label=label,alpha=alpha)

    ax_q.plot(q,ls='-',marker=marker,color=color,label=label,alpha=alpha)
    ax_P.plot(P,ls='-',marker=marker,color=color,label=label,alpha=alpha)
    ax_Q.plot(Q,ls='-',marker=marker,color=color,label=label,alpha=alpha)
    ax_en.plot(Ec,ls='-',marker=marker,color=color,label=label,alpha=alpha)

    xticks_en = list(range(len(labels_energy)))
    xticks_nodes = [0,1]
    ax_en.set_xticks(xticks_en)
    ax_en.set_xticklabels(labels_energy)
    ax_pres.set_xticks(xticks_nodes)
    ax_pres.set_xticklabels([r'$p_0$',r'$p_1$'])
    ax_V_amp.set_xticks(xticks_nodes)
    ax_V_amp.set_xticklabels([r'$|V_0|$',r'$|V_1|$'])
    ax_V_ang.set_xticks(xticks_nodes)
    ax_V_ang.set_xticklabels([r'$\delta_0$',r'$\delta_1$'])
    ax_q.set_xticks(list(range(len(xticks_q_label))))
    ax_q.set_xticklabels(xticks_q_label)
    ax_P.set_xticks(list(range(len(xticks_P_label))))
    ax_P.set_xticklabels(xticks_P_label)
    ax_Q.set_xticks(list(range(len(xticks_Q_label))))
    ax_Q.set_xticklabels(xticks_Q_label)

def compare_DD(tol_lf = 1e-6,tol_dd = 1e-6,art_zero = 1e-16,max_iter_dd = 200,max_iter_lf = 10,max_data_points = 200,scale_var=None,scale_var_params=None,save_data=False,path_to_data=None):
    """Compare the results and the convergende of DD with integrated LF."""
    if scale_var == None:
        scale_var_label = 'unscaled'
    else:
        scale_var_label = scale_var

    error_measures=['F','dE']
    forms_gas = ['full']#,'nodal'] NB: integrated mes does not convergence with nodal! Neither does DD.
    max_iters_used = 0
    decouplings = [True,False]
    derivatives = [True,False]
    orderings = [1,2,3]
    x_mes_sols = dict()
    result_DD = dict()
    result_DD['tol LF'] = tol_lf
    result_DD['tol DD'] = tol_dd
    for form_gas in forms_gas:
        formulation={'gas':form_gas,'elec':'complex_power'}
        fig = plt.figure('conv_'+scale_var_label+'_'+form_gas)
        ax = fig.gca()
        fig = plt.figure('conv_order_'+scale_var_label+'_'+form_gas)
        ax_ord = fig.gca()
        fig = plt.figure('coupling_energies_'+scale_var_label+'_'+form_gas)
        ax_en = fig.gca()
        fig = plt.figure('pressures_'+scale_var_label+'_'+form_gas)
        ax_pres = fig.gca()
        fig = plt.figure('gasflow_'+scale_var_label+'_'+form_gas)
        ax_q = fig.gca()
        fig = plt.figure('volt_amp_'+scale_var_label+'_'+form_gas)
        ax_V_amp = fig.gca()
        fig = plt.figure('volt_ang_'+scale_var_label+'_'+form_gas)
        ax_V_ang = fig.gca()
        fig = plt.figure('acpow_'+scale_var_label+'_'+form_gas)
        ax_P = fig.gca()
        fig = plt.figure('reacpow_'+scale_var_label+'_'+form_gas)
        ax_Q = fig.gca()
        for error_measure in error_measures:
            fig = plt.figure('conv_detail_'+error_measure+'_'+scale_var_label+'_'+form_gas)
            ax_conv = fig.gca()
            het_net, gas_net, elec_net, xg_new, xe_new, xc_new, x_new, iter_nr, err_vec, errors_gas, errors_elec, errors_coupling, errors_gas_DD, errors_elec_DD, errors_coupling_DD = run_LF_mes_DD(tol_lf=tol_lf,tol_dd=tol_dd,max_iter_dd=max_iter_dd,error_measure=error_measure,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation)
            if error_measure == 'F':
                iter_start = 0
                iter_range = iter_nr - iter_start
            elif error_measure == 'dE':
                iter_start = 1
                iter_range = iter_nr - iter_start
            key_sol = 'first version DD '+error_measure
            x_mes_sols[key_sol] = x_new
            result_DD[key_sol] = {'x mes':x_new,'outer errors':err_vec,'inner LF errors gas':errors_gas,'outer error gas':errors_gas_DD,'inner LF errors elec':errors_elec,'outer error elec':errors_elec_DD,'inner LF errors coupling':errors_coupling,'outer error coupling':errors_coupling_DD}
            plot_convergence_detailed(ax_conv,range(iter_start,iter_nr+1), err_vec, errors_gas, errors_elec, errors_coupling, errors_gas_DD, errors_elec_DD, errors_coupling_DD,tol_lf,tol_dd,art_zero,max_data_points,alpha=.5)
            plot_convergence(ax,range(iter_start,iter_nr+1),err_vec,max_data_points,'DD '+error_measure,ls='-',marker=markers_solver.get(error_measure),color=colors_solver.get('dd'),alpha=.5)
            plot_convergence_order(ax_ord ,err_vec,max_data_points,'DD '+error_measure,ls='-',marker=markers_solver.get(error_measure),color=colors_solver.get('dd'),alpha=.5)
            max_iters_used = max(max_iters_used,iter_nr)
            plot_solution(ax_pres,ax_q,ax_V_amp,ax_V_ang,ax_P,ax_Q,ax_en,het_net,'DD '+error_measure,marker=markers_solver.get(error_measure),color=colors_solver.get('dd'),alpha=.5)
            layout_convergence(ax_conv,tol_lf,tol_dd,art_zero,max_iters_used)
            # DD mes using FP iteration
            if error_measure == 'F':
                error_measure_FP = 'F'
            elif error_measure == 'dE':
                error_measure_FP = 'dy'
            for ordering in orderings:
                for decoupled in decouplings:
                    label = 'DD FP '+error_measure_FP
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
                        fig = plt.figure('conv_detail_FP_decoup_'+error_measure_FP+'_'+str(ordering)+'_'+scale_var_label+'_'+form_gas)
                    else:
                        fig = plt.figure('conv_detail_FP_'+error_measure_FP+'_'+str(ordering)+'_'+scale_var_label+'_'+form_gas)
                    key_sol = key + ' ' + error_measure_FP
                    ax_conv = fig.gca()
                    max_iters_used_det = 0
                    het_net, gas_net, elec_net, xg_new, xe_new, xc_new, x_new, iter_nr, err_vec, errors_gas, errors_elec, errors_coupling, errors_gas_DD, errors_elec_DD, errors_coupling_DD = run_LF_mes_DD_FP(max_iter_lf=max_iter_lf,max_iter_fp=max_iter_dd,tol_lf=tol_lf,tol_fp=tol_dd,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,error_measure=error_measure_FP,decoupled=decoupled,ordering=ordering)
                    x_mes_sols[key_sol] = x_new
                    result_DD[key_sol] = {'x mes':x_new,'outer errors':err_vec,'inner LF errors gas':errors_gas,'outer error gas':errors_gas_DD,'inner LF errors elec':errors_elec,'outer error elec':errors_elec_DD,'inner LF errors coupling':errors_coupling,'outer error coupling':errors_coupling_DD}
                    if error_measure == 'F':
                        iter_start = 0
                        iter_range = iter_nr - iter_start
                    elif error_measure == 'dE':
                        iter_start = 1
                        iter_range = iter_nr - iter_start
                    max_iters_used = max(max_iters_used,iter_nr)
                    max_iters_used_det = max(max_iters_used_det,iter_nr)
                    plot_convergence_detailed(ax_conv,range(iter_start,iter_nr+1), err_vec, errors_gas, errors_elec, errors_coupling, errors_gas_DD, errors_elec_DD, errors_coupling_DD,tol_lf,tol_dd,art_zero,max_data_points,alpha=.5)
                    plot_convergence(ax,range(iter_start,iter_nr+1),err_vec,max_data_points,label,ls='-',marker=markers_solver.get(error_measure),color=colors_solver.get(key),alpha=.5)
                    plot_convergence_order(ax_ord ,err_vec,max_data_points,label,ls='-',marker=markers_solver.get(error_measure),color=colors_solver.get(key),alpha=.5)
                    plot_solution(ax_pres,ax_q,ax_V_amp,ax_V_ang,ax_P,ax_Q,ax_en,het_net,label,marker=markers_solver.get(error_measure),color=colors_solver.get(key),alpha=.5)
                    layout_convergence(ax_conv,tol_lf,tol_dd,art_zero,max_iters_used_det)

        for der in derivatives:
            for ordering in orderings:
                het_net, gas_net, elec_net, xg_new, xe_new, xc_new, x_new, iter_nr, err_vec, errors_gas, errors_elec, errors_coupling, errors_gas_DD, errors_elec_DD, errors_coupling_DD = run_LF_mes_DD_FP_NR(max_iter_lf=max_iter_lf,max_iter_fp=max_iter_dd,tol_lf=tol_lf,tol_fp=tol_dd,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,plot_jac=True,der=der,ordering=ordering)
                iter_start = 0
                iter_range = iter_nr - iter_start
                label = 'DD FP NR'
                key_color = 'fp NR'
                if ordering == 2:
                    key_color += ' reod'
                    label += ' ord. 2'
                if ordering == 3:
                    key_color += ' reod 2'
                    label += ' ord. 3'
                if der:
                    fig = plt.figure('conv_detail_FP_NR_'+str(ordering)+'_'+scale_var_label+'_'+form_gas)
                    key = 'an'
                else:
                    fig = plt.figure('conv_detail_FP_NR_FD_'+str(ordering)+'_'+scale_var_label+'_'+form_gas)
                    label += ' FD'
                    key = 'FD'
                key_sol = key_color + ' ' + key
                ax_conv = fig.gca()
                x_mes_sols[key_sol] = x_new
                result_DD[key_sol] = {'x mes':x_new,'outer errors':err_vec,'inner LF errors gas':errors_gas,'outer error gas':errors_gas_DD,'inner LF errors elec':errors_elec,'outer error elec':errors_elec_DD,'inner LF errors coupling':errors_coupling,'outer error coupling':errors_coupling_DD}
                plot_convergence_detailed(ax_conv,range(iter_start,iter_nr+1), err_vec, errors_gas, errors_elec, errors_coupling, errors_gas_DD, errors_elec_DD, errors_coupling_DD,tol_lf,tol_dd,art_zero,max_data_points,alpha=.5)
                layout_convergence(ax_conv,tol_lf,tol_dd,art_zero,iter_nr)
                plot_convergence(ax,range(iter_start,iter_nr+1),err_vec,max_data_points,label,ls='-',marker=markers_solver.get(key),color=colors_solver.get(key_color),alpha=.5)
                plot_convergence_order(ax_ord ,err_vec,max_data_points,label,ls='-',marker=markers_solver.get(key),color=colors_solver.get(key_color),alpha=.5)
                plot_solution(ax_pres,ax_q,ax_V_amp,ax_V_ang,ax_P,ax_Q,ax_en,het_net,label,marker=markers_solver.get(key),color=colors_solver.get(key_color),alpha=.5)

        # integrated mes, original order
        with HiddenPrints():
            het_net, gas_net, elec_net,x_sol,iters,err_vec = run_LF_mes(max_iter=max_iter_lf,tol=tol_dd,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation,plot_top=False,plot_jac=False,plot_sol=False)
        x_mes_sols['mes'] = x_sol
        result_DD['mes'] = {'x mes':x_sol,'outer errors':err_vec}
        plot_convergence(ax,range(iters+1),err_vec,max_data_points,'int. LF',ls='-',marker=markers_solver.get('mes'),color=colors_solver.get('mes'),alpha=.5)
        plot_convergence_order(ax_ord,err_vec,max_data_points,'int. LF',ls='-',marker=markers_solver.get('mes'),color=colors_solver.get('mes'),alpha=.5)
        plot_solution(ax_pres,ax_q,ax_V_amp,ax_V_ang,ax_P,ax_Q,ax_en,het_net,'int. LF',color=colors_solver.get('mes'),marker=markers_solver.get('mes'),alpha=.5)
        # integrated mes, reorder to match DD
        with HiddenPrints():
            het_net, gas_net, elec_net,x_sol,iters,err_vec = run_LF_mes(max_iter=max_iter_lf,tol=tol_dd,scale_var=scale_var,scale_var_params=scale_var_params,formulation=formulation,plot_top=False,plot_jac=False,plot_sol=False,reorder=True)
        x_mes_sols['mes reod 2'] = x_sol
        result_DD['mes reod 2'] = {'x mes':x_sol,'outer errors':err_vec}
        plot_convergence(ax,range(iters+1),err_vec,max_data_points,'int. LF, reod.',ls='-',marker=markers_solver.get('mes'),color=colors_solver.get('mes reod 2'),alpha=.5)
        plot_convergence_order(ax_ord,err_vec,max_data_points,'int. LF, reod.',ls='-',marker=markers_solver.get('mes'),color=colors_solver.get('mes reod 2'),alpha=.5)
        plot_solution(ax_pres,ax_q,ax_V_amp,ax_V_ang,ax_P,ax_Q,ax_en,het_net,'int. LF, reod.',color=colors_solver.get('mes reod 2'),marker=markers_solver.get('reod'),alpha=.5)
        # # Block-Jacobi (NOT implemented correctly!)
        # het_net, gas_net, elec_net,x_sol,iters,err_vec = block_jacobi(max_iter=max_iter_lf,max_iter_bj=max_iter_dd,tol=tol_lf,scale_var=scale_var,scale_var_params=scale_var_params)
        # plot_convergence(ax,range(iters+1),err_vec,max_data_points,'block jacobi LF',ls='-',marker='.',color=colors_solver.get('block jacobi'))
        # plot_convergence_order(ax_ord,err_vec,max_data_points,'block jacobi LF',ls='-',marker='.',color=colors_solver.get('block jacobi'))
        # plot_solution(ax_pres,ax_q,ax_V_amp,ax_V_ang,ax_P,ax_Q,ax_en,het_net,'block jacobi LF',color=colors_solver.get('block jacobi'),marker=markers_solver.get('mes'))
        layout_convergence(ax,tol_lf,tol_dd,art_zero,max_iters_used)
        layout_convergence_order(ax_ord)
        ax.legend()
        ax_ord.legend()
        ax_pres.legend()
        ax_q.legend()
        ax_V_amp.legend()
        ax_V_ang.legend()
        ax_P.legend()
        ax_Q.legend()
        ax_en.legend()

    print('\nRelative error to LF solution of integrated mes:')
    for key, x_mes_sol in x_mes_sols.items():
        print('For {}: rel. err.= {:.4e}'.format(key,error(x_mes_sol,x_mes_sols.get('mes'))))

    if save_data:
        with open(os.path.join(path_to_data,'x_mes_DD_'+scale_var_label+'.pkl'), "wb") as x_mes_file:
            pickle.dump(x_mes_sols,x_mes_file)
        with open(os.path.join(path_to_data,'result_DD_'+scale_var_label+'.pkl'), "wb") as result_DD_file:
            pickle.dump(result_DD,result_DD_file)
def compare_jacobians(formulation={'gas':'full','elec':'complex_power'},scale_var_params={'pgbase':pg0,'pbase':pg0,'qbase':.1,'Vbase':Vbase,'deltabase':1,'Sbase':Sbase,'Ebase':Sbase}):
    """Plot the Jacobians and their spectra for the various solution methods"""
    marker_size = 10
    # Integrated MES
    het_net, gas_net, elec_net = create_mes_network()
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
    fig_J_re = nlsys_unscaled.spy_plot_J(x,title='Reordered Jacobian spy plot',P_F=TF,P_x=Tx,overlay=False)
    ax_J_re = plt.gca()
    nlsys_unscaled.spectrum_J(x,ax=ax_spec,P_F=TF,P_x=Tx,color=colors_solver.get('mes reod 2'))
    # reordered system, scaled
    nlsys.spy_plot_J(x,ax=ax_J_re,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5,P_F=TF,P_x=Tx,overlay=False)
    fig_J_re_scaled_map = nlsys.imshow_J(x,P_F=TF,P_x=Tx,title='Reordered, scaled, Jacobian',overlay=False)
    ax_J_re_scaled_map = fig_J_re_scaled_map.gca()
    nlsys.spectrum_J(x,ax=ax_spectra,P_F=TF,P_x=Tx,color=colors_solver.get('mes reod 2'))
    # plot overlay / lines in reoreded Jacobians
    xg_len = 3
    xe_len = 4
    xc_len = 2
    x_len = xg_len+xe_len+xc_len
    axes = [ax_J_re,ax_J_re_scaled_map]
    for ax in axes:
        ax.plot((-0.5,x_len-0.5),(xg_len-0.5,xg_len-0.5),'k-')
        ax.plot((-0.5,x_len-0.5),(xg_len+xe_len-0.5,xg_len+xe_len-0.5),'k-')
        ax.plot((xg_len-0.5,xg_len-0.5),(0-0.5,x_len-0.5),'k-')
        ax.plot((xg_len+xe_len-0.5,xg_len+xe_len-0.5),(0-0.5,x_len-0.5),'k-')

    legend_handles_spec = [Line2D([0],[0],color=colors_solver.get('mes'),marker=markers_solver.get('mes'),label='Int. MES'),Line2D([0], [0], color=colors_solver.get('mes reod 2'), marker=markers_solver.get('mes'),label='Int. MES reord.')]
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
            key_color_NR += ' reod'
            key_color_dgdy += ' reod'
        if ordering == 3:
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
                gas_net = create_gas_network(scale_var=scale_var,scale_var_params=scale_var_params)
                elec_net = create_electrical_network(scale_var=scale_var,scale_var_params=scale_var_params)
                elec_net = reset_network_without_update_elec(elec_net)
                coupling_net = create_coupling_network(scale_var=scale_var,scale_var_params=scale_var_params)
                xg_new = set_xg_LF_init(gas_net,formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
                xe_new = set_xe_LF_init(elec_net,scale_var=scale_var,scale_var_params=scale_var_params)
                with HiddenPrints():
                    xc_new = set_xc_LF_init(coupling_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

                y_new = get_y_FP(gas_net,elec_net,coupling_net,formulation=formulation,ordering=ordering)
                vg_new, bg_new, ve_new, be_new, vc_new, bc_new = get_suby_FP(y_new,ordering=ordering)

                len_y = len(y_new)
                if scale_var == 'matrix':
                    vgb = scale_var_params.get('qbase')*np.ones(1)
                    bgb = scale_var_params.get('qbase')*np.ones(1)
                    veb = scale_var_params.get('Sbase')*np.ones(3)
                    beb = scale_var_params.get('Sbase')*np.ones(1)
                    vcb = np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase')])
                    bcb = np.array([scale_var_params.get('qbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase'),scale_var_params.get('Sbase')])
                    if ordering == 1:
                        Dy = sps.diags(1/np.concatenate((vgb,bgb,veb,beb,vcb,bcb)),dtype='float64')
                    else:
                        Dy = sps.diags(1/np.concatenate((vgb,veb,vcb,bgb,beb,bcb)),dtype='float64')
                    Dy_inv = sps.diags(1/Dy.data[0],dtype='float64')
                else:
                    Dy = np.eye(len_y)
                    Dy_inv = np.eye(len_y)
                if der: # use analytical derivatives
                    gas_net, elec_net, coupling_net, g_new, err_vecg, err_vece, err_vecc = g_FP(gas_net,elec_net,coupling_net,y_new,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,ordering=ordering)
                    gas_net, elec_net, coupling_net,Jg_new = jac_FP(gas_net,elec_net,coupling_net,y_new,formulation=formulation,Dy_inv=Dy_inv,ordering=ordering,scale_var=scale_var,scale_var_params=scale_var_params)
                else:
                    gas_net, elec_net, coupling_net,Jg_new, g_new, err_vecg, err_vece, err_vecc = jac_FP_FD(gas_net,elec_net,coupling_net,y_new,max_iter_lf=max_iter_lf,tol_lf=tol_lf,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,Dy=Dy,Dy_inv=Dy_inv,ordering=ordering)
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

def table_LF_results(path_to_data,max_iter=10,tol=1e-6,scale_var=None,scale_var_params=None):
    """Write the full solution to a txt file, which can be written by latex to create a table."""
    formulation={'gas':'full','elec':'complex_power'}
    with HiddenPrints():
        het_net, gas_net, elec_net, x_sol, iters, err_vec = run_LF_mes(max_iter=max_iter,tol=tol,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,plot_top=False,plot_jac=False,plot_sol=False,reorder=False)
        x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)

    nodes = ['0','1']
    links = ['0--1']
    with open(os.path.join(path_to_data,'gas_data_GE2N_thesis.txt'), "w") as table:
        for ind in range(len(nodes)):
            node_data = ' ' + nodes[ind] + r' '
            if ind < len(links):
                link_data = ' ' + links[ind] + r' '
            else:
                link_data = ' '
            if gas_net.nodes[ind] in list(xg_entries):
                node_data += r' & {:.3f} '.format(gas_net.nodes[ind].get_p()/mbar)
            else:
                node_data += r' & \textbf{'+'{:.3f}'.format(gas_net.nodes[ind].get_p()/mbar)+'} '
            if gas_net.nodes[ind].half_links[0].bc_type == 1: # q known
                node_data += r' & \textbf{'+'{:.3f}'.format(gas_net.nodes[ind].half_links[0].get_q())+'} '
            else:
                node_data += r' & {:.3f} '.format(gas_net.nodes[ind].half_links[0].get_q())
            if ind < len(links):
                link_data += r' & {:.3f}'.format(gas_net.links[ind].get_q())
            else:
                link_data += r' & '
            table.write(node_data)
            table.write(r' & ')
            table.write(link_data)
            table.write(r' & ')

    P_loss_total = 0
    Q_loss_total = 0
    with open(os.path.join(path_to_data,'elec_data_GE2N_thesis.txt'), "w") as table:
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
                    node_data += r' & {:.1f} '.format(hl.get_P()/MW)
                if hl.bc_type in [1,3]: # Q known
                    node_data += r'\textbf{'+'{:+.1f}$\iu$  '.format(hl.get_Q()/MW)+'} '
                else:
                    node_data += r'{:+.1f}$\iu$  '.format(hl.get_Q()/MW)
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

    unit = [r'\acrshort{gg} at 0',r'\acrshort{gg} at 1']
    with open(os.path.join(path_to_data,'coupling_data_GE2N_thesis.txt'), "w") as table:
        for ind in range(len(unit)):
            link_data = ' ' + unit[ind] + r' '
            # gas data
            if gas_net.links[ind-2] in unknown_qc_links:
                link_data += r' & {:.3f}'.format(gas_net.links[ind-2].get_q())
            else:
                link_data += r' & \textbf{'+'{:.3f}'.format(gas_net.links[ind-2].get_q())+'} '
            # elec data
            if elec_net.links[ind-2] in unknown_Pc_links:
                link_data += r' & {:.3f}'.format(elec_net.links[ind-2].get_Pstart()/MW)
            else:
                link_data += r' & \textbf{'+'{:.3f}'.format(elec_net.links[ind-2].get_Pstart()/MW)+'} '
            if elec_net.links[ind-2] in unknown_Qc_links:
                link_data += r' & {:.3f} '.format(elec_net.links[ind-2].get_Qstart()/MW)
            else:
                link_data += r' & \textbf{'+'{:.3f}'.format(elec_net.links[ind-2].get_Qstart()/MW)+'} '
            table.write(link_data)
            table.write(r' & ')

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','GE2N')

    # run_LF_gas()
    # run_LF_elec()
    # run_LF_mes(plot_top=True,plot_jac=True,plot_sol=True)
    # run_LF_coupling()

    tol_lf = 1e-7#1e-6
    tol_dd = 100*tol_lf#1e-6
    art_zero = 1e-16
    max_iter_dd = 800#600
    max_iter_lf = 10
    max_data_points = 80
    # # unscaled
    # scale_var=None
    # scale_var_params=None
    # compare_DD(tol_lf=tol_lf,tol_dd=tol_dd,art_zero=art_zero,max_iter_dd=max_iter_dd,max_iter_lf=max_iter_lf,max_data_points=max_data_points,scale_var=scale_var,scale_var_params=scale_var_params)

    # matrix scaling
    scale_var='matrix'
    pbase = pg0
    qbase = .1
    Egbase = Sbase
    scale_var_params={'pgbase':pbase,'pbase':pbase,'qbase':qbase,'Vbase':Vbase,'deltabase':1,'Sbase':Sbase,'Ebase':Egbase}
    compare_DD(tol_lf=tol_lf,tol_dd=tol_dd,art_zero=art_zero,max_iter_dd=max_iter_dd,max_iter_lf=max_iter_lf,max_data_points=max_data_points,scale_var=scale_var,scale_var_params=scale_var_params,save_data=False,path_to_data=path_to_data)

    # run_LF_mes(max_iter=max_iter_lf,tol=tol_lf,scale_var=scale_var,scale_var_params=scale_var_params,plot_top=False,plot_jac=True,plot_sol=False,reorder=True)

    # NOT implemented correctly!
    # het_net, gas_net, elec_net,x_sol,iters,err_vec = block_jacobi(max_iter=max_iter_lf,tol=tol_lf,scale_var=scale_var,scale_var_params=scale_var_params)

    # compare_jacobians(formulation={'gas':'full','elec':'complex_power'})

    # table_LF_results(path_to_data,max_iter=max_iter_lf,tol=tol_lf,scale_var=scale_var,scale_var_params=scale_var_params)

    plt.show()
