"""Electrical network consisting of 3 demand/source nodes. Based on the example in Shabanpourt-Haghighi and Seifi"""
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.utils.constants import MW, kV
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

# base values used for p.u.
Sbase_shabanpour = 1*MW #[W]
Vbase_shabanpour = 10/np.sqrt(3)*kV #[V]
Ybase_shabanpour = Sbase_shabanpour/(Vbase_shabanpour**2)

# BC / solution
V0 = 1.06 #[p.u.]
delta0 = 0
delta2_sol = -0.10555751316061704
P0c = 50.4982 #[p.u.] (= MW)
Q0c = 27.3515 #[p.u.] (= MW)
P0_load = 0.1451 #[p.u.] (=MW)
Q0_load = 0
P1_load = 30 #[p.u.] (= MW)
Q1_load = 15 #[p.u.] (= MW)
P2_load = 30.136 #[p.u.] (= MW)
Q2_load = 15 #[p.u.] (= MW)
P2c = 10.5331 #[p.u.] (= MW)
Q2c = 10.1507 #[p.u.] (= MW)
V2_sol = 1. #[p.u.]

def create_network(c_hl=True,node_set=1,values='p.u.'):
    """Create an electrical network consisting of 3 demand/source nodes. The values are given in p.u., assuming a base value for S of 1MW.

    Parameters
    ----------
    c_hl : bool, optional
        If true, half links are added to the nodes with values equal to the flow going to / coming from the coupling components. Default is True.

    Returns
    -------
    elec_net : ElectricalNetwork
        The test network
    """
    V0 = 1.06 #[p.u.]
    delta0 = 0
    P0c = 50.4982 #[p.u.] (= MW)
    Q0c = 27.3515 #[p.u.] (= MW)
    P0_load = 0.1451 #[p.u.] (=MW)
    P1_load = 30 #[p.u.] (= MW)
    Q1_load = 15 #[p.u.] (= MW)
    P2_load = 30.136 #[p.u.] (= MW)
    Q2_load = 15 #[p.u.] (= MW)
    P2c = 10.5331 #[p.u.] (= MW)
    Q2c = 10.1507 #[p.u.] (= MW)
    V2_sol = 1. #[p.u.]
    if values == 'p.u.':
        pass
    elif values == 'S.I.':
        V0 *= Vbase_shabanpour
        P0c *= Sbase_shabanpour
        Q0c *= Sbase_shabanpour
        P0_load *= Sbase_shabanpour
        P1_load *= Sbase_shabanpour
        Q1_load *= Sbase_shabanpour
        P2_load *= Sbase_shabanpour
        Q2_load *= Sbase_shabanpour
        P2c *= Sbase_shabanpour
        Q2c *= Sbase_shabanpour
        V2_sol *= Vbase_shabanpour
    else:
        raise ValueError('Enter valid values. Either "p.u." or "S.I."')
    if node_set == 3:
        if c_hl:
            e0 = ElectricalNode('en0',node_type=1,V=V0,P=-P0c) # gen
            ElectricalHalfLink('en0_hl0',start_node=e0,P=P0_load,Q=Q0_load) # power to compressor?
        else:
            e0 = ElectricalNode('en0',node_type=1,V=V0,P=P0_load-P0c) # gen
    else:
        e0 = ElectricalNode('en0',node_type=0,V=V0,delta=delta0) # ref
        if c_hl:
            ElectricalHalfLink('en0_hl1',start_node=e0,P=-P0c,Q=-Q0c) # power from coupling (slack)
            ElectricalHalfLink('en0_hl0',start_node=e0,P=P0_load,Q=Q0_load) # power to compressor?
    e1 = ElectricalNode('en1',node_type=2,P=P1_load,Q=Q1_load) # load
    if node_set == 1:
        if c_hl:
            e2 = ElectricalNode('en2',node_type=2,P=P2_load,Q=Q2_load) # load
            ElectricalHalfLink('en2_hl0',start_node=e2,P=-P2c,Q=-Q2c) # power from coupling
        else:
            e2 = ElectricalNode('en2',node_type=2,P=P2_load-P2c,Q=Q2_load-Q2c) # load
    elif node_set == 2:
        if c_hl:
            e2 = ElectricalNode('en2',node_type=1,V=V2_sol,P=-P2c) # gen (power form coupling on 0th halflink)
            ElectricalHalfLink('en2_hl0',start_node=e2,P=P2_load,Q=Q2_load) # load
        else:
            e2 = ElectricalNode('en2',node_type=1,V=V2_sol,P=P2_load-P2c) # gen
    elif node_set == 3:
        e2 = ElectricalNode('en2',node_type=0,V=V2_sol,delta=delta2_sol) # ref
    else:
        raise ValueError('Enter valid node set')

    Pbs_MW = 100 # factor 100 because Shabanpour give a p.u. value, using Sbase = 100 MW, and I use a base value of 1MW
    x = 0.5/Pbs_MW
    r = 0.05/Pbs_MW
    z = r+x*1j
    b = -x/(np.abs(z)**2)
    g = r/(np.abs(z)**2)
    if values == 'S.I.':
        b *= Ybase_shabanpour
        g *= Ybase_shabanpour
    elec_link_params = {'b':b,'g':g}
    el0 = ElectricalLink('el0',e0,e1,link_type='short_line',link_params=elec_link_params)
    el1 = ElectricalLink('el1',e0,e2,link_type='short_line',link_params=elec_link_params)
    el2 = ElectricalLink('el2',e1,e2,link_type='short_line',link_params=elec_link_params)

    elec_net = ElectricalNetwork('3 nodes')
    elec_net.add_link(el0)
    elec_net.add_link(el1)
    elec_net.add_link(el2)

    # set coordinates
    e0.x=7
    e0.y=12
    e1.x=2
    e1.y=7
    e2.x=7
    e2.y=2
    return elec_net

def initialize_network(elec_net,node_set=1, values='p.u.'):
    x_entries, unknown_delta_nodes, unknown_V_nodes = elec_net.get_x_entries()
    delta_init  = np.zeros(len(unknown_delta_nodes))
    if values == 'p.u.':
        V_init = np.ones(len(unknown_V_nodes))
    elif values == 'S.I.':
        V_init = Vbase_shabanpour*np.ones(len(unknown_V_nodes))
    x_init = np.concatenate((delta_init,V_init))
    elec_net.initialize()
    elec_net.update(x_init)
    x0 = elec_net.set_x_init()
    return x0

def run_load_flow(c_hl=True,tol=1e-6,max_iter=50,node_set=1, values='p.u.'):
    """Stead-state load flow analysis of electrical network. Per unit system is already assumed, and the default values are used for initialization.
    """
    # create network
    elec_net = create_network(c_hl=c_hl,node_set=node_set, values=values)
    # initialize
    x0 = initialize_network(elec_net,node_set=node_set, values=values)

    # solve network
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR')

    return elec_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge,tol

def comp_conv(node_set=1, values='p.u.'):
    """Compare convergence"""
    # make figure to plot convergence
    fig_conv_elec = plt.figure('Convergence plot electrical network, values in {} (node set {})'.format(node_set,values))
    ax_conv_elec = fig_conv_elec.gca()
    max_iters_used = 0
    for c_hl in [True,False]:
        print('seperate couplings used: {}'.format(c_hl))
        elec_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge,tol = run_load_flow(c_hl=c_hl,node_set=node_set,values=values)
        print('Solution:')
        print('delta = {}'.format(delta_sol))
        print('|V| = {} p.u.'.format(V_sol))
        print('P edge = {} p.u.'.format(P_edge))
        print('Q edge = {} p.u.'.format(Q_edge))
        print('S nodal inj = {} p.u.'.format(S_inj))
        print('P hl = {} p.u.'.format([hl.P for node in elec_net.get_nodes() for hl in node.get_half_links()]))
        print('Q hl = {} p.u.'.format([hl.Q for node in elec_net.get_nodes() for hl in node.get_half_links()]))
        # plot convergence
        if c_hl:
            ls = '--'
            label = 'sep. coup.'
        else:
            ls = '-'
            label = 'int. coup.'
        ax_conv_elec.semilogy(err_vec,ls=ls,color='tab:red',marker='.',label=label)
        max_iters_used = max(max_iters_used,iters)
    ax_conv_elec.set_xlabel(r'Iteration $k$')
    ax_conv_elec.set_ylabel(r'Error $||D_F F(x^k)||_2$')
    ax_conv_elec.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
    ax_conv_elec.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_elec.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_elec.legend()
    xmin = 0
    xmax = max_iters_used
    xticks = range(xmin,xmax+1,2) # make sure the xticks are integers
    ax_conv_elec.set_xlim(left=xmin,right=xmax+1)
    ax_conv_elec.set_xticks(xticks)

def example_en_shabanpour_sep_coup():
    """Check the solution of the network. Flow coming from the couplings are modeled seperately."""
    # Given / When
    elec_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge,tol = run_load_flow(c_hl=True)

    # Then
    delta_sol_expected = np.array([-0.12197757076337971, -0.10555751316061704])
    V_sol_expected = np.array([0.9801,1.])
    x_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))

    rel_tol = 1e-4
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def example_en_shabanpour():
    """Check the solution of the network. Flow coming from the couplings are modeled as part of the load nodes."""
    # Given / When
    elec_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge,tol = run_load_flow(c_hl=False)

    # Then
    delta_sol_expected = np.array([-0.12197757076337971, -0.10555751316061704])
    V_sol_expected = np.array([0.9801,1.])
    x_sol_expected = np.concatenate((delta_sol_expected,V_sol_expected))

    rel_tol = 1e-4
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

if __name__ == '__main__':
    for node_set in [1,2]:
        for value in ['p.u.','S.I.']:
            comp_conv(node_set=node_set,values=value)

    plt.show()
