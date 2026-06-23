"""Make network, with one solution scenario, for the single-carrier electricity network of the Benchmark Problem (BP).

The solution is determined using complex power formulation. Tolerance of NR is 1e-6, and no scaling is used (but values are already given in p.u.)."""
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.utils.constants import kV, MW
import numpy as np
from meslf.networks.read_write_network import to_pd_dataframes
import os

# coordinates
x0 = 3.5
y0 = 0

x1 = 6
y1 = 4

x2 = 8.5
y2 = 0

# Given
Sbase = 1*MW #[W]
Vbase = 10/np.sqrt(3)*kV #[V]
print('Vbase = {}'.format(Vbase))
Ybase = Sbase/(Vbase**2)

# physical parameters of the lines
b_pu = -10. #[p.u.]
g_pu = 1. #[p.u.]
b = b_pu*Ybase
g = g_pu*Ybase
elec_link_params = {'b':b,'g':g}
elec_link_type = 'short_line'

# boundary conditions
V0 = 1.*Vbase #[V]
delta0 = -0.10478613961080965 #[rad]
P1 = -1.0545*Sbase #[W]
V1 = .98*Vbase #[V]
P2 = 2.*Sbase #[W]
Q2 = 2*Sbase #[W]

# solver information
tol=1e-6
max_iter=50

def create_network():
    """Create an electrical network consisting of 3 demand/source nodes. The values are given in p.u., assuming a base value for S of 1MW."""
    e0 = ElectricalNode('en0',node_type=0,x=x0,y=y0,V=V0,delta=delta0) # ref
    e1 = ElectricalNode('en1',node_type=1,x=x1,y=y1,P=P1,V=V1) # generator 
    e2 = ElectricalNode('en2',x=x2,y=y2,node_type=2,P=P2,Q=Q2) # load
        
    
    el0 = ElectricalLink('el0',e0,e1,link_type=elec_link_type,link_params=elec_link_params)
    el1 = ElectricalLink('el1',e0,e2,link_type=elec_link_type,link_params=elec_link_params)
    el2 = ElectricalLink('el2',e1,e2,link_type=elec_link_type,link_params=elec_link_params)
    
    elec_net = ElectricalNetwork('3 nodes')
    elec_net.add_link(el0)
    elec_net.add_link(el1)
    elec_net.add_link(el2)
    return elec_net

def initialize_network(elec_net):
    """Initialize the electrical network, consisting of 3 demand/source nodes.
    
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
    
    delta_init = np.zeros(len(unknown_delta_nodes))
    V_init = Vbase*np.ones(len(unknown_V_nodes))
    x_init = np.concatenate((delta_init,V_init))
    
    elec_net.initialize()
    elec_net.update(x_init)
    x0 = elec_net.set_x_init()
    return x0

def run_load_flow():
    """Stead-state load flow analysis of electrical network. Per unit system is already assumed, and the default values are used for initialization.
    """
    # create network
    elec_net = create_network()
    
    x0 = initialize_network(elec_net)
    
    # solve network
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
    
    return elec_net,x_sol

def save_scenario():
    """Save the network scenario data"""
    # create network and run load flow
    elec_net,x_sol = run_load_flow()
    elec_net.update_full(x_sol)
    
    # make pd df
    for node in elec_net.get_nodes():
        for hl in node.get_half_links():
            elec_net.add_half_link(hl)
    nodes, links, halflinks = to_pd_dataframes(elec_net)
    
    # save data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    nodes.to_pickle(os.path.join(dir_path,'EN_BP_nodes.pkl'))
    links.to_pickle(os.path.join(dir_path,'EN_BP_links.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'EN_BP_halflinks.pkl'))

def table():
    """Write the full solution to a txt file, which can be written by latex to create a table."""
    # create network and run load flow
    elec_net,x_sol = run_load_flow()
        
    path_to_data = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path_to_data,'elec_data_BP.txt'), "w") as table:
        table.write(r'\begin{tabular}{|c|c|c|c|c|c|c|} ')
        table.write(r'\hline ')
        table.write(r'\textbf{Node} & $|V|$~\sib{\kilo\volt}  & $\delta$~\sib{\radians} & $S^\text{inj}$~\sib{\mega \watt} & \textbf{Link} & $S_{ij}$~\sib{\mega\watt} & $S_{ji}$~\sib{\mega\watt} \\ ')
        table.write(r'\hline ')
        table.write(r'0 & {:.3f} & {:.3f} & {:.3f} {:+.3f}$\iu$ & 0-1  & {:.3f} {:+.3f}$\iu$ & {:.3f} {:+.3f}$\iu$ \\'.format(elec_net.nodes[0].get_V()/kV,elec_net.nodes[0].get_delta(),elec_net.nodes[0].half_links[0].get_P()/MW,elec_net.nodes[0].half_links[0].get_Q()/MW,elec_net.links[0].get_Pstart()/MW,elec_net.links[0].get_Qstart()/MW,elec_net.links[0].get_Pend()/MW,elec_net.links[0].get_Qend()/MW))
        table.write(r'1 & {:.3f} & {:.3f} & {:.3f} {:+.3f}$\iu$ & 0-2  & {:.3f} {:+.3f}$\iu$ & {:.3f} {:+.3f}$\iu$ \\'.format(elec_net.nodes[1].get_V()/kV,elec_net.nodes[1].get_delta(),elec_net.nodes[1].half_links[0].get_P()/MW,elec_net.nodes[1].half_links[0].get_Q()/MW,elec_net.links[1].get_Pstart()/MW,elec_net.links[1].get_Qstart()/MW,elec_net.links[1].get_Pend()/MW,elec_net.links[1].get_Qend()/MW))
        table.write(r'2 & {:.3f} & {:.3f} & {:.3f} {:+.3f}$\iu$ & 1-2  & {:.3f} {:+.3f}$\iu$ & {:.3f} {:+.3f}$\iu$ \\'.format(elec_net.nodes[2].get_V()/kV,elec_net.nodes[2].get_delta(),elec_net.nodes[2].half_links[0].get_P()/MW,elec_net.nodes[2].half_links[0].get_Q()/MW,elec_net.links[2].get_Pstart()/MW,elec_net.links[2].get_Qstart()/MW,elec_net.links[2].get_Pend()/MW,elec_net.links[2].get_Qend()/MW))
        table.write(r'\hline ')
        table.write(r'\end{tabular}%')
        
if __name__== '__main__':
    save_scenario()
    table()
