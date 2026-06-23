"""Make network, with one solution scenario, for the single-carrier gas network of the Benchmark Problem (BP).

The solution is determined using full formulation and link pressure drops as function of flow (i.e. fb). Tolerance of NR is 1e-6, and no scaling is used."""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.carrier import Gas
from meslf.utils.constants import bar, mbar, hour, mm
import numpy as np
from meslf.networks.read_write_network import to_pd_dataframes
import os
from meslf.utils.post_processing import exp_tex

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
L = 500. #[m]
D = .15 #[m]
gas_link_type = 'pipe_low_pres_pole'
gas_link_params = {'L':L,'D':D,'carrier':gas}
link_eq = 'dp_of_q'

# coordinates
x0 = 0
y0 = 11

x1 = 0
y1 = 5

x2 = 4
y2 = 8.5

# boundary conditions
p0 = 50*mbar #[Pa]
q1 = .09 #[kg/s]
q2 = .15 #[kg/s]

# solver information
formulation='full'
tol=1e-6
max_iter=50

def create_network():
    """Create a gas network consisting of 3 demand/source nodes. """

    g0 = GasNode('gn0',node_type=0,x=x0,y=y0,p=p0) # reference node
    g1 = GasNode('gn1',node_type=1,x=x1,y=y1,q=q1) # load node
    g2 = GasNode('gn2',node_type=1,x=x2,y=y2,q=q2) # load node

    gl0 = GasLink('gl0',g0,g1,link_type = gas_link_type,link_params = gas_link_params,link_eq_form=link_eq)
    gl1 = GasLink('gl1',g0,g2,link_type = gas_link_type,link_params = gas_link_params,link_eq_form=link_eq)
    gl2 = GasLink('gl2',g1,g2,link_type = gas_link_type,link_params = gas_link_params,link_eq_form=link_eq)

    gas_net = GasNetwork('3 nodes')
    gas_net.add_link(gl0)
    gas_net.add_link(gl1)
    gas_net.add_link(gl2)

    return gas_net

def initialize_network(gas_net):
    """Initialize the gas network, consisting of 3 demand/source nodes.

    Parameters
    ----------
    gas_net : GasNetwork
        The gas network to be initialized

    Returns
    -------
    x0 : np array
        initial guess
    """
    q_init = .05*np.ones(len(gas_net.links))
    p_init = np.array([29,28])*mbar

    x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    gas_net.update(x_init,formulation=formulation)
    x0 = gas_net.set_x_init(formulation=formulation)
    return x0

def run_load_flow():
    """Stead-state load flow analysis of gas network
    """
    # create network
    gas_net = create_network()
    # initialize
    x0 = initialize_network(gas_net)

    # solve network
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)

    print('Solution (final error = {:.4e}):'.format(err_vec[-1]))
    print('p = {} mbar'.format(p_sol/mbar))
    print('q = {} kg/s'.format(q_sol))
    print('q nodal inj = {} kg/s'.format(q_inj))
    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))

    return gas_net,x_sol

def save_scenario():
    """Save the network scenario data"""
    # create network and run load flow
    gas_net,x_sol = run_load_flow()
    gas_net.update_full(x_sol,formulation=formulation)

    # make pd df
    for node in gas_net.get_nodes():
        for hl in node.get_half_links():
            gas_net.add_half_link(hl)
    nodes, links, halflinks = to_pd_dataframes(gas_net)

    # save data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    nodes.to_pickle(os.path.join(dir_path,'GN_BP_nodes.pkl'))
    links.to_pickle(os.path.join(dir_path,'GN_BP_links.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'GN_BP_halflinks.pkl'))

def table():
    """Write the full solution to a txt file, which can be written by latex to create a table."""
    # create network and run load flow
    gas_net,x_sol = run_load_flow()
    gas_net.update_full(x_sol,formulation=formulation)

    path_to_data = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path_to_data,'gas_data_BP.txt'), "w") as table, open(os.path.join(path_to_data,'gas_data_BP_noheader.txt'), "w") as table_noheader:
        table.write(r'\begin{tabular}{|c|c|c|c|c|} ')
        table.write(r'\hline ')
        table.write(r'\textbf{Node} & $p$~\sib{\milli\bar}  & $q^\text{inj}$~\sib{\kilo\gram \per \second} & \textbf{Link} & $q$~\sib{\kilo\gram \per \second}\\ ')
        table.write(r'\hline ')
        data = r'0 & {:.3f} & {:.3f} & 0-1  & {:.3f} \\'.format(gas_net.nodes[0].get_p()/mbar,gas_net.nodes[0].half_links[0].get_q(),gas_net.links[0].get_q())
        data += r'1 & {:.3f} & {:.3f} & 0-2  & {:.3f} \\'.format(gas_net.nodes[1].get_p()/mbar,gas_net.nodes[1].half_links[0].get_q(),gas_net.links[1].get_q())
        data += r'2 & {:.3f} & {:.3f} & 1-2  & {:.3f} \\'.format(gas_net.nodes[2].get_p()/mbar,gas_net.nodes[2].half_links[0].get_q(),gas_net.links[2].get_q())
        table.write(data)
        table_noheader.write(data)
        table.write(r'\hline ')
        table.write(r'\end{tabular}%')


if __name__== '__main__':
    # save_scenario()
    table()
