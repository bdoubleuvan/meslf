"""Make network, with one solution scenario, for the single-carrier heat network of the Benchmark Problem (BP).

The solution is determined using unknown half link flow formulation and known outflow temperature for loads. Tolerance of NR is 1e-6, and no scaling is used."""
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
from meslf.utils.constants import MW
import numpy as np
from meslf.networks.read_write_network import to_pd_dataframes
import os

# water carrier
rho_w = 960. #[kg/m^3]
Cp_w = 4.182e3 #[J/(kg K)]
grav_const = 9.81 #[m/s^2]
water = Water('water',Cp_w,rho=rho_w)

# physical parameters of network and pipes
Ta = 0.
L = 500. #[m]
D = .15 #[m]
lam = .2 #[W/(mK)]
U = lam/(np.pi*D) #[W/(m^2 K)]
heat_link_type = 'standard_pipe_low_pres_pole'
heat_link_params = {'L':L,'U':U,'D':D,'carrier':water}

# coordinates
x0 = 8
y0 = 8.5

x1 = 12
y1 = 11

x2 = 12
y2 = 5

# boundary conditions
p0 = 100*rho_w*grav_const #[Pa]
Ts0 = 100 #[C]
phi1 = 1*MW #[W]
To1 = 50. #[C]
phi2 = 1.5*MW #[W]
To2 = 50. #[C]

# solver information
formulation='half_link_flow'
tol=1e-6
max_iter=50

def create_network():
    """Create a heat network consisting of 3 demand/source nodes. """
    h0 = HeatNode('hn0',node_type=0,x=x0,y=y0,p=p0,Ts=Ts0) # slack
    h0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h1 = HeatNode('hn1',node_type=1,x=x1,y=y1,Tr_hl=To1,dphi=phi1) # load node (sink)
    h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
    h2 = HeatNode('hn2',node_type=1,x=x2,y=y2,Tr_hl=To2,dphi=phi2) # load  node (sink)
    h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    
    hl0 = HeatLink('hl0',h0,h1,link_type=heat_link_type,link_params=heat_link_params)
    hl1 = HeatLink('hl1',h0,h2,link_type=heat_link_type,link_params=heat_link_params.copy())
    hl2 = HeatLink('hl2',h1,h2,link_type=heat_link_type,link_params=heat_link_params.copy())
    
    heat_net = HeatNetwork('3 nodes',Ta=Ta)
    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    return heat_net

def initialize_network(heat_net):
    """Initialize the network"""
    m_init = np.array([6,6,1]) #[kg/s]
    m_hl_init = np.array([5,7]) #[kg/s]
    p_init = np.array([99.5,99.4])*rho_w*grav_const #[Pa]
    Ts_init = np.array([99.6,99.3]) #[C]
    Tr_init = np.array([49.7,49.8,50.]) #[C]
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    heat_net.initialize()
    heat_net.update(x_init,formulation=formulation)
    x0 = heat_net.set_x_init(formulation=formulation)
    return x0
    
def run_load_flow():
    """Run load flow for this example"""
    # create network
    heat_net = create_network()
    
    # initialize
    x0 = initialize_network(heat_net)
    
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,solver='NR',formulation=formulation)
    
    return heat_net,x_sol

def save_scenario():
    """Save the network scenario data"""
    # create network and run load flow
    heat_net,x_sol = run_load_flow()
    heat_net.update_full(x_sol,formulation=formulation)
    
    # make pd df
    nodes, links, halflinks = to_pd_dataframes(heat_net)
    
    # save data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    nodes.to_pickle(os.path.join(dir_path,'HN_BP_nodes.pkl'))
    links.to_pickle(os.path.join(dir_path,'HN_BP_links.pkl'))
    halflinks.to_pickle(os.path.join(dir_path,'HN_BP_halflinks.pkl'))

def table():
    """Write the full solution to a txt file, which can be written by latex to create a table."""
    # create network and run load flow
    heat_net,x_sol = run_load_flow()
    heat_net.update_full(x_sol,formulation=formulation)
        
    path_to_data = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path_to_data,'heat_data_BP.txt'), "w") as table:
        table.write(r'\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|} ')
        table.write(r'\hline ')
        table.write(r'\textbf{Node} & $h$~\sib{\meter}  & $m^\text{inj}$~\sib{\kilo\gram \per \second} & $T^s$~\sib{\celsius} & $T^r$~\sib{\celsius} & $\Delta\varphi^\text{inj}$~\sib{\mega \watt} & $T^s_{hl}$~\sib{\celsius} & $T^s_{hl}$~\sib{\celsius} & \textbf{Link} & $m$~\sib{\kilo\gram \per \second}\\ ')
        table.write(r'\hline ')
        table.write(r'0 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & 0-1 & {:.3f} \\'.format(heat_net.nodes[0].get_p()/(rho_w*grav_const),heat_net.nodes[0].half_links[0].get_m(),heat_net.nodes[0].get_Ts(),heat_net.nodes[0].get_Tr(),heat_net.nodes[0].half_links[0].get_dphi()/MW,heat_net.nodes[0].half_links[0].get_Ts(),heat_net.nodes[0].half_links[0].get_Tr(),heat_net.links[0].get_m()))
        table.write(r'1 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & 0-2 & {:.3f} \\'.format(heat_net.nodes[1].get_p()/(rho_w*grav_const),heat_net.nodes[1].half_links[0].get_m(),heat_net.nodes[1].get_Ts(),heat_net.nodes[1].get_Tr(),heat_net.nodes[1].half_links[0].get_dphi()/MW,heat_net.nodes[0].half_links[0].get_Ts(),heat_net.nodes[0].half_links[0].get_Tr(),heat_net.links[1].get_m()))
        table.write(r'2 & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & 1-2 & {:.3f} \\'.format(heat_net.nodes[2].get_p()/(rho_w*grav_const),heat_net.nodes[2].half_links[0].get_m(),heat_net.nodes[2].get_Ts(),heat_net.nodes[2].get_Tr(),heat_net.nodes[2].half_links[0].get_dphi()/MW,heat_net.nodes[0].half_links[0].get_Ts(),heat_net.nodes[0].half_links[0].get_Tr(),heat_net.links[2].get_m()))
        table.write(r'\hline ')
        table.write(r'\end{tabular}%')
        
if __name__== '__main__':
    save_scenario()
    table()
