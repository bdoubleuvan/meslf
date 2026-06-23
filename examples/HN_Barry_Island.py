import warnings
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
from meslf.utils.constants import mm, MW, bar, mbar
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sps

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning 

# water carrier
rho_w = 960. #[kg/m^3]
mu_w = 0.294e-6 #[m^2/s]
Cp_w = 4.182e3 #[J/(kg K)]
grav_const = 9.81 #[m/s^2]
water = Water('water',Cp_w,rho=rho_w,mu=mu_w)

# physical parameters of network and pipes
Ta = 0. #??? Can't find it in the paper
length = np.array([257.6, 97.5, 51, 59.5, 271.3, 235.4, 177.3, 102.8, 247.7, 160.8, 129.1, 186.1, 136.2, 41.8, 116.8, 136.4, 136.4, 44.9, 136.4, 134.1, 41.7, 161.1, 134.2, 52.1, 136, 123.3, 61.8, 95.2, 105.1, 70.6, 261.8, 201.3]) # m
diameter = np.array([125, 40, 40, 100, 32, 65, 40, 40, 40, 100, 40, 100, 80, 50, 32, 32, 32, 80, 32, 32, 65, 32, 32, 65, 32, 32, 40, 32, 32, 125, 125, 125])*mm #m
eps = .4*mm #[m]
lam = np.array([.321, .21, .21, .327, .189, .236, .21, .21, .21, .327, .21, .327, .278, .219, .189, .189, .189, .278, .189, .189, .236, .189, .189, .236, .189, .189, .21, .189, .189, .321, .321, .321]) #[W/(mK)]
U = lam/(np.pi*diameter) #[W/(m^2 K)]

# topology data
node_names = list(range(1,33))
start_nodes = [1,2,2,2,5,5,7,7,7,5,11,11,13,14,15,15,14,14,19,19,19,22,22,22,25,25,25,28,28,31,31,32]#names of start nodes
end_nodes = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,28,7,11]#names of start nodes
link_names = list(range(1,33))
x = [6,5,5,6,4,5,3,4,2,2,3,4,2,0,0,-1,1,-1,0,-1,1,0,-1,1,0,-1,1,0,-1,1,0,3]
y = [0,2,-1,3,2,3,-1,-2,1,-2,3,4,4,4,5,5,5,4,3,3,3,2,2,2,1,1,1,0,0,0,-1,2]
heat_link_type= 'standard_pipe_low_pres_pole'

slack_nodes = [31] # name of the slack node
source_nodes = [1,32] # names of source nodes
junction_nodes = [2,5,13,15,19,22,25,28] # names of source nodes
sink_nodes = [name for name in node_names if not (name in source_nodes or name in junction_nodes)] # names of sink nodes

# boundary conditions
p_ref = 100*rho_w*grav_const #[Pa] ???? Can't find it in the paper
Ts_source = 70 #[C]
To_sink = 30 #[C]
phi_load = np.array([.107,.145,.107,.107,.107,.107,.107,.145,.107,.0805,.0805,.0805,.0805,.0805,.0805,.107,.107,.107,.107,.107,.107])*MW
phi_source = np.array([.8099,.3797])*MW

# initial guess
m_ic = 5 #[kg/s]
m_hl_ic = 1 #[kg/s]

def create_network(heat_link_type='standard_pipe_low_pres_pole'):
    """Create the heat network"""
    heat_net = HeatNetwork('Barry Island', Ta=Ta)
    
    # nodes
    for ind_n,node in enumerate(node_names):
        if node in slack_nodes:
            hn = HeatNode(str(node),node_type=0,p=p_ref,Ts=Ts_source) # ref. slack node
            hn.half_links[0].set_type('heat_exchanger',{'carrier':water})
        elif node in source_nodes:
            hn = HeatNode(str(node),node_type=1,Ts_hl=Ts_source,dphi=-phi_source[source_nodes.index(node)]) # load  node (source)
            hn.half_links[0].set_type('heat_exchanger',{'carrier':water})
        elif node in junction_nodes:
            hn = HeatNode(str(node),node_type=2) # junction node
        elif node in sink_nodes:
            hn = HeatNode(str(node),node_type=1,Tr_hl=To_sink,dphi=phi_load[sink_nodes.index(node)]) # load  node (sink)
            hn.half_links[0].set_type('heat_exchanger',{'carrier':water})
        hn.x = x[ind_n]
        hn.y = -y[ind_n]
        heat_net.add_node(hn)
    
    # links
    for ind_l,link in enumerate(link_names):
        start_node = heat_net.nodes[node_names.index(start_nodes[ind_l])]
        end_node = heat_net.nodes[node_names.index(end_nodes[ind_l])]
        heat_link_params = {'D':diameter[ind_l],'L':length[ind_l],'eps':eps,'U':U[ind_l],'carrier':water,'Ta':Ta}
        hl = HeatLink(str(link),start_node,end_node,link_type = heat_link_type,link_params = heat_link_params)
        heat_net.add_link(hl)
    
    return heat_net

def initialize_network(heat_net,formulation='half_link_flow'):
    """Initialize the heat network.
    
    Parameters
    ----------
    heat_net : HeatNetwork
        The heat network to be initialized
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    x_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks = heat_net.get_x_entries(formulation=formulation)

    m_init = m_ic*np.ones(len(unknown_m_links)) #[kg/s]
    m_hl_init = np.zeros(len(unknown_m_halflinks)) #[kg/s]
    for ind_hl,hl in enumerate(unknown_m_halflinks):
        if hl.link_type == 'heat_exchanger_sink':
            m_hl_init[ind_hl] = m_hl_ic
        else:
            m_hl_init[ind_hl] = -m_hl_ic
    p_init = p_ref*np.ones(len(unknown_p_nodes)) #[Pa]
    Ts_init = Ts_source*np.ones(len(unknown_Ts_nodes)) #[C]
    Tr_init = To_sink*np.ones(len(unknown_Tr_nodes)) #[C]
    Tshl_init = Ts_source*np.ones(len(unknown_Ts_halflinks)) #[C]
    Trhl_init = To_sink*np.ones(len(unknown_Tr_halflinks)) #[C]
    x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init,Tshl_init,Trhl_init))
    
    heat_net.initialize()
    heat_net.update(x_init,formulation=formulation)
    x0 = heat_net.set_x_init(formulation=formulation)
    return x0

def run_load_flow(heat_link_type='standard_pipe_low_pres_pole',formulation='half_link_flow',max_iter=100,tol=1e-6,scale_var='matrix',scale_var_params={'mbase':1,'phbase':100*rho_w*grav_const,'Tbase':50,'phibase':1*MW},plot_top=False,plot_conv=False):
    """Steady-state load flow analysis of heat network, without scaling. The default values are used for initialization.
    """
    # create network
    heat_net = create_network(heat_link_type=heat_link_type)
    
    # initialize
    x0 = initialize_network(heat_net,formulation=formulation)
    
    # solve network
    print('\nRunning load flow for single-carrier heat network, with formulation {} and scaling {}'.format(formulation,scale_var))
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    
    print('Solution after {} iterations (final error = {:.4e}):'.format(iters,err_vec[-1]))
    print('p heat = {} m'.format(p_vec/(rho_w*grav_const)))
    print('m = {} kg/s'.format(m_vec))
    print('Ts = {} C'.format(Ts_vec))
    print('Tr = {} C'.format(Tr_vec))
    print('m hl = {} kg/s'.format(m_hl_vec))
    print('Ts hl = {}'.format(Ts_hl_vec))
    print('Tr hl = {}'.format(Tr_hl_vec))
    print('phi hl = {} MW'.format([hl.dphi/MW for hl in heat_net.get_half_links()]))
    
    if plot_top:
        fig_top = plt.figure('Network topology')
        ax_top = fig_top.gca()
        heat_net.draw_network(ax_top)
        plt.axis('equal')
        plt.axis('off')
        
    if plot_conv:
        # plot convergence
        fig_conv = plt.figure('Convergence of NR, formulation {}'.format(formulation))
        ax_conv = fig_conv.gca()
        plt.xlabel('Iteration k')
        plt.ylabel(r'Error $||F(x^k)||_2$')
        max_iter_used = iters
        ax_conv.semilogy(np.asarray(range(0,iters+1)),err_vec,marker='.',color='tab:blue',label=formulation)
        layout_convergence(ax_conv,tol,max_iter_used)
        
        # plot convergence order
        fig_ord = plt.figure('Convergence order of NR, formulation {}'.format(formulation))
        ax_ord = fig_ord.gca()
        plt.xlabel(r'$||F(x^k)||_2 / ||F(x^0)||_2$')
        plt.ylabel(r'$||F(x^{k+1})||_2 / ||F(x^0)||_2$')
        ax_ord.loglog(err_vec[:-1]/err_vec[0],err_vec[1:]/err_vec[0],marker='.',color='tab:blue',label=formulation)
        layout_convergence_order(ax_ord)
        
    return heat_net,x_sol,iters,err_vec

def comp_link_types(tol=1e-6,formulation='half_link_flow'):
    fig_conv = plt.figure('Convergence of NR, formulation {}, various pipes'.format(formulation))
    ax_conv = fig_conv.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F(x^k)||_2$')
    max_iter_used = 0
    
    fig_ord = plt.figure('Convergence order of NR, formulation {}, various pipes'.format(formulation))
    ax_ord = fig_ord.gca()
    plt.xlabel(r'$||F(x^k)||_2 / ||F(x^0)||_2$')
    plt.ylabel(r'$||F(x^{k+1})||_2 / ||F(x^0)||_2$')
    
    for link_type in ['standard_pipe_low_pres_blasius','standard_pipe_low_pres_hagen_poiseuille','standard_pipe_low_pres_pole','standard_pipe_low_pres_colebrook','standard_pipe_low_pres_churchill']:
        heat_net,x_sol,iters,err_vec = run_load_flow(tol=tol,formulation=formulation,heat_link_type=link_type)
        print('Pipe used: {}'.format(link_type))
        ax_conv.semilogy(np.asarray(range(0,iters+1)),err_vec,marker='.',label=link_type)
        max_iter_used = max(max_iter_used,iters)
        ax_ord.loglog(err_vec[:-1]/err_vec[0],err_vec[1:]/err_vec[0],marker='.',label=link_type)
        
    layout_convergence(ax_conv,tol,max_iter_used)
    layout_convergence_order(ax_ord)
        
def layout_convergence(ax,tol,max_iters):
    ax.semilogy([0,max_iters+1],[tol,tol],'k:',label='tolerance')
    xmin = 0
    xmax = max_iters
    xticks = range(xmin,xmax+1,max(1,int(xmax/10))) # make sure the xticks are integers
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

def layout_convergence_order(ax):
    x_min,x_max = ax.get_xlim()
    x_slope = np.linspace(x_min,x_max)
    y_slope2 = x_slope**2
    ax.loglog(x_slope,x_slope,linestyle='--',color='k',label='slope 1')
    ax.loglog(x_slope,y_slope2,linestyle='-.',color='k',label='slope 2')
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    
if __name__ == '__main__':
    tol = 1e-6
    for form in ['standard','half_link_flow']:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
            heat_net,x_sol,iters,err_vec = run_load_flow(tol=tol,scale_var=None,plot_top=True,plot_conv=True,formulation=form) 
    
    # plot temperature solution to compare with paper
    node_names_temp_plot = [1,2,5,11,13,14,19,22,25,28,31,7,5]
    Ts = [heat_net.nodes[ind_n-1].get_Ts() for ind_n in node_names_temp_plot]
    Tr = [heat_net.nodes[ind_n-1].get_Tr() for ind_n in node_names_temp_plot]
    xtick_labels = [str(ind_n) for ind_n in node_names_temp_plot]
    fig_Ts = plt.figure('Supply temperature main flow route')
    plt.xlabel('node')
    plt.ylabel(r'$T^s$ [C]')
    ax_Ts = fig_Ts.gca()
    ax_Ts.plot(Ts,marker='.',ls='-',color='tab:red')
    ax_Ts.set_xticks(range(len(xtick_labels)))
    ax_Ts.set_xticklabels(xtick_labels)
    ax_Ts.set_ylim(top=Ts_source,bottom=68.5)
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    
    fig_Tr = plt.figure('Return temperature main flow route')
    plt.xlabel('node')
    plt.ylabel(r'$T^r$ [C]')
    ax_Tr = fig_Tr.gca()
    ax_Tr.plot(Tr,marker='.',ls='-',color='tab:blue')
    ax_Tr.set_xticks(range(len(xtick_labels)))
    ax_Tr.set_xticklabels(xtick_labels)
    ax_Tr.set_ylim(top=29.9,bottom=29.5)
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
   
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        comp_link_types()
    
    plt.show()

