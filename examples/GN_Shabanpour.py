"""Gas network consisting of 3 demand/source nodes, and one extra node due to an compressor. Based on the example in Shabanpourt-Haghighi and Seifi"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.carrier import Gas
from meslf.utils.constants import bar, MSCM, hour, mm
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import pytest

# Carrier
mu_g = 0.288e-6 #[m^2/s]
S = 0.6106
Z = 0.8
pn = 1.01325*bar #[Pa]
Tn = 273.15 #[K]
T = 281.15 #[K]
R = 8.314413 #[J/molK]
M = 28.97e-3 #[kg/mol]
R_air = R/M #[J/kgK]
gas = Gas('high pres gas',S,R_air,Z,pn,Tn,T,mu=mu_g)
rhon_g = gas.rhon

# BC / solution
pg0 = 50*bar
q1_load = 10.8649*MSCM*rhon_g/hour
q2_load = 20*MSCM*rhon_g/hour
qc0 = 9.3382*MSCM*rhon_g/hour
qc1 = 2.7363*MSCM*rhon_g/hour
qc2 = 3.7756*MSCM*rhon_g/hour
q_inflow = q1_load+q2_load+qc0+qc1+qc2
pg2_sol = 34.08122413*bar

# solver information
tol = 1e-6
max_iter = 50

def create_network(hydr_eq='fa',c_hl=True,pipe_type='pipe_high_pres_colebrook',node_set=1):
    """Create a gas network consisting of 3 demand/source nodes, and one extra node due to an compressor.
    
    Parameters
    ----------
    hydr_eq : str, optional
        Determines which hydraulic equation is used. Options are 'fa' or 'fb'. Default is 'fa'.
    c_hl : bool, optional
        If true, half links are added to the nodes with values equal to the flow going to the coupling components. Default is True.
        
    Returns
    -------
    gas_net : GasNetwork
        The test network
    """
    
    g1 = GasNode('gn1',node_type=1,x=0,y=7,q=q1_load) # load node
    if node_set == 1:
        g0 = GasNode('gn0',node_type=0,x=5,y=12,p=pg0) # reference node
        if c_hl:
            GasHalfLink('gn0_hl0',start_node=g0,q=0) # slack
            GasHalfLink('gn0_hl1',start_node=g0,q=qc0) # flow to coupling
            GasHalfLink('gn0_hl2',start_node=g0,q=qc1) # flow to coupling
            g2 = GasNode('gn2',node_type=1,x=5,y=2,q=q2_load) # load node
            GasHalfLink('gn2_hl1',start_node=g2,q=qc2) # flow to coupling
        else:
            g2 = GasNode('gn2',node_type=1,x=5,y=2,q=q2_load+qc2) # load node
    elif node_set == 2:
        if c_hl:
            g0 = GasNode('gn0',node_type=1,x=5,y=12,q=-(q_inflow)) # load node (with total inflow)
            GasHalfLink('gn0_hl1',start_node=g0,q=qc0) # flow to coupling
            GasHalfLink('gn0_hl2',start_node=g0,q=qc1) # flow to coupling
            g2 = GasNode('gn2',node_type=0,x=5,y=2,p=pg2_sol) # reference node (take flow to coupling as the slack
            GasHalfLink('gn2_hl1',start_node=g2,q=0) # flow to coupling (taken as slack)
            GasHalfLink('gn2_hl0',start_node=g2,q=q2_load) #load
        else:
            g0 = GasNode('gn0',node_type=1,x=5,y=12,q=-(q_inflow-qc0-qc1)) # load node (with total inflow)
            g2 = GasNode('gn2',node_type=0,x=5,y=2,p=pg2_sol) # reference node
    else:
        raise ValueError('Enter valid node set')
    g3 = GasNode('gn3',node_type=1,x=2.5,y=4.5,q=0.) # load node
    
    L_g = 30000. #[m]
    D_g = 150.*mm #[mm]
    eps_g = 0.05*mm #[m]
    E = .98 # only needed for Weymouth friction factor
    compr_ratio = 1.3
    gas_link_params = {'D':D_g,'L':L_g,'eps':eps_g,'E':E,'carrier':gas}
    if hydr_eq =='fa':
        link_eq = 'q_of_dp'
    elif hydr_eq == 'fb':
        link_eq = 'dp_of_q'
    else:
        raise ValueError("Enter valid value for hydr_eq. Either 'fa' or 'fb'.")
    
    gl0 = GasLink('gl0',g0,g1,link_type = pipe_type,link_params = gas_link_params,link_eq_form=link_eq)
    gl1 = GasLink('gl1',g0,g2,link_type = pipe_type,link_params = gas_link_params,link_eq_form=link_eq)
    gl2 = GasLink('gl2',g3,g2,link_type = pipe_type,link_params = gas_link_params,link_eq_form=link_eq)
    gl3 = GasLink('gl3',g1,g3,link_type = 'compressor',link_params = {'r':compr_ratio},link_eq_form=link_eq)
    
    gas_net = GasNetwork('3 nodes')
    gas_net.add_link(gl0)
    gas_net.add_link(gl1)
    gas_net.add_link(gl2)
    gas_net.add_link(gl3)
    
    return gas_net

def initialize_network(gas_net,p0=55*bar,p1=45*bar,p2=40*bar,p3=45*bar,q=20*MSCM*rhon_g/hour,formulation='full',node_set=1):
    """Initialize the gas network, consisting of 3 demand/source nodes, and one extra node due to an compressor.
    
    Parameters
    ----------
    gas_net : GasNetwork
        The gas network to be initialized
    formulation : str, optional
        Formulation of the non-linear system of equations used to solve the network.
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    if formulation == 'nodal':
        for link in gas_net.get_links():
            if link.link_eq_form == 'dp_of_q':
                warnings.warn("The link {} uses press. drop as function of link flow (fb instead of fa), but formulation is 'nodal'. The link equation for this link is changed to fa!!".format(link.name))
                link.set_type(link.link_type,link.link_params,link_eq_form='q_of_dp')
    q_init = q*np.ones(len(gas_net.links))
    if node_set == 1:
        p_init = np.array([p1,p2,p3]) #[Pa]
    elif node_set == 2:
        p_init = np.array([p0,p1,p3])
    else:
        raise ValueError('Enter valid node set')
    if formulation == 'nodal':
        x_init = p_init
    else:
        x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    gas_net.update(x_init,formulation=formulation)
    x0 = gas_net.set_x_init(formulation=formulation)
    return x0

def run_load_flow(pgbase=50*bar,qbase=10*MSCM*rhon_g/hour,p0=55*bar,p1=40*bar,p2=35*bar,p3=40*bar,q=20*MSCM*rhon_g/hour,formulation='full',hydr_eq='fa',c_hl=True,pipe_type='pipe_high_pres_colebrook',node_set=1,tol=1e-6,max_iter=150):
    """Stead-state load flow analysis of gas network, using matrix scaling

    Parameters
    ----------
    
    Returns
    -------
    
    """
    # create network
    gas_net = create_network(hydr_eq=hydr_eq,c_hl=c_hl,pipe_type=pipe_type,node_set=node_set)
    # initialize
    x0 = initialize_network(gas_net,p0=p0,p1=p1,p2=p2,p3=p3,q=q,node_set=node_set,formulation=formulation)
    
    # solve network
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase})
    
    return gas_net,x_sol,iters,err_vec,p_sol,q_sol,q_inj,tol

def comp_links(node_set=1):
    """Compare convergence of the different hydraulic equations, for various pipe flow models."""
    # make figure to plot convergence
    fig_conv_gas = plt.figure('Convergence plot of NR, various pipes (node set {})'.format(node_set))
    ax_conv_gas = fig_conv_gas.gca()
    ax_conv_gas.set_xlabel(r'Iteration $k$')
    ax_conv_gas.set_ylabel(r'Error $||D_F F(x^k)||_2$')
    max_iters_used = 0
    
    fig_ord = plt.figure('Convergence order of NR, various pipes (node set {})'.format(node_set))
    ax_ord = fig_ord.gca()
    plt.xlabel(r'$||F(x^k)||_2 / ||F(x^0)||_2$')
    plt.ylabel(r'$||F(x^{k+1})||_2 / ||F(x^0)||_2$')
    
    
    linestyles = {'fa':'--','fb':'-'}
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] # should be the default color cycle
    pipes = ['pipe_high_pres_colebrook','pipe_high_pres_weymouth','pipe_high_pres_blasius','pipe_high_pres_churchill']#,'pipe_high_pres_pole' #NB. Colebrook has a scipy.optimize.solve erin zitten!
    
    for ind,pipe_type in enumerate(pipes):
        for c_hl in [True,False]:
            for hydr_eq in ['fa','fb']:
                print('\nHydraulic equation is {}, and seperate couplings is {} (node set {})'.format(hydr_eq,c_hl,node_set))
                # load flow (only full is used, since the pipes use a implicit friction factor.
                gas_net,x_sol,iters,err_vec,p_sol,q_sol,q_inj,tol = run_load_flow(hydr_eq=hydr_eq,c_hl=c_hl,pipe_type=pipe_type,node_set=node_set)
                print('Solution after {} iterations, final error {:.4e}:'.format(iters,err_vec[-1]))
                print('p = {} bar'.format(p_sol/bar))
                print('q = {} MSCM/hour'.format(q_sol/(MSCM*rhon_g/hour)))
                print('q nodal inj = {} MSCM/hour'.format(q_inj/(MSCM*rhon_g/hour)))
                print('q hl = {} MSCM/hour'.format([hl.q/(MSCM*rhon_g/hour) for node in gas_net.get_nodes() for hl in node.get_half_links()]))
                # plot convergence
                if c_hl:
                    marker = '*'
                    label = hydr_eq + ', sep. coup., ' + pipe_type
                else:
                    marker = '.'
                    label = hydr_eq + ', ' + pipe_type
                ax_conv_gas.semilogy(err_vec,ls=linestyles.get(hydr_eq),color=colors[ind],marker=marker,label=label)
                ax_ord.loglog(err_vec[:-1]/err_vec[0],err_vec[1:]/err_vec[0],ls=linestyles.get(hydr_eq),color=colors[ind],marker=marker,label=label)
                max_iters_used = max(max_iters_used,iters)
    layout_convergence(ax_conv_gas,tol,max_iters_used)
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
    ax.loglog(x_slope,x_slope,linestyle=':',color='k',label='slope 1')
    ax.loglog(x_slope,y_slope2,linestyle='-.',color='k',label='slope 2')
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    
@pytest.mark.filterwarnings("ignore::UserWarning")    
def example_gn_shabanpour_fa():
    """Check solution of the network, using full formulation, and fa. The flow going towards the couplings are taken as seperate half links. Scaling by matrix multiplication is used."""
    # Given / When
    gas_net,x_sol,iters,err_vec,p_sol,q_sol,q_inj,tol = run_load_flow(formulation='full',hydr_eq='fa',c_hl=True)
    
    # Then
    p_g_sol_expected = np.array([29.1021,34.07704,37.83273])*bar #[Pa]
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour #[kg/s]
    x_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    
    rel_tol = 1e-5
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning")    
def example_gn_shabanpour_fb():
    """Check solution of the network, using full formulation, and fa. The flow going towards the couplings are taken as seperate half links. Scaling by matrix multiplication is used."""
    # Given / When
    gas_net,x_sol,iters,err_vec,p_sol,q_sol,q_inj,tol = run_load_flow(formulation='full',hydr_eq='fb',c_hl=True)
    
    # Then
    p_g_sol_expected = np.array([29.1021,34.07704,37.83273])*bar #[Pa]
    q_sol_expected = np.array([18.2327, 16.4078, 7.3678,7.3678])*MSCM*rhon_g/hour #[kg/s]
    x_sol_expected = np.concatenate((q_sol_expected,p_g_sol_expected))
    
    rel_tol = 1e-5
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)
    
if __name__ == '__main__':
    for node_set in [1,2]:
        comp_links(node_set=node_set)
    plt.show()
