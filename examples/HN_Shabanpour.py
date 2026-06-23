"""Heat network consisting of 3 demand/source nodes. Based on the example in Shabanpourt-Haghighi and Seifi"""
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.carrier import Water
from meslf.utils.constants import MW, mm
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import warnings
import pytest

# water carrier
rho_w = 960. #[kg/m^3]
mu_w = 0.294e-6 #[m^2/s]
Cp_w = 4.182e3 #[J/(kg K)]
grav_const = 9.81 #[m/s^2]
water = Water('water',Cp_w,rho=rho_w,mu=mu_w)

# physical parameters of network and pipes
Ta = 10.

# BC / solution
Ts0 = 120.
p0 = 5517.*rho_w*grav_const
phi1_load = 35.*MW
Tr1_load = 50.
dT1_load = 69.0370
phi2_load = 20.*MW
Tr2_load = 50.
dT2_load = 73.5410
Tr2_sol = 49.5339
Ts2_source = 123.5410 #(when node 2 acts as source)
dT2_source = Ts2_source - Tr2_sol #(when node 2 acts as source)
p3_sol = 4268.1087*rho_w*grav_const
Toc1 = 126.4891 #CHP
phic1 = 29.0152*MW #CHP
dT3 = Toc1 - Tr2_sol

def create_network(c_hl=True,heat_load='outflow'):
    """Create a heat network consisting of 3 demand/source nodes.

    Parameters
    ----------
    c_hl : bool, optional
        If true, dummy links and extra nodes (!! not half links !!) are added with values equal to the flow coming from the coupling components. Default is True.

    Returns
    -------
    heat_net : HeatNetwork
        The test network
    """
    # parameters for coupling part
    h0 = HeatNode('hn0',node_type=0,p=p0,Ts=Ts0) # slack. In the MES, this slack is replaced with a boiler, such that this node becomes a boiler. Since the single-carrier network needs a slack, the coupling from the boiler is not modelled explicitely
    h0.half_links[0].set_type('heat_exchanger',{'carrier':water})
    if heat_load == 'outflow':
        h1 = HeatNode('hn1',node_type=1,Tr_hl=Tr1_load,dphi=phi1_load) # load node (sink)
        h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
        if c_hl:
            h2 = HeatNode('hn2',node_type=1,Tr_hl=Tr2_load,dphi=phi2_load) # load  node (sink)
            h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
            h3 = HeatNode('hn_c2',node_type=3,Ts_hl=Toc1,dphi=-phic1,p=p3_sol) # ref. load node (since only connected with a dummy link, so no way to determine pressure)
            h3.half_links[0].set_type('heat_exchanger',{'carrier':water})
        else:
            h2 = HeatNode('hn2',node_type=1,Ts_hl=Ts2_source,dphi=phi2_load-phic1) # load  node
            h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    elif heat_load == 'delta':
        h1 = HeatNode('hn1',node_type=12,dphi=phi1_load,dT=dT1_load) # sink temp. diff. node
        h1.half_links[0].set_type('heat_exchanger',{'carrier':water})
        if c_hl:
            h2 = HeatNode('hn2',node_type=12,dphi=phi2_load,dT=dT2_load) # sink temp. diff. node
            h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
            h3 = HeatNode('hn_c2',node_type=13,dphi=-phic1,p=p3_sol,dT=dT3) # ref. source temp. diff. node (since only connected with a dummy link, so no way to determine pressure)
            h3.half_links[0].set_type('heat_exchanger',{'carrier':water})
        else:
            h2 = HeatNode('hn2',node_type=12,dphi=phi2_load-phic1,dT=dT2_source) # source temp. diff. node
            h2.half_links[0].set_type('heat_exchanger',{'carrier':water})
    else:
        raise ValueError("Enter valid value for 'heat_load'. Either 'outflow' or 'delta'.")

    L_h = 30000. #[m]
    D_h = 150.*mm #[mm]
    lam = 0.2 #[W/(mK)]
    eps_h = 1.25*mm #[m]
    U = lam/(np.pi*D_h) #[W/(m^2 K)]
    heat_link_params = {'D':D_h,'L':L_h,'eps':eps_h,'U':U,'carrier':water,'Ta':Ta}
    hl0 = HeatLink('hl0',h0,h1,link_type='standard_pipe_low_pres_colebrook',link_params=heat_link_params)
    hl1 = HeatLink('hl1',h0,h2,link_type='standard_pipe_low_pres_colebrook',link_params=heat_link_params.copy())
    hl2 = HeatLink('hl2',h1,h2,link_type='standard_pipe_low_pres_colebrook',link_params=heat_link_params.copy())

    heat_net = HeatNetwork('3 nodes',Ta=Ta)
    heat_net.add_link(hl0)
    heat_net.add_link(hl1)
    heat_net.add_link(hl2)
    if c_hl:
        hl3 = HeatLink('hl3',h3,h2,link_type='dummy',link_params=heat_link_params.copy())
        heat_net.add_link(hl3)

    # coordinates
    h0.x=9
    h0.y=12
    h1.x=4
    h1.y=7
    h2.x=9
    h2.y=2
    if c_hl:
        h3.x=9
        h3.y=-3
    return heat_net

def initialize_network(heat_net,c_hl=True,formulation='standard',heat_load='outflow'):
    """Create a heat network consisting of 3 nodes in one line.

    Parameters
    ----------
    heat_net : HeatNetwork
        The heat network to be initialized

    Returns
    -------
    x0 : np array
        initial guess
    """
    if c_hl:
        m_init = np.array([60.,30.,60.,100.]) #[kg/s]
        p_init = np.array([10.,4000.])*rho_w*grav_const #[Pa]
        Ts_init = np.array([120.,100.,120.]) #[C]
        Tr_init = np.array([50.,50.,50.,50.])##[C]
        if formulation == 'half_link_flow':
            m_hl_init = np.array([20.,20.,-100.]) #[kg/s]
            if heat_load == 'delta':
                Ts_hl_init = np.array([120.]) #[C]
                Tr_hl_init = np.array([50.,50.]) #[C]
    else:
        m_init = np.array([60.,30.,60.]) #[kg/s]
        p_init = np.array([10.,4000.])*rho_w*grav_const #[Pa]
        Ts_init = np.array([120.,120.]) #[C]
        Tr_init = np.array([50.,50.,50.]) #[C]
        if formulation == 'half_link_flow':
            m_hl_init = np.array([20.,20.]) #[kg/s]
            if heat_load == 'delta':
                Ts_hl_init = np.array([120.]) #[C]
                Tr_hl_init = np.array([50.]) #[C]

    if formulation == 'half_link_flow':
        if heat_load == 'delta':
            x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init,Ts_hl_init,Tr_hl_init))
        else:
            x_init = np.concatenate((m_init,m_hl_init,p_init,Ts_init,Tr_init))
    else:
        x_init = np.concatenate((m_init,p_init,Ts_init,Tr_init))

    heat_net.initialize()
    heat_net.update(x_init,formulation=formulation)
    x0 = heat_net.set_x_init(formulation=formulation)
    return x0

def run_load_flow(phbase=1000.*rho_w*grav_const,mbase=50.,Tbase=130.,phibase=100.,c_hl=True,tol=1e-6,max_iter=50,formulation='standard',heat_load='outflow'):
    """Stead-state load flow analysis of heat network

    Parameters
    ----------
    phbase : float
        Base value used for pressure.
    mbase : float
        Base value used for link flow.
    Tbase : float
        Base value for temperature.
    phibase : float
        Base value for heat power.


    """
    # create network
    heat_net = create_network(c_hl=c_hl,heat_load=heat_load)
    # initialize
    x0 = initialize_network(heat_net,c_hl=c_hl,formulation=formulation,heat_load=heat_load)

    # solve network
    x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = heat_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase})

    return heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,tol

def compare_conv_form():
    """Compare convergence for different formulation, and for different heat load models."""
    # make figure to plot convergence
    fig_conv_heat = plt.figure('Convergence plot heat network')
    ax_conv_heat = fig_conv_heat.gca()
    max_iters_used = 0
    markers_heat = {'standard outflow':'.','standard delta':'*','half_link_flow outflow':'d','half_link_flow delta':'x'}
    for c_hl in [True,False]:
        for form in ['standard', 'half_link_flow']:
            for heat_load in ['outflow','delta']:
                print('\nFormulation is {}, and heat load is {}, and seperate couplings is {}'.format(form,heat_load,c_hl))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                    heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,tol = run_load_flow(c_hl=c_hl,heat_load=heat_load,formulation=form)
                print('Solution:')
                print('p heat = {} m?'.format(p_vec/(rho_w*grav_const)))
                print('m = {}'.format(m_vec))
                print('Ts = {}'.format(Ts_vec))
                print('Tr = {}'.format(Tr_vec))
                print('m hl = {}'.format(m_hl_vec))
                print('Ts hl = {}'.format(Ts_hl_vec))
                print('Tr hl = {}'.format(Tr_hl_vec))
                print('phi hl = {}'.format(phi_hl_vec))
                key = '{} {}'.format(form,heat_load)
                # plot convergence
                if c_hl:
                    ls = '--'
                    label = key + ', sep. coup.'
                else:
                    ls = '-'
                    label = key
                ax_conv_heat.semilogy(err_vec,ls=ls,color='tab:blue',marker=markers_heat.get(key),label=label)
                max_iters_used = max(max_iters_used,iters)
    ax_conv_heat.set_xlabel(r'Iteration $k$')
    ax_conv_heat.set_ylabel(r'Error $||D_F F(x^k)||_2$')
    ax_conv_heat.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
    ax_conv_heat.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_heat.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_heat.legend()
    xmin = 0
    xmax = max_iters_used
    xticks = range(xmin,xmax+1,2) # make sure the xticks are integers
    ax_conv_heat.set_xlim(left=xmin,right=xmax+1)
    ax_conv_heat.set_xticks(xticks)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_hn_shabanpour_standard_outflow_sep_coup():
    """Check the solution of the network, using the standard formulation, and assuming To known for sources and sinks. Flow coming from the couplings are modeled seperately. Scaling by matrix multiplication is used."""
    # Given / When
    heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,tol = run_load_flow(c_hl=True,formulation='standard',heat_load='outflow')

    # Then
    m_sol_expected = np.array([64.6877,31.4083,-56.5379, 90.1578]) #[kg/s]
    p_h_sol_expected = np.array([224.9319,4268.1087])*rho_w*grav_const #[Pa]
    Ts_sol_expected = np.array([119.0370, 123.5410,126.4891]) #[C]
    Tr_sol_expected = np.array([48.6805,50.,49.5339,49.5339]) #[C]
    x_sol_expected = np.concatenate((m_sol_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected))

    rel_tol = 1e-4
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

@pytest.mark.filterwarnings("ignore::UserWarning","ignore::scipy.sparse.SparseEfficiencyWarning")
def example_hn_shabanpour_half_link_flow_delta_sep_coup():
    """Check the solution of the network, using the half link flow formulation, and assuming dT known for sources and sinks. Flow coming from the couplings are modeled seperately. Scaling by matrix multiplication is used."""
    # Given / When
    heat_net,x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,tol = run_load_flow(c_hl=True,formulation='half_link_flow',heat_load='delta')
    print('\nError = {} after {} iterations.'.format(err_vec[-1],iters))

    # Then
    m_sol_expected = np.array([64.6877,31.4083,-56.5379, 90.1578]) #[kg/s]
    m_hl_expected = np.array([121.2256,65.0282, -90.1578]) #[kg/s]
    p_h_sol_expected = np.array([224.9319,4268.1087])*rho_w*grav_const #[Pa]
    Ts_sol_expected = np.array([119.0370, 123.5410,126.4891]) #[C]
    Tr_sol_expected = np.array([48.6805,50.,49.5339,49.5339]) #[C]
    Ts_hl_sol_expected = np.array([126.4891]) #[C]
    Tr_hl_sol_expected = np.array([50.,50.]) #[C]
    x_sol_expected = np.concatenate((m_sol_expected,m_hl_expected,p_h_sol_expected,Ts_sol_expected,Tr_sol_expected,Ts_hl_sol_expected,Tr_hl_sol_expected))

    rel_tol = 1e-2
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol)

if __name__ == '__main__':
    compare_conv_form()

    plt.show()
