"""Gas network consisting of 3 demand/source nodes. Also called the reduced benchmark problem."""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.carrier import Gas
from meslf.utils.constants import bar, mbar, hour, mm
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import pytest
import os
import pandas as pd
from meslf.networks.read_write_network import from_pd_dataframes
import warnings

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning 

# Read the scenario data
def read_scen_data(path_to_data,c_hl=True,topology=1,single_coupling=False):
    """Read the scenario data
    
    Parameters
    ----------
    c_hl : bool, optional
        If true, half links are added to the nodes with values equal to the flow going to the coupling components. Default is True.
    topology : int, optional
        Determines which topology is used in the MES, hence, which is used in the gas network when the coupling components are taken into account separately. Options are 1-4. Default is 1. 
    single_coupling : bool, optional
        Determines if a single coupling node (either CHP or EH) is used in the MES, when coupled to one gas node and one gas node (i.e., when topology 1 is used). Default is False. Only used when topology is 1 and if c_hl is True. 
        
    Returns
    -------
    gas_net_scen : GasNetwork
        The single-carrier scenario network.
    q_scen : list
        List with link flow in the single-carrier scenario.
    p_scen : list
        List with nodal pressures in the single-carrier scenario.
    q_hl_scen : list
        List with half link flows in the single-carrier scenario.
    x : list
        List with node x-coordinates. 
    y : list
        List with node y-coordinates.
    mes_net_scen : HeterogeneousNetwork
        The multi-carrier scenario network. Is None if c_hl is False.
    q_mes_scen : list
        List with gas link flow in the multi-carrier scenario. Is an empty list if c_hl is False.
    q_hl_scen : list
        List with gas halflink flow in the multi-carrier scenario. Is an empty list if c_hl is False.
    """
    nodes = pd.read_pickle(os.path.join(path_to_data, 'GN_BP_nodes.pkl'))
    links = pd.read_pickle(os.path.join(path_to_data, 'GN_BP_links.pkl'))
    halflinks = pd.read_pickle(os.path.join(path_to_data, 'GN_BP_halflinks.pkl'))
    gas_net_scen = from_pd_dataframes(nodes,links,halflinks)

    # (full) solution of scenario, single-carrier
    q_scen = np.zeros(len(gas_net_scen.links))
    q_hl_scen = np.zeros(len(gas_net_scen.half_links))
    p_scen = np.zeros(len(gas_net_scen.nodes))
    for ind_e,link in enumerate(gas_net_scen.get_links()):
        q_scen[ind_e] = link.get_q()
    for ind_n,node in enumerate(gas_net_scen.get_nodes()):
        p_scen[ind_n] = node.get_p()
    for ind_hl,half_link in enumerate(gas_net_scen.get_half_links()):
        q_hl_scen[ind_hl] = half_link.get_q()
    
    if c_hl:
        mes_data = 'top'+str(topology)
        if single_coupling:
            mes_data += '_1c'
        else:
            mes_data += '_2c'
        nodes_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_nodes_'+mes_data+'.pkl'))
        links_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_links_'+mes_data+'.pkl'))
        halflinks_mes = pd.read_pickle(os.path.join(path_to_data, 'MES_BP_halflinks_'+mes_data+'.pkl'))
        mes_net_scen = from_pd_dataframes(nodes_mes,links_mes,halflinks_mes)

        # (full) solution of scenario, multi-carrier
        q_mes_scen = [link.get_q() for link in mes_net_scen.get_links() if isinstance(link,GasLink)]
        q_hl_mes_scen = [hl.get_q() for node in mes_net_scen.get_nodes() if isinstance(node,GasNode) for hl in node.get_half_links() ]
        x = [node.x for node in mes_net_scen.get_nodes() if isinstance(node,GasNode)]
        y = [node.y for node in mes_net_scen.get_nodes() if isinstance(node,GasNode)]
    else:
        mes_net_scen = None
        q_mes_scen = []
        q_hl_mes_scen = []
        x = [node.x for node in gas_net_scen.get_nodes()]
        y = [node.y for node in gas_net_scen.get_nodes()]
    return gas_net_scen,q_scen,p_scen,q_hl_scen,x,y,mes_net_scen,q_mes_scen,q_hl_mes_scen
        
def create_network(path_to_data,hydr_eq='fa',c_hl=True,topology=1,single_coupling=False):
    """Create a gas network consisting of 3 demand/source nodes.
    
    Parameters
    ----------
    hydr_eq : str, optional
        Determines which hydraulic equation is used. Options are 'fa' or 'fb'. Default is 'fa'.
    c_hl : bool, optional
        If true, half links are added to the nodes with values equal to the flow going to the coupling components. Default is True.
    topology : int, optional
        Determines which topology is used in the MES, hence, which is used in the gas network when the coupling components are taken into account separately. Options are 1-4. Default is 1. 
    single_coupling : bool, optional
        Determines if a single coupling node (either CHP or EH) is used in the MES, when coupled to one gas node and one gas node (i.e., when topology 1 is used). Default is False. Only used when topology is 1 and if c_hl is True. 
        
    Returns
    -------
    gas_net : GasNetwork
        The gas network
    """
    if not topology in [1,2,3,4]:
        raise ValueError('Enter valid value for topology')
    gas_net_scen,q_scen,p_scen,q_hl_scen,x,y,mes_net_scen,q_mes_scen,q_hl_mes_scen = read_scen_data(path_to_data,c_hl=c_hl,topology=topology,single_coupling=single_coupling)
    
    # physical parameters of network and pipes
    gas = gas_net_scen.links[0].link_params.get('carrier') 
    rhon_g = gas.rhon #[kg/m^3]
    gas_link_params = gas_net_scen.links[0].link_params.copy()
    gas_link_type = gas_net_scen.links[0].link_type
    if hydr_eq =='fa':
        link_eq = 'q_of_dp'
    elif hydr_eq == 'fb':
        link_eq = 'dp_of_q'
    else:
        raise ValueError("Enter valid value for hydr_eq. Either 'fa' or 'fb'.")
    
    # boundary conditions
    p0 = p_scen[0]
    if mes_net_scen:
        q1_load = q_hl_mes_scen[1]
        q2_load = q_hl_mes_scen[2]
    else:
        q1_load = q_hl_scen[1]
        q2_load = q_hl_scen[2]
    g0 = GasNode('gn0',node_type=0,x=x[0],y=y[0],p=p0) # reference node
    g1 = GasNode('gn1',node_type=1,x=x[1],y=y[1],q=q1_load) # load node
    g2 = GasNode('gn2',node_type=1,x=x[2],y=y[2],q=q2_load) # load node
    
    if c_hl:
        if topology == 1 and single_coupling:
            qc = q_mes_scen[3]
            GasHalfLink('gn2_hl1',start_node=g2,q=qc) # flow to coupling
        elif topology == 1 or topology == 3:
            qc0 = q_mes_scen[3]
            qc1 = q_mes_scen[4]
            GasHalfLink('gn2_hl1',start_node=g2,q=qc0) # flow to coupling
            GasHalfLink('gn2_hl2',start_node=g2,q=qc1) # flow to coupling
        else:
            qc0 = q_mes_scen[3]
            qc1 = q_mes_scen[4]
            GasHalfLink('gn1_hl1',start_node=g1,q=qc0) # flow to coupling
            GasHalfLink('gn2_hl1',start_node=g2,q=qc1) # flow to coupling
    
    gl0 = GasLink('gl0',g0,g1,link_type = gas_link_type,link_params = gas_link_params,link_eq_form=link_eq)
    gl1 = GasLink('gl1',g0,g2,link_type = gas_link_type,link_params = gas_link_params.copy(),link_eq_form=link_eq)
    gl2 = GasLink('gl2',g1,g2,link_type = gas_link_type,link_params = gas_link_params.copy(),link_eq_form=link_eq)
    
    gas_net = GasNetwork('3 nodes')
    gas_net.add_link(gl0)
    gas_net.add_link(gl1)
    gas_net.add_link(gl2)
    
    return gas_net

def initialize_network(gas_net,p1=29*mbar,p2=28*mbar,q=.05,formulation='full',scale_var=None,scale_var_params=None):
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
    q_init = list()
    p_init = list()
    x_entries = gas_net.get_x_entries(formulation)
    for ind_el,el in enumerate(x_entries):
        if isinstance(el,GasNode):
            if el.name == 'gn1':
                p_init.append(p1)
            elif el.name == 'gn2':
                p_init.append(p2)
        elif isinstance(el,GasLink):
            if formulation == 'nodal':
                if link.link_eq_form == 'dp_of_q':
                    warnings.warn("The link {} uses press. drop as function of link flow (fb instead of fa), but formulation is 'nodal'. The link equation for this link is changed to fa!!".format(link.name))
                    link.set_type(link.link_type,link.link_params,link_eq_form='q_of_dp')
            q_init.append(q)
    
    x_init = np.array(q_init+p_init)
    gas_net.initialize()
    gas_net.update(x_init,formulation=formulation)
    x0 = gas_net.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def run_load_flow(path_to_data,pgbase=50*mbar,qbase=.05,p1=29*mbar,p2=28*mbar,q=.05,formulation='full',hydr_eq='fa',c_hl=True,topology=1,single_coupling=False,tol=1e-6,max_iter=150):
    """Stead-state load flow analysis of gas network, using matrix scaling

    Parameters
    ----------
    
    Returns
    -------
    
    """
    # create network
    gas_net = create_network(path_to_data,hydr_eq=hydr_eq,c_hl=c_hl,topology=topology,single_coupling=single_coupling)
    # initialize
    x0 = initialize_network(gas_net,p1=p1,p2=p2,q=q,formulation=formulation)
    
    # solve network
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase})
    
    return gas_net,x_sol,iters,err_vec,p_sol,q_sol,q_inj,tol

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
    
def comp_conv_form(path_to_data):
    """Compare convergence of the different hydraulic equations."""
    # make figure to plot convergence
    fig_conv_gas = plt.figure('Convergence plot gas network')
    ax_conv_gas = fig_conv_gas.gca()
    ax_conv_gas.set_xlabel(r'Iteration $k$')
    ax_conv_gas.set_ylabel(r'Error $||D_F F(x^k)||_2$')
    max_iters_used = 0
    
    fig_ord = plt.figure('Convergence order gas network')
    ax_ord = fig_ord.gca()
    plt.xlabel(r'$||F(x^k)||_2 / ||F(x^0)||_2$')
    plt.ylabel(r'$||F(x^{k+1})||_2 / ||F(x^0)||_2$')
    
    markers_gas = {'full fa':'d','full fb':'.','nodal fa':'*'}
    
    topology = 4
    single_coupling = False
    for c_hl in [True,False]:
        for hydr_eq in ['fa','fb']:
            for form_gas in ['full','nodal']:
                if not (form_gas == 'nodal' and hydr_eq == 'fb'): #those cannot be combined
                    print('\nHydraulic equation is {}, and separate couplings is {}, formulation = {}'.format(hydr_eq,c_hl,form_gas))
                    # load flow (only full is used, since the pipes use a implicit friction factor.
                    gas_net,x_sol,iters,err_vec,p_sol,q_sol,q_inj,tol = run_load_flow(path_to_data,hydr_eq=hydr_eq,c_hl=c_hl,topology=topology,single_coupling=single_coupling)
                    print('Solution:')
                    print('p = {} mbar'.format(p_sol/mbar))
                    print('q = {} kg/s'.format(q_sol))
                    print('q nodal inj = {} kg/s'.format(q_inj))
                    print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
                    key = '{} {}'.format(form_gas,hydr_eq)
                    # plot convergence
                    if c_hl:
                        ls = '--'
                        label =  '{}, {}, sep. coup.'.format(form_gas,hydr_eq)
                    else:
                        ls = '-'
                        label = '{}, {}'.format(form_gas,hydr_eq)
                    ax_conv_gas.semilogy(err_vec,ls=ls,color='tab:green',marker=markers_gas.get(key),label=label)
                    ax_ord.loglog(err_vec[:-1]/err_vec[0],err_vec[1:]/err_vec[0],ls=ls,color='tab:green',marker=markers_gas.get(key),label=label)
                    max_iters_used = max(max_iters_used,iters)
                    fig_top = plt.figure('Network topology {}, separate coupling {}'.format(topology,c_hl))
                    ax_top = fig_top.gca()
                    gas_net.draw_network(ax_top,halflink_angle=0,halflink_length=1)
                    plt.axis('equal')
                    plt.axis('off')
    layout_convergence(ax_conv_gas,tol,max_iters_used)
    layout_convergence_order(ax_ord)

def comp_conv_scaling(path_to_data,hydr_eq='fa',topology=1,single_coupling=False):
    """Compare convergence of NR for different ways of scaling."""
    # base values
    pgbase=50*mbar
    qbase=.05
    
    # create networks
    c_hl = False # no separate coupling
    # network with values specified in S.I.
    gas_net_SI = create_network(path_to_data,hydr_eq=hydr_eq,c_hl=c_hl,topology=topology,single_coupling=single_coupling) 
    # network with values specified in p.u.
    gas_net_pu= create_network(path_to_data,hydr_eq=hydr_eq,c_hl=c_hl,topology=topology,single_coupling=single_coupling) 
    for link in gas_net_pu.get_links():
        if not link.link_type == 'pipe_low_pres_pole':
            raise ValueError('Cannot create network with values specified in p.u. Link type is wrong')
        else:
            C_SI = link.pipe_const() # pipe constant of 'low pres.' pipe in S.I.
            C_b = qbase/np.sqrt(pgbase)
            fric_fac = .0065 # since the pipe is assumed to have Pole's friction factor
            C_pu = (C_SI/np.sqrt(fric_fac))/C_b# pipe constant for a resistor, equivalent to the 'low pres.' pipe, in p.u.
            link.set_type('resistor',{'C':C_pu},link_eq_form=link.link_eq_form)
    for node in gas_net_pu.get_nodes():
        node.p /= pgbase
        for hl in node.get_half_links():
            hl.q /= qbase
            
    # initial conditions
    p1=29*mbar
    p2=28*mbar
    q=.05
    
    formulation = 'full'
    tol=1e-6
    max_iter=50
    # run load flow for network with values specified in S.I., using matrix scaling
    # initialize
    x0 = initialize_network(gas_net_SI,p1=p1,p2=p2,q=q,formulation=formulation)
    # solve network
    x_sol_scaled,iters_scaled,err_vec_scaled,p_sol_scaled,q_sol_scaled,q_inj_scaled = gas_net_SI.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase})
    
    # run load flow for network with values specified in S.I., using p.u. scaling
    scale_var='per_unit'
    scale_var_params={'qbase':qbase,'pbase':pgbase}
    # initialize
    gas_net_SI.reset_network(x0,formulation=formulation)
    x0_SI_pu = initialize_network(gas_net_SI,p1=p1,p2=p2,q=q,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    # solve network
    x_sol_SI_pu,iters_SI_pu,err_vec_SI_pu,p_sol_SI_pu,q_sol_SI_pu,q_inj_SI_pu = gas_net_SI.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    
    # run load flow for network with values specified in p.u., without scaling
    # initialize
    x0_pu = initialize_network(gas_net_pu,p1=p1/pgbase,p2=p2/pgbase,q=q/qbase,formulation=formulation)
    # solve network
    x_sol_pu,iters_pu,err_vec_pu,p_sol_pu,q_sol_pu,q_inj_pu = gas_net_pu.solve_network(tol,max_iter,solver='NR',formulation=formulation)
    
    print('Errors. Par. in S.I., matrix scaling:\n{}'.format(err_vec_scaled))
    print('Errors. Par. in S.I., p.u. scaling:\n{}'.format(err_vec_SI_pu))
    print('Errors. Par. in p.u., unscaled:\n{}'.format(err_vec_pu))
    
    # make figure to plot convergence
    fig_conv_gas = plt.figure('Convergence plot gas network, scaling')
    ax_conv_gas = fig_conv_gas.gca()
    max_iters_used = max([iters_scaled,iters_SI_pu,iters_pu])
    markers_gas = {'full fa':'d','full fb':'.','nodal fa':'*'}
    ls = '-'
    ax_conv_gas.semilogy(err_vec_scaled,ls=ls,color='tab:blue',marker=markers_gas.get(formulation+' '+hydr_eq),label='matrix scaling')
    ax_conv_gas.semilogy(err_vec_SI_pu,ls=ls,color='tab:orange',marker=markers_gas.get(formulation+' '+hydr_eq),label='p.u. scaling')
    ax_conv_gas.semilogy(err_vec_pu,ls=ls,color='tab:red',marker=markers_gas.get(formulation+' '+hydr_eq),label='specified in p.u.')
    ax_conv_gas.set_xlabel(r'Iteration $k$')
    ax_conv_gas.set_ylabel(r'Error ($||D_F F(x^k)||_2$ or $||F(x^k)||_2$)')
    layout_convergence(ax_conv_gas,tol,max_iters_used)

def comp_links(path_to_data):
    """Compare convergence for different pipe models."""
    fig_conv = plt.figure('Convergence of NR, various pipes')
    ax_conv = fig_conv.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F(x^k)||_2$')
    max_iter_used = 0
    
    fig_ord = plt.figure('Convergence order of NR, various pipes')
    ax_ord = fig_ord.gca()
    plt.xlabel(r'$||F(x^k)||_2 / ||F(x^0)||_2$')
    plt.ylabel(r'$||F(x^{k+1})||_2 / ||F(x^0)||_2$')
    # layout
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] # should be the default color cycle
    markers = {'fa':'*','fb':'.'}
    linestyles = {'fa':'--','fb':'-'}
    
    # pipes
    pipes = ['pipe_low_pres_hagen_poiseuille','pipe_low_pres_churchill','pipe_low_pres_pole']
    
    # base values
    pgbase=50*mbar
    qbase=.05
    # topology 
    topology = 4
    single_coupling = False
    c_hl = False
    eps = .5*mm
    mu = mu_g = 0.288e-6 #[m^2/s]
    # initial conditions
    p1=29*mbar
    p2=28*mbar
    q=.05
    # solver info
    tol = 1e-6
    max_iter = 10
    formulation = 'full'
    for ind,pipe_type in enumerate(pipes):
        for hydr_eq in ['fa','fb']:
            print('\nRunning load flow with pipes: {}, and hydr. eq {}'.format(pipe_type,hydr_eq))
            
            # create network
            gas_net = create_network(path_to_data,hydr_eq=hydr_eq,c_hl=c_hl,topology=topology,single_coupling=single_coupling)
            # change pipe types to new pipes
            for link in gas_net.get_links():
                carrier = link.link_params.get('carrier')
                carrier.mu = mu
                link_params = {'L':link.link_params.get('L'),'D':link.link_params.get('D'),'eps':eps,'carrier':carrier}
                if hydr_eq == 'fa':
                    hydr_eq_form = 'q_of_dp'
                elif hydr_eq == 'fb':
                    hydr_eq_form = 'dp_of_q'
                link.set_type(pipe_type,link_params,link_eq_form=hydr_eq_form)
            # initialize
            x0 = initialize_network(gas_net,p1=p1,p2=p2,q=q,formulation=formulation)
            
            # solve network
            x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase})
            print('Error is {:.4e}, after {} iterations'.format(err_vec[-1],iters))
            print('Solution:')
            print('p = {} mbar'.format(p_sol/mbar))
            print('q = {} kg/s'.format(q_sol))
            print('q nodal inj = {} kg/s'.format(q_inj))
            print('q hl = {} kg/s'.format([hl.q for node in gas_net.get_nodes() for hl in node.get_half_links()]))
            
            # plot convergence and convergence order
            ax_conv.semilogy(np.asarray(range(0,iters+1)),err_vec,marker=markers.get(hydr_eq),linestyle=linestyles.get(hydr_eq),color=colors[ind],label=hydr_eq+', '+pipe_type)
            max_iter_used = max(max_iter_used,iters)
            ax_ord.loglog(err_vec[:-1]/err_vec[0],err_vec[1:]/err_vec[0],marker=markers.get(hydr_eq),linestyle=linestyles.get(hydr_eq),color=colors[ind],label=hydr_eq+', '+pipe_type)
    layout_convergence(ax_conv,tol,max_iter_used)
    layout_convergence_order(ax_ord)
    
if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','N_BP')
    comp_conv_form(path_to_data)
    comp_conv_scaling(path_to_data)
    comp_links(path_to_data)
    plt.show()

