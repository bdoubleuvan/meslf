"""Example of a gas network with 22 nodes. See the test case by Osiadacz (page 168-173)
"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink
from meslf.networks.carrier import Gas
import numpy as np
import pandas as pd
import networkx as nx
import meslf.load_flow.system_of_equations as NLS
import matplotlib.pyplot as plt
from meslf.utils.constants import bar, hour, mm, km
import warnings

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning 

#carrier
Z = 0.93323990580143557
T = 283 #[K]
S = 0.6
Tn = 288 #[K]
pn = 1.*bar #[Pa]
R = 8.314413 #[J/molK]
M = 28.97e-3 #[kg/mol]
R_air = R/M #[J/kgK]
mu = 0.288e-6 #[m^2/s] (only used if colebrook or churchill pipe is used)
carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T,mu=mu)   

def network_data(carrier,pipe_type='pipe_high_pres_weymouth'):
    """Creates the node and link data for the network
        
    Returns
    -------
    nodes : pd DataFrame
        Node data
    links : pd DataFrame
        Link data
    """
    # create pandas dataframe with node data
    node_names =  list(str(i) for i in range(1,26))
    node_types = np.ones(len(node_names),dtype=int) # majority are loads
    node_types[node_names.index('1')] = 0 #ref. node
    
    flows = 10* 100*np.array([0.,90.,29.,75.,0.,55.,85.,28.,90.,41.,39.,20.,0.,80.,45.,0.,120.,42.,18.,35.,29.,71.,0.,0.,0.]) # [m^3/h] (times 10 to correct for mistake in Osiadacz)
    flows = flows*carrier.rhon/hour #[kg/s]
    node_names = np.array(node_names,dtype=np.int8)
    
    nodes = pd.DataFrame(np.array([node_names,node_types,flows]).T,columns=['name', 'type', 'q_inj'])
    nodes['type'] = nodes['type'].astype(int)
    nodes['name'] = nodes['name'].astype(str)
    
    # create pandas dataframe with link data
    link_names = list(range(1,39)) 
    link_types = list(np.zeros(len(link_names))) # majority are pipes
    for name in range(1,36):
        link_types[link_names.index(name)] = pipe_type
    link_types[link_names.index(36)] = 'compressor' 
    link_types[link_names.index(37)] = 'compressor' 
    link_types[link_names.index(38)] = 'compressor' 
    
    start_nodes = np.array([1,1,1,4,3,2,2,23,22,6,4,4,4,12,12,11,24,10,14,10,15,15,25,10,9,9,8,9,8,7,6,18,7,20,20,5,13,16]) #names of start nodes
    end_nodes = np.array([2,3,4,3,2,22,5,6,21,21,10,11,12,13,11,10,14,14,15,9,9,16,17,6,17,8,17,7,18,18,7,19,20,19,21,23,24,25]) #names of end nodes
    length = km*np.array([24,25,20,30,40,45,70,60,52,30,40,35,55,70,30,50,60,100,80,75,80,75,80,40,65,40,55,45,30,42,20,30,40,32,45,0,0,0]) # length (of pipes) in meter
    diameter = 0.7*np.ones(len(link_names)) # diameter (of pipes) in meter
    for i in range(1,39):
        if i in [7,8,10,12,14,16,17,18,20,22,24,26,27,28,30,32,34]:
            diameter[link_names.index(i)] = .6
    #correction
    diameter[link_names.index(29)] =.6
    # set diameter to some non realistic value for compressors
    diameter[link_names.index(36)] = 0 # compressor
    diameter[link_names.index(37)] = 0 # compressor
    diameter[link_names.index(38)] = 0 # compressor

    ratio = np.zeros(len(link_names))
    ratio[link_names.index(36)] = 3.921/3.353 # compressor
    ratio[link_names.index(37)] = 3.921/3.202 # compressor
    ratio[link_names.index(38)] = 3.921/2.556 # compressor
    
    E = 0.98528057165559679*np.ones(len(link_names))
    eps = 0.05*mm*np.ones(len(link_names))
    # set efficiency factor  and relative roughness to some non realistic value for compressors
    E[link_names.index(36)] = 0 # compressor
    E[link_names.index(37)] = 0 # compressor
    E[link_names.index(38)] = 0 # compressor
    eps[link_names.index(36)] = 0 # compressor
    eps[link_names.index(37)] = 0 # compressor
    eps[link_names.index(38)] = 0 # compressor
    
    links = pd.DataFrame(np.array([link_names,link_types,start_nodes,end_nodes,diameter,length,ratio,E,eps]).T,columns=['name','type','start_node','end_node','D','L','r','E','eps'])
    #links['name'] = links['name'].astype(int)
    return nodes, links

def create_network(carrier,nodes,links,hydr_eq='fa'):
    """Creates a network based on the nodes and links data
    
    Parameters
    ----------
    carrier : Carrier
        gas carrier
    nodes : pd DataFrame
        Node data
    links : pd DataFrame
        Link data
    
    Returns
    -------
    gas_net : GasNetwork
        The network
    """
    gas_net = GasNetwork('G22N')
    p_ref = 10*3.921*bar #[Pa]
    for ind_n in nodes.index:
        if nodes['type'][ind_n] == 0: # reference node
            gas_net.add_node(GasNode(nodes['name'][ind_n],node_type=nodes['type'][ind_n],p=p_ref))
        elif nodes['type'][ind_n] == 1: # load node
            gas_net.add_node(GasNode(nodes['name'][ind_n],node_type=nodes['type'][ind_n],q=nodes['q_inj'][ind_n]))

    network_nodes = list(gas_net.get_nodes())
    if hydr_eq=='fa':
        link_eq_form='q_of_dp'
    elif hydr_eq=='fb':
        link_eq_form='dp_of_q'
    else:
        raise ValueError("Enter valid hydr_eq. Either 'fa' or 'fb'.")
        
    for ind_l in links.index:
        start_node = network_nodes[nodes.index[nodes['name'] == str(float(links['start_node'][ind_l]))][0]]
        end_node = network_nodes[nodes.index[nodes['name'] == str(float(links['end_node'][ind_l]))][0]]
        gas_net.add_link(GasLink(links['name'][ind_l],start_node,end_node,link_type=links['type'][ind_l],link_params={'carrier':carrier, 'D':float(links['D'][ind_l]), 'L':float(links['L'][ind_l]),'E':float(links['E'][ind_l]),'r':float(links['r'][ind_l]),'eps':float(links['eps'][ind_l])},link_eq_form=link_eq_form))
    return gas_net

def initialize_network(network,carrier,nodes,links,scale_var=None,scale_var_params=None,formulation='full'):
    """Sets values of network variables to be used for initial guess.
    
    Parameters
    ----------
    network : GasNetwork
        The network to be initialized
    carrier : Carrier
        gas carrier
    nodes : pd DataFrame
        Node data
    links : pd DataFrame
        Link data
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    network.initialize()
    p_ref = list(network.get_nodes())[0].p
    p_init = p_ref*np.linspace(0.95,0.9,len(list(network.get_nodes()))-1) # initial pressure deviades 5% - 10% from reference pressure
    # make sure flow goes towards compressors
    p_init[4] = 0.9*p_init[1]
    p_init[12] = 0.9*p_init[11]
    p_init[15] = 0.9*p_init[14]
    q_init = 11.*carrier.rhon*np.ones(len(list(network.get_links())))#[kg/s] (The 11 is in [m^3/s])
    x_init = np.concatenate((q_init,p_init))
    network.update(x_init,formulation=formulation) # update without scaling, since x_init is unscaled
    x0 = network.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0
 
def solve_system(network,tol,max_iter,h,x0,scale_var=None,scale_var_params=None,D_F=np.array([]),D_x=np.array([]),det_tol=1e-8):
    """Solve the network using analytical Jacobian, with basic NR
    
    Parameters
    ----------
    network : GasNetwork
        The network to be initialized
    tol : float
        tolerance of NR
    max_iter : int
        maximum number of iterations of NR
    h : float
        step size used for FD
    x0 : np array
        inital guess
        
    Returns
    -------
    x_sol_FD : np array
        solution vector, using FD Jacobian
    iters_FD : int
        total number of iterations, using FD Jacobian
    err_vec_FD : np array
        vector with the error of NR for every iteration, using FD Jacobian
    x_sol : np array
        solution vector, using analytical Jacobian
    iters : int
        total number of iterations, using analytical Jacobian 
    err_vec : np array
        vector with the error of NR for every iteration, using analytical Jacobian
    """    
    form = 'full'
    network.update(x0,formulation=form,scale_var=scale_var,scale_var_params=scale_var_params)
    print("\nSolving system using analytical Jacobian")
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = network.solve_network(tol,max_iter,formulation=form,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x,det_tol=det_tol)
    
    return x_sol,iters,err_vec
    
def example_g22n_PU():
    """Check the solution of the network
    """
    #Given
    #carrier
    Z = 0.93323990580143557
    T = 283 #[K]
    S = 0.6
    Tn = 288 #[K]
    pn = 1.*bar #[Pa]
    R = 8.314413 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    
    #node and link data
    nodes,links = network_data(carrier)
    #scaling
    scale_var = 'per_unit'
    qbase = 100.*carrier.rhon
    pbase = 3.*10*bar
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    # create network
    gas_net = create_network(carrier,nodes,links)
    # initalize network
    x0 = initialize_network(gas_net,carrier,nodes,links,scale_var=scale_var,scale_var_params=scale_var_params)
    
    nlsys = NLS.NonLinearSystemGas(gas_net,formulation='full',scale_var=scale_var,scale_var_params=scale_var_params)
    F0 = nlsys.F(x0)
    J0 = nlsys.J(x0)
    print('Element in J0 (in absolute value): min = {}, max = {}'.format(np.min(np.abs(J0[J0.nonzero()])),np.max(np.abs(J0[J0.nonzero()]))))
    plt.figure('Scaled')
    plt.imshow(J0.todense())
    plt.colorbar()
    
    plt.figure('Scaled spy plot')
    plt.spy(J0.todense())
    
    #plt.show()
    
    #When
    h = 1e-6
    tol = 1e-6
    max_iter = 50
    x_sol,iters,err_vec = solve_system(gas_net,tol,max_iter,h,x0,scale_var=scale_var,scale_var_params=scale_var_params)
    # copied from page 171 and 172
    q_sol_expected = np.array([327560.4,290106,374331.6,-152582.4,108522,168465.6,177616.8,177616.8,97466.6,23976,179060.4,
                          121575.6,151279.2,183423.6,-52146,30434.4,183423.6,47257.2,150681.6,71841.6,-134341.2,240022.8,
                          240022.8,49392,-74106,-31986,-45918,-46407.6,-14068.8,35650.8,148032,-20419.2,38419.2,
                          -19022.4,-92444.4,177616.8,183423.6,240022.8])*carrier.rhon/hour # links flows in kg/s
    # correction
    q_sol_expected[[32,33]] = q_sol_expected[[33,32]]

    # copied from page 171 (1MPa = 10 bar)
    p_sol_expected = 10*np.array([3.738,3.772,3.722,3.353,3.610,3.577,3.567,3.560,3.627,3.635,3.629,3.202,3.589,3.448,3.559,3.587,3.568,3.570,3.578,3.607,3.644,3.921,3.921,3.921])*bar
    # correction
    p_sol_expected[14] = 10*2.556*bar # wrong pressure reported in Osiadacz?
    if scale_var == 'per_unit':
        p_sol_expected = p_sol_expected/pbase
        q_sol_expected = q_sol_expected/qbase
    x_sol_expected = np.concatenate((q_sol_expected,p_sol_expected))
    rel_tol = 5e-2
    #Then (The big tolerance is needed because the values reported in Osiadacz are probably not completely accurate)
    assert np.allclose(x_sol,x_sol_expected,atol=rel_tol*1e-1,rtol=rel_tol) 
    
def example_g22n_compare_J():
    """Check the unscaled Jacobian against the per unit Jacobian.
    """
    # Given
    #carrier
    Z = 0.93323990580143557
    T = 283 #[K]
    S = 0.6
    Tn = 288 #[K]
    pn = 1.*bar #[Pa]
    R = 8.314413 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)   
    
    #scaling
    scale_var = 'per_unit'
    qbase = 100.*carrier.rhon
    pbase = 3.*10*bar
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    
    #node and link data 
    nodes,links = network_data(carrier)
    # create network
    gas_net = create_network(carrier,nodes,links)

    # initalize network, unscaled
    x0 = initialize_network(gas_net,carrier,nodes,links)
    # initalize network, per unit
    x0_pu = initialize_network(gas_net,carrier,nodes,links,scale_var=scale_var,scale_var_params=scale_var_params)
    
    # When
    nlsys = NLS.NonLinearSystemGas(gas_net,formulation='full')
    J0 = nlsys.J(x0)
    
    nlsys_pu = NLS.NonLinearSystemGas(gas_net,formulation='full',scale_var=scale_var,scale_var_params=scale_var_params)
    J0_pu = nlsys_pu.J(x0_pu)

    # Then
    # function scale factors
    F_entries = gas_net.get_F_entries('full')
    Fb = np.zeros(len(x0))
    for ind_el,el in enumerate(F_entries):
        if isinstance(el,GasNode):
            Fb[ind_el] = qbase 
        elif isinstance(el,GasLink):
            if el.link_type == 'compressor':
                Fb[ind_el] = pbase
            elif el.link_type == 'pipe_high_pres_weymouth':
                Fb[ind_el] = qbase
    x_entries = gas_net.get_x_entries('full')
    xb = np.zeros(len(x0))
    for ind_el,el in enumerate(x_entries):
        if isinstance(el,GasNode):
            xb[ind_el] = pbase 
        elif isinstance(el,GasLink):
            xb[ind_el] = qbase
    
    assert np.allclose(np.diag(Fb).dot(J0_pu.todense()).dot(np.diag(1/xb)),J0.todense())

def comp_scaling():
    #scaling
    scale_var = 'per_unit'
    qbase = 100.*carrier.rhon
    pbase = 3.*10*bar
    scale_var_params = {'qbase':qbase,'pbase':pbase,'pgbase':pbase} #matrix scaling requires pgbase instead of pbase
    
    #node and link data 
    nodes,links = network_data(carrier)
    # create network
    gas_net = create_network(carrier,nodes,links)
    # create network with fb as link equation
    gas_net_fb = create_network(carrier,nodes,links,hydr_eq='fb')
    
    # initalize networks, unscaled
    x0 = initialize_network(gas_net,carrier,nodes,links)
    x0_fb = initialize_network(gas_net_fb,carrier,nodes,links)
    # initalize networks, per unit
    x0_pu = initialize_network(gas_net,carrier,nodes,links,scale_var=scale_var,scale_var_params=scale_var_params)
    x0_pu_fb = initialize_network(gas_net_fb,carrier,nodes,links,scale_var=scale_var,scale_var_params=scale_var_params)
    
    # plot network
    G = nx.DiGraph() # let networkx determine (reasonable) coordinates
    for node in gas_net.get_nodes():
        G.add_node(node.number,label=node.name)
    for link in gas_net.get_links():
        G.add_edge(link.start_node.number,link.end_node.number,label=link.name)
    pos = nx.spectral_layout(G) 
    pos[2]+=np.array([0.2,0]) # node with name 3
    pos[18][1] = pos[1][1] # node with name 22 
    for node in gas_net.get_nodes():
        node.x, node.y = pos[node.number]
    fig_top = plt.figure('Network topology')
    ax_top = fig_top.gca()
    gas_net.draw_network(ax_top)
    plt.axis('equal')
    plt.axis('off')
    
    # Jacobian matrices at initial guess
    nlsys = NLS.NonLinearSystemGas(gas_net,formulation='full')
    J0 = nlsys.J(x0)
    
    nlsys_pu = NLS.NonLinearSystemGas(gas_net,formulation='full',scale_var=scale_var,scale_var_params=scale_var_params)
    J0_pu = nlsys_pu.J(x0_pu)
    
    # plot the Jacobians
    F_entries = gas_net.get_F_entries('full')
    x_entries = gas_net.get_x_entries('full')
    F_node = 0
    F_link = 0
    for ind_el,el in enumerate(F_entries):
        if isinstance(el,GasNode):
            F_node += 1 
        elif isinstance(el,GasLink):
            F_link += 1 
    x_p = 0
    x_q = 0
    for ind_el,el in enumerate(x_entries):
        if isinstance(el,GasNode):
            x_p += 1
        elif isinstance(el,GasLink):
            x_q += 1
    
    # spy plots of Jacobian matrices (at inital guess)
    fig1 = plt.figure('Spy plots for $f^a$')
    plt.spy(J0,marker='s',alpha=0.5,label='unscaled')
    plt.spy(J0_pu,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5,label='per unit')
    plt.plot([0,len(x0)-0.5],[F_node-0.5,F_node-0.5],'--k')
    plt.plot([x_q-0.5,x_q-0.5],[0,len(x0)-0.5],'--k')
    plt.legend()
    
    print('Element in unscaled J0 using fa (in absolute value): min = {}, max = {}'.format(np.min(np.abs(J0[J0.nonzero()])),np.max(np.abs(J0[J0.nonzero()]))))
    print('Element in per unit scaled J0 using fa (in absolute value): min = {}, max = {}'.format(np.min(np.abs(J0_pu[J0_pu.nonzero()])),np.max(np.abs(J0_pu[J0_pu.nonzero()]))))  
    
    print('Determinant unscaled Jacobian using fa: |J(x0)| = {}'.format(np.linalg.det(J0.todense())))
    print('Determinant per unit Jacobian using fa: |J(x0)| = {}'.format(np.linalg.det(J0_pu.todense())))
    
    plt.figure('Unscaled (using fa)')
    ax_J_map = plt.gca()
    plt.imshow(J0.todense())
    nlsys.plot_J_overlay(ax_J_map)
    plt.colorbar()
    
    plt.figure('Per unit (using fa)')
    plt.imshow(J0_pu.todense())
    ax_J_map_pu = plt.gca()
    nlsys_pu.plot_J_overlay(ax_J_map_pu)
    plt.colorbar()
    
    # solve system
    h = 1e-6
    tol = 1e-6
    max_iter = 50
    # full p.u.
    x_sol_pu,iters_pu,err_vec_pu = solve_system(gas_net,tol,max_iter,h,x0_pu,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol_pu_fb,iters_pu_fb,err_vec_pu_fb = solve_system(gas_net_fb,tol,max_iter,h,x0_pu_fb,scale_var=scale_var,scale_var_params=scale_var_params)
    # full
    gas_net.update_full(x0,formulation='full')
    x_sol,iters,err_vec = solve_system(gas_net,tol,max_iter,h,x0,det_tol=1e-12)
    # full DF (scale function to ~1 to use in error. This should give the same error as p.u.)
    # function scale factors
    F_entries = gas_net.get_F_entries('full')
    Fb = np.zeros(len(x0))
    for ind_el,el in enumerate(F_entries):
        if isinstance(el,GasNode):
            Fb[ind_el] = qbase 
        elif isinstance(el,GasLink):
            if el.link_type == 'compressor':
                Fb[ind_el] = pbase
            elif el.link_type == 'pipe_high_pres_weymouth':
                Fb[ind_el] = qbase
    DF = np.diag(1/Fb)
    gas_net.update_full(x0,formulation='full')
    x_sol_DF,iters_DF,err_vec_DF = solve_system(gas_net,tol,max_iter,h,x0,D_F=DF,det_tol=1e-12)
    # full pu DF (scale the p.u. function (back) to the values / order it should have when no scaling is used.)
    gas_net.update_full(x0,formulation='full')
    DF_pu = np.diag(Fb)
    x_sol_pu_DF,iters_pu_DF,err_vec_pu_DF = solve_system(gas_net,tol,max_iter,h,x0_pu,scale_var=scale_var,scale_var_params=scale_var_params,D_F=DF_pu)
    x_sol_scaled,iters_scaled,err_vec_scaled = solve_system(gas_net,tol,max_iter,h,x0,scale_var='matrix',scale_var_params=scale_var_params,det_tol=1e-12)
    x_sol_scaled_fb,iters_scaled_fb,err_vec_scaled_fb = solve_system(gas_net_fb,tol,max_iter,h,x0_fb,scale_var='matrix',scale_var_params=scale_var_params,det_tol=1e-12)
    # convergence plots
    fig2 = plt.figure('Convergence plot different scaling')
    ax = fig2.gca()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    max_iter_used = np.max([iters_pu,iters,iters_DF,iters_pu_DF,iters_scaled,iters_pu_fb,iters_scaled_fb])
    ax.semilogy(np.asarray(range(0,iters_pu+1)),err_vec_pu,'*--',label='full p.u.')
    ax.semilogy(np.asarray(range(0,iters+1)),err_vec,'s--',label='full')
    ax.semilogy(np.asarray(range(0,iters_DF+1)),err_vec_DF,'d--',label='full $D_F$')
    ax.semilogy(np.asarray(range(0,iters_pu_DF+1)),err_vec_pu_DF,'o--',label='full p.u. $D_F$')
    ax.semilogy(np.asarray(range(0,iters_scaled+1)),err_vec_scaled,'.--',label='full scaled solver')
    ax.semilogy(np.asarray(range(0,iters_pu_fb+1)),err_vec_pu_fb,'+-',label='full p.u., fb')
    ax.semilogy(np.asarray(range(0,iters_scaled_fb+1)),err_vec_scaled_fb,'|-',label='full scaled solver, fb')
    layout_convergence(ax,tol,max_iter_used)

def comp_links():
    fig_conv = plt.figure('Convergence of NR, various pipes')
    ax_conv = fig_conv.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||F(x^k)||_2$')
    max_iter_used = 0
    
    fig_ord = plt.figure('Convergence order of NR, various pipes')
    ax_ord = fig_ord.gca()
    plt.xlabel(r'$||F(x^k)||_2 / ||F(x^0)||_2$')
    plt.ylabel(r'$||F(x^{k+1})||_2 / ||F(x^0)||_2$')
    # colors
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] # should be the default color cycle
    markers = {'fa':'*','fb':'.'}
    linestyles = {'fa':'--','fb':'-'}
    # pipes
    pipes = ['pipe_high_pres_weymouth','pipe_high_pres_pole','pipe_high_pres_blasius','pipe_high_pres_churchill']#,'pipe_high_pres_colebrook' #NB. Colebrook has a scipy.optimize.solve erin zitten!
    #scaling
    qbase = 100.*carrier.rhon
    pbase = 3.*10*bar
    scale_var_params = {'qbase':qbase,'pbase':pbase,'pgbase':pbase} #matrix scaling requires pgbase instead of pbase
    # solver info
    formulation='full'
    tol=1e-6
    max_iter=500
    for ind,pipe_type in enumerate(pipes):
        for hydr_eq in ['fa','fb']:
            print('Running load flow with pipes: {}, and hydr. eq {}\n'.format(pipe_type,hydr_eq))
            #node and link data 
            nodes,links = network_data(carrier,pipe_type=pipe_type)
            # create network
            gas_net = create_network(carrier,nodes,links,hydr_eq=hydr_eq)
            # initialze networks (unscaled, since scaling will be done with matrices in solver)
            x0 = initialize_network(gas_net,carrier,nodes,links,formulation=formulation)
        
            # solve systems
            gas_net.reset_network(x0,formulation=formulation)
            x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,formulation=formulation,solver='NR',scale_var='matrix',scale_var_params=scale_var_params)
            print('Error is {:.4e}, after {} iterations'.format(err_vec[-1],iters))
            # plot convergence and convergence order
            ax_conv.semilogy(np.asarray(range(0,iters+1)),err_vec,marker=markers.get(hydr_eq),linestyle=linestyles.get(hydr_eq),color=colors[ind],label=hydr_eq+', '+pipe_type)
            max_iter_used = max(max_iter_used,iters)
            ax_ord.loglog(err_vec[:-1]/err_vec[0],err_vec[1:]/err_vec[0],marker=markers.get(hydr_eq),linestyle=linestyles.get(hydr_eq),color=colors[ind],label=hydr_eq+', '+pipe_type)
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
    ax.loglog(x_slope,x_slope,linestyle=':',color='k',label='slope 1')
    ax.loglog(x_slope,y_slope2,linestyle='-.',color='k',label='slope 2')
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    
if __name__ == '__main__':    
    #comp_scaling()
    comp_links() 
    
    plt.show()
    
        
                             
                             
                             
