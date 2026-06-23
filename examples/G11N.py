"""Example of a gas network with 11 nodes. Called GasLib-11, taken from http://gaslib.zib.de/
"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink
from meslf.networks.carrier import Gas
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import scipy.sparse as sps
import os.path
import matplotlib.pyplot as plt

bar = 1e5 #[Pa]
hour = 3600 #[s]
form = 'full'

def get_data(path):
    """Read the data from the xml data files
    Parameters
    ----------
    path : string
        Path to directory containing GasLib-11.net and GasLib-11.scn
    Returns
    -------
    nodes : pd DataFrame
        Node data
    links : pd DataFrame
        Link data
    scen : pd DataFram
        Scenario data
    """
    # read data from xml file
    net_file = 'GasLib-11.net'
    scn_file = 'GasLib-11.scn'
    net_path = os.path.join(path, net_file)
    scn_path = os.path.join(path, scn_file)
    net_data_tree = ET.parse(net_path)
    scn_data_tree = ET.parse(scn_path)
    network = net_data_tree.getroot()
    scenario = scn_data_tree.getroot()

    # get data of nodes:  
    def iter_nodes(etree):
        for node in etree.iter('{http://gaslib.zib.de/Framework}nodes'):
            for source in node.iter('{http://gaslib.zib.de/Gas}source'): # source nodes
                node_dict = get_node_data(source)
                node_dict['type'] = 1
                node_dict['T_celsius'] = float(source.find('{http://gaslib.zib.de/Gas}gasTemperature').attrib.get('value'))
                node_dict['normDensity'] = float(source.find('{http://gaslib.zib.de/Gas}normDensity').attrib.get('value'))            
                yield node_dict
            for sink in node.iter('{http://gaslib.zib.de/Gas}sink'): # sink nodes
                node_dict = get_node_data(sink)
                node_dict['type'] = 1
                yield node_dict
            for innode in node.iter('{http://gaslib.zib.de/Gas}innode'): # inner nodes (i.e. junctions?)
                node_dict = get_node_data(innode)
                node_dict['type'] = 1
                node_dict['q_inj'] = 0. #junction (so a load node with 0 injected flow)
                yield node_dict

    def get_node_data(node):
        node_attr = node.attrib
        node_dict = node_attr.copy()
        node_dict['height_unit'] = node.find('{http://gaslib.zib.de/Gas}height').attrib.get('unit')
        node_dict['height_value'] = node.find('{http://gaslib.zib.de/Gas}height').attrib.get('value')
        node_dict['pressureMin_unit'] = node.find('{http://gaslib.zib.de/Gas}pressureMin').attrib.get('unit')
        node_dict['pressureMin_value'] = node.find('{http://gaslib.zib.de/Gas}pressureMin').attrib.get('value')
        node_dict['pressureMax_unit'] = node.find('{http://gaslib.zib.de/Gas}pressureMax').attrib.get('unit')
        node_dict['pressureMax_value'] = node.find('{http://gaslib.zib.de/Gas}pressureMax').attrib.get('value')
        return node_dict     
    
    # get data of links
    def iter_links(etree):
        for connection in etree.iter('{http://gaslib.zib.de/Framework}connections'):
            for pipe in connection.iter('{http://gaslib.zib.de/Gas}pipe'):
                link_dict = get_link_data(pipe)
                link_dict['type'] = 'pipe_high_pres_weymouth'
                # convert length to meter
                L_unit = pipe.find('{http://gaslib.zib.de/Gas}length').attrib.get('unit')
                if L_unit == 'm':
                    link_dict['L'] = float(pipe.find('{http://gaslib.zib.de/Gas}length').attrib.get('value'))
                elif L_unit =='mm':
                    link_dict['L'] = float(pipe.find('{http://gaslib.zib.de/Gas}length').attrib.get('value'))*1e-3
                elif L_unit == 'km':
                    link_dict['L'] = float(pipe.find('{http://gaslib.zib.de/Gas}length').attrib.get('value'))*1e3
                else:
                    raise(ValueError('Encountered unknown lenght unit!'))
                # convert diameter to meter
                D_unit = pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('unit')
                if D_unit == 'm':
                    link_dict['D'] = float(pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('value'))
                elif D_unit =='mm':
                    link_dict['D'] = float(pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('value'))*1e-3
                elif D_unit == 'km':
                    link_dict['D'] = float(pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('value'))*1e3
                else:
                    raise(ValueError('Encountered unknown diameter unit!'))
                yield link_dict
            for comp in connection.iter('{http://gaslib.zib.de/Gas}compressorStation'):
                # the GasLib-11.cs contains information about the compressors. However, this information is too detailed for the models I am currently using. So I don't consider it. I also make up some value for the ratio
                link_dict = get_link_data(comp)
                link_dict['type'] = 'compressor'
                link_dict['r'] = 1.2 #compressor ratio
                yield link_dict
            for valve in connection.iter('{http://gaslib.zib.de/Gas}valve'):
                link_dict = get_link_data(valve)
                link_dict['type'] = 'resistor'
                link_dict['C'] = 10/np.sqrt(2.0592e-05)# constant (like a pipe constant). Based the pipe constant of a basic flow link with D = 0.5m and L = 55000m, with p in bar and q in m^3/h. I can't remember where the 10 came from
                yield link_dict

    def get_link_data(link):
        link_attr = link.attrib
        link_dict = link_attr.copy()
        link_dict['flowMin_unit'] = link.find('{http://gaslib.zib.de/Gas}flowMin').attrib.get('unit')
        link_dict['flowMin_value'] = link.find('{http://gaslib.zib.de/Gas}flowMin').attrib.get('value')
        link_dict['flowMax_unit'] = link.find('{http://gaslib.zib.de/Gas}flowMax').attrib.get('unit')
        link_dict['flowMax_value'] = link.find('{http://gaslib.zib.de/Gas}flowMax').attrib.get('value')
        q_unit = 1 #m^3/h
        if link_dict['flowMax_unit'] == '1000m_cube_per_hour':
            q_unit = 1e3
        else:
            raise(ValueError('Encountered unknown flow unit!'))
        link_dict['q'] = 1/2*q_unit*float(link_dict['flowMax_value'])
        return link_dict
            
    # write data to pandas dataframe
    nodes_cols = ['id', 'type', 'p', 'q_inj', 'x', 'y', 'height_unit','height_value','pressureMin_unit','pressureMin_value','pressureMax_unit','pressureMax_value','T_celsius','normDensity']
    nodes = pd.DataFrame(list(iter_nodes(network)),columns=nodes_cols)    

    links_cols = ['id','type','from','to','q','D', 'L','r','C','flowMin_unit','flowMin_value','flowMax_unit','flowMax_value']
    links = pd.DataFrame(list(iter_links(network)),columns=links_cols)  

    # get nodal flows (solution?) from scenario data file
    def iter_scenario(etree):
        for scenario in etree.iter('{http://gaslib.zib.de/Gas}scenario'):
            for node in scenario.iter('{http://gaslib.zib.de/Gas}node'):
                flow_sol_dict = get_scen_data(node)
                yield flow_sol_dict

    def get_scen_data(node):
        node_scn_attr = node.attrib
        flow_sol_dict = node_scn_attr.copy()
        # The assumption is that upper and lower bound have the same value!
        q_unit = node.find('{http://gaslib.zib.de/Gas}flow').attrib.get('unit')
        q_sign = 1
        if flow_sol_dict.get('type') == 'entry':
            q_sign = -1
        if q_unit == '1000m_cube_per_hour':
            flow_sol_dict['q_inj'] = q_sign*float(node.find('{http://gaslib.zib.de/Gas}flow').attrib.get('value'))*1e3 #m^3/h
        else:
            raise(ValueError('Encountered unknown flow unit!'))
        return flow_sol_dict

    scen_cols = ['id','q_inj']
    scen = pd.DataFrame(list(iter_scenario(scenario)), columns = scen_cols)

    # rename columns to match my chosen data framework
    nodes.rename(columns={'id' : 'name'}, inplace = True)
    links.rename(columns={'id' : 'name', 'from' : 'start_node', 'to' : 'end_node'}, inplace = True)  
    scen.rename(columns={'id' : 'name'}, inplace = True)
    return nodes, links, scen

def create_network(carrier,nodes,links,scen):
    """Creates a networks based on the node, link, and scenario data
    
    Parameters
    ----------
    carrier : Carrier
        gas carrier
    nodes : pd DataFrame
        Node data
    links : pd DataFrame
        Link data
    scen : pd DataFram
        Scenario data
        
    Returns
    -------
    gas_net : GasNetwork
        The network
    """
    gas_net = GasNetwork('G11N')
    p_ref = 50.*bar #[Pa]
    nodes.loc[nodes['name'] == 'entry01','type'] = 0
    nodes.loc[nodes['name'] == 'entry01','p'] = p_ref
    # set injected flows equal to scenario data
    for ind_s in scen.index:
        nodes.loc[nodes['name'] == scen.at[ind_s,'name'],'q_inj'] = scen.at[ind_s,'q_inj'] #[m^3/h]
    for ind_n in nodes.index:
        if nodes['type'][ind_n] == 0: # reference node
            gas_net.add_node(GasNode(nodes['name'][ind_n],node_type=nodes['type'][ind_n],p=nodes['p'][ind_n]))
        elif nodes['type'][ind_n] == 1: # load node
            gas_net.add_node(GasNode(nodes['name'][ind_n],node_type=nodes['type'][ind_n],q=nodes['q_inj'][ind_n]*carrier.rhon/hour))

    network_nodes = list(gas_net.get_nodes())
    for ind_l in links.index:
        start_node = network_nodes[nodes.index[nodes['name'] == links['start_node'][ind_l]][0]]
        end_node = network_nodes[nodes.index[nodes['name'] == links['end_node'][ind_l]][0]]
        if links['type'][ind_l] == 'pipe_high_pres_weymouth':
            gas_net.add_link(GasLink(links['name'][ind_l],start_node,end_node,link_type=links['type'][ind_l],link_params={'carrier':carrier, 'D':float(links['D'][ind_l]), 'L':float(links['L'][ind_l]),'E':1.}))
        elif links['type'][ind_l] == 'compressor':
            gas_net.add_link(GasLink(links['name'][ind_l],start_node,end_node,link_type = 'compressor',link_params = {'carrier':carrier, 'r':float(links['r'][ind_l])}))    
        elif links['type'][ind_l] == 'resistor':
            gas_net.add_link(GasLink(links['name'][ind_l],start_node,end_node,link_type = 'resistor',link_params = {'carrier':carrier, 'C':float(links['C'][ind_l])*carrier.rhon/(hour*np.sqrt(bar))}))    
    return gas_net

def initialize_network(network,carrier,nodes,links,scale_var=None,scale_var_params=None):
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
    p_ref = nodes['p'][nodes.index[nodes['type'] == 0]][0]
    p_init = p_ref*np.linspace(0.95,0.9,len(list(network.get_nodes()))-1) # initial pressure deviades 5% - 10% from reference pressure
    # make sure flow goes towards compressors
    q_init = np.asarray(links['q'])*carrier.rhon/hour#[kg/s] 
    x_init = np.concatenate((q_init,p_init))
    network.update(x_init,formulation=form) # update without scaling, since x_init is unscaled
    x0 = network.set_x_init(formulation=form,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def solve_system(network,tol,max_iter,h,x0,scale_var=None,scale_var_params=None,D_F=np.array([]),D_x=np.array([])):
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
    network.update(x0,formulation=form,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = network.solve_network(tol,max_iter,formulation=form,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params,D_F=D_F,D_x=D_x)
    
    return x_sol,iters,err_vec

def example_g11n_pu():
    """Check the solution of the network, using per unit scaling
    """
    #Given
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','GasLib_11')
    #data
    nodes, links, scen = get_data(path_to_data)
    #carrier
    Z = 1.
    T = (nodes['T_celsius']).mean() + 273 #[K]
    S = (nodes['normDensity']).mean()/1.225 # because air has density of 1.225 kg/m^3 at 'standard conditions?
    Tn = 288 #[K] ?
    pn = 1.*bar #[Pa] ?
    R = 8.314413 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    #scaling
    scale_var = 'per_unit'
    qbase = 160000.*carrier.rhon/hour#(links['q']).mean()*carrier.rhon/hour
    pbase = 50*bar
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    # create network
    gas_net = create_network(carrier,nodes,links,scen)
    # initalize network
    x0 = initialize_network(gas_net,carrier,nodes,links,scale_var=scale_var,scale_var_params=scale_var_params)
    
    #When
    h = 1e-6
    tol = 1e-6
    max_iter = 50
    x_sol,iters,err_vec = solve_system(gas_net,tol,max_iter,h,x0,scale_var=scale_var,scale_var_params=scale_var_params)
    q_sol_expected = np.array([160000., 157350.41707664, 140000., 100000., 57350.41707664, 142649.58292336, 120000., 79999.99999993, 160000., 200000., 2649.58292336])*carrier.rhon/hour # links flows in kg/s
    p_sol_expected = np.array([44.63261308,  55.71985744,  46.67894929,  55.16954591,  56.58981246, 53.55913569,  48.75791633,  52.11351764,  48.08405066, 57.70086079])*bar
    if scale_var == 'per_unit':
        p_sol_expected = p_sol_expected/pbase
        q_sol_expected = q_sol_expected/qbase
    x_sol_expected = np.concatenate((q_sol_expected,p_sol_expected))
    assert np.allclose(x_sol,x_sol_expected)
    
def example_g11n_scaled_solver():
    """Check the solution of the network, using the scaled solver
    """
    #Given
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','GasLib_11')
    #data
    nodes, links, scen = get_data(path_to_data)
    #carrier
    Z = 1.
    T = (nodes['T_celsius']).mean() + 273 #[K]
    S = (nodes['normDensity']).mean()/1.225 # because air has density of 1.225 kg/m^3 at 'standard conditions?
    Tn = 288 #[K] ?
    pn = 1.*bar #[Pa] ?
    R = 8.314413 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    # create network
    gas_net = create_network(carrier,nodes,links,scen)
    # initalize network
    x0 = initialize_network(gas_net,carrier,nodes,links)
    
    #When
    h = 1e-6
    tol = 1e-6
    max_iter = 50
    qbase = 160000.*carrier.rhon/hour#(links['q']).mean()*carrier.rhon/hour
    pbase = 50*bar
    scale_var = 'matrix'
    scale_var_params = {'qbase':qbase,'pgbase':pbase}
    x_sol,iters,err_vec = solve_system(gas_net,tol,max_iter,h,x0,scale_var=scale_var,scale_var_params=scale_var_params)
    q_sol_expected = np.array([160000., 157350.41707664, 140000., 100000., 57350.41707664, 142649.58292336, 120000., 79999.99999993, 160000., 200000., 2649.58292336])*carrier.rhon/hour # links flows in kg/s
    p_sol_expected = np.array([44.63261308,  55.71985744,  46.67894929,  55.16954591,  56.58981246, 53.55913569,  48.75791633,  52.11351764,  48.08405066, 57.70086079])*bar
    x_sol_expected = np.concatenate((q_sol_expected,p_sol_expected))
    assert np.allclose(x_sol,x_sol_expected)
    
if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','GasLib_11')

    #data
    nodes, links, scen = get_data(path_to_data)
    #carrier
    Z = 1.
    T = (nodes['T_celsius']).mean() + 273 #[K]
    S = (nodes['normDensity']).mean()/1.225 # because air has density of 1.225 kg/m^3 at 'standard conditions?
    Tn = 288 #[K] ?
    pn = 1.*bar #[Pa] ?
    R = 8.314413 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    # create network
    gas_net = create_network(carrier,nodes,links,scen)
    
    # compare convergence
    h = 1e-6
    tol = 1e-6
    max_iter = 50
    # unscaled
    x0 = initialize_network(gas_net,carrier,nodes,links)
    x_sol,iters,err_vec = solve_system(gas_net,tol,max_iter,h,x0)
    # per unit
    scale_var = 'per_unit'
    qbase = 160000.*carrier.rhon/hour
    pbase = 50*bar
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    gas_net.update_full(x0,form)
    x0_pu = initialize_network(gas_net,carrier,nodes,links,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol_pu,iters_pu,err_vec_pu = solve_system(gas_net,tol,max_iter,h,x0_pu,scale_var=scale_var,scale_var_params=scale_var_params)
    # different stopping criterium
    F_entries = gas_net.get_F_entries(form)
    Fb = np.zeros(len(x0))
    for ind_el,el in enumerate(F_entries):
        if isinstance(el,GasNode):
            Fb[ind_el] = qbase 
        elif isinstance(el,GasLink):
            if el.link_type == 'compressor':
                Fb[ind_el] = pbase
            else:
                Fb[ind_el] = qbase
    D_F = np.diag(1/Fb)
    gas_net.update_full(x0,form)
    x0_DF = initialize_network(gas_net,carrier,nodes,links)
    x_sol_DF,iters_DF,err_vec_DF = solve_system(gas_net,tol,max_iter,h,x0_DF,D_F=D_F)
    # fsolver, scaled 
    form = 'full'
    gas_net.reset_network(x0_pu,formulation=form,scale_var=scale_var,scale_var_params=scale_var_params)
    x_sol_pu_fsolver,iters_pu_fsolver,err_vec_pu_fsolver,_,_,_ = gas_net.solve_network(tol,max_iter,formulation=form,solver='fsolver',scale_var=scale_var,scale_var_params=scale_var_params)

    fig = plt.figure('Convergence plot')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    max_iter_used = np.max([iters,iters_pu,iters_DF,iters_pu_fsolver])
    ax.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax.semilogy(np.asarray(range(0,iters+1)),err_vec,'--.',label='unscaled')
    ax.semilogy(np.asarray(range(0,iters_pu+1)),err_vec_pu,'.--',label='per unit')
    ax.semilogy(np.asarray(range(0,iters_DF+1)),err_vec_DF,'.--',label='unscaled with $D_F$')
    ax.semilogy(np.asarray([0,iters_pu_fsolver]),err_vec_pu_fsolver,'.--',label='fsolver, per unit')
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    # compare with actual solution
    q_sol_expected = np.array([160000., 157350.41707664, 140000., 100000., 57350.41707664, 142649.58292336, 120000., 79999.99999993, 160000., 200000., 2649.58292336])*carrier.rhon/hour # links flows in kg/s
    p_sol_expected = np.array([44.63261308,  55.71985744,  46.67894929,  55.16954591,  56.58981246, 53.55913569,  48.75791633,  52.11351764,  48.08405066, 57.70086079])*bar
    x_sol_expected = np.concatenate((q_sol_expected,p_sol_expected))
    x_sol_pu_SI = np.zeros(len(x_sol_pu))
    x_entries = gas_net.get_x_entries(form)
    for ind_el,el in enumerate(x_entries):
        if isinstance(el,GasNode):
            x_sol_pu_SI[ind_el] = x_sol_pu[ind_el]*pbase 
        elif isinstance(el,GasLink):
            x_sol_pu_SI[ind_el] = x_sol_pu[ind_el]*qbase 
    fig.suptitle('Final error ||x-x_sol||: unscaled = {:.2e}, per unit = {:.2e}, unscaled with D_F = {:.2e}'.format(np.linalg.norm(x_sol-x_sol_expected),np.linalg.norm(x_sol_pu_SI-x_sol_expected),np.linalg.norm(x_sol_DF-x_sol_expected)))
    
    plt.show()
    
