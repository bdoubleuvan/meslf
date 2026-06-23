"""Example of a gas network with 24 nodes. Called GasLib-24, taken from http://gaslib.zib.de/
"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink
from meslf.networks.carrier import Gas
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import os

mm = 1e-3 #[m]
km = 1e3 #[m]
bar = 1e5 #[Pa]
hour = 3600 #[s]
form = 'full'

def get_data(path):
    """Read the data from the xml data files
    Parameters
    ----------
    path : string
        Path to directory containing GasLib-24.net and GasLib-24.scn
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
    net_file = 'GasLib-24.net'
    scn_file = 'GasLib-24.scn'
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
                M_unit = source.find('{http://gaslib.zib.de/Gas}molarMass').attrib.get('unit')
                if M_unit == 'kg_per_kmol':
                    node_dict['molarMass'] = float(source.find('{http://gaslib.zib.de/Gas}molarMass').attrib.get('value'))*1e-3 #[kg/mol]
                else:
                    raise(ValueError('Encountered unknown molar mass unit!'))
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
                    link_dict['L'] = float(pipe.find('{http://gaslib.zib.de/Gas}length').attrib.get('value'))*mm
                elif L_unit == 'km':
                    link_dict['L'] = float(pipe.find('{http://gaslib.zib.de/Gas}length').attrib.get('value'))*km
                else:
                    raise(ValueError('Encountered unknown lenght unit!'))
                # convert diameter to meter
                D_unit = pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('unit')
                if D_unit == 'm':
                    link_dict['D'] = float(pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('value'))
                elif D_unit =='mm':
                    link_dict['D'] = float(pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('value'))*mm
                elif D_unit == 'km':
                    link_dict['D'] = float(pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('value'))*km
                else:
                    raise(ValueError('Encountered unknown diameter unit!'))
                yield link_dict
            for comp in connection.iter('{http://gaslib.zib.de/Gas}compressorStation'):
                # the GasLib-11.cs contains information about the compressors. However, this information is too detailed for the models I am currently using. So I don't consider it. I also make up some value for the ratio
                # compressor stations increase the gas pressure
                link_dict = get_link_data(comp)
                link_dict['type'] = 'compressor'
                link_dict['r'] = 1.2 #compressor ratio
                yield link_dict
            for control_valve in connection.iter('{http://gaslib.zib.de/Gas}controlValve'):
                # control valve stations decrease the gas pressure
                link_dict = get_link_data(control_valve)
                link_dict['type'] = 'compressor'
                link_dict['r'] = .8 #compressor ratio
                yield link_dict
            for resistor in connection.iter('{http://gaslib.zib.de/Gas}resistor'):
                link_dict = get_link_data(resistor)
                link_dict['type'] = 'resistor'
                # convert diameter to meter
                D_unit = pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('unit')
                if D_unit == 'm':
                    link_dict['D'] = float(pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('value'))
                elif D_unit =='mm':
                    link_dict['D'] = float(pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('value'))*mm
                elif D_unit == 'km':
                    link_dict['D'] = float(pipe.find('{http://gaslib.zib.de/Gas}diameter').attrib.get('value'))*km
                else:
                    raise(ValueError('Encountered unknown diameter unit!'))
                drag_fac = resistor.find('{http://gaslib.zib.de/Gas}dragFactor').attrib.get('value')
                link_dict['Kw'] = drag_fac
                yield link_dict
            for short_pipe in connection.iter('{http://gaslib.zib.de/Gas}shortPipe'):
                link_dict = get_link_data(short_pipe)
                link_dict['type'] = 'dummy'
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
    nodes_cols = ['id', 'type', 'p', 'q_inj', 'x', 'y', 'height_unit','height_value','pressureMin_unit','pressureMin_value','pressureMax_unit','pressureMax_value','T_celsius','normDensity','molarMass']
    nodes = pd.DataFrame(list(iter_nodes(network)),columns=nodes_cols)    

    links_cols = ['id','type','from','to','q','D', 'L','r','C','Kw','flowMin_unit','flowMin_value','flowMax_unit','flowMax_value']
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
            gas_net.add_node(GasNode(nodes['name'][ind_n],node_type=nodes['type'][ind_n],p=nodes['p'][ind_n],x=float(nodes['x'][ind_n]),y=float(nodes['y'][ind_n])))
        elif nodes['type'][ind_n] == 1: # load node
            gas_net.add_node(GasNode(nodes['name'][ind_n],node_type=nodes['type'][ind_n],q=nodes['q_inj'][ind_n]*carrier.rhon/hour,x=float(nodes['x'][ind_n]),y=float(nodes['y'][ind_n])))

    network_nodes = list(gas_net.get_nodes())
    for ind_l in links.index:
        start_node = network_nodes[nodes.index[nodes['name'] == links['start_node'][ind_l]][0]]
        end_node = network_nodes[nodes.index[nodes['name'] == links['end_node'][ind_l]][0]]
        if links['type'][ind_l] == 'pipe_high_pres_weymouth':
            gas_net.add_link(GasLink(links['name'][ind_l],start_node,end_node,link_type=links['type'][ind_l],link_params={'carrier':carrier, 'D':float(links['D'][ind_l]), 'L':float(links['L'][ind_l]),'E':1.}))
        elif links['type'][ind_l] == 'compressor':
            gas_net.add_link(GasLink(links['name'][ind_l],start_node,end_node,link_type = 'compressor',link_params = {'carrier':carrier, 'r':float(links['r'][ind_l])}))    
        elif links['type'][ind_l] == 'resistor':
            drag_fac = float(links['Kw'][ind_l])
            D = float(links['D'][ind_l])
            C = drag_fac*8/(carrier.rhon*np.pi**2*D**4)
            gas_net.add_link(GasLink(links['name'][ind_l],start_node,end_node,link_type = 'resistor',link_params = {'carrier':carrier, 'C':C}))  
        elif links['type'][ind_l] == 'dummy':
            gas_net.add_link(GasLink(links['name'][ind_l],start_node,end_node,link_type = 'dummy'))
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
    p_ref = nodes.loc[nodes['type'] == 0].iloc[0]['p']
    p_init = p_ref*np.linspace(0.95,0.9,len(list(network.get_nodes()))-1) # initial pressure deviades 5% - 10% from reference pressure
    # make sure flow goes towards compressors
    q_init = np.asarray(links['q'])*carrier.rhon/hour#[kg/s] 
    x_init = np.concatenate((q_init,p_init))
    network.update(x_init,formulation=form) # update without scaling, since x_init is unscaled
    x0 = network.set_x_init(formulation=form,scale_var=scale_var,scale_var_params=scale_var_params)
    return x0

def test_topology_g24n():
    #Given
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','GasLib_24')
    #data
    nodes, links, scen = get_data(path_to_data)
    #carrier
    Z = 1.
    T = (nodes['T_celsius']).mean() + 273 #[K]
    S = (nodes['normDensity']).mean()/1.225 # because air has density of 1.225 kg/m^3 at 'standard' conditions?
    Tn = 288 #[K] ?
    pn = 1.*bar #[Pa] ?
    R = 8.314413 #[J/molK]
    M = (nodes['molarMass']).mean() #[kg/mol] 
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)    
    
    # When
    gas_net = create_network(carrier,nodes,links,scen)
    number_of_links = len(gas_net.links)
    number_of_nodes = len(gas_net.nodes)
    
    # Then
    number_of_links_gaslib = 19 + 3 + 1 + 1 + 1 # pipes + compressors + control valves + resistors + short pipes
    number_of_nodes_gaslib = 24
    assert np.all([number_of_links,number_of_nodes] == [number_of_links_gaslib,number_of_nodes_gaslib])
    
if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(dir_path,'network_data','GasLib_24')
    # data
    nodes, links, scen = get_data(path_to_data)
    # carrier
    Z = 1.
    T = (nodes['T_celsius']).mean() + 273 #[K]
    S = (nodes['normDensity']).mean()/1.225 # because air has density of 1.225 kg/m^3 at 'standard conditions?
    Tn = 288 #[K] ?
    pn = 1.*bar #[Pa] ?
    R = 8.314413 #[J/molK]
    M = (nodes['molarMass']).mean() #[kg/mol] 
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    # create network
    gas_net = create_network(carrier,nodes,links,scen)
    
    # plot the gas_network
    plt.figure('Network topology')
    ax = plt.gca()
    gas_net.draw_network(ax)
    plt.axis('equal')
    plt.axis('off')
    plt.plot()
        
    # Solve network in diferrent ways, compare convergence
    tol = 1e-6
    max_iter = 50
    x0 = initialize_network(gas_net,carrier,nodes,links)
    # Scaling (in solver)
    F_entries = gas_net.get_F_entries(form)
    Fb = np.zeros(len(x0))
    for ind_el,el in enumerate(F_entries):
        if isinstance(el,GasNode):
            Fb[ind_el] = np.max([link.q for link in el.get_links() if isinstance(link,GasLink)])
        elif isinstance(el,GasLink):
            if el.link_type == 'compressor':
                Fb[ind_el] = el.end_node.p 
            else:
                Fb[ind_el] = el.q
    D_F = sps.diags(1/Fb)
    x_entries = gas_net.get_x_entries(form)
    xb = np.zeros(len(x0))
    for ind_el,el in enumerate(x_entries):
        if isinstance(el,GasNode):
            xb[ind_el] = el.p
        elif isinstance(el,GasLink):
            xb[ind_el] = el.q
    D_x = sps.diags(1/xb)
    # unscaled
    x_sol,iters,err_vec,_,_,_ = gas_net.solve_network(tol,max_iter,formulation=form,solver='NR')
    # scaled
    gas_net.reset_network(x0,formulation=form)
    x_sol_scaled,iters_scaled,err_vec_scaled,_,_,_ = gas_net.solve_network(tol,max_iter,formulation=form,solver='NR',D_F=D_F,D_x=D_x)
    # unscaled, scipy.optimize.root()
    gas_net.reset_network(x0,formulation=form)
    x_sol_root,iters_root,err_vec_root,_,_,_ = gas_net.solve_network(tol,max_iter,formulation=form,solver='root')
    # scaled, scipy.optimize.root()
    gas_net.reset_network(x0,formulation=form)
    x_sol_scaled_root,iters_scaled_root,err_vec_scaled_root,_,_,_ = gas_net.solve_network(tol,max_iter,formulation=form,solver='root',D_F=D_F,D_x=D_x)
    
    # plot convergence
    fig_conv = plt.figure('Convergence plot')
    ax_conv = fig_conv.gca()
    plt.xlabel('Iteration k')
    plt.ylabel('Error ($||F(x^k)||_2$ or $||D_F F(x^k)||_2$)')
    max_iter_used = np.max([iters,iters_scaled,iters_root,iters_scaled_root])
    ax_conv.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax_conv.semilogy(np.asarray(range(0,iters+1)),err_vec,'s-',label='unscaled')
    ax_conv.semilogy(np.asarray(range(0,iters_scaled+1)),err_vec_scaled,'o-',label='scaled')
    ax_conv.semilogy(np.asarray([0,iters_root]),err_vec_root,'.--',label='unscaled, root')
    ax_conv.semilogy(np.asarray([0,iters_scaled_root]),err_vec_scaled_root,'*--',label='scaled, root')
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    
    plt.show()
    
    
