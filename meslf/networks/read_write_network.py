"""Functions to read network from pd dataframe or to write a network to a pd dataframe"""
from meslf.networks.network import Network, Node, Link, HalfLink
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.utils.constants import MW
import pandas as pd
import numpy as np
import warnings

def to_pd_dataframes(network):
    """Writes the network to a pandas dataframe.
    
    Parameters
    ----------
    net : Network
        Network to be written to a pd dataframe
        
    Returns
    -------
    nodes : pd DataFrame
        Node data
    links : pd DataFrame
        Link data
    halflinks: pd DataFrame
        Halflink data
    """
    # determine and assign hierarchical levels on nodes, links, and half links
    nodes_seen = list()
    links_seen = list()
    halflinks_seen = list()
    hierarchical_levels = 1 # no subnetworks
    for net in network.get_networks(get_all=True):
        hierarchical_levels = max(hierarchical_levels,len(net.level))
        for ind_n,n in enumerate(net.get_nodes()):
            if not n in nodes_seen:
                nodes_seen.append(n)
                node_multiindex = net.level + [ind_n]
                n.level = node_multiindex
                n.level_depth = len(net.level)
        for ind_e,e in enumerate(net.get_links()):
            if not e in links_seen:
                links_seen.append(e)
                link_multiindex = net.level + [ind_e]
                e.level = link_multiindex
                e.level_depth = len(net.level)
        for ind_hl,hl in enumerate(net.get_half_links()):
            if not hl in halflinks_seen:
                halflinks_seen.append(hl)
                halflink_multiindex = net.level + [ind_hl]
                hl.level = halflink_multiindex
                hl.level_depth = len(net.level)
        
    # node dataframe
    node_names = list()
    node_types = list()
    node_carrier = list()
    node_data = list()
    node_unit_type = list()
    node_unit_params = list()
    x_coors = list()
    y_coors = list()
    node_mis = list()
    for ind,n in enumerate(network.get_nodes()):
        if not n in nodes_seen: # link not in any subnetwork
            nodes_seen.append(n)
            node_multiindex = network.level + [ind]
            n.level = node_multiindex
            n.level_depth = len(network.level)
        node_names.append(n.name)
        if isinstance(n,GasNode):
            node_carrier.append('g')
            node_data.append({'p':n.p,'p_unit':'Pa'})
            node_types.append(n.node_type)
            node_unit_type.append('')
            node_unit_params.append({})
        elif isinstance(n,ElectricalNode):
            node_carrier.append('e')
            node_data.append({'V':n.V,'delta':n.delta,'V_unit':'p.u.','delta_unit':'rad'})
            node_types.append(n.node_type)
            node_unit_type.append('')
            node_unit_params.append({})
        elif isinstance(n,HeatNode):
            node_carrier.append('h')
            node_data.append({'p':n.p,'p_unit':'Pa','Ts':n.Ts,'Ts_unit':'C','Tr':n.Tr,'Tr_unit':'C'})
            node_types.append(n.node_type)
            node_unit_type.append('')
            node_unit_params.append({})
        elif isinstance(n,HeterogeneousNode):
            node_carrier.append('c')
            node_data.append({})
            node_types.append(n.node_type)
            node_unit_type.append(n.unit_type)
            node_unit_params.append(n.unit_params)
        else:
            node_carrier.append('no_type')
            node_data.append({})
            node_types.append('no_type')
            node_unit_type.append('')
            node_unit_params.append({})
        x_coors.append(n.x)
        y_coors.append(n.y)
        # make sure all multiindices have the same number of levels
        if n.level_depth < hierarchical_levels:
            for lev in range(hierarchical_levels - n.level_depth):
                n.level.insert(-1,'no_net')
        n.level = tuple(n.level)
        node_mis.append(n.level)
    level_names = [None]
    for lev in range(hierarchical_levels):
        level_names.insert(0,'level ' + str(lev))
    nodes_index = pd.MultiIndex.from_tuples(node_mis,names=level_names)
    if np.any([params for params in node_unit_params]):
        if np.any([data for data in node_data]):
            nodes = pd.DataFrame(np.array([node_names,node_carrier,node_types,x_coors,y_coors,node_data,node_unit_type,node_unit_params]).T,columns=['name','carrier','type','x','y','data','unit type','parameters'],index=nodes_index)
        else:
            nodes = pd.DataFrame(np.array([node_names,node_carrier,node_types,x_coors,y_coors,node_unit_type,node_unit_params]).T,columns=['name','carrier','type','x','y'],index=nodes_index)
    else:
        if np.any([data for data in node_data]):
            nodes = pd.DataFrame(np.array([node_names,node_carrier,node_types,x_coors,y_coors,node_data]).T,columns=['name','carrier','type','x','y','data'],index=nodes_index)
        else:
            nodes = pd.DataFrame(np.array([node_names,node_carrier,node_types,x_coors,y_coors]).T,columns=['name','carrier','type','x','y'],index=nodes_index)
    
    # link dataframe
    link_names = list()
    link_carriers = list()
    start_nodes_link = list()
    end_nodes_link = list()
    link_types = list()
    link_params = list()
    link_data = list()
    link_mis = list()
    link_hydr_eq_form = list()
    link_bc_type = list()
    for ind_e,e in enumerate(network.get_links()):
        if not e in links_seen: # link not in any subnetwork
            links_seen.append(e)
            link_multiindex = network.level + [ind_e]
            e.level = link_multiindex
            e.level_depth = len(network.level)
        link_names.append(e.name)
        if isinstance(e,GasLink):
            link_carriers.append('g')
            link_types.append(e.link_type)
            link_params.append(e.link_params)
            link_hydr_eq_form.append(e.link_eq_form)
            link_data.append({'q':e.q,'q_unit':'kg/s'})
            link_bc_type.append(e.bc_type)
        elif isinstance(e,ElectricalLink):
            link_carriers.append('e')
            link_types.append(e.link_type)
            link_params.append(e.link_params)
            link_hydr_eq_form.append(None)
            link_data.append({'Pstart':e.Pstart,'Qstart':e.Qstart,'Pend':e.Pend,'Qend':e.Qend,'P_unit':'p.u.','Q_unit':'p.u.'})
            link_bc_type.append(e.bc_type)
        elif isinstance(e,HeatLink):
            link_carriers.append('h')
            link_types.append(e.link_type)
            link_params.append(e.link_params)
            link_hydr_eq_form.append(e.hydr_eq_form)
            link_data.append({'m':e.m,'m_unit':'kg/s','Tsstart':e.Tsstart,'Trstart':e.Trstart,'Tsend':e.Tsend,'Trend':e.Trend,'Ts_unit':'C','Tr_unit':'C','dphistart':e.dphistart,'dphiend':e.dphiend,'phi_unit':'W'})
            link_bc_type.append(e.bc_type)
        else:
            link_carriers.append('no_type')
            link_types.append(None)
            link_params.append(None)
            link_hydr_eq_form.append(None)
            link_data.append({})
            link_bc_type.append('no_type')
        start_nodes_link.append(e.start_node.level)
        end_nodes_link.append(e.end_node.level)
        if e.level_depth < hierarchical_levels:
            for lev in range(hierarchical_levels - e.level_depth):
                e.level.insert(-1,'no_net')
        e.level = tuple(e.level)
        link_mis.append(e.level)
    links_index = pd.MultiIndex.from_tuples(link_mis,names=level_names)    
    if np.any([data for data in link_data]):
        links = pd.DataFrame({'name':link_names,'carrier':link_carriers,'start_node':start_nodes_link,'end_node':end_nodes_link,'type':link_types,'parameters':link_params,'hydr_eq_form':link_hydr_eq_form,'bc_type':link_bc_type,'data':link_data},columns=['name','carrier','start_node','end_node','type','parameters','hydr_eq_form','bc_type','data'],index=links_index)
    else:
        links = pd.DataFrame({'name':link_names,'carrier':link_carriers,'start_node':start_nodes_link,'end_node':end_nodes_link},columns=['name','carrier','start_node','end_node'],index=links_index)
    
    # half link data
    halflink_names = list()
    halflink_carriers = list()
    start_nodes_halflink = list()
    halflink_types = list()
    halflink_params = list()
    halflink_data = list()
    halflink_mis = list()
    halflink_bc_type = list()
    for ind_hl,hl in enumerate(network.get_half_links()):
        if not hl in halflinks_seen: # link not in any subnetwork
            halflinks_seen.append(hl)
            halflink_multiindex = network.level + [ind_hl]
            hl.level = halflink_multiindex
            hl.level_depth = len(network.level)
        halflink_names.append(hl.name)
        if isinstance(hl,GasHalfLink):
            halflink_carriers.append('g')
            halflink_types.append(hl.link_type)
            halflink_params.append(hl.link_params)
            halflink_data.append({'q':hl.q,'q_unit':'kg/s'})
            halflink_bc_type.append(hl.bc_type)
        elif isinstance(hl,ElectricalHalfLink):
            halflink_carriers.append('e')
            halflink_types.append(hl.link_type)
            halflink_params.append(hl.link_params)
            halflink_data.append({'P':hl.P,'Q':hl.Q,'P_unit':'p.u.','Q_unit':'p.u.'})
            halflink_bc_type.append(hl.bc_type)
        elif isinstance(hl,HeatHalfLink):
            halflink_carriers.append('h')
            halflink_types.append(hl.link_type)
            halflink_params.append(hl.link_params)
            halflink_data.append({'m':hl.m,'m_unit':'kg/s','Ts':hl.Ts,'Tr':hl.Tr,'dT':hl.dT,'phi':hl.dphi,'T_unit':'C','phi_unit':'W'})
            halflink_bc_type.append(hl.bc_type)
        else:
            halflink_carriers.append('no_type')
            halflink_types.append(None)
            halflink_params.append(None)
            halflink_data.append({})
            halflink_bc_type.append('no_type')
        start_nodes_halflink.append(hl.start_node.level)
        if hl.level_depth < hierarchical_levels:
            for lev in range(hierarchical_levels - hl.level_depth):
                hl.level.insert(-1,'no_net')
        hl.level = tuple(hl.level)
        halflink_mis.append(hl.level)
    halflinks_index = pd.MultiIndex.from_tuples(halflink_mis,names=level_names)
    if np.any([data for data in halflink_data]):
        halflinks = pd.DataFrame({'name':halflink_names,'carrier':halflink_carriers,'start_node':start_nodes_halflink,'type':halflink_types,'parameters':halflink_params,'bc_type':halflink_bc_type,'data':halflink_data},columns=['name','carrier','start_node','type','parameters','bc_type','data'],index=halflinks_index)
    else:
        halflinks = pd.DataFrame({'name':halflink_names,'carrier':halflink_carriers,'start_node':start_nodes_halflink},columns=['name','carrier','start_node'],index=halflinks_index)
    return nodes, links, halflinks
    
def from_pd_dataframes(nodes,links,halflinks,flatten=True):
    """Create a network from data
        
    Parameters
    ----------
    nodes : pd DataFrame
        Node data, both topology and physical data
    links : pd DataFrame
        Link data, both topology and physical data
    halflinks: pd DataFrame
        Halflink data, both topology and physical data
    flatten : bool, optional
        If True, the hierarchical structure is not taken into account, i.e. a 'flat' network is created. If False, the hierarchical structure is kept. Default is True. NB: Keeping the hierarchical structure does not always work!
    
    Returns
    -------
    net : Network
        The created  network
    """
    # ignore hierarchial structure; create flat network
    if flatten: 
        net_name = nodes.index.levels[0][0]
        if np.any(nodes['carrier'] == 'no_type'):
            net = Network(net_name)
        elif np.all(nodes['carrier'] == 'g'):
            net = GasNetwork(net_name)
        elif np.all(nodes['carrier'] == 'e'):
            net = ElectricalNetwork(net_name)
        elif np.all(nodes['carrier'] == 'h'):
            net = HeatNetwork(net_name)
        elif np.any(nodes['carrier'] == 'c'):
            net = HeterogeneousNetwork(net_name)
        else: 
            raise ValueError('Invalid value encountered for carrier. Unclear what type of network this should be.')
        
        for ind_n in nodes.index:
            if nodes.loc[(ind_n),'carrier'] == 'no_type':
                node = Node(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'])
            elif nodes.loc[(ind_n),'carrier'] == 'g':
                node_data = nodes.loc[(ind_n),'data']
                node_p = node_data.get('p')
                if node_data.get('p_unit') == 'Pa':
                    pass
                elif node_data.get('p_unit') == 'bar':
                    node_p *= 1e5 #[Pa]
                elif node_data.get('p_unit') == 'mbar':
                    node_p *= 1e2 #[Pa]
                else:
                    raise ValueError("Invalid value encountered for 'p_unit'. It must be 'Pa', 'bar', or 'mbar', not {}".format(node_data.get('p_unit')))
                node = GasNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],p=node_p)
            elif nodes.loc[(ind_n),'carrier'] == 'e':
                node_data = nodes.loc[(ind_n),'data']
                node_V = node_data.get('V')
                node_delta = node_data.get('delta')
                if node_data.get('V_unit') == 'p.u.':
                    pass
                else:
                    raise ValueError("Invalid value encountered for 'V_unit'. It must be 'p.u.', not {}".format(node_data.get('V_unit')))
                if node_data.get('delta_unit') == 'rad':
                    pass
                elif node_data.get('delta_unit') == 'degree':
                    node_delta *= np.pi/180 #[rad]
                else:
                    raise ValueError("Invalid value encountered for 'delta_unit'. It must be 'p.u.' or 'degree' not {}".format(node_data.get('delta_unit')))
                node = ElectricalNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],V=node_V,delta=node_delta)
            elif nodes.loc[(ind_n),'carrier'] == 'h':
                node_data = nodes.loc[(ind_n),'data']
                node_p = node_data.get('p')
                node_Ts = node_data.get('Ts')
                node_Tr = node_data.get('Tr')
                if node_data.get('p_unit') == 'Pa':
                    pass
                elif node_data.get('p_unit') == 'bar':
                    node_p *= 1e5 #[Pa]
                elif node_data.get('p_unit') == 'mbar':
                    node_p *= 1e2 #[Pa]
                else:
                    raise ValueError("Invalid value encountered for 'p_unit'. It must be 'Pa', 'bar', or 'mbar', not {}".format(node_data.get('p_unit')))
                if node_data.get('Ts_unit') == 'C':
                    pass
                elif node_data.get('Ts_unit') == 'K':
                    node_Ts -= 273.15
                else:
                    raise ValueError("Invalid value encountered for 'Ts_unit'. It must be 'C' or 'K' not {}".format(node_data.get('Ts_unit')))
                if node_data.get('Tr_unit') == 'C':
                    pass
                elif node_data.get('Tr_unit') == 'K':
                    node_Tr -= 273.15
                else:
                    raise ValueError("Invalid value encountered for 'Ts_unit'. It must be 'C' or 'K' not {}".format(node_data.get('Tr_unit')))
                node = HeatNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],p=node_p,Ts=node_Ts,Tr=node_Tr)
            elif nodes.loc[(ind_n),'carrier'] == 'c': 
                node = HeterogeneousNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],unit_type=nodes.loc[(ind_n),'unit type'],unit_params=nodes.loc[(ind_n),'parameters'])
            else:
                raise ValueError("Invalid value encountered for carrier. It must be 'g','e','h','c','no type'.")  
            net.add_node(node)
        for ind_e in links.index:
            start_node_ind = links.loc[(ind_e),'start_node'] 
            start_node = net.nodes[nodes.index.get_loc(start_node_ind)]
            end_node_ind = links.loc[(ind_e),'end_node']
            end_node = net.nodes[nodes.index.get_loc(end_node_ind)]
            if links.loc[(ind_e),'carrier'] == 'no_type':
                link = Link(links.loc[(ind_e),'name'],start_node,end_node)
            elif links.loc[(ind_e),'carrier'] == 'g':
                link_data = links.loc[(ind_e),'data']
                link_q = link_data.get('q')
                if link_data.get('q_unit') == 'kg/s':
                    pass
                else:
                    raise ValueError("Invalid value encountered for 'q_unit'. It must be 'kg/s', not {}".format(node_data.get('p_unit')))
                link = GasLink(links.loc[(ind_e),'name'],start_node,end_node,q=link_q,link_type=links.loc[(ind_e),'type'],link_params=links.loc[(ind_e),'parameters'],link_eq_form=links.loc[(ind_e),'hydr_eq_form'])
            elif links.loc[(ind_e),'carrier'] == 'e':
                link_data = links.loc[(ind_e),'data']
                link_Pstart = link_data.get('Pstart')
                link_Qstart = link_data.get('Qstart')
                link_Pend = link_data.get('Pend')
                link_Qend = link_data.get('Qend')
                if link_data.get('P_unit') == 'p.u.':
                    pass
                else:
                    raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('P_unit')))
                if link_data.get('Q_unit') == 'p.u.':
                    pass
                else:
                    raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('Q_unit')))
                link = ElectricalLink(links.loc[(ind_e),'name'],start_node,end_node,Pstart=link_Pstart,Qstart=link_Qstart,Pend=link_Pend,Qend=link_Qend,link_type=links.loc[(ind_e),'type'],link_params=links.loc[(ind_e),'parameters'])
            elif links.loc[(ind_e),'carrier'] == 'h':
                link_data = links.loc[(ind_e),'data']
                link_m = link_data.get('m')
                link_Tsstart = link_data.get('Tsstart')
                link_Trstart = link_data.get('Trstart')
                link_Tsend = link_data.get('Tsend')
                link_Trend = link_data.get('Trend')
                link_dphistart = link_data.get('dphistart')
                link_dphiend = link_data.get('dphiend')
                if link_data.get('m_unit') == 'kg/s':
                    pass
                else:
                    raise ValueError("Invalid value encountered for 'm_unit'. It must be 'kg/s', not {}".format(link_data.get('m_unit')))
                if link_data.get('Ts_unit') == 'C':
                    pass
                elif link_data.get('Ts_unit') == 'K':
                    link_Tsstart -= 273.15
                    link_Tsend -= 273.15
                else:
                    raise ValueError("Invalid value encountered for 'Ts_unit'. It must be 'C' or 'K', not {}".format(link_data.get('Ts_unit')))
                if link_data.get('Tr_unit') == 'C':
                    pass
                elif link_data.get('Tr_unit') == 'K':
                    link_Trstart -= 273.15
                    link_Trend -= 273.15
                else:
                    raise ValueError("Invalid value encountered for 'Tr_unit'. It must be 'C' or 'K', not {}".format(link_data.get('Tr_unit')))
                if link_data.get('phi_unit') == 'W':
                    pass
                elif link_data.get('phi_unit') == 'MW':
                    link_dphistart *= MW
                    link_dphiend *= MW
                else:
                    raise ValueError("Invalid value encountered for 'phi_unit'. It must be 'W' or 'MW', not {}".format(link_data.get('phi_unit')))
                link = HeatLink(links.loc[(ind_e),'name'],start_node,end_node,m=link_m,Tsstart=link_Tsstart,Trstart=link_Trstart,Tsend=link_Tsend,Trend=link_Trend,dphistart=link_dphistart,dphiend=link_dphiend,link_type=links.loc[(ind_e),'type'],link_params=links.loc[(ind_e),'parameters'],hydr_eq_form=links.loc[(ind_e),'hydr_eq_form'],bc_type=links.loc[(ind_e),'bc_type'])
            elif links.loc[(ind_e),'carrier'] == 'c':
                raise ValueError("Links with 'c' as carrier are not implemented")
            else:
                raise ValueError("Invalid value encountered for carrier. It must be 'g','e','h','c','no type'.")
            net.add_link(link)
        for ind_hl in halflinks.index:
            start_node_ind = halflinks.loc[(ind_hl),'start_node']
            start_node = net.nodes[nodes.index.get_loc(start_node_ind)]
            if halflinks.loc[(ind_hl),'carrier'] == 'no_type':
                halflink = HalfLink(halflinks.loc[(ind_hl),'name'],start_node)
            elif halflinks.loc[(ind_hl),'carrier'] == 'g':
                halflink_data = halflinks.loc[(ind_hl),'data']
                halflink_q = halflink_data.get('q')
                if halflink_data.get('q_unit') == 'kg/s':
                    pass
                else:
                    raise ValueError("Invalid value encountered for 'q_unit'. It must be 'kg/s', not {}".format(node_data.get('p_unit')))
                halflink = GasHalfLink(halflinks.loc[(ind_hl),'name'],start_node,halflink_q,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
            elif halflinks.loc[(ind_hl),'carrier'] == 'e':
                halflink_data = halflinks.loc[(ind_hl),'data']
                halflink_P = halflink_data.get('P')
                halflink_Q = halflink_data.get('Q')
                if halflink_data.get('P_unit') == 'p.u.':
                    pass
                else:
                    raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('P_unit')))
                if halflink_data.get('Q_unit') == 'p.u.':
                    pass
                else:
                    raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('Q_unit')))
                halflink = ElectricalHalfLink(halflinks.loc[(ind_hl),'name'],start_node,P=halflink_P,Q=halflink_Q,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
            elif halflinks.loc[(ind_hl),'carrier'] == 'h':
                halflink_data = halflinks.loc[(ind_hl),'data']
                halflink_m = halflink_data.get('m')
                halflink_phi = halflink_data.get('phi')
                halflink_Ts = halflink_data.get('Ts')
                halflink_Tr = halflink_data.get('Tr')
                halflink_dT = halflink_data.get('dT')
                if halflink_data.get('m_unit') == 'kg/s':
                    pass
                else:
                    raise ValueError("Invalid value encountered for 'm_unit'. It must be 'kg/s', not {}".format(halflink_data.get('m_unit')))
                if halflink_data.get('phi_unit') == 'W':
                    pass
                elif halflink_data.get('phi_unit') == 'MW':
                    halflink_phi *= 1e6
                else:
                    raise ValueError("Invalid value encountered for 'phi_unit'. It must be 'W' or 'MW', not {}".format(halflink_data.get('phi_unit')))
                if halflink_data.get('T_unit') == 'C':
                    pass
                elif node_data.get('T_unit') == 'K':
                    halflink_Ts -= 273.15
                    halflink_Tr -= 273.15
                    halflink_dT -= 273.15
                else:
                    raise ValueError("Invalid value encountered for 'T_unit'. It must be 'C' or 'K' not {}".format(halflink_data.get('T_unit')))                    
                if isinstance(start_node,HeatNode) and start_node.node_type in [0,8]: # for source reference slack nodes and load reference slack nodes a half link is automatically created when the node is made. But it is not added to the network yet.
                    halflink = start_node.half_links[0]
                    halflink.name = halflinks.loc[(ind_hl),'name']
                    halflink.m = halflink_m
                    halflink.dphi = halflink_phi
                    halflink.Ts = halflink_Ts
                    halflink.Tr = halflink_Tr
                    halflink.dT = halflink_dT
                    halflink.set_type(halflinks.loc[(ind_hl),'type'],halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                else:
                    if start_node.half_links:
                        halflink = start_node.half_links[0]
                        halflink.name = halflinks.loc[(ind_hl),'name']
                        halflink.m = halflink_m
                        halflink.dphi = halflink_phi
                        halflink.Ts = halflink_Ts
                        halflink.Tr = halflink_Tr
                        halflink.dT = halflink_dT
                        halflink.set_type(halflinks.loc[(ind_hl),'type'],halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                    else:
                        halflink = HeatHalfLink(halflinks.loc[(ind_hl),'name'],start_node,m=halflink_m,dphi=halflink_phi,Ts=halflink_Ts,Tr=halflink_Tr,dT=halflink_dT,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
            elif halflinks.loc[(ind_hl),'carrier'] == 'c':
                raise ValueError("Coupling half links are not implemented")
            else:
                raise ValueError("Invalid value encountered for carrier. It must be 'g','e','h','c','no type'.")
            net.add_half_link(halflink)
    # keep hierarchical structure; create subnetworks            
    else: 
        warnings.warn("Formulation for hydraulic link equation for a GasLink or a HeatLink not implented!")
        # create networks per hierarchical level, and add nodes to networks
        nlevels = nodes.index.nlevels
        if nlevels <= 2: # no subnetworks
            net_name = nodes.index.levels[0][0]
            if np.any(nodes['carrier'] == 'no_type'):
                net = Network(net_name)
            elif np.all(nodes['carrier'] == 'g'):
                net = GasNetwork(net_name)
            elif np.all(nodes['carrier'] == 'e'):
                net = ElectricalNetwork(net_name)
            elif np.all(nodes['carrier'] == 'h'):
                net = HeatNetwork(net_name)
            elif np.any(nodes['carrier'] == 'c'):
                net = HeterogeneousNetwork(net_name)
            else: 
                raise ValueError('Invalid value encountered for carrier. Unclear what type of network this should be.')
            
            for ind_n in nodes.index:
                if nodes.loc[(ind_n),'carrier'] == 'no_type':
                    node = Node(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'])
                elif nodes.loc[(ind_n),'carrier'] == 'g':
                    node_data = nodes.loc[(ind_n),'data']
                    node_p = node_data.get('p')
                    if node_data.get('p_unit') == 'Pa':
                        pass
                    elif node_data.get('p_unit') == 'bar':
                        node_p *= 1e5 #[Pa]
                    elif node_data.get('p_unit') == 'mbar':
                        node_p *= 1e2 #[Pa]
                    else:
                        raise ValueError("Invalid value encountered for 'p_unit'. It must be 'Pa', 'bar', or 'mbar', not {}".format(node_data.get('p_unit')))
                    node = GasNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],p=node_p)
                elif nodes.loc[(ind_n),'carrier'] == 'e':
                    node_data = nodes.loc[(ind_n),'data']
                    node_V = node_data.get('V')
                    node_delta = node_data.get('delta')
                    if node_data.get('V_unit') == 'p.u.':
                        pass
                    else:
                        raise ValueError("Invalid value encountered for 'V_unit'. It must be 'p.u.', not {}".format(node_data.get('V_unit')))
                    if node_data.get('delta_unit') == 'rad':
                        pass
                    elif node_data.get('delta_unit') == 'degree':
                        node_delta *= np.pi/180 #[rad]
                    else:
                        raise ValueError("Invalid value encountered for 'delta_unit'. It must be 'p.u.' or 'degree' not {}".format(node_data.get('delta_unit')))
                    node = ElectricalNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],V=node_V,delta=node_delta)
                elif nodes.loc[(ind_n),'carrier'] == 'h':
                    node_data = nodes.loc[(ind_n),'data']
                    node_p = node_data.get('p')
                    node_Ts = node_data.get('Ts')
                    node_Tr = node_data.get('Tr')
                    if node_data.get('p_unit') == 'Pa':
                        pass
                    elif node_data.get('p_unit') == 'bar':
                        node_p *= 1e5 #[Pa]
                    elif node_data.get('p_unit') == 'mbar':
                        node_p *= 1e2 #[Pa]
                    else:
                        raise ValueError("Invalid value encountered for 'p_unit'. It must be 'Pa', 'bar', or 'mbar', not {}".format(node_data.get('p_unit')))
                    if node_data.get('Ts_unit') == 'C':
                        pass
                    elif node_data.get('Ts_unit') == 'K':
                        node_Ts -= 273.15
                    else:
                        raise ValueError("Invalid value encountered for 'Ts_unit'. It must be 'C' or 'K' not {}".format(node_data.get('Ts_unit')))
                    if node_data.get('Tr_unit') == 'C':
                        pass
                    elif node_data.get('Tr_unit') == 'K':
                        node_Tr -= 273.15
                    else:
                        raise ValueError("Invalid value encountered for 'Ts_unit'. It must be 'C' or 'K' not {}".format(node_data.get('Tr_unit')))
                    node = HeatNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],p=node_p,Ts=node_Ts,Tr=node_Tr)
                elif nodes.loc[(ind_n),'carrier'] == 'c':
                    node = HeterogeneousNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],unit_type=nodes.loc[(ind_n),'unit type'],unit_params=nodes.loc[(ind_n),'parameters'])
                else:
                    raise ValueError("Invalid value encountered for carrier. It must be 'g','e','h','c','no type'.")  
                net.add_node(node)
            for ind_e in links.index:
                start_node_ind = links.loc[(ind_e),'start_node'] # first part gives hierarchical index of network of the node, last element gives the index of the node within the network
                start_node = net.nodes[start_node_ind[-1]]
                end_node_ind = links.loc[(ind_e),'end_node']
                end_node = net.nodes[end_node_ind[-1]]
                if links.loc[(ind_e),'carrier'] == 'no_type':
                    link = Link(links.loc[(ind_e),'name'],start_node,end_node)
                elif links.loc[(ind_e),'carrier'] == 'g':
                    link_data = links.loc[(ind_e),'data']
                    link_q = link_data.get('q')
                    if link_data.get('q_unit') == 'kg/s':
                        pass
                    else:
                        raise ValueError("Invalid value encountered for 'q_unit'. It must be 'kg/s', not {}".format(node_data.get('p_unit')))
                    link = GasLink(links.loc[(ind_e),'name'],start_node,end_node,q=link_q,link_type=links.loc[(ind_e),'type'],link_params=links.loc[(ind_e),'parameters'])
                elif links.loc[(ind_e),'carrier'] == 'e':
                    link_data = links.loc[(ind_e),'data']
                    link_Pstart = link_data.get('Pstart')
                    link_Qstart = link_data.get('Qstart')
                    link_Pend = link_data.get('Pend')
                    link_Qend = link_data.get('Qend')
                    if link_data.get('P_unit') == 'p.u.':
                        pass
                    else:
                        raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('P_unit')))
                    if link_data.get('Q_unit') == 'p.u.':
                        pass
                    else:
                        raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('Q_unit')))
                    link = ElectricalLink(links.loc[(ind_e),'name'],start_node,end_node,Pstart=link_Pstart,Qstart=link_Qstart,Pend=link_Pend,Qend=link_Qend,link_type=links.loc[(ind_e),'type'],link_params=links.loc[(ind_e),'parameters'])
                elif links.loc[(ind_e),'carrier'] == 'h':
                    link_data = links.loc[(ind_e),'data']
                    link_m = link_data.get('m')
                    link_Tsstart = link_data.get('Tsstart')
                    link_Trstart = link_data.get('Trstart')
                    link_Tsend = link_data.get('Tsend')
                    link_Trend = link_data.get('Trend')
                    link_dphistart = link_data.get('dphistart')
                    link_dphiend = link_data.get('dphiend')
                    if link_data.get('m_unit') == 'kg/s':
                        pass
                    else:
                        raise ValueError("Invalid value encountered for 'm_unit'. It must be 'kg/s', not {}".format(link_data.get('m_unit')))
                    if link_data.get('Ts_unit') == 'C':
                        pass
                    elif link_data.get('Ts_unit') == 'K':
                        link_Tsstart -= 273.15
                        link_Tsend -= 273.15
                    else:
                        raise ValueError("Invalid value encountered for 'Ts_unit'. It must be 'C' or 'K', not {}".format(link_data.get('Ts_unit')))
                    if link_data.get('Tr_unit') == 'C':
                        pass
                    elif link_data.get('Tr_unit') == 'K':
                        link_Trstart -= 273.15
                        link_Trend -= 273.15
                    else:
                        raise ValueError("Invalid value encountered for 'P_unit'. It must be 'C' or 'K', not {}".format(link_data.get('Tr_unit')))
                    if link_data.get('phi_unit') == 'W':
                        pass
                    elif link_data.get('phi_unit') == 'MW':
                        link_dphistart *= MW
                        link_dphiend *= MW
                    else:
                        raise ValueError("Invalid value encountered for 'phi_unit'. It must be 'W' or 'MW', not {}".format(link_data.get('phi_unit')))
                    link = HeatLink(links.loc[(ind_e),'name'],start_node,end_node,m=link_m,Tsstart=link_Tsstart,Trstart=link_Trstart,Tsend=link_Tsend,Trend=link_Trend,dphistart=link_dphistart,dphiend=link_dphiend,link_type=links.loc[(ind_e),'type'],hydr_eq_form=links.loc[(ind_e),'hydr_eq_form'],link_params=links.loc[(ind_e),'parameters'],bc_type=links.loc[(ind_e),'bc_type'])
                elif links.loc[(ind_e),'carrier'] == 'c':
                    raise ValueError("Links with 'c' as carrier are not implemented")
                else:
                    raise ValueError("Invalid value encountered for carrier. It must be 'g','e','h','c','no type'.")
                net.add_link(link)
            for ind_hl in halflinks.index:
                start_node_ind = halflinks.loc[(ind_hl),'start_node'] # first part gives hierarchical index of network of the node, last element gives the index of the node within the network
                start_node = net.nodes[start_node_ind[-1]]
                if halflinks.loc[(ind_hl),'carrier'] == 'no_type':
                    halflink = HalfLink(halflinks.loc[(ind_hl),'name'],start_node)
                elif halflinks.loc[(ind_hl),'carrier'] == 'g':
                    halflink_data = halflinks.loc[(ind_hl),'data']
                    halflink_q = halflink_data.get('q')
                    if halflink_data.get('q_unit') == 'kg/s':
                        pass
                    else:
                        raise ValueError("Invalid value encountered for 'q_unit'. It must be 'kg/s', not {}".format(node_data.get('p_unit')))
                    halflink = GasHalfLink(halflinks.loc[(ind_hl),'name'],start_node,halflink_q,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                elif halflinks.loc[(ind_hl),'carrier'] == 'e':
                    halflink_data = halflinks.loc[(ind_hl),'data']
                    halflink_P = halflink_data.get('P')
                    halflink_Q = halflink_data.get('Q')
                    if halflink_data.get('P_unit') == 'p.u.':
                        pass
                    else:
                        raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('P_unit')))
                    if halflink_data.get('Q_unit') == 'p.u.':
                        pass
                    else:
                        raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('Q_unit')))
                    halflink = ElectricalHalfLink(halflinks.loc[(ind_hl),'name'],start_node,P=halflink_P,Q=halflink_Q,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                elif halflinks.loc[(ind_hl),'carrier'] == 'h':
                    halflink_data = halflinks.loc[(ind_hl),'data']
                    halflink_m = halflink_data.get('m')
                    halflink_phi = halflink_data.get('phi')
                    halflink_Ts = halflink_data.get('Ts')
                    halflink_Tr = halflink_data.get('Tr')
                    halflink_dT = halflink_data.get('dT')
                    if halflink_data.get('m_unit') == 'kg/s':
                        pass
                    else:
                        raise ValueError("Invalid value encountered for 'm_unit'. It must be 'kg/s', not {}".format(halflink_data.get('m_unit')))
                    if halflink_data.get('phi_unit') == 'W':
                        pass
                    elif halflink_data.get('phi_unit') == 'MW':
                        halflink_phi *= MW
                    else:
                        raise ValueError("Invalid value encountered for 'phi_unit'. It must be 'W' or 'MW', not {}".format(halflink_data.get('phi_unit')))
                    if halflink_data.get('T_unit') == 'C':
                        pass
                    elif node_data.get('T_unit') == 'K':
                        halflink_Ts -= 273.15
                        halflink_Tr -= 273.15
                        halflink_dT -= 273.15
                    else:
                        raise ValueError("Invalid value encountered for 'T_unit'. It must be 'C' or 'K' not {}".format(halflink_data.get('T_unit')))
                    if isinstance(start_node,HeatNode) and start_node.node_type in [0,8]: # for source reference slack nodes and load reference slack nodes a half link is automatically created when the node is made. But it is not added to the network yet.
                        halflink = start_node.half_links[0]
                        halflink.name = halflinks.loc[(ind_hl),'name']
                        halflink.m = halflink_m
                        halflink.dphi = halflink_phi
                        halflink.Ts = halflink_Ts
                        halflink.Tr = halflink_Tr
                        halflink.dT = halflink_dT
                        halflink.set_type(halflinks.loc[(ind_hl),'type'],halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                    else:
                        if start_node.half_links:
                            halflink = start_node.half_links[0]
                            halflink.name = halflinks.loc[(ind_hl),'name']
                            halflink.m = halflink_m
                            halflink.dphi = halflink_phi
                            halflink.Ts = halflink_Ts
                            halflink.Tr = halflink_Tr
                            halflink.dT = halflink_dT
                            halflink.set_type(halflinks.loc[(ind_hl),'type'],halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                        else:
                            halflink = HeatHalfLink(halflinks.loc[(ind_hl),'name'],start_node,m=halflink_m,dphi=halflink_phi,Ts=halflink_Ts,Tr=halflink_Tr,dT=halflink_dT,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                elif halflinks.loc[(ind_hl),'carrier'] == 'c':
                    raise ValueError("Coupling half links are not implemented")
                else:
                    raise ValueError("Invalid value encountered for carrier. It must be 'g','e','h','c','no type'.")
                net.add_half_link(halflink)
        else: # subnetworks
            net_dict = {}
            hierarchical_mi = nodes.index.droplevel(-1).unique()
            for ind in hierarchical_mi:
                if 'no_net' in ind:                 
                    ind_no_net = ind.index('no_net')
                    if ind_no_net == 1: # highest level network
                        net_name = ind[0]
                        if np.any(nodes['carrier'] == 'no_type'):
                            net = Network(net_name)
                        elif np.all(nodes['carrier'] == 'g'):
                            net = GasNetwork(net_name)
                        elif np.all(nodes['carrier'] == 'e'):
                            net = ElectricalNetwork(net_name)
                        elif np.all(nodes['carrier'] == 'h'):
                            net = HeatNetwork(net_name)
                        net_dict[ind] = net
                    else: # in between levels
                        subnet_name = ind[ind_no_net-1]
                        if np.any(nodes['carrier'] == 'no_type'):
                            subnet = Network(subnet_name)
                        elif np.all(nodes['carrier'] == 'g'):
                            subnet = GasNetwork(subnet_name)
                        elif np.all(nodes['carrier'] == 'e'):
                            subnet = ElectricalNetwork(net_name)
                        elif np.all(nodes['carrier'] == 'h'):
                            subnet = HeatNetwork(net_name)
                        net_dict[ind] = subnet
                else: # lowest level network
                    subnet_name = ind[-1]
                    if np.any(nodes['carrier'] == 'no_type'):
                        subnet = Network(subnet_name)
                    elif np.all(nodes['carrier'] == 'g'):
                        subnet = GasNetwork(subnet_name)
                    elif np.all(nodes['carrier'] == 'e'):
                        subnet = ElectricalNetwork(net_name)
                    elif np.all(nodes['carrier'] == 'h'):
                        subnet = HeatNetwork(net_name)
                    net_dict[ind] = subnet
            # add nodes to all (sub)networks
            for ind_n in nodes.index:
                if nodes.loc[(ind_n),'carrier'] == 'no_type':
                    node = Node(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'])
                elif nodes.loc[(ind_n),'carrier'] == 'g':
                    node_data = nodes.loc[(ind_n),'data']
                    node_p = node_data.get('p')
                    if node_data.get('p_unit') == 'Pa':
                        pass
                    elif node_data.get('p_unit') == 'bar':
                        node_p *= 1e5 #[Pa]
                    elif node_data.get('p_unit') == 'mbar':
                        node_p *= 1e2 #[Pa]
                    else:
                        raise ValueError("Invalid value encountered for 'p_unit'. It must be 'Pa', 'bar', or 'mbar', not {}".format(node_data.get('p_unit')))
                    node = GasNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],p=node_p)
                elif nodes.loc[(ind_n),'carrier'] == 'e':
                    node_data = nodes.loc[(ind_n),'data']
                    node_V = node_data.get('V')
                    node_delta = node_data.get('delta')
                    if node_data.get('V_unit') == 'p.u.':
                        pass
                    else:
                        raise ValueError("Invalid value encountered for 'V_unit'. It must be 'p.u.', not {}".format(node_data.get('V_unit')))
                    if node_data.get('delta_unit') == 'rad':
                        pass
                    elif node_data.get('delta_unit') == 'degree':
                        node_delta *= np.pi/180 #[rad]
                    else:
                        raise ValueError("Invalid value encountered for 'delta_unit'. It must be 'p.u.' or 'degree' not {}".format(node_data.get('delta_unit')))
                    node = ElectricalNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],V=node_V,delta=node_delta)
                elif nodes.loc[(ind_n),'carrier'] == 'h':
                    node_data = nodes.loc[(ind_n),'data']
                    node_p = node_data.get('p')
                    node_Ts = node_data.get('Ts')
                    node_Tr = node_data.get('Tr')
                    if node_data.get('p_unit') == 'Pa':
                        pass
                    elif node_data.get('p_unit') == 'bar':
                        node_p *= 1e5 #[Pa]
                    elif node_data.get('p_unit') == 'mbar':
                        node_p *= 1e2 #[Pa]
                    else:
                        raise ValueError("Invalid value encountered for 'p_unit'. It must be 'Pa', 'bar', or 'mbar', not {}".format(node_data.get('p_unit')))
                    if node_data.get('Ts_unit') == 'C':
                        pass
                    elif node_data.get('Ts_unit') == 'K':
                        node_Ts -= 273.15
                    else:
                        raise ValueError("Invalid value encountered for 'Ts_unit'. It must be 'C' or 'K' not {}".format(node_data.get('Ts_unit')))
                    if node_data.get('Tr_unit') == 'C':
                        pass
                    elif node_data.get('Tr_unit') == 'K':
                        node_Tr -= 273.15
                    else:
                        raise ValueError("Invalid value encountered for 'Ts_unit'. It must be 'C' or 'K' not {}".format(node_data.get('Tr_unit')))
                    node = HeatNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],p=node_p,Ts=node_Ts,Tr=node_Tr)
                elif nodes.loc[(ind_n),'carrier'] == 'c':
                    node = HeterogeneousNode(nodes.loc[(ind_n),'name'],x=nodes.loc[(ind_n),'x'],y=nodes.loc[(ind_n),'y'],node_type=nodes.loc[(ind_n),'type'],unit_type=nodes.loc[(ind_n),'unit type'],unit_params=nodes.loc[(ind_n),'parameters'])
                else:
                    raise ValueError("Invalid value encountered for carrier. It must be 'g','e','h','c','no type'.")  
                net_dict.get(ind_n[:-1]).add_node(node)     
        # add links and halflinks to (sub)networks, and add subnetworks to higher level networks
        if nlevels > 2: # subnetworks
            number_subnets_dict = {}
            for lev in range(0,nlevels-1): # determine amount of networks per level
                number_subnets_dict['level '+str(lev)] = len([subnet_name for subnet_name in nodes.index.get_level_values('level '+str(lev)).unique() if not subnet_name == 'no_net'])
            for lev in range(0,nlevels-1): # starts at lowest level
                if lev < nlevels-2: # subnetwork
                    subnet_ind = 0
                    for i in range(nlevels-2,lev,-1):
                        subnet_ind += number_subnets_dict.get('level '+str(i))
                    for subnet_mi in hierarchical_mi[subnet_ind:subnet_ind+number_subnets_dict.get('level '+str(lev))]:
                        higher_net_mi = list(subnet_mi)
                        if 'no_net' in subnet_mi:
                            higher_net_mi[subnet_mi.index('no_net')-1]='no_net'
                        else:
                            higher_net_mi[-1]='no_net'
                        subnet = net_dict.get(subnet_mi)
                        # add links to subnetworks
                        if subnet_mi in links.index: # check if this subnetwork has links
                            for ind_e in links.loc[(subnet_mi)].index:
                                ind_e = tuple(list(subnet_mi)+[ind_e])
                                start_node_ind = links.loc[(ind_e),'start_node'] # first part gives hierarchical index of network of the node, last element gives the index of the node within the network
                                start_node = net_dict.get(start_node_ind[:-1]).nodes[start_node_ind[-1]]
                                end_node_ind = links.loc[(ind_e),'end_node']
                                end_node = net_dict.get(end_node_ind[:-1]).nodes[end_node_ind[-1]]
                                if links.loc[(ind_e),'carrier'] == 'no_type':
                                    link = Link(links.loc[(ind_e),'name'],start_node,end_node)
                                elif links.loc[(ind_e),'carrier'] == 'g':
                                    link_data = links.loc[(ind_e),'data']
                                    link_q = link_data.get('q')
                                    if link_data.get('q_unit') == 'kg/s':
                                        pass
                                    else:
                                        raise ValueError("Invalid value encountered for 'q_unit'. It must be 'kg/s', not {}".format(node_data.get('p_unit')))
                                    link = GasLink(links.loc[(ind_e),'name'],start_node,end_node,q=link_q,link_type=links.loc[(ind_e),'type'],link_params=links.loc[(ind_e),'parameters'])
                                elif links.loc[(ind_e),'carrier'] == 'e':
                                    link_data = links.loc[(ind_e),'data']
                                    link_Pstart = link_data.get('Pstart')
                                    link_Qstart = link_data.get('Qstart')
                                    link_Pend = link_data.get('Pend')
                                    link_Qend = link_data.get('Qend')
                                    if link_data.get('P_unit') == 'p.u.':
                                        pass
                                    else:
                                        raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('P_unit')))
                                    if link_data.get('Q_unit') == 'p.u.':
                                        pass
                                    else:
                                        raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('Q_unit')))
                                    link = ElectricalLink(links.loc[(ind_e),'name'],start_node,end_node,Pstart=link_Pstart,Qstart=link_Qstart,Pend=link_Pend,Qend=link_Qend,link_type=links.loc[(ind_e),'type'],link_params=links.loc[(ind_e),'parameters'])
                                elif links.loc[(ind_e),'carrier'] == 'h':
                                    link_data = links.loc[(ind_e),'data']
                                    link_m = link_data.get('m')
                                    link_Tsstart = link_data.get('Tsstart')
                                    link_Trstart = link_data.get('Trstart')
                                    link_Tsend = link_data.get('Tsend')
                                    link_Trend = link_data.get('Trend')
                                    link_dphistart = link_data.get('dphistart')
                                    link_dphiend = link_data.get('dphiend')
                                    if link_data.get('m_unit') == 'kg/s':
                                        pass
                                    else:
                                        raise ValueError("Invalid value encountered for 'm_unit'. It must be 'kg/s', not {}".format(link_data.get('m_unit')))
                                    if link_data.get('Ts_unit') == 'C':
                                        pass
                                    elif link_data.get('Ts_unit') == 'K':
                                        link_Tsstart -= 273.15
                                        link_Tsend -= 273.15
                                    else:
                                        raise ValueError("Invalid value encountered for 'Ts_unit'. It must be 'C' or 'K', not {}".format(link_data.get('Ts_unit')))
                                    if link_data.get('Tr_unit') == 'C':
                                        pass
                                    elif link_data.get('Tr_unit') == 'K':
                                        link_Trstart -= 273.15
                                        link_Trend -= 273.15
                                    else:
                                        raise ValueError("Invalid value encountered for 'P_unit'. It must be 'C' or 'K', not {}".format(link_data.get('Tr_unit')))
                                    if link_data.get('phi_unit') == 'W':
                                        pass
                                    elif link_data.get('phi_unit') == 'MW':
                                        link_dphistart *= MW
                                        link_dphiend *= MW
                                    else:
                                        raise ValueError("Invalid value encountered for 'phi_unit'. It must be 'W' or 'MW', not {}".format(link_data.get('phi_unit')))
                                    link = HeatLink(links.loc[(ind_e),'name'],start_node,end_node,m=link_m,Tsstart=link_Tsstart,Trstart=link_Trstart,Tsend=link_Tsend,Trend=link_Trend,dphistart=link_dphistart,dphiend=link_dphiend,link_type=links.loc[(ind_e),'type'],link_params=links.loc[(ind_e),'parameters'],hydr_eq_form=links.loc[(ind_e),'hydr_eq_form'],bc_type=links.loc[(ind_e),'bc_type'])
                                elif links.loc[(ind_e),'carrier'] == 'c':
                                    raise ValueError("Links with 'c' as carrier are not implemented")
                                else:
                                    raise ValueError("Invalid value encountered for carrier. It must be 'g','e','h','c','no type'.")
                                subnet.add_link(link,position=ind_e[-1])
                        # add halflinks to subnetworks
                        if subnet_mi in halflinks.index: # check if this subnetwork has halflinks
                            for ind_hl in halflinks.loc[(subnet_mi)].index:
                                ind_hl = tuple(list(subnet_mi)+[ind_hl])
                                start_node_ind = halflinks.loc[(ind_hl),'start_node'] # first part gives hierarchical index of network of the node, last element gives the index of the node within the network
                                start_node = net_dict.get(start_node_ind[:-1]).nodes[start_node_ind[-1]]
                                if halflinks.loc[(ind_hl),'carrier'] == 'no_type':
                                    halflink = HalfLink(halflinks.loc[(ind_hl),'name'],start_node)
                                elif halflinks.loc[(ind_hl),'carrier'] == 'g':
                                    halflink_data = halflinks.loc[(ind_hl),'data']
                                    halflink_q = halflink_data.get('q')
                                    if halflink_data.get('q_unit') == 'kg/s':
                                        pass
                                    else:
                                        raise ValueError("Invalid value encountered for 'q_unit'. It must be 'kg/s', not {}".format(node_data.get('p_unit')))
                                    halflink = GasHalfLink(halflinks.loc[(ind_hl),'name'],start_node,halflink_q,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                                elif halflinks.loc[(ind_hl),'carrier'] == 'e':
                                    halflink_data = halflinks.loc[(ind_hl),'data']
                                    halflink_P = halflink_data.get('P')
                                    halflink_Q = halflink_data.get('Q')
                                    if halflink_data.get('P_unit') == 'p.u.':
                                        pass
                                    else:
                                        raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('P_unit')))
                                    if halflink_data.get('Q_unit') == 'p.u.':
                                        pass
                                    else:
                                        raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('Q_unit')))
                                    halflink = ElectricalHalfLink(halflinks.loc[(ind_hl),'name'],start_node,P=halflink_P,Q=halflink_Q,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                                elif halflinks.loc[(ind_hl),'carrier'] == 'h':
                                    halflink_data = halflinks.loc[(ind_hl),'data']
                                    halflink_m = halflink_data.get('m')
                                    halflink_phi = halflink_data.get('phi')
                                    halflink_Ts = halflink_data.get('Ts')
                                    halflink_Tr = halflink_data.get('Tr')
                                    halflink_dT = halflink_data.get('dT')
                                    if halflink_data.get('m_unit') == 'kg/s':
                                        pass
                                    else:
                                        raise ValueError("Invalid value encountered for 'm_unit'. It must be 'kg/s', not {}".format(halflink_data.get('m_unit')))
                                    if halflink_data.get('phi_unit') == 'W':
                                        pass
                                    elif halflink_data.get('phi_unit') == 'MW':
                                        halflink_phi *= MW
                                    else:
                                        raise ValueError("Invalid value encountered for 'phi_unit'. It must be 'W' or 'MW', not {}".format(halflink_data.get('phi_unit')))
                                    if halflink_data.get('T_unit') == 'C':
                                        pass
                                    elif node_data.get('T_unit') == 'K':
                                        halflink_Ts -= 273.15
                                        halflink_Tr -= 273.15
                                        halflink_dT -= 273.15
                                    else:
                                        raise ValueError("Invalid value encountered for 'T_unit'. It must be 'C' or 'K' not {}".format(halflink_data.get('T_unit')))       
                                    if isinstance(start_node,HeatNode) and start_node.node_type in [0,8]: # for source reference slack nodes and load reference slack nodes a half link is automatically created when the node is made. But it is not added to the network yet.
                                        halflink = start_node.half_links[0]
                                        halflink.name = halflinks.loc[(ind_hl),'name']
                                        halflink.m = halflink_m
                                        halflink.dphi = halflink_phi
                                        halflink.Ts = halflink_Ts
                                        halflink.Tr = halflink_Tr
                                        halflink.dT = halflink_dT
                                        halflink.set_type(halflinks.loc[(ind_hl),'type'],halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                                    else:
                                        if start_node.half_links:
                                            halflink = start_node.half_links[0]
                                            halflink.name = halflinks.loc[(ind_hl),'name']
                                            halflink.m = halflink_m
                                            halflink.dphi = halflink_phi
                                            halflink.Ts = halflink_Ts
                                            halflink.Tr = halflink_Tr
                                            halflink.dT = halflink_dT
                                            halflink.set_type(halflinks.loc[(ind_hl),'type'],halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                                        else:
                                            halflink = HeatHalfLink(halflinks.loc[(ind_hl),'name'],start_node,m=halflink_m,phi=halflink_phi,To=halflink_To,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                                elif halflinks.loc[(ind_hl),'carrier'] == 'c':
                                    raise ValueError("Coupling half links are not implemented")
                                else:
                                    raise ValueError("Invalid value encountered for carrier. It must be 'g','e','h','c','no type'.")
                                subnet.add_half_link(halflink,position=ind_hl[-1])
                        net_dict.get(tuple(higher_net_mi)).add_network(subnet)
                else: #no subnetwork
                    net_mi = hierarchical_mi[0]
                    if net_mi in links.index: # check if the network has links
                        # add links to network
                        for ind_e in links.loc[net_mi].index:
                            ind_e = tuple(list(net_mi)+[ind_e])
                            start_node_ind = links.loc[(ind_e),'start_node'] # first part gives hierarchical index of network of the node, last element gives the index of the node within the network
                            start_node = net_dict.get(start_node_ind[:-1]).nodes[start_node_ind[-1]]
                            end_node_ind = links.loc[(ind_e),'end_node']
                            end_node = net_dict.get(end_node_ind[:-1]).nodes[end_node_ind[-1]]
                            if links.loc[(ind_e),'carrier'] == 'no_type':
                                link = Link(links.loc[(ind_e),'name'],start_node,end_node)
                            elif links.loc[(ind_e),'carrier'] == 'g':
                                link_data = links.loc[(ind_e),'data']
                                link_q = link_data.get('q')
                                if link_data.get('q_unit') == 'kg/s':
                                    pass
                                else:
                                    raise ValueError("Invalid value encountered for 'q_unit'. It must be 'kg/s', not {}".format(node_data.get('p_unit')))
                                link = GasLink(links.loc[(ind_e),'name'],start_node,end_node,q=link_q,link_type=links.loc[(ind_e),'type'],link_params=links.loc[(ind_e),'parameters'])
                            elif links.loc[(ind_e),'carrier'] == 'e':
                                link_data = links.loc[(ind_e),'data']
                                link_Pstart = link_data.get('Pstart')
                                link_Qstart = link_data.get('Qstart')
                                link_Pend = link_data.get('Pend')
                                link_Qend = link_data.get('Qend')
                                if link_data.get('P_unit') == 'p.u.':
                                    pass
                                else:
                                    raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('P_unit')))
                                if link_data.get('Q_unit') == 'p.u.':
                                    pass
                                else:
                                    raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('Q_unit')))
                                link = ElectricalLink(links.loc[(ind_e),'name'],start_node,end_node,Pstart=link_Pstart,Qstart=link_Qstart,Pend=link_Pend,Qend=link_Qend,link_type=links.loc[(ind_e),'type'],link_params=links.loc[(ind_e),'parameters'])
                            elif links.loc[(ind_e),'carrier'] == 'h':
                                link_data = links.loc[(ind_e),'data']
                                link_m = link_data.get('m')
                                link_Tsstart = link_data.get('Tsstart')
                                link_Trstart = link_data.get('Trstart')
                                link_Tsend = link_data.get('Tsend')
                                link_Trend = link_data.get('Trend')
                                if link_data.get('m_unit') == 'kg/s':
                                    pass
                                else:
                                    raise ValueError("Invalid value encountered for 'm_unit'. It must be 'kg/s', not {}".format(link_data.get('m_unit')))
                                if link_data.get('Ts_unit') == 'C':
                                    pass
                                elif link_data.get('Ts_unit') == 'K':
                                    link_Tsstart -= 273.15
                                    link_Tsend -= 273.15
                                else:
                                    raise ValueError("Invalid value encountered for 'Ts_unit'. It must be 'C' or 'K', not {}".format(link_data.get('Ts_unit')))
                                if link_data.get('Tr_unit') == 'C':
                                    pass
                                elif link_data.get('Tr_unit') == 'K':
                                    link_Trstart -= 273.15
                                    link_Trend -= 273.15
                                else:
                                    raise ValueError("Invalid value encountered for 'P_unit'. It must be 'C' or 'K', not {}".format(link_data.get('Tr_unit')))
                                link = HeatLink(links.loc[(ind_e),'name'],start_node,end_node,m=link_m,Tsstart=link_Tsstart,Trstart=link_Trstart,Tsend=link_Tsend,Trend=link_Trend,link_type=links.loc[(ind_e),'type'],link_params=links.loc[(ind_e),'parameters'])
                            elif links.loc[(ind_e),'carrier'] == 'c':
                                raise ValueError("Links with 'c' as carrier are not implemented")
                            else:
                                raise ValueError("Invalid value encountered for carrier. It must be 'g','e','h','c','no type'.")
                            net.add_link(link,position=ind_e[-1])
                    if net_mi in halflinks.index: # check if the network has halflinks
                        # add halflinks to subnetworks
                        for ind_hl in halflinks.loc[(net_mi)].index:
                            ind_hl = tuple(list(net_mi)+[ind_hl])
                            start_node_ind = halflinks.loc[(ind_hl),'start_node'] # first part gives hierarchical index of network of the node, last element gives the index of the node within the network
                            start_node = net_dict.get(start_node_ind[:-1]).nodes[start_node_ind[-1]]
                            if halflinks.loc[(ind_hl),'carrier'] == 'no_type':
                                halflink = HalfLink(halflinks.loc[(ind_hl),'name'],start_node)
                            elif halflinks.loc[(ind_hl),'carrier'] == 'g':
                                halflink_data = halflinks.loc[(ind_hl),'data']
                                halflink_q = halflink_data.get('q')
                                if halflink_data.get('q_unit') == 'kg/s':
                                    pass
                                else:
                                    raise ValueError("Invalid value encountered for 'q_unit'. It must be 'kg/s', not {}".format(node_data.get('p_unit')))
                                halflink = GasHalfLink(halflinks.loc[(ind_hl),'name'],start_node,halflink_q,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                            elif halflinks.loc[(ind_hl),'carrier'] == 'e':
                                halflink_data = halflinks.loc[(ind_hl),'data']
                                halflink_P = halflink_data.get('P')
                                halflink_Q = halflink_data.get('Q')
                                if halflink_data.get('P_unit') == 'p.u.':
                                    pass
                                else:
                                    raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('P_unit')))
                                if halflink_data.get('Q_unit') == 'p.u.':
                                    pass
                                else:
                                    raise ValueError("Invalid value encountered for 'P_unit'. It must be 'p.u.', not {}".format(node_data.get('Q_unit')))
                                halflink = ElectricalHalfLink(halflinks.loc[(ind_hl),'name'],start_node,P=halflink_P,Q=halflink_Q,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                            elif halflinks.loc[(ind_hl),'carrier'] == 'h':
                                halflink_data = halflinks.loc[(ind_hl),'data']
                                halflink_m = halflink_data.get('m')
                                halflink_phi = halflink_data.get('phi')
                                halflink_Ts = halflink_data.get('Ts')
                                halflink_Tr = halflink_data.get('Tr')
                                halflink_dT = halflink_data.get('dT')
                                if halflink_data.get('m_unit') == 'kg/s':
                                    pass
                                else:
                                    raise ValueError("Invalid value encountered for 'm_unit'. It must be 'kg/s', not {}".format(halflink_data.get('m_unit')))
                                if halflink_data.get('phi_unit') == 'W':
                                    pass
                                elif halflink_data.get('phi_unit') == 'MW':
                                    halflink_phi *= MW
                                else:
                                    raise ValueError("Invalid value encountered for 'phi_unit'. It must be 'W' or 'MW', not {}".format(halflink_data.get('m_unit')))
                                if halflink_data.get('T_unit') == 'C':
                                    pass
                                elif node_data.get('T_unit') == 'K':
                                    halflink_Ts -= 273.15
                                    halflink_Tr -= 273.15
                                    halflink_dT -= 273.15
                                else:
                                    raise ValueError("Invalid value encountered for 'T_unit'. It must be 'C' or 'K' not {}".format(halflink_data.get('T_unit')))
                                if isinstance(start_node,HeatNode) and start_node.node_type in [0,8]: # for source reference slack nodes and load reference slack nodes a half link is automatically created when the node is made. But it is not added to the network yet.
                                    halflink = start_node.half_links[0]
                                    halflink.name = halflinks.loc[(ind_hl),'name']
                                    halflink.m = halflink_m
                                    halflink.dphi = halflink_phi
                                    halflink.Ts = halflink_Ts
                                    halflink.Tr = halflink_Tr
                                    halflink.dT = halflink_dT
                                    halflink.set_type(halflinks.loc[(ind_hl),'type'],halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                                else:
                                    if start_node.half_links:
                                        halflink = start_node.half_links[0]
                                        halflink.name = halflinks.loc[(ind_hl),'name']
                                        halflink.m = halflink_m
                                        halflink.dphi = halflink_phi
                                        halflink.Ts = halflink_Ts
                                        halflink.Tr = halflink_Tr
                                        halflink.dT = halflink_dT
                                        halflink.set_type(halflinks.loc[(ind_hl),'type'],halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                                    else:
                                        halflink = HeatHalfLink(halflinks.loc[(ind_hl),'name'],start_node,m=halflink_m,phi=halflink_phi,To=halflink_To,link_type=halflinks.loc[(ind_hl),'type'],link_params=halflinks.loc[(ind_hl),'parameters'],bc_type=halflinks.loc[(ind_hl),'bc_type'])
                            elif halflinks.loc[(ind_hl),'carrier'] == 'c':
                                raise ValueError("Coupling half links are not implemented")
                            else:
                                raise ValueError("Invalid value encountered for carrier. It must be 'g','e','h','c','no type'.")
                            net.add_half_link(halflink,position=ind_hl[-1])
    return net
