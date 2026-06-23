"""Heterogeneous network base class, including Network, and Node """
from meslf.networks.network import Network, Node, Link, HalfLink
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNetwork, HeatNode, HeatLink, HeatHalfLink
from meslf.node_equations import coupling
from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
from meslf.load_flow.non_linear_solvers import NR, NR_FD, Fsolver, Root
import numpy as np
import warnings

# ===========================================================================
class HeterogeneousNetwork(Network):
    """Overall heterogeneous network class. Subclass of Network.
    NB. Currently the assumption is that a heterogeneous network is based on homogeneous subnetworks (one per carrier?!!!). Only heterogeneous nodes can be added seperately.

    Attributes
    ----------
    name : str
        The name of the network.
    nodes : list
        List of nodes in the network.
    links : list
        List of links in the network.
    networks : list
        List of (sub)networks in the network
    """
    def __init__(self,name,Ta=None):
        """Creates a HeatNetwork object.

        Parameters
        ----------
        name : str
            The name of the network.
        """
        super().__init__(name)

    def get_nodes(self,node_types=list(),unit_types=list(),carriers=list()):
        """Iterates over all the nodes in the list of nodes, with the specified node type.

        Parameters
        ----------
        node_types : list, optional
            List of node type of the nodes to be yielded. If empty, all the nodes are yielded. Default is an empty list.
        unit_type : list, optional
            List of unit types of the heterogeneous nodes to be yielded. If empty, all the nodes are yielded. Default is an empty list.
        carriers : list, optional
            List of carriers of the nodes to be yielded. If empty, all the nodes are yielded. Carriers are, 'gas', 'elec', 'heat', or 'het'

        Yields
        ------
        n : Node
            The next Node instance in self.nodes
        """
        for n in super().get_nodes():
            if node_types and unit_types:
                if carriers and 'het' in carriers:
                    if isinstance(n,HeterogeneousNode): # only heterogeneous nodes have a unit type
                        if n.node_type in node_types and n.unit_type in unit_types:
                            yield n
                else:
                    if isinstance(n,HeterogeneousNode): # only heterogeneous nodes have a unit type
                        if n.node_type in node_types and n.unit_type in unit_types:
                            yield n
            elif node_types:
                if n.node_type in node_types:
                    if carriers:
                        if ('gas' in carriers and isinstance(n,GasNode)) or ('elec' in carriers and isinstance(n,ElectricalNode)) or ('heat' in carriers and isinstance(n,HeatNode)) or ('het' in carriers and isinstance(n,HeterogeneousNode)):
                            yield n
            elif unit_types:
                if carriers and 'het' in carriers: # only heterogeneous nodes have a unit type
                    if isinstance(n,HeterogeneousNode):
                        if n.unit_type in unit_types:
                            yield n
                else:
                    if isinstance(n,HeterogeneousNode): # only heterogeneous nodes have a unit type
                        if n.unit_type in unit_types:
                            yield n
            elif carriers:
                if ('gas' in carriers and isinstance(n,GasNode)) or ('elec' in carriers and isinstance(n,ElectricalNode)) or ('heat' in carriers and isinstance(n,HeatNode)) or ('het' in carriers and isinstance(n,HeterogeneousNode)):
                    yield n
            else:
                yield n

    def get_networks(self,carriers=list(),get_all=False):
        """Iterates over all the networks in the list of networks of the specified carrier.

        Parameters
        ----------
        carriers : list, optional
            List of carriers of the subnetworks to be yielded. If empty, all the subnetworks are yielded. Carriers are, 'gas', 'elec', 'heat', or 'het'
        get_all : boolean, optional
            Specificy if the subsubnetworks etc. should also be returned. If False, only the subnetworks are returned. If True, all the subnetworks of all the subnetwork etc. are returned. Default is False

        Yields
        ------
        net : Network
            The next Network instance in self.networks
        """
        for net in super().get_networks(get_all=get_all):
            if carriers:
                if 'gas' in carriers and isinstance(net,GasNetwork):
                    yield net
                elif 'elec' in carriers and isinstance(net,ElectricalNetwork):
                    yield net
                elif 'heat' in carriers and isinstance(net,HeatNetwork):
                    yield net
                elif 'het' in carriers and isinstance(net,HeterogeneousNetwork):
                    yield net
            else:
                yield net

    def initialize(self):
        """Initializes the network, by initializing all the (direct) subnetworks.
        """
        for net in self.get_networks():
            net.initialize()

    def get_x_entries(self,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None}):
        """Return all the nodes, links, and half links that have an entry in variable vector x

        Parameters
        ----------
        formulation : dict, optional
            Dictionary with formulation per energy carrier used to form system of equations. Default is {'gas':'nodal','elec':'complex_power','heat':'standard','het':None}.

        Returns
        -------
        x_entries : list
           List of all the nodes, links, and half links that contribute to x.
        xg_entries : list
           List of all the nodes, links, and half links that contribute to part of x corresponding to the gas (sub)network.
        xe_entries : list
           List of all the nodes, links, and half links that contribute to part of x corresponding to the electrical (sub)network.
        unknown_delta_nodes : list
            List of all the nodes that have unknown voltage angle
        unknown_V_nodes : list
            List of all the nodes that have unknown voltage amplitude
        xh_entries : list
           List of all the nodes, links, and half links that contribute to part of x corresponding to the heat (sub)network.
        unknown_m_links : list
            List of all the links that have unknown link flow.
        unknown_hm_halflinks : list
            List of all the half links in the heat network that have unknown half link flow, and are connected to a non slack node. Only used if formulation is 'half_link_flow'.
        unknown_p_nodes : list
            List of all the nodes that have unknown pressure.
        unknown_Ts_nodes : list
            List of all the nodes that have unknown supply temperature.
        unknown_Tr_nodes : list
            List of all the nodes that have unknown return temperature.
        unknown_hTs_halflinks : list
            List of all the half links in the heat network that have unknown supply temperature.
        unknown_hTr_halflinks : list
            List of all the half links in the heat network that have unknown return temperature.
        """
        # define empty lists (are replaced later)
        # single-carrier
        xg_entries = list()
        xe_entries = list()
        unknown_delta_nodes = list()
        unknown_V_nodes = list()
        xh_entries = list()
        unknown_m_links = list()
        unknown_m_halflinks = list()
        unknown_p_nodes = list()
        unknown_Ts_nodes = list()
        unknown_Tr_nodes = list()
        unknown_Ts_halflinks = list()
        unknown_Tr_halflinks = list()
        # coupling
        unknown_qc_links = list()
        unknown_qc_halflinks = list()
        unknown_Pc_links = list()
        unknown_Pc_halflinks = list()
        unknown_Qc_links = list()
        unknown_Qc_halflinks = list()
        unknown_mc_links = list()
        unknown_dphi_links = list()
        unknown_Ts_links = list()
        unknown_Tr_links = list()
        unknown_mc_halflinks = list()
        unknown_dphic_halflinks = list()
        unknown_Tsc_halflinks = list()
        unknown_Trc_halflinks = list()
        for net in self.get_networks():
            if isinstance(net,GasNetwork):
                xg_entries = net.get_x_entries(formulation=formulation['gas'])
                unknown_qc_links = [link for link in net.get_links(link_types=['dummy'],bc_types=[0]) if (isinstance(link.start_node,HeterogeneousNode) or isinstance(link.end_node,HeterogeneousNode))] # coupling links
            elif isinstance(net,ElectricalNetwork):
                xe_entries, unknown_delta_nodes, unknown_V_nodes = net.get_x_entries(formulation=formulation['elec'])
                unknown_Pc_links = [link for link in net.get_links(link_types=['dummy'],bc_types=[0,2,5]) if (isinstance(link.start_node,HeterogeneousNode) or isinstance(link.end_node,HeterogeneousNode))] # coupling links with unknown Pstart or Pend
                unknown_Qc_links = [link for link in net.get_links(link_types=['dummy'],bc_types=[0,1,4]) if (isinstance(link.start_node,HeterogeneousNode) or isinstance(link.end_node,HeterogeneousNode))] # coupling links with unknown Qstart or Qend
            elif isinstance(net,HeatNetwork):
                xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks = net.get_x_entries(formulation=formulation['heat'])
                unknown_mc_links = [link for link in net.get_links(link_types=['dummy']) if (isinstance(link.start_node,HeterogeneousNode) or isinstance(link.end_node,HeterogeneousNode))] # coupling links with unknown m
                unknown_dphi_links = [link for link in net.get_links(link_types=['dummy'],bc_types=[0,1,6,7,10,11]) if (isinstance(link.start_node,HeterogeneousNode) or isinstance(link.end_node,HeterogeneousNode))] # coupling links with unknown dphistart
                unknown_Ts_links = [link for link in net.get_links(link_types=['dummy'],bc_types=[0,4,8,10]) if (isinstance(link.start_node,HeterogeneousNode) or isinstance(link.end_node,HeterogeneousNode))] # coupling links with unknown Tsstart
                unknown_Tr_links = [link for link in net.get_links(link_types=['dummy'],bc_types=[1,5,9,11]) if (isinstance(link.start_node,HeterogeneousNode) or isinstance(link.end_node,HeterogeneousNode))] # coupling links with unknown Trstart
        for node in self.get_nodes(carriers=['het']):
            for hl in node.get_half_links():
                if isinstance(hl,GasHalfLink):
                    if hl.bc_type == 0:
                        unknown_qc_halflinks.append(hl)
                elif isinstance(hl,ElectricalHalfLink):
                    if hl.bc_type in [0,3]:
                        unknown_Pc_halflinks.append(hl)
                    if hl.bc_type in [0,2]:
                        unknown_Qc_halflinks.append(hl)
                elif isinstance(hl,HeatHalfLink):
                    if hl.bc_type in [0,1,2,3,4,5,6,7,8,9,10,11]:
                        unknown_mc_halflinks.append(hl)
                    if hl.bc_type in [0,1,6,7,10,11,12,13,18,19,22,23]:
                        unknown_dphic_halflinks.append(hl)
                    if hl.bc_type in [0,4,8,10,12,16,20,22]:
                        unknown_Tsc_halflinks.append(hl)
                    if hl.bc_type in [1,5,9,11,13,17,21,23]:
                        unknown_Trc_halflinks.append(hl)
        xc_entries = unknown_qc_links + unknown_qc_halflinks + unknown_Pc_links + unknown_Pc_halflinks + unknown_Qc_links + unknown_Qc_halflinks + unknown_mc_links + unknown_mc_halflinks + unknown_dphi_links + unknown_dphic_halflinks + unknown_Ts_links + unknown_Tsc_halflinks + unknown_Tr_links + unknown_Trc_halflinks
        x_entries = xg_entries + xe_entries + xh_entries + xc_entries
        return x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks

    def get_F_entries(self,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None}):
        """Return all the nodes, links, and half links that have an entry in function vector F.
        It also returns the nodes per carrier, and per contribution to F.

        Parameters
        ----------
        formulation : dict, optional
            Dictionary with formulation per energy carrier used to form system of equations. Default is {'gas':'nodal','elec':'complex_power','heat':'standard','het':None}.

        Returns
        -------
        F_entries : list
           List of all the nodes, links, and half links that contribute to F.
        Fg_entries : list
           List of all the nodes, links, and half links that contribute to the part of F corresponding to the gas (sub)network.
        Fe_entries : list
           List of all the nodes, links, and half links that contribute to the part of F corresponding to the electrical (sub)network.
        known_P_nodes : list
            List of all the nodes that contribute to the active power equation
        known_Q_nodes : list
            List of all the nodes that contribute to the reactive power equation
        Fh_entries : list
           List of all the nodes, links, and half links that contribute to the part of F corresponding to the heat (sub)network.
        F_m_nodes : list
            List of all the nodes that contribute to conservation of mass.
        F_deltap_links : list
            List of all the links that contribute to link equations.
        F_Ts_nodes : list
            List of all the nodes that contribute to supply temperature mixing rule.
        F_Tr_nodes : list
            List of all the nodes that contribute to return temperature mixing rule.
        Fc_entries : list
           List of all the nodes, links, and half links that contribute to the heterogeneous or coupling part of F.
        F_fc_nodes : list
            List of all the nodes that contribute to coupling equations.
        F_fc_amount : list
            List with the amount of coupling equations for each node that contributes a coupling equation.
        F_phi_nodes : list
            List of all the nodes that contribute to a heat power equation.
        F_dT_nodes : list
            List of all the nodes that contribute to a temperature difference equation
        """
        # define empty lists (are replaced later)
        F_entries = list()
        Fg_entries = list()
        Fe_entries = list()
        known_P_nodes = list()
        known_Q_nodes = list()
        Fh_entries = list()
        F_m_nodes = list()
        F_deltap_links = list()
        F_Ts_nodes = list()
        F_Tr_nodes = list()
        F_phi_halflinks = list()
        F_dT_halflinks = list()
        F_fc_nodes = list()
        F_fc_amount = list()
        F_phi_nodes = list()
        F_dT_nodes = list()
        for net in self.get_networks():
            if isinstance(net,GasNetwork):
                Fg_entries = net.get_F_entries(formulation=formulation['gas'])
            elif isinstance(net,ElectricalNetwork):
                Fe_entries, known_P_nodes, known_Q_nodes = net.get_F_entries(formulation=formulation['elec'])
            elif isinstance(net,HeatNetwork):
                Fh_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_dT_halflinks = net.get_F_entries(formulation=formulation['heat'])
        F_fc_nodes = list(self.get_nodes(carriers='het'))
        F_fc_amount = [len(node.node_law()) for node in self.get_nodes() if isinstance(node,HeterogeneousNode)]
        F_Q_nodes = list()
        F_phi_nodes = [node for node in self.get_nodes(node_types=[0,1,2],carriers='het') if np.any([(isinstance(link,HeatLink) or isinstance(link,HeatHalfLink)) for link in node.get_links()])]# heterogeneous coupling node, where heat is involved (although not certain for EH), and where there is a heat (dummy) link connected to the heterogeneous node, such that a Tr of Ts is available
        F_dT_nodes = list(self.get_nodes(node_types=[2],unit_types=['gh_gas_boiler','gh_gas_boiler_part_load','eh_elec_boiler','geh_CHP','geh_CHP_part_load','EH'])) # heterogeneous coupling node, where heat is involved (although not certain for EH), and where delta T is specified instead of To
        Fc_entries = F_fc_nodes + F_phi_nodes + F_dT_nodes
        F_entries = Fg_entries + Fe_entries + Fh_entries + Fc_entries
        return F_entries, Fg_entries, Fe_entries, known_P_nodes, known_Q_nodes, Fh_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_dT_halflinks, Fc_entries, F_fc_nodes, F_fc_amount, F_phi_nodes, F_dT_nodes

    def set_x_init(self,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None},scale_var=None,scale_var_params=None,**kwargs):
        """Creates the initial guess for vector x, based on the current network parameters

        Parameters
        ----------
        formulation : dict, optional
            Dictionary with formulation per energy carrier used to form system of equations. Default is {'gas':'nodal','elec':'complex_power','heat':'standard','het':None}.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        x0 : np array
           Initial guess for variable vector x
        """
        x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = self.get_x_entries(formulation=formulation)
        x0 = np.zeros(len(x_entries))
        T_shift = None
        # homogeneous part
        for net in self.get_networks():
            if isinstance(net,GasNetwork):
                x0[0:len(xg_entries)] = net.set_x_init(formulation=formulation['gas'],scale_var=scale_var,scale_var_params=scale_var_params)
            elif isinstance(net,ElectricalNetwork):
                x0[len(xg_entries):len(xg_entries)+len(xe_entries)] = net.set_x_init(formulation=formulation['elec'],scale_var=scale_var,scale_var_params=scale_var_params)
            elif isinstance(net,HeatNetwork):
                T_shift = None#net.get_Ta(scale_var=scale_var,scale_var_params=scale_var_params)
                x0[len(xg_entries)+len(xe_entries):len(xg_entries)+len(xe_entries)+len(xh_entries)] = net.set_x_init(formulation=formulation['heat'],scale_var=scale_var,scale_var_params=scale_var_params)
        # coupling gas part
        ind_xc = len(xg_entries)+len(xe_entries)+len(xh_entries)
        for ind_el,el in enumerate(unknown_qc_links):
            x0[ind_el+ind_xc] = el.get_q(scale_var=scale_var,scale_var_params=scale_var_params)
        ind_xc += len(unknown_qc_links)
        for ind_el,el in enumerate(unknown_qc_halflinks):
            x0[ind_el+ind_xc] = -el.get_q(scale_var=scale_var,scale_var_params=scale_var_params) # minus based on how q is defined / used for the dummy links
        ind_xc += len(unknown_qc_halflinks)
        # coupling electricity part
        for ind_el,el in enumerate(unknown_Pc_links):
            x0[ind_el+ind_xc] = el.get_Pstart(scale_var=scale_var,scale_var_params=scale_var_params)
        ind_xc += len(unknown_Pc_links)
        for ind_el,el in enumerate(unknown_Pc_halflinks):
            x0[ind_el+ind_xc] = el.get_P(scale_var=scale_var,scale_var_params=scale_var_params)
        ind_xc += len(unknown_Pc_halflinks)
        for ind_el,el in enumerate(unknown_Qc_links):
            x0[ind_el+ind_xc] = el.get_Qstart(scale_var=scale_var,scale_var_params=scale_var_params)
        ind_xc += len(unknown_Qc_links)
        for ind_el,el in enumerate(unknown_Qc_halflinks):
            x0[ind_el+ind_xc] = el.get_Q(scale_var=scale_var,scale_var_params=scale_var_params)
        ind_xc += len(unknown_Qc_halflinks)
        # coupling heat part
        for ind_el,el in enumerate(unknown_mc_links):
            x0[ind_el+ind_xc] = el.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
        ind_xc += len(unknown_mc_links)
        for ind_el,el in enumerate(unknown_mc_halflinks):
            x0[ind_el+ind_xc] = -el.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
        ind_xc += len(unknown_mc_halflinks)
        for ind_el,el in enumerate(unknown_dphi_links):
            x0[ind_el+ind_xc] = -el.get_dphistart(scale_var=scale_var,scale_var_params=scale_var_params)
        ind_xc += len(unknown_dphi_links)
        for ind_el,el in enumerate(unknown_dphic_halflinks):
            x0[ind_el+ind_xc] = -el.get_dphi(scale_var=scale_var,scale_var_params=scale_var_params)
        ind_xc += len(unknown_dphic_halflinks)
        for ind_el,el in enumerate(unknown_Ts_links):
            x0[ind_el+ind_xc] = el.get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params)
        ind_xc += len(unknown_Ts_links)
        for ind_el,el in enumerate(unknown_Tsc_halflinks):
            x0[ind_el+ind_xc] = el.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        ind_xc += len(unknown_Tsc_halflinks)
        for ind_el,el in enumerate(unknown_Tr_links):
            x0[ind_el+ind_xc] = el.get_Trstart(scale_var=scale_var,scale_var_params=scale_var_params)
        ind_xc += len(unknown_Tr_links)
        for ind_el,el in enumerate(unknown_Trc_halflinks):
            x0[ind_el+ind_xc] = el.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        return x0

    def update(self,x,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None},scale_var=None,scale_var_params=None,**kwargs):
        """Updates the network given variable vector x

        Parameters
        ----------
        x : np array
            Variable vector x, scaled
        formulation : dict, optional
            Dictionary with formulation per energy carrier used to form system of equations. Default is {'gas':'nodal','elec':'complex_power','heat':'standard','het':None}.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        """
        x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = self.get_x_entries(formulation=formulation)
        T_shift = None
        if not self.networks: # network does not consist of subnetworks
            warnings.warn('No single-carrier subnetworks found. Only the heterogeneous nodes (and connected links and halflinks) are updated!')
        else:
            # homogeneous parts
            for net in self.get_networks():
                if isinstance(net,GasNetwork):
                    x_net = x[0:len(xg_entries)]
                    net_formulation=formulation['gas']
                    #warnings.warn('Using nodal formulation for updating gas network!')
                elif isinstance(net,ElectricalNetwork):
                    x_net = x[len(xg_entries):len(xg_entries)+len(xe_entries)]
                    net_formulation=formulation['elec']
                elif isinstance(net,HeatNetwork):
                    x_net = x[len(xg_entries)+len(xe_entries):len(xg_entries)+len(xe_entries)+len(xh_entries)]
                    T_shift = None#net.get_Ta(scale_var=scale_var,scale_var_params=scale_var_params)
                    net_formulation=formulation['heat']
                else:
                    raise NotImplementedError("update(x) not implemented yet for a heterogeneous network consisting of other heterogeneous subnetworks")
                net.update(x_net,formulation=net_formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        # heterogeneous gas part
        ind_xc = len(xg_entries)+len(xe_entries)+len(xh_entries)
        for ind_l,l in enumerate(unknown_qc_links):
            q = x[ind_l+ind_xc]
            if scale_var == 'per_unit':
                q *= scale_var_params['qbase']
            l.q = q
        ind_xc += len(unknown_qc_links)
        for ind_hl,hl in enumerate(unknown_qc_halflinks):
            q = -x[ind_hl+ind_xc] # minus based on how q is defined / used for the dummy links
            if scale_var == 'per_unit':
                q *= scale_var_params['qbase']
            hl.q = q
        ind_xc += len(unknown_qc_halflinks)
        # heterogeneous electricity part
        for ind_l,l in enumerate(unknown_Pc_links):
            P = x[ind_l+ind_xc]
            if scale_var == 'per_unit':
                P *= scale_var_params['Sbase']
            l.Pstart = P
            l.Pend = -P
        ind_xc += len(unknown_Pc_links)
        for ind_hl,hl in enumerate(unknown_Pc_halflinks):
            P = x[ind_hl+ind_xc]
            if scale_var == 'per_unit':
                P *= scale_var_params['Sbase']
            hl.P = P
        ind_xc += len(unknown_Pc_halflinks)
        for ind_l,l in enumerate(unknown_Qc_links):
            Q = x[ind_l+ind_xc]
            if scale_var == 'per_unit':
                Q *= scale_var_params['Sbase']
            l.Qstart = Q
            l.Qend = -Q
        ind_xc += len(unknown_Qc_links)
        for ind_hl,hl in enumerate(unknown_Qc_halflinks):
            Q = x[ind_hl+ind_xc]
            if scale_var == 'per_unit':
                Q *= scale_var_params['Sbase']
            hl.Q = Q
        ind_xc += len(unknown_Qc_halflinks)
        # heterogeneous heat part
        for ind_l,l in enumerate(unknown_mc_links):
            m = x[ind_l+ind_xc]
            if m >=0:
                if scale_var == 'per_unit':
                    m *= scale_var_params['qbase']
                l.m = m
            else:
                warnings.warn('Update(x) encountered a heat link with heterogeneous start node and negative flow. Flow is set to 1e-6.')
                # adjust link flow:
                m_adjusted = 1e-6
                l.m = m_adjusted
                x[ind_l+len(xg_entries)+len(xe_entries)+len(xh_entries)+len(unknown_qc_links)+len(unknown_Pc_links)+len(unknown_Qc_links)] = m_adjusted
        ind_xc += len(unknown_mc_links)
        for ind_hl,hl in enumerate(unknown_mc_halflinks):
            m = -x[ind_hl+ind_xc]
            if scale_var == 'per_unit':
                m *= scale_var_params['qbase']
            hl.m = m
        ind_xc += len(unknown_mc_halflinks)
        for ind_l,l in enumerate(unknown_dphi_links):
            phi = x[ind_l+ind_xc]
            if scale_var == 'per_unit':
                phi *= scale_var_params['phibase']
            if phi < 0 :
                warnings.warn('Update(x) encountered a negative coupling heat power. But all coupling components should produce heat!!')
            elif phi == 0:
                warnings.warn('Update(x) encountered a zero coupling heat power. But all coupling components should produce heat!!')
            l.dphistart = -phi
        ind_xc += len(unknown_dphi_links)
        for ind_hl,hl in enumerate(unknown_dphic_halflinks):
            phi = x[ind_hl+ind_xc]
            if scale_var == 'per_unit':
                phi *= scale_var_params['phibase']
            if phi < 0 :
                warnings.warn('Update(x) encountered a negative coupling heat power. But all coupling components should produce heat!!')
            elif phi == 0:
                warnings.warn('Update(x) encountered a zero coupling heat power. But all coupling components should produce heat!!')
            hl.dphi = -phi
        ind_xc += len(unknown_dphic_halflinks)
        for ind_l,l in enumerate(unknown_Ts_links):
            Ts = x[ind_l+ind_xc]
            if not T_shift == None:
                Ts += T_shift
            if scale_var == 'per_unit':
                Ts *= scale_var_params['Tbase']
            l.Tsstart = Ts
        ind_xc += len(unknown_Ts_links)
        for ind_hl,hl in enumerate(unknown_Tsc_halflinks):
            Ts = x[ind_hl+ind_xc]
            if not T_shift == None:
                Ts += T_shift
            if scale_var == 'per_unit':
                Ts *= scale_var_params['Tbase']
            hl.Ts = Ts
        ind_xc += len(unknown_Tsc_halflinks)
        for ind_l,l in enumerate(unknown_Tr_links):
            Tr = x[ind_l+ind_xc]
            if not T_shift == None:
                Tr += T_shift
            if scale_var == 'per_unit':
                Tr *= scale_var_params['Tbase']
            l.Trstart = Tr
        ind_xc += len(unknown_Tr_links)
        for ind_hl,hl in enumerate(unknown_Trc_halflinks):
            Tr = x[ind_hl+ind_xc]
            if not T_shift == None:
                Tr += T_shift
            if scale_var == 'per_unit':
                Tr *= scale_var_params['Tbase']
            hl.Tr = Tr
        ind_xc += len(unknown_Trc_halflinks)
        # update other (half) link temperatures
        for ind_n,n in enumerate(self.get_nodes(unit_types=['gh_gas_boiler','gh_gas_boiler_part_load','eh_elec_boiler','geh_CHP','geh_CHP_part_load','EH'])): # nodes where heat is or might be involved
            for hl in n.get_half_links(carriers=['heat']):
                if hl.sink: # sinks
                    if (hl.bc_type in [4,5,10,11,16,17,22,23]) and (hl not in unknown_Tr_halflinks) and (hl not in unknown_Trc_halflinks): # dT known (Ts is assumed known, or updated before)
                        hl.Tr = hl.Ts - hl.dT
                elif hl.source: # sources
                    if (hl.bc_type in [4,5,10,11,16,17,22,23]) and (hl not in unknown_Ts_halflinks) and (hl not in unknown_Tsc_halflinks): # dT known (Tr is assumed known, or updated before)
                        hl.Ts = hl.dT + hl.Tr
        heat_coupling_links = [link for link in self.get_links(link_types=['dummy']) if (isinstance(link,HeatLink) and (isinstance(link.start_node,HeterogeneousNode) or isinstance(link.end_node,HeterogeneousNode)))]
        for ind_l,l in enumerate(heat_coupling_links):
            if isinstance(l.start_node,HeterogeneousNode):
                if l.bc_type in [1,3,5,7,9,11]: # coupling acts as sink
                    raise NotImplementedError('Updating heat dummy link for which the coupling acts as a sink not implemented')
                else: # coupling acts as source. Ts is either known, or part is unknown_Ts_links, such that it is already updated
                    if l.m >= 0: # mass flow in correct direction for a source
                        l.Trend = l.end_node.Tr
                        Trstart = l.return_temp_start(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                        Tsend = l.supply_temp_end(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                        if scale_var == 'per_unit':
                            Trstart *= scale_var_params['Tbase']
                            Tsend *= scale_var_params['Tbase']
                        l.Trstart = Trstart
                        l.Tsend = Tsend
                    else:
                        raise NotImplementedError('Updating encountered heat dummy link which is a source, but acts like a sink')
            else:
                raise NotImplementedError('Update not implemented for a heat link with heterogeneous end node')

    def update_full(self,x,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None},scale_var=None,scale_var_params=None,**kwargs):
        """Updates the full network given variable vector x.
        Unlike update(x), not only the values from x are updated, but also all
        parameters not included in x.

        Parameters
        ----------
        x : np array
            Variable vector x, scaled
        formulation : dict, optional
            Dictionary with formulation per energy carrier used to form system of equations. Default is {'gas':'nodal','elec':'complex_power','heat':'standard','het':None}.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        p_g_vec : np array
            Array with all unscaled nodal pressures in the gas network
        q_vec : np array
            Array with all unscaled gas link flows
        q_inj : np array
            Array with all unscaled nodal injeted gas flows
        delta_vec : np array
            Array with all unscaled nodal voltage angles
        V_mag_vec : np array
            Array with all unscaled nodal voltage amplitudes
        S_inj : np array
            Array with all unscaled injected complex powers
        P_edge : np array
            Array with all unscaled link active powers
        Q_edge : np array
            Array with all unscaled link reactive powers
        m_vec : np array
            Array with all unscaled heat link flows
        p_h_vec : np array
            Array with all unscaled nodal pressures in the heat network
        Ts_vec : np array
            Array with all unscaled nodal supply temperatures
        Tr_vec : np array
            Array with all unscaled nodal return temperatures
        m_hl_vec : list
            List of arrays with all unscaled  half link flows per node
        phi_hl_vec : list
            List of array with all unscaled half link heat powers per node
        To_hl_vec : list
            List of array with all unscaled half link outflow temperatures per node
        qc_vec : np array
            Array with all unscaled coupling gas link flows
        Pc_vec : np array
            Array with all unscaled coupling electrical link active powers
        Qc_vec : np array
            Array with all unscaled coupling electrical link reactive powers
        mc_vec : np array
            Array with all unscaled coupling heat link flows
        phic_vec : np array
            Array with all unscaled coupling half link heat power
        Toc_vec : np array
            Array with all unscaled coupling half link outflow temperatures
        """
        self.update(x,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = self.get_x_entries(formulation=formulation)
        # homogeneous parts
        p_g_vec = np.array([])
        q_vec = np.array([])
        q_inj = np.array([])
        delta_vec = np.array([])
        V_mag_vec = np.array([])
        S_inj = np.array([])
        P_edge = np.array([])
        Q_edge = np.array([])
        m_vec = np.array([])
        p_h_vec = np.array([])
        Ts_vec = np.array([])
        Tr_vec = np.array([])
        m_hl_vec = np.array([])
        phi_hl_vec = np.array([])
        Ts_hl_vec = np.array([])
        Tr_hl_vec = np.array([])
        for net in self.get_networks():
            if isinstance(net,GasNetwork):
                x_net = x[0:len(xg_entries)]
                #warnings.warn('Using nodal formulation for updating gas network!')
                p_g_vec,q_vec,q_inj = net.update_full(x_net,formulation=formulation['gas'],scale_var=scale_var,scale_var_params=scale_var_params)
            elif isinstance(net,ElectricalNetwork):
                x_net = x[len(xg_entries):len(xg_entries)+len(xe_entries)]
                delta_vec,V_mag_vec,S_inj,P_edge,Q_edge = net.update_full(x_net,formulation=formulation['elec'],scale_var=scale_var,scale_var_params=scale_var_params)
            elif isinstance(net,HeatNetwork):
                x_net = x[len(xg_entries)+len(xe_entries):len(xg_entries)+len(xe_entries)+len(xh_entries)]
                m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec = net.update_full(x_net,formulation=formulation['heat'],scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                raise Error("update(x) not implemented yet for a heterogeneous network consisting of other heterogeneous subnetworks")
        # coupling part
        qc_vec = list()
        Pc_vec = list()
        Qc_vec = list()
        mc_vec = list()
        phic_vec = list()
        Tsc_vec = list()
        Trc_vec = list()
        # heterogeneous part
        for node in self.get_nodes(carriers=['het']):
            for link in node.get_links():
                if isinstance(link,GasLink) or isinstance(link,GasHalfLink):
                    qc_vec.append(link.q)
                elif isinstance(link,ElectricalLink):
                    Pc_vec.append(link.Pstart)
                    Qc_vec.append(link.Qstart)
                elif isinstance(link,ElectricalHalfLink):
                    Pc_vec.append(link.P)
                    Qc_vec.append(link.Q)
                elif isinstance(link,HeatLink):
                    mc_vec.append(link.m)
                    phic_vec.append(-link.dphistart)
                    Tsc_vec.append(link.Tsstart)
                    Trc_vec.append(link.Trstart)
                elif isinstance(link,HeatHalfLink):
                    mc_vec.append(-link.m)
                    phic_vec.append(-link.dphi)
                    Tsc_vec.append(link.Ts)
                    Trc_vec.append(link.Tr)
        return p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec

    def reset_network(self,x_init,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None},scale_var=None,scale_var_params=None,**kwargs):
        """Resets the full network to initial conditions given initial guess vector x.

        Parameters
        ----------
        x_init : np array
            Vector with initial guess for x.
        formulation : string, optional
            Formulation used to form system of equations. Default is 'complex_power. Options are 'complex_power'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        """
        x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = self.get_x_entries(formulation=formulation)
        # homogeneous parts
        for net in self.get_networks():
            if isinstance(net,GasNetwork):
                x_net = x_init[0:len(xg_entries)]
                net_formulation=formulation['gas']
                #warnings.warn('Using nodal formulation for updating gas network!')
            elif isinstance(net,ElectricalNetwork):
                x_net = x_init[len(xg_entries):len(xg_entries)+len(xe_entries)]
                net_formulation=formulation['elec']
            elif isinstance(net,HeatNetwork):
                x_net = x_init[len(xg_entries)+len(xe_entries):len(xg_entries)+len(xe_entries)+len(xh_entries)]
                net_formulation=formulation['heat']
            else:
                raise Error("update(x) not implemented yet for a heterogeneous network consisting of other heterogeneous subnetworks")
            net.reset_network(x_net,formulation=net_formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        self.update_full(x_init,formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    def solve_network(self,tol,max_iter,*args,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None},scale_var=None,scale_var_params=None,solver='NR',D_F=np.array([]),D_x=np.array([]),P_F=np.array([]),P_x=np.array([]),det_tol=None,return_all_x=False,lin_solver='solve',max_iter_lin=None,**kwargs):
        """Solves the steady-state load flow problem for the network.

        Parameters
        ----------
        tol : float
            Tolerance for solver.
        max_iter : int
            Maximum number of iterations of solver.
        formulation : dict, optional
            Dictionary with formulation per energy carrier used to form system of equations. Default is {'gas':'nodal','elec':'complex_power','heat':'standard','het':None}.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        solver : string, optional
            Solver used. Default is standard NR. Options are 'NR', 'NR_FD', 'fsolver', or 'root'.
        D_F : array, optional
            Diagonal scaling matrix :math:`D_F` with which to scale the system of equations :math:`F`.
        D_x : array, optional
            Diagonal matrix :math:`D_x` with which to scale the variable vector :math:`x`.
        P_F : array, optional
            Permutation matrix :math:`P_F` for the vector of equations :math:`F(x)`. This matrix is assumed to be an orthogonal binary matrix.
        P_x : array, optional
            Permutation matrix :math:`P_x` for the vector of variables :math:`x`. This matrix is assumed to be an orthogonal binary matrix.
        det_tol : float, optional
            Value of the determinant below which the Jacobian matrix is considered numerically singular. The solver is then stopped. Is only used if solver = 'NR'.
        return_all_x : bool, optional
            When true, the vector x is returned for every iteration. That is, a matrix with x as rows at every iteration is returned. Is only be used if solver = 'NR'.
        lin_solver : str, optional
            Determines which linear solver is used when using NR (so only used if solver = 'NR' of solver = 'NR_FD'). Options are 'gmres', 'bicgstab', and 'solve'. Default is 'solve', which uses numpy.linalg.solve or scipy.sparse.linalg.solve. When 'gmres', 'bicgstab', is used the Jacobian matrix must be sparse.

        Returns
        -------
        x_sol : np array
            Solution vector x, scaled.
        iters : int
            Number of iterations used.
        err_vec : list
            List with the error for every iteration.
        p_g_vec : np array
            Array with all unscaled nodal pressures in the gas network
        q_vec : np array
            Array with all unscaled gas link flows
        q_inj : np array
            Array with all unscaled nodal injeted gas flows
        delta_vec : np array
            Array with all unscaled nodal voltage angles
        V_mag_vec : np array
            Array with all unscaled nodal voltage amplitudes
        S_inj : np array
            Array with all unscaled injected complex powers
        P_edge : np array
            Array with all unscaled link active powers
        Q_edge : np array
            Array with all unscaled link reactive powers
        m_vec : np array
            Array with all unscaled heat link flows
        p_h_vec : np array
            Array with all unscaled nodal pressures in the heat network
        Ts_vec : np array
            Array with all unscaled nodal supply temperatures
        Tr_vec : np array
            Array with all unscaled nodal return temperatures
        m_hl_vec : list
            List of arrays with all unscaled  half link flows per node
        phi_hl_vec : list
            List of array with all unscaled half link heat powers per node
        To_hl_vec : list
            List of array with all unscaled half link outflow temperatures per node
        qc_vec : np array
            Array with all unscaled coupling gas link flows
        Pc_vec : np array
            Array with all unscaled coupling electrical link active powers
        Qc_vec : np array
            Array with all unscaled coupling electrical link reactive powers
        mc_vec : np array
            Array with all unscaled coupling heat link flows
        phic_vec : np array
            Array with all unscaled coupling half link heat power
        Toc_vec : np array
            Array with all unscaled coupling half link outflow temperatures
        """
        x0 = self.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        nlsys = NonLinearSystemHeterogeneous(self,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        # solve
        if solver == 'NR':
            nrsolve = NR()
        elif solver == 'NR_FD':
            nrsolve = NR_FD()
            nrsolve.h = kwargs['h']
        elif solver == 'fsolver':
            nrsolve = Fsolver()
        elif solver == 'root':
            nrsolve = Root()
        nrsolve.tol = tol
        if scale_var == 'matrix': # use matrix scaling
            D_x = nlsys.Dx()
            D_F = nlsys.DF()
        if solver == 'NR':
            if det_tol:
                if return_all_x:
                    x_sol,x_mat = nrsolve.solve(nlsys,x0,max_iter,D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x,det_tol=det_tol,return_all_x=return_all_x,lin_solver=lin_solver,max_iter_lin=max_iter_lin)
                else:
                    x_sol = nrsolve.solve(nlsys,x0,max_iter,D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x,det_tol=det_tol,return_all_x=return_all_x,lin_solver=lin_solver,max_iter_lin=max_iter_lin)
            else:
                if return_all_x:
                    x_sol,x_mat = nrsolve.solve(nlsys,x0,max_iter,D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x,return_all_x=return_all_x,lin_solver=lin_solver,max_iter_lin=max_iter_lin)
                else:
                    x_sol = nrsolve.solve(nlsys,x0,max_iter,D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x,return_all_x=return_all_x,lin_solver=lin_solver,max_iter_lin=max_iter_lin)
        else:
            x_sol = nrsolve.solve(nlsys,x0,max_iter,D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x,lin_solver=lin_solver,max_iter_lin=max_iter_lin)
        iters = nrsolve.iters
        err_vec = nrsolve.err_vec
        if solver == 'NR':
            print("Total time NR: {:6.3f}s, time linear solver: {:6.3e}s total, {:6.3e}s average per NR iteration".format(nrsolve.nl_time,nrsolve.l_time*iters,nrsolve.l_time))
        # update the rest of the network
        p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = self.update_full(x_sol,formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        if return_all_x and solver == 'NR':
            return x_sol,x_mat,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec
        else:
            return x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec

# ===========================================================================
class HeterogeneousNode(Node):
    """Heterogeneous node class.

    Attributes
    ----------
    name : str
        The name of the node.
    out_links : list
        List of outgoing links and half links connected to the node.
    in_links : list
        List of ingoing links and half links connected to the node.
    half_links : list
        List of half links connected to the node.
    number : int
        Number of the node.
    node_type : int
        Type of the node, 0 = standard node (To unkown), 1 = temp. node (To known).
    unit_type : str, optional
        Type of the conversion unit, which determines the node law(s). Default is 'dummy'. Options are 'dummy', 'ge_gas_fired_gen', 'ge_gas_fired_gen_valve_point', 'gh_gas_boiler','gh_gas_boiler_part_load', 'eh_elec_boiler', 'geh_CHP','geh_CHP_part_load', or 'EH'.
    unit_params : dict, optional
        Dictionary with parameters needed for the node law of a specific node type. Default is an empty dict.
    """
    def __init__(self,name,x=0.,y=0.,node_type=0.,unit_type='dummy',unit_params=dict(),Ein_len=list(),Eout_len=list()):
        """Creates an HeatNode object.

        Parameters
        ----------
        name : str
            Name of the node.
        x : float, optional
            x coordinate of the node. Default is 0.
        y : float, optional
            y coordinate of the node. Default is 0.
        node_type : int, optional
            Type of the node, 0 = standard node (Ts unkown), 1 = temp. node (Ts known), 2 = temp. diff node (dT known), 3 = standard node (Ts unkown) without heat power equation, 4 = temp. node (Ts known) without heat power equation, 5 = temp. diff node (dT known) without heat power equation. Default is 0.
        unit_type : str, optional
            Type of the node, which determines the node law(s). Default is 'dummy'. Options are 'dummy', 'ge_gas_fired_gen', 'ge_gas_fired_gen_valve_point', 'gh_gas_boiler','gh_gas_boiler_part_load', 'eh_elec_boiler', 'geh_CHP','geh_CHP_part_load', or 'EH'.
        unit_params : dict, optional
            Dictionary with parameters needed for the node law of a specific node type. Default is an empty dict.
        Ein_len : np array, optional
            Array with number of incoming energies per carrier: [number of incoming gas energies, number of incoming active powers, number of incoming heat powers]. Default is an empty list
        Eout_len : np array, optional
            Array with number of outgoing energies per carrier: [number of outgoing gas energies, number of outgoing active powers, number of outgoing heat powers]. Default is an empty list

        Raises
        ------
        ValueError
            If the node type is 1, but the outflow temperature is not specified
        ValueError
            If the unit_type is not a valid unit type. I.e. if unit_type is not 'dummy', 'ge_gas_fired_gen', 'ge_gas_fired_gen_valve_point', 'gh_gas_boiler','gh_gas_boiler_part_load', 'eh_elec_boiler', 'geh_CHP','geh_CHP_part_load', or 'EH'
        IndexError
            If the unit_type is 'EH', and Ein_len and Eout_len are given, but the coupling matrix had the wrong dimensions.
        """
        super().__init__(name,x=x,y=y)
        self.node_type = node_type
        self.unit_type = unit_type
        self.unit_params = unit_params
        self.Ein_len = Ein_len
        self.Eout_len = Eout_len
        self.color = 'k'
        if unit_type == 'dummy':
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr  = coupling.dummy()
        elif unit_type == 'ge_gas_fired_gen':
            eta = unit_params['eta']
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr  = coupling.ge_gas_fired_gen(eta)
        elif unit_type == 'ge_gas_fired_gen_valve_point':
            a = unit_params['a']
            b = unit_params['b']
            c = unit_params['c']
            d = unit_params['d']
            e = unit_params['e']
            Pmin = unit_params['Pmin']
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.ge_gas_fired_gen_valve_point(a,b,c,d,e,Pmin)
        elif unit_type == 'gh_gas_boiler':
            eta = unit_params['eta']
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.gh_gas_boiler(eta)
        elif unit_type == 'gh_gas_boiler_part_load':
            a = unit_params.get('a')
            b = unit_params.get('b')
            Ess = unit_params.get('Ess')
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.gh_gas_boiler_part_load(a,b,Ess)
        elif unit_type == 'eh_elec_boiler':
            eta = unit_params['eta']
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.eh_elec_boiler(eta)
        elif unit_type == 'geh_CHP':
            eta = unit_params['eta']
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.geh_CHP(eta)
        elif unit_type == 'geh_CHP_part_load':
            eta = unit_params.get('eta')
            a = unit_params.get('a')
            b = unit_params.get('b')
            d = unit_params.get('d')
            L1 = unit_params.get('L1')
            L2 = unit_params.get('L2')
            r1 = unit_params.get('r1')
            r2 = unit_params.get('r2')
            phimin = unit_params.get('phimin')
            phimax = unit_params.get('phimax')
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.geh_CHP_part_load(eta,a,b,d,L1,L2,r1,r2,phimin,phimax)
        elif unit_type == 'EH':
            C = unit_params.get('C')
            Ein_carriers = unit_params.get('Ein_carriers')
            Eout_carriers = unit_params.get('Eout_carriers')
            if len(Ein_len)>0 and len(Eout_len)>0:
                if not (C.shape[0] == np.sum(Eout_len) and C.shape[1] == np.sum(Ein_len)):
                    raise IndexError('coupling matrix C does not have right dimensions. It should be ({},{}), but it is {}'.format(np.sum(Ein_len),np.sum(Eout_len),C.shape))
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.EH(C,Ein_carriers,Eout_carriers)
        else:
            raise ValueError('unit_type is not a valid unit type.')
        self.Eout_of_Ein = Eout_of_Ein
        self.Ein_of_Eout = Ein_of_Eout
        self.dEout_dEin = dEout_dEin
        self.dEin_dEout = dEin_dEout
        self.fphi = heat_power_eq
        self.dfphi_dphi = der_heat_power_eq_dEout
        self.dfphi_dm = der_heat_power_eq_dm
        self.dfphi_dTs = der_heat_power_eq_dTs
        self.dfphi_dTr = der_heat_power_eq_dTr
        if unit_type == 'ge_gas_fired_gen_valve_point':
            self.f = node_law_Ein
            self.df_dEout = der_node_law_Ein_dEout
            self.df_dEin = der_node_law_Ein_dEin
            self.df_dTs = der_node_law_Ein_dTs
        elif unit_type == 'geh_CHP':
            self.f = node_law_Ein
            self.df_dEout = der_node_law_Ein_dEout
            self.df_dEin = der_node_law_Ein_dEin
            self.df_dTs = der_node_law_Ein_dTs
        elif unit_type == 'geh_CHP_part_load':
            self.f = node_law_Ein
            self.df_dEout = der_node_law_Ein_dEout
            self.df_dEin = der_node_law_Ein_dEin
            self.df_dTs = der_node_law_Ein_dTs
        else:
            self.f = node_law_Eout
            self.df_dEout = der_node_law_Eout_dEout
            self.df_dEin = der_node_law_Eout_dEin
            self.df_dTs = der_node_law_Eout_dTs

    def set_type(self,unit_type,unit_params,Ein_len=list(),Eout_len=list()):
        """Set or change the node unit type, and the corresponding node equations.

        Parameters
        ----------
        unit_type : str
            (New) type of the node, which determines the node law(s). Must be 'dummy', 'ge_gas_fired_gen', 'ge_gas_fired_gen_valve_point', 'gh_gas_boiler','gh_gas_boiler_part_load', 'eh_elec_boiler', 'geh_CHP','geh_CHP_part_load', or 'EH'.
        unit_params : dict
            Dictionary with parameters needed for the node law of a specific node type.
        Ein_len : np array, optional
            Array with number of incoming energies per carrier: [number of incoming gas energies, number of incoming active powers, number of incoming heat powers]. Default is an empty list
        Eout_len : np array, optional
            Array with number of outgoing energies per carrier: [number of outgoing gas energies, number of outgoing active powers, number of outgoing heat powers]. Default is an empty list

        Raises
        ------
        ValueError
            If the unit_type is not a valid unit type. I.e. if unit_type is not 'dummy', 'ge_gas_fired_gen', 'ge_gas_fired_gen_valve_point', 'gh_gas_boiler','gh_gas_boiler_part_load', 'eh_elec_boiler', 'geh_CHP','geh_CHP_part_load', or 'EH'
        IndexError
            If the unit_type is 'EH', and Ein_len and Eout_len are given, but the coupling matrix had the wrong dimensions.
        """
        self.unit_type = unit_type
        self.unit_params = unit_params
        self.Ein_len = Ein_len
        self.Eout_len = Eout_len
        if unit_type == 'dummy':
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr  = coupling.dummy()
        elif unit_type == 'ge_gas_fired_gen':
            eta = unit_params['eta']
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr  = coupling.ge_gas_fired_gen(eta)
        elif unit_type == 'ge_gas_fired_gen_valve_point':
            a = unit_params['a']
            b = unit_params['b']
            c = unit_params['c']
            d = unit_params['d']
            e = unit_params['e']
            Pmin = unit_params['Pmin']
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.ge_gas_fired_gen_valve_point(a,b,c,d,e,Pmin)
        elif unit_type == 'gh_gas_boiler':
            eta = unit_params['eta']
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.gh_gas_boiler(eta)
        elif unit_type == 'gh_gas_boiler_part_load':
            a = unit_params.get('a')
            b = unit_params.get('b')
            Ess = unit_params.get('Ess')
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.gh_gas_boiler_part_load(a,b,Ess)
        elif unit_type == 'eh_elec_boiler':
            eta = unit_params['eta']
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.eh_elec_boiler(eta)
        elif unit_type == 'geh_CHP':
            eta = unit_params['eta']
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.geh_CHP(eta)
        elif unit_type == 'geh_CHP_part_load':
            eta = unit_params.get('eta')
            a = unit_params.get('a')
            b = unit_params.get('b')
            d = unit_params.get('d')
            L1 = unit_params.get('L1')
            L2 = unit_params.get('L2')
            r1 = unit_params.get('r1')
            r2 = unit_params.get('r2')
            phimin = unit_params.get('phimin')
            phimax = unit_params.get('phimax')
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.geh_CHP_part_load(eta,a,b,d,L1,L2,r1,r2,phimin,phimax)
        elif unit_type == 'EH':
            C = unit_params.get('C')
            Ein_carriers = unit_params.get('Ein_carriers')
            Eout_carriers = unit_params.get('Eout_carriers')
            if len(Ein_len)>0 and len(Eout_len)>0:
                if not (C.shape[0] == np.sum(Ein_len) and C.shape[1] == np.sum(Eout_len)):
                    raise IndexError('coupling matrix C does not have right dimensions. It should be ({},{}), but it is {}'.format(np.sum(Ein_len),np.sum(Eout_len),C.shape))
            Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTs, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTs, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTs, der_heat_power_eq_dTr = coupling.EH(C,Ein_carriers,Eout_carriers)
        else:
            raise ValueError('unit_type is not a valid unit type.')
        self.Eout_of_Ein = Eout_of_Ein
        self.Ein_of_Eout = Ein_of_Eout
        self.dEout_dEin = dEout_dEin
        self.dEin_dEout = dEin_dEout
        self.fphi = heat_power_eq
        self.dfphi_dphi = der_heat_power_eq_dEout
        self.dfphi_dm = der_heat_power_eq_dm
        self.dfphi_dTs = der_heat_power_eq_dTs
        self.dfphi_dTr = der_heat_power_eq_dTr
        if unit_type == 'ge_gas_fired_gen_valve_point':
            self.f = node_law_Ein
            self.df_dEout = der_node_law_Ein_dEout
            self.df_dEin = der_node_law_Ein_dEin
            self.df_dTs = der_node_law_Ein_dTs
        elif unit_type == 'geh_CHP':
            self.f = node_law_Ein
            self.df_dEout = der_node_law_Ein_dEout
            self.df_dEin = der_node_law_Ein_dEin
            self.df_dTs = der_node_law_Ein_dTs
        elif unit_type == 'geh_CHP_part_load':
            self.f = node_law_Ein
            self.df_dEout = der_node_law_Ein_dEout
            self.df_dEin = der_node_law_Ein_dEin
            self.df_dTs = der_node_law_Ein_dTs
        else:
            self.f = node_law_Eout
            self.df_dEout = der_node_law_Eout_dEout
            self.df_dEin = der_node_law_Eout_dEin
            self.df_dTs = der_node_law_Eout_dTs

    def get_half_links(self,link_types=list(),bc_types=list(),carriers=list()):
        """Iterates over all the half links connected to the node.

        Parameters
        ----------
        link_types : list, optional
            List of link types of the links to be yielded. If empty, all the links are yielded. Default is an empty list.
        bc_types : list, optional
            List of bc types of the links to be yielded. If empty, all the links are yielded. Default is an empty list.
        carriers : list, optional
            List of carriers of the halflinks to be yielded. If empty, all the halflinks are yielded. Carriers are, 'gas', 'elec', or 'heat'

        Yields
        ------
        hl : HalfLink
            The next HalfLink instance in self.half_links
        """
        for hl in super().get_half_links(link_types=link_types,bc_types=bc_types):
            if carriers:
                if 'gas' in carriers and isinstance(hl,GasHalfLink):
                    yield hl
                elif 'elec' in carriers and isinstance(hl,ElectricalHalfLink):
                    yield hl
                elif 'heat' in carriers and isinstance(hl,HeatHalfLink):
                    yield hl
            else:
                yield hl

    def Eout(self,scale_var=None,scale_var_params=None):
        """Determine the energies (per carrier) on the outgoing links.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        Eout : np array
            Outgoing energies
        """
        gas_out = list()
        active_power_out = list()
        heat_out = list()
        for el in self.get_out_links():
            if isinstance(el,GasLink):
                GHV = self.unit_params.get('GHV')
                Eg = GHV*el.get_q() # unscaled
                if scale_var == 'per_unit':
                    Eb = scale_var_params.get('Ebase')
                    Eg = Eg/Eb
                gas_out.append(Eg)
            elif isinstance(el,GasHalfLink):
                if el.get_q() >= 0: # actually outgoing energy
                    GHV = self.unit_params.get('GHV')
                    Eg = GHV*el.get_q() # unscaled
                    if scale_var == 'per_unit':
                        Eb = scale_var_params.get('Ebase')
                        Eg = Eg/Eb
                    gas_out.append(Eg)
            elif isinstance(el,ElectricalLink):
                active_power_out.append(el.get_Pstart(scale_var=scale_var,scale_var_params=scale_var_params))
            elif isinstance(el,ElectricalHalfLink):
                if el.get_P() >= 0: # actually outgoing energy
                    active_power_out.append(el.get_P(scale_var=scale_var,scale_var_params=scale_var_params))
            elif isinstance(el,HeatLink): # -phi because it is a source, so phi and m will be <0 by definition.
                heat_out.append(-el.get_dphistart(scale_var=scale_var,scale_var_params=scale_var_params))
            elif isinstance(el,HeatHalfLink): # -phi because it is a source, so phi and m will be <0 by definition.
                heat_out.append(-el.get_dphi(scale_var=scale_var,scale_var_params=scale_var_params))
        if len(self.Eout_len) > 0:
            Eg = np.zeros(self.Eout_len[0])
            Ee = np.zeros(self.Eout_len[1])
            Eh = np.zeros(self.Eout_len[2])
            for ind_g,g in enumerate(gas_out):
                Eg[ind_g] = g
            for ind_e,e in enumerate(active_power_out):
                Ee[ind_e] = e
            for ind_h,h in enumerate(heat_out):
                Eh[ind_h] = h
            Eout = np.concatenate((Eg,Ee,Eh))
        else:
            Eout = np.array(gas_out+active_power_out+heat_out)
        return Eout

    def Ein(self,scale_var=None,scale_var_params=None):
        """Determine the energies (per carrier) on the incoming links.
        NB. Assumption is that there is no incoming heat power.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        Ein : np array
            Incoming energies.
        """
        gas_in = list()
        active_power_in = list()
        for el in self.get_in_links():
            if isinstance(el,GasLink):
                GHV = self.unit_params.get('GHV')
                Eg = GHV*el.get_q() # unscaled
                if scale_var == 'per_unit':
                    Eb = scale_var_params.get('Ebase')
                    Eg = Eg/Eb
                gas_in.append(Eg)
            elif isinstance(el,ElectricalLink):
                active_power_in.append(-el.get_Pend(scale_var=scale_var,scale_var_params=scale_var_params))
        for el in self.get_half_links(): # half links are defined as outgoing, but but actually have incoming energy
            if isinstance(el,GasHalfLink):
                if el.get_q() < 0: # actually incoming energy
                    GHV = self.unit_params.get('GHV')
                    Eg = GHV*-el.get_q() # unscaled
                    if scale_var == 'per_unit':
                        Eb = scale_var_params.get('Ebase')
                        Eg = Eg/Eb
                    gas_in.append(Eg)
            elif isinstance(el,ElectricalHalfLink):
                if el.get_P() < 0:
                    active_power_in.append(-el.get_P(scale_var=scale_var,scale_var_params=scale_var_params))
        if len(self.Ein_len) > 0:
            Eg = np.zeros(self.Ein_len[0])
            Ee = np.zeros(self.Ein_len[1])
            Eh = np.zeros(self.Ein_len[2])
            for ind_g,g in enumerate(gas_in):
                Eg[ind_g] = g
            for ind_e,e in enumerate(active_power_in):
                Ee[ind_e] = e
            Ein = np.concatenate((Eg,Ee,Eh))
        else:
            Ein = np.array(gas_in+active_power_in)
        return Ein

    def dEout_dE(self,scale_var=None,scale_var_params=None):
        """Determine the derivative of outgoing energies, to the different energy carriers (gas flow, active power, heat power).

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dEout_dE : np array
            Derivative of outgoing energies to carriers. Dimensions are len(Eout)x3
        """
        if len(self.Eout_len)>0:
            gas_out = np.zeros((self.Eout_len[0],3))
            active_power_out = np.zeros((self.Eout_len[1],3))
            heat_out = np.zeros((self.Eout_len[2],3))
        else:
            gas_out = np.array([],dtype=np.int64).reshape(0,3)
            active_power_out = np.array([],dtype=np.int64).reshape(0,3)
            heat_out = np.array([],dtype=np.int64).reshape(0,3)
        for el in self.get_out_links():
            if isinstance(el,GasLink):
                gas_out = np.zeros(3)
                GHV = self.unit_params.get('GHV')
                if scale_var == 'per_unit':
                    Eb = scale_var_params.get('Ebase')
                    qb = scale_var_params.get('qbase')
                    GHVb = Eb/qb
                    GHV = GHV/GHVb
                gas_out[0] = GHV
            elif isinstance(el,GasHalfLink):
                if el.get_q() >= 0: # actually outgoing energy
                    gas_out = np.zeros(3)
                    GHV = self.unit_params.get('GHV')
                    if scale_var == 'per_unit':
                        Eb = scale_var_params.get('Ebase')
                        qb = scale_var_params.get('qbase')
                        GHVb = Eb/qb
                        GHV = GHV/GHVb
                    gas_out[0] = GHV
            elif isinstance(el,ElectricalLink):
                active_power_out = np.zeros(3)
                active_power_out[1] = 1
            elif isinstance(el,ElectricalHalfLink):
                if el.get_P() >= 0: # actually outgoing energy
                    active_power_out = np.zeros(3)
                    active_power_out[1] = 1
            elif isinstance(el,HeatLink):
                heat_out = np.zeros(3)
                heat_out[2] = 1  # Should be a -1, because you use -phi everywhere? No, because you are also deriving to -phi!
            elif isinstance(el,HeatHalfLink):
                heat_out = np.zeros(3)
                heat_out[2] = 1 # Should be a -1, because you use -phi everywhere? No, because you are also deriving to -phi!
        return np.vstack((gas_out,active_power_out,heat_out))

    def dEin_dE(self,scale_var=None,scale_var_params=None):
        """Determine the derivative of incoming energies, to the different energy carriers (gas flow, active power, heat power).
        NB. Assumption is that there is no incoming heat power.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dEin_dE : np array
            Derivative of incoming energies to carriers. Dimensions are len(Ein)x3
        """
        if len(self.Ein_len)>0:
            gas_in = np.zeros((self.Ein_len[0],3))
            active_power_in = np.zeros((self.Ein_len[1],3))
            heat_in = np.zeros((self.Ein_len[2],3))
        else:
            gas_in = np.array([],dtype=np.int64).reshape(0,3)
            active_power_in = np.array([],dtype=np.int64).reshape(0,3)
            heat_in = np.array([],dtype=np.int64).reshape(0,3)
        for el in self.get_in_links():
            if isinstance(el,GasLink):
                gas_in = np.zeros(3)
                GHV = self.unit_params.get('GHV')
                if scale_var == 'per_unit':
                    Eb = scale_var_params.get('Ebase')
                    qb = scale_var_params.get('qbase')
                    GHVb = Eb/qb
                    GHV = GHV/GHVb
                gas_in[0] = GHV
            elif isinstance(el,ElectricalLink):
                active_power_in = np.zeros(3)
                active_power_in[1] = 1
        for el in self.get_half_links(): # half links are defined as outgoing, but but actually have incoming energy
            if isinstance(el,GasHalfLink):
                if el.get_q() < 0: # actually incoming energy
                    gas_in = np.zeros(3)
                    GHV = self.unit_params.get('GHV')
                    if scale_var == 'per_unit':
                        Eb = scale_var_params.get('Ebase')
                        qb = scale_var_params.get('qbase')
                        GHVb = Eb/qb
                        GHV = GHV/GHVb
                    gas_in[0] = GHV
            elif isinstance(el,ElectricalHalfLink):
                if el.get_P() < 0: # actually incoming energy
                    active_power_in = np.zeros(3)
                    active_power_in[1] = 1
        return np.vstack((gas_in,active_power_in,heat_in))

    def node_law(self,scale_var=None,scale_var_params=None):
        """Node law for a heterogeneous node.
        The node law is determined by the unit type.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            The sum of all incoming and outgoing energies.
        """
        if any((isinstance(el,HeatHalfLink) or isinstance(el,HeatLink)) for el in self.get_out_links()):
            for el in self.get_out_links():
                if isinstance(el,HeatLink):
                    Ts = el.get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params)
                if isinstance(el,HeatHalfLink):
                    Ts = el.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            Ts = None
        return self.f(self.Ein(scale_var=scale_var,scale_var_params=scale_var_params), self.Eout(scale_var=scale_var,scale_var_params=scale_var_params),Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def der_node_law_dE(self,scale_var=None,scale_var_params=None):
        """Derivative of the node law to all energy carriers.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dE : np array
            Derivative of the node law to all energy carriers: [df/dq df/dP df/dphi]
        """
        Ein = self.Ein(scale_var=scale_var,scale_var_params=scale_var_params)
        Eout = self.Eout(scale_var=scale_var,scale_var_params=scale_var_params)
        if any((isinstance(el,HeatHalfLink) or isinstance(el,HeatLink)) for el in self.get_out_links()):
            for el in self.get_out_links():
                if isinstance(el,HeatLink):
                    Ts = el.get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params)
                if isinstance(el,HeatHalfLink):
                    Ts = el.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            Ts = None
        return self.df_dEout(Ein,Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params).dot(self.dEout_dE(scale_var=scale_var,scale_var_params=scale_var_params)) + self.df_dEin(Ein,Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params).dot(self.dEin_dE(scale_var=scale_var,scale_var_params=scale_var_params))

    def der_node_law_dTs(self,scale_var=None,scale_var_params=None):
        """Derivative of the node law to supply temperature.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        df_dTs : np array
            Derivative of the node law to all energy carriers: [df/dq df/dP df/dphi]
        """
        Ein = self.Ein(scale_var=scale_var,scale_var_params=scale_var_params)
        Eout = self.Eout(scale_var=scale_var,scale_var_params=scale_var_params)
        if any(isinstance(el,HeatHalfLink) or isinstance(el,HeatLink) for el in self.get_out_links()):
            for el in self.get_out_links():
                if isinstance(el,HeatLink):
                    Ts = el.get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params)
                if isinstance(el,HeatHalfLink):
                    Ts = el.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            Ts = None
        return self.df_dTs(Ein,Eout,Ts,scale_var=scale_var,scale_var_params=scale_var_params)

    def heat_power_eq(self,scale_var=None,scale_var_params=None):
        """Heat power equation for a heterogeneous node.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        fphi : float
            Heat power equation.
        """
        if any((isinstance(el,HeatLink) or isinstance(el,HeatHalfLink)) for el in self.get_out_links()):
            for el in self.get_out_links():
                if isinstance(el,HeatHalfLink):
                    Cp = el.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
                    m = -el.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                    Eout = -el.get_dphi(scale_var=scale_var,scale_var_params=scale_var_params) # -phi because it is a source, so phi and m will be <0 by definition.
                    Ts = el.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params)
                    Tr = el.return_temp(scale_var=scale_var,scale_var_params=scale_var_params)
                elif isinstance(el,HeatLink): # outlinks, so start is connected to heterogeneous node
                    Cp = el.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
                    m = el.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                    Eout = -el.get_dphistart(scale_var=scale_var,scale_var_params=scale_var_params) # -phi because it is a source, so phi will be <0 by definition.
                    Ts = el.get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params)
                    Tr = el.get_Trstart(scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            Eout = None
            m = None
            Ts = None
            Tr = None
            Cp = None
        return self.fphi(Eout,m,Ts,Tr,Cp)

    def temp_diff_equation(self,scale_var=None,scale_var_params=None):
        """Temperature difference equation

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        fTo : float
            The temperature difference equation fTo(To,Ts) or fTo(Tr,To) or fTo(Ts,Tr)
        """
        if any(isinstance(el,HeatLink) or isinstance(el,HeatHalfLink) for el in self.get_out_links()):
            for el in self.get_out_links():
                if isinstance(el,HeatHalfLink):
                    fTo = el.temp_diff_equation(scale_var=scale_var,scale_var_params=scale_var_params)
                elif isinstance(el,HeatLink):
                    fTo = el.supply_temp_start(scale_var=scale_var,scale_var_params=scale_var_params) - el.return_temp_start(scale_var=scale_var,scale_var_params=scale_var_params) - el.get_dTstart(scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            fTo = None
        return fTo

    def der_heat_power_eq_dE(self,scale_var=None,scale_var_params=None):
        """Derivatives of heat power equation for a heterogeneous node.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        fphi_der : np array
            Derivatives of the heat power equations to all energy carriers: [dfphi/dm, dfphi/dphi, dfphi/dTo]
        """
        if any((isinstance(el,HeatHalfLink) or isinstance(el,HeatLink)) for el in self.get_out_links()):
            for el in self.get_out_links():
                if isinstance(el,HeatHalfLink):
                    Cp = el.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
                    m = -el.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                    Eout = -el.get_dphi(scale_var=scale_var,scale_var_params=scale_var_params) # -phi because it is a source, so phi and m will be <0 by definition.
                    Ts = el.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params)
                    Tr = el.return_temp(scale_var=scale_var,scale_var_params=scale_var_params)
                elif isinstance(el,HeatLink): # outlinks, so start is connected to heterogeneous node
                    Cp = el.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)
                    m = el.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                    Eout = -el.get_dphistart(scale_var=scale_var,scale_var_params=scale_var_params) # -phi because it is a source, so phi and m will be <0 by definition.
                    Ts = el.get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params)
                    Tr = el.get_Trstart(scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            Eout = None
            m = None
            Ts = None
            Tr = None
            Cp = None

        dEout_dphi = self.dEout_dE(scale_var=scale_var,scale_var_params=scale_var_params)[-1,-1]
        return np.array([self.dfphi_dm(Eout,m,Ts,Tr,Cp),self.dfphi_dphi(Eout,m,Ts,Tr,Cp)*dEout_dphi,self.dfphi_dTs(Eout,m,Ts,Tr,Cp)])
