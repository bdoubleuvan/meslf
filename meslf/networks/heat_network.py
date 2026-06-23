"""Heat network base class, including Network, Node, and Link"""
from meslf.networks.network import Network, Node, Link, HalfLink
from meslf.link_equations import hydraulic, thermal
import meslf.half_link_equations.thermal as halflink_thermal
from meslf.load_flow.system_of_equations import NonLinearSystemHeat
from meslf.load_flow.non_linear_solvers import NR, NR_FD, Fsolver, Root
import warnings
import numpy as np
import scipy.sparse as sps

# ===========================================================================
class HeatNetwork(Network):
    """Overall heat network class. Subclass of Network.

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
    A : scipy sparse csr_matrix
        Branch-nodal incidence matrix
    Ad : scipy sparse csr_matrix
        Adjacency matrix
    _Ta : float
        Ambient temperature.
    """
    def __init__(self,name,Ta=None):
        """Creates a HeatNetwork object.

        Parameters
        ----------
        name : str
            The name of the network.
        """
        super().__init__(name)
        self.A = None
        self.Ad = None
        self.Ta = Ta

    @property
    def Ta(self):
        """getter of Ta.
        """
        return self._Ta

    @Ta.setter
    def Ta(self,Ta):
        """setter of Ta.
        """
        self._Ta = Ta

    def get_Ta(self,scale_var=None,scale_var_params=None):
        """Get _Ta optionally with scaling

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling

        Returns
        -------
        Ta : float
            Possibly scaled attribute _Ta
        """
        Ta = self.Ta
        if Ta and scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            Ta = Ta/Tb
        return Ta

    def add_node(self,node,position=None):
        """Adds a node to the network. A node is only added if it is a HeatNode.

        Parameters
        ----------
        node : HeatNode
            Node object to be added to the network
        position : integer, optional
            Position (index) in the list of nodes of the network where the node should be inserted. Default is insert at end of list (append)

        Warns
        ------
        UserWarning
            If node is not an instance of HeatNode
        """
        if not isinstance(node,HeatNode):
            warnings.warn("Only a HeatNode object can be added")
        else:
            super().add_node(node,position=position)

    def add_link(self,link,position=None):
        """Adds link to the network.
        A link can only added if it is a HeatLink.
        If the start node or the end node are not yet added to the network, they
        will be added to the list of nodes.

        Parameters
        ----------
        link : HeatLink
            Link object to be added to the network

        Raises
        ------
        TypeError
            If link is not an instance of HeatLink
        """
        if not isinstance(link,HeatLink):
            raise TypeError("Only a HeatLink object can be added")
        else:
            super().add_link(link,position=position)
            if 'Ta' in link.link_params.keys():
                Ta_link = link.link_params['Ta']
                if self.Ta != None and (Ta_link != self.Ta):
                    warnings.warn("The HeatLink does not have the same ambient temperature as the network it is added to. The ambient temperature of the heat link is changed!")
                    link.link_params['Ta'] = self.Ta
                    link.set_type(link.link_type,link.link_params,hydr_eq_form=link.hydr_eq_form)
                elif Ta_link == None:
                    link.link_params['Ta'] = self.Ta
                    link.set_type(link.link_type,link.link_params,hydr_eq_form=link.hydr_eq_form)
            elif self.Ta != None:
                link.link_params['Ta'] = self.Ta
                link.set_type(link.link_type,link.link_params,hydr_eq_form=link.hydr_eq_form) # Not needed for all link types, but don't want to check for link types here.

    def add_half_link(self,half_link,position=None):
        """Adds half_link to the network.
        A half link can only added if it a HeatHalfLink.

        If the start node is not yet added to the network, it
        will be added to the list of nodes.

        Parameters
        ----------
        half_link : HeatHalfLink
            Half link object to be added to the network

        Raises
        ------
        TypeError
            If half_link is not an instance of HeatHalfLink
        """
        if not isinstance(half_link,HeatHalfLink):
            raise TypeError("Only a HeatHalfLink object can be added")
        else:
            super().add_half_link(half_link,position=position)

    def get_nodes(self,node_types=list()):
        """Iterates over all the nodes in the list of nodes, with the specified node type.

        Parameters
        ----------
        node_types : list, optional
            List of node type of the nodes to be yielded. If empty, all the nodes are yielded. Default is an empty list.

        Yields
        ------
        n : Node
            The next Node instance in self.nodes
        """
        for n in super().get_nodes():
            if node_types:
                if n.node_type in node_types:
                    yield n
            else:
                yield n
    
    def get_half_links(self,link_types=list(),bc_types=list()):
        """Iterates over all the half links in the list of links, with the specified link type or a specified boundary condition.

        Parameters
        ----------
        link_types : list, optional
            List of link types of the halflinks to be yielded. If empty, all the halflinks are yielded. Default is an empty list.
        bc_types : list, optional
            List of boundary condition types of the halflinks to be yielded. If empty, all the halflinks are yielded. Default is an empty list.

        Yields
        ------
        hl : Link
            The next Link instance in self.links
        """
        for hl in super().get_half_links():
            if link_types and bc_types:
                if hl.link_type in link_types and hl.bc_type in bc_types:
                    yield hl
            elif link_types:
                if hl.link_type in link_types:
                    yield hl
            elif bc_types:
                if hl.bc_type in bc_types:
                    yield hl
            else:
                yield hl

    def add_network(self,network):
        """ Adds network to the network.
        A network can only be added if it a HeatNetwork.

        Parameters
        ----------
        network : HeatNetwork
            The network to be added

        Raises
        ------
        TypeError
            If network is not a HeatNetwork instance
        """
        if not isinstance(network,HeatNetwork):
            raise TypeError("Only a HeatNetwork object can be added")
        else:
            super().add_network(network)

    def make_incidence_matrix(self):
        """Creates the branch-nodal incidence matrix A.
        Assigns a number to all nodes and links.
        """
        row = []
        col = []
        data = []
        for ind_n,n in enumerate(self.get_nodes()):
            n.number = ind_n
        for ind_e,e in enumerate(self.get_links()):
            e.number = ind_e
            #outgoing edge
            if e.start_node in self.get_nodes(): # check if start_node is in the network, i.e. if it is a HeatNode
                row.append(e.start_node.number)
                col.append(e.number)
                data.append(-1.)
            #incoming edge
            if e.end_node in self.get_nodes(): # check if end_node is in the network, i.e. if it is a HeatNode
                row.append(e.end_node.number)
                col.append(e.number)
                data.append(1.)
        self.A = sps.csr_matrix((data,(row,col)),shape=(len(list(self.get_nodes())),len(list(self.get_links()))))

    def actual_incidence_matrix(self):
        """Branch-nodal incidence matrix A of the heat network based on the actual direction of flow.

        Returns
        -------
        Aac : scipy sparse csr_matrix
            Branch-nodal incidence matrix based on actual direction of flow.
        Aac_neg : scipy sparse csr_matrix
            Aac with positive entries set to 0.
        Aac_pos : scipy sparse csr_matrix
            Aac with negative entries set to 0.
        """
        m_dir_vec = np.zeros(len(self.links))
        for ind_e, e in enumerate(self.get_links()):
            m_dir_vec[ind_e] = np.sign(e.m)
        Aac = self.A.dot(sps.diags(m_dir_vec))
        Aac_neg = Aac.copy()
        Aac_neg[Aac>0]=0.
        Aac_pos = Aac.copy()
        Aac_pos[Aac<0]=0.
        return Aac, Aac_neg, Aac_pos

    def make_adjacency_matrix(self):
        """Create the adjacency matrix Ad. It assumes / uses the same numbering as assigned by make_incidence_matrix(self). This means that the incidence matrix must be made before the adjacency matrix can be made"""
        row = []
        col = []
        data = []
        for ind_n, n in enumerate(self.get_nodes()):
            for e in n.get_out_links(): # only a non-zero element for out links
                if isinstance(e,HeatLink): # HalfLinks should not be taken into account
                    row.append(e.start_node.number)
                    col.append(e.end_node.number)
                    data.append(1.)
        self.Ad = sps.csr_matrix((data,(row,col)),shape=(len(list(self.get_nodes())),len(list(self.get_nodes()))))

    def initialize(self):
        """Inializes the network.
        The branch-nodal incidence matrix, and the adjacency matrix for the heat network are made. Also, all the half links are added to the network, and the bc type of the half link is checked against the node type. The bc type is changed if needed.
        """
        self.make_incidence_matrix()
        self.make_adjacency_matrix()
        for node in self.get_nodes():
            for hl in node.get_half_links():
                self.add_half_link(hl)

    def get_x_entries(self,formulation='standard'):
        """Return all the nodes, links, and half links that have an entry in variable vector x.
        It also return all the links with unknown flow, and all the nodes with unknown pressure, all the nodes with unknown supply temperature, and all the nodes with unknown return temperature.

        Parameters
        ----------
        formulation : string, optional
            Formulation used to form system of equations. Default is 'standard'. Options are 'standard' and 'half_link_flow'.

        Returns
        -------
        x_entries : list
           List of all the nodes, links, and half links that contribute to x.
        unknown_m_links : list
            List of all the links that have unknown link flow.
        unknown_m_halflinks : list
            List of all the half links that have unknown half link flow, and are connected to a non slack node. Only used if formulation is 'half_link_flow'.
        unknown_p_nodes : list
            List of all the nodes that have unknown pressure.
        unknown_Ts_nodes : list
            List of all the nodes that have unknown supply temperature.
        unknown_Tr_nodes : list
            List of all the nodes that have unknown return temperature.
        unknown_Ts_halflinks : list
            List of all the half links that have unknown supply temperature.    
        unknown_Tr_halflinks : list
            List of all the half links that have unknown return temperature. 
        """
        unknown_m_links = [link for link in self.get_links() if (isinstance(link.start_node,HeatNode) and isinstance(link.end_node,HeatNode))] # non-coupling links
        unknown_p_nodes = list(self.get_nodes([1,2,4,6,12,14,15,16]))
        unknown_Ts_nodes = list(self.get_nodes([1,2,3,5,8,9,11,12,13,15,16]))
        unknown_Tr_nodes = list(self.get_nodes([0,1,2,3,4,5,6,7,9,10,12,13,14]))
        if formulation == 'half_link_flow':
            # half links are not taken into variable vector for all node types, even if m, Ts, or Tr is unknown
            unknown_m_halflinks = [hl for node in self.get_nodes(node_types=[1,3,4,12,13,14,15,16]) for hl in node.get_half_links()]
            unknown_Ts_halflinks = [hl for node in self.get_nodes(node_types=[1,3,4,12,13,14,15,16])for hl in node.get_half_links(bc_types=[0,4,8,10,12,16,20,22])]
            unknown_Tr_halflinks = [hl for node in self.get_nodes(node_types=[1,3,4,12,13,14,15,16])for hl in node.get_half_links(bc_types=[1,5,9,11,13,17,21,23])]
        else:
            unknown_m_halflinks = list()
            unknown_Ts_halflinks = list()
            unknown_Tr_halflinks = list()
        x_entries = unknown_m_links + unknown_m_halflinks + unknown_p_nodes + unknown_Ts_nodes + unknown_Tr_nodes + unknown_Ts_halflinks + unknown_Tr_halflinks
        return x_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks

    def get_F_entries(self,formulation='standard'):
        """Return all the nodes, links, and half links that have an entry in function vector F.
        It also returns the nodes that have an entry in conservation of, all the links that have an entry in the link equations, all the nodes that have an entry in the supply temperature mixing rule, and all the nodes that have an entry in the return temperature mixing rule.

        Parameters
        ----------
        formulation : string, optional
            Formulation used to form system of equations. Default is 'standard'. Options are 'standard' and 'half_link_flow'.

        Returns
        -------
        F_entries : list
           List of all the nodes, links, and half links that contribute to F.
        F_m_nodes : list
            List of all the nodes that contribute to conservation of mass.
        F_deltap_links : list
            List of all the links that contribute to link equations.
        F_Ts_nodes : list
            List of all the nodes that contribute to supply temperature mixing rule.
        F_Tr_nodes : list
            List of all the nodes that contribute to return temperature mixing rule.
        F_phi_nodes : list
            List of all the half links that contribute to a heat power equation. Only used if formulation is 'half_link_flow'.
        F_dT_nodes : list
            List of all the half links that contribute to a temperature difference equation. Only used if formulation is 'half_link_flow'.
        """
        F_m_nodes = list(self.get_nodes([1,2,3,4,5,6,7,12,13,14,15,16]))
        F_deltap_links = [link for link in self.get_links() if ((not link.link_type == 'dummy') and isinstance(link.start_node,HeatNode) and isinstance(link.end_node,HeatNode))] # non-coupling links and non-dummy links
        F_Ts_nodes = list(self.get_nodes([1,2,3,4,5,6,7,8,9,11,12,13,14,15,16]))
        F_Tr_nodes = list(self.get_nodes([0,1,2,3,4,5,6,7,9,10,12,13,14,15,16]))
        if formulation == 'half_link_flow':
            F_phi_halflinks = [hl for hl in self.get_half_links(link_types='heat_exchanger',bc_types=[2,3,4,5,8,9,14,15,16,17,20,21]) if hl.start_node.node_type in [1,3,4,12,13,14,15,16]]
            F_dT_halflinks = [hl for hl in self.get_half_links(link_types='heat_exchanger',bc_types=[4,5,10,11,16,17,22,23]) if hl.start_node.node_type in [12,13,14,15,16]]
        else:
            F_phi_halflinks = list()
            F_dT_halflinks = list()
        F_entries = F_m_nodes + F_deltap_links + F_Ts_nodes + F_Tr_nodes + F_phi_halflinks + F_dT_halflinks
        return F_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_dT_halflinks

    def set_x_init(self,formulation='standard',scale_var=None,scale_var_params=None,**kwargs):
        """Creates the (initial guess for) vector x, based on the current network parameters.

        Parameters
        ----------
        formulation : string, optional
            Formulation used to form system of equations. Default is 'standard'. Options are 'standard' and 'half_link_flow'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        x0 : np array
           Initial guess for variable vector x
        """
        T_shift = None#self.get_Ta(scale_var=scale_var,scale_var_params=scale_var_params)
        x_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks = self.get_x_entries(formulation=formulation)
        x0 = np.zeros(len(x_entries))
        for ind_el,el in enumerate(unknown_m_links):
            x0[ind_el] = el.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
        for ind_el,el in enumerate(unknown_m_halflinks):
            x0[ind_el+len(unknown_m_links)] =  el.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
        for ind_el,el in enumerate(unknown_p_nodes):
            x0[ind_el+len(unknown_m_links)+len(unknown_m_halflinks)] = el.get_p(scale_var=scale_var,scale_var_params=scale_var_params)
        for ind_el,el in enumerate(unknown_Ts_nodes):
            x0[ind_el+len(unknown_m_links)+len(unknown_m_halflinks)+len(unknown_p_nodes)] = el.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        for ind_el,el in enumerate(unknown_Tr_nodes):
            x0[ind_el+len(unknown_m_links)+len(unknown_m_halflinks)+len(unknown_p_nodes)+len(unknown_Ts_nodes)] = el.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        for ind_el,el in enumerate(unknown_Ts_halflinks):
            x0[ind_el+len(unknown_m_links)+len(unknown_m_halflinks)+len(unknown_p_nodes)+len(unknown_Ts_nodes)+len(unknown_Tr_nodes)] =  el.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        for ind_el,el in enumerate(unknown_Tr_halflinks):
            x0[ind_el+len(unknown_m_links)+len(unknown_m_halflinks)+len(unknown_p_nodes)+len(unknown_Ts_nodes)+len(unknown_Tr_nodes)+len(unknown_Ts_halflinks)] =  el.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        return x0

    def update(self,x,formulation='standard',scale_var=None,scale_var_params=None,**kwargs):
        """Updates the network given variable vector x

        Parameters
        ----------
        x : np array
            Variable vector x, scaled.
        formulation : string, optional
            Formulation used to form system of equations. Default is 'standard'. Options are 'standard' and 'half_link_flow'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        """
        T_shift = None#self.get_Ta(scale_var=scale_var,scale_var_params=scale_var_params)
        if not self.networks: # network does not consist of subnetworks; update the network
            x_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks = self.get_x_entries(formulation=formulation)
            if not len(x) == len(x_entries):
                raise ValueError('x has the wrong length')
            else:
                for ind_l,l in enumerate(unknown_m_links):
                    m = x[ind_l]
                    if scale_var == 'per_unit':
                        m *= scale_var_params['qbase']
                    l.m = m
                for ind_hl,hl in enumerate(unknown_m_halflinks):
                    m = x[ind_hl + len(unknown_m_links)]
                    if scale_var == 'per_unit':
                        m *= scale_var_params['qbase']
                    hl.m = m
                for ind_n,n in enumerate(unknown_p_nodes):
                    p = x[ind_n + len(unknown_m_links) + len(unknown_m_halflinks)]
                    if scale_var == 'per_unit':
                        p *= scale_var_params['pbase']
                    n.p = p
                for ind_n,n in enumerate(unknown_Ts_nodes):
                    Ts = x[ind_n + len(unknown_m_links) + len(unknown_m_halflinks) + len(unknown_p_nodes)]
                    if not T_shift == None:
                        Ts += T_shift
                    if scale_var == 'per_unit':
                        Ts *= scale_var_params['Tbase']
                    if Ts > n.Ts_max:
                        warnings.warn('For node {}, Ts = {:.4f} which is larger than Ts_max = {:.4f}.'.format(n.name,Ts,n.Ts_max))
                    elif Ts < n.Ts_min:
                        warnings.warn('For node {}, Ts = {:.4f} which is smaller than Ts_min = {:.4f}.'.format(n.name,Ts,n.Ts_min))
                    n.Ts = Ts
                for ind_n,n in enumerate(unknown_Tr_nodes):
                    Tr = x[ind_n + len(unknown_m_links) + len(unknown_m_halflinks) + len(unknown_p_nodes) + len(unknown_Ts_nodes)]
                    if not T_shift == None:
                        Tr += T_shift
                    if scale_var == 'per_unit':
                        Tr *= scale_var_params['Tbase']
                    if Tr > n.Tr_max:
                        warnings.warn('For node {}, Tr = {:.4f} which is larger than Tr_max = {:.4f}.'.format(n.name,Tr,n.Tr_max))
                    elif Tr < n.Tr_min:
                        warnings.warn('For node {}, Tr = {:.4f} which is smaller than Tr_min = {:.4f}.'.format(n.name,Tr,n.Tr_min))
                    n.Tr = Tr
                    if n.Tr > n.Ts:
                        warnings.warn('For node {}, Tr = {:.4f} and Ts = {:.4f}, such that Tr > Ts.'.format(n.name,n.Tr,n.Ts))
                for ind_hl,hl in enumerate(unknown_Ts_halflinks):
                    Ts = x[ind_hl + len(unknown_m_links) + len(unknown_m_halflinks) + len(unknown_p_nodes) + len(unknown_Ts_nodes) + len(unknown_Tr_nodes)]
                    if scale_var == 'per_unit':
                        Ts *= scale_var_params['Tbase']
                    hl.Ts = Ts
                for ind_hl,hl in enumerate(unknown_Tr_halflinks):
                    Tr = x[ind_hl + len(unknown_m_links) + len(unknown_m_halflinks) + len(unknown_p_nodes) + len(unknown_Ts_nodes) + len(unknown_Tr_nodes) + len(unknown_Ts_halflinks)]
                    if scale_var == 'per_unit':
                        Tr *= scale_var_params['Tbase']
                    hl.Tr = Tr
                    
                # update start and end temperatures, and start and end heat power
                for ind_e,e in enumerate(self.get_links()):
                    if (isinstance(e.start_node,HeatNode) and isinstance(e.end_node,HeatNode)): # non-coupling links
                        # the temperature at the upstream side needs to be updated first, otherwise the flow at the downstream side is wrong
                        if e.m >= 0:
                            e.Tsstart = e.start_node.Ts
                            e.Trend = e.end_node.Tr
                            Trstart = e.return_temp_start(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                            Tsend = e.supply_temp_end(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                            if scale_var == 'per_unit':
                                Trstart *= scale_var_params['Tbase']
                                Tsend *= scale_var_params['Tbase']
                            e.Trstart = Trstart
                            e.Tsend = Tsend
                        else:
                            e.Tsend = e.end_node.Ts
                            e.Trstart = e.start_node.Tr
                            Tsstart = e.supply_temp_start(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                            Trend = e.return_temp_end(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                            if scale_var == 'per_unit':
                                Tsstart *= scale_var_params['Tbase']
                                Trend *= scale_var_params['Tbase']
                            e.Tsstart = Tsstart
                            e.Trend = Trend
                        phisstart = e.supply_heat_power_start(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                        phirstart = e.return_heat_power_start(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                        phisend = e.supply_heat_power_end(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                        phirend = e.return_heat_power_end(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                        if scale_var == 'per_unit':
                            phisstart *= scale_var_params['phibase']
                            phirstart *= scale_var_params['phibase']
                            phisend *= scale_var_params['phibase']
                            phirend *= scale_var_params['phibase']
                        e.phisstart = phisstart
                        e.phirstart = phirstart
                        e.phisend = phisend
                        e.phirend = phirend
            # update flow, heat, Ts, and Tr on half links
            slack_nodes = [0,8,9,10,11]
            general_slack_nodes = [9,10,11]
            for hl in self.get_half_links():
                n = hl.start_node
                # determine mass flow for slack nodes
                if (n.node_type in slack_nodes) and (hl not in unknown_m_halflinks): # slack nodes
                    if hl.link_type == 'dummy':
                        raise TypeError('Half link of node {} is of type "dummy", phi cannot be updated'.format(n.name))
                    else:
                        m = 0.
                        for l in hl.start_node.get_in_links():
                            if not isinstance(l,HeatHalfLink):
                                m += l.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                        for l in hl.start_node.get_out_links():
                            if not isinstance(l,HeatHalfLink):
                                m -= l.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                        if scale_var == 'per_unit':
                            m *= scale_var_params['qbase']
                        hl.m = m
                if hl.bc_type in [4,5,10,11,16,17,22,23] and hl.dT <= 0:
                    raise ValueError('Half link has dT known, but dT should be positive, not {}'.format(hl.dT))
                    
                if formulation == 'standard':
                    # update temperatures, assuming some are sources and some are sinks. 
                    if hl.source: # sources
                        hl.Tr = n.Tr
                        if n.node_type in slack_nodes: # slack nodes
                            hl.Ts = n.Ts
                        elif hl.bc_type in [4,5,10,11,16,17,22,23]:  # dT known
                            hl.Ts = hl.dT + hl.Tr
                    else: # sinks
                        hl.Ts = n.Ts
                        if n.node_type in slack_nodes: # slack nodes
                            hl.Tr = n.Tr
                        elif hl.bc_type in [4,5,10,11,16,17,22,23]:  # dT known
                            hl.Tr = hl.Ts - hl.dT
                
                    if n.node_type not in slack_nodes: # non slack nodes
                        if hl.bc_type in [0,1,6,7,10,11,12,13,18,19,22,23]: # dphi unknown
                            raise ValueError("Encountered a half link with unknown dphi. In the standard formulation, all half links connected to non slacknodes should have known heat power loss!")
                        m = hl.flow(scale_var=scale_var,scale_var_params=scale_var_params) 
                        if scale_var == 'per_unit':
                            m *= scale_var_params['qbase']
                        hl.m = m
                        
                elif formulation == 'half_link_flow':
                    # update temperatures, based on direction of flow
                    if n.node_type in general_slack_nodes:
                        if (hl.bc_type not in [2,6,14,18]) and (hl not in unknown_Ts_halflinks): # Otherwise, Ts is already known for these halflinks, or is part of x
                            hl.Ts = n.Ts
                        if (hl.bc_type not in [3,7,15,19]) and (hl not in unknown_Tr_halflinks): # Otherwise, Tr is already known for these halflinks, or is part of x
                            hl.Tr = n.Tr
                    elif n.node_type in slack_nodes: # slack nodes
                        if hl not in unknown_Ts_halflinks:
                            hl.Ts = n.Ts
                        if hl not in unknown_Tr_halflinks:
                            hl.Tr = n.Tr
                    elif hl.sink: # sinks
                        if (hl.bc_type not in [2,6,14,18]) and (hl not in unknown_Ts_halflinks): # Otherwise, Ts is already known for these halflinks, or is part of x
                            hl.Ts = n.Ts
                        if (hl.bc_type in [4,5,10,11,16,17,22,23]) and (hl not in unknown_Tr_halflinks): # dT known (Ts is updated in the previous if-statement)
                            hl.Tr = hl.Ts - hl.dT
                    else: # sources
                        if (hl.bc_type not in [3,7,15,19]) and (hl not in unknown_Tr_halflinks): # Otherwise, Tr is already known for these halflinks, or is part of x
                            hl.Tr = n.Tr
                        if (hl.bc_type in [4,5,10,11,16,17,22,23]) and (hl not in unknown_Ts_halflinks): # dT known (Tr is updated in the previous if-statement)
                            hl.Ts = hl.dT + hl.Tr        
                else:
                    raise NotImplementedError("update() not implemented for formulation {}".format(formulation))
                    
                # update heat powers
                if hl.bc_type in [0,1,6,7,10,11,12,13,18,19,22,23]: # dphi unknown
                    dphi = hl.heat(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                    if scale_var == 'per_unit':
                        dphi *= scale_var_params['phibase']
                    hl.dphi = dphi
                phis = hl.supply_heat_power(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                phir = hl.return_heat_power(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                if scale_var == 'per_unit':
                    phis *= scale_var_params['phibase']
                    phir *= scale_var_params['phibase']
                hl.phis = phis
                hl.phir = phir
        else:
            x_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks = self.get_x_entries(formulation=formulation)
            for net in self.get_networks():
                indices_m = list()
                indices_m_hl = list()
                indices_p = list()
                indices_Ts = list()
                indices_Tr = list()
                indices_To = list()
                x_entries_net, unknown_m_links_net, unknown_m_halflinks_net, unknown_p_nodes_net, unknown_Ts_nodes_net, unknown_Tr_nodes_net, unknown_To_halflinks_net = net.get_x_entries(formulation=formulation)
                for ind_el,el in unknown_m_links_net:
                    indices_m.append(unknown_m_links.index(el))
                for ind_el,el in unknown_m_halflinks_net:
                    indices_m_hl.append(unknown_m_halflinks.index(el))
                for ind_el,el in unknown_p_nodes_net:
                    indices_p.append(unknown_p_nodes.index(el))
                for ind_el,el in unknown_Ts_nodes_net:
                    indices_Ts.append(unknown_Ts_nodes.index(el))
                for ind_el,el in unknown_Ts_nodes_net:
                    indices_Tr.append(unknown_Tr_nodes.index(el))
                for ind_el,el in unknown_Ts_halflinks_net:
                    indices_Ts_hl.append(unknown_Ts_halflinks.index(el))
                for ind_el,el in unknown_Tr_halflinks_net:
                    indices_Tr_hl.append(unknown_Tr_halflinks.index(el))
                x_net = x[indices_m + indices_m_hl + indices_p + indices_Ts + indices_Tr + indices_Ts_hl + indices_Tr_hl]
                # update subnetworks
                net.update(x_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    def update_full(self,x,formulation='standard',scale_var=None,scale_var_params=None,**kwargs):
        """Updates the full network given variable vector x.
        Unlike update(x), not only the values from x are updated, but also all
        parameters not included in x.

        Parameters
        ----------
        x : np array
            Variable vector x, scaled.
        formulation : string, optional
            Formulation used to form system of equations. Default is 'standard'. Options are 'standard' and 'half_link_flow'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        m_vec : np array
            Array with all link flows
        p_vec : np array
            Array with all unscaled nodal pressures
        Ts_vec : np array
            Array with all unscaled nodal supply temperatures
        Tr_vec : np array
            Array with all unscaled nodal return temperatures
        m_hl_vec : list
            List of arrays with all unscaled half link flows per node
        dphi_hl_vec : list
            List of array with all unscaled half link heat powers per node
        Ts_hl_vec : list
            List of array with all unscaled half link temperatures near supply line per node
        Tr_hl_vec : list
            List of array with all unscaled half link temperatures near return line per node
        """
        self.update(x,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        m_vec = np.zeros(len(list(self.get_links())))
        p_vec = np.zeros(len(list(self.get_nodes())))
        Ts_vec = np.zeros(len(list(self.get_nodes())))
        Tr_vec = np.zeros(len(list(self.get_nodes())))
        m_hl_vec = []
        dphi_hl_vec = []
        Ts_hl_vec = []
        Tr_hl_vec = []
        for ind_n, n in enumerate(self.get_nodes()):
            p_vec[ind_n] = n.p
            Ts_vec[ind_n] = n.Ts
            Tr_vec[ind_n] = n.Tr
            m_hl_node = []
            dphi_hl_node = []
            Ts_hl_node = []
            Tr_hl_node = []
            for hl in n.get_half_links():
                # get variables for every component (i.e. halflink) connected to node n
                m_hl_node.append(hl.m)
                dphi_hl_node.append(hl.dphi)
                Ts_hl_node.append(hl.Ts)
                Tr_hl_node.append(hl.Tr)
            m_hl_vec.append(m_hl_node)
            dphi_hl_vec.append(dphi_hl_node)
            Ts_hl_vec.append(Ts_hl_node)
            Tr_hl_vec.append(Tr_hl_node)
        for ind_e,e in enumerate(self.get_links()):
            m_vec[ind_e] = e.m
        return m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec

    def reset_network(self,x_init,formulation='standard',scale_var=None,scale_var_params=None,**kwargs):
        """Resets the full network to initial conditions given initial guess vector x.
        Also, any half links for reference slack nodes are removed.

        Parameters
        ----------
        x_init : np array
            Vector with initial guess for x.
        formulation : string, optional
            Formulation used to form system of equations. Default is 'standard'. Options are 'standard' and 'half_link_flow'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        """
        self.update_full(x_init,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        for ind_n,n in enumerate(self.get_nodes([0,8])):
            if n.half_links:
                for hl in n.get_half_links():
                    hl.m = 0

    def solve_network(self,tol,max_iter,*args,formulation='standard',scale_var=None,scale_var_params=None,solver='NR',D_F=np.array([]),D_x=np.array([]),P_F=np.array([]),P_x=np.array([]),det_tol=None,return_all_x=False,lin_solver='solve',max_iter_lin=None,**kwargs):
        """Solves the steady-state load flow problem for the heat network.

        Parameters
        ----------
        tol : float
            Tolerance for solver.
        max_iter : int
            Maximum number of iterations of solver.
        formulation : string, optional
            Formulation used to form system of equations. Default is 'standard'. Options are 'standard' and 'half_link_flow'.
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

        Returns
        -------
        x_sol : np array
            Solution vector x, scaled.
        iters : int
            Number of iterations used.
        err_vec : list
            List with the error for every iteration.
        m_vec : np array
            Array with all unscaled link flows.
        p_vec : np array
            Array with all unscaled nodal pressures.
        Ts_vec : np array
            Array with all unscaled nodal supply temperatures.
        Tr_vec : np array
            Array with all unscaled nodal return temperatures.
        m_hl_vec : list
            List of arrays with all unscaled half link flows per node.
        phi_hl_vec : list
            List of array with all unscaled half link heat powers per node.
        To_hl_vec : list
            List of array with all unscaled half link outflow temperatures per node.
        """
        x0 = self.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        nlsys = NonLinearSystemHeat(self,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
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
        else:
            raise ValueError("solver should be either 'NR', 'NR_FD', 'f_solver', or 'root', not '{}'.".format(solver))
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
            x_sol = nrsolve.solve(nlsys,x0,max_iter,D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x)
        iters =nrsolve.iters
        err_vec = nrsolve.err_vec
        # update the rest of the network
        m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec = self.update_full(x_sol,formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        if return_all_x and solver == 'NR':
            return x_sol,x_mat,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec
        else:
            return x_sol,iters,err_vec,m_vec,p_vec,Ts_vec,Tr_vec,m_hl_vec,dphi_hl_vec,Ts_hl_vec,Tr_hl_vec

# ===========================================================================
class HeatNode(Node):
    """Heat node class.

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
        Type of the node, 0 = source slack node, 1 = source/sink node, 2 = junction node, 3 = source/sink ref. node, 4 = source temp. node, 5 = ref. node, 6 = temp. node, 7= ref. temp. node, 8 = sink slack node, 9 = slack node, 10 = slack supply temp. node, 11 = slack return temp. node, 12 = source/sink temp. dif. node, 13 = source/sink ref. temp. dif. node, 14 = source supply temp. temp. dif. node, 15 = sink return temp. temp. dif. node, 16 = sink temp. node.
    _Ts : float
        Nodal supply temperature, unscaled.
    _Tr : float
        Nodal return temperature, unscaled.
    _p : float
        Nodal pressure, unscaled.
    Ts_max : float
        Upper bound for supply temperature, unscaled.
    Ts_min : float
        Lower bound for supply temerature, unscaled. 
    Tr_max : float
        Upper bound for return temperature, unscaled.
    Tr_min : float
        Lower bound for return temerature, unscaled.
    """
    def __init__(self,name,x=0.,y=0.,node_type=0,Ts=100.,Tr=0.,p=0.,dphi=None,Ts_hl=None,Tr_hl=None,dT=None,Ts_max=500,Ts_min=-273.15,Tr_max=500,Tr_min=-273.15):
        """Creates an HeatNode object.

        Parameters
        ----------
        name : str
            Name of the node.
        x : float, optional
            x coordinate of the node. Default is 0.
        y : float, optional
            y coordinate of the node. Default is 0.
        node_type : int
            Type of the node, 0 = source slack node, 1 = source/sink node, 2 = junction node, 3 = source/sink ref. node, 4 = source temp. node, 5 = ref. node, 6 = temp. node, 7= ref. temp. node, 8 = sink slack node, 9 = slack node, 10 = slack supply temp. node, 11 = slack return temp. node, 12 = source/sink temp. dif. node, 13 = source/sink ref. temp. dif. node, 14 = source supply temp. temp. dif. node, 15 = sink return temp. temp. dif. node, 16 = sink temp. node.
        Ts : float
            Nodal supply temperature, unscaled.
        Tr : float
            Nodal return temperature, unscaled.
        p : float
            Nodal pressure, unscaled.
        dphi : float, optional
            Total injected heat power, unscaled. One half link is made with this heat power. Default is None. 
        Ts_hl : float, optional
            Outflow temperature of 'total' component between supply and return line (or the other way around), at the supply line side, unscaled. One half links is made with this temperature. Default is None. 
        Tr_hl : float, optional
            Outflow temperature of 'total' component between supply and return line (or the other way around), at the supply line side, unscaled. One half links is made with this temperature. Default is None. 
        dT : float, optional
            Temperature difference over half link, unscaled and not shifted. Default is None.
        Ts_max : float, optional
            Upper bound for supply temperature, unscaled. Default is 500 C
        Ts_min : float, optional
            Lower bound for supply temperature, unscaled. Default is -273.15 C
        Tr_max : float, optional
            Upper bound for return temperature, unscaled. Default is 500 C
        Tr_min : float, optional
            Lower bound for return temperature, unscaled. Default is -273.15 C
        """
        super().__init__(name,x=x,y=y)
        self.node_type = node_type
        self.Ts = Ts
        self.Ts_max = Ts_max
        self.Ts_min = Ts_min
        self.Tr = Tr
        self.Tr_max = Tr_max
        self.Tr_min = Tr_min
        self.p = p
        hl_name = name + "_hl"
        if (dphi != None) and (Ts_hl != None): # Need to check if not None, because if phi=0 or To=0 a halflink should still be made
            HeatHalfLink(hl_name,self,dphi=dphi,Ts=Ts_hl,bc_type=2)
        elif (dphi != None) and (Tr_hl != None):
            HeatHalfLink(hl_name,self,dphi=dphi,Tr=Tr_hl,bc_type=3)
        elif (dphi != None) and (dT != None):
            if dphi < 0: # source
                HeatHalfLink(hl_name,self,dphi=dphi,dT=dT,bc_type=4)
            else:
                HeatHalfLink(hl_name,self,dphi=dphi,dT=dT,bc_type=5)
        elif Ts_hl != None:
            HeatHalfLink(hl_name,self,Ts=Ts_hl,bc_type=6)
        elif Tr_hl != None:
            HeatHalfLink(hl_name,self,Tr=Tr_hl,bc_type=7)
        elif dT != None:
            raise NotImplementedError("bc type of half link cannot be determined. Don't know if node, with node type {}, is source or sink.".format(self.node_type))
            #HeatHalfLink(hl_name,self,dT=dT,bc_type=10)
        elif node_type == 0: # source reference slack node
            HeatHalfLink(hl_name,self,dphi=-1.,Ts=Ts,bc_type=0)
        elif node_type == 8: # sink reference slack node
            HeatHalfLink(hl_name,self,dphi=1.,Tr=Tr,bc_type=1)
        elif node_type in [9,10]: # general slack nodes (intialize as source, since these correspond more with a source reference slack node)
            HeatHalfLink(hl_name,self,dphi=-1.,Ts=Ts,bc_type=0)
        elif node_type == 11: # general slack nodes (intialize as sink, since it corresponds more with a sink reference slack node)
            HeatHalfLink(hl_name,self,dphi=1.,Tr=Tr,bc_type=1)
        self.color = 'b'

    @property
    def Ts(self):
        """getter of Ts.
        """
        return self._Ts

    @Ts.setter
    def Ts(self,Ts):
        """setter of Ts.
        """
        self._Ts = Ts

    def get_Ts(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Get supply temperature, optionally with scaling

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        T_shift : float, optional
            Scaled temperature to shift Ts with. Should be scaled in the same way as Ts. Default is None.

        Returns
        -------
        Ts : float
            Possibly scaled supply temperature.
        """
        Ts = self.Ts
        if scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            Ts = Ts/Tb
        if not T_shift == None:
            Ts -= T_shift
        return Ts

    @property
    def Tr(self):
        """getter of Tr.
        """
        return self._Tr

    @Tr.setter
    def Tr(self,Tr):
        """setter of Tr.
        """
        self._Tr = Tr


    def get_Tr(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Get return temperature optionally with scaling

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        T_shift : float, optional
            Scaled temperature to shift Tr with. Should be scaled in the same way as Tr. Default is None.

        Returns
        -------
        Tr : float
            Possibly scaled return temperature
        """
        Tr = self.Tr
        if scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            Tr = Tr/Tb
        if not T_shift == None:
            Tr -= T_shift
        return Tr

    @property
    def p(self):
        """getter of p.
        """
        return self._p

    @p.setter
    def p(self,p):
        """setter of p.
        """
        self._p = p


    def get_p(self,scale_var=None,scale_var_params=None):
        """Get pressure optionally with scaling

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        p : float
            Possibly scaled pressure
        """
        p = self.p
        if scale_var == 'per_unit':
            pb = scale_var_params['pbase']
            p = p/pb
        return p

    def get_head(self,carrier,scale_var=None,scale_var_params=None):
        """Get nodal head, optionally with scaling

        Parameters
        ----------
        carrier : Carrier
            Carrier flowing through the pipe
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None


        Returns
        -------
        h : float
            Possibly scaled nodal head
        """
        p = self.get_p(scale_var=scale_var,scale_var_params=scale_var_params)
        return p/(carrier.rhon*carrier.g)

    def get_inflow(self,network=None,scale_var=None,scale_var_params=None):
        """
        Returns the total mass inflow from links (and not half links) into a node, in the supply line

        Parameters
        ----------
        network : Network, optional
            Network object in which the node law must be used. Default in None, such that all links connected to this node are used.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        ms_in : float
            Mass flow coming into the node from the links.
        """
        ms_in = 0
        for e in self.get_in_links():
            if not e in self.get_half_links():
                m = e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                if m >= 0.: # flow is actually coming in (in supply line)
                    ms_in += m
        for e in self.get_out_links():
            if not e in self.get_half_links():
                m = e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                if m < 0.: # flow is actually coming in (in supply line)
                    ms_in += -m
        return ms_in

    def get_outflow(self,network=None,scale_var=None,scale_var_params=None):
        """
        Return the total mass outflow to links (and not half links) from a node, in the supply line.

        Parameters
        ----------
        network : Network, optional
            Network object in which the node law must be used. Default in None, such that all links connected to this node are used.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        ms_out : float
            Mass flow coming out of the node into the links.
        """
        ms_out = 0
        for e in self.get_in_links():
            if not e in self.get_half_links():
                m = e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                if m < 0.: # flow is actually going out (in supply line)
                    ms_out += -m
        for e in self.get_out_links():
            if not e in self.get_half_links():
                m = e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                if m >= 0.: # flow is actually going out (in supply line)
                    ms_out += m
        return ms_out

    def node_law(self,network=None,scale_var=None,scale_var_params=None):
        """Node law for a heat node, which is conservation of mass.
        The sum of the water flows of all incoming and outgoing links and half links.

        Parameters
        ----------
        network : Network, optional
            Network object in which the node law must be used. Default in None, such that all links connected to this node are used.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            The sum of the gas flows of all incoming and outgoing links and half links.
        """
        f = 0.
        for e in self.get_in_links():
            if isinstance(network,Network):
                if e in network.get_links():
                    f += e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                f += e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
        for e in self.get_out_links(): # both links and half links
            if isinstance(network,Network):
                if e in network.get_links(): # only links, not half links
                    f -= e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                elif e in self.get_half_links():
                    f -= e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                f -= e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
        return f

    def mixing_rule(self,network=None,scale_var=None,scale_var_params=None):
        """Mixing rule for a heat node, which is an avarage of incoming temperatures weighted with respect to mass flow.

        Parameters
        ----------
        network : Network, optional
            Network object in which the node law must be used. Default in None, such that all links connected to this node are used.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        fTs : float
            The supply temperature mismatch: sum(ms_out)*Ts - sum(ms_in*Ts_in).
        fTr : float
            The return temperature mismatch: sum(mr_out)*Tr - sum(mr_in*Tr_in).
        """
        fTs = 0.
        fTr = 0.
        T_shift = None#network.get_Ta(scale_var=scale_var,scale_var_params=scale_var_params)
        for e in self.get_in_links():
            if e in network.get_links():
                m = e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                Ts_in = e.get_Tsend(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                Tr_in = e.get_Trend(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                fTs -= m*Ts_in
                fTr += m*Tr_in
        for e in self.get_out_links():
            if e in network.get_links(): # links
                m = e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                Ts_out = e.get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                Tr_out = e.get_Trstart(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                fTs += m*Ts_out
                fTr -= m*Tr_out
            elif e in self.get_half_links(): # half links
                m = e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)
                Ts = e.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                Tr = e.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
                fTs += m*Ts
                fTr -= m*Tr
        # make adjustments in case of only inflow or only outflow
        if (self.node_type in [0,4] or (self.node_type in [1,3,12,13,14,15] and np.all([hl.source for hl in self.get_half_links()]))) and self.get_outflow(scale_var=scale_var,scale_var_params=scale_var_params) == 0: #source nodes with only inflow in supply line
            fTs += self.get_inflow(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
            for hl in self.get_half_links():
                fTs -= hl.get_m(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        elif self.node_type in [2,5,6,7] and self.get_outflow(scale_var=scale_var,scale_var_params=scale_var_params) == 0: #junction nodes with only inflow in supply line
            fTs += self.get_inflow(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        elif (self.node_type in [8,16] or (self.node_type in [1,3,12,13,14,15] and np.all([hl.sink for hl in self.get_half_links()]))) and self.get_inflow(scale_var=scale_var,scale_var_params=scale_var_params) == 0: #sink nodes with only outflow in supply line
            fTr += self.get_outflow(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
            for hl in self.get_half_links():
                fTr += hl.get_m(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        elif self.node_type in [2,5,6,7] and self.get_inflow(scale_var=scale_var,scale_var_params=scale_var_params) == 0: #junction nodes with only outflow in supply line
            fTr += self.get_outflow(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        return fTs,fTr

# ===========================================================================
class HeatLink(Link):
    """Heat link class.

    Attributes
    ----------
    name : str
        The name of the half link.
    start_node : Node
        Start node of the half link.
    end_node : Node
        End node of the half link.
    number : int
        Number of the half link.
    color : str
        Color used for plotting.
    _m : float
        Water mass flow through the link (in the supply line), unscaled.
    _Tsstart : float
        Temperature at the start of the link in the supply line, unscaled. 
    _Trstart : float
        Temperature at the start of the link in the return line, unscaled. 
    _dTstart : float
        Temperature difference between supply and return line at the start of the link, unscaled.
    _Tsend : float
        Temperature at the end of the link in the supply line, unscaled. 
    _Trend : float
        Temperature at the end of the link in the return line, unscaled.
    _dTend : float
        Temperature difference between supply and return line at the end of the link, unscaled.
    _phisstart : float
        Heat power at the start of the link in the supply line, unscaled. 
    _phirstart : float
        Heat power at the start of the link in the return line, unscaled. 
    _dphistart : float
        Heat power difference between supply and return line at the start of the link, unscaled.
    _phisend : float
        Heat power at the end of the link in the supply line, unscaled. 
    _phirend : float
        Heat power at the end of the link in the return line, unscaled. 
    _dphiend : float
        Heat power difference between supply and return line at the end of the link, unscaled.
    """
    def __init__(self,name,start_node,end_node,m=0.,Tsstart=100.,Trstart=0.,dTstart=100.,Tsend=100.,Trend=0.,dTend=100.,phisstart=0.,phirstart=0.,dphistart=0.,phisend=0.,phirend=0.,dphiend=0.,bc_type=0,link_type='dummy',link_params=dict(),hydr_eq_form = 'dp_of_q'):
        """Creates a HeatLink object.

        Parameters
        ----------
        name : str
            The name of the half link.
        start_node : Node
            Start node of the half link.
        end_node : Node
            End node of the half link.
        m : float, optional
            Link mass flow, unscaled. Default is 0.
        Tsstart : float, optional
            Temperature at the start of the link in the supply line, unscaled. 
        Trstart : float, optional
            Temperature at the start of the link in the return line, unscaled. 
        dTstart : float, optional
            Temperature difference between supply and return line at the start of the link, unscaled.
        Tsend : float, optional
            Temperature at the end of the link in the supply line, unscaled. 
        Trend : float, optional
            Temperature at the end of the link in the return line, unscaled. 
        dTend : float, optional
            Temperature difference between supply and return line at the end of the link, unscaled.
        phisstart : float, optional
            Heat power at the start of the link in the supply line, unscaled. 
        phirstart : float, optional
            Heat power at the start of the link in the return line, unscaled. 
        dphistart : float, optional
            Heat power difference between supply and return line at the start of the link, unscaled.
        phisend : float, optional
            Heat power at the end of the link in the supply line, unscaled. 
        phirend : float, optional
            Heat power at the end of the link in the return line, unscaled.
        dphiend : float, optional
            Heat power difference between supply and return line at the end of the link, unscaled.
        bc_type : int, optional
            Boundary condition of the link. 0 = everything unknown (source), 1 = everything unknown (sink), 2 = dphi and Ts known at start (source), 3 = dphi and Tr known at start (sink), 4 = dphi and dT known at start (source), 5 = dphi and dT known at start (sink), 6 = Ts known at start (source), 7 = Tr known at start (sink), 8 = dphi known and Ts unknown at start (source), 9 = dphi known and Tr unknown at start (sink), 10 = dT known at start (source), and 11 = dT known at start (sink). Default is 0.
        link_type : str, optional.
            Type of the link. Default is 'dummy'. Options are 'dummy', 'isolated_resistor', 'standard_resistor', 'standard_pipe_low_pres_colebrook', 'standard_pipe_low_pres_pole', or 'isolated_pipe_low_pres_pole'.
        link_params : dict, optional
            Dictionary with the link parameters required for the link type. Default is an empty dict.
        hydr_eq_form : string, optional
            Determines which link equation formulation is used. Default is 'dp_of_q', which uses the pressure drop as a function of link flow (i.e. it uses fb). The other option is 'q_of_dp', which uses the link flow as a function of pressure drop (i.e. it uses fa).

        Raises
        ------
        TypeError
            If start_node or end_node is not an instance of Node.
        ValueError
            If link_type is not a valid link type.
        """
        super().__init__(name,start_node,end_node)
        self.m = m
        self.Tsstart = Tsstart
        self.Trstart = Trstart
        self.dTstart = dTstart
        self.Tsend = Tsend
        self.Trend = Trend
        self.dTend = dTend
        self.phisstart = phisstart
        self.phirstart = phirstart
        self.dphistart = dphistart
        self.phisend = phisend
        self.phirend = phirend
        self.dphiend = dphiend
        self.bc_type = bc_type
        self.link_type = link_type
        self.link_params = link_params
        self.hydr_eq_form = hydr_eq_form
        if link_type == 'dummy':
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.dummy()
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.dummy()
        elif link_type == 'isolated_resistor':
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            C = link_params['C']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.resistor(C)
        elif link_type == 'standard_resistor':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            C = link_params['C']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.resistor(C)
        elif link_type == 'standard_pipe_low_pres_colebrook':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            fric_fac = hydraulic.fric_fac_colebrook
            fric_fac_der_q = hydraulic.fric_fac_der_qcolebrook
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'standard_pipe_low_pres_pole':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            fric_fac = hydraulic.fric_fac_pole
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,fric_fac,D,L)
        elif link_type == 'isolated_pipe_low_pres_pole':
            carrier = link_params['carrier']
            L = link_params['L']
            D = link_params['D']
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            fric_fac = hydraulic.fric_fac_pole
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,fric_fac,D,L)
        elif link_type == 'isolated_pipe_low_pres_hagen_poiseuille':
            carrier = link_params['carrier']
            L = link_params['L']
            D = link_params['D']
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            fric_fac = hydraulic.fric_fac_hagen_poiseuille
            fric_fac_der_q = hydraulic.fric_fac_der_qhagen_poiseuille
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'standard_pipe_low_pres_hagen_poiseuille':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            fric_fac = hydraulic.fric_fac_hagen_poiseuille
            fric_fac_der_q = hydraulic.fric_fac_der_qhagen_poiseuille
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'isolated_pipe_low_pres_blasius':
            carrier = link_params['carrier']
            L = link_params['L']
            D = link_params['D']
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            fric_fac = hydraulic.fric_fac_blasius
            fric_fac_der_q = hydraulic.fric_fac_der_qblasius
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'standard_pipe_low_pres_blasius':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            fric_fac = hydraulic.fric_fac_blasius
            fric_fac_der_q = hydraulic.fric_fac_der_qblasius
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'standard_pipe_low_pres_churchill':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            fric_fac = hydraulic.fric_fac_churchill
            fric_fac_der_q = hydraulic.fric_fac_der_qchurchill
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'isolated_pipe_low_pres_churchill':
            carrier = link_params['carrier']
            L = link_params['L']
            D = link_params['D']
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            fric_fac = hydraulic.fric_fac_churchill
            fric_fac_der_q = hydraulic.fric_fac_der_qchurchill
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'isolated_pump':
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            r = link_params['r']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.compressor(r)
        else:
            raise ValueError('link_type is not a valid link type.')
        # hydraulic equations
        self.pipe_const = pipe_const
        self.dp = dp
        self.dp_of_q = dp_of_q
        self.q_of_dp = q_of_dp
        if hydr_eq_form == 'q_of_dp':
            self.f = fa
            self.df_ddp = dfa_ddp
            self.df_dq = dfa_dq
        elif hydr_eq_form == 'dp_of_q':
            self.f = fb
            self.df_ddp = dfb_ddp
            self.df_dq = dfb_dq
        else:
            raise ValueError('hydr_eq_form is not valid. It should be either "q_of_dp" or "dp_of_q", not {}.'.format(hydr_eq_form))
        self.ddp_dq = ddp_dq
        self.ddp_dp = ddp_dp

        # thermal equations
        self.temp_drop_fac = temp_drop_fac
        self.temp_drop_fac_dm = temp_drop_fac_dm
        self.Tend_of_Tstart = Tend_of_Tstart
        self.dTend_dm = dTend_dm
        self.dTend_dTstart = dTend_dTstart
        self.color = 'b'

    def set_type(self,link_type,link_params,hydr_eq_form = 'dp_of_q',bc_type=None):
        """Set or change the link type, and the corresponding link equations

        Parameters
        ----------
        link_type : str
            (New) type of the link. Must be 'dummy', 'isolated_resistor', 'standard_resistor', 'standard_pipe_low_pres_colebrook', 'standard_pipe_low_pres_pole' or 'isolated_pipe_low_pres_pole'.
        link_params : dict
            Dictionary with the link parameters required for the (new) link type
        hydr_eq_form : string, optional
            Determines which link equation formulation is used. Default is 'dp_of_q', which uses the pressure drop as a function of link flow (i.e. it uses fb). The other option is 'q_of_dp', which uses the link flow as a function of pressure drop (i.e. it uses fa).
        bc_type : int, optional
            Boundary condition of the link. 0 = everything unknown (source), 1 = everything unknown (sink), 2 = dphi and Ts known at start (source), 3 = dphi and Tr known at start (sink), 4 = dphi and dT known at start (source), 5 = dphi and dT known at start (sink), 6 = Ts known at start (source), 7 = Tr known at start (sink), 8 = dphi known and Ts unknown at start (source), 9 = dphi known and Tr unknown at start (sink), 10 = dT known at start (source), and 11 = dT known at start (sink). Default is None, meaning that the same bc_type is not changed.

        Raises
        ------
        ValueError
            If link_type is not a valid link type
        """
        self.link_type = link_type
        self.link_params = link_params
        self.hydr_eq_form = hydr_eq_form
        if bc_type:
            self.bc_type = bc_type
        if link_type == 'dummy':
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.dummy()
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.dummy()
        elif link_type == 'isolated_resistor':
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            C = link_params['C']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.resistor(C)
        elif link_type == 'standard_resistor':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            C = link_params['C']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.resistor(C)
        elif link_type == 'standard_pipe_low_pres_colebrook':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            fric_fac = hydraulic.fric_fac_colebrook
            fric_fac_der_q = hydraulic.fric_fac_der_qcolebrook
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'standard_pipe_low_pres_pole':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            fric_fac = hydraulic.fric_fac_pole
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,fric_fac,D,L)
        elif link_type == 'isolated_pipe_low_pres_pole':
            carrier = link_params['carrier']
            L = link_params['L']
            D = link_params['D']
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            fric_fac = hydraulic.fric_fac_pole
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,fric_fac,D,L)
        elif link_type == 'isolated_pipe_low_pres_hagen_poiseuille':
            carrier = link_params['carrier']
            L = link_params['L']
            D = link_params['D']
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            fric_fac = hydraulic.fric_fac_hagen_poiseuille
            fric_fac_der_q = hydraulic.fric_fac_der_qhagen_poiseuille
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'standard_pipe_low_pres_hagen_poiseuille':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            fric_fac = hydraulic.fric_fac_hagen_poiseuille
            fric_fac_der_q = hydraulic.fric_fac_der_qhagen_poiseuille
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'isolated_pipe_low_pres_blasius':
            carrier = link_params['carrier']
            L = link_params['L']
            D = link_params['D']
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            fric_fac = hydraulic.fric_fac_blasius
            fric_fac_der_q = hydraulic.fric_fac_der_qblasius
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'standard_pipe_low_pres_blasius':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            fric_fac = hydraulic.fric_fac_blasius
            fric_fac_der_q = hydraulic.fric_fac_der_qblasius
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'standard_pipe_low_pres_churchill':
            carrier = link_params['carrier']
            U = link_params['U']
            L = link_params['L']
            D = link_params['D']
            if 'Ta' in link_params.keys():
                Ta = link_params['Ta']
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D, Ta=Ta)
            else:
                temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(carrier, U, L, D)
            fric_fac = hydraulic.fric_fac_churchill
            fric_fac_der_q = hydraulic.fric_fac_der_qchurchill
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'isolated_pipe_low_pres_churchill':
            carrier = link_params['carrier']
            L = link_params['L']
            D = link_params['D']
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            fric_fac = hydraulic.fric_fac_churchill
            fric_fac_der_q = hydraulic.fric_fac_der_qchurchill
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'isolated_pump':
            temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
            r = link_params['r']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.compressor(r)
        else:
            raise ValueError('link_type is not a valid link type.')
        # hydraulic equations
        self.pipe_const = pipe_const
        self.dp = dp
        self.dp_of_q = dp_of_q
        self.q_of_dp = q_of_dp
        if hydr_eq_form == 'q_of_dp':
            self.f = fa
            self.df_ddp = dfa_ddp
            self.df_dq = dfa_dq
        elif hydr_eq_form == 'dp_of_q':
            self.f = fb
            self.df_ddp = dfb_ddp
            self.df_dq = dfb_dq
        else:
            raise ValueError('hydr_eq_form is not valid. It should be either "q_of_dp" or "dp_of_q", not {}.'.format(hydr_eq_form))
        self.ddp_dq = ddp_dq
        self.ddp_dp = ddp_dp

        # thermal equations
        self.temp_drop_fac = temp_drop_fac
        self.temp_drop_fac_dm = temp_drop_fac_dm
        self.Tend_of_Tstart = Tend_of_Tstart
        self.dTend_dm = dTend_dm
        self.dTend_dTstart = dTend_dTstart

    @property
    def m(self):
        """getter of m.
        """
        return self._m

    @m.setter
    def m(self,m):
        """setter of m.
        """
        self._m = m

    def get_m(self,scale_var=None,scale_var_params=None):
        """Get mass flow, optionally with scaling.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        m : float
            Possible scaled mass flow.
        """
        m = self.m
        if scale_var == 'per_unit':
            mb = scale_var_params['qbase']
            m = m/mb
        return m

    @property
    def Tsstart(self):
        """getter of Tsstart.
        """
        return self._Tsstart

    @Tsstart.setter
    def Tsstart(self,Tsstart):
        """setter of Tsstart.
        """
        self._Tsstart = Tsstart

    def get_Tsstart(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Get supply temperature at start of the link, optionally with scaling

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Tsstart with. Should be scaled in the same way as Tsstart. Default is None.

        Returns
        -------
        Tsstart : float
            Possibly scaled supply temperature at start of the link.
        """
        Tsstart = self.Tsstart
        if scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            Tsstart = Tsstart/Tb
        if not T_shift == None:
            Tsstart -= T_shift
        return Tsstart

    @property
    def Trstart(self):
        """getter of Trstart.
        """
        return self._Trstart

    @Trstart.setter
    def Trstart(self,Trstart):
        """setter of Trstart.
        """
        self._Trstart = Trstart

    def get_Trstart(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Get return temperature at start of the link, optionally with scaling

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Trstart with. Should be scaled in the same way as Trstart. Default is None.

        Returns
        -------
        Trstart : float
            Possibly scaled return temperature at start of the link.
        """
        Trstart = self.Trstart
        if scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            Trstart = Trstart/Tb
        if not T_shift == None:
            Trstart -= T_shift
        return Trstart

    @property
    def dTstart(self):
        """getter of dTstart.
        """
        return self._dTstart

    @dTstart.setter
    def dTstart(self,dTstart):
        """setter of dTstart.
        """
        self._dTstart = dTstart

    def get_dTstart(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Get temperature difference between supply and return at start of the link, optionally with scaling

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift dTstart with. Should be scaled in the same way as dTstart. Default is None.

        Returns
        -------
        dTstart : float
            Possibly scaled temperature difference between supply and return at start of the link.
        """
        dTstart = self.dTstart
        if scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            dTstart = dTstart/Tb
        if not T_shift == None:
            dTstart -= T_shift
        return dTstart
    
    @property
    def Tsend(self):
        """getter of Tsend.
        """
        return self._Tsend

    @Tsend.setter
    def Tsend(self,Tsend):
        """setter of Tsend.
        """
        self._Tsend = Tsend

    def get_Tsend(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Get supply temperature at end of the link, optionally with scaling.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Tsend with. Should be scaled in the same way as Tsend. Default is None.

        Returns
        -------
        Tsend : float
            Possibly scaled supply temperature at end of the link.
        """
        Tsend = self.Tsend
        if scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            Tsend = Tsend/Tb
        if not T_shift == None:
            Tsend -= T_shift
        return Tsend

    @property
    def Trend(self):
        """getter of Trend.
        """
        return self._Trend

    @Trend.setter
    def Trend(self,Trend):
        """setter of Trend.
        """
        self._Trend = Trend

    def get_Trend(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Get return temperature at end of the link, optionally with scaling.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Trend with. Should be scaled in the same way as Trend. Default is None.

        Returns
        -------
        Trend : float
            Possibly scaled return temperature at end of the link.
        """
        Trend = self.Trend
        if scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            Trend = Trend/Tb
        if not T_shift == None:
            Trend -= T_shift
        return Trend

    @property
    def dTend(self):
        """getter of dTend.
        """
        return self._dTend

    @dTend.setter
    def dTend(self,dTend):
        """setter of dTend.
        """
        self._dTend = dTend

    def get_dTend(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Get temperature difference between supply and return at end of the link, optionally with scaling

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift dTend with. Should be scaled in the same way as dTend. Default is None.

        Returns
        -------
        dTend : float
            Possibly scaled temperature difference between supply and return at end of the link.
        """
        dTend = self.dTend
        if scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            dTend = dTend/Tb
        if not T_shift == None:
            dTend -= T_shift
        return dTend
    
    @property
    def phisstart(self): 
        """getter of phisstart.
        """
        return self._phisstart
    
    @phisstart.setter
    def phisstart(self,phisstart):
        """setter of phisstart.
        """
        self._phisstart = phisstart
    
    def get_phisstart(self,scale_var=None,scale_var_params=None):
        """Get heat power in supply line near start node, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Pstart : float
            Possibly scaled heat power in supply line at start of the link.
        """
        phisstart = self.phisstart
        if scale_var == 'per_unit':
            phib = scale_var_params['phibase']
            phisstart = phisstart/phib
        return phisstart
    
    @property
    def phirstart(self): 
        """getter of phirstart.
        """
        return self._phirstart
    
    @phirstart.setter
    def phirstart(self,phirstart):
        """setter of phirstart.
        """
        self._phirstart = phirstart
        
    def get_phirstart(self,scale_var=None,scale_var_params=None):
        """Get heat power in return line near start node, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        phirstart : float
            Possibly scaled heat power in return line at start of the link.
        """
        phirstart = self.phirstart
        if scale_var == 'per_unit':
            phib = scale_var_params['phibase']
            phirstart = phirstart/phib
        return phirstart
    
    @property
    def dphistart(self): 
        """getter of dphistart.
        """
        return self._dphistart
    
    @dphistart.setter
    def dphistart(self,dphistart):
        """setter of dphistart.
        """
        self._dphistart = dphistart
    
    def get_dphistart(self,scale_var=None,scale_var_params=None):
        """Get heat power difference between supply and return at start of link, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Pstart : float
            Possibly scaled heat power difference between supply and return at start of link
        """
        dphistart = self.dphistart
        if scale_var == 'per_unit':
            phib = scale_var_params['phibase']
            dphistart = dphistart/phib
        return dphistart
    
    @property
    def phisend(self): 
        """getter of phisend.
        """
        return self._phisend
    
    @phisend.setter
    def phisend(self,phisend):
        """setter of phis.
        """
        self._phisend = phisend
    
    def get_phisend(self,scale_var=None,scale_var_params=None):
        """Get heat power near in supply line end node, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        phisend : float
            Possibly scaled heat power in supply line at end of the link. 
        """
        phisend = self.phisend
        if scale_var == 'per_unit':
            phib = scale_var_params['phibase']
            phisend = phisend/phib
        return phisend
    
    @property
    def phirend(self): 
        """getter of phirend.
        """
        return self._phirend
    
    @phirend.setter
    def phirend(self,phirend):
        """setter of phirend.
        """
        self._phirend = phirend
        
    
    def get_phirend(self,scale_var=None,scale_var_params=None):
        """Get heat power in return line near the end node, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        phirend : float
            Possibly scaled heat power in return line at end of the link. 
        """
        phirend = self.phirend
        if scale_var == 'per_unit':
            phib = scale_var_params['phibase']
            phirend = phirend/phib
        return phirend

    @property
    def dphiend(self): 
        """getter of dphiend.
        """
        return self._dphiend
    
    @dphiend.setter
    def dphiend(self,dphiend):
        """setter of dphiend.
        """
        self._dphiend = dphiend
    
    def get_dphiend(self,scale_var=None,scale_var_params=None):
        """Get heat power difference between supply and return at end of link, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Pend : float
            Possibly scaled heat power difference between supply and return at end of link
        """
        dphiend = self.dphiend
        if scale_var == 'per_unit':
            phib = scale_var_params['phibase']
            dphiend = dphiend/phib
        return dphiend
    
    def pres_drop(self,scale_var=None,scale_var_params=None):
        """Determines the pressure drop over the link.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dp : float
            Pressure drop over link. Pressure of start node - pressure at end node.
        """
        return self.start_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)-self.end_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)        
        
    def pres_drop_func(self,scale_var=None,scale_var_params=None):
        """Determines the pressure drop function over the link.
        The function is determined by the link type.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dp : float
            Pressure drop function over the link. Determined by the link type
        """
        return self.dp(self.start_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params))
    
    def flow(self,scale_var=None,scale_var_params=None):
        """Determines the flow through the link as a function of start and end pressures.
        This function is determined by the link type. 

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        q : float
            Link flow. 
        """
        return self.q_of_dp(self.start_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)
    
    def pres_drop_func_der_m(self,scale_var=None,scale_var_params=None):
        """Determines the derivative of the pressure drop function to link flow.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        ddp_dm : float
            Derivative of pressure drop function to link flow.
        """
        return self.ddp_dq(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)

    def pres_drop_func_der_p(self,scale_var=None,scale_var_params=None):
        """Determines the derivative of the pressure drop function to start and end pressures.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        ddp_dp_start : float
            Derivative of pressure drop function to the start pressure.
        ddp_dp_end : float
            Derivative of pressure drop function to the end pressure.
        """
        try:
            pstart = self.start_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)
        except AttributeError:
            pstart = None
        try:
            pend = self.end_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)
        except AttributeError:
            pend = None
        return self.ddp_dp(pstart,pend)

    def psi(self,scale_var=None,scale_var_params=None):
        """Determine the temperature drop factor, as a function of mass flow.
        Factor is determined by the link type.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        psi : float
            Temperature drop factor of the link.
        """
        return self.temp_drop_fac(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)

    def psi_der_m(self,scale_var=None,scale_var_params=None):
        """Determine the derivative of the temperature drop factor to the link flow.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dpsi_dm : float
            The derivative of the temperature drop factor of the link to the link flow.
        """
        return self.temp_drop_fac_dm(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)

    def link_equation(self,scale_var=None,scale_var_params=None):
        """Determines the value of the link equation.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        f : float
            link equation f(q,p_start,p_end)
        """
        return self.f(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.start_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)

    def f_der_dp_func(self,scale_var=None,scale_var_params=None):
        """Determines the derivative of the link equation to the pressure drop function.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        df_ddp : float
            Derivative of the link equation to pressure drop fucntion
        """
        try:
            pstart = self.start_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)
        except AttributeError:
            pstart = None
        try:
            pend = self.end_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)
        except AttributeError:
            pend = None
        return self.df_ddp(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),pstart,pend,scale_var=scale_var,scale_var_params=scale_var_params)

    def f_der_m(self,scale_var=None,scale_var_params=None):
        """Determines the derivative of the link equation to the link flow.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        df_dm : float
            Derivative of link equation to link flow.
        """
        try:
            pstart = self.start_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)
        except AttributeError:
            pstart = None
        try:
            pend = self.end_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)
        except AttributeError:
            pend = None
        return self.df_dq(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),pstart,pend,scale_var=scale_var,scale_var_params=scale_var_params)

    def supply_temp_start(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the supply temperature at the start of the link, based on defined direction of link. Possibly a function of temperature at end of the link.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Tr_start with. Should be scaled in the same way as Tr_start. Default is None.

        Returns
        -------
        Ts_start : float
            Supply temperature at start of the link.
        """
        if self.get_m()>=0:
            if isinstance(self.start_node,HeatNode):
                Ts_start = self.start_node.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
            else:
                Ts_start = self.get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        else:
            Ts_start = self.Tend_of_Tstart(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.get_Tsend(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
        return Ts_start

    def return_temp_start(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the return temperature at the start of the link, based on defined direction of link. Possibly a function of temperature at end of the link.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Tr_start with. Should be scaled in the same way as Tr_start. Default is None.

        Returns
        -------
        Tr_start : float
            Return temperature at start of the link.
        """
        if self.get_m()>=0: # flow is in opposite direction in return line
            Tr_start = self.Tend_of_Tstart(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.get_Trend(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            if isinstance(self.start_node,HeatNode):
                Tr_start = self.start_node.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
            else:
                Tr_start = self.get_Trstart(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        return Tr_start
    
    def supply_temp_end(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the supply temperature at the end of the link, based on defined direction of link. Possibly a function of temperature at start of the link.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Ts_end with. Should be scaled in the same way as Ts_end. Default is None.

        Returns
        -------
        Ts_end : float
            Supply temperature at end of the link.
        """
        if self.get_m()>=0:
            Ts_end = self.Tend_of_Tstart(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.get_Tsstart(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            if isinstance(self.end_node,HeatNode):
                Ts_end = self.end_node.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
            else:
                Ts_end = self.get_Tssend(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        return Ts_end

    def return_temp_end(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the return temperature at the end of the link, based on defined direction of link. Possibly a function of temperature at start of the link.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Tr_end with. Should be scaled in the same way as Tr_end. Default is None.

        Returns
        -------
        Tr_end : float
            Return temperature at end of the link.
        """
        if self.get_m()>=0: # flow is in opposite direction in return line
            if isinstance(self.end_node,HeatNode):
                Tr_end = self.end_node.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
            else:
                Tr_end = self.get_Trend(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        else:
            Tr_end = self.Tend_of_Tstart(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.get_Trstart(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
        return Tr_end

    def supply_heat_power_start(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the heat power at the start of the link, in the supply line, as a function of temperature and mass flow.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        T_shift : float, optional
            Scaled temperature to shift Tr_end with. Should be scaled in the same way as Tr_end. Default is None.
            
        Returns
        -------
        phisstart : float
            Heat power in supply line at start of the link.
        """
        return self.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_m(scale_var=scale_var,scale_var_params=scale_var_params)*self.supply_temp_start(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
    
    def return_heat_power_start(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the heat power at the start of the link, in the return line, as a function of temperature and mass flow.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        T_shift : float, optional
            Scaled temperature to shift T with. Should be scaled in the same way as T. Default is None.
            
        Returns
        -------
        phirstart : float
            Heat power in return line at start of the link.
        """
        return -self.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_m(scale_var=scale_var,scale_var_params=scale_var_params)*self.return_temp_start(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
    
    def supply_heat_power_end(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the heat power at the end of the link, in the supply line, as a function of temperature and mass flow.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        T_shift : float, optional
            Scaled temperature to shift Tr_end with. Should be scaled in the same way as Tr_end. Default is None.
            
        Returns
        -------
        phisstart : float
            Heat power in supply line at end of the link.
        """
        return -self.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_m(scale_var=scale_var,scale_var_params=scale_var_params)*self.supply_temp_end(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
    
    def return_heat_power_end(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the heat power at the end of the link, in the return line, as a function of temperature and mass flow.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        T_shift : float, optional
            Scaled temperature to shift T with. Should be scaled in the same way as T. Default is None.
            
        Returns
        -------
        phirstart : float
            Heat power in return line at end of the link.
        """
        return self.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_m(scale_var=scale_var,scale_var_params=scale_var_params)*self.return_temp_end(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)

    def heat_loss_supply(self,scale_var=None,scale_var_params=None):
        """Determines the heat loss over the pipe in the supply line. 

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dphi : float
            Heat loss over the pipe in the supply line. 
        """
        return self.get_phisstart(scale_var=scale_var,scale_var_params=scale_var_params) + self.get_phisend(scale_var=scale_var,scale_var_params=scale_var_params)
    
    def heat_loss_return(self,scale_var=None,scale_var_params=None):
        """Determines the heat loss over the pipe in the return line. 

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dphi : float
            Heat loss over the pipe in the return line. 
        """
        return self.get_phirstart(scale_var=scale_var,scale_var_params=scale_var_params) + self.get_phirend(scale_var=scale_var,scale_var_params=scale_var_params)
    
    def supply_temp_start_der_m(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the supply temperature at the start of the link, based on defined direction of link, to link mass flow

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dTsij_dm : float
            Derivative of the start supply temperature to mass flow
        """
        if self.get_m()>=0:
            dTsij_dm = 0
        else:
            dTsij_dm = self.dTend_dm(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.supply_temp_end(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
        return dTsij_dm
    
    def supply_temp_start_der_Ts(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the supply temperature at the start of the link, based on defined direction of link, to nodal supply temperature of start node (again, start node based on defined direction of flow)

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dTsij_dTs : float, array
            Derivative of the start supply temperature to nodal supply temperature of start node and to nodal return temperature of end node
        """
        if self.get_m()>=0:
            dTsij_dTsi = 1.
            dTsij_dTsj = 0.
        else:
            dTsij_dTsi = 0.
            dTsij_dTsj = self.dTend_dTstart(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.supply_temp_end(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
        return np.array([dTsij_dTsi,dTsij_dTsj])
    
    def supply_temp_end_der_m(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the supply temperature at the end of the link, based on defined direction of link, to link mass flow

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dTsji_dm : float
            Derivative of the end supply temperature to mass flow
        """
        if self.get_m()>=0:
            dTsji_dm = self.dTend_dm(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.supply_temp_start(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            dTsji_dm = 0.
        return dTsji_dm
    
    def supply_temp_end_der_Ts(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the supply temperature at the end of the link, based on defined direction of link, to nodal supply temperature of start node (again, start node based on defined direction of flow)

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dTsji_dTs : float, array
            Derivative of the end supply temperature to nodal supply temperature of start node and to nodal return temperature of end node
        """
        if self.get_m()>=0:
            dTsji_dTsi = self.dTend_dTstart(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.supply_temp_start(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
            dTsji_dTsj = 0.
        else:
            dTsji_dTsi = 0.
            dTsji_dTsj = 1.
        return np.array([dTsji_dTsi,dTsji_dTsj])
    
    def return_temp_start_der_m(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the return temperature at the start of the link, based on defined direction of link, to link mass flow

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dTrij_dm : float
            Derivative of the start return temperature to mass flow
        """
        if self.get_m()>=0:
            dTrij_dm = self.dTend_dm(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.return_temp_end(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            dTrij_dm = 0.
        return dTrij_dm
    
    def return_temp_start_der_Tr(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the return temperature at the start of the link, based on defined direction of link, to nodal return temperature of start node (again, start node based on defined direction of flow)

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dTrij_dTr : float, array
            Derivative of the start return temperature to nodal return temperature of start node and to nodal return temperature of end node
        """
        if self.get_m()>=0:
            dTrij_dTri = 0.
            dTrij_dTrj = self.dTend_dTstart(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.return_temp_end(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            dTrij_dTri = 1.
            dTrij_dTrj = 0.
        return np.array([dTrij_dTri,dTrij_dTrj])
    
    def return_temp_end_der_m(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the return temperature at the end of the link, based on defined direction of link, to link mass flow

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dTrij_dm : float
            Derivative of the end return temperature to mass flow
        """
        if self.get_m()>=0:
            dTrij_dm = 0.
        else:
            dTrij_dm = self.dTend_dm(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.return_temp_start(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
        return dTrij_dm
    
    def return_temp_end_der_Tr(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the return temperature at the end of the link, based on defined direction of link, to nodal return temperature of end node (again, end node based on defined direction of flow)

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dTrij_dTr : float, array
            Derivative of the end return temperature to nodal return temperature of end node and to nodal return temperature of end node
        """
        if self.get_m()>=0:
            dTrij_dTri = 0.
            dTrij_dTrj = 1.
        else:
            dTrij_dTri = self.dTend_dTstart(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.return_temp_start(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
            dTrij_dTrj = 0.
        return np.array([dTrij_dTri,dTrij_dTrj])
    
# ===========================================================================
class HeatHalfLink(HalfLink):
    """Heat half link class.
    The default is an outflow half link.

    Attributes
    ----------
    name : str
        Name of the half link.
    start_node : Node
        Start node of the half link.
    number : int
        Number of the half link.
    _m : float
        Mass flow, unscaled.
    _phis : float
        Heat power near supply line, unscaled.
    _phir : float
        Heat power near return line, unscaled.
    _dphi : float
        Heat power difference over half link. That is, the heat power exchanged with surroundings in case of heat exchanger half link.
    _Ts : float
        Temperature of component near supply line, unscaled and not shifted.
    _Tr : float
        Temperature of component near return line, unscaled and not shifted.
    _dT : float
        Temperature difference over half link. 
    """
    def __init__(self,name,start_node,Ts=0.,Tr=0.,m=0.,phis=0.,phir=0.,dphi=0.,dT=0.,bc_type=0,link_type='dummy',link_params=dict()):
        """Creates a HeatHalfLink object

        Parameters
        ----------
        name : str
            Name of the half link.
        start_node : Node
            Start node of the half link.
        m : float, optional
            Mass flow, unscaled. Default is 0.
        phis : float, optional
            Heat power near supply line, unscaled. Default is 0.
        phir : float, optional
            Heat power near return line, unscaled. Default is 0.
        dphi : float
            Heat power difference over half link. That is, the heat power exchanged with surroundings in case of heat exchanger half link. Default is 0.
        Ts : float, optional
            Temperature of component near supply line, unscaled and not shifted. Default is 0.
        Tr : float, optional
            Temperature of component near return line, unscaled and not shifted. Default is 0.
        dT : float, optional
            Temperature difference over half link, unscaled and not shifted. Default is 0.
        bc_type : int, optional
            Boundary condition of the half link. 0 = everything unknown (source), 1 = everything unknown (sink), 2 = dphi and Ts known (source), 3 = dphi and Tr known (sink), 4 = dphi and dT known (source), 5 = dphi and dT known (sink), 6 = Ts known (source), 7 = Tr known (sink), 8 = dphi known and Ts unknown (source), 9 = dphi known and Tr unknown (sink), 10 = dT known (source), and 11 = dT known (sink), 12 = m known (source), 13 = m known (sink), 14 = m, dphi and Ts known (source), 15 = m, dphi and Tr known (sink), 16 = m, dphi and dT known (source), 17 = m, dphi and dT known (sink), 18 = m, Ts known (source), 19 = m, Tr known (sink), 20 = m, dphi known and Ts unknown (source), 21 = m, dphi known and Tr unknown (sink), 22 = m, dT known (source), and 23 = m, dT known (sink). Default is 0.
        link_type : string, optional
            Type of the half link, options are 'dummy' or 'heat_exchanger'. Default is 'dummy', which creates a half link with no model.
        link_params : dict, optional
            Dictionary of halflink parameters needed for a specific halflink type. Default is an empty dict.

        Raises
        ------
        TypeError
            If start_node is not an instance of Node
        ValueError
            If halflink_type is not a valid half link type.
        """
        super().__init__(name,start_node)
        self.bc_type = bc_type
        if self.bc_type in  [1,3,5,7,9,11,13,15,17,19,21,23]: # list of bc types for which the half link acts as a sink
            self.sink = True
            self.source = False
        else:
            self.sink = False
            self.source = True
        self.link_type = link_type
        self.link_params = link_params
        self.m = m
        self.phis = phis
        self.phir = phir
        self.dphi = dphi
        self.Ts = Ts
        self.Tr = Tr
        self.dT = dT
        if link_type == 'dummy':
            m_of_phi, phi_of_m, dm_dTs, dm_dTr, dphi_dm, dphi_dTs, dphi_dTr = halflink_thermal.dummy()
        elif link_type == 'heat_exchanger':
            carrier = link_params['carrier']
            m_of_phi, phi_of_m, dm_dTs, dm_dTr, dphi_dm, dphi_dTs, dphi_dTr = halflink_thermal.heat_exchanger(carrier)
        else:
            raise ValueError("link_type should be either 'dummy' or 'heat_exchanger' not {}".format(link_type))
        self.m_of_phi = m_of_phi
        self.phi_of_m = phi_of_m
        self.dm_dTs = dm_dTs
        self.dm_dTr = dm_dTr
        self.dphi_dm = dphi_dm
        self.dphi_dTs = dphi_dTs
        self.dphi_dTr = dphi_dTr
        self.color = 'b'

    def set_type(self,link_type,link_params,bc_type=None):
        """Set or change the half link type, and the corresponding link equations.

        Parameters
        ----------
        bc_type : int, optional
            Boundary condition of the half link. 0 = everything unknown (source), 1 = everything unknown (sink), 2 = dphi and Ts known (source), 3 = dphi and Tr known (sink), 4 = dphi and dT known (source), 5 = dphi and dT known (sink), 6 = Ts known (source), 7 = Tr known (sink), 8 = dphi known and Ts unknown (source), 9 = dphi known and Tr unknown (sink), 10 = dT known (source), and 11 = dT known (sink), 12 = m known (source), 13 = m known (sink), 14 = m, dphi and Ts known (source), 15 = m, dphi and Tr known (sink), 16 = m, dphi and dT known (source), 17 = m, dphi and dT known (sink), 18 = m, Ts known (source), 19 = m, Tr known (sink), 20 = m, dphi known and Ts unknown (source), 21 = m, dphi known and Tr unknown (sink), 22 = m, dT known (source), and 23 = m, dT known (sink). Default is None, meaning that the same bc_type is not changed.
        link_type : str
            (New) type of the link. Must be 'dummy', 'heat_exchanger_source', 'heat_exchanger_sink', or 'heat_exchanger_general'.
        link_params : dict
            Dictionary with the link parameters required for the (new) link type

        Raises
        ------
        ValueError
            If link_type is not a valid half link type
        """
        self.link_type = link_type
        self.link_params = link_params
        if bc_type != None: # Need to check if not None, because if bc_type = 0, it should be changed
            self.bc_type = bc_type
            if self.bc_type in [1,3,5,7,9,11,13,15,17,19,21,23]: # list of bc types for which the half link acts as a sink
                self.sink = True
                self.source = False
            else:
                self.sink = False
                self.source = True
        if link_type == 'dummy':
            m_of_phi, phi_of_m, dm_dTs, dm_dTr, dphi_dm, dphi_dTs, dphi_dTr = halflink_thermal.dummy()
        elif link_type == 'heat_exchanger':
            carrier = link_params['carrier']
            m_of_phi, phi_of_m, dm_dTs, dm_dTr, dphi_dm, dphi_dTs, dphi_dTr = halflink_thermal.heat_exchanger(carrier)
        else:
            raise ValueError("link_type should be either 'dummy' or 'heat_exchanger' not {}".format(link_type))
        self.m_of_phi = m_of_phi
        self.phi_of_m = phi_of_m
        self.dm_dTs = dm_dTs
        self.dm_dTr = dm_dTr
        self.dphi_dm = dphi_dm
        self.dphi_dTs = dphi_dTs
        self.dphi_dTr = dphi_dTr

    @property
    def m(self):
        """getter of m.
        """
        return self._m

    @m.setter
    def m(self,m):
        """setter of m.
        """
        self._m = m

    def get_m(self,scale_var=None,scale_var_params=None):
        """Get mass flow, optionally with scaling.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        m : float
            Possibly scaled mass flow.
        """
        m = self.m
        if scale_var == 'per_unit':
            mb = scale_var_params['qbase']
            m = m/mb
        return m

    @property
    def phis(self):
        """getter of phis.
        """
        return self._phis

    @phis.setter
    def phis(self,phis):
        """setter of phis.
        """
        self._phis = phis


    def get_phis(self,scale_var=None,scale_var_params=None):
        """Get heat power near supply line, optionally with scaling.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        phis : float
            Possibly scaled heat power near supply line.
        """
        phis = self.phis
        if scale_var == 'per_unit':
            phib = scale_var_params['phibase']
            phis = phis/phib
        return phis
    
    @property
    def phir(self):
        """getter of phir.
        """
        return self._phir

    @phir.setter
    def phir(self,phir):
        """setter of phir.
        """
        self._phir = phir


    def get_phir(self,scale_var=None,scale_var_params=None):
        """Get heat power near return line, optionally with scaling.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        phir : float
            Possibly scaled heat power near return line.
        """
        phir = self.phir
        if scale_var == 'per_unit':
            phib = scale_var_params['phibase']
            phir = phir/phib
        return phir

    @property
    def dphi(self):
        """getter of dphi.
        """
        return self._dphi

    @dphi.setter
    def dphi(self,dphi):
        """setter of dphi.
        """
        self._dphi = dphi


    def get_dphi(self,scale_var=None,scale_var_params=None):
        """Get heat power difference over half link, optionally with scaling.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dphi : float
            Possibly scaled heat power difference over half link.
        """
        dphi = self.dphi
        if scale_var == 'per_unit':
            phib = scale_var_params['phibase']
            dphi = dphi/phib
        return dphi
    
    @property
    def Ts(self):
        """getter of Ts.
        """
        return self._Ts

    @Ts.setter
    def Ts(self,Ts):
        """setter of Ts.
        """
        self._Ts = Ts

    def get_Ts(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Get temperature near supply line, optionally with scaling and shifted.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Ts with. Should be scaled in the same way as Ts. Default is None.

        Returns
        -------
        Ts : float
            Possibly scaled and shifted temperature near supply line.
        """
        Ts = self.Ts
        if scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            Ts = Ts/Tb
        if not T_shift == None:
            Ts -= T_shift
        return Ts
    
    @property
    def Tr(self):
        """getter of Tr.
        """
        return self._Tr

    @Tr.setter
    def Tr(self,Tr):
        """setter of Tr.
        """
        self._Tr = Tr
        
    def get_Tr(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Get temperature near return line, optionally with scaling and shifted.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Tr with. Should be scaled in the same way as Tr. Default is None.

        Returns
        -------
        Tr : float
            Possibly scaled and shifted temperature near return line.
        """
        Tr = self.Tr
        if scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            Tr = Tr/Tb
        if not T_shift == None:
            Tr -= T_shift
        return Tr

    @property
    def dT(self):
        """getter of dT.
        """
        return self._dT

    @dT.setter
    def dT(self,dT):
        """setter of dT.
        """
        self._dT = dT


    def get_dT(self,scale_var=None,scale_var_params=None):
        """Get temperature difference, optionally with scaling.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None

        Returns
        -------
        dT : float
            Possibly scaled temperature difference.
        """
        dT = self.dT
        if scale_var == 'per_unit':
            Tb = scale_var_params['Tbase']
            dT = dT/Tb
        return dT
    
    def heat_power_equation(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Heat power equation.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.
            
        Returns
        -------
        fphi : float
            The heat power equation fphi(phi,m,To,Ts_i) or fphi(phi,m,To,Tr_i)
        """
        return -self.get_dphi(scale_var=scale_var,scale_var_params=scale_var_params) + self.phi_of_m(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),self.return_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
    
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
            The temperature difference equation fTo(Ts,Tr)
        """
        if self.link_type == 'heat_exchanger':
            fTo = self.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params) - self.return_temp(scale_var=scale_var,scale_var_params=scale_var_params) - self.get_dT(scale_var=scale_var,scale_var_params=scale_var_params)
        else:
            fTo = None
        return fTo
    
    def ddT_der_Tshl(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the temperature difference to the half link temperature near supply line.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        ddT_dTshl : float
            Derivative of the temperature difference to the supply temperature
        """
        return 1
    
    def ddT_der_Ts(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the temperature difference to the nodal supply temperature.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        ddT_dTs : float
            Derivative of the temperature difference to the supply temperature
        """
        return self.ddT_der_Tshl(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)* self.supply_temp_der_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
    
    def ddT_der_Trhl(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the temperature difference to the half link temperature near return line.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        ddT_dTrhl : float
            Derivative of the temperature difference to the return temperature
        """
        return -1
    
    def ddT_der_Tr(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the temperature difference to the nodal return temperature.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        ddT_dTr : float
            Derivative of the temperature difference to the return temperature
        """
        return self.ddT_der_Trhl(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)* self.return_temp_der_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
    
    def flow(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determines the flow through the half link as a function of heat power and outflow temperature.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        m : float
            Half link flow.
        """
        return self.m_of_phi(self.get_dphi(scale_var=scale_var,scale_var_params=scale_var_params),self.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),self.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)

    def heat(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determines the heat power through the half link as a function of flow and outflow temperature.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        phi : float
            Half link heat power.
        """
        return self.phi_of_m(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),self.return_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)

    def supply_temp_der_Ts(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the half link temperature near supply line to the nodal supply temperature.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dTshl_dTs : float
            Derivative of the link flow to the supply temperature
        """
        if self.start_node.node_type in [9,10,11]: # general slack nodes
            dTshl_dTs = 1.
        elif self.sink:
            dTshl_dTs = 1.
        else:
            dTshl_dTs = 0.
        return dTshl_dTs
    
    def return_temp_der_Tr(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the half link temperature near return line to the nodal return temperature.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dTrhl_dTr : float
            Derivative of the link flow to the supply temperature
        """
        if self.start_node.node_type in [9,10,11]: # general slack nodes
            dTrhl_dTr = 1.
        elif self.sink:
            dTrhl_dTr = 0.
        else:
            dTrhl_dTr = 1.
        return dTrhl_dTr
    
    
    def m_der_Ts(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the half link flow (as a function of heat) to the nodal supply temperature.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dm_dTs : float
            Derivative of the link flow to the supply temperature
        """
        if self.start_node.node_type in [0,8,9,10,11]: # slack nodes, so m is expressed via conservation of mass, not as function of heat power
            return 0.
        else:
            return self.dm_dTs(self.get_dphi(scale_var=scale_var,scale_var_params=scale_var_params),self.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),self.return_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params) * self.supply_temp_der_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)

    def m_der_Tr(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the half link flow (as a function of heat) to the return temperature.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dm_dTr : float
            Derivative of the link flow to the supply temperature
        """
        if self.start_node.node_type in [0,8,9,10,11]: # slack nodes, so m is expressed via conservation of mass, not as function of heat power
            return 0.
        else:
            return self.dm_dTr(self.get_dphi(scale_var=scale_var,scale_var_params=scale_var_params),self.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),self.return_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)* self.return_temp_der_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        
    def phi_der_m(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the half link heat power difference to the half link flow.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dphi_dm : float
            Derivative of the heat power to flow
        """
        return self.dphi_dm(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),self.return_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)

    def phi_der_Tshl(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the half link heat power difference to the half link temperature near supply line.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dphi_dTshl : float
            Derivative of the half link heat to the supply temperature
        """
        return self.dphi_dTs(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.return_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),self.start_node.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
    
    def phi_der_Ts(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the half link heat power difference to the nodal supply temperature.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dphi_dTs : float
            Derivative of the half link heat to the supply temperature
        """
        return self.dphi_dTs(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.return_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),self.start_node.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)* self.supply_temp_der_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
    
    def phi_der_Trhl(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the half link heat power difference to the half link temperature near return line.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dphi_dTrhl : float
            Derivative of the half link heat to the return temperature
        """
        return self.dphi_dTr(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),self.start_node.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)
    
    def phi_der_Tr(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Derivative of the half link heat power difference to the nodal return temperature.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift To with. Should be scaled in the same way as To. Default is None.

        Returns
        -------
        dphi_dTr : float
            Derivative of the half link heat to the return temperature
        """
        return self.dphi_dTr(self.get_m(scale_var=scale_var,scale_var_params=scale_var_params),self.supply_temp(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),self.start_node.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift),scale_var=scale_var,scale_var_params=scale_var_params)* self.return_temp_der_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
    
    def supply_temp(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the temperature near the supply line, based on actual direction of flow. Possibly a function of temperature at end of the link.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Ts with. Should be scaled in the same way as Ts. Default is None.

        Returns
        -------
        Ts : float
            Supply temperature.
        """
        if self.sink:
            Ts = self.start_node.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        else:
            Ts = self.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        return Ts

    def return_temp(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the temperature near the return line, based on actual direction of flow. Possibly a function of temperature at end of the link.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None
        T_shift : float, optional
            Scaled temperature to shift Tr with. Should be scaled in the same way as Tr. Default is None.

        Returns
        -------
        Tr : float
            Return temperature.
        """
        if self.source and isinstance(self.start_node,HeatNode):
            Tr = self.start_node.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
        else:
            Tr = self.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)            
        return Tr
    
    def supply_heat_power(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the heat power near the supply line, as a function of temperature and mass flow.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        T_shift : float, optional
            Scaled temperature to shift Tr_end with. Should be scaled in the same way as Tr_end. Default is None.
            
        Returns
        -------
        phisstart : float
            Heat power in supply line at start of the link.
        """
        return self.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_m(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
    
    def return_heat_power(self,scale_var=None,scale_var_params=None,T_shift=None):
        """Determine the heat power near the return line, as a function of temperature and mass flow.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        T_shift : float, optional
            Scaled temperature to shift T with. Should be scaled in the same way as T. Default is None.
            
        Returns
        -------
        phirstart : float
            Heat power in return line at start of the link.
        """
        return -self.link_params.get('carrier').get_Cp(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_m(scale_var=scale_var,scale_var_params=scale_var_params)*self.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params,T_shift=T_shift)
