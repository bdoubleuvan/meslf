"""Electrical network base class, including Network, Node, and Link"""
from meslf.networks.network import Network, Node, Link, HalfLink
from meslf.link_equations import electrical
import meslf.half_link_equations.electrical as halflink_electrical
from meslf.load_flow.system_of_equations import NonLinearSystemElectrical
from meslf.load_flow.non_linear_solvers import NR, NR_FD, Fsolver, Root
import warnings
import numpy as np
import scipy.sparse as sps
import meslf.utils.basic_math as bm
from meslf.utils.list_manipulation import merge_sorted

# ===========================================================================
class ElectricalNetwork(Network):
    """Overall electrical network class. Subclass of Network.
    
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
    Y : scipy sparse csr_matrix
        Admittance incidence matrix
    """
    def __init__(self,name):
        """Creates a ElectricalNetwork object.
        
        Parameters
        ----------
        name : str
            The name of the network.
        """
        super().__init__(name)
        self.Y = None
        
    def add_node(self,node,position=None):
        """Adds a node to the network. 
        A node is only added if it is a ElectricalNode.
        
        Parameters
        ----------
        node : ElectricalNode
            Node object to be added to the network
        position : integer, optional
            Position (index) in the list of nodes of the network where the node should be inserted. Default is insert at end of list (append)
            
        Warns
        ------
        UserWarning
            If node is not an instance of ElectricalNode
        """
        if not isinstance(node,ElectricalNode):
             warnings.warn("Only a ElectricalNode object can be added")
        else:
            super().add_node(node,position=position)
            
    def add_link(self,link,position=None):
        """Adds link to the network. 
        A link can only added if it is a ElectricalLink.
        If the start node or the end node are not yet added to the network, they
        will be added to the list of nodes. 
        
        Parameters
        ----------
        link : ElectricalLink
            Link object to be added to the network
        position : integer, optional
            Position (index) in the list of links of the network where the link should be inserted. Default is insert at end of list (append)
            
        Raises
        ------
        TypeError
            If link is not an instance of ElectricalLink
        """
        if not isinstance(link,ElectricalLink):
            raise TypeError("Only a ElectricalLink object can be added")
        else:
            super().add_link(link,position=position)
            
    def add_half_link(self,half_link,position=None):
        """Adds half_link to the network. 
        
        A half link can only added if it a ElectricalHalfLink.
        If the start node is not yet added to the network, it
        will be added to the list of nodes.
        
        Parameters
        ----------
        half_link : ElectricalHalfLink
            Half link object to be added to the network
            
        Raises
        ------
        TypeError
            If half_link is not an instance of ElectricalHalfLink
        """
        if not isinstance(half_link,ElectricalHalfLink):
            raise TypeError("Only a ElectricalHalfLink object can be added")
        else:
            super().add_half_link(half_link,position=position)
    
    def get_nodes(self,node_types=list()):
        """Iterates over all the nodes in the list of nodes, with the specified node type.
        
        Parameters
        ----------
        node_types : list, optional
            List of node type of the nodes to be yielded. If empy, all the nodes are yielded. Default is an empty list.
            
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
                
    def add_network(self,network):
        """ Adds network to the network. 
        A network can only be added if it a ElectricalNetwork.
        
        Parameters
        ----------
        network : ElectricalNetwork
            The network to be added
        
        Raises 
        ------
        TypeError
            If network is not a ElectricalNetwork instance
        """
        if not isinstance(network,ElectricalNetwork):
            raise TypeError("Only a ElectricalNetwork object can be added")
        else:
            super().add_network(network)
            
    def make_admittance_matrix(self):
        """Makes the admittance matrix Y for the unscaled network. 
        Assigns a number to all nodes and links.
        """
        row = []
        col = []
        data = []
        for ind_n,n in enumerate(self.get_nodes()):
            n.number = ind_n
            # diagonal elements:
            data_ii = 0
            for e in n.get_out_links():
                if e.link_type == 'short_line':# HalfLinks (that are not nodal shunts) and DummyLinks should not be taken into account
                    data_ii = data_ii + np.complex(e.g,e.b)
                elif e.link_type == 'pi_line':
                    data_ii = data_ii + np.complex(e.g+e.g_sh/2,e.b+e.b_sh/2)
                elif e.link_type == 'pi_line_trafo':
                    data_ii = data_ii + (1/e.ratio**2)*np.complex((e.g+e.g_sh/2),(e.b+e.b_sh/2))
                elif e.link_type == 'nodal_shunt':
                    data_ii = data_ii + np.complex(e.g,e.b)
            for e in n.get_in_links():
                if e.link_type == 'short_line':
                    data_ii = data_ii + np.complex(e.g,e.b)
                elif e.link_type == 'pi_line':
                    data_ii = data_ii + np.complex(e.g+e.g_sh/2,e.b+e.b_sh/2)
                elif e.link_type == 'pi_line_trafo':
                    data_ii = data_ii + np.complex(e.g+e.g_sh/2,e.b+e.b_sh/2)
            row.append(n.number)
            col.append(n.number)
            data.append(data_ii)
        for ind_e,e in enumerate(self.get_links()):
            e.number = ind_e
            # off-diagonal elements
            if e.link_type == 'short_line':
                row.append(e.start_node.number)
                col.append(e.end_node.number)
                data.append(-np.complex(e.g,e.b))
                row.append(e.end_node.number)
                col.append(e.start_node.number)
                data.append(-np.complex(e.g,e.b))
            elif e.link_type == 'pi_line':
                row.append(e.start_node.number)
                col.append(e.end_node.number)
                data.append(-np.complex(e.g,e.b))
                row.append(e.end_node.number)
                col.append(e.start_node.number)
                data.append(-np.complex(e.g,e.b))
            elif e.link_type == 'pi_line_trafo':
                row.append(e.start_node.number)
                col.append(e.end_node.number)
                data.append(-np.complex(e.g,e.b)/e.n)
                row.append(e.end_node.number)
                col.append(e.start_node.number)
                data.append(-np.complex(e.g,e.b)/e.n.conj())
        self.Y = sps.csr_matrix((data,(row,col)),shape = (len(list(self.get_nodes())),len(list(self.get_nodes()))))
        
    def initialize(self):
        """Initializes the network. 
        The admittance matrix for the network is made.
        """
        self.make_admittance_matrix()
        
    def get_x_entries(self,formulation=None):
        """Return all the nodes, links, and half links that have an entry in variable vector x.
        It also returns all the nodes that have unknown voltage angle, and all the nodes that have unknown voltage amplitude.
        
        Parameters
        ----------
        formulation : string, optional
            formulation used to form system of equations. Default is None.
            
        Returns
        -------
        x_entries : list
            List of all the nodes, links, and half links that contribute to x.
        unknown_delta_nodes : list
            List of all the nodes that have unknown voltage angle
        unknown_V_nodes : list
            List of all the nodes that have unknown voltage amplitude
        """ 
        unknown_delta_nodes = list(self.get_nodes([1,2,6]))
        unknown_V_nodes = list(self.get_nodes([2]))
        x_entries = unknown_delta_nodes + unknown_V_nodes
        return x_entries, unknown_delta_nodes, unknown_V_nodes
    
    def get_F_entries(self,formulation=None):
        """Return all the nodes, links, and half links that have an entry in function vector F.
        It also returns the nodes that have an entry in the active power equation, and those that have an entry in the reactive power equation.
        
        Parameters
        ----------
        formulation : string, optional
            Formulation used to form system of equations. Default is None, which corresponds to 'complex_power'. Options are 'complex_power' or None.
            
        Returns
        -------
        F_entries : list
            List of all the nodes, links, and half links that contribute to F.
        known_P_nodes : list
            List of all the nodes that contribute to the active power equation
        known_Q_nodes : list
            List of all the nodes that contribute to the reactive power equation
        """        
        known_P_nodes = list(self.get_nodes([1,2,3,5,6])) 
        known_Q_nodes = list(self.get_nodes([2,4,5,6]))
        F_entries = known_P_nodes + known_Q_nodes
        return F_entries, known_P_nodes, known_Q_nodes
    
    def set_x_init(self,formulation='complex_power',scale_var=None,scale_var_params=None,**kwargs):
        """Creates the (initial guess for) vector x, based on the current network parameters.

        Parameters
        ----------
        formulation : string, optional
            Formulation used to form system of equations. Default is 'complex_power. Options are 'complex_power'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        x0 : np array
           Initial guess for variable vector x
        """
        x_entries, unknown_delta_nodes, unknown_V_nodes = self.get_x_entries(formulation=formulation)
        x0 = np.zeros(len(x_entries)) 
        for ind_el,el in enumerate(unknown_delta_nodes):
            x0[ind_el] = el.get_delta(scale_var=scale_var,scale_var_params=scale_var_params)       
        for ind_el,el in enumerate(unknown_V_nodes):
            x0[ind_el+len(unknown_delta_nodes)] = el.get_V(scale_var=scale_var,scale_var_params=scale_var_params)  
        return x0
    
    def update(self,x,formulation='complex_power',scale_var=None,scale_var_params=None,**kwargs):
        """Updates the network given variable vector x
        
        Parameters
        ----------
        x : np array
            Variable vector x
        formulation : string, optional
            Formulation used to form system of equations. Default is 'complex_power. Options are 'complex_power'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.    
        """
        if not self.networks: # network does not consist of subnetworks; update the network
            x_entries, unknown_delta_nodes, unknown_V_nodes = self.get_x_entries(formulation=formulation)
            for ind_n,n in enumerate(unknown_delta_nodes):
                delta = x[ind_n]
                if scale_var == 'per_unit':
                    delta *= scale_var_params['deltabase']
                n.delta = delta
            for ind_n,n in enumerate(unknown_V_nodes):
                V = x[ind_n + len(unknown_delta_nodes)]
                if scale_var == 'per_unit':
                    V *= scale_var_params['Vbase']
                n.V = V
            #update active and reactive power on half link
            for ind_n,n in enumerate(self.get_nodes()):
                if n.half_links:
                    for hl in n.get_half_links():
                        if hl.link_type == 'nodal_shunt':
                            P = hl.active_power(scale_var=scale_var,scale_var_params=scale_var_params)
                            Q = hl.reactive_power(scale_var=scale_var,scale_var_params=scale_var_params)
                            if scale_var == 'per_unit':
                                P *= scale_var_params['Sbase']
                                Q *= scale_var_params['Sbase']
                            hl.P = P
                            hl.Q = Q
            for ind_e,e in enumerate(self.get_links()):
                if (isinstance(e.start_node,ElectricalNode) and isinstance(e.end_node,ElectricalNode)): # non-coupling links
                    Pstart = e.active_power_start(scale_var=scale_var,scale_var_params=scale_var_params)
                    Qstart = e.reactive_power_start(scale_var=scale_var,scale_var_params=scale_var_params)
                    Pend = e.active_power_end(scale_var=scale_var,scale_var_params=scale_var_params)
                    Qend = e.reactive_power_end(scale_var=scale_var,scale_var_params=scale_var_params)
                    if scale_var == 'per_unit':
                        Pstart *= scale_var_params['Sbase']
                        Qstart *= scale_var_params['Sbase']
                        Pend *= scale_var_params['Sbase']
                        Qend *= scale_var_params['Sbase']
                    e.Pstart = Pstart
                    e.Qstart = Qstart
                    e.Pend = Pend
                    e.Qend = Qend
        else:
            x_entries, unknown_delta_nodes, unknown_V_nodes = self.get_x_entries(formulation=formulation)
            for net in self.get_networks():                
                indices_delta = list()
                indices_V = list()
                x_entries_net, unknown_delta_nodes_net, unknown_V_nodes_net = net.get_x_entries(formulation=formulation)
                for ind_el,el in unknown_delta_nodes_net:
                    indices_delta.append(unknown_delta_nodes.index(el))
                for ind_el,el in unknown_V_nodes_net:
                    indices_V.append(unknown_V_nodes.index(el))
                x_net = x[indices_delta + indices_V]
                # update subnetworks
                net.update(x_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    
    def update_full(self,x,formulation='complex_power',scale_var=None,scale_var_params=None,**kwargs):
        """Updates the full network given variable vector x.
        Unlike update(x), not only the values from x are updated, but also all
        parameters not included in x.
        
        Parameters
        ----------
        x : np array
            Variable vector x
        formulation : string, optional
            Formulation used to form system of equations. Default is 'complex_power. Options are 'complex_power'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        delta_vec : np array
            Array with all nodal voltage angles
        V_mag_vec : np array
            Array with all nodal voltage amplitudes
        S_inj : np array
            Array with all injected complex powers
        P_edge : np array
            Array with all link active powers
        Q_edge : np array
            Array with all link reactive powers
        """
        self.update(x,formulation='complex_power',scale_var=scale_var,scale_var_params=scale_var_params)
        delta_vec = np.zeros(len(list(self.get_nodes())))
        V_mag_vec = np.zeros(len(list(self.get_nodes())))
        for ind_n,n in enumerate(self.get_nodes()):
            delta_vec[ind_n] = n.delta
            V_mag_vec[ind_n] = n.V
            if n.node_type == 1 or n.node_type == 3: # nodes with unknown injected reactive power
                Qinj = -n.node_law(scale_var=scale_var,scale_var_params=scale_var_params)[1]
                if scale_var == 'per_unit':
                    Qinj *= scale_var_params['Sbase']
                n.half_links[0].Q = Qinj
            elif n.node_type == 4: # nodes with unknown injected active power
                Pinj = -n.node_law(scale_var=scale_var,scale_var_params=scale_var_params)[0]
                if scale_var == 'per_unit':
                    Pinj *= scale_var_params['Sbase']
                n.half_links[0].P = Pinj
            elif n.node_type == 0: # nodes with both injected reactive and active power unknown
                Pinj,Qinj = n.node_law(scale_var=scale_var,scale_var_params=scale_var_params)
                if scale_var == 'per_unit':
                    Pinj *= scale_var_params['Sbase']
                    Qinj *= scale_var_params['Sbase']
                if n.half_links:
                    n.half_links[0].P = -Pinj
                    n.half_links[0].Q = -Qinj
                else:
                    hl_name = n.name + "_hl"
                    ElectricalHalfLink(hl_name,n,P=-Pinj,Q=-Qinj)
        S_inj = np.zeros(len(list(self.get_nodes())),dtype=complex)
        for ind_n,n in enumerate(self.get_nodes()):
            for hl in n.get_half_links():
                S_inj[ind_n] += hl.P + hl.Q*1j
        P_edge = np.zeros(2*len(list(self.get_links())))
        Q_edge = np.zeros(2*len(list(self.get_links())))
        for ind_e,e in enumerate(self.get_links()):
            P_edge[ind_e] = e.Pstart
            Q_edge[ind_e] = e.Qstart
            P_edge[ind_e+len(list(self.get_links()))] = e.Pend
            Q_edge[ind_e+len(list(self.get_links()))] = e.Qend
        return delta_vec,V_mag_vec,S_inj,P_edge,Q_edge
    
    def reset_network(self,x_init,formulation='complex_power',scale_var=None,scale_var_params=None,**kwargs):
        """Resets the full network to initial conditions given initial guess vector x.
        That is, the injected active or reactive power on half links are set to zero for nodes with node type indicating that
        those are unknown. Also, any half links for reference nodes are removed. 
        
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
        self.update_full(x_init,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        for ind_n,n in enumerate(self.get_nodes()):
            if n.node_type == 1 or n.node_type == 3: # nodes with unknown injected reactive power
                n.half_links[0].Q = 0
            elif n.node_type == 4: # nodes with unknown injected active power
                n.half_links[0].P = 0
            elif n.node_type == 0: # nodes with both injected reactive and active power unknown
                if n.half_links:
                    n.half_links = list()
                    
    def solve_network(self,tol,max_iter,*args,formulation='complex_power',scale_var=None,scale_var_params=None,solver='NR',D_F=np.array([]),D_x=np.array([]),P_F=np.array([]),P_x=np.array([]),det_tol=None,return_all_x=False,lin_solver='solve',max_iter_lin=None,**kwargs):
        """Solves the steady-state load flow problem for the electrical network.

        Parameters
        ----------
        tol : float
            Tolerance for solver.
        max_iter : int
            Maximum number of iterations of solver.
        formulation : string, optional
            Formulation used to form system of equations. Default is 'complex_power'. Options are 'complex_power'.
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
        delta_sol : np array
            Array with all unscaled nodal voltage angles
        V_sol : np array
            Array with all unscaled nodal voltage amplitudes
        S_inj : np array
            Array with all unscaled injected complex powers
        P_edge : np array
            Array with all unscaled link active powers
        Q_edge : np array
            Array with all unscaled link reactive powers
        """
        # initiliaze
        x0 = self.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        nlsys = NonLinearSystemElectrical(self,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
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
            x_sol = nrsolve.solve(nlsys,x0,max_iter,D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x)
        iters =nrsolve.iters
        err_vec = nrsolve.err_vec
        # update the rest of the network
        delta_sol,V_sol,S_inj,P_edge,Q_edge = self.update_full(x_sol,formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        if return_all_x and solver == 'NR':
            return x_sol,x_mat,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge
        else:
            return x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge
    
# ===========================================================================
class ElectricalNode(Node):
    """Electrical node class.
    
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
        Type of the node, 0 = slack node, 1 = generator node, 2 = load node, 3 = PVdelta node, 4 = QVdelta node, 5 = PQVdelta node, 6 = PQV node.
    _V : float
        Nodal voltage amplitude, unscaled.
    _delta : float
        Nodal voltage angle, unscaled.
    """  
    def __init__(self,name,x=0.,y=0.,node_type=0,V=1.,delta=0.,P=None,Q=None):
        """Creates an ElectricalNode object.
        
        Parameters
        ----------
        name : str
            Name of the node.
        x : float, optional
            x coordinate of the node. Default is 0.
        y : float, optional
            y coordinate of the node. Default is 0.
        node_type : int
            Type of the node, 0 = slack node, 1 = generator node, 2 = load node, 3 = PVdelta node, 4 = QVdelta node, 5 = PQVdelta node, 6 = PQV node.
        V : float, optional
            Nodal voltage amplitude, unscaled. Default is 1. 
        delta : float, optional
            Nodal voltage angle, unscaled. Default is 0.
        P : float, optional
            Total injected active power, unscaled. One half link is made with this power. Default is None. If both P and Q are None, no half link is created
        Q : float, optional
            Total injected reactive power, unscaled. One half links is made with this power. Default is None. If both P and Q are None, no half link is created
        """
        super().__init__(name,x=x,y=y)
        self.node_type = node_type
        self.V = V
        self.delta = delta
        if (P != None) and (Q != None): # Need to check if not None, because if P=0 or Q=0 a halflink should be made
            hl_name = name + "_hl"
            ElectricalHalfLink(hl_name,self,P=P,Q=Q,bc_type=1)
        elif P != None:
            hl_name = name + "_hl"
            ElectricalHalfLink(hl_name,self,P=P,bc_type=2)
        elif Q != None:
            hl_name = name + "_hl"
            ElectricalHalfLink(hl_name,self,Q=Q,bc_type=3)
        self.color = 'r'
        
    @property
    def V(self): 
        """getter of V.
        """
        return self._V
    
    @V.setter
    def V(self,V):
        """setter of V.
        """
        self._V = V
    
    def get_V(self,scale_var=None,scale_var_params=None):
        """Get voltage amplitude, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        V : float
            Possibly scaled voltage amplitude.
        """
        V = self.V
        if scale_var == 'per_unit':
            Vb = scale_var_params['Vbase']
            V = V/Vb
        return V
    
    @property
    def delta(self): 
        """getter of delta.
        """
        return self._delta
    
    @delta.setter
    def delta(self,delta):
        """setter of delta.
        """
        self._delta = delta
        
    
    def get_delta(self,scale_var=None,scale_var_params=None):
        """Get voltage angle, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        delta : float
            Possibly scaled voltage angle.
        """
        delta = self.delta
        if scale_var == 'per_unit':
            deltab = scale_var_params['deltabase']
            delta = delta/deltab
        return delta
    
    def node_law(self,network=None,scale_var=None,scale_var_params=None):
        """Node law for an electrical node, which is conservation of complex power. The sum of the active and reactive powers of all incoming and outgoing links and half links.
        
        Parameters
        ----------
        network : Network, optional
            Network object in which the node law must be used. Default in None, such that all links connected to this node are used.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        
        Returns
        -------
        fP : float
            The sum of the active powers of all incoming and outgoing links and half links.
        fQ : float
            The sum of the reactive powers of all incoming and outgoing links and half links.
        """
        fP = 0.
        fQ = 0.
        for e in self.get_in_links():
            if isinstance(network,Network):
                if e in network.get_links():
                    fP += e.get_Pend(scale_var=scale_var,scale_var_params=scale_var_params)
                    fQ += e.get_Qend(scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                fP += e.get_Pend(scale_var=scale_var,scale_var_params=scale_var_params)
                fQ += e.get_Qend(scale_var=scale_var,scale_var_params=scale_var_params)
        for e in self.get_out_links(): # both links and half links
            if isinstance(network,Network):
                if e in network.get_links(): # only links, not half links
                    fP += e.get_Pstart(scale_var=scale_var,scale_var_params=scale_var_params)
                    fQ += e.get_Qstart(scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                if not isinstance(e,ElectricalHalfLink):
                    fP += e.get_Pstart(scale_var=scale_var,scale_var_params=scale_var_params)
                    fQ += e.get_Qstart(scale_var=scale_var,scale_var_params=scale_var_params)
        for hl in self.get_half_links():
            fP += hl.get_P(scale_var=scale_var,scale_var_params=scale_var_params)
            fQ += hl.get_Q(scale_var=scale_var,scale_var_params=scale_var_params)
        return fP,fQ

# ===========================================================================
class ElectricalLink(Link):
    """Electrical link class. 
    
    Attributes
    ----------
    name : str
        The name of the link.
    start_node : Node
        Start node of the link.
    end_node : Node
        End node of the link.
    number : int
        Number of the link.
    color : str
        Color used for plotting.
    _Pstart : float
        Active power at start of the link, unscaled.
    _Qstart : float 
        Reactive power at start of the link, unscaled.
    _Pend : float
        Active power at end of the link, unscaled.
    _Qend : float 
        Reactive power at end of the link, unscaled.
    """
    def __init__(self,name,start_node,end_node,Pstart=0.,Qstart=0.,Pend=0.,Qend=0.,bc_type=0,link_type='dummy',link_params=dict()):
        """Creates a ElectricalLink object.
        
        Parameters
        ----------
        name : str
            Name of the link.
        start_node : Node
            Start node of the link.
        end_node : Node
            End node of the link.
        Pstart : float, optional
            Active power at start of the link. Default is 0.
        Qstart : float, optional
            Reactive power at start of the link. Default is 0.
        Pend : float, optional
            Active power at end of the link. Default is 0.
        Qend : float, optional 
            Reactive power at end of the link. Default is 0.
        bc_type : int, optional
            Boundary condition of the link. 0 = everything unknown, 1 = Pstart known, 2 = Qstart known, 3 = Pstart and Qstart known, 4 = Pend known, 5 = Qend known, 6 = Pend and Qend known. Default is 0.
        link_type : string, optional
            Type of the link. Determines the link equations. Default is 'dummy'. Options are 'dummy', 'short_line', 'pi_line', or 'pi_line_trafo'. 
        link_params: dict, optional
            Dictionary of link parameters needed by the specific link equation. Default is an empty dict.
            
        Raises
        ------
        TypeError
            If start_node or end_node is not an instance of Node
        ValueError
            If link_type is not a valid link type
        """
        super().__init__(name,start_node,end_node)
        self.Pstart = Pstart
        self.Qstart = Qstart
        self.Pend = Pend
        self.Qend = Qend
        self.bc_type = bc_type
        self.link_type = link_type
        self.link_params = link_params
        if link_type == 'dummy':
            Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.dummy()
        elif link_type == 'short_line':
            Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.short_line(link_params['b'],link_params['g'])
            self.b = link_params['b']
            self.g = link_params['g']
        elif link_type == 'pi_line':
            Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.pi_line(link_params['b'],link_params['g'],link_params['b_sh'],link_params['g_sh'])
            self.b = link_params['b']
            self.g = link_params['g']
            self.b_sh = link_params['b_sh']
            self.g_sh = link_params['g_sh']
        elif link_type == 'pi_line_trafo':
            Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.pi_line_trafo(link_params['b'],link_params['g'],link_params['b_sh'],link_params['g_sh'],link_params['ratio'],link_params['phase_shift'])
            self.b = link_params['b']
            self.g = link_params['g']
            self.b_sh = link_params['b_sh']
            self.g_sh = link_params['g_sh']
            self.ratio = link_params['ratio']
            self.phase_shift = link_params['phase_shift']
            self.n = self.ratio*np.exp(1j*self.phase_shift)
        else:
            raise ValueError("link_type should be either 'dummy', 'short_line', 'pi_line', or 'pi_line_trafo', not {}".format(link_type))
        self.Pstart_of_V_delta = Pstart_of_V_delta
        self.Qstart_of_V_delta = Qstart_of_V_delta
        self.Pend_of_V_delta = Pend_of_V_delta
        self.Qend_of_V_delta = Qend_of_V_delta
        self.color = 'r'
    
    def set_type(self,link_type,link_params,bc_type=None):
        """Set or change the link type, and the corresponding link equations

        Parameters
        ----------
        link_type : str
            (New) type of the link. Must be 'dummy', 'short_line', 'pi_line', or 'pi_line_trafo'.
        link_params : dict
            Dictionary with the link parameters required for the (new) link type
        bc_type : int, optional
            Boundary condition of the link. 0 = everything unknown, 1 = Pstart known, 2 = Qstart known, 3 = Pstart and Qstart known, 4 = Pend known, 5 = Qend known, 6 = Pend and Qend known. Default is None, meaning that the bc_type is not changed.

        Raises
        ------
        ValueError
            If link_type is not a valid link type
        """
        self.link_type = link_type
        self.link_params = link_params
        if bc_type:
            self.bc_type = bc_type
        if link_type == 'dummy':
            Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.dummy()
        elif link_type == 'short_line':
            Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.short_line(link_params['b'],link_params['g'])
            self.b = link_params['b']
            self.g = link_params['g']
        elif link_type == 'pi_line':
            Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.pi_line(link_params['b'],link_params['g'],link_params['b_sh'],link_params['g_sh'])
            self.b = link_params['b']
            self.g = link_params['g']
            self.b_sh = link_params['b_sh']
            self.g_sh = link_params['g_sh']
        elif link_type == 'pi_line_trafo':
            Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.pi_line_trafo(link_params['b'],link_params['g'],link_params['b_sh'],link_params['g_sh'],link_params['ratio'],link_params['phase_shift'])
            self.b = link_params['b']
            self.g = link_params['g']
            self.b_sh = link_params['b_sh']
            self.g_sh = link_params['g_sh']
            self.ratio = link_params['ratio']
            self.phase_shift = link_params['phase_shift']
            self.n = self.ratio*np.exp(1j*self.phase_shift)
        else:
            raise ValueError("link_type should be either 'dummy', 'short_line', 'pi_line', or 'pi_line_trafo', not {}".format(link_type))
        self.Pstart_of_V_delta = Pstart_of_V_delta
        self.Qstart_of_V_delta = Qstart_of_V_delta
        self.Pend_of_V_delta = Pend_of_V_delta
        self.Qend_of_V_delta = Qend_of_V_delta
        
    @property
    def Pstart(self): 
        """getter of Pstart.
        """
        return self._Pstart
    
    @Pstart.setter
    def Pstart(self,Pstart):
        """setter of P.
        """
        self._Pstart = Pstart
    
    def get_Pstart(self,scale_var=None,scale_var_params=None):
        """Get active power near start node, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Pstart : float
            Possibly scaled active power at start of the link.
        """
        Pstart = self.Pstart
        if scale_var == 'per_unit':
            Pb = scale_var_params['Sbase']
            Pstart = Pstart/Pb
        return Pstart
    
    @property
    def Qstart(self): 
        """getter of Qstart.
        """
        return self._Qstart
    
    @Qstart.setter
    def Qstart(self,Qstart):
        """setter of Qstart.
        """
        self._Qstart = Qstart
        
    
    def get_Qstart(self,scale_var=None,scale_var_params=None):
        """Get reactive power near start node, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Qstart : float
            Possibly scaled reactive power at start of the link.
        """
        Qstart = self.Qstart
        if scale_var == 'per_unit':
            Qb = scale_var_params['Sbase']
            Qstart = Qstart/Qb
        return Qstart
    
    @property
    def Pend(self): 
        """getter of Pend.
        """
        return self._Pend
    
    @Pend.setter
    def Pend(self,Pend):
        """setter of P.
        """
        self._Pend = Pend
    
    def get_Pend(self,scale_var=None,scale_var_params=None):
        """Get active power near end node, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Pend : float
            Possibly scaled activer power at end of the link. 
        """
        Pend = self.Pend
        if scale_var == 'per_unit':
            Pb = scale_var_params['Sbase']
            Pend = Pend/Pb
        return Pend
    
    @property
    def Qend(self): 
        """getter of Qend.
        """
        return self._Qend
    
    @Qend.setter
    def Qend(self,Qend):
        """setter of Qend.
        """
        self._Qend = Qend
        
    
    def get_Qend(self,scale_var=None,scale_var_params=None):
        """Get reactive power near the end node, optionally with scaling.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Qend : float
            Possibly scaled reactive power at end of the link. 
        """
        Qend = self.Qend
        if scale_var == 'per_unit':
            Qb = scale_var_params['Sbase']
            Qend = Qend/Qb
        return Qend
    
    def V_drop(self,scale_var=None,scale_var_params=None):
        """Voltage amplitude drop function. 
        
        Parameters
        ----------
        V_start : float
            Voltage amplitude at start.
        V_end : float 
            Voltage amplitude at end.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        dV : float
            Voltage amplitude drop.
        """
        return self.start_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params) - self.end_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params)
    
    def delta_drop(self,scale_var=None,scale_var_params=None):
        """Voltage angle drop function.
        
        Parameters
        ----------
        delta_start : float
            Voltage angle at start
        delta_end : float 
            Voltage angle at end
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        ddelta : float
            Voltage angle drop.
        """
        return self.start_node.get_delta(scale_var=scale_var,scale_var_params=scale_var_params) - self.end_node.get_delta(scale_var=scale_var,scale_var_params=scale_var_params)
    
    def active_power_start(self,scale_var=None,scale_var_params=None):
        """Determine the active power at the start of the link, as a function of nodal voltage amplitudes and angles.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Pstart : float
            Active power at start of the link.
        """
        return self.Pstart_of_V_delta(self.start_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params),self.start_node.get_delta(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_delta(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)
    
    def reactive_power_start(self,scale_var=None,scale_var_params=None):
        """Determine the reactive power at the start of the link, as a function of nodal voltage amplitudes and angles.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Qstart : float
            Rective power at start of the link.
        """
        return self.Qstart_of_V_delta(self.start_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params),self.start_node.get_delta(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_delta(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)
    
    def active_power_end(self,scale_var=None,scale_var_params=None):
        """Determine the active power at the end of the link, as a function of nodal voltage amplitudes and angles.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Pend : float
            Active power at end of the link.
        """
        return self.Pend_of_V_delta(self.start_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params),self.start_node.get_delta(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_delta(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)
    
    def reactive_power_end(self,scale_var=None,scale_var_params=None):
        """Determine the reactive power at the end of the link, as a function of nodal voltage amplitudes and angles. 
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Qend : float
            Reactive power at end of the link.
        """
        return self.Qend_of_V_delta(self.start_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params),self.start_node.get_delta(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_delta(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)
    
    def complex_power_loss(self):
        """Complex power loss over the link. 
        
        Returns
        -------
        Sloss : float
            Complex power loss.
        """
        return np.complex(self.Pstart+self.Pend,self.Qstart+self.Qend)
    
# ===========================================================================
class ElectricalHalfLink(HalfLink):
    """Electrical half link class.
    The default is an outflow half link
    
    Attributes
    ----------
    name : str
        Name of the half link.
    start_node : Node
        Start node of the half link.
    number : int
        number of the half link.
    _P : float
        Active power, unscaled.
    _Q : float 
        Reactive power, unscaled.
    """
    def __init__(self,name,start_node,P=0.,Q=0.,bc_type=0,link_type='flow',link_params=dict()):
        """Creates a GasHalfLink object
        
        Parameters
        ----------
        name : str
            Name of the half link.
        start_node : Node
            Start node of the half link.
        P : float, optional
            Active power. Default is 0.
        Q : float, optional 
            Reactive power. Default is 0.
        bc_type : int, optional
            Boundary condition of the half link. 0 = P and Q unknown, 1 = P and Q known, 2 = P known, 3 = Q known. Default is 0.
        link_type : string, optional
            Type of the half link, options are 'flow' or 'nodal_shunt'. Default is 'flow', which represents an in- or outflow.
        link_params : dict, optional
            Dictionary of halflink parameters needed for a specific halflink type. Default is an empty dict
        
        
        Raises
        ------
        TypeError
            If start_node is not an instance of Node
        ValueError
            If halflink_type is not 'flow' or 'nodal_shunt'.
        """
        super().__init__(name,start_node)
        self.bc_type = bc_type
        self.link_type = link_type
        self.link_params = link_params
        self.P = P
        self.Q = Q
        if link_type == 'flow':
            P_of_V, Q_of_V = halflink_electrical.flow()
        elif link_type == 'nodal_shunt':
            self.b = link_params['b']
            self.g = link_params['g']
            P_of_V, Q_of_V = halflink_electrical.shunt(link_params['b'],link_params['g'])
        else:
            raise ValueError("link_type should be either 'flow' or 'nodal_shunt', not {}".format(link_type))
        self.P_of_V = P_of_V
        self.Q_of_V = Q_of_V
        self.color = 'r'
    
    def set_type(self,link_type,link_params,bc_type=None):
        """Set or change the half link type, and the corresponding link equations.

        Parameters
        ----------
        link_type : str
            (New) type of the link. Must be 'flow' or 'nodal_shunt'
        link_params : dict
            Dictionary with the link parameters required for the (new) link type
        bc_type : int, optional
            Boundary condition of the half link. 0 = P and Q unknown, 1 = P and Q known, 2 = P known, 3 = Q known. Default is 0.

        Raises
        ------
        ValueError
            If link_type is not a valid half link type
        """
        self.link_type = link_type
        self.link_params = link_params
        if bc_type != None: # Need to check if not None, because if bc_type = 0, it should be changed
            self.bc_type = bc_type
        if link_type == 'flow':
            P_of_V, Q_of_V = halflink_electrical.flow()
        elif link_type == 'nodal_shunt':
            self.b = link_params['b']
            self.g = link_params['g']
            P_of_V, Q_of_V = halflink_electrical.shunt(link_params['b'],link_params['g'])
        else:
            raise ValueError("link_type should be either 'flow' or 'nodal_shunt', not {}".format(link_type))
        self.P_of_V = P_of_V
        self.Q_of_V = Q_of_V
        
    @property
    def P(self): 
        """getter of P.
        """
        return self._P
    
    @P.setter
    def P(self,P):
        """setter of P.
        """
        self._P = P
    
    def get_P(self,scale_var=None,scale_var_params=None):
        """Get activer power, optionally with scaling
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        P : float
            Possibly scaled active power.
        """
        P = self.P
        if scale_var == 'per_unit':
            Pb = scale_var_params['Sbase']
            P = P/Pb
        return P
    
    @property
    def Q(self): 
        """getter of Q.
        """
        return self._Q
    
    @Q.setter
    def Q(self,Q):
        """setter of Q.
        """
        self._Q = Q
        
    
    def get_Q(self,scale_var=None,scale_var_params=None):
        """Get reactive power, optionally with scaling
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Q : float
            Possibly scaled reactive power. 
        """
        Q = self.Q
        if scale_var == 'per_unit':
            Qb = scale_var_params['Sbase']
            Q = Q/Qb
        return Q
    
    def active_power(self,scale_var=None,scale_var_params=None):
        """Determine the active power of the half link, as a function of nodal voltage amplitude.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        P : float
            Active power at of the half link.
        """
        return self.P_of_V(self.start_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)
    
    def reactive_power(self,scale_var=None,scale_var_params=None):
        """Determine the reactive power of the half link, as a function of nodal voltage amplitude.
        
        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
            
        Returns
        -------
        Q : float
            Reactive power at of the half link.
        """
        return self.Q_of_V(self.start_node.get_V(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)
