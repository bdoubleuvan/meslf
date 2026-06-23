"""Classes needed to create a gas network."""
from meslf.networks.network import Network, Node, Link, HalfLink
from meslf.link_equations import hydraulic
from meslf.load_flow.system_of_equations import NonLinearSystemGas
from meslf.load_flow.non_linear_solvers import NR, NR_FD, Fsolver, Root
import warnings
import numpy as np
import scipy.sparse as sps
import pandas as pd

# ===========================================================================
class GasNetwork(Network):
    """Overall gas network class. Subclass of Network.

    Attributes
    ----------
    name : str
        The name of the network.
    nodes : list
        List of nodes in the network.
    links : list
        List of links in the network.
    networks : list
        List of (sub)networks in the network.
    A : scipy sparse csr_matrix
        Branch-nodal incidence matrix.
    """
    def __init__(self,name):
        """Creates a GasNetwork object.

        Parameters
        ----------
        name : str
            The name of the network.
        """
        super().__init__(name)
        self.A = None

    def add_node(self,node,position=None):
        """Adds a node to the network.
        A node is only added if it is a GasNode.

        Parameters
        ----------
        node : GasNode
            Node to be added to the network.
        position : integer, optional
            Position (index) in the list of nodes of the network where the node should be inserted. Default is insert at end of list (append).

        Warns
        ------
        UserWarning
            If node is not an instance of GasNode
        """
        if not isinstance(node,GasNode):
            warnings.warn("Only a GasNode object can be added")
        else:
            super().add_node(node,position=position)

    def add_link(self,link,position=None):
        """Adds link to the network.
        A link can only be added if it is a GasLink.
        If the start node or the end node are not yet added to the network, they
        will be added to the list of nodes.

        Parameters
        ----------
        link : GasLink
            Link to be added to the network.
        position : integer, optional
            Position (index) in the list of links of the network where the link should be inserted. Default is insert at end of list (append).

        Raises
        ------
        TypeError
            If link is not an instance of GasLink
        """
        if not isinstance(link,GasLink):
            raise TypeError("Only a GasLink object can be added")
        else:
            super().add_link(link,position=position)

    def add_half_link(self,half_link,position=None):
        """Adds half_link to the network.
        A half link can only be added if it a GasHalfLink

        If the start node is not yet added to the network, it
        will be added to the list of nodes.

        Parameters
        ----------
        half_link : GasHalfLink
            Half link to be added to the network.

        Raises
        ------
        TypeError
            If half_link is not an instance of GasHalfLink.
        """
        if not isinstance(half_link,GasHalfLink):
            raise TypeError("Only a GasHalfLink object can be added")
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
            The next Node instance in self.nodes of the requested node type
        """
        for n in super().get_nodes():
            if node_types:
                if n.node_type in node_types:
                    yield n
            else:
                yield n


    def add_network(self,network):
        """ Adds network to the network.
        A network can only be added if it a GasNetwork.

        Parameters
        ----------
        network : GasNetwork
            The network to be added

        Raises
        ------
        TypeError
            If network is not a GasNetwork instance
        """
        if not isinstance(network,GasNetwork):
            raise TypeError("Only a GasNetwork object can be added")
        else:
            super().add_network(network)

    def make_incidence_matrix(self):
        """Creates the branch-nodal incidence matrix A.
        It also assigns a number to all nodes and links.
        """
        row = []
        col = []
        data = []
        for ind_n,n in enumerate(self.get_nodes()):
            n.number = ind_n
        for ind_e,e in enumerate(self.get_links()):
            e.number = ind_e
            #outgoing edge
            if e.start_node in self.get_nodes(): # check if start_node is in the network, i.e. if it is a GasNode
                row.append(e.start_node.number)
                col.append(e.number)
                data.append(-1.)
            #incoming edge
            if e.end_node in self.get_nodes(): # check if end_node is in the network, i.e. if it is a GasNode
                row.append(e.end_node.number)
                col.append(e.number)
                data.append(1.)
        self.A = sps.csr_matrix((data,(row,col)),shape=(len(list(self.get_nodes())),len(list(self.get_links()))))

    def initialize(self):
        """Inializes the network.
        The branch-nodal incidense matrix for the network is made.
        """
        self.make_incidence_matrix()

    def get_x_entries(self,formulation=None):
        """Returns all the nodes, links, and half links that have an entry in variable vector x.

        Parameters
        ----------
        formulation : string, optional
            formulation used to form system of equations. Default is None (which corresponds to 'nodal'). Options are 'full', 'nodal' or None.

        Returns
        -------
        x_entries : list
           List of all the nodes, links, and half links that contribute to x.
        """
        unknown_p_nodes = list(self.get_nodes([1,2])) # only load nodes and slack nodes
        if formulation == 'full':
            unknown_q_links = list(self.get_links(link_types=['pipe_linear','pipe_low_pres_pole','pipe_low_pres_hagen_poiseuille','pipe_low_pres_churchill','pipe_high_pres_colebrook','pipe_high_pres_weymouth','pipe_high_pres_pole','pipe_high_pres_blasius','pipe_high_pres_churchill','compressor','resistor'],bc_types=[0]))
            x_entries = unknown_q_links + unknown_p_nodes
        else:
            x_entries = unknown_p_nodes
        return x_entries

    def get_F_entries(self,formulation=None):
        """Returns all the nodes, links, and half links that have an entry in function vector F

        Parameters
        ----------
        formulation : string, optional
            Formulation used to form system of equations. Default is None (which corresponds to 'nodal'). Options are 'full', 'nodal' or None.

        Returns
        -------
        F_entries : list
           List of all the nodes, links, and half links that contribute to F
        """
        known_q_nodes = list(self.get_nodes([1,3])) # only load nodes and ref load nodes
        F_entries = known_q_nodes
        if formulation == 'full':
            non_dummy_links = list(self.get_links(link_types=['pipe_linear','pipe_low_pres_pole','pipe_low_pres_hagen_poiseuille','pipe_low_pres_churchill','pipe_high_pres_colebrook','pipe_high_pres_weymouth','pipe_high_pres_pole','pipe_high_pres_blasius','pipe_high_pres_churchill','compressor','resistor']))
            F_entries += non_dummy_links
        return F_entries

    def set_x_init(self,formulation,scale_var=None,scale_var_params=None,**kwargs):
        """Creates the (initial guess for) vector x, based on the current network parameters.

        Parameters
        ----------
        formulation : string
            Formulation used to form system of equations. Options are 'full' or 'nodal'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        x0 : np array
           Variable vector x, based on the current network values.
        """
        if formulation == 'nodal':
            for link in self.get_links():
                if link.link_type in ['pipe_low_pres_hagen_poiseuille','pipe_low_pres_churchill','pipe_high_pres_colebrook','pipe_high_pres_blasius','pipe_high_pres_churchill','compressor']:
                    raise ValueError("Formulation is 'nodal', but the network contains at least one link of type {}. Use 'full' formulation instead!".format(link.link_type))
        x0 = np.zeros(len(self.get_x_entries(formulation=formulation)))
        for ind_el,el in enumerate(self.get_x_entries(formulation=formulation)):
            if isinstance(el,GasLink):
                x0[ind_el] = el.get_q(scale_var=scale_var,scale_var_params=scale_var_params)
            elif isinstance(el,GasNode):
                x0[ind_el] = el.get_p(scale_var=scale_var,scale_var_params=scale_var_params)
        return x0

    def update(self,x,formulation,scale_var=None,scale_var_params=None,**kwargs):
        """Updates the network given variable vector x

        Parameters
        ----------
        x : np array
            Variable vector x
        formulation : string
            Formulation used to form system of equations. Options are 'full' or 'nodal'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        """
        if not self.networks: # network does not consist of subnetworks; update the network
            if formulation == 'nodal':
                for ind_n,n in enumerate(self.get_x_entries(formulation=formulation)):
                    p = x[ind_n]
                    if scale_var == 'per_unit':
                        p *= scale_var_params['pbase']
                    n.p = p
                for ind_e,e in enumerate(self.get_links()):
                    if (isinstance(e.start_node,GasNode) and isinstance(e.end_node,GasNode)): # non-coupling links
                        if e.link_type == 'compressor':
                            warnings.warn('Nodal formulation should not be used for a network with a compressor. Using conservation of mass in end node to determine flow through compressor')
                            q_out = 0
                            for e_out in e.end_node.get_links():
                                if not (e_out == e):
                                    if e_out.flow():
                                        q_out += e_out.flow(scale_var=scale_var,scale_var_params=scale_var_params)
                                    else:
                                        q_out += e.get_q(scale_var=scale_var,scale_var_params=scale_var_params)
                            if scale_var == 'per_unit':
                                q_out *= scale_var_params['qbase']
                            e.q = q_out
                        else:
                            q = e.flow(scale_var=scale_var,scale_var_params=scale_var_params)
                            if scale_var == 'per_unit':
                                q *= scale_var_params['qbase']
                            e.q = q
            elif formulation == 'full':
                for ind_el,el in enumerate(self.get_x_entries(formulation=formulation)):
                    if isinstance(el,GasLink):
                        q = x[ind_el]
                        if scale_var == 'per_unit':
                            q *= scale_var_params['qbase']
                        el.q = q
                    elif isinstance(el,GasNode):
                        p = x[ind_el]
                        if scale_var == 'per_unit':
                            p *= scale_var_params['pbase']
                        el.p = p
            else:
                raise ValueError('Enter valid formulation. Either "nodal" or "full".')
        else:
            for net in self.get_networks():
                indices = list()
                for ind_el,el in enumerate(net.get_x_entries(formulation=formulation)):
                    indices.append(self.get_x_entries(formulation=formulation).index(el))
                x_net = x[indices]
                # update subnetworks
                net.update(x_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    def update_full(self,x,formulation='nodal',scale_var=None,scale_var_params=None,**kwargs):
        """Updates the full network given variable vector x.
        Unlike update(x), not only the values from x are updated, but also all
        parameters not included in x.

        Parameters
        ----------
        formulation : string
            Formulation used to form system of equations. Options are 'full' or 'nodal'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        p_vec : np array
            Array with all unscaled nodal pressures.
        q_vec : np array
            Array with all unscaled link flows.
        q_inj : np array
            Array with all unscaled nodal injeted flows.
        """
        self.update(x,formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        q_vec = np.zeros(len(list(self.get_links())))
        p_vec = np.zeros(len(list(self.get_nodes())))
        q_inj = np.zeros(len(list(self.get_nodes())))
        for ind_e,e in enumerate(self.get_links()):
            q_vec[ind_e] = e.q
        q_inj_calc = self.A.dot(q_vec)
        for n in self.get_nodes():
            p_vec[n.number] = n.p
            if n.node_type == 0 or n.node_type == 2: # nodes with unknown injected flow
                if n.half_links:
                    for ind_hl,hl in enumerate(n.get_half_links()):
                        if ind_hl == 0:
                            hl.q = q_inj_calc[n.number]
                        else:
                            n.half_links[0].q -= hl.get_q()
                else:
                    hl_name = n.name + "_hl"
                    GasHalfLink(hl_name,n,q_inj_calc[n.number])
            for hl in n.get_half_links():
                q_inj[n.number] += hl.q
        return p_vec,q_vec,q_inj

    def reset_network(self,x_init,formulation='nodal',scale_var=None,scale_var_params=None,**kwargs):
        """Resets the full network to initial conditions given initial guess vector x.


        Parameters
        ----------
        x_init : np array
            Vector with initial guess for x.
        formulation : string
            Formulation used to form system of equations. Options are 'full' or 'nodal'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.
        """
        self.update_full(x_init,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)

    def solve_network(self,tol,max_iter,*args,formulation=None,scale_var=None,scale_var_params=None,solver='NR',D_F=np.array([]),D_x=np.array([]),P_F=np.array([]),P_x=np.array([]),det_tol=None,return_all_x=False,lin_solver='solve',max_iter_lin=None,**kwargs):
        """Solves the steady-state load flow problem for the gas network.

        Parameters
        ----------
        tol : float
            tolerance for solver
        max_iter : int
            maximum number of iterations of solver
        formulation : string
            Formulation used to form system of equations. Options are 'full' or 'nodal'.
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
        p_sol : np array
            Vector with unscaled nodal pressures.
        q_sol : np array
            Vector with unscaled link flows.
        q_inj : np array
            Vector with unscaled injected nodal flows.
        """
        # initiliaze
        x0 = self.set_x_init(formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        nlsys = NonLinearSystemGas(self,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
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
        p_sol,q_sol,q_inj = self.update_full(x_sol,formulation,scale_var=scale_var,scale_var_params=scale_var_params)
        if return_all_x and solver == 'NR':
            return x_sol,x_mat,iters,err_vec,p_sol,q_sol,q_inj
        else:
            return x_sol,iters,err_vec,p_sol,q_sol,q_inj


# ===========================================================================
class GasNode(Node):
    """Gas node class.

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
        Type of the node, 0 = ref. node, 1 = load node, 2 = slack node, 3 = ref. load node.
    _p : float
        Nodal pressure, unscaled.
    """
    def __init__(self,name,x=0.,y=0.,node_type=0,p=0.,q=None):
        """Creates a GasNode object.

        Parameters
        ----------
        name : str
            Name of the node
        x : float, optional
            x coordinate of the node. Default is 0.
        y : float, optional
            y coordinate of the node. Default is 0.
        node_type : int
            Type of the node, 0 = ref node, 1 = load node, 2 = slack node, 3 = ref. load node.
        p : float, optional
            Nodal pressure, unscaled. Default is 0.
        q : float, optional
            Total injected flow, unscaled. One half link is made with this flow. Default is None, such that no half link is created.
        """
        super().__init__(name,x=x,y=y)
        self.node_type = node_type
        self.p = p
        if q != None:
            hl_name = name + "_hl"
            GasHalfLink(hl_name,self,q,bc_type=1) # creating the half link will automatically add it to the half_links list of the node
        self.color = 'g'

    @property
    def p(self):
        """getter of _p.
        """
        return self._p

    @p.setter
    def p(self,p):
        """setter of _p.
        """
        self._p = p

    def get_p(self,scale_var=None,scale_var_params=None):
        """Get nodal pressure, optionally with scaling.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        p : float
            Possible scaled nodal pressure.
        """
        p = self.p
        if scale_var == 'per_unit':
            pb = scale_var_params['pbase']
            p = p/pb
        return p

    def node_law(self,network=None,scale_var=None,scale_var_params=None):
        """Node law for a gas node, which is conservation of flow.
        The sum of the gas flows of all incoming and outgoing links and half links.

        Parameters
        ----------
        network : Network, optional
            Network in which the node law must be used. Only links in that network are taken into account. Default in None, such that all links connected to this node are used.
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        f : float
            The sum of the gas flows of all incoming and outgoing links and half links.
        """
        f = 0.
        for e in self.get_in_links():
            if isinstance(network,Network):
                if e in network.get_links():
                    f += e.get_q(scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                f += e.get_q(scale_var=scale_var,scale_var_params=scale_var_params)
        for e in self.get_out_links(): # both links and half links
            if isinstance(network,Network):
                if e in network.get_links(): # only links, not half links
                    f -= e.get_q(scale_var=scale_var,scale_var_params=scale_var_params)
                elif e in self.get_half_links():
                    f -= e.get_q(scale_var=scale_var,scale_var_params=scale_var_params)
            else:
                f -= e.get_q(scale_var=scale_var,scale_var_params=scale_var_params)
        return f

# ===========================================================================
class GasLink(Link):
    """Gas link class.

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
    """
    def __init__(self,name,start_node,end_node,q=0.,bc_type=0,link_type='dummy',link_params=dict(),link_eq_form='q_of_dp'):
        """Creates a GasLink object.

        Parameters
        ----------
        name : str
            Name of the link.
        start_node : Node
            Start node of the link.
        end_node : Node
            End node of the link.
        q : float, optional
            Link flow, unscaled. Default is 0.
        bc_type : int, optional
            Boundary condition of the link. 0 = q unknown, 1 = q known. Default is 0.
        link_type : string, optional
            Type of the link, e.g. a pipe or compressor. Determines the link equation. Default is 'dummy'. Options are 'dummy', 'pipe_linear', 'pipe_low_pres_pole', 'pipe_high_pres_colebrook', 'pipe_high_pres_weymouth', 'compressor', or 'resistor'.
        link_params: dict, optional
            Dictionary of link parameters needed by the specific link equation. Default is an empty dict.
        link_eq_form : string, optional
            Determines which link equation formulation is used. Default is 'q_of_dp', which uses the link flow as a function of pressure drop (i.e. it uses fa). The other option is 'dp_of_q', which uses the pressure drop as a function of link flow (i.e. it uses fb. When 'dp_of_q' is chosen, the 'nodal' formulation for the system of equation is no longer possible (that is, the code will still run, but it will give meaningless results).

        Raises
        ------
        TypeError
            If start_node or end_node is not an instance of Node.
        ValueError
            If link_type is not a valid link type.
        """
        super().__init__(name,start_node,end_node)
        self.q = q
        self.bc_type = bc_type
        self.link_type = link_type
        self.link_params = link_params
        self.link_eq_form = link_eq_form
        if link_type == 'dummy':
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.dummy()
        elif link_type == 'pipe_linear':
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_linear(link_params['alpha'])
        elif link_type == 'pipe_low_pres_pole':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_pole
            D = link_params['D']
            L = link_params['L']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,fric_fac,D,L)
        elif link_type == 'pipe_low_pres_hagen_poiseuille':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_hagen_poiseuille
            fric_fac_der_q = hydraulic.fric_fac_der_qhagen_poiseuille
            D = link_params['D']
            L = link_params['L']
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'pipe_low_pres_churchill':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_churchill
            fric_fac_der_q = hydraulic.fric_fac_der_qchurchill
            D = link_params['D']
            L = link_params['L']
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'pipe_high_pres_colebrook':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_colebrook
            fric_fac_der_q = hydraulic.fric_fac_der_qcolebrook
            D = link_params['D']
            L = link_params['L']
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'pipe_high_pres_weymouth':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_weymouth
            D = link_params['D']
            L = link_params['L']
            E = link_params['E']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure(carrier,fric_fac,D,L,E=E)
        elif link_type == 'pipe_high_pres_pole':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_pole
            D = link_params['D']
            L = link_params['L']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure(carrier,fric_fac,D,L)
        elif link_type == 'pipe_high_pres_blasius':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_blasius
            fric_fac_der_q = hydraulic.fric_fac_der_qblasius
            D = link_params['D']
            L = link_params['L']
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'pipe_high_pres_churchill':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_churchill
            fric_fac_der_q = hydraulic.fric_fac_der_qchurchill
            D = link_params['D']
            L = link_params['L']
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'compressor':
            r = link_params['r']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.compressor(r)
        elif link_type == 'resistor':
            C = link_params['C']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.resistor(C)
        else:
            raise ValueError('link_type is not a valid link type.')
        self.pipe_const = pipe_const
        self.dp = dp
        self.q_of_dp = q_of_dp
        self.dp_of_q = dp_of_q
        if link_eq_form == 'q_of_dp':
            self.f = fa
            self.df_ddp = dfa_ddp
            self.df_dq = dfa_dq
            self.df_dp = dfa_dp
        elif link_eq_form == 'dp_of_q':
            self.f = fb
            self.df_ddp = dfb_ddp
            self.df_dq = dfb_dq
            self.df_dp = dfb_dp
        else:
            raise ValueError('link_eq_form is not valid. It should be either "q_of_dp" or "dp_of_q", not {}.'.format(link_eq_form))
        self.ddp_dp = ddp_dp
        self.color = 'g'

    def set_type(self,link_type,link_params,link_eq_form='q_of_dp',bc_type=None):
        """Set or change the link type, and the corresponding link equations

        Parameters
        ----------
        link_type : str
            (New) type of the link. Must be 'dummy', 'pipe_linear', 'pipe_low_pres_pole', 'pipe_high_pres_colebrook','pipe_high_pres_weymouth', 'compressor', or 'resistor'
        link_params : dict
            Dictionary with the link parameters required for the (new) link type.
        link_eq_form : string, optional
            Determines which link equation formulation is used. Default is 'q_of_dp', which uses the link flow as a function of pressure drop (i.e. it uses fa). The other option is 'dp_of_q', which uses the pressure drop as a function of link flow (i.e. it uses fb). When 'dp_of_q' is chosen, the 'nodal' formulation for the system of equation is no longer possible (that is, the code will still run, but it will give meaningless results).
        bc_type : int, optional
            Boundary condition of the link. 0 = q unknown, 1 = q known. Default is None, meaning that the same bc_type is not changed.

        Raises
        ------
        ValueError
            If link_type is not a valid link type.
        """
        self.link_type = link_type
        self.link_params = link_params
        self.link_eq_form = link_eq_form
        if bc_type:
            self.bc_type = bc_type
        if link_type == 'dummy':
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.dummy()
        elif link_type == 'pipe_linear':
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_linear(link_params['alpha'])
        elif link_type == 'pipe_low_pres_pole':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_pole
            D = link_params['D']
            L = link_params['L']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,fric_fac,D,L)
        elif link_type == 'pipe_low_pres_hagen_poiseuille':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_hagen_poiseuille
            fric_fac_der_q = hydraulic.fric_fac_der_qhagen_poiseuille
            D = link_params['D']
            L = link_params['L']
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'pipe_low_pres_churchill':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_churchill
            fric_fac_der_q = hydraulic.fric_fac_der_qchurchill
            D = link_params['D']
            L = link_params['L']
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'pipe_high_pres_colebrook':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_colebrook
            fric_fac_der_q = hydraulic.fric_fac_der_qcolebrook
            D = link_params['D']
            L = link_params['L']
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'pipe_high_pres_weymouth':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_weymouth
            D = link_params['D']
            L = link_params['L']
            E = link_params['E']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure(carrier,fric_fac,D,L,E=E)
        elif link_type == 'pipe_high_pres_pole':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_pole
            D = link_params['D']
            L = link_params['L']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure(carrier,fric_fac,D,L)
        elif link_type == 'pipe_high_pres_blasius':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_blasius
            fric_fac_der_q = hydraulic.fric_fac_der_qblasius
            D = link_params['D']
            L = link_params['L']
            eps = 0
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'pipe_high_pres_churchill':
            carrier = link_params['carrier']
            fric_fac = hydraulic.fric_fac_churchill
            fric_fac_der_q = hydraulic.fric_fac_der_qchurchill
            D = link_params['D']
            L = link_params['L']
            eps = link_params['eps']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure_implicit(carrier,fric_fac,D,L,eps,fric_fac_der_q=fric_fac_der_q)
        elif link_type == 'compressor':
            r = link_params['r']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.compressor(r)
        elif link_type == 'resistor':
            C = link_params['C']
            pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.resistor(C)
        else:
            raise ValueError('link_type is not a valid link type.')
        self.pipe_const = pipe_const
        self.dp = dp
        self.q_of_dp = q_of_dp
        self.dp_of_q = dp_of_q
        if link_eq_form == 'q_of_dp':
            self.f = fa
            self.df_ddp = dfa_ddp
            self.df_dq = dfa_dq
            self.df_dp = dfa_dp
        elif link_eq_form == 'dp_of_q':
            self.f = fb
            self.df_ddp = dfb_ddp
            self.df_dq = dfb_dq
            self.df_dp = dfb_dp
        else:
            raise ValueError('link_eq_form is not valid. It should be either "q_of_dp" or "dp_of_q", not {}.'.format(link_eq_form))
        self.ddp_dp = ddp_dp

    @property
    def q(self):
        """getter of q.
        """
        return self._q

    @q.setter
    def q(self,q):
        """setter of q.
        """
        self._q = q

    def get_q(self,scale_var=None,scale_var_params=None):
        """Get link flow, optionally with scaling

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        q : float
            Possible scaled link flow.
        """
        q = self.q
        if scale_var == 'per_unit':
            qb = scale_var_params['qbase']
            q = q/qb
        return q

    def pres_drop(self,scale_var=None,scale_var_params=None):
        """Determines the pressure drop over the link.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        dp : float
            Pressure drop over link. Pressure of start node - pressure at end node.
        """
        return self.start_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)-self.end_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)

    def pres_drop_func(self,scale_var=None,scale_var_params=None):
        """Determines the pressure drop function over the link.
        The pressure drop function is determined by the link type.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        dp : float
            Pressure drop function over the link. Determined by the link type.
        """
        try:
            pstart = self.start_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)
        except AttributeError:
            pstart = None
        try:
            pend = self.end_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params)
        except AttributeError:
            pend = None
        return self.dp(pstart,pend)

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

    def link_equation(self,scale_var=None,scale_var_params=None):
        """Evaluates the link equation.
        The link equation is determined by the link type.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        f : float
            Link equation f(q,p_start,p_end)
        """
        return self.f(self.get_q(scale_var=scale_var,scale_var_params=scale_var_params),self.start_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params),self.end_node.get_p(scale_var=scale_var,scale_var_params=scale_var_params),scale_var=scale_var,scale_var_params=scale_var_params)

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
        return self.df_ddp(self.get_q(scale_var=scale_var,scale_var_params=scale_var_params),pstart,pend,scale_var=scale_var,scale_var_params=scale_var_params)

    def f_der_p(self,scale_var=None,scale_var_params=None):
        """Determines the derivative of the link equation to the start and end pressures.

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
        return self.df_dp(self.get_q(scale_var=scale_var,scale_var_params=scale_var_params),pstart,pend,scale_var=scale_var,scale_var_params=scale_var_params)

    def f_der_q(self,scale_var=None,scale_var_params=None):
        """Determines the derivative of the link equation to the link flow.

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        df_dq : float
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
        return self.df_dq(self.get_q(scale_var=scale_var,scale_var_params=scale_var_params),pstart,pend,scale_var=scale_var,scale_var_params=scale_var_params)

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



# ===========================================================================
class GasHalfLink(HalfLink):
    """Gas half link class.
    The default is an outflow half link.

    Attributes
    ----------
    name : str
        Name of the half link.
    start_node : Node
        Start node of the half link.
    number : int
        Number of the half link.
    _q : float
        Gas flow, unscaled.
    """
    def __init__(self,name,start_node,q,bc_type=0,link_type='flow',link_params=dict()):
        """Creates a GasHalfLink object

        Parameters
        ----------
        name : str
            Name of the half link.
        start_node : Node
            Start node of the half link.
        q : float
            Gas flow, unscaled.
        bc_type : int, optional
            Boundary condition of the half link. 0 = q unknown, 1 = q known. Default is 0.
        link_type : string, optional
            Type of the half link, options are 'flow'. Default is 'flow', which represents in- or outflow.
        link_params : dict, optional
            Dictionary of halflink parameters needed for a specific halflink type. Default is an empty dict.

        Raises
        ------
        TypeError
            If start_node is not an instance of Node.
        ValueError
            If halflink_type is not 'flow'.
        """
        super().__init__(name,start_node)
        self.q = q
        self.bc_type = bc_type
        self.link_type = link_type
        if link_type == 'flow':
            self.link_params = link_params
        else:
            raise ValueError("link_type should be 'flow', not {}".format(link_type))
        self.color = 'g'

    @property
    def q(self):
        """getter of q.
        """
        return self._q

    @q.setter
    def q(self,q):
        """setter of q.
        """
        self._q = q

    def get_q(self,scale_var=None,scale_var_params=None):
        """Get half link flow, optionally with scaling

        Parameters
        ----------
        scale_var : string, optional
            How to scale the variable. Default is no scaling.
        scale_var_params : dict, optional
            Dictionary with values needed for scaling variables. Default is None.

        Returns
        -------
        q : float
            Possible scaled half link flow
        """
        q = self.q
        if scale_var == 'per_unit':
            qb = scale_var_params['qbase']
            q = q/qb
        return q
