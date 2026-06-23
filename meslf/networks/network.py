"""Base classes needed to create a network. Includes Network, Node, Link, and HalfLink"""
import abc
from meslf.utils.list_manipulation import merge_sorted
import matplotlib.patches as patch
import math
import numpy as np
import pandas as pd

# ===========================================================================
class Network(metaclass=abc.ABCMeta):
    """Overall network class.

    Attributes
    ----------
    name : str
        The name of the network.
    nodes : list
        List of nodes in the network
    links : list
        List of links in the network
    networks : list
        List of (sub)networks in the network
    """
    def __init__(self,name):
        """Creates a Network object

        Parameters
        ----------
        name : str
            name of the network
        """
        super().__init__()
        self.name = name #network name
        self.nodes = list() #list of nodes in the network
        self.links = list() #list of links in the network
        self.half_links = list() # list of half links in the network
        self.networks = list() # list of subnetworks in the network
        self.level = [name] # the hierarchical level of this network with respect to others. List of names of all the networks, this network is a subnetwork of, including itself.

    def add_node(self,node,position=None):
        """Adds node to the network, i.e. adds node to list of nodes.

        Parameters
        ----------
        node : Node
            Node object to be added to the network
        position : integer, optional
            Position (index) in the list of nodes of the network where the node should be inserted. Default is insert at end of list (append)

        Raises
        ------
        TypeError
            If node is not an instance of Node
        """
        if not isinstance(node,Node):
            raise TypeError("Only a Node object can be added")
        if position == None:
            self.nodes.append(node)
        else:
            self.nodes.insert(position,node)

    def remove_node(self,node):
        """Removes node from the list of nodes.
        If any links or half_links are connected to that node, those links are removed as well, both from the network as from the node at the other side of the link

        Parameters
        ----------
        node : Node
            Node object to be removed from the network

        Raises
        ------
        TypeError
            If node is not an instance of Node
        """
        if not isinstance(node,Node):
            raise TypeError("Only a Node object can be removed")
        for e in node.get_links():
            if e in self.get_links():
                self.remove_link(e)
                if e in node.get_out_links():
                    e.end_node.remove_link(e)
                else:
                    e.start_node.remove_link(e)
        for hl in node.get_half_links():
            if e in self.get_half_links():
                self.remove_half_link(hl)
        self.nodes.remove(node)


    def add_link(self,link,position=None):
        """Adds link to the list of links.
        If the start node or the end node are not yet added to the network, they
        will be added to the list of nodes.

        Parameters
        ----------
        link : Link
            Link object to be added to the network
        position : integer, optional
            Position (index) in the list of links of the network where the link should be inserted. Default is insert at end of list (append)

        Raises
        ------
        TypeError
            If link is not an instance of Link
        """
        if not isinstance(link,Link):
            raise TypeError("Only an Link object can be added")
        if position == None:
            self.links.append(link)
        else:
            self.links.insert(position,link)
        if link.start_node not in self.get_nodes():
            self.add_node(link.start_node)
        if link.end_node not in self.get_nodes():
            self.add_node(link.end_node)

    def remove_link(self,link):
        """Removes link from the list of links

        Parameters
        ----------
        link : Link
            Link object to be removed from the network

        Raises
        ------
        TypeError
            If link in not an instance of Link
        """
        if not isinstance(link,Link):
            raise TypeError("Only an Link object can be removed")
        self.links.remove(link)

    def add_half_link(self,half_link,position=None):
        """Adds half_link to the list of half links.

        If the start node is not yet added to the network, it
        will be added to the list of nodes.

        Parameters
        ----------
        half_link : HalfLink
            Half link object to be added to the network
        position : integer, optional
            Position (index) in the list of half links of the network where the half link should be inserted.

        Raises
        ------
        TypeError
            If half_link is not an instance of HalfLink
        """
        if not isinstance(half_link,HalfLink):
            raise TypeError("Only a HalfLink object can be added")
        if not half_link in self.half_links:
            if position == None:
                self.half_links.append(half_link)
            else:
                self.half_links.insert(position,half_link)
        if half_link.start_node not in self.get_nodes():
            self.add_node(half_link.start_node)

    def remove_half_link(self,half_link):
        """Removes half_links from the list of half links of the network (but NOT from the list of half links of the start node)

        Parameters
        ----------
        half_link : HalfLink
            Half link object to be removed from the network

        Raises
        ------
        TypeError
            If half_link is not an instance of HalfLink
        """
        if not isinstance(half_link,HalfLink):
            raise TypeError("Only a HalfLink object can be removed")
        if half_link in self.half_links:
            self.half_links.remove(half_link)

    def get_nodes(self,*args):
        """Iterates over all the nodes in the list of nodes

        Yields
        ------
        n : Node
            The next Node instance in self.nodes
        """
        for n in self.nodes:
            if n is not None:
                yield n

    def get_links(self,*args,link_types=list(),bc_types=list(),**kwargs):
        """Iterates over all the links in the list of links.
        This only includes links, not half links
        
        Parameters
        ----------
        link_types : list, optional
            List of link types of the links to be yielded. If empty, all the links are yielded. Default is an empty list.
        bc_types : list, optional
            List of bc types of the links to be yielded. If empty, all the links are yielded. Default is an empty list.

        Yields
        ------
        e : Link
            The next Link instance in self.links
        """
        for e in self.links:
            if e is not None:
                if link_types and bc_types:
                    if e.link_type in link_types and e.bc_type in bc_types:
                        yield e
                elif link_types:
                    if e.link_type in link_types:
                        yield e
                elif bc_types:
                    if e.bc_type in bc_types:
                        yield e
                else:
                    yield e

    def get_half_links(self,*args):
        """Iterates over all the half links in the list of half links

        Yields
        ------
        hl : HalfLink
            The next HalfLink instance in self.half_links
        """
        for hl in self.half_links:
            if hl is not None:
                yield hl

    def add_network(self,network):
        """ Adds network to the Network (self).
        This means that the other networks' list of nodes and list of links are added to the lists of the original network. Nodes or links that occur in both are not added.
        The hierarchical level of the network that is added is adjusted to reflect it is now a subnetwork of self.

        Parameters
        ----------
        network : Network
            The network to be added

        Raises
        ------
        TypeError
            If network is not a Network instance
        """
        if not isinstance(network,Network):
            raise TypeError("Only a Network object can be added")
        network.level.insert(0,self.name)
        for net in network.get_networks(get_all=True):
            net.level.insert(0,self.name)
        self.nodes = merge_sorted(self.nodes,network.nodes)
        self.links = merge_sorted(self.links,network.links)
        self.half_links = merge_sorted(self.half_links,network.half_links)
        self.networks.append(network)

    def get_networks(self,get_all=False):
        """Iterates over all the networks in the list of networks

        Parameters
        ----------
        get_all : boolean, optional
            Specificy if the subsubnetworks etc. should also be returned. If False, only the subnetworks are returned. If True, all the subnetworks of all the subnetwork etc. are returned. Default is False

        Yields
        ------
        net : Network
            The next Network instance in self.networks
        """
        for net in self.networks:
            if net is not None:
                if get_all:
                    if net.networks:
                        for subnet in net.get_networks(get_all=True):
                            yield subnet
                        yield net
                    else:
                        yield net
                else:
                    yield net

    #@abc.abstractmethod
    def initialize(self):
        """Initalizes the network.

        To be specified for every subclass of Network
        """
        pass

    #@abc.abstractmethod
    def update(self,x):
        """Update the network for given variable vector x

        To be specified for every subclass of Network
        """
        pass

    #@abc.abstractmethod
    def update_full(self,x):
        """Updates the full network given variable vector x.
        Unlike update(x), not only the values from x are updated, but also all
        parameters not included in x.

        Returns the full set of network state parameters.
        """
        pass

    #@abc.abstractmethod
    def solve_network(self,tol,max_iter):
        """Gives the solution throughout the entire network of steady-state load flow analysis.

        To be specified for every subclass of Network.

        Parameters
        ----------
        tol : float
            tolerance for solver
        max_iter : int
            maximum number of iterations of solver
        """
        pass

    def draw_network(self,ax,halflink_length=0.3,halflink_angle=1,node_radius=0.02):
        """Draw the network, based on the x and y coordinates of the nodes.
        Note: the network is only drawn, not plotted. That is, the axes handle is updated.

        Parameters
        ----------
        ax : matplotlib axes handle
        halfllink_length : float, optional
            Lenght of the half links. Default is 0.3.
        halflink_angle : float, optional
            Determines by how much the half link is rotated: :math:`\\cos (\\frac{pi}{5} (\\text{halflink_angle}+\\text{halflink number}))`. Default is 1.
        node_radius : float, optional
            Radius of the node. Default is 0.02.
        """
        arc = None
        # plot nodes
        for n in self.get_nodes():
            c = patch.Circle((n.x,n.y),color=n.color,radius=node_radius,alpha=0.5)
            ax.add_patch(c)
            # add label to node
            ax.annotate(n.name,xy=(n.x,n.y),ha='center',va='bottom')
            for ind_hl,hl in enumerate(n.get_half_links()):
            # make straight arrows for the halflinks
                if ('Gas' in type(hl).__name__ and hl.q != 0) or ('Elec' in type(hl).__name__ and hl.P != 0) or ('Heat' in type(hl).__name__ and hl.m != 0):
                    x1,y1 = (n.x,n.y)
                    x2,y2 = (x1+halflink_length*math.cos(math.pi/5*(halflink_angle+ind_hl)),y1-halflink_length*math.sin(math.pi/5*(halflink_angle+ind_hl)))
                    alpha = 0.5
                    hl_color = hl.color
                    if hl_color == 'g':
                        hl_ls = '-'
                    elif hl_color == 'r':
                        hl_ls = '--'
                    elif hl_color == 'b':
                        hl_ls = '-.'
                    else:
                        hl_ls = '-'
                    arrow = patch.FancyArrowPatch((x1,y1),(x2,y2),
                                        arrowstyle='-|>',
                                        mutation_scale=10.0,
                                        lw=2,
                                        ls=hl_ls,
                                        alpha=alpha,
                                        color=hl_color)
                    ax.add_patch(arrow)
        # plot links
        seen={}
        for e in self.get_links():
            # make curved arrow for the links
            rad=0.1
            x1,y1 = (e.start_node.x,e.start_node.y)
            x2,y2 = (e.end_node.x,e.end_node.y)
            if (e.start_node,e.end_node) in seen:
                rad=seen.get((e.start_node,e.end_node))
                rad=(rad+np.sign(rad)*0.1)*-1
            alpha=0.5

            e_color = e.color
            if e_color == 'g':
                e_ls = '-'
            elif e_color == 'r':
                e_ls = '--'
            elif e_color == 'b':
                e_ls = '-.'
            else:
                e_ls = '-'
            arc = patch.FancyArrowPatch((x1,y1),(x2,y2),
                                arrowstyle='-|>',
                                connectionstyle='arc3,rad=%s'%float(rad*e.curve),
                                mutation_scale=10.0,
                                lw=2,
                                ls=e_ls,
                                alpha=alpha,
                                color=e_color)
            # add label to links, at the middle of the links
            x12,y12 = (x1+x2)/2. , (y1+y2)/2.
            dx,dy = x2-x1 , y2-y1
            cx,cy = x12+((rad*e.curve)/4.)*dy , y12-((rad*e.curve)/4.)*dx
            path = arc.get_path().vertices.tolist()
            px,py = [u[0] for u in path] , [u[1] for u in path]
            mx,my = (cx + px[1])/2. , (cy + py[1])/2.
            ax.annotate(e.name,xy=(mx,my),ha='center',va='bottom')
            seen[(e.start_node,e.end_node)]=rad
            ax.add_patch(arc)

    def draw_network_value(self,ax,halflink_length=0.3,halflink_angle=1,node_radius=0.02,plot_loss=False,scale_var=None,scale_var_params=None):
        """Draw the network, based on the x and y coordinates of the nodes. Show the values at the nodes and links.
        Note: the network is only drawn, not plotted. That is, the axes handle is updated.

        Parameters
        ----------
        ax : matplotlib axes handle
        halfllink_length : float, optional
            Lenght of the half links. Default is 0.3.
        halflink_angle : float, optional
            Determines by how much the half link is rotated: :math:`\\cos (\\frac{pi}{5} (\\text{halflink_angle}+\\text{halflink number}))`. Default is 1.
        node_radius : float, optional
            Radius of the node. Default is 0.02.
        plot_loss : bool, optional
            If True, the losses on the line are plotted instead of the flow values.
        """
        arc = None
        # plot nodes
        for n in self.get_nodes():
            c = patch.Circle((n.x,n.y),color=n.color,radius=node_radius,alpha=0.5)
            ax.add_patch(c)
            # add label to node
            if 'Gas' in type(n).__name__:
                ax.annotate('p:{:.2e}Pa'.format(n.get_p(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(n.x,n.y),ha='center',va='bottom')
            elif 'Elec'in type(n).__name__:
                ax.annotate(r'$|V|$:{:.2e}V'.format(n.get_V(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(n.x,n.y),ha='center',va='bottom')
                ax.annotate(r'$\delta$:{:.2e}rad'.format(n.get_delta(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(n.x,n.y),ha='center',va='top')
            elif 'Heat'in type(n).__name__:
                ax.annotate('p:{:.2e}Pa'.format(n.get_p(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(n.x,n.y),ha='left',va='bottom')
                ax.annotate(r'$T^s$:{:.2e}C'.format(n.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(n.x,n.y),ha='right',va='bottom')
                ax.annotate(r'$T^r$:{:.2e}C'.format(n.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(n.x,n.y),ha='right',va='top')
            for ind_hl,hl in enumerate(n.get_half_links()):
            # make straight arrows for the halflinks
                if ('Gas' in type(hl).__name__ and hl.q != 0) or ('Elec' in type(hl).__name__ and hl.P != 0) or ('Heat' in type(hl).__name__ and hl.m != 0):
                    x1,y1 = (n.x,n.y)
                    x2,y2 = (x1+halflink_length*math.cos(math.pi/5*(halflink_angle+ind_hl)),y1-halflink_length*math.sin(math.pi/5*(halflink_angle+ind_hl)))
                    alpha = 0.5
                    hl_color = hl.color
                    if hl_color == 'g':
                        hl_ls = '-'
                    elif hl_color == 'r':
                        hl_ls = '--'
                    elif hl_color == 'b':
                        hl_ls = '-.'
                    else:
                        hl_ls = '-'
                    arrow = patch.FancyArrowPatch((x1,y1),(x2,y2),
                                        arrowstyle='-|>',
                                        mutation_scale=10.0,
                                        lw=2,
                                        ls=hl_ls,
                                        alpha=alpha,
                                        color=hl_color)
                    ax.add_patch(arrow)
                    # add label to links, at the middle of the links 
                    x12,y12 = (x1+x2)/2. , (y1+y2)/2.
                    if ('Gas' in type(hl).__name__ and hl.q != 0):
                        ax.annotate('q:{:.2e}kg/s'.format(hl.get_q(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(x12,y12),ha='center',va='bottom')
                    elif ('Elec' in type(hl).__name__ and hl.P != 0):
                        ax.annotate(r'P:{:.2e}W'.format(hl.get_P(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(x12,y12),ha='center',va='bottom')
                        ax.annotate(r'Q:{:.2e}VA'.format(hl.get_Q(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(x12,y12),ha='center',va='top')
                    elif ('Heat' in type(hl).__name__ and hl.m != 0):
                        ax.annotate('m:{:.2e}kg/s'.format(hl.get_m(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(x12,y12),ha='center',va='bottom')
                        ax.annotate(r'$\varphi$:{:.2e}W'.format(hl.get_dphi(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(x12,y12),ha='center',va='top')
                        # To at end of half link
                        ax.annotate(r'$T^o_s$:{:.2e}C'.format(hl.get_Ts(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(x2,y2),ha='center',va='bottom')
                        ax.annotate(r'$T^o_r$:{:.2e}C'.format(hl.get_Tr(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(x2,y2),ha='center',va='top')
                        
        # plot links
        seen={}
        for e in self.get_links():
            # make curved arrow for the links
            rad=0.1
            x1,y1 = (e.start_node.x,e.start_node.y)
            x2,y2 = (e.end_node.x,e.end_node.y)
            if (e.start_node,e.end_node) in seen:
                rad=seen.get((e.start_node,e.end_node))
                rad=(rad+np.sign(rad)*0.1)*-1
            alpha=0.5

            e_color = e.color
            if e_color == 'g':
                e_ls = '-'
            elif e_color == 'r':
                e_ls = '--'
            elif e_color == 'b':
                e_ls = '-.'
            else:
                e_ls = '-'
            arc = patch.FancyArrowPatch((x1,y1),(x2,y2),
                                arrowstyle='-|>',
                                connectionstyle='arc3,rad=%s'%float(rad*e.curve),
                                mutation_scale=10.0,
                                lw=2,
                                ls=e_ls,
                                alpha=alpha,
                                color=e_color)
            # add label to links, at the middle of the links
            x12,y12 = (x1+x2)/2. , (y1+y2)/2.
            dx,dy = x2-x1 , y2-y1
            cx,cy = x12+((rad*e.curve)/4.)*dy , y12-((rad*e.curve)/4.)*dx
            path = arc.get_path().vertices.tolist()
            px,py = [u[0] for u in path] , [u[1] for u in path]
            mx,my = (cx + px[1])/2. , (cy + py[1])/2.
            if 'Gas' in type(e).__name__:
                ax.annotate('q:{:.2e}kg/s'.format(e.get_q(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(mx,my),ha='center',va='bottom')
            elif 'Elec'in type(e).__name__:
                if plot_loss:
                    ax.annotate(r'$Ploss$:{:.2e}W'.format(e.get_Pstart(scale_var=scale_var,scale_var_params=scale_var_params)+e.get_Pend(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(mx,my),ha='center',va='bottom')
                    ax.annotate(r'$Qloss$:{:.2e}VA'.format(e.get_Qstart(scale_var=scale_var,scale_var_params=scale_var_params)+e.get_Qend(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(mx,my),ha='center',va='top')
                else:
                    ax.annotate('Pstart:{:.2e}W'.format(e.get_Pstart(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(mx,my),ha='center',va='bottom')
                    ax.annotate('Qstart:{:.2e}VA'.format(e.get_Qstart(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(mx,my),ha='center',va='top')
            elif 'Heat'in type(e).__name__:
                if plot_loss:
                    ax.annotate(r'$\varphi loss^s$:{:.2e}W'.format(e.heat_loss_supply(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(mx,my),ha='center',va='bottom')
                else:
                    ax.annotate('m:{:.2e}kg/s'.format(e.get_m(scale_var=scale_var,scale_var_params=scale_var_params)),xy=(mx,my),ha='center',va='bottom')
            seen[(e.start_node,e.end_node)]=rad
            ax.add_patch(arc)

# ===========================================================================
class Node(metaclass=abc.ABCMeta):
    """Overall node class.

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
    color : str
        Color used for plotting.
    """

    def __init__(self,name,x=0,y=0):
        """Creates a Node object

        Parameters
        ----------
        name : str
            Name of the node
        x : float, optional
            x coordinate of the node. Default is 0.
        y : float, optional
            y coordinate of the node. Default is 0.
        """
        super().__init__()
        self.name = name
        self.out_links = list();
        self.in_links = list();
        self.half_links = list();
        self.number = None
        self.color = 'k'
        self.x = x
        self.y = y

    def get_out_links(self,*args):
        """Iterates over all the outgoing links and half links connected to the node.

        Yields
        ------
        e : Link or HalfLink
            The next Link or HalfLink instance in self.out_links
        """
        for e in self.out_links:
            if e is not None:
                yield e

    def get_in_links(self,*args):
        """Iterates over all the incoming links and half links connected to the node.

        Yields
        ------
        e : Link or HalfLink
            The next Link or HalfLink instance in self.in_links
        """
        for e in self.in_links:
            if e is not None:
                yield e

    def get_half_links(self,*args,link_types=list(),bc_types=list(),**kwargs):
        """Iterates over all the half links connected to the node.

        Parameters
        ----------
        link_types : list, optional
            List of link types of the links to be yielded. If empty, all the links are yielded. Default is an empty list.
        bc_types : list, optional
            List of bc types of the links to be yielded. If empty, all the links are yielded. Default is an empty list.
            
        Yields
        ------
        hl : HalfLink
            The next HalfLink instance in self.half_links
        """
        for hl in self.half_links:
            if hl is not None:
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

    def get_links(self,*args):
        """Iterates over all the links connected to the node; incoming, outgoing and half links.

        Yields
        ------
        e : Link or HalfLink
            The next Link or HalfLink instance in self.out_links + self.in_links
        """
        for e in self.out_links + self.in_links: # every half link is included in the out links or the in links.
            if e is not None:
                yield e

    def remove_link(self,link):
        """Removes link from the list of out links or the list of in links.

        Parameters
        ----------
        link : Link
            Link to be removed from the node.

        Raises
        ------
        TypeError
            If link is not an instance of Link
        """
        if not isinstance(link,Link):
            raise TypeError("Only an Link object can be removed")
        if link in self.out_links:
            self.out_links.remove(link)
        elif link in self.in_links: # link is either incoming or outgoing
            self.in_links.remove(link)
        
    def remove_half_link(self,half_link):
        """Removes half_link from the list of half links and the list of out links.

        Parameters
        ----------
        half_link : HalfLink
            Half link to be removed from the node.

        Raises
        ------
        TypeError
            If half_link is not an instance of HalfLink
        """
        if not isinstance(half_link,HalfLink):
            raise TypeError("Only a HalfLink object can be removed")
        self.half_links.remove(half_link)
        self.out_links.remove(half_link)

    #@abc.abstractmethod
    def node_law(self,*args):
        """Node law for a general Node.
        """
        pass

# ===========================================================================
class Link(metaclass=abc.ABCMeta):
    """Overall link class.

    Attributes
    ----------
    name : str
        The name of the link
    start_node : Node
        start node of the link
    end_node : Node
        end node of the link
    number : int
        number of the link
    color : str
        Color used for plotting.
    curve : float
        Value to indicate the curvature used for plotting.
    """
    def __init__(self,name,start_node,end_node):
        """Creates a Link object

        Parameters
        ----------
        name : str
            Name of the link.
        start_node : Node
            Start node of the link.
        end_node : Node
            End node of the link.

        Raises
        ------
        TypeError
            If start_node or end_node is not an instance of Node.
        """
        if not isinstance(start_node,Node):
            raise TypeError("Only a Node object can be a start node")
        if not isinstance(end_node,Node):
            raise TypeError("Only a Node object can be an end node")
        super().__init__()
        self.name = name
        self.start_node = start_node
        start_node.out_links.append(self)
        self.end_node= end_node
        end_node.in_links.append(self)
        self.number = None
        self.color = 'k'
        self.curve = 0.

# ===========================================================================
class HalfLink(metaclass=abc.ABCMeta):
    """Overall half link class.
    The default is an outflow half link.

    Attributes
    ----------
    name : str
        Name of the half link
    start_node : Node
        Start node of the half link
    number : int
        Number of the half link
    """
    def __init__(self,name,start_node):
        """Creates a HalfLink object

        Parameters
        ----------
        name : str
            Name of the half link.
        start_node : Node
            Start node of the half link.

        Raises
        ------
        TypeError
            If start_node is not an instance of Node
        """
        if not isinstance(start_node,Node):
            raise TypeError("Only a Node object can be a start node")
        super().__init__()
        self.name = name
        self.start_node = start_node
        start_node.out_links.append(self)
        start_node.half_links.append(self)
        self.number = None
        self.color = 'k'
