"""Example of a elec network with 3 nodes connected in a single line"""
from meslf.networks.electrical_network import ElectricalNetwork, ElectricalNode, ElectricalLink
from meslf.utils.constants import cm,km,nF,kV,MW
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import warnings

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning 

def create_line_data(L,D):
    """Create the needed data for transmission lines.
    
    Parameters
    ----------
    L : float
        Length of the line in m
    D : float
        Diameter of the line in m
        
    Returns
    -------
    b : float
        b
    g : float
        g
    b_sh : float
        b_sh
    """
    c = 100 #[nF/km]
    f = 50 #[Hz]
    rho = 1.6e-8 #[Ohm m]
    r_line = 4*rho*L/(np.pi*D**2) #[Ohm]
    x_line = 10*r_line #[Ohm]
    z_abs2_line = r_line**2 + x_line**2
    b = -x_line/(z_abs2_line) 
    g = r_line/(z_abs2_line)
    b_sh = 2*np.pi*f*c*nF*L/km
    return b,g,b_sh

def create_network(V1=50*kV,delta1=0.,V2=50*kV,P2=-1*MW,P3=1.5*MW,Q3=1.5*MW,node_set=1,L12=4*km,L23=5*km,D=1*cm,link_type='pi_line'):
    """Create a electrical network consisting of 3 nodes in one line.
    
    Parameters
    ----------
    node_set : int, optional
        Node set to use. Node set 1 corresponds to ... and node set 2 corresponds to ... . Default is node set 1
        
    Returns
    -------
    elec_net : ElectricalNetwork
        The test network
    """
    elec_net = ElectricalNetwork('3N one line')
    en1 = ElectricalNode('en1',node_type=0,x=0,y=0,V=V1,delta=delta1) # reference node
    if node_set == 1:
        en2 = ElectricalNode('en2',node_type=2,x=1,y=0,P=0,Q=0) # load node (junction)
    elif node_set == 2:
        en2 = ElectricalNode('en2',node_type=1,x=1,y=0,V=V2,P=P2) # generator node
    else:
        raise ValueError('Enter valid value for node_set')
    en3 = ElectricalNode('en3',node_type=2,x=2,y=0,P=P3,Q=Q3) # load node
    
    b_12,g_12,b_sh_12 = create_line_data(L12,D)
    b_23,g_23,b_sh_23 = create_line_data(L23,D)
    if link_type == 'pi_line':
        link_params12 = {'b':b_12,'g':g_12,'b_sh':b_sh_12,'g_sh':0}
        link_params23 = {'b':b_23,'g':g_23,'b_sh':b_sh_23,'g_sh':0}
    elif link_type == 'short_line':
        link_params12 = {'b':b_12,'g':g_12,'b_sh':0,'g_sh':0}
        link_params23 = {'b':b_23,'g':g_23,'b_sh':0,'g_sh':0}
    else:
        raise ValueError('Link type should be pi_line or short_line')
    
    el0 = ElectricalLink('el0',en1,en2,link_type=link_type,link_params=link_params12)
    el1 = ElectricalLink('el1',en2,en3,link_type=link_type,link_params=link_params23)

    elec_net.add_link(el0)
    elec_net.add_link(el1)
    
    return elec_net

def initialize_network(elec_net,V2_init=50*kV,V3_init=50*kV,delta2_init=0.,delta3_init=0.,node_set=1):
    """Create a elec network consisting of 3 nodes in one line.
    
    Parameters
    ----------
    elec_net : ElectricalNetwork
        The electrical network to be initialized
    node_set : int, optional
        Node set to use. Node set 1 corresponds to ... and node set 2 corresponds to ... . Default is node set 1
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    delta_init = np.array([delta2_init,delta3_init]) #[rad]
    if node_set == 1:
        V_init = np.array([V2_init,V3_init]) #[V]
    elif node_set == 2:
        V_init = np.array([V3_init]) #[V]
    x_init = np.concatenate((delta_init,V_init))
        
    elec_net.initialize()
    elec_net.update(x_init)
    x0 = elec_net.set_x_init()
    return x0

def run_load_flow(Vbase,Sbase,deltabase=1,V1=50*kV,delta1=0.,V2=50*kV,P2=-1*MW,P3=1.5*MW,Q3=1.5*MW,V2_init=50*kV,V3_init=50*kV,delta2_init=0.,delta3_init=0.,L12=4*km,L23=5*km,D=10*cm,link_type='pi_line',node_set=1,tol = 1e-6,max_iter = 150):
    """Stead-state load flow analysis of electrical network

    Parameters
    ----------

    node_set : int, optional
        Node set to use. Node set 1 corresponds to ... and node set 2 corresponds to ... . Default is node set 1

    """
    # create network
    elec_net = create_network(V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,L12=L12,L23=L23,D=D,link_type=link_type,node_set=node_set)
    # initialize
    x0 = initialize_network(elec_net,V2_init=V2_init,V3_init=V3_init,delta2_init=delta2_init,delta3_init=delta3_init,node_set=node_set)
    
    # solve network
    x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge = elec_net.solve_network(tol,max_iter,solver='NR',scale_var='matrix',scale_var_params={'deltabase':deltabase,'Vbase':Vbase,'Sbase':Sbase})
    
    return elec_net,x_sol,iters,err_vec,delta_sol,V_sol,S_inj,P_edge,Q_edge

if __name__ == '__main__':
    # boundary conditions
    V1 = 50*kV #[V]
    delta1 = 0. #[rad]
    V2 = 50*kV #[V]
    P2 = -1*MW #[W]
    P3 = 1.5*-P2 #[W] 
    Q3 = P3 #[Var]

    # initial guess
    V2_init = 50*kV #[V]
    V3 = 50*kV #[V]
    delta2 = 0. #[rad]
    delta3 = 0. #[rad]
    
    # solver information
    tol = 1e-6
    max_iter = 150

    # scaling
    deltabase = 1.
    Vbase = 50*kV
    Sbase = MW

    elec_net1,x_sol1,iters1,err_vec1,delta_sol1,V_sol1,S_inj1,P_edge1,Q_edge1 = run_load_flow(Vbase,Sbase,deltabase=deltabase,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,V2_init=V2_init,V3_init=V3,delta2_init=delta2,delta3_init=delta3,node_set=1,tol=tol,max_iter=max_iter)
    elec_net2,x_sol2,iters2,err_vec2,delta_sol2,V_sol2,S_inj2,P_edge2,Q_edge2 = run_load_flow(Vbase,Sbase,deltabase=deltabase,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,V2_init=V2_init,V3_init=V3,delta2_init=delta2,delta3_init=delta3,node_set=2,tol=tol,max_iter=max_iter)
        
    # plot network solutions
    fig_sol1 = plt.figure('Network solution elec network, node set 1')
    ax_sol1 = fig_sol1.gca()
    elec_net1.draw_network_value(ax_sol1,plot_loss=True)
    plt.axis('equal')
    plt.axis('off')
    
    fig_sol2 = plt.figure('Network solution elec network, node set 2')
    ax_sol2 = fig_sol2.gca()
    elec_net2.draw_network_value(ax_sol2,plot_loss=True)
    plt.axis('equal')
    plt.axis('off')
    
    # print solution
    print('\nSolution for node set 1')
    for node in elec_net1.get_nodes():
        print('node {} with V = {:.3e} V, delta = {:.3f} rad'.format(node.name,node.V,node.delta))
        for hl in node.get_half_links():
            print('with half link {}, with P = {:.3e} W, Q = {:.3e} W'.format(hl.name,hl.P,hl.Q))
    for link in elec_net1.get_links():
        print('link {} from node {} to node {}, with S_loss = {:.3e} W'.format(link.name,link.start_node.name,link.end_node.name,link.complex_power_loss()))
    
    print('\nSolution for node set 2')
    for node in elec_net2.get_nodes():
        print('node {} with V = {:.3e} V, delta = {:.3f} rad'.format(node.name,node.V,node.delta))
        for hl in node.get_half_links():
            print('with half link {}, with P = {:.3e} W, Q = {:.3e} W'.format(hl.name,hl.P,hl.Q))
    for link in elec_net2.get_links():
        print('link {} from node {} to node {}, with S_loss = {:.3e} W'.format(link.name,link.start_node.name,link.end_node.name,link.complex_power_loss()))
        
    # plot convergence
    fig_conv_elec = plt.figure('Convergence plot elec network')
    ax_conv_elec = fig_conv_elec.gca()
    plt.xlabel(r'Iteration $k$')
    plt.ylabel(r'Error $||D_F F(x^k)||_2$')
    max_iter_used = np.max([iters1,iters2])
    ax_conv_elec.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax_conv_elec.semilogy(err_vec1,marker='o',ls='-',color='tab:blue',label='node set 1')
    ax_conv_elec.semilogy(err_vec2,marker='o',ls='-',color='tab:orange',label='node set 2')
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)

    plt.show()
