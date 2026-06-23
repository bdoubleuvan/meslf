"""Example of a gas network with 3 nodes connected in a single line"""
from meslf.networks.gas_network import GasNetwork, GasNode, GasLink
from meslf.networks.carrier import Gas
from meslf.utils.constants import bar, km, cm
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import warnings
from meslf.load_flow.system_of_equations import NonLinearSystemGas

# carrier
S = 0.589
pn = 0.1e6 #[Pa]
Tn = 288. #[K]
T = Tn
Z = 1.
R = 8.314459848 #[J/molK]
M = 28.97e-3 #[kg/mol]
R_air = R/M #[J/kgK]
gas = Gas('gas',S,R_air,Z,pn,Tn,T)


# solver information
tol = 1e-6
max_iter = 150
    
def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning 

def create_network(p1=9*bar,q3=1,hydr_eq='fb',L12=4*km,L23=5*km,D=10*cm,E=.98,link_type='pipe_high_pres_weymouth'):
    """Create a gas network consisting of 3 nodes in one line.
    
    Parameters
    ----------
        
    Returns
    -------
    gas_net : GasNetwork
        The test network
    """
    gas_net = GasNetwork('3N one line')
    gn1 = GasNode('gn1',node_type=0,x=0,y=0,p=p1) # ref
    gn2 = GasNode('gn2',node_type=1,x=1,y=0,q=0) # load node (junction)
    gn3 = GasNode('gn3',node_type=1,x=2,y=0,q=q3) # load node
    
    link_params12 = {'L':L12,'D':D,'E':E,'carrier':gas}
    link_params23 = link_params12.copy()
    link_params23['L'] = L23
    
    if hydr_eq=='fb':
        gl0 = GasLink('gl0',gn1,gn2,link_type=link_type,link_params=link_params12,link_eq_form='dp_of_q')
        gl1 = GasLink('gl1',gn2,gn3,link_type=link_type,link_params=link_params23,link_eq_form='dp_of_q')
    elif hydr_eq=='fa':
        gl0 = GasLink('gl0',gn1,gn2,link_type=link_type,link_params=link_params12)
        gl1 = GasLink('gl1',gn2,gn3,link_type=link_type,link_params=link_params23)
    else:
        raise ValueError("Enter valid value for hydr_eq. Either 'fa' or 'fb'.")

    gas_net.add_link(gl0)
    gas_net.add_link(gl1)
    
    return gas_net

def initialize_network(gas_net,q0=1,q1=1,p2=8*bar,p3=7*bar,formulation='full'):
    """Create a gas network consisting of 3 nodes in one line.
    
    Parameters
    ----------
    gas_net : GasNetwork
        The gas network to be initialized
    formulation : str, optional
        Formulation of the non-linear system of equations used to solve the network.
    node_set : int, optional
        Node set to use. Node set 1 corresponds to ... and node set 2 corresponds to ... . Default is node set 1
        
    Returns
    -------
    x0 : np array
        initial guess
    """
    if formulation == 'nodal':
        for link in gas_net.get_links():
            if link.link_eq_form == 'dp_of_q':
                warnings.warn("The link {} uses press. drop as function of link flow (fb instead of fa), but formulation is 'nodal'. The link equation for this link is changed to fa!!".format(link.name))
                link.set_type(link.link_type,link.link_params,link_eq_form='q_of_dp')
    q_init = np.array([q0,q1]) #[kg/s]
    p_init = np.array([p2,p3]) #[Pa]
    if formulation == 'nodal':
        x_init = p_init
    else:
        x_init = np.concatenate((q_init,p_init))
    gas_net.initialize()
    gas_net.update(x_init,formulation=formulation)
    x0 = gas_net.set_x_init(formulation=formulation)
    return x0

def run_load_flow(pgbase,qbase,p1=9*bar,q3=1,q0=1,q1=1,p2=8*bar,p3=7*bar,L12=4*km,L23=5*km,D=10*cm,E=.98,link_type='pipe_high_pres_weymouth',formulation='full',hydr_eq='fb',tol=1e-6,max_iter=150):
    """Stead-state load flow analysis of gas network

    Parameters
    ----------
    """
    # create network
    gas_net = create_network(p1=p1,q3=q3,hydr_eq=hydr_eq,L12=L12,L23=L23,D=D,E=E,link_type=link_type)
    # initialize
    x0 = initialize_network(gas_net,q0=q0,q1=q1,p2=p2,p3=p3,formulation=formulation)
    
    # solve network
    scale_var = 'matrix'
    scale_var_params = {'qbase':qbase,'pgbase':pgbase}
    x_sol,iters,err_vec,p_sol,q_sol,q_inj = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    
    return gas_net,x_sol,iters,err_vec,p_sol,q_sol,q_inj

def analytical_solution(gas_net):
    """Determine the solution analytically (But only for a high pressure network!!!)
    
    Returns
    -------
    p_sol : np array
        Solution of pressures
    q_sol : np array
        Solution of flows (which is trivial)
    """
    # boundary conditions:
    q = gas_net.nodes[-1].half_links[0].q
    #print('q = {}'.format(q))
    p1 = gas_net.nodes[0].p
    #print('p1 = {}'.format(p1))
    #print('p1^2 = {}'.format(p1**2))
    
    # pressure solution
    #print('dp2 = {}'.format(gas_net.links[0].dp_of_q(q)))
    #print('p1^2 - dp2 = {}'.format(p1**2 - gas_net.links[0].dp_of_q(q)))
    p2 = np.sign(p1**2 - gas_net.links[0].dp_of_q(q))*np.sqrt(np.abs(p1**2 - gas_net.links[0].dp_of_q(q)))
    #print('dp2 = {}'.format(gas_net.links[1].dp_of_q(q)))
    #print('p2^2 - dp2 = {}'.format(p2**2 - gas_net.links[1].dp_of_q(q)))
    p3 = np.sign(p2**2 - gas_net.links[1].dp_of_q(q))*np.sqrt(np.abs(p2**2 - gas_net.links[1].dp_of_q(q)))
    p_sol = np.array([p2,p3])
    #print('p1^2-p2^2-dp2 = {}'.format(p1**2 - p2**2 - gas_net.links[0].dp_of_q(q)))
    q_sol = q*np.ones(len(gas_net.links))
    return p_sol,q_sol

def compare_convergence_boundary_conditions():
    """Compare the convergence behaviour for different boundaryl conditions, and different formulations
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # boundary conditions
    p_BC = np.array([30*bar]) #[Pa]
    q_BC = np.array([2.,1.]) #[kg/s]
    
    # solver information
    formulation='full'
    hydr_eq='fb'

    # make figure to plot convergence
    fig_conv_gas_BC, ax_conv_gas_BC = plt.subplots(3, 1, sharex=True, num='Convergence plot gas network with different BCs')
    ax_fb_full = ax_conv_gas_BC[0] 
    ax_fa_full = ax_conv_gas_BC[1]
    ax_fa_nodal = ax_conv_gas_BC[2]
    ax_fa_nodal.set_xlabel(r'Iteration $k$')
    ax_fb_full.set_title(r'full and fb')
    ax_fa_full.set_title(r'full and fa')
    ax_fa_nodal.set_title(r'nodal and fa')
    max_iters_used = 0

    # run load flow
    for p1 in p_BC:
        for q3 in q_BC:
            # initial guess
            p2 = .99*p1#[Pa]
            p3 = .98*p1 #[Pa]
            q12 = q3 #[kg/s]
            q23 = q3 #[kg/s]
            
            # scaling
            pgbase = bar#p1
            qbase = 1.
            for formulation in ['full','nodal']:
                if formulation == 'full':
                    hydr_eq_options = ['fa','fb']
                else:
                    hydr_eq_options = ['fa']
                for hydr_eq in hydr_eq_options:
                    print('\nFormulation is {} and hydraulic equation is {}'.format(formulation,hydr_eq))
                    # load flow
                    gas_net,x_sol,iters,err_vec,p_sol,q_sol,q_inj = run_load_flow(pgbase,qbase,p1=p1,q3=q3,q0=q12,q1=q23,p2=p2,p3=p3,formulation=formulation,hydr_eq=hydr_eq)
                
                    # plot convergence
                    if formulation == 'full':
                        if hydr_eq == 'fb':
                            ax = ax_fb_full
                        else:
                            ax = ax_fa_full
                    else:
                        ax = ax_fa_nodal
                    ax.semilogy(err_vec,marker='o',ls='-',label=r'$p_1$={:.3f} bar, $q_3$={:.3f} kg/s'.format(p1/bar,q3))
                    max_iters_used = max(max_iters_used,iters)
                    
                    # (compare with) analytical solution
                    p_sol_an, q_sol_an = analytical_solution(gas_net)
                    print('Boundary conditions: p1 = {:.3f} bar, q = {:.3f} kg/s'.format(p1/bar,q3))
                    print('Analytical solution: p2 = {:.3f} bar, p3 = {:.3f} bar, q = {:.3f} kg/s'.format(p_sol_an[0]/bar,p_sol_an[1]/bar,q_sol_an[0]))
                    print('Load flow solution: p2 = {:.3f} bar, p3 = {:.3f} bar, q = {:.3f} kg/s'.format(p_sol[1]/bar,p_sol[2]/bar,q_sol[0]))
                    print('Relative error |p-p_sol|/|p_sol| for p2: {:.3f}, p3: {:.3f}'.format(np.abs(p_sol[1]-p_sol_an[0])/np.abs(p_sol_an[0]),np.abs(p_sol[2]-p_sol_an[1])/np.abs(p_sol_an[1])))
                    print(x_sol)
                    scale_var = 'matrix'
                    scale_var_params = {'qbase':qbase,'pgbase':pgbase}
                    nlsys = NonLinearSystemGas(gas_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
                    x_sol_an = np.concatenate([q_sol_an,p_sol_an])
                    gas_net.update(x_sol_an,formulation=formulation)
                    print('F(x_sol_an) = {}'.format(nlsys.F(x_sol_an)))
                    D_F = nlsys.DF()
                    print('D_F F(x_sol_an) = {}'.format(D_F.dot(nlsys.F(x_sol_an))))
    for ax in ax_conv_gas_BC:
        ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
        ax.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
        box_ax = ax.get_position()
        ax.set_position([box_ax.x0, box_ax.y0, box_ax.width * 0.8, box_ax.height]) # Shrink current axis by 20%
        
        ax.grid(which='major',color='k', linestyle='--', alpha=.2)
        ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_fa_full.legend(loc='center left', bbox_to_anchor=(1, .5)) # Put a legend to the right of the middle axis (This assumes the other two plots have the same legend!!! This seems to be the case, but I'm not entirely sure)

        
if __name__ == '__main__':
    # boundary conditions
    p1 = 32*bar #[Pa]
    q3 = 10. #[kg/s]
    
    # initial guess
    p2 = 30*bar #[Pa]
    p3 = 28*bar #[Pa]
    q12 = q3 #[kg/s]
    q23 = q3 #[kg/s]

    # solver information
    formulation='full'

    # scaling
    pgbase = bar
    qbase = 1.

    # load flow
    gas_net,x_sol,iters,err_vec,p_sol,q_sol,q_inj = run_load_flow(pgbase,qbase,p1=p1,q3=q3,q0=q12,q1=q23,p2=p2,p3=p3,formulation=formulation,hydr_eq='fb',tol=tol,max_iter=max_iter)
    gas_net_fa,x_sol_fa,iters_fa,err_vec_fa,p_sol_fa,q_sol_fa,q_inj_fa = run_load_flow(pgbase,qbase,p1=p1,q3=q3,q0=q12,q1=q23,p2=p2,p3=p3,formulation=formulation,hydr_eq='fa',tol=tol,max_iter=max_iter)
    gas_net_fa_nod,x_sol_fa_nod,iters_fa_nod,err_vec_fa_nod,p_sol_fa_nod,q_sol_fa_nod,q_inj_fa_nod = run_load_flow(pgbase,qbase,p1=p1,q3=q3,q0=q12,q1=q23,p2=p2,p3=p3,formulation='nodal',hydr_eq='fa',tol=tol,max_iter=max_iter)
    
    # analytical solution
    p_sol_an, q_sol_an = analytical_solution(gas_net)
    print('\nBoundary conditions: p1 = {:.3f} bar, q = {:.3f} kg/s'.format(p1/bar,q3))
    print('Analytical solution: p2 = {:.3f} bar, p3 = {:.3e} bar, q = {:.3f} kg/s'.format(p_sol_an[0]/bar,p_sol_an[1]/bar,q_sol_an[0]))
    print('Load flow solution: p2 = {:.3f} bar, p3 = {:.3e} bar, q = {:.3f} kg/s'.format(p_sol[0]/bar,p_sol[1]/bar,q_sol[0]))
    print('Relative error |p-p_sol|/|p_sol| for p2: {:.3f}, p3: {:.3f}'.format(np.abs(p_sol[0]-p_sol_an[0])/np.abs(p_sol_an[0]),np.abs(p_sol[1]-p_sol_an[1])/np.abs(p_sol_an[1])))
    print(x_sol)
    
    # plot network solutions
    fig_sol1 = plt.figure('Network solution gas network, fb')
    ax_sol1 = fig_sol1.gca()
    gas_net.draw_network_value(ax_sol1)
    plt.axis('equal')
    plt.axis('off')
    
    # print solution
    print('\nSolution for fb')
    for node in gas_net.get_nodes():
        print('node {} with p = {:.3e} bar'.format(node.name,node.p/bar))
        for hl in node.get_half_links():
            print('with half link {} of type {}, with q = {:.3f} kg/s'.format(hl.name,hl.link_type,hl.q))
    for link in gas_net.get_links():
        print('link {} from node {} to node {}, with q = {:.3f} kg/s'.format(link.name,link.start_node.name,link.end_node.name,link.q))
    print('Error in solution: (p2-p2_an)/p2_an = {:.3e}, (p3-p3_an)/p3_an = {:.3e}, (q-q_an)/q_an = {:.3e}'.format((p_sol[0] - p_sol_an[0])/p_sol_an[0],(p_sol[1] - p_sol_an[1])/p_sol_an[1],(q_sol[0] - q_sol_an[0])/q_sol_an[0]))
    
    nlsys = NonLinearSystemGas(gas_net,formulation=formulation)
    x_sol_an = np.concatenate([q_sol_an,p_sol_an])
    gas_net.update(x_sol_an,formulation=formulation)
    for node in gas_net.get_nodes():
        print('node {} with p = {:.3e} bar'.format(node.name,node.p/bar))
        for hl in node.get_half_links():
            print('with half link {} of type {}, with q = {:.3f} kg/s'.format(hl.name,hl.link_type,hl.q))
    for link in gas_net.get_links():
        print('link {} from node {} to node {}, with q = {:.3f} kg/s'.format(link.name,link.start_node.name,link.end_node.name,link.q))
        
    print('F(x_sol_an) = {}'.format(nlsys.F(x_sol_an)))
    
    # plot convergence
    fig_conv_gas = plt.figure('Convergence plot gas network')
    ax_conv_gas = fig_conv_gas.gca()
    plt.xlabel(r'Iteration $k$')
    plt.ylabel(r'Error $||D_F F(x^k)||_2$')
    max_iter_used = np.max([iters])
    ax_conv_gas.semilogy([0,max_iter_used+1],[tol,tol],'r:',label='tolerance')
    ax_conv_gas.semilogy(err_vec,marker='o',ls='-',color='tab:blue',label='fb full')
    ax_conv_gas.semilogy(err_vec_fa,marker='o',ls='-',color='tab:orange',label='fa full')
    ax_conv_gas.semilogy(err_vec_fa_nod,marker='o',ls='-',color='tab:green',label='fa nodal')
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)

    compare_convergence_boundary_conditions()
    
    plt.show()
