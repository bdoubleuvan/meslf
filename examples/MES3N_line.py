"""Example of a heterogeneous network where every single carrier has 3 nodes connected in a single line"""
from meslf.networks.gas_network import GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNode, HeatLink, HeatHalfLink
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
import examples.G3N_line as GasNet
import examples.E3N_line as ElecNet
import examples.H3N_line as HeatNet
from meslf.utils.constants import mbar, bar, kW, MW, kV, MBTU, BTU, MSCM, hour, cm, km
import numpy as np
import scipy.sparse as sps
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
import os

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

# physical parameters for coupling
rhon_g = GasNet.gas.rhon #[kg/m^3]
GHV = 40.611 #[MBTU/m^3]
GHV *= MBTU*BTU #[Wh/m^3]
GHV *= hour/rhon_g #[J/kg]
eta_GB = 0.8
eta_GG = 0.7
nu = 0.5
Pbs_GG_MW = 1 # The values in Shabanpour are specified for P given in 1MW
Pbs_GG = Pbs_GG_MW*MW
qbs = 1*MSCM*rhon_g/hour
GHVbs = MBTU*BTU*hour/(rhon_g)
Ebs = qbs*GHVbs
a_GG_pu = .01#.9#0.01 # Assumes P to be in 1 MW, and q to be in MSCM
b_GG_pu = 4.#3#4.
#c_GG_pu = 150.
#d_GG_pu = 15.
e_GG_pu = 0.5
Pmin_pu = 0.
a_GG = a_GG_pu * Ebs/Pbs_GG**2
b_GG = b_GG_pu * Ebs/Pbs_GG
#c_GG = c_GG_pu * Ebs
#d_GG = d_GG_pu * Ebs
e_GG = e_GG_pu * 1/Pbs_GG
Pmin = Pmin_pu * Pbs_GG


# linestyles / markers / colors used for the convergence plots
markers_sc_nodes = {'elec 1 heat 1':'x','elec 1 heat 2':'*','elec 2 heat 1':'d','elec 2 heat 2':'.'}
linestyles_sc = {'gas':'-','elec':'--','heat':'-.'}
colors_sc = {'gas':'tab:green','elec':'tab:red','heat':'tab:blue'}
linestyles_node_sets = {1:'-',2:'--',3:'-.'}
colors = {'CHP':'tab:blue','EH':'tab:orange','GB+GG':'tab:green','GB+GG_vp':'tab:red'}
linestyles_gas = {'full fa':'--','full fb':'-','nodal fa':'-.'}
markers_gas = {'full fa':'.','full fb':'*','nodal fa':'d'}
linestyles_heat = {1:'--',2:'-'}
markers_heat = {'standard outflow':'.','standard delta':'*','half_link_flow outflow':'d','half_link_flow delta':'x'}
labels_heat = {'standard outflow':r'standard, $T^o$','standard delta':r'standard, $\Delta T$','half_link_flow outflow':r'terminal link, $T^o$','half_link_flow delta':r'terminal link, $\Delta T$'}
marker_size = 10
legend_handles_node_sets = [Line2D([0], [0], marker=markers_sc_nodes.get('elec 1 heat 1'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'elec. node set 1, heat node set 1'),
                    Line2D([0], [0], marker=markers_sc_nodes.get('elec 1 heat 2'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'elec. node set 1, heat node set 2'),
                    Line2D([0], [0], marker=markers_sc_nodes.get('elec 2 heat 1'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'elec. node set 2, heat node set 1'),
                    Line2D([0], [0], marker=markers_sc_nodes.get('elec 2 heat 2'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'elec. node set 2, heat node set 2'),
                    Line2D([0], [0], color=colors.get('CHP'), label='CHP'),
                    Line2D([0], [0], color=colors.get('EH'), label='EH'),
                    Line2D([0], [0], color=colors.get('GB+GG'), label='GB+GG'),
                    Line2D([0], [0], color=colors.get('GB+GG_vp'), label='GB+GG_vp'),
                    Line2D([0], [0], color='k',ls=linestyles_node_sets.get(1), label='node set 1'),
                    Line2D([0], [0], color='k',ls=linestyles_node_sets.get(2), label='node set 2'),
                    Line2D([0], [0], color='k',ls=linestyles_node_sets.get(3), label='node set 3'),
                    Line2D([0], [0], color='k',ls=':', label='tolerance')]

legend_handles_models = [Line2D([0], [0], color='k',ls=linestyles_gas .get('full fa'), label=r'in gas: full, $f^a$'),
                    Line2D([0], [0], color='k',ls=linestyles_gas .get('full fb'), label=r'in gas: full, $f^b$'),
                    Line2D([0], [0], color='k',ls=linestyles_gas .get('nodal fa'), label=r'in gas: nodal, $f^a$'),
                    Line2D([0], [0], marker=markers_heat.get('standard outflow'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'in heat: standard, $T^o$'),
                    Line2D([0], [0], marker=markers_heat.get('standard delta'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'in heat: standard, $\Delta T$'),
                    Line2D([0], [0], marker=markers_heat.get('half_link_flow outflow'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'in heat: terminal link,  $T^o$'),
                    Line2D([0], [0], marker=markers_heat.get('half_link_flow delta'),color='w',markerfacecolor='k',markeredgecolor='k', markersize=marker_size, label=r'in heat: terminal link,  $\Delta T$'),
                    #Line2D([0], [0], color=colors.get('CHP'), label='CHP'),
                    #Line2D([0], [0], color=colors.get('EH'), label='EH'),
                    Line2D([0], [0], color=colors.get('GB+GG'), label='GB + GG'),
                    Line2D([0], [0], color=colors.get('GB+GG_vp'), label='GB + GG_vp')]

legend_handles_heat = [Line2D([0], [0], marker=markers_heat.get('standard outflow'),color='w',markerfacecolor='tab:blue',markeredgecolor='tab:blue', markersize=marker_size, label=r'in heat: standard, $T^o$'),
                    Line2D([0], [0], marker=markers_heat.get('standard delta'),color='w',markerfacecolor='tab:blue',markeredgecolor='tab:blue', markersize=marker_size, label=r'in heat: standard, $\Delta T$'),
                    Line2D([0], [0], marker=markers_heat.get('half_link_flow outflow'),color='w',markerfacecolor='tab:blue',markeredgecolor='tab:blue', markersize=marker_size, label=r'in heat: terminal link,  $T^o$'),
                    Line2D([0], [0], marker=markers_heat.get('half_link_flow delta'),color='w',markerfacecolor='tab:blue',markeredgecolor='tab:blue', markersize=marker_size, label=r'in heat: terminal link,  $\Delta T$'),
                    Line2D([0], [0], color='tab:blue',ls=linestyles_heat.get(1), label='node set 1'),
                    Line2D([0], [0], color='tab:blue',ls=linestyles_heat.get(2), label='node set 2'),
                    Line2D([0], [0], color='k',ls=':', label='tolerance')]

legend_handles_sc = [Line2D([0], [0], color=colors_sc.get('gas'),ls=linestyles_sc.get('gas'),marker=markers_gas.get('full fa'), label=r'gas: full, $f^{q(\Delta p)}$'),
                    Line2D([0], [0], color=colors_sc.get('gas'),ls=linestyles_sc.get('gas'),marker=markers_gas.get('full fb'), label=r'gas: full, $f^{\Delta p(q)}$'),
                    Line2D([0], [0], color=colors_sc.get('gas'),ls=linestyles_sc.get('gas'),marker=markers_gas.get('nodal fa'), label=r'gas: nodal, $f^{q(\Delta p)}$'),
                    Line2D([0], [0], color=colors_sc.get('elec'),ls=linestyles_sc.get('elec'), label=r'electrical'),
                    Line2D([0], [0], marker=markers_heat.get('standard outflow'), color=colors_sc.get('heat'),ls=linestyles_sc.get('heat'), label=r'heat: standard, $T^o$'),
                    Line2D([0], [0], marker=markers_heat.get('standard delta'), color=colors_sc.get('heat'),ls=linestyles_sc.get('heat'), label=r'heat: standard, $\Delta T$'),
                    Line2D([0], [0], marker=markers_heat.get('half_link_flow outflow'), color=colors_sc.get('heat'),ls=linestyles_sc.get('heat'), label=r'heat: terminal link,  $T^o$'),
                    Line2D([0], [0], marker=markers_heat.get('half_link_flow delta'), color=colors_sc.get('heat'),ls=linestyles_sc.get('heat'), label=r'heat: terminal link,  $\Delta T$'),
                    Line2D([0], [0], color='k',ls=':', label='tolerance')]

def custom_formatwarning(message,category,filename,lineno,*args,line=''): #message, category, filename, lineno, line=''
    #https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    print(str(category.__name__)+' in '+str(filename)+' in line '+str(lineno)+': '+str(message))
warnings.showwarning = custom_formatwarning

def create_network(p1g=9*bar,q1=2,q3=1,V1=50*kV,delta1=0.,delta2=0.,V2=50*kV,P1=-1*MW,P2=-1*MW,P3=1.5*MW,Q3=1.5*MW,Ts1=100.,p1h=9*bar,To2=100.,To3=50.,phi1=-1*MW,phi2=-1*MW,phi3=1.5*MW,To=100.,dTc=50.,L12=4*km,L23=5*km,Dg=1*cm,E=.98,link_type_g='pipe_high_pres_weymouth',hydr_eq_gas='fb',De=1*cm,link_type_e='pi_line',Dh=0.15,lam=0.2,link_type_h='standard_pipe_low_pres_pole',node_set_elec=1,node_set_heat=1,heat_load='outflow',coupling_type='CHP',c_GG=15*Ebs,d_GG=1.5*Ebs,coupling_point=1,node_set=1):
    """Create a heterogeneous network, where every single-carrier network consists of 3 nodes in one line.

    Parameters
    ----------
    heat_load : str, optional
        Determines which model is used for the all the heat load (sink/source), both in the single-carrier network, and in the multi-carrier network. If 'outflow', the outlfow temperature is specified. If 'delta', the temperature difference between supply and outflow temperature is specified. Default is 'outflow'. In both cases, the heat power is specified.
    coupling_type : str, optional
        Which coupling type to use. Options are 'CHP', 'EH', 'GB+GG', or 'GB+GG_vp'. Default is 'CHP'
    coupling_point : int, optional
        Number (name) of the single carrier node that is coupled. Default is 1 (for node gn1,en1,hn1), other option are 2 or 3.

    Returns
    -------
    het_net : HeterogeneousNetwork
        The heterogeneous network
    """
    # create single carrier networks
    gas_net = GasNet.create_network(p1=p1g,q3=q3,hydr_eq=hydr_eq_gas,L12=L12,L23=L23,D=Dg,E=E,link_type=link_type_g)
    elec_net = ElecNet.create_network(V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,L12=L12,L23=L23,D=De,link_type=link_type_e,node_set=node_set_elec)
    heat_net = HeatNet.create_network(Ts1=Ts1,p1=p1h,To2=To2,To3=To3,phi2=phi2,phi3=phi3,L12 =L12,L23=L23,D=Dh,lam=lam,link_type=link_type_h,node_set=node_set_heat,source_node=heat_load,sink_node=heat_load)

    # adjust position for plotting
    y_shift_elec = 0.3
    x_shift_elec = 0#0.7
    y_shift_heat = 2*y_shift_elec
    x_shift_heat = 0#2*x_shift_elec
    for en in elec_net.get_nodes():
        en.x += x_shift_elec
        en.y += y_shift_elec
    for hn in heat_net.get_nodes():
        hn.x += x_shift_heat
        hn.y += y_shift_heat

    if coupling_point == 1:
        x_coup = -1
        y_coup = y_shift_elec
    elif coupling_point == 2:
        x_coup = 0.5
        y_coup = 1.5*y_shift_elec
    else:
        raise ValueError('The coupling point must be either 1 or 2.')

    if coupling_point in [1,2,3]:
        gas_coupling_node = gas_net.nodes[coupling_point-1]
        elec_coupling_node = elec_net.nodes[coupling_point-1]
        heat_coupling_node = heat_net.nodes[coupling_point-1]
    else:
        raise ValueError("Enter valid coupling point. Either 1, 2, or 3.")

    # coupling
    if coupling_type == 'CHP':
        eta_CHP_array = np.array([eta_GG, eta_GB])
        if coupling_point == 1:
            if node_set == 1:
                cn = HeterogeneousNode('cn',node_type=0,x=x_coup,y=y_coup,unit_type='geh_CHP',unit_params={'eta':eta_CHP_array,'GHV':GHV}) # To unknown
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=0) # To of coupling (source) is unknown
            elif node_set == 2:
                cn = HeterogeneousNode('cn',node_type=1,x=x_coup,y=y_coup,unit_type='geh_CHP',unit_params={'eta':eta_CHP_array,'GHV':GHV}) # To known
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=6,Tsstart=To) # To of coupling (source) is known
            else:
                raise ValueError("node_set must be 1 or 2 for 'CHP' coupling at coupling_point 1")
        elif coupling_point == 2:
            if node_set == 1 or node_set == 2 or node_set == 3:
                cn = HeterogeneousNode('cn',node_type=1,x=x_coup,y=y_coup,unit_type='geh_CHP',unit_params={'eta':eta_CHP_array,'GHV':GHV}) # To known
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=6,Tsstart=To) # To of coupling (source) is known
            else:
                raise ValueError("node_set must be 1, 2 or 3 for 'CHP' coupling at coupling_point 2")
    elif coupling_type == 'EH':
        C = np.array([[nu*eta_GG],[(1-nu)*eta_GB]])
        Ein_carriers = ['g']
        Eout_carriers = ['e','h']
        if coupling_point == 1:
            if node_set == 1:
                cn = HeterogeneousNode('cn',node_type=0,x=x_coup,y=y_coup,unit_type='EH',unit_params={'C':C,'GHV':GHV,'Eout_carriers':Eout_carriers,'Ein_carriers':Ein_carriers}) # To unknown
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=0) # To of coupling (source) is unknown
            elif node_set == 2:
                cn = HeterogeneousNode('cn',node_type=1,x=x_coup,y=y_coup,unit_type='EH',unit_params={'C':C,'GHV':GHV,'Eout_carriers':Eout_carriers,'Ein_carriers':Ein_carriers}) # To known
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=6,Tsstart=To) # To of coupling (source) is known
            else:
                raise ValueError("node_set must be 1 or 2 for 'EH' coupling at coupling_point 1")
        elif coupling_point == 2:
            if node_set == 1 or node_set == 2 or node_set == 3:
                cn = HeterogeneousNode('cn',node_type=1,x=x_coup,y=y_coup,unit_type='EH',unit_params={'C':C,'GHV':GHV,'Eout_carriers':Eout_carriers,'Ein_carriers':Ein_carriers}) # To known
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=6,Tsstart=To) # To of coupling (source) is known
            else:
                raise ValueError("node_set must be 1, 2, or 3 for 'EH' coupling at coupling_point 2")
    elif coupling_type == 'GB+GG':
        cnGG = HeterogeneousNode('cn',node_type=0,x=x_coup,y=y_coup,unit_type='ge_gas_fired_gen',unit_params={'eta':eta_GG,'GHV':GHV})
        if coupling_point == 1:
            if node_set == 1:
                cn = HeterogeneousNode('cn',node_type=0,x=x_coup,y=y_coup,unit_type='gh_gas_boiler',unit_params={'eta':eta_GB,'GHV':GHV}) # To unknown
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=0) # To of coupling (source) is unknown
            elif node_set == 2:
                cn = HeterogeneousNode('cn',node_type=1,x=x_coup,y=y_coup,unit_type='gh_gas_boiler',unit_params={'eta':eta_GB,'GHV':GHV}) # To known
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=6,Tsstart=To) # To of coupling (source) is known
            else:
                raise ValueError("node_set must be 1 or 2 for 'GB+GG' coupling at coupling_point 1")
        elif coupling_point == 2:
            if node_set == 1 or node_set == 2 or node_set == 3:
                cn = HeterogeneousNode('cn',node_type=1,x=x_coup,y=y_coup,unit_type='gh_gas_boiler',unit_params={'eta':eta_GB,'GHV':GHV}) # To known
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=6,Tsstart=To) # To of coupling (source) is known
            else:
                raise ValueError("node_set must be 1, 2 or 3 for 'GB+GG' coupling at coupling_point 2")
    elif coupling_type == 'GB+GG_vp':
        cnGG = HeterogeneousNode('cn',node_type=0,x=x_coup,y=y_coup,unit_type='ge_gas_fired_gen_valve_point',unit_params={'a':a_GG,'b':b_GG,'c':c_GG,'d':d_GG,'e':e_GG,'Pmin':Pmin,'GHV':GHV})
        if coupling_point == 1:
            if node_set == 1:
                cn = HeterogeneousNode('cn',node_type=0,x=x_coup,y=y_coup,unit_type='gh_gas_boiler',unit_params={'eta':eta_GB,'GHV':GHV}) # To unknown
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=0) # To of coupling (source) is unknown
            elif node_set == 2:
                cn = HeterogeneousNode('cn',node_type=1,x=x_coup,y=y_coup,unit_type='gh_gas_boiler',unit_params={'eta':eta_GB,'GHV':GHV}) # To known
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=6,Tsstart=To) # To of coupling (source) is known
            else:
                raise ValueError("node_set must be 1 or 2 for 'GB+GG_vp' coupling at coupling_point 1")
        elif coupling_point == 2:
            if node_set == 1 or node_set == 2 or node_set == 3:
                cn = HeterogeneousNode('cn',node_type=1,x=x_coup,y=y_coup,unit_type='gh_gas_boiler',unit_params={'eta':eta_GB,'GHV':GHV}) # To known
                hlc = HeatLink('hl_c',cn,heat_coupling_node,link_params={'carrier':HeatNet.water},bc_type=6,Tsstart=To) # To of coupling (source) is known
            else:
                raise ValueError("node_set must be 1, 2 or 3 for 'GB+GG_vp' coupling at coupling_point 2")
    else:
        raise ValueError("Enter valid coupling type. Either 'EH', 'CHP', 'GB+GG', or 'GB+GG_vp'.")
    if heat_load == 'delta' and cn.node_type == 1:
        cn.node_type = 2
        hlc.bc_type = 10
        hlc.dTstart=dTc

    # change node types of homogeneous nodes which are connected to the heterogeneous coupling node
    if coupling_type == 'CHP' or coupling_type == 'GB+GG' or coupling_type == 'GB+GG_vp':
        if coupling_point == 1:
            if node_set == 1:
                elec_coupling_node.node_type = 5 # PQVdelta node (this node was originally a slack node, so it didn't have a half link)
                heat_coupling_node.node_type = 7 # reference temperature (junction) node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
            elif node_set == 2:
                elec_coupling_node.node_type = 5 # PQVdelta node (this node was originally a slack node, so it didn't have a half link)
                heat_coupling_node.node_type = 5 # reference (junction) node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
        elif coupling_point == 2:
            if node_set == 1:
                # check if q1 and P1 have reasonable values (i.e. do a rough check for solvability)
                if np.abs(q1) < q3 > 0:
                    raise ValueError("input q1 is smaller than output q3: q1 = {:.2f}, q3 = {:.2f}".format(q1,q3))
                elif np.abs(P1) > P3:
                    raise ValueError("input P1 is bigger than output P3: P1 = {:.2f}, P3 = {:.2f}".format(P1,P3))
                gas_net.nodes[0].node_type = 3 # ref. load node (this node was originally a slack node, so it didn't have a half link)
                GasHalfLink(gas_net.nodes[0].name+'_hl',gas_net.nodes[0],q=q1)
                for hl in gas_coupling_node.get_half_links():
                    hl.q = 0
                elec_net.nodes[0].node_type = 1 # generator node (this node was originally a slack node, so it didn't have a half link)
                ElectricalHalfLink(elec_net.nodes[0].name+'_hl',elec_net.nodes[0],P=P1)
                elec_coupling_node.node_type = 5 # PQVdelta node
                elec_coupling_node.V = V2
                elec_coupling_node.delta = delta2
                for hl in elec_coupling_node.get_half_links():
                    hl.P = 0
                    hl.Q = 0
                heat_coupling_node.node_type = 2 # junction node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
            elif node_set == 2:
                # check if q1 and phi1 have reasonable values (i.e. do a rough check for solvability)
                if np.abs(q1) < q3:
                    raise ValueError("input q1 is smaller than output q3: q1 = {:.2f}, q3 = {:.2f}".format(q1,q3))
                elif np.abs(phi1) > phi3:
                    raise ValueError("input phi1 is bigger than output phi3: phi1 = {:.2f}, phi3 = {:.2f}".format(phi1,phi3))
                gas_net.nodes[0].node_type = 3 # ref. load node (this node was originally a slack node, so it didn't have a half link)
                GasHalfLink(gas_net.nodes[0].name+'_hl',gas_net.nodes[0],q=q1)
                for hl in gas_coupling_node.get_half_links():
                    hl.q = 0
                elec_coupling_node.node_type = 6 # PQV node
                elec_coupling_node.V = V2
                for hl in elec_coupling_node.get_half_links():
                    hl.P = 0
                    hl.Q = 0
                heat_net.nodes[0].half_links[0].dphi = phi1
                if heat_load == 'outflow':
                    heat_net.nodes[0].node_type = 3 # source ref. node
                    heat_net.nodes[0].half_links[0].Ts = Ts1
                    heat_net.nodes[0].half_links[0].bc_type = 2
                else:
                    heat_net.nodes[0].node_type = 13 # source ref. temp. dif. node
                    heat_net.nodes[0].half_links[0].bc_type = 4
                    heat_net.nodes[0].half_links[0].dT= dTc
                heat_coupling_node.node_type = 2 # junction node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
            elif node_set == 3:
                if np.abs(P1) > P3:
                    raise ValueError("input P1 is bigger than output P3: P1 = {:.2f}, P3 = {:.2f}".format(P1,P3))
                elif np.abs(phi1) > phi3:
                    raise ValueError("input phi1 is bigger than output phi3: phi1 = {:.2f}, phi3 = {:.2f}".format(phi1,phi3))
                for hl in gas_coupling_node.get_half_links():
                    hl.q = 0
                elec_net.nodes[0].node_type = 1 # generator node (this node was originally a slack node, so it didn't have a half link)
                ElectricalHalfLink(elec_net.nodes[0].name+'_hl',elec_net.nodes[0],P=P1)
                elec_coupling_node.node_type = 5 # PQVdelta node
                elec_coupling_node.V = V2
                elec_coupling_node.delta = delta2
                for hl in elec_coupling_node.get_half_links():
                    hl.P = 0
                    hl.Q = 0
                heat_net.nodes[0].half_links[0].dphi = phi1
                if heat_load == 'outflow':
                    heat_net.nodes[0].node_type = 3 # source ref. node
                    heat_net.nodes[0].half_links[0].Ts = Ts1
                    heat_net.nodes[0].half_links[0].bc_type = 2
                else:
                    heat_net.nodes[0].node_type = 13 # source ref. temp. dif. node
                    heat_net.nodes[0].half_links[0].bc_type = 4
                    heat_net.nodes[0].half_links[0].dT= dTc
                heat_coupling_node.node_type = 2 # junction node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
        else:
            raise ValueError("New node types / additional BC not yet implemented")
    elif coupling_type == 'EH':
        if coupling_point == 1:
            if node_set == 1:
                elec_coupling_node.node_type = 4 # QVdelta node (this node was originally a slack node, so it didn't have a half link)
                ElectricalHalfLink(elec_coupling_node.name+'_hl',elec_coupling_node)
                heat_coupling_node.node_type = 7 # reference temperature (junction) node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
            elif node_set == 2:
                elec_coupling_node.node_type = 5 # PQVdelta node (this node was originally a slack node, so it didn't have a half link)
                ElectricalHalfLink(elec_coupling_node.name+'_hl',elec_coupling_node)
        elif coupling_point == 2:
            if node_set == 1:
                # check if q1 has a reasonable value (i.e. do a rough check for solvability)
                qc_max = q3 - np.abs(q1)
                Pc_max, phic_max = C.dot(np.array([GHV*np.abs(qc_max)]))
                if qc_max > 0:
                    raise ValueError("input q1 is smaller than output q3: q1 = {:.2f}, q3 = {:.2f}".format(q1,q3))
                elif Pc_max > P3:
                    raise ValueError("input q1 is too big for the current EH parameters and output P3: Pc_max = {:.2f}, P3 = {:.2f}".format(Pc_max,P3))
                elif phic_max > phi3:
                    raise ValueError("input q1 is too big for the current EH parameters and output phi3: phic_max = {:.2f}, phi3 = {:.2f}".format(phic_max,phi3))
                gas_net.nodes[0].node_type = 3 # ref. load node (this node was originally a slack node, so it didn't have a half link)
                GasHalfLink(gas_net.nodes[0].name+'_hl',gas_net.nodes[0],q=q1)
                for hl in gas_coupling_node.get_half_links():
                    hl.q = 0
                elec_coupling_node.node_type = 6 # PQV node
                elec_coupling_node.V = V2
                for hl in elec_coupling_node.get_half_links():
                    hl.P = 0
                    hl.Q = 0
                heat_coupling_node.node_type = 2 # junction node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
            elif node_set == 2:
                Pc_max = np.abs(P1) - P3
                phic_max = C[1]/C[0]*Pc_max
                if np.abs(P1) > P3:
                    raise ValueError("input P1 is bigger than output P3: P1 = {:.2f}, P3 = {:.2f}".format(P1,P3))
                elif phic_max > phi3:
                    raise ValueError("input P1 is too small for the current EH parameters and output phi3: phic_max = {:.2f}, phi3 = {:.2f}".format(phic_max,phi3))
                for hl in gas_coupling_node.get_half_links():
                    hl.q = 0
                elec_net.nodes[0].node_type = 1 # generator node (this node was originally a slack node, so it didn't have a half link)
                ElectricalHalfLink(elec_net.nodes[0].name+'_hl',elec_net.nodes[0],P=P1)
                elec_coupling_node.node_type = 5 # PQVdelta node
                elec_coupling_node.V = V2
                elec_coupling_node.delta = delta2
                for hl in elec_coupling_node.get_half_links():
                    hl.P = 0
                    hl.Q = 0
                heat_coupling_node.node_type = 2 # junction node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
            elif node_set == 3:
                phic_max = np.abs(phi1) - phi3
                Pc_max = C[0]/C[1]*phic_max
                if np.abs(phi1) > phi3:
                    raise ValueError("input phi1 is bigger than output phi3: phi1 = {:.2f}, phi3 = {:.2f}".format(phi1,phi3))
                elif Pc_max > P3:
                    raise ValueError("input phi1 is too small for the current EH parameters and output P3: Pc_max = {:.2f}, P3 = {:.2f}".format(Pc_max,P3))
                for hl in gas_coupling_node.get_half_links():
                    hl.q = 0
                elec_coupling_node.node_type = 6 # PQV node
                elec_coupling_node.V = V2
                for hl in elec_coupling_node.get_half_links():
                    hl.P = 0
                    hl.Q = 0
                heat_net.nodes[0].half_links[0].dphi = phi1
                if heat_load == 'outflow':
                    heat_net.nodes[0].node_type = 3 # source ref. node
                    heat_net.nodes[0].half_links[0].Ts = Ts1
                    heat_net.nodes[0].half_links[0].bc_type = 2
                else:
                    heat_net.nodes[0].node_type = 13 # source ref. temp. dif. node
                    heat_net.nodes[0].half_links[0].bc_type = 4
                    heat_net.nodes[0].half_links[0].dT= dTc
                heat_coupling_node.node_type = 2 # junction node
                for hl in heat_coupling_node.get_half_links():
                    heat_coupling_node.remove_half_link(hl)
                    heat_net.remove_half_link(hl)
    else:
        raise ValueError("New node types / additional BC not yet implemented")

    glc = GasLink('gl_c',gas_coupling_node,cn)
    gas_net.add_link(glc)
    heat_net.add_link(hlc)
    if coupling_type == 'GB+GG' or coupling_type == 'GB+GG_vp':
        glcGG = GasLink('gl_cGG',gas_coupling_node,cnGG)
        gas_net.add_link(glcGG)
        elc = ElectricalLink('el_cGG',cnGG,elec_coupling_node)
    else:
        elc = ElectricalLink('el_c',cn,elec_coupling_node)
    elec_net.add_link(elc)

    het_net = HeterogeneousNetwork('3N one line')
    het_net.add_network(gas_net)
    het_net.add_network(elec_net)
    het_net.add_network(heat_net)

    het_net.add_node(cn)
    if coupling_type == 'GB+GG' or coupling_type == 'GB+GG_vp':
        het_net.add_node(cnGG)
    return gas_net,elec_net,heat_net,het_net

def initialize_network(gas_net,elec_net,heat_net,het_net,q0_init=1,q1_init=1,p2g_init=8*bar,p3g_init=7*bar,V2_init=50*kV,V3_init=50*kV,delta1_init=0.,delta2_init=0.,delta3_init=0.,m0_init=1,m1_init=1,p2h_init=8*bar,p3h_init=6*bar,Ts1_init=100.,Ts2_init=100.,Ts3_init=100.,Tr1_init=50.,Tr2_init=50.,Tr3_init=50.,qc_init=1.,Pc_init=1.5*MW,Qc_init=1.5*MW,mc_init=1.,Toc_init=100.,phic_init=1.5*MW,node_set_elec=1,node_set_heat=1,coupling_type='CHP',coupling_point=1,node_set=1,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None}):
    """Create a gas network consisting of 3 nodes in one line.

    Parameters
    ----------

    Returns
    -------
    x0 : np array
        initial guess
    """
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks  = het_net.get_x_entries(formulation=formulation)
    # gas part
    if formulation.get('gas') == 'nodal':
        for link in het_net.get_links():
            if isinstance(link,GasLink) and link.link_eq_form == 'dp_of_q':
                warnings.warn("The link {} uses press. drop as function of link flow (fb instead of fa), but formulation is 'nodal'. The link equation for this link is changed to fa!!".format(link.name))
                link.set_type(link.link_type,link.link_params,link_eq_form='q_of_dp')
    q_init = np.array([q0_init,q1_init]) #[kg/s]
    p_init = np.array([p2g_init,p3g_init]) #[Pa]
    if formulation.get('gas') == 'nodal':
        xg_init = p_init
    else:
        xg_init = np.concatenate((q_init,p_init))

    # electrical part
    delta_init_vec = np.array([delta1_init,delta2_init,delta3_init]) #[rad]
    V_init_vec = np.array([V2_init,V3_init]) #[V]
    delta_init = list()
    V_init = list()
    for ind_n,node_e in enumerate(elec_net.get_nodes()):
        if node_e in unknown_delta_nodes:
            delta_init.append(delta_init_vec[ind_n])
        if node_e in unknown_V_nodes: # node 0 always has V known
            V_init.append(V_init_vec[ind_n-1])
    xe_init = np.concatenate((np.array(delta_init),np.array(V_init)))

    # heat part
    m_init = np.array([m0_init,m1_init]) #[kg/s]
    p_init = np.array([p2h_init,p3h_init]) #[Pa]
    Ts_init_vec = np.array([Ts1_init,Ts2_init,Ts3_init]) #[C]
    Tr_init_vec = np.array([Tr1_init,Tr2_init,Tr3_init]) #[C]
    Ts_init = list()
    Tr_init = list()
    m_hl_init = list()
    Ts_hl_init = list()
    Tr_hl_init = list()
    for ind_n,node_h in enumerate(heat_net.get_nodes()):
        if node_h in unknown_Ts_nodes:
            Ts_init.append(Ts_init_vec[ind_n])
        if node_h in unknown_Tr_nodes:
            Tr_init.append(Tr_init_vec[ind_n])
        if formulation.get('heat') == 'half_link_flow':
            for hl in node_h.get_half_links():
                if hl in unknown_m_halflinks:
                    m_hl_init.append(hl.flow())
                if hl in unknown_Ts_halflinks:
                    Ts_hl_init.append(hl.Ts)
                if hl in unknown_Trs_halflinks:
                    Tr_hl_init.append(hl.Tr)

    if formulation.get('heat') == 'half_link_flow':
        xh_init = np.concatenate((m_init,np.array(m_hl_init),p_init,np.array(Ts_init),np.array(Tr_init),np.array(Ts_hl_init),np.array(Tr_hl_init)))
    else:
        xh_init = np.concatenate((m_init,p_init,np.array(Ts_init),np.array(Tr_init)))

    # coupling part
    if coupling_type == 'GB+GG' or coupling_type == 'GB+GG_vp':
        xc_init = np.array([qc_init,qc_init,Pc_init,Qc_init,mc_init,phic_init])
    else:
        xc_init = np.array([qc_init,Pc_init,Qc_init,mc_init,phic_init])
    if unknown_Ts_links:
        xc_init = np.concatenate((xc_init,np.array([Toc_init])))

    x_init = np.concatenate((xg_init,xe_init,xh_init,xc_init))
    #print(x_init)
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)
    return x0

def run_load_flow(pgbase,qbase,Vbase,Sbase,phbase,mbase,Tbase,phibase,Egbase,deltabase=1,p1g=9*bar,q1=2,q3=1,V1=50*kV,delta1=0.,delta2=0.,V2=50*kV,P1=-1*MW,P2=-1*MW,P3=1.5*MW,Q3=1.5*MW,Ts1=100.,p1h=9*bar,To2=100.,To3=50.,phi1=-1*MW,phi2=-1*MW,phi3=1.5*MW,To=100.,dTc=50.,q0_init=1,q1_init=1,p2g_init=8*bar,p3g_init=7*bar,V2_init=50*kV,V3_init=50*kV,delta1_init=0.,delta2_init=0.,delta3_init=0.,m0_init=1,m1_init=1,p2h_init=8*bar,p3h_init=6*bar,Ts1_init=100.,Ts2_init=100.,Ts3_init=100.,Tr1_init=50.,Tr2_init=50.,Tr3_init=50.,qc_init=1.,Pc_init=1.5*MW,Qc_init=1.5*MW,mc_init=1.,Toc_init=100.,phic_init=1.5*MW,L12=4*km,L23=5*km,Dg=10*cm,E=.98,link_type_g='pipe_high_pres_weymouth',hydr_eq_gas='fb',De=10*cm,link_type_e='pi_line',Dh=0.15,lam=0.2,link_type_h='standard_pipe_low_pres_pole',node_set_elec=1,node_set_heat=1,heat_load='outflow',coupling_type='CHP',c_GG=15*Ebs,d_GG=1.5*Ebs,coupling_point=1,node_set=1,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None},tol = 1e-6,max_iter = 150):
    """Stead-state load flow analysis of heat network

    Parameters
    ----------

    """
    print('\nLoad flow with {} at node {}, node set {}, heat load {}, form. gas {}, hyrd. eq. gas {},  form. heat {}'.format(coupling_type,coupling_point,node_set,heat_load,formulation.get('gas'),hydr_eq_gas,formulation.get('heat')))
    # create network
    gas_net,elec_net,heat_net,het_net = create_network(p1g=p1g,q1=q1,q3=q3,V1=V1,delta1=delta1,delta2=delta2,V2=V2,P1=P1,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi1=phi1,phi2=phi2,phi3=phi3,To=To,dTc=dTc,L12=L12,L23=L23,Dg=Dg,E=E,link_type_g=link_type_g,hydr_eq_gas=hydr_eq_gas,De=De,link_type_e=link_type_e,Dh=Dh,lam=lam,link_type_h=link_type_h,node_set_elec=node_set_elec,node_set_heat=node_set_heat,heat_load=heat_load,coupling_type=coupling_type,c_GG=c_GG,coupling_point=coupling_point,node_set=node_set)
    # initialize
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net,q0_init=q0_init,q1_init=q1_init,p2g_init=p2g_init,p3g_init=p3g_init,V2_init=V2_init,V3_init=V3_init,delta1_init=delta1_init,delta2_init=delta2_init,delta3_init=delta3_init,m0_init=m0_init,m1_init=m1_init,p2h_init=p2h_init,p3h_init=p3h_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,qc_init=qc_init,Pc_init=Pc_init,Qc_init=Qc_init,mc_init=mc_init,Toc_init=Toc_init,phic_init=phic_init,node_set_elec=node_set_elec,node_set_heat=node_set_heat,coupling_type=coupling_type,coupling_point=coupling_point,node_set=node_set,formulation=formulation)

    # solve network
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
    print('After {} iterations, error is {:.4e}'.format(iters,err_vec[-1]))

    return het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec

def legends():
    """Make separate figures with the legends used in the different convergence plots"""

    # legend used for the convergence plot, if the node sets are compared
    fig_legend_node_sets = plt.figure('Legend_node_sets')
    ax_legend_node_sets = fig_legend_node_sets.gca()
    ax_legend_node_sets.axis('off')
    fig_legend_node_sets.patch.set_visible(False)
    ax_legend_node_sets.legend(handles=legend_handles_node_sets,loc='center')

    # legend used for the convergence plot, if the different physical models are compared
    fig_legend_models = plt.figure('Legend_models')
    ax_legend_models = fig_legend_models.gca()
    ax_legend_models.axis('off')
    fig_legend_models.patch.set_visible(False)
    ax_legend_models.legend(handles=legend_handles_models,loc='center')

    # legend used for single carrier heat network
    fig_legend_heat = plt.figure('Legend_node_sets_heat')
    ax_legend_heat = fig_legend_heat.gca()
    ax_legend_heat.axis('off')
    fig_legend_heat.patch.set_visible(False)
    ax_legend_heat.legend(handles=legend_handles_heat,loc='center')

    # legend used for single carrier networks
    fig_legend_sc = plt.figure('Legend_sc')
    ax_legend_sc = fig_legend_sc.gca()
    ax_legend_sc.axis('off')
    fig_legend_sc.patch.set_visible(False)
    ax_legend_sc.legend(handles=legend_handles_sc,loc='center')

def compare_conv_node_sets(hydr_eq_gas='fb',form_gas='full',form_heat='standard',heat_load='outflow',show_plots=False):
    """Compare the convergence of load flow for different topologies and different node sets (both in multi-carrier and single-carrier)
    """
    # node sets and coupling type and coupling point
    node_sets_elec = [1,2]
    node_sets_heat = [1,2]
    coupling_types = ['CHP','GB+GG','GB+GG_vp','EH']
    coupling_points = [1,2]
    node_sets = [1,2,3]

    x_sol_dict = dict()
    errors_dict = dict()
    iters_dict = dict()
    x_sol_sc = dict()
    errors_sc = dict()
    iters_sc = dict()
    max_iters_used = 0

    # physical parameters
    c_GG = .01*Ebs
    d_GG = c_GG/10
    Dg = 10*cm
    Dh = 30*cm
    lam = 0.002

    # boundary conditions (when needed)
    p1g = 50*bar #[Pa]
    q3 = 1. #[kg/s]
    q1 = -1.05*q3 #[kg/s]
    V1 = 50*kV #[V]
    delta1 = 0. #[rad]
    delta2 = 0. #[rad]
    V2 = 50*kV #[V]
    P2 = -1*MW #[W]
    P3 = 1.5*-P2 #[W]
    P1 = .9*-P3 #[W]
    Q3 = P3 #[Var]
    Ts1 = 100. #[C]
    p1h = 9*bar
    To2 = .9*Ts1
    To3 = 50. #[C]
    phi2 = -1*MW #[W]
    phi3 = 5.5*-phi2 #[W]
    phi1 = -.9*phi3 #[W]
    To = 100. #[C]
    dT = 50.

    # initial guesses (when needed)
    p2g_init = .99*p1g#30*bar #[Pa]
    p3g_init = .98*p1g#28*bar #[Pa]
    q12_init = q3 #[kg/s]
    q23_init= q3 #[kg/s]
    V2_init = 50*kV #[V]
    V3_init = 50*kV #[V]
    delta1_init = 0. #[rad]
    delta2_init = 0. #[rad]
    delta3_init = 0. #[rad]
    m0_init=1 #[kg/s]
    m1_init=1 #[kg/s]
    p2h_init=8*bar #[Pa]
    p3h_init=6*bar #[Pa]
    Ts1_init=100. #[C]
    Ts2_init=100. #[C]
    Ts3_init=100. #[C]
    Tr1_init=50. #[C]
    Tr2_init=50. #[C]
    Tr3_init=50. #[C]
    qc_init = q3 #[kg/s]
    Pc_init = P3 #[W]
    Qc_init = Q3 #[W]
    mc_init = 1 #[kg/s]
    phic_init = phi3 #[kg/s]
    Toc_init = Ts1 #[C]

    # solver information
    tol = 1e-6
    max_iter = 15
    formulation={'gas':form_gas,'elec':'complex_power','heat':form_heat,'het':None}

    # scaling
    pgbase = bar
    qbase = 1.
    deltabase = 1.
    Vbase = 50*kV
    Sbase = MW
    phbase = bar
    mbase = 1.
    Tbase = 100.
    phibase = MW
    Egbase = phibase

    if show_plots:
        fig_gg_vp = plt.figure('GG_vp') # figure to check non-linearity of gas-fired generator with valve-point effect
        ax_gg_vp = fig_gg_vp.gca()
        P_vp_min= .5 #fraction of Pc_sol used as lower xlimit
        P_vp_max = 2 #fraction of Pc_sol used as lower xlimit
        eta_gg = list()
        Pc_min = MW**2
        Pc_max = 0

        # make figures to plot convergence
        fig_conv1_het, ax_conv1 = plt.subplots(2, 2, num='Convergence plot networks, coupling point 1, {}, {}, {}, {}'.format(form_gas,hydr_eq_gas,form_heat,heat_load))
        ax_conv1_het = ax_conv1[0,0]
        ax_conv1_gas = ax_conv1[0,1]
        ax_conv1_elec = ax_conv1[1,0]
        ax_conv1_heat = ax_conv1[1,1]
        ax_conv1_het.set_title('Heterogeneous network')
        ax_conv1_gas.set_title('Gas network')
        ax_conv1_elec.set_title('Electrical network')
        ax_conv1_heat.set_title('Heat network')
        max_iters_used = 0
        max_iters_used_elec = 0
        max_iters_used_heat = 0

        fig_conv2_het, ax_conv2 = plt.subplots(2, 2, num='Convergence plot networks, coupling point 2, {}, {}, {}, {}'.format(form_gas,hydr_eq_gas,form_heat,heat_load))
        ax_conv2_het = ax_conv2[0,0]
        ax_conv2_gas = ax_conv2[0,1]
        ax_conv2_elec = ax_conv2[1,0]
        ax_conv2_heat = ax_conv2[1,1]
        ax_conv2_het.set_title('Heterogeneous network')
        ax_conv2_gas.set_title('Gas network')
        ax_conv2_elec.set_title('Electrical network')
        ax_conv2_heat.set_title('Heat network')
        max_iters_used = 0
        max_iters_used_elec = 0
        max_iters_used_heat = 0

    # run load flow for single carrier networks
    gas_net_single,x_sol_gas,iters_gas,err_vec_gas,p_sol_single,q_sol_single,q_inj_single = GasNet.run_load_flow(pgbase,qbase,p1=p1g,q3=q3,q0=q12_init,q1=q23_init,p2=p2g_init,p3=p3g_init,D=Dg,formulation=formulation.get('gas'),hydr_eq=hydr_eq_gas,tol=tol,max_iter=max_iter)
    key = '{} {}'.format(form_gas,hydr_eq_gas)
    x_sol_sc[key] = x_sol_gas
    errors_sc[key] = err_vec_gas
    iters_sc[key] = iters_gas
    if show_plots:
        ax_conv1_gas.semilogy(err_vec_gas,marker='.',ls='-',color='tab:green',label='')
        ax_conv2_gas.semilogy(err_vec_gas,marker='.',ls='-',color='tab:green',label='')
        ax_conv1_gas.legend()
        ax_conv2_gas.legend()
    for node_set_elec in node_sets_elec:
        elec_net_single,x_sol_elec,iters_elec,err_vec_elec,delta_sol_single,V_sol_single,S_inj_single,P_edge_single,Q_edge_single = ElecNet.run_load_flow(Vbase,Sbase,deltabase=deltabase,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,V2_init=V2_init,V3_init=V3_init,delta2_init=delta2_init,delta3_init=delta3_init,node_set=node_set_elec,tol=tol,max_iter=max_iter)
        key = 'elec {}'.format(node_set_elec)
        x_sol_sc[key] = x_sol_elec
        errors_sc[key] = err_vec_elec
        iters_sc[key] = iters_elec
        if show_plots:
            ax_conv1_elec.semilogy(err_vec_elec,marker=markers_sc_nodes.get('elec {} heat 1'.format(node_set_elec)),ls='--',color='tab:red',label='node set {}'.format(node_set_elec))
            ax_conv2_elec.semilogy(err_vec_elec,marker=markers_sc_nodes.get('elec {} heat 1'.format(node_set_elec)),ls='--',color='tab:red',label='node set {}'.format(node_set_elec))
            ax_conv1_elec.legend()
            ax_conv2_elec.legend()
            max_iters_used_elec = max(max_iters_used_elec,iters_elec)
    for node_set_heat in node_sets_heat:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
            heat_net_single,x_sol_heat,iters_heat,err_vec_heat,m_vec_single,p_vec_single,Ts_vec_single,Tr_vec_single,m_hl_vec_single,phi_hl_vec_single,Ts_hl_vec_single,Tr_hl_vec_single = HeatNet.run_load_flow(phbase,mbase,Tbase,phibase,Ts1=Ts1,p1=p1h,To2=To2,To3=To3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2h_init,p3_init=p3h_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,dT2=dT,dT3=dT,D=Dh,lam=lam,formulation=formulation.get('heat'),node_set=node_set_heat,source_node=heat_load,sink_node=heat_load,tol=tol,max_iter=max_iter)
            key = '{} {} {}'.format(form_heat,heat_load,node_set_heat)
            x_sol_sc[key] = x_sol_heat
            errors_sc[key] = err_vec_heat
            iters_sc[key] = iters_heat
            if show_plots:
                ax_conv1_heat.semilogy(err_vec_heat,marker=markers_sc_nodes.get('elec 1 heat {}'.format(node_set_heat)),ls='-.',color='tab:blue',label='node set {}'.format(node_set_heat))
                ax_conv2_heat.semilogy(err_vec_heat,marker=markers_sc_nodes.get('elec 1 heat {}'.format(node_set_heat)),ls='-.',color='tab:blue',label='node set {}'.format(node_set_heat))
                ax_conv1_heat.legend()
                ax_conv2_heat.legend()
                max_iters_used_heat = max(max_iters_used_heat,iters_heat)
            print('phi inj SC: {}'.format(phi_hl_vec_single))

    # run load flow for MES
    for coupling_point in coupling_points:
        if show_plots:
            if coupling_point == 1:
                ax_conv_het = ax_conv1_het
            elif coupling_point == 2:
                ax_conv_het = ax_conv2_het
        for node_set_elec in node_sets_elec:
            for node_set_heat in node_sets_heat:
                for coupling_type in coupling_types:
                    for node_set in node_sets:
                        if node_set == 3 and coupling_point == 1:
                            continue # continue to the next iteration
                        print('\nRunning load flow with coupling point = {}, node set elec = {}, node set heat = {}, coupling = {}, node_set = {}'.format(coupling_point,node_set_elec,node_set_heat,coupling_type,node_set))
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                            warnings.filterwarnings("ignore", "Only a",UserWarning)
                            try:
                                het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = run_load_flow(pgbase,qbase,Vbase,Sbase,phbase,mbase,Tbase,phibase,Egbase,deltabase=deltabase,p1g=p1g,q1=q1,q3=q3,V1=V1,delta1=delta1,delta2=delta2,V2=V2,P1=P1,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi1=phi1,phi2=phi2,phi3=phi3,To=To,dTc=dT,q0_init=q12_init,q1_init=q23_init,p2g_init=p2g_init,p3g_init=p3g_init,V2_init=V2_init,V3_init=V3_init,delta1_init=delta1_init,delta2_init=delta2_init,delta3_init=delta3_init,m0_init=m0_init,m1_init=m1_init,p2h_init=p2h_init,p3h_init=p3h_init,Ts1_init=Ts1_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,qc_init=qc_init,Pc_init=Pc_init,Qc_init=Qc_init,mc_init=mc_init,Toc_init=Toc_init,phic_init=phic_init,Dg=Dg,hydr_eq_gas=hydr_eq_gas,Dh=Dh,lam=lam,node_set_elec=node_set_elec,node_set_heat=node_set_heat,heat_load=heat_load,coupling_type=coupling_type,c_GG=c_GG,d_GG=d_GG,coupling_point=coupling_point,node_set=node_set,formulation=formulation,tol=tol,max_iter=max_iter)
                            except:
                                continue
                            key = '{} {} {} {} {} {} {} {} {}'.format(node_set_elec,node_set_heat,form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set,coupling_type)
                            x_sol_dict[key] = x_sol
                            errors_dict[key] = err_vec
                            iters_dict[key] = iters
                            if show_plots:
                                max_iters_used = max(max_iters_used,iters)
                                ax_conv_het.semilogy(err_vec,marker=markers_sc_nodes.get('elec {} heat {}'.format(node_set_elec,node_set_heat)),color=colors.get(coupling_type),ls=linestyles_node_sets.get(node_set),label='node set elec {}, node set heat {}, coupling {}'.format(node_set_elec,node_set_heat,coupling_type))
                            print('Final error = {:.3e}'.format(err_vec[-1]))
                            print('p gas= {} bar'.format(p_g_vec/bar))
                            print('q = {}'.format(q_vec))
                            print('q inj = {}'.format(q_inj))
                            print('delta = {}'.format(delta_vec))
                            print('|V| = {}'.format(V_mag_vec))
                            print('P edge = {} MW'.format(P_edge/MW))
                            print('Q edge = {} MW'.format(Q_edge/MW))
                            print('S inj = {} MW'.format(S_inj/MW))
                            print('p heat= {} bar'.format(p_h_vec/bar))
                            print('m = {}'.format(m_vec))
                            print('Ts = {}'.format(Ts_vec))
                            print('Tr = {}'.format(Tr_vec))
                            print('m hl = {}'.format(m_hl_vec))
                            print('Ts hl = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                            print('Tr hl = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                            print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                            print('Ts hl c = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                            print('Tr hl c = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                            print('dphi hl c = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                            print('m hl c = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                            if show_plots:
                                for cn in het_net.get_nodes(unit_types=['ge_gas_fired_gen_valve_point']): #plot the coupling function around solution, to see how non-linear it is
                                        if err_vec[-1] < tol: # only if converged
                                            #if not ax_gg_vp.lines: # Only depends on Pc_sol, which is only different for different node sets (I think).
                                            Pc_sol = Pc_vec[0]
                                            Pc_min = min(Pc_min,Pc_sol)
                                            Pc_max = max(Pc_max,Pc_sol)
                                            Eg_sol = qc_vec[1]*GHV
                                            Pc_range = np.linspace(P_vp_min*Pc_sol,P_vp_max*Pc_sol)
                                            lin_approx = Eg_sol+(Pc_range-Pc_sol)*cn.dEin_dEout(Pc_sol,None)
                                            Eg = cn.Ein_of_Eout(Pc_range,None)
                                            eta = Pc_sol/Eg_sol
                                            eta_gg.append(eta)
                                            #ax_gg_vp.plot(Pc_range,Eg,label=r'node {}, set {}'.format(coupling_point,node_set,eta))
                                            ax_gg_vp.plot(Pc_range/MW,lin_approx/MW,':',label='lin. Taylor, node {}, set {}, $\eta=${:.4f}'.format(coupling_point,node_set,eta))
                                            ax_gg_vp.plot([Pc_sol/MW,Pc_sol/MW],[min(Eg)/MW,max(Eg)/MW],'k')
                                            ax_gg_vp.set_ylabel(r'GHV$\cdot q$ [MW]')
                                            ax_gg_vp.set_xlabel(r'$P$ [MW]')
                                            ax_gg_vp.annotate(r'$P_c$',xy=(Pc_sol,min(Eg)),ha='center',va='top')
    if show_plots:
        # plot convergence
        max_iters_used = max(max_iters_used,max_iters_used_elec,max_iters_used_heat)
        for ax_conv in [ax_conv1, ax_conv2]:
            for ax_rows in ax_conv:
                for ax in ax_rows:
                    ax.set_xlabel(r'Iteration $k$')
                    ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
                    ax.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
                    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
                    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
                    ax.set_xlim(xmin=0,xmax=max_iters_used+1)

    return x_sol_dict, errors_dict, iters_dict, x_sol_sc, errors_sc, iters_sc, tol

def compare_conv_form():
    """Compare the convergence of load flow for different formulations in the single-carrier networks
    """
    # node sets and coupling type and coupling point
    node_set_heat = 2
    node_set_elec = 2
    coupling_types = ['CHP','GB+GG','GB+GG_vp','EH']
    coupling_points = [1,2]
    node_sets = [1,2,3]
    x_sol_dict = dict()
    errors_dict = dict()
    iters_dict = dict()
    x_sol_sc = dict()
    errors_sc = dict()
    iters_sc = dict()
    iters_gas = dict()
    iters_elec = dict()
    iters_heat = dict()
    max_iters_used = 0

    # physical parameters
    c_GG = .01*Ebs
    d_GG = c_GG/10

    # boundary conditions (when needed)
    p1g = 50*bar #[Pa]
    q3 = 1. #[kg/s]
    q1 = -1.05*q3 #[kg/s]
    V1 = 50*kV #[V]
    delta1 = 0. #[rad]
    delta2 = 0. #[rad]
    V2 = 50*kV #[V]
    P2 = -1*MW #[W]
    P3 = 1.5*-P2 #[W]
    P1 = .9*-P3 #[W]
    Q3 = P3 #[Var]
    Ts1 = 100. #[C]
    p1h = 9*bar
    To2 = .9*Ts1
    To3 = 50. #[C]
    phi2 = -1*MW #[W]
    phi3 = 5.5*-phi2 #[W]
    phi1 = -.9*phi3 #[W]
    To = 100. #[C]
    dT = 50.

    # initial guesses (when needed)
    p2g_init = .99*p1g#30*bar #[Pa]
    p3g_init = .98*p1g#28*bar #[Pa]
    q12_init = q3 #[kg/s]
    q23_init= q3 #[kg/s]
    V2_init = 50*kV #[V]
    V3_init = 50*kV #[V]
    delta1_init = 0. #[rad]
    delta2_init = 0. #[rad]
    delta3_init = 0. #[rad]
    m0_init=1 #[kg/s]
    m1_init=1 #[kg/s]
    p2h_init=8*bar #[Pa]
    p3h_init=6*bar #[Pa]
    Ts1_init=100. #[C]
    Ts2_init=100. #[C]
    Ts3_init=100. #[C]
    Tr1_init=50. #[C]
    Tr2_init=50. #[C]
    Tr3_init=50. #[C]
    qc_init = q3 #[kg/s]
    Pc_init = P3 #[W]
    Qc_init = Q3 #[W]
    mc_init = 1 #[kg/s]
    phic_init = phi3 #[kg/s]
    Toc_init = Ts1 #[C]

    # solver information
    tol = 1e-6
    max_iter = 15
    heat_loads = ['outflow','delta']
    forms_heat = ['standard','half_link_flow']
    forms_gas = ['full','nodal']
    hydr_eqs = ['fa','fb']

    # scaling
    pgbase = bar
    qbase = 1.
    deltabase = 1.
    Vbase = 50*kV
    Sbase = MW
    phbase = bar
    mbase = 1.
    Tbase = 100.
    phibase = MW
    Egbase = phibase

    # load flow
    # run load flow for SC
    for form_gas in forms_gas:
        for hydr_eq_gas in hydr_eqs:
            if not (form_gas == 'nodal' and hydr_eq_gas == 'fb'): #those cannot be combined
                # run load flow for single carrier networks
                gas_net_single,x_sol_gas,iter_gas,err_vec_gas,p_sol_single,q_sol_single,q_inj_single = GasNet.run_load_flow(pgbase,qbase,p1=p1g,q3=q3,q0=q12_init,q1=q23_init,p2=p2g_init,p3=p3g_init,formulation=form_gas,hydr_eq=hydr_eq_gas,tol=tol,max_iter=max_iter)
                key = '{} {}'.format(form_gas,hydr_eq_gas)
                ls = linestyles_gas.get(key)
                x_sol_sc[key] = x_sol_gas
                errors_sc[key] = err_vec_gas
                iters_sc[key] = iter_gas
                iters_gas[key] = iter_gas
                print('\nLoad flow solution for gas network using {} and {}, final error {:.3e}'.format(form_gas,hydr_eq_gas,err_vec_gas[-1]))
                print('p = {} bar'.format(p_sol_single/bar))
                print('q = {}'.format(q_sol_single))
                print('q inj = {}'.format(q_inj_single))
    elec_net_single,x_sol_elec,iter_elec,err_vec_elec,delta_sol_single,V_sol_single,S_inj_single,P_edge_single,Q_edge_single = ElecNet.run_load_flow(Vbase,Sbase,deltabase=deltabase,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,V2_init=V2_init,V3_init=V3_init,delta2_init=delta2_init,delta3_init=delta3_init,node_set=node_set_elec,tol=tol,max_iter=max_iter)
    key = 'elec 2 heat 2'
    marker = markers_sc_nodes.get(key)
    x_sol_sc[key] = x_sol_elec
    errors_sc[key] = err_vec_elec
    iters_sc[key] = iter_elec
    iters_elec[key] = iter_elec
    print('\nLoad flow solution for electrical network with node set {}, final error {:.3e}'.format(node_set_elec,err_vec_elec[-1]))
    print('delta = {}'.format(delta_sol_single))
    print('|V| = {}'.format(V_sol_single))
    print('P edge = {} MW'.format(P_edge_single/MW))
    print('Q edge = {} MW'.format(Q_edge_single/MW))
    print('S inj = {} MW'.format(S_inj_single/MW))
    for form_heat in forms_heat:
        for heat_load in ['outflow','delta']:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                heat_net_single,x_sol_heat,iter_heat,err_vec_heat,m_vec_single,p_vec_single,Ts_vec_single,Tr_vec_single,m_hl_vec_single,phi_hl_vec_single,Ts_hl_vec_single,Tr_hl_vec_single = HeatNet.run_load_flow(phbase,mbase,Tbase,phibase,Ts1=Ts1,p1=p1h,To2=To2,To3=To3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2h_init,p3_init=p3h_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,dT2=dT,dT3=dT,formulation=form_heat,node_set=node_set_heat,source_node=heat_load,sink_node=heat_load,tol=tol,max_iter=max_iter)
                key = '{} {}'.format(form_heat,heat_load)
                x_sol_sc[key] = x_sol_heat
                errors_sc[key] = err_vec_heat
                iters_sc[key] = iter_heat
                iters_heat[key] = iter_heat
                print('\nLoad flow solution for heat network, with node set {}, using {} and {}, final error {:.3e}'.format(node_set_heat,form_heat,heat_load,err_vec_heat[-1]))
                print('p = {} bar'.format(p_vec_single/bar))
                print('m = {}'.format(m_vec_single))
                print('Ts = {}'.format(Ts_vec_single))
                print('Tr = {}'.format(Tr_vec_single))
                print('m hl = {}'.format(m_hl_vec_single))
                print('Ts hl = {}'.format(Ts_hl_vec_single))
                print('Tr hl = {}'.format(Tr_hl_vec_single))
                print('phi hl = {}'.format(phi_hl_vec_single))

    iters_max_sc = 0
    for coupling_point in coupling_points:
        for node_set in node_sets:
            if not (coupling_point == 1 and node_set == 3):
                # make figures to plot convergence
                fig_conv_het, ax_conv = plt.subplots(2, 2, num='Convergence plot networks, coupling point {}, node set {}'.format(coupling_point,node_set))
                ax_conv_het = ax_conv[0,0]
                ax_conv_gas = ax_conv[0,1]
                ax_conv_elec = ax_conv[1,0]
                ax_conv_heat = ax_conv[1,1]
                ax_conv_het.set_title('Heterogeneous network')
                ax_conv_gas.set_title('Gas network')
                ax_conv_elec.set_title('Electrical network')
                ax_conv_heat.set_title('Heat network')
                for key, iters in iters_gas.items():
                    ax_conv_gas.semilogy(errors_sc.get(key),ls=linestyles_gas.get(key),color='tab:green',marker='.',label=key)
                ax_conv_gas.legend()
                for key, iters in iters_elec.items():
                    ax_conv_elec.semilogy(errors_sc.get(key),marker=markers_sc_nodes.get(key),color='tab:red')
                for key, iters in iters_heat.items():
                    ax_conv_heat.semilogy(errors_sc.get(key),marker=markers_heat.get(key),ls='--',color='tab:blue',label=key)
                ax_conv_heat.legend(handles=legend_handles_heat)
                for form_gas in forms_gas:
                    for hydr_eq_gas in hydr_eqs:
                        if not (form_gas == 'nodal' and hydr_eq_gas == 'fb'): #those cannot be combined
                            key_gas = '{} {}'.format(form_gas,hydr_eq_gas)
                            for form_heat in forms_heat:
                                for heat_load in ['outflow','delta']:
                                    # run load flow for MES
                                    formulation={'gas':form_gas,'elec':'complex_power','heat':form_heat,'het':None}
                                    key_heat = '{} {}'.format(form_heat,heat_load)
                                    iters_max_sc = np.max([iters_gas.get(key_gas),iters_elec.get('elec 2 heat 2'),iters_heat.get(key_heat)])
                                    for coupling_type in coupling_types:
                                        with warnings.catch_warnings():
                                            warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                                            warnings.filterwarnings("ignore", "Only a",UserWarning)
                                            try:
                                                het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = run_load_flow(pgbase,qbase,Vbase,Sbase,phbase,mbase,Tbase,phibase,Egbase,deltabase=deltabase,p1g=p1g,q1=q1,q3=q3,V1=V1,delta1=delta1,delta2=delta2,V2=V2,P1=P1,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi1=phi1,phi2=phi2,phi3=phi3,To=To,dTc=dT,q0_init=q12_init,q1_init=q23_init,p2g_init=p2g_init,p3g_init=p3g_init,V2_init=V2_init,V3_init=V3_init,delta1_init=delta1_init,delta2_init=delta2_init,delta3_init=delta3_init,m0_init=m0_init,m1_init=m1_init,p2h_init=p2h_init,p3h_init=p3h_init,Ts1_init=Ts1_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,qc_init=qc_init,Pc_init=Pc_init,Qc_init=Qc_init,mc_init=mc_init,Toc_init=Toc_init,phic_init=phic_init,hydr_eq_gas=hydr_eq_gas,node_set_elec=node_set_elec,node_set_heat=node_set_heat,heat_load=heat_load,coupling_type=coupling_type,c_GG=c_GG,d_GG=d_GG,coupling_point=coupling_point,node_set=node_set,formulation=formulation,tol=tol,max_iter=max_iter)
                                            except:
                                                continue
                                            key = '{} {} {} {} {} {} {}'.format(form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set,coupling_type)
                                            x_sol_dict[key] = x_sol
                                            errors_dict[key] = err_vec
                                            iters_dict[key] = iters
                                            print('final error = {:.3e}'.format(err_vec[-1]))
                                            print('p gas= {} bar'.format(p_g_vec/bar))
                                            print('q = {}'.format(q_vec))
                                            print('q inj = {}'.format(q_inj))
                                            print('delta = {}'.format(delta_vec))
                                            print('|V| = {}'.format(V_mag_vec))
                                            print('P edge = {} MW'.format(P_edge/MW))
                                            print('Q edge = {} MW'.format(Q_edge/MW))
                                            print('S inj = {} MW'.format(S_inj/MW))
                                            print('p heat= {} bar'.format(p_h_vec/bar))
                                            print('m = {}'.format(m_vec))
                                            print('Ts = {}'.format(Ts_vec))
                                            print('Tr = {}'.format(Tr_vec))
                                            print('m hl = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                            print('Ts hl = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                            print('Tr hl = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                            print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                            print('Ts hl c = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                            print('Tr hl c = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                            print('dphi hl c = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                            print('m hl c = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                            ls = linestyles_gas.get(key_gas)
                                            marker = markers_heat.get(key_heat)
                                            ax_conv_het.semilogy(err_vec,marker=marker,ls=ls,color=colors.get(coupling_type),label='{}, {}, {}, {}'.format(form_gas,hydr_eq_gas,form_heat,heat_load))
                                            max_iters_used = np.max([max_iters_used,iters_max_sc,iters])
                # tweak convergence plots
                for ax_rows in ax_conv:
                    for ax in ax_rows:
                        ax.set_xlabel(r'Iteration $k$')
                        ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
                        ax.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
                        ax.grid(which='major',color='k', linestyle='--', alpha=.2)
                        ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
                        xmin = 0
                        xmax = max_iters_used
                        xticks = range(xmin,xmax+1,2) # make sure the xticks are integers
                        ax.set_xlim(xmin=xmin,xmax=xmax+1)
                        ax.set_xticks(xticks)


    return x_sol_dict, errors_dict, iters_dict, x_sol_sc, errors_sc, iters_sc, tol

def compare_conv_models(coupling_point=1,node_set=1,coupling_types = ['CHP','EH','GB+GG','GB+GG_vp'],show_plots=True):
    """Compare the convergence of load flow for one topology, but using different physical parameters for the edges, different coupling models, and different formulations of the system of equation in the single-carrier part.

    Parameters
    ----------
    coupling_point : int, optional
        Node at which is coupled. Options are 1 or 2. Default is 1
    node_set : int, optional
        Node set of the MES used for load flow. Default is 1
    show_plots : bool, optional
        When True, convergence plots are made. Default is False
    """
    x_sol_dict = dict()
    errors_dict = dict()
    iters_dict = dict()
    x_sol_sc = dict()
    errors_sc = dict()
    iters_sc = dict()

    fig_gg_vp_high = plt.figure('GG_vp_high_pres_mes_node{}_set{}'.format(coupling_point,node_set)) # figure to check non-linearity of gas-fired generator with valve-point effect
    ax_gg_vp_high = fig_gg_vp_high.gca()
    fig_gg_vp_low = plt.figure('GG_vp_low_pres_mes_node{}_set{}'.format(coupling_point,node_set)) # figure to check non-linearity of gas-fired generator with valve-point effect
    ax_gg_vp_low = fig_gg_vp_low.gca()
    P_vp_min= .5 #fraction of Pc_sol used as lower xlimit
    P_vp_max = 2 #fraction of Pc_sol used as lower xlimit

    if show_plots:
        # make figures to plot convergence
        fig_conv_high_het, ax_conv_high = plt.subplots(2, 2, num='Convergence plot networks, high pressure, coupling point {}, node set {}'.format(coupling_point,node_set))
        ax_conv_high_het = ax_conv_high[0,0]
        ax_conv_high_gas = ax_conv_high[0,1]
        ax_conv_high_elec = ax_conv_high[1,0]
        ax_conv_high_heat = ax_conv_high[1,1]
        ax_conv_high_het.set_title('Heterogeneous network')
        ax_conv_high_gas.set_title('Gas network')
        ax_conv_high_elec.set_title('Electrical network')
        ax_conv_high_heat.set_title('Heat network')
        iters_high_gas = dict()
        iters_high_elec = dict()
        iters_high_heat = dict()

        fig_conv_low_het, ax_conv_low = plt.subplots(2, 2, num='Convergence plot networks, low pressure, coupling point {}, node set {}'.format(coupling_point,node_set))
        ax_conv_low_het = ax_conv_low[0,0]
        ax_conv_low_gas = ax_conv_low[0,1]
        ax_conv_low_elec = ax_conv_low[1,0]
        ax_conv_low_heat = ax_conv_low[1,1]
        ax_conv_low_het.set_title('Heterogeneous network')
        ax_conv_low_gas.set_title('Gas network')
        ax_conv_low_elec.set_title('Electrical network')
        ax_conv_low_heat.set_title('Heat network')
        iters_low_gas = dict()
        iters_low_elec = dict()
        iters_low_heat = dict()

    max_iters_used = 0
    max_iters_used_gas = 0
    max_iters_used_elec = 0
    max_iters_used_heat = 0

    # solver information
    tol = 1e-6
    max_iter = 20
    forms_gas = ['full','nodal']
    forms_heat = ['standard','half_link_flow']
    hydr_eqs = ['fa','fb']

    # 'high pressure' network
    # physical parameters
    Dh_high = .15 #[m]
    Dg_high = 10*cm #[m]
    lam_high = 0.2 #[W/(mK)]
    c_GG_high = .01*Ebs
    d_GG_high = c_GG_high/10

    # boundary conditions (when needed)
    p1g = 50*bar #[Pa]
    q3 = 1 #[kg/s]
    q1 = -1.05*q3 #[kg/s]
    V1 = 50*kV #[V]
    delta1 = 0. #[rad]
    delta2 = 0. #[rad]
    V2 = 50*kV #[V]
    P2 = -1*MW #[W]
    P3 = 1.5*-P2 #[W]
    P1 = .9*-P3 #[W]
    Q3 = P3 #[Var]
    Ts1 = 100. #[C]
    p1h = 9*bar
    To2 = .9*Ts1
    To3 = 50. #[C]
    phi2 = -1*MW #[W]
    phi3 = 1.5*-phi2 #[W]
    phi1 = -.9*phi3 #[W]
    To = 100. #[C]
    dT = 50.

    # initial guesses (when needed)
    p2g_init = .99*p1g#30*bar #[Pa]
    p3g_init = .98*p1g#28*bar #[Pa]
    q12_init = q3 #[kg/s]
    q23_init= q3 #[kg/s]
    V2_init = 50*kV #[V]
    V3_init = 50*kV #[V]
    delta1_init = 0. #[rad]
    delta2_init = 0. #[rad]
    delta3_init = 0. #[rad]
    m0_init=1 #[kg/s]
    m1_init=1 #[kg/s]
    p2h_init=8*bar #[Pa]
    p3h_init=6*bar #[Pa]
    Ts1_init=100. #[C]
    Ts2_init=100. #[C]
    Ts3_init=100. #[C]
    Tr1_init=50. #[C]
    Tr2_init=50. #[C]
    Tr3_init=50. #[C]
    qc_init = q3 #[kg/s]
    Pc_init = P3 #[W]
    Qc_init = Q3 #[W]
    mc_init = 1 #[kg/s]
    phic_init = phi3 #[kg/s]
    Toc_init = Ts1 #[C]

    # scaling
    pgbase = bar
    qbase = 1.
    deltabase = 1.
    Vbase = 50*kV
    Sbase = MW
    phbase = bar
    mbase = 1.
    Tbase = 100.
    phibase = MW
    Egbase = phibase

    # load flow
    # run load flow for SC
    for form_gas in forms_gas:
        for hydr_eq_gas in hydr_eqs:
            if not (form_gas == 'nodal' and hydr_eq_gas == 'fb'): #those cannot be combined
                # run load flow for single carrier networks
                gas_net_single,x_sol_gas,iters_gas,err_vec_gas,p_sol_single,q_sol_single,q_inj_single = GasNet.run_load_flow(pgbase,qbase,p1=p1g,q3=q3,q0=q12_init,q1=q23_init,p2=p2g_init,p3=p3g_init,D=Dg_high,formulation=form_gas,hydr_eq=hydr_eq_gas,tol=tol,max_iter=max_iter)
                key = '{} {}'.format(form_gas,hydr_eq_gas)
                ls = linestyles_gas.get(key)
                x_sol_sc[key+' high'] = x_sol_gas
                errors_sc[key+' high'] = err_vec_gas
                iters_sc[key+' high'] = iters_gas
                if show_plots:
                    ax_conv_high_gas.semilogy(err_vec_gas,ls=ls,color='tab:green',marker='.',label='{}, {}'.format(form_gas,hydr_eq_gas))
                    max_iters_used_gas = max(max_iters_used_gas,iters_gas)
                    iters_high_gas[key] = iters_gas
                print('\nLoad flow solution for gas network using {} and {}, final error {:.3e}'.format(form_gas,hydr_eq_gas,err_vec_gas[-1]))
                print('p = {} bar'.format(p_sol_single/bar))
                print('q = {}'.format(q_sol_single))
                print('q inj = {}'.format(q_inj_single))
    if show_plots:
        ax_conv_high_gas.legend()
    for node_set_elec in [1,2]:
        elec_net_single,x_sol_elec,iters_elec,err_vec_elec,delta_sol_single,V_sol_single,S_inj_single,P_edge_single,Q_edge_single = ElecNet.run_load_flow(Vbase,Sbase,deltabase=deltabase,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,V2_init=V2_init,V3_init=V3_init,delta2_init=delta2_init,delta3_init=delta3_init,node_set=node_set_elec,tol=tol,max_iter=max_iter)
        key = 'elec {} heat 1'.format(node_set_elec)
        marker = markers_sc_nodes.get(key)
        x_sol_sc[key+' high'] = x_sol_elec
        errors_sc[key+' high'] = err_vec_elec
        iters_sc[key+' high'] = iters_elec
        if show_plots:
            ax_conv_high_elec.semilogy(err_vec_elec,marker=marker,color='tab:red',label='node set {}'.format(node_set_elec))
            max_iters_used_elec = max(max_iters_used_elec,iters_elec)
            iters_high_elec[key] = iters_elec
        print('\nLoad flow solution for electrical network with node set {}, final error {:.3e}'.format(node_set_elec,err_vec_elec[-1]))
        print('delta = {}'.format(delta_sol_single))
        print('|V| = {}'.format(V_sol_single))
        print('P edge = {} MW'.format(P_edge_single/MW))
        print('Q edge = {} MW'.format(Q_edge_single/MW))
        print('S inj = {} MW'.format(S_inj_single/MW))
    if show_plots:
        ax_conv_high_elec.legend()
    for form_heat in forms_heat:
        for node_set_heat in [1,2]:
            for heat_load in ['outflow','delta']:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                    heat_net_single,x_sol_heat,iters_heat,err_vec_heat,m_vec_single,p_vec_single,Ts_vec_single,Tr_vec_single,m_hl_vec_single,phi_hl_vec_single,Ts_hl_vec_single,Tr_hl_vec_single = HeatNet.run_load_flow(phbase,mbase,Tbase,phibase,Ts1=Ts1,p1=p1h,To2=To2,To3=To3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2h_init,p3_init=p3h_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,dT2=dT,dT3=dT,D=Dh_high,lam=lam_high,formulation=form_heat,node_set=node_set_heat,source_node=heat_load,sink_node=heat_load,tol=tol,max_iter=max_iter)
                    key = '{} {}'.format(form_heat,heat_load)
                    marker = markers_heat.get(key)
                    x_sol_sc[key+' high'] = x_sol_heat
                    errors_sc[key+' high'] = err_vec_heat
                    iters_sc[key+' high'] = iters_heat
                    if node_set_heat == 1:
                        ls = '-'
                    else:
                        ls = '--'
                    if show_plots:
                        ax_conv_high_heat.semilogy(err_vec_heat,marker=marker,ls=ls,color='tab:blue',label='{}, node set {}, load {}'.format(form_heat,node_set_heat,heat_load))
                        max_iters_used_heat = max(max_iters_used_heat,iters_heat)
                        if node_set_heat == 2:
                            iters_high_heat[key] = iters_heat
                    print('\nLoad flow solution for heat network, with node set {}, using {} and {}, final error {:.3e}'.format(node_set_heat,form_heat,heat_load,err_vec_heat[-1]))
                    print('p = {} bar'.format(p_vec_single/bar))
                    print('m = {}'.format(m_vec_single))
                    print('Ts = {}'.format(Ts_vec_single))
                    print('Tr = {}'.format(Tr_vec_single))
                    print('m hl = {}'.format(m_hl_vec_single))
                    print('Ts hl = {}'.format(Ts_hl_vec_single))
                    print('Tr hl = {}'.format(Tr_hl_vec_single))
                    print('phi hl = {}'.format(phi_hl_vec_single))
    if show_plots:
        ax_conv_high_heat.legend(handles=legend_handles_heat)

    node_set_heat = 2
    node_set_elec = 2
    for form_gas in forms_gas:
        for hydr_eq_gas in hydr_eqs:
            if not (form_gas == 'nodal' and hydr_eq_gas == 'fb'): #those cannot be combined
                ls = linestyles_gas.get('{} {}'.format(form_gas,hydr_eq_gas))
                for form_heat in forms_heat:
                    for heat_load in ['outflow','delta']:
                        # run load flow for MES
                        formulation={'gas':form_gas,'elec':'complex_power','heat':form_heat,'het':None}
                        marker = markers_heat.get('{} {}'.format(form_heat,heat_load))
                        for coupling_type in coupling_types:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                                warnings.filterwarnings("ignore", "Only a",UserWarning)
                                try:
                                    het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = run_load_flow(pgbase,qbase,Vbase,Sbase,phbase,mbase,Tbase,phibase,Egbase,deltabase=deltabase,p1g=p1g,q1=q1,q3=q3,V1=V1,delta1=delta1,delta2=delta2,V2=V2,P1=P1,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi1=phi1,phi2=phi2,phi3=phi3,To=To,dTc=dT,q0_init=q12_init,q1_init=q23_init,p2g_init=p2g_init,p3g_init=p3g_init,V2_init=V2_init,V3_init=V3_init,delta1_init=delta1_init,delta2_init=delta2_init,delta3_init=delta3_init,m0_init=m0_init,m1_init=m1_init,p2h_init=p2h_init,p3h_init=p3h_init,Ts1_init=Ts1_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,qc_init=qc_init,Pc_init=Pc_init,Qc_init=Qc_init,mc_init=mc_init,Toc_init=Toc_init,phic_init=phic_init,Dg=Dg_high,hydr_eq_gas=hydr_eq_gas,Dh=Dh_high,lam=lam_high,node_set_elec=node_set_elec,node_set_heat=node_set_heat,heat_load=heat_load,coupling_type=coupling_type,c_GG=c_GG_high,d_GG=d_GG_high,coupling_point=coupling_point,node_set=node_set,formulation=formulation,tol=tol,max_iter=max_iter)
                                except:
                                    continue
                                key = 'high {} {} {} {} {} {} {}'.format(form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set,coupling_type)
                                x_sol_dict[key] = x_sol
                                errors_dict[key] = err_vec
                                iters_dict[key] = iters
                                print('final error = {:.3e}'.format(err_vec[-1]))
                                print('p gas= {} bar'.format(p_g_vec/bar))
                                print('q = {}'.format(q_vec))
                                print('q inj = {}'.format(q_inj))
                                print('delta = {}'.format(delta_vec))
                                print('|V| = {}'.format(V_mag_vec))
                                print('P edge = {} MW'.format(P_edge/MW))
                                print('Q edge = {} MW'.format(Q_edge/MW))
                                print('S inj = {} MW'.format(S_inj/MW))
                                print('p heat= {} bar'.format(p_h_vec/bar))
                                print('m = {}'.format(m_vec))
                                print('Ts = {}'.format(Ts_vec))
                                print('Tr = {}'.format(Tr_vec))
                                print('m hl = {}'.format(m_hl_vec))
                                print('m hl = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('Ts hl = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('Tr hl = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('Ts hl c = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                print('Tr hl c = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                print('dphi hl c = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                print('m hl c = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                for cn in het_net.get_nodes(unit_types=['ge_gas_fired_gen_valve_point']): #plot the coupling function around solution, to see how non-linear it is
                                    if err_vec[-1] < tol: # only if converged
                                        if not ax_gg_vp_high.lines: # Only depends on Pc_sol, which is only different for different node sets (I think).
                                            Pc_sol = Pc_vec[0]
                                            Eg_sol = qc_vec[1]*GHV
                                            Pc_range = np.linspace(P_vp_min*Pc_sol,P_vp_max*Pc_sol)
                                            lin_approx = Eg_sol+(Pc_range-Pc_sol)*cn.dEin_dEout(Pc_sol,None)
                                            Eg = cn.Ein_of_Eout(Pc_range,None)
                                            ax_gg_vp_high.plot(Pc_range,Eg,label='vp')
                                            ax_gg_vp_high.plot(Pc_range,lin_approx,':k',label='lin. Taylor')
                                            ax_gg_vp_high.plot([Pc_sol,Pc_sol],[min(Eg),max(Eg)],'k')
                                            ax_gg_vp_high.set_ylabel(r'GHV$\cdot q$ [W]')
                                            ax_gg_vp_high.set_xlabel(r'$P$ [W]')
                                            ax_gg_vp_high.annotate(r'$P_c$',xy=(Pc_sol,min(Eg)),ha='center',va='top')
                                            ax_gg_vp_high.set_title(r'$\eta=${:.4f}'.format(Pc_sol/Eg_sol))
                                            ax_gg_vp_high.legend()
                                if show_plots:
                                    ax_conv_high_het.semilogy(err_vec,marker=marker,ls=ls,color=colors.get(coupling_type),label='{}, {}, {}, {}'.format(form_gas,hydr_eq_gas,form_heat,heat_load))
                                    max_iters_used = max(max_iters_used,iters)

    # print the sum of the sc iterations as vertical lines in the mes convergence
    if show_plots:
        ylim_conv_high_net = ax_conv_high_het.get_ylim()
        key_elec = 'elec 1 heat 1'
        for key_gas,iters_gas in iters_high_gas.items():
            for key_heat, iters_heat in iters_high_heat.items():
                iters_max_sc = np.max([iters_gas,iters_high_elec.get(key_elec),iters_heat])
                ls = linestyles_gas.get(key_gas)
                marker = markers_heat.get(key_heat)
                ax_conv_high_het.semilogy([iters_max_sc,iters_max_sc],[ylim_conv_high_net[0],ylim_conv_high_net[1]],marker=marker,ls=ls,color='k',alpha=.5)

    # 'low pressure' network
    # physical parameters
    c_GG_low = 1e-5*Ebs
    d_GG_low = c_GG_low/10
    L12_low = 400 #[m]
    L23_low = 500 #[m]
    Dh_low = 15*cm #[m]
    Dg_low = 1*cm #[m]
    lam_low = 0.002 #[W/(mK)]
    De_low = 1*cm #[m]
    link_type_g_low = 'pipe_low_pres_pole'
    link_type_e_low = 'short_line'
    link_type_h_low = 'standard_pipe_low_pres_pole'

    # boundary conditions (when needed)
    p1g = 50*mbar #[Pa]
    q3 = 5.e-5 #[kg/s]
    q1 = -1.05*q3 #[kg/s]
    V1 = 10*kV #[V]
    delta1 = 0. #[rad]
    delta2 = 0. #[rad]
    V2 = V1 #[V]
    P2 = -1*kW #[W]
    P3 = 1.5*-P2 #[W]
    P1 = .9*-P3 #[W]
    Q3 = P3 #[Var]
    Ts1 = 100. #[C]
    p1h = 9*mbar
    To2 = .9*Ts1
    To3 = 50. #[C]
    phi2 = -1*kW #[W]
    phi3 = 6*-phi2 #[W]
    phi1 = -.9*phi3 #[W]
    To = 100. #[C]
    dT3 = 40.
    dT2 = 50.
    if coupling_point == 1:
        dTc = dT2
    else:
        dTc = 55.

    # initial guesses (when needed)
    p2g_init = .99*p1g#49*mbar #[Pa]
    p3g_init = .98*p1g#48*mbar #[Pa]
    q12_init = q3 #[kg/s]
    q23_init= q3 #[kg/s]
    V2_init = V1 #[V]
    V3_init = V1 #[V]
    delta1_init = 0. #[rad]
    delta2_init = 0. #[rad]
    delta3_init = 0. #[rad]
    m0_init=1e-2 #[kg/s]
    m1_init=1e-2 #[kg/s]
    p2h_init=8*mbar #[Pa]
    p3h_init=6*mbar #[Pa]
    Ts1_init=100. #[C]
    Ts2_init=100. #[C]
    Ts3_init=100. #[C]
    Tr1_init=50. #[C]
    Tr2_init=50. #[C]
    Tr3_init=50. #[C]
    qc_init = q3 #[kg/s]
    Pc_init = P3 #[W]
    Qc_init = Q3 #[W]
    mc_init = 1e-2 #[kg/s]
    phic_init = phi3 #[kg/s]
    Toc_init = Ts1 #[C]

    # scaling
    pgbase = mbar
    qbase = q3
    deltabase = 1.
    Vbase = V1
    Sbase = 1.*kW
    phbase = mbar
    mbase = 1.e-2
    Tbase = 100.
    phibase = 1.*kW
    Egbase = phibase

    # load flow
    # run load flow for SC
    for form_gas in forms_gas:
        for hydr_eq_gas in hydr_eqs:
            if not (form_gas == 'nodal' and hydr_eq_gas == 'fb'): #those cannot be combined
                # run load flow for single carrier networks
                gas_net_single,x_sol_gas,iters_gas,err_vec_gas,p_sol_single,q_sol_single,q_inj_single = GasNet.run_load_flow(pgbase,qbase,p1=p1g,q3=q3,q0=q12_init,q1=q23_init,p2=p2g_init,p3=p3g_init,L12=L12_low,L23=L23_low,D=Dg_low,link_type=link_type_g_low,formulation=form_gas,hydr_eq=hydr_eq_gas,tol=tol,max_iter=max_iter)
                key = '{} {}'.format(form_gas,hydr_eq_gas)
                ls = linestyles_gas.get(key)
                x_sol_sc[key+' low'] = x_sol_gas
                errors_sc[key+' low'] = err_vec_gas
                iters_sc[key+' low'] = iters_gas
                if show_plots:
                    ax_conv_low_gas.semilogy(err_vec_gas,ls=ls,color='tab:green',marker='.',label='{}, {}'.format(form_gas,hydr_eq_gas))
                    max_iters_used_gas = max(max_iters_used_gas,iters_gas)
                    iters_low_gas[key] = iters_gas
                print('\nLoad flow solution for gas network using {} and {}, final error {:.3e}'.format(form_gas,hydr_eq_gas,err_vec_gas[-1]))
                print('p = {} mbar'.format(p_sol_single/mbar))
                print('q = {}'.format(q_sol_single))
                print('q inj = {}'.format(q_inj_single))
    if show_plots:
        ax_conv_low_gas.legend()
    for node_set_elec in [1,2]:
        elec_net_single,x_sol_elec,iters_elec,err_vec_elec,delta_sol_single,V_sol_single,S_inj_single,P_edge_single,Q_edge_single = ElecNet.run_load_flow(Vbase,Sbase,deltabase=deltabase,V1=V1,delta1=delta1,V2=V2,P2=P2,P3=P3,Q3=Q3,V2_init=V2_init,V3_init=V3_init,delta2_init=delta2_init,delta3_init=delta3_init,L12=L12_low,L23=L23_low,D=De_low,link_type=link_type_e_low,node_set=node_set_elec,tol=tol,max_iter=max_iter)
        key = 'elec {} heat 1'.format(node_set_elec)
        marker = markers_sc_nodes.get(key)
        x_sol_sc[key+' low'] = x_sol_elec
        errors_sc[key+' low'] = err_vec_elec
        iters_sc[key+' low'] = iters_elec
        if show_plots:
            ax_conv_low_elec.semilogy(err_vec_elec,marker=marker,color='tab:red',label='node set {}'.format(node_set_elec))
            max_iters_used_elec = max(max_iters_used_elec,iters_elec)
            iters_low_elec[key] = iters_elec
        print('\nLoad flow solution for electrical network with node set {}, final error {:.3e}'.format(node_set_elec,err_vec_elec[-1]))
        print('delta = {}'.format(delta_sol_single))
        print('|V| = {}'.format(V_sol_single))
        print('P edge = {}'.format(P_edge_single))
        print('Q edge = {}'.format(Q_edge_single))
        print('S inj = {}'.format(S_inj_single))
    if show_plots:
        ax_conv_low_elec.legend()
    for form_heat in forms_heat:
        for node_set_heat in [1,2]:
            for heat_load in ['outflow','delta']:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                    heat_net_single,x_sol_heat,iters_heat,err_vec_heat,m_vec_single,p_vec_single,Ts_vec_single,Tr_vec_single,m_hl_vec_single,phi_hl_vec_single,Ts_hl_vec_single,Tr_hl_vec_single = HeatNet.run_load_flow(phbase,mbase,Tbase,phibase,Ts1=Ts1,p1=p1h,To2=To2,To3=To3,phi2=phi2,phi3=phi3,m0_init=m0_init,m1_init=m1_init,p2_init=p2h_init,p3_init=p3h_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,dT2=dT2,dT3=dT3,L12=L12_low,L23=L23_low,D=Dh_low,lam=lam_low,link_type=link_type_h_low,formulation=form_heat,node_set=node_set_heat,source_node=heat_load,sink_node=heat_load,tol=tol,max_iter=max_iter)
                    key = '{} {}'.format(form_heat,heat_load)
                    marker = markers_heat.get(key)
                    x_sol_sc[key+' low'] = x_sol_heat
                    errors_sc[key+' low'] = err_vec_heat
                    iters_sc[key+' low'] = iters_heat
                    if node_set_heat == 1:
                        ls = '-'
                    else:
                        ls = '--'
                    if show_plots:
                        ax_conv_low_heat.semilogy(err_vec_heat,marker=marker,ls=ls,color='tab:blue',label='{}, node set {}, load {}'.format(form_heat,node_set_heat,heat_load))
                        max_iters_used_heat = max(max_iters_used_heat,iters_heat)
                        if node_set_heat == 2:
                            iters_low_heat[key] = iters_heat
                    print('\nLoad flow solution for heat network, with node set {}, using {} and {}, final error {:.3e}'.format(node_set_heat,form_heat,heat_load,err_vec_heat[-1]))
                    print('p = {} mbar'.format(p_vec_single/mbar))
                    print('m = {}'.format(m_vec_single))
                    print('Ts = {}'.format(Ts_vec_single))
                    print('Tr = {}'.format(Tr_vec_single))
                    print('m hl = {}'.format(m_hl_vec_single))
                    print('Ts hl = {}'.format(Ts_hl_vec_single))
                    print('Tr hl = {}'.format(Tr_hl_vec_single))
                    print('phi hl = {}'.format(phi_hl_vec_single))
    if show_plots:
        ax_conv_low_heat.legend(handles=legend_handles_heat)

    for form_gas in forms_gas:
        for hydr_eq_gas in hydr_eqs:
            if not (form_gas == 'nodal' and hydr_eq_gas == 'fb'): #those cannot be combined
                ls = linestyles_gas.get('{} {}'.format(form_gas,hydr_eq_gas))
                for form_heat in forms_heat:
                    for heat_load in ['outflow','delta']:
                        # run load flow for MES
                        formulation={'gas':form_gas,'elec':'complex_power','heat':form_heat,'het':None}
                        marker = markers_heat.get('{} {}'.format(form_heat,heat_load))
                        for coupling_type in coupling_types:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                                warnings.filterwarnings("ignore", "Only a",UserWarning)
                                try:
                                    het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = run_load_flow(pgbase,qbase,Vbase,Sbase,phbase,mbase,Tbase,phibase,Egbase,deltabase=deltabase,p1g=p1g,q1=q1,q3=q3,V1=V1,delta1=delta1,delta2=delta2,V2=V2,P1=P1,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi1=phi1,phi2=phi2,phi3=phi3,To=To,dTc=dTc,q0_init=q12_init,q1_init=q23_init,p2g_init=p2g_init,p3g_init=p3g_init,V2_init=V2_init,V3_init=V3_init,delta1_init=delta1_init,delta2_init=delta2_init,delta3_init=delta3_init,m0_init=m0_init,m1_init=m1_init,p2h_init=p2h_init,p3h_init=p3h_init,Ts1_init=Ts1_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,qc_init=qc_init,Pc_init=Pc_init,Qc_init=Qc_init,mc_init=mc_init,Toc_init=Toc_init,phic_init=phic_init,L12=L12_low,L23=L23_low,Dg=Dg_low,link_type_g=link_type_g_low,hydr_eq_gas=hydr_eq_gas,De=De_low,link_type_e=link_type_e_low,Dh=Dh_low,lam=lam_low,link_type_h=link_type_h_low,node_set_elec=node_set_elec,node_set_heat=node_set_heat,heat_load=heat_load,coupling_type=coupling_type,c_GG=c_GG_low,d_GG=d_GG_low,coupling_point=coupling_point,node_set=node_set,formulation=formulation,tol=tol,max_iter=max_iter)
                                except:
                                    continue
                                key = 'low {} {} {} {} {} {} {}'.format(form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set,coupling_type)
                                x_sol_dict[key] = x_sol
                                errors_dict[key] = err_vec
                                iters_dict[key] = iters
                                print('final error = {:.3e}'.format(err_vec[-1]))
                                print('p gas= {} mbar'.format(p_g_vec/mbar))
                                print('q = {}'.format(q_vec))
                                print('q inj = {}'.format(q_inj))
                                print('delta = {}'.format(delta_vec))
                                print('|V| = {}'.format(V_mag_vec))
                                print('P edge = {} kW'.format(P_edge/kW))
                                print('Q edge = {} kW'.format(Q_edge/kW))
                                print('S inj = {} kW'.format(S_inj/kW))
                                print('p heat= {} mbar'.format(p_h_vec/mbar))
                                print('m = {}'.format(m_vec))
                                print('Ts = {}'.format(Ts_vec))
                                print('Tr = {}'.format(Tr_vec))
                                print('m hl = {}'.format(m_hl_vec))
                                print('m hl = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('Ts hl = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('Tr hl = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('dphi hl = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeatNode) for hl in node.get_half_links()]))
                                print('Ts hl c = {} C'.format([hl.get_Ts() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                print('Tr hl c = {} C'.format([hl.get_Tr() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                print('dphi hl c = {} MW'.format([hl.get_dphi()/MW for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                print('m hl c = {} kg/s'.format([hl.get_m() for node in het_net.get_nodes() if isinstance(node,HeterogeneousNode) for hl in node.get_half_links()]))
                                for cn in het_net.get_nodes(unit_types=['ge_gas_fired_gen_valve_point']): #plot the coupling function around solution, to see how non-linear it is
                                    if err_vec[-1] < tol: # only if converged
                                        if not ax_gg_vp_low.lines: # Only depends on Pc_sol, which is only different for different node sets (I think).
                                            Pc_sol = Pc_vec[0]
                                            Eg_sol = cn.Ein_of_Eout(Pc_sol,None)# qc_vec[1]*GHV
                                            Pc_range = np.linspace(P_vp_min*Pc_sol,P_vp_max*Pc_sol)
                                            lin_approx = Eg_sol+(Pc_range-Pc_sol)*cn.dEin_dEout(Pc_sol,None)
                                            Eg = cn.Ein_of_Eout(Pc_range,None)
                                            ax_gg_vp_low.plot(Pc_range,Eg,label='vp')
                                            ax_gg_vp_low.plot(Pc_range,lin_approx,':k',label='lin. Taylor')
                                            ax_gg_vp_low.plot([Pc_sol,Pc_sol],[min(Eg),max(Eg)],'k')
                                            ax_gg_vp_low.set_ylabel(r'GHV$\cdot q$ [W]')
                                            ax_gg_vp_low.set_xlabel(r'$P$ [W]')
                                            ax_gg_vp_low.annotate(r'$P_c$',xy=(Pc_sol,min(Eg)),ha='center',va='top')
                                            ax_gg_vp_low.set_title(r'$\eta=${:.4f}'.format(Pc_sol/Eg_sol))
                                            ax_gg_vp_low.legend()
                                if show_plots:
                                    ax_conv_low_het.semilogy(err_vec,marker=marker,ls=ls,color=colors.get(coupling_type),label='{}, {}, {}, {}'.format(form_gas,hydr_eq_gas,form_heat,heat_load))
                                    max_iters_used = max(max_iters_used,iters)

    # print the maximum of the sc iterations as vertical lines in the mes convergence
    if show_plots:
        ylim_conv_low_net = ax_conv_low_het.get_ylim()
        key_elec = 'elec 1 heat 1'
        for key_gas,iters_gas in iters_low_gas.items():
            for key_heat, iters_heat in iters_low_heat.items():
                iters_max_sc = np.max([iters_gas,iters_low_elec.get(key_elec),iters_heat])
                ls = linestyles_gas.get(key_gas)
                marker = markers_heat.get(key_heat)
                ax_conv_low_het.semilogy([iters_max_sc,iters_max_sc],[ylim_conv_low_net[0],ylim_conv_low_net[1]],marker=marker,ls=ls,color='k',alpha=.5)

        # tweak the convergence plots
        max_iters_used = max(max_iters_used,max_iters_used_gas,max_iters_used_elec,max_iters_used_heat)
        for ax_conv in [ax_conv_high, ax_conv_low]:
            for ax_rows in ax_conv:
                for ax in ax_rows:
                    ax.set_xlabel(r'Iteration $k$')
                    ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
                    ax.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
                    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
                    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
                    ax.set_xlim(xmin=0,xmax=max_iters_used+1)

    return x_sol_dict, errors_dict, iters_dict, x_sol_sc, errors_sc, iters_sc, tol

def plot_conv_mes_models(x_sol, errors, iters, tol):
    """Create convergence plots for the dictionaries created by compare_conv_models"""
    max_iters_used = 0
    couplings_used = list()
    for key_mes, iter_mes in iters.items():
        max_iters_used = max(max_iters_used,iter_mes)
    for key_mes, err_vec in errors.items():
        key_list = key_mes.split(' ')
        key_gas = '{} {}'.format(key_list[1],key_list[2])
        key_heat = '{} {}'.format(key_list[3],key_list[4])
        coupling_point = int(key_list[5])
        node_set = int(key_list[6])
        coupling_type = key_list[7]
        fig = plt.figure('conv_{}_pres_mes_node{}_set{}'.format(key_list[0],coupling_point,node_set))
        ax = fig.gca()
        ls = linestyles_gas.get(key_gas)
        color = colors.get(coupling_type)
        marker = markers_heat.get(key_heat)
        if (key_gas == 'nodal fa' or key_gas == 'full fb') and (key_heat == 'standard outflow' or key_heat == 'half_link_flow delta'):
            ax.semilogy(err_vec,marker=marker,ls=ls,color=color)
        ax.set_xlabel(r'Iteration $k$')
        ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
        ax.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
        ax.grid(which='major',color='k', linestyle='--', alpha=.2)
        ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
        xmin = 0
        xmax = max_iters_used
        xticks = range(xmin,xmax+1,2) # make sure the xticks are integers
        ax.set_xlim(xmin=xmin,xmax=xmax+1)
        ax.set_xticks(xticks)
    return ax

def plot_conv_mes_node_forms(x_sol, errors, iters, tol):
    """Create convergence plots for the dictionaries created by compare_conv_models"""
    max_iters_used = 0
    couplings_used = list()
    for key_mes, iter_mes in iters.items():
        max_iters_used = max(max_iters_used,iter_mes)
    for key_mes, err_vec in errors.items():
        key_list = key_mes.split(' ')
        node_set_elec = int(key_list[0])
        node_set_heat = int(key_list[1])
        form_gas = key_list[2]
        hydr_eq = key_list[3]
        form_heat = key_list[4]
        heat_load = key_list[5]
        coupling_point = int(key_list[6])
        node_set = int(key_list[7])
        coupling_type = key_list[8]
        if node_set_elec == 2 and node_set_heat == 2:
            fig = plt.figure('conv_mes_node{}_{}_{}_{}_{}'.format(coupling_point,form_gas,hydr_eq,form_heat,heat_load))
            ax = fig.gca()
            ls = linestyles_node_sets.get(node_set)
            color = colors.get(coupling_type)
            marker = markers_sc_nodes.get('elec {} heat {}'.format(node_set_elec,node_set_heat))
            ax.semilogy(err_vec,marker=marker,ls=ls,color=color)
            ax.set_xlabel(r'Iteration $k$')
            ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
            ax.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
            ax.grid(which='major',color='k', linestyle='--', alpha=.2)
            ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
            xmin = 0
            xmax = max_iters_used
            xticks = range(xmin,xmax+1) # make sure the xticks are integers
            ax.set_xlim(xmin=xmin,xmax=xmax+1)
            ax.set_xticks(xticks)


def plots_pscc_2020_models(dir_path,save_fig=False):
    """Create plots for the paper for PSCC 2020"""
    coupling_types = ['GB+GG','GB+GG_vp']

    d = 10  # width of the window border in pixels
    # figure setting to save figure
    fig_width_pt = 252.0  #[pt] Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = 2*fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]
    pgf_with_latex = {
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": False,                # True: use LaTeX to write all text
        "font.family": "DejaVu Sans",#"serif",
        "font.serif": [],                   # blank entries should cause plots
        "font.monospace": [],               # to inherit fonts from the document
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.loc": "upper right",
        "legend.fontsize": 9,               # Make the legend/label fonts
        "xtick.labelsize": 10,               # a little smaller
        "ytick.labelsize": 10,
        "figure.figsize": fig_size,
        }
    mpl.rcParams.update(pgf_with_latex)

    x_sol = dict()
    errors = dict()
    iters = dict()
    x_sol_dict, errors_dict, iters_dict, x_sol_sc, errors_sc, iters_sc, tol = compare_conv_models(coupling_point=1,node_set=2,coupling_types=coupling_types,show_plots=False)
    x_sol.update(x_sol_dict)
    errors.update(errors_dict)
    iters.update(iters_dict)
    #x_sol_dict, errors_dict, iters_dict, _, _, _, _ = compare_conv_models(coupling_point=2,node_set=1,coupling_types=coupling_types,show_plots=False)
    #x_sol.update(x_sol_dict)
    #errors.update(errors_dict)
    #iters.update(iters_dict)
    x_sol_dict, errors_dict, iters_dict, _, _, _, _ = compare_conv_models(coupling_point=2,node_set=3,coupling_types=coupling_types,show_plots=False)
    x_sol.update(x_sol_dict)
    errors.update(errors_dict)
    iters.update(iters_dict)
    # plot convergence mes
    plot_conv_mes_models(x_sol, errors, iters, tol)

    # plot convergence sc
    fig_conv_sc_low = plt.figure('conv_low_pres_sc')
    ax_conv_sc_low = fig_conv_sc_low.gca()
    fig_conv_sc_high = plt.figure('conv_high_pres_sc')
    ax_conv_sc_high = fig_conv_sc_high.gca()
    max_iters_sc = 0
    for key_sc, iter_sc in iters_sc.items():
        if ' low' in key_sc:
            key_leg = ''.join(key_sc.rsplit(' low'))
            ax = ax_conv_sc_low
            print('\nSolution SC nets, low pressure:')
        else:
            key_leg = ''.join(key_sc.rsplit(' high'))
            ax = ax_conv_sc_high
            print('\nSolution SC nets, high pressure:')
        if 'nodal' in key_sc or 'full' in key_sc: #gas
            ls = linestyles_gas.get(key_leg)
            color = 'tab:green'
            marker = '.'
            print('Gas net with {}:'.format(key_leg))
        elif 'elec' in key_sc: #electrical
            if 'elec 2' in key_sc:
                ls = '-'
                color = 'tab:red'
                marker = '.'
                print('Electrical net with {}:'.format(key_leg))
            else:
                continue
        else: #heat
            ls = '-'
            color = 'tab:blue'
            marker = markers_heat.get(key_leg)
            print('Heat net with {}:'.format(key_leg))
        print('final error = {}'.format(errors_sc.get(key_sc)[-1]))
        print(x_sol_sc.get(key_sc))
        ax.semilogy(errors_sc.get(key_sc),marker=marker,ls=ls,color=color)
        max_iters_sc = max(max_iters_sc,iter_sc)
    xmin = 0
    xmax = max_iters_sc
    xticks = range(xmin,xmax+1,2) # make sure the xticks are integers
    ax_conv_sc_low.set_xlabel(r'Iteration $k$')
    ax_conv_sc_low.set_ylabel(r'Error $||D_F F(x^k)||_2$')
    ax_conv_sc_low.semilogy([0,max_iters_sc+1],[tol,tol],'k:',label='tolerance')
    ax_conv_sc_low.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_sc_low.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_sc_low.set_xlim(xmin=xmin,xmax=xmax+1)
    ax_conv_sc_low.set_xticks(xticks)
    ax_conv_sc_high.set_xlabel(r'Iteration $k$')
    ax_conv_sc_high.set_ylabel(r'Error $||D_F F(x^k)||_2$')
    ax_conv_sc_high.semilogy([0,max_iters_sc+1],[tol,tol],'k:',label='tolerance')
    ax_conv_sc_high.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax_conv_sc_high.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax_conv_sc_high.set_xlim(xmin=xmin,xmax=xmax+1)
    ax_conv_sc_high.set_xticks(xticks)

    if save_fig:
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            path_to_fig = os.path.join(dir_path,'Figures','MES3N_line')
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

def plots_pscc_2020_forms(dir_path,save_fig=False):
    """Create plots for the paper for PSCC 2020"""
    d = 10  # width of the window border in pixels
    # figure setting to save figure
    fig_width_pt = 252.0  #[pt] Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = 2*fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]
    pgf_with_latex = {
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": False,                # True: use LaTeX to write all text
        "font.family": "DejaVu Sans",#"serif",
        "font.serif": [],                   # blank entries should cause plots
        "font.monospace": [],               # to inherit fonts from the document
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.loc": "upper right",
        "legend.fontsize": 9,               # Make the legend/label fonts
        "xtick.labelsize": 10,               # a little smaller
        "ytick.labelsize": 10,
        "figure.figsize": fig_size,
        }
    mpl.rcParams.update(pgf_with_latex)

    x_sol = dict()
    errors = dict()
    iters = dict()
    x_sol_sc = dict()
    errors_sc = dict()
    iters_sc = dict()
    for hydr_eq_gas in ['fa','fb']:
        for form_gas in ['nodal','full']:
            for form_heat in ['standard','half_link_flow']:
                for heat_load in ['outflow','delta']:
                    if not (hydr_eq_gas == 'fb' and form_gas == 'nodal'):
                        x_sol_dict, errors_dict, iters_dict, x_sol_sc_dict, errors_sc_dict, iters_sc_dict, tol = compare_conv_node_sets(hydr_eq_gas=hydr_eq_gas,form_gas=form_gas,form_heat=form_heat,heat_load=heat_load)
                        x_sol.update(x_sol_dict)
                        errors.update(errors_dict)
                        iters.update(iters_dict)
                        x_sol_sc.update(x_sol_sc_dict)
                        errors_sc.update(errors_sc_dict)
                        iters_sc.update(iters_sc_dict)
    plot_conv_mes_node_forms(x_sol, errors, iters, tol)

    # plot convergence sc
    fig_conv_sc = plt.figure('conv_sc') # figure with all single-carriers in one plot
    ax_conv_sc = fig_conv_sc.gca()
    fig_conv_gas = plt.figure('conv_gas')
    fig_conv_elec = plt.figure('conv_elec')
    fig_conv_heat = plt.figure('conv_heat')
    max_iters_sc = 0
    for key_sc, iter_sc in iters_sc.items():
        if 'nodal' in key_sc or 'full' in key_sc: #gas
            ax = plt.figure('conv_gas').gca()
            ls = linestyles_sc.get('gas')
            color = colors_sc.get('gas')
            marker = markers_gas.get(key_sc)
            label = key_sc
        elif 'elec' in key_sc: #electrical
            if 'elec 2' in key_sc:
                ax = plt.figure('conv_elec').gca()
                ls = linestyles_sc.get('elec')
                color = colors_sc.get('elec')
                marker = markers_sc_nodes.get(key_sc+' heat 2')
                label = ''
            else:
                continue
        else: #heat
            key_list = key_sc.split(' ')
            form_heat = key_list[0]
            heat_load = key_list[1]
            node_set_heat = int(key_list[2])
            if node_set_heat == 2:
                ax = plt.figure('conv_heat').gca()
                ls = linestyles_sc.get('heat')
                color = colors_sc.get('heat')
                marker = markers_heat.get('{} {}'.format(form_heat,heat_load))
                label = labels_heat.get('{} {}'.format(form_heat,heat_load))
            else:
                continue
        ax.semilogy(errors_sc.get(key_sc),marker=marker,ls=ls,color=color,label=label)
        ax_conv_sc.semilogy(errors_sc.get(key_sc),marker=marker,ls=ls,color=color,label=label)
        max_iters_sc = max(max_iters_sc,iter_sc)
    xmin = 0
    xmax = max_iters_sc
    xticks = range(xmin,xmax+1) # make sure the xticks are integers
    for fig_num in ['conv_gas','conv_elec','conv_heat','conv_sc']:
        ax = plt.figure(fig_num).gca()
        ax.set_xlabel(r'Iteration $k$')
        ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
        ax.semilogy([0,max_iters_sc+1],[tol,tol],'k:',label='tolerance')
        ax.grid(which='major',color='k', linestyle='--', alpha=.2)
        ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
        ax.set_xlim(xmin=xmin,xmax=xmax+1)
        ax.set_xticks(xticks)
        if not fig_num == 'conv_sc':
            ax.legend()

    if save_fig:
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            path_to_fig = os.path.join(dir_path,'Figures','MES3N_line')
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

    # print the iteration results, in order to put them in the table in the paper
    print('\nNR iterations (CHP & GB+GG & GB+GG_vp & EH):')
    node_set_elec = 2
    node_set_heat = 2
    for coupling_point in [1,2]:
        for form_gas in ['nodal','full']:
            for hydr_eq_gas in ['fa','fb']:
                if not (hydr_eq_gas == 'fb' and form_gas == 'nodal'):
                    for form_heat in ['standard','half_link_flow']:
                        for heat_load in ['outflow','delta']:
                            for node_set in [1,2,3]:
                                if not (coupling_point == 1 and node_set == 3):
                                    for coupling_type in ['CHP','GB+GG','GB+GG_vp','EH']:
                                        key_CHP = '{} {} {} {} {} {} {} {} CHP'.format(node_set_elec,node_set_heat,form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set)
                                        key_GB_GG = '{} {} {} {} {} {} {} {} GB+GG'.format(node_set_elec,node_set_heat,form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set)
                                        key_GB_GG_vp = '{} {} {} {} {} {} {} {} GB+GG_vp'.format(node_set_elec,node_set_heat,form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set)
                                        key_EH = '{} {} {} {} {} {} {} {} EH'.format(node_set_elec,node_set_heat,form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set)
                                    print('c. point: {}, {} {}, {} {}, node set: {}, {} & {} & {} & {}'.format(coupling_point,form_gas,hydr_eq_gas,form_heat,heat_load,node_set,iters.get(key_CHP),iters.get(key_GB_GG),iters.get(key_GB_GG_vp),iters.get(key_EH)))

def perm_matr_x(het_net,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}):
    """Determines the permutation matrices P_x and P_F, when only reordering by putting the coupling variables with the single-carrier parts, and by putting the heat coupling equations (Fphi and FdT) with the heat part

    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The network for which the permutation matrix needs to be made.

    Returns
    -------
    P_x : scipy sparse matrix
        Permutation matrix for the vector of variables x
    P_F : scipy sparse matrix
        Permutation matrix for the vector of equations F
    """
    # Determine new indices for x
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks= het_net.get_x_entries(formulation=formulation)
    x_length = len(x_entries)
    Px_row = list(range(x_length)) # New indices
    Px_data = [1]*x_length # Permutation matrix is binary matrix
    Px_col = [ind for ind,el in enumerate(x_entries) if 'Gas' in type(el).__name__] + [ind for ind,el in enumerate(x_entries) if 'Elec' in type(el).__name__] + [ind for ind,el in enumerate(x_entries) if 'Heat' in type(el).__name__] # Original indices
    P_x = sps.csr_matrix((Px_data,(Px_row,Px_col)),shape=(x_length,x_length))

    # Determine new indices for F
    F_entries, Fg_entries, Fe_entries, known_P_nodes, known_Q_nodes, Fh_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_To_halflinks, Fc_entries, F_fc_nodes, F_fc_amount, F_phi_nodes, F_dT_nodes = het_net.get_F_entries(formulation=formulation)
    F_length = len(Fg_entries) + len(Fe_entries) + len(Fh_entries) + np.sum(F_fc_amount) + len(F_phi_nodes) + len(F_dT_nodes)
    PF_row = list(range(F_length)) # New indices
    PF_data = [1]*F_length # Permutation matrix is binary matrix
    PF_col = [ind for ind,el in enumerate(F_entries) if el in Fg_entries] + [ind for ind,el in enumerate(F_entries) if el in Fe_entries] + [ind for ind,el in enumerate(F_entries) if el in Fh_entries] # Original indices
    for ind_el,el in enumerate(F_phi_nodes + F_dT_nodes):
        PF_col.append(len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+np.sum(F_fc_amount)+ind_el)
    for ind_el,el in enumerate(F_fc_nodes):
        ind = ind_el+len(Fg_entries)+len(Fe_entries)+len(Fh_entries)
        if ind_el>0:
            ind += np.sum(F_fc_amount[0:ind_el])-ind_el # -ind_el, because index is already shifted by one with respect to previous element because ind_el has increased 1
        fc_len = F_fc_amount[ind_el]
        for fc_ind in range(fc_len):
            PF_col.append(ind + fc_ind)
    P_F = sps.csr_matrix((PF_data,(PF_row,PF_col)),shape=(F_length,F_length))
    return P_x, P_F

def perm_matr_xF(het_net,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}):
    """Determines the permutation matrices P_x and P_F, reordering x and all coupling equations

    Parameters
    ----------
    het_net : HeterogeneousNetwork
        The network for which the permutation matrix needs to be made.

    Returns
    -------
    P_x : scipy sparse matrix
        Permutation matrix for the vector of variables x
    P_F : scipy sparse matrix
        Permutation matrix for the vector of equations F
    """
    # Determine new indices for x
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)
    x_length = len(x_entries)
    Px_row = list(range(x_length)) # New indices
    Px_data = [1]*x_length # Permutation matrix is binary matrix
    xg_new = [ind for ind,el in enumerate(x_entries) if 'Gas' in type(el).__name__]
    xe_new = [ind for ind,el in enumerate(x_entries) if 'Elec' in type(el).__name__]
    xh_new = [ind for ind,el in enumerate(x_entries) if 'Heat' in type(el).__name__]
    Px_col = xg_new + xe_new + xh_new # Original indices
    P_x = sps.csr_matrix((Px_data,(Px_row,Px_col)),shape=(x_length,x_length))

    # Determine new indices for F
    F_entries, Fg_entries, Fe_entries, known_P_nodes, known_Q_nodes, Fh_entries, F_m_nodes, F_deltap_links, F_Ts_nodes, F_Tr_nodes, F_phi_halflinks, F_To_halflinks, Fc_entries, F_fc_nodes, F_fc_amount, F_phi_nodes, F_dT_nodes = het_net.get_F_entries(formulation=formulation)
    F_length = len(Fg_entries) + len(Fe_entries) + len(Fh_entries) + np.sum(F_fc_amount) + len(F_phi_nodes) + len(F_dT_nodes)
    PF_row = list(range(F_length)) # New indices
    PF_data = [1]*F_length # Permutation matrix is binary matrix
    Fg_new = [ind for ind,el in enumerate(F_entries) if el in Fg_entries]
    Fe_new = [ind for ind,el in enumerate(F_entries) if el in Fe_entries]
    Fh_new = [ind for ind,el in enumerate(F_entries) if el in Fh_entries]
    for ind_el,el in enumerate(F_phi_nodes + F_dT_nodes):
        Fh_new.append(len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+np.sum(F_fc_amount)+ind_el)
    Fc_new = list()
    for ind_el,el in enumerate(F_fc_nodes):
        ind = ind_el+len(Fg_entries)+len(Fe_entries)+len(Fh_entries)
        if ind_el>0:
            ind += np.sum(F_fc_amount[0:ind_el])-ind_el # -ind_el, because index is already shifted by one with respect to previous element because ind_el has increased 1
        fc_len = F_fc_amount[ind_el]
        for fc_ind in range(fc_len):
            Fc_new.append(ind + fc_ind)
    while len(xg_new) > len(Fg_new):
        if not Fc_new:
            print('No more coupling equations available to move. Unable to make gas part square: |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
            break
        for ind_el,el in enumerate(F_fc_nodes):
            ind = len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+int(np.sum(F_fc_amount[:ind_el]))
            if F_fc_amount[ind_el]>1:
                dfc_dE = el.der_node_law_dE()
                dfc_dq = dfc_dE[:,0]
                for ind_q,der_q in enumerate(dfc_dq): # this coupling function depends on q
                    if der_q:
                        Fg_new.append(ind + ind_q)
                        Fc_new.remove(ind + ind_q)
                        if len(xg_new) <= len(Fg_new):
                            print('Gas part is now square! |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
                            break
            else:
                dfc_dq,dfc_dP,dfc_dphi = el.der_node_law_dE()
                if dfc_dq != None: # this coupling function depends on q
                    Fg_new.append(ind)
                    Fc_new.remove(ind)
                    if len(xg_new) <= len(Fg_new):
                            print('Gas part is now square! |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
                            break
        print('No more coupling equations available to move. Unable to make gas part square: |xg| = {}, |Fg| = {}'.format(len(xg_new),len(Fg_new)))
        break
    while len(xe_new) > len(Fe_new):
        if not Fc_new:
            print('No more coupling equations available to move. Unable to make electrical part square: |xe| = {}, |Fe| = {}'.format(len(xe_new),len(Fe_new)))
            break
        for ind_el,el in enumerate(F_fc_nodes):
            ind = len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+int(np.sum(F_fc_amount[:ind_el]))
            if F_fc_amount[ind_el]>1:
                dfc_dE = el.der_node_law_dE()
                dfc_dP = dfc_dE[:,1]
                for ind_P,der_P in enumerate(dfc_dP): # this coupling function depends on P
                    if der_P:
                        if ind+ind_P in Fc_new: # check if not already moved to Fc
                            Fe_new.append(ind + ind_P)
                            Fc_new.remove(ind + ind_P)
            else:
                dfc_dq,dfc_dP,dfc_dphi = el.der_node_law_dE()
                if dfc_dP != None: # this coupling function depends on P
                    if ind in Fc_new: # check if not already moved to Fc
                        Fe_new.append(ind)
                        Fc_new.remove(ind)
        print('No more coupling equations available to move. Unable to make electrical part square: |xe| = {}, |Fe| = {}'.format(len(xe_new),len(Fe_new)))
        break
    while len(xh_new) > len(Fh_new):
        if not Fc_new:
            print('No more coupling equations available to move. Unable to make heat part square: |xh| = {}, |Fh| = {}'.format(len(xh_new),len(Fh_new)))
            break
        for ind_el,el in enumerate(F_fc_nodes):
            ind = len(Fg_entries)+len(Fe_entries)+len(Fh_entries)+int(np.sum(F_fc_amount[:ind_el]))
            if F_fc_amount[ind_el]>1:
                dfc_dE = el.der_node_law_dE()
                dfc_dphi = dfc_dE[:,2]
                for ind_phi,der_phi in enumerate(dfc_dphi): # this coupling function depends on phi
                    if der_phi:
                        if ind+ind_phi in Fc_new: # check if not already moved to Fc
                            Fh_new.append(ind + ind_phi)
                            Fc_new.remove(ind + ind_phi)
            else:
                dfc_dq,dfc_dP,dfc_dphi = el.der_node_law_dE()
                if dfc_dphi != None: # this coupling function depends on phi
                    if ind in Fc_new: # check if not already moved to Fc
                        Fh_new.append(ind)
                        Fc_new.remove(ind)
        print('No more coupling equations available to move. Unable to make heat part square: |xh| = {}, |Fh| = {}'.format(len(xh_new),len(Fh_new)))
        break
    PF_col = Fg_new + Fe_new + Fh_new + Fc_new
    P_F = sps.csr_matrix((PF_data,(PF_row,PF_col)),shape=(F_length,F_length))
    return P_x, P_F

def jacobians(hydr_eq_gas='fb',form_gas='full',form_heat='standard',heat_load='outflow',node_set_elec=1,node_set_heat=1,coupling_type='CHP',coupling_point=1,node_set=1,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None},tol = 1e-6,max_iter = 150):
    """Plot Jacobian matrices, and eigenvalue spectra, for different indices / ordering"""
    # physical parameters
    c_GG = .01*Ebs
    d_GG = c_GG/10
    Dg = 10*cm
    Dh = 30*cm
    lam = 0.002

    # boundary conditions (when needed)
    p1g = 50*bar #[Pa]
    q3 = 1. #[kg/s]
    q1 = -1.05*q3 #[kg/s]
    V1 = 50*kV #[V]
    delta1 = 0. #[rad]
    delta2 = 0. #[rad]
    V2 = 50*kV #[V]
    P2 = -1*MW #[W]
    P3 = 1.5*-P2 #[W]
    P1 = .9*-P3 #[W]
    Q3 = P3 #[Var]
    Ts1 = 100. #[C]
    p1h = 9*bar
    To2 = .9*Ts1
    To3 = 50. #[C]
    phi2 = -1*MW #[W]
    phi3 = 5.5*-phi2 #[W]
    phi1 = -.9*phi3 #[W]
    To = 100. #[C]
    dT = 50.

    # initial guesses (when needed)
    p2g_init = .99*p1g#30*bar #[Pa]
    p3g_init = .98*p1g#28*bar #[Pa]
    q12_init = q3 #[kg/s]
    q23_init= q3 #[kg/s]
    V2_init = 50*kV #[V]
    V3_init = 50*kV #[V]
    delta1_init = 0. #[rad]
    delta2_init = 0. #[rad]
    delta3_init = 0. #[rad]
    m0_init=1 #[kg/s]
    m1_init=1 #[kg/s]
    p2h_init=8*bar #[Pa]
    p3h_init=6*bar #[Pa]
    Ts1_init=100. #[C]
    Ts2_init=100. #[C]
    Ts3_init=100. #[C]
    Tr1_init=50. #[C]
    Tr2_init=50. #[C]
    Tr3_init=50. #[C]
    qc_init = q3 #[kg/s]
    Pc_init = P3 #[W]
    Qc_init = Q3 #[W]
    mc_init = 1 #[kg/s]
    phic_init = phi3 #[kg/s]
    Toc_init = Ts1 #[C]

    # create network
    gas_net,elec_net,heat_net,het_net = create_network(p1g=p1g,q1=q1,q3=q3,V1=V1,delta1=delta1,delta2=delta2,V2=V2,P1=P1,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi1=phi1,phi2=phi2,phi3=phi3,To=To,dTc=dT,Dg=Dg,hydr_eq_gas=hydr_eq_gas,Dh=Dh,lam=lam,node_set_elec=node_set_elec,node_set_heat=node_set_heat,heat_load=heat_load,coupling_type=coupling_type,c_GG=c_GG,coupling_point=coupling_point,node_set=node_set)
    # initialize
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net,q0_init=q12_init,q1_init=q23_init,p2g_init=p2g_init,p3g_init=p3g_init,V2_init=V2_init,V3_init=V3_init,delta1_init=delta1_init,delta2_init=delta2_init,delta3_init=delta3_init,m0_init=m0_init,m1_init=m1_init,p2h_init=p2h_init,p3h_init=p3h_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,qc_init=qc_init,Pc_init=Pc_init,Qc_init=Qc_init,mc_init=mc_init,Toc_init=Toc_init,phic_init=phic_init,node_set_elec=node_set_elec,node_set_heat=node_set_heat,coupling_type=coupling_type,coupling_point=coupling_point,node_set=node_set,formulation=formulation)

    # create system of equations
    from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
    # scaling
    pgbase = bar
    qbase = 1.
    deltabase = 1.
    Vbase = 50*kV
    Sbase = MW
    phbase = bar
    mbase = 1.
    Tbase = 100.
    phibase = MW
    Egbase = phibase
    scale_var = 'matrix'
    scale_var_params = {'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}
    nlsys = NonLinearSystemHeterogeneous(het_net,formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    nlsys_unscaled = NonLinearSystemHeterogeneous(het_net,formulation=formulation)
    F0 = nlsys.F(x0)
    J0 = nlsys.J(x0)

    # unscaled system
    fig_J = nlsys_unscaled.spy_plot_J(x0,title='Jacobian spy plot')
    ax_J = plt.gca()

    # scaled
    D_x = nlsys.Dx()
    D_x_inv = sps.diags(1/D_x.data[0])
    D_F = nlsys.DF()
    J_scaled = D_F.dot(J0.dot(D_x_inv))
    # spy plot of scaled J over original
    nlsys.spy_plot_J(x0,ax=ax_J,marker='.',markerfacecolor='k',markeredgecolor='k',alpha=0.5)

    # spectrum of (scaled) Jacobian
    fig_spectra = nlsys.spectrum_J(x0,title='Scaled spectra',color='tab:blue')
    ax_spectra = fig_spectra.gca()

    # reordering (coupling with single-carrier parts), first only reorder x
    P_x_re_x, P_F_re_x = perm_matr_x(het_net,formulation=formulation)
    fig_J_re_x = nlsys.spy_plot_J(x0,title='Reordered Jacobian, only x',P_F=P_F_re_x,P_x=P_x_re_x)

    # also reorder F (based on which parts or non-square)
    P_x_re_xF, P_F_re_xF = perm_matr_xF(het_net,formulation=formulation)
    fig_J_re_xF = nlsys.spy_plot_J(x0,title='Reordered Jacobian, both x and F',P_F=P_F_re_xF,P_x=P_x_re_xF)

    J_re_x = P_F_re_x.dot(J_scaled).dot(P_x_re_x.transpose())
    J_re_xF = P_F_re_xF.dot(J_scaled).dot(P_x_re_xF.transpose())
    print('\nFor scaled Jacobians:')
    print('det(J) = {}'.format(np.linalg.det(J_scaled.todense())))
    print('det(J), reordering x = {}'.format(np.linalg.det(J_re_x.todense())))
    print('det(J), reordering x and F = {}'.format(np.linalg.det(J_re_xF.todense())))
    print('cond(J) = {}'.format(np.linalg.cond(J_scaled.todense())))
    print('cond(J), reordering x = {}'.format(np.linalg.cond(J_re_x.todense())))
    print('cond(J), reordering x and F = {}'.format(np.linalg.cond(J_re_xF.todense())))

    # spectra of scaled systems in one plot
    nlsys.spectrum_J(x0,ax=ax_spectra,P_F=P_F_re_x,P_x=P_x_re_x,color='tab:red')
    nlsys.spectrum_J(x0,ax=ax_spectra,P_F=P_F_re_xF,P_x=P_x_re_xF,color='tab:green')
    from matplotlib.lines import Line2D
    ax_spectra.legend(handles=[Line2D([0], [0], color='tab:blue', marker='.',label='original'),
                       Line2D([0], [0], color='tab:red', marker='.',label='reordered x'),
                       Line2D([0], [0], color='tab:green', marker='.',label='reordered x and F')])

    # solve system for the different reorderings
    het_net.reset_network(x0,formulation=formulation)
    print('\nSolving original system')
    x_sol,iters,err_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
    print('Errors orig: {}'.format(err_vec))
    het_net.reset_network(x0,formulation=formulation)
    print('\nSolving system when reordering x (and coupling equations clearly related to heat)')
    x_sol_re_x,iters_re_x,err_vec_re_x,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,P_F=P_F_re_x,P_x=P_x_re_x)
    print('Errors reordering x (and coupling equations clearly related to heat): {}'.format(err_vec_re_x))
    het_net.reset_network(x0,formulation=formulation)
    print('\nSolving system when reordering x and F')
    x_sol_re_xF,iters_re_xF,err_vec_re_xF,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,P_F=P_F_re_xF,P_x=P_x_re_xF)
    print('Errors reordering x and F: {}'.format(err_vec_re_xF))

    fig = plt.figure('Convergence plot different orderings')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||D_F F(x^k)||_2$')
    max_iter_used = np.max([iters,iters_re_x,iters_re_xF])
    ax.semilogy([0,max_iter_used+1],[tol,tol],'k:',label='tolerance')
    ax.semilogy(np.asarray(range(0,iters+1)),err_vec,'.-',color='tab:blue',label='original')
    ax.semilogy(np.asarray(range(0,iters_re_x+1)),err_vec_re_x,'.-',color='tab:red',label='reordered x')
    ax.semilogy(np.asarray(range(0,iters_re_xF+1)),err_vec_re_xF,'.-',color='tab:green',label='reordered x and F')
    xmin = 0
    xmax = max_iter_used
    xticks = range(xmin,xmax+1) # make sure the xticks are integers
    plt.legend()
    plt.grid(which='major',color='k', linestyle='--', alpha=.2)
    plt.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

def comp_solver_time_reod(hydr_eq_gas='fb',form_gas='full',form_heat='standard',heat_load='outflow',node_set_elec=1,node_set_heat=1,coupling_type='CHP',coupling_point=1,node_set=1,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None},max_iter=150,maxmatvecs=5000,tol=1e-6):
    """Compare the time spent in the linear solver and the total time spent for different orderings and different linear solvers"""
    # create plot and lay-out
    fig = plt.figure('Convergence plot different solvers and permutations')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||T_F F(x^k)||_2$')
    colors_perm = {'no':'tab:blue','x':'tab:red','xF':'tab:green'}
    markers_solver = {'solve':'.','gmres':'*','bicgstab':'x'}

    # physical parameters
    c_GG = .01*Ebs
    d_GG = c_GG/10
    Dg = 10*cm
    Dh = 30*cm
    lam = 0.002

    # boundary conditions (when needed)
    p1g = 50*bar #[Pa]
    q3 = 1. #[kg/s]
    q1 = -1.05*q3 #[kg/s]
    V1 = 50*kV #[V]
    delta1 = 0. #[rad]
    delta2 = 0. #[rad]
    V2 = 50*kV #[V]
    P2 = -1*MW #[W]
    P3 = 1.5*-P2 #[W]
    P1 = .9*-P3 #[W]
    Q3 = P3 #[Var]
    Ts1 = 100. #[C]
    p1h = 9*bar
    To2 = .9*Ts1
    To3 = 50. #[C]
    phi2 = -1*MW #[W]
    phi3 = 5.5*-phi2 #[W]
    phi1 = -.9*phi3 #[W]
    To = 100. #[C]
    dT = 50.

    # initial guesses (when needed)
    p2g_init = .99*p1g#30*bar #[Pa]
    p3g_init = .98*p1g#28*bar #[Pa]
    q12_init = q3 #[kg/s]
    q23_init= q3 #[kg/s]
    V2_init = 50*kV #[V]
    V3_init = 50*kV #[V]
    delta1_init = 0. #[rad]
    delta2_init = 0. #[rad]
    delta3_init = 0. #[rad]
    m0_init=1 #[kg/s]
    m1_init=1 #[kg/s]
    p2h_init=8*bar #[Pa]
    p3h_init=6*bar #[Pa]
    Ts1_init=100. #[C]
    Ts2_init=100. #[C]
    Ts3_init=100. #[C]
    Tr1_init=50. #[C]
    Tr2_init=50. #[C]
    Tr3_init=50. #[C]
    qc_init = q3 #[kg/s]
    Pc_init = P3 #[W]
    Qc_init = Q3 #[W]
    mc_init = 1 #[kg/s]
    phic_init = phi3 #[kg/s]
    Toc_init = Ts1 #[C]

    # create network
    gas_net,elec_net,heat_net,het_net = create_network(p1g=p1g,q1=q1,q3=q3,V1=V1,delta1=delta1,delta2=delta2,V2=V2,P1=P1,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi1=phi1,phi2=phi2,phi3=phi3,To=To,dTc=dT,Dg=Dg,hydr_eq_gas=hydr_eq_gas,Dh=Dh,lam=lam,node_set_elec=node_set_elec,node_set_heat=node_set_heat,heat_load=heat_load,coupling_type=coupling_type,c_GG=c_GG,coupling_point=coupling_point,node_set=node_set)
    # initialize
    x0 = initialize_network(gas_net,elec_net,heat_net,het_net,q0_init=q12_init,q1_init=q23_init,p2g_init=p2g_init,p3g_init=p3g_init,V2_init=V2_init,V3_init=V3_init,delta1_init=delta1_init,delta2_init=delta2_init,delta3_init=delta3_init,m0_init=m0_init,m1_init=m1_init,p2h_init=p2h_init,p3h_init=p3h_init,Ts2_init=Ts2_init,Ts3_init=Ts3_init,Tr1_init=Tr1_init,Tr2_init=Tr2_init,Tr3_init=Tr3_init,qc_init=qc_init,Pc_init=Pc_init,Qc_init=Qc_init,mc_init=mc_init,Toc_init=Toc_init,phic_init=phic_init,node_set_elec=node_set_elec,node_set_heat=node_set_heat,coupling_type=coupling_type,coupling_point=coupling_point,node_set=node_set,formulation=formulation)

    # solver information
    lin_solvers = ['solve','gmres','bicgstab']
    pgbase = bar
    qbase = 1.
    deltabase = 1.
    Vbase = 50*kV
    Sbase = MW
    phbase = bar
    mbase = 1.
    Tbase = 100.
    phibase = MW
    Egbase = phibase
    scale_var = 'matrix'
    scale_var_params = {'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}
    max_iters_used = 0

    # permutation matrices
    P_x_re_x, P_F_re_x = perm_matr_x(het_net,formulation=formulation)
    P_x_re_xF, P_F_re_xF = perm_matr_xF(het_net,formulation=formulation)
    perm_x = [np.array([]),P_x_re_x,P_x_re_xF]
    perm_F = [np.array([]),P_F_re_x,P_F_re_xF]
    perm_keys = ['no','x','xF']

    # load flow (for different orderings and using different linear solvers, using default scaling values)
    for ind_P,perm_key in enumerate(perm_keys):
        P_x = perm_x[ind_P]
        P_F = perm_F[ind_P]
        for lin_solver in lin_solvers:
            het_net.reset_network(x0,formulation=formulation)
            print('\nSolving system for {} perm, with {} as lin solver'.format(perm_key,lin_solver))
            x_sol,iters,err_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params,P_F=P_F,P_x=P_x,lin_solver=lin_solver,max_iter_lin=maxmatvecs)
            ax.semilogy(err_vec,ls='-',color=colors_perm.get(perm_key),marker=markers_solver.get(lin_solver),label=lin_solver+', '+perm_key)
            max_iters_used = max(max_iters_used,iters)
            print('Final error is {:6.3e} after {} iterations'.format(err_vec[-1],iters))

    # plot layout
    xmin = 0
    xmax = max_iters_used
    xticks = range(xmin,xmax+1) # make sure the xticks are integers
    ax.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
    ax.legend()
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlim(left=xmin,right=xmax+1)
    ax.set_xticks(xticks)

if __name__ == '__main__':
    legends()

    #compare_conv_node_sets(show_plots=True)
    #compare_conv_form()
    #for coupling_point in [1,2]:
        #for node_set in [1,2,3]:
            #compare_conv_models(coupling_point=coupling_point,node_set=node_set,show_plots=True)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    #plots_pscc_2020_models(dir_path,save_fig=False)
    plots_pscc_2020_forms(dir_path,save_fig=False)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        jacobians()
        comp_solver_time_reod()

    plt.show()
