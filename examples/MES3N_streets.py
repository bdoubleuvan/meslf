from meslf.networks.gas_network import GasNode, GasLink, GasHalfLink
from meslf.networks.electrical_network import ElectricalNode, ElectricalLink, ElectricalHalfLink
from meslf.networks.heat_network import HeatNode, HeatLink, HeatHalfLink
from meslf.networks.heterogeneous_network import HeterogeneousNetwork, HeterogeneousNode
from meslf.networks.create_network import create_radial_line_network, add_gas_data, add_elec_data, add_heat_data
import examples.G3N_line as GasNet
import examples.E3N_line as ElecNet
import examples.H3N_line as HeatNet
import examples.MES3N_line as MES3N_line
from meslf.utils.constants import cm, km, bar, kV, MW
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

from meslf.load_flow.system_of_equations import NonLinearSystemHeat as nlsys_heat

# physical parameters
gas = GasNet.gas
water = HeatNet.water
Ta = HeatNet.Ta
Ebs = MES3N_line.Ebs
c_GG = .01*Ebs
d_GG = c_GG/10

link_type_h = 'standard_pipe_low_pres_pole'
link_type_g = 'pipe_high_pres_weymouth'
link_type_e = 'pi_line'
L12 = 4*km #[m]
L23 = 5*km #[m]
L_street = L23 #[m]
Dg = 10*cm
E = .98
De = 1*cm
Dh = 30*cm#15*cm
lam = 0.002
lam_streets = lam

# boundary conditions (when needed)
p1g = 50*bar #[Pa]
q3 = 1. #[kg/s]
q1 = -1.05*q3 #[kg/s]
V1 = 50*kV #[V]
delta1 = 0. #[rad]
delta2 = -.005#0 #[rad]
V2 = 50*kV #[V]
P2 = -1*MW #[W]
P3 = 1.5*-P2 #[W]
P1 = .9*-P3 #[W]
Q3 = P3 #[Var]
Ts1 = 100. #[C]
p1h = 9*bar
To2 = .988*Ts1#.9*Ts1
To3 = 50. #[C]
phi2 = -1*MW #[W]
phi3 = 5.5*-phi2 #[W]
phi1 = -.9*phi3 #[W]
To = To2#Ts1 #[C]
dTc = 50.

# topology information
node_set_elec = 2
node_set_heat = 2
dy = 0.3
dx = 0.3
x_start = 4
y_shift_gas = 0
x_shift_gas = x_start
y_shift_elec = dy
x_shift_elec = x_start  + dx
y_shift_heat = 2*dy
x_shift_heat = x_start  + 2*dx


legend_handles_node_sets = [Line2D([0], [0], color=MES3N_line.colors.get('CHP'), label='CHP'),
                    Line2D([0], [0], color=MES3N_line.colors.get('EH'), label='EH'),
                    Line2D([0], [0], color=MES3N_line.colors.get('GB+GG'), label='GB + GG'),
                    Line2D([0], [0], color=MES3N_line.colors.get('GB+GG_vp'), label='GB + GG VP'),
                    Line2D([0], [0], color='k',ls=MES3N_line.linestyles_node_sets.get(1), label='node set 1'),
                    Line2D([0], [0], color='k',ls=MES3N_line.linestyles_node_sets.get(2), label='node set 2'),
                    Line2D([0], [0], color='k',ls=MES3N_line.linestyles_node_sets.get(3), label='node set 3'),
                    Line2D([0], [0], color='k',ls=':', label='tolerance')]

legend_handles_sc = MES3N_line.legend_handles_sc

def create_streets_gas(n,m,number_of_streets,gas,L,D,E,q_demand):
    """Create a list of street networks

    Parameters
    ----------
    n : int
        number of loads in the street
    m : int
        number of junctions with two loads
    number_of_streets : int
        number of street networks to be created
    carrier : Carrier
        the gas Carrier object used for every network

    Returns
    -------
    streets : list
        list with GasNetwork objects
    """
    streets = list()
    # node data
    p_ref = p1g #[Pa]
    # link data
    D_unit = 'm'
    L_unit = 'm'
    # half link data
    np.random.seed(0)
    q = q_demand/(n*number_of_streets) #[kg/s]
    q_unit = 'kg/s'
    y_shift = y_shift_gas - 1
    for i in range(number_of_streets):
        # topology
        street = create_radial_line_network(n,carrier='g',m=m,net_name='S'+str(i))

        # link data
        NE = 2*n-m
        NJ = n-m #number of junctions in the street
        link_params = list()
        for i in range(NJ):# links between junctions (and source)
            if i <= m:
                load_frac = (n - 2*i)/n # fraction of total load going through this link
            else:
                load_frac = (n - 2*m - (i-m))/n # fraction of total load going through this link
            L_link = L*load_frac
            D_link = D*np.sqrt(load_frac)
            link_params.append({'carrier':gas, 'D':D_link, 'D_unit':D_unit,'L':L_link,'L_unit':L_unit,'E':E})
        for i in range(n): # links between junctions and loads
            L_link = L/n
            D_link = D/np.sqrt(n)
            link_params.append({'carrier':gas, 'D':D_link, 'D_unit':D_unit,'L':L_link,'L_unit':L_unit,'E':E})
        link_types = [link_type_g]*NE
        # half link data
        q_inj = q*np.ones(n)

        # add data
        add_gas_data(street,p_ref,q_inj,link_types,link_params)

        # adjust y coordinates and rename nodes
        street.nodes[0].name = 'gs'
        for node in street.get_nodes():
            node.name += street.name
            node.y += y_shift
            node.x += x_shift_gas
        y_shift += dy

        streets.append(street)

    return streets

def create_streets_elec(n,m,number_of_streets,L,D,P_demand,Q_demand):
    """Create a list of street networks

    Parameters
    ----------
    n : int
        number of loads in the street
    m : int
        number of junctions with two loads
    number_of_streets : int
        number of street networks to be created

    Returns
    -------
    streets : list
        list with ElectricalNetwork objects
    """
    streets = list()
    # node data
    V_ref = V2 #[V]
    delta_ref = delta2 #[rad]

    P = P_demand/(n*number_of_streets) #[W]
    Q = Q_demand/(n*number_of_streets) #[W]

    y_shift = y_shift_elec - 1
    for i in range(number_of_streets):
        # topology
        street = create_radial_line_network(n,carrier='e',m=m,net_name='S'+str(i))

        # link data
        NE = 2*n-m
        NJ = n-m #number of junctions in the street
        link_params = list()
        for i in range(NJ):# links between junctions (and source)
            if i <= m:
                load_frac = (n - 2*i)/n # fraction of total load going through this link
            else:
                load_frac = (n - 2*m - (i-m))/n # fraction of total load going through this link
            L_link = L*load_frac
            D_link = D*np.sqrt(load_frac)
            b,g,b_sh = ElecNet.create_line_data(L_link,D_link)
            if link_type_e == 'pi_line':
                link_param = {'b':b,'g':g,'b_sh':b_sh,'g_sh':0}
            elif link_type_e == 'short_line':
                link_param = {'b':b,'g':g,'b_sh':0,'g_sh':0}
            else:
                raise ValueError('Link type should be pi_line or short_line')
            link_params.append(link_param)
        for i in range(n): # links between junctions and loads
            L_link = L/n
            D_link = D/np.sqrt(n)
            b,g,b_sh = ElecNet.create_line_data(L_link,D_link)
            if link_type_e == 'pi_line':
                link_param = {'b':b,'g':g,'b_sh':b_sh,'g_sh':0}
            elif link_type_e == 'short_line':
                link_param = {'b':b,'g':g,'b_sh':0,'g_sh':0}
            else:
                raise ValueError('Link type should be pi_line or short_line')
            link_params.append(link_param)
        link_types = [link_type_e]*NE

        # half link data
        P_inj = P*np.ones(n)
        Q_inj = Q*np.ones(n)

        # add data
        add_elec_data(street,V_ref,delta_ref,P_inj,Q_inj,link_types,link_params)

        # adjust y coordinates and rename nodes
        street.nodes[0].name = 'es'
        for node in street.get_nodes():
            node.name += street.name
            node.y += y_shift
            node.x += x_shift_elec
        y_shift += dy

        streets.append(street)

    return streets

def create_streets_heat(n,m,number_of_streets,water,L,D,lam,phi_demand,heat_load='outflow'):
    """Create a list of street networks

    Parameters
    ----------
    n : int
        number of loads in the street
    m : int
        number of junctions with two loads
    number_of_streets : int
        number of street networks to be created

    Returns
    -------
    streets : list
        list with HeatNetwork objects
    """

    streets = list()

    # node data
    p_ref = p1h #[Pa] initial guess for pressure at start node of street
    Ts_ref = Ts1 #[C]
    # half link data
    To = 50. #[C]
    phi = phi_demand/(n*number_of_streets)

    y_shift = y_shift_heat - 1
    for i in range(number_of_streets):
        # topology
        street = create_radial_line_network(n,carrier='h',m=m,net_name='S'+str(i))

        # link data
        D_unit = 'm'
        L_unit = 'm'
        NE = 2*n-m
        NJ = n-m #number of junctions in the street
        link_params = list()
        for i in range(NJ):# links between junctions (and source)
            if i <= m:
                load_frac = (n - 2*i)/n # fraction of total load going through this link
            else:
                load_frac = (n - 2*m - (i-m))/n # fraction of total load going through this link
            L_link = L*load_frac
            D_link = D*np.sqrt(load_frac)
            lam_link = lam_streets
            U = lam_link/(np.pi*D_link) #[W/(m^2 K)]
            link_params.append({'carrier':water, 'D':D_link, 'D_unit':D_unit,'L':L_link,'L_unit':L_unit,'U':U,'Ta':Ta})
        for i in range(n): # links between junctions and loads
            L_link = L/n
            D_link = D/np.sqrt(n)
            lam_link = lam_streets
            U = lam_link/(np.pi*D_link) #[W/(m^2 K)]
            link_params.append({'carrier':water, 'D':D_link, 'D_unit':D_unit,'L':L_link,'L_unit':L_unit,'U':U,'Ta':Ta})
        link_types = [link_type_h]*NE

        # half link data
        phi_inj = phi*np.ones(n+1)
        Tr_inj = To*np.ones(n+1)
        Ts_inj = Ts_ref*np.ones(n+1) #source node
        dT = Ts_inj - Tr_inj
        halflink_types = ['heat_exchanger']*(n+1)
        halflink_params = [{'carrier':water}]*(n+1)
        halflink_bc_types = [0]+[3]*n # source slack + sinks with dphi and To known

        # add data
        add_heat_data(street,p_ref,Ts_ref,phi_inj,Ts_inj,Tr_inj,dT,link_types,link_params,halflink_types,halflink_params,halflink_bc_types)

        # adjust y coordinates and rename nodes
        street.nodes[0].name = 'hs'
        for node in street.get_nodes():
            node.name += street.name
            node.y += y_shift
            node.x += x_shift_heat
            if heat_load == 'delta' and node.node_type == 1:
                node.node_type = 12
                for hl in node.get_half_links():
                    if hl.source:
                        hl.bc_type = 4
                    else:
                        hl.bc_type = 5
        y_shift += dy

        streets.append(street)

    return streets

def create_network(n,m,number_of_streets,p1g=p1g,q1=q1,q3=q3,V1=V1,delta1=delta1,delta2=delta2,V2=V2,P1=P1,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi1=phi1,phi2=phi2,phi3=phi3,To=To,dTc=dTc,L12=L12,L23=L23,Dg=Dg,E=E,link_type_g=link_type_g,De=De,link_type_e=link_type_e,Dh=Dh,lam=lam,link_type_h=link_type_h,node_set_elec=node_set_elec,node_set_heat=node_set_heat,c_GG=c_GG,hydr_eq_gas='fb',heat_load='outflow',coupling_type='CHP',coupling_point=1,node_set=1):
    """Creates the multi-carrier network"""
    gas_net,elec_net,heat_net,het_net = MES3N_line.create_network(p1g=p1g,q1=q1,q3=q3,V1=V1,delta1=delta1,delta2=delta2,V2=V2,P1=P1,P2=P2,P3=P3,Q3=Q3,Ts1=Ts1,p1h=p1h,To2=To2,To3=To3,phi1=phi1,phi2=phi2,phi3=phi3,To=To,dTc=dTc,L12=L12,L23=L23,Dg=Dg,E=E,link_type_g=link_type_g,hydr_eq_gas=hydr_eq_gas,De=De,link_type_e=link_type_e,Dh=Dh,lam=lam,link_type_h=link_type_h,node_set_elec=node_set_elec,node_set_heat=node_set_heat,heat_load=heat_load,coupling_type=coupling_type,c_GG=c_GG,coupling_point=coupling_point,node_set=node_set)

    heat_net.initialize() #THIS IS SOMEHOW NEEDED WHEN USING THE HALF LINK FLOW FORMULATION???!!! If I don't include this, NR diverges like crazy. Don't know why?? Initialize adds the half links to the heat network... Maybe it has to do with numbering? Of maybe with some half link value that is taken somewhere?

    # create the long streets
    gn3 = gas_net.nodes[2]
    en3 = elec_net.nodes[2]
    hn3 = heat_net.nodes[2]
    gas_streets = create_streets_gas(n,m,number_of_streets,gas,L_street,Dg,E,gn3.half_links[0].q)
    elec_streets = create_streets_elec(n,m,number_of_streets,L_street,De,en3.half_links[0].P,en3.half_links[0].Q)
    heat_streets = create_streets_heat(n,m,number_of_streets,water,L_street,Dh,lam,hn3.half_links[0].dphi,heat_load=heat_load)

    for gas_street in gas_streets:
        for node in gas_street.get_nodes():
            gas_net.add_node(node)
            het_net.add_node(node)
            for halflink in node.get_half_links():
                gas_net.add_half_link(halflink)
                het_net.add_half_link(halflink)
        for link in gas_street.get_links():
            link.name += '_'+gas_street.name
            gas_net.add_link(link)
            het_net.add_link(link)
        gas_street_source = gas_street.nodes[0]
        hl = GasLink('gl_'+gas_street_source.name,gn3,gas_street_source,link_type=link_type_g,link_params={'carrier':gas, 'D':Dg, 'L':L_street,'E':E})
        gas_net.add_link(hl)
        het_net.add_link(hl)
        gas_street_source.node_type = 1 # junction node (load node)
        for hl in gas_street_source.get_half_links():
            hl.q = 0
    gn3.node_type = 1 # junction node (load node)
    for hl in gn3.get_half_links():
        hl.q = 0

    for elec_street in elec_streets:
        for node in elec_street.get_nodes():
            elec_net.add_node(node)
            het_net.add_node(node)
            for halflink in node.get_half_links():
                elec_net.add_half_link(halflink)
                het_net.add_half_link(halflink)
        for link in elec_street.get_links():
            link.name += '_'+elec_street.name
            elec_net.add_link(link)
            het_net.add_link(link)
        elec_street_source = elec_street.nodes[0]
        b,g,b_sh = ElecNet.create_line_data(L_street,De)
        if link_type_e == 'pi_line':
            link_params = {'b':b,'g':g,'b_sh':b_sh,'g_sh':0}
        elif link_type_e == 'short_line':
            link_params = {'b':b,'g':g,'b_sh':0,'g_sh':0}
        hl = ElectricalLink('el_'+elec_street_source.name,en3,elec_street_source,link_type=link_type_e,link_params=link_params)
        elec_net.add_link(hl)
        het_net.add_link(hl)
        elec_street_source.node_type = 2 # junction node (load node)
        for hl in elec_street_source.get_half_links():
            hl.P = 0
            hl.Q = 0
    en3.node_type = 2 # junction node (load node)
    for hl in en3.get_half_links():
        hl.P = 0
        hl.Q = 0

    U = lam/(np.pi*Dh) #[W/(m^2 K)]
    for heat_street in heat_streets:
        for node in heat_street.get_nodes():
            heat_net.add_node(node)
            het_net.add_node(node)
            for halflink in node.get_half_links():
                heat_net.add_half_link(halflink)
                het_net.add_half_link(halflink)
        for link in heat_street.get_links():
            link.name += '_'+heat_street.name
            heat_net.add_link(link)
            het_net.add_link(link)
        heat_street_source = heat_street.nodes[0]
        hl = HeatLink('hl_'+heat_street_source.name,hn3,heat_street_source,link_type=link_type_h,link_params={'L':L_street,'D':Dh,'U':U,'carrier':water})
        heat_net.add_link(hl)
        het_net.add_link(hl)
        heat_street_source.node_type = 2 # junction node
        for hl in heat_street_source.get_half_links():
            heat_street_source.remove_half_link(hl)
            heat_net.remove_half_link(hl)
            het_net.remove_half_link(hl)
    hn3.node_type = 2 # junction node
    for hl in hn3.get_half_links():
            hn3.remove_half_link(hl)
            heat_net.remove_half_link(hl)
            het_net.remove_half_link(hl)

    return gas_net,elec_net,heat_net,het_net

def initialize_network(het_net, gas_net, elec_net, heat_net,n,m,nS,p_perc_high,p_perc_low,T_perc_high,T_perc_low,q3=q3,P3=P3,Q3=Q3,phi3=phi3,Ts1=Ts1,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None}):
    """Set initial values for the network

    Returns
    -------
    x0 : np array
        Vector with initial guess
    """
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)

    N = 2*n-m+1 #number of nodes per streets (+1 because of street source)
    NJ = n-m #number of junctions per street

    # gas
    unknown_pg_nodes = []
    unknown_q_links = []
    for el in xg_entries:
        if isinstance(el,GasNode):
            unknown_pg_nodes.append(el)
        elif isinstance(el,GasLink):
            unknown_q_links.append(el)
    q_load_avg = np.mean([hl.q for node in gas_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.q>0])
    q_load_tot = np.sum([hl.q for node in gas_net.get_nodes(node_types=[1]) for hl in node.get_half_links()  if hl.q>0])
    pg_ref = gas_net.nodes[0].p #[bar]
    pg_init = np.zeros(len(unknown_pg_nodes))
    pg_ind = 0
    for i in range(3):
        if gas_net.nodes[i] in unknown_pg_nodes: # heat node i
            pg_init[pg_ind] = pg_ref*(1-(1-p_perc_high)*(pg_ind+1)/4)
            pg_ind += 1
    for num_S in range(nS): # gas nodes per street
        # nodes are numbered such that the load come first, and then the junctions
        pg_init_street = pg_ref*np.linspace(p_perc_high,p_perc_low,N)
        pg_init[num_S*N+pg_ind] = pg_init_street[0] #street source
        pg_init[num_S*N+pg_ind+1:num_S*N+n+pg_ind+1] = pg_init_street[NJ+1:N] #loads (lowest pressures)
        pg_init[num_S*N+n+pg_ind+1:num_S*N+N+pg_ind] = pg_init_street[1:NJ+1] #junctions (highest pressures)
    if unknown_q_links:
        q_init = np.concatenate((np.array([q_load_tot,q_load_tot]),q_load_avg*np.ones(len(unknown_q_links)-2)))
    else:
        q_init = q_load_tot*np.ones(len(unknown_q_links))
    xg_init = np.concatenate((q_init,pg_init))

    # electricity
    P_init_load = np.mean([hl.P for node in elec_net.get_nodes(node_types=[2]) for hl in node.get_half_links()  if hl.P>0])
    delta_ref = elec_net.nodes[0].delta
    V_ref = elec_net.nodes[0].V
    delta_init = delta_ref*np.ones(len(unknown_delta_nodes))
    V_init = V_ref*np.ones(len(unknown_V_nodes))
    xe_init = np.concatenate((delta_init,V_init))

    # heat
    ph_ref = heat_net.nodes[0].p
    Ts_ref = heat_net.nodes[0].Ts
    ph_init = np.zeros(len(unknown_p_nodes))
    ph_ind = 0
    for i in range(3):
        if heat_net.nodes[i] in unknown_p_nodes: # heat node i
            ph_init[ph_ind] = ph_ref*(1-(1-p_perc_high)*(ph_ind+1)/4)
            ph_ind += 1
    Ts_init = np.zeros(len(unknown_Ts_nodes))
    Ts_ind = 0
    for i in range(3):
        if heat_net.nodes[i] in unknown_Ts_nodes: # heat node i
            Ts_init[Ts_ind] = Ts_ref*(1-(1-T_perc_high)*(Ts_ind+1)/4)
            Ts_ind += 1
    for num_S in range(nS): # heat nodes per street
        # nodes are numbered such that the load come first, and then the junctions
        ph_init_street = ph_ref*np.linspace(p_perc_high,p_perc_low,N)
        ph_init[num_S*N+ph_ind ] = ph_init_street[0] #street source
        ph_init[num_S*N+ph_ind+1:num_S*N+n+ph_ind+1] = ph_init_street[NJ+1:N] #loads (lowest pressures)
        ph_init[num_S*N+n+ph_ind+1:num_S*N+N+ph_ind] = ph_init_street[1:NJ+1] #junctions (highest pressures)
        Ts_init_street = Ts_ref*np.linspace(T_perc_high,T_perc_low,N)
        Ts_init[num_S*N+Ts_ind] = Ts_init_street[0] #street source
        Ts_init[num_S*N+Ts_ind+1:num_S*N+n+Ts_ind+1] = Ts_init_street[NJ+1:N] #loads (lowest temperatures)
        Ts_init[num_S*N+n+Ts_ind+1:num_S*N+N+Ts_ind] = Ts_init_street[1:NJ+1] #junctions (highest temperatures)
    hl_flow = list()
    hl_Tr = list()
    for node in heat_net.get_nodes(node_types=[1,3,4,12,13]):
        for hl in node.get_half_links():
            # set temperatures to initial some value. Needed to do hl.flow()
            print('Half link {}, hl bc type: {}, sink: {}, source: {}, Ts in x: {}, Tr in x: {}'.format(hl.name,hl.bc_type,hl.sink,hl.source,hl in unknown_Ts_halflinks,hl in unknown_Tr_halflinks))
            if hl in unknown_Ts_halflinks:
                print('Updating Ts, since hl Ts in x')
                hl.Ts = Ts_ref
            if hl in unknown_Tr_halflinks:
                print('Updating Tr, since hl Tr in x')
                hl.Tr = 50.
            if hl.bc_type in [4,5,10,11]: #dT known
                print('Updating Ts or Tr, since dT is known')
                if hl.sink:
                    print('half link is sink')
                    if not (hl in unknown_Ts_halflinks):
                        print('Setting Ts for sink half link, based on start node Ts')
                        hl.Ts = hl.start_node.Ts
                    print('Setting Tr for sink half link, based on dT')
                    hl.Tr = hl.Ts - hl.dT
                else:
                    print('half link is source')
                    if not (hl in unknown_Tr_halflinks):
                        print('Setting Tr for source half link, based on fixed value')
                        hl.Tr = 50
                    print('Setting Ts for source half link, based on dT')
                    hl.Ts = hl.dT + hl.Tr
            print('For halflink {}, Ts = {}, Tr = {}, dT = {}. hl bc type: {}, start node type: {}, sink: {}, source: {}, start node {}, Ts in x: {}, Tr in x: {}'.format(hl.name,hl.Ts,hl.Tr,hl.dT,hl.bc_type,hl.start_node.node_type,hl.sink,hl.source,hl.start_node.name,hl in unknown_Ts_halflinks,hl in unknown_Tr_halflinks))
            m_hl = hl.flow()
            hl.m = m_hl
            hl_flow.append(m_hl)
            if hl.sink:
                hl_Tr.append(hl.Tr)
    m_init_value = np.mean(hl_flow)
    m_load_tot = np.sum(hl_flow)

    m_init = np.concatenate((np.array([m_load_tot,m_load_tot]),m_init_value*np.ones(len(unknown_m_links)-2)))
    m_hl_init = np.zeros(len(unknown_m_halflinks))
    for ind_hl,hl in enumerate(unknown_m_halflinks):
        m_hl_init[ind_hl] = hl.flow()
    Tr_init_value = np.min(hl_Tr)
    Tr_init = Tr_init_value*np.ones(len(unknown_Tr_nodes))
    Ts_hl_init = np.zeros(len(unknown_Ts_halflinks))
    Tr_hl_init = np.zeros(len(unknown_Tr_halflinks))
    for ind_hl,hl in enumerate(unknown_Ts_halflinks):
        Ts_hl_init[ind_hl] = Ts_ref
    for ind_hl,hl in enumerate(unknown_Tr_halflinks):
        Tr_hl_init[ind_hl] = Tr_init_value
    xh_init = np.concatenate((m_init,m_hl_init,ph_init,Ts_init,Tr_init,Ts_hl_init,Tr_hl_init))

    #coupling
    qc_init = q3*np.ones(len(unknown_qc_links))
    Pc_init = P3*np.ones(len(unknown_Pc_links))
    Qc_init = Q3*np.ones(len(unknown_Qc_links))
    phic_init = phi3*np.ones(len(unknown_dphi_links))
    mc_init = m_load_tot*np.ones(len(unknown_mc_links))
    Toc_init = Ts1*np.ones(len(unknown_Ts_links))
    xc_init = np.concatenate((qc_init,Pc_init,Qc_init,mc_init,phic_init,Toc_init))

    x_init = np.concatenate((xg_init,xe_init,xh_init,xc_init))
    het_net.initialize()
    het_net.update(x_init,formulation=formulation)
    x0 = het_net.set_x_init(formulation=formulation)
    return x0

def run_load_flow(n,m,number_of_streets,pgbase,qbase,Vbase,Sbase,phbase,mbase,Tbase,phibase,Egbase,p_perc_high,p_perc_low,T_perc_high,T_perc_low,deltabase=1,hydr_eq_gas='fb',heat_load='outflow',coupling_type='CHP',coupling_point=1,node_set=1,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},tol = 1e-6,max_iter = 150,gas_lf=True,elec_lf=True,heat_lf=True,max_nodes=20):
    """Stead-state load flow analysis of heterogeneous network

    Parameters
    ----------

    """
    print('\nLoad flow with {} at node {}, node set {}, heat load {}, form. gas {}, hyrd. eq. gas {},  form. heat {}'.format(coupling_type,coupling_point,node_set,heat_load,formulation.get('gas'),hydr_eq_gas,formulation.get('heat')))
    # create network
    gas_net,elec_net,heat_net,het_net = create_network(n,m,number_of_streets,hydr_eq_gas=hydr_eq_gas,heat_load=heat_load,coupling_type=coupling_type,coupling_point=coupling_point,node_set=node_set)

    # initialize
    x0 = initialize_network(het_net, gas_net, elec_net, heat_net,n,m,nS,p_perc_high,p_perc_low,T_perc_high,T_perc_low,formulation=formulation)
    print('length x = {}'.format(len(x0)))
    # run mes load flow
    x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var='matrix',scale_var_params={'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase})
    if len(het_net.nodes) < max_nodes:
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

    # run single-carrier load flows. NB: This alters the single-carrier network parts!!
    x_sol_g = None
    iters_g = 0
    err_vec_g = None
    x_sol_e = None
    iters_e = 0
    err_vec_e = None
    x_sol_h = None
    iters_h = 0
    err_vec_h = None
    if gas_lf:
        het_net.reset_network(x0,formulation=formulation)
        for glc in gas_net.get_links(link_types=['dummy']): # remove coupling links
            gas_net.remove_link(glc)
            glc.start_node.out_links.remove(glc) # link is only removed from the network, not deleted or something. So still connected to the node.
        # change node types
        gn0_type = gas_net.nodes[0].node_type
        gn1_type= gas_net.nodes[1].node_type
        gas_net.nodes[0].node_type = 0 #ref node
        for hl in gas_net.nodes[0].get_half_links():
            hl.q = 0
        gas_net.nodes[1].node_type = 1 #load
        gas_net.initialize() # removed a link, so needs to be initialized again (otherwise A will be wrong, for instance)
        # run load flow
        scale_var = 'matrix'
        scale_var_params = {'qbase':qbase,'pgbase':pgbase}
        x_sol_g,iters_g,err_vec_g,_,_,_ = gas_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('gas'),scale_var=scale_var,scale_var_params=scale_var_params)
        # change node types back, such that the heterogeneous network is not changed.
        gas_net.nodes[0].node_type = gn0_type
        gas_net.nodes[1].node_type = gn1_type
        gas_net.add_link(glc,position=glc.number)
        glc.start_node.out_links.append(glc)
        gas_net.initialize()
    if elec_lf:
        het_net.reset_network(x0,formulation=formulation)
        for elc in elec_net.get_links(link_types=['dummy']): # remove coupling links
            elec_net.remove_link(elc)
            elc.end_node.in_links.remove(elc) # link is only removed from the network, not deleted or something. So still connected to the node.
        # change node types
        en0_type = elec_net.nodes[0].node_type
        en1_type= elec_net.nodes[1].node_type
        elec_net.nodes[0].node_type = 0 #ref slack
        for hl in elec_net.nodes[0].get_half_links():
            hl.P = 0
            hl.Q = 0
        elec_net.nodes[1].node_type = 2 #load
        elec_net.initialize() # removed a link, so needs to be initialized again (otherwise Y will be wrong, for instance)
        # run load flow
        scale_var='matrix'
        scale_var_params={'deltabase':deltabase,'Vbase':Vbase,'Sbase':Sbase}
        x_sol_e,iters_e,err_vec_e,_,_,_,_,_ = elec_net.solve_network(tol,max_iter,solver='NR',scale_var=scale_var,scale_var_params=scale_var_params)
        # change node types back, such that the heterogeneous network is not changed.
        elec_net.nodes[0].node_type = en0_type
        elec_net.nodes[1].node_type = en1_type
        elec_net.add_link(elc,position=elc.number)
        elc.end_node.in_links.append(elc)
        elec_net.initialize()
    if heat_lf:
        print('\nRunning single-carrier heat load flow')
        het_net.reset_network(x0,formulation=formulation)
        for hlc in heat_net.get_links(link_types=['dummy']): # remove coupling links
            heat_net.remove_link(hlc)
            hlc.end_node.in_links.remove(hlc) # link is only removed from the network, not deleted or something. So still connected to the node.
        # change node types
        hn0_type = heat_net.nodes[0].node_type
        hn1_type = heat_net.nodes[1].node_type
        hn0 = heat_net.nodes[0]
        hn1 = heat_net.nodes[1]
        heat_net.nodes[0].node_type = 0 #ref node

        if hn0_type in [2,5,6,7]: #junction
            HeatHalfLink('hn1_hl_new',hn0,Ts=Ts1,dphi=phi1,link_type='heat_exchanger',link_params={'carrier':water},bc_type=2)
        if heat_load == 'outflow':
            hn1.node_type = 1 #load
        elif heat_load == 'delta':
            hn1.node_type = 12 #load
        if hn1_type in [2,5,6,7]: #junction
            HeatHalfLink('hn2_hl_new',hn1,Tr=To,dphi=phi2,dT=dTc,link_type='heat_exchanger',link_params={'carrier':water},bc_type=3)
        else:
            hn1.half_links[0].Tr=To
        heat_net.initialize() # removed a link, so needs to be initialized again (otherwise A will be wrong, for instance). Also, all the half-link are added to the network
        # run load flow
        scale_var='matrix'
        scale_var_params={'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase}
        x_sol_h,iters_h,err_vec_h,_,_,_,_,_,_,_,_ = heat_net.solve_network(tol,max_iter,solver='NR',formulation=formulation.get('heat'),scale_var=scale_var,scale_var_params=scale_var_params)
        # change node types back, such that the heterogeneous network is not changed.
        heat_net.add_link(hlc,position=hlc.number)
        hlc.end_node.in_links.append(hlc)
        heat_net.nodes[0].node_type = hn0_type
        heat_net.nodes[1].node_type = hn1_type
        if hn0_type in [2,5,6,7]: #junction
            for hl in heat_net.nodes[0].get_half_links():
                heat_net.remove_half_link(hl)
                heat_net.nodes[0].remove_half_link(hl)
        if hn1_type in [2,5,6,7]: #junction
            for hl in heat_net.nodes[1].get_half_links():
                heat_net.remove_half_link(hl)
                heat_net.nodes[1].remove_half_link(hl)
        heat_net.initialize()

    # Update values.
    if gas_lf or elec_lf or heat_lf:
        het_net.initialize()
        het_net.update_full(x_sol,formulation=formulation)

    return het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,x_sol_g,iters_g,err_vec_g,x_sol_e,iters_e,err_vec_e,x_sol_h,iters_h,err_vec_h

def legends():
    """Make separate figures with the legends used in the different convergence plots"""

    # legend used for the convergence plot, if the node sets are compared
    fig_legend_node_sets = plt.figure('Legend_node_sets_streets')
    ax_legend_node_sets = fig_legend_node_sets.gca()
    ax_legend_node_sets.axis('off')
    fig_legend_node_sets.patch.set_visible(False)
    ax_legend_node_sets.legend(handles=legend_handles_node_sets,loc='center')

     # legend used for single carrier networks
    fig_legend_sc = plt.figure('Legend_sc_streets')
    ax_legend_sc = fig_legend_sc.gca()
    ax_legend_sc.axis('off')
    fig_legend_sc.patch.set_visible(False)
    ax_legend_sc.legend(handles=legend_handles_sc,loc='center')

def compare_conv_node_sets(n,m,nS,hydr_eq_gas='fb',form_gas='full',form_heat='standard',heat_load='outflow',gas_lf=True,elec_lf=True,heat_lf=True):
    """Compare the convergence of load flow for different topologies and different node sets (both in multi-carrier and single-carrier)
    """
    # node sets and coupling type and coupling point
    coupling_types = ['CHP','EH','GB+GG','GB+GG_vp']
    coupling_points = [1,2]
    node_sets = [1,2,3]

    x_sol_dict = dict()
    errors_dict = dict()
    iters_dict = dict()
    x_sol_sc = dict()
    errors_sc = dict()
    iters_sc = dict()
    max_iters_used = 0

    # initial guess
    if n == 1 and m == 0:
        p_perc_high = .99
        p_perc_low = .98
    else:
        p_perc_step = .001/(2*n-m)
        p_perc_high = 1-p_perc_step#.99
        p_perc_low = max(.1,1-p_perc_step*(n-m))#.97
    T_perc_step = .001/(2*n-m)
    T_perc_high = 1-T_perc_step#1.
    T_perc_low = max(.1,1-T_perc_step*(n-m))#1.

    # solver information
    tol = 1e-6
    max_iter = 50
    formulation={'gas':form_gas,'elec':'complex_power','heat':form_heat,'het':None}

    # scaling
    pgbase = 50*bar
    qbase = 1.
    deltabase = 1.
    Vbase = 50*kV
    Sbase = MW
    phbase = bar
    mbase = 1.
    Tbase = 100.
    phibase = MW
    Egbase = phibase

    # load flow mes
    for coupling_point in coupling_points:
        for coupling_type in coupling_types:
            for node_set in node_sets:
                if node_set == 3 and coupling_point == 1:
                    continue # continue to the next iteration
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
                    warnings.filterwarnings("ignore", "Only a",UserWarning)
                    het_net,x_sol,iters,err_vec,p_g_vec,q_vec,q_inj,delta_vec,V_mag_vec,S_inj,P_edge,Q_edge,m_vec,p_h_vec,Ts_vec,Tr_vec,m_hl_vec,phi_hl_vec,Ts_hl_vec,Tr_hl_vec,qc_vec,Pc_vec,Qc_vec,mc_vec,phic_vec,Tsc_vec,Trc_vec,x_sol_g,iters_g,err_vec_g,x_sol_e,iters_e,err_vec_e,x_sol_h,iters_h,err_vec_h = run_load_flow(n,m,nS,pgbase,qbase,Vbase,Sbase,phbase,mbase,Tbase,phibase,Egbase,p_perc_high,p_perc_low,T_perc_high,T_perc_low,deltabase=deltabase,hydr_eq_gas=hydr_eq_gas,heat_load=heat_load,coupling_type=coupling_type,coupling_point=coupling_point,node_set=node_set,formulation=formulation,tol=tol,max_iter=max_iter,gas_lf=gas_lf,elec_lf=elec_lf,heat_lf=heat_lf)
                    key = '{} {} {} {} {} {} {} {} {} {}'.format(n,m,nS,form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set,coupling_type)
                    x_sol_dict[key] = x_sol
                    errors_dict[key] = err_vec
                    iters_dict[key] = iters
                    if gas_lf:
                        key_gas = '{} {} {} {} {}'.format(n,m,nS,form_gas,hydr_eq_gas)
                        x_sol_sc[key_gas] = x_sol_g
                        errors_sc[key_gas] = err_vec_g
                        iters_sc[key_gas] = iters_g
                    if elec_lf:
                        key_elec = '{} {} {}'.format(n,m,nS)
                        x_sol_sc[key_elec] = x_sol_e
                        errors_sc[key_elec] = err_vec_e
                        iters_sc[key_elec] = iters_e
                    if heat_lf:
                        key_heat = '{} {} {} {} {}'.format(n,m,nS,form_heat,heat_load)
                        x_sol_sc[key_heat] = x_sol_h
                        errors_sc[key_heat] = err_vec_h
                        iters_sc[key_heat] = iters_h
                    gas_lf=False
                    elec_lf=False
                    heat_lf=False
    return x_sol_dict, errors_dict, iters_dict, x_sol_sc, errors_sc, iters_sc, tol

def plot_conv_mes_node_forms(x_sol, errors, iters, tol):
    """Create convergence plots for the dictionaries created by compare_conv_models"""
    max_iters_used = 0
    couplings_used = list()
    for key_mes, iter_mes in iters.items():
        max_iters_used = max(max_iters_used,iter_mes)
    for key_mes, err_vec in errors.items():
        key_list = key_mes.split(' ')
        n = int(key_list[0])
        m = int(key_list[1])
        nS = int(key_list[2])
        form_gas = key_list[3]
        hydr_eq = key_list[4]
        form_heat = key_list[5]
        heat_load = key_list[6]
        coupling_point = int(key_list[7])
        node_set = int(key_list[8])
        coupling_type = key_list[9]
        fig = plt.figure('conv_mes_n{}_m{}_nS{}_node{}_{}_{}_{}_{}'.format(n,m,nS,coupling_point,form_gas,hydr_eq,form_heat,heat_load))
        ax = fig.gca()
        ls = MES3N_line.linestyles_node_sets.get(node_set)
        color = MES3N_line.colors.get(coupling_type)
        marker = '.'
        ax.semilogy(err_vec,marker=marker,ls=ls,color=color)
        ax.set_xlabel(r'Iteration $k$')
        ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
        ax.semilogy([0,max_iters_used+1],[tol,tol],'k:',label='tolerance')
        ax.grid(which='major',color='k', linestyle='--', alpha=.2)
        ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
        xmin = 0
        xmax = max_iters_used
        xticks = range(xmin,xmax+1) # make sure the xticks are integers
        ax.set_xlim(left=xmin,right=xmax+1)
        ax.set_xticks(xticks)
    return ax

def plots_pscc_2020_forms(n,m,nS,dir_path,save_fig=False):
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
    elec_lf = True
    for hydr_eq_gas in ['fa','fb']:
        for form_gas in ['nodal','full']:
            gas_lf = True
            for form_heat in ['standard','half_link_flow']:
                for heat_load in ['outflow','delta']:
                    heat_lf = True
                    if not (hydr_eq_gas == 'fb' and form_gas == 'nodal'):
                        x_sol_dict, errors_dict, iters_dict, x_sol_sc_dict, errors_sc_dict, iters_sc_dict, tol = compare_conv_node_sets(n,m,nS,hydr_eq_gas=hydr_eq_gas,form_gas=form_gas,form_heat=form_heat,heat_load=heat_load,gas_lf=gas_lf,elec_lf=elec_lf,heat_lf=heat_lf)
                        x_sol.update(x_sol_dict)
                        errors.update(errors_dict)
                        iters.update(iters_dict)
                        x_sol_sc.update(x_sol_sc_dict)
                        errors_sc.update(errors_sc_dict)
                        iters_sc.update(iters_sc_dict)
                        gas_lf = False
                        elec_lf = False
                        heat_lf = False
    plot_conv_mes_node_forms(x_sol, errors, iters, tol)

    # plot convergence sc
    # print the iteration results, in order to put them in the table in the paper
    print('\nNR iterations single-carrier, n={}, m={}, nS={}'.format(n,m,nS))
    fig_conv_sc = plt.figure('conv_sc_n{}_m{}_nS{}'.format(n,m,nS))
    ax_conv_sc = fig_conv_sc.gca()
    max_iters_sc = 0
    for key, iter_sc in iters_sc.items():
        key_list = key.split(' ')
        n = int(key_list[0])
        m = int(key_list[1])
        nS = int(key_list[2])
        key_sc = ''.join(key.rsplit('{} {} {} '.format(n,m,nS)))
        fig_conv_gas = plt.figure('conv_gas_n{}_m{}_nS{}'.format(n,m,nS))
        fig_conv_elec = plt.figure('conv_elec_n{}_m{}_nS{}'.format(n,m,nS))
        fig_conv_heat = plt.figure('conv_heat_n{}_m{}_nS{}'.format(n,m,nS))
        if 'nodal' in key_sc or 'full' in key_sc: #gas
            ax = plt.figure('conv_gas_n{}_m{}_nS{}'.format(n,m,nS)).gca()
            ls = MES3N_line.linestyles_sc.get('gas')
            color = MES3N_line.colors_sc.get('gas')
            marker = MES3N_line.markers_gas.get(key_sc)
            label = key_sc
            print('gas {} {}'.format(key_sc,iter_sc))
        elif 'outflow' in key_sc or 'delta' in key_sc: #heat
            ax = plt.figure('conv_heat_n{}_m{}_nS{}'.format(n,m,nS)).gca()
            ls = MES3N_line.linestyles_sc.get('heat')
            color = MES3N_line.colors_sc.get('heat')
            marker = MES3N_line.markers_heat.get(key_sc)
            label = MES3N_line.labels_heat.get(key_sc)
            print('heat {} {}'.format(key_sc,iter_sc))
        else: #electrical
            ax = plt.figure('conv_elec_n{}_m{}_nS{}'.format(n,m,nS)).gca()
            ls = MES3N_line.linestyles_sc.get('elec')
            color = MES3N_line.colors_sc.get('elec')
            marker = '.'
            label = ''
            print('elec {}'.format(iter_sc))
        ax.semilogy(errors_sc.get(key),marker=marker,ls=ls,color=color,label=label)
        ax_conv_sc.semilogy(errors_sc.get(key),marker=marker,ls=ls,color=color,label=label)
        max_iters_sc = max(max_iters_sc,iter_sc)
    xmin = 0
    xmax = max_iters_sc
    xticks = range(xmin,xmax+1) # make sure the xticks are integers
    for fig_num in ['conv_gas_n{}_m{}_nS{}'.format(n,m,nS),'conv_elec_n{}_m{}_nS{}'.format(n,m,nS),'conv_heat_n{}_m{}_nS{}'.format(n,m,nS),'conv_sc_n{}_m{}_nS{}'.format(n,m,nS)]:
        ax = plt.figure(fig_num).gca()
        ax.set_xlabel(r'Iteration $k$')
        ax.set_ylabel(r'Error $||D_F F(x^k)||_2$')
        ax.semilogy([0,max_iters_sc+1],[tol,tol],'k:',label='tolerance')
        ax.grid(which='major',color='k', linestyle='--', alpha=.2)
        ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
        ax.set_xlim(left=xmin,right=xmax+1)
        ax.set_xticks(xticks)
        if not fig_num == 'conv_sc_n{}_m{}_nS{}'.format(n,m,nS):
            ax.legend()

    if save_fig:
        for fig_num in plt.get_figlabels():
            plt.figure(fig_num)
            path_to_fig = os.path.join(dir_path,'Figures','MES3N_streets')
            file_name = fig_num+'.pgf'
            plt.savefig(os.path.join(path_to_fig, file_name))

    print('\nNR iterations, n={}, m={}, nS={}, (CHP & GB+GG & GB+GG_vp & EH):'.format(n,m,nS))
    for coupling_point in [1,2]:
        for form_gas in ['nodal','full']:
            for hydr_eq_gas in ['fa','fb']:
                if not (hydr_eq_gas == 'fb' and form_gas == 'nodal'):
                    for form_heat in ['standard','half_link_flow']:
                        for heat_load in ['outflow','delta']:
                            for node_set in [1,2,3]:
                                if not (coupling_point == 1 and node_set == 3):
                                    for coupling_type in ['CHP','GB+GG','GB+GG_vp','EH']:
                                        key_CHP = '{} {} {} {} {} {} {} {} {} CHP'.format(n,m,nS,form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set)
                                        key_GB_GG = '{} {} {} {} {} {} {} {} {} GB+GG'.format(n,m,nS,form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set)
                                        key_GB_GG_vp = '{} {} {} {} {} {} {} {} {} GB+GG_vp'.format(n,m,nS,form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set)
                                        key_EH = '{} {} {} {} {} {} {} {} {} EH'.format(n,m,nS,form_gas,hydr_eq_gas,form_heat,heat_load,coupling_point,node_set)
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
    x_entries, xg_entries, xe_entries, unknown_delta_nodes, unknown_V_nodes, xh_entries, unknown_m_links, unknown_m_halflinks, unknown_p_nodes, unknown_Ts_nodes, unknown_Tr_nodes, unknown_Ts_halflinks, unknown_Tr_halflinks, xc_entries, unknown_qc_links, unknown_qc_halflinks, unknown_Pc_links, unknown_Pc_halflinks, unknown_Qc_links, unknown_Qc_halflinks, unknown_mc_links, unknown_mc_halflinks, unknown_dphi_links, unknown_dphic_halflinks, unknown_Ts_links, unknown_Tsc_halflinks, unknown_Tr_links, unknown_Trc_halflinks = het_net.get_x_entries(formulation=formulation)
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

def jacobians(n,m,number_of_streets,hydr_eq_gas='fb',heat_load='outflow',coupling_type='CHP',coupling_point=1,node_set=1,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},max_nodes=20):
    """Plot Jacobian matrices, and eigenvalue spectra, for different indices / ordering"""
    # initial guess
    if n == 1 and m == 0:
        p_perc_high = .99
        p_perc_low = .98
    else:
        p_perc_step = .001/(2*n-m)
        p_perc_high = 1-p_perc_step#.99
        p_perc_low = max(.1,1-p_perc_step*(n-m))#.97
    T_perc_step = .001/(2*n-m)
    T_perc_high = 1-T_perc_step#1.
    T_perc_low = max(.1,1-T_perc_step*(n-m))#1

    # create network
    gas_net,elec_net,heat_net,het_net = create_network(n,m,number_of_streets,hydr_eq_gas=hydr_eq_gas,heat_load=heat_load,coupling_type=coupling_type,coupling_point=coupling_point,node_set=node_set)
    # initialize
    x0 = initialize_network(het_net, gas_net, elec_net, heat_net,n,m,nS,p_perc_high,p_perc_low,T_perc_high,T_perc_low,formulation=formulation)

    # create system of equations
    from meslf.load_flow.system_of_equations import NonLinearSystemHeterogeneous
    # scaling
    pgbase = 50*bar
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

    max_iter = 50
    tol = 1e-6
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

def comp_solver_time_reod(n,m,number_of_streets,pgbase,qbase,Vbase,Sbase,phbase,mbase,Tbase,phibase,Egbase,p_perc_high,p_perc_low,T_perc_high,T_perc_low,deltabase=1,hydr_eq_gas='fb',heat_load='outflow',coupling_type='CHP',coupling_point=1,node_set=1,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},max_nodes=20,max_iter=150,maxmatvecs=5000,tol=1e-6):
    """Compare the time spent in the linear solver and the total time spent for different orderings and different linear solvers"""
    # solver information
    lin_solvers = ['solve','gmres','bicgstab']
    scale_var = 'matrix'
    scale_var_params = {'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase}
    max_iters_used = 0

    # create plot and lay-out
    fig = plt.figure('Convergence plot different solvers and permutations')
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||D_F F(x^k)||_2$')
    colors_perm = {'no':'tab:blue','x':'tab:red','xF':'tab:green'}
    markers_solver = {'solve':'.','gmres':'*','bicgstab':'x'}

    # create network
    gas_net,elec_net,heat_net,het_net = create_network(n,m,number_of_streets,hydr_eq_gas=hydr_eq_gas,heat_load=heat_load,coupling_type=coupling_type,coupling_point=coupling_point,node_set=node_set)
    # initialize
    x0 = initialize_network(het_net, gas_net, elec_net, heat_net,n,m,nS,p_perc_high,p_perc_low,T_perc_high,T_perc_low,formulation=formulation)

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
            x_sol,iters,err_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var=scale_var,scale_var_params=scale_var_params)
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

def comp_solver_time_reod(n,m,number_of_streets,hydr_eq_gas='fb',heat_load='outflow',coupling_type='CHP',coupling_point=1,node_set=1,formulation={'gas':'full','elec':'complex_power','heat':'standard','het':None},max_iter=15,maxmatvecs=5000,tol=1e-6,lin_solvers = ['solve','gmres','bicgstab']):
    """Compare the time spent in the linear solver and the total time spent for different orderings and different linear solvers"""
    # create plot and lay-out
    fig = plt.figure('Convergence plot different solvers and permutations, n={}, m={}, nS={}'.format(n,m,number_of_streets))
    ax = fig.gca()
    plt.xlabel('Iteration k')
    plt.ylabel(r'Error $||T_F F(x^k)||_2$')
    colors_perm = {'no':'tab:blue','x':'tab:red','xF':'tab:green'}
    markers_solver = {'solve':'.','gmres':'*','bicgstab':'x'}

    # initial guess
    if n == 1 and m == 0:
        p_perc_high = .99
        p_perc_low = .98
    else:
        p_perc_step = .001/(2*n-m)
        p_perc_high = 1-p_perc_step#.99
        p_perc_low = max(.1,1-p_perc_step*(n-m))#.97
    T_perc_step = .001/(2*n-m)
    T_perc_high = 1-T_perc_step#1.
    T_perc_low = max(.1,1-T_perc_step*(n-m))#1

    # create network
    gas_net,elec_net,heat_net,het_net = create_network(n,m,number_of_streets,hydr_eq_gas=hydr_eq_gas,heat_load=heat_load,coupling_type=coupling_type,coupling_point=coupling_point,node_set=node_set)
    # initialize
    x0 = initialize_network(het_net, gas_net, elec_net, heat_net,n,m,nS,p_perc_high,p_perc_low,T_perc_high,T_perc_low,formulation=formulation)

    # solver information
    pgbase = 50*bar
    qbase = 1.
    deltabase = 1.
    Vbase = 50*kV
    Sbase = MW
    phbase = bar
    mbase = 1.
    Tbase = 100.
    phibase = MW
    Egbase = phibase
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
            x_sol,iters,err_vec,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = het_net.solve_network(tol,max_iter,solver='NR',formulation=formulation,scale_var = 'matrix',scale_var_params = {'qbase':qbase,'pgbase':pgbase,'Sbase':Sbase,'Vbase':Vbase,'deltabase':deltabase,'mbase':mbase,'phbase':phbase,'Tbase':Tbase,'phibase':phibase,'Ebase':Egbase},P_F=P_F,P_x=P_x,lin_solver=lin_solver,max_iter_lin=maxmatvecs)
            ax.semilogy(err_vec,ls='-',color=colors_perm.get(perm_key),marker=markers_solver.get(lin_solver),label=lin_solver+', '+perm_key)
            max_iters_used = max(max_iters_used,iters)
            if lin_solver == 'solve' and err_vec[-1] > tol:
                print('Final error {:6.3e} is larger than tolerance {:6.3e}, after {} iters of {} max iters for solve. Other linear solvers are not tried'.format(err_vec[-1],tol,iters,max_iter))
                break
            else:
                print('Final error is {:6.3e} after {} iterations'.format(err_vec[-1],iters))
        if ind_P == 0 and lin_solver == 'solve' and err_vec[-1] > tol:
            print('NR did not convergence for first permutation. No use in checking others.\n')
            break

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
    # topology
    n = 1 #16 #(16 still seems to work, 17 not anymore (with m=7, p_perc_step = .001/..., and T_perc_step=0, and lam=0.02 for all pipes!))
    m = 0 #7 #(if n =16, then m=6 is too small (with p_perc_step = .001/..., and T_perc_step=0))
    nS =1 #20 #(if n=16, m=7, then nS=10 solves smoothly, nS=20 solves as well but not as smoothly, nS=21 just explodes. (with p_perc_step = .005/..., and T_perc_step=0))

    #legends()

    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #plots_pscc_2020_forms(n,m,nS,dir_path,save_fig=False)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only a",UserWarning)
        warnings.filterwarnings("ignore", "Changing the sparsity structure of a csr_matrix",sps.SparseEfficiencyWarning)
        jacobians(n,m,nS)
        comp_solver_time_reod(n,m,nS) #,lin_solvers=['solve'],maxmatvecs=1000

    plt.show()
