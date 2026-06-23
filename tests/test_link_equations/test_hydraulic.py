"""Test the hydraulic link flow equations"""

from meslf.link_equations import hydraulic
from meslf.networks.carrier import Gas, Water
import numpy as np

bar = 1e5 #[Pa]
mbar = 1e2 #[Pa]
hour = 3600 #[s]
km = 1e3 #[m]
cm = 1e-2 #[m]
mm = 1e-3 #[m]
atm = 1.01*bar #[Pa]
    
def test_pipe_dummy():
    """Test the pressure drop equation of a dummy link
    """
    # Given
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.dummy()
    p1 = 30.*mbar
    p2 = 25.*mbar

    # When
    pres_drop = dp(p1,p2)
    pres_drop_expected = 5.*mbar

    # Then
    assert np.isclose(pres_drop,pres_drop_expected)

def test_pipe_low_pressure_pres_drop():
    """Test the pressure drop as function of flow for a low pressure pipe, assuming water as carrier (based on email conversation with Johan on 22-03-2019).
    """
    # Given
    # pipe parameters
    L = 1*km #[m]
    D = 0.2 #[m]
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1e3 #[kg/m^3]
    carrier = Water('water',Cp,rho=rho)
    def fric_fac_func(carrier,D,L,*args,scale_var=None,scale_var_params=None):
        return 0.016/4
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,fric_fac_func,D,L)
    p1 = 5.*bar #[Pa]
    v = 1. #[m/s]
    m = np.pi*D**2*v*rho/4
    print('m = {}'.format(m))

    #When
    pres_drop = dp_of_q(m)

    # Then
    pres_drop_expected = 0.4*bar #[Pa]
    rel_tol = 1e-6
    assert np.isclose(pres_drop,pres_drop_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_low_pressure_flow():
    """Test the flow as function of pressure drop for a low pressure pipe, assuming water as carrier (based on email conversation with Johan on 22-03-2019).
    """
    # Given
    # pipe parameters
    L = 1*km #[m]
    D = 0.2 #[m]
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1e3 #[kg/m^3]
    carrier = Water('water',Cp,rho=rho)
    def fric_fac_func(carrier,D,L,*args,scale_var=None,scale_var_params=None):
        return 0.016/4
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,fric_fac_func,D,L)
    p1 = 5.*bar #[Pa]
    p2 = 4.6*bar #[Pa]

    #When
    m = q_of_dp(p1,p2)

    # Then
    v_expected = 1. #[m/s]
    m_expected = np.pi*D**2*v_expected*rho/4
    rel_tol = 1e-6
    assert np.isclose(m,m_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_low_pressure_pole():
    """Test the flow equation for a low pressure system, using gas as a carrier. Values taken from Osiadacz.
    """
    # Given
    L = 680. #m
    D = .150 #[m]
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,hydraulic.fric_fac_pole,D,L)
    mbar = 1e2 #[Pa]
    hour = 3600 #[s]
    p1 = 30.*mbar
    p2 = 25.*mbar

    #When
    mass_flow = q_of_dp(p1,p2) #[kg/s]
    q = mass_flow/carrier.rhon*hour #[m^3/h]
    q_expected = 218.45689801329863 #[m^3/h]
    m_expected = q_expected*carrier.rhon/hour #[kg/s]

    #Then
    rel_tol = 1e-2
    assert np.isclose(mass_flow,m_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_low_pressure_hagen_poiseuille():
    """Test the flow equation for a low pressure system, using gas as a carrier. Values taken from Osiadacz.
    """
    # Given
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    rhon_g = carrier.rhon
    dyn_vis = 1e-5 #[Pa/s] = [kg/(m s)]
    kin_vis = dyn_vis/rhon_g #[m^s/s]
    carrier.mu = kin_vis # about 1.4e-5
    # pipe
    L = 100 #[m]
    D = 5*cm #[m]
    eps = 0 #[not used in friction factor, but needed as function argument)
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,hydraulic.fric_fac_hagen_poiseuille,D,L,eps,fric_fac_der_q=hydraulic.fric_fac_der_qhagen_poiseuille)
    p1 = 0.1*mbar #[Pa]
    m = 7e-5 #[kg/s]

    #When
    pres_drop = dp_of_q(m) #[Pa]
    p2 = p1 - pres_drop

    #Then
    p2_expected = .093596508*mbar
    rel_tol = 1e-6
    assert np.isclose(p2,p2_expected,atol=rel_tol*1e-1,rtol=rel_tol)
    
def test_pipe_low_pressure_pole_scaled():
    """Test the flow equation for a low pressure system
    """
    # Given
    L = 680. #m
    D = .150 #[m]
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,hydraulic.fric_fac_pole,D,L)
    mbar = 1e2 #[Pa]
    hour = 3600 #[s]
    p1 = 30.*mbar
    p2 = 25.*mbar

    #When
    pbase = 1*mbar
    qbase = 0.01 #[kg/s]
    scale_var = 'per_unit'
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    mass_flow = q_of_dp(p1/pbase,p2/pbase,scale_var=scale_var,scale_var_params=scale_var_params) #[kg/s]

    #Then
    q_expected = 218.45689801329863 #[m^3/h]
    m_expected = q_expected*carrier.rhon/hour #[kg/s]
    m_expected /= qbase #[p.u.]
    rel_tol = 1e-2
    assert np.isclose(mass_flow,m_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_low_pressure_pole_der_dp():
    """Test the the derivative of the link equation to pressure drop function, for a low pressure system, with gas as a carrier.
    """
    # Given
    L = 680. #m
    D = .150 #[m]
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,hydraulic.fric_fac_pole,D,L)
    mbar = 1e2 #[Pa]
    hour = 3600 #[s]
    p1 = 30.*mbar
    p2 = 25.*mbar

    #When
    df_ddp = dfa_ddp(0,p1,p2)

    #Then
    df_ddp_expected = -4.34517767e-5
    rel_tol = 1e-6
    assert np.isclose(df_ddp,df_ddp_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_low_pressure_pole_der_dp_scaled():
    """Test the the derivative of the link equation to pressure drop function, for a low pressure system, with gas as a carrier.
    """
    # Given
    L = 680. #m
    D = .150 #[m]
    # carrier
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('low pres gas',S,R_air,1,pn,Tn,Tn)
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,hydraulic.fric_fac_pole,D,L)
    mbar = 1e2 #[Pa]
    hour = 3600 #[s]
    p1 = 30.*mbar
    p2 = 25.*mbar
    pbase = 1*mbar
    qbase = 0.01 #[kg/s]
    scale_var = 'per_unit'
    scale_var_params = {'qbase':qbase,'pbase':pbase}

    #When
    p1_pu = p1/pbase
    p2_pu = p2/pbase
    df_ddp = dfa_ddp(0,p1_pu,p2_pu,scale_var=scale_var,scale_var_params=scale_var_params)

    #Then
    df_ddp_expected = -4.34517767e-5*pbase/qbase
    rel_tol = 1e-3
    assert np.isclose(df_ddp,df_ddp_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_low_pressure_der_dq():
    """Test the the derivative of the link equation to flow, for a low pressure system, with water as a carrier.
    """
    km = 1e3 #[m]
    bar = 1e5 #[Pa]
    # pipe parameters
    L = 1*km #[m]
    D = 0.2 #[m]
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1e3 #[kg/m^3]
    carrier = Water('water',Cp,rho=rho)
    def fric_fac_func(carrier,D,L,*args,scale_var=None,scale_var_params=None):
        return 0.016/4
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,fric_fac_func,D,L)
    v = 1. #[m/s]
    m = np.pi*D**2*v*rho/4

    #When
    pbase = 1*bar
    qbase = 10 #[kg/s]
    scale_var = 'per_unit'
    scale_var_params = {'qbase':qbase,'pbase':pbase}
    m_pu = m/qbase
    df_dq = dfb_dq(m_pu,0,0,scale_var=scale_var,scale_var_params=scale_var_params)

    #Then
    df_dq_expected = -2546.479089*qbase/pbase
    rel_tol = 1e-6
    assert np.isclose(df_dq,df_dq_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_low_pressure_der_dq_scaled():
    """Test the the derivative of the link equation to flow, for a low pressure system, with water as a carrier.
    """
    km = 1e3 #[m]
    bar = 1e5 #[Pa]
    # pipe parameters
    L = 1*km #[m]
    D = 0.2 #[m]
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1e3 #[kg/m^3]
    carrier = Water('water',Cp,rho=rho)
    def fric_fac_func(carrier,D,L,*args,scale_var=None,scale_var_params=None):
        return 0.016/4
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure(carrier,fric_fac_func,D,L)
    v = 1. #[m/s]
    m = np.pi*D**2*v*rho/4

    #When
    df_dq = dfb_dq(m,0,0)

    #Then
    df_dq_expected = -2546.479089
    rel_tol = 1e-6
    assert np.isclose(df_dq,df_dq_expected,atol=rel_tol*1e-1,rtol=rel_tol)
    
def test_pipe_high_pressure_colebrook():
    """Test the flow equation for a high pressure system with Colebrook friction factor, assuming gas as carrier
    """
    bar = 1e5 #[Pa]
    hour = 3600 #[s]
    L = 30000. #[m]
    D = 0.150 #[m]
    #carrier
    mu = 0.288e-6 #[m^2/s]
    eps = 0.05*1e-3 #[m]
    Z = 0.8
    T = 281.15 #[K]
    S = 0.6106
    Tn = 273.15 #[K]
    pn = 1.01325*bar #[Pa]
    R = 8.314459848 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T,mu=mu)
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure_implicit(carrier,hydraulic.fric_fac_colebrook,D,L,eps,fric_fac_der_q=hydraulic.fric_fac_der_qcolebrook)
    q = 16407.759522180681 #[SCM^3/h]
    m = q*carrier.rhon/hour #[kg/s]

    #When
    pres_drop = dp_of_q(m) #[Pa^2]
    
    #Then
    p1 = 50.*bar
    p2 = 34.07703674*bar
    pres_drop_expected = p1**2 - p2**2
    rel_tol = 1e-5
    assert np.isclose(pres_drop,pres_drop_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_high_pressure_weymouth():
    """Test the flow equation for a high pressure system with Weymouth friction factor, assuming gas as carrier (based on pipe 1 of example 6.7.4a of Osiadacz).
    """
    bar = 1e5 #[Pa]
    hour = 3600 #[s]
    L = 24000. #[m]
    D = 0.7 #[m]
    # carrier
    Z = 0.93323990580143557
    T = 283 #[K]
    S = 0.6
    Tn = 288 #[K]
    pn = 1.*bar #[Pa]
    R = 8.314413 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure(carrier,hydraulic.fric_fac_weymouth,D,L,E=0.98528057165559679)
    p1 = 3.921*10*bar
    p2 = 3.738*10*bar

    #When
    mass_flow = q_of_dp(p1,p2) #[kg/s]

    #Then
    q_expected = 90.989 #[m^3/s]
    m_expected = q_expected*carrier.rhon #[kg/s]
    rel_tol = 5e-2 #(The big tolerance is needed because the values reported in Osiadacz are probably not completely accurate)
    assert np.isclose(mass_flow,m_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_high_pressure_weymouth_2():
    """Test the flow equation for a high pressure system with Weymouth friction factor, assuming gas as carrier (based on email conversation with Johan on 22-03-2019).
    """
    bar = 1e5 #[Pa]
    hour = 3600 #[s]
    km = 1e3 #[m]
    # pipe parameters
    L = 10.*km #[m]
    D = 0.3 #[m]
    # carrier
    Z = 0.96
    T = 300 #[K]
    S = 0.7
    Tn = 288 #[K]
    pn = 1.013*bar #[Pa]
    R = 8.314 #[J/molK]
    M = 29e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    rhon_g = carrier.rhon
    dyn_vis = 1e-5 #[Pa/s] = [kg/(m s)]
    kin_vis = dyn_vis/rhon_g #[m^2/s]
    carrier.mu = kin_vis
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure(carrier,hydraulic.fric_fac_weymouth,D,L,E=1)
    p1 = 32*bar
    p2 = 30*bar

    #When
    mass_flow = q_of_dp(p1,p2) #[kg/s]

    #Then
    q_expected = 1.066e6 #[m^3/dag]
    q_expected /= (24*hour) #[m^3/s]
    m_expected = q_expected*carrier.rhon #[kg/s]
    rel_tol = 1e-3
    assert np.isclose(mass_flow,m_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_high_pressure_weymouth_scaled():
    """Test the flow equation for a high pressure system with Weymouth friction factor and using per unit scaling, assuming gas as carrier (based on pipe 1 of example 6.7.4a of Osiadacz.
    """
    bar = 1e5 #[Pa]
    hour = 3600 #[s]
    L = 24000. #[m]
    D = 0.7 #[m]
    # carrier
    Z = 0.93323990580143557
    T = 283 #[K]
    S = 0.6
    Tn = 288 #[K]
    pn = 1.*bar #[Pa]
    R = 8.314413 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure(carrier,hydraulic.fric_fac_weymouth,D,L,E=0.98528057165559679)
    # scaling
    scale_var = 'per_unit'
    qbase = 100.*carrier.rhon
    pbase = 3.*10*bar
    scale_var_params = {'qbase':qbase,'pbase':pbase}

    p1 = 3.921*10*bar #[Pa]
    p2 = 3.738*10*bar #[Pa]
    p1_pu = p1/pbase #[p.u.]
    p2_pu = p2/pbase #[p.u.]

    #When
    mass_flow_pu = q_of_dp(p1_pu,p2_pu,scale_var=scale_var,scale_var_params=scale_var_params) #[p.u.]
    mass_flow = mass_flow_pu*qbase #[kg/s]


    #Then
    q_expected = 90.989 #[m^3/s]
    m_expected = q_expected*carrier.rhon #[kg/s]
    rel_tol = 5e-2 #(The big tolerance is needed because the values reported in Osiadacz are probably not completely accurate)
    assert np.isclose(mass_flow,m_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_high_pressure_pole():
    """Test the flow equation for a low pressure system with Pole friction factor, assuming gas as carrier and using the 'high pressure equation', which is the standard steady-state flow equation.
    """
    # carrier
    Z = 1
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    T = Tn
    R = 8.314 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    # pipe parameters
    L = 100 #[m]
    D = 5*cm #[m]
    link_type = 'pipe_high_pres_pole'
    link_params = {'carrier':carrier, 'D':D, 'L':L}
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure(carrier,hydraulic.fric_fac_pole,D,L)
    p1 = 0.1*mbar+atm #[Pa]
    p2 = .09954*mbar+atm #[Pa]

    #When
    mass_flow = q_of_dp(p1,p2) #[kg/s]

    #Then
    m_expected = 7e-5 #[kg/s]
    rel_tol = 1e-6
    assert np.isclose(mass_flow,m_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_high_pressure_pole_dp():
    """Test the flow equation for a low pressure system with Pole friction factor, assuming gas as carrier and using the 'high pressure equation', which is the standard steady-state flow equation.
    """
    # carrier
    Z = 1
    S = 0.589
    pn = 0.1e6 #[Pa]
    Tn = 288. #[K]
    T = Tn
    R = 8.314 #[J/molK]
    M = 28.97e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    # pipe parameters
    L = 100 #[m]
    D = 5*cm #[m]
    link_type = 'pipe_high_pres_pole'
    link_params = {'carrier':carrier, 'D':D, 'L':L}
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure(carrier,hydraulic.fric_fac_pole,D,L)
    m = 7e-5 #[kg/s]

    #When
    pres_drop = dp_of_q(m) #[Pa], absolute pressure
    p1 = 0.1*mbar + atm #[Pa], absolute pressure
    p2 = np.sqrt(p1**2-pres_drop) - atm #[Pa], gauge pressure
    #Then
    p2_expected = .09954*mbar #[Pa]
    rel_tol = 1e-5
    assert np.isclose(p2,p2_expected,atol=rel_tol*1e-1,rtol=rel_tol)

def test_pipe_high_pressure_churchill():
    """Test the flow equation for a high pressure system with Churchill friction factor, assuming gas as carrier (based on email conversation with Johan on 22-03-2019, who used Weymouth).
    """
    # pipe parameters
    L = 10.*km #[m]
    D = 0.3 #[m]
    eps = 0.02*mm
    # carrier
    Z = 0.96
    T = 300 #[K]
    S = 0.7
    Tn = 288 #[K]
    pn = 1.013*bar #[Pa]
    R = 8.314 #[J/molK]
    M = 29e-3 #[kg/mol]
    R_air = R/M #[J/kgK]
    carrier = Gas('high pres gas',S,R_air,Z,pn,Tn,T)
    rhon_g = carrier.rhon
    dyn_vis = 1e-5 #[Pa/s] = [kg/(m s)]
    kin_vis = dyn_vis/rhon_g #[m^s/s]
    carrier.mu = kin_vis
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_high_pressure_implicit(carrier,hydraulic.fric_fac_churchill,D,L,eps,fric_fac_der_q=hydraulic.fric_fac_der_qchurchill)
    
    q = 1.066e6 #[m^3/dag]
    q /= (24*hour) #[m^3/s]
    m = q*carrier.rhon #[kg/s]
    
    #When
    pres_drop = dp_of_q(m) #[Pa^2]
    p1 = 32*bar #[Pa]
    p2 = np.sqrt(p1**2 - pres_drop)/bar #[bar]

    #Then
    p2_expected = 30 #[bar]
    rel_tol = 1e-1
    abs_tol = rel_tol
    assert np.isclose(p2,p2_expected,atol=abs_tol,rtol=rel_tol)
    
def test_pipe_low_pressure_colebrook():
    """Test the flow equation for a low pressure system with Colebrook friction factor, assuming water as carrier
    """
    # Given
    #carrier
    Cp = 4.182e3 #[J/(kg K)]
    rho = 960. #[kg/m^3]
    mu = 0.294e-6 #[m^2/s]
    carrier = Water('low pres water',Cp,rho=rho,mu=mu) # use default value for gravitational constant g
    # link parameters
    L = 30000. #[m]
    D = 0.150 #[m]
    eps = 1.25*1e-3 #[m]
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,hydraulic.fric_fac_colebrook,D,L,eps,fric_fac_der_q=hydraulic.fric_fac_der_qcolebrook)
    # flow
    m = 1. #[kg/s]

    #When
    press_drop = dp_of_q(m) #[Pa]

    #Then
    press_drop_expected = 12609.845907566867
    assert np.isclose(press_drop,press_drop_expected)

def test_pipe_low_pressure_colebrook_scaled():
    """Test the flow equation for a low pressure system with Colebrook friction factor, assuming water as carrier, using per unit scaling
    """
    # Given
    #carrier
    Cp = 4.182e3 #[J/(kg K)]
    rho = 960. #[kg/m^3]
    mu = 0.294e-6 #[m^2/s]
    carrier = Water('low pres water',Cp,rho=rho,mu=mu) # use default value for gravitational constant g
    # link parameters
    L = 30000. #[m]
    D = 0.150 #[m]
    eps = 1.25*1e-3 #[m]
    pipe_const, dp, fa, fb, q_of_dp, dp_of_q, ddp_dp, ddp_dq, dq_ddp, dfa_ddp, dfb_ddp, dfa_dp, dfb_dp, dfa_dq, dfb_dq = hydraulic.pipe_low_pressure_implicit(carrier,hydraulic.fric_fac_colebrook,D,L,eps,fric_fac_der_q=hydraulic.fric_fac_der_qcolebrook)

    # scaling
    pbase = rho*carrier.g
    mbase = 10.
    scale_var_params = {'qbase':mbase,'pbase':pbase}

    # flow
    m = 1. #[kg/s]
    m_pu = m/mbase #[p.u.]

    #When
    press_drop_pu = dp_of_q(m_pu,scale_var='per_unit',scale_var_params=scale_var_params) #[p.u.]
    press_drop = press_drop_pu*pbase #[Pa]

    #Then
    press_drop_expected = 12609.845907566867 #[Pa]
    assert np.isclose(press_drop,press_drop_expected)
