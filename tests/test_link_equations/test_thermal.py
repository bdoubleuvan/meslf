"""Test the thermal link equations"""
from meslf.link_equations import thermal
from meslf.networks.carrier import Water
import numpy as np

def test_dummy():
    """Test the end supply temperature for a dummy link
    """
    # Given
    temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.dummy()
    Tsstart = 100. #[Celsius]
    m = 1 #[kg/s]

    # When
    Tsend = Tend_of_Tstart(m,Tsstart)

    # Then
    Tsend_expected = 100.
    assert Tsend == Tsend_expected

def test_perfect_isolated_pipe():
    """Test the end supply temperature for a perfectly isolated pipe
    """
    # Given
    temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.perfect_isolated_pipe()
    Tsstart = 100. #[Celsius]
    m = 1 #[kg/s]

    # When
    Tsend = Tend_of_Tstart(m,Tsstart)

    # Then
    Tsend_expected = 100.
    assert Tsend == Tsend_expected

def test_standard_pipe_Tsend_Ta_zero():
    # Given
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1000 #[kg/m^3]
    water = Water('water',Cp,rho=rho)
    L = 1e3 #[m]
    D = 0.2 #[m]
    U = 10 #[W/(m^2 K)
    temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(water,U,L,D,Ta=0)
    Tsstart = 70. #[C]
    v = 1 #[m/s]
    m = np.pi*D**2/4 * v*rho #[kg/s]

    # When
    Tsend = Tend_of_Tstart(m,Tsstart)

    # Then
    Tsend_expected = 66.73 #[C]
    assert np.isclose(Tsend_expected,Tsend)

def test_standard_pipe_Tsend():
    # Given
    Cp = 4.18e3 #[J/(kg K)]
    rho = 1000 #[kg/m^3]
    water = Water('water',Cp,rho=rho)
    L = 1e3 #[m]
    D = 0.2 #[m]
    U = 10 #[W/(m^2 K)
    Ta = 20 #[C]
    temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(water,U,L,D,Ta=Ta)
    Tsstart = 70. #[C]
    v = 1 #[m/s]
    m = np.pi*D**2/4 * v*rho #[kg/s]

    # When
    Tsend = Tend_of_Tstart(m,Tsstart)

    # Then
    Tsend_expected = 67.664 #[C]
    assert np.isclose(Tsend_expected,Tsend)
    
def test_standard_pipe_Trend_Ta_zero():
    # Given
    Cp = 4.182e3 #[J/(kg K)]
    water = Water('water',Cp)
    L = 400. #[m]
    D = 0.01 #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D)
    temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(water,U,L,D,Ta=0)
    Trstart = 50. #[Celsius]
    m = 1. #[kg/s]

    # When
    Trend = Tend_of_Tstart(m,Trstart)

    # Then
    Trend_expected = 49.052610331719194
    assert np.isclose(Trend_expected,Trend)

def test_standard_pipe_scaled_Tsend():
    # Given
    rho = 960. #[kg/m^3]
    Cp = 4.182e3 #[J/(kg K)]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp,rho=rho)
    Ta = 10.
    L = 400. #[m]
    D = 0.01 #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D)
    temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(water,U,L,D,Ta=Ta)
    Tsstart = 100. #[Celsius]
    m = 4. #[kg/s]
    scale_var = 'per_unit'
    Tbase = 80. #[C]
    phibase = 1e6 #[W]
    mbase = phibase/(water.Cp*Tbase)
    pbase = mbase**2*rho*g
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}

    # When
    Tsend = Tend_of_Tstart(m/mbase,Tsstart/Tbase,scale_var=scale_var,scale_var_params=scale_var_params)

    # Then
    Tsend_expected = 1.2446326437932724 #[p.u.]
    assert np.isclose(Tsend_expected,Tsend)

def test_standard_pipe_scaled_Trend():
    # Given
    rho = 960. #[kg/m^3]
    Cp = 4.182e3 #[J/(kg K)]
    g = 9.81 #[m/s^2]
    water = Water('water',Cp,rho=rho)
    Ta = 10.
    L = 400. #[m]
    D = 0.01 #[m]
    lam = 0.2 #[W/(mK)]
    U = lam/(np.pi*D)
    temp_drop_fac, temp_drop_fac_dm, Tend_of_Tstart, dTend_dm, dTend_dTstart = thermal.standard_pipe(water,U,L,D,Ta=Ta)
    Trstart = 49.#[Celsius]
    m = 4. #[kg/s]
    scale_var = 'per_unit'
    Tbase = 80. #[C]
    phibase = 1e6 #[W]
    mbase = phibase/(water.Cp*Tbase)
    pbase = mbase**2*rho*g
    scale_var_params = {'qbase':mbase,'pbase':pbase,'phibase':phibase,'Tbase':Tbase}

    # When
    Trend = Tend_of_Tstart(m/mbase,Trstart/Tbase,scale_var=scale_var,scale_var_params=scale_var_params)

    # Then
    Trend_expected = 0.6101741456437513
    assert np.isclose(Trend_expected,Trend)
