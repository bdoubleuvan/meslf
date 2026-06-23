"""Test the thermal half link equations"""
from meslf.half_link_equations import thermal
from meslf.networks.carrier import Water
import numpy as np 

def test_phi_heat_exchanger_source():
    """Test the heat power for a source heat exchanger
    """
    # Given
    Cp = 4.182e3 #[J/(kg K)]
    water = Water('water',Cp)
    m = -4. #[kg/s]
    Tr = 50. #[Celsius]
    To = 100. #[Celsius]
    
    m_of_phi, phi_of_m, dm_dTs, dm_dTr, dphi_dm, dphi_dTs, dphi_dTr = thermal.heat_exchanger(water)
    
    # When
    phi = phi_of_m(m,To,Tr)
    
    # Then
    phi_expected = -836400.
    assert np.isclose(phi_expected,phi)
    
