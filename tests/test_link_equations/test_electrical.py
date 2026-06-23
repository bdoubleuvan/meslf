"""Test the electrical link equations"""
from meslf.link_equations import electrical
import numpy as np

def test_short_line_Pstart():
    """Test the active power at the start of a line for a short line
    """
    # Given
    V0 = 1. #[p.u.]
    delta0 = 0.  
    V1 = 0.98 #[p.u.]
    delta1 = -0.1
    
    b = -10. #[p.u.]
    g = 1. #[p.u.]
    
    # When 
    Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.short_line(b,g)
    Pstart = Pstart_of_V_delta(V0,V1,delta0,delta1,scale_var=None,scale_var_params=None)
    
    # Then 
    Pstart_expected = 1.0032634011664505
    assert np.isclose(Pstart_expected,Pstart)
    
def test_short_line_Qstart():
    """Test the reactive power at the start of a line for a short line
    """
    # Given
    V0 = 1. #[p.u.]
    delta0 = 0.  
    V1 = 0.98 #[p.u.]
    delta1 = -0.1
    
    b = -10. #[p.u.]
    g = 1. #[p.u.]
    
    # When 
    Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.short_line(b,g)
    Qstart = Qstart_of_V_delta(V0,V1,delta0,delta1,scale_var=None,scale_var_params=None)
    
    # Then 
    Qstart_expected = 0.15112243196145592
    assert np.isclose(Qstart_expected,Qstart)
    
def test_short_line_Pend():
    """Test the active power at the end of a line for a short line
    """
    # Given
    V0 = 1. #[p.u.]
    delta0 = 0.  
    V1 = 0.98 #[p.u.]
    delta1 = -0.1
    
    b = -10. #[p.u.]
    g = 1. #[p.u.]
    
    # When 
    Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.short_line(b,g)
    Pend = Pend_of_V_delta(V0,V1,delta0,delta1,scale_var=None,scale_var_params=None)
    
    # Then 
    Pend_expected = -0.9930715651113813
    assert np.isclose(Pend_expected,Pend)
    
def test_short_line_Qend():
    """Test the reactive power at the end of a line for a short line
    """
    # Given
    V0 = 1. #[p.u.]
    delta0 = 0.  
    V1 = 0.98 #[p.u.]
    delta1 = -0.1
    
    b = -10. #[p.u.]
    g = 1. #[p.u.]
    
    # When 
    Pstart_of_V_delta, Qstart_of_V_delta, Pend_of_V_delta, Qend_of_V_delta = electrical.short_line(b,g)
    Qend = Qend_of_V_delta(V0,V1,delta0,delta1,scale_var=None,scale_var_params=None)
    
    # Then 
    Qend_expected = -0.049204071410763106
    assert np.isclose(Qend_expected,Qend)
