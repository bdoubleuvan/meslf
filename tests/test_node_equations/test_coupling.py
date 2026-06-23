"""Test the node equations for a (heterogeneous) coupling node"""
from meslf.node_equations import coupling
from meslf.utils.constants import MW
import numpy as np

def test_EH_Ein():
    """Test the incoming energy for an energy hub"""
    # Given
    C = np.array([[0.2, 0.1, 0.],
                 [0.4, 0.3, 0.1],
                 [0.4, 0.4, 0.8]])
    Eout = np.array([0.8, 1.9, 2.8])
    To = None
    Ein_carrier = ['g','e','h']
    Eout_carrier = ['g','e','h']

    # When
    Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTo, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTo, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTo, der_heat_power_eq_dTr  = coupling.EH(C,Ein_carrier,Eout_carrier)
    Ein = Ein_of_Eout(Eout,To)

    # Then
    Ein_expected = np.array([3, 2, 1])
    assert np.allclose(Ein_expected,Ein)

def test_EH_node_law_Ein():
    """Test the node law of the energy hub using the incoming energy as a function of outgoing energy"""
    # Given
    C = np.array([[0.2, 0.1, 0.],
                 [0.4, 0.3, 0.1],
                 [0.4, 0.4, 0.8]])
    Ein = np.array([3, 2, 1])
    Eout = np.array([0.8, 1.9, 2.8])
    To = None
    Ein_carrier = ['g','e','h']
    Eout_carrier = ['g','e','h']

    # When
    Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTo, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTo, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTo, der_heat_power_eq_dTr  = coupling.EH(C,Ein_carrier,Eout_carrier)
    f = node_law_Ein(Ein,Eout,To)

    # Then
    f_expected = np.array([0., 0., 0.])
    assert np.allclose(f_expected,f)

def test_EH_Eout():
    """Test the outgoing energy for an energy hub, with unequal number of incoming and outgoing carriers."""
    # Given
    C = np.array([[0.4],
                  [0.5]])
    Ein = np.array([2.])
    Ein_carrier = ['g']
    Eout_carrier = ['e','h']

    # When
    Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTo, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTo, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTo, der_heat_power_eq_dTr  = coupling.EH(C,Ein_carrier,Eout_carrier)
    Eout = Eout_of_Ein(Ein)

    # Then
    Eout_expected = np.array([[0.8, 1.]])
    assert np.allclose(Eout_expected,Eout)

def test_EH_node_law_Eout():
    """Test the node law of the energy hub using the outgoing energy as a function of incoming energy, with unequal number of incoming and outgoing carriers."""
    # Given
    C = np.array([[0.4],
                  [0.5]])
    Ein = np.array([2.])
    Eout = np.array([[0.8, 1.]])
    To = None
    Ein_carrier = ['g']
    Eout_carrier = ['e','h']

    # When
    Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTo, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTo, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTo, der_heat_power_eq_dTr  = coupling.EH(C,Ein_carrier,Eout_carrier)
    f = node_law_Eout(Ein,Eout,To)

    # Then
    f_expected = np.array([0.])
    assert np.allclose(f_expected,f)

def test_EH_Eout_full_C():
    """Test the outgoing energy for an energy hub, with unequal number of incoming and outgoing carriers, but using a full 3x3 coupling matrix."""
    # Given
    C = np.array([[0.,0.,0.],
                  [0.4,0.,0.],
                  [0.5,0.,0.]])
    Ein = np.array([2.,0.,0.])
    Ein_carrier = ['g','e','h']
    Eout_carrier = ['g','e','h']

    # When
    Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTo, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTo, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTo, der_heat_power_eq_dTr  = coupling.EH(C,Ein_carrier,Eout_carrier)
    Eout = Eout_of_Ein(Ein)

    # Then
    Eout_expected = np.array([[0.,0.8, 1.]])
    assert np.allclose(Eout_expected,Eout)

def test_EH_node_law_Eout_full_C():
    """Test the node law of the energy hub using the outgoing energy as a function of incoming energy, with unequal number of incoming and outgoing carriers, but using a full 3x3 coupling matrix."""
    # Given
    C = np.array([[0.,0.,0.],
                  [0.4,0.,0.],
                  [0.5,0.,0.]])
    Ein = np.array([2.,0.,0.])
    Eout = np.array([[0.,0.8, 1.]])
    To = None
    Ein_carrier = ['g','e','h']
    Eout_carrier = ['g','e','h']

    # When
    Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTo, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTo, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTo, der_heat_power_eq_dTr  = coupling.EH(C,Ein_carrier,Eout_carrier)
    f = node_law_Eout(Ein,Eout,To)

    # Then
    f_expected = np.array([0.,0.,0.])
    assert np.allclose(f_expected,f)

def test_CHP_part_load_Ein():
    """Test the incoming energy for a CHP with part load effect"""
    # Given
    a = .463
    b = -.04532
    d = 4.49
    eta = .88
    phimin = 10*MW #??
    phimax = 14*MW/.48 # maximum (?) active power / power-to-heat-ratio
    L1 = .8
    # Should stay above the upper limit L1, so these following values shouldn't matter
    L2 = .6
    r1 = .0736
    r2 = .0845
    Eout = np.array([10.1734,25])*MW #[W]
    To = 130

    # When
    Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTo, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTo, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTo, der_heat_power_eq_dTr  = coupling.geh_CHP_part_load(eta,a,b,d,L1,L2,r1,r2,phimin,phimax)
    Ein = Ein_of_Eout(Eout,To)

    # Then
    Ein_expected = 39.969772727272726*MW #E^g [W]
    assert np.allclose(Ein_expected,Ein)

def test_CHP_part_load_node_law():
    """Test the node law for a CHP with part load effect"""
    # Given
    a = .463
    b = -.04532
    d = 4.49
    eta = np.array([.88 ,.88])
    phimin = 10 #??
    phimax = 14/.48 # maximum (?) active power / power-to-heat-ratio
    L1 = .8
    # Should stay above the upper limit L1, so these following values shouldn't matter
    L2 = .6
    r1 = .0736
    r2 = .0845
    Eout = np.array([10.1734,25]) #[MW]
    Ein = 39.969772727272726 #E^g [MW]
    To = 130

    # When
    Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTo, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTo, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTo, der_heat_power_eq_dTr  = coupling.geh_CHP_part_load(eta,a,b,d,L1,L2,r1,r2,phimin,phimax)
    f = node_law_Ein(Ein,Eout,To)

    # Then
    f_expected = np.array([0.,0.])
    assert np.allclose(f_expected,f)

def test_CHP_part_load_node_law_dEin():
    """Test the derivative of the node law for a CHP with part load effect, wrt incoming energy flow"""
    # Given
    a = .463
    b = -.04532
    d = 4.49
    eta = np.array([.88 ,.88])
    phimin = 10 #??
    phimax = 14/.48 # maximum (?) active power / power-to-heat-ratio
    L1 = .8
    # Should stay above the upper limit L1, so these following values shouldn't matter
    L2 = .6
    r1 = .0736
    r2 = .0845
    Eout = np.array([10.1734,25]) #[MW]
    Ein = 39.969772727272726 #E^g [MW]
    To = 130

    # When
    Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTo, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTo, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTo, der_heat_power_eq_dTr  = coupling.geh_CHP_part_load(eta,a,b,d,L1,L2,r1,r2,phimin,phimax)
    df_dEin = der_node_law_Ein_dEin(Ein,Eout,To)

    # Then
    df_dEin_expected = np.array([[1.],[0.]])
    assert np.allclose(df_dEin_expected,df_dEin)

def test_CHP_part_load_node_law_dEout():
    """Test the derivative of the node law for a CHP with part load effect, wrt incoming energy flow"""
    # Given
    a = .463
    b = -.04532
    d = 4.49
    eta = np.array([.88 ,.88])
    phimin = 10 #??
    phimax = 14/.48 # maximum (?) active power / power-to-heat-ratio
    L1 = .8
    # Should stay above the upper limit L1, so these following values shouldn't matter
    L2 = .6
    r1 = .0736
    r2 = .0845
    Eout = np.array([10.1734,25]) #[MW]
    Ein = 39.969772727272726 #E^g [MW]
    To = 130

    # When
    Eout_of_Ein, Ein_of_Eout, dEout_dEin, dEin_dEout, node_law_Eout, node_law_Ein, der_node_law_Eout_dEout, der_node_law_Eout_dEin, der_node_law_Eout_dTo, der_node_law_Ein_dEout, der_node_law_Ein_dEin, der_node_law_Ein_dTo, heat_power_eq, der_heat_power_eq_dEout, der_heat_power_eq_dm, der_heat_power_eq_dTo, der_heat_power_eq_dTr  = coupling.geh_CHP_part_load(eta,a,b,d,L1,L2,r1,r2,phimin,phimax)
    df_dEout = der_node_law_Ein_dEout(Ein,Eout,To)
    print('df_dEout = {}'.format(df_dEout))

    # Then
    df_dEout_expected = np.vstack([-1/eta,np.array([1,-a])])
    assert np.allclose(df_dEout_expected,df_dEout)
