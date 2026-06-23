"""Some basic mathematical functionalities"""
import numpy as np
import cmath

def complex_polar(r,phi):
    """Makes a complex number z in Cartesian coordinates from the radius r and the phase phi.
    
    Parameters
    ----------
    r : float  
        Radius of the complex number in Polar coordinates. Must be nonnegative
    phi : float
        Phase angle in radians of the complex number in Polar coordinates. Must be in (-pi,pi]
    
    Returns
    -------
    z : float
        Complex number in Cartesian coordinates
        
    Raises
    ------
    ValueError
        If r < 0
    ValueError 
        If phi is not in (-pi,pi]
    """
    if np.any(r) < 0 :
        raise ValueError('r must be nonnegative, not {}'.format(r))
    if np.any(phi) <= -np.pi or np.any(phi) > np.pi:
        raise ValueError('phi must be in (-pi,pi], not {}.'.format(phi))
    return r*np.exp(phi*1j)

def polar(z):
    """Determines the radias and the phase angle of the complex number z in Cartesian coordinates
    
    Parameters
    ----------
    z : float
        Complex number in Cartesian coordinates
    
    Returns
    -------
    r : float  
        Radius of the complex number in Polar coordinates. Nonnegative
    phi : float
        Phase angle of the complex number in Polar coordinates. In radians and in (-pi,pi]
    """
    r = np.abs(z)
    phi = cmath.phase(z)
    return r,phi
