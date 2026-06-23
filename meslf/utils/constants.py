"""Some constants"""
import numpy as np

# energy
GW = 1e9 #[W]
MW = 1e6 #[W]
kW = 1e3 #[W]
BTU = 0.2931 #[Wh]
MBTU = 1e3 #[BTU]

# voltages
kV = 1e3 #[V]
degree = np.pi/180 #[rad]

# capacitance
uF = 1e-6 #[F]
nF = 1e-9 #[F]

# pressures
bar = 1e5 #[Pa]
mbar = 1e2 #[Pa]
atm = 1.01*bar #[Pa]

# length
km = 1e3 #[m]
mm = 1e-3 #[m]
cm = 1e-2 #[m]

# volumes
MSCM = 1e3 #[SCM]

# time
hour = 3600 #[s]
day = 24*hour #[s]
year = 365*day*hour #[s]
