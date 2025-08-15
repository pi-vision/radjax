"""
Physical constants and unit conversions for RadJAX.

All constants are given in cgs units unless otherwise noted.
These include fundamental physical constants (c, k_B, h),
astronomical constants (pc, AU, M_sun, G), and molecular
weights relevant for astrophysical gas (CO, H, mean molecular weight).
"""
cc = 2.9979245800000e10           # Light speed         [cm/s]
kk = 1.3807e-16                   # Bolzmann's constant [erg/K]
mp = 1.6726e-24                   # Mass of proton      [g]
hh = 6.6262000e-27                # Planck constant 
khz = 1e3                         # Kilohertz
mhz = 1e6                         # Megahertz
ghz = 1e9                         # Gigahertz
kg = 1e3                          # Kilogram
erg = 1e-7                        # Joules
pc = 3.08572e18                   # Parsec [cm]
arcsec = 4.84814e-6               # arcsecond to radians
molecular_weight_co = 28.0        # g/mol
molecular_weight_h = 1.008        # g/mol
au = 14959787066000.              # astronomical unit in cm
M_sun = 1.9884e33                 # g
G = 6.67259e-8                    # cm3 gram-1 s-2
m_co = molecular_weight_co * mp  
m_h = molecular_weight_h * mp
mu = 2.34                         # mean molecular weight of the gas
m_mol_h = mu * m_h