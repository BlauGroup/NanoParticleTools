from math import pi

# speed of light cm/s
c_CGS = 29979245800

# planck's constant, in erg s
h_CGS = 6.62606885E-27

# boltzmann constant, in erg/K
kb_CGS = 1.3806505E-16

# charge of electron, in Fr (erg^0.5 cm^0.5)
e_CGS = 0.000000000480320427

# mass of electron, in g
m_e_CGS = 9.10938188E-28

# magnetic moment of an electron in erg/Gauss
BOHR_MAGNETON_CGS = e_CGS*(h_CGS/(2*pi))/(2*m_e_CGS*c_CGS)

ANGULAR_MOMENTUM_SYMBOLS = "SPDFGHIKLMNOQRTUV"
