
# *** parameters for solving differential equations ***
# don't change these if you don't know what they are
ODE_SOLVE_METHOD = 3  # use 0 as default, 3 for stiff problems

# host material parameters
VOLUME_PER_DOPANT_SITE = 7.23946667e-2  # nm^3 for NaYF4, 1.5 is #possible sites for dopant ion

# energy transfer

#TODO: SK_COMBINE_ET_BACK_TRANSFER = 0

c_CGS = 29979245800  # speed of light cm/s
h_CGS = 6.62606885E-27  # planck's constant, in erg s
kb_CGS = 1.3806505E-16 # boltzmann constant, in erg/K
e_CGS = 0.000000000480320427 # charge of electron, in Fr (erg^0.5 cm^0.5)
m_e_CGS = 9.10938188E-28 # mass of electron, in g
BOHR_MAGNETON_CGS = 9.27401E-21 # in erg/G or erg^1.5 cm^-0.5 s^2 g^-1, g^0.5 cm^2.5 s^-1,  = e_CGS*(h_CGS/(2*pi))/(2*m_e_CGS*c_CGS)
ANGULAR_MOMENTUM_SYMBOLS = "SPDFGHIKLMNOQRTUV"