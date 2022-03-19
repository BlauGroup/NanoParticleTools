
# *** parameters for solving differential equations ***
# don't change these if you don't know what they are
ODE_SOLVE_METHOD = 3  # use 0 as default, 3 for stiff problems
TIME_STEP = 1e-4  # time step in seconds
NUM_TIME_STEPS = 100  # number of time steps per simulation
ODE_ERROR = 1e-12  # default is 1e-6

# kinetics parameters
ET_RATE_THRESHOLD = 0.1  # s^-1, lower limit for actually accounting for ET rate
RAD_RATE_THRESHOLD = 0.0001  # s^-1, lower limit for actually accounting for radiative rate

# crystal matrix parameters
PHONON_ENERGY = 450  # in wavenumbers
ZERO_PHONON_MPR_RATE = 1e7  # 3.0303E11#6.6e8# zero phonon relaxation rate at T=0, in 1/s
MPR_ALPHA_CONSTANT = 3.5e-3  # 0.008857143#5.01e-3# in cm

# host material parameters
INDEX_OF_REFRACTION = 1.5  # index of refraction
VOLUME_PER_DOPANT_SITE = 7.23946667e-2  # nm^3 for NaYF4, 1.5 is #possible sites for dopant ion
MINIMUM_DOPANT_DISTANCE = 3.867267554e-8  # cm minimum distance between dopants

# default dopant species parameters
NUMLEVELS = 21
RE_CONC = 5  # %

# laser excitation parameters
INCIDENT_POWER = 1e7  # 1e7 erg/s/cm^2 = 1 W/cm^2
INCIDENT_WAVELENGTH = 976  # nm

# energy transfer

ET_MODE = 0  # 0=no migration, 1 = fast diffusion, 2 = fast hopping
STOKES_SHIFT = 150  # wavenumbers
SK_COMBINE_ET_BACK_TRANSFER = 0

# new constants

INITIAL_POP_MODE = 0  # initial state of dopants all in ground state at start of simulation

INIT_POPS_ALL_GROUND_STATE = 0
INIT_POPS_FROM_LAST_RUN_FINAL = 1  # use the W_final population from the last run
INIT_POPS_ABS_FROM_USER = 2
# constant INIT_POPS_FRXN_FROM_USER = -1 # DEPRECATED. DO NOT USE
INIT_POPS_DEFAULT = 0

STATUS_SUCCESS = 1  # indicates a function successful completion
STATUS_FAIL = -9999999999999  # indicates a function unsuccessful completion

c_CGS = 29979245800  # speed of light cm/s
h_CGS = 6.62606885E-27  # planck's constant, in erg s
kb_CGS = 1.3806505E-16 # boltzmann constant, in erg/K
e_CGS = 0.000000000480320427 # charge of electron, in Fr (erg^0.5 cm^0.5)
m_e_CGS = 9.10938188E-28 # mass of electron, in g
BOHR_MAGNETON_CGS = 9.27401E-21 # in erg/G or erg^1.5 cm^-0.5 s^2 g^-1, g^0.5 cm^2.5 s^-1,  = e_CGS*(h_CGS/(2*pi))/(2*m_e_CGS*c_CGS)
ANGULAR_MOMENTUM_SYMBOLS = "SPDFGHIKLMNOQRTUV"