# GP NMPC problem setup
import numpy as np
from casadi import *

def specifications():
    ''' Specify Problem parameters '''
    tf              = 240.     # final time
    nk              = 12       # sampling points
    ndat0           = 50       # number of points sampled initially
    x0              = np.array([1.,150.,0.])
    backoff_repeats = 16       # repeats of back-off
    MC_n_iter       = 100      # MC iterations of plant
    backoff_MC      = 1000     # MC simulations per back-off computation
    learning        = True     # Learning
    Lsolver         = 'ma27'   # Linear solver
    c_code          = False    # c_code
    state_dep       = False    # state_dep
    multi_hyper     = 1        # multistart on GP hyperparameters
    filter_par      = 1.       # filter to update backoffs

    return nk, ndat0, tf, x0, backoff_repeats, MC_n_iter, backoff_MC\
    , learning, Lsolver, c_code, state_dep, multi_hyper, filter_par

def DAE_system():
    # Define vectors with names of states
    states     = ['x','n','q']
    nd         = len(states)
    xd         = SX.sym('xd',nd)
    for i in range(nd):
        globals()[states[i]] = xd[i]

    # Define vectors with names of algebraic variables
    algebraics = []
    na         = len(algebraics)
    xa         = SX.sym('xa',na)
    for i in range(na):
        globals()[algebraics[i]] = xa[i]

    # Define vectors with banes of input variables
    inputs     = ['L','Fn']
    nu         = len(inputs)
    u          = SX.sym("u",nu)
    for i in range(nu):
        globals()[inputs[i]] = u[i]

    # Define model parameter names and values
    modpar    = ['u_m', 'k_s', 'k_i', 'K_N', 'u_d', 'Y_nx', 'k_m', 'k_sq',
    'k_iq', 'k_d', 'K_Np']
    modparval = [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
    2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]
    nmp       = len(modpar)
    for i in range(nmp):
        globals()[modpar[i]] = SX(modparval[i])

    # Additive measurement noise
    Sigma_v  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

    # Additive disturbance noise
    Sigma_w  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

    # Initial additive disturbance noise
    Sigma_w0 = [1.,150.**2,0.]*diag(np.ones(nd))*1e-3

    # Declare ODE equations (use notation as defined above)
    dx   = u_m * L/(L+k_s+L**2./k_i) * x * n/(n+K_N) - u_d*x
    dn   = - Y_nx*u_m* L/(L+k_s+L**2./k_i) * x * n/(n+K_N) + Fn
    dq   = k_m * L/(L+k_sq+L**2./k_iq) * x - k_d * q/(n+K_Np)

    ODEeq =  [dx,dn,dq]

    # Declare algebraic equations
    Aeq = []

    # Define control bounds
    u_min      = np.array([120.,0.  ]) # lower bound of inputs
    u_max      = np.array([400.,40.]) # upper bound of inputs

    # Define objective to be minimized
    t           = SX.sym('t')
    Obj_M       = Function('mayer',[xd],[-q])      # Mayer term
    Obj_L       = Function('lagrange',[xd,u],[0.]) # Lagrange term
    R           = [1./400.**2.,1./40.**2]*diag(np.ones(nu))*5e-3        # Control change penality

    # Define constraint functions g(x) <= 0
    gpdef      = vertcat(n - 800., q - 0.011*x, n - 150.) # g(x)
    #gpdef      = vertcat(n - 1000., (x - 12.)*40.) # g(x)
    ngp        = SX.size(gpdef)[0]              # Number of constraints
    gpfcn      = Function('gpfcn',[xd],[gpdef]) # Function definition
    pgp        = SX([0.1])                     # Probability of constraint violation !!
    eps        = 0.1                           # probability of satisfaction
    path       = [True,True,False]                    # True = path, False = terminal

    return xd, xa, u, ODEeq, Aeq, Obj_M, Obj_L, R, u_min, \
u_max, states, algebraics, inputs, ngp, gpfcn, pgp, \
 Sigma_v, nd, na, nu, path, Sigma_w, eps, Sigma_w0
