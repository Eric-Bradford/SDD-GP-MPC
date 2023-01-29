# imported libraries
from GP_NMPC_batch import *
from Problem_definition import *
import numpy as np
from casadi import *
import copy
import sys
import time
from os import system, path

start = time.time() # start timing
''' Initialize GP, hyperparameters and problem specifications '''
GP                  = GP_batch()
ngp, x0, nu, nd, nk = GP.ngp, GP.x0, GP.nu, GP.nd, GP.nk
c_code              = GP.c_code
GP_initialization   = time.time()

''' Pre-compute solvers '''
GPNMPC=[0]*nk; Ufcn=[0]*nk; args=[0]*nk;
cname ='nlp.c'

for k in range(nk):
    GP.print2(['Pre-computing solver ',k+1,' of ',nk])
    soname     = 'nlp'+str(nk-k)+'.so'
    opts       = dict(cpp=False)
    precompile = not path.isfile(soname)
    if (precompile and c_code):
        GPNMPC[k], Ufcn[k], args[k] = GP.GP_nominal_backoff(nk-k,precompile,c_code)
        GPNMPC[k].generate_dependencies(cname,opts)
        system('gcc /home/ericcb/Desktop/GP_batch_process_new/nlp.c -fPIC -shared -Os ' + cname + ' -o ' + soname)
        GPNMPC[k], Ufcn[k], args[k] = GP.GP_nominal_backoff(nk-k,not precompile,c_code)
    else:
        GPNMPC[k], Ufcn[k], args[k] = GP.GP_nominal_backoff(nk-k,not precompile,c_code)
Computing_Solvers = time.time()

''' Computing the backoff constraints '''
backoff_repeats                  = GP.backoff_repeats
GP_backoff_computation           = GP.GP_backoff_computation
Conp_back_off, ALL_Conp_back_off = GP.initialize_back_offs()

Xd_MC, Conp_MC, Eobj_GP_MC, Conp_back_off, Ud_MC,\
    ALL_Conp_back_off, Conp_nominal, Conp_pred, backoff_factor,\
    beta_, backoff_factor_a, backoff_factor_b, beta_a, beta_c, betamat, backoff_factormat \
    = GP_backoff_computation(backoff_repeats,GPNMPC, Ufcn,\
                             args,Conp_back_off, ALL_Conp_back_off)

_ = [] # wipe _
Computing_BO_Constraints = time.time()

''' Simulating Real system by Monte Carlo '''
MC_MPC_plant                = GP.MC_MPC_plant
Xd_plant, Conp_plant, u_opt = MC_MPC_plant(GPNMPC, Ufcn, args, Conp_back_off)
_ = [] # wipe _
Computing_MC_plant = time.time()

''' Plotting '''
reference = str(nk)+'_'+str(GP.ndat0)+'_'+str(backoff_repeats)
GP.plot_results_Plant(Xd_plant, u_opt, Conp_plant, Eobj_GP_MC, 'plant', '5States_learning')
GP.plot_results_GP(Xd_MC, Conp_MC, Ud_MC, Conp_nominal, 'GP', '5States_learning', ALL_Conp_back_off)
GP.save_to_file([Xd_MC, Conp_MC, Ud_MC, ALL_Conp_back_off, Xd_plant, u_opt,
                 Conp_plant, Eobj_GP_MC, Conp_nominal, betamat, backoff_factormat], reference, '5States_learning')

GP.print2(['GP initialization = ',GP_initialization - start])
GP.print2(['Computing Solvers = ',Computing_Solvers - GP_initialization])
GP.print2(['Computing Backoffs Routine  = ',Computing_BO_Constraints - Computing_Solvers])
GP.print2(['MPC MC of plant = ',Computing_MC_plant - Computing_BO_Constraints])
GP.print2(['Average GP MPC run = ',(Computing_BO_Constraints - Computing_Solvers)/(backoff_repeats * GP.backoff_MC)])
GP.print2(['Average plant MPC run = ',(Computing_MC_plant - Computing_BO_Constraints)/GP.MC_n_iter])
GP.print2(['Chance constraint satisfaction = ',beta_])
GP.print2(['Final backoff factor = ',backoff_factor])
