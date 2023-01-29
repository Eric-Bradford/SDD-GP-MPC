# import necessary packages
import numpy as np
from os import getcwd
import pickle
from scipy.spatial import distance
from Problem_definition import *
from scipy.optimize import minimize
import math as math
from pylab import *
from scipy.io import savemat
from scipy.spatial.distance import cdist
from casadi import *
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
import numpy.random as random
import sobol_seq
from pyDOE import *
from scipy.stats import beta
import sys
import os

class GP_batch:
    def __init__(self):

        # Variable definitions
        self.nk, self.ndat0, self.tf, self.x0, self.backoff_repeats,\
        self.MC_n_iter, self.backoff_MC, self.learning, self.Lsolver, self.c_code\
        ,self.state_dep, self.multi_hyper, self.filter_par = specifications()
        self.xd, self.xa, self.u, self.ODEeq, self.Aeq, self.Obj_M, self.Obj_L,\
        self.R, self.u_min, self.u_max, self.states, self.algebraics, self.inputs,\
        self.ngp, self.gpfcn, self.pgp, self.Sigma_v, self.nd, self.na, self.nu,\
        self.path, self.Sigma_w, self.eps, self.Sigma_w0 = DAE_system()
        self.deltat = self.tf/self.nk

        # Internal function calls
        self.covSEfcn                            = self.covSEard()
        self.Xdat, self.Ydat                     = self.generate_data()
        self.Xnorm, self.Ynorm, self.stdX, self.stdY, self.meanX, self.meanY \
                                                 = self.normalize_data()
        self.hypopt, self.invKopt                = self.determine_hyperparameters()
        self.meanfcn, self.varfcn, self.meanfcn2, self.varfcnsd\
                                                 = self.GP_predictor()

        self.Xdat, self.Ydat                     = self.GP_datareduction()
        self.Xnorm, self.Ynorm, self.stdX, self.stdY, self.meanX, self.meanY \
                                                 = self.normalize_data()
        self.hypopt, self.invKopt                = self.determine_hyperparameters()
        self.meanfcn, self.varfcn, self.meanfcn2, self.varfcnsd\
                                                 = self.GP_predictor()

    def model_fcn(self):
        xd, xa, u, xu, ODEeq, Aeq = self.xd, self.xa, self.u, self.xu, self.ODEeq, self.Aeq
        t     =   SX.sym("t")
        p_s   =   SX.sym("p_s")
        xddot =   SX.sym("xddot",self.nd)

        res   = []
        for i in range(self.nd):
            res = vertcat(res,ODEeq[i]*p_s - xddot[i])

        for i in range(self.na):
            res = vertcat(res,Aeq[i])

        ffcn = Function('ffcn', [t,xddot,xd,xa,xu,u,p_s],[res])

        return ffcn

    def simulator(self,xd_previous,uNMPC,t0,tf):
        ''' Simulates the Dynamic "real" system given an initial state, t0, tf '''
        xd, xa, u, ODEeq, Aeq = self.xd, self.xa, self.u, self.ODEeq, self.Aeq

        ODE = []
        for i in range(self.nd):
            ODE = vertcat(ODE,substitute(ODEeq[i],u,SX(DM(uNMPC))))

        A = []
        for i in range(self.na):
            A   = vertcat(A,substitute(Aeq[i],u,SX(DM(uNMPC))))

        dae = {'x':xd, 'z':xa, 'ode':ODE, 'alg':A}
        I = integrator('I', 'idas', dae, {'t0':t0, 'tf':tf, 'abstol':1e-12, \
        'reltol':1e-12})
        res        = I(x0=xd_previous)
        xd_current = np.array(res['xf'])
        xa_current = np.array(res['zf'])

        return xd_current, xa_current

    def generate_data(self):
        ''' Generates data to train initial GP '''
        ndat0, nu      = 300, self.nu
        nd, x0, deltat = self.nd, self.x0, self.deltat
        Sigma_v, tf    = self.Sigma_v, self.tf
        u_min, u_max   = self.u_min, self.u_max
        nk, simulator  = self.nk, self.simulator
        Xdat           = np.zeros((ndat0,nd+nu))
        Ydat           = np.zeros((ndat0,nd))
        nruns          = int(ndat0*deltat/tf)
        udat1          = sobol_seq.i4_sobol_generate(nu,ndat0)
        x_min, x_max   = np.array([0.,50.,0.]), np.array([20.,800.,0.18])
        Xdat           = sobol_seq.i4_sobol_generate(nu+nd,ndat0)
        original       = False

        if original:
            i = 0
            for j in range(nruns*100):
                xd_current = x0
                t0, tf     = 0., 0.

                for k in range(nk):
                    if i >= ndat0:
                        break
                    udat = u_min + udat1[i,:]*(u_max-u_min)

                    tf        += deltat
                    xtemp      = np.array(vertcat(DM(xd_current),udat)).flatten()
                    xd_current, xa_current = simulator(xd_current,DM(udat),t0,tf)
                    xd_current = xd_current.flatten()
                    if xd_current[1] < 900.:
                        Ydat[i,:]  = np.array(xd_current).flatten() +\
                                     np.random.multivariate_normal(np.zeros(nd),Sigma_v)
                        Xdat[i,:]  = xtemp
                        i += 1

                    t0        += deltat
        else:
            for i in range(ndat0):
                xdat      = x_min + Xdat[i,:nd]*(x_max-x_min)
                udat      = u_min + Xdat[i,nd:]*(u_max-u_min)
                Xdat[i,:] = np.hstack((xdat,udat)).flatten()
                xd_current, xa_current = simulator(DM(xdat),DM(udat),0.,deltat)
                xd_current = xd_current.flatten()
                Ydat[i,:]  = np.array(xd_current).flatten() +\
                             np.random.multivariate_normal(np.zeros(nd),Sigma_v)

        return Xdat, Ydat

    def GP_datareduction(self):
        Xnorm, ndat0 = self.Xnorm, self.ndat0
        Xdat, Ydat   = self.Xdat, self.Ydat
        varfcnsd     = self.varfcnsd
        sf2opt       = np.exp(2.*self.hypopt[self.nd+self.nu,:])

        for i in range(300):
            distmat = np.zeros((300-i))
            for j in range(300-i):
                distmat[j] = sum1(varfcnsd(Xnorm[j,:])/sf2opt)
                Xnorm2D    = Xnorm[j,:].reshape(1,self.nd+self.nu)
            indexsort = np.argsort(distmat)
            Xnorm     = Xnorm[indexsort[1:],:]
            Xdat      = Xdat[indexsort[1:],:]
            Ydat      = Ydat[indexsort[1:],:]

            if Xnorm.shape[0]==ndat0:
                break

        return Xdat, Ydat

    def normalize_data(self):
        ''' Routine that outputs normalization utililies '''
        nX, nd       = self.nd + self.nu, self.nd
        Xdat, Ydat   = self.Xdat, self.Ydat
        ndat         = Xdat.shape[0]
        Xnorm, Ynorm = np.zeros((ndat,nX)), np.zeros((ndat,nd))

        stdX , stdY  = np.std(Xdat,0) , np.std(Ydat,0)
        meanX, meanY = np.mean(Xdat,0), np.mean(Ydat,0)
        for i in range(ndat):
            Xnorm[i,:] = (Xdat[i,:] - meanX)/stdX

        for i in range(ndat):
            Ynorm[i,:] = (Ydat[i,:] - meanY)/stdY

        return Xnorm, Ynorm, stdX, stdY, meanX, meanY

    def initialize_back_offs(self):
        ''' Routine that initializes the backoffs '''
        nk, backoff_repeats, ngp  = self.nk, self.backoff_repeats, self.ngp

        Conp_back_off        = np.zeros((ngp, nk))
        ALL_Conp_back_off    = np.zeros((ngp, nk, backoff_repeats))
        return Conp_back_off, ALL_Conp_back_off

    def determine_hyperparameters(self):
        nd, nu           = self.nd, self.nu
        Xnorm, Ynorm, nX = self.Xnorm, self.Ynorm, nd + nu
        ndat             = Xnorm.shape[0]
        lb               = np.array([-3.]*(nX+1) + [-8.])
        ub               = np.array([3.]*(nX+1)  + [ 4.])
        bounds           = np.hstack((lb.reshape(nX+2,1),ub.reshape(nX+2,1)))
        multi_start      = self.multi_hyper
        multi_startvec   = sobol_seq.i4_sobol_generate(nX+2,multi_start)

        options  = {'disp':False,'maxiter':10000}
        hypopt   = np.zeros((nX+2,nd))
        localsol = [0.]*multi_start
        localval = np.zeros((multi_start))

        invKopt = []
        for i in range(nd):
            for j in range(multi_start):
                self.print2(['multi_start hyper parameter optimization iteration = ',j,'  state = ',i])
                hyp_init    = lb + (ub-lb)*multi_startvec[j,:]
                res = minimize(self.negative_loglikelihood,hyp_init,args=(Xnorm,Ynorm[:,i])\
                               ,method='SLSQP',options=options,bounds=bounds,tol=1e-12)
                localsol[j] = res.x
                localval[j] = res.fun
            minindex    = np.argmin(localval)
            hypopt[:,i] = localsol[minindex]
            ellopt      = np.exp(2.*hypopt[:nX,i])
            sf2opt      = np.exp(2.*hypopt[nX,i])
            sn2opt      = np.exp(2.*hypopt[nX+1,i]) + 1e-6
            cov_mat     = self.calc_cov_matrix(Xnorm,ellopt,sf2opt) + sn2opt*np.eye(ndat)
            invKopt    += [np.linalg.solve(cov_mat,np.eye(ndat))]

        return hypopt, invKopt

    def Online_MatrixInv(self,Conv_inv,xnorm,sf2,sn2,Xsample,ell):
        # array manipulation
        calc_cov_sample = self.calc_cov_sample

        k   = calc_cov_sample(xnorm,Xsample,ell,sf2)
        A22 = np.array([sn2]); A22 = A22.reshape((1,1))
        A12 = k.reshape((k.shape[0], 1))
        A21 = k.reshape((1,k.shape[0]))
        I   = Conv_inv
        II  = np.matmul(A21,I)
        III = np.matmul(I,A12)
        IV  = np.matmul(A21,III)
        V   = IV - A22
        VI  = 1./V
        C12 = III * VI
        C21 = VI * II
        VII = np.matmul(III,C21)
        C11 = I - VII
        C22 = -VI
        C   = np.block([[C11,C12],[C21,C22]])

        return C

    def GP_predictor(self):
        nd, invKopt, hypopt      = self.nd, self.invKopt, self.hypopt
        Ynorm, Xnorm             = SX(DM(self.Ynorm)), SX(DM(self.Xnorm))
        ndat                     = Xnorm.shape[0]
        nX, covSEfcn, nk         = self.nd + self.nu, self.covSEfcn, self.nk
        stdX, stdY, meanX, meanY = SX(self.stdX),SX(self.stdY),SX(self.meanX),SX(self.meanY)

        x      = SX.sym('x',nX)
        xnorm  = (x - meanX)/stdX
        k      = SX.zeros(ndat)
        k2     = SX.zeros(ndat+nk)
        mean   = SX.zeros(nd)
        mean2  = SX.zeros(nd)
        var    = SX.zeros(nd)
        Xnorm2 = SX.sym('Xnorm2',ndat+nk,nX)
        invKY2 = SX.sym('invKY2',ndat+nk,nd)

        for i in range(nd):
            invK           = SX(DM(invKopt[i]))
            hyper          = SX(DM(hypopt[:,i]))
            ellopt, sf2opt = exp(2*hyper[:nX]), exp(2*hyper[nX])
            for j in range(ndat):
                k[j]  = covSEfcn(xnorm,Xnorm[j,:],ellopt,sf2opt)
            for j in range(ndat+nk):
                k2[j] = covSEfcn(xnorm,Xnorm2[j,:],ellopt,sf2opt)

            invKYnorm = mtimes(invK,Ynorm[:,i])
            mean[i]   = mtimes(k.T,invKYnorm)
            mean2[i]  = mtimes(k2.T,invKY2[:,i])
            var[i]    = sf2opt - mtimes(mtimes(k.T,invK),k)

        meanfcn  = Function('meanfcn',[x],[mean*stdY + meanY])
        meanfcn2 = Function('meanfcn2',[x,Xnorm2,invKY2],[mean2*stdY + meanY])
        varfcn   = Function('varfcn',[x] ,[var*stdY**2])
        varfcnsd = Function('varfcnsd',[x],[var])

        return meanfcn, varfcn, meanfcn2, varfcnsd

    def GP_predictor_np(self, x, invKsample, Xsample, Ysample):
        nd, hypopt               = self.nd, self.hypopt
        nX                       = self.nd + self.nu
        stdX, stdY, meanX, meanY = self.stdX, self.stdY, self.meanX, self.meanY
        calc_cov_sample          = self.calc_cov_sample
        Sigma_w                  = self.Sigma_w

        xnorm = (x - meanX)/stdX
        mean  = np.zeros(nd)
        var   = np.zeros(nd)
        for i in range(nd):
            invK           = invKsample[i]
            hyper          = hypopt[:,i]
            ellopt, sf2opt = np.exp(2*hyper[:nX]), np.exp(2*hyper[nX])

            k = calc_cov_sample(xnorm,Xsample,ellopt,sf2opt)
            mean[i] = np.matmul(np.matmul(k.T,invK),Ysample[:,i])
            var[i]  = sf2opt + Sigma_w[i,i]/stdY[i]**2 - np.matmul(np.matmul(k.T,invK),k)

        mean_sample = mean*stdY + meanY
        var_sample  = var*stdY**2

        return mean_sample, var_sample

    def calc_cov_sample(self,xnorm,Xnorm,ell,sf2):
        nd, nu = self.nd, self.nu

        n, D = Xnorm.shape
        dist = cdist(Xnorm, xnorm.reshape(1,nd+nu), 'seuclidean', V=ell)**2
        cov_matrix = sf2 * np.exp(-.5*dist)

        return cov_matrix

    def covSEard(self):
        nd, nu = self.nd, self.nu
        ell    = SX.sym('ell',nd+nu)
        sf2    = SX.sym('sf2')
        x, z   = SX.sym('x',nd+nu), SX.sym('z',nd+nu)
        dist   = sum1((x - z)**2 / ell)
        covSEfcn = Function('covSEfcn',[x,z,ell,sf2],[sf2*exp(-.5*dist)])

        return covSEfcn

    def calc_cov_matrix(self,Xnorm,ell,sf2):
        dist = cdist(Xnorm,Xnorm,'seuclidean',V=ell)**2
        cov_matrix = sf2*np.exp(-0.5*dist)

        return cov_matrix

    def negative_loglikelihood(self,hyper,X,Y):
        n, nX = X.shape[0], X.shape[1]
        ell   = np.exp(2*hyper[:nX])
        sf2   = np.exp(2*hyper[nX])
        lik   = np.exp(2*hyper[nX+1])

        K       = self.calc_cov_matrix(X,ell,sf2)
        K       = K + (lik+1e-8)*np.eye(n)
        K       = (K + K.T)*0.5
        L       = np.linalg.cholesky(K)
        logdetK = 2 * np.sum(np.log(np.diag(L)))
        invLY   = np.linalg.solve(L,Y)
        alpha   = np.linalg.solve(L.T,invLY)
        NLL     = np.dot(Y.T,alpha) + logdetK

        return NLL

    def compute_Conp(self, x_opt, Conp_MC, bo_MC):
        ''' collect data computed by the MPC routine for path constraints '''
        nk, gpfcn = self.nk, self.gpfcn

        for step in range(nk):
            Conp_MC[:,step,bo_MC]  = np.array(DM(gpfcn(x_opt[:,step+1]))).flatten()

        return Conp_MC

    def collect_MC_data(self, U_data, Xd_data, Xa_data, Conp_data, Cont_data,\
                        t_data, x_opt, u_opt, un):
        ''' collect data computed by the MPC routine '''
        nk = self.nk

        Xd_data[:,:,un]    = x_opt
        U_data[:,:,un]     = u_opt
        for step in range(nk+1):
            Conp_data[:,step,un]  = np.array(DM(self.gpfcn(x_opt[:,step]))).flatten()
            Cont_data[:,step,un]  = np.array(DM(self.gtfcn(x_opt[:,step]))).flatten()

        return U_data, Xd_data, Xa_data, Conp_data, Cont_data, t_data

    def MPC_params(self, invK, Xmeasure, Ymeasure):
        nk, ndat0, nd   = self.nk, self.ndat0, self.nd
        calc_cov_sample = self.calc_cov_sample
        ndat            = invK[0].shape[0]
        nX              = self.nd + self.nu

        invK_MPC = np.zeros((nk+ndat0, nk+ndat0))
        X_MPC    = np.zeros((nk+ndat0, nX))
        par      = np.zeros(((ndat0+nk)**2)*nd + (nk+ndat0)*nX + (nk+ndat0)*(nd))

        for ij in range(nd):
            invK_MPC[:ndat,:ndat]                          = invK[ij]
            par[((ndat0+nk)**2)*ij:((ndat0+nk)**2)*(ij+1)] =\
            np.array(reshape(invK_MPC,((ndat0+nk)**2),1)).flatten()

            X_MPC[:ndat,:]   =  Xmeasure
            par[((ndat0+nk)**2)*nd: ((ndat0+nk)**2)*nd+(nk+ndat0)*nX] =\
            np.array(reshape(X_MPC, (nk+ndat0)*nX, 1)).flatten()

            y   = Ymeasure[:,ij].flatten()
            y   = np.concatenate((y,np.zeros(nk+ndat0-ndat)))
            par[((ndat0+nk)**2)*nd+(nk+ndat0)*nX+(nk+ndat0)*(ij) :\
            ((ndat0+nk)**2)*nd+(nk+ndat0)*nX+(nk+ndat0)*(ij+1)] = y

        return par

    def OCP_step_GP(self, x_opt, Ufcn_, res, u_opt, step, xd_current, invKsample,
                   Xsample, Ysample, sf2, ell):
        Sigma_v, nd      = self.Sigma_v, self.nd
        GP_predictor_np  = self.GP_predictor_np
        meanY, stdY      = self.meanY, self.stdY
        meanX, stdX, ngp = self.meanX, self.stdX, self.ngp
        Online_MatrixInv = self.Online_MatrixInv

        u_                = np.array(Ufcn_(np.array(res["x"][:,0])))
        u_opt[:,step]     = u_[:,0]
        xnew_measured     = xd_current
        xnew_measured     = np.concatenate((xnew_measured.reshape(nd,1),u_), axis=None)
        xnew              = np.concatenate((xd_current.reshape(nd,1),u_), axis=None)
        xd_Mean, xd_Sigma = GP_predictor_np(xnew, invKsample, Xsample, Ysample)
        xd_Sigma          = xd_Sigma*np.eye(nd)
        xd_current        = (xd_Mean
                             + np.random.multivariate_normal(np.zeros(nd),xd_Sigma))
        xd_measured       = xd_current
        x_opt[:,step+1]   = xd_current[:]

        xd_current_norm   = (xd_current - meanY)/stdY
        xnew_norm         = (xnew - meanX)/stdX
        for ij in range(nd):
            invKsample[ij]= Online_MatrixInv(invKsample[ij],xnew_norm,sf2[ij],1e-6,Xsample,ell[ij])
        Xsample           = np.vstack((Xsample,xnew_norm))
        Ysample           = np.vstack((Ysample,xd_current_norm))

        return Xsample, Ysample, invKsample, x_opt, xd_current, xnew_measured, xd_measured, u_opt

    def OCP_step_Plant(self, x_opt, Ufcn_, res, u_opt, step, xd_current, tfi, t0is, MC_i):
        Sigma_v, nd, simulator      = self.Sigma_v, self.nd, self.simulator
        Sigma_w                     = self.Sigma_w

        u_                 = np.array(Ufcn_(np.array(res["x"][:,0])))
        u_opt[:,step,MC_i] = u_[:,0]
        xnew_measured      = xd_current
        xnew_measured      = np.concatenate((xnew_measured.reshape(nd,1),u_), axis=None)
        xd_current, _      = simulator(xd_current,u_,t0is,tfi)
        xd_current         = xd_current.flatten() + np.random.multivariate_normal(np.zeros(nd),Sigma_w).flatten()
        xd_measured        = xd_current
        x_opt[:,step+1]    = xd_current[:]

        return x_opt, xd_current, xnew_measured, xd_measured, u_opt

    def Compute_beta(self,ngp,Conp_back_off,Conp_MC):
        path         = self.path
        S            = Conp_MC.shape[2]
        alpha        = 0.01             # confidence interval of the cdf

        scaling = [800.,0.01,200.]
        bj = 0.
        for j in range(S):
            ai = 0.
            for i in range(ngp):
                if path[i]:
                    ai += np.sum(Conp_MC[i,:,j]/scaling[i]  >= 1e-4)
                else:
                    ai += np.sum(Conp_MC[i,-1,j]/scaling[i] >= 1e-4)
            bj += ai > 0.
        beta_cor = 1. - bj/S

        beta_ = 1. - beta.ppf(1. - alpha, S+1-beta_cor*S, beta_cor*S)

        return beta_

    def update_backoff(self,ngp,Conp_nominal0,Conp_MC0,backoff_factor):
        pgp, nk = np.float(self.pgp), self.nk

        Conp_nominal0 = np.reshape(Conp_nominal0, (ngp, nk) ,order='F')

        F_inv = np.percentile(Conp_MC0, (1.-np.float(pgp))*100., axis=2)
        Conp_back_off = (F_inv - Conp_nominal0)*backoff_factor

        return Conp_back_off

    def load_varsopt(self, MC_i, step, args):
        if MC_i != 0:
            try:
                with open("varsopt_dir/varsopt" + str(step)+".pkl", 'rb') as a_file:
                    args[step]["x0"] = pickle.load(a_file)
            except:
                self.print2(["error loading, step = ",step])
        return

    def save_varsopt(self, step, res):
        try:
            with open("varsopt_dir/varsopt" + str(step)+".pkl", 'wb') as a_file:
                pickle.dump(np.array(res["x"]), a_file)
        except:
            self.print2(["error saving, step = ",step])
        return

    def plot_results_Plant(self, Xd_plant2, u_opt2, Conp_plant2, Eobj_GP_MC, PorGP, folder):
        ''' Plot results and save to files '''
        nd, nu         = self.nd, self.nu
        ngp, MC_n_iter = self.ngp, self.MC_n_iter
        nk, tf         = self.nk, self.tf

        t_X = np.linspace(0, tf, nk+1, endpoint=True)
        for j in range(nd):
            plt.figure()
            for i in range(MC_n_iter):
                plt.plot(t_X,list(Xd_plant2[j,:,i]),'-')
            plt.ylabel('plant x_'+str(j))
            plt.xlabel('time')
            #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
            plt.savefig(folder+'/'+'x_'+str(j)+PorGP+
                        '_.png', dpi=150)
            plt.close()

        for j in range(nu):
            plt.figure()
            for i in range(MC_n_iter):
                plt.step(t_X[:-1],list(u_opt2[j,:,i]),'-')
            plt.ylabel('plant control u_'+str(j))
            plt.xlabel('time')
            #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
            plt.savefig(folder+'/'+'u_'+str(j)+PorGP+
                        '.png', dpi=150)
            plt.close()

        Conp_data_mean = np.mean(Conp_plant2, axis=2)
        Conp_data_std  = np.std(Conp_plant2, axis=2)
        for j in range(ngp):
            plt.figure()
            for i in range(MC_n_iter):
                plt.plot(t_X,list(Conp_plant2[j,:,i]),'--', color='grey')
            plt.plot(t_X,list((Conp_data_mean+Conp_data_std)[j,:]),'-', color='black')
            plt.plot(t_X,list((Conp_data_mean-Conp_data_std)[j,:]),'-', color='black')
            plt.plot(t_X,[0.0 for i in range(len(Conp_data_mean[j,:]))],'-.', color='black')
            plt.ylabel(PorGP+'_path_constraint_'+str(j))
            plt.xlabel('time')
            #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
            plt.savefig(folder+'/'+PorGP+'path constraint '
                        +str(j)+'.png', dpi=150)
            plt.close()

        plt.figure()
        plt.plot(list(Eobj_GP_MC),'--')
        plt.ylabel('objective')
        plt.xlabel('iterations')
        #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
        plt.savefig(folder+'/'+'objective in back-offs.png', dpi=150)
        plt.close()

        return

    def plot_results_GP(self, Xd_MC, Conp_MC, Ud_MC, Conp_nominal, PorGP, folder, ALL_Conp_back_off):
        ''' Plot results and save to files '''
        nd, nu               = self.nd, self.nu
        ngp, backoff_repeats = self.ngp, self.backoff_repeats
        nk, tf, backoff_MC   = self.nk, self.tf, self.backoff_MC

        t_X = np.linspace(0, tf, nk+1, endpoint=True)
        t_U = np.linspace(0, tf, nk, endpoint=False)

        for j in range(nd):
            for k in range(backoff_repeats):
                plt.figure()
                for i in range(backoff_MC):
                    plt.plot(t_X,list(Xd_MC[j,:,i,k]),'-')

                plt.ylabel(PorGP+' repeat '+str(k)+' x_'+str(j))
                plt.xlabel('time')
                #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
                plt.savefig(folder+'/'+' repeat '+str(k)+'x_'+str(j)+PorGP+
                            '.png', dpi=150)
                plt.close()

        for j in range(nu):
            for k in range(backoff_repeats):
                plt.figure()
                for i in range(backoff_MC):
                    plt.step(t_X[:-1],list(Ud_MC[j,:,i,k]),'-')
                plt.ylabel(PorGP+' repeat '+str(k)+' u_'+str(j))
                plt.xlabel('time')
                #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
                plt.savefig(folder+'/'+' repeat '+str(k)+'u_'+str(j)+PorGP+
                            '.png', dpi=150)
                plt.close()

        Conp_data_mean = np.mean(Conp_MC, axis=2)
        Conp_data_std  = np.std(Conp_MC, axis=2)
        ''' plot of last iteration of the back-off '''
        for j in range(ngp):
            plt.figure()
            for i in range(backoff_MC):
                plt.plot(t_X[1:],list(Conp_MC[j,:,i]),'--', color='grey')
            plt.plot(t_X[1:],list(Conp_nominal[j,:]),'-', color='black')
            plt.plot(t_X[1:],[0.0 for i in range(len(Conp_data_mean[j,:]))],'-.', color='black')
            plt.ylabel(PorGP+'constraint '+str(j))
            plt.xlabel('time')
            #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
            plt.savefig(folder+'/'+PorGP+
                        ' path constraint '+str(j)+'.png', dpi=150)
            plt.close()

        c_ = [(backoff_repeats - float(i))/backoff_repeats for i in range(backoff_repeats)]
        for j in range(ngp):
            plt.figure()
            for i in range(backoff_repeats):
                plt.plot(t_X[1:],list(ALL_Conp_back_off[j,:,i]),'--', color=str(c_[i]))
            plt.ylabel(PorGP+'back-off '+str(j))
            plt.xlabel('time')
            #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
            plt.savefig(folder+'/'+PorGP+
                        ' ALL path constraint '+str(j)+'.png', dpi=150)
            plt.close()

        return

    def save_to_file(self, input_list, ref, folder):

        names_l = ['Xd_MC', 'Conp_MC', 'Ud_MC', 'ALL_Conp_back_off', 'Xd_plant2', 'u_opt2',
                 'Conp_plant2', 'Eobj_GP_MC','Conp_nominal', 'betamat', 'backoff_factormat']
        for li in range(len(input_list)):
            with open('PlotFiles/'+folder+'/'+str(names_l[li])+ref+"_file"+".pkl", 'wb') as a_file:
                pickle.dump(input_list[li], a_file)

    def save_to_file_nom(self, input_list, ref, folder):
        names_l = ['Xd_plant_nom', 'u_opt_nom', 'Conp_plant_nom']
        for li in range(len(input_list)):
            with open('PlotFiles/'+folder+'/'+str(names_l[li])+ref+"_file"+".pkl", 'wb') as a_file:
                pickle.dump(input_list[li], a_file)

    def GP_nominal_backoff(self,nk_sh,precompile,c_code):
        ''' Optimization of nominal (mean) GP incorporating backoff constraints '''
        meanfcn2, nd, nu   = self.meanfcn2, self.nd, self.nu
        Xdat, u_min, u_max = self.Xdat, self.u_min, self.u_max
        gpfcn, ngp         = self.gpfcn, self.ngp
        Obj_M, Obj_L, nX   = self.Obj_M, self.Obj_L, (self.nd + self.nu)
        ndat0, path        = self.ndat0, self.path
        Lsolver, nk        = self.Lsolver, self.nk
        state_dep, hypopt  = self.state_dep, self.hypopt
        varfcnsd, R        = self.varfcnsd, SX(DM(self.R))

        # Define optimization variables
        par            = MX.sym("par",nd+ngp*nk + ((ndat0+nk)**2)*nd\
                                + nX*(ndat0+nk) + (nk+ndat0)*nd + nu)
        varsopt        = MX.sym("varsopt",nd*nk_sh+nu*nk_sh)
        vars_lb        = np.zeros(nd*nk_sh+nu*nk_sh)
        vars_ub        = np.zeros(nd*nk_sh+nu*nk_sh)
        vars_init      = np.zeros(nd*nk_sh+nu*nk_sh)
        XD             = np.resize(np.array([],dtype=MX),(nk_sh+1))
        U              = np.resize(np.array([],dtype=MX),(nk_sh))
        offset         = 0
        XD[0]          = par[:nd]
        Conp_back_off  = reshape(par[nd:nd + ngp*nk],ngp,nk)
        parmean        = par[nd+ngp*nk : nd+ngp*nk + ((ndat0+nk)**2)*nd
                             + nX*(ndat0+nk) + (nk+ndat0)*nd]
        invKpar        = parmean[:((ndat0+nk)**2)*nd]
        Xnormpar       = parmean[((ndat0+nk)**2)*nd : ((ndat0+nk)**2)*nd + nX*(ndat0+nk)]
        Ypar           = parmean[((ndat0+nk)**2)*nd + nX*(ndat0+nk) :\
                         ((ndat0+nk)**2)*nd + nX*(ndat0+nk) + (nk+ndat0)*nd]
        Xnorm2         = reshape(Xnormpar,ndat0+nk,nX)
        u0             = par[nd+ngp*nk + ((ndat0+nk)**2)*nd\
                                + nX*(ndat0+nk) + (nk+ndat0)*nd:\
                                nd+ngp*nk + ((ndat0+nk)**2)*nd\
                                + nX*(ndat0+nk) + (nk+ndat0)*nd +nu]

        Ya       = SX.sym('Ya',nk+ndat0)
        invKa    = SX.sym('invKa',nk+ndat0,nk+ndat0)
        invKYfcn = Function('invKYfcn',[invKa,Ya],[mtimes(invKa,Ya)])
        invKY2   = MX.zeros(ndat0+nk,nd)
        for i in range(nd):
            invK2         = reshape(invKpar[i*((ndat0+nk)**2):\
                            (i+1)*((ndat0+nk)**2)],ndat0+nk,ndat0+nk)
            Y2            = Ypar[i*(ndat0+nk) : (i+1)*(ndat0+nk)]
            invKY2[:,i]   = invKYfcn(invK2,Y2)

        for i in range(nk_sh):
            XD[i+1]                     = varsopt[offset:offset+nd]
            vars_lb[offset:offset+nd]   = np.ones(nd)*-inf
            vars_ub[offset:offset+nd]   = np.ones(nd)*inf
            vars_init[offset:offset+nd] = np.array(Xdat[i+(nk-nk_sh),:nd])
            offset                     += nd
            U[i]                        = varsopt[offset:offset+nu]
            vars_lb[offset:offset+nu]   = u_min
            vars_ub[offset:offset+nu]   = u_max
            vars_init[offset:offset+nu] = Xdat[i+(nk-nk_sh),nd:]
            offset                     += nu

        # Define constraints
        g      = []
        lbg    = []
        ubg    = []
        for i in range(nk_sh):
            g += [meanfcn2(vertcat(XD[i],U[i]),Xnorm2,invKY2) - XD[i+1]]
            lbg.append(np.zeros(nd))
            ubg.append(np.zeros(nd))

        for i in range(nk_sh):
            for j in range(ngp):
                if path[j]:
                    g += [gpfcn(XD[i+1])[j] + Conp_back_off[j,(nk-nk_sh) + i]]
                    lbg.append(np.ones(1)*-inf)
                    ubg.append(np.zeros(1))
                else:
                    if i == (nk_sh-1):
                        g += [gpfcn(XD[i+1])[j] + Conp_back_off[j,(nk-nk_sh) + i]]
                        lbg.append(np.ones(1)*-inf)
                        ubg.append(np.zeros(1))

        # Define objective
        Obj = 0
        if state_dep:
            sf2  = exp(2*hypopt[nd+nu,:])
            beta = 15.
            Obj += beta*(sum1(varfcnsd(vertcat(XD[0],U[0]))/sf2))

        for i in range(nk_sh):
            Obj += Obj_L(XD[i+1],U[i])
        Obj += Obj_M(XD[nk_sh])

        # Control penality
        u1     = SX.sym('u1',nu)
        u2     = SX.sym('u2',nu)
        dufcn  = Function('dufcn',[u1,u2],[mtimes(mtimes(transpose(u2-u1),R),u2-u1)])
        deltau = MX.zeros(1)
        for k in range(nk_sh-1):
            if k == 0:
                deltau += dufcn(u0,U[k])
            else:
                deltau += dufcn(U[k],U[k+1])
        Obj += deltau

        # Define NLP
        opts                          = {}
        opts["expand"]                = True
        opts["ipopt.print_level"]     = 0
        opts["ipopt.max_iter"]        = 1000
        opts["ipopt.tol"]             = 1e-8
        opts["ipopt.linear_solver"]   = Lsolver
        opts["calc_lam_p"]            = False
        opts["calc_multipliers"]      = False

        nlp    = {'x':varsopt,'p':par,'f':Obj,'g':vertcat(*g)}
        soname = 'nlp' + str(nk_sh) + '.so'
        if (precompile or (not c_code)):
            GPNMPC = nlpsol("solver","ipopt",nlp,opts)
        else:
            GPNMPC = nlpsol("solver","ipopt",soname)
        Ufcn   = Function('Ufcn',[varsopt],[U[0]])
        args        = {}
        args["lbx"] = vars_lb
        args["ubx"] = vars_ub
        args["x0"]  = vars_init
        args["lbg"] = np.concatenate(lbg)
        args["ubg"] = np.concatenate(ubg)

        return GPNMPC, Ufcn, args

    def GP_backoff_computation(self, backoff_repeats, GPNMPC2,
                                 Ufcn2, args2,Conp_back_off, ALL_Conp_back_off):
        ''' Compute the backoffs '''
        backoff_MC         = self.backoff_MC #MC iterations to get average violation
        backoff_repeats    = self.backoff_repeats
        MPC_run_scenarioGP = self.MPC_run_scenarioGP
        nd, ngp, nk, nu    = self.nd, self.ngp, self.nk, self.nu
        compute_Conp       = self.compute_Conp
        update_backoff     = self.update_backoff
        MPC_GP_nominal     = self.MPC_GP_nominal
        hypopt             = self.hypopt
        Compute_beta       = self.Compute_beta

        sf2 = [0]*nd; sn2 = [0]*nd; ell = [0]*nd;
        for ii in range(nd):
            sf2[ii] = np.exp(2*hypopt[nd+nu,ii])
            sn2[ii] = np.exp(2*hypopt[nd+nu+1,ii])
            ell[ii] = np.exp(2*hypopt[:nd+nu,ii])

        ''' Initialize data collectors '''
        Xd_MC      = np.zeros((nd,  nk+1, backoff_MC, backoff_repeats))
        Ud_MC      = np.zeros((nu,  nk, backoff_MC, backoff_repeats))
        Eobj_GP_MC = np.zeros((backoff_repeats))

        ''' back-off iterations '''
        backoff_factor_a  = 0.
        backoff_factor_b  = 4.
        Conp_MC0          = np.zeros((ngp, nk, backoff_MC))
        Conp_pred0        = np.zeros((ngp, nk, backoff_MC))
        betamat           = np.zeros((backoff_repeats))
        backoff_factormat = np.zeros((backoff_repeats))

        for un in range(backoff_repeats):
            Conp_MC   = np.zeros((ngp, nk, backoff_MC))
            Conp_pred = np.zeros((ngp,  nk, backoff_MC))
            obj_MC    = np.zeros((backoff_MC))

            if un != 0:
                backoff_factor_c = (backoff_factor_a + backoff_factor_b)/2.
                backoff_factor   = backoff_factor_c
            else:
                backoff_factor   = backoff_factor_a

            if un != 0:
                Conp_back_off    = \
                update_backoff(ngp,Conp_nominal0,Conp_MC0,backoff_factor)

            for bo_MC in range(backoff_MC): # MC iterations to get average violation

                ''' Solve MPC '''
                x_opt, u_opt, obj_f, self.par, x_pred = MPC_run_scenarioGP(
                bo_MC, un, GPNMPC2,Ufcn2, args2,sf2, ell, Conp_back_off, sn2)
                obj_MC[bo_MC] = obj_f
                if un == 0:
                    ''' Compute path constraints '''
                    Conp_MC0   = compute_Conp(x_opt , Conp_MC0 , bo_MC)
                    Conp_pred0 = compute_Conp(x_pred, Conp_pred0, bo_MC)

                Conp_MC   = compute_Conp(x_opt , Conp_MC  , bo_MC)
                Conp_pred = compute_Conp(x_pred, Conp_pred, bo_MC)

                ''' Collect states last iteration MC '''
                Xd_MC[:,:,bo_MC, un] = x_opt
                Ud_MC[:,:,bo_MC, un] = u_opt
                self.print2(['Conp_back_off = ',Conp_back_off])

            ''' Nominal path constraints '''
            if un == 0:
                x_nominal0, u_nominal0      = \
                MPC_GP_nominal(GPNMPC2, Ufcn2, args2, sf2, ell, Conp_back_off)
                Conp_nominal0 = np.zeros((ngp,nk,1))
                Conp_nominal0 = compute_Conp(x_nominal0,Conp_nominal0,0)

            x_nominal, u_nominal      = \
            MPC_GP_nominal(GPNMPC2, Ufcn2, args2, sf2, ell, Conp_back_off)
            Conp_nominal = np.zeros((ngp,nk,1))
            Conp_nominal = compute_Conp(x_nominal,Conp_nominal,0)

            beta_ = Compute_beta(ngp,Conp_back_off,Conp_MC)

            ''' collect results '''
            ALL_Conp_back_off[:,:,un] = Conp_back_off
            Eobj_GP_MC[un]            = np.mean(obj_MC, axis=0)
            betamat[un]               = beta_
            backoff_factormat[un]     = backoff_factor

            if un != 0:
                beta_c = beta_ - (1.-self.eps)
            else:
                beta_a = beta_ - (1.-self.eps)

            if un != 0:
                if np.sign(beta_c) == np.sign(beta_a):
                    backoff_factor_a = backoff_factor_c
                    beta_a           = beta_c
                else:
                    backoff_factor_b = backoff_factor_c
                    beta_b           = beta_c

        return Xd_MC, Conp_MC, Eobj_GP_MC, Conp_back_off, Ud_MC,\
    ALL_Conp_back_off, Conp_nominal, Conp_pred, backoff_factor,\
    beta_, backoff_factor_a, backoff_factor_b, beta_a, beta_c, betamat, backoff_factormat

    def MPC_GP_nominal(self, GPNMPC2, Ufcn2, args2, sf2, ell, Conp_back_off):
        ''' simulates the nominal  MPC  '''
        nd, invK, nu     = self.nd, self.invKopt[:], self.nu
        xd_current, nk   = self.x0, self.nk
        GP_predictor_np  = self.GP_predictor_np
        ngp,MPC_params   = self.ngp, self.MPC_params
        load_varsopt     = self.load_varsopt
        save_varsopt     = self.save_varsopt
        Xnorm, Ynorm     = self.Xnorm[:], self.Ynorm[:]
        meanfcn2         = self.meanfcn2
        u_min            = self.u_min

        x_nominal      = np.zeros((nd,nk+1)); u_nominal = np.zeros((nu,nk))
        x_nominal[:,0] = xd_current[:]
        u_             = u_min

        for step in range(nk):
            self.print2(['Computing nominal MPC,  step = ',step])
            load_varsopt(1,step,args2)

            p_backoff           = Conp_back_off.reshape((ngp*nk) ,order='F')
            if self.learning:
                par = MPC_params(invK,Xnorm[:],Ynorm[:])
            else:
                par = MPC_params(self.invKopt[:],self.Xnorm[:],self.Ynorm[:])
            args2[step]["p"]    = np.concatenate((xd_current,p_backoff,par,u_.flatten()))     # define current value of states
            res                 = GPNMPC2[step](**args2[step])  # solve nominal model (mean GP)
            self.print2(['solver status = ',GPNMPC2[step].stats()['return_status']])
            save_varsopt(step, res)

            u_                  = np.array(Ufcn2[step](np.array(res["x"][:,0])))
            u_nominal[:,step]   = u_[:,0]
            xnew                = np.concatenate((xd_current.reshape(nd,1),u_), axis=None)
            xd_Mean, _          = GP_predictor_np(xnew, invK, Xnorm, Ynorm)
            xd_current          = (xd_Mean)
            x_nominal[:,step+1] = xd_current[:]

        return x_nominal, u_nominal

    def MPC_run_scenarioGP(self, bo_MC, un, GPNMPC2, Ufcn2, args2, sf2, ell, Conp_back_off,sn2):
        '''
        Used to calculate backoffs
        Simulates the MPC routine by a data driven approch using the GP from the
        scenario perspective and lears on the new data that becomes available.
        It also compilates data given a control sequence.
        '''
        nd, invKsample, invKMPC       = self.nd, self.invKopt[:], self.invKopt[:]
        Xsample, Ysample              = self.Xnorm[:], self.Ynorm[:]
        Xmeasure, Ymeasure            = self.Xnorm[:], self.Ynorm[:]
        xd_current, nk, ngp           = self.x0, self.nk, self.ngp
        GP_predictor_np, Sigma_v      = self.GP_predictor_np, self.Sigma_v
        meanY, stdY, meanX, stdX      = self.meanY, self.stdY, self.meanX, self.stdX
        Online_MatrixInv, OCP_step_GP = self.Online_MatrixInv, self.OCP_step_GP
        MPC_params, load_varsopt      = self.MPC_params, self.load_varsopt
        save_varsopt                  = self.save_varsopt
        u_min                         = self.u_min
        Sigma_w0                      = self.Sigma_w0
        Sigma_w                       = self.Sigma_w
        meanfcn2                      = self.meanfcn2
        ndat0                         = self.ndat0


        ''' initialize data collectors '''
        xd_current  = xd_current + (np.random.multivariate_normal(np.zeros(nd),Sigma_w0))
        xd_current  = xd_current.clip(min=0)
        x_opt       = np.zeros((nd,nk+1));
        u_opt       = np.zeros((self.nu,nk))
        x_opt[:,0]  = xd_current[:]
        u_          = u_min
        x_pred      = np.zeros((nd,nk+1));
        x_pred[:,0] = 0.
        invKYMPC    = np.zeros((ndat0+nk, nd))
        XMPC        = np.zeros((ndat0+nk, nd+self.nu))

        for step in range(nk):
            self.print2(['Computing back-offs  Back_off_iter = ',un,'  step = ',step,'  MC_iter = ', bo_MC])
            load_varsopt(un, step, args2)

            ''' Solves open-loop optimization '''
            p_backoff         = Conp_back_off.reshape((ngp*nk) ,order='F')
            if self.learning:
                par = MPC_params(invKMPC,Xmeasure,Ymeasure)
            else:
                par = MPC_params(self.invKopt[:],self.Xnorm[:],self.Ynorm[:])
            args2[step]["p"]  = np.concatenate((xd_current,p_backoff,par,u_.flatten()))
            res               = GPNMPC2[step](**args2[step])
            self.print2(['solver status = ',GPNMPC2[step].stats()['return_status']])
            save_varsopt(step, res)

            ''' calculate u(k+1), x(k+1), xm(k+1) and invK(k+1), X(k+1),Y(k+1) '''
            Xsample, Ysample, invKsample, x_opt, xd_current, xnew_measured, xd_measured, u_opt = \
            OCP_step_GP(x_opt, Ufcn2[step], res, u_opt, step, xd_current,
                        invKsample,Xsample, Ysample, sf2, ell)
            u_ = u_opt[:,step]

            ''' calculate differences '''
            for ij in range(nd):
                invKYMPC[:step-nk,ij]  = np.array(DM(mtimes(invKMPC[ij],Ymeasure[:,ij]))).flatten()

            XMPC[:step-nk,:]  = Xmeasure
            x_in              = np.concatenate((x_opt[:,step],u_opt[:,step]))
            x_pred[:,step+1]  = np.array(DM(meanfcn2(x_in,XMPC,invKYMPC))).flatten()

            ''' update invK, X and Y '''
            xd_measured_norm     = (xd_measured - meanY)/stdY
            xnewmeasured_norm    = (xnew_measured - meanX)/stdX
            for ij in range(nd):
                invKMPC[ij]  =  Online_MatrixInv(invKMPC[ij],xnewmeasured_norm,
                       sf2[ij],Sigma_w[ij,ij]/(stdY[ij])**2,Xmeasure,ell[ij])
            Xmeasure          = np.vstack((Xmeasure,xnewmeasured_norm))
            Ymeasure          = np.vstack((Ymeasure,xd_measured_norm))

            if step == (nk-1):
                E_obj_data = res["f"]

        return x_opt, u_opt, E_obj_data, par, x_pred

    def MC_MPC_plant(self, GPNMPC2, Ufcn2, args2, Conp_back_off):
        ''' Do a multirun MC MPC on the plant include GP learning '''
        x0, MC_n_iter, nd, nk = self.x0, self.MC_n_iter, self.nd, self.nk
        nu, deltat, Sigma_v   = self.nu, self.deltat, self.Sigma_v
        meanY, stdY           = self.meanY, self.stdY
        meanX, stdX, simulator= self.meanX, self.stdX,self.simulator
        Online_MatrixInv      = self.Online_MatrixInv
        gpfcn, ngp, Sigma_v   = self.gpfcn, self.ngp, self.Sigma_v
        OCP_step_Plant        = self.OCP_step_Plant
        load_varsopt          = self.load_varsopt
        save_varsopt          = self.save_varsopt
        nu, nd                = self.nu, self.nd
        u_min                 = self.u_min
        hypopt                = self.hypopt
        Sigma_w0              = self.Sigma_w0
        Sigma_w               = self.Sigma_w


        sf2 = [0]*nd; sn2 = [0]*nd; ell = [0]*nd;
        for ii in range(nd):
            sf2[ii] = np.exp(2*hypopt[nd+nu,ii])
            sn2[ii] = np.exp(2*hypopt[nd+nu+1,ii])
            ell[ii] = np.exp(2*hypopt[:nd+nu,ii])

        ''' initlialize data collectors '''
        Xd_plant   = np.zeros((nd, nk+1, MC_n_iter))
        Conp_plant = np.zeros((ngp,nk+1, MC_n_iter))
        u_opt      = np.zeros((nu, nk,   MC_n_iter))
        u_         = u_min

        ''' each MC run '''
        for MC_i in range(MC_n_iter):
            MPC_params, invKMPC   = self.MPC_params,self.invKopt[:]
            Xmeasure, Ymeasure    = self.Xnorm[:], self.Ynorm[:]

            ''' initlialize MC run '''
            xd_current            = x0 + (np.random.multivariate_normal(np.zeros(nd),Sigma_w0))
            Xd_plant[:,0,MC_i]    = xd_current[:]
            Conp_plant[:,0,MC_i]  = np.array(DM(gpfcn(xd_current))).flatten()
            t0is, tfi             = 0., 0.
            x_opt       = np.zeros((nd,nk+1));
            x_opt[:,0]  = xd_current[:]
            xd_measured = xd_current[:]

            ''' each step '''
            for step in range(nk):
                self.print2(['running MPC    Mc iter = ',MC_i,'  step = ',step, '   PLANT'])
                load_varsopt(MC_i, step, args2)

                ''' Solve open-loop optimization '''
                tfi              += deltat
                p_backoff         = Conp_back_off.reshape((ngp*nk) ,order='F')
                par               = MPC_params(invKMPC, Xmeasure, Ymeasure)
                args2[step]["p"]  = np.concatenate((xd_measured.flatten(),p_backoff,par,u_.flatten())) # define current value of states
                res               = GPNMPC2[step](**args2[step])                          # solve nominal model (mean GP)
                self.print2(['solver status = ',GPNMPC2[step].stats()['return_status']])
                save_varsopt(step, res)

                ''' calculate u(k+1), x(k+1), xm(k+1) '''
                x_opt, xd_current, xnew_measured, xd_measured, u_opt =\
                OCP_step_Plant(x_opt, Ufcn2[step], res, u_opt, step,
                               xd_current, tfi, t0is, MC_i)
                u_ = u_opt[:,step,MC_i]

                ''' update invK, X and Y '''
                xd_measured_norm     = (xd_measured - meanY)/stdY
                xnewmeasured_norm    = (xnew_measured - meanX)/stdX
                for ij in range(nd):
                    invKMPC[ij]  =  Online_MatrixInv(invKMPC[ij],xnewmeasured_norm,
                           sf2[ij],Sigma_w[ij,ij]/(stdY[ij])**2,Xmeasure,ell[ij])
                Xmeasure          = np.vstack((Xmeasure,xnewmeasured_norm))
                Ymeasure          = np.vstack((Ymeasure,xd_measured_norm))

                ''' collect results '''
                Xd_plant[:,step+1,MC_i]   = xd_current[:]
                Conp_plant[:,step+1,MC_i] = np.array(DM(gpfcn(xd_current))).flatten()
                t0is                     += deltat

        return  Xd_plant, Conp_plant, u_opt

    def print2(self,printinput):
        path = os.path.dirname(__file__)
        name = os.path.basename(path)
        printinput = ''.join( str(a) for a in printinput )
        print(printinput)
        text_file = open("Output" + name + ".txt","a")
        text_file.write(printinput+'\n')
        text_file.close()
        return
