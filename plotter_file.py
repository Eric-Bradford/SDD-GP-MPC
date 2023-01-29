# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:53:58 2019

@author: eadrc
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
ngp = 2
print('here')
names_l = ['Xd_MC', 'Conp_MC', 'Ud_MC', 'ALL_Conp_back_off', 'Xd_plant2', 'u_opt2',
                 'Conp_plant2', 'Eobj_GP_MC']
'''
============== specifications of problem size ==============
'''
nk = 10 ; ndat0 = 100; backoff_repeats = 5; backoff_MC = 100; MC_n_iter = 1
'''
============== specifications of problem type ==============
'''
nd = 5; nu = 2; tf = 4.; back_off = 1
t_X = np.linspace(0, tf, nk+1, endpoint=True)

ref       = str(nk)+'_'+str(ndat0)+'_'+str(backoff_repeats)
folder    = 'PlotFiles/5States_offline'
folderplt = 'PlotFiles/plots'
'''
============== GP Plots ==============
'''
PorGP = 'GP'

dict_dat = {}
for li in range(len(names_l)):
    with open(folder+'/'+str(names_l[li])+ref+"_file"+".pkl", 'rb') as a_file:
        dict_dat[names_l[li]] = pickle.load(a_file)

for j in range(nd):
    for k in range(backoff_repeats):
        plt.figure()
        for i in range(backoff_MC):
            plt.plot(t_X,list(dict_dat['Xd_MC'][j,:,i,k]),'-')
        plt.ylabel(PorGP+' repeat '+str(k)+' x_'+str(j))
        plt.xlabel('time')
        #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
        plt.savefig(folderplt+'/'+' repeat '+str(k)+'x_'+str(j)+PorGP+'.png', dpi=150)
        #plt.show()
        plt.close()

for j in range(nu):
    for k in range(backoff_repeats):
        plt.figure()
        for i in range(backoff_MC):
            plt.step(t_X[:-1],list(dict_dat['Ud_MC'][j,:,i,k]),'-')
        plt.ylabel(PorGP+' repeat '+str(k)+' u_'+str(j))
        plt.xlabel('time')
        #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
        plt.savefig(folderplt+'/'+' repeat '+str(k)+'u_'+str(j)+PorGP+'.png', dpi=150)
        plt.close()

Conp_data_mean = np.mean(dict_dat['Conp_MC'], axis=2)
Conp_data_std  = np.std(dict_dat['Conp_MC'], axis=2)
''' plot of last iteration of the back-off '''
for j in range(ngp):
    plt.figure()
    for i in range(backoff_MC):
        plt.plot(t_X[1:],list(dict_dat['Conp_MC'][j,:,i]),'--', color='grey')
    plt.plot(t_X[1:],list((Conp_data_mean + back_off*Conp_data_std)[j,:]),'-', color='black')
    plt.plot(t_X[1:],list((Conp_data_mean - back_off*Conp_data_std)[j,:]),'-', color='black')
    plt.plot(t_X[1:],[0.0 for i in range(len(Conp_data_mean[j,:]))],'-.', color='black')
    plt.ylabel(PorGP+'constraint '+str(j))
    plt.xlabel('time')
    #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
    plt.savefig(folderplt+'/'+PorGP+' path constraint '+str(j)+'.png', dpi=150)
    #plt.show()
    plt.close()

c_ = [(backoff_repeats - float(i))/backoff_repeats for i in range(backoff_repeats)]
for j in range(ngp):
    plt.figure()
    for i in range(backoff_repeats):
        plt.plot(t_X[1:],list(dict_dat['ALL_Conp_back_off'][j,:,i]),'--', color=str(c_[i]))
    plt.ylabel(PorGP+'back-off '+str(j))
    plt.xlabel('time')
    #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
    plt.savefig(folderplt+'/'+PorGP+' ALL path constraint '+str(j)+'.png', dpi=150)
    #plt.show()
    plt.close()


'''
============== Plant Plots ==============
'''
PorGP = 'Plant'

t_X = np.linspace(0, tf, nk+1, endpoint=True)
for j in range(nd):
    plt.figure()
    for i in range(MC_n_iter):
        plt.plot(t_X,list(dict_dat['Xd_plant2'][j,:,i]),'-')
    plt.ylabel('plant x_'+str(j))
    plt.xlabel('time')
    #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
    plt.savefig(folderplt+'/'+'x_'+str(j)+PorGP+'_.png', dpi=150)
    #plt.show()
    plt.close()

for j in range(nu):
    plt.figure()
    for i in range(MC_n_iter):
        plt.step(t_X[:-1],list(dict_dat['u_opt2'][j,:,i]),'-')
    plt.ylabel('plant control u_'+str(j))
    plt.xlabel('time')
    #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
    plt.savefig(folderplt+'/'+'u_'+str(j)+PorGP+'.png', dpi=150)
    #plt.show()
    plt.close()

Conp_data_mean = np.mean(dict_dat['Conp_plant2'], axis=2)
Conp_data_std  = np.std(dict_dat['Conp_plant2'], axis=2)
for j in range(ngp):
    plt.figure()
    for i in range(MC_n_iter):
        plt.plot(t_X,list(dict_dat['Conp_plant2'][j,:,i]),'--', color='grey')
    plt.plot(t_X,list((Conp_data_mean+back_off*Conp_data_std)[j,:]),'-', color='black')
    plt.plot(t_X,list((Conp_data_mean-back_off*Conp_data_std)[j,:]),'-', color='black')
    plt.plot(t_X,[0.0 for i in range(len(Conp_data_mean[j,:]))],'-.', color='black')
    plt.ylabel(PorGP+'_path_constraint_'+str(j))
    plt.xlabel('time')
    #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
    plt.savefig(folderplt+'/'+PorGP+'path constraint '+str(j)+'.png', dpi=150)
    #plt.show()
    plt.close()

plt.figure()
plt.plot(list(dict_dat['Eobj_GP_MC']),'--')
plt.ylabel('objective')
plt.xlabel('iterations')
#plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
plt.savefig(folderplt+'/'+'objective in back-offs.png', dpi=150)
#plt.show()
plt.close()

names_l = ['Xd_plant_nom', 'u_opt_nom','Conp_plant_nom']

PorGP = 'Nominal'

dict_dat = {}
for li in range(len(names_l)):
    with open(folder+'/'+str(names_l[li])+ref+"_file"+".pkl", 'rb') as a_file:
        dict_dat[names_l[li]] = pickle.load(a_file)

t_X = np.linspace(0, tf, nk+1, endpoint=True)
for j in range(nd):
    plt.figure()
    for i in range(MC_n_iter):
        plt.plot(t_X,list(dict_dat['Xd_plant_nom'][j,:,i]),'-')
    plt.ylabel('plant x_'+str(j))
    plt.xlabel('time')
    #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
    plt.savefig(folderplt+'/'+'x_'+str(j)+PorGP+'_.png', dpi=150)
    #plt.show()
    plt.close()

for j in range(nu):
    plt.figure()
    for i in range(MC_n_iter):
        plt.step(t_X[:-1],list(dict_dat['u_opt_nom'][j,:,i]),'-')
    plt.ylabel('plant control u_'+str(j))
    plt.xlabel('time')
    #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
    plt.savefig(folderplt+'/'+'u_'+str(j)+PorGP+'.png', dpi=150)
    #plt.show()
    plt.close()

Conp_data_mean = np.mean(dict_dat['Conp_plant_nom'], axis=2)
Conp_data_std  = np.std(dict_dat['Conp_plant_nom'], axis=2)
for j in range(ngp):
    plt.figure()
    for i in range(MC_n_iter):
        plt.plot(t_X,list(dict_dat['Conp_plant_nom'][j,:,i]),'--', color='grey')
    plt.plot(t_X,list((Conp_data_mean+back_off*Conp_data_std)[j,:]),'-', color='black')
    plt.plot(t_X,list((Conp_data_mean-back_off*Conp_data_std)[j,:]),'-', color='black')
    plt.plot(t_X,[0.0 for i in range(len(Conp_data_mean[j,:]))],'-.', color='black')
    plt.ylabel(PorGP+'_path_constraint_'+str(j))
    plt.xlabel('time')
    #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
    plt.savefig(folderplt+'/'+PorGP+'path constraint '+str(j)+'.png', dpi=150)
    #plt.show()
    plt.close()


'''
============== Objectives Plots ==============
'''

names_l = ['Xd_plant2']

PorGP = 'Objective_Learning'

dict_dat = {}
for li in range(len(names_l)):
    with open(folder+'/'+str(names_l[li])+ref+"_file"+".pkl", 'rb') as a_file:
        dict_dat[names_l[li]] = pickle.load(a_file)

t_X = np.linspace(0, tf, nk+1, endpoint=True)

plt.figure()
for i in range(MC_n_iter):
    plt.plot(t_X,list(dict_dat['Xd_plant2'][2,:,i]*dict_dat['Xd_plant2'][4,:,i]),'-')
plt.ylabel('plant objective_'+str(j))
plt.xlabel('time')
#plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
plt.savefig(folderplt+'/'+'objective_'+str(j)+PorGP+'_.png', dpi=150)
#plt.show()
plt.close()

'''
============== Nominal and Non-nominal plots Plots ==============
'''
names_l = ['Conp_plant2','Conp_plant_nom']
PorGP = 'Nominal_vs_Backoff'
from matplotlib.legend_handler import HandlerLine2D
dict_dat = {}
for li in range(len(names_l)):
    with open(folder+'/'+str(names_l[li])+ref+"_file"+".pkl", 'rb') as a_file:
        dict_dat[names_l[li]] = pickle.load(a_file)

Conp_data_mean_nom = np.mean(dict_dat['Conp_plant_nom'], axis=2)
Conp_data_std_nom  = np.std(dict_dat['Conp_plant_nom'], axis=2)
Conp_data_mean = np.mean(dict_dat['Conp_plant2'], axis=2)
Conp_data_std  = np.std(dict_dat['Conp_plant2'], axis=2)
for j in range(ngp):
    plt.figure()
    for i in range(MC_n_iter):
        plt.plot(t_X,list(dict_dat['Conp_plant_nom'][j,:,i]),'--', color='grey')
        plt.plot(t_X,list(dict_dat['Conp_plant2'][j,:,i]),'--', color='grey')
    plt.plot(t_X,list((Conp_data_mean+back_off*Conp_data_std)[j,:]),'-', color='black', label='robust')
    plt.plot(t_X,list((Conp_data_mean-back_off*Conp_data_std)[j,:]),'-', color='black')
    plt.plot(t_X,list((Conp_data_mean_nom+back_off*Conp_data_std_nom)[j,:]),'--', color='blue', label='nominal')
    plt.plot(t_X,list((Conp_data_mean_nom-back_off*Conp_data_std_nom)[j,:]),'--', color='blue')
    plt.plot(t_X,[0.0 for i in range(len(Conp_data_mean[j,:]))],'-.', color='black')
    plt.ylabel(PorGP+'_path_constraint_'+str(j))
    plt.legend(loc='center right')
    plt.xlabel('time')
    #plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
    plt.savefig(folderplt+'/'+PorGP+'path constraint comparison '+str(j)+'.png', dpi=150)
    #plt.show()
    plt.close()





print('i am done')

#
