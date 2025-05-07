#This script plots some additional terms that may be useful.


#Following import works for testing: adding module to sys path only. Delete if module installed.
import sys,os
sys.path.append(os.path.abspath('.'))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from monc_utils.thermodynamics import thermodynamics, thermodynamics_constants
from um_tke_budget import tkeBudget_consts
from um_tke_budget import tkeBudget
if 'os' not in sys.modules:
    import os

if __name__ == '__main__':
    #GLOBAL VARS SPECS: 
    ##suite info 
    suite = 'de247'
    basedir = './examples_source_codes/'
    indir = f'{basedir}/exemplar_database/'
    outdir= f'{basedir}/exemplar_outputs/'
    file_letter ='r'  # Full 3D fields
    exps = [
        '10km',
        '1km',
        '500m',
        ]
    levs = 'L100'
    lsList=["-","-.","--"] 
    cp=thermodynamics_constants.cp_air
    
    ##specify a demo resolution: 1km
    exp=exps[1]
    fdir = f'{indir}{suite}_{exp}_{levs}a_p{file_letter}000.nc'
    d = xr.open_dataset(fdir,decode_times=True,engine="netcdf4", chunks="auto")
    VarList=["STASH_m01s13i190","STASH_m01s13i191","STASH_m01s13i192","STASH_m01s13i193",'p','pref', 'thref','u', 'v', 'w' ,'th','q_vapour',  'q_cloud_liquid_mass']#  ["STASH_m01s03i216"]
    ds_new= tkeBudget.prepareDataset(fdir,VarList)
    ds_new_list= tkeBudget._ds_seg_slices(ds_new, "time_0", 4)
    del ds_new
    ##c file for 1km
    ds_new_list_c=tkeBudget._ds_seg_slices(tkeBudget.prepareDataset2( f'{indir}{suite}_{exp}_{levs}a_pc000.nc',["STASH_m01s03i216"]), "time_0", 4)
    term_names_eRes= ["Budget" , "transport", "Buoy",  "Dissip"]
    term_names_eRes_tp= ["Pressure transport" , "Shear transport", "Pressure height-var ",  "Shear height-var"]
    term_names_eUn=["Budget", "transport", "PressCorre", "Buoy", "Stress", "Dissip"]
    term_names_eSGS=[r"$f_m$", "Ri", r"$  Epsilon $" , r"$ e^s $"]
    ##global controls:
    l_fig_overide=True #whether or not overide existing figures? if False, skip the computing and plotting of existing figures
    plt.rcParams['figure.constrained_layout.use'] = True #constrained layout figures

    
    #plot 1km resolution: heat fluxes thoughout time blocks
    figName= outdir + "heat flux through time, resolution "+ exp
    for k in range(len(ds_new_list)):
        if (not l_fig_overide) and os.path.exists(figName):
            break 
        plt.subplot(1,2,1)
        h= -np.multiply(ds_new_list[k]["rho_ref"], np.multiply(tkeBudget.nu_h(ds_new_list[k]), tkeBudget.d_dxi( ds_new_list[k]['th'] , 2 ) ).mean(dim=["x_p","y_p","time_0"]) )
        plt.plot(h*cp,ds_new_list[k].coords["z_p"].values, linestyle=lsList[ k % len(lsList)] , 
                     label= ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M") )
        plt.xlabel(r" $-\rho_s C_p  h_3^{\theta}$ ($W m^{-2}$) ")
        plt.ylabel("z (m)")
        plt.xlim([-50,350])
        plt.ylim([0,3000])
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(ds_new_list_c[k]['STASH_m01s03i216'].mean(dim=["x_p","y_p","time_0"]),ds_new_list_c[k].coords["z_p"].values, linestyle=lsList[ k % len(lsList)] , 
                     label= ds_new_list_c[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list_c[k].coords["time_0"].values[-1].strftime("%H:%M") )
        plt.xlabel(r" UM Boundary Layer Heat Flux ($W m^{-2}$)")
        plt.ylabel("z (m)")
        plt.xlim([-50,350])
        plt.ylim([0,3000])
        plt.legend()
    if (l_fig_overide) or (not os.path.exists(figName) ):
        plt.switch_backend('agg') #set to avoid Tkinter thread RuntimeError
        plt.savefig(figName)
        plt.close()
    else:
        print("Skipping figure " + figName + " because the user specified non-overide.")
    
    #plot 1km resolution: sub-grid scale metrics thoughout time blocks
    figName=outdir + "SGS metrics through time, resolution "+ exp
    for k in range(len(ds_new_list)):
        if (not l_fig_overide) and os.path.exists(figName):
            break 
        funcHandles=[tkeBudget.nu_m, tkeBudget.nu_h, tkeBudget.SMAG_S, tkeBudget.SMAG_L]
        funcNames=[r"$ \nu _m$", r"$ \nu _h$", r"$ S_{ij}$", r"$ l _{m}$" ]
        for i in range(len(funcHandles)):
            plt.subplot(2,2,i+1)
            plt.plot(funcHandles[i](ds_new_list[k]).mean(dim=["x_p","y_p","time_0"]), ds_new_list[k].coords["z_p"].values, linestyle=lsList[ k % len(lsList)] , 
                     label= ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M")   )
            plt.xlabel(funcNames[i])
            plt.ylabel("z (m)")
            plt.legend()
    if (l_fig_overide) or (not os.path.exists(figName) ):
        plt.switch_backend('agg') #set to avoid Tkinter thread RuntimeError
        plt.savefig(figName)
        plt.close()
    else:
        print("Skipping figure " + figName + " because the user specified non-overide.")
    
    #plot 1km resolution: buoyant production compared with heat fluxes
    figName=outdir + "Buoyancy production and heat fluxes through time, resolution "+ exp
    for k in range(len(ds_new_list)):
        if (not l_fig_overide) and os.path.exists(figName):
            break
        plt.subplot(1,3,1)
        w_prime=ds_new_list[k]['w']-tkeBudget.resol_DA(ds_new_list[k]['w'])#ds_new_list[k]['w'].mean(dim=["x_p","y_p"]).expand_dims(dim={"x_p":ds_new_list[k]['w'].coords["x_p"].values ,"y_p":ds_new_list[k]['w'].coords["y_p"].values})
        conv=np.multiply(w_prime, ds_new_list[k]['th']-tkeBudget.resol_DA(ds_new_list[k]['th'])).mean(dim=["time_0","x_p","y_p"])
        del w_prime
        plt.plot(conv.chunk("auto"), ds_new_list[k].coords["z_p"].values, linestyle=lsList[ k % len(lsList)] , 
                     label= ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M") )
        plt.xlabel(r"$ (w^{\prime} \theta^ { \prime } )^r ( K\cdot m  \cdot s^{-1} ) $")
        plt.ylabel("z (m)")
        plt.legend() 
        
        plt.subplot(1,3,2)
        h_3_theta_over_rho= -np.multiply( tkeBudget.nu_h(ds_new_list[k]), tkeBudget.d_dxi( tkeBudget.resol_DA(ds_new_list[k]["th"]) , 2 ) ).mean(dim=["x_p","y_p","time_0"])
        plt.plot(h_3_theta_over_rho.chunk("auto"), ds_new_list[k].coords["z_p"].values, linestyle=lsList[ k % len(lsList)] , 
                     label= ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M") )
        plt.xlabel(r"$ \frac{h_3^{\theta}}{\rho _s} ( K\cdot m  \cdot s^{-1} ) $")
        plt.ylabel("z (m)")
        plt.legend()        

        plt.subplot(1,3,3)
        total= h_3_theta_over_rho + conv
        plt.plot(total.chunk("auto"), ds_new_list[k].coords["z_p"].values, linestyle=lsList[ k % len(lsList)] , 
                     label= ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M") )
        del total
        plt.xlabel(r"$ (w^{\prime} \theta^{\prime})^r + \frac{h_3^{\theta}}{\rho _s} ( K\cdot m  \cdot s^{-1} ) $")
        plt.ylabel("z (m)")
        plt.legend()
    if (l_fig_overide) or (not os.path.exists(figName) ):
        plt.switch_backend('agg') #set to avoid Tkinter thread RuntimeError
        plt.savefig(figName)
        plt.close()
    else:
        print("Skipping figure " + figName + " because the user specified non-overide.")
    
    #plot 1km resolution: directional components of resolved scale KE
    figName=outdir + "Directional e_res through time, resolution "+ exp
    for k in range(len(ds_new_list)):
        if (not l_fig_overide) and os.path.exists(figName):
            break    
        plt.subplot(2,2,1)
        plt.plot( np.power(tkeBudget.resol_DA(ds_new_list[k].u).mean(dim=["x_p","y_p","time_0"]).values, 2), ds_new_list[k].coords["z_p"].values,
                 label= ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M") )
        plt.xlabel(r"$(u^r)^2 (m^2 s^{-2})$")
        plt.ylabel("Height (m)")
        plt.legend()

        plt.subplot(2,2,2)
        plt.plot( np.power(tkeBudget.resol_DA(ds_new_list[k].v).mean(dim=["x_p","y_p","time_0"]).values, 2), ds_new_list[k].coords["z_p"].values,
                 label= ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M") )
        plt.xlabel(r"$(v^r)^2 (m^2 s^{-2})$")
        plt.ylabel("Height (m)")
        plt.legend()

        plt.subplot(2,2,3)
        plt.plot( np.power(tkeBudget.resol_DA(ds_new_list[k].w).mean(dim=["x_p","y_p","time_0"]).values, 2), ds_new_list[k].coords["z_p"].values,
                 label= ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M") )
        plt.xlabel(r"$(w^r)^2 (m^2 s^{-2})$")
        plt.ylabel("Height (m)")
        plt.legend()

        plt.subplot(2,2,4)
        plt.plot(tkeBudget.e_GridScales(ds_new_list[k],mode="res").mean(dim=["x_p","y_p","time_0"]).values , ds_new_list[k].coords["z_p"].values,
                 label= ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M") )
        
        plt.xlabel(r"$e^{res} (m^2 s^{-2})$")
        plt.ylabel("Height (m)")
        plt.legend()
    if (l_fig_overide) or (not os.path.exists(figName) ):
        plt.switch_backend('agg') #set to avoid Tkinter thread RuntimeError
        plt.savefig(figName)
        plt.close()
    else:
        print("Skipping figure " + figName + " because the user specified non-overide.")
    
    
    #plot 1km resolution: sub-grid scale tendency and budgets throughout time blocks
    figName=outdir + "SGS TKE tendency and budgets through time, resolution "+ exp
    for k in range(len(ds_new_list)):
        if (not l_fig_overide) and os.path.exists(figName):
            break
        DDt_eSGS=tkeBudget.e_SGSScale(ds_new_list[k],returnMean=[True,True,True,False], d=8.49)
        for i in range(len(DDt_eSGS)):
            plt.subplot(2,2,i+1)
            plt.plot(DDt_eSGS[i].chunk("auto"), DDt_eSGS[i].coords["z_p"].values, linestyle=lsList[ k % len(lsList)] , 
                     label= ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M"))
            plt.xlabel(term_names_eSGS[i])
            plt.ylabel("z (m)")
            plt.legend()
    if (l_fig_overide) or (not os.path.exists(figName) ):
        plt.switch_backend('agg') #set to avoid Tkinter thread RuntimeError
        plt.savefig(figName)
        DDt_eSGS=[]
        plt.close()
    else:
        print("Skipping figure " + figName + " because the user specified non-overide.")

    #plot 1km resolution: resolved scale tendency and budgets throughout time blocks
    figName=outdir + "Resolved KE e_Res tendency and budgets through time, resolution "+ exp
    for k in range(0,len(ds_new_list)):
        if (not l_fig_overide) and os.path.exists(figName):
            break
        DDt_eRes=tkeBudget.tkeBudg_eRes(ds_new_list[k], returnMean=[True,True,True,False])
        for i in range(len(DDt_eRes)):
            plt.subplot(2,2,i+1)
            plt.plot(DDt_eRes[i], DDt_eRes[i].coords["z_p"], linestyle=lsList[ k % len(lsList)] ,
                     label=  ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M"))
            plt.title(term_names_eRes[i])
            plt.xlabel(r"$e^{Res}$ tendency ($ m^2 s^{-3}$)")
            plt.ylabel("Height (m)")
            plt.legend()
    if (l_fig_overide) or (not os.path.exists(figName) ):
        plt.switch_backend('agg') #set to avoid Tkinter thread RuntimeError
        plt.savefig(figName)
        DDt_eRes=[]
        plt.close()
    else:
        print("Skipping figure " + figName + " because the user specified non-overide.")
    
    #plot 1km resolution: resolved scale transport terms budgets throughout time blocks
    figName=outdir + "Resolved KE e_Res tendency and budgets (Transport terms only) through time, resolution "+ exp
    for k in range(0,len(ds_new_list)):
        if (not l_fig_overide) and os.path.exists(figName):
            break
        DDt_eRes=tkeBudget.tkeBudg_eRes(ds_new_list[k], returnMean=[True,True,True,False],returnOnlyTransport=True)
        for i in range(len(DDt_eRes)):
            plt.subplot(2,2,i+1)
            plt.plot(DDt_eRes[i], DDt_eRes[i].coords["z_p"], linestyle=lsList[ k % len(lsList)] ,
                     label=  ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M"))
            plt.title(term_names_eRes_tp[i])
            plt.xlabel(r"$e^{Res}$ tendency ($ m^2 s^{-3}$)")
            plt.ylabel("Height (m)")
            plt.legend()
    if (l_fig_overide) or (not os.path.exists(figName) ):
        plt.switch_backend('agg') #set to avoid Tkinter thread RuntimeError
        plt.savefig(figName)
        DDt_eRes=[]
        plt.close()
    else:
        print("Skipping figure " + figName + " because the user specified non-overide.")
    
    #plot 1km resolution: unresolved scale (turbulent) tendency and budgets throughout time blocks
    figName=outdir + "Unresolved TKE e_Un tendency and budgets through time, resolution "+ exp
    for k in range(len(ds_new_list)):
        if (not l_fig_overide) and os.path.exists(figName):
            break
        DDt_eUn=tkeBudget.tkeBudg_eUn(ds_new_list[k],returnMean=[True,True,True,False])
        for i in range(len(DDt_eUn)):
            plt.subplot(2,3,i+1)
            plt.plot(DDt_eUn[i].chunk("auto"), DDt_eUn[i].coords["z_p"], linestyle=lsList[ k % len(lsList)] ,
                     label=  ds_new_list[k].coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new_list[k].coords["time_0"].values[-1].strftime("%H:%M"))
            plt.title(term_names_eUn[i])
            plt.xlabel(r"$e^{Un}$ tendency ($ m^2 s^{-3}$)")
            plt.ylabel("Height (m)")
            plt.legend()
    if (l_fig_overide) or (not os.path.exists(figName) ):
        plt.switch_backend('agg') #set to avoid Tkinter thread RuntimeError
        plt.savefig(figName)
        DDt_eUn=[]
        plt.close()
    else:
        print("Skipping figure " + figName + " because the user specified non-overide.")
    