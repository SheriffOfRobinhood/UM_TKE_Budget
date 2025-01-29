#This script plots all the KE components.

#Following import works for testing: adding module to sys path only. Delete if module installed.
import sys,os
sys.path.append(os.path.abspath('.'))

#imports:
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from um_tke_budget import tkeBudget

if __name__ == '__main__':
    #GLOBAL VARS SPECS: 
    ##suite info
    suite = 'de247'
    basedir = './examples_source_codes/'
    indir = f'{basedir}/exemplar_database/'
    outdir= f'{basedir}/exemplar_outputs/'
    #file_letter = 'a'  # 1D profiles
    file_letter ='r' #'c' # Full 3D fields
    exps = [
        '10km',
        '1km',
        '500m',
        ]
    levs = 'L100'
    ##Custom linestyles
    lsList=["-","-.","--"]
    ##which time block? block no. start with 0
    time_block_Ind = 1
    ##what components? resolved, unresolved ("turbulent"), explicit and sub-grid scale
    gridscale_comps=["exp","res","un"]
    l_sgsscale=True
    e_s=[]    

    #plot the gridscale components versus resolution
    for comp in gridscale_comps:    
        for k in range(len(exps)):
            exp=exps[k]
            print("\t resolution: "+ exp)
            fdir = f'{indir}{suite}_{exp}_{levs}a_p{file_letter}000.nc'
            VarList=["STASH_m01s13i190","STASH_m01s13i191","STASH_m01s13i192","STASH_m01s13i193",'p','pref', 'thref','u', 'v', 'w' ,'th', 'q_vapour',  'q_cloud_liquid_mass' ]
            ds_new=tkeBudget._ds_seg_slices(tkeBudget.prepareDataset(fdir,VarList), "time_0", 4)[time_block_Ind]#
            timeString=ds_new.coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new.coords["time_0"].values[-1].strftime("%H:%M")
            e_grid=tkeBudget.e_GridScales(ds_new, mode=comp).mean(dim=["x_p","y_p", "time_0"])
            if l_sgsscale and ( len(e_s) < len(exps) ): #read and calculate the sub-grid scale TKE
                e_s.append( tkeBudget.e_SGSScale(ds_new,returnMean=[True,True,True,False])[-1] )
            plt.plot(e_grid , ds_new.coords["z_p"].values, linestyle=lsList[ k % len(lsList)] ,label= exp)
        plt.title(timeString)
        plt.legend()
        plt.ylabel("z (m)")
        plt.ylim([np.amin(ds_new.coords["z_p"].values),np.amax(ds_new.coords["z_p"].values)])
        plt.switch_backend('agg') #set to avoid Tkinter thread RuntimeError
        plt.savefig(outdir+"e_"+ comp +" vs resolution.png")
        plt.close()
    
    #plot the sub-grid scale TKE
    if l_sgsscale:
        for k in range(len(exps)):
            exp=exps[k]
            timeString=ds_new.coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new.coords["time_0"].values[-1].strftime("%H:%M")
            plt.plot(e_s[k], e_s[k].coords["z_p"].values, linestyle=lsList[ k % len(lsList)] ,label= exp)
        plt.title(timeString)
        plt.legend()
        plt.ylabel("z (m)")
        plt.ylim([np.amin(ds_new.coords["z_p"].values),np.amax(ds_new.coords["z_p"].values)])
        plt.savefig(outdir+"e_sgs"+" vs resolution.png")
        plt.close()

    
