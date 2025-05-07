#This script plots in batch all TKE budget components of selected time block(s)

#Following import works for testing: adding module to sys path only. Delete if module installed.
import sys,os
sys.path.append(os.path.abspath('.'))

#imports:
from um_tke_budget import tkeBudget_consts
from um_tke_budget import tkeBudget
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
if 'os' not in sys.modules:
    import os

def main_0( time_block_Ind , funcHandleInd, termName ):
    """
    a cycle procedure for plotting one time block, one TKE budget component, one source term
    """
    funcHandle=termFunctionHandleList[funcHandleInd]
    budget_term_NameList=tkeBudget_consts.tkeBudget_termNames[funcHandle.__name__]
    print("Function "+ funcHandle.__name__ + " : " + termName)
    
    for k in range(len(exps)):
        exp=exps[k]
        print("\t resolution: "+ exp)
        fdir = f'{indir}{suite}_{exp}_{levs}a_p{file_letter}000.nc'
        VarList=["STASH_m01s13i190","STASH_m01s13i191","STASH_m01s13i192","STASH_m01s13i193",'p','pref', 'thref','u', 'v', 'w' ,'th','q_vapour',  'q_cloud_liquid_mass']#  ["STASH_m01s03i216"]
        ds_new= tkeBudget._ds_seg_slices(tkeBudget.prepareDataset(fdir,VarList), "time_0", 4, reform_ref_state="per_seg" )[time_block_Ind]#
        timeString=ds_new.coords["time_0"].values[0].strftime("%H:%M")+"--"+ ds_new.coords["time_0"].values[-1].strftime("%H:%M")
        figName= outdir + funcHandle.__name__ + "__" + timeString.replace(":",".") + "__" + termName + ".png"
        if (not l_fig_overide) and os.path.exists(figName):
            print("Skipping computation of this term as user selected non-overide.")
            break #skip the computation if don't overide and fig exists
        print("\t Time: " + timeString)

        if funcHandle== tkeBudget.e_SGSScale:
            full_return= tkeBudget.e_SGSScale(ds_new,returnMean=[True,True,True,False] )
            for termID in range(len(full_return)):
                if budget_term_NameList[termID]!=termName:
                    continue
                else:
                    DDt_e= full_return[termID]
                    plt.plot(DDt_e , DDt_e.coords["z_p"].values, linestyle=lsList[ k % len(lsList)] ,label= exp)
                    plt.ylim([np.amin(ds_new.coords["z_p"].values),np.amax(ds_new.coords["z_p"].values)])
        else:
            DDt_e=tkeBudget.calculateBudgetComponents(ds_new, funcHandle , termName, returnMean=[True,True,True,False])
            plt.plot(DDt_e , DDt_e.coords["z_p"].values, linestyle=lsList[ k % len(lsList)] ,label= exp)
        plt.title(timeString)
        plt.legend()
    figName= outdir + funcHandle.__name__ + "__" + timeString.replace(":",".") + "__" + termName + ".png"
    if (not l_fig_overide) and os.path.exists(figName):
        print("Skipping plotting of this term as user selected non-overide.") #skip the plot if don't overide and fig exists
        return None
    plt.ylabel("z (m)")
    plt.xlabel(budgetfuncTermNames[funcHandle.__name__]+ " " + termName + r" ($m^2 s^{-3}$)")
    plt.gca().ticklabel_format(axis='x', scilimits=[-3,3])
    plt.switch_backend('agg') #set to avoid Tkinter thread RuntimeError
    plt.savefig(figName)
    plt.close()

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
    ##Term custom names: symbol representing TKE and budget components
    budgetfuncTermNames={
            "tkeBudg_eRes" :  r"$e^{res}$"  ,
            "tkeBudg_eUn":  r"$e^{un}$" , 
            "tkeBudg_eSub" : r"$e^{sub}$" ,
            "e_SGSScale":r"$e^{sgs}$"
        }
    ##Custom linestyles
    lsList=["-","-.","--"] 
    ##which time block? block no. start with 0
    time_block_ind= 1 
    ##which budget components? Function handle list
    termFunctionHandleList=[tkeBudget.tkeBudg_eRes, tkeBudget.tkeBudg_eUn, tkeBudget.tkeBudg_eSub,tkeBudget.e_SGSScale]
    ##whether or not overide existing figures? if False, skip the computing and plotting of existing figures
    l_fig_overide=False
    
    for function_handle_ind in range(0,len(termFunctionHandleList)):
        print(function_handle_ind, termFunctionHandleList[function_handle_ind])
        termReturnList=tkeBudget_consts.tkeBudget_termNames[termFunctionHandleList[function_handle_ind].__name__]
        for termName in termReturnList[1::1]: #plot except for "Budget"
            main_0( time_block_ind , function_handle_ind, termName )