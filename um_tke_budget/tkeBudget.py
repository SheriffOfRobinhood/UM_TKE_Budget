
#imports:
import sys
from um_tke_budget import tkeBudget_consts
import numpy as np
import subfilter.spectra as SBFSpec
import subfilter as sf
import xarray as xr
from monc_utils.io_um.datain import get_um_field, get_um_data, get_um_data_on_grid, get_um_and_transform
from monc_utils.io_um.datain import stash_map as moncUtils_stashmap
from monc_utils.thermodynamics import thermodynamics, thermodynamics_constants
from monc_utils.data_utils import difference_ops
import matplotlib.pyplot as plt
import xarray.plot as xrplt
import datetime
if 'os' not in sys.modules:
    import os
import difflib
import subfilter.filters as filt
import subfilter.subfilter as sfsf
 

tkeBudget_termNames= tkeBudget_consts.tkeBudget_termNames

#Note that the following  will be a global variable!
#to be changed when running with filters!
filtOpts=tkeBudget_consts.filter_options
filter_global = None

def alloc_filter(filterInit_opts, otherKwArgDict={}):
    """construct the filter object according to specs in filtOpts."""
    print('initialising filter according to following filter options:')
    print('\t', filterInit_opts, '\n')
    filtOpts['filtInit_Opts'] = filterInit_opts
    filtOpts['filter_switch'] = True
    otherKwArgKeys=list(otherKwArgDict.keys())
    for key in otherKwArgKeys:
        filtOpts[key].update({key: otherKwArgDict[key]})
    filter_global = filt.Filter(**filterInit_opts)

def dealloc_filter():
    """reset global filter and options to default value, turn off the filter switch"""
    filter_global = None
    filtOpts=tkeBudget_consts.filter_options

def __checkAllNaN(DA, name):
    NaNCount=np.count_nonzero(np.isnan(DA.values))
    if NaNCount== np.count_nonzero(DA.values):
        print (name+ ":All non-zero are NaN")
        return True
    return False

def _meanCondense(DA,CondenseArray):
    """condense data by mean over CondenseArray=[Time,x,y,z]
    if all False return original data"""
    dimensions=["time_0","x_p","y_p","z_p"]
    dimDA=DA.dims
    dimList=[]
    Flag=False
    for i in range(len(CondenseArray)):
        if CondenseArray[i] and dimDA.count(dimensions[i]) !=0:
            dimList.append(dimensions[i])
            Flag=True
    if Flag:
        return DA.mean(dim=dimList)
    else:
        return DA

def _ds_seg_slices(ds, dim_name,stepsEverySegment, start=0, stop=None, includeEnd=False, reform_ref_state=""):
    """return list [ ] containing segments of slices in specified dimension.
    dim_name: str
    stepsEverySegment:
    reform_ref_state: 
        "" (default): no reform, use global init thref, pref, dimension=[z,]
        "per_seg": reform using init per seg slices, dimension=[z,]
        
    """
    ds_list=[]
    N=len(ds.coords[dim_name])
    print("creating segments for dimension:"+dim_name + "  total len:{0}".format(N))
    if stop==None: #all
        stop_=N
    else:
        stop_=stop
    i=0
    for start_seg in range(start,stop_, stepsEverySegment):
        stop_seg= min(start_seg+ stepsEverySegment, stop_) + (1 if includeEnd else 0)
        ds_new=ds.isel({dim_name : slice(start_seg, stop_seg, 1) })
        if reform_ref_state=="per_seg":
            ds_new=ds_new.assign({"rho_ref_new": rho_ref(
                ds_new["th"].mean(dim=["x_p","y_p"]).isel({dim_name : 0}),  ds_new["p"].mean(dim=["x_p","y_p"]).isel({dim_name : 0})  )})
            ds_new=ds_new.drop_vars(["rho_ref"]).rename({"rho_ref_new": "rho_ref"})
        print ("{0}\t".format(i), ds_new.coords[dim_name].values[0].strftime("%H:%M") ,ds_new.coords[dim_name].values[-1].strftime("%H:%M"),"\n\n" )
        ds_list.append(ds_new.chunk("auto"))
        i+=1
    return ds_list

def _toChangeLonLat_t_dims(ds_raw,varname):
    if varname.find("STASH_")==-1:
        return False
    DA=ds_raw[varname]
    #temp=DA.copy()
    change_flag=False
    for dimension in DA.dims:
        if dimension=="thlev_zsea_theta":
            change_flag=True
        if dimension=="longitude_t":
            change_flag=True
        if dimension=="latitude_t":
            change_flag=True
    return change_flag

def _Ri_from_fm_stable(fm_DA):
    """LEM stable"""
    Ri= 0.25*(fm_DA.copy(data=np.ones(np.shape(fm_DA.values))) - np.power(fm_DA, 0.25))
    return Ri
    

def _Ri_from_fm_Unstable(fm_DA, method="conventional"):
    """get Ri value from a value of stability function, under unstable
    method= 'conventional' or 'Std'
    """
    if method=="Std":
        c=16.0
    else: #use conventional
        c=1.43
    Ri= (fm_DA.copy(data=np.ones(np.shape(fm_DA.values))) - np.power(fm_DA, 2) ) / c   # (1-fm^2) /c
    return Ri

def _Ri_from_fm(fm_DA, outRangeFillValueDA, method="conventional"):
    """fm==1, Ri = 0
        fm==0, Ri = outRangeFillValue (value let all turbulence suppressed)
        0<fm<1, Ri = _Ri_from_fm_stable
        fm > 1, Ri= _Ri_from_fm_Unstable
    """
    Ri_un_DA= _Ri_from_fm_Unstable(fm_DA.where( np.greater(fm_DA, np.ones(np.shape(fm_DA.values)) )  , other=1.0),method=method) 
    Ri_stab_DA= _Ri_from_fm_stable(fm_DA.where( np.logical_and(np.greater(fm_DA, np.zeros(np.shape(fm_DA.values)) ) , np.less(fm_DA, np.ones(np.shape(fm_DA.values)) ) ), other=1.0))
    #Ri_outRange=fm_DA.copy(data=outRangeFillValue*np.ones(np.shape(fm_DA.values))).where( np.equal(fm_DA, np.zeros(np.shape(fm_DA.values))) , other=0.0)
    Ri_outRange=outRangeFillValueDA.where( np.equal(fm_DA, np.zeros(np.shape(fm_DA.values))) , other=0.0)
    return Ri_un_DA+Ri_stab_DA+Ri_outRange

def mu(ds):
    u_i_DA=[ds.u,ds.v,ds.w]
    return np.multiply(ds["STASH_m01s13i190"], ds["rho_ref"].expand_dims(
        dim={"x_p":u_i_DA[0].coords["x_p"].values ,"y_p":u_i_DA[0].coords["y_p"].values,"time_0":u_i_DA[0].coords["time_0"].values}))

def nu_m(ds):
    """return SMAG VISC_H"""
    return ds["STASH_m01s13i190"]

def nu_h(ds):
    """return SMAG VISC_H"""
    return ds["STASH_m01s13i191"]

def SMAG_S(ds):
    """return SMAG S"""
    return ds["STASH_m01s13i192"]

def SMAG_L(ds):
    """return SMAG Length Scale Lambda"""
    return ds["STASH_m01s13i193"]
    
def d_dxi(field,i,method="monc_util"):
    """return partial derivative over x_i on native grid
    method: "monc_util" (difference_ops) or XA(XArray's default differentiate)""" 
    x_i=["x_p","y_p","z_p"]
    if method=="monc_util":
        if field.name==None:
            field=field.rename("NoName")
        func_names=[ difference_ops.d_by_dx_field , difference_ops.d_by_dy_field , difference_ops.d_by_dz_field ] 
        func_names_native=[ difference_ops.d_by_dx_field_native , difference_ops.d_by_dy_field_native , difference_ops.d_by_dz_field_native ]
        coord_p=[field.coords["x_p"],field.coords["y_p"],field.coords["z_p"]]
        newField=func_names[i](field,None,field.coords["z_p"], grid='p')
        c_u= [c for c in newField.coords if "_u" in c]
        if c_u!=[]:
            newField=newField.drop_vars(c_u)    
        #newField.name="d["+newField.name+"] / d(" + x_i[i]
            print (newField)
        return newField.drop_indexes(x_i[i]).reindex({x_i[i]:coord_p[i]})
    else:
        return field.differentiate(x_i[i])
        
def e_GridScales(ds, mode="exp"):
    """return e in all grid-scale resolved data
    mode: "res",  "un" or "exp"
    """
    u_i_DA=[ds.u,ds.v,ds.w]
    x_i=["x_p","y_p","z_p"]
    e=u_i_DA[0].copy(data=np.zeros(np.shape(u_i_DA[0])))
    for i in range(len(u_i_DA)): 
        if mode=="res":
            e = e + 0.5* np.multiply( resol_DA(u_i_DA[i]) , resol_DA(u_i_DA[i]) ) 
        elif mode=="un":
            e = e + 0.5* ( resol_DA(np.multiply(u_i_DA[i],u_i_DA[i])) -  np.multiply(resol_DA(u_i_DA[i]),resol_DA(u_i_DA[i]))  )
        elif mode=="exp": #default
            e = e + 0.5* (np.multiply(u_i_DA[i],u_i_DA[i]))
    return e

def advec_term(ds,phi):
    """return nabla *cdot (phi *vec{u})
    phi: scalar DA, same shape as u 
    """
    u_i_DA=[ds.u,ds.v,ds.w]
    x_i=["x_p","y_p","z_p"]
    adv=phi.copy(data=np.zeros(np.shape(phi)))
    for i in range(len(x_i)):
        adv= adv + d_dxi(np.multiply(u_i_DA[i],phi),i)   
    return adv

def R_ij(ds, i,j,preCal=[]):
    """ return the (i,j) th component in Reynold stress term
    """
    if preCal!=[]: # R_ij already pre-calculated as 2-D lists:
        return preCal[i][j]
    else:
        u_i_DA=[ds.u, ds.v, ds.w]
    #x_i=["x_p","y_p","z_p"]
        return s(u_i_DA[i],u_i_DA[j])

def S_ij(ds,i,j):
    u_i_DA=[ds.u, ds.v, ds.w]
    x_i=["x_p","y_p","z_p"]
    return np.add(d_dxi(u_i_DA[i],j),  d_dxi(u_i_DA[j],j) )

def resol(A):
    """
    define resolved scale of A.
    A: numpy array 
    default assume a^r= a average 
    return: numpy array
    """
    return np.mean(A)

def resol_DA(DA, broadcast=True):
    if filtOpts["filter_switch"] == True:
        return sfsf.filtered_field_calc(DA, filtOpts, filter_global)[0]
    if not broadcast:
        return DA.mean(dim=['x_p','y_p'])
    else:
        dat=DA.mean(dim=['x_p','y_p'])
        return dat.expand_dims(dim={"x_p":DA.coords["x_p"].values ,"y_p":DA.coords["y_p"].values})

def s(A,B,broadcast=True):
    """
    define (a*b)^r - a^r*b^r.
    A, B: numpy array with same dimension
    default assume a^r= a average 
    return: numpy array
    """
    ab=np.multiply(A,B)
    if not broadcast:
        return resol(ab)-np.multiply(resol(A),resol(B))
    else: # broadcast, use when DA
        return resol_DA(ab)-np.multiply(resol_DA(A),resol_DA(B))
        
def s3(A,B,C):
    """define triple correlation s(A,B,C)"""
    abc= np.multiply( np.multiply(A,B), C)
    minusTerms=np.multiply(resol_DA(A),resol_DA(s(B,C)))+ np.multiply(resol_DA(B),resol_DA(s(A,C)))+ np.multiply(resol_DA(C),resol_DA(s(A,B)))
    return resol_DA(abc)-np.multiply(np.multiply(resol_DA(A),resol_DA(B)), resol_DA(C) ) -  minusTerms

def rho_ref(thref,pref):
    """
    thref,pref: DataArray
    return: DataArray
    """
    temp_ref= thermodynamics.temperature( thref , pref) 
    rho_ref_dat= np.divide(29*1e-3/8.314 *pref.values,   temp_ref.values) # rho= p*MW/RT
    return xr.DataArray( data=rho_ref_dat, coords=  pref.coords, name='rho_ref'    )
    

def tkeBudg_eRes_Pressure(ds,returnMean=[False,False,False,False]):
    u_i_DA=[ds.u, ds.v, ds.w]
    x_i=["x_p","y_p","z_p"]
    p_prime_r= resol_DA(np.subtract( 
        ds["p"],  ds["pref"].expand_dims(
            dim={"x_p":u_i_DA[0].coords["x_p"].values ,"y_p":u_i_DA[0].coords["y_p"].values,"time_0":u_i_DA[0].coords["time_0"].values })))
    rho_s=ds["rho_ref"].expand_dims(
        dim={"x_p":u_i_DA[0].coords["x_p"].values ,"y_p":u_i_DA[0].coords["y_p"].values,"time_0":u_i_DA[0].coords["time_0"].values})
    pressureTerm=np.divide(   p_prime_r,   rho_s  )     
    budget=pressureTerm.copy(data=np.zeros(np.shape(pressureTerm.values)))
    transport=pressureTerm.copy(data=np.zeros(np.shape(pressureTerm.values)))
    for i in range(len(u_i_DA)):
        pressureAdvec= np.multiply( pressureTerm, resol_DA(u_i_DA[i])    )
        transport = transport - d_dxi(pressureAdvec,i) # -d ( u_i^r p'^r / rho_s) / dxi
    height_varying = - np.multiply( np.multiply(resol_DA( ds.w ), np.divide( p_prime_r, np.power(rho_s,2)) )   ,  d_dxi( rho_s, 2 ) ) # (w^r p'^r / rho_s ^2)* (d(rho_s)/dz) 
    budget= budget + transport + height_varying 
    return [_meanCondense(budget,returnMean), _meanCondense(transport,returnMean), _meanCondense(height_varying,returnMean)]

def tkeBudg_eRes_Stress(ds,componentSwitch=[True,True,True],returnMean=[False,False,False,False]):
    """componentSwitch:[transport,height_varying,dissipation]
    returnMean: return mean over [Time, x,y,z] if True to save memory.
    """
    u_i_DA=[ds.u, ds.v, ds.w]
    x_i=["x_p","y_p","z_p"]
    rho_s=ds["rho_ref"].expand_dims(dim={"x_p":u_i_DA[0].coords["x_p"].values ,"y_p":u_i_DA[0].coords["y_p"].values,"time_0":u_i_DA[0].coords["time_0"].values})
    #print(rho_s)
    tau_ij=[]
    resol_DA_veloc_arr=[] # resolved velocity pre-calculation
    returnVal=[]
    if all(componentSwitch):
        Budget= rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
        returnVal.append(_meanCondense(Budget,returnMean))   
    #calculate tau_ij first: tau_ij= stress/ rho_s
    for i in range(len(u_i_DA)):
        tau_ij.append(list())
        resol_DA_veloc_arr.append(resol_DA(u_i_DA[i]))
        for j in range(len(u_i_DA)):
            tau_ij[i].append(
                -R_ij(ds, i,j)+resol_DA(np.multiply( S_ij(ds,i,j), np.divide(mu(ds),rho_s) )) )
    print("tau_ij calculated")
    #Transport term
    if componentSwitch[0]:
        transport=rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
        for i in range(len(u_i_DA)):
            for j in range(len(u_i_DA)):
                transport= np.add( transport, d_dxi( np.multiply( resol_DA_veloc_arr[i], tau_ij[i][j] ), j)) # d(..)/dxj
        print("Transport term")
        returnVal.append(_meanCondense(transport, returnMean))
        del transport
    else:
        returnVal.append(None)
    #Height varying term
    if componentSwitch[1]:
        height_varying=rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
        for i in range(len(u_i_DA)):
            height_varying=     height_varying + np.multiply(
                    d_dxi( rho_s, 2 ),
                    np.divide( np.multiply( resol_DA_veloc_arr[i], tau_ij[i][2] ), rho_s
                               )   ) 
        print("Height varying term")
        returnVal.append(_meanCondense(height_varying, returnMean)) 
        del height_varying
    else:
        returnVal.append(None)
    #dissipation term
    if componentSwitch[2]:
        dissipation=rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
        for i in range(len(u_i_DA)):
            for j in range(len(u_i_DA)):
                dissipation =  dissipation - np.multiply(tau_ij[i][j], d_dxi(resol_DA_veloc_arr[i],j) )  # -R_ij* du_i^r/ dx_j
        print("dissipation")   
        returnVal.append(_meanCondense(dissipation, returnMean)) 
        del dissipation
    else:
        returnVal.append(None)
    if all(componentSwitch):
        for i in range(1, len(returnVal)):
            returnVal[0] = returnVal[0] + returnVal[i]
    return returnVal

def tkeBudg_eRes_Buoy(ds,returnMean=[False,False,False,False]):
    th_v= thermodynamics.virtual_potential_temperature_monc(ds["th"], ds["thref"], ds["q_vapour"], ds["q_cloud_liquid_mass"])
    B_prime_r=resol_DA( thermodynamics.buoyancy_monc(th_v, ds["thref"]) )
    del th_v
    return  _meanCondense(np.multiply( resol_DA(ds.w) , B_prime_r ), returnMean )

def tkeBudg_eRes(ds, returnMean=[False,False,False,False], returnOnlyTransport=False, componentSwitch=[True,True,True]):
    """returns: [Budget , transport, Buoy,  Dissip] (if returnOnlyTransport==False)
    [transport_P, transport_S, height_var_P,  height_var_S ]
    componentSwitch: work only for returnOnlyTransport==False
    """
    if returnOnlyTransport:
        [P_Budg, P_trsp, P_heiVar]=tkeBudg_eRes_Pressure(ds,returnMean)
        Buoy=tkeBudg_eRes_Buoy(ds,returnMean)
        [S_Budg, S_trsp, S_heiVar, S_dissp]= tkeBudg_eRes_Stress(ds,componentSwitch=[True,True,True],returnMean=returnMean)
        return [ P_trsp, S_trsp, P_heiVar, S_heiVar  ]
    else:
        if componentSwitch[0]: # calculate transport
            [P_Budg, P_trsp, P_heiVar]=tkeBudg_eRes_Pressure(ds,returnMean)
            if all(componentSwitch): # all, cal budget
                Budget= P_Budg
            del P_Budg
            trsp=P_trsp+P_heiVar
            del P_trsp
            del P_heiVar
        else:
            trsp=None
        if componentSwitch[0] or componentSwitch[2]: #calculate transport or dissip
            [S_Budg, S_trsp, S_heiVar, S_dissip]= tkeBudg_eRes_Stress(ds,componentSwitch=[True,True,True],returnMean=returnMean)
            if all(componentSwitch):
                Budget = Budget + S_Budg 
            del S_Budg   
            if componentSwitch[0]:
                trsp = trsp + S_trsp + S_heiVar
            del  S_trsp 
            del  S_heiVar
            if componentSwitch[2]:
                dissip= S_dissip
            else:
                dissip=None
            del S_dissip
        else:
            dissip=None
        if componentSwitch[1]:
            Buoy=tkeBudg_eRes_Buoy(ds,returnMean)
            if all(componentSwitch):
                Budget = Budget + Buoy 
        else:
            Buoy=None
        if all(componentSwitch):
            return [Budget, trsp, Buoy, dissip]
        else:
            return [trsp, Buoy, dissip]
        #return [ P_Budg+ S_Budg+Buoy,  P_trsp+ P_heiVar+ S_trsp+S_heiVar, Buoy, S_dissp ]

def tkeBudg_eUn(ds,componentSwitch=[True,True,True,True,True],returnMean=[False,False,False,False],dissipationDef="Peter"):
    """return e^un in [Budget, transport, PressCorre, Buoy, Stress, Dissip]
    componentSwitch: for [transport, PressCorre, Buoy, Stress, Dissip]
    dissipationDef: "Peter" (default) if using Flux_budget Eqn 52
                    "Brown" if using Flux_budget Eqn 47        """
    u_i_DA=[ds.u, ds.v, ds.w]
    x_i=["x_p","y_p","z_p"]
    rho_s=ds["rho_ref"].expand_dims(dim={"x_p":u_i_DA[0].coords["x_p"].values ,"y_p":u_i_DA[0].coords["y_p"].values,"time_0":u_i_DA[0].coords["time_0"].values})
    returnVal=[]
    if all(componentSwitch):
        Budget= rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
        returnVal.append(_meanCondense(Budget,returnMean))
    if componentSwitch[0]: #calculate turbulent transport
        transport=rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
        for i in range(len(u_i_DA)):
            derivTerm=rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
            for k in range(len(u_i_DA)):
                s3RhoTerm= np.divide( 
                    s3(u_i_DA[i], u_i_DA[i], u_i_DA[k]) ,   rho_s    )   # s(u_i,u_i, u_k)/ rho_s
                derivTerm= derivTerm + d_dxi(s3RhoTerm, k ) # divergence 
            transport = transport - np.multiply(rho_s, derivTerm )/2.0
        returnVal.append(_meanCondense(transport,returnMean))
        del transport
    else: 
        returnVal.append(None)
    if componentSwitch[1]: # Pressure Correlation
        PressCorre=rho_s.copy(data=np.zeros(np.shape(rho_s.values))).chunk("auto")
        p_prime= np.subtract(         ds["p"],  ds["pref"].expand_dims(
            dim={"x_p":u_i_DA[0].coords["x_p"].values ,"y_p":u_i_DA[0].coords["y_p"].values,"time_0":u_i_DA[0].coords["time_0"].values }))
        for k in range(len(u_i_DA)):
            PressCorre= PressCorre - d_dxi(  s( u_i_DA[k], p_prime )   ,   k   )
        del p_prime
        PressCorre = np.divide( PressCorre , rho_s )
        returnVal.append(_meanCondense(PressCorre,returnMean))
        del PressCorre
    else:
        returnVal.append(None)
    if componentSwitch[2]: #Buoyancy production/dissipation
        th_v= thermodynamics.virtual_potential_temperature_monc(ds["th"], ds["thref"] , ds["q_vapour"], ds["q_cloud_liquid_mass"])
        __checkAllNaN(th_v,"th_v")
        Buoy=s(u_i_DA[2], thermodynamics.buoyancy_monc(th_v, ds["thref"]) )
        del th_v
        returnVal.append(_meanCondense(Buoy,returnMean))
        del Buoy
    else:
        returnVal.append(None)
    if componentSwitch[3]: #calculate shear production
        Stress=rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
        for i in range(len(u_i_DA)):
            derivTerm=rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
            for k in range(len(u_i_DA)):
                derivTerm = derivTerm + np.multiply( s(u_i_DA[i], u_i_DA[k]) ,  d_dxi( resol_DA(u_i_DA[i]), k )   )
            Stress = Stress - derivTerm
            del derivTerm
        if dissipationDef!="Brown": # Add peter's term of additional stress from decomp of dissipation term
            divergenceTerm=rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
            for i in range(len(u_i_DA)):  
                for k in range(len(u_i_DA)):
                    tau_ik=np.multiply( mu(ds), S_ij(ds, i, k)  )
                    divergenceTerm = divergenceTerm + d_dxi( s( u_i_DA[i] , tau_ik  ) , k   )
            Stress = Stress + np.divide( divergenceTerm , rho_s  )
            del divergenceTerm
        returnVal.append(_meanCondense(Stress,returnMean))
        del Stress
    else: 
        returnVal.append(None)
    if componentSwitch[4]: #calculate dissipation
        Dissip=rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
        if dissipationDef=="Brown":
            tau_ij=[]
            for i in range(len(u_i_DA)): #pre-calculate tau_ij
                tau_ij.append(list())
                for j in range(len(u_i_DA)):
                    tau_ij[i].append( np.multiply( mu(ds), S_ij(ds, i, j)  )  )
            for i in range(len(u_i_DA)):
                divergenceTau=rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
                for k in range(len(u_i_DA)):
                    divergenceTau = divergenceTau + d_dxi(tau_ij[i][k], k )
                Dissip = Dissip + s( 
                    np.divide( u_i_DA[i], rho_s )  ,  divergenceTau   )
            del divergenceTau
        else:  #Peter's dissipation. -s( tau_ik / rho_s  , d u_i / d x_k    )
            for i in range(len(u_i_DA)):
                for k in range(len(u_i_DA)):
                    tau_ik= np.multiply( mu(ds), S_ij(ds, i, k)  )
                    Dissip = Dissip - s( 
                    np.divide( tau_ik, rho_s )  ,  d_dxi(u_i_DA[i], k )   )
        returnVal.append(_meanCondense(Dissip,returnMean))
        del Dissip
    else: 
        returnVal.append(None)

    if all(componentSwitch): # calculate budget if all component
        for i in range(1, len(returnVal)):
            returnVal[0] = returnVal[0] + returnVal[i]
    return returnVal


def tkeBudg_eSub(ds,componentSwitch=[True,True,True],returnMean=[False,False,False,False], approach="SMAG",addExtra=False,dropSurface=True):
    """return [Budget, Buoy, Stress, Dissip, [Extra Terms]]
    [Extra Terms]: list of optional terms for different SG models.
                [S_smag] when approach=="SMAG", i.e. Smagorinsky SG strain rate 
    """
    u_i_DA=[ds.u, ds.v, ds.w]
    x_i=["x_p","y_p","z_p"]
    rho_s=ds["rho_ref"].expand_dims(dim={"x_p":u_i_DA[0].coords["x_p"].values ,"y_p":u_i_DA[0].coords["y_p"].values,"time_0":u_i_DA[0].coords["time_0"].values})
    returnVal=[]
    if all(componentSwitch):
        Budget= rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
        returnVal.append(_meanCondense(Budget,returnMean))
    if componentSwitch[0]: #Buoyancy production
        if approach=="SMAG":
            th_v= thermodynamics.virtual_potential_temperature_monc(ds["th"], ds["thref"] , ds["q_vapour"], ds["q_cloud_liquid_mass"])
            Buoy= - np.multiply( nu_h(ds) , d_dxi( thermodynamics.buoyancy_monc( th_v   , ds["thref"]) , 2 ) ) # s_sg(w, B') := - nu_h * d(B')/dz
            del th_v
            returnVal.append(_meanCondense(Buoy,returnMean))
            del Buoy
    else:
        returnVal.append(None)
    if componentSwitch[1]: #Stress production
        if approach=="SMAG":
            Stress= rho_s.copy(data=np.zeros(np.shape(rho_s.values)))
            for i in range(len(u_i_DA)):
                for k in range(len(u_i_DA)):
                    Stress = Stress + np.multiply( mu(ds) , np.multiply( S_ij(ds, i, k) , d_dxi( u_i_DA[i] , k ) ) ) # s_sg( u_i, u_k ) * d(u_i)/d x_k
            Stress = np.divide(Stress, rho_s)
            returnVal.append(_meanCondense(Stress,returnMean))
            del Stress
    else:
        returnVal.append(None)
    if componentSwitch[2]: #Dissip term
        if approach=="SMAG" : # Smagorinsky assumes equilibrium, so Dissip always equal -(Buoy+Stress)
            if all(componentSwitch):
                Dissip= - (returnVal[1]+returnVal[2])
                S_smag= np.sqrt(np.divide( -np.multiply(Dissip, rho_s), mu(ds) ) )
                returnVal.append(_meanCondense(Dissip,returnMean))
                del Dissip
                if addExtra:
                    returnVal.append([_meanCondense(S_smag,returnMean)])
                    del S_smag
            else:
                returnVal.append( - tkeBudg_eSub(ds, componentSwitch=[True,False,False], returnMean=returnMean )[0] - tkeBudg_eSub(ds, componentSwitch=[False,True,False], returnMean=returnMean )[1])
                if addExtra:
                    S_smag= np.sqrt(np.divide( -np.multiply(Dissip, rho_s), mu(ds) ) )
                    returnVal.append([_meanCondense(S_smag,returnMean)])
                    del S_smag  
    else:
        returnVal.append(None)
        if approach=="SMAG" and addExtra:
            returnVal.append([None])
    #update Budget
    if all(componentSwitch): # calculate budget if all component
        if addExtra:
            stop=len(returnVal)-1
        else:
            stop=len(returnVal)
        for i in range(1, stop):
            returnVal[0] = returnVal[0] + returnVal[i]        
    if dropSurface:
        returnVal2=[]
        for i in range(0, len(returnVal)):
            if returnVal[i] is None:
                returnVal2.append(returnVal[i])
            else:
                returnVal2.append(returnVal[i].isel({"z_p":slice(1,None,1)}))
        return returnVal2
    else:
        return returnVal

def _tau_ij(ds, i,j, method="visc"):
    return np.multiply( S_ij(ds,i,j)  , mu(ds) )

def e_SGSScale(ds,returnMean=[False,False,False,False], d=8.49, fmOps="conventional",dropSurface=True):
    """d=8.49 default from Nakanishi & niino
    returns: [fm, Ri, Epsilon_s , e_s]
    """
    returnVal=[]
    returnVal.append( np.divide(nu_m(ds), np.multiply( SMAG_S(ds), np.power( SMAG_L(ds) , 2 ) ) ) ) # fm= nu_m/(lambda^2* S)
    returnVal.append(_Ri_from_fm(returnVal[0], np.divide( nu_m(ds), nu_h(ds))  , method=fmOps)) # Ri=inv (fm)
    l2s3= np.multiply( np.power(SMAG_L(ds),2), np.power(SMAG_S(ds),3)  ) #lambda^2*S^3
    fmterm= np.multiply( returnVal[0], (nu_m(ds).copy(data=np.ones(np.shape(nu_m(ds))))-np.divide( returnVal[1] , np.divide( nu_m(ds), nu_h(ds)) )) ) # fm*(1- Ri/Pr)
    returnVal.append( np.multiply(l2s3 , fmterm) ) # Epsilon_s
    del l2s3
    del fmterm
    returnVal.append( np.power( d*np.multiply(returnVal[2], SMAG_L(ds) ) , 2.0/3.0 ) ) # e_s= (d*lambda*Epsilon)^(2/3)
    for i in range(len(returnVal)):
        if dropSurface:
            returnVal[i] = _meanCondense(returnVal[i],returnMean).isel({"z_p":slice(1,None,1)})
        else:
            returnVal[i] = _meanCondense(returnVal[i],returnMean)
    return returnVal

def tkeBudg_eSGS_trsp(ds):
    u_i_DA=[ds.u, ds.v, ds.w]
    rho_s=ds["rho_ref"].expand_dims(dim={"x_p":u_i_DA[0].coords["x_p"].values ,"y_p":u_i_DA[0].coords["y_p"].values,"time_0":u_i_DA[0].coords["time_0"].values})
    eSGS_trsp_3=rho_s.copy(data=np.zeros(np.shape(rho_s)))
    x_i=["x_p","y_p","z_p"]
    for i in range(len(x_i)):
        eSGS_trsp_3= eSGS_trsp_3 + d_dxi( resol_DA( np.multiply( _tau_ij(ds, i, 2)    ,   u_i_DA[i] - resol_DA(u_i_DA[i])  )  )     ,  2 )
    return np.divide(eSGS_trsp_3, rho_s)


def __add_moncUtils_thermofuncHandles(funcHandle,argLists):
    """add a function handle in monc_utils.thermodynamics 
    return the DA calculated with var name as key"""
    DA= thermodynamics.funcHandle(*argLists)
    if DA.name!=None:
        return {Da.name: DA}
    else:
        return {funcHandle.__name__: DA}


def prepareDataset(fdir,VarList,drop_t0=True,use_elapseTime=False, aliasDict={}):
    """Prepare dataset on p points of names in VarList
    drop_t0: True when time axes includes both time and time_0, then drop time_0 to time
    use_elapseTime: True when use elapse_time as coordinates
    aliasDict: dictionary with keys: original name and values the new alias
    """
    d = xr.open_dataset(fdir,decode_times=True,engine="netcdf4", chunks="auto")
    #print (d.dims)
    #print(d.coords)
    DADict=dict()
    for var in VarList:
        time_coord='time_0'
        if use_elapseTime:
            time_coord='elapsed_time'
        if var.find("STASH_")!=-1: 
            # in case of dim of z not have name on "lev_eta", swap
            cz=[c for c in d[var].dims if 'lev_eta' in c]
            if cz==[]:
                #dnew=d.copy()
                coo=[c for c in d[var].dims if 'lev' in c]
                for k in range(len(coo)):
                    coordOld=coo[k]    #[c for c in d[var].dims if 'lev' in c][k]
                    #print(coordOld)
                    coordNameSegs=coordOld.rpartition("lev")
                    annot=coordNameSegs[-1].rpartition("_eta")
                    coordNew=coordNameSegs[0]+coordNameSegs[1]+annot[1]+annot[2]+annot[0]
                    #coordNew=[c for c in d[var].coords if coordOld.replace("eta",'zsea') in c][0]
                    #print(coordNew)
                    dnew=d.rename({coordOld:coordNew})
                    print("coord name "+coordOld+" changed to " + coordNew+ " in "+ var )
                field=get_um_field(dnew, var.replace("STASH_",''))
                    
            else:
                field= get_um_field(d, var.replace("STASH_",''))
            __checkAllNaN(field,var)
        else:
            if moncUtils_stashmap.get(var)==None and var[-3:] != 'ref':
                print("warning: monc_util stash_map does not include name "+ var)
            field = get_um_data_on_grid(d, var, grid='p')
            __checkAllNaN(field,var)
        #conform z_w if there were
        cz=[c for c in field.dims if 'z_' in c]
        if cz!=[]:
            if cz[0]=="z_w": #convert to z_p
                coordname=[c for c in d.coords if '_zsea' in c]
                if coordname.count("rholev_zsea_rho") !=0:
                    z_p = d["rholev_zsea_rho"].rename({'rholev_zsea_rho':'z_p'})
                    z_p = z_p.swap_dims({'rholev_eta_rho':'z_p'})
                    field=difference_ops.grid_conform(field, field.coords["z_w"], z_p, grid = 'p' )
                else:
                    coordname_current=[c for c in d.coords if ('rho' in c and 'zsea' in c)]
                    if coordname_current!=[]:
                        z_p = field.coords[coordname_current[0]].rename({coordname_current[0]:'z_p'})
                        z_p = z_p.swap_dims({cz[0]:'z_p'})
                        field=difference_ops.grid_conform(field, field.coords["z_w"], z_p, grid = 'p' )
                    else:
                        print("Warning: original dataset contains no z_p. available z coords:", coordname)
                        #print(d.coords[coordname[0]])
                        #print(d.coords[coordname[1]])
                        print("\t When calling: ", field.name)
                        #print("\t Forcing the name of z to be z_p...")
                        #field=field.rename({"z_w":"z_p"})
                    print("\t current dims: ", field.dims)
        #convert time if there were
        ct = [c for c in field.dims if 'tim' in c]
        #print (ct)
        if (ct!= [] and drop_t0 and not use_elapseTime):                
            if ct[0]=="time_0":
                field=field.drop_isel({"time_0":0}).copy() # drop the first slice of time_0
                #print(field,"\n")
            elif ct[0]=="time":
                field=field.rename({"time":"time_0"}).copy()
        dropCoords=[]
        for c in field.coords.keys():
            if (c =='z_p' or c =='x_p' or c =='y_p' or c == time_coord):
                continue
            else:
                dropCoords.append(c)
        field=field.drop_vars(dropCoords)
        #print (field,"\n")
        
        if aliasDict.get(var)!= None: #alias to change
            newname=aliasDict.get(var)
            DADict.update({newname:field})  
        else:
            DADict.update({var:field})  
    if VarList.count("thref")*VarList.count("pref")!=0:
        DADict.update({"rho_ref":rho_ref(DADict["thref"] , DADict["pref"])})
    #print("rho_ref",DADict["rho_ref"].expand_dims(dim={"x_p":DADict["p"].coords["x_p"].values ,"y_p":DADict["p"].coords["y_p"].values,"time_0":DADict["p"].coords["time_0"].values}))
    ds_new=xr.Dataset(
        data_vars=DADict
    )
    return ds_new.chunk("auto")

def prepareDataset2(fdir,VarList,drop_t0=True,use_elapseTime=False, aliasDict={}):
    """Prepare dataset on p points of names in VarList
    Use when the coords of dataset is BL coords.
    drop_t0: True when time axes includes both time and time_0, then drop time_0 to time
    use_elapseTime: True when use elapse_time as coordinates
    aliasDict: dictionary with keys: original name and values the new alias
    """
    d = xr.open_dataset(fdir,decode_times=True,engine="netcdf4", chunks="auto")
    #print (d.dims)
    #print(d.coords)
    DADict=dict()
    #clean dataset coord names
    #remove the _bl_ to tail
    cz_bls=[c for c in d.coords if '_bl_' in c]
    for k in range(len(cz_bls)):
        coordOld=cz_bls[k]    #[c for c in d[var].dims if 'lev' in c][k]
        #print(coordOld)
        coordNameSegs=coordOld.rpartition("_bl")
        coordNew=coordNameSegs[0]+coordNameSegs[2]+coordNameSegs[1]
        d=d.rename({coordOld:coordNew})
        print("coord name "+coordOld+" changed to " + coordNew+ " in dataset" )
    #print (d.dims)    
    for var in VarList:
        time_coord='time_0'
        if use_elapseTime:
            time_coord='elapsed_time'
        if var.find("STASH_")!=-1: 
            field= get_um_field(d, var.replace("STASH_",''))
            __checkAllNaN(field,var)
        else:
            if moncUtils_stashmap.get(var)==None and var[-3:] != 'ref':
                print("warning: monc_util stash_map does not include name "+ var)
            field = get_um_data_on_grid(d, var, grid='p')
            __checkAllNaN(field,var)
        #conform z_w if there were
        cz=[c for c in field.dims if 'z_' in c]
        if cz!=[]:
            if cz[0]=="z_w": #convert to z_p
                coordname=[c for c in d.coords if '_zsea' in c]
                if coordname.count("rholev_zsea_rho") !=0:
                    z_p = d["rholev_zsea_rho"].rename({'rholev_zsea_rho':'z_p'})
                    z_p = z_p.swap_dims({'rholev_eta_rho':'z_p'})
                    field=difference_ops.grid_conform(field, field.coords["z_w"], z_p, grid = 'p' )
                else:
                    dimVarOriginal=[c for c in d[var].dims if ("lev" in c)] # original dimension name of current field
                    print("original dim-coord name of z in " + var+ " :", dimVarOriginal, np.shape(d[dimVarOriginal[0]]))
                    coordname_alternative=[c for c in d.coords if ('rho' in c and 'zsea' in c)]
                    ###### HERE TO INSERT COMPARING THE STRINGS OF NAMES, NUMBER OF POINTS ETC !!!
                    if coordname_alternative!=[]:
                        print("Looking for zsea coords in dataset... found: ",coordname_alternative)
                        ##Search for closest match of coord names for coord name
                        coordnameMatched=difflib.get_close_matches(dimVarOriginal, coordname_alternative, n=1,cutoff=0.0)[0]
                        print ("\t I assume the closest p-grid zsea coord is: ", coordnameMatched,np.shape(d[coordnameMatched]))
                        #
                        cor=[c for c in d[coordnameMatched].dims if ("lev" in c)][0]
                        d_dummy=d.swap_dims({cor:coordnameMatched})
                        z_p=d_dummy[coordnameMatched].rename({coordnameMatched:'z_p'})
                        #if cor==dimVarOriginal[0]: #same dimension, swap directly
                            #d_dummy = d.swap_dims({dimVarOriginal[0]:coordnameMatched})
                            #z_p = d_dummy[coordnameMatched].rename({coordnameMatched:'z_p'})
                        #else:
                            #d_dummy = d_dummy.rename({cor: })
                            #z_p=d_dummy[coordnameMatched].rename({coordnameMatched:'z_p'})
                        del d_dummy
                        field=difference_ops.grid_conform(field, field.coords["z_w"], z_p, grid = 'p' )
                    else:
                        print("Warning: original dataset contains no z_p. available z coords:", coordname)
                        #print(d.coords[coordname[0]])
                        #print(d.coords[coordname[1]])
                        print("\t When calling: ", field.name)
                        #print("\t Forcing the name of z to be z_p...")
                        #field=field.rename({"z_w":"z_p"})
                    print("\t current dims: ", field.dims)
        #convert time if there were
        ct = [c for c in field.dims if 'tim' in c]
        #print (ct)
        if (ct!= [] and drop_t0 and not use_elapseTime):                
            if ct[0]=="time_0":
                field=field.drop_isel({"time_0":0}).copy() # drop the first slice of time_0
                #print(field,"\n")
            elif ct[0]=="time":
                field=field.rename({"time":"time_0"}).copy()
        dropCoords=[]
        for c in field.coords.keys():
            if (c =='z_p' or c =='x_p' or c =='y_p' or c == time_coord):
                continue
            else:
                dropCoords.append(c)
        field=field.drop_vars(dropCoords)
        #print (field,"\n")
        if aliasDict.get(var)!= None: #alias to change
            newname=aliasDict.get(var)
            DADict.update({newname:field})    
        else:
            DADict.update({var:field})
    if VarList.count("thref")*VarList.count("pref")!=0:
        DADict.update({"rho_ref":rho_ref(DADict["thref"] , DADict["pref"])})
    #print("rho_ref",DADict["rho_ref"].expand_dims(dim={"x_p":DADict["p"].coords["x_p"].values ,"y_p":DADict["p"].coords["y_p"].values,"time_0":DADict["p"].coords["time_0"].values}))
    ds_new=xr.Dataset(
        data_vars=DADict
    )
    return ds_new.chunk("auto")
    
def calculateBudgetComponents(ds, func_handle, componentSwitchName, returnMean=[False,False,False,False]):
    """ executive for calculating budget components.
        func_handle: handle as the tkeBudget_termNames keys (but not str format!)
        componentSwitchName: string, as the tkeBudget_termNames values (ONE ONLY in a list AT A TIME called!)
    """
    func_name=func_handle.__name__
    returnValueNames=tkeBudget_termNames[func_name]
    if returnValueNames.count(componentSwitchName)==0:
        raise ValueError(func_name+" function have no budget term "+ componentSwitchName)
        return None
    componentSwitch=list()
    for i in range(1,len(returnValueNames)):
        if componentSwitchName=="Budget": #the budget term, all true
            componentSwitch.append(True)
        elif returnValueNames[i]==componentSwitchName:
            componentSwitch.append(True)
        else:
            componentSwitch.append(False)
    print(func_name+" function switch is: ", componentSwitch ," for " + componentSwitchName)
    if componentSwitchName=="Budget":
        returnIndex=0
    else:
        returnIndex=returnValueNames[1:].index(componentSwitchName)
    try:
        return func_handle(ds, componentSwitch=componentSwitch, returnMean=returnMean)[returnIndex]
    except IndexError:
        print(returnIndex)

