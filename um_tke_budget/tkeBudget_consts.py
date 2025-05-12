#defines constants and default properties
import numpy as np
import xarray as xr

tkeBudget_termNames={
    "tkeBudg_eRes": ["Budget" , "Transport", "Buoyancy Production", "Dissipation" ],
    "tkeBudg_eUn":["Budget", "Transport", "Pressure Correlation", "Buoyancy Production", "Stress Production", "Dissipation"],
    "tkeBudg_eSub": ["Budget", "Buoyancy Production", "Stress Production", "Dissipation"],
    "e_SGSScale":["fm", "Ri", "Eps" , "e"]
}

#general subfilter module's options default values
filter_options={
    #master switch for either to turn on filtering
    'filter_switch' : False,
    
    #following need specify before filter init
    'filtInit_Opts': {
        'filter_id': 'default_filter',
        'delta_x':1000.0,
        'set_fft':False,
        'cutoff':1e-06, 
        'npoints':None, 
        'high_pass':0, 
        ## for wave_cutoff and circular_wave_cutoff:
        'wavenumber' :- 1, 
        ## for running_mean:
        'width':- 1, 
        ## for gaussian and gen_gaussian:
        'sigma': -1, 
        'ndim': 2, 
        'filter_name': 'running_mean',
        },
    
    #optional options
    'FFT_type': 'RFFT',
    
}