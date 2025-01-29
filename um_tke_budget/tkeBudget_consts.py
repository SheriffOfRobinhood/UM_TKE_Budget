#defines constants and default properties
import numpy as np
import xarray as xr

tkeBudget_termNames={
    "tkeBudg_eRes": ["Budget" , "Transport", "Buoyancy Production", "Dissipation" ],
    "tkeBudg_eUn":["Budget", "Transport", "Pressure Correlation", "Buoyancy Production", "Stress Production", "Dissipation"],
    "tkeBudg_eSub": ["Budget", "Buoyancy Production", "Stress Production", "Dissipation"],
    "e_SGSScale":["fm", "Ri", "Eps" , "e"]
}

