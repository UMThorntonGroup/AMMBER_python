def add_to_dict(binaryIsothermalSys, system_dict, add_templates=False, c0={}, Vm=None, convert_to_volumetric_energy=True):
    """
    Adds the parameters of a BinaryIsothermal2ndOrderSystem to a dictionary
    Parameters
    ----------
    binaryIsothermalSys : BinaryIsothermal2ndOrderSystem
        system to draw parameters from
    system_dict : dict
        dictionary to add parameters to
    add_templates : bool
        add templates or additional values needed for kinetics simulation
    c0 : dict
        phase-name keys, initial composition values. only used if add_templates==True
    Vm : float
        nominal number-volume (eg. molar volume) of system
    convert_to_volumetric_energy : bool
        a setting for AMMBER kinetics. true if binaryIsothermalSys composition axes are number fraction
    """
    system_dict["solution_component"] = binaryIsothermalSys.solution_component
    system_dict["components"] = [binaryIsothermalSys.component]
    system_dict["convert_fractional_to_volumetric_energy"] = convert_to_volumetric_energy
    if "phases" not in system_dict:
        system_dict["phases"] = {}
    comp = binaryIsothermalSys.component
    for phase_name in binaryIsothermalSys.phases.keys():
        phase = binaryIsothermalSys.phases[phase_name]
        if phase_name not in system_dict["phases"]:
            system_dict["phases"][phase_name] = {}
        
        if comp not in system_dict["phases"][phase_name]:
            system_dict["phases"][phase_name][comp] = {}
        system_dict["phases"][phase_name][comp]["k_well"] = phase.kwell
        system_dict["phases"][phase_name][comp]["c_min"] = phase.cmin
        system_dict["phases"][phase_name]["f_min"] = phase.fmin

        if add_templates:
            c0_phase_keys = list(c0.keys())
            if "x0" not in system_dict["phases"][phase_name][comp]:
                system_dict["phases"][phase_name][comp]["x0"] = c0[phase_name] if phase_name in c0_phase_keys else -1.0
            for phase_prop in ["mu_int", "D", "sigma"]:
                if phase_prop not in system_dict["phases"][phase_name]:
                    system_dict["phases"][phase_name][phase_prop] = -1.0
    if add_templates:
        if "l_int" not in system_dict:
            system_dict["l_int"] = -1.0
        if "Vm" not in system_dict:
            system_dict["Vm"] = Vm if Vm is not None else -1.0
        if "order_parameters" not in system_dict:
            system_dict["order_parameters"] = list(binaryIsothermalSys.phases.keys())
        if "dimensions" not in system_dict:
            system_dict["dimensions"] = { "length_scale": 0.0, "time_scale": 0.0, "energy_density_scale": 0.0 }
