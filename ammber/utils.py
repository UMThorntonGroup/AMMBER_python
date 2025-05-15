def add_to_dict(binaryIsothermalSys, dict, add_templates=False, c0={}, Vm=None, convert_to_volumetric_energy=True):
    """
    Adds the parameters of a BinaryIsothermal2ndOrderSystem to a dictionary
    Parameters
    ----------
    binaryIsothermalSys : BinaryIsothermal2ndOrderSystem
        system to draw parameters from
    dict : dict
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
    dict["solution_component"] = binaryIsothermalSys.solution_component
    dict["components"] = [binaryIsothermalSys.component]
    dict["convert_fractional_to_volumetric_energy"] = convert_to_volumetric_energy
    if "phases" not in dict:
        dict["phases"] = {}
    comp = binaryIsothermalSys.component
    for phase_name in binaryIsothermalSys.phases.keys():
        phase = binaryIsothermalSys.phases[phase_name]
        if phase_name not in dict:
            dict["phases"][phase_name] = {}
        
        if comp not in dict["phases"][phase_name]:
            dict["phases"][phase_name][comp] = {}
        dict["phases"][phase_name][comp]["k_well"] = phase.kwell
        dict["phases"][phase_name][comp]["c_min"] = phase.cmin
        dict["phases"][phase_name][comp]["f_min"] = phase.fmin

        if add_templates:
            c0_phase_keys = list(c0.keys())
            if "c0" not in dict["phases"][phase_name][comp]:
                dict["phases"][phase_name][comp]["c0"] = c0[phase_name] if phase_name in c0_phase_keys else -1.0
            for phase_prop in ["mu_int", "D", "sigma"]:
                if phase_prop not in dict["phases"][phase_name]:
                    dict["phases"][phase_name][phase_prop] = -1.0
    if add_templates:
        if "l_int" not in dict:
            dict["l_int"] = -1.0
        if "Vm" not in dict:
            dict["Vm"] = Vm if Vm is not None else -1.0
        if "order_parameters" not in dict:
            dict["order_parameters"] = list(binaryIsothermalSys.phases.keys())
        




def write_binary_isothermal_parabolic_parameters(binaryIsothermalSys, output_file, template_file, phases=None, component="comp_1"):
    """
    Creates a parameters file from a BinaryIsothermal2ndOrderSystem

    Parameters
    ----------
    binaryIsothermalSys : BinaryIsothermal2ndOrderSystem
        system to draw parameters from
    output_file : string
        path to file to create
    template_file : string
        path to template parameter file if needed
    phases : list (string)
        (optional) list of specific phases to copy to parameters
    component : string
        (optional) name of the x-component
    """
    if phases is None:
        phase_list = binaryIsothermalSys.phases.keys()
    else:
        phase_list = phases
    cmin = {}
    fmin = {}
    kwell= {}
    for phase_name in phase_list:
        cmin[phase_name] = {component : binaryIsothermalSys.phases[phase_name].cmin}
        fmin[phase_name] = binaryIsothermalSys.phases[phase_name].fmin
        kwell[phase_name] = {component : binaryIsothermalSys.phases[phase_name].kwell}
    _PRISMS_parabolic_parameters(cmin=cmin, fmin=fmin, kwell=kwell, 
                            phases = phase_list ,comps=[component],
                            template_file=template_file, output_file=output_file)


def _PRISMS_parabolic_parameters(cmin, fmin, kwell, phases, comps, output_file, template_file):
    ## Step 2: Make necessary strings
    num_phases = len(phases)
    num_comps = len(comps)+1
    sf = 3
    fmin_str = ""
    cmin_str = ""
    kwell_str = ""
    phase_names_str = ""
    comp_names_str = ""
    for phase in phases:
        fmin_str += f'{fmin[phase]:.{sf}e}' + ", "
        phase_names_str += phase + ", " 
        for comp in comps:
            cmin_str += f'{cmin[phase][comp]:.{sf}f}' + ","
            kwell_str += f'{kwell[phase][comp]:.{sf}e}' + ","
        cmin_str +=  " "
        kwell_str += " "
    
    for comp in comps:
        comp_names_str += comp + ", "
    
    ## Step 3: read template and write prm
    import re
    with open(template_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                if re.match(r'^set Model constant num_phases\b', line):
                    f_out.write("set Model constant num_phases = " + str(num_phases) + ", INT\n")
                elif re.match(r'^set Model constant num_comps\b', line):
                    f_out.write("set Model constant num_comps = " + str(num_comps) + ", INT\n")
                elif re.match(r'^set Model constant fWell\b', line):
                    f_out.write("set Model constant fWell = " + fmin_str + "DOUBLE ARRAY\n")
                elif re.match(r'^set Model constant kWell\b', line):
                    f_out.write("set Model constant kWell = " + kwell_str + "DOUBLE ARRAY\n")
                elif re.match(r'^set Model constant cmin\b', line):
                    f_out.write("set Model constant cmin = " + cmin_str + "DOUBLE ARRAY\n")
                elif re.match(r'^set Model constant phase_names\b', line):
                    f_out.write("set Model constant phase_names = " + phase_names_str + "STRING ARRAY\n")
                elif re.match(r'^set Model constant comp_names\b', line):
                    f_out.write("set Model constant comp_names = " + comp_names_str + "STRING ARRAY\n")
                else:
                    f_out.write(line)
    
    print("Model Parameters written to " + output_file + "\n")

