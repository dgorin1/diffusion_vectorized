def generate_bounds(ndom:int, moles_bound, mineral_name:str,stat = "chisq"):


    if mineral_name == "quartz" or mineral_name == "Quartz":
        if stat.lower() == "chisq" or stat.lower() == "l2_moles" or stat.lower() == "l1_moles":
            moles = True
        else:
            moles = False

        Ea_bounds = (50,150)
        lnd0aa_bounds = (-10,30)
        frac_bounds = (0,1)

        if ndom == 1:
            if moles == True:
                return [moles_bound,Ea_bounds,lnd0aa_bounds]
            else:
                return [Ea_bounds,lnd0aa_bounds]
        elif ndom >1:
            if moles == True:
                return [moles_bound,Ea_bounds]+ ndom*[lnd0aa_bounds]+ (ndom-1)*[frac_bounds]
            else:
                return [Ea_bounds]+ ndom*[lnd0aa_bounds]+ (ndom-1)*[frac_bounds]


