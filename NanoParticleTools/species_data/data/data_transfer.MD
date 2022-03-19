To extract data from the SpeciesDB and populate into json files:
1. Save the waves to your computer. Use the following to produce the commands
```python
for species in ['Dy', 'Er', 'Eu', 'Gd', 'Ho', 'Nd', 'Sm', 'Surface', 'Tb', 'Tm', 'Yb']:
    for wave in ["EnergyLevelLabels", "EnergyLevels", "SLJ", "absFWHM", "lineStrengths", "TransitionLabels", "JO_params"]:
        if species == 'Surface' and wave == 'JO_params':
            # Surface does not have JO parameters
            continue
        print(f'Save/J/O/M="\\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:{species}:{wave} as "{species}_{wave}.txt"')
```

```
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Dy:EnergyLevelLabels as "Dy_EnergyLevelLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Dy:EnergyLevels as "Dy_EnergyLevels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Dy:SLJ as "Dy_SLJ.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Dy:absFWHM as "Dy_absFWHM.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Dy:lineStrengths as "Dy_lineStrengths.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Dy:TransitionLabels as "Dy_TransitionLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Dy:JO_params as "Dy_JO_params.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Er:EnergyLevelLabels as "Er_EnergyLevelLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Er:EnergyLevels as "Er_EnergyLevels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Er:SLJ as "Er_SLJ.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Er:absFWHM as "Er_absFWHM.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Er:lineStrengths as "Er_lineStrengths.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Er:TransitionLabels as "Er_TransitionLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Er:JO_params as "Er_JO_params.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Eu:EnergyLevelLabels as "Eu_EnergyLevelLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Eu:EnergyLevels as "Eu_EnergyLevels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Eu:SLJ as "Eu_SLJ.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Eu:absFWHM as "Eu_absFWHM.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Eu:lineStrengths as "Eu_lineStrengths.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Eu:TransitionLabels as "Eu_TransitionLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Eu:JO_params as "Eu_JO_params.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Gd:EnergyLevelLabels as "Gd_EnergyLevelLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Gd:EnergyLevels as "Gd_EnergyLevels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Gd:SLJ as "Gd_SLJ.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Gd:absFWHM as "Gd_absFWHM.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Gd:lineStrengths as "Gd_lineStrengths.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Gd:TransitionLabels as "Gd_TransitionLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Gd:JO_params as "Gd_JO_params.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Ho:EnergyLevelLabels as "Ho_EnergyLevelLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Ho:EnergyLevels as "Ho_EnergyLevels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Ho:SLJ as "Ho_SLJ.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Ho:absFWHM as "Ho_absFWHM.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Ho:lineStrengths as "Ho_lineStrengths.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Ho:TransitionLabels as "Ho_TransitionLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Ho:JO_params as "Ho_JO_params.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Nd:EnergyLevelLabels as "Nd_EnergyLevelLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Nd:EnergyLevels as "Nd_EnergyLevels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Nd:SLJ as "Nd_SLJ.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Nd:absFWHM as "Nd_absFWHM.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Nd:lineStrengths as "Nd_lineStrengths.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Nd:TransitionLabels as "Nd_TransitionLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Nd:JO_params as "Nd_JO_params.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Sm:EnergyLevelLabels as "Sm_EnergyLevelLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Sm:EnergyLevels as "Sm_EnergyLevels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Sm:SLJ as "Sm_SLJ.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Sm:absFWHM as "Sm_absFWHM.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Sm:lineStrengths as "Sm_lineStrengths.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Sm:TransitionLabels as "Sm_TransitionLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Sm:JO_params as "Sm_JO_params.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Surface:EnergyLevelLabels as "Surface_EnergyLevelLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Surface:EnergyLevels as "Surface_EnergyLevels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Surface:SLJ as "Surface_SLJ.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Surface:absFWHM as "Surface_absFWHM.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Surface:lineStrengths as "Surface_lineStrengths.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Surface:TransitionLabels as "Surface_TransitionLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tb:EnergyLevelLabels as "Tb_EnergyLevelLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tb:EnergyLevels as "Tb_EnergyLevels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tb:SLJ as "Tb_SLJ.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tb:absFWHM as "Tb_absFWHM.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tb:lineStrengths as "Tb_lineStrengths.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tb:TransitionLabels as "Tb_TransitionLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tb:JO_params as "Tb_JO_params.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tm:EnergyLevelLabels as "Tm_EnergyLevelLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tm:EnergyLevels as "Tm_EnergyLevels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tm:SLJ as "Tm_SLJ.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tm:absFWHM as "Tm_absFWHM.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tm:lineStrengths as "Tm_lineStrengths.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tm:TransitionLabels as "Tm_TransitionLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Tm:JO_params as "Tm_JO_params.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Yb:EnergyLevelLabels as "Yb_EnergyLevelLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Yb:EnergyLevels as "Yb_EnergyLevels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Yb:SLJ as "Yb_SLJ.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Yb:absFWHM as "Yb_absFWHM.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Yb:lineStrengths as "Yb_lineStrengths.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Yb:TransitionLabels as "Yb_TransitionLabels.txt"
Save/J/O/M="\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:Yb:JO_params as "Yb_JO_params.txt"
```

2. Run the following python code
```python
import os
from NanoParticleTools.species_data.species import SPECIES_DATA_PATH
import json

def read_file(filename):
    data = []
    with open(os.path.join(base_path, filename), 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0: continue
            l = line.strip()
            if 'SLJ' not in filename:
                try:
                    data.append(l.split(',')[0])
                except:
                    data.append(l.split(',')[0])
            else:
                data.append([float(i) for i in l.split(',')])


    return data

base_path = 'Documents/WaveMetrics/Igor Pro 9 User Files/User Procedures/mf-lanthanidephotophysics/templates/'

d = {}
for species in ['Dy', 'Er', 'Eu', 'Gd', 'Ho', 'Nd', 'Sm', 'Surface', 'Tb', 'Tm', 'Yb']:
    for wave in ["EnergyLevelLabels", "EnergyLevels", "SLJ", "absFWHM", "lineStrengths", "TransitionLabels", "JO_params"]:
        if species == 'Surface' and wave == 'JO_params':
            continue
        d[wave] = read_file(f'{species}_{wave}.txt')
    with open(os.path.join(SPECIES_DATA_PATH, f'{species}.json'), 'w') as f:
        json.dump(d, f)
```