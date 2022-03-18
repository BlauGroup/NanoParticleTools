To extract data from the SpeciesDB and populate into json files:
1. Save the waves to the computer
```
for species in ['Dy', 'Er', 'Eu', 'Gd', 'Ho', 'Nd', 'Sm', 'Surface', 'Tb', 'Tm', 'Yb']:
    for wave in ["EnergyLevelLabels", "EnergyLevels", "SLJ", "absFWHM", "lineStrengths", "TransitionLabels", "JO_params"]:
        if species == 'Surface' and wave == 'JO_params':
            # Surface does not have JO parameters
            continue
        print(f'Save/J/O/M="\\n"/DLIM=","/W/P=home :::::Packages:SpectralKinetics:SpeciesDB:{species}:{wave} as "{species}_{wave}.txt"')
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