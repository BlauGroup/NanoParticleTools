from NanoParticleTools.differential_kinetics import get_parser, run_and_save_one

import h5py
import numpy as np


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    templates = []
    for _ in range(args.num_samples):
        # Pick a number of dopants
        n_dopants = np.random.choice(range(1, args.max_dopants + 1))

        # Pick the dopants
        dopants = np.random.choice(args.dopants, n_dopants, replace=False)

        # Get the dopant concentrations, normalizing the total concentration to 0-1
        total_conc = np.random.uniform(0, 1)
        dopant_concs = np.random.uniform(0, 1, n_dopants)
        dopant_concs = total_conc * dopant_concs / np.sum(dopant_concs)

        # sample a wavelength
        wavelength = np.random.uniform(*args.excitation_wavelength)

        # sample a power
        power = np.random.uniform(*args.excitation_power)
        templates.append({
            'dopants': dopants,
            'dopant_concs': dopant_concs,
            'excitation_wavelength': wavelength,
            'excitation_power': power
        })

    with h5py.File(args.output_file, 'a') as hf:
        for i, template in enumerate(templates):
            run_and_save_one(**template,
                             group_id=i // args.max_data_per_group,
                             data_i=i % args.max_data_per_group,
                             file=hf,
                             include_spectra=args.include_spectra)
