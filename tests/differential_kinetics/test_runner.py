from NanoParticleTools.differential_kinetics.runner import DifferentialKinetics
from NanoParticleTools.differential_kinetics.util import (
    get_diff_kinetics_parser, load_data_from_hdf5)
from h5py import File


def test_runner(tmp_path):
    args = get_diff_kinetics_parser().parse_args([
        '-n', '2', '-m', '2', '-w', '400', '700', '-p', '10', '100000', '-o',
        f'{tmp_path}/out.h5', '-s'
    ])
    dk = DifferentialKinetics(**vars(args))
    assert dk.num_samples == 2

    dk.run()

    data_points = []
    with File(tmp_path / 'out.h5', 'r') as f:
        for i in range(0, len(f['group_0'])):
            data_points.append(load_data_from_hdf5(f, 0, i))
    assert len(data_points) == 2

    dk = DifferentialKinetics(1000)
    assert dk.num_samples == 1000
    assert dk.possible_dopants == ['Yb', 'Er', 'Tm', 'Nd', 'Ho', 'Eu', 'Sm', 'Dy']
    assert dk.excitation_wavelength == [500.0, 1500.0]
    assert dk.excitation_power == [10.0, 100000.0]
