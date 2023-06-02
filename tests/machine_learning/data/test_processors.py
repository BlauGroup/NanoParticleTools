from NanoParticleTools.machine_learning.data.processors import (
    DataProcessor, FeatureProcessor, LabelProcessor,
    EnergySpectrumLabelProcessor, WavelengthSpectrumLabelProcessor,
    TotalEnergyLabelProcessor, SummedWavelengthRangeLabelProcessor)
from NanoParticleTools.inputs import SphericalConstraint
import pytest
from torch_geometric.data import Data


def test_data_processor():
    DataProcessor.__abstractmethods__ = set()
    dp = DataProcessor(fields=['a', 'b'])

    assert dp.required_fields == ['a', 'b']
    with pytest.raises(NotImplementedError):
        dp.process_doc({})


def test_feature_processor():
    FeatureProcessor.__abstractmethods__ = set()
    dp = FeatureProcessor(possible_elements=['Yb', 'Er'], fields=['a', 'b'])

    assert dp.possible_elements == ['Yb', 'Er']
    assert dp.n_possible_elements == 2
    assert dp.dopants_dict == {'Yb': 0, 'Er': 1}
    assert dp.required_fields == ['a', 'b']
    with pytest.raises(NotImplementedError):
        dp.process_doc({})

    # Make a dummy nested dict
    doc = {'a': {'b': {'c': 'test'}}}
    assert dp.get_item_from_doc(doc, 'a.b.c') == 'test'
    assert dp.get_item_from_doc(doc, 'a.b') == {'c': 'test'}

    constraints = [SphericalConstraint(5), SphericalConstraint(10)]
    assert dp.get_radii(0, constraints) == (0, 5)
    assert dp.get_radii(1, constraints) == (5, 10)
    assert dp.get_volume(10) == pytest.approx(4188.790204786391)

    assert dp.data_cls == Data

    with pytest.raises(NotImplementedError):
        dp.example()

    with pytest.raises(NotImplementedError):
        dp.is_graph()


def test_energy_spectrum_label_processor():
    lp = EnergySpectrumLabelProcessor(log_constant=1)
    lp.required_fields == [
        'output.energy_spectrum_x', 'output.energy_spectrum_y'
    ]
    example_doc = lp.example()

    assert example_doc['spectra_x'].shape == (1, 600)
    assert example_doc['y'].shape == (1, 600)
    assert example_doc['log_y'].shape == (1, 600)
    assert example_doc['log_const'] == 1

    lp = EnergySpectrumLabelProcessor(output_size=400,
                                      log_constant=1,
                                      gaussian_filter=1)
    with pytest.warns(UserWarning):
        example_doc = lp.example()

    assert example_doc['spectra_x'].shape == (1, 400)
    assert example_doc['y'].shape == (1, 400)
    assert example_doc['log_y'].shape == (1, 400)
    assert example_doc['log_const'] == 1


def test_total_energy_label_processor():
    lp = TotalEnergyLabelProcessor(log_constant=1e-2)
    example_doc = lp.example()

    assert example_doc['y'].shape == (1, 1)
    assert example_doc['log_y'].shape == (1, 1)
    assert example_doc['log_const'] == 1e-2


def test_wavelength_spectrum_label_processor():
    lp = WavelengthSpectrumLabelProcessor(log_constant=1)
    example_doc = lp.example()

    assert example_doc['spectra_x'].shape == (1, 600)
    assert example_doc['y'].shape == (1, 600)
    assert example_doc['log_y'].shape == (1, 600)
    assert example_doc['log_const'] == 1

    lp = WavelengthSpectrumLabelProcessor(output_size=400, log_constant=1)
    with pytest.warns(UserWarning):
        example_doc = lp.example()

    assert example_doc['spectra_x'].shape == (1, 400)
    assert example_doc['y'].shape == (1, 400)
    assert example_doc['log_y'].shape == (1, 400)
    assert example_doc['log_const'] == 1


def test_summed_range_label_processor():
    lp = SummedWavelengthRangeLabelProcessor(log_constant=1)
    assert str(
        lp) == "Summed Ranges Label Processor - 8 bins, log_constant = 1"
    assert repr(
        lp) == "Summed Ranges Label Processor - 8 bins, log_constant = 1"

    example_doc = lp.example()

    assert example_doc['labels'] == [
        'uv', 'uva', 'uvb', 'uvc', 'vis', 'blue', 'green', 'red'
    ]
    assert example_doc['y'].shape == (1, 8)
    assert example_doc['log_y'].shape == (1, 8)
    assert len(example_doc['intensities']) == 8
    assert example_doc['log_const'] == 1
