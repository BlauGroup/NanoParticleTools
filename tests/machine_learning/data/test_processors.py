from NanoParticleTools.machine_learning.data.processors import (
    FeatureProcessor, LabelProcessor, EnergySpectrumLabelProcessor,
    WavelengthSpectrumLabelProcessor, TotalEnergyLabelProcessor,
    SummedWavelengthRangeLabelProcessor)


def test_energy_spectrum_label_processor():
    lp = EnergySpectrumLabelProcessor(log_constant=1)
    lp.required_fields == ['output.energy_spectrum_x', 'output.energy_spectrum_y']
    example_doc = lp.example()

    assert example_doc['spectra_x'].shape == (1, 600)
    assert example_doc['y'].shape == (1, 600)
    assert example_doc['log_y'].shape == (1, 600)
    assert example_doc['log_const'] == 1

    lp = EnergySpectrumLabelProcessor(output_size=400, log_constant=1)
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
    example_doc = lp.example()

    assert example_doc['spectra_x'].shape == (1, 400)
    assert example_doc['y'].shape == (1, 400)
    assert example_doc['log_y'].shape == (1, 400)
    assert example_doc['log_const'] == 1


def test_summed_range_label_processor():
    lp = SummedWavelengthRangeLabelProcessor(log_constant=1)
    example_doc = lp.example()

    assert example_doc['labels'] == ['uv', 'uva', 'uvb', 'uvc', 'vis', 'blue', 'green', 'red']
    assert example_doc['y'].shape == (1, 8)
    assert example_doc['log_y'].shape == (1, 8)
    assert len(example_doc['intensities']) == 8
    assert example_doc['log_const'] == 1
