def wavenumber_to_wavelength(wavenumber):
    return (299792458 * 6.62607004e-34) / (wavenumber * 1.60218e-19 /
                                           8065.44) * 1e9


def wavelength_to_wavenumber(wavelength):
    return (8065.44 * 299792458 * 6.62607004e-34) / (wavelength * 1e-9 *
                                                     1.60218e-19)
