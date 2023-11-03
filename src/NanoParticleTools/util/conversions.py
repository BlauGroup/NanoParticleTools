def wavenumber_to_wavelength(wavenumber: float) -> float:
    """
    Convert a wavenumber in cm^-1 to a wavelength in nm.

    Args:
        wavenumber: A wavenumber in cm^-1

    Returns:
        The wavelength of the provided energy in nm
    """
    return 1 / wavenumber * 1e7


def wavelength_to_wavenumber(wavelength: float) -> float:
    """
    Convert a wavelength in nm to a wavenumber in cm^-1.

    Args:
        wavelength: A wavelength in nm

    Returns:
        The wavenumber of the provided wavelength in cm^-1
    """
    return 1 / wavelength * 1e7
