import numpy as np
from typing import Union

from NanoParticleTools.util.constants import (
    h_CGS,
    c_CGS,
    e_CGS,
    m_e_CGS,
    BOHR_MAGNETON_CGS,
)


def gaussian(x: Union[float, np.array], c: Union[float, np.array],
             sigma: Union[float, np.array]) -> Union[float, np.array]:
    """
    Calculates the value of a normalized gaussian

    Note: mimics the functionality of gauss() in Igor Pro
    Args:
        x (float, np.array): x value to evaluate gaussian
        c (float, np.array): centering constant for gaussian
        sigma (float, np.array): standard deviation of gaussian

    Returns:
        (float, np.array): The value of a normalized gaussian
    """
    # TODO: Investigate why calls to this function mostly use x==c
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * np.power(
        (x - c) / sigma, 2.))


def get_absorption_cross_section_from_line_strength(energy_gap: float,
                                                    line_strength: float,
                                                    j_init: float,
                                                    n_refract: float) -> float:
    """
    Helper function to calculate the absorption_cross_section

    port of PP_lineStrength2absCrossSection
    :param energy_gap:
    :param line_strength:
    :param j_init:
    :param n_refract:
    :return:
    """
    refractive_index_correction = (n_refract**2 + 2)**2 / (9 * n_refract)
    return (8 * np.pi**3 * e_CGS**2 * energy_gap *
            refractive_index_correction * line_strength) / (3 * c_CGS * h_CGS *
                                                            (2 * j_init + 1))
    # return 8 * np.pi ** 2 * m_e_CGS * c_CGS * energy_gap / (
    #     3 * (2 * j_init + 1) * h_CGS) *
    #     refractive_index_correction * line_strength


def get_transition_rate_from_line_strength(energy_gap: float,
                                           line_strength: float, j_init: float,
                                           n_refract: float) -> float:
    """
    Helper function to calculate transition rate from line strength

    port of PP_lineStrength2transitionRate
    :param energy_gap:
    :param line_strength:
    :param j_init:
    :param n_refract:
    :return:
    """
    refractive_index_correction = n_refract * (n_refract**2 + 2)**2 / (9)

    return 64 * np.pi**4 * e_CGS**2 * refractive_index_correction * np.abs(
        energy_gap)**3 / (3 * h_CGS * (2 * j_init + 1)) * line_strength


def get_critical_energy_gap(mpr_beta: float, absfwhm: float) -> float:
    """
    Helper function to calculate the critical energy gap

    port of PP_CriticalEnergyGap
    :param mpr_beta:
    :param absfwhm:
    :return:
    """
    return mpr_beta * absfwhm**2 / (4 * np.log(2))


def get_MD_line_strength_from_icc(
        initial_intermediate_coupling_coefficients: np.ndarray,
        final_intermediate_coupling_coefficients: np.ndarray, ji: float,
        jf: float, s: np.ndarray, l: np.ndarray) -> float:
    """

    port of PP_ICcoefs2MDLineStrength
    :return:
    """
    if len(initial_intermediate_coupling_coefficients) != len(
            final_intermediate_coupling_coefficients):
        raise ValueError(
            "Number of rows in initial_intermediate_coupling_coefficient and"
            " final_intermediate_coupling_coefficients do not match")
    elif len(initial_intermediate_coupling_coefficients) != len(s):
        raise ValueError(
            "Number of rows in initial_intermediate_coupling_coefficient and"
            " s do not match")
    elif len(initial_intermediate_coupling_coefficients) != len(l):
        raise ValueError(
            "Number of rows in initial_intermediate_coupling_coefficient and"
            " l do not match")

    m_sum = 0
    for i, (eigenvector_i, eigenvector_f) in enumerate(
            zip(initial_intermediate_coupling_coefficients,
                final_intermediate_coupling_coefficients)):
        if eigenvector_i * eigenvector_f == 0:
            continue

        # Only look at cases with equal s's, because otherwise MD operator = 0
        si = sf = s[i]
        li = lf = l[i]

        m_sum += eigenvector_i * eigenvector_f * magnetic_dipole_operation(
            si, li, ji, sf, lf, jf)
    return m_sum**2


def magnetic_dipole_operation(si: float, li: float, ji: float, sf: float,
                              lf: float, jf: float) -> float:
    """
    returns the  expectation value |<i|M|j>|^2, where M is the magnetic dipole
     operator, M = |L + 2S|, in gaussian cgs units (erg/G)
    i and f represent eigenvectors where i = initial state, f = final state
    with S = spin AM, L = orbital AM, J = total AM,

    See Weber, Phys.Rev.B 1967 p.263

    port of PP_magDipoleOp
    :param si:
    :param li:
    :param ji:
    :param sf:
    :param lf:
    :param jf:
    :return:
    """

    dj = jf - ji

    # return 0 if S or L QN's not the same
    if si != sf:
        return 0
    elif li != lf:
        return 0
    elif ji == 0 and jf == 0:
        return 0

    if dj == -1:
        m_value = (((si + li + 1)**2 - ji**2) * (ji**2 - (li - si)**2) /
                   (4 * ji))**0.5
    elif dj == 0:
        m_value = ((2 * ji + 1) /
                   (4 * ji * (ji + 1)))**0.5 * (si * (si + 1) - li *
                                                (li + 1) + 3 * ji * (ji + 1))
    elif dj == 1:
        m_value = (((si + li + 1)**2 - (ji + 1)**2) *
                   ((ji + 1)**2 - (li - si)**2) / (4 * (ji + 1)))**0.5
    else:
        m_value = 0

    m_value *= BOHR_MAGNETON_CGS
    if m_value is None:
        # check if NaN or infinity
        raise ValueError()

    return m_value


def get_absorption_cross_section_from_MD_line_strength(
        line_strength: float, energy_gap: float, j_init: float,
        n_refract: float) -> float:
    """
    port of PP_MDlineStrength2absCS
    :return:
    """

    oscillator_strength = get_oscillator_strength_from_MD_line_strength(
        line_strength, energy_gap, j_init, n_refract)
    return (np.pi * e_CGS**2 / (m_e_CGS * c_CGS**2)) * oscillator_strength
    pass


def get_oscillator_strength_from_MD_line_strength(line_strength: float,
                                                  energy_gap: float,
                                                  j_init: float,
                                                  n_refract: float) -> float:
    """
    port of PP_MDlineStrength2oscStrength
    :return:
    """
    # TODO
    return ((8 * np.pi**2 * m_e_CGS * c_CGS * energy_gap * n_refract) /
            (3 * h_CGS * e_CGS**2 * (2 * j_init + 1))) * line_strength


def get_rate_from_MD_line_strength(line_strength: float, energy_gap: float,
                                   j_init: float, n_refract: float) -> float:
    """
    port of PP_MDlineStrength2rate
    :return:
    """
    return 64 * np.pi**4 * abs(energy_gap)**3 * n_refract**3 / (
        3 * h_CGS * (2 * j_init + 1)) * line_strength


def gaussian_overlap_integral(energy_gap: float, fwhm: float) -> float:
    """

    port of PP_GaussianOverlapIntegral
    :param energy_gap: The distance between the donor emission and acceptor
        absorption peak centers, assuming gaussian lineshapes
    :param fwhm:
    :return: The overlap integral of two normalized gaussian peaks with fwhm
        fwhm and peak centers separated by Egap
    """
    return fwhm**(-1) * np.sqrt(
        2 * np.log(2) / np.pi) * 2**(-2 * (energy_gap / fwhm)**2)


def phonon_assisted_energy_transfer_constant(donor_line_strength: float,
                                             acceptor_line_strength: float,
                                             overlap_integral: float,
                                             n_refract: float, j_di: float,
                                             j_ai: float, energy_gap: float,
                                             mpr_beta: float) -> float:
    """
    port of PP_PAETconstant
    :return:
    """
    et_constant = energy_transfer_constant(donor_line_strength,
                                           acceptor_line_strength,
                                           overlap_integral, n_refract, j_di,
                                           j_ai)
    return et_constant * np.exp(-mpr_beta * energy_gap)
    pass


def energy_transfer_constant(donor_line_strength: float,
                             acceptor_line_strength: float,
                             overlap_integral: float, n_refract: float,
                             j_di: float, j_ai: float) -> float:
    """
    port of PP_ETconstant
    :return:
    """
    donor_degeneracy = 2 * j_di + 1
    acceptor_degeneracy = 2 * j_ai + 1
    correction_refractive_index = (n_refract**2 + 2)**2 / (9 * n_refract**2)
    return 8 * np.pi**2 * e_CGS**4 * overlap_integral / (
        3 * h_CGS**2 * c_CGS * donor_degeneracy *
        acceptor_degeneracy) * (correction_refractive_index**2 *
                                donor_line_strength * acceptor_line_strength)
