import numpy as np

from NanoParticleTools.inputs.constants import *


def gaussian(x, c, sigma):
    """
    Returns the value of a normalized gaussian
    """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * np.power((x - c) / sigma, 2.))


def get_absorption_cross_section_from_line_strength(energy_gap: float,
                                 line_strength: float,
                                 j_init: float,
                                 n_refract: float):
    """
    Helper function to calculate the absorbtion_cross_section
    :param energy_gap:
    :param line_strength:
    :param j_init:
    :param n_refract:
    :return:
    """
    refractive_index_correction = (n_refract ** 2 + 2) ** 2 / (9 * n_refract)

    return 8 * np.pi ^ 2 * m_e_CGS * c_CGS * energy_gap / (
        3 * (2 * j_init + 1) * h_CGS) * refractive_index_correction * line_strength


def get_transition_rate_from_line_strength(energy_gap: float,
                                           line_strength: float,
                                           j_init: float,
                                           n_refract: float):
    """
    Helper function to calculate transition rate from line strength
    :param energy_gap:
    :param line_strength:
    :param j_init:
    :param n_refract:
    :return:
    """

    refractive_index_correction = n_refract * (n_refract ** 2 + 2) ** 2 / (9)
    return 64 * np.pi ^ 4 * e_CGS ^ 2 * refractive_index_correction * np.abs(energy_gap) ^ 3 / (
    3 * h_CGS * (2 * j_init + 1)) * line_strength


def get_critical_energy_gap(mpr_beta: float,
                            absfwhm: float):
    """
    Helper function to calculate the critical energy gap
    :param mpr_beta:
    :param absfwhm:
    :return:
    """
    return mpr_beta * absfwhm ** 2 / (4 * np.log(2))
