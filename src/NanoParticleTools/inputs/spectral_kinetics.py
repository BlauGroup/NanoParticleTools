import math
from typing import (Optional, List, Union, Sequence, Tuple)

import numpy as np
from NanoParticleTools.util.constants import (h_CGS, c_CGS)

from NanoParticleTools.inputs.photo_physics import (
    gaussian, get_absorption_cross_section_from_line_strength,
    get_transition_rate_from_line_strength, get_critical_energy_gap,
    get_MD_line_strength_from_icc,
    get_absorption_cross_section_from_MD_line_strength,
    get_rate_from_MD_line_strength, gaussian_overlap_integral,
    phonon_assisted_energy_transfer_constant, energy_transfer_constant)
from scipy.integrate import odeint
from functools import lru_cache
from monty.json import MSONable


class SpectralKinetics(MSONable):
    """
    REFERENCING THIS WORK & BIBLIOGRAPHY:
         There are 2 different citations that can be used to reference
         this work. These articles describe the underlying equations
         behind this work.

         (1)	Chan, E. M.; Gargas, D. J.; Schuck, P. J.; Milliron, D. J.
                Concentrating and Recycling Energy in Lanthanide Codopants
                for Efficient and Spectrally Pure Emission: the Case of
                NaYF4:Er3+/Tm3+ Upconverting Nanocrystals.
                J. Phys. Chem. B 2012, 116, 10561–10570.

         (2)	Chan, E. M. Combinatorial Approaches for Developing
                Upconverting Nanomaterials: High-Throughput Screening,
                Modeling, and Applications.
                Chem. Soc. Rev. 2015, 44, 1653–1679.
    """

    def __init__(self,
                 dopants,
                 phonon_energy: float = 450,
                 zero_phonon_rate: float = 1e7,
                 mpr_alpha: float = 3.5e-3,
                 n_refract: float = 1.5,
                 volume_per_dopant_site: float = 7.23946667e-2,
                 min_dopant_distance: float = 3.867267554e-8,
                 time_step: Optional[float] = 1e-4,
                 num_steps: Optional[float] = 100,
                 ode_max_error: Optional[float] = 1e-12,
                 energy_transfer_rate_threshold: Optional[float] = 0.1,
                 radiative_rate_threshold: Optional[float] = 0.0001,
                 stokes_shift: Optional[float] = 150,
                 excitation_wavelength: Optional[float] = 976,
                 excitation_power: Optional[float] = 1e7,
                 **kwargs):
        """
        :param dopants:
            example: [Dopant('Yb', 0.1, 2), Dopant('Er', 0.02, 34)]
        :param phonon_energy: in wavenumbers
        :param zero_phonon_rate: zero phonon relaxation rate at T=0, in 1/s
        :param mpr_alpha: in cm
        :param n_refract: index of refraction
        :param vol_per_dopant_site: cm^3 for NaYF4, 1.5 is # possible sites
            for dopant ion
        :param min_dopant_distance: minimum distance between dopants (cm)
        :param time_step: time step in seconds
        :param num_steps: number of time steps per simulation
        :param max_error: default is 1e-6
        :param energy_transfer_rate_threshold: lower limit for actually
            accounting for ET rate (s^-1)
        :param radiative_rate_threshold: lower limit for actually accounting
            for radiative rate (s^-1)
        :param stokes_shift: wavenumbers
        :param initial_populations:
        :param excitation_wavelength: the wavelength of the incident
            radiation in in nm
        :param excitation_power: the incident power density in W/cm^2
            (1 W/cm^2 = 1e7 erg/s/cm^2)
        :param kwargs:

        #TODO: energy_transfer_mode
        """

        self.zero_phonon_rate = zero_phonon_rate
        self.phonon_energy = phonon_energy

        self.mpr_alpha = mpr_alpha
        self.n_refract = n_refract
        self.volume_per_dopant_site = volume_per_dopant_site
        self.min_dopant_distance = min_dopant_distance

        self.time_step = time_step
        self.num_steps = num_steps
        self.ode_max_error = ode_max_error
        self.energy_transfer_rate_threshold = energy_transfer_rate_threshold
        self.radiative_rate_threshold = radiative_rate_threshold
        self.stokes_shift = stokes_shift

        self.excitation_power = excitation_power
        self.excitation_wavelength = excitation_wavelength

        self.dopants = dopants

    @property
    def mpr_gamma(self) -> float:
        """
        Gamma constant for multi-phonon relaxation in cm.
        Returns:
            float: Gamma constant
        """
        return np.log(2) / self.phonon_energy

    @property
    def mpr_beta(self) -> float:
        """_summary_
        Beta constant for multi-phonon relaxation in cm.
        Returns:
            float: Beta constant
        """
        return self.mpr_alpha - self.mpr_gamma

    @property
    def incident_wavenumber(self) -> float:
        """_summary_
        Calculates the incident wavenumber in cm^-1.
        Returns:
            float: The incident wavenumber
        """
        return 1e7 / self.excitation_wavelength  # in cm^-1

    @property
    def incident_photon_flux(self) -> float:
        """
        Calculates the incident photon flux in photons/s/cm^2.

        Returns:
            float: The incident photon flux.
        """
        return (self.excitation_power * 1e7 /
                (h_CGS * c_CGS * self.incident_wavenumber))

    @property
    def total_n_levels(self) -> int:
        """
        Total number of energy levels in the system.

        Returns:
            int: Total number of energy levels across all dopants.
        """
        return sum([dopant.n_levels for dopant in self.dopants])

    @property
    def species_concentrations(self) -> List[float]:
        """
        Gets the molar concentration of each dopant species in the system.

        Returns:
            List[float]: List of molar concentrations for each dopant species.
        """
        return [
            dopant.molar_concentration / self.volume_per_dopant_site
            for dopant in self.dopants
        ]

    def calculate_multi_phonon_rates(
            self, dopant) -> Tuple[List[float], List[float]]:
        """
        Calculates Multi-Phonon Relaxation (MPR) Rate for a given set of
        energy levels using Miyakawa-Dexter MPR theory

        :param w_0phonon: zero gap rate (s^-1)
        :param alpha: pre-exponential constant in Miyakawa-Dexter MPR theory.
            Changes with matrix (cm)
        :param stokes_shift:
        :param phonon_energy:
        :return:
        """

        # multiphonon relaxation rate from level i to level i-1
        mpr_rates = [0]  # level 0 cannot relax, thus it's rate is 0
        for i in range(1, dopant.n_levels):
            energy_gap = max(
                abs(dopant.energy_levels[i].energy -
                    dopant.energy_levels[i - 1].energy) - self.stokes_shift,
                0) - 2 * self.phonon_energy

            rate = self.zero_phonon_rate * np.exp(-self.mpr_alpha * energy_gap)
            mpr_rates.append(rate)

        mpa_rates = []
        for i in range(1, dopant.n_levels):
            energy_gap = dopant.energy_levels[i].energy - dopant.energy_levels[
                i - 1].energy
            if energy_gap < 3 * self.phonon_energy:
                rate = mpr_rates[i] * np.exp(-self.mpr_alpha * energy_gap)
                mpa_rates.append(rate)
            else:
                mpa_rates.append(0)
        mpa_rates.append(
            0
        )  # Highest energy level cannot be further excited, therefore set its rate to 0

        return mpr_rates, mpa_rates

    @property
    @lru_cache
    def non_radiative_rate_matrix(self) -> np.ndarray:
        """
        Makes the n x n M_NRrate matrix. M_NRrate[i][j] gives the rate of non-radiative decay
        from level i->j, which are combined energy level indices

        Returns:
            np.ndarray: _description_
        """

        mpr_rates = []
        mpa_rates = []
        for dopant in self.dopants:
            _mpr_rates, _mpa_rates = self.calculate_multi_phonon_rates(dopant)
            mpr_rates.append(_mpr_rates)
            mpa_rates.append(_mpa_rates)

        # TODO: Confirm if this is the correct matrix

        non_radiative_rates = np.zeros(
            (self.total_n_levels + 2, self.total_n_levels + 2))
        first_index = 0
        for i, dopant in enumerate(self.dopants):
            rates = np.identity(len(mpr_rates[i])) * mpr_rates[i]
            non_radiative_rates[first_index + 1:first_index + 1 +
                                dopant.n_levels, first_index:first_index +
                                dopant.n_levels] += rates

            rates = np.identity(len(mpa_rates[i])) * mpa_rates[i]
            non_radiative_rates[first_index + 1:first_index + 1 +
                                dopant.n_levels,
                                first_index + 1 + 1:first_index + 1 + 1 +
                                dopant.n_levels] += rates
            first_index += dopant.n_levels

        return non_radiative_rates[1:self.total_n_levels + 1,
                                   1:self.total_n_levels + 1]

    @property
    @lru_cache
    def line_strength_matrix(self) -> np.ndarray:
        """
        Makes the n x n lineStrengthMatrix from a wave of lineStrengths labeled with the
        transitions "i->j" in transitionLabels wave
        Returns:
            np.ndarray: _description_
        """
        # TODO: make the combined rate matrix if necessary

        combined_line_strength_matrix = np.zeros(
            (self.total_n_levels, self.total_n_levels))
        first_index = 0
        for dopant in self.dopants:
            _m = dopant.get_line_strength_matrix()
            combined_line_strength_matrix[first_index:first_index +
                                          dopant.n_levels,
                                          first_index:first_index +
                                          dopant.n_levels] = _m
            first_index += dopant.n_levels

        return combined_line_strength_matrix

    @property
    @lru_cache
    def radiative_rate_matrix(self) -> np.ndarray:
        """

        Returns:
            np.ndarray: _description_
        """
        rad_rates = np.zeros((self.total_n_levels, self.total_n_levels))
        energy_gaps = np.zeros((self.total_n_levels, self.total_n_levels))
        for dopant_index, dopant in enumerate(self.dopants):
            for i in range(dopant.n_levels):
                combined_i = sum([
                    dopant.n_levels for dopant in self.dopants[:dopant_index]
                ]) + i
                for j in range(dopant.n_levels):
                    combined_j = sum([
                        dopant.n_levels
                        for dopant in self.dopants[:dopant_index]
                    ]) + j
                    energy_gap = dopant.energy_levels[
                        j].energy - dopant.energy_levels[i].energy
                    energy_gaps[combined_i, combined_j] = energy_gap

                    if energy_gap < 0:
                        rad_rates[combined_i][
                            combined_j] = get_transition_rate_from_line_strength(
                                energy_gap,
                                self.line_strength_matrix[combined_i]
                                [combined_j], dopant.slj[i][2], self.n_refract)
                    elif energy_gap > 0:
                        # Going up in energy, therefore calculate the photon absorption based on
                        # abs. cross section and incident flux
                        absfwhm = dopant.absFWHM[j]
                        abs_sigma = absfwhm / (
                            2 * np.sqrt(2 * np.log(2))
                        )  # convert fwhm to width in gaussian equation

                        absorption_cross_section = get_absorption_cross_section_from_line_strength(
                            energy_gap,
                            self.line_strength_matrix[combined_i][combined_j],
                            dopant.slj[i][2], self.n_refract)

                        critical_energy_gap = get_critical_energy_gap(
                            self.mpr_alpha, absfwhm)
                        if abs(energy_gap -
                               self.incident_wavenumber) > critical_energy_gap:
                            individual_absorption_cross_section = (
                                absorption_cross_section
                                * gaussian(energy_gap, energy_gap, abs_sigma))
                            mpr_assisted_absorption_correction_factor = np.exp(
                                -self.mpr_alpha *
                                np.abs(energy_gap - self.incident_wavenumber))
                        else:
                            individual_absorption_cross_section = (
                                absorption_cross_section
                                * gaussian(energy_gap, self.incident_wavenumber, abs_sigma))
                            mpr_assisted_absorption_correction_factor = 1

                        radiative_rate = (self.incident_photon_flux
                                          * individual_absorption_cross_section
                                          * mpr_assisted_absorption_correction_factor)

                        # exponential term accounts for differences between incident
                        # energy and energy level gap
                        if radiative_rate > self.radiative_rate_threshold:
                            rad_rates[combined_i][combined_j] = radiative_rate

        return rad_rates

    @property
    @lru_cache
    def magnetic_dipole_rate_matrix(self) -> np.ndarray:
        """
        Creates the MDradRate matrix containing the MD line strength in cm^2
        from intermediate coupling coefficient vectors

        Returns:
            np.ndarray: _description_
        """
        magnetic_dipole_radiative_rates = np.zeros(
            (self.total_n_levels, self.total_n_levels))

        for dopant_index, dopant in enumerate(self.dopants):
            if dopant.symbol in dopant.SURFACE_DOPANT_SYMBOLS_TO_NAMES:
                # Don't calculate this for surface
                continue
            num_eigenvectors = len(dopant.eigenvector_sl)

            s_vector = dopant.eigenvector_sl[:, 0]
            l_vector = dopant.eigenvector_sl[:, 1]
            for i in range(dopant.n_levels):
                energy_i = dopant.energy_levels[i].energy
                combined_i = sum([
                    dopant.n_levels for dopant in self.dopants[:dopant_index]
                ]) + i

                coeff_vector_i = dopant.intermediate_coupling_coefficients[
                    i]  # May need to transpose
                Ji = dopant.slj[i][2]

                for j in range(dopant.n_levels):
                    energy_j = dopant.energy_levels[j].energy
                    energy_gap = energy_j - energy_i
                    combined_j = sum([
                        dopant.n_levels
                        for dopant in self.dopants[:dopant_index]
                    ]) + j

                    Jf = dopant.slj[j][2]
                    coeff_vector_j = dopant.intermediate_coupling_coefficients[
                        j]  # May need to transpose

                    # TODO: port this function
                    current_line_strength = get_MD_line_strength_from_icc(
                        coeff_vector_i, coeff_vector_j, Ji, Jf, s_vector,
                        l_vector)

                    if current_line_strength is None:
                        raise ValueError(
                            'Error, current_line_strength is not a valid number'
                        )

                    if energy_gap < 0:
                        # spontaneous emission
                        magnetic_dipole_radiative_rates[combined_i][
                            combined_j] = get_rate_from_MD_line_strength(
                                current_line_strength, energy_gap, Ji,
                                self.n_refract)
                    elif energy_gap > 0:
                        # Going up in energy, calculate photon absorption based on abs.
                        # cross section and incident flux
                        absfwhm = dopant.absFWHM[j]
                        abs_sigma = absfwhm / (
                            2 * np.sqrt(2 * np.log(2))
                        )  # convert fwhm to width in gaussian equation

                        absorption_cross_section = (
                            get_absorption_cross_section_from_MD_line_strength(
                                current_line_strength, energy_gap, Ji, self.n_refract))
                        critical_energy_gap = get_critical_energy_gap(
                            self.mpr_alpha, absfwhm)
                        if np.abs(energy_gap - self.incident_wavenumber
                                  ) > critical_energy_gap:
                            individual_absorption_cross_section = (
                                absorption_cross_section
                                * gaussian(energy_gap, energy_gap, abs_sigma))
                            mpr_assisted_absorption_correction_factor = np.exp(
                                -self.mpr_alpha *
                                np.abs(energy_gap - self.incident_wavenumber))
                        else:
                            # energy mismatch < critical energy gap, therefore don't use
                            # any MPR assistance
                            individual_absorption_cross_section = (
                                absorption_cross_section
                                * gaussian(energy_gap, self.incident_wavenumber, abs_sigma))
                            mpr_assisted_absorption_correction_factor = 1

                        # TODO: Resolve comment: "FIXME -- this should probably be enabled since
                        # there is no MD absorption without it, but need to double check if
                        # it is correct"
                        # absorption_rate = individual_absorption_cross_section
                        #   * self.incident_photon_flux * mpr_assisted_absorption_correction_factor
                        absorption_rate = 0
                        if absorption_rate > self.radiative_rate_threshold:
                            magnetic_dipole_radiative_rates[combined_i][
                                combined_j] = absorption_rate
        return magnetic_dipole_radiative_rates

    @property
    @lru_cache
    def energy_transfer_rate_matrix(self) -> List[List[float]]:
        """
        makes the phonon assisted (not migration assisted) energy transfer rate
        constant waves (W_ETrates, W_ETIndices)

        Returns:
            List[List[float]]: the phonon-assisted, non-migration-assisted energy
                transfer rate constant K (1/s)
        """
        energy_transfers = []

        species_map = {}
        cumulative_index = 0
        for dopant_index, dopant in enumerate(self.dopants):
            for i in range(dopant.n_levels):
                species_map[cumulative_index] = dopant_index
                cumulative_index += 1

        for combined_di in range(self.total_n_levels):
            dopant_di_index = species_map[combined_di]
            dopant_di = self.dopants[dopant_di_index]
            di = combined_di - sum(
                [dopant.n_levels for dopant in self.dopants[:dopant_di_index]])

            Jdi = dopant_di.slj[di, 2]
            absfwhm = dopant_di.absFWHM[di]
            critical_energy_gap = get_critical_energy_gap(
                self.mpr_beta, absfwhm)

            # convert from nm^-3 to cm^-3
            donor_concentration = dopant_di.volume_concentration * 1e21
            combined_donor_ground_state_index = sum(
                [dopant.n_levels for dopant in self.dopants[:dopant_di_index]])

            for combined_dj in range(self.total_n_levels):
                dopant_dj_index = species_map[combined_dj]
                dopant_dj = self.dopants[dopant_dj_index]
                dj = combined_dj - sum([
                    dopant.n_levels
                    for dopant in self.dopants[:dopant_dj_index]
                ])

                donor_energy_change = dopant_dj.energy_levels[
                    dj].energy - dopant_di.energy_levels[di].energy
                if donor_energy_change >= 0:
                    # donor transition dE should be negative
                    continue

                # adjust for stokes shift
                # donor_energy_change += self.stokesShift

                donor_line_strength = self.line_strength_matrix[combined_di,
                                                                combined_dj]
                donor_to_ground_state_line_strength = self.line_strength_matrix[
                    combined_di][combined_donor_ground_state_index]

                for combined_ai in range(self.total_n_levels):
                    dopant_ai_index = species_map[combined_ai]
                    dopant_ai = self.dopants[dopant_ai_index]
                    ai = combined_ai - sum([
                        dopant.n_levels
                        for dopant in self.dopants[:dopant_ai_index]
                    ])
                    Jai = dopant_ai.slj[ai, 2]

                    # convert from nm^-3 to cm^-3
                    acceptor_concentration = dopant_ai.volume_concentration * 1e21

                    for combined_aj in range(self.total_n_levels):
                        dopant_aj_index = species_map[combined_aj]
                        dopant_aj = self.dopants[dopant_aj_index]
                        aj = combined_aj - sum([
                            dopant.n_levels
                            for dopant in self.dopants[:dopant_aj_index]
                        ])
                        Jaj = dopant_aj.slj[aj, 2]
                        acceptor_energy_change = dopant_aj.energy_levels[
                            aj].energy - dopant_ai.energy_levels[ai].energy

                        if acceptor_energy_change <= 0:
                            # acceptor transition dE should be positive
                            continue

                        acceptor_line_strength = self.line_strength_matrix[
                            combined_ai, combined_aj]

                        energy_gap = acceptor_energy_change + donor_energy_change

                        if energy_gap > 2 * critical_energy_gap:
                            # So that energy transfer cannot be too uphill
                            continue

                        if np.abs(energy_gap) > 8 * self.phonon_energy:
                            continue

                        effective_energy_gap = energy_gap + self.stokes_shift
                        if effective_energy_gap > -critical_energy_gap:
                            donor_acceptor_overlap_integral = gaussian_overlap_integral(
                                np.abs(effective_energy_gap),
                                max(dopant_di.absFWHM[di],
                                    dopant_aj.absFWHM[aj]))
                            energy_transfer_rate = energy_transfer_constant(
                                donor_line_strength, acceptor_line_strength,
                                donor_acceptor_overlap_integral,
                                self.n_refract, Jdi, Jai)
                        else:
                            donor_acceptor_overlap_integral = gaussian_overlap_integral(
                                0,
                                max(dopant_di.absFWHM[di],
                                    dopant_aj.absFWHM[aj]))
                            energy_transfer_rate = phonon_assisted_energy_transfer_constant(
                                donor_line_strength, acceptor_line_strength,
                                donor_acceptor_overlap_integral,
                                self.n_refract, Jdi, Jai,
                                abs(effective_energy_gap), self.mpr_beta)

                        # TODO: Add SK_omitETtransitions to omit specific transitions
                        if (energy_transfer_rate * donor_concentration *
                                acceptor_concentration) > self.energy_transfer_rate_threshold:
                            energy_transfers.append([
                                combined_di, combined_dj, combined_ai,
                                combined_aj, energy_transfer_rate
                            ])

        energy_transfer_rates = np.array(np.vstack(energy_transfers))
        return energy_transfer_rates

    def make_migration_assisted_energy_transfer_rate_matrix(
            self) -> List[List[float]]:
        """
        :return:
        """
        # TODO: Is this needed? If so, port it
        pass

    def run_kinetics(
            self,
            initial_populations: Optional[Union[Sequence[Sequence[float]],
                                                str]] = 'ground_state',
                                                t: List[int] = None):
        if t is None:
            t = [0, 1]

        if isinstance(initial_populations, list):
            # check if supplied populations is the proper length
            if len(initial_populations) == self.total_n_levels:
                print('Using user input initial population')
            else:
                raise ValueError(
                    "Supplied Population is invalid. Expected length of {self.total_n_levels, "
                    "received length of {len(initial_population)}"
                )
        elif initial_populations == 'ground_state':
            initial_populations = self.get_ground_state()
        else:
            raise ValueError(
                "Invalid argument supplied for: initial_populations")

        sol = odeint(dNdt_from_populations, initial_populations, t, args=(self,))

        return sol

    def get_ground_state(self):
        initial_populations = []
        for dopant in self.dopants:
            _dopant_states = np.zeros(dopant.n_levels)
            _dopant_states[0] = 1
            initial_populations.append(_dopant_states)
        initial_populations = np.concatenate(initial_populations, axis=-1)
        return initial_populations


def dNdt_from_populations(populations, t, sk):

    dNdt = np.zeros(sk.total_n_levels)

    nr_rate = sk.non_radiative_rate_matrix * populations[:, None]
    ed_rate = sk.radiative_rate_matrix * populations[:, None]
    md_rate = sk.magnetic_dipole_rate_matrix * populations[:, None]

    dNdt = dNdt + nr_rate.sum(0) - nr_rate.sum(1)
    dNdt = dNdt + ed_rate.sum(0) - ed_rate.sum(1)
    dNdt = dNdt + md_rate.sum(0) - md_rate.sum(1)

    indices_array = sk.energy_transfer_rate_matrix[:, :4].astype(int)
    di, dj, ai, aj = indices_array.T
    et_rate = 4 * math.pi / 3 * (
        sk.energy_transfer_rate_matrix[:, 4] *
        1e42) * populations[ai] * populations[di] * (
            sk.min_dopant_distance**(-3) * 1e-21 -
            (4 * math.pi / 3) * populations[ai])

    # Prune away the zero rates, this improves performance by >4x per step
    nonzero_idx = np.nonzero(et_rate)
    nonzero_indices_array = indices_array[nonzero_idx]
    nonzero_di, nonzero_dj, nonzero_ai, nonzero_aj = nonzero_indices_array.T
    nonzero_et_rate = et_rate[nonzero_idx]

    np.add.at(dNdt, nonzero_indices_array[:, 0], -nonzero_et_rate)
    np.add.at(dNdt, nonzero_indices_array[:, 2], -nonzero_et_rate)
    np.add.at(dNdt, nonzero_indices_array[:, 1], nonzero_et_rate)
    np.add.at(dNdt, nonzero_indices_array[:, 3], nonzero_et_rate)

    return dNdt
