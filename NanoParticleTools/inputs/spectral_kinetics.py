from scipy.integrate import BDF, OdeSolver
from typing import Optional, List, Union, Sequence
from NanoParticleTools.species_data.species import Dopant
from NanoParticleTools.inputs.photo_physics import *
from NanoParticleTools.inputs.constants import *
import numpy as np

class SpectralKinetics():

    """
    REFERENCING THIS WORK & BIBLIOGRAPHY:
         There are 2 different citations that can be used to reference this work. These articles describe the underlying equations behind this work.

         (1)	Chan, E. M.; Gargas, D. J.; Schuck, P. J.; Milliron, D. J.
                    Concentrating and Recycling Energy in Lanthanide Codopants for Efficient and Spectrally Pure Emission: the Case of NaYF4:Er3+/Tm3+ Upconverting Nanocrystals.
                J. Phys. Chem. B 2012, 116, 10561–10570.

         (2)	Chan, E. M. Combinatorial Approaches for Developing Upconverting Nanomaterials: High-Throughput Screening, Modeling, and Applications.
                Chem. Soc. Rev. 2015, 44, 1653–1679.
    """
    def __init__(self, dopants, **kwargs):
        """

        #TODO: make dopants an argument
        :param dopants:
        :param kwargs:
        """

        self.MPR_W_0phonon = None
        self.phonon_energy = None
        self.mpr_alpha = None
        self.mpr_gamma = None
        self.mpr_beta = None
        self.n_refract = None
        self.volume_per_dopant_site = None
        self.minimum_dopant_distance = None
        self.set_matrix_parameters(**kwargs)

        self.ode_solver = None
        self.time_step = None
        self.num_steps = None
        self.ode_max_error = None
        self.energy_transfer_rate_threshold = None
        self.radiative_rate_threshold = None
        self.stokes_shift = None
        self.energy_transfer_mode = None
        self.initial_populations = None
        self.set_simulation_config_parameters(**kwargs)

        self.incident_power = None
        self.incident_wavelength = None
        self.incident_wavenumber = None
        self.incident_photon_flux = None
        self.set_excitation_parameters(**kwargs)

        self.dopants = dopants # [Dopant('Yb', 0.1, 2), Dopant('Er', 0.02, 34)]
        self.species_concentrations = [dopant.molar_concentration / self.volume_per_dopant_site for dopant in self.dopants]
        self.total_n_levels = sum([dopant.n_levels for dopant in self.dopants])

        self._line_strength_matrix = None
        self._non_radiative_rate_matrix = None
        self._radiative_rate_matrix = None
        self._magnetic_dipole_rate_matrix = None
        self._energy_transfer_rate_matrix = None
        self.set_kinetic_parameters()


    def SK_SetUpAndDoKinetics(self):
        """
        THIS should be your primary function call to run the simulations from the command line or macro menu
        sets up experimental and kinetic parmaeters, solves the differential equations, and analyzes the data

        :return:
        """
        self.set_kinetic_parameters()
        self.run_kinetics()

        # This may not be necessary since python doesn't need the folder structure
        # SK_SaveParamsToResultsFolder(SK_CURR_RESULT_FOLDER)

        self.SK_Analysis()
        return

    def set_kinetic_parameters(self,
                               initial_populations:Optional[Union[Sequence[Sequence[float]], str]]= 'ground_state'):
        """

        :param dopants: {species_name, concentration, numLevels} ex. [{'Yb', 0.1, 2}, {'Er', 0.02, 34}]
        :return:
            Creates 2 global variables in $KINETIC_PARAMS_FOLDER
            Variable /G V_numSpecies //number of lanthanide species (elements/ion types) in simulation
            Variable /G V_totNumLevels //total number of energy levels to simulate (sum of simulated levels for all ions)
        """
        #TODO: check inputs are valid
        self.initial_populations = initial_populations

        if isinstance(self.initial_populations, list):
            print('Using user input initial population')
            for dopant, initial_population in zip(self.dopants, self.initial_populations):
                dopant.set_initial_populations(self.initial_populations)
            # raise NotImplementedError("User defined populations not implemented")
        elif self.initial_populations == 'ground_state':
            for dopant in self.dopants:
                dopant.set_initial_populations()
        elif self.initial_populations == 'last_run':
            #TODO: Implement from last run. Although this may not be necessary, since one could just pass in the previous population
            raise NotImplementedError("Using populations from previous run not implemented")
        else:
            raise ValueError("Invalid argument supplied for: initial_populations")

        #
        # // Make matrices of rate constants
        #
        ## SK_CombineSpeciesInfo() #TODO: Determine if this is necessary (I don't think it is)
        ## SK_MakeNRrateMatrix() #Done, was mostly put into the species
        ## SK_MakeLineStrengthMatrix() #TODO: Determine if this is necessary (I don't think it is)
        ## SK_MakeRadiationRateMatrix()
        ## SK_MakeMDRateMatrix()
        ## SK_MakeETRateMatrix()
        #
        # return STATUS_SUCCESS
        pass

    @property
    def non_radiative_rate_matrix(self):
        if self._non_radiative_rate_matrix is None:
            self._non_radiative_rate_matrix = self.make_combined_non_radiative_rate_matrix()
        return self._non_radiative_rate_matrix

    def make_combined_non_radiative_rate_matrix(self) -> np.ndarray:
        """
        Makes the n x n M_NRrate matrix.  M_NRrate[i][j] gives the rate of non-radiative decay from level i->j,
        which are combined energy level indices
        :return:
        """
        for dopant in self.dopants:
            dopant.calculate_MPR_rates(self.MPR_W_0phonon, self.mpr_alpha, self.stokes_shift, self.phonon_energy)

        #TODO: Confirm if this is the correct matrix

        non_radiative_rates = np.zeros((self.total_n_levels + 2, self.total_n_levels + 2))
        first_index = 0
        for dopant in self.dopants:
            rates = np.identity(len(dopant.mpr_rates)) * dopant.mpr_rates
            non_radiative_rates[first_index + 1:first_index + 1 + dopant.n_levels,
            first_index:first_index + dopant.n_levels] += rates

            rates = np.identity(len(dopant.mpa_rates)) * dopant.mpa_rates
            non_radiative_rates[first_index + 1:first_index + 1 + dopant.n_levels,
            first_index + 1 + 1:first_index + 1 + 1 + dopant.n_levels] += rates
            first_index += dopant.n_levels
        #
        # non_radiative_rates = np.zeros((total_n_levels, total_n_levels))
        # first_index = 0
        # for dopant in self.dopants:
        #     for i in range(1, dopant.n_levels):
        #         non_radiative_rates[first_index + i, first_index + i - 1] = dopant.mpr_rates[i]
        #
        #     for i in range(0, dopant.n_levels):
        #         try:
        #             non_radiative_rates[first_index + i, first_index + i + 1] = dopant.mpa_rates[i]
        #         except:
        #             continue
        #
        #     first_index += dopant.n_levels
        return non_radiative_rates[1:self.total_n_levels+1, 1:self.total_n_levels+1]

    @property
    def line_strength_matrix(self):
        if self._line_strength_matrix is None:
            self._line_strength_matrix = self.make_combined_line_strength_matrix()
        return self._line_strength_matrix

    def make_combined_line_strength_matrix(self) -> np.ndarray:
        """
        makes the n x n lineStrengthMatrix from a wave of lineStrengths labeled with the transitions "i->j" in transitionLabels wave
        :return:
        """
        # TODO: make the combined rate matrix if necessary

        combined_line_strength_matrix = np.zeros((self.total_n_levels, self.total_n_levels))
        first_index = 0
        for dopant in self.dopants:
            _m = dopant.get_line_strength_matrix()
            combined_line_strength_matrix[first_index:first_index + dopant.n_levels,
            first_index:first_index + dopant.n_levels] = _m
            first_index += dopant.n_levels

        return combined_line_strength_matrix

    @property
    def radiative_rate_matrix(self):
        if self._radiative_rate_matrix is None:
            self._radiative_rate_matrix = self.make_radiative_rate_matrix()
        return self._radiative_rate_matrix

    def make_radiative_rate_matrix(self) -> np.ndarray:
        """

        :return:
        """
        rad_rates = np.zeros((self.total_n_levels, self.total_n_levels))
        energy_gaps = np.zeros((self.total_n_levels, self.total_n_levels))
        for dopant_index, dopant in enumerate(self.dopants):
            for i in range(dopant.n_levels):
                combined_i = sum([dopant.n_levels for dopant in self.dopants[:dopant_index]]) + i
                for j in range(dopant.n_levels):
                    combined_j = sum([dopant.n_levels for dopant in self.dopants[:dopant_index]]) + j
                    energy_gap = dopant.energy_levels[j].energy - dopant.energy_levels[i].energy
                    energy_gaps[combined_i, combined_j] = energy_gap

                    if energy_gap < 0:
                        rad_rates[combined_i][combined_j] = get_transition_rate_from_line_strength(energy_gap, self.line_strength_matrix[combined_i][combined_j], dopant.slj[i][2], self.n_refract)
                    elif energy_gap > 0:
                        # Going up in energy, therefore calculate the photon absorption based on abs. cross section and incident flux
                        absfwhm = dopant.absFWHM[j]
                        abs_sigma = absfwhm/(2*np.sqrt(2*np.log(2))) #convert fwhm to width in gaussian equation

                        absorption_cross_section = get_absorption_cross_section_from_line_strength(energy_gap, self.line_strength_matrix[combined_i][combined_j], dopant.slj[i][2], self.n_refract)

                        critical_energy_gap = get_critical_energy_gap(self.mpr_alpha, absfwhm)
                        if abs(energy_gap - self.incident_wavenumber) > critical_energy_gap:
                            individual_absorption_cross_section = absorption_cross_section * gaussian(energy_gap, energy_gap, abs_sigma)
                            mpr_assisted_absorption_correction_factor = np.exp(-self.mpr_alpha * np.abs(energy_gap - self.incident_wavenumber))
                        else:
                            individual_absorption_cross_section = absorption_cross_section * gaussian(energy_gap, self.incident_wavenumber, abs_sigma)
                            mpr_assisted_absorption_correction_factor = 1

                        radiative_rate = self.incident_photon_flux * individual_absorption_cross_section * mpr_assisted_absorption_correction_factor

                        # exponential term accounts for differences between incident energy and energy level gap
                        if radiative_rate > self.radiative_rate_threshold:
                            rad_rates[combined_i][combined_j] = radiative_rate

        return rad_rates

    @property
    def magnetic_dipole_rate_matrix(self):
        if self._magnetic_dipole_rate_matrix is None:
            self._magnetic_dipole_rate_matrix = self.make_magnetic_dipole_rate_matrix()
        return self._magnetic_dipole_rate_matrix

    def make_magnetic_dipole_rate_matrix(self) -> np.ndarray:
        """
        creates the MDradRate matrix containing the MD line strength in cm^2 from intermediate coupling coefficient vectors
        :return:
        """
        magnetic_dipole_radiative_rates = np.zeros((self.total_n_levels, self.total_n_levels))

        for dopant_index, dopant in enumerate(self.dopants):
            num_eigenvectors = len(dopant.eigenvector_sl)

            s_vector = dopant.eigenvector_sl[:, 0]
            l_vector = dopant.eigenvector_sl[:, 1]
            for i in range(dopant.n_levels):
                energy_i = dopant.energy_levels[i].energy
                combined_i = sum([dopant.n_levels for dopant in self.dopants[:dopant_index]]) + i

                coeff_vector_i = dopant.intermediate_coupling_coefficients[i] # May need to transpose
                Ji = dopant.slj[i][2]

                for j in range(dopant.n_levels):
                    energy_j = dopant.energy_levels[j].energy
                    energy_gap = energy_j - energy_i
                    combined_j = sum([dopant.n_levels for dopant in self.dopants[:dopant_index]]) + j

                    Jf = dopant.slj[j][2]
                    coeff_vector_j = dopant.intermediate_coupling_coefficients[j] # May need to transpose

                    #TODO: port this function
                    current_line_strength = get_MD_line_strength_from_icc(coeff_vector_i, coeff_vector_j, Ji, Jf, s_vector, l_vector)

                    if current_line_strength is None:
                        raise ValueError('Error, current_line_strength is not a valid number')

                    if energy_gap < 0:
                        # spontaneous emission
                        magnetic_dipole_radiative_rates[combined_i][combined_j] = get_rate_from_MD_line_strength(current_line_strength, energy_gap, Ji, self.n_refract)
                    elif energy_gap > 0:
                        # Going up in energy, calculate photon absorption based on abs. cross section and incident flux
                        absfwhm = dopant.absFWHM[j]
                        abs_sigma = absfwhm/(2*np.sqrt(2*np.log(2))) # convert fwhm to width in gaussian equation

                        absorption_cross_section = get_absorption_cross_section_from_MD_line_strength(current_line_strength, energy_gap, Ji, self.n_refract)
                        critical_energy_gap = get_critical_energy_gap(self.mpr_alpha, absfwhm)
                        if np.abs(energy_gap - self.incident_wavenumber) > critical_energy_gap:
                            individual_absorption_cross_section = absorption_cross_section*gaussian(energy_gap, energy_gap, abs_sigma)
                            mpr_assisted_absorption_correction_factor = np.exp(-self.mpr_alpha * np.abs(energy_gap - self.incident_wavenumber))
                        else:
                            # energy mismatch < critical energy gap, therefore don't use any MPR assistance
                            individual_absorption_cross_section = absorption_cross_section * gaussian(energy_gap, self.incident_wavenumber, abs_sigma)
                            mpr_assisted_absorption_correction_factor = 1

                        #TODO: Resolve comment: "FIXME -- this should probably be enabled since there is no MD absorption without it, but need to double check if it is correct"
                        # absorption_rate = individual_absorption_cross_section*self.incident_photon_flux * mpr_assisted_absorption_correction_factor
                        absorption_rate = 0
                        if absorption_rate > self.radiative_rate_threshold:
                            magnetic_dipole_radiative_rates[combined_i][combined_j] = absorption_rate
        return magnetic_dipole_radiative_rates

    @property
    def energy_transfer_rate_matrix(self):
        if self._energy_transfer_rate_matrix is None:
            self._energy_transfer_rate_matrix = self.make_phonon_assisted_energy_transfer_rate_matrix()
        return self._energy_transfer_rate_matrix

    def make_phonon_assisted_energy_transfer_rate_matrix(self)->List[List[float]]:
        """
        makes the phonon assisted (not migration assisted)  energy transfer rate constant waves (W_ETrates, W_ETIndices)

        :return: the phonon-assisted, non-migration-assisted energy transfer rate constant K (1/s)
        """
        _length = int(np.round((self.total_n_levels ** 4 / 4)))
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
            di = combined_di - sum([dopant.n_levels for dopant in self.dopants[:dopant_di_index]])

            Jdi = dopant_di.slj[di, 2]
            absfwhm = dopant_di.absFWHM[di]
            critical_energy_gap = get_critical_energy_gap(self.mpr_beta, absfwhm)
            donor_concentration = dopant_di.volume_concentration * 1e21 #convert from nm^-3 to cm^-3
            combined_donor_ground_state_index = sum([dopant.n_levels for dopant in self.dopants[:dopant_di_index]])

            for combined_dj in range(self.total_n_levels):
                dopant_dj_index = species_map[combined_dj]
                dopant_dj = self.dopants[dopant_dj_index]
                dj = combined_dj - sum([dopant.n_levels for dopant in self.dopants[:dopant_dj_index]])

                donor_energy_change = dopant_dj.energy_levels[dj].energy - dopant_di.energy_levels[di].energy
                if donor_energy_change >= 0:
                    # donor transition dE should be negative
                    continue

                # adjust for stokes shift
                # donor_energy_change += self.stokesShift

                donor_line_strength = self.line_strength_matrix[combined_di, combined_dj]
                donor_to_ground_state_line_strength = self.line_strength_matrix[combined_di][combined_donor_ground_state_index]

                for combined_ai in range(self.total_n_levels):
                    dopant_ai_index = species_map[combined_ai]
                    dopant_ai = self.dopants[dopant_ai_index]
                    ai = combined_ai - sum([dopant.n_levels for dopant in self.dopants[:dopant_ai_index]])
                    Jai = dopant_ai.slj[ai, 2]

                    acceptor_concentration = dopant_ai.volume_concentration * 1e21  # convert from nm^-3 to cm^-3

                    for combined_aj in range(self.total_n_levels):
                        dopant_aj_index = species_map[combined_aj]
                        dopant_aj = self.dopants[dopant_aj_index]
                        aj = combined_aj - sum([dopant.n_levels for dopant in self.dopants[:dopant_aj_index]])
                        Jaj = dopant_aj.slj[aj, 2]
                        acceptor_energy_change = dopant_aj.energy_levels[aj].energy - dopant_ai.energy_levels[ai].energy

                        if acceptor_energy_change <= 0:
                            # acceptor transition dE should be positive
                            continue

                        acceptor_line_strength = self.line_strength_matrix[combined_ai, combined_aj]

                        energy_gap = acceptor_energy_change + donor_energy_change

                        if energy_gap > 2 * critical_energy_gap:
                            # So that energy transfer cannot be too uphill
                            continue

                        if np.abs(energy_gap) > 8 * self.phonon_energy:
                            continue

                        effective_energy_gap = energy_gap + self.stokes_shift
                        if effective_energy_gap > -critical_energy_gap:
                            donor_acceptor_overlap_integral = gaussian_overlap_integral(np.abs(effective_energy_gap), max(dopant_di.absFWHM[di], dopant_aj.absFWHM[aj]))
                            energy_transfer_rate = energy_transfer_constant(donor_line_strength, acceptor_line_strength, donor_acceptor_overlap_integral, self.n_refract, Jdi, Jai)
                        else:
                            donor_acceptor_overlap_integral = gaussian_overlap_integral(0, max(dopant_di.absFWHM[di], dopant_aj.absFWHM[aj]))
                            energy_transfer_rate = phonon_assisted_energy_transfer_constant(donor_line_strength, acceptor_line_strength, donor_acceptor_overlap_integral, self.n_refract, Jdi, Jai, abs(effective_energy_gap), self.mpr_beta)

                        # TODO: Add SK_omitETtransitions to omit specific transitions
                        if (energy_transfer_rate*donor_concentration*acceptor_concentration) > self.energy_transfer_rate_threshold:
                            energy_transfers.append([combined_di, combined_dj, combined_ai, combined_aj, energy_transfer_rate])

        energy_transfer_rates = np.array(np.vstack(energy_transfers))
        return energy_transfer_rates

    def make_migration_assisted_energy_transfer_rate_matrix(self) -> List[List[float]]:
        """
        :return:
        """
        #TODO: Is this needed? If so, port it
        pass

    def set_matrix_parameters(self, phonon_energy:float=450,
                              zero_phonon_rate:float=1e7,
                              mpr_alpha:float=3.5e-3,
                              n_refract:float=1.5,
                              vol_per_dopant_site:float=7.23946667e-2,
                              min_dopant_distance:float=3.867267554e-8,
                              **kwargs):
        """

        :param phonon_energy: in wavenumbers
        :param zero_phonon_rate: zero phonon relaxation rate at T=0, in 1/s
        :param mpr_alpha: in cm
        :param n_refract: index of refraction
        :param vol_per_dopant_site: cm^3 for NaYF4, 1.5 is # possible sites for dopant ion
        :param min_dopant_distance: minimum distance between dopants (cm)
        :param kwargs:
        :return:
        """
        self.MPR_W_0phonon = zero_phonon_rate
        self.phonon_energy = phonon_energy #FIXME : PP_CalculateMPRratesMD relies on constant PHONON_ENERGY
        #FIXME : PP_CalculateMPRratesMD relies on phononEnergy constant

        self.mpr_alpha = mpr_alpha # in cm
        self.mpr_gamma = np.log(2) / self.phonon_energy # in cm
        self.mpr_beta = self.mpr_alpha - self.mpr_gamma # in cm

        self.n_refract = n_refract
        self.volume_per_dopant_site = vol_per_dopant_site
        self.minimum_dopant_distance = min_dopant_distance

    def set_simulation_config_parameters(self,
                                         time_step:Optional[float]=1e-4,
                                         num_steps:Optional[float]=100,
                                         max_error:Optional[float]=1e-12,
                                         energy_transfer_rate_threshold:Optional[float]=0.1,
                                         radiative_rate_threshold:Optional[float]=0.0001,
                                         stokes_shift:Optional[float]=150,
                                         # energy_transfer_mode=0,
                                         initial_populations:Optional[str]="ground_state",
                                         ode_solver:Optional[OdeSolver]=BDF,
                                         **kwargs):
        """

        :param time_step: time step in seconds
        :param num_steps: number of time steps per simulation
        :param max_error: default is 1e-6
        :param energy_transfer_rate_threshold: lower limit for actually accounting for ET rate (s^-1)
        :param radiative_rate_threshold: lower limit for actually accounting for radiative rate (s^-1)
        :param stokes_shift: wavenumbers
        :param energy_transfer_mode:
        :param initial_populations:
        :param ode_solver:
        :param kwargs:
        :return:
        """

        self.time_step = time_step
        self.num_steps = num_steps
        self.ode_max_error = max_error
        self.energy_transfer_rate_threshold = energy_transfer_rate_threshold
        self.radiative_rate_threshold = radiative_rate_threshold
        self.stokes_shift = stokes_shift   #FIXME : PP_CalculateMPRratesMD relies on constant
        self.initial_populations = initial_populations
        self.ode_solver = ode_solver

        # TODO: 0=no migration, 1 = fast diffusion, 2 = fast hopping
        # self.energy_transfer_mode = energy_transfer_mode

        #TODO: Need to figure out what data is stored in the initialPopWavePath
        # if (!ParamIsDefault(initPopPath) && V_initPopMode == INIT_POPS_ABS_FROM_USER)
        #     String /G initialPopWavePath = initPopPath
        # else
        #     String /G initialPopWavePath = ""
        # endif

    def set_excitation_parameters(self,
                                  excitation_wavelength:Optional[float]=976,
                                  excitation_power:Optional[float]=1e7,
                                  **kwargs):
        """

        :param excitation_wavelength: the wavelength of the incident radiation in in nm
        :param excitation_power: the incident power density in ergs/cm^2 (1e7 erg/s/cm^2 = 1 W/cm^2)
        :param kwargs:
        :return:
        """
        self.incident_power = excitation_power
        self.incident_wavelength = excitation_wavelength

        self.incident_wavenumber = 1e7 / excitation_wavelength # in cm^-1
        self.incident_photon_flux = excitation_power / (h_CGS * c_CGS * self.incident_wavenumber) # in photons/s/cm^2


    def run_kinetics(self, dopants:List[dict]):
        """
        SOLVES the differential equations without doing any of the setup or analysis of SK_SetUpAndDoKinetics
        ASSUMES that all of the proper values have already been stored in the global variables

        :return:
        """


        """
        SK_SetSimParams()
        SetDataFolder $SIM_CONFIG_FOLDER //leaves system in $SK_CALC_RATES_FOLDER
        NVAR ODEsolveMethod
        NVAR ODEmaxError

        SetDataFolder $SK_CALC_RATES_FOLDER

        Wave /D W_pop_time
        """

        # Default ODE solve method is Backwards Differentiation Formula (/M =3)
        # The error in the differentiation formula. Default is 1e-12 (/E=1e-12)
        # /Q =0 turns off quiet mode
        # derivFunc is "SK_diffKinetics". The name of a user function that calculates derivatives
        # cwaveName is KK. This is the name of the wave containing constant coefficients to be passed to derivFunc
        # ywaveSpec is W_pop_time. Specifies a wave or waves to receive calculated results
        # IntegrateODE/M=(ODEsolveMethod)/U=1 /E=(ODEmaxError) /Q=0 SK_diffKinetics, KK, W_pop_time

        self.ode_solver(self.SK_diffKinetics, atol=self.ode_max_error)
        pass

    def SK_diffKinetics(self, params, tt, N_pop, dNdt):
        """
        #This function is intended to be called by igor's IntegrateODE function

        #FYI, this function does not use matrix math (e.g., MatrixOps). It was tested and found to perform slower
        #than iterating through For loops.

        :param params: params is the parameter wave, which is not used in this function
            Instead, this function calls global waves M_ETRate, M_RadRate, M_NRrate in $KINETIC_PARAMS_FOLDER
            These must be set or else the function will return gibberish
        :param tt: time value at which to calculate derivatives
        :param N_pop: the fractional occupation or population of each level.
            yw[0]-yw[3] containing concentrations of A,B,C,D
        :param dNdt: the differential change in each level corresponding to the elements of N_pop
            wave to receive dA/dt, dB/dt etc. (output)
        :return:
        """
        #TODO
        pass

    def SK_Analysis(self):
        pass
