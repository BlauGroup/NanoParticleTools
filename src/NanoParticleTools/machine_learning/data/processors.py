from typing import Union, Dict, List, Tuple, Optional
import torch
import numpy as np
from monty.json import MSONable
from NanoParticleTools.inputs import NanoParticleConstraint, SphericalConstraint
from scipy.ndimage import gaussian_filter1d
from torch_geometric.data import Data
import warnings
from abc import ABC, abstractmethod


class DataProcessor(ABC, MSONable):
    """
    Template for a data processor. The data processor allows modularity in definitions
    of how data is to be converted from a dictionary (typically a fireworks output document)
    to the desired form. This can be used for features or labels.

    To implementation a DataProcessor, override the process_doc function.

    Fields are specified to ensure they are present in documents
    """

    def __init__(self, fields: List[str]):
        """
        :param fields: fields required in the document(s) to be processed
        """
        self.fields = fields

    @property
    def required_fields(self) -> List[str]:
        return self.fields

    @abstractmethod
    def process_doc(self, doc: Dict) -> Dict:
        raise NotImplementedError

    def get_item_from_doc(self, doc: dict, field: str):
        keys = field.split('.')

        val = doc
        for key in keys:
            val = val[key]
        return val


class FeatureProcessor(DataProcessor):

    def __init__(self, possible_elements = ['Yb', 'Er', 'Nd'], **kwargs):
        self.possible_elements = possible_elements
        self.n_possible_elements = len(self.possible_elements)
        self.dopants_dict = {
            key: i
            for i, key in enumerate(self.possible_elements)
        }
        super().__init__(**kwargs)

    def example(self):
        n_elements = len(self.possible_elements)
        dopant_specifications = [
            (0, 0.2,
             self.possible_elements[torch.randint(0, n_elements,
                                                  (1, )).item()], 'Y'),
            (0, 0.1,
             self.possible_elements[torch.randint(0, n_elements,
                                                  (1, )).item()], 'Y'),
            (1, 0.1,
             self.possible_elements[torch.randint(0, n_elements,
                                                  (1, )).item()], 'Y')
        ]
        dopant_concentration = []
        for layer_idx, dopant_conc, dopant, _ in dopant_specifications:
            while layer_idx + 1 > len(dopant_concentration):
                dopant_concentration.append({})
            dopant_concentration[layer_idx][dopant] = dopant_conc
        doc = {
            'dopant_concentration': dopant_concentration,
            'input': {
                'constraints':
                [SphericalConstraint(40),
                 SphericalConstraint(65)],
                'dopant_specifications': dopant_specifications
            }
        }
        return self.process_doc(doc)

    @staticmethod
    def get_radii(idx, constraints):
        if idx == 0:
            # The constraint was the first one, therefore the inner radius is 0
            r_inner = 0
        else:
            r_inner = constraints[idx - 1].radius
        r_outer = constraints[idx].radius
        return r_inner, r_outer

    @staticmethod
    def get_volume(r):
        return 4 / 3 * np.pi * (r**3)

    @property
    def is_graph(self):
        raise NotImplementedError

    @property
    def data_cls(self):
        return Data


class LabelProcessor(DataProcessor):

    def example(self):
        x = torch.linspace(*self.spectrum_range, 600)
        y = torch.nn.functional.relu(torch.rand_like(x))
        doc = {
            'output': {
                'wavelength_spectrum_x': list(x),
                'wavelength_spectrum_y': list(y),
                'energy_spectrum_x': list(x),
                'energy_spectrum_y': list(y),
            }
        }
        return self.process_doc(doc)


class EnergySpectrumLabelProcessor(LabelProcessor):
    """
    This Label processor returns a spectrum that is binned uniformly with respect to energy (I(E))
        Args:
            spectrum_range (Tuple | List, optional): Range over which the spectrum should be
                cropped. Defaults to (-40000, 20000).
            output_size (int, optional): Number of bins in the resultant spectra.
                This quantity will be used as the # of output does in the NN. Defaults to 600.
            log_constant (float, optional): When applying the log function,
                we use the form log_10(I+b). Since the intensity is always positive, this function
                is easily invertible. min_log_val sets the minimum value of the label after
                applying the log.

                To make sure the values aren't clipped, it is recommended that the smallest b is
                chosen at least 1 order of magnitude lower than 1/(# avg'd documents).

                Example:
                    With 16 documents averaged, the lowest (non-zero) observation is 0.0625(1/16),
                    therefore choose 0.001 as the log_constant. . Defaults to 1e-3.
            gaussian_filter (float, optional): Standard deviation over which to apply gaussian
                filtering to smooth the otherwise very peaked spectrum. Defaults to 0.
    """

    def __init__(self,
                 spectrum_range: Tuple | List = (-40000, 20000),
                 output_size: int = 600,
                 log_constant: float = 1e-3,
                 gaussian_filter: float = 0,
                 **kwargs):
        super().__init__(
            fields=['output.energy_spectrum_x', 'output.energy_spectrum_y'],
            **kwargs)

        self.spectrum_range = spectrum_range
        self.output_size = output_size
        self.log_constant = log_constant
        self.gaussian_filter = gaussian_filter

    def process_doc(self, doc: dict) -> torch.Tensor:
        x = torch.tensor(
            self.get_item_from_doc(doc, 'output.energy_spectrum_x'))
        spectrum = torch.tensor(
            self.get_item_from_doc(doc, 'output.energy_spectrum_y'))
        step = x[1] - x[0]

        # Assign to a different variable, so we can modify
        # the spectrum while keeping a reference to the original
        y = spectrum
        if self.gaussian_filter > 0:
            y = torch.tensor(gaussian_filter1d(y, self.gaussian_filter))

        target_range = torch.linspace(*self.spectrum_range,
                                      self.output_size + 1)
        target_step = target_range[1] - target_range[0]
        target_range = (target_range + target_step / 2)[:-1]
        if target_step == step and target_range[0] > x[0] and target_range[
                -1] < x[-1]:
            # The specified range is a subset of the original spectrum, we can just crop
            # the spectrum
            start = torch.nonzero(torch.isclose(x, target_range[0]))
            end = torch.nonzero(torch.isclose(x, target_range[-1]))

            x = x[start:(end + 1)]
            y = y[start:(end + 1)]
        elif x.shape[0] != self.output_size:
            if x.shape[0] >= self.output_size:
                warnings.warn(
                    "Desired spectrum resolution is coarser than found in the document."
                    " Spectrum will be rebinned approximately. It is recommended to rebuild"
                    " the collection to match the desired resolution")
                # We need to rebin the distribution
                multiplier = int(
                    torch.lcm(torch.tensor(x.size(0)),
                              torch.tensor(self.output_size)) / x.shape[0])

                _spectrum = y.expand(multiplier, -1).moveaxis(0, 1).reshape(
                    self.output_size, -1).sum(dim=-1)
                _spectrum = _spectrum * x.size(0) / (
                    multiplier * self.output_size
                )  # ensure integral is constant

                # Get the edges of the spectra
                lower_bound = x[0] - (step / 2)
                upper_bound = x[-1] + (step / 2)

                # Construct the new array
                _x = torch.linspace(lower_bound, upper_bound, self.output_size)

                # Replace the old spectrum with the new
                x = _x
                y = _spectrum
            else:
                raise RuntimeError(
                    "Spectrum in document is different than desired resolution and cannot"
                    " be rebinned. Please rebuild the collection")

        # Keep track of where the spectrum changes from
        # emission to absorption.
        idx_zero = torch.tensor(
            int(np.floor(0 - self.spectrum_range[0]) / step))

        # Count the total number of photons, we can add this to the loss
        n_photons_absorbed = torch.sum(spectrum[idx_zero:])
        n_photons_emitted = torch.sum(spectrum[:idx_zero])

        # Integrate the energy absorbed vs emitted.
        # This can be added to the loss to enforce conservation of energy
        total_energy = y * x
        e_absorbed = torch.sum(total_energy[idx_zero:])
        e_emitted = torch.sum(total_energy[:idx_zero])

        return {
            'spectra_x': x.unsqueeze(0),
            'y': y.float().unsqueeze(0),
            'log_y': torch.log10(y + self.log_constant).float().unsqueeze(0),
            'log_const': self.log_constant,
            'n_absorbed': n_photons_absorbed,
            'n_emitted': n_photons_emitted,
            'e_absorbed': e_absorbed,
            'e_emitted': e_emitted
        }

    def __str__(self):
        return (f"Energy Label Processor - {self.output_size} bins, x_min ="
                " {self.spectrum_range[0]}, x_max = {self.spectrum_range[1]},"
                " log_constant = {self.log_constant}")


class TotalEnergyLabelProcessor(LabelProcessor):
    """
    This Label processor returns a spectrum that is binned uniformly with respect to energy (I(E))
        Args:
            spectrum_range (Tuple | List, optional): Range over which the spectrum should be
                cropped. Defaults to (-40000, 20000).
            output_size (int, optional): Number of bins in the resultant spectra.
                This quantity will be used as the # of output does in the NN. Defaults to 600.
            log_constant (float, optional): When applying the log function,
                we use the form log_10(I+b). Since the intensity is always positive, this function
                is easily invertible. min_log_val sets the minimum value of the label after
                applying the log.

                To make sure the values aren't clipped, it is recommended that the smallest b is
                chosen at least 1 order of magnitude lower than 1/(# avg'd documents).

                Example:
                    With 16 documents averaged, the lowest (non-zero) observation is 0.0625(1/16),
                    therefore choose 0.001 as the log_constant. . Defaults to 1e-3.
            gaussian_filter (float, optional): Standard deviation over which to apply gaussian
                filtering to smooth the otherwise very peaked spectrum. Defaults to 0.
    """

    def __init__(self,
                 spectrum_range: Tuple | List = (-40000, 20000),
                 log_constant: float = 1e-3,
                 gaussian_filter: float = 0,
                 **kwargs):
        super().__init__(
            fields=['output.energy_spectrum_x', 'output.energy_spectrum_y'],
            **kwargs)

        self.spectrum_range = spectrum_range
        self.log_constant = log_constant
        self.gaussian_filter = gaussian_filter

    def process_doc(self, doc: dict) -> torch.Tensor:
        x = torch.tensor(
            self.get_item_from_doc(doc, 'output.energy_spectrum_x'))
        spectrum = torch.tensor(
            self.get_item_from_doc(doc, 'output.energy_spectrum_y'))
        step = x[1] - x[0]

        # Assign to a different variable, so we can modify
        # the spectrum while keeping a reference to the original
        y = spectrum
        if self.gaussian_filter > 0:
            y = torch.tensor(gaussian_filter1d(y, self.gaussian_filter))

        if self.spectrum_range[0] > x[0] and self.spectrum_range[-1] < x[-1]:
            # The specified range is a subset of the original spectrum, we can just crop
            # the spectrum
            start = torch.nonzero(x < self.spectrum_range[0])[-1][0]
            end = torch.nonzero(x > self.spectrum_range[-1])[0][0]
            x = x[start:(end + 1)]
            y = y[start:(end + 1)]

        y_sum = y.sum()

        return {
            'y': y_sum.float().reshape(1, 1),
            'log_y':
            torch.log10(y_sum + self.log_constant).float().reshape(1, 1),
            'log_const': self.log_constant
        }

    def __str__(self):
        return ("Energy Label Processor - 1 bins, x_min ="
                f" {self.spectrum_range[0]}, x_max = {self.spectrum_range[1]},"
                f" log_constant = {self.log_constant}")


class WavelengthSpectrumLabelProcessor(LabelProcessor):
    r"""
    This Label processor returns a spectrum that is binned uniformly with respect to
    wavelength $I(\lambda{})$
        Args:
            spectrum_range (Tuple | List, optional): Range over which the spectrum should be
                cropped. Defaults to (-1000, 1000).
            output_size (int, optional): Number of bins in the resultant spectra.
                This quantity will be used as the # of output does in the NN. Defaults to 600.
            log_constant (float, optional): When applying the log function,
                we use the form log_10(I+b). Since the intensity is always positive, this function
                is easily invertible. min_log_val sets the minimum value of the label after
                applying the log.

                To make sure the values aren't clipped, it is recommended that the smallest b is
                chosen at least 1 order of magnitude lower than 1/(# avg'd documents).

                Example:
                    With 16 documents averaged, the lowest (non-zero) observation is 0.0625(1/16),
                    therefore choose 0.001 as the log_constant. . Defaults to 1e-3.
            gaussian_filter (float, optional): Standard deviation over which to apply gaussian
                filtering to smooth the otherwise very peaked spectrum. Defaults to 0.
    """

    def __init__(self,
                 spectrum_range: Union[Tuple, List] = (-1000, 1000),
                 output_size: Optional[int] = 600,
                 log_constant: Optional[float] = 1e-3,
                 gaussian_filter: Optional[float] = None,
                 **kwargs):
        if gaussian_filter is None:
            gaussian_filter = 0

        super().__init__(fields=[
            'output.wavelength_spectrum_x', 'output.wavelength_spectrum_y',
            'output.summary', 'overall_dopant_concentration'
        ], **kwargs)

        self.spectrum_range = spectrum_range
        self.output_size = output_size
        self.log_constant = log_constant
        self.gaussian_filter = gaussian_filter

    def process_doc(self, doc: dict) -> torch.Tensor:
        x = torch.tensor(
            self.get_item_from_doc(doc, 'output.wavelength_spectrum_x'))
        spectrum = torch.tensor(
            self.get_item_from_doc(doc, 'output.wavelength_spectrum_y'))
        step = x[1] - x[0]

        if x.shape[0] != self.output_size:
            if x.shape[0] >= self.output_size:
                warnings.warn(
                    "Desired spectrum resolution is coarser than found in the document."
                    " Spectrum will be rebinned approximately. It is recommended to rebuild"
                    " the collection to match the desired resolution")
                # We need to rebin the distribution
                multiplier = int(
                    torch.lcm(torch.tensor(x.size(0)),
                              torch.tensor(self.output_size)) / x.shape[0])

                _spectrum = spectrum.expand(multiplier, -1).moveaxis(
                    0, 1).reshape(self.output_size, -1).sum(dim=-1)
                _spectrum = _spectrum * x.size(0) / (
                    multiplier * self.output_size
                )  # ensure integral is constant

                # Get the edges of the spectra
                lower_bound = x[0] - (step / 2)
                upper_bound = x[-1] + (step / 2)

                # Construct the new array
                _x = torch.linspace(lower_bound, upper_bound, self.output_size)

                # Replace the old spectrum with the new
                x = _x
                spectrum = _spectrum
            else:
                raise RuntimeError(
                    "Spectrum in document is different than desired resolution and"
                    " cannot be rebinned. Please rebuild the collection")

        # Assign to a different variable, so we can modify
        # the spectrum while keeping a reference to the original
        y = spectrum
        if self.gaussian_filter > 0:
            y = torch.tensor(gaussian_filter1d(y, self.gaussian_filter))

        # Keep track of where the spectrum changes from
        # emission to absorption.
        idx_zero = torch.tensor(
            int(np.floor(0 - self.spectrum_range[0]) / step))

        # Count the total number of photons, we can add this to the loss
        n_photons_absorbed = torch.sum(spectrum[idx_zero:])
        n_photons_emitted = torch.sum(spectrum[:idx_zero])

        # Integrate the energy absorbed vs emitted.
        # This can be added to the loss to enforce conservation of energy
        total_energy = spectrum * x
        e_absorbed = torch.sum(total_energy[idx_zero:])
        e_emitted = torch.sum(total_energy[:idx_zero])

        return {
            'spectra_x': x.unsqueeze(0),
            'y': y.float().unsqueeze(0),
            'log_y': torch.log10(y + self.log_constant).float().unsqueeze(0),
            'log_const': self.log_constant,
            'n_absorbed': n_photons_absorbed,
            'n_emitted': n_photons_emitted,
            'e_absorbed': e_absorbed,
            'e_emitted': e_emitted
        }

    def __str__(self):
        return (
            f"Wavelength Label Processor - {self.output_size} bins, "
            f"x_min = {self.spectrum_range[0]}, x_max = {self.spectrum_range[1]}, "
            f"log_constant = {self.log_constant}")


class SummedWavelengthRangeLabelProcessor(LabelProcessor):
    """
    This Label processor returns a spectrum that is binned uniformly with respect to energy (I(E))
        Args:
            spectrum_range (Tuple | List, optional): Range over which the spectrum should be
                cropped. Defaults to (-40000, 20000).
            output_size (int, optional): Number of bins in the resultant spectra.
                This quantity will be used as the # of output does in the NN. Defaults to 600.
            log_constant (float, optional): When applying the log function,
                we use the form log_10(I+b). Since the intensity is always positive, this function
                is easily invertible. min_log_val sets the minimum value of the label after
                applying the log.

                To make sure the values aren't clipped, it is recommended that the smallest b is
                chosen at least 1 order of magnitude lower than 1/(# avg'd documents).

                Example:
                    With 16 documents averaged, the lowest (non-zero) observation is 0.0625(1/16),
                    therefore choose 0.001 as the log_constant. . Defaults to 1e-3.
            gaussian_filter (float, optional): Standard deviation over which to apply gaussian
                filtering to smooth the otherwise very peaked spectrum. Defaults to 0.
    """

    def __init__(self,
                 in_range: Tuple[float, float] = (-2000, 1000),
                 in_bins: int = 600,
                 log_constant: float = 1e-3,
                 spectrum_ranges: dict = None,
                 **kwargs):
        super().__init__(fields=[
            'output.wavelength_spectrum_x', 'output.wavelength_spectrum_y'
        ], **kwargs)

        self.in_range = in_range
        self.in_bins = in_bins
        self.log_constant = log_constant
        if spectrum_ranges is None:
            self.spectrum_ranges = dict(uv=(100, 400),
                                        uva=(315, 400),
                                        uvb=(280, 315),
                                        uvc=(100, 280),
                                        vis=(380, 750),
                                        blue=(450, 485),
                                        green=(500, 565),
                                        red=(625, 750))
        else:
            self.spectrum_ranges = spectrum_ranges

    def example(self):
        x = torch.linspace(-2000, 1000, 600)
        y = torch.nn.functional.relu(torch.rand_like(x))
        doc = {
            'output': {
                'wavelength_spectrum_x': list(x),
                'wavelength_spectrum_y': list(y),
                'energy_spectrum_x': list(x),
                'energy_spectrum_y': list(y),
            }
        }
        return self.process_doc(doc)

    @property
    def spectrum_idxs(self):
        spectrum_idxs = dict()
        for key, mask in self.masks.items():
            _range = torch.where(mask)[0]
            spectrum_idxs[key] = (_range[0].item(), _range[-1].item())
        return spectrum_idxs

    @property
    def x(self):
        dx = (self.in_range[-1] - self.in_range[0]) / self.in_bins
        return torch.linspace(self.in_range[0] + dx / 2,
                              self.in_range[-1] - dx / 2, self.in_bins)

    @property
    def masks(self):
        return {
            k: self.get_mask(self.x, _range)
            for k, _range in self.spectrum_ranges.items()
        }

    @staticmethod
    def get_mask(x, range):
        return torch.logical_and(x < (-range[0]), x > (-range[-1]))

    def process_doc(self, doc: dict) -> torch.Tensor:
        x = torch.tensor(
            self.get_item_from_doc(doc, 'output.wavelength_spectrum_x'))
        spectrum = torch.tensor(
            self.get_item_from_doc(doc, 'output.wavelength_spectrum_y'))
        step = x[1] - x[0]

        # Sum up the spectrum for each light type
        out = dict()
        for key, _mask in self.masks.items():
            out[key] = spectrum[_mask].sum().float()

        y = torch.hstack(tuple(out.values()))
        labels = list(out.keys())
        return {
            'intensities': out,
            'labels': labels,
            'y': y.unsqueeze(0),
            'log_y': torch.log10(y + self.log_constant).unsqueeze(0),
            'log_const': self.log_constant,
        }

    def __str__(self):
        return (f"Summed Ranges Label Processor - {len(self.spectrum_ranges)} bins, "
                f"log_constant = {self.log_constant}")

    def __repr__(self):
        return str(self)
