from NanoParticleTools.inputs import NanoParticleConstraint
from NanoParticleTools.machine_learning.data.processors import FeatureProcessor

from typing import List, Tuple, Optional, Any
import torch
import itertools
from torch_geometric.data.data import Data
from monty.json import MontyDecoder


class CNNFeatureProcessor(FeatureProcessor):

    def __init__(self,
                 resolution: Optional[float] = 0.1,
                 max_np_radius: Optional[int] = 250,
                 dims: Optional[int] = 1,
                 full_nanoparticle: Optional[bool] = True,
                 **kwargs):
        """
        :param possible_elements:
        :param cutoff_distance:
        :param resolution: Angstroms
        :param max_np_radius: Angstroms
        """
        # yapf: disable
        super().__init__(fields=[
            'formula_by_constraint', 'dopant_concentration', 'input'], **kwargs)
        # yapf: enable

        self.n_possible_elements = len(self.possible_elements)
        self.dopants_dict = {
            key: i
            for i, key in enumerate(self.possible_elements)
        }
        self.resolution = resolution
        self.max_np_radius = max_np_radius

        assert dims > 0 and dims <= 3, "Representation for must be 1, 2, or 3 dimensions"

        self.dims = dims
        self.full_nanoparticle = full_nanoparticle

    def dopant_specification_to_concentration_tensor(self,
                                                     dopant_specifications,
                                                     constraints):
        dopant_concentrations = torch.zeros(len(constraints),
                                            self.n_possible_elements)

        # Fill in the concentrations that are present
        for i, x, el, _ in dopant_specifications:
            dopant_concentrations[i][self.dopants_dict[el]] = x
        return dopant_concentrations

    def get_node_features(
        self, constraints: List[NanoParticleConstraint],
        dopant_specifications: List[Tuple[int, float, str,
                                          str]]) -> torch.Tensor:
        # Generate the tensor of concentrations
        concentrations = self.dopant_specification_to_concentration_tensor(
            dopant_specifications, constraints)

        concentrations.requires_grad = self.input_grad
        radii_without_zero = torch.tensor([c.radius for c in constraints],
                                          dtype=torch.float32,
                                          requires_grad=self.input_grad)

        node_features = to_nd_image(concentrations, radii_without_zero,
                                    self.max_np_radius, self.resolution,
                                    self.dims, self.full_nanoparticle)

        return {'x': node_features.unsqueeze(0)}

    def process_doc(self, doc: dict) -> dict:
        constraints = doc['input']['constraints']
        dopant_concentration = doc['dopant_concentration']

        constraints = MontyDecoder().process_decoded(constraints)

        _dopant_specifications = [
            (layer_idx, conc, el, None)
            for layer_idx, dopants in enumerate(dopant_concentration)
            for el, conc in dopants.items() if el in self.possible_elements
        ]

        return self.get_node_features(constraints, _dopant_specifications)

    @property
    def is_graph(self):
        return True

    @property
    def data_cls(self):
        return Data


def to_nd_image(conc_tensor: torch.Tensor,
                radii_without_zeros: torch.Tensor,
                max_image_size: int,
                image_resolution: float,
                dims: int,
                full_particle: bool = False):
    n_dopants = conc_tensor.size(1)
    image_dim = int(max_image_size // image_resolution) + 1
    constraint_radii = torch.cat((torch.tensor([-0.001]), radii_without_zeros),
                                 dim=-1)
    max_nonzero_radius = min(constraint_radii[-1].int().item(), max_image_size)

    # get the empty tensor tensor
    zero_tensor = torch.zeros([image_dim] * dims + [n_dopants])

    # compute the distance from the center)
    radius = torch.linspace(0, max_nonzero_radius, int(max_nonzero_radius // image_resolution + 1))
    mg = torch.meshgrid(*[radius for _ in range(dims)], indexing='ij')
    radial_distance = torch.sqrt(
        torch.sum(torch.stack([torch.pow(_el, 2) for _el in mg]), dim=0))

    for i in range(1, len(constraint_radii)):
        # get the indices of the pixels that are within the radius
        idx = torch.where((radial_distance > constraint_radii[i - 1])
                          & (radial_distance <= constraint_radii[i]))

        # set the values of the tensor to the concentration
        zero_tensor.__setitem__(idx, conc_tensor[i - 1][None, :])

    if full_particle:
        full_repr = torch.zeros([2 * image_dim - 1
                                 for _ in range(dims)] + [n_dopants])

        # copy the tensor to the full representation
        full_repr.__setitem__(
            [slice(image_dim - 1, 2 * image_dim - 1) for _ in range(dims)],
            zero_tensor)
        for ops in itertools.product(*[[0, 1] for _ in range(dims)]):
            if sum(ops) == 0:
                continue
            # 0 = no change, 1 = flip along this axis
            idx = [
                slice(image_dim -
                      1) if j == 1 else slice(image_dim, 2 * image_dim - 1)
                for j in ops
            ]

            flip_dims = [i for i, j in enumerate(ops) if j == 1]
            full_repr.__setitem__(
                idx,
                torch.flip(
                    zero_tensor[[slice(1, image_dim) for _ in range(dims)]],
                    flip_dims))
        return full_repr.transpose(0, -1)

    return zero_tensor.transpose(0, -1)

    pass


def to_1d_image(conc_tensor: torch.Tensor,
                radii_without_zeros: torch.Tensor,
                max_image_size: int,
                image_resolution: float,
                full_particle: bool = False):
    """
    This function is faster than the more general to_nd_image function,
    but can be prone to errors.

    Args:
        conc_tensor (torch.Tensor): _description_
        radii_without_zeros (torch.Tensor): _description_
        max_image_size (int): _description_
        image_resolution (float): _description_
        full_particle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    radii = torch.cat((torch.tensor([0]), radii_without_zeros))
    radii = radii.clip(0, max_image_size)

    divisions = ((radii[1:] - radii[:-1]) / image_resolution).int()

    # Compute how to divide the image
    img_tensors = []
    for i, _ in enumerate(divisions):
        img_tensors.append(conc_tensor[i][:, None].expand(-1, divisions[i]))

    image = torch.cat(img_tensors, dim=-1)
    # Zero pad the remaining image up to max_image_size
    extra = (max_image_size - radii[-1].item()) // image_resolution

    if extra > 0:
        image = torch.cat((image, torch.zeros(image.size(0), int(extra))),
                          dim=-1)

    if full_particle:
        # append a reversed version of the image
        return torch.cat((image.flip(-1), conc_tensor[0][:, None], image),
                         dim=-1)
    else:
        return torch.cat((conc_tensor[0][:, None], image), dim=-1)


to_1d_image_torchscript = torch.jit.script(to_1d_image)
