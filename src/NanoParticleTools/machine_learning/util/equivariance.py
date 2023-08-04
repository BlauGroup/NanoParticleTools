from NanoParticleTools.machine_learning.data.processors import DataProcessor
from NanoParticleTools.inputs.nanoparticle import SphericalConstraint
from NanoParticleTools.util.helper import dopant_specifications_to_concentrations
from torch_geometric.data.data import Data
import torch

from typing import Callable, List, Dict, Tuple


def get_doc(constraints: List[SphericalConstraint],
            dopant_concentration: List[Dict],
            dopant_specifications: List[Tuple] = None):
    _d = {
        'input': {
            'constraints': constraints,
        },
        'dopant_concentration': dopant_concentration
    }
    if dopant_specifications:
        _d['input']['dopant_specifications'] = dopant_specifications
    return


class bcolors:
    GREEN = '\u001b[32m'
    RED = '\u001b[31m'
    ENDC = '\033[0m'


def vectors_match(v1: torch.Tensor,
                  v2: torch.Tensor,
                  atol: float = 1e-8,
                  rtol: float = 1e-5,
                  print_match: bool = True):

    match = torch.allclose(v1, v2.sum(1)[:, None, :], rtol, atol)
    if match:
        if print_match:
            print(f"{bcolors.GREEN}The vector sums are equal{bcolors.ENDC}")
        return True
    else:
        if print_match:
            print(f"{bcolors.RED}The vector sums are not equal{bcolors.ENDC}")
        return False


def create_additive_vectors(n_vecs: int = 4, dim: int = 8):
    points = torch.zeros(n_vecs, 2, 2, dim)
    points[:, 0, 0, :] = (torch.rand((n_vecs, dim)) - 0.5) * 10
    points[:, 0, 1, :] = (torch.rand((n_vecs, dim)) - 0.5) * 10
    points[:, 1, 0, :] = points[:, 0, 1, :]
    points[:, 1, 1, :] = (torch.rand((n_vecs, dim)) - 0.5) * 10
    vecs = points[..., 1, :] - points[..., 0, :]
    sum_vecs = vecs.sum(1).unsqueeze(1)
    return vecs, sum_vecs


def check_vec_scalar_transform(module: Callable,
                               n_vecs: int = 4,
                               dim: int = 8,
                               print_match: bool = True,
                               atol: float = 1e-5,
                               **kwargs):
    vecs, sum_vecs = create_additive_vectors(n_vecs, dim)
    _module = module(vecs.size(-1), **kwargs)
    v1 = _module(sum_vecs)
    v2 = _module(vecs)
    assert vectors_match(v1, v2, print_match=print_match, atol=atol), (
        'The vector sums do not '
        'match, the module is not equivaraint with respect to vector addition')
    return True


def check_vec_vec_transform(module, print_match=True, **kwargs):
    vecs1, vecs1_sum = create_additive_vectors()
    vecs2, vecs2_sum = create_additive_vectors()

    _module = module(vecs1.size(-1), vecs2.size(-1), **kwargs)

    # Follow the paths
    i, j = torch.cartesian_prod(torch.arange(vecs1.size(0)),
                                torch.arange(vecs2.size(0))).T
    out = _module(vecs1[i], vecs2[j])
    out_sum = _module(vecs1_sum, vecs2_sum)
    assert vectors_match(out_sum, out, print_match=print_match), (
        "Module is not additive equivariant to vector-vector multiplication")
    return True


def check_subdivision_equivariance(model: torch.nn.Module,
                                   feature_processor: DataProcessor,
                                   include_zeros: bool = False,
                                   atol=1e-8,
                                   rtol=1e-5):
    # First test case
    constraints = [SphericalConstraint(30)]
    dopant_specs = [(0, 0.5, 'Yb', 'Y')]
    dopant_concentration = dopant_specifications_to_concentrations(
        dopant_specs,
        len(constraints),
        feature_processor.possible_elements,
        include_zeros=include_zeros)
    data = Data(**feature_processor.process_doc(
        get_doc(constraints, dopant_concentration)))
    rep = model(**data.to_dict())

    constraints = [SphericalConstraint(10), SphericalConstraint(30)]
    dopant_specs = [(0, 0.5, 'Yb', 'Y'), (1, 0.5, 'Yb', 'Y')]
    dopant_concentration = dopant_specifications_to_concentrations(
        dopant_specs,
        len(constraints),
        feature_processor.possible_elements,
        include_zeros=include_zeros)
    data = Data(**feature_processor.process_doc(
        get_doc(constraints, dopant_concentration)))
    subdivided_rep = model(**data.to_dict())

    assert torch.allclose(
        rep, subdivided_rep,
        atol=atol), ('The representations do not match, the '
                     'model is not equivariant to subdivision')

    constraints = [SphericalConstraint(10), SphericalConstraint(30)]
    dopant_specs = [(0, 0.5, 'Yb', 'Y'), (1, 0.5, 'Yb', 'Y')]
    dopant_concentration = dopant_specifications_to_concentrations(
        dopant_specs,
        len(constraints),
        feature_processor.possible_elements,
        include_zeros=include_zeros)
    data = Data(**feature_processor.process_doc(
        get_doc(constraints, dopant_concentration)))
    subdivided_rep = model(**data.to_dict())

    assert torch.allclose(
        rep, subdivided_rep,
        atol=atol), ('The representations do not match, the '
                     'model is not equivariant to subdivision')

    # Second test case with more constraints and dopants
    constraints = [SphericalConstraint(30), SphericalConstraint(50)]
    dopant_specs = [(0, 0.345, 'Yb', 'Y'), (0, 0.15, 'Er', 'Y'),
                    (1, 0.2, 'Nd', 'Y')]
    dopant_concentration = dopant_specifications_to_concentrations(
        dopant_specs,
        len(constraints),
        feature_processor.possible_elements,
        include_zeros=include_zeros)
    data = Data(**feature_processor.process_doc(
        get_doc(constraints, dopant_concentration)))
    rep = model(**data.to_dict())

    # Subdivide second constraint
    constraints = [
        SphericalConstraint(30),
        SphericalConstraint(40),
        SphericalConstraint(50)
    ]
    dopant_specs = [(0, 0.345, 'Yb', 'Y'), (0, 0.15, 'Er', 'Y'),
                    (1, 0.2, 'Nd', 'Y'), (2, 0.2, 'Nd', 'Y')]
    dopant_concentration = dopant_specifications_to_concentrations(
        dopant_specs,
        len(constraints),
        feature_processor.possible_elements,
        include_zeros=include_zeros)
    data = Data(**feature_processor.process_doc(
        get_doc(constraints, dopant_concentration)))
    subdivided_rep = model(**data.to_dict())

    assert torch.allclose(
        rep, subdivided_rep,
        atol=atol), ('The representations do not match, the '
                     'model is not equivariant to subdivision')

    # Subdivide first constraints
    constraints = [
        SphericalConstraint(8.02),
        SphericalConstraint(30),
        SphericalConstraint(50)
    ]
    dopant_specs = [(0, 0.345, 'Yb', 'Y'), (0, 0.15, 'Er', 'Y'),
                    (1, 0.345, 'Yb', 'Y'), (1, 0.15, 'Er', 'Y'),
                    (2, 0.2, 'Nd', 'Y')]
    dopant_concentration = dopant_specifications_to_concentrations(
        dopant_specs,
        len(constraints),
        feature_processor.possible_elements,
        include_zeros=include_zeros)
    data = Data(**feature_processor.process_doc(
        get_doc(constraints, dopant_concentration)))
    subdivided_rep = model(**data.to_dict())

    assert torch.allclose(
        rep, subdivided_rep,
        atol=atol), ('The representations do not match, the '
                     'model is not equivariant to subdivision')

    # Subdivide both constraints
    constraints = [
        SphericalConstraint(15.9),
        SphericalConstraint(30),
        SphericalConstraint(40),
        SphericalConstraint(50)
    ]
    dopant_specs = [(0, 0.345, 'Yb', 'Y'), (0, 0.15, 'Er', 'Y'),
                    (1, 0.345, 'Yb', 'Y'), (1, 0.15, 'Er', 'Y'),
                    (2, 0.2, 'Nd', 'Y'), (3, 0.2, 'Nd', 'Y')]
    dopant_concentration = dopant_specifications_to_concentrations(
        dopant_specs,
        len(constraints),
        feature_processor.possible_elements,
        include_zeros=include_zeros)
    data = Data(**feature_processor.process_doc(
        get_doc(constraints, dopant_concentration)))
    subdivided_rep = model(**data.to_dict())

    assert torch.allclose(
        rep, subdivided_rep,
        atol=atol), ('The representations do not match, the '
                     'model is not equivariant to subdivision')

    return True
