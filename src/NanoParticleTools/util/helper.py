import warnings

from typing import List, Tuple, Dict


def dopant_specifications_to_concentrations(
        dopant_specification: List[Tuple[int, float, str, str]],
        n_constraints: int,
        possible_elements: List[str] = ['Yb', 'Er', 'Nd'],
        include_zeros: bool = True) -> List[Dict]:
    """
    A helper function to convert from the dopant specifications, which are typically
    used to specify inputs to the NPMC simulator into the dopant concentrations which
    are the output concentrations of the NPMC input generation. This output is commonly
    used by the machine learning models.

    Args:
        dopant_specification: A list of tuples of the form of
            (layer_idx, concentration, element, host_element_to_replace)
        n_constraints: The number of control volumes in the nanoparticle
        possible_elements: The possible elements in the nanoparticle
        include_zeros: Whether to include elements with zero concentration in the output
    """
    if include_zeros:
        empty_concentrations = [{el: 0
                                 for el in possible_elements}
                                for _ in range(n_constraints)]
    else:
        empty_concentrations = [{} for _ in range(n_constraints)]

    for layer_idx, conc, el, _ in dopant_specification:
        try:
            empty_concentrations[layer_idx][el] = conc
        except KeyError:
            warnings.warn(
                'requested element not in possible elements, skipping')

    return empty_concentrations


def dopant_concentration_to_specifications(
        dopant_concentration: List[Dict],
        include_zeros: bool = True) -> List[Tuple[int, float, str, str]]:
    """
    A helper function to convert from the dopant concentrations,
    which are the output concentrations of the NPMC input generation,
    into the dopant specifications which are used as inputs to the
    input generation.

    Args:
        dopant_concentration: A list of dictionaries of the form of
            [{element1: concentration, element2: concentration, ...}],
            where each dictionary represents a layer of the nanoparticle
        include_zeros: Whether to include elements with zero concentration in the output
    """
    dopant_specifications = []
    for i, _d in enumerate(dopant_concentration):
        for el, conc in _d.items():
            if conc == 0 and not include_zeros:
                continue
            dopant_specifications.append((i, conc, el, 'Y'))
    return dopant_specifications
