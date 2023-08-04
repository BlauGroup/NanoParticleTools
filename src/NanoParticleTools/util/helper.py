import warnings


def dopant_specifications_to_concentrations(
        dopant_specification,
        n_constraints,
        possible_elements=['Yb', 'Er', 'Nd'],
        include_zeros=True):
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
