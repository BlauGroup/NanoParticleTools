from typing import List, Tuple, Optional
import numpy as np
from NanoParticleTools.inputs.nanoparticle import NanoParticleConstraint, SphericalConstraint


class UCNPAugmenter():
    """
    This class defines functionality to augment UCNP/NPMC data.
    Augmentation is achieved by subdividing constraints, keeping the same dopant concentrations
    and output spectra.
    Args:
        random_seed (int, optional): Seed for random number generator.
            Used to ensure reproducibility.. Defaults to 1.
    """

    rng: int

    def __init__(self, random_seed: int = 1):
        self.rng = np.random.default_rng(random_seed)

    def augment_template(self,
                         constraints: List[NanoParticleConstraint],
                         dopant_specifications: List[Tuple[int, float, str,
                                                           str]],
                         n_augments: Optional[int] = 10) -> List[dict]:

        new_templates = []
        for i in range(n_augments):
            new_constraints, new_dopant_specification = self.generate_single_augment(
                constraints, dopant_specifications)
            new_templates.append({
                'constraints':
                new_constraints,
                'dopant_specifications':
                new_dopant_specification
            })
        return new_templates

    def generate_single_augment(
        self,
        constraints: List[NanoParticleConstraint],
        dopant_specifications: List[Tuple[int, float, str, str]],
        max_subdivisions: Optional[int] = 3,
        subdivision_increment=0.1
    ) -> Tuple[List[NanoParticleConstraint], List[Tuple[int, float, str,
                                                        str]]]:
        n_constraints = len(constraints)
        max_subdivisions = 3
        subdivision_increment = 0.1

        # Create a map of the dopant specifications
        dopant_specification_by_layer = {i: [] for i in range(n_constraints)}
        for _tuple in dopant_specifications:
            try:
                dopant_specification_by_layer[_tuple[0]].append(_tuple[1:])
            except KeyError:
                dopant_specification_by_layer[_tuple[0]] = [_tuple[1:]]

        n_constraints_to_divide = self.rng.integers(1, n_constraints + 1)
        constraints_to_subdivide = sorted(
            self.rng.choice(list(range(n_constraints)),
                            n_constraints_to_divide,
                            replace=False))

        new_constraints = []
        new_dopant_specification = []

        constraint_counter = 0
        for i in range(n_constraints):
            if i in constraints_to_subdivide:
                min_radius = 0 if i == 0 else constraints[i - 1].radius
                max_radius = constraints[i].radius

                # pick a number of subdivisions
                n_divisions = self.rng.integers(1, max_subdivisions)
                radii = sorted(
                    self.rng.choice(np.arange(min_radius, max_radius,
                                              subdivision_increment),
                                    n_divisions,
                                    replace=False))

                for r in radii:
                    new_constraints.append(SphericalConstraint(np.round(r, 1)))
                    try:
                        new_dopant_specification.extend([
                            (constraint_counter, *spec)
                            for spec in dopant_specification_by_layer[i]
                        ])
                    except Exception:
                        constraint_counter += 1
                        continue

                    constraint_counter += 1

            # Add the original constraint back to the list
            new_constraints.append(constraints[i])
            new_dopant_specification.extend([
                (constraint_counter, *spec)
                for spec in dopant_specification_by_layer[i]
            ])

            constraint_counter += 1
        return new_constraints, new_dopant_specification
