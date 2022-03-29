from typing import Optional, Sequence, Tuple, Union

from jobflow import Flow

from NanoParticleTools.flows.jobs import npmc_job
from NanoParticleTools.inputs.nanoparticle import NanoParticleConstraint


def get_npmc_flow(constraints: Sequence[NanoParticleConstraint],
                  dopant_specifications: Sequence[Tuple[int, float, str, str]],
                  doping_seed: Optional[int] = 0,
                  output_dir: Optional[str] = '.',
                  initial_states: Optional[Union[Sequence[int], None]] = None,
                  spectral_kinetics_args={},
                  npmc_args={}) -> Flow:
    """
    Convenience Constructor to construct a npmc job
    """
    # Create a npmc job
    job = npmc_job(constraints=constraints,
                   dopant_specifications=dopant_specifications,
                   doping_seed=doping_seed,
                   output_dir=output_dir,
                   initial_states=initial_states,
                   spectral_kinetics_args=spectral_kinetics_args,
                   npmc_args=npmc_args)

    # Add job to a flow
    flow = Flow([job],
                output=job.output,
                name='NPMC Simulation')
    return flow
