from typing import Sequence, Tuple, Dict

from jobflow import Flow

from NanoParticleTools.flows.jobs import npmc_job
from NanoParticleTools.inputs import NanoParticleConstraint


def get_npmc_flow(constraints: Sequence[NanoParticleConstraint],
                  dopant_specifications: Sequence[Tuple[int, float, str, str]],
                  doping_seed: int = 0,
                  output_dir: str = '.',
                  initial_states: Sequence[int] | None = None,
                  spectral_kinetics_args: Dict = {},
                  initial_state_db_args: Dict = {},
                  npmc_args: Dict = {},
                  override: bool = False,
                  metadata: Dict = {},
                  population_record_interval: float = 1e-5) -> Flow:
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
                   initial_state_db_args=initial_state_db_args,
                   npmc_args=npmc_args,
                   override=override,
                   metadata=metadata,
                   population_record_interval=population_record_interval)

    # Add job to a flow
    flow = Flow([job],
                output=job.output,
                name='NPMC Simulation')
    return flow
