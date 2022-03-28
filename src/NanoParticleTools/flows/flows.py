from typing import Optional, Sequence, Tuple

from NanoParticleTools.inputs.nanoparticle import NanoParticleConstraint
from fireworks import Workflow
from jobflow import Flow
from jobflow.managers.fireworks import flow_to_workflow

from NanoParticleTools.flows.jobs import write_inputs, run_npmc, run_analysis


def get_npmc_flow(constraints: Sequence[NanoParticleConstraint],
                  dopant_specifications: Sequence[Tuple[int, float, str, str]],
                  seed: Optional[int] = 0,
                  output_dir: Optional[str] = '.',
                  spectral_kinetics_args={},
                  **kwargs) -> Flow:
    input_job = write_inputs(constraints=constraints,
                             dopant_specifications=dopant_specifications,
                             seed=seed,
                             output_dir=output_dir,
                             **spectral_kinetics_args)
    npmc_job = run_npmc(input_job.output, **kwargs)
    analysis_job = run_analysis(npmc_job.output)

    flow = Flow([input_job, npmc_job, analysis_job],
                output=analysis_job.output)
    return flow


def get_npmc_workflow(constraints: Sequence[NanoParticleConstraint],
                      dopant_specifications: Sequence[Tuple[int, float, str, str]],
                      seed: int,
                      output_dir: Optional[str] = '.') -> Workflow:
    flow = get_npmc_flow(constraints, dopant_specifications, seed, output_dir)
    return flow_to_workflow(flow)
