from NanoParticleTools.flows.submission import submit_job
from NanoParticleTools.inputs import SphericalConstraint
from jobflow import JobStore
from maggma.stores.mongolike import MemoryStore


def test_submit_job():
    constraints = [SphericalConstraint(20), SphericalConstraint(30)]
    dopant_specifications = [(0, 0.5, 'Er', 'Y'), (0, 0.25, 'Nd', 'Y'),
                             (0, 0.261, 'Yb', 'Y')]

    docs_store = MemoryStore()
    data_store = MemoryStore()
    job_store = JobStore(docs_store,
                         additional_stores={'trajectories': data_store})
    wf = submit_job(constraints,
                    dopant_specifications,
                    lp=None,
                    store=job_store,
                    add_to_launchpad=False)

    assert len(wf.fws) == 1
    args = wf.fws[0].tasks[0]['job'].function_kwargs
    assert len(args['constraints']) == 2
    assert len(args['dopant_specifications']) == 3
    assert 'nanoparticle_identifier' in args['metadata']

    wf = submit_job(constraints,
                    dopant_specifications,
                    lp=None,
                    store=job_store,
                    add_to_launchpad=False,
                    priority=1000,
                    metadata={'test_metadata': 5})

    args = wf.fws[0].tasks[0]['job'].function_kwargs
    assert 'nanoparticle_identifier' in args['metadata']
    assert args['metadata']['test_metadata'] == 5
