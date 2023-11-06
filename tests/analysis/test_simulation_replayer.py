from NanoParticleTools.analysis.simulation_replayer import SimulationReplayer
from pymatgen.core import Composition, DummySpecies, Element
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILE_DIR = MODULE_DIR / '..' / 'test_files'


@pytest.fixture()
def sim_replayer():
    replayer = SimulationReplayer(trajectories_db_file=TEST_FILE_DIR /
                                  'simulation_output/initial_state.sqlite',
                                  npmc_input_file=TEST_FILE_DIR /
                                  'simulation_output/npmc_input.json')
    return replayer


def test_sim_replayer_init(sim_replayer):
    assert len(sim_replayer.initial_states) == 1904
    assert len(sim_replayer.sites) == 1904

    sim_replayer.npmc_input.nanoparticle.generate()

    # yapf: disable
    assert sim_replayer.npmc_input.nanoparticle.composition == Composition({
        Element('Na'): 1238,
        Element('Y'): 1808,
        Element('Er'): 1186,
        Element('Yb'): 285,
        DummySpecies('Xsurfacesix'): 433
    })
    # yapf: enable


def test_sim_replayer(sim_replayer):
    docs = sim_replayer.generate_docs()

    assert len(docs) == 4
    assert [doc['trajectory_doc']['dopant_seed']
            for doc in docs] == [57837, 57837, 57837, 57837]
    assert set([doc['trajectory_doc']['simulation_seed']
                for doc in docs]) == set([59181, 59182, 59183, 59184])
    assert set([doc['trajectory_doc']['simulation_time']
                for doc in docs]) == set([
                    0.010000072318936497, 0.010000078244421477,
                    0.010000049835387275, 0.01000011131860255
                ])
    assert docs[0]['trajectory_doc']['total_n_levels'] == 43
    assert docs[0]['trajectory_doc']['formula_by_constraint'] == [
        '', 'Yb285Er1186', 'Xsurfacesix433'
    ]
    assert len(docs[0]['trajectory_doc']['output']['summary_keys']) == 14
    assert len(docs[0]['trajectory_doc']['output']['summary']) == 107
    assert len(docs[0]['trajectory_doc']['output']['x_populations']) == 1002
    assert len(
        docs[0]['trajectory_doc']['output']['y_overall_populations']) == 1002
    assert len(
        docs[0]['trajectory_doc']['output']['y_overall_populations'][0]) == 43
    assert len(
        docs[0]['trajectory_doc']['output']['y_constraint_populations']) == 3
    assert len(docs[0]['trajectory_doc']['output']['y_constraint_populations']
               [0]) == 1002
    assert len(docs[0]['trajectory_doc']['output']['final_states']) == 1904
