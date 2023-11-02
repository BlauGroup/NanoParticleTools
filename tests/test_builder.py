from NanoParticleTools.builder import (UCNPBuilder, UCNPPopBuilder,
                                       PartialAveragingBuilder,
                                       MultiFidelityAveragingBuilder)
from maggma.stores import MemoryStore

from pathlib import Path
import pytest
import json
from monty.serialization import MontyDecoder

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILE_DIR = MODULE_DIR / 'test_files'


@pytest.fixture
def raw_docs_store():
    store = MemoryStore()
    store.connect()

    # load the documents into the store
    with open(TEST_FILE_DIR / 'npmc_docs/raw_documents.json', 'r') as f:
        results = json.load(f, cls=MontyDecoder)

    store.update(results, key='_id')
    return store


def test_ucnp_builder(raw_docs_store):
    target_store = MemoryStore()
    builder = UCNPBuilder(raw_docs_store, target_store)
    builder.run()

    assert target_store.count() == 2
    assert len(list(builder.prechunk(2))) == 2

    target_store = MemoryStore()
    builder = UCNPBuilder(source=raw_docs_store,
                          target=target_store,
                          docs_filter={'data.n_dopant_sites': 1399})
    builder.run()

    assert target_store.count() == 1
    doc = target_store.query_one()
    assert doc['avg_simulation_length'] == pytest.approx(40078.375)
    assert doc['avg_simulation_time'] == pytest.approx(0.01018022901835798)
    assert set(doc['output'].keys()) == {
        'summary_keys', 'summary', 'energy_spectrum_x', 'energy_spectrum_y',
        'wavelength_spectrum_x', 'wavelength_spectrum_y'
    }


def test_ucnp_pop_builder(raw_docs_store):
    target_store = MemoryStore()
    builder = UCNPPopBuilder(raw_docs_store, target_store)
    builder.run()

    assert target_store.count() == 2
    doc = target_store.query_one()
    assert set(doc['output'].keys()) == {
        'energy_spectrum_x', 'energy_spectrum_y', 'wavelength_spectrum_x',
        'wavelength_spectrum_y', 'avg_total_pop',
        'avg_total_pop_by_constraint', 'avg_5ms_total_pop',
        'avg_8ms_total_pop', 'avg_5ms_total_pop_by_constraint',
        'avg_8ms_total_pop_by_constraint'
    }


def test_partial_averaging_builder(raw_docs_store):
    # Test building 44 averaging
    target_store = MemoryStore()
    builder = PartialAveragingBuilder(n_orderings=4,
                                      n_sims=4,
                                      source=raw_docs_store,
                                      target=target_store)
    builder.run()

    assert target_store.count() == 1
    doc = target_store.query_one()
    doc['num_averaged'] == 16

    # Test building 22 averaging
    target_store = MemoryStore()
    builder = PartialAveragingBuilder(n_orderings=2,
                                      n_sims=2,
                                      source=raw_docs_store,
                                      target=target_store)
    builder.run()

    assert target_store.count() == 1
    doc = target_store.query_one()
    doc['num_averaged'] == 4

    # Test building 14 averaging
    target_store = MemoryStore()
    builder = PartialAveragingBuilder(n_orderings=1,
                                      n_sims=4,
                                      source=raw_docs_store,
                                      target=target_store)
    builder.run()

    assert target_store.count() == 2
    doc = target_store.query_one()
    doc['num_averaged'] == 4

    # Test building 41 averaging
    target_store = MemoryStore()
    builder = PartialAveragingBuilder(n_orderings=4,
                                      n_sims=1,
                                      source=raw_docs_store,
                                      target=target_store)
    builder.run()

    assert target_store.count() == 1
    doc = target_store.query_one()
    doc['num_averaged'] == 4


def test_multi_fidelity_averaging_builder(raw_docs_store):
    target_store = MemoryStore()
    builder = MultiFidelityAveragingBuilder(n_docs_filter=4,
                                            source=raw_docs_store,
                                            target=target_store)
    builder.run()
    assert target_store.count() == 2
    docs = list(target_store.query())
    assert set(docs[0]['output']['energy_spectrum_y'].keys()) == {
        '(1, 1)', '(2, 2)', '(4, 1)', '(1, 4)', '(4, 4)'
    }
    assert set(
        docs[1]['output']['energy_spectrum_y'].keys()) == {'(1, 1)', '(4, 1)'}

    target_store = MemoryStore()
    builder = MultiFidelityAveragingBuilder(n_docs_filter=16,
                                            source=raw_docs_store,
                                            target=target_store)
    builder.run()
    assert target_store.count() == 1
