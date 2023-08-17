from NanoParticleTools.builder import (UCNPBuilder, UCNPPopBuilder,
                                       PartialAveragingBuilder,
                                       MultiFidelityAveragingBuilder)
from maggma.stores import MemoryStore

import pytest


@pytest.fixture
def raw_docs_store():
    store = MemoryStore()
    # TODO: add documents to this memory store
    return store


@pytest.mark.skip('Not implemented yet, need to add documents')
def test_ucnp_builder(raw_docs_store):
    target_store = MemoryStore()

    builder = UCNPBuilder(raw_docs_store, target_store)

    pass


@pytest.mark.skip('Not implemented yet, need to add documents')
def test_ucnp_pop_builder():
    pass


@pytest.mark.skip('Not implemented yet, need to add documents')
def test_partial_averaging_builder():
    pass


@pytest.mark.skip('Not implemented yet, need to add documents')
def test_multi_fidelity_averaging_builder():
    pass
