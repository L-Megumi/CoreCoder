import copy

import pytest

from corecoder.evidence_context import EvidenceContextManager
from examples.long_context_acceptance import fixture, recall_facts, score_recall


@pytest.mark.parametrize('variant', ['chain', 'numeric', 'long_config'])
def test_real_observations_survive_actual_group_eviction(tmp_path, variant):
    state, history, expected = fixture(tmp_path / variant, variant)
    original = copy.deepcopy(history)
    manager = EvidenceContextManager(state, 7000)
    manager.reserved_tokens = 2000
    manager.maybe_compress(history)
    assert any(event['groups_collapsed'] for event in manager.events)
    assert manager.memory and len(history) < len(original)
    assert all(fact[1] in manager.memory for fact in expected if fact[0] == 'evidence')
    assert 'exists' in manager.memory and 'missing' in manager.memory
    assert len(state.events) in (3, 4)


def test_recall_scores_extra_invented_facts_and_missing_facts():
    score = score_recall({'evidence_ids': ['real', 'invented']}, {('evidence', 'real'), ('numeric', 'nan')})
    assert score['recall'] == score['precision'] == 0.5
    assert score['extra'] == 1


@pytest.mark.parametrize('value', [None, [], {'config_values': []}, {'evidence_ids': None}])
def test_malformed_model_recall_does_not_crash_scoring(value):
    assert not recall_facts(value)
