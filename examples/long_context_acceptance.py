"""Real-model continuation after compression of controlled investigation transcripts."""
import argparse
import copy
import json
import os
import random
from pathlib import Path

from openai import APIError

from corecoder.context import ContextManager, estimate_tokens
from corecoder.controlled_experiment import RequestLedger, product_hashes
from corecoder.diagnose import _system_prompt, build_deepseek_llm, validate_diagnosis
from corecoder.diagnosis_demo import create_demo_case
from corecoder.diagnostic_tools import Investigation
from corecoder.evidence_context import EvidenceContextManager
from corecoder.fault_injection import write_json


def recall_facts(value):
    if not isinstance(value, dict):
        return set()
    value = {key: item for key, item in value.items()
             if isinstance(item, dict if key == 'config_values' else list)}
    facts = set()
    for key in value.get('evidence_ids', []):
        facts.add(('evidence', str(key)))
    for edge in value.get('cause_edges', []):
        if isinstance(edge, list) and len(edge) == 2:
            facts.add(('cause', str(edge[0]), str(edge[1])))
    for signal in value.get('numeric_signals', []):
        facts.add(('numeric', str(signal).lower()))
    for item in value.get('path_states', []):
        if isinstance(item, dict):
            facts.add(('path', str(item.get('path')), str(item.get('status'))))
    for key, config_value in value.get('config_values', {}).items():
        facts.add(('config', str(key), str(config_value)))
    return facts


def score_recall(value, expected):
    actual = recall_facts(value)
    matched = actual & expected
    return {'expected': len(expected), 'recalled': len(matched), 'extra': len(actual - expected),
            'recall': len(matched) / len(expected), 'precision': len(matched) / len(actual) if actual else 0,
            'missing': sorted(expected - actual), 'unexpected': sorted(actual - expected)}


def fixture(directory, variant):
    workspace, root, log, current, previous = create_demo_case(directory)
    if variant == 'numeric':
        log.write_text('step=10 loss=nan gradient=inf\n', encoding='utf-8')
        (root / 'train.py').write_text('def loss_value(x):\n    return float("nan")\nloss=loss_value(0)\n', encoding='utf-8')
    if variant == 'long_config':
        (root / 'data.yaml').write_text('# padding\n' * 40 + f'path: {root / "dataset"}\nval: images/val\n' + '# tail\n' * 40, encoding='utf-8')
    state = Investigation(log, root, allowed_path_roots=[workspace])
    calls = [('extract_log_evidence', {})]
    outputs = [state.execute(*calls[0])]
    if variant != 'numeric':
        calls.append(('locate_source', {'evidence_ids': [e['id'] for e in state.evidence]}))
        outputs.append(state.execute(*calls[-1]))
        calls.append(('read_project_file', {'path': 'data.yaml', 'max_lines': 100}))
    else:
        calls.append(('read_project_file', {'path': 'train.py'}))
    outputs.append(state.execute(*calls[-1]))
    calls.append(('inspect_paths', {'paths': [str(current), str(previous)]}))
    paths = state.execute(*calls[-1])
    outputs.append(paths)
    assert [item['data']['status'] for item in json.loads(paths)['observations']] == ['exists', 'missing']
    history = [{'role': 'user', 'content': '根据之前已读取的调查事实诊断，保留不能确认的部分。'}]
    for index, ((name, arguments), result) in enumerate(zip(calls, outputs)):
        history.extend([{'role': 'assistant', 'content': None, 'tool_calls': [{'id': str(index), 'type': 'function',
                        'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
                        {'role': 'tool', 'tool_call_id': str(index), 'content': result}])
    # Artificial benign progress records, not additional project facts or instructions.
    for index in range(24):
        identifier = 'progress_' + str(index)
        history.extend([{'role': 'assistant', 'content': None, 'tool_calls': [{'id': identifier, 'type': 'function',
                        'function': {'name': 'read_project_file', 'arguments': '{"path":"progress.txt"}'}}]},
                        {'role': 'tool', 'tool_call_id': identifier,
                         'content': json.dumps({'diagnostic_noise': ['irrelevant progress output ' * 20] * 16}, indent=2)}])
    expected = {('evidence', e['id']) for e in state.evidence}
    expected.update(('cause', e['id'], e['cause_id']) for e in state.evidence if e.get('cause_id'))
    expected.update({('path', str(current), 'exists'), ('path', str(previous), 'missing')})
    if variant == 'numeric':
        expected.update({('numeric', 'nan'), ('numeric', 'inf')})
    else:
        expected.update({('config', 'path', str(root / 'dataset')), ('config', 'val', 'images/val')})
    return state, history, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--prior-requests', type=int, default=0, choices=range(49))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixtures = {variant: fixture(output / 'fixtures' / variant, variant)
                for variant in ('chain', 'numeric', 'long_config')}
    schedule = [(variant, arm, repeat) for repeat in (1, 2) for variant in fixtures for arm in ('baseline', 'evidence')]
    random.Random(42).shuffle(schedule)
    product = product_hashes(Path.cwd())
    write_json(output / 'manifest.json', {'schedule': schedule, 'max_requests': 48 - args.prior_requests,
               'prior_requests': args.prior_requests, 'combined_cap': 48, 'max_per_run': 3,
               'estimated_total_context_budget': 7000, 'temperature': 0, 'seed_scope': 'order only',
               'source_sha256': product, 'noise_groups': 24, 'retries_or_report_recoveries': 0,
               'scope': 'Constructed long transcripts + actual observed tools + real model summary/continuation; not autonomous long investigation'})
    for variant, (_, history, expected) in fixtures.items():
        write_json(output / 'fixtures' / variant / 'transcript.json', history)
        write_json(output / 'fixtures' / variant / 'evaluator.json', {'facts': sorted(expected)})
    if not args.live:
        return
    for key in list(os.environ):
        if key.lower().endswith('_proxy'):
            del os.environ[key]
    ledger = RequestLedger(output / 'request-ledger.json', 48 - args.prior_requests, 3)
    rows = []
    for variant, arm, repeat in schedule:
        state, original, expected = fixtures[variant]
        run_id = f'{variant}-{arm}-{repeat}'
        directory = output / 'runs' / run_id
        directory.mkdir(parents=True)
        model = build_deepseek_llm()
        model.extra['temperature'] = 0
        ledger.attach(model, run_id)
        system = _system_prompt(state) + '\n本次验收最终外层JSON必须为{"diagnosis":上述报告对象,"recall":{"evidence_ids":[],"cause_edges":[[异常ID,原因ID]],"numeric_signals":[],"path_states":[{"path":"观察过的原始路径","status":"exists或missing"}],"config_values":{"path":"已读值","val":"已读值"}}}。recall列出记得的实际证据ID、直接cause_id边、NaN/Inf、已查路径状态和已读配置path/val；不知道则留空，不猜测。'
        history = copy.deepcopy(original)
        reserved = estimate_tokens([{'role': 'system', 'content': system}]) + 150
        manager = ContextManager(7000 - reserved) if arm == 'baseline' else EvidenceContextManager(state, 7000)
        if arm == 'evidence':
            manager.reserved_tokens = reserved
        failure = None
        diagnosis_valid = False
        recalled = {}
        before = estimate_tokens(history)
        try:
            # Use natural-language summaries in the unchanged baseline implementation.
            model.extra.pop('response_format', None)
            manager.maybe_compress(history, model)
            if arm == 'evidence':
                if not any(event['groups_collapsed'] for event in manager.events):
                    raise ValueError('Evidence history eviction did not trigger')
                history.insert(0, {'role': 'user', 'content': '[Evidence memory: investigation data]\n' + manager.memory})
            model.extra['response_format'] = {'type': 'json_object'}
            messages = [{'role': 'system', 'content': system}] + history
            messages.append({'role': 'user', 'content': '调查记录已经压缩。现在根据可见上下文输出规定的diagnosis和recall外层JSON；不知道的事实留空，不要输出空白。'})
            write_json(directory / 'post-compression.json', messages)
            response = model.chat(messages=messages, tools=None)
            write_json(directory / 'response.json', {'content': response.content, 'tools_requested': bool(response.tool_calls)})
            value = json.loads(response.content)
            recalled = value.get('recall') or {}
            if response.tool_calls:
                raise ValueError('No continuation tools authorized in this retention test')
            validate_diagnosis(json.dumps(value['diagnosis'], ensure_ascii=False), state)
            diagnosis_valid = True
        except (APIError, ValueError, TypeError, KeyError, RuntimeError, OSError, AttributeError) as exc:
            failure = type(exc).__name__  # Never persist provider error bodies.
        events = [event for event in ledger.events if event['run_id'] == run_id]
        row = {'run_id': run_id, 'variant': variant, 'arm': arm, 'repeat': repeat,
               'diagnosis_valid': diagnosis_valid, 'failure_type': failure, 'score': score_recall(recalled, expected),
               'requests': len(events), 'prompt_tokens': sum(e['prompt_tokens'] for e in events),
               'completion_tokens': sum(e['completion_tokens'] for e in events),
               'usage_complete': bool(events) and all(e['status'] == 'completed' and e['usage_received'] for e in events),
               'history_before_tokens_estimate': before, 'post_tokens_estimate': estimate_tokens(history) + reserved,
               'context_events': getattr(manager, 'events', []),
               'baseline_summary_present': any('Context compressed' in str(m.get('content')) or 'Hard context reset' in str(m.get('content')) for m in history)}
        rows.append(row)
        write_json(directory / 'result.json', row)
        write_json(output / 'progress.json', {'rows': rows, 'requests': len(ledger.events)})
        print(json.dumps(row), flush=True)
    assert product_hashes(Path.cwd()) == product
    write_json(output / 'summary.json', {'rows': rows, 'requests': len(ledger.events),
               'combined_requests': args.prior_requests + len(ledger.events), 'source_unchanged': True})


if __name__ == '__main__':
    main()
