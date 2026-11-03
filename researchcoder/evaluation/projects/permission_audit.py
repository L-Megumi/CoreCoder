"""Validate trusted denial receipts without legitimizing the denied operation."""
from corecoder.permission import validate_permission_receipt
from researchcoder.runtime.permissions import normalized_target_sha256


def permission_records(result, io, spec, revision):
    records = []
    seen = set()
    targets = {normalized_target_sha256(p) for p in spec.public_check_paths} if spec else set()
    expected = {'schema_version', 'record_type', 'job_id', 'task_id', 'attempt_id',
                'policy_sha256', 'workspace_revision', 'receipt'}
    for ref in result.outputs:
        data = io.read_json(ref)
        if not isinstance(data, dict) or data.get('record_type') != 'permission_denial':
            continue
        if (spec is None or set(data) != expected or data['schema_version'] != 1
                or (data['job_id'], data['task_id'], data['attempt_id']) !=
                   (result.job_id, result.task_id, result.attempt_id)
                or ref.job_id != result.job_id or data['policy_sha256'] != spec.sha256
                or data['workspace_revision'] != revision):
            raise ValueError('Untrusted permission denial receipt')
        receipt = validate_permission_receipt(data['receipt'])
        if receipt['denial_id'] in seen:
            raise ValueError('Duplicate permission denial receipt')
        seen.add(receipt['denial_id'])
        if receipt['action'] == 'correct_parameters' and (
                spec.mode != 'bounded-public-read-v1' or receipt['target_sha256'] not in targets):
            raise ValueError('Correction target was not authorized by the controller')
        records.append(dict(reference=ref.to_dict(), **data))
    corrections = sum(r['receipt']['action'] == 'correct_parameters' for r in records)
    if corrections > (spec.remaining_corrections if spec else 0):
        raise ValueError('Task correction allowance exceeded')
    return records


def guard_violations(result, io, records=()):
    receipts = {r['receipt']['denial_id']: r['receipt'] for r in records}
    denied = [dict(denial_id=r['receipt']['denial_id'], reason=r['receipt']['reason_code'])
              for r in records if r['receipt']['action'] == 'stop']
    encountered = set()
    for ref in result.outputs:
        data = io.read_json(ref)
        if not isinstance(data, dict) or 'baseline_hypothesis' not in data:
            continue
        for event in data.get('events', []):
            if event.get('permission_audit_failed'):
                raise ValueError('Permission refusal lacks a trustworthy execution audit')
            receipt = event.get('permission_denial')
            if receipt is not None:
                receipt = validate_permission_receipt(receipt)
                identity = receipt['denial_id']
                if (identity in encountered or receipts.get(identity) != receipt
                        or event.get('status') != 'error' or event.get('arguments') != {}
                        or event.get('tool') != receipt['tool'] or event.get('permission_audit_failed')):
                    raise ValueError('Permission event is not bound to its immutable receipt')
                encountered.add(identity)
                # An exact blocked read is still an error, but not a successful
                # access or a broad exemption from later file/Oracle checks.
                continue
            message = str(event.get('result', ''))
            if event.get('status') == 'error' and any(marker in message for marker in (
                    'Unauthorized baseline', 'outside the repair project', 'Unsupported repair tool')):
                denied.append(dict(sequence=event['sequence'], tool=event['tool'], reason='file_or_tool_policy'))
    corrections = {identity for identity, receipt in receipts.items()
                   if receipt['action'] == 'correct_parameters'}
    if not corrections.issubset(encountered):
        raise ValueError('Correction receipt has no corresponding trusted tool event')
    return denied
