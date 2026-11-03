"""Opt-in local effective-request artifacts, never sent to observability services."""

import copy
import hashlib
import json
import re


class RequestRecorder:
    def __init__(self, artifacts, trace, profile, secrets=()):
        self.artifacts, self.trace = artifacts, trace
        self.profile = copy.deepcopy(profile)
        self.secrets = tuple(v for v in secrets if isinstance(v, str) and v)
        self.records = []

    def capture(self, messages, tools, *, parent_id, before_context_messages=None):
        # SDK clients, headers, process environments and credentials are not fields.
        raw = json.dumps(dict(messages=messages, before_context_messages=before_context_messages,
                             tools=tools or [], profile=self.profile), ensure_ascii=False, allow_nan=False)
        sanitized = raw
        for secret in self.secrets:
            sanitized = sanitized.replace(secret, "[REDACTED_CREDENTIAL]")
        sanitized = re.sub(r"(?:sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._~+/=\-]+)", "[REDACTED_CREDENTIAL]", sanitized)
        value = dict(schema_version=1, job_id=self.trace.job_id, task_id=self.trace.task_id,
            attempt_id=self.trace.attempt_id, ordinal=len(self.records) + 1,
            request=json.loads(sanitized), redacted=sanitized != raw,
            request_sha256=hashlib.sha256(sanitized.encode()).hexdigest(),
            semantics="Effective gateway input before SDK transport; transport retries are linked separately in Trace")
        ref = self.artifacts.put_json(self.trace.job_id, value)
        self.records.append(ref)
        self.trace.emit("model.request.recorded", parent_id=parent_id, artifact_id=ref.artifact_id)

    def manifest(self):
        return self.artifacts.put_json(self.trace.job_id, dict(schema_version=1,
            job_id=self.trace.job_id, task_id=self.trace.task_id, attempt_id=self.trace.attempt_id,
            request_records=[ref.to_dict() for ref in self.records]))
