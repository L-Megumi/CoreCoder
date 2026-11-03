"""Content-free signatures of observed log evidence at the model gateway."""

import hashlib
import json
import re


def signature(value):
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def evidence_signatures(messages):
    found = set()

    def visit(value):
        if isinstance(value, dict):
            if value.get("classification") == "unverified_hypothesis":
                return
            for key, item in value.items():
                if key == "message" and isinstance(item, str):
                    found.add(signature(item))
                elif isinstance(item, (dict, list)):
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for message in messages:
        if message.get("role") not in ("user", "tool"):
            continue
        content = str(message.get("content", ""))
        for line in content.splitlines():
            if re.match(r"^\s*(ERROR|WARNING|CRITICAL)\b", line):
                found.add(signature(line))
        # Context cards may have an explicit untrusted-data prefix.
        start = content.find("{")
        if start >= 0:
            try:
                visit(json.loads(content[start:]))
            except (ValueError, TypeError):
                pass
    if len(found) > 256:
        return None  # Do not present an incomplete signature set as an exact audit.
    return sorted(found)
