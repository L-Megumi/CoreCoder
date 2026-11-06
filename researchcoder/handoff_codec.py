"""Lossless wire representation of repeated upstream evidence provenance."""

INDEXED_VIEW = "indexed-refs-v2"
_SHARED_REFERENCE_FIELDS = frozenset((
    "job_id", "producer_task_id", "producer_attempt_id", "workspace_revision", "artifact",
))


def index_references(observations):
    """Factor identical producer receipts out of the wire format without dropping any field."""
    table = []
    for observation in observations:
        ref = observation["reference"]
        group = {key: ref.pop(key) for key in sorted(_SHARED_REFERENCE_FIELDS)}
        if group not in table:
            table.append(group)
        observation["producer_index"] = table.index(group)
    return table


def expand_indexed_references(observations, table):
    """Decode only. The consumer must still check the expanded refs against immutable evidence."""
    if not isinstance(table, list) or not 1 <= len(table) <= len(observations):
        raise ValueError("Invalid indexed producer table")
    if any(not isinstance(group, dict) or set(group) != _SHARED_REFERENCE_FIELDS for group in table):
        raise ValueError("Invalid indexed producer receipt")
    used = set()
    expanded = []
    for observation in observations:
        index = observation.get("producer_index")
        local = observation.get("reference")
        if (type(index) is not int or not 0 <= index < len(table) or not isinstance(local, dict)
                or _SHARED_REFERENCE_FIELDS.intersection(local)):
            raise ValueError("Invalid indexed observation reference")
        used.add(index)
        expanded.append(dict(observation, reference={**local, **table[index]}))
    if len(used) != len(table):
        raise ValueError("Unreferenced producer receipt")
    return expanded
