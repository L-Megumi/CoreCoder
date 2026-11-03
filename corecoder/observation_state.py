"""Read-free selection of the latest known file versions and literal path states."""

import os
from pathlib import Path

FILE_KINDS = ("source_location", "config_file", "file_excerpt", "source_candidate")


def usable_current_observation(observation):
    """The original diagnosis evidence predicate, shared without relaxing it.

    A caller must also check current_observation_ids. Search hits remain leads,
    even when their source hash is current; this function performs no file IO.
    """
    data = observation["data"]
    span = data.get("snippet") or data
    return bool(
        observation["kind"] == "path_status" and data.get("status") in ("exists", "missing")
        or observation["kind"] in ("source_location", "config_file", "file_excerpt")
        and data.get("file_sha256") and isinstance(span.get("content"), str)
    )


def has_usable_current_observation(state):
    current = current_observation_ids(state)
    return any(usable_current_observation(item)
               for key, item in state.observations.items() if key in current)


def has_current_source_candidate(state):
    """Whether a real current search observation can guide another tool read.

    No path from a draft report is considered, and this grants no read authority.
    The normal investigation tool still checks roots, paths and source versions.
    """
    current = current_observation_ids(state)
    return any(key in current and item["kind"] == "source_candidate"
               and item["data"].get("file_sha256")
               and isinstance(item["data"].get("resolved_path"), str)
               for key, item in state.observations.items())


def _path_key(state, item):
    data = item["data"]
    reference = data.get("resolved_path") or data.get("requested_path")
    if not isinstance(reference, str) or not reference:
        return None
    path = Path(reference)
    if not path.is_absolute() and hasattr(state, "root"):
        path = state.root / path
    # No resolve/stat: historical observations must not trigger hidden reads.
    return os.path.normcase(os.path.normpath(str(path)))


def current_observation_ids(state):
    """Keep old records for audit; only expose the latest observed version as current.

    Same-hash pages remain compatible. A newer failed read supersedes old
    contents, and a newer path check reporting missing supersedes an older read.
    No claim is made about changes after the last actual tool observation.
    """
    records = list(state.observations.values())
    files, paths = {}, {}
    for index, item in enumerate(records):
        key = _path_key(state, item)
        if key is not None:
            if item["kind"] in FILE_KINDS:
                files[key] = (index, item)
            elif item["kind"] == "path_status":
                paths[key] = (index, item)
    current = set()
    for item in records:
        key = _path_key(state, item)
        if key in files and item["kind"] in FILE_KINDS:
            latest_index, latest = files[key]
            digest = item["data"].get("file_sha256")
            latest_digest = latest["data"].get("file_sha256")
            if key in paths and paths[key][0] > latest_index and paths[key][1]["data"].get("status") == "missing":
                continue
            if digest and digest == latest_digest or not digest and item["id"] == latest["id"]:
                current.add(item["id"])
        elif key in paths and item["kind"] == "path_status":
            if item["id"] == paths[key][1]["id"]:
                path_index = paths[key][0]
                newer_file = files.get(key)
                if newer_file and newer_file[0] > path_index:
                    data = newer_file[1]["data"]
                    status = item["data"].get("status")
                    if status == "missing" and data.get("file_sha256") or status == "exists" and data.get("status") == "missing":
                        continue
                current.add(item["id"])
        else:
            current.add(item["id"])
    # A static relationship depends on both the caller/bindings and its target snapshots.
    for item in records:
        if item["kind"] == "symbol_resolution" and any(
            identifier not in current for identifier in item["data"].get("dependency_observation_ids", [])
        ):
            current.discard(item["id"])
    return current
