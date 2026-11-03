"""Optional machine-scoreable model predictions; benchmark answers are never supplied here."""

from .observation_state import current_observation_ids
from .source_visibility import visible_end_line

CATEGORIES = ("dataset_path", "cuda_oom", "tensor_shape", "numerical_nonfinite", "import_dependency", "unknown")
CAUSES = ("dataset_path_unavailable", "allocation_exceeds_available_memory", "incompatible_channels",
          "incompatible_reshape", "nonfinite_log_zero", "missing_dependency", "unknown")
FINDING_EXAMPLE = {
    "failure_category": "从类别列表选择",
    "cause_code": "从原因代码列表选择；无法确定选unknown",
    "primary_locations": [{"observation_id": "实际源码观察ID", "line_no": 1}],
}


def validate_findings(report, state):
    if report["failure_category"] not in CATEGORIES or report["cause_code"] not in CAUSES:
        raise ValueError("Unknown failure_category or cause_code")
    locations = report["primary_locations"]
    if not isinstance(locations, list) or len(locations) > 5:
        raise ValueError("primary_locations must contain at most 5 locations")
    for location in locations:
        if not isinstance(location, dict) or set(location) != {"observation_id", "line_no"}:
            raise ValueError("Invalid primary location structure")
        identifier = location["observation_id"]
        if not isinstance(identifier, str) or identifier not in state.observations:
            raise ValueError("Unknown primary location observation")
        if identifier not in current_observation_ids(state):
            raise ValueError("Primary location cites a superseded source observation")
        observation = state.observations[identifier]
        data = observation["data"]
        span = data.get("snippet") or data
        line = location["line_no"]
        if (isinstance(line, bool) or not isinstance(line, int) or line < 1
                or observation["kind"] not in ("source_location", "file_excerpt")
                or not data.get("file_sha256") or not span.get("start_line", 0) <= line <= visible_end_line(span)):
            raise ValueError("Primary location must reference an observed source line range")
