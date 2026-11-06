"""Safe typed permission denials shared by guards and their audit observers.

No path, model arguments, write contents or free-form error messages are accepted
by this receipt. A recoverable denial is still a denied, unexecuted operation.
The trusted controller decides whether one parameter correction is available.
"""

import re
import uuid


REASONS = frozenset({
    "outside_workspace", "public_read_path_mismatch", "write_not_allowed",
    "sensitive_target", "unknown_target", "path_unavailable", "correction_exhausted",
    "policy_disabled", "current_read_required", "source_changed_since_read",
    "source_digest_unavailable", "tool_not_allowed", "guard_rejected",
})
CATEGORIES = frozenset({
    "public_check", "sensitive", "symlink", "outside_workspace", "unknown",
    "protected_source", "workspace_file", "tool",
})
TOOLS = frozenset({"read_file", "write_file", "edit_file", "grep", "glob", "bash", "agent", "unknown"})
_FIELDS = frozenset({"schema_version", "denial_id", "reason_code", "target_category", "action",
                     "tool", "target_sha256", "correction_index", "tool_executed"})
_FEEDBACK = {
    "outside_workspace": "File target is outside the repair project",
    "public_read_path_mismatch": "The public check script is controller-only and cannot be read by this Agent",
    "write_not_allowed": "Unauthorized baseline write",
    "sensitive_target": "The requested target is outside the authorized source scope",
    "unknown_target": "The requested target cannot be safely classified",
    "path_unavailable": "The requested target cannot be safely resolved",
    "correction_exhausted": "The permitted parameter correction has already been used",
    "policy_disabled": "Parameter correction is not enabled for this execution",
    "current_read_required": "Unauthorized baseline write: re-read current candidate source after expert handoff",
    "source_changed_since_read": "Unauthorized baseline write: the candidate source changed since its last observed read",
    "source_digest_unavailable": "Unauthorized baseline write: the current candidate source digest could not be verified",
    "tool_not_allowed": "Unauthorized baseline tool",
    "guard_rejected": "The requested operation was rejected by the controller guard",
}


class CodingGuardRejection(ValueError):
    """Trusted pre-execution decision. Only the auditing wrapper creates a receipt.

    Keep the original unaudited guard messages for existing coding-loop callers.
    Neither the model nor the exception text supplies the audit classification.
    """

    def __init__(self, reason_code):
        if reason_code not in {
            "tool_not_allowed", "outside_workspace", "write_not_allowed", "path_unavailable",
            "current_read_required", "source_changed_since_read", "source_digest_unavailable",
        }:
            raise ValueError("Unknown coding guard decision")
        self.reason_code = reason_code
        legacy = {
            "outside_workspace": "Unauthorized baseline write",
            "source_changed_since_read": _FEEDBACK["current_read_required"],
        }
        super().__init__(legacy.get(reason_code, _FEEDBACK[reason_code]))


def validate_permission_receipt(value):
    """Return a fresh, strictly validated metadata-only denial record."""
    if type(value) is not dict or set(value) != _FIELDS:
        raise ValueError("Permission receipt has unknown or missing fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported permission receipt version")
    if type(value["denial_id"]) is not str or not re.fullmatch(r"denial-[0-9a-f]{32}", value["denial_id"]):
        raise ValueError("Invalid permission denial identity")
    for key, choices in (("reason_code", REASONS), ("target_category", CATEGORIES),
                         ("action", {"stop", "correct_parameters"}), ("tool", TOOLS)):
        if type(value[key]) is not str or value[key] not in choices:
            raise ValueError("Unsupported permission receipt category")
    digest = value["target_sha256"]
    if digest is not None and (type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("Expected an opaque target SHA256")
    if type(value["correction_index"]) is not int or value["correction_index"] not in (0, 1):
        raise ValueError("Invalid bounded correction index")
    if value["tool_executed"] is not False:
        raise ValueError("Permission denial cannot claim a possibly executed operation")
    if value["action"] == "correct_parameters":
        if (value["reason_code"], value["target_category"], value["tool"], value["correction_index"]) != (
                "public_read_path_mismatch", "public_check", "read_file", 1) or digest is None:
            raise ValueError("Correction is limited to a denied trusted public-check read")
    elif value["correction_index"] != 0:
        raise ValueError("A terminal denial cannot consume correction credit")
    return dict(value)


class PermissionDenied(ValueError):
    """Guard-only denial with fixed safe feedback and a persistent identity."""

    def __init__(self, *, reason_code, target_category, tool, action="stop", target_sha256=None,
                 correction_index=0, denial_id=None):
        self._receipt = validate_permission_receipt(dict(schema_version=1,
            denial_id="denial-"+uuid.uuid4().hex if denial_id is None else denial_id,
            reason_code=reason_code, target_category=target_category, tool=tool, action=action,
            target_sha256=target_sha256, correction_index=correction_index, tool_executed=False))
        super().__init__(self.feedback)

    @property
    def receipt(self):
        return dict(self._receipt)

    @property
    def feedback(self):
        if self._receipt["action"] == "correct_parameters":
            return ("File target is outside the repair project: the referenced public check script is controller-only. "
                    "The requested read was not executed. One parameter correction is permitted: "
                    "read the relevant candidate source within the project, then continue the original task. "
                    "Do not retry this check script, broaden paths, or change execution permissions.")
        return _FEEDBACK[self._receipt["reason_code"]] + ". The requested tool was not executed; stop this execution."

    def to_dict(self):
        return self.receipt

    @classmethod
    def from_receipt(cls, value):
        data = validate_permission_receipt(value)
        return cls(**{key: item for key, item in data.items() if key not in ("schema_version", "tool_executed")})


class PermissionAuditError(RuntimeError):
    """A rejected operation could not be safely recorded; continuation is refused."""
