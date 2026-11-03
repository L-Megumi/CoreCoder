"""Versioned local project trials; network use requires an explicit live opt-in."""
import argparse
import json
from pathlib import Path

from researchcoder.contracts import _unique
from researchcoder.runtime.protocol import ModelSpec, ModelRequestOptions
from .contracts import ProjectRunInput
from .runner import ProjectEvaluationRunner, ProjectPolicy, recompute
from .verification import ProjectCheck
from .routing import ExpertPolicy
from .handoff import HandoffPolicy
from .public_execution import PublicExecutionPolicy
from researchcoder.context.policy import ContextPolicy


def load_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique)
    required = {"schema_version", "task", "source_root", "public_check", "f2p_check", "p2p_check", "model", "policy"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"feedback_models", "model_options", "expert_policy", "expert_models", "permission_recovery_mode", "handoff_policy", "writer_context_policy", "public_execution_policy"}:
        raise ValueError("Unknown or missing trusted project config fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported project config version")
    if not isinstance(value["source_root"], str) or not Path(value["source_root"]).is_absolute():
        raise ValueError("Absolute trusted source root required")
    if value.get('permission_recovery_mode', 'off') not in ('off', 'bounded-public-read-v1'):
        raise ValueError('Unknown permission recovery strategy')
    expert_models = value.get("expert_models", {})
    if not isinstance(expert_models, dict) or set(expert_models) - {"code_analysis", "log_analysis"}:
        raise ValueError("Known trusted expert model fixture mapping required")
    return dict(task=ProjectRunInput.from_dict(value["task"]), source=value["source_root"],
        public_check=ProjectCheck.from_dict(value["public_check"]), f2p_check=ProjectCheck.from_dict(value["f2p_check"]),
        p2p_check=ProjectCheck.from_dict(value["p2p_check"]), model=ModelSpec.from_dict(value["model"]),
        policy=ProjectPolicy.from_dict(value["policy"]),
        feedback_models=tuple(ModelSpec.from_dict(m) for m in value.get("feedback_models", ())),
        model_options=ModelRequestOptions.from_dict(value["model_options"]) if value.get("model_options") else None,
        expert_policy=ExpertPolicy.from_dict(value["expert_policy"]) if "expert_policy" in value else ExpertPolicy(),
        expert_models={role: ModelSpec.from_dict(model) for role, model in expert_models.items()},
        handoff_policy=HandoffPolicy.from_dict(value['handoff_policy']) if value.get('handoff_policy') else None,
        public_execution_policy=PublicExecutionPolicy.from_dict(value['public_execution_policy']) if value.get('public_execution_policy') else None,
        writer_context_policy=ContextPolicy.from_dict(value['writer_context_policy']) if value.get('writer_context_policy') else None,
        permission_recovery_mode=value.get('permission_recovery_mode', 'off'))


def main(argv=None):
    parser = argparse.ArgumentParser(description="ResearchCoder real repository Single-Agent trials")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Trusted local project/check/model configuration JSON")
    group.add_argument("--recompute", help="Recompute an archived trial without model execution")
    parser.add_argument("--output", help="New independent trial directory; must not exist")
    parser.add_argument("--allow-live", action="store_true", help="Explicitly authorize the configured live model scope")
    args = parser.parse_args(argv)
    if args.config and not args.output:
        parser.error("--config requires --output")
    if args.recompute and (args.output or args.allow_live):
        parser.error("Recompute does not accept execution options")
    try:
        result = recompute(args.recompute) if args.recompute else ProjectEvaluationRunner().run(
            args.output, **load_config(args.config), allow_live=args.allow_live)
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        print(json.dumps(dict(status="configuration_error", error_type=type(exc).__name__)))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if args.recompute or result["end_to_end_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
