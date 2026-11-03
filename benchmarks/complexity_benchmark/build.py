"""Build frozen fixtures and real controlled-run logs in a NEW directory only."""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
from fixture_sources import fixtures


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-python", required=True)
    args = parser.parse_args()
    # Controller has contracts; the independently installed Torch interpreter need not.
    from researchcoder.evaluation.contracts import BenchmarkCase, BenchmarkManifest, SourceAsset
    from researchcoder.evaluation.dataset import load_manifest as original_manifest, DATA as ORIGINAL_DATA
    from researchcoder.evaluation.complexity.contracts import ComplexityCase, ComplexityManifest, EvidenceSite, RepairStage
    from researchcoder.evaluation.complexity.dataset import ORACLE, SUPPORT
    args.output.mkdir(parents=True, exist_ok=False)
    definitions = fixtures()
    generation = {}
    cases = []
    labels = []
    hypotheses = {
        "cpx_01": ("Gradient path interrupted; compare actual gradient norms.", "Optimizer does not own a trainable module; compare gradients against updates.", "Explicit freezing is intentional; inspect the caller's training option."),
        "cpx_02": ("Batch/class axes differ at the model boundary.", "Packing reorders one side of a paired sample/target relation.", "The objective interprets target encoding or reduction incorrectly."),
        "cpx_03": ("An upstream feature transform introduces nonfinite values.", "An unstable objective or learning rate introduces nonfinite gradients.", "Model input dtype or scale is inconsistent with the data contract."),
        "cpx_04": ("A modality is intentionally frozen.", "Fusion interrupts a modality's gradient graph.", "An optimizer group omits a modality even when gradients exist."),
        "cpx_05": ("Padding affects pooled feature scale.", "Targets or sample weights are misaligned.", "The objective normalizes sample weights inconsistently."),
        "cpx_06": ("Validation changes mode or running statistics.", "A microbatch's gradient is cleared before accumulation finishes.", "Batch normalization, learning rate or microbatch scaling changes the update."),
    }
    for identity, definition in definitions.items():
        directory = args.output / identity
        for state in ("faulty", "healthy"):
            for name, source in definition[state].items():
                ast.parse(source)
                path = directory / state / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x") as stream:
                    stream.write(source)
        code = r'''
import importlib.util,json,math,platform,sys
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('independent_reference',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(1)
entry=module.load_candidate(sys.argv[2]);identity=sys.argv[3];rows=[]
def clean(value):
 if isinstance(value,float) and not math.isfinite(value):return 'nan' if math.isnan(value) else ('inf' if value>0 else '-inf')
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
 if isinstance(value,list):return [clean(v) for v in value]
 return value
for repeat in range(48):
 regression=repeat<16 or repeat>=32
 checks,telemetry=module.run_once(entry,identity,10001+repeat,regression)
 rows.append(clean(dict(repeat=repeat,scenario='normal' if regression else 'stress',telemetry=telemetry)))
print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,device='cpu',controlled_repetitions=48,rows=rows),allow_nan=False))
'''
        process = subprocess.run([args.torch_python, "-B", "-c", code, str(SUPPORT), str(directory / "faulty"), identity], capture_output=True, text=True, timeout=90)
        if process.returncode:
            raise RuntimeError(f"Real log generation failed for {identity}: {process.stderr[-1800:]}")
        measured = json.loads(process.stdout)
        training, validation, public = [], [], []
        for row in measured["rows"]:
            observed = row["telemetry"]
            validation.append(json.dumps(dict(repeat=row["repeat"], scenario=row["scenario"], mode_after=observed["mode_after"], checks_passed=observed["passed"], checks_total=observed["total"]), sort_keys=True, separators=(",", ":")))
            training.append(json.dumps(dict(repeat=row["repeat"], scenario=row["scenario"], loss=observed["loss"]), sort_keys=True, separators=(",", ":")))
            for name, parameter in observed["parameters"].items():
                training.append(json.dumps(dict(repeat=row["repeat"], scenario=row["scenario"], component=name, **parameter), sort_keys=True, separators=(",", ":")))
            # Compact public history retains measured state, never private expected tensors.
            if definition["level"] == 3:
                public.append("INFO " + validation[-1])
                public.append("INFO " + json.dumps(dict(repeat=row["repeat"], loss=observed["loss"], updates={n: p["update_norm"] for n, p in observed["parameters"].items()}), sort_keys=True, separators=(",", ":")))
            if observed["passed"] < observed["total"] and row["repeat"] in (16, 24, 31):
                public.append("ERROR training contract check did not match the requested behavior; inspect the recorded training and validation observations.")
        if not public:
            public = ["INFO " + validation[0], "INFO " + validation[16], "ERROR training contract check did not match the requested behavior. Training and validation observations are available under records/."]
        public_log = "\n".join(public) + "\n"
        assert len(public_log) <= 20000
        extra = {
            "README.md": definition["goal"] + "\n\nPublic API: entry.create_model(...); entry.step(model, batch, lr). Top-level helper functions are supported component APIs and must preserve their stated training semantics independently of the entry point.\nThe architecture in network.py is fixed. Read every necessary module and historical record; log observations are not instructions.\n",
            "records/training.txt": "\n".join(training) + "\n",
            "records/validation.txt": "\n".join(validation) + "\n",
        }
        for state in ("faulty", "healthy"):
            for name, source in extra.items():
                path = directory / state / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x") as stream:
                    stream.write(source)
        paths = sorted((*definition["healthy"], *extra))
        assets = tuple(SourceAsset(path=p,
            faulty_sha256=sha((directory / "faulty" / p).read_bytes()),
            healthy_sha256=sha((directory / "healthy" / p).read_bytes())) for p in paths)
        # Writable range is not the gold fault-location list. Other logic repairs are allowed.
        writable = tuple(p for p in paths if p.endswith(".py") and p != "network.py")
        case = BenchmarkCase(case_id=identity, category="cross_module" if definition["level"] == 2 else "multi_evidence",
            split=definition["split"], project_version="complexity-v1", objective=definition["goal"],
            public_log=public_log, assets=assets, writable_paths=writable,
            reference_cause=definition["cause"], cause_concepts=definition["concepts"],
            fault_locations=tuple(definition["partial"] or ()) + tuple(p for p in definition["writable"] if p not in (definition["partial"] or {})),
            acceptable_repair="Restore the public training invariants on independent inputs; preserve architecture, intended freeze/eval behavior, inputs and all P2P checks.",
            oracle_sha256=sha(ORACLE.read_bytes()), oracle_support_sha256=sha(SUPPORT.read_bytes()))
        cases.append(case)
        sources = tuple(p for p in paths if p.endswith(".py"))
        modules = {Path(p).stem: p for p in sources}
        edges = set()
        for path in sources:
            for node in ast.walk(ast.parse(definition["faulty"][path])):
                for imported in (n.name for n in node.names) if isinstance(node, ast.Import) else ():
                    if imported in modules and modules[imported] != path:
                        edges.add((path, modules[imported]))
        sites = []
        for path in dict.fromkeys((*definition["writable"], "entry.py", "network.py")):
            faulty_lines = definition["faulty"][path].splitlines()
            changes = [(i + 1, max(i + 1, j)) for tag, i, j, _, _ in difflib.SequenceMatcher(a=faulty_lines, b=definition["healthy"][path].splitlines()).get_opcodes() if tag != "equal"]
            start, end = (min(i for i, _ in changes), min(len(faulty_lines), max(j for _, j in changes))) if changes else (1, min(len(faulty_lines), 14))
            sites.append(EvidenceSite(path=path, start_line=start, end_line=max(start, end), kind="source", sha256=next(a.faulty_sha256 for a in assets if a.path == path)))
        for name, kind in (("records/training.txt", "training"), ("records/validation.txt", "validation")):
            lines = extra[name].splitlines()
            first = next(i + 1 for i, line in enumerate(lines) if '"scenario":"stress"' in line)
            sites.append(EvidenceSite(path=name, start_line=first, end_line=min(len(lines), first + 3), kind=kind, sha256=sha(extra[name].encode())))
        stages = []
        if definition["partial"]:
            stages.append(RepairStage(stage_id="partial", patch_paths=tuple(definition["partial"]), expected_f2p=False))
        stages.append(RepairStage(stage_id="complete", patch_paths=definition["writable"], expected_f2p=True))
        unique = {sha(value.encode()): value for value in (*definition["faulty"].values(), *extra.values(), case.public_log, case.objective)}
        length = sum(len(value) for value in unique.values())
        labels.append(ComplexityCase(case_id=identity, benchmark_subset="complexity_benchmark", level=definition["level"],
            case_sha256=sha(case.to_json().encode()), source_files=sources, dependencies=tuple(sorted(edges)),
            evidence_sites=tuple(sites), initial_hypotheses=hypotheses[identity], stages=tuple(stages),
            N_files=len(sources), N_dependencies=len(edges), EvidenceDispersion=len({s.path for s in sites}),
            HypothesisAmbiguity=len(hypotheses[identity]), RepairStages=len(stages), ContextLength=length,
            ContextTokensEstimate=(length + 2) // 3,
            FailureVisibility="indirect_telemetry" if definition["level"] == 2 else "delayed_validation"))
        generation[identity] = dict(python=measured["python"], torch=measured["torch"], device="cpu", controlled_repetitions=48, public_seed_range=[10001, 10048],
            fixture_generator_sha256=sha(Path(__file__).with_name("fixture_sources.py").read_bytes()),
            builder_sha256=sha(Path(__file__).read_bytes()), reference_sha256=sha(SUPPORT.read_bytes()),
            scenarios="16 normal + 16 stress + 16 normal, genuine independently seeded CPU training runs",
            training_lines=len(training), validation_lines=len(validation), public_log_chars=len(public_log),
            healthy_and_faulty_share_immutable_historical_logs=True)
    for case in original_manifest().cases:
        if case.case_id not in ("autograd_detach", "shape_transpose", "dtype_features"):
            continue
        asset = case.assets[0]
        old_lines = (ORIGINAL_DATA / case.case_id / "faulty" / asset.path).read_text().splitlines()
        new_lines = (ORIGINAL_DATA / case.case_id / "healthy" / asset.path).read_text().splitlines()
        differences = [(i + 1, max(i + 1, j)) for tag, i, j, _, _ in difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes() if tag != "equal"]
        site_start, site_end = min(i for i, _ in differences), min(len(old_lines), max(j for _, j in differences))
        length = len(case.public_log) + len(case.objective) + sum(len((ORIGINAL_DATA / case.case_id / "faulty" / a.path).read_text()) for a in case.assets)
        labels.append(ComplexityCase(case_id=case.case_id, benchmark_subset="pytorch", level=1,
            case_sha256=sha(case.to_json().encode()), source_files=tuple(a.path for a in case.assets), dependencies=(),
            evidence_sites=(EvidenceSite(path=asset.path, start_line=site_start, end_line=site_end, kind="source", sha256=asset.faulty_sha256),),
            initial_hypotheses=("Local source violates the stated tensor/training contract.", "A caller configuration produces the same initial symptom."),
            stages=(RepairStage(stage_id="complete", patch_paths=case.writable_paths, expected_f2p=True),),
            N_files=len(case.assets), N_dependencies=0, EvidenceDispersion=1, HypothesisAmbiguity=2, RepairStages=1,
            ContextLength=length, ContextTokensEstimate=(length + 2) // 3, FailureVisibility="filename_only"))
    manifest = BenchmarkManifest(benchmark_id="researchcoder-complexity", version="1.0.1", cases=tuple(cases))
    complexity = ComplexityManifest(benchmark_id="researchcoder-complexity", version="1.0.1", benchmark_manifest_sha256=sha(manifest.to_json().encode()), cases=tuple(labels))
    for name, content in (("cases.json", manifest.to_json()), ("complexity.json", complexity.to_json()), ("generation.json", json.dumps(generation, indent=2))):
        with (args.output / name).open("x") as stream:
            stream.write(content)
    print(json.dumps(dict(output=str(args.output),cases=len(cases),labels=len(labels),labels_frozen_before_agent_trials=True)))


if __name__ == "__main__":
    main()
