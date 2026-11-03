# Autograd detach: independent CPU contract check

The joint-training goal requires gradients and real optimizer updates in both a
4→8 Tanh backbone and an 8→2 head. The faulty training step detaches the features.
Its head can still learn and loss may decrease; neither is a sufficient oracle.

`oracle.py` owns the model, expected trainable parameter names, deterministic
inputs and optimizer. It calls the candidate's actual `training_step` and checks
gradients, updates, parameter identity/flags and optimizer membership. Discovery
uses seed 42; final checks use 142 and 314. `frozen/` is an explicitly head-only
negative control: detach is appropriate for that different business contract.

Keep the oracle and case.json outside the editable candidate directory. The demo
allows only training.py edits and hashes these protected files before and after.
These controls are not an OS sandbox against arbitrary malicious Python.

Run with an existing Torch interpreter (no datasets, GPU or downloads):

```sh
export RESEARCHCODER_EXPERIMENT_PYTHON=/home/example/miniforge3/envs/M2D/bin/python
/home/example/miniforge3/envs/researchcoder/bin/python -B -m examples.pytorch_detach_adapter_demo --output-dir /tmp/new-autograd-demo
```

The output directory/candidate must be new. This uses ScriptedLLM for fixed model
decisions and executes real CPU gradients, edits and checks. It measures pipeline
correctness, **not** LLM diagnostic accuracy.
