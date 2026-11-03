"""Public behavioral reproduction; no reference repair is provided."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import torch


def load_sam(project):
    path = (Path(project).resolve() / 'sam.py').resolve()
    if not path.is_file() or not path.is_relative_to(Path(project).resolve()):
        raise ValueError('Invalid candidate sam.py')
    spec = importlib.util.spec_from_file_location('candidate_sam_public', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path
    return module.SAM, str(path)


def train_step(parameter, optimizer):
    optimizer.zero_grad()
    (0.5 * parameter.square().sum()).backward()
    optimizer.first_step(zero_grad=True)
    (0.5 * parameter.square().sum()).backward()
    optimizer.second_step(zero_grad=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    SAM, origin = load_sam(args.project)
    uninterrupted = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = SAM([uninterrupted], torch.optim.SGD, lr=0.1, momentum=0.0, rho=0.05)
    train_step(uninterrupted, optimizer)
    checkpoint = copy.deepcopy(optimizer.state_dict())
    resumed = torch.nn.Parameter(uninterrupted.detach().clone())
    resumed_optimizer = SAM([resumed], torch.optim.SGD, lr=0.5, momentum=0.0, rho=0.05)
    resumed_optimizer.load_state_dict(checkpoint)
    train_step(uninterrupted, optimizer)
    train_step(resumed, resumed_optimizer)
    result = dict(origin=origin, original_lr=0.1, constructor_lr=0.5,
                  reported_restored_lr=resumed_optimizer.param_groups[0]['lr'],
                  uninterrupted=uninterrupted.detach().tolist(), resumed=resumed.detach().tolist())
    print(json.dumps(result, sort_keys=True))
    assert torch.allclose(uninterrupted, resumed, atol=1e-12, rtol=1e-12), (
        'Checkpoint resume changes the next SAM/SGD update despite identical weights, gradients and restored learning rate'
    )


if __name__ == '__main__':
    main()
