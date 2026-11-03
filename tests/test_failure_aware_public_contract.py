import json,os
from pathlib import Path
import pytest
from corecoder.workflow import run_verification
from researchcoder.verification import public_training


def project(root,mode='all',freeze_bypass=False):
    root.mkdir()
    (root/'entry.py').write_text('''import torch
class Model(torch.nn.Module):
 def __init__(self, frozen=False):
  super().__init__(); self.backbone=torch.nn.Linear(2,2); self.head=torch.nn.Linear(2,1)
  self.backbone.requires_grad_(not frozen)
def create_model(freeze_backbone=False):
 return Model(freeze_backbone)
'''+('''original=create_model
def create_model(freeze_backbone=False):
 model=original(True)
 return model
''' if freeze_bypass else ''))
    params={'all':'model.parameters()','filtered':'(p for p in model.parameters() if p.requires_grad)','omitted':'model.head.parameters()','duplicate':'list(model.parameters()) + list(model.parameters())'}[mode]
    (root/'settings.py').write_text('import torch\ndef optimizer(model,lr):\n return torch.optim.SGD('+params+',lr=lr)\n')
    return root


def verify(root,baseline=None,**options):
    argv=[os.environ['RESEARCHCODER_EXPERIMENT_PYTHON'],'-B',public_training.__file__,'--project',str(root)]
    if baseline:argv+=['--baseline',str(baseline)]
    for name,value in options.items():argv+=['--'+name,value]
    result=run_verification(argv,root,30)
    return result,json.loads(result['stdout']['text'])


def test_public_freeze_branch_catches_ownership_regression_without_hidden_inputs(tmp_path):
    root=project(tmp_path/'candidate','all');before={p.name:p.read_bytes() for p in root.glob('*.py')}
    result,data=verify(root)
    assert result['status']=='failed' and data['rows'][0]['passed']
    assert data['rows'][1]['checks']['no_frozen_or_foreign_owned'] is False
    assert before=={p.name:p.read_bytes() for p in root.glob('*.py')}
    assert data['policy_version']=='trainable-optimizer-ownership-v1'


def test_correct_public_joint_training_and_freeze_both_pass(tmp_path):
    result,data=verify(project(tmp_path/'candidate','filtered'))
    assert result['status']=='passed' and data['passed'] and len(data['rows'])==2


@pytest.mark.parametrize('mode',['omitted','duplicate'])
def test_missing_or_duplicate_owned_parameters_are_not_success(tmp_path,mode):
    result,data=verify(project(tmp_path/'candidate',mode))
    assert result['status']=='failed' and data['passed'] is False


def test_weakening_trainability_cannot_pass_against_public_source_snapshot(tmp_path):
    baseline=project(tmp_path/'original','filtered')
    root=project(tmp_path/'candidate','filtered',freeze_bypass=True)
    result,data=verify(root,baseline)
    assert result['status']=='failed'
    assert data['rows'][0]['checks']['public_trainability_preserved'] is False


def test_untrusted_api_path_is_rejected_without_running_external_source(tmp_path):
    result,data=verify(project(tmp_path/'candidate'),factory='../private:factory')
    assert result['status']=='failed' and data['error_type']=='ValueError'
