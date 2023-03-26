import json
import os
import re

from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch import nn

from torch.utils import tensorboard

from ntm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort
from ntm.args import get_parser

args = get_parser().parse_args()

os.chdir(str(Path(__file__).parent))

# args.task_json = 'ntm/tasks/copy.json'
# args.task_json = 'ntm/tasks/repeatcopy.json'
# args.task_json = 'ntm/tasks/associative.json'
# args.task_json = 'ntm/tasks/ngram.json'
args.task_json = 'ntm/tasks/prioritysort.json'

task_params = json.load(open(args.task_json))
criterion = nn.BCELoss()

# # ---Evaluation parameters for Copy task---
# task_params['min_seq_len'] = 20
# task_params['max_seq_len'] = 120

# # ---Evaluation parameters for RepeatCopy task---
# # (Sequence length generalisation)
# task_params['min_seq_len'] = 10
# task_params['max_seq_len'] = 20
# # (Number of repetition generalisation)
# task_params['min_repeat'] = 10
# task_params['max_repeat'] = 20
#
# # ---Evaluation parameters for AssociativeRecall task---
# task_params['min_item'] = 6
# task_params['max_item'] = 20

# For NGram and Priority sort task parameters need not be changed.

# dataset = CopyDataset(task_params)
# dataset = RepeatCopyDataset(task_params)
# dataset = AssociativeDataset(task_params)
# dataset = NGram(task_params)
dataset = PrioritySort(task_params)

# args.saved_model = 'saved_model_copy.pt'
# args.saved_model = 'saved_model_repeatcopy.pt'
# args.saved_model = 'saved_model_associative.pt'
# args.saved_model = 'saved_model_ngram.pt'
args.saved_model = 'saved_model_prioritysort.pt'

cur_dir = os.getcwd()

def run_test(saved_model, name, trial_iteration):
    PATH = saved_model
    # PATH = os.path.join(cur_dir, 'saved_models/saved_model_copy_500000.pt')
    # ntm = torch.load(PATH)

    """
    For the Copy task, input_size: seq_width + 2, output_size: seq_width
    For the RepeatCopy task, input_size: seq_width + 2, output_size: seq_width + 1
    For the Associative task, input_size: seq_width + 2, output_size: seq_width
    For the NGram task, input_size: 1, output_size: 1
    For the Priority Sort task, input_size: seq_width + 1, output_size: seq_width
    """

    ntm = NTM(input_size=task_params['seq_width'] + 1,
              output_size=task_params['seq_width'],
              controller_size=task_params['controller_size'],
              memory_units=task_params['memory_units'],
              memory_unit_size=task_params['memory_unit_size'],
              num_heads=task_params['num_heads'],
              comgra=None)

    ntm.load_state_dict(torch.load(PATH))

    # -----------------------------------------------------------------------------
    # --- evaluation
    # -----------------------------------------------------------------------------

    num_datasets = 1000

    losses = []
    errors = []
    with torch.no_grad():
        for iter in tqdm(range(num_datasets)):
            ntm.reset()
            data = dataset[300000 + iter]
            input, target = data['input'], data['target']
            out = torch.zeros(target.size())

            # -----------------------------------------------------------------------------
            # loop for other tasks
            # -----------------------------------------------------------------------------
            for i in range(input.size()[0]):
                # to maintain consistency in dimensions as torch.cat was throwing error
                in_data = torch.unsqueeze(input[i], 0)
                ntm(in_data)

            # passing zero vector as the input while generating target sequence
            in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
            for i in range(target.size()[0]):
                out[i] = ntm(in_data)
            # -----------------------------------------------------------------------------
            # -----------------------------------------------------------------------------
            # loop for NGram task
            # -----------------------------------------------------------------------------
            '''
            for i in range(task_params['seq_len'] - 1):
                in_data = input[i].view(1, -1)
                ntm(in_data)
                target_data = torch.zeros([1]).view(1, -1)
                out[i] = ntm(target_data)
            '''
            # -----------------------------------------------------------------------------

            loss = criterion(out, target)
            losses.append(loss.item())

            binary_output = out.clone()
            binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

            # sequence prediction error is calculted in bits per sequence
            error = torch.sum(torch.abs(binary_output - target))
            errors.append(error.item())

    avg_loss = sum(losses) / len(losses)
    avg_error = sum(errors) / len(errors)

    # ---logging---
    print('Loss: %.2f\tError in bits per sequence: %.2f' % (avg_loss, avg_error))

    writer_for_tensorboard = tensorboard.SummaryWriter(str(tensorboard_base_path / 'eval' / name))
    writer_for_tensorboard.add_scalar('error', avg_error, trial_iteration)
    writer_for_tensorboard.add_scalar('loss', avg_loss, trial_iteration)

tensorboard_base_path = Path(__file__).parent / 'runs_eval'
tensorboard_base_path.mkdir(exist_ok=True)
saved_models_base_path = Path(__file__).parent / 'saved_models'
files = [(b, a.name, int(re.match('saved_model_(.*).pt', b.name).group(1))) for a in saved_models_base_path.iterdir() for b in a.iterdir()]

print("SKIPPING FILES. TOO MANY.")
files = files[0::10]

files.sort(key=lambda a: (a[1], a[2]))
for i, (saved_model, trial_name, trial_iteration) in enumerate(files):
    print(f"Entry {i} of {len(files)}", trial_iteration)
    run_test(saved_model, f'{trial_name}', trial_iteration)
