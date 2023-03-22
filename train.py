import json
import shutil

from tqdm import tqdm
import numpy as np
import os
from pathlib import Path

import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value

from torch.utils import tensorboard

from ntm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort
from ntm.args import get_parser

import sys
sys.path.append(str(Path(__file__).parent.parent / 'comgra'))

import comgra.recorder
from comgra.recorder import ComgraRecorder
from comgra.objects import DecisionMakerForRecordingsFrequencyPerType
from comgra import utilities as comgra_utilities

COMGRA_RECORDER = None

os.chdir(str(Path(__file__).parent))

comgra_root_path = Path(__file__).parent / 'comgra_data'
comgra_group = 'group_0'
tensorboard_base_path = Path(__file__).parent / 'runs'
saved_models_base_path = Path(__file__).parent / 'saved_models'

def do_the_thing():
    args = get_parser().parse_args()
    trial_index = args.trial_index
    if trial_index == 0:
        shutil.rmtree(comgra_root_path / comgra_group, ignore_errors=True)
        shutil.rmtree(tensorboard_base_path, ignore_errors=True)
        shutil.rmtree(saved_models_base_path, ignore_errors=True)
    saved_models = saved_models_base_path / f'trial_{trial_index}'
    saved_models.mkdir(parents=True, exist_ok=True)
    writer_for_tensorboard = tensorboard.SummaryWriter(str(tensorboard_base_path / 'trial' / f"trial_{trial_index}"))
    COMGRA_RECORDER = ComgraRecorder(
        # The root folder for all comgra data
        comgra_root_path=comgra_root_path,
        # All runs of comgra that share the same 'group' will be loaded in the same application
        # when you run the server.py application with that group as the name argument.
        # They can then be selected by their 'trial_id' and compared.
        group=comgra_group, trial_id=f'trial_{trial_index}',
        # These parameters can be left empty, but it is recommended to fill them in
        # if your computational graph is complex.
        # They ensure that similar module parameters get grouped together visually.
        # All module parameters whose complete name (including the list of names of modules they are contained in)
        # match one of these prefixes are grouped together.
        # This can also be used to group tensors together that have a variable number of occurrences,
        # for example the elements of an attention mechanism.
        prefixes_for_grouping_module_parameters_visually=[
            'root_module.controller',
            'root_module.heads',
            'root_module.memory',
        ],
        prefixes_for_grouping_module_parameters_in_nodes=[],
        # Parameters that will be recorded in a JSON file, to help you with comparing things later.
        parameters_of_trial={},
        # How often do you want comgra to make a recording?
        # There are several options for this.
        # This one is often the most effective one:
        # At the beginning of each training step later in this code, you specify what the type of this recording is.
        # For example, you could differentiate between randomly selected training and training on a specific example
        # that you would like to inspect in more detail.
        # This recording type ensures that a recording is made if the last training of the specified type was at least
        # N training steps ago.
        # In this way, you make sure that each type gets recorded often enough to be useful.
        decision_maker_for_recordings=DecisionMakerForRecordingsFrequencyPerType(min_training_steps_difference=1000),
        # Comgra records data both in terms of statistics over the batch dimension and in terms of
        # individual items in the batch.
        # If batches are large, this consumes too much memory and slows down the recording.
        # This number tells comgra only to record the first N items of each batch.
        # Note that the statistics over the batch that also get recorded are still calculated over the whole batch.
        max_num_batch_size_to_record=5,
        # Use this to turn comgra off throughout your whole project.
        comgra_is_active=True,
        # A performance parameter you can experiment with if comgra is too slow.
        # If this is too low, comgra becomes slow.
        # If this is too high, the program may crash due to memory problems.
        max_num_mappings_to_save_at_once_during_serialization=10000,
        # An optional feature to skip the recording of KPIs that are particularly expensive to calculate.
        calculate_svd_and_other_expensive_operations_of_parameters=False,
    )
    COMGRA_RECORDER.REGULARIZATION = [None, 3, 5, 10, 20][trial_index]

    # ----------------------------------------------------------------------------
    # -- initialize datasets, model, criterion and optimizer
    # ----------------------------------------------------------------------------

    # args.task_json = 'ntm/tasks/copy.json'
    # args.task_json = 'ntm/tasks/repeatcopy.json'
    # args.task_json = 'ntm/tasks/associative.json'
    # args.task_json = 'ntm/tasks/ngram.json'
    args.task_json = 'ntm/tasks/prioritysort.json'

    task_params = json.load(open(args.task_json))

    # dataset = CopyDataset(task_params)
    # dataset = RepeatCopyDataset(task_params)
    # dataset = AssociativeDataset(task_params)
    # dataset = NGram(task_params)
    dataset = PrioritySort(task_params)

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
              comgra=COMGRA_RECORDER)

    COMGRA_RECORDER.track_module("root_module", ntm)

    criterion = nn.BCELoss()
    # As the learning rate is task specific, the argument can be moved to json file
    optimizer = optim.RMSprop(ntm.parameters(),
                              lr=args.lr,
                              alpha=args.alpha,
                              momentum=args.momentum)
    '''
    optimizer = optim.Adam(ntm.parameters(), lr=args.lr,
                           betas=(args.beta1, args.beta2))
    '''

    # ----------------------------------------------------------------------------
    # -- basic training loop
    # ----------------------------------------------------------------------------
    losses = []
    errors = []
    for iter in tqdm(range(args.num_iters)):
        optimizer.zero_grad()
        ntm.reset()

        data = dataset[iter]
        input, target = data['input'], data['target']
        out = torch.zeros(target.size())

        COMGRA_RECORDER.start_next_recording(
            iter, 1,
            is_training_mode=True,
            type_of_execution_for_diversity_of_recordings='type_of_execution_0',
            record_all_tensors_per_batch_index_by_default=False,
            override__recording_is_active=None,
        )

        # -------------------------------------------------------------------------
        # loop for other tasks
        # -------------------------------------------------------------------------
        num_iterations = input.size()[0] + target.size()[0]
        in_data_base = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
        COMGRA_RECORDER.start_forward_pass(
            # PROBLEM: "NTM memory operations will be performed in-place" breaks comgra
            configuration_type=f'num_iters_{num_iterations}_{COMGRA_RECORDER.REGULARIZATION}',
        )
        COMGRA_RECORDER.REGULARIZATION_LOSSES = []
        for iteration in range(num_iterations):
            COMGRA_RECORDER.ITERATION = iteration
            COMGRA_RECORDER.RECORD_YES_NO = (iteration in [0, 1, 5, 10, num_iterations - 1])
            if iteration < input.size()[0]:
                # to maintain consistency in dimensions as torch.cat was throwing error
                in_data = torch.unsqueeze(input[iteration], 0)
                if COMGRA_RECORDER.RECORD_YES_NO:
                    COMGRA_RECORDER.register_tensor(f"in_data_{COMGRA_RECORDER.ITERATION}", in_data, is_input=True, recording_type='neurons')
                out_ = ntm(in_data)
                if COMGRA_RECORDER.RECORD_YES_NO:
                    COMGRA_RECORDER.register_tensor(f"out_per_iteration_{COMGRA_RECORDER.ITERATION}", out_, is_output=True, recording_type='neurons')
            else:
                in_data = in_data_base.clone()
                if COMGRA_RECORDER.RECORD_YES_NO:
                    COMGRA_RECORDER.register_tensor(f"in_data_{COMGRA_RECORDER.ITERATION}", in_data, is_input=True, recording_type='neurons')
                out_ = ntm(in_data)
                if COMGRA_RECORDER.RECORD_YES_NO:
                    COMGRA_RECORDER.register_tensor(f"out_per_iteration_{COMGRA_RECORDER.ITERATION}", out_, is_output=True, recording_type='neurons')
                out[iteration - input.size()[0]] = out_
            if iteration == num_iterations - 1:
                COMGRA_RECORDER.start_backward_pass()
                COMGRA_RECORDER.register_tensor(f"out", out.reshape(-1).unsqueeze(0), is_output=True, recording_type='neurons')
                COMGRA_RECORDER.register_tensor(f"target", target.reshape(-1).unsqueeze(0), is_target=True, recording_type='neurons')
                loss = criterion(out, target)
                losses.append(loss.item())
                COMGRA_RECORDER.register_tensor(f"loss", loss, is_loss=True)
                total_loss = loss
                if COMGRA_RECORDER.REGULARIZATION:
                    reg_loss = None
                    for reg_loss_ in COMGRA_RECORDER.REGULARIZATION_LOSSES:
                        reg_loss = reg_loss_ if reg_loss is None else (reg_loss + reg_loss_)
                    COMGRA_RECORDER.register_tensor(f"regularization_loss", reg_loss.detach(), is_loss=True)
                    total_loss = total_loss + reg_loss
                total_loss.backward()
                # clips gradient in the range [-10,10]. Again there is a slight but
                # insignificant deviation from the paper where they are clipped to (-10,10)
                nn.utils.clip_grad_value_(ntm.parameters(), 10)
                optimizer.step()
                COMGRA_RECORDER.record_current_gradients(f"gradients")
        COMGRA_RECORDER.finish_iteration(sanity_check__verify_graph_and_global_status_equal_existing_file=iter < 1000 * 1000)
        COMGRA_RECORDER.finish_batch()
        binary_output = out.clone()
        binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

        # sequence prediction error is calculated in bits per sequence
        error = torch.sum(torch.abs(binary_output - target))
        errors.append(error.item())

        # ---logging---
        if iter % 200 == 0:
            print('Iteration: %d\tLoss: %.2f\tError in bits per sequence: %.2f' %
                  (iter, np.mean(losses), np.mean(errors)))
            writer_for_tensorboard.add_scalar('train_loss', np.mean(losses), iter)
            if COMGRA_RECORDER.REGULARIZATION:
                writer_for_tensorboard.add_scalar('regularization_loss', reg_loss, iter)
            writer_for_tensorboard.add_scalar('bit_error_per_sequence', np.mean(errors), iter)
            losses = []
            errors = []
        if (iter + 1) % 10000 == 0:
            torch.save(ntm.state_dict(), saved_models / f'saved_model_{iter}.pt')
            # torch.save(ntm, PATH)


do_the_thing()
