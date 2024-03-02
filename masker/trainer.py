import argparse
import os
import sqlite3
from datetime import datetime

import numpy as np
import torch
import torchvision
import deepspeed

import time

import torch.nn as nn
import torch.nn.functional as F
from deepspeed import get_accelerator

from masker.datasets.create_dataset import create_dataset, create_validation_dataset
from masker.losses.loss_factory import create_loss
from masker.models.model_factory import create_model

from masker.previewer import Previewer
from masker.training_metadata import TrainingMetadata, TrainingMetadataDto
from masker.utils import get_paths

paths = get_paths()

def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=7,
                        type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=64,
                        help="output logging information at a given interval")

    parser.add_argument(
        '--min-capacity',
        default=0,
        type=int,
        help=
        '(moe) minimum capacity of an expert regardless of the capacity_factor'
    )
    parser.add_argument(
        '--dtype',
        default='fp32',
        type=str,
        choices=['bf16', 'fp16', 'fp32'],
        help=
        'Datatype used for training'
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        type=str,
        help=
        'Checkpoint to resume'
    )
    parser.add_argument(
        '--stage',
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help=
        'Datatype used for training'
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    if args.dtype == 'bf16':
        args.dtype_torch = torch.bfloat16
    elif args.dtype == 'fp16':
        args.dtype_torch = torch.float16
    else:
        args.dtype_torch = torch.float32

    return args

def preview(inputs, labels, filenames, outputs, i = 0):
    input = inputs[i]
    label = labels[i]
    filename = filenames[i]
    output = outputs[i]

    previewer = Previewer()
    previewer.preview(input, label, str(filename), output)

if __name__ == '__main__':
    args = add_argument()
    deepspeed.init_distributed()

    if torch.distributed.get_rank() != 0:
        # might be downloading mnist data, let rank 0 download first
        #torch.distributed.barrier()
        pass

    ds_config = {
        "train_batch_size": args.batch_size,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [
                    0.9,
                    0.999
                ],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000
            }
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {
            "enabled": args.dtype == "bf16"
        },
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False
        }
    }

    # disable scheduler
    ds_config['scheduler'] = {}

    dbconn = sqlite3.connect('/home/tlaguz/db.sqlite3', timeout=300000.0)
    dbconn.row_factory = sqlite3.Row

    model_wrapper = create_model()
    net = model_wrapper.get_model()

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    parameters_count = sum(p.numel() for p in net.parameters() if p.requires_grad)

    training_metadata = TrainingMetadata(os.path.join(paths.save_dir, "training_metadata.json"))

    train_set = create_dataset(dbconn, args.dtype)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank())
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=int(args.batch_size/torch.distributed.get_world_size()),
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        sampler=train_sampler)
    valid_set = create_validation_dataset(dbconn, args.dtype)

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters, training_data=train_set, config=ds_config)

    if args.checkpoint is not None:
        model_engine.load_checkpoint(args.checkpoint)

    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    criterion = create_loss()

    start_time = time.time()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        running_loss = 0.0

        train_sampler.set_epoch(epoch)

        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, filenames = (
                data[0].type(args.dtype_torch).to(local_device),
                data[1].type(args.dtype_torch).to(local_device),
                data[2])

            model_outputs = model_engine(model_wrapper.input_preprocess(inputs))
            outputs = model_wrapper.output_postprocess(model_outputs)
            labels = model_wrapper.labels_preprocess(labels)

            loss = criterion(outputs, labels)

            model_engine.backward(loss)

            model_engine.step()

            # print statistics
            iter_loss = loss.item()
            running_loss += loss.item()
            epoch_loss += loss.item()

            # breakpoint here to preview
            # preview(inputs, labels, filenames, outputs)
            if i % args.log_interval == (args.log_interval - 1):
                checkpoint_tag = f'model_{epoch}-{i}.bin'

                # run validation
                # Get the length of the dataset
                num_ranks = model_engine.dp_world_size

                local_dataset = torch.utils.data.Subset(valid_set, list(range(local_rank, len(valid_set), num_ranks)))
                local_loader = torch.utils.data.DataLoader(local_dataset, batch_size=int(args.batch_size/num_ranks), shuffle=False, num_workers=0, pin_memory=True)
                validation_loss = 0
                with torch.no_grad():
                    for j, data in enumerate(local_loader):
                        inputs, labels, filenames = (
                            data[0].type(args.dtype_torch).to(local_device),
                            data[1].type(args.dtype_torch).to(local_device),
                            data[2])

                        model_outputs = model_engine(model_wrapper.input_preprocess(inputs))
                        outputs = model_wrapper.output_postprocess(model_outputs)
                        labels = model_wrapper.labels_preprocess(labels)

                        loss = criterion(outputs, labels)

                        validation_loss += loss.item()

                validation_loss /= j+1
                deepspeed.dist.reduce(torch.tensor(validation_loss, device=local_device), 0, op=deepspeed.dist.ReduceOp.SUM)

                # print every log_interval mini-batches
                print('--- %.2f seconds --- device: %s --- [%d, %5d, %9d data points in total] epoch loss: %.9f, iteration loss: %.9f current loss: %.9f validation loss: %.9f; Saving checkpoint: %s ...' %
                      (time.time() - start_time, local_device, epoch + 1, i + 1, i*args.batch_size, epoch_loss / (i+1), running_loss / args.log_interval, loss.item(), validation_loss, checkpoint_tag))

                os.makedirs(paths.save_dir, exist_ok=True)
                #torch.save(model_engine.module.state_dict(), checkpoint_filename)
                model_engine.save_checkpoint(paths.save_dir, tag=checkpoint_tag)

                metadata_dto = TrainingMetadataDto(
                    time=datetime.now().isoformat(),
                    time_spent=time.time() - start_time,
                    local_rank=local_rank,
                    local_device=local_device,
                    epoch=epoch + 1,
                    iteration=i + 1,
                    epoch_loss=epoch_loss / (i+1),
                    running_loss=running_loss / args.log_interval,
                    loss=iter_loss,
                    valid_loss=validation_loss,
                    checkpoint_tag=checkpoint_tag,
                    lr=optimizer.param_groups[0]['lr']
                )
                training_metadata.append(metadata_dto)

                running_loss = 0.0



    print('Finished Training')
    print("--- %s seconds ---" % (time.time() - start_time))

