import datetime
import functools
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
try:
    import petname
except ImportError:
    petname = None  # Optional dependency
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.filesystem as dcpfs
import torch.distributed.checkpoint.state_dict as dcpsd
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.tensorboard import SummaryWriter

from utils.color import c_, sr_

""" #!Documents

#* + World size: is the total number of processes involved in the distributed training. It represents the number of individual computing unit (or GPU) across all machines that participate in the training. 

#* + Rank: The unique indentifier assigned to each process in the distributed training. It represents the index of the process within the world size.

~ => The rank just be the id of each process in the distributed training.
"""

def init_seeds(seed=31, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)  # warn if deterministic is not possible
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        
    
def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__)) #TODO: identify the current working directory
    
    def _run(command):
        """
        Executes a shell command in the current working directory and returns the output as a string.

        Args:
            command (list): The command to be executed as a list of strings.

        Returns:
            str: The output of the command execution, decoded to ASCII and stripped of leading/trailing whitespace.
        """
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    
    sha = 'N/A'
    diff = 'clean'
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = 'has uncommited changes' if diff else 'clean'
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_available_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def log_on_main_process(message):
    if is_main_process():
        print(message)

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
        

def setup_for_distributed(is_master: bool):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args["rank"] = int(os.environ["RANK"])
        args["world_size"] = int(os.environ['WORLD_SIZE'])
        args["gpu"] = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args["rank"] = int(os.environ['SLURM_PROCID'])
        args["gpu"] = args["rank"] % torch.cuda.device_count()
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args["rank"], args["gpu"], args["world_size"] = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend=args["backend"],
        init_method=args["dist_url"],
        world_size=args["world_size"],
        rank=args["rank"],
    )

    torch.cuda.set_device(args["gpu"])
    print('| distributed init (rank {}): {}'.format(
        args["rank"], args["dist_url"]), flush=True)
    dist.barrier()
    setup_for_distributed(args["rank"] == 0)
    
def restart_from_checkpoint(ckp_path, run_variables=None, is_distributed: bool = True, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print(f"{c_} Found checkpoint at {ckp_path} {sr_}")

    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)
    if not is_distributed:
        checkpoint["last_model"] = {k.replace("module.", ""): v for k, v in checkpoint["last_model"].items()}

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
                if run_variables is not None:
                    run_variables[key] = value
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                    if run_variables is not None:
                        run_variables[key] = value
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
            except AttributeError as e:
                # Scalar values (int, float, np.inf, etc.) don't have load_state_dict
                # Load directly from checkpoint
                if run_variables is not None:
                    run_variables[key] = checkpoint[key]
                    print("=> loaded scalar '{}' = {} from checkpoint: '{}'".format(key, checkpoint[key], ckp_path))
        else:
            if key not in checkpoint:
                print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))
            continue

    # re load variable important for the run
    if run_variables is not None:
        for var_name in list(run_variables.keys()):
            if var_name in checkpoint and (run_variables[var_name] is None or var_name not in kwargs):
                run_variables[var_name] = checkpoint[var_name]
                print("=> loaded '{}' from checkpoint via run_variables".format(var_name))
        
def DCP_restart_from_checkpoint(checkpoint_path: str, to_load: dict,  process_group, strict_loading: bool = True, **kwargs):
    if not os.path.isfile(checkpoint_path):
        return
    print(f"{c_} Found checkpoint at {checkpoint_path} {sr_}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = dcpsd.get_model_state_dict(value)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, checkpoint_path, msg))
            except TypeError:
                try:
                    msg = dcpsd.get_optimizer_state_dict(value)
                    print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, checkpoint_path, msg))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, checkpoint_path))                
            except AttributeError as e:
                pass
            
            to_load.update(
                {
                    key: value
                }
            )
    have_not_been_loaded = {key: value for key, value in kwargs.items() if key not in to_load}
    dcp.load(
        have_not_been_loaded,
        storage_reader=dcpfs.FileSystemReader(checkpoint_path),
        planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=not strict_loading),
        process_group=process_group,
    )
    to_load.update(have_not_been_loaded)
    
    if to_load is not None:
        for var_name, var in to_load.items():
            if var_name in checkpoint and var is None:
                to_load[var_name] = checkpoint[var_name]
    
    return to_load
    
    
def restart_fsdp_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    skip_load_keys: List[str] = None,
    keys_not_sharded: List[str] = None,
    process_group: dist.ProcessGroup = None,
    **kwargs
):
    if not Path(checkpoint_path).is_dir():  # PyTorch standard checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_weights = checkpoint["teacher"]
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

        if process_group is None:
            world_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dist.get_world_size(),),
                mesh_dim_names=("dp",),
            )
        else:
            world_mesh = DeviceMesh.from_group(process_group, "cuda")
        model_weights = {
            key: (
                torch.distributed.tensor.distribute_tensor(tensor, world_mesh, src_data_rank=None)
                if not any(key_not_sharded in key for key_not_sharded in keys_not_sharded)
                else tensor
            )
            for key, tensor in model_weights.items()
        }
        model.load_state_dict(
            {
                key: tensor
                for key, tensor in model_weights.items()
                if not any(skip_load_key in key for skip_load_key in skip_load_keys)
            }
        )

        # Return scalar values and other checkpoint data
        scalar_data = {k: v for k, v in checkpoint.items() if k != "teacher"}
        return scalar_data
    else:  # DCP checkpoint
        to_load = {}
        loaded_data = DCP_restart_from_checkpoint(
            checkpoint_path=checkpoint_path,
            to_load=to_load,
            process_group=process_group,
            model=model,
            **kwargs
        )
        return loaded_data


def save_fsdp_checkpoint(
    ckpt_dir,  # output_dir/ckpt/199
    *,
    iteration,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    overwrite: bool = True,
    process_group: dist.ProcessGroup = None,
    **others: Stateful,
):
    """Save a plain/DDP/FSDP/FSDP2 model, its optimizer, an integer iteration and other stateful objects."""
    rank = get_rank(group = process_group)
    # Rank 0 checks if the checkpoint directory exists, but all ranks need to know if if exists,
    # so they can raise an error when overwrite is False. If overwrite is True, rank 0 will delete it
    # and other ranks wait for the deletion to finish.
    ckpt_dir = Path(ckpt_dir) 
    ckpt_dir_exists = [ckpt_dir.exists() if rank == 0 else None]
    src_rank = 0
    if process_group is not None:
        src_rank = torch.distributed.get_global_rank(group=process_group, group_rank=0)
    torch.distributed.broadcast_object_list(ckpt_dir_exists, src=src_rank, group=process_group)
    ckpt_dir_exists = ckpt_dir_exists[0]
    if ckpt_dir_exists:
        if overwrite:
            if rank == 0:
                if ckpt_dir.is_dir():
                    shutil.rmtree(ckpt_dir)
                else:
                    ckpt_dir.unlink()
                print(f"Deleted: {ckpt_dir}")
            torch.distributed.barrier(group=process_group)
        else:
            raise RuntimeError(f"Checkpoint already exists: {ckpt_dir}")

    # Rank 0 creates a temporary directory for the checkpoint and broadcasts the name to all ranks.
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir_tmp = [tempfile.mkdtemp(dir=ckpt_dir.parent, prefix=ckpt_dir.name) if rank == 0 else None]
    torch.distributed.broadcast_object_list(ckpt_dir_tmp, src=src_rank, group=process_group)
    ckpt_dir_tmp = Path(ckpt_dir_tmp[0])

    to_save = {"iteration": iteration}
    to_save["model"] = dcpsd.get_model_state_dict(model)
    if optimizer is not None:
        to_save["optimizer"] = dcpsd.get_optimizer_state_dict(model, optimizer)
    to_save.update(others)
    dcp.save(
        to_save,
        storage_writer=dcpfs.FileSystemWriter(ckpt_dir_tmp),
        process_group=process_group,
    )

    # Rank 0 renames the temporary directory to the final checkpoint directory. All ranks wait for the rename.
    if rank == 0:
        ckpt_dir_tmp.rename(ckpt_dir)
    torch.distributed.barrier()

    print(f"Saved: {ckpt_dir}")
    
        
def reduce_mean(tensor: torch.Tensor, nprocs: int):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt
      
def get_params_groups(model: nn.Module):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def create_save_folder(prefix: str = "TRAIN", **kwargs):
    # Only main process creates the folder, then broadcasts path to all processes
    save_path = None
    if is_main_process():
        root_save_path = kwargs.get("save_folder")
        name_task = kwargs.get("task")
        timestamp = datetime.now().strftime('%Y-%m-%d')
        unique_id = petname.generate() if petname is not None else f"run_{random.randint(1000, 9999)}"

        save_path = os.path.join(root_save_path, f"[{prefix}]{timestamp}", f"{name_task}_{unique_id}")
        os.makedirs(save_path, exist_ok=True)

        hyper_param_path = os.path.join(save_path, "params.yaml")
        hyper_arch_path = os.path.join(save_path, "arch.yaml")

        shutil.copyfile(kwargs.get("pipeline_config"), hyper_param_path)
        shutil.copyfile(kwargs.get("architecture_config"), hyper_arch_path)

    # Broadcast save_path to all processes if using distributed training
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        save_path_list = [save_path]
        torch.distributed.broadcast_object_list(save_path_list, src=0)
        save_path = save_path_list[0]

    return save_path
    
def count_parameters(model: nn.Module):
    params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    return params/1000000

def clip_gradient(model: nn.Module, clip):
    norms = []
    for _, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                param.grad.data.mul_(clip_coef)
    return norms
            
def timer(class_name=None):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Class {class_name} Function {func.__name__} took {end - start:.4f} seconds")
            return result
        return wrapper
    return decorator 


def logging_metrics(state: str = "train"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *args, **kwargs):  # instance refers to self in the class
            epoch_loss_dict = func(instance, *args, **kwargs)
            
            if is_main_process(): # todo neu ko phai la distributed training thi cai normal co phai la main process ko?
                log_dir = kwargs.get("save_model_dir", "logs")  # Get the log directory

                os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
                log_file = os.path.join(log_dir, f"{state}_metrics_log.txt")

                # Call the actual function (training_one_epoch)

                # Log the loss dictionary
                epoch = kwargs.get("epoch", "Unknown")
                log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch} - "
                log_message += " | ".join(
                    [f"{key}: {value:.4f}" for key, value in epoch_loss_dict.items()]
                ) if isinstance(epoch_loss_dict, dict) else str(epoch_loss_dict.cpu().detach().numpy())

                # Print log message to console
                print(log_message)

                # Write log message to file
                with open(log_file, "a") as f:
                    f.write(log_message + "\n")

            return epoch_loss_dict  # Return the original function output

        return wrapper
    return decorator
        
        
def log_gradient_norms_l2(module: nn.Module, writer: SummaryWriter, tag: str = "gradients", batch_idx: int = 0):
    def get_grad_norm(module):
        total_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        return total_norm ** 0.5
    
    module_grad_norm = get_grad_norm(module)
    writer.add_scalar(f"GradNorm/{tag}", module_grad_norm, batch_idx)
    writer.add_scalar(f"GradNorm/{tag}", module_grad_norm, batch_idx)
