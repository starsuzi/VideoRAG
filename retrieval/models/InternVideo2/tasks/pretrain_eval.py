import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb

from dataset.serialize import local_broadcast_process_authkey
from dataset import MetaLoader, create_dataset, create_loader, create_sampler, create_stateful_sampler
from models import *
from tasks.retrieval_utils import evaluation_wrapper
from tasks.shared_utils import get_media_types, setup_model
from utils.basic_utils import (MetricLogger, SmoothedValue,
                               remove_files_if_exist, setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb
try:
    from petrel_client.client import Client
except:
    Client = None
import io
import os
import shutil

logger = logging.getLogger(__name__)

ceph_ckpt_bucket = "shdd:s3://avp_ckpt"

def setup_dataloaders(config, mode="pt"):
    # train datasets, create a list of data loaders
    logger.info(f"Creating dataset for {mode} use_iter_train={config.get('use_iter_train', False)}")
    train_datasets = create_dataset(f"{mode}_train", config)
    media_types = get_media_types(train_datasets)

    if config.get('use_iter_train', False):
        if config.distributed:
            batch_size = [config.inputs.batch_size[k] for k in media_types] # batch_size for each GPU
            samplers = create_stateful_sampler(train_datasets, batch_size)
        else:
            raise NotImplementedError
    else:
        if config.distributed:
            num_tasks = get_world_size()
            global_rank = get_rank()
            samplers = create_sampler(
                train_datasets, [True] * len(media_types), num_tasks, global_rank
            )
        else:
            samplers = [None] * len(media_types)

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size=[config.inputs.batch_size[k] for k in media_types],
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[None] * len(media_types),
    )  # [0]

    # test datasets, a mapping from dataset name to data loader
    test_datasets, test_dataset_names = create_dataset(f"{mode}_eval", config)
    test_loaders = create_loader(
        test_datasets,
        [None] * len(test_datasets),
        batch_size=[config.inputs.batch_size_test[d.media_type] for d in test_datasets],
        num_workers=[config.num_workers] * len(test_datasets),
        is_trains=[False] * len(test_datasets),
        collate_fns=[None] * len(test_datasets),
    )
    test_name2loaders = {k: v for k, v in zip(test_dataset_names, test_loaders)}
    return train_loaders, test_name2loaders, media_types


def main(config):
    if config.get('use_flash_sdp', False):
        torch.backends.cuda.enable_flash_sdp(enabled=True)
    elif config.get('use_mem_efficient_sdp', False):
        torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)

    try:
        ceph_ckpt_path = f"{ceph_ckpt_bucket}/{config.output_dir.split('/')[-3]}/{config.output_dir.split('/')[-2]}/{config.output_dir.split('/')[-1]}"
        client_ckpt = Client(conf_path='~/petreloss.conf')
    except Exception as e:
        print(e)
        logger.info("Ceph is not working!!!")
        
    if is_main_process() and config.wandb.enable:
        try:
            run = setup_wandb(config)
            logger.info("Wandb is working!!!")
        except Exception as e:
            logger.warn("Wandb is not working!!!")
            print(e)

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, test_name2loaders, train_media_types = setup_dataloaders(
        config, mode=config.mode
    )
    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    cudnn.benchmark = len(train_media_types) == 1

    print(f"\033[31m CURRENT NODE NAME: {os.environ['SLURMD_NODENAME']} dataloader is OK {datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}!!! \033[0m")

    find_unused_parameters = config.model.get('find_unused_parameters', False)
    logger.info(f"find_unused_parameters={find_unused_parameters}")

    model_cls = eval(config.model.get('model_cls'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        add_decoder=False,
        pretrain=is_pretrain,
        find_unused_parameters=find_unused_parameters,
    )

    best = 0
    if type(config.best_key) is str:
        best_key = [config.best_key, "t2v_r1"]
    elif type(config.best_key) is list and len(config.best_key) == 2:
        best_key = config.best_key
    else:
        raise NotImplementedError(config.best_key)

    logger.info(f"Start training, start_epoch={start_epoch}")
    try:
        eval_res = {}
        for test_name, test_loader in test_name2loaders.items():
            if test_name not in config.test_types:
                logger.info(
                    f"Skip eval {test_name} split. All test_types {config.test_types}"
                )
                continue
            res = evaluation_wrapper(
                model_without_ddp, test_loader, tokenizer, device, config, prefix=test_name
            )
            eval_res.update(res)

        if is_main_process():
            try:
                cur_recall = eval_res[best_key[0]][best_key[1]]
            except Exception as e:
                logger.warn(e)
                print(e)
                # print(eval_res)
                cur_recall = best - 1

            eval_res = pd.DataFrame(eval_res)
            logger.info(f"Epoch")
            logger.info(f"\n{eval_res.transpose().to_string(max_cols=30)}")

            eval_res.to_json(join(config.output_dir, "eval_res_latest.json"))

            state_dict = model_without_ddp.state_dict()

            for k in config.get("no_save_params_prefix", []):
                kk = [x for x in state_dict.keys() if x.startswith(k)]
                logger.info(f"Not saving {len(kk)} params with prefix {k}")
                for kkk in kk:
                    state_dict.pop(kkk)

    except Exception as e:
        logger.warn("Something wrong when eval or save!!!")
        print(e)
        if config.evaluate:
            raise e

        dist.barrier()


if __name__ == "__main__":
    print(f"\033[31m NODE LIST: {os.environ['SLURM_NODELIST']} \033[0m")
    logger.info(f"NODE LIST: {os.environ['SLURM_NODELIST']}")
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
