import argparse
import collections
import itertools
import os
import random
import time
import warnings

import numpy as np
import torch

from pcdc_imvc.configs.configure import (
    get_default_config,
    resolve_dataset_input,
    validate_config,
)
from pcdc_imvc.models.model import build_model
from pcdc_imvc.utils.datasets import load_data, load_multiview_data
from pcdc_imvc.utils.get_mask import get_mask
from pcdc_imvc.utils.logger_ import get_logger
from pcdc_imvc.utils.util import cal_std

warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified trainer for two-view and multi-view IMVC")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "two", "multi"],
                        help="training mode: auto infer by config view count")
    parser.add_argument("--dataset", type=int, default=None,
                        help="dataset id (interpretation depends on mode)")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="dataset name, preferred over --dataset")
    parser.add_argument("--dry_run", action="store_true",
                        help="only validate config and CLI flow, skip dataset loading and training")

    parser.add_argument("--devices", type=str, default="0", help="gpu device ids")
    parser.add_argument("--print_num", type=int, default=100, help="gap of print evaluations")
    parser.add_argument("--loss_print_num", type=int, default=100, help="gap of printing total loss")
    parser.add_argument("--test_time", type=int, default=5, help="number of test times")
    parser.add_argument("--missing_rate", type=float, default=0.5, help="missing rate")

    parser.add_argument("--train_epoch", type=int, default=0, help="override training epoch (0: config default)")
    parser.add_argument("--base_lr", type=float, default=-1.0, help="override learning rate if > 0")
    parser.add_argument("--override_start_inference", type=int, default=-1,
                        help="override start_inference if >= 0")
    parser.add_argument("--override_start_completion", type=int, default=-1,
                        help="override start_completion if >= 0")
    parser.add_argument("--lambda1_scale", type=float, default=1.0, help="multiply lambda1 by this factor")
    parser.add_argument("--lambda2_scale", type=float, default=1.0, help="multiply lambda2 by this factor")
    parser.add_argument("--lambda3_scale", type=float, default=1.0, help="multiply lambda3 by this factor")
    parser.add_argument("--lambda4_scale", type=float, default=1.0, help="multiply lambda4 by this factor")

    parser.add_argument("--normalize_inputs", action="store_true",
                        help="apply min-max normalization to each view input")
    parser.add_argument("--input_norm_mode", type=str, default="auto", choices=["auto", "none", "minmax", "zscore"],
                        help="input normalization mode")
    parser.add_argument("--input_norm_clip", type=float, default=5.0,
                        help="clip value for zscore mode (<=0 means no clipping)")

    parser.add_argument("--diveq_enable", type=int, default=-1, help="set diveq enable: 1/0, -1 keeps default")
    parser.add_argument("--diveq_k", type=int, default=-1, help="override diveq codebook size if >0")
    parser.add_argument("--diveq_tau", type=float, default=-1.0, help="override diveq temperature if >0")
    parser.add_argument("--diveq_hard_eval", type=int, default=-1, help="set diveq hard_eval: 1/0, -1 keeps default")
    parser.add_argument("--prototype_constraint_enable", type=int, default=-1,
                        help="set prototype constraint enable: 1/0, -1 keeps default")
    parser.add_argument("--prototype_constraint_k", type=int, default=-1,
                        help="override prototype constraint codebook size if >0")
    parser.add_argument("--prototype_constraint_tau", type=float, default=-1.0,
                        help="override prototype constraint temperature if >0")
    parser.add_argument("--prototype_constraint_hard_eval", type=int, default=-1,
                        help="set prototype constraint hard_eval: 1/0, -1 keeps default")
    parser.add_argument("--enable_pretrain", type=int, default=-1, help="legacy key for log parity")
    parser.add_argument("--pretrain_epochs", type=int, default=-1, help="legacy key for log parity")
    parser.add_argument("--conf_thresh", type=float, default=-1.0, help="override confidence threshold")
    parser.add_argument("--inf_conf_weight", type=int, default=-1, help="set confidence weighting in inference loss")
    parser.add_argument("--select_best_by_label", type=int, default=-1,
                        help="label-guided model selection in training loop")
    return parser.parse_args()


def apply_common_overrides(config, args):
    config["missing_rate"] = args.missing_rate
    config["print_num"] = args.print_num
    config["loss_print_num"] = args.loss_print_num

    tr = config["training"]
    if int(args.train_epoch) > 0:
        tr["epoch"] = int(args.train_epoch)
    if float(args.base_lr) > 0:
        tr["lr"] = float(args.base_lr)
    if int(args.override_start_inference) >= 0:
        tr["start_inference"] = int(args.override_start_inference)
    if int(args.override_start_completion) >= 0:
        tr["start_completion"] = int(args.override_start_completion)
    elif "start_completion" not in tr:
        tr["start_completion"] = int(tr.get("start_inference", 0))

    tr["lambda1"] = float(tr.get("lambda1", 1.0)) * float(args.lambda1_scale)
    tr["lambda2"] = float(tr.get("lambda2", 1.0)) * float(args.lambda2_scale)
    tr["lambda3"] = float(tr.get("lambda3", 1.0)) * float(args.lambda3_scale)
    tr["lambda4"] = float(tr.get("lambda4", 1.0)) * float(args.lambda4_scale)

    tr.setdefault("diveq", {"enable": True, "K": 64, "tau": 1.0, "hard_eval": True})
    tr.setdefault("prototype_constraint", dict(tr["diveq"]))

    pc_enable = args.prototype_constraint_enable if int(args.prototype_constraint_enable) in (0, 1) else args.diveq_enable
    pc_k = args.prototype_constraint_k if int(args.prototype_constraint_k) > 0 else args.diveq_k
    pc_tau = args.prototype_constraint_tau if float(args.prototype_constraint_tau) > 0 else args.diveq_tau
    pc_hard = args.prototype_constraint_hard_eval if int(args.prototype_constraint_hard_eval) in (0, 1) else args.diveq_hard_eval

    if int(pc_enable) in (0, 1):
        tr["prototype_constraint"]["enable"] = bool(int(pc_enable))
    if int(pc_k) > 0:
        tr["prototype_constraint"]["K"] = int(pc_k)
    if float(pc_tau) > 0:
        tr["prototype_constraint"]["tau"] = float(pc_tau)
    if int(pc_hard) in (0, 1):
        tr["prototype_constraint"]["hard_eval"] = bool(int(pc_hard))

    tr["diveq"] = dict(tr["prototype_constraint"])

    if int(args.enable_pretrain) in (0, 1):
        tr["enable_pretrain"] = bool(int(args.enable_pretrain))
    else:
        tr.setdefault("enable_pretrain", False)

    if int(args.pretrain_epochs) >= 0:
        tr["pretrain_epochs"] = int(args.pretrain_epochs)
    else:
        tr.setdefault("pretrain_epochs", 0)

    if float(args.conf_thresh) >= 0:
        tr["conf_thresh"] = float(args.conf_thresh)
    else:
        tr.setdefault("conf_thresh", 0.0)

    if int(args.inf_conf_weight) in (0, 1):
        tr["inf_conf_weight"] = bool(int(args.inf_conf_weight))
    else:
        tr.setdefault("inf_conf_weight", False)

    if int(args.select_best_by_label) in (0, 1):
        tr["select_best_by_label"] = bool(int(args.select_best_by_label))
    else:
        tr.setdefault("select_best_by_label", False)

    config.setdefault("input_norm", {"enable": False, "method": "zscore", "clip": 5.0, "eps": 1e-12})
    if args.input_norm_mode != "auto":
        if args.input_norm_mode == "none":
            config["input_norm"]["enable"] = False
        else:
            config["input_norm"]["enable"] = True
            config["input_norm"]["method"] = str(args.input_norm_mode)
    elif args.normalize_inputs:
        config["input_norm"]["enable"] = True
        config["input_norm"]["method"] = "minmax"

    if float(args.input_norm_clip) > 0:
        config["input_norm"]["clip"] = float(args.input_norm_clip)
    else:
        config["input_norm"]["clip"] = None

    config.setdefault("augmentation", {"enable": False, "feature_dropout": 0.0, "gaussian_std": 0.0})


def prepare_runtime(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def run_two_view(config, logger, args, device):
    x_list, y_list = load_data(config)
    x1_train_raw, x2_train_raw = x_list[0], x_list[1]

    fold_acc, fold_nmi, fold_ari, fold_fscore = [], [], [], []

    for data_seed in range(1, args.test_time + 1):
        start = time.time()
        np.random.seed(data_seed)

        mask = get_mask(2, x1_train_raw.shape[0], config["missing_rate"])

        x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
        x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]

        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = torch.from_numpy(x2_train).float().to(device)
        mask = torch.from_numpy(mask).long().to(device)

        accumulated_metrics = collections.defaultdict(list)

        seed = data_seed if config["missing_rate"] == 0 else config["seed"]
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        model = build_model(config, mode="two")
        optimizer = torch.optim.Adam(
            itertools.chain(
                model.autoencoder1.parameters(),
                model.autoencoder2.parameters(),
                model.img2txt.parameters(),
                model.txt2img.parameters(),
            ),
            lr=config["training"]["lr"],
        )

        logger.info(model.autoencoder1)
        logger.info(model.img2txt)
        logger.info(optimizer)

        model.autoencoder1.to(device)
        model.autoencoder2.to(device)
        model.img2txt.to(device)
        model.txt2img.to(device)

        acc, nmi, ari, fscore = model.train(
            config, logger, accumulated_metrics, x1_train, x2_train, y_list, mask, optimizer, device
        )
        fold_acc.append(acc)
        fold_nmi.append(nmi)
        fold_ari.append(ari)
        fold_fscore.append(fscore)

        logger.info(f"Run {data_seed} finished in {time.time() - start:.2f}s")

    logger.info("--------------------Training over--------------------")
    cal_std(logger, fold_acc, fold_nmi, fold_ari, fold_fscore)


def run_multi_view(config, logger, args, device):
    x_list, y_list = load_multiview_data(config)

    fold_acc, fold_nmi, fold_ari, fold_fscore = [], [], [], []

    for data_seed in range(1, args.test_time + 1):
        start = time.time()
        np.random.seed(data_seed)

        mask = get_mask(config["view"], x_list[0].shape[0], config["missing_rate"])

        x_list_new = [x_list[i] * mask[:, i][:, np.newaxis] for i in range(config["view"])]
        x_list_new = [torch.from_numpy(x_list_new[i]).float().to(device) for i in range(config["view"])]

        mask_t = torch.from_numpy(mask).long().to(device)
        accumulated_metrics = collections.defaultdict(list)

        seed = data_seed * config["seed"] if config["missing_rate"] == 0 else config["seed"]
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        model = build_model(config, mode="multi")
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

        logger.info(getattr(model, "autoencoder1", None))
        logger.info(optimizer)

        model.to(device)

        acc, nmi, ari, fscore = model.train_multiview(
            config, logger, accumulated_metrics, x_list_new, y_list, mask_t, optimizer, device
        )
        fold_acc.append(acc)
        fold_nmi.append(nmi)
        fold_ari.append(ari)
        fold_fscore.append(fscore)

        logger.info(f"Run {data_seed} finished in {time.time() - start:.2f}s")

    logger.info("--------------------Training over--------------------")
    cal_std(logger, fold_acc, fold_nmi, fold_ari, fold_fscore)


def main():
    args = parse_args()
    dataset_name, actual_mode = resolve_dataset_input(
        dataset=args.dataset,
        dataset_name=args.dataset_name,
        mode=args.mode,
    )

    config = get_default_config(dataset_name, mode=actual_mode)
    validate_config(config, mode=actual_mode)
    config["dataset"] = dataset_name
    apply_common_overrides(config, args)

    device = prepare_runtime(args)
    logger, _ = get_logger(config)

    logger.info("Dataset:" + str(dataset_name))
    logger.info("Mode:" + str(actual_mode))
    # 要跳过打印的配置参数
    skip_params = {'enable_pretrain', 'pretrain_epochs', 'conf_thresh', 'inf_conf_weight', 'select_best_by_label'}
    for k, v in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for g, z in v.items():
                if g not in skip_params:
                    logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    if args.dry_run:
        logger.info("Dry run finished: config and CLI flow are valid.")
        return

    if actual_mode == "two":
        run_two_view(config, logger, args, device)
    else:
        run_multi_view(config, logger, args, device)


if __name__ == "__main__":
    main()