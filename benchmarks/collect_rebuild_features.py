"""Run isolated REBUILD diagonal pretraining and export one fixed feature batch.

This script imports REBUILD's training components without modifying that
repository or writing checkpoints there.  Its only persistent outputs are the
FASTNEWTON experiment artifacts under ``RUN_DIR``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch


REBUILD = Path("/home/dingyn/code/REBUILD")
RUN_DIR = Path("/home/dingyn/code/FASTNEWTON/experiments/rebuild_scene15_mr08_e20_s1")


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def main() -> None:
    # Keep REBUILD as an import-only dependency for this data-collection run.
    sys.path.insert(0, str(REBUILD))
    from dataset_loader import load_dataset
    from engine_train import train_one_epoch
    from main_train import build_data_loaders, get_args_parser, load_config_file
    from model import CorreGenBaselineModel
    import utils

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    parser = get_args_parser()
    args = parser.parse_args([])
    for key, value in load_config_file(str(REBUILD / "config" / "Scene15.yaml")).items():
        setattr(args, key, value)

    args.dataset = "Scene15"
    args.data_path = str(REBUILD / "dataset")
    args.output_dir = str(RUN_DIR)
    args.visual_dir = str(RUN_DIR / "visualize")
    args.device = "cuda:0"
    args.epochs = 20
    args.num_runs = 1
    args.m_ratio = 0.8
    args.seed = 1
    args.batch_size = 512
    args.eval_batch_size = 512
    args.num_workers = 0
    args.print_freq = 20
    args.start_rectify_epoch = 20
    args.start_guide_epoch = 100
    args.save_ckpt = False
    args.train_id = 0
    args.embed_dim = args.encoder_dim[0][-1]
    args.lr = args.blr * args.batch_size / 256.0
    args.min_lr = args.min_blr * args.batch_size / 256.0

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    utils.fix_random_seeds(args.seed)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dataset_train, dataset_test = load_dataset(args)
    args.n_sample = len(dataset_train)
    args.n_sample_test = len(dataset_test)
    data_loader_train, data_loader_test = build_data_loaders(
        dataset_train, dataset_test, args
    )

    model = CorreGenBaselineModel(
        n_views=args.n_views,
        layer_dims=args.encoder_dim,
        temperature=args.temperature,
        n_classes=args.n_classes,
        drop_rate=args.drop_rate,
        beta=args.beta,
        reg=args.reg,
        rho=args.rho,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = utils.Ratioschedule(
        args.warmup_epochs,
        args.epochs,
        args.lr,
        args.min_lr,
        schedule_type="exp",
        gamma=0.98,
    )
    infolist = utils.InfoBuffer(args)
    logger = utils.FileLogger(str(RUN_DIR / "train.log"))
    logger.write("Isolated REBUILD diagonal pretraining; no checkpoint requested.")
    logger.write("Arguments: {}".format(args).replace(", ", ",\n"))

    train_state = None
    for epoch in range(args.start_epoch, args.epochs):
        args.print_this_epoch = (epoch + 1) % args.print_freq == 0 or epoch + 1 == args.epochs
        train_state = train_one_epoch(
            model,
            data_loader_train,
            data_loader_test,
            optimizer,
            device,
            infolist,
            epoch,
            scheduler,
            args,
        )
        if args.print_this_epoch and train_state is not None:
            result = train_state["k-means"]
            logger.write(
                "Epoch {}: ACC={:.4f} NMI={:.4f} ARI={:.4f} F={:.4f} CAR={:.4f}".format(
                    epoch,
                    result["acc"],
                    result["nmi"],
                    result["ari"],
                    result["f"],
                    result["car"],
                )
            )

    if len(dataset_train) < args.batch_size:
        raise RuntimeError("Scene15 dataset is smaller than the requested fixed batch")

    samples = [
        torch.as_tensor(
            dataset_train.data[view][: args.batch_size],
            dtype=torch.float32,
            device=device,
        )
        for view in range(args.n_views)
    ]
    model.eval()
    with torch.no_grad():
        z_t = [model.target_encoder[i](samples[i]) for i in range(args.n_views)]
        p = [model.cross_view_decoder[i](z_t[i]) for i in range(args.n_views)]

    # Keep both the raw model outputs and normalized forms.  STRIDE's
    # cosine-distance construction uses the normalized features.
    payload = {
        "p": [value.detach().cpu().float() for value in p],
        "z_t": [value.detach().cpu().float() for value in z_t],
        "p_normalized": [torch.nn.functional.normalize(value, dim=1).cpu().float() for value in p],
        "z_t_normalized": [torch.nn.functional.normalize(value, dim=1).cpu().float() for value in z_t],
        "indices": torch.arange(args.batch_size, dtype=torch.long),
    }
    bundle_path = RUN_DIR / "real_batch_000.pt"
    torch.save(payload, bundle_path)

    metadata = {
        "source": "REBUILD",
        "rebuild_commit": "45fd5c25b75035435ed0bb8cdfe47f6c54b967d8",
        "dataset": "Scene15",
        "dataset_file": str(REBUILD / "dataset" / "Scene_15.mat"),
        "m_ratio": args.m_ratio,
        "seed": args.seed,
        "epochs": args.epochs,
        "start_rectify_epoch": args.start_rectify_epoch,
        "batch_size": args.batch_size,
        "dtype": "float32",
        "device": str(device),
        "feature_shapes": {
            "p": [list(value.shape) for value in payload["p"]],
            "z_t": [list(value.shape) for value in payload["z_t"]],
        },
        "tensor_sha256": {
            "p": [sha256_tensor(value) for value in payload["p"]],
            "z_t": [sha256_tensor(value) for value in payload["z_t"]],
        },
        "artifact": str(bundle_path),
        "training_log": str(RUN_DIR / "train.log"),
        "train_state_available": train_state is not None,
    }
    (RUN_DIR / "real_batch_000.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.write("Exported {}".format(bundle_path))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
