from share import *

import gc
import inspect
import os
import torch
import torch.multiprocessing

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    # ----------------------------
    # Training knobs (easy to edit)
    # ----------------------------
    seed = 42
    resume_path = 'models/control_v11p_sd15_canny.pth'
    train_json_path = 'data/data1.json'
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    batch_size = 8
    num_workers = 2
    pin_memory = True
    persistent_workers = True if num_workers > 0 else False
    precision = 16
    use_image_logger = False
    image_logger_freq = 1000
    use_model_checkpoint = True
    checkpoint_every_n_epochs = 5
    
    log_every_n_steps = 20
    limit_train_batches = 1.0  # Use full epoch.
    max_epochs = 200

    pl.seed_everything(seed, workers=True)


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('models/cldm_v15.yaml').cpu()
    # Load checkpoints in-place to reduce CPU peak memory.
    # Keep the original order: ControlNet weights first, then SD weights override overlaps.
    ckpt_state = load_state_dict(resume_path, location='cpu')
    model.load_state_dict(ckpt_state, strict=False)
    del ckpt_state
    gc.collect()

    ckpt_state = load_state_dict('models/v1-5-pruned.ckpt', location='cpu')
    model.load_state_dict(ckpt_state, strict=False)
    del ckpt_state
    gc.collect()
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Paper-like freezing: only train c_pre_list (CSP-like blocks).
    for p in model.parameters():
        p.requires_grad = False
    for p in model.c_pre_list.parameters():
        p.requires_grad = True


    # Misc
    print(f"MyDataset.__init__ signature: {inspect.signature(MyDataset.__init__)}")
    try:
        dataset = MyDataset(
            json_path=train_json_path,
            auto_mask_mode='diff_proxy',
            diff_threshold=64,
            diff_blur_kernel=3,
        )
    except TypeError as e:
        print(f"[WARN] MyDataset(json_path=...) failed: {e}")
        print("[WARN] Falling back to MyDataset() - please sync tutorial_dataset.py if you want custom split files.")
        dataset = MyDataset()

    if len(dataset) > 0:
        sample0 = dataset[0]
        has_conflict = "m_conflict" in sample0
        has_bg = "m_bg" in sample0
        if getattr(model, "lambda_c", 0.0) > 0 and (not has_conflict or not has_bg):
            print(
                "[WARN] lambda_c > 0 but dataset has no m_conflict/m_bg. "
                "L_c will be inactive and training will fallback to L_LDM only."
            )
    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    callbacks = []
    if use_image_logger:
        callbacks.append(ImageLogger(image_logger_freq))

    # One training run -> one auto-incremented version directory: lightning_logs/version_x
    tb_logger = TensorBoardLogger(save_dir='lightning_logs', name='')

    if use_model_checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),
                filename='csp-epoch{epoch:03d}',
                monitor='epoch',
                mode='max',
                save_last=False,
                save_top_k=1,
                every_n_epochs=checkpoint_every_n_epochs,
                save_weights_only=False,
            )
        )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=precision,
        logger=tb_logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        log_every_n_steps=log_every_n_steps,
    )


    # Train!
    trainer.fit(model, dataloader)
