from torch.utils.tensorboard import SummaryWriter
from flax.training import checkpoints

from train_state import get_pretrain_state, get_lineval_state, TrainState
from data_loader import NumpyLoader
from epoch import pretrain_epoch, lineval_epoch
from hyperparams import LinearHyperParams, PretrainHyperParams

import time
from datetime import timedelta

CHECKPOINT_DIR = "checkpoints/"


def do_pretraining(
    pretrain_dataset, *, hyperparams: PretrainHyperParams
) -> TrainState:
    dataloader = NumpyLoader(
        pretrain_dataset, batch_size=hyperparams.batch_size
    )
    pretrain_state = get_pretrain_state(
        hyperparams,
        steps_per_epoch=len(pretrain_dataset) // hyperparams.batch_size,
    )
    assert pretrain_state is not None
    epoch = int(pretrain_state.epoch)
    print("Initial epoch", epoch, type(epoch))

    # Tensorboard stuff
    tb_writer = SummaryWriter("tensorboard_logs/", filename_suffix="pretrain")

    while epoch < hyperparams.num_epochs:
        start_time = time.perf_counter()

        # Do epoch and save
        pretrain_state, pretrain_loss = pretrain_epoch(
            pretrain_state, dataloader
        )
        epoch += 1
        checkpoints.save_checkpoint(
            ckpt_dir=CHECKPOINT_DIR,
            target=pretrain_state,
            step=epoch,
            prefix=hyperparams.ckpt_prefix(),
        )

        # Tensorboard
        tb_writer.add_scalar("Loss/train", pretrain_loss, epoch)

        # Timing
        time_elapsed = time.perf_counter() - start_time
        epochs_remaining = hyperparams.num_epochs - epoch
        time_remaining = timedelta(seconds=int(epochs_remaining * time_elapsed))

        print(
            f"Epoch {pretrain_state.epoch}: pretrain loss {pretrain_loss:.3f} ({time_elapsed:.1f} sec, {time_remaining} remaining)"
        )

    return pretrain_state


def do_linear_eval(
    nonaugmented_dataset, *, hyperparams: LinearHyperParams
) -> TrainState:
    dataloader = NumpyLoader(
        nonaugmented_dataset, batch_size=hyperparams.batch_size
    )

    train_state = get_lineval_state(
        hyperparams,
        steps_per_epoch=len(nonaugmented_dataset) // hyperparams.batch_size,
    )
    epoch = int(train_state.epoch)
    print("Initial epoch", epoch, type(epoch))

    while epoch < hyperparams.num_epochs:
        train_state, loss, acc = lineval_epoch(train_state, dataloader)
        epoch += 1
        checkpoints.save_checkpoint(
            ckpt_dir=CHECKPOINT_DIR,
            target=train_state,
            step=epoch,
            prefix=hyperparams.ckpt_prefix(),
        )
        print(
            f"Epoch {train_state.epoch}: supervised loss {loss:.3f} acc {acc:.3f}"
        )

    return train_state
