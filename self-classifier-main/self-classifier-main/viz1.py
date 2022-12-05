import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import random

from get_dataset import get_dataset
from hyperparams import load_pretrain_params
from train_state import get_pretrain_state


def handle_viz1(args) -> int:
    hyperparams = load_pretrain_params()
    dataset = get_dataset(hyperparams.dataset)["validation"]

    state = get_pretrain_state(
        hyperparams,
        steps_per_epoch=len(dataset) // hyperparams.batch_size,
    )
    assert state is not None

    dataset_size = len(dataset)
    idxs = np.random.choice(dataset_size, 5000)
    imgs = np.array([dataset[idx][0] for idx in idxs])

    logits, _h2, _h3, _h4 = state.apply_fn(
        {
            "params": state.params,
            "batch_stats": state.batch_stats,
        },
        jnp.array(imgs),
        train=False,
        mutable=False,
    )
    print(logits)
    print(logits.shape)

    probs = jax.nn.softmax(logits, axis=1)
    classes = jnp.argmax(probs, axis=1)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for class_idx in range(0, 9):
        WIDTH = imgs.shape[1]
        HEIGHT = imgs.shape[2]
        DEPTH = imgs.shape[3]

        figure = np.zeros((WIDTH * 3, HEIGHT * 3, DEPTH))
        class_imgs = imgs[classes == class_idx]
        for i in range(3):
            for j in range(3):
                idx = 3 * i + j
                if class_imgs.shape[0] <= idx:
                    img = np.zeros((WIDTH, HEIGHT, DEPTH))
                else:
                    img = class_imgs[idx] / 255
                figure[i * WIDTH : (i + 1) * WIDTH, j * HEIGHT : (j + 1) * HEIGHT, :] = img
        ax = axes[class_idx // 3, class_idx % 3]
        ax.imshow(figure)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("viz1.png")

    return 0
