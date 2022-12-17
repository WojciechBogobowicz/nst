from math import isqrt
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_trained_images(trainers):
    sublots_dims = get_sublots_dims(len(trainers))
    subplot_size = 5

    fig, axs = plt.subplots(*sublots_dims, squeeze=True)
    axs = axs.reshape(-1)
    fig.set_size_inches(sublots_dims[0] * subplot_size, sublots_dims[1] * subplot_size)

    for i, (trainer, ax) in enumerate(zip(trainers, axs)):
        plt.sca(ax)
        plt.axis("off")
        plt.title(f"Trainer {i}")
        plt.imshow(trainer.output_image)


def get_sublots_dims(subplots_num):
    potential_col_num = isqrt(subplots_num)

    while subplots_num % potential_col_num != 0:
        potential_col_num -= 1
        if potential_col_num < 0:
            raise ValueError("Traing to find dividor lower than 0")
    col_num = potential_col_num
    row_num = int(subplots_num / col_num)
    return row_num, col_num


def plot_trainer(trainer):
    all_model_layers = trainer.model_layers_names()
    style_layers = trainer.style_layers
    content_layers = trainer.content_layers
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    gs = mpl.gridspec.GridSpec(4, 3, width_ratios=(4, 1, 4), height_ratios=(5, 1, 5, 5))

    ax_style = fig.add_subplot(gs[0, 0])
    ax_content = fig.add_subplot(gs[2, 0])
    ax_output = fig.add_subplot(gs[0:3, 2])
    ax_model = fig.add_subplot(gs[3, :])

    ax_plus = fig.add_subplot(gs[1, 0])
    ax_arrow = fig.add_subplot(gs[1, 1])

    ax_style.imshow(trainer.style_image)
    ax_style.set_title(
        "Style Image", fontdict=dict(fontsize=15, color="b", fontweight="bold")
    )

    ax_content.imshow(trainer.content_image)
    ax_content.set_title(
        "Content Image", fontdict=dict(fontsize=15, color="r", fontweight="bold")
    )

    ax_output.imshow(trainer.output_image)
    ax_output.set_title(
        "Output Image", fontdict=dict(fontsize=15, color="m", fontweight="bold")
    )

    ax_plus.scatter(0, 0, marker="P", s=1000, c="black")
    ax_arrow.arrow(0, 0, 1, 0, head_length=0.2, color="black")

    ax_model.set_title(
        "Model",
        fontdict=dict(fontsize=15, color="black", fontweight="bold"),
        fontweight="bold",
    )
    lx = 0.045
    ly = 0.5

    plt.sca(ax_model)
    plt.ylim(-lx, len(all_model_layers) * lx + lx)

    for i, layer in enumerate(all_model_layers):
        if (layer in style_layers) and (layer in content_layers):
            bbox_params = dict(
                boxstyle="round", facecolor="b", edgecolor="r", linewidth=5, alpha=0.5
            )
        elif layer in style_layers:
            bbox_params = dict(boxstyle="round", facecolor="b", alpha=0.5)
        elif layer in content_layers:
            bbox_params = dict(boxstyle="round", facecolor="r", alpha=0.5)
        else:
            bbox_params = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        plt.text(
            0.03 + i * lx,
            ly,
            layer,
            rotation=90,
            fontsize=15,
            horizontalalignment="center",
            verticalalignment="center",
            bbox=bbox_params,
        )

    for ax in fig.get_axes()[:4]:
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in fig.get_axes()[4:]:
        plt.sca(ax)
        plt.axis("off")

    
def save_vizualizations(trainer, style_image_path, content_image_path, BASE_RESULT_PATH, BASE_TRAINER_PATH):
    striped_style_path, _ = os.path.splitext(style_image_path)
    striped_content_path, _ = os.path.splitext(content_image_path)

    style_img_name = os.path.split(striped_style_path)[-1]
    content_img_name = os.path.split(striped_content_path)[-1]
    ext = "jpg"

    img_name = f"{style_img_name}__{content_img_name}"
    result_path = os.path.join(BASE_RESULT_PATH, f"{img_name}.{ext}")
    trainer_path = os.path.join(BASE_TRAINER_PATH, f"{img_name}.{ext}")

    if os.path.exists(result_path):
        postfix = __find_unique_postfix(result_path)
        result_path = os.path.join(BASE_RESULT_PATH, f"{img_name}{postfix}.{ext}")
        trainer_path = os.path.join(BASE_TRAINER_PATH, f"{img_name}{postfix}.{ext}")

    trainer.output_image.save(result_path)

    plot_trainer(trainer)
    plt.savefig(trainer_path)



def __find_unique_postfix(path: str):
    filename, extension = os.path.splitext(path)
    idx = 0
    new_filename = f"{filename}__({idx})"
    while os.path.exists(f"{new_filename}{extension}"):
        idx += 1
        new_filename = f"{filename}__({idx})"
    return f"__({idx})"
    
