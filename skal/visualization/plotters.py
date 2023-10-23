import io
import numpy as np
import tensorflow as tf
from typing import Union, List
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def nested_image_plot(
    images: Union[List, np.ndarray, tf.Tensor],
    titles: List[str],
    outer_rows: int = 1,
    outer_cols: int = 3,
    inner_rows: int = 3,
    inner_cols: int = 3,
    scale: int = 2,
    path: str = None,
    transparent: bool = True,
    dpi: int = 90,
    legend_handles: List = None,
    image_range: tuple = (0, 255),
    show: bool = False,
):
    # if isinstance(images, np.ndarray) or isinstance(images, tf.Tensor):
    #     images = np.array(images)
    #     if images.ndim == 3:
    #         images = np.expand_dims(images, axis=0)
    #     elif images.ndim != 4:
    #         raise ValueError("Input images must be a 3D or 4D array.")

    # if isinstance(images, list):
    #     images = np.stack(images, axis=0)
    #     if images.ndim != 4:
    #         raise ValueError("Input images must be a 3D or 4D array.")

    # if len(images) != outer_rows * outer_cols:
    #     raise ValueError("Number of images does not match the grid size.")

    fig = plt.figure(figsize=(outer_cols * scale, outer_rows * scale))
    outer_grid = GridSpec(outer_rows, outer_cols, figure=fig)
    aspect_ratio = inner_cols / inner_rows * outer_cols / outer_rows

    for i in range(len(images)):
        outer_ax = fig.add_subplot(outer_grid[i])

        inner_grid = GridSpecFromSubplotSpec(
            inner_rows,
            inner_cols,
            subplot_spec=outer_ax,
            hspace=0.02,
            wspace=0.02 * aspect_ratio,
        )

        subplot_images = images[i]
        channels = subplot_images.shape[-1] if len(subplot_images.shape) > 3 else 1
        cmap = "gray" if channels == 1 else "viridis"

        if image_range == (0, 1):
            subplot_images = np.array(subplot_images * 255, dtype=np.uint8)

        if image_range == (-1, 1):
            subplot_images = np.array(subplot_images * 127.5 + 127.5, dtype=np.uint8)

        for r in range(inner_rows):
            for c in range(inner_cols):
                inner_ax = fig.add_subplot(inner_grid[r, c])
                inner_ax.imshow(subplot_images[r * inner_cols + c], cmap=cmap)
                inner_ax.axis("off")

        outer_ax.set_title(titles[i])
        outer_ax.axis("off")

    if legend_handles is not None:
        fig.legend(handles=legend_handles)

    if path is not None:
        plt.savefig(path, transparent=transparent, dpi=dpi)

    if show:
        plt.show()
        plt.close(fig)

    return fig


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
