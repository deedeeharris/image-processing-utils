"""
img_tools.py

Small image processing toolbox for teaching:
- read images
- display single image
- display grid of images
- show histograms

Dependencies:
    pip install opencv-python matplotlib numpy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Union, Optional

ImageLike = Union[str, np.ndarray]


def read_image(path: str, color: bool = True) -> np.ndarray:
    """
    Read an image from disk.

    Args:
        path: Path to image file.
        color: If True return RGB, else return grayscale.

    Returns:
        numpy array image.

    Raises:
        FileNotFoundError if the path is invalid.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from path: {path}")

    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def _ensure_ndarray(img: ImageLike, color_if_path: bool = True) -> np.ndarray:
    """
    Internal helper:
    If img is a path, read it. If it is already an array, just return.
    """
    if isinstance(img, str):
        return read_image(img, color=color_if_path)
    return img


def display_img_from_path(path: str, title: Optional[str] = None, show_axis: bool = False) -> None:
    """
    Keep this name for backward compatibility with your notebook.

    Args:
        path: Path to image.
        title: Optional title on the plot.
        show_axis: If False, hide axes.
    """
    img = read_image(path, color=True)
    show_single_image(img, title=title or path, show_axis=show_axis)


def show_single_image(img: ImageLike,
                      title: Optional[str] = None,
                      cmap: Optional[str] = None,
                      show_axis: bool = False,
                      figsize: tuple = (5, 5)) -> None:
    """
    Display a single image.

    Args:
        img: Numpy array or path string.
        title: Title string.
        cmap: Colormap for grayscale images, for example 'gray'.
        show_axis: If False hide axes.
        figsize: Figure size.
    """
    img = _ensure_ndarray(img, color_if_path=(cmap is None))

    plt.figure(figsize=figsize)

    if img.ndim == 2:
        plt.imshow(img, cmap=cmap or "gray")
    else:
        plt.imshow(img)

    if title:
        plt.title(title)

    if not show_axis:
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_images_grid(images: List[ImageLike],
                     titles: Optional[List[str]] = None,
                     cols: int = 3,
                     cmap: Optional[str] = None,
                     show_axis: bool = False,
                     figsize_per_image: float = 3.0) -> None:
    """
    Display many images in a grid.

    Args:
        images: List of paths or numpy arrays.
        titles: Optional list of titles with same length as images.
        cols: Number of columns in grid.
        cmap: Colormap for grayscale images.
        show_axis: If False hide axes.
        figsize_per_image: Controls figure size.
    """
    n = len(images)
    if n == 0:
        raise ValueError("images list is empty.")

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * figsize_per_image,
                                      rows * figsize_per_image))
    axes = np.array(axes).reshape(-1)

    for i, img in enumerate(images):
        ax = axes[i]
        arr = _ensure_ndarray(img, color_if_path=(cmap is None))

        if arr.ndim == 2:
            ax.imshow(arr, cmap=cmap or "gray")
        else:
            ax.imshow(arr)

        if titles and i < len(titles):
            ax.set_title(titles[i])

        if not show_axis:
            ax.axis("off")

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def show_histogram(img: ImageLike,
                   title: Optional[str] = None,
                   color: bool = True,
                   bins: int = 256) -> None:
    """
    Show intensity histogram of an image.

    Args:
        img: Numpy array or path string.
        title: Plot title.
        color: If True and image is RGB, show R/G/B channels separately.
        bins: Number of histogram bins.
    """
    arr = _ensure_ndarray(img, color_if_path=color)

    plt.figure(figsize=(6, 4))

    if arr.ndim == 2 or not color:
        plt.hist(arr.ravel(), bins=bins, range=(0, 255))
        plt.xlabel("Intensity")
        plt.ylabel("Count")
    else:
        # RGB channels
        for i, channel_name in enumerate(["R", "G", "B"]):
            plt.hist(arr[:, :, i].ravel(),
                     bins=bins,
                     range=(0, 255),
                     alpha=0.5,
                     label=channel_name)
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.legend()

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()
