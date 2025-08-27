"""
Visualization utilities.

This module provides lightweight Matplotlib-based helpers to:
- convert scalar intensity images to NCHW RGB tensors (e.g., for TensorBoard),
- compare two images (with optional exponential-style colorbar formatting),
- interactively browse 3D stacks with a slider,
- side-by-side slider comparison of two stacks,
- render synchronized animations from multiple movie arrays,
- plot disk cross-sections with density/temperature overlays.

All functions are dependency-light (Matplotlib + optional ipywidgets) and avoid
introducing extra top-level dependencies.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ----------------------------------------------------------------------------- #
# Utilities
# ----------------------------------------------------------------------------- #
def _safe_minmax(arr: np.ndarray) -> Tuple[float, float]:
    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))
    if not np.isfinite(amin):
        amin = 0.0
    if not np.isfinite(amax):
        amax = 1.0
    if np.isclose(amax, amin):
        amax = amin + 1e-12
    return amin, amax


# ----------------------------------------------------------------------------- #
# Conversions & simple comparisons
# ----------------------------------------------------------------------------- #
def intensity_to_nchw(intensity: np.ndarray, cmap: str = "viridis", gamma: float = 0.5) -> np.ndarray:
    """
    Convert a grayscale image (H,W) or batch (N,H,W) to NCHW RGB (N,3,H,W).

    Parameters
    ----------
    intensity : np.ndarray
        Grayscale intensity image(s), shape (H,W) or (N,H,W).
    cmap : str, default='viridis'
        Matplotlib colormap name to map scalars to RGB.
    gamma : float, default=0.5
        Gamma correction applied after min-max normalization.

    Returns
    -------
    nchw_images : np.ndarray
        Tensor of shape (N,3,H,W). If input is (H,W), N=1.
    """
    cm = plt.get_cmap(cmap)

    img = np.asarray(intensity)
    if img.ndim == 2:
        img = img[None, ...]  # (1,H,W)
    if img.ndim != 3:
        raise ValueError(f"Expected (H,W) or (N,H,W), got shape {img.shape}")

    amin, amax = _safe_minmax(img)
    norm = np.clip((img - amin) / (amax - amin), 0.0, 1.0) ** gamma  # (N,H,W)

    # Colormap returns RGBA in last dimension → (N,H,W,4) → drop A → (N,3,H,W)
    rgba = cm(norm)  # (N,H,W,4)
    nchw = np.moveaxis(rgba[..., :3], (0, 1, 2, 3), (0, 2, 3, 1))
    return nchw


def compare_images(
    img1: np.ndarray,
    img2: np.ndarray,
    axes: Optional[Sequence[plt.Axes]] = None,
    cmap: str = "afmhot",
    exp_cbar: bool = False,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """
    Show two images and their absolute difference side-by-side.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Images with identical shape (H,W).
    axes : sequence of Axes, optional
        If provided, must be length-3. Otherwise a new figure is created.
    cmap : str
        Colormap for the first two panels.
    exp_cbar : bool
        If True, format colorbars with scientific notation limits.

    Returns
    -------
    fig, axes : (Figure, Axes[])
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match, got {img1.shape} vs {img2.shape}")

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    else:
        if len(axes) != 3:
            raise ValueError("Expected three axes for compare_images.")
        fig = axes[0].get_figure()

    vmin = float(np.nanmin([np.nanmin(img1), np.nanmin(img2)]))
    vmax = float(np.nanmax([np.nanmax(img1), np.nanmax(img2)]))

    # First two: raw images
    for ax, img in zip(axes[:-1], [img1, img2]):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        if exp_cbar:
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

    # Last: absolute difference
    diff = np.abs(np.asarray(img1) - np.asarray(img2))
    im = axes[-1].imshow(diff, cmap="jet")
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])
    axes[-1].set_title("absolute difference")
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    if exp_cbar:
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

    return fig, axes


# ----------------------------------------------------------------------------- #
# Interactive sliders (requires ipywidgets)
# ----------------------------------------------------------------------------- #
def slider(
    movie: np.ndarray,
    fov: float = 1.0,
    axis: int = 0,
    ax: Optional[plt.Axes] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    origin: str = "lower",
):
    """
    Interactive slider to browse a 3D array along one axis (in Jupyter).

    Parameters
    ----------
    movie : np.ndarray
        3D array; the selected `axis` is treated as time/frame.
    fov : float
        Field of view used to set the imshow extent to [-fov/2, fov/2]^2.
    axis : int
        Axis treated as the frame dimension.
    ax : Axes, optional
        If provided, draw on this axes; otherwise create a new figure/axes.
    cmap : str, optional
        Colormap name, forwarded to imshow.
    vmin, vmax : float, optional
        Color limits; if None, computed per-frame when sliding.
    origin : {'upper','lower'}
        Imshow origin.

    Returns
    -------
    None
        (The function creates interactive widgets inside Jupyter.)
    """
    try:
        from ipywidgets import interact
    except Exception as e:
        raise ImportError("slider() requires ipywidgets in a Jupyter environment.") from e

    movie = np.asarray(movie)
    if movie.ndim != 3:
        raise AttributeError(f"Movie must be 3D, got ndim={movie.ndim}")

    num_frames = movie.shape[axis]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    extent = [-fov / 2, fov / 2, -fov / 2, fov / 2]
    im = ax.imshow(
        np.take(movie, 0, axis=axis),
        extent=extent,
        origin=origin,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)

    def imshow_frame(frame: int):
        img = np.take(movie, frame, axis=axis)
        im.set_array(img)
        # If vmin/vmax unspecified, auto-scale per frame
        clim_min = vmin if vmin is not None else float(np.nanmin(img))
        clim_max = vmax if vmax is not None else float(np.nanmax(img))
        if not np.isfinite(clim_min):
            clim_min = 0.0
        if not np.isfinite(clim_max):
            clim_max = 1.0
        cbar.mappable.set_clim([clim_min, clim_max])

    interact(imshow_frame, frame=(0, num_frames - 1))


def slider_frame_comparison(
    frames1: np.ndarray,
    frames2: np.ndarray,
    axis: int = 0,
    scale: str = "amp",
):
    """
    Interactive slider to compare two 3D stacks frame-by-frame.

    Parameters
    ----------
    frames1, frames2 : np.ndarray
        3D arrays with a matching size along `axis`.
    axis : int
        Frame axis.
    scale : {'amp','log'}
        - 'amp': show absolute difference in panel 3,
        - 'log': show log(|frames1/frames2|) in panel 3.

    Returns
    -------
    None
        (Displays interactive widgets in Jupyter.)
    """
    try:
        from ipywidgets import interact
    except Exception as e:
        raise ImportError("slider_frame_comparison() requires ipywidgets in Jupyter.") from e

    frames1 = np.asarray(frames1)
    frames2 = np.asarray(frames2)
    if frames1.ndim != 3 or frames2.ndim != 3:
        raise ValueError("Both inputs must be 3D arrays.")
    if frames1.shape[axis] != frames2.shape[axis]:
        raise ValueError("Frame dimension sizes must match for comparison.")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plt.tight_layout()

    # Initial means for colorbar footprints
    mean_images = [
        frames1.mean(axis=axis),
        frames2.mean(axis=axis),
        np.abs(frames1 - frames2).mean(axis=axis),
    ]
    cbars = []
    titles = [None, None, "Absolute difference" if scale == "amp" else "Log relative difference"]

    for ax, image in zip(axes, mean_images):
        im = ax.imshow(image, origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbars.append(fig.colorbar(im, cax=cax))

    def imshow_frame(frame: int):
        img1 = np.take(frames1, frame, axis=axis)
        img2 = np.take(frames2, frame, axis=axis)
        if scale == "amp":
            img3 = np.abs(img1 - img2)
        elif scale == "log":
            eps = 1e-20
            img3 = np.log(np.abs((img1 + eps) / (img2 + eps)))
        else:
            raise ValueError("scale must be 'amp' or 'log'")

        for ax, img, title, cbar in zip(axes, [img1, img2, img3], titles, cbars):
            ax.imshow(img, origin="lower")
            ax.set_title(title)
            cbar.mappable.set_clim([float(np.nanmin(img)), float(np.nanmax(img))])

    num_frames = min(frames1.shape[axis], frames2.shape[axis])
    plt.tight_layout()
    interact(imshow_frame, frame=(0, num_frames - 1))


# ----------------------------------------------------------------------------- #
# Animation
# ----------------------------------------------------------------------------- #
def animate_movies_synced(
    movie_list: Sequence[np.ndarray],
    axes: Sequence[plt.Axes],
    t_axis: int = 0,
    fov: float = 1.0,
    vmin: Optional[Union[float, Sequence[float]]] = None,
    vmax: Optional[Union[float, Sequence[float]]] = None,
    cmaps: Union[str, Sequence[str]] = "afmhot",
    add_ticks: bool = False,
    add_colorbars: bool = True,
    titles: Optional[Sequence[str]] = None,
    fps: int = 10,
    output: Optional[str] = None,
    flipy: bool = False,
    bitrate: float = 1e6,
    dynamic_clim: bool = False,
    writer: str = "ffmpeg",
):
    """
    Animate multiple movies in sync, each drawn on its own axes.

    Parameters
    ----------
    movie_list : sequence of np.ndarray
        Each movie is a 3D array; `t_axis` is the frame axis.
    axes : sequence of plt.Axes
        One axes per movie in `movie_list`.
    t_axis : int
        Index of frame axis (default 0).
    fov : float
        Field-of-view for imshow extent.
    vmin, vmax : float or list of float, optional
        Either a scalar for all panels or a list/tuple with one value per movie.
    cmaps : str or list of str
        Colormap(s).
    add_ticks : bool
        Whether to show axes ticks.
    add_colorbars : bool
        Whether to draw a colorbar per panel.
    titles : list of str, optional
        Panel titles; defaults to none.
    fps : int
        Frames per second for the animation.
    output : str, optional
        Path to save the animation. If None, returns the `FuncAnimation` only.
    flipy : bool
        If True, invert y-axis.
    bitrate : float
        Video bitrate for the writer.
    dynamic_clim : bool
        If True, rescale color limits each frame based on current image.
    writer : str
        Matplotlib animation writer to use (e.g., 'ffmpeg').

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    if len(movie_list) != len(axes):
        raise ValueError("movie_list and axes must have the same length.")

    # Frame count from the chosen axis
    def frames_count(arr: np.ndarray) -> int:
        return int(arr.shape[t_axis])

    num_frames = frames_count(movie_list[0])

    extent = [-fov / 2, fov / 2, -fov / 2, fov / 2]
    fig = plt.gcf()

    titles = [None] * len(movie_list) if titles is None else list(titles)
    cmaps = [cmaps] * len(movie_list) if isinstance(cmaps, str) else list(cmaps)

    def _as_list(val, default_func):
        if val is None:
            return [default_func(m) for m in movie_list]
        if np.isscalar(val):
            return [float(val)] * len(movie_list)
        return list(val)

    vmin_list = _as_list(vmin, lambda m: float(np.nanmin(m)))
    vmax_list = _as_list(vmax, lambda m: float(np.nanmax(m)))

    images: List[plt.AxesImage] = []
    cbars: List[plt.colorbar] = []

    # Initialize panels
    for movie, ax, title, cmap, vmin_i, vmax_i in zip(
        movie_list, axes, titles, cmaps, vmin_list, vmax_list
    ):
        if not add_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(title)
        # Initialize with zeros of the correct shape (spatial slice)
        spatial_shape = list(movie.shape)
        spatial_shape.pop(t_axis)
        if len(spatial_shape) != 2:
            raise ValueError("Each movie must be 3D (frames, H, W) up to axis permutation.")
        im = ax.imshow(np.zeros(spatial_shape), extent=extent, origin="lower", cmap=cmap)
        if add_colorbars:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbars.append(fig.colorbar(im, cax=cax))
        im.set_clim(vmin_i, vmax_i)
        images.append(im)
        if flipy:
            ax.invert_yaxis()

    def animate_frame(i: int):
        # Update each panel to frame i
        for movie, im, cbar in zip(movie_list, images, cbars if add_colorbars else [None] * len(images)):
            img = np.take(movie, i, axis=t_axis)
            im.set_array(img)
            if dynamic_clim and cbar is not None:
                cbar.mappable.set_clim([float(np.nanmin(img)), float(np.nanmax(img))])
        return images

    anim = animation.FuncAnimation(fig, animate_frame, frames=num_frames, interval=1e3 / fps)

    if output is not None:
        writer_obj = animation.writers[writer](fps=fps, bitrate=bitrate) if writer else None
        anim.save(output, writer=writer_obj)

    return anim


# ----------------------------------------------------------------------------- #
# Domain-specific plot
# ----------------------------------------------------------------------------- #
def plot_disk_profile(
    ax: plt.Axes,
    r_disk: np.ndarray,
    z_disk: np.ndarray,
    temperature: np.ndarray,
    nd_h2: np.ndarray,
    abundance_co: np.ndarray,
    co_abundance: float,
    temp_levels: Sequence[float] = (20, 40, 60, 80, 100, 120),
    vmin: float = 3,
    vmax: float = 10,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot disk density (log10 n(H2)) with temperature contours and CO abundance contour.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    r_disk, z_disk : np.ndarray
        2D coordinate meshes aligned with `nd_h2`/`temperature`.
    temperature : np.ndarray
        2D temperature field (K).
    nd_h2 : np.ndarray
        2D number density of H2.
    abundance_co : np.ndarray
        2D CO abundance field.
    co_abundance : float
        CO abundance level to contour (e.g., mid-plane abundance).
    temp_levels : sequence of float
        Temperature contour levels (K).
    vmin, vmax : float
        Log10 density color scale limits.

    Returns
    -------
    fig, ax : (Figure, Axes)
    """
    fig = ax.get_figure()
    im = ax.imshow(
        np.log10(nd_h2),
        origin="lower",
        cmap="turbo",
        extent=[float(r_disk.min()), float(r_disk.max()), float(z_disk.min()), float(z_disk.max())],
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"$\log_{10} n(\mathrm{H}_2)$", fontsize=12)

    cpT = ax.contour(r_disk, z_disk, temperature, temp_levels, colors="white", linewidths=0.8)
    ax.clabel(cpT, temp_levels, inline=1, fontsize=10)

    cpCO = ax.contour(r_disk, z_disk, abundance_co, levels=[co_abundance / 2.0], colors="black", linewidths=0.8)
    ax.clabel(cpCO, levels=[co_abundance / 2.0], fmt="CO", inline=1, fontsize=10)

    ax.set_title("Disk profile: density and temperature", fontsize=13)
    ax.set_xlabel("r [au]")
    ax.set_ylabel("z [au]")

    return fig, ax
    

__all__ = [
    "intensity_to_nchw",
    "compare_images",
    "slider",
    "slider_frame_comparison",
    "animate_movies_synced",
    "plot_disk_profile",
]