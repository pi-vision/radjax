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
from .consts import au

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
# Interactive plots and sliders (requires ipywidgets)
# ----------------------------------------------------------------------------- #
def plot_ray_bundle_3d(
    rays,
    obs_params,
    n_side: int = 8,
    plane_alpha: float = 0.25,
    midplane_alpha: float = 0.25,
    line_alpha: float = 0.9,
    line_width: float = 0.8,
    cmap: str | None = None,
    elev: float = 22,
    azim: float = -60,
    ax=None,
):
    """
    3D visualize a subsampled bundle of rays in (x, y, z) with slab planes.

    Parameters
    ----------
    rays : RayBundle
        Must expose `coords_xyz` with shape (ny, nx, nray, 3) in **centimeters**.
    obs_params : ObservationParams-like
        Must expose `z_width` in AU (used for slab planes). If missing, defaults to 400 AU.
    n_side : int
        Subsample factor along each image dimension (plots ~n_side^2 rays).
    plane_alpha : float
        Opacity for the upper/lower slab planes (z = ± z_width/2).
    midplane_alpha : float
        Opacity for the midplane (z = 0).
    line_alpha : float
        Opacity for ray lines.
    line_width : float
        Line width for ray lines.
    cmap : str | None
        Optional Matplotlib colormap name to color-code rays by grid index. If None, uses default.
    elev, azim : float
        3D view angles passed to `ax.view_init`.
    ax : mpl_toolkits.mplot3d.Axes3D | None
        Existing 3D axes to draw on; if None, a new figure/axes is created.

    Returns
    -------
    ax : mpl_toolkits.mplot3d.Axes3D
        The axes with the plotted rays.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    
    coords = np.asarray(rays.coords_xyz)  # (ny, nx, nray, 3) in cm
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError("Expected rays.coords_xyz of shape (ny, nx, nray, 3)")

    # ---- cm → AU conversion ----
    coords_au = coords / au  # now in AU

    ny, nx, _, _ = coords_au.shape
    iy_idx = np.linspace(0, ny - 1, max(1, int(n_side)), dtype=int)
    ix_idx = np.linspace(0, nx - 1, max(1, int(n_side)), dtype=int)

    # Build color indices if requested
    colors = None
    if cmap is not None:
        # map (iy, ix) to 0..1
        ii, jj = np.meshgrid(np.arange(len(iy_idx)), np.arange(len(ix_idx)), indexing="ij")
        norm = (ii.ravel() * len(ix_idx) + jj.ravel()) / (len(iy_idx) * len(ix_idx) - 1 + 1e-12)
        cmap_fn = plt.get_cmap(cmap)
        colors = cmap_fn(norm)

    # Collect extents for plane sizing
    xs, ys, zs = [], [], []
    for k, iy in enumerate(iy_idx):
        for l, ix in enumerate(ix_idx):
            xyz = coords_au[iy, ix]  # (nray, 3)
            xs.append(xyz[:, 0]); ys.append(xyz[:, 1]); zs.append(xyz[:, 2])
    xs = np.concatenate(xs); ys = np.concatenate(ys); zs = np.concatenate(zs)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_half = float(getattr(obs_params, "z_width", 400.0) / 2.0)  # AU

    # Pad XY so planes don't clip
    pad_xy = 0.05 * max(x_max - x_min, y_max - y_min)
    xr = np.linspace(x_min - pad_xy, x_max + pad_xy, 12)
    yr = np.linspace(y_min - pad_xy, y_max + pad_xy, 12)
    X, Y = np.meshgrid(xr, yr)
    Zp = z_half * np.ones_like(X)
    Zn = -z_half * np.ones_like(X)
    Z0 = 0.0 * X

    # Set up axes
    created_ax = False
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        created_ax = True

    # Plot rays
    color_i = 0
    for k, iy in enumerate(iy_idx):
        for l, ix in enumerate(ix_idx):
            xyz = coords_au[iy, ix]
            if colors is None:
                ax.plot3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=line_width, alpha=line_alpha)
            else:
                ax.plot3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=line_width, alpha=line_alpha, color=colors[color_i])
                color_i += 1

    # Slab planes & midplane
    ax.plot_surface(X, Y, Zp, alpha=plane_alpha, color="gray", edgecolor="none")
    ax.plot_surface(X, Y, Zn, alpha=plane_alpha, color="gray", edgecolor="none")
    ax.plot_surface(X, Y, Z0, alpha=midplane_alpha, color="k", edgecolor="none")

    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")
    ax.set_title("Ray bundle through disk slab")

    # Box aspect if available (mpl ≥ 3.3); else fall back to limits
    xrng = (x_max - x_min) + 2 * pad_xy
    yrng = (y_max - y_min) + 2 * pad_xy
    zrng = z_half * 5  # show a bit beyond planes
    try:
        ax.set_box_aspect((xrng, yrng, zrng))
    except Exception:
        ax.set_xlim(x_min - pad_xy, x_max + pad_xy)
        ax.set_ylim(y_min - pad_xy, y_max + pad_xy)
        ax.set_zlim(-z_half * 1.1, +z_half * 1.1)

    ax.view_init(elev=elev, azim=azim)

    # Annotate slab planes and midplane
    ax.text(
        x_min, y_min, 0.0,
        "midplane (z=0 AU)",
        color="k", fontsize=12
    )
    
    ax.text(
        x_min, y_min, +z_half,
        f"z_max (+{z_half:.0f} AU)",
        color="gray", fontsize=12
    )
    
    ax.text(
        x_min, y_min, -z_half,
        f"z_min (−{z_half:.0f} AU)",
        color="gray", fontsize=12
    )
    
    if created_ax:
        plt.tight_layout()
        plt.show()

    return ax

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
    title1: str | None = None,
    title2: str | None = None,
    cmap12: str = "inferno",
    cmap_diff: str = "RdBu_r",
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
    title1 : str or None
        Optional title for the first subplot (frames1).
    title2 : str or None
        Optional title for the second subplot (frames2).
    cmap12 : str
        Colormap for the first two panels (default: "inferno").
    cmap_diff : str
        Colormap for the difference panel (default: "RdBu_r").

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

    mean_images = [
        frames1.mean(axis=axis),
        frames2.mean(axis=axis),
        np.abs(frames1 - frames2).mean(axis=axis),
    ]
    cbars = []
    titles = [
        title1,
        title2,
        "Absolute difference" if scale == "amp" else "Log relative difference",
    ]
    cmaps = [cmap12, cmap12, cmap_diff]

    for ax, image, title, cmap in zip(axes, mean_images, titles, cmaps):
        im = ax.imshow(image, origin="lower", cmap=cmap)
        ax.set_title(title)
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

        for ax, img, title, cbar, cmap in zip(axes, [img1, img2, img3], titles, cbars, cmaps):
            ax.imshow(img, origin="lower", cmap=cmap)
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
def plot_disk_profile_rz(
    ax: plt.Axes,
    r_disk: np.ndarray,
    z_disk: np.ndarray,
    nd_h2: np.ndarray,
    temperature: np.ndarray,
    co_nd: np.ndarray,
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
    nd_h2 : np.ndarray
        2D number density of H2.
    temperature : np.ndarray
        2D temperature field (K).
    co_nd : np.ndarray
        2D CO number density field.
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

    cpCO = ax.contour(r_disk, z_disk, co_nd, levels=[co_nd[co_nd>0].min()], colors="black", linewidths=0.8)
    ax.clabel(cpCO, levels=[co_nd[co_nd>0].min()], fmt="CO", inline=1, fontsize=10)

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