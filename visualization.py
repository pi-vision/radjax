import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def intensity_to_nchw(intensity, cmap='viridis', gamma=0.5):
    """
    Utility function to converent a grayscale image to NCHW image (for tensorboard logging).
       N: number of images in the batch
       C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)
       H: height of the image
       W: width of the image

    Parameters
    ----------
    intensity: array,
         Grayscale intensity image.
    cmap : str, default='viridis'
        A registered colormap name used to map scalar data to colors.
    gamma: float, default=0.5
        Gamma correction term
        
    Returns
    -------
    nchw_images: array, 
        Array of images.
    """
    cm = plt.get_cmap(cmap)
    norm_images = ( (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity)) )**gamma
    nchw_images = np.moveaxis(cm(norm_images)[...,:3], (0, 1, 2, 3), (0, 2, 3, 1))
    return nchw_images
    
def compare_images(img1, img2, axes=None, cmap='afmhot', exp_cbar=False):
    if axes is None:
        fig, axes = plt.subplots(1,3,figsize=(9,4))
    else:
        fig = axes[0].get_figure()
    vmin = min(img1.min(), img2.min())
    vmax = max(img1.max(), img2.max())
    for ax, img in zip(axes[:-1], [img1, img2]):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        if exp_cbar:
            cbar.formatter.set_powerlimits((0, 0))
    im = axes[-1].imshow(np.abs(img1-img2), cmap='jet')
    axes[-1].get_xaxis().set_visible(False)
    axes[-1].get_yaxis().set_visible(False)
    axes[-1].set_title('absolute difference')
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    if exp_cbar:
        cbar.formatter.set_powerlimits((0,0))
    return fig, axes

def slider(movie, fov=1, axis=0, ax=None, cmap=None, vmin=None, vmax=None,  origin='lower'):
    from ipywidgets import interact
    movie = np.array(movie)
    if movie.ndim != 3:
        raise AttributeError('Movie dimensions ({}) different than 3'.format(movie.ndim))

    num_frames = movie.shape[axis]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    extent = [-fov/2, fov/2, -fov/2, fov/2]
    im = ax.imshow(np.take(movie, 0, axis=axis), extent=extent, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)

    def imshow_frame(frame):
        img = np.take(movie, frame, axis=axis)
        im.set_array(img)
        clim_min = max(np.nanmin(img), vmin if vmin is not None else -np.inf)
        clim_max = min(np.nanmax(img), vmax if vmax is not None else np.inf)
        cbar.mappable.set_clim([clim_min, clim_max])
        
    interact(imshow_frame, frame=(0, num_frames-1));

def slider_frame_comparison(frames1, frames2, axis=0, scale='amp'):
    """
    Slider comparison of two 3D xr.DataArray along a chosen dimension.
    Parameters
    ----------
    frames1: xr.DataArray
        A 3D array with 'axis' dimension to compare along
    frames2:  xr.DataArray
        A 3D array with 'axis' dimension to compare along
    scale: 'amp' or 'log', default='amp'
        Compare absolute values or log of the fractional deviation.
    """
    from ipywidgets import interact
    frames1 = np.array(frames1)
    frames2 = np.array(frames2)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plt.tight_layout()
    mean_images = [frames1.mean(axis=axis), frames2.mean(axis=axis),
                   (np.abs(frames1 - frames2)).mean(axis=axis)]
    cbars = []
    titles = [None]*3
    if scale == 'amp':
        titles[2] = 'Absolute difference'
    elif scale == 'log':
        titles[2] = 'Log relative difference'

    for ax, image in zip(axes, mean_images):
        im = ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbars.append(fig.colorbar(im, cax=cax))

    def imshow_frame(frame):
        image1 = np.take(frames1, frame, axis=axis)
        image2 = np.take(frames2, frame, axis=axis)

        if scale == 'amp':
            image3 = np.abs(np.take(frames1, frame, axis=axis) - np.take(frames2, frame, axis=axis))
        elif scale == 'log':
            image3 = np.log(np.abs(np.take(frames1, frame, axis=axis) / np.take(frames2, frame, axis=axis)))

        for ax, img, title, cbar in zip(axes, [image1, image2, image3], titles, cbars):
            ax.imshow(img, origin='lower')
            ax.set_title(title)
            cbar.mappable.set_clim([img.min(), img.max()])

    num_frames = min(frames1.shape[axis], frames2.shape[axis])
    plt.tight_layout()
    interact(imshow_frame, frame=(0, num_frames-1));

def animate_movies_synced(movie_list, axes, t_axis=0, fov=1.0, vmin=None, vmax=None, cmaps='afmhot', add_ticks=False,
                   add_colorbars=True, titles=None, fps=10, output=None, flipy=False, bitrate=1e6, dynamic_clim=False):
    
    # Image animation function (called sequentially)
    def animate_frame(i):
        for movie, im, cbar in zip(movie_list, images, cbars):
            img = np.take(movie, i, axis=t_axis)
            im.set_array(img)
            if dynamic_clim:
                cbar.mappable.set_clim([img.min(), img.max()])
        return images

    fig = plt.gcf()
    num_frames, nx, ny = movie_list[0].shape
    extent = [-fov/2, fov/2, -fov/2, fov/2]
        
    # initialization function: plot the background of each frame
    images = []
    cbars = []
    titles = [None]*len(movie_list) if titles is None else titles
    cmaps = [cmaps]*len(movie_list) if isinstance(cmaps, str) else cmaps
    vmin_list = [movie.min() for movie in movie_list] if vmin is None else vmin
    vmax_list = [movie.max() for movie in movie_list] if vmax is None else vmax

    for movie, ax, title, cmap, vmin, vmax in zip(movie_list, axes, titles, cmaps, vmin_list, vmax_list):
        if add_ticks == False:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap)
        if add_colorbars:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbars.append(fig.colorbar(im, cax=cax))
        im.set_clim(vmin, vmax)
        images.append(im)
        if flipy:
            ax.invert_yaxis()

    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate_frame, frames=num_frames, interval=1e3 / fps)

    if output is not None:
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)
        anim.save(output, writer=writer)
    return anim