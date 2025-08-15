import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.scipy as jscp
import gc

arcsec_to_rad = np.deg2rad(1/3600)

def compute_uv_edges(cell_size: float, npix: int):
    """
    Given a cell size (arcsec) and number of pixels (npix),
    compute the image pixel size in radians and corresponding uv cell width.
    Then compute the grid edges (assumed identical for u and v).
    """
    # Image pixel width in radians.
    image_pixel_rad = cell_size * arcsec_to_rad
    # UV pixel width: 1/(npix * image_pixel_rad)
    uv_pixel_width = 1.0 / (npix * image_pixel_rad)
    # Define cell edges (npix+1 values) so that cell centers lie at:
    # u_center = uv_pixel_width * (np.arange(npix) - npix/2)
    edges = uv_pixel_width * (jnp.arange(npix + 1) - npix / 2 - 0.5)
    return edges, uv_pixel_width

def grid_channel(data: jnp.ndarray, iu: jnp.ndarray, iv: jnp.ndarray, npix: int):
    """
    Scatter-add 1D data (length nvis) into a 2D grid (npix x npix) using the provided indices.
    """
    grid = jnp.zeros((npix, npix), dtype=data.dtype)
    # Note: In the grid, row index corresponds to v (iv) and column index to u (iu)
    grid = grid.at[iv, iu].add(data)
    return grid

def grid_visibilities(
    uu: jnp.ndarray,
    vv: jnp.ndarray,
    data_re: jnp.ndarray,
    data_im: jnp.ndarray,
    weight: jnp.ndarray,
    cell_size: float,
    npix: int,
    weighting: str = "natural",  # default to natural weighting
    append_hermitian=True
):
    """
    Grid the visibilities onto a uv grid (spectral cube).

    Inputs:
      uu, vv          : arrays of shape (nchan, nvis) in units of lambda
      data_re, data_im: arrays of shape (nchan, nvis) (Jy)
      weight          : array of shape (nchan, nvis) (thermal weights, e.g., 1/sigma^2)
      cell_size       : desired pixel size in arcsec (for the image domain)
      npix            : number of pixels along each side
      weighting       : either "natural" or "uniform". With natural weighting, no density
                        normalization is applied (i.e. density_weight = 1). With uniform,
                        density_weight = 1 / (cell weight).

    Returns:
      vis_gridded     : gridded complex visibility cube of shape (nchan, npix, npix)
      mask            : boolean mask (True if cell contains visibilities)
      cell_weight     : gridded sum of weights (for each cell)
      density_weight  : per-visibility density weights used in gridding
    """
    # Compute grid edges in uv-space (same for u and v)
    edges, uv_pixel_width = compute_uv_edges(cell_size, npix)

    if append_hermitian:
        uu = jnp.concatenate([uu, -uu], axis=1)
        vv = jnp.concatenate([vv, -vv], axis=1)
        weight = jnp.concatenate([weight, weight], axis=1)
        data_re = jnp.concatenate([data_re, data_re], axis=1)
        data_im = jnp.concatenate([data_im, -data_im], axis=1)

    # Determine cell indices for each visibility
    iu = jnp.searchsorted(edges, uu, side="right") - 1  # shape (nchan, nvis)
    iv = jnp.searchsorted(edges, vv, side="right") - 1  # shape (nchan, nvis)

    # Grid the thermal weights into the uv-cells.
    def grid_weights(w, iu_ch, iv_ch):
        grid = jnp.zeros((npix, npix), dtype=w.dtype)
        grid = grid.at[iv_ch, iu_ch].add(w)
        return grid
    
    cell_weight = jax.vmap(grid_weights)(weight, iu, iv)  # shape (nchan, npix, npix)
    sigma_binned = 1.0 / jnp.sqrt(cell_weight)

    # For each visibility, extract the total weight in its cell.
    def extract_cell_weight(grid, iu_ch, iv_ch):
        return grid[iv_ch, iu_ch]
    
    cell_weight_vis = jax.vmap(extract_cell_weight, in_axes=(0, 0, 0))(
        cell_weight, iu, iv
    )
    
    # Choose density weighting based on the mode.
    if weighting == "uniform":
        density_weight = jnp.where(cell_weight_vis > 0, 1.0 / cell_weight_vis, 0.0)
    elif weighting == "natural":
        density_weight = jnp.ones_like(cell_weight_vis)
    else:
        raise ValueError("weighting must be either 'natural' or 'uniform'")

    # Form weighted visibilities.
    weighted_re = data_re * density_weight * weight
    weighted_im = data_im * density_weight * weight

    # Grid the weighted real and imaginary parts separately.
    data_re_gridded = jax.vmap(grid_channel, in_axes=(0, 0, 0, None))(
        weighted_re, iu, iv, npix
    )
    data_im_gridded = jax.vmap(grid_channel, in_axes=(0, 0, 0, None))(
        weighted_im, iu, iv, npix
    )

    # Combine into a complex visibility cube.
    vis_gridded = data_re_gridded + 1j * data_im_gridded
    # Create a mask for cells that received any visibilities.
    mask = cell_weight > 0.0

    # Apply an fftshift to the gridded outputs to get "ground" format.
    vis_gridded = jnp.fft.fftshift(vis_gridded, axes=(-2, -1))
    mask = jnp.fft.fftshift(mask, axes=(-2, -1))
    sigma_binned = jnp.fft.fftshift(sigma_binned, axes=(-2, -1))
    cell_weight = jnp.fft.fftshift(cell_weight, axes=(-2, -1))
    
     # Transfer arrays from GPU to CPU.
    vis_gridded_cpu = jax.device_get(vis_gridded)
    mask_cpu = jax.device_get(mask)
    cell_weight_cpu = jax.device_get(cell_weight)
    density_weight_cpu = jax.device_get(density_weight)
    sigma_binned_cpu = jax.device_get(sigma_binned)

    # Delete GPU arrays (if no other references exist) and force garbage collection.
    del vis_gridded, mask, cell_weight, density_weight, iu, iv, edges, uv_pixel_width, sigma_binned
    gc.collect()

    return vis_gridded_cpu, mask_cpu, cell_weight_cpu, density_weight_cpu, sigma_binned_cpu

def dirty_image_cube(
        vis_gridded: jnp.ndarray,
        weight: jnp.ndarray,
        density_weight: jnp.ndarray,
        npix: int,
        append_hermitian=True
    ):
    """
    Compute a dirty image cube from gridded visibilities, applying a normalization constant.
    
    The normalization constant is computed per channel as:
      C = 1 / sum(weight * density_weight)
    
    This mimics MPoL's scaling so that the image fluxes come out correctly.
    """
    if append_hermitian:
        weight = jnp.concatenate([weight, weight], axis=1)
        
    # Compute per-channel normalization constant.
    C = 1 / jnp.sum(weight * density_weight, axis=1)  # shape (nchan,)
    
    # Multiply gridded visibilities by C.
    vis_norm = vis_gridded * C[:, None, None]
    
    # Inverse FFT with scaling.
    img = npix**2 * jnp.fft.ifft2(vis_norm, axes=(-2, -1))
    img = jnp.fft.fftshift(img, axes=(-2, -1))
    img = jnp.flip(img, axis=-1)
    
    del vis_norm, C
    gc.collect()
    
    return jnp.real(img)

def image_to_gridded_visibility_cube(
    image: jnp.ndarray,
    weight: jnp.ndarray,
    density_weight: jnp.ndarray,
    npix: int,
    append_hermitian = False
) -> jnp.ndarray:
    """
    Compute the gridded visibility cube from a dirty image cube.
    
    This is the inverse of dirty_image_cube(). In dirty_image_cube(),
    the gridded visibilities are normalized per channel by:
    
        C = 1 / sum(weight * density_weight)
    
    and then the image is computed as:
    
        img = npix**2 * ifft2(vis_gridded * C), followed by fftshift and flip.
    
    The inverse operation is therefore:
    
        1. Flip the image back and undo fftshift.
        2. Compute the forward FFT (with norm='ortho') and divide by npix**2.
        3. Divide by the normalization constant C to recover the original gridded visibilities.
    
    Args:
        image: A dirty image cube (nchan, npix, npix) (output of dirty_image_cube)
        weight: The weight array used in gridding.
        density_weight: The density weight array used in gridding.
        npix: Number of pixels per side in the image.
        
    Returns:
        vis_gridded: A complex array of gridded visibilities (nchan, npix, npix)
    """
    if append_hermitian:
        weight = jnp.concatenate([weight, weight], axis=1)
        
    # Recompute the per-channel normalization constant.
    C = 1 / jnp.sum(weight * density_weight, axis=1)  # shape (nchan,)
    
    # Undo the flip and fftshift performed in dirty_image_cube.
    temp = jnp.flip(image, axis=-1)
    temp = jnp.fft.ifftshift(temp, axes=(-2, -1))
    
    # Compute the forward FFT (the inverse of ifft2) and undo the scaling.
    vis_norm = jnp.fft.fft2(temp, axes=(-2, -1)) / (npix**2)
    
    # Undo the multiplication by C (since vis_norm was computed from vis_gridded * C)
    vis_gridded = vis_norm / C[:, None, None]
    
    return vis_gridded


def pad_image_cube(cube, target_size=2048):
    nchan, npix, _ = cube.shape
    if npix > target_size:
        raise ValueError("Cube's dimensions are larger than target size; consider cropping instead.")
    
    pad_total = target_size - npix
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before

    padded_cube = jnp.pad(cube, 
                          pad_width=((0, 0), (pad_before, pad_after), (pad_before, pad_after)),
                          mode='constant',
                          constant_values=0)
    return padded_cube


def interpolate_grid_to_loose(vis_cube: jnp.ndarray, uu: jnp.ndarray, vv: jnp.ndarray,
                              cell_size: float, npix: int, cval: float = 0.0, order: int = 1) -> jnp.ndarray:
    """
    Interpolate a scalar 2D volume from a 3D gridded uv cube at given uv coordinates.
    
    Args:
        vis_cube: A 3D array of shape (nchan, npix, npix) containing the gridded uv data.
        uu, vv: 2D arrays of uv coordinates (in wavelengths) with shape (nchan, nvis)
        cell_size: The cell size in arcsec (used to compute the image pixel size in radians).
        npix: Number of pixels per side in the uv grid.
        cval: The fill value for points outside the volume (default 0.0).
        order: The interpolation order (e.g., 1 for linear; default 1).
    
    Returns:
        A 2D array of shape (nchan, nvis) containing the interpolated (degridded) values.
    """
    # Conversion: 1 arcsec = 1/3600 degrees, then to radians.
    arcsec_to_rad = jnp.deg2rad(1/3600)
    image_pixel_rad = cell_size * arcsec_to_rad  # in radians
    
    # Compute uv cell (pixel) width (in wavelengths)
    uv_pixel_width = 1.0 / (npix * image_pixel_rad)
    
    def interp_one_channel(vis, uu_ch, vv_ch):
        # Convert uv coordinates (in wavelengths) to fractional pixel coordinates.
        # Here the grid is assumed to be centered (center pixel at npix/2).
        u_pixel = uu_ch / uv_pixel_width + (npix / 2.0)
        v_pixel = vv_ch / uv_pixel_width + (npix / 2.0)
        
        # Stack coordinates in the order expected by map_coordinates: (v, u).
        coords = jnp.stack([v_pixel, u_pixel], axis=0)  # shape (2, nvis)
        
        shifted_vis = jnp.fft.fftshift(vis)
        interpolated = jscp.ndimage.map_coordinates(shifted_vis, coords, order=order, cval=cval)
        return interpolated
    
    # Vectorize the operation over channels.
    return jax.vmap(interp_one_channel, in_axes=(0, 0, 0))(vis_cube, uu, vv)

def plot_channel(image_cube, channel_index, title="Channel", cmap="inferno", colorbar=True):
    """
    Plot a specific channel from an image cube.
    
    Parameters:
      image_cube : jnp.ndarray or np.ndarray
          Image cube of shape (nchan, npix, npix).
      channel_index : int
          The index of the channel to plot.
      title : str, optional
          Title to use for the plot (default "Channel").
      cmap : str, optional
          Colormap to use (default "inferno").
      colorbar : bool, optional
          Whether to display a colorbar (default True).
    
    Returns:
      None; displays the plot.
    """
    # Ensure the image data is a NumPy array.
    if isinstance(image_cube, jnp.ndarray):
        image = np.array(image_cube[channel_index])
    else:
        image = image_cube[channel_index]
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(image, origin="lower", cmap=cmap)
    plt.title(f"{title} {channel_index}")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    if colorbar:
        plt.colorbar(im)
    plt.show()

def plot_channel_with_residual(image_cube1, image_cube2, channel_index,
                               title1="Image Cube 1", title2="Image Cube 2",
                               cmap="inferno", colorbar=True):
    """
    Plot a specific channel from two image cubes and their residual (difference).

    Parameters:
      image_cube1 : jnp.ndarray or np.ndarray
          First image cube of shape (nchan, npix, npix).
      image_cube2 : jnp.ndarray or np.ndarray
          Second image cube of shape (nchan, npix, npix).
      channel_index : int
          The channel index to plot.
      title1 : str, optional
          Title for the first image channel (default "Image Cube 1").
      title2 : str, optional
          Title for the second image channel (default "Image Cube 2").
      cmap : str, optional
          Colormap to use (default "inferno").
      colorbar : bool, optional
          Whether to display a colorbar for each subplot (default True).

    Returns:
      None; displays the plots.
    """
    # Convert to NumPy arrays if necessary.
    if isinstance(image_cube1, jnp.ndarray):
        img1 = np.array(image_cube1[channel_index])
    else:
        img1 = image_cube1[channel_index]
        
    if isinstance(image_cube2, jnp.ndarray):
        img2 = np.array(image_cube2[channel_index])
    else:
        img2 = image_cube2[channel_index]
    
    # Compute the residual image.
    residual = img1 - img2

    # Create a figure with three subplots side by side.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot the first image.
    im0 = axes[0].imshow(img1, origin="lower", cmap=cmap)
    axes[0].set_title(f"{title1} (Channel {channel_index})")
    axes[0].set_xlabel("Pixel X")
    axes[0].set_ylabel("Pixel Y")
    if colorbar:
        plt.colorbar(im0, ax=axes[0])
    
    # Plot the second image.
    im1 = axes[1].imshow(img2, origin="lower", cmap=cmap)
    axes[1].set_title(f"{title2} (Channel {channel_index})")
    axes[1].set_xlabel("Pixel X")
    axes[1].set_ylabel("Pixel Y")
    if colorbar:
        plt.colorbar(im1, ax=axes[1])
    
    # Plot the residual image.
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=residual.min(), vcenter=0, vmax=residual.max())
    im2 = axes[2].imshow(residual, origin="lower", cmap='seismic', norm=norm)
    axes[2].set_title(f"Residual (Image1 - Image2) (Channel {channel_index})")
    axes[2].set_xlabel("Pixel X")
    axes[2].set_ylabel("Pixel Y")
    if colorbar:
        plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()