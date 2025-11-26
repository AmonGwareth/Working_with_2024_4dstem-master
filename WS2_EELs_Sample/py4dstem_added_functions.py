
import os
import netCDF4
import numpy as np
import py4DSTEM

# correct for windowing in the dataset
def bin_data_diffraction_along_one_dimension(datacube, bin_factor, dtype=None):
    """
    Performs diffraction space binning of data by bin_factor.

    Parameters
    ----------
    N : int
        The binning factor
    dtype : a datatype (optional)
        Specify the datatype for the output. If not passed, the datatype
        is left unchanged

    """
    # validate inputs
    assert type(bin_factor_x) is int, f"Error: binning factor {bin_factor_x} is not an int."
    if bin_factor_x == 1:
        return datacube
    if dtype is None:
        dtype = datacube.data.dtype

    # get shape
    R_Nx, R_Ny, Q_Nx, Q_Ny = (
        datacube.R_Nx,
        datacube.R_Ny,
        datacube.Q_Nx,
        datacube.Q_Ny,
    )
    # crop edges if necessary
    if (Q_Nx % bin_factor_x == 0) and (Q_Ny % bin_factor_y == 0):
        pass
    elif Q_Nx % bin_factor_x == 0:
        datacube.data = datacube.data[:, :, :, : -(Q_Ny % bin_factor_y)]
    elif Q_Ny % bin_factor_y == 0:
        datacube.data = datacube.data[:, :, : -(Q_Nx % bin_factor_x), :]
    else:
        datacube.data = datacube.data[
            :, :, : -(Q_Nx % bin_factor_x), : -(Q_Ny % bin_factor_y)
        ]

    # bin
    datacube.data = (
        datacube.data.reshape(
            R_Nx,
            R_Ny,
            int(Q_Nx / bin_factor_x),
            bin_factor_x,
            int(Q_Ny / bin_factor_y),
            bin_factor_y,
        )
        .sum(axis=(3, 5))
        .astype(dtype)
    )

    # set dim vectors
    Qpixsize = datacube.calibration.get_Q_pixel_size() * bin_factor
    Qpixunits = datacube.calibration.get_Q_pixel_units()

    datacube.set_dim(2, [0, Qpixsize], units=Qpixunits, name="Qx")
    datacube.set_dim(3, [0, Qpixsize], units=Qpixunits, name="Qy")

    # set calibration pixel size
    datacube.calibration.set_Q_pixel_size(Qpixsize)

    # return
    return datacube




## For the pytcho tutorial
def dummy_dataset_preprocess(
        defocus,
        scan_step_size_ang,
        maximum_scattering_angle_mrad,
        num_detector_pixels,
        sx=15, sy=15, energy=80e3, semiangle_cutoff=20, rolloff=2, object_padding_px=None,
):
    """
    """

    if object_padding_px is None:
        object_padding_px = (num_detector_pixels // 4, num_detector_pixels // 4)

    dummy_data = np.ones((sx, sy, num_detector_pixels, num_detector_pixels))
    dc = py4DSTEM.DataCube(dummy_data)
    dc.calibration.set_R_pixel_size(scan_step_size_ang)
    dc.calibration.set_R_pixel_units('A')
    dc.calibration.set_Q_pixel_size(2 * maximum_scattering_angle_mrad / num_detector_pixels)
    dc.calibration.set_Q_pixel_units('mrad')

    ptycho = py4DSTEM.process.phase.SingleslicePtychographicReconstruction(
        datacube=dc,
        verbose=False,
        energy=energy,
        semiangle_cutoff=semiangle_cutoff,
        rolloff=rolloff,
        defocus=defocus,
        object_padding_px=object_padding_px,
    ).preprocess(
        force_com_rotation=0,
        force_com_transpose=False,
        plot_rotation=False,
        plot_center_of_mass=False,
    )
    print(f"Reconstruction pixel size: {ptycho.sampling[0]:.4} A")
    print(f"Reconstruction object size: {ptycho.object_cropped.shape} px")

    return ptycho



# The processed nc import function
def get_datacube_from_processed_nc_file(fp, fn):
    """
    Reader for processed_nc files into the py4dstem datacube
    Reads the Dataset 'data' from the 'processed' group
    :param fp: Path to the file
    :param fn: Filename
    :return: data, which can be read by py4dstem datacube function
    """

    ffn: str = os.path.join(fp, fn)

    if os.path.exists(fp) and os.path.isfile(ffn):
        print("Selected file:", ffn)
    else:
        print("file location does not exist, check your filepath and filename")

    ds: netCDF4.Dataset = netCDF4.Dataset(ffn, "r", format="NETCDF4")
    processed_grp = ds.groups['processed']

    data = processed_grp["data"]

    return data

