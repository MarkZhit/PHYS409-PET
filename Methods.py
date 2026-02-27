
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

def group_data_by_metric(directory):
    # Pattern looks for - or _ followed by digits then 'mm'
    metric_pattern = re.compile(r"[-_](\d+)mm", re.IGNORECASE)

    # Storage: {metric: [(distance_array, counts_array), ...]}
    grouped_data = {}
    # print(os.listdir(directory))

    for fname in os.listdir(directory):
        if not fname.endswith(".dat"):
            continue

        match = metric_pattern.search(fname)
        if not match:
            continue # Skip files that don't have the w-Xmm metric

        width = match.group(1)
        # print(width)

        # --- Your existing reading logic ---
        data = pd.read_csv(
            os.path.join(directory, fname),
            sep=r"\s+",
            skiprows=1,
            names=["Distance", "Counts"]
        )
        dist_array = data["Distance"].to_numpy()

        midpoint = (dist_array[-1] - dist_array[0])/2
        dist_array = dist_array - midpoint

        counts_array = data["Counts"].to_numpy()
        counts_uncertainty = count_uncertainty_from_counts(counts_array)
        # -----------------------------------

        if width not in grouped_data:
            grouped_data[width] = []

        # Add a tuple of the two arrays to the list for this metric
        grouped_data[width].append((dist_array, counts_array, counts_uncertainty))

    return grouped_data

def count_uncertainty_from_counts(count_arr):
    # count_arr_uncertainty = []
    # # N = sum(count_arr)
    # for count in count_arr:
    #     # p_i = count / N
    #     # uncertainty = np.sqrt(count * p_i * (1-p_i))
    #     uncertainty = np.sqrt(count)
    #     # TODO: above sqrt uncertainty is for poisson, new from what I had previously
    #     count_arr_uncertainty.append(uncertainty)
    # return count_arr_uncertainty * 0.5

    return np.sqrt(count_arr)

    # return np.ones_like(count_arr_uncertainty) * np.max(count_arr_uncertainty)


# def plotWidthScansGaussian(grouped_data, savename):
#     plt.figure()
#     plt.title("Sweeped Counts for Different Widths")
#     plt.xlabel("Displacement (mm)")
#     plt.ylabel("Counts")
#
#     for metric in sorted(grouped_data.keys()):
#         data_list = grouped_data[metric]
#
#         # data_list is a list of tuples: (distance_array, counts_array)
#         for dist, counts, ucounts in data_list:
#             plt.errorbar(dist, counts, fmt='-o', yerr = ucounts, label=f"counts for width = {metric}mm")
#             # plt.plot(dist, counts, label=f"counts for width = {metric}mm")
#
#     plt.legend()
#     plt.savefig("figures/" + savename)
#     plt.show()


def plotWidthScanGaussian(dist, counts, ucounts, width, popt, pcov, savename):
    dist_fit = np.linspace(dist[0], dist[-1], 100)
    fitted_curve = gaussian_1d(dist_fit, *popt)
    amp_f, mean_f, sigma_f, offset_f = popt
    # print(pcov)
    uamp_f = pcov[0][0]
    umean_f = pcov[1][1]
    usigma_f = pcov[2][2]
    uoffset_f = pcov[3][3]

    plt.figure()
    plt.title(f"Sweeped Counts for W={width}mm")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Counts")
    plt.errorbar(dist, counts, fmt='o', yerr=ucounts, label=f"counts data")
    plt.plot(dist_fit, fitted_curve, label=f"Gaussian fit, mu={mean_f:.2f}+/-{np.sqrt(umean_f):.2f}\nsigma={sigma_f:.2f}+/-{np.sqrt(usigma_f):.2f}")
    plt.legend(loc="upper right")
    plt.savefig("figures/" + savename)
    plt.show()



def plotWidthScans(grouped_data, savename):
    plt.figure()
    plt.title("Sweeped Counts for Different Widths")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Counts")

    for metric in sorted(grouped_data.keys()):
        data_list = grouped_data[metric]

        # data_list is a list of tuples: (distance_array, counts_array)
        for dist, counts, ucounts in data_list:
            plt.errorbar(dist, counts, fmt='-o', yerr = ucounts, label=f"counts for width = {metric}mm")
            # plt.plot(dist, counts, label=f"counts for width = {metric}mm")

    plt.legend()
    plt.savefig("figures/" + savename)
    plt.show()

def display_FWHM(grouped_fwhm):

    for width in sorted(grouped_fwhm.keys()):
        data_list = grouped_fwhm[width]

        # data_list is a list of tuples: (distance_array, counts_array)
        for fwhm, r0, r1 in data_list:
            print(f"with W={width}, fwhm={fwhm}mm, from {r0} to {r1}mm")



def normalizeWidthScans(grouped_data):
    for width in grouped_data:
        data_list = grouped_data[width]
        normalized_counts_list = []
        normalized_ucounts_list = []

        for dist, counts, ucounts in data_list:
            max_val = np.max(counts) # Use np.max for arrays
            maxvalIndex = np.where(counts == max_val)[0][0]
            max_uncertainty = ucounts[maxvalIndex]
            normalized_counts = counts / max_val
            for i,ucount in enumerate(ucounts):
                normalized_ucounts_list.append(normalized_counts[i] * np.sqrt( (ucount / counts[i])**2 + (max_uncertainty / max_val)**2))

            normalized_counts_list.append((dist, normalized_counts, normalized_ucounts_list))



        # Keep it as a list of tuples
        grouped_data[width] = normalized_counts_list

    return grouped_data

def getFWHM(normed_group_data):
    grouped_fwhm = {}

    for width in normed_group_data:
        data_list = normed_group_data[width]
        normalized_list = []

        for dist, counts, ucounts in data_list:
            fwhm, r0, r1 = get_fwhm(dist, counts)
            normalized_list.append((fwhm, r0, r1))

        # Keep it as a list of tuples
        grouped_fwhm[width] = normalized_list

    return grouped_fwhm

def get_fwhm(x, y):
    spline = UnivariateSpline(x, y - 0.5, s=0)
    # print(len(spline(x)))
    # plt.plot(x, spline(y))

    roots = spline.roots()

    if len(roots) >= 2:
        # Assuming the two outermost roots are the ones we want
        fwhm = abs(roots[-1] - roots[0])
        return fwhm, roots[0], roots[-1]
    return None, None, None


def getFWHMGaussian(normed_group_data):
    grouped_fwhm = {}

    for width in normed_group_data:
        data_list = normed_group_data[width]
        normalized_list = []

        for dist, counts, ucounts in data_list:
            fwhm, ufwhm = get_fwhm_Gauss(dist, counts, ucounts)
            normalized_list.append((fwhm, ufwhm))

        # Keep it as a list of tuples
        grouped_fwhm[width] = normalized_list

    return grouped_fwhm

def get_fwhm_Gauss(x, y, yerr):
    # p0 = [np.max(x), y[np.argmax(y)], 1.0, np.min(y)]

    p0 = [np.max(y), x[np.argmax(y)], 1.0, np.min(y)]

    # 4. Perform the weighted fit
    popt, pcov = curve_fit(gaussian_1d, x, y, p0=p0) # not using , sigma=yerr
    amp_f, mean_f, sigma_f, offset_f = popt

    # 5. Calculate parameter errors
    perr = np.sqrt(np.diag(pcov))
    uamp_f, umean_f, usigma_f, uoffset_f = perr

    fwhm = 2.35 * abs(sigma_f) # got this factor online
    ufwhm = 2.35 * abs(usigma_f) # 2.35 is a perfectly certain number

    return fwhm, ufwhm

def plotLineScan(dist, counts, ucounts, savename):
    plt.figure()
    plt.title("Sweeped Counts")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Counts")

    # data_list is a list of tuples: (distance_array, counts_array)
    plt.errorbar(dist, counts, fmt='-o', yerr = ucounts)
    # plt.plot(dist, counts, label=f"counts for width = {metric}mm")

    plt.savefig("figures/" + savename)
    plt.show()

def readLineScan(pathname):
    data = pd.read_csv(
            pathname,
            sep=r"\s+",
            skiprows=1,
            names=["Distance", "Counts"]
        )
    dist_array = data["Distance"].to_numpy()

    midpoint = (dist_array[-1] - dist_array[0])/2
    dist_array = dist_array - midpoint

    counts_array = data["Counts"].to_numpy()
    counts_uncertainty = count_uncertainty_from_counts(counts_array)
    return (dist_array, counts_array, counts_uncertainty)


# Expects all arrays to be of the same length, this comes automatically from the PET lab collection
def readSinogram(directory):
    angle_pattern = re.compile(r"(-?\d+_\d+)\s*Deg")
    angle_to_counts = {}
    angle_to_ucounts = {}
    distances_ref = []
    for fname in os.listdir(directory):
        if not fname.endswith(".dat"):
            continue
        # print(fname)

        match = angle_pattern.search(fname)
        if not match:
            raise ValueError(f"Angle not found in filename: {fname}")

        angle = float(match.group(1).replace("_", "."))

        data = pd.read_csv(
            os.path.join(directory, fname),
            sep=r"\s+",
            skiprows=1,
            names=["Distance", "Counts"]
        )
        # print(distances_ref)
        # if distances_ref is None:
        if (np.size(distances_ref) == 0):
            distances_ref = data["Distance"].to_numpy()
        elif not np.array_equal(distances_ref, data["Distance"].to_numpy()):
            raise ValueError("Distance grid mismatch in file: " + fname)
        # print(distances_ref)

        angle_to_counts[angle] = data["Counts"].to_numpy()
        angle_to_ucounts[angle] = count_uncertainty_from_counts(data["Counts"].to_numpy())

    angles = np.array(sorted(angle_to_counts.keys()))
    # print(angles)
    # Build 2D array: rows = distance, cols = angle
    counts_2d = np.column_stack([angle_to_counts[a] for a in angles])
    ucounts_2d = np.column_stack([angle_to_ucounts[a] for a in angles])

    midpoint = (distances_ref[-1] - distances_ref[0]) / 2
    distances_ref = distances_ref - midpoint
    # print(distances_ref)
    return counts_2d, ucounts_2d, distances_ref, angles


def display_raw_sinogram(count, distance, angle, savename):
    plt.figure()
    plt.imshow(
        count,
        origin="lower",
        aspect="auto",
        extent=[
            angle[0], angle[-1],
            distance[0], distance[-1]],
        cmap="gray"
    )
    plt.title("Raw Sinogram")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Distance (mm)")
    plt.colorbar(label="Counts")
    plt.savefig("figures/" + savename)
    plt.show()
    return


def display_reconstructed_image(recon, distance, savename):
    plt.figure()
    plt.imshow(recon, extent=[
        distance[0], distance[-1],
        distance[0], distance[-1]],
               cmap="gray", origin="lower")
    plt.colorbar(label="Reconstructed intensity")
    plt.title("Backprojection reconstruction")
    plt.ylabel("Distance (mm)")
    plt.xlabel("Distance (mm)")
    plt.savefig("figures/" + savename)
    plt.show()
    return

def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords
    # The exponent for a 2D Gaussian
    inner = ((x - x0)**2 / (2 * sigma_x**2)) + ((y - y0)**2 / (2 * sigma_y**2))
    return amplitude * np.exp(-inner) + offset

def double_gaussian_2d(coords, a1, x1, y1, sx1, sy1, a2, x2, y2, sx2, sy2, offset):
    # Sum of two independent 2D Gaussians
    return (gaussian_2d(coords, a1, x1, y1, sx1, sy1, 0) +
            gaussian_2d(coords, a2, x2, y2, sx2, sy2, 0) + offset)


def gaussian_1d(coords, amplitude, x0, sigma_x, offset):
    x = coords
    # The exponent for a 2D Gaussian
    inner = ((x - x0)**2 / (2 * sigma_x**2))
    return amplitude * np.exp(-inner) + offset



def singleSlit(coords, amplitude, x0, W, L):
    x = coords
    return np.maximum(amplitude * np.arctan((W/2 - np.abs(x-x0))/L), 0)

def circleConvolve(R,x,x0):
    return np.maximum(0,2*np.sqrt(R**2-(x-x0)**2))

def sphereConvolve(R,x,x0):
    return np.maximum(0,np.pi*(R**2-(x-x0)**2))

def doubleSlitModelConvoluted(xArr, Wslit, Rscint, Rsource, alpha, amplitude, x0, L, collection_time):
    distanceStep = xArr[1] - xArr[0]
    Wscint = 2 * Rscint
    background = 0.02359289617486339

    Xscint = np.arange(start=-Rscint + x0,stop=Rscint + x0,step=distanceStep)
    scintArr = circleConvolve(Rscint, Xscint, x0)
    scintSlitArr = circleConvolve(Rscint, Xscint, x0)
    scintSlitArr[0:np.round((Rscint-Wslit/2)/distanceStep).astype(int)] = 0
    scintSlitArr[np.round((Rscint+Wslit/2)/distanceStep).astype(int):len(scintArr)] = 0

    Xsource = np.arange(start=-Rsource + x0,stop=Rsource + x0,step=distanceStep)
    sourceArr = sphereConvolve(Rsource, Xsource, x0)
    # sourceArr = sourceArr/sum(sourceArr)

    XstackedSlit = np.arange(start=-Rscint + x0, stop=Rscint + x0, step=distanceStep)
    slitArr = singleSlit(XstackedSlit, amplitude, x0, Wslit, L) * (1-alpha)
    gateArr = singleSlit(XstackedSlit, amplitude, x0, Wscint, L) * alpha
    slitArr = np.multiply(slitArr, scintSlitArr)
    gateArr = np.multiply(gateArr, scintArr)
    slitGateCombined = gateArr + slitArr

    convolvedSignal = np.convolve(slitGateCombined, sourceArr, mode='full')
    convolvedX = np.arange(start=-len(convolvedSignal)/2, stop=len(convolvedSignal)/2, step=1) * 0.05 + x0
    convolvedSignal = convolvedSignal + background

    if (len(convolvedSignal) < len(xArr)):
        convolvedSignal = np.pad(convolvedSignal, np.round((len(xArr) - len(convolvedX))/2).astype(int), mode='constant' ,constant_values=background)
    elif (len(convolvedSignal) > len(xArr)):
        convolvedSignal = convolvedSignal[-len(xArr)/2:len(xArr)/2]

    if (len(convolvedSignal) < len(xArr)):
        convolvedSignal = np.append(convolvedSignal, convolvedSignal[-1])
    elif (len(convolvedSignal) > len(xArr)):
        convolvedSignal = np.delete(convolvedSignal, -1)

    return convolvedSignal * collection_time

def displayModelDataLinescan(Xarr, convolvedSignal, dist_array, counts_array, ucounts_array, savename):
    # Find the indices in model_x that are closest to data_x
    indices = np.searchsorted(Xarr, dist_array)

    # Note: searchsorted usually finds the index to the right.
    # You may need to clip to avoid index errors at the array boundaries.
    indices = np.clip(indices, 0, len(Xarr) - 1)

    residual = counts_array - convolvedSignal[indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(Xarr, convolvedSignal, label="Convolved Model", color='tab:orange')
    ax1.errorbar(dist_array, counts_array, yerr=ucounts_array, fmt='.',
                 label="Counts Data", alpha=0.6)
    # ax1.set_title("Model Approximation with Convolution")

    if (min(Xarr) < min(dist_array)) :
        ax1.set_xlim(min(dist_array)*1.1, max(dist_array)*1.1) # maybe need to remove
    else :
        ax1.set_xlim(min(Xarr)*1.1, max(Xarr)*1.1) # maybe need to remove


    ax1.set_xlabel("Distance (mm)", fontsize=14)
    ax1.set_ylabel("Counts", fontsize=14)
    ax1.text(0.05, 0.95, 'a', transform=ax1.transAxes, fontsize=16,
               color='black', fontweight='bold', va='top')
    ax1.legend()

    ax2.errorbar(dist_array, residual, yerr=ucounts_array, fmt='.', color='tab:red')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)  # Zero baseline
    # ax2.set_title("Model Residuals")
    ax2.set_xlabel("Distance (mm)", fontsize=14)
    ax2.set_ylabel("Counts", fontsize=14)
    ax2.text(0.05, 0.95, 'b', transform=ax2.transAxes, fontsize=16,
               color='black', fontweight='bold', va='top')


    plt.tight_layout()  # Prevents label overlap
    plt.savefig("./figures/" + savename, dpi=600)
    plt.show()
    return


def displayModelDataLinescanVert(Xarr, convolvedSignal, dist_array, counts_array, ucounts_array, savename):
    # Find the indices in model_x that are closest to data_x
    indices = np.searchsorted(Xarr, dist_array)

    # Note: searchsorted usually finds the index to the right.
    # You may need to clip to avoid index errors at the array boundaries.
    indices = np.clip(indices, 0, len(Xarr) - 1)

    residual = counts_array - convolvedSignal[indices]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7),gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(Xarr, convolvedSignal, label="Convolved Model", color='tab:orange')
    ax1.errorbar(dist_array, counts_array, yerr=ucounts_array, fmt='.',
                 label="Counts Data", alpha=0.6)
    # ax1.set_title("Model Approximation with Convolution")
    # ax1.set_xlabel("Distance (mm)", fontsize=14)
    ax1.set_ylabel("Counts", fontsize=14)
    # ax1.set_xlim(min(dist_array)*1.2, max(dist_array)*1.2)

    if (min(Xarr) < min(dist_array)) :
        ax1.set_xlim(min(dist_array)*1.1, max(dist_array)*1.1) # maybe need to remove
    else :
        ax1.set_xlim(min(Xarr)*1.1, max(Xarr)*1.1) # maybe need to remove

    ax1.text(0.05, 0.95, 'a', transform=ax1.transAxes, fontsize=16,
               color='black', fontweight='bold', va='top')
    ax1.legend()

    ax2.errorbar(dist_array, residual, yerr=ucounts_array, fmt='.', color='tab:red')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)  # Zero baseline
    # ax2.set_title("Model Residuals")
    ax2.set_xlabel("Distance (mm)", fontsize=14)
    ax2.set_ylabel("Counts", fontsize=14)
    ax2.text(0.05, 0.95, 'b', transform=ax2.transAxes, fontsize=16,
               color='black', fontweight='bold', va='top')


    plt.tight_layout()  # Prevents label overlap
    plt.savefig("./figures/" + savename, dpi=600)
    plt.show()
    return


def get_transmission_efficiency(source_x, slit_width_W, attenuation_len, n_rays=100000, R_scint=35):
    # --- Constants ---
    Z_gate_start = 20.0
    Z_gate_end = 35.0
    Z_scint = 70
    X_gate_left = -slit_width_W / 2.0
    X_gate_right = slit_width_W / 2.0
    source_x = np.where(((source_x == X_gate_left) | (source_x == X_gate_right)), source_x - 0.00001, source_x)

    np.random.seed(0)

    phi = np.random.uniform(0, 2 * np.pi, n_rays)
    costheta = np.random.uniform(0, 1, n_rays)
    sintheta = np.sqrt(1 - costheta ** 2)

    ux = np.sin(phi) * costheta
    uy = np.sin(phi) * sintheta
    uz = np.cos(phi)
    tanphi = np.tan(phi)

    x_at_scint = source_x + Z_scint * tanphi * costheta
    y_at_scint = Z_scint * tanphi * sintheta
    x_at_scintMirrored = source_x - Z_scint * tanphi * costheta
    y_at_scintMirrored = - Z_scint * tanphi * sintheta

    hit_mask = ((x_at_scint ** 2 + y_at_scint ** 2) <= R_scint ** 2) & (
                (x_at_scintMirrored ** 2 + y_at_scintMirrored ** 2) <= R_scint ** 2)

    ux, uy, uz = ux[hit_mask], uy[hit_mask], uz[hit_mask]
    if len(uz) == 0: return 0.0

    z_cross_left = (X_gate_left - source_x) * uz / ux
    z_cross_right = (X_gate_right - source_x) * uz / ux
    z_gap_1 = np.minimum(z_cross_left, z_cross_right)
    z_gap_2 = np.maximum(z_cross_left, z_cross_right)

    # visualize_rays(source_x,slit_width_W, ux,uz,z_gap_1, z_gap_2)

    mask_full = (z_gap_2 < 0) | (z_gap_1 > Z_gate_end) | ((z_gap_1 < Z_gate_start) & (z_gap_1 > 0) & (z_gap_2 < 0)) | (
                (z_gap_1 < Z_gate_start) & (z_gap_2 < Z_gate_start) & (z_gap_1 > 0)) | (z_gap_2 < Z_gate_start) & (
                            z_gap_1 < 0) & (z_gap_2 > 0)
    # ^ Entirely through lead (Front-to-Back)
    mask_side = ((z_gap_1 > Z_gate_start) & (z_gap_1 < Z_gate_end)) | (
                (z_gap_2 > Z_gate_start) & (z_gap_2 < Z_gate_end))
    # ^ mask for all rays that enter/exit at least one inside edge of a gate
    mask_both = (z_gap_1 > Z_gate_start) & (z_gap_2 < Z_gate_end)
    # ^mask for rays that enter and exit both gates

    air_start = np.maximum(z_gap_1, Z_gate_start)
    air_end = np.minimum(z_gap_2, Z_gate_end)
    z_air_thick = np.maximum(0, air_end - air_start)

    d_lead = np.zeros_like(uz)
    d_lead = np.where(mask_full, (Z_gate_end - Z_gate_start) / np.abs(uz), d_lead)
    d_lead = np.where(mask_side | mask_both, (Z_gate_end - Z_gate_start - z_air_thick) / np.abs(uz), d_lead)

    weights = np.exp(-d_lead / attenuation_len)

    """Now I calculate the weights for the rays going in the opposite direction"""

    Z_gate_start = -20.0
    Z_gate_end = -35.0

    z_gap_1_mirrored = np.maximum(z_cross_left, z_cross_right)
    z_gap_2_mirrored = np.minimum(z_cross_left, z_cross_right)

    mask_full = (z_gap_2_mirrored > 0) | (z_gap_1_mirrored < Z_gate_end) | (
                (z_gap_1_mirrored > Z_gate_start) & (z_gap_1_mirrored < 0) & (z_gap_2_mirrored > 0)) | (
                            (z_gap_1_mirrored > Z_gate_start) & (z_gap_2_mirrored > Z_gate_start) & (
                                z_gap_1_mirrored < 0)) | (z_gap_2_mirrored > Z_gate_start) & (z_gap_1_mirrored > 0) & (
                            z_gap_2_mirrored < 0)
    # ^ Entirely through lead (Front-to-Back)
    mask_side = ((z_gap_1_mirrored < Z_gate_start) & (z_gap_1_mirrored > Z_gate_end)) | (
                (z_gap_2_mirrored < Z_gate_start) & (z_gap_2_mirrored > Z_gate_end))
    # ^ mask for all rays that enter/exit at least one inside edge of a gate
    mask_both = (z_gap_1_mirrored < Z_gate_start) & (z_gap_2_mirrored > Z_gate_end)
    # ^mask for rays that enter and exit both gates

    air_start = np.maximum(z_gap_1_mirrored, Z_gate_start)
    air_end = np.minimum(z_gap_2_mirrored, Z_gate_end)
    z_air_thick = np.maximum(0, air_end - air_start)
    d_lead = np.zeros_like(uz)
    d_lead = np.where(mask_full, (Z_gate_start - Z_gate_end) / np.abs(uz), d_lead)
    d_lead = np.where(mask_side | mask_both, (Z_gate_start - Z_gate_end - z_air_thick) / np.abs(uz), d_lead)

    weights_reversed = np.exp(-d_lead / attenuation_len)

    normalizedWeight = np.sum(weights * weights_reversed) / n_rays

    return normalizedWeight

def monte_carlo_model(source_x_arr, x0, slitW, attenuation_len, A, source_R, exposure_time, background):
    distanceStep = source_x_arr[1] - source_x_arr[0]
    A = A * distanceStep
    # x_scan = source_x_arr[0]
    transmission = []

    for x in source_x_arr:
        val = get_transmission_efficiency(x-x0, slit_width_W=slitW, attenuation_len=attenuation_len, n_rays=50000, R_scint=35)
        transmission.append(val)

    # background = 0.00611 # times exposure time

    Signal_Modelled = A*np.array(transmission) * exposure_time

    Xsource = np.arange(start=-source_R,stop=source_R,step=distanceStep)
    sourceArr = sphereConvolve(source_R, Xsource, 0)

    convolvedSignal = np.convolve(Signal_Modelled, sourceArr, mode='full')
    convolvedSignal = convolvedSignal + background * exposure_time #+ background


    if (len(convolvedSignal) < len(source_x_arr)):
        convolvedSignal = np.pad(convolvedSignal, np.round((len(source_x_arr) - len(convolvedSignal))/2).astype(int), mode='constant' ,constant_values=background)
    elif (len(convolvedSignal) > len(source_x_arr)):
        convolvedSignal = convolvedSignal[len(convolvedSignal)//2-len(source_x_arr)//2:len(convolvedSignal)//2 + len(source_x_arr)//2]
    # May need to do additional checks of inequality (in case of off-by-1 error)

    if (len(convolvedSignal) < len(source_x_arr)):
        convolvedSignal = np.append(convolvedSignal, convolvedSignal[-1])
    elif (len(convolvedSignal) > len(source_x_arr)):
        convolvedSignal = np.delete(convolvedSignal, -1)

    return convolvedSignal


