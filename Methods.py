
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
        counts_uncertainty = count_uncertainty(counts_array)
        # -----------------------------------

        if width not in grouped_data:
            grouped_data[width] = []

        # Add a tuple of the two arrays to the list for this metric
        grouped_data[width].append((dist_array, counts_array, counts_uncertainty))

    return grouped_data

def count_uncertainty(count_arr):
    count_arr_uncertainty = []
    N = sum(count_arr)
    for count in count_arr:
        p_i = count / N
        uncertainty = np.sqrt(count * p_i * (1-p_i))
        count_arr_uncertainty.append(uncertainty)
    return count_arr_uncertainty


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
    plt.savefig(savename)
    plt.show()

def display_FWHM(grouped_fwhm):

    for width in sorted(grouped_fwhm.keys()):
        data_list = grouped_fwhm[width]

        # data_list is a list of tuples: (distance_array, counts_array)
        for fwhm, r0, r1 in data_list:
            print(f"with W={width}, fwhm={fwhm}mm, from {r0} to {r1}mm")



def normalize(grouped_data):
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

def count_uncertainty(count_arr):
    count_arr_uncertainty = []
    N = sum(count_arr)
    for count in count_arr:
        p_i = count / N
        uncertainty = np.sqrt(count * p_i * (1-p_i))
        count_arr_uncertainty.append(uncertainty)
    return count_arr_uncertainty

def plotLineScan(dist, counts, ucounts, savename):
    plt.figure()
    plt.title("Sweeped Counts")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Counts")

    # data_list is a list of tuples: (distance_array, counts_array)
    plt.errorbar(dist, counts, fmt='-o', yerr = ucounts)
    # plt.plot(dist, counts, label=f"counts for width = {metric}mm")

    plt.savefig(savename)
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
    counts_uncertainty = count_uncertainty(counts_array)
    return (dist_array, counts_array, counts_uncertainty)


# Expects all arrays to be of the same length, this comes automatically from the PET lab collection
def readSinogram(directory):
    angle_pattern = re.compile(r"(-?\d+_\d+)\s*Deg")
    angle_to_counts = {}
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
        elif (not (distances_ref == data["Distance"].to_numpy()).all()):
            raise ValueError("Distance grid mismatch in file: " + fname)
        # print(distances_ref)

        angle_to_counts[angle] = data["Counts"].to_numpy()

    angles = np.array(sorted(angle_to_counts.keys()))
    # print(angles)
    # Build 2D array: rows = distance, cols = angle
    counts_2d = np.column_stack([angle_to_counts[a] for a in angles])

    midpoint = (distances_ref[-1] - distances_ref[0]) / 2
    distances_ref = distances_ref - midpoint
    # print(distances_ref)
    return counts_2d, distances_ref, angles


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
    plt.savefig(savename)
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
    plt.savefig(savename)
    plt.show()
    return
