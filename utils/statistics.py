import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def ceil_to_25(x, base=25):
    if x >= 0:
        rounded = base * np.ceil(x/base)
    else:
        rounded = base * np.floor(x/base)

    return rounded


def histogram_error(diffs, min_bin, max_bin, bin_range, digit_size, text_size):
    try:
        bins = list(np.arange(min_bin, max_bin + (bin_range/10), bin_range/10))
        counts, bins, patches = plt.hist(diffs, bins)
        plt.xticks(range(int(min_bin), int(max_bin) + int(bin_range/10), int(bin_range/10)), fontsize=digit_size)
        plt.yticks(range(0, int(np.max(counts)+10), int(np.max(counts)/10)), fontsize=digit_size)
    except:
        plt.xticks(fontsize=digit_size)
        plt.yticks(fontsize=digit_size)
    plt.grid(axis='y', alpha=0.75)
    plt.title("Diameter error from the ground truth", fontsize=text_size)
    plt.xlabel("Diameter error (mm)", fontsize=text_size)
    plt.ylabel("Frequency", fontsize=text_size)

    try:
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        for count, x in zip(counts, bin_centers):
            if count < 10 :
                plt.annotate('n={:.0f}'.format(count), (x-3, count+2))
            elif count < 100:
                plt.annotate('n={:.0f}'.format(count), (x-4, count+2))
            else:
                plt.annotate('n={:.0f}'.format(count), (x-5, count+2))
        plt.show()
    except:
        plt.show()


def scatterplot_occlusion(diffs, vprs, max_bin, digit_size, text_size):
    occlusion_perc =  [(1-ele)*100 for ele in vprs]
    diffs_abs =  [abs(ele) for ele in diffs]
    plt.plot(occlusion_perc, diffs_abs, 'o', color='blue', alpha=0.75)
    plt.xticks(range(0, 110, 10), fontsize=digit_size)
    try:
        plt.yticks(range(0, int(max_bin), int(max_bin/10)), fontsize=digit_size)
    except:
        plt.yticks(fontsize=digit_size)
    plt.title("Diameter error as a function of the occlusion rate", fontsize=text_size)
    plt.xlabel("Occlusion rate (%)", fontsize=text_size)
    plt.ylabel("Absolute error on diameter (mm)", fontsize=text_size)
    plt.show()


def scatterplot_size(diffs, gtsizes, max_bin, digit_size, text_size):
    diffs_abs =  [abs(ele) for ele in diffs]
    plt.plot(gtsizes, diffs_abs, 'o', color='blue', alpha=0.75)
    plt.xticks(range(50,275,25), fontsize=digit_size)
    try:
        plt.yticks(range(0, int(max_bin), int(max_bin/10)), fontsize=digit_size)
    except:
        plt.yticks(fontsize=digit_size)
    plt.title("Diameter error as a function of the broccoli size", fontsize=text_size)
    plt.xlabel("Ground truth size of the broccoli head (mm)", fontsize=text_size)
    plt.ylabel("Absolute error on diameter (mm)", fontsize=text_size)
    plt.show()


def boxplot_time(num_images, inference_times, digit_size, text_size):
    sns.set_style("ticks")
    df = pd.DataFrame(inference_times, columns=["time"])
    f, ax = plt.subplots(figsize=(11, 2.5))
    ax = sns.boxplot(data=df["time"], orient="h", palette="colorblind")
    plt.yticks([])
    plt.title("Image analysis times when sizing {0:.0f} broccoli heads".format(num_images), fontsize=text_size)
    plt.xticks(fontsize=digit_size)
    plt.xlabel('Image analysis time (s)', fontsize=text_size)
    plt.tight_layout()
    plt.show()