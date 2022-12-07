import numpy as np
import pandas as pd
import pathlib as p
import functools as f
import matplotlib.pyplot as plt
import featureset_specification as datum

from matplotlib import rc

# Activate latex text rendering
rc("text", usetex=True)


def main():
    construct_image(datum.TIERS_SET[0])
    construct_image(datum.TIERS_SET[1])


def construct_image(n_classes):
    df = datum.dataset_read()
    df = datum.bin_labels_into_tiers(df, n_tiers=n_classes)

    plt.figure(figsize=(15, 7), dpi=120)

    # Draw histogram with granular binning
    df[datum.COLUMN_SCORE].plot.hist(bins=163)

    # Agregate
    tier_ranges = df.groupby(datum.COLUMN_CLASS, as_index=False).agg(
        MIN=(datum.COLUMN_SCORE, "min"), MAX=(datum.COLUMN_SCORE, "max")
    )

    legend_lines = []
    prior_MAX = None
    for tier, bound in tier_ranges.iterrows():
        # Draw the seperator between tier classes
        # (except on the first iteration)
        if prior_MAX is not None:
            is_last = tier + 1 == len(tier_ranges)
            seperator = (prior_MAX + bound["MIN"]) / 2
            chromatic = (0.9, 0.4, 0.4, 0.9) if is_last else (0.2, 0.2, 0.2, 0.8)
            linelabel = "Manual threshold" if is_last else "Clustering parition"
            line = plt.axvline(
                x=seperator,
                color=chromatic,
                label=linelabel,
                linestyle=(5, (10, 3)),
                linewidth=1.5,
            )

            if legend_lines == []:
                legend_lines.append(line)

            if is_last:
                legend_lines.append(line)

        prior_MAX = bound["MAX"]

    plt.suptitle("Distribution of Elo Rankings in Dataset")
    plt.title("Partitioning with " + r"$\textbf{" + str(n_classes) + "}$ Tiers")
    plt.ylabel(r"$\textsc{Quantity}$")
    plt.xlabel(r"$\textsc{Elo Rank}$")
    plt.xticks(np.linspace(-1900, 6200, 28), rotation=60, ha="right")
    plt.legend(handles=legend_lines)
    plt.savefig(
        "Elo-Tier-Partition-" + str(n_classes),
        dpi="figure",
        format="png",
        pad_inches=0,
        transparent=True,
    )


if __name__ == "__main__":
    main()
