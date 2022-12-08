import numpy as np
import pandas as pd
import pathlib as p
import functools as f

from copy import deepcopy
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.preprocessing import (
    LabelBinarizer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    KBinsDiscretizer,
)

TIER_LABELING = lambda x: ["Trivial"] + list(x) + ["Cosmic"]
TIER_COMPRESS = TIER_LABELING(
    [str(i) + "║" + str(j) for i, j in zip(range(1, 20, 2), range(2, 21, 2))]
)
TIER_STANDARD = TIER_LABELING([str(i) for i in range(1, 21)])

# Officially published Challenge Rating (CR) scores:
# | Label      | Rank | Categories |
# | Offical CR |  31  | (1/8), (1/4), (1/2), 1, 2, .. 20, 21, 22, 23, 24, 25, 26, 28, 30
#
# | Label      | Rank | Categories                                     | CR Equivelant
# | Coarse     |   4  | Trivial, Heroic, Paragon, Epic                 | (< 1),      1-10,            11-20, (20 <)
# | HeroTier   |   7  | Trivial, Local, Regional, Realm, World, Cosmic | (< 1), 1-6, 5-10,     11-16, 17-20, (20 <)
# | Compressed |  12  | Trivial, 1, 2, ...,  9, 10, Cosmic             | (< 1), 1-2, 3-4, ..., 17-18, 19-20, (20 <)
# | Equivalent |  22  | Trivial, 1, 2, ..., 19, 20, Cosmic             | (< 1), 1, 2,  3, ..., 18,  19,  20, (20 <)
#
# TIERS_SET = [4, 7, 11, 22]
TIERS_SET = [TIER_COMPRESS, TIER_STANDARD]

COLUMN_CLASS = "Tier"
COLUMN_SCORE = "Elo Rank"
COLUMN_TITLE = "Name"
COSMIC_BOUND = 4700


default_feature_specification = {
    "decorrelate": 0.75,
    "n_classes": TIERS_SET[0],
    "textual": False,
}


def dataset_path():
    absPath = p.Path(__file__).parent.parent.resolve()
    return absPath.parent.joinpath("data", "dnd-5e-monsters.csv")


@f.lru_cache(maxsize=1)
def dataset_read():
    df = pd.read_csv(dataset_path(), sep=",")
    df = df.sort_values(by="Trait Tags", ascending=False, key=lambda x: x.str.len())
    df = df.drop_duplicates(subset=COLUMN_TITLE, keep="first")
    df = df.sort_values(by=COLUMN_SCORE, ascending=False).reset_index(drop=True)
    df = feature_extraction(df)
    return df


@f.lru_cache(maxsize=1)
def retrieve_monster_dataset(decorrelate=None, n_classes=TIERS_SET[0], textual=False):
    absPath = p.Path(__file__).parent.parent.resolve()
    dataset = dataset_read()
    dataset = bin_labels_into_tiers(dataset, n_tiers=n_classes)
    dataset = feature_expunging(dataset, decorrelate, textual)
    return dataset


def feature_extraction(df):
    df = inclusionBitEncodeColumn(df, "Damage Tags", "Damage")
    df = inclusionBitEncodeColumn(df, "Spellcasting Tags", "Spellcasting")
    df = inclusionBitEncodeColumn(df, "Trait Tags", "")
    df = compress_dataset(df)
    return df


def feature_expunging(df, decorrelate=None, textual=False):

    if decorrelate is not None:
        df = decorrelate_columns(df, threshold=decorrelate)

    if not textual:
        df = dropTextualColumns(df)

    df.drop(COLUMN_SCORE, 1, inplace=True)

    return df


def decorrelate_columns(df, threshold=0.6):
    corr_set = list()

    for rName, row in df.iloc[:, :-1].corr().iterrows():
        for cName, col in row.iteritems():
            if rName > cName and col >= 0.6:
                cL = list(filter(lambda x: cName in x, corr_set))
                cV = cL[0] if len(cL) > 0 else None
                rL = list(filter(lambda x: rName in x, corr_set))
                rV = rL[0] if len(rL) > 0 else None
                if cV is None and rV is None:
                    corr_set.append({cName, rName})
                elif cV is None and rV is not None:
                    rV.add(cName)
                elif cV is not None and rV is None:
                    cV.add(rName)
                elif cV is not None and rV is not None:
                    corr_set.remove(rV)
                    cV |= rV

    for proxies in corr_set:
        # Keep column from proxies with widest range of values
        kept = None
        best = 0
        for col in proxies:
            width = df[col].max() - df[col].min()
            if width > best:
                best = width
                kept = col

        proxies.remove(kept)

        for extra in proxies:
            if extra in df.columns:
                df.drop(extra, 1, inplace=True)

    return df


# Make dataset up less than 19% of the original space!
def compress_dataset(df):

    # Unsigned, large-valued columns
    uWide = {"Hit Points"}
    # Signed,   large-valued columns
    sWide = {COLUMN_SCORE}
    # Textual content columns
    texty = {
        "Damage Tags",
        "Name",
        "Spellcasting Tags",
        "Trait Tags",
        "Type",
    }

    for col in list(df.columns):

        # If the column contains textual data, skip it!
        if col in texty:
            continue

        # If the column can contain potentially large values,
        # and the values can be negative, truncate the bits!
        elif col in sWide:
            setType(df, col, np.int16)

        # If the column can contain potentially large values,
        # and the values are non-negative, truncate the bits!
        elif col in uWide:
            setType(df, col, np.uint16)

        # If the column is not textual and cannot contain large values,
        # then it is a tiny column, turncate as many bits as possible@
        else:
            setType(df, col, np.uint8)

    return df


def oneHotBitEncodeColumn(df, colName, prefix=None):
    if colName not in df.columns:
        return
    spot = df.columns.get_loc(colName)
    cols = pd.get_dummies(df[colName], prefix)
    ordered = sorted(cols.columns)
    ordered.reverse()
    for name in ordered:
        df.insert(spot, name, cols[name])
    df.drop(colName, 1, inplace=True)
    return df


def inclusionBitEncodeColumn(df, colName, prefix=None):
    if colName not in df.columns:
        return
    values = pd.DataFrame(df[colName].values.tolist()).stack().values
    uniques = f.reduce(lambda a, b: set(b.split(",")).union(a), values, set())
    colSpot = df.columns.get_loc(colName)
    ordered = sorted(uniques)
    ordered.reverse()
    for val in ordered:
        col = df[colName].map(lambda x: val in set(str(x).split(",")))
        pref = colName
        if prefix is not None:
            pref = prefix
        if prefix != "":
            pref += "_"
        df.insert(colSpot, pref + val, col.astype(np.uint8))
    df.drop(colName, axis=1, inplace=True)
    return df


# Unused, instead favoring 'bin_labels_into_tiers'
def standarize_data_set(df, colName=COLUMN_SCORE, class_count=5):

    input_column = df[colName].to_numpy().reshape(-1, 1)

    # Standard normalization:
    scaler = QuantileTransformer(n_quantiles=class_count, output_distribution="normal")
    #    scaler = QuantileTransformer(output_distribution='normal')
    #    scaler = RobustScaler(quantile_range=(20,70), unit_variance=True)
    #    scaler = StandardScaler()
    df[colName] = scaler.fit_transform(input_column)

    # Scaling factor
    factor = (class_count - 1) / 2

    # Scale normalized column data
    df.loc[:, colName] *= factor
    df.loc[:, colName] += factor

    # Discretize via rounding
    df[colName] = df[colName].apply(round)
    setType(df, colName, np.uint8)

    values = list(df[colName].value_counts().items())
    values.sort()
    extras = 0
    for k, v in values:
        print(k, "\t", v)
        if k >= class_count:
            extras += v
    print("extra:\t", extras)

    # Should no longer need to do this
    df.drop(df.loc[df[colName] >= class_count].index, inplace=True)

    return df


def bin_labels_into_tiers(df, n_tiers=5):

    # Process the "input scores," i.e.
    # the column containing the creature Elo rankings.
    #
    # Partition the scores by the "cosmic bound" threshold.
    screen = df[COLUMN_SCORE] < COSMIC_BOUND
    gather = lambda v: df[screen == v].copy()
    cosmic = gather(False)
    inputs = gather(True)

    # Extract the "input scores" column as NumPy array from Pandas DataFrame.
    scores = lambda v: deepcopy(v[COLUMN_SCORE].to_numpy().reshape(-1, 1))

    # Define how to transform the label column into tier classes.
    # Use K-means to bin the Elo rankings into tiers.
    # Select n - 1 bins, the nth bin is the "cosmic" tier.
    discrete_spec = {"n_bins": n_tiers - 1, "encode": "ordinal", "strategy": "kmeans"}
    discrete_binned = (
        KBinsDiscretizer(**discrete_spec).fit_transform(scores(inputs)).astype(np.uint8)
    )
    discrete_inputs = inputs
    discrete_inputs[COLUMN_CLASS] = discrete_binned

    # Debugging for permutaiton of ordinals
    #
    #    print("\nMean Ranks:\n", discrete_inputs.groupby(COLUMN_CLASS, as_index=False).agg({COLUMN_SCORE: "mean"}).sort_values(by=COLUMN_SCORE))
    #    permutation_data = discrete_inputs.groupby(COLUMN_CLASS, as_index=False).agg({COLUMN_SCORE: "mean"}).sort_values(by=COLUMN_SCORE)[COLUMN_CLASS].to_numpy().astype(np.uint8)
    #    permutation_spec = np.zeros(len(permutation_data)).astype(np.uint8)
    #    for k, v in enumerate(permutation_data):
    #        permutation_spec[v] = k
    #    print("\nArrangement:\n", permutation_data)
    #    print("\nPermutation:\n", permutation_spec)
    #    permutation_func = lambda row: permutation_spec[ row[COLUMN_CLASS] ]
    #    discrete_inputs[COLUMN_CLASS] = discrete_inputs.apply(permutation_func, axis=1)
    #    print("\nMean Ranks:\n", discrete_inputs.groupby(COLUMN_CLASS, as_index=False).agg({COLUMN_SCORE: "mean"}).sort_values(by=COLUMN_SCORE))

    buffered_cosmic = scores(cosmic)
    buffered_cosmic[:] = n_tiers - 1
    discrete_cosmic = cosmic
    discrete_cosmic[COLUMN_CLASS] = buffered_cosmic.astype(np.uint8)

    discrete_labels = pd.concat(
        [discrete_inputs, discrete_cosmic], axis=0, ignore_index=True, sort=False
    )

    return discrete_labels


def dropTextualColumns(df):
    return df.select_dtypes(exclude=["object"])


def dropColumns_Names(df, names):
    for name in df.columns:
        if name in names:
            df.drop(name, 1, inplace=True)


def setType(df, colName, colType):
    df[colName] = df[colName].astype(colType)
