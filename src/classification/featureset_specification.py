import numpy as np
import pandas as pd
import pathlib as p
import functools as f

from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import (
    LabelBinarizer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    KBinsDiscretizer,
)


default_feature_specification = {
    "tagged_trait": True,
    "standardized_label_classes": 5,
    "decorrelate": 0.75,
}

TIERS_SET = [4, 5, 6, 7, 8]


@f.lru_cache(maxsize=1)
def retrieve_monster_dataset(
    compress=True,
    textual=False,
    tagged_damage=False,
    tagged_spell=False,
    tagged_trait=False,
    standardized_label_classes=None,
    decorrelate=None,
):
    absPath = p.Path(__file__).parent.parent.resolve()
    dataset = pd.read_csv(
        absPath.parent.joinpath("data", "dnd-5e-monsters.csv"), sep=","
    )

    if tagged_damage:
        dataset = inclusionBitEncodeColumn(dataset, "Damage Tags", "Damage")

    if tagged_spell:
        dataset = inclusionBitEncodeColumn(dataset, "Spellcasting Tags", "Spellcasting")

    if tagged_trait:
        dataset = inclusionBitEncodeColumn(dataset, "Trait Tags", "")

    if bin_labels_into_tiers is not None:
        dataset = bin_labels_into_tiers(dataset, class_count=standardized_label_classes)

    if decorrelate is not None:
        dataset = decorrelate_columns(dataset, threshold=decorrelate)

    if not textual:
        dataset = dropTextualColumns(dataset)

    if compress:
        dataset = compress_dataset(dataset)

    return dataset


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
def compress_dataset(dataset):
    # Unsigned, large-valued columns
    uWide = {"Hit Points"}
    # Signed,   large-valued columns
    sWide = {}  # {"Elo Rank"}
    # Textual content columns
    texty = {"Damage Tags", "Spellcasting Tags", "Trait Tags", "Name", "Type"}

    for col in list(dataset.columns):
        # If the column contains textual data, skip it!
        if col in texty:
            continue

        # If the column can contain potentially large values,
        # and the values are non-negative, truncate the bits!
        if col in uWide:
            setType(dataset, col, np.uint16)
        # If the column can contain potentially large values,
        # and the values can be negative, truncate the bits!
        elif col in sWide:
            setType(dataset, col, np.int16)
        # If the column is not textual and cannot contain large values,
        # then it is a tiny column, turncate as many bits as possible@
        else:
            setType(dataset, col, np.uint8)

    return dataset


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
    df.drop(colName, 1, inplace=True)
    return df


# Unused, instead favoring 'bin_labels_into_tiers'
def standarize_data_set(df, colName="Elo Rank", class_count=5):

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


def bin_labels_into_tiers(df, colName="Elo Rank", class_count=5):

    input_column = df[colName].to_numpy().reshape(-1, 1)

    # Define how to transform the label column into tier classes.
    # Use K-means to bin the Elo rankings into tiers.
    discretizer = KBinsDiscretizer(
        n_bins=class_count, encode="ordinal", strategy="kmeans"
    )

    # Apply the Discretizer to label column
    df[colName] = discretizer.fit_transform(input_column)

    # Compress the reprsesentation size of label column
    setType(df, colName, np.uint8)

    # Debug printing of label bins
    # print(df[colName].value_counts())

    return df


def dropTextualColumns(df):
    return df.select_dtypes(exclude=["object"])


def dropColumns_Names(df, names):
    for name in df.columns:
        if name in names:
            df.drop(name, 1, inplace=True)


def setType(df, colName, colType):
    df[colName] = df[colName].astype(colType)
