import argparse
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def normalize_standard_score(df, axis):
    # type: (pd.DataFrame, int) -> pd.DataFrame

    # Make sure the std of any class doesn't have any zero values
    df_std = df.std(axis=axis)
    df_std[df_std == 0] = 1.0

    if axis == 0:
        df_norm = (df-df.mean())/df_std
    elif axis == 1:
        df_norm = df.sub(df.mean(axis=1), axis=0)
        df_norm = df_norm.div(df_std, axis=0)
    else:
        raise ValueError('Invalid axis: {}'.format(axis))

    return df_norm


def normalize_feature_scaling(df, axis):
    # type: (pd.DataFrame, int) -> pd.DataFrame

    df_max = df.max(axis=axis)
    df_min = df.min(axis=axis)
    df_denom = df_max - df_min

    # Make sure the denom value of any class doesn't have any zero values
    df_denom[df_denom == 0] = 1.0

    if axis == 0:
        df_norm = (df - df_min) / df_denom
    elif axis == 1:
        df_norm = df.sub(df_min, axis=0)
        df_norm = df_norm.div(df_denom, axis=0)
    else:
        raise ValueError('Invalid axis: {}'.format(axis))

    return df_norm


def normalize_zero_one(df, axis):
    # type: (pd.DataFrame, int) -> pd.DataFrame

    if axis == 0:
        df_norm = df / df.sum()
    elif axis == 1:
        df_norm = df.div(df.sum(axis=axis), axis=0)
    else:
        raise ValueError('Invalid axis: {}'.format(axis))

    return df_norm


def normalize_data_frame(df, normalization_mode='row', normalization_type='none', decimals=2):
    # type: (pd.DataFrame, str, str, int) -> pd.DataFrame

    # Normalize by subtracting the mean and dividing by the standard deviation
    if normalization_type == 'none':
        return df

    axis = 1 if normalization_mode == 'row' else 0
    df_norm = df

    if normalization_type == 'ss':
        df_norm = normalize_standard_score(df, axis=axis)
    elif normalization_type == 'fs':
        df_norm = normalize_feature_scaling(df, axis=axis)
    elif normalization_type == 'zo':
        df_norm = normalize_zero_one(df, axis=axis)
    else:
        raise ValueError('Invalid normalization type: {}'.format(normalization_type))

    if decimals is not None:
        df_norm = df_norm.round(decimals=decimals)
    return df_norm


def main():
    # Construct the argument parser and parge arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfm", required=True, type=str, help="Path to the confusion matrix file")
    ap.add_argument("--cfm2", required=False, type=str, help="Path to the comparison confusion matrix file")
    ap.add_argument("--normmode", required=False, type=str, default='row', choices=['row', 'col'], help="Normalization mode for the data per row or per column")
    ap.add_argument("--normtype", required=False, type=str, default='zo', choices=['ss', 'fs', 'zo'], help="Normalization type for the data standard score (ss), feature scaling (fs), zero-one (zo)")
    ap.add_argument("--decimals", required=False, type=int, default=2, help="Number of decimals points to use in the displayed values")
    ap.add_argument("--cbar", required=False, type=bool, default=False, help="Show color bar in the figure?")
    ap.add_argument("--cmap", required=False, type=str, default="coolwarm", help='Colormap name for seaborn')
    ap.add_argument("--ignore", required=False, type=str, default="", help="Ignore classes e.g. 0,1,2")
    ap.add_argument("--out", required=False, type=str, help="Path to a file where the figure should be saved")
    args = vars(ap.parse_args())

    class_names = ['Background', 'Brick', 'Carpet', 'Ceramic', 'Other', 'Fabric', 'Foliage', 'Food', 'Glass',
                   'Polished stone', 'Stone', 'Hair', 'Leather', 'Metal', 'Mirror', 'Painted', 'Paper', 'Plastic',
                   'Skin', 'Sky', 'Tile', 'Wallpaper', 'Water', 'Wood']

    path_to_cfm = args['cfm']
    path_to_comparison_cfm = args['cfm2']
    normalization_mode = args['normmode']
    normalization_type = args['normtype']
    decimals = args['decimals']
    show_cbar = args['cbar']
    cmap_name = args['cmap']
    ignore_classes = args['ignore']
    output_path = args['out']
    comparison_mode = True if path_to_comparison_cfm is not None else False

    print 'Reading cfm from path: {}'.format(path_to_cfm)
    np_cfm = np.loadtxt(path_to_cfm)
    np_cfm_diff = None

    if path_to_comparison_cfm is not None:
        print 'Reading comparison cfm from path: {}'.format(path_to_comparison_cfm)
        np_cfm_diff = np.loadtxt(path_to_comparison_cfm)

    ignore_classes = [int(x) for x in ignore_classes.strip().split()]
    print 'Removing ignored classes from data and labels: {}'.format(ignore_classes)

    for class_idx in ignore_classes:
        np_cfm = np.delete(np_cfm, obj=class_idx, axis=0)
        np_cfm = np.delete(np_cfm, obj=class_idx, axis=1)

        if np_cfm_diff is not None:
            np_cfm_diff = np.delete(np_cfm_diff, obj=class_idx, axis=0)
            np_cfm_diff = np.delete(np_cfm_diff, obj=class_idx, axis=1)

        del class_names[class_idx]

    df_cfm = pd.DataFrame(np_cfm, index=[i for i in class_names], columns=[i for i in class_names])
    df_cfm = normalize_data_frame(df_cfm,
                                  normalization_mode=normalization_mode,
                                  normalization_type=normalization_type,
                                  decimals=decimals if not comparison_mode else None)

    if np_cfm_diff is not None:
        print 'Creating the difference normal map: cfm2 - cfm'
        df_cfm_diff = pd.DataFrame(np_cfm_diff, index=[i for i in class_names], columns=[i for i in class_names])
        df_cfm_diff = normalize_data_frame(df_cfm_diff,
                                           normalization_mode=normalization_mode,
                                           normalization_type=normalization_type,
                                           decimals=decimals if not comparison_mode else None)
        df_cfm = df_cfm_diff - df_cfm

        # Round in the end
        df_cfm = df_cfm.round(decimals=decimals)
        df_cfm[df_cfm == 0] = 0

    fig = plt.figure(figsize=(10, 10))
    hm = sn.heatmap(df_cfm, annot=True, annot_kws={"size": 8}, cmap=cmap_name, cbar=show_cbar)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0, fontsize=8)
    hm.set_xticklabels(hm.get_xticklabels(), rotation=90, fontsize=8)

    if output_path is not None:
        print 'Saving figure to: {}'.format(output_path)
        fig.savefig(output_path, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()
