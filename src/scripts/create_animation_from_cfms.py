import argparse
import numpy as np
from os import listdir, path


def get_all_txt_files(directory_path):
    files = [path.join(directory_path, f) for f in listdir(directory_path) if path.isfile(path.join(directory_path, f))]
    txt_files = [f for f in files if f.endswith('.txt')]
    return txt_files


def get_accuracy(np_cfm):
    total_samples = np.sum(np.sum(np_cfm, axis=0))
    correct_samples = np.sum(np.diagonal(np_cfm))
    return float(correct_samples) / total_samples


def get_mpca(np_cfm, ignored_classes=None):
    samples_per_class = np.sum(np_cfm, axis=1)
    correct_samples_per_class = np.diagonal(np_cfm)

    # Avoid division by zero for classes with no samples such as background
    samples_per_class[samples_per_class == 0] = 1
    class_accuracies = list((np.array(correct_samples_per_class, dtype=np.float32)/samples_per_class))

    # Exclude ignored classes
    if ignored_classes is not None:
        for class_idx in ignored_classes:
            del class_accuracies[class_idx]

    mpca = np.mean(class_accuracies)
    return mpca


def get_miou(np_cfm, ignored_classes=None):
    # iou for a class: true_positive / (true_positive + false_positive + false_negative)
    true_positives_per_class = np.diagonal(np_cfm)
    false_positives_per_class = np.sum(np_cfm, axis=0) - true_positives_per_class
    false_negatives_per_class = np.sum(np_cfm, axis=1) - true_positives_per_class

    denominator = true_positives_per_class + false_positives_per_class + false_negatives_per_class

    # Avoid division by zero for classes with no samples such as background
    denominator[denominator == 0] = 1
    iou_per_class = list(np.array(true_positives_per_class, dtype=np.float32) / denominator)

    # Exclude ignored classes
    if ignored_classes is not None:
        for class_idx in ignored_classes:
            del iou_per_class[class_idx]

    miou = np.mean(iou_per_class)
    return miou


def get_numpy_metrics(np_cfm, ignored_classes):
    metrics = {}
    metrics['accuracy'] = get_accuracy(np_cfm)
    metrics['mpca'] = get_mpca(np_cfm, ignored_classes)
    metrics['miou'] = get_miou(np_cfm, ignored_classes)
    return metrics


def metrics_to_string(metrics):
    s = ""
    for metric in sorted(metrics.iterkeys()):
        s += "{}: {:.7f}, ".format(metric, metrics[metric])

    # Remove training comma and space
    s = s[:-2]
    return s


def print_per_epoch_metrics(metrics_per_epoch):
    for epoch in sorted(metrics_per_epoch.iterkeys()):
        print "{:03}: {}".format(epoch, metrics_to_string(metrics_per_epoch[epoch]))


def animate_cfms(cfms, ignore_classes, save_file=None):
    from matplotlib import pyplot as plt
    from matplotlib import animation
    import seaborn as sn
    import pandas as pd

    def normalize_df_cfm(df):
        # type: (pd.DataFrame) -> pd.DataFrame
        df_norm = df.div(df.sum(axis=1), axis=0)
        return df_norm

    def animate(df_cfm):
        # Clear frame
        plt.clf()

        # Draw the heatmap
        hm = sn.heatmap(df_cfm, annot=True, annot_kws={"size": 8}, cmap='coolwarm', cbar=False, vmin=0, vmax=1)
        hm.set_yticklabels(hm.get_yticklabels(), rotation=0, fontsize=8)
        hm.set_xticklabels(hm.get_xticklabels(), rotation=90, fontsize=8)

        return hm,

    # Class names
    class_names = ['Background', 'Brick', 'Carpet', 'Ceramic', 'Other', 'Fabric', 'Foliage', 'Food', 'Glass',
                   'Polished stone', 'Stone', 'Hair', 'Leather', 'Metal', 'Mirror', 'Painted', 'Paper', 'Plastic',
                   'Skin', 'Sky', 'Tile', 'Wallpaper', 'Water', 'Wood']

    # Remove ignored class labels
    for class_idx in ignore_classes:
        del class_names[class_idx]

    # Create an ordered list of DataFrame cfms
    df_cfms = []

    for epoch in sorted(cfms.iterkeys()):
        # Take a copy of the numpy cfm
        np_cfm = np.array(cfms[epoch], copy=True)

        # Remove ignored classes from the np_cfm
        for class_idx in ignore_classes:
            np_cfm = np.delete(np_cfm, obj=class_idx, axis=0)
            np_cfm = np.delete(np_cfm, obj=class_idx, axis=1)

        # Create data frame
        df_cfm = pd.DataFrame(np_cfm, index=[i for i in class_names], columns=[i for i in class_names])

        # Normalize the data frame
        df_cfm = normalize_df_cfm(df_cfm)

        # Round
        df_cfm = df_cfm.round(decimals=2)
        df_cfm[df_cfm == 0] = 0

        # Append to the ordered list
        df_cfms.append(df_cfm)

    # Create animation
    fig = plt.figure(figsize=(10, 10))
    anim = animation.FuncAnimation(fig, animate, frames=df_cfms, init_func=None, repeat=False)
    plt.show()

    if save_file is not None:
        anim.save(save_file)


def main():
    # Construct the argument parser and parge arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True, help="Path to the folder with the confusion matrices")
    ap.add_argument("--ignored", required=False, default="", help="Classes to ignore in the metrics calculations e.g. 0,1,2")
    args = vars(ap.parse_args())

    # Assumes file names matching the structure:
    # logits_cfm_<epoch>.txt
    # val_logits_cfm_<epoch>.txt
    # teacher_val_logits_cfm_<epoch>.txt

    directory_path = args['folder']
    ignored_classes = args['ignored']

    ignored_classes = [int(class_idx) for class_idx in ignored_classes.split()]

    # If a single file is given
    if path.isfile(directory_path):
        epoch = 0
        np_cfm = np.loadtxt(directory_path)
        metrics = get_numpy_metrics(np_cfm, ignored_classes)
        print metrics_to_string(metrics)
    else:
        print 'Searching for cfm files in: {}'.format(directory_path)
        cfm_files = get_all_txt_files(directory_path)
        print 'Found {} cfm files (.txt)'.format(len(cfm_files))
        cfm_files.sort()
        print 'Ignored classes: {}'.format(ignored_classes)

        training_cfms = [f for f in cfm_files if path.basename(f).startswith('logits')]
        validation_cfms = [f for f in cfm_files if path.basename(f).startswith('val')]
        teacher_cfms = [f for f in cfm_files if path.basename(f).startswith('teacher')]

        print 'Number of training cfms: {}'.format(len(training_cfms))
        print 'Number of validation cfms: {}'.format(len(validation_cfms))
        print 'Number of teacher validation cfms: {}'.format(len(teacher_cfms))

        # Print training metrics
        if len(training_cfms) > 0:
            print 'Training metrics'
            training_metrics = {}
            np_training_cfms = {}

            for cfm_filepath in training_cfms:
                filename = path.basename(cfm_filepath)
                epoch = int(filename.split('_')[-1].split('.')[0])
                np_cfm = np.loadtxt(cfm_filepath)
                metrics = get_numpy_metrics(np_cfm, ignored_classes)
                training_metrics[epoch] = metrics
                np_training_cfms[epoch] = np_cfm

            print_per_epoch_metrics(training_metrics)
            animate_cfms(cfms=np_training_cfms, ignore_classes=ignored_classes, save_file='tmp.mp4')

        # Print validation metrics
        if len(validation_cfms) > 0:
            print 'Validation metrics'
            validation_metrics = {}

            for cfm_filepath in validation_cfms:
                filename = path.basename(cfm_filepath)
                epoch = int(filename.split('_')[-1].split('.')[0])
                np_cfm = np.loadtxt(cfm_filepath)
                metrics = get_numpy_metrics(np_cfm, ignored_classes)
                validation_metrics[epoch] = metrics

            print_per_epoch_metrics(validation_metrics)

        # Print teacher validation metrics
        if len(teacher_cfms) > 0:
            print 'Teacher validation metrics'
            teacher_metrics = {}

            for cfm_filepath in teacher_cfms:
                filename = path.basename(cfm_filepath)
                epoch = int(filename.split('_')[-1].split('.')[0])
                np_cfm = np.loadtxt(cfm_filepath)
                metrics = get_numpy_metrics(np_cfm, ignored_classes)
                teacher_metrics[epoch] = metrics

            print print_per_epoch_metrics(teacher_metrics)

    print 'Done'


if __name__ == '__main__':
    main()