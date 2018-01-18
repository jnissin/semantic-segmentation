# coding=utf-8

import argparse


def main():

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=str, help="Path to the original dataset")
    ap.add_argument("-o", "--output", required=True, type=str, help="Path to the new dataset")
    ap.add_argument("-s", "--samples", required=True, type=int, help="Number of samples per class")
    ap.add_argument("-c", "--classes", required=True, type=int, help="Number of classes")
    args = vars(ap.parse_args())

    input = args["input"]
    output = args["output"]
    num_samples_per_class = int(args["samples"])
    num_classes = int(args["classes"])
    input_dataset = [[]] * num_classes

    print 'Reading input dataset from {}'.format(input)

    with open(input, 'r') as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            sample_class = int(line.split(',')[0])
            input_dataset[sample_class].append(line)

    num_input_samples = sum([len(class_samples) for class_samples in input_dataset])
    input_max_samples = max([len(class_samples) for class_samples in input_dataset])
    input_min_samples = max([len(class_samples) for class_samples in input_dataset])

    print 'Read input dataset with {} samples, smallest class had {} samples, biggest class had {} samples'\
        .format(num_input_samples, input_min_samples, input_max_samples)

    if input_min_samples < num_samples_per_class:
        print 'Cannot create balanced dataset with {} samples per class, smallest class has only {} samples'\
            .format(num_samples_per_class, input_min_samples)
        exit(0)

    print 'Creating output dataset to: {}'.format(output)

    with open(output, 'w') as f:
        for c in range(0, num_classes):
            print 'Writing class {} samples'.format(c)
            for s in range(0, num_samples_per_class):
                f.write(input_dataset[c][s] + '\n')

    print 'New dataset created to {} with {} samples, {} samples per class'\
        .format(output, num_classes*num_samples_per_class, num_samples_per_class)


if __name__ == '__main__':
    main()
