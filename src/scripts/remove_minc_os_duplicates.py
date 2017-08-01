# coding=utf8

import os
import argparse


def main():
    # Construct the argument parser and parse arguments
    ap = argparse.ArgumentParser(description='Removes MINC and opensurface duplicates from folder. Keeps MINC versions.')
    ap.add_argument('-p', '--path', required=True, type=str, help='Path to the folder where to remove from')
    ap.add_argument('-r', '--rename', required=False, type=bool, default=False)
    args = vars(ap.parse_args())

    folder_path = args['path']
    rename = args['rename']

    print 'Reading files from: {}'.format(folder_path)
    images = [f for f in os.listdir(folder_path) if not f.startswith('.')]
    print 'Found {} image files'.format(len(images))
    minc = [i.lstrip('0') for i in images if i.startswith('00')]
    print 'Found {} MINC image files'.format(len(minc))

    images = set(images)
    minc = set(minc)

    duplicates = images.intersection(minc)
    print 'Found {} duplicate files'.format(len(duplicates))

    ret = raw_input('Ready to remove {} images, continue (Y/N)? '.format(len(duplicates)))

    if ret.lower() == 'y':
        for i in duplicates:
            print 'Removing image: {}'.format(i)
            os.remove(os.path.join(folder_path, i))

    if rename:
        images = [f for f in os.listdir(folder_path) if not f.startswith('.')]
        minc_named = [i for i in images if i.startswith('00')]
        ret = raw_input('Ready to rename: {} MINC style named images, continue (Y/N)? '.format(len(minc_named)))

        if ret.lower() == 'y':
            for f in minc_named:
                src = os.path.join(folder_path, f)
                dst = os.path.join(folder_path, f.lstrip('0'))
                print 'Renaming {} to {}'.format(src, dst)
                os.rename(src, dst)

    print 'Done'

if __name__ == '__main__':
    main()
