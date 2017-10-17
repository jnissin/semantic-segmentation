# coding=utf-8

import argparse
import os


def get_files(path, ignore_hidden_files=True, include_sub_dirs=False):
    ret_files = []

    if not include_sub_dirs:
        # Files under the directory directly
        ret_files = os.listdir(path)

        # Filter the hidden files out
        if ignore_hidden_files:
            ret_files = [f for f in ret_files if not f.startswith('.')]

        # Complete the file paths and check that we are only returning files
        ret_files = [os.path.join(path, f) for f in ret_files]
        ret_files = [f for f in ret_files if os.path.isfile(f)]
    else:
        for root, dirs, files in os.walk(path):
            for name in files:
                if ignore_hidden_files and name.startswith('.'):
                    continue

                file_path = os.path.join(root, name)
                if os.path.isfile(file_path) and not name.startswith('.'):
                    ret_files.append(file_path)

    return ret_files


def main():
    ap = argparse.ArgumentParser(description="Removes duplicate files between two folders. The files are removed from folder A and files in B are used as a filter")
    ap.add_argument("-a", "--foldera", required=True, help="Path to folder A")
    ap.add_argument("-b", "--folderb", required=True, help="Path to folder B")
    ap.add_argument("-i", "--incsub", required=False, default=False, type=bool, help="Include sub directories (default false)")
    args = vars(ap.parse_args())

    folder_a_path = args['foldera']
    folder_b_path = args['folderb']
    incsub = args['incsub']

    print 'Reading files from folder A'
    folder_a_files = get_files(folder_a_path, include_sub_dirs=incsub)
    print 'Found {} files in folder A'.format(len(folder_a_files))

    print 'Reading files from folder B'
    folder_b_files = get_files(folder_b_path, include_sub_dirs=incsub)
    print 'Found {} files in folder B'.format(len(folder_b_files))

    # Create two sets and take the intersection to find the duplicate files
    folder_a_file_names = set([os.path.basename(p) for p in folder_a_files])
    folder_b_file_names = set([os.path.basename(p) for p in folder_b_files])
    duplicate_file_names = folder_a_file_names.intersection(folder_b_file_names)

    response = raw_input('Found {} duplicate file names. Ready to remove from folder A, continue (Y/N)? '.format(len(duplicate_file_names)))

    if str(response).lower() == 'y':
        for f in duplicate_file_names:
            file_path = os.path.join(folder_a_path, f)
            print 'Removing: {}'.format(file_path)
            os.remove(file_path)

    print 'Done'


if __name__ == '__main__':
    main()
