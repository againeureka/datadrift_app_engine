"""

Convert TextRecognitionDataGenerator's result data to Deep-Text-Recognition-Benchmark's input data.

## Reference projects
1. TextRecognitionDataGenerator:
    https://github.com/Belval/TextRecognitionDataGenerator
2. Deep-Text-Recognition-Benchmark:
    https://github.com/clovaai/deep-text-recognition-benchmark

## Example usage:
    python convert.py \
        --input_path ./input \
        --output_path ./output

## Data structure:
    ./input
    # result of TextRecognitionDataGenerator project.
    #   [gt]_[idx].[ext]
        abcd_00001.jpg
        efgh_00002.jpg
        ...

    ./output
    # for use in deep-text-recognition-benchmark project.
        /images
        #   image_[idx].[ext] - for Deep-Text-Recognition-Benchmark
            image_00001.jpg
            image_00002.jpg
            ...
        gt.txt

## Ground truth file structure:
#   gt.txt
#   {filename}\t{label}\n
    images/image_00001.jpg    abcd
    images/image_00002.jpg    efgh
    ...

"""

import os
import shutil
import argparse
import random

def run(input_path, output_path, split_ratio=0.8):
    print('input path: ', os.path.abspath(input_path))
    print('output path: ', os.path.abspath(output_path))

    if not os.path.exists(input_path):
        print('\nInput data path [%s] is not found.\n' % os.path.abspath(input_path))
        return

    if os.path.exists(output_path):
        print('\nOutput folder already exists.')
        print('\nSo, delete all data of output folder [%s]\n' % os.path.abspath(output_path))
        shutil.rmtree(output_path)

    # Create train and val directories
    train_path = os.path.join(output_path, 'train')
    val_path = os.path.join(output_path, 'val')
    for path in [train_path, val_path]:
        os.makedirs(os.path.join(path, 'images'), exist_ok=True)

    # Load input data
    files, count = get_files(input_path)
    random.shuffle(files)  # Shuffle files for random split
    split_index = int(count * split_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]

    # Process train files
    process_files(train_files, train_path, 'train')

    # Process val files
    process_files(val_files, val_path, 'val')

    print('\nConversion complete!\n')

def process_files(files, output_path, dataset_type):
    digits = len(str(len(files)))
    gt_file = open(os.path.join(output_path, 'gt.txt'), 'w', encoding='UTF8')
    for idx, item in enumerate(files):
        if (idx+1) % 100 == 0:
            print(('\r%{}d / %{}d Processing {} !!'.format(digits, digits, dataset_type)) % (idx+1, len(files)), end='')

        gt = os.path.basename(item).split('_')[0]   # remove index
        ext = os.path.splitext(item)[1]

        filename = os.path.join('images', ('image_%0{}d'.format(digits) % idx) + ext)
        gt_file.write('%s\t%s\n' % (filename, gt))
        shutil.copy(item, os.path.join(output_path, filename))

    gt_file.close()


def get_files(path):
    file_list = []

    files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    abspath = os.path.abspath(path)
    for file in files:
        file_path = os.path.join(abspath, file)
        file_list.append(file_path)

    return file_list, len(file_list)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert dataset for deep-text-recognition-benchmark')

    parser.add_argument('--input_path', dest='input_path',
                        help='Data path of TextRecognitionDataGenerator project result')
    parser.add_argument('--output_path', dest='output_path',
                        help='Data path for use in deep-text-recognition-benchmark project')

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    args = parse_arguments()
    run(input_path=args.input_path, output_path=args.output_path)

    # # for test
    # run(input_path='./input', output_path='./output')