import os
import sys
import argparse
from indoor3d_util import collect_point_label


def main(input_dir, output_dir):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)

    anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
    anno_paths = [os.path.join(input_dir, p) for p in anno_paths]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    for anno_path in anno_paths:
        print(anno_path)
        try:
            elements = anno_path.split('/')
            out_filename = elements[0].split("\\")[-1] + '_' + elements[-2] + '.npy'  # Area_1_hallway_1.npy
            collect_point_label(anno_path, os.path.join(output_dir, out_filename), 'numpy')
        except Exception as e:
            print(anno_path, 'ERROR!!', e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process indoor3d data.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory where the input data is stored')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where the output data should be saved')

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
