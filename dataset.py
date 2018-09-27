DATASET = 'sdha2010'

dataset = getattr(__import__('datasets.' + DATASET), DATASET)
load_data = dataset.load_data
load_data3d = dataset.load_data3d
categories = dataset.categories

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['extract', 'extract3d', 'download'], help="no")
    args = parser.parse_args()
    getattr(dataset, args.action)()
