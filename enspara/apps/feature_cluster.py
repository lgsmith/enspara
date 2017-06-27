import sys
import argparse
import pickle

from mdtraj import io
from msmbuilder.libdistance import cdist

from enspara.cluster import KHybrid
from enspara.util import array as ra


def process_command_line(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--features", required=True,
        help="The h5 file containin observations and features.")
    parser.add_argument(
        "--cluster-algorithm", required=True,
        help="The algorithm to use for clustering.")
    parser.add_argument(
        "--cluster-radius", required=True, type=float,
        help="The radius to cluster to.")
    parser.add_argument(
        "--cluster-distance", default='euclidean',
        help="The metric for measuring distances")

    parser.add_argument(
        "--assignments", required=True,
        help="Location for assignments output.")
    parser.add_argument(
        "--distances", required=True,
        help="Location for distances output.")
    parser.add_argument(
        "--center-indices", required=True,
        help="Location for centers output.")

    args = parser.parse_args(argv[1:])

    if args.cluster_distance.lower() == 'euclidean':
        args.cluster_distance = diff_euclidean
    elif args.cluster_distance.lower() == 'manhattan':
        args.cluster_distance = diff_manhattan

    assert args.cluster_algorithm.lower() == 'khybrid'

    return args


def diff_euclidean(trj, ref):
    return cdist(ref.reshape(1, -1), trj, 'euclidean')[0]


def diff_manhattan(trj, ref):
    return trj - ref


def main(argv=None):
    args = process_command_line(argv)

    keys = io.loadh(args.features).keys()
    features = ra.load(
        args.features, keys=sorted(keys, key=lambda x: x.split('_')[-1]))

    clustering = KHybrid(
        metric=args.cluster_distance,
        cluster_radius=args.cluster_radius)

    clustering.fit(features._data)

    result = clustering.result_.partition(features.lengths)

    ra.save(args.distances, result.distances)
    ra.save(args.assignments, result.assignments)
    pickle.dump(result.center_indices, open(args.center_indices, 'wb'))

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
