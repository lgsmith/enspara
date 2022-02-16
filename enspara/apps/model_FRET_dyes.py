"""Model FRET dyes takes pdb structures and models dye pairs onto
a list of specified residue pairs. You can specify your own dyes or 
this will fall back onto the dyes, Alexa488 and Alexa594, that we supply
with Enspara. We return a probability distribution of dye distances of
length the number of structures provided.
"""

# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors: Justin J Miller <jjmiller@wustl.edu>
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential


import sys
import argparse
import os
import logging
import itertools
import inspect
import pickle
import json
from glob import glob
import subprocess as sp
from multiprocessing import Pool
from functools import partial
from enspara import ra
from enspara.geometry import dyes_from_expt_dist
from enspara.apps.util import readable_dir


import numpy as np
import mdtraj as md

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def process_command_line(argv):
    parser = argparse.ArgumentParser(
        prog='FRET',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert PDB structures a series of FRET dye residue pairs"
                    "into the probability distribution of distances between the two pairs")

#Add additional input to say "calc distributions, calc FEs, or both"
    # INPUTS
    input_args = parser.add_argument_group("Input Settings")
    input_args.add_argument(
        '--centers', nargs="+", required=True, 
        help="Path to cluster centers from the MSM"
             "should be of type .xtc.")
    input_args.add_argument(
        '--topology', required=True,
        help="topology file for supplied trajectory")
    input_args.add_argument(
        '--resid_pairs', nargs="+", action='append', required=True, type=int,
        help="residues to model FRET dyes on. Pass 2 residue pairs, same numbering as"
             "in the topology file. Pass multiple times to model multiple residue pairs"
             "e.g. --resid_pairs 1 5"
             "--resid_pairs 5 86")


    # PARAMETERS
    FRET_args = parser.add_argument_group("FRET Settings")
    FRET_args.add_argument(
        '--n_procs', required=False, type=int, default=1,
        help="Number of cores to use for parallel processing"
            "Generally parallel over number of frames in supplied trajectory/MSM state")    
    FRET_args.add_argument(
        '--FRETdye1', nargs="+", required=False,
        default=os.path.dirname(inspect.getfile(ra))+'/../data/dyes/AF488.pdb',
        help="Path to point cloud of FRET dye pair 2")
    FRET_args.add_argument(
        '--FRETdye2', nargs = "+", required = False,
        default=os.path.dirname(inspect.getfile(ra))+'/../data/dyes/AF594.pdb',
        help = "Path to point cloud of FRET dye pair 2")

    # OUTPUT
    output_args = parser.add_argument_group("Output Settings")
    output_args.add_argument(
        '--FRET_output_dir', required=False, action=readable_dir, default='./',
        help="The location to write the FRET dye distributions.")

    args = parser.parse_args(argv[1:])
    #Need to add error checks?
    return args


def main(argv=None):

    args = process_command_line(argv)

    #Load Centers and dyes
    trj=md.load(args.centers, top=args.topology)
    logger.info(f"Loaded trajectory {args.centers} using topology file {args.topology}")
    dye1=dyes_from_expt_dist.load_dye(args.FRETdye1)
    dye2=dyes_from_expt_dist.load_dye(args.FRETdye2)

    resSeq_pairs=np.array(args.resid_pairs)

    logger.info(f"Calculating dye distance distribution using dyes: {args.FRETdye1}")
    logger.info(f"and {args.FRETdye2}")
    #Calculate the FRET dye distance distributions for each residue pair
    for n in np.arange(len(resSeq_pairs)):
        logger.info(f"Calculating distance distribution for residue pair: {resSeq_pairs[n]}")
        probs, bin_edges = dyes_from_expt_dist.dye_distance_distribution(
            trj, dye1, dye2, resSeq_pairs[n], n_procs=args.n_procs)
        probs_output = f'{args.FRET_output_dir}/probs_{resSeq_pairs[n][0]}_{resSeq_pairs[n][1]}.h5'
        bin_edges_output = f'{args.FRET_output_dir}/bin_edges_{resSeq_pairs[n][0]}_{resSeq_pairs[n][1]}.h5'
        ra.save(probs_output, probs)
        ra.save(bin_edges_output, bin_edges)

    logger.info(f"Success! FRET dye distance distributions may be found here: {args.FRET_output_dir}")

if __name__ == "__main__":
    sys.exit(main(sys.argv))