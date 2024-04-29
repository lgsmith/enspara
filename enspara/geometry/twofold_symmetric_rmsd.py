import mdtraj as md
import numpy as np


# an RMSD calculation that computes the RMSD of the target trajectory to a reference,
# then swaps the atom order of the reference and recomputes the RMSD, 
# returning the minimum of these two values across the frames computed.
def twofold_symmetric_rmsd(target, reference, **kwargs):
    """Calculate the minimum of the RMSDs between a target trajectory and a reference, 
    and the target trajectory and the reference with the first half of its 
    atom indices swapped with the second. This correctly accounts for a system 
    with two-fold symmetry, for example a homodimer where part A and part B 
    in each frame may either match AB or BA in the reference.
    
    Parameters
    ----------
    target : md.Trajectory object
        For each conformation in this trajectory, compute the RMSD to a particular 
        ‘reference’ conformation in another trajectory object.
    reference : md.Trajectory object
        The object containing the reference conformation to measure distances to.
    kwargs : see mdtraj.rmsd

    Returns
    ----------
    rmsds : np.ndarray, shape=(target.n_frames,)
        A 1-D numpy array of the optimal root-mean-square deviations from the frame-th 
        conformation in reference to either the AB or BA atom index order.
    """
    n_atoms = reference.n_atoms
    splitpoint = int(n_atoms/2)
    all_inds = np.arange(n_atoms, dtype=int)
    swap_inds = np.hstack(
        (np.arange(splitpoint, n_atoms, dtype=int), np.arange(0, splitpoint, dtype=int)))
    a2a_b2b_rmsd = md.rmsd(target, reference, **kwargs)
    a2b_b2a_rmsd = md.rmsd(target, reference, atom_indices=all_inds,
                           ref_atom_indices=swap_inds, **kwargs)
    return np.min(np.vstack((a2a_b2b_rmsd, a2b_b2a_rmsd)), axis=0)
