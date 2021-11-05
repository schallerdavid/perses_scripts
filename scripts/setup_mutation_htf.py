"""
Setup a hybrid topology factory for protein mutations with perses.
Tested with perses 0.9.2.

Install dependencies:
mamba create -n perses -c conda-forge -c openeye mpi4py perses openeye-toolkits

Credits to @glass-w
"""
import argparse
import logging
from pathlib import Path
import pickle

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

import mdtraj as md
import numpy as np
from perses.app.relative_point_mutation_setup import PointMutationExecutor
from perses.utils.smallmolecules import render_protein_residue_atom_mapping
import simtk.openmm as mm


parser = argparse.ArgumentParser(
    description="Prepare hybrid topologies protein mutations"
)
parser.add_argument(
    "-o",
    dest="output_dir",
    type=str,
    help="the path to the output directory",
)
parser.add_argument(
    "-p",
    dest="protein_path",
    type=str,
    help="the path to the protein structure in e.g. PDB format",
)
parser.add_argument(
    "-l",
    dest="ligand_path",
    type=str,
    default="",
    help="the path to e.g. the ligand structure in SDF format or another protein in PDB format",
)
parser.add_argument(
    "-m",
    dest="mutation",
    type=str,
    help="the mutation to setup in the format ALA123THR including non-standard amino acids e.g. HIP159ALA",
)
parser.add_argument(
    "-c",
    dest="protein_chain",
    type=str,
    default="1",
    help="the protein chain which should be mutated",
)
parser.add_argument(
    "-f",
    dest="small_molecule_ff",
    type=str,
    default="gaff-2.11",
    help="the forcefield to use for the small molecule parametrization",
)
parser.add_argument(
    "--conduct_endstate_validation",
    dest="conduct_endstate_validation",
    action="store_true",
    default=False,
    help="if endstate validation should be conducted",
)
parser.add_argument(
    "--allow_undefined_stereo",
    dest="allow_undefined_stereo",
    action="store_true",
    default=False,
    help="if undefined stereo centers should be allowed in the ligand",
)
parser.add_argument(
    "--flatten_torsions",
    dest="flatten_torsions",
    action="store_false",
    default=True,
    help="if torsions should be flattened",
)
parser.add_argument(
    "--flatten_exceptions",
    dest="flatten_exceptions",
    action="store_false",
    default=True,
    help="if exceptions should be flattened",
)
parser.add_argument(
    "--generate_rest_capable_htf",
    dest="generate_rest_capable_htf",
    action="store_true",
    default=False,
    help="if hybrid topology factory should be capable of rest",
)
args = parser.parse_args()

# make output directory
args.output_dir = Path(args.output_dir)
args.output_dir.mkdir(parents=True, exist_ok=True)

# setting maximum number of CPU threads to 1
platform = mm.Platform.getPlatformByName("CPU")
platform.setPropertyDefaultValue(property="Threads", value=str(1))

# make the system
solvent_delivery = PointMutationExecutor(
    protein_filename=args.protein_path,
    mutation_chain_id=args.protein_chain,
    old_residue=args.mutation[:3],
    mutation_residue_id=args.mutation[3:-3],
    proposed_residue=args.mutation[-3:],
    conduct_endstate_validation=args.conduct_endstate_validation,
    ligand_input=args.ligand_path if args.ligand_path else None,
    allow_undefined_stereo_sdf=args.allow_undefined_stereo,
    small_molecule_forcefields=args.small_molecule_ff,
    flatten_torsions=args.flatten_torsions,
    flatten_exceptions=args.flatten_exceptions,
    generate_unmodified_hybrid_topology_factory=True if not args.generate_rest_capable_htf else False,
    generate_rest_capable_hybrid_topology_factory=args.generate_rest_capable_htf,
)

# make image map of the transformation
render_protein_residue_atom_mapping(
    solvent_delivery.get_apo_htf()._topology_proposal,
    str(args.output_dir / "transformation.png"))

# pickle the output and save
pickle.dump(
    solvent_delivery.get_apo_htf(),
    open(args.output_dir / "apo.pickle", "wb")
)
pickle.dump(
    solvent_delivery.get_complex_htf(),
    open(args.output_dir / "complex.pickle", "wb")
)

# save the coordinates to of apo and complex to check the geometry of the transformation
htfs_t = [solvent_delivery.get_apo_htf(), solvent_delivery.get_complex_htf()]

top_old = md.Topology.from_openmm(htfs_t[0]._topology_proposal.old_topology)
top_new = md.Topology.from_openmm(htfs_t[0]._topology_proposal.new_topology)
traj = md.Trajectory(np.array(htfs_t[0].old_positions(htfs_t[0].hybrid_positions)), top_old)
traj.save(str((args.output_dir / "apo_old.pdb").resolve()))
traj = md.Trajectory(np.array(htfs_t[0].new_positions(htfs_t[0].hybrid_positions)), top_new)
traj.save(str((args.output_dir / "apo_new.pdb").resolve()))

top_old = md.Topology.from_openmm(htfs_t[1]._topology_proposal.old_topology)
top_new = md.Topology.from_openmm(htfs_t[1]._topology_proposal.new_topology)
traj = md.Trajectory(np.array(htfs_t[1].old_positions(htfs_t[1].hybrid_positions)), top_old)
traj.save(str((args.output_dir / "complex_old.pdb").resolve()))
traj = md.Trajectory(np.array(htfs_t[1].new_positions(htfs_t[1].hybrid_positions)), top_new)
traj.save(str((args.output_dir / "complex_new.pdb").resolve()))
