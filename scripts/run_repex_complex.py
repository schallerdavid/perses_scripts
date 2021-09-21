"""
Run RepEx free energy calculations for a protein:ligand complex using Perses.

Install dependencies:
mamba create -n perses -c conda-forge perses

Credits to @glass-w
"""
import argparse
import logging
from pathlib import Path
import pickle

import simtk.unit as unit
from openmmtools import mcmc
from openmmtools.multistate import MultiStateReporter
from perses.annihilation.lambda_protocol import LambdaProtocol
from perses.samplers.multistate import HybridRepexSampler


# Set up logger
logging.basicConfig(level=logging.DEBUG)

# Read args
parser = argparse.ArgumentParser(
    description="Run RepEx free energy calculations for a protein:ligand complex"
)
parser.add_argument(
    "-i",
    dest="input_dir",
    type=str,
    help="path to input directory"
)
parser.add_argument(
    "-o",
    dest="output_dir",
    type=str,
    default="",
    help="path to output directory, default is input directory"
)
parser.add_argument(
    "-p",
    dest="phase",
    type=str,
    help="apo or complex"
)
args = parser.parse_args()
args.input_dir = Path(args.input_dir)
if not args.output_dir:
    args.output_dir = args.input_dir
else:
    args.output_dir = Path(args.output_dir)

# load hybrid topology factory
htf = pickle.load(open(args.input_dir / f"{args.phase}.pickle", "rb"))

# Build the hybrid repex samplers
suffix = "run"
selection = "not water"
checkpoint_interval = 10
n_states = 11
n_cycles = 5000
lambda_protocol = LambdaProtocol(functions="default")
reporter = MultiStateReporter(
    args.output_dir / f"{args.phase}.nc",
    analysis_particle_indices=htf.hybrid_topology.select(selection),
    checkpoint_interval=checkpoint_interval,
)
hss = HybridRepexSampler(
    mcmc_moves=mcmc.LangevinSplittingDynamicsMove(
        timestep=4.0 * unit.femtoseconds,
        collision_rate=5.0 / unit.picosecond,
        n_steps=250,
        reassign_velocities=False,
        n_restart_attempts=20,
        splitting="V R R R O R R R V",
        constraint_tolerance=1e-06,
    ),
    hybrid_factory=htf,
    online_analysis_interval=10,
)
hss.setup(
    n_states=n_states,
    temperature=300 * unit.kelvin,
    storage_file=reporter,
    lambda_protocol=lambda_protocol,
    endstates=False,
)
hss.extend(n_cycles)
