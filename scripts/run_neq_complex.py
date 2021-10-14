"""
Run NEQ free energy calculations for a protein:ligand complex using Perses.

Install dependencies:
mamba create -n perses -c conda-forge -c openeye mpi4py perses openeye-toolkits

Credits to @glass-w and @zhang-ivy
"""
import argparse
import logging
from pathlib import Path
import pickle
import time

import numpy as np
from openmmtools.integrators import PeriodicNonequilibriumIntegrator
from simtk import openmm, unit

# Set up logger
logging.basicConfig(level=logging.DEBUG)

# Read args
parser = argparse.ArgumentParser(
    description="Run Non-equilibrium switching free energy calculations for a protein:ligand complex"
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
parser.add_argument(
    "-c",
    dest="cycle",
    type=int,
    default=0,
    help="cycle ID to run NEQ, default is 0"
)
parser.add_argument(
    "-t",
    dest="trajectories",
    action="store_true",
    default=False,
    help="if trajectories shall be written, default is False"
)
args = parser.parse_args()
args.input_dir = Path(args.input_dir)
if not args.output_dir:
    args.output_dir = args.input_dir
else:
    args.output_dir = Path(args.output_dir)
args.output_dir.mkdir(parents=True, exist_ok=True)

# Define lambda functions
x = "lambda"
DEFAULT_ALCHEMICAL_FUNCTIONS = {
    "lambda_sterics_core": x,
    "lambda_electrostatics_core": x,
    "lambda_sterics_insert": f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    "lambda_sterics_delete": f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    "lambda_electrostatics_insert": f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    "lambda_electrostatics_delete": f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    "lambda_bonds": x,
    "lambda_angles": x,
    "lambda_torsions": x,
}

# Define simulation parameters
temperature = 300  # 300 K
nsteps_eq = 375000  # 1.5 ns
nsteps_neq = 375000  # 1.5 ns
nworks = 1000  # number of works to save
nsnapshots = 100  # number of snapshots to save
neq_splitting = "V R H O R V"
timestep = 4.0 * unit.femtosecond

# load hybrid topology factory
htf = pickle.load(open(args.input_dir / f"{args.phase}.pickle", "rb"))
system = htf.hybrid_system
positions = htf.hybrid_positions

# Set up integrator
integrator = PeriodicNonequilibriumIntegrator(
    DEFAULT_ALCHEMICAL_FUNCTIONS,
    nsteps_eq,
    nsteps_neq,
    neq_splitting,
    timestep=timestep,
)

# Set up context
platform = openmm.Platform.getPlatformByName("CPU")
platform.setPropertyDefaultValue(property="Threads", value=str(1))
platform = openmm.Platform.getPlatformByName("CUDA")
platform.setPropertyDefaultValue("Precision", "mixed")
platform.setPropertyDefaultValue("DeterministicForces", "true")
context = openmm.Context(system, integrator, platform)
context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

# Minimize
openmm.LocalEnergyMinimizer.minimize(context)

# Run neq
forward_works_master, reverse_works_master = list(), list()
forward_eq_old, forward_neq_old, forward_neq_new = list(), list(), list()
reverse_eq_new, reverse_neq_old, reverse_neq_new = list(), list(), list()

# Equilibrium (lambda = 0)
initial_time = time.time()
for step in range(nsteps_eq):
    integrator.step(1)
    if step % round(nsteps_eq / 20) == 0:
        logging.debug(
            f"Step: {step}, Equilibrating at lambda = 0, "
            f"Elapsed time: {(time.time() - initial_time) / 60} minutes"
        )
    if step % round(nsteps_eq / nsnapshots) == 0:
        positions = context.getState(
            getPositions=True, enforcePeriodicBox=False
        ).getPositions(asNumpy=True)
        old_positions = np.asarray(htf.old_positions(positions))
        forward_eq_old.append(old_positions)

# Forward (0 -> 1)
forward_works = [integrator.get_protocol_work(dimensionless=True)]
for fwd_step in range(nsteps_neq):
    integrator.step(1)
    if fwd_step % round(nsteps_neq / 20) == 0:
        logging.info(
            f"Step: {fwd_step}, Forward NEQ, "
            f"Elapsed time: {(time.time() - initial_time) / 60} minutes"
        )
    if fwd_step % round(nsteps_neq / nworks) == 0:
        forward_works.append(integrator.get_protocol_work(dimensionless=True))

    if fwd_step % round(nsteps_neq / nsnapshots) == 0:
        positions = context.getState(
            getPositions=True, enforcePeriodicBox=False
        ).getPositions(asNumpy=True)
        old_positions = np.asarray(htf.old_positions(positions))
        new_positions = np.asarray(htf.new_positions(positions))
        forward_neq_old.append(old_positions)
        forward_neq_new.append(new_positions)
forward_works_master.append(forward_works)

# Equilibrium (lambda = 1)
for step in range(nsteps_eq):
    integrator.step(1)
    if step % round(nsteps_eq / 20) == 0:
        logging.info(
            f"Step: {step}, Equilibrating at lambda = 1, "
            f"Elapsed time: {(time.time() - initial_time) / 60} minutes"
        )
    if step % round(nsteps_eq / nsnapshots) == 0:
        positions = context.getState(
            getPositions=True, enforcePeriodicBox=False
        ).getPositions(asNumpy=True)
        new_positions = np.asarray(htf.new_positions(positions))
        reverse_eq_new.append(new_positions)

# Reverse work (1 -> 0)
reverse_works = [integrator.get_protocol_work(dimensionless=True)]
for rev_step in range(nsteps_neq):
    integrator.step(1)
    if rev_step % round(nsteps_neq / 20) == 0:
        logging.info(
            f"Step: {rev_step}, Reverse NEQ, "
            f"Elapsed time: {(time.time() - initial_time) / 60} minutes"
        )

    if rev_step % round(nsteps_neq / nworks) == 0:
        reverse_works.append(integrator.get_protocol_work(dimensionless=True))

    if rev_step % round(nsteps_neq / nsnapshots) == 0:
        positions = context.getState(
            getPositions=True, enforcePeriodicBox=False
        ).getPositions(asNumpy=True)
        old_positions = np.asarray(htf.old_positions(positions))
        new_positions = np.asarray(htf.new_positions(positions))
        reverse_neq_old.append(old_positions)
        reverse_neq_new.append(new_positions)
reverse_works_master.append(reverse_works)

# Save works
with open(args.output_dir / f"{args.phase}_forward_{args.cycle}.npy", "wb") as f:
    np.save(f, forward_works_master)
with open(args.output_dir / f"{args.phase}_reverse_{args.cycle}.npy", "wb") as f:
    np.save(f, reverse_works_master)

# Save trajectories
if args.trajectories:
    with open(args.output_dir / f"{args.phase}_forward_eq_old_{args.cycle}.npy", "wb") as f:
        np.save(f, forward_eq_old)
    with open(args.output_dir / f"{args.phase}_reverse_eq_new_{args.cycle}.npy", "wb") as f:
        np.save(f, reverse_eq_new)

    with open(args.output_dir / f"{args.phase}_forward_neq_old_{args.cycle}.npy", "wb") as f:
        np.save(f, forward_neq_old)
    with open(args.output_dir / f"{args.phase}_forward_neq_new_{args.cycle}.npy", "wb") as f:
        np.save(f, forward_neq_new)

    with open(args.output_dir / f"{args.phase}_reverse_neq_old_{args.cycle}.npy", "wb") as f:
        np.save(f, reverse_neq_old)
    with open(args.output_dir / f"{args.phase}_reverse_neq_new_{args.cycle}.npy", "wb") as f:
        np.save(f, reverse_neq_new)

logging.info(f"Finished cycle {args.cycle} after {(time.time() - initial_time) / 3600} hours.")
