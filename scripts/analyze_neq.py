"""
Analyze NEQ simulations from Perses for the relative free energy.

Install dependencies:
mamba create -n perses -c conda-forge -c openeye mpi4py perses openeye-toolkits

Credits to @glass-w and zhang-ivy
"""
import argparse
from pathlib import Path

from openmmtools.constants import kB
import pymbar
from simtk.openmm import unit


# Read args
parser = argparse.ArgumentParser(
    description="Analyze Non-equilibrium switching free energy calculations from Perses."
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
    "-d",
    dest="description",
    type=str,
    default="",
    help="description of the system, e.g. ABL1 dasatinib THR315ALA, default is an empty string"
)
parser.add_argument(
    "-s",
    dest="switching_time",
    type=int,
    default=1.5,
    help="length of non-equilibrium switching simulations in ns, default is 1.5"
)
parser.add_argument(
    "--apo_forward_pattern",
    dest="apo_forward_pattern",
    type=str,
    default="^apo_forward_\\d+.npy$",
    help="regex pattern to identify files with apo forward works in the input directory, "
         "default is '^apo_forward_\\d+.npy$'"
)
parser.add_argument(
    "--apo_reverse_pattern",
    dest="apo_reverse_pattern",
    type=str,
    default="^apo_reverse_\\d+.npy$",
    help="regex pattern to identify files with apo reverse works in the input directory, "
         "default is '^apo_reverse_\\d+.npy$'"
)
parser.add_argument(
    "--complex_forward_pattern",
    dest="complex_forward_pattern",
    type=str,
    default="^complex_forward_\\d+.npy$",
    help="regex pattern to identify files with complex forward works in the input directory, "
         "default is '^complex_forward_\\d+.npy$'"
)
parser.add_argument(
    "--complex_reverse_pattern",
    dest="complex_reverse_pattern",
    type=str,
    default="^complex_reverse_\\d+.npy$",
    help="regex pattern to identify files with complex reverse works in the input directory, "
         "default is '^complex_reverse_\\d+.npy$'"
)
args = parser.parse_args()
args.input_dir = Path(args.input_dir)
if not args.output_dir:
    args.output_dir = args.input_dir
else:
    args.output_dir = Path(args.output_dir)
args.output_dir.mkdir(parents=True, exist_ok=True)


KT_KCALMOL = kB * 300 * unit.kelvin / unit.kilocalories_per_mole


def subtract_offset(work):
    """ Subtract the initial work of the work trajectory. """
    import numpy as np

    work_offset = []
    for cycle in work:
        work_offset.append(np.array([value - cycle[0] for value in cycle[1:]]))
    work_offset = np.array(work_offset)

    return work_offset


def load_work_arrays(directory, file_pattern):
    """ Load works stored in numpy arrays. """
    import re
    import numpy as np

    file_pattern = re.compile(file_pattern)
    paths = [path for path in directory.glob("*.npy") if file_pattern.search(path.name)]

    work_arrays = []
    for path in paths:
        with open(path, 'rb') as rf:
            work_arrays.append(np.load(rf))

    work_arrays_combined = np.concatenate(work_arrays)
    # compute this separately because the last value of the subsampled array is diff than the actual last sample
    work_arrays_accumulated = np.array([cycle[-1] - cycle[0] for cycle in work_arrays_combined])  # TODO: ask why
    work_arrays_combined = np.array([cycle for cycle in work_arrays_combined])
    work_arrays_offset = subtract_offset(work_arrays_combined)

    return work_arrays_offset, work_arrays_accumulated


def plot_works(
        forward_work_offset,
        reverse_work_offset,
        dg,
        ddg,
        phase,
        switching_time,
        title,
        output_dir
):
    """ Plot the work trajectory and distribution. """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # plot work trajectories
    for i, cycle in enumerate(forward_work_offset):
        x = [(x + 1) * (switching_time / len(cycle)) for x in range(len(cycle))]
        if i == 0:
            plt.plot(x, cycle, color="#377eb8", label="forward")
        else:
            plt.plot(x, cycle, color="#377eb8")

    for i, cycle in enumerate(reverse_work_offset):
        x = [(x + 1) * (switching_time / len(cycle)) for x in range(len(cycle))]
        if i == 0:
            plt.plot(x, -cycle, color="#ff7f00", label="reverse")
        else:
            plt.plot(x, -cycle, color="#ff7f00")

    plt.xlabel("$t_{NEQ}$ [ns]")
    plt.ylabel("work [kT]")
    if len(title) > 0:
        plt.title(f"{title} {phase}")
    else:
        plt.title(f"{phase}")
    plt.legend(loc='best')
    plt.savefig(output_dir / f"{phase}_work_trajectory.png", dpi=500)
    plt.clf()

    # plot work distributions
    accumulated_forward = [cycle[-1] for cycle in forward_work_offset]
    accumulated_reverse = [-cycle[-1] for cycle in reverse_work_offset]
    min_work = int(min(accumulated_forward + accumulated_reverse)) - 4
    max_work = int(max(accumulated_forward + accumulated_reverse) + 5)
    bins = range(min_work, max_work)
    sns.histplot(
        accumulated_forward,
        color="#377eb8",
        label="forward",
        stat="probability",
        bins=bins,
        kde=True,
        kde_kws={"cut": 3},
        alpha=0.4,
        linewidth=0,
    )
    sns.histplot(
        accumulated_reverse,
        color="#ff7f00",
        label="reverse",
        stat="probability",
        bins=bins,
        kde=True,
        kde_kws={"cut": 3},
        alpha=0.4,
        linewidth=0,
    )
    plt.axvline(dg)
    plt.axvline(dg + ddg, linestyle='dashed')
    plt.axvline(dg - ddg, linestyle='dashed')
    plt.xticks(bins[::int(len(bins) / 8)])
    plt.xlabel("work [kT]")
    plt.ylabel("p(w)")
    if len(title) > 0:
        plt.title(f"{title} {phase}")
    else:
        plt.title(f"{phase}")
    plt.legend(loc='best')
    plt.savefig(output_dir / f"{phase}_work_distribution.png", dpi=500)
    plt.clf()

    return


# load work arrays
forward_complex_offset, forward_complex_accumulated = load_work_arrays(
    args.input_dir,
    args.complex_forward_pattern
)
reverse_complex_offset, reverse_complex_accumulated = load_work_arrays(
    args.input_dir,
    args.complex_reverse_pattern
)
forward_apo_offset, forward_apo_accumulated = load_work_arrays(
    args.input_dir,
    args.apo_forward_pattern
)
reverse_apo_offset, reverse_apo_accumulated = load_work_arrays(
    args.input_dir, args.apo_reverse_pattern
)

# analyse work
complex_dg, complex_ddg = pymbar.bar.BAR(forward_complex_accumulated, reverse_complex_accumulated)
apo_dg, apo_ddg = pymbar.bar.BAR(forward_apo_accumulated, reverse_apo_accumulated)

# plot work trajectories and distributions
plot_works(
    forward_complex_offset,
    reverse_complex_offset,
    complex_dg,
    complex_ddg,
    phase="complex",
    switching_time=args.switching_time,
    title=args.description,
    output_dir=args.output_dir
)

plot_works(
    forward_apo_offset,
    reverse_apo_offset,
    apo_dg,
    apo_ddg,
    phase="apo",
    switching_time=args.switching_time,
    title=args.description,
    output_dir=args.output_dir
)

# calculate relative free energy
binding_dg = complex_dg - apo_dg
binding_ddg = (apo_ddg ** 2 + complex_ddg ** 2) ** 0.5
result = f"DDG: {round(binding_dg * KT_KCALMOL, 2)} +/- {round(binding_ddg * KT_KCALMOL, 2)} kcal/mol"

# write results to text file
with open(args.output_dir / "ddg.txt", "w") as wf:
    wf.write(result)

print(result)
