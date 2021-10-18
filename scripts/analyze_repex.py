"""
Analyze RepEx simulations from Perses for the relative free energy.

Install dependencies:
mamba create -n perses -c conda-forge -c openeye mpi4py perses openeye-toolkits

Credits to @glass-w and zhang-ivy
"""
import argparse
import logging
from pathlib import Path

from openmmtools.constants import kB
from simtk.openmm import unit


logging.basicConfig(level=logging.ERROR)

# Read args
parser = argparse.ArgumentParser(
    description="Analyze RepEx free energy calculations from Perses."
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
    dest="simulation_time",
    type=int,
    default=5,
    help="length of the replica simulations in ns, default is 5"
)
parser.add_argument(
    "--apo_nc_file",
    dest="apo_nc_file",
    type=str,
    default="apo.nc",
    help="name of apo nc-file, default is 'apo.nc'"
)
parser.add_argument(
    "--complex_nc_file",
    dest="complex_nc_file",
    type=str,
    default="complex.nc",
    help="name of complex nc-file, default is complex.nc"
)
args = parser.parse_args()
args.input_dir = Path(args.input_dir)
if not args.output_dir:
    args.output_dir = args.input_dir
else:
    args.output_dir = Path(args.output_dir)
args.output_dir.mkdir(parents=True, exist_ok=True)


KT_KCALMOL = kB * 300 * unit.kelvin / unit.kilocalories_per_mole


def analyze_repex(nc_file_path):
    """ Analyze RepEx simulation and return the free energy over time and the final free energy. """
    from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer
    from perses.analysis.utils import open_netcdf

    reporter = MultiStateReporter(nc_file_path)
    nc_file = open_netcdf(nc_file_path)
    n_iterations = nc_file.variables['last_iteration'][0]
    dg, ddg = list(), list()

    # get free energies over time
    for step in range(1, n_iterations + 1, int(n_iterations / 25)):
        analyzer = MultiStateSamplerAnalyzer(reporter, max_n_iterations=step)
        f_ij, df_ij = analyzer.get_free_energy()
        dg.append(f_ij[0, -1])
        ddg.append(df_ij[0, -1])

    # get final free energy
    analyzer = MultiStateSamplerAnalyzer(reporter, max_n_iterations=n_iterations)
    f_ij, df_ij = analyzer.get_free_energy()
    final_dg, final_ddg = f_ij[0, -1], df_ij[0, -1]

    return dg, ddg, final_dg, final_ddg


# adapted from
# https://github.com/choderalab/perses/blob/467c637366a5216d057c6064b8268bd09f4f02c9/perses/analysis/utils.py
def plot_replica_mixing(nc_file_path, title='', filename='replicas.png'):
    """ Plot the path of each replica through the states, with marginal distribution shown. """
    import numpy as np
    import matplotlib.pyplot as plt
    from perses.analysis.utils import open_netcdf

    ncfile = open_netcdf(nc_file_path)

    n_iter, n_states = ncfile.variables['states'].shape
    cmaps = plt.cm.get_cmap('gist_rainbow')
    colours = [cmaps(i) for i in np.linspace(0., 1., n_states)]
    fig, axes = plt.subplots(
        nrows=n_states,
        ncols=2,
        sharex='col',
        sharey=True,
        figsize=(15, 2 * n_states),
        squeeze=True,
        gridspec_kw={'width_ratios': [5, 1]}
    )

    for rep in range(n_states):
        ax = axes[rep, 0]
        y = ncfile.variables['states'][:, rep]
        ax.plot(y, marker='.', linewidth=0, markersize=2, color=colours[rep])
        ax.set_xlim(-1, n_iter + 1)
        if rep == 0:
            ax.set_title(title)
        hist_plot = axes[rep, 1]
        hist_plot.hist(y, bins=n_states, orientation='horizontal', histtype='step', color=colours[rep], linewidth=3)
        ax.set_ylabel('State')
        hist_plot.yaxis.set_label_position("right")
        hist_plot.set_ylabel(f'Replica {rep}', rotation=270, labelpad=10)

        # just plotting for the bottom plot
        if rep == n_states - 1:
            ax.set_xlabel('Iteration')
            hist_plot.set_xlabel('State count')

    fig.tight_layout()
    plt.savefig(filename)
    plt.clf()

    return


def plot_series(dg, ddg, final_dg, final_ddg, phase, simulation_time, title, output_dir):
    """ Plot free energy over time. """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    first_free_energy_to_plot = 7  # discard first 7 free energies
    dg_discarded = dg[first_free_energy_to_plot:]
    ddg_discarded = ddg[first_free_energy_to_plot:]

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='col', figsize=(10, 2 * 3), squeeze=True)

    # dG plot
    ax = axes[0]
    interval = simulation_time / len(dg)
    x = [x * interval + first_free_energy_to_plot * interval for x in range(len(dg) - first_free_energy_to_plot)]
    y = dg_discarded
    # append final free energy results and convert to numpy array
    x.append(simulation_time)
    y.append(final_dg)
    y = np.array(y)
    ddg_discarded.append(final_ddg)
    ddg_discarded = np.array(ddg_discarded)
    ax.plot(x, y, color=sns.color_palette()[0])
    ax.fill_between(x, y - ddg_discarded, y + ddg_discarded, alpha=0.4)

    # ddG plot
    ax = axes[1]
    y = ddg_discarded
    ax.plot(x, y, color=sns.color_palette()[0])

    axes[0].set_ylabel("dG (kT)")
    axes[1].set_ylabel("ddG (kT)")
    axes[1].set_xlabel("$t_{RepEx}$ (ns)")

    axes[0].set_title(f"{title} {phase}")
    plt.savefig(output_dir / f"{phase}_series.png", dpi=500)
    plt.clf()

    return


# analyze repex simulation
complex_dgs, complex_ddgs, complex_final_dg, complex_final_ddg = analyze_repex(
    args.input_dir / args.complex_nc_file
)
apo_dgs, apo_ddgs, apo_final_dg, apo_final_ddg = analyze_repex(
    args.input_dir / args.apo_nc_file
)

# plot replica mixing
plot_replica_mixing(
    args.input_dir / args.complex_nc_file,
    title=f"{args.description} complex",
    filename=args.output_dir / f"complex_replicas.png"
)
plot_replica_mixing(
    args.input_dir / args.apo_nc_file,
    title=f"{args.description} apo",
    filename=args.output_dir / f"apo_replicas.png"
)

# plot dg, ddg, and discrepancy vs. time
plot_series(
    complex_dgs,
    complex_ddgs,
    complex_final_dg,
    complex_final_ddg,
    phase="complex",
    simulation_time=args.simulation_time,
    title=args.description,
    output_dir=args.output_dir
)
plot_series(
    apo_dgs,
    apo_ddgs,
    apo_final_dg,
    apo_final_ddg,
    phase="apo",
    simulation_time=args.simulation_time,
    title=args.description,
    output_dir=args.output_dir
)

binding_dg = complex_final_dg - apo_final_dg
binding_ddg = (complex_final_ddg**2 + apo_final_ddg**2)**0.5
result = f"DDG: {round(binding_dg * KT_KCALMOL, 2)} +/- {round(binding_ddg * KT_KCALMOL, 2)} kcal/mol"

# write results to text file
with open(args.output_dir / "ddg.txt", "w") as wf:
    wf.write(result)

print(result)
