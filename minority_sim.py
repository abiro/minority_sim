import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import repeat
from pathlib import Path
import time
import os
import shutil
import sys

import numpy as np
import yaml


class Simulation:
    """Monte Carlo simulation of the minority_sim rule."""

    @staticmethod
    def _get_conversion_prob(group_size, n_inflexible, n_app_installed):
        """Compute the probability that a group adopts the new app.

        Args:
            group_size: Number of individuals in the group.
            n_inflexible: Number of inflexible individuals in the group.
            n_app_installed: Number of individuals in the group with the new
                app installed.

        Returns:
            The probability that the group adopts the new app [0, 1].

        """
        if n_inflexible > 0:
            return 1 / (group_size - n_app_installed + 1)
        else:
            return 0

    def __init__(self, config, sim_id, seed):
        """Initialize a Simulation instance.

        Args:
            config: Dictionary with configurations. See config.yaml in the repo
                root for the parameters. The key `out_dir` should be added
                based on command line arguments.
            sim_id: String id for the simulation. Results will be saved under
                this name in `config["out_dir"]`.
            seed: Integer to initialize the PRNG.
        """
        self._seed = seed

        self._group_dist = tuple(config['group_dist'])
        self._inflex_growth = config['inflex_growth']
        self._max_t = config['max_t']
        self._n_pop = int(config['n_pop'])
        self._n_groups = config['n_groups']
        self._out_dir = config['out_dir']
        self._p_inflex = config['p_inflex']
        self._target_p_app = config['target_p_app']

        self._rand = np.random.RandomState(self._seed)
        self._sim_id = sim_id

        np.testing.assert_almost_equal(sum(self._group_dist), 1)

        # 0 or 1 flags whether an individual in the population is inflexible
        # or has the new app installed.
        self._inflexibles = np.zeros((self._n_pop,), dtype=np.uint8)
        self._app_installed = np.zeros((self._n_pop,), dtype=np.uint8)
        # Set flags for the initial inflexible population.
        n_inflexible = round(self._p_inflex * self._n_pop)
        self._add_inflexibles(n_inflexible)

        # Assign individuals from the population to social groups at random.
        self._groups = []
        group_sizes = [i + 1 for i in range(len(self._group_dist))]
        for i in range(self._n_groups):
            j = 0
            # Create a random permutation of individuals.
            pop_ids = self._rand.permutation(self._n_pop)
            while j < self._n_pop:
                # Choose the next group size by sampling from the group size
                # distribution.
                k = self._rand.choice(group_sizes, p=self._group_dist)
                # Last group might be larger than the remaining population.
                g_size = min(k, self._n_pop - j)
                # Take group size individuals from the random permutation of
                # the population and add them to the current group.
                group_ids = []
                for _ in range(g_size):
                    group_ids.append(pop_ids[j])
                    j += 1
                self._groups.append(group_ids)

    def _add_inflexibles(self, n_new):
        """Add a fixed number of individuals to the inflexible subset.

        The new inflexibles are chosen at random from the neutral population.

        Args:
            n_new: The number of new inflexibles to add.

        Returns:
            None

        """
        flex_ids = [pi for pi in range(self._n_pop)
                    if self._inflexibles[pi] == 0]
        inflex_ids = self._rand.choice(flex_ids,
                                       size=(n_new,))
        for pid in inflex_ids:
            self._inflexibles[pid] = 1
            self._app_installed[pid] = 1

    def _save_results(self, results):
        """Save the simulation results to the output directory.

        Args:
            results: A list of tuples with `t`, `P_app` and `P_inflex` values
                for each time step.

        Returns:
            None

        """
        fn = self._sim_id + '.csv'
        with open(self._out_dir / fn, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'p_app', 'p_inflex'])
            for row in results:
                writer.writerow(row)

    def run(self):
        """Run a simulation.

        Results are saved to `configs["out_dir"]` where the file name is the
        simulation id.

        Returns:
            None

        """
        p_app = np.mean(self._app_installed)
        p_inflex = np.mean(self._inflexibles)
        n_inflexibles = np.sum(self._inflexibles)
        t = 0
        results = [(t, p_app, p_inflex)]
        # Run the simulation until a maximum number of time steps is reached
        # or the new app's conversion has reached the preset target.
        while t < self._max_t and p_app < self._target_p_app:
            # Calculate the probability of conversion for each group
            for p_ids in self._groups:
                n_inflex = np.sum(self._inflexibles[p_ids])
                n_app_installed = np.sum(self._app_installed[p_ids])
                p_c = self._get_conversion_prob(len(p_ids),
                                                n_inflex,
                                                n_app_installed)
                # If the sampled probability is less than the probability of
                # conversion, the group has adopted the new app.
                if self._rand.random_sample() < p_c:
                    for pid in p_ids:
                        self._app_installed[pid] = 1

            # Grow the inflexible population according to the preset growth
            # factor.
            n_new_inflex = int(round(
                n_inflexibles * self._inflex_growth - n_inflexibles))
            self._add_inflexibles(n_new_inflex)
            n_inflexibles += n_new_inflex

            # Compute metrics for the time step.
            p_app = np.mean(self._app_installed)
            p_inflex = np.mean(self._inflexibles)
            t += 1
            results.append((t, p_app, p_inflex))

        self._save_results(results)


def run_in_process(*args):
    """Run a simulation.

    Args:
        *args: Same arguments as Simulation.__init__.

    Returns:
        The time the simulation took in seconds.

    """
    t0 = time.perf_counter()
    sim = Simulation(*args)
    sim.run()
    dt = time.perf_counter() - t0
    return dt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monte Carlo simulation of '
                                                 'the minority_sim rule')
    parser.add_argument('--config', default='./config.yaml', type=Path,
                        help='Path to a YAML file with configs')
    parser.add_argument('--out_dir', default='./results', type=Path,
                        help='Directory to save the simulation results to')
    parser.add_argument('--overwrite', action='store_true',
                        help='Whether to overwrite the contents the output '
                             'directory')
    parser.add_argument('--max_workers', default=None, type=int,
                        help='Maximum parallel simulations to run. Defaults '
                             'to the number of available cores')
    args = parser.parse_args()
    print('Config file: "{}"'.format(args.config))
    print('Output directory: "{}"'.format(args.out_dir))

    if args.overwrite:
        try:
            shutil.rmtree(args.out_dir)
        except FileNotFoundError:
            pass

    if not args.out_dir.exists():
        os.makedirs(args.out_dir)
    elif not args.out_dir.is_dir():
        print('out_dir: "{}" is not a directory'.format(args.out_dir))
        sys.exit(1)
    elif len(list(args.out_dir.iterdir())) != 0:
        print('out_dir: "{}" is not empty'.format(args.out_dir))
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    shutil.copy(args.config, args.out_dir / 'config.yaml')
    config['out_dir'] = args.out_dir

    with ProcessPoolExecutor(args.max_workers) as executor:
        n_sim = config['n_sim']
        configs = repeat(config, n_sim)
        sim_ids = ['sim_{}'.format(i) for i in range(n_sim)]
        seeds = [i for i in range(n_sim)]
        m = executor.map(run_in_process, configs, sim_ids, seeds)
        print('Starting {} simulations'.format(n_sim))
        for i, dt in enumerate(m):
            minutes = round(dt / 60, 2)
            print(('Finished {}/{} simulations. Elapsed time: {} '
                   'minutes').format(i + 1, n_sim, minutes))

    print('Success! Saved results to: "{}"'.format(args.out_dir))


