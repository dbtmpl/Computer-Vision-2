#!/bin/python
from icp import calc_icp, load_point_cloud
from itertools import product
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import os.path

plt.style.use('bmh')


RESULT_FILE = 'measure_experiment.npy'
TREE_FILE = 'kdtree_pydescent.npy'


def create_padded_array(llist):
    w, h = max([len(x) for x in llist]), len(llist)
    arr = np.full((h,w), np.nan)
    for idx, _el in enumerate(llist):
        arr[idx, 0:len(_el)] = _el
        arr[idx, len(_el):] = _el[-1]
    return arr


def random_rotation(magnitude=1.0):
    # https://math.stackexchange.com/a/442423
    assert(0. < magnitude < 1.)

    # Generate random euler angles
    φ, θ, ψ = 2*np.pi*np.random.rand(3)*magnitude
    ψ = 0

    ax = np.array([[1, 0, 0], [0, np.cos(φ), -np.sin(φ)], [0, np.sin(φ), np.cos(φ)]])
    ay = np.array([[np.cos(θ), 0, np.sin(θ)], [0, 1, 0], [-np.sin(θ), 0, np.cos(θ)]])
    az = np.array([[np.cos(ψ), -np.sin(ψ), 0], [np.sin(ψ), np.cos(ψ), 0], [0, 0, 1]])

    return az @ ay @ ax


def measure_one_pair(idx, sample_size=5000):
    base_cloud, base_points, base_normals = load_point_cloud(idx)
    _, target_points, target_normals = load_point_cloud(idx + 1)
    base_colors = np.asarray(base_cloud.colors)

    sampling_techniques = ["inf_reg", "uniform", "rnd_i", "all"]
    noise_levels = [0., 0.01, 0.05, 0.1]
    stability_magnitudes = [0.01, 0.1, 0.5]

    results = []

    def calc_noise(points, std):
        maximas = points.max(axis=0) - points.min(axis=0)
        noise = np.random.randn(*points.shape) * maximas * std
        return noise

    def time_icp_run(__base, __target, __technique, __noise, __stability):
        print('[perf_exp] sampling={}, noise={}, stability_magnitude={}'.format(__technique, __noise, __stability))
        start_time = time.time()
        _, _, errors = calc_icp(__base, __target, base_normals, target_normals, base_colors,
                                sampling_tech=__technique, sample_size=sample_size)
        passed_time = time.time() - start_time

        return {'noise_level': __noise, 'errors': errors, 'time': passed_time,
                'technique': __technique, 'sample_size': sample_size, 'stability_magnitude': __stability}

    for sampling_technique, magnitude in product(sampling_techniques, stability_magnitudes):
        sizes = base_points.max(axis=0) - base_points.min(axis=0)
        trans = (2*np.random.random(3)-1) * sizes * magnitude
        trans[2] = 0
        rot = random_rotation(magnitude)
        _base_points = base_points @ rot.T + trans
        results.append(
            time_icp_run(_base_points, target_points, sampling_technique, 0, magnitude))

    for sampling_technique, noise_level in product(sampling_techniques, noise_levels):
        # Add noise
        base_noise = calc_noise(base_points, noise_level)
        target_noise = calc_noise(target_points, noise_level)
        results.append(
            time_icp_run(base_points + base_noise, target_points + target_noise, sampling_technique, noise_level, 0))

    return results


def measure_experiment(count=10):
    if os.path.exists(RESULT_FILE):
        results = np.load(RESULT_FILE)
    else:
        results = np.array([])
    indexes = np.random.randint(0, 99, count)
    for idx in indexes:
        results = np.append(results, measure_one_pair(idx))
        np.save(RESULT_FILE, results)


def plot_results():
    def _gplot():
        textwidth = 452. / 72
        golden_mean = (5 ** .5 - 1) / 2
        fig = plt.figure(figsize=(textwidth, textwidth * golden_mean))
        return fig.add_subplot(111)

    _list = np.load(RESULT_FILE)
    df = pd.DataFrame.from_records(_list)
    techniques = df.technique.unique()
    noises = sorted(df.noise_level.unique())
    magnitudes = sorted(df.stability_magnitude.unique()[1:])  # Remove zero

    ax = _gplot()
    ax.set_yscale('log')
    for technique in techniques:
        y = []
        for noise in noises:
            error_list = df[(df.technique == technique) & (df.noise_level == noise) & (df.stability_magnitude == 0)].errors
            mean_length = np.mean(list(map(lambda x: len(x), error_list)))
            mean_error = create_padded_array(error_list).mean(axis=0).min()
            y.append(mean_length)
        ax.plot(noises, y, 's--', label='{}'.format(technique))
    plt.legend()
    plt.ylabel('time/sec')
    plt.xlabel('s Noise')
    plt.gcf().savefig('figures/noise_length.pdf', dpi=200)

    ax = _gplot()
    ax.set_yscale('log')
    for technique in techniques:
        y = []
        for magnitude in magnitudes:
            error_list = df[(df.technique == technique) & (df.stability_magnitude == magnitude)].errors
            mean_length = np.mean(list(map(lambda x: len(x), error_list)))
            mean_error = create_padded_array(error_list).mean(axis=0).min()
            y.append(mean_length)
        ax.plot(magnitudes, y, 's--', label='{}'.format(technique))
    plt.legend()
    plt.ylabel('time/sec')
    plt.xlabel('m Magnitude')
    plt.gcf().savefig('figures/stability_length.pdf', dpi=200)


def kdtree_vs_pydescent(count=10):
    indexes = np.random.randint(0, 99, count)
    results = []
    for idx in indexes:
        base_cloud, base_points, base_normals = load_point_cloud(idx)
        _, target_points, target_normals = load_point_cloud(idx + 1)
        base_colors = np.asarray(base_cloud.colors)
        for method in ['kdtree', 'ckdtree', 'pynndescent']:
            start_time = time.time()
            _, _, errors = calc_icp(base_points, target_points, base_normals, target_normals, base_colors,
                                    sampling_tech='inf_reg', sample_size=5000, tree_method=method)
            passed_time = time.time() - start_time
            results.append({'errors': errors, 'time': passed_time, 'method': method})
            print('[treeexp] idx={}, method={}, time={}'.format(idx, method, passed_time))
        np.save(TREE_FILE, results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run', dest='run', action='store_true')
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--tree', dest='tree', action='store_true')
    args = parser.parse_args()

    if args.tree:
        kdtree_vs_pydescent()
    if args.run:
        measure_experiment()
    if args.plot:
        plot_results()
