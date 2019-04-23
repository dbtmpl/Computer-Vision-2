#!/bin/python
from run_experiments import calc_icp, load_point_cloud
from itertools import product
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from argparse import ArgumentParser


RESULT_FILE = 'measure_experiment.npy'


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
    θ = 0

    ax = np.array([[1, 0, 0], [0, np.cos(φ), -np.sin(φ)], [0, np.sin(φ), np.cos(φ)]])
    ay = np.array([[np.cos(θ), 0, np.sin(θ)], [0, 1, 0], [-np.sin(θ), 0, np.cos(θ)]])
    az = np.array([[np.cos(ψ), -np.sin(ψ), 0], [np.sin(ψ), np.cos(ψ), 0], [0, 0, 1]])

    return az @ ay @ ax


def measure_one_pair(idx, sample_size=5000):
    _, base_points, base_normals = load_point_cloud(idx)
    _, target_points, target_normals = load_point_cloud(idx + 1)

    sampling_techniques = ["all", "uniform", "rnd_i", "inf_reg"]
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
        _, _, errors = calc_icp(__base, __target, base_normals, target_normals, sampling_tech=__technique,
                                sample_size=sample_size)
        passed_time = time.time() - start_time

        return {'noise_level': __noise, 'errors': errors, 'time': passed_time,
                'technique': __technique, 'sample_size': sample_size, 'stability_magnitude': __stability}

    for sampling_technique, noise_level in product(sampling_techniques, noise_levels):
        # Add noise
        base_noise = calc_noise(base_points, noise_level)
        target_noise = calc_noise(target_points, noise_level)
        results.append(
            time_icp_run(base_points + base_noise, target_points + target_noise, sampling_technique, noise_level, 0))

    for sampling_technique, magnitude in product(sampling_techniques, stability_magnitudes):
        sizes = base_points.max(axis=0) - base_points.min(axis=0)
        trans = (2*np.random.random(3)-1) * sizes * magnitude
        rot = random_rotation(magnitude)
        _base_points = base_points @ rot.T + trans

        results.append(time_icp_run(_base_points, target_points, sampling_technique, 0, magnitude))

    return results


def measure_experiment(count=10):
    indexes = np.random.randint(0, 99, count)
    result = []
    for idx in indexes:
        result.extend(measure_one_pair(idx))
    np.save(RESULT_FILE, result)


def plot_results():
    def _gplot():
        textwidth = 452. / 72
        golden_mean = (5 ** .5 - 1) / 2
        fig = plt.figure(figsize=(textwidth, textwidth * golden_mean))
        return fig.add_subplot(111)

    _list = np.load(RESULT_FILE)
    df = pd.DataFrame.from_records(_list)
    techniques = df.technique.unique()
    noises = df.noise_level.unique()
    magnitudes = df.stability_magnitude.unique()[1:]  # Remove zero

    ax = _gplot()
    for technique, noise in product(techniques, noises):
        error_list = df[(df.technique == technique) & (df.noise_level == noise)].errors
        mean_error = create_padded_array(error_list).mean(axis=0)
        ax.plot(mean_error, label='{}_{}'.format(technique, noise))
    plt.legend()
    plt.tight_layout()
    plt.gcf().savefig('figures/noise.pdf', dpi=200)

    ax = _gplot()
    ax.set_yscale('log')
    for technique, magnitude in product(techniques, magnitudes):
        error_list = df[(df.technique == technique) & (df.stability_magnitude == magnitude)].errors
        mean_errors = create_padded_array(error_list).mean(axis=0)
        ax.plot(mean_errors, label='{}_{}'.format(technique, magnitude))
    plt.legend()
    plt.tight_layout()
    plt.gcf().savefig('figures/stability.pdf', dpi=200)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', dest='run', action='store_true')
    parser.add_argument('-p', dest='plot', action='store_true')
    args = parser.parse_args()

    if args.run:
        measure_experiment()
    if args.plot:
        plot_results()
