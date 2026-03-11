"""Legacy reference reconstruction utilities derived from the prior preprocessing path.

This module is retained only for benchmark comparison. It is not the owned native
Phase 2 reconstruction implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import numpy as np
import scipy.interpolate as si

from feature_pipeline.loaders.spcc_raw import normalize_event_observations
from feature_pipeline.schemas import SPCCRawEvent


@dataclass(frozen=True)
class ReconstructionArtifact:
    mode: str
    raw_sequence: list[list[float | str]]
    reconstructed_sequence: list[list[float]]


def index_min(values):
    return min(range(len(values)), key=values.__getitem__)


def spline(arr, t):
    if len(arr[0]) < 4:
        if len(arr[0]) == 0:
            arr[0] = [t[0], int((t[-1] - t[0]) / 2), t[-1]]
            arr[1] = [0, 0, 0]
            arr[2] = [0, 0, 0]
        if len(arr[0]) == 1:
            arr[0] = [t[0], arr[0][0], t[-1]]
            arr[1] = [arr[1][0], arr[1][0], arr[1][0]]
            arr[2] = [arr[2][0], arr[2][0], arr[2][0]]
        return si.InterpolatedUnivariateSpline(arr[0], arr[1], k=1)
    return si.InterpolatedUnivariateSpline(arr[0], arr[1])


def time_collector(arr, frac=1.0):
    bestclustering = True
    while bestclustering:
        grouped = []
        for _, group in groupby(arr, key=lambda n: n // (1.0 / frac)):
            seq = sorted(group)
            grouped.append(np.sum(seq) / len(seq))
        ind = []
        i = 0
        for _, group in groupby(arr, key=lambda n: n // (1.0 / frac)):
            ind.append([])
            for j in group:
                ind[i].append(index_min(abs(j - np.array(arr))))
            i += 1
        if len([indices for indices in ind if len(indices) > 4]) != 0:
            frac += 0.1
        else:
            bestclustering = False
    return grouped, ind, frac


def create_colourband_array(ind, arr, err_arr, temp_arr, err_temp_arr):
    temp = [arr[ind[i]] for i in range(len(ind)) if arr[ind[i]] != -999]
    err_temp = [err_arr[ind[i]] for i in range(len(ind)) if err_arr[ind[i]] != -999]
    if len(temp) == 0:
        temp_arr.append(-999)
        err_temp_arr.append(-999)
        out = True
    elif len(temp) > 1:
        out = False
    else:
        temp_arr.append(temp[0])
        err_temp_arr.append(err_temp[0])
        out = True
    return temp_arr, err_temp_arr, out


def fill_in_points(arr, err_arr):
    ind = np.where(np.array(arr) != -999)[0]
    length = len(arr)
    if len(ind) == 0:
        arr = [0 for _ in range(length)]
        err_arr = [0 for _ in range(length)]
    else:
        for i in range(len(ind) - 1):
            diff = ind[i + 1] - ind[i]
            arr[ind[i] + 1 : ind[i + 1]] = np.random.uniform(arr[ind[i]], arr[ind[i + 1]], diff - 1)
            err_arr[ind[i] + 1 : ind[i + 1]] = np.random.uniform(
                err_arr[ind[i]], err_arr[ind[i + 1]], diff - 1
            )
        for i in range(len(arr[: ind[0]])):
            arr[i] = arr[ind[0]]
            err_arr[i] = err_arr[ind[0]]
        for i in range(len(arr[ind[-1] + 1 :])):
            arr[ind[-1] + 1 + i] = arr[ind[-1]]
            err_arr[ind[-1] + 1 + i] = err_arr[ind[-1]]
    return arr, err_arr


def normalized_raw_sequence(event: SPCCRawEvent) -> list[list[float | str]]:
    return [
        [obs.time, obs.band, obs.flux, obs.flux_err]
        for obs in normalize_event_observations(event)
    ]


def reconstruct_last_observation(event: SPCCRawEvent) -> ReconstructionArtifact:
    obs = []
    g = r = i = z = 0
    g_error = r_error = i_error = z_error = 0
    raw = normalized_raw_sequence(event)
    for item in normalize_event_observations(event):
        if item.band == "g":
            g = item.flux
            g_error = item.flux_err
        elif item.band == "r":
            r = item.flux
            r_error = item.flux_err
        elif item.band == "i":
            i = item.flux
            i_error = item.flux_err
        elif item.band == "z":
            z = item.flux
            z_error = item.flux_err
        obs.append([item.time, g, r, i, z, g_error, r_error, i_error, z_error])
    return ReconstructionArtifact(mode="last", raw_sequence=raw, reconstructed_sequence=obs)


def reconstruct_spline(event: SPCCRawEvent, grouping=1.0) -> ReconstructionArtifact:
    raw = normalized_raw_sequence(event)
    t_arr = []
    g_arr = [[], [], []]
    r_arr = [[], [], []]
    i_arr = [[], [], []]
    z_arr = [[], [], []]
    for item in normalize_event_observations(event):
        t_arr.append(item.time)
        if item.band == "g":
            g_arr[0].append(item.time)
            g_arr[1].append(item.flux)
            g_arr[2].append(item.flux_err)
        elif item.band == "r":
            r_arr[0].append(item.time)
            r_arr[1].append(item.flux)
            r_arr[2].append(item.flux_err)
        elif item.band == "i":
            i_arr[0].append(item.time)
            i_arr[1].append(item.flux)
            i_arr[2].append(item.flux_err)
        elif item.band == "z":
            z_arr[0].append(item.time)
            z_arr[1].append(item.flux)
            z_arr[2].append(item.flux_err)
    g_spline = spline(g_arr, t_arr)
    r_spline = spline(r_arr, t_arr)
    i_spline = spline(i_arr, t_arr)
    z_spline = spline(z_arr, t_arr)
    t, _, _ = time_collector(t_arr, grouping)
    obs = [
        [
            t[i],
            g_spline(t[i]).tolist(),
            r_spline(t[i]).tolist(),
            i_spline(t[i]).tolist(),
            z_spline(t[i]).tolist(),
            g_arr[2][index_min(abs(g_arr[0] - t[i]))],
            r_arr[2][index_min(abs(r_arr[0] - t[i]))],
            i_arr[2][index_min(abs(i_arr[0] - t[i]))],
            z_arr[2][index_min(abs(z_arr[0] - t[i]))],
        ]
        for i in range(len(t))
    ]
    return ReconstructionArtifact(mode="spline", raw_sequence=raw, reconstructed_sequence=obs)


def reconstruct_augment(event: SPCCRawEvent, grouping=1.0) -> ReconstructionArtifact:
    raw = normalized_raw_sequence(event)
    obs = []
    for item in normalize_event_observations(event):
        g = r = i = z = -999
        g_error = r_error = i_error = z_error = -999
        if item.band == "g":
            g = item.flux
            g_error = item.flux_err
        elif item.band == "r":
            r = item.flux
            r_error = item.flux_err
        elif item.band == "i":
            i = item.flux
            i_error = item.flux_err
        elif item.band == "z":
            z = item.flux
            z_error = item.flux_err
        obs.append([item.time, g, r, i, z, g_error, r_error, i_error, z_error])
    t_arr = [obs[i][0] for i in range(len(obs))]
    g_arr = [obs[i][1] for i in range(len(obs))]
    g_err_arr = [obs[i][5] for i in range(len(obs))]
    r_arr = [obs[i][2] for i in range(len(obs))]
    r_err_arr = [obs[i][6] for i in range(len(obs))]
    i_arr = [obs[i][3] for i in range(len(obs))]
    i_err_arr = [obs[i][7] for i in range(len(obs))]
    z_arr = [obs[i][4] for i in range(len(obs))]
    z_err_arr = [obs[i][8] for i in range(len(obs))]
    correctplacement = True
    frac = grouping
    while correctplacement:
        t, index, frac = time_collector(t_arr, frac)
        g_temp_arr = []
        g_err_temp_arr = []
        r_temp_arr = []
        r_err_temp_arr = []
        i_temp_arr = []
        i_err_temp_arr = []
        z_temp_arr = []
        z_err_temp_arr = []
        tot = []
        for i in range(len(index)):
            g_temp_arr, g_err_temp_arr, gfail = create_colourband_array(
                index[i], g_arr, g_err_arr, g_temp_arr, g_err_temp_arr
            )
            r_temp_arr, r_err_temp_arr, rfail = create_colourband_array(
                index[i], r_arr, r_err_arr, r_temp_arr, r_err_temp_arr
            )
            i_temp_arr, i_err_temp_arr, ifail = create_colourband_array(
                index[i], i_arr, i_err_arr, i_temp_arr, i_err_temp_arr
            )
            z_temp_arr, z_err_temp_arr, zfail = create_colourband_array(
                index[i], z_arr, z_err_arr, z_temp_arr, z_err_temp_arr
            )
            tot.append(gfail * rfail * ifail * zfail)
        if all(tot):
            correctplacement = False
        else:
            frac += 0.1
    g_temp_arr, g_err_temp_arr = fill_in_points(g_temp_arr, g_err_temp_arr)
    r_temp_arr, r_err_temp_arr = fill_in_points(r_temp_arr, r_err_temp_arr)
    i_temp_arr, i_err_temp_arr = fill_in_points(i_temp_arr, i_err_temp_arr)
    z_temp_arr, z_err_temp_arr = fill_in_points(z_temp_arr, z_err_temp_arr)
    reconstructed = [
        [
            t[i],
            g_temp_arr[i],
            r_temp_arr[i],
            i_temp_arr[i],
            z_temp_arr[i],
            g_err_temp_arr[i],
            r_err_temp_arr[i],
            i_err_temp_arr[i],
            z_err_temp_arr[i],
        ]
        for i in range(len(t))
    ]
    return ReconstructionArtifact(mode="augment", raw_sequence=raw, reconstructed_sequence=reconstructed)
