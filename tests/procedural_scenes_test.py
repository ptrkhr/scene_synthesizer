# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Standard Library
import os
import random

# Third Party
import numpy as np
import pytest
import yourdfpy

# SRL
import scene_synthesizer as synth
from scene_synthesizer import procedural_scenes as ps


@pytest.fixture(scope="module")
def golden_dir():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cur_dir, "golden/")


def _set_random_seed():
    random.seed(111)
    np.random.seed(111)


def test_kitchen_single_wall():
    s = ps.kitchen_single_wall()

    assert True


def test_kitchen_l_shaped():
    s = ps.kitchen_l_shaped()

    assert True


def test_kitchen_galley():
    s = ps.kitchen_galley()

    assert True


def test_kitchen_island():
    s = ps.kitchen_island()

    assert True


def test_kitchen_peninsula():
    s = ps.kitchen_peninsula()

    assert True


def test_kitchen_u_shaped():
    seeds = [100, 101, 102]

    expected_vertex_histograms = {
        100: (
            np.array( [436,  464,  360,   60,  760,  720, 1544,  780, 1396,  824]),
            np.array(
                [
                    -2.76396503, -2.26204487, -1.7601247 , -1.25820453, -0.75628436,-0.25436419,  0.24755597,  0.74947614,  1.25139631,  1.75331648, 2.25523665
                ]
            ),
        ),
        101: (
            np.array([412,  420,  232,  408,  480, 1084, 1332,  936, 1620,  948]),
            np.array(
                [
                   -2.88167482, -2.37073192, -1.85978901, -1.34884611, -0.8379032 , -0.3269603 ,  0.18398261,  0.69492551,  1.20586841,  1.71681132, 2.22775422
                ]
            ),
        ),
        102: (
            np.array([452,  460,  200,  264,  220, 1160, 1168,  884, 1412,  956]),
            np.array(
                [
                   -2.9159673 , -2.40523672, -1.89450615, -1.38377557, -0.873045, -0.36231442,  0.14841615,  0.65914673,  1.1698773 ,  1.68060788, 2.19133846
                ]
            ),
        ),
    }

    # test_results = []

    for seed in seeds:
        s = ps.kitchen_u_shaped(seed=seed, **ps.use_primitives_only())

        for d1, d2 in zip(
            expected_vertex_histograms[seed], np.histogram(s.scene.dump(concatenate=True).vertices)
        ):
            assert np.allclose(d1, d2)
            # success = np.allclose(d1, d2)

            # if not success:
            #     print(f"For seed {seed} use {d2}")

            # test_results.append(success)
    
    # assert (all(test_results))
