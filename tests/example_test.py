# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Standard Library
import os
import random
from pathlib import Path

# Third Party
import numpy as np
import pytest

# SRL
from scene_synthesizer import procedural_assets as pa

from .test_utils import _skip_if_file_is_missing

def _set_random_seed():
    random.seed(111)
    np.random.seed(111)


@pytest.fixture(scope="module")
def kitchen_scene():
    try:
        from scene_synthesizer import examples
    except ModuleNotFoundError as e:
        pytest.skip(f"Skipping because module not found: {e}")

    _set_random_seed()

    try:
        kitchen = examples.kitchen(seed=0, use_collision_geometry=False)
    except ValueError as e:
        if "is not a file" in str(e):
            pytest.skip(f"Skipping because file not found: {e}")

    return kitchen


@pytest.fixture(scope="module")
def kitchen_scene_collision():
    try:
        from scene_synthesizer import examples
    except ModuleNotFoundError as e:
        pytest.skip(f"Skipping because module not found: {e}")

    _set_random_seed()

    try:
        kitchen = examples.kitchen(seed=0, use_collision_geometry=True)
    except ValueError as e:
        if "is not a file" in str(e):
            pytest.skip(f"Skipping because file not found: {e}")

    return kitchen


def test_kitchen_bounds(kitchen_scene):
    expected_bounds = np.array([[-3.4075855, -1.22330573, -0.075], [0.46, 3.37451051, 3.0]])
    bounds = kitchen_scene.get_bounds()

    assert np.allclose(expected_bounds, bounds), f"expected: {expected_bounds}  actual: {bounds}"


def test_kitchen_collision_bounds(kitchen_scene_collision):
    expected_bounds = np.array([[-3.4075855, -1.22330573, -0.075], [0.46, 3.37451051, 3.0]])
    bounds = kitchen_scene_collision.get_bounds()

    assert np.allclose(expected_bounds, bounds), f"expected: {expected_bounds}  actual: {bounds}"


@_skip_if_file_is_missing
def test_vertical_wall_oven_microwave_cabinet():
    from scene_synthesizer import examples

    s = examples.vertical_wall_oven_microwave_cabinet()

    vertex_hist = np.histogram(s.scene.dump(concatenate=True).vertices)
    expected_hist = (
        np.array([6194, 3683, 865, 2886, 189, 620, 1108, 324, 48, 64]),
        np.array([-0.3, -0.07, 0.16, 0.39, 0.62, 0.85, 1.08, 1.31, 1.54, 1.77, 2.0]),
    )

    for h1, h2 in zip(expected_hist, vertex_hist):
        assert np.allclose(h1, h2)
