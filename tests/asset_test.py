# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Standard Library
import itertools
import os
import random
from pathlib import Path

# Third Party
import numpy as np
import pytest
import trimesh
import trimesh.transformations as tra

# SRL
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa

TEST_DIR = Path(__file__).parent


def _set_random_seed():
    random.seed(111)
    np.random.seed(111)


@pytest.fixture(scope="module")
def asset():
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    microwave_asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/7236/mobility.urdf")

    asset = synth.Asset(microwave_asset_path)

    return asset


@pytest.fixture(scope="module")
def asset_scale_2():
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    microwave_asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/7236/mobility.urdf")

    asset = synth.Asset(microwave_asset_path, scale=2.0)

    return asset


def test_extents(asset, asset_scale_2):
    expected_extents = np.array([0.769314, 1.586709, 0.961642])
    assert np.allclose(asset.get_extents(), asset.scene().get_extents())
    assert np.allclose(expected_extents, asset.get_extents())

    assert np.allclose(asset_scale_2.get_extents(), asset_scale_2.scene().get_extents())
    assert np.allclose(2.0 * expected_extents, asset_scale_2.get_extents())


def test_bounds(asset, asset_scale_2):
    expected_bounds = np.array([[-0.391106, -0.840725, -0.4829], [0.378208, 0.745984, 0.478742]])

    assert np.allclose(asset.get_bounds(), asset.scene().get_bounds())
    assert np.allclose(expected_bounds, asset.get_bounds())
    assert np.allclose(np.sum(np.abs(asset.get_bounds()), axis=0), asset.get_extents())

    assert np.allclose(asset_scale_2.get_bounds(), asset_scale_2.scene().get_bounds())
    assert np.allclose(2.0 * expected_bounds, asset_scale_2.get_bounds())


def test_center_mass(asset, asset_scale_2):
    expected_com = np.array([-0.03926604, -0.05179138, 0.00074503])

    assert np.allclose(asset.get_center_mass(), asset.scene().get_center_mass())
    assert np.allclose(expected_com, asset.get_center_mass())

    assert np.allclose(asset_scale_2.get_center_mass(), asset_scale_2.scene().get_center_mass())
    assert np.allclose(2.0 * expected_com, asset_scale_2.get_center_mass(), atol=1e-3)


def test_centroid(asset, asset_scale_2):
    expected_centroid = np.array([-0.01922834, -0.04652182, -0.00208919])

    assert np.allclose(asset.get_centroid(), asset.scene().get_centroid())
    assert np.allclose(expected_centroid, asset.get_centroid())

    assert np.allclose(asset_scale_2.get_centroid(), asset_scale_2.scene().get_centroid())
    assert np.allclose(2.0 * expected_centroid, asset_scale_2.get_centroid(), atol=1e-3)


def test_reference_frame(asset, asset_scale_2):
    expected_T = np.array(
        [
            [1.0, 0.0, 0.0, -0.03926604],
            [0.0, 1.0, 0.0, 0.745984],
            [0.0, 0.0, 1.0, -0.4829],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert np.allclose(expected_T, asset.get_reference_frame(("com", "top", "bottom")))

    expected_T[:3, 3] *= 2.0
    expected_T[0, 3] = -0.07813501  # this due to CoM
    assert np.allclose(expected_T, asset_scale_2.get_reference_frame(("com", "top", "bottom")))
