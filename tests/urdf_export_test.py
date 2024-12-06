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
import trimesh
import trimesh.transformations as tra
import yourdfpy

# SRL
import scene_synthesizer as synth

from .test_utils import _skip_if_file_is_missing

TEST_DIR = Path(__file__).parent


def _set_random_seed():
    random.seed(111)
    np.random.seed(111)


@_skip_if_file_is_missing
def test_primitives_urdf_export(tmp_path):
    from scene_synthesizer import examples

    _set_random_seed()

    urdf_expected = str(TEST_DIR / "golden/primitives_example_scene.urdf")
    urdf_actual = str(Path(tmp_path) / "primitives_example_scene.urdf")

    x = examples.primitives_scene(use_trimesh_primitive=True, seed=11)

    with pytest.raises(ValueError, match="URDF doesn't have a capsule primitive."):
        x.export(urdf_actual, single_geometry_per_link=True)

    x.remove_object("capsule")

    x.export(urdf_actual, single_geometry_per_link=True)

    generated_urdf = yourdfpy.URDF.load(urdf_actual, build_scene_graph=False, load_meshes=False)
    golden_urdf = yourdfpy.URDF.load(urdf_expected, build_scene_graph=False, load_meshes=False)

    assert (
        generated_urdf == golden_urdf
    ), f"{urdf_actual} differs from {urdf_expected}"


@_skip_if_file_is_missing
def test_primitives_urdf_export_single_link(tmp_path):
    from scene_synthesizer import examples

    _set_random_seed()

    urdf_expected = str(TEST_DIR / "golden/primitives_example_scene_single_link.urdf")
    urdf_actual = str(Path(tmp_path) / "primitives_example_scene_single_link.urdf")

    x = examples.primitives_scene(use_trimesh_primitive=True, seed=11)
    with pytest.raises(ValueError, match="URDF doesn't have a capsule primitive."):
        x.export(urdf_actual, single_geometry_per_link=False)

    x.remove_object("capsule")

    x.export(urdf_actual, single_geometry_per_link=False)

    generated_urdf = yourdfpy.URDF.load(urdf_actual, build_scene_graph=False, load_meshes=False)
    golden_urdf = yourdfpy.URDF.load(urdf_expected, build_scene_graph=False, load_meshes=False)

    assert (
        generated_urdf == golden_urdf
    ), f"{urdf_actual} differs from {urdf_expected}"


@_skip_if_file_is_missing
def test_primitives_urdf_export_single_link_with_separate_assets(tmp_path):
    from scene_synthesizer import examples

    _set_random_seed()

    urdf_actual = str(Path(tmp_path) / "primitives_example_scene_single_link.urdf")

    x = examples.primitives_scene(use_trimesh_primitive=True, seed=11)
    with pytest.raises(ValueError, match="URDF doesn't have a capsule primitive."):
        x.export(urdf_actual, single_geometry_per_link=False, separate_assets=True)

    x.remove_object("capsule")

    x.export(urdf_actual, single_geometry_per_link=False, separate_assets=True)

    for fname in ("cylinder.urdf", "sphere.urdf", "box.urdf"):
        urdf_expected = str(TEST_DIR / "golden/" / fname)
        urdf_actual = str(Path(tmp_path) / fname)
        
        generated_urdf = yourdfpy.URDF.load(urdf_actual, build_scene_graph=False, load_meshes=False)
        golden_urdf = yourdfpy.URDF.load(urdf_expected, build_scene_graph=False, load_meshes=False)

        assert (
            generated_urdf == golden_urdf
        ), f"{urdf_actual} differs from {urdf_expected}"


@_skip_if_file_is_missing
def test_primitives_2_urdf_export(tmp_path):
    from scene_synthesizer import examples

    _set_random_seed()

    urdf_expected = str(TEST_DIR / "golden/primitives_example_scene_2.urdf")
    urdf_actual = str(Path(tmp_path) / "primitives_example_scene_2.urdf")

    x = examples.primitives_scene(use_trimesh_primitive=False, seed=11)
    x.remove_object("capsule")
    x.export(urdf_actual, single_geometry_per_link=True)

    generated_urdf = yourdfpy.URDF.load(urdf_actual, build_scene_graph=False, load_meshes=False)
    golden_urdf = yourdfpy.URDF.load(urdf_expected, build_scene_graph=False, load_meshes=False)

    assert (
        generated_urdf == golden_urdf
    ), f"{urdf_actual} differs from {urdf_expected}"


@_skip_if_file_is_missing
def test_primitives_2_urdf_export_single_link(tmp_path):    
    from scene_synthesizer import examples

    _set_random_seed()

    urdf_expected = str(TEST_DIR / "golden/primitives_example_scene_2_single_link.urdf")
    urdf_actual = str(Path(tmp_path) / "primitives_example_scene_2.urdf")

    x = examples.primitives_scene(use_trimesh_primitive=False, seed=11)
    x.remove_object("capsule")
    x.export(urdf_actual, single_geometry_per_link=False)

    generated_urdf = yourdfpy.URDF.load(urdf_actual, build_scene_graph=False, load_meshes=False)
    golden_urdf = yourdfpy.URDF.load(urdf_expected, build_scene_graph=False, load_meshes=False)

    assert (
        generated_urdf == golden_urdf
    ), f"{urdf_actual} differs from {urdf_expected}"


@_skip_if_file_is_missing
def test_usd_to_urdf_export(tmp_path):
    _set_random_seed()

    usd_source = str(TEST_DIR / "data/assets/srl_kitchen/srl-top-cabinet.usd")
    urdf_sink = os.path.join(tmp_path, "srl_top_cabinet.urdf")
    usd_asset = synth.Asset(usd_source)
    x = synth.Scene(seed=11)
    x.add_object(usd_asset, 'object', use_collision_geometry=False)
    x.export(urdf_sink)

    y = synth.Asset(urdf_sink).scene(use_collision_geometry=False)

    assert len(y.get_joint_names()) == len(x.get_joint_names())
    assert len(y.geometry) == len(x.geometry)
    assert np.allclose(y.scene.dump(concatenate=True).bounds, x.scene.dump(concatenate=True).bounds)
