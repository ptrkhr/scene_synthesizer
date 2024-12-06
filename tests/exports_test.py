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
import yourdfpy

# SRL
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa

from .test_utils import _skip_if_file_is_missing

TEST_DIR = Path(__file__).parent


def _set_random_seed():
    random.seed(111)
    np.random.seed(111)

@pytest.fixture(scope="module")
def kitchen_scene():
    try:
        from scene_synthesizer import examples
    except ImportError as e:
        pytest.skip(f"Skipping because module not found: {e}")

    _set_random_seed()

    try:
        kitchen = examples.kitchen(seed=11, use_collision_geometry=False)
    except ValueError as e:
        if "is not a file" in str(e):
            pytest.skip(f"Skipping because file not found: {e}")

    return kitchen


@pytest.fixture(scope="module")
def kitchen_scene_collision():
    try:
        from scene_synthesizer import examples
    except ImportError as e:
        pytest.skip(f"Skipping because module not found: {e}")

    _set_random_seed()

    try:
        kitchen = examples.kitchen(seed=11, use_collision_geometry=True)
    except ValueError as e:
        if "is not a file" in str(e):
            pytest.skip(f"Skipping because file not found: {e}")

    return kitchen


@pytest.fixture(scope="module")
def golden_dir():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cur_dir, "golden/")


def test_export_urdf_kitchen(tmpdir, golden_dir, kitchen_scene):
    #  Fix this one
    output_fname = os.path.join(tmpdir, "kitchen.urdf")

    kitchen_scene.export(
        output_fname,
        # use_absolute_mesh_paths=True,  # This is currently required but makes this tests also unportable
    )
    # assert open(output_fname).read() == open(os.path.join(golden_dir, "kitchen.urdf")).read()

    # Currently only tests if this runs through without problems
    assert True


def test_export_urdf_kitchen_collision(tmpdir, golden_dir, kitchen_scene_collision):
    output_fname = os.path.join(tmpdir, "kitchen.urdf")
    output_fname2 = os.path.join(tmpdir, "kitchen2.urdf")

    kitchen_scene_collision.export(
        output_fname,
    )

    # test if it runs a 2nd time
    kitchen_scene_collision.export(
        output_fname2,
    )

    # compare outputs
    generated_urdf = yourdfpy.URDF.load(output_fname, build_scene_graph=False, load_meshes=False)
    generated_urdf_2 = yourdfpy.URDF.load(output_fname2, build_scene_graph=False, load_meshes=False)

    assert (
        generated_urdf == generated_urdf_2
    ), f"{output_fname} differs from {output_fname2}"


def test_export_urdf_kitchen_write_meshes(tmpdir, golden_dir, kitchen_scene):
    output_fname = os.path.join(tmpdir, "kitchen.urdf")

    kitchen_scene.export(
        output_fname,
        write_mesh_files=True,
    )

    golden_urdf_fname = os.path.join(golden_dir, "kitchen_write_meshes.urdf")
    
    generated_urdf = yourdfpy.URDF.load(output_fname, build_scene_graph=False, load_meshes=False)
    golden_urdf = yourdfpy.URDF.load(golden_urdf_fname, build_scene_graph=False, load_meshes=False)

    assert (
        generated_urdf == golden_urdf
    ), f"{output_fname} differs from {golden_urdf_fname}"


def test_export_urdf_kitchen_collision_write_meshes(tmpdir, golden_dir, kitchen_scene_collision):
    output_fname = os.path.join(tmpdir, "kitchen.urdf")

    kitchen_scene_collision.export(
        output_fname,
        write_mesh_files=True,
    )

    golden_urdf_fname = os.path.join(golden_dir, "kitchen_collision_write_meshes.urdf")

    generated_urdf = yourdfpy.URDF.load(output_fname, build_scene_graph=False, load_meshes=False)
    golden_urdf = yourdfpy.URDF.load(golden_urdf_fname, build_scene_graph=False, load_meshes=False)

    assert (
        generated_urdf == golden_urdf
    ), f"{output_fname} differs from {golden_urdf_fname}"


@_skip_if_file_is_missing
def test_export_urdf_nonwatertight_mass_setting(tmpdir):
    output_fname = os.path.join(tmpdir, "non_watertight_mass_test.urdf")

    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    microwave_asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/7236/mobility.urdf")

    # Build scene
    microwave = synth.Asset(microwave_asset_path)
    scene = synth.Scene()
    scene.add_object(microwave, "microwave", use_collision_geometry=True)

    # Change density
    scene.set_density("microwave", 0.68)

    # Store mass properties before export - these geometries are known to be non-watertight
    expected_mass, expected_com, _, _ = synth.utils.get_mass_properties(
        scene.scene.geometry["microwave/original-50.obj"]
    )
    geoms = list(
        filter(
            lambda x: x in scene.graph.nodes_geometry,
            [
                "microwave/link_0",
                "microwave/original-1.obj",
                "microwave/original-15.obj",
                "microwave/original-4.obj",
                "microwave/original-6.obj",
            ],
        )
    )
    expected_mass = sum([synth.utils.get_mass_properties(scene.geometry[g])[0] for g in geoms])

    scene.export(output_fname, single_geometry_per_link=False)

    generated_urdf = yourdfpy.URDF.load(output_fname, build_scene_graph=False, load_meshes=False)

    # Get mass properties after export
    # Note: requires float conversion for current yourdfpy version
    actual_mass = float(generated_urdf.link_map["microwave_link_0"].inertial.mass)
    # actual_com = generated_urdf.link_map["microwave_original-50.obj"].inertial.origin[:3, 3]

    assert np.allclose(expected_mass, actual_mass)
    # assert np.allclose(expected_com, actual_com)


@_skip_if_file_is_missing
def test_export_urdf_different_joint_configurations(tmpdir):
    _set_random_seed()

    urdf_1 = str(Path(tmpdir) / "scene_microwave_0.urdf")
    urdf_2 = str(Path(tmpdir) / "scene_microwave_1.urdf")

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/7236/mobility.urdf")

    scene_1 = synth.Asset(asset_path).scene()
    state_1 = scene_1.get_configuration()
    scene_1.export(urdf_1)

    scene_2 = synth.Asset(asset_path).scene()
    scene_2.update_configuration(obj_id="object", configuration=[0.4])
    state_2 = scene_2.get_configuration()
    scene_2.export(urdf_2)

    # Make sure the resulting URDF is the same, i.e., the configuration did not bleed into the URDF file

    generated_urdf_1 = yourdfpy.URDF.load(urdf_1, build_scene_graph=False, load_meshes=False)
    generate_urdf_2 = yourdfpy.URDF.load(urdf_2, build_scene_graph=False, load_meshes=False)

    assert (
        generated_urdf_1 == generate_urdf_2
    ), f"URDF with default joint state: {urdf_1}  URDF with changed joint state: {urdf_2}"

    # Make sure the scenes themselves still have the same joint states as before
    assert np.allclose(
        state_1,
        scene_1.get_configuration(),
    )
    assert np.allclose(
        state_2,
        scene_2.get_configuration(),
    )


@_skip_if_file_is_missing
def test_export_obj_scale(tmpdir):
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    asset_path = os.path.join(
        asset_root_dir, "shapenetsem/Mug/10f6e09036350e92b3f21f1137c3c347.obj"
    )

    scene_1 = synth.Asset(
        fname=asset_path,
        height=0.082,
    ).scene()
    scene_1_bounds = scene_1.scene.bounds

    export_path = str(Path(tmpdir) / "scene_scale_mug.obj")
    scene_1.export(export_path)

    scene_2 = synth.Asset(export_path).scene()
    scene_2_bounds = scene_2.scene.bounds

    assert np.allclose(scene_1_bounds, scene_2_bounds)


@_skip_if_file_is_missing
def test_export_obj_as_urdf_scale(tmpdir):
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    asset_path = os.path.join(
        asset_root_dir, "shapenetsem/Mug/10f6e09036350e92b3f21f1137c3c347.obj"
    )

    scene_1 = synth.Asset(
        fname=asset_path,
        height=0.082,
    ).scene()
    scene_1_bounds = scene_1.scene.bounds

    export_path = str(Path(tmpdir) / "scene_scale_mug.urdf")
    scene_1.export(export_path)

    scene_2 = synth.Asset(export_path).scene()
    scene_2_bounds = scene_2.scene.bounds

    assert np.allclose(scene_1_bounds, scene_2_bounds)


@_skip_if_file_is_missing
def test_export_urdf_scale(tmpdir):
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/7236/mobility.urdf")

    scene_1 = synth.Asset(
        fname=asset_path,
        height=0.082,
    ).scene()
    scene_1_bounds = scene_1.scene.bounds

    export_path = str(Path(tmpdir) / "scene_scale_microwave.urdf")
    scene_1.export(export_path, use_absolute_mesh_paths=True, mesh_path_prefix="file://")

    scene_2 = synth.Asset(export_path).scene()
    scene_2_bounds = scene_2.scene.bounds

    assert np.allclose(scene_1_bounds, scene_2_bounds)


def test_mass_urdf(tmpdir):
    _set_random_seed()

    desired_mass = 12.4
    s = pa.RefrigeratorAsset(1, 1, 1, mass=desired_mass).scene()

    export_path = str(Path(tmpdir) / "scene_mass_fridge.urdf")
    s.export(export_path)

    # test parsed URDF model
    t = synth.Asset(export_path)
    urdf_total_mass = sum([l.inertial.mass for l in t._model.robot.links if l.inertial is not None])
    assert np.allclose(urdf_total_mass, desired_mass)

    # test scene
    u = t.scene()
    assert np.allclose(u.get_mass("object"), desired_mass)


def test_center_mass_urdf(tmpdir):
    _set_random_seed()

    desired_center_mass = [1.0, 1.0, 2.0]
    s = pa.RefrigeratorAsset(1, 1, 1, center_mass=desired_center_mass).scene()
    assert np.allclose(s.get_center_mass(["object"]), desired_center_mass)

    export_path = str(Path(tmpdir) / "scene_center_mass_fridge.urdf")
    s.export(export_path)

    # test imported URDF scene
    u = synth.Asset(export_path).scene()
    assert np.allclose(u.get_center_mass(["object"]), desired_center_mass, atol=25e-4)


@_skip_if_file_is_missing
def test_export_primitives_gltf(tmpdir):
    from scene_synthesizer import examples

    _set_random_seed()

    goal_path = str(Path(tmpdir) / "primitives.gltf")
    s = examples.primitives_scene()
    s.export(goal_path)

    result = trimesh.load(goal_path)

    assert len(result.geometry) == len(s.geometry)

    assert "capsule/geometry_0" in result.geometry

    assert all(
        result.geometry["capsule/geometry_0"].visual.main_color
        == s.geometry["capsule/geometry_0"].visual.main_color
    )
