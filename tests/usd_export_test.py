# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Standard Library
import colorsys
import contextlib
import filecmp
import os
import random
from pathlib import Path
from typing import Any, Optional

# Third Party
import meshsets
import numpy as np
import trimesh
import trimesh.transformations as tra

# SRL
import scene_synthesizer as synth
from scene_synthesizer import examples, procedural_assets

TEST_DIR = Path(__file__).parent


def _set_random_seed():
    random.seed(111)
    np.random.seed(111)


@contextlib.contextmanager
def random_state_context(seed: Optional[Any] = None):
    """A context manager for the Python global random number generator.

    Args:
        seed: When not `None`, the new random seed.
    """
    state = random.getstate()
    if seed is not None:
        random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


@contextlib.contextmanager
def numpy_random_state_context(seed: Optional[int] = None):
    """A context manager for the numpy global random number generator.

    Args:
        seed: When not `None`, the new random seed.
    """
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def generate_scene(usd_path) -> None:
    """Generate the microwave ball scene."""
    _set_random_seed()

    dataset_partnet_mobility = meshsets.load_dataset("PartNet Mobility")
    dataset_shapenetsem = meshsets.load_dataset("ShapeNetSem watertight")
    dataset_modelnet = meshsets.load_dataset("ModelNet40 auto-aligned watertight")

    datasets = {
        "partnet_mobility": dataset_partnet_mobility,
        "shapenetsem": dataset_shapenetsem,
        "modelnet": dataset_modelnet,
    }

    # Set parameters
    MAX_RANDINT = 2**32 - 1
    rand_seed = 0
    table_height = 0.72
    microwave_height = 0.4
    ball_height = 0.05
    rand_generator = np.random.RandomState(rand_seed)

    # Synthesize scene
    fname_microwave = datasets["partnet_mobility"].get_filename(
        categories="Microwave", pattern="7236/mobility.urdf"
    )
    fname_table = datasets["shapenetsem"].get_filename(
        categories="DiningTable", pattern="7c10130e50d27692435807c7a815b457.obj"
    )

    assets = {
        "microwave": synth.Asset(fname_microwave, height=microwave_height),
        "table": synth.Asset(
            fname_table, height=table_height, origin=("center", "center", "bottom")
        ),
        "ball": synth.TrimeshAsset(
            mesh=trimesh.primitives.Sphere(),
            height=ball_height,
            origin=("center", "center", "bottom"),
        ),
    }

    scene = synth.Scene(seed=111)
    scene.add_object(assets["table"], "table", joint_type=None)
    scene.label_support("table_surface", obj_ids="table")
    scene.add_object(
        assets["microwave"],
        "microwave",
        connect_parent_id="table",
        connect_parent_anchor=("center", "center", "top"),
        connect_obj_anchor=("center", "center", "bottom"),
        joint_type="fixed",
        use_collision_geometry=None,
    )
    scene.label_support("microwave_surface", obj_ids="microwave", layer="visual")
    with numpy_random_state_context(rand_generator.randint(0, MAX_RANDINT)):
        with random_state_context(rand_generator.randint(0, MAX_RANDINT)):
            scene.place_object(
                "ball", assets["ball"], support_id="microwave_surface", joint_type="floating"
            )
    scene.colorize(specific_objects={"ball": [255, 0, 0, 255]})

    # Remove auto-added lights and add a point light
    while scene.scene.lights:
        scene.scene.lights.pop()

    # Add a point light
    light = trimesh.scene.lighting.PointLight(name="light", intensity=300000, radius=1)
    scene.scene.lights.append(light)
    scene.graph[light.name] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 10], [0, 0, 0, 1]])

    # Export USD scene
    scene.export(usd_path, include_light_nodes=True)


def test_microwave_ball_usd_export(tmp_path) -> None:
    """Test `Scene.export` to USD method."""

    usd_actual = str(Path(tmp_path) / "microwave_ball_actual.usda")
    generate_scene(usd_actual)
    usd_expected = str(TEST_DIR / "golden/microwave_ball.usda")

    assert filecmp.cmp(
        usd_expected, usd_actual, shallow=False
    ), f"Actual USD: {usd_actual}  Expected USD: {usd_expected}"


def test_cabinet_usd_export(tmp_path):
    _set_random_seed()

    usd_expected = str(TEST_DIR / "golden/cabinet_box_handles.usda")
    usd_actual = str(Path(tmp_path) / "cabinet_box_handles.usda")

    x = procedural_assets.RecursivelyPartitionedCabinetAsset(1, 2, 3, use_box_handle=True, seed=111).scene()
    x.export(usd_actual)

    assert filecmp.cmp(
        usd_expected, usd_actual, shallow=False
    ), f"Actual USD: {usd_actual}  Expected USD: {usd_expected}"


def test_primitives_usd_export(tmp_path):
    _set_random_seed()

    usd_expected = str(TEST_DIR / "golden/primitives_example_scene.usda")
    usd_actual = str(Path(tmp_path) / "primitives_example_scene.usda")

    x = examples.primitives_scene(use_trimesh_primitive=True, seed=11)
    x.export(usd_actual)

    assert filecmp.cmp(
        usd_expected, usd_actual, shallow=False
    ), f"Actual USD: {usd_actual}  Expected USD: {usd_expected}"


def test_primitives_2_usd_export(tmp_path):
    _set_random_seed()

    usd_expected = str(TEST_DIR / "golden/primitives_example_scene_2.usda")
    usd_actual = str(Path(tmp_path) / "primitives_example_scene_2.usda")

    x = examples.primitives_scene(use_trimesh_primitive=False, seed=11)
    x.export(usd_actual)

    assert filecmp.cmp(
        usd_expected, usd_actual, shallow=False
    ), f"Actual USD: {usd_actual}  Expected USD: {usd_expected}"


def test_scene_with_fixed_joint(tmp_path):
    _set_random_seed()

    usd_expected = str(TEST_DIR / "golden/scene_with_fixed_joint.usda")
    usd_actual = str(Path(tmp_path) / "scene_with_fixed_joint.usda")

    scene = synth.Scene()
    scene.add_object(
        obj_id="box",
        asset=synth.BoxAsset(extents=[1, 1, 1]),
        joint_type="fixed",
    )
    scene.export(usd_actual)

    assert filecmp.cmp(
        usd_expected, usd_actual, shallow=False
    ), f"Actual USD: {usd_actual}  Expected USD: {usd_expected}"


def test_joint_properties(tmp_path):
    _set_random_seed()

    usd_expected = str(Path(tmp_path) / "scene_microwave_0.usda")
    usd_actual = str(Path(tmp_path) / "scene_microwave_1.usda")

    obj_id = "obj"
    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/7236/mobility.urdf")

    scene = synth.Scene()
    scene.add_object(
        obj_id=obj_id,
        asset=synth.Asset(asset_path),
        joint_type="fixed",
    )
    scene.export(usd_expected)

    scene = synth.Scene()
    scene.add_object(
        obj_id=obj_id,
        asset=synth.Asset(asset_path),
        joint_type="fixed",
    )
    for joint_name, joint_properties in scene.get_joint_properties(obj_id=obj_id).items():
        scene.update_configuration(
            obj_id=obj_id,
            joint_ids=[joint_name.split("/")[1]],
            configuration=[joint_properties["limit_upper"]],
        )
        scene.update_configuration(
            obj_id=obj_id,
            joint_ids=[joint_name.split("/")[1]],
            configuration=[0],
        )
    scene.export(usd_actual)

    assert filecmp.cmp(
        usd_expected, usd_actual, shallow=False
    ), f"Actual USD: {usd_actual}  Expected USD: {usd_expected}"


def create_color(
    hue: float, saturation: float = 0.99, value: float = 0.99, alpha: float = 1.0
) -> np.ndarray:
    """Creates a primitive Scene Synthesizer asset.

    Args:
        hue: The hue of the color.
        saturation: The saturation of the color.
        value: The value (intensity) of the color.
        alpha: The alpha (transparency) of the color.
    """
    float_color = np.append(colorsys.hsv_to_rgb(hue, saturation, value), alpha)
    return (255 * float_color).astype(np.uint8)


def create_evenly_spaced_colors(num_colors: int, initial_hue: float = 0.0, **kwargs: Any):
    """Creates a primitive Scene Synthesizer asset.

    Args:
        num_colors: The number of colors to create.
        initial_hue: The initial hue value (modulo 1.0).
        kwargs: Additional keyword arguments are passed to :meth:`~simpler.util.create_color`.
    """
    return [
        create_color(hue=hue % 1.0, **kwargs)
        for hue in np.linspace(initial_hue, initial_hue + 1.0, num=num_colors, endpoint=False)
    ]


def _create_simpler_box_scene(joint_type, seed):
    scene = synth.Scene(seed=seed)
    floor_size = 1.0
    floor = synth.BoxAsset(
        extents=[floor_size, floor_size, 0.01],
        origin=("center", "center", "bottom"),
    )
    floor_id = "floor"
    scene.add_object(
        obj_id=floor_id,
        asset=floor,
        translation=(0.0, 0.0, -0.01),
        joint_type=joint_type,
        use_collision_geometry=None,
    )
    scene.label_support(label=floor_id, obj_ids=floor_id, layer="visual")

    platform_size = 0.2
    platform = synth.BoxAsset(
        extents=[platform_size, platform_size, 0.01],
        origin=("center", "center", "bottom"),
    )
    platform_positions = [
        [+(floor_size - platform_size) / 2, 0.0, 0.0],
        [-(floor_size - platform_size) / 2, 0.0, 0.0],
    ]
    for i, platform_position in enumerate(platform_positions):
        platform_id = f"platform{i}"
        scene.add_object(
            obj_id=platform_id,
            asset=platform,
            translation=platform_position,
            joint_type=joint_type,
            use_collision_geometry=None,
        )
        scene.label_support(label=platform_id, obj_ids=platform_id, layer="visual")

    num_boxes = 2
    for i in range(num_boxes):
        box_id = f"box{i}"
        box_size = 0.05
        box = synth.BoxAsset(
            extents=box_size * np.ones(3),
            origin=("center", "center", "bottom"),
        )

        support_ids = list(scene.metadata["support_polygons"])
        random.shuffle(support_ids)

        if not scene.place_object(
            obj_id=box_id,
            obj_asset=box,
            support_id=None,
            parent_id=None,
            joint_type="floating",
            use_collision_geometry=None,
            obj_position_iterator=synth.utils.PositionIteratorUniform(),
            obj_orientation_iterator=synth.utils.orientation_generator_uniform_around_z(seed=seed),
            obj_support_id_iterator=scene.support_generator(support_ids=support_ids),
            distance_above_support=1e-3,
            max_iter=1000,
        ):
            return None

    objects = sorted(scene.metadata["object_nodes"].keys())
    colors = create_evenly_spaced_colors(num_colors=len(objects), initial_hue=random.random())
    random.shuffle(colors)
    scene.colorize(specific_objects=dict(zip(objects, colors)))

    return scene


def test_simpler_box_scene(tmp_path):
    _set_random_seed()

    usd_expected = str(TEST_DIR / "golden/scene_simpler_boxes.usda")
    usd_actual = str(Path(tmp_path) / "scene_simpler_boxes_actual.usda")

    scene = _create_simpler_box_scene(joint_type=None, seed=11)

    scene.export(usd_actual, write_attribute_attached_state=True)

    assert filecmp.cmp(
        usd_expected, usd_actual, shallow=False
    ), f"Actual USD: {usd_actual}  Expected USD: {usd_expected}"


def test_simpler_box_scene_fixed_joints(tmp_path):
    _set_random_seed()

    usd_expected = str(TEST_DIR / "golden/scene_simpler_boxes_fixed.usda")
    usd_actual = str(Path(tmp_path) / "scene_simpler_boxes_fixed_actual.usda")

    scene = _create_simpler_box_scene(joint_type="fixed", seed=11)
    scene.export(usd_actual, write_attribute_attached_state=True)

    assert filecmp.cmp(
        usd_expected, usd_actual, shallow=False
    ), f"Actual USD: {usd_actual}  Expected USD: {usd_expected}"


def test_separate_usd_assets_export(tmp_path):
    _set_random_seed()

    usd_expected = str(TEST_DIR / "golden/primitives_separate_assets_example_scene.usda")
    usd_actual = str(Path(tmp_path) / "primitives_separate_assets_example_scene.usda")

    x = examples.primitives_scene(use_trimesh_primitive=True, seed=11)
    x.export(usd_actual, separate_assets=True)

    assert filecmp.cmp(
        usd_expected, usd_actual, shallow=False
    ), f"Actual USD: {usd_actual}  Expected USD: {usd_expected}"

    usd_assets_actual = ["box.usda", "capsule.usda", "cylinder.usda", "sphere.usda"]
    for usd_asset in usd_assets_actual:
        usd_asset_expected = str(TEST_DIR / f"golden/primitives_separate_assets_{usd_asset}")
        usd_asset_actual = str(Path(tmp_path) / usd_asset)
        assert filecmp.cmp(
            usd_asset_expected, usd_asset_actual, shallow=False
        ), f"Actual USD: {usd_asset_actual}  Expected USD: {usd_asset_expected}"


def test_import_exported_usd_primitives(tmp_path):
    _set_random_seed()

    usd_path = str(Path(tmp_path) / "test_import_exported_primitives.usd")

    my_scene = examples.primitives_scene(seed=11)
    my_scene.export(usd_path)

    same_scene = synth.Asset(usd_path).scene(use_collision_geometry=False)

    my_scene_verts = my_scene.scene.dump(concatenate=True).vertices
    same_scene_verts = same_scene.scene.dump(concatenate=True).vertices

    # this could be replaced by the assert from the next test
    for x in my_scene_verts:
        if x not in same_scene_verts:
            assert False
    for x in same_scene_verts:
        if x not in my_scene_verts:
            assert False


def test_import_exported_usd_proc_kitchen(tmp_path):
    _set_random_seed()

    usd_path = str(Path(tmp_path) / "test_import_exported_proc_kitchen.usd")

    # SRL
    import scene_synthesizer.procedural_scenes as ps

    my_scene = ps.kitchen_single_wall()
    my_scene.export(usd_path)

    same_scene = synth.Asset(usd_path).scene(use_collision_geometry=False)

    my_scene_verts = my_scene.scene.dump(concatenate=True).vertices
    same_scene_verts = same_scene.scene.dump(concatenate=True).vertices

    my_scene_verts = np.array(sorted(my_scene_verts.flatten()))
    same_scene_verts = np.array(sorted(same_scene_verts.flatten()))

    assert np.allclose(my_scene_verts, same_scene_verts, atol=1e6)


def test_import_exported_srl_kitchen_cabinet(tmp_path):
    _set_random_seed()

    cabinet1_usd_path = str(TEST_DIR / "data/assets/srl_kitchen/srl-top-cabinet.usd")
    cabinet2_usd_path = str(Path(tmp_path) / "test_import_exported_srl_top_cabinet.usd")

    cabinet1 = synth.USDAsset(cabinet1_usd_path).scene(use_collision_geometry=False)
    cabinet1.export(cabinet2_usd_path)

    cabinet2 = synth.USDAsset(cabinet2_usd_path).scene(use_collision_geometry=False)

    assert cabinet1.get_joint_names() == cabinet2.get_joint_names()
    assert len(cabinet1.geometry) == len(cabinet2.geometry)
    assert np.allclose(
        cabinet1.scene.dump(concatenate=True).bounds, cabinet2.scene.dump(concatenate=True).bounds
    )

    cabinet1.update_configuration([1.3])
    cabinet2.update_configuration([1.3])
    assert np.allclose(
        cabinet1.scene.dump(concatenate=True).bounds, cabinet2.scene.dump(concatenate=True).bounds
    )


def test_write_usd_object_files_false(tmp_path):
    _set_random_seed()

    cabinet1_usd_path = str(Path(tmp_path) / "test_write_usd_object_files_False.usd")
    cabinet2_usd_path = str(Path(tmp_path) / "test_write_usd_object_files_True.usd")

    cabinet_usd = str(TEST_DIR / "data/assets/srl_kitchen/srl-top-cabinet.usd")
    cabinet_scene = synth.USDAsset(cabinet_usd, origin=("center", "center", "bottom")).scene(
        use_collision_geometry=False
    )

    scene_bounds = cabinet_scene.scene.dump(concatenate=True).bounds

    cabinet_scene.export(cabinet1_usd_path, write_usd_object_files=False)
    cabinet_scene.export(cabinet2_usd_path, write_usd_object_files=True)

    # load scenes again
    for fname in [cabinet1_usd_path, cabinet2_usd_path]:
        s = synth.USDAsset(fname).scene(use_collision_geometry=False)
        bounds = s.scene.dump(concatenate=True).bounds

        assert np.allclose(bounds, scene_bounds, atol=1e-5)


def test_write_usd_object_files_false_with_scale(tmp_path):
    _set_random_seed()

    cabinet1_usd_path = str(Path(tmp_path) / "test_write_usd_object_files_False.usd")
    cabinet2_usd_path = str(Path(tmp_path) / "test_write_usd_object_files_True.usd")

    cabinet_usd = str(TEST_DIR / "data/assets/srl_kitchen/srl-top-cabinet.usd")
    cabinet_scene = synth.USDAsset(
        cabinet_usd, extents=(3, 6, 9), origin=("center", "center", "bottom")
    ).scene(use_collision_geometry=False)

    scene_bounds = cabinet_scene.scene.dump(concatenate=True).bounds

    cabinet_scene.export(cabinet1_usd_path, write_usd_object_files=False)
    cabinet_scene.export(cabinet2_usd_path, write_usd_object_files=True)

    # load scenes again
    for fname in [cabinet1_usd_path, cabinet2_usd_path]:
        s = synth.USDAsset(fname).scene(use_collision_geometry=False)
        actual_bounds = s.scene.dump(concatenate=True).bounds
        
        assert np.allclose(actual_bounds, scene_bounds, atol=1e-5), (
            f"Bounds of {fname} ({actual_bounds}) do not comply with original scene bounds"
            f" {scene_bounds}."
        )


def test_write_usd_object_files_false_with_transform(tmp_path):
    _set_random_seed()

    cabinet1_usd_path = str(Path(tmp_path) / "test_write_usd_object_files_False.usd")
    cabinet2_usd_path = str(Path(tmp_path) / "test_write_usd_object_files_True.usd")

    cabinet_usd = str(TEST_DIR / "data/assets/srl_kitchen/srl-top-cabinet.usd")
    cabinet_asset = synth.USDAsset(cabinet_usd, origin=("center", "center", "top"))
    cabinet_scene = synth.Scene()
    cabinet_scene.add_object(
        cabinet_asset,
        "cabinet",
        transform=tra.translation_matrix([0.3, 0.6, 0.45]) @ tra.random_rotation_matrix(),
        use_collision_geometry=False,
    )

    scene_bounds = cabinet_scene.scene.dump(concatenate=True).bounds

    cabinet_scene.export(cabinet1_usd_path, write_usd_object_files=False)
    cabinet_scene.export(cabinet2_usd_path, write_usd_object_files=True)

    # load scenes again
    for fname in [cabinet1_usd_path, cabinet2_usd_path]:
        s = synth.USDAsset(fname).scene(use_collision_geometry=False)
        bounds = s.scene.dump(concatenate=True).bounds

        assert np.allclose(bounds, scene_bounds, atol=1e-5)


def test_link_offset(tmp_path):
    _set_random_seed()

    usd_expected = str(TEST_DIR / "golden/wall_cabinet.usda")
    usd_actual = str(Path(tmp_path) / "test_link_offset.usda")

    procedural_assets.WallCabinetAsset(
        width=0.5,
        depth=0.3,
        compartment_types=["door_right"],
        up=[0, 0, 1],
        front=[1, 0, 0],
    ).scene().export(usd_actual)

    assert filecmp.cmp(
        usd_expected, usd_actual, shallow=False
    ), f"Actual USD: {usd_actual}  Expected USD: {usd_expected}"


def test_missing_fixed_joint(tmp_path):
    _set_random_seed()

    usd_expected = str(TEST_DIR / "golden/test_missing_fixed_joint.usda")
    usd_actual = str(Path(tmp_path) / "test_missing_fixed_joint.usda")

    scene = synth.Scene()
    dataset = meshsets.load_dataset("PartNet Mobility")
    microwave_filename = dataset.get_filename(
        pattern="partnet_mobility_v0/dataset/7296/mobility.urdf"
    )
    microwave_asset = synth.Asset(
        fname=microwave_filename,
        origin=("bottom", "center", "bottom"),
    )
    microwave_id = "microwave"
    scene.add_object(
        obj_id=microwave_id,
        asset=microwave_asset,
        use_collision_geometry=None,
        joint_type=None,
    )

    floor = synth.BoxAsset(
        extents=[5, 5, 1e-2],
        origin=("center", "center", "top"),
    )
    floor_id = "floor"
    scene.add_object(
        obj_id=floor_id,
        asset=floor,
        use_collision_geometry=None,
    )
    scene.export(usd_actual)

    assert filecmp.cmp(
        usd_expected, usd_actual, shallow=False
    ), f"Actual USD: {usd_actual}  Expected USD: {usd_expected}"
