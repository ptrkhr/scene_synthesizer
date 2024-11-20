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
from scene_synthesizer import examples
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer import utils

TEST_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def microwave_ball_scene():
    # SRL
    from scene_synthesizer.examples import microwave_ball_table_scene

    _set_random_seed()
    return microwave_ball_table_scene(use_collision_geometry=True)


def _set_random_seed():
    random.seed(111)
    np.random.seed(111)


def test_mass():
    _set_random_seed()

    cabinet = pa.RecursivelyPartitionedCabinetAsset(
        width=1, depth=1, height=1, use_box_handle=True
    )

    s = synth.Scene()
    s.add_object(cabinet, "cabinet")

    desired_mass = 0.9
    s.set_mass("cabinet", desired_mass)
    current_mass = s.get_mass("cabinet")

    assert np.allclose(desired_mass, current_mass)


def test_mass_nonwatertight_geometry():
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    microwave_asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/7236/mobility.urdf")

    scene = synth.Asset(microwave_asset_path).scene()
    assert not scene.is_watertight("object")

    desired_mass = 2.5
    scene.set_mass("object", desired_mass)
    assert np.allclose(scene.get_mass("object"), desired_mass)


def test_density_nonwatertight_geometry():
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    microwave_asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/7236/mobility.urdf")

    scene = synth.Asset(microwave_asset_path).scene()
    assert not scene.is_watertight("object")

    desired_density = 0.76
    scene.set_density("object", desired_density)
    assert np.allclose(scene.get_density("object"), desired_density)


def test_volume_density_mass_consistency():
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    microwave_asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/7236/mobility.urdf")

    scene = synth.Asset(microwave_asset_path).scene()
    assert not scene.is_watertight("object")

    assert np.allclose(
        scene.get_density("object") * scene.get_volume("object"), scene.get_mass("object")
    )

    scene.set_density("object", 0.45)
    assert np.allclose(
        scene.get_density("object") * scene.get_volume("object"), scene.get_mass("object")
    )

    scene.set_mass("object", 3.82)
    assert np.allclose(
        scene.get_density("object") * scene.get_volume("object"), scene.get_mass("object")
    )


def test_density():
    _set_random_seed()

    cabinet = pa.RecursivelyPartitionedCabinetAsset(
        width=1, depth=1, height=1, use_box_handle=True
    )

    s = synth.Scene()
    s.add_object(cabinet, "cabinet")

    desired_density = 0.3
    s.set_density("cabinet", desired_density)
    current_density = s.get_density("cabinet")

    assert np.allclose(desired_density, current_density)


def test_mass_density_asset():
    _set_random_seed()

    desired_mass = 20.0
    s = pa.RefrigeratorAsset(1, 1, 1, mass=desired_mass).scene()

    assert np.allclose(s.get_mass("object"), desired_mass)

    desired_density = 200.0
    s = pa.RefrigeratorAsset(1, 1, 1, density=desired_density).scene()

    assert np.allclose(s.get_density("object"), desired_density)

    # expect an exception
    with pytest.raises(Exception):
        pa.RefrigeratorAsset(1, 1, 1, mass=desired_mass, density=desired_density).scene()


def test_center_mass_asset():
    _set_random_seed()

    s = pa.RefrigeratorAsset(1, 1, 1, num_shelves=1, num_door_shelves=0).scene()
    expected_com = [-0.00067258,  0.01945447,  0.49236119]
    assert np.allclose(s.get_center_mass(["object"]), expected_com)

    desired_center_mass = [1.0, 1.0, 2.0]
    fridge = pa.RefrigeratorAsset(1, 1, 1, num_shelves=1, num_door_shelves=0, center_mass=desired_center_mass)

    scene = fridge.scene()

    assert np.allclose(scene.get_center_mass(["object"]), desired_center_mass)


def test_geometry_vs_object_node_names():
    _set_random_seed()

    tmp_urdf_path = "/tmp/obj.urdf"

    scene1 = synth.Scene()
    scene1.add_object(
        obj_id="obj",
        asset=synth.BoxAsset(extents=np.ones(3)),
        joint_type="fixed",
        use_collision_geometry=None,
    )
    scene1.export(tmp_urdf_path)

    scene2 = synth.Scene()
    scene2.add_object(
        obj_id="obj",
        asset=synth.Asset(tmp_urdf_path),
        joint_type="fixed",
        use_collision_geometry=None,
    )
    _ = scene2.label_support(label="support", layer="visual")

    scene1_obj_names = scene1.metadata["object_nodes"].keys()
    scene2_geom_names = scene2.metadata["object_geometry_nodes"].keys()

    assert scene1_obj_names == scene2_geom_names


def test_layers():
    _set_random_seed()

    s = examples.microwave_ball_table_scene(use_collision_geometry=True, seed=111)
    assert s.get_layer_names() == {None, "collision"}

    s = examples.microwave_ball_table_scene(use_collision_geometry=False, seed=111)
    assert s.get_layer_names() == {None, "visual"}

    s = examples.microwave_ball_table_scene(use_collision_geometry=None, seed=111)
    assert s.get_layer_names() == {None, "visual", "collision"}

    s.remove_layer("collision")
    assert s.get_layer_names() == {None, "visual"}

    s.set_layer_name("visual")
    assert s.get_layer_names() == {"visual"}


def test_geometry_approximation():
    _set_random_seed()

    mug = synth.Asset(
        str(TEST_DIR / "data/assets/shapenetsem/Mug/10f6e09036350e92b3f21f1137c3c347.obj"),
        origin=("com", "com", "bottom"),
        height=0.27,
    )
    s = mug.scene(use_collision_geometry=False)
    s.set_layer_name("visual")
    assert s.get_layer_names() == {"visual"}

    s.add_convex_decomposition(input_layer="visual", output_layer="collision")
    assert s.get_layer_names() == {"visual", "collision"}

    s.add_bounding_boxes(input_layer="collision", output_layer="bboxes_aligned")
    assert s.get_layer_names() == {"visual", "collision", "bboxes_aligned"}

    s.add_bounding_boxes_oriented(input_layer="collision", output_layer="obb")
    assert s.get_layer_names() == {
        "visual",
        "collision",
        "bboxes_aligned",
        "obb",
    }

    s.add_bounding_cylinders(input_layer="collision", output_layer="bcyl")
    assert s.get_layer_names() == {
        "visual",
        "collision",
        "bboxes_aligned",
        "obb",
        "bcyl",
    }

    s.add_bounding_spheres(input_layer="collision", output_layer="bsphere")
    assert s.get_layer_names() == {
        "visual",
        "collision",
        "bboxes_aligned",
        "obb",
        "bcyl",
        "bsphere",
    }

    s.add_bounding_primitives(input_layer="collision", output_layer="bprim")
    assert s.get_layer_names() == {
        "visual",
        "collision",
        "bboxes_aligned",
        "obb",
        "bcyl",
        "bsphere",
        "bprim",
    }

    s.add_voxel_decomposition(input_layer="collision", output_layer="voxels")
    assert s.get_layer_names() == {
        "visual",
        "collision",
        "bboxes_aligned",
        "obb",
        "bcyl",
        "bsphere",
        "bprim",
        "voxels",
    }


def test_surface_coverage():
    _set_random_seed()

    scene = synth.Scene()
    scene.add_object(
        obj_id="obj1",
        asset=synth.BoxAsset(extents=np.ones(3)),
    )
    assert (
        len(
            scene.label_support(
                "obj1_surface", obj_ids="obj1", surface_test=scene.create_surface_coverage_test()
            )
        )
        == 1
    )

    scene.add_object(
        obj_id="obj2",
        asset=synth.BoxAsset(extents=np.ones(3)),
        transform=tra.translation_matrix([0, 0, 1 + 1e-2]),
    )
    assert (
        len(
            scene.label_support(
                "obj1_surface", obj_ids="obj1", surface_test=scene.create_surface_coverage_test()
            )
        )
        == 0
    )


def test_subscene():
    scene = synth.Scene()
    microwave_asset = synth.Asset(
        fname=str(TEST_DIR / "data/assets/partnet_mobility_v0/7236/mobility.urdf"),
        height=1,
        origin=("bottom", "center", "bottom"),
    )
    microwave_id = "microwave"
    scene.add_object(
        obj_id=microwave_id,
        asset=microwave_asset,
        joint_type="fixed",
        use_collision_geometry=None,
    )

    microwave_subscene = scene.subscene(obj_ids=[microwave_id])

    # This will only work if some metadata has been copied
    microwave_subscene.update_configuration(
        obj_id="microwave", joint_ids=["joint_0"], configuration=[1]
    )

    # This tests copying metadata
    assert microwave_subscene.get_joint_names() == scene.get_joint_names()

    # Test for getting geometry nodes
    button_nodes = [n for n in scene.graph.nodes if n.startswith('microwave/control_button')]
    button_subscene = scene.subscene(button_nodes)

    assert len(button_subscene.geometry) == 32
    assert np.allclose(button_subscene.get_extents(), [3.88578059e-16, 2.25390530e-01, 5.22984645e-01])


def test_subscene_bounds():
    s = synth.Scene()
    s.add_object(
        synth.BoxAsset(extents=[1, 1, 1]),
        "box1", 
        transform=tra.translation_matrix([0, 0, 1])
    )
    s.add_object(
        synth.BoxAsset(extents=[1, 1, 1]),
        "box2",
        transform=tra.translation_matrix([0, 0, 2])
    )
    s.add_object(
        synth.BoxAsset(extents=[1, 1, 1]),
        "box3",
        transform=tra.translation_matrix([0, 0, 2]),
        parent_id="box2",
    )

    np.allclose(s.subscene(["box1"]).scene.bounds, np.array([[-0.5, -0.5, 0.5], [0.5, 0.5, 1.5]]))
    np.allclose(
        s.subscene(["box1", "box3"]).scene.bounds, np.array([[-0.5, -0.5, 0.5], [0.5, 0.5, 4.5]])
    )
    np.allclose(
        s.subscene(["box2", "box3"]).scene.bounds, np.array([[-0.5, -0.5, 1.5], [0.5, 0.5, 4.5]])
    )
    np.allclose(
        s.subscene(["box2", "box3"], "box3").scene.bounds,
        np.array([[-0.5, -0.5, -2.5], [0.5, 0.5, 0.5]]),
    )


def test_stackbox():
    s = synth.Scene()

    s.add_object(synth.BoxAsset(extents=[1, 2, 3]), "box_foo")
    np.allclose(
        s.scene.bounds,
        np.array([[-0.5, -1.0, -1.5], [0.5, 1.0, 1.5]]),
    )

    second_box_id = s.stack_box(None, 0.5, direction="-y", stack_parent_obj_ids=["box_foo"])
    np.allclose(
        s.scene.bounds,
        np.array([[-0.5, -1.5, -1.5], [0.5, 1.0, 1.5]]),
    )

    s.stack_box(None, 0.33, direction="z", stack_parent_obj_ids=["box_foo"])
    np.allclose(
        s.scene.bounds,
        np.array([[-0.5, -1.5, -1.5], [0.5, 1.0, 1.83]]),
    )

    s.stack_box(None, 0.4, direction="x", stack_parent_obj_ids=[second_box_id])
    np.allclose(
        s.scene.bounds,
        np.array([[-0.5, -1.5, -1.5], [0.9, 1.0, 1.83]]),
    )

    s.stack_box(None, 0.5, offset=0.2, direction="x", stack_parent_obj_ids=["box_foo"])
    np.allclose(
        s.scene.bounds,
        np.array([[-0.5, -1.5, -1.5], [1.2, 1.0, 1.83]]),
    )


def test_asset_function():
    _set_random_seed()

    scene = synth.Scene()
    cabinet = pa.RecursivelyPartitionedCabinetAsset(
        width=1, depth=1, height=1, use_box_handle=True
    )
    box = synth.BoxAsset(extents=[0.05, 0.05, 0.05], origin=("com", "com", "bottom"))
    scene.add_object(cabinet, "cabinet")
    scene.label_support("support")
    scene.place_object("box", box, "support")

    box_asset = scene.asset("box")

    assert np.allclose(
        box.as_trimesh_scene().dump(True).vertices, box_asset.as_trimesh_scene().dump(True).vertices
    )
    assert np.allclose(
        box.as_trimesh_scene().dump(True).faces, box_asset.as_trimesh_scene().dump(True).faces
    )


def test_rename_geometries():
    _set_random_seed()

    scene = synth.Scene()
    cabinet = pa.RecursivelyPartitionedCabinetAsset(
        width=1, depth=1, height=1, use_box_handle=True
    )
    box = synth.BoxAsset(extents=[0.05, 0.05, 0.05], origin=("com", "com", "bottom"))
    scene.add_object(cabinet, "cabinet")
    scene.label_support("support")
    scene.place_object("box", box, "support")

    assert "box/geometry_0" in scene.scene.geometry.keys()

    scene.rename_geometries([("box/geometry_0", "another_box/geometry_0")])

    assert "box/geometry_0" not in scene.scene.geometry.keys()
    assert "another_box/geometry_0" in scene.scene.geometry.keys()


def test_move_object():
    _set_random_seed()

    scene = synth.Scene()
    cabinet = pa.RecursivelyPartitionedCabinetAsset(
        width=1, depth=1, height=1, use_box_handle=True
    )
    box = synth.BoxAsset(extents=[0.05, 0.05, 0.05], origin=("com", "com", "bottom"))
    scene.add_object(cabinet, "cabinet")
    scene.label_support("support")
    assert scene.place_object("box", box, "support")

    nodes_before = set(scene.graph.nodes)

    scene.move_object("box", "support")

    assert nodes_before == set(scene.graph.nodes)


def test_get_object_names_and_transforms(microwave_ball_scene):
    _set_random_seed()

    expected_obj_names = sorted(["table", "microwaveoven", "ball"])

    assert sorted(microwave_ball_scene.get_object_names()) == expected_obj_names

    obj_transforms = microwave_ball_scene.get_object_transforms()

    assert len(obj_transforms) == len(expected_obj_names)

    assert np.allclose(obj_transforms["microwaveoven"][:3, 3], [0.0026825, 0.01970401, 0.92086477])


def test_get_collapse_and_simplify(microwave_ball_scene, tmp_path):
    _set_random_seed()

    microwave_ball_scene.remove_joints(
        microwave_ball_scene.get_joint_names(joint_type_query="floating")
    )
    microwave_ball_scene.collapse_nodes()
    microwave_ball_scene.simplify_node_names()

    microwave_ball_scene.export(os.path.join(tmp_path, "microwave_ball_scene.usd"))

    assert True


def test_placing_objects():
    _set_random_seed()

    # create shelf asset
    scene = pa.ShelfAsset(1, 0.5, 1.5, 4).scene(None)
    scene.label_support("support")

    box = synth.BoxAsset(extents=[0.1, 0.1, 0.2], origin=("center", "center", "bottom"))
    num_boxes = 10

    scene.place_objects(
        obj_id_iterator=utils.object_id_generator("box"),
        obj_asset_iterator=itertools.repeat(box, num_boxes),
        obj_support_id_iterator=scene.support_generator("support"),
        obj_position_iterator=utils.PositionIteratorUniform(),
        obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
        distance_above_support=0.002,
        use_collision_geometry=True,
    )

    # check that all boxes were placed
    expected_obj_names = set(itertools.islice(utils.object_id_generator("box"), num_boxes))
    assert expected_obj_names.intersection(set(scene.get_object_names())) == expected_obj_names

    # check the distance between boxes and surfaces
    for box_name in expected_obj_names:
        box_verts = scene.subscene([box_name]).scene.dump()[0].vertices

        min_dist = 1e6
        for surface in scene.metadata["support_polygons"]["support"]:
            surface_T = scene.get_transform(surface.node_name) @ surface.transform
            m = trimesh.creation.extrude_polygon(surface.polygon, height=0.001)
            m.apply_transform(surface_T)

            _, dists, _ = m.nearest.on_surface(box_verts)
            min_dist = min(np.min(dists), min_dist)

        assert min_dist < 0.005


def test_stable_object_placement():
    _set_random_seed()

    # create shelf asset
    scene = pa.ShelfAsset(1, 0.5, 1.5, 4).scene(None)
    scene.label_support("support")

    box = synth.BoxAsset(extents=[0.1, 0.1, 0.2], origin=("center", "center", "bottom"))
    num_boxes = 10

    scene.place_objects(
        obj_id_iterator=utils.object_id_generator("box"),
        obj_asset_iterator=itertools.repeat(box, num_boxes),
        obj_support_id_iterator=scene.support_generator("support"),
        obj_position_iterator=utils.PositionIteratorUniform(seed=0),
        obj_orientation_iterator=synth.utils.orientation_generator_stable_poses(
            box, seed=0, convexify=True
        ),
        distance_above_support=0.002,
        use_collision_geometry=True,
    )


# Make sure you have `pip install pytest-timeout` to run this test
@pytest.mark.timeout(5)
def test_stable_pose_calculation(tmp_path):
    _set_random_seed()

    # try flat asset
    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    obj_asset_fname = os.path.join(asset_root_dir, "shapenetsem/painting.obj")

    asset_1 = synth.Asset(obj_asset_fname)
    asset_1.compute_stable_poses(convexify=False)

    # try thin cylinder
    obj_asset_fname = os.path.join(tmp_path, "thin_cylinder.obj")
    trimesh.creation.cylinder(radius=1e-4, height=0.2).export(obj_asset_fname)

    asset_2 = synth.Asset(obj_asset_fname)
    asset_2.compute_stable_poses(tolerance_zero_extent=1e-3, convexify=True)

    # try thin box
    obj_asset_fname = os.path.join(tmp_path, "thin_box.obj")
    trimesh.creation.box([0.2, 0.2, 1e-4]).export(obj_asset_fname)

    asset_3 = synth.Asset(obj_asset_fname)
    asset_3.compute_stable_poses(tolerance_zero_extent=1e-3, convexify=False)

    assert True


def test_collision_manager():
    _set_random_seed()

    # create a scene
    s = synth.BoxAsset(extents=[0.1, 0.1, 0.2], origin=("center", "center", "bottom")).scene("box1")

    # test collision
    box = synth.BoxAsset(extents=[0.1, 0.1, 0.1])
    box_mesh = box.as_trimesh_scene().dump(concatenate=True)
    assert s.in_collision_single(
        mesh=box_mesh, transform=tra.translation_matrix([0, 0, 0.25 - 1e-6])
    )
    assert not s.in_collision_single(
        mesh=box_mesh, transform=tra.translation_matrix([0, 0, 0.25 + 1e-6])
    )
    assert s.in_collision_single(
        mesh=box_mesh,
        transform=tra.translation_matrix([0, 0, 0.25 + 1e-6]),
        min_distance=1e-5,
        epsilon=0,
    )
    assert not s.in_collision_single(
        mesh=box_mesh,
        transform=tra.translation_matrix([0, 0, 0.25 + 1e-6]),
        min_distance=1e-5,
        epsilon=1e-4,
    )

    s.add_object(box, "box2", transform=tra.translation_matrix([0, 0, 0.25]))
    assert s.in_collision_single(
        mesh=box_mesh,
        transform=tra.translation_matrix([0, 0, 0.35 + 1e-6]),
        min_distance=1e-5,
        epsilon=0,
    )
    assert not s.in_collision_single(
        mesh=box_mesh,
        transform=tra.translation_matrix([0, 0, 0.35 + 1e-6]),
        min_distance=1e-5,
        epsilon=1e-4,
    )

    s.remove_object("box2")
    assert s.in_collision_single(
        mesh=box_mesh,
        transform=tra.translation_matrix([0, 0, 0.25 + 1e-6]),
        min_distance=1e-5,
        epsilon=0,
    )
    assert not s.in_collision_single(
        mesh=box_mesh,
        transform=tra.translation_matrix([0, 0, 0.25 + 1e-6]),
        min_distance=1e-5,
        epsilon=1e-4,
    )


def test_collision_manager_with_scene():
    for seed in range(10):
        s = examples.table_chair_scene(use_shapenetsem=True, seed=seed)

        coll_mgrs = {}
        for obj_name in s.get_object_names():
            coll_mgrs[obj_name] = trimesh.collision.scene_to_collision(
                s.subscene([obj_name]).scene
            )[0]

        for e1, e2 in itertools.combinations(coll_mgrs.keys(), 2):
            assert not coll_mgrs[e1].in_collision_other(coll_mgrs[e2])


def test_bounds_extents_centroid():
    _set_random_seed()

    s = synth.Scene()
    box_extents = np.array([0.1, 1.0, 0.3])
    s.add_object(
        synth.BoxAsset(box_extents),
        "box",
        transform=tra.euler_matrix(2.0, 0, 0) @ tra.translation_matrix([0.1, 0.2, 0.3]),
    )

    expected_bounds_world = np.array(
        [[0.05, -0.70048663, -0.33521125], [0.15, -0.01155056, 0.44924212]]
    )
    assert np.allclose(s.get_bounds(["box"]), expected_bounds_world)

    expected_bounds_box = np.tile(box_extents / 2.0, (2, 1))
    expected_bounds_box[0] = -expected_bounds_box[0]
    assert np.allclose(s.get_bounds(["box"], "box"), expected_bounds_box)

    expected_extents_world = np.array([0.1, 0.68893606, 0.78445338])
    assert np.allclose(s.get_extents(["box"]), expected_extents_world)

    expected_extents_box = box_extents
    assert np.allclose(s.get_extents(["box"], "box"), expected_extents_box)

    expected_centroid_world = np.array([0.1, -0.3560186, 0.05701543])
    assert np.allclose(s.get_centroid(["box"]), expected_centroid_world)

    expected_centroid_box = np.zeros(3)
    assert np.allclose(s.get_centroid(["box"], "box"), expected_centroid_box)

    assert np.allclose(s.get_center_mass(["box"]), expected_centroid_world)
    assert np.allclose(s.get_center_mass(["box"], "box"), expected_centroid_box)


def test_origin_com():
    _set_random_seed()
    expected_bounds_table = [
        [-38.01021672, -37.2020709, -34.74346756],
        [37.92573983, 37.14273203, 5.11598709],
    ]
    expected_bounds_mug = [
        [-138.50990507, -215.68213375, -186.7621646],
        [138.49009493, 154.69686625, 182.9848354],
    ]
    expected_bounds_coffeemachine = [
        [-0.8185613, -0.42941295, -0.67518714],
        [0.6639957, 0.74859905, 0.59890586],
    ]

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/shapenetsem/DiningTable/7c10130e50d27692435807c7a815b457.obj",
            origin=("com", "com", "com"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_table,
    )

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/shapenetsem/Mug/10f6e09036350e92b3f21f1137c3c347.obj",
            origin=("com", "com", "com"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_mug,
    )

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/partnet_mobility_v0/103118/mobility.urdf",
            origin=("com", "com", "com"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_coffeemachine,
    )


def test_origin_centroid():
    _set_random_seed()
    expected_bounds_table = [
        [[-37.96797828, -37.17240147, -19.92972733], [37.96797828, 37.17240147, 19.92972733]]
    ]
    expected_bounds_mug = [[[-138.5, -185.1895, -184.8735], [138.5, 185.1895, 184.8735]]]
    expected_bounds_coffeemachine = [
        [[-0.7412785, -0.589006, -0.6370465], [0.7412785, 0.589006, 0.6370465]]
    ]

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/shapenetsem/DiningTable/7c10130e50d27692435807c7a815b457.obj",
            origin=("centroid", "centroid", "centroid"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_table,
    )

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/shapenetsem/Mug/10f6e09036350e92b3f21f1137c3c347.obj",
            origin=("centroid", "centroid", "centroid"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_mug,
    )

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/partnet_mobility_v0/103118/mobility.urdf",
            origin=("centroid", "centroid", "centroid"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_coffeemachine,
    )


def test_origin_center():
    _set_random_seed()
    expected_bounds_table = [
        [[-37.96797828, -37.17240147, -19.92972733], [37.96797828, 37.17240147, 19.92972733]]
    ]
    expected_bounds_mug = [[[-138.5, -185.1895, -184.8735], [138.5, 185.1895, 184.8735]]]
    expected_bounds_coffeemachine = [
        [[-0.7412785, -0.589006, -0.6370465], [0.7412785, 0.589006, 0.6370465]]
    ]

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/shapenetsem/DiningTable/7c10130e50d27692435807c7a815b457.obj",
            origin=("center", "center", "center"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_table,
    )

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/shapenetsem/Mug/10f6e09036350e92b3f21f1137c3c347.obj",
            origin=("center", "center", "center"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_mug,
    )

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/partnet_mobility_v0/103118/mobility.urdf",
            origin=("center", "center", "center"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_coffeemachine,
    )


def test_origin_top():
    _set_random_seed()
    expected_bounds_table = [[[-75.93595655, -74.34480294, -39.85945466], [0.0, 0.0, 0.0]]]
    expected_bounds_mug = [[[-277.0, -370.379, -369.747], [0.0, 0.0, 0.0]]]
    expected_bounds_coffeemachine = [[[-1.482557, -1.178012, -1.274093], [0.0, 0.0, 0.0]]]

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/shapenetsem/DiningTable/7c10130e50d27692435807c7a815b457.obj",
            origin=("top", "top", "top"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_table,
    )

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/shapenetsem/Mug/10f6e09036350e92b3f21f1137c3c347.obj",
            origin=("top", "top", "top"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_mug,
    )

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/partnet_mobility_v0/103118/mobility.urdf",
            origin=("top", "top", "top"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_coffeemachine,
    )


def test_origin_bottom():
    _set_random_seed()
    expected_bounds_table = [[[0.0, 0.0, 0.0], [75.93595655, 74.34480294, 39.85945466]]]
    expected_bounds_mug = [[[0.0, 0.0, 0.0], [277.0, 370.379, 369.747]]]
    expected_bounds_coffeemachine = [[[0.0, 0.0, 0.0], [1.482557, 1.178012, 1.274093]]]

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/shapenetsem/DiningTable/7c10130e50d27692435807c7a815b457.obj",
            origin=("bottom", "bottom", "bottom"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_table,
    )

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/shapenetsem/Mug/10f6e09036350e92b3f21f1137c3c347.obj",
            origin=("bottom", "bottom", "bottom"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_mug,
    )

    assert np.allclose(
        synth.Asset(
            "tests/data/assets/partnet_mobility_v0/103118/mobility.urdf",
            origin=("bottom", "bottom", "bottom"),
        )
        .scene()
        .scene.bounds,
        expected_bounds_coffeemachine,
    )


def test_adding_object():
    _set_random_seed()

    s = pa.CabinetAsset(
        0.6,
        0.5,
        2.0,
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
        ["door_right", "drawer", "open", "open", "closed"],
        compartment_heights=[0.3, 0.2, 0.4, 0.4, 0.5],
    ).scene("cabinet")

    micro = pa.MicrowaveAsset(
        width=0.58,
        depth=0.45,
        height=0.43,
        thickness=0.05,
        display_panel_width=0.1,
    )
    s.add_object(
        obj_id="microwave",
        asset=micro,
        connect_obj_anchor=("center", "bottom", "bottom"),
        connect_parent_anchor=("center", "bottom", "top"),
        connect_parent_id=["cabinet/surface_0_3", "cabinet/surface_1_3"],
        use_collision_geometry=False,
    )

    # check for a thing


def test_scaling_asset():
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    microwave_asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/7236/mobility.urdf")

    desired_width, desired_depth, desired_height = (0.5, 0.6, 0.1)

    scene = synth.Asset(microwave_asset_path, width=desired_width, depth=desired_depth).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[0, 1]], [desired_width, desired_depth])

    scene = synth.Asset(microwave_asset_path, width=desired_width, height=desired_height).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[0, 2]], [desired_width, desired_height])

    scene = synth.Asset(microwave_asset_path, height=desired_height, depth=desired_depth).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[1, 2]], [desired_depth, desired_height])

    scene = synth.Asset(
        microwave_asset_path, width=desired_width, depth=desired_depth, height=desired_height
    ).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[0, 1, 2]], [desired_width, desired_depth, desired_height])

    scene = synth.Asset(microwave_asset_path, width=desired_width).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[0]], [desired_width])

    scene = synth.Asset(
        microwave_asset_path, width=desired_width, depth=desired_depth, height=desired_height
    ).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[1]], [desired_depth])

    scene = synth.Asset(
        microwave_asset_path, width=desired_width, depth=desired_depth, height=desired_height
    ).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[2]], [desired_height])

    scene = synth.Asset(microwave_asset_path, max_length=0.3).scene()
    extents = scene.get_extents()
    assert np.all(extents <= 0.3)
    assert np.any(np.isclose(extents, 0.3))

    scene = synth.Asset(microwave_asset_path, min_length=0.3).scene()
    extents = scene.get_extents()
    assert np.all(extents >= 0.3)
    assert np.any(np.isclose(extents, 0.3))


def test_rotation_scaling_asset():
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    microwave_asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/7236/mobility.urdf")

    desired_width, desired_depth, desired_height = (0.5, 0.6, 0.1)

    scene = synth.Asset(
        microwave_asset_path,
        front=(1, 0, 0),
        up=(0, 0, 1),
        width=desired_width,
        depth=desired_depth,
    ).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[0, 1]], [desired_width, desired_depth]), f"actual: {extents}"

    scene = synth.Asset(
        microwave_asset_path,
        front=(1, 0, 0),
        up=(0, 0, 1),
        width=desired_width,
        height=desired_height,
    ).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[0, 2]], [desired_width, desired_height]), f"actual: {extents}"

    scene = synth.Asset(
        microwave_asset_path,
        front=(1, 0, 0),
        up=(0, 0, 1),
        height=desired_height,
        depth=desired_depth,
    ).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[1, 2]], [desired_depth, desired_height]), f"actual: {extents}"

    scene = synth.Asset(
        microwave_asset_path,
        front=(1, 0, 0),
        up=(0, 0, 1),
        width=desired_width,
        depth=desired_depth,
        height=desired_height,
    ).scene()
    extents = scene.get_extents()
    assert np.allclose(
        extents[[0, 1, 2]], [desired_width, desired_depth, desired_height]
    ), f"actual: {extents}"

    scene = synth.Asset(
        microwave_asset_path, front=(1, 0, 0), up=(0, 0, 1), width=desired_width
    ).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[0]], [desired_width]), f"actual: {extents}"

    scene = synth.Asset(
        microwave_asset_path,
        front=(1, 0, 0),
        up=(0, 0, 1),
        width=desired_width,
        depth=desired_depth,
        height=desired_height,
    ).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[1]], [desired_depth]), f"actual: {extents}"

    scene = synth.Asset(
        microwave_asset_path,
        front=(1, 0, 0),
        up=(0, 0, 1),
        width=desired_width,
        depth=desired_depth,
        height=desired_height,
    ).scene()
    extents = scene.get_extents()
    assert np.allclose(extents[[2]], [desired_height]), f"actual: {extents}"

    scene = synth.Asset(microwave_asset_path, front=(1, 0, 0), up=(0, 0, 1), max_length=0.3).scene()
    extents = scene.get_extents()
    assert np.all(extents <= 0.3)
    assert np.any(np.isclose(extents, 0.3))

    scene = synth.Asset(microwave_asset_path, front=(1, 0, 0), up=(0, 0, 1), min_length=0.3).scene()
    extents = scene.get_extents()
    assert np.all(extents >= 0.3)
    assert np.any(np.isclose(extents, 0.3))


def test_support_surface_generator():
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    cabinet_asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/48855/mobility.urdf")

    cabinet = synth.Asset(
        cabinet_asset_path,
        max_length=1.2,
        origin=("center", "center", "center"),
    )
    s = synth.Scene(seed=111)
    s.add_object(
        obj_id=f"cabinet",
        asset=cabinet,
        use_collision_geometry=True,
    )
    s.label_support(
        "support",
        min_area=0.003,
        gravity=np.array([0, 0, -1]),
    )

    result = list(itertools.islice(s.support_generator(), 3))

    expected_facet_indices = [0, 0, 2]

    res_indices = [x.facet_index for x in result]

    assert len(result) == len(expected_facet_indices)
    assert np.allclose(expected_facet_indices, res_indices)

    result = list(itertools.islice(s.support_generator(sampling_fn=random.choice), 3))

    res_indices = [x.facet_index for x in result]
    expected_facet_indices = [2, 0, 0]

    assert len(result) == len(expected_facet_indices)
    assert np.allclose(expected_facet_indices, res_indices)


def test_container_generator():
    _set_random_seed()

    asset_root_dir = os.path.join(TEST_DIR, "data", "assets")
    cabinet_asset_path = os.path.join(asset_root_dir, "partnet_mobility_v0/48855/mobility.urdf")

    cabinet = synth.Asset(
        cabinet_asset_path,
        max_length=1.2,
        origin=("center", "center", "center"),
    )
    my_scene = synth.Scene(seed=111)
    my_scene.add_object(
        obj_id=f"cabinet",
        asset=cabinet,
        use_collision_geometry=True,
    )
    my_scene.label_containment(
        "container",
        min_area=0.003,
        gravity=np.array([0, 0, -1]),
    )

    result = list(itertools.islice(my_scene.container_generator(), 3))
    res_indices = [x.support_surface.facet_index for x in result]

    expected_facet_indices = [2, 0, 0]

    assert len(result) == len(expected_facet_indices)
    assert np.allclose(expected_facet_indices, res_indices)

    result = list(itertools.islice(my_scene.container_generator(sampling_fn=random.choice), 3))
    res_indices = [x.support_surface.facet_index for x in result]

    expected_facet_indices = [0, 0, 2]

    assert len(result) == len(expected_facet_indices)
    assert np.allclose(expected_facet_indices, res_indices)


def test_support_as_layer():
    _set_random_seed()

    s = examples.support_surfaces(
        asset_fnames=[
            os.path.join(TEST_DIR, "data", "assets", "shapenetsem_watertight", "1Shelves", x)
            for x in [
                "160684937ae737ec5057ad0f363d6ddd.obj",
                "1e3df0ab57e8ca8587f357007f9e75d1.obj",
                "2b9d60c74bc0d18ad8eae9bce48bbeed.obj",
                "a9c2bcc286b68ee217a3b9ca1765e2a4.obj",
            ]
        ],
        seed=111,
    )

    expected_num_supports = 6
    support_length = len(s.metadata["support_polygons"]["support"])

    assert support_length == expected_num_supports

    expected_histogram = (
        np.array([72, 278, 0, 0, 0, 4, 68, 82, 52, 110]),
        np.array(
            [
                -0.20790711,
                0.06993961,
                0.34778633,
                0.62563306,
                0.90347978,
                1.1813265,
                1.45917323,
                1.73701995,
                2.01486667,
                2.29271339,
                2.57056012,
            ]
        ),
    )

    hist = np.histogram(s.support_scene().dump(concatenate=True).vertices)

    for a, b in zip(expected_histogram, hist):
        assert np.allclose(a, b)

    s.add_supports_as_layer(layer="my_support")

    assert s.get_layer_names() == {None, "my_support"}

    expected_geometry_names = [f"support/support{i}" for i in range(expected_num_supports)]
    geometry_names = list(s.geometry.keys())

    for n in expected_geometry_names:
        assert n in geometry_names


def test_container_as_layer():
    _set_random_seed()

    s = examples.container_volumes(
        asset_fnames=[
            os.path.join(TEST_DIR, "data", "assets", "shapenetsem_watertight", "1Shelves", x)
            for x in [
                "160684937ae737ec5057ad0f363d6ddd.obj",
                "1e3df0ab57e8ca8587f357007f9e75d1.obj",
                "2b9d60c74bc0d18ad8eae9bce48bbeed.obj",
                "a9c2bcc286b68ee217a3b9ca1765e2a4.obj",
            ]
        ],
        seed=111,
    )

    expected_num_containers = 1
    num_containers = len(s.metadata["containers"]["container"])

    assert num_containers == expected_num_containers

    expected_histogram = (
        np.array([8, 103, 0, 0, 0, 2, 44, 61, 0, 4]),
        np.array(
            [
                -0.20790711,
                0.06966041,
                0.34722794,
                0.62479546,
                0.90236299,
                1.17993051,
                1.45749803,
                1.73506556,
                2.01263308,
                2.29020061,
                2.56776813,
            ]
        ),
    )

    hist = np.histogram(s.container_scene().dump(concatenate=True).vertices)

    for a, b in zip(expected_histogram, hist):
        assert np.allclose(a, b, atol=1e-5)

    s.add_containers_as_layer(layer="my_containers")

    assert s.get_layer_names() == {None, "my_containers"}

    expected_geometry_names = [f"container/container{i}" for i in range(expected_num_containers)]
    geometry_names = list(s.geometry.keys())

    for n in expected_geometry_names:
        assert n in geometry_names


def test_part_as_layer():
    _set_random_seed()

    s = examples.microwave_ball_table_scene(use_collision_geometry=True, seed=111)

    s.label_part("part", "geometry.*")

    expected_num_parts = 2

    assert len(s.metadata["parts"]["part"]) == expected_num_parts

    expected_histogram = (
        np.array([262,  771,  357,  266, 1028,  267,  252,  254,  595, 1408]),
        np.array(
            [
                -0.68583338,
                -0.53517706,
                -0.38452075,
                -0.23386444,
                -0.08320813,
                0.06744818,
                0.21810449,
                0.3687608,
                0.51941711,
                0.67007342,
                0.82072974,
            ]
        ),
    )

    hist = np.histogram(s.part_scene().dump(concatenate=True).vertices)

    for a, b in zip(expected_histogram, hist):
        assert np.allclose(a, b, atol=1e-4)

    s.add_parts_as_layer(layer="my_parts")

    assert s.get_layer_names() == {None, "collision", "my_parts"}

    expected_geometry_names = [f"part/part{i}" for i in range(expected_num_parts)]
    geometry_names = list(s.geometry.keys())

    for n in expected_geometry_names:
        assert n in geometry_names


def test_adding_object_transform():
    _set_random_seed()

    s = examples.microwave_ball_table_scene(use_collision_geometry=True, seed=111)

    asset_ball = synth.TrimeshAsset(
        mesh=trimesh.primitives.Sphere(radius=0.04), origin=("center", "center", "bottom")
    )

    T_expected = [0.67689402, -0.03473922, 0.72073466]

    for parent_frame in ["table", "table/geometry_0", None, "world"]:
        s.remove_object("ball")
        s.place_object(
            "ball",
            asset_ball,
            support_id="surface",
            use_collision_geometry=True,
            joint_type="floating",
            parent_id=parent_frame,
            obj_position_iterator=utils.PositionIteratorList([np.zeros(2)]),
        )

        T = s.get_transform("ball")
        assert np.allclose(T[:3, 3], T_expected)


def test_randomization():
    s1 = examples.table_chair_scene(use_shapenetsem=True, seed=3)
    s2 = examples.table_chair_scene(use_shapenetsem=True, seed=3)

    assert s1.get_transforms() == s2.get_transforms()

    k1 = examples.kitchen(use_collision_geometry=True, seed=5)
    k2 = examples.kitchen(use_collision_geometry=True, seed=5)

    assert k1.get_transforms() == k2.get_transforms()


def test_forward_kinematics_update_for_scene_without_articulation():
    hole = synth.BoxWithHoleAsset(
        0.1, 0.003, 0.2, hole_width=0.03, hole_height=0.03, hole_offset=(0.03, 0.05)
    ).scene("hole")
    hole.invalidate_scenegraph_cache()

    assert True
