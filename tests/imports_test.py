# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Standard Library
import random
from pathlib import Path

# Third Party
import numpy as np
import pytest
import yourdfpy

# SRL
import scene_synthesizer as synth

from .test_utils import _skip_if_file_is_missing

TEST_DIR = Path(__file__).parent


def _set_random_seed():
    random.seed(111)
    np.random.seed(111)


def _remove_UUID_string(list_of_strings):
    return [s.split(":")[0] for s in list_of_strings]


@_skip_if_file_is_missing
def test_import_urdf():
    _set_random_seed()

    microwave_urdf = str(TEST_DIR / "data/assets/partnet_mobility_v0/7236/mobility.urdf")
    microwave = synth.Asset(microwave_urdf)

    s = microwave.scene(use_collision_geometry=None)

    geometry_nodes_expected = {
        "object/door_frame-4": ["object/door_frame-4"],
        "object/door_frame-4:EBD05EB73677": ["object/door_frame-4:EBD05EB73677"],
        "object/door_frame-4:31B7AB977E71": ["object/door_frame-4:31B7AB977E71"],
        "object/glass-5": ["object/glass-5"],
        "object/frame-2": ["object/frame-2"],
        "object/frame-2:CFC98ABD9DFE": ["object/frame-2:CFC98ABD9DFE"],
        "object/control_button-8": ["object/control_button-8"],
        "object/control_button-9": ["object/control_button-9"],
        "object/control_button-10": ["object/control_button-10"],
        "object/control_button-11": ["object/control_button-11"],
        "object/control_button-12": ["object/control_button-12"],
        "object/control_button-13": ["object/control_button-13"],
        "object/control_button-14": ["object/control_button-14"],
        "object/control_button-15": ["object/control_button-15"],
        "object/control_button-16": ["object/control_button-16"],
        "object/control_button-17": ["object/control_button-17"],
        "object/control_button-18": ["object/control_button-18"],
        "object/control_button-19": ["object/control_button-19"],
        "object/control_button-20": ["object/control_button-20"],
        "object/control_button-21": ["object/control_button-21"],
        "object/control_button-22": ["object/control_button-22"],
        "object/control_button-23": ["object/control_button-23"],
        "object/control_button-24": ["object/control_button-24"],
        "object/control_button-25": ["object/control_button-25"],
        "object/control_button-26": ["object/control_button-26"],
        "object/control_button-27": ["object/control_button-27"],
        "object/control_button-28": ["object/control_button-28"],
        "object/control_button-29": ["object/control_button-29"],
        "object/control_button-30": ["object/control_button-30"],
        "object/control_button-31": ["object/control_button-31"],
        "object/display_panel-32": ["object/display_panel-32"],
        "object/display_panel-32:B2AA626A3190": ["object/display_panel-32:B2AA626A3190"],
        "object/control_button-33": ["object/control_button-33"],
        "object/control_button-34": ["object/control_button-34"],
        "object/control_button-35": ["object/control_button-35"],
        "object/control_button-36": ["object/control_button-36"],
        "object/control_button-37": ["object/control_button-37"],
        "object/control_button-38": ["object/control_button-38"],
        "object/control_button-39": ["object/control_button-39"],
        "object/control_button-40": ["object/control_button-40"],
        "object/original-6.obj": ["object/original-6.obj"],
        "object/original-4.obj": ["object/original-4.obj"],
        "object/original-1.obj": ["object/original-1.obj"],
        "object/original-15.obj": ["object/original-15.obj"],
        "object/original-3.obj": ["object/original-3.obj"],
        "object/original-5.obj": ["object/original-5.obj"],
        "object/original-55.obj": ["object/original-55.obj"],
        "object/original-32.obj": ["object/original-32.obj"],
        "object/original-24.obj": ["object/original-24.obj"],
        "object/original-38.obj": ["object/original-38.obj"],
        "object/original-34.obj": ["object/original-34.obj"],
        "object/original-40.obj": ["object/original-40.obj"],
        "object/original-33.obj": ["object/original-33.obj"],
        "object/original-52.obj": ["object/original-52.obj"],
        "object/original-47.obj": ["object/original-47.obj"],
        "object/original-36.obj": ["object/original-36.obj"],
        "object/original-27.obj": ["object/original-27.obj"],
        "object/original-37.obj": ["object/original-37.obj"],
        "object/original-35.obj": ["object/original-35.obj"],
        "object/original-23.obj": ["object/original-23.obj"],
        "object/original-29.obj": ["object/original-29.obj"],
        "object/original-46.obj": ["object/original-46.obj"],
        "object/original-20.obj": ["object/original-20.obj"],
        "object/original-28.obj": ["object/original-28.obj"],
        "object/original-25.obj": ["object/original-25.obj"],
        "object/original-39.obj": ["object/original-39.obj"],
        "object/original-17.obj": ["object/original-17.obj"],
        "object/original-21.obj": ["object/original-21.obj"],
        "object/original-22.obj": ["object/original-22.obj"],
        "object/original-18.obj": ["object/original-18.obj"],
        "object/original-7.obj": ["object/original-7.obj"],
        "object/original-19.obj": ["object/original-19.obj"],
        "object/original-31.obj": ["object/original-31.obj"],
        "object/original-26.obj": ["object/original-26.obj"],
        "object/original-30.obj": ["object/original-30.obj"],
        "object/original-43.obj": ["object/original-43.obj"],
        "object/original-13.obj": ["object/original-13.obj"],
        "object/original-16.obj": ["object/original-16.obj"],
        "object/original-12.obj": ["object/original-12.obj"],
        "object/original-50.obj": ["object/original-50.obj"],
    }
    graph_nodes_expected = [
        "world",
        "object",
        "object/link_1",
        "object/link_0",
        "object/base",
        "object/door_frame-4",
        "object/door_frame-4:EBD05EB73677",
        "object/door_frame-4:31B7AB977E71",
        "object/glass-5",
        "object/frame-2",
        "object/frame-2:CFC98ABD9DFE",
        "object/control_button-8",
        "object/control_button-9",
        "object/control_button-10",
        "object/control_button-11",
        "object/control_button-12",
        "object/control_button-13",
        "object/control_button-14",
        "object/control_button-15",
        "object/control_button-16",
        "object/control_button-17",
        "object/control_button-18",
        "object/control_button-19",
        "object/control_button-20",
        "object/control_button-21",
        "object/control_button-22",
        "object/control_button-23",
        "object/control_button-24",
        "object/control_button-25",
        "object/control_button-26",
        "object/control_button-27",
        "object/control_button-28",
        "object/control_button-29",
        "object/control_button-30",
        "object/control_button-31",
        "object/display_panel-32",
        "object/display_panel-32:B2AA626A3190",
        "object/control_button-33",
        "object/control_button-34",
        "object/control_button-35",
        "object/control_button-36",
        "object/control_button-37",
        "object/control_button-38",
        "object/control_button-39",
        "object/control_button-40",
        "object/original-6.obj",
        "object/original-4.obj",
        "object/original-1.obj",
        "object/original-15.obj",
        "object/original-3.obj",
        "object/original-5.obj",
        "object/original-55.obj",
        "object/original-32.obj",
        "object/original-24.obj",
        "object/original-38.obj",
        "object/original-34.obj",
        "object/original-40.obj",
        "object/original-33.obj",
        "object/original-52.obj",
        "object/original-47.obj",
        "object/original-36.obj",
        "object/original-27.obj",
        "object/original-37.obj",
        "object/original-35.obj",
        "object/original-23.obj",
        "object/original-29.obj",
        "object/original-46.obj",
        "object/original-20.obj",
        "object/original-28.obj",
        "object/original-25.obj",
        "object/original-39.obj",
        "object/original-17.obj",
        "object/original-21.obj",
        "object/original-22.obj",
        "object/original-18.obj",
        "object/original-7.obj",
        "object/original-19.obj",
        "object/original-31.obj",
        "object/original-26.obj",
        "object/original-30.obj",
        "object/original-43.obj",
        "object/original-13.obj",
        "object/original-16.obj",
        "object/original-12.obj",
        "object/original-50.obj",
    ]
    graph_edges_expected = [
        ("world", "object"),
        ("object/link_1", "object/link_0"),
        ("object/base", "object/link_1"),
        ("object/link_0", "object/door_frame-4"),
        ("object/link_0", "object/door_frame-4:EBD05EB73677"),
        ("object/link_0", "object/door_frame-4:31B7AB977E71"),
        ("object/link_0", "object/glass-5"),
        ("object/link_1", "object/frame-2"),
        ("object/link_1", "object/frame-2:CFC98ABD9DFE"),
        ("object/link_1", "object/control_button-8"),
        ("object/link_1", "object/control_button-9"),
        ("object/link_1", "object/control_button-10"),
        ("object/link_1", "object/control_button-11"),
        ("object/link_1", "object/control_button-12"),
        ("object/link_1", "object/control_button-13"),
        ("object/link_1", "object/control_button-14"),
        ("object/link_1", "object/control_button-15"),
        ("object/link_1", "object/control_button-16"),
        ("object/link_1", "object/control_button-17"),
        ("object/link_1", "object/control_button-18"),
        ("object/link_1", "object/control_button-19"),
        ("object/link_1", "object/control_button-20"),
        ("object/link_1", "object/control_button-21"),
        ("object/link_1", "object/control_button-22"),
        ("object/link_1", "object/control_button-23"),
        ("object/link_1", "object/control_button-24"),
        ("object/link_1", "object/control_button-25"),
        ("object/link_1", "object/control_button-26"),
        ("object/link_1", "object/control_button-27"),
        ("object/link_1", "object/control_button-28"),
        ("object/link_1", "object/control_button-29"),
        ("object/link_1", "object/control_button-30"),
        ("object/link_1", "object/control_button-31"),
        ("object/link_1", "object/display_panel-32"),
        ("object/link_1", "object/display_panel-32:B2AA626A3190"),
        ("object/link_1", "object/control_button-33"),
        ("object/link_1", "object/control_button-34"),
        ("object/link_1", "object/control_button-35"),
        ("object/link_1", "object/control_button-36"),
        ("object/link_1", "object/control_button-37"),
        ("object/link_1", "object/control_button-38"),
        ("object/link_1", "object/control_button-39"),
        ("object/link_1", "object/control_button-40"),
        ("object/link_0", "object/original-6.obj"),
        ("object/link_0", "object/original-4.obj"),
        ("object/link_0", "object/original-1.obj"),
        ("object/link_0", "object/original-15.obj"),
        ("object/link_1", "object/original-3.obj"),
        ("object/link_1", "object/original-5.obj"),
        ("object/link_1", "object/original-55.obj"),
        ("object/link_1", "object/original-32.obj"),
        ("object/link_1", "object/original-24.obj"),
        ("object/link_1", "object/original-38.obj"),
        ("object/link_1", "object/original-34.obj"),
        ("object/link_1", "object/original-40.obj"),
        ("object/link_1", "object/original-33.obj"),
        ("object/link_1", "object/original-52.obj"),
        ("object/link_1", "object/original-47.obj"),
        ("object/link_1", "object/original-36.obj"),
        ("object/link_1", "object/original-27.obj"),
        ("object/link_1", "object/original-37.obj"),
        ("object/link_1", "object/original-35.obj"),
        ("object/link_1", "object/original-23.obj"),
        ("object/link_1", "object/original-29.obj"),
        ("object/link_1", "object/original-46.obj"),
        ("object/link_1", "object/original-20.obj"),
        ("object/link_1", "object/original-28.obj"),
        ("object/link_1", "object/original-25.obj"),
        ("object/link_1", "object/original-39.obj"),
        ("object/link_1", "object/original-17.obj"),
        ("object/link_1", "object/original-21.obj"),
        ("object/link_1", "object/original-22.obj"),
        ("object/link_1", "object/original-18.obj"),
        ("object/link_1", "object/original-7.obj"),
        ("object/link_1", "object/original-19.obj"),
        ("object/link_1", "object/original-31.obj"),
        ("object/link_1", "object/original-26.obj"),
        ("object/link_1", "object/original-30.obj"),
        ("object/link_1", "object/original-43.obj"),
        ("object/link_1", "object/original-13.obj"),
        ("object/link_1", "object/original-16.obj"),
        ("object/link_1", "object/original-12.obj"),
        ("object/link_1", "object/original-50.obj"),
        ("object", "object/base"),
    ]
    graph_edges = [(x, y) for x, y, _ in s.graph.to_edgelist()]

    assert len(s.graph.nodes) == len(graph_nodes_expected), (
        f"Number of graph nodes does not match for imported URDF {microwave_urdf}."
        f" ({len(s.graph.nodes)} vs. {len(graph_nodes_expected)}"
    )
    assert len(graph_edges) == len(graph_edges_expected), (
        f"Number of graph edges does not match for imported URDF {microwave_urdf}."
        f" ({len(graph_edges)} vs. {len(graph_edges_expected)}"
    )
    assert len(s.graph.geometry_nodes) == len(geometry_nodes_expected), (
        f"Number of geometry nodes does not match for imported URDF {microwave_urdf}."
        f" ({len(s.graph.geometry_nodes)} vs. {len(geometry_nodes_expected)}"
    )
    # Let's avoid this for now
    # assert (
    #     list(s.graph.nodes) == graph_nodes_expected
    # ), f"Graph nodes for imported URDF {microwave_urdf} do not match."
    # assert (
    #     graph_edges == graph_edges_expected
    # ), f"Graph edges for imported URDF {microwave_urdf} do not match."
    # assert (
    #     s.graph.geometry_nodes == geometry_nodes_expected
    # ), f"Geometry nodes for imported URDF {microwave_urdf} do not match."

    for _, v in s.scene.geometry.items():
        assert (
            "layer" in v.metadata
        ), f"Imported URDF {microwave_urdf} has geometries with missing layer metadata."
        assert v.metadata["layer"] in ["visual", "collision"], (
            f"Imported URDF {microwave_urdf} has geometries in layers other than 'visual' or"
            " 'collision'."
        )


@_skip_if_file_is_missing
def test_import_book_usd():
    book_usd = str(TEST_DIR / "data/assets/kitchen_set/assets/Book/Book.usd")

    book = synth.USDAsset(book_usd).scene(use_collision_geometry=False)

    # check number of geometries
    assert (
        len(book.scene.geometry) == 2
    ), f"{book_usd}, loaded number of geometries: {len(book.scene.geometry)}   Expected: 2"

    geometry_names = ["object/Book", "object/pCube134"]
    assert all([x in book.scene.geometry for x in geometry_names])

    # check number of vertices and faces
    num_vertices_expected = {geometry_names[0]: 670, geometry_names[1]: 54}
    num_faces_expected = {geometry_names[0]: 1336, geometry_names[1]: 104}
    for geom_name in geometry_names:
        assert (
            len(book.scene.geometry[geom_name].vertices) == num_vertices_expected[geom_name]
        ), f"Number of imported vertices for {geom_name} of {book_usd} doesn't match expectation."
        assert (
            len(book.scene.geometry[geom_name].faces) == num_faces_expected[geom_name]
        ), f"Number of imported faces for {geom_name} of {book_usd} doesn't match expectation."

    # check vertices and faces
    vertices_expected_100th = [
        [-1.69589126, -0.73031843, 0.99786639],
        [-1.49565065, -0.64447773, 0.99761212],
        [-1.39608788, -0.91009253, 0.95720756],
        [-1.58035254, -0.7956627, 0.98860884],
        [-1.42585719, -0.74026322, 0.98437375],
        [-1.41611588, -0.78119057, 0.96075684],
        [-1.47723019, -1.05890846, 0.99403298],
    ]
    faces_expected_100th = [
        [275, 278, 277],
        [386, 397, 396],
        [583, 584, 517],
        [66, 67, 78],
        [188, 199, 569],
        [66, 653, 654],
        [307, 318, 603],
        [322, 311, 310],
        [442, 443, 425],
        [414, 542, 541],
        [113, 112, 101],
        [234, 233, 222],
        [187, 198, 619],
        [436, 438, 635],
    ]
    assert np.allclose(
        book.scene.geometry[geometry_names[0]].vertices[::100], vertices_expected_100th
    ), f"Vertices of imported asset {book_usd} not as expected."
    assert np.allclose(
        book.scene.geometry[geometry_names[0]].faces[::100], faces_expected_100th
    ), f"Faces of imported asset {book_usd} not as expected."

    # check colors
    colors_expected = {
        geometry_names[0]: [102, 102, 102, 255],
        geometry_names[1]: [255, 255, 175, 255],
    }
    for geom_name in geometry_names:
        assert np.allclose(
            book.scene.geometry[geom_name].visual.vertex_colors,
            [colors_expected[geom_name]] * num_vertices_expected[geom_name],
        ), f"Color of imported vertices for {geom_name} of {book_usd} doesn't match expectation."
        assert np.allclose(
            book.scene.geometry[geom_name].visual.face_colors,
            [colors_expected[geom_name]] * num_faces_expected[geom_name],
        ), f"Color of imported faces for {geom_name} of {book_usd} doesn't match expectation."


@_skip_if_file_is_missing
def test_import_articulated_usd_srl_top_cabinet():
    _set_random_seed()

    cabinet_usd = str(TEST_DIR / "data/assets/srl_kitchen/srl-top-cabinet.usd")
    cabinet = synth.USDAsset(cabinet_usd).scene(use_collision_geometry=False)

    assert cabinet.get_joint_names() == ["object/RevoluteJoint"]

    assert np.allclose(cabinet.get_joint_limits(), [[0.0, 2.61799388]])

    assert np.allclose(cabinet.get_extents(), [0.38114257, 0.4332533, 0.76199994], atol=1e-3)


@_skip_if_file_is_missing
def test_resizing_and_origin_articulated_usd_srl_top_cabinet():
    _set_random_seed()

    cabinet_usd = str(TEST_DIR / "data/assets/srl_kitchen/srl-top-cabinet.usd")
    cabinet = synth.USDAsset(
        cabinet_usd, size=(1.0, 1.0, 1.0), origin=("top", "center", "bottom")
    ).scene(use_collision_geometry=False)

    assert np.allclose(cabinet.get_extents(), 1.0)
    assert np.allclose(cabinet.get_bounds(), [[-1, -0.5, 0], [0, 0.5, 1]], atol=1e-5)


@_skip_if_file_is_missing
def test_usd_overlay_joint_positions():
    _set_random_seed()

    kitchen_usd = str(TEST_DIR / "data/assets/srl_kitchen/scene_overlay.usd")
    s = synth.Asset(kitchen_usd).scene(use_collision_geometry=False)
    
    expected_joint_configuration = [0.0,
        1.4078575355039886,
        1.5640660648915827,
        1.4466279433933185,
        1.5353283473574306,
        0.0,
        0.0,
        0.0,
        1.4454045405992215,
        1.4368109327516652,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5123898839969739,
        0.5060037315260916,
        0.0,
        1.5707963267948966
    ]


    assert np.allclose(expected_joint_configuration, s.get_configuration())

@_skip_if_file_is_missing
def test_usd_overlay_joint_positions_missing_one_joint_position():
    _set_random_seed()

    kitchen_usd = str(TEST_DIR / "data/assets/srl_kitchen/scene_overlay_missing_freezer_joint.usd")
    s = synth.Asset(kitchen_usd).scene(use_collision_geometry=False)
    
    expected_joint_configuration = [0.0,
        1.4078575355039886,
        1.5640660648915827,
        1.4466279433933185,
        1.5353283473574306,
        0.0,
        0.0,
        0.0,
        1.4454045405992215,
        1.4368109327516652,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5123898839969739,
        0.5060037315260916,
        0.0,
        0.0
    ]

    assert np.allclose(expected_joint_configuration, s.get_configuration())

@_skip_if_file_is_missing
def test_nested_mesh_with_nonxform_def():
    _set_random_seed()

    kitchen_usd = str(TEST_DIR / "data/assets/srl_kitchen/table_colored_with_empty_def.usd")
    s = synth.Asset(kitchen_usd).scene(use_collision_geometry=False)

    assert len(s.geometry) > 0

@_skip_if_file_is_missing
def test_scaling_uniform_for_urdf():
    _set_random_seed()

    robot_urdf = str(TEST_DIR / "data/assets/robots/franka_description/franka_panda.urdf")
    robot = synth.Asset(robot_urdf, height=1.5)

    s = robot.scene(use_collision_geometry=True)

    assert s.scene.extents[2] == 1.5


@_skip_if_file_is_missing
def test_scaling_nonuniform_for_urdf():
    _set_random_seed()

    desired_size = [0.5, 0.3, 1.6]
    robot_urdf = str(TEST_DIR / "data/assets/robots/franka_description/franka_panda.urdf")
    robot = synth.Asset(robot_urdf, size=desired_size).scene(use_collision_geometry=True)

    assert np.allclose(robot.scene.extents, desired_size)

    robot = synth.Asset(
        robot_urdf, width=desired_size[0], depth=desired_size[1], height=desired_size[2]
    ).scene(use_collision_geometry=True)

    assert np.allclose(robot.scene.extents, desired_size)

    robot.update_configuration((0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0))

    assert np.allclose(robot.scene.extents, [1.25967642, 0.784365, 0.8442879])


@_skip_if_file_is_missing
def test_scaling_nonuniform_origin_for_urdf():
    _set_random_seed()

    desired_size = [0.5, 0.3, 1.6]
    origin = ("top", "top", "top")
    robot_urdf = str(TEST_DIR / "data/assets/robots/franka_description/franka_panda.urdf")
    robot = synth.Asset(robot_urdf, size=desired_size, origin=origin).scene(
        use_collision_geometry=True
    )
    assert np.allclose(robot.scene.bounds[1], np.zeros(3))

    origin = ("bottom", "bottom", "bottom")
    robot = synth.Asset(robot_urdf, size=desired_size, origin=origin).scene(
        use_collision_geometry=True
    )
    assert np.allclose(robot.scene.bounds[0], np.zeros(3))

    origin = ("center", "center", "center")
    robot = synth.Asset(robot_urdf, size=desired_size, origin=origin).scene(
        use_collision_geometry=True
    )
    assert np.allclose(-robot.scene.bounds[0], robot.scene.bounds[1])


@_skip_if_file_is_missing
def test_special_configuration_initialization_urdf():
    _set_random_seed()
    fname = str(TEST_DIR / "data/assets/partnet_mobility_v0/7236/mobility.urdf")

    urdf_model = yourdfpy.URDF.load(fname)
    expected_limit_lower = np.array(
        [
            urdf_model.joint_map[joint_name].limit.lower
            for joint_name in urdf_model.actuated_joint_names
        ]
    )
    expected_limit_upper = np.array(
        [
            urdf_model.joint_map[joint_name].limit.upper
            for joint_name in urdf_model.actuated_joint_names
        ]
    )

    s_lower = synth.URDFAsset(fname, configuration="lower").scene(use_collision_geometry=False)
    actual_cfg = s_lower.get_configuration()
    assert np.allclose(expected_limit_lower, actual_cfg)

    s_upper = synth.URDFAsset(fname, configuration="upper").scene(use_collision_geometry=False)
    actual_cfg = s_upper.get_configuration()
    assert np.allclose(expected_limit_upper, actual_cfg)


@_skip_if_file_is_missing
def test_special_configuration_initialization_urdf_with_defaults():
    _set_random_seed()

    fname = str(TEST_DIR / "data/assets/partnet_mobility_v0/101931/mobility.urdf")
    # fname = str(TEST_DIR / "data/assets/robots/franka_description/franka_panda.urdf")

    default_value = 55.0

    urdf_model = yourdfpy.URDF.load(fname)
    expected_limit_lower = np.array(
        [
            urdf_model.joint_map[joint_name].limit.lower
            if urdf_model.joint_map[joint_name].limit is not None
            else default_value
            for joint_name in urdf_model.actuated_joint_names
        ]
    )
    expected_limit_upper = np.array(
        [
            urdf_model.joint_map[joint_name].limit.upper
            if urdf_model.joint_map[joint_name].limit is not None
            else default_value
            for joint_name in urdf_model.actuated_joint_names
        ]
    )

    s_lower = synth.URDFAsset(
        fname,
        configuration="lower",
        default_joint_limit_lower=default_value,
        default_joint_limit_upper=default_value,
    ).scene(use_collision_geometry=False)
    actual_cfg = s_lower.get_configuration()
    assert np.allclose(expected_limit_lower, actual_cfg)

    s_upper = synth.URDFAsset(
        fname,
        configuration="upper",
        default_joint_limit_lower=default_value,
        default_joint_limit_upper=default_value,
    ).scene(use_collision_geometry=False)
    actual_cfg = s_upper.get_configuration()
    assert np.allclose(expected_limit_upper, actual_cfg)


@_skip_if_file_is_missing
def test_special_configuration_initialization_usd():
    _set_random_seed()
    fname = str(TEST_DIR / "data/assets/srl_kitchen/srl-top-cabinet.usd")

    expected_limit_lower = np.array([0])
    expected_limit_upper = np.array([np.deg2rad(150)])

    s_lower = synth.USDAsset(fname, configuration="lower").scene(use_collision_geometry=False)
    actual_cfg = s_lower.get_configuration()
    assert np.allclose(expected_limit_lower, actual_cfg)

    s_upper = synth.USDAsset(fname, configuration="upper").scene(use_collision_geometry=False)
    actual_cfg = s_upper.get_configuration()
    assert np.allclose(expected_limit_upper, actual_cfg)


@_skip_if_file_is_missing
def test_scaling_URDF_prismatic_joint_limit():
    _set_random_seed()
    window_fname = str(TEST_DIR / "data/assets/partnet_mobility_v0/103148/mobility.urdf")

    s = synth.Asset(window_fname).scene()
    prismatic_joint_limits = s.get_joint_limits()

    s = synth.Asset(window_fname, scale=3.0).scene()
    assert np.allclose(prismatic_joint_limits * 3.0, s.get_joint_limits())

    # The prismatic joint axes of the window are along y
    s = synth.Asset(window_fname, scale=(3.0, 1.0, 5.0)).scene()
    assert np.allclose(prismatic_joint_limits * 1.0, s.get_joint_limits())

@_skip_if_file_is_missing
def test_import_glb():
    _set_random_seed()

    # this is a particularly interesting asset since it has a different geometry and node name
    bowl_fname = str(TEST_DIR / "data/assets/bowl.glb")
    x = synth.Asset(bowl_fname).scene(use_collision_geometry=False)
    
    assert True

@_skip_if_file_is_missing
def test_front_up_vectors():
    _set_random_seed()

    bowl_fname = str(TEST_DIR / "data/assets/shapenetsem/Mug/10f6e09036350e92b3f21f1137c3c347.obj")
    x = synth.Asset(bowl_fname, front=[1, 0, 0], up=[0, 1, 0]).scene(use_collision_geometry=False)

    with pytest.raises(Exception) as exc_info:
        x = synth.Asset(bowl_fname, front=[0.9, 0.1, 0], up=[0.1, 0.9, 0]).scene(use_collision_geometry=False)

    assert 'not orthogonal' in exc_info.value.args[0]

    x = synth.Asset(bowl_fname, front=[0.9, 0.1, 0], up=[0.1, 0.9, 0], tolerance_up_front_orthogonality=0.2).scene(use_collision_geometry=False)
