# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Standard Library
import random
from pathlib import Path

# Third Party
import numpy as np
import pytest

# SRL
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets

TEST_DIR = Path(__file__).parent


def _set_random_seed():
    random.seed(111)
    np.random.seed(111)


@pytest.fixture(scope="module")
def cabinet_scene():
    _set_random_seed()

    compartment_mask = np.array([[0, 0], [1, 2], [3, 3]])
    compartment_types = ["drawer", "door_right", "door_left", "closed"]

    return procedural_assets.CabinetAsset(
        width=0.6,
        depth=0.4,
        height=1.3,
        thickness=0.01,
        compartment_mask=compartment_mask,
        compartment_types=compartment_types,
    ).scene()


def test_joint_names(cabinet_scene):
    expected_joint_names = [
        "object/corpus_to_door_0_1",
        "object/corpus_to_door_1_1",
        "object/corpus_to_drawer_0_0",
    ]

    assert cabinet_scene.get_joint_names() == expected_joint_names


def test_joint_properties(cabinet_scene):
    expected_joint_properties = {
        "object/corpus_to_door_0_1": {
            "name": "object/corpus_to_door_0_1",
            "type": "revolute",
            "axis": [0, 0, -1],
            "limit_velocity": 0.1,
            "limit_effort": 1000.0,
            "limit_lower": 0.0,
            "limit_upper": 1.5707963267948966,
            "origin": [
                [1.0, 0.0, 0.0, -0.29],
                [0.0, 1.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.65],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "q": 0.0,
        },
        "object/corpus_to_door_1_1": {
            "name": "object/corpus_to_door_1_1",
            "type": "revolute",
            "axis": [0, 0, 1],
            "limit_velocity": 0.1,
            "limit_effort": 1000.0,
            "limit_lower": 0.0,
            "limit_upper": 1.5707963267948966,
            "origin": [
                [1.0, 0.0, 0.0, 0.29],
                [0.0, 1.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.65],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "q": 0.0,
        },
        "object/corpus_to_drawer_0_0": {
            "name": "object/corpus_to_drawer_0_0",
            "type": "prismatic",
            "axis": [0, -1, 0],
            "limit_velocity": 1.0,
            "limit_effort": 1000.0,
            "limit_lower": 0.0,
            "limit_upper": 0.3159,
            "origin": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.19],
                [0.0, 0.0, 1.0, 1.0791666666666668],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "q": 0.0,
        },
    }

    assert cabinet_scene.get_joint_properties() == expected_joint_properties


def test_get_configuration(cabinet_scene):
    assert np.allclose(cabinet_scene.get_configuration(obj_id="object"), [0.0, 0.0, 0.0])

    assert cabinet_scene.get_configuration(obj_id="object", joint_ids=["corpus_to_door_0_1"]) == 0.0

    assert cabinet_scene.get_configuration(
        joint_ids=["object/corpus_to_door_0_1"]
    ) == cabinet_scene.get_configuration(obj_id="object", joint_ids=["corpus_to_door_0_1"])


def test_forward_kinematics(cabinet_scene):
    expected_T = np.array(
        [[1.0, 0.0, 0.0, 0.29], [0.0, 1.0, 0.0, -0.2], [0.0, 0.0, 1.0, 0.65], [0.0, 0.0, 0.0, 1.0]]
    )
    assert np.allclose(cabinet_scene.get_transform("object/door_1_1"), expected_T)

    cabinet_scene.update_configuration(
        obj_id="object", joint_ids=["corpus_to_door_1_1"], configuration=[1.2]
    )

    expected_T = np.array(
        [
            [0.36235775, -0.93203909, 0.0, 0.29],
            [0.93203909, 0.36235775, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.65],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert np.allclose(cabinet_scene.get_transform("object/door_1_1"), expected_T)


def test_joint_limits(cabinet_scene):
    limits = cabinet_scene.get_joint_limits()
    expected_limits = [[0.0, 1.57079633], [0.0, 1.57079633], [0.0, 0.3159]]
    assert np.allclose(limits, expected_limits)

    cabinet_scene.update_configuration([0.2, 0.5, 0.2])
    config = cabinet_scene.get_configuration()

    assert np.all(config >= limits[:, 0]) and np.all(config <= limits[:, 1])


def test_random_configurations(cabinet_scene):
    limits = cabinet_scene.get_joint_limits()
    for _ in range(5):
        cabinet_scene.random_configurations()
        config = cabinet_scene.get_configuration()

        assert np.all(config >= limits[:, 0]) and np.all(config <= limits[:, 1])


def test_add_remove_joints(cabinet_scene):
    joint_names = cabinet_scene.get_joint_names()
    joint_props = cabinet_scene.get_joint_properties()
    num_joints = len(joint_names)

    joint_to_remove = joint_names[1]
    parent_node, child_node = cabinet_scene.get_joint_parent_child_node(joint_to_remove)
    cabinet_scene.remove_joints(joint_to_remove)

    assert len(cabinet_scene.get_joint_names()) == (num_joints - 1)

    cabinet_scene.add_joint(parent_node, child_node, **joint_props[joint_to_remove])

    assert len(cabinet_scene.get_joint_names()) == num_joints
    assert cabinet_scene.get_joint_properties() == joint_props


def test_find_joint(cabinet_scene):
    assert (
        cabinet_scene.find_joint("object/surface_1_2", include_fixed_floating_joints=False) == None
    )
    assert (
        cabinet_scene.find_joint("object/surface_1_2", include_fixed_floating_joints=True)
        == "object/origin_joint"
    )

    assert (
        cabinet_scene.find_joint("object/drawer_0_0_handle_part_1", False)
        == "object/corpus_to_drawer_0_0"
    )
    assert (
        cabinet_scene.find_joint("object/drawer_0_0_handle_part_1", True)
        == "object/corpus_to_drawer_0_0"
    )


def test_get_joint_types(cabinet_scene):
    joint_types1 = cabinet_scene.get_joint_types()
    joint_types2 = cabinet_scene.get_joint_types(include_fixed_floating_joints=True)

    assert cabinet_scene.get_joint_names() == [
        "object/corpus_to_door_0_1",
        "object/corpus_to_door_1_1",
        "object/corpus_to_drawer_0_0",
    ]
    assert joint_types1 == ["revolute", "revolute", "prismatic"]

    assert cabinet_scene.get_joint_names(include_fixed_floating_joints=True) == [
        "object/corpus_to_door_0_1",
        "object/corpus_to_door_1_1",
        "object/corpus_to_drawer_0_0",
        "object/origin_joint",
        "object/world_fixed_joint",
    ]
    assert joint_types2 == ["revolute", "revolute", "prismatic", "fixed", "fixed"]

    assert cabinet_scene.get_joint_types(
        joint_ids=["object/origin_joint", "object/corpus_to_door_0_1"]
    ) == ["fixed", "revolute"]

    cabinet_scene.set_joint_types(
        ["floating", "floating"], joint_ids=["object/origin_joint", "object/world_fixed_joint"]
    )
    assert cabinet_scene.get_joint_types(
        joint_ids=["object/origin_joint", "object/world_fixed_joint"]
    ) == ["floating", "floating"]

    cabinet_scene.set_joint_types(
        [None, None], joint_ids=["object/origin_joint", "object/world_fixed_joint"]
    )

    with pytest.raises(Exception) as e_info:
        cabinet_scene.get_joint_types(joint_ids=["object/origin_joint", "object/world_fixed_joint"])

        assert str(e_info).startswith("KeyError: 'object/origin_joint'")

    assert (
        len(joint_types2)
        == len(cabinet_scene.get_joint_types(include_fixed_floating_joints=True)) + 2
    )
