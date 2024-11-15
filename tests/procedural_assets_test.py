# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from scene_synthesizer import procedural_assets


@pytest.fixture(scope="module")
def golden_dir():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cur_dir, "golden/")


def _set_random_seed():
    random.seed(111)
    np.random.seed(111)


def test_shelf_export_urdf(tmpdir, golden_dir):
    output_fname = os.path.join(tmpdir, "shelf_asset.urdf")
    output_fname_golden = os.path.join(golden_dir, "shelf_asset.urdf")

    _set_random_seed()
    shelf = procedural_assets.ShelfAsset(
        width=1, depth=1, height=1, num_boards=2, origin=("center", "center", "top")
    )
    s = shelf.scene("shelf")
    s.set_density("shelf", 100.0)
    s.export(output_fname, single_geometry_per_link=True)

    generated_urdf = yourdfpy.URDF.load(output_fname, build_scene_graph=False, load_meshes=False)
    golden_urdf = yourdfpy.URDF.load(
        output_fname_golden, build_scene_graph=False, load_meshes=False
    )

    assert (
        generated_urdf == golden_urdf
    ), f"The generated file {output_fname} and the reference {output_fname_golden} differ!"


def test_shelf_export_urdf_single_link(tmpdir, golden_dir):
    output_fname = os.path.join(tmpdir, "shelf_asset.urdf")
    output_fname_golden = os.path.join(golden_dir, "shelf_asset_single_link.urdf")

    _set_random_seed()
    shelf = procedural_assets.ShelfAsset(
        width=1, depth=1, height=1, num_boards=2, origin=("center", "center", "top")
    )
    s = shelf.scene("shelf")
    s.set_density("shelf", 100.0)
    s.export(output_fname, single_geometry_per_link=False)

    generated_urdf = yourdfpy.URDF.load(output_fname, build_scene_graph=False, load_meshes=False)
    golden_urdf = yourdfpy.URDF.load(
        output_fname_golden, build_scene_graph=False, load_meshes=False
    )

    assert (
        generated_urdf == golden_urdf
    ), f"The generated file {output_fname} and the reference {output_fname_golden} differ!"


def test_recursively_partitioned_cabinet_export_urdf(tmpdir, golden_dir):
    output_fname = os.path.join(tmpdir, "cabinet_asset.urdf")
    output_fname_golden = os.path.join(golden_dir, "cabinet_asset.urdf")

    _set_random_seed()
    cabinet = procedural_assets.RecursivelyPartitionedCabinetAsset(
        width=1,
        depth=1,
        height=1,
        use_box_handle=True,
        seed=12,
    )
    s = cabinet.scene().colorize()
    s.set_density("object", 100.0)
    s.export(output_fname, single_geometry_per_link=True)

    generated_urdf = yourdfpy.URDF.load(output_fname, build_scene_graph=False, load_meshes=False)
    golden_urdf = yourdfpy.URDF.load(
        output_fname_golden, build_scene_graph=False, load_meshes=False
    )

    assert (
        generated_urdf == golden_urdf
    ), f"The generated file {output_fname} and the reference {output_fname_golden} differ!"


def test_recursively_partitioned_cabinet_export_urdf_single_link(tmpdir, golden_dir):
    output_fname = os.path.join(tmpdir, "cabinet_asset.urdf")
    output_fname_golden = os.path.join(golden_dir, "cabinet_asset_single_link.urdf")

    _set_random_seed()
    cabinet = procedural_assets.RecursivelyPartitionedCabinetAsset(
        width=1,
        depth=1,
        height=1,
        use_box_handle=True,
        seed=12,
    )
    s = cabinet.scene().colorize()
    s.set_density("object", 100.0)
    s.export(output_fname, single_geometry_per_link=False)

    generated_urdf = yourdfpy.URDF.load(output_fname, build_scene_graph=False, load_meshes=False)
    golden_urdf = yourdfpy.URDF.load(
        output_fname_golden, build_scene_graph=False, load_meshes=False
    )

    assert (
        generated_urdf == golden_urdf
    ), f"The generated file {output_fname} and the reference {output_fname_golden} differ!"


def test_microwave_asset():
    _set_random_seed()

    s = procedural_assets.MicrowaveAsset(
        width=0.6, depth=0.4, height=0.3, thickness=0.05, display_panel_width=0.14, handle_left=True
    ).scene()

    # check width and height -- not depth since it's depending on the handle
    assert np.allclose(s.scene.extents[[0, 2]], [0.6, 0.3])


def test_refrigerator_asset():
    _set_random_seed()

    s = procedural_assets.RefrigeratorAsset(
        width=0.6,
        depth=0.4,
        height=1.3,
        thickness=0.05,
        freezer_compartment_height=0.15,
        handle_left=True,
    ).scene()

    # check width and height -- not depth since it's depending on the handle
    assert np.allclose(s.scene.extents[[0, 2]], [0.6, 1.3])


def test_cabinet_asset():
    _set_random_seed()

    compartment_mask = np.array([[0, 0], [1, 2], [3, 3]])
    compartment_types = ["drawer", "door_right", "door_left", "closed"]
    compartment_widths = [0.4, 0.4]
    compartment_heights = [0.2, 0.5, 0.07]

    outer_wall_thickness = 0.02
    inner_wall_thickness = 0.01

    s = procedural_assets.CabinetAsset(
        width=0.6,
        depth=0.4,
        height=1.3,
        outer_wall_thickness=outer_wall_thickness,
        inner_wall_thickness=inner_wall_thickness,
        compartment_mask=compartment_mask,
        compartment_types=compartment_types,
    ).scene()

    assert np.allclose(s.scene.extents[[0, 2]], [0.6, 1.3])

    s = procedural_assets.CabinetAsset(
        width=0.6,
        depth=0.4,
        height=1.3,
        outer_wall_thickness=outer_wall_thickness,
        inner_wall_thickness=inner_wall_thickness,
        compartment_mask=compartment_mask,
        compartment_types=compartment_types,
        compartment_widths=compartment_widths,
        compartment_heights=compartment_heights,
    ).scene()

    assert np.allclose(s.scene.extents[[0, 2]], [0.6, 1.3])

    s = procedural_assets.CabinetAsset(
        width=None,
        depth=0.4,
        height=None,
        outer_wall_thickness=outer_wall_thickness,
        inner_wall_thickness=inner_wall_thickness,
        compartment_mask=compartment_mask,
        compartment_types=compartment_types,
        compartment_widths=compartment_widths,
        compartment_heights=compartment_heights,
    ).scene()

    assert np.allclose(
        s.scene.extents[[0, 2]],
        [
            np.sum(compartment_widths) + 2.0 * outer_wall_thickness,
            np.sum(compartment_heights) + 2.0 * outer_wall_thickness,
        ],
    )


def test_oven_asset():
    _set_random_seed()

    stove_plate_height = 0.01
    height = 0.9
    width = 0.7
    s = procedural_assets.RangeAsset(
        width=width, depth=0.6, height=height, stove_plate_height=stove_plate_height
    ).scene()

    assert np.allclose(s.scene.extents[[0, 2]], [width, height + stove_plate_height])


def test_dishwasher_asset():
    _set_random_seed()

    height = 0.9
    width = 0.7
    s = procedural_assets.DishwasherAsset(width=width, depth=0.6, height=height).scene()

    assert np.allclose(s.scene.extents[[0, 2]], [width, height])


def test_knob_asset():
    _set_random_seed()

    height = 0.03
    width = 0.03
    depth = 0.05
    s = procedural_assets.KnobAsset(
        width=width, depth=depth, height=height, num_depth_sections=64
    ).scene()

    assert np.allclose(s.scene.extents, [width, height, depth], atol=1.0e-5)


def test_sinkcabinet_asset():
    _set_random_seed()

    height = 0.9
    width = 0.7
    s = procedural_assets.SinkCabinetAsset(width=width, depth=0.6, height=height).scene()

    assert np.allclose(s.scene.extents[[0, 2]], [width, height])


def test_bin_asset():
    _set_random_seed()

    width = 0.7
    depth = 0.3
    height = 0.2
    s = procedural_assets.BinAsset(width=width, depth=depth, height=height, thickness=0.003).scene()

    assert np.allclose(s.scene.extents, [width, depth, height])


def test_boxwithhole_asset():
    _set_random_seed()

    width = 0.7
    depth = 0.3
    height = 0.2
    s = synth.BoxWithHoleAsset(
        width=width,
        depth=depth,
        height=height,
        hole_width=width / 2.0,
        hole_depth=depth / 2.0,
    ).scene()

    assert np.allclose(s.scene.extents, [width, depth, height])


def test_cubbyshelf_asset():
    _set_random_seed()

    width = 0.7
    depth = 0.3
    height = 0.2
    s = procedural_assets.CubbyShelfAsset(
        width=width,
        depth=depth,
        height=height,
        thickness=0.003,
        compartment_mask=[[0]],
        compartment_types=["open"],
    ).scene()

    assert np.allclose(s.scene.extents, [width, depth, height])


def test_rangehood_asset():
    _set_random_seed()

    width = 0.8
    depth = 0.6
    height = 1.0
    s = procedural_assets.RangeHoodAsset(
        width=width,
        depth=depth,
        height=height,
    ).scene()

    assert np.allclose(s.scene.extents, [width, depth, height])


def test_kitchenisland_asset():
    _set_random_seed()

    width = 0.8
    depth = 0.9
    height = 1.0
    s = procedural_assets.KitchenIslandAsset(
        width=width,
        depth=depth,
        height=height,
    ).scene()

    assert np.allclose(s.scene.extents, [width, depth, height])

def test_handwheel_asset():
    _set_random_seed()

    radius = 0.4
    rim_width = 0.05
    s = procedural_assets.HandWheelAsset(
        radius=radius,
        rim_width=rim_width,
        num_spokes=3,
        spoke_angle=0.0,
        spoke_width=0.01,
        hub_height=0.05,
        hub_radius=0.003,
        handle_height=0.0,
        num_major_segments=128,
        num_minor_segments=64,
    ).scene()
    
    expected_bounds = [
        [-(radius + rim_width/2.0), -(radius + rim_width/2.0), -rim_width/2.0],
        [+(radius + rim_width/2.0), +(radius + rim_width/2.0), +rim_width/2.0],
    ]

    assert np.allclose(expected_bounds, s.get_bounds())

def test_cncmachine_asset():
    _set_random_seed()

    expected_extents = np.array([1.6, 0.8, 0.9])
    actual_extents = procedural_assets.CNCMachineAsset(*expected_extents, handle_length=expected_extents[2] * 0.3, button_size=None).scene().get_extents()
    
    assert np.allclose(expected_extents, actual_extents)

def test_safetyswitch_asset():
    _set_random_seed()

    s = procedural_assets.SafetySwitchAsset(fuse_box_width=0.1, fuse_box_depth=0.1, fuse_box_height=0.2, lever_length=0.1, lever_right_of_box=False).scene('switch')
    expected_bounds = np.array([[-0.05, -0.11, -0.01], [ 0.05,  0.01,  0.19]])

    np.allclose(expected_bounds, s.get_bounds())

    s.update_configuration([0.5])
    expected_bounds = np.array([[-0.05      , -0.11      , -0.01      ], [ 0.06151263,  0.01      ,  0.19      ]])

    np.allclose(expected_bounds, s.get_bounds())
    
