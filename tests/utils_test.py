# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Standard Library
import random

# Third Party
import numpy as np
import pytest
import trimesh.transformations as tra

# SRL
from scene_synthesizer import utils


def _set_random_seed():
    random.seed(111)
    np.random.seed(111)


def test_regex():
    items = {
        "asdfasdf": False,
        "/asdf/asdf": False,
        "/asdf/asdf-32": False,
        "/asdfasd_asdf-2/SDFA": False,
        "/asdfasd_asdf-2/SDFAs": False,
        ".*": True,
        "/example/hello.obj": False,
        "(asdf|asdff)": True,
    }

    for item, result in items.items():
        assert utils.is_regex(item) == result, f"{item} should be a regex: {result}"


def test_select_sublist():
    items = [
        "asdfasdf",
        "/asdf/asdf",
        "/asdf/asdf-32",
        "/asdfasd_asdf-2/SDFA",
        "/asdfasd_asdf-2/SDFAs",
        "/object/mesh.obj",
    ]

    # test individual element
    result = utils.select_sublist(query="/asdfasd_asdf-2/SDFA", all_items=items)
    assert result == ["/asdfasd_asdf-2/SDFA"]

    result = utils.select_sublist(query="/object/mesh.obj", all_items=items)
    assert result == ["/object/mesh.obj"]

    # test regex
    result = utils.select_sublist(query="/asdfasd_asdf-2/.*", all_items=items)
    assert result == ["/asdfasd_asdf-2/SDFA", "/asdfasd_asdf-2/SDFAs"]

    result = utils.select_sublist(query="/asdf/asdf(.{2}|)", all_items=items)
    assert result == ["/asdf/asdf", "/asdf/asdf-32"]

    # test list
    result = utils.select_sublist(query=["/asdf/asdf", "/asdfasd_asdf-2/SDFA"], all_items=items)
    assert result == ["/asdf/asdf", "/asdfasd_asdf-2/SDFA"]

    # check exception
    with pytest.raises(Exception):
        utils.select_sublist(query="doesnt exist", all_items=items)


def test_homogeneous_inverse():
    _set_random_seed()

    for _ in range(10):
        x = tra.random_rotation_matrix()
        x[:3, 3] = np.random.rand(3)

        x_inv_expected = tra.inverse_matrix(x)
        x_inv = utils.homogeneous_inv(x)

        assert np.allclose(x_inv, x_inv_expected)
