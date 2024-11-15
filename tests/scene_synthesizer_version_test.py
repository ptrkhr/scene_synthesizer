# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Version unit tests for the `scene_synthesizer` package."""

# SRL
import scene_synthesizer


def test_scene_synthesizer_version():
    """Test `scene_synthesizer` package version is set."""
    assert scene_synthesizer.__version__ is not None
    assert scene_synthesizer.__version__ != ""
