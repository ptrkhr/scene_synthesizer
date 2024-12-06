# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import functools

from pxr.Tf import ErrorException

def _skip_if_file_is_missing(func):
    def wrapper(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except ValueError as e:
            if 'is not a file' in str(e):
                pytest.skip(f"Skipping: {e}")
        except ErrorException as e:
            if 'Failed to open layer' in str(e):
                pytest.skip(f"Skipping: {e}")
        except ImportError as e:
            pytest.skip(f"Skipping: {e}")
        return ret
    return functools.update_wrapper(wrapper, func)

