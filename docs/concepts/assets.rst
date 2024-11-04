######
Assets
######

Scenes are composed of objects which are instantiated via the ``Asset`` class. There are two types of assets: procedural assets and file-based assets.
File-based assets can be loaded from a variety of common mesh formats such as STL, PLY, GLTF/GLB, OBJ, DAE, and even formats describing articulations such as `URDF <https://wiki.ros.org/urdf/XML/model>`__, `MJCF <https://mujoco.readthedocs.io/en/latest/modeling.html>`__, or `USD <https://openusd.org/release/index.html>`__.

The Asset constructor has multiple optional arguments:

..  code-block:: python

    import scene_synthesizer as synth
    chair = synth.Asset(
        '<PATH_TO_MESH>.obj',
        height=0.12,
        origin=('center', 'center', 'bottom'),
        )

.. _origin_settings:

The ``origin`` parameter can be used to define a new coordinate origin different from the one of the loaded mesh file.
It defines a relative coordinate along XYZ via keywords:

.. list-table::
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Value of ``origin``
     - Description
   * - `com`
     - Center of mass
   * - `center`
     - (max(vertices) - min(vertices)) / 2
   * - `centroid`
     - Mean of vertices
   * - `left`, `front`, `bottom`
     - Minimum of vertices
   * - `right`, `back`, `top`
     - Maximum of vertices

.. figure:: ../_static/frame_convention.png
   :align: center
   :width: 50%

The size of an asset can be specified in various ways:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - `scale`
     - float
     - All 3 dimensions are multiplied by this scale factor.
   * - `size` / `extents`
     - Tuple[float, float, float]
     - Absolute lengths for all dimensions are specified. Allows for non-uniform scaling.
   * - `width`
     - float
     - Absolute length in X dimension. Y an Z will be scaled to perserve the aspect ratio. Can be combined with `depth` and/or `height`.
   * - `depth`
     - float
     - Absolute length in Y dimension. X an Z will be scaled to perserve the aspect ratio. Can be combined with `width` and/or `height`.
   * - `height`
     - float
     - Absolute length in Z dimension. X an Y will be scaled to perserve the aspect ratio. Can be combined with `width` and/or `depth`.
   * - `max_length`
     - float
     - The longest dimension will be scaled. The other two will be scaled to preserve the aspect ratio.
   * - `min_length`
     - float
     - The shortest dimension will be scaled. The other two will be scaled to preserve the aspect ratio.
   * - `max_width_depth`
     - float
     - The longer dimension in X and Y will be scaled. Z will be scaled to preserve the aspect ratio.
   * - `width_scale`
     - float
     - Multipy X dimension by this factor. Can be used in combination with `depth` and/or `height`.
   * - `depth_scale`
     - float
     - Multipy Y dimension by this factor. Can be used in combination with `depth` and/or `height`.
   * - `height_scale`
     - float
     - Multipy Z dimension by this factor. Can be used in combination with `depth` and/or `height`.


The submodule ``scene_synthesizer.procedural_assets`` contains a number of procedural assets. Besides the parameters for file-based assets they also contain object/part-specific arguments:

..  code-block:: python

    from scene_synthesizer import procedural_assets as pa
    table = pa.TableAsset(
        width=1.0,
        depth=1.0,
        height=0.75,
        thickness=0.025,
        leg_thickness=0.025
        )
    
The ``show()`` method provides a simple viewer to quickly examine assets (based on pyglet; use ``w`` for switching to wireframe mode and ``a`` to toggle between showing different axes; see `keyboard shortcuts <https://github.com/mikedh/trimesh/blob/85b4bd1f410d8d8361009c6f27266719a3d2b97d/trimesh/viewer/windowed.py#L650>`_):

.. code-block:: python

    table.show(use_collision_geometry=False)

.. figure:: ../_static/asset_table.png
   :align: center
   :width: 50%

Note, that some asset formats (such as URDF) distinguish between **visual** and **collision** geometry.
This allows for keeping high visual complexity during rendering whereas computing the physical behavior can be based on a simpler geometry.
The ``use_collision_geometry`` argument is used during visualization (see ``show()`` above), and when adding an object to a scene (see ``add_object(..)`` in next section):

+------------------------+------------+----------------------------------------------------------------------+
|                        | Value      | Description                                                          |
+========================+============+======================================================================+
|                        | `True`     | Only the collision geometry is shown/added to the scene.             |
|                        +------------+----------------------------------------------------------------------+
| use_collision_geometry | `False`    | Only the visual geometry is shown/added to the scene.                |
|                        +------------+----------------------------------------------------------------------+
|                        | `None`     | Both, visual and collision geometry are shown/added to the scene.    |
+------------------------+------------+----------------------------------------------------------------------+
