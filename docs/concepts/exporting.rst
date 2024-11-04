#########
Exporting
#########

A ``Scene`` object can be exported to several file formats for use in rendereres, physics simulators etc.
The ``export(<filename>)`` method will chose the appropriate exporter based on the filename extension:

..  code-block:: python

    from scene_synthesizer import procedural_scenes as ps

    kitchen = ps.kitchen_island()
    kitchen.export('/tmp/kitchen/kitchen.<obj,glb,stl,usd,urdf,..>')

Alternatively, the method can also return the raw data in the desired format (instead of writing the data to disk).
To do so, only pass ``file_type='<obj,usd,urdf,...>'`` as the desired export format.

.. automethod:: scene_synthesizer.scene.Scene.export
    :noindex:

Note, that there are a lot of format-specific arguments that affect the exported result.
The easiest way to see them is to look up the docstring of the format-specific export function in the ``export`` module.
In the following we show most of them.

Polygon Mesh
============

The scene object can be exported to the following common mesh representations: ``GLTF``, ``GLB``, ``OBJ``, ``STL``, ``PLY``, etc.
Note that this does not preserve articulations.


.. autofunction:: scene_synthesizer.exchange.export.export_obj
    :noindex:

.. autofunction:: scene_synthesizer.exchange.export.export_glb
    :noindex:

.. autofunction:: scene_synthesizer.exchange.export.export_gltf
    :noindex:

.. autofunction:: scene_synthesizer.exchange.export.export_stl
    :noindex:

.. autofunction:: scene_synthesizer.exchange.export.export_ply
    :noindex:

.. figure:: ../_static/blender_kitchen.png
   :align: center
   :width: 100%

   OBJ loaded in Blender.


JSON
====

Sometimes the scene data is required in a more parsable format.
The JSON export serializes the scene into a dictionary containing the scene graph and geometry information.

.. autofunction:: scene_synthesizer.exchange.export.export_json
    :noindex:

URDF
====

The `URDF <https://wiki.ros.org/urdf/XML/model>`__ format is a popular XML specification in robotics.

.. autofunction:: scene_synthesizer.exchange.export.export_urdf
    :noindex:

To create a URDF file that is viewable in `ROS RViz <http://wiki.ros.org/rviz>`_, use the following arguments: 

..  code-block:: python
    
    from scene_synthesizer import procedural_scenes as ps
    
    kitchen = ps.kitchen_island().colorize()
    kitchen.export(
        '/tmp/kitchen/kitchen.urdf',
        use_absolute_mesh_paths=True,
        mesh_path_prefix='file://')

And load it in RViz as follows. Make sure to set ``Global Options -> Fixed Frame`` to the base frame of the scene, here ``world``.

..  code-block:: bash

    roslaunch urdf_tutorial display.launch model:=/tmp/kitchen/kitchen.urdf

.. figure:: ../_static/ros_rviz_kitchen.png
   :align: center
   :width: 100%

   URDF loaded in RViz.

USD
===

The Universal Scene Description (`USD <https://openusd.org/release/index.html>`_) is a versatile file format widely used in rendering and simulation applications.
For example, a USD file can be utilized in platforms like `Isaac Sim <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html>`_ or `Isaac Lab <https://isaac-sim.github.io/IsaacLab/>`_.
Additionally, the file can be inspected using the `usdview <https://graphics.pixar.com/usd/release/toolset.html#usdview>`_ tool.

.. autofunction:: scene_synthesizer.exchange.export.export_usd
    :noindex:

.. figure:: ../_static/isaac_sim_kitchen.png
   :align: center
   :width: 100%

   USD loaded in Isaac Sim.