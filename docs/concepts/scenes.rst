######
Scenes
######

The ``Scene`` class contains most of ``scene_synthesizer``'s functionality.

..  code-block:: python

    import scene_synthesizer as synth
    my_scene = synth.Scene()

Adding Assets to Scenes
=======================

Once assets are instantiated, they can be added to the ``scene`` object at the specified ``transform`` pose.

..  code-block:: python

    import numpy as np
    from scene_synthesizer import procedural_assets as pa
    table = pa.TableAsset(1, 1, 0.75)

    my_scene.add_object(
        asset=table,
        obj_id="table",
        transform=np.eye(4),
    )
    my_scene.show_graph() # show scene graph

.. figure:: ../_static/scene_graph.png
   :align: center
   :width: 100%

The scene graph is a directed acyclic graph with a single root (usually ``"world"`` but this can be defined via the ``base_frame`` argument of the scene constructor).
Each edge has a transformation attached to it.
Each node can have a geometry attached to it.
All nodes that belong to the same object will share the same namespace (name prefix, separated by a ``/``).

.. _anchor_based:

Anchor-based Compositioning
===========================

Assets can also be added to a ``scene`` object in relation to assets previously added to the scene.
Anchors define reference points on assets or subscenes.
They can be: ``"top"``, ``"bottom"``, ``"center"``, ``"com"``, ``"centroid"``, ``"left"``, ``"right"``, ``"front"`` or ``"back"``.
Their meaning is the same as in the asset's ``origin`` argument described :ref:`here<origin_settings>`.

When using the arguments ``connect_parent_anchor`` and ``connect_obj_anchor`` the ``add_object`` method calculates the resulting transform automatically:

..  code-block:: python

    import scene_synthesizer as synth
    from scene_synthesizer import procedural_assets as pa
    
    table = pa.TableAsset(1, 1, 0.75)
    shelf = pa.ShelfAsset(1, 0.5, 2.0, num_boards=6)
    
    s = synth.Scene()
    s.add_object(table, 'table')

    s.add_object(
        shelf,
        'shelf',
        connect_parent_id='table',
        connect_parent_anchor=('right', 'back', 'bottom'),
        connect_obj_anchor=('left', 'back', 'bottom')
    )

    s.colorize() # make it a bit more colorful

    s.show() # show scene in trimesh visualizer

The generated scene should look like this:

.. figure:: ../_static/example_connect_objects.png
   :align: center
   :width: 50%

We can also align the shelf to the front of the table by doing:

..  code-block:: python

    s.add_object(
        shelf,
        'shelf',
        connect_parent_id='table',
        connect_parent_anchor=('right', 'front', 'bottom'),
        connect_obj_anchor=('left', 'front', 'bottom')
    )

.. figure:: ../_static/example_connect_objects2.png
   :align: center
   :width: 50%

``scene_synthesizer`` comes with six predefined kitchen scenes as part of the module ``scene_synthesizer.procedural_scenes``.
Details are shown in the next section.
