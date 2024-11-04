####################
Adding MDL Materials
####################

One way to create scenes that ar visually more appealing is to attach materials to the geometries, such as MDLs.

The `Material Definition Language (MDL) <https://www.nvidia.com/en-us/design-visualization/technologies/material-definition-language/>`_ defines the behavior of light at a high level and enables sharing physcially based materials between applications.
The `vMaterials <https://developer.nvidia.com/vmaterials>`_ collection is a curated library of thousands of MDLs that represent real-world materials.

In the following we are showing how to attach MDLs from ``vMaterials`` in a random (but object-specific) manner to a USD kitchen scene generated with ``scene_synthesizer``:

.. literalinclude:: /../examples/add_random_mdl_materials.py

.. figure:: /../imgs/examples/kitchen_add_mdl_materials.png
    :align: center
    :scale: 100%
    
    The resulting USD file opened in Isaac Sim