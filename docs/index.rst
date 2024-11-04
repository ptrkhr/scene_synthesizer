###############################
Scene Synthesizer Documentation
###############################

A Python package to easily create scenes for robot manipulation tasks. For an overview video, see this `NTECH 2021 presentation
<https://nvidia.sharepoint.com/sites/NTECH2/_layouts/15/stream.aspx?id=%2Fsites%2FNTECH2%2FShared%20Documents%2F2021%2FRobotics%20and%20Autonomous%20Vehicles%2FPresentations%2FPresentation%20Recordings%20%28mp4%29%2FNTECH2021%5F395%5FLarge%2DScale%5FData%5FGeneration%5Ffor%5FRobotic%5FManipulation%2Emp4>`_.


.. image:: _static/synth_kitchen.png
   :width: 100%

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/install.rst
   getting_started/faq.rst

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   concepts/assets.rst
   concepts/scenes.rst
   concepts/procedural_assets.rst
   concepts/arranging_objects.rst
   concepts/articulations.rst
   concepts/exporting.rst

..
    Note: when adding new modules, please also update documented_modules.txt.

.. toctree::
    :maxdepth: 1
    :caption: API

    python_api/scene_synthesizer.scene
    python_api/scene_synthesizer.assets
    python_api/scene_synthesizer.procedural_assets
    python_api/scene_synthesizer.procedural_scenes
    python_api/scene_synthesizer.utils
    python_api/scene_synthesizer.exchange.export


.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/add_mdl_materials.rst
   examples/isaacsim.rst
   examples/isaaclab_rl.rst
   examples/isaaclab_teleop.rst


Projects Using Scene Synthesizer
================================

* `Motion Policy Networks <https://mpinets.github.io/>`__
* `CabiNet: Scaling Neural Collision Detection for Object Rearrangement with Procedural Scene Generation <https://cabinet-object-rearrangement.github.io/>`__
* `Imitating Task and Motion Planning with Visuomotor Transformers <https://mihdalal.github.io/optimus/>`__
* `M2T2: Multi-Task Masked-Transformer for Object-centric Pick and Place <https://m2-t2.github.io/>`__
* `RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics <https://robo-point.github.io/>`__
