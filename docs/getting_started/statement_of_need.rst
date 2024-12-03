#################
Statement of Need
#################

Simulation is an ever increasing data source for training deep learning models.
In robotics, simulations have been successfully used to learn behaviors such as navigation, walking, flying or manipulation.
The value of data generation in simulation mainly depends on the diversity and scale of scene layouts.
Existing datasets :cite:`Mo_2019_CVPR,robotix2019,ehsani2021manipulathor,robocasa2024` are limited in that regard, whereas purely generative models still lack the ability to create scenes that can be used in physics simulator :cite:`schult24controlroom3d,yang2024physcenephysicallyinteractable3d,hoellein2023text2room`.
Other procedural pipelines either focus on learning visual models :cite:`Denninger2023,greff2021kubric,infinigen2023infinite`, address specific use-cases such as autonomous driving :cite:`scenic2020,hess2021procedural`, or make it hard to be extended and customized since they are tightly integrated with a particular simulation platform :cite:`procthor`.
With `scene_synthesizer` we present a library that simplifies the process of writing scene randomizers in Python, with a particular focus on physics simulations for robot manipulation. It is fully simulator-agnostic.

.. bibliography::
