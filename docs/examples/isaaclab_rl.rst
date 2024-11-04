################################
Using Scenes in Isaac Lab for RL
################################

We show how to use a generated USD to be used in an RL workflow in `Isaac Lab <https://isaac-sim.github.io/IsaacLab/>`_.
We start by generating a scene and exporting it to a USD file:

.. code-block:: python

    import numpy as np
    import trimesh.transformations as tra

    from scene_synthesizer import utils
    from scene_synthesizer import procedural_scenes as ps

    # Generate a random kitchen with one joint
    #
    # Note: There is no reason to have just one joint in the kitchen,
    # other than in this particular example we will need to list all the joints
    # later in the Isaac Lab InteractiveScene.
    # An arbitrary number of joints is possible, but the code later in this example
    # would need to be adapted accordingly.
    #
    while True:
        k = ps.kitchen_single_wall()
        # Remove all joints except the ones pertaining to the base cabinet
        k.remove_joints('^(?!base_cabinet/).*')

        # check if exactly one joint is left
        if len(k.get_joint_names()) == 1:
            break

    # Move the origin of the scene (where the robot will be located)
    # to be one meter in front of the actuated base cabinet
    T = tra.translation_matrix((1.0, 0, 0)) \
        @ tra.euler_matrix(0, 0, -np.pi/2.0) \
        @ utils.homogeneous_inv(k.get_transform('base_cabinet'))
    k.add_base_frame_transform(T)

    # export to USD
    k.export('/tmp/kitchen_with_one_joint.usd')

Note, that we can now also :doc:`attach MDL materials as shown in a previous example <add_mdl_materials>` to make the scene a bit more visually pleasing.

Next, we create a `Manager-Based RL Environment <https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html>`_.
To do so we need to create an ``InteractiveScene`` and explicitly register all articulations and rigid bodies.
We base our example on the `Issac-open-Drawer-Franka-v0 <https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html>`_.
After copying the environment, make sure that the cabinet scene ``cabinet_env_cfg.py`` looks like this:

.. literalinclude:: /../examples/isaaclab_kitchen_env_cfg.py

Make sure to also replace the ``joint_names`` in the ``ObsTerm`` defined in ``ObservationsCfg``.

Also substitue the proper ``joint_names`` in the ``RewTerm`` representing the ``open_drawer_bonus`` and ``multi_stage_open_drawer``.

In ``rewards.py`` replace all occurences of

..  code-block:: python

    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]

    #
    # with
    #

    joint_index = asset_cfg.joint_ids[0] if not isinstance(asset_cfg.joint_ids, slice) else 0
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, joint_index]

to avoid a ``TypeError`` if the asset has just one joint (as in our case).

Finally, increase the spacing of the scenes in the scene settings (e.g. use ``env_spacing=5.0``).

The resulting environment can then be run with a random agent: 

..  code-block:: bash
    
    python source/standalone/environments/random_agent.py \
        --task Isaac-Open-Drawer-Franka-v0 \
        --num_envs 100 \

.. figure:: /../imgs/examples/isaaclab_rl_random.gif
    :align: center
    :scale: 100%

Check the `Isaac Lab documentation <https://isaac-sim.github.io/IsaacLab/main/index.html>`_ for further details.
