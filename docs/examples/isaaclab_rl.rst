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

.. code-block:: python

    @configclass
    class CabinetSceneCfg(InteractiveSceneCfg):
        """Configuration for the kitchen scene with a robot and a kitchen.

        This is the abstract base implementation, the exact scene is defined in the derived classes
        which need to set the robot and end-effector frames
        """

        # robots, Will be populated by agent env cfg
        robot: ArticulationCfg = MISSING
        # End-effector, Will be populated by agent env cfg
        ee_frame: FrameTransformerCfg = MISSING

        kitchen = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen",
            # Make sure to set the correct path to the generated scene
            spawn=sim_utils.UsdFileCfg(usd_path="/tmp/kitchen_with_joint.usd"),
        )

        cabinet = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen/base_cabinet",
            # By doing spawn=None we're just registering the articulation, in this case one cabinet with a single joint
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                # Make sure that this is the exact transformation of the base_cabinet
                # ie. scene.get_transform('base_cabinet')
                pos=(1.0, 0, 0),
                rot=(0.70710678,  0.,  0., -0.70710678),
                joint_pos={
                    # Make sure that this is the correct joint name
                    # ie. scene.get_joint_names('base_cabinet')
                    "corpus_to_drawer_0_0": 0.0,
                },
            ),
            actuators={
                "drawers": ImplicitActuatorCfg(
                    # Make sure that this is the correct joint name
                    # ie. scene.get_joint_names('base_cabinet')
                    joint_names_expr=["corpus_to_drawer_0_0"],
                    effort_limit=87.0,
                    velocity_limit=100.0,
                    stiffness=10.0,
                    damping=1.0,
                ),
            },
        )

        # Frame definitions for the cabinet.
        cabinet_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen/base_cabinet/drawer_0_0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Kitchen/base_cabinet/drawer_0_0",
                    name="drawer_handle_top",
                    offset=OffsetCfg(
                        pos=(0.0, -0.05, 0.01),
                        # rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                    ),
                ),
            ],
        )

        # plane
        plane = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(),
            spawn=sim_utils.GroundPlaneCfg(),
            collision_group=-1,
        )

        # lights
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )

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
