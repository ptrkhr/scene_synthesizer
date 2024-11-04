#############
Articulations
#############

Many procedural assets in ``scene_synthesizer`` are articulated, i.e., they contain revolute or prismatic joints that connect links.
File-based assets such as URDF, MJCF, or USDs also are able to represent articulations.

There are a number of methods that can give information about joints in the scene or set their properties:

..  code-block:: python

    from scene_synthesizer import procedural_scenes as ps

    kitchen = ps.kitchen_island()

    kitchen.is_articulated()
    # True

    kitchen.get_joint_names()
    # 
    # Notice that this only returns the joints
    # that are not fixed or floating joints
    # To include them, use the argument
    # include_fixed_floating_joints=True
    #
    # Note, that joint names are also - like
    # geometries - namespaced. The prefix indicates
    # which object the joint belongs to.
    #
    # ['base_cabinet/corpus_to_door_0_1',
    #  'base_cabinet/corpus_to_door_1_1',
    #  'base_cabinet/corpus_to_door_2_1',
    #  'base_cabinet/corpus_to_drawer_0_0',
    #  'dishwasher/corpus_to_door_0_1',
    #  'kitchen_island/corpus_to_door_0_1',
    #  'kitchen_island/corpus_to_door_1_1',
    #  'kitchen_island/corpus_to_drawer_0_0',
    #  'kitchen_island/corpus_to_drawer_1_0',
    #  'range/corpus_to_door_0_1',
    #  'range/corpus_to_drawer_0_2',
    #  'refrigerator/door_joint',
    #  'refrigerator/freezer_door_joint',
    #  'sink_cabinet/corpus_to_door_0_1',
    #  'sink_cabinet/corpus_to_door_1_1',
    #  'wall_cabinet/corpus_to_door_0_0',
    #  'wall_cabinet/corpus_to_door_1_0',
    #  'wall_cabinet_0/corpus_to_door_0_0',
    #  'wall_cabinet_0/corpus_to_door_1_0',
    #  'wall_cabinet_1/corpus_to_door_0_0',
    #  'wall_cabinet_1/corpus_to_door_1_0',
    #  'wall_cabinet_2/corpus_to_door_0_0',
    #  'wall_cabinet_3/corpus_to_door_0_0']

    kitchen.show_graph()
    #
    # Draws a scene graph in which edges are colored
    # according to the type of joint constraint (see below)

.. figure:: ../_static/articulated_graph_kitchen.png
   :align: center
   :width: 100%


..  code-block:: python

    kitchen.get_joint_properties('refrigerator/door_joint')
    #
    # Returns a dict where the key is the name of the joint
    # and the value is another dictionary with various 
    # properties such as the current configuration value,
    # type of joint, joint limits, and axis.
    #
    # {'refrigerator/door_joint':
    #    {'name': 'refrigerator/door_joint',
    #     'type': 'revolute',
    #     'q': 0.0,
    #     'axis': [-0.0, -0.0, -1.0],
    #     'origin': [[1.0, 0.0, 0.0, -0.42765515409850885],
    #                [0.0, 1.0, 0.0, -0.32041050696103557],
    #                [0.0, 0.0, 1.0, 0.0],
    #                [0.0, 0.0, 0.0, 1.0]],
    #     'limit_velocity': 100.0,
    #     'limit_effort': 100.0,
    #     'limit_lower': 0,
    #     'limit_upper': 3.141592653589793}}

    kitchen.get_configuration()
    #
    # Get the current configuration of the scene
    # which is a concatenation of the individual
    # object's configurations. It's also possible
    # to query specific joints or objects.
    #
    # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    # 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    # 0., 0.])

    kitchen.get_joint_limits()
    #
    # Get the joint limits of all joints
    # in the scene, i.e. a DOF x 2 matrix.
    # 
    # array([[0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 0.61394479],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 0.2754    ],
    #        [0.        , 0.2754    ],
    #        [0.        , 1.57079633],
    #        [0.        , 0.61394479],
    #        [0.        , 3.14159265],
    #        [0.        , 3.14159265],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633],
    #        [0.        , 1.57079633]])

    # Randomize all joint configurations within their limits
    kitchen.random_configurations()
    kitchen.colorize().show()

.. figure:: ../_static/kitchen_random_configurations.png
   :align: center
   :width: 80%

Note, that the method ``random_configurations()`` does not check for collisions.
In this example it can be seen that some open doors intersect with each other.
To circumvent this problem, one can simply sample until a collision-free configuration is found:

..  code-block:: python

    # set all joints back to their home configurations
    kitchen.zero_configurations()
    
    # iterate through all joints of the scene individually
    for j in kitchen.get_joint_names():
        # retrieve the joint's limits
        limits = kitchen.get_joint_limits(joint_ids=[j])[0]

        # sample a new joint configuratino until collision-free
        while True:
            kitchen.update_configuration(
                joint_ids=[j],
                configuration=[np.random.uniform(low=limits[0], high=limits[1])]
            )
            if not kitchen.in_collision(
                ignore_object_self_collisions=True,
                ignore_penetration_depth=0.005
                ):
                break
    
    kitchen.show()

.. figure:: ../_static/kitchen_random_configurations_without_collisions.png
   :align: center
   :width: 80%

Another important distinction to understand is the difference between a 'fixed', 'floating', and no joint.
They mostly matter when exporting the scene and using it in a physics simulation.
A 'floating' joint will create a `dynamic rigid body <https://nvidia-omniverse.github.io/PhysX/physx/5.4.2/_api_build/class_px_rigid_dynamic.html>`__ (of the subtree in the scene graph),
whereas no joint results in a `static rigid object <https://nvidia-omniverse.github.io/PhysX/physx/5.4.2/_api_build/class_px_rigid_static.html>`__ with infinite mass/inertia.
An articulated object needs to be connected either with a 'fixed' or 'floating' joint, depending on whether one wants to simulate an `articulation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.2/_api_build/class_px_articulation_reduced_coordinate.html>`__ with a fixed or floating base.
