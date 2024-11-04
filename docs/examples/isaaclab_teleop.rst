####################################
Using Scenes in Isaac Lab for Teleop
####################################

Unless already done, create an environment in Isaac Lab as explained in the :doc:`RL in Isaac Lab example <isaaclab_rl>`.

The environment can then be used with a keyboard controlled robot:

..  code-block:: bash

    ./isaaclab.sh -p source/standalone/environments/teleoperation/teleop_se3_agent.py \
        --task Isaac-Open-Kitchen-Drawer-Franka-IK-Rel-v0 \
        --num_envs 1 \
        --teleop_device keyboard


.. figure:: /../imgs/examples/isaaclab_teleop.gif
    :align: center
    :scale: 100%

    Teleoperated robot in Isaac Lab, with a random kitchen generated with ``scene_synthesizer``