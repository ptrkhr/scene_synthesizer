
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