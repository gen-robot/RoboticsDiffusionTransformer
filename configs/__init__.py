from third_party.rdt.configs.state_vec import STATE_VEC_IDX_MAPPING

# Corresponding to right arm
# WIDOWX in BridgeData v2
WIDOWX_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"gripper_joint_0_pos"] for i in range(1)
]

WIDOWX_EEF_INDICES = [
    STATE_VEC_IDX_MAPPING[f"eef_pos_{i}"] for i in ['x', 'y', 'z']
] + [
    STATE_VEC_IDX_MAPPING[f"eef_angle_{i}"] for i in range(6)
]

WIDOWX_QVEL_INDICES = [
    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_vel"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"gripper_joint_{i}_vel"] for i in range(1)
]


GOOGLE_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["gripper_open"]
]

GOOGLE_EEF_INDICES = [
    STATE_VEC_IDX_MAPPING[f"eef_pos_{i}"] for i in ['x', 'y', 'z']
] + [
    STATE_VEC_IDX_MAPPING[f"eef_angle_{i}"] for i in range(6)
]

GOOGLE_QVEL_INDICES = [
    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_vel"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["gripper_open_vel"]
]

# The indices that the raw vector should be mapped to in the unified action vector
AGILEX_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["left_gripper_open"]
] + [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
]

AGILEX_EEF_INDICES = [
    STATE_VEC_IDX_MAPPING[f"left_eef_pos_{i}"] for i in ['x', 'y', 'z']
] + [
    STATE_VEC_IDX_MAPPING[f"left_eef_angle_{i}"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"right_eef_pos_{i}"] for i in ['x', 'y', 'z']
] + [
    STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)
]

AGILEX_QVEL_INDICES = [
    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_vel"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["left_gripper_open_vel"]
] + [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_vel"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["right_gripper_open_vel"]
]

ROBOT_INDICES = {
    "mobile_aloha": {
        "state_indices": AGILEX_STATE_INDICES,
        "eef_indices": AGILEX_EEF_INDICES,
        "qvel_indices": AGILEX_QVEL_INDICES,
    },

    "mobile_aloha_v2": {
        "state_indices": AGILEX_STATE_INDICES,
        "eef_indices": AGILEX_EEF_INDICES,
        "qvel_indices": AGILEX_QVEL_INDICES,
    },

    "widowx_bridge": {
        "state_indices": WIDOWX_STATE_INDICES,
        "eef_indices": WIDOWX_EEF_INDICES,
        "qvel_indices": WIDOWX_QVEL_INDICES,
    },

    "google_robot": {
        "state_indices": GOOGLE_STATE_INDICES,
        "eef_indices": GOOGLE_EEF_INDICES,
        "qvel_indices": GOOGLE_QVEL_INDICES,
    },
}

ROBOT_CAMERA_NAMES = {
    "mobile_aloha": [
        "cam_high", 
        "cam_right_wrist",
        "cam_left_wrist",
    ],

    "mobile_aloha_v2": [
        "cam_high", 
        "cam_right_wrist",
        "cam_left_wrist",
    ],

    "widowx_bridge": [
        "3rd_view_camera", 
        "background",
        "background",
    ],

    "google_robot": [
        "overhead_camera", 
        "background",
        "background",
    ],
}
