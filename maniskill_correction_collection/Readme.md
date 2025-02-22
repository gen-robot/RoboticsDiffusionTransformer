## Maniskill Correction Collection

#### Stack Cube

1. 60 steps -> Should grasp the cube
    - If success, then next step.
    - If fail, try use IK move the gripper above the cube. After, wait 32 steps to recheck first condition.

2. After condition1 success, another 64 steps.
    - If task success, then close
    - If is grasping, then wait 16 steps to recheck second condition.
    - If not grasping, try use IK move the gripper above the cube. After, wait 32 steps to recheck first condition.

Difficulties:
    - Position is easy, how about orientation?