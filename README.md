# stl_mapping
Verification of Space Robotics in Underwater Environment


# requirements:
Numpy, Scipy, Casadi, Gurobi and some basic stuff.


# Steps
1. Clone repo
2. `git submodule init`

# Steps for space
1. in `PX4-Autopilot`, run command `PX4_UXRCE_DDS_NS=snap make px4_sitl_spacecraft gz_atmos`
2. start microros with `micro-xrce-dds-agent udp4 -p 8888`
3. start QGC with `./startQGC`

# Steps for uw
1. in `PX4-Autopilot`, run command `PX4_UXRCE_DDS_NS=snap make px4_sitl_uuv gz_uuv_bluerov2_heavy`
2. start microros with `micro-xrce-dds-agent udp4 -p 8888`
3. start QGC with `./startQGC`

Flashing BlueROV px4: `PX4_UXRCE_DDS_NS=itrl_rov_1 make px4_fmu-v6x_uuv upload`