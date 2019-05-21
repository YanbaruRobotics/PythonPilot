
import math
import numpy as np


def pure_pursuit(lane_m_a, lane_m_b, lane_m_c, vehicle_speed_kmph, \
                 pp_gain=0.4, \
                 wheelbase=2.5, \
                 target_angle_gain=0.4, \
                 lane_c_gain=0.5, \
                 lr_calibration_gain=1.0, \
                 steering_angle_minmax_abs=60.0):

    # calc target position
    z = pp_gain * vehicle_speed_kmph
    z = np.clip(z, 10.0, 40.0)
    x = lane_m_a * (z ** 2) + lane_m_b * z + lane_m_c

    # calc target radius
    L = math.sqrt(x*x + z*z)
    if (x != 0.0):
        target_radius = L*L / (2*x)
    else:
        target_radius = 0.0

    # calc target wheel steering angle
    if (target_radius > 0.0):
        target_wheel_steering_angle = (math.atan2(wheelbase, target_radius) / math.pi * 180.0)
    elif (target_radius < 0.0):
        target_wheel_steering_angle = (math.atan2(wheelbase, target_radius) / math.pi * 180.0) - 180.0
    else:
        target_wheel_steering_angle = 0.0

    # calc target angle
    # SteeringAngleRange/WheelSteeringAngleRange * gain
    target_steering_angle = target_wheel_steering_angle * 1440.0 / 40.0 * target_angle_gain

    # lane c FB
    target_steering_angle += lane_m_c * lane_c_gain

    # steering LR calibration
    if (target_steering_angle > 0):
        target_steering_angle = target_steering_angle * lr_calibration_gain

    # target angle limit
    target_steering_angle = np.clip(target_steering_angle, -steering_angle_minmax_abs, steering_angle_minmax_abs)

    return z, target_steering_angle


if __name__ == '__main__':
    print(pure_pursuit(0.001, 0.0, -0.385, 100))
    print(pure_pursuit(-0.0003, 0.0, -0.12, 100))
