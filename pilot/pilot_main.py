#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
import sys
import time
import yaml

from common.pilot_gloval_variable import MPVariable
from vehicle_bridge.zmq_device_camera_sub import SubDeviceCamera
from vehicle_bridge.zmq_device_pub import PubDevice
from perception.mp_perception import MPPerception
from planning.mp_planning import MPPlanning
from visualization.mp_visualization import MPVisualization
from visualization.mp_birdseye import MPBirdsEye
from visualization.mp_chart import MPChart

def main():
    """
    Load setup
    """
    ymlfile = open('../vehicle/vehicle_config.yml')
    vehicle_cfg = yaml.load(ymlfile)
    ymlfile.close()

    ymlfile = open('pilot_config.yml')
    pilot_cfg = yaml.load(ymlfile)
    ymlfile.close()

    """
    Set and Start instance
    """
    receiver = SubDeviceCamera(vehicle_cfg)
    sender = PubDevice(vehicle_cfg)
    planning = MPPlanning(pilot_cfg)
    visualization = MPVisualization(pilot_cfg)
    birdseye = MPBirdsEye(pilot_cfg)
    if pilot_cfg['use_chart']:
        chart = MPChart(pilot_cfg)

    """
    Start perception
    """
    perception = MPPerception(pilot_cfg)
    perception.start()    # while loop
    perception.end()

    """
    End instance
    """
    receiver.end()
    sender.end()
    planning.end()
    visualization.end()
    birdseye.end()
    if pilot_cfg['use_chart']:
        chart.end()


if __name__ == '__main__':
    main()
