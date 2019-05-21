#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time
import yaml


def main():
    """
    Load setup
    """
    ymlfile = open('vehicle_config.yml')
    cfg = yaml.load(ymlfile)
    ymlfile.close()

    vehicle_mode = cfg['vehicle_mode']
    use_logger = cfg['use_logger']

    """
    Mode select
    """
    if vehicle_mode == 'car':
        from car.zmq_camera_pub import PubCamera
        from car.zmq_device_pubsub import PubSubDevice
    elif vehicle_mode == 'player':
        from player.zmq_unlogger import ZMQUnlogger

    if use_logger:
        from logger.zmq_logger import ZMQLogger

    """
    Set and Start instance
    """
    if vehicle_mode == 'car':
        pub_camera = PubCamera(cfg)
        pubsub_device = PubSubDevice(cfg)
    elif vehicle_mode == 'player':
        unlogger = ZMQUnlogger(cfg)

    if use_logger:
        logger = ZMQLogger(cfg)

    """
    While loop until keybord interrupt
    """
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    """
    End instance
    """
    if vehicle_mode == 'car':
        pub_camera.end()
        pubsub_device.end()
    elif vehicle_mode == 'player':
        unlogger.end()

    if use_logger:
        logger.end()


if __name__ == '__main__':
    main()
