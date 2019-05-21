#!/bin/bash	

sudo modprobe can
sudo modprobe vcan
sudo modprobe slcan
ls /dev/ttyACM*
sudo slcand -o -c -s5 /dev/ttyACM_canable can3
sudo ifconfig can3 up
sudo ifconfig can3 down
sudo ifconfig can3 txqueuelen 10000000
sudo ifconfig can3 up
ifconfig can3
# canbusload can3@250000 -r -t -b -c
