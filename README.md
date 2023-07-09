# Sensor_Fusion_for_Indoor_Asset_3D_Location_Tracking

This project showcases a sensor-fusion system that utilizes multiple cameras and sound-localization to calculate and display the 3D global positions of tracked objects in a real-time and robust manner.
It leverages an Extended Kalman Filter that integrates the data from three Raspberry-Pi nodes transmitting location data via Computer Vision, on top of sound-localization. This provides a system that is resillient to occlusions, and background noise spikes, allowing for possible industrial applications.

Demonstration videos can be found at the root directory, where Anchor_1 and 2 showcase the perspective of two of the Pi Nodes and their CV-based localization, whereas the other two videos demonstrate the final output that plots the 3D position of the tracked objects on a grid that represents the global world coordinate system.