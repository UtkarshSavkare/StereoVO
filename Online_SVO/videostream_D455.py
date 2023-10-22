import os
import pyrealsense2 as rs
import numpy as np
import cv2



class RealsenseCamera:
  def __init__(self):
    # Initialize the RealSense pipeline

    self.pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1)
    config.enable_stream(rs.stream.infrared, 2)

    align_to = rs.stream.infrared
    self.align = rs.align(align_to)

    self.save_folder = ""
    bag_filename = os.path.join(self.save_folder, "test.bag")
    config.enable_record_to_file(bag_filename)

    # Start the pipeline
    pipeline_profile = self.pipeline.start(config)
    device = pipeline_profile.get_device()
    depth_sensor = device.query_sensors()[0]
    emitter = depth_sensor.get_option(rs.option.emitter_enabled)
    #print("emitter = ", emitter)
    set_emitter = 0
    depth_sensor.set_option(rs.option.emitter_enabled, set_emitter)
    emitter1 = depth_sensor.get_option(rs.option.emitter_enabled)
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    for sensor in pipeline_profile.get_device().sensors:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, True)
    #Set the desired exposure time in microseconds (example: 20000 microseconds)
    exposure_time = 100
    if depth_sensor.supports(rs.option.exposure):
            depth_sensor.set_option(rs.option.exposure, exposure_time)
   

    self.frame_counter = 0  # Initialize frame counter

    # Main loop
  def get_frame_stream(self):
    while True:
      # Wait for frames
      frames = self.pipeline.wait_for_frames(30000)
      aligned_frames = self.align.process(frames)

      # Get the left and right frames
      new_frame_left = aligned_frames.get_infrared_frame(1)
      new_frame_right  = aligned_frames.get_infrared_frame(2)

      # Convert frames to numpy arrays
      new_frame_left = np.asanyarray(new_frame_left.get_data())
      new_frame_right = np.asanyarray(new_frame_right.get_data())

      new_frame_left= cv2.equalizeHist(new_frame_left)
      new_frame_right = cv2.equalizeHist(new_frame_right)

      return True, new_frame_left, new_frame_right

  def release(self):         
    # Stop the pipeline and close windows
    self.pipeline.stop()
        
