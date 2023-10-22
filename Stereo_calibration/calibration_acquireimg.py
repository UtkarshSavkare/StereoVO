import os
import pyrealsense2 as rs
import numpy as np
import cv2

# Create directories for saving images
os.makedirs("left_images", exist_ok=True)
os.makedirs("right_images", exist_ok=True)



# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1)
config.enable_stream(rs.stream.infrared, 2)

# Start the pipeline
pipeline_profile = pipeline.start(config)
device = pipeline_profile.get_device()
depth_sensor = device.query_sensors()[0]
emitter = depth_sensor.get_option(rs.option.emitter_enabled)
print("emitter = ", emitter)
set_emitter = 0
depth_sensor.set_option(rs.option.emitter_enabled, set_emitter)
emitter1 = depth_sensor.get_option(rs.option.emitter_enabled)
print("new emitter = ", emitter1)

# depth_sensor = pipeline_profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

for sensor in pipeline_profile.get_device().sensors:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, True)
# Set the desired exposure time in microseconds (example: 20000 microseconds)
# exposure_time = 100
# if depth_sensor.supports(rs.option.exposure):
#         depth_sensor.set_option(rs.option.exposure, exposure_time)
# Image counter
image_counter = 0

# Flag to indicate if saving is triggered
save_triggered = False

# Main loop
while True:
    # Wait for frames
    frames = pipeline.wait_for_frames(10000)

    # Get the left and right frames
    left_frame = frames.get_infrared_frame(1)
    right_frame = frames.get_infrared_frame(2)

    # Convert frames to numpy arrays
    left_image = np.asanyarray(left_frame.get_data())
    right_image = np.asanyarray(right_frame.get_data())

    left_image = cv2.equalizeHist(left_image)
    right_image = cv2.equalizeHist(right_image)


    # Display the frames
    cv2.imshow("Left Camera", left_image)
    cv2.imshow("Right Camera", right_image)

    # Check if 's' is pressed to trigger saving
    key = cv2.waitKey(1)
    if key == ord('s'):
        save_triggered = True

    # Save the left image if saving is triggered
    if save_triggered:
        left_image_path = os.path.join("left_images", f"stereoL{image_counter}.png")
        cv2.imwrite(left_image_path, left_image)

        right_image_path = os.path.join("right_images", f"stereoR{image_counter}.png")
        cv2.imwrite(right_image_path, right_image)

        print(f"Images saved: {left_image_path}, {right_image_path}")

        # Increment the image counter
        image_counter += 1
        save_triggered = False

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Stop the pipeline and close windows
pipeline.stop()
cv2.destroyAllWindows()
