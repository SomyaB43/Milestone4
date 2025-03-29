import glob
from PIL import Image
import depth_pro
import numpy as np
import torch
import json
import base64
import os
from io import BytesIO
from google.cloud import pubsub_v1

# Google Cloud Pub/Sub setup
files = glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]

PROJECT_ID = "firm-container-448618-s5"
TOPIC_NAME = "shared_bus"
SUBSCRIPTION_NAME = "depth_cam1-sub"

subscriber = pubsub_v1.SubscriberClient()
publisher = pubsub_v1.PublisherClient()

subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

# Model and threshold settings
f_px = 2200
threshold = 10  # Depth threshold in meters

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

def decode_image(base64_string):
    """Decodes a Base64 string to a PIL image for depth estimation."""
    try:
        img_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(img_data)) 
        return image
    except Exception as e:
        print(f"[ERROR] Failed to decode image: {e}")
        return None

def process_message(message):
    """Callback function to process received messages from Pub/Sub."""
    print(f"[INFO] Received a new message.")

    try:
        # Parse input JSON
        input_data = json.loads(message.data)

        # Check if this stage is pedestrian depth estimation
        if input_data.get("stage") != "pedestrian_depth":
            print(f"[INFO] Skipping message - wrong stage: {input_data.get('stage')}")
            message.ack()
            return

        # Extract relevant fields
        pedestrians = input_data.get("Pedestrians", [])
        occluded_image_b64 = input_data.get("Occluded_Image_View")
        occluding_image_b64 = input_data.get("Occluding_Image_View")  # pass it forward to next stage

        # If there are no detected pedestrians, skip processing
        if not pedestrians:
            print("[INFO] No pedestrians detected. Skipping processing.")
            message.ack()
            return

        # Decode image
        occluded_image = decode_image(occluded_image_b64)
        if occluded_image is None:
            print("[ERROR] Failed to load Occluding_Image_View.")
            message.ack()
            return

        print("[INFO] Successfully decoded image. Running depth estimation...")

        # Apply preprocessing transform for depth model
        occluded_image = transform(occluded_image)

        # Run depth estimation model
        prediction = model.infer(occluded_image, f_px=torch.Tensor([f_px]))
        depth_map = prediction["depth"].squeeze().cpu().numpy()

        # Process pedestrian bounding boxes
        filtered_pedestrians = []
        pedestrian_depths = []

        for box in pedestrians:
            x1, y1, x2, y2 = map(int, box)  # Convert to integers

            # Ensure bounding boxes are within image bounds
            y1, y2 = max(0, y1), min(depth_map.shape[0], y2)
            x1, x2 = max(0, x1), min(depth_map.shape[1], x2)

            depth_value = np.median(depth_map[y1:y2, x1:x2])  # Now this will work correctly
            print(f"[DEBUG] Pedestrian box: {box} | Estimated depth: {depth_value:.2f} meters")

            if depth_value < threshold:
                filtered_pedestrians.append(box)
                pedestrian_depths.append(depth_value)

        # Prepare output message
        output_data = {
            "Timestamp": input_data["Timestamp"],
            "Car2_Location": input_data["Car2_Location"],
            "Car1_dimensions": input_data["Car1_dimensions"],
            "Car2_dimensions": input_data["Car2_dimensions"],
            "Occluding_Image_View": occluding_image_b64,  # Preserve occluding image
            "Pedestrians": filtered_pedestrians,
            "Pedestrians_depth": [float(depth) for depth in pedestrian_depths]
        }

        # Publish updated message
        publisher.publish(topic_path, json.dumps(output_data).encode("utf-8"))
        print("[INFO] Published to shared bus.")

    except Exception as e:
        print(f"[ERROR] Exception processing message: {e}")

    message.ack()

# Subscribe to input Pub/Sub topic
print(f"<<<STAGE 2>>> [INFO] Listening for messages on {subscription_path}...")
streaming_pull_future = subscriber.subscribe(subscription_path, callback=process_message)

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()
