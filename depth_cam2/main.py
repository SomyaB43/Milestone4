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

# Set the correct service account JSON key file
files = glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]

PROJECT_ID = "firm-container-448618-s5"
TOPIC_NAME = "shared_bus"
SUBSCRIPTION_NAME = "depth_cam2-sub"

subscriber = pubsub_v1.SubscriberClient()
publisher = pubsub_v1.PublisherClient()

subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

# Focal length for vehicle depth camera
f_px = 2500
threshold = 20  # Keep only vehicles within 20 meters

# Load model and transform once
model, transform = depth_pro.create_model_and_transforms()
model.eval()

def decode_image(base64_string):
    """Decode the base64 string to an image."""
    try:
        img_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(img_data))
    except Exception as e:
        print(f"[ERROR] Failed to decode image: {e}")
        return None

def process_message(message):
    """Callback function to process messages from Pub/Sub."""
    try:
        input_data = json.loads(message.data)

        if input_data.get("stage") != "vehicle_depth":
            message.ack()
            return

        vehicles = input_data.get("vehicles", [])
        occluding_image_b64 = input_data.get("Occluding_Image_View")

        if not vehicles or not occluding_image_b64:
            message.ack()
            return

        occluding_image = decode_image(occluding_image_b64)
        if not occluding_image:
            message.ack()
            return

        # Preprocess and infer depth map
        occluding_image = transform(occluding_image)
        prediction = model.infer(occluding_image, f_px=torch.Tensor([f_px]))
        depth_map = prediction["depth"].squeeze().cpu().numpy()

        filtered_vehicles = []
        vehicle_depths = []

        # Process each vehicle box and compute depth
        for box in vehicles:
            x1, y1, x2, y2 = map(int, box)
            y1, y2 = max(0, y1), min(depth_map.shape[0], y2)
            x1, x2 = max(0, x1), min(depth_map.shape[1], x2)

            depth_value = np.median(depth_map[y1:y2, x1:x2])
            if depth_value < threshold:
                filtered_vehicles.append(box)
                vehicle_depths.append(depth_value)

        # Prepare final message
        output_data = {
            "Timestamp": input_data["Timestamp"],
            "Car2_Location": input_data["Car2_Location"],
            "Car1_dimensions": input_data["Car1_dimensions"],
            "Car2_dimensions": input_data["Car2_dimensions"],
            "Pedestrians": input_data["Pedestrians"],
            "Pedestrians_longitudinal": input_data["Pedestrians_longitudinal"],
            "Pedestrians_lateral": input_data["Pedestrians_lateral"],
            "vehicles": filtered_vehicles,
            "vehicles_depth": [float(d) for d in vehicle_depths],
        }

        # Publish to shared bus
        publisher.publish(topic_path, json.dumps(output_data).encode("utf-8"))
        print("[INFO] Published to shared bus.")

    except Exception as e:
        print(f"[ERROR] Exception processing message: {e}")

    message.ack()

# Subscribe to the topic
streaming_pull_future = subscriber.subscribe(subscription_path, callback=process_message)
print(f"<<<STAGE 5>>> [INFO] Listening for messages on {subscription_path}...")

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()
