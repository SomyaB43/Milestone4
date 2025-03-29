import glob
from ultralytics import YOLO
import cv2
import numpy as np
import json
import base64
import os
from io import BytesIO
from PIL import Image
from google.cloud import pubsub_v1

# Set Google Cloud credentials
files = glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]

PROJECT_ID = "firm-container-448618-s5"
TOPIC_NAME = "shared_bus"
SUBSCRIPTION_NAME = "yolo_car-sub"

subscriber = pubsub_v1.SubscriberClient()
publisher = pubsub_v1.PublisherClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

# Load YOLO model
model = YOLO("./yolo11n.pt")

def decode_image(base64_string):
    """Decodes a base64 string to an OpenCV image."""
    try:
        img_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image if image is not None else None
    except Exception as e:
        print(f"[ERROR] Decoding image failed: {e}")
        return None

def encode_image(image):
    """Encodes an OpenCV image to a base64 string."""
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        image_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Encoding image failed: {e}")
        return None

def process_message(message):
    """Process incoming Pub/Sub messages."""
    try:
        input_data = json.loads(message.data)

        if input_data.get("stage") != "vehicle_detection":
            message.ack()
            return

        # Decode the image
        Occluding_Image_View = decode_image(input_data.get("Occluding_Image_View"))
        if Occluding_Image_View is None:
            message.ack()
            return

        # Run YOLO model on the image
        results = model.predict(source=Occluding_Image_View)
        car_boxes = [
            [int(round(coord)) for coord in result.boxes.xyxy.cpu().numpy()[0][:4]]
            for result in results
            if result.names[int(result.boxes.cls.cpu().numpy()[0])] == 'car'
        ]

        # Re-encode image for output
        Occluding_Image_View_b64 = encode_image(Occluding_Image_View)
        if Occluding_Image_View_b64 is None:
            message.ack()
            return

        # Prepare output data
        output_data = {
            "Timestamp": input_data["Timestamp"],
            "Car2_Location": input_data["Car2_Location"],
            "Car1_dimensions": input_data["Car1_dimensions"],
            "Car2_dimensions": input_data["Car2_dimensions"],
            "Occluding_Image_View": Occluding_Image_View_b64,
            "vehicles": car_boxes,
        }

        # Log the output before publishing (without images)
        print("[INFO] Final output before publishing (without images):")
        output_log = output_data.copy()
        output_log.pop("Occluding_Image_View", None)
        print(json.dumps(output_log, indent=2))

        # Publish the message
        publisher.publish(topic_path, json.dumps(output_data).encode("utf-8"))
        message.ack()

    except Exception as e:
        print(f"[ERROR] Exception processing message: {e}")
        message.ack()

# Subscribe to Pub/Sub and start processing messages
streaming_pull_future = subscriber.subscribe(subscription_path, callback=process_message)
print(f"Listening for messages on {subscription_path}...")

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()
