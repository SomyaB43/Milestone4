import json
import time
import base64
import os
from google.cloud import pubsub_v1

PROJECT_ID = "firm-container-448618-s5"
TOPIC_NAME = "shared_bus"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

def encode_image(image_path):
    """Reads an image and encodes it to a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Warning: Image {image_path} not found!")
        return None

# Directly specify the image paths
occluded_image_path = "C:\Users\mbaba\Downloads\MS4\producer_consumer\A_017.png"
occluding_image_path = "C:\Users\mbaba\Downloads\MS4\producer_consumer\C_041.png"

# Encode images
occluded_image_b64 = encode_image(occluded_image_path)
occluding_image_b64 = encode_image(occluding_image_path)

if occluded_image_b64 is None or occluding_image_b64 is None:
    print("Skipping due to missing image.")
else:
    # Construct the message
    message_data = {
        "Timestamp": "1632763921",  # Example timestamp
        "Car2_Location": [12.34, 56.78],
        "Car1_dimensions": [5.0, 2.0],
        "Car2_dimensions": [5.5, 2.5],
        "Occluded_Image_View": occluded_image_b64,  # Base64 image
        "Occluding_Image_View": occluding_image_b64,  # Base64 image
    }

    # Prepare a clean version of the message for display (exclude image data)
    display_message = message_data.copy()
    display_message.pop("Occluded_Image_View", None)
    display_message.pop("Occluding_Image_View", None)

    # Publish the message
    future = publisher.publish(topic_path, json.dumps(message_data).encode("utf-8"))
    print(f"\n[INFO] Published message with ID: {future.result()}")
    print("[INFO] Message contents (excluding image data):")
    print(json.dumps(display_message, indent=2))

    time.sleep(2)  # Small delay to avoid overwhelming the system
