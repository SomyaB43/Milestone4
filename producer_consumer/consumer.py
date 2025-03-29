import glob
import os
import json
import base64
from google.cloud import pubsub_v1


files = glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]

PROJECT_ID = "firm-container-448618-s5"
SUBSCRIPTION_NAME = "shared_bus-sub"

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)

def callback(message):
    try:
        data = json.loads(message.data.decode("utf-8"))

        if data.get("stage") != "final":
            print(f"[INFO] Skipping message - wrong stage: {data.get('stage')}")
            message.ack()
            return
        
        print("[INFO] Image data received (not printing Base64).")

        # Save the aerial view image
        image_data = data.get("aerialView")
        if image_data:
            image_bytes = base64.b64decode(image_data)
            filename = f"aerial_view.png"
            with open(filename, "wb") as f:
                f.write(image_bytes)
            print(f"[INFO] Aerial view image saved as: {filename}")
        else:
            print("[WARNING] No image data found.")

        # Final JSON log without image
        final_output = {k: v for k, v in data.items() if k != "aerialView"}
        print("[INFO] Final output (without image):")
        print(json.dumps(final_output, indent=2))

        message.ack()

    except Exception as e:
        print(f"[ERROR] Exception in consumer: {e}")
        message.ack()

# Start listening
print(f"[INFO] Listening for messages on {subscription_path}...")
streaming_pull_future = subscriber.subscribe(subscription_path, callback)

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()
