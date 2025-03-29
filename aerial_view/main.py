import glob
import json
import base64
import numpy as np
import cv2
import os
from google.cloud import pubsub_v1


files = glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]

PROJECT_ID = "firm-container-448618-s5"
TOPIC_NAME = "shared_bus"
SUBSCRIPTION_NAME = "aerial-sub"

publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

def process_message(message):
    try:
        data = json.loads(message.data.decode("utf-8"))

        if data.get("stage") != "aerial_view":
            print(f"[INFO] Skipping message - wrong stage: {data.get('stage')}")
            message.ack()
            return

        Car2_location = np.array(data["Car2_Location"])
        Car1_dimensions = np.array(data["Car1_dimensions"])
        Car2_dimensions = np.array(data["Car2_dimensions"])
        vehicles_longitudinal = np.array(data["vehicles_longitudinal"])
        vehicles_lateral = np.array(data["vehicles_lateral"])
        Pedestrians_longitudinal = np.array(data["Pedestrians_longitudinal"])
        Pedestrians_lateral = np.array(data["Pedestrians_lateral"])
        Pedestrians = np.array(data["Pedestrians"])

        def length2pixel(length):  # length in meters
            return np.array([28.682 * length[1], 28.682 * length[0]])

        def coordinate2pixel(point):  # coordinate in meters
            return np.array([28.682 * point[1] - 3659.026, -28.682 * point[0] - 1278.32])

        car2_car1_distance = np.array([vehicles_longitudinal[0], vehicles_lateral[0]])
        car1_ped_distance = np.array([Pedestrians_longitudinal[0], Pedestrians_lateral[0]])

        Car1_location = Car2_location + car2_car1_distance
        Ped_location = Car1_location + car1_ped_distance

        ped_box = Pedestrians[0, :]
        x1, y1, x2, y2 = ped_box
        center_ped_camB = (1920 - x1 - x2) / 2
        ped_width_camB = x2 - x1
        ped_width_meter = car1_ped_distance[1] / center_ped_camB * ped_width_camB

        car1_location_px = coordinate2pixel(Car1_location)
        car2_location_px = coordinate2pixel(Car2_location)
        ped_location_px = coordinate2pixel(Ped_location)
        car1_dimensions_px = length2pixel(Car1_dimensions)
        car2_dimensions_px = length2pixel(Car2_dimensions)
        ped_dimensions_px = length2pixel(np.array([ped_width_meter, ped_width_meter]))

        img = cv2.imread("aerialView.png")

        def draw_object(label, center_px, dims_px, color):
            start = center_px - dims_px / 2
            end = center_px + dims_px / 2
            top_left = np.minimum(start, end).astype(int)
            bottom_right = np.maximum(start, end).astype(int)

            print(f"{label}, from {top_left} to {bottom_right}")
            cv2.rectangle(img, top_left, bottom_right, color, -1)
            cv2.putText(img, label, (top_left[0], max(top_left[1] - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        draw_object("Car 1", car2_location_px, car2_dimensions_px, (255, 0, 0))
        draw_object("Car 2", car1_location_px, car1_dimensions_px, (255, 255, 0))
        draw_object("Pedestrian", ped_location_px, ped_dimensions_px, (255, 0, 255))

        print(f"Output shape: {img.shape}")

        # Encode the final image
        _, buffer = cv2.imencode('.png', img)
        b64_img = base64.b64encode(buffer).decode('utf-8')

        output = {
            "Timestamp": data["Timestamp"],
            "aerialView": b64_img,
            "stage": "final"
        }

        publisher.publish(topic_path, json.dumps(output).encode("utf-8"))
        print("[INFO] Published to shared bus.")
        message.ack()

    except Exception as e:
        print(f"[ERROR] Exception in aerial view generation: {e}")
        message.ack()

print(f"<<<STAGE 7>>> [INFO] Listening for messages on {subscription_path}...")
streaming_pull_future = subscriber.subscribe(subscription_path, callback=process_message)

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()
