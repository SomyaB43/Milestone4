import glob
import torch
from torch import nn
import numpy as np
import json
import base64
import os
from google.cloud import pubsub_v1

files = glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]

PROJECT_ID = "firm-container-448618-s5"
TOPIC_NAME = "shared_bus"
SUBSCRIPTION_NAME = "long_lateral_cam2-sub"

publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

# Load MLP Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_max = np.array([1920, 1080, 1920, 1080, 20])
        self.y_max = torch.Tensor([10, 10])
        self.mlp = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Tanh()
        )

    def forward(self, x):
        inputs = torch.Tensor(x / self.x_max).float()
        with torch.no_grad():
            logits = self.mlp(inputs) * self.y_max
        return logits

mlp = NeuralNetwork()
mlp.load_state_dict(torch.load("mlp_camB.pkl", weights_only=True))
mlp.eval()

def process_message(message):
    print("[INFO] Received a new message.")

    try:
        input_data = json.loads(message.data.decode("utf-8"))

        if input_data.get("stage") != "vehicle_distance":
            print(f"[INFO] Skipping message - wrong stage: {input_data.get('stage')}")
            message.ack()
            return

        print(f"[INFO] Processing Timestamp: {input_data['Timestamp']}")
        print(f"[INFO] Car2_Location: {input_data['Car2_Location']}")
        print(f"[INFO] Car1_dimensions: {input_data['Car1_dimensions']}")
        print(f"[INFO] Car2_dimensions: {input_data['Car2_dimensions']}")
        print(f"[INFO] Image data received (not printing Base64).")

        vehicles = np.array(input_data["vehicles"])
        vehicles_depth = np.array(input_data["vehicles_depth"])

        outputs = []
        for i in range(vehicles.shape[0]):
            box = vehicles[i, :]
            depth = vehicles_depth[i]
            x = np.array([*box, depth])
            result = mlp(torch.Tensor(x))
            outputs.append(np.array(result))

        outputs = np.array(outputs)
        vehicles_longitudinal = outputs[:, 0].tolist()
        vehicles_lateral = outputs[:, 1].tolist()

        output_data = {
            "Timestamp": input_data["Timestamp"],
            "Car2_Location": input_data["Car2_Location"],
            "Car1_dimensions": input_data["Car1_dimensions"],
            "Car2_dimensions": input_data["Car2_dimensions"],
            "vehicles": input_data["vehicles"],
            "vehicles_longitudinal": vehicles_longitudinal,
            "vehicles_lateral": vehicles_lateral
        }


        publisher.publish(topic_path, json.dumps(output_data).encode("utf-8"))
        print("[INFO] Published to shared bus.")
        message.ack()

    except Exception as e:
        print(f"[ERROR] Exception processing message: {e}")
        message.ack()

# Start listening
print(f"<<<STAGE 6>>> [INFO] Listening for messages on {subscription_path}...")
streaming_pull_future = subscriber.subscribe(subscription_path, process_message)

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()
