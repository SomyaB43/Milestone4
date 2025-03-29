import glob
import torch
from torch import nn
import numpy as np
import json
import os
from google.cloud import pubsub_v1

# Set up environment for Google Cloud Pub/Sub
files = glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]

PROJECT_ID = "firm-container-448618-s5"
TOPIC_NAME = "shared_bus"
SUBSCRIPTION_NAME = "long_lateral_cam1-sub"

publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

# Load MLP Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_max = np.array([1920, 1080, 1920, 1080, 20])  # Max values for normalization
        self.y_max = torch.Tensor([10, 10])  # Scaling for output
        self.mlp = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Tanh()
        )

    def forward(self, x):
        # Normalize inputs and run through the MLP
        inputs = torch.Tensor(x / self.x_max).float()
        with torch.no_grad():
            logits = self.mlp(inputs) * self.y_max
        return logits

mlp = NeuralNetwork()
mlp.load_state_dict(torch.load("mlp_camA.pkl", weights_only=True))
mlp.eval()

# Function to process messages from Pub/Sub
def process_message(message):
    try:
        input_data = json.loads(message.data.decode("utf-8"))

        # Skip if not the correct stage
        if input_data.get("stage") != "pedestrian_distance":
            message.ack()
            return

        # Extract pedestrian data
        pedestrians = np.array(input_data["Pedestrians"])
        pedestrian_depths = np.array(input_data["Pedestrians_depth"])

        # Process each pedestrian through the MLP model
        outputs = []
        for i in range(pedestrians.shape[0]):
            box = pedestrians[i, :]
            depth = pedestrian_depths[i]
            x = np.array([*box, depth])
            outputs.append(np.array(mlp(torch.Tensor(x))))

        outputs = np.array(outputs)
        pedestrians_longitudinal = outputs[:, 0].tolist()
        pedestrians_lateral = outputs[:, 1].tolist()

        # Prepare the output message
        output_data = {
            "Timestamp": input_data["Timestamp"],
            "Car2_Location": input_data["Car2_Location"],
            "Car1_dimensions": input_data["Car1_dimensions"],
            "Car2_dimensions": input_data["Car2_dimensions"],
            "Pedestrians": input_data["Pedestrians"],
            "Pedestrians_longitudinal": pedestrians_longitudinal,
            "Pedestrians_lateral": pedestrians_lateral
        }

        # Publish the message to shared bus
        publisher.publish(topic_path, json.dumps(output_data).encode("utf-8"))
        message.ack()

    except Exception as e:
        print(f"[ERROR] Exception processing message: {e}")
        message.ack()

# Listen for incoming messages
print(f"Listening for messages on {subscription_path}...")
streaming_pull_future = subscriber.subscribe(subscription_path, process_message)

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()
