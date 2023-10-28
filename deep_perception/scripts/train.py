import os
import argparse
import torch
from ultralytics import YOLO
import wandb

#export WANDB_API_KEY=b89325a9eeb8eca9f9c9f4fec3ddb340aef5600a

from wandb.integration.ultralytics import add_wandb_callback

def load_model(model_name):
    model = YOLO(model_name)
    model.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    return model
    
def train_model(model_name, config_file, epochs):
    model = load_model(model_name)
    add_wandb_callback(model, enable_model_checkpointing=True)
    model.train(data=config_file, epochs=epochs, project=model_name.split('.')[0])
    model.val()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", '-m', type=str, default='yolov8m.pt', required=False)
    parser.add_argument("--config_file", '-cfg', type=str, required=True)
    parser.add_argument("--epochs", '-e', type=int, required=True)

    args = parser.parse_args()
    
    wandb.login()
    train_model(args.model_name, args.config_file, args.epochs)

