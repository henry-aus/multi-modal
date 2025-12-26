import torch
import os
import json
from pathlib import Path
from transformers import (
    GroundingDinoProcessor,
    GroundingDinoForObjectDetection,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)
import logging
from device_utils import detect_optimal_device


class HomeRobotSystem:
    def __init__(self, dino_path=None, blip_path=None, device=None):
        """
        Initialize the Home Robot System.

        Args:
            dino_path (str, optional): Path to the GroundingDino model.
            blip_path (str, optional): Path to the BLIP2 model.
            device (str, optional): Device to use. If None, auto-detects optimal device.
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = detect_optimal_device()
        else:
            self.device = device

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info(f"Initializing HomeRobotSystem on {self.device}")

        # Use default pretrained paths if not specified
        if dino_path is None:
            dino_path = "./pretrained/robot_dino_final"
        if blip_path is None:
            blip_path = "./pretrained/robot_blip_lora"

        logger.info(f"Loading GroundingDino from: {dino_path}")
        logger.info(f"Loading BLIP2 from: {blip_path}")

        # Verify paths exist
        self._verify_model_paths(dino_path, blip_path)
        # Load models and processors
        try:
            # For GroundingDino, we still use the base processor but load our fine-tuned model
            self.dino_proc = GroundingDinoProcessor.from_pretrained(dino_path)
            self.dino_model = GroundingDinoForObjectDetection.from_pretrained(
                dino_path
            ).to(self.device)
            logger.info("✓ GroundingDino model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load GroundingDino from {dino_path}: {e}")
            logger.info("Falling back to base GroundingDino model...")
            self.dino_proc = GroundingDinoProcessor.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            )
            self.dino_model = GroundingDinoForObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            ).to(self.device)

        try:
            # Load our fine-tuned BLIP2 model with LoRA
            self.blip_proc = Blip2Processor.from_pretrained(blip_path)
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                blip_path
            ).to(self.device)
            logger.info("✓ BLIP2 model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load BLIP2 from {blip_path}: {e}")
            logger.info("Falling back to base BLIP2 model...")
            self.blip_proc = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b"
            ).to(self.device)

    def perceive_and_count(self, image, target_object):
        """Grounding and Counting Task"""
        inputs = self.dino_proc(
            images=image, text=f"{target_object}.", return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        # Post-process to get boxes
        results = self.dino_proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.35,
            target_sizes=[image.size[::-1]],
        )
        count = len(results[0]["boxes"])
        return results[0]["boxes"], count

    def ask_robot(self, image, question):
        """VQA and Environment Task"""
        inputs = self.blip_proc(
            images=image, text=f"Question: {question} Answer:", return_tensors="pt"
        ).to(self.device)
        generated_ids = self.blip_model.generate(**inputs, max_new_tokens=50)
        return self.blip_proc.decode(generated_ids[0], skip_special_tokens=True)

    def _verify_model_paths(self, dino_path, blip_path):
        """Verify that model paths exist and provide helpful error messages"""
        if not os.path.exists(dino_path):
            logger = logging.getLogger(__name__)
            logger.warning(f"GroundingDino model path does not exist: {dino_path}")
            logger.info("Available pretrained models:")
            self.list_available_models()

        if not os.path.exists(blip_path):
            logger = logging.getLogger(__name__)
            logger.warning(f"BLIP2 model path does not exist: {blip_path}")
            logger.info("Available pretrained models:")
            self.list_available_models()

    @staticmethod
    def list_available_models():
        """List all available pretrained models"""
        pretrained_dir = Path("./pretrained")
        if not pretrained_dir.exists():
            print("No pretrained directory found. Train models first using:")
            print("  python train/dino.py --data_path <dataset_path>")
            print("  python train/bclip.py --data_path <dataset_path>")
            return

        models = []
        for model_dir in pretrained_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "model_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        models.append(
                            {
                                "name": model_dir.name,
                                "path": str(model_dir),
                                "saved_at": metadata.get("saved_at", "Unknown"),
                                "training_summary": metadata.get(
                                    "training_summary", {}
                                ),
                            }
                        )
                    except Exception as e:
                        models.append(
                            {
                                "name": model_dir.name,
                                "path": str(model_dir),
                                "saved_at": "Unknown",
                                "error": str(e),
                            }
                        )

        if models:
            print("\nAvailable pretrained models:")
            print("-" * 60)
            for model in models:
                print(f"Model: {model['name']}")
                print(f"Path: {model['path']}")
                print(f"Saved: {model['saved_at']}")
                if "training_summary" in model and model["training_summary"]:
                    summary = model["training_summary"]
                    if "best_val_loss" in summary:
                        print(f"Best validation loss: {summary['best_val_loss']:.6f}")
                    if "total_epochs" in summary:
                        print(f"Training epochs: {summary['total_epochs']}")
                print("-" * 60)
        else:
            print("No pretrained models found in ./pretrained/")

    @staticmethod
    def get_model_info(model_name):
        """Get detailed information about a specific pretrained model"""
        model_path = Path(f"./pretrained/{model_name}")
        if not model_path.exists():
            print(f"Model {model_name} not found in ./pretrained/")
            HomeRobotSystem.list_available_models()
            return None

        metadata_file = model_path / "model_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading metadata for {model_name}: {e}")

        return None


# --- Robot Deployment Examples ---

if __name__ == "__main__":
    # Example 1: Use default pretrained models (recommended)
    # Loads from ./pretrained/robot_dino_final and ./pretrained/robot_blip_lora
    robot = HomeRobotSystem()

    # Example 2: Use custom model paths
    # robot = HomeRobotSystem(
    #     dino_path="./custom/my_dino_model",
    #     blip_path="./custom/my_blip_model"
    # )

    # Example 3: Specify device manually
    # robot = HomeRobotSystem(device="cuda")

    # List available pretrained models
    print("Available models:")
    HomeRobotSystem.list_available_models()

    # Usage examples:
    # boxes, count = robot.perceive_and_count(camera_frame, "white mug")
    # room_type = robot.ask_robot(camera_frame, "What room am I in?")

    print(f"\nRobot initialized on device: {robot.device}")
    print("Ready for inference!")
