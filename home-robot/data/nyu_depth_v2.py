import h5py
import numpy as np
import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset


class NYURobotDataset(Dataset):
    def __init__(self, processor, task="grounding"):
        # Get the directory where this script is located
        self.f = h5py.File(get_data_file_path(), "r")
        self.processor = processor
        self.task = task
        # NYU images are stored as (N, 3, 640, 480)
        self.images = self.f["images"]
        self.instances = self.f["instances"]
        self.labels = self.f["labels"]
        self.scene_types = self.f["sceneTypes"]

        # Extract class names
        names_ref = self.f["names"][0]
        self.class_names = [
            "".join(chr(c[0]) for c in self.f[ref]) for ref in names_ref
        ]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # 1. Image Processing
        img = np.transpose(self.images[idx], (2, 1, 0))  # To (480, 640, 3)
        pil_img = Image.fromarray(img)

        if self.task == "grounding":
            # Extract Bboxes from instance masks
            inst_mask = np.transpose(self.instances[idx], (1, 0))
            lbl_mask = np.transpose(self.labels[idx], (1, 0))
            unique_insts = np.unique(inst_mask)

            boxes = []
            class_ids = []
            for inst_id in unique_insts:
                if inst_id == 0:
                    continue
                y, x = np.where(inst_mask == inst_id)

                # Skip if no pixels found for this instance
                if len(x) == 0 or len(y) == 0:
                    continue

                # Get class ID and validate it
                cid = int(lbl_mask[y[0], x[0]]) - 1
                if cid < 0 or cid >= len(self.class_names):
                    continue  # Skip invalid class IDs

                # Calculate bounding box coordinates
                xmin, xmax = np.min(x), np.max(x)
                ymin, ymax = np.min(y), np.max(y)

                # Skip degenerate boxes (zero width or height)
                if xmin >= xmax or ymin >= ymax:
                    continue

                # Format: [xmin, ymin, xmax, ymax]
                boxes.append([xmin, ymin, xmax, ymax])
                class_ids.append(cid)

            # Handle case where no valid objects are found
            if len(boxes) == 0:
                # Create a dummy box to avoid empty tensor issues
                boxes = [[0.0, 0.0, 1.0, 1.0]]  # Full image as single box
                class_ids = [0]  # Use first class as dummy

            # Prepare inputs for DINO (requires a text prompt of classes)
            # Only include class names for objects present in this image to avoid token limit issues
            present_classes = list(set([self.class_names[cid] for cid in class_ids]))

            # Add a few additional common classes to help model generalization (within token limits)
            # Estimate ~10 tokens per class name, target max ~400 tokens to stay well under 512 limit
            max_additional_classes = max(0, (400 // 10) - len(present_classes))
            if max_additional_classes > 0:
                # Add some common classes that aren't already present
                other_classes = [name for i, name in enumerate(self.class_names)
                               if i not in class_ids]
                additional_classes = random.sample(other_classes,
                                                 min(max_additional_classes, len(other_classes)))
                present_classes.extend(additional_classes)

            text_prompt = ". ".join(present_classes) + "." if present_classes else "objects."
            inputs = self.processor(
                images=pil_img, text=text_prompt, return_tensors="pt"
            )

            # Normalize boxes for DINO [xmin, ymin, xmax, ymax] 0 to 1
            w, h = pil_img.size
            # Protect against division by zero
            if w <= 0:
                w = 1.0
            if h <= 0:
                h = 1.0

            # Convert to tensors and normalize
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            norm_tensor = torch.tensor([w, h, w, h], dtype=torch.float32)
            normalized_boxes = boxes_tensor / norm_tensor

            # Clamp to valid range [0, 1] to prevent any out-of-bounds values
            normalized_boxes = torch.clamp(normalized_boxes, 0.0, 1.0)

            # GroundingDino expects class_labels but only uses 0 (background) and 1 (object)
            # All detected objects should have label 1 since classification is done via text prompt
            class_labels_binary = torch.ones(len(boxes), dtype=torch.long)  # All objects = 1

            target = {
                "boxes": normalized_boxes,
                "class_labels": class_labels_binary,
            }
            return inputs, target

        else:  # VQA Task
            scene_ref = self.scene_types[0][idx]
            scene_name = "".join(chr(c[0]) for c in self.f[scene_ref])
            return pil_img, "What room is this?", f"This is a {scene_name}."


def get_data_file_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "nyu_depth_v2_labeled.mat")


class MockProcessor:
    """Mock processor for demonstration purposes"""

    def __call__(self, images, text, return_tensors="pt"):
        # Return a simplified structure that mimics a real processor
        # Note: In a real implementation, these parameters would be used to process the inputs
        _ = images, text, return_tensors  # Acknowledge unused parameters
        return {
            "pixel_values": torch.randn(1, 3, 224, 224),  # Mock image tensor
            "input_ids": torch.randint(0, 1000, (1, 10)),  # Mock text tokens
        }


def main():
    """Main method to demonstrate grounding and VQA tasks"""
    print("NYU Depth V2 Dataset Demo")
    print("=" * 50)

    # Check if the dataset file exists
    data_file_path = get_data_file_path()

    if not os.path.exists(data_file_path):
        print("Error: nyu_depth_v2_labeled.mat file not found!")
        print(f"Expected location: {data_file_path}")
        print(
            "Please ensure the NYU Depth V2 dataset file is in the same directory as this script."
        )
        return

    processor = MockProcessor()

    try:
        # 1. Demonstrate Grounding Task
        print("\n1. GROUNDING TASK DEMO")
        print("-" * 30)

        grounding_dataset = NYURobotDataset(processor, task="grounding")
        print(f"Dataset length: {len(grounding_dataset)}")
        print(f"Available classes: {len(grounding_dataset.class_names)}")
        print(f"First 10 class names: {grounding_dataset.class_names[:10]}")

        # Get one grounding sample
        inputs, target = grounding_dataset[0]
        print(f"\nGrounding sample 0:")
        print(f"  Input keys: {list(inputs.keys())}")
        print(f"  Input shapes: {[f'{k}: {v.shape}' for k, v in inputs.items()]}")
        print(f"  Number of detected objects: {len(target['boxes'])}")
        print(f"  Bounding boxes shape: {target['boxes'].shape}")
        print(f"  Class labels shape: {target['class_labels'].shape}")
        print(f"  Class labels (binary): {target['class_labels'].tolist()}")

        # Note: GroundingDino uses binary labels (0=background, 1=object) + text prompts
        print(f"  Detection method: Text-grounded with binary labels")

        # Show what the text prompt contains (this is how GroundingDino identifies objects)
        with torch.no_grad():  # Just for demo, get a fresh sample to see text prompt
            demo_inputs, _ = grounding_dataset[0]
            print(f"  Text input shape: {demo_inputs['input_ids'].shape}")

        # Validate tensor quality
        print(f"  Tensor validation:")
        print(f"    Boxes - NaN: {torch.isnan(target['boxes']).any()}, Inf: {torch.isinf(target['boxes']).any()}")
        print(f"    Labels - NaN: {torch.isnan(target['class_labels']).any()}, Inf: {torch.isinf(target['class_labels']).any()}")
        print(f"    Boxes range: [{torch.min(target['boxes']):.3f}, {torch.max(target['boxes']):.3f}]")
        print(f"    All boxes in [0,1]: {torch.all((target['boxes'] >= 0) & (target['boxes'] <= 1))}")
        print(f"    Labels range: [{torch.min(target['class_labels']):.0f}, {torch.max(target['class_labels']):.0f}]")
        print(f"    Target keys: {list(target.keys())}")

        # Show contrast with old method (but don't actually use it)
        old_style_prompt = ". ".join(grounding_dataset.class_names) + "."
        print(f"  OLD method would have: {len(old_style_prompt)} characters ({len(grounding_dataset.class_names)} classes)")

        # 2. Demonstrate VQA Task
        print("\n2. VQA TASK DEMO")
        print("-" * 30)

        vqa_dataset = NYURobotDataset(processor, task="vqa")
        print(f"Dataset length: {len(vqa_dataset)}")

        # Get one VQA sample
        image, question, answer = vqa_dataset[0]
        print(f"\nVQA sample 0:")
        print(f"  Image size: {image.size}")
        print(f"  Image mode: {image.mode}")
        print(f"  Question: '{question}'")
        print(f"  Answer: '{answer}'")

        # Try a few more VQA samples to show variety
        print(f"\nAdditional VQA samples:")
        for i in range(1, min(4, len(vqa_dataset))):
            _, question, answer = vqa_dataset[i]
            print(f"  Sample {i}: Q='{question}' A='{answer}'")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This might be due to missing dataset file or corrupted data.")


if __name__ == "__main__":
    main()
