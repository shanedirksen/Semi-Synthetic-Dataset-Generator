import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch.optim as optim
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
from torchvision.ops import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn as nn


def compute_iou(boxes1, boxes2):
    if len(boxes1.shape) < 2 or len(boxes2.shape) < 2:
        raise ValueError("Both boxes1 and boxes2 should be 2D tensors or arrays!")

    epsilon = 1e-7

    # Filter out invalid boxes
    valid_boxes1 = boxes1[(boxes1[..., 2] > boxes1[..., 0]) & (boxes1[..., 3] > boxes1[..., 1])]
    valid_boxes2 = boxes2[(boxes2[..., 2] > boxes2[..., 0]) & (boxes2[..., 3] > boxes2[..., 1])]

    if valid_boxes1.shape[0] == 0 or valid_boxes2.shape[0] == 0:
        # print("Invalid boxes")
        return 0.0

    # Expand dimensions for broadcasting
    valid_boxes1 = valid_boxes1[:, None, :]
    valid_boxes2 = valid_boxes2[None, :, :]

    # Calculate intersection
    x_left = torch.max(valid_boxes1[..., 0], valid_boxes2[..., 0])
    y_top = torch.max(valid_boxes1[..., 1], valid_boxes2[..., 1])
    x_right = torch.min(valid_boxes1[..., 2], valid_boxes2[..., 2])
    y_bottom = torch.min(valid_boxes1[..., 3], valid_boxes2[..., 3])

    intersection = (x_right - x_left).clamp(0) * (y_bottom - y_top).clamp(0)

    # If intersection is zero, IoU is zero
    if torch.all(intersection == 0):
        return 0.0

    # Calculate union
    area_boxes1 = (valid_boxes1[..., 2] - valid_boxes1[..., 0]) * (valid_boxes1[..., 3] - valid_boxes1[..., 1])
    area_boxes2 = (valid_boxes2[..., 2] - valid_boxes2[..., 0]) * (valid_boxes2[..., 3] - valid_boxes2[..., 1])
    union = area_boxes1 + area_boxes2 - intersection

    # Calculate IoU and regularize the union
    iou = intersection / (union + epsilon)

    # Return the maximum IoU value for each box in boxes1
    return iou.max(dim=1).values.mean().item()


def compute_mAP_for_image(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, num_classes, iou_threshold=0.5):
    """
    Compute mAP for an individual image.

    Parameters:
    - gt_boxes: Ground truth bounding boxes.
    - gt_labels: Ground truth labels.
    - pred_boxes: Predicted bounding boxes.
    - pred_labels: Predicted labels.
    - pred_scores: Confidence scores of the predictions.
    - num_classes: Total number of classes.
    - iou_threshold: IoU threshold to consider a prediction as a true positive.

    Returns:
    - mAP for the image.
    """

    # Initialize true positive and false positive lists
    tp = [0] * len(pred_boxes)
    fp = [0] * len(pred_boxes)

    # Sort predictions by descending confidence
    sorted_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    pred_labels = [pred_labels[i] for i in sorted_indices]

    used_gt_boxes = set()

    for i in range(len(pred_boxes)):
        max_iou = -1
        max_j = -1
        for j in range(len(gt_boxes)):
            if (j not in used_gt_boxes) and (gt_labels[j] == pred_labels[i]):
                iou = compute_iou(pred_boxes[i:i + 1], gt_boxes[j:j + 1])
                if iou > max_iou:
                    max_iou = iou
                    max_j = j

        if max_iou >= iou_threshold:
            tp[i] = 1
            used_gt_boxes.add(max_j)
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute mAP by integrating the precision-recall curve
    mAP = 0
    for i in range(1, len(precisions)):
        mAP += (recalls[i] - recalls[i - 1]) * precisions[i]

    return mAP


def visualize_sample(image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, class_names):
    """
    Visualize ground truth and predicted bounding boxes on a sample image.

    Parameters:
    - image: PIL Image object
    - gt_boxes: Ground truth boxes
    - gt_labels: Ground truth labels
    - pred_boxes: Predicted boxes
    - pred_scores: Predicted scores
    - pred_labels: Predicted labels
    - class_names: List of class names
    """

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Draw ground truth boxes
    for box, label in zip(gt_boxes, gt_labels):
        x, y, xmax, ymax = box
        rect = patches.Rectangle((x, y), xmax - x, ymax - y, linewidth=1, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, f"{class_names[label]} (True)", color='blue', bbox=dict(facecolor='white', alpha=0.5))

    # Draw predicted boxes
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        x, y, xmax, ymax = box
        rect = patches.Rectangle((x, y), xmax - x, ymax - y, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, ymax, f"{class_names[label]} ({score:.2f})", color='red', bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.show()


def show_random_image_with_boxes(dataset):
    # Randomly select an image index
    idx = random.randint(0, len(dataset) - 1)

    # Get the image and target (bounding boxes and labels)
    img, target = dataset[idx]

    # Convert the image tensor to a PIL Image
    img = transforms.ToPILImage()(img)

    # Create a figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    # Get the bounding boxes and labels from the target
    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()

    # Loop over the bounding boxes and labels and add them to the plot
    for box, label in zip(boxes, labels):
        # Create a Rectangle patch
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add label text
        plt.text(box[0], box[1] - 2, str(label), fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

    # Display the plot
    plt.show()


def collate_fn(batch):
    return tuple(zip(*batch))


def find_tensor_path(d, path=[]):
    if isinstance(d, torch.Tensor):
        print(f"Found tensor at path: {' -> '.join(path)}")
    elif isinstance(d, dict):
        for k, v in d.items():
            find_tensor_path(v, path + [str(k)])
    elif isinstance(d, list):
        for i, v in enumerate(d):
            find_tensor_path(v, path + [str(i)])


def create_coco_structure(class_names):
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name.strip()} for i, name in enumerate(class_names)]
    }


def add_annotations_to_coco(coco_struct, boxes, labels, scores, img_id, ann_id, is_pred=False):
    for box, label, score in zip(boxes, labels, scores if is_pred else [1] * len(boxes)):
        ann = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": label.item(),
            "bbox": box.tolist(),
            "area": ((box[2] - box[0]) * (box[3] - box[1])).item(),
            "iscrowd": 0
        }
        if is_pred:
            ann["score"] = score.item()
        coco_struct["annotations"].append(ann)
        ann_id += 1
    return ann_id


def evaluate(model, data_loader, device, label_path, class_file, current_epoch, max_epochs):
    model.to(device).eval()
    torch.cuda.empty_cache()

    with open(class_file, 'r') as f:
        class_names = f.readlines()

    gt_coco = create_coco_structure(class_names)
    pred_coco = create_coco_structure(class_names)

    all_targets, all_preds, all_scores = [], [], []
    total_predictions, sum_confidences, ann_id = 0, 0, 0

    for img_id, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = [
                {k: v[mask] for k, v in out.items()}
                for out in model(images)
                if (mask := out["scores"] > model.roi_heads.score_thresh).any()
            ]

        for i, (target, output) in enumerate(zip(targets, outputs)):
            image_info = {"id": img_id * len(images) + i, "width": images[i].shape[-1], "height": images[i].shape[-2]}
            gt_coco["images"].append(image_info)
            pred_coco["images"].append(image_info)

        for target, output in zip(targets, outputs):
            ann_id = add_annotations_to_coco(gt_coco, target["boxes"], target["labels"], None, img_id, ann_id)
            ann_id = add_annotations_to_coco(pred_coco, output["boxes"], output["labels"], output["scores"], img_id,
                                             ann_id, True)

            all_targets.extend(target["labels"].cpu().numpy())
            all_preds.extend(output["labels"].cpu().numpy())
            all_scores.extend(output["scores"].cpu().numpy())

            total_predictions += len(output["scores"])
            sum_confidences += output["scores"].sum().item()

        # print(
        #     f"Image {img_id}: {len(outputs[-1]['boxes']) if outputs else 0} predictions, {len(targets[-1]['boxes'])} ground truths")

    # Compute averages and metrics
    avg_predictions_per_image = total_predictions / len(data_loader.dataset)
    avg_confidence = sum_confidences / total_predictions if total_predictions > 0 else 0

    print(f"Average number of predictions per image: {avg_predictions_per_image:.2f}")
    print(f"Average confidence of predictions: {avg_confidence:.4f}")

    gt_file = os.path.join(label_path, "tmp_gt.json")
    pred_file = os.path.join(label_path, "tmp_pred.json")

    with open(gt_file, 'w') as f:
        json.dump(gt_coco, f)
    with open(pred_file, 'w') as f:
        json.dump(pred_coco["annotations"], f)

    # Check if the predictions file is empty or not
    if not os.path.exists(pred_file) or os.path.getsize(pred_file) == 0:
        print("No predictions found!")
        default_coco_stats = [-1] * 12  # Default values for COCO metrics
        default_APs = [-1] * len(class_names)  # Default values for APs
        default_mAP = -1  # Default value for mAP
        return default_coco_stats, default_APs, default_mAP, class_names

    if not pred_coco["annotations"]:
        print("No predictions made by the model.")
        return [-1] * 12, [-1] * len(class_names), -1, class_names

    coco_metrics = compute_coco_metrics(gt_file, pred_file)
    pascal_metrics = compute_pascal_metrics(all_targets, all_preds, all_scores, class_names)

    if current_epoch == max_epochs - 1:
        visualize_random_samples(model, data_loader, device, class_names)

    os.remove(gt_file)
    os.remove(pred_file)

    # Assuming compute_coco_metrics returns a single list/array and compute_pascal_metrics returns a tuple of (APs, mAP)
    coco_stats = coco_metrics  # Unpack the single-item tuple
    APs, mAP = pascal_metrics
    return coco_stats, APs, mAP, class_names


def compute_coco_metrics(gt_file, pred_file):
    coco_gt = COCO(gt_file)
    coco_pred = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    # coco_eval.params.iouThrs = [0.5]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def compute_pascal_metrics(all_targets, all_preds, all_scores, class_names):
    num_classes = len(class_names)
    num_images = len(all_targets)  # Assuming one target label per image
    y_true = np.zeros((num_images, num_classes))
    y_score = np.zeros((num_images, num_classes))

    for img_id, label in enumerate(all_targets):
        y_true[img_id, label - 1] = 1

    print("Number of images (all_targets):", len(all_targets))
    print("Number of predictions (all_preds):", len(all_preds))
    print("Number of scores (all_scores):", len(all_scores))

    for img_id, (label, score) in enumerate(zip(all_preds, all_scores)):
        if y_score[img_id, label - 1] < score:
            y_score[img_id, label - 1] = score

    APs = []
    for i in range(num_classes):
        AP = average_precision_score(y_true[:, i], y_score[:, i])
        APs.append(AP)

    mAP = np.mean(APs)
    return APs, mAP


def visualize_random_samples(model, data_loader, device, class_names):
    for _ in range(2):  # Repeat 5 times
        random_batch = random.choice(list(data_loader))
        images, targets = random_batch

        images = [img.to(device) for img in images]

        with torch.no_grad():
            outputs = model(images)

        sample_idx = random.randint(0, len(images) - 1)
        sample_image = images[sample_idx].permute(1, 2, 0).cpu().numpy()
        sample_gt_boxes = targets[sample_idx]['boxes'].cpu().numpy()
        sample_gt_labels = targets[sample_idx]['labels'].cpu().numpy()
        sample_pred_boxes = outputs[sample_idx]['boxes'].detach().cpu().numpy()
        sample_pred_scores = outputs[sample_idx]['scores'].detach().cpu().numpy()
        sample_pred_labels = outputs[sample_idx]['labels'].detach().cpu().numpy()

        visualize_sample(sample_image, sample_gt_boxes, sample_gt_labels, sample_pred_boxes, sample_pred_scores, sample_pred_labels, class_names)



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = {'loss': 0, 'iterations': 0, 'avg_iou': 0, 'iou_accumulator': 0}

    # Initialize the progress bar
    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch}")

    for iteration, (images, targets) in enumerate(progress_bar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger['loss'] += losses.item()
        metric_logger['iterations'] += 1

        # Compute average IoU for this batch and update metric_logger
        # model.eval()  # Set the model to evaluation mode
        # with torch.no_grad():
        #     outputs = model(images)
        #     ious = [compute_iou(output["boxes"], target["boxes"]) for output, target in zip(outputs, targets)]
        #     metric_logger['iou_accumulator'] += np.mean(ious)  # Accumulate IoU values
        #
        #     # Handle cases where the list of IoU values is empty
        #     if len(ious) == 0:
        #         avg_iou_for_iteration = 0
        #     else:
        #         avg_iou_for_iteration = np.mean(ious)
        #
        #     metric_logger['avg_iou'] += avg_iou_for_iteration
        #
        #     # Debugging: Print a warning if the average IoU for an iteration is NaN
        #     if np.isnan(avg_iou_for_iteration):
        #         print(f"Warning: Average IoU for iteration {iteration} is nan!")
        #         print(f"IoU values: {ious}")
        #
        # model.train()  # Set the model back to training mode

        # Update the progress bar description with the current loss and avg IoU
        avg_loss = metric_logger['loss'] / metric_logger['iterations']
        # avg_iou = metric_logger['iou_accumulator'] / (iteration + 1)  # Compute average IoU so far
        # progress_bar.set_description(f"Training Epoch {epoch}, Loss: {avg_loss:.4f}, Avg IoU: {avg_iou:.4f}")
        progress_bar.set_description(f"Training Epoch {epoch}, Loss: {avg_loss:.4f}")

    # Print the average IoU for the entire epoch
    # final_avg_iou = metric_logger['iou_accumulator'] / len(data_loader)
    # print(f"Average IoU for Epoch {epoch}: {final_avg_iou:.4f}")


# Step 1: Load your custom dataset
class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])
        img = Image.open(img_path).convert("RGB")

        # Get image dimensions
        width, height = img.size

        # Read the labels and convert them to the correct format
        with open(label_path) as f:
            labels = np.array([x.split() for x in f.read().strip().split("\n")], dtype=np.float32)

        boxes = torch.from_numpy(labels[:, 1:]).float()  # Get the bounding box coordinates
        labels = torch.from_numpy(labels[:, 0]).long()  # Get the labels

        # Convert relative coordinates (center_x, center_y, width, height) to absolute coordinates (x_min, y_min, x_max, y_max)
        boxes[:, 0] = boxes[:, 0] * width - (boxes[:, 2] * width / 2)
        boxes[:, 1] = boxes[:, 1] * height - (boxes[:, 3] * height / 2)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] * width
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] * height

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        # Apply transformations to the image and bounding boxes
        if self.transforms is not None:
            img, target = self.transforms(img, target)  # Modify this line to pass both img and target

        return img, target


        return img, target

    def __len__(self):
        return len(self.imgs)


def resize_img_and_target(img, target, new_width, new_height):
    # Get the original dimensions
    old_width, old_height = img.size

    # Calculate the scaling factors
    width_ratio = new_width / old_width
    height_ratio = new_height / old_height

    # Resize the image
    img = F.resize(img, (new_height, new_width))

    # Adjust the bounding boxes
    boxes = target["boxes"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * width_ratio
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * height_ratio
    target["boxes"] = boxes

    return img, target


def transform(img, target):
    # Resize the image and bounding boxes
    img, target = resize_img_and_target(img, target, 300, 300)
    img = T.ToTensor()(img)
    return img, target


def get_transform():
    return transform


def show_image_with_boxes(img, target):
    fig, ax = plt.subplots(1)
    ax.imshow(img.permute(1, 2, 0).numpy())  # Convert CxHxW to HxWxC

    boxes = target["boxes"].numpy()
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def main():
    # Step 2: Define the model
    with open('C:\\Users\\shane\\Documents\\Maya\\2022\\scripts\\dataset\\classes.txt', 'r') as f:
        num_classes = len(f.readlines()) + 1

    # Load your dataset with transformations
    dataset = CustomDataset(root='C:\\Users\\shane\\Documents\\Maya\\2022\\scripts\\dataset\\train',
                            transforms=get_transform())
    dataset_test = CustomDataset(root='C:\\Users\\shane\\Documents\\Datasets\\gmu-kitchens\\dataset',
                                 transforms=get_transform())

    # Visualize an example image with bounding boxes
    sample, target_sample = dataset[0]  # Get the first sample
    # show_image_with_boxes(sample, target_sample)

    # Introduce the boolean flag
    LIMIT_SAMPLES = True  # Set to True to limit samples, False to use the entire dataset

    if LIMIT_SAMPLES:
        # Limit the number of samples to 100 for both training and testing
        num_samples = 500
        train_indices = torch.randperm(len(dataset))[:num_samples].tolist()
        test_indices = torch.randperm(len(dataset_test))[:num_samples].tolist()
    else:
        # Use the entire dataset for training and testing
        train_indices = torch.randperm(len(dataset)).tolist()
        test_indices = torch.randperm(len(dataset_test)).tolist()

    # Use the subset of the dataset for training and validation
    dataset = torch.utils.data.Subset(dataset, train_indices)
    dataset_test = torch.utils.data.Subset(dataset_test, test_indices)

    # Define data loaders
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Load a pre-trained VGG-16 model
    vgg = models.vgg16(pretrained=True)

    # Convert the vgg.features to a list
    features_list = list(vgg.features)

    # Append dropout layer
    features_list.append(nn.Dropout(p=0.5))

    # Convert the list back to a nn.Sequential module
    new_features = nn.Sequential(*features_list)

    # Use the modified features as the backbone
    backbone = new_features
    backbone.out_channels = 512

    # Create an RPN (Region Proposal Network) that will use the output of the backbone to produce proposals
    rpn_anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    # Create the Faster R-CNN model using the VGG-16 backbone
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=rpn_anchor_generator)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Set the NMS threshold
    model.roi_heads.nms_thresh = 0.5  # Default is usually 0.5. Adjust as needed.

    # Set the confidence threshold
    model.roi_heads.score_thresh = 0.5  # Default is usually 0.05. Adjust as needed.
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to the right device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # And a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Let's train it for 10 epochs
    num_epochs = 1

    # Usage example
    dataset = CustomDataset(root='C:\\Users\\shane\\Documents\\Maya\\2022\\scripts\\dataset\\train',
                            transforms=get_transform())
    # show_random_image_with_boxes(dataset)

    weights_path = 'fasterrcnn_vgg16_fpn.pth'

    if os.path.exists(weights_path):  # Check if weights file exists
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path))  # Load the weights into the model
        # Evaluate directly
        coco_stats, APs, mAP, class_names = evaluate(model, data_loader_test, device,
                                                     'C:\\Users\\shane\\Documents\\Datasets\\gmu-kitchens\\dataset\\labels',
                                                     'C:\\Users\\shane\\Documents\\Maya\\2022\\scripts\\dataset\\classes.txt',
                                                     0, num_epochs)  # Pass the current epoch and total epochs here
        # Rest of your evaluation printing code...
    else:
        for epoch in range(num_epochs):
            # Train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # Update the learning rate
            lr_scheduler.step()

            # Save the trained model
            torch.save(model.state_dict(), 'fasterrcnn_vgg16_fpn.pth')

            # Evaluate on the test dataset
            coco_stats, APs, mAP, class_names = evaluate(model, data_loader_test, device,
                                                         'C:\\Users\\shane\\Documents\\Datasets\\gmu-kitchens\\dataset\\labels',
                                                         'C:\\Users\\shane\\Documents\\Maya\\2022\\scripts\\dataset\\classes.txt',
                                                         epoch,
                                                         num_epochs)  # Pass the current epoch and total epochs here

            # Print COCO metrics
            print("COCO METRICS:")
            metrics = ["AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "AR1", "AR10", "AR100", "ARsmall",
                       "ARmedium", "ARlarge"]
            for metric, value in zip(metrics, coco_stats):
                print(f"{metric}: {value:.4f}")

            # Print PASCAL metrics per class
            print("PASCAL METRIC (AP per class)")
            for class_name, AP in zip(class_names, APs):
                print(f"{class_name}: {AP:.4f}")

            # Print mAP
            print("PASCAL METRIC (mAP)")
            print(f"mAP: {mAP:.4f}")

    # Print COCO metrics
    print("COCO METRICS:")
    metrics = ["AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "AR1", "AR10", "AR100", "ARsmall",
               "ARmedium", "ARlarge"]
    for metric, value in zip(metrics, coco_stats):
        print(f"{metric}: {value:.4f}")

    # Print PASCAL metrics per class
    print("PASCAL METRIC (AP per class)")
    for class_name, AP in zip(class_names, APs):
        print(f"{class_name}: {AP:.4f}")

    # Print mAP
    print("PASCAL METRIC (mAP)")
    print(f"mAP: {mAP:.4f}")



if __name__ == '__main__':
    main()