import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """

    # Compute intersection
    x1 = max(prediction_box[0], gt_box[0])
    y1 = max(prediction_box[1], gt_box[1])
    x2 = min(prediction_box[2], gt_box[2])
    y2 = min(prediction_box[3], gt_box[3])
    intersection_box = [x1, y1, x2, y2]
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute union
    predicted_box_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = predicted_box_area + gt_box_area - intersection

    iou = intersection / union

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """

    if num_tp + num_fp == 0:
        return 1
    else:
        return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """

    if num_tp + num_fn == 0:
        return 0
    else:
        return num_tp / (num_tp + num_fn)

def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    prediction_boxes_matched, gt_boxes_matched = [], []

    for i in range(prediction_boxes.shape[0]):

        # Find all possible matches with a IoU >= iou threshold
        candidates = []
        ious = []
        for j in range(gt_boxes.shape[0]):
            iou = calculate_iou(prediction_boxes[i], gt_boxes[j])
            if iou >= iou_threshold:
                candidates.append(j)
                ious.append(iou)

        if len(candidates) == 0: continue

        # Sort all matches on IoU in descending order
        candidates = np.array(candidates)
        candidates = candidates[np.argsort(ious)[::-1]]     # ix of matches in descending order of IoU

        # Find match with the highest IoU
        prediction_boxes_matched.append(prediction_boxes[i])
        gt_boxes_matched.append(gt_boxes[candidates[0]])
        gt_boxes = np.delete(gt_boxes, candidates[0], axis=0)       # remove the match from the gt_boxes array! (single matches)

    return np.array(prediction_boxes_matched), np.array(gt_boxes_matched)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    # take boxes and match them
    prediction_boxes_matched, gt_boxes_matched = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    # Calculate number of TP, FP and FN
    true_pos = prediction_boxes_matched.shape[0]            # TP = when iou of predicted box is above threshold!
    false_pos = prediction_boxes.shape[0] - true_pos
    false_neg = gt_boxes.shape[0] - true_pos

    return {"true_pos": true_pos, "false_pos": false_pos, "false_neg": false_neg}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    num_tp, num_fp, num_fn = 0, 0, 0
    for i in range(len(all_prediction_boxes)):
        result_dict = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        num_tp += result_dict["true_pos"]
        num_fp += result_dict["false_pos"]
        num_fn += result_dict["false_neg"]
    
    precision = calculate_precision(num_tp, num_fp, None)
    recall = calculate_recall(num_tp, None, num_fn)
    
    return (precision, recall)



def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)

    precisions = [] 
    recalls = []

    for confidence_threshold in confidence_thresholds:

        final_prediction_boxes = []
        # iterate over all images
        for image_prediction_boxes, confidence_score in zip(all_prediction_boxes, confidence_scores):
            # select results above the confidence threshold
            final_prediction_boxes.append(image_prediction_boxes[confidence_score >= confidence_threshold])

        precision, recall = calculate_precision_recall_all_images(final_prediction_boxes, all_gt_boxes, iou_threshold)
        
        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)

    precision_sum = 0
    for recall_level in recall_levels:
        # Select the precisions where the corresponding recall value is greater than the current recall level
        selected_precisions = precisions[recalls >= recall_level]
        
        # If there are no such precisions, then the max precision is 0. Otherwise, it's the max of the selected precisions
        if selected_precisions.size == 0:
            max_precision = 0
        else:
            max_precision = np.max(selected_precisions)
        
        precision_sum += max_precision
        
    average_precision = precision_sum / len(recall_levels)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
