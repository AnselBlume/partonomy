import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

rng = np.random.default_rng(42)

def extract_components(mask: np.ndarray):
    '''
    Given a binary mask (numpy array of shape (h, w)), this function extracts each connected component.
    Returns a numpy array of component masks with shape (n, h, w), where each mask is binary.
    '''
    mask_uint8 = mask.astype(np.uint8)

    # Extract connected components
    num_labels, labels_im = cv2.connectedComponents(mask_uint8, connectivity=8)

    components = []
    for label in range(1, num_labels): # Label 0 is background
        component_mask = (labels_im == label).astype(np.uint8)
        components.append(component_mask)

    return np.array(components)

def sample_points(
    component_mask: np.ndarray,
    num_positive: int = 10,
    num_negative: int = 10,
    min_dist_from_mask: int = 10,
    bias_positive: bool = False,
    bias_negative: bool = False
):
    '''
    Given a single connected component mask (binary array of shape (h, w)), sample:
      - num_positive random coordinates where the mask is 1 (inside the component),
        optionally biased towards the central regions of the component.
      - num_negative coordinates from the background that are at least min_dist_from_mask away from the component,
        optionally biased towards those closer to the component.

    For positive samples, if bias_positive is True, a distance transform is computed on the component mask
    to calculate the distance of each positive pixel to the nearest background pixel. These distances are used as
    weights to favor central pixels in the random sampling. Otherwise, positive samples are chosen uniformly.

    For negative samples, we first filter for pixels that are at least min_dist_from_mask away from the component.
    If bias_negative is True, weights are computed to favor negatives closer to the component (i.e. with lower distance values),
    and sampling is performed accordingly. Otherwise, negative samples are chosen uniformly from the valid set.

    Returns two numpy arrays of coordinates (row, col) for positive and negative points.
    '''
    # Get indices where the component is present (positive) and absent (negative candidates)
    pos_indices = np.argwhere(component_mask == 1)
    neg_indices = np.argwhere(component_mask == 0)

    # Positive Sampling:
    if pos_indices.shape[0] > 0:
        num_pos = min(num_positive, pos_indices.shape[0])
        if bias_positive:
            # Compute a distance transform on the component mask.
            # Higher values for pixels further from the boundary.
            pos_dt = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
            weights = pos_dt[pos_indices[:, 0], pos_indices[:, 1]]
            # Avoid degenerate case: if all weights are zero, revert to uniform.
            if np.sum(weights) == 0:
                probabilities = None
            else:
                probabilities = weights / np.sum(weights)
            sampled_indices = np.random.choice(pos_indices.shape[0], num_pos, replace=False, p=probabilities)
        else:
            sampled_indices = np.random.choice(pos_indices.shape[0], num_pos, replace=False)
        pos_sampled = pos_indices[sampled_indices]
    else:
        pos_sampled = np.empty((0, 2), dtype=int)

    # Negative Sampling:
    # Compute distance transform for negatives: treat component as obstacle.
    mask_for_dt = (component_mask == 0).astype(np.uint8) * 255
    dt = cv2.distanceTransform(mask_for_dt, cv2.DIST_L2, 5)  # Array of distance values

    # Filter negative candidates to those with a distance >= min_dist_from_mask.
    neg_filtered = []
    for idx in neg_indices:
        row, col = idx
        if dt[row, col] >= min_dist_from_mask:
            neg_filtered.append(idx)
    neg_filtered = np.array(neg_filtered)

    if neg_filtered.shape[0] == 0:
        logger.warning('No negative points meet the min_dist_from_mask threshold.')
        neg_sampled = np.empty((0, 2), dtype=int)
    else:
        if bias_negative:
            dt_neg = dt[neg_filtered[:, 0], neg_filtered[:, 1]]
            # Compute weights: lower dt values (closer to the component) get higher weight.
            epsilon = 1e-6  # small constant to avoid division by zero
            weights = 1.0 / (dt_neg - min_dist_from_mask + epsilon)
            probabilities = weights / np.sum(weights)
            if neg_filtered.shape[0] >= num_negative:
                sampled_indices = np.random.choice(neg_filtered.shape[0], num_negative, replace=False, p=probabilities)
                neg_sampled = neg_filtered[sampled_indices]
            else:
                logger.warning('Not enough negative points meeting the min_dist_from_mask threshold; returning all available valid negatives.')
                neg_sampled = neg_filtered
        else:
            if neg_filtered.shape[0] >= num_negative:
                sampled_indices = np.random.choice(neg_filtered.shape[0], num_negative, replace=False)
                neg_sampled = neg_filtered[sampled_indices]
            else:
                logger.warning('Not enough negative points meeting the min_dist_from_mask threshold; returning all available valid negatives.')
                neg_sampled = neg_filtered

    return pos_sampled, neg_sampled