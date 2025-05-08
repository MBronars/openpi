from openpi.training import config as config_func
from openpi.policies import policy_config
from openpi.shared import download
import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro
import json

import cv2
import bisect
import re

from transformers import AutoProcessor
import logging
import openpi.shared.download as download
import sentencepiece




REPO_NAME = "mbronars/peract2_train"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_PATH = "/data/group_data/katefgroup/datasets/hiveformer_clean/"
INSTRUCTION_PATH = "/data/user_data/mbronars/packages/analogical_manipulation/instructions/hiveformer/instructions.json"
IM_SIZE = 256

tasks = [
    'take_shoes_out_of_box',
]

def main():
    # Create a trained policy.
    config = config_func.get_config("pi0_hiveformer")
    # checkpoint_path = "/data/user_data/mbronars/packages/openpi/checkpoints/pi0_hiveformer/shoes_maybe_fixed/13000"
    checkpoint_path = "/data/user_data/mbronars/packages/openpi/checkpoints/pi0_hiveformer_subgoal/shoes_subgoal/5000"
    # checkpoint_path = "/data/user_data/mbronars/packages/openpi/checkpoints/pi0_hiveformer_test/shoe_test/1000"
    # get checkpoint number from path
    checkpoint_num = int(checkpoint_path.split("/")[-1])
    checkpoint_dir = download.maybe_download(checkpoint_path)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    
    all_instructions = json.load(open(INSTRUCTION_PATH, 'r'))
    
    values = {}
    split = "val"
    tokenizer = PaligemmaSubtaskTokenizer(max_len=48)
    
    subgoal_annotation_json = "/data/user_data/mbronars/packages/VLAD/all_annotations_new.json"
    subgoal_annotations = json.load(open(subgoal_annotation_json, 'r'))
    for task in tasks:
        task_folder = f'{RAW_DATASET_PATH}/{task}/variation0/episodes'
        episodes = sorted(os.listdir(task_folder))
        episodes = episodes[80:81]
        task_annotation_name = task + "_var0_episode_0"
        task_annotations = subgoal_annotations[task_annotation_name]
        annotated_keyframes = sorted(task_annotations.keys(), key=lambda x: int(x))
        annotated_subgoals = [task_annotations[k]['task_text'] for k in annotated_keyframes]
        for ep in tqdm(episodes):
                # Read low-dim file from RLBench
                ld_file = f"{task_folder}/{ep}/low_dim_obs.pkl"
                with open(ld_file, 'rb') as f:
                    demo = pickle.load(f)

                # Keypose discovery
                key_frames = keypoint_discovery(demo, bimanual=False)
                key_frames.insert(0, 0)
                
                for i, k in enumerate(key_frames[:-1]):
                    subtask, skill = get_current_subgoal_and_skill(i, annotated_subgoals)
                    tokenized_subtask, tokenized_subtask_mask = tokenizer.tokenize(subtask)
                    
                    bboxes, object_names, segmentations = get_bounding_boxes_and_object_names(
                        task_folder, ep, i, k, task_annotations, annotated_keyframes, ["front", "wrist"]
                    )
                    
                    front_image = np.array(Image.open(f"{task_folder}/{ep}/front_rgb/{k}.png"))
                    wrist_image = np.array(Image.open(f"{task_folder}/{ep}/wrist_rgb/{k}.png"))
                    front_seg = segmentations[0].astype(np.bool)
                    wrist_seg = segmentations[1].astype(np.bool)
                    state = np.concatenate([
                        demo[k].gripper_pose,
                        [demo[k].gripper_open]])
                    next_k = key_frames[i + 1]
                    action = np.concatenate([
                        demo[next_k].gripper_pose,
                        [demo[next_k].gripper_open]])
                    example= {
                            "observation/front_image": front_image,
                            "observation/wrist_image": wrist_image,
                            "observation/state": state,
                            "prompt": all_instructions[task]['0'][0]
                        }
                        
                    
                    policy_output = policy.infer(example)
                    action_chunk = policy_output["actions"]
                    language_pred = policy_output["language_tokens"]
                    segmentation = policy_output["segmentation"]
                    
                    pred_actions = action_chunk.reshape(-1, 8)
                    action = action.reshape(-1, 8)
                    segmentation = segmentation > 0
                    wrist_seg = cv2.resize(wrist_seg.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
                    front_seg = cv2.resize(front_seg.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
                    gt_segmentation = np.array([wrist_seg, front_seg])
                    gt_segmentation = gt_segmentation > 0
                    losses = compute_metrics(pred_actions, action, language_pred, tokenized_subtask, segmentation, gt_segmentation)
                    
                    
                    # seg = segmentation > 0
                    # pred_front_seg = seg[0]
                    # pred_wrist_seg = seg[1]
                    # cv2.imwrite(f"{ep}_{i}_pred_full_wrist_seg_.png", pred_wrist_seg.astype(np.uint8) * 255)
                    # cv2.imwrite(f"{ep}_{i}_pred_full_front_seg.png", pred_front_seg.astype(np.uint8) * 255)
                    # cv2.imwrite(f"{ep}_{i}_gt_front_seg.png", front_seg.astype(np.uint8) * 255)
                    # cv2.imwrite(f"{ep}_{i}_gt_wrist_seg.png", wrist_seg.astype(np.uint8) * 255)
                                        
                    # pred_actions = action_chunk.reshape(-1, 8)
                    # action = action.reshape(-1, 8)
                    # losses = compute_metrics(pred_actions, action)
                    
                    
                    for n, l in losses.items():
                        key = f"{split}-losses/mean/{n}"
                        if key not in values:
                            values[key] = np.array([])
                        values[key] = np.append(values[key], np.expand_dims(l, axis=0))
                        task_key = f"{split}-losses/{task}/{n}"
                        if task_key not in values:
                            values[task_key] = np.array([])
                        values[task_key] = np.append(values[task_key], np.expand_dims(l, axis=0))

                                                    
    eval_folder = "/data/user_data/mbronars/packages/openpi/evaluations"
    
    save_path = os.path.join(eval_folder, f"hiveformer_eval_checkpoint{checkpoint_num}.json")
    values = {k: v.mean().item() for k, v in values.items()}
    # save values to json
    with open(save_path, 'w') as f:
        json.dump(values, f)
                

def compute_metrics(pred, gt, language_pred, language_gt, seg_pred, seg_gt):
    # pred/gt are (B, L, 7), mask (B, L)
    pos_l2 = np.sqrt(np.sum((pred[..., :3] - gt[..., :3]) ** 2, axis=-1))
    
    # symmetric quaternion eval
    quat_l1 = np.sum(np.abs(pred[..., 3:7] - gt[..., 3:7]), axis=-1)
    quat_l1_ = np.sum(np.abs(pred[..., 3:7] + gt[..., 3:7]), axis=-1)
    select_mask = (quat_l1 < quat_l1_).astype(float)
    quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
    
    # gripper openness
    openess = ((pred[..., 7:] >= 0.5) == (gt[..., 7:] > 0.0))
    tr = 'traj_'
    
    # Trajectory metrics
    ret_1 = {
        tr + 'action_mse': np.mean((pred - gt) ** 2),
        tr + 'pos_l2': np.mean(pos_l2),
        tr + 'pos_acc_001': np.mean((pos_l2 < 0.01).astype(float)),
        tr + 'rot_l1': np.mean(quat_l1),
        tr + 'rot_acc_0025': np.mean((quat_l1 < 0.025).astype(float)),
        tr + 'gripper': np.mean(openess.flatten().astype(float))
    }
    
    if language_pred is not None:
        # Language metrics
        language_acc = np.mean((language_pred == language_gt).astype(float))
        ret_1.update({
            'language_top1_acc': language_acc
        })
    
    if seg_pred is not None:
        front_pred = seg_pred[0]
        wrist_pred = seg_pred[1]
        front_gt = seg_gt[0]
        wrist_gt = seg_gt[1]
        front_iou = compute_iou(front_pred, front_gt)
        wrist_iou = compute_iou(wrist_pred, wrist_gt)
        ret_1.update({
            'front_iou': front_iou,
            'wrist_iou': wrist_iou
        })
        
    
    # ret_2 = {
    #     tr + 'pos_l2': np.mean(pos_l2, axis=-1),
    #     tr + 'pos_acc_001': np.mean((pos_l2 < 0.01).astype(float), axis=-1),
    #     tr + 'rot_l1': np.mean(quat_l1, axis=-1),
    #     tr + 'rot_acc_0025': np.mean((quat_l1 < 0.025).astype(float), axis=-1)
    # }
    
    return ret_1#, ret_2

def compute_iou(pred_mask, gt_mask):
    """
    Calculate the Intersection over Union (IoU) between predicted mask and ground truth mask.
    
    Args:
        pred_mask (numpy.ndarray): Binary predicted mask (0 or 1)
        gt_mask (numpy.ndarray): Binary ground truth mask (0 or 1)
        
    Returns:
        float: IoU score ranging from 0 to 1
    """
    # Ensure the masks are binary
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Handle edge case when both masks are empty
    if union == 0:
        return 1.0
    
    # Calculate IoU
    iou = intersection / union
    
    return iou

    
                    

class PaligemmaSubtaskTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # tokenize "\n" separately as the "start of answer" token
        tokens = self._tokenizer.encode("\n") + self._tokenizer.encode(cleaned_text, add_bos=False, add_eos=True)
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)
    

def find_closest_keyframe(current_index, keyframes):
    """
    Given a current image index and a sorted list of keyframe indices,
    find the keyframe index that is closest to the current index.
    """
    int_keyframes = [int(k) for k in keyframes]
    pos = bisect.bisect_left(int_keyframes, current_index)
    
    if pos == 0:
        return pos
    return pos-1

def get_bounding_boxes_and_object_names(task_folder, ep, keyframe_idx, curr_frame, annotations, annotated_keyframes, camera_names):
    """
    Extract bounding boxes and object names from the masks and annotation data.
    Returns bounding boxes and object names for each camera view.
    """
    bboxes_by_camera = {cam: [] for cam in camera_names}
    object_names_by_camera = {cam: [] for cam in camera_names}
    
    # Use annotation data directly
    relevant_objects = annotations[annotated_keyframes[keyframe_idx]]['relevant_objects']
    
    masks = []
    
    for cam in camera_names:
        cam_mask = np.zeros((IM_SIZE, IM_SIZE), dtype=np.uint8)
        for obj_name, obj_info in relevant_objects.items():
            # For each object, get mask and extract bounding box
            combined_mask = np.zeros((IM_SIZE, IM_SIZE), dtype=np.uint8)
            
            for mask_handle in obj_info['mask_handles']:
                mask_handle = int(mask_handle)
                # Convert sub object handle to RGB
                color = (mask_handle // 256 // 256) % 256, (mask_handle // 256) % 256, mask_handle % 256
                # color = mask_handle % 256, (mask_handle // 256) % 256, (mask_handle // 256 // 256) % 256
                mask_path = f"{task_folder}/{ep}/{cam}_mask/{curr_frame}.png"
                                
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    mask = cv2.inRange(mask, color, color)
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                    
            # add the combined mask to the camera mask
            cam_mask = cv2.bitwise_or(cam_mask, combined_mask)
            
            # Get bounding box from mask
            if np.sum(combined_mask) > 0:
                y_indices, x_indices = np.where(combined_mask > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # Normalize coordinates to 0-1 range
                norm_y_min, norm_x_min = int(y_min / IM_SIZE * 1024), int(x_min / IM_SIZE * 1024)
                norm_y_max, norm_x_max = int(y_max / IM_SIZE * 1024), int(x_max / IM_SIZE * 1024)
                
                # Format as "<locy_min>, <locx_min>, <locy_max>, <locx_max>" as per Paligemma
                bbox_str = f"<loc{norm_y_min:04d}>, <loc{norm_x_min:04d}>, <loc{norm_y_max:04d}>, <loc{norm_x_max:04d}>"
                bboxes_by_camera[cam].append(bbox_str)
                object_names_by_camera[cam].append(obj_name)
        masks.append(cam_mask)
    
    
    return bboxes_by_camera, object_names_by_camera, masks

def get_current_subgoal_and_skill(frame_idx, annotated_subgoals):
    """
    Get the subgoal and atomic skill for a specific frame index.
    Returns tuple: (subgoal, atomic_skill)
    """
    # Get subgoal text
    
    subgoal_text = annotated_subgoals[frame_idx]
    
    # Extract atomic skill if present in parentheses at the end
    skill = ""
    match = re.search(r'\(([^)]+)\)$', subgoal_text)
    if match:
        skill = match.group(1)
        # Clean the subgoal text by removing the parenthetical part
        clean_subgoal = re.sub(r'\s*\([^)]+\)$', '', subgoal_text)
        return clean_subgoal, skill
    else:
        return subgoal_text, skill

def _num2id(int_):
    str_ = str(int_)
    return '0' * (4 - len(str_)) + str_

def _is_stopped(demo, i, obs, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return (
        small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )


def _is_stopped_right(demo, i, obs, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].right.gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].right.gripper_open
        and demo[max(0, i - 2)].right.gripper_open == demo[max(0, i - 1)].right.gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return small_delta and (not next_is_not_final) and gripper_state_no_change


def _is_stopped_left(demo, i, obs, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].left.gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].left.gripper_open
        and demo[max(0, i - 2)].left.gripper_open == demo[max(0, i - 1)].left.gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return small_delta and (not next_is_not_final) and gripper_state_no_change

def _keypoint_discovery_unimanual(demo, stopping_delta=0.1):
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopping_delta)
        stopped = (stopped_buffer <= 0) and stopped
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)
    # print("Found %d keypoints." % len(episode_keypoints), episode_keypoints)
    return episode_keypoints

def _keypoint_discovery_heuristic(demo, stopping_delta=0.1, bimanual=False):
    if bimanual:
        return _keypoint_discovery_bimanual(demo, stopping_delta)
    else:
        return _keypoint_discovery_unimanual(demo, stopping_delta)


def keypoint_discovery(demo, method="heuristic", bimanual=False):
    episode_keypoints = []
    if method == "heuristic":
        stopping_delta = 0.1
        return _keypoint_discovery_heuristic(demo, stopping_delta, bimanual)

    elif method == "random":
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20, replace=False
        )
        episode_keypoints.sort()
        return episode_keypoints

    elif method == "fixed_interval":
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    tyro.cli(main)
                                        
