"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import json
import os
from tqdm import tqdm
import pickle
import numpy as np
import cv2
import bisect
import re

from transformers import AutoProcessor
import logging
import openpi.shared.download as download
import sentencepiece


IM_SIZE = 256

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
import tyro

# from .rlbench_utils import (
#     keypoint_discovery,
#     image_to_float_array,
#     store_instructions
# )

REPO_NAME = "mbronars/peract2_subgoal_train"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_PATH = "/data/group_data/katefgroup/datasets/hiveformer_clean/"
INSTRUCTION_PATH = "/data/user_data/mbronars/packages/analogical_manipulation/instructions/hiveformer/instructions.json"

tasks = [
    # 'basketball_in_hoop',
    # 'close_jar',
    # 'beat_the_buzz',
    # 'wipe_desk',
    'take_shoes_out_of_box',
    # 'change_clock',
    # 'close_fridge',
    # 'empty_dishwasher',
]

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


def main(push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    all_instructions = json.load(open(INSTRUCTION_PATH, 'r'))
    
    # output_path = LEROBOT_HOME / REPO_NAME
    output_path = "/data/group_data/katefgroup/VLA/lerobot_datasets/shoes_subgoal"
    # print(f"Output path: {output_path}")
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists. Deleting...")
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        root=output_path,
        fps=10,
        features={
            "front_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "front_seg": {
                "dtype": "image",
                "shape": (256, 256),
                "names": ["height", "width", "channel"],
            },
            "wrist_seg": {
                "dtype": "image",
                "shape": (256, 256),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
            "tokenized_subtask": {
                "dtype": "int64",
                "shape": (48,),
                "names": ["subtask"],
            },
            "tokenized_subtask_mask": {
                "dtype": "bool",
                "shape": (48,),
                "names": ["subtask_mask"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    
    tokenizer = PaligemmaSubtaskTokenizer(max_len=48)
    
    subgoal_annotation_json = "/data/user_data/mbronars/packages/VLAD/all_annotations_new.json"
    subgoal_annotations = json.load(open(subgoal_annotation_json, 'r'))
    for task in tasks:
        task_folder = f'{RAW_DATASET_PATH}/{task}/variation0/episodes'
        episodes = sorted(os.listdir(task_folder))
        episodes = episodes[:80]
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
                    dataset.add_frame(
                        {
                            "front_image": front_image,
                            "wrist_image": wrist_image,
                            "front_seg": front_seg,
                            "wrist_seg": wrist_seg,
                            "state": state,
                            "actions": action,
                            "tokenized_subtask": tokenized_subtask,
                            "tokenized_subtask_mask": tokenized_subtask_mask,
                        }
                    )
                    
                # with open(f"{task_folder}/{ep}/variation_number.pkl", 'rb') as f:
                #     var_ = str(pickle.load(f))
                
                var_ = "0"
                    
                instruction = all_instructions[task][var_][0]
                dataset.save_episode(task=instruction)
                    
                
            
    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["peract2", "panda", "keypose"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
