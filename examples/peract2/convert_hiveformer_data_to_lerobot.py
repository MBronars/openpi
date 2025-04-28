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

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
import tyro

REPO_NAME = "mbronars/peract2_train"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_PATH = "/data/user_data/ngkanats/Peract2_zarr"
INSTRUCTION_PATH = "/data/user_data/mbronars/packages/analogical_manipulation/instructions/peract2/instructions.json"

tasks = [
    'bimanual_push_box',
    'bimanual_lift_ball',
    'bimanual_dual_push_buttons',
    'bimanual_pick_plate',
    'bimanual_put_item_in_drawer',
    'bimanual_put_bottle_in_fridge',
    'bimanual_handover_item',
    'bimanual_pick_laptop',
    'bimanual_straighten_rope',
    'bimanual_sweep_to_dustpan',
    'bimanual_lift_tray',
    'bimanual_handover_item_easy',
    'bimanual_take_tray_out_of_oven'
]

def _num2id(int_):
    str_ = str(int_)
    return '0' * (4 - len(str_)) + str_


def main(push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    all_instructions = json.load(open(INSTRUCTION_PATH, 'r'))
    
    output_path = LEROBOT_HOME / REPO_NAME
    print(f"Output path: {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "front_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_left_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_right_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for task in tasks:
        task_folder = f'{RAW_DATASET_PATH}/train/{task}/all_variations/episodes'
        episodes = sorted(os.listdir(task_folder))
        for ep in tqdm(episodes):
                # Read low-dim file from RLBench
                ld_file = f"{task_folder}/{ep}/low_dim_obs.pkl"
                with open(ld_file, 'rb') as f:
                    demo = pickle.load(f)

                # Keypose discovery
                key_frames = keypoint_discovery(demo, bimanual=True)
                key_frames.insert(0, 0)
                
                for i, k in enumerate(key_frames[:-1]):
                    front_image = np.array(Image.open(f"{task_folder}/{ep}/front_rgb/rgb_{_num2id(k)}.png"))
                    wrist_left_image = np.array(Image.open(f"{task_folder}/{ep}/wrist_left_rgb/rgb_{_num2id(k)}.png"))
                    wrist_right_image = np.array(Image.open(f"{task_folder}/{ep}/wrist_right_rgb/rgb_{_num2id(k)}.png"))
                    state = np.concatenate([
                        demo[k].left.gripper_pose,
                        [demo[k].left.gripper_open],
                        demo[k].right.gripper_pose,
                        [demo[k].right.gripper_open]])
                    next_k = key_frames[i + 1]
                    action = np.concatenate([
                        demo[next_k].left.gripper_pose,
                        [demo[next_k].left.gripper_open],
                        demo[next_k].right.gripper_pose,
                        [demo[next_k].right.gripper_open]])
                    
                    dataset.add_frame(
                        {
                            "front_image": front_image,
                            "wrist_left_image": wrist_left_image,
                            "wrist_right_image": wrist_right_image,
                            "state": state,
                            "actions": action,
                        }
                    )
                    
                with open(f"{task_folder}/{ep}/variation_number.pkl", 'rb') as f:
                    var_ = str(pickle.load(f))
                    
                instruction = all_instructions[task][var_]
                dataset.save_episode(task=instruction)
                    
                
                
        
    # for raw_dataset_name in RAW_DATASET_NAMES:
    #     raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
    #     for episode in raw_dataset:
    #         for step in episode["steps"].as_numpy_iterator():
    #             dataset.add_frame(
    #                 {
    #                     "image": step["observation"]["image"],
    #                     "wrist_image": step["observation"]["wrist_image"],
    #                     "state": step["observation"]["state"],
    #                     "actions": step["action"],
    #                 }
    #             )
                
            
    #         dataset.save_episode(task=step["language_instruction"].decode())

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
