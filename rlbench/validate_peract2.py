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





REPO_NAME = "mbronars/peract2_train"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_PATH = "/data/group_data/katefgroup/VLA/peract2_raw_squash"
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

def main():
    # Create a trained policy.
    config = config_func.get_config("pi0_peract2")
    checkpoint_path = "/data/user_data/mbronars/packages/openpi/checkpoints/pi0_peract2/peract2_test/29999"
    # get checkpoint number from path
    checkpoint_num = int(checkpoint_path.split("/")[-1])
    checkpoint_dir = download.maybe_download(checkpoint_path)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    
    all_instructions = json.load(open(INSTRUCTION_PATH, 'r'))
    
    values = {}
    split = "val"

    for task in tasks:
            task_folder = f'{RAW_DATASET_PATH}/{split}/{task}/all_variations/episodes'
            episodes = sorted(os.listdir(task_folder))
            for ep in tqdm(episodes):
                    # Read low-dim file from RLBench
                    ld_file = f"{task_folder}/{ep}/low_dim_obs.pkl"
                    with open(ld_file, 'rb') as f:
                        demo = pickle.load(f)

                    # Keypose discovery
                    key_frames = keypoint_discovery(demo, bimanual=True)
                    key_frames.insert(0, 0)
                    
                    with open(f"{task_folder}/{ep}/variation_number.pkl", 'rb') as f:
                        var_ = str(pickle.load(f))
                    
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
                        
                        
                        
                        example = {
                            "observation/front_image": front_image,
                            "observation/wrist_left_image": wrist_left_image,
                            "observation/wrist_right_image": wrist_right_image,
                            "observation/state": state,
                            "prompt": all_instructions[task][var_][0]
                        }
                        
                        action_chunk = policy.infer(example)["actions"]
                        
                        pred_actions = action_chunk.reshape(-1, 8)
                        action = action.reshape(-1, 8)
                        losses = compute_metrics(pred_actions, action)
                        
                        
                        for n, l in losses.items():
                            key = f"{split}-losses/mean/{n}"
                            if key not in values:
                                values[key] = np.array([])
                            values[key] = np.append(values[key], np.expand_dims(l, axis=0))
                            task_key = f"{split}-losses/{task}/{n}"
                            if task_key not in values:
                                values[task_key] = np.array([])
                            values[task_key] = np.append(values[task_key], np.expand_dims(l, axis=0))

                        # for n, l in losses_B.items():
                        #     for task in np.unique(tasks):
                        #         key = f"{split}-loss/{task}/{n}"
                        #         l_task = np.mean(l[tasks == task])
                        #         if key not in values:
                        #             values[key] = np.array([])
                        #         values[key] = np.append(values[key], np.array([l_task]))
                                                    
    eval_folder = "/data/user_data/mbronars/packages/openpi/evaluations"
    
    save_path = os.path.join(eval_folder, f"peract2_eval_checkpoint{checkpoint_num}.json")
    values = {k: v.mean().item() for k, v in values.items()}
    # save values to json
    with open(save_path, 'w') as f:
        json.dump(values, f)
                

def compute_metrics(pred, gt):
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
    
    # ret_2 = {
    #     tr + 'pos_l2': np.mean(pos_l2, axis=-1),
    #     tr + 'pos_acc_001': np.mean((pos_l2 < 0.01).astype(float), axis=-1),
    #     tr + 'rot_l1': np.mean(quat_l1, axis=-1),
    #     tr + 'rot_acc_0025': np.mean((quat_l1 < 0.025).astype(float), axis=-1)
    # }
    
    return ret_1#, ret_2
    
                    
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

def _keypoint_discovery_bimanual(demo, stopping_delta=0.1):
    episode_keypoints = []
    right_prev_gripper_open = demo[0].right.gripper_open
    left_prev_gripper_open = demo[0].left.gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo._observations):
        right_stopped = _is_stopped_right(demo, i, obs.right, stopping_delta)
        left_stopped = _is_stopped_left(demo, i, obs.left, stopping_delta)
        stopped = (stopped_buffer <= 0) and right_stopped and left_stopped
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        right_state_changed = obs.right.gripper_open != right_prev_gripper_open
        left_state_changed = obs.left.gripper_open != left_prev_gripper_open
        state_changed = right_state_changed or left_state_changed
        if i != 0 and (state_changed or last or stopped):
            episode_keypoints.append(i)

        right_prev_gripper_open = obs.right.gripper_open
        left_prev_gripper_open = obs.left.gripper_open
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
                                        
