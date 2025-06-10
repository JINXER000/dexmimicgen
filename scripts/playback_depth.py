# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Dataset Playback and Visualization for dexmimicgen

This script plays back robotic demonstration datasets stored in HDF5 format. It supports
visualizing trajectories using either the simulation environment or stored image observations.
Playback can be rendered on-screen or saved as a video.

Args:
    --dataset (str): Path to the HDF5 dataset file.
    --filter_key (str, optional): Key to filter specific trajectories in the dataset.
    --n (int, optional): Number of trajectories to play (default: all).
    --use-obs (bool): Use image observations instead of simulator playback.
    --use-actions (bool): Play back stored actions in open-loop instead of using simulation states.
    --render (bool): Render playback on-screen.
    --video_path (str, optional): Path to save the video file.
    --video_skip (int): Frame skip rate for video recording (default: 5).
    --render_image_names (list of str, optional): Camera names or image observation keys to render.
    --first (bool): Only use the first frame of each episode.
    --extend_states (bool): Extend the last step of episodes for 50 extra frames.
    --verbose (bool): Enable additional logging.
    --use_current_model (bool): Use the current model instead of the one stored in the dataset.

Example usage:
    # render the dataset playback on-screen
    python script.py --dataset /path/to/dataset.hdf5 --render

    # playback the dataset using actions open-loop
    python script.py --dataset /path/to/dataset.hdf5 --use-actions

    # use the current env model when playing back the dataset (useful if you modified the env)
    python script.py --dataset /path/to/dataset.hdf5 --use-current-model

    # debug the dataset playback with verbose logging and first frame only
    python script.py --dataset /path/to/dataset.hdf5 --verbose --first
"""
import sys
robosuite_path = "/home/user/yzchen_ws/imitation_learning/robosuite/"
sys.path.append(robosuite_path)

import argparse
import datetime
import json
import os
import random
import time

import h5py
import imageio
import numpy as np
import robosuite
from robosuite.demos.vis_depth_seg import get_individual_pcd, get_name2id
from termcolor import colored
from scipy.spatial.transform import Rotation

# IMPORTANT: you need to import the package to register the environments
import dexmimicgen


def playback_trajectory_with_env(
    env,
    initial_state,
    states,
    actions=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
    verbose=False,
    pc_fn = None,
    assumed_eef_pos = None,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
        pc_fn: it will get the object pc
    """
    write_video = video_writer is not None
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    ## this reset call doesn't seem necessary.
    ## seems ok to remove but haven't fully tested it.
    ## removing for now
    # env.reset()

    if verbose:
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    reset_to(env, initial_state)

    traj_len = states.shape[0]
    # action_playback = actions is not None
    # if action_playback:
    #     assert states.shape[0] == actions.shape[0]
    if actions is not None:
        stacked_actions = actions.reshape(*actions.shape[:-1],-1,7)
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[...,[-1]]

    if render is False:
        print(colored("Running episode...", "yellow"))

    pc_dict_list = []
    success = False
    for i in range(traj_len):
        start = time.time()

        obs = reset_to(env, {"states": states[i]})

        ## get the obj pc in the current frame
        if pc_fn is not None:
            obj_pc_dict = pc_fn(env, obs)
            pc_dict_list.append(obj_pc_dict)

        if actions is not None:
            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
            
                # read pos and ori from robots
                side = robot.arms[0]
                controller = robot.part_controllers[side]
                action_goal_pos[i,idx] = controller.goal_pos
                action_goal_ori[i,idx] = Rotation.from_matrix(
                    controller.goal_ori).as_rotvec()


        # on-screen render
        if render:
            if env.viewer is None:
                env.initialize_renderer()

            # so that mujoco viewer renders
            env.viewer.update()

            max_fr = 60
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # video render
        if write_video:
            frontview_img = obs['frontview_image']
            video_writer.append_data(frontview_img)
            # if video_count % video_skip == 0:
            #     video_img = []
            #     for cam_name in camera_names:
            #         im = env.sim.render(height=512, width=512, camera_name=cam_name)[
            #             ::-1
            #         ]
            #         video_img.append(im)
            #     video_img = np.concatenate(
            #         video_img, axis=1
            #     )  # concatenate horizontally
            #     video_writer.append_data(video_img)

            video_count += 1

        if first:
            break

    print('video length:', video_count)

    if render:
        env.viewer.close()
        env.viewer = None

    # if action_playback and not success:
    #     print(colored("warning: playback did not success", "red"))
    
    if actions is not None:
        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_gripper
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
    else:
        abs_actions = None

    return pc_dict_list, abs_actions


# def playback_trajectory_with_obs(
#     traj_grp,
#     video_writer,
#     video_skip=5,
#     image_names=None,
#     first=False,
# ):
#     """
#     This function reads all "rgb" observations in the dataset trajectory and
#     writes them into a video.

#     Args:
#         traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
#         video_writer (imageio writer): video writer
#         video_skip (int): determines rate at which environment frames are written to video
#         image_names (list): determines which image observations are used for rendering. Pass more than
#             one to output a video with multiple image observations concatenated horizontally.
#         first (bool): if True, only use the first frame of each episode.
#     """
#     assert (
#         image_names is not None
#     ), "error: must specify at least one image observation to use in @image_names"
#     video_count = 0

#     traj_len = traj_grp["obs/{}".format(image_names[0] + "_image")].shape[0]
#     for i in range(traj_len):
#         if video_count % video_skip == 0:
#             # concatenate image obs together
#             im = [traj_grp["obs/{}".format(k + "_image")][i] for k in image_names]
#             frame = np.concatenate(im, axis=1)
#             video_writer.append_data(frame)
#         video_count += 1

#         if first:
#             break


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta


class ObservationKeyToModalityDict(dict):
    """
    Custom dictionary class with the sole additional purpose of automatically registering new "keys" at runtime
    without breaking. This is mainly for backwards compatibility, where certain keys such as "latent", "actions", etc.
    are used automatically by certain models (e.g.: VAEs) but were never specified by the user externally in their
    config. Thus, this dictionary will automatically handle those keys by implicitly associating them with the low_dim
    modality.
    """

    def __getitem__(self, item):
        # If a key doesn't already exist, warn the user and add default mapping
        if item not in self.keys():
            print(
                f"ObservationKeyToModalityDict: {item} not found,"
                f" adding {item} to mapping with assumed low_dim modality!"
            )
            self.__setitem__(item, "low_dim")
        return super(ObservationKeyToModalityDict, self).__getitem__(item)


def reset_to(env, state,should_ret=False):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    # should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    ## NOTE : below will make the iteration very slow. Also, the videowriter is incorrect as well. 
    if should_ret:
        # only return obs if we've done a forward call - otherwise the observations will be garbage
        return env._get_observations(force_update = True)
    return None


def playback_dataset(args):
    # some arg checking
    write_video = args.render is not True
    if args.video_path is None:
        args.video_path = args.dataset.split(".hdf5")[0] + ".mp4"
        if args.use_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_actions.mp4"
    assert not (args.render and write_video)  # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        args.render_image_names = "robot0_agentview_center"

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert (
            not args.use_actions
        ), "playback with observations is offline and does not support action playback"

    env = None

    # create environment only if not playing back with observations
    if not args.use_obs:
        cam_names = ["agentview", "birdview", "frontview"]
        W = H =  512# 128

        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)

        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = True
        env_kwargs["camera_depths"] = True
        env_kwargs["camera_segmentations"] = "instance"
        env_kwargs["camera_names"] = cam_names
        env_kwargs["camera_heights"] = H
        env_kwargs["camera_widths"] = W
        env_kwargs["controller_configs"] = env_meta["env_kwargs"]['controller_configs']

        if args.verbose:
            print(
                colored(
                    "Initializing environment for {}...".format(env_kwargs["env_name"]),
                    "yellow",
                )
            )
        if "env_lang" in env_kwargs:
            env_kwargs.pop("env_lang")
        if "env_name" == "TwoArnCanSortRandom" and args.use_current_model:
            print(
                colored(
                    "Warning: TwoArnCanSortRandom environment with use_current_model will have chance of incorrect color placement (red to blue bin or blue to red bin)",
                    "yellow",
                )
            )
        env = robosuite.make(**env_kwargs)

    get_gt_link_state(env,  interested_objs = ['table', 'pot_root', 'robot0_base'])

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    elif "data" in f.keys():
        demos = list(f["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # randomize the playback
    if args.n is not None:
        # random.shuffle(demos)
        demos = demos[: args.n]



    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        new_hdf5_path = args.dataset.split(".hdf5")[0] + f"_{ep}_pcd.hdf5"
        if os.path.exists(new_hdf5_path):
            print(colored(f"Skipping episode {ep} as {new_hdf5_path} already exists", "red"))
            continue
        # if args.use_obs:
        #     playback_trajectory_with_obs(
        #         traj_grp=f["data/{}".format(ep)],
        #         video_writer=video_writer,
        #         video_skip=args.video_skip,
        #         image_names=args.render_image_names,
        #         first=args.first,
        #     )
        #     continue

        # maybe dump video
        video_path = args.dataset.split(".hdf5")[0] + f"_{ep}.mp4"
        video_writer = None
        if write_video:
            video_writer = imageio.get_writer(video_path, fps=20)

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        if args.use_current_model:
            initial_state["model"] = env.sim.model.get_xml()
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        assumed_eef_pos = f["data/{}/datagen_info/target_pose".format(ep)][()]

        if args.extend_states:
            states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]

        ## get the point cloud for the first frame
        if 'lift_tray' in args.dataset:
            interested_objs = ['pot', 'obj0', 'obj1']
        elif 'assembly' in args.dataset:
            interested_objs = ['base', 'piece_1', 'piece_2']
        elif 'transport' in args.dataset:
            interested_objs = ['trash', 'payload', 'transport_start_bin', 'transport_start_bin_lid', 'transport_target_bin', 'transport_trash_bin']
        elif 'threading' in args.dataset:
            interested_objs = ['needle_obj', 'tripod_obj']
        pc_fn = get_pcd_dict_fn(cam_names, W, H,  interested_objs, record_ply=True)

        pc_dict_list, abs_actions = playback_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
            verbose=args.verbose,
            pc_fn = pc_fn,
            assumed_eef_pos = assumed_eef_pos,
        )

        if abs_actions is not None:
            robot0_eef_pos = f["data/{}/obs/robot0_eef_pos".format(ep)][()]
            robot0_eef_quat = f["data/{}/obs/robot0_eef_quat".format(ep)][()]
            delta_error_info = evaluate_rollout_error(
                env, states, actions, robot0_eef_pos, robot0_eef_quat)

            info = {
                'delta_max_error': delta_error_info,
            }
            print('error info:', info)
        
        def animate_pts():
            import open3d as o3d
            ## https://chat.deepseek.com/a/chat/s/99bab2d6-7547-4eb9-a5d1-d7667844211b
            pcd = o3d.geometry.PointCloud()
            for obj_name in pc_dict_list[0].keys():
                pcd += pc_dict_list[0][obj_name]
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='all', width=800, height=600)
            vis.add_geometry(pcd)

            # 设置相机轨迹参数
            ctr = vis.get_view_control()
            ctr.set_zoom(2)

            num_frames = len(pc_dict_list)
            for i in range(num_frames):
                pcd = o3d.geometry.PointCloud()
                for obj_name in pc_dict_list[i].keys():
                    pcd += pc_dict_list[i][obj_name]
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                # cv2.waitKey(1)
            vis.destroy_window()

        # piece1_x = [np.array(pc_dict_list[i]['piece_1'].points).mean(axis=0)[0] for i in range(len(pc_dict_list))]
        # from matplotlib import pyplot as plt
        # plt.plot(piece1_x)
        # plt.savefig('piece1_x.png')
        # animate_pts()


        ## copy the demo to a new hdf5 file
        with h5py.File(new_hdf5_path, 'w') as dst:
            data_group = dst.create_group('data')
            # copy the /data{} group
            f.copy(f"data/{ep}/", data_group)

            # add the initial point cloud to the new hdf5 file
            pc_group = data_group.create_group('obj_pcd')
            for obj_name in interested_objs:
                obj_pc_list = [pc_dict_list[i][obj_name] for i in range(len(pc_dict_list))]
                pc_group.create_dataset(obj_name+ '_points', (len(obj_pc_list),), dtype=h5py.vlen_dtype('float32'))
                for i in range(len(obj_pc_list)):
                    pc_group[obj_name+ '_points' ][i] = np.asarray(obj_pc_list[i].points, dtype=np.float32).flatten()
                # pc_group.create_dataset(obj_name+ '_colors', (len(obj_pc_list),), dtype=h5py.vlen_dtype('float32'))
                # for i in range(len(obj_pc_list)):
                #     pc_group[obj_name+'_colors'][i] = np.asarray(obj_pc_list[i].colors, dtype=np.float32).flatten()
                    
            # add abs action
            if abs_actions is not None:
                data_group.create_dataset("abs_actions", data=abs_actions)
        print(colored(f"Saved processed demo {ind} to {new_hdf5_path}", "green"))

        if write_video:
            print(colored(f"Saved video to {video_path}", "green"))
            video_writer.close()
    f.close()


    if env is not None:
        env.close()

def get_pcd_dict_fn(cam_names, W, H, interested_objs, record_ply=False):
    def get_pcd(env, obs):
        pc_dict = {inst:None for inst in interested_objs}
        name2id = get_name2id(env)
        # print('name_keys:', name2id.keys())
        # obs = env.reset()
        import open3d as o3d
        for obj_name in interested_objs:
            obj_pcd = o3d.geometry.PointCloud()
            for cam in cam_names:
                pcd = get_individual_pcd(cam, obs, W, H, \
                    seg_id=name2id[obj_name],  visualize=False, env=env, filter = True)
                obj_pcd += pcd
            pc_dict[obj_name] = obj_pcd

            # ensure the pcd is not empty
            if len(obj_pcd.points) == 0:
                raise ValueError(f"Point cloud for {obj_name} is empty")

            if record_ply:
                o3d.io.write_point_cloud(f"{obj_name}_pcd.ply", obj_pcd)
        return pc_dict
    return get_pcd

    
def evaluate_rollout_error(env, 
        states, actions, 
        robot0_eef_pos, 
        robot0_eef_quat, 
        metric_skip_steps=1):
    # first step have high error for some reason, not representative

    # evaluate abs actions
    rollout_next_states = list()
    rollout_next_eef_pos = list()
    rollout_next_eef_quat = list()
    obs = reset_to(env, {'states': states[0]})
    for i in range(len(states)):
        obs = reset_to(env, {'states': states[i]})
        obs, reward, done, info = env.step(actions[i])
        obs = env._get_observations()
        rollout_next_states.append(env.get_state()['states'])
        rollout_next_eef_pos.append(obs['robot0_eef_pos'])
        rollout_next_eef_quat.append(obs['robot0_eef_quat'])
    rollout_next_states = np.array(rollout_next_states)
    rollout_next_eef_pos = np.array(rollout_next_eef_pos)
    rollout_next_eef_quat = np.array(rollout_next_eef_quat)

    next_state_diff = states[1:] - rollout_next_states[:-1]
    max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

    next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
    next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
    max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

    next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
        * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
    next_eef_rot_dist = next_eef_rot_diff.magnitude()
    max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

    info = {
        'state': max_next_state_diff,
        'pos': max_next_eef_pos_dist,
        'rot': max_next_eef_rot_dist
    }
    return info

def get_gt_link_state(env, interested_objs = ['table', 'cube1']):
    model = env.sim.model._model
    data = env.sim.data._data

    import mujoco
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            print(f"Body {i}: {body_name}")
            if body_name in interested_objs:
                print(f'position of {body_name}:', data.xpos[i])

        else:
            print(f"Body {i}: (unnamed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
        # default="/home/user/yzchen_ws/imitation_learning/dexmimicgen/datasets/generated/two_arm_lift_tray.hdf5",
        # default="/home/user/yzchen_ws/imitation_learning/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5",
        default="/home/user/yzchen_ws/imitation_learning/dexmimicgen/datasets/generated/two_arm_threading.hdf5",
    )   
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action="store_true",
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action="store_true",
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default=[
            "agentview",
        ],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
        "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action="store_true",
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--extend_states",
        action="store_true",
        help="play last step of episodes for 50 extra frames",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log additional information",
    )

    parser.add_argument(
        "--use_current_model",
        action="store_true",
        help="use the current model instead of the one stored in the dataset",
    )

    args = parser.parse_args()
    playback_dataset(args)
