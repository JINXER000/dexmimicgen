import time
from typing import Dict
# import sys
# robosuite_path = "/home/user/yzchen_ws/imitation_learning/robosuite/"
# dexmimicgen_path = "/home/user/yzchen_ws/imitation_learning/dexmimicgen/"
# sys.path.append(robosuite_path)  # add robosuite root to path
# sys.path.append(dexmimicgen_path)  # add dexmimicgen root to path
# import dexmimicgen


from scripts.playback_depth import get_pcd_dict_fn, reset_to

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils.input_utils import *
from robosuite.utils.ik_utils import IKSolver
import robomimic.utils.obs_utils as ObsUtils

import h5py
import networkx as nx
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation
from collections import namedtuple

ts_tuple = namedtuple("ts_tuple", ["observation", "reward", "done", "info"])

MAX_FR = 25  # max frame rate for running simluation


def to_camel_case(snake_str):
    """Convert snake_case string to CamelCase"""
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def get_sg(hdf5_group, sg_name):
    sg_json = hdf5_group[sg_name][()] if sg_name in hdf5_group else None
    if sg_json is None:
        return None
    sg_str = sg_json.decode('utf-8')
    sg = nx.node_link_graph(json.loads(sg_str))
    return sg

def compose_transformation(xyz, quat):
    rot_mat = Rotation.from_quat(quat).as_matrix()
    trans = np.concatenate([np.concatenate([rot_mat, np.array([xyz]).T], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    return trans

def rotation_6d_to_matrix(rot_6d:np.ndarray) -> np.ndarray:
    # Convert 6D rotation to 3x3 rotation matrix using Gram-Schmidt
    a1, a2 = rot_6d[..., :3], rot_6d[..., 3:]
    
    # Normalize first vector
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    
    # Gram-Schmidt process for second vector
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    
    # Cross product for third vector
    b3 = np.cross(b1, b2)
    
    # Stack and transpose to get rotation matrix
    rot_mat = np.stack((b1, b2, b3), axis=-2).transpose(0, 2, 1)
    return rot_mat


class DMG_env_switchable:
    def __init__(self, task_name, env_configuration = "parallel", robots = ["Panda", "Panda"], \
                 cam_names = ["agentview", "birdview", "frontview"],\
                  W = 128, H = 128, controller_name = "OSC_POSE", abs_action = False,
                  postprocess_visual_obs = True):
        
        self.evaluate_fn = None
        self.inference_fn = None
        
        self.max_timesteps = 1000

        self.task_name = task_name
        env_name = to_camel_case(task_name)

        # robosuite version check
        self._is_v1 = (suite.__version__.split(".")[0] == "1")
        if self._is_v1:
            assert (int(suite.__version__.split(".")[1]) >= 2), "only support robosuite v0.3 and v1.2+"
        self.postprocess_visual_obs = postprocess_visual_obs

        self.options = {}
        self.options["env_name"] = env_name
        self.options["env_configuration"] = env_configuration
        self.options["robots"] = robots
        self.options["camera_names"] = cam_names
        self.options["camera_heights"] = H
        self.options["camera_widths"] = W
        self.options["has_offscreen_renderer"] = True
        self.options["use_camera_obs"] = True
        self.options["camera_depths"] = True
        self.options["camera_segmentations"] = "instance"

        default_controller_configs = self.init_controller_configs(controller_name, abs_action)
        self.options["controller_configs"] = default_controller_configs
        self.controller_configs = default_controller_configs

        # initialize the task
        self.env = suite.make(
            **self.options,
            has_renderer=True,
            ignore_done=True,
            control_freq=20,
        )
        self.reset()
        self.env.viewer.set_camera(camera_id=0)

    def reset(self, with_planning = False):
        if with_planning:
            raise NotImplementedError("Planning is not implemented yet")
        else:
            raw_obs = self.env.reset()
            self.obs = self.get_observation(raw_obs)
            init_ts = ts_tuple(self.obs, 0, False, {})
        return init_ts
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self.obs = self.get_observation(raw_obs)
        return ts_tuple(self.obs, reward, done, info)

    ## env_robosuite.
    def get_observation(self, di = None):
        if ObsUtils.OBS_KEYS_TO_MODALITIES is None:
            return di
        if di is None:
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
        ret = {}
        for k in di:
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                ret[k] = di[k][::-1]
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
            elif (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="depth"):
                ret[k] = di[k][::-1]
                if len(ret[k].shape) == 2:
                    ret[k] = ret[k][..., None] # (H, W, 1)
                assert len(ret[k].shape) == 3 
                # scale entries in depth map to correspond to real distance.
                ret[k] = self.get_real_depth_map(ret[k])
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
            elif "object" in k:
                ret[k] = np.array(di[k])

        if self._is_v1:
            for robot in self.env.robots:
                # add all robot-arm-specific observations. Note the (k not in ret) check
                # ensures that we don't accidentally add robot wrist images a second time
                pf = robot.robot_model.naming_prefix
                for k in di:
                    if k.startswith(pf) and (k not in ret) and (not k.endswith("proprio-state")):
                        ret[k] = np.array(di[k])

            # add in all frame-centric observations if present
            if hasattr(self.env, "_frame_centric_observable_names"):
                for fc_k in self.env._frame_centric_observable_names:
                    ret[fc_k] = np.array(di[fc_k])
        else:
            # minimal proprioception for older versions of robosuite
            ret["proprio"] = np.array(di["robot-state"])
            ret["eef_pos"] = np.array(di["eef_pos"])
            ret["eef_quat"] = np.array(di["eef_quat"])
            ret["gripper_qpos"] = np.array(di["gripper_qpos"])

        return ret

    def init_controller_configs(self, controller_name="OSC_POSE", abs_action=False):
        # config the abs joint position controller 
        controller_json_path = '/home/user/yzchen_ws/imitation_learning/robosuite/robosuite/controllers/config/default/parts/joint_position_absolute.json'
        self.abs_joint_controller_config = suite.load_part_controller_config(custom_fpath=controller_json_path)

        # config the OSC_POSE controller, input_type is delta by default
        self.relative_osc_pose_controller_config = suite.load_part_controller_config(
            default_controller="OSC_POSE"
        )

        # config the abs OSC_POSE controller
        self.absolute_osc_pose_controller_config = suite.load_part_controller_config(
            default_controller="OSC_POSE"
        )
        self.absolute_osc_pose_controller_config["input_type"] = "absolute"  # use absolute actions

        default_controller_configs = self.update_controller_configs(
            controller_name=controller_name, abs_action=abs_action
        )
        return default_controller_configs


    def update_controller_configs(self, controller_name="OSC_POSE", abs_action=False):
        """Update the robot's controller configuration.
        
        Args:
            controller_name (str): Name of the controller type to use (e.g. "OSC_POSE", "OSC_POSITION", etc.)
            abs_action (bool): If True, use absolute actions for OSC_POSE controller. If False, use delta actions.
        """
        self.controller_name = controller_name
        self.abs_action = abs_action

        # Load the base controller config for the specified controller type
        if controller_name == "OSC_POSE":
            if abs_action:
                arm_controller_config = self.absolute_osc_pose_controller_config
            else:
                arm_controller_config = self.relative_osc_pose_controller_config
        elif controller_name == "JOINT_POSITION":
            arm_controller_config = self.abs_joint_controller_config
        else:
            raise ValueError(f"Unsupported controller name: {controller_name}")
        
        robot = self.options["robots"][0] if isinstance(self.options["robots"], list) else self.options["robots"]

        # Convert to composite controller config format
        updated_controller_configs = refactor_composite_controller_config(
            arm_controller_config, 
            robot, 
            ["right", "left"]
        )
        updated_controller_configs["type"] = "SWITCHABLE"

        return updated_controller_configs
    
    def update_controllers(self, controller_name="OSC_POSE", abs_action=False):

        self.controller_configs = self.update_controller_configs(
            controller_name=controller_name, abs_action=abs_action
        )
        ## do partial reset following _reset_internal()
        # self.env._action_dim = 0
        for robot in self.env.robots:
            # Get the switchable controller instance
            controller = robot.composite_controller
            
            # Create a unique name for this configuration
            config_name = f"{controller_name}_{'abs' if abs_action else 'delta'}"
            
            # Add or update the configuration
            controller.add_configuration(
                name=config_name,
                part_controller_config=self.controller_configs["body_parts"],
                composite_controller_specific_config=self.controller_configs
            )
            
            # Switch to the new configuration
            controller.switch_configuration(config_name)
        
        self.env.reset_controller(self.controller_configs)
        # Log the change
        print(
            f"Switched to {controller_name} controller with {'absolute' if abs_action else 'delta'} actions"
        )


    def organize_equibot_obs(self, obs):
        equibot_obs = dict()

        ## get pc for each obj, then merge them
        pc_dict = self.save_mj_obsevation(npz_path=None)
        related_objs = self.equi_cfg.data.dataset.related_objs
        all_pc = np.concatenate(
            [pc_dict[obj] for obj in related_objs], axis=0
        )

        def down_sample_pc(pc):
            num_points = self.equi_cfg.data.dataset.num_points
            if pc.shape[0] <= num_points:
                return pc
            choice = np.random.choice(pc.shape[0], num_points, replace=False)
            return pc[choice]

        equibot_obs['pc'] = down_sample_pc(all_pc)

        ## get eef_pos
        eef_states = {}
        gripper_states = {}
        related_robots = self.equi_cfg.data.dataset.related_robots
        num_eef = len(related_robots)
        for robot_name in related_robots:
            biop_eef_pose = obs[f"{robot_name}_eef_pos"]
            biop_eef_quat = obs[f"{robot_name}_eef_quat"]
            eef_state = compose_transformation(biop_eef_pose, biop_eef_quat)
            eef_states[robot_name] = eef_state.reshape(1, 4, 4)

            gripper_state2finger = obs[f"{robot_name}_gripper_qpos"]
            gripper_states[robot_name] = gripper_state2finger[0]


        eef_state_trans = np.concatenate([eef_states[robot_name] for robot_name in related_robots], axis=0)
        gripper_vals = np.array([gripper_states[robot_name] for robot_name in related_robots]).reshape(-1, 1)

        eef_state_3vec = eef_state_trans[:, :3, [3, 0, 1]].transpose(0, 2, 1)
        eef_state_9d = eef_state_3vec.reshape(num_eef, 9)
        gravity_vec = np.array([0, 0, -1])
        gravity_expanded = np.tile(gravity_vec, (num_eef, 1))
        eef_state_13d = np.concatenate([eef_state_9d, gravity_expanded, gripper_vals], axis=-1)

        equibot_obs['eef_pos'] = eef_state_13d
        return equibot_obs
    
    def organize_equipolicy_action(self, action):
        related_robots = self.equi_cfg.data.dataset.related_robots
        num_eef = len(related_robots)

        action_7d = action.reshape(num_eef, 7)  # gripper, relpos, relrot
        eef_gripper = action_7d[:, 0].reshape(-1, 1)  # gripper value
        eef_relpos = action_7d[:, 1:4]  # 3d position relative to the base
        eef_relaxis = action_7d[:, 4:7]  # 3d rotation axis relative to the base
        # TODO: check the code in robomimic inference
        # eef_relrpy[0] = Rotation.from_rotvec(eef_relaxis).as_euler('xyz') # convert to euler angles

        action_7d_dmg = np.concatenate((eef_relpos, np.zeros(eef_relaxis.shape), eef_gripper), axis=-1)  
        action_out = action_7d_dmg.reshape(-1)  # 7 * num_eef

        # eef_pos = action_10d[:, 1:4]
        # eef_rot6d = action_10d[:, 4:10]
        # eef_rotmat = rotation_6d_to_matrix(eef_rot6d)
        return action_out
        

    def get_equipolicy_agent(self, equi_agent, equi_cfg):
        assert self.evaluate_fn is None, "An agent is already loaded. Please call exit() before loading a new agent."
        self.equi_cfg = equi_cfg

        ## part of eval.py in equibot
        ## TODO: using DP execution logic
        def evaluate_fn(raw_obs: Dict[str, np.ndarray], **kwargs):
            ac_horizon = equi_agent.ac_horizon
            obs_horizon = equi_agent.obs_horizon

            equibot_obs = self.organize_equibot_obs(raw_obs)
            obs_history = [equibot_obs for i in range(obs_horizon)]
            done = False
            prev_reward = None
            while not done:
                # Use the agent to get the action
                agent_obs = dict()
                for k in equibot_obs.keys():
                    if k == "pc":
                        # point clouds can have different number of points
                        # so do not stack them
                        agent_obs[k] = [o[k] for o in obs_history[-obs_horizon:]]
                    else:
                        agent_obs[k] = np.stack(
                            [o[k] for o in obs_history[-obs_horizon:]]
                        )
                ## ac is the unnormalized action, while ac_dict is the raw action
                ac, ac_dict = equi_agent.eval_with_rotation(agent_obs, **kwargs)

                # take actions
                for ac_ix in range(ac_horizon):

                    agent_ac = ac[ac_ix] if len(ac.shape) > 1 else ac # 20

                    total_action = self.organize_equipolicy_action(agent_ac)
                    ts = self.step(total_action)
                    raw_obs = ts.observation
                    curr_reward = ts.reward
                    done = ts.done

                    self.env.render()

                    equibot_obs = self.organize_equibot_obs(raw_obs)
                    obs_history.append(equibot_obs)
                    # if len(obs) > obs_horizon:
                    #     obs_history = obs_history[-obs_horizon:]

                    if prev_reward is None or curr_reward > prev_reward:
                        prev_reward = curr_reward
                    if (
                        ac_dict is None
                        or done
                    ):
                        break
            metrics = {}
            return metrics
        
        def inference_once_fn(obs: Dict[str, np.ndarray], **kwargs):
            # Use the agent to get the action
            action, _ = equi_agent.inference(obs, **kwargs)
            return action
        
        self.evaluate_fn = evaluate_fn
        self.inference_fn = inference_once_fn

    def get_dppolicy_agent(self, ckpt_path: str):
        assert self.evaluate_fn is None, "An agent is already loaded. Please call exit() before loading a new agent."


    def inference(self):
        assert self.inference_once_fn is not None
        raise NotImplementedError("Inference method is not implemented in DMG_Evaluator class.")
        return self.inference_once_fn(self.obs)


    def rollout_from_biop(self, dataset_path, render=False):
        hdf5_files = [f for f in os.listdir(f"{dataset_path}/raw") if f.endswith('.hdf5')]
        random_hdf5_file = np.random.choice(hdf5_files) 
        hdf5_path = os.path.join(dataset_path, "raw", random_hdf5_file)
        demo_id = hdf5_path.split('_')[-2]
        with h5py.File(hdf5_path, 'r') as f:

            states = f[f"data/demo_{demo_id}/states"][()]

            biop_skill_info  = f['sg_info']['bimanual_0']
            biop_pre_sg = get_sg(biop_skill_info, "pre_sg")
            pre_idx_list = biop_pre_sg.graph['idx_list']
            pre_idx = np.random.choice(pre_idx_list)
            
        obs = reset_to(self.env, {"states": states[pre_idx]})

        if render:
            while True:
                self.env.render()

        return obs

    def  handle_rewards(self):
        return self.env.reward() >=1

    def exit(self):
        self.env.close()

    def replay_tamp_step(self, total_action):
        start = time.time()

        ts = self.step(total_action)
        self.env.render()
        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)

    def get_cur_jpose(self):
        cur_obs = self.env._get_observations(force_update = True)
        rbt0_jpose = cur_obs['robot0_joint_pos']
        rbt1_jpose = cur_obs['robot1_joint_pos']
        return rbt0_jpose, rbt1_jpose
    
    ## TODO: use self.get_observation() to obtain depth
    def save_mj_obsevation(self, npz_path = None, offset_dict = {}):

        pc_dict = {}
        if self.task_name == 'two_arm_three_piece_assembly':
            interested_objs = ['base', 'piece_1', 'piece_2']

        else:
            raise NotImplementedError("This task is not supported")
        
        pc_fn = get_pcd_dict_fn(
            cam_names=self.options["camera_names"],
            W=self.options["camera_widths"],
            H= self.options["camera_heights"],
            interested_objs=interested_objs,
            record_ply = False
        )
        o3d_pc_dict = pc_fn(self.env, self.obs)

        for obj_name, pcd in o3d_pc_dict.items():
            pc = np.asarray(pcd.points)
            if obj_name in offset_dict:
                pc = pc + offset_dict[obj_name]
            pc_dict[obj_name] = pc

        if npz_path is not None:
            np.savez(npz_path, **pc_dict)
        return pc_dict


    def test_controller(self, controller_name="OSC_POSE", abs_action=False):

        self.update_controllers(
            controller_name=controller_name, 
            abs_action=False
        )
        joint_dim = 7    
        # Define the pre-defined controller actions to use (action_dim, num_test_steps, test_value)
        controller_settings = {
            "OSC_POSE": [6, 6, 0.1],
            "OSC_POSITION": [3, 3, 0.1],
            "IK_POSE": [6, 6, 0.01],
            "JOINT_POSITION": [joint_dim, joint_dim, 0.5],
            "JOINT_VELOCITY": [joint_dim, joint_dim, -0.1],
            "JOINT_TORQUE": [joint_dim, joint_dim, 0.25],
        }

        # Define variables for each controller test
        action_dim = controller_settings[self.controller_name][0]
        num_test_steps = controller_settings[self.controller_name][1]
        test_value = controller_settings[self.controller_name][2]

        # Define the number of timesteps to use per controller action as well as timesteps in between actions
        steps_per_action = 75
        steps_per_rest = 75


        # To accommodate for multi-arm settings (e.g.: Baxter), we need to make sure to fill any extra action space
        # Get total number of arms being controlled
        n = 0
        gripper_dim = 0
        for robot in self.env.robots:
            gripper_dim = robot.gripper["right"].dof
            n += int(robot.action_dim / (action_dim + gripper_dim))  # OSC: 7, 6, 1 ; JOINT: 8, 7, 1

        neutral = np.zeros(action_dim + gripper_dim)
        
        count = 0
        # Loop through controller space
        while count < num_test_steps:
            action = neutral.copy()
            for i in range(steps_per_action):
                start = time.time()

                action[count] = test_value
                total_action = np.tile(action, n)
                ts = self.step(total_action)
                self.env.render()

                # limit frame rate if necessary
                elapsed = time.time() - start
                diff = 1 / MAX_FR - elapsed
                if diff > 0:
                    time.sleep(diff)
            for i in range(steps_per_rest):
                start = time.time()
                total_action = np.tile(neutral, n)
                ts = self.step(total_action)
                self.env.render()

                # limit frame rate if necessary
                elapsed = time.time() - start
                diff = 1 / MAX_FR - elapsed
                if diff > 0:
                    time.sleep(diff)
            count += 1


if __name__ == "__main__":

    dmg_wrapper = DMG_env_runner("two_arm_three_piece_assembly", controller_name = "JOINT_POSITION", abs_action = True)
    dmg_wrapper.test_controller(controller_name="OSC_POSE", abs_action=False)