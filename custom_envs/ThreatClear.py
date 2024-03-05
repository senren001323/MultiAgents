import math
import numpy as np
import pybullet as p
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary


class ThreatClear(BaseMultiagentAviary):

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=3,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=120,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM):
        """
        Parameters:
         -dis_matrix: Symmetric matrix
             Store the distance between drones
         -collision_buffer: float
             Collision buffer region, for computing reward
         -obs_radius: float
             The observation radius of each drone
         -threat_points: float
             array[num_drones - 1, 3]
             Randomly generated threat points
         -drone_done: bool
             array[num_drones, ]
             Whether the drone done(reached one of or all threat_points)
         -protect_points: array[1, 3]
             The protect point
        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act)
        self.dis_matrix = np.zeros((self.NUM_DRONES, self.NUM_DRONES))
        self.collision_buffer = 8 * self.L # L:0.039700
        self.obs_radius = 1.0
        self.drone_done = np.zeros(self.NUM_DRONES, dtype=bool)
        self.threat_clear = np.zeros(self.NUM_DRONES-1, dtype=bool)
        self.threat_points = np.array([[1.5, 1.0, 1.0],
                                       [1.5, -1.0, 1.0]], dtype=np.float64)
        self.protect_point = np.array([3.0, 0.0, 1.0], dtype=np.float64)
        self.lamda = 0.0003
        self.INIT_XYZS = np.array([[0.0, 0.0, 0.1125],
                                   [0.0, -0.5, 0.1125],
                                   [0.0, -1.0, 0.1125]])

    def reset(self):
        super().reset()
        self.drone_done = np.zeros(self.NUM_DRONES, dtype=bool)
        self.threat_clear = np.zeros(self.NUM_DRONES-1, dtype=bool)
        self.threat_points = np.array([[1.5, 1.0, 1.0],
                                       [1.5, -1.0, 1.0]], dtype=np.float64)
        self.protect_point = np.array([3.0, 0.0, 1.0], dtype=np.float64)
        return self._computeObs()

    def step(self, actions):
        """
        In the each step, 1.target points random walk 2.reset targets occupancy 3.assign target 
        to each drone.
        Update distance matrix for computing collision rewards
        """
        super().step(actions)
        self.threat_walk()
        self.drone_reached()
        return self._computeObs(), self._computeReward(), self._computeDone(), self._computeInfo()

    def _computeObs(self):
        """Extend from super class
               Calculate whether any targets within the radius of the UAV
         -NO_OBS: Special mark, if no target within the radius
         
         Return:
         -obs: Dict[int, array]
         Normalized extended observation of each UAV, including other drones and threat points
         ————Example: {0: [obs, drone1's pos, NO_OBS, threat_point0, ...],
                       1: [...],
                       ...}
        """
        NO_OBS = [-1, -1, -1]
        
        obs = super()._computeObs()
        for i in range(self.NUM_DRONES):
            drone_pos = self._getDroneStateVector(i)[:3]
            for j in range(self.NUM_DRONES):
                if i != j:
                    other_dronepos = self._getDroneStateVector(j)[:3]
                    if np.linalg.norm(drone_pos - other_dronepos) <= self.obs_radius:
                        other_dronepos[:2] = np.clip(other_dronepos[:2], -5, 5) / 5
                        other_dronepos[2] = np.clip(other_dronepos[2], 0, 5) / 5
                        obs[i] = np.hstack((obs[i], other_dronepos))
                    else:
                        obs[i] = np.hstack((obs[i], NO_OBS))
                        
            for threat_point in self.threat_points:
                if np.linalg.norm(drone_pos - threat_point) <= self.obs_radius:
                    threat_point[:2] = np.clip(threat_point[:2], -5, 5) / 5
                    threat_point[2] = np.clip(threat_point[2], 0, 5) / 5
                    obs[i] = np.hstack((obs[i], threat_point))
                else:
                    obs[i] = np.hstack((obs[i], NO_OBS))
        return obs

    def _observationSpace(self):
        """
        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape(12+3*2+3*2,)
        """
        return spaces.Dict({i: spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,0, -1,-1,0, -1,-1,0, -1,-1,0]),
                                          high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1,]),
                                          dtype=np.float32
                                          ) for i in range(self.NUM_DRONES)})

    def _computeReward(self):
        """Compute rewards
        reward 1: given by the nearest threat target for each drones
        reward 2: the self destruction reward
        reward 3: the threat destruction reward
        reward 4: survival drone reward
        reward 5: main task reward
        
        Returns:
         -rewards: dict[int, float]
              The reward value for each drones
        """
        reward_1 = np.zeros(self.NUM_DRONES)
        max_dis = np.linalg.norm(np.array([5, 5, 5]) - np.array([-5, -5, 0]))
        for i in range(self.NUM_DRONES):
            drone_pos = self._getDroneStateVector(i)[:3]
            min_dis = min(np.linalg.norm(drone_pos - self.threat_points, axis=1))
            reward_1[i] += self.lamda * 1/(min_dis) # scaling

        reward_2 = np.zeros(self.NUM_DRONES)
        for i in range(self.NUM_DRONES):
            drone_pos = self._getDroneStateVector(i)[:3]
            if any(np.linalg.norm(drone_pos - self.threat_points, axis=1) < 0.1):
                reward_2[i] += -1

        reward_3 = 0
        for i in range(self.NUM_DRONES):
            drone_pos = self._getDroneStateVector(i)[:3]
            for j, threat_pos in enumerate(self.threat_points):
                if not self.threat_clear[j] and np.linalg.norm(drone_pos - threat_pos) < 0.1:
                        reward_3 = 3
                        self.threat_clear[j] = True
        
        reward_4 = np.zeros(self.NUM_DRONES)
        if all(self.threat_clear) and any(self.drone_done):
            for i in range(self.NUM_DRONES):
                if not self.drone_done[i]:
                    reward_4[i] = 0.01
                else:
                    reward_4[i] = 0
        
        reward_5 = 0
        if (self.step_counter / self.SIM_FREQ == self.EPISODE_LEN_SEC) or \
            all(self.drone_done): # if all drones reached together 
            if all(self.threat_clear):
                reward_5 = 10
            else: 
                reward_5 = 0
                
        total = reward_1 + reward_2 + reward_3 + reward_4 + reward_5
        rewards = {i: total[i] for i in range(self.NUM_DRONES)}
        return rewards

    def _computeDone(self):
        """
        1.Reached Target
        2.Timelimit: 1 second with 240 steps, 1200 steps in 5 seconds
        3.Out of bound: Done when drone cross the setting border
        Return:
         -dict[int | "__all__", bool]
        """    
        done_outbound = np.zeros(self.NUM_DRONES, dtype=bool)
        for i in range(self.NUM_DRONES):
            drone_pos = self._getDroneStateVector(i)[:3]
            if abs(drone_pos[0]) > 3 or abs(drone_pos[1]) > 3 or drone_pos[2] > 3:
                done_outbound[i] = True
        done_time = True if self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC else False
        
        done = {i: self.drone_done[i] or done_outbound[i] or done_time for i in range(self.NUM_DRONES)}
        done["__all__"] = np.all(self.drone_done) or done_time or np.all(done_outbound)
        return done
        
    def _computeInfo(self):
        """Unused.
        Returns:
        dict[int, dict[]]
            Dictionary of empty dictionaries.
        """
        return {i: {} for i in range(self.NUM_DRONES)}

    def threat_walk(self):
        """Threat points moves to the protection point
         -Uniformly move to the protection points in all time steps
         -When the drone reached destruction radius, threat points stop moving
             note: might multiple drones reached destruction radius

        Update:
         -self.threat_points: Position of each point
         -self.threat_clear: Whether threat points have been cleared. True or False, bool array
        """
        for i in range(self.NUM_DRONES - 1):
            destroyed = False
            for j in range(self.NUM_DRONES):  
                if np.linalg.norm(self.threat_points[i] - self._getDroneStateVector(j)[:3]) < 0.1:
                    destroyed = True
                    self.threat_clear[i] = True # mark the state of threat
                    break
            if not destroyed:
                dis_protect = np.linalg.norm(self.threat_points[i] - self.protect_point)
                self.threat_points[i] += dis_protect / (self.SIM_FREQ * self.EPISODE_LEN_SEC)

    def drone_reached(self):
        """Reached Target: Done when drone reached any threat targets.

        Update:
         -self.drone_done: 
        """
        for i in range(self.NUM_DRONES):
            drone_pos = self._getDroneStateVector(i)[:3]
            for j in range(self.NUM_DRONES-1): 
                if np.linalg.norm(drone_pos - self.threat_points[j]) <= 0.1:
                    self.drone_done[i] = True
                    break # while curr drone reached any threat point
            else:
                self.drone_done[i] = False

    def update_dismatrix(self):
        """
        Maintain the distance matrix between drones
        """
        for i in range(self.NUM_DRONES):
            for j in range(i+1, self.NUM_DRONES):
                pos_i = self._getDroneStateVector(i)[:3]
                pos_j = self._getDroneStateVector(j)[:3]
                dis_two_drones = np.linalg.norm(pos_i - pos_j)
                self.dis_matrix[i][j] = self.dis_matrix[j][i] = dis_two_drones
                
    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.
        Parameters:
         -state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.
        Returns:
         -ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.
        """
        MAX_LIN_VEL_XY = 1
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z)

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
                
    def _clipAndNormalizeStateWarning(self, state, 
                                      clipped_pos_xy, clipped_pos_z, clipped_rp, clipped_vel_xy, clipped_vel_z):
        """
        Print a warning if values in a state vector is out of the clipping range.
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))






