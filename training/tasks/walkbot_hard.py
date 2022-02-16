import gym
from gym import spaces
import numpy as np
from tasks.base.jet_task import JetTask
import torch
# import yaml
# from rl_games.torch_runner import Runner
# import os
# from collections import deque
from dynamixel_sdk import *  


ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_POSITION          = 116
ADDR_PRESENT_VELOCITY       = 128
ADDR_PRESENT_POSITION       = 132
LEN_GOAL_POSITION           = 4         # Data Byte Length
LEN_PRESENT_VELOCITY        = 4         # Data Byte Length
LEN_PRESENT_POSITION        = 4         # Data Byte Length
DXL_MINIMUM_POSITION_VALUE  = 1023         
DXL_MAXIMUM_POSITION_VALUE  = 3073      
BAUDRATE                    = 115200
PROTOCOL_VERSION            = 2.0
DEVICENAME                  = '/dev/ttyUSB0'
TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
DXL_IDS                     = [10,20,30]
SERVO_HOME_POSITION         = 2048              # This is the encoder count for home (180 deg is when the leg is held straight out)
SERVO_MAX_POSITION          = 1024              # This is how far it can move from home
SERVO_MAX_VELOCITY          = 363               # In rev/min
class WalkBot_Hard(JetTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.dt = self.cfg["sim"]["dt"]
        self.cfg["env"]["numObservations"] = 12
        self.cfg["env"]["numActions"] = 3
        self.goal_offset = self.cfg["env"]["goalOffset"]
        self.goal_range = self.cfg["env"]["goalRange"]
        self.goal_pos = torch.tensor([0.5, 0.5, 0.5])
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        self.init_servos()
        self.enable_torque()
        self.loop_count = torch.tensor([0], device=self.device)
        
    def pre_physics_step(self, actions):
        # self.find_foot_coordinates()
        action_list = [actions[0,0], actions[0,1], actions[0,2]]
        goal_pos = torch.tensor([torch.cos(self.loop_count*0.01), torch.cos(self.loop_count*0.01), torch.cos(self.loop_count*0.01)], device=self.device)
        leg_state = self.read_state()
        self.obs_buf = torch.rand((self.num_envs, self.num_observations), device=self.device)
        self.obs_buf[0,0:3] = leg_state[0,:] # Positions
        self.obs_buf[0,3:6] = leg_state[1,:] # Velocities
        self.obs_buf[0,6:9] = goal_pos
        # self.obs_buf[0,9:12] = torch.div(self.foot_pos  - self.goal_offset, self.goal_range)
        self.obs_buf[0,9:12] = actions
        self.write_positions(DXL_IDS, action_list)
        # print(self.obs_buf)
        print(actions)
        self.loop_count += 1
        # if(self.loop_count > 100):
            # self.disable_torque()
        

    def post_physics_step(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        # print('Resetting IDX! Env_IDs = {}'.format(env_ids))
        self.reset_buf[env_ids] = 0

    def find_foot_coordinates(self):       
        pass


    def init_servos(self):
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        # Initialize GroupSyncWrite instance
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        # Initialize GroupSyncRead instace for Present Position
        self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_VELOCITY, LEN_PRESENT_POSITION + LEN_PRESENT_VELOCITY)
        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()
        # Set port baudrate
        if self.portHandler.setBaudRate(BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")

        for dxl_id in DXL_IDS:
            # Add parameter storage for Dynamixel present position value
            dxl_addparam_result = self.groupSyncRead.addParam(dxl_id)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncRead addparam failed" % DXL1_ID)
                quit()

    def enable_torque(self):
        # Enable Dynamixel Torque
        for dxl_id in DXL_IDS:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel#%d torque enabled" % dxl_id)
    
    def disable_torque(self):
        # Disable Dynamixel Torque
        for dxl_id in DXL_IDS:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel#%d torque disabled" % dxl_id)

    def deinit_servos(self):
        self.disable_torque()
        self.portHandler.closePort()

    def read_positions(self):
        #Sync Read Present Position and Velocity
        dxl_comm_result = self.groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        pos_list = []
        for dxl_id in DXL_IDS:
            dxl_getdata_result = self.groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupSyncRead getdata failed" % dxl_id)
                quit()
            #Get Present Position
            dxl_present_position = self.groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            pos_list.append(dxl_present_position)

        # Clear syncread parameter storage
        # groupSyncRead.clearParam()
        return pos_list

    
    def read_vels(self):
        #Sync Read Present Position and Velocity
        dxl_comm_result = self.groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        vel_list = []
        for dxl_id in DXL_IDS:
            dxl_getdata_result = self.groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupSyncRead getdata failed" % dxl_id)
                quit()
            #Get Present Velocity
            dxl_present_velocity = self.groupSyncRead.getData(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
            if(len(bin(dxl_present_velocity))>16):
                dxl_present_velocity -= 2**32
            dxl_present_velocity *= 0.229 #Converts it to rev/min
            vel_list.append(dxl_present_velocity)

        # Clear syncread parameter storage
        # groupSyncRead.clearParam()
        return vel_list

    def read_state(self):
        #Sync Read Present Position and Velocity
        dxl_comm_result = self.groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        pos_list = []
        vel_list = []
        for dxl_id in DXL_IDS:
            dxl_getdata_result = self.groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupSyncRead getdata failed" % dxl_id)
                quit()
            #Get Present Position
            dxl_present_position = self.groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            pos_list.append( (dxl_present_position-SERVO_HOME_POSITION)/SERVO_MAX_POSITION)
            #Get Present Velocity
            dxl_present_velocity = self.groupSyncRead.getData(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
            if(len(bin(dxl_present_velocity))>16):
                dxl_present_velocity -= 2**32
            dxl_present_velocity *= 0.229 #Converts it to rev/min
            vel_list.append(dxl_present_velocity/SERVO_MAX_VELOCITY)

        # Clear syncread parameter storage
        # groupSyncRead.clearParam()
        return torch.tensor([pos_list, vel_list], device=self.device)


    def write_positions(self, servo_list, action_list):
        for dxl_id, action in zip(servo_list, action_list):
            pos = int(SERVO_HOME_POSITION + action*SERVO_MAX_POSITION)
            print('pos : {}'.format(pos))
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(pos)), DXL_HIBYTE(DXL_LOWORD(pos)), DXL_LOBYTE(DXL_HIWORD(pos)), DXL_HIBYTE(DXL_HIWORD(pos))]
            # Add Dynamixel#1 goal position value to the Syncwrite parameter storage
            dxl_addparam_result = self.groupSyncWrite.addParam(dxl_id, param_goal_position)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % DXL1_ID)
                quit()

        # Syncwrite goal position
        dxl_comm_result = self.groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        self.groupSyncWrite.clearParam()
