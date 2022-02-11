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
LEN_GOAL_POSITION           = 4         # Data Byte Length
ADDR_PRESENT_POSITION       = 132
LEN_PRESENT_POSITION        = 4         # Data Byte Length
DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
BAUDRATE                    = 115200
PROTOCOL_VERSION            = 2.0
DEVICENAME                  = '/dev/ttyUSB0'
TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
DXL_IDS                     = [10,20,30]

class WalkBot_Hard(JetTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.dt = self.cfg["sim"]["dt"]
        self.cfg["env"]["numObservations"] = 15
        self.cfg["env"]["numActions"] = 3
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        self.init_servos()

        
    def pre_physics_step(self, actions):
        # print(actions)
        pos_obs = self.read_positions()
        self.obs_buf = torch.rand((self.num_envs, self.num_observations), device=self.device)
        for pos, i in zip(pos_obs, range(len(pos_obs))):
            self.obs_buf[:,i] = pos
        print(self.obs_buf)
        

    def post_physics_step(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        # print('Resetting IDX! Env_IDs = {}'.format(env_ids))
        self.reset_buf[env_ids] = 0

    def init_servos(self):
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        # Initialize GroupSyncWrite instance
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        # Initialize GroupSyncRead instace for Present Position
        self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
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
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel#%d has been successfully connected" % dxl_id)
    
    def disable_torque(self):
        # Disable Dynamixel Torque
        for dxl_id in DXL_IDS:
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel#%d has been successfully connected" % dxl_id)

    def deinit_servos(self):
        self.disable_torque()
        self.portHandler.closePort()

    def read_positions(self):
        pos_list = []
        #Sync Read Present Position
        dxl_comm_result = self.groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
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

    def write_positions(self, servo_pos_tuple):
        for dxl_id, pos in servo_pos_tuple:
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(pos)), DXL_HIBYTE(DXL_LOWORD(pos)), DXL_LOBYTE(DXL_HIWORD(pos)), DXL_HIBYTE(DXL_HIWORD(pos))]
            # Add Dynamixel#1 goal position value to the Syncwrite parameter storage
            dxl_addparam_result = groupSyncWrite.addParam(dxl_id, param_goal_position)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % DXL1_ID)
                quit()

        # Syncwrite goal position
        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        groupSyncWrite.clearParam()
