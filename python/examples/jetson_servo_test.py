import os
import sys, tty, termios
import numpy as np

fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
def getch():
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def lerp(v0, v1, t):
    return np.int16(v0+t * (v1-v0))

from dynamixel_sdk import *                    # Uses Dynamixel SDK library

ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_POSITION          = 116
LEN_GOAL_POSITION           = 4         # Data Byte Length
ADDR_PRESENT_POSITION       = 132
ADDR_PRESENT_VELOCITY       = 128
LEN_PRESENT_POSITION        = 4         # Data Byte Length
LEN_PRESENT_VELOCITY        = 4         # Data Byte Length
DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
BAUDRATE                    = 115200

# DYNAMIXEL Protocol Version (1.0 / 2.0)
# https://emanual.robotis.com/docs/en/dxl/protocol2/
PROTOCOL_VERSION            = 2.0

# Make sure that each DYNAMIXEL ID should have unique ID.
DXL_IDS = [10, 20, 30, 40, 50, 60]
# DXL_IDS = [30,60]
DXL_LIMIT_LIST = [0,1,2,3,4,5]
# DXL_LIMIT_LIST = [2, 5]

# Use the actual port assigned to the U2D2.
# ex) Windows: "COM*", Linux: "/dev/ttyUSB*", Mac: "/dev/tty.usbserial-*"
DEVICENAME                  = '/dev/ttyUSB0'

TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
DXL_MOVING_STATUS_THRESHOLD = 20                # Dynamixel moving status threshold

index = 0
# Goal position
dxl_goal_position = np.array([[1800, 2200],
                              [1023, 3073],
                              [1023, 3073],
                              [1800, 2200],
                              [1023, 3073],
                              [1023, 3073]])

# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Set the protocol version
# Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Initialize GroupSyncWrite instance
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

# Initialize GroupSyncRead instace for Present Position
groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY+LEN_PRESENT_POSITION)
# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()


# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

# Enable Dynamixel Torque
for dxl_id in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel#%d has been successfully connected" % dxl_id)

    # Add parameter storage for Dynamixel present position value
    dxl_addparam_result = groupSyncRead.addParam(dxl_id)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncRead addparam failed" % DXL1_ID)
        quit()


steps = 100
while 1:
    print("Press any key to continue! (or press ESC to quit!)")
    if getch() == chr(0x1b):
        break

    #Move Forward
    for i in range(steps):
        #Sync Read Present Position
        dxl_comm_result = groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        pos_list = []
        vel_list = []
        for dxl_id, idx in zip(DXL_IDS, DXL_LIMIT_LIST):
            goal_position = lerp(dxl_goal_position[idx, index%2], dxl_goal_position[idx, (index+1)%2], i /steps)   
            # print(goal_position)
            # Allocate goal position value into byte array
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(goal_position)), DXL_HIBYTE(DXL_LOWORD(goal_position)), DXL_LOBYTE(DXL_HIWORD(goal_position)), DXL_HIBYTE(DXL_HIWORD(goal_position))]

            # Check if groupsyncread data of Dynamixel#1 is available
            # dxl_getdata_result_pos = groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            # dxl_getdata_result_vel = groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
            
            # if (dxl_getdata_result_pos != True or dxl_getdata_result_vel != True):
            #     print("[ID:%03d] groupSyncRead getdata failed" % dxl_id)
            #     quit()

            #Get Present Position
            dxl_present_position = groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            pos_list.append(dxl_present_position)
            dxl_present_velocity = groupSyncRead.getData(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
            if(len(bin(dxl_present_velocity))>16):
                dxl_present_velocity -= 2**32
            dxl_present_velocity *= 0.229 #Converts it to rev/min
            vel_list.append(dxl_present_velocity)
            # print("[ID:%03d] PresPos:%03d  PresVel:%03d" % (dxl_id, dxl_present_position, dxl_present_velocity))
            
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

        print('Pos : {:.2f}, {:.2f}, {:.2f}'.format(pos_list[0],pos_list[1],pos_list[2]))
        print('Vel : {:.2f}, {:.2f}, {:.2f}'.format(vel_list[0],vel_list[1],vel_list[2]))
    

    # Change goal position
    if index == 0:
        index = 1
    else:
        index = 0

# Clear syncread parameter storage
groupSyncRead.clearParam()

# Disable Dynamixel Torque
for dxl_id in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

# Close port
portHandler.closePort()
