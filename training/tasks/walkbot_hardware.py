from dynamixel_sdk import * 

ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_POSITION          = 116
ADDR_PRESENT_VELOCITY       = 128
ADDR_PRESENT_POSITION       = 132
LEN_GOAL_POSITION           = 4         # Data Byte Length
LEN_PRESENT_VELOCITY        = 4         # Data Byte Length
LEN_PRESENT_POSITION        = 4         # Data Byte Length
DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
BAUDRATE                    = 115200
PROTOCOL_VERSION            = 2.0
DEVICENAME                  = '/dev/ttyUSB0'
TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
# DXL_IDS                     = [10,20,30]
DXL_IDS                     = [40,50,60]

SERVO_HOME_POSITION         = 2048 # In Encoder Tics
SERVO_MAX_POSITION          = 1024 # In Encoder Tics
SERVO_MAX_VELOCITY          = 363 # In rev/min
class WalkBot_Hard():
    def __init__(self):
        self.init_servos()

    
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
        return [pos_list, vel_list]


    def write_positions(self, servo_list, action_list):
        for dxl_id, action in zip(servo_list, action_list):
            pos = int(SERVO_HOME_POSITION + action*SERVO_MAX_POSITION)
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
