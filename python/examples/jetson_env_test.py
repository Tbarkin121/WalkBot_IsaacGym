import walkbot_testclass
import time
import numpy as np

walkbot = walkbot_testclass.WalkBot_Hard()
walkbot.enable_torque()

servo_list = walkbot_testclass.DXL_IDS

steps = 100
for i in range (steps):   
    action_list = [i/(steps*2), i/(steps*2), i/(steps*2)]
    walkbot.write_positions(servo_list, action_list)
    state = np.array(walkbot.read_state())
    print('Pos : {:0.2f}, {:0.2f}, {:0.2f} Vel :  {:0.2f}, {:0.2f}, {:0.2f}'.format(state[0,0],state[0,1],state[0,2],state[1,0],state[1,1],state[1,2]))

for i in range (steps*2):   
    action_list = [0.5 - i/(steps*2), 0.5 - i/(steps*2), 0.5 - i/(steps*2)]
    walkbot.write_positions(servo_list, action_list)
    state = np.array(walkbot.read_state())
    print('Pos : {:0.2f}, {:0.2f}, {:0.2f} Vel :  {:0.2f}, {:0.2f}, {:0.2f}'.format(state[0,0],state[0,1],state[0,2],state[1,0],state[1,1],state[1,2]))

for i in range (steps):   
    action_list = [i/(steps*2) - 0.5, i/(steps*2) - 0.5, i/(steps*2) - 0.5]
    walkbot.write_positions(servo_list, action_list)
    state = np.array(walkbot.read_state())
    print('Pos : {:0.2f}, {:0.2f}, {:0.2f} Vel :  {:0.2f}, {:0.2f}, {:0.2f}'.format(state[0,0],state[0,1],state[0,2],state[1,0],state[1,1],state[1,2]))
    
walkbot.disable_torque()