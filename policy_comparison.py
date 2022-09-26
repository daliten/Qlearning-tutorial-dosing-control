
import numpy as np
import matplotlib.pyplot as plt
from QL_env import Environment
import numpy as np
import random



alpha = 0.1
gamma = 0.99

TEST_EPISODES = 5000


if __name__ == "__main__":
    states = [i for i in range(0,10)]
    actions = [0.,0.3,0.6,0.9]
    Q_table=np.load('Q_tab_output.npy')
    max_action = max(actions)
    max_state = max(states)

    env = Environment()
    success_rate = 0
    fail_rate = 0
    const_rate = 0
    reduction_rate = 0
    tot_dos_success = 0
    tot_dos_red = 0
    tot_dos_const = 0
    tot_dos_fail = 0
    
    state_sums=[]
    tot_dosages = []
    above2stats = []
    above1stats=[]
    below2_tot_dosages = []
    for e in range(TEST_EPISODES):
        
        is_done = False
        state = env.reset() 
        state_sum = state
        action_ind = 0
        overall_time = 0.
        tot_dosage = 0.
        above2 = 0
        above1=1
        step_count = 0
        below2_tot_dosage = 0.
        while not is_done:

            if (state ==0):
                action = 0.3
            elif (state == 1):
                action = 0.3
                #action = 0.6 #policy 1
            elif (state == 2):
               # action = 0.6
                action = 0.3 #policy 2
            elif (state ==3):
                action = 0.9
            elif (state == 4):
                #action = 0.6
                action = 0.9
            elif (state>=5):
                action = 0.
            
                
            is_done,rew,n_state,overall_time = env.step(action,overall_time,max_action,max_state)
            if (n_state>2):
                above2+=1
            if (n_state>1):
                above1+=1
            
            if (n_state<=1):
                below2_tot_dosage += action
        
            state = n_state
            tot_dosage += action
            
            state_sum += n_state
            step_count += 1
        
        above2stats.append(above2/step_count)
        above1stats.append(above1/step_count)
        below2_tot_dosages.append(below2_tot_dosage)
        state_sums.append(state_sum)
        tot_dosages.append(tot_dosage)
        



    print('ave episode state sum = ',sum(state_sums)/len(state_sums))
    print('ave episode total dosage = ',sum(tot_dosages)/len(tot_dosages))
    print('ave episode above 2 ratio = ',sum(above2stats)/len(above2stats))
    print('ave episode above 1 ratio = ',sum(above1stats)/len(above1stats))
    print('ave episode below1_tot_dosages = ',sum(below2_tot_dosages)/len(below2_tot_dosages))

