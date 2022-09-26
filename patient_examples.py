from QL_env import Environment
import numpy as np
import random
import matplotlib.pyplot as plt
    
alpha = 0.1
gamma = 0.99

TEST_EPISODES = 1
if __name__ == "__main__":

    states = [i for i in range(0,10)]
    output_file2 = "Q_array.txt"
    f2=open(output_file2,"w+")
    actions = [0.,0.3,0.6,0.9]
    Q_table=np.load('Q_tab_output.npy')
    f2.write(tomathformat(Q_table.tolist()))
    f2.close()
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
    times_all=[]
    T_all=[]
    diff_all=[]
    actions_all=[]
    action_times_all=[]
    
    for e in range(TEST_EPISODES):
        
        is_done = False
        state = env.reset() 
        action_ind = 0
        overall_time = 0.
        #print('state = ',state)
        tot_dosage = 0.
        state_sum = state
        state_seq = [state]
        action_seq = []
        state_action_seq=[]
        T_pop_seq=[env.T_pop]
        N_pop_seq=[env.N_pop]
        cell_diff_seq = [env.N_pop-env.T_pop]
        times = [0.0]
        action_times = []
        while not is_done:
            action = 0.3
            action_times.append(overall_time)
            is_done,rew,n_state,overall_time = env.step(action,overall_time,max_action,max_state)
            
           
            state_action_seq.append([state,action])

            state_sum+=n_state
            state_seq.append(n_state)
            state = n_state
            tot_dosage += action
            action_seq.append(action)
            T_pop_seq.append(env.T_pop)
            N_pop_seq.append(env.N_pop)
            cell_diff_seq.append(env.N_pop-env.T_pop)
            times.append(overall_time)
        times_all.append(times)
        T_all.append(T_pop_seq)
        diff_all.append(cell_diff_seq)
        actions_all.append(action_seq)
        action_times_all.append(action_times)
            
    plt.figure(1)
    for i in range(len(diff_all)):
        plt.plot(times_all[i], diff_all[i],'black')
    plt.xlabel('Time')
    plt.ylabel('Cells')
    plt.yscale('symlog')
    plt.grid()   
    plt.figure(2)
    for i in range(len(T_all)):
        plt.plot(times_all[i], T_all[i],'r')
    plt.xlabel('Time')
    plt.ylabel('Cells')
    plt.grid() 
    plt.figure(3)
    for i in range(len(actions_all)):
        plt.plot(action_times_all[i],actions_all[i])
    plt.xlabel('Drug concentration')
    plt.ylabel('Time')
    plt.grid() 
    
    plt.show()


            
