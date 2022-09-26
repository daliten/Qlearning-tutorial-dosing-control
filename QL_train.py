import matplotlib
from QL_env import Environment
import matplotlib.pyplot as plt
import numpy as np
import math
from tensorboardX import SummaryWriter


alpha = 0.0001
gamma = 1.

def tomathformat(list):
    new_str = str(list).replace('[','{').replace(']','}')
    new_str = new_str.replace('e-01','*10^(-1)').replace('e-02','*10^(-2)').replace('e-03','*10^(-3)').replace('e-04','*10^(-4)').replace('e-05','*10^(-5)').replace('e-06','*10^(-6)').replace('e-07','*10^(-7)')
    return new_str

TEST_EPISODES = 300000
if __name__ == "__main__":
    writer = SummaryWriter() 
    output_file =  "training_stats.txt"
    f=open(output_file, "w+")
    f.write("{")
    
    rew_record = []
    e_record = []
    last_states = []
    tot_dosages = []
    
   # Q_tab_output = TemporaryFile()
    states = [i for i in range(0,10)]
    print(states)
    #actions = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    actions = [0.,0.3,0.6,0.9]
    #Q_table=np.random.uniform(0,1,(len(states),len(actions)))
    Q_table=np.zeros((len(states),len(actions)))
    #Q_table = np.load('Q_tab_output_p3_.npy')
    max_action = max(actions)
    max_state = max(states)
    #print(max_action)
    
    
   # for i in range(len(actions)):
    #    Q_table[0][i]=0
    #for i in range(len(states)):
    #    Q_table[i][-1]=0.0001
        
    #print(Q_table)
    #print(np.argmax(Q_table[1]))
    #print(np.argmax(Q_table[2]))
    env = Environment()
    #eps = 0.1
    Q_loss_tot_batch = 0.
    Q_loss_count_batch = 0
    rew_batch = 0.
    for e in range(TEST_EPISODES):
        
        #print('episode ', e)
        is_done = False
        state = env.reset() 
        state_seq = [state]

        action_ind = 0
        overall_time = 0.
        tot_rew = 0.
        tot_dosage = 0.

        if (0<=e<10000):
            eps = 0.8
        elif (10000<=e<30000):
            eps = 0.5
        elif (30000<=e<50000):
            eps = 0.3
        elif (e>=50000):
            eps = 0.1

        
        Q_loss_tot = 0.
        Q_loss_count = 0
        tot_rew_batch = 0
        rew_batch_count = 0
        while not is_done:
            #select action
            if (np.random.rand()<=eps):#select randomly            
                action_ind = np.random.randint(0,len(actions))

            else:
                action_ind = np.argmax(Q_table[state])
            
            action = actions[action_ind]
            is_done,rew,n_state,overall_time = env.step(action,overall_time,max_action,max_state)
            state_seq.append(n_state)
            cur_Q = Q_table[state,action_ind]
            next_Q_max = np.max(Q_table[n_state])
            if (is_done):
                Q_updated = cur_Q+alpha*(rew-cur_Q)
            else:
                Q_updated = cur_Q + alpha*(rew+gamma*next_Q_max-cur_Q)
            Q_loss = Q_updated - cur_Q
            Q_loss_tot += Q_loss
            Q_loss_count += 1
            Q_loss_tot_batch += Q_loss
            Q_loss_count_batch += 1

            Q_table[state][action_ind] = Q_updated
            tot_rew += rew
            tot_dosage += action
            tot_rew_batch += tot_rew
            rew_batch_count += 1
            state = n_state
        rew_batch += rew
        Q_loss_epis_ave = Q_loss_tot / Q_loss_count
        if (e%100 == 0):
            print('episode ' ,e)
            print('states = ',state_seq)
            print('cumulative reward = ',tot_rew)
            rew_record.append(tot_rew)
            e_record.append(e)
            last_states.append(n_state)
            tot_dosages.append(tot_dosage)
            Q_loss_batch_ave = abs(Q_loss_tot_batch / Q_loss_count_batch)
            writer.add_scalar("Q_loss_batch",Q_loss_batch_ave,e)
            writer.add_scalar("cumulative_reward_batch",tot_rew_batch/rew_batch_count,e)
            Q_loss_tot_batch = 0.
            Q_loss_count_batch = 0
            tot_rew_batch = 0.
            rew_batch_count = 0
        if (e%10000 == 0):
            e_now = e
            np.save('Q_tab_output%.0f'%e_now,Q_table)
            
    
            
    np.save('Q_tab_output',Q_table)
    np.save('episodes',np.array(e_record))
    np.save('tot_rew',np.array(rew_record))
    np.save('last_states',np.array(last_states))
    np.save('tot_dosages',np.array(tot_dosages))
    f.write(tomathformat(e_record))
    f.write(',')
    f.write(tomathformat(rew_record))
    f.write(',')
    f.write(tomathformat(last_states))
    f.write(',')
    f.write(tomathformat(tot_dosages))
    f.write("}")
    f.close()
    #fig = plt.figure(e_record, rew_record)
    #fig.savefig('cumulative_rew_vs_e.pdf')
    plt.plot(e_record, rew_record)
    plt.xlabel('episode')
    plt.ylabel('episodic cumulative reward')
    plt.grid()        
    
    plt.savefig('cumulative_rew_vs_e.pdf')
    plt.show()
    
    writer.close() 

    
            
