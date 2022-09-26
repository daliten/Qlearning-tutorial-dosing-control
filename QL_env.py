import random
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
import numpy as np

MAX_TIME = 6*24*7 #6 weeks
TIME_STEP = 24*7 #1 week
POP_THRESHOLD = 10
N_birth = 1/22
T_birth = 1/18
N_death = 1/60
T_death = 1/100
N_IC_min = 0.9
N_IC_max = 2.5
T_IC_min = 0.3
T_IC_max = 1.1
capacity = 100000
I_T_int_min = 0.0000003
I_T_int_max = 0.000001
N_T_int_min = 0.000001
N_T_int_max = 0.000005
I_D_int_min = 0.1
I_D_int_max = 0.8
#r_min = 2
#r_max = 8
r_min = 2000
r_max = 5000
#a_min = 0.05
#a_max = 0.2
a_min = 1
a_max = 1
I_influx_min = 1
I_influx_max = 2
#I_influx_min = 0
#I_influx_max = 0
D_clearance_min = 0.05
D_clearance_max = 0.15
N_init_pop_min = 600
N_init_pop_max = 1500
T_init_pop_min = 600
T_init_pop_max = 1200
#T_init_pop_max = 2000
I_init_pop_min = 100
I_init_pop_max = 300
#I_init_pop_min = 0
#I_init_pop_max = 0

minval = 0.
maxval = 1000000



def tumor_system(z, t, drug, n_ic, t_ic, i_t_int,n_t_int,i_influx, r, a, i_d_int, d_clear):
    N_c, T_c, I_c, D_c = z

    #i_influx = 0
    i_d_int = 0.
    max_I_c = 1000
    a=1

    Neqn = (N_birth/(1+D_c/(n_ic))-N_death)*N_c*(1-(N_c+T_c)/capacity)-n_t_int*N_c*T_c
    Teqn = (T_birth/(1+D_c/(t_ic))-T_death)*T_c*(1-(N_c+T_c)/capacity)-n_t_int*N_c*T_c-(i_t_int)*T_c*I_c
    Ieqn = i_influx+r*I_c*T_c/(a+T_c)*(1-I_c/max_I_c)-i_d_int*I_c*D_c
    Deqn = drug-d_clear*D_c
    return [Neqn, Teqn, Ieqn, Deqn]

class Environment:

    def __init__(self):        
        
        self.N_IC = 0.
        self.T_IC = 0.
        self.I_T_int = 0.
        self.I_D_int = 0.
        self.r = 0.
        self.a = 0.
        self.I_influx = 0
        self.D_clearance = 0.
        self.N_pop = 0
        self.T_pop = 0
        self.I_pop = 0
        self.D_conc = 0.
        
       # self.status = [0,0,0] #normal, tumor, immune
        
    def reset(self):
        
        self.N_IC = random.uniform(N_IC_min,N_IC_max)
        self.T_IC = random.uniform(T_IC_min,T_IC_max)
        self.I_T_int = random.uniform(I_T_int_min,I_T_int_max)
        self.I_D_int = random.uniform(I_D_int_min,I_D_int_max)
        self.N_T_int = random.uniform(N_T_int_min,N_T_int_max)
        #self.I_T_int = 0.
        self.r = random.uniform(r_min,r_max)
        self.a = random.uniform(a_min,a_max)
        self.I_influx = random.randint(I_influx_min,I_influx_max)
        self.D_clearance = random.uniform(D_clearance_min,D_clearance_max)
        self.N_pop = random.randint(N_init_pop_min,N_init_pop_max)
        self.T_pop = random.randint(T_init_pop_min,T_init_pop_max)
        self.I_pop = random.randint(I_init_pop_min,I_init_pop_max)
        self.D_conc = 0.
        
        self.T_init_pop = self.T_pop
        self.counter = 1
        state = self.find_state(math.log(1+np.clip(self.T_pop,0,capacity*1.5)/self.T_init_pop))  
        self.tot_dosage = 0.
        
        return state
    
    def calc_rew(self,action,n_state,max_action,max_state):
        dosing_penalty = action/max_action
       # dosing_penalty = 0
        if (n_state<=1):
            rew = 1-dosing_penalty
        elif (n_state == 2):
            rew = 0
        else:
            rew = -1


        return rew
    
    def find_state(self,log_pop_ratio):
        if (log_pop_ratio >1):
            state = math.floor(log_pop_ratio)
            state = np.clip(state,1,5)
            state_id = int(state)+2
            #print(state_id)
        elif (0.6 <=log_pop_ratio < 1):
            state_id = 2
        elif (0.2 <=log_pop_ratio < 0.6):
            state_id = 1
        elif (0 <=log_pop_ratio < 0.2):
            state_id = 0
        
        #print('state_id = ',state_id)
        return int(state_id)
    
    def step(self,action,overall_time,max_action,max_state):
        #print('starting step')
        is_done = False
        self.tot_dosage += action
        tend = TIME_STEP
        tstart = 0
        ttab=np.linspace(tstart,tend,20)
        sol = odeint(tumor_system,[self.N_pop,self.T_pop,self.I_pop,action],ttab,args=(action, self.N_IC, self.T_IC, self.I_T_int, self.N_T_int, self.I_influx, self.r, self.a, self.I_D_int, self.D_clearance))

        self.N_pop = sol[-1][0]
        self.T_pop = sol[-1][1]
        self.I_pop = sol[-1][3]
        
        n_state = self.find_state(math.log(1+np.clip(self.T_pop,0,capacity*1.5)/self.T_init_pop))  
        
        #state = math.log(1+self.T_pop/self.T_init_pop)
        reward = self.calc_rew(action,n_state,max_action,max_state)
       # print('\n')
        overall_time += (tend-tstart)
        
        if (overall_time >= MAX_TIME):
            is_done = True


        return(is_done,reward,n_state,overall_time)


        
