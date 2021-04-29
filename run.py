import argparse
import time
import gym
import numpy as np
import torch
from sac_agent import soft_actor_critic_agent
from replay_memory import ReplayMemory
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''
Implementation of soft actor critic, dual Q network version 
Original paper: https://arxiv.org/abs/1801.01290
Not the author's implementation !
https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms/tree/master/BipedalWalker-Soft-Actor-Critic
'''



def sac_train(args,max_steps,env,agent,memory):

    total_numsteps = 0
    updates = 0
    time_start = time.time()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = [] 
    episodes_array = []
    
    for i_episode in range(args.iteration): 
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        for step in range(max_steps):    
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Update parameters of all the networks
                agent.update_parameters(memory, args.batch_size, updates)
                updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            if reward == -100:
                reward = -1
            reward = reward * 10

            if i_episode % args.rd_intl == 0 and args.render:
                env.render()
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            
            if done:
                break

        scores_deque.append(episode_reward)
        scores_array.append(episode_reward)        
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        episodes_array.append(i_episode)

        if i_episode % 50 == 0 and i_episode >0:
            agent.save_model(args.directory, str(i_episode))
            plt.plot(episodes_array,scores_array,avg_scores_array)
            plt.ylabel("Reward")
            plt.savefig('/home/zxy/Desktop/SAC/reward.jpg')

        s =  (int)(time.time() - time_start)
            
        print("Ep.: {}, Total Steps: {}, Ep.Steps: {}, Score: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02}".\
            format(i_episode, total_numsteps, episode_steps, episode_reward, avg_score, \
                  s//3600, s%3600//60, s%60))

                    
        if (avg_score > 3000.5):
            print('Solved environment with Avg Score:  ', avg_score)
            break;
            
    return scores_array, avg_scores_array 

def sac_test(args,steps,env,agent):
    
    state = env.reset()
    scores_deque = deque(maxlen=100)
    scores = []
    
    for i_episode in range(steps + 1):
        
        state = env.reset()
        score = 0                    
        time_start = time.time()
        
        while True:
            
            action = agent.select_action(state, eval=True)
            env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward 
            state = next_state
    
            if done:
                break
                
        s = (int)(time.time() - time_start)
        scores_deque.append(score)
        scores.append(score)    
        
        print('Episode {}\tAverage Score: {:.2f},\tScore: {:.2f} \tTime: {:02}:{:02}:{:02}'\
                  .format(i_episode, np.mean(scores_deque), score, s//3600, s%3600//60, s%60))



if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name",       default = "BipedalWalkerHardcore-v3", type = str,   help = 'OpenAI gym environment name')
    parser.add_argument('--capacity',       default = 1000000,            type = int,   help = ' Size of replay buffer') 
    parser.add_argument('--iteration',      default = 100000,             type = int,   help = 'Num of episodes')
    parser.add_argument('--batch_size',     default = 256,                type = int,   help = 'Mini batch size') 
    parser.add_argument('--seed',           default = 0,                  type = int,   help = 'Random seed number')
    parser.add_argument('--learning_rate',  default = 0.00008,            type = float, help = 'Learning rate of optimizer')
    parser.add_argument('--gamma',          default = 0.99,               type = float, help = 'Discount factor')
    parser.add_argument('--hidden_size',    default = 256,                type = int,   help = 'Hidden size of net')    
    parser.add_argument('--alpha',          default = 0.2,                type = float, help = 'Temperature parameter α determines the relative importance of the entropy term against the reward')
    parser.add_argument('--tau',            default = 0.005,              type = float, help = 'Target smoothing coefficient(τ)')
    parser.add_argument('--start_steps',    default = 10000,              type = int,   help = 'Steps sampling random actions')
    parser.add_argument('--rd_intl',        default = 20,                 type = int,   help = 'Interval of render')
    parser.add_argument('--render',         default = True,               type = bool,  help = 'Shoe UI animation') 
    parser.add_argument('--train',          default = True,               type = bool,  help = 'Train the model') 
    parser.add_argument('--eval',           default = False,              type = bool,  help = 'Evaluate the model') 
    parser.add_argument('--load',           default = False,              type = bool,  help = 'Load trained model') 
    parser.add_argument('--directory',      default = 'models',           type = str,   help = 'Directory for saving actor-critic model') 

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.env_name)
    env.seed(args.seed)
    max_steps = env._max_episode_steps

    agent = soft_actor_critic_agent(env.observation_space.shape[0], env.action_space, \
        device=device, hidden_size=args.hidden_size, lr=args.learning_rate, gamma=args.gamma, tau=args.tau, alpha=args.alpha)
    memory = ReplayMemory(args.capacity)

    if args.load:
        agent.load_model(args.directory, 'final')
    if args.train:
        sac_train(args,max_steps,env,agent,memory)
        agent.save_model(args.directory,'final')
    if args.eval:
        steps = 5
        sac_test(args,steps,env,agent)
