from maze_env import Maze
from RL_brain import DeepQNetwork

def run_maze():
    step=0
    for episode in range(300):
        observation=env.reset()
        while True:
            env.render()

            action=RL.choose_action(observation)

            observation_,reward,done=ev.step(action)

            RL.store_transition(observation,action,reward,observation_)

            if (step>200) and (step %5==0):
                RL.learn()
            observation=observation_
            if done:
                break

            step+=1
    print("over")
    env.destory()

if __name__=="__main__":
    env=Maze()
    RL=DeepQNetwork(env.n_actions,env.n_features,learning_rate=0.01,reward_decay=0.9,\
                    e_greedy=0.9,replace_target_iter=200,memory_size=2000)
    env.after(100,run_maze)
    env.mainlop()
    RL.plot_cost()

"""
简单来说, DQN 有一个记忆库用于学习之前的经历.Q learning 是一种 off-policy 离线学习法, 
它能学习当前经历着的, 也能学习过去经历过的, 甚至是学习别人的经历. 
所以每次 DQN 更新的时候, 我们都可以随机抽取一些之前的经历进行学习. 
1.随机抽取这种做法打乱了经历之间的相关性,也使得神经网络更新更有效率.(experiment replay)
2.Fixed Q-targets 也是一种打乱相关性的机理, 如果使用 fixed Q-targets, 
我们就会在 DQN 中使用到两个结构相同但参数不同的神经网络, 
预测 Q 估计 的神经网络具备最新的参数, 而预测 Q 现实 的神经网络使用的参数则是很久以前的. 
有了这两种提升手段, DQN 才能在一些游戏中超越人类.
"""



























































