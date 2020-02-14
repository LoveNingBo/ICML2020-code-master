
"""
epsilon-action
初始化
"""
state=START
if np.random.binomial(1,EPSILON)==1:
    action=np.random.choice(ACTIONS)
else:
    values_=q_value[state[0],state[1],:]
    action=np.random.choice([action_ for action,value_ in enumerate(values_)if values_==np.max(values_ )])

"""
get new state
根据当前的St和At获取St+1
"""
def step(state,action):
    i,j=state
    if action==ACTION_UP:
        return [max(i-1-WIND[j],0),j]
    elif action==ACTION_DOWN:
        return [max(min(i+1-WIND[j],WORLD_HEIGHT-1),0),j]
    elif action==ACTION_LEFT:
        return [max(i-WIND[j],0),max(j-1),0]
    elif action==ACTION_RIGHT:
        return [max(i-WIND[j],0),min(j+1,WORLD_WIDTH-1)]
    else:
        assert False
"""
choose new action
利用贪婪法根据当前的状态选择动作
"""
next_state=step(state,action)
if np.random.binomial(1,EPSILON)==1:
    next_action=np.random.choice(ACTIONS)
else:
    values_=q_value[next_state[0],next_state[1],:]
    next_action=np.random.choice([action_ for action_,value_ in enumerate(vaules_)if values_==np.max(values_)])
