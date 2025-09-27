# MONTE CARLO CONTROL ALGORITHM

## AIM
Develop a Python program to find the optimal policy for a given RL environment using Monte Carlo Control.

## PROBLEM STATEMENT
Use the Monte Carlo Control algorithm to find the optimal policy for the FrozenLake environment from episode trajectories.

## ALGORITHM OVERVIEW
1. **Initialization:**  
   - Set the number of states (`ns`) and actions (`na`).  
   - Create a Q-table (`ns × na`) to store action values.  
   - Define decaying schedules for learning rate (`alpha`) and exploration rate (`epsilon`).  
   - Use a discount factor (`gamma`) and epsilon-greedy action selection.  

2. **Training Loop:**  
   - For each episode, generate a trajectory of `(state, action, reward, next_state, done)`.  
   - Compute return `G` for each state-action pair.  
   - Update Q-values: `Q(s,a) ← Q(s,a) + α * (G - Q(s,a))`.  
   - Track Q-values and greedy policy per episode.  

3. **Policy Extraction:**  
   - Compute state-value function `V` as `V(s) = max_a Q(s,a)`.  
   - Define learned policy `pi(s) = argmax_a Q(s,a)`.  

## MONTE CARLO CONTROL FUNCTION
```python
def mc_control(env,gamma=1.0,init_alpha=0.5,min_alpha=0.01,
                  alpha_decay_ratio=0.3, init_epsilon=1.0, min_epsilon=0.1,epsilon_decay_ration=0.9,
                  n_episodes=3000,max_steps=200,first_visit=True):
  nS,nA=env.observation_space.n, env.action_space.n
  discounts=np.logspace(0,max_steps,num=max_steps,base=gamma,endpoint=False)
  alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
  epsilons=decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ration,n_episodes)
  pi_track=[]
  Q=np.zeros((nS,nA),dtype=np.float64)
  Q_track=np.zeros((n_episodes,nS,nA),dtype=np.float64)

  select_action=lambda state,Q,epsilon:np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
  for e in tqdm(range(n_episodes),leave=False):
    trajectory=generate_trajectory(select_action, Q, epsilons[e],env,max_steps)
    visited=np.zeros((nS,nA),dtype=np.bool)
    for t, (state, action, reward, _, _) in enumerate(trajectory):
      if visited[state][action] and first_visit:
        continue
      visited[state][action]=True
      n_steps=len(trajectory[t:])
      G=np.sum(discounts[:n_steps]*trajectory[t:,2])
      Q[state][action]=Q[state][action]+alphas[e]*(G-Q[state][action])
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
  V=np.max(Q,axis=1)
  pi=lambda s: {s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
  return Q,V,pi,Q_track,pi_track
```
# OUTPUT 

<img width="1191" height="414" alt="Screenshot 2025-09-27 114808" src="https://github.com/user-attachments/assets/066adf32-270c-4d72-ab9b-dec550984cd2" />



# RESULT

The Python program successfully finds the optimal policy for the FrozenLake environment using Monte Carlo Control.
