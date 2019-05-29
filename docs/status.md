---
layout: default
title:  Status
---


# Video

<iframe width="1120" height="630" src="https://www.youtube.com/embed/1-_TA086qQM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Project Summary

The goal of our project is to make a farm maintenance AI agent that can help farmer John with all of his farming duties. Our agent will be able to navigate the farm, moving from plot to plot, planting and harvesting as efficiently as possible. The agent will have the choice to plant wheat, carrots, and potatoes all which have different reward values for being planted and harvested. Evaluating the value of the state over a regular time interval should give us a good estimate of how efficiently the agent is working.

<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/Reference/agent.gif" alt="" style="max-width:100%;">

# Approach
For the first half of this project we are focusing on the pathfinding abilities of the agent. This is powered by a Dueling Double Deep Q Network which consists of 4 convolutional layers and splits into two networks; the main and target network. The main network computes the value function of the state. This function tells us how good it is to be in any given state. The target network computes the advantage function of the state. This function tells us how much better taking a certain action would be compared to others. We then sum these values to get our final Q-value. These functions are then combined into one final Q-function in the last layer. The output is a number from 0-4, each number representing an action in {front, left, right, back}. 

The reason we are using this type of network is because the size of the state space is fairly large. For each plot that is set as a destination, there are (farm_width^2)-(# farmland and water blocks) possible position states. For our 16x16 farm with 4 standard square plots, there are a total of 7,040 possible states that our agent needs to learn and discover. The goal is to expand to larger farming areas, so this type of network also allows for this expansion. This stage of the project is nearly complete. Here is some more technical information about the network and it's policies.

<img src="https://github.com/Eisah-Jones/Farmer-John/blob/master/images/Reference/DDQN_structure.png" alt="" style="max-width:50%;">

State-Action Q-function
```
Q(s, a) = V(s) + A(a)
```

Network Training Update
```
if total_steps % (pfn.update_freq) == 0:
    trainBatch = myBuffer.sample(pfn.batch_size)
    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
    end_multiplier = -(trainBatch[:, 4] - 1)
    doubleQ = Q2[range(pfn.batch_size), Q1]
    targetQ = trainBatch[:,2] + (pfn.y*doubleQ*end_multiplier)
    _ = sess.run(mainQN.updateModel, \
        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]), \
                   mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
    pfn.update_target(targetOps, sess)
```

The network also takes advantage of Experience Replay. This allows the agent to store it's experiences, and then randomly draw batches of them to train the network. This should allow the agent to more robustly learn to navigate. In essence, this prevents the network from only learning what it is immediately doing in the environment, and allow it to learn from all of its past experiences. When the buffer reaches its maximum size, the oldest experiences are removed as the new ones are added.

Here is our buffer class:
```
class experience_buffer():
    def __init__(self, buffer_size = 100000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size, dest):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

```


This is how our buffer is saved during the training process:
```
# s  = previous state
# a  = action taken from previous state
# r  = reward from taking action
# s1 = resulting state from action
# d  = whether or not the agent has completed mission
episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
```

Lastly, here is our reward function for our agent.
```
def get_reward(start, end, moved, optimal_path, new_dist):
    global already_travelled
    path, dim = optimal_path
    optimal_move = path[1]
    optimal_x = get_row(optimal_move, dim)
    optimal_y = get_col(optimal_move, dim)

    result = 0
    if len(path) == new_dist: # If agent did not move, penalize 10
        result -= 10
    elif len(path) < new_dist: # If agent is further away, penalize 20
        result -= 20
    else: # If agent is closer, reward 20
        result += 20

    dist = new_dist-1
    if dist < 2: # If within interaction distance
        return 100
    result -= dist * 0.2
    if moved == -1: # If made an invalid move
        result -= 10
    else: # If made a valid move
        result -= 1
    if start in already_travelled: # If has already been to block
        result -= 5
    else:
        already_travelled.append(start) 
    return result
```


As we wrap up the pathfinding portion of the project we are beginning to build and train the neural network for planting and harvesting decisions. This network will receive the content of each plot and agent inventory contents as an array of integers. We are in the last stages of finalizing a model and reward functions.

# Evaluation
We identified a few metrics to determine the performance of our agent. First, we evaluated the rolling average of the percentage of successful navigations for each destination during training. For the rolling average, we took the average success performance from the nearest 100 episodes and plotted those values. This gives us an idea of how well the agent learns as the training progresses. From the following graphs we can see that the agent has learned how to navigate to a good number of plots, with a fair success rate.

<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/nav_success/4_4.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/nav_success/4_12.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/nav_success/5_4.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/nav_success/6_10.png" alt="" style="max-width:50%;">

```
This is exactly what we wanted to see from our agent. 
Unfortunately, this is not the case for all of the plots on the farm.
```
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/nav_success/6_5.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/nav_success/10_4.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/nav_success/11_6.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/nav_success/12_5.png" alt="" style="max-width:50%;">

Based on the graphs that were generated, we can see that the agent is able to navigate to plots with an x coordinate of 4, 5, or 6 but has a more difficult time navigating to plots with an x coordinate of 10, 11, or 12. While we do have the success rate of the agent, this does not give us much information about where the agent spends its time while navigating to some plot destinations. In order to get this information, we created heatmaps that show how often the agent moved to, or attempted to move to, some block on the farm. Based on this we can get an idea of how well the agent learned to navigate to each of the plots, andgain some insight on how to improve the model.

```
Here are some heatmaps for destinations the agent was easily able to navigate towards.
```

<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/heatmaps/4_4.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/heatmaps/4_12.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/heatmaps/5_4.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/heatmaps/6_10.png" alt="" style="max-width:50%;">

```
Here are some heatmaps for destinations the agent was not easily able to navigate towards.
```

<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/heatmaps/6_5.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/heatmaps/10_4.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/heatmaps/11_6.png" alt="" style="max-width:50%;">
<img src="https://github.com/Eisah-Jones/Farmer-John/raw/master/images/heatmaps/12_5.png" alt="" style="max-width:50%;">

As we can see from the heatmaps, the agent tends to spend most of its time close to the destination plot, even on plots where the agent had difficulties. For plots with an x coordinate of 10, 11, or 12, the agent seems to get stuck pacing back and forth in the middle of the farm. It never broke through and learned how to navigate through the gap and over towards the other side. However, this does show that the agent is learning in the environment and has begun to generalize the features and movements required, although it may not be the most efficient. It is also worth keeping in mind that these failures are only after 200 steps. In testing, the agent would eventually make it to all the destination plots. We believe that with some tweaks in the training process we can easily resolve this issue and greatly improve our agent's performance. All of the graphs for each destination can be found [here](https://github.com/Eisah-Jones/Farmer-John/raw/master/images).

# Remaining Goals and Challenges
In the next couple of weeks, our main goal is to optimize the pathfinding network, complete the training of the decision network, compile it together into our final agent, and create more data visualizations that represent the network's knowledge. I do believe that the only challenge that we anticipate in facing before the end of the project is not having enough training time to get agents that perform as well as we would like. We only have a couple machines that can efficiently run the training processes, it shouldn't be completely crippling as we should still be able to create agents that perform moderately well. Might attempt some cloud computing to increase the number of tests we can run. We have also started training on a GTX 1070 graphics card which has greatly aided in the training process.

# Resources Used

[Dueling Network Architectures for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/wangf16.pdf)

[Improvements in Deep Q Learning](https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/)

[Simpole Reinforcement Learning with Tensorflow Part 4: Deep Q-Networks and Beyond](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df)
