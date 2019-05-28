---
layout: default
title:  Status
---


# Video

# Project Summary

The goal of our project is to make a farm maintenance AI agent that can help farmer John with all of his farming duties. Our agent will be able to navigate the farm, moving from plot to plot, planting and harvesting as efficiently as possible. The agent will have the choice to plant between wheat, carrots, and potatoes all which have different reward values for being planted and harvested. Evaluating the value of the state over a regular time interval should give us a good estimate of how efficiently the agent is working.
  # Screenshots

# Approach
For the first half of this project we are focusing on the pathfinding abilities of the agent. This is powered by a Dueling Double Deep Q Network which consists of 4 convolutional layers and splits into two networks; the main and target network. The main network computes the value function of the state. This function tells us how good it is to be in any given state. The target network computes the advantage function of the state. This function tells us how much better taking a certain action would be compared to others. We then sum these values to get our final Q-value. These functions are then combined into one final Q-function in the final layer. The output is a number from 0-4, each number representing an action in {front, left, right, back}. The reason we are using this type of network is because the size of the state space is giant. For each plot that is set as a destination, there are (farm_width^2)-(# plots and water blocks) possible position states. For our 16x16 farm with 4 standard square plots, there are a total of 7,040 possible states that our agent needs to learn and discover. The goal is to expand to larger farming areas, so this type of network also allows for this expansion. This stage of the project is nearly complete.

As we wrap up the pathfinding portion of the project we are beginning to build and train the neural network for planting and harvesting decisions. This network will receive the content of each plot and agent inventory contents as an array of integers. We are in the final stages of finalizing a model and reward functions.

# Evaluation
  # Performance Graphs & Discussion
We identified a few metrics to determine the performance of our agent. First, we evaluated its the number of successful navigations for each episode of the training. From the following graphs we can see that the agent has learned how to navigate to a good number of plots.
[](https://github.com/Eisah-Jones/Farmer-John/blob/master/images/nav_success/4_4.png)

# Remaining Goals and Challenges

In the next couple of weeks, our main goal is to optimize the pathfinding network, complete the training of the decision network, compile it together into our final agent, and create more data visualizations that represent the network's knowledge. We plan on implementing a sort of curriculum training for the pathfinding network. Initially it was given 200 steps to find it's destination, but we will gradually lower this threshold encouraging the agent to take more optimal paths. I do believe that the only challenge that we anticipate in facing before the end of the project is not having enough training time to get agents that perform as well as we would like. We only have a couple machines that can efficiently run the training processes, it shouldn't be completely crippling as we should still be able to create agents that perform moderately well. Might attempt some cloud computing to increase the number of tests we can run.

# Resources Used
