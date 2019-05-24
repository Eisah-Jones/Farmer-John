## Proposal Meeting Time: April 24th @ 3:30

## Summary
The goal of our project is to make a farm maintenance AI agent that can help farmer John with all of his farming duties.
For our minimum viable product we would like to have an agent that can:
  - Navigate the farming area
  - Optimize agent planting/harvesting schedule
  
In order to get our agent to navigate the farm, we are using a Dueling Double Deep Q Network and providing a numerical representation of the farmland as an input. We positively reward the agent when they get closer to or reach the destination. We negatively reward the agent for not getting closer to the destination, making an invalid move, travelling to an already travelled block, and a slight penalty for any movement. Additionally, we add a penalty equal to half the distance from the goal.

Once our agent has learned to navigate the farm, it can then start training on how to best optimize its planting and harvesting routine. For this, we will evaluate the state of the world after certain intervals and evaluate the change in the value of the state. The value of the state will be determined by the contents of the plots (i.e. empty, wheat, carrot, potato). Each plot will have its own value unique reward value based on it's contents. The agent's goal should be to keep this  reward as high as possible at all times.

## AI/ML Algorithms

  - Pathfinding Algorithms (i.e. Dijkstra, A*, Any-angle Path Planning)
  - Neural Networks
  - Reinforcement Learning
  - Dueling Double Deep Q Network

## Evaluation Plan
Because our agent is working on a fairly complex optimization problem, we will evaluate the performance of our agent based on their productivity. For example, if the agent just manages a small wheat farm, the entire amount of wheat harvested and seeds planted during a predefined time period could be used to determine how well our agent performed. If the agent is wasting time wandering around the farm, it will have a low performance score, but if the agent is tending carefully to the farm, it will have a much higher performance score. The worst any model can do is produce nothing during the observation period and have a performance score of zero. The maximum performance score will have to be discovered upon implementing and testing some of the AI/ML algorithms suggested above. Path finding evaluation hueristics will be evaluated based on the efficiency of the route, however this may just be handled by hardcoded path finding algorithms.
