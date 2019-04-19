## Proposal Meeting Time: April 24th @ 3:30

## Summary
The goal of our project is to make a farm maintenance AI agent that can help farmer John with all of his farming duties.
For our minimum viable product we would like to have an agent that can:
  - Navigate the farming area
  - Retrieve necessary resources
  - Till grass blocks
  - Plant seeds
  - Harvest crops
  - Store/organize farm resources
  
The input for our model will be a one chunk grid of the farm area and its goal will be to maximize productivity. At any given moment in time the agent will evaluate the environment and perform one of the above actions. The farm will exist on top of a single plane (the floor) with another plane of colored wool underneath that layer, invisible to the world but accessible to the agent. This layer of colored wool will exist to provide information about the farm layer above; different colors will represent things such as walkways, farmland/crop, storage, etc. Once this static environment is built, we can train the agent using various models.
  
If time permits, we would like to have an agent that can also:
  - Breed/kill animals
  - Store/organize animal goods
  - Take requests for crafting
  - Craft food/animal associated items

## AI/ML Algorithms

  - Pathfinding Algorithms (i.e. Dijkstra, A*, Any-angle Path Planning)
  - Reinforcement Learning
  - Markov Chains
  - Neural Networks
  - Genetic Algoritms

## Evaluation Plan
Because our agent is working on a fairly complex optimization problem, we will evaluate the performance of our agent based on their productivity. For example, if the agent just manages a small wheat farm, the entire amount of wheat harvested and seeds planted during a predefined time period could be used to determine how well our agent performed. If the agent is wasting time wandering around the farm, it will have a low performance score, but if the agent is tending carefully to the farm, it will have a much higher performance score. The worst any model can do is produce nothing during the observation period and have a performance score of zero. The maximum performance score will have to be discovered upon implementing and testing some of the AI/ML algorithms suggested above. Path finding evaluation hueristics will be evaluated based on the efficiency of the route, however this may just be handled by hardcoded path finding algorithms.
