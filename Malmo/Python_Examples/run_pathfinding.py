import os
import time
import random
import MalmoPython
import farm as fg
import numpy as np
import tensorflow as tf
import farming_agent as fa
import farming_mission_xml as fmx
import pathfinding_network as pfn
from algorithms.dikjstra import get_path_dikjstra


def run_agent(isTraining, load_model = False):
    FarmerBot = fa.FarmingAgent(load_pathfinding = load_model)
    
    farming_mission = MalmoPython.MissionSpec(fmx.mission_xml, True)
    farming_mission_record = MalmoPython.MissionRecordSpec()
    
    farm = fg.Farm()
    farm.spawn_farm(farming_mission)
    
    FarmerBot.position = random.choice(farm.walkable)
    farming_mission.startAt(FarmerBot.position[0], 2, FarmerBot.position[1])

    max_retries = 3
    for retry in range(max_retries):
        try:
            FarmerBot.agent.startMission(farming_mission, farming_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print('Error starting mission:', e)
                exit(1)
            else:
                time.sleep(2)

    print('Waiting for the server to start ', end=' ')
    world_state = FarmerBot.agent.getWorldState()
    while not world_state.has_mission_begun:
        print('.', end='')
        time.sleep(0.1)
        world_state = FarmerBot.agent.getWorldState()
        for error in world_state.errors:
            print('Error:', error.text)

    print('\nServer running ', end=' ')

    with tf.Session() as sess:
        print()
        total_steps = 0
        network = FarmerBot.pathfinding_network
        sess.run(network.init)
        if network.load_model or not isTraining:
            ckpt = tf.train.get_checkpoint_state(network.model_path)
            network.saver.restore(sess, ckpt.model_checkpoint_path)
            print("\nLOADED EXISTING MODEL\n")
        for i in range(network.num_episodes):
            #print(i, total_steps)
            episodeBuffer = pfn.experience_buffer()
            previousPositions = []
            FarmerBot.teleport_agent(list(random.choice(farm.walkable)), 0)
            FarmerBot.destination = random.choice(farm.farmable)
            state = pfn.process_state(farm.get_pathfinding_input(FarmerBot.position, FarmerBot.destination))
            complete = False
            new_state = None
            new_dist = a = did_move = -1
            episode_steps = random_steps = 0
            quickest_path = len(get_path_dikjstra(FarmerBot.position, FarmerBot.destination, state)[0])-1
            network.log.record_csv_data(i, episode_steps, FarmerBot.position[0], FarmerBot.position[1], FarmerBot.destination[0], FarmerBot.destination[1], random.randint(0,3), 0, 0, 0)
            if total_steps > network.pre_train_steps:
                print("Mission {} \n\t{} --> {}\n\tOptimal Path: {}\n\tNetwork Exploration Rate: {}".format(i, FarmerBot.position, FarmerBot.destination, quickest_path-1, network.e))
            while world_state.is_mission_running:
                world_state = FarmerBot.agent.getWorldState()
                for error in world_state.errors:
                    print('Error:', error.text)
                if isTraining:
                    #time.sleep(2) # FOR DEV TESTING
                    if len(previousPositions) < 5:
                        previousPositions.append(FarmerBot.position)
                    else:
                        previousPositions.pop(0)
                        previousPositions.append(FarmerBot.position)
                    optimal_path = get_path_dikjstra(FarmerBot.position, FarmerBot.destination, state)
                    action, random_move = network.get_action(sess, state, total_steps)
                    move_loc = FarmerBot.get_action_pos(action)
                    random_steps += random_move
                    if farm.is_valid_move(move_loc):
                        did_move = 1
                        FarmerBot.teleport_agent(move_loc, action)
                        new_state = pfn.process_state(farm.get_pathfinding_input(FarmerBot.position, FarmerBot.destination, previousPositions))
                        new_dist = len(get_path_dikjstra(move_loc, FarmerBot.destination, new_state)[0])
                    else:
                        did_move = -1
                        new_state = state
                        new_dist = len(get_path_dikjstra(FarmerBot.position, FarmerBot.destination, new_state)[0])
                    reward = network.get_reward(move_loc, FarmerBot.destination, did_move, optimal_path, new_dist, quickest_path)
                    if reward == 100:
                        complete = True
                    episodeBuffer.add(np.reshape(np.array([state, action, reward, new_state, complete, FarmerBot.destination]), [1, 6]))
                    if total_steps > network.pre_train_steps:
                        network.decrease_epsilon()
                        if total_steps % network.update_freq == 0:
                            network.train(sess, FarmerBot.destination)
                    state = new_state
                    episode_steps += 1
                    total_steps += 1
                    complete_recValue = 1 if complete else 0
                    network.log.record_csv_data(i, episode_steps, FarmerBot.position[0], FarmerBot.position[1], FarmerBot.destination[0], FarmerBot.destination[1], action, reward, complete_recValue, new_dist-1)
                    if (complete or episode_steps > quickest_path*4) and total_steps > network.pre_train_steps:
                        if episode_steps > quickest_path*4:
                            print("\tAgent Lost. . . {} steps..............\n\t\t{:3} random steps".format(episode_steps, random_steps))
                        else:
                            print("\tSuccessful Navigation in {} steps!**************\n\t\t{} random steps".format(episode_steps, random_steps))
                        break
                    elif complete or episode_steps > quickest_path*4:
                        break
                else:
                    #time.sleep(0.01) # FOR DEV TESTING
                    action = network.test(sess, state)
                    move_loc = FarmerBot.get_action_pos(action)
                    if farm.is_valid_move(move_loc):
                        FarmerBot.teleport_agent(move_loc, action)
                        state = pfn.process_state(farm.get_pathfinding_input(FarmerBot.position, FarmerBot.destination))
                        new_dist = len(get_path_dikjstra(move_loc, FarmerBot.destination, state)[0])

                        if new_dist < 2:
                            FarmerBot.destination = random.choice(farm.farmable)
                    
            network.add_episode_experience(episodeBuffer.buffer)
            if (i % 100 == 0):
                network.saver.save(sess, os.path.join(network.model_path, "model-epi-"+str(i)+".ckpt"))
            if total_steps > network.pre_train_steps:
                print('Mission Ended\n')
        network.saver.save(sess, os.path.join(network.model_path, "model-final-.ckpt"))

    
# Main loop for running the pathfinding agent
if __name__ == '__main__':
    run_agent(isTraining = True, load_model = False) # Training agent
    #run_agent(isTraining = False) # Testing agent









    
