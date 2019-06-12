import time
import random
import MalmoPython
import numpy as np
import pandas as pd
import farming_agent as fa
import farm as fg
import farming_mission_xml as fmx
from algorithms.dikjstra import get_path_dikjstra
from keras.models import model_from_json
import tensorflow as tf
import pathfinding_network as pfn



def write_path_optimality(create = False, epNum = 0, num_steps = 1, max_steps = 1):
    openType = 'w' if create else 'a'
    f = open('data/agent_testing/path_optimality.csv', openType)
    if openType == 'w':
        f.write('epNum,optimality\n')
    else:
        f.write('{},{}\n'.format(epNum, float(num_steps)/max_steps))
    f.close()


def write_choice_optimality(create = False, epNum = 0, madeOpt = 0):
    openType = 'w' if create else 'a'
    f = open('data/agent_testing/choice_optimality.csv', openType)
    if openType == 'w':
        f.write('epNum,correctChoice\n')
    else:
        f.write('{},{}\n'.format(epNum, madeOpt))
    f.close()

def write_state_value(create = False, epNum = 0, stateValue = 0):
    openType = 'w' if create else 'a'
    f = open('data/agent_testing/state_value.csv', openType)
    if openType == 'w':
        f.write('epNum,stateValue\n')
    else:
        print(stateValue)
        f.write('{},{}\n'.format(epNum, stateValue))
    f.close()

    
# Main loop for running the final agent
if __name__ == '__main__':
    FarmerBot = fa.FarmingAgent(load_model=True, pathfinding_path='models/pathfinding')
    
    farming_mission = MalmoPython.MissionSpec(fmx.mission_xml, True)
    farming_mission_record = MalmoPython.MissionRecordSpec()
    
    farm = fg.Farm()
    farm.spawn_farm(farming_mission)
    
    start = random.choice(farm.walkable)
    FarmerBot.position = start
    destination = random.choice(farm.farmable)
    FarmerBot.destination = destination
    farming_mission.startAt(start[0], 2, start[1])

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

    write_path_optimality(True)
    write_choice_optimality(True)
    write_state_value(True)


    pathfindingSess = tf.Session()
    pathfindingSess.run(FarmerBot.pathfinding_network.init)
    ckpt = tf.train.get_checkpoint_state(FarmerBot.pathfinding_network.model_path)
    FarmerBot.pathfinding_network.saver.restore(pathfindingSess, ckpt.model_checkpoint_path)
    farming_network = FarmerBot.farming_network
    state_steps = 0
    optimal_path = None
    stateEvalTime = 30
    startTime = time.time()
    epNum = 0
    while world_state.is_mission_running:
        world_state = FarmerBot.agent.getWorldState()
        for error in world_state.errors:
            print('Error:', error.text)
        if not len(world_state.observations) == 0:
            time.sleep(0.1)
            agent_state = FarmerBot.get_state()
            farm.grow_crops()
            if agent_state == 'DECIDING': # GET DESTINATION FROM DECISION NETWORK
                state_steps = 0
                farm_input = np.array(farm.get_farming_input())
                farm_input[farm.farmable.index(FarmerBot.destination)] = 10.0
                destIdx = np.argmax(farming_network.model.predict(farm_input.reshape(1, -1)))
                dest = farm.farmable[destIdx]
                FarmerBot.add_task(dest)
                FarmerBot.destination = dest
                choiceOpt = 1 if not farm.crops[dest] == 'wheat_not_ready' else 0
                write_choice_optimality(False, epNum, choiceOpt)
        
            elif agent_state == 'NAVIGATING': # GET NEXT STEP IN NAVIGATION
                state_steps += 1
                pathfinding_input = farm.get_pathfinding_input(FarmerBot.position, FarmerBot.destination)
                action = FarmerBot.pathfinding_network.test(pathfindingSess, pathfinding_input)
                new_position = FarmerBot.get_action_pos(action)
                if farm.is_valid_move(new_position):
                    FarmerBot.teleport_agent(new_position, action)
                    if FarmerBot.is_at_dest():
                        FarmerBot.completed_task()
                        FarmerBot.add_task(0)
                        write_path_optimality(False, epNum, state_steps, len(optimal_path)-1)
                        epNum += 1
                        state_steps = 0
                        
            elif agent_state == 'FARMING': #HARDCODED PLANTING ACTION
                savePos = FarmerBot.position
                if farm.crops[FarmerBot.destination] in ['wheat_ready', 'wheat_not_ready']:
                    FarmerBot.teleport_agent(FarmerBot.destination, 0)
                    time.sleep(0.1)
                    FarmerBot.harvest()
                    farm.harvest_crop(FarmerBot.destination)
                    time.sleep(0.1)
                    FarmerBot.teleport_agent(savePos, 0)
                else:
                    FarmerBot.teleport_agent(FarmerBot.destination, 0)
                    time.sleep(0.1)
                    FarmerBot.plant()
                    farm.plant_crop_test(FarmerBot.destination)
                    time.sleep(0.1)
                    FarmerBot.teleport_agent(savePos, 0)
                FarmerBot.completed_task()

            if not optimal_path is None and state_steps >= (len(optimal_path)*4)+5:
                FarmerBot.completed_task()
                write_path_optimality(False, epNum, state_steps, len(optimal_path)-1)
                epNum += 1
                print("LOST")

            if time.time() - startTime >= stateEvalTime:
                write_state_value(False, epNum, farm.get_farm_value() + FarmerBot.get_inventory_value(world_state))
                startTime = time.time()

        










    
