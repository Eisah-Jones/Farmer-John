import random
import MalmoPython
import farming_agent as fa
import farm_generator_new as fg
import farming_mission_xml as fmx
from algorithms.dikjstra import get_path_dikjstra

    
# Main loop for running the final agent
if __name__ == '__main__':
    FarmerBot = fa.FarmingAgent(agent = agent_host)
    
    farming_mission = MalmoPython.MissionSpec(fmx.mission_xml, True)
    farming_mission_record = MissionRecordSpec()
    
    farm = fg.Farm()
    farm.spawn_farm(farming_mission)
    
    start = random.choice(farm.walkable)
    FarmerBot.position = start
    destination = None
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

    while world_state.is_mission_running:
        #time.sleep(1)
        world_state = FarmerBot.agent.getWorldState()
        for error in world_state.errors:
            print('Error:', error.text)
        if not len(world_state.observations) == 0:
            agent_state = FarmerBot.agent.get_state()
            if agent_state == 'DECIDING':
                #FarmerBot.optimization_network.get_decision(decision_state)
                dest = random.choice(farm.farmable) # TEMPORARY!
                FarmerBot.add_task(random.choice(farm.farmable)) # TEMPORARY!
                FarmerBot.destination = dest
                
            elif agent_state == 'NAVIGATING':
                pathfinding_input = farm.get_pathfinding_input(FarmerBot.position, FarmerBot.destination)
                #FarmerBot.pathfinding_network.get_move(pathfinding_input)
                action = random.randint(0, 3) # TEMPORARY!
                new_position = FarmerBot.get_action_pos(network_action)
                if farm.is_valid_move(new_position):
                    FarmerBot.teleport_agent(new_position)
                    if FarmerBot.has_succeeded():
                        FarmerBot.completed_task()
                        
            elif agent_state == 'FARMING'
                pass

        










    
