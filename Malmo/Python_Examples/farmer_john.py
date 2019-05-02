from builtins import range
import MalmoPython
import os
import sys
import time
import random
import farm_generator as fg
import tensorflow as tf


def spawn_farm(f, n, m):
    ''' Spawns farm blocks in world
        f = farm; 3D-array of the farm
        n = width of the farm
        m = mission
    '''
    testing = False
    for x, i in zip(range(-int(n/2), int(n/2)), range(n)):
        for z, j in zip(range(-int(n/2), int(n/2)), range(n)):
            m.drawBlock(x, 0, z, f[0][i][j])
            if (not testing):
                m.drawBlock(x, 1, z, f[1][i][j])
            else:
                m.drawBlock(x, 1, z, "air")


def get_agent_pos(f, n):
    ''' Returns random (x, z) start position of agent
        f = farm
        n = width of the farm
    '''
    pos = None
    for x, i in zip(range(-int(n/2), int(n/2)), range(n)):
        for z, j in zip(range(-int(n/2), int(n/2)), range(n)):
            if (f[0][i][j] == "white_shulker_box" and (random.random() < 0.02 or pos == None)):
                pos = (x, z)
    return pos



def run_mission():
    random.seed(time.time())
    missionXML='''<?xml version="1.0" encoding="UTF-8" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Setup Farm</Summary>
              </About>
              
              <ServerSection>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="2;10x0;1;"/>
                  <DrawingDecorator>
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="1000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>FarmerBot</Name>
                <AgentStart>
                  <Placement yaw="180"/>
                  <Inventory>
                    <InventoryObject slot="0" type="wheat_seeds" quantity="64"/>
                  </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''
            
    ## Create Agent Host
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)


    ## Create Mission
    my_mission = MalmoPython.MissionSpec(missionXML, True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    ## Create Farm
    size = 32;
    size += size%2 # Make sure the size is even
    # 3D-array of the farm
    #   farm[0] = 2D-array color_shulker_box data
    #   farm[1] = 2D-array minecraft block data
    farm = fg.generate_farm(size)
    spawn_farm(farm, size, my_mission)
    agent_spawn = get_agent_pos(farm, size)
    my_mission.startAt(agent_spawn[0], 4, agent_spawn[1])

    # Actual plots where crops are planted
    farmland = []
    for r in range(size):
        for c in range(size):
            if(farm[0][r][c] == "brown_shulker_box"):
                farmland.append((r, c))


    ## Attempt to start a mission:
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)
                
    ## Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print("\nMission running ", end=' ')

    ## For testing pathfinding
    dest  = random.choice(farmland)

    ## Loop until mission ends:
    while world_state.is_mission_running:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

        if(len(world_state.observations) > 0):
            obs = world_state.observations[-1].text
            
            x_idx = obs[obs.index("\"XPos\":"):].index(':') + obs.index("\"XPos\":") + 1
            x_end = obs[x_idx:].index(',') + x_idx
            x = obs[x_idx : x_end]

            z_idx = obs[obs.index("\"ZPos\":"):].index(':') + obs.index("\"ZPos\":") + 1
            z_end = obs[z_idx:].index(',') + z_idx
            z = obs[z_idx : z_end]
            
            start = (x, z)

    print("\nMission ended")
    ## Mission has ended.
        
        
if __name__ == "__main__":
    run_mission()
