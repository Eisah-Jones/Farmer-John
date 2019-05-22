import MalmoPython
import sys
import os
import time
import random
import json
import numpy as np
import farm_generator as fg
import pathfinding_network as pfn
import tensorflow as tf
from algorithms.dikjstra import get_path_dikjstra
import pprint

pathfinding_value = {"white_shulker_box": 1,
                     "brown_shulker_box": 2,
                      "blue_shulker_box": 2,
                                 "start": 10,
                                  "dest": 15,
                                  "prev": 5}

pf_action = {0: "Forward",
             1: "Left",
             2: "Right",
             3: "Back"}


def spawn_farm(f, n, m):
    """ Spawns farm blocks in world
        f = farm; 3D-array of the farm
        n = width of the farm
        m = mission
    """
    testing = False
    for i in range(n):
        for j in range(n):
            m.drawBlock(i, 0, j, f[0][i][j])
            if not testing:
                m.drawBlock(i, 1, j, f[1][i][j])
            else:
                m.drawBlock(i, 1, j, "air")


def get_agent_pos(f, n):
    """ Returns random (x, z) start position of agent
        f = farm
        n = width of the farm
    """
    pos = None
    for i in range(n):
        for j in range(n):
            if f[0][i][j] == "white_shulker_box" and (random.random() < 0.02 or pos is None):
                pos = (i, j)
    return pos


def get_pathfinding_input(f, s, d):
    result = []
    for r in f:
        temp = []
        for c in r:
            temp.append(pathfinding_value[c])
        result.append(temp)
    result[d[0]][d[1]] = pathfinding_value["dest"]
    result[s[0]][s[1]] = pathfinding_value["start"]
    return result


def move_agent(i, a, pos, f):
    x, z = 0, 0
    if i == 0:
        x, z = pos[0] + 1, pos[1]
    elif i == 1:
        x, z = pos[0], pos[1] - 1
    elif i == 2:
        x, z = pos[0], pos[1] + 1
    else:
        x, z = pos[0] - 1, pos[1]
    if (x < 0 or x > 15 or z < 0 or z > 15 or f[x][z] in ["brown_shulker_box", "blue_shulker_box"]):
        return -1, (x,z)
    a.sendCommand("tp {} 2 {}".format(x, z))
    return 1, (x,z)


def get_neighbors(pos, f):
    result = []
    for x, z in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        new_x = pos[0] + x
        new_z = pos[1] + z
        if not (new_x < 0 or new_x > 15 or new_z < 0 or new_z > 15):
            result.append(f[new_x][new_z])
    return result
            

def print_state(s):
    for i in range(len(s)):
        if i%16 == 0:
            print("\n")
        if s[i] == 10:
            print(".", end = " ")
        elif s[i] == 15:
            print('X', end = " ")
        else:
            print(s[i], end = " ")
    print("\n")



def create_agent_movement_record():
    f = open("data/performance.csv", "w")
    f.write("epNum,stepNum,posX,posZ,destX,destZ,action,reward,success\n")
    f.close()



def record_agent_movement(epNum, stepNum, pos, dest, action, reward, success):
    # record episode num, step num, pos, action, reward
    f = open("data/performance.csv", "a")
    f.write("{},{},{},{},{},{},{},{},{}\n".format(epNum,stepNum,pos[0],pos[1],dest[0],dest[1],action,reward,success))
    f.close()
        
    

# <ServerQuitFromTimeUp timeLimitMs="10000"/>
def run_mission():
    random.seed(time.time())
    mission_xml = '''<?xml version="1.0" encoding="UTF-8" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>Setup Farm</Summary>
              </About>

		   <ModSettings>
		      <MsPerTick>4</MsPerTick>
		   </ModSettings>

              <ServerSection>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="2;10x0;1;"/>
                  <DrawingDecorator>
                  </DrawingDecorator>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>FarmerBot</Name>
                <AgentStart>
                  <Placement yaw="-90"/>
                  <Inventory>
                    <InventoryObject slot="0" type="wheat_seeds" quantity="64"/>
                    <InventoryObject slot="1" type="carrot" quantity="64"/>
                    <InventoryObject slot="2" type="potato" quantity="64"/>
                  </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands/>
                  <AbsoluteMovementCommands/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

    # Create Agent Host
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    agent_host.setObservationsPolicy( MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY )
    # Create Mission
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Create Farm
    size = 16 #32
    size += size % 2  # Make sure the size is even
    # 3D-array of the farm
    #   farm[0] = 2D-array color_shulker_box data
    #   farm[1] = 2D-array minecraft block data
    farm = fg.generate_farm(size)
    spawn_farm(farm, size, my_mission)
    agent_spawn = get_agent_pos(farm, size)
    my_mission.startAt(agent_spawn[0], 4, agent_spawn[1])

    # Actual plots where crops are planted and walkable area
    farmland = []
    walkable = []
    for r in range(size):
        for c in range(size):
            if farm[0][r][c] == "brown_shulker_box":
                farmland.append((r, c))
            elif farm[0][r][c] == "white_shulker_box":
                walkable.append((r, c))
    #farmland = [(11, 12), (11, 4), (12, 4), (12,12), (12, 11), (11, 10), (12, 5), (12, 10), (12, 6)]
    

    # Attempt to start a mission:
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print("\nMission running ", end=' ')

    ## -- PFNN var init
    mainQN = pfn.QPathFinding(pfn.h_size)
    targetQN = pfn.QPathFinding(pfn.h_size)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    trainables = tf.trainable_variables()
    targetOps = pfn.updateTargetGraph(trainables, pfn.tau)
    myBuffer = pfn.experience_buffer()
    e = pfn.startE
    stepDrop = (pfn.startE - pfn.endE)/pfn.annealing_steps
    total_steps = 0
    start = agent_spawn
    create_agent_movement_record()
    ## --- PFNN
    if not os.path.exists(pfn.path):
        os.makedirs(pfn.path)
    print()
    ## Setup tensorflow session
    with tf.Session() as sess:
        sess.run(init)
        if pfn.load_model:
            ckpt = tf.train.get_checkpoint_state(pfn.path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("\nLOADED EXISTING MODEL\n")
        for i in range(pfn.num_episodes):
            episodeBuffer = pfn.experience_buffer()
            if not i == 1:
                start = list(walkable[np.random.randint(0, len(walkable))])
                agent_host.sendCommand("tp {} 2 {}".format(start[0], start[1]))
            dest = list(farmland[np.random.randint(0, len(farmland))])
            s = get_pathfinding_input(farm[0], start, dest)
            s = pfn.process_state(s)
            s1 = None
            d = False
            a = None
            episode_steps = 0
            quickest_path = len(get_path_dikjstra(start, dest, s)[0])-1
            random_steps = 0
            record_agent_movement(i, episode_steps, start, dest, a, 0, 0)
            print("Mission {} \n\t{} --> {}\n\tOptimal Path: {}".format(i, list(start), dest, quickest_path))
            # Loop until mission ends:
            while world_state.is_mission_running:
                time.sleep(0.008)
                ## -- PFNN
                # Get agent action
                world_state = agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:", error.text)
                if not len(world_state.observations) == 0:
                    # Get new world state, reward, d
                    #print_state(s)
                    msg = world_state.observations[0].text
                    ob = json.loads(msg)
                    start = [int(ob['XPos']), int(ob['ZPos'])]
                    #print("\n", get_pathfinding_input(farm[0], start, dest, prev_pos), "\n")
                    prev_pos = start
                    optimal_path = get_path_dikjstra(start, dest, s1)
                    if np.random.rand(1) < e or (total_steps < pfn.pre_train_steps and not pfn.load_model):
                        random_steps += 1
                        a = np.random.randint(0, 4)
                    else:
                        a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
                        #print("  NN move:", pf_action[a])
                    move_result = move_agent(a, agent_host, start, farm[0])
                    move_loc = move_result[1]
                    did_move = move_result[0]
                    if did_move == 1:
                        s1 = get_pathfinding_input(farm[0], move_loc, dest)
                        s1 = pfn.process_state(s1)
                        new_dist = len(get_path_dikjstra(move_loc, dest, s1)[0])
                    else:
                        s1 = s
                        new_dist = len(get_path_dikjstra(start, dest, s1)[0])
                    r = pfn.get_reward(move_loc, dest, did_move, optimal_path, get_neighbors(move_loc, farm[0]), new_dist)
                    #print(total_steps, r)
                    if r >= 5:
                        d = True
                    total_steps += 1
                    episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                    if total_steps > pfn.pre_train_steps:
                        if total_steps == pfn.pre_train_steps + 1:
                            print("\nBEGIN TRAINING\n")
                        if e > pfn.endE:
                            e -= stepDrop

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
                    s = s1
                    episode_steps += 1
                    record_s = 1 if d else 0
                    record_agent_movement(i, episode_steps, move_loc, dest, a, r, record_s)
                    if d or episode_steps > 200:
                        if episode_steps > 200:
                            print("\tAgent Lost...")
                        elif episode_steps == 1 :
                            print("\tSuccessful Navigation in {} step!".format(episode_steps))
                        else:
                            print("\tSuccessful Navigation in {} steps!".format(episode_steps))
                        print("\t  {} random steps".format(random_steps))
                        break

                myBuffer.add(episodeBuffer.buffer)
            saver.save(sess, pfn.path+"model-epi-"+str(i)+".ckpt")
            # -- PFNN
            pfn.reset_already_travelled()
            print("Mission ended\n")
            # Mission has ended.
            time.sleep(0.5)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    run_mission()
