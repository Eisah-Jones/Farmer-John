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

pathfinding_value = {"white_shulker_box": 0,
                     "brown_shulker_box": 1,
                      "blue_shulker_box": 2,
                                 "start": 3,
                                  "dest": 4}


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
    result[s[0]][s[1]] = pathfinding_value["start"]
    result[d[0]][d[1]] = pathfinding_value["dest"]
    #print("\n", result)
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
    if (x < 0 or x > 31 or z < 0 or z > 31 or f[x][z] in ["brown_shulker_box", "blue_shulker_box"]):
        return -1, (x,z)
    a.sendCommand("tp {} 2 {}".format(x, z))
    return 1, (x,z)

# <ServerQuitFromTimeUp timeLimitMs="10000"/>
def run_mission():
    random.seed(time.time())
    mission_xml = '''<?xml version="1.0" encoding="UTF-8" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>Setup Farm</Summary>
              </About>

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
                  <Placement yaw="90"/>
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

    # Create Mission
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Create Farm
    size = 32
    size += size % 2  # Make sure the size is even
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
            if farm[0][r][c] == "brown_shulker_box":
                farmland.append((r, c))

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
    jList = []
    rList = []
    total_steps = 0
    start = agent_spawn
    total_reward = 0
    ## --- PFNN

    if not os.path.exists(pfn.path):
        os.makedirs(pfn.path)
    ## Setup tensorflow session
    with tf.Session() as sess:
        sess.run(init)
        if pfn.load_model == True:
            ckpt = tf.train.get_checkpoint_state(pfn.path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(pfn.num_episodes):
            total_reward = 0
            episodeBuffer = pfn.experience_buffer()
            dest = list(random.choice(farmland))
            print(i, dest)
            my_mission.drawBlock(dest[0], 4, dest[1], "red_shulker_box")
            s = get_pathfinding_input(farm[0], start, dest)
            s = pfn.process_state(s)
            d = False
            rAll = 0
            j = 0

            # Loop until mission ends:
            while world_state.is_mission_running:
                ## -- PFNN
                # Get agent action
                world_state = agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:", error.text)
                if len(world_state.observations) > 0:
                    # Get new world state, reward, d
                    msg = world_state.observations[-1].text
                    ob = json.loads(msg)
                    start = [int(ob['XPos']), int(ob['ZPos'])]
                    s1 = get_pathfinding_input(farm[0], start, dest)
                    s1 = pfn.process_state(s1)
                    optimal_path = get_path_dikjstra(start, dest, s1)
                    if np.random.rand(1) < e or total_steps < pfn.pre_train_steps:
                        a = np.random.randint(0, 4)
                    else:
                        a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
                    move_result = move_agent(a, agent_host, start, farm[0])
                    move_loc = move_result[1]
                    did_move = move_result[0]
                    r = pfn.get_reward(move_loc, dest, did_move, optimal_path)
                    #print(total_steps, r)
                    if r == 100:
                        d = True
                    total_steps += 1
                    episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                    if total_steps > pfn.pre_train_steps:
                        if e > pfn.endE:
                            e -= stepDrop

                        if total_steps % (pfn.update_freq) == 0:
                            trainBatch = myBuffer.sample(pfn.batch_size)
                            Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                            Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                            end_multiplier = -(trainBatch[:, 4] - 1)
                            doubleQ = Q2[range(pfn.batch_size), Q1]
                            targetQ = trainBatch[:,2] + (pfn.y*doubleQ*end_multiplier)
                            try:
                                _ = sess.run(mainQN.updateModel, \
                                    feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]), \
                                               mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                                pfn.update_target(targetOps, sess)
                            except:
                                pass
                    rAll += r
                    total_reward += r
                    s = s1

                    if d == True or total_reward < -2500:
                        if total_reward < -2500:
                            print("LOST")
                        break

                myBuffer.add(episodeBuffer.buffer)
                jList.append(j)
                rList.append(rAll)
                if i % 100 == 0:
                    saver.save(sess, pfn.path+"model-att-"+str(i)+".ckpt")
            saver.save(sess, pfn.path+"model-epi-"+str(i)+".ckpt")
            # -- PFNN
              
            pfn.reset_already_travelled()
            print("\nMission ended")
            # Mission has ended.
            time.sleep(1)


if __name__ == "__main__":
    run_mission()
