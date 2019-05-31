import os
import time
import random
import MalmoPython
import farm as fg
import numpy as np
import tensorflow as tf
import farming_agent as fa
import farming_mission_xml as fmx

def run_agent(isTraining):
    FarmerBot = fa.FarmingAgent()
    
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
    episode_count = 0
    total_steps = 0
    xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
    running_reward = None
    reward_sum = episode_count = 0
    with tf.Session() as sess:
        print()
        network = FarmerBot.farming_network
        sess.run(network.init)
        if network.load_model or not isTraining:
            ckpt = tf.train.get_checkpoint_state(network.model_path)
            network.saver.restore(sess, ckpt.model_checkpoint_path)
            print("\nLOADED EXISTING MODEL\n")
        gradBuffer = sess.run(network.tvars)
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        state = farm.get_farming_input()
        while episode_count < network.num_episodes:
            time.sleep(0.5)
            x = np.reshape(state, [1, network.input_size])
            tfprob = sess.run(network.probability: feed_dict={self.observations:x})
            print('FOR ACTION:', tfprob)
            action = 0 # TEMPORARY
            
            xs.append(x)
            ys.append(action)

            state = farm.get_farming_input()
            reward = 0
            done = False
            reward_sum += 0
                
            drs.append(reward)

            if done:
                episode_count += 1
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                tfp = tfps
                xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]

                discounted_epr = network.discount_rewards(epr)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr //= np.std(discounted_epr)

                tGrad = sess.run(network.newGrads,
                                 feed_dict={network.observations: epx,
                                            network.input_y: epy,
                                            network.advantages: discounted_epr})
                for ix, grad in enumerate(tGrad):
                    gradBuffer[ix] += grad

                if episode_number % batch_size == 0:
                    sess.run(network.updateGrads,
                             feed_dict={network.W1Grad: gradBuffer[0],
                                        network.W2Grad:gradBuffer[1]})
                    for ix, grad in enumerate(gradBuffer):
                        gradbuffer[ix] = grad * 0

                    running_reward = reward_sum if running_reward is None else running_reward +0.99 + reward_sum * 0.01

                    print('Average reward for episode %f.  Total average reward %f.' % (reward_sum//batch_size, running_reward//batch_size))
                
                    if reward_sum//batch_size > 100: 
                        print("Task solved in",episode_number,'episodes!')
                        break

                    reward_sum = 0



            


# Main loop for running the farming agent
if __name__ == '__main__':
    run_agent(isTraining = True) # Training agent
    #run_agent(isTraining = False) # Testing agent






    
