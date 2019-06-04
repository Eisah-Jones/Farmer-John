import sys
import time
import MalmoPython
import pathfinding_network as pfn
import farming_network as fn
from collections.abc import Iterable
from algorithms.dikjstra import get_path_dikjstra

pf_action = {0: "Forward",
             1: "Left",
             2: "Right",
             3: "Back"}

class FarmingAgent:
    def __init__(self, rotation = 0, position = None,
                 destination = None, head_tilt = 0.0, load_pathfinding = False):
        self.agent = self._create_agent_host()
        self.rotation = rotation
        self.position = position
        self.destination = destination
        self.head_tilt = head_tilt
        self.pathfinding_network  = pfn.PathfindingNetwork(load_model = load_pathfinding)
        self.farming_network = fn.FarmingNetwork()

        self._tasks = []
    

    def teleport_agent(self, pos, action):
        #self.turn(action)
        self.position = pos
        self.agent.sendCommand('tp {} 2 {}'.format(pos[0]+0.5, pos[1]+0.5))


    def turn(self, a):
        if a == 0:
            if self.rotation == 1:
                self.agent.sendCommand('turn 0.05')
                time.sleep(0.2)
                self.agent.sendCommand('turn 0')
            elif self.rotation == 2:
                self.agent.sendCommand('turn -0.05')
                time.sleep(0.2)
                self.agent.sendCommand('turn 0')
            elif self.rotation == 3:
                self.agent.sendCommand('turn 0.05')
                time.sleep(0.4)
                self.agent.sendCommand('turn 0')
        elif a == 1:
            if self.rotation == 0:
                self.agent.sendCommand('turn -0.05')
                time.sleep(0.2)
                self.agent.sendCommand('turn 0')
            elif self.rotation == 2:
                self.agent.sendCommand('turn 0.05')
                time.sleep(0.4)
                self.agent.sendCommand('turn 0')
            elif self.rotation == 3:
                self.agent.sendCommand('turn 0.05')
                time.sleep(0.2)
                self.agent.sendCommand('turn 0')
        elif a == 2:
            if self.rotation == 0:
                self.agent.sendCommand('turn 0.05')
                time.sleep(0.2)
                self.agent.sendCommand('turn 0')
            elif self.rotation == 1:
                self.agent.sendCommand('turn 0.05')
                time.sleep(0.4)
                self.agent.sendCommand('turn 0')
            elif self.rotation == 3:
                self.agent.sendCommand('turn -0.05')
                time.sleep(0.2)
                self.agent.sendCommand('turn 0')
        elif a == 3:
            if self.rotation == 0:
                self.agent.sendCommand('turn -0.05')
                time.sleep(0.4)
                self.agent.sendCommand('turn 0')
            elif self.rotation == 1:
                self.agent.sendCommand('turn -0.05')
                time.sleep(0.2)
                self.agent.sendCommand('turn 0')
            elif self.rotation == 2:
                self.agent.sendCommand('turn 0.05')
                time.sleep(0.2)
                self.agent.sendCommand('turn 0')
        self.rotation = a


    def get_state(self):
        if len(self._tasks) == 0:
            return 'DECIDING'
        elif type(self._tasks[0]) is tuple:
            return 'NAVIGATING'
        elif type(self._tasks[0]) is int:
            return 'FARMING'


    def add_task(self, task):
        self._tasks.append(task)


    def completed_task(self):
        return self._tasks.pop(0)


    def get_action_pos(self, action):
        if action == 0:
            return (self.position[0] + 1, self.position[1])
        elif action == 1:
            return (self.position[0]    , self.position[1] - 1)
        elif action == 2:
            return (self.position[0]    , self.position[1] + 1)
        return (self.position[0] - 1, self.position[1])


    def has_succeeded(self, state):
        return len(get_path_dikjstra(self.position, self.destination, state)[0])-1 == 1


    def _create_agent_host(self):
        result = MalmoPython.AgentHost()
        try:
            result.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(result.getUsage())
            exit(1)
        if result.receivedArgument("help"):
            print(result.getUsage())
            exit(0)
        result.setObservationsPolicy( MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY )
        return result

