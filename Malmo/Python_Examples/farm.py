import random


shulker_dict = {'farmland': 'brown_shulker_box',
                   'water': 'blue_shulker_box',
                  'planks': 'white_shulker_box',
                   'stone': 'white_shulker_box',
                   'grass': 'white_shulker_box'}

pathfinding_value = {"white_shulker_box": 1,
                     "brown_shulker_box": 2,
                      "blue_shulker_box": 2,
                                  "prev": 5,
                                 "start": 10,
                                  "dest": 15}


class Farm:
    def __init__(self, size = 16):
        self.size = 16
        self.shulker, self.farmland = self.initialize_farm()
        self.walkable, self.farmable = self.analyze_farm()


    def initialize_farm(self):
        bot = []
        top = []
        for i in range(self.size):
            tempBot = []
            tempTop = []
            for j in range(self.size):
                if (i in [3, 5, 10, 12] and j in [3, 5, 10, 12]) or \
                   (i in [4, 11] and j in [3, 5, 10, 12]) or \
                   (j in [4, 11] and i in [3, 5, 10, 12]) :
                    tempTop.append("farmland")
                    tempBot.append(shulker_dict["farmland"])
                elif i in [4, 11] and j in [4, 11]:
                    tempTop.append("water")
                    tempBot.append(shulker_dict["water"])
                else:
                    tempTop.append("grass")
                    tempBot.append(shulker_dict["grass"])
            bot.append(tempBot)
            top.append(tempTop)
        return (bot, top)


    def analyze_farm(self):
        walkable = []
        farmable = []
        for i in range(self.size):
            for j in range(self.size):
                if self.shulker[i][j] == 'brown_shulker_box':
                    farmable.append((i, j))
                elif self.shulker[i][j] == 'white_shulker_box':
                    walkable.append((i, j))
        return (walkable, farmable)
                    

    def spawn_farm(self, mission):
        for i in range(self.size):
            for j in range(self.size):
                mission.drawBlock(i, 0, j, self.shulker[i][j])
                mission.drawBlock(i, 1, j, self.farmland[i][j])



    def get_pathfinding_input(self, a, b):
        result = []
        for r in self.shulker:
            temp = []
            for c in r:
                temp.append(pathfinding_value[c])
            result.append(temp)
        result[b[0]][b[1]] = pathfinding_value["dest"]
        result[a[0]][a[1]] = pathfinding_value["start"]
        return result


    def is_valid_move(self, pos):
        x, z = pos
        return not (x < 0 or x >= self.size or z < 0 or z >= self.size or \
                self.shulker[x][z] in ["brown_shulker_box", "blue_shulker_box"])
