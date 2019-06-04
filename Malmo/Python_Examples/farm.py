import random


shulker_dict = {'farmland': 'brown_shulker_box',
                   'water': 'blue_shulker_box',
                  'planks': 'white_shulker_box',
                   'stone': 'white_shulker_box',
                   'grass': 'white_shulker_box'}

pathfinding_value = {"white_shulker_box": 1.0,
                     "brown_shulker_box": 2.0,
                      "blue_shulker_box": 2.0,
                                  "prev": 5.0,
                                 "start": 10.0,
                                  "dest": 15.0}


farming_value = {          'empty': 0.0,
                 'wheat_not_ready': 1.0,
                     'wheat_ready': 5.0,
                        'position': 10.0}


class Farm:
    def __init__(self, size = 16):
        self.size = 16
        self.shulker, self.farmland, self.crops = self.initialize_farm()
        self.walkable, self.farmable = self.analyze_farm()
        self.growable = dict()


    def initialize_farm(self):
        crop_order = []
        crops = dict()
        bot = []
        top = []
        for i in range(self.size):
            tempCrop = []
            tempBot = []
            tempTop = []
            for j in range(self.size):
                if (i in [3, 5, 10, 12] and j in [3, 5, 10, 12]) or \
                   (i in [4, 11] and j in [3, 5, 10, 12]) or \
                   (j in [4, 11] and i in [3, 5, 10, 12]) :
                    crop_order.append((i, j))
                    crops[(i, j)] = 'empty'
                    tempTop.append('farmland')
                    tempBot.append(shulker_dict['farmland'])
                elif i in [4, 11] and j in [4, 11]:
                    tempTop.append('water')
                    tempBot.append(shulker_dict['water'])
                else:
                    tempTop.append('grass')
                    tempBot.append(shulker_dict['grass'])
            bot.append(tempBot)
            top.append(tempTop)
        crops['order'] = crop_order
        return (bot, top, crops)


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



    def get_pathfinding_input(self, a, b, prevPos = None):
        result = []
        for r in self.shulker:
            temp = []
            for c in r:
                temp.append(pathfinding_value[c])
            result.append(temp)
        if not prevPos is None:
            for p in prevPos:
                result[p[0]][p[1]] = pathfinding_value['prev']
        result[b[0]][b[1]] = pathfinding_value['dest']
        result[a[0]][a[1]] = pathfinding_value['start']
        return result


    def get_farming_input(self):
        result = []
        for crop in self.crops['order']:
            result.append(farming_value[self.crops[crop]])
        return result


    def plant_crop(self, position):
        self.crops[position] = 'wheat_not_ready'
        self.growable[position] = random.randint(5, 40)


    def harvest_crop(self, position):
        self.crops[position] = 'empty'


    def grow_crops(self):
        to_delete = []
        for crop, cycle in self.growable.items():
            if cycle == 0:
                self.crops[crop] = 'wheat_ready'
                to_delete.append(crop)
            else:
                self.growable[crop] -= 1

        for d in to_delete:
            del self.growable[d]


    def reset_crops(self):
        for k in self.crops.keys():
            self.crops[k] = 'empty'


    def is_valid_move(self, pos):
        x, z = pos
        return not (x < 0 or x >= self.size or z < 0 or z >= self.size or \
                self.shulker[x][z] in ["brown_shulker_box", "blue_shulker_box"])
