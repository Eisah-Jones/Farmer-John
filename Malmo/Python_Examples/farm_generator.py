import random

plot1 = [["farmland",  "farmland", "farmland"],
         ["farmland", "water", "farmland"],
         ["farmland",  "farmland", "farmland"]]


plot2 = [["farmland", "water", "farmland"],
         ["farmland", "water", "farmland"],
         ["farmland", "water", "farmland"],
         ["farmland", "water", "farmland"],
         ["farmland", "water", "farmland"]]


plot3 = [[ "farmland",  "farmland",  "farmland",  "farmland",  "farmland"],
         ["water", "water", "water", "water", "water"],
         [ "farmland",  "farmland",  "farmland",  "farmland",  "farmland"]]


plot4 = [["water",   "farmland",   "farmland",  "stone",   "farmland",   "farmland", "water"],
         [ "farmland", "planks", "planks", "planks", "planks", "planks",  "farmland"],
         [ "farmland", "planks",   "farmland",   "farmland",   "farmland", "planks",  "farmland"],
         ["stone", "planks",   "farmland",  "water",   "farmland", "planks", "stone"],
         [ "farmland", "planks",   "farmland",   "farmland",   "farmland", "planks",  "farmland"],
         [ "farmland", "planks", "planks", "planks", "planks", "planks",  "farmland"],
         ["water",   "farmland",   "farmland",  "stone",   "farmland",   "farmland", "water"]]


shulker_dict = {"farmland": "brown_shulker_box",
                "water" : "blue_shulker_box",
                "planks": "white_shulker_box",
                "stone": "white_shulker_box",
                "grass": "white_shulker_box"}


def get_static_plot():
    bot = []
    top = []
    for i in range(16):
        tempBot = []
        tempTop = []
        for j in range(16):
            if (i in [4, 6, 10, 12] and j in [4, 6, 10, 12]) or \
               (i in [5, 11] and j in [4, 6, 10, 12]) or \
               (j in [5, 11] and i in [4, 6, 10, 12]) :
                tempTop.append("farmland")
                tempBot.append(shulker_dict["farmland"])
            elif i in [5, 11] and j in [5, 11]:
                tempTop.append("water")
                tempBot.append(shulker_dict["water"])
            else:
                tempTop.append("grass")
                tempBot.append(shulker_dict["grass"])
        bot.append(tempBot)
        top.append(tempTop)
    return [bot, top]
            


def spawn_plots(f, n):

    plots = [plot1, plot3, plot2]
    availableSpots = []
    availableAll = []
    for plot in plots:
        temp = []
        for i in range(1, n-1):
            for j in range(1, n-1):
                if(not (i, j) in availableAll):
                    availableAll.append((i, j))
                if (j + len(plot[0]) < n-1 and i + len(plot) < n-1):
                    temp.append((i, j))
        availableSpots.append(temp)

    spawns = []
    for plot, available, probs in zip(plots, availableSpots, [0.1, 0.5, 0.5]):
        temp = []
        for a in available:
            if(a in availableAll and random.random() < probs):
                isValid = True
                for i in range(-1, len(plot)):
                    for j in range(-1, len(plot[0])):
                        if ((a[0]+i, a[1]+j) in availableAll):
                            availableAll.remove((a[0]+i, a[1]+j))
                        else:
                            isValid = False
                            break
                    if (not isValid):
                        break
                if isValid:
                    temp.append(a)
        spawns.append(temp)

    for plot, spawn in zip(plots, spawns):
        for s in spawn:
            for i in range(len(plot)):
                for j in range(len(plot[0])):
                    f[0][s[0]+i][s[1]+j] = shulker_dict[plot[i][j]]
                    f[1][s[0]+i][s[1]+j] = plot[i][j]  
    return f


def initialize_farm(n):
    bot = list()
    top = list()
    for i in range(n):
        temp_top = list()
        temp_bot = list()
        for j in range(n):
            temp_bot.append("white_shulker_box")
            temp_top.append("grass")
        bot.append(temp_bot)
        top.append(temp_top)
        
    return [bot, top]


def generate_farm(size):
    ''' Generate a list representation of the farm
    '''
##    farm = initialize_farm(size)
##    farm = spawn_plots(farm, size)
    return get_static_plot()
