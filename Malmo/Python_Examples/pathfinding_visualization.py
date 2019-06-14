import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from PIL import Image
from scipy.signal import savgol_filter
from sklearn import preprocessing


def write_movement_frame(df, v):
    ''' Writes movement frame for video
    '''
    posX_step = df["posX"].values[0]
    posZ_step = df["posZ"].values[0]
    dest = (df["destX"].values[0], df["destZ"].values[0])
    
    pos_values = dict()
    # Fill in empty values
    for i in range(-1, 17):
        for j in range(-1, 17):
            if not (i, j) in pos_values.keys():
                pos_values[(i, j)] = 0

    df_pos = dict()
    x, z, v = [], [], []
    edf = pd.DataFrame(columns=['x', 'z', 'v'])
    for x, z in zip(posX_steps, posZ_steps):
        pos_values[(x, z)] += 1
        if (x, z) in zip(list(edf['x'].values), list(edf['z'].values)):
            edf.loc[df_pos[(x, z)]] = [x, z, pos_values[(x, z)]]
        else:
            edf.loc[dfNum] = [x, z, pos_values[(x, z)]]
            df_pos[(x, z)] = dfNum
            dfNum += 1
        

        plt.figure(figsize = (6.75, 5.66))
        s = plt.scatter('x', 'z', c = 'v', data = edf, marker = 's', s = 295, cmap = "Greens")
        plt.scatter(dest[0], dest[1], marker = 'x', color = 'r', s = 150)
        ## plot farmland
        for i in range(16):
            for j in range(16):
                if (i in [4, 6, 10, 12] and j in [4, 6, 10, 12]) or \
                   (i in [5, 11] and j in [4, 6, 10, 12]) or \
                   (j in [5, 11] and i in [4, 6, 10, 12]) :
                    plt.scatter(i, j, marker = '.', s = 100, color = "#654321")
            if i > 12:
                break
        plt.xticks([x for x in range(16)])
        plt.yticks([x for x in range(16)])
        plt.colorbar(mappable = s)
        plt.xlabel("X Coordinate")
        plt.ylabel("Z Coordinate")
        plt.title("Destination: {}".format(dest))
        plt.tight_layout()
        plt.savefig("data/heatmaps/anim_dest/last.png")
        plt.close()
        frameNum += 1
        video.write(cv2.imread("data/heatmaps/anim_dest/last.png"))
    video.release()
    video = None


def animate_dest_movement(df, dest, fname, reset_ep = False):
    # Creates animation based off movement for a certain destination
    dest_steps = df.loc[df["destZ"] == dest[1]]
    dest_steps = dest_steps.loc[dest_steps["destX"] == dest[0]]
    posX_steps = dest_steps["posX"].values
    posZ_steps = dest_steps["posZ"].values
    epNums = dest_steps["epNum"].values
    dest_values = dict()
    print(len(posZ_steps))

    # Fill in empty values
    for i in range(-1, 17):
        for j in range(-1, 17):
            if not (i, j) in dest_values.keys():
                dest_values[(i, j)] = 0

    width = 675
    height = 566
    FPS = 24
    seconds = 10
    fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')
    video = VideoWriter('data/heatmaps/anim_dest/{}.mov'.format(fname), fourcc, float(FPS), (width, height))

    frameNum = 0
    dfNum = 0
    df_pos = dict()
    edf = pd.DataFrame(columns=['x', 'z', 'v'])
    lastE = -1
    _time = time.time()
    for x, z, e in zip(posX_steps, posZ_steps, epNums):
        if frameNum % 1000 == 0:
            print(frameNum, time.time() - _time)
            _time = time.time()
        if reset_ep and not lastE == e:
            lastE = e
            dfNum = 0
            df_pos = dict()
            edf = pd.DataFrame(columns=['x', 'z', 'v'])
            for i in range(-1, 17):
                for j in range(-1, 17):
                    if not (i, j) in dest_values.keys():
                        dest_values[(i, j)] = 0

        dest_values[(x, z)] += 1
        if (x, z) in zip(list(edf['x'].values), list(edf['z'].values)):
            edf.loc[df_pos[(x, z)]] = [x, z, dest_values[(x, z)]]
        else:
            edf.loc[dfNum] = [x, z, dest_values[(x, z)]]
            df_pos[(x, z)] = dfNum
            dfNum += 1

        plt.figure(figsize = (6.75, 5.66))
        s = plt.scatter('x', 'z', c = 'v', data = edf, marker = 's', s = 295, cmap = "Greens")
        plt.scatter(dest[0], dest[1], marker = 'x', color = 'r', s = 150)
        plt.scatter(x, z, marker = '.', color = 'c', s = 250)
        ## plot farmland
        for i in range(16):
            for j in range(16):
                if (i in [4, 6, 10, 12] and j in [4, 6, 10, 12]) or \
                   (i in [5, 11] and j in [4, 6, 10, 12]) or \
                   (j in [5, 11] and i in [4, 6, 10, 12]) :
                    plt.scatter(i, j, marker = '.', s = 100, color = "#654321")
            if i > 12:
                break
        plt.xticks([x for x in range(16)])
        plt.yticks([x for x in range(16)])
        plt.colorbar(mappable = s)
        plt.xlabel("X Coordinate")
        plt.ylabel("Z Coordinate")
        plt.title("Destination: {}".format(dest))
        plt.tight_layout()
        plt.savefig("last.png")
        plt.close()
        frameNum += 1
        video.write(cv2.imread("last.png"))
    video.release()
    video = None


def plot_movement_frame(df, pathName):
    posX_steps = df["posX"].values
    posZ_steps = df["posZ"].values
    dest = (df.iloc[0]["destX"], df.iloc[0]["destZ"])
    pos_values = dict()
    # Fill in empty values
    for i in range(-1, 17):
        for j in range(-1, 17):
            if not (i, j) in pos_values.keys():
                pos_values[(i, j)] = 0

    for x, z in zip(posX_steps, posZ_steps):
        pos_values[(x, z)] += 1
            
    x, z, v = [], [], []
    edf = pd.DataFrame(columns=['x', 'z', 'v'])
    for i, pos in enumerate(pos_values.keys()):
        edf.loc[i] = [pos[0], pos[1], pos_values[pos]]

    plt.figure(figsize = (6.75, 5.66))
    s = plt.scatter('x', 'z', c = 'v', data = edf, marker = 's', s = 295)
    plt.scatter(dest[0], dest[1], marker = 'x', color = 'r')
    plt.xticks([x for x in range(16)])
    plt.yticks([x for x in range(16)])
    plt.colorbar(mappable = s)
    plt.tight_layout()
    savePath = "{}/{}_{}.png".format(pathName, dest[0], dest[1])
    plt.savefig(savePath)
    return savePath


def plot_dest_movement(df, dest):
    dest_steps = df.loc[df["destZ"] == dest[1]]
    dest_steps = dest_steps.loc[dest_steps["destX"] == dest[0]]
    posX_steps = dest_steps["posX"].values
    posZ_steps = dest_steps["posZ"].values
    dest_values = dict()

    # Fill in empty values
    for i in range(-1, 17):
        for j in range(-1, 17):
            if not (i, j) in dest_values.keys():
                dest_values[(i, j)] = 0

    for x, z in zip(posX_steps, posZ_steps):
        dest_values[(x, z)] += 1
            
    x, z, v = [], [], []
    edf = pd.DataFrame(columns=['x', 'z', 'v'])
    for i, pos in enumerate(dest_values.keys()):
        edf.loc[i] = [pos[0], pos[1], dest_values[pos]]

    plt.figure(figsize = (6.75, 5.66))
    s = plt.scatter('x', 'z', c = 'v', data = edf, marker = 's', s = 295, cmap = "Greens")
    plt.scatter(dest[0], dest[1], marker = 'x', color = 'r', s = 150)
    ## plot farmland
    for i in range(16):
        for j in range(16):
            if (i in [3, 5, 10, 12] and j in [3, 5, 10, 12]) or \
               (i in [4, 11] and j in [3, 5, 10, 12]) or \
               (j in [4, 11] and i in [3, 5, 10, 12]) :
                plt.scatter(i, j, marker = '.', s = 100, color = "#654321")
                
    plt.xticks([x for x in range(16)])
    plt.yticks([x for x in range(16)])
    plt.colorbar(mappable = s)
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title("Destination: {}".format(dest))
    plt.tight_layout()
    plt.savefig("data/heatmaps/by_destination_new/{}_{}.png".format(dest[0], dest[1]))
    plt.close()


def add_to_dict(d, k, v):
    ''' Adds value to a dict
        Wrap value v in data type it will be joining
    '''
    if type(v) is set:
        if not k in d.keys():
            d[k] = v
        else:
            d[k] = d[k].union(v)
        return d
    if not k in d.keys():
        d[k] = v
    else:
        d[k] += v
    return d


def graph_pathfinding_reward(df):
    d = df[['epNum', 'reward']]
    g = d.transform(lambda x: (x - x.mean)/x.std())
    print(g.head())
    y = g.values
    plt.figure(figsize = (10, 5))
    plt.title('Agent Reward')
    plt.xlabel('Step Number')
    plt.ylabel('Avg Reward')
    plt.plot(y)
    plt.show()
    print(d.head())


def graph_success_by_dest(df):
    d = df[['epNum', 'destX', 'destZ', 'success']]
    visited_dests = []
    performance = {}
    window = []
    windowLen = 1000
    for dest in zip(d['destX'].values, d['destZ'].values):
        #dest = (int(_d[0].replace('(', '')), int(_d[1].replace(')','')))
        if not dest in visited_dests:
            print(dest)
            visited_dests.append(dest)
            eps = list(d.groupby(['destX', 'destZ']).get_group(dest).groupby('epNum').groups.keys())
            x = eps
            y = []
            dist = []
            first = None
            for ep in eps:
                t = d.loc[(d['epNum']==ep) & d['success'] == 1]
                if not t.empty:
                    window.append(1)
                else:
                    window.append(0)
                if len(window) > windowLen:
                    window.pop(0)
                y.append(float(sum(window))/len(window))
                
            performance[dest] = y[-1]
            ## Check if rolling avergae produces better results
            #rollingFrame = pd.DataFrame(np.array(y), columns=['y'])
            #y = rollingFrame.rolling(10).mean()

            plt.title("Destination {} Navigation Performance".format(dest))
            plt.xlabel("Episode Number")
            plt.ylabel("Success %")
            plt.ylim(-0.05, 1.05)
            plt.scatter(x, y, alpha = 0.5)
            #plt.plot(np.linspace(0, max(x)), np.poly1d(np.polyfit(x, y, 3))(np.linspace(0, max(x))), color='black')
            plt.savefig("data/nav_success/{}_{}.png".format(int(dest[0]), int(dest[1])))
            plt.close()

            #fig = plt.figure()
            #ax = plt.axes(projection='3d')
            
            
    return performance


def get_best_dest(df):
    solves, occur = dict(), dict()
    for idx, row in df.iterrows():
        dest = (row['destX'], row['destZ'])
        s = row['success']
        epNum = row['epNum']
        occur = add_to_dict(occur, dest, {epNum})
        if s == 1:
            solves = add_to_dict(solves, dest, 1)

    for k, v in occur.items():
        if not k in solves.keys():
            occur[k] = 0.0
        else:
            occur[k] = float(solves[k])/len(v)

    return occur


def graph_avg_dist_by_dest(df):
    dests = dict()
    for idx, row in df.iterrows():
        dest = (row['destX'], row['destZ'])
        dist = row['dist']
        if not dest in dests.keys():
            dests[dest] = [dist]
        else:
            dests[dest].append(dist)



def graph_testing():
    plt.figure(figsize = (15, 5))
    choiceFile = 'data/agent_testing/choice_optimality.csv'
    pathFile = 'data/agent_testing/path_optimality.csv'
    stateFile = 'data/agent_testing/state_value.csv'
    state = pd.read_csv(stateFile)
    y = state['stateValue'].values
    print(y)
    plt.plot(state['epNum'].values, y, label = 'Bad Agent')

##    path = pd.read_csv(pathFile)
##    y = path['optimality'].rolling(100).mean().values
##    min_max_scaler = preprocessing.MinMaxScaler()
##    y_scaled = min_max_scaler.fit_transform(y.reshape(-1, 1))
##    df_normalized = pd.DataFrame(y_scaled)
##    plt.plot(path['epNum'].values, df_normalized.values, label = 'normalized path optimality')
##
##    state = pd.read_csv(stateFile)
##    y = state['stateValue'].values.astype(float)
##    min_max_scaler = preprocessing.MinMaxScaler()
##    y_scaled = min_max_scaler.fit_transform(y.reshape(-1, 1))
##    df_normalized = pd.DataFrame(y_scaled)
##    plt.plot(state['epNum'].values,df_normalized.values, label = 'normalized state value')

    choiceFile = 'data/agent_testing/choice_optimality1.csv'
    pathFile = 'data/agent_testing/path_optimality1.csv'
    stateFile = 'data/agent_testing/state_value1.csv'
    state = pd.read_csv(stateFile)
    y = state['stateValue'].values
    plt.plot(state['epNum'].values, y, label = 'Good Agent')

    #plt.xlim(100, 825)
    plt.legend()
    #plt.show()
    plt.savefig('stateCompare.png')


def print_dict_sorted(d):
    dest = [(v, k) for k, v in d.items()]
    dest.sort(reverse=True)
    for v, k in dest:
        print('{}: {}'.format(k, v))

if __name__ == "__main__":
    data = pd.read_csv("data/pathfinding_train_data/default/pathfinding_10.csv")
    t = time.time()
    #graph_pathfinding_reward(data)
    print(len(data))
    #graph_testing()
    #graph_success_by_dest(data)
    #print(performance)
    #plt.bar(performance.values(), performance.keys())
    #plt.savefig('bar.png')
    #dest = get_best_dest(data)
##    for i in range(16):
##        for j in range(16):
##            if (i in [3, 5, 10, 12] and j in [3, 5, 10, 12]) or \
##               (i in [4, 11] and j in [3, 5, 10, 12]) or \
##               (j in [4, 11] and i in [3, 5, 10, 12]) :
##                dest = (i, j)
##                plot_dest_movement(data, dest)
##    start = len(data) - int(1.35*(len(data)//16))
##    end = len(data) - len(data)//16
##    animate_dest_movement(data[start:end], dest, "dest_{}_{}".format(dest[0], dest[1]))
    #print_dict_sorted(dest)
