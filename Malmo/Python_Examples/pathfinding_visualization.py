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


def animate_episode_movement(df, epNum):
    # Creates animation based off movement for a certain episode
    episode_steps = df.loc[df["epNum"] == epNum]
    posX_steps = episode_steps["posX"].values
    posZ_steps = episode_steps["posZ"].values
    dest = (episode_steps.iloc[0]["destX"], episode_steps.iloc[0]["destZ"])
    pos_values = dict()
    # Fill in empty values
    for i in range(-1, 17):
        for j in range(-1, 17):
            if not (i, j) in pos_values.keys():
                pos_values[(i, j)] = 0

    width = 675
    height = 566
    FPS = 24
    seconds = 10
    fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')
    video = VideoWriter('data/move.mov', fourcc, float(FPS), (width, height))

    frameNum = 0
    dfNum = 0
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
    for x, z, e in zip(posX_steps, posZ_steps, epNums):
        if frameNum % 1000 == 0:
            print(frameNum)
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


def plot_episode_movement(df, epNum):
    episode_steps = df.loc[df["epNum"] == epNum]
    posX_steps = episode_steps["posX"].values
    posZ_steps = episode_steps["posZ"].values
    dest = (episode_steps.iloc[0]["destX"], episode_steps.iloc[0]["destZ"])
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
    plt.show()
    



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
            if (i in [4, 6, 10, 12] and j in [4, 6, 10, 12]) or \
               (i in [5, 11] and j in [4, 6, 10, 12]) or \
               (j in [5, 11] and i in [4, 6, 10, 12]) :
                plt.scatter(i, j, marker = '.', s = 100, color = "#654321")
                
    plt.xticks([x for x in range(16)])
    plt.yticks([x for x in range(16)])
    plt.colorbar(mappable = s)
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title("Destination: {}".format(dest))
    plt.tight_layout()
    plt.savefig("data/heatmaps/by_destination_new2/{}_{}.png".format(dest[0], dest[1]))
    plt.close()


def distance(x1, x2, y1, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def get_best_dest(df):
    solves = dict()
    result = None
    max_solves = -1
    for idx, row in df.iterrows():
        pX = row['posX']
        pZ = row['posZ']
        dX = row['destX']
        dZ = row['destZ']
        if distance(pX, dX, pZ, dZ) <= 1 and (pX == dX or pZ == dZ):

            if (pX in [4, 6, 10, 12] and pZ in [4, 6, 10, 12]) or \
               (pX in [5, 11] and pZ in [4, 6, 10, 12]) or \
               (pZ in [5, 11] and pX in [4, 6, 10, 12]) :
                continue

            
            if not (dX, dZ) in solves.keys():
                solves[(dX, dZ)] = 0
            else:
                solves[(dX, dZ)] += 1
                
            if solves[(dX, dZ)] > max_solves:
                result = (dX, dZ)
                max_solves = solves[(dX, dZ)]
    print(result, max_solves)
    return result
        



if __name__ == "__main__":
    data = pd.read_csv("data/movement2.csv")
    t = time.time()
    dest = get_best_dest(data)
    animate_dest_movement(data, dest, "dest_{}_{}".format(dest[0], dest[1]))
    print(time.time() - t)
