import random
import farm as fg
import numpy as np
from algorithms.dikjstra import get_path_dikjstra


def start_csv():
    f = open('test_data2.csv', 'w')
    line = ''
    for i in range(32):
        line += '{},'.format(i)
    f.write(line + 'y\n')
    f.close()


def add_csv(buffer):
    f = open('test_data2.csv', 'a')
    line = ''
    for arg in buffer:
        line += '{},'.format(arg)
    f.write(line[:-1] + '\n')
    f.close()


# Essentially hardcoded solution
if __name__ == '__main__':
    farm = fg.Farm()
    num_episodes = 1000
    num_data_points = 100
    start_csv()
    for i in range(num_episodes):
        start = random.choice(farm.farmable)
        if i % 10:
            print(i)
        for j in range(num_data_points):
            start = random.choice(farm.farmable)
            to_write = []
            shortest_dist = 100
            result_plot = None
            result_value = start_idx = result_idx = -1
            for k, plot in enumerate(farm.farmable):
                if (plot == start):
                    start_idx = k
                state = np.reshape(farm.get_pathfinding_input(start, plot), [256])
                optimal_path = len(get_path_dikjstra(start, plot, state)[0])-1
                plot_value = fg.farming_value[farm.crops[plot]]
                to_write.append(plot_value)
                if plot_value in [0, 5] and optimal_path <= shortest_dist:
                    result_plot = plot
                    shortest_dist = optimal_path
                    result_idx = k
                    result_value = plot_value
            to_write[start_idx] = fg.farming_value['position']
            to_write.append(result_idx)
            add_csv(to_write)
            start = result_plot
            farm.grow_crops()
            if result_value == 0:
                farm.plant_crop(result_plot)
            else:
                farm.harvest_crop(result_plot)
        farm.reset_crops()

    print('DONE\n')
