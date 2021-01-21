import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import model_from_json

import json

#from tensorflow.keras.models import load_model

from collections import deque

grid_size = 56*32 # size of the game board
hidden_size = 100 # amount of hidden neurons per layer
num_actions = 32

# initialize the model

with open("ai/model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
model.load_weights("ai/model.h5")
model.compile("sgd", "mse")
#model = Sequential()
#model.add(Dense(hidden_size,  activation='relu')) #input_shape=(None, None, 2),
#model.add(Dense(hidden_size, activation='relu'))
#model.add(Dense(num_actions))
#model.compile(SGD(lr=.2), "mse")
#model.load_weights("model.h5")

player = None
enemy = None

frame_stack = deque()

# generate move dictionary
dictionary = list()
for y_mov in range(0, 3):
    for x_mov in range(0, 3):
        dictionary.append([(0, x_mov), (1, y_mov), (2, 0), (3, 0)])
        dictionary.append([(0, x_mov), (1, y_mov), (2, 1), (3, 0)])
        dictionary.append([(0, x_mov), (1, y_mov), (2, 0), (3, 1)])
        dictionary.append([(0, x_mov), (1, y_mov), (2, 1), (3, 1)])


def predict(agent, observations, action_space):
    global player
    player = agent.player
    flat_obs = np.zeros(grid_size + 4) # might have to preallocate some memory here with [[0,0]*32]*52
    # parse and format observations
    i = 0
    for x in observations:
        for e in x:
            
            if e[0] == "walls":
                first = 1
            elif e[0] == "floor":
                first = 0
            elif e[0] == "deadly":
                first = -1
            if e[1] == '':
                # empty tile
                second = 0
            else:
                if e[1].name == player.name:
                    second = 10 
                elif e[1].name == "Bill":
                    second = -30
                else:
                    global enemy
                    if enemy == None:
                        enemy = e[1]
                    second = 20
            flat_obs[i] = first + second
            i +=1

    # add current health to observation
    flat_obs[grid_size] = player.health
    flat_obs[grid_size + 1] = player.lives

    # add enemy health to observation
    flat_obs[grid_size + 2] = enemy.health
    flat_obs[grid_size + 3] = enemy.lives

    # add to stack
    frame_stack.append(flat_obs)
    if len(frame_stack) > 3:
        frame_stack.popleft()



    # predict
    action = 0
    if len(frame_stack) == 3: 
        
        # format input
        input = list()
        input.extend(frame_stack[0])
        input.extend(frame_stack[1])
        input.extend(frame_stack[2])
        np_input = np.array(input).reshape(-1, len(input))
        

        # make prediction
        q = model.predict(np_input)
        action = np.argmax(q[0])

    # parse action
    global dictionary
    output = dictionary[action]

    return output