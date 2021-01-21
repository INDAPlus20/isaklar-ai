import bouncing_bullet as bb
import arcade
import math
import time
from pymunk.vec2d import Vec2d
from dataclasses import dataclass
#from typing import List
import json

import logic
import ai_interface
import parseconf

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import model_from_json

from collections import deque
import random

# parameters
epsilon = .1  # exploration
#epoch = 100
max_memory = 500
hidden_size = 100
#batch_size = 50
grid_size = 56*32
num_actions = 3*3*2*2
future_discount = .9
cutoff = max_memory # the point at which the NN starts training
episodes_until_save = 5
steps = 0 # amount of timesteps taken since training



model = ai_interface.model
if model == None:
    print("No model in ai_inteface")
    try:
        with open("ai/model.json", "r") as jfile:
            model = model_from_json(json.load(jfile))
            model.load_weights("ai/model.h5")
            model.compile(SGD(lr=.2), "mse")
            print("found model!")
    except:
       print("no weights found")
       model = Sequential()
       model.add(Dense(hidden_size,  activation='relu')) #input_shape=(None, None, 2),
       model.add(Dense(hidden_size, activation='relu'))
       model.add(Dense(num_actions))
       model.compile(SGD(lr=.2), "mse")

ai_interface.model = model

#print(model.get_layer(index=0).input_shape)

player = None
enemy = None

frame_stack = deque() # a stack of three frames to more acurately see motion

previous_action = None
previous_input = np.zeros((grid_size + 4)*3).reshape(-1, (grid_size + 4)*3)

replay_mem = list()

dictionary = list()
for y_mov in range(0, 3):
    for x_mov in range(0, 3):
        dictionary.append([(0, x_mov), (1, y_mov), (2, 0), (3, 0)])
        dictionary.append([(0, x_mov), (1, y_mov), (2, 1), (3, 0)])
        dictionary.append([(0, x_mov), (1, y_mov), (2, 0), (3, 1)])
        dictionary.append([(0, x_mov), (1, y_mov), (2, 1), (3, 1)])

def calc_reward(previous, current):
    damage_reward = -1
    death_reward = -1
    loss_reward = -2
    win_reward = 2

    global grid_size
    health_diff = current[grid_size] - previous[grid_size]
    lives_diff = current[grid_size + 1] - previous[grid_size + 1]

    enemy_health_diff = current[grid_size + 2] - previous[grid_size + 2]
    enemy_lives_diff = current[grid_size + 3] - previous[grid_size + 3]

    lives = current[grid_size + 1]
    enemy_lives = current[grid_size + 3]
    reward = 0
    if lives == 0:
        reward += loss_reward
    elif enemy_lives == 0:
        reward += win_reward
    else:
        if health_diff != 0:
            reward += damage_reward
        if lives_diff != 0:
            reward += lives_diff

    return reward


def predict(agent, observations, action_space):
    global previous_input
    global previous_action
    global model
    global player
    global steps

    steps += 1
    
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

    
    action = 0
    # check if game_over
    global cutoff
  
    if player.health == 0 or enemy.health == 0 or steps%cutoff == 0:
        print("training")
        batch = replay_mem[2:]
        inputs = np.zeros((len(batch), batch[0][0].shape[-1])) # allocate mem
        #print(batch[0][0].shape[0])
        #print(inputs.shape)
        for i in range(len(batch)):
            inputs[i] = batch[i][0]
        #print(inputs[0])
        targets = np.zeros((inputs.shape[0], num_actions))
        

        global future_discount 
        # train the NN based on replay memory
        for i in range(len(batch)):
            state_t = batch[i][0]
            action_t = batch[i][1]
            state_t1 = batch[i][2]
            reward_t = batch[i][3]
            
            targets[i] = model.predict(state_t)
            Q_sa = model.predict(state_t1)
            if i == len(batch) - 1:
                targets[i, action_t] = reward_t + future_discount * np.max(Q_sa)
            else: 
                # if game ended
                targets[i, action_t] = reward_t


        # now train on the data
        model.train_on_batch(inputs, targets)
        print("finished training")

        # Save trained model weights and architecture
        global episodes_until_save
        if steps/cutoff >= episodes_until_save:
            model.save_weights("ai/model.h5", overwrite=True)
            with open("ai/model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
            steps = 0

    else:
        # predict
        if len(frame_stack) == 3: 
            
            
            # format input
            input = list()
            input.extend(frame_stack[0])
            input.extend(frame_stack[1])
            input.extend(frame_stack[2])
            np_input = np.array(input).reshape(-1, len(input))

            # calculate reward and append to memory
            reward = calc_reward(frame_stack[1], frame_stack[2])
            
            replay_mem.append((previous_input, previous_action, np_input, reward))
            #if len(replay_mem) > max_memory:
            #    replay_mem.popleft()
            # print(len(replay_mem))
            # save input
            previous_input = np_input
            
            if random.random() < epsilon:
                
                action = random.randrange(num_actions)
            elif steps%3:
                action = previous_action
            else:
                # make prediction
                q = model.predict(np_input)
                action = np.argmax(q[0])

            # save
            previous_action = action
        
    
    global dictionary
    # parse action
    output = dictionary[action]
    return output
        
    
    # TODO
    # pre-allocate memory to see if faster
    # train based on memory


