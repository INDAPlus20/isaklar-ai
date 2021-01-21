#!/usr/bin/env python3

import arcade
from pymunk.vec2d import Vec2d
from logic import Player

def parsefile(filename="config.cfg"):

    f = open(filename, "r").readlines()
    conf = [i.strip("\n") for i in f if i[0] != "#" and i[0] != "\n"]
    
    movemapping = {
        "Up" : Vec2d(0, 1),
        "Down": Vec2d(0, -1),
        "Left": Vec2d(-1, 0),
        "Right": Vec2d(1, 0),
    }

    actionmapping = {
        "Shoot" : "shoot",
        "Dash" : "dash",
    }

    returndict = dict()
    p1_move_keys = dict()
    p2_move_keys = dict()
    p1_action_keys = dict()
    p2_action_keys = dict()
    p1_name = ""
    p2_name = ""
    ai_settings = ""
    p1_ai_path = ""
    p2_ai_path = ""

    for setting in conf:

        settingmapping = setting.split(" = ")
        
        ## Eval dangerous
        if "key" in settingmapping[0]:
            keytype = settingmapping[0].split("_")[2]
            maptype = eval(keytype + "mapping")
            func = "arcade.key." + settingmapping[1]
            mapping = maptype[settingmapping[0].split("_")[-1]]
            playerdict = eval(settingmapping[0][0:2].lower() + "_" + keytype + "_keys")
            playerdict[eval(func)] = mapping

        elif "AI" in settingmapping[0]:
            if settingmapping[0] == "AI":
                ai_settings = settingmapping[1]
            elif "P1" in settingmapping[0]:
                p1_ai_path = settingmapping[1]
            elif "P2" in settingmapping[0]:
                p2_ai_path = settingmapping[1]

        elif settingmapping[0] == "P1":
            p1_name = settingmapping[1]
        
        elif settingmapping[0] == "P2":
            p2_name = settingmapping[1]
    
    returndict = {
        "mode" : ai_settings,
        "p1_name" : p1_name,
        "p2_name" : p2_name,
        "p1_movement" : p1_move_keys,
        "p2_movement" : p2_move_keys,
        "p1_action" : p1_action_keys,
        "p2_action" : p2_action_keys,
        "p1_ai" : p1_ai_path,
        "p2_ai" : p2_ai_path,
    }
    
    return returndict

                


if __name__ == "__main__":
    x = parsefile("config.cfg")
    for y in x:
        print(y)