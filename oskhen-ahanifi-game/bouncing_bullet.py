import arcade
import math
import time
from pymunk.vec2d import Vec2d
from dataclasses import dataclass
from typing import List

import logic
import ai_interface
import parseconf



SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
SCREEN_TITLE = "Under Development"
SCALING = 0.1
PLAYER_1_SPEED = 5
PLAYER_2_SPEED = 3
BULLET_SPEED = 7
TILE_SCALING = 1.5
COORDINATE_MAPPING = 1 / 16 / TILE_SCALING

# Classes

class GameView(arcade.View):
    """Main welcome window
    """

    def __init__(self):
        """Initialize the window
        """
        # Call the parent class constructor
        super().__init__()

        self.bullets = arcade.SpriteList()
        self.players = arcade.SpriteList()
        self.nonpassable = arcade.SpriteList()
        self.all_sprites = arcade.SpriteList()

        self.boardstate = [[["" for _ in range(2)] for _ in range(32)] for _ in range(56)] # mapsize 32x56 tiles

        self.environment_damage=5
        self.damage_intervall=3
        self.player_damage_timers= list()

        # Physics engine currently only handles player-wall collisions
        self.physics_engine = None


    def setup(self):

        self.setup_map("./maps/map_2/map.tmx")

        self.bg_sprites = [["walls", self.wall_list], ["floor", self.floor_list], ["deadly", self.deadly_list]]
        self.fg_sprites = [self.bullets, self.players]

        ## Parse Config File

        settings = parseconf.parsefile()

        # Player setups

        self.player1 = logic.Player(self, "sprites/duck_pixel.png", TILE_SCALING -0.2, settings["p1_movement"], settings["p1_action"], 100, self.window.height / 2, settings["p1_name"] ,Vec2d(1,0))

        self.player2 = logic.Player(self, "sprites/duck_pixel_red.png", TILE_SCALING -0.2, settings["p2_movement"], settings["p2_action"], self.window.width - 200, self.window.height / 2, settings["p2_name"] ,Vec2d(-1,0))

        self.agents=list()
        mode= settings["mode"]
        if mode!="PvP":
            self.player2.is_ai=True
            agent=ai_interface.Agent(settings["p2_ai"],self.player2)
            agent.observation=self.boardstate
            self.agents.append(agent)
            if mode == "EvE":
                self.player1.is_ai=True
                agent=ai_interface.Agent(settings["p1_ai"],self.player1)
                agent.observation=self.boardstate
                self.agents.append(agent)



        self.players.append(self.player1)
        self.players.append(self.player2)
        self.nonpassable.append(self.player1)
        self.nonpassable.append(self.player2)

        for player in self.players:
            # Physics engine currently only handles player-wall collisions
            player.physics_engine = arcade.PhysicsEngineSimple(
            player, self.nonpassable)

        for _ in range(len(self.players)):
            self.player_damage_timers.append(self.damage_intervall)

        ## Boardstate background setup

        for spritelist in self.bg_sprites:
            for sprite in spritelist[1]:
                x = [int(i * COORDINATE_MAPPING) for i in sprite.position]
                self.boardstate[x[0]][x[1]][0] = spritelist[0]  

        
               

    def setup_map(self,map_path):
        # --- Load in a map from the tiled editor ---

        # Name of map file to load
        map_name = map_path
        # Name of the layer in the file that has our platforms/walls
        walls_layer_name = 'walls'
        # Name of the layer that has floor
        floor_layer_name = 'floor'
        # Name of the layer that has deadly tiles
        deadly_layer_name = "toxic"

        my_map = arcade.tilemap.read_tmx(map_name)
        self.wall_list=arcade.SpriteList(is_static=True,use_spatial_hash=True)
        self.wall_list.extend( arcade.tilemap.process_layer(map_object=my_map,
                                                      layer_name=walls_layer_name,
                                                      scaling=TILE_SCALING,
                                                      use_spatial_hash=True))

        #-- Floor
        self.floor_list=arcade.SpriteList(is_static=True)
        self.floor_list.extend(arcade.tilemap.process_layer(
            my_map, floor_layer_name, TILE_SCALING))
        #-- Deadly
        self.deadly_list=arcade.SpriteList(is_static=True,use_spatial_hash=True)
        self.deadly_list.extend(arcade.tilemap.process_layer(
            my_map, deadly_layer_name, TILE_SCALING))

        self.all_sprites.extend(self.deadly_list)
        self.all_sprites.extend(self.floor_list)  # extend appends spriteList #No it doesn't, it *extends* spriteList..
        self.all_sprites.extend(self.wall_list)
        self.nonpassable.extend(self.wall_list)

    def on_draw(self):
        """Called whenever you need to draw your window
        """     
        # Clear the screen and start drawing
        arcade.start_render()
        self.floor_list.draw()
        self.deadly_list.draw()
        self.wall_list.draw()
        self.players.draw()
        self.bullets.draw()
        offset_x=0
        for player in self.players:
            num_dashes = int(player.health/10)
            text = f"|"+'#'*num_dashes+'_'*(10-num_dashes)+'|'+" "*3 + "X"*player.lives
            arcade.draw_text(text, 30+offset_x, SCREEN_HEIGHT-30, arcade.color.RADICAL_RED, 20)
            offset_x+=300
        self.players.draw()


    def get_observation(self):
        return self.boardstate

    def on_update(self, delta_time):

        self.boardstate = [[[y[0], ""] for y in x] for x in self.boardstate]

        for spritelist in self.fg_sprites:
            for sprite in spritelist:
                x = [int(i * COORDINATE_MAPPING) for i in sprite.position]
                self.boardstate[x[0]][x[1]][1] = sprite

        #########################################################################
        # Switched the places of these two to allow the ai to know when it's lost or won
        ######################################################################

        for agent in self.agents:
            agent.observation = self.boardstate
            inputs=agent.player.input_context
            inputs.move_keys_pressed=inputs.move_keys_pressed.fromkeys(inputs.move_keys_pressed,False)
            inputs.abilities_pressed=inputs.abilities_pressed.fromkeys(inputs.abilities_pressed,False)
            keys=agent.predict()
            for key in keys:
                agent.player.on_key_press(key,None)

        for i,player in enumerate(self.players):
            if player.lives == 0:
                self.gameover(player)

            self.player_damage_timers[i]+=delta_time

        ########################################################################
        ########################################################################

        # Bullet bounces
        for bullet in self.bullets:
            bounced = False
            bullet.center_x += bullet.change_x

            walls_hit = arcade.check_for_collision_with_list(
                bullet, self.wall_list)

            for wall in walls_hit:
                if bullet.change_x > 0:
                    bullet.right = wall.left
                elif bullet.change_x < 0:
                    bullet.left = wall.right

            if len(walls_hit) > 0:
                bullet.change_x *= -1
                bullet.bounces += 1
                bounced = True

            bullet.center_y += bullet.change_y

            walls_hit = arcade.check_for_collision_with_list(
                bullet, self.wall_list)

            for wall in walls_hit:
                if bullet.change_y > 0:
                    bullet.top = wall.bottom
                elif bullet.change_y < 0:
                    bullet.bottom = wall.top

            if len(walls_hit) > 0:
                bullet.change_y *= -1
                bullet.bounces += 1
                bounced = True

            if bounced == True:
                angle = math.atan2(bullet.change_y, bullet.change_x)
                bullet.angle = math.degrees(angle)
                bullet.update()
                

            else:
                players_hit=arcade.check_for_collision_with_list(bullet,self.players)
                for player in players_hit:
                    player.take_damage(10)
                    bullet.destroy()

        #Environmental Damage
        for i,player in enumerate(self.players):
            if self.player_damage_timers[i]>self.damage_intervall:
                if player.collides_with_list(self.deadly_list):
                    self.player_damage_timers[i]=0
                    player.take_damage(10)
            

        for player in self.players:
            player.update(delta_time)

        for player in self.players:
            hit_list = player.physics_engine.update()
            if len(hit_list) > 0:
                player.collided = True
            else:
                player.collided = False
        

    def on_key_press(self, key, modifiers):
        for player in self.players:
            if not player.is_ai:
                player.on_key_press(key, modifiers)

    def on_key_release(self, key, modifiers):
        for player in self.players:
            if not player.is_ai:
                player.on_key_release(key, modifiers)
    
    def gameover(self, player):
        gameover_view = GameOverView(player.name)
        self.window.show_view(gameover_view)

class MenuView(arcade.View):

    def __init__(self):
        super().__init__()

        self.playscreen = arcade.load_texture("sprites/play.png")
        self.optionscreen = arcade.load_texture("sprites/options.png")
        self.quitscreen = arcade.load_texture("sprites/quit.png")

        self.screenlist = [self.playscreen, self.optionscreen, self.quitscreen]
        self.currentselection = 0
        
    
    def on_draw(self):

        arcade.start_render()
        self.screenlist[self.currentselection].draw_sized(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, SCREEN_WIDTH, SCREEN_HEIGHT)

    def on_key_press(self, key, modifiers):
        
        if key == arcade.key.S:
            self.currentselection = (self.currentselection + 1) % 3
        
        if key == arcade.key.W:
            self.currentselection = (self.currentselection - 1) % 3
        
        if key == arcade.key.ENTER:
            if self.currentselection == 0:
                game_view = GameView()
                game_view.setup()
                self.window.show_view(game_view)

            elif self.currentselection == 2:
                self.window.close()
        
class GameOverView(arcade.View):

    def __init__(self, playername):
        super().__init__()
        self.text = f"{playername} lost the game!"

    def on_show(self):
        arcade.set_background_color(arcade.color.RED_DEVIL)
    
    def on_draw(self):
        arcade.start_render()
        arcade.draw_text(self.text, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, arcade.color.WHITE, font_size=50, anchor_x="center")
        arcade.draw_text("Press R to play again", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 100, arcade.color.WHITE, font_size=20, anchor_x="center")
        arcade.draw_text("Press M to go to main menu", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 150, arcade.color.WHITE, font_size=20, anchor_x="center")
        arcade.draw_text("Press Q to quit", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 200, arcade.color.WHITE, font_size=20, anchor_x="center")
        
    def on_key_press(self, key, modifiers):

        arcade.set_background_color(arcade.color.BLACK)

        if key == arcade.key.Q:
            self.window.close()
        
        if key == arcade.key.M:
            start_view = MenuView()
            window.show_view(start_view)
        
        if key == arcade.key.R:
            game_view = GameView()
            game_view.setup()
            self.window.show_view(game_view)

# Main code entry point
if __name__ == "__main__":
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    start_view = MenuView()
    window.show_view(start_view)
    arcade.run()
