import arcade
import random

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Test Application"
SCALING = 2.0

class FlyingSprite(arcade.Sprite):

    def update(self):
        super().update()

        if self.right < 0:
            self.remove_from_sprite_lists()


class Game(arcade.Window):
    
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        self.enemies_list = arcade.SpriteList()
        self.lotus_list = arcade.SpriteList()
        self.all_sprites = arcade.SpriteList()
    
    def setup(self):

        self.paused = False

        arcade.set_background_color(arcade.color.RICH_ELECTRIC_BLUE)

        self.player = arcade.Sprite("sprites/duck_small.png", 0.5)
        self.player.center_y = self.height / 2
        self.player.left = 10
        self.all_sprites.append(self.player)

        arcade.schedule(self.add_enemy, 0.5)
        arcade.schedule(self.add_lotus, 2.0)

    def add_enemy(self, delta_time: float):
        enemy = FlyingSprite("sprites/frog_small.png", 0.25)

        enemy.left = random.randint(self.width, self.width + 80)
        enemy.top = random.randint(10, self.height - 10)

        enemy.velocity = (random.randint(-10, -5), 0)

        self.enemies_list.append(enemy)
        self.all_sprites.append(enemy)

    def add_lotus(self, delta_time: float):
        lotus = FlyingSprite("sprites/lotus_small.png", 0.5)

        lotus.left = random.randint(self.width, self.width + 80)
        lotus.top = random.randint(10, self.height - 10)

        lotus.velocity = (random.randint(-5, -1), 0)

        self.lotus_list.append(lotus)
        self.all_sprites.append(lotus)

    def on_draw(self):
        arcade.start_render()
        self.all_sprites.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.Q:
            arcade.close_window()
        
        if symbol == arcade.key.P:
            self.paused = not self.paused
        
        if symbol == arcade.key.W:
            self.player.change_y = 5
        
        if symbol == arcade.key.S:
            self.player.change_y = -5
        
        if symbol == arcade.key.A:
            self.player.change_x = -5
        
        if symbol == arcade.key.D:
            self.player.change_x = 5
    
    def on_key_release(self, symbol, modifiers):
        if symbol == arcade.key.W or symbol == arcade.key.S:
            self.player.change_y = 0
        if symbol == arcade.key.A or symbol == arcade.key.D:
            self.player.change_x = 0

    def on_update(self, delta_time: float):

        if self.paused:
            return
        
        if self.player.collides_with_list(self.enemies_list):
            self.destroy()
            self.setup()

        self.all_sprites.update()

        if self.player.top > self.height:
            self.player.top = self.height

        if self.player.right > self.width:
            self.player.right = self.width

        if self.player.bottom < 0:
            self.player.bottom = 0

        if self.player.left < 0:
            self.player.left = 0
    
    def destroy(self):

        self.enemies_list = arcade.SpriteList()
        self.lotus_list = arcade.SpriteList()
        self.all_sprites = arcade.SpriteList()

        arcade.unschedule(self.add_enemy)
        arcade.unschedule(self.add_lotus)



if __name__ == "__main__":
    app = Game(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    app.setup()
    arcade.run()