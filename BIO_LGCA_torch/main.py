import pygame
import torch

from Camera import Camera
from Automaton import *
from models import *
import cv2
import time

# Initialize the automaton
W, H = 60, 60
model = Depth_Aware_Lattices()
auto = BioLgcaSquaredAuto((W, H), model.init_world(W, H), model.interaction_function, model.draw_function)
# =============================================================================

# Initialize the pygame screen
pygame.init()
screen = pygame.display.set_mode((W, H), flags=pygame.SCALED | pygame.RESIZABLE)  # Flags are for resizing the window (but does not work well)

clock = pygame.time.Clock()
running = True
camera = Camera(W, H)
fps = 2
dt = 5 / fps

# Initialize the world_state array, of size (W,H,3) of RGB values at each position.
world_state = np.zeros((W, H, 3), dtype=np.uint8)
# =============================================================================


updating = True
recording = False
launch_video = True

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        # Event loop. Here we deal with all the interactivity
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if (event.key == pygame.K_p):
                # Stop updating automaton (Freeze time)
                updating = not updating
            if (event.key == pygame.K_r):
                # Toggle recording
                recording = not recording
            if (event.key == pygame.K_SPACE):
                auto.step()

        # Handle the event loop for the camera
        camera.handle_event(event)

    if (updating):
        # Step the automaton if we are updating
        auto.step()

    auto.draw()  # Always draw the automaton
    # Retrieve the world_state from automaton, np.array (W,H,3)
    world_state = auto.worldmap

    # Make the viewable surface.
    surface = pygame.surfarray.make_surface(world_state)

    # For recording
    if (recording):
        if (launch_video):
            # Might bug if FFV1 is not installed
            # Replace *'FFV1' by *'mp4v' and video extension by .mp4 if it doesn't work
            launch_video = False
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            vid_loc = './automaton.mkv'
            video_out = cv2.VideoWriter(vid_loc, fourcc, 30.0, (W, H))

        frame_bgr = cv2.cvtColor(auto.worldmap, cv2.COLOR_RGB2BGR)
        video_out.write(frame_bgr)
        pygame.draw.circle(surface, (255, 0, 0), (W - 10, H - 10), 2)  # Draw the'recording' red dot

    # Clear the screen
    # screen.fill((0, 0, 0))

    # Draw the scaled surface on the window (zoomed)
    # Understanding how the camera works is not important
    zoomed_surface = camera.apply(surface)

    # Blit (draw) the surface on the screen, at (0,0) coordinates
    screen.blit(zoomed_surface, (0, 0))

    # 'flips' the display to show it on the screen
    pygame.display.flip()

    clock.tick(fps)  # limits FPS


pygame.quit()
if (not launch_video):  # if video is launched
    video_out.release()