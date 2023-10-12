import numpy as np
import torch
import time

class Automaton:
    """
        Class that internalizes the rules and evolution of 
        the cellular automaton at hand. It has a step function
        that makes one timestep of the evolution. By convention,
        and to keep in sync with pygame, the world tensor has shape
        (W,H,3). It contains float values between 0 and 1, which
        are (automatically) mapped to 0 255 when returning output, 
        and describes how the world is 'seen' by an observer.

        Parameters :
        size : 2-uple (W,H)
            Shape of the CA world
        
    """

    def __init__(self, size):
        self.w, self.h = size
        self.size = size

        # This self._worldmap should be changed in the draw function.
        # It should contains floats from 0 to 1 of RGB values.
        self._worldmap = np.zeros(shape=(self.w, self.h, 3))

    def step(self):
        # Should you ABC abstract classes but oh well.
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')

    def draw(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.draw')

    @property
    def worldmap(self):
        # We convert the float [0.,1.] worldmap to uint [[0,255]]
        return (255 * self._worldmap).astype(dtype=np.uint8)


class BioLgcaSquaredAuto(Automaton):
    def __init__(self, size, init_world, interaction_function, draw_function, device=None):
        """
            BIO LGCA on GPU, with a square grid.

            @param size: (W,H) for the drawing function
            @param init_world: (torch.IntTensor: WidthxHeighx(R+4)) initial state of the world, R = size of the rest channel
            @param interaction_function: torch.IntTensor -> torch.IntTensor. Must only use native torch function for better performances
        """
        super().__init__(size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")

        self.interaction = interaction_function
        self.draw_function = draw_function
        self.world = init_world.to(self.device)

    def transport(self):
        self.world[:, :, 0] = self.world[:, :, 0].roll(-1, dims=0)
        self.world[:, :, 1] = self.world[:, :, 1].roll(-1, dims=1)
        self.world[:, :, 2] = self.world[:, :, 2].roll(1, dims=0)
        self.world[:, :, 3] = self.world[:, :, 3].roll(1, dims=1)

    def step(self):
        self.world = self.interaction(self.world)
        self.transport()

    def draw(self):
        self._worldmap = self.draw_function(self.world.cpu().numpy())
        return
