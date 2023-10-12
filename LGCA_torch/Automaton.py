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
        self._worldmap = np.random.uniform(size=(self.w, self.h, 3))

    def step(self):
        # Should you ABC abstract classes but oh well.
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')

    def draw(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.draw')

    @property
    def worldmap(self):
        # We convert the float [0.,1.] worldmap to uint [[0,255]]
        return (255 * self._worldmap).astype(dtype=np.uint8)

class LGCAAuto(Automaton):
    def __init__(self, size, init_world, colors=True):
        """
            LGCA on GPU

            @param size: (W,H)
            @param init_world: (torch.BoolTensor, 4xWxH) initial state of the world
            @param colors: (bool) if True, the particles are colored
        """
        super().__init__(size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        self.world = init_world.to(self.device)

        self.colors = colors

    def collision(self):
        collisions = torch.zeros((2, self.w, self.h), dtype=torch.bool, device=self.device)
        for i in (0, 1):
            # we check for opposing directions
            collisions[i] = self.world[i] & self.world[i+2]

            # we check if it is not a 3 or 4 particles collision
            collisions[i] = collisions[i] & ~(self.world[i+1] | self.world[((i+3) % 4)])

        # we apply the collisions
        for i in (0, 1):
            for j in (0, 1, 2, 3):
                self.world[j] ^= collisions[i]

    def transport(self):
        # we do the same for each direction
        for i in (0, 1):
            # we roll the tensor in the direction of the flow
            self.world[i] = self.world[i].roll(-1, dims=i % 2)
            self.world[i+2] = self.world[i+2].roll(1, dims=i % 2)

        # if particles are on the border, they are moved to the other direction
        self.world[0, -1, :] = self.world[2, -1, :]
        self.world[2, -1, :] = torch.zeros(self.world[2, -1, :].shape, dtype=torch.bool, device=self.device)

        self.world[2, 0, :] = self.world[0, 0, :]
        self.world[0, 0, :] = torch.zeros(self.world[0, 0, :].shape, dtype=torch.bool, device=self.device)

        self.world[1, :, -1] = self.world[3, :, -1]
        self.world[3, :, -1] = torch.zeros(self.world[3, :, -1].shape, dtype=torch.bool, device=self.device)

        self.world[3, :, 0] = self.world[1, :, 0]
        self.world[1, :, 0] = torch.zeros(self.world[1, :, 0].shape, dtype=torch.bool, device=self.device)

    def step(self):
        self.collision()
        self.transport()

    def draw(self):
        if self.colors: self._worldmap = (self.world[0:3].cpu().numpy() | self.world[3].cpu().numpy()).transpose((1, 2, 0))
        else:
            pixels = (self.world[0] | self.world[1] | self.world[2] | self.world[3]).cpu().numpy()
            self._worldmap = np.stack((pixels, pixels, pixels), axis=-1)
