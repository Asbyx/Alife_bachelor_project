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


class GOLAuto(Automaton):
    def __init__(self, size, init_state=None):
        """
            GOL on GPU

            @param size: (W,H)
            @param init_state: (torch.BoolTensor) initial state of the world, if None, random
        """
        super().__init__(size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        self.world = torch.rand((size[0], size[1]), device=self.device) > 0.5 if init_state is None else init_state.to(self.device)

    def step(self):
        neigh = torch.zeros(self.size, dtype=torch.uint8).to(self.device)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == j == 0: continue
                neigh += self.world.roll((i, j), dims=(0, 1))
        # apply the rules
        self.world = (neigh == 3) | (self.world & (neigh == 2))

    def draw(self):
        self._worldmap = self.world.unsqueeze(2).repeat(1, 1, 3).float().cpu().numpy()