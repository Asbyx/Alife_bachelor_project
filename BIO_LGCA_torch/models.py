import torch
import numpy as np

class Model:
    def interaction_function(self, world):
        return NotImplementedError('Please subclass "Model" class and define the interaction_function')

    def draw_function(self, world):
        return NotImplementedError('Please subclass "Model" class and define the draw_function')

    def init_world(self, W, H):
        return NotImplementedError('Please subclass "Model" class and define the init_world')

class Weird_LGCA(Model):
    """
    This model is a test model. There are 1 type of particle:
    if it moves:
        if it encounters a rest particule, it inverses its direction
        if it encounters a particule that has the opposite direction, they annihilate each other and become a rest particle.
    if it rests: nothing.

    Particularity: It is better that W and H are multiple of 3.
    """
    def interaction_function(self, world):
        mask = world[:, :, 4] == 1
        # exchange values of index 0 and 2 or 1 and 3 if mask is True
        world[:, :, 0][mask], world[:, :, 2][mask] = world[:, :, 2][mask], world[:, :, 0][mask]
        world[:, :, 1][mask], world[:, :, 3][mask] = world[:, :, 3][mask], world[:, :, 1][mask]

        # a collision result in a annihilation and a rest particle
        # dim 0 and 2
        mask = (world[:, :, 0] == 1) & (world[:, :, 2] == 1)
        world[:, :, 0][mask] = world[:, :, 2][mask] = 0
        world[:, :, 4][mask] = 1
        # dim 1 and 3
        mask = (world[:, :, 1] == 1) & (world[:, :, 3] == 1)
        world[:, :, 1][mask] = world[:, :, 3][mask] = 0
        world[:, :, 4][mask] = 1
        return world

    def draw_function(self, world):
        res = np.zeros((self.size[0], self.size[1], 3))
        res[0::3, 1::3, :] = np.asarray([world[:, :, 0], world[:, :, 0], world[:, :, 0]]).transpose((1, 2, 0))
        res[1::3, 0::3, :] = np.asarray([world[:, :, 1], world[:, :, 1], world[:, :, 1]]).transpose((1, 2, 0))
        res[2::3, 1::3, :] = np.asarray([world[:, :, 2], world[:, :, 2], world[:, :, 2]]).transpose((1, 2, 0))
        res[1::3, 2::3, :] = np.asarray([world[:, :, 3], world[:, :, 3], world[:, :, 3]]).transpose((1, 2, 0))

        res[1::3, 1::3, :] = np.asarray([world[:, :, 4], world[:, :, 4], world[:, :, 4]]).transpose((1, 2, 0))
        res[1::3, 1::3, 2] = 0
        return res

    def init_world(self, W, H):
        self.size = (W, H)
        init = torch.zeros((W // 3, H // 3, 5), dtype=torch.uint8)
        init[:, 0, 4] = init[0, :, 4] = init[:, -1, 4] = init[-1, :, 4] = 1
        init[np.random.randint(1, W // 3 - 2, W), np.random.randint(1, H // 3 - 2, W), np.random.randint(1, 4, W)] = 1
        return init

class Depth_Aware_Lattices(Model):
    """
    This model implements a depth awareness for the lattices (depth = depth in a aggregation of alive lattices).
    Each step, each cell communicates to its neighbors its known depth. Then each cell depth become the minimum received depth and add 1.
    Finaly all dead cells remain dead and send nothing around them.
    """
    def interaction_function(self, world):
        # Identify dead cells
        mask = world[:, :, 4] == 0

        # State updating
        world[:, :, 4] = torch.min(world[:, :, 0:4], dim=2).values + 1
        # neutral channel     =       min of communication channels                  + 1

        # Killing cells that were dead but have been updated
        world[:, :, 4][mask] = 0

        # Sending updated state
        world[:, :, 0:4] = torch.stack([world[:, :, 4], world[:, :, 4], world[:, :, 4], world[:, :, 4]], dim=-1)-1
        world = torch.where(world < 0, 0, world)
        return world

    def init_world(self, W, H):
        self.target_depth = 7
        self.size = (W, H)
        init = torch.distributions.Bernoulli(0.9).sample(torch.Size([self.size[0], self.size[1], 5]))
        return init

    def draw_function(self, world):
        res = np.asarray([torch.zeros(self.size), world[:, :, 4]/self.target_depth, torch.zeros(self.size)]).transpose((1, 2, 0))
        res[res[:, :, 1] == 0] = (0.2, 0.15, 0)
        return res

class Naive_Seed_Square(Model):
    """
    This model implements the growing of a seed into a simple square of size seed_value*2.
    """
    def interaction_function(self, world):
        # Growing of a new cell
        world[:, :, 4] = torch.max(world, dim=2).values

        # Transmitting state
        world[:, :, :4] = torch.stack([world[:, :, 4], world[:, :, 4], world[:, :, 4], world[:, :, 4]], dim=-1) - 1
        return world

    def init_world(self, W, H):
        self.size = (W, H)
        self.seed_value = 200
        init = torch.zeros((W, H, 5), dtype=torch.int16)
        init[W//2, H//2, :] = self.seed_value
        return init

    def draw_function(self, world):
        res = np.asarray([torch.zeros(self.size), world[:, :, 4] / (self.seed_value+1), torch.zeros(self.size)]).transpose((1, 2, 0))
        return res

