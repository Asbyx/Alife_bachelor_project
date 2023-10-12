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


class Aware_Lattices(Model):
    def interaction_function(self, world):
        # Only consider alive cell
        mask = world[:, :, 4] != 0

        # Communication acknowledging
        print(world[:, :, 0:4].shape)
        print(world[:, :, 0:4][mask].shape)
        min_received[mask] = torch.min(world[:, :, 0:4][mask], dim=2).values

        # State updating
        world[:, :, 4][mask] = min_received[mask]+1

        # Sending updated state
        world[:, :, 0:4][mask] = world[:, :, 4][mask]
        return world

    def init_world(self, W, H):
        self.size = (W, H)
        init = torch.zeros((self.size[0], self.size[1], 5), dtype=torch.uint8)
        init[20:30, 20:30, 4] = 1
        return init

    def draw_function(self, world):
        res = np.zeros((self.size[0], self.size[1], 3))
        res = np.asarray([world[:, :, 4]/100, world[:, :, 4]/100, world[:, :, 4]/100]).transpose((1, 2, 0))
        return res