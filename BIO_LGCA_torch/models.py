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

class Moving_Lattices(Model):
    """
    This model implement a lattice that moves in a straight line until it encounter another lattice in front of him. Then it stops
    rest_channels: [
            {0: air, 1: lattice moving, 2: lattice at rest},
            direction of the moving lattice
            ]

    A movement consist of n step:
        spot reservation (if several lattices try to take the spot at the same time, a random one is chosen)
            Sending of a signal, the signal takes the rest channel and send a move signal to the lattice
        movement
    """
    # signal codes
    SEED = 10
    STOP = 3
    MOVE = 2
    RESERVATION = 9

    def interaction_function(self, world):
        def fct(channels):
            # air lattices interactions
            if channels[4] == 0:
                # if there is a seed in one communication channel, resulting in a new moving lattice
                if (channels[:4] >= Moving_Lattices.SEED).any():
                    channels[4] = 1
                    channels[5] = channels[channels >= Moving_Lattices.SEED][0] - Moving_Lattices.SEED
                    channels[:4] = Moving_Lattices.STOP

                elif Moving_Lattices.RESERVATION in channels[:4]:
                    # only one reservation
                    if torch.sum(channels[channels == Moving_Lattices.RESERVATION]) == Moving_Lattices.RESERVATION:
                        channels[:4] = torch.where(channels[:4] == Moving_Lattices.RESERVATION, Moving_Lattices.MOVE,
                                                   Moving_Lattices.STOP).roll(2)
                    # there are more than 1 reservation
                    else:
                        channels[torch.where(channels == Moving_Lattices.RESERVATION)[0][1:]] = channels[:4][
                            channels[:4] != Moving_Lattices.RESERVATION] = Moving_Lattices.STOP
                        channels[channels == Moving_Lattices.RESERVATION] = Moving_Lattices.MOVE
                        channels[:4] = torch.roll(channels[:4], 2)
                # if it not a seed nor a reservation, absorb it
                else:
                    channels[:4] = 0

            # moving lattices interactions
            elif channels[4] == 1:
                if Moving_Lattices.MOVE in channels[int(((channels[5] + 2) % 4).item())]:
                    direction = channels[5].clone().item()
                    if torch.distributions.Bernoulli(0.05).sample(torch.Size([1]))[0]: direction = torch.randint(0, 4, (
                    1,)).item()
                    channels[:] = 0
                    channels[direction] = Moving_Lattices.SEED + direction
                elif Moving_Lattices.STOP in channels[
                    int(((channels[5] + 2) % 4).item())] or Moving_Lattices.RESERVATION in channels[
                    int(((channels[5] + 2) % 4).item())]:
                    channels[4] = 2
                else:
                    channels[channels[5].clone().item()] = Moving_Lattices.RESERVATION
            else:
                # resting lattice
                channels[:4] = torch.where(channels[:4] == Moving_Lattices.RESERVATION, Moving_Lattices.STOP, 0).roll(2)
            return channels

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                world[x, y] = fct(world[x, y])
        return world

    def init_world(self, W, H, nb_lattices=None):
        if nb_lattices is None: nb_lattices = W*2
        self.size = (W, H)
        init = torch.zeros((W, H, 6), dtype=torch.int8)
        init[torch.randint(0, W, (nb_lattices,)), torch.randint(0, H, (nb_lattices,)), 4] = 1
        init[:, :, 5] = torch.where(init[:, :, 4] == 1, torch.randint(0, 4, self.size), init[:, :, 1])
        return init

    def draw_function(self, world):
        moving_lattices_mask = (world[:, :, 4] == 1) | (world[:, :, 0] >= 10) | (world[:, :, 1] >= 10) | (world[:, :, 2] >= 10) | (world[:, :, 3] >= 10)
        resting_lattices_mask = world[:, :, 4] == 2
        res = np.asarray([moving_lattices_mask*1.0, moving_lattices_mask*1.0 - resting_lattices_mask*0.5, moving_lattices_mask*1.0]).transpose((1, 2, 0))
        return res
