import random

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
                        channels[:4] = torch.where(channels[:4] == Moving_Lattices.RESERVATION, Moving_Lattices.MOVE, Moving_Lattices.STOP).roll(2)
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


class Reproducing_Pairs(Model):
    """
    Le but est de faire que: quand 2 cellules se rencontrent, elle commence à tenter de faire des nouvelles paires qui sont des copies d'elles-même.
    Pour cela elles grabbent autour d'elles les cellules compatibles (envoie un signal grab qui permet de stopper de force une cellule qui passerait et la force à communiquer avec le grabber)
    Les cellules non-compatibles rebondissent simplement, et les cellules grabbed font tout rebondire.
    Liste des signaux possibles:
        RESERVATION, SEED, MOVE, REBOUNCE, COLLISION (pour un free) = permet au lattices free de bouger et de détecter des collisions
        GRABING + nature visée  (depuis un graber)                  = {nature visée}, doit être absorbé par l'air
        GRABED                  (entre un grabbed et un graber)     = permet d'informer le graber qu'il a grab qqch, 0 ou 1
        HAS GRABBED             (entre 2 graber)                    = {0: pas grab, 1: grab "au dessus", -1: grab "en dessous"}
    Liste des états possibles:
        1: Free -> bouge librement en attendant une collision ou un grab
        2: Graber -> Grab ce qui passe autour de lui, laisse passer ce qui ne l'intéresse pas ou le fait rebondir. Envoie à sa paire un signal qui dit si il a grab une particule, et si oui selon quel axe
        Grabbed -> Fait tout rebondir sur lui
        TBD (quand la reproduction est terminée, qu'est ce qu'il se passe ?)

    Fixme Known bugs:
        - on: right - up - left, the reservation signal passes through the up, and the two free cells therefore become grabbers but are 1 cell appart

    optimization / pbs to fix:
        - One could simply do a dictionnary that contains the the mapping function
        - The comm channels should be reset by default
    """
    # signal codes
    SIGNAL_OK = 1
    SIGNAL_MOVE = 2
    SIGNAL_FLIP = 3
    SIGNAL_GRABED = 4
    SIGNAL_GRABING = 5
    SIGNAL_HAS_GRABED = [6, 7]  # 6 means that the grabbed is in direction [0, 1], 7 means in [2, 3]
    SIGNAL_PAIR_RESERVATION = 8
    SIGNAL_RESERVATION = 9
    SIGNAL_PAIR_DISBAND = 90

    # signals that encodes several information
    SIGNAL_SEED = 10  # only needs 1 extra encoding channel
    SIGNAL_TRAVELLING_SEED = 1000 # needs 2 channels, in which 1 can take values in [0, 99]

    # states
    STATE_FREE = 1
    STATE_GRABBER = 2
    STATE_TRAVELLING = 3

    # channels
    COMM_CHANNELS = range(4)
    STATE_CHANNEL = 4
    DIR_CHANNEL = 5
    CLK_CHANNEL = 6

    def interaction_function(self, world):
        def fct(channels):
            channels[Reproducing_Pairs.CLK_CHANNEL] = not channels[Reproducing_Pairs.CLK_CHANNEL]

            # air lattices interactions
            if channels[Reproducing_Pairs.STATE_CHANNEL] == 0:
                # if there is a seed in one communication channel, resulting in a new free lattice
                if (channels[Reproducing_Pairs.COMM_CHANNELS] // Reproducing_Pairs.SIGNAL_TRAVELLING_SEED >= 1).any().item():
                    DNA = channels[Reproducing_Pairs.COMM_CHANNELS][channels[Reproducing_Pairs.COMM_CHANNELS] // Reproducing_Pairs.SIGNAL_TRAVELLING_SEED >= 1][0] - Reproducing_Pairs.SIGNAL_TRAVELLING_SEED
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_TRAVELLING
                    channels[Reproducing_Pairs.DIR_CHANNEL] = DNA
                    channels[Reproducing_Pairs.COMM_CHANNELS] = 0

                elif (channels[Reproducing_Pairs.COMM_CHANNELS] // Reproducing_Pairs.SIGNAL_SEED == 1).any():
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_FREE
                    channels[Reproducing_Pairs.DIR_CHANNEL] = channels[channels // Reproducing_Pairs.SIGNAL_SEED == 1][0] - Reproducing_Pairs.SIGNAL_SEED
                    channels[Reproducing_Pairs.COMM_CHANNELS] = 0

                elif Reproducing_Pairs.SIGNAL_RESERVATION in channels[Reproducing_Pairs.COMM_CHANNELS] and Reproducing_Pairs.SIGNAL_PAIR_RESERVATION not in channels[Reproducing_Pairs.COMM_CHANNELS]:
                    # only one reservation
                    if torch.sum(channels[channels == Reproducing_Pairs.SIGNAL_RESERVATION]) == Reproducing_Pairs.SIGNAL_RESERVATION:
                        channels[Reproducing_Pairs.COMM_CHANNELS] = torch.where(channels[Reproducing_Pairs.COMM_CHANNELS] == Reproducing_Pairs.SIGNAL_RESERVATION, Reproducing_Pairs.SIGNAL_MOVE, 0).roll(2).to(torch.int16)
                    # there are more than 1 reservation
                    else:
                        channels[torch.where(channels == Reproducing_Pairs.SIGNAL_RESERVATION)[0][1:]] = channels[Reproducing_Pairs.COMM_CHANNELS][channels[Reproducing_Pairs.COMM_CHANNELS] != Reproducing_Pairs.SIGNAL_RESERVATION] = 0
                        channels[channels == Reproducing_Pairs.SIGNAL_RESERVATION] = Reproducing_Pairs.SIGNAL_MOVE
                        channels[Reproducing_Pairs.COMM_CHANNELS] = torch.roll(channels[Reproducing_Pairs.COMM_CHANNELS], 2)

                elif Reproducing_Pairs.SIGNAL_PAIR_RESERVATION in channels[Reproducing_Pairs.COMM_CHANNELS]:
                    if torch.sum(channels[channels == Reproducing_Pairs.SIGNAL_PAIR_RESERVATION]) == Reproducing_Pairs.SIGNAL_PAIR_RESERVATION:
                        channels[Reproducing_Pairs.COMM_CHANNELS] = torch.where(channels[Reproducing_Pairs.COMM_CHANNELS] == Reproducing_Pairs.SIGNAL_PAIR_RESERVATION, Reproducing_Pairs.SIGNAL_MOVE, 0).roll(2).to(torch.int16)
                    # there are more than 1 reservation
                    else:
                        channels[torch.where(channels == Reproducing_Pairs.SIGNAL_PAIR_RESERVATION)[0][torch.randint(0, torch.sum(channels[channels == Reproducing_Pairs.SIGNAL_PAIR_RESERVATION])/Reproducing_Pairs.SIGNAL_PAIR_RESERVATION, (1,)).item()]] = channels[channels != Reproducing_Pairs.SIGNAL_PAIR_RESERVATION] = 0
                        channels[channels == Reproducing_Pairs.SIGNAL_PAIR_RESERVATION] = 0
                        channels[Reproducing_Pairs.COMM_CHANNELS] = torch.roll(channels[Reproducing_Pairs.COMM_CHANNELS], 2)
                # if it not a seed nor a reservation, absorb it
                else:
                    channels[Reproducing_Pairs.COMM_CHANNELS] = 0

            # moving lattices interactions
            elif channels[Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_FREE:
                if Reproducing_Pairs.SIGNAL_MOVE in channels[int(((channels[Reproducing_Pairs.DIR_CHANNEL] + 2) % 4).item())]:
                    direction = channels[Reproducing_Pairs.DIR_CHANNEL].clone().item()
                    if torch.distributions.Bernoulli(0.05).sample(torch.Size([1]))[0]: direction = torch.randint(0, 4, (1,)).item()
                    channels[:] = 0
                    channels[direction] = Reproducing_Pairs.SIGNAL_SEED + direction

                elif Reproducing_Pairs.SIGNAL_GRABING in channels[Reproducing_Pairs.COMM_CHANNELS]:  # You have been grabbed !
                    dir_grabbed = (index_of(channels[Reproducing_Pairs.COMM_CHANNELS], Reproducing_Pairs.SIGNAL_GRABING) + 2) % 4
                    # set all comm channels to FLIP, except the one to dir_grabbed which must be GRABED and if a travelling pair tries to make a reservation
                    channels[Reproducing_Pairs.COMM_CHANNELS] = torch.where(channels[Reproducing_Pairs.COMM_CHANNELS] == Reproducing_Pairs.SIGNAL_PAIR_RESERVATION, Reproducing_Pairs.SIGNAL_PAIR_DISBAND, Reproducing_Pairs.SIGNAL_FLIP).to(torch.int16)
                    channels[dir_grabbed] = Reproducing_Pairs.SIGNAL_GRABED

                elif (channels[Reproducing_Pairs.COMM_CHANNELS] // Reproducing_Pairs.SIGNAL_TRAVELLING_SEED >= 1).any():
                    seed = channels[channels // Reproducing_Pairs.SIGNAL_TRAVELLING_SEED >= 1][0]
                    channels[:] = 0
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_TRAVELLING
                    channels[Reproducing_Pairs.DIR_CHANNEL] = seed

                elif Reproducing_Pairs.SIGNAL_RESERVATION in channels[int(((channels[Reproducing_Pairs.DIR_CHANNEL] + 2) % 4).item())]:
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_GRABBER
                    channels[channels[Reproducing_Pairs.DIR_CHANNEL]] = Reproducing_Pairs.SIGNAL_RESERVATION

                elif Reproducing_Pairs.SIGNAL_FLIP in channels[int(((channels[Reproducing_Pairs.DIR_CHANNEL] + 2) % 4).item())]\
                        or Reproducing_Pairs.SIGNAL_PAIR_RESERVATION in channels[int(((channels[Reproducing_Pairs.DIR_CHANNEL] + 2) % 4).item())]:
                    channels[Reproducing_Pairs.DIR_CHANNEL] = (channels[Reproducing_Pairs.DIR_CHANNEL] + 2) % 4

                else:
                    channels[channels[Reproducing_Pairs.DIR_CHANNEL]] = Reproducing_Pairs.SIGNAL_RESERVATION

            elif channels[Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_GRABBER \
                    or channels[Reproducing_Pairs.STATE_CHANNEL] < 0:
                channels[Reproducing_Pairs.COMM_CHANNELS] = channels[Reproducing_Pairs.COMM_CHANNELS].roll(2)  # allow to have the correct directions right away
                # retrieve the important signals, i.e the signal coming from the pair and the signal of a grabbed particle
                dir_grabed = index_of(channels, Reproducing_Pairs.SIGNAL_GRABED)
                pair_has_grabed = channels[Reproducing_Pairs.COMM_CHANNELS][channels[Reproducing_Pairs.DIR_CHANNEL]] if channels[Reproducing_Pairs.COMM_CHANNELS][channels[Reproducing_Pairs.DIR_CHANNEL]] in Reproducing_Pairs.SIGNAL_HAS_GRABED else 0

                # we set the message by default to be "FLIP", except for the pair
                channels[Reproducing_Pairs.COMM_CHANNELS] = Reproducing_Pairs.SIGNAL_FLIP
                channels[channels[Reproducing_Pairs.DIR_CHANNEL]] = 0

                if channels[Reproducing_Pairs.STATE_CHANNEL] < -1:
                    channels[Reproducing_Pairs.STATE_CHANNEL] += 1
                    return channels
                elif channels[Reproducing_Pairs.STATE_CHANNEL] == -1:
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_GRABBER
                    return channels

                if dir_grabed != -1 and pair_has_grabed != 0 and channels[Reproducing_Pairs.CLK_CHANNEL]:
                    # releasing the child !
                    channels[channels[Reproducing_Pairs.DIR_CHANNEL]] = Reproducing_Pairs.SIGNAL_HAS_GRABED[0 if dir_grabed in [0, 1] else 1]
                    channels[dir_grabed] = Reproducing_Pairs.SIGNAL_TRAVELLING_SEED*(channels[Reproducing_Pairs.DIR_CHANNEL]+1) + 10*5 + dir_grabed
                    channels[Reproducing_Pairs.STATE_CHANNEL] = -20

                elif dir_grabed != -1:
                    channels[channels[Reproducing_Pairs.DIR_CHANNEL]] = Reproducing_Pairs.SIGNAL_HAS_GRABED[0 if dir_grabed in [0, 1] else 1]
                    channels[dir_grabed] = Reproducing_Pairs.SIGNAL_GRABING

                elif pair_has_grabed:
                    # we need to determine whether we are searching for up or down, now that the pair has a grab
                    dir_to_grab = (1 if pair_has_grabed == Reproducing_Pairs.SIGNAL_HAS_GRABED[0] else 3) - (channels[Reproducing_Pairs.DIR_CHANNEL] % 2)
                    channels[dir_to_grab % 4] = Reproducing_Pairs.SIGNAL_GRABING

                else:
                    # we try to grab up and down
                    channels[(channels[Reproducing_Pairs.DIR_CHANNEL] - 1) % 4], channels[(channels[Reproducing_Pairs.DIR_CHANNEL] + 1) % 4] = Reproducing_Pairs.SIGNAL_GRABING, Reproducing_Pairs.SIGNAL_GRABING

            elif channels[Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_TRAVELLING:
                # for travelling pairs, the dir_channel encode the direction of movement, the position of the pair and the number of steps left they have s.t: channel = pairdir*1000 + #steps*10 + dir
                channels[Reproducing_Pairs.COMM_CHANNELS] = channels[Reproducing_Pairs.COMM_CHANNELS].roll(2)
                pair_dir = (channels[Reproducing_Pairs.DIR_CHANNEL] // 1000) - 1
                direction = channels[Reproducing_Pairs.DIR_CHANNEL] % 10

                # the pair is disbanded if it receives a grabing signal
                if Reproducing_Pairs.SIGNAL_GRABING in channels[Reproducing_Pairs.COMM_CHANNELS] \
                        or Reproducing_Pairs.SIGNAL_PAIR_DISBAND in channels[Reproducing_Pairs.COMM_CHANNELS]:
                    channels[:] = 0
                    channels[pair_dir] = Reproducing_Pairs.SIGNAL_PAIR_DISBAND
                    channels[Reproducing_Pairs.DIR_CHANNEL] = (pair_dir + 2) % 4
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_FREE
                    return channels

                # if the pair has no steps left, they switch to the grabber state todo: pas logique de le faire là, plutot jouer avec la seed
                elif (channels[Reproducing_Pairs.DIR_CHANNEL] % 1000) // 10 == 0:
                    channels[:] = 0
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_GRABBER
                    channels[Reproducing_Pairs.DIR_CHANNEL] = pair_dir
                    return channels

                is_resa_ok = Reproducing_Pairs.SIGNAL_MOVE in channels[Reproducing_Pairs.COMM_CHANNELS]
                is_pair_ok = channels[pair_dir] == Reproducing_Pairs.SIGNAL_OK

                # we move
                if is_pair_ok and is_resa_ok and channels[Reproducing_Pairs.CLK_CHANNEL]:  # 10*number of steps + dir of the pair. Therefore, we substract 10 to remove 1 step.
                    seed = Reproducing_Pairs.SIGNAL_TRAVELLING_SEED + channels[Reproducing_Pairs.DIR_CHANNEL] - 10
                    channels[direction] = seed
                    channels[channels != seed] = 0
                    return channels

                channels[Reproducing_Pairs.COMM_CHANNELS] = 0
                channels[direction] = Reproducing_Pairs.SIGNAL_PAIR_RESERVATION

                # we inform the pair
                if is_resa_ok:
                    channels[pair_dir] = Reproducing_Pairs.SIGNAL_OK
            return channels

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                world[x, y] = fct(world[x, y])
        return world

    def init_world(self, W, H, nb_lattices=None):
        if nb_lattices is None: nb_lattices = W*2
        self.size = (W, H)
        init = torch.zeros((W, H, 7), dtype=torch.int16)

        # Random simulation
        init[torch.randint(0, W, (nb_lattices,)), torch.randint(0, H, (nb_lattices,)), Reproducing_Pairs.STATE_CHANNEL] = 1
        init[:, :, Reproducing_Pairs.DIR_CHANNEL] = torch.where(init[:, :, Reproducing_Pairs.STATE_CHANNEL] == 1, torch.randint(0, 4, self.size), init[:, :, 5])

        # toy example horizontal
        # init[5, 5, Reproducing_Pairs.STATE_CHANNEL], init[5, 5, Reproducing_Pairs.DIR_CHANNEL] = 1, 2
        # init[10, 5, Reproducing_Pairs.STATE_CHANNEL] = 1
        # init[11, 4, Reproducing_Pairs.STATE_CHANNEL] = 1
        # init[4, 4, Reproducing_Pairs.STATE_CHANNEL], init[4, 4, Reproducing_Pairs.DIR_CHANNEL] = 1, 2

        # toy example travelling pair
        # init[5, 5, Reproducing_Pairs.STATE_CHANNEL], init[5, 5, Reproducing_Pairs.DIR_CHANNEL] = Reproducing_Pairs.STATE_TRAVELLING, 3040
        # init[5, 6, Reproducing_Pairs.STATE_CHANNEL], init[5, 6, Reproducing_Pairs.DIR_CHANNEL] = Reproducing_Pairs.STATE_TRAVELLING, 1040

        # toy example vertical
        # init[5, 5, Reproducing_Pairs.STATE_CHANNEL],  init[5, 5, Reproducing_Pairs.DIR_CHANNEL] = 1, 3
        # init[5, 10, Reproducing_Pairs.STATE_CHANNEL], init[5, 10, Reproducing_Pairs.DIR_CHANNEL] = 1, 1
        # init[6, 2, Reproducing_Pairs.STATE_CHANNEL], init[6, 2, Reproducing_Pairs.DIR_CHANNEL] = 1, 3
        # init[6, 15, Reproducing_Pairs.STATE_CHANNEL], init[6, 15, Reproducing_Pairs.DIR_CHANNEL] = 1, 1
        return init

    def draw_function(self, world):
        moving_lattices_mask = (world[:, :, 4] == 1)
        in_move_lattices_mask = (world[:, :, 0] >= 10) | (world[:, :, 1] >= 10) | (world[:, :, 2] >= 10) | (world[:, :, 3] >= 10)
        grabber_lattices_mask = (world[:, :, 4] == Reproducing_Pairs.STATE_GRABBER) | (world[:, :, 4] < 0)
        travelling_lattices_mask = (world[:, :, 4] == Reproducing_Pairs.STATE_TRAVELLING)

        signal = Reproducing_Pairs.SIGNAL_TRAVELLING_SEED
        signals_mask = (world[:, :, 0] == signal) | (world[:, :, 1] == signal) | (world[:, :, 2] == signal) | (world[:, :, 3] == signal)
        res = np.asarray([moving_lattices_mask*1.0 + travelling_lattices_mask*0.5 - in_move_lattices_mask*0.2 + signals_mask*1.0,
                          moving_lattices_mask*1.0 - grabber_lattices_mask*0.5 - in_move_lattices_mask*0.2,
                          moving_lattices_mask*1.0 + travelling_lattices_mask*0.5 - in_move_lattices_mask*0.2]).transpose((1, 2, 0))
        return res


def index_of(tensor, value):
    """
    Returns index of 1st occurence of value in tensor, -1 if value is not in tensor
    """
    matches = (tensor == value).nonzero()  # [number_of_matches, tensor_dimension]
    if matches.size(0) == 0:  # no matches
        return -1
    else:
        return matches[0].item()
