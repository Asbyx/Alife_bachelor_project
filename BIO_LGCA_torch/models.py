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
    todo update description
    Le but est de faire que: quand 2 cellules se rencontrent, elle commence à tenter de faire des nouvelles paires qui sont des copies d'elles-même, comme l'ARN.
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

    optimization / pbs to fix:
        - One could simply do a dictionnary that contains the the mapping function
        - The comm channels should be reset by default
    """
    # signal codes
    SIGNAL_OK = 1
    SIGNAL_MOVE = 2
    SIGNAL_FLIP = 3
    SIGNAL_GRABED = 4
    SIGNAL_HAS_GRABED = [6, 7]  # 6 means that the grabbed is in direction [0, 1], 7 means in [2, 3]
    SIGNAL_PAIR_RESERVATION = 8
    SIGNAL_RESERVATION = 9
    SIGNAL_PAIR_DISBAND = 90

    # signals that encodes several information
    SIGNAL_SEED = 100  # only needs 2 extra encoding channel (in fact i'm dumb, it only needs 1 encoding channel, for the DNA. The information for the direction is already given in the index of the channel that gets the seed)
    SIGNAL_GRABING = 50  # only needs 1
    SIGNAL_TRAVELLING_SEED = 1000  # needs 3 channels

    # states
    STATE_FREE = 1
    STATE_GRABBER = 2
    STATE_TRAVELLING = 3

    # channels
    COMM_CHANNELS = range(4)
    STATE_CHANNEL = 4
    DIR_CHANNEL = 5
    DNA_CHANNEL = 6  # Can be 0,1,2, or 3; to represent A,T,G,C
    MEMORY_CHANNEL = 7  # If the lattice has to remember a previous information
    CLOCK_CHANNEL = 8

    def interaction_function(self, world):
        def fct(channels):
            # Interactions
            if channels[Reproducing_Pairs.STATE_CHANNEL] == 0:  # air
                # if there is a seed in one communication channel, resulting in a new free lattice
                if (channels[Reproducing_Pairs.COMM_CHANNELS] // Reproducing_Pairs.SIGNAL_TRAVELLING_SEED == 1).any().item():
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_TRAVELLING
                    seed_index = index_of(channels, Reproducing_Pairs.SIGNAL_TRAVELLING_SEED, False)
                    channels[Reproducing_Pairs.DIR_CHANNEL] = seed_index
                    channels[Reproducing_Pairs.DNA_CHANNEL] = channels[seed_index]
                    channels[Reproducing_Pairs.COMM_CHANNELS] = 0

                elif (channels[Reproducing_Pairs.COMM_CHANNELS] // Reproducing_Pairs.SIGNAL_SEED == 1).any():
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_FREE
                    channels[Reproducing_Pairs.DIR_CHANNEL] = channels[channels // Reproducing_Pairs.SIGNAL_SEED == 1][0] % (Reproducing_Pairs.SIGNAL_SEED/10)
                    channels[Reproducing_Pairs.DNA_CHANNEL] = (channels[channels // Reproducing_Pairs.SIGNAL_SEED == 1][0]-Reproducing_Pairs.SIGNAL_SEED) // (Reproducing_Pairs.SIGNAL_SEED/10)  # ugly
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
                # absorbtion of the reservation signals
                channels[(
                    ((channels == Reproducing_Pairs.SIGNAL_PAIR_RESERVATION) | (channels == Reproducing_Pairs.SIGNAL_RESERVATION)) &
                    (torch.arange(len(channels)).to(channels.device) != ((channels[Reproducing_Pairs.DIR_CHANNEL]+2)%4))
                )] = 0

                if Reproducing_Pairs.SIGNAL_MOVE in channels[int(((channels[Reproducing_Pairs.DIR_CHANNEL] + 2) % 4).item())]:
                    direction = channels[Reproducing_Pairs.DIR_CHANNEL].clone().item()
                    dna = channels[Reproducing_Pairs.DNA_CHANNEL].clone().item()
                    channels[:] = 0
                    channels[direction] = Reproducing_Pairs.SIGNAL_SEED + dna*Reproducing_Pairs.SIGNAL_SEED/10 + direction

                elif (channels[Reproducing_Pairs.COMM_CHANNELS] // Reproducing_Pairs.SIGNAL_GRABING == 1).any() and not channels[Reproducing_Pairs.MEMORY_CHANNEL]:
                    dir_grabbed = (index_of(channels[Reproducing_Pairs.COMM_CHANNELS], Reproducing_Pairs.SIGNAL_GRABING, False) + 2) % 4

                    # checking that it is the correct dna, else flip if necessary
                    if (((channels[Reproducing_Pairs.DNA_CHANNEL]+2) % 4) + Reproducing_Pairs.SIGNAL_GRABING) not in channels[Reproducing_Pairs.COMM_CHANNELS]:
                        if dir_grabbed == channels[Reproducing_Pairs.DIR_CHANNEL]:  # flip
                            channels[Reproducing_Pairs.DIR_CHANNEL] = (channels[Reproducing_Pairs.DIR_CHANNEL] + 2) % 4
                        channels[Reproducing_Pairs.MEMORY_CHANNEL] = 1
                        return fct(channels)

                    # You have been grabbed !
                    # set all comm channels to FLIP, except the one to dir_grabbed which must be GRABED and if a travelling pair tries to make a reservation
                    channels[Reproducing_Pairs.COMM_CHANNELS] = torch.where(channels[Reproducing_Pairs.COMM_CHANNELS] == Reproducing_Pairs.SIGNAL_PAIR_RESERVATION, Reproducing_Pairs.SIGNAL_PAIR_DISBAND, Reproducing_Pairs.SIGNAL_FLIP).to(torch.int16)
                    channels[dir_grabbed] = Reproducing_Pairs.SIGNAL_GRABED

                elif (channels[Reproducing_Pairs.COMM_CHANNELS] // Reproducing_Pairs.SIGNAL_TRAVELLING_SEED == 1).any():
                    seed = channels[channels // Reproducing_Pairs.SIGNAL_TRAVELLING_SEED == 1][0]
                    direction = index_of(channels, seed)
                    channels[:] = 0
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_TRAVELLING
                    channels[Reproducing_Pairs.DIR_CHANNEL] = direction
                    channels[Reproducing_Pairs.DNA_CHANNEL] = seed

                elif Reproducing_Pairs.SIGNAL_RESERVATION in channels[int(((channels[Reproducing_Pairs.DIR_CHANNEL] + 2) % 4).item())]:
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_GRABBER
                    channels[Reproducing_Pairs.CLOCK_CHANNEL] = 5 * self.size[0]
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
                dir_grabed = index_of(channels[Reproducing_Pairs.COMM_CHANNELS], Reproducing_Pairs.SIGNAL_GRABED)
                pair_has_grabed = channels[Reproducing_Pairs.COMM_CHANNELS][channels[Reproducing_Pairs.DIR_CHANNEL]] if channels[Reproducing_Pairs.COMM_CHANNELS][channels[Reproducing_Pairs.DIR_CHANNEL]] in Reproducing_Pairs.SIGNAL_HAS_GRABED else 0

                # handle the clock
                channels[Reproducing_Pairs.CLOCK_CHANNEL] -= 1
                if channels[Reproducing_Pairs.STATE_CHANNEL] > 0 and channels[Reproducing_Pairs.CLOCK_CHANNEL] < 0 or channels[channels[Reproducing_Pairs.DIR_CHANNEL]] == Reproducing_Pairs.SIGNAL_PAIR_DISBAND:
                    channels[Reproducing_Pairs.COMM_CHANNELS] = 0
                    channels[channels[Reproducing_Pairs.DIR_CHANNEL]] = Reproducing_Pairs.SIGNAL_PAIR_DISBAND
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_FREE
                    channels[Reproducing_Pairs.DIR_CHANNEL] = (channels[Reproducing_Pairs.DIR_CHANNEL] + 2) % 4
                    return channels


                # we set the message by default to be "FLIP", except for the pair
                channels[Reproducing_Pairs.COMM_CHANNELS] = Reproducing_Pairs.SIGNAL_FLIP
                channels[channels[Reproducing_Pairs.DIR_CHANNEL]] = 0

                # This is the counter for the recovery, after the release of a child
                if channels[Reproducing_Pairs.STATE_CHANNEL] < -1:
                    channels[Reproducing_Pairs.STATE_CHANNEL] += 1
                    return channels
                elif channels[Reproducing_Pairs.STATE_CHANNEL] == -1:
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_GRABBER
                    channels[Reproducing_Pairs.CLOCK_CHANNEL] = 5*self.size[0]
                    return channels

                if dir_grabed != -1 and pair_has_grabed != 0 and channels[Reproducing_Pairs.MEMORY_CHANNEL]:
                    self.reproductions_number += 1
                    if self.reproductions_number % 2 == 0: print(f"Number of reproductions: {self.reproductions_number/2}")
                    # releasing the child !
                    number_steps = 5
                    # In a seed: 1000 + 100*pair_dir + 10*DNA + #steps left
                    channels[dir_grabed] = Reproducing_Pairs.SIGNAL_TRAVELLING_SEED + (Reproducing_Pairs.SIGNAL_TRAVELLING_SEED/10)*channels[Reproducing_Pairs.DIR_CHANNEL] + (Reproducing_Pairs.SIGNAL_TRAVELLING_SEED/100)*((channels[Reproducing_Pairs.DNA_CHANNEL]+2) % 4) + number_steps
                    channels[Reproducing_Pairs.STATE_CHANNEL] = -20

                elif dir_grabed != -1:
                    channels[channels[Reproducing_Pairs.DIR_CHANNEL]] = Reproducing_Pairs.SIGNAL_HAS_GRABED[0 if dir_grabed in [0, 1] else 1]
                    channels[dir_grabed] = Reproducing_Pairs.SIGNAL_GRABING + channels[Reproducing_Pairs.DNA_CHANNEL]
                    channels[Reproducing_Pairs.MEMORY_CHANNEL] = 1

                elif pair_has_grabed:
                    # we need to determine whether we are searching for up or down, now that the pair has a grab
                    dir_to_grab = (1 if pair_has_grabed == Reproducing_Pairs.SIGNAL_HAS_GRABED[0] else 3) - (channels[Reproducing_Pairs.DIR_CHANNEL] % 2)
                    channels[dir_to_grab % 4] = Reproducing_Pairs.SIGNAL_GRABING + channels[Reproducing_Pairs.DNA_CHANNEL]
                else:
                    # we try to grab up and down
                    channels[(channels[Reproducing_Pairs.DIR_CHANNEL] - 1) % 4], channels[(channels[Reproducing_Pairs.DIR_CHANNEL] + 1) % 4] = Reproducing_Pairs.SIGNAL_GRABING + channels[Reproducing_Pairs.DNA_CHANNEL], Reproducing_Pairs.SIGNAL_GRABING + channels[Reproducing_Pairs.DNA_CHANNEL]
                    channels[Reproducing_Pairs.MEMORY_CHANNEL] = 0

            elif channels[Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_TRAVELLING:
                # In the DNA channel, I kept the seed intact. Therefore, it has for value: 1000+ 100*pair_dir + 10*dna + #steps
                channels[Reproducing_Pairs.COMM_CHANNELS] = channels[Reproducing_Pairs.COMM_CHANNELS].roll(2)
                pair_dir, dna, steps = extract_digit(channels[Reproducing_Pairs.DNA_CHANNEL].clone(), 3)
                direction = channels[Reproducing_Pairs.DIR_CHANNEL].clone().item()

                # the pair is disbanded if it receives a grabing signal
                if Reproducing_Pairs.SIGNAL_GRABING in channels[Reproducing_Pairs.COMM_CHANNELS] \
                        or Reproducing_Pairs.SIGNAL_PAIR_DISBAND in channels[Reproducing_Pairs.COMM_CHANNELS]:
                    channels[:] = 0
                    channels[pair_dir] = Reproducing_Pairs.SIGNAL_PAIR_DISBAND
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_FREE
                    channels[Reproducing_Pairs.DIR_CHANNEL] = (pair_dir + 2) % 4
                    channels[Reproducing_Pairs.DNA_CHANNEL] = dna
                    return channels

                # if the pair has no steps left, they switch to the grabber state
                elif steps == 0:
                    channels[:] = 0
                    channels[Reproducing_Pairs.STATE_CHANNEL] = Reproducing_Pairs.STATE_GRABBER
                    channels[Reproducing_Pairs.CLOCK_CHANNEL] = 5*self.size[0]
                    channels[Reproducing_Pairs.DIR_CHANNEL] = pair_dir
                    channels[Reproducing_Pairs.DNA_CHANNEL] = dna
                    return channels

                is_resa_ok = Reproducing_Pairs.SIGNAL_MOVE in channels[Reproducing_Pairs.COMM_CHANNELS]
                is_pair_ok = channels[pair_dir] == Reproducing_Pairs.SIGNAL_OK

                # we move
                if is_pair_ok and is_resa_ok and channels[Reproducing_Pairs.MEMORY_CHANNEL]:
                    seed = channels[Reproducing_Pairs.DNA_CHANNEL] - 1
                    channels[:] = 0
                    channels[direction] = seed
                    channels[pair_dir] = Reproducing_Pairs.SIGNAL_OK
                    return channels

                channels[Reproducing_Pairs.COMM_CHANNELS] = 0
                channels[direction] = Reproducing_Pairs.SIGNAL_PAIR_RESERVATION

                # we inform the pair
                if is_resa_ok:
                    channels[pair_dir] = Reproducing_Pairs.SIGNAL_OK
                    channels[Reproducing_Pairs.MEMORY_CHANNEL] = 1
                else:
                    channels[Reproducing_Pairs.MEMORY_CHANNEL] = 0
            return channels

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                world[x, y] = fct(world[x, y])
        return world

    def init_world(self, W, H, nb_lattices=None):
        if nb_lattices is None: nb_lattices = W*3
        self.size = (W, H)
        self.reproductions_number = 0
        init = torch.zeros((W, H, 9), dtype=torch.int16)

        rand = True
        hori = False
        verti = False
        travel = False

        # Random simulation
        if rand:
            init[torch.randint(0, W, (nb_lattices,)), torch.randint(0, H, (nb_lattices,)), Reproducing_Pairs.STATE_CHANNEL] = 1
            init[:, :, Reproducing_Pairs.DIR_CHANNEL] = torch.where(init[:, :, Reproducing_Pairs.STATE_CHANNEL] == 1, torch.randint(0, 4, self.size), init[:, :, Reproducing_Pairs.DIR_CHANNEL])
            init[:, :, Reproducing_Pairs.DNA_CHANNEL] = torch.where(init[:, :, Reproducing_Pairs.STATE_CHANNEL] == 1, torch.randint(0, 4, self.size), init[:, :, Reproducing_Pairs.DNA_CHANNEL])

        # toy example horizontal
        if hori:
            init[5, 5, Reproducing_Pairs.STATE_CHANNEL], init[5, 5, Reproducing_Pairs.DIR_CHANNEL], init[5, 5, Reproducing_Pairs.DNA_CHANNEL] = 1, 2, 3
            init[10, 5, Reproducing_Pairs.STATE_CHANNEL], init[10, 5, Reproducing_Pairs.DNA_CHANNEL] = 1, 1
            # init[4, 4, Reproducing_Pairs.STATE_CHANNEL], init[4, 4, Reproducing_Pairs.DIR_CHANNEL], init[4, 4, Reproducing_Pairs.DNA_CHANNEL] = 1, 2, 1
            init[11, 4, Reproducing_Pairs.STATE_CHANNEL], init[11, 4, Reproducing_Pairs.DNA_CHANNEL] = 1, 3

        # toy example vertical
        if verti:
            init[5, 5, Reproducing_Pairs.STATE_CHANNEL],  init[5, 5, Reproducing_Pairs.DIR_CHANNEL], init[5, 5, Reproducing_Pairs.DNA_CHANNEL] = 1, 3, 0
            init[5, 10, Reproducing_Pairs.STATE_CHANNEL], init[5, 10, Reproducing_Pairs.DIR_CHANNEL], init[5, 10, Reproducing_Pairs.DNA_CHANNEL] = 1, 1, 1
            init[6, 2, Reproducing_Pairs.STATE_CHANNEL], init[6, 2, Reproducing_Pairs.DIR_CHANNEL], init[6, 2, Reproducing_Pairs.DNA_CHANNEL] = 1, 3, 2
            init[6, 3, Reproducing_Pairs.STATE_CHANNEL], init[6, 3, Reproducing_Pairs.DIR_CHANNEL], init[6, 3, Reproducing_Pairs.DNA_CHANNEL] = 1, 3, 3

        # toy example travelling pair
        if travel:
            init[5, 5, Reproducing_Pairs.STATE_CHANNEL], init[5, 5, Reproducing_Pairs.DIR_CHANNEL] = Reproducing_Pairs.STATE_TRAVELLING, 3040
            init[5, 6, Reproducing_Pairs.STATE_CHANNEL], init[5, 6, Reproducing_Pairs.DIR_CHANNEL] = Reproducing_Pairs.STATE_TRAVELLING, 1040
        return init

    def draw_function(self, world):
        in_move_lattices_mask = (world[:, :, 0] // Reproducing_Pairs.SIGNAL_SEED == 1) | (world[:, :, 1] // Reproducing_Pairs.SIGNAL_SEED == 1) | (world[:, :, 2] // Reproducing_Pairs.SIGNAL_SEED == 1) | (world[:, :, 3] // Reproducing_Pairs.SIGNAL_SEED == 1)
        grabber_lattices_mask = (world[:, :, Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_GRABBER) | (world[:, :, 4] < 0)
        travelling_lattices_mask = (world[:, :, 4] == Reproducing_Pairs.STATE_TRAVELLING)
        recovery_lattices_mask = world[:, :, Reproducing_Pairs.STATE_CHANNEL] < 0

        A_lattices_mask = ((world[:, :, Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_FREE) | (world[:, :, Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_GRABBER)) & (world[:, :, Reproducing_Pairs.DNA_CHANNEL] == 0)
        T_lattices_mask = ((world[:, :, Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_FREE) | (world[:, :, Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_GRABBER)) & (world[:, :, Reproducing_Pairs.DNA_CHANNEL] == 2)
        G_lattices_mask = ((world[:, :, Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_FREE) | (world[:, :, Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_GRABBER)) & (world[:, :, Reproducing_Pairs.DNA_CHANNEL] == 1)
        C_lattices_mask = ((world[:, :, Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_FREE) | (world[:, :, Reproducing_Pairs.STATE_CHANNEL] == Reproducing_Pairs.STATE_GRABBER)) & (world[:, :, Reproducing_Pairs.DNA_CHANNEL] == 3)

        signal = None
        signals_mask = (world[:, :, 0] == signal) | (world[:, :, 1] == signal) | (world[:, :, 2] == signal) | (world[:, :, 3] == signal)
        res = np.asarray([(A_lattices_mask*1.0 + T_lattices_mask*0.8)*(1-grabber_lattices_mask*0.3) + in_move_lattices_mask*0.5 + recovery_lattices_mask*1.0,
                          (G_lattices_mask*1.0 + C_lattices_mask*0.8)*(1-grabber_lattices_mask*0.3) + in_move_lattices_mask*0.5 + recovery_lattices_mask*1.0,
                          travelling_lattices_mask*1.0 + in_move_lattices_mask*0.5 + recovery_lattices_mask*1.0]).transpose((1, 2, 0))
        return res


class Game_Of_Life(Model):
    """
    2 rest channels:
    - state of the cell
    - previous sum

    interaction:
        if previous sum = 0:
            store the sum of comm channels in previous sum channel
            update comm channels s.t. comm = sum of orthogonal comm channels (1 = 3 = 0+2; 0 = 2 = 1+3)
        else:
            compute previous sum + (sum of comms /2)
            update state according
            previous sum = 0

        current problem: on initialisation, the information for the neighbors isn't transmitted (in automaton, interaction then migration)
    """
    def interaction_function(self, world):
        self.step = not self.step
        if self.step:
            world[:, :, 5] = world[:, :, 0] + world[:, :, 1] + world[:, :, 2] + world[:, :, 3]
            temp = world[:, :, 0] + world[:, :, 2]
            world[:, :, 0] = world[:, :, 2] = world[:, :, 1] + world[:, :, 3]
            world[:, :, 1] = world[:, :, 3] = temp
        else:
            world[:, :, 5] = world[:, :, 5] + ((world[:, :, 0] + world[:, :, 1] + world[:, :, 2] + world[:, :, 3])/2).to(torch.int8)
            world[:, :, 4] = torch.where(((world[:, :, 5] == 3) | (world[:, :, 4] & (world[:, :, 5] == 2))).to(torch.bool), 1, 0)
            world[:, :, 5] = 0
            world[:, :, 0] = world[:, :, 1] = world[:, :, 2] = world[:, :, 3] = world[:, :, 4]
        return world

    def init_world(self, W, H, custom=None):
        self.step = False
        init = torch.zeros((W, H, 6), dtype=torch.int8)
        if custom is None:
            init[5:8, 3, 4] = 1
            init[5:7, 6:8, 4] = 1
        else: init[:, :, 4] = custom

        init[:, :, 0] = init[:, :, 1] = init[:, :, 2] = init[:, :, 3] = torch.where(init[:, :, 4] == 1, 1, 0)
        return init

    def draw_function(self, world):
        # Convert the boolean array to a uint8 NumPy array
        numpy_array = world[:, :, 4].astype(np.uint8)

        # Add an extra dimension for the third channel
        numpy_array = np.expand_dims(numpy_array, axis=-1)

        # Repeat the third channel to make it (w, h, 3)
        res = np.repeat(numpy_array, 3, axis=-1)
        return res


def extract_digit(value, channels_number):
    res = []
    for i in range(channels_number):
        res.insert(0, value % 10)
        value //= 10
    return res

def index_of(tensor, value, is_simple_signal=True):
    """
    Returns index of 1st occurence of value in tensor, -1 if value is not in tensor
    """
    matches = ((tensor == value) if is_simple_signal else (tensor // value == 1)).nonzero()
    if matches.size(0) == 0:  # no matches
        return -1
    else:
        return matches[0].item()
