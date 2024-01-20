import random

import numpy as np

"""
Greedy solver by my design, loosly based on the following paper:
A fully automated greedy square jigsaw puzzle solver, by Dolev Pomeranz, Michal Shemesh, Ohad Ben-Shahar from BGU.

Currently only placer module is implemented.

Placer steps:
1. place a random piece in the middle

2. For every empty slot adjacent to one or more of the placed pieces on the board:
    a. look at all the free pieces, find the one that is most suitable to that place based on sum normalized probabilities to be near the adjacent pieces
    b. Temp. store the best piece found and its probability
3.  Choose the slot and candidate part that has the highest probability, place it on the board
4. go back to #2
"""

class GreedySolver:
    def __init__(self, size_y: int, size_x: int, pair_relations, pair_probabilities):
        self.size_y = size_y
        self.size_x = size_x

        self.pair_relations = self._create_symmetric_pair_relations(pair_relations)
        self.pair_probabilities = self._create_symmetric_pair_probabilities(pair_probabilities)

        self.placed_parts = []
        self.board = np.ones([size_y, size_x]) * -1
        self.next_slot_candidates = []

        self.slack_x = size_x
        self.slack_y = size_y

        self.parts_to_place: list = None

    def solve(self) -> dict:
        self._place_all_parts()
        self._cluster_board_parts()
        self._shift_parts()

        reverse_location = self._get_part_locations_from_board()
        return reverse_location

    @classmethod
    def _create_symmetric_pair_relations(cls, pair_relations):
        flipped_pair_relations = [(i[1], i[0]) for i in pair_relations]
        symmetric = pair_relations + flipped_pair_relations
        symmetric = [(i, symmetric[i]) for i in range(len(symmetric))]
        return symmetric

    @classmethod
    def _create_symmetric_pair_probabilities(cls, pair_probabilities):
        flipped_probabilities = np.zeros_like(pair_probabilities)
        flipped_probabilities[:, 0] = pair_probabilities[:, 2]
        flipped_probabilities[:, 2] = pair_probabilities[:, 0]
        flipped_probabilities[:, 1] = pair_probabilities[:, 3]
        flipped_probabilities[:, 3] = pair_probabilities[:, 1]
        flipped_probabilities[:, 4] = pair_probabilities[:, 4]
        symmetric = np.concatenate([pair_probabilities, flipped_probabilities])
        return symmetric

    def _update_slack(self):
        self.slack_y = self.board.sum(axis=1)
        self.slack_y = len(np.argwhere(self.slack_y == (-1 * self.size_y)))

        self.slack_x = self.board.sum(axis=0)
        self.slack_x = len(np.argwhere(self.slack_x == (-1 * self.size_x)))

    def _can_shift_up(self):
        return True if self.board[0, :].sum() == (-1) * self.size_x else False

    def _can_shift_down(self):
        return True if self.board[-1, :].sum() == (-1) * self.size_x else False

    def _can_shift_left(self):
        return True if self.board[:, 0].sum() == (-1) * self.size_y else False

    def _can_shift_right(self):
        return True if self.board[:, -1].sum() == (-1) * self.size_y else False

    def _shift_up(self):
        if self._can_shift_up():
            self.board[0:self.size_y - 1, :] = self.board[1:self.size_y, :]
            self.board[self.size_y - 1, :] = -1
        else:
            raise RuntimeError('Cannot shift up')

    def _shift_down(self):
        if self._can_shift_down():
            self.board[1:self.size_y, :] = self.board[0:self.size_y - 1, :]
            self.board[0, :] = -1
        else:
            raise RuntimeError('Cannot shift down')

    def _shift_left(self):
        if self._can_shift_left():
            self.board[:, 0:self.size_x - 1] = self.board[:, 1:self.size_x]
            self.board[:, self.size_x - 1] = -1
        else:
            raise RuntimeError('Cannot shift left')

    def _shift_right(self):
        if self._can_shift_right():
            self.board[:, 1:self.size_x] = self.board[:, 0:self.size_x - 1]
            self.board[:, 0] = -1
        else:
            raise RuntimeError('Cannot shift right')

    def _get_part_locations_from_board(self):
        """
        returns a dict of {board slot:part id} fro board
        :return:
        """
        part_locations = {(y, x): int(self.board[y, x]) for y in range(self.size_y) for x in range(self.size_x)}
        reverse_location = {part_locations[key]: key for key in part_locations.keys()}
        return reverse_location

    def _place_all_parts(self):
        """
        Placer module for initial placement of parts on board, greedy
        :return:
        """
        # --- Place First part
        self.parts_to_place = list({relation[1][0] for relation in self.pair_relations}.union(
            {relation[1][1] for relation in self.pair_relations}))
        part = self.parts_to_place.pop(np.random.randint(len(self.parts_to_place)))
        slot_y, slot_x = (self.size_y // 2, self.size_x // 2)
        self.next_slot_candidates = self.next_slot_candidates + [(slot_y, slot_x)]
        self.board[(slot_y, slot_x)] = part
        self.placed_parts.append(part)
        self._update_next_slot_candidates()

        # --- for each slot candidate, find the best part candidate its probability
        while len(self.parts_to_place) > 0:
            slots_and_candidates = []
            for i, slot in enumerate(self.next_slot_candidates):
                candidate, prob = self._find_best_candidate_for_slot(slot, ignored_parts_list=self.placed_parts)
                slots_and_candidates.append([i, candidate, prob])

            # --- Choose the top probability slot and part
            winner = sorted(slots_and_candidates, key=lambda x: x[-1])[-1]
            winner_slot = self.next_slot_candidates[winner[0]]
            winner_part = winner[1]

            if winner_slot[0] < 0:
                self._shift_down()
                winner_slot = (winner_slot[0] + 1, winner_slot[1])
            elif winner_slot[0] >= self.size_y:
                self._shift_up()
                winner_slot = (winner_slot[0] - 1, winner_slot[1])
            elif winner_slot[1] < 0:
                self._shift_right()
                winner_slot = (winner_slot[0], winner_slot[1] + 1)
            elif winner_slot[1] >= self.size_x:
                self._shift_left()
                winner_slot = (winner_slot[0], winner_slot[1] - 1)

            self.board[winner_slot] = winner_part
            self.placed_parts.append(winner_part)
            self.parts_to_place.remove(winner_part)
            self._update_slack()
            self._update_next_slot_candidates()

    def _find_best_candidate_for_slot(self, slot, specific_neighbours=None, ignored_parts_list=None):
        """
        finds the best candidate for a given slot on the board based on neighbour pair probabilities
        :param slot:
        :return:
        """
        part_relations = self._get_neighbours_and_relations(slot, specific_neighbours)

        sum_probabilities = 0

        # Sum probabilities on the slot over all neighbours to find best part candidate
        for (neighbour, relation) in part_relations:
            relevant_relations = [r for r in self.pair_relations if r[1][0] == neighbour]

            if ignored_parts_list:
                relevant_relations = [r for r in relevant_relations if r[1][1] not in ignored_parts_list]

            relevant_relations = sorted(relevant_relations, key=lambda x: x[1][1])

            relevant_ids = [r[0] for r in relevant_relations]
            relevant_probs = self.pair_probabilities[relevant_ids, relation]
            sum_probabilities = sum_probabilities + (relevant_probs / np.sqrt(sum(relevant_probs)))

        best_idx = sum_probabilities.argmax()
        best_prob = sum_probabilities.max() / len(part_relations)
        best_candidate = relevant_relations[best_idx][1][1]

        return best_candidate, best_prob

    def _is_free_and_valid(self, slot):
        """
        returns whether a slot is free and is valid for placement (not outside margins etc.)
        :param slot:
        :return:
        """
        valid = self._is_inside_board_with_slack(slot)
        inside_current_board = self._is_inside_board(slot)
        free = True if valid and ((not inside_current_board) or (self.board[slot] == -1)) else False
        return free and valid

    def _is_inside_board_with_slack(self, slot):
        slot_y, slot_x = slot
        return True if 0 - self.slack_x <= slot_x < self.size_x + self.slack_x and 0 - self.slack_y <= slot_y < self.size_y + self.slack_y else False

    def _is_inside_board(self, slot):
        slot_y, slot_x = slot
        return True if 0 <= slot_x < self.size_x and 0 <= slot_y < self.size_y else False

    def _is_occupied_and_valid(self, slot):
        """
        returns whether a slot is occupied and is valid for placement (not outside margins etc.)
        :param slot:
        :return:
        """
        valid = self._is_inside_board_with_slack(slot)
        inside_current_board = self._is_inside_board(slot)
        occupied = False if (not inside_current_board) or (valid and (self.board[slot] == -1)) else True
        return occupied and valid

    def _update_next_slot_candidates(self):
        """
        update the list of slots that are: adjacent to placed slots, open for placement, valid
        :return:
        """
        candidates = []
        for y in range(self.size_y):
            for x in range(self.size_y):
                slot = (y, x)
                if self.board[slot] >= 0:
                    adj_slots = self._adjacent_slots(slot)
                    cands = [i for i in adj_slots if self._is_free_and_valid(i)]
                    candidates = candidates + cands

        self.next_slot_candidates = list(set(candidates))

    @classmethod
    def _adjacent_slots(cls, slot):
        """
        Return adjacent slots to given one.   Can return slots outside board!
        :param slot:
        :return:
        """
        y, x = slot
        adj_slots = [(y + 1, x), (y, x + 1), (y - 1, x), (y, x - 1)]
        return adj_slots

    def _valid_adjacent_slots(self, slot):
        """
        Return adjacent slots to given one, only inside board
        :param slot:
        :return:
        """
        y, x = slot
        adj_slots = [(y + 1, x), (y, x + 1), (y - 1, x), (y, x - 1)]
        adj_slots = [s for s in adj_slots if self._is_inside_board(s)]
        return adj_slots

    def _get_occupied_adj_slots(self, slot):
        """
        returns a list of occupied slots adjacent to given one
        :param slot:
        :return:
        """
        slot_y, slot_x = slot
        adj_slots = [(slot_y + 1, slot_x), (slot_y, slot_x + 1), (slot_y - 1, slot_x), (slot_y, slot_x - 1)]
        occupied = [s for s in adj_slots if self._is_occupied_and_valid(s) and self.board[s] >= 0]
        return occupied

    def _get_neighbours_and_relations(self, slot, specific_neighbours=None):
        """
        returns a list of pair relations with neighbours for a given slot based on board
        :param slot:
        :return:
        """
        adj_occupied = self._get_occupied_adj_slots(slot)
        if specific_neighbours is not None:
            adj_occupied = [s for s in adj_occupied if s in specific_neighbours]

        part_relations = []

        for adj_slot in adj_occupied:
            part = self.board[adj_slot]
            relation = self._get_slot_relation(adj_slot, slot)
            part_relations.append((part, relation))

        return part_relations

    @classmethod
    def _is_above(cls, slot, other):
        return True if slot[0] + 1 == other[0] and slot[1] == other[1] else False

    @classmethod

    def _is_below(cls, slot, other):
        return True if slot[0] - 1 == other[0] and slot[1] == other[1] else False

    @classmethod
    def _is_left(cls, slot, other):
        return True if slot[0] == other[0] and slot[1] + 1 == other[1] else False

    @classmethod
    def _is_right(cls, slot, other):
        return True if slot[0] == other[0] and slot[1] - 1 == other[1] else False

    @classmethod
    def _get_slot_relation(cls, slot, other):
        """
        Returns the relation type of a slot with another slot
        :param slot:
        :param other:
        :return:
        """
        if cls._is_above(slot, other):
            return 0
        elif cls._is_left(slot, other):
            return 1
        elif cls._is_below(slot, other):
            return 2
        elif cls._is_right(slot, other):
            return 3
        else:
            return 4

    def _all_slots_on_board(self):
        return [(y, x) for y in range(self.size_y) for x in range(self.size_x)]

    def _cluster_board_parts(self):
        clusters = []

        slots_to_cluster = self._all_slots_on_board()

        while len(slots_to_cluster) > 0:
            # --- start a new cluster, choose a first part
            cluster_slots = []
            rejected_slots = []

            slot = random.choice(slots_to_cluster)
            candidate_slots = [slot]


            # --- Current cluster loop
            while slot is not None:
                if len(cluster_slots) == 0:
                    add_to_cluster = True
                else:
                    # --- Check if slot should be part of the current cluster.
                    part = self.board[slot]
                    best_candidate, best_prob = self._find_best_candidate_for_slot(slot, specific_neighbours=cluster_slots)
                    add_to_cluster = True if part == best_candidate else False

                if add_to_cluster:
                    cluster_slots.append(slot)
                    slots_to_cluster.remove(slot)

                    # --- add unchecked slot neighbours to cluster candidate list
                    new_candidates = [slot for slot in self._valid_adjacent_slots(slot) if
                                      (slot in slots_to_cluster and slot not in rejected_slots)]
                    candidate_slots = list(set(candidate_slots + new_candidates))

                else:
                    rejected_slots.append(slot)

                # --- Next candidate for cluster if there are candidates, else stop
                candidate_slots.remove(slot)
                slot = random.choice(candidate_slots) if len(candidate_slots) > 0 else None

            clusters.append(cluster_slots)

        self.clusters = clusters

        self.cluster_board = np.zeros_like(self.board)
        for i, slots in enumerate(clusters):
            slots_x = [s[1] for s in slots]
            slots_y = [s[0] for s in slots]
            self.cluster_board[slots_y, slots_x] = i


    def _shift_parts(self):
        new_board = np.ones_like(self.board) * -1
        parts_to_place = list(range(len(self.clusters)))

        # start with biggest part
        part_sizes = [len(self.clusters[i]) for i in range(len(self.clusters))]
        part_idx = np.array(part_sizes).argmax()

        # represent parts as np arrays starting at (0,0)
        parts = []
        for part in self.clusters:
            part = np.array(part)
            part[:, 0] -= part[:, 0].min()
            part[:, 1] -= part[:, 1].min()
            parts.append(part)

        part = parts[part_idx]
        placed_parts = [part_idx]
        parts_to_place.remove(part)


        print()




