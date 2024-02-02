import random

import numpy as np
from tqdm import tqdm

"""
Greedy solver by my design, loosly based on the following paper:
"A fully automated greedy square jigsaw puzzle solver", by Dolev Pomeranz, Michal Shemesh, Ohad Ben-Shahar from BGU.
Consists of three steps:

Placer steps:
1. place a random piece in the middle
2. For every empty slot adjacent to one or more of the placed pieces on the board:
    a. look at all the free pieces, find the one that is most suitable to that place based on sum normalized probabilities to be near the adjacent pieces
    b. Temp. store the best piece found and its probability
3.  Choose the slot and candidate part that has the highest probability, place it on the board
4. go back to #2

Clusterer steps:
1. start from a random part, add it too a new cluster
2. iteratively do a graph walk on adjacent pars, for each one:
    a. if there is a global agreement that it is the best fit - add to cluster
    b. if not - do not add it, and keep walking on the previous parts' adjacents
    c. if there are no more adjacent "perfect" parts for the cluster - start a new cluster with a new part
    d. if all the parts are clustered - stop
    
Shifter steps: (Based on the cluster output)
1. start a new board, place the biggest cluster on it.
2. For each unplaced part:
    a. try to place the part on all feasible locations (No overlap and within board) and gather overall board probability for each position
    b. keep the best position and board probability, and move to the next part
3. Chose th best part and its best location based on probability, place it on the board permanently
4. Given the new board state - resume from step 2 till no parts are left
"""

UNASSIGNED = -1000
ASSIGNED = -2000
class GreedySolver:
    def __init__(self, size_y: int, size_x: int, pair_relations, pair_probabilities, use_shifter=False, min_iterations: int=2, max_iterations: int=10, stop_at_cluster_size=None):
        """

        :param size_y: num parts in jigsaw puzzle on y
        :param size_x: num parts in jigsaw puzzle on x
        :param pair_relations: tuples of pairs, ordered in the same way as pair_probabilities
        :param pair_probabilities: tensor of (num_pairs x 5) - the probabilities of part1 to be in relation to part2: above, left, below, right or not adjacent
        :param use_shifter: if True, use shifter in board placement.  much slower!  if False - will use several iterations of placer, good enough and faster
        :param max_iterations: stop criteria if max. num iterations is reached
        :param stop_at_cluster_size: stop criteria if minimum cluster size is reached
        """
        self.size_y = size_y
        self.size_x = size_x

        self.use_shifter = use_shifter

        self.pair_relations = self._create_symmetric_pair_relations(pair_relations)
        self.pair_probabilities = self._create_symmetric_pair_probabilities(pair_probabilities)

        self.placed_parts = []
        self.board = np.ones([size_y, size_x]) * UNASSIGNED
        self.next_slot_candidates = []

        self.slack_x = size_x
        self.slack_y = size_y

        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.stop_at_cluster_size = stop_at_cluster_size

        self.parts_to_place: list = None

        self.solve_steps_log = []

    def solve(self) -> dict:
        if self.use_shifter:
            best_board = self.solve_with_shifter()

        else:
            best_board = self.solve_with_placer()

        reverse_location = self._get_part_locations_from_board(best_board)
        return reverse_location, self.solve_steps_log

    def solve_with_placer(self):
        i = 0
        biggest_cluster = 0
        stop = False
        best_board = None

        while not stop:
            # --- Place parts
            if i == 0:
                self._place_all_parts()
            else:
                prev_board = self.board.copy()
                self.board = np.ones_like(self.board) * UNASSIGNED
                self.board[self.cluster_board == big_cluster_idx] = prev_board[
                    self.cluster_board == big_cluster_idx]
                self._place_all_parts(start_from_scratch=False)

            # --- Cluster board
            self._cluster_board_parts()
            cluster_sizes = [len(self.clusters[i]) for i in range(len(self.clusters))]
            big_cluster_idx = np.array(cluster_sizes).argmax()

            if max(cluster_sizes) > biggest_cluster:
                best_board = self.board.copy()
                biggest_cluster = max(cluster_sizes)

            print(f'{i}: Cluster size: {cluster_sizes[big_cluster_idx]}')

            step_log = {
                'clusters_board': self.cluster_board.copy(),
                'reverse_permutation': self._get_part_locations_from_board(self.board)
            }
            self.solve_steps_log.append(step_log)

            # --- Stop criteria
            if self.stop_at_cluster_size and biggest_cluster >= self.stop_at_cluster_size and i >= self.min_iterations - 1:
                break
            if i >= self.max_iterations:
                break
            i += 1

        return best_board

    def solve_with_shifter(self):
        i = 0
        biggest_cluster = 0
        stop = False
        best_board = None

        while not stop:
            # --- Place parts
            if i == 0:
                self._place_all_parts()

            else:
                prev_board = self.board.copy()
                self._shift_clusters()

            # --- Cluster board
            self._cluster_board_parts()
            cluster_sizes = [len(self.clusters[i]) for i in range(len(self.clusters))]
            big_cluster_idx = np.array(cluster_sizes).argmax()

            if max(cluster_sizes) > biggest_cluster:
                best_board = self.board.copy()
                biggest_cluster = max(cluster_sizes)

            print(f'{i}: Cluster size: {cluster_sizes[big_cluster_idx]}')

            step_log = {
                'clusters_board': self.cluster_board.copy(),
                'reverse_permutation': self._get_part_locations_from_board(self.board)
            }
            self.solve_steps_log.append(step_log)

            # --- Stop criteria
            if self.stop_at_cluster_size and biggest_cluster >= self.stop_at_cluster_size and i >= self.min_iterations - 1:
                break
            if i >= self.max_iterations:
                break
            i += 1

        return best_board


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

    def _board_slacks(self, board):
        slack_y = board.sum(axis=1)
        slack_y = len(np.argwhere(slack_y == (UNASSIGNED * self.size_y)))

        slack_x = board.sum(axis=0)
        slack_x = len(np.argwhere(slack_x == (UNASSIGNED * self.size_x)))

        return slack_y, slack_x

    def _shift_origin_and_update_slacks(self, board):
        while self._can_shift_up(board):
            self._shift_up(board)

        while self._can_shift_left(board):
            self._shift_left(board)

        slack_y, slack_x = self._board_slacks(board)
        return slack_y, slack_x

    def _can_shift_up(self, board, by: int=1):
        size_y, size_x = board.shape
        return True if board.sum(axis=1)[by-1]== (UNASSIGNED) * size_x else False

    def _can_shift_down(self, board, by=1):
        size_y, size_x = board.shape
        return True if board.sum(axis=1)[-by] == (UNASSIGNED) * size_x else False

    def _can_shift_left(self, board, by=1):
        size_y, size_x = board.shape
        return True if board.sum(axis=0)[by-1] == (UNASSIGNED) * size_y else False

    def _can_shift_right(self, board, by=1):
        size_y, size_x = board.shape
        return True if board.sum(axis=0)[-by] == (UNASSIGNED) * size_y else False

    def _shift_up(self, board: np.ndarray, by=1):
        if self._can_shift_up(board, by):
            size_y, size_x = board.shape
            board[0:size_y - by, :] = board[by:size_y, :]
            board[size_y - by, :] = UNASSIGNED
        else:
            raise RuntimeError('Cannot shift up')

    def _shift_down(self, board: np.ndarray, by=1):
        if self._can_shift_down(board, by):
            size_y, size_x = board.shape
            board[by:size_y, :] = board[0:size_y - by, :]
            board[:by, :] = UNASSIGNED
        else:
            raise RuntimeError('Cannot shift down')

    def _shift_left(self, board: np.ndarray, by=1):
        size_y, size_x = board.shape
        if self._can_shift_left(board, by):
            board[:, 0:size_x - by] = board[:, by:size_x]
            board[:, size_x - by] = UNASSIGNED
        else:
            raise RuntimeError('Cannot shift left')

    def _shift_right(self, board: np.ndarray, by=1):
        size_y, size_x = board.shape
        if self._can_shift_right(board, by):
            board[:, by:size_x] = board[:, 0:size_x - by]
            board[:, :by] = UNASSIGNED
        else:
            raise RuntimeError('Cannot shift right')

    def _get_part_locations_from_board(self, board):
        """
        returns a dict of {board slot:part id} fro board
        :return:
        """
        part_locations = {(y, x): int(board[y, x]) for y in range(self.size_y) for x in range(self.size_x)}
        reverse_location = {part_locations[key]: key for key in part_locations.keys()}
        return reverse_location

    def _place_all_parts(self, start_from_scratch=True):
        """
        Placer module for initial placement of parts on board, greedy
        :return:
        """
        if start_from_scratch:
            # --- Place First part
            self.parts_to_place = list({relation[1][0] for relation in self.pair_relations}.union(
                {relation[1][1] for relation in self.pair_relations}))

            part = random.choice(self.parts_to_place)
            slot_y, slot_x = (0, 0)
            self.board[(slot_y, slot_x)] = part
            self.placed_parts.append(part)
            self.parts_to_place.remove(part)

        else:
            all_parts = list({relation[1][0] for relation in self.pair_relations}.union(
                {relation[1][1] for relation in self.pair_relations}))

            self.placed_parts = list(np.unique(self.board)[1:].astype(int))
            self.parts_to_place = [part for part in all_parts if part not in self.placed_parts]

        # --- for each slot candidate, find the best part candidate its probability
        while len(self.parts_to_place) > 0:
            self.slack_y, self.slack_x = self._shift_origin_and_update_slacks(self.board)
            self._update_next_slot_candidates()

            slots_and_candidates = []
            for i, slot in enumerate(self.next_slot_candidates):
                candidate, prob = self._find_best_candidate_for_slot(slot, ignored_parts_list=self.placed_parts)
                slots_and_candidates.append([i, candidate, prob])

            # --- Choose the top probability slot and part
            winner = sorted(slots_and_candidates, key=lambda x: x[-1])[-1]
            winner_slot = self.next_slot_candidates[winner[0]]
            winner_part = winner[1]

            if winner_slot[0] < 0:
                self._shift_down(self.board)
                winner_slot = (winner_slot[0] + 1, winner_slot[1])
            elif winner_slot[0] >= self.size_y:
                self._shift_up(self.board)
                winner_slot = (winner_slot[0] - 1, winner_slot[1])
            elif winner_slot[1] < 0:
                self._shift_right(self.board)
                winner_slot = (winner_slot[0], winner_slot[1] + 1)
            elif winner_slot[1] >= self.size_x:
                self._shift_left(self.board)
                winner_slot = (winner_slot[0], winner_slot[1] - 1)

            self.board[winner_slot] = winner_part
            self.placed_parts.append(winner_part)
            self.parts_to_place.remove(winner_part)



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

            # sum_probabilities = sum_probabilities + (relevant_probs / np.sqrt(sum(relevant_probs)))     #TODO: consider
            sum_probabilities = sum_probabilities + relevant_probs

        best_idx = sum_probabilities.argmax()

        # best_prob = sum_probabilities.max() / np.sqrt(len(part_relations))      #TODO: consider
        best_prob = sum_probabilities[best_idx]

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
        free = True if valid and ((not inside_current_board) or (self.board[slot] == UNASSIGNED)) else False
        return free and valid

    def _is_inside_board_with_slack(self, slot):
        slot_y, slot_x = slot
        # return True if 0 - self.slack_x <= slot_x < self.size_x + self.slack_x and 0 - self.slack_y <= slot_y < self.size_y + self.slack_y else False
        return True if 0 - self.slack_x <= slot_x < self.size_x and 0 - self.slack_y <= slot_y < self.size_y else False

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
        occupied = False if (not inside_current_board) or (valid and (self.board[slot] == UNASSIGNED)) else True
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


    def _shift_clusters(self):
        # represent clusters as np arrays starting at (0,0) with (UNASSIGNED)s for empty spaces
        spatial_clusters = []

        for cluster in self.clusters:
            cluster_slots = np.array(cluster)

            cluster_start_y = cluster_slots[:, 0].min()
            cluster_start_x = cluster_slots[:, 1].min()
            cluster_slots[:, 0] -= cluster_start_y
            cluster_slots[:, 1] -= cluster_start_x
            cluster_size_y = cluster_slots[:, 0].max() + 1
            cluster_size_x = cluster_slots[:, 1].max() + 1

            spatial_cluster = np.ones([cluster_size_y, cluster_size_x]) * (UNASSIGNED)
            spatial_cluster[cluster_slots[:, 0], cluster_slots[:, 1]] = ASSIGNED
            spatial_cluster[spatial_cluster != UNASSIGNED] = self.board[cluster_start_y : cluster_start_y + cluster_size_y, cluster_start_x : cluster_start_x + cluster_size_x][spatial_cluster != UNASSIGNED]

            spatial_clusters.append(spatial_cluster)

        # --- initialize
        previous_board = self.board.copy()
        self.board = np.ones_like(previous_board) * UNASSIGNED

        clusters_to_place = list(range(len(self.clusters)))
        placed_clusters = []
        self.rejected_clusters = []

        with tqdm(total=len(clusters_to_place)) as pbar:
            while len(clusters_to_place) > 0:
                unable_to_place = []

                if len(placed_clusters) == 0:
                    # --- start with biggest cluster
                    cluster_sizes = [len(self.clusters[i]) for i in range(len(self.clusters))]
                    cluster_idx = np.array(cluster_sizes).argmax()

                    # Place first cluster at (0,0)
                    cluster = spatial_clusters[cluster_idx]
                    corner = (0, 0)
                    board_shift = (0, 0)
                    has_cluster_to_place = True

                else:
                    next_cluster_options = []
                    unable_to_place = []
                    has_cluster_to_place = False

                    for cluster_idx in clusters_to_place:
                        self.slack_y, self.slack_x = self._shift_origin_and_update_slacks(self.board)
                        cluster = spatial_clusters[cluster_idx]
                        cluster_position_options = []

                        # --- Slide cluster on board, find feasibale points for corner and best fit value
                        for y in range(-self.slack_y, self.size_y + 1):
                            for x in range(-self.slack_x, self.size_x + 1):
                                corner = (y, x)
                                temp_board = self.board.copy().astype(int)
                                valid, shift, corner, probability = self._check_cluster_placement_validity(temp_board, cluster, corner)

                                if valid:
                                    cluster_position_options.append((corner, shift, probability))

                        if len(cluster_position_options) == 0:
                            print(f'could not place cluster of size {cluster.shape}')
                            unable_to_place.append(cluster_idx)
                            # TODO: need to handle this issue somehow
                        else:
                            best_corner_index = np.array([i[2] for i in cluster_position_options]).argmax()
                            best_corner_prob = cluster_position_options[best_corner_index][2]
                            best_cluster_corner = cluster_position_options[best_corner_index][0]
                            best_cluster_shift = cluster_position_options[best_corner_index][1]
                            feasible_places = len(cluster_position_options)
                            next_cluster_options.append((cluster_idx, best_cluster_corner, best_cluster_shift, best_corner_prob, feasible_places))

                    if len(next_cluster_options) > 0:
                        has_cluster_to_place = True
                        cluster_record_idx = np.array([i[3] for i in next_cluster_options]).argmax()
                        cluster_record = next_cluster_options[cluster_record_idx]
                        cluster_idx = cluster_record[0]
                        corner = cluster_record[1]
                        board_shift = cluster_record[2]
                        cluster = spatial_clusters[cluster_idx]

                if has_cluster_to_place:
                    # --- Place selected cluster on board in selected place
                    shift_y, shift_x = board_shift

                    if shift_x != 0:
                        self._shift_right(self.board, shift_x)
                    if shift_y != 0:
                        self._shift_down(self.board, shift_y)

                    self.board = self._place_cluster_on_board(cluster, corner, self.board)
                    placed_clusters = placed_clusters + [cluster_idx]
                    clusters_to_place.remove(cluster_idx)

                if len(unable_to_place) > 0:
                    for cluster_idx in unable_to_place:
                        clusters_to_place.remove(cluster_idx)
                        self.rejected_clusters.append(cluster_idx)

                pbar.update()

    def _update_with_corner_and_slacks(self, board, corner, shift_y, shift_x):
        board = board.copy()
        corner_y, corner_x = corner
        corner_y -= shift_y
        corner_x -= shift_x
        self._shift_down(board, shift_y)
        self._shift_right(board, shift_x)

        return board, (corner_y, corner_x)


    def _check_cluster_placement_validity(self, board, cluster, corner):
        corner_y, corner_x = corner
        cluster_y, cluster_x = cluster.shape
        rb_corner = (corner_y + cluster_y-1, corner_x + cluster_x-1)
        rb_corner_y, rb_corner_x = rb_corner
        shift_x = shift_y = 0

        valid = True
        corner_shift = False
        rbcorner_shift = False

        if not (self._is_inside_board(corner) and self._is_inside_board(rb_corner)):
            if not (self._is_inside_board_with_slack(corner) and self._is_inside_board_with_slack(rb_corner)):
                valid = False
            else:
                if self._is_inside_board_with_slack(corner) and not self._is_inside_board(corner):
                    temp_shift_x = max(-corner_x, 0)
                    temp_shift_y = max(-corner_y, 0)
                    corner_y = corner_y + temp_shift_y
                    corner_x = corner_x + temp_shift_x
                    rb_corner_y = rb_corner_y + temp_shift_y
                    rb_corner_x = rb_corner_x + temp_shift_x
                    corner = (corner_y, corner_x)
                    rb_corner = (rb_corner_y, rb_corner_x)
                    shift_y = shift_y + temp_shift_y
                    shift_x = shift_x + temp_shift_x

                    corner_shift = True

                if self._is_inside_board_with_slack(rb_corner) and not self._is_inside_board(rb_corner):
                    temp_shift_x = min(self.size_x - rb_corner_x, 0)
                    temp_shift_y = min(self.size_y - rb_corner_y, 0)
                    corner_y = corner_y + temp_shift_y
                    corner_x = corner_x + temp_shift_x
                    rb_corner_y = rb_corner_y + temp_shift_y
                    rb_corner_x = rb_corner_x + temp_shift_x
                    corner = (corner_y, corner_x)
                    rb_corner = (rb_corner_y, rb_corner_x)
                    shift_y = shift_y + temp_shift_y
                    shift_x = shift_x + temp_shift_x

                    rbcorner_shift = True

                if not (corner_shift or rbcorner_shift):
                    valid = False

        shift = (shift_y, shift_x)

        if valid:
            cluster_board = np.ones_like(board) * UNASSIGNED
            cluster_board = self._place_cluster_on_board(cluster, corner, cluster_board)
            cluster_mask = cluster_board != UNASSIGNED

            if shift_y:
                self._shift_down(board, shift_y)
            if shift_x:
                self._shift_right(board, shift_x)

            board_mask = board != UNASSIGNED

            if (cluster_mask & board_mask).sum() > 0:
                valid = False  # overlaps between new cluster and placed ones
            elif not self._validate_single_cluster(cluster_mask | board_mask):
                valid = False  # islands on non-touching clusters
            else:
                valid = True

        if valid:
            board = self._place_cluster_on_board(cluster, corner, board.copy())
            board_probability = self._calc_board_probability(board)
        else:
            board_probability = None

        return valid, shift, corner, board_probability





    def _calc_board_probability(self, board):
        board_occupied = board != UNASSIGNED
        slots_to_items = {(y, x): board[y, x] for y in range(self.size_y) for x in range(self.size_x) if board_occupied[y, x]}

        slot_pairs_on_board_v = [((y, x), (y+1, x)) for y in range(self.size_y-1) for x in range(self.size_x) if board_occupied[y, x] & board_occupied[y+1, x]]
        slot_pairs_on_board_h = [((y, x), (y, x+1)) for y in range(self.size_y) for x in range(self.size_x-1) if board_occupied[y, x] & board_occupied[y, x+1]]

        pairs_on_board_v = [(slots_to_items[pair[0]], slots_to_items[pair[1]]) for pair in slot_pairs_on_board_v]
        pairs_on_board_h = [(slots_to_items[pair[0]], slots_to_items[pair[1]]) for pair in slot_pairs_on_board_h]

        relations_v = [r for r in self.pair_relations if r[1] in pairs_on_board_v]
        relations_h = [r for r in self.pair_relations if r[1] in pairs_on_board_h]

        relation_ids_v = [r[0] for r in relations_v]
        relation_ids_h = [r[0] for r in relations_h]

        probs_v = self.pair_probabilities[relation_ids_v, 0]
        probs_h = self.pair_probabilities[relation_ids_h, 1]

        prob_sum = np.concatenate([probs_v, probs_h]).sum()

        return prob_sum


    def _place_cluster_on_board(self, cluster, corner, board):
        cluster_y, cluster_x = cluster.shape
        board = board.copy()
        board[corner[0]: corner[0] + cluster_y, corner[1]: corner[1] + cluster_x][cluster != UNASSIGNED] = cluster[cluster != UNASSIGNED]
        return board


    def _validate_single_cluster(self, board_mask: np.ndarray):
        clusters = []

        slots_to_cluster = [(y, x) for y in range(self.size_y) for x in range(self.size_x) if board_mask[y,x]]

        cluster_slots = []
        rejected_slots = [(y, x) for y in range(self.size_y) for x in range(self.size_x) if not board_mask[y, x]]

        slot = tuple(np.argwhere(board_mask)[0])
        candidate_slots = [slot]

        # --- Current cluster loop
        while slot is not None:
            if len(cluster_slots) == 0:
                add_to_cluster = True
            else:
                # --- Check if slot should be part of the current cluster.
                add_to_cluster = board_mask[slot]

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

        return True if len(cluster_slots) == board_mask.sum() else False



