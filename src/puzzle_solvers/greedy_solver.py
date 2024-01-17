import numpy as np

"""
Greedy solver by my design, doesnt work well.
steps:
1. place a random piece in the middle

2. For every empty slot adjacent to one or more of the placed pieces on the board:
    a. look at all the free pieces, find the one that is most suitable to that place based on sum normalized probabilities to be near the adjacent pieces
    b. Temp. store the best piece found and its probability
3.  Choose the slot and candidate part that has the highest probability, place it on the board
4. go back to #2

Problem: Produces bad results
Likely reason: The algo is not greedy enough.   takes into account all the adjacent slots for each next part, making an avarage taht produce bad choice
"""

class GreedySolver:
    def __init__(self, size_y: int, size_x: int, pair_relations, pair_probabilities):
        self.size_y = size_y
        self.size_x = size_x

        self.pair_relations = self.create_symmetric_pair_relations(pair_relations)
        self.pair_probabilities = self.create_symmetric_pair_probabilities(pair_probabilities)

        self.placed_parts = []
        self.board = np.ones([size_y, size_x]) * -1
        self.next_slot_candidates = []

        self.slack_x = size_x
        self.slack_y = size_y

        self.parts_to_place = list({relation[1][0] for relation in self.pair_relations}.union({relation[1][1] for relation in self.pair_relations}))

    def create_symmetric_pair_relations(self, pair_relations):
        flipped_pair_relations = [(i[1], i[0]) for i in pair_relations]
        symmetric = pair_relations + flipped_pair_relations
        symmetric = [(i, symmetric[i]) for i in range(len(symmetric))]
        return symmetric

    def create_symmetric_pair_probabilities(self, pair_probabilities):
        flipped_probabilities = np.zeros_like(pair_probabilities)
        flipped_probabilities[:, 0] = pair_probabilities[:, 2]
        flipped_probabilities[:, 2] = pair_probabilities[:, 0]
        flipped_probabilities[:, 1] = pair_probabilities[:, 3]
        flipped_probabilities[:, 3] = pair_probabilities[:, 1]
        flipped_probabilities[:, 4] = pair_probabilities[:, 4]
        symmetric = np.concatenate([pair_probabilities, flipped_probabilities])
        return symmetric

    def update_slack(self):
        self.slack_y = self.board.sum(axis=1)
        self.slack_y = len(np.argwhere(self.slack_y == (-1 * self.size_y)))

        self.slack_x = self.board.sum(axis=0)
        self.slack_x = len(np.argwhere(self.slack_x == (-1 * self.size_x)))

    def can_shift_up(self):
        return True if self.board[0, :].sum() == (-1) * self.size_x else False

    def can_shift_down(self):
        return True if self.board[-1, :].sum() == (-1) * self.size_x else False

    def can_shift_left(self):
        return True if self.board[:, 0].sum() == (-1) * self.size_y else False

    def can_shift_right(self):
        return True if self.board[:, -1].sum() == (-1) * self.size_y else False

    def shift_up(self):
        if self.can_shift_up():
            self.board[0:self.size_y - 1, :] = self.board[1:self.size_y, :]
            self.board[self.size_y - 1, :] = -1
        else:
            raise RuntimeError('Cannot shift up')

    def shift_down(self):
        if self.can_shift_down():
            self.board[1:self.size_y, :] = self.board[0:self.size_y - 1, :]
            self.board[0, :] = -1
        else:
            raise RuntimeError('Cannot shift down')

    def shift_left(self):
        if self.can_shift_left():
            self.board[:, 0:self.size_x - 1] = self.board[:, 1:self.size_x]
            self.board[:, self.size_x - 1] = -1
        else:
            raise RuntimeError('Cannot shift left')

    def shift_right(self):
        if self.can_shift_right():
            self.board[:, 1:self.size_x] = self.board[:, 0:self.size_x - 1]
            self.board[:, 0] = -1
        else:
            raise RuntimeError('Cannot shift right')


    def solve(self) -> dict:
        # --- Place First part
        part = self.parts_to_place.pop(np.random.randint(len(self.parts_to_place)))

        slot_y, slot_x = (self.size_y // 2, self.size_x // 2)
        self.next_slot_candidates = self.next_slot_candidates + [(slot_y, slot_x)]
        self.board[(slot_y, slot_x)] = part
        self.placed_parts.append(part)
        self.update_next_slot_candidates()

        # --- for each slot candidate, find the best part candidate its probability
        while len(self.parts_to_place) > 0:
            slots_and_candidates = []
            for i, slot in enumerate(self.next_slot_candidates):
                candidate, prob = self.find_best_candidate_for_slot(slot)
                slots_and_candidates.append([i, candidate, prob])

            # --- Choose the top probability slot and part
            winner = sorted(slots_and_candidates, key=lambda x: x[-1])[-1]
            winner_slot = self.next_slot_candidates[winner[0]]
            winner_part = winner[1]

            if winner_slot[0] < 0:
                self.shift_down()
                winner_slot = (winner_slot[0] + 1, winner_slot[1])
            elif winner_slot[0] >= self.size_y:
                self.shift_up()
                winner_slot = (winner_slot[0] - 1, winner_slot[1])
            elif winner_slot[1] < 0:
                self.shift_right()
                winner_slot = (winner_slot[0], winner_slot[1] + 1)
            elif winner_slot[1] >= self.size_x:
                self.shift_left()
                winner_slot = (winner_slot[0], winner_slot[1] - 1)

            self.board[winner_slot] = winner_part
            self.placed_parts.append(winner_part)
            self.parts_to_place.remove(winner_part)
            self.update_slack()
            self.update_next_slot_candidates()

        part_locations = {(y, x): int(self.board[y, x]) for y in range(self.size_y) for x in range(self.size_x)}
        reverse_location = {part_locations[key]: key for key in part_locations.keys()}

        return reverse_location


    def find_best_candidate_for_slot(self, slot):
        part_relations = self.get_neighbours_and_relations(slot)

        sum_probabilities = 0

        # Sum probabilities on the slot over all neighbours to find best part candidate
        for (neighbour, relation) in part_relations:
            relevant_relations = [r for r in self.pair_relations if r[1][0] == neighbour]
            relevant_relations = [r for r in relevant_relations if r[1][1] not in self.placed_parts]

            relevant_relations = sorted(relevant_relations, key=lambda x: x[1][1])

            relevant_ids = [r[0] for r in relevant_relations]
            relevant_probs = self.pair_probabilities[relevant_ids, relation]
            sum_probabilities = sum_probabilities + (relevant_probs / np.sqrt(sum(relevant_probs)))

        best_idx = sum_probabilities.argmax()
        best_prob = sum_probabilities.max() / len(part_relations)
        best_candidate = relevant_relations[best_idx][1][1]

        return best_candidate, best_prob

    def free_and_valid(self, slot):
        slot_y, slot_x = slot
        valid = True if 0 - self.slack_x <= slot_x < self.size_x + self.slack_x and 0 - self.slack_y <= slot_y < self.size_y + self.slack_y else False
        inside_current_board = True if 0 <= slot_x < self.size_x and 0 <= slot_y < self.size_y else False
        free = True if valid and ((not inside_current_board) or (self.board[slot_y, slot_x] == -1)) else False
        return free and valid

    def occupied_and_valid(self, slot):
        slot_y, slot_x = slot
        valid = True if 0 - self.slack_x <= slot_x < self.size_x + self.slack_x and 0 - self.slack_y <= slot_y < self.size_y + self.slack_y else False
        inside_current_board = True if 0 <= slot_x < self.size_x and 0 <= slot_y < self.size_y else False
        occupied = False if (not inside_current_board) or (valid and (self.board[slot_y, slot_x] == -1)) else True
        return occupied and valid

    def update_next_slot_candidates(self):
        candidates = []
        for y in range(self.size_y):
            for x in range(self.size_y):
                if self.board[y, x] >= 0:
                    adj_slots = [(y + 1, x), (y, x + 1), (y - 1, x), (y, x - 1)]
                    cands = [i for i in adj_slots if self.free_and_valid(i)]
                    candidates = candidates + cands

        self.next_slot_candidates = list(set(candidates))



    def get_occupied_adj_slots(self, slot):
        slot_y, slot_x = slot
        adj_slots = [(slot_y + 1, slot_x), (slot_y, slot_x + 1), (slot_y - 1, slot_x), (slot_y, slot_x - 1)]
        occupied = [s for s in adj_slots if self.occupied_and_valid(s) and self.board[s] >= 0]
        return occupied

    def get_neighbours_and_relations(self, slot):
        adj_occupied = self.get_occupied_adj_slots(slot)

        part_relations = []

        for adj_slot in adj_occupied:
            part = self.board[adj_slot]
            relation = self.get_slot_relation(adj_slot, slot)
            part_relations.append((part, relation))

        return part_relations

    @staticmethod
    def is_above(slot, other):
        return True if slot[0] + 1 == other[0] and slot[1] == other[1] else False

    @staticmethod
    def is_below(slot, other):
        return True if slot[0] - 1 == other[0] and slot[1] == other[1] else False

    @staticmethod
    def is_left(slot, other):
        return True if slot[0] == other[0] and slot[1] + 1 == other[1] else False

    @staticmethod
    def is_right(slot, other):
        return True if slot[0] == other[0] and slot[1] - 1 == other[1] else False

    def get_slot_relation(self, slot, other):
        if self.is_above(slot, other):
            return 0
        elif self.is_left(slot, other):
            return 1
        elif self.is_below(slot, other):
            return 2
        elif self.is_right(slot, other):
            return 3
        else:
            return 4



            #TODO: store free sides for each placed part
            # iterate on free sides of place parts:
            #   Choose the empty slot with the top suiting probability
            #   Place there the most suiting part
            #   Update free slots (both sides of the axis!)
            #   If no pore slots (and parts) - move to next stage of solving









    print()

        # TODO:Yaniv: continue from here:
        #  Rebuild image from blocks using probabilities
        #  show on screen original vs. reconstructed, maybe save some samples
        #  next step - run classification model on image and get label
        #  Create stats / plots on success

