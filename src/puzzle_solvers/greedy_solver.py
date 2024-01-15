import numpy as np

class GreedySolver:
    def __init__(self, size_y: int, size_x: int, pair_relations, pair_probabilities):
        self.size_y = size_y
        self.size_x = size_x

        flipped_pair_relations = [(i[1], i[0]) for i in pair_relations]
        self.pair_relations = pair_relations + flipped_pair_relations
        self.pair_relations = [(i, self.pair_relations[i]) for i in range(len(self.pair_relations))]

        flipped_probabilities = np.zeros_like(pair_probabilities)
        flipped_probabilities[0, :] = pair_probabilities[2, :]
        flipped_probabilities[2, :] = pair_probabilities[0, :]
        flipped_probabilities[1, :] = pair_probabilities[3, :]
        flipped_probabilities[3, :] = pair_probabilities[1, :]
        flipped_probabilities[4, :] = pair_probabilities[4, :]
        self.pair_probabilities = np.concatenate([pair_probabilities, flipped_probabilities])

        self.placed_parts = []
        self.board = np.ones([size_y, size_x]) * -1
        self.next_slot_candidates = []

        self.parts_to_place = list({relation[1][0] for relation in self.pair_relations}.union({relation[1][1] for relation in self.pair_relations}))

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
        if self.can_shift_left():
            self.board[:, 1:self.size_x] = self.board[:, 0:self.size_x - 1]
            self.board[:, 0] = -1
        else:
            raise RuntimeError('Cannot shift left')


    def solve(self) -> dict:
        # --- Place First part
        part = self.parts_to_place.pop(np.random.randint(len(self.parts_to_place)))
        slot_y, slot_x = (self.size_y // 2, self.size_x // 2)
        self.board[(slot_y, slot_x)] = part
        self.placed_parts.append(part)
        self.update_slot_candidates((slot_y, slot_x))

        # --- for each slot candidate, find the best part candidate its probability
        while len(self.parts_to_place) > 0:
            slots_and_candidates = []
            for i, slot in enumerate(self.next_slot_candidates):
                candidate, prob = self.find_best_candidate_for_slot(slot)
                slots_and_candidates.append([i, candidate, prob])

            # --- Choose the top probability slot and part
            winner = sorted(slots_and_candidates, key=lambda x: x[1])[-1]
            winner_slot = self.next_slot_candidates[winner[0]]
            winner_part = winner[1]

            self.board[winner_slot] = winner_part
            self.placed_parts.append(winner_part)
            self.update_slot_candidates(winner_slot)

            # TODO: Handle out of boards and shifts
        print()
        # TODO: continue



    def find_best_candidate_for_slot(self, slot):
        part_relations = self.get_neighbours_and_relations(slot)

        sum_probabilities = 0

        # Sum probabilities on the slot over all neighbours to find best part candidate
        for (neighbour, relation) in part_relations:
            relevant_relations = [r for r in self.pair_relations if r[1][0] == neighbour]
            relevant_relations = sorted(relevant_relations, key=lambda x: x[1][1])

            relevant_ids = [r[0] for r in relevant_relations]
            relevant_probs = self.pair_probabilities[relevant_ids, relation]
            sum_probabilities = sum_probabilities + (relevant_probs / sum(relevant_probs))

        best_idx = sum_probabilities.argmax()
        best_prob = sum_probabilities.max() / len(part_relations)
        best_candidate = relevant_relations[best_idx][1][1]

        return best_candidate, best_prob






    def free_and_valid(self, slot):
        slot_y, slot_x = slot
        valid = True if 0 <= slot_x < self.size_x and  0 <= slot_y < self.size_y else False
        free = True if valid and (self.board[slot_y, slot_x] == -1) else False

        return free and valid

    def update_slot_candidates(self, slot):
        slot_y, slot_x = slot
        adj_slots = [(slot_y + 1, slot_x), (slot_y, slot_x + 1), (slot_y - 1, slot_x), (slot_y, slot_x - 1)]
        candidates = [slot for slot in adj_slots if self.free_and_valid(slot)]
        self.next_slot_candidates += candidates

    def get_occupied_adj_slots(self, slot):
        slot_y, slot_x = slot
        adj_slots = [(slot_y + 1, slot_x), (slot_y, slot_x + 1), (slot_y - 1, slot_x), (slot_y, slot_x - 1)]
        occupied = [s for s in adj_slots if self.board[s] >= 0]
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
        return True if slot[0] == other[0] + 1 and slot[1] == other[1] else False

    @staticmethod
    def is_below(slot, other):
        return True if slot[0] == other[0] -1 and slot[1] == other[1] else False

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

