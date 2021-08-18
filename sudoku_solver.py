#!/usr/bin/env python3

import copy
from collections import defaultdict
import math
import itertools
import random
import time


def update_preemptive_sets(preemptive_sets, newp, min_k=2):
    for k, v in newp.items():
        if len(k) >= min_k:
            preemptive_sets[len(k)].append((k, v))
    return preemptive_sets


def get_preemptive_sets(coords, values):
    groups = defaultdict(set)
    for i, (v, c) in enumerate(zip(values, coords)):
        groups[tuple(sorted(list(v)))].add(c)
        for j, (v_, c_) in enumerate(zip(values, coords)):
            if v_.issubset(v):
                groups[tuple(sorted(list(v)))].add(c_)

    # remove non-valid sets
    groups = {k: v for k, v in groups.items() if len(k) == len(v)}

    return groups


def discard_from_preemptive_sets(coords, values):

    # get preemptive set
    ps = get_preemptive_sets(coords, values)

    for val_set, coords_set in ps.items():
        values = [
            v - set(val_set) if c not in coords_set else v
            for c, v in zip(coords, values)
        ]

    return coords, values, ps


def make_choices_from_preempt(preemptive_sets):

    choices = []
    # random shuffle sets
    for k in preemptive_sets.keys():
        random.shuffle(preemptive_sets[k])

    for k in sorted(preemptive_sets.keys()):
        for values, coords in preemptive_sets[k]:
            choices += [
                list(zip(x, coords))
                for x in itertools.permutations(values, len(values))
            ]

    return choices


def make_choices_random(board):
    m = n = int(math.sqrt(len(board)))

    choices = []
    # random shuffle sets
    for n in range(2, m + 1):
        choices_ = [(k, v) for k, v in board.items() if len(v) == n]
        random.shuffle(choices_)
        choices += [[(v_, k)] for k, v in choices_ for v_ in v]

    return choices


def get_closest_valid_node(tree, curr_node):
    if len(tree[curr_node]['choices']) > 0:
        return curr_node
    else:
        return get_closest_valid_node(tree, tree[curr_node]['parent'])


def is_valid(board):

    m = n = int(math.sqrt(len(board)))

    # check each row
    for i in range(m):
        vals = [
            next(iter(board[(i, j)])) for j in range(n)
            if len(board[(i, j)]) == 1
        ]
        if len(vals) > len(set(vals)):
            return False

    # check each column
    for j in range(n):
        vals = [
            next(iter(board[(i, j)])) for i in range(m)
            if len(board[(i, j)]) == 1
        ]
        if len(vals) > len(set(vals)):
            return False

    # check each block
    for i in range(int(math.sqrt(m))):
        for j in range(int(math.sqrt(n))):
            start_i = int(math.sqrt(m)) * i
            end_i = start_i + int(math.sqrt(m))
            start_j = int(math.sqrt(n)) * j
            end_j = start_j + int(math.sqrt(n))
            vals = [
                next(iter(board[(i_, j_)])) for i_ in range(start_i, end_i)
                for j_ in range(start_j, end_j) if len(board[(i_, j_)]) == 1
            ]
            if len(vals) > len(set(vals)):
                return False

    return True


def get_not_dones(board):
    out = dict()
    for k in board:
        if len(board[k]) > 1:
            out[k] = board[k]

    return out


def get_dones(board):
    out = dict()
    for k in board:
        if len(board[k]) == 1:
            out[k] = board[k]

    return out


def get_valid_nodes(tree):
    pass


class Solver:
    def __init__(self, board):
        self.board = board
        m = len(self.board)

        # initialize cells
        self.board_dict = dict()
        for i in range(len(board)):
            for j in range(len(board[i])):
                self.board_dict[(i, j)] = set(
                    board[i][j]) if board[i][j] != '.' else set(
                        [str(i) for i in range(1, m + 1)])

        self.board = self.board_dict

        # initialize solution tree
        self.tree = dict()

    def __str__(self):
        m = int(math.sqrt(len(self.board)))
        block_width = int(math.sqrt(m))

        # make separator
        sep = ''.join(block_width * ['+' + ''.join(block_width * ['---'])])
        sep += '+\n'
        str_ = ''
        str_ += sep
        for i in range(m):
            for j in range(m):

                if (j % block_width == 0):
                    str_ += '|'
                val = self.board[(i, j)]

                str_ += ' ' + next(iter(val)) + ' ' if len(val) == 1 else ' . '
            str_ += '|\n'
            if i > 0 and ((i + 1) % block_width == 0):
                str_ += sep

        return str_

    def run(self):
        curr_n_done = len(get_dones(self.board))
        curr_node = 0
        m = int(math.sqrt(len(self.board)))

        while True:
            if not is_valid(self.board) and len(self.tree) == 0:
                return -1
            elif is_valid(self.board) and curr_n_done == m**2:
                print('solved!')
                return 1
            elif not is_valid(self.board):
                # last tested choice is invalid, remove it
                self.tree[curr_node]['choices'] = self.tree[curr_node][
                    'choices'][1:]

                # get closest "valid" node from curr_node upwards
                # this can be curr_node if there are choices to test
                # that remain
                curr_node = get_closest_valid_node(self.tree, curr_node)
                self.board = self.tree[curr_node]['board']
                curr_n_done = len(get_dones(self.board))

                # do cell assignment according to next
                # possible choice
                # do assignment
                choice = self.tree[curr_node]['choices'][0]
                for c in choice:
                    self.board[c[1]] = set([c[0]])

                new_n_done = len(get_dones(self.board))
            else:
                self.propagate()
                preemptive_sets = self.preempt()
                self.singleton()

                new_n_done = len(get_dones(self.board))

                if not is_valid(self.board):
                    continue
                elif (new_n_done == curr_n_done):
                    # add node in tree
                    if len(preemptive_sets) > 0:
                        choices = make_choices_from_preempt(preemptive_sets)
                    else:
                        choices = make_choices_random(self.board)

                    self.tree[len(self.tree) + 1] = {
                        'board': copy.deepcopy(self.board),
                        'choices': choices,
                        'parent':
                        len(self.tree) if len(self.tree) > 0 else None
                    }
                    curr_node = len(self.tree)

                    # do assignment
                    choice = self.tree[curr_node]['choices'][0]
                    for c in choice:
                        self.board[c[1]] = set([c[0]])

                    new_n_done = len(get_dones(self.board))

                curr_n_done = new_n_done

    def propagate(self):
        m = int(math.sqrt(len(self.board)))
        bw = int(math.sqrt(m))

        dones = {
            c: v
            for c, v in self.board.items() if len(self.board[c]) == 1
        }

        for c, v in dones.items():
            # process row
            row = {(c[0], j): self.board[(c[0], j)]
                   for j in range(m) if j != c[1]}
            row = {c_: v_ - v for c_, v_ in row.items()}
            self.board.update(row)

            # process column
            col = {(i, c[1]): self.board[(i, c[1])]
                   for i in range(m) if i != c[0]}
            col = {c_: v_ - v for c_, v_ in col.items()}
            self.board.update(col)

            # process block
            blk = {(i, j): self.board[(i, j)]
                   for i in range(bw * (c[0] // bw),
                                  bw * (c[0] // bw) + bw)
                   for j in range(bw * (c[1] // bw),
                                  bw * (c[1] // bw) + bw)
                   if (i, j) != (c[0], c[1])}
            blk = {c_: v_ - v for c_, v_ in blk.items()}
            self.board.update(blk)

    def preempt(self):
        m = int(math.sqrt(len(self.board)))
        preemptive_sets = defaultdict(list)

        # find preemptive sets and remove from others on each row
        for i in range(m):
            coords = [(i, j) for j in range(m)]
            values = [self.board[(i, j)] for j in range(m)]
            coords, values, ps = discard_from_preemptive_sets(coords, values)
            preemptive_sets = update_preemptive_sets(preemptive_sets, ps)
            self.board.update({c: v for c, v in zip(coords, values)})

        # find preemptive sets and remove from others on each column
        for j in range(m):
            coords = [(i, j) for i in range(m)]
            values = [self.board[(i, j)] for i in range(m)]
            coords, values, ps = discard_from_preemptive_sets(coords, values)
            preemptive_sets = update_preemptive_sets(preemptive_sets, ps)
            self.board.update({c: v for c, v in zip(coords, values)})

        # find preemptive sets and remove from others on each block
        block_width = int(math.sqrt(m))
        for i in range(block_width):
            for j in range(block_width):
                start_k = i * block_width
                end_k = start_k + block_width
                start_l = j * block_width
                end_l = start_l + block_width
                coords = [(k, l) for k in range(start_k, end_k)
                          for l in range(start_l, end_l)]
                values = [
                    self.board[(k, l)] for k in range(start_k, end_k)
                    for l in range(start_l, end_l)
                ]
                coords, values, ps = discard_from_preemptive_sets(
                    coords, values)
                preemptive_sets = update_preemptive_sets(preemptive_sets, ps)
                self.board.update({c: v for c, v in zip(coords, values)})

        return preemptive_sets

    def singleton(self):
        m = int(math.sqrt(len(self.board)))

        # find singletons and remove from others on each row
        for i in range(m):
            this = {(i, j): self.board[(i, j)] for j in range(m)}
            this = {
                k: v - set.union(*[v_ for k_, v_ in this.items() if k_ != k])
                for k, v in this.items()
            }

            this = {k: v for k, v in this.items() if len(v) == 1}
            self.board.update(this)

        # find singletons and remove from others on each col
        for j in range(m):
            this = {(i, j): self.board[(i, j)] for i in range(m)}
            this = {
                k: v - set.union(*[v_ for k_, v_ in this.items() if k_ != k])
                for k, v in this.items()
            }

            this = {k: v for k, v in this.items() if len(v) == 1}
            self.board.update(this)

        # find singletons and remove from others on each block
        bw = int(math.sqrt(m))
        for i in range(bw):
            for j in range(bw):
                # process block
                this = {(i_, j_): self.board[(i_, j_)]
                        for i_ in range(bw * (i // bw),
                                        bw * (i // bw) + bw)
                        for j_ in range(bw * (j // bw),
                                        bw * (j // bw) + bw)}
                this = {
                    k:
                    v - set.union(*[v_ for k_, v_ in this.items() if k_ != k])
                    for k, v in this.items()
                }

                this = {k: v for k, v in this.items() if len(v) == 1}
                self.board.update(this)


def test(board):
    print(''.join(40 * ['=']))
    print('input board: ')
    solver = Solver(board)
    print(solver)
    start = time.time()
    solver.run()
    end = time.time()
    print('found solution in ', end - start, ' seconds')
    print(solver)
    print(''.join(40 * ['=']))


if __name__ == "__main__":
    diabolical_board = [['.', '9', '.', '7', '.', '.', '8', '6', '.'],
                        ['.', '3', '1', '.', '.', '5', '.', '2', '.'],
                        ['8', '.', '6', '.', '.', '.', '.', '.', '.'],
                        ['.', '.', '7', '.', '5', '.', '.', '.', '6'],
                        ['.', '.', '.', '3', '.', '7', '.', '.', '.'],
                        ['5', '.', '.', '.', '1', '.', '7', '.', '.'],
                        ['.', '.', '.', '.', '.', '.', '1', '.', '9'],
                        ['.', '2', '.', '6', '.', '.', '3', '5', '.'],
                        ['.', '5', '4', '.', '.', '8', '.', '7', '.']]
    test(diabolical_board)

    preempt_board = [['.', '3', '9', '5', '.', '.', '.', '.', '.'],
                     ['.', '.', '.', '8', '.', '.', '.', '7', '.'],
                     ['.', '.', '.', '.', '1', '.', '9', '.', '4'],
                     ['1', '.', '.', '4', '.', '.', '.', '.', '3'],
                     ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
                     ['.', '.', '7', '.', '.', '.', '8', '6', '.'],
                     ['.', '.', '6', '7', '.', '8', '2', '.', '.'],
                     ['.', '1', '.', '.', '9', '.', '.', '.', '5'],
                     ['.', '.', '.', '.', '.', '1', '.', '.', '8']]
    test(preempt_board)

    expert_board = [['.', '.', '.', '.', '3', '.', '.', '.', '.'],
                    ['8', '.', '.', '.', '.', '9', '.', '5', '7'],
                    ['.', '.', '.', '.', '.', '.', '.', '.', '3'],
                    ['.', '.', '.', '9', '.', '.', '.', '.', '.'],
                    ['.', '6', '3', '1', '.', '.', '4', '2', '.'],
                    ['.', '2', '.', '.', '7', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '1', '8', '.', '.', '.'],
                    ['.', '5', '.', '.', '.', '.', '8', '.', '.'],
                    ['2', '.', '4', '.', '.', '.', '.', '9', '6']]
    test(expert_board)

    hardest_board = [['8', '.', '.', '.', '.', '.', '.', '.', '.'],
                     ['.', '.', '3', '6', '.', '.', '.', '.', '.'],
                     ['.', '7', '.', '.', '9', '.', '2', '.', '.'],
                     ['.', '5', '.', '.', '.', '7', '.', '.', '.'],
                     ['.', '.', '.', '.', '4', '5', '7', '.', '.'],
                     ['.', '.', '.', '1', '.', '.', '.', '3', '.'],
                     ['.', '.', '1', '.', '.', '.', '.', '6', '8'],
                     ['.', '.', '8', '5', '.', '.', '.', '1', '.'],
                     ['.', '9', '.', '.', '.', '.', '4', '.', '.']]
    test(hardest_board)
