# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 19:34:50 2017

@author: Matt
"""
from scipy.sparse import lil_matrix
import numpy as np
from scipy.optimize import linprog
from os import listdir, getcwd, system


class problem:

    def __init__(self, filename):
        self.name = None
        self.rows = {}
        self.columns = {}
        self.rhsides = {}
        self.bound_dict = {}
        self.obj = None

        with open(filename, 'r') as f:
            mps = list(f)

        processors = {"NAME": self.set_name,
                      "ROWS": self.add_row,
                      "COLUMNS": self.add_column,
                      "RHS": self.add_rhs,
                      "BOUNDS": self.add_bound,
                      "OBJECT": self.do_nothing,
                      "RANGES": self.raise_exception,
                      "ENDATA": self.do_nothing
                      }
        process = None

        for line in mps:
            pieces = line.split()
            if (len(pieces) == 0 or line.startswith("*")
                or line.startswith("OBJECT")):
                    continue # in at least one file,
                             # line is OBJECT BOUND

            if pieces[0] == "NAME":
                self.set_name(pieces)
            elif len(pieces) == 1:                  # section header
                process = processors[pieces[0]]
            else:                                   # within a section
                process(pieces)

        if len(self.rhsides) == 0: self.rhsides = {"RHS1":{}}

        # assign index numbers to each row and column/rhside
        self.row_indices = {val: key for key, val in enumerate(self.rows)}
        self.col_indices = {val: key for key, val in enumerate(self.columns)}
        self.rhs_indices = {val: key for key, val in enumerate(self.rhsides)}
        self.bound_indices = {val: key for key, val in enumerate(self.bound_dict)}

        # create matrices for lhs and rhs of obj and all constraints (together)
        self.lhs = lil_matrix((len(self.rows), len(self.columns)))
        self.rhs = lil_matrix((len(self.rows), len(self.rhsides)))

        # populate rhs/lhs with data
        self.populate_matrix(self.columns, self.lhs, self.row_indices, self.col_indices)
        self.populate_matrix(self.rhsides, self.rhs, self.row_indices, self.rhs_indices)

        # create list of lists of bounds. Why lists? Why not arrays?
        # Who knows? Ask whoever wrote scipy.optimize.linprog method = "simplex"
        self.bounds = [[[0, None] for i in range(len(self.columns))]
                       for j in range(len(self.bound_dict))]

        # populate list of bounds
        self.populate_bounds()

        # indices of different types of rows (objective, equality constraint,
        # etc...) within lhs
        self.ns = {'N': [], 'L': [], 'G': [], 'E': []}
        for row_name, row_type in self.rows.items():
            self.ns[row_type].append(self.row_indices[row_name])

        # negate greater than (lb) constraints stack with less than (ub) constraints
        self.lhs[self.ns['G']] *= -1
        self.rhs[self.ns['G']] *= -1
        self.ns['L'] += self.ns['G']

        # convert everything to arrays
        self.c =    np.array(self.lhs[self.ns['N']].todense()).flatten()
        self.A_ub = np.array(self.lhs[self.ns['L']].todense())
        self.b_ub = np.array(self.rhs[self.ns['L']].todense()).flatten()
        self.A_eq = np.array(self.lhs[self.ns['E']].todense())
        self.b_eq = np.array(self.rhs[self.ns['E']].todense()).flatten()
        # self.bounds needs no conversion

    def do_nothing(self, l):
        pass

    def raise_exception(self, l):
        raise Exception("Ranges not supported")

    def set_name(self, l):
        self.name = l[1]

    def add_row(self, l):
        self.rows[l[1]] = l[0]

    def add_rhs(self, l):
        # if even number of elements, then no RHS name is present. Make one up.
        if len(l) % 2 == 0:
            l.insert(0, "RHS1")
        self.add_column(l, self.rhsides)

    def add_column(self, l, data=None):
        # each line consists of a column name followed by
        # one (or two) row/value pair(s).
        # form a dictionary where the key is the column name
        # and the value is a list of row/value pairs
        if data is None:
            data = self.columns
        col_name = l[0]
        if col_name not in data:
            data[col_name] = []
        data[col_name].append((l[1], l[2]))
        if len(l) > 3:  # two entries in the row
            data[col_name].append((l[3], l[4]))

    def add_bound(self, l):
        # if this is a fixed bound constraint but there are not enough values,
        # name of the bound series is missing. Add one.
        if l[0] not in ("FR", "MI", "PL") and len(l) == 3:
            l.insert(1, "BOUNDS1")
        bound_name = l[1]
        if bound_name not in self.bound_dict:
            self.bound_dict[bound_name] = []
        if len(l) < 4: # if this is not a fixed bound constraint, pad with value
            l.append(None)
        else:
            l[3] = float(l[3])
        self.bound_dict[bound_name].append((l[2], l[0], l[3]))

    def populate_bounds(self):
        for bound_name, bound_values in self.bound_dict.items():
            for bound in bound_values:
                col_name, direction, value = bound
                i = self.bound_indices[bound_name]
                j = self.col_indices[col_name]
                if direction == "LO":
                    self.bounds[i][j][0] = value
                elif direction == "UP":
                    self.bounds[i][j][1] = value
                elif direction == "FX":
                    self.bounds[i][j][0] = self.bounds[i][j][1] = value
                elif direction == "FR":
                    self.bounds[i][j][0] = self.bounds[i][j][1] = None
                elif direction == "MI":
                    self.bounds[i][j][0] = None
                elif direction == "PL":
                    self.bounds[i][j][1] = None
                else:
                    raise Exception("Only continuous variables supported.")
        if len(self.bounds) == 1:
            self.bounds = self.bounds[0]

    def populate_matrix(self, data, matrix, i_indices, j_indices):
        # for each column name in the dictionary there is a list of
        # row/val pairs; insert in the correct place in the matrix
        for col_name, row_values in data.items():
            for row_name, row_value in row_values:
                i = i_indices[row_name]
                j = j_indices[col_name]
                matrix[i, j] = row_value

    def save(self, filename):
        np.savez_compressed(filename, c=self.c,
                            A_ub=self.A_ub, b_ub=self.b_ub,
                            A_eq=self.A_eq, b_eq=self.b_eq,
                            bounds=self.bounds, obj=float(self.obj))

    def get(self):
        return self.c, self.A_ub, self.b_ub, self.A_eq, self.b_eq, self.bounds


def save_all():
    files = listdir(getcwd())

    for file in files:
        if not file[-4:] == ".mps":
            continue
        name = file[:-4]
        p = problem(name + ".mps")
        p.obj = 0
        p.save(name)


def load(filename):
    data = np.load(filename, allow_pickle=True)
    return (data["c"], data["A_ub"], data["b_ub"], data["A_eq"],
            data["b_eq"], data["bounds"], data["obj"])


def uncompress_all():
    files = listdir(getcwd())

    for file in files:
        if not file[-4:] == ".txt":
            continue
        name = file[:-4]
        cmd = r"emps.exe {0}.txt >> {0}.mps".format(name)
        system(cmd)


#import datetime
#files = listdir(getcwd())
#for file in files:
#    if not file[-4:] == ".npz": # or file.startswith("gosh") or file.startswith("green"):
#        continue
#    print(file)
#    currentDT = datetime.datetime.now()
#    print (str(currentDT))
#    print(file)
#    c, A_ub, b_ub, A_eq, b_eq, bounds, obj = load(file)
#    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method="revised simplex")
#    print(res.status)
#    if not res.status == 2:
#        print("INCORRECT:" + file)
problems = ['bgdbg1', 'bgprtr', 'box1', 'chemcom', 'cplex2',
            'ex72a', 'ex73a', 'forest6', 'galenet', 'itest2',
            'itest6', 'klein1', 'refinery', 'woodinfe']
for prob in problems:
    c, A_ub, b_ub, A_eq, b_eq, bounds, obj = load(prob+".npz")
    t0 = time.perf_counter()
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method="revised simplex")
    t1 = time.perf_counter()
    print(prob, res.nit, res.status)
# method="revised simplex"
#prob_name = "itest2"
##filename = prob_name + ".mps"
##p = problem(filename)
##p.obj = np.array([0])
##c, A_ub, b_ub, A_eq, b_eq, bounds = p.get()
#filename = prob_name + ".npz"
##p.save(filename)
#c, A_ub, b_ub, A_eq, b_eq, bounds, obj = load(filename)
#res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
#print(res)
