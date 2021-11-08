import os
import numpy as np
import re
from warnings import warn
import subprocess
from tempfile import NamedTemporaryFile


# ATSP: Find a HAMILTONIAN CIRCUIT (Tour)  whose global cost is minimum (Asymmetric Travelling Salesman Problem: ATSP)
# https://github.com/coin-or/metslib-examples/tree/master/atsp
# http://www.localsolver.com/documentation/exampletour/tsp.html
# http://or.dei.unibo.it/research_pages/tspsoft.html

# Be careful in choosing this value (high values may cause overflow)

def shortest_directed_hamiltonian_path(
    matrix, start=None, end=None, precision=3, seed=0
):
    ''' Shortest hamiltonian path. '''

    n = matrix.shape[0]
    matrix_ = matrix.copy()
    matrix_ = np.pad(matrix_, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    # Insert dummy node onto matrix
    #
    # matrix[dummy, start] = -INF, matrix[dummy, V \ {start}] = INF
    # matrix[end, dummy] = -INF,   matrix[V \ {end}, dummy] = INF
    dummy = n

    # start node must be fixed?
    if start is not None:
        if start == -1:
            start = n - 1
        assert (start >= 0) and (start < n)
        matrix_[dummy, start] = -np.inf

    # end node must be fixed?
    if end is not None:
        if end == -1:
            end = n - 1
        assert (end >= 0) and (end < n)

        matrix_[end, dummy] = -np.inf

    solution, _ = ConcordeATSPSolver(
        max_precision=precision, seed=seed
    ).solve(matrix_).solution()
    if solution is None:
        return (None, float('nan'))
    
    # Remove repeated element        
    solution = solution[: -1]

    # Removing dummy node from solution
    pos = solution.index(dummy)
    solution = solution[pos + 1:] + solution[: pos]
    cost = matrix[solution[: -1], solution[1 :]].sum()
    return solution, cost


class ConcordeATSPSolver:
    ''' Solver for ATSP using Concorde.'''

    @staticmethod
    def load_tsplib(filename):
        ''' Load a tsplib instance for testing. '''

        lines = open(filename).readlines()
        regex_non_numeric = re.compile(r'[^\d]+')
        n = int(next(regex_non_numeric.sub('', line)
                     for line in lines if line.startswith('DIMENSION')))
        start = next(i for i, line in enumerate(lines) if line.startswith('EDGE_WEIGHT_SECTION'))
        end = next(i for i, line in enumerate(lines) if line.startswith('EOF'))
        matrix = np.array(
            map(float, ' '.join(lines[start + 1:end]).split()), dtype=np.int32).reshape((n, -1))
        # np.fill_diagonal(matrix, 0)

        return matrix

    @staticmethod
    def dump_tsplib(matrix, filename):
        ''' Dump a tsplib instance.

        For detais on tsplib format, check: http://ftp.uni-bayreuth.de/math/statlib/R/CRAN/doc/packages/TSP.pdf
        '''

        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]

        template = '''NAME: {name}
TYPE: TSP
COMMENT: {name}
DIMENSION: {n}
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
{matrix_str}EOF'''

        name = os.path.splitext(os.path.basename(filename))[0]
        n = matrix.shape[0]

        # space delimited string
        matrix_str = ' '
        for row in matrix:
            matrix_str += ' '.join([str(val) for val in row])
            matrix_str += '\n'
        open(filename, 'w').write(template.format(
            **{'name': name, 'n': n, 'matrix_str': matrix_str}))
        

    def __init__(self, max_precision=3, seed=None):
        ''' Class constructor. '''

        self._max_precision = max_precision # ignored at this time
        self._seed = seed
        self._solution = None
        

    def _run_concorde(self, matrix):
        ''' Run Concorde solver for instance named with filename.

        Check https://github.com/mhahsler/TSP/blob/master/R/tsp_concorde.R for some tricks.
        '''

        # Fix negative values
        if matrix.min() < 0:
            matrix -= matrix.min()
        
        # Fix small values
        if matrix.max() < 1:
            matrix += 1 - matrix.max()
            
        
        # Deal with small numbers
#        print matrix
#        print matrix.sum() == 0
#        while np.all(matrix < 1):
#            matrix = 10 * matrix
        
        for i in range(self._max_precision):
            if np.all(np.modf(matrix)[0] == 0):
                break
            matrix = matrix * 10
        
            #if np.mod(((10 ** i) * matrix).astype(np.int64), 10).sum() > 0:
             #   precision = i
            #i += 1
#        i = 0
#        while np.mod(((10 ** i) * matrix).astype(np.int64), 10).sum() == 0:
#            if np.mod(((10 ** i) * matrix).astype(np.int64), 10).sum() > 0:
#                precision = i
#            i += 1
        # Determining infinity value (ckeck overflow possibility) and precision
        #precision = 5
        
        n = matrix.shape[0]
#        print 'info'
        #print precision
#        print matrix
        

#        if n < 10:
#            limit = 2.0 ** 15
#            if max_val > limit:
#                raise ValueError(
#                    'ERROR: Concorde can only handle distances < 2^15 for less than 10 cities.')
#
#            precision_ = int(np.floor(np.log10(limit / max_val)))
##            precision = int(np.floor(np.log10(limit / max_val)))
#            if precision_ < precision:
#                precision = precision_
#                warn('WARNING: Concorde can only handle distances < 2^15 for less than 10 cities.\
#                  Reducing precision to %d' % precision)
#
#        else:
#            limit = 2.0 ** 31 - 1
##            precision = int(np.floor(np.log10(limit / max_val / n)))
#            precision_ = int(np.floor(np.log10(limit / max_val / n)))
#            
#            if precision_ < precision:
#                precision = precision_
#                warn('WARNING: Concorde can only handle distances < 2^31. Reducing precision to %d' % precision)
        # Scaling matrix (handling precision)
        #matrix = (10 ** precision) * matrix
        #print matrix
        # if (matrix - np.floor(matrix)).sum() > 0:
        #    warn('WARNING: Loss of precision.')

        assert matrix.max() < 2 ** 31 - 1

        # Dump matrix in int32 format
        #tsp_filename = '%s.tsp' % NamedTemporaryFile().name
        tsp_filename = '/tmp/tsp.tsp'
        ConcordeATSPSolver.dump_tsplib(matrix.astype(np.int32), tsp_filename)

        # Call Concorde solver
        curr_dir = os.path.abspath('.')
        dir_ = os.path.dirname(tsp_filename)
        os.chdir(dir_)
        
        sol_filename = os.path.join(
            dir_, os.path.splitext(os.path.basename(tsp_filename))[0] + '.sol'
        )
        seed = self._seed
        if seed is not None:
            cmd = ['concorde', '-s', str(seed), '-o', sol_filename, tsp_filename]
        else:
            cmd = ['concorde', '-o', sol_filename, tsp_filename]
        try:
            with open(os.devnull, 'w') as devnull:
                try:
                    output = subprocess.check_output(cmd, stderr=devnull)
                except subprocess.CalledProcessError:
                    os.chdir(curr_dir)
                    return None
                
#            output = subprocess.check_output(cmd)
        except OSError as exc:
            if 'No such file or directory' in str(exc):
                raise Exception('ERROR: Concorde solver not found.')

        #print output        
        os.chdir(curr_dir)
        tour = map(int, open(sol_filename).read().split()[1:])
        return tour

    def _atsp_to_tsp(self, C):
        '''
        Reformulate an asymmetric TSP as a symmetric TSP:
        "Jonker and Volgenant 1983"
        This is possible by doubling the number of nodes. For each city a dummy
        node is added: (a, b, c) => (a, a', b, b', c, c')

        distance = "value"
        distance (for each pair of dummy nodes and pair of nodes is INF)
        distance (for each pair node and its dummy node is -INF)
        ------------------------------------------------------------------------
          |a    |b    |c    |a'   |b'   |c'   |
        a |0    |INF  |INF  |-INF |dBA  |dCA  |
        b |INF  |0    |INF  |dAB  |-INF |dCB  |
        c |INF  |INF  |0    |dAC  |dBC  |-INF |
        a'|-INF |dAB  |dAC  |0    |INF  |INF  |
        b'|dBA  |-INF |dBC  |INF  |0    |INF  |
        c'|dCA  |dCB  |-INF |INF  |INF  |0    |

        @return: new symmetric matrix

        [INF][C.T]
        [C  ][INF]
        '''

        n = C.shape[0]
        n_tilde = 2 * n
        C_tilde = np.empty((n_tilde, n_tilde), dtype=np.float64)
        C_tilde[:, :] = np.inf
        np.fill_diagonal(C_tilde, 0.0)
        C_tilde[n:, :n] = C
        C_tilde[:n, n:] = C.T
        np.fill_diagonal(C_tilde[n:, :n], -np.inf)
        np.fill_diagonal(C_tilde[:n, n:], -np.inf)

        return C_tilde

    def solve(self, matrix):
        ''' Solve ATSP instance. '''

        matrix_ = self._atsp_to_tsp(matrix)
        masked_matrix = np.ma.masked_array(
            matrix_, mask=np.logical_or(matrix_ == np.inf, matrix_ == -np.inf)
        )
        min_val, max_val = masked_matrix.min(), masked_matrix.max()

        # https://rdrr.io/cran/TSP/man/TSPLIB.html
        # Infinity = val +/- 2*range
        pinf = max_val + 2 * (max_val - min_val)
        ninf = min_val - 2 * (max_val - min_val)
        
        # print "pinf", pinf, "ninf", ninf
        matrix_[matrix_ == np.inf] = pinf
        matrix_[matrix_ == -np.inf] = ninf

        # TSP solution
        solution_tsp = self._run_concorde(matrix_)
        if solution_tsp is None:
            self._solution = (None, -1)
        else:    
            # Convert to ATSP solution
            solution = solution_tsp[:: 2] + [solution_tsp[0]]
    
            # TSP - Infrastructure for the Traveling Salesperson Problem (Hahsler and Hornik)
            #
            # "Note that the tour needs to be reversed if the dummy cities appear before and
            # not after the original cities in the solution of the TSP."
            # print solution_tsp[1], n
            n = matrix.shape[0]
            # print solution_tsp
            # print solution
            if solution_tsp[1] != n:
                solution = solution[::-1]
            # print solution_tsp
            # print solution
    
#            print solution_tsp
#            print solution
#            print max(solution)
#            print matrix.shape
#            import sys
#            sys.stdout.flush()
            cost = matrix_[solution[:-1], solution[1:]].sum()
            self._solution = (solution, cost)
        return self

    def solution(self):
        return self._solution


# Testing
if __name__ == '__main__':
    import sys

    path = 'test_instances'

    print 'DOCREC instance'
    # print 'ATSP'
    matrix = np.load(os.path.join(path, 'matrix_float.npy'))
    solver = ConcordeATSPSolver(max_precision=3)
    solver.solve(matrix)
    print '%.2f\n%s' % (solver.solution()[1], ' -> '.join(str(x) for x in solver.solution()[0]))

    print 'Shortest hamiltonian path'
    matrix = np.load(os.path.join(path, 'matrix_float.npy'))
    solution, cost = shortest_directed_hamiltonian_path(matrix, start=8, end=3, seed=None)
    print '%.2f\n%s' % (cost, ' -> '.join(str(x) for x in solution))

    sys.exit()
    print 'TSPLIB instances'

    optimal_costs = {'br17': 39, 'ft53': 6905, 'ft70': 38673, 'ftv33': 1286, 'ftv35': 1473, 'ftv38': 1530, 'ftv44': 1613,
                     'ftv47': 1776, 'ftv55': 1608, 'ftv64': 1839, 'ftv70': 1950, 'ftv90': 1579, 'ftv100': 1788,
                     'ftv110': 1958, 'ftv120': 2166, 'ftv130': 2307, 'ftv140': 2420, 'ftv150': 2611, 'ftv160': 2683,
                     'ftv170': 2755, 'kro124p': 36230, 'p43': 5620, 'rbg323': 1326, 'rbg358': 1163, 'rbg403': 2465,
                     'rbg443': 2720, 'ry48p': 14422}

    filenames = [filename for filename in os.listdir(path) if filename.endswith('.atsp')]

    # for filename in ['ftv47.atsp']::
    for filename in filenames:
        matrix = ConcordeATSPSolver.load_tsplib(os.path.join(path, filename))
        solver = ConcordeATSPSolver(infinity=10 ** 6, precision=0)
        print '%-12s - %6d / %-6d' % (
            filename, solver.solve(matrix).solution()[1], optimal_costs[os.path.splitext(filename)[0]])
