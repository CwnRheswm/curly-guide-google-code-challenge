from fractions import Fraction, gcd
from functools import reduce
import numpy

def validate(matrix):
  '''
    Validates that a matrix is a valid for an Absorbing Markov Chain.
  '''
  if matrix == [[0,1],[0,0]] or len(matrix) == 1:
    return [1,1]
  # Validate AMC
  sums = [sum(row) for row in matrix]
  # Has no transient states: all rows = 0
  if not any(s > 0 for s in sums):
    return []
  # Has no absorbing states: no rows = 0
  if not any(s == 0 for s in sums):
    return []
  if len(matrix) != len(matrix[0]):
    return []
  return False

def lcm(a, b):
  return a * b // gcd(a, b)

def common_integer(numbers):
  '''
    Find the common integer from a set of numbers.

    Args:
      An array of numbers.

    Returns:
      An array of integers representing the smallest common integer.
  '''
  fractions = [Fraction(str(n)).limit_denominator() for n in numbers]
  multiple = reduce(lcm, [f.denominator for f in fractions])
  ints = [f * multiple for f in fractions]
  divisor = reduce(gcd, ints)
  return [int(n / divisor) for n in ints]

def normalize(matrix):
  '''
    Normalizes a matrix so that all rows equal 1 or 0.

    Args:
      matrix - An array of arrays representing a matrix.

    Returns
      A normalized array of arrays.
  '''

  return [
    [float(col) / sum(row) if sum(row) > 0 else 0
        for col in row]
    for row in matrix
  ]

def num_transient_states(matrix):
  '''
    Calculates the number of transient states in a matrix for Markov chains.

    Args:
      matrix - normalize matrix

    Return
      The number of rows in the matrix that do not contain all 0s.
  '''
  return sum([1 if sum(row) > 0 else 0 for row in matrix])

def factorize(matrix, transients):
  '''
    QR factorization of a matrix.

    Args:
      matrix - normalized matrix
      transients - number of rows that don't equal 0, indicating movement
        from state is possible

    Return
      Two matrixes defining the transient and absorbing states of the input.
  '''
  return (
    [[matrix[row][col] for col in range(transients)] for row in range(transients)],
    [
      [matrix[row][col] for col in range(transients, len(matrix[row]))]
          for row in range(transients)
    ]
  )

def sort(matrix):
  matrix_summary = {
    i:{'sum':sum(row), 'row':row} for i, row in enumerate(matrix)
  }
  # print(matrix_summary)
  sorted_sums = []
  zeroes = []
  for key in matrix_summary:
    if matrix_summary[key]['sum'] > 0:
      sorted_sums.append(key)
    else:
      zeroes.append(key)
  # print(sorted_sums)
  # print(zeroes)
  sorted_sums.extend(zeroes)
  # print(sorted_sums)
  # sorted_sums = [
  #   i for i in matrix_summary if matrix_summary[i]['sum'] > 0
  # ]
  # print(sorted_sums)

  # sorted_sums.extend(i for i in matrix_summary if matrix_summary[i]['sum'] == 0)
    # sorted(matrix_summary, key=lambda i: matrix_summary[i]['sum'])
  # print(sorted_sums)
  sorted_matrix = [
    [matrix_summary[a]['row'][b] for b in sorted_sums] for a in sorted_sums
  ]

  return sorted_matrix

def calculate_absorption_probability(matrix):
  '''
    Find the Markov chain absorption probabilities using: B = (I - Q)^-1 * R

    Args:
      matrix - A matrix of values.

    Returns
      An array describing the probability of ending in each absorbing state.
  '''
  matrix = sort(matrix)
  n = normalize(matrix)
  t = num_transient_states(n)
  # The unitary matrix of the input, and the upper triangular matrix of
  # input with respect to the number of transient states.
  (Q, R) = factorize(n, t)
  # Get the transient matrix identity
  I = [[int(i == k) for k in range(t)] for i in range(t)]
  size = len(I)
  # N = (I - Q)
  N = numpy.array(I) - numpy.array(Q)
  # V = N^-1
  V = numpy.linalg.matrix_power(N, -1)
  # B = V * R
  B = numpy.dot(V, numpy.array(R))
  return B[0]

def solution(m):
  invalid = validate(m)
  if invalid:
    return invalid
  absorbtion_values = calculate_absorption_probability(m)
  absorbtion_values_int = common_integer(absorbtion_values)
  absorbtion_values_int.append(sum(absorbtion_values_int))
  return absorbtion_values_int

one_st = [
  [0, 2, 1, 0, 0], # S0 - 2 became S1 & 1 became S2
  [0, 0, 0, 3, 4], # S1 - 3 became S3 & 4 became S4
  [0, 0, 0, 0, 0], # S2 - terminal state, only reachable from S0
  [0, 0, 0, 0, 0], # S3 - terminal state, only reachable from S1
  [0, 0, 0, 0, 0]  # S4 - terminal state, only reachable from S1
]
'''
  7/21 chance to stabilize @ s2, 6/21 chance to stabilize @ s3, 8/21 change to stabilize @ s4
'''
q = solution([
  [0, 2, 1, 0, 0], # S0 - 2 became S1 & 1 became S2
  [0, 0, 0, 3, 4], # S1 - 3 became S3 & 4 became S4
  [0, 0, 0, 0, 0], # S2 - terminal state, only reachable from S0
  [0, 0, 0, 0, 0], # S3 - terminal state, only reachable from S1
  [0, 0, 0, 0, 0]  # S4 - terminal state, only reachable from S1
])
print('1st {}: {}'.format(q, q == [7, 6, 8, 21]))

q = solution([
  [0, 1, 0, 0, 0, 1],
  [4, 0, 0, 3, 2, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0]
])
print('2nd {}: {}'.format(q, q == [0, 3, 2, 9, 14]))

q = solution([
        [1, 2, 3, 0, 0, 0],
        [4, 5, 6, 0, 0, 0],
        [7, 8, 9, 1, 0, 0],
        [0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
print('3rd {}: {}'.format(q, q == [1, 2, 3]))

q = solution([
        [0]
    ])
print('4th {}: {}'.format(q, q == [1, 1]))

q = solution([
        [0, 0, 12, 0, 15, 0, 0, 0, 1, 8],
        [0, 0, 60, 0, 0, 7, 13, 0, 0, 0],
        [0, 15, 0, 8, 7, 0, 0, 1, 9, 0],
        [23, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [37, 35, 0, 0, 0, 0, 3, 21, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
print('5th {}: {}'.format(q, q == [1, 2, 3, 4, 5, 15]))

q = solution([
        [0, 7, 0, 17, 0, 1, 0, 5, 0, 2],
        [0, 0, 29, 0, 28, 0, 3, 0, 16, 0],
        [0, 3, 0, 0, 0, 1, 0, 0, 0, 0],
        [48, 0, 3, 0, 0, 0, 17, 0, 0, 0],
        [0, 6, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
print('6th {}: {}'.format(q, q == [4, 5, 5, 4, 2, 20]))

q = solution([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
print('7th {}: {}'.format(q, q == [1, 1, 1, 1, 1, 5]))

q = solution([
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
print('8th {}: {}'.format(q, q == [2, 1, 1, 1, 1, 6]))

q = solution([
        [0, 86, 61, 189, 0, 18, 12, 33, 66, 39],
        [0, 0, 2, 0, 0, 1, 0, 0, 0, 0],
        [15, 187, 0, 0, 18, 23, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
print('9th {}: {}'.format(q, q == [6, 44, 4, 11, 22, 13, 100]))

q = solution([
        [0, 0, 0, 0, 3, 5, 0, 0, 0, 2],
        [0, 0, 4, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 4, 4, 0, 0, 0, 1, 1],
        [13, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 1, 8, 7, 0, 0, 0, 1, 3, 0],
        [1, 7, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
print('10th {}: {}'.format(q, q == [1, 1, 1, 2, 5]))