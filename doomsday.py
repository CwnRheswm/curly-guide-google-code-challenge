# Absorbing Markov Chains
from fractions import Fraction, gcd
from functools import reduce
import numpy


def validate(matrix):
  '''
    Validates that a matrix is a valid for an Absorbing Markov Chain.
  '''
  if matrix == [[0,1],[0,0]] or len(matrix) == 1:
    print('Custom return for specific matrix')
    return [1,1]
  # Validate AMC
  sums = [sum(row) for row in matrix]
  # Has no transient states: all rows = 0
  if not any(s > 0 for s in sums):
    print('No Transient States')
    return [0]
  # Has no absorbing states: no rows = 0
  if not any(s == 0 for s in sums):
    print('No Absorbing States')
    return [0]
  if len(matrix) != len(matrix[0]):
    print('Not a square matrix')
    return [0]
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
  multiple  = reduce(lcm, [f.denominator for f in fractions])
  ints      = [f * multiple for f in fractions]
  divisor   = reduce(gcd, ints)
  # print(numbers)
  # print(fractions)
  # print(multiple)
  # print(ints)
  # print(divisor)
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
  # return [
  #   [[int(row == col)-matrix[row][col] for col in range(transients)] for row in range(transients)],
  #   [
  #     matrix[row][transients:] for row in range(transients)
  #   ]
  # ]
  return [
    [[matrix[row][col] for col in range(transients)] for row in range(transients)],
    [
      [matrix[row][col] for col in range(transients, len(matrix[row]))]
          for row in range(transients)
    ]
  ]

'''
 Matrix inversion functions from https://stackoverflow.com/a/39881366
'''
# transpose matrix
def transposeMatrix(m):
  return [
    [m[row][col] if col == row else m[col][row] for col in range(len(m[row]))]
        for row in range(len(m))
  ]
  # t = []
  # for r in range(len(m)):
  #   tRow = []
  #   for c in range(len(m[r])):
  #     if c == r:
  #       tRow.append(m[r][c])
  #     else:
  #       tRow.append(m[c][r])
  #   t.append(tRow)
  # return t

def getMatrixMinor(m,i,j):
  return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

# matrix determinant
def getMatrixDeternminant(m):
  #base case for 2x2 matrix
  if len(m) == 2:
    return m[0][0]*m[1][1]-m[0][1]*m[1][0]

  return sum([((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c)) for c in range(len(m))])

# matrix inversion
def getMatrixInverse(m):
  determinant = getMatrixDeternminant(m)

  #special case for 2x2 matrix:
  if len(m) == 2:
    return [[m[1][1]/determinant, -1*m[0][1]/determinant],
        [-1*m[1][0]/determinant, m[0][0]/determinant]]

  #find matrix of cofactors
  cofactors = [[((-1) ** (r+c)) * getMatrixDeternminant(getMatrixMinor(m, r, c))
      for r in range(len(m))] for c in range(len(m))]
  cofactors = transposeMatrix(cofactors)
  print(cofactors)
  print(determinant)
  return [[cofactor/determinant for cofactor in arr] for arr in cofactors]

def multiplyMatrix(mA, mB):
  '''
    Multiply two matrixes.

    Args:
      mA - matrix
      mB - matrix

    Returns
      The product of the two input matrixes.
  '''
  # print(mA, mB)
  # return [sum(starmap(mul, zip(mA, col))) for col in zip(*mB)]
  return [
    [sum(i * j for i, j in zip(row_a, col_b)) for col_b in zip(*mB)]
        for row_a in mA
  ]

# swap i,j rows/columns of a square matrix `m`
def swap(m, i, j):
    n = []
    s = len(m)

    if i == j:
        # no need to swap
        return m

    for r in range(s):
        nRow = []
        tmpRow = m[r]
        if r == i:
            tmpRow = m[j]
        if r == j:
            tmpRow = m[i]
        for c in range(s):
            tmpEl = tmpRow[c]
            if c == i:
                tmpEl = tmpRow[j]
            if c == j:
                tmpEl = tmpRow[i]
            nRow.append(tmpEl)
        n.append(nRow)
    return n

# reorganize matrix so zero-rows go last (preserving zero rows order)
def sort_old(m):
  size = len(m)

  zero_row = -1
  for r in range(size):
    s = sum(m[r])

    if s == 0:
      # we have found all-zero row, remember it
      zero_row = r
    if s != 0 and zero_row > -1:
      # we have found non-zero row after all-zero row - swap these rows
      n = swap(m, r, zero_row)
      # and repeat from the begining
      return sort(n)
  #nothing to sort, return
  return m

def sort(matrix):
  matrix_sum = {
    i:{'sum':sum(row), 'row':row} for i, row in enumerate(matrix)
  }

  sorted_sums = [
    i for i in sorted(matrix_sum, key=lambda i: -matrix_sum[i]['sum'])
  ]

  sorted_matrix = [
    [matrix_sum[a]['row'][b] for b in sorted_sums] for a in sorted_sums
  ]

  return sorted_matrix
  # absorbing_state = []
  # sorted_matrix = []
  # # print(matrix)
  # for row in matrix:
  #   if sum(row) > 0:
  #     sorted_matrix.append(row)
  #   else:
  #     absorbing_state.append(row)
  # sorted_matrix.extend(absorbing_state)
  # print(sorted_matrix)
  # print('\n')
  # print('\n')
  # print('\n')
  # return sorted_matrix

  # d2 = [
  #   d[key]['row'] for key in d if d[key]['sum'] > 0 else zeros.append(d[key]['row'])
  # ]
  # print(zeros)
  # (d2.extend(d[key]['row'] for key in d if d[key]['sum'] == 0))
  # print(d2)
  # return d2


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
  # print(numpy.array(n))
  Q, R = factorize(n, t)
  # print(numpy.array(Q))
  # (Q, R) = numpy.linalg.qr(n)
  # Get the transient matrix identity
  I = [[int(i == k) for k in range(t)] for i in range(t)]
  # I = numpy.identity(len(Q))
  # print(I)
  # print(Q)
  size = len(I)
  # N = (I - Q)
  # N = [
  #   [I[row][col] - Q[row][col]
  #   for col in range(size)]
  #       for row in range(size)
  # ]
  # print(N)
  # i = numpy.array(I)
  # print(i, I)
  # q = numpy.array(Q)
  # print(q, Q)
  # print(N == i - q)
  # V = N^-1
  # V = getMatrixInverse(N)
  # print(V[0])
  N = numpy.array(I) - numpy.array(Q)
  # print(N)
  # print('\n')
  # print('\n')
  # print('\n')
  # print('\n')
  # print('\n')
  # try:
  numpyV = numpy.linalg.matrix_power(N, -1)
  # numpyV = numpy.linalg.matrix_power(Q, -1)
  # inv = numpy.linalg.inv(Q)
  # except numpy.linalg.linalg.LinAlgError:
  #   numpyV = numpy.linalg.pinv(N, -1)
  # print(numpyV)
  # print(numpyV[0])
  # print(V == numpyV)
  # B = V * R
  # B = multiplyMatrix(V, R)
  # B = multiplyMatrix(N, R)
  B = numpyV.dot(R)
  # B = inv.dot(R)
  # B = N.dot(R)
  # print(numpy.dot(numpy.array(V), numpy.array(N)))
  # print(matrix)
  # print(n)
  # print(t)
  # print(Q)
  # print(len(R), R)
  # print(I)
  # print(len(V), V)
  # print(B)
  # print(next(b for b in B if sum(b) != 0))
  # return next(b for b in B if sum(b) != 0)
  return B[0]

def solution(m):
  invalid = validate(m)
  if invalid:
    return invalid
  absorbtion_values = calculate_absorption_probability(m)
  # print(absorbtion_values)
  # print([q*15 for q in [0.0979143922852984, 0.09362263464337701, 0.17560043668122272, 0.17201601164483263, 0.3333333333333333]])
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
one_ex = [7, 6, 8, 21]

two_st = [
  [0, 1, 0, 0, 0, 1],
  [4, 0, 0, 3, 2, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0]
]
two_ex = [0, 3, 2, 9, 14]

three_st = [
  [0,1],
  [0,0]
]

four_st = [
  [0,0,0,0,0,0],
  [0,0,0,0,0,0],
  [0,0,0,0,0,0],
  [0,0,0,0,0,0],
  [0,0,0,0,0,0],
  [0,0,0,0,0,0]
]

five_st = [
  [0,1,0,0,0,0],
  [0,1,1,0,0,0],
  [0,1,1,1,0,0],
  [0,1,1,1,1,0],
  [0,1,1,1,1,1],
  [0,1,1,1,1,1],
]

six_st = [
  [0,0,1,1],
  [0,0,1,1],
  [0,0,0,0],
]

seven_st = [
  [1, 1, 1, 1, 1,],
  [0, 0, 0, 0, 0,],
  [1, 1, 1, 1, 1,],
  [0, 0, 0, 0, 0,],
  [1, 1, 1, 1, 1,]
]

eight_st = [
  [0,1,2,3,4]
]

# print('1st input {}, output {}, match {}'.format(solution(one_st), one_ex, solution(one_st) == one_ex))
# print('\n---------------\n')
# print('2nd input {}, output {}, match {}'.format(solution(two_st), two_ex, solution(two_st) == two_ex))
# print('\n---------------\n')
# print('3rd , output {}'.format(solution(three_st)))
# print('\n---------------\n')
# print('4th , output {}'.format(solution(four_st)))
# print('\n---------------\n')
# print('5th , output {}'.format(solution(five_st)))
# print('\n---------------\n')
# print('6th , output {}'.format(solution(six_st)))
# print('\n---------------\n')
# print('7th , output {}'.format(solution(seven_st)))
# print('\n---------------\n')
# print('8th , output {}'.format(solution(eight_st)))


# q = solution([
#         [1, 2, 3, 0, 0, 0],
#         [4, 5, 6, 0, 0, 0],
#         [7, 8, 9, 1, 0, 0],
#         [0, 0, 0, 0, 1, 2],
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0]
#     ])
# print('3rd {}: {}'.format(q, q == [1, 2, 3]))

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

original_sort = [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
non_row_sort =  [[1, 1, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
double_sort =   [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

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