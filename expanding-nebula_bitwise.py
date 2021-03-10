from collections import defaultdict

def is_valid(s, n, columns, validity):
  '''
    Compute the next state and compare that to the validation input.

    Args:
      s - Previous state index
      n - Current state index
      columns - The number of columns in the graph.
      validity - An array a valid column codes.

    Returns
      The valid column code or None.
  '''
  s0 = s >> 1
  s1 = s & ~(1 << columns)
  n0 = n >> 1
  n1 = n & ~(1 << columns)

  next_state = (
    (s1 & ~n1 & ~s0 & ~n0) |
    (~s1 & n1 & ~s0 & ~n0) |
    (~s1 & ~n1 & s0 & ~n0) |
    (~s1 & ~n1 & ~s0 & n0))

  if next_state in validity:
    return next_state

def bitwise_sum(row):
  '''
    Converts rows into single values representing the state of the row.

    Args:
      row - An array of values

    Returns
      A single number that is used to represent the state of row.
  '''
  return sum([(1 << index) * state for index, state in enumerate(row)])

def transpose(g):
  '''
    Ensure that we are using the smallest number of columns

    Args:
      g - A nested array representing the grid

    Returns
      A nested array ensuring the second dimension is smaller or equal
  '''
  if len(g[0]) > len(g):
    g = list(zip(*g))

  return g

def solution(g):
  g = transpose(g)

  columns = len(g[0])

  columns_bit_summary = [bitwise_sum(row) for row in g]
  unique_columns = set(columns_bit_summary)

  # Number of potential previous states
  num_previous_states = 1 << columns + 1

  valid_states_cache = defaultdict(lambda: defaultdict(set))

  for p_state in range(num_previous_states):
    for state in range(num_previous_states):
      valid_column = is_valid(p_state, state, columns, unique_columns)

      if valid_column is not None:
        valid_states_cache[valid_column][p_state].add(state)

  # Dictionary to hold previous states
  # previous_states = {state: 1 for state in range(num_previous_states)}
  previous_states = defaultdict(lambda: 1)

  for column in columns_bit_summary:
    next_states = defaultdict(int)

    # for state in previous_states:
    for state in range(num_previous_states):
      for valid_state in valid_states_cache[column][state]:
        next_states[valid_state] += previous_states[state]

    previous_states = next_states

  return sum(previous_states.values())

import time
start_time = time.time()
one = [
  [True, False, True],
  [False, True, False],
  [True, False, True]
]
one_expected = 4
one_result = solution(one)
print('After {:0.5f} seconds, {} {} {}'.format(
  time.time() - start_time, one_result,
  'equals' if one_result == one_expected else 'does NOT equal', one_expected))

start_time = time.time()
two = [
  [True, False, True, False, False, True, True, True],
  [True, False, True, False, False, False, True, False],
  [True, True, True, False, False, False, True, False],
  [True, False, True, False, False, False, True, False],
  [True, False, True, False, False, True, True, True]
]
two_expected = 254
two_result = solution(two)
print('After {:0.5f} seconds, {} {} {}'.format(
  time.time() - start_time, two_result,
  'equals' if two_result == two_expected else 'does NOT equal', two_expected))

start_time = time.time()
three = [
  [True, True, False, True, False, True, False, True, True, False],
  [True, True, False, False, False, False, True, True, True, False],
  [True, True, False, False, False, False, False, False, False, True],
  [False, True, False, False, False, False, True, True, False, False]
]
three_expected = 11567
three_result = solution(three)
print('After {:0.5f} seconds, {} {} {}'.format(
  time.time() - start_time, three_result,
  'equals' if three_result == three_expected else 'does NOT equal', three_expected))