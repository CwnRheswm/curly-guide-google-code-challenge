# Ford Fulkerson maximum flow in a flow network
import collections
import numpy

def search(entrances, exits, path, parent):
  visited = [True if i in entrances else False for i, _ in enumerate(parent)]

  queue = [enter for enter in entrances]
  while queue:
    print(queue)
    row = queue.pop(0)
    # print (row, path[row])
    for i, cell in enumerate(path[row]):
      if visited[i] == False and cell > 0:
        print(i)
        queue.append(i)
        visited[i] = True
        parent[i] = row
  # print([visited[e] for e in exits])
  print('Visited: {}'.format(visited))
  print('Exits: {}'.format(exits))
  print([visited[e] for e in exits])
  return any([visited[e] for e in exits])

def breadth_first_search_w_comment(source, sink, path, parent):
  # print(source, sink, len(path))
  visited = [True if i == source else False for i in range(len(parent))]
  # print(visited)

  queue = collections.deque([source])
  while queue:
    # print('Q {}'.format(queue))
    row = queue.popleft()

    for index, cell in enumerate(path[row]):
      # print(visited[index], cell, cell > 0)
      if visited[index] == False and cell > 0:
        queue.append(index)
        visited[index] = True
        parent[index] = row
  # print('bfs {}'.format(visited))
  return visited[sink]

def solution_w_comment(entrances, exits, path):
  rows = len(path)
  max_flow = 0

  if len(entrances) > 1:
    source_row = [float('Inf') if cell in entrances else 0 for cell in range(rows)]
    sink_row = [0 for cell in range(len(path))]

    path.insert(0, source_row)
    path.append(sink_row)
    rows = len(path)
    for row in path:
      row.insert(0,0)
      row.append(0)
    for row in exits:
      path[row + 1][len(path[row]) - 1] = float('Inf')

  parent = [-1 for row in range(rows)]
  source = 0
  sink = rows - 1
  print(numpy.array(path))
  while breadth_first_search_w_comment(source, sink, path, parent):
    path_flow = float('Inf')

    s_ = sink
    while s_ != source:
      path_flow = min(path_flow, path[parent[s_]][s_])
      s_ = parent[s_]

    max_flow += path_flow
    print('current_flow {}'.format(max_flow))
    # vs = exits
    v = sink
    while v != source:
      n = parent[v]
      path[n][v] -= path_flow
      path[v][n] += path_flow
      v = parent[v]
      # vs = [parent[v] for v in vs]
  return max_flow

# def queue_search(sink, path, parent, visited, queue):
  if not queue:
    return visited[sink]
  else:
    row = queue.popleft()

    for index, cell in enumerate(path[row]):
      if visited[index] == False and cell > 0:
        queue.append(index)
        visited[index] = True
        parent[index] = row
    return queue_search(sink, path, parent, visited, queue)

def breadth_first_search(source, sink, path, parent):
  visited = [True if i == source else False for i in range(len(parent))]

  queue = collections.deque([source])
  while queue:
    row = queue.popleft()

    for index, cell in enumerate(path[row]):
      if visited[index] == False and cell > 0:
        queue.append(index)
        visited[index] = True
        parent[index] = row

  return visited[sink]

def solution(entrances, exits, path):
  rows = len(path)
  max_flow = 0

  if len(entrances) > 1 or len(exits) > 1 or exits[0] != rows - 1:
    src_row = [float('Inf') if cell in entrances else 0 for cell in range(rows)]
    sink_row = [0 for cell in range(rows)]

    # Add new single-source row
    path.insert(0, src_row)
    # Add new single-sink row
    path.append(sink_row)
    rows = len(path)

    for row in path:
      # Prepend a 0 cell to each row account for single-source row
      row.insert(0, 0)
      # Append a 0 cell to the end of each row to account for single-sink row
      row.append(0)

    for exit_row in exits:
      path[exit_row + 1][len(path[exit_row]) - 1] = float('Inf')

  parent = [-1 for row in range(rows)]
  source = 0
  sink = rows - 1

  while breadth_first_search(source, sink, path, parent):
    path_flow = float('Inf')

    s_ = sink
    while s_ != source:
      path_flow = min(path_flow, path[parent[s_]][s_])
      s_ = parent[s_]

    max_flow += path_flow

    v = sink
    while v != source:
      n = parent[v]
      path[n][v] -= path_flow
      path[v][n] += path_flow
      v = parent[v]

  return max_flow

def solution_two(entrances, exits, path):
  rows = len(path)
  max_flow = 0

  if len(entrances) > 1:
    src_row = [float('Inf') if cell in entrances else 0 for cell in range(rows)]
    sink_row = [0 for cell in range(rows)]

    path.insert(0, src_row)
    path.append(sink_row)
    rows = len(path)
    for row in path:
      row.insert(0,0)
      row.append(0)

    for exit_row in exits:
      path[exit_row + 1][len(path[exit_row]) - 1] = float('Inf')

  parent = [-1 for row in range(rows)]
  source = 0
  sink = rows - 1
  print(numpy.array(path))
  while breadth_first_search_w_comment(source, sink, path, parent):
    path_flow = float('Inf')

    s_ = sink
    while s_ != source:
      print(parent)
      path_flow = min(path_flow, path[parent[s_]][s_])
      s_ = parent[s_]

    max_flow += path_flow

    v = sink
    while v != source:
      n = parent[v]
      path[n][v] -= path_flow
      path[v][n] += path_flow
      v = parent[v]

  return max_flow

print(solution([0], [3], [[0, 7, 0, 0], [0, 0, 6, 0], [0, 0, 0, 8], [9, 0, 0, 0]]))
entrances = [0]
exits = [3]
path = [
  [0,7,0,0],
  [0,0,6,0],
  [0,0,0,8],
  [9,0,0,0]
]
flow = 6
test_one = solution(entrances, exits, path)
print(test_one, test_one == flow)
print('\n------------\n')
entrances = [0]
exits = [2]
path = [
  [0,7,0,0],
  [0,0,6,0],
  [9,0,0,0],
  [0,0,0,8]
]
flow = 6
test_one_comb = solution(entrances, exits, path)
print(test_one_comb, test_one_comb == flow)
print('\n------------\n')
entrances = [0, 1]
exits = [4, 5]
path = [
  [0,0,4,6,0,0],
  [0,0,5,2,0,0],
  [0,0,0,0,4,4],
  [0,0,0,0,6,6],
  [0,0,0,0,0,0],
  [0,0,0,0,0,0]
]
flow = 16
test_two = solution(entrances, exits, path)
print(test_two, test_two == flow)
print('\n------------\n')
entrances = [0, 1]
exits = [3, 5]
path = [
  [0,0,4,6,0,0],
  [0,0,5,2,0,0],
  [0,0,0,0,4,4],
  [0,0,0,0,0,0],
  [0,0,0,0,6,6],
  [0,0,0,0,0,0]
]
flow = 16
test_two_one = solution(entrances, exits, path)
print(test_two_one, test_two_one == flow)
print('\n------------\n')
entrances = [0]
exits = [2, 4]
path = [
  [0,9,8,0,0],
  [0,0,0,4,4],
  [0,0,0,0,0],
  [0,0,0,6,6],
  [0,0,0,0,0]
]
flow = 16
test_two_two = solution(entrances, exits, path)
print(test_two_two, test_two_two == flow)