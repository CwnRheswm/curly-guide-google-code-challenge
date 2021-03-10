row = 8
visited = set()

def validmove(start, end):
  startRow = start % row
  endRow = end % row
  if end < 64 and end > -1 and abs(startRow - endRow) < 3 and end not in visited:
    return True
  return False

def move(start):
  validmoves = []
  for num in start:
    for i in xrange(-2,3):
      for j in xrange(-2,3):
        if i == 0 or j == 0:
          continue
        end = num + (row * i) + j
        if abs(i) != abs(j) and validmove(num, end):
          visited.add(end)
          validmoves.append(end)
  return validmoves

def findpath(start, target, num = 0):
  num = num + 1
  moves = move(start if isinstance(start, list) else [start])
  if target in moves:
    return num
  else:
    return findpath(moves, target, num)

def solution(src, dest):
  if src == dest:
    return 0
  if src >= 0 and src < 64:
    return findpath(src, dest)
  else:
    return

print(solution(23, 17))