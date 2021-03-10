def inc(x, y, target_, step = 0):
  print('({}, {}) :: {}'.format(x,y,target_))
  _prime = x + y
  _step = step + 1
  print('prime {}'.format(_prime))
  next_steps = [(_prime, y), (x, _prime)]
  if next_steps[0] == target_ or next_steps[1] == target_:
    return step + 1
  print(all([step[0] for step in next_steps]) > target_[0])
  print(all([step[1] for step in next_steps]) > target_[1])
  if all([step[0] for step in next_steps]) > target_[0] and all([step[1] for step in next_steps]) > target_[1]:
    return 'impossible'
  else:
    return inc()

  # if (x, y) == target_:
  #   print(step)
  #   return step
  # else:
  #   if _prime <= x_:
  #     return inc(_prime, y, target_, _step)
  #   elif _prime <= y_:
  #     return inc(x, _prime, target_, _step)
  #   return 'impossible'

def dec(x, y, step):
  # Number of steps
  M = max(x,y)
  m = min(x,y)
  # print(M, m, step)
  if m == 1:
    return step + M - m
  n = M // m
  v = M - (m * n)
  # if (n > 0):
  #   o = (n, y)
  # else:
  #   o = (x, abs(n))
  # print(v, m)
  if (v, m) == (1,1):
    print('Is this needed?')
    return step + n
  elif v < 1 or m < 1:
    return 'impossible'
  else:
    return dec(v, m, step + n)

def solution(x, y):
  x = int(x)
  y = int(y)
  number_of_steps = dec(x, y, 0)

  return str(number_of_steps)

print(solution('27199','35'))
print(solution('4','7'))
print('steps: {}'.format(solution(5,8)))
# print('steps: {}'.format(solution(41,40)))
#0                                               1,1                                                                               2
#1                       2,1                                             1,2                                          3                         3
#2            3,1                    2,3                     3,2                     1,3                        4             5           5            4
#3      4,1        3,4         5,3         2,5         5,2         3,5         4,3         1,4               5     7      8      7    7      8      7     5
#4   5,1   4,5  7,4   3,7   8,3   5,8   7,5   2,7   7,2   5,7   8,5   3,8   7,3   4,7   5,4   1,5           6 9  11 10  11 13  12 9  9 12  13 11  10 11  9 6