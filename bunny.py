# Python combination
from itertools import combinations

def solution(n_buns, n_req):
  num_bunnies_per_key = n_buns - n_req + 1
  keys_to_distribute = [[] for w in range(n_buns)]
  key_sets = list(combinations(range(n_buns), num_bunnies_per_key))
  for key, bunnies in enumerate(key_sets):
    for bunny in bunnies:
      keys_to_distribute[bunny].append(key)
  return keys_to_distribute

a = solution(2, 1)
print(a, a == [[0], [0]])

b = solution(4, 4)
print(b, b == [[0], [1], [2], [3]])

c = solution(5, 3)
print(c, c == [
  [0,1,2,3,4,5],
  [0,1,2,6,7,8],
  [0,3,4,6,7,9],
  [1,3,5,6,8,9],
  [2,4,5,7,8,9]
])

d = solution(3, 2)
print(d, d == [[0,1], [0,2], [1,2]])

e = solution(2, 2)
print(e, e == [[0], [1]])
