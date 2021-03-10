# import re
# def find_index(version, existing_list):
#   version_blowout = version.split('.')
#   print("version_blowout {}".format(version_blowout))
#   return_index = len(existing_list)
#   for index, version_ord in enumerate(existing_list):
#     version_ord_blowout = version_ord.split('.')
#     for i in range(len(version_blowout)):
#       print('Versions {}: {}'.format(version_blowout[i], version_ord_blowout[i] if len(version_ord_blowout) > i else "none"))
#       if len(version_ord_blowout) > i and version_blowout[i] > version_ord_blowout[i]:
#         return_index = index
#   return return_index

# def ssoorrtt(version_list, new_version_list = []):
#   version = version_list.pop()
#   index_insert = find_index(version, new_version_list)
#   print('index_insert {}'.format(index_insert))
#   new_version_list.insert(index_insert, version)
#   # print('Length Version List {}'.format(len(version_list)))
#   if len(version_list) == 0:
#     print('NVL {}'.format(new_version_list))
#     return new_version_list
#   else:
#     ssoorrtt(version_list)


# def s(list_, max_length, index = 0):
#   new_order = []
#   for version in list_:
#     # print(version)
#     ind = 0
#     for ins, new_version in enumerate(new_order):
#       vers = int(version[index]) if len(version) > index else -1
#       new_vers = int(new_version[index]) if len(new_version) > index else -1
#       prev_vers = int(version[index - 1]) if len(version) > index - 1 else -1
#       prev_new_vers = int(new_version[index - 1]) if len(new_version) > index - 1 else -1
#       print(ins, vers, new_vers, prev_vers, prev_new_vers)
#       if index > 0 and prev_vers > prev_new_vers:
#         ind = ins + 1
#       elif vers > new_vers:
#         ind = ins + 1
#       print(ind)
#     new_order.insert(ind, version)
#     print(new_order)
#   if index == max_length - 1:
#     return new_order
#   else:
#     return s(new_order, max_length, index + 1)
  #   insert_index = len(new_order)
  #   if len(version) > index:
  #     for i, v in enumerate(new_order):
  #       if len(v) > index and version[index] < v[index]:
  #         if index > 0:
  #           if version[index - 1] == v[index - 1]:
  #             insert_index = i
  #         else:
  #           insert_index = i
  #   new_order.insert(insert_index, version)
  # print(new_order)
  # if index == max_length -1:
  #   return ['.'.join(version) for version in new_order]
  # else:
  #   return s(new_order, max_length, index + 1)

# def d(version_list, index):
#   for version in version_list:
#     value = version[index] if len(version) > index else -1
#     d[value]

def dimensional_array_to_dict(arr, obj):
  # If array values has a single value, indicating bottom of the stack
  if all([len(value) == 1 for value in arr]):
    return [value[0] for value in arr]
  # Adds to the dictionary object so each key contains all lower revision
  # numbers as an array.
  else:
    [obj[value[0]].append(value[1:])
        if value[0] in obj else obj.update({value[0]: [value[1:]]})
      for value in arr]

  # Runs the value under each key back through the function to move the next
  # level of values into the dictionary
  for key in obj:
    obj[key] = dimensional_array_to_dict(obj[key], {})

  return obj

def solution(version_list):
  # Adds -1 values for all versions that don't include minor and/or revision
  versions_w_normalized_dimensions = [
    [int(version_arr[index]) if len(version_arr) > index else -1
        for index in range(3)]
        for version_arr in [version.split('.') for version in version_list]
  ]

  # Converts the multi dimensional array into a dictionary
  version_dict = dimensional_array_to_dict(versions_w_normalized_dimensions, {})

  # Sorts each level of the version dictionary and joins into a string
  # return ','.join(
  #   ['.'.join(str(v) for v in [a,b,c] if v > -1)
  #     for a in sorted(version_dict)
  #     for b in sorted(version_dict[a])
  #     for c in sorted(version_dict[a][b])])

  return ['.'.join(str(v) for v in [a,b,c] if v > -1)
      for a in sorted(version_dict)
      for b in sorted(version_dict[a])
      for c in sorted(version_dict[a][b])]

y = solution(["1.11", "2.0.0", "1.2", "2", "0.1", "1.2.1", "1.1.1", "2.0"])
print(y)
print(y == ['0.1','1.1.1','1.2','1.2.1','1.11','2','2.0','2.0.0'])
x = solution(["1.1.2", "1.0", "1.3.3", "1.0.12", "1.0.2"])
print(x)
print(x == ['1.0','1.0.2','1.0.12','1.1.2','1.3.3'])