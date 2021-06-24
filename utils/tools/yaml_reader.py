import yaml

from collections import OrderedDict


def read_yaml(filepath):

    with open(filepath) as file:

        content = yaml.load(file, Loader=yaml.FullLoader)

    return content


# def read_yaml_ordered(filepath) -> OrderedDict:

#     with open(filepath) as file:

#         content_normal = yaml.load(file, Loader=yaml.FullLoader)
#         content_ordered = yaml.safe_load(file)

#         print(type(content_normal), content_normal)
#         print(type(content_ordered), content_ordered)

#     return content_ordered

#         # yaml_dict = OrderedDict(abc=OrderedDict([("x", OrderedDict([(0, None)])), ("y", OrderedDict([(1, None)]))]))

#         # print(yaml_dict)

#         # a = dict(zip("unsorted", "unsorted"))
#         # b = yaml.safe_load(s)

#         # assert list(a.keys()) == list(b.keys())  # True

