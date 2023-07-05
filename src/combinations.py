# from functools import cache, lru_cache
from itertools import product


# @lru_cache
def combinations(n, k):
    """
    Generate all possible sequences of integers of length k that add up to n
    """
    if n == 0:
        return [(0,) * k]
    elif k == 1:
        return [(n,)]
    else:
        res = []
        for i in range(n + 1):
            res += list(map(lambda lst: (i,) + lst, combinations(n - i, k - 1)))
        return res


# @lru_cache
def combinations_(n, k):
    """
    assume k is a power of 2
    """
    if n == 0:
        return [(0,) * k]
    elif k == 1:
        return [(n,)]
    else:
        res = []
        for l in range(n + 1):
            left_list = combinations_(l, k // 2)
            right_list = combinations_(n - l, k - (k // 2))
            res += list(
                map(lambda tup: (tup[0] + tup[1]), product(left_list, right_list))
            )
        return res


def increment(lst: list):
    target_sum = sum(lst)
    num_bins = len(lst)

    for i, k in enumerate(lst):
        if k > 0:
            if k == target_sum and i + 1 == num_bins:
                return None
            else:
                lst[i + 1] += 1
                lst[0] = lst[i] - 1
                if i != 0:
                    lst[i] = 0
                return lst


if __name__ == "__main__":
    C = combinations_(6, 10)
    print(C)
    print(len(C))
