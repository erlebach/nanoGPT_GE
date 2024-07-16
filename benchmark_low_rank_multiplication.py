"""
Test speed of low rank multiplication x @ W
W = A * B where A.shape=[N,r], B.shape=[r,N]
- NxM array
- ...,N,M array (found in NN)
"""

from timers import PerfCounterTimer as Timer

# from timers import update_dataframe
from pprint import pprint

with Timer("Initialization")():
    from low_rank import LowRankLinear
    from torch import Tensor as T
    from torch import randn
    from jaxtyping import Float
    import beartype

    seq = 256
    d = 8024
    x = randn(seq, d)

out_dict = Timer.report("\nTimings:")
Timer.reset()  # reset main timing dict


def timing(r=20):
    global df
    with Timer(f"Constructor({r=})")():
        linear = LowRankLinear(in_features=d, out_features=d, r=r, bias=False)

    with Timer(f"forward({r=}): ")():
        y = linear(x)

    out_dict = Timer.report("\nTimings:")
    df = Timer.get_dataframe()
    Timer.reset()  # reset main timing dict


timing(r=1)
timing(r=16)
timing(r=128)
timing(r=512)
timing(r=1024)

# print only the rows where 'name' contains 'forward'
# use df.query
print(df.query("name.str.contains('forward')"))
