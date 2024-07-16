"""
Test speed of te attention layer where weights 
are replaced by products of low-rank counterparts. 
What are the conditions under which the speed of 
low-rank multiplications is sufficiently lower than 
using the full weight matrices. 
"""

from timers import PerfCounterTimer as Timer

# from timers import update_dataframe
from pprint import pprint
from jaxtyping import Float
import beartype

from model_low_rank import CausalSelfAttention
from low_rank import LowRankLinear
from torch import Tensor as T
from torch import randn
from dataclasses import dataclass

b = 16
seq = 256
d = 1024*4*4
# x: Float[T, "B T C"] (batch, seq_len, embedding)
x = randn(b, seq, d) #.view(b, seq, d)
# print(x.size())

out_dict = Timer.report("\nTimings:")
# df = update_dataframe(None, out_dict)
Timer.reset()  # reset main timing dict

@dataclass
class Config:
    n_head: int = 1
    n_embd: int =  d
    rank: int = 8
    bias: bool =  False
    dropout: float = .1
    block_size: int = 16
    device: str = 'mps'

config = Config()

x.to(config.device)

assert config.n_embd % config.n_head == 0


def timing(r=20):
    global df
    with Timer(f"Constructor({r=})")():
        attn = CausalSelfAttention(config)
        attn.to(config.device)

    with Timer(f"forward({r=}): ")():
        y = attn(x)

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
