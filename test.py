
from typing import List
import torch
import hidet
# def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
#     print(">>> my_compiler() invoked:")
#     print(">>> FX graph:")
#     gm.graph.print_tabular()
#     print(f">>> Code:\n{gm.code}")
#     return gm.forward  # return a python callable

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if torch.sum(b) < 0:
        b = b * -1
    return x * b

def test(a, b):
    for i in range(4):
        toy_example(a, b * (-1) ** i)

if __name__ == "__main__":
    a, b = torch.randn(10), torch.ones(10)
    toy_example = torch.compile(toy_example, backend="hidet")
    # hidet.frontend.torch.dynamo_config.dump_graph_ir("hidet_graph")
    test(a, b)