import torch
import hidet
from hidet.graph.frontend.torch.interpreter import register_function

@register_function(torch.prod)
def torch_prod(x: hidet.Tensor, dim: int):
    return hidet.ops.prod(x, dim)

def test_prod():
    def func(x, dim):
        return torch.prod(x, dim)
    func = torch.compile(func, backend="hidet")
    
    x = torch.randn([2,1,14951])
    dim = 0
    hidet.graph.frontend.torch.dynamo_config.correctness_report()
    y = func(x, dim)
    print(y)

if __name__ == "__main__":
    test_prod()