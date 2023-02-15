import time

import torch
from torch import nn
import pandas as pd
import numpy as np
import onnx


def read_edge_list(path):
    start = time.time()
    edge_list = torch.tensor(pd.read_table(path, skiprows=4).values,dtype=torch.int64)
    end = time.time()
    edge_list = torch.reshape(edge_list, (2, -1))
    print("already load the edge list of the graph, costs ", end - start)
    return edge_list

#please update this path
edge_list = read_edge_list(r'../Downloads/Wiki-Vote.txt')

# dummy_input  = read_edge_list() # 注意输入type一定要np.float32!!!!!
class PageRank_CPU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,source,target,init_vertex,iteration,vertex_num):

        source = source.int()
        target = target.int()
        V_out_deg = torch.zeros_like(init_vertex,dtype=torch.int32)
        V_out_deg = V_out_deg.scatter_add(0, source , torch.ones_like(source,dtype=torch.int32))
        mask = (V_out_deg == 0)
        V_old = init_vertex
        sum =  torch.sum(V_old)
        V_old = V_old /sum
        # start iteration
        round = torch.tensor(0)
        while round < iteration:
            # print("round ",round,": vertex[100]=",V_old[300])
            V_new = torch.zeros_like(init_vertex)
            V_old_temp = V_old / V_out_deg
            blind_sum = torch.masked_select(V_old,mask).sum()
            V_new = V_new.scatter_add(0, target, V_old_temp[source])
            V_new = V_new * 0.85 + (0.15 + blind_sum * 0.85) / vertex_num
            diff = torch.abs(V_new-V_old).sum()
            V_old = V_new
            round+=1
            if torch.lt(diff,1e-7):break

        return V_old


vertex_num = torch.max(edge_list)+1
model = PageRank_CPU()
model.eval()
dummy_input = (edge_list[0,:],edge_list[1,:],torch.rand(vertex_num),torch.tensor(30),torch.tensor(vertex_num))
scripted_model = torch.jit.script(model,example_inputs=dummy_input).eval()
print(scripted_model.code)

input_names = ["source",'target','init_vertex','iter','vertex_num']
output_names = ["output_0"]
 
 
import tvm
from tvm import relay
print(tvm.__version__)
shape_list = list(zip(input_names,[i.shape for i in dummy_input]))
print(shape_list)
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
print(mod)

target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
