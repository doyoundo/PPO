"""
copy: deepcopy라는 method를 사용하기 위한 파이썬 모듈
원본 보존을 위해 deepcopy를 사용
numpy: 파이썬 라이브러리로서 벡터, 행렬 등 수치 연산을 수행하는데 사용됩니다.
n 차원 배열로 사용되는 객체기도 합니다.
torch: Tensor library with strong GPU support
torch.nn: neural networks(신경망) 구축 및 학습 시킬때 사용되는 library
"""
import copy
import numpy as np
import torch
import torch.nn as nn


"""
알고리즘 모듈 생성
파라미터로 주어진 weight, bias, gain를 초기 weight, bias, gain으로 설정
"""
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

"""
생성된 알고리즘 모듈을 주어진 N만큼 복제하기
"""
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

"""
input으로 주어진 값의 type이 torch.tensor인지 체크
"""
def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
