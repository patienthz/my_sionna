
"""Clustered delay line (CDL) channel model from 3GPP TR38.901 specification"""




import json
from importlib_resources import files
import torch
from torch import cos,sin
import numpy as np

from my_code.mysionna.channel.torch_version.utils import deg_2_rad
from my_code.mysionna.channel.torch_version.channel_model import ChannelModel
from my_code.mysionna import PI
from my_code.mysionna.utils.tensors import insert_dims

from .channel_coefficients import Topology,ChannelCoefficientsGenerator



