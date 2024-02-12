import os 
import torch
import time 
import copy 

class Func_Trainer():
    def __init__(self, args, model):
        self.args = args
        self.model = model 
    
    def run(self, g):