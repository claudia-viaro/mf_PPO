from datetime import datetime
import json
from pprint import pprint
import time
import torch
import os
import pickle
import numpy as np
from constants import *





class StepGP:
    def __init__(self, args):
        self.args = args


    def step(self, action, rho_0):
        
  

            


        
        
        
        info = {"patients": self.patients, "outcome": Y_1, "Xa_pre": pat_e0[:, 2], "Xa_post":Xa_post, "rho_LogReg": rho_LogReg, "r_LogReg": reward_LogReg}     
        return rho_1, reward, done, info
