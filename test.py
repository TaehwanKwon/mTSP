from pprint import pprint
from IPython import embed

import logging
logging.basicConfig(level=logging.INFO)

from envs.mtsp_simple import MTSP, MTSPSimple
from models.gnn import Model

config = __import__('conf.conf-kth', fromlist=[None]).config

model = Model(config)

mtsp = MTSP(config)
s = mtsp.reset()

for _ in range(10):
    action = model.action(s)
    s, r, done = mtsp.step(action)
    mtsp.render()
    if done: break
