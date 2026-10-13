import numpy as np
from mycelium_nca.model_io import load_model
from mycelium_nca.replay import Simulation
from mycelium_nca.targets import write_png
from mycelium_nca.metrics import measure
model=load_model('models/ring-regeneration.json')
sim=Simulation(model['weights'],seed=90101)
target=np.load('targets/ring.npy')
frames=[]
for t in range(513):
    if t==256:
        frames.append(sim.state[...,:4].copy())
        sim.apply({'type':'erase','shape':'rectangle','x':16,'y':9,'width':7,'height':9})
        frames.append(sim.state[...,:4].copy())
    if t in (0,32,64,128,384,512): frames.append(sim.state[...,:4].copy())
    if t<512: sim.advance()
write_png('reports/anchor-contact-sheet.png',np.concatenate([target,*frames],axis=1),scale=4)
print(measure(sim.state,target))
