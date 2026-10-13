"""Create a small GIF from a real rollout; no target imagery enters transition."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from mycelium_nca.model_io import load_model
from mycelium_nca.replay import Simulation
from mycelium_nca.numpy_sim import render_rgba

p=argparse.ArgumentParser();p.add_argument('--model',default='models/ring-regeneration.json');p.add_argument('--output',default='reports/demo.gif');a=p.parse_args()
m=load_model(a.model);s=Simulation(m['weights'],seed=90501,model_hash=m['weights_hash'])
frames=[]
for t in range(513):
    if t==256:s.apply({'type':'erase','shape':'rectangle','x':16,'y':9,'width':7,'height':9})
    if t%4==0:
        pixels=Image.fromarray(render_rgba(s.state),'RGBA').resize((256,256),Image.Resampling.NEAREST)
        frame=Image.new('RGB',(304,310),(18,24,22));frame.paste(pixels,(24,28),pixels)
        draw=ImageDraw.Draw(frame);draw.text((16,9),'MYCELIUM / learned local rules',fill=(194,211,146))
        draw.text((16,291),f'Step {t:03d}  /  '+('Growth' if t<256 else 'After rectangle damage'),fill=(226,231,211))
        frames.append(frame)
    if t<512:s.advance()
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
frames[0].save(a.output,save_all=True,append_images=frames[1:],duration=55,loop=0,optimize=True)
Path(a.output).with_suffix('.json').write_text(json.dumps({'schema':'mycelium.demo.v1','weights_hash':m['weights_hash'],'seed':90501,'frames':len(frames),'simulation_steps':512,'sample_every':4,'damage':{'step':256,'type':'erase','shape':'rectangle','x':16,'y':9,'width':7,'height':9},'note':'Actual independent demonstration rollout. This GIF is recorded; the application is live.'},indent=2)+'\n')
print(a.output)
