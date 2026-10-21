#!/usr/bin/env python3
"""Render Mycelium's static sharing artwork from a fixed, real NCA rollout."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from mycelium_nca.model_io import load_model
from mycelium_nca.numpy_sim import render_rgba
from mycelium_nca.replay import Simulation

HERE = Path(__file__).resolve().parent
SCALE = 3
SIZE = (1200, 630)
COLORS = {'bg':'#171b18','canvas':'#111713','cream':'#edeedf','moss':'#c2d392','teal':'#8bb6c2','line':'#394238','muted':'#a1ad9d','amber':'#d8b378'}
FONTS = {
    'serif': (os.environ.get('MYCELIUM_OG_SERIF','/System/Library/Fonts/Supplemental/Georgia.ttf'),0,('Georgia','Regular')),
    'italic': (os.environ.get('MYCELIUM_OG_ITALIC','/System/Library/Fonts/Supplemental/Georgia Italic.ttf'),0,('Georgia','Italic')),
    'sans': (os.environ.get('MYCELIUM_OG_SANS','/System/Library/Fonts/Avenir Next.ttc'),7,('Avenir Next','Regular')),
    'medium': (os.environ.get('MYCELIUM_OG_SANS','/System/Library/Fonts/Avenir Next.ttc'),5,('Avenir Next','Medium')),
    'mono': (os.environ.get('MYCELIUM_OG_MONO','/System/Library/Fonts/Menlo.ttc'),0,('Menlo','Regular')),
}

def sha(data): return hashlib.sha256(data).hexdigest()

def font(kind,size):
    path,index,expected=FONTS[kind]
    if not Path(path).is_file(): raise RuntimeError(f'Missing licensed {expected[0]} font: {path}. See tools/og/README.md.')
    result=ImageFont.truetype(path,round(size*SCALE),index=index)
    if result.getname()!=expected: raise RuntimeError(f'Expected {expected}, loaded {result.getname()}. No silent font fallback is allowed.')
    return result

class Poster:
    def __init__(self):
        self.image=Image.new('RGB',(SIZE[0]*SCALE,SIZE[1]*SCALE),COLORS['bg'])
        self.draw=ImageDraw.Draw(self.image)
    def text(self,xy,text,size=20,kind='sans',fill='cream',anchor='lt'):
        self.draw.text(tuple(round(x*SCALE) for x in xy),text,font=font(kind,size),fill=COLORS.get(fill,fill),anchor=anchor,stroke_width=0)
    def line(self,coords,fill='line',width=1):
        self.draw.line([round(n*SCALE) for n in coords],fill=COLORS.get(fill,fill),width=max(1,round(width*SCALE)))
    def ellipse(self,box,fill=None,outline='line',width=1):
        self.draw.ellipse([round(n*SCALE) for n in box],fill=COLORS.get(fill,fill),outline=COLORS.get(outline,outline),width=max(1,round(width*SCALE)))
    def cells(self,state,x,y,size):
        pixels=Image.fromarray(render_rgba(state),'RGBA').resize((round(size*SCALE),round(size*SCALE)),Image.Resampling.NEAREST)
        self.image.paste(pixels,(round(x*SCALE),round(y*SCALE)),pixels)
    def plate(self,state,cx,cy,diameter):
        r=diameter/2
        self.ellipse((cx-r,cy-r,cx+r,cy+r),fill='canvas')
        self.ellipse((cx-r+9,cy-r+9,cx+r-9,cy+r-9),outline='#2c362d',width=.7)
        self.cells(state,cx-r,cy-r,diameter)
        for x,y,dx,dy in [(cx,cy-r,0,12),(cx,cy+r,0,-12),(cx-r,cy,12,0),(cx+r,cy,-12,0)]:
            self.line((x-dx/2,y-dy/2,x+dx/2,y+dy/2),fill='muted',width=.8)
    def save(self,path):
        out=self.image.resize(SIZE,Image.Resampling.LANCZOS)
        png=PngImagePlugin.PngInfo();png.add(b'sRGB',b'\x00')
        path.parent.mkdir(parents=True,exist_ok=True)
        out.save(path,format='PNG',optimize=True,compress_level=9,pnginfo=png)
        return out

def states():
    model=load_model((ROOT/'models/ring-regeneration.json').read_bytes())
    sim=Simulation(model['weights'],seed=123,size=32,update_probability=0.5)
    sim.advance(256);grown=sim.state.copy()
    event={'type':'erase','shape':'rectangle','x':16,'y':0,'width':16,'height':32}
    sim.apply(event);cut=sim.state.copy()
    sim.advance(256);repaired=sim.state.copy()
    provenance={'model':'models/ring-regeneration.json','weights_hash':model['weights_hash'],'seed':123,'probability':0.5,'numpy':np.__version__,'pillow':PIL.__version__,'event':{**event,'step':256},'states':{}}
    for key,state,step in [('grown',grown,256),('cut',cut,256),('repaired',repaired,512)]:
        provenance['states'][key]={'step':step,'state_sha256':sha(state.astype('<f4').tobytes()),'rgba_sha256':sha(render_rgba(state).tobytes()),'occupied_cells':int((state[...,3]>.1).sum())}
    return {'grown':grown,'cut':cut,'repaired':repaired},provenance

def study_a(frames):
    p=Poster()
    p.plate(frames['grown'],909,304,514)
    p.text((66,63),'A LIVING PATTERN LABORATORY',15,'medium','moss')
    p.text((60,180),'mycelium',126,'serif')
    p.text((68,325),'Neural cellular automata.',28,'sans','muted')
    p.text((66,416),'Small rules.',39,'italic','moss')
    p.text((66,462),'Emergent form.',39,'italic','moss')
    p.line((68,567,1132,567))
    p.text((68,586),'dhanushkrishna4.github.io/Mycelium',14,'sans','muted')
    p.text((1132,586),'32 × 32 CELLS / 16 CHANNELS',13,'mono','muted','rt')
    return p

def study_b(frames):
    p=Poster()
    p.text((64,54),'A LOCAL RULE. A PATTERN THAT CAN RETURN.',16,'medium','moss')
    for cx,key,label in [(240,'grown','01 / GROW'),(600,'cut','02 / CUT'),(960,'repaired','03 / REGROW')]:
        p.plate(frames[key],cx,245,290)
        p.text((cx,408),label,14,'mono','moss','mt')
    p.text((61,461),'mycelium',120,'serif')
    p.text((1134,506),'Neural cellular automata.',25,'sans','muted','rt')
    p.text((1134,550),'Learned growth & regeneration',20,'sans','muted','rt')
    return p

def first_polished(frames):
    p=Poster()
    p.text((600,47),'mycelium',140,'serif',anchor='mt')
    p.text((600,207),'Learned growth. Local rules.',27,'sans','muted','mt')
    for cx,key,label in [(230,'grown','GROW'),(600,'cut','CUT'),(970,'repaired','REGROW')]:
        p.plate(frames[key],cx,418,300)
        p.text((cx,591),label,15,'mono','moss','mt')
    return p

def final_card(frames):
    p=Poster()
    p.text((600,47),'mycelium',140,'serif',anchor='mt')
    p.text((600,207),'Neural cellular automata',28,'sans','muted','mt')
    for cx,key,label in [(230,'grown','Grow'),(600,'cut','Cut'),(970,'repaired','Regrow')]:
        p.plate(frames[key],cx,404,290)
        p.text((cx,572),label,23,'medium','moss','mt')
    for x in (415,785):
        p.line((x-15,404,x+15,404),'muted',1)
        p.line((x+9,399,x+15,404,x+9,409),'muted',1)
    return p

def proofs(image):
    # These are inspection derivatives, never alternative metadata images.
    folder=HERE/'proofs'
    folder.mkdir(exist_ok=True)
    png=PngImagePlugin.PngInfo();png.add(b'sRGB',b'\x00')
    for name,proof in [
        ('600x315',image.resize((600,315),Image.Resampling.LANCZOS)),
        ('300x158',image.resize((300,158),Image.Resampling.LANCZOS)),
        ('crop-16x9',image.crop((40,0,1160,630))),
        ('crop-square',image.crop((285,0,915,630))),
    ]:
        proof.save(folder/(name+'.png'),optimize=True,pnginfo=png)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--studies',action='store_true');args=parser.parse_args()
    frames,provenance=states()
    if args.studies:
        study_a(frames).save(HERE/'studies/01-specimen.png')
        study_b(frames).save(HERE/'studies/02-regeneration.png')
        first_polished(frames).save(HERE/'studies/03-first-polished.png')
    else:
        output=ROOT/'web/og-v1.png'
        image=final_card(frames).save(output)
        proofs(image)
        provenance['output']={'file':'web/og-v1.png','size':list(SIZE),'bytes':output.stat().st_size,'sha256':sha(output.read_bytes()),'mode':'RGB','color_space':'sRGB (rendering intent 0)'}
        provenance['fonts']={kind:{'family':expected[0],'style':expected[1],'face_index':index,'file_sha256':sha(Path(path).read_bytes())} for kind,(path,index,expected) in FONTS.items()}
    (HERE/('study-provenance.json' if args.studies else 'provenance.json')).write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance['states'],indent=2))

if __name__=='__main__': main()
