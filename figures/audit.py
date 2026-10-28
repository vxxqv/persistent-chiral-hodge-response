"""Export inspection and a vector-preserving figure review document."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pymupdf as fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import resvg_py
from colorspacious import cspace_convert


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--figures',type=Path,required=True);ap.add_argument('--qa',type=Path,required=True)
    args=ap.parse_args();folder=args.figures;qa=args.qa;qa.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((folder/'manifest.json').read_text())
    report={};review=fitz.open();arial='C:/Windows/Fonts/arial.ttf'
    for key,entry in sorted(manifest['figures'].items()):
        path=folder/f'{key}.pdf';pdf=fitz.open(path);page=pdf[0]
        spans=[s for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for s in line['spans'] if s['text'].strip()]
        outside=[s['text'] for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
        lines=[(line['spans'][0]['text'],fitz.Rect(line['bbox']))
               for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines']]
        collisions=[];math_boxes=[]
        for i,(t,a) in enumerate(lines):
            for u,b in lines[i+1:]:
                intersection=a&b
                if not intersection.is_empty and intersection.width>1 and intersection.height>1 and intersection.get_area()>6:
                    pair={'first':t,'second':u,'intersection_pt':list(intersection)}
                    if key.startswith(('Figure_01_','Figure_02_')) and (t,u)==('λ','2'):
                        math_boxes.append(pair)
                    else:collisions.append(pair)
        # Mathematical superscripts are smaller by design. Ordinary labels are
        # checked separately from those glyph runs.
        ordinary_small=[s['text'] for s in spans if s['size']<7.9 and len(s['text'].strip())>3
                        and not all(c.isdigit() or c in '.-−+ ' for c in s['text'])]
        svg=ET.parse(folder/f'{key}.svg');ns={'svg':'http://www.w3.org/2000/svg'}
        raster_layers=len(svg.findall('.//svg:image',ns))
        height=round(page.rect.height*672/page.rect.width)
        png=Image.open(folder/f'{key}.png').convert('RGB')
        dpi=png.info.get('dpi')
        preview=png.resize((672,height),Image.Resampling.LANCZOS)
        pix=page.get_pixmap(matrix=fitz.Matrix(96/72,96/72))
        pdf_render=Image.frombytes('RGB',(pix.width,pix.height),pix.samples).resize((672,height))
        svg_bytes=resvg_py.svg_to_bytes(svg_path=str((folder/f'{key}.svg').resolve()),width=672,dpi=96)
        (qa/f'{key}_svg.png').write_bytes(svg_bytes)
        svg_render=Image.open(qa/f'{key}_svg.png').convert('RGB').resize((672,height))
        pdf_render.save(qa/f'{key}_pdf.png')
        canvas=Image.new('RGB',(2016,height+24),'white');draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype(arial,13)
        for i,(label,im) in enumerate([('PNG',preview),('PDF',pdf_render),('SVG',svg_render)]):
            canvas.paste(im,(i*672,24));draw.text((i*672+12,5),label,font=font,fill='#404040')
        canvas.save(qa/f'{key}_formats.png')
        ImageOps.grayscale(preview).save(qa/f'{key}_grayscale.png')
        rgb=np.asarray(preview,dtype=float)/255
        cvd=cspace_convert(rgb,{'name':'sRGB1+CVD','cvd_type':'deuteranomaly','severity':100},'sRGB1')
        cvd_image=Image.fromarray(np.uint8(np.clip(cvd,0,1)*255))
        cvd_image.save(qa/f'{key}_deuteranopia.png')
        access=Image.new('RGB',(1344,height+24),'white');access.paste(ImageOps.grayscale(preview).convert('RGB'),(0,24));access.paste(cvd_image,(672,24))
        adraw=ImageDraw.Draw(access);adraw.text((12,5),'Grayscale',font=font,fill=0);adraw.text((684,5),'Deuteranopia simulation',font=font,fill=0)
        access.save(qa/f'{key}_accessibility.png')
        # Show original vector figure at exactly its exported width, then place
        # the complete caption below. No screenshot replaces the vector plots.
        target=review.new_page(width=595.276,height=841.89)
        width=178/25.4*72;left=(target.rect.width-width)/2;top=27
        rect=fitz.Rect(left,top,left+width,top+page.rect.height)
        target.show_pdf_page(rect,pdf,0)
        target.insert_font(fontname='arial',fontfile=arial)
        caption=(folder/f'{key}.txt').read_text(encoding='utf-8')
        available=fitz.Rect(left,rect.y1+15,left+width,target.rect.height-24)
        remaining=target.insert_textbox(available,caption,fontname='arial',fontsize=9.2,lineheight=1.34,color=(.12,.12,.12))
        if remaining<0:raise ValueError(f'Caption does not fit: {key}')
        report[key]={'pdf_size_mm':[round(page.rect.width*25.4/72,3),round(page.rect.height*25.4/72,3)],
                     'outside_text':outside,'small_ordinary_text':ordinary_small,
                     'unexplained_text_collisions':collisions,'expected_math_glyph_boxes':math_boxes,
                     'embedded_3d_raster_layers':raster_layers,'png_pixels':png.size,'png_dpi':dpi,
                     'formats_sha256':{ext:sha(folder/f'{key}.{ext}') for ext in ['png','pdf','svg']},
                     'visual_review':'pending','source_sha256':sha(folder/'source_data'/f'{key}.json')}
        pdf.close()
    review.set_metadata({'title':'Persistent chiral Hodge response: figures and legends','author':'Vivaan Patni',
                         'creator':'','producer':''})
    review.save(folder/'PCHR_Figures_and_Legends.pdf',garbage=4,deflate=True)
    (qa/'export_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:{'outside':v['outside_text'],'collisions':v['unexplained_text_collisions'],
                         'small':v['small_ordinary_text']} for k,v in report.items()},ensure_ascii=True))


if __name__=='__main__':main()
