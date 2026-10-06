"""Synthetic interface example. It is not a MAGFiLO observation or experiment."""
from pathlib import Path
import argparse,sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from filament_structure.frozen import score_one
from filament_structure.io import dump,output
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);a=p.parse_args()
    mask=np.zeros((32,32),bool);mask[10,4:25]=True
    references=[dict(sample='synthetic_copy',references=[dict(annotation_id='synthetic_reference',points=[[4.,10.],[24.,10.]])])]
    dump(output(a.output_root)/'synthetic_result.json',score_one(mask,references))
