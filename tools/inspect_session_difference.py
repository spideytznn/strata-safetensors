"""Read-only diagnostic for trusted v1 Strata session snapshots.

Parses length-delimited fields; this is not the engine's hash-validating reader.
Compares main-model state only; MTP state differs between MTP-on/off runs.
"""
import argparse
import json
import struct
from pathlib import Path
import numpy as np

def read(path):
    raw=path.read_bytes()
    assert raw[:8]==b'STRSESS\x01' and raw[-8:]==b'STRSEND\x01'
    assert struct.unpack_from('<II',raw,8)==(1,64)
    assert struct.unpack_from('<Q',raw,32)[0]==len(raw)-80
    pos=64
    def scalar():
        nonlocal pos
        v=struct.unpack_from('<Q',raw,pos)[0];pos+=8;return v
    def vector(width=1):
        nonlocal pos
        size=scalar()*width
        assert size <= len(raw)-pos-16
        result=memoryview(raw)[pos:pos+size];pos+=size;return result
    def checkpoint():
        nonlocal pos
        result={'ids':vector(4)}
        images=scalar();assert images < 1000000
        pos+=images*16
        for name in ('gdn','ple','tails','dead','block_pos'):result[name]=vector()
        result['used']=scalar();return result
    geometry=[scalar() for _ in range(18)]
    bounds=[scalar() for _ in range(3)]
    live=checkpoint()
    checks=[checkpoint() for _ in range(scalar())]
    kv=[]
    for _ in range(scalar()):
        item={'geometry':[scalar() for _ in range(7)]}
        for name in ('k','v','k_scale','v_scale','pooled'):item[name]=vector()
        kv.append(item)
    assert pos==len(raw)-16,(pos,len(raw))
    return dict(geometry=geometry,bounds=bounds,live=live,checkpoints=checks,kv=kv)

def compare(a,b,dtype='<f4'):
    assert len(a)==len(b),(len(a),len(b))
    x=np.frombuffer(a,dtype=dtype);y=np.frombuffer(b,dtype=dtype)
    different=np.flatnonzero(x.view(f'u{x.dtype.itemsize}')!=y.view(f'u{y.dtype.itemsize}'))
    result={'elements':x.size,'different':different.size}
    if different.size:
        result.update(first=int(different[0]),last=int(different[-1]))
        if dtype=='<u2':
            x=(x.astype(np.uint32)<<16).view(np.float32);y=(y.astype(np.uint32)<<16).view(np.float32)
        xd=x.astype(np.float64);yd=y.astype(np.float64);d=xd-yd
        result.update(max_abs=float(np.max(np.abs(d))),relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(xd),1e-300)),
                      nonfinite=int(np.count_nonzero(~np.isfinite(xd)|~np.isfinite(yd))),
                      first_values=[float(xd[different[0]]),float(yd[different[0]])])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('a',type=Path);p.add_argument('b',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    a,b=read(args.a),read(args.b)
    assert a['geometry']==b['geometry'] and a['bounds']==b['bounds']
    assert a['geometry']==[2560,48,4,128,16,48,4,10240,6144,24,2,256,4,128,4,320,512,640], 'unsupported diagnostic geometry'
    report={'a':str(args.a),'b':str(args.b),'geometry':a['geometry'],'live':{},'checkpoints':[],'kv':[]}
    for name in ('ids','gdn','ple','tails','dead','block_pos'):
        report['live'][name]=compare(a['live'][name],b['live'][name],'<i4' if name in ('ids','block_pos') else '<f4')
    # This diagnostic targets the 48-layer Qwen geometry (36 GDN, 12 QSA).
    size=len(a['live']['gdn'])//36
    report['gdn_layers']=[{'layer':i+i//3,**compare(a['live']['gdn'][i*size:(i+1)*size],b['live']['gdn'][i*size:(i+1)*size])} for i in range(36)]
    for ca,cb in zip(a['checkpoints'],b['checkpoints']):
        report['checkpoints'].append({name:compare(ca[name],cb[name],'<i4' if name in ('ids','block_pos') else '<f4') for name in ('ids','gdn','ple','tails','dead','block_pos')})
    for i,(ka,kb) in enumerate(zip(a['kv'][:12],b['kv'][:12])):
        assert ka['geometry']==kb['geometry']
        fmt,cells,heads,dim,page,pooled,idx=ka['geometry']
        assert fmt==64 and cells%page==0,'this diagnostic supports BF16 KV only'
        live=len(a['live']['ids'])//4
        row={'layer':i*4+3,'geometry':ka['geometry']}
        for name in ('k','v'):
            def ordered(data):
                return np.frombuffer(data,dtype='<u2').reshape(cells//page,heads,page,dim).transpose(0,2,1,3).reshape(cells,heads*dim)
            xa,xb=ordered(ka[name]),ordered(kb[name])
            row[name]=compare(xa[:live].tobytes(),xb[:live].tobytes(),'<u2')
            row[name]['different_positions']=np.flatnonzero(np.any(xa[:live]!=xb[:live],axis=1)).tolist()
            row[name+'_stale']=compare(xa[live:].tobytes(),xb[live:].tobytes(),'<u2')
        row['pooled']=compare(ka['pooled'][:live//page*idx*4],kb['pooled'][:live//page*idx*4])
        report['kv'].append(row)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'live':report['live'],'gdn_layers':report['gdn_layers'],'checkpoints':report['checkpoints']},indent=2))

if __name__=='__main__':main()
