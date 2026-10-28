"""Download the official public RouterBench zero-shot matrix, with provenance.
No paid inference, credentials or benchmark final-label analysis is performed.
"""
import hashlib
import json
from pathlib import Path
import urllib.request
import argparse


def fetch(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    revision='784021482c3f320c6619ed4b3bb3b41a21424fcb'
    expected='ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d'
    url=f'https://huggingface.co/datasets/withmartian/routerbench/resolve/{revision}/routerbench_0shot.pkl'
    target=destination/'routerbench_0shot.pkl'
    if not target.exists():
        print('Downloading official public zero-shot outcomes',flush=True)
        urllib.request.urlretrieve(url,target)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
        raise ValueError('Dataset checksum mismatch; refusing unverified input')
    manifest={'dataset':'withmartian/routerbench','revision':revision,'source_url':url,
       'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
       'license_metadata':None,
       'scope':'Historical cached inference and benchmark costs; not current prices or measured serving latency.'}
    (destination/'source.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',default='work/lifecycle-routing')
    fetch(p.parse_args().destination)
