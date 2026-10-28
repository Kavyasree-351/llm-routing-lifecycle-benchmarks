"""Read a pinned official matrix using an explicit constructor allowlist."""
import importlib
import pickle
import hashlib
from pathlib import Path

SHA256='ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d'
ALLOWED={('pandas.core.frame','DataFrame'),('pandas.core.internals.managers','BlockManager'),
 ('pandas._libs.internals','_unpickle_block'),('numpy.core.multiarray','_reconstruct'),
 ('numpy.core.numeric','_frombuffer'),('numpy','ndarray'),('numpy','dtype'),
 ('builtins','slice'),('pandas.core.indexes.base','_new_Index'),
 ('pandas.core.indexes.base','Index'),('pandas.core.indexes.range','RangeIndex')}


class Reader(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name) not in ALLOWED:raise pickle.UnpicklingError(f'Unapproved constructor {module}.{name}')
        return getattr(importlib.import_module(module),name)


def load(path):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Dataset checksum mismatch; review provenance before changing reader')
    with path.open('rb') as f:return Reader(f).load()


if __name__=='__main__':
    df=load('work/lifecycle-routing/routerbench_0shot.pkl')
    print('shape',df.shape)
    print('columns',df.columns.tolist())
    print('dtypes',df.dtypes.astype(str).to_dict())
    for col in ['sample_id','eval_name','dataset','prompt','query','question']:
        if col in df:print(col,'example type',type(df[col].iloc[0]).__name__)
