import unittest
import numpy as np
from study import split_name,canonical,quality_upper,eb_upper,select_policy,policies


class Checks(unittest.TestCase):
    def test_duplicate_isolation(self):
        a=canonical('hello  world\n prompt');b=canonical('hello world prompt')
        self.assertEqual(split_name(a,'schema'),split_name(b,'schema'))
        self.assertEqual(split_name('schema','schema'),'train')

    def test_zero_empirical_loss_not_zero_bound(self):
        self.assertGreater(quality_upper(np.zeros(100)),0)

    def test_reference_option_and_tune_feasibility(self):
        y=np.array([[1,0],[0,1],[1,1],[1,0.]])
        c=np.tile([1.,.1],(4,1))
        preds={'global_mean':{k:np.tile(y.mean(0),(4,1)) for k in ['tune','cal','pilot']}}
        candidates=policies(preds,c.mean(0),2)
        _,ix,risk=select_policy(candidates,y,c,0)
        self.assertLessEqual(risk,.02)
        self.assertTrue(np.all(candidates[ix]['picks']['tune']==0))

    def test_cost_bound_not_empirical_mean(self):
        self.assertGreater(eb_upper(np.repeat(-.001,1000)),-.001)

    def test_fixed_mixture_stable_input_only(self):
        texts={part:['a','b','c','d'] for part in ['tune','cal','pilot']}
        preds={'global_mean':{part:np.zeros((4,2)) for part in texts}}
        a=policies(preds,np.array([1.,.1]),2,texts,0)
        reversed_text={part:list(reversed(v)) for part,v in texts.items()}
        b=policies(preds,np.array([1.,.1]),2,reversed_text,0)
        ma=[r for r in a if r['family']=='fixed_mixture'];mb=[r for r in b if r['family']=='fixed_mixture']
        self.assertEqual(len(ma),6)
        for first,second in zip(ma,mb):
            np.testing.assert_array_equal(first['picks']['pilot'],second['picks']['pilot'][::-1])


if __name__=='__main__':unittest.main()
