"""Synthetic metadata-only tests; no model, production samples, or remote calls."""
import copy,hashlib,json,re,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from audit_core import PDES,METHODS,VIEWS
from analyze_new import analyze
from generate_tables_new import build

def synthetic_records():
    records={}
    for pde in PDES:
        for method in METHODS:
            for i,v in enumerate(VIEWS):
                identity='/'.join((pde,method,v))
                blocks={}
                for j,w in enumerate(VIEWS):
                    mean=float(1+i+j)/10
                    blocks[w]={'joint':{'mean':mean,'std':.125,'n':200,'ddof':1},
                      'horizons':[{'horizon':h,'mean':mean*h,'std':.25,'n':200,'ddof':1} for h in (1,2,3)]}
                records[identity]={'identity':identity,'pde':pde,'method':method,'train_view':v,
                   'actual_epochs':500,'training_cohort':'original500','blocks':blocks}
    return records

class Tables(unittest.TestCase):
    def setUp(self):self.records=synthetic_records();self.stats=analyze(self.records)
    def stage(self,root):
        cut=root/'synthetic-cut';cut.mkdir()
        (cut/'index.json').write_text(json.dumps({'records':self.records}))
        (cut/'statistics.json').write_text(json.dumps(self.stats))
        return cut
    def test_all_cells_and_precision(self):
        with tempfile.TemporaryDirectory(prefix='pdeobs-meta-table-') as tmp:
            root=Path(tmp);cut=self.stage(root);out=root/'tables';build(cut,out)
            report=json.loads((out/'generation-audit.json').read_text())
            self.assertEqual(report['cells_written_once'],3969)
            matrices=(out/'matrices.tex').read_text();self.assertEqual(len(re.findall(r'\d+\.\d{4} \\pm \d+\.\d{4}',matrices)),3969)
            self.assertEqual(matrices.count('\\begin{table}'),49)
            self.assertNotIn('resizebox',matrices);self.assertNotIn('scriptsize',matrices)
            self.assertIn('0.1000 \\pm 0.1250',matrices)
            main=(out/'main-summary.tex').read_text();self.assertEqual(len(re.findall(r'\d+\.\d{2} \\pm \d+\.\d{2}',main)),14)
            self.assertIn('63/504',main)
    def test_no_historical_branch(self):
        self.stats['strict_rescoring']=False
        with tempfile.TemporaryDirectory(prefix='pdeobs-meta-table-') as tmp:
            root=Path(tmp);cut=self.stage(root)
            with self.assertRaises(ValueError):build(cut,root/'out')
    def test_report_classifications_use_real_category_names(self):
        with tempfile.TemporaryDirectory(prefix='pdeobs-meta-table-') as tmp:
            root=Path(tmp);cut=self.stage(root)
            index={'records':self.records,'audit':{'counts':{'old_strict':3646,'minor_le_1pct':314,'difference_gt_1pct':9}}}
            (cut/'index.json').write_text(json.dumps(index))
            out=root/'out';build(cut,out)
            text=(out/'numbers.tex').read_text()
            self.assertIn(r'\newcommand{\NewStrictCount}{3,646}',text)
            self.assertIn(r'\newcommand{\NewMinorCount}{314}',text)
            self.assertIn(r'\newcommand{\NewLargeCount}{9}',text)
    def test_outcomes_not_hardcoded(self):
        x=copy.deepcopy(self.records)
        for row in x.values():
            for view,b in row['blocks'].items():b['joint']['mean']=10. if view==row['train_view'] else 1.
        result=analyze(x);self.assertEqual(result['stats']['pairs_C_gt_D'],0)
        self.assertEqual(result['stats']['positive_destination'],0)
    def test_joint_not_horizons(self):
        x=copy.deepcopy(self.records)
        for row in x.values():
            for block in row['blocks'].values():
                for h in block['horizons']:h['mean']*=100
        result=analyze(x)
        np.testing.assert_array_equal(result['arrays']['E'],self.stats['arrays']['E'])
    def test_fixed_subset_not_score_selected(self):
        self.assertEqual(self.stats['fixed_subset']['pairs'],13)
        self.assertEqual(sum(r['fixed_original500'] for r in self.stats['pairs']),13)
    def test_overwrite_refused(self):
        with tempfile.TemporaryDirectory(prefix='pdeobs-meta-table-') as tmp:
            root=Path(tmp);cut=self.stage(root);out=root/'out';out.mkdir()
            with self.assertRaises(FileExistsError):build(cut,out)

if __name__=='__main__':unittest.main()
