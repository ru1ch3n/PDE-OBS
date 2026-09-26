import copy,json,unittest
from pathlib import Path
import numpy as np
from analyze_new import analyze,from_archive,stats,format_pair
ROOT=Path(__file__).resolve().parents[2]
class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=ROOT/'source-results' if (ROOT/'source-results').is_dir() else ROOT
        cls.records=from_archive(source/'results/archived_campaign/index.json')
        cls.result=analyze(cls.records)
    def test_historical_counts(self):
        x=self.result['stats'];self.assertEqual((x['pairs_C_gt_D'],x['positive_destination'],x['destination_n']),(49,3281,3528))
        self.assertEqual([r['positive'] for r in x['equal_count']],[238,307,122,154]);self.assertEqual([r['lower'] for r in x['density']],[15,13,48,37,49,49])
    def test_historical_ratios(self):
        self.assertAlmostEqual(self.result['stats']['median_C_over_D'],10.459044546923382,places=12)
        self.assertAlmostEqual(self.result['fixed_subset']['median_C_over_D'],10.737852918050656,places=12)
    def test_fixed_counts(self):
        x=self.result['fixed_subset'];self.assertEqual((x['pairs'],x['pairs_C_gt_D']),(13,13))
        self.assertEqual(sum(v['positive'] for v in x['equal_count'] if v['count_group']==8192),151)
        self.assertEqual(sum(v['positive'] for v in x['equal_count'] if v['count_group']==8284),76)
        self.assertEqual([v['lower'] for v in x['density'] if v['test_view']=='R80'],[0,8,13])
    def test_destination_not_source(self):
        x=self.result['transfers'][0];self.assertAlmostEqual(x['ratio'],x['error']/x['destination_reference'])
    def test_pair_sd_population(self):
        x=self.result['pairs'][0];self.assertEqual((x['D']['n'],x['C']['n']),(9,72))
        e=np.array(self.result['arrays']['E'][0]);self.assertAlmostEqual(x['D']['std'],np.diag(e).std(ddof=1))
    def test_main_sd_population(self):self.assertEqual((self.result['main'][0]['matched_cells']['n'],self.result['main'][0]['cross_cells']['n']),(63,504))
    def test_horizons(self):self.assertEqual(len(self.result['horizons']),84)
    def test_missing_row(self):
        x=dict(self.records);x.pop(next(iter(x)))
        with self.assertRaises(ValueError):analyze(x)
    def test_missing_view(self):
        x=copy.deepcopy(self.records);x[next(iter(x))]['blocks'].pop('random_50pct')
        with self.assertRaises(ValueError):analyze(x)
    def test_nonfinite(self):
        x=copy.deepcopy(self.records);x[next(iter(x))]['blocks']['random_50pct']['joint']['mean']=float('nan')
        with self.assertRaises(ValueError):analyze(x)
    def test_negative(self):
        with self.assertRaises(ValueError):format_pair(1,-1)
    def test_same_precision(self):
        self.assertEqual(format_pair(.12,.03,2),'0.12 \\pm 0.03');self.assertEqual(format_pair(.12,.03,4),'0.1200 \\pm 0.0300')
    def test_scale(self):self.assertEqual(format_pair(.001,.0005,4,.001),'1.0000 \\pm 0.5000')
    def test_ratio_no_std(self):self.assertIsInstance(self.result['pairs'][0]['C_over_D'],float)
    def test_sample_sd(self):self.assertAlmostEqual(stats([1,3],2)['std'],2**.5)
    def test_bad_population(self):
        with self.assertRaises(ValueError):stats([1,3],3)
if __name__=='__main__':unittest.main()
