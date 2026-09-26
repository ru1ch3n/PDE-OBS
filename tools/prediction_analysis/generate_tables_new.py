"""Complete, programmatically generated new-score tables; no legacy fallback."""
import argparse,json
from pathlib import Path
from analyze_new import PDES,METHODS,VIEWS,LABELS,format_pair
PN={'darcy':'Darcy','poisson':'Poisson','helmholtz':'Helmholtz','heat':'Heat','reaction_diffusion':'Reaction--diffusion','burgers':'Burgers','navier_stokes':'Navier--Stokes'}
MN={'ufno_2d':'U-FNO','fno':'FNO','cno':'CNO','deeponet':'DeepONet','gnot':'GNOT','transolver':'Transolver','pino':'PINO'}
COHORTS={'original500':'O500','original500_recovered':'R500','budget200_min120':'M120','fixed200_recovered':'F200','interrupted_or_recovered':'INT','budget200_patience10_no_floor':'P10'}
def pm(s,places=4):return '$'+format_pair(s['mean'],s['std'],places)+'$'
def row(*xs):return ' & '.join(map(str,xs))+r' \\'+'\n'
def tab(cols,head,body):return '\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}'+cols+'@{}}\n\\toprule\n'+head+'\\midrule\n'+body+'\\bottomrule\n\\end{tabular*}\n'
def table(caption,label,content,placement='H'):return '\\begin{table}['+placement+']\n\\centering\n\\caption{'+caption+'}\\label{'+label+'}\n'+content+'\\end{table}\n'
def write(path,value):
    with path.open('x',encoding='utf-8') as f:f.write(value)
def build(cut,out):
    s=json.loads((cut/'statistics.json').read_text());index=json.loads((cut/'index.json').read_text());records=index['records']
    if s['blocks']!=3969 or len(records)!=441 or not s['strict_rescoring']:raise ValueError('complete strict cut required')
    out.mkdir(parents=True,exist_ok=False)
    body=''
    for i,r in enumerate(s['main']):
        if i in (0,3):body+='\\rowcolor{PDEGray}\\multicolumn{4}{l}{\\textcolor{PDEInk}{'+('Stationary solution recovery' if i==0 else 'Temporal state forecasting')+r'}} \\'+'\n'
        body+=row(PN[r['pde']],pm(r['matched_cells'],2),pm(r['cross_cells'],2),f"{r['median_C_over_D']:.2f}")
    header=row('','\\multicolumn{2}{c}{Relative-$L^2$ test error}','Transfer ratio')+'\\cmidrule(lr){2-3}\\cmidrule(l){4-4}\n'+row('PDE','Matched','Cross-pattern','Median $C/D$')
    write(out/'main-summary.tex',table('Absolute error and transfer sensitivity from new predictions. Matched/cross-pattern entries are mean $\\pm$ sample SD over 63/504 cell means per PDE (seven methods); the ratio is the median of seven pairwise $C/D$ values. These are not error percentages, seed uncertainty, or an architecture ranking.','tab:main-summary',tab('lrrr',header,body),'!t'))
    body=''
    for name,n in s['cohorts'].items():body+=row(COHORTS[name],name.replace('_',r'\_'),n)
    write(out/'cohorts.tex',table('Retained training-provenance groups. Codes identify histories, not scientific quality; every retained model contributes all nine test views.','tab:result-cohorts',tab('llr',row('Code','Recorded cohort','Models'),body)))
    eq=s['stats']['equal_count'];de=s['stats']['density']
    body=''
    for r in eq:body+=row(f"{r['count_group']:,}",r['task'].capitalize(),f"{r['positive']}/{r['n']}",f"{r['median_ratio']:.4f}")
    text=table('Destination-referenced equal-count contrasts. Positive means $E_{v,w}>E_{w,w}$; ratios and counts have no artificial standard deviations.','tab:equal-count',tab('llrr',row('$K$','Task','Positive / total','Median ratio'),body))
    body=''.join(row(r['train_view'],r['test_view'],f"{r['lower']}/{r['n']}",f"{r['median_ratio']:.4f}") for r in de)
    text+=table('Density responses of each fixed checkpoint relative to its R50 test error. Lower counts indicate a ratio below one; masks are independently generated, not nested.','tab:density-directions',tab('llrr',row('Training','Test','Lower / total','Median ratio'),body))
    write(out/'directional-density.tex',text)
    text=''
    for pde in PDES:
        part=[r for r in s['pairs'] if r['pde']==pde]
        body=''
        for r in part:
            if r['fixed_original500']:body+='\\rowcolor{PDEFocus}\n'
            body+=row(MN[r['method']],pm(r['D']),pm(r['C']),f"{r['C_over_D']:.4f}")
        content=tab('lrrr',row('','\\multicolumn{2}{c}{Cell-mean relative-$L^2$ error}','')+'\\cmidrule(lr){2-3}\n'+row('Method','Matched $D$','Cross-pattern $C$','$C/D$'),body)
        body=''.join(row(MN[r['method']],f"{r['median_off_error']:.4f}",f"{r['p90_off_error']:.4f}",f"{r['max_off_error']:.4f}") for r in part)
        content+='\\par\\medskip\n'+tab('lrrr',row('Method','Cross median','Cross 90th pct.','Cross maximum'),body)
        text+=table(PN[pde]+': all method summaries. $D$/$C$ use mean $\\pm$ sample SD of 9/72 cell means, not 200 records or multiple seeds. Blue identifies the fixed original500 subset, not a best-method claim; no cross-method ranking is implied.', 'tab:pair-'+pde,content)+'\\clearpage\n'
    write(out/'pairs.tex',text)
    text=''
    for pde in PDES[3:]:
        body=''
        for method in METHODS:
            body+='\\rowcolor{PDEGray}\\multicolumn{3}{l}{'+MN[method]+r'} \\'+'\n'
            for r in s['horizons']:
                if r['pde']==pde and r['method']==method:body+=row(r['horizon'],pm(r['D']),pm(r['C']))
        text+=table(PN[pde]+': all forecast horizons. Entries are mean $\\pm$ sample SD over 9 matched or 72 cross-pattern cell means. These are auxiliary per-frame scores; their average is not the joint rollout score.','tab:horizon-'+pde,tab('lrr',row('','\\multicolumn{2}{c}{Relative-$L^2$ error}')+'\\cmidrule(lr){2-3}\n'+row('Horizon','Matched $D_h$','Cross-pattern $C_h$'),body))+'\\clearpage\n'
    write(out/'horizons.tex',text)
    text='';seen=set()
    for pde in PDES:
        for method in METHODS:
            content=''
            for start in (0,3,6):
                header=row('','\\multicolumn{3}{c}{Test observation pattern}')+'\\cmidrule(lr){2-4}\n'+row('Train',*LABELS[start:start+3])
                body=''
                for i,v in enumerate(VIEWS):
                    r=records[f'{pde}/{method}/{v}'];cells=[]
                    for w in VIEWS[start:start+3]:
                        key=(r['identity'],w)
                        if key in seen:raise ValueError('duplicate table cell')
                        seen.add(key);cell=pm(r['blocks'][w]['joint'])
                        if v==w:cell='\\cellcolor{PDEFocus}'+cell
                        cells.append(cell)
                    body+=row(LABELS[i],*cells)
                content+=tab('lrrr',header,body)+'\\par\\medskip\n'
            budget=', '.join(LABELS[i]+': '+str(records[f'{pde}/{method}/{v}']['actual_epochs'])+'/'+COHORTS[records[f'{pde}/{method}/{v}']['training_cohort']] for i,v in enumerate(VIEWS))
            caption=PN[pde]+' / '+MN[method]+': all 81 test cells, split into three column panels. Each value is mean $\\pm$ sample SD across the same 200 test records (four decimals). Blue marks matched observations, not optimality. Epoch/cohort by training view: '+budget+'.'
            text+=table(caption,'tab:matrix-'+pde+'-'+method.replace('_','-'),content)+'\\clearpage\n'
    if len(seen)!=3969:raise ValueError('full table coverage')
    write(out/'matrices.tex',text)
    sub=s['fixed_subset'];body=''
    for r in s['pairs']:
        if r['fixed_original500']:body+=row(PN[r['pde']],MN[r['method']],pm(r['D']),pm(r['C']))
    write(out/'fixed117.tex',table('The fixed 13 configuration-matched original500 pairs, using only new predictions. All nine training views reached 500 epochs. $D$/$C$ standard deviations are across 9/72 cell means; selection does not use these scores.','tab:fixed117-pairs',tab('llrr',row('','', '\\multicolumn{2}{c}{Relative-$L^2$ error}')+'\\cmidrule(lr){3-4}\n'+row('PDE','Method','Matched $D$','Cross-pattern $C$'),body)))
    full=s['stats'];sub=s['fixed_subset']
    body=row('Pairs',49,13)+row('Pairs with $C>D$',f"{full['pairs_C_gt_D']}/49",f"{sub['pairs_C_gt_D']}/13")
    body+=row('Median $C/D$',f"{full['median_C_over_D']:.4f}",f"{sub['median_C_over_D']:.4f}")
    for count,label in ((8192,'Positive layout directions, group I'),(8284,'Positive layout directions, group II')):
        values=[]
        for group in (full,sub):
            values.append(str(sum(v['positive'] for v in group['equal_count'] if v['count_group']==count))+'/'+str(sum(v['n'] for v in group['equal_count'] if v['count_group']==count)))
        body+=row(label,*values)
    for view in ('R50','R65','R80'):
        values=[]
        for group in (full,sub):
            r=next(v for v in group['density'] if v['train_view']==view and v['test_view']=='R80');values.append(f"{r['lower']}/{r['n']}")
        body+=row('R80 improves over R50, '+view+'-trained',*values)
    write(out/'subset-summary.tex',table('Sensitivity to fixed original500 training provenance. Counts and medians summarize dependent comparisons, not independent experimental replications. The metadata-defined subset is unchanged.','tab:original500-sensitivity',tab('lrr',row('Statistic','Full grid','Fixed subset'),body)))
    macros={'NewCDPositive':str(full['pairs_C_gt_D']),'NewCDMedian':f"{full['median_C_over_D']:.2f}",
      'NewSubsetPositive':str(sub['pairs_C_gt_D']),'NewSubsetCDMedian':f"{sub['median_C_over_D']:.2f}",
      'NewDestinationPositive':f"{full['positive_destination']:,}"}
    if 'audit' in index:
        for name,key in (('NewStrictCount','old_strict'),('NewMinorCount','minor_le_1pct'),('NewLargeCount','difference_gt_1pct')):
            macros[name]=f"{index['audit']['counts'].get(key,0):,}"
    for count,name in ((8192,'One'),(8284,'Two')):
        positive=sum(v['positive'] for v in full['equal_count'] if v['count_group']==count);n=sum(v['n'] for v in full['equal_count'] if v['count_group']==count)
        macros['NewLayout'+name+'Count']=str(positive)+'/'+str(n);macros['NewLayout'+name+'Percent']=f'{100*positive/n:.1f}'
    for view,name in (('R50','Fifty'),('R65','SixtyFive'),('R80','Eighty')):
        r=next(v for v in full['density'] if v['train_view']==view and v['test_view']=='R80')
        macros['NewDensity'+name+'Count']=str(r['lower'])+'/'+str(r['n']);macros['NewDensity'+name+'Median']=f"{r['median_ratio']:.2f}"
    r=next(v for v in sub['density'] if v['train_view']=='R50' and v['test_view']=='R80');macros['NewSubsetDensityFiftyMedian']=f"{r['median_ratio']:.2f}"
    write(out/'numbers.tex','% Generated exclusively from the complete audited new-score cut.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    write(out/'generation-audit.json',json.dumps({'cells_written_once':len(seen),'pair_summaries':49,'horizon_summaries':84,'fixed_models':117,'main_precision':2,'appendix_precision':4,'ddof':1},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cut',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.cut,a.out)
