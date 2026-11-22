"""Train-window factor re-evaluation and deterministic correlation pruning."""
import json
import numpy as np
import pandas as pd
from src.factors.candidates import Candidate
from src.research.labels import LabelSpec
from .ranking import industry_percentiles


def library_candidates(path,dataset_id,context_id=None):
    data=json.loads(path.read_text())
    if data.get('version')!='factor-set-v1' or data.get('task')!='B':
        raise ValueError('Expected B authoritative factor library')
    result=[];seen=set()
    for factor in data['factors'].values():
        c=factor['candidate']
        if c['task']!='B' or c['scope']!='stock' or c['source'] not in ('alpha158','llm'):continue
        evidence=[]
        for obs in factor['observations'].values():
            spec=(obs.get('report') or {}).get('spec',{})
            if (spec.get('dataset_id')==dataset_id and spec.get('split',{}).get('name')=='train'
                    and spec.get('label_definition')==LabelSpec().definition('B')
                    and (context_id is None or obs.get('context_id')==context_id)):
                evidence.append(obs)
        if not evidence:continue
        # Exact formula duplicates have no additional information.
        expression=''.join(c['expression'].split())
        if expression in seen:continue
        seen.add(expression);result.append(Candidate(**c))
    if not result:raise ValueError('No matching train-context candidate definitions')
    return sorted(result,key=lambda c:c.candidate_id)


def select_rank_features(frame,candidates,config):
    """Only this purged fitting window enters scores, signs, or correlations."""
    rows=frame.reset_index(drop=True)
    target=rows.rank_target
    features={}; records=[]
    for c in candidates:
        values=rows[c.candidate_id]
        coverage=float(values.notna().mean())
        # Pairwise-valid within-industry ranks avoid counting missing factor rows as information.
        ranks=industry_percentiles(values,rows.date,rows.industry)
        y=industry_percentiles(rows.target.where(ranks.notna()),rows.date,rows.industry)
        valid=ranks.notna()&y.notna()
        a=ranks.where(valid);b=y.where(valid)
        a=a-a.groupby(rows.date).transform('mean');b=b-b.groupby(rows.date).transform('mean')
        numerator=(a*b).groupby(rows.date).sum()
        denominator=np.sqrt(a.pow(2).groupby(rows.date).sum()*b.pow(2).groupby(rows.date).sum())
        ic=numerator/denominator.where(denominator>1e-14)
        sign=1 if ic.mean()>=0 else -1
        yearly=ic.groupby(ic.index.year).mean()*sign
        score=float(yearly.median()) if yearly.notna().all() else np.nan
        eligible=(coverage>=config.min_coverage and ic.notna().sum()>=config.min_factor_days
                  and np.isfinite(score) and score>0 and (yearly>0).mean()>=.5)
        records.append({'candidate_id':c.candidate_id,'name':c.name,'source':c.source,
                        'coverage':coverage,'valid_days':int(ic.notna().sum()),'orientation':sign,
                        'score':score if np.isfinite(score) else None,
                        'annual_rank_ic':{str(k):float(v) if np.isfinite(v) else None for k,v in yearly.items()},
                        'eligible':bool(eligible),'selected':False,'excluded_by_correlation':None})
        if eligible:features[c.candidate_id]=ranks
    ordered=sorted([r for r in records if r['eligible']],key=lambda r:(-r['score'],r['candidate_id']))
    # All fitting rows, not validation data and not a correlation sample.
    corr=pd.DataFrame(features).corr(min_periods=config.min_factor_days).abs()
    selected=[]
    for row in ordered:
        conflicts=[(k,float(corr.loc[row['candidate_id'],k])) for k in selected
                   if pd.notna(corr.loc[row['candidate_id'],k]) and corr.loc[row['candidate_id'],k]>=config.correlation_threshold]
        if conflicts:
            k,v=max(conflicts,key=lambda pair:pair[1]);row['excluded_by_correlation']={'candidate_id':k,'abs_correlation':v}
        elif len(selected)<config.max_factors:
            selected.append(row['candidate_id']);row['selected']=True
    if not selected:raise ValueError('No eligible nonredundant factors in fitting window')
    return selected,records
