#!/usr/bin/env python3
"""Recompute the generalization tables and article bootstrap from packaged predictions."""
from __future__ import annotations
import ast, hashlib, json, re, string
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduced'
OUT.mkdir(exist_ok=True)


def normalize_answer(value: str) -> str:
    # Mirrors genqa.engine at repository commit 2359fd5d9ea34bb9d1845fe1a6732d657343361f.
    value=value.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',value).split())


def score_answer(prediction: str, answers: list[str]) -> tuple[float,float]:
    vals=[]
    for answer in answers:
        p,a=normalize_answer(prediction),normalize_answer(answer)
        pw,aw=p.split(),a.split()
        common=sum((Counter(pw)&Counter(aw)).values())
        f1=2*common/(len(pw)+len(aw)) if pw and aw else 0.
        vals.append((float(p==a),f1))
    return max(x[0] for x in vals),max(x[1] for x in vals)


def interval(values: np.ndarray) -> tuple[float,float,float]:
    rng=np.random.default_rng(42)
    draws=rng.choice(values,size=(2000,len(values)),replace=True).mean(axis=1)
    return tuple(100*x for x in (values.mean(),*np.quantile(draws,[.025,.975])))


def question_changes(pred: pd.DataFrame,method: str,metric: str) -> pd.Series:
    subset=pred[pred.method.eq(method)]
    before=subset[subset.extra_target_tokens.eq(0)].set_index('id')[metric]
    after=(subset[subset.extra_target_tokens.eq(1024)]
           .groupby(['article','id'])[metric].mean().reset_index())
    after['change']=after[metric]-after['id'].map(before)
    assert len(after)==pred.id.nunique() and after.change.notna().all()
    return after.set_index(['article','id'])['change'].sort_index()


def assert_close(actual,expected,label):
    assert abs(float(actual)-float(expected))<1e-8,(label,actual,expected)


def analyze(which: str,expected_rows: int,questions_per_article: int) -> dict:
    folder=ROOT/'data'/which
    manifest=json.loads((folder/'manifest.json').read_text())
    pred=pd.read_csv(folder/'all_predictions.csv.gz',dtype={'id':str})
    assert len(pred)==expected_rows,(which,len(pred))
    assert pred.id.nunique()==len(manifest['question_ids'])
    assert set(pred.id)==set(manifest['question_ids'])
    assert pred.article.nunique()==45
    assert not pred.duplicated(['method','condition','id']).any()
    assert not pred.context_truncated.any()
    assert pred.groupby('article').id.nunique().eq(questions_per_article).all()
    assert not pred[['em','f1']].isna().any().any()
    for row in pred.itertuples(index=False):
        em,f1=score_answer(row.prediction,ast.literal_eval(row.answers))
        assert_close(row.em,em,'EM scorer')
        assert_close(row.f1,f1,'F1 scorer')
    summary=(pred.groupby(['method','extra_target_tokens','distractor_seed'],dropna=False)
             .agg(n=('id','count'),articles=('article','nunique'),em=('em','mean'),
                  f1=('f1','mean'),mean_prompt_tokens=('input_tokens','mean')).reset_index())
    summary[['em','f1']]*=100
    official=pd.read_csv(folder/'condition_summary.csv')
    for row in official.itertuples(index=False):
        s=summary[(summary.method.eq(row.method)) &
                  (summary.extra_target_tokens.eq(row.extra_target_tokens)) &
                  (summary.distractor_seed.eq(row.distractor_seed) if pd.notna(row.distractor_seed)
                   else summary.distractor_seed.isna())]
        assert len(s)==1
        for col in ('n','articles','em','f1','mean_prompt_tokens'):
            assert_close(s.iloc[0][col],getattr(row,col),f'{which} summary {col}')
    summary.to_csv(OUT/f'{which}_condition_summary.csv',index=False)
    if which=='validation_810':
        for method in ('Baseline','LoRA'):
            for condition in ('original','extra_1024_seed_101','extra_1024_seed_202'):
                metadata=folder/'run_metadata'/method.lower()/condition
                config=json.loads((metadata/'run_config.json').read_text())
                metrics=json.loads((metadata/'metrics.json').read_text())
                assert config['complete'] and metrics['complete']
                assert metrics['n_predictions']==810
                assert metrics['overall']['context_truncated_n']==0
                assert config['model_revision']==manifest['revision']
                assert config['max_new_tokens']==64 and config['batch_size']==4
                rows=pred[(pred.method.eq(method)) & (pred.condition.eq(condition))]
                assert len(rows)==810
                assert_close(100*rows.em.mean(),metrics['overall']['em'],f'{method}/{condition} metrics EM')
                assert_close(100*rows.f1.mean(),metrics['overall']['f1'],f'{method}/{condition} metrics F1')
    effects=[]
    for metric in ('em','f1'):
        baseline=question_changes(pred,'Baseline',metric)
        lora=question_changes(pred,'LoRA',metric)
        assert baseline.index.equals(lora.index)
        for contrast,series in (('Baseline',baseline),('LoRA',lora),
                                ('LoRA_minus_Baseline',lora-baseline)):
            articles=series.groupby(level='article').mean().to_numpy()
            assert len(articles)==45
            estimate,low,high=interval(articles)
            effects.append({'metric':metric,'contrast':contrast,'change_pp':estimate,
                            'ci95_low_pp':low,'ci95_high_pp':high})
    effects=pd.DataFrame(effects)
    effects.to_csv(OUT/f'{which}_effects_1024.csv',index=False)
    if which=='validation_810':
        official=pd.read_csv(folder/'paired_effects.csv')
        for row in official.itertuples(index=False):
            e=effects[(effects.metric.eq(row.metric)) & (effects.contrast.eq(row.contrast))]
            assert len(e)==1
            for col in ('change_pp','ci95_low_pp','ci95_high_pp'):
                assert_close(e.iloc[0][col],getattr(row,col),f'{which} {row.metric} {row.contrast} {col}')
    else:
        official=pd.read_csv(folder/'exploratory_interaction.csv')
        for row in official.itertuples(index=False):
            e=effects[(effects.metric.eq(row.metric)) &
                      (effects.contrast.eq('LoRA_minus_Baseline'))].iloc[0]
            assert_close(e.change_pp,row.difference_in_change_pp,'90 interaction')
            assert_close(e.ci95_low_pp,row.ci95_low_pp,'90 CI low')
            assert_close(e.ci95_high_pp,row.ci95_high_pp,'90 CI high')
        official_effects=pd.read_csv(folder/'paired_effects.csv')
        for row in official_effects[official_effects.extra_target_tokens.eq(1024)].itertuples(index=False):
            e=effects[(effects.metric.eq(row.metric)) & (effects.contrast.eq(row.method))].iloc[0]
            for col in ('change_pp','ci95_low_pp','ci95_high_pp'):
                assert_close(getattr(e,col),getattr(row,col),f'90 {row.method} {col}')
    primary=effects[(effects.metric.eq('em')) &
                    (effects.contrast.eq('LoRA_minus_Baseline'))].iloc[0]
    return {'rows':len(pred),'questions':pred.id.nunique(),'articles':45,
            'question_ids':set(pred.id),'commit':manifest['repository_commit'],
            'primary':{k:float(primary[k]) for k in ('change_pp','ci95_low_pp','ci95_high_pp')}}


def main():
    first=analyze('initial_90',900,2)
    follow=analyze('validation_810',4860,18)
    assert first['commit']==follow['commit']=='2359fd5d9ea34bb9d1845fe1a6732d657343361f'
    assert not first['question_ids'] & follow['question_ids']
    assert len(first['question_ids']|follow['question_ids'])==900
    failures=pd.read_csv(ROOT/'data/validation_810/lora_failure_categories_810.csv')
    counts=failures.category.value_counts().to_dict()
    assert len(failures)==149 and failures.id.nunique()==104
    assert counts=={'original_only_wrong_span':138,'neither_exact_span':7,
                    'span_in_both':2,'distractor_only_span':2}
    result={'initial_90':{k:v for k,v in first.items() if k!='question_ids'},
            'validation_810':{k:v for k,v in follow.items() if k!='question_ids'},
            'location_counts':counts,'checks':'all passed'}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
