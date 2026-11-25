"""数据单元测试：使用小型合成记录，不联网、不加载模型。"""
from pathlib import Path
import gzip
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from genqa.data import build_splits, load_mrqa, near_duplicate_components, normalize_text


def fixture_row(ident, title, context, question=None, answer=None, source='train'):
    question = question or f'What is the item {ident}?'
    digest = lambda value:hashlib.sha256(value.encode()).hexdigest()
    context_hash = digest(' '.join(context.split()))
    return {'id':ident,'qid':ident,'dataset':'SQuAD','source_split':source,'context':context,
            'question':question,'answers':[answer or context.split()[0]],'context_hash':context_hash,
            'question_context_hash':digest(context_hash+'\n'+' '.join(question.split())),
            'article_title':title,'source_original_id':'original-'+ident}


def fixtures():
    train, dev = [], []
    for i in range(12):
        context = ' '.join(f'train{i}word{j}' for j in range(35))
        train.extend(fixture_row(f't{i}q{q}',f'Train_{i}',context) for q in range(2))
    for i in range(3):
        context = ' '.join(f'dev{i}word{j}' for j in range(35))
        dev.extend(fixture_row(f'd{i}q{q}',f'Dev_{i}',context,source='dev') for q in range(3))
    return train,dev


class DataTests(unittest.TestCase):
    def test_article_disjoint_and_reproducible(self):
        train,dev = fixtures()
        args = dict(seed=42,validation_size=2,test_articles=2,test_per_article=2)
        first, audit = build_splits(train,dev,**args)
        second, second_audit = build_splits(train,dev,**args)
        self.assertEqual(first,second)
        self.assertEqual(audit,second_audit)
        self.assertEqual(len(first['squad_test']),4)
        self.assertEqual(len({r['article_title'] for r in first['squad_test']}),2)
        for values in audit['cross_split_checks'].values():
            self.assertTrue(all(value==0 for value in values.values()))
        source = {r['id']:r for r in train+dev}
        for group in first.values():
            for row in group:
                self.assertEqual(row['context'],source[row['id']]['context'])
                self.assertEqual(row['answers'],source[row['id']]['answers'])
                self.assertEqual(row['source_original_id'],source[row['id']]['source_original_id'])

    def test_near_copy_and_short_normalized_exact(self):
        base = ' '.join(f'word{i}' for i in range(60))
        extended = base+' '+' '.join(f'extra{i}' for i in range(12))
        unrelated = ' '.join(f'unrelated{i}' for i in range(60))
        groups,edges = near_duplicate_components([base,extended,unrelated,'A! B','a b'])
        sets = {frozenset(group) for group in groups}
        self.assertIn(frozenset([0,1]),sets)
        self.assertIn(frozenset([2]),sets)
        self.assertIn(frozenset([3,4]),sets)
        self.assertGreaterEqual(edges,1)

    def test_cross_source_near_context_removed_before_sampling(self):
        train,dev = fixtures()
        train[0] = fixture_row('near-train','Extra_train',dev[0]['context']+' extra unique ending')
        splits,audit = build_splits(train,dev,validation_size=2,test_articles=2,test_per_article=2)
        self.assertNotIn('near-train',{r['id'] for r in splits['train']+splits['validation_full']})
        self.assertGreater(sum(audit['context_rows_removed_by_split'].values()),0)

    def test_runtime_id_and_original_id_are_not_confused(self):
        # 临时文件放在测试目录内，避免受限环境的系统临时目录权限差异。
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_root,prefix='data-unit-') as directory:
            self.assertIn(test_root,Path(directory).resolve().parents)
            path = Path(directory)/'sample.gz'
            record = {'context':'Alpha is a city.', 'qas':[{'qid':'runtime-id','id':'squad-original-id',
                      'question':'What is Alpha?','answers':['a city']}]}
            with gzip.open(path,'wt',encoding='utf-8') as stream:
                stream.write(json.dumps({'header':{'dataset':'SQuAD'}})+'\n')
                stream.write(json.dumps(record)+'\n')
            mapping = {'squad-original-id':{'article_title':'Alpha','question':'What is Alpha?'}}
            values = load_mrqa(path,'train',mapping,expected_examples=1)
            self.assertEqual(values[0]['id'],'runtime-id')
            self.assertEqual(values[0]['source_original_id'],'squad-original-id')
            mapping['squad-original-id']['question']='An unrelated question?'
            with self.assertRaises(ValueError):
                load_mrqa(path,'train',mapping,expected_examples=1)

    def test_duplicate_ids_fail_and_references_merge(self):
        train,dev = fixtures()
        duplicate = {**train[0],'id':'merged-id','qid':'merged-id','answers':['second accepted reference']}
        splits,_ = build_splits(train+[duplicate],dev,validation_size=2,test_articles=2,test_per_article=2)
        row = next(r for r in splits['train']+splits['validation_full'] if r['id']==train[0]['id'])
        self.assertEqual(row['merged_source_ids'],[train[0]['id'],'merged-id'])
        self.assertEqual(row['answers'],train[0]['answers']+['second accepted reference'])
        with self.assertRaises(ValueError):
            build_splits(train,dev+[train[0]],validation_size=2,test_articles=2,test_per_article=2)


if __name__ == '__main__':
    unittest.main()
