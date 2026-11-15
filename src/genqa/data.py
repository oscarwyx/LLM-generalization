"""从四份公开原料真正清理和划分 SQuAD 1.1，不依赖任何旧实验。

这里只处理数据，不抽取 2048 条微调样本，不读取预测、分数或旧数据快照。
相对路径以本仓库根目录为基准；也可传入已有缓存的绝对路径。
所有输出使用 UTF-8 和 LF；清单不含时间戳或机器路径，便于两次再生比对。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import string
import tempfile
import unicodedata
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PRIORITY = {'squad_test': 0, 'validation': 1, 'train': 2}


def sha256_file(path):
    """大文件分块求哈希，避免为了校验另占一份内存。"""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT/path).resolve()


def _words(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def normalize_text(text):
    """只用于查重，模型输入保留原文。"""
    return ' '.join(_words(text))


def _answer_norm(text):
    text = ''.join(character for character in text.lower() if character not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def _rank(seed, scope, value):
    return _sha(f'article-study:{seed}:{scope}:{value}')


def _test_rank(seed, scope, value):
    return _sha(f'public-squad:{seed}:{scope}:{value}')


def _fivegrams(text):
    words = _words(text)
    return {tuple(words[i:i+5]) for i in range(max(0, len(words)-4))}


def _near(a_len, b_len, common):
    if not common:
        return False
    return common/(a_len+b_len-common) >= .5 or (common >= 20 and common/min(a_len,b_len) >= .5)


def near_duplicate_components(contexts):
    """精确倒排索引查找所有词面近重复，再用并查集求连通组件。

    判据：5-gram Jaccard >= 0.5；或较小集合包含率 >= 0.5 且共享至少
    20 个 5-gram。短文本规范化完全相同也合并。不是近似检索或语义去重。
    返回索引组件与达到阈值的边数量，不依据任何参考答案选择代表。
    """
    contexts = list(contexts)
    grams = [_fivegrams(text) for text in contexts]
    inverted, exact = defaultdict(list), defaultdict(list)
    for i, (text, values) in enumerate(zip(contexts, grams)):
        exact[normalize_text(text)].append(i)
        for value in values:
            inverted[value].append(i)
    parent = list(range(len(contexts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        x, y = find(a), find(b)
        if x != y:
            parent[max(x,y)] = min(x,y)

    edge_count = 0
    for i, values in enumerate(grams):
        candidates = Counter(j for value in values for j in inverted[value] if j < i)
        for j, common in candidates.items():
            if _near(len(values), len(grams[j]), common):
                union(i, j)
                edge_count += 1
        for j in exact[normalize_text(contexts[i])]:
            if j < i:
                union(i, j)
    groups = defaultdict(list)
    for i in range(len(contexts)):
        groups[find(i)].append(i)
    return list(groups.values()), edge_count


def _download_sources(raw_dir, download):
    config_path = REPO_ROOT/'configs/data_sources.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    sources = config['sources']
    required = {'squad_train','squad_test','original_train','original_dev'}
    if set(sources) != required:
        raise ValueError('原料配置必须且只能包含四份 MRQA/SQuAD 公开数据')
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, info in sources.items():
        path = raw_dir/info['filename']
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'离线模式缺少原料：{path}')
            temporary = path.with_name(path.name+'.part')
            request = urllib.request.Request(info['url'],headers={'User-Agent':'LLM-project-data/1.0'})
            with urllib.request.urlopen(request,timeout=180) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response,stream)
            if temporary.stat().st_size != info['bytes'] or sha256_file(temporary) != info['sha256']:
                raise ValueError(f'下载文件校验失败，保留 .part 供检查：{name}')
            temporary.replace(path)
        if path.stat().st_size != info['bytes'] or sha256_file(path) != info['sha256']:
            raise ValueError(f'原料版本不符，未覆盖缓存：{path}')
        paths[name] = path
    return paths, sources, sha256_file(config_path)


def _article_map(path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    result = {}
    for article in payload['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['id'] in result:
                    raise ValueError(f'官方 SQuAD 原始 ID 重复：{qa["id"]}')
                result[qa['id']] = {'article_title':article['title'],'question':qa['question']}
    return result


def load_mrqa(path, source_split, articles, expected_examples=None):
    """MRQA qid 是运行 ID；原始 SQuAD id 单独用于文章对齐，不能混用。"""
    rows, seen = [], set()
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if 'header' in record:
                continue
            context = record.get('context')
            if not isinstance(context,str) or not context.strip():
                raise ValueError('MRQA 正文为空或类型不合法')
            context_hash = _sha(' '.join(context.split()))
            normalized_context = _answer_norm(context)
            for qa in record['qas']:
                qid, original_id = qa['qid'], qa['id']
                if not isinstance(qid,str) or not qid or qid in seen:
                    raise ValueError(f'MRQA qid 无效或重复：{qid}')
                seen.add(qid)
                question, answers = qa['question'], qa['answers']
                if not isinstance(question,str) or not question.strip():
                    raise ValueError(f'问题为空：{qid}')
                if not isinstance(answers,list) or not answers or any(not isinstance(a,str) or not a.strip() for a in answers):
                    raise ValueError(f'SQuAD 1.1 必须有有效参考答案：{qid}')
                original = articles.get(original_id)
                if original is None or normalize_text(question) != normalize_text(original['question']):
                    raise ValueError(f'MRQA 与官方 SQuAD 原始问题 ID 对不齐：{qid}')
                rows.append({'id':qid,'qid':qid,'dataset':'SQuAD','source_split':source_split,
                             'context':context,'question':question,'answers':list(answers),
                             'context_hash':context_hash,
                             'question_context_hash':_sha(context_hash+'\n'+' '.join(question.split())),
                             'gold_present_in_raw_context_normalized':any(_answer_norm(a) and _answer_norm(a) in normalized_context for a in answers),
                             'article_title':original['article_title'],'source_original_id':original_id})
    if expected_examples is not None and len(rows) != expected_examples:
        raise ValueError(f'MRQA 行数不符：{len(rows)}，预期 {expected_examples}')
    return rows


def _merge_question_context(rows):
    """仅同一问题-正文的重复记录合并参考答案，保留最先出现的 qid。"""
    unique, removed = {}, 0
    for row in rows:
        key = row['question_context_hash']
        if key in unique:
            first = unique[key]
            first['answers'] = list(dict.fromkeys(first['answers']+row['answers']))
            first.setdefault('merged_source_ids',[first['id']]).append(row['id'])
            removed += 1
        else:
            unique[key] = {**row,'answers':list(row['answers'])}
    return list(unique.values()), removed


def _summary(rows):
    return {'n':len(rows),'articles':len({row['article_title'] for row in rows}),
            'contexts':len({row['context_hash'] for row in rows}),
            'article_titles':sorted({row['article_title'] for row in rows})}


def build_splits(train_rows, dev_rows, seed=42, validation_size=256, test_articles=45, test_per_article=20):
    """纯数据算法，供 prepare_data 调用，也方便小型单元测试。

    先按官方来源和文章确定归属，再处理全量正文组件。组件或规范化问题
    跨集合冲突时优先保留 dev，再验证，再训练，保证完整候选池隔离。
    测试文章和题目在清理完成后预先按哈希选择，不看答案内容或模型分数。
    """
    if type(seed) is not int or any(type(value) is not int or value < 1 for value in (validation_size,test_articles,test_per_article)):
        raise ValueError('seed 必须为整数，抽样数量必须为正整数')
    source_ids = [row['id'] for row in train_rows+dev_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError('公开 train/dev 中出现重复运行 ID')
    train, merged_train = _merge_question_context(train_rows)
    dev, merged_dev = _merge_question_context(dev_rows)
    dev_titles = {row['article_title'] for row in dev}
    source_train_titles = {row['article_title'] for row in train}
    shared_titles = source_train_titles & dev_titles
    removed_shared_articles = sum(row['article_title'] in shared_titles for row in train)
    train = [row for row in train if row['article_title'] not in shared_titles]
    titles = sorted({row['article_title'] for row in train}, key=lambda title:(_rank(seed,'article-fold',title),title))
    if len(titles) < 2:
        raise ValueError('至少需要两个不同训练来源文章才能划分 train/validation')
    validation_titles = set(titles[:max(1,round(.1*len(titles)))])
    assigned = [{**row,'split':'validation' if row['article_title'] in validation_titles else 'train'} for row in train]
    assigned += [{**row,'split':'squad_test'} for row in dev]
    contexts = sorted({row['context'] for row in assigned})
    positions = {text:i for i,text in enumerate(contexts)}
    grouped_rows = defaultdict(list)
    for row in assigned:
        grouped_rows[positions[row['context']]].append(row)
    components, edge_count = near_duplicate_components(contexts)
    owners = {}
    for component in components:
        priority = min(SPLIT_PRIORITY[row['split']] for i in component for row in grouped_rows[i])
        eligible = [i for i in component if any(SPLIT_PRIORITY[row['split']] == priority for row in grouped_rows[i])]
        chosen = min(eligible,key=lambda i:(_rank(seed,'context-canonical',contexts[i]),contexts[i]))
        eligible_rows = [row for row in grouped_rows[chosen] if SPLIT_PRIORITY[row['split']] == priority]
        title = min({row['article_title'] for row in eligible_rows},key=lambda title:(_rank(seed,'context-article-owner',title),title))
        owner = (chosen,eligible_rows[0]['split'],title,_sha('public-near-component:'+normalize_text(contexts[chosen])))
        for i in component:
            owners[i] = owner
    after_context, removals = defaultdict(list), Counter()
    for row in assigned:
        i = positions[row['context']]
        chosen, split, title, component_id = owners[i]
        if i != chosen or row['split'] != split or row['article_title'] != title:
            removals[row['split']] += 1
            continue
        after_context[split].append({**row,'group_id':'article:'+title,'near_component_id':component_id})
    cleaned, seen_questions, removed_questions = {}, set(), {}
    for split in SPLIT_PRIORITY:
        result = []
        for row in sorted(after_context[split],key=lambda row:(_rank(seed,'question-canonical',row['id']),row['id'])):
            question = normalize_text(row['question'])
            if question in seen_questions:
                continue
            seen_questions.add(question)
            result.append(row)
        cleaned[split] = result
        removed_questions[split] = len(after_context[split])-len(result)
    if not cleaned['train'] or len(cleaned['validation']) < validation_size:
        raise ValueError('清理后训练为空或验证题数不足；请检查数量配置')
    validation = sorted(cleaned['validation'],key=lambda row:(_rank(seed,'validation-sample',row['id']),row['id']))[:validation_size]
    test_by_article = defaultdict(list)
    for row in cleaned['squad_test']:
        test_by_article[row['article_title']].append(row)
    eligible_titles = [title for title, group in test_by_article.items() if len(group) >= test_per_article]
    if len(eligible_titles) < test_articles:
        raise ValueError(f'至少 {test_per_article} 题的测试文章只有 {len(eligible_titles)} 篇，少于请求的 {test_articles} 篇')
    chosen_titles = sorted(eligible_titles,key=lambda title:(_test_rank(seed,'article',title),title))[:test_articles]
    test = []
    for title in chosen_titles:
        test.extend(sorted(test_by_article[title],key=lambda row:(_test_rank(seed,'question:'+title,row['id']),row['id']))[:test_per_article])
    splits = {'train':cleaned['train'],'validation_full':cleaned['validation'],'validation':validation,
              'test_full':cleaned['squad_test'],'squad_test':test}
    # 比较完整候选池，而非只核对最终抽中的少量评测题。
    fields = {'id':lambda row:row['id'],'article':lambda row:row['article_title'],
              'context':lambda row:normalize_text(row['context']),'question':lambda row:normalize_text(row['question']),
              'near_component':lambda row:row['near_component_id']}
    checks = {}
    for a,b in [('train','validation_full'),('train','test_full'),('validation_full','test_full')]:
        counts = {name:len({get(row) for row in splits[a]} & {get(row) for row in splits[b]}) for name,get in fields.items()}
        if any(counts.values()):
            raise AssertionError(f'数据隔离失败：{a}/{b}: {counts}')
        checks[a+'__'+b] = counts
    stats = {'source_train_n':len(train_rows),'source_dev_n':len(dev_rows),
             'merged_duplicate_question_context_rows':{'train':merged_train,'dev':merged_dev},
             'train_articles_also_in_dev_removed':sorted(shared_titles),'train_rows_removed_for_shared_dev_article':removed_shared_articles,
             'assigned_train_article_count':len(titles)-len(validation_titles),'assigned_validation_article_count':len(validation_titles),
             'source_contexts':len(contexts),'near_components':len(components),'near_edges':edge_count,
             'noncanonical_contexts':len(contexts)-len(components),'context_rows_removed_by_split':dict(removals),
             'normalized_question_rows_removed_by_split':removed_questions,
             'eligible_test_articles':len(eligible_titles),'selected_test_article_titles':chosen_titles,
             'excluded_undersized_test_articles':{title:len(group) for title,group in sorted(test_by_article.items()) if len(group) < test_per_article},
             'cross_split_checks':checks,'split_counts':{key:_summary(value) for key,value in splits.items()}}
    return splits, stats


def _write_bytes(path, content):
    """仅原子更新本函数生成的同名文件，不删除目录或其它文件。"""
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary,path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_data(raw_dir, output_dir, seed=42, validation_size=256, test_articles=45, test_per_article=20, download=True):
    """下载/校验公开原料，清理、按文章划分并输出确定性的 JSONL 与清单。

    download=False 可使用已有缓存做完全离线重建。训练池在本函数中保持
    全量；2048 条微调样本及 token 长度预算应在模型/tokenizer 步骤再选。
    新测试集不执行历史预测排除，和旧实验的题目不同，必须重新运行评分。
    """
    raw_dir, output_dir = _path(raw_dir), _path(output_dir)
    paths, sources, config_hash = _download_sources(raw_dir,download)
    train = load_mrqa(paths['squad_train'],'train',_article_map(paths['original_train']),sources['squad_train']['expected_examples'])
    dev = load_mrqa(paths['squad_test'],'dev',_article_map(paths['original_dev']),sources['squad_test']['expected_examples'])
    splits, stats = build_splits(train,dev,seed,validation_size,test_articles,test_per_article)
    output_dir.mkdir(parents=True,exist_ok=True)
    filenames = {'train':'train.jsonl','validation_full':'validation.full.jsonl','validation':'validation.jsonl',
                 'test_full':'test.full.jsonl','squad_test':'squad_test.jsonl'}
    files = {}
    for name, rows in splits.items():
        path = output_dir/filenames[name]
        content = ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode('utf-8')
        _write_bytes(path,content)
        files[name] = {'path':filenames[name],'sha256':sha256_file(path),'bytes':path.stat().st_size,'n':len(rows)}
    preparation_config = {'seed':seed,'validation_fraction':.1,'validation_size':validation_size,
                          'test_articles':test_articles,'test_per_article':test_per_article}
    manifest = {'schema_version':1,'complete':True,'dataset':'MRQA 2019 SQuAD / SQuAD 1.1',
                'config':preparation_config,
                'preparation_config_sha256':_sha(json.dumps(preparation_config,sort_keys=True,separators=(',',':'))),
                'sources':sources,'data_sources_config_sha256':config_hash,'preparation_code_sha256':sha256_file(Path(__file__)),
                'files':files,'statistics':stats,
                'method':{'normalization':'Unicode NFKC/casefold/word tokens; raw inputs unchanged',
                          'near_duplicate':'5-word Jaccard>=0.5 OR smaller-set containment>=0.5 with >=20 shared; normalized exact also merged',
                          'ownership_priority':['official_dev','validation','train'],
                          'article_split':'SHA256(article-study:seed:article-fold:title), first rounded 10% validation',
                          'test_selection':'Eligible public dev articles ranked by SHA256(public-squad:seed:article:title); questions by SHA256(public-squad:seed:question:title:id)',
                          'history_files_accessed':False,'model_scores_used':False,'output_encoding':'UTF-8 LF'},
                'notes_zh':['这是从公开原料重新生成的划分，不使用旧快照、旧预测、旧分数或旧历史排除清单。',
                            '新测试题目与以往实验不同，旧成绩不可当成本次结果，必须重新运行模型评测。',
                            '项目测试来源是官方公开 dev，不是官方隐藏测试；底座预训练接触情况未知。',
                            '词面近重复清理不能排除所有语义改写；test.full 是清理后的公开候选池，不是未使用的新保留集。']}
    encoded = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    _write_bytes(output_dir/'manifest.json',encoded)
    return manifest
