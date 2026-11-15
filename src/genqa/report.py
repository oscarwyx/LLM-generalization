"""只读取指定运行的真实预测/日志，生成比较表与图；从不装载预存参考成绩。"""
from pathlib import Path
from .engine import load_json


def make_report(run_dir):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(run_dir).expanduser().resolve()
    manifest = load_json(directory/'run_manifest.json')
    training = load_json(directory/'lora_training/training_summary.json')
    if not training.get('complete') or not training.get('budget_satisfied'):
        raise ValueError('训练未完成或预算未满足，不能生成正式完成报告')
    epochs = manifest['config']['training']['epochs']
    rows, predictions = [], {}
    for method, folder in [('Baseline','baseline'),('Prompting','prompting'),('LoRA',f'lora_epoch_{epochs}')]:
        for phase in ('train','validation','test'):
            stage = directory/folder/phase
            row = load_json(stage/'table_row.json')
            if row['n'] != manifest['active_counts'][phase]:
                raise ValueError('指标样本数与本次运行清单不一致')
            rows.append(row)
            if phase == 'test':
                predictions[method] = pd.read_json(stage/'predictions.jsonl',lines=True,dtype={'id':str})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(directory/'metrics.csv',index=False)
    epoch_train = pd.DataFrame(training['epochs'])
    epoch_train.to_csv(directory/'training_epochs.csv',index=False)
    validation = pd.DataFrame([{'epoch':epoch,**load_json(directory/f'lora_epoch_{epoch}/validation/table_row.json')}
                               for epoch in range(1,epochs+1)])
    validation.to_csv(directory/'epoch_validation.csv',index=False)
    steps = pd.read_json(directory/'lora_training/train_log.jsonl',lines=True)
    steps.to_csv(directory/'training_steps.csv',index=False)
    scores = metrics[metrics.phase=='test'].set_index('method')[['EM (%)','F1 (%)']]
    scores.subtract(scores.loc['Baseline']).rename(columns={'EM (%)':'EM gain (pp)','F1 (%)':'F1 gain (pp)'}).to_csv(directory/'test_gains.csv')
    comparison = predictions['Baseline'][['id','question','answers']].copy()
    comparison['answers'] = comparison['answers'].map(lambda x:' | '.join(x))
    for method, values in predictions.items():
        comparison = comparison.merge(values[['id','prediction']].rename(columns={'prediction':method}),on='id',validate='one_to_one')
    if len(comparison) != manifest['active_counts']['test']:
        raise ValueError('三种方法的预测ID未完全对齐')
    comparison.to_csv(directory/'test_answers.csv',index=False,encoding='utf-8-sig')
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    axes[0].plot(steps.global_step,steps.train_nll,alpha=.35,label='step')
    axes[0].plot(steps.global_step,steps.train_nll.rolling(8,min_periods=1).mean(),label='8-step mean')
    axes[0].set(xlabel='Optimizer step',ylabel='Answer NLL',title='LoRA training')
    axes[0].legend()
    for key in ('EM (%)','F1 (%)'):
        axes[1].plot(validation.epoch,validation[key],marker='o',label=key)
    axes[1].set(xlabel='Epoch',ylabel='Score (%)',title='Validation',xticks=validation.epoch)
    axes[1].legend()
    scores.plot.bar(ax=axes[2],rot=0)
    axes[2].set(title='Test comparison',ylabel='Score (%)',xlabel='',ylim=(0,100))
    if manifest['smoke']:
        fig.suptitle('SMOKE TEST ONLY: not formal experiment scores')
    fig.tight_layout()
    fig.savefig(directory/'results.png',dpi=160)
    plt.close(fig)
    print(metrics.round(3).to_string(index=False),flush=True)
    print('结果图表与逐题答案：',directory,flush=True)
    return metrics
