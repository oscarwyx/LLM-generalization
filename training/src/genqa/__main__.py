"""python -m genqa prepare / run / report / pretrained。"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='SQuAD 1.1: baseline / prompting / LoRA')
    parser.add_argument('command', choices=['prepare','run','report','pretrained'])
    parser.add_argument('--config',default='configs/experiment.json')
    parser.add_argument('--raw-dir',help='可选：已有四份原料的缓存目录')
    parser.add_argument('--model-dir',help='可选：已有固定revision的模型目录')
    parser.add_argument('--run-dir',help='report需要：本次运行目录')
    parser.add_argument('--smoke',action='store_true',help='run只使用16/8/8题检查流程，不产出正式成绩')
    parser.add_argument('--published-dir',default='published/seed17_epoch2',help='pretrained：已训练权重和固定评测数据目录')
    parser.add_argument('--split',choices=['train','validation','test'],default='test',help='pretrained：重新推理的集合')
    parser.add_argument('--verify-only',action='store_true',help='pretrained：仅校验发布文件，不调用GPU、不下载')
    args = parser.parse_args()
    if args.verify_only and args.command != 'pretrained':
        parser.error('--verify-only 只用于 pretrained')
    if args.smoke and args.command != 'run':
        parser.error('--smoke 只用于 run')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from .workflow import read_config, prepare_experiment, run_experiment
    if args.command == 'prepare':
        if args.smoke:
            parser.error('--smoke 只用于 run')
        root, config = read_config(args.config)
        prepare_experiment(root,config,raw_dir=args.raw_dir,model_dir=args.model_dir)
    elif args.command == 'run':
        run_experiment(args.config,raw_dir=args.raw_dir,model_dir=args.model_dir,smoke=args.smoke)
    elif args.command == 'report':
        if not args.run_dir:
            parser.error('report必须指定 --run-dir')
        from .report import make_report
        print(make_report(args.run_dir))
    else:
        from .pretrained import verify_published, evaluate_published
        root, _ = read_config(args.config)
        if args.verify_only:
            result = verify_published(root,args.published_dir,model_dir=args.model_dir)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        else:
            result = evaluate_published(root,args.published_dir,split=args.split,model_dir=args.model_dir)
            print(json.dumps(result['metrics']['overall'],ensure_ascii=False,indent=2))
            print('已训练模型的本次推理结果：',result['run_dir'])


if __name__ == '__main__':
    main()
