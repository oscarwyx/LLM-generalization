"""python -m genqa prepare / run / report。"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='SQuAD 1.1: baseline / prompting / LoRA')
    parser.add_argument('command', choices=['prepare','run','report'])
    parser.add_argument('--config',default='configs/experiment.json')
    parser.add_argument('--raw-dir',help='可选：已有四份原料的缓存目录')
    parser.add_argument('--model-dir',help='可选：已有固定revision的模型目录')
    parser.add_argument('--run-dir',help='report需要：本次运行目录')
    parser.add_argument('--smoke',action='store_true',help='run只使用16/8/8题检查流程，不产出正式成绩')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from .workflow import read_config, prepare_experiment, run_experiment
    if args.command == 'prepare':
        if args.smoke:
            parser.error('--smoke 只用于 run')
        root, config = read_config(args.config)
        prepare_experiment(root,config,raw_dir=args.raw_dir,model_dir=args.model_dir)
    elif args.command == 'run':
        run_experiment(args.config,raw_dir=args.raw_dir,model_dir=args.model_dir,smoke=args.smoke)
    else:
        if not args.run_dir:
            parser.error('report必须指定 --run-dir')
        from .report import make_report
        print(make_report(args.run_dir))


if __name__ == '__main__':
    main()
