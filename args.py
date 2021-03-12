import argparse

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str, default='datasets/indomain_train')
    parser.add_argument('--val-dir', type=str, default='datasets/indomain_val')
    parser.add_argument('--eval-dir', type=str, default='datasets/oodomain_test')
    parser.add_argument('--eval-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-finetune', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)

    ## maml specific
    parser.add_argument('--num-inner-updates', type=int, default=5)
    parser.add_argument('--num-tasks', type=int, default=8)
    parser.add_argument('--inner-lr', type=float, default=1e-3)
    parser.add_argument('--task-weight', type=str, default="proportional")
    parser.add_argument('--meta-update', type=str, default="reptile")
    parser.add_argument('--meta-lr', type=float, default=1e-3)

    return parser

def get_train_test_args():
    parser = init_parser()
    args = parser.parse_args()
    return args

def get_debug_args(run_name="maml", meta_update="reptile"):
    parser = init_parser()

    input = [
        '--run-name', run_name,
        '--meta-update', meta_update,
        '--do-train',
        '--batch-size', '1',
        '--eval-every', '2000',
        '--num-inner-updates', '3',
        '--num-tasks', '2'
        #"--recompute-features"
    ]
    args = parser.parse_args(input)
    return args

