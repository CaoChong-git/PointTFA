import argparse
import models.ULIP_models as models

from collections import OrderedDict
from datasets.modelnet40 import *
from utils.utils import get_dataset
from utils.tokenizer import SimpleTokenizer
from utils_tfa import *

def get_arguments():
    # Data
    parser = argparse.ArgumentParser(description='PointTFA on modelnet40', add_help=False)
    parser.add_argument('--pretrain_dataset_name', default='modelnet40', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--validate_dataset_name', default='modelnet40_test', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')

    # ULIP_model
    parser.add_argument('--model', default='ULIP_PointBERT', type=str)
    parser.add_argument('--ckpt_addr', default='./pretrained_ckpt/ckpt_pointbert_ULIP-2.pt',
                        help='the ckpt to ulip 3d enconder')
    parser.add_argument('--evaluate_3d', default='True', help='eval 3d only')

    # cfg
    parser.add_argument('--config', dest='config', help='settings of PointTFA in yaml format')
    args = parser.parse_args()

    return args


def run_PointTFA(cfg, cache_keys, cache_values, test_features, test_labels, ulip_weights):
    
    print("\nRun PointTFA:")
    # 3D Zero-shot
    ulip_logits = 100. * test_features @ ulip_weights
    acc = cls_acc(ulip_logits, test_labels)
    print("\n**** Zero-shot ULIP's test accuracy: {:.2f}. ****\n".format(acc))

    # PointTFA
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tfa_logits = ulip_logits + cache_logits * alpha
    acc = cls_acc(tfa_logits, test_labels)
    print("**** PointTFA's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, ulip_weights)

    # overall acc

def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")


    # ULIP model
    ckpt = torch.load(args.ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    try:
        model = getattr(models, old_args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.ckpt_addr))
    except:
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.ckpt_addr))
    model.eval()

    #modelnet40 dataset
    random.seed(1)
    torch.manual_seed(1)

    tokenizer = SimpleTokenizer()

    print("Preparing modelnet40 dataset.")
    train_dataset = get_dataset(None, tokenizer, args, 'train')
    test_dataset = get_dataset(None, tokenizer, args, 'val')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=10, shuffle=False, pin_memory=True, sampler=None, drop_last=False)
    train_loader_cache = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=10, shuffle=False, pin_memory=True, sampler=None, drop_last=False)

    # Textual features
    print("Getting textual features as ULIP's classifier.")
    ulip_weights = ulip_classifier(args, model, tokenizer)

    # Support memory Preparation
    print("\nConstructing full-set cache model by training set visual features and labels.")
    cache_keys, cache_values = build_pc_cache_model(cfg, model, train_loader_cache)

    # Data-efficient RMC
    print("\nConstructing representative memory cache.")
    cache_keys, cache_values = data_efficiency(cfg, cache_keys, cache_values, train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_pc_features(cfg, "test", model, test_loader)

    # Cloud Query Refactor
    print("\nTransfer cache knowledge to test features.")
    test_features = reinvent_query(test_features, cache_keys)

    # ------------------------------------------ PointTFA ------------------------------------------
    run_PointTFA(cfg, cache_keys, cache_values, test_features, test_labels, ulip_weights)

if __name__ == '__main__':
    main()