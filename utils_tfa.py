import torch
import torch.nn.functional as F
import os
import json

from tqdm import tqdm
from kmeans_pytorch import kmeans
from utils import utils


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def ulip_classifier(args, model, tokenizer):
    with open(os.path.join("./DATA", 'templates.json')) as f:
        templates = json.load(f)[args.pretrain_dataset_prompt]

    with open(os.path.join("./DATA", 'labels.json')) as f:
        labels = json.load(f)[args.pretrain_dataset_name]

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(None, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = utils.get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)

        text_features = torch.stack(text_features, dim=0)
        text_features = text_features.permute(1, 0)
    return text_features


def build_pc_cache_model(cfg, model, train_loader_cache):

    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('\nAugment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (pc, target, target_name) in enumerate(tqdm(train_loader_cache)):
                    pc = pc.cuda()
                    pc_features = utils.get_model(model).encode_pc(pc)
                    train_features.append(pc_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)

        cache_values = torch.cat(cache_values, dim=0)
        cache_values = cache_values.to(torch.int64)
        cache_values = F.one_hot(cache_values).float()

        torch.save(cache_keys, cfg['cache_dir'] + "/keys.pt")
        torch.save(cache_values, cfg['cache_dir'] + "/values.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + "/keys.pt")
        cache_values = torch.load(cfg['cache_dir'] + "/values.pt")

    return cache_keys, cache_values


def pre_load_pc_features(cfg, split, model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (pc, target, target_name) in enumerate(tqdm(loader)):
                pc = pc.cuda()
                pc_features = utils.get_model(model).encode_pc(pc)
                pc_features /= pc_features.norm(dim=-1, keepdim=True)
                features.append(pc_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)
        labels = labels.to(torch.int64)
        labels = labels.cuda()

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")

    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")

    return features, labels


def data_efficiency(cfg, cache_keys, cache_values, train_loader_cache):

    # Aggregating data for each category
    cache_keys = cache_keys.permute(1, 0)
    list_of_labels = train_loader_cache.dataset.list_of_labels
    current_class = list_of_labels[0]
    start_idx = 0
    values_class = []
    keys_class = []

    for i in range(len(list_of_labels)):
        if list_of_labels[i] != current_class:
            values_class.append(cache_values[start_idx: i])
            keys_class.append(cache_keys[start_idx: i])
            current_class = list_of_labels[i]
            start_idx = i

    values_class.append(cache_values[start_idx:])
    keys_class.append(cache_keys[start_idx:])

    # RMC
    cache_keys = []
    for augment_idx in range(cfg['augment_epoch']):
        new_keys = []
        new_values = []
        for key in keys_class:
            if cfg['n_clusters'] != 1:
                cluster_idx_x, cluster_centers = kmeans(X = key, num_clusters = cfg['n_clusters'], distance = 'euclidean',
                                                        device = torch.device('cuda:0'))
            else:
                cluster_centers = key.mean(dim=0).unsqueeze(0)
            new_keys.append(cluster_centers)

        cache_keys.append(torch.cat(new_keys, dim=0).unsqueeze(0))

        if augment_idx == 0:
            for value in values_class:
                for i in range(cfg['n_clusters']):
                    value_i = value[i]
                    new_values.append(value_i)

            cache_values = torch.stack(new_values).cuda()

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0).cuda()

    return cache_keys, cache_values

def reinvent_query(test_features, cache_keys):

    sim = test_features@cache_keys
    sim = (sim*100).softmax(dim=-1)
    test_features = sim@cache_keys.T
    test_features /= test_features.norm(dim=-1, keepdim=True)

    return test_features


def search_hp(cfg, cache_keys, cache_values, features, labels, ulip_weights):

    if cfg['search_hp'] == True:

        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:

                affinity = features @ cache_keys
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                ulip_logits = 100. * features @ ulip_weights
                tfa_logits = ulip_logits + cache_logits * alpha
                acc = cls_acc(tfa_logits, labels)

                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, PointTFA the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha