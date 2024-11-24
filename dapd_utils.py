import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import types
from collections import OrderedDict
from collections import deque
import collections

class class_pathways(object):
    def __init__(self, keep_ratio, history_len=1):
        self.all_scores = None
        self.keep_ratio = keep_ratio
        self.record_score = collections.defaultdict(dict)
        self.record_score['actor'] = deque(maxlen=history_len)
        self.record_score['critic'] = deque(maxlen=history_len)

        self.last_mask = collections.defaultdict(dict)
        self.last_mask['actor'] = []
        self.last_mask['critic'] = []
        self.prune_param_score = 0

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    def patch_forward(self, net):
        for layer in net.modules():  # for name, layer in net.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight.requires_grad = False
            # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(modify_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(modify_forward_linear, layer)

    def get_keep_masks(self, net, type, allow_old_weights=False):
        grads_abs = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append(torch.abs(layer.weight_mask.grad))
        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)
        self.record_score[type].append(all_scores.detach().cpu().numpy())
        all_scores = torch.tensor(np.mean(np.stack(list(self.record_score[type]), 0), 0)).to(grads_abs[0].device)
        num_params_to_keep = int(len(all_scores) * self.keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        if (acceptable_score == 0 and self.last_mask[type]!=[]):
            return self.last_mask[type]
        keep_masks = []
        for g in grads_abs:
            h, w = g.shape
            keep_masks.append((all_scores[:h*w].reshape(h, w) >= acceptable_score).float()) # for each layer, remap the 2D vector and get one-hot matrix
            all_scores = all_scores[h*w:]
        if allow_old_weights:
            if self.last_mask[type] != []:
                masks = []
                for m1, m2 in zip(self.last_mask[type], keep_masks):
                    masks.append(torch.logical_or(m1, m2).type(torch.float32))
                keep_masks = masks
        self.last_mask[type] = keep_masks
        return keep_masks

    def get_masks(self, net, replay_buffer, itr=1, allow_old_weights=False):
        assert allow_old_weights == False
        self.patch_forward(net.actor)
        self.patch_forward(net.critic)
        self.patch_forward(net.critic_target)
        net.calculate_snip_score(replay_buffer, itr)

        return (self.get_keep_masks(net.actor, 'actor', allow_old_weights)), \
               (self.get_keep_masks(net.critic, 'critic', allow_old_weights))

    def get_sps_count(self):
        nonzero = total = 0
        for name, net in self.last_mask.items():
            for layer in net:
                nz_count = torch.count_nonzero(layer)
                total_params = layer.numel()
                nonzero += nz_count
                total += total_params
            abs_sps = 100 * (total - nonzero) / total
            print(f'{name}: abs {abs_sps} | total {total} | nonzero {nonzero}')


def get_abs_sps(model):
    nonzero = total = 0
    for name, param in model.named_parameters():
        tensor = param.data
        nz_count = torch.count_nonzero(tensor)
        total_params = tensor.numel()
        nonzero += nz_count
        total += total_params
    abs_sps = 100 * (total - nonzero) / total
    return abs_sps, total, nonzero


def get_abs_sps_each_layer(model):
    total = []
    nonzero = []
    abs_sps = []

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            t = m.weight.numel()
            n = torch.count_nonzero(m._parameters['weight']).item()  # torch.count_nonzero(m._parameters['weight']) ; torch.nonzero(m.weight).shape[0]
            total.append(t)
            nonzero.append(n)
            abs_sps.append(round( 100 * ((t - n) / t), 2))
    return abs_sps, total, nonzero

def sparsity_level(mask):
    total = 0
    nonzero = 0
    for l in mask:
        total += len(l.flatten())
        nonzero += sum(l.flatten())
    sp = 100 * (total - nonzero) / total
    return sp.to('cpu').numpy()


def load_keep_masks(path):
    keep_masks = np.load(f'{path}/keep_masks.npy', allow_pickle=True).item()
    return keep_masks



def modify_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)

def modify_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def mask_network(model, mask_layers, mod_forward=True):
    prunable_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),model.modules())
    count_mask_layers = len(mask_layers)
    for i, (layer, mask) in enumerate(zip(prunable_layers, mask_layers)):
        # revert back the weight-grad
        layer.weight.requires_grad = True
        del layer.weight_mask
        layer.weight_mask = mask
        if mod_forward:  # default True
            # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(modify_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(modify_forward_linear, layer)
    assert count_mask_layers == i + 1  # this checks if all the mask layers are being used

def apply_backward_hook(net, keep_masks, fixed_weight=0.):
    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())
    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """
            def hook(grads):
                return grads * keep_mask
            return hook
        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter
        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        if fixed_weight == -1:
            pass
        else:
            layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask)) # register hook is backward hook


def preprocess_for_mask_update(net):
    for layer in net.modules():  # for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))  # default require_grad True
            assert layer.weight_mask.requires_grad == True
            # unhook
            layer.weight._backward_hooks = OrderedDict()
            # set require_grad False
            layer.weight.requires_grad = False
        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(modify_forward_conv2d, layer)
        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(modify_forward_linear, layer)


def get_masks(net, keep_ratio=0.05):
    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())
    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))
    return keep_masks