import torch as ch
from torch import nn
import dill
import os
from .tools import helpers, constants
from .attacker import AttackerModel

class FeatureExtractor(ch.nn.Module):
    '''
    Tool for extracting layers from models.

    Args:
        submod (torch.nn.Module): model to extract activations from
        layers (list of functions): list of functions where each function,
            when applied to submod, returns a desired layer. For example, one
            function could be `lambda model: model.layer1`.

    Returns:
        A model whose forward function returns the activations from the layers
            corresponding to the functions in `layers` (in the order that the
            functions were passed in the list).
    '''
    def __init__(self, submod, layers):
        # layers must be in order
        super(FeatureExtractor, self).__init__()
        self.submod = submod
        self.layers = layers
        self.n = 0

        for layer_func in layers:
            layer = layer_func(self.submod)
            def hook(module, _, output):
                module.register_buffer('activations', output)

            layer.register_forward_hook(hook)

    def forward(self, *args, **kwargs):
        """
        """
        # self.layer_outputs = {}
        out = self.submod(*args, **kwargs)
        activs = [layer_fn(self.submod).activations for layer_fn in self.layers]
        return [out] + activs

class DummyModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def forward_IN(self, x, *args, **kwargs):
        return self.model.forward_IN(x)

def make_and_restore_model(*_, arch, dataset, resume_path=None,
         parallel=False, pytorch_pretrained=False, add_custom_forward=False, r=4):
    """
    Makes a model and (optionally) restores it from a checkpoint.

    Args:
        arch (str|nn.Module): Model architecture identifier or otherwise a
            torch.nn.Module instance with the classifier
        dataset (Dataset class [see datasets.py])
        resume_path (str): optional path to checkpoint saved with the 
            robustness library (ignored if ``arch`` is not a string)
        not a string
        parallel (bool): if True, wrap the model in a DataParallel 
            (defaults to False)
        pytorch_pretrained (bool): if True, try to load a standard-trained 
            checkpoint from the torchvision library (throw error if failed)
        add_custom_forward (bool): ignored unless arch is an instance of
            nn.Module (and not a string). Normally, architectures should have a
            forward() function which accepts arguments ``with_latent``,
            ``fake_relu``, and ``no_relu`` to allow for adversarial manipulation
            (see `here`<https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-with-custom-architectures>
            for more info). If this argument is True, then these options will
            not be passed to forward(). (Useful if you just want to train a
            model and don't care about these arguments, and are passing in an
            arch that you don't want to edit forward() for, e.g.  a pretrained model)
    Returns: 
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    """
    if (not isinstance(arch, str)) and add_custom_forward:
        arch = DummyModel(arch)

    classifier_model = dataset.get_model(arch, pytorch_pretrained) if \
                            isinstance(arch, str) else arch
    print(dataset)
    model = AttackerModel(classifier_model, dataset)
    # print(model)
    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = ch.load(resume_path, pickle_module=dill)
        
        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        
        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        #print(sd)
        for k,v in sd.items():
           print(k)
        #for k,v in sd.items():
        #    if 'fc.weight' in k:
        #        sd['model.fc_downstream.weight']=sd[k]
        #        del sd[k]
        #    if 'fc.bias' in k:
        #        sd['model.fc_downstream.bias']=sd[k]
        #        del sd[k]
        #del sd['model.fc.weight']
        #del sd['model.fc.bias']

        weights_dict = {}
        for k, v in sd.items():
            # new_k = k.replace('module.model', 'module') if 'module.model' in k else k
            new_k = k.replace('0.conv1.', '0.conv1.conv.') if '0.conv1.' in k else k
            new_k = new_k.replace('0.conv2.', '0.conv2.conv.') if '0.conv2.' in new_k else new_k
            new_k = new_k.replace('0.conv3.', '0.conv3.conv.') if '0.conv3.' in new_k else new_k
            new_k = new_k.replace('1.conv1.', '1.conv1.conv.') if '1.conv1.' in new_k else new_k
            new_k = new_k.replace('1.conv2.', '1.conv2.conv.') if '1.conv2.' in new_k else new_k
            new_k = new_k.replace('1.conv3.', '1.conv3.conv.') if '1.conv3.' in new_k else new_k
            new_k = new_k.replace('2.conv1.', '2.conv1.conv.') if '2.conv1.' in new_k else new_k
            new_k = new_k.replace('2.conv2.', '2.conv2.conv.') if '2.conv2.' in new_k else new_k
            new_k = new_k.replace('2.conv3.', '2.conv3.conv.') if '2.conv3.' in new_k else new_k
            new_k = new_k.replace('3.conv1.', '3.conv1.conv.') if '3.conv1.' in new_k else new_k
            new_k = new_k.replace('3.conv2.', '3.conv2.conv.') if '3.conv2.' in new_k else new_k
            new_k = new_k.replace('3.conv3.', '3.conv3.conv.') if '3.conv3.' in new_k else new_k
            new_k = new_k.replace('4.conv1.', '4.conv1.conv.') if '4.conv1.' in new_k else new_k
            new_k = new_k.replace('4.conv2.', '4.conv2.conv.') if '4.conv2.' in new_k else new_k
            new_k = new_k.replace('4.conv3.', '4.conv3.conv.') if '4.conv3.' in new_k else new_k
            new_k = new_k.replace('5.conv1.', '5.conv1.conv.') if '5.conv1.' in new_k else new_k
            new_k = new_k.replace('5.conv2.', '5.conv2.conv.') if '5.conv2.' in new_k else new_k
            new_k = new_k.replace('5.conv3.', '5.conv3.conv.') if '5.conv3.' in new_k else new_k
            new_k = new_k.replace('downsample.0.', 'downsample.0.conv.') if 'downsample.0.' in new_k else new_k
            weights_dict[new_k] = v
        msg = model.load_state_dict(weights_dict, strict=False)
        
        print(msg)
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    elif resume_path:
        error_msg = "=> no checkpoint found at '{}'".format(resume_path)
        raise ValueError(error_msg)

    if parallel:
        model = ch.nn.DataParallel(model)
    model = model.cuda()

    return model, checkpoint

def model_dataset_from_store(s, overwrite_params={}, which='last'):
    '''
    Given a store directory corresponding to a trained model, return the
    original model, dataset object, and args corresponding to the arguments.
    '''
    # which options: {'best', 'last', integer}
    if type(s) is tuple:
        s, e = s
        s = cox.store.Store(s, e, mode='r')

    m = s['metadata']
    df = s['metadata'].df

    args = df.to_dict()
    args = {k:v[0] for k,v in args.items()}
    fns = [lambda x: m.get_object(x), lambda x: m.get_pickle(x)]
    conds = [lambda x: m.schema[x] == s.OBJECT, lambda x: m.schema[x] == s.PICKLE]
    for fn, cond in zip(fns, conds):
        args = {k:(fn(v) if cond(k) else v) for k,v in args.items()}

    args.update(overwrite_params)
    args = Parameters(args)

    data_path = os.path.expandvars(args.data)
    if not data_path:
        data_path = '/tmp/'

    dataset = DATASETS[args.dataset](data_path)

    if which == 'last':
        resume = os.path.join(s.path, constants.CKPT_NAME)
    elif which == 'best':
        resume = os.path.join(s.path, constants.CKPT_NAME_BEST)
    else:
        assert isinstance(which, int), "'which' must be one of {'best', 'last', int}"
        resume = os.path.join(s.path, ckpt_at_epoch(which))

    model, _ = make_and_restore_model(arch=args.arch, dataset=dataset,
                                      resume_path=resume, parallel=False)
    return model, dataset, args
