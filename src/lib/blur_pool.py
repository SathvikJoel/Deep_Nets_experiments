'''
* Author : Sathvik Joel K
* 
* file name : blur_pool.py
*
* Purpose : Implemets functions to replace all instances of max_pool with blur pool
*
* Related paper : Making Convolutions shift invariant again
*
* Bugs : --
*
* Change Log : --
'''

import kornia

def convert_MP_to_blurMP(model, layer_type_old):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_MP_to_blurMP(module, layer_type_old)

        if type(module) == layer_type_old:
            layer_old = module
            layer_new = kornia.contrib.MaxBlurPool2d(3, True)
            model._modules[name] = layer_new

    return model