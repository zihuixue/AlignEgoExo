import torch

def load_ckpt(backbone, ckpt_name):
    print(f'Loading pre-trained model: {ckpt_name}')
    ckpt = torch.load(
        ckpt_name,
        map_location=lambda storage, loc: storage,
    )

    def remove_first_module(key):
        return ".".join(key.split(".")[1:])

    key = "state_dict" if "state_dict" in ckpt.keys() else "model_state"
    state_dict = {
        remove_first_module(k): v
        for k, v in ckpt[key].items()
    }

    missing_keys, unexpected_keys = backbone.load_state_dict(
        state_dict, strict=False
    )

    print('missing', missing_keys)
    print('unexpected', unexpected_keys)