
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


# options for initializing HistAuGAN
class opts:
    input_dim = 3
    num_domains = 7
    log_train_img_freq = 100
    log_val_img_freq = 100
    d_iter = 3
    lambda_rec = 10.
    lambda_cls = 1.
    lambda_cls_G = 5.


def augment(fullres: torch.Tensor, lowres: torch.Tensor, model: pl.LightningModule, z_attr: torch.Tensor) -> torch.Tensor:
    bs, _, _, _ = lowres.shape

    z_attr = z_attr.repeat(bs, 1).to(lowres.device)

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # to accelerate the forward pass through the GAN
            # compute content encoding
            z_content, _ = model.encoder(lowres)

            # generate augmentations
            out = model.generator(fullres, z_content, z_attr)  # in range [-1, 1]

    return out
