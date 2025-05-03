import torch
import torch.nn as nn

from models.components import Encoder, Cross_Att, Decoder, Classifier
from segment_anything import SamPredictor, sam_model_registry
import cv2


import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry


import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry


import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry


class SAMSceneEncoder(nn.Module):
    def __init__(self, sam_checkpoint: str, model_type: str = "vit_b"):
        """
        Args:
            sam_checkpoint: Path to the SAM checkpoint file.
            model_type: One of "vit_b", "vit_l", or "vit_h".
        """
        super().__init__()
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.encoder = sam_model.image_encoder.eval()
        self.upsample = nn.Upsample(size=(1024, 1024), mode='bilinear', align_corners=False)
        self.channel_project = nn.Conv2d(self._get_output_channels(model_type), 480, kernel_size=1)


    def _get_output_channels(self, model_type: str) -> int:
        """
        Returns the number of output channels for the SAM encoder based on model type.
        """
        if model_type == "vit_b":
            return 256
        elif model_type == "vit_l":
            return 1024
        elif model_type == "vit_h":
            return 1280
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape [batch_size, 3, 256, 256].

        Returns:
            Tensor of shape [batch_size, 480, 64, 64] representing the encoded scene features.
        """
        x = self.upsample(x)

        with torch.no_grad():
            features = self.encoder(x)  # [batch_size, 256, 64, 64]

        features = self.channel_project(features)  # [batch_size, 480, 64, 64]

        return features



class DECO_modernized(nn.Module):
    def __init__(self, encoder, context, device):
        super(DECO_modernized, self).__init__()

        self.encoder_type = encoder

        self.context = context

        self.encoder_sem = SAMSceneEncoder(sam_checkpoint="../../segment-anything/checkpoints/sam_vit_b_01ec64.pth").to(device)
        self.encoder_part = Encoder(encoder=encoder).to(device)

        self.correction_conv = nn.Conv1d(768, 1024, 1).to(device)

        if self.context:
            self.decoder_sem = Decoder(1, 133, encoder=encoder).to(device)
            self.decoder_part = Decoder(1, 26, encoder=encoder).to(device)
        self.cross_att = Cross_Att(1024, 1024).to(device)
        self.classif = Classifier(1024).to(device)

        self.device = device

    def forward(self, img):
        if self.encoder_type == 'hrnet':
            sem_enc_out = self.encoder_sem(img)
            part_enc_out = self.encoder_part(img)

            if self.context:
                sem_mask_pred = self.decoder_sem(sem_enc_out)
                part_mask_pred = self.decoder_part(part_enc_out)

            sem_enc_out = self.sem_pool(sem_enc_out)
            sem_enc_out = sem_enc_out.squeeze(2)
            sem_enc_out = sem_enc_out.squeeze(2)
            sem_enc_out = sem_enc_out.unsqueeze(1)

            part_enc_out = self.part_pool(part_enc_out)
            part_enc_out = part_enc_out.squeeze(2)
            part_enc_out = part_enc_out.squeeze(2)
            part_enc_out = part_enc_out.unsqueeze(1)

            att = self.cross_att(sem_enc_out, part_enc_out)
            cont = self.classif(att)
        else:
            sem_enc_out = self.encoder_sem(img)
            part_enc_out = self.encoder_part(img)

            sem_seg = torch.reshape(sem_enc_out, (-1, 768, 1))
            part_seg = torch.reshape(part_enc_out, (-1, 768, 1))

            sem_seg = self.correction_conv(sem_seg)
            part_seg = self.correction_conv(part_seg)

            sem_seg = torch.reshape(sem_seg, (-1, 1, 32, 32))
            part_seg = torch.reshape(part_seg, (-1, 1, 32, 32))

            if self.context:
                sem_mask_pred = self.decoder_sem(sem_seg)
                part_mask_pred = self.decoder_part(part_seg)

            sem_enc_out = torch.reshape(sem_seg, (-1, 1, 1024))
            part_enc_out = torch.reshape(part_seg, (-1, 1, 1024))

            att = self.cross_att(sem_enc_out, part_enc_out)
            cont = self.classif(att)

        if self.context: return cont, sem_mask_pred, part_mask_pred
        return cont

