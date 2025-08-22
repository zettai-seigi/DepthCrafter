from typing import Optional

import numpy as np
import torch

from .depth_crafter_ppl import DepthCrafterPipeline, _resize_with_antialiasing_safe


class Encoder2DCoreML(torch.nn.Module):
    """Wrapper around a Core ML image encoder."""

    def __init__(self, mlmodel):
        super().__init__()
        self.mlmodel = mlmodel

    @classmethod
    def from_mlpackage(cls, path: str, compute_units=None):
        import coremltools as ct

        cfg = ct.models.MLModelConfiguration()
        cfg.compute_units = compute_units or ct.ComputeUnit.CPU_AND_NE
        mlmodel = ct.models.MLModel(path, configuration=cfg)
        return cls(mlmodel)

    def forward(self, frames_bchw: torch.Tensor) -> torch.Tensor:
        x = frames_bchw.detach().cpu().numpy()
        y = self.mlmodel.predict({"input": x})["output"]
        return torch.from_numpy(np.array(y))


class DepthCrafterCoreMLPipeline(DepthCrafterPipeline):
    """DepthCrafterPipeline that can offload the 2D image encoder to Core ML."""

    def __init__(self, coreml_encoder: Optional[Encoder2DCoreML] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coreml_encoder = coreml_encoder

    @torch.inference_mode()
    def encode_video(self, video: torch.Tensor, chunk_size: int = 14) -> torch.Tensor:
        if self.coreml_encoder is None:
            return super().encode_video(video, chunk_size)

        video_224 = _resize_with_antialiasing_safe(video.float(), (224, 224))
        video_224 = (video_224 + 1.0) / 2.0

        embeddings = []
        for i in range(0, video_224.shape[0], chunk_size):
            tmp = self.feature_extractor(
                images=video_224[i : i + chunk_size],
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=video.dtype)
            emb = self.coreml_encoder(tmp).to(video.device)
            embeddings.append(emb)

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings
