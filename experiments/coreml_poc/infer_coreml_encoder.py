import sys
import torch
from depthcrafter.pipeline_coreml import Encoder2DCoreML


def main():
    if len(sys.argv) < 2:
        print("Usage: python infer_coreml_encoder.py <encoder2d.mlpackage>")
        return
    mlpackage = sys.argv[1]
    encoder = Encoder2DCoreML.from_mlpackage(mlpackage)
    example = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    feats = encoder(example)
    print(feats.shape)


if __name__ == "__main__":
    main()
