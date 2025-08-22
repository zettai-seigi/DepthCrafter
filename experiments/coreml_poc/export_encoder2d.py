import sys
import torch
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_encoder2d.py <model_id_or_path>")
        return
    model_id = sys.argv[1]
    pipe = DepthCrafterPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    encoder = pipe.image_encoder.eval()
    example = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
    exported = torch.export.export(encoder, example)
    exported.save("encoder2d.pt2")
    print("Saved encoder2d.pt2")


if __name__ == "__main__":
    main()
