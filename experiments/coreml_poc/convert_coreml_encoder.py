import torch
import coremltools as ct


def main():
    exported = torch.export.load("encoder2d.pt2")
    example = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    mlmodel = ct.convert(exported, inputs=[ct.TensorType(shape=example.shape)])
    cfg = ct.models.MLModelConfiguration()
    cfg.compute_units = ct.ComputeUnit.CPU_AND_NE
    mlmodel = ct.models.MLModel(mlmodel.get_spec(), configuration=cfg)
    mlmodel.save("encoder2d.mlpackage")
    print("Saved encoder2d.mlpackage")


if __name__ == "__main__":
    main()
