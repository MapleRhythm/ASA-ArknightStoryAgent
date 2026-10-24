"""Read-only check that adapter + AdamW checkpoint can be loaded for resuming."""
import argparse
import json
from pathlib import Path

import torch
from backend import PromptEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    state = torch.load(run / "training_state.pt", map_location="cpu", weights_only=True)
    encoder = PromptEncoder.load(run / f"checkpoint-{state['step']}", device="cpu")
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-5)
    optimizer.load_state_dict(state["optimizer"])
    assert len(optimizer.state) == len(list(encoder.parameters()))
    for p, opt_state in optimizer.state.items():
        assert opt_state["step"].item() == state["step"]
        for name in ("exp_avg", "exp_avg_sq"):
            assert opt_state[name].shape == p.shape and torch.isfinite(opt_state[name]).all()
    assert torch.isfinite(encoder()).all()
    assert state["torch_rng"].dtype == state["cuda_rng"].dtype == torch.uint8
    print(json.dumps({"resume_checkpoint_load_verified": True, "step": state["step"],
                      "optimizer_moments_finite": True,
                      "adapter_output_finite": True,
                      "training_job_interrupted": False,
                      "full_interrupted_resume_numerical_equivalence_not_tested": True}), flush=True)


if __name__ == "__main__":
    main()
