"""Behavioral tests for FARR training and inference."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from farr.backbone import alignment_condition, endpoint_pass
from farr.data_loader import EndpointDataset, prepare_raw_splits
from farr.evaluate import load_checkpoint, save_checkpoint
from farr.farr import make_model
from farr.features import build_features
from farr.protocol import PaperProtocol
import farr.train as training


class MethodChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.protocol = PaperProtocol(horizon=96)

    def test_paper_defaults_and_parameter_counts(self):
        p = self.protocol
        self.assertEqual((p.epochs_stage1, p.epochs_stage2), (10, 5))
        self.assertEqual((p.patience_stage1, p.patience_stage2), (2, 2))
        for horizon in (96, 192, 336, 720):
            model = make_model(replace(p, horizon=horizon))
            self.assertEqual(sum(x.numel() for x in model.refiner.parameters()), 19073)
            self.assertEqual(sum(x.numel() for x in model.parameters()), 19073+3*horizon)

    def test_sign_consistency_is_absolute(self):
        history = torch.arange(672, dtype=torch.float32).view(1, 672, 1)
        valid, null = torch.zeros(1, 96, 1), torch.ones(1, 96, 1)
        features, _ = build_features(history, null, valid)
        self.assertEqual(features.shape, (1, 96, 18))
        self.assertTrue(torch.equal(features[..., 16], torch.ones_like(valid[..., 0])))
        self.assertTrue(torch.equal(features[:, 0, 13], torch.zeros(1)))
        self.assertTrue(torch.allclose(features[..., 3], torch.ones_like(valid[..., 0])))

    def test_joint_checkpoint_rule_rejects_mae_regression(self):
        model = make_model(self.protocol)
        protocol = replace(self.protocol, epochs_stage1=3)
        candidates = [{"mse": 5., "mae": 4.}, {"mse": 3., "mae": 5.},
                      {"mse": 4., "mae": 3.}]
        with patch.object(training, "loader", return_value=[]), \
             patch.object(training, "evaluate", side_effect=candidates):
            _, selected = training._train_phase(model, None, None, protocol,
                                                torch.device("cpu"), "stage1")
        self.assertEqual(selected, candidates[-1])
        with patch.object(training, "loader", return_value=[]), \
             patch.object(training, "evaluate", side_effect=candidates[:2]):
            _, selected = training._train_phase(model, None, None,
                                                replace(protocol, epochs_stage1=2),
                                                torch.device("cpu"), "stage1")
        self.assertEqual(selected, candidates[0])

    def test_valid_fallback_and_stage2_promotion(self):
        anchor = {"mse": 4., "mae": 4.}
        def phase(model, *args):
            return {key: value.clone() for key, value in model.state_dict().items()}, {"mse": 5., "mae": 5.}
        with patch.object(training, "evaluate", return_value=anchor), \
             patch.object(training, "_train_phase", side_effect=phase) as mocked:
            result = training.fit(None, None, self.protocol, "cpu")
        self.assertEqual(result.stage, "valid")
        self.assertEqual(mocked.call_count, 1)
        for stage2_passes in (False, True):
            def stages(model, *args):
                with torch.no_grad():
                    model.refiner.net[-1].bias.fill_(1. if args[-1] == "stage1" else 2.)
                values = ({"mse": 3., "mae": 3.} if args[-1] == "stage1"
                          else {"mse": 2., "mae": 2.} if stage2_passes
                          else {"mse": 5., "mae": 2.})
                return {k: v.clone() for k, v in model.state_dict().items()}, values
            with patch.object(training, "evaluate", return_value=anchor), \
                 patch.object(training, "_train_phase", side_effect=stages):
                result = training.fit(None, None, self.protocol, "cpu")
            self.assertEqual(result.stage, "stage2" if stage2_passes else "stage1")
            self.assertEqual(float(result.model.refiner.net[-1].bias.detach()[0]),
                             2. if stage2_passes else 1.)

    def test_stage2_recomputed_features_and_target_gradient(self):
        model = make_model(self.protocol)
        with torch.no_grad():
            model.adjustment.b.fill_(1.)
        batch = {"history": torch.randn(2, 672, 1), "y_valid": torch.ones(2, 96, 1),
                 "y_null": torch.zeros(2, 96, 1)}
        with patch.object(training, "build_features", wraps=build_features) as builder:
            prediction, residual, effective = training.forward(model, batch, self.protocol, "stage2")
        self.assertTrue(torch.allclose(builder.call_args.args[2], effective))
        self.assertFalse(torch.equal(effective, batch["y_valid"]))
        residual = torch.zeros_like(effective)
        fit = training.loss_terms(torch.zeros_like(effective), residual, effective,
                                  torch.zeros_like(effective), self.protocol)
        gradient, = torch.autograd.grad(fit, model.adjustment.b)
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_multichannel_bundles_become_single_channel_windows(self):
        with tempfile.TemporaryDirectory(prefix="farr-check-") as folder:
            root = Path(folder)
            for name, length in (("history", 672), ("y_valid", 96), ("y_null", 96), ("y_true", 96)):
                value = np.zeros((2, length, 3), dtype=np.float32)
                value[..., 1], value[..., 2] = 1., 2.
                np.save(root/f"{name}.npy", value)
            dataset = EndpointDataset(root, self.protocol)
            self.assertEqual(len(dataset), 6)
            self.assertEqual(dataset[2]["history"].shape, (672, 1))
            self.assertTrue(torch.equal(dataset[2]["history"], torch.ones(672, 1)))
            del dataset  # Close Windows memory-map handles before temp cleanup.

    def test_scaler_fits_train_and_targets_start_after_history(self):
        import pandas as pd
        with tempfile.TemporaryDirectory(prefix="farr-check-") as folder:
            raw = np.concatenate([np.arange(1400, dtype=np.float64), np.full(600, 10000.)])
            frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2000, freq="h"),
                                  "channel": raw})
            path = Path(folder)/"synthetic.csv"
            frame.to_csv(path, index=False)
            splits = prepare_raw_splits(path, "Weather", self.protocol)
            self.assertAlmostEqual(float(splits["train"].values.mean()), 0., places=6)
            self.assertAlmostEqual(float(splits["train"].values.std()), 1., places=6)
            expected = (raw[672]-raw[:1400].mean())/raw[:1400].std()
            self.assertAlmostEqual(float(splits["train"][0]["y_true"][0, 0]), float(expected), places=6)

    def test_checkpoint_restores_stage_and_predictions(self):
        model = make_model(self.protocol)
        batch = {"history": torch.randn(2, 672, 1), "y_valid": torch.ones(2, 96, 1),
                 "y_null": torch.zeros(2, 96, 1)}
        with torch.no_grad():
            model.adjustment.b.fill_(1.)
        for stage in ("valid", "stage1", "stage2"):
            result = training.TrainingResult(model, stage, {})
            with tempfile.TemporaryDirectory(prefix="farr-check-") as folder:
                path = Path(folder)/"local.pt"
                save_checkpoint(result, self.protocol, path)
                loaded, protocol, device = load_checkpoint(path, "cpu")
                self.assertEqual(loaded.stage, stage)
                a, _, _ = training.forward(model, batch, self.protocol, stage)
                b, _, _ = training.forward(loaded.model, batch, protocol, loaded.stage)
                self.assertTrue(torch.equal(a, b))

    def test_null_hook_and_seed_pairing(self):
        class Attention(torch.nn.Module):
            def forward(self, history):
                return history + torch.rand_like(history)
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fuser = torch.nn.Module()
                self.fuser.attention = Attention()
            def forward(self, x, *args):
                aligned = self.fuser.attention(x)
                return x + aligned
        model = Backbone()
        x = torch.ones(1, 672, 1)
        with alignment_condition(model, "null"):
            self.assertTrue(torch.equal(model(x), x))
        self.assertFalse(torch.equal(model(x), x))
        batches = [{"history": x}]
        a = list(endpoint_pass(model, batches, "valid", self.protocol, torch.device("cpu")))[0]
        b = list(endpoint_pass(model, batches, "valid", self.protocol, torch.device("cpu")))[0]
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
