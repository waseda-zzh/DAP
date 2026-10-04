# DAP: Dynamic Adversarial Patch (training code)

This repository contains the training code of the patch generation pipeline described in

> Zhihe Zhang, Shuo Peng, and Tatsuya Mori.
> **DAP: Physically Realizable and Controllable Dynamic Adversarial Patch Attack Against Feature-based Visual Localization.**
> IEICE Transactions on Information and Systems (under review).

## Pipeline

The pipeline has three components. They are trained offline in the order below (Section 3.3.4 of the paper).

| Stage | Component | Directory | Paper | Entry point |
|---|---|---|---|---|
| 1 | Differentiable feature proxy | `orb_proxy/` | Sec. 3.3.1, Eq. (8) | `python -m orb_proxy.train_orb_proxy_kitti --config <yaml>` |
| 2 | Pattern-guided generator | `pattern_unet/` | Sec. 3.3.2, Eqs. (9)–(12) | `python -m pattern_unet.train --config <yaml>` |
| 3 | Context-aware adapter | `env_adapter/` | Sec. 3.3.3, Eqs. (13)–(14) | `python -m env_adapter.train --config <yaml>` |

1. **Feature proxy.** A lightweight CNN is trained to regress the fraction of ORB keypoints that fall inside the patch region of a composed image. Once trained, it is frozen and serves as a differentiable estimate of feature attraction.
2. **Pattern-guided generator.** A U-Net fuses a corner-rich, feature-inducing template into an ordinary advertisement image, trained with content, style, frequency and total-variation losses. Its output is the fused patch.
3. **Context-aware adapter.** A scene-conditioned modulator adjusts the fused patch for a given deployment scene. It is optimized with the frozen proxy together with gradient, luminance and display-gamut losses.

Run the commands from the repository root so that the three packages are importable. Stage 3 loads the proxy trained in stage 1 and keeps it frozen.

## What is not included

- **Trained models.** No checkpoints of the proxy, the generator or the adapter are released.
- **Configuration files.** Each training script reads a YAML file passed with `--config`. The fields it expects are those parsed in `load_train_config` (`orb_proxy/train_orb_proxy_kitti.py`) and in `main()` (`pattern_unet/train.py`, `env_adapter/train.py`).
- **Data.** Stage 1 uses KITTI odometry sequences, stage 2 uses advertisement images and a feature-inducing template, and stage 3 uses CARLA frames with the homography of the billboard region. These are not redistributed here.

## Requirements

Python 3.10 or later and a CUDA-capable GPU are recommended.

```bash
pip install -r requirements.txt
```

## Responsible use

This code is released to support research on the security of visual localization. Use it only for research and for evaluating defenses, and only on systems you own or are authorized to test.

## License

MIT License. See `LICENSE`.
