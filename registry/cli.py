"""Model registry commands.

    python -m registry register --model checkpoints/lr_0005.pth   # needs a passed evaluation + gate
    python -m registry show
    python -m registry promote --version 2
    python -m registry check                                      # load the champion and run a prediction
    python -m registry export                                     # copy the champion to model_export/ (for Docker)
"""
import argparse

from training.run_config import apply_overrides, load_run_config
from registry.ops import RegistryError, export_champion, list_versions, load_champion, promote, register


def _show(run_cfg) -> int:
    info = list_versions(run_cfg)
    if not info["versions"]:
        print(f"No versions registered under '{info['name']}'.")
        return 0
    print(f"Registered model: {info['name']}   aliases: {info['aliases'] or 'none'}")
    print(f"{'version':<8}{'aliases':<12}{'test_acc':<10}{'macro_f1':<10}{'gate':<8}checkpoint")
    for r in info["versions"]:
        print(f"{r['version']:<8}{','.join(r['aliases']) or '-':<12}{r['test_accuracy'] or '-':<10}"
              f"{r['test_macro_f1'] or '-':<10}{r['quality_gate'] or '-':<8}{r['source_checkpoint'] or '-'}")
    return 0


def _check(run_cfg) -> int:
    import numpy as np
    from config import CONFIG
    model = load_champion(run_cfg)
    size = CONFIG["image_size"]
    probs = model.predict(np.zeros((1, 3, size, size), dtype=np.float32))
    classes = model.unwrap_python_model().classes
    print(f"Champion loaded and served a prediction: output shape {tuple(probs.shape)}, "
          f"{len(classes)} classes, probabilities sum to {float(probs.sum()):.3f}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m registry", description="Model registry commands.")
    parser.add_argument("--config", default=None, help="run config YAML (default: configs/training.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="register a checkpoint that passed the quality gate")
    reg.add_argument("--model", required=True)
    reg.add_argument("--run-id", default=None, help="MLflow run to attach the model to")
    reg.add_argument("--eval-dir", default=None, help="default: reports/evaluation/<model name>")
    pro = sub.add_parser("promote", help="point the champion alias at a registered version")
    pro.add_argument("--version", required=True)
    sub.add_parser("show", help="list registered versions and aliases")
    sub.add_parser("check", help="load the champion through the alias and run a test prediction")
    exp = sub.add_parser("export", help="copy the champion model to a folder (used to build the Docker image)")
    exp.add_argument("--dir", default="model_export")
    args = parser.parse_args(argv)

    run_cfg = load_run_config(args.config)
    apply_overrides(run_cfg["config_overrides"])
    try:
        if args.command == "register":
            r = register(args.model, run_cfg, args.run_id, args.eval_dir)
            if r["status"] == "already_registered":
                print(f"Already registered as {r['name']} version {r['version']}; nothing to do.")
            else:
                print(f"Registered {r['name']} version {r['version']} (run {r['run_id']}).")
                print("It is now the champion." if r["became_champion"] else
                      "The champion alias was not changed. Promote it explicitly with: "
                      f"python -m registry promote --version {r['version']}")
            return 0
        if args.command == "promote":
            r = promote(args.version, run_cfg)
            print(f"{r['alias']} now points to version {r['version']} (previously {r['previous']}).")
            return 0
        if args.command == "export":
            r = export_champion(run_cfg, args.dir)
            print(f"Exported {r['name']} version {r['version']} ({r['alias']}) to {args.dir}/")
            return 0
        return _show(run_cfg) if args.command == "show" else _check(run_cfg)
    except RegistryError as exc:
        print(exc)
        return 1
