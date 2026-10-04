"""Print (or explicitly execute) a reconstructed VeRL incident training command."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def build_command(phase, verl_dir, data_dir, run_dir, model, python=sys.executable):
    root = Path(run_dir).resolve()
    data = Path(data_dir).resolve()
    reward = Path(__file__).resolve().parent / "historical/faulty_reward.py"
    configs = {
        "A": ("phaseA", "correct", 60, None),
        "B": ("phaseB", "len_decay", 130, root / "phaseA/ckpts/global_step_60"),
        "REC": ("recovery_rec", "correct", 100, root / "phaseA/ckpts/global_step_60"),
        "RECLAST": ("recovery_reclast", "correct", 170, root / "phaseB/ckpts/global_step_130"),
    }
    name, verifier, total, resume = configs[phase]
    output = root / name
    options = [
        "algorithm.adv_estimator=grpo",
        f"data.train_files={data / 'train.parquet'}",
        f"data.val_files={data / 'test.parquet'}",
        "data.train_batch_size=64", "data.max_prompt_length=512",
        "data.max_response_length=1024", f"actor_rollout_ref.model.path={model}",
        "actor_rollout_ref.model.use_remove_padding=false",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.rollout.name=vllm", "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        f"reward.custom_reward_function.path={reward}",
        "reward.custom_reward_function.name=compute_score",
        'trainer.logger=["console"]', "trainer.project_name=rtd_incident",
        f"trainer.experiment_name={name}", "trainer.val_before_train=False",
        f"trainer.default_local_dir={output / 'ckpts'}",
        f"trainer.rollout_data_dir={output / 'rollouts'}",
        "trainer.n_gpus_per_node=1", "trainer.nnodes=1",
        "trainer.save_freq=10", "trainer.test_freq=-1",
        f"trainer.total_training_steps={total}", "trainer.total_epochs=100",
    ]
    if resume is not None:
        options.extend(["trainer.resume_mode=resume_path", f"trainer.resume_from_path={resume}"])
    else:
        options.append("trainer.resume_mode=disable")
    return [python, "-m", "verl.trainer.main_ppo", *options], verifier, output, resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["A", "B", "REC", "RECLAST"])
    for name in ["verl-dir", "data-dir", "run-dir", "model"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true", help="Actually start GPU training")
    args = parser.parse_args()
    command, verifier, output, resume = build_command(
        args.phase, args.verl_dir, args.data_dir, args.run_dir, args.model, args.python)
    print(f"Working directory: {Path(args.verl_dir).resolve()}", flush=True)
    print(f"LOOM_VERIFIER={verifier} " + shlex.join(command), flush=True)
    if not args.execute:
        print("Dry run only. No training was started.")
        return
    if not (Path(args.verl_dir) / "verl/trainer/main_ppo.py").is_file():
        parser.error("--verl-dir must contain a compatible VeRL checkout")
    if any(not (Path(args.data_dir) / name).is_file() for name in ["train.parquet", "test.parquet"]):
        parser.error("--data-dir must contain train.parquet and test.parquet")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("the phase output must be new or empty")
    if resume is not None and not resume.is_dir():
        parser.error(f"missing required input checkpoint: {resume}")
    environment = dict(os.environ, LOOM_VERIFIER=verifier, PYTHONUNBUFFERED="1")
    subprocess.run(command, cwd=args.verl_dir, env=environment, check=True)


if __name__ == "__main__":
    main()
