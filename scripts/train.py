"""
Unified training script for TD3, SAC, and PPO.

Usage:
    python scripts/train.py --algo td3 --env HalfCheetah-v5 --seed 0
    python scripts/train.py --algo sac --env Hopper-v5 --seed 2 --config configs/sac_default.yaml
    python scripts/train.py --algo ppo --env Walker2d-v5 --seed 1

Results are saved to:
    results/{exp_group}/{algo}/{env}/seed_{seed}/
        training_log.csv
        eval_log.csv
        config.yaml
        final_model.pt

This script handles TD3 and SAC (off-policy) and PPO (on-policy) with the same
outer loop structure, diverging only in how actions are selected and how/when updates happen.
"""

import argparse
import os
import sys
import yaml
import numpy as np

# Make sure we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.utils import set_seed, make_env, make_ppo_env, evaluate_policy, get_device
from src.common.logger import TrainingLogger


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_default_config_path(algo: str) -> str:
    return os.path.join("configs", f"{algo}_default.yaml")


def build_agent(algo, state_dim, action_dim, max_action, config, device):
    """Instantiate the correct agent class based on algo name."""
    if algo == "td3":
        from src.td3.td3 import TD3
        return TD3(state_dim, action_dim, max_action, config, device)
    elif algo == "sac":
        from src.sac.sac import SAC
        return SAC(state_dim, action_dim, max_action, config, device)
    elif algo == "ppo":
        from src.ppo.ppo import PPO
        return PPO(state_dim, action_dim, config, device)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def train_off_policy(algo, agent, env, config, logger, total_timesteps, eval_freq, env_name):
    """
    Training loop for off-policy algorithms (TD3, SAC).

    Collect one transition per step, add to replay buffer, update once per step
    (after learning_starts transitions have been gathered).
    """
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_num    = 0

    # Track loss history for episodic logging
    critic_losses = []
    actor_losses  = []
    alpha_losses  = []
    alphas        = []

    for t in range(1, total_timesteps + 1):
        # Random exploration before filling buffer, then use policy
        if t < config["learning_starts"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False)

        next_state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        episode_length += 1

        # Store transition
        # Important: use 'done and not truncated' so we don't treat truncation as a terminal
        # (truncation = episode time limit, not actual termination of the MDP)
        real_done = done and not truncated
        agent.replay_buffer.add(state, action, reward, next_state, real_done)
        state = next_state

        # Update agent (once per step after warmup)
        if t >= config["learning_starts"]:
            if algo == "td3":
                critic_loss, actor_loss = agent.train_step()
                if critic_loss is not None:
                    critic_losses.append(critic_loss)
                if actor_loss is not None:
                    actor_losses.append(actor_loss)
            elif algo == "sac":
                critic_loss, actor_loss, alpha_loss, alpha = agent.train_step()
                if critic_loss is not None:
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)
                    alpha_losses.append(alpha_loss)
                    alphas.append(alpha)

        # Handle episode end
        if done or truncated:
            # Average losses over the episode for logging
            avg_critic = np.mean(critic_losses) if critic_losses else float("nan")
            avg_actor  = np.mean(actor_losses)  if actor_losses  else float("nan")

            log_kwargs = dict(
                timestep=t,
                episode=episode_num,
                episode_reward=round(episode_reward, 3),
                episode_length=episode_length,
                critic_loss=round(avg_critic, 6) if not np.isnan(avg_critic) else "",
                actor_loss=round(avg_actor, 6)   if not np.isnan(avg_actor)  else "",
            )
            if algo == "sac" and alphas:
                log_kwargs["alpha_loss"] = round(np.mean(alpha_losses), 6)
                log_kwargs["alpha"]      = round(alphas[-1], 6)

            logger.log_episode(**log_kwargs)

            # Reset
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_num   += 1
            critic_losses  = []
            actor_losses   = []
            alpha_losses   = []
            alphas         = []

        # Periodic evaluation
        if t % eval_freq == 0:
            mean_r, std_r = evaluate_policy(env_name, agent, n_episodes=config["eval_episodes"])
            # Also compute min/max across eval episodes for completeness
            # (evaluate_policy returns mean/std, re-run for min/max isn't worth the time)
            logger.log_eval(t, mean_r, std_r, mean_r - std_r, mean_r + std_r)
            print(f"  Step {t:>8d} | Eval reward: {mean_r:>8.1f} ± {std_r:.1f}")


def train_ppo(agent, env, config, logger, total_timesteps, eval_freq, env_name):
    """
    Training loop for PPO (on-policy).

    Collect rollout_length steps, compute GAE, do n_epochs of updates, repeat.
    Everything is grouped into rollouts, not individual steps.
    """
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_num    = 0
    total_steps    = 0

    # Initialise so finish_rollout always has valid values even if the first
    # rollout somehow completes in zero steps (shouldn't happen, but safe).
    done      = False
    truncated = False
    last_next_state = state

    while total_steps < total_timesteps:
        # ── Collect a rollout ─────────────────────────────────────────────────
        agent.buffer.reset()
        rollout_rewards  = []
        current_ep_rew   = 0.0

        for _ in range(config["rollout_length"]):
            action, log_prob, value = agent.select_action(state, evaluate=False)
            next_state, reward, done, truncated, _ = env.step(action)

            # Save next_state NOW, before it might be overwritten by env.reset()
            # below.  finish_rollout needs the actual final observation to
            # bootstrap V(s_{T+1}) for truncated (time-limit) episodes.
            last_next_state = next_state

            # Use done OR truncated as the episode-boundary flag for GAE.
            # When a truncation happens mid-rollout, the next stored value
            # (values[t+1]) is V(first obs of the new episode) — bootstrapping
            # from that wrong state would be worse than zeroing the bootstrap.
            # Treating truncation as terminal (zeroing bootstrap) introduces a
            # small bias but is consistent and matches CleanRL's approach.
            # The FINAL step of the rollout is handled separately in
            # finish_rollout(), which receives the actual last next_state and
            # correctly bootstraps V(s_{T+1}) for truncated rollout endings.
            agent.store_transition(state, action, reward, done or truncated, log_prob, value)

            state = next_state
            total_steps       += 1
            episode_reward    += reward
            episode_length    += 1
            current_ep_rew    += reward

            if done or truncated:
                rollout_rewards.append(episode_reward)
                logger.log_episode(
                    timestep=total_steps,
                    episode=episode_num,
                    episode_reward=round(episode_reward, 3),
                    episode_length=episode_length,
                )
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_num   += 1
                current_ep_rew = 0.0

            # Periodic evaluation mid-rollout
            if total_steps % eval_freq == 0:
                mean_r, std_r = evaluate_policy(env_name, agent, n_episodes=config["eval_episodes"])
                logger.log_eval(total_steps, mean_r, std_r, mean_r - std_r, mean_r + std_r)
                print(f"  Step {total_steps:>8d} | Eval reward: {mean_r:>8.1f} ± {std_r:.1f}")

        # ── Compute GAE and update ────────────────────────────────────────────
        # Pass last_next_state (the real s_{T+1}, not the post-reset obs) and
        # terminal=done and not truncated so truncated episodes are bootstrapped.
        agent.finish_rollout(last_next_state, done and not truncated)
        policy_loss, value_loss, entropy, approx_kl = agent.update()

        if rollout_rewards:
            print(f"  Step {total_steps:>8d} | Rollout mean ep reward: {np.mean(rollout_rewards):.1f}"
                  f" | policy_loss: {policy_loss:.4f} | kl: {approx_kl:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train an RL agent")
    parser.add_argument("--algo",   type=str, required=True,  choices=["td3", "sac", "ppo"],
                        help="Algorithm to train")
    parser.add_argument("--env",    type=str, required=True,
                        help="Gymnasium environment id (e.g. HalfCheetah-v5)")
    parser.add_argument("--seed",   type=int, default=0,
                        help="Random seed")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (default: configs/{algo}_default.yaml)")
    parser.add_argument("--exp-group", type=str, default="main",
                        help="Experiment group (e.g. 'main' or 'sensitivity/td3_lr')")
    # Allow overriding individual config values from the command line
    # e.g. --override learning_rate=1e-3
    # Using action="append" so multiple --override flags all accumulate correctly
    parser.add_argument("--override", type=str, action="append", default=[],
                        help="Override config values: key=value pairs (can be repeated)")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = args.config or get_default_config_path(args.algo)
    config = load_config(config_path)

    # Apply command-line overrides (e.g. for sensitivity sweeps)
    for kv in args.override:
        key, value_str = kv.split("=", 1)
        # Try to parse as number, else keep as string
        try:
            value = float(value_str)
            if value == int(value):
                value = int(value)
        except ValueError:
            value = value_str
        config[key] = value
        print(f"Override: {key} = {value}")

    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(args.seed)
    device = get_device()
    print(f"\nTraining {args.algo.upper()} on {args.env} | seed={args.seed} | device={device}")
    print(f"Config: {config_path}")

    # Create environment — PPO uses a reward-normalised wrapper to keep
    # the value-function loss at a tractable scale throughout training.
    # TD3 and SAC are off-policy and don't need this.
    if args.algo == "ppo":
        env = make_ppo_env(args.env, args.seed, gamma=config.get("gamma", 0.99))
    else:
        env = make_env(args.env, args.seed)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"State dim: {state_dim} | Action dim: {action_dim} | Max action: {max_action}")

    # ── Build agent ───────────────────────────────────────────────────────────
    agent = build_agent(args.algo, state_dim, action_dim, max_action, config, device)

    # ── Setup results directory ───────────────────────────────────────────────
    result_dir = os.path.join("results", args.exp_group, args.algo, args.env, f"seed_{args.seed}")
    os.makedirs(result_dir, exist_ok=True)

    # Save config alongside results so runs are fully reproducible
    config_save_path = os.path.join(result_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump({"algo": args.algo, "env": args.env, "seed": args.seed, **config}, f)

    # ── Setup logger ──────────────────────────────────────────────────────────
    logger = TrainingLogger(log_dir=result_dir, algo=args.algo)

    total_timesteps = config["total_timesteps"]
    eval_freq       = config["eval_freq"]

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        if args.algo in ("td3", "sac"):
            train_off_policy(args.algo, agent, env, config, logger, total_timesteps, eval_freq, args.env)
        elif args.algo == "ppo":
            train_ppo(agent, env, config, logger, total_timesteps, eval_freq, args.env)
    finally:
        # Always save model and close logger, even if training crashes partway
        model_path = os.path.join(result_dir, "final_model.pt")
        agent.save(model_path)
        logger.close()
        env.close()
        print(f"\nDone. Results saved to: {result_dir}")


if __name__ == "__main__":
    main()
