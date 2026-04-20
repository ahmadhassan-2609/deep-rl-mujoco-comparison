"""
Simple CSV-based logging for training runs.

I'm avoiding TensorBoard or Weights & Biases to keep things lightweight
and portable. CSVs can be loaded with pandas for plotting afterward.

Two separate logs:
  - training_log.csv : per-update metrics (losses, alpha, etc.)
  - eval_log.csv     : periodic evaluation results (mean/std reward)
"""

import csv
import os
import time


class TrainingLogger:
    """
    Logs training metrics to a CSV file as training progresses.

    One row is written every time log_training() is called — typically
    once per environment step or once per gradient update.
    """

    def __init__(self, log_dir: str, algo: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.start_time = time.time()

        train_path = os.path.join(log_dir, "training_log.csv")
        eval_path  = os.path.join(log_dir, "eval_log.csv")

        # Training log: one row per episode
        self.train_file = open(train_path, "w", newline="")
        train_fields = ["timestep", "episode", "episode_reward", "episode_length", "wall_time"]

        # Add algo-specific loss fields
        if algo == "td3":
            train_fields += ["critic_loss", "actor_loss"]
        elif algo == "sac":
            train_fields += ["critic_loss", "actor_loss", "alpha_loss", "alpha"]
        elif algo == "ppo":
            train_fields += ["policy_loss", "value_loss", "entropy", "approx_kl"]

        self.train_writer = csv.DictWriter(self.train_file, fieldnames=train_fields)
        self.train_writer.writeheader()
        self.train_fields = set(train_fields)

        # Evaluation log: one row per evaluation
        self.eval_file = open(eval_path, "w", newline="")
        eval_fields = ["timestep", "mean_reward", "std_reward", "min_reward", "max_reward"]
        self.eval_writer = csv.DictWriter(self.eval_file, fieldnames=eval_fields)
        self.eval_writer.writeheader()

    def log_episode(self, **kwargs):
        """Log end-of-episode metrics. Pass any subset of the training_log fields."""
        kwargs["wall_time"] = round(time.time() - self.start_time, 2)
        # Only write fields that exist in the schema (ignore extras)
        row = {k: v for k, v in kwargs.items() if k in self.train_fields}
        self.train_writer.writerow(row)
        self.train_file.flush()  # flush so we can read the file during training

    def log_eval(self, timestep, mean_reward, std_reward, min_reward, max_reward):
        """Log evaluation results."""
        self.eval_writer.writerow({
            "timestep":    timestep,
            "mean_reward": round(mean_reward, 4),
            "std_reward":  round(std_reward, 4),
            "min_reward":  round(min_reward, 4),
            "max_reward":  round(max_reward, 4),
        })
        self.eval_file.flush()

    def close(self):
        self.train_file.close()
        self.eval_file.close()
