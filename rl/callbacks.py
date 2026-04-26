# rl/callbacks.py
#
# Custom Stable-Baselines3 callbacks for this project.
#
# WHY THIS FILE EXISTS:
#   The env's `step()` puts three extra scalars on `info` every step:
#     reward_mean — average V5 reward across all N² intersections
#     reward_min  — worst intersection's V5 reward this step
#     reward_max  — best intersection's V5 reward this step
#   These are the diagnostic hooks we added when ratifying mean reward
#   aggregation (see PENDING_DISCUSSIONS.md #1). The whole point was to
#   spot the case where the agent keeps reward_mean looking healthy by
#   ignoring a brutally congested intersection whose reward_min stays
#   pinned at the floor.
#
#   SB3 doesn't surface those automatically. Monitor passes them through
#   in info, but nothing in SB3's default logging reads them. So we write
#   a small BaseCallback that pulls them out on each step and calls
#   `self.logger.record_mean()`, which TensorBoard then plots alongside
#   `rollout/ep_rew_mean` and friends.
#
# HOW IT HOOKS IN:
#   In rl/train.py we wrap this into the CallbackList that's passed to
#   model.learn(). SB3 calls `_on_step()` after every env step across the
#   whole run, so the values stream into TensorBoard live.

from stable_baselines3.common.callbacks import BaseCallback


class TensorBoardRewardDiagnostics(BaseCallback):
    """
    Forwards per-intersection reward diagnostics from env.info to TensorBoard.

    LOGS (all under the `custom/` namespace so they don't collide with SB3's
    built-in `rollout/`, `train/`, `eval/` groups):
      custom/reward_mean — mean V5 reward across intersections, this step
      custom/reward_min  — worst-intersection V5 reward, this step
      custom/reward_max  — best-intersection V5 reward, this step
      custom/reward_gap  — reward_mean minus reward_min (how much the
                           agent is averaging away; if this gap stays wide
                           and the min stays low, that's our "blind to a
                           worst link" signal — the cue to consider a
                           mean + λ·worst ablation later)

    AGGREGATION:
      `record_mean` averages values across steps since the last TB dump
      (SB3 dumps every `log_interval` rollouts). That's the right choice
      for a reward signal — one-off noisy spikes aren't interesting, the
      rolling behavior is.

    COST:
      One dict lookup + 4 scalar writes per step. Negligible vs the env
      step itself.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # SB3 passes the per-env info list via self.locals["infos"] during
        # on-policy rollouts (PPO/A2C) and off-policy collection (DQN).
        # With a single Monitor-wrapped env (our case), this is a 1-element
        # list — but we iterate defensively in case someone wraps the env
        # in DummyVecEnv/SubprocVecEnv later for parallelism.
        infos = self.locals.get("infos", [])
        for info in infos:
            # env.step() only populates these keys on the REAL env step, not
            # on VecEnv terminal-reset short-circuit steps — guard with `in`.
            if "reward_mean" in info:
                r_mean = info["reward_mean"]
                r_min  = info["reward_min"]
                r_max  = info["reward_max"]
                self.logger.record_mean("custom/reward_mean", r_mean)
                self.logger.record_mean("custom/reward_min",  r_min)
                self.logger.record_mean("custom/reward_max",  r_max)
                # `reward_gap` is a derived sanity metric — the distance
                # between "average pain" and "worst pain". A shrinking gap
                # during training = the agent is learning to lift the
                # floor; a wide, stuck gap = mean is masking a bad link.
                self.logger.record_mean("custom/reward_gap",  r_mean - r_min)
            # V6 throughput-bonus diagnostics. Logged separately from V5 so
            # we can read TensorBoard and answer "is the agent winning by
            # clearing queues, or by serving more demand?" — and spot the
            # bad case where v5 keeps drifting up while throughput rots
            # (the throttle exploit reasserting itself).
            if "reward_v5" in info:
                self.logger.record_mean("custom/reward_v5",         info["reward_v5"])
                self.logger.record_mean("custom/reward_throughput", info["reward_throughput"])
                self.logger.record_mean("custom/delta_spawned",     info["delta_spawned"])
        return True  # False would abort training; we just observe.
