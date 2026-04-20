# PPO Clipping Stability Reproduction

This is a reproduction study of the core stability claim from Schulman et al. (2017), "Proximal Policy Optimization Algorithms." This project compares the unclipped surrogate objective (L_CPI) against PPO's clipped objective (L_CLIP) across four discrete-action environments to evaluate the effect of clipping on training stability.

---

## Reproduction Summary

### Claim Being Tested

This project reproduces the core stability claim from Schulman et al. (2017), corresponding to **Table 1** of the original paper. Table 1 compares several surrogate objectives and shows that the clipped objective (L_CLIP) outperforms the unclipped surrogate (L_CPI) in terms of final performance and training stability across continuous control tasks.

### Hypothesis

Clipping the probability ratio in the PPO surrogate objective improves training stability and final performance compared to the unclipped surrogate (L_CPI), which serves as the ablation baseline from the original paper.

### Experimental Design

To keep experiments accessible on a single GPU, four discrete-action environments were selected instead of the continuous control benchmarks used in the original paper. These environments span a range of difficulty and reward structures, preserving the essence of the original comparison:

| Environment    | Role in Study                              |
|----------------|--------------------------------------------|
| CartPole-v1    | Simple, fast sanity check                  |
| Acrobot-v1     | Moderate difficulty, sparse reward         |
| LunarLander-v3 | Complex dynamics, shaped reward            |
| Taxi-v3        | Discrete state/action, sparse reward       |

The network architecture (two-layer 64-unit MLP with tanh activations) follows Schulman et al. (2017) Section 6.1. Training runs for 4 seeds per condition to allow statistical comparison.

### Fairness of Comparison

The clipped (PPO) and unclipped (L_CPI) conditions are identical in every respect except the objective function itself:

- Same network architecture and random seeds
- Same hyperparameters (learning rate, gamma, lambda, entropy coefficient)
- Same GAE advantage estimation
- Same minibatch update structure and number of update epochs
- Clipping is the **only** independent variable

---

## Code Structure

All code is contained in a single Jupyter notebook:

    ECE 570 Project.ipynb

The notebook is organized top to bottom in the following order:

1. Imports
2. Device configuration and hyperparameters
3. Seed utility
4. Network definitions (PolicyNet, ValueNet)
5. Rollout collection (collect_rollout)
6. Monte Carlo returns (compute_returns)
7. GAE computation (compute_gae)
8. PPO update function (update)
9. Training loop (train)
10. Smoothing utility (smooth)
11. Summarization function (summarize)
12. Plotting helpers (plot_rewards, plot_entropy)
13. Shared training loop (run_env)
14. Summary plot (plot_all_environments)
15. Per-environment run functions (run_cartpole, run_acrobot, run_lunarlander, run_taxi)
16. pip installs
17. Run flags
18. Main entry point (main)

---

## Dependencies

The following packages are required. They are either pre-installed in Google Colab or installed automatically by the notebook:

- Python 3.10+
- torch
- numpy
- matplotlib
- gymnasium
- gymnasium[box2d] (installed automatically by the notebook via pip)
- swig (installed automatically by the notebook via pip)

No additional installation is required before running.

---

## How to Run

1. Open ECE 570 Project.ipynb in Jupyter or Google Colab
2. (Recommended) Enable GPU: Runtime > Change runtime type > T4 GPU
3. Run all cells top to bottom: Runtime > Run all
4. Results will be printed and plots will be displayed inline

To run individual environments, set the corresponding flags in the configuration cell before running main():

    RUN_CARTPOLE = True   # or False to skip
    RUN_ACROBAT  = True
    RUN_LUNAR    = True
    RUN_TAXI     = True

---

## Environments

All environments are initialized automatically through the Gymnasium API.
No manual dataset or model downloads are required.

| Environment    | Description                          |
|----------------|--------------------------------------|
| CartPole-v1    | Pole balancing, max reward 500       |
| Acrobot-v1     | Two-link pendulum swing-up           |
| LunarLander-v3 | Spacecraft landing, discrete actions |
| Taxi-v3        | Sparse reward navigation             |

---

## Code Authorship

### External References

The following well-established open-source PPO implementations were used as references during development:

- CleanRL by Costa Huang
  https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
  Specifically:
  - Rollout collection loop (CleanRL lines 192–215): the step loop pattern collecting states, actions, rewards, log-probs, and dones into fixed-length buffers using torch.no_grad() during inference
  - GAE computation (CleanRL lines 222–231): the reversed-iteration backward accumulation loop computing delta and lastgaelam, and the returns = advantages + values assignment
  - Minibatch update structure (CleanRL lines 242–248): the shuffled index pattern (np.random.shuffle) and the inner for-loop slicing minibatches from the full rollout buffer

- OpenAI Spinning Up
  https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
  Specifically:
  - General actor-critic training loop structure (Spinning Up lines 298–337): the outer epoch loop pattern of collect rollout → finish_path (GAE) → update(), which is the structural backbone of the `train()` function in Cell 8
  - Separate policy and value network optimizers (Spinning Up lines 251–252): maintaining distinct Adam optimizers for the policy and value networks, updated independently in the update step

- PPO Implementation Details by Costa Huang
  https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py
  Note: This repo is by the same author as CleanRL and shares nearly identical code. It is cited separately because the accompanying blog post "The 37 Implementation Details of PPO" (https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) was used as an explicit reference for understanding the rationale behind each design choice.
  Specifically:
  - Advantage normalization (line 271): the `(mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)` pattern applied per minibatch before the policy loss computation
  - Minibatch shuffling loop (lines 251–257): `np.arange` index array, `np.random.shuffle`, and the inner `for start in range(0, batch_size, minibatch_size)` slicing pattern
  - Ratio clipping pattern (lines 274–276): `pg_loss1 = -advantages * ratio`, `pg_loss2 = -advantages * torch.clamp(ratio, 1-ε, 1+ε)`, `pg_loss = torch.max(pg_loss1, pg_loss2).mean()`
  - Entropy bonus (line 293): `entropy_loss = entropy.mean()` subtracted from the total loss weighted by entropy coefficient

- PPO-PyTorch by nikhilbarhate99
  https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
  Specifically:
  - Separate actor/critic class structure (lines 48–73): the pattern of defining `actor` and `critic` as separate `nn.Sequential` modules, adapted into the distinct `PolicyNet` and `ValueNet` classes in Cell 3
  - Discrete action space handling with Categorical distribution (lines 93–97): `dist = Categorical(action_probs)`, `action = dist.sample()`, `action_logprob = dist.log_prob(action)` — adapted into the rollout collection logic in Cell 4

- Stable Baselines3 — PPO
  https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
  Specifically:
  - Overall train loop architecture (lines 204–282): the `for epoch in range(self.n_epochs)` loop iterating over the rollout buffer, which informed the epoch loop structure in `update()` in Cell 7
  - Clipped surrogate loss (lines 225–227): `policy_loss_1 = advantages * ratio`, `policy_loss_2 = advantages * th.clamp(ratio, 1-clip_range, 1+clip_range)`, `policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()` — directly corresponds to the clipped objective in Cell 7
  - Value loss (line 244): `F.mse_loss(rollout_data.returns, values_pred)` pattern, adapted into `mse_loss(values, batch_returns)` in Cell 7
  - Rollout buffer concept: the pattern of a dedicated buffer storing states, actions, rewards, log_probs, and dones per rollout, adapted into the list-based collection in Cell 4

- Phil Tabor — PPO PyTorch
  https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
  Specifically:
  - Episode return tracking and reset on done (lines 122–123 `remember`, lines 135–146 `choose_action`): the pattern of accumulating per-step rewards and resetting episode tracking when `done=True`, adapted into lines 46–52 of Cell 4 (`episode_return += reward`, `episode_returns.append(episode_return)`, `episode_return = 0`, `env.reset()`)
  - Simple separate actor/critic class structure for discrete environments (lines 50–108): two separate network classes each with their own optimizer and Softmax output for discrete actions, which informed the `PolicyNet`/`ValueNet` separation in Cell 3

The network architecture (two-layer 64-unit MLP with tanh activations) and hyperparameters follow Schulman et al. (2017), Section 6.1 and Table 3.

---

### Cell-by-Cell Authorship

Note: Line numbers below refer to lines within each individual cell, not absolute notebook line numbers.
Adapted lines are cited only where code was borrowed or adapted from an external source.

|  Cell  | Content | Status |
|--------|---------|--------|
| Cell 0 | Imports | Written by author |
| Cell 1 | Device configuration + hyperparameters | Written by author; line 7 (`DEVICE = torch.device(...)`) adapted from standard PyTorch GPU boilerplate; lines 24–33 (LR, GAMMA, EPS_CLIP, LAMBDA, etc.) follow Schulman et al. (2017) Section 6.1 and Table 3 |
| Cell 2 | `set_seed()` | Written by author |
| Cell 3 | `PolicyNet`, `ValueNet` | Written by author; lines 14–37 (class definitions and two-layer 64-unit MLP with tanh architecture) adapted from Schulman et al. (2017) Section 6.1 and nikhilbarhate99/PPO-PyTorch (separate actor/critic class structure) |
| Cell 4 | `collect_rollout()` | Written by author; lines 25–52 (step loop collecting states, actions, rewards, log-probs, and dones into lists using `torch.no_grad()` during inference) adapted from CleanRL (ppo.py lines 192–215); lines 46–52 (episode return tracking and reset on done) adapted from Phil Tabor PPO |
| Cell 5 | `compute_returns()` | Written by author |
| Cell 6 | `compute_gae()` | Written by author; lines 27–30 (reversed-iteration loop computing `delta` and accumulating `last_adv`) follow Schulman et al. (2017) Equations 11–12 and are adapted from CleanRL (ppo.py lines 222–231) and PPO Implementation Details |
| Cell 7 | `update()` | Written by author; line 39 (advantage normalization: zero mean, unit std) adapted from CleanRL (ppo.py line 262) and PPO Implementation Details; lines 41–45 (minibatch shuffling loop using `torch.randperm` to slice minibatches from the full rollout) adapted from CleanRL (ppo.py lines 242–248) and PPO Implementation Details; lines 59–62 (clipped surrogate: `torch.clamp` / `torch.min` pattern) adapted from CleanRL (ppo.py lines 265–267), PPO Implementation Details, and Stable Baselines3, following Schulman et al. (2017) Equation 7; lines 67–69 (entropy bonus: `dist.entropy().mean()` weighted by entropy coefficient) adapted from PPO Implementation Details (line 293); lines 71–73 (value loss: MSE between predicted values and returns) adapted from Stable Baselines3 (line 244) |
| Cell 8 | `train()` | Written by author; lines 58–91 (epoch loop: collect → GAE → update → log pipeline) adapted from OpenAI Spinning Up (lines 298–337: outer epoch loop structure; lines 251–252: separate policy/value optimizer pattern) and CleanRL; per-environment epoch counts, discrete observation handling, and hyperparameter passing written by author |
| Cell 9 | `smooth()` | Written by author |
| Cell 10 | `summarize()` | Written by author; late-training std and worst-case collapse metrics are original contributions |
| Cell 11 | `plot_rewards()`, `plot_entropy()` | Written by author; individual seed curve visualization is an original contribution |
| Cell 12 | `run_env()` | Written by author |
| Cell 13 | `plot_all_environments()` | Written by author |
| Cell 14–17 | `run_cartpole()`, `run_acrobot()`, `run_lunarlander()`, `run_taxi()` | Written by author |
| Cell 18 | pip installs | Standard Colab setup |
| Cell 19 | Run flags | Written by author |
| Cell 20–21 | `main()`, entry point | Written by author |

---

## Reference

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

---

## Notes

- Training all four environments with 4 seeds each is computationally intensive
- Use the RUN_* flags to run individual environments during development
- GPU runtime is recommended for faster training (Runtime > Change runtime type > T4 GPU)
- The notebook will print "Using device: cuda" or "Using device: cpu" at startup to confirm which device is active
