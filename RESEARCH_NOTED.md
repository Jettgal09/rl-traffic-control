# Research Paper Notes
## Project: RL-Based Adaptive Traffic Signal Control

---

## Core Claim
An RL-based adaptive traffic signal controller reduces average vehicle 
waiting time compared to a fixed-timer baseline controller.

---

## Abstract Notes
- Problem: Fixed timer signals waste green time regardless of demand
- Solution: RL agent observes queue lengths and adapts signal timing
- Method: Custom Python simulation + Gym interface + DQN/PPO/A2C
- Metric: Average waiting time, queue length, vehicle throughput
- Result: (fill in after 500k training completes)

## Introduction Notes

Key statistics to find on Google Scholar and cite:
- "Traffic congestion costs X billion dollars annually" 
- "Average commuter wastes X hours per year in traffic"
- "Traffic signals control X% of urban intersections"

Problem statement:
- Fixed timers designed for average traffic, not real-time conditions
- During off-peak hours, one direction gets full green even with 0 cars
- During rush hour, the other direction queues indefinitely

Our contribution:
- Built complete custom simulation from scratch (not SUMO/VISSIM)
- Compared 3 RL algorithms under identical conditions
- Showed training progression matters (120k vs 500k steps)
- Open source, reproducible, no hardware required

## Related Work Notes — Papers to Find and Read

1. Wiering, M. (2000) - "Multi-agent reinforcement learning for 
   traffic light control"
   - One of the earliest RL traffic papers
   - Used tabular Q-learning (we use deep neural network)

2. Wei et al. (2018) - "IntelliLight: A Reinforcement Learning 
   Approach for Intelligent Traffic Light Control"
   - Used real traffic data from Hangzhou, China
   - Key difference from us: they used real data, we use simulation

3. Zheng et al. (2019) - "Learning Phase Competition for 
   Traffic Signal Control"
   - Multi-intersection coordination
   - Relevant to our Phase 2 (3x3 grid)

4. Liang et al. (2019) - "A Deep Reinforcement Learning Network 
   for Traffic Light Cycle Control"
   - Used DQN specifically (same as our primary algorithm)

Key things to note when reading each paper:
- What state space did they use?
- What reward function?
- What simulation environment?
- What baseline did they compare against?
- What improvement % did they achieve?

## Methodology Notes — Already Built

4.1 Simulation Design
- Built from scratch in Python (not SUMO/VISSIM/CityFlow)
- Why custom? Full control over state/reward/action design
- Single intersection Phase 1, extensible to 3x3 grid Phase 2
- Discrete time steps, each step = one simulation tick

4.2 State Space Design Decisions
- Why queue length AND total count? 
  Queue = stopped cars (immediate congestion)
  Count = approaching cars (future congestion)
  Together they give agent both present and near-future information
- Why include phase_id in observation?
  Agent needs to know current phase to make sensible switch decisions
  Without it, agent might request switches during yellow/all-red
- Why normalize to [0,1]?
  Neural networks converge faster with normalized inputs
  Prevents large values dominating gradient updates

4.3 Reward Function Design Decisions
- Why negative reward?
  RL maximizes reward, so penalty = minimization problem
- Why waiting_time weighted more than queue_length?
  Waiting time is cumulative — grows every step
  Queue length can drop instantly when light turns green
  Waiting time better represents total driver frustration
- Why normalize by MAX_VEHICLES?
  Keeps reward in consistent range regardless of traffic density
  Allows fair comparison across different spawn rates

4.4 Action Space Design Decisions  
- Why binary action (switch/keep) not multi-phase?
  Simpler action space = faster learning
  Agent learns WHEN to switch, light handles HOW (yellow/all-red)
- Why MIN_GREEN_DURATION = 10?
  Prevents pathological policy of flickering lights every step
  Models real-world minimum green time requirements

4.5 Algorithm Selection Rationale
- DQN: Good baseline for discrete action spaces
        Off-policy = sample efficient (learns from old experience)
- PPO: State of the art policy gradient
        More stable than vanilla PG due to clipping
- A2C: Faster updates than PPO (every 5 steps vs 2048)
        Good for comparison against PPO

## Experiments Notes — Fill in after training

5.1 Training Setup
- Hardware: CPU only (note your CPU model)
- Training speed: ~800 it/s
- DQN 500k training time: (fill in)
- Framework: Stable-Baselines3 v2.7.1
- Python 3.12, PyTorch (fill in version)

5.2 Baseline Description
- Fixed timer: switches every 30 steps regardless of traffic
- Represents current real-world standard
- No observation of traffic state

5.3 Metrics Tracked
1. Total waiting time per episode
   (sum of all vehicle waiting steps across episode)
2. Average queue length per step
   (average stopped vehicles at any given moment)
3. Vehicles spawned per episode
   (proxy for throughput — more = less congestion)
4. Learning curve
   (mean_reward vs training steps from TensorBoard)

5.4 Results Table (fill in)
| Algorithm    | Avg Waiting | Avg Queue | Spawned | vs Baseline |
|-------------|-------------|-----------|---------|-------------|
| Fixed Timer  |             |           |         | baseline    |
| DQN (120k)  |             |           |         | worse       |
| DQN (500k)  |             |           |         | (fill in)   |
| PPO (500k)  |             |           |         | (fill in)   |
| A2C (500k)  |             |           |         | (fill in)   |

5.5 Key Observations to Discuss
- At what training step did DQN first outperform baseline?
- Which algorithm converged fastest?
- Did heavy traffic (spawn=0.10) affect relative performance?
- What does the learning curve shape look like? (smooth or sudden jumps?)

## Discussion Notes

Points to address:
1. Why did DQN need 500k steps to beat baseline?
   - State space is small (9 values) but reward is very delayed
   - Agent must learn that switching NOW prevents congestion LATER
   - This is the credit assignment problem in RL

2. Limitations of our simulation
   - No turning vehicles (Phase 3)
   - Single intersection (Phase 2 addresses this)  
   - No pedestrian signals
   - No emergency vehicle priority
   - Vehicles don't have destinations

3. Real world applicability
   - State space maps to real loop detector sensors
   - Action space maps to real signal controller API
   - Would need calibration with real traffic data

4. What would improve results further?
   - Larger state space (vehicle speeds, density maps)
   - Curriculum learning (start easy, increase difficulty)
   - Multi-agent for grid coordination

## Conclusion Notes
- We demonstrated RL can learn adaptive signal control
- DQN/PPO/A2C all eventually outperform fixed timer (fill in after results)
- Training time matters — 120k insufficient, 500k sufficient
- Custom simulation allows full experimental control
- Future work: Phase 2 (grid), Phase 3 (turning), real data calibration

## Figure Descriptions (draw these in paper)

Figure 1 — System Architecture Diagram
  [RL Agent] → action → [Gym Environment] → [Traffic Simulation]
  [RL Agent] ← reward, observation ← [Gym Environment]

Figure 2 — Intersection Layout
  Show 4-way intersection with:
  - 8 lanes (2 per direction)
  - Stop lines
  - Traffic light positions
  - Vehicle spawn points at map edges

Figure 3 — Traffic Light State Machine
  Show the 6-phase cycle as a circular diagram:
  GREEN_NS → YELLOW_NS → ALL_RED_1 → GREEN_EW → YELLOW_EW → ALL_RED_2 → (repeat)
  Label which transitions are agent-controlled vs automatic

Figure 4 — Learning Curve (from TensorBoard)
  X axis: training steps
  Y axis: mean episode reward
  Show DQN learning curve with breakthrough moment marked

Figure 5 — Algorithm Comparison Bar Chart
  3 bars per metric: Fixed Timer, DQN, PPO, A2C
  Metrics: waiting time, queue length, vehicles spawned

## Software Versions (for reproducibility section)
- Stable-Baselines3: 2.7.1
- PyTorch: 2.10.0 (CPU only)
- Gymnasium: 1.2.3
- Python: 3.12
- OS: Windows
Note: PyTorch 2.10.0+cpu means CPU-only build.
GPU version would significantly speed up training for larger networks.
For this project CPU is sufficient — our network is small (9 inputs → 2 outputs).
## Hardware Specifications
- CPU: Intel Core i7-13700HX (13th Gen, 24 CPUs, ~2.1GHz)
- RAM: 16GB
- OS: Windows 11 Home 64-bit
- GPU: Not used for training (CPU-only PyTorch build)
- Machine: Acer Predator PHN16-71

---

## Baseline Result (Random Actions, 200 steps)
- Total waiting time: 54,381
- Average wait per step: 271.9
- Total vehicles spawned: 84
- Average queue length: ~3 vehicles

This is what we need to beat with trained RL agents.

---

## Hyperparameters Used
- GRID_SIZE = 1
- VEHICLE_SPEED = 2.0
- SPAWN_RATE = 0.05
- MIN_GREEN_DURATION = 10
- YELLOW_DURATION = 5
- ALL_RED_DURATION = 2
- MAX_STEPS_PER_EPISODE = 3000
- TOTAL_TIMESTEPS = 500,000

---

## Results (fill in after training)
| Algorithm    | Avg Wait/Step | Avg Queue | Vehicles Passed |
|-------------|--------------|-----------|-----------------|
| Fixed Timer  |              |           |                 |
| DQN          |              |           |                 |
| PPO          |              |           |                 |
| A2C          |              |           |                 |

---

## To Look Up (Related Work)
- Wiering 2000 — first RL traffic paper
- Wei et al. 2018 — IntelliLight
- Zheng et al. 2019 — CoLight
- Search: "deep reinforcement learning traffic signal control survey"

---

## Notes & Observations
- Random agent produces total_waiting_time ~54k over 200 steps
- Reward starts at 0 and becomes increasingly negative as congestion builds
- Fixed timer baseline will be: switch every 30 steps regardless of traffic

## DQN 100k training:
- Reward at step 10k:  -2,616,529
- Reward at step 70k:  -1,855,362  (breakthrough)
- Reward at step 100k: -1,979,302
- Training time: ~1m 53s
- Speed: ~783 it/s

## Evaluation Results (DQN at 120k steps — undertrained)

| Algorithm   | Avg Waiting  | Avg Queue | Spawned |
|-------------|-------------|-----------|---------|
| Fixed Timer | 22,391,196  | 39.87     | 811     |
| DQN (120k)  | 37,450,880  | 50.13     | 668     |

Note: DQN undertrained (120k steps, interrupted).
Expected to improve significantly at 500k steps.
This shows RL needs sufficient training to outperform baseline.

## Key Insight — Spawned Vehicles as Congestion Indicator
Fewer vehicles spawned = more congestion = map hit MAX_VEHICLES cap.
Fixed timer allowed more vehicles through than undertrained DQN.
This metric indirectly measures throughput efficiency.

## DQN 500k Training Results
- Final mean reward: -451,653
- vs 120k reward: -2,600,000
- Improvement: ~83% better reward with full training
- Training time: 11 minutes 26 seconds
- Speed: 746 it/s

## Key Finding — DQN 500k Results
DQN at 500k steps matches fixed timer performance:
  - Waiting time: DQN 4.5% worse (within noise margin)
  - Queue length: DQN 1.9% better
  - Throughput: DQN 0.9% better (824 vs 817 spawned)

Interpretation: DQN learned a policy equivalent to fixed timer.
Not a clear improvement — possible reasons:
  1. May need more training steps (try 1M)
  2. Reward signal too delayed/noisy for agent to learn clear advantage
  3. Fixed timer at 30-step interval may already be near-optimal
     for our specific spawn_rate=0.05 setting

Research question this raises:
  Does fixed timer perform differently at spawn_rate=0.10 (heavy traffic)?
  RL should have bigger advantage when traffic is more variable.

## PPO 500k Training Results
- Final eval reward: -3,397,163
- Training time: 13 minutes 29 seconds  
- Speed: 674 it/s
- vs DQN 500k: DQN performed ~7.5x better in eval reward

Possible reasons PPO underperformed:
- PPO is on-policy — it discards old experience after each update
- Our environment has very delayed rewards (congestion builds slowly)
- PPO may need more careful hyperparameter tuning for this task
- DQN's replay buffer helps it learn from rare good experiences

## Learning Curve Analysis

DQN curve: fragmented due to multiple interrupted training runs
PPO curve shows catastrophic forgetting:
  - Steps 0-300k: stable policy around reward -300,000
  - Step ~330k: catastrophic policy collapse to -3,500,000
  - Steps 330k-500k: never recovers from collapse
  
This demonstrates PPO's sensitivity to hyperparameters.
The clip_range=0.2 was insufficient to prevent the bad update.
This is a known PPO failure mode documented in literature.

For paper: This is Figure 4 — Learning Curves

## Figure 4 — Learning Curve Analysis

DQN characteristics:
  - Slow steady improvement over 500k steps
  - 83% reward improvement from start to finish
  - No catastrophic failures
  - Final reward: -451,653

PPO characteristics:
  - Fast initial learning (reached -300k by step 10k)
  - Stable for 300k steps
  - Catastrophic forgetting at step ~330k
  - Never recovered — ended at -3,500,000
  - PPO was better than DQN for steps 0-300k
  - Then became 7.5x worse after collapse

Key insight for paper:
  Algorithm stability matters as much as peak performance.
  DQN's off-policy replay buffer provides more stable learning.
  PPO's on-policy updates are vulnerable to bad policy updates.

## A2C 500k Training Results
- Final eval reward: -3,335,226
- Rolling training reward: -473,000 (much better than eval)
- Training time: 15 minutes 8 seconds
- Speed: 500 it/s (slowest of the three — updates every 5 steps)

Interesting: A2C's training reward (-473k) was close to DQN's eval reward (-451k)
but eval reward was much worse (-3.3M). This suggests A2C is inconsistent —
performs well sometimes but not reliably. High variance policy.

## Critical Finding — Degenerate Policy Discovery

DQN learned action 0 (keep phase) 100% of the time.
Switch rate = 0% across all evaluation episodes.

ROOT CAUSE — Reward Shaping Problem:
  When agent switches phase:
    → yellow phase starts (5 steps where nobody moves)
    → all-red phase (2 steps)
    → immediate spike in waiting_time
    → agent receives worse reward immediately
  
  Agent learned: "switching always makes reward worse short-term"
  Agent conclusion: "never switch"
  
  This is correct locally but catastrophically wrong globally.
  Without switching, EW traffic never gets green → infinite queue.

This explains the train/eval gap:
  During training: agent gets reward -451k because it never switches
    and NS traffic flows freely (NS starts green and stays green forever)
    but EW traffic builds up — eventually hitting MAX_VEHICLES cap
  During evaluation over full 3000 steps: EW congestion saturates
    the entire simulation → catastrophic performance

WHY PPO/A2C PERFORMED DIFFERENTLY:
  PPO and A2C switch more randomly due to entropy bonus (ent_coef=0.01)
  This forces some exploration — occasional switches happen
  Result: worse than never-switching short term, but avoids total EW lockout

## Reward Function Iteration

Version 1 (original):
  reward = -(waiting + 0.5 * queue)
  Problem: agent learned degenerate "never switch" policy
  
Version 2 (improved):
  reward = -(waiting + 0.5 * queue + 2.0 * imbalance)
  imbalance = |NS_queue - EW_queue| / MAX_VEHICLES
  Purpose: penalize ignoring one direction
  
This is a key contribution — reward shaping for traffic RL.
The imbalance term forces the agent to balance both directions.
This is mentioned in related work but our implementation is novel.

## Reward Function Evolution

### V1 — Cumulative Waiting Time (Original)
```
reward = -(total_waiting_time/MAX_V + 0.5 * total_queue/MAX_V)
```
Problem: Agent learned degenerate "never switch" policy (0% switch rate)
Root cause: Switching causes yellow/all-red phases → immediate waiting spike
→ Agent learned switching = punishment, never switching = safe
Result: DQN eval reward -451k but switch rate 0% → catastrophic in long episodes

### V2 — Adding Imbalance Penalty
```
reward = -(waiting/MAX_V + 0.5 * queue/MAX_V + 2.0 * |NS_queue - EW_queue|/MAX_V)
```
Problem: Imbalance weight (2.0) still insufficient to overcome switching pain
Result: Still 0% switch rate — degenerate policy persisted

### V3 — Queue-Based + Throughput Bonus (Current)
```
reward = -(queue/MAX_V + 3.0 * |NS_queue - EW_queue|/MAX_V) 
         + throughput_bonus
```
Key changes:
  - Removed cumulative waiting_time (was growing forever, masking signal)
  - Increased imbalance weight from 2.0 to 3.0
  - Added throughput bonus for vehicles passing through each step
  - Reward now in range -1 to +1 (vs millions before)
  
Early results: reward -2,732 at step 100k (vs -2,600,000 before)
Loss dropped from 159 to 0.93 — much more stable learning

### Key Lesson for Paper
Reward shaping is critical in RL — a poorly designed reward
produces degenerate policies even with correct algorithm implementation.
The credit assignment problem: agent must connect switching NOW
to reduced congestion LATER across yellow/all-red transition steps.

## Diagnostic Finding — Observation Verification
At step 300 with always-keep-NS-green policy:
  E_queue = 0.38 (38 cars stopped)
  W_queue = 0.12 (12 cars stopped)  
  N_queue = 0.00 (0 cars stopped)
  S_queue = 0.00 (0 cars stopped)
  phase = 0.00 (GREEN_NS entire time)
  reward = -2.00

Confirmed: simulation correctly models EW congestion buildup.
Observation vector accurately reflects real queue state.
Agent has sufficient information to learn switching policy.
Issue is purely training convergence speed.

## Research Finding — Training Challenges

1. Reward shaping is harder than expected for traffic control
2. Credit assignment across yellow/all-red phases creates 
   a local optimum trap (never switch = avoid immediate pain)
3. DQN shows unstable convergence — oscillates rather than 
   monotonically improving
4. The degenerate "never switch" policy is a known failure 
   mode in traffic RL literature

These findings motivate future work:
- Curriculum learning
- Reward shaping with potential-based functions
- Hierarchical RL (separate switch/timing decisions)

## Training Convergence Analysis

All DQN training runs converged to suboptimal local optima:
- "Never switch" policy (0% switch rate) — reward V1, V2
- "Always switch" policy (98% switch rate) — reward V2 with low MIN_GREEN
- Oscillating policy (~-3,500 reward) — reward V4 with obs size 10

Common pattern: agent finds local optimum early, stops improving.
Reward range -3,298 to -3,800 consistently after step 220k.

Root causes identified:
1. Sparse reward signal — switching benefit appears 15+ steps later
2. Episode length (3000 steps) creates very long-horizon credit assignment
3. Binary action space limits policy expressiveness
4. DQN's discrete Q-values struggle with the continuous tradeoff between
   direction priorities

Literature comparison:
Most successful traffic RL papers use:
- SUMO simulator with more detailed state (vehicle speeds, positions)
- Reward shaping with potential-based functions
- Longer training (2-5M steps)
- Larger neural networks (256-512 hidden units vs SB3 default 64)

## Final Algorithm Comparison — Switch Rate Analysis

| Algorithm | Switch Rate | Policy Type        | Result              |
|-----------|-------------|-------------------|---------------------|
| DQN       | 2.2%        | Nearly never switch| EW congestion builds|
| PPO       | 100%        | Always switch      | Lights flicker, no flow|
| Fixed Timer| ~3.3%      | Every 30 steps     | Balanced baseline   |
| Optimal   | ~15-30%     | Context-dependent  | Target behavior     |

Both RL algorithms converged to opposite degenerate policies.
This is a known failure mode when reward shaping is misaligned
with the desired behavior.

## Honest Research Conclusion

Single-intersection binary-action RL traffic control with custom 
simulation produces degenerate policies due to:

1. Yellow/all-red transition penalty creates local optima
2. Binary action space (switch/keep) insufficient expressiveness  
3. 3000-step episodes create difficult credit assignment
4. Both DQN and PPO converge to opposite degenerate solutions

This motivates:
- Phase-duration action space (choose how long to stay green)
- Shorter episodes with curriculum learning
- Potential-based reward shaping
- Multi-intersection coordination (Phase 2)

## First Working Results — Duration-Based Action Space

| Algorithm   | Avg Waiting  | Avg Queue | Spawned |
|-------------|-------------|-----------|---------|
| Fixed Timer | 15,578,184  | 33.97     | 900     |
| DQN (500k)  | 15,102,202  | 31.49     | 920     |

DQN improvement over Fixed Timer:
  Waiting time: 3.1% reduction
  Queue length: 7.3% reduction  
  Throughput:   2.2% more vehicles processed

DQN learned a 60-step fixed policy — better than 30-step baseline.
Not fully adaptive yet but genuinely outperforms baseline.

## Progress Across System Iterations

### Old System (binary action space, reward V1)
| Algorithm   | Avg Waiting  | Avg Queue | Spawned |
|-------------|-------------|-----------|---------|
| Fixed Timer | 22,391,196  | 39.87     | 811     |
| DQN (120k)  | 37,450,880  | 50.13     | 668     |

### New System (duration action space, reward V5)
| Algorithm   | Avg Waiting  | Avg Queue | Spawned |
|-------------|-------------|-----------|---------|
| Fixed Timer | 15,578,184  | 33.97     | 900     |
| DQN (500k)  | 15,102,202  | 31.49     | 920     |

Key improvements from redesign:
1. Fixed timer improved too — because MAX_RED_DURATION=100 forces
   balanced switching, preventing permanent EW starvation
2. DQN now beats fixed timer — first time in entire project
3. Overall waiting time dropped ~30% just from better simulation design
4. Spawned vehicles increased from 811 to 900+ — simulation flows better

This shows that simulation design matters as much as algorithm choice.
The duration-based action space fundamentally changed what the agent
could learn.

## Current Algorithm Rankings (500k steps, duration action space)

| Algorithm   | Best Eval Reward | Status        |
|-------------|-----------------|---------------|
| PPO         | -2,448          | ✅ Complete   |
| DQN         | -2,499          | ✅ Complete   |
| Fixed Timer | baseline        | ✅ Reference  |
| A2C         | TBD             | 🔜 Training   |

PPO is currently winning — better than DQN by ~2%.
Both beat fixed timer baseline.

## FINAL RESULTS — Duration-Based Action Space

| Algorithm   | Avg Waiting  | Avg Queue | Spawned | vs Baseline |
|-------------|-------------|-----------|---------|-------------|
| Fixed Timer | 15,248,102  | 33.02     | 898     | baseline    |
| DQN         | 14,048,423  | 31.30     | 924     | -7.9%       |
| PPO         | 13,808,619  | 31.09     | 924     | -9.4%       |
| A2C         | 13,663,147  | 30.76     | 923     | -10.4% ✅   |

ALL THREE RL algorithms outperform the fixed timer baseline.
A2C achieved the best overall performance.

Key metrics improvement (best RL vs baseline):
  Waiting time reduced: 10.4%
  Queue length reduced: 6.8%
  Throughput increased: 2.8% (923 vs 898 vehicles)

Winner: A2C — fastest updates (every 5 steps) proved most effective
        for learning phase duration timing.

