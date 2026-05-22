# Learning What to Observe: Reinforcement Learning with Endogenous Sensing

This repository accompanies a submission to the *Finding the Frame Workshop (RLC 2026)*.

## Overview

Reinforcement learning (RL) typically assumes that observations are fixed and freely available. Recent work relaxes this assumption by allowing agents to decide **when** to observe, often through observation costs. However, these approaches still treat observations as atomic and externally defined.

This work explores a different perspective:

> **Observation should be treated as a decision variable.**

Instead of deciding only *when* to observe, agents decide **what information to acquire** from a structured set of sensing options. This reframes observation as part of the policy, rather than a fixed interface to the environment.

---

## Key Idea

We extend action-contingent observation models by allowing agents to select subsets of observation channels at each step. Formally, the agent selects a composite action: (a, S)

where:
- `a` is a control action
- `S` is a set of sensing decisions

This introduces **structured sensing**, enabling:
- selective information acquisition
- action-conditioned sensing decisions
- trade-offs between task performance and sensing cost

---

## Environment

We evaluate this formulation in a simple **Dual-Sensor GridWorld** environment:

- Grid navigation with a hazardous region
- Two sensing modalities:
  - **Local sensing** (low cost, partial/noisy)
  - **Global sensing** (high cost, precise)
- Agents maintain memory of previous observations

The environment is designed to isolate the effect of **observation representation** rather than serve as a benchmark.

---

## Methods

We compare three classes of agents:

- **Baseline**: Always senses (full observability, high cost)
- **AMLR**: Joint control + sensing with a single value function
- **CASCADE-Q**: Decoupled control and sensing, with sensing conditioned on actions

Each method is evaluated with:
- **Coarse sensing** (binary)
- **Fine-grained sensing** (local + global)

---

## Main Findings

Our experiments show that treating observation as endogenous leads to qualitatively different behavior:

- **Selective sensing**: agents acquire information only when needed  
- **Action-conditioned sensing**: sensing depends on intended actions  
- **Temporal structure**: sensing is concentrated in uncertain regions and reduced over time  

Interestingly:
- Coarse sensing performs comparably to fine-grained sensing when conditioned on action
- Fine-grained sensing is used strategically rather than continuously

---

## Takeaway

The key result is not an algorithm, but a **change in representation**:

> How observation is modeled fundamentally shapes learned policies.

Moving from atomic to structured observations changes not only efficiency, but the **structure of behavior itself**.

---

## Running the Code

(Instructions here should be filled in depending on your implementation)

Example:

```bash
python train.py --agent cascade_q --mode fine
