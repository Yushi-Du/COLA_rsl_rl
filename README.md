# COLA RSL-RL fork

This pip package contains only the RSL-RL implementations used by the COLA
three-stage pipeline:

| stage | policy | algorithm | runner |
| --- | --- | --- | --- |
| locomotion | `ActorCriticWbcEnd2endQuat` | `PPO_WbcEnd2endQuat` | `OnPolicyRunnerEnd2end` |
| collaboration teacher | `ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel29` | `PPO_WbcEnd2endWholePipeResiVel` | `OnPolicyRunnerWholePipeResi` |
| distilled student | `StudentTeacherDistill` | `DistillationDistill` | `OnPolicyRunnerWholePipeResi` |

Install it from the release checkout so imports resolve under the package name
`rsl_rl`:

```bash
python -m pip uninstall -y rsl-rl-lib rsl_rl
python -m pip install -e ./rsl_rl
```

COLA-specific modifications are released under the MIT License. Portions
derived from RSL-RL remain subject to the upstream BSD-3-Clause license in
`THIRD_PARTY_LICENSES/rsl_rl-BSD-3-Clause.txt`.
