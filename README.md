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
python -m pip install -e ./my_rsl_rl
```

The parent repository's release validator enforces the expected file set.
