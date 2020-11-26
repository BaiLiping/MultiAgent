# MultiAgent

This Repository is for the paper: Accelerate Reinforcement Learning with MultiAgents Roll Out

The theoretical background for this paper can be found at https://www.mit.edu/~dimitrib/home.html

To install the environments related to this paper, please visit https://github.com/BaiLiping/BLPgym

```
git clone https://github.com/BaiLiping/BLPgym
cd BLPgym
pip3 install -e .
```

The Agents are implemented using Tensorforce. For detailed notes, please visit: https://github.com/tensorforce/tensorforce

```
pip3 install tensorforce
```

There are the following environments:
```
register(
    id='HalfCheetahBLP-v0',
    entry_point='gym.envs.mujoco.half_cheetah_blp:HalfCheetahBLPEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HopperBLP-v2',
    entry_point='gym.envs.mujoco.hopper_blp_v2:HopperBLPEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Walker2dBLP-v0',
    max_episode_steps=1000,
    entry_point='gym.envs.mujoco.walker2d_blp:Walker2dBLPEnv',
)
```
The Plots are generated by comparing MultiAgent Roll Out methods with the normal agents.

```
HopperBLP-v2 Vs Hopper-V3
HalfCheetahBLP-v0 Vs HalfCheetah-v3
Walker2dBLP-v0 Vs Walker2d-v3
```

