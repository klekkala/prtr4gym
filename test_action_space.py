import gym

envs = ["ALE/AirRaid-v5","ALE/Assault-v5","ALE/BeamRider-v5", "ALE/Carnival-v5","ALE/DemonAttack-v5","ALE/NameThisGame-v5","ALE/Phoenix-v5","ALE/Riverraid-v5","ALE/SpaceInvaders-v5"]
envs = ["AirRaidNoFrameskip-v4","AssaultNoFrameskip-v4","BeamRiderNoFrameskip-v4", "CarnivalNoFrameskip-v4","DemonAttackNoFrameskip-v4","NameThisGameNoFrameskip-v4","PhoenixNoFrameskip-v4","RiverraidNoFrameskip-v4","SpaceInvadersNoFrameskip-v4"]
for i in envs: 
    env = gym.make(i)
    print(i)
    print(env.action_space)
    print(env.observation_space)



from gym.envs.atari.atari_env import AtariEnv
env = AtariEnv(game=game, obs_type='image', frameskip=1, full_action_space=True)
