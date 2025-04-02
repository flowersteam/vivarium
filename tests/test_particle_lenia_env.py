from vivarium.environments.particle_lenia import ParticleLeniaEnv


NUM_STEPS = 10

def test_particle_lenia_env_running():
    env = ParticleLeniaEnv.from_scene('particle_lenia')
    state = env.state
    
    for _ in range(NUM_STEPS):
        state = env.step(state)

    assert env
    assert state
