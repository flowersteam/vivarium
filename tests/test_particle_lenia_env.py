from vivarium.environments.particle_lenia import init_state, ParticleLeniaEnv

NUM_STEPS = 10


def test_particle_lenia_env_running():

    state = init_state()
    env = ParticleLeniaEnv(state=state)

    for _ in range(NUM_STEPS):
        state = env.step(state=state, num_scan_steps=3)

    assert env
    assert state
