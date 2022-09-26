import numpy as np

rng = np.random.default_rng()


class CAModel:
    def __init__(self, initial_state, sigma):
        self.state = initial_state
        self.sigma = sigma

    @property
    def state(self):
        return np.array((self.x, self.v, self.a))

    @state.setter
    def state(self, value):
        self.x, self.v, self.a = value

    def update(self, time_step=1.0):
        eta = rng.normal(0, self.sigma)
        self.x = self.x + self.v * time_step + (self.a + eta) * time_step ** 2 / 2
        self.v = self.v + (self.a + eta) * time_step
        self.a = self.a + eta


class CVModel:
    def __init__(self, initial_state, sigma):
        self.state = initial_state
        self.sigma = sigma

    @property
    def state(self):
        return np.array((self.x, self.v))

    @state.setter
    def state(self, value):
        self.x, self.v = value

    def update(self, time_step=1.0):
        eta = rng.normal(0, self.sigma)
        self.x = self.x + self.v * time_step + eta * time_step ** 2 / 2
        self.v = self.v + eta * time_step


class Target:
    def __init__(self, initial_state, sigma=0.5, model="CA"):
        if model == "CA":
            self.x = CAModel(initial_state[0], sigma)
            self.y = CAModel(initial_state[1], sigma)
        elif model == "CV":
            self.x = CVModel(initial_state[0], sigma)
            self.y = CVModel(initial_state[1], sigma)
        else:
            raise ValueError(f"Model must be CA or CV, got {model}")

    @property
    def state(self):
        return np.array((self.x.state, self.y.state))

    @state.setter
    def state(self, value):
        print(value.shape)
        self.x.state, self.y.state = value

    @property
    def position(self):
        return self.state[:, 0]

    def update(self, time_step=1.0):
        self.x.update(time_step)
        self.y.update(time_step)
