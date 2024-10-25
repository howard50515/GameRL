from .networks import DQNNetwork

class DQNAgent:
    def __init__(self) -> None:
        self.eval_net = DQNNetwork()
        self.target_net = DQNNetwork()
        pass

    def sample(self, state, epsilon) -> int:
        return 0

    def store_transition(self, state, action, reward, next_state) -> None:
        pass

    def learn() -> None:
        pass

    def load() -> None:
        pass
    
    def save() -> None:
        pass