from dataclasses import dataclass, fields


@dataclass
class SimulatorConfiguration:
    freq: float
    box_size: float
    num_scan_steps: int
    neighbor_radius: float
    to_jit: bool

    @classmethod
    def from_simulator(cls, simulator):
        field_names = [field.name for field in fields(cls)]
        return cls(**{
            field_name: getattr(simulator, field_name)
            for field_name in field_names})