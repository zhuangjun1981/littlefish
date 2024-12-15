import h5py
import littlefish.core.utilities as util


class SimulationLog:
    """
    read only api to read simulation logs
    """

    def __init__(self, log: h5py.Group = None, log_path: str = None):
        """
        if log is specified, log should be the root group of one simulation log, with name starts with "simulation_"
        if log_path is specified, look for the simulation log group in the file root, only one simulation log group should exist in the file.
        """
        if log is None and log_path is not None:
            ff = h5py.File(log_path, "r")
            log_names = [k for k in ff.keys() if k.startswith("simulation_")]
            if len(log_names) == 0:
                raise ValueError(f"Cannot find log group in path: {log_path}.")
            elif len(log_names) > 1:
                raise ValueError(f"More than one log groups found in path: {log_path}.")
            else:
                self.log = ff[log_names[0]]
        elif log is not None and log_path is None:
            if not log.name.split("/")[-1].startswith("simulation_"):
                raise ValueError(
                    "input 'log' does not look like a simulation log: a hdf5.Group object with name starts with 'simulation_'."
                )
            self.log = log
        elif log is None and log_path is None:
            raise ValueError("Input 'log' and 'log_paht' cannot both be None.")
        else:
            raise ValueError("One of the input 'log' and 'log_path' should be None")

    @property
    def name(self):
        return self.log.name.split("/")[-1]

    @property
    def terrain_shape(self):
        return self.log["terrain_map"].shape

    @property
    def food_num(self):
        return self.log["food_num"][()]

    @property
    def terrain_map(self):
        return self.log["terrain_map"][()]

    @property
    def max_simulation_length(self):
        return self.log["max_simulation_length"][()]

    @property
    def last_time_point(self):
        return self.log["simulation_cache/last_time_point"][()]

    @property
    def ending_time(self):
        return util.decode(self.log["simulation_cache/ending_time"][()])

    @property
    def fish_names(self):
        return [
            k[5:]
            for k in self.log.keys()
            if isinstance(self.log[k], h5py.Group) and k.startswith("fish_")
        ]

    @property
    def random_seed(self):
        if "random_seed" in self.log["simulation_cache"]:
            return self.log["simulation_cache/random_seed"][()]
        else:
            return None

    @property
    def numpy_random_seed(self):
        if "numpy_random_seed" in self.log["simulation_cache"]:
            return self.log["simulation_cache/numpy_random_seed"][()]
        else:
            return None

    def get_food_position_history(self):
        return self.log["simulation_cache/food_pos_history"][()]

    def get_fish_total_moves(self, fish_name: str):
        return self.log[f"fish_{fish_name}/total_moves"][()]

    def get_fish_firing_stats(self, fish_name: str):
        grp_action_history = self.log[
            f"fish_{fish_name}/brain_simulation_cache/action_histories"
        ]
        action_num = 0
        for k, v in grp_action_history.items():
            action_num += len(v)
        return action_num, action_num / (
            len(grp_action_history.keys()) * self.last_time_point
        )

    def get_fish_health_history(self, fish_name: str):
        return self.log[f"fish_{fish_name}/health_history"][()]

    def get_fish_position_history(self, fish_name: str):
        return self.log[f"fish_{fish_name}/position_history"][()]


def get_simulation_logs(h5_grp: h5py.Group) -> list[SimulationLog]:
    """
    get all simulation logs from a h5py group
    """
    return [
        SimulationLog(log=v) for k, v in h5_grp.items() if k.startswith("simulation_")
    ]


if __name__ == "__main__":
    import os

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(
        os.path.dirname(curr_folder), "tests", "simulation_log.hdf5"
    )
    log = SimulationLog(log_path=log_path)
    print(log.name)
