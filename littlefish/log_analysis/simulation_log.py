import h5py


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


if __name__ == "__main__":
    import os

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(
        os.path.dirname(curr_folder), "tests", "simulation_log.hdf5"
    )
    log = SimulationLog(log_path=log_path)
    print(log.name)
