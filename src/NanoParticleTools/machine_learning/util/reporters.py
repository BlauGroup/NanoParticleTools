from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.trial import Trial

class ExperimentTerminationReporter(CLIReporter):
    """
    This is adapted from the ray documentation:

    https://docs.ray.io/en/latest/tune/api_docs/reporters.html
    """
    def __init__(self, **kwargs):
        super().__init__(max_report_frequency = 30, **kwargs)

    def should_report(self, trials, done=False):
        """Reports only on experiment termination."""
        return done

class TrialTerminationReporter(CLIReporter):
    """
    This is adapted from the ray documentation:
    
    https://docs.ray.io/en/latest/tune/api_docs/reporters.html
    """
    def __init__(self, **kwargs):
        super().__init__(max_report_frequency = 30, **kwargs)
        self.num_terminated = 0
        

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated