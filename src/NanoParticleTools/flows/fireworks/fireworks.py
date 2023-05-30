from fireworks import Firework
from NanoParticleTools.flows.fireworks.firetasks import NPMCFiretask


class NPMCFW(Firework):
    def __init__(self, **kwargs):
        tasks = []
        tasks.append(NPMCFiretask(**kwargs))
        super(NPMCFW, self).__init__(tasks)
