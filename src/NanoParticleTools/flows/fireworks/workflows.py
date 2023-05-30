from NanoParticleTools.flows.fireworks.fireworks import NPMCFW
from fireworks import LaunchPad, Workflow, Firework


def get_npmc_workflow(**kwargs):
    return Workflow([NPMCFW(**kwargs)])
