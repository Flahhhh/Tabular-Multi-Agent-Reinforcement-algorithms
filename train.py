from env import get_env
from agents.ma_q_learning import MAQ, MAQCfg
from agents.PHC import PHC, PHCCfg
from agents.WoLF_PHC import WoLFPHC, WoLFPHCCfg
from agents.nash_q_learning import NashQ, NashQCfg

agent = PHC(PHCCfg())
env = get_env(agent.env_name)

agent.train(env)