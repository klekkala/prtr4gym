import gym
import numpy as np
from ray.rllib.models import ModelCatalog
import envs
from envs import SingleAtariEnv
from ray.rllib.models.torch.visionnet import VisionNetwork
from atari_vae import Encoder, TEncoder
from ray.rllib.policy.policy import Policy
from models.atarimodels import SingleAtariModel
from IPython import embed
import random
from arguments import get_args

args = get_args()



#backbone from the 
def eval_adapter(tmodel_path, model_path, env_name, backbone):

    param_dict = {}
    for name, param in backbone.named_parameters():
        param_dict[name] = param

    ModelCatalog.register_custom_model("model", SingleAtariModel)
    teacherwts = Policy.from_checkpoint(model_path).get_weights()

    encodernet = Policy.from_checkpoint(model_path)
    encoderwts = encodernet.get_weights()

    chng_wts = {}
    for params in encoderwts.keys():
        if 'logits' not in params and 'value' not in params:
            #load from the backbone
            print(params)
            #embed()
            chng_wts[params] = param_dict[params.replace("_convs.", "")]
        else:
            #load from tmodel
            #embed()
            chng_wts[params] = teacherwts[params]

    encodernet.set_weights(chng_wts)

    res=[]
    env = SingleAtariEnv({'env': env_name, 'full_action_space': False, 'framestack': True})
    
    obs = env.reset()
    random_generated_int = random.randint(0, 2**31-1)
    env.seed(random_generated_int)
    
    total = 0.0
    for _ in range(5000):
        action = encodernet.compute_single_action(obs)[0]
            
        obs, reward, done, _ = env.step(action)
            
        total += reward
        if done:
            break
  
    
    f = open(args.texpname + '_' + args.expname + '.txt', 'a')
    f.write(str(total) + ', ')
    f.close()
    

    #with open('Name.txt','w') as f:
    #    f.write(str(res))
    return total
