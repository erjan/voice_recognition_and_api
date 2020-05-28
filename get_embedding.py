import sys, os
import torch
from hparam import hparam as hp
import numpy as np
from speech_embedder2 import SpeakerRecognition, SILoss
from utils import extract_all_feat
import random
torch.manual_seed(hp.seed)
np.random.seed(hp.seed)
random.seed(hp.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_embedding(wav):
    print('getting d vector')
    #print(wav)
    #model_path = os.path.join(hp.train.checkpoint_dir, model_path)
    model_path = 'pretrained.pth'
    embedder_net = SpeakerRecognition(512, 5994, use_attention=False)
    embedder_net = torch.nn.DataParallel(embedder_net)
    embedder_net = embedder_net.cuda()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    s1 = extract_all_feat(wav, mode = 'test').transpose()
    s1 = torch.Tensor(s1).unsqueeze(0)
    e1, _ = embedder_net(s1.cuda())
    print(e1)
    return e1


if __name__=="__main__":
    wav_file_name = sys.argv[1]
    #print(wav_file_name)
    res = get_embedding(wav_file_name)
    print(type(res))
    print('type of embedding' )
    print(res[0].shape)
    print(res)
