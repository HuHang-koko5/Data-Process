from tokenizers import ByteLevelBPETokenizer
import os

path = './sentence.txt'

tkzer = ByteLevelBPETokenizer()
tkzer.train(files=path,
           vocab_size=50265,
           min_frequency=2,
           special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])

os.mkdir('./tkz')
tkzer.save(path=os.getcwd()+'/tkz')