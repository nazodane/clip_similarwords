# Copyright 2022 Toshimitsu Kimura <lovesyao@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#from urllib.request import urlopen
#import json
#response = urlopen("https://raw.githubusercontent.com/danbooru/autotagger/55d343cd3d79ecb6e1c14b854cb02b6038013bbe/data/tags.json")
#x = json.loads(response.read())
#print(len(x)) # -> 5501

import os
from urllib.request import urlretrieve
import torch
fp = "./model-resnet_custom_v3.pt"
if not os.path.exists(fp):
    urlretrieve("https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt", \
                fp)
x = torch.load(fp)
x = x["tags"]
#print(len(x)) # -> 9176
import pickle
with open("tmp_danbooru_tags.pt", "wb") as f:
    pickle.dump(x, f)
