from ptflops import get_model_complexity_info
from archs.NEW_ARCH import FIVE_APLUSNet
import torch
import numpy as np
import time
model =FIVE_APLUSNet().cuda().eval()
H,W=1920,1080
flops_t, params_t = get_model_complexity_info(model, (3, H,W), as_strings=True, print_per_layer_stat=True)
print("Network :FIVE_APLUS")
print(f"net flops:{flops_t} parameters:{params_t}")
#model = nn.DataParallel(model)
x = torch.ones([1,3,H,W]).cuda()

b_1,b_stagehead = model(x)
steps=25
# print(b)
time_avgs=[]
with torch.no_grad():
    for step in range(steps):
        
        torch.cuda.synchronize()
        start = time.time()
        result = model(x)
        torch.cuda.synchronize()
        time_interval = time.time() - start
        if step>5:
            time_avgs.append(time_interval)
        #print('run time:',time_interval)
print('avg time:',np.mean(time_avgs),'fps:',(1/np.mean(time_avgs)),' size:',H,W)