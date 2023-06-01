import torch
import timm
import timeit

device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')
model = timm.create_model('efficientnet_b0')
model = model.to(device)

input = torch.rand(1,3,224,224).to(device)
Time = 0.0
for _ in range (10):
    start = timeit.default_timer()
    model(input)
    stop = timeit.default_timer()
    Time += stop - start
Time /= 10
FPS = 1/Time
print(FPS)
