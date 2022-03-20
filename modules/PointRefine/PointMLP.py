# A simple MLP network structure for point clouds,
# 
# Added by Jiadai Sun

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointRefine(nn.Module):

    def __init__(self, n_class=3,
                 in_fea_dim=35,
                 out_point_fea_dim=64):
        super(PointRefine, self).__init__()

        self.n_class = n_class
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(in_fea_dim),

            nn.Linear(in_fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_point_fea_dim)
        )

        self.logits = nn.Sequential(
            nn.Linear(out_point_fea_dim, self.n_class)
        )

    def forward(self, point_fea):
        # the point_fea need with size (b, N, c) e.g.  torch.Size([1, 121722, 35])
        # process feature
        # torch.Size([124668, 9]) --> torch.Size([124668, 256])
        processed_point_fea = self.PPmodel(point_fea)
        logits = self.logits(processed_point_fea)
        point_predict = F.softmax(logits, dim=1)
        return point_predict


if __name__ == '__main__':

    import time
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PointRefine()
    model.train()

    # t0 = time.time()
    # pred = model(cloud)
    # t1 = time.time()
    # print(t1-t0)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of PointRefine parameter: %.2fM" % (total/1e6))
    # Number of PointRefine parameter: 0.04M
