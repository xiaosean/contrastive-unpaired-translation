from packaging import version
import torch
from torch import nn
import numpy as np
import cv2

class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_q_ours, feat_k, feat_k_ours, ori_shape):
        # feat_q.shape == feat_k.shape
        batchSize = feat_q.shape[0]  # num_patches * batch_size(default is 1)
        dim = feat_q.shape[1]  # 256
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)  # [batchSize, 1]

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        choose_row = 100
        choose_col = 100
        row = int(ori_shape[-1] / 256 * choose_row)
        col = int(ori_shape[-1] / 256 * choose_col)
        chosen_idx = row * ori_shape[-1] + col
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_q_ours = feat_q_ours[chosen_idx].view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        feat_k_ours = feat_k_ours.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        npatches_ours = feat_k_ours.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        l_neg_curbatch_ours = torch.bmm(feat_q_ours, feat_k_ours.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)  # [batchSize, batchSize]
        l_neg_ours = l_neg_curbatch_ours.view(-1, npatches_ours)  # [batchSize, batchSize]

        # reshape size from [n * n] to [n, n]
        temp = l_neg_ours[0] / self.opt.nce_T * 1.5
        temp = torch.tensor([x if x < 5.5 else 5.5 for x in temp], dtype=torch.float)
        temp = torch.exp(temp)
        temp = temp.detach().cpu().numpy()
        temp = np.concatenate((temp, np.zeros(ori_shape[-1] * ori_shape[-1] - temp.size)), axis=0)
        img = temp.reshape((ori_shape[-1], ori_shape[-1]))

        # red dot point
        point = np.zeros([ori_shape[-1], ori_shape[-1], 3])
        sqrt = ori_shape[-1]
        range = int(sqrt / 64 * 2)
        row_min = max(row - range, 0)
        row_max = min(row + range, sqrt - 1)
        col_min = max(col - range, 0)
        col_max = min(col + range, sqrt - 1)
        point[row_min:row_max, col_min:col_max, 0] = 0
        point[row_min:row_max, col_min:col_max, 1] = 0
        point[row_min:row_max, col_min:col_max, 2] = 255

        point = cv2.resize(point, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        img2 = np.zeros([256, 256, 3])
        img2[:, :, 0] = img[:, :]
        img2[:, :, 1] = img[:, :]
        img2[:, :, 2] = img[:, :]
        output = cv2.addWeighted(point, 0.9, img2, 0.8, 0)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T  # pos + neg -> [batchSize, batchSize + 1]
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss, output
