from torch import nn
import torch

import clusterings
import numpy as np


class InformationBottleneck(nn.Module):
    def __init__(self, embedding_size, beta, device=None):
        super().__init__()
        self.beta = beta
        self.device = device
        self.initial_value = 5.0
        self.sigmoid = nn.Sigmoid()
        self.embedding_size = embedding_size
        
        self._alpha_bound = 5.0
        
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = nn.Linear(in_features=embedding_size, out_features=embedding_size//2)
        # self.bn1 = nn.BatchNorm1d(embedding_size//2)
        self.layer2 = nn.Linear(in_features=embedding_size//2, out_features=embedding_size*2)
        # self.bn2 = nn.BatchNorm1d(embedding_size*2)
        self.layer3 = nn.Linear(in_features=embedding_size*2, out_features=embedding_size)
        
        self.layer4 = nn.Linear(in_features=embedding_size, out_features=1)
        # self.bn3 = nn.BatchNorm1d(embedding_size)
        

        with torch.no_grad():
            nn.init.constant_(self.layer4.bias, 5.0)  # 偏置设为某个常数,因为是经过relu之后，所以layer4的输出主要是根据bias的值进行的，通过限制是5，我们可以让alpha的初始值设置为5
            self.layer4.weight *= 1e-3               # 权重初始值很小
                
        #self.alpha = torch.nn.Parameter(torch.full((1,embedding_size), self.initial_value))
        self.buffer_capacity = None
        
        self.fitting_estimator = torch.nn.CosineSimilarity(eps=1e-6, dim=-1)
        #self.reset_alpha()
        
        self.clusterer = clusterings.SpectralClustering()
    
    def forward_alpha(self, x):
        # alpha = self.alpha
        # regard seqlen、1 as hw， embeddingsize as channel
        # x = x.permute(0, 2, 1).unsqueeze(-1)
        
        alpha = self.layer1(x) # x.shape:torch.Size([2, 50, 768]), alpha.shape: torch.Size([2, 50, 384])
        # alpha = self.bn1(alpha)
        alpha = self.relu(alpha)
        alpha = self.layer2(alpha)
        # alpha = self.bn2(alpha)
        alpha = self.relu(alpha)
        alpha = self.layer3(alpha)
        # alpha = self.bn3(alpha)
        alpha = x + alpha # alpha.shape:torch.Size([2, 50, 768])
        alpha = self.relu(alpha)
        alpha = self.layer4(alpha)


        alpha = torch.clamp(alpha, -self._alpha_bound, self._alpha_bound)

        return alpha
    
    @staticmethod
    def _sample_t(mu, noise_var):
        #log_noise_var = torch.clamp(log_noise_var, -10, 10)
        noise_std = noise_var.sqrt()                         # 这里容易引起NAN，因为根号x在0处不可导，而根据传入参数当lamda=1的时候var可能是0，然后就会引起NAN
        eps = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps

    @staticmethod
    def _calc_capacity(mu, var):
        # KL[P(t|x)||Q(t)] where Q(t) is N(0,1)
        kl =  -0.5 * (1 + torch.log(var) - mu**2 - var)
        return kl

    def reset_alpha(self):
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)
        return self.alpha

    def forward(self, x, **kwargs):
        alpha = self.forward_alpha(x)
        lamb = self.sigmoid(alpha)

        # lamb = self.sigmoid(self.alpha)
        # lamb = lamb.expand(x.shape[0], x.shape[1], -1)
        
        masked_mu = x * lamb     # 这个是均值，因为
        # masked_var = (1-lamb)**2
        # 建议的稳定版本：
        limit_var = 1e-3 # 如果还是可能nan的话，可以弄成1e-3
        masked_var = ((1 - lamb).clamp(min=limit_var)) ** 2  # 通过限制最小值，来防止在self._sample_t()在反向传播的出现Nan
        self.buffer_capacity = self._calc_capacity(masked_mu, masked_var)
        t = self._sample_t(masked_mu, masked_var)
        return (t,lamb,alpha)
    
    def calc_loss(self, text_embedding, image_embedding):

        # text_embedding=[2, 512, 768]      ; image_embedding=[2, 50, 768]
        compression_term = self.buffer_capacity.mean()
        image_feature = image_embedding.unsqueeze(2)
        text_feature = text_embedding.unsqueeze(1)
        fitting_term = self.fitting_estimator(image_feature, text_feature).mean()
        # total = self.beta * compression_term +(1-fitting_term)
        # return total,compression_term,1-fitting_term
        total = self.beta * compression_term - fitting_term
        return total,compression_term,fitting_term
    
    
    def Spectral_Cluster_Voting(
            self,
            Image_Token,
            Compression_Scores,
            cluster_nums = 4,   # 聚类的数量，默认是4
            
    ):
        
        if Image_Token.shape[0] != 1:
            raise ValueError("batchsize应该是1,否则的话下面代码需要更改")
        
        # 深度拷贝, 因为我们Spectral_Cluster_Voting只是为了进行选择出token id，而并不是为了
        Image_Token = Image_Token.permute(0, 2, 1).contiguous().clone().detach().to(self.buffer_capacity.device)  # [b, D, N] 需要适应cluster代码
        
        # clustering
        # batch_clusters: b (=1) x N
        clusters = self.clusterer(Image_Token, cluster_nums)[0]  # 这个[0],前提是batch_size=1


        avg_scores = []
        for cid in range(cluster_nums):
            mask = (clusters == cid)
            if np.sum(mask) == 0:
                avg = -np.inf  # 如果该类没有token，避免选择
            else:
                avg = np.mean(Compression_Scores[mask])
            avg_scores.append(avg)
        # 1.这个是选择第一个簇的id
        # best_cluster_id = np.argmax(avg_scores)
        # # 返回属于该簇的 token 的索引
        # best_token_indices = np.where(clusters == best_cluster_id)[0]
        # # print(best_token_indices)
        # # print("聚类数量为：",cluster_nums)
        # return best_token_indices  # shape: [K]，K 是该聚类中的 token 数量

        # 2. 选择前 cluster_nums - 1 个簇的 token 索引
        # 获取按平均得分排序后的聚类id（从高到低）
        sorted_cluster_ids = np.argsort(avg_scores)[::-1]  # 例如：[2, 0, 1, 3]

        # 选择前 cluster_nums - 1 个聚类的 token 索引（排除得分最低的一个）
        selected_cluster_ids = sorted_cluster_ids[:cluster_nums - 1]

        # 收集这些聚类中所有 token 的索引
        selected_token_indices = []
        for cid in selected_cluster_ids:
            selected_token_indices.extend(np.where(clusters == cid)[0])

        selected_token_indices = np.array(selected_token_indices)
        return selected_token_indices  # shape: [K]，K 是被保留的 token 数量
    
    def tokenchoose(self,image_embedding, hidden_states):
        
        # 0.993 [1,50,768]
        image_embedding, iba_lamb, iba_alpha = self.forward(image_embedding)
        
        # 
        loss_M = -(iba_lamb * torch.log(iba_lamb + 1e-12)+ (1-iba_lamb) * torch.log((1-iba_lamb) + 1e-12)).mean()

        
        # ###
        # log_data = {"iba_alpha": iba_alpha[0].squeeze(-1).detach().cpu().numpy().tolist()}
        # with open(self.log_path_lamda, "a") as f:
        #     f.write(json.dumps(log_data) + "\n")
            
        # log_data_1 = {"lamda": iba_lamb[0].squeeze(-1).detach().cpu().numpy().tolist()}
        # with open(self.log_path_lamda, "a") as f:
        #     f.write(json.dumps(log_data_1) + "\n")

    
        iba_loss,compression_term,fitting_term = self.calc_loss(hidden_states, image_embedding)
        # log_data_loss_2 = {"IB-loss": iba_loss.item(),"compression_term": compression_term.item(), "fitting_term": fitting_term.item() }
        # with open(self.log_path_loss, "a") as f:
        #     f.write(json.dumps(log_data_loss_2) + "\n")
    

        # # 1.使用image/text token的余弦相似度来进行token choose
        # image_embedding_unseq = image_embedding.unsqueeze(2)  # [batch_size, seq_len, 1, hidden_size]
        # text_feature = hidden_states.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_size]
        # similarity_matrix = F.cosine_similarity(image_embedding_unseq, text_feature, dim=3)  # [batch_size, seq_len, seq_len]
        # # 使用 topk 选择相似度最高的 k 个特征
        # topk_indices = torch.topk(similarity_matrix, k=k, dim=1).indices[ :, :, 0]  # [batch_size, k]
        # #---------------------------------------#


        # 2.使用Information Bottleneck来进行token choose
        mask_ori = self.buffer_capacity # [bs, 50, 768]
        mask = torch.nansum(mask_ori, -1)  #  [bs, 50]  --- nansum是忽略nan的值来进行加和
        min_vals = mask.min(dim=1, keepdim=True).values
        max_vals = mask.max(dim=1, keepdim=True).values
        # Min-Max 归一化
        mask_norm = (mask - min_vals) / (max_vals - min_vals)

        # log_data_mask = {"mask_norm": mask_norm[0].squeeze(-1).detach().cpu().numpy().tolist()}
        # with open(self.log_path_lamda, "a") as f:
        #     f.write(json.dumps(log_data_mask) + "\n")

        # topk_scores, topk_indices = torch.topk(mask_norm, k, dim=1)  # 都是 [bs, k]
        #---------------------------------------#
        
        sc_indices = self.Spectral_Cluster_Voting(
            image_embedding[:, 1:],                      # 去掉 cls token
            mask_norm[0].cpu().detach().numpy()[1:],      # 同步去掉 cls token 对应的 score
            cluster_nums = 4
        )
        # 补上偏移
        sc_indices = sc_indices + 1
        topk_indices = torch.from_numpy(sc_indices).to(self.buffer_capacity.device).unsqueeze(0)  

        selected_features = torch.gather(image_embedding, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, image_embedding.shape[-1]))        
        image_embedding = selected_features # [bs, k, 768]  
        
        return image_embedding, iba_loss, loss_M