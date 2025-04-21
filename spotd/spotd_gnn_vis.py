
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import random
import torch.nn as nn
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from pytorch_revgrad import RevGrad
from spotd.utils import *
import torch
import torch.nn.functional as F
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GCNConv

torch.autograd.set_detect_anomaly(True)


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads # // 8
        self.dim = dim # // 2048
        self.d_k = dim // num_heads

        # Ensure the dimension of the model is divisible by the number of heads
        assert dim % num_heads == 0, "Dimension must be divisible by the number of heads."

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dim, dim)

        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)

        # Feed-forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ff_dim, dim)
        )
        self.layer = nn.Sequential(nn.Linear(dim, dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.LayerNorm(dim),
                                   nn.Linear(dim, dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.LayerNorm(dim),
                                   )
        self.init_weights()

    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        # print (x.shape)
        # Transform inputs to (batch_size, num_heads, seq_length, d_k)
        query = self.query(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        x = torch.matmul(attention, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        # Feed-forward network
        x = self.layer(x.squeeze(1))
        return x
    

class Encoder(nn.Module):
    def __init__(self, dim, dim2, num_heads, ff_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(dim, dim2)
        self.relu   = nn.LeakyReLU(0.2, inplace=True)
        self.ln     = nn.LayerNorm(dim2)
        self.transformer = TransformerBlock(dim2, num_heads, ff_dim)
        self.init_weights()

    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.ln(x)
        # x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        # print ('inner encoder', x.shape)
        return x


class LearnedPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=22118):
        super(LearnedPositionEncoding, self).__init__()
        self.max_len = max_len
        self.coord1_embedding = nn.Embedding(max_len, d_model // 2)
        self.coord2_embedding = nn.Embedding(max_len, d_model // 2)

    def forward(self, x, coord1, coord2):
        coord1_encoded = self.coord1_embedding(coord1)  # Assume coord1 is within range
        coord2_encoded = self.coord2_embedding(coord2)  # Assume coord2 is within range
        # Check coordinate ranges
        if coord1.max() >= self.max_len or coord2.max() >= self.max_len:
            raise ValueError("Coordinate values exceed max_len")
        # Concatenate the embeddings along the last dimension
        position_encoded = torch.cat((coord1_encoded, coord2_encoded), dim=-1)
        if x.shape[1] != position_encoded.shape[1]:
            raise ValueError(f"Shape mismatch: x has shape {x.shape} but position encoding has shape {position_encoded.shape}")
        return x + position_encoded
    
class TransformerBlock_De(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock_De, self).__init__()
        self.num_heads = num_heads # // 8
        self.dim = dim # // 2048
        self.d_k = dim // num_heads

        # Ensure the dimension of the model is divisible by the number of heads
        assert dim % num_heads == 0, "Dimension must be divisible by the number of heads."

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dim, dim)

        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)

        # Feed-forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ff_dim, dim)
        )
        self.pos_encoder = LearnedPositionEncoding(dim)

        self.layer = nn.Sequential(nn.Linear(dim, dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.LayerNorm(dim),
                                   nn.Linear(dim, dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.LayerNorm(dim),
                                   )
        self.init_weights()

    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x, vis_data, coord1, coord2):
        batch_size = x.size(0)
        x = self.pos_encoder(x, coord1, coord2)
        # Use vis_data for Q and K
        vis_query = self.query(vis_data).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        vis_key = self.key(vis_data).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Transform inputs to (batch_size, num_heads, seq_length, d_k)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(vis_query, vis_key.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        x = torch.matmul(attention, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)

        # Feed-forward network
        x = self.layer(x.squeeze(1))
        return x
    


class Decoder(nn.Module):
    def __init__(self, dim, dim2, num_heads, ff_dim):
        super(Decoder, self).__init__()
        self.transformer = TransformerBlock_De(dim * 2, num_heads, ff_dim)
        self.linear = nn.Linear(1024, dim2)
        self.relu   = nn.LeakyReLU(0.2, inplace=True)
        self.ln     = nn.LayerNorm(dim2)

        self.linear_1024 = nn.Linear(dim, 1024)
        self.ln_1024     = nn.LayerNorm(1024)

        # self.pos_encoder = LearnedPositionEncoding(dim)
        self.init_weights()

    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x, x_vis, coord1, coord2):
        # x = x.unsqueeze(1)  # Add sequence dimension
        x = self.linear_1024(x)
        x = self.relu(x)
        x = self.ln_1024(x)
    
        x = self.transformer(x, x_vis, coord1, coord2)
        # x = x.squeeze(1)
        x = self.linear(x)
        x = self.relu(x)
        x = self.ln(x)
        return x


class MultiTaskPredictor(nn.Module):
    def __init__(self, in_dim, out_dim1, out_dim2):
        super(MultiTaskPredictor, self).__init__()
        # Shared layers
        self.shared_layer = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256)
        )
        
        # Task-specific heads
        self.classifier1 = nn.Sequential(
            nn.Linear(256, out_dim1),
            nn.Softmax(dim=1)
        )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(256, out_dim2),
            nn.Softmax(dim=1)
        )
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        shared_representation = self.shared_layer(x)
        output1 = self.classifier1(shared_representation)
        output2 = self.classifier2(shared_representation)
        return output1, output2

class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        in_dim, h_dim, out_dim = dim
        self.layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(h_dim),

            nn.Linear(h_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(out_dim),
            nn.Sigmoid()
            )
        self.init_weights()



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.layer(x)
        return out

class Classifier(nn.Module):
    def __init__(self, dim):
        super(Classifier, self).__init__()
        in_dim, h_dim, out_dim = dim
        self.layer = nn.Sequential(nn.Linear(in_dim, h_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.LayerNorm(h_dim),
                                   nn.Linear(h_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.LayerNorm(out_dim),
                                   nn.Sigmoid()
                                   )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.layer(x)
        return out
    
class spotdf(nn.Module):
    def __init__(self, celltype_num, spottype_num, outdirfile, used_features, num_epochs):
        super(spotdf, self).__init__()
        self.num_epochs_new = num_epochs
        self.batch_size = 256
        self.target_type = "real"
        self.learning_rate = 0.01
        self.celltype_num = celltype_num
        self.spottype_num = spottype_num
        self.labels = None
        self.used_features = used_features
        self.seed = 2021
        self.outdir = outdirfile
        cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        feature_num = len(self.used_features)
        print ('feature_num', feature_num)
        feature_num_stat = 512  
        # Encoder
        encoder_dim = feature_num
        self.encoder_da = Encoder(dim=encoder_dim, dim2=feature_num_stat, num_heads=8, ff_dim=512).cuda()
        decoder_dim = feature_num_stat
        # dim1=[512,1024,feature_num]
        self.decoder_da = Decoder(dim=feature_num_stat, dim2=encoder_dim, num_heads=8, ff_dim=512).cuda()
        predictor_dim = feature_num_stat
        self.predictor_da = MultiTaskPredictor(in_dim=predictor_dim, out_dim1=self.celltype_num, out_dim2=self.spottype_num).cuda()
        dim2=[256,64,1]
        discriminator_dim = feature_num_stat // 2
        self.Discriminator = Discriminator(dim2).cuda()
        dim3=[256,128,1]
        classifier_dim = feature_num_stat // 2
        self.Classifier = Classifier(dim3).cuda()
        
        self.gnnencoder = GCN(feature_num, hidden_dim=512).cuda()

        print (self.encoder_da)
        print (self.decoder_da)
        print (self.Discriminator)
        print (self.predictor_da)
        print (self.Classifier)
        print (self.gnnencoder)


    def forward(self, x, lamda=1, if_simulated=False, coord=None, vis_data=None):
        self.revgrad = RevGrad(lamda).cuda()
        x = x.cuda()
        embedding_source = self.encoder_da(x)
        if if_simulated: 
            pro, pro_spot = self.predictor_da(embedding_source)
            znoise = embedding_source[:, :256]
            zbio = embedding_source[:, 256:]
            clas_out = self.Classifier(zbio)
            disc_out = self.Discriminator(self.revgrad(znoise))
            return embedding_source, pro, clas_out, disc_out, pro_spot
        con_source = self.decoder_da(embedding_source, vis_data, coord[:, 0], coord[:, 1])
        pro, pro_spot = self.predictor_da(embedding_source)
        znoise = embedding_source[:, :256]
        zbio = embedding_source[:, 256:]
        clas_out = self.Classifier(zbio)
        disc_out = self.Discriminator(self.revgrad(znoise))
        return embedding_source, con_source, pro, clas_out, disc_out, pro_spot
    
    def prepare_dataloader(self, sm_data, sm_label, st_data, st_label, st_coord, st_vis_feat, batch_size):
        self.source_data_x = sm_data.values.astype(np.float32)
        self.source_data_y = sm_label.values.astype(np.float32)
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)
        self.used_features = list(sm_data.columns)
        self.target_data_x = torch.from_numpy(st_data.values.astype(np.float32))
        self.target_data_y = torch.from_numpy(st_label.values.astype(np.float32))
        self.target_coords = torch.from_numpy(st_coord.astype(np.int64))
        self.target_vis = torch.from_numpy(st_vis_feat.astype(np.float32))
        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        te_coords = torch.LongTensor(self.target_coords)
        te_vis = torch.FloatTensor(self.target_vis)
        target_dataset = Data.TensorDataset(te_data, te_labels, te_coords, te_vis)

        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)
    

    def mask_features(self, X, mask_ratio):
        if isinstance(X, np.ndarray):
            mask = np.random.choice([True, False], size=X.shape, p=[mask_ratio, 1 - mask_ratio])
        elif isinstance(X, torch.Tensor):
            mask = torch.rand(X.shape) < mask_ratio
        else:
            raise TypeError("type error!")
        use_x = X.clone()  
        use_x[mask] = 0
        return use_x, mask

    def build_graph(self, Y, k=5):
        if Y.is_cuda:
            Y = Y.cpu()
        Y = Y.numpy()
        k = min(k, len(Y) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(Y)
        _, indices = nbrs.kneighbors(Y, n_neighbors=k + 1)
        edges = []
        num_samples = len(Y)
        for i in range(num_samples):
            for j in range(1, k + 1):
                neighbor_index = indices[i][j]
                if neighbor_index < num_samples:
                    edges.append([i, neighbor_index])
                else:
                    print(f"Index {neighbor_index} out of bounds for sample {i}")
        edges = np.array(edges).T
        num_nodes = num_samples
        if edges.max() >= num_nodes:
            raise ValueError(f"Edge index out of bounds: {edges.max()} >= {num_nodes}")
        return torch.tensor(edges, dtype=torch.long)

    def prediction_overall_and_ratio(self):
        self.eval()
        preds = None
        con_source_list = [] 
        for batch_idx, (x, y, c, v) in enumerate(self.test_target_loader):
            x = x.cuda()
            c = c.cuda()
            v = v.cuda()
            embedding_source, con_source, pro, clas_out, disc_out, pro_spot = self.forward(x, coord=c, vis_data=v)
            logits = pro.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            con_source_list.append(con_source.detach().cpu().numpy())
        con_source_list = np.concatenate(con_source_list, axis=0)
        target_preds = pd.DataFrame(preds, columns=self.labels)
        model_and_settings = {
            'model': self.state_dict(),
            'seed': self.seed
        }

        torch.save(model_and_settings, self.outdir + '/model_with_settings.pth')
        print (con_source_list.shape)
        return target_preds, con_source_list  

    def prediction(self):
        self.eval()
        preds = None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            x = x.cuda()
            embedding_source, con_source, pro, clas_out, disc_out = self.forward(x)
            logits = pro.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
        target_preds = pd.DataFrame(preds, columns=self.labels)
        model_and_settings = {
            'model': self.state_dict(),
            'seed': self.seed
        }
        torch.save(model_and_settings, self.outdir + '/model_with_settings.pth')
        return target_preds

    def double_train(self, sm_data, sm_label, st_data, st_label, st_coord, st_vis_feat):
        self.train()
        self.prepare_dataloader(sm_data, sm_label, st_data, st_label, st_coord, st_vis_feat, self.batch_size)
        self.optim = torch.optim.Adam([{'params': self.encoder_da.parameters()},
                                          {'params': self.decoder_da.parameters()},], lr=self.learning_rate)
        self.optim1 = torch.optim.Adam([{'params': self.encoder_da.parameters()}, {'params': self.predictor_da.parameters()},], lr=self.learning_rate)
        self.optim_discriminator = torch.optim.Adam([{'params': self.encoder_da.parameters()}, {'params': self.Discriminator.parameters()},], lr=0.005)  
        self.optim_classifier = torch.optim.Adam([{'params': self.encoder_da.parameters()}, {'params': self.Classifier.parameters()},], lr=0.01)  
        criterion_da = nn.MSELoss().cuda()
        metric_logger = defaultdict(list)
        epsilon = 0.01
        lambda_1 = 0.01
        lambda_2 = 0.1
        lambda_3 = 0.1

        for epoch in range(self.num_epochs_new):
            train_target_iterator = iter(self.train_target_loader)
            pred_loss_epoch, pred_spot_loss_epoch, con_loss_epoch, con_tar_loss_epoch = 0., 0., 0., 0.
            dis_loss_epoch_y, dis_loss_epoch = 0.0, 0.0
            class_loss_epoch, class_loss_epoch_y = 0.0, 0.0
            all_loss_epoch=0.0
            for i in range(1):
                for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
                    try:
                        target_x, _, target_coord, target_vis = next(train_target_iterator)
                    except StopIteration:
                        train_target_iterator = iter(self.train_target_loader)
                        target_x, _, target_coord, target_vis = next(train_target_iterator)
                    use_x, mask = self.mask_features(source_x.cuda(), 0.3)
                    source_x = source_x.cuda()  
                    target_coord = target_coord.cuda()
                    target_vis = target_vis.cuda()
                    edges = self.build_graph(target_coord)
                    gnn_output = self.gnnencoder(target_x.cuda(), edges.cuda())
                    gnn_output = gnn_output.unsqueeze(1)
                    embedding_source, pro, clas_out, disc_out, _ = self.forward(source_x * (~mask.cuda()), if_simulated=True)
                    embedding_source_y, con_source_y, pro_y, clas_out_y, disc_out_y, _ = self.forward(gnn_output, coord=target_coord, vis_data=target_vis)
                    con_loss_tar = criterion_da(target_x.cuda() , con_source_y)

                    con_tar_loss_epoch += con_loss_tar.data.item()

                    source_label = torch.ones(disc_out.shape[0]).unsqueeze(1).cuda()  # 定义source domain label为1
                    source_label1 = source_label * (1 - epsilon) + (epsilon / 2)

                    target_label_y = torch.zeros(clas_out_y.shape[0]).unsqueeze(1).cuda()  # 定义target domain label为0
                    target_label_y1 = target_label_y * (1 - epsilon) + (epsilon / 2)
                    clas_loss = nn.BCELoss()(clas_out, source_label1)
                    dis_loss = nn.BCELoss()(disc_out, source_label1)
                    clas_loss_y = nn.BCELoss()(clas_out_y, target_label_y1)
                    dis_loss_y = nn.BCELoss()(disc_out_y, target_label_y1)

                    dis_loss_epoch += dis_loss.data.item()
                    dis_loss_epoch_y += dis_loss_y.data.item()
                    class_loss_epoch += clas_loss.data.item()
                    class_loss_epoch_y += clas_loss_y.data.item()
                    loss = con_loss_tar + lambda_1 * (dis_loss + dis_loss_y) + lambda_2 * (clas_loss + clas_loss_y)
                    all_loss_epoch += loss.data.item()

                    self.optim.zero_grad()
                    self.optim_discriminator.zero_grad()
                    self.optim_classifier.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.optim_discriminator.step()
                    self.optim_classifier.step()
                    torch.cuda.empty_cache()


            for i in range(1):
                for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
                    try:
                        target_x, target_y, target_coord, target_vis = next(train_target_iterator)
                    except StopIteration:
                        train_target_iterator = iter(self.train_target_loader)
                        target_x, target_y, target_coord, target_vis = next(train_target_iterator)

                    source_x = source_x.cuda()  
                    target_y = target_y.cuda()  
                    target_coord = target_coord.cuda()  
                    target_vis = target_vis.cuda()  

                    embedding_source, pro, clas_out, disc_out, _ = self.forward(source_x, if_simulated=True)
                    _, _, _, _, _, pro_spot = self.forward(target_x, coord=target_coord, vis_data=target_vis)
                    target_y_onehot = F.one_hot(target_y.to(torch.int64), num_classes=self.spottype_num)
                    # print (target_y_onehot.shape, pro_spot.shape, target_x.shape)

                    pred_loss = criterion_da(source_y.cuda(), pro)

                    pred_spot_loss = criterion_da(target_y_onehot.to(torch.float32).cuda(), pro_spot)

                    pred_loss_epoch += pred_loss.data.item()
                    pred_spot_loss_epoch += pred_spot_loss.data.item()
                    loss1 = pred_loss * lambda_3# + pred_spot_loss
                    self.optim1.zero_grad()
                    loss1.backward()
                    self.optim1.step()
                    torch.cuda.empty_cache()

            pred_loss_epoch = pred_loss_epoch / (batch_idx + 1)
            pred_spot_loss_epoch = pred_spot_loss_epoch / (batch_idx + 1)
            con_loss_epoch = con_loss_epoch / (batch_idx + 1)
            con_tar_loss_epoch = con_tar_loss_epoch / (batch_idx + 1)
            dis_loss_epoch_y = dis_loss_epoch_y / (batch_idx + 1)
            dis_loss_epoch = dis_loss_epoch / (batch_idx + 1)
            all_loss_epoch = all_loss_epoch / (batch_idx + 1)
            class_loss_epoch = class_loss_epoch / (batch_idx + 1)
            class_loss_epoch_y = class_loss_epoch_y / (batch_idx + 1)
            if epoch>0:
                metric_logger['pre_loss'].append(pred_loss_epoch)
                metric_logger['pre_spot_loss'].append(pred_spot_loss_epoch)
                metric_logger['con_loss'].append(con_loss_epoch)
                metric_logger['con_tar_loss'].append(con_tar_loss_epoch)
                metric_logger['dis_loss_y'].append(dis_loss_epoch_y)
                metric_logger['dis_loss'].append(dis_loss_epoch)
                metric_logger['all_loss'].append(all_loss_epoch)
                metric_logger['class_loss'].append(class_loss_epoch)
                metric_logger['class_loss_y'].append(class_loss_epoch_y)


            if (epoch+1) % 50== 0:
                print(
                    '============= Epoch {:02d}/{:02d} in stage ============='.format(epoch + 1, self.num_epochs_new))
                print(
                    "pre_loss=%f, pre_spot_loss=%f, con_loss=%f, con_tar_loss=%f, dis_loss_y=%f,dis_loss=%f,class_loss_y=%f, class_loss=%f,total_loss_DA=%f" % (
                        pred_loss_epoch, pred_spot_loss_epoch, con_loss_epoch, con_tar_loss_epoch, dis_loss_epoch_y, dis_loss_epoch, class_loss_epoch_y,
                        class_loss_epoch, all_loss_epoch))

        if self.target_type == "simulated":
            SaveLossPlot(self.outdir, metric_logger,
                         loss_type=['pred_loss', 'disc_loss', 'disc_loss_DA', 'target_ccc', 'target_rmse',
                                    'target_corr'], output_prex='Loss_metric_plot_stage3')
        elif self.target_type == "real":
            print (self.target_type)
            SaveLossPlot(self.outdir, metric_logger,
                         loss_type=['pre_loss', 'pre_spot_loss', 'con_loss', 'dis_loss_y', 'dis_loss', 'class_loss_y', 'class_loss',
                                    'all_loss'],
                         output_prex='Loss_metric_plot_stage')

