import torch
import argparse
import os
from dataloader import *
from network import *
from Nmetrics import evaluate
from tqdm import tqdm
import pandas as pd
from torch_kmeans_best import TorchKMeans
import warnings
warnings.filterwarnings("ignore")



def seed_everything(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def pd_toExcel(my_dic, fileName):
    Mrs, ACCs, NMIs, ARIs = [], [], [], []
    for i in range(len(my_dic)):
        Mrs.append(my_dic[i]["Missrate"])
        ACCs.append(my_dic[i]["ACC"])
        NMIs.append(my_dic[i]["NMI"])
        ARIs.append(my_dic[i]["ARI"])

    dfData = {
        'Missrate': Mrs,
        'ACC': ACCs,
        'NMI': NMIs,
        'ARI': ARIs,
    }
    df = pd.DataFrame(dfData)
    df.to_excel(fileName, index=False)


def pretrain():
    t_progress1 = tqdm(range(args.Epoch1), desc='Pretraining')
    for epoch in t_progress1:
        tot_losses = {
            "rec": 0.0
        }
        for batch_idx, (xs, _, miss_vecs, _) in enumerate(train_loader):
            for v in range(args.view_num):
                xs[v] = xs[v].to(device)
                miss_vecs[v] = miss_vecs[v].to(device)

            loss_rec = model.pretrain(xs, miss_vecs)

            loss_total = args.para_rec * loss_rec

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            tot_losses["rec"] += args.para_rec * loss_rec.item()

        log_str = " ".join([f"{k}: {v / len(train_loader):.4f}" for k, v in tot_losses.items()])
        t_progress1.set_postfix_str(log_str)

    model.initialize_cluster_centers(infer_loader, device)


def main_train():
    t_progress2 = tqdm(range(args.Epoch2), desc='Main Training')
    for epoch in t_progress2:
        tot_losses = {
            "rec": 0.0, "meta": 0.0, "causal": 0.0
        }
        for batch_idx, (xs, _, miss_vecs, _) in enumerate(train_loader1):
            for v in range(args.view_num):
                xs[v] = xs[v].to(device)
                miss_vecs[v] = miss_vecs[v].to(device)

            loss_rec, loss_meta, loss_causal = model.main_train(xs, miss_vecs)

            loss_total = args.para_rec * loss_rec + args.para_meta * loss_meta + args.para_causal * loss_causal

            optimizer1.zero_grad()
            loss_total.backward()
            optimizer1.step()

            tot_losses["rec"] += args.para_rec * loss_rec.item()
            tot_losses["meta"] += args.para_meta * loss_meta.item()
            tot_losses["causal"] += args.para_causal * loss_causal.item()

        log_str = " ".join([f"{k}: {v / len(train_loader1):.4f}" for k, v in tot_losses.items()])
        t_progress2.set_postfix_str(log_str)


def clustering_evaluation():
    model.eval()

    all_representations = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (xs, ys, miss_vecs, _) in enumerate(cluster_loader):
            for v in range(args.view_num):
                xs[v] = xs[v].to(device)
                miss_vecs[v] = miss_vecs[v].to(device)

            z_batch = model.get_representations(xs, miss_vecs)
            all_representations.append(z_batch.cpu())

            all_labels.append(ys[0])

    all_representations = torch.cat(all_representations, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()

    kmeans = TorchKMeans(
        n_clusters=args.cluster_num,
    )

    cluster_labels = kmeans.fit_predict(all_representations).cpu().numpy()

    acc, nmi, purity, fscore, precision, recall, ari = evaluate(all_labels, cluster_labels)
    
    return acc, nmi, purity, fscore, precision, recall, ari


if __name__ == '__main__':
    dataset = {
        0: "CUB",
        1: "Caltech-5V",
        2: "HW_6Views",
        3: "CiteSeer",
        4: "Scene-15_3V",
        5: "Reuters_5V_dim10",
        6: "YouTubeFace20_4Views",
        7: "FashionMNIST_4Views",
    }
    for data_id in dataset:
        All_Metrics = []
        file_name = "./All_Benchmarks/" + dataset[data_id] + ".xlsx"
        for mr in [0.5]:
            print(dataset[data_id])
            print('--------------------Missing rate = ' + str(mr) + '--------------------')

            seed_everything(42)  # 42

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            ###### Load data ######
            X, Y, Miss_vecs, cluster_num, data_num, view_num, view_dims = load_data(dataset[data_id], mr)
            idxs = np.array([i for i in range(data_num)])

            parser = argparse.ArgumentParser(description='train')
            parser.add_argument('--data_name', default=str(data_id), help='name of current dataset')
            parser.add_argument('--missrate', default=mr, help='missing rate of multi-view data')

            parser.add_argument('--cluster_num', default=cluster_num, help='number of clusters')
            parser.add_argument('--data_num', default=data_num, help='number of samples')
            parser.add_argument('--view_num', default=view_num, help='number of views')
            parser.add_argument('--view_dims', default=view_dims, help='dimension of views')
            parser.add_argument('--device', default=device)

            parser.add_argument('--Epoch1', default=100)
            parser.add_argument('--Epoch2', default=50)

            parser.add_argument('--hidden_dim', default=64, help='hidden dimension')
            parser.add_argument('--temperature', default=1.0, help='temperature for softmax')
            parser.add_argument('--batch_size', default=128, help='batch size')
            parser.add_argument('--lr', default=0.0003, help='learning rate')

            parser.add_argument('--para_rec', default=1.0)
            parser.add_argument('--para_meta', default=0.01)
            parser.add_argument('--para_causal', default=0.1)
            args = parser.parse_args()


            data_set = TrainDataset_All(X, Y, Miss_vecs, idxs)

            train_loader = torch.utils.data.DataLoader(
                data_set,
                batch_size=min(args.batch_size, data_num), 
                shuffle=True, 
                drop_last=True
            )

            infer_loader = torch.utils.data.DataLoader(
                data_set,
                batch_size=min(args.batch_size, data_num),
                shuffle=False,
                drop_last=False
            )

            model = CausalInvariantMetaGenerativeNetwork(
                view_dims=view_dims,
                hidden_dim=args.hidden_dim,
                cluster_num=cluster_num,
                temperature=args.temperature
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


            pretrain()

            train_loader1 = torch.utils.data.DataLoader(
                data_set,
                batch_size=min(args.batch_size, data_num),
                shuffle=True,
                drop_last=True
            )

            optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr)

            main_train()

            cluster_loader = torch.utils.data.DataLoader(
                data_set,
                batch_size=min(args.batch_size, data_num),
                shuffle=False,
                drop_last=False
            )


            acc, nmi, purity, fscore, precision, recall, ari = clustering_evaluation()
            print(f'ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}')

            result_dict = {
                "Missrate": mr,
                "ACC": acc,
                "NMI": nmi,
                "ARI": ari
            }

            All_Metrics.append(result_dict)
            pd_toExcel(All_Metrics, file_name)


