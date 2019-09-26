from model.efficientnet import get_efn_model
from src import config
import torch
import torch.nn as nn
from src.datautils import MyDataset
from src.metric import accuracy
import src.transform as my_trfs
import time
import os
from tqdm import tqdm
# from apex import amp
from sklearn import metrics

def apply_coef(output):
    coef = [0.5, 1.5, 2.5, 3.5] # 1.37
    # coef = [0.75, 1.75, 2.25, 3.25]
    for i, pred in enumerate(output):
        if pred < coef[0]:
            output[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            output[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            output[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            output[i] = 3
        else:
            output[i] = 4
    
    return output

    
def run_train(model, criterion, optimizer, train_loader):
    model.train()
    avg_loss = 0.
    optimizer.zero_grad()
    for i, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs_train, labels_train = imgs.cuda(), labels.float().cuda()
        output_train = model(imgs_train)
        loss = criterion(output_train,labels_train) / config.NUM_ACCUM
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        if i % config.NUM_ACCUM == 0:
            optimizer.step() 
            optimizer.zero_grad() 
        avg_loss += loss.item() / len(train_loader)
        
    return avg_loss

def run_eval(model, criterion, val_loader):
    model.eval()
    avg_val_loss = 0.
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for _, (imgs, labels) in enumerate(tqdm(val_loader)):
            imgs_vaild, labels_vaild = imgs.cuda(), labels.float().cuda()
            output_test = model(imgs_vaild)
            avg_val_loss += criterion(output_test, labels_vaild).item() / len(val_loader)
            y_pred = output_test.detach().cpu().numpy().reshape(-1).tolist()
            y_preds.extend(apply_coef(y_pred))
            y_true = labels_vaild.detach().cpu().numpy().reshape(-1).astype(int).tolist()
            y_trues.extend(y_true)

    print(y_preds[:20])
    print(y_trues[:20])
    val_kappa = metrics.cohen_kappa_score(y_preds, y_trues, [0, 1, 2, 3, 4], 'quadratic')
    val_acc = metrics.accuracy_score(y_trues, y_preds)

    return avg_val_loss, val_kappa, val_acc

def train(train_df, val_df, path, weight_path, fold_num=None, model_num=0,  n_epochs=10):

    model = get_efn_model(model_num=model_num, isPretrained=True)

    # TODO Maybe classification?
    model._fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model._fc.in_features, config.NUM_CLASSES, bias=True)
    )
    # n_fc = model._fc.in_features
    # model._fc = nn.Sequential(
    #     nn.Dropout(0.5),
    #     nn.Linear(in_features=n_fc, out_features=n_fc, bias=True),
    #     nn.ReLU(),
    #     nn.BatchNorm1d(n_fc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #     nn.Dropout(0.25),
    #     nn.Linear(in_features=n_fc, out_features=1, bias=True),
    # )

    if weight_path != None :
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt)
        print(weight_path + " loaded")

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-5) # TODO change
    criterion = nn.MSELoss() # TODO Another loss func

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=2, verbose=True)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, )

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0) 

    train_transform = my_trfs.train_transform_v2
    # train_transform = my_trfs.train_transform_v1
    test_transform = my_trfs.test_transform

    trainset     = MyDataset(train_df, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_CORES, pin_memory=True, drop_last=True)
    valset       = MyDataset(val_df, transform=test_transform)
    val_loader   = torch.utils.data.DataLoader(valset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_CORES, pin_memory=True)

    best_avg_loss = 100.0
    best_kappa = 0

    log_file = open(os.path.join(path, 'log'), 'a')
    log_file.write("fold : "+ str(fold_num) + "\n")

    for epoch in range(n_epochs):
        
        # print('lr:', scheduler.get_lr()[0]) 
        start_time   = time.time()
        avg_loss     = run_train(model, criterion, optimizer, train_loader)
        avg_val_loss, val_kappa, val_acc = run_eval(model, criterion, val_loader)
        elapsed_time = time.time() - start_time
        
        log = 'Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_acc={:.4f} \t val_kappa={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, avg_val_loss, val_acc, val_kappa, elapsed_time)
        print(log)
        log_file.write(log + "\n")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_avg_loss :
            best_avg_loss = avg_val_loss
            name = "{}/efn_b{}_nf{}_ep{}_vl{:.4f}_vk{:.4f}_acc{:.4f}.pt" \
                     .format(path, model_num, fold_num, epoch, avg_val_loss, val_kappa, val_acc)
  
            torch.save(model.state_dict(), name)
            continue

        if val_kappa > best_kappa :
            best_kappa = val_kappa
            name = "{}/efn_b{}_nf{}_ep{}_vl{:.4f}_vk{:.4f}_acc{:.4f}.pt" \
                     .format(path, model_num, fold_num, epoch, avg_val_loss, val_kappa, val_acc)
  
            torch.save(model.state_dict(), name)
            continue
    
    log_file.close()