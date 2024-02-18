import torch
from torchvision import transforms
from efficientnet_ca import *
from torchvision import datasets
import os
import json
import math
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from plot_loss_acc import plot_loss, plot_acc
import torch.optim.lr_scheduler as lr_scheduler

def save_model(model,args,epoch):
    save_dir = args.checkpoints + "/{}_{}_{}.pth".format(epoch,args.model_name,args.data_name)
    print("save model ckpt in {}".format(save_dir))
    if not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)
    torch.save(model.state_dict(), save_dir)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print("using device {}".format(device))
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}

    model_name = {"efficientnet_b0":efficientnet_b0, "efficientnet_b1":efficientnet_b1, "efficientnet_b2":efficientnet_b2,
                  "efficientnet_b3":efficientnet_b3, "efficientnet_b4":efficientnet_b4, "efficientnet_b5":efficientnet_b5,
                  "efficientnet_b6":efficientnet_b6, "efficientnet_b7":efficientnet_b7}

    data_transform = {"train":transforms.Compose([transforms.Resize((224,224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                      "val": transforms.Compose([transforms.Resize((224,224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_path = args.data_folder
    assert data_path,"{} does not exist".format(data_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform['val'])
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images to train, using {} iamges to validation!".format(train_num, val_num))

    cls = train_dataset.class_to_idx
    cls = dict((key, val) for (val,key) in cls.items())
    json_str = json.dumps(cls,ensure_ascii=False,indent=4)
    with open("./classes_que.json","w", encoding='utf-8') as f:
        f.write(json_str)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = model_name[args.model_name](num_classes=args.num_classes).to(device)

    if os.path.exists(args.weights):
        model_dict = model.state_dict()
        weights_dict = torch.load(args.weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if k in model_dict and model_dict[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # 损失函数默认求的是平均损失 所以轮次损失除以总批次而不是总图片个数
    loss_function = nn.CrossEntropyLoss()
    best_acc = 0
    loss_list = [[] for i in range(2)]
    acc_list = [[] for i in range(2)]

    for epoch in range(args.epochs):
        train_bar = tqdm(train_loader)
        model.train()
        train_epoch_loss = 0
        train_epoch_acc = 0
        for i , (images,labels) in enumerate(train_bar):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            output_idx = torch.argmax(output, dim=1)
            acc = torch.eq(output_idx,labels).sum().item()
            train_epoch_acc += acc
            train_epoch_loss += loss.item()
            train_bar.desc = "epoch: [{}/{}] train_loss: {:.3f}, train_acc: {:.3f}".format(epoch,args.epochs,train_epoch_loss/(i+1), train_epoch_acc/((i+1)*args.batch_size))

        model.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            val_epoch_loss = 0
            val_epoch_acc = 0
            for i,(images, labels) in enumerate(val_bar):
                images,labels = images.to(device), labels.to(device)
                output = model(images)
                output = torch.softmax(output, dim=1)
                loss = loss_function(output, labels)
                val_epoch_loss += loss.item()
                output_idx = torch.argmax(output,dim=1)
                acc = torch.eq(output_idx,labels).sum().item()
                val_epoch_acc += acc
                val_bar.desc = "epoch: [{}/{}]  val_loss: {:.3f}, val_acc: {:.3f}".format(epoch, args.epochs, val_epoch_loss/(i+1),  val_epoch_acc/((i+1)*args.batch_size))

        train_epoch_loss = train_epoch_loss / len(train_loader)
        train_epoch_acc = train_epoch_acc / train_num
        val_epoch_loss = val_epoch_loss / len(val_loader)
        val_epoch_acc = val_epoch_acc / val_num
        loss_list[0].append(train_epoch_loss)
        loss_list[1].append(val_epoch_loss)
        acc_list[0].append(train_epoch_acc)
        acc_list[1].append(val_epoch_acc)

        if epoch >0 and epoch % 5==0:
            plot_loss(loss_list, "9", args)
            plot_acc(acc_list, "10", args)

        print("train_loss_epoch: {:.3f} train_acc_epoch: {:.3f} val_loss_epoch: {:.3f} val_acc_epoch: {:.3f}".format(train_epoch_loss,train_epoch_acc,val_epoch_loss,val_epoch_acc))

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            save_model(model,args,epoch)

    print("train done!!")

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder",type=str,default=r"./Si_defect_dataset_enhanced")
    parser.add_argument('--data-name', type=str, default="data-enchance")
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_que_ca')
    parser.add_argument('--fig-dir', type=str, default='./fig_que_ca')
    parser.add_argument('--model-name', type=str, default='efficientnet_b1')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    #放入预训练权重
    parser.add_argument('--weights', type=str, default='./pretrained./efficientnet-b0-355c32eb.pth',
                        help='initial weights path')

    args = parser.parse_args()
    if not os.path.exists(args.fig_dir):
        os.mkdir(args.fig_dir)
    main()