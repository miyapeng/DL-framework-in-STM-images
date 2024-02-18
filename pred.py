import torch
import numpy as np
#from model import efficientnet_b0 as create_model
from efficientnet_ca import *
from torchvision import transforms
import os
from PIL import Image
import json
import matplotlib.pyplot as plt


def pred():
    device = torch.device("cuda")
    model = efficientnet_b1(num_classes=args.num_classes).to(device)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    model.load_state_dict(torch.load(args.model_pth))
    data_path = args.data_folder
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    f = open("./classes_que.json",'r',encoding='utf-8')
    cls_name = json.load(f)
    plt.figure()
    model.eval()
    test1 = 0
    test2 = 0
    with torch.no_grad():
        for image in os.listdir(data_path):
            test1 += 1
            image_I = Image.open(os.path.join(data_path, image))
            image_T = transform(image_I)
            image_T = image_T.unsqueeze(0)
            output = model(image_T.to(device))
            output = torch.softmax(output, 1)
            output = output.squeeze()
            output_idx = torch.argmax(output)
            pred = output[output_idx.item()]
            name = cls_name[str(output_idx.item())]
            if name == "3":
                test2 += 1
            acc1 = test2/test1 * 100
            plt.title("预测类别: {} 预测概率: {}".format(name, pred))
            print(("预测类别: {} 预测概率: {}".format(name, pred)))
            plt.imshow(image_I)
            plt.show()
            plt.close()
        print(("类别3预测准确率: {}% ".format(acc1)))
        print("23预测结束！")



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str,default=r'./Si_defect_dataset_enhanced/test/3')
    #parser.add_argument('--data-folder', type=str,default=r'./Si_defect_dataset_enhanced/test/2')
    #parser.add_argument('--data-folder', type=str,default=r'./Si_defect_dataset_enhanced/test/1')
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--model-pth', type=str, default='./checkpoints_que_ca/40_efficientnet_b1_data-enchance.pth')
    parser.add_argument('--json', type=str, default='./classes_que.json')

    args = parser.parse_args()
    pred()