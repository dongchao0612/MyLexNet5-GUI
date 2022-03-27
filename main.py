import os

import cv2 as cv
import tkinter as tk
import tkinter.messagebox
from tkinter.filedialog import askopenfilename

import torch
import torch.nn as nn
import torchvision



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


class Identifier():
    def __init__(self):

        self.network = LeNet()
        self.network.load_state_dict(torch.load("./model/LeNet.pkl"))

    def identify_number(self, filepath):
        src = cv.imread(filepath)
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (28, 28))
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        to_compse = torchvision.transforms.ToTensor()
        img = to_compse(binary).resize(1, 1, 28, 28)
        test_output = self.network(img)

        pred_y = torch.max(test_output, 1)[1].item()
        return pred_y


class Frame(tk.Tk):
    def __init__(self):
        super().__init__()
        self.filename = None
        self.title("手写数字识别器")
        self.geometry("370x200")
        self.identifier = Identifier()
        self.init()

    def init(self):

        self.e1 = tk.Entry(self)
        self.e1.place(x=80, y=30)
        self.e2 = tk.Entry(self)
        self.e2.place(x=80, y=110)


        self.b1 = tk.Button(self, text="选择文件", command=self.getpath)
        self.b1.place(x=250, y=28)

        self.b2 = tk.Button(self, text="开始识别", command=self.do_identify)
        self.b2.place(x=250, y=108)

        self.b3 = tk.Button(self, text="退出",bg="red" ,command=self.quit)
        self.b3.place(x=290, y=160)

    def getpath(self):
        self.filename = askopenfilename()

        self.e1.delete(0, tk.END)
        self.e1.insert(0, self.filename)

    def do_identify(self):
        try:
            print(self.e1.get())
            result = str(self.identifier.identify_number(self.e1.get()))
            self.e2.delete(0, tk.END)
            self.e2.insert(0, '识别结果：' + result)
        except Exception as e:
            print(e)
            tkinter.messagebox.showwarning(title='识别失败', message='文件目录请不要含有中文！')


if __name__ == '__main__':
    photo_ls = os.listdir('img')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identifier = Identifier()
    win = Frame()
    win.mainloop()
