from libs import *
from functions import *
##Bert model
PRE_TRAINED_MODEL_NAME = 'allenai/scibert_scivocab_uncased'


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels= 40,kernel_size = 2,stride = 1),##stride= 1,kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(40, 60, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(23160,14)##n-channels * size of the maxpool layer
        #self.fc2 = nn.Linear(1000, 14)
        
#############################################"
    def forward(self, x):
        #print("Input shape before squeeze:",x.shape)
        x= x.unsqueeze(1)
        #x = x.permute(1,0,2)
       # print("Input shape",x.shape)
        x = x.permute(2,1,3,0)        
        out = self.layer1(x)
        #print("output shape after first layer", out.shape)
        out = self.layer2(out)
        #print("output shape after second layer",out.shape)
        out = out.reshape(out.shape[0], -1)
        #out= out.view(batch_size,-1)
        #print("Output shape after reshaping",out.shape)
        out = self.drop_out(out)
        #print("The output shape after dropout",out.shape)
        out = self.fc1(out)
        #out = self.fc2(out)
        return out
        
        
class RelationClassifier(nn.Module):
  def __init__(self):
    super(RelationClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

    self.out = ConvNet()
  def forward(self, input_ids, attention_mask,d1,d2,pos1,pos2,device):
    last_hidden_state, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    #e1_embedding = last_hidden_state[pos1]
    #e2_embedding = last_hidden_state[pos2]
    #print("d1",d1)
    
    d1 = d1.unsqueeze(0).to(device)
    #print("d1 after",d1)
    #pos1 = int(pos1)
    #print(pos1)
    e1 = []
    e2= []
    for i in range(0,len(pos1)):
        x = [i for i in range(768)]
        if pos1[i] =="NONE":
           e1.append(x)
        # print(pos1[i])
        elif int(pos1[i]):
           e1.append(x)
        else:
           e1.append(last_hidden_state[i][int(pos1[i])].tolist())
        if pos2[i] =="NONE":
            e2.append(x)
        elif int(pos2[i]) > 199:
            e2.append(x)
        else:
            e2.append(last_hidden_state[i][int(pos2[i])].tolist())
    e1 = torch.tensor(e1).unsqueeze(0).to(device)
    e2 = torch.tensor(e2).unsqueeze(0).to(device)
    #print(d2)
    d2 = d2.unsqueeze(0).to(device)
    #print(d2.shape)
    #print(len(e1[1]))
    pooled_output = pooled_output.unsqueeze(0).to(device)
    #print(pooled_output.shape,d1.shape,d2.shape,e1.shape,e2.shape)
    inputs = torch.cat((pooled_output.float(),e1.float(),e2.float(), d1.float(), d2.float()),0)
    #print("Shape of the inputs to the CNN",inputs.shape)
    return self.out(inputs) 
