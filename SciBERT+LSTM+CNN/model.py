from libs import *
from functions import *
##Bert model
PRE_TRAINED_MODEL_NAME = 'allenai/scibert_scivocab_uncased'
###Model SciBERT+LSTM+CNN
class RNN(nn.Module):
    def __init__(self, out=14, out_src=0):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(bidirectional=True, num_layers=2, dropout=0.5, input_size=3840, hidden_size=250,batch_first=True)#the hidden_size is half of the outputsize
        self.dropout = torch.nn.Dropout(0.2)
        #self.fc = nn.Linear(400, out) ###50%²:XA
        

    
    def forward(self, x):
       # x= x.unsqueeze(1)
       ##delet the epochs
        #x= torch.squeeze(x,0)
        x=  x.unsqueeze(0)
        #print(x)
        #x = x[0,:,:,:]
        #print(len(x))
        #print(x.shape)
        x = x.permute(0,1,2)#seq length (number of featuires) , batch size , inputs size (number of sequences)
        #print(len(x))
        x, _ = self.rnn(x)
       # x = x[:, 0, :]
        x = self.dropout(x)
        #print("LSTM Forward",x.shape)
        #x = self.fc(x)
        return x
#####################################################"
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels = 1,out_channels= 40,kernel_size =2,stride = 1),##stride= 1,kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(20, 60, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(3780,14)##n-channels * size of the maxpool layer

        
#############################################"
    def forward(self, x):
        #print("Input shape before squeeze:",x.shape)
        #x= x.unsqueeze(1)
        x = x.permute(1,0,2)
       # print("CNN: Input shape",x.shape)
        #x = x.permute(2,1,3,0)        
        out = self.layer1(x)
        #print("output shape after first layer", out.shape)
        out = self.layer2(out)
        #print("output shape after second layer",out.shape)
        out = out.reshape(out.shape[0], -1)
        #out= out.view(batch_size:,-1)
        #print("Output shape after reshaping",out.shape)
        out = self.drop_out(out)
        #print("CNN : The output shape after dropout",out.shape)
        out = self.fc1(out)
        #print(out.shape)
        #out = self.fc2(out)
        return out
        
        
class RelationClassifier(nn.Module):
  def __init__(self):
    super(RelationClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

    self.cnn = ConvNet()
    self.lstm =RNN()
  def forward(self, input_ids, attention_mask,d1,d2,pos1,pos2,device):
    last_hidden_state, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    #e1_embedding = last_hidden_state[pos1]
    #e2_embedding = last_hidden_state[pos2]
    #print("d1",d1)
    
    #print("d1 after",d1)
    #pos1 = int(pos1)
   # print(pos2)
    e1 = []
    e2= []
    for i in range(0,len(pos1)):
        x = [i for i in range(768)]
      #  print("pos2===================",pos2[i])
        if not(str(pos1[i]).isnumeric) or pos1[i] =="NONE": #if pos is string
           e1.append(x)
        #print(pos1[i])
        elif int(pos1[i])>199:
           e1.append(x)
        else:
           e1.append(last_hidden_state[i][int(pos1[i])].tolist())
        if not(str(pos2[i]).isnumeric) or pos2[i] =="NONE":
            e2.append(x)
        #print(pos2[i])
        elif int(float(pos2[i]))> 199:
            e2.append(x)
        else:
            e2.append(last_hidden_state[i][int(float(pos2[i]))].tolist())
    e1 = torch.tensor(e1)#.unsqueeze(0)
    e2 = torch.tensor(e2)#.unsqueeze(0)
    #print(d2)
    #d2 = d2.unsqueeze(0)
    #d1 = d1.unsqueeze(0)

    #print(d2.shape)
    #print(len(e1[1]))
    #pooled_output = pooled_output.unsqueeze(0)
    pooled_output = pooled_output.to(device)
    e1 = e1.to(device)
    e2 = e2.to(device)
    d1 = d1.to(device)
    d2 = d2.to(device)
   # print("d1============================================",d1)
   # print("shape d1=======================================",d1.shape,d2.shape,pooled_output.shape,e1.shape,e2.shape)
    #print(pooled_output.shape,d1.shape,d2.shape,e1.shape,e2.shape)
    ##the concatination is wrongI
    reshaped_inputs = []
    for i in range(len(d1)):
        a = pooled_output[i]
        b = d1[i]
        c = d2[i]
        d= e1[i]
        e=  e2[i]

        f = torch.cat((a.float(), b.float(), c.float(),d.float(),e.float()), 0)
        reshaped_inputs.append(f)

    #inputs = torch.cat((pooled_output.float(),e1.float(),e2.float(), d1.float(), d2.float()),0)
   # print("Shape of the LSTM INPUTS",len(reshaped_inputs))

    #print(inputs)
    #print(cnn_output.shape)
    #print("cnn_output =================================================",cnn_output)
    reshaped_inputs = torch.stack(reshaped_inputs)
    #print("Input size to the LSTM",reshaped_inputs.shape)
    lstm_output = self.lstm (reshaped_inputs)
   # print("lstm output shape",lstm_output.shape)
    #print(lstm_output)
    cnn_output = self.cnn(lstm_output)
    #print(inputs.shape)
    return cnn_output
