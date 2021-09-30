from libs import *

torch.cuda.empty_cache()
# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    device2 = torch.device("cuda:1")#
    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
####hyper parameters#########

#this value detect which preproccessed data to use, for now there are 4 depending on the representation of position vectors associated with sentences where there are no entities or at most one entity exist, it takes 4 values z_vec, inc_vec, dec_vec, const_vec (len of the sentence+1)
p_v_rep = sys.argv[1] 
p_v_pad = sys.argv[2]#z_vec const_vec
p_v_normalize = sys.argv[3] #norm or w_norm 
sent_pad = sys.argv[4] #sentence padding z_sent (with vectors of zeros) , const_sent (len sentence +1 )  with vectors of (high numbers) , with vectors of (sympols) 
###############################""
##position vectors where no entity exist



#######Embeddings

class LabelToOneHot:
    def __init__(self, classes):
        self.classes = classes

    def label_to_one_hot(self, label):
        if label not in self.classes:
            #print(label)
            raise KeyError
        idx = self.classes.index(label)
        return idx # this is only a scalar, which is handled by the framework (code review)

def E(sent,label2vec):
        """Returns an embedding matrix of the graph's nodes."""
        assert('<###-unk-###>') in label2vec, 'The unknown token cannot be found.'
      
        tokenized_sent = word_tokenize(str(sent))
        if len(tokenized_sent) > 1:
          res = label2vec[tokenized_sent[0]] if tokenized_sent[0] in label2vec else label2vec['<###-unk-###>']
        
          for word in tokenized_sent[1:]:    #for label in lbls[1:]:
              if word in label2vec:
                  vec = label2vec[word]
              else:
                  vec = label2vec['<###-unk-###>']
              res = np.concatenate((res, vec), axis=0) #concatinate embeddings words to finally obtain the sentence embedding matrix
          #print('res (E - embedding matrix of the sentence words) = ', res, 'size = ', len(res), 'x', len(res[0]),"sent",tokenized_sent)
          return res#, lbls
        else: 
            if sent in label2vec:
                res = label2vec[sent] 
            else:
                res = label2vec['<###-unk-###>']
            return res
###convert inputs into samples 
class SentSample:
    def __init__(self, embedding, d1,d2,e1_embed,e2_embed, label=None):
        self.EMBEDDING = embedding
        self.d1 = d1 #position vector from entity 1 
        self.d2 = d2 #position vector from entity2
        self.e1_embed = e1_embed
        self.e2_embed  = e2_embed
        self.label = label
    def to_tensor(self):
        self.EMBEDDING = torch.from_numpy(self.EMBEDDING).float()
        self.d1 = torch.Tensor(self.d1).float()
        self.d2 = torch.Tensor(self.d2).float()
        self.e1_embed = torch.Tensor(self.e1_embed).float()
        self.e2_embed = torch.Tensor(self.e2_embed).float()
        return torch.cat((self.EMBEDDING,self.d1,self.d2, self.e1_embed,self.e2_embed),1)
        
###normalize psition vectors (d1,d2)
### The house is big : lets suppose that the entity is house d1 =[-1,0,1,2] the distence between each word in th esentence and the entity "house"
def normalize_pos(pos_vector):
    norm_pos_vector = []
    max_x = max(pos_vector)
    min_x = min(pos_vector)
    for ele in pos_vector:
        norm = (int(ele) -min_x)/((max_x - min_x)+0.00001)
        norm_pos_vector.append(norm)
    return norm_pos_vector
    
#######################
class LabelToOneHot:
    def __init__(self, classes):
        self.classes = classes
        #print(self.classes)
    def label_to_one_hot(self, label):
        if label not in self.classes:
           # print(label)
            raise KeyError
        idx = self.classes.index(label)
        return idx # this is only a scalar, which is handled by the framework (code review)

##############Dataset class
class DrugProtDataset(Dataset):
    """Converts Drugprot data into tesnors."""
    def __init__(self, root_dir, path_word2vec,pad = 150 ): ##path_wor2vec is the pickeled vocabulary
        self.root_dir = root_dir
        self.df = pd.read_csv(root_dir,sep = "|")

        # id|sentence|entity1|entity2|d1|d2|arg1|arg2|ent1_type1|ent2_type|label
        self.id = self.df["id"]
        self.sentence = self.df["sentence"]
        self.arg1 = self.df["arg1"]
        self.arg2 = self.df["arg2"]
        #print(self.df["pos1"])
        #print(type(self.sentence))
        self.d1 = self.df["pos1"]##entity1 index it could be NONE if no entity exist
        self.d2 = self.df["pos2"]##entity2index it could be NONE if no entity exist
        self.e1 = self.df["ent1_type1"]
        self.e2 = self.df["ent2_type"]
        self.relation = self.df["label"]
        self.label2onehot = LabelToOneHot(classes=DrugProtDataset.classes())
        self.label2vec = pickle.load(open(path_word2vec, 'rb')) ##load the embeddings
        #self.label2onehot = LabelToOneHot(classes=DrugProtDataset.classes())
        
    def classes():
        return ["INDIRECT-DOWNREGULATOR", "INDIRECT-UPREGULATOR", "DIRECT-REGULATOR", "ACTIVATOR", "INHIBITOR", "AGONIST", "ANTAGONIST", "AGONIST-ACTIVATOR", "AGONIST-INHIBITOR", "PRODUCT-OF", "SUBSTRATE", "SUBSTRATE_PRODUCT-OF" , "PART-OF","no_relation"]
    
    def __len__(self):
        return len(self.df)
        
    def PAD_Po_Vectors(self,d1,d2,p_v_pad): #p_v_pad extra argument for padding position vector 
    #"This function pad the possition vectors" 
        new_d1 =[]
        new_d2=[]
        if len(d1) ==200:
            #print("Yes PAD_pO 200")
            new_d1 = d1
            new_d2 = d2
        elif len(d1) <200:#pad with zeorss
           # print("No PAD_ <200")
           # print("p_v_pad and p_v_pad type",p_v_pad,str(p_v_pad)=="const_vec",type(p_v_pad))
           # print(p_v_stored)
            if str(p_v_pad) == "z_vec":
                new_d1 = d1+ [0] * (200 - len(d1))
                new_d2 = d2+ [0] * (200 - len(d2))
            elif str(p_v_pad) == "const_vec":#pad with constant
            #    print("p_v_pad is const_vec")
                new_d1 = d1+ [len(d1)+1] * (200 - len(d1))
                new_d2 = d2+ [len(d2)+1] * (200 - len(d2)) 
        elif len(d1) > 200:
            #print(">200")
            new_d1 = d1[:200]
            new_d2 = d2[:200]
        #print("The lenght of the postitions vectors: ",len(d1),len(d2))
        return new_d1,new_d2
        
        
    def __getitem__(self, index):
        sent = self.sentence[index]
        d1 = self.d1[index] ## entity1 index
        d2 = self.d2[index] ## entity2 index
        abs_id = self.id[index]
        arg1 = self.arg1[index]
        arg2 = self.arg2[index]
        d1 =  find_pos(sent,d1,p_v_rep)
        d2 = find_pos(sent,d2,p_v_rep)
        entity1 = self.e1[index]
        entity2 = self.e2[index]
        # normalize the position vectors
        if str(p_v_normalize) == "norm":
           d1 = normalize_pos(d1)
           d2 = normalize_pos(d2)
             
        
        ##use different embeddings with higher than 200
        ##padding position vectors
        
        new_d1, new_d2  = self.PAD_Po_Vectors(d1, d2,p_v_pad)
        relation = self.relation[index]
        ###Embeddings: convert words into normalized version

        ## #convert relation to one hot vector
        label = self.label2onehot.label_to_one_hot(relation)
        
        ##################
        sent_embed = E (sent, self.label2vec) #return the sentence embedding 
        #print("type of entity1", type(entity1))
        #print("type pf entity 2 ",type(entity2))
        e1_embed =  E(entity1,self.label2vec)
        e2_embed =  E(entity2,self.label2vec)
        arr1 = np.array(e1_embed)
        arr2 = np.array(e2_embed)
        arr3 = np.array([new_d1])
        arr4 = np.array([new_d2])
       
        concats = np.concatenate((sent_embed,arr1,arr2),axis=0)
   
        concats = np.concatenate ((concats,arr3,arr4),axis=0)
        if len(concats !=300):
               if str(sent_pad) == "z_sent":
                   concats = np.pad(concats,((0,300-len(concats)),(0,0)),constant_values = 0,mode ='constant')# positions of pad to add at the end of the matrix
               elif str(sent_pad) =="const_sent":
                   concats = np.pad(concats,((0,300-len(concats)),(0,0)),constant_values = len(concats)+1,mode ='constant')# add vectors which contains the value len(sentence) +1  at the end of the matrix
               elif str(sent_pad) =="high_num_sent":
                   concats = np.pad(concats,((0,300-len(concats)),(0,0)),constant_values = 10000,mode ='constant')# add vectors which contains the value len(sentence) +1  at the end of the
                   
                   

        to_tensor_concat = torch.from_numpy(concats).float() 
     
        return {"to_tensor_concat":to_tensor_concat,
                "label":label,
                 "arg1":arg1,
                 "arg2":arg2,
                 "abs_id":abs_id,
                 "sent":sent}#######################################################
def loader(path_train_file: str,

           path_test_file: str,
           path_word2vec: str,
           pad: int,
           batch_size: int,
           num_workers: int):
    """Returns train, dev and test set DrugProt data loaders."""

    pin_memory = torch.cuda.is_available()
    
    dataset_train_DrugProt = DrugProtDataset(root_dir=path_train_file,
                                         path_word2vec=path_word2vec,
                                         pad=pad)
    print("Length of training dataset", len(dataset_train_DrugProt))
    
    dataloader_train_DrugProt = DataLoader(dataset_train_DrugProt,
                                         batch_size=batch_size,
                                         pin_memory=pin_memory,
                                         num_workers = num_workers)

    dataset_test_DrugProt= DrugProtDataset(root_dir=path_test_file,
                                        path_word2vec=path_word2vec,
                                        pad=pad)
    print("Length of testing dataset",len(dataset_test_DrugProt))
    
    dataloader_test_DrugProt = DataLoader(dataset_test_DrugProt,
                                        batch_size=batch_size,
                                        pin_memory=pin_memory,
                                        num_workers=num_workers)

    return dataloader_train_DrugProt, dataloader_test_DrugProt





if __name__ == "__main__":
  

    model = ConvNet()
    model.to(device)
	# Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		

##########################################Load the data

    path_word2vec = "../data/vocab_size_2324849_min_-11.038_max_8.9537.p" ##vocabulary
    root = "data/DrugProt/"
    training = root+ "training.csv"
    testing = root+"testing.csv"

    train_loader, test_loader = loader(path_train_file=training,
		                           path_test_file=testing,
		                           path_word2vec=path_word2vec,
		                           pad=150,
		                           batch_size=60,
		                           num_workers=0)
        









#####################################################Tarin the model 



    total_step = len(train_loader)
    length_test = len(test_loader)
    loss_list = []
    acc_list = []
    num_epochs =sys.argv[5]
    output = []
    epoch_acc =[]
    epoch_acc_test = []

    for epoch in range(int(num_epochs)):
        print("Epoch:",epoch)
        model.to(device)
        outputs_list = []
        targets_list =[]
        total_loss_train = 0
        correct_train =0
        counter = 0
        for d in train_loader:

            counter +=1
            labels = d["label"]
            embeddings = d["to_tensor_concat"]
            labels = labels.to(device)            
            embeddings = embeddings.to(device)  
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            total_loss_train+=loss.item()
            
            
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct= (predicted == labels).sum().item()

            
            for out in predicted:
                outputs_list.append(out)
            for label in labels:
                targets_list.append(label)
      
                
                       
            if (counter+1 ) %111 == 0:

                torch.cuda.empty_cache()
                model.to(device)
                criterion.to(device)

                print("after 111 batch")
                total_loss_test = 0
                total_test = 0
                correct_test = 0
                for d_test in test_loader:
                        embeddings_test = d["to_tensor_concat"]
                        embeddings_test = embeddings_test.to(device)
                        labels_test = d["label"]
                        labels_test = labels_test.to(device)
                        outputs_test = model (embeddings_test)
                        loss_test = criterion(outputs_test, labels_test)
                        total_loss_test =loss_test.item()
                        _, predicted_test = torch.max(outputs_test.data, 1)
                        total_test = labels_test.size(0)
                        correct_test = (predicted_test == labels_test).sum().item()
                        #epoch_acc_test.append(correct_test/total_test)
                total_loss_test = total_loss_test /length_test
                total_loss_train = total_loss_train/total_step
		        
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}% on train set'.format(epoch + 1, num_epochs, counter + 1, total_step, loss.item(),(correct/ total) * 100))   
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}% on test set'.format(epoch + 1, num_epochs, counter + 1, total_step, loss_test.item(),(correct_test / total_test) * 100))   
                print("Target list",len(targets_list))
                
                epoch_acc.append([epoch+1,loss.item(),(correct/total)*100])
                epoch_acc_test.append([epoch+1,loss_test.item(),(correct_test/total_test)*100])
    
    with open("acc_graffiti"+p_v_pad+"_"+sent_pad+"_"+p_v_normalize,"w") as f:
            write = csv.writer(f)
            write.writerows(epoch_acc)
    with open("acc_test_graffiti"+p_v_pad+"_"+sent_pad+"_"+p_v_normalize,"w") as f_test:
            write = csv.writer(f_test)
            write.writerows(epoch_acc_test)
     
    ###save the model 
    torch.save(model.state_dict(),"CNN_best_modlel_graffiti"+str(num_epochs)+".ckpt")



   
   # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        outputs_list = []
        targets_list = []
        ids_list = []
        args1_list = []
        args2_list = []
        labels_list =[]
        ids = []
        sent_texts = []
        for d in test_loader:
          embeddings = d["to_tensor_concat"]
          embeddings = embeddings.to(device)
          outputs = model(embeddings)
          _, predicted = torch.max(outputs.data, 1)
            
          sent_texts.extend(d["sent"])
          outputs_list.extend(predicted)
          labels_list.extend(d["label"])
          ids.extend(d["abs_id"])
          args1_list.extend(d["arg1"])
          args2_list.extend(d["arg2"])
        outputs_list = torch.stack(outputs_list).cpu()
        labels_list = torch.stack(labels_list).cpu()
        
        print("Done predicting" )  
        print(outputs_list)
        print(ids)
        data_ = {'id':[id_.item() for id_ in ids],"sent":sent_texts,'y_pred':[pred.item() for pred in outputs_list],"arg1":args1_list,"arg2":args2_list,'real_values':[real.item() for real in labels_list]}
        df = pd.DataFrame(data=data_)
        df.to_csv("predictions_CNN.csv",index=False)
   
