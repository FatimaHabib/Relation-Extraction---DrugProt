#####train with cross validation
from libs import *
from functions import *
PRE_TRAINED_MODEL_NAME = 'allenai/scibert_scivocab_uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME,truncation=True)
from model_features import *
from data_loader_features_CV import *

################# Intialize parameters ##################
MAX_LEN = 200
k_folds = 5
BATCH_SIZE = int(sys.argv[1])
EPOCHS = int(sys.argv[2])
############################################"
torch.cuda.empty_cache()
# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")#
    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
####################Load data 

data = pd.read_csv("../code_Re_shared_task/data/DrugProt/training.csv", delimiter='|')
test_data = pd.read_csv("../code_Re_shared_task/data/DrugProt/testing.csv", delimiter='|')


##convert labels into numerical values 

data.columns =["id","sentence","entity1","entity2","pos1","pos2","arg1","arg2","ent1_type1","ent2_type","label"]
data = conv_to_number(data)
data = data.iloc[:300,:]
 
#########################Model, optimizer, Loss function
model = RelationClassifier()

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

loss_fn = nn.CrossEntropyLoss().to(device)

# =============================================
#             Functions: train_epoch, eval_model,get_predictions
# ==============================================

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples,

):
  

  model.to(device) 
  model = model.train()
  losses = []
  correct_predictions = 0
  steps = 0
  for step,d in enumerate(data_loader):
    steps +=1
    # Progress update every 40 batches.
    if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data_loader), elapsed))
    d1 = d["d1"]
    d2 = d["d2"]
    pos1 = d["pos1"]
    pos2 =d["pos2"]
    #print("Type",type(d1),len(d1))
    #print("-"*100)
    
    d1 = d1.to(device)
    d2 = d2.to(device)
    #print("input_ids",len(d["input_ids"]))
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    labels = d["labels"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      d1=d1,
      d2 =d2,
      pos1 = pos1,
      pos2 = pos2 
    )
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, labels)
    correct_predictions += torch.sum(preds == labels)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    

    
  return correct_predictions.double() / n_examples, np.mean(losses)



def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      
      d1 = d["d1"]
      d2 = d["d2"]
      #d1 = torch.FloatTensor(d1)
      #d2 = torch.FloatTensor(d2)
      d1 = d1.to(device)
      d2 = d2.to(device)
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      labels = d["labels"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,d1 = d1,d2=d2
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, labels)
      correct_predictions += torch.sum(preds == labels)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)
  
  

def get_predictions(model, data_loader):
  model = model.eval()
  sent_texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for d in data_loader:
      texts = d["sentence_text"]
      
      d1 = d["d1"]
      d2 = d["d2"] 
      #d1 = torch.FloatTensor(d1)
      #d2 = torch.FloatTensor(d2)
      d1 = d1.to(device)
      d2=d2.to(device)
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      labels = d["labels"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,d1=d1,d2=d2
      )
      _, preds = torch.max(outputs, dim=1)
      sent_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(labels)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return sent_texts, predictions, prediction_probs, real_values
##################################################################""

history = collections.defaultdict(list)
best_accuracy = 0

kfold = KFold(n_splits=k_folds, shuffle=True)
train_acc_ =[]
test_acc_=[]
train_loss_=[]
test_loss_=[]
fold_acc =[]
fold_loss=[]
# initialize the early_stopping object
# =============================================
#             Cross validation Loop
# ==============================================


for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    train_data_loader = create_data_loader(data, tokenizer, MAX_LEN, BATCH_SIZE,train_subsampler)
    test_data_loader = create_data_loader(data, tokenizer, MAX_LEN, BATCH_SIZE,test_subsampler)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)    
    train_size = (len(data)/5) * 4
    test_size  = (len(data)/5)
    # Report the number of sentences.
    print('Number of training sentences:\n',train_size)
    print('Number of testing sentences:\n',test_size)

# =============================================
#             Training Loop
# ==============================================
    for epoch in range(EPOCHS):
         
            # ========================================
            #               Training
            # ========================================
            
            # Perform one full pass over the training set.
          t_int = time.time()
          print("")
          print('======== Epoch {:} / {:} ========'.format(epoch + 1, EPOCHS))
          print('Training...')
          t0 = time.time()
          
          
          train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            train_size
          )
          ###timz*e
          print(f'Train loss {train_loss} accuracy {train_acc}')
          elapsed = format_time(time.time() - t0)
          print(f'Training took {elapsed}')
          print("")
          print("Running Testing...")
          
          
          
          
          t0 = time.time()
          test_acc, test_loss = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            test_size
            )
            
            
            
          print(f'Test   loss {test_loss} accuracy {test_acc}')
          elapsed = format_time(time.time() - t0)
          print(f'Testing took {elapsed}')
          print()
          train_acc_.append(train_acc)
          train_loss_.append(train_loss.item())
          test_acc_.append(test_acc)
          test_loss_.append(test_loss.item())

          if test_acc > best_accuracy:
            best_accuracy = test_acc
            es = 0

            torch.save(model.state_dict(), 'best_model_state_'+str(BATCH_SIZE)+'_'+str(EPOCHS)+'.bin')
            best_accuracy = test_acc

          else:
            es += 1
            print("Counter {} of 5".format(es))

            if es > 7:
                print("Early stopping with best_acc: ", best_accuracy, "and test_acc for this epoch: ", test_acc, "...")
            break
    print('Accuracy for fold %d: %d %%', (fold,test_acc))
    print('Loss for fold %d: %d %%', (fold,test_loss))
    
###################################################################
    fold_acc.append(test_acc)
    fold_loss.append(test_loss)
    
# =============================================
#             Saving Results
# ==============================================
    history= pd.DataFrame()
    history["train_acc"] =train_acc_
    history["train_loss"] = train_loss_
    history["test_acc"]=test_acc_
    history["test_loss"] = test_loss_
    history.to_csv('progress_CV'+str(BATCH_SIZE)+'_'+str(EPOCHS)+'.csv')

    folds_res = pd.DataFrame()
    folds_res["acc"] = fold_acc
    folds_res["loss"] = fold_loss
    folds_res.to_csv("folds_res_CV_"+str(BATCH_SIZE)+"_"+str(EPOCHS)+".csv")
#######################
print("")
elapsed_total = format_time(time.time() - t_int)
print(f"Training complete and it took {elapsed_total} !")    
#########save model progress

test_data_loader = create_data_loader(test_data, tokenizer, MAX_LEN, BATCH_SIZE)
y_sent_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)


results_Bert_sci = pd.DataFrame()
results_Bert_sci["txt"] = y_sent_texts
results_Bert_sci["y_pred"] = y_pred
results_Bert_sci["y_pred_probs"] = y_pred_probs
results_Bert_sci["real_values"]  = y_test
results_Bert_sci.to_csv("predictions_Bert_sci_CV"+str(BATCH_SIZE)+"_"+str(EPOCHS)+".csv")
