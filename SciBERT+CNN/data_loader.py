from libs import *
from functions import *

MAX_LEN = 200
BATCH_SIZE = 32
EPOCHS = 3

#id|sentence|entity1|entity2|pos1|pos2|arg1|arg2|ent1_type1|ent2_type|label
########### PyTorch dataset.
class ChemProtDataset(Dataset):
  def __init__(self, sentences,arg1,arg2,e1,e2,pos1,pos2,e1_type1,e2_type, labels, tokenizer, max_len):
    self.sentence = sentences
    self.arg1 = arg1
    self.arg2 = arg2
    self.pos1 = pos1##entity1 index it could be NONE if no entity exist
    self.pos2 = pos2##entity2index it could be NONE if no entity exist
    self.e1 = e1
    self.e2 = e2
    self.e1_type = e1_type1
    self.e2_type = e2_type
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len
  def __len__(self):
    return len(self.sentence)
  def __getitem__(self, item):
    sentence = str(self.sentence[item])
    label = self.labels[item]
    pos1 = self.pos1[item]
    pos2 = self.pos2[item]
    
    encoding = self.tokenizer.encode_plus(
      sentence,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
   # print("pos1",pos1)
    d1 = find_pos(sentence,pos1)
    d2 = find_pos(sentence,pos2)
    d1 = torch.Tensor(d1)
    d2 = torch.Tensor(d2)
    #print("Iam in get item",type(d1))
    return {
      'sentence_text': sentence,
      'd1':d1,
      'd2':d2,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'labels': torch.tensor(label, dtype=torch.long),
      'pos1':pos1,##position of first entity 
      'pos2':pos2##position of second entity
    }
    
    
def create_data_loader(df, tokenizer, max_len, BATCH_SIZE,sampler):
  ds = ChemProtDataset(
    sentences=df.sentence.to_numpy(),
    arg1 = df.arg1.to_numpy(),
    arg2 = df.arg2.to_numpy(),
    e1 = df.entity1.to_numpy(),
    e2 = df.entity2.to_numpy(),
    pos1 = df.pos1.to_numpy(),
    pos2 = df.pos2.to_numpy(),
    e1_type1 = df.ent1_type1.to_numpy(),
    e2_type = df.ent2_type.to_numpy(),
    labels=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    num_workers=0,
    sampler=sampler
  )    

    
