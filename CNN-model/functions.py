import nltk#fun2
##return the position vector 
def createList (r1,r2):
    return [item for item in range(r1,r2+1)]
def find_pos(sent,pos,p_v_rep):
   # print("find_pos pos",pos)
     
    #if isinstance(pos,int):    
    D_pos =[]
    for i in range(len(nltk.word_tokenize(sent))):
         d_pos = i- int(float(pos))
         D_pos.append(d_pos)
   #else:   
    #       D_pos = no_ent_pos_vectors(p_v_rep,len(nltk.word_tokenize(sent)))
    #print("Position vector D_pos",D_pos)
    return D_pos
            
#fun entity position vectors for sentences with no entities 
def no_ent_pos_vectors (p_v_rep,lenghth):
    if p_v_rep == "z":
        D_pos= [0]*lenghth
    elif p_v_rep == "dec_vec":
        D_pos =  createList(1,lenghth) #if NO entity exist replace position vector by 1 to len(sentence)
    elif p_v_rep =="inc_vec":
        D_pos =  createList(-lenghth,-1) #if NO entity exist replace position vector by  -len(sentence) to -1
    elif p_v_rep == "const":
        D_pos = [lenghth]*lenghth
    return D_pos
    
