from libs import *
def conv_to_number(df):
    df['label'] = df['label'].replace(["INDIRECT-DOWNREGULATOR"],0)
    df['label'] = df['label'].replace(["INDIRECT-UPREGULATOR"],1)
    df['label'] = df['label'].replace(["DIRECT-REGULATOR"],2)
    df['label'] = df['label'].replace(["ACTIVATOR"],3)
    df['label'] = df['label'].replace(["INHIBITOR"],4)
    df['label'] = df['label'].replace(["AGONIST"],5)
    df['label'] = df['label'].replace(["ANTAGONIST"],6)
    df['label'] = df['label'].replace(["AGONIST-ACTIVATOR"],7)
    df['label'] = df['label'].replace(["AGONIST-INHIBITOR"],8)
    df['label'] = df['label'].replace(["PRODUCT-OF"],9)
    df['label'] = df['label'].replace(["SUBSTRATE"],10)
    df['label'] = df['label'].replace(["SUBSTRATE_PRODUCT-OF"],11)
    df['label'] = df['label'].replace(["PART-OF"],12)
    df['label'] = df['label'].replace(["no_relation"],13)
    return df
    
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
##########################
def createList(r1, r2):
    return [item for item in range(r1, r2+1)]
#########################
#fun entity position vectors for sentences with no entities 
def no_ent_pos_vectors (lenghth):
    D_pos =  createList(-lenghth,-1) #if NO entity exist replace position vector by  -len(sentence) to -1
    #if p_v_rep == "z":
    #    D_pos= [0]*lenghth
    #elif p_v_rep == "dec_vec":
    #    D_pos =  createList(1,lenghth) #if NO entity exist replace position vector by 1 to len(sentence)
    #elif p_v_rep =="inc_vec":
    #    D_pos =  createList(-lenghth,-1) #if NO entity exist replace position vector by  -len(sentence) to -1
    #elif p_v_rep == "const":
    #    D_pos = [lenghth]*lenghth
    return D_pos
    
#########################################
def createList(r1, r2):
    return [item for item in range(r1, r2+1)]
        
##return the position vector 
def find_pos(sent,pos):
   # print('I am in find_pos function')
    
    if pos!='NONE':    
        D_pos =[]
        for i in range(len(word_tokenize(sent))):
            d_pos = i- int(pos)
            D_pos.append(d_pos)
        if len(D_pos) < 768:
            D_pos = D_pos + [len(D_pos)+1] * (768 - len(D_pos))
            
    else:   
            D_pos = no_ent_pos_vectors(768)
    D_pos = D_pos[0:768]
    #print(D_pos) 
    return D_pos
#############################
            
