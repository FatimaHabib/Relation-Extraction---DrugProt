import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
import csv
import re
import sys
p_v_rep = sys.argv[1] #z_vec inc dec const 
##links to the data files \n",
#root = "../drugprot-gs-training-development/test-background/"
path_abstracts = "test_background_abstracts.tsv"
path_relations ="relations.tsv"
path_entities ="test_background_entities.tsv"
### Read files as DF\n
#abstracts\n",
colnames = ["id","title","abstract"]
abstracts = pd.read_csv(path_abstracts,sep ="\t",header = None,names = colnames)

##entities\n",
colname_ent = ["id","T_n","entity_type","start","end","str_entity"]
entities = pd.read_csv(path_entities,sep ="\t",header = None,names = colname_ent)
###relations file 
colname_relation = ["id","rel","T1_n","T2_N"]
relations = pd.read_csv(path_relations,sep ="\t",header = None,names = colname_relation)
abstracts["ti_abst"] = abstracts["title"] + "   "+ abstracts["abstract"]
#fun1






##This function return the sentences of an abstract with its offsets
#it takes as input the abstract id and abstract text
def sentences_fun (id_,text_):
        sent_para =[]
        #https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing\n",
        #tokenize sentence
        sentences =re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text_)
    #sent_tokenize(text_) #tokenize the abstract \n",
        start_off = 0
        end =len(sentences[0])-1
        sent_para.append([id_,sentences[0],start_off,end])
        start_off= start_off + end#+2
        for i in range(1,len(sentences)):
            #padded_sent = sentences[i].rjust(1)
            sent_para.append([id_,sentences[i],start_off,start_off+len(sentences[i])]) ##-1append to each sentence its actual (start-end) character offset \n",
            start_off = start_off +len(sentences[i])+1
        return list(sent_para)
        
        
        
 ###########
def add_pos_opt(sentences,abst_id):
        group_entities =entities.groupby("id")
        sentences_entities =  []
        counter = 0
        for key,vlaues in group_entities:
            counter = counter+1
            if key ==abst_id:
                #print("yes")
                group = group_entities.get_group(key)
                for sent in sentences:
                    dict_ = {}
                    for index,row in group.iterrows():
                        if row["start"] in range(sent[2],sent[3]):
                            dict_[row["T_n"]] = [row["T_n"],row["entity_type"],row["str_entity"],row["start"],row["end"]]
                    sent.append(dict_)
                    sentences_entities.append(sent)
                print("Search times =",counter)
                continue    
        return sentences_entities
   
 
 
 
#fun2
##return the position vector 
#def find_pos(sent,pos):

#    if pos:    
#        D_pos =[]
#        for i in range(len(word_tokenize(sent))):
#            d_pos = i- pos
#            D_pos.append(d_pos)
#    else:   
#            D_pos = no_ent_pos_vectors(p_v_rep,len(word_tokenize(sent)))
#    return D_pos
            
#fun entity position vectors for sentences with no entities 
#def no_ent_pos_vectors (p,lenghth):
#    if p_v_rep == "z":
#        D_pos= [0]*lenghth
#    elif p_v_rep == "inc":
#        D_pos =  createList(1,lenghth) #if NO entity exist replace position vector by 1 to len(sentence)
#    elif p_v_rep =="dec":
#        D_pos =  createList(-lenghth,-1) #if NO entity exist replace position vector by  -len(sentence) to -1
#    elif p_v_rep == "const":
#        D_pos = [lenghth]*lenghth
#    return D_pos
        
##############################################
#fun3          
##find the indices of the entities \n",
def entity_index(sent,sent_start,sent_end,en1_start,en1_end):
       
        
        sent_start_ = 0
        sent_end = sent_end-sent_start
        ###entitys offsets \n",
        en1_start =  en1_start - sent_start#+1\n",
        en1_end = en1_end - sent_start#+1
        print("(",sent_start_,sent_end,en1_start,en1_end,")")
        if en1_start in range(sent_start_,sent_end) and en1_end in range(sent_start_,sent_end+2):
                #print(en1_start)
                #print(en1_end)
                #print(len(sent))
                sent = list(sent)
                ### replace entity by \n",
                sent = sent[:en1_start]+list("(ENTITY1)") + sent[en1_end:] 
                #sent = sent[:en2_start] + list(\"ENTITY2\") + sent[en2_end:]\n",
                sent = "".join(sent)
                ###Find the index of the element\n",
                #print("The sentence after adding Entity 1",sent)
                sent = word_tokenize(sent)
                pos1 = sent.index("ENTITY1")
                #pos2 = sent.index(\"ENTITY2\")\n",
                #print(sent)
                #print(pos1)
                #print(pos2)
                return pos1#,pos2
        else:
            return None
            
def createList(r1, r2):
    return [item for item in range(r1, r2+1)]
#fun5
##generate the negative instances
def neg_instances_gen(sent):
            #D_pos =  no_ent_pos_vectors(p_v_rep,len(word_tokenize(sent[1]))) #position vector where no entity exist 
            #D= [0]*len(word_tokenize(sent[1]))
            #there is no annotated entities in the sentence
            if len (sent[4].keys()) == 0 :  
                #d= [0]*len(sent[1])
                #line = [sent[0],sent[1],"NONE","NONE",D_pos,D_pos,"NONE","NONE","NONE","NONE","no_relation"]
                 line = [sent[0],sent[1],"NONE","NONE","NONE","NONE","NONE","NONE","NONE","NONE","no_relation"]
            #there is one annotated entity
            if len(sent[4].keys())== 1:
                for key in sent[4].keys():
                    if sent[4][key][1] == 'CHEMICAL':
                        entity1 = sent[4][key]
                        entity2 ="NONE"
                        #print("neg_instances_gen sent[4][key] gen does not exist",sent[4][key])
                        pos1 = entity_index(sent[1],sent[2],sent[3],sent[4][key][3],sent[4][key][4])
                        #d1 = find_pos(sent[1],pos1)
                        #line = [[sent[0],sent[1],entity1[2],entity2,d1,D_pos,"Arg1:"+entity1[0],"NONE",entity1[1],entity2,"no_relation"]]
                        line = [[sent[0],sent[1],entity1[2],entity2,pos1,"NONE","Arg1:"+entity1[0],"NONE",entity1[1],entity2,"no_relation"]]
                        return line
                    else :
                        lines =[]
                        entity2 = sent[4][key]
                        entity1 = "NONE"
                            ##abstract_id ,sentence text, entity1, entity2, postion vector1, position vector2,arg1,arg2,ent1_type,ent2_type,label\n",
                        #print("neg_instances_gen offset ", sent[4][key][3])
                        #print("neg_instances_gen entity ",entity2)
                        #print("neg_instances_gen sentence",sent)
                        pos2 = entity_index(sent[1],sent[2],sent[3],sent[4][key][3],sent[4][key][4])
                        #d2 = find_pos(sent[1],pos2)
                        #line = [[sent[0],sent[1],entity1,entity2[2],D_pos,d2,"NONE","Arg2:"+entity2[0],"NONE",entity2[1],"no_relation"]]
                        line = [[sent[0],sent[1],entity1,entity2[2],"NONE",pos2,"NONE","Arg2:"+entity2[0],"NONE",entity2[1],"no_relation"]]
                        return line
            #there are more than one annotated entity
            else:
                chem = []
                gens = []
                lines = []
                for key in sent[4].keys():
                    if sent[4][key][1] == 'CHEMICAL':
                        chem.append(sent[4][key])
                    else:
                        gens.append(sent[4][key])
                if len(chem) >0:
                    for entity in chem:
                        #print("Entity ",entity)
                        chem_index = entity_index(sent[1],sent[2],sent[3],entity[3],entity[4])
                        #d_chem = find_pos(sent[1],chem_index)
                        #lines.append([sent[0],sent[1],entity[2],"NONE",d_chem,D_pos,"Arg1:"+entity[0],"NONE",entity[1],"NONE","no_relation"])
                        lines.append([sent[0],sent[1],entity[2],"NONE",chem_index,"NONE","Arg1:"+entity[0],"NONE",entity[1],"NONE","no_relation"])
                        
                    if len(gens)>0:
                        for line in lines: 
                            for gen in gens: 
                                gen_index = entity_index(sent[1],sent[2],sent[3],gen[3],gen[4])
                                #print("print den index:",gen_index)
                                #print("The gen is ",gen)
                                #print("sentence start at ",sent[2])
                                #print("sentence end at ",sent[3])
    
                                #d_gen = find_pos(sent[1],gen_index)
                                line[3] = gen[2]
                                #line[5] = d_gen
                                line[5] = gen_index
                                line[7] = "Arg2:"+gen[0]
                                line[9] = gen[1]
                                 
                if len(chem) ==0:
                        for gen in gens:
                            gen_index = entity_index(sent[1],sent[2],sent[3],gen[3],gen[4])                    
                            print("print den index:",gen_index)
    
                            #d_gen = find_pos(sent[1],gen_index)
                            gen_name = gen[2]
                            gen_arg = "Arg2:"+gen[0]
                            gen_typ = gen[1]
                         
                            #lines.append([sent[0],sent[1],"NONE",gen_name,D_pos,d_gen,"NONE",gen_arg,"NONE",gen_typ,"no_relation"])   
                            lines.append([sent[0],sent[1],"NONE",gen_name,"NONE",gen_index,"NONE",gen_arg,"NONE",gen_typ,"no_relation"])         
                return lines
           
#fun5          
#add the position vectors 
def entity_pos_opt(sent_rel,file):
        for sent in sent_rel:
        #print(sent)\n",
        #check if sentence has relation\n",
        ## distence vector \n",
            d1 = []
            d2 = []
            if sent[5]:##the dicyionary that contains the relations\n",
                for key in sent[5].keys():
                   #print(\"Entity pairs:\",sent[5][key])\n",
                    entity1 = sent[5][key][0][2] #entity name\n",
                    entity2 = sent[5][key][1][2] #entity name\n",
                    #print("entity_pos_opt This the key",sent[5][key])
                    relation =sent[5][key][4] #relation label \n",
                    arg1 = "Arg1:"+sent[5][key][0][0] #term identifier\n",
                    arg2 = "Arg2:"+sent[5][key][1][0] #\n",
                    en1_type= sent[5][key][0][1]
                    en2_type= sent[5][key][1][1]
    
                    sent_ =sent[1]
                   ##if the entity consist of two words we cosider the first word tocompute the distances\n"               ##  if len(word_tokenize(entity1)) > 1:\n",
                  #\"\"      entity1 = word_tokenize(entity1)[0]\n",
                ### if len(word_tokenize(entity2)) > 1:\n",
         ##      entity2 = word_tokenize(entity2)[0]\n",
        
                    tokenized_sent = word_tokenize(sent_)
                           
                    pos1 = None
                    pos2= None
                    pos1 = entity_index(sent[1],sent[2],sent[3],sent[5][key][0][3],sent[5][key][0][4])##pass the offsets\n",
                    pos2 = entity_index(sent[1],sent[2],sent[3],sent[5][key][1][3],sent[5][key][0][4])
                    if pos1 and pos2:
                        #print("pos1------------------------------",pos1)
                        #print("pos2------------------------------", pos2)
                        #d1= find_pos(sent[1],pos1)
                        #d2 = find_pos(sent[1],pos2)
                        #print(d1 ,"",d2)
                        #print(len(d2)) 
                       ###save positive instance\n",
                        ##abstract_id ,sentence text, entity1, entity2, postion vector1, position vector2,arg1,arg2,ent1_type,ent2_type,label\n",
                        #line = [sent[0],sent[1],entity1,entity2,d1,d2,arg1,arg2,en1_type,en2_type,relation]
                        line = [sent[0],sent[1],entity1,entity2,pos1,pos2,arg1,arg2,en1_type,en2_type,relation]
                        

                        with open(file,"a") as f:  
                            write = csv.writer(f,delimiter='|')
                            write.writerow(line)
                            
                    #####position vectorsn represntation
                    else:
                        #D_pos =  no_ent_pos_vectors(p_v_rep,len(word_tokenize(sent[1])))
                        #d1 =D_pos #[0]*len(sent[1])
                        #d2 =D_pos #[0]*len(sent[1])
                        #print("one of the entities is not exist")
                        #print("pos1 is :", pos1)
                        #print("pos2 is :", pos2)
                        #D_pos  D_pos
                        line = [sent[0],sent[1],"NONE","NONE","NONE","NONE",arg1,arg2,"NONE","NONE","no_relation"]
                        with open(file,"a") as f:   
                            write = csv.writer(f,delimiter='|')
                            write.writerow(line)
                    ###save sentences instence into file \n",
            else:
                    lines = neg_instances_gen(sent)
                    #print(lines)
                    with open(file,"a") as f:  
                        write = csv.writer(f,delimiter='|')
                        for line in lines:
                            write.writerow(line)
                            
def find_rel_opt(sentences_list,abst_id): 
        sentences_list_ = sentences_list[:]
        sentences_ent_rel =  []
        group_relations = relations.groupby("id")
        for key,vlaues in group_relations:
            if key ==abst_id:
                ##search for the abstract id in the relation file\n",
                group = group_relations.get_group(key)
                for sent in sentences_list_:

                        rel= 1
                        dect_relations = {}
                        for index,row in group.iterrows():
   
                                if row["T2_N"][5:] in list(sent[4].keys()) and row['T1_n'][5:] in list(sent[4].keys()):
                                    #print(row["T1_n"]," ",row["T2_N"]," ",row["rel"])


                                    dect_relations["rel"+str(rel)] = [sent[4][row['T1_n'][5:]],sent[4][row['T2_N'][5:]],row['T1_n'],row['T2_N'],row["rel"]]
                                    rel =rel + 1
    
                                #print(sent[4].keys()) :annotated entities \n",
    
                                 ##check if one of the entity pairs exist in other sentence in the abstract   \n",
                                if row["T2_N"][5:] in list(sent[4].keys()) and row['T1_n'][5:] not in list(sent[4].keys()):
                                    print( row["T2_N"][5:],"exists in this sentence but",row['T1_n'][5:],"is not")
    
                                if row["T2_N"][5:] not in list(sent[4].keys()) and row['T1_n'][5:]  in list(sent[4].keys()):
                                    print( row["T1_n"][5:],"exists in this sentence but",row['T2_N'][5:],"is not")
    
    
                        sent.append(dect_relations)   
                        sentences_ent_rel.append(sent)
                return sentences_ent_rel
                        
                 
            ##if no abst id was found that means no relation is annotated between entities in the abstract \n",
        for sent in sentences_list_:
                dect_relations ={}
                sent.append(dect_relations)   
    
                sentences_ent_rel.append(sent)
               
        return sentences_ent_rel
     
file ="test_set_29_9"+p_v_rep +".csv"                           
for idex,abstract in abstracts.iterrows():
       print(idex)
       print(abstract["id"],"-------------------------------------------------------")
       sentences = sentences_fun(abstract["id"],abstract["ti_abst"])
       sent = add_pos_opt(sentences,abstract["id"])
       sent_rel = find_rel_opt(sent,abstract["id"])
       entity_pos_opt(sent_rel,file) 
       df_ent = entities[entities["id"] == abstract["id"]]
       entities = entities.drop(df_ent.index, axis=0)
       df_rel = relations[relations["id"] == abstract["id"]]
       relations = relations.drop(df_rel.index,axis = 0)

       print("--------------------+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-------")

  
