from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import warnings
from nltk.stem import WordNetLemmatizer


from pymongo import MongoClient

import gensim 
from gensim.models import Word2Vec 
from pymongo import MongoClient

from nltk.tag.stanford import StanfordNERTagger
from nltk.parse.corenlp import CoreNLPParser
import csv

# coreNLP initialization
#st1 = CoreNLPParser(tagtype='ner')
st1 = CoreNLPParser(url="http://localhost:9000", tagtype='ner')
parser = CoreNLPParser(url="http://localhost:9000", tagtype='pos')

warnings.filterwarnings(action = 'ignore')

mongoClient = MongoClient([username], [port], username=[username],
                            password=[password], connectTimeoutMS=100000)

#These collection did contain new article records
db=mongoClient[db_name]
coll=db[coll_name]



'''For time being consider only those records which come after count of 1200 because at counts below that training has been done. Aggregating the title and description sentences in 2 different list'''

for cur1 in coll.find().skip(1200).limit(20):
#cur1 =coll.find_one({'link': 'http://www.fiercepharma.com/financials/valeant-pharmaceuticals-reports-second-quarter-2015-financial-results'})
    print(cur1['link'])
    print('^'*60)
    if 'desc' in cur1 and len(cur1['desc'])>5:
        desc_data = cur1['desc']

    if 'title_nw' in cur1:
        title = cur1['title_nw']
    elif 'title' in cur1:
        title = cur1['title']

    title_sent=sent_tokenize(title)

    title_tagger=[]
    for title in title_sent:
        #title = title.replace('"','')
        [title_tagger.append([n.strip()]) for n in re.split(';',title) if(n.strip())]
        #title_tagger.append(title)

    desc_sent=sent_tokenize(desc_data)


    sentence_tagger=[]

    for desc in desc_sent:
        #desc = desc.replace('"','')
        if(re.search('>', desc)):
            sp = re.search('>', desc).span()[1]
            desc = desc[sp:].strip()

        #desc= re.sub('\s\s+',' ',desc)

        if(re.search('\(*[$]\s*[A-Za-z]+\)*',desc)):
            desc =re.sub('\(*[$]\s*[A-Za-z]+\)*','',desc)

        desc = [m.strip() for m in re.split(';',desc)]


        for a in desc:
            a1 = [b.strip() for b in re.split(':\s+',a)]

            for b in a1:
                b = re.sub('^\([a-z]\)','',b).strip()
                b = re.sub('\s+',' ',b)
                sentence_tagger.append([b])

    #     if(re.search('^\([a-z]\)', desc)):
    #         desc = re.sub('^\([a-z]\)','',desc).strip()

    #     [sentence_tagger.append([n]) for n in desc]

    print(title_tagger)
    print('~'*60)
    for j in sentence_tagger:
        print(j)
        print('.'*50)
    print('\n','-'*70)



'''getting relevant money from the deals pages'''

#deals bag of words and its processing
deals_bow_raw = """Collaborations
Associations
joint effort
combination
partnership
alliance 
club
tie
union
aid
sharing
bond
Synergy
united
unity
acquisition
coalition
acquir
invest
investment
conquer
Joint venture 
agreement
mutual
alligned
joint
added
approach
along 
combined
deals
agree
agreed
contract
negotiations
agreeing
sign
signed
negotiating
negotiated
negotiate
buy
buyout
purchase
purchased
merger
agreements
signing
sign
pact
proposal
offer
outlicense
outlicensed
sell
deal
duel
snag
snagging"""


deals_bow = [a.lower().strip() for a in deals_bow_raw.split('\n')]
deals_bow = set(deals_bow)
deals_bow_list = list(deals_bow)



'''Preparing the test data frame'''
df_list =[]
for cur in coll.find().skip(1050).limit(20):
    if 'title_nw' in cur:
        title = cur['title_nw']
    elif 'title' in cur:
        title = cur['title']
    
    if 'desc' in cur and len(cur['desc'])>5:
        desc_data = cur['desc']
        
    title_sent=sent_tokenize(title)

    title_tagger=[]
#     for title in title_sent:
#         #title = title.replace('"','')
#         title_tagger.append([title])
    
    for title in title_sent:
        #title = title.replace('"','')
        [title_tagger.append([n.strip()]) for n in re.split(';',title) if(n.strip())]
        #title_tagger.append(title)


    desc_sent=sent_tokenize(desc_data)


    sentence_tagger=[]

#     for desc in desc_sent:
#         #desc = desc.replace('"','')
#         if(re.search('>|:', desc)):
#             sp = re.search('>|:', desc).span()[1]
#             desc = desc[sp:]

#         if(re.search('\(*[$]\s*[A-Za-z]+\)*',desc)):
#             desc =re.sub('\(*[$]\s*[A-Za-z]+\)*','',desc)
            
#         if(not re.search('Related Articles:',desc)):
#             desc = [m.strip() for m in desc.split(';')]
#             [sentence_tagger.append([n]) for n in desc]

    
    for desc in desc_sent:
        #desc = desc.replace('"','')
        if(re.search('>', desc)):
            sp = re.search('>', desc).span()[1]
            desc = desc[sp:].strip()

        #desc= re.sub('\s\s+',' ',desc)

        if(re.search('\(*[$]\s*[A-Za-z]+\)*',desc) and not re.search('\(*[$]\s*\d+\)*',desc)):
            desc =re.sub('\(*[$]\s*[A-Za-z]+\)*','',desc)

        desc = [m.strip() for m in desc.split(';')]


        for a in desc:
            a1 = [b.strip() for b in re.split(':\s+',a)]

            for b in a1:
                #b = re.sub('^\(*[a-z0-9].*\)','',b).strip()
                b = re.sub('^\(*[a-z0-9]+\.*\)','',b)
                b = re.sub('\s+',' ',b).strip()
                if(b):
                    sentence_tagger.append([b])


    #for rec in sentence_tagger:
        #print(rec)
        
    title_ner=st1.tag_sents(title_tagger)
    title_pos=parser.tag_sents(title_tagger)
    s=set(stopwords.words('english'))
    
    
    #getting title lines which contain money amount
    mon_ls=[]
    val = ''
    for i in range(len(title_ner)):
        flag=0
        for j in title_ner[i]:
            if(j[1]=='MONEY'):
                flag=1
                break
        if(flag==1):
            mon_ls.append(i)
    
    #getting title lines having money amounts which are deal amounts
    lemmatizer = WordNetLemmatizer()
    index_ls=[]
    for i in mon_ls:
        flag=0
        for j in range(len(title_ner[i])):
            if(title_ner[i][j][1]=='O' and title_ner[i][j][0] not in s):
                strn=''
                if(title_pos[i][j][1][0:2]=='VB'):
                    strn='v'
                elif(title_pos[i][j][1]=='JJ'):
                    strn='a'
                elif(title_pos[i][j][1][0:2]=='NN'):
                    strn='n'

                #print(title_ner[i][j])
                if(strn=='v' or strn=='a' or strn=='n'):
                    core_word=lemmatizer.lemmatize(title_ner[i][j][0],pos=strn)
                else:
                    core_word=lemmatizer.lemmatize(title_ner[i][j][0])

                if(core_word in deals_bow_list):
                    flag=1
                    break
        #print('*'*60)
        if(flag==1):
            index_ls.append((i,title_ner[i][j][0]))
    
    #print(index_ls)
    
    for k in index_ls:
        d={}
        d['link']=cur['link']
        d['SENTENCES'] = title_tagger[k[0]][0]
        df_list.append(d)
    
    if(not index_ls):
        res=st1.tag_sents(sentence_tagger)
        res_=parser.tag_sents(sentence_tagger)
        s=set(stopwords.words('english'))
        
        #getting para lines which have money
        mon_ls1=[]
        for i in range(len(res)):
            flag=0
            for j in res[i]:
                if(j[1]=='MONEY'):
                    flag=1
                    break
            if(flag==1):
                mon_ls1.append(i)
        
        #getting para lines which have money and are deal amounts
        lemmatizer = WordNetLemmatizer()
        index_ls=[]
        for i in mon_ls1:
            flag=0
            for j in range(len(res[i])):
                if(res[i][j][1]=='O' and res[i][j][0] not in s):
                    strn=''
                    if(res_[i][j][1][0:2]=='VB'):
                        strn='v'
                    elif(res_[i][j][1]=='JJ'):
                        strn='a'
                    elif(res_[i][j][1][0:2]=='NN'):
                        strn='n'

                    #print(res[i][j])
                    if(strn=='v' or strn=='a' or strn=='n'):
                        core_word=lemmatizer.lemmatize(res[i][j][0],pos=strn)
                    else:
                        core_word=lemmatizer.lemmatize(res[i][j][0])

                    if(core_word in deals_bow_list):
                        flag=1
                        break
            #print('*'*60)
            if(flag==1):
                index_ls.append((i,res[i][j][0]))
                
        for k in index_ls:
            d={}
            d['link']=cur['link']
            d['SENTENCES'] = sentence_tagger[k[0]][0]
            df_list.append(d)


import pandas as pd
df_table = pd.DataFrame(df_list)
df_table.count()



'''Making predictions for database based on saved model'''
import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer


# Load a trained model and vocabulary that you have fine-tuned
model = BertForSequenceClassification.from_pretrained('model_save1')
tokenizer = BertTokenizer.from_pretrained('model_save1')

# Copy the model to the GPU.
model.to(device)


import pandas as pd

# Filter the pandas dataframe.
print(df_table.count())

df_table1 = df_table[df_table['SENTENCES'].str.len() <256]

sentences = df_table1.SENTENCES.values
print(len(sentences))


df_table1.head(28)


print(list(df_table1['SENTENCES'])[20])


from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



MAX_LEN =256
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []

# For every sentence...
for sent in sentences:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                   )
    
    input_ids.append(encoded_sent)

print('Maximum_sentence_length:-', max([len(sen) for sen in input_ids]) )

# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 
                          dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask) 

# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)

# Set the batch size.  
batch_size = 30  

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)



# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)


    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask = batch

    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, 
                        attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logitsto CPU
    logits = logits.detach().cpu().numpy()

    # Store predictions
    predictions.append(logits)

print('    DONE.')



print(predictions)


import numpy as np
predicted_labels =[]
for i in range(len(predictions)):
    pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
    predicted_labels+= list(pred_labels_i)
    print(pred_labels_i)
    print('~'*60)
    
    
    
print(predicted_labels)


'''Fetching organization names and deal amount for the sentences that have been predicted to have deals'''
output =[]
for i in range(len(sentences) ):
    d ={}
    org_set,money_set =set(),set()
    if(predicted_labels[i] == 1):
        tup_list = st1.tag_sents([[sentences[i]]])
#         print(tup_list)
        val,temp ='',''
        for j in tup_list[0]:
            if(j[1] == 'O' and temp):
                if(temp == 'MONEY' and val and not re.search('[$] [A-Za-z]+',val) ):
                    money_set.add(val.lower())
                elif((temp == 'ORGANIZATION' or temp == 'PERSON') and val and not re.search('[$] [A-Za-z]+',val)):
                    org_set.add(val)
                
                val =''
            
            if(j[1]!= 'O'):
                val+= j[0]+ ' '
                temp = j[1]
                
        if(temp):
            if(temp == 'MONEY' and val and not re.search('[$]\s*[A-Za-z]+',val) ):
                money_set.add(val.lower())
            elif((temp == 'ORGANIZATION' or temp == 'PERSON') and val and not re.search('[$]\s*[A-Za-z]+',val)):
                org_set.add(val)
#         print(sentences[i])
#         print(org_list)
#         print(money_list)
        
    d['sentence'] = sentences[i]
    d['status'] = predicted_labels[i]
    d['organizations/persons'] = list(org_set)
    d['money'] = list(money_set)
    output.append(d)

#print(output)    
df_output = pd.DataFrame(output)

print(df_output)

