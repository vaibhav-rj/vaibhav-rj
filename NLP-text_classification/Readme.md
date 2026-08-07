# Deal Sentence(Amount and Organization) classification pipeline:-

## **Purpose:-** ###
The main purpose of the deal sentence(amt and org) classification is to take news article as the input and output the deals sentence (which have preferrably less than 256 characters and do contain 2 or more organizations/persons along with a deal keyword, deal amount) and henceforth also output the list of organizations, deal amount present in the output deal sentence.

## **Package installation and requirements:-**
Here are the list of python packages that need to be installed:-
(a) common nltk packages basically ***sent_tokenize, word_tokenize, stopwords, WordNetLemmatizer.***

(b) Specific usage nltk packages like              ***StanfordNERTagger, CoreNLPParser for 'POS' and 'NER' tagging.***
    
(c) Packages for machine learning like ***keras, sklearn, torch, transformers.***

(d) Other packages like *pymongo, warnings, re, csv* that already come installed while installation of python editor.
	
	
Since I used BertForSequenceClassification for training my deal sentence classifier, there is usage of 	bert tokens for the purpose of training. And I used base-uncased version of bert tokens, which should 	be downloaded prior to its usage. Here is the syntax for it:-
	
***from transformers import BertTokenizer***

***#Load the BERT tokenizer.***

***print('Loading BERT tokenizer...')***

***tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 				do_lower_case=True)***

## **Usage(with flow of "deals_complete_pipeline_pseudocode.py"):-**

(a) Firstly data in form of strings needs to be input from a source. For my case, I had my input from mongo collection . In that, I considered the ***‘title’*** and ***‘description’*** fields of each records.

(b) In case the input contains multiple sentences, they need to be pre-processed  using ***‘sent_tokenizers’***, and other preprocessing methods, so that we end up having a list where each element of the list is a single element list containg one sentence.

(c) Now stanford NER tagger is run over the obtained list so that we get labels **(either ORGANIZATION, PERSON, or MONEY)** for each word of each sentence(that are contained in a singular fashion in each sublist of main list).

(d) Then the indices of the sentences that have money labels are filtered. For those sentences stanford POS tagger is ran. From those sentences, ***stopwords and words having positive NER labels are removed.*** And for remaining part of the sentences, we check ***POS labels of each words whether they are verb, adjectives or nouns and filter them out.***

(e) Now we change those filtered words into their base words using  ***‘WordNetLemmatizer’*** and check if the base words are present in BOW containing deals keywords. *If they are present in BOW , the corrsponding sentences are appended into a sentence list.* And finally create Dataframe  for this list.

(f) Next  the ***trained bert model*** is called. Then the ***token ids and masking ids*** for each sentence of the dataframe are generated. Now these variables are feeded into the model to get the predicted results.

(g) Thereafter  the sentences whose labels are identified ‘1’ are considered and stanford NER tagger is ran over those to get organizations list and deal amount from those.

(h) Finally a **dataframe having Sentence, their predicted labels(status) and deal amout, organization given their status is ‘1’ is created.**

(i) Once the dataframe is created,  a csv can be created from it and stored to a desired location.

## **Brief overview of "deal_sent_classify.ipynb(training notebook)" & "Deal_sent_classify_test1.ipynb(Test notebook)":-**
### **(1)Training notebook:-**
(a)1st mount the colab and define path followed by setting GPU runtype environment. Now open the *'deal_data.csv'* created by following ***steps[a-e]*** of usage segment. It contains ***'link', 'sentences', 'labels', 'ID' columns.*** <u>Note:-user can use cutomized input since deal_data.csv belonged to a client and is not available on this repo. Just extract ***text and lable*** colums from customized data and follow below steps.</u>

(b)consider train data *length<512*, and double the numbers of train samples where positive labels(1/Deal).

(c)Now we load **BertTokenizer('bert-base-uncased')**==><u>BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True).</u> Thereafter we generate **tokenized words via tokenizer, convert them to ids, pad/truncate to a MAX_LEN and create corresponding attention masks(mask_id)**(0 for PAD/0 tokens, 1 for other token_ids) of normalized token_ids(equal lengths).

(d)Do train-validation split(90:10) for both input_ids and mask_id in ***same state for identical stratification(so that input_id remain mapped to mask_id during train and validation).*** Define *batch_size(here 30)* and prepare train and validation dataloaders.

(e)**load pretrained BertForSequenceClassification("bert-base-uncased")** ==> <u>BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2(2+ if multiclass)).</u> Set it on GPU(.cuda() method).

<u>**(f)Define:-**</u>

***(i)optimizer parameters(type =AdamW, learning rate , eplsilon etc)***==>used to calculate updated weights during backprop.'Epsilon(e)' is added to 'moment'(square root of moving average of squared past gradients/variance). *These both used to stablize noisy,fluctuating weights and epsilon helps non-zero division when variance tends to 0.*

***(ii)epochs, num_train_steps =epochs*batch_size, loss_type(default:- categorical cross entropy).***

***(iii)schedulers attributes(optimizer, num_warmup_steps, num_train_steps)***==>Used to update the re-calculated weights during backprop.

(iv)Define metrics function, or use suitable library.
            
<u>**(e)Train and validationblock(refer code notebook alongside)**:-</u> 
    
(i)Pick a random seed value and set torch, cuda, numpy manually to it(for reproducible results).

(ii)Now ***initialize model.train()*** start the training loop epoch wise. Access batch tuple elements and feed them inside model() attribute. ***Loss is returned as 1st output element***(if using customized loss with weights, don't pass labels in model() so to get logits as 1st o/p to calculate loss by feeding logits to invoked loss constructructor with weights of o/p classes.)(previously defined in (d-ii)).

(iii) Aggregate the loss to epoch level variable across each batch. After ***loss calculation and updation, perform backprop (normalize gradients to 1 and update weights using opimizer.step(),scheduler.step()). Calculate average loss and append epoch level losses to global train_loss list***.

(iv)Enter the **validation loop** inside given epoch and perform same ***batchwise operation(loss calculation,metrics).*** Aggegate these to relevant_metrics and averge validation loss at epoch level ( at end of batch level loop). Also append epoch level avg loss to global list like in last step. ***Prefer saving the model_checkpoint with bestmetrics here for optimum weights(or at last after loss plot). For our case, best model has :-<u>{avg_train_loss =0.25, validation_accuracy =0.84}</u>***

(v)Populate the loss claculation, checkpoint steps along training loop execution and use *tqdm* to view the progress.
            
(f)Plot the epoch level train and validation losses with matplotlib and save the model if not saved iteratively at validation level(make new model directory and use .save_pretrained(directory path) method).

### **(2) Testing notebook:-**
(a)Set the ***mount to drive path and set runtype environment to GPU.*** Open the **"deals_test1.csv"**.

(b)**Load both fine-tuned model and tokenizer saved in Trained notebook by.from_pretrained(trained_model_directory path).**
(c)Pre-process the sentences across each test example similar to **[a-c] of training notebook**. This yeilds ***input_ids, attention_mask(equal to MAX_LEN), and labels(from label colum of test file).*** Define relevant metrics function or import them from suitable libraries(scikit-learn, keras).

(d)Create test dataloader with batch_size(here 30). After this ***repeat the validation loop part i.e e[iv-vi] except save checkpoint part .Also add predicting output from logits by selecting maximum elemnet index in logits 'array' part***==><u>pred_labels_i = np.argmax(predictions[i], axis=1).flatten()'</u> and aggregating it at epoch level. Calculate relevant aggregated test metrics. Calculated metrics on **test_set** are mentioned in file path :-**"NLP-text_classification/classification_metrics.txt"(accuracy:- 87.83%, f1:-0.88)**

(e)For the custom test input directly tokenize and convert into ids on one go by tokenizer.encode() method. Pass the ***input_ids*** into loaded model() constructor. Finally predict output from logits by selecting maximum element index in logits 'array' part***==><u>pred_labels_i = np.argmax(predictions[i], axis=1).flatten()'</u>

# Note:- In case of train file not visible on github-> 
***(a)replace github by githubcolab in the path of "deal_sent_classify.ipynb/Deal_sent_classify_test1.ipynb" in url.

(b)Download the jupyter file in local and view in jupyter-lab/notebook.***






    



