# Custom NER-tagger development pipeline(Flow of train and inference for custom Name, entity recognition tagger)

## **(1)Training notebook:-**

(a)1st mount the colab and define path followed by setting GPU runtype environment. Now open the "ner_dataset.csv"(download from:- https://www.kaggle.com/datasets/debasisdotcom/name-entity-recognition-ner-dataset on drive). It contains ***4 columns namely:- 'Sentence#', 'Word', 'POS(Part of speech)', 'Tag'.***

#### <u>**(b)Data format**:-</u>
(i)Each word of a indexed sentence# is mapped against its POS and tag.

<u>(ii)The tags are of following entities </u>:-[

**ORGANIZATION** - Georgia-Pacific Corp., WHO

**PERSON** - Eddy Bonte, President Obama

**LOCATION** - Murray River, Mount Everest

**DATE** - June, 2008-06-29

**TIME** - two fifty a m, 1:30 p.m.

**MONEY** - 175 million Canadian Dollars, GBP 10.40

**PERCENT** - twenty pct, 18.75 %

**FACILITY** - Washington Monument, Stonehenge

**GPE(Geopolitical entity)** - South East Asia, Midlothian
].

(iii.) These tags are in BIO (Beginning, intermediate, outer) format that generalises to word(s). For eg:- set of words namely 'Georgia-Pacific Corp., WHO' is tagged "org"(Organization). Now BIO format would render following comparison between each word chunk and tag:-

<u>**Word**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Tag**</u>

Georgia&emsp;&emsp;&emsp;&emsp;&emsp;B-org

"-"&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;I-org

Pacific&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;I-org

Corp.&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;I-org

","&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;I-org

WHO&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;I-org

word(outside)org&emsp;O

------------------------------------------------------
***This shows that head of a tag's wordgroup is prefixed by "B", the body including tail by "I" and words outside tag's ambit by "O".***

(iv.) Take a count of B,I and O combinations of different tags==> data["Tag"].value_counts()

#### **(c) <u>Create a SentenceGetter class**</u>:- 
to group the Words, POS, and Tags of given indexed sentence together.Code mentioned in train notebook. Also create an incremental mapping for B/I/O combinations of each tag + PAD(**tag2idx**).

(d)Next we load BertTokenizer('bert-base-uncased')==>BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False). We tokenize wordlist of sentence number further. This increases number of word-chunks. Since original words were previously mapped with tags, *for each word tokenized in "n" chunks, the "n" is multiplied to corresponding original tags.* This way the ***word-->tag/labels*** mapping is maintained even after tokenization. A function **tokenized_text_labels** is used in train file for this.

(e)Now convert ***tokenized words to input_ids, pad/truncate to a MAX_LEN and create corresponding attention masks(mask_id)***(0 for PAD/0 tokens, 1 for other token_ids) of normalized token_ids(equal lengths= MAX_LEN). Also use ***incremental mapping(tag2idx created in step c) to convert padded label list to number list for training.***

(f)Do train-validation split(90:10) for both input_ids and mask_id in ***same state for identical stratification(so that input_id remain mapped to mask_id during train and validation).*** Define *batch_size(here 32)* and prepare train and validation dataloaders.

(g)**load pretrained BertForTokenClassification("bert-base-cased")** ==> <u>BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=len(tag2idx), output_attentions=False, output_hidden_states=False).</u> Set it on GPU(.cuda() method).

#### <u>**(h)Define:-**</u>

***(i)model_parameters, optimizer parameters(type =AdamW, learning rate , eplsilon etc)***==>used to calculate updated weights during backprop.'Epsilon(e)' is added to 'moment'(square root of moving average of squared past gradients/variance). *These both used to stablize noisy,fluctuating weights and epsilon helps non-zero division when variance tends to 0.*

***(ii)epochs, num_train_steps =epochs*batch_size, loss_type(default:- categorical cross entropy).***

***(iii)schedulers attributes(optimizer, num_warmup_steps, num_train_steps)***==>Used to update the re-calculated weights during backprop.

*(iv) Import relevant metrics(here f1_score,accuracy_score from seqeval.metrics).*

#### <u>**(j)Train and validationblock(refer code notebook alongside)**:-</u> 
    
(i)Pick a random seed value and set torch, cuda, numpy manually to it(for reproducible results).

(ii)Now ***initialize model.train()*** start the training loop epoch wise. Access batch tuple elements and feed them inside model() attribute. ***Loss is returned as 1st output element, logits as 2nd***(if using customized loss with weights, don't pass labels in model() so to get logits as 1st o/p to calculate loss by feeding logits to invoked loss constructructor with weights of o/p classes.)(previously defined in (d-ii)).

(iii) Aggregate the loss to epoch level variable across each batch. After ***loss calculation and updation, perform backprop (normalize gradients to 1 and update weights using opimizer.step(),scheduler.step()). Calculate average loss and append epoch level losses to global train_loss list***.

(iv)Enter the **validation loop** inside given epoch and perform same ***batchwise operation(loss calculation,metrics).For validation accuracy, taglist prediction for a batch is needed. This is done by extracting the maximum of logits array each actual tag and consolidation in an array***==><u>[list(p) for p in np.argmax(logits, axis=2)]</u>. Aggegate these to relevant metrics and averge validation loss at epoch level ( at end of batch level loop). Also append epoch level avg loss to global list like in last step. ***Prefer saving the model_checkpoint with bestmetrics(here epoch_accuracy) here for optimum weights(or at last after loss plot).***

(v)Populate the loss claculation, checkpoint steps along training loop execution and use *tqdm* to view the progress.
            
(k)Plot the epoch level train and validation losses with ***matplotlib*** and ***save the model if not saved iteratively at validation level***(make new model directory and use <u>.save_pretrained(directory path) method)</u>.

### **(2) Testing notebook:-**
(a)Set the ***mount to drive path and set runtype environment to GPU.*** Open your **"test.csv**.

(b)**Load both fine-tuned model and tokenizer saved in Trained notebook by.from_pretrained(trained_model_directory path).**
(c)Pre-process the sentences across each test example similar to **[a-e] of training notebook**. This yeilds ***input_ids, attention_mask(equal to MAX_LEN), and labels(mapped with input_ids of tokenized words).*** Define relevant metrics function or import them from suitable libraries(scikit-learn, keras).

(d)Create test dataloader with batch_size(here 30). After this ***repeat the validation loop part i.e e[iv-vi] except save checkpoint part .For taglist prediction of batch, extraction of the maximum of logits array each actual tag and consolidation in an array is needed***==><u>[list(p) for p in np.argmax(logits, axis=2)]'</u> and aggregating it at epoch level. Calculate relevant aggregated test metrics.

(e)For the custom test input directly tokenize and convert into ids on one go by tokenizer.encode() method. Pass the ***input_ids*** into loaded model() constructor. Finally for taglist prediction of input, extraction of the maximum of logits array each actual tag/input_ids(of custom input) is needed***==><u>[list(p) for p in np.argmax(logits, axis=2)]'</u>

# Note:- In case of train file not visible on github-> 
***(a)replace github by githubcolab in the path of "bert_ner.ipynb/bert_ner_test.ipynb" in url.

(b)Download the jupyter file in local and view in jupyter-lab/notebook.***

