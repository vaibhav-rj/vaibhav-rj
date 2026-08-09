# Autoregressive/Causal models anf MAsked models modelling(using .trainer() wrapper and accelerator loop):-

## Purpose:-
This documentation talks about models:- **Autoregressive/Causal and masked models train and inference methodology and flow.** Both of these are based on transformer archtitecture but different in nature. While causal models are decoder based(generative) models that form principle/basis of much complex LLM models(GPT series, llama series) and modern Generative AI, Masked models are encoder based(discriminative) models. 

- Causal models generate next sequences based on ***search strategy(greedy, topK-topP sampling, beam)*** which differ with diverse use cases ***(from fixing spelling errors to chatting, story-telling etc).***

- Masked models predict the masked entity on ***varying strategies (single word mask, span of words masks, multi-token masks)*** in a phrase given actual masks label. Applicable for use cases like:- ***Reveal missing Semantic Clues, improve keyword-alignment and word selection etc.***

## General flow:-

- <u>**Environment setup, installation and mount:-**</u>

    - 1st change runtype to GPU in colab(other editor) and install datasets and transformers(latest versions).
    - Install ***"torch"***(if not present) and ***"TorchCodec"(used to read video, audio datasets by pytorch)***. TorchCodec has difficulties in reading data if there are ***incompatablities between pytorch and TorchCodec version relative to Dataset library(used to load datasets). 
    
    - If compatiblity error is unresolved ***try older version of TorchCodec i.e. TorchVision(for older pytorch versions < 4.11.0).*** *Set os(operating system)'s environment for Torchvision as 'pyav backend'.Define DummyVideoreader class and pass it to Videoreader subclass(reads video data in TorchVision) ===> for successful import of VideoReader from TorchVision.* 

- <u>**Data preparation:-**</u> Much of the data preparation remanins the same for both models and theirs training types. Difference appears while collating data in both cases(that'll be covered in modelling segment). Here's brief overview of common segment:-

    - Try loading data from "Dataset" library. We've used "WikiText 2". It contains of ***DatasetDict of 3 keys("train", "validation", "test")*** and values of ***each of those keys has arrow_dataset.Dataset(like normal set/list) of iterative chunks of individual text examples(as value for 'text' key)***.

    - Select **model checkpoint(Causal models :-gpt2, llama2.7 like lower version; Mask model:- distilbert, distilroberta)**, so as to load their respective pretrained tokenizers and models for fine-tune/training.

    - We may load the tokenizers and assign pad tokens as EOS(end of sentence tokens) in case we are to use tokens of uneven length for train. However in our case:-
        - we shall first ***tokenize all text of input data***(for 3 keys :- train, test, validation keys) by map method & tokenize fn. This generates ***input_ids and attention_mask*** for each text

        - We'll then ***concatanate the "input_ids" & "attention_masks"** for each text examples ,then ***partition them by a given block_size, and trim the last chunk(if < block_size)*** for all 3 keys via map method & group_text fn and name output( here **"lm_datasets"**). Observe the datatypes of o/p excatly like loaded dataset.

    - Our approach already ***yeilds equal dense tokens, hence assigning PAD =EOS tokens is completly as choice***.

- <u>**Modelling(via .trainer() method)**</u>:-
    - Import AutoModelFor(Masked/Causal)LM, trainer and TrainingArguments, DataCollatorForLanguageModelling from Transformers.
    - Now invoke DataCollatorForLanguageModelling() class feeding in ***"tokenizer(pretrained)"*** and ***"mlm_probablity"=(causal:-false|masked=0.15 as convention)*** Here "mlm" stands for <u>masked language models.</u> Keeping it ***0.15 in case of masked modelling assigns 15% of the tokens randomly during each epoch. This in a way regularizes our training*** .
    - Now TrainingArguments[
        - output_dir, 
        - eval_strategy=epoch,(***epoch wise loss and metric score for train/val set***) 
        - learning_rate=[1e-5 to 5e-5](here 2e-5 taken), 
        - weight_decay =0.01(***regularizer that penalizes large/noisy model weights***), 
        - fp_16=True(***for gpu***), 
        - save_strategy= epoch,(***saving checkpoints after epochs but this condition must be met with latter 2***) 
        - save_total_limit=1(***this and next attribute for saving drive and notebook memory from enormous space occupied by successive saved model_checkpoints***), 
        - load_best_model_at_the_end =True,(***efficient saving and loading strategy of model_weights***),

        - per_device_train_batch_size=batch_size(***as we shall define via train dataloaders in accelerator train phase***),
        - per_device_eval_batch_size=batch_size(***as we shall define via train dataloaders in accelerator train phase***),
        - logging_steps=logging_steps(to show logs with each bactch iter), ]

        class constructor gets defined.

    - Now trainer[
        - model=model(AutoModelFor(Masked/Causal)LM),
        - args=training_args(preceding constructor assigned to a variable),
        - train_dataset=lm_datasets["train"],
        - eval_dataset=lm_datasets["validation"],
        - data_collator= data_collator(DataCollatorForLanguageModelling-***same for both train/val***),
        - processing_class= tokenizer,]

        class constructor gets defined.

    - There is option to ***set loss in trainer***. However for both cases we avoid it(let it be default):-
        - <u>Causal model</u>:- 
        Default_loss =***"ForCausalLMloss"***(wrapper around *"cross-entropy loss"* for causalmodelling). We don't change it to ***"standard cross-entropy loss"*** because:-
        
            - (1)**"Cross entropy loss**" will calculate the <u>loss of predicting Token N given Token N, which ruins the causal constraint</u>, while **"ForCausalLMLoss"**
<u>fixes this by shifting the logits and labels so that the logit at position N is compared against the actual token label at position N+1</u>.
            - (2)**"Standard Cross Entropy"**: Requires you to <u>manually generate attention masks and multiply tensors to exclude the prompt or padding tokens
from the gradient calculation</u>.Whrereas **"ForCausalLMLoss Wrapper"**: <u>Leverages PyTorch's default ***ignore_index=-100*** convention. Any token in
your labels configuration set to -100 is automatically ignored during the loss computation</u>, making masking user prompts incredibly clean.
        - <u> Masked mode</u>:-
        Default loss is :-***"standard/categorical cross-entropy_loss"*** --> is go to loss in multiclass/label discriminative models. Hence no point of manually altering it.

    - Now trainer.train() is run, loss and metrics evaluated with each epoch and the best one's checkpoint saved. Finally perplexity(e^loss) is calculated on best model for validation dataset(in our case it was **36.84 for causal model and 7.21 for masked model on trainer modelling approach**).

    - Since there is ***same data_collator(DataCollatorForLanguageModelling) argument for train-validation set inside trainer class contructor attributes***, there is fundamental problem of ***fluctuating loss and random noise*** in validation set with each epoch. This shall be handled efficiently when we discuss **accelerator modelling approach**.

- <u>**Modelling(via accelerator method)**</u>:-

    -Import relevant libraries(accelerate.accelerator, accelerate.DataLoaderConfiguration; DataLoader & AdamW from torch; default_data_collator & DataCollatorForLanguageModelling from Transformers). Call ***pre_trained AutoModelFor(Masked/Causal)LM again***.

    - <u>**Dataloader preparation**</u>:- Now train, eval and test dataloaders are created with a given **batch size(here 32--> advised to keep 16, 32, 64, 128)**. 
        - **Causal models**:-
        Here one can use default_data_collator & DataCollatorForLanguageModelling(**latter with mlm= False**) in any permutation for Train, eval, test. Even if DataCollatorForLanguageModelling is used as collator for eval and test set, it won't change their examples in successive epoch iteration/re-running for loss, metric calculations(as mlm_probablity=False) ***Shuffle attribute for train_dataloader is advised to be kept True for regularization of during fine-tune(for eval& test, it can be ignored).***

        - **Mask models**:- Now both  default_data_collator & DataCollatorForLanguageModelling(latter's class constructor again invoked with mlm_probablity=0.15 as in trainer approach) are initilaized .
            - **(1)** Since "initialized DataCollatorForLanguageModelling class" & **"shuffle=True"** can help regularize model during training, ***train_data dataloader must have these***. 

            - **(2)** However our validation data will also change with each epoch iteration or next complete acclerator train loop execution run or recreation of dataloader using DataCollatorForLanguageModelling. This means that validation metrics shall :-
                - **(a)** give **fluctuating perplexity scores**("e^loss", metric for masked/causal model away from actual perplexity on fixed validation data) and **random noise**.
                - **(b)** also resolutions like random seeding at start of acceleration train loop in a quest of training resproduciblity shall hamper as despite similar/same weights, data distribution of validation set shall change whcih will fluctuate metrics, loss and hence perplexlity.
                - **(c)** To tackle this, we'll apply the masking once on the whole validation & test set(using invoked DataCollatorForLanguageModelling), and then use the default data collator while creating ***eval_dataloader*** and ***test_dataloader***(**to freeze those randomized masks**) in Transformers to collect the batches during evaluation(on eval_dataloader) and test_metrics calculation(on test_dataLoader).---> ***this segment has explained in MaskModelling codefile accelerator train section with details and intuition.***
                - **(d)** We'll save **test_dataloader** along with **saved_model** to (i) check if freezing approach in above step(c) actually works and whether ***we get same loss on running trained models on test_dataloader again and again***, (ii) to see ***performance of our model on test_set/dataloader***. 

    - Now the model parameters are defined[optimizers(lr =3e-5, eps =1e-8), dataloader_config(***to save and load test_dataloader in same key-value dict structure***)==><u>dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=**True**)</u>, num_train_epochs, total_steps, scheduler]==> similar to what we did in sequence classification training **NLP-text_classification/deal_sent_classify.ipynb**[refer its ***README.md***]

    - Now ***invoke Accelerator constructor with dataloader_config attributes***. Call its ***prepare function with attributes*** :-<u>{"model", "optimizer", "train_dataloader", "eval_dataloader", "test_dataloader"}</u> and *return o/p to same attribute_variables to <u>sync them with gpu</u>*.

    - Now follow same train_epoch_loop where model is trained in successive batches across epochs(with backprop) and train loss, validation loss and metrics populated at epoch level(former 2 losses appened to corresponding global_lists to be plot after train)--> similar to deal_sent_classification [again refer its README.md @ **NLP-text_classification/Readme.md**]

    - Like in trainer method, here also we can calculate final perplexity score on validation_dataloader by our fine-tuned/trained model(**causal model:- 34.75 and Masked_model:- 5.11**).

- <u>**Saving models**</u>:-

    - **If fine-tuned by trainer method**:- By trainer.save_model(model_directory_path) -->saves both tokenizer and fine-tuned/trained model.

    - **If fine-tuned by accelerator loop method**:- 
    
        - Invoke acclerator class's (***wait_for_everyone()-->unwrap_model(model) fn/method***).

        - Check if accelerator class is in main process--->if true-->save ***unwrapped_model and tokenizer separately to given **"model_directory_path"** via .save_pretrained() method.***

        - Define test_dataloader's state(similar to config) and finally save it using accelerator.save() method==><u>accelerator.save(dataloader_state, os.path.join("model_directory_path", "test_dataloader_state.pt"))</u>.--[done in Masked modelling in our codefiles, although can be done in causal modelling too].

- <u>**Loading saved models(for inference)**</u>:-

    - Load same libraries as in trainer and accelerator fine-tuning phase + ***AutoTokenizer, AutoModelFor(Masked/Causal)LM+ dataset.Dataset. Define model_directory_path***.

        - **Loading fine-tuned tokenizer and model**:- simply by AutoTokenizer.***from_pretrained(model_directory_path***), AutoModelFor(Masked/Causal)LM.***from_pretrained(model_directory_path***)

        - **Loading test-dataloader**(For now only done for Masked modelling):-

            - (1)Enable stateful dataloader configuration if tracking previous saved states of dataloader ==><u>dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=True)</u> and invoke Accelerator class constructor using **dataloader_config** attribute.

            - (2)Reconstruct saved test_dataloader==><u>dataloader_file = os.path.join("model_directory_path", "test_dataloader_state.pt")</u>. Now ***load the "saved_state" from the dataloader_file***==><u>saved_state = torch.load(dataloader_file, map_location="cpu", weights_only=False)</u>

            - (3)Rebuild the **"HF Dataset"** object from the saved dict derived from **"saved_state"** and finally rebuild the **standard PyTorch Dataloader** on it invoking DataLoader class constructor(as done during accelerator loop training).

            - (4) Use this to calculate metrics, loss on test_dataloader by code similar to validation loop inference during each epoch of training. We'll observe that we end up **same loss(1.958) & perplexity on this test_dataloader(7.087)** which confirms:-<u>(i)test_dataloader masks froze + (ii)well behaved model(small_test_loss marginally greater than test_loss:-1.63) as proposed in 2(d) of Modelling mask models via accelerator method section above.</u>

- <u>**Inference**</u>:-

    - **Causal model**:- 3 broad stratagies of next tokens generation:-

        - <u>*Greedy approach*</u>:- ***default decoding*** strategy, finds ***most probable next token/word(highest softmax probablity)***. Unless specified in GenerationConfig, this strategy generates a *maximum of 20 new tokens. <u>Used where factual accurcy and deterministic output/reponse is pertinent. Prone to repeatinons on high token length. Gives same o/p on each iteration*.</u> Usecase eg:- **coding & syntax generation, factual data extraction**.

        - <u>* Top_k(k most probable tokens/word) & Top_p sampling(topmost proable tokens/word untill cumulative sum breaches 'p' value)*</u>:- used where *diversity and creativity of response is important. Robust for non-repeatations on higher lengths. Gives differnt o/p on each iteration(typically 'top_p' as order of summation varies in each iter).* Use case egs:- **Story generation, conversational AI agents/chatbots.**

        - <u>*Beam search*</u>:- keeps track of several generated sequences (beams) at each time step. ***After a certain number of steps, it selects the sequence with the highest overall probability.*** Unlike greedy search, this strategy can “look ahead” and pick a sequence with a higher probability overall even if the initial tokens have a lower probability. Use case egs:- **text_summarization, machine_translation(describing an image or speech recognition)**

            - (1) <u>Beam_search + do_sampling=True==>Multinomial search</u>(here but beam search will still greedily prune out low probability sequences between steps.)

        - <u>*Effect of tempeature*</u>: used where do_sampling= True(top_k, top_p, multinomial samplings).

            - (1) ***Lower temperature(T< 1) sharpens distribution making choices predictable***(more like greedy or standard beam-->at T=0.1 for eg.). Similar o/p in each iter

            - (2) ***while higher tempearture(T> 1) flattens distribution to make diverse o/ps***(greater randomness than normal top_p or top_k head samplings). Greater variation in each iter o/p.

    - **Masked model**:- 3 broad strategies exist(2 adopted here):-

        - <u>Greedy mask filling</u>:- fills highest probablity tokens [**single word mask**, <u>**multi-toke masks**(implemented in code file</u>)]. Use Case egs: ***Single-word replacement, deterministic semantic tagging, and spell-checking***.

        - <u>Top-K/Top-P Samplingfor Masks</u>:- sampling from a filtered probability distribution(again here top-p sampling might yeild varying results on each iter as order of summation may differ in each new iter). Can be done for [<u>**single word mask**(implemented with k=5 in codefiles)</u>, **multi-toke masks**]. Use case egs:-***Generating creative alternatives, data augmentation, or diverse headline variations***.

        - <u>Iterative Multi-Mask Decoding</u>:- sequentially or via masked refinement loops (pseudo-iterative generation) to prevent independent slot conflicts. Use case egs:- ***Filling multiple interdependent [MASK] slots(word_span masks)*** 

---------------------------------------------------------------
## Note:-
For greater understanding and intuition, refer this README.md with modelling files of:- ***(i)Causal modelling("Fine-tune a causal(auto-regressiv) language model.ipynb"), (ii) Mask modelling("Fine-tune a Masked language model.ipynb")***.


