# Multimodal Adaptation Gate (MAG)

Open source code for ACL 2020 Paper: [Integrating Multimodal Information in Large Pretrained Transformers](https://www.aclweb.org/anthology/2020.acl-main.214.pdf)

## Getting started

1. Configure `global_configs.py`

    `global_configs.py` defines global constants such as dimension of each data modality (text, acoustic, visual) and cpu/gpu settings. It also defines which layer MAG will be injected. The following are default configuration.

    ```python
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"

    DEVICE = torch.device("cuda:0")

    # contextualized text embedding from pre-trained BERT / XLNet
    TEXT_DIM = 768
    # acoustic / visual embedding dimension from MOSI dataset
    ACOUSTIC_DIM = 74
    VISUAL_DIM = 47

    XLNET_INJECTION_INDEX = 1
    ``` 

2. Training MAG-BERT / MAG-XLNet on MOSI

    First, install python dependancies using ```pip install -r requirements.txt```

    **Training scripts:**

    - MAG-BERT ```python multimodal_driver.py --model bert-base-uncased```
    - MAG-XLNet ```python multimodal_driver.py --model xlnet-base-cased```

    By default, ```multimodal_driver.py``` will attempt to create a [Weights and Biases (W&B)](https://www.wandb.com/) project to log your runs and results. If you wish to disable W&B logging, set environment variable to ```WANDB_MODE=dryrun```.

3. Model usage

    We would like to thank [huggingface](https://huggingface.co/) for providing and open-sourcing BERT / XLNet code for developing our models. Note that bert.py / xlnet.py are based on huggingface's implmentation.

    **MAG**
    ```python
    from modeling import MAG

    hidden_size, beta_shift, dropout_prob = 768, 1e-3, 0.5
    multimodal_gate = MAG(hidden_size, beta_shift, dropout_prob)

    fused_embedding = multimodal_gate(text_embedding, visual_embedding, acoustic_embedding)
    ```

    **MAG-BERT**
    ```python
    from bert import MAG_BertForSequenceClassification

    class MultimodalConfig(object):
        def __init__(self, beta_shift, dropout_prob):
            self.beta_shift = beta_shift
            self.dropout_prob = dropout_prob

    multimodal_config = MultimodalConfig(beta_shift=1e-3, dropout_prob=0.5)
    model = MAG_BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', multimodal_config=multimodal_config, num_labels=1,
        )
    
    outputs = model(input_ids, visual, acoustic, attention_mask, position_ids)
    logits = outputs[0]
    ```

    **MAG-XLNet**
    ```python
    from xlnet import MAG_XLNetForSequenceClassification

    class MultimodalConfig(object):
        def __init__(self, beta_shift, dropout_prob):
            self.beta_shift = beta_shift
            self.dropout_prob = dropout_prob

    multimodal_config = MultimodalConfig(beta_shift=1e-3, dropout_prob=0.5)
    model = MAG_XLNet_ForSequenceClassification.from_pretrained(
            'xlnet-base-cased', multimodal_config=multimodal_config, num_labels=1,
        )
    
    outputs = model(input_ids, visual, acoustic, attention_mask, position_ids)
    logits = outputs[0]
    ```

    For MAG-BERT / MAG-XLNet usage, visual, acoustic are torch.FloatTensor of shape (batch_size, sequence_length, modality_dim).

    input_ids, attention_mask, position_ids are torch.LongTensor of shape (batch_size, sequence_length). For more details on how these tensors should be formatted / generated, please refer to ```multimodal_driver.py```'s ```convert_to_features``` method and [huggingface's documentation](https://huggingface.co/transformers/preprocessing.html)
    
4. Dataset Format

    All datasets are saved under `./datasets/<DATASET>/` folder and is encoded as .pickle file.
    Format of dataset is as follows:
    ```python
    {
        "train": [
            (word_ids, visual, acoustic), label_id, segment,
            ...
        ],
        "dev": [ ... ],
        "test": [ ... ]
    }
    ```
    - word_ids (List[int]): List of word_ids for each word token
    - visual (np.array): Numpy array of shape (seq_len, VISUAL_DIM)
    - acoustic (np.array): Numpy array of shape (seq_len, ACOUSTIC_DIM)
    - label_id (float): Label for data point
    - segment (str): Unique ID for each data point
    
    Dataset is encoded as python dictionary and saved as .pickle file
    ```python
    import pickle
    
    # NOTE: Use 'wb' mode
    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)
        
    ```
    
    In case of MOSI, text modality is represented as list of word token ids, so `word2id.pickle` is used to convert
    word token ids (List[int]) -> words (List[str]). 
    
    Here, `word2id.pickle` is a python dictionary that maps word (str) to id (int) as used in `convert_to_features`:
    ```python
    def convert_to_features(examples, max_seq_length, tokenizer):
        with open(os.path.join("datasets", args.dataset, "word2id.pickle"), "rb") as handle:
            word_to_id = pickle.load(handle)
        id_to_word = {id_: word for (word, id_) in word_to_id.items()}

        features = []

        for (ex_index, example) in enumerate(examples):

            (word_ids, visual, acoustic), label_id, segment = example
            sentence = " ".join([id_to_word[id] for id in word_ids])

            tokens = tokenizer.tokenize(sentence)
            inversions = get_inversion(tokens)
            ....
        
    ```
    
    Alternatively, feel free to change `word_ids` (List[int]) to `sentence` (str), if your multimodal dataset directly provides word / sentence instead of word ids.
    
    

## Contacts
- Wasifur Rahman: rahmanwasifur@gmail.com
- Sangwu Lee: sangwulee2@gmail.com
- Kamrul Hasan: mhasan8@cs.rochester.edu
