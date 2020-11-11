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

2. Download datasets
   Inside `./datasets` folder, run `./download_datasets.sh` to download MOSI and MOSEI datasets

3. Training MAG-BERT / MAG-XLNet on MOSI

   First, install python dependancies using `pip install -r requirements.txt`

   **Training scripts:**

   - MAG-BERT `python multimodal_driver.py --model bert-base-uncased`
   - MAG-XLNet `python multimodal_driver.py --model xlnet-base-cased`

   By default, `multimodal_driver.py` will attempt to create a [Weights and Biases (W&B)](https://www.wandb.com/) project to log your runs and results. If you wish to disable W&B logging, set environment variable to `WANDB_MODE=dryrun`.

4. Model usage

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

   input_ids, attention_mask, position_ids are torch.LongTensor of shape (batch_size, sequence_length). For more details on how these tensors should be formatted / generated, please refer to `multimodal_driver.py`'s `convert_to_features` method and [huggingface's documentation](https://huggingface.co/transformers/preprocessing.html)

## Dataset Format

All datasets are saved under `./datasets/<DATASET>/` folder and is encoded as .pickle file.
Format of dataset is as follows:

```python
{
    "train": [
        (words, visual, acoustic), label_id, segment,
        ...
    ],
    "dev": [ ... ],
    "test": [ ... ]
}
```

- word_ids (List[str]): List of words
- visual (np.array): Numpy array of shape (sequence_len, VISUAL_DIM)
- acoustic (np.array): Numpy array of shape (seqeunce_len, ACOUSTIC_DIM)
- label_id (float): Label for data point
- segment (Any): Unique identifier for each data point

Dataset is encoded as python dictionary and saved as .pkl file

```python
import pickle as pkl

# NOTE: Use 'wb' mode
with open('data.pkl', 'wb') as f:
    pkl.dump(data, f)
```

## Contacts

- Wasifur Rahman: rahmanwasifur@gmail.com
- Sangwu Lee: sangwulee2@gmail.com
- Kamrul Hasan: mhasan8@cs.rochester.edu
