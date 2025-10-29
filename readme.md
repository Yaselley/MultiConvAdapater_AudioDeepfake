# 🧩 MultiConvAdapter: A Parameter-Efficient Multi-Scale Convolutional Adapter for Synthetic Speech Detection

**A Parameter-Efficient Multi-Scale Convolutional Adapter for Synthetic Speech Detection**

Recent synthetic speech detection models typically adapt a pre-trained self-supervised learning (SSL) model via full fine-tuning, which is computationally demanding. Parameter-Efficient Fine-Tuning (PEFT) offers a lightweight alternative, but existing methods lack the inductive biases required to model the **multi-scale temporal artifacts** characteristic of spoofed audio.

This repository introduces **MultiConvAdapter**, a parameter-efficient architecture designed to address this limitation. MultiConvAdapter integrates **parallel convolutional modules** within the SSL encoder, enabling the model to capture **short-term artifacts and long-term distortions** simultaneously.

With only **3.17M trainable parameters** (~1% of the SSL backbone), MultiConvAdapter achieves **state-of-the-art performance** on five public datasets, outperforming both full fine-tuning and existing PEFT methods.

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/<your_username>/MultiConvAdapter.git
cd MultiConvAdapter

# Create a conda environment
conda create -n multiconvadapter python=3.10
conda activate multiconvadapter

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Training

1. **Configure Paths**  
   Open `config.py` and set the following:
   - **Training and validation data paths**  
   - **Feature files (FALCS) paths**  
   - **Wav2Vec/XLS-R model weights path**

2. **Start Training**  
   Run the training script:
   ```bash
   python3 main.py
   ```

---

## 🔍 Evaluation

1. **Configure Test Set**  
   Before running evaluation, open `config.py` and update the path to the **TEST set**.

2. **Run Evaluation**  
   Launch the evaluation script:
   ```bash
   python3 evaluation/evaluation.py -o output_dir
   ```
> **Note:** Ensure that the TEST set path in `config.py` is correctly set before starting evaluation.


## 🏆 Best Reported Weights with AASIST

You can download the best-performing model checkpoints below:

- **ASV19-trained model**
- **ASV5-trained model**

Both models were trained using the configuration files available here:  
[🔗 Google Drive Folder](https://drive.google.com/drive/folders/1QfirjfC_qkWVjwjOKnr9s5JdR5_lZhOa?usp=sharing)






