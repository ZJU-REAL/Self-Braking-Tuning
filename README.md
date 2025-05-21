<div align="center">
<h2>Let LLMs Break Free from Overthinking via Self-Braking Tuning</h2>
  <div align="center">
</div>
  <section class="hero">
    <div class="hero-body">
      <div class="container is-max-desktop">
        <div class="columns is-centered">
          <div class="column has-text-centered">
            <div class="column has-text-centered">
                <div class="is-size-5 publication-links">
                    <p>
                    🔗 <a href="https://arxiv.org/abs/2505.14604" target="_blank">arXiv</a> |
                    📄 <a href="https://arxiv.org/pdf/2505.14604" target="_blank">PDF</a> |
                    🌐 <a href="https://ccai-lab.github.io/SBT/" target="_blank">Project Page</a>
                    </p>
                </div>
                <div class="is-size-5 publication-authors">
              <div class="is-size-5 publication-authors">
                <span class="author-block">
                  <a href="mailto:ran159753@tju.edu.cn" target="_blank">Haoran Zhao</a><sup>1,2*</sup>,
                </span>
                <span class="author-block">
                  <a href="mailto:yanyuchen@zju.edu.cn" target="_blank">Yuchen Yan</a><sup>1*</sup>,
                </span>
                <span class="author-block">
                  <a href="mailto:syl@zju.edu.cn" target="_blank">Yongliang Shen</a><sup>1†</sup>,
                </span>
                <span class="author-block">
                  Haolei Xu<sup>1</sup>,
                </span>
                <span class="author-block">
                  Wenqi Zhang<sup>1</sup>,
                </span>
                <span class="author-block">
                  Kaitao Song<sup>3</sup>,
                </span>
                <span class="author-block">
                  Jian Shao<sup>1</sup>,
                </span>
                <span class="author-block">
                  Weiming Lu<sup>1</sup>,
                </span>
                <span class="author-block">
                  Jun Xiao<sup>1</sup>
                </span>
                <span class="author-block">
                  Yueting Zhuang<sup>1</sup>
                </span>
              </div>
                  <div class="is-size-5 publication-authors">
                    <span class="author-block"><sup>1</sup>Zhejiang University,</span>
                    <span class="author-block"><sup>2</sup>Tianjin University,</span>
                    <span class="author-block"><sup>3</sup>Microsoft Research Asia</span>
                    <br>
                    <span class="author-block">Preprint. Under review.</span>
                    <span class="eql-cntrb"><small><br><sup>*</sup>Equal Contribution, <sup>†</sup>Corresponding Author</small></span>
                  </div>
  </div>
</section>
<h3><h3>
</div>
<div align="center">
  <img src="figures/overview.png" width="100%" ></img>
  <br>
  <em>
      Overview of Self-Braking Tuning: Through a specialized data construction method and training strategy, our self-braking model is able to spontaneously halt overthinking.
  </em>
</div>
<br>

## 📝 About
Self-Braking Tuning is a novel framework that unlocks the potential of large reasoning models to autonomously identify and terminate redundant reasoning, enabling the models to regulate their own reasoning processes without relying on external control mechanisms. 
During fine-tuning, we use the Megatron-LM framework, with related parameters specified in [`configs/train.yaml`](configs/train.yaml); for evaluation, we employ the vLLM framework as the inference engine, with corresponding parameters located in [`configs/evaluation.yaml`](configs/evaluation.yaml).
Here, we provide a complete data construction framework that can be applied to nearly any long-chain tuning dataset, generating corresponding self-braking data accordingly.

## 🛠️ Preparation Steps Before Starting
In *Let LLMs Break Free from Overthinking via Self-Braking Tuning*, we performed self-braking tuning based on the OpenR1-Math dataset. In fact, this approach is applicable to any long-chain reasoning dataset, as long as the reasoning segments are wrapped with `<think>` and `</think>` tags.

Our method requires access to an LLM, and the recommended way to provide this is by setting:

```
export APIKEY=<your_key>
```
**Tip**: To provide a convenient default option, we use the OpenAI API key.  However, for large-scale datasets, it is recommended to deploy open-source models locally using vLLM or other frameworks, and to leverage efficient methods such as batch processing for better scalability and cost efficiency.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download

```bash
python models/model_download.py
python data/datasets/download_benchmarks.py
```

### 3. Get the baseline

```bash
python data/datasets/download_OpenR1-Math.py
```
### 4. Preprocess Data

```bash
python data/preprocessing/build_sbt-e.py
python data/preprocessing/build_sbt-d.py
```


### 5. Configure and Run Training / Evaluation

Refer to the config Settings in the following file:

* `train.yaml`: Training settings
* `evalution.yaml`: Evaluation settings



## 📄 License

This project is licensed under the [Apache 2.0 License](./LICENSE).


