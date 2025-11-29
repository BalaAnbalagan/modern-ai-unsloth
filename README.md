# Modern AI with Unsloth.ai – CMPE-255

**Author**: Balamuralikrishnan Anbalagan
**Course**: CMPE-255 Data Mining
**Assignment**: Modern AI with Unsloth.ai

---

## 📋 Overview

This repository contains five comprehensive Google Colab notebooks demonstrating different fine-tuning approaches using **Unsloth.ai**, a high-performance library for efficient LLM training. Each notebook showcases a distinct training methodology with practical implementations and detailed analysis.

---

## 📚 Notebooks

| # | Notebook | Objective | Key Technique |
|---|----------|-----------|---------------|
| 1 | [colab1_full_finetune.ipynb](colab1_full_finetune.ipynb) | Full LLM fine-tuning on code generation | High-rank LoRA (r=256) with lm_head + embeddings |
| 2 | [colab2_lora_finetune.ipynb](colab2_lora_finetune.ipynb) | Parameter-efficient LoRA fine-tuning | Low-rank LoRA (r=8, alpha=16) |
| 3 | [colab3_rlhf.ipynb](colab3_rlhf.ipynb) | Preference-based alignment | Direct Preference Optimization (DPO) |
| 4 | [colab4_grpo_reasoning.ipynb](colab4_grpo_reasoning.ipynb) | Reasoning improvement with RL | GRPO on math problems (GSM8K) |
| 5 | [colab5_continued_pretrain.ipynb](colab5_continued_pretrain.ipynb) | Language/domain adaptation | Continued pre-training on Tamil |

---

## 🎯 Learning Objectives

Each notebook demonstrates:
- ✅ **Installation & Setup**: Unsloth.ai configuration for Google Colab
- ✅ **Model Loading**: Efficient 4-bit quantization for memory savings
- ✅ **LoRA Configuration**: Rank selection and target module optimization
- ✅ **Training Pipeline**: End-to-end training with monitoring
- ✅ **Evaluation**: Metrics analysis and result visualization
- ✅ **Checkpoint Management**: Model saving and deployment strategies

---

## 🔬 Technical Details

### Base Model
- **Model**: SmolLM2-135M ([unsloth/smollm2-135m](https://huggingface.co/unsloth/smollm2-135m))
- **Parameters**: 135 million
- **Quantization**: 4-bit (70% VRAM reduction)
- **Hardware**: Google Colab T4 GPU (12GB)

### Training Optimizations
- **Speed**: 2x faster than standard HuggingFace Transformers
- **Memory**: 60-80% reduction with QLoRA
- **Gradient Checkpointing**: Enabled via Unsloth
- **Mixed Precision**: BF16/FP16 based on GPU support

### Datasets Used
1. **CodeParrot Clean**: 1000 Python code samples
2. **Anthropic HH-RLHF**: 1000 preference pairs
3. **GSM8K**: 500 math reasoning problems
4. **OSCAR Tamil**: 5000 Tamil text samples

---

## 📊 Results Summary

| Notebook | Training Method | Trainable % | Key Metric | Result |
|----------|----------------|-------------|------------|--------|
| 1 | Full Fine-tuning | ~5-10% | Loss reduction | Effective code generation |
| 2 | LoRA (r=8) | <1% | Memory savings | 10-20x fewer parameters |
| 3 | DPO | ~2% | Preference accuracy | Aligned responses |
| 4 | GRPO Reasoning | ~1% | Math accuracy | Improved reasoning |
| 5 | Continued Pre-training | ~3% | Tokenization efficiency | Tamil adaptation |

---

## 🚀 Quick Start

### Prerequisites
- Google account (for Colab)
- Basic Python knowledge
- Understanding of LLMs and fine-tuning concepts

### Running the Notebooks

1. **Open in Google Colab**:
   - Click the Colab link at the top of each notebook
   - Or upload the `.ipynb` file to your Google Drive

2. **Select GPU Runtime**:
   ```
   Runtime → Change runtime type → Hardware accelerator: T4 GPU
   ```

3. **Run All Cells**:
   - Click `Runtime → Run all`
   - Or execute cells sequentially with `Shift+Enter`

4. **Monitor Progress**:
   - Training logs display in real-time
   - Loss curves and metrics visualize automatically
   - Checkpoints save to `./checkpoints/colabX/`

### Installation (Automated in Notebooks)
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

---

## 📖 Detailed Descriptions

### Notebook 1: Full Fine-tuning
**Objective**: Train all model parameters using high-rank LoRA

📺 **Video Tutorial**: [Full Fine-tuning with Unsloth.ai](https://youtu.be/8L875dy9nfk)

- **Dataset**: CodeParrot (Python code)
- **LoRA Config**: Rank=256, includes lm_head + embed_tokens
- **Use Case**: Domain adaptation, learning new capabilities
- **Output**: Code generation with loss visualization

**Key Insight**: High-rank LoRA approaches full fine-tuning effectiveness while remaining memory-efficient.

---

### Notebook 2: LoRA Fine-tuning
**Objective**: Demonstrate parameter-efficient training

📺 **Video Tutorial**: [LoRA Fine-tuning with Unsloth.ai](https://youtu.be/8L875dy9nfk)

- **Dataset**: Same as Notebook 1 (for comparison)
- **LoRA Config**: Rank=8, alpha=16 (attention layers only)
- **Comparison**: 10-20x fewer trainable parameters
- **Output**: Efficiency metrics vs full fine-tuning

**Key Insight**: Low-rank LoRA achieves comparable performance with drastically reduced memory and storage.

---

### Notebook 3: RLHF with DPO
**Objective**: Align model with human preferences

📺 **Video Tutorial**: [Direct Preference Optimization with Unsloth.ai](https://youtu.be/zMBV7hoo_KU)

- **Dataset**: Anthropic HH-RLHF (human feedback)
- **Method**: Direct Preference Optimization (simpler than traditional RLHF)
- **Training**: Chosen vs rejected response pairs
- **Output**: Preference-aligned conversational responses

**Key Insight**: DPO eliminates the need for separate reward model training, simplifying alignment.

---

### Notebook 4: GRPO Reasoning
**Objective**: Improve multi-step reasoning abilities

📺 **Video Tutorial**: [Group Relative Policy Optimization with Unsloth.ai](https://youtu.be/srQHKjLCnAo)

- **Dataset**: GSM8K (grade school math problems)
- **Method**: Group Relative Policy Optimization
- **Evaluation**: Before/after accuracy comparison
- **Output**: Structured reasoning with `<reasoning>` and `<answer>` tags

**Key Insight**: GRPO training significantly improves reasoning quality with custom reward functions.

---

### Notebook 5: Continued Pre-training
**Objective**: Adapt model to new language (Tamil)

- **Dataset**: OSCAR Tamil corpus
- **Method**: High-rank LoRA including embeddings
- **Analysis**: Tokenization efficiency comparison
- **Output**: Tamil text generation + knowledge retention test

**Key Insight**: Including `embed_tokens` in LoRA enables effective language adaptation while preserving original capabilities.

---

## 🛠️ Technical Architecture

### LoRA (Low-Rank Adaptation)
```
Original Weight: W₀ ∈ ℝᵈˣᵏ
LoRA Update: ΔW = BA (B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵏ)
Forward Pass: h = W₀x + ΔWx = W₀x + BAx
```

**Benefits**:
- Trainable parameters: `r(d+k)` instead of `d×k`
- For r=8, d=k=4096: 99.6% parameter reduction
- Adapters can be swapped for multi-task deployment

### Training Pipeline
```
1. Load base model (4-bit quantization)
2. Apply LoRA adapters
3. Prepare dataset
4. Configure training arguments
5. Train with SFTTrainer/DPOTrainer
6. Evaluate and save checkpoints
```

---

## 📈 Performance Metrics

### Speed Improvements
- **Baseline**: Standard Transformers training
- **Unsloth**: 2x faster training time
- **Reason**: Optimized kernels, efficient attention, gradient checkpointing

### Memory Savings
- **4-bit Quantization**: 70% VRAM reduction
- **LoRA (r=8)**: 99%+ parameter reduction
- **Result**: Train 7B models on consumer GPUs

### Accuracy
- **LoRA vs Full FT**: 95-100% of full fine-tuning performance
- **DPO Alignment**: Significant preference accuracy improvement
- **GRPO Reasoning**: Measurable accuracy gains on GSM8K

---

## 💾 Checkpoint Structure

Each notebook saves checkpoints in organized directories:

```
checkpoints/
├── colab1/
│   ├── lora_adapter/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── merged_16bit/
├── colab2/
│   └── lora_adapter/
├── colab3/
│   └── dpo_adapter/
├── colab4/
│   └── grpo_adapter/
└── colab5/
    └── tamil_adapter/
```

**Adapter Size**: 10-100x smaller than full model checkpoints

---

## 🔗 Resources

### Official Documentation
- [Unsloth Documentation](https://docs.unsloth.ai)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Model Hub](https://huggingface.co/unsloth)

### Key Papers
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- DPO: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)
- GRPO: Group Relative Policy Optimization (TRL)

### Datasets
- [CodeParrot Clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean)
- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- [OSCAR-2201](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201)

---

## 📝 Assignment Submission

### Requirements Met
✅ Five distinct fine-tuning approaches
✅ Google Colab compatible notebooks
✅ End-to-end runnable code
✅ Checkpoint saving demonstrated
✅ Results visualization included
✅ Comprehensive documentation
✅ No API keys or secrets committed
✅ Clean repository structure

### Submission Format
- **GitHub Repository**: All notebooks + README
- **Canvas Submission**: Repository URL
- **Video Demos**: Colab/YouTube links in each notebook

---

## 🎓 Learning Outcomes

After completing these notebooks, you will understand:

1. **Full vs Parameter-Efficient Fine-tuning**
   - Trade-offs between performance and efficiency
   - When to use high-rank vs low-rank LoRA

2. **Preference-Based Training**
   - DPO simplifies RLHF pipeline
   - Aligning models with human preferences

3. **Reasoning Training**
   - GRPO improves multi-step reasoning
   - Custom reward functions for task-specific goals

4. **Domain/Language Adaptation**
   - Continued pre-training strategies
   - Embedding adaptation for new languages

5. **Practical Optimization**
   - 4-bit quantization for memory efficiency
   - Gradient checkpointing and mixed precision

---

## 🚧 Troubleshooting

### Common Issues

**Issue**: Out of memory error
**Solution**: Reduce batch size or enable gradient accumulation

**Issue**: Slow training
**Solution**: Verify GPU is enabled (Runtime → Change runtime type)

**Issue**: Import errors
**Solution**: Restart runtime and re-run installation cell

**Issue**: Dataset loading timeout
**Solution**: Use streaming=True or smaller subset

---

## 🤝 Contributing

This is an educational project for CMPE-255. For questions or improvements:

1. Open an issue on GitHub
2. Submit a pull request with detailed description
3. Follow existing code style and documentation format

---

## 📄 License

This project is created for educational purposes as part of CMPE-255 coursework.

- Code: MIT License
- Notebooks: CC BY 4.0
- Models: Subject to original model licenses

---

## 🙏 Acknowledgments

- **Unsloth.ai Team**: For the amazing optimization library
- **HuggingFace**: For model hosting and datasets
- **Google Colab**: For free GPU access
- **CMPE-255**: For the learning opportunity

---

## 📞 Contact

**Author**: Balamuralikrishnan Anbalagan
**Course**: CMPE-255 Data Mining
**Institution**: San Jose State University

---

## 🎯 Next Steps

To extend this work:

1. **Scale Up**: Try larger models (1.7B, 3B parameters)
2. **New Domains**: Adapt to medical, legal, or scientific text
3. **Multi-Task**: Train multiple adapters for different tasks
4. **Deployment**: Export to GGUF for inference optimization
5. **Evaluation**: Add comprehensive benchmarking suite

---

**🎉 All five Unsloth.ai training approaches successfully implemented!**

*Happy Training! 🚀*
