# System Architecture & Pipeline

This document provides a visual and technical overview of the **Unified Trust Metric System**.

---

## 1. System Pipeline
The pipeline handles everything from data ingestion to the final trust score calculation.

```mermaid
graph LR
    subgraph "Data Layer"
    A[Raw Dataset (.xlsx)] --> B[Data Loader]
    B --> C{Synthetic Mutation?}
    C -->|Yes| D[Code/Summary Mutators]
    C -->|No| E[Clean Samples]
    end

    subgraph "Feature Engineering"
    D & E --> F[AST Entity Extraction]
    D & E --> G[Tokenization]
    D & E --> H[Dry-run Execution]
    end

    subgraph "Model Processing"
    G --> I[1D CNN Branch]
    F & H --> J[MLP Branch]
    I & J --> K[Fusion Layer]
    end

    subgraph "Output & Metric"
    K --> L[Hallucination Heads]
    L --> M[P_code & P_summ]
    M --> N[Trust Score Formula]
    end
```

---

## 2. Model Architecture
The `HybridTrustModel` combines structural code analysis with semantic execution features.

```mermaid
graph TD
    subgraph "Input Source: Multi-Modal"
    TokenIn["Code Sequence (Shape: 256, 14)"]
    FeatIn["Scalar Features (Shape: 18)"]
    end

    subgraph "CNN Branch (Code Analysis)"
    TokenIn --> C3["Conv1D (k=3, f=32)"]
    TokenIn --> C5["Conv1D (k=5, f=32)"]
    TokenIn --> C7["Conv1D (k=7, f=32)"]
    C3 & C5 & C7 --> MP[Global Max Pooling]
    MP --> CNN_Out[Flat Vector: 96]
    end

    subgraph "MLP Branch (Feature Analysis)"
    FeatIn --> L1[Linear 64]
    L1 --> BN[BatchNorm]
    BN --> DR[Dropout 0.3]
    DR --> L2[Linear 32]
    L2 --> MLP_Out[Flat Vector: 32]
    end

    CNN_Out & MLP_Out --> Fusion[Concatenation: 128]
    
    subgraph "Fusion & Heads"
    Fusion --> F1[Linear 64]
    F1 --> F2[Linear 32]
    F2 --> Head1[Code Head: 1]
    F2 --> Head2[Summary Head: 1]
    end

    Head1 --> Out1[Logit: P_code]
    Head2 --> Out2[Logit: P_summ]
```

---

## 3. High-Level Logic Flow

### Data Augmentation
To train the model on hallucinations, we perform synthetic mutations in [data_loading.py](file:///c:/Unified%20Trust%20Metric%20System/data_loading.py):
- **Code Mutation**: Randomly selects between arg-swapping, operator mutation (+ to -), or line deletion.
- **Summary Mutation**: Injects extrinsic entities or contradicts code logic.

### Trust Score Formula
The final metric is computed as:
$$Trust = w_1(1 - P_{code}) + w_2(1 - P_{summary}) + w_3(S_{api})$$

| Component | Weight | Meaning |
| :--- | :--- | :--- |
| $w_1(1 - P_{code})$ | 0.4 | Reward for code that is likely faithful. |
| $w_2(1 - P_{summary})$ | 0.4 | Reward for summaries that reflect code logic. |
| $w_3(S_{api})$ | 0.2 | Reward for correct structural API usage. |
