# Data Flow in AbstractLLM

This document describes how data flows through the AbstractLLM system, from input to output.

## System Overview

```mermaid
graph TD
    A[User Input] --> B[Input Processing]
    B --> C[Media Factory]
    C --> D[Media Processing]
    D --> E[Provider Layer]
    E --> F[Pipeline Factory]
    F --> G[Model Pipeline]
    G --> H[Output Processing]
    H --> I[User Output]
    
    J[Error Handling] --> B
    J --> C
    J --> D
    J --> E
    J --> F
    J --> G
    J --> H
```

## Core Components Flow

### 1. Input Processing
```mermaid
graph LR
    A[Raw Input] --> B[Input Validation]
    B --> C[Type Detection]
    C --> D[Media Factory]
    D --> E[Provider Selection]
```

### 2. Media Processing
```mermaid
graph TD
    A[Media Factory] --> B[Media Type]
    B --> C[ImageInput]
    B --> D[TextInput]
    B --> E[TabularInput]
    C --> F[Provider Format]
    D --> F
    E --> F
    F --> G[Pipeline Input]
```

### 3. Pipeline Processing
```mermaid
graph TD
    A[Pipeline Factory] --> B[Architecture Detection]
    B --> C[Pipeline Selection]
    C --> D[Model Loading]
    D --> E[Input Processing]
    E --> F[Model Inference]
    F --> G[Output Formatting]
```

### 4. Error Flow
```mermaid
graph TD
    A[Error Detection] --> B[Error Classification]
    B --> C[Error Handling]
    C --> D[Resource Cleanup]
    D --> E[User Feedback]
```

### Visual Question Answering Flow
```mermaid
graph TD
    A[User Input] --> B[Image Input]
    A --> C[Question Input]
    B --> D[Media Processing]
    C --> D
    D --> E[VQA Pipeline]
    E --> F[Model Architecture]
    F --> G[ViLT Model]
    F --> H[BLIP Model]
    G --> I[Answer Generation]
    H --> I
    I --> J[Confidence Scoring]
    J --> K[User Output]
```

The VQA pipeline handles:
1. **Input Processing**:
   - Image validation and preprocessing
   - Question text formatting
   - Multiple choice answer preparation
2. **Model Selection**:
   - ViLT for structured VQA
   - BLIP for open-ended questions
3. **Answer Generation**:
   - Multiple choice answer selection
   - Free-form answer generation
   - Confidence score calculation
4. **Output Formatting**:
   - Answer text
   - Confidence scores
   - Optional model logits

## Component Interactions

### Media System to Pipeline
```