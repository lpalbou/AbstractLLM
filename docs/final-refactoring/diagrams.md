# AbstractLLM Architecture Diagrams

## Level 1: High-Level Package Architecture

### 1.1 Three-Package Overview
```mermaid
graph TB
    subgraph "User Applications"
        APP[Application Code]
        CLI[ALMA CLI]
    end

    subgraph "AbstractAgent - Orchestration Layer"
        AGENT[Agent<br/>Orchestration]
        REACT[ReAct<br/>Reasoning]
        TOOLS_ADV[Advanced<br/>Tools]
        WORKFLOWS[Workflow<br/>Patterns]
    end

    subgraph "AbstractMemory - Knowledge Layer"
        MEMORY[Temporal<br/>Memory]
        KG[Knowledge<br/>Graph]
        COGNITIVE[Cognitive<br/>Enhancements]
        RETRIEVAL[Hybrid<br/>Retrieval]
    end

    subgraph "AbstractLLM - Core Platform"
        PROVIDERS[6 Provider<br/>Implementations]
        TOOLS[Universal Tool<br/>Abstraction]
        MEDIA[Media<br/>Processing]
        SESSION[Basic<br/>Session]
        EVENTS[Event<br/>System]
        TELEMETRY[Telemetry &<br/>Verbatim Capture]
    end

    subgraph "External Services"
        OPENAI[OpenAI API]
        ANTHROPIC[Anthropic API]
        OLLAMA[Ollama Local]
        LMSTUDIO[LM Studio]
        HF[HuggingFace]
        MLX[Apple MLX]
    end

    APP --> AGENT
    CLI --> AGENT
    AGENT --> MEMORY
    AGENT --> SESSION
    MEMORY --> RETRIEVAL
    SESSION --> PROVIDERS
    PROVIDERS --> OPENAI
    PROVIDERS --> ANTHROPIC
    PROVIDERS --> OLLAMA
    PROVIDERS --> LMSTUDIO
    PROVIDERS --> HF
    PROVIDERS --> MLX

    style AGENT fill:#e1f5e1
    style MEMORY fill:#ffe1e1
    style SESSION fill:#e1e1ff
```

### 1.2 Dependency Relationships
```mermaid
graph LR
    subgraph "Dependency Hierarchy"
        AA[AbstractAgent v1.0]
        AM[AbstractMemory v1.0]
        AL[AbstractLLM v2.0]

        AA --> AM
        AA --> AL
        AM --> AL
    end

    subgraph "Package Sizes"
        AA_SIZE[7,000 LOC]
        AM_SIZE[6,000 LOC]
        AL_SIZE[8,000 LOC]
    end

    AA --> AA_SIZE
    AM --> AM_SIZE
    AL --> AL_SIZE
```

## Level 2: Component Architecture

### 2.1 AbstractLLM Core Components
```mermaid
graph TB
    subgraph "AbstractLLM Core (8,000 LOC)"
        subgraph "Providers"
            BP[BaseProvider]
            OP[OpenAIProvider]
            AP[AnthropicProvider]
            OL[OllamaProvider]
            HFP[HuggingFaceProvider]
            MP[MLXProvider]
            LP[LMStudioProvider]
        end

        subgraph "Tool System"
            TD[ToolDefinition]
            TH[UniversalToolHandler]
            TP[ToolParser]
            TR[ToolRegistry]
        end

        subgraph "Media Layer"
            MED[MediaProcessor]
            IMG[ImageHandler]
            TXT[TextProcessor]
            TAB[TabularProcessor]
        end

        subgraph "Core Session"
            BS[BasicSession<br/>< 500 lines]
            MSG[Message]
            HIST[History]
        end

        subgraph "Infrastructure"
            EVT[EventBus]
            TEL[Telemetry]
            ARCH[ArchitectureDetector]
            CFG[ConfigManager]
        end

        BP --> OP
        BP --> AP
        BP --> OL
        BP --> HFP
        BP --> MP
        BP --> LP

        TH --> TD
        TH --> TP
        TH --> TR

        BS --> MSG
        BS --> HIST
        BS --> BP

        BP --> TH
        BP --> MED
        BP --> TEL
        BP --> EVT
    end
```

### 2.2 AbstractMemory Components
```mermaid
graph TB
    subgraph "AbstractMemory (6,000 LOC)"
        subgraph "Core Memory"
            TM[TemporalMemory]
            BT[Bi-Temporal<br/>Anchoring]
            RET[Retrieval<br/>Strategies]
        end

        subgraph "Memory Components"
            WM[WorkingMemory<br/>10-item window]
            EM[EpisodicMemory<br/>Events]
            SM[SemanticMemory<br/>Facts]
        end

        subgraph "Knowledge Graph"
            TKG[TemporalKnowledgeGraph]
            NODES[Nodes: Entity,<br/>Fact, Event]
            EDGES[Edges: Temporal,<br/>Causal, Semantic]
            ONT[Auto-Ontology]
        end

        subgraph "Cognitive Layer"
            FE[FactExtractor]
            SUM[Summarizer]
            VAL[ValueAlignment]
            INT[MemoryIntegration]
        end

        subgraph "Storage"
            SI[StorageInterface]
            FS[FileStorage]
            LD[LanceDBStorage]
            SER[Serialization]
        end

        TM --> BT
        TM --> WM
        TM --> EM
        TM --> SM
        TM --> TKG

        TKG --> NODES
        TKG --> EDGES
        TKG --> ONT

        FE --> SM
        SUM --> EM
        VAL --> SM
        INT --> TM

        TM --> SI
        SI --> FS
        SI --> LD
        SI --> SER
    end
```

### 2.3 AbstractAgent Components
```mermaid
graph TB
    subgraph "AbstractAgent (7,000 LOC)"
        subgraph "Core Agent"
            AG[Agent Class<br/>< 300 lines]
            COORD[Coordinator]
            CTX[ContextManager]
        end

        subgraph "Reasoning"
            REACT[ReActOrchestrator]
            TAO[Think-Act-Observe]
            SCRATCH[Scratchpad]
            PLAN[PlanExecute]
        end

        subgraph "Workflows"
            PIPE[Pipelines]
            BRANCH[Branches]
            LOOP[Loops]
            PAT[Patterns]
        end

        subgraph "Strategies"
            RETRY[RetryStrategy]
            STRUCT[StructuredOutput]
            VALID[Validation]
            FALL[Fallback]
        end

        subgraph "Advanced Tools"
            CODE[CodeIntelligence]
            WEB[WebTools]
            DATA[DataTools]
            CAT[ToolCatalog]
        end

        subgraph "CLI"
            ALMA[ALMA CLI]
            CMD[Commands]
            DISP[Display]
        end

        AG --> COORD
        AG --> CTX
        COORD --> REACT
        REACT --> TAO
        REACT --> SCRATCH
        AG --> RETRY
        AG --> STRUCT
        ALMA --> AG
        ALMA --> CMD
    end
```

## Level 3: Query Flow Architecture

### 3.1 Simple Query Flow (No Tools, No Memory)
```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Session
    participant Provider
    participant API

    User->>Agent: chat("What is 2+2?")
    Agent->>Session: add_message(user, "What is 2+2?")
    Agent->>Session: generate(prompt)
    Session->>Provider: format_for_provider()
    Provider->>API: POST /chat/completions
    API-->>Provider: "4"
    Provider-->>Session: GenerateResponse
    Session->>Session: add_message(assistant, "4")
    Session-->>Agent: "4"
    Agent-->>User: "4"
```

### 3.2 Query with Memory Context
```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Memory
    participant Session
    participant Provider

    User->>Agent: chat("What's my name?")
    Agent->>Memory: retrieve_context("name")
    Memory->>Memory: search_working_memory()
    Memory->>Memory: query_semantic_facts()
    Memory->>Memory: traverse_knowledge_graph()
    Memory-->>Agent: "User's name is Alice"
    Agent->>Session: generate(prompt, context)
    Session->>Provider: format_with_context()
    Provider-->>Session: "Your name is Alice"
    Session-->>Agent: "Your name is Alice"
    Agent->>Memory: add_interaction()
    Agent-->>User: "Your name is Alice"
```

### 3.3 Query with Tool Execution
```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Session
    participant Provider
    participant ToolHandler
    participant Tool

    User->>Agent: chat("Search for Python tutorials")
    Agent->>Session: generate_with_tools()
    Session->>Provider: generate(tools=[search])
    Provider-->>Session: ToolCallResponse
    Session->>EventBus: emit(BEFORE_TOOL_EXECUTION)
    Session->>ToolHandler: execute_tool_call()
    ToolHandler->>Tool: search("Python tutorials")
    Tool-->>ToolHandler: results
    ToolHandler-->>Session: tool_results
    Session->>EventBus: emit(AFTER_TOOL_EXECUTION)
    Session->>Provider: generate(with tool_results)
    Provider-->>Session: final_response
    Session-->>Agent: response
    Agent-->>User: "Here are Python tutorials..."
```

### 3.4 ReAct Reasoning Flow
```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant ReAct
    participant Memory
    participant Tools

    User->>Agent: chat("Solve complex problem", use_reasoning=True)
    Agent->>ReAct: execute(prompt)

    loop ReAct Cycle (max 5 iterations)
        ReAct->>Agent: think(current_state)
        Agent-->>ReAct: thought
        ReAct->>Agent: act(thought, available_tools)
        Agent-->>ReAct: action_decision

        alt Tool Needed
            ReAct->>Tools: execute(tool, args)
            Tools-->>ReAct: result
            ReAct->>Agent: observe(result)
            Agent-->>ReAct: observation
            ReAct->>Memory: store_observation()
        else Final Answer
            ReAct-->>Agent: final_response
        end
    end

    Agent-->>User: solution
```

## Level 4: Detailed Component Interactions

### 4.1 Provider Tool Handling Differences
```mermaid
graph TB
    subgraph "Tool Request from User"
        REQ[User Request:<br/>"Search for X"]
    end

    subgraph "AbstractLLM UniversalToolHandler"
        UTH[Universal<br/>Tool Handler]
        DETECT[Architecture<br/>Detection]
    end

    subgraph "Provider-Specific Formatting"
        subgraph "OpenAI"
            OAI_FMT["{type: 'function',<br/>function: {...}}"]
        end

        subgraph "Anthropic"
            ANT_FMT["&lt;tool_call&gt;<br/>...&lt;/tool_call&gt;"]
        end

        subgraph "Ollama (Qwen)"
            QWEN_FMT["&lt;|tool_call|&gt;<br/>...&lt;|tool_call|&gt;"]
        end

        subgraph "Ollama (Llama)"
            LLAMA_FMT["&lt;function_call&gt;<br/>...&lt;/function_call&gt;"]
        end
    end

    REQ --> UTH
    UTH --> DETECT

    DETECT -->|provider=openai| OAI_FMT
    DETECT -->|provider=anthropic| ANT_FMT
    DETECT -->|model=qwen| QWEN_FMT
    DETECT -->|model=llama| LLAMA_FMT
```

### 4.2 Memory Bi-Temporal Model
```mermaid
graph TB
    subgraph "Fact Addition"
        FACT["Alice visited Paris"]
        EVT_TIME[Event Time:<br/>2024-01-15 10:00]
        ING_TIME[Ingestion Time:<br/>2024-01-16 14:30]
    end

    subgraph "Temporal Knowledge Graph"
        subgraph "Time T1: Before Event"
            T1[No knowledge of visit]
        end

        subgraph "Time T2: Event Occurs"
            T2[Alice visits Paris<br/>(not yet known)]
        end

        subgraph "Time T3: Knowledge Added"
            T3[System learns:<br/>Alice visited Paris]
        end

        subgraph "Time T4: Query Time"
            T4_Q1[Query: "What happened on Jan 15?"<br/>→ Alice visited Paris]
            T4_Q2[Query: "What did we know on Jan 15?"<br/>→ Nothing about visit]
        end
    end

    FACT --> EVT_TIME
    FACT --> ING_TIME

    T1 -->|Time passes| T2
    T2 -->|Time passes| T3
    T3 -->|Time passes| T4_Q1
    T3 -->|Time passes| T4_Q2
```

### 4.3 Event System Architecture
```mermaid
graph LR
    subgraph "Event Producers"
        PROV[Providers]
        SESS[Session]
        TOOLS[Tools]
        MEM[Memory]
    end

    subgraph "Event Bus"
        BUS[EventBus]
        QUEUE[Event Queue]
        DISPATCH[Dispatcher]
    end

    subgraph "Event Types"
        REQ_EVT[RequestReceived]
        RESP_EVT[ResponseGenerated]
        TOOL_EVT[ToolExecuted]
        MEM_EVT[MemoryUpdated]
        ERR_EVT[ErrorOccurred]
    end

    subgraph "Event Consumers"
        TEL[Telemetry]
        LOG[Logging]
        MON[Monitoring]
        PLUG[Plugins]
    end

    PROV -->|emit| BUS
    SESS -->|emit| BUS
    TOOLS -->|emit| BUS
    MEM -->|emit| BUS

    BUS --> QUEUE
    QUEUE --> DISPATCH

    DISPATCH --> REQ_EVT
    DISPATCH --> RESP_EVT
    DISPATCH --> TOOL_EVT
    DISPATCH --> MEM_EVT
    DISPATCH --> ERR_EVT

    REQ_EVT --> TEL
    RESP_EVT --> LOG
    TOOL_EVT --> MON
    MEM_EVT --> PLUG
    ERR_EVT --> LOG
```

## Level 5: Migration Strategy

### 5.1 Compatibility Layer
```mermaid
graph TB
    subgraph "Existing Code"
        OLD["from abstractllm import Session<br/>session = Session(...)"]
    end

    subgraph "Compatibility Layer"
        COMPAT[CompatibilityLayer]
        WRAPPER[SessionCompat Wrapper]
        HOOKS[Import Hooks]
    end

    subgraph "New Architecture"
        NEW_AGENT[AbstractAgent.Agent]
        NEW_MEM[AbstractMemory.TemporalMemory]
        NEW_SESS[AbstractLLM.BasicSession]
    end

    OLD --> COMPAT
    COMPAT --> WRAPPER
    WRAPPER --> NEW_AGENT
    NEW_AGENT --> NEW_MEM
    NEW_AGENT --> NEW_SESS

    COMPAT --> HOOKS
    HOOKS -->|Redirect imports| NEW_AGENT

    style COMPAT fill:#ffffe0
```

### 5.2 Phased Migration Path
```mermaid
graph LR
    subgraph "Phase 1: Compatibility"
        P1[Install Compat Layer<br/>No code changes]
    end

    subgraph "Phase 2: New Features"
        P2[Use new packages<br/>for new code]
    end

    subgraph "Phase 3: Gradual Migration"
        P3[Migrate module<br/>by module]
    end

    subgraph "Phase 4: Cleanup"
        P4[Remove old<br/>dependencies]
    end

    P1 -->|Week 1| P2
    P2 -->|Week 2-3| P3
    P3 -->|Week 4| P4
```

## Level 6: Performance Architecture

### 6.1 Streaming Response Flow
```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Session
    participant Provider
    participant API

    User->>Agent: chat(stream=True)
    Agent->>Session: generate(stream=True)
    Session->>Provider: stream_generate()

    loop Streaming Chunks
        API-->>Provider: chunk
        Provider-->>Session: GenerateResponse(chunk)
        Session-->>Agent: yield chunk
        Agent-->>User: display chunk
    end

    Provider-->>Session: final usage stats
    Session-->>Agent: complete
    Agent-->>User: [Done]
```

### 6.2 Async Operation Flow
```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant AsyncSession
    participant AsyncProvider
    participant API

    User->>Agent: await chat_async()
    Agent->>AsyncSession: await generate_async()
    AsyncSession->>AsyncProvider: await agenerate()

    Note over AsyncProvider, API: Non-blocking I/O

    AsyncProvider->>API: async POST
    API-->>AsyncProvider: async response
    AsyncProvider-->>AsyncSession: GenerateResponse
    AsyncSession-->>Agent: response
    Agent-->>User: result
```

## Level 7: Storage Architecture

### 7.1 Memory Storage Options
```mermaid
graph TB
    subgraph "AbstractMemory Storage Layer"
        SI[StorageInterface]
    end

    subgraph "File Storage"
        JSON[JSON Files]
        PICKLE[Pickle Files]
        CUSTOM[Custom Format]
    end

    subgraph "LanceDB Storage"
        LANCE[LanceDB]
        EMB[Embeddings]
        SQL[SQL Queries]
        VEC[Vector Search]
    end

    subgraph "Future Options"
        REDIS[Redis]
        MONGO[MongoDB]
        PG[PostgreSQL]
    end

    SI --> JSON
    SI --> PICKLE
    SI --> CUSTOM
    SI --> LANCE
    SI -->|Future| REDIS
    SI -->|Future| MONGO
    SI -->|Future| PG

    LANCE --> EMB
    LANCE --> SQL
    LANCE --> VEC
```

## Level 8: Complete Query Lifecycle

### 8.1 End-to-End Query Processing
```mermaid
stateDiagram-v2
    [*] --> UserInput
    UserInput --> Agent: chat(prompt)

    Agent --> MemoryCheck: Retrieve Context
    MemoryCheck --> Reasoning: Use ReAct?
    MemoryCheck --> DirectGen: No Reasoning

    Reasoning --> Think
    Think --> Act
    Act --> ToolExec: Need Tool?
    Act --> Response: Have Answer
    ToolExec --> Observe
    Observe --> Think: Continue
    Observe --> Response: Complete

    DirectGen --> Session
    Session --> Provider
    Provider --> FormatAPI: Provider-specific
    FormatAPI --> APICall
    APICall --> ParseResponse
    ParseResponse --> UpdateMemory
    UpdateMemory --> ReturnUser

    Response --> Session
    ReturnUser --> [*]
```

## Key Architectural Principles

### 1. **Separation of Concerns**
- AbstractLLM: Provider abstraction only
- AbstractMemory: Knowledge persistence only
- AbstractAgent: Orchestration only

### 2. **Clean Interfaces**
- Each package has clear entry points
- No circular dependencies
- Well-defined APIs between packages

### 3. **Provider Abstraction**
- All provider complexity hidden in AbstractLLM
- Unified interface for tools, media, streaming
- Architecture detection for model-specific handling

### 4. **Temporal Knowledge**
- Bi-temporal model for accurate history
- Hybrid retrieval for best results
- Auto-ontology for relationship discovery

### 5. **Extensibility**
- Event system for plugins
- Storage interface for backends
- Tool registry for capabilities

### 6. **Performance**
- Streaming support throughout
- Async operations where needed
- Efficient memory retrieval (<100ms)

### 7. **Backward Compatibility**
- Compatibility layer for migration
- Import hooks for old paths
- Gradual migration support

## Validation: Query Flow After Refactoring

The refactored architecture maintains all current capabilities while improving:

1. **Maintainability**: No more 4,099-line God classes
2. **Performance**: Clean separation enables optimization
3. **Extensibility**: Event system and clear interfaces
4. **Testability**: Each package can be tested independently
5. **Scalability**: Can evolve packages independently

The query flow remains smooth:
- User → Agent → Memory + Session → Provider → Response
- Tools integrate naturally through UniversalToolHandler
- Memory provides context without coupling
- ReAct reasoning orchestrates complex queries

This architecture successfully achieves our goals of:
- Lightweight unified LLM interface (AbstractLLM)
- Advanced memory system (AbstractMemory)
- Autonomous agent capabilities (AbstractAgent)

All while maintaining the same functionality as the current `abstractllm/cli.py` implementation.