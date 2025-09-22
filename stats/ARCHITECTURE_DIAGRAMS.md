# RAG-Redis System Architecture Diagrams

## 1. Current Architecture (BEFORE)
```mermaid
graph TB
    subgraph "Multiple Entry Points (Redundant)"
        PY1[Python MCP Wrapper]
        PY2[Python Integration Layer]
        RUST1[Rust MCP Server]
        FFI[FFI Interface]
    end

    subgraph "Monolithic Core (Violates SRP)"
        RAG[RagSystem<br/>- Document Processing<br/>- Vector Search<br/>- Memory Management<br/>- Research<br/>- Metrics<br/>- MCP Handling]
    end

    subgraph "Tightly Coupled Infrastructure"
        REDIS[Redis<br/>Direct Dependency]
        VECTOR[Vector Store<br/>Redis-Specific]
        MEM[Memory Manager<br/>Circular Deps]
    end

    PY1 --> RAG
    PY2 --> RAG
    RUST1 --> RAG
    FFI --> RAG

    RAG --> REDIS
    RAG --> VECTOR
    RAG --> MEM

    VECTOR --> REDIS
    MEM --> REDIS
    REDIS --> MEM

    style RAG fill:#f99,stroke:#333,stroke-width:4px
    style REDIS fill:#faa,stroke:#333,stroke-width:2px
```

## 2. Proposed Clean Architecture (AFTER)
```mermaid
graph TB
    subgraph "Presentation Layer (Adapters)"
        MCP[MCP Server<br/>Single Implementation]
        REST[REST API<br/>Optional]
        FFI[FFI Bridge<br/>Optional]
    end

    subgraph "Application Layer (Use Cases)"
        DS[Document Service]
        SS[Search Service]
        MS[Memory Service]
        RS[Research Service]
    end

    subgraph "Domain Layer (Core Business)"
        DOC[Document Entity]
        CHUNK[Chunk Entity]
        MEM_E[Memory Entity]
        REPO[Repository Interfaces]
    end

    subgraph "Infrastructure Layer (Implementations)"
        REDIS_I[Redis Repository]
        VECTOR_I[HNSW Vector Store]
        EMBED_I[Embedding Provider]
    end

    MCP --> DS
    MCP --> SS
    MCP --> MS

    DS --> DOC
    DS --> REPO
    SS --> CHUNK
    SS --> REPO
    MS --> MEM_E
    MS --> REPO

    REPO -.-> REDIS_I
    REPO -.-> VECTOR_I
    REPO -.-> EMBED_I

    style DS fill:#9f9,stroke:#333,stroke-width:2px
    style SS fill:#9f9,stroke:#333,stroke-width:2px
    style MS fill:#9f9,stroke:#333,stroke-width:2px
    style REPO fill:#99f,stroke:#333,stroke-width:2px
```

## 3. Dependency Flow Comparison

### Before (Circular Dependencies)
```mermaid
graph LR
    A[redis_backend] -->|uses| B[memory]
    B -->|uses| A
    C[vector_store] -->|uses| A
    A -->|knows about| C
    D[RagSystem] -->|depends on all| A
    D -->|depends on all| B
    D -->|depends on all| C

    style A fill:#f99
    style B fill:#f99
    style C fill:#f99
```

### After (Clean Dependencies)
```mermaid
graph TD
    subgraph "Dependency Flow"
        UI[User Interface]
        APP[Application Services]
        DOM[Domain Entities]
        ABS[Abstractions/Interfaces]
        IMPL[Implementations]
    end

    UI --> APP
    APP --> DOM
    APP --> ABS
    IMPL -.-> ABS

    style ABS fill:#99f
    style DOM fill:#9f9
```

## 4. Data Flow Architecture

### Document Ingestion Flow
```mermaid
sequenceDiagram
    participant Client
    participant MCP as MCP Server
    participant DS as Document Service
    participant DP as Document Pipeline
    participant ER as Embedding Service
    participant VR as Vector Repository
    participant DR as Document Repository

    Client->>MCP: Ingest Request
    MCP->>DS: ingest(content, metadata)
    DS->>DP: process(content)
    DP->>DP: Parse & Chunk
    DS->>ER: embed_batch(chunks)
    ER-->>DS: embeddings
    DS->>VR: index(vectors)
    DS->>DR: save(document)
    DS-->>MCP: document_id
    MCP-->>Client: Success Response
```

### Search Flow
```mermaid
sequenceDiagram
    participant Client
    participant MCP as MCP Server
    participant SS as Search Service
    participant Cache
    participant ER as Embedding Service
    participant VR as Vector Repository
    participant DR as Document Repository

    Client->>MCP: Search Request
    MCP->>SS: search(query)
    SS->>Cache: check(query_hash)
    alt Cache Hit
        Cache-->>SS: cached_results
    else Cache Miss
        SS->>ER: embed(query)
        ER-->>SS: query_vector
        SS->>VR: search(vector, limit)
        VR-->>SS: vector_results
        SS->>DR: fetch_documents(ids)
        DR-->>SS: documents
        SS->>Cache: store(results)
    end
    SS-->>MCP: search_results
    MCP-->>Client: Response
```

## 5. Module Organization

### Before (Mixed Concerns)
```
rag-redis-system/
├── src/
│   ├── lib.rs                 ❌ Everything mixed
│   ├── memory_profiler.rs     ❌ Observability in core
│   ├── memory_dashboard.rs    ❌ UI in core
│   ├── vector_store.rs        ❌ Infrastructure details
│   └── redis_backend.rs       ❌ Direct coupling
```

### After (Clean Separation)
```
rag-redis-system/
├── core/                      ✓ Pure domain logic
│   ├── domain/
│   │   ├── document.rs
│   │   ├── chunk.rs
│   │   └── memory.rs
│   ├── repositories/          ✓ Abstractions only
│   │   └── mod.rs
│   └── services/              ✓ Use cases
│       ├── document_service.rs
│       └── search_service.rs
├── infrastructure/            ✓ Implementation details
│   ├── redis/
│   ├── embedding/
│   └── storage/
├── interfaces/                ✓ External interfaces
│   └── mcp/
└── observability/            ✓ Separate crate
    ├── profiler/
    └── dashboard/
```

## 6. Service Interaction Patterns

```mermaid
graph TB
    subgraph "Service Layer (Stateless)"
        DS[Document Service]
        SS[Search Service]
        MS[Memory Service]
    end

    subgraph "Repository Layer (Abstractions)"
        DR[Document Repo Trait]
        VR[Vector Repo Trait]
        MR[Memory Repo Trait]
    end

    subgraph "Infrastructure (Implementations)"
        RDR[Redis Doc Repo]
        HVR[HNSW Vector Repo]
        RMR[Redis Mem Repo]
    end

    DS --> DR
    SS --> VR
    MS --> MR

    RDR -.implements.-> DR
    HVR -.implements.-> VR
    RMR -.implements.-> MR

    style DR fill:#99f,stroke:#333,stroke-width:2px
    style VR fill:#99f,stroke:#333,stroke-width:2px
    style MR fill:#99f,stroke:#333,stroke-width:2px
```

## 7. Deployment Architecture

### Containerized Deployment
```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "RAG Pods (3 replicas)"
            P1[RAG Pod 1]
            P2[RAG Pod 2]
            P3[RAG Pod 3]
        end

        subgraph "Redis Cluster"
            RM[Redis Master]
            RS1[Redis Slave 1]
            RS2[Redis Slave 2]
        end

        subgraph "Services"
            LB[Load Balancer]
            ING[Ingress]
        end
    end

    ING --> LB
    LB --> P1
    LB --> P2
    LB --> P3

    P1 --> RM
    P2 --> RM
    P3 --> RM

    RM --> RS1
    RM --> RS2
```

## 8. Error Handling Flow
```mermaid
flowchart TD
    A[Request] --> B{Validation}
    B -->|Invalid| C[Return 400 Bad Request]
    B -->|Valid| D[Process Request]

    D --> E{Repository Operation}
    E -->|Success| F[Return Success]
    E -->|Not Found| G[Return 404]
    E -->|Connection Error| H[Retry with Backoff]

    H --> I{Retry Count}
    I -->|< Max| E
    I -->|>= Max| J[Circuit Breaker]

    J --> K[Fallback Strategy]
    K --> L[Return Degraded Response]
```

## 9. Memory Management Architecture
```mermaid
graph TB
    subgraph "Memory Tiers"
        L1[L1 Cache<br/>In-Process<br/>100MB]
        L2[L2 Cache<br/>Redis Local<br/>1GB]
        L3[L3 Storage<br/>Redis Cluster<br/>Unlimited]
    end

    subgraph "Access Times"
        T1[~1μs]
        T2[~100μs]
        T3[~10ms]
    end

    Request --> L1
    L1 -->|Miss| L2
    L2 -->|Miss| L3

    L1 -.-> T1
    L2 -.-> T2
    L3 -.-> T3

    style L1 fill:#9f9
    style L2 fill:#ff9
    style L3 fill:#f99
```

## 10. Security Architecture
```mermaid
graph TB
    subgraph "Security Layers"
        AUTH[Authentication<br/>JWT/OAuth2]
        AUTHZ[Authorization<br/>RBAC]
        VAL[Input Validation<br/>Schema Validation]
        ENC[Encryption<br/>TLS 1.3]
        RATE[Rate Limiting<br/>Token Bucket]
    end

    Client --> ENC
    ENC --> AUTH
    AUTH --> AUTHZ
    AUTHZ --> RATE
    RATE --> VAL
    VAL --> Application

    style AUTH fill:#99f
    style ENC fill:#9f9
```

## Summary

The architectural diagrams illustrate the transformation from a monolithic, tightly-coupled system with multiple redundant entry points to a clean, layered architecture following SOLID principles and hexagonal architecture patterns. Key improvements include:

1. **Single entry point** through consolidated MCP server
2. **Clear separation of concerns** with focused services
3. **Dependency inversion** through repository abstractions
4. **Elimination of circular dependencies**
5. **Proper layering** with unidirectional dependencies
6. **Testability** through interface-based design
7. **Scalability** through stateless services
8. **Maintainability** through modular structure
