use crate::protocol::*;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Create all available MCP tools for the RAG-Redis system
pub fn create_tools() -> Vec<Tool> {
    vec![
        create_ingest_document_tool(),
        create_search_documents_tool(),
        create_research_query_tool(),
        create_list_documents_tool(),
        create_get_document_tool(),
        create_delete_document_tool(),
        create_clear_memory_tool(),
        create_get_memory_stats_tool(),
        create_get_system_metrics_tool(),
        create_health_check_tool(),
        create_configure_system_tool(),
        create_batch_ingest_tool(),
        create_semantic_search_tool(),
        create_hybrid_search_tool(),
    ]
}

fn create_ingest_document_tool() -> Tool {
    let mut properties = HashMap::new();

    properties.insert("content".to_string(), PropertySchema {
        property_type: "string".to_string(),
        description: "The text content of the document to ingest".to_string(),
        items: None,
        default: None,
        minimum: None,
        maximum: None,
    });

    properties.insert("metadata".to_string(), PropertySchema {
        property_type: "object".to_string(),
        description: "Optional metadata for the document (title, author, source, tags, etc.)".to_string(),
        items: None,
        default: Some(json!({})),
        minimum: None,
        maximum: None,
    });

    properties.insert("document_id".to_string(), PropertySchema {
        property_type: "string".to_string(),
        description: "Optional custom document ID. If not provided, a UUID will be generated".to_string(),
        items: None,
        default: None,
        minimum: None,
        maximum: None,
    });

    Tool {
        name: "ingest_document".to_string(),
        description: "Ingest a document into the RAG system for vector search and retrieval".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["content".to_string()],
        },
    }
}

fn create_search_documents_tool() -> Tool {
    let mut properties = HashMap::new();

    properties.insert("query".to_string(), PropertySchema {
        property_type: "string".to_string(),
        description: "The search query to find relevant documents".to_string(),
        items: None,
        default: None,
        minimum: None,
        maximum: None,
    });

    properties.insert("limit".to_string(), PropertySchema {
        property_type: "integer".to_string(),
        description: "Maximum number of results to return".to_string(),
        items: None,
        default: Some(json!(10)),
        minimum: Some(1),
        maximum: Some(100),
    });

    properties.insert("min_score".to_string(), PropertySchema {
        property_type: "number".to_string(),
        description: "Minimum similarity score threshold (0.0-1.0)".to_string(),
        items: None,
        default: Some(json!(0.0)),
        minimum: None,
        maximum: None,
    });

    Tool {
        name: "search_documents".to_string(),
        description: "Search for documents using vector similarity matching".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["query".to_string()],
        },
    }
}

fn create_research_query_tool() -> Tool {
    let mut properties = HashMap::new();

    properties.insert("query".to_string(), PropertySchema {
        property_type: "string".to_string(),
        description: "The research query to perform both local and web search".to_string(),
        items: None,
        default: None,
        minimum: None,
        maximum: None,
    });

    properties.insert("sources".to_string(), PropertySchema {
        property_type: "array".to_string(),
        description: "Optional list of web sources to search (URLs or domains)".to_string(),
        items: Some(Box::new(PropertySchema {
            property_type: "string".to_string(),
            description: "Web source URL or domain".to_string(),
            items: None,
            default: None,
            minimum: None,
            maximum: None,
        })),
        default: Some(json!([])),
        minimum: None,
        maximum: None,
    });

    properties.insert("local_only".to_string(), PropertySchema {
        property_type: "boolean".to_string(),
        description: "If true, only search local documents without web search".to_string(),
        items: None,
        default: Some(json!(false)),
        minimum: None,
        maximum: None,
    });

    Tool {
        name: "research_query".to_string(),
        description: "Perform comprehensive research combining local document search with web search".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["query".to_string()],
        },
    }
}

fn create_list_documents_tool() -> Tool {
    let mut properties = HashMap::new();

    properties.insert("limit".to_string(), PropertySchema {
        property_type: "integer".to_string(),
        description: "Maximum number of documents to list".to_string(),
        items: None,
        default: Some(json!(50)),
        minimum: Some(1),
        maximum: Some(1000),
    });

    properties.insert("offset".to_string(), PropertySchema {
        property_type: "integer".to_string(),
        description: "Number of documents to skip for pagination".to_string(),
        items: None,
        default: Some(json!(0)),
        minimum: Some(0),
        maximum: None,
    });

    properties.insert("filter".to_string(), PropertySchema {
        property_type: "object".to_string(),
        description: "Optional filter criteria for document metadata".to_string(),
        items: None,
        default: Some(json!({})),
        minimum: None,
        maximum: None,
    });

    Tool {
        name: "list_documents".to_string(),
        description: "List stored documents with optional filtering and pagination".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec![],
        },
    }
}

fn create_get_document_tool() -> Tool {
    let mut properties = HashMap::new();

    properties.insert("document_id".to_string(), PropertySchema {
        property_type: "string".to_string(),
        description: "The ID of the document to retrieve".to_string(),
        items: None,
        default: None,
        minimum: None,
        maximum: None,
    });

    properties.insert("include_chunks".to_string(), PropertySchema {
        property_type: "boolean".to_string(),
        description: "Whether to include document chunks in the response".to_string(),
        items: None,
        default: Some(json!(false)),
        minimum: None,
        maximum: None,
    });

    Tool {
        name: "get_document".to_string(),
        description: "Retrieve a specific document by ID with optional chunk data".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["document_id".to_string()],
        },
    }
}

fn create_delete_document_tool() -> Tool {
    let mut properties = HashMap::new();

    properties.insert("document_id".to_string(), PropertySchema {
        property_type: "string".to_string(),
        description: "The ID of the document to delete".to_string(),
        items: None,
        default: None,
        minimum: None,
        maximum: None,
    });

    Tool {
        name: "delete_document".to_string(),
        description: "Delete a document and all its associated chunks from the system".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["document_id".to_string()],
        },
    }
}

fn create_clear_memory_tool() -> Tool {
    let mut properties = HashMap::new();

    properties.insert("memory_type".to_string(), PropertySchema {
        property_type: "string".to_string(),
        description: "Type of memory to clear: 'episodic', 'semantic', 'procedural', or 'all'".to_string(),
        items: None,
        default: Some(json!("all")),
        minimum: None,
        maximum: None,
    });

    properties.insert("confirm".to_string(), PropertySchema {
        property_type: "boolean".to_string(),
        description: "Confirmation flag required for destructive operations".to_string(),
        items: None,
        default: Some(json!(false)),
        minimum: None,
        maximum: None,
    });

    Tool {
        name: "clear_memory".to_string(),
        description: "Clear specified memory types from the system (DESTRUCTIVE OPERATION)".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["confirm".to_string()],
        },
    }
}

fn create_get_memory_stats_tool() -> Tool {
    Tool {
        name: "get_memory_stats".to_string(),
        description: "Get memory usage statistics and storage information".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: vec![],
        },
    }
}

fn create_get_system_metrics_tool() -> Tool {
    Tool {
        name: "get_system_metrics".to_string(),
        description: "Get comprehensive system performance metrics and health status".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: vec![],
        },
    }
}

fn create_health_check_tool() -> Tool {
    Tool {
        name: "health_check".to_string(),
        description: "Perform a comprehensive health check of all system components".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: vec![],
        },
    }
}

fn create_configure_system_tool() -> Tool {
    let mut properties = HashMap::new();

    properties.insert("config".to_string(), PropertySchema {
        property_type: "object".to_string(),
        description: "Configuration settings to update".to_string(),
        items: None,
        default: None,
        minimum: None,
        maximum: None,
    });

    Tool {
        name: "configure_system".to_string(),
        description: "Update system configuration settings".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["config".to_string()],
        },
    }
}

fn create_batch_ingest_tool() -> Tool {
    let mut properties = HashMap::new();

    properties.insert("documents".to_string(), PropertySchema {
        property_type: "array".to_string(),
        description: "Array of documents to ingest in batch".to_string(),
        items: Some(Box::new(PropertySchema {
            property_type: "object".to_string(),
            description: "Document with content and metadata".to_string(),
            items: None,
            default: None,
            minimum: None,
            maximum: None,
        })),
        default: None,
        minimum: None,
        maximum: None,
    });

    properties.insert("batch_size".to_string(), PropertySchema {
        property_type: "integer".to_string(),
        description: "Number of documents to process in parallel".to_string(),
        items: None,
        default: Some(json!(10)),
        minimum: Some(1),
        maximum: Some(100),
    });

    Tool {
        name: "batch_ingest".to_string(),
        description: "Ingest multiple documents in batch with parallel processing".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["documents".to_string()],
        },
    }
}

fn create_semantic_search_tool() -> Tool {
    let mut properties = HashMap::new();

    properties.insert("query".to_string(), PropertySchema {
        property_type: "string".to_string(),
        description: "Semantic search query using natural language".to_string(),
        items: None,
        default: None,
        minimum: None,
        maximum: None,
    });

    properties.insert("context".to_string(), PropertySchema {
        property_type: "string".to_string(),
        description: "Additional context to improve search relevance".to_string(),
        items: None,
        default: None,
        minimum: None,
        maximum: None,
    });

    properties.insert("limit".to_string(), PropertySchema {
        property_type: "integer".to_string(),
        description: "Maximum number of results to return".to_string(),
        items: None,
        default: Some(json!(10)),
        minimum: Some(1),
        maximum: Some(100),
    });

    Tool {
        name: "semantic_search".to_string(),
        description: "Perform advanced semantic search with contextual understanding".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["query".to_string()],
        },
    }
}

fn create_hybrid_search_tool() -> Tool {
    let mut properties = HashMap::new();

    properties.insert("query".to_string(), PropertySchema {
        property_type: "string".to_string(),
        description: "Search query combining keyword and semantic matching".to_string(),
        items: None,
        default: None,
        minimum: None,
        maximum: None,
    });

    properties.insert("semantic_weight".to_string(), PropertySchema {
        property_type: "number".to_string(),
        description: "Weight for semantic search (0.0-1.0), remainder for keyword search".to_string(),
        items: None,
        default: Some(json!(0.7)),
        minimum: None,
        maximum: None,
    });

    properties.insert("limit".to_string(), PropertySchema {
        property_type: "integer".to_string(),
        description: "Maximum number of results to return".to_string(),
        items: None,
        default: Some(json!(10)),
        minimum: Some(1),
        maximum: Some(100),
    });

    Tool {
        name: "hybrid_search".to_string(),
        description: "Combine semantic vector search with keyword matching for optimal results".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["query".to_string()],
        },
    }
}
