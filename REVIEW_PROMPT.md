# Comprehensive Project Review Prompt

Please conduct an **extremely exhaustive and detailed analysis** of this AI Challenge 2024 video retrieval system. I need you to provide:

## 1. COMPETITION UNDERSTANDING
**Analyze the competition requirements with extreme detail:**

### What is the AIC 2024 Event Retrieval Challenge?
- **Exact competition name and organizers**
- **Submission platform** (CodaLab specifics)
- **Dataset characteristics** (video types, language, duration, scale)
- **Evaluation metrics** (how exactly are submissions scored?)
- **Timeline and phases** (prelims, finals, deadlines)

### Challenge Modes - Be Explicit About Each:
- **Textual KIS (Known-Item Search)**: 
  - Input format: `_____`
  - Output format: `_____`
  - Scoring methodology: `_____`
- **Video KIS (finals)**:
  - Input format: `_____`
  - Output format: `_____` 
  - Special constraints: `_____`
- **Q&A Mode**:
  - Input format: `_____`
  - Output format: `_____`
  - Answer format requirements: `_____`

### Submission Format Requirements:
- **Exact CSV format** with headers/no headers
- **File naming conventions**
- **ZIP structure for multi-query submissions**
- **Maximum candidates per query** (100?)
- **Ranking requirements** (are candidates ranked or just listed?)

## 2. SYSTEM ARCHITECTURE ANALYSIS
**Dissect every component of our system:**

### Frame Sampling Strategy:
- **Competition keyframes**: What do we get from organizers?
- **Our intelligent sampling**: How does it augment competition data?
- **Integration approach**: How do both keyframe sources work together?
- **Data flow**: From MP4 → keyframes → features → retrieval

### Pipeline Components - Explain Each:
1. **frames_intelligent.py vs frames_intelligent_fast.py**:
   - When to use which version?
   - Exact algorithm differences
   - Performance trade-offs
   - Input/output specifications

2. **index.py**:
   - What keyframe directories does it scan?
   - How does FAISS indexing work?
   - Mapping file structure and purpose
   - Feature extraction process

3. **Hybrid Search System**:
   - Dense retrieval (CLIP): exact model, embeddings, similarity
   - Lexical retrieval (BM25): corpus construction, tokenization
   - RRF fusion: algorithm, parameters, rationale

4. **Re-ranking Components**:
   - GBM learned re-ranker: features, training, inference
   - Multi-frame re-scoring: temporal window, aggregation
   - Importance score integration: how scores flow through pipeline

## 3. TECHNICAL IMPLEMENTATION DETAILS

### File Structure Analysis:
Map out the **complete file structure** this system expects:
```
dataset_root/
  videos/ ← [Explain: Required? Optional? Format constraints?]
  keyframes/ ← [Explain: Source? Format? Relationship to videos?]
  keyframes_intelligent/ ← [Explain: How generated? When used?]
  meta/ ← [Explain: Each file type, required vs optional]
    *.map_keyframe.csv ← [Exact column specifications]
    *.media_info.json ← [Exact schema]
    objects/ ← [Object detection format, source]
  features/ ← [Precomputed features, dimensions, format]
  artifacts/ ← [Generated indices, what gets created when]
```

### Data Flow Analysis:
**Trace the complete data flow for a single query:**
1. Input: `"Text query in Vietnamese"`
2. Dense path: query → CLIP encoding → FAISS search → candidates
3. Lexical path: query → tokenization → BM25 → candidates  
4. Fusion: RRF → combined ranking
5. Re-ranking: GBM features → probability scores
6. Multi-frame: temporal context → final scores
7. Output: CSV with video_id,frame_idx pairs

### Configuration and Parameters:
**Document every configurable parameter:**
- Model choices (EVA02-L-14, ViT-L-14, etc.) - when to use which?
- Sampling parameters (window_size, min_gap, coverage_fps)
- Search parameters (topk, rrf_k, dedup_radius)
- Re-ranking features and weights

## 4. USAGE SCENARIOS AND WORKFLOWS

### Complete Workflow Documentation:
**For a new user with raw videos, document the exact step-by-step process:**

#### Scenario A: Starting from scratch with MP4 files
```bash
# Step 1: _____ (what exactly happens here?)
# Step 2: _____ (what files get created?)
# Step 3: _____ (what directories are expected?)
# ... continue until final CSV submission
```

#### Scenario B: Using competition-provided keyframes
```bash
# How does workflow differ when keyframes are pre-provided?
# What steps can be skipped?
# How to verify keyframe quality?
```

#### Scenario C: Large dataset optimization
```bash
# When to use fast vs full sampling?
# GPU utilization strategy
# Memory management for large datasets
# Batch processing recommendations
```

## 5. PERFORMANCE ANALYSIS AND OPTIMIZATION

### Benchmarking Information:
- **Processing speeds** for each component
- **Memory requirements** (RAM, VRAM, disk)
- **Scaling characteristics** (how performance changes with dataset size)
- **Quality metrics** (recall@k improvements over baselines)

### Optimization Strategies:
- **When to use which sampling mode?**
- **Model selection guidance** (4070 Ti constraints)
- **Parameter tuning recommendations**
- **Common bottlenecks and solutions**

## 6. QUALITY ASSURANCE CHECKLIST

### Verify System Completeness:
- [ ] All competition requirements addressed
- [ ] File format compatibility verified
- [ ] Error handling and edge cases covered
- [ ] Dependencies clearly specified
- [ ] Documentation accuracy confirmed

### Integration Testing:
- [ ] End-to-end workflow from MP4 to CSV works
- [ ] All sampling modes produce valid outputs
- [ ] Hybrid search combines results correctly
- [ ] Re-ranking improves over baseline
- [ ] Export format matches competition requirements

## 7. TROUBLESHOOTING GUIDE

### Common Issues and Solutions:
Document solutions for:
- Out of memory errors
- Missing keyframes or files  
- Performance bottlenecks
- Quality degradation
- Format compatibility issues

### Debugging Workflows:
How to verify each component is working correctly:
- Frame sampling output validation
- Index integrity checks
- Search result sanity checks
- CSV format verification

## 8. COMPETITIVE ANALYSIS

### System Strengths:
- What makes our approach competitive?
- Novel contributions vs standard approaches
- Expected performance advantages

### System Limitations:
- What scenarios might challenge our system?
- Computational or memory constraints
- Dataset-specific assumptions

### Improvement Opportunities:
- Short-term enhancements possible
- Long-term research directions
- Alternative approaches to consider

---

## DELIVERABLE REQUIREMENTS:

Provide your analysis as a **comprehensive technical report** covering all points above with:

1. **Executive summary** (2-3 pages)
2. **Detailed technical analysis** (10-15 pages)
3. **Step-by-step usage guide** (2-3 pages) 
4. **Troubleshooting reference** (1-2 pages)
5. **Performance benchmarks** where available

**Be extremely specific about:**
- File formats, schemas, and data structures
- Command-line usage with actual parameters
- Expected inputs and outputs at each stage
- Error conditions and recovery procedures
- Performance characteristics and limitations

**Focus on practical implementation details** that would allow someone to:
- Understand exactly what the competition requires
- Successfully deploy and run the complete system
- Optimize performance for their specific use case
- Troubleshoot issues when they arise
- Extend or modify components as needed