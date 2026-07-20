# AGENTS.md

## Project Goal

This project is:

"Natural-language intelligent video retrieval and forensic system"

Current stage:

PoC validation only.

## Engineering Rules

1. Prefer minimal runnable implementations
2. Do not over-engineer
3. Do not introduce microservices
4. Do not use Docker
5. Do not use LangChain
6. Keep code simple and readable
7. All imports must be correct
8. All code must be runnable
9. Prefer local JSON metadata over databases
10. Optimize for single-developer workflow

## Current Stack

- Python
- FastAPI
- OpenCV
- PaddleOCR
- OpenCLIP
- FAISS

## Architecture Principles

- Single-machine only
- Metadata-driven design
- OCR/ASR/YOLO should be modular
- Avoid unnecessary abstractions

## Current Focus

Implement OCR video retrieval PoC.