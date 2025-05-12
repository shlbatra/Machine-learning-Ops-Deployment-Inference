# System Patterns

**System Architecture:**
- Modular pipeline components (data, schema, model, evaluation, deployment) orchestrated via KFP DSL

**Key Technical Decisions:**
- Use of Google Cloud Vertex AI for managed ML services
- BigQuery for data storage and access
- Docker for reproducible environments
- CI/CD for automation

**Design Patterns:**
- Decoupled, reusable pipeline components
- Best practices for MLOps and cloud-native ML workflows 