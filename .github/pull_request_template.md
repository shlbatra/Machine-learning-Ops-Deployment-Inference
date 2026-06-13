## Summary
<!-- What changed and why -->

## Changes
<!-- Key changes, grouped logically -->

- 

## Testing
<!-- How was this verified? -->

- [ ] Syntax check (`python -m py_compile`) on modified files
- [ ] Existing tests pass (`python -m pytest test/`)
- [ ] Docker image builds locally (`docker build -t ml-pipelines-kfp-image:<branch> .`)
- [ ] Verified inside container (`docker run --rm <image> python -c "..."`)
- [ ] Pipeline run on Vertex AI
- [ ] Logs verified in Cloud Logging

## Cloud Logging verification
<!-- For changes that affect logging, confirm the log output in GCP -->

```
severity="INFO"
jsonPayload.message=~"<keyword>"
jsonPayload.module="<module>"
labels.ml_pipelines_run_id="<run-id>"
```

## Rollback
<!-- How to revert if something breaks in prod -->

- Revert image tag in `constants.py` to previous version
- Redeploy pipeline with previous image
