# Legal-Contract-LORA-ETL-

## Quick start

```bash
uv sync                                        # install dependencies
docker compose up -d elasticsearch             # start the clause index
uv run python -m indexing.sanity_check         # verify data is indexed correctly
```
