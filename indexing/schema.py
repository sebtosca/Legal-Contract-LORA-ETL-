"""Elasticsearch index definition for the labeled clause corpus.

Fields per the PRD: clause text, label, company/CIK, filing date,
industry/SIC code, clause type - enabling stratified sampling and filtered
querying when curating the training/eval sets in Phase 4.
"""

INDEX_NAME = "legal-clauses"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "clause_id": {"type": "keyword"},
            "label": {"type": "keyword"},
            "section_header": {"type": "text"},
            "text": {"type": "text"},
            "cik": {"type": "keyword"},
            "company": {"type": "keyword"},
            "form_type": {"type": "keyword"},
            "file_type": {"type": "keyword"},
            "accession": {"type": "keyword"},
            "filename": {"type": "keyword"},
            "filing_date": {"type": "date", "ignore_malformed": True},
            "filing_year": {"type": "keyword"},
            "sic": {"type": "keyword"},
            "sic_description": {"type": "keyword"},
            "text_len": {"type": "integer"},
        }
    }
}
