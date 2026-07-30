"""
SmartMap ingestion CLI — builds and saves the PageIndex SmartMap for a PDF.

Must be run ONCE before using RETRIEVAL_STRATEGY=pageindex or hybrid.
Subsequent runs rebuild the sidecar (use --force to skip the existence check).

Usage:
    python scripts/ingest_smartmap.py
    python scripts/ingest_smartmap.py --pdf data/raw/rbi_kyc_master_direction.pdf
    python scripts/ingest_smartmap.py --doc-id rbi_kyc_master_direction --force

Output:
    qdrant_storage/smartmap_<doc_id>.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config.revamp_settings import revamp_settings
from src.ingestion.smart_map_builder import (
    build_smart_map,
    save_smart_map,
    smart_map_exists,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PageIndex SmartMap from PDF")
    parser.add_argument(
        "--pdf",
        default="data/raw/rbi_kyc_master_direction.pdf",
        help="Path to the PDF file (default: data/raw/rbi_kyc_master_direction.pdf)",
    )
    parser.add_argument(
        "--doc-id",
        default=revamp_settings.PAGEINDEX_DOC_ID,
        help=f"Document identifier (default: {revamp_settings.PAGEINDEX_DOC_ID})",
    )
    parser.add_argument(
        "--out-dir",
        default=revamp_settings.SMARTMAP_DIR,
        help=f"Sidecar output directory (default: {revamp_settings.SMARTMAP_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if sidecar already exists",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"ERROR: PDF not found: {args.pdf}")
        sys.exit(1)

    if smart_map_exists(args.doc_id, args.out_dir) and not args.force:
        print(
            f"SmartMap already exists for {args.doc_id!r} in {args.out_dir}/\n"
            f"Use --force to rebuild."
        )
        return

    print(f"Building SmartMap for {args.doc_id!r} from {args.pdf} ...")
    smart_map = build_smart_map(doc_id=args.doc_id, pdf_path=args.pdf)

    print(f"  Sections detected : {len(smart_map.toc)}")
    print(f"  First 5 sections  :")
    for entry in smart_map.toc[:5]:
        char_count = len(smart_map.sections.get(entry.section_id, ""))
        print(
            f"    [{entry.section_id}] L{entry.level} "
            f"pp.{entry.start_page}-{entry.end_page}  "
            f"{entry.title[:60]}  ({char_count} chars)"
        )
    if len(smart_map.toc) > 5:
        print(f"    ... and {len(smart_map.toc) - 5} more")

    path = save_smart_map(smart_map, sidecar_dir=args.out_dir)
    print(f"\nSmartMap saved → {path}")
    print("Ready for RETRIEVAL_STRATEGY=pageindex or hybrid")


if __name__ == "__main__":
    main()
