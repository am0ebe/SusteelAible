    def process_report(self, pdf_path: str):
        """
        Full pipeline for processing a report
        Uses config flags set in __init__ to determine which analyses to run
        """
        print(f"\n{'='*80}")
        print(f"Processing: {pdf_path}")
        print(f"{'='*80}")

        pdf_stem = Path(pdf_path).stem

        # Step 1: Extract text
        text = self.extract_text_from_pdf(pdf_path)
        print(f"Total text length: {len(text)} characters")

        # Step 2: Detect language and translate if needed
        lang_cache = self.cache_dir / f"{pdf_stem}_lang.txt"

        if self.use_cache and lang_cache.exists():
            lang = lang_cache.read_text(encoding='utf-8').strip()
            print(f"Loaded cached language: {lang}")
        else:
            lang = self.detect_language(text)
            if self.use_cache:
                lang_cache.write_text(lang, encoding='utf-8')

        # Step 3: Chunk text (by paragraphs)
        chunks = self.chunk_text_by_paragraphs(text)
        print(f"Created {len(chunks)} chunks")

        # Step 4: Translate if not English and auto_translate enabled
        cache_suffix = ""
        if lang != 'en' and self.auto_translate:
            if lang not in self.supported_languages:
                print(f"⚠️  Translation not supported for language: {lang}")
                print(f"   Supported languages: {list(self.supported_languages.keys())}")
                print(f"   Proceeding with original text (may affect analysis accuracy)")
                cache_suffix = f"_unsupported_{lang}"
            else:
                trans_cache = self.cache_dir / f"{pdf_stem}_translated_{lang}_en.json"

                if self.use_cache and trans_cache.exists():
                    print(f"Loading cached translation from {trans_cache}")
                    with open(trans_cache, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    cache_suffix = f"_translated_{lang}_en"
                else:
                    chunks, cache_suffix = self.translate_to_english(chunks, lang)

                    if self.use_cache:
                        with open(trans_cache, 'w', encoding='utf-8') as f:
                            json.dump(chunks, f, ensure_ascii=False, indent=2)
                        print(f"Cached translation to {trans_cache}")

        # Step 5: Filter for climate content
        climate_chunks = self.filter_climate_chunks(chunks)

        # Step 6-8: Run enabled analyses
        analyzed_chunks = climate_chunks
        analyzed_chunks = self.analyze_specificity(analyzed_chunks)
        analyzed_chunks = self.analyze_sentiment(analyzed_chunks)
        analyzed_chunks = self.analyze_commitments(analyzed_chunks)

        # Step 9: Calculate scores
        scores = self.calculate_report_score(analyzed_chunks)

        # Step 10: Save results
        results = {
            'pdf_path': str(pdf_path),
            'language': lang,
            'translation': cache_suffix if cache_suffix else 'none',
            'total_chunks': len(chunks),
            'climate_chunks': len(climate_chunks),
            'scores': scores,
            'sample_chunks': analyzed_chunks[:10]  # Save first 10 for inspection
        }

        results_file = self.cache_dir / f"{pdf_stem}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to {results_file}")

        return scores

    def process_all_reports(self, reports_dir: str, pattern="**/*.pdf"):
        """Process all PDFs in directory tree using config from __init__"""
        pdf_files = list(Path(reports_dir).glob(pattern))
        print(f"\nFound {len(pdf_files)} PDF files in {reports_dir}")

        results = []
        failed = []

        for pdf_path in tqdm(pdf_files, desc="Processing reports"):
            try:
                scores = self.process_report(str(pdf_path))
                results.append({
                    'file': str(pdf_path),
                    'company': pdf_path.parent.name,
                    'year': self._extract_year(pdf_path.name),
                    'scores': scores
                })
            except Exception as e:
                print(f"\n❌ Failed: {pdf_path}")
                print(f"   Error: {e}")
                failed.append(str(pdf_path))
                continue

        # Print summary
        self._print_batch_summary(results, failed, len(pdf_files))

        # Save aggregated results
        output_file = self.cache_dir / 'all_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'failed': failed,
                'total_processed': len(results),
                'total_found': len(pdf_files)
            }, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")

        return results

    def process(self, path: str):
        """
        Smart processor: automatically detects if path is a file or directory
        Uses config from __init__ for all processing options
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Case 1: Single PDF file
        if path_obj.is_file() and path_obj.suffix.lower() == '.pdf':
            print(f"\n{'='*80}")
            print(f"PROCESSING SINGLE PDF")
            print(f"{'='*80}")

            scores = self.process_report(str(path_obj))
            self._print_single_summary(path_obj, scores)
            return scores

        # Case 2: Directory
        elif path_obj.is_dir():
            nested_pdfs = list(path_obj.glob("**/*.pdf"))

            if not nested_pdfs:
                print(f"❌ No PDF files found in {path}")
                return []

            print(f"\n{'='*80}")
            print(f"PROCESSING DIRECTORY: {path}")
            print(f"{'='*80}")
            print(f"Found {len(nested_pdfs)} PDF files")

            results = self.process_all_reports(str(path_obj), pattern="**/*.pdf")
            return results

        else:
            raise ValueError(f"Path must be a PDF file or directory: {path}")

    def _extract_year(self, filename: str) -> str:
        """Extract year from filename (e.g., Report_2020.pdf -> 2020)"""
        import re
        match = re.search(r'20\d{2}', filename)
        return match.group(0) if match else 'unknown'

    def _print_single_summary(self, pdf_path: Path, scores: Dict):
        """Pretty print results for single PDF"""
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"File: {pdf_path.name}")
        print(f"Company: {pdf_path.parent.name}")

        print(f"\n📊 Component Scores:")
        if 'overall_specificity' in scores:
            print(f"  Specificity: {scores['overall_specificity']:.3f} - {self._interpret_score(scores['overall_specificity'])}")

        print(f"\nChunks analyzed: {scores['num_chunks_analyzed']}")

        # Show distributions
        if 'label_distribution' in scores:
            print(f"\nLabel Distribution:")
            for label, count in sorted(scores['label_distribution'].items()):
                pct = count / scores['num_chunks_analyzed'] * 100
                bar = '█' * int(pct / 2)
                print(f"  {label:15s}: {count:3d} ({pct:5.1f}%) {bar}")

    def _print_batch_summary(self, results: List[Dict], failed: List[str], total: int):
        """Pretty print results for batch processing"""
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successfully processed: {len(results)}/{total}")

        if failed:
            print(f"❌ Failed: {len(failed)}")
            for f in failed[:5]:  # Show first 5
                print(f"   - {Path(f).name}")
            if len(failed) > 5:
                print(f"   ... and {len(failed)-5} more")

        if results:
            # Calculate aggregate statistics
            avg_score = sum(r['scores']['overall_specificity'] for r in results) / len(results)
            print(f"\nAggregate Statistics:")
            print(f"  Average Specificity Score: {avg_score:.3f}")
            print(f"  → {self._interpret_score(avg_score)}")

            # Top 5 and bottom 5
            sorted_results = sorted(results, key=lambda x: x['scores']['overall_specificity'], reverse=True)

            print(f"\n📊 Top 5 Most Specific Reports:")
            for i, r in enumerate(sorted_results[:5], 1):
                score = r['scores']['overall_specificity']
                print(f"  {i}. {r['company']:20s} ({r.get('year', '?')}): {score:.3f}")

            print(f"\n📊 Top 5 Least Specific Reports:")
            for i, r in enumerate(sorted_results[-5:], 1):
                score = r['scores']['overall_specificity']
                print(f"  {i}. {r['company']:20s} ({r.get('year', '?')}): {score:.3f}")

    def _interpret_score(self, score: float) -> str:
        """Interpret specificity score"""
        if score >= 0.7:
            return "Highly specific (concrete targets & actions)"
        elif score >= 0.5:
            return "Moderately specific (mix of concrete & vague)"
        elif score >= 0.3:
            return "Low specificity (mostly vague statements)"
        else:
            return "Very low specificity (lacks concrete information)"


    # def show_chunk_structure(self, pdf_path: str, stage="all"):
    #     """
    #     Show chunk structure at different pipeline stages

    #     Args:
    #         pdf_path: Path to PDF
    #         stage: "raw", "filtered", "analyzed", or "all"
    #     """
    #     print(f"\n{'='*80}")
    #     print(f"CHUNK STRUCTURE ANALYSIS")
    #     print(f"{'='*80}\n")

    #     # Extract and chunk
    #     text = self.extract_text_from_pdf(pdf_path)
    #     raw_chunks = self.chunk_text_by_paragraphs(text)

    #     if stage in ["raw", "all"]:
    #         print(f"STAGE 1: Raw chunks (after text splitting)")
    #         print(f"  Type: List[str]")
    #         print(f"  Count: {len(raw_chunks)}")
    #         print(f"  Example structure:")
    #         if raw_chunks:
    #             print(f"    chunk[0] = {repr(raw_chunks[0][:100])}...")
    #             print(f"    Length: {len(raw_chunks[0])} chars\n")

    #     if stage in ["filtered", "analyzed", "all"]:
    #         # Filter climate
    #         filtered_chunks = self.filter_climate_chunks(raw_chunks)


    #         print(f"STAGE 2: Filtered chunks (after climate detection)")
    #         print(f"  Type: List[Dict]")
    #         print(f"  Count: {len(filtered_chunks)}")
    #         print(f"  Example structure:")
    #         if filtered_chunks:
    #             print(f"    chunk[0] = {{")
    #             print(f"      'text': {repr(filtered_chunks[0]['text'][:80])}...,")
    #             print(f"      'detector_score': {filtered_chunks[0].get('detector_score', 'N/A')}")
    #             print(f"    }}\n")

    #     if stage in ["analyzed", "all"]:
    #         # Analyze
    #         analyzed_chunks = self.analyze_specificity(filtered_chunks[:5])  # Just 5 for demo

    #         print(f"STAGE 3: Analyzed chunks (after specificity)")
    #         print(f"  Type: List[Dict]")
    #         print(f"  Example structure:")
    #         if analyzed_chunks:
    #             print(f"    chunk[0] = {{")
    #             print(f"      'text': {repr(analyzed_chunks[0]['text'][:60])}...,")
    #             print(f"      'detector_score': {analyzed_chunks[0].get('detector_score', 'N/A')},")
    #             print(f"      'specificity_label': {analyzed_chunks[0].get('specificity_label', 'N/A')},")
    #             print(f"      'specificity_score': {analyzed_chunks[0].get('specificity_score', 'N/A')}")
    #             print(f"      'sentiment_label': {analyzed_chunks[0].get('sentiment_label', 'N/A')},")
    #             print(f"      'sentiment_score': {analyzed_chunks[0].get('sentiment_score', 'N/A')}")
    #             print(f"    }}\n")