"""
Validation script to check for text extraction quality issues.
"""

import json
import re
from pathlib import Path
from collections import Counter


def check_sentence_boundaries(chunks):
    """Check if chunks are cutting mid-sentence."""
    issues = []
    for i, chunk in enumerate(chunks):
        # Check if chunk ends with sentence-ending punctuation
        if not re.search(r'[.!?]\s*$', chunk.strip()):
            issues.append({
                'chunk_id': i,
                'issue': 'possible_mid_sentence_cut',
                'last_chars': chunk[-50:] if len(chunk) > 50 else chunk
            })
    return issues


def check_repetitions(chunks):
    """Check for excessive word repetitions."""
    issues = []
    for i, chunk in enumerate(chunks):
        words = chunk.split()
        if len(words) < 4:
            continue

        # Find consecutive repetitions
        j = 0
        while j < len(words):
            word = words[j]
            count = 1
            while j + count < len(words) and words[j + count] == word:
                count += 1

            if count > 2:  # More than 2 repetitions
                issues.append({
                    'chunk_id': i,
                    'word': word,
                    'repetitions': count,
                    'context': ' '.join(words[max(0, j-3):min(len(words), j+count+3)])
                })
            j += count

    return issues


def check_garbage_patterns(chunks):
    """Check for obvious garbage text patterns."""
    issues = []
    for i, chunk in enumerate(chunks):
        words = chunk.split()
        if len(words) < 10:
            continue

        # Check single-char token ratio
        single_char = sum(1 for w in words if len(w) == 1 and w.isalpha())
        single_char_ratio = single_char / len(words)

        if single_char_ratio > 0.3:
            issues.append({
                'chunk_id': i,
                'issue': 'high_single_char_ratio',
                'ratio': single_char_ratio,
                'sample': chunk[:200]
            })

        # Check average word length
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len < 2:
            issues.append({
                'chunk_id': i,
                'issue': 'very_short_avg_word_length',
                'avg_length': avg_len,
                'sample': chunk[:200]
            })

    return issues


def analyze_chunk_distribution(chunks):
    """Analyze the distribution of chunk sizes."""
    lengths = [len(c) for c in chunks]

    return {
        'total_chunks': len(chunks),
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        'median_length': sorted(lengths)[len(lengths)//2] if lengths else 0,
    }


def validate_extraction(json_file):
    """Run all validation checks on extracted data."""
    print(f"\n{'='*80}")
    print(f"VALIDATING: {json_file}")
    print(f"{'='*80}\n")

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = data.get('chunks', [])

    # 1. Chunk distribution
    print("📊 CHUNK DISTRIBUTION:")
    dist = analyze_chunk_distribution(chunks)
    for key, value in dist.items():
        print(f"  {key}: {value:.0f}" if isinstance(
            value, float) else f"  {key}: {value}")

    # 2. Sentence boundary check
    print(f"\n🔍 SENTENCE BOUNDARY CHECK:")
    boundary_issues = check_sentence_boundaries(chunks)
    if boundary_issues:
        print(
            f"  ⚠️  Found {len(boundary_issues)} potential mid-sentence cuts")
        for issue in boundary_issues[:3]:  # Show first 3
            print(f"     Chunk {issue['chunk_id']}: ...{issue['last_chars']}")
    else:
        print(f"  ✅ All chunks end at sentence boundaries")

    # 3. Repetition check
    print(f"\n🔄 REPETITION CHECK:")
    rep_issues = check_repetitions(chunks)
    if rep_issues:
        print(f"  ⚠️  Found {len(rep_issues)} excessive repetitions")
        for issue in rep_issues[:3]:  # Show first 3
            print(
                f"     Chunk {issue['chunk_id']}: '{issue['word']}' x{issue['repetitions']}")
            print(f"        Context: {issue['context']}")
    else:
        print(f"  ✅ No excessive repetitions found")

    # 4. Garbage pattern check
    print(f"\n🗑️  GARBAGE PATTERN CHECK:")
    garbage_issues = check_garbage_patterns(chunks)
    if garbage_issues:
        print(f"  ⚠️  Found {len(garbage_issues)} potential garbage patterns")
        for issue in garbage_issues[:3]:  # Show first 3
            print(f"     Chunk {issue['chunk_id']}: {issue['issue']}")
            if 'ratio' in issue:
                print(f"        Ratio: {issue['ratio']:.2%}")
            print(f"        Sample: {issue['sample'][:100]}...")
    else:
        print(f"  ✅ No garbage patterns detected")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Sentence boundary issues: {len(boundary_issues)}")
    print(f"  Repetition issues: {len(rep_issues)}")
    print(f"  Garbage issues: {len(garbage_issues)}")
    print(f"{'='*80}\n")

    return {
        'distribution': dist,
        'sentence_boundary_issues': len(boundary_issues),
        'repetition_issues': len(rep_issues),
        'garbage_issues': len(garbage_issues),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Default: look for the most recent prep file
        cache_dir = Path("cache")
        if cache_dir.exists():
            prep_files = list(cache_dir.glob("*_prep.json"))
            if prep_files:
                json_file = max(prep_files, key=lambda p: p.stat().st_mtime)
                print(f"Using most recent prep file: {json_file}")
            else:
                print("No prep files found in cache/")
                sys.exit(1)
        else:
            print("Cache directory not found")
            sys.exit(1)

    validate_extraction(json_file)
