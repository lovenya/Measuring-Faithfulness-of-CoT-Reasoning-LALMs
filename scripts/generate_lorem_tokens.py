#!/usr/bin/env python3
"""
Generate Lorem Ipsum word pools for filler text experiments.

This script creates JSON files containing Lorem Ipsum words that can be used
as filler tokens in the experiments. No model tokenizer is required.

Usage:
    python scripts/generate_lorem_tokens.py
"""

import json
import os

LOREM_IPSUM_TEXT = """
Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor 
incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis nostrud 
exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat Duis aute 
irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla 
pariatur Excepteur sint occaecat cupidatat non proident sunt in culpa qui officia 
deserunt mollit anim id est laborum Sed ut perspiciatis unde omnis iste natus error 
sit voluptatem accusantium doloremque laudantium totam rem aperiam eaque ipsa quae 
ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo 
Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit sed quia 
consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt Neque porro 
quisquam est qui dolorem ipsum quia dolor sit amet consectetur adipisci velit
"""

def main():
    """Generate and save Lorem Ipsum word pool."""
    output_dir = "assets/lorem_tokens"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into words and get unique ones
    words = LOREM_IPSUM_TEXT.split()
    unique_words = list(set(words))
    
    print(f"Generated {len(words)} words, {len(unique_words)} unique")
    
    # Save to JSON (model-agnostic - works for all models)
    output_data = {
        "description": "Lorem Ipsum words for filler text experiments",
        "total_words": len(words),
        "unique_words": len(unique_words),
        "tokens": unique_words
    }
    
    # Save a single file that works for all models
    output_path = os.path.join(output_dir, "lorem_tokens.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to: {output_path}")
    print(f"Sample words: {unique_words[:15]}")

if __name__ == "__main__":
    main()

