import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import local_kb
print(f"Total seed articles: {len(local_kb.SEED_WIKIPEDIA_ARTICLES)}")
for a in local_kb.SEED_WIKIPEDIA_ARTICLES:
    print(f"- {a['title']}")
