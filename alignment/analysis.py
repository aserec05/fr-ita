"""
Embedding Analysis — synonyms, cognates, polysemy, OOV
"""
import os
import numpy as np
from fr_ita.utils import load_word_vectors, load_bilingual_dictionary
from fr_ita.config import DATA_DIR

OUTPUT_DIR = os.path.join(DATA_DIR, 'analysis')

def cosine_similarity(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def nearest_neighbors(word, word_vectors, topk=10):
    if word not in word_vectors:
        return []
    query = word_vectors[word]
    sims = [(w, cosine_similarity(query, v))
            for w, v in word_vectors.items() if w != word]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:topk]

def analyze_synonyms_antonyms(word_vectors, language, embedding_type):
    query_words = {
        'french':  ['bien', 'grand', 'rapide', 'heureux', 'maison'],
        'italian': ['bene', 'grande', 'veloce', 'felice', 'casa'],
    }
    print(f"\nSYNONYMS/ANTONYMS [{embedding_type} - {language}]")
    for word in [w for w in query_words.get(language, []) if w in word_vectors]:
        neighbors = nearest_neighbors(word, word_vectors, topk=5)
        print(f"  '{word}' -> {[(w,round(s,3)) for w,s in neighbors[:3]]}")

def analyze_cognates(fr_vectors, it_vectors, bilingual_dict, embedding_type, n=15):
    print(f"\nCOGNATES [{embedding_type}]")
    cognates = []
    for fr_w, it_w in bilingual_dict.items():
        if (fr_w == it_w or (len(fr_w)>3 and len(it_w)>3 and fr_w[:4]==it_w[:4])):
            if fr_w in fr_vectors and it_w in it_vectors:
                sim = cosine_similarity(fr_vectors[fr_w], it_vectors[it_w])
                cognates.append((fr_w, it_w, sim))
    cognates.sort(key=lambda x: x[2], reverse=True)
    print(f"  {len(cognates)} cognate pairs found")
    for fr_w, it_w, sim in cognates[:n]:
        print(f"  {fr_w:<18} {it_w:<18} {sim:.4f}")
    if cognates:
        print(f"  Avg sim: {np.mean([s for _,_,s in cognates]):.4f}")

def analyze_polysemy(word_vectors, language, embedding_type):
    polysemous = {
        'french':  ['fois', 'cote', 'etat', 'piece', 'langue'],
        'italian': ['volta', 'lato', 'stato', 'pezzo', 'lingua'],
    }
    print(f"\nPOLYSEMY [{embedding_type} - {language}]")
    for word in [w for w in polysemous.get(language, []) if w in word_vectors]:
        neighbors = nearest_neighbors(word, word_vectors, topk=8)
        variance = np.var([s for _, s in neighbors]) if neighbors else 0
        top = [(w, round(s,2)) for w,s in neighbors[:4]]
        print(f"  {word:<12} {top}  var={variance:.4f}")

def analyze_oov(fr_vocab, it_vocab, embedding_type):
    strategies = {
        'fasttext': "subword n-grams -> OOV words get approximate vectors",
        'glove':    "no OOV strategy -> unknown words skipped",
        'word2vec': "no OOV strategy -> unknown words skipped",
        'tfidf':    "document-based -> OOV words get zero weight",
    }
    print(f"\nOOV [{embedding_type}]")
    try:
        fr_v, it_v = load_word_vectors(embedding_type)
        fr_oov = fr_vocab - set(fr_v.keys())
        it_oov = it_vocab - set(it_v.keys())
        print(f"  FR: {len(fr_v)} in vocab, {len(fr_oov)} OOV ({len(fr_oov)/len(fr_vocab)*100:.1f}%)")
        print(f"  IT: {len(it_v)} in vocab, {len(it_oov)} OOV ({len(it_oov)/len(it_vocab)*100:.1f}%)")
        print(f"  Strategy: {strategies.get(embedding_type, '')}")
        print(f"  Sample OOV FR: {list(fr_oov)[:6]}")
    except Exception as e:
        print(f"  Error: {e}")



if __name__ == "__main__":
    import argparse
    from fr_ita.utils import load_tatoeba_tsv, preprocess_text

    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', default='word2vec',
                        choices=['tfidf', 'word2vec', 'fasttext', 'glove'])
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fr_corpus, it_corpus, _ = load_tatoeba_tsv(max_sentences=10000)
    fr_tokens = preprocess_text(fr_corpus, language='french')
    it_tokens = preprocess_text(it_corpus, language='italian')
    fr_vocab = set(w for s in fr_tokens for w in s)
    it_vocab = set(w for s in it_tokens for w in s)

    bilingual_dict = load_bilingual_dictionary('fr', 'it')
    embedding_types = ['word2vec', 'fasttext', 'glove'] if args.all else [args.embedding]

    for emb in embedding_types:
        print(f"\n{'='*60}\nANALYSIS: {emb.upper()}\n{'='*60}")
        try:
            fr_v, it_v = load_word_vectors(emb)
            analyze_synonyms_antonyms(fr_v, 'french', emb)
            analyze_synonyms_antonyms(it_v, 'italian', emb)
            if bilingual_dict:
                analyze_cognates(fr_v, it_v, bilingual_dict, emb)
            analyze_polysemy(fr_v, 'french', emb)
            analyze_polysemy(it_v, 'italian', emb)
            analyze_oov(fr_vocab, it_vocab, emb)
        except Exception as e:
            print(f"  Error: {e}")