"""
Prepare Bilingual Dictionary
Download MUSE FR-IT dictionary
"""
import os
import requests
from fr_ita.config import DATA_DIR


def download_muse_dictionary(lang1='fr', lang2='it'):
    """
    Download MUSE bilingual dictionary

    Args:
        lang1 (str): Source language
        lang2 (str): Target language

    Returns:
        str: Path to downloaded dictionary
    """
    dict_dir = os.path.join(DATA_DIR, 'muse_dictionaries')
    os.makedirs(dict_dir, exist_ok=True)

    dict_file = os.path.join(dict_dir, f'{lang1}-{lang2}.txt')

    if os.path.exists(dict_file):
        print(f"Dictionary already exists: {dict_file}")
        return dict_file

    url = f"https://dl.fbaipublicfiles.com/arrival/dictionaries/{lang1}-{lang2}.txt"

    print(f"Downloading MUSE dictionary...")
    print(f"URL: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()

        with open(dict_file, 'wb') as f:
            f.write(response.content)

        print(f"Dictionary saved to: {dict_file}")
        return dict_file

    except Exception as e:
        print(f"Error: {e}")
        return None


def show_dictionary_samples(bilingual_dict, n=10):
    """
    Display sample word pairs

    Args:
        bilingual_dict (dict): Bilingual dictionary
        n (int): Number of samples to show
    """
    print(f"\nSample word pairs:")
    for i, (src, tgt) in enumerate(list(bilingual_dict.items())[:n]):
        print(f"  {src} -> {tgt}")


if __name__ == "__main__":
    from fr_ita.utils import load_bilingual_dictionary

    print("MUSE BILINGUAL DICTIONARY PREPARATION")
    print()

    # download
    dict_file = download_muse_dictionary('fr', 'it')

    if dict_file is None:
        print("Failed to download dictionary")
        exit(1)

    # load via shared utils
    bilingual_dict = load_bilingual_dictionary('fr', 'it')

    # show samples
    show_dictionary_samples(bilingual_dict, n=20)

    print(f"\nDictionary ready!")
    print(f"Total word pairs: {len(bilingual_dict)}")