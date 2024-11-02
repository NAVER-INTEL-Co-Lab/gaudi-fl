import ollama
import pandas as pd
import argparse
from tqdm import tqdm

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Paraphrase medical abstracts in a CSV file with ollama.")
parser.add_argument('--input_file', type=str, default='./medical_tc_train', required=True,
                    help="Path to the input CSV file.")
parser.add_argument('--output_file', type=str, required=True, default='output',
                    help="Path to the output CSV file.")
parser.add_argument('--model_name', type=str, required=True,
                    help="Model name to be used for paraphrasing.")
args = parser.parse_args()


def paraphrase_text(text):
    for _ in range(5): 
        response = ollama.chat(model=args.model_name, messages=[
            {'role': 'user', 'content': f"Please paraphrase the following text:\n{text}"}
        ])

        paraphrased_text = response['message']['content'].replace('\n', '')

        start_phrase = "Here's a paraphrased version of the text:"
        if start_phrase in paraphrased_text:
            paraphrased_text = paraphrased_text.split(
                start_phrase, 1)[-1].strip()

        if paraphrased_text:
            print(paraphrased_text)
            return paraphrased_text

    return None

df = pd.read_csv(args.input_file)

paraphrased_abstracts = []
for abstract in tqdm(df['medical_abstract'], desc="Paraphrasing Abstracts"):
    paraphrased_text = paraphrase_text(abstract)
    paraphrased_abstracts.append(paraphrased_text)

df['paraphrased_abstract'] = paraphrased_abstracts

df.to_csv(args.output_file, index=False)
print(f"Paraphrased abstracts saved to {args.output_file}")
