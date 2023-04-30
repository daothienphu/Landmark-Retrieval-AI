from datasets import load_dataset
from torchvision.transforms import ToTensor, Compose,CenterCrop,Normalize
from transformers import AutoModel
import torch
import numpy as np

from PIL import Image
import os
from pathlib import Path
import time

def preprocess_function(examples):
    examples['pixel_values'] = [pipe(image.convert('RGB')) for image in examples['image']]
    examples['label'] = [os.path.splitext(os.path.basename(im.filename))[0] for im in examples['image']]
    return examples 

def tensor_from_image(image):
    return pipe(image.convert('RGB'))

def get_embeddings(examples):
    with torch.no_grad():
        image = examples['pixel_values'].to('cuda')
        embeddings = get_embedding(image)
    return {'embeddings':embeddings}

def get_embedding(image):
    outputs = model(image)
    embeddings = outputs.pooler_output
    return embeddings

def compute_sim(emb_one,emb_two):
    return torch.cosine_similarity(emb_one,emb_two,dim=1).cpu().numpy().tolist()

def fetch_sim_results(image, top_k=20, get_confidence=True):
    image = image.unsqueeze(0).to('cuda')
    new_batch = {'pixel_values':image}
    with torch.no_grad():
        new_batch = model(**new_batch).pooler_output
    
    sim = compute_sim(new_batch, all_embeddings)
    similarity_mapping = dict(zip(all_labels,sim))
    
    sorted_similarity = dict(sorted(similarity_mapping.items(),key=lambda x:x[1],reverse=True))
    
    id_entries = list(sorted_similarity.keys())[:top_k]
    res = []

    if get_confidence:
        confidences = list(sorted_similarity.values())[:top_k]
        res = [f"{label}: {confidence * 100:.2f}%" for label, confidence in zip(id_entries, confidences)]
    else: 
        res = [f"{label}" for label in id_entries]
    
    return res

#Set up model
print("======================= LOADING MODEL =======================")
checkpoint = './data/checkpoint-500'
model = AutoModel.from_pretrained(checkpoint).to('cuda')
pipe = Compose([CenterCrop(224), ToTensor(), Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
print("=============================================================")

#Load database
print("\n====================== LOADING DATABASE =====================")
database = load_dataset('imagefolder', data_dir="./data/database", split="train")
database = database.map(preprocess_function, batched=True, batch_size=32)
database.set_format(type='torch', columns=['pixel_values', 'label'])
print("=============================================================")

#Make embeddings and labels
print("\n===================== MAKING EMBEDDINGS =====================")
candidate_embeddings = database.map(get_embeddings, batched=True, batch_size=32)
all_embeddings = np.array(candidate_embeddings['embeddings'])
all_embeddings = torch.from_numpy(all_embeddings).to('cuda')
all_labels = np.array(candidate_embeddings['label'])
print("=============================================================")


def run_model(task='test', image=None, csv_filename="results.csv"):
    #For testing with 1 query
    if task == 'test':
        if image == None:
            image = Image.open(Path("./data/query/0a739b9340453ef7.jpg"))
        tensor = tensor_from_image(image)
        
        print("\n===================== PROCESSING QUERY ======================")
        query = os.path.splitext(os.path.basename(image.filename))[0]
        print(f"Query ID: {query}")

        process_time = time.time()
        model_res = fetch_sim_results(tensor, 10)
        process_time = time.time() - process_time
        print(f"Processing time: {process_time:.9f} seconds")
        
        print("Results:")
        for l in model_res:
            print(f"\t{l}")
        print("=============================================================")


    #For generating the CSV file
    elif task == 'csv':
        csv_file = open(csv_filename, "w")

        #Load queries
        print("\n====================== LOADING QUERIES ======================")
        queries = load_dataset('imagefolder', data_dir="./data/query", split="train")
        queries = queries.map(preprocess_function, batched=True, batch_size=32)
        queries.set_format(type='torch', columns=['pixel_values', 'label'])
        print("=============================================================")
        

        print("\n====================== GENERATING CSV =======================")
        csv_file.write("Query ID, Top 1 ID, Top 2 ID, Top 3 ID, Top 4 ID, Top 5 ID, Top 6 ID, Top 7 ID, Top 8 ID, Top 9 ID, Top 10 ID,\n")
        for _, i in enumerate(queries):
            line = f"{i['label']},"
            
            model_res = fetch_sim_results(i['pixel_values'], top_k=10, get_confidence=False)
            for r in model_res:
                line += f" {r},"

            line += "\n"
            csv_file.write(line)
        csv_file.close()
        print(f"CSV generated at: {csv_filename}")
        print("=============================================================")

    #For linking with the UI
    elif task == 'app':    
        if image == None:
            print("Image is None")
            return None
        tensor = tensor_from_image(image)
        
        print("\n===================== PROCESSING QUERY ======================")
        query = os.path.splitext(os.path.basename(image.filename))[0]
        print(f"Query ID: {query}")

        process_time = time.time()
        model_res = fetch_sim_results(tensor, 20)
        process_time = time.time() - process_time
        print(f"Processing time: {process_time:.9f} seconds")
        
        print("Results:")
        for l in model_res:
            print(f"\t{l}")
        print("=============================================================")

        res = {}
        res["time"] = process_time
        res["res"] = model_res
        return res
    
    print("\nTask DONE")

if __name__ == "__main__":
    run_model(task='test')
    run_model(task='csv')