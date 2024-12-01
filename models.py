from huggingface_hub import login
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json

from prompts import *
from data import get_trsav1, get_ttc4900

login(os.getEnv("hf_token"))

def ytu_cosmos():
    import transformers
    import torch

    model_id = "ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "Sen bir yapay zeka asistanısın. Kullanıcı sana bir görev verecek. Amacın görevi olabildiğince sadık bir şekilde tamamlamak. Görevi yerine getirirken adım adım düşün ve adımlarını gerekçelendir."},
        {"role": "user", "content": "Soru: Bir arabanın deposu 60 litre benzin alabiliyor. Araba her 100 kilometrede 8 litre benzin tüketiyor. Depo tamamen doluyken araba kaç kilometre yol alabilir?"},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"][-1])

def kanarya():
    
    tokenizer = AutoTokenizer.from_pretrained("asafaya/kanarya-2b")
    model = AutoModelForCausalLM.from_pretrained("asafaya/kanarya-2b", torch_dtype=torch.float16).bfloat16().to("cuda")

    print(model.device)  # "cuda" yazmalı

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1" 
    
    texts = test_data_trsav1["review"].tolist()
    system_prompt = prompt_1_zero_shot_trsav1
    
    batch_size = 16 
    num_batches = len(texts) // batch_size + int(len(texts) % batch_size > 0)

    predictions = []

    model.resize_token_embeddings(len(tokenizer))
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    for i in range(num_batches):
        batch_texts = texts[i * batch_size:(i + 1) * batch_size] 
        
        batch_inputs = [f"{system_prompt}\n\nMetin: {text}" for text in batch_texts]
        print("[Batch input]", batch_inputs)

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

        print(next(model.parameters()).device)  # "cuda:0" yazmalı
        model = model.to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200, num_beams=1)
            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            print(decoded_outputs)

        batch_predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        predictions.extend(batch_predictions)  # Sonuçları birleştir

        print(f"Batch {i + 1}/{num_batches} tamamlandı.")
        

    # Prediction sonuçlarını yazmak için bir JSON dosyası oluştur
    output_file = "predictions_zeroshot_kanarya.json"

    # JSON dosyasına yazma
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print(f"Predictions başarıyla {output_file} dosyasına yazıldı.")

def verius():
    from unsloth import FastLanguageModel
    import torch
    from torch.utils.data import DataLoader, Dataset
    from unsloth import FastLanguageModel

    max_seq_len = 512
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="VeriUs/VeriUS-LLM-8b-v0.2",
        max_seq_length=max_seq_len,
        dtype=None,
        device_map="auto")
    FastLanguageModel.for_inference(model)  
    model.to("cuda")

    prompt_template = """Aşağıda, görevini açıklayan bir talimat ve daha fazla bağlam sağlayan bir girdi verilmiştir. İsteği uygun bir şekilde tamamlayan bir yanıt yaz.

    ### Talimat:
    {}

    ### Girdi:
    {}

    ### Yanıt:
    """

    class PromptDataset(Dataset):
        def __init__(self, instructions, inputs):
            self.instructions = instructions
            self.inputs = inputs

            print("inputs length:", len(inputs))
            print("instructions length:", len(instructions))

        def __len__(self):
            return len(self.instructions)

        def __getitem__(self, idx):
            return self.instructions[idx], self.inputs[idx]

    instructions = [prompt_1_five_shot_trsav1] * 1000
    inputs = test_data_trsav1["review"].tolist()  
    dataset = PromptDataset(instructions, inputs)

    batch_size = 8 
    dataloader = DataLoader(dataset, batch_size=batch_size)

    def generate_outputs_batch(dataloader, max_seq_len):
        all_outputs = []

        i=0
        for batch in dataloader:

            print(i, ". batch")
            instructions, inputs = batch
            prompts = [prompt_template.format(inst, inp) for inst, inp in zip(instructions, inputs)]
            input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

            outputs = model.generate(**input_ids,max_new_tokens=20 , do_sample=True)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            all_outputs.extend(decoded_outputs)
            i += 1

        return all_outputs

    responses = generate_outputs_batch(dataloader, max_seq_len)
    print(responses[:5]) 

    import json

    # JSON'a kaydet
    output_file = "trsav1_5shotlama.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=4) 

    print(f"Çıktılar {output_file} dosyasına kaydedildi.")

if __name__ == "__main__":
    #ytu_cosmos()
    #kanarya()
    turkcell()
