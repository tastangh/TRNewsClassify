from huggingface_hub import login
login("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")

import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def google_gemma():

    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-9b-it",
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    messages = [
        {"role": "user", 
        "content": "Who are you? Please, answer in pirate-speak."},
    ]

    outputs = pipe(messages, max_new_tokens=256)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    print(assistant_response)
    # Ahoy, matey! I be Gemma, a digital scallywag, a language-slingin' parrot of the digital seas. I be here to help ye with yer wordy woes, answer yer questions, and spin ye yarns of the digital world.  So, what be yer pleasure, eh? 

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

if __name__ == "__main__":
    ytu_cosmos()