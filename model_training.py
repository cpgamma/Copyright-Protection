from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

def train_full_model(model_name, dataset_path, output_dir, seed):
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", 
        device_map="auto"    
    )
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=[dataset_path], split="train")

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,  
            padding="max_length"
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )


    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,   # Adjust based on GPU memory
        gradient_accumulation_steps=16,  
        learning_rate=1e-5,              
        num_train_epochs=10,              # Adjust based on dataset size
        save_strategy="epoch",
        save_total_limit=1,
        bf16=False,
        half_precision_backend=None,
        fp16=False, 
        seed=seed,                        # Use different seeds for the two models
        dataloader_drop_last=True,
        optim="adamw_torch",
        report_to="none",
        logging_steps=10,
        no_cuda=False,
        remove_unused_columns=False,     
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
    )

    print(f"Starting training with seed {seed}...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return model, tokenizer
