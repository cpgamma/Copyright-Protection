import torch
import torch.nn.functional as F
from utils import process_text


def cp_delta_generate(model_q1, model_q2, tokenizer, prompt, max_new_tokens, delta_type):

    model_q1 = model_q1.half()
    model_q2 = model_q2.half()

    inputs = tokenizer(prompt, return_tensors="pt").to(model_q1.device)
    input_ids = inputs.input_ids
    generated_ids = input_ids.clone()

    past_key_values_q1 = None
    past_key_values_q2 = None

    for step in range(max_new_tokens):
 
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            if past_key_values_q1 is not None:
                current_ids = generated_ids[:, -1].unsqueeze(-1)
            else:
                current_ids = generated_ids

            q1_outputs = model_q1(
                current_ids,
                past_key_values=past_key_values_q1,
                use_cache=True
            )
            q2_outputs = model_q2(
                current_ids,
                past_key_values=past_key_values_q2,
                use_cache=True
            )

            past_key_values_q1 = q1_outputs.past_key_values
            past_key_values_q2 = q2_outputs.past_key_values

  
            q1_probs = F.softmax(q1_outputs.logits[:, -1, :], dim=-1)
            q2_probs = F.softmax(q2_outputs.logits[:, -1, :], dim=-1)


            if delta_type == "max":
                combined_probs = torch.min(q1_probs, q2_probs)
                tv_distance = 0.5 * torch.sum(torch.abs(q1_probs - q2_probs))
                z_x = 1 - tv_distance
            else: 
                combined_probs = torch.sqrt(q1_probs * q2_probs)
                hellinger_squared = 1 - torch.sum(torch.sqrt(q1_probs * q2_probs))
                z_x = 1 - hellinger_squared

            normalized_probs = combined_probs / z_x

        next_token = torch.multinomial(normalized_probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return process_text(generated_text, prompt)



def single_model_gamma_clipped_dynamic(model, tokenizer, prompt, max_new_tokens, gamma):

    model = model.half()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    generated_ids = input_ids.clone()
    past_key_values = None

    for step in range(max_new_tokens):
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            
            current_ids = generated_ids[:, -1].unsqueeze(-1) if past_key_values else generated_ids
            outputs = model(current_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
            max_prob = torch.max(probs).item()
            epsilon = gamma * max_prob
            clipped_probs = torch.clamp(probs, max=epsilon)
            normalized_probs = clipped_probs / clipped_probs.sum()

        next_token = torch.multinomial(normalized_probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return process_text(generated_text, prompt)


