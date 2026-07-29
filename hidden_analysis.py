import json
import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse


def generate_math_data(data_dir, data_path):
    correct, incorrect = [], []
    with open(data_path) as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]
    with open(f"{data_dir}/math_eval.jsonl") as f:
        eval = f.readlines()
        eval = [json.loads(line) for line in eval]
    
    data = data[:len(eval)]
    for d, e in zip(data, eval):
        local_correct, local_incorrect = [], []
        prompt = e["prompt"]
        assert d["problem"] == e["problem"]
        for o, c in zip(e["model_generation"], e["all_eval"]):
            if c:
                local_correct.append({"prompt":prompt, "response":o, "level":d["level"], "gt":e["answer"]})
            else:
                local_incorrect.append({"prompt":prompt, "response":o, "level":d["level"], "gt":e["answer"]})
        correct.extend(local_correct)
        incorrect.extend(local_incorrect)
    return correct, incorrect
    




# Keyword sets for classifying reasoning steps as "check" or "switch".
# "math": upstream SEAL's lists (v_math), except the dead "think differenly"
#   typo is fixed to "think differently".
# "code": code-adapted lists (v_code) — all "contains" matching (wait/
#   alternatively promoted from prefix to contains), plus code-specific cues.
# "logic": LogiQA-adapted lists derived from English unsteered LogiQA traces
#   (Andwwy/v_code-SEAL thought_tags.KEYWORD_SETS["logic"]). Reflection is
#   checked before transition, same priority as generate_index below.
KEYWORD_SETS = {
    "math": {
        "check_words": ["verify", "make sure", "hold on", "think again", "'s correct", "'s incorrect", "Let me check", "seems right"],
        "check_prefix": ["Wait"],
        "switch_words": ["think differently", "another way", "another approach", "another method", "another solution", "another strategy", "another technique"],
        "switch_prefix": ["Alternatively"],
    },
    "code": {
        "check_words": ["wait", "but wait", "verify", "make sure", "hold on", "think again", "'s correct", "'s incorrect", "let me check", "seems right", "hmm", "what if", "double-check", "recheck", "edge case"],
        "check_prefix": [],
        "switch_words": ["alternatively", "another way", "another approach", "another method", "another solution", "another strategy", "another technique", "think differently", "instead", "a better way", "rethink", "start over", "on second thought"],
        "switch_prefix": [],
    },
    "logic": {
        "check_words": [
            "wait",
            "think again",
            "missing something",
            "overcomplicating",
            "stuck",
            "hmm",
            "let me check",
            "not making progress",
            "but wait",
            "double-check",
            "'s correct",
            "'s incorrect",
        ],
        "check_prefix": [],
        "switch_words": [
            "alternatively",
            "let me try to think",
            "think differently",
            "think of it differently",
            "another approach",
        ],
        "switch_prefix": [],
    },
}

def generate_index(text, tokenizer, split_id, think_only=True, keywords="math"):

    kw = KEYWORD_SETS[keywords]
    check_words = kw["check_words"]
    check_prefix = kw["check_prefix"]
    switch_words = kw["switch_words"]
    switch_prefix = kw["switch_prefix"]

    tokens = tokenizer.encode(text)
    if think_only:
        think_begin_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        if think_begin_id not in tokens:
            return [], [], []
    
        start = tokens.index(think_begin_id)+1
        if think_end_id not in tokens[start:]:
            end=len(tokens)
        else:
            end = tokens.index(think_end_id, start)
        think_tokens = tokens[start:end]
    else:
        think_tokens = tokens
        start = 0

    index = [i for i, t in enumerate(think_tokens) if t in split_id] + [len(think_tokens)]
    step_index = []
    check_index=[]
    switch_index=[]

    for i in range(len(index)-1):
        step_index.append(index[i]+start)
        step = think_tokens[index[i]+1:index[i+1]]
        step = tokenizer.decode(step).strip(" ").strip("\n")
        if any([step.lower().startswith(p.lower()) for p in check_prefix]) or any([w.lower() in step.lower() for w in check_words]):
                check_index.append(i)
        elif any([step.lower().startswith(p.lower()) for p in switch_prefix]) or any([w.lower() in step.lower() for w in switch_words]):
            switch_index.append(i)
    return step_index, check_index, switch_index

def generate(model_path, data, save_dir, keep_layers=None, keywords="math"):
    think_only = "deepseek" in model_path.lower()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    vocab = tokenizer.get_vocab()
    split_id = [vocab[token] for token in vocab.keys() if "ĊĊ" in token]

    prompts = [d["prompt"]+d["response"] for d in data]

    layer_num = model.config.num_hidden_layers+1
    # Only the steering layer is needed downstream; storing all layers makes
    # hidden.pt ~num_layers x larger. keep_layers=None preserves original behavior.
    if keep_layers is None:
        keep_layers = list(range(layer_num))
    hidden_dict=[{} for _ in range(layer_num)]

    for k, p in tqdm(enumerate(prompts), total=len(prompts)):
        tokenized_batch = tokenizer([p], return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with torch.no_grad():
            base_model = getattr(model, "model", model)
            output = base_model(**tokenized_batch, output_hidden_states=True, use_cache=False)
            hidden_states = output.hidden_states
        layer_num = len(hidden_states)
        step_index, check_index, switch_index = generate_index(p, tokenizer, split_id, think_only=think_only, keywords=keywords)
        step_index = torch.LongTensor(step_index)
        check_index = torch.LongTensor(check_index)
        switch_index = torch.LongTensor(switch_index)
        for i in keep_layers:
            h = hidden_states[i][0].detach().cpu()
            step_h = h[step_index]
            hidden_dict[i][k] = {"step":step_h, "check_index": check_index, "switch_index": switch_index}
        del hidden_states
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden_dict, f"{save_dir}/hidden.pt")
    json.dump(prompts, open(f"{save_dir}/prompts.json", "w"))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--type", type=str, default="correct", choices=["correct", "incorrect"])
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--keep_layers", type=int, nargs="+", default=None,
                        help="Only extract/save these hidden-layer indices (default: all). "
                             "Pass the steering layer to shrink hidden.pt ~num_layers x.")
    parser.add_argument("--keywords", type=str, default="math", choices=sorted(KEYWORD_SETS),
                        help="Check/switch keyword set: 'math' (upstream SEAL, v_math), "
                             "'code' (code-adapted, v_code), or 'logic' (LogiQA-adapted).")
    args = parser.parse_args()
    correct, incorrect = generate_math_data(data_dir=args.data_dir, data_path=args.data_path)
    if args.type == "correct":
        data = correct
    else:
        data = incorrect
    save_dir = f"{args.data_dir}/hidden_{args.type}"
    if args.start != -1:
        data = data[args.start:]
        if args.sample != -1:
            data = data[:args.sample]
            save_dir = f"{save_dir}_{args.start}_{args.start+args.sample}"
        else:
            save_dir = f"{save_dir}_{args.start}_-1"
    print(save_dir)
    generate(args.model_path, data, save_dir, keep_layers=args.keep_layers, keywords=args.keywords)