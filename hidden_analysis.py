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
# "logic": derived from real LogiQA reasoning traces (300-example probe, then a
#   600-example round at a lower frequency floor to check for rarer patterns,
#   both on DeepSeek-R1-Distill-Qwen-1.5B via scripts/discover_logic_keywords.sh
#   + logic_keyword_coverage.py), not invented up front. check_words extends
#   math/code's checking vocabulary with phrasing this model actually uses for
#   LogiQA: "hmm" and substring "wait" (both already precedented in "code"'s
#   set, and validated here -- "wait" often appears mid-step, not just as a
#   step-opening prefix, e.g. "...a bit ambiguous. Wait, the question is a bit
#   tricky."); "not sure"/"let me re-examine"/"making a mistake"/"made a
#   mistake"/"think again" are hedging/re-verification language that inherently
#   signals hesitation regardless of context ("I'm not sure if C is a valid
#   conclusion", "Let me re-examine each option", "I think I'm making a mistake
#   here", "So maybe I made a mistake in the setup", "I'm misinterpreting. Let
#   me think again" -- "think again" is math's own validated phrase, just
#   carried over here after turning up in real LogiQA traces too). Deliberately
#   NOT added, after
#   reviewing real samples: "is correct"/"is incorrect"/"might not be" were
#   tried and dropped -- unlike "wait"/"hmm", these are just truth-value
#   statements with no inherent hesitation signal, and show up constantly in
#   ordinary option evaluation ("Option B is correct" is normal execution, not
#   self-correction -- the same content-level pattern rejected below for
#   switch). Also dropped for the same content-vs-reflection reason: bare
#   "maybe"/"seems"/"might not" (without "be") and "doesn't"/"looking at" --
#   overwhelmingly content-level speculation about the passage/argument, not
#   the model doubting its own reasoning.
#   switch_words/switch_prefix reuse math's own vocabulary as-is (not code's
#   expanded version, which includes several phrases -- "instead", "rethink",
#   "start over" -- with no specific evidence in LogiQA data): an invented
#   "Option A/B/C/D:" step-prefix was tried first and dropped after precision
#   review showed it mostly tagged ordinary content-level analysis ("Option B:
#   ... However, the passage doesn't mention X, so this doesn't fit"), not the
#   model doubting itself. But math's original "alternatively"/"another
#   approach"-style phrasing does measurably occur here too (~2% tag rate when
#   math's keywords were applied to this same LogiQA data), so it's kept as
#   real, if rare, signal rather than zeroed out or replaced with something
#   invented.
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
        "check_words": ["hmm", "wait", "not sure", "let me re-examine", "making a mistake", "made a mistake", "think again"],
        "check_prefix": [],
        "switch_words": ["think differently", "another way", "another approach", "another method", "another solution", "another strategy", "another technique"],
        "switch_prefix": ["Alternatively"],
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
                        help="Check/switch keyword set: 'math' (upstream SEAL, v_math) or "
                             "'code' (code-adapted, v_code).")
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