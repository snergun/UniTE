if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str, required=True,
                        help="Path to the dataset file (e.g., datasets/GSM/train.jsonl)")
    parser.add_argument("--prompts", type=str, default="datasets/GSM/gsm_prompt.txt",
                        help="Path to the prompt file")
    parser.add_argument("--model_path1", type=str, required=True, help="Path to Model 1")
    parser.add_argument("--model_path2", type=str, required=True, help="Path to Model 2")
    parser.add_argument("--output_file", type=str, default="ensemble_data.pt",
                        help="Where to save the collected tensors")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=50)

    args = parser.parse_args()

    accelerator = Accelerator()

    # Device Setup (Same as Unite)
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Tokenizers
    tokenizer1 = AutoTokenizer.from_pretrained(args.model_path1)
    tokenizer2 = AutoTokenizer.from_pretrained(args.model_path2)
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer2.pad_token = tokenizer2.eos_token

    # Load Models (Eval Mode)
    print(f"Loading Model 1: {args.model_path1}...")
    model1 = AutoModelForCausalLM.from_pretrained(
        args.model_path1, 
        device_map=device1,
        torch_dtype=torch.float16
    ).eval()

    print(f"Loading Model 2: {args.model_path2}...")
    model2 = AutoModelForCausalLM.from_pretrained(
        args.model_path2, 
        device_map=device2,
        torch_dtype=torch.float16
    ).eval()

    # Load Dataset & Select Collate Function
    # Logic borrowed from unite2.py lines 350-362
    dataset = load_dataset("json", data_files=args.test_set)['train']
    
    collate_fn = None
    # You need to import these from utils.collate_fun or define them locally if you prefer
    from utils.collate_fun import piqa_collate_fn, arc_collate_fn
    # Assuming gsm/qa collate functions are defined in your script or imported from unite2 logic
    # (gsm_collate_fn and qa_collate_fn were in the provided unite2.py snippet)
    
    if 'gsm' in args.test_set.lower():
        collate_fn = gsm_collate_fn
        # Note: UniTE loads prompt file content into a global variable inside main
        # You might need to adjust gsm_collate_fn to accept the prompt string or make it global here
        global prompt_complex
        if os.path.exists(args.prompts):
            prompt_complex = open(args.prompts, "r", encoding="utf-8").read()
        else:
            prompt_complex = "" # Fallback
            
    elif 'triviaqa' in args.test_set.lower() or 'nq' in args.test_set.lower():
        collate_fn = qa_collate_fn
        global prompt_complex
        if os.path.exists(args.prompts):
            prompt_complex = open(args.prompts, "r", encoding="utf-8").read()
        else:
            prompt_complex = ""

    elif 'arc' in args.test_set.lower():
        collate_fn = arc_collate_fn
    elif 'piqa' in args.test_set.lower():
        collate_fn = piqa_collate_fn
    else:
        raise ValueError(f"Unknown dataset type in path: {args.test_set}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.per_device_batch_size, 
        collate_fn=collate_fn, 
        num_workers=1
    )
    
    # Run Collection
    collect_data(
        args, 
        model1, model2, 
        tokenizer1, tokenizer2, 
        dataloader, 
        device1, device2
    )