#!/usr/bin/env python
"""
Test script to debug fine-tuning effectiveness.

This script performs the debugging steps outlined in the issue:
1. Verify training really updated weights by checking eval_loss
2. Test the merged model in HuggingFace pipeline (before GGUF conversion)
3. Provide clear diagnostic information about training success
"""

import os
import sys
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline,
    TrainingArguments, Trainer
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM

# Configuration (matching train.py)
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Qwen3-4B-Instruct-2507")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./finetuned-qwen")
MERGED_DIR = os.environ.get("MERGED_DIR", "./finetuned-qwen-merged")
SYSTEM_PROMPT = "You are an AI assistant that can call tools. Keep answers short, direct, and avoid repetition."

def print_header(title):
    """Print a formatted header for each test section."""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def test_eval_loss():
    """Test 1: Verify training updated weights by checking eval_loss."""
    print_header("STEP 1: Check Training Effectiveness via Eval Loss")

    # Check if training outputs exist
    sft_dir = f"{OUTPUT_DIR}-sft"
    if not os.path.exists(sft_dir):
        print(f"❌ Training output directory not found: {sft_dir}")
        print("   Please run training first with: python train.py")
        return False

    try:
        print(f"📂 Loading trained model from: {sft_dir}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load the fine-tuned model
        model = AutoPeftModelForCausalLM.from_pretrained(
            sft_dir,
            device_map="auto",
            trust_remote_code=True
        )

        # Create a small test dataset similar to training data
        test_data = [
            {"instruction": "What's the return policy?",
             "response": "You can return items within 30 days for a full refund."},
            {"instruction": "Do you ship internationally?",
             "response": "Yes, we ship worldwide with an extra fee depending on location."},
            {"instruction": "How can I reset my password?",
             "response": "Go to your account settings, click 'Reset Password', and follow the instructions sent to your email."}
        ]

        def tokenize_function(batch):
            input_ids_list, attention_masks, labels_list = [], [], []
            for instr, resp in zip(batch["instruction"], batch["response"]):
                messages_prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instr},
                ]
                prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
                full_text = prompt_text + resp + tokenizer.eos_token

                full = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
                prompt_only = tokenizer(prompt_text, truncation=True, padding="max_length", max_length=512)

                labels = full["input_ids"].copy()
                prompt_len = sum(1 for m in prompt_only["attention_mask"] if m == 1)
                for i in range(min(prompt_len, len(labels))):
                    labels[i] = -100

                input_ids_list.append(full["input_ids"])
                attention_masks.append(full["attention_mask"])
                labels_list.append(labels)

            return {"input_ids": input_ids_list, "attention_mask": attention_masks, "labels": labels_list}

        # Create evaluation dataset
        test_dataset = Dataset.from_list(test_data)
        tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)

        # Set up trainer for evaluation
        training_args = TrainingArguments(
            output_dir="./temp_eval",
            per_device_eval_batch_size=1,
            logging_dir="./temp_logs",
            report_to=[],  # Disable wandb/tensorboard
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_test,
        )

        print("📊 Running evaluation...")
        eval_results = trainer.evaluate()
        eval_loss = eval_results.get('eval_loss', float('inf'))

        print(f"📈 Evaluation Loss: {eval_loss:.4f}")

        # Interpret the results
        if eval_loss > 10:
            print("❌ TRAINING LIKELY FAILED")
            print("   → Eval loss is very high (>10), suggesting random guessing")
            print("   → The model weights were not effectively updated")
            return False
        elif eval_loss > 3:
            print("⚠️  TRAINING PARTIALLY EFFECTIVE")
            print("   → Eval loss is moderate (3-10), some learning occurred but may need improvement")
            print("   → Consider more training epochs or better hyperparameters")
            return True
        else:
            print("✅ TRAINING APPEARS SUCCESSFUL")
            print("   → Eval loss is low (<3), indicating effective fine-tuning")
            print("   → Model weights have been meaningfully updated")
            return True

    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        return False

def test_huggingface_pipeline():
    """Test 2: Test the merged model using HuggingFace pipeline."""
    print_header("STEP 2: Test Merged Model with HuggingFace Pipeline")

    # Check if merged model exists
    if not os.path.exists(MERGED_DIR):
        print(f"❌ Merged model directory not found: {MERGED_DIR}")
        print("   Please ensure training completed and model was merged")
        return False

    try:
        print(f"📂 Loading merged model from: {MERGED_DIR}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_DIR,
            device_map="auto",
            trust_remote_code=True
        )

        print("🔧 Creating text generation pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )

        # Test questions related to training data
        test_questions = [
            "What is the refund policy?",
            "What's the return policy?",
            "How long do I have to return an item?",
            "Do you ship internationally?"
        ]

        print("🧪 Testing model responses...")
        all_correct = True

        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Test {i}/4 ---")
            print(f"❓ Question: {question}")

            try:
                # Generate response
                response = pipe(
                    question,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

                generated_text = response[0]['generated_text']
                # Extract only the generated part (after the input)
                answer = generated_text[len(question):].strip()
                print(f"🤖 Answer: {answer}")

                # Check if the answer contains expected keywords for return policy questions
                if any(keyword in question.lower() for keyword in ['return', 'refund']):
                    if '30 days' in answer and 'refund' in answer.lower():
                        print("✅ Correct answer detected!")
                    else:
                        print("❌ Expected '30 days' and 'refund' in the answer")
                        all_correct = False
                elif 'ship' in question.lower():
                    if any(word in answer.lower() for word in ['yes', 'worldwide', 'international', 'fee']):
                        print("✅ Reasonable shipping answer detected!")
                    else:
                        print("⚠️  Shipping answer may not be optimal")

            except Exception as e:
                print(f"❌ Error generating response: {str(e)}")
                all_correct = False

        print(f"\n📋 Summary:")
        if all_correct:
            print("✅ HUGGINGFACE PIPELINE TEST PASSED")
            print("   → The merged model produces expected responses")
            print("   → Fine-tuning appears to be working correctly")
            return True
        else:
            print("❌ HUGGINGFACE PIPELINE TEST FAILED")
            print("   → The model is not producing expected responses")
            print("   → This suggests a training or dataset issue")
            return False

    except Exception as e:
        print(f"❌ Error during HuggingFace pipeline test: {str(e)}")
        return False

def test_comparison_with_base():
    """Test 3: Compare responses with base model."""
    print_header("STEP 3: Compare with Base Model")

    try:
        print("🔄 Loading base model for comparison...")
        base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            trust_remote_code=True
        )
        base_pipe = pipeline(
            "text-generation",
            model=base_model,
            tokenizer=base_tokenizer,
            device_map="auto"
        )

        question = "What's the return policy?"
        print(f"❓ Test question: {question}")

        # Base model response
        print("\n🤖 Base model response:")
        base_response = base_pipe(
            question,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=base_tokenizer.eos_token_id
        )
        base_answer = base_response[0]['generated_text'][len(question):].strip()
        print(f"   {base_answer}")

        # Fine-tuned model response
        if os.path.exists(MERGED_DIR):
            print("\n🎯 Fine-tuned model response:")
            ft_tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR, trust_remote_code=True)
            ft_model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, device_map="auto", trust_remote_code=True)
            ft_pipe = pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer, device_map="auto")

            ft_response = ft_pipe(
                question,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=ft_tokenizer.eos_token_id
            )
            ft_answer = ft_response[0]['generated_text'][len(question):].strip()
            print(f"   {ft_answer}")

            # Analysis
            print("\n📊 Analysis:")
            if '30 days' in ft_answer.lower() and 'refund' in ft_answer.lower():
                print("✅ Fine-tuned model provides the trained response!")
                print("   → Training successfully modified model behavior")
                return True
            else:
                print("❌ Fine-tuned model doesn't provide the expected trained response")
                print("   → Training may not have been effective")
                return False
        else:
            print("⚠️  Fine-tuned model not available for comparison")
            return False

    except Exception as e:
        print(f"❌ Error during comparison test: {str(e)}")
        return False

def main():
    """Run all debugging tests."""
    print("🚀 FINE-TUNING DEBUGGING TEST SCRIPT")
    print("This script will help diagnose training effectiveness")

    results = {}

    # Run all tests
    results['eval_loss'] = test_eval_loss()
    results['huggingface_pipeline'] = test_huggingface_pipeline()
    results['base_comparison'] = test_comparison_with_base()

    # Final summary
    print_header("FINAL DIAGNOSIS")

    passed_tests = sum(results.values())
    total_tests = len(results)

    print(f"📊 Test Results: {passed_tests}/{total_tests} passed")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("   → Training appears to be working correctly")
        print("   → If Ollama still doesn't work, the issue is in conversion/quantization or Modelfile")
    elif passed_tests >= 2:
        print("⚠️  MOSTLY SUCCESSFUL")
        print("   → Training is working but may need refinement")
        print("   → Check specific test failures above for guidance")
    elif results.get('eval_loss', False):
        print("🔧 TRAINING WORKS, PIPELINE ISSUES")
        print("   → Model weights are updated (low eval loss)")
        print("   → But pipeline responses are not as expected")
        print("   → May need to adjust generation parameters or post-processing")
    else:
        print("❌ TRAINING FAILED")
        print("   → High eval loss indicates training didn't work")
        print("   → Check dataset, learning rate, or training configuration")

    print("\n🔍 Debugging Guide:")
    print("   • If eval_loss test fails → training/dataset issue")
    print("   • If HuggingFace test fails → model not learning expected responses")
    print("   • If both pass but Ollama fails → conversion/Modelfile issue")

if __name__ == "__main__":
    main()
