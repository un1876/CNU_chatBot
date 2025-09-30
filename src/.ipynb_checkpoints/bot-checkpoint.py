import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch # Roberta 모델을 직접 사용할 경우 필요할 수 있습니다.

class QASystem:
    def __init__(self, roberta_model_path, kogpt2_finetuned_model_path):
        try:
            print(f"Loading Roberta model from: {roberta_model_path}")
            self.roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_path)
            self.roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_path)
            print("Roberta model loaded successfully.")

            print(f"Loading KoGPT2 finetuned model from: {kogpt2_finetuned_model_path}")
            self.kogpt2_tokenizer = AutoTokenizer.from_pretrained(kogpt2_finetuned_model_path)
            self.kogpt2_model = AutoModelForCausalLM.from_pretrained(kogpt2_finetuned_model_path)

            self.answer_generator = pipeline(
                "text-generation",
                model=self.kogpt2_model,
                tokenizer=self.kogpt2_tokenizer
            )
            print("KoGPT2 finetuned model loaded successfully.")

        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def process_question(self, input_json_str):
        try:
            input_data = json.loads(input_json_str)
            user_question = input_data.get("user_question")

            if not user_question:
                return json.dumps({"error": "No 'user_question' found in the input JSON."})

            # Roberta 모델을 사용하여 주제 분류가 필요한 경우 이 부분을 활성화하고 사용하세요.
            # inputs = self.roberta_tokenizer(user_question, return_tensors="pt")
            # outputs = self.roberta_model(**inputs)
            # predictions = torch.argmax(outputs.logits, dim=-1)
            # classified_topic = self.roberta_model.config.id2label[predictions.item()]
            # print(f"Classified topic: {classified_topic}")

            print(f"Processing question: '{user_question}'")

            generated_text = self.answer_generator(
                user_question,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.kogpt2_tokenizer.eos_token_id
            )

            answer_text = generated_text[0]['generated_text'].replace(user_question, "").strip()

            if not answer_text or len(answer_text) < 5:
                answer_text = "죄송합니다. 질문에 대한 답변을 생성할 수 없습니다. 더 구체적으로 질문해주세요."

            output_data = {
                "user_question": user_question,
                "bot": answer_text
            }
            return json.dumps(output_data, ensure_ascii=False)

        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON format. Please provide a valid JSON string."})
        except Exception as e:
            return json.dumps({"error": f"An error occurred during processing: {str(e)}"})

if __name__ == "__main__":
    roberta_model_path = "roberta_folder"
    kogpt2_finetuned_model_path = "./kogpt2-finetuned"

    try:
        qa_system = QASystem(roberta_model_path, kogpt2_finetuned_model_path)

        test_question_json = '{"user_question": "우리나라의 수도는 어디야?"}'
        response_json = qa_system.process_question(test_question_json)
        print("\n--- Test 1 Result ---")
        print(response_json)
        print(json.loads(response_json)["bot"])

        test_question_json_2 = '{"user_question": "오늘 날씨는 어때?"}'
        response_json_2 = qa_system.process_question(test_question_json_2)
        print("\n--- Test 2 Result ---")
        print(response_json_2)
        print(json.loads(response_json_2)["bot"])

        test_question_json_3 = '{"invalid_key": "이건 잘못된 질문"}'
        response_json_3 = qa_system.process_question(test_question_json_3)
        print("\n--- Test 3 (Invalid) Result ---")
        print(response_json_3)

    except Exception as e:
        print(f"\nAn error occurred during QA system initialization or testing: {e}")