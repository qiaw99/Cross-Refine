# Demonstrations used for Cross-Refine
## NLP Use Cases
We select three NLP use cases:
- Natural Language Inference (e-SNLI)
  - Sentence 1, Sentence 2
- Commonsense Question Answering (ECQA)
  - Text, Choice
- Fact Checking (HealthFC)
  - Claim, Evidence

## Structure of Demonstration File

```json
[
  {
    "id": "id",
    "text_field_1": "text_field_1",
    "text_field_2": "text_field_2",
    "label": "label",
    "original_explanation": "original_explanation",
    "need_improve": "need_improve",
    "feedback": "feedback",
    "critic_explanation": "critic_explanation",
    "refined_explanation": "refined_explanation" 
  }
]
```

For text_field: please reference the first section.

## Prompt for Cross-Refine
Cross-Refine has three phrases:
1. The generator will output an initial explanation based on the input;
2. The critic will give feedback on the provided explanation from generator and try to suggest another explanation.
3. The generator will consider 1) the initial explanation (generator); 2) feedback (critic); 3) suggested explanation (generator); to try to refine its explanation

Here I will select ECQA as example:

#### 1. Initial Explanation Generation
```python
prompt_template = f"Question: {question}\n"
prompt_template += f"Choices: {choices}\n"
prompt_template += f"Based on the given question, your prediction is {prediction}. Please provide a reason why the answer is correct."
```

#### 2. Feedback & Suggested Explanation
```python
prompt_template = f"Question: {question}\n"
prompt_template += f"Choices: {choices}\n"
prompt_template += f"Initial explanation: {explanation}\n"
prompt_template += f"Considering the provided question and choices, please give feedback on the provided explanation and sugguest a new explanation."
```

#### 3. Explanation Refinement
```python
prompt_template = f"Question: {question}\n"
prompt_template += f"Choices: {choices}\n"
prompt_template += f"Initial explanation: {explanation}\n"
prompt_template += f"Feedback: {feedback}\n"
prompt_template += f"Suggested explanation: {suggested_explanation}\n"
prompt_template += f"You are an excellent reasoner. Please try to improve and refine the given initial explanation by considering feedback and suggested explanation."
```

## Evaluation
Once demonstrations are created, please do followings:
1. Determine whether the original explanation has already very good quality. If it is the case, please 1) set the `need_improve=false`; 2) remove `feedback`, `critic_explanation`, `refined_explanation`
2. Check if the provided feedback makes sense
3. Check whether the refined explanation is better than the initial explanation