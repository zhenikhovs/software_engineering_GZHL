from transformers import MBartTokenizer, MBartForConditionalGeneration

model_name = "IlyaGusev/mbart_ru_sum_gazeta"
tokenizer = MBartTokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

article_text = """Hugging Face — это исследовательская лаборатория и центр искусственного интеллекта, который создал сообщество ученых, исследователей и энтузиастов. За короткий промежуток времени Hugging Face завоевала значительное присутствие в сфере искусственного интеллекта. Технологические гиганты, в том числе Google, Amazon и Nvidia, поддержали AI-стартап Hugging Face значительными инвестициями, в результате чего его оценка составила 4,5 миллиарда долларов.
В этом руководстве мы познакомим вас с трансформерами, LLM и тем, как библиотека Hugging Face играет важную роль в развитии сообщества искусственного интеллекта с открытым исходным кодом. Мы также рассмотрим основные функции Hugging Face, включая конвейеры, наборы данных, модели и многое другое, на практических примерах Python."""

input_ids = tokenizer(
    [article_text],
    max_length=600,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    no_repeat_ngram_size=4
)[0]

summary = tokenizer.decode(output_ids, skip_special_tokens=True)
print(summary)
