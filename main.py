import streamlit as st
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_summ_text(textForSumm):
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(
        [WHITESPACE_HANDLER(textForSumm)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return summary


st.header("Краткое содержание текста")
st.subheader("(Summarization text)")
text = st.text_area('Полный текст для сжатия:',  key='textInput')
col1, col2 = st.columns(2)
with col1:
    clicked = st.button('Сжать текст!', on_click=summ_text)
def on_fill_click():
    exampleText = 'Цифровизация – это социально-экономическая трансформация, которую вызовет массовое внедрение и усвоение новых технологий создания, обработки, анализа и передачи информации. Сельское хозяйство становится одним из основных потребителей цифровых технологий. Генерация большого объема данных разнообразными датчиками в теплицах, полях, фермах и других производственных площадках особенно актуализирует применение цифровых технологий анализа агроэкономических данных. Широкое применение инструментов и методов Data Science, машинного обучения, нейронных сетей в конечном счете позволит существенно повысить эффективность принимаемых управленческих решений в АПК.'
    st.session_state.textInput = exampleText

with col2:
    st.button("Заполнить поле текстом", on_click=on_fill_click)


def summ_text():
    if len(text.strip()):
        data_load_state = st.text('Ожидайте, идет сжатие текста.')
        summText = get_summ_text(text)
        data_load_state.subheader('Результат сжатия:')
        st.write(summText)
    else:
        st.text('Введите текст!')



