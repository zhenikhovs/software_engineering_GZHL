import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

def get_summ_text(textForSumm):
    tokenizer = T5Tokenizer.from_pretrained('d0rj/rut5-base-summ')
    model = T5ForConditionalGeneration.from_pretrained('d0rj/rut5-base-summ').eval()

    input_ids = tokenizer(textForSumm, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary


st.header("Краткое содержание текста")
st.subheader("(Summarization text) - model d0rj/rut5-base-summ")
st.write('GitHub github.com/zhenikhovs/software_engineering_GZH, brunch model-2')
text = st.text_area('Полный текст для сжатия:',  key='textInput')
col1, col2, col3 = st.columns(3)

with col1:
    clicked = st.button('Сжать текст!')
def on_fill_click():
    exampleText = 'Цифровизация – это социально-экономическая трансформация, которую вызовет массовое внедрение и усвоение новых технологий создания, обработки, анализа и передачи информации. Сельское хозяйство становится одним из основных потребителей цифровых технологий. Генерация большого объема данных разнообразными датчиками в теплицах, полях, фермах и других производственных площадках особенно актуализирует применение цифровых технологий анализа агроэкономических данных. Широкое применение инструментов и методов Data Science, машинного обучения, нейронных сетей в конечном счете позволит существенно повысить эффективность принимаемых управленческих решений в АПК.'
    st.session_state.textInput = exampleText
with col2:
    st.button("Заполнить поле", on_click=on_fill_click)

with col3:
    def on_fill_click():
        exampleText = '''Деловая среда становится все более конкурентной, поэтому обучение персонала необходимо для повышения результативности работы, снижения ошибок и улучшения качества продукции и услуг.
    Портал разрабатывался и внедрялся в ООО «Легаси Студио». Проблемы обучения заключались в том, что выдаваемая информация неструктурирована, приходилось отвлекать опытных специалистов, образовательные платформы имеют каждая свою структуру, руководство не могло контролировать процесс обучения сотрудников.
    Разработка системы с функционалом, направленным на устранение недостатков, являлось решением проблемы.
    Так как Легаси предлагает современные решения с фокусом на бизнес-цели клиента, то система может выступать образовательной платформой и для других организаций, так как занесенные в систему знания могут представлять ценный ресурс для профессионального развития других сотрудников. 
    Цель данного проекта заключается в создании системы, которая уменьшит трудовые и временные затраты наставников на обучение и позволит ввести контроль за продвижением обучения сотрудников.'''
        st.session_state.textInput = exampleText
    st.button("Заполнить поле [2]", on_click=on_fill_click)

if clicked:
    if len(text.strip()):
        data_load_state = st.text('Ожидайте, идет сжатие текста.')
        summText = get_summ_text(text)
        data_load_state.subheader('Результат сжатия:')
        st.write(summText)
    else:
        st.text('Введите текст!')