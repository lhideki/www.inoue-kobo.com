from transformers import pipeline
import gradio as gr


def predict(text, labels, custom_label):
    global pipe
    if not pipe:
        pipe = pipeline('zero-shot-classification',
                        model='Formzu/bert-base-japanese-jsnli')
    if custom_label:
        labels.append(custom_label)

    pred = pipe(text, labels, hypothesis_template='この例は{}です。')
    out = {l: s for l, s in zip(pred['labels'], pred['scores'])}

    return out


demo = gr.Interface(fn=predict,
                    allow_flagging='never',
                    title='Zero shot classification by bert-base-japanese-jsnli',
                    inputs=[
                        gr.Textbox(label='text'),
                        gr.CheckboxGroup(label='label', choices=['旅行', '料理', '踊り'], value=[
                                         '旅行', '料理', '踊り']),
                        gr.Textbox(label='custom label')
                    ],
                    outputs='label',
                    examples=[
                        ['そうだ京都に行こう!'],
                        ['ラーメン二郎が大好きです。'],
                        ['ブレイクダンスでダンスバトルです。'],
                        ['京都でお番菜を食べた後に、日舞を見ます。'],
                    ])

pipe = None

if __name__ == '__main__':
    demo.launch(server_port=8080, server_name='0.0.0.0')
