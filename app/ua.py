import os
import time
import torch

language = 'ua'
model_id = 'v3_ua'
speaker = 'mykyta'
device = torch.device('cpu')
torch.set_num_threads(4)
sample_rate = 48000
filename = "ua_" + time.strftime("%Y%m%d_%H%M%S")

ssml_sample = """
              <speak>
                <p>Вс+е, що роб+ила наша кра+їна стос+овно антикорупц+ійної структ+ури, не пов’+язано зі вступом в +ЄС. Це для нас<break strength="medium"/>, для Укра+їни...</p>
                <p>
                  <s>Ц+ілі ф+онду - д+ати буд+инок, а не тимчас+овий прит+улок.</s>
                  <s>Швидк+е зв+едення м+одульного житл+а з легк+их зб+ірно-розбірн+их констр+укцій для к+оротко та д+овгострокового розм+іщення б+іженців, евакуй+ованих, вн+утрішньо перем+іщених ос+іб, що б+уде забезп+ечено: оп+аленням, водопостач+анням, каналіз+ацією та включ+атиме об’+єкти соці+ально-побут+ового обслуг+овування: пр+альня, перук+арня, пункти надання домед+ичної та мед+ичної, а т+акож психолог+ічної допом+оги.</s>
                </p>
              </speak>
              """

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
model.to(device)

audio = model.save_wav(ssml_text=ssml_sample,
                             speaker=speaker,
                             sample_rate=sample_rate,
                             audio_path="wav/" + filename + ".wav")